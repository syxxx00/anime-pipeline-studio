"""An explicit pixel-to-model operation, not an automatic model-role policy."""
from __future__ import annotations

import hashlib
from io import BytesIO
from pathlib import Path
from typing import Literal
import uuid

from PIL import Image
from pydantic import BaseModel, Field

from .config import Settings
from .generation_request import prepare_generation_request
from .lora_catalog import describe_lora
from .sampling_profiles import PROFILES
from .model_conditioning import negative_for_target
from .schemas import GenerateRequest
from .visual_material import VisualMaterial, load_visual_material, verify_visual_material
from .workflow import build_workflow


class HandoffOptions(BaseModel):
    model_config = {'extra':'forbid'}
    target: Literal['newbie','anima','illustrious','z_image_turbo']
    prompt: str | None = Field(default=None, min_length=1, max_length=32000)
    negative_prompt: str | None = Field(default=None, max_length=16000)
    denoise: float = Field(default=.25, gt=0, le=1)
    resize_to: tuple[int, int] | None = None
    steps: int | None = Field(default=None, ge=1, le=60)
    cfg: float | None = Field(default=None, ge=1, le=10)
    seed: int | None = Field(default=None, ge=0, le=2**63-1)
    sampler: str | None = None
    scheduler: str | None = None
    shift: float | None = Field(default=None, ge=0, le=20)
    cfg_rescale: float | None = Field(default=None, ge=0, le=1)
    weight_dtype: Literal['default', 'fp8_e4m3fn', 'fp8_e5m2'] | None = None
    lora_name: str | None = None
    lora_strength: float = Field(default=.75, ge=0, le=2)
    style_preset: str = Field(default='', max_length=120)


def build_handoff_workflow(request: GenerateRequest, settings: Settings, input_name: str, prefix: str) -> dict:
    if request.image_handoff is None or request.batch_size != 1:
        raise ValueError('模型交接必须声明输入，且一次只执行一张图片')
    if request.render_pipeline.value not in PROFILES or PROFILES[request.render_pipeline.value]['model_id'] != request.model_id.value:
        raise ValueError('交接目标必须是明确的单模型')
    if Path(input_name).name != input_name or '/' in input_name or '\\' in input_name:
        raise ValueError('交接输入名不能包含目录')
    if request.face_detail_enabled and request.model_id.value != 'newbie':
        raise ValueError('当前目标的交接构建器不能执行已选面部阶段；未静默关闭设置。')
    workflow = build_workflow(request, settings, prefix)
    # Replace the base sampler's empty latent only. Each target's own highres
    # (and NewBie's face stage) keeps its parameters and condition connections.
    samplers = [(key,node) for key,node in workflow.items() if node['class_type']=='KSampler'
        and workflow.get(node['inputs']['latent_image'][0], {}).get('class_type')
            in {'EmptyLatentImage', 'EmptySD3LatentImage'}]
    decoders = [node for node in workflow.values() if node['class_type'] in {'VAEDecode','VAEDecodeTiled'}]
    if len(samplers)!=1 or not decoders or len({tuple(n['inputs']['vae']) for n in decoders}) != 1:
        raise ValueError('该工作流没有唯一可验证的初始采样与目标 VAE')
    # Derive the target VAE from its own decoder. SDXL's checkpoint VAE and
    # other models' separately loaded VAEs must never be cross-connected.
    vae = list(decoders[0]['inputs']['vae'])
    workflow['handoff_load'] = {'class_type':'LoadImage','inputs':{'image':input_name}}
    pixels = ['handoff_load',0]
    if request.image_handoff.resize_applied:
        workflow['handoff_resize'] = {'class_type':'ImageScale','inputs':{
            'image':pixels,'upscale_method':'lanczos','width':request.width,'height':request.height,'crop':'disabled'}}
        pixels = ['handoff_resize',0]
    workflow['handoff_encode'] = {'class_type':'VAEEncode','inputs':{'pixels':pixels,'vae':vae}}
    sampler = samplers[0][1]['inputs']
    empty_id = sampler['latent_image'][0]
    sampler.update(latent_image=['handoff_encode',0],denoise=request.image_handoff.denoise)
    if workflow[empty_id]['class_type'] in {'EmptyLatentImage','EmptySD3LatentImage'}:
        del workflow[empty_id]
    return workflow


def prepare_handoff(artwork, parent_job, payload: dict, settings: Settings, *, materialize: bool = True,
                    context_request: GenerateRequest | None = None):
    options = HandoffOptions.model_validate(payload)
    source = Path(artwork.file_path).resolve()
    if not source.is_relative_to(Path(settings.output_dir).resolve()) or not source.is_file():
        raise ValueError('原图不在作品输出目录内或文件不存在；请先保留阶段图为作品')
    source_bytes = source.read_bytes()
    with Image.open(BytesIO(source_bytes)) as image:
        if image.width*image.height > 40_000_000:
            raise ValueError('原图尺寸过大')
        source_size = image.size
    size = options.resize_to or ((context_request.width, context_request.height)
        if context_request is not None else source_size)
    if any(value < 512 or value > 1536 or value % 16 for value in size):
        raise ValueError('本地交接输出需为 512–1536 且为 16 的倍数；大图请显式提供 resize_to，不会自动缩小')
    profile = PROFILES[options.target]
    if context_request is not None and (context_request.model_id.value != profile['model_id']
            or context_request.render_pipeline.value != options.target):
        raise ValueError('完整当前条件需先准备为交接目标模型；不会继承另一模型的采样参数。')
    if context_request is not None and context_request.batch_size != 1:
        raise ValueError('当前交接一次只支持一张；未将已选批次数悄悄改为 1。')
    channel = {'z_image_turbo':'z_image'}.get(options.target,options.target)
    pixel_parent = parent_job.request
    parent = context_request.model_dump(mode='json') if context_request is not None else pixel_parent
    native = parent.get('native_prompts') or {}
    native_target = dict(native.get(channel) or {})
    positive = options.prompt if options.prompt is not None else native_target.get('positive')
    if not positive and parent.get('model_id') == profile['model_id']:
        positive = parent.get('prompt')
    if not positive or not str(positive).strip():
        raise ValueError('缺少目标模型的原生文本，请明确填写 prompt；不会借用其他模型格式')
    negative = negative_for_target(parent, native_target, options.target, override=options.negative_prompt)
    changed = positive != native_target.get('positive') or negative != native_target.get('negative')
    if changed and (options.prompt is not None or options.negative_prompt is not None):
        native_target['compiled_trace'] = native_target.get('trace',[])
        native_target['trace'] = []
        native_target['trace_status'] = 'unverified_manual_override'
    native_target.update(positive=positive,negative=negative)
    native = {**native,channel:native_target}
    proposal = parent.get('author_proposal')
    semantic, contract = parent.get('semantic_spec'), parent.get('scene_contract')
    if changed and proposal:
        proposal = {**proposal, 'condition_status': 'historical_after_native_override'}
        semantic, contract = None, None
    own_parameters = options.model_dump(exclude_none=True,include={
        'steps','cfg','seed','sampler','scheduler','shift','cfg_rescale','weight_dtype','lora_name','lora_strength'})
    input_dir = (Path(settings.engine_base_dir)/'input').resolve()
    name = f'handoff_{uuid.uuid4().hex}_source.png'
    source_input = input_dir/name
    base = (context_request.model_dump(mode='json') if context_request is not None else {
        **profile, 'seed': parent.get('seed',42),
        'highres_enabled':False, 'face_detail_enabled':False,
        'hand_repair_enabled':False, 'text_clean_enabled':False, 'check_enabled':True})
    if context_request is not None:
        own_parameters = {key:value for key,value in own_parameters.items() if key in payload}
    request = GenerateRequest.model_validate({
        **base, **own_parameters,
        'render_pipeline':options.target, 'parameter_policy':'explicit',
        'prompt':positive,'negative_prompt':negative,'use_default_negative_prompt':False,
        'native_prompts':native,'semantic_spec':semantic,'scene_contract':contract, 'author_proposal':proposal,
        'creation_intent':parent.get('creation_intent'), 'iteration_context':parent.get('iteration_context'),
        'width':size[0],'height':size[1],'batch_size':1,
        'style_preset':options.style_preset,
        'parent_artwork_id':artwork.id,'operation':'image_handoff',
        'output_group':'handoffs','output_label':f'{options.target}__from-{artwork.id}',
        'image_handoff':{'source_file':str(source_input),'source_sha256':hashlib.sha256(source_bytes).hexdigest(),
            'source_artwork_id':artwork.id,'source_model_id':pixel_parent.get('model_id','unknown'),
            'denoise':options.denoise,'resize_applied':size!=source_size,'requested_style_preset':options.style_preset},
    })
    if request.lora_name:
        root = settings.path_map['loras'].resolve()
        lora = (root/request.lora_name).resolve()
        if not lora.is_relative_to(root) or not lora.is_file() or request.model_id.value not in describe_lora(lora,root)['compatible_models']:
            raise ValueError('LoRA 缺少目标模型的兼容架构证据')
    request, _, _ = prepare_generation_request(request,settings)
    # Freeze already styled text so submit_workflow cannot inject a second
    # style layer; keep the user's chosen preset in the handoff provenance.
    request = request.model_copy(update={'style_preset':''})
    workflow = build_handoff_workflow(request,settings,name,'newbie_lab/'+name.removesuffix('_source.png'))
    if materialize:
        input_dir.mkdir(parents=True,exist_ok=True)
        source_input.write_bytes(source_bytes)
    return request,workflow


def prepare_visual_material_handoff(request: GenerateRequest, material: VisualMaterial,
                                   settings: Settings, denoise: float = .35, *, materialize: bool = True):
    """Use a real programme sketch as pixels, with the target's full conditions.

    This is not a generated-work ancestry edge. A caller must supply the actual
    complete single-model request; this function does not invent art conditions
    or convert one model's native text into another model's format.
    """
    source_bytes = verify_visual_material(material, settings)
    if (request.parameter_policy != 'explicit'
            or request.render_pipeline.value not in PROFILES
            or PROFILES[request.render_pipeline.value]['model_id'] != request.model_id.value):
        raise ValueError('视觉材料消费必须采用显式的单目标模型与完整参数')
    if request.image_handoff or request.parent_artwork_id or request.repair_source_file or request.mask_file:
        raise ValueError('视觉材料不是生成作品祖先，不能混入已有交接或区域重绘来源')
    if (not request.creation_intent or request.creation_intent.original_text != material.original_text
            or request.creation_intent.original_sha256 != material.original_sha256):
        raise ValueError('视觉材料必须与请求的完整原句一致')
    target = request.render_pipeline.value
    channel = {'z_image_turbo': 'z_image'}.get(target, target)
    native = (request.native_prompts or {}).get(channel) or {}
    if native.get('positive') is not None and native['positive'] != request.prompt:
        raise ValueError('目标原生正向条件与当前完整请求不同，请先明确当前版本')
    negative = negative_for_target(request.model_dump(mode='json'), native, target)
    if native.get('negative') is not None and request.negative_prompt != negative:
        raise ValueError('目标原生负向条件与当前完整请求不同，请先明确当前版本')
    input_dir = (Path(settings.engine_base_dir) / 'input').resolve()
    name = f'handoff_{uuid.uuid4().hex}_source.png'
    requested_style = request.style_preset
    updated = request.model_dump(mode='json') | {
        'negative_prompt': negative, 'use_default_negative_prompt': False,
        'batch_size': 1, 'highres_enabled': False, 'face_detail_enabled': False,
        'hand_repair_enabled': False, 'text_clean_enabled': False,
        'parent_artwork_id': None, 'operation': 'image_handoff',
        'image_handoff': {
            'source_kind': 'visual_material', 'source_file': str(input_dir / name),
            'source_sha256': material.png_sha256, 'source_artwork_id': None,
            'source_model_id': material.generation_method, 'source_material_id': material.id,
            'source_material_record_file': material.record_file,
            'source_material_record_sha256': hashlib.sha256(Path(material.record_file).read_bytes()).hexdigest(),
            'source_original_sha256': material.original_sha256,
            'denoise': denoise, 'resize_applied': (request.width, request.height) != (material.width, material.height),
            'requested_style_preset': requested_style,
        },
    }
    prepared, _, _ = prepare_generation_request(GenerateRequest.model_validate(updated), settings)
    # As for generated-work handoff, freeze the applied style layer so queueing
    # does not append it twice. The exact selection remains in provenance.
    prepared = prepared.model_copy(update={'style_preset': ''})
    workflow = build_handoff_workflow(prepared, settings, name, 'newbie_lab/' + name.removesuffix('_source.png'))
    if materialize:
        input_dir.mkdir(parents=True, exist_ok=True)
        with (input_dir / name).open('xb') as handle:
            handle.write(source_bytes)
    return prepared, workflow


def verify_handoff_input(request: GenerateRequest, settings: Settings) -> None:
    info = request.image_handoff
    if info is None:
        return
    path = Path(info.source_file).resolve()
    root = (Path(settings.engine_base_dir)/'input').resolve()
    if not path.is_relative_to(root) or hashlib.sha256(path.read_bytes()).hexdigest()!=info.source_sha256:
        raise ValueError('模型交接输入缺失、越界或在排队后变化')
    if info.source_kind == 'visual_material':
        material = load_visual_material(info.source_material_id, settings)
        if (material.record_file != info.source_material_record_file
                or material.png_sha256 != info.source_sha256
                or material.original_sha256 != info.source_original_sha256
                or request.creation_intent is None
                or request.creation_intent.original_text != material.original_text
                or hashlib.sha256(Path(material.record_file).read_bytes()).hexdigest()
                    != info.source_material_record_sha256):
            raise ValueError('视觉材料来源在排队后变化，或与本次原句不同')


def verify_handoff_output(payload: bytes, request: GenerateRequest) -> dict:
    size = (request.width, request.height)
    if request.highres_enabled:
        # Match this local Core's LatentUpscaleBy: round in latent pixels,
        # then decode using the same 8x-spatial target VAE. Do not round the
        # image dimensions first or reject a correctly enlarged handoff.
        size = tuple(round(value // 8 * request.highres_scale) * 8 for value in size)
    with Image.open(BytesIO(payload)) as image:
        if image.size != size:
            raise ValueError('模型交接输出尺寸与记录不一致')
    source_name = '程序视觉材料' if request.image_handoff.source_kind == 'visual_material' else '模型交接来源'
    return {'name':source_name,'status':'warning',
            'message':'从记录的源像素经目标模型自身 VAE 重新编码；全图可能改变，尚未验证语义、角色或审美改善。',
            'details':request.image_handoff.model_dump()}


def cleanup_handoff_input(request: GenerateRequest, settings: Settings) -> str | None:
    if not request.image_handoff:
        return
    path = Path(request.image_handoff.source_file).resolve()
    root = (Path(settings.engine_base_dir)/'input').resolve()
    # Only delete the single temporary input created by this operation.
    import re
    if path.parent==root and re.fullmatch(r'handoff_[0-9a-f]{32}_source\.png',path.name):
        try:
            path.unlink(missing_ok=True)
        except OSError as exc:
            return f'交接临时输入未能清理：{path}（{type(exc).__name__}）；原图与已保存版本未删除。'
    return None
