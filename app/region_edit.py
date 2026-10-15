"""Prepare explicit user edits and verify their pixel boundary before delivery."""
from __future__ import annotations

import base64
import binascii
import hashlib
from io import BytesIO
from pathlib import Path
import uuid

from PIL import Image, ImageChops
from pydantic import BaseModel, Field

from .config import Settings
from .lora_catalog import describe_lora
from .schemas import GenerateRequest
from .workflow import build_region_repair_workflow


class RegionEditOptions(BaseModel):
    instruction: str = Field(min_length=1, max_length=4000)
    mask_data_url: str = Field(max_length=28_000_000)
    steps: int = Field(default=18, ge=5, le=40)
    denoise: float = Field(default=.42, ge=.15, le=.75)
    source_sha256: str | None = Field(default=None, pattern=r'^[0-9a-f]{64}$')


def generation_ancestor(artwork, artworks, jobs):
    seen = set()
    while artwork and artwork.id not in seen:
        seen.add(artwork.id)
        job = jobs.get(artwork.job_id)
        if job is not None:
            return job
        artwork = artworks.get(artwork.parent_artwork_id) if artwork.parent_artwork_id else None
    return None


def repair_parameters(parent: dict, settings: Settings) -> tuple[dict, list[str]]:
    # This adapts to the currently implemented repair engine, not a claim that
    # NewBie is the best repair model. Do not pass SDXL/Turbo settings into it.
    values = {}
    notes = ['局部修改当前使用 NewBie；是否适合此处的风格和细节仍需比较原图确认。']
    if parent.get('model_id', 'newbie') == 'newbie':
        keys = ('steps', 'cfg', 'shift', 'cfg_rescale', 'sampler', 'scheduler',
                'negative_prompt', 'use_default_negative_prompt', 'lora_strength')
        values.update({key: parent[key] for key in keys if key in parent})
        name = parent.get('lora_name')
        if name:
            root = (Path(settings.model_root) / 'loras').resolve()
            path = (root / str(name)).resolve()
            if path.is_relative_to(root) and path.is_file() and 'newbie' in describe_lora(path, root)['compatible_models']:
                values['lora_name'] = name
            else:
                notes.append('未继承原 LoRA：缺少可验证的 NewBie 架构兼容信息。')
    else:
        notes.append(f"原图来自 {parent.get('model_id')}；采样参数和 LoRA 未跨架构继承，使用 NewBie 修补默认值。")
    for key in ('seed', 'semantic_spec', 'scene_contract', 'creation_intent', 'author_proposal', 'check_enabled'):
        if key in parent:
            values[key] = parent[key]
    return values, notes


def protection_fields(source: Path, mask: Path) -> dict:
    return dict(repair_source_file=str(source.resolve()), mask_file=str(mask.resolve()),
                repair_source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                repair_mask_sha256=hashlib.sha256(mask.read_bytes()).hexdigest())


def prepare_region_edit(artwork, parent_job, payload: dict, settings: Settings, *,
                        context_request: GenerateRequest | None = None, materialize: bool = True):
    options = RegionEditOptions.model_validate(payload)
    instruction = options.instruction.strip()
    if not instruction:
        raise ValueError('请说明圈选区域要怎样修改')
    if not options.mask_data_url.startswith('data:image/png;base64,'):
        raise ValueError('遮罩必须是 PNG')
    try:
        mask_bytes = base64.b64decode(options.mask_data_url.split(',', 1)[1], validate=True)
    except (ValueError, binascii.Error) as exc:
        raise ValueError('遮罩数据无效') from exc
    if not mask_bytes or len(mask_bytes) > 20 * 1024 * 1024:
        raise ValueError('遮罩为空或超过 20MB')
    source = Path(artwork.file_path).resolve()
    if not source.is_relative_to(Path(settings.output_dir).resolve()) or not source.is_file():
        raise ValueError('原图不在作品输出目录内或已不存在')
    source_bytes = source.read_bytes()
    if options.source_sha256 is not None and hashlib.sha256(source_bytes).hexdigest() != options.source_sha256:
        raise ValueError('圈选对应的原图像素已变化，请重新打开当前原图并圈选')
    try:
        with Image.open(BytesIO(mask_bytes)) as raw, Image.open(BytesIO(source_bytes)) as image:
            if raw.format != 'PNG' or raw.size != image.size:
                raise ValueError('遮罩必须是与原图尺寸一致的 PNG')
            if image.width * image.height > 40_000_000:
                raise ValueError('图片尺寸超过局部修改支持范围')
            # Transparent RGB values must never authorize edits. The browser
            # uses white paint on a transparent canvas; grayscale masks also work.
            rgba = raw.convert('RGBA')
            mask = ImageChops.multiply(rgba.convert('L'), rgba.getchannel('A'))
            if not mask.getbbox():
                raise ValueError('请先在图片上圈出需要修改的区域')
            normalized = BytesIO()
            mask.save(normalized, format='PNG')
            mask_bytes = normalized.getvalue()
    except (OSError, Image.DecompressionBombError) as exc:
        raise ValueError('图片或遮罩数据无法解码') from exc
    parameters, notes = repair_parameters(parent_job.request, settings)
    if context_request is not None:
        if context_request.model_id != 'newbie' or not context_request.prompt.strip():
            raise ValueError('循环局部修改需要完整的当前 NewBie 场景条件')
        # A previous repair's encoded prompt contains its local operation, not
        # just the scene. Never nest old overrides into another scene context:
        # the current source pixels carry that history, and a new local intent
        # may explicitly replace it inside the new mask.
        scene_context = (context_request.repair_context_prompt
                         if context_request.repair_context_policy == 'scene_context_with_local_override'
                         and context_request.repair_context_prompt else context_request.prompt)
        for key in ('steps', 'cfg', 'shift', 'cfg_rescale', 'sampler', 'scheduler',
                    'seed', 'lora_name', 'lora_strength', 'weight_dtype'):
            parameters[key] = getattr(context_request, key)
        for key in ('semantic_spec', 'scene_contract', 'creation_intent', 'author_proposal', 'check_enabled'):
            parameters[key] = getattr(context_request, key)
        parameters.update(negative_prompt=context_request.negative_prompt,
                          use_default_negative_prompt=context_request.use_default_negative_prompt,
                          repair_context_prompt=scene_context,
                          repair_context_policy='scene_context_with_local_override')
        notes.append('完整当前场景仅作上下文；圈内以本次局部意图覆盖冲突的旧细节，是否达成仍需比较。')
    input_dir = (Path(settings.engine_base_dir) / 'input').resolve()
    edit_id = f'edit_{uuid.uuid4().hex[:16]}'
    source_name, mask_name = f'{edit_id}_source.png', f'{edit_id}_mask.png'
    source_input, mask_input = input_dir / source_name, input_dir / mask_name
    request = GenerateRequest.model_validate({**parameters,
        'parameter_policy': 'explicit', 'model_id': 'newbie', 'render_pipeline': 'newbie',
        'prompt': instruction, 'batch_size': 1, 'output_group': 'edits',
        'output_label': f'{artwork.id}__region-repair',
        'highres_enabled': False, 'face_detail_enabled': False,
        'hand_repair_enabled': False, 'text_clean_enabled': False,
        'parent_artwork_id': artwork.id, 'operation': 'region_repair',
        'repair_instruction': instruction, 'region_repair_steps': options.steps,
        'region_repair_denoise': options.denoise, 'repair_notes': notes,
        'repair_source_file': str(source_input), 'mask_file': str(mask_input),
        'repair_source_sha256': hashlib.sha256(source_bytes).hexdigest(),
        'repair_mask_sha256': hashlib.sha256(mask_bytes).hexdigest(),
    })
    workflow = build_region_repair_workflow(request, settings, source_name, mask_name, f'newbie_lab/{edit_id}')
    if context_request is not None:
        request = GenerateRequest.model_validate(request.model_dump(mode='json') | {
            'prompt': workflow['6']['inputs']['text'], 'negative_prompt': workflow['7']['inputs']['text'],
            'use_default_negative_prompt': False,
            'native_prompts': {'newbie': {'positive': workflow['6']['inputs']['text'],
                'negative': workflow['7']['inputs']['text'], 'trace': [],
                'trace_status': 'local_edit_context_and_override_unverified'}},
        })
    if not materialize:
        return request, workflow, [source_input, mask_input]
    input_dir.mkdir(parents=True, exist_ok=True)
    try:
        # Unique names prevent two queued edits with the same mask from sharing
        # writable input files. Validation above happens before any file write.
        source_input.write_bytes(source_bytes)
        mask_input.write_bytes(mask_bytes)
    except OSError:
        source_input.unlink(missing_ok=True)
        mask_input.unlink(missing_ok=True)
        raise
    return request, workflow, [source_input, mask_input]


def protected_inputs(request: GenerateRequest, settings: Settings) -> tuple[bytes, bytes]:
    root = (Path(settings.engine_base_dir) / 'input').resolve()
    result = []
    for filename, expected in ((request.repair_source_file, request.repair_source_sha256),
                               (request.mask_file, request.repair_mask_sha256)):
        path = Path(filename or '').resolve()
        if not path.is_relative_to(root) or not expected:
            raise ValueError('缺少有效的局部修改输入保护记录')
        payload = path.read_bytes()
        if hashlib.sha256(payload).hexdigest() != expected:
            raise ValueError('局部修改输入在排队后发生变化，已停止使用')
        result.append(payload)
    return tuple(result)


def verify_region_output(payload: bytes, request: GenerateRequest, settings: Settings) -> dict:
    source_bytes, mask_bytes = protected_inputs(request, settings)
    with Image.open(BytesIO(source_bytes)) as original, Image.open(BytesIO(mask_bytes)) as mask, Image.open(BytesIO(payload)) as output:
        if output.size != original.size or mask.size != original.size:
            raise ValueError('局部修改输出尺寸变化，未保存为成功作品')
        differences = ImageChops.difference(original.convert('RGBA'), output.convert('RGBA')).split()
        difference = differences[0]
        for channel in differences[1:]:
            difference = ImageChops.lighter(difference, channel)
        outside = mask.convert('L').point(lambda value: 0 if value else 255)
        if ImageChops.multiply(difference, outside).getbbox():
            raise ValueError('局部修改改变了未圈选区域，未保存为成功作品')
        changed = difference.getbbox()
    return dict(name='局部修改像素边界', status='pass' if changed else 'warning',
                message='未圈选区域逐像素一致；圈内质量仍需人工比较。' if changed else '边界一致，但图像没有发生可见像素变化，请比较修改是否达成。',
                details=dict(verification='outside_mask_pixels_verified', changed_pixel_bbox=changed,
                             source_sha256=request.repair_source_sha256, mask_sha256=request.repair_mask_sha256))
