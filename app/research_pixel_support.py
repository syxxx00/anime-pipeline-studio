"""Explicit whole-image research pixels with the complete current conditions.

This consumer neither selects material nor certifies artwork/model authorship.
Its caller must also validate the frozen selection against the live registries,
persist the returned request/receipt, and reserve dispatch before queueing.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
from io import BytesIO
import os
from pathlib import Path
from typing import Literal
import uuid

from PIL import Image
from pydantic import BaseModel, Field, field_validator

from .creative_art_support import ResearchArtSupport, _saved_native, _source_original
from .creative_continuity_context import _digest, _raw_words
from .image_handoff import build_handoff_workflow, verify_handoff_input
from .pipeline_styles import prepare_request_styles
from .sampling_profiles import PROFILES
from .schemas import GenerateRequest


class ResearchPixelUse(BaseModel):
    model_config = {'extra': 'forbid', 'strict': True, 'revalidate_instances': 'always'}
    source_scope: Literal['whole_image']
    denoise: float = Field(gt=0, le=1, allow_inf_nan=False)
    resize_policy: Literal['exact', 'to_current']
    reason: str = Field(min_length=1, max_length=4000)

    @field_validator('reason')
    @classmethod
    def nonblank_reason(cls, value):
        if not value.strip():
            raise ValueError('像素采用理由不能为空；原话不改写。')
        return value


def _frozen_png(snapshot, settings):
    """Check saved own-job provenance and bounded exact pixels, not a live job."""
    try:
        if not isinstance(snapshot, dict):
            raise ValueError('研究像素需要完整冻结快照。')
        saved = deepcopy(snapshot)
        signature = saved.pop('snapshot_sha256')
        if (saved['schema_version'] != 1 or saved['kind'] != 'internal_research_art_support'
                or _digest(saved) != signature or saved['condition_scope'] != 'own_completed_job'
                or saved['grants_image_handoff'] is not False):
            raise ValueError('研究快照校验失败；原观察用途不能隐式授予像素交接。')
        selection = ResearchArtSupport.model_validate(saved['selection'])
        work, job, image = saved['source_artwork'], saved['source_job'], saved['image']
        if (not isinstance(work, dict) or not isinstance(job, dict) or not isinstance(image, dict)
                or work['id'] != selection.artwork_id or work['job_id'] != job['id']
                or job['status'] != 'completed' or work.get('source_stage') is not None
                or type(work['image_index']) is not int or work['image_index'] < 0
                or not isinstance(job['request'], dict)
                or _digest(job['request']) != job['request_sha256']
                or image['sha256'] != selection.expected_image_sha256
                or saved['source_native'] != _saved_native(job['request'])
                or saved['source_original'] != _source_original(job['request'])):
            raise ValueError('研究像素不是快照中自己完成任务的确切原件。')
        reviews = saved['feedback']['artwork_reviews']
        if (work.get('reviews', []) != reviews
                or _digest({'artwork_id': work['id'], 'reviews': _raw_words(reviews)})
                   != saved['feedback']['original_words_signature']):
            raise ValueError('研究评价原话与冻结来源不同。')
        path = Path(work['file_path'])
        root = Path(settings.output_dir).resolve()
        if (not path.is_absolute() or path.is_symlink() or not path.is_file()
                or not path.resolve().is_relative_to(root)
                or Path(image['file_path']).resolve() != path.resolve()
                or image['format'] != 'PNG' or type(image['bytes']) is not int
                or not 0 < image['bytes'] <= 8 * 1024 * 1024
                or type(image['width']) is not int or type(image['height']) is not int
                or min(image['width'], image['height']) <= 0
                or image['width'] * image['height'] > 16_000_000):
            raise ValueError('研究原PNG缺失、越界或超过原件上限；不换缩略图。')
        before = path.stat()
        with path.open('rb') as stream:
            opened = os.fstat(stream.fileno())
            raw = stream.read(image['bytes'] + 1)
            after_read = os.fstat(stream.fileno())
        after = path.stat()
        identity = lambda stat: (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns)
        file_identity = {'device': before.st_dev, 'inode': before.st_ino, 'mtime_ns': before.st_mtime_ns}
        if (not identity(before) == identity(opened) == identity(after_read) == identity(after)
                or before.st_size != image['bytes'] or len(raw) != image['bytes']
                or file_identity != image['file_identity']
                or hashlib.sha256(raw).hexdigest() != image['sha256']):
            raise ValueError('研究原PNG在冻结后或读取中变化；不采用旧快照。')
        with Image.open(BytesIO(raw)) as picture:
            if (picture.format != 'PNG' or getattr(picture, 'n_frames', 1) != 1
                    or picture.size != (image['width'], image['height'])):
                raise ValueError('研究原件不是所选单帧PNG或尺寸不同。')
            picture.verify()
        with Image.open(BytesIO(raw)) as picture:
            picture.load()
        return saved, signature, raw
    except (KeyError, TypeError, OSError, SyntaxError, Image.DecompressionBombError) as exc:
        raise ValueError('研究像素原件或冻结出处不能完整核验；未创建输入。') from exc


def _complete_current(request, settings):
    if not isinstance(request, GenerateRequest):
        raise ValueError('研究像素消费需要已验证的完整当前GenerateRequest。')
    actual = GenerateRequest.model_validate(request.model_dump(mode='json'))
    route = actual.render_pipeline.value
    if (actual.parameter_policy != 'explicit' or route not in PROFILES
            or actual.model_id.value != PROFILES[route]['model_id']
            or actual.creation_intent is None or actual.batch_size != 1
            or actual.operation not in {'generate', 'iteration_generate'}
            or actual.source_parameters is not None):
        raise ValueError('研究像素消费需完整当前原句、显式单模型和单张基础请求。')
    if (actual.highres_enabled or actual.face_detail_enabled or actual.hand_repair_enabled
            or actual.text_clean_enabled or actual.regional_conditioning is not None
            or actual.image_handoff is not None or actual.mask_file or actual.repair_source_file
            or actual.repair_source_sha256 or actual.repair_mask_sha256
            or actual.repair_instruction or actual.repair_context_prompt or actual.repair_notes):
        raise ValueError('当前已选附加阶段或已有修改不受此基础消费者支持；不关闭或丢弃。')
    channel = {'z_image_turbo': 'z_image'}.get(route, route)
    native = (actual.native_prompts or {}).get(channel) or {}
    if (not isinstance(native.get('positive'), str) or not native['positive'].strip()
            or not isinstance(native.get('negative'), str)
            or native['positive'] != actual.prompt or native['negative'] != actual.negative_prompt
            or (not actual.negative_prompt.strip() and actual.use_default_negative_prompt)):
        raise ValueError('当前目标完整原生正负文本与实际请求不一致或存在负面默认回退。')
    if prepare_request_styles(actual, settings).model_dump(mode='json') != actual.model_dump(mode='json'):
        raise ValueError('当前风格尚未完整编译；不能在像素消费中另注入或改变条件。')
    if actual.lora_name:
        from .lora_catalog import describe_lora
        root = settings.path_map['loras'].resolve()
        path = (root / actual.lora_name).resolve()
        if (not path.is_relative_to(root) or not path.is_file()
                or actual.model_id.value not in describe_lora(path, root)['compatible_models']):
            raise ValueError('当前LoRA缺少目标模型兼容证据；不移除所选角色或风格条件。')
    return actual


def prepare_research_pixel_handoff(request, snapshot, options, settings, *, materialize=False):
    """Keep the full new request; use old pixels only by explicit whole-image use.

    ``exact`` refuses different dimensions. ``to_current`` uses Lanczos with
    no crop; a changed aspect ratio stretches the entire image. Neither mode
    isolates style from layout/identity or promises to satisfy current intent.
    Live registry freshness and dispatch/resume ownership remain caller duties.
    """
    if type(materialize) is not bool:
        raise ValueError('materialize必须显式为布尔值。')
    use = ResearchPixelUse.model_validate(options)
    current = _complete_current(request, settings)
    saved, signature, raw = _frozen_png(snapshot, settings)
    source_size = (saved['image']['width'], saved['image']['height'])
    target_size = (current.width, current.height)
    if use.resize_policy == 'exact' and source_size != target_size:
        raise ValueError('exact研究像素尺寸与当前画幅不同；不改变当前尺寸或静默缩放。')
    input_root = (Path(settings.engine_base_dir) / 'input').resolve()
    name = f'handoff_{uuid.uuid4().hex}_source.png'
    input_path = input_root / name
    current_data = current.model_dump(mode='json')
    receipt = {'version': 1, 'kind': 'explicit_whole_research_pixel_use',
        'use': use.model_dump(mode='json'), 'use_for': saved['selection']['use_for'],
        'source_snapshot_sha256': signature, 'source_artwork_id': saved['source_artwork']['id'],
        'source_job_id': saved['source_job']['id'], 'source_request_sha256': saved['source_job']['request_sha256'],
        'source_image_sha256': saved['image']['sha256'], 'source_size': list(source_size),
        'target_size': list(target_size), 'resize_applied': source_size != target_size,
        'resize_method': 'lanczos_no_crop' if source_size != target_size else 'unchanged_pixels',
        'aspect_ratio_changed': source_size[0] * target_size[1] != source_size[1] * target_size[0],
        'current_request_sha256': _digest(current_data),
        'current_request_hash_scope': 'complete_input_before_operation_and_receipt_not_final_dispatch',
        'dispatch_hash_responsibility': 'caller_hashes_entire_final_request_after_pixel_operation_receipt_and_output_labeling',
        'authoring_scope': 'current_saved_request_not_model_authorship',
        'source_condition_scope': 'historical_snapshot_not_new_native_or_character_conditions',
        'source_observation_grants_image_handoff': False,
        'adoption_authority': 'new_explicit_pixel_use_not_old_observation_permission',
        'live_registry_revalidation_required': True,
        'scope_warning': '全幅像素可能带入旧人物、姿态、构图和画风；不是纯画风迁移，也不认证当前作品。',
        'quality_verified': False, 'identity_accuracy_verified': False,
        'semantic_understanding_verified': False}
    context = deepcopy(current_data['iteration_context']) if current_data['iteration_context'] is not None else {
        'original_text': current.creation_intent.original_text,
        'original_sha256': current.creation_intent.original_sha256}
    if 'research_pixel_use' in context:
        raise ValueError('当前请求已有研究像素采用回执；不覆盖或重新解释旧操作。')
    context['research_pixel_use'] = receipt
    prepared_data = deepcopy(current_data)
    prepared_data.update(operation='image_handoff', parent_artwork_id=saved['source_artwork']['id'],
        iteration_context=context, image_handoff={
            'source_file': str(input_path), 'source_sha256': saved['image']['sha256'],
            'source_kind': 'generated_artwork', 'source_artwork_id': saved['source_artwork']['id'],
            'source_model_id': str(saved['source_job']['request'].get('model_id', 'unknown')),
            'denoise': use.denoise, 'resize_applied': source_size != target_size,
            'requested_style_preset': current.style_preset, 'semantic_verified': False})
    actual = GenerateRequest.model_validate(prepared_data)
    workflow = build_handoff_workflow(actual, settings, name,
        'newbie_lab/' + name.removesuffix('_source.png'))
    if materialize:
        input_root.mkdir(parents=True, exist_ok=True)
        owned = False
        try:
            with input_path.open('xb') as stream:
                owned = True
                stream.write(raw)
            verify_handoff_input(actual, settings)
        except BaseException:
            if owned:
                input_path.unlink(missing_ok=True)
            raise
    return actual, workflow, raw
