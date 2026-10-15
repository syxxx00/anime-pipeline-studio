"""Opt-in full-canvas regional text predictions, not an identity or layout proof.

Global conditioning is retained. Masks only weight additional model predictions;
they do not pin an object, gaze, pose, or anatomical detail to a location.
"""
from __future__ import annotations

from copy import deepcopy
from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator


NativeTarget = Literal['newbie', 'illustrious', 'anima', 'z_image']
MODEL_TARGETS = {
    'newbie': 'newbie',
    'illustrious': 'illustrious',
    'anima_turbo': 'anima',
    'z_image_turbo': 'z_image',
}
SINGLE_PIPELINES = {'newbie', 'illustrious', 'anima', 'z_image_turbo'}


class RegionalTextCondition(BaseModel):
    """One explicitly targeted native positive and normalized rectangle."""
    model_config = {'extra': 'forbid'}
    target: NativeTarget
    positive: str = Field(min_length=1, max_length=32_000)
    x: float = Field(ge=0, lt=1, strict=True, allow_inf_nan=False)
    y: float = Field(ge=0, lt=1, strict=True, allow_inf_nan=False)
    width: float = Field(gt=0, le=1, strict=True, allow_inf_nan=False)
    height: float = Field(gt=0, le=1, strict=True, allow_inf_nan=False)
    strength: float = Field(default=1.0, gt=0, le=2, strict=True, allow_inf_nan=False)

    @field_validator('positive')
    @classmethod
    def nonempty_native_positive(cls, value: str) -> str:
        if not value.strip():
            raise ValueError('区域原生正向条件不能为空')
        return value

    @model_validator(mode='after')
    def inside_canvas(self) -> 'RegionalTextCondition':
        if self.x + self.width > 1 or self.y + self.height > 1:
            raise ValueError('区域矩形必须完整位于归一化画布内')
        return self

    def pixel_box(self, width: int, height: int) -> tuple[int, int, int, int]:
        x, y = round(self.x * width), round(self.y * height)
        right = round((self.x + self.width) * width)
        bottom = round((self.y + self.height) * height)
        if right <= x or bottom <= y:
            raise ValueError('区域矩形转换为当前图片像素后不能为空')
        return x, y, right - x, bottom - y


def validate_regional_request(request) -> list[RegionalTextCondition]:
    """Check at schema admission and again before building a copied request."""
    raw = request.regional_conditioning
    if raw is None:
        return []
    if not isinstance(raw, list) or not 1 <= len(raw) <= 4:
        raise ValueError('区域条件需要 1 至 4 个明确区域')
    if (request.parameter_policy != 'explicit'
            or request.render_pipeline.value not in SINGLE_PIPELINES
            or request.operation not in {'generate', 'iteration_generate'}
            or request.batch_size != 1 or request.highres_enabled
            or request.face_detail_enabled or request.hand_repair_enabled
            or request.text_clean_enabled or request.image_handoff is not None
            or request.repair_source_file or request.mask_file
            or request.repair_instruction.strip() or request.source_parameters is not None
            or request.source_prompt.strip()):
        raise ValueError('区域文本条件仅适用于 explicit 单模型单张基础生成，不能混入交接、重绘或自动修改')
    target = MODEL_TARGETS.get(request.model_id.value)
    expected_pipeline = 'anima' if request.model_id.value == 'anima_turbo' else request.model_id.value
    if request.render_pipeline.value != expected_pipeline:
        raise ValueError('区域条件的单模型管线与模型不一致')
    channel = (request.native_prompts or {}).get(target)
    if (not isinstance(channel, dict)
            or not isinstance(channel.get('positive'), str)
            or not isinstance(channel.get('negative'), str)
            or not channel['positive'].strip()):
        raise ValueError('区域条件需要对应模型完整的原生正负通道')
    if channel['positive'].strip() != request.prompt.strip():
        raise ValueError('区域条件不能沿用与主请求不同的原生正向条件')
    # Import lazily: schemas imports this module and prompting imports schemas.
    from .model_conditioning import NEGATIVE_BASELINES
    negative = request.negative_prompt.strip()
    fallback_key = 'anima' if request.model_id.value == 'anima_turbo' else request.model_id.value
    effective_negative = (negative or NEGATIVE_BASELINES[fallback_key]) if request.use_default_negative_prompt else negative
    if channel['negative'].strip() != effective_negative:
        raise ValueError('区域条件不能沿用与实际全局负面不同的原生通道')
    regions = [RegionalTextCondition.model_validate(
        item.model_dump() if isinstance(item, RegionalTextCondition) else item
    ) for item in raw]
    for region in regions:
        if region.target != target:
            raise ValueError('区域正向原生通道与当前模型不一致')
        region.pixel_box(request.width, request.height)
    return regions


def apply_regional_conditioning(workflow: dict, request) -> dict:
    """Add Core mask predictions while leaving every base node value intact."""
    regions = validate_regional_request(request)
    if not regions:
        return workflow
    samplers = [node for node in workflow.values() if node.get('class_type') == 'KSampler']
    if len(samplers) != 1:
        raise ValueError('区域条件必须连接唯一的基础采样步骤')
    sampler = samplers[0]
    positive_link = sampler['inputs']['positive']
    encoder = workflow.get(positive_link[0])
    if (positive_link[1] != 0 or not encoder
            or encoder.get('class_type') != 'CLIPTextEncode'):
        raise ValueError('区域条件需要可追溯的全局文本编码器')
    names = ['regional_empty_mask']
    names.extend(f'regional_{index}_{part}' for index in range(1, len(regions) + 1)
                 for part in ('text', 'rectangle', 'mask', 'conditioning', 'combine'))
    if any(name in workflow for name in names):
        raise ValueError('区域节点标识与已有工作流冲突')
    result = deepcopy(workflow)
    sampler_id = next(key for key, node in result.items() if node.get('class_type') == 'KSampler')
    result['regional_empty_mask'] = {'class_type': 'SolidMask', 'inputs': {
        'value': 0.0, 'width': request.width, 'height': request.height,
    }}
    combined = list(positive_link)
    for index, region in enumerate(regions, start=1):
        stem = f'regional_{index}_'
        x, y, width, height = region.pixel_box(request.width, request.height)
        positive = region.positive
        if region.target == 'newbie':
            from .prompting import ensure_system_prefix
            positive = ensure_system_prefix(positive)
        elif region.target == 'anima':
            positive = 'You are an assistant designed to generate anime images based on textual prompts.\n<Prompt Start>\n' + positive.strip()
        else:
            positive = positive.strip()
        result[stem + 'text'] = {'class_type': 'CLIPTextEncode', 'inputs': {
            'clip': deepcopy(encoder['inputs']['clip']), 'text': positive,
        }}
        result[stem + 'rectangle'] = {'class_type': 'SolidMask', 'inputs': {
            'value': 1.0, 'width': width, 'height': height,
        }}
        result[stem + 'mask'] = {'class_type': 'MaskComposite', 'inputs': {
            'destination': ['regional_empty_mask', 0], 'source': [stem + 'rectangle', 0],
            'x': x, 'y': y, 'operation': 'add',
        }}
        result[stem + 'conditioning'] = {'class_type': 'ConditioningSetMask', 'inputs': {
            'conditioning': [stem + 'text', 0], 'mask': [stem + 'mask', 0],
            'strength': region.strength, 'set_cond_area': 'default',
        }}
        result[stem + 'combine'] = {'class_type': 'ConditioningCombine', 'inputs': {
            'conditioning_1': combined, 'conditioning_2': [stem + 'conditioning', 0],
        }}
        combined = [stem + 'combine', 0]
    result[sampler_id]['inputs']['positive'] = combined
    return result
