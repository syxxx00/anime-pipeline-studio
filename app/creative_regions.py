"""Mechanical consumption of explicitly authored regions, not spatial reasoning.

This opt-in adapter has no model, service, default-method or quality decision.
The caller supplies rectangles bound to the complete CURRENT semantic draft.
Only that entity's saved fields enter its extra prediction; the complete global
request, native channels, negative and parameters remain value-for-value intact.
It does not certify the semantic draft against a historical workflow or pixels.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
import re
from typing import Annotated
from xml.etree import ElementTree as ET

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .picture_event import (CanonicalPictureChoice, canonical_projection_receipt,
                            project_canonical_picture)
from .regional_conditioning import MODEL_TARGETS, RegionalTextCondition
from .schemas import GenerateRequest, NewbieSemanticSpec


_Ref = Annotated[str, Field(min_length=1, max_length=80,
                           pattern=r'^[A-Za-z][A-Za-z0-9_-]*$')]
_Sha = Annotated[str, Field(pattern=r'^[0-9a-f]{64}$')]


class AuthoredRegion(BaseModel):
    """A declared entity and normalized rectangle; no free text or name alias."""
    model_config = ConfigDict(extra='forbid', strict=True, frozen=True)
    ref: _Ref
    x: float = Field(ge=0, lt=1, allow_inf_nan=False)
    y: float = Field(ge=0, lt=1, allow_inf_nan=False)
    width: float = Field(gt=0, le=1, allow_inf_nan=False)
    height: float = Field(gt=0, le=1, allow_inf_nan=False)
    strength: float = Field(gt=0, le=2, allow_inf_nan=False)

    @model_validator(mode='after')
    def rectangle_inside_canvas(self):
        if self.x + self.width > 1 or self.y + self.height > 1:
            raise ValueError('作者区域必须完整位于归一化画布内；未推断或移动坐标')
        return self


class RegionalConditioningSpec(BaseModel):
    """An opt-in author declaration, bound to one complete semantic snapshot."""
    model_config = ConfigDict(extra='forbid', strict=True, frozen=True)
    source_semantic_sha256: _Sha
    source_request_sha256: _Sha
    regions: list[AuthoredRegion] = Field(min_length=1, max_length=4)


@dataclass(frozen=True, slots=True)
class RegionalProjection:
    request: GenerateRequest
    receipt: dict


def _sha(value) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
        separators=(',', ':'), allow_nan=False).encode('utf-8')).hexdigest()


def _current_semantic(request: GenerateRequest) -> dict:
    raw = request.semantic_spec
    proposal = request.author_proposal
    if (not isinstance(raw, dict) or not isinstance(proposal, dict)
            or request.creation_intent is None
            or proposal.get('condition_status') == 'historical_after_native_override'):
        raise ValueError('区域消费需要当前完整作者稿的有效关联；历史或缺失规格未采用')
    if set(raw) != set(NewbieSemanticSpec.model_fields):
        raise ValueError('区域来源必须是完整的当前semantic_spec；未补默认字段或丢弃未知字段')
    for name, keys in [('objects', {'id', 'name', 'appearance', 'position'}),
                       ('relations', {'subject', 'predicate', 'object'})]:
        if (not isinstance(raw[name], list)
                or any(not isinstance(item, dict) or set(item) != keys for item in raw[name])):
            raise ValueError(f'区域来源包含不完整或未支持的{name}字段；未猜补')
    parsed = NewbieSemanticSpec.model_validate(deepcopy(raw)).model_dump(mode='json')
    if parsed != raw:
        raise ValueError('区域来源不能通过类型转换或默认值改写当前稿件')
    if (proposal.get('spec') != raw or proposal.get('semantic_sha256') != _sha(raw)
            or raw['source_sentence'] != request.creation_intent.original_text):
        raise ValueError('区域来源与当前作者稿或完整原句不同；未借用历史规格')
    ids = [obj['id'] for obj in raw['objects']]
    if (len(ids) != len(set(ids)) or set(ids) & {'character', 'environment', 'composition', 'style'}
            or any(not re.fullmatch(r'[A-Za-z][A-Za-z0-9_-]{0,79}', ref) for ref in ids)):
        raise ValueError('区域来源对象ID不明确、重复或与保留引用冲突；未按名称匹配')
    return deepcopy(raw)


def _character_position_field(request: GenerateRequest, spec: dict) -> dict:
    """Recompute the actual current projection; do not interpret prose/coordinates.

    SemanticSpec alone does not retain canonical character_position. A legacy
    draft without this explicit projection remains usable for object regions,
    but this adapter cannot safely invent its character position association.
    """
    proposal = request.author_proposal
    original = proposal.get('canonical_picture_choice')
    translated = proposal.get('translated_canonical_picture_choice')
    if not isinstance(original, dict) or (translated is not None and not isinstance(translated, dict)):
        raise ValueError('人物区域缺少当前canonical机械投影来源；未从位置文字或历史稿猜补')
    CanonicalPictureChoice.model_validate(deepcopy(original))
    effective = translated if translated is not None else original
    choice = CanonicalPictureChoice.model_validate(deepcopy(effective))
    event = project_canonical_picture(choice).model_dump(mode='json')
    receipt = canonical_projection_receipt(choice, original_choice=original)
    if (proposal.get('picture_event') != event or proposal.get('picture_projection') != receipt
            or proposal.get('positions') != event['positions']
            or proposal.get('scene_narrative') != event['scene_narrative']
            or proposal.get('layout') != event['layout']
            or proposal.get('scene_narrative_origin') != 'host_mechanical_projection_not_model_authored_prose'
            or any(spec[key] != value for key, value in event['visual_spec'].items())):
        raise ValueError('人物位置、当前规格或机械投影记录不同稿；未选取一份位置替代其他条件')
    field = 'translated_canonical_picture_choice' if translated is not None else 'canonical_picture_choice'
    return {'source_pointer': f'/author_proposal/{field}/character_position',
            'value': choice.character_position, 'native_slot': 'position',
            'matching_source_pointers': ['/author_proposal/positions/character',
                                        '/author_proposal/picture_event/positions/character'],
            'projection_receipt_pointer': '/author_proposal/picture_projection',
            'effective_choice_sha256': receipt['effective_choice_sha256']}


def _fields(spec: dict, ref: str, request: GenerateRequest) -> list[dict]:
    """Return exact values and pointers, without names/shape-based interpretation."""
    if ref == 'character':
        selected = [(f'/semantic_spec/{key}', spec[key], slot) for key, slot in [
            ('character_name', 'n'), ('character_tag', 'n'), ('source', 'n'),
            ('count', 'gender'), ('appearance', 'appearance'), ('clothing', 'clothing'),
            ('expression', 'expression'), ('body_action', 'action'),
            ('left_hand', 'left_hand'), ('right_hand', 'right_hand')]]
    else:
        matches = [(index, obj) for index, obj in enumerate(spec['objects']) if obj['id'] == ref]
        if len(matches) != 1:
            raise ValueError(f'区域引用不是character或当前声明对象ID：{ref}')
        index, obj = matches[0]
        selected = [(f'/semantic_spec/objects/{index}/{key}', obj[key], key)
                    for key in ('name', 'appearance', 'position')]
    result = []
    for pointer, raw, slot in selected:
        values = raw if isinstance(raw, list) else [raw]
        for index, value in enumerate(values):
            if not isinstance(value, str) or (value and not value.strip()):
                raise ValueError(f'区域字段必须是明确原文，不能猜补空白或转换类型：{pointer}')
            if value:
                field = {'source_pointer': pointer + (f'/{index}' if isinstance(raw, list) else ''),
                         'value': value, 'native_slot': slot}
                if slot in {'left_hand', 'right_hand'}:
                    field.update(native_slot='action', serialization_prefix=slot.replace('_', ' ') + ': ')
                result.append(field)
    if ref == 'character':
        result.append(_character_position_field(request, spec))
    if not result:
        raise ValueError(f'区域对象没有可对应的实际文本条件：{ref}')
    return result


def _positive(target: str, ref: str, fields: list[dict]) -> str:
    """Published NewBie slots; other targets keep their own text consumer form.

    No quality, identity, pose, style or relationship defaults are added.
    The existing regional consumer adds the NewBie system/Anima wrapper once.
    """
    if target == 'newbie':
        root = ET.Element('img')
        if ref == 'character':
            character = ET.SubElement(root, 'character_1')
            for slot in dict.fromkeys(field['native_slot'] for field in fields):
                ET.SubElement(character, slot).text = ', '.join(
                    field.get('serialization_prefix', '') + field['value']
                    for field in fields if field['native_slot'] == slot)
        else:
            general = ET.SubElement(root, 'general_tags')
            parts = []
            for slot in ('name', 'appearance', 'position'):
                values = [field['value'] for field in fields if field['native_slot'] == slot]
                if values:
                    parts.append(', '.join(values) if slot == 'name' else f'{slot}: ' + ', '.join(values))
            ET.SubElement(general, 'objects').text = '; '.join(parts)
        return ET.tostring(root, encoding='unicode', short_empty_elements=False)
    if target in {'illustrious', 'anima', 'z_image'}:
        parts = []
        for field in fields:
            # Field labels preserve laterality rather than guessing from content.
            parts.append(field.get('serialization_prefix', '') + field['value'])
        return ', '.join(parts)
    raise ValueError(f'区域对象暂无可支持的原生格式：{target}')


def compile_authored_regions(request: GenerateRequest,
                            declarations: RegionalConditioningSpec | dict) -> RegionalProjection:
    """Change ONLY regional_conditioning on a validated current request copy.

    Binding hashes are input correspondence, not source/identity/art approval.
    The existing regional validator retains single-model/operation exclusions.
    """
    if not isinstance(request, GenerateRequest):
        raise TypeError('区域消费需要一个完整GenerateRequest，不接收局部参数模板')
    base = request.model_dump(mode='json')
    checked = GenerateRequest.model_validate(deepcopy(base))
    if checked.model_dump(mode='json') != base:
        raise ValueError('当前请求重校验会改变条件；未简化或静默修复')
    if checked.regional_conditioning is not None:
        raise ValueError('当前请求已有明确区域条件；未覆盖另一份区域选择')
    spec = _current_semantic(checked)
    raw_declarations = (declarations.model_dump(mode='json') if isinstance(declarations, RegionalConditioningSpec)
                        else deepcopy(declarations))
    declared = RegionalConditioningSpec.model_validate(deepcopy(raw_declarations))
    semantic_sha = _sha(spec)
    if declared.source_semantic_sha256 != semantic_sha:
        raise ValueError('区域几何绑定的作者稿SHA已变化；未沿用旧区域')
    if declared.source_request_sha256 != _sha(base):
        raise ValueError('区域几何绑定的完整当前请求SHA已变化；未沿用旧区域或旧全局条件')
    target = MODEL_TARGETS.get(checked.model_id.value)
    if target is None:
        raise ValueError('当前模型没有可消费的区域原生通道')
    native = (checked.native_prompts or {}).get(target) or {}
    if native.get('trace_status') in {'unverified_manual_override', 'explicit_user_override_unverified',
                                     'historical_projection_after_native_override'}:
        raise ValueError('原生条件已手动覆盖而规格没有有效当前关联；未借用历史对象条件')
    conditions, bindings = [], []
    for index, region in enumerate(declared.regions):
        fields = _fields(spec, region.ref, checked)
        positive = _positive(target, region.ref, fields)
        values = region.model_dump(mode='json', exclude={'ref'})
        condition = RegionalTextCondition(target=target, positive=positive, **values)
        condition.pixel_box(checked.width, checked.height)
        conditions.append(condition.model_dump(mode='json'))
        bindings.append({'region_index': index, 'ref': region.ref, 'target': target,
            'geometry_source_pointer': f'/regions/{index}', 'geometry': region.model_dump(mode='json'),
            'source_fields': fields, 'positive_sha256': hashlib.sha256(positive.encode('utf-8')).hexdigest(),
            'scope': 'exact_current_entity_fields_not_inferred_pixel_observation'})
    output = GenerateRequest.model_validate({**deepcopy(base), 'regional_conditioning': conditions})
    restored = output.model_dump(mode='json')
    restored['regional_conditioning'] = None
    if restored != base:
        raise ValueError('区域投影改变了完整全局条件或参数；没有返回部分请求')
    return RegionalProjection(output, {
        'source_semantic_sha256': semantic_sha, 'base_request_sha256': _sha(base),
        'declarations_sha256': _sha(raw_declarations), 'output_request_sha256': _sha(output.model_dump(mode='json')),
        'bindings': bindings, 'scope': 'mechanical_entity_condition_projection_only',
        'historical_workflow_equivalence_verified': False,
        'semantic_correctness_verified': False, 'quality_verified': False,
        'method_selected_or_dispatched': False,
    })
