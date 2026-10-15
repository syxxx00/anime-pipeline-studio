"""Traceable input conditions, separate from replaceable author proposals.

Exact quotes establish where an interpretation came from, not that the
translation, intent interpretation or resulting character image is correct.
This module neither asks the user for expert knowledge nor assigns model roles.
"""
from __future__ import annotations

import hashlib
import json
import re
from typing import Literal, Sequence

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .knowledge import declared_visual_fact_records
from .schemas import (NewbieCompileRequest, NewbieObjectSpec, NewbieRelationSpec,
                      NewbieSemanticSpec, NewbieStyleMode)


ConditionField = Literal['count', 'appearance', 'clothing', 'expression', 'body_action',
    'left_hand', 'right_hand', 'framing', 'background', 'lighting', 'style', 'exclusions']
Origin = Literal['user_quote', 'knowledge_fact', 'professional_edit']


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra='forbid')


class QuotedValue(_StrictModel):
    value: str = Field(min_length=1, max_length=1200)
    source_quote: str = Field(min_length=1, max_length=4000)


class QuotedUserCondition(QuotedValue):
    field: ConditionField
    entity_ref: str | None = Field(default=None, max_length=80)


class QuotedUserObject(_StrictModel):
    id: str = Field(pattern=r'^[a-zA-Z][a-zA-Z0-9_-]{0,79}$')
    name: str = Field(min_length=1, max_length=160)
    source_quote: str = Field(min_length=1, max_length=4000)
    appearance: list[QuotedValue] = Field(default_factory=list, max_length=12)
    position: list[QuotedValue] = Field(default_factory=list, max_length=8)


class QuotedUserRelation(_StrictModel):
    subject: str = Field(min_length=1, max_length=80)
    predicate: str = Field(min_length=1, max_length=180)
    object: str = Field(min_length=1, max_length=80)
    source_quote: str = Field(min_length=1, max_length=4000)


class SourceGoal(_StrictModel):
    description: str = Field(min_length=1, max_length=1000)
    source_quote: str = Field(min_length=1, max_length=4000)
    priority: Literal['required', 'preferred', 'context'] = 'required'


class SourceContext(_StrictModel):
    description: str = Field(min_length=1, max_length=1500)
    source_quote: str = Field(min_length=1, max_length=4000)


class ProfessionalSourceEdit(_StrictModel):
    """An explicit later choice; it never impersonates an original quote."""
    field: ConditionField | Literal['position']
    value: str = Field(min_length=1, max_length=1200)
    entity_ref: str = Field(default='character', max_length=80)
    supersedes: list[str] = Field(default_factory=list, max_length=60)
    edit_note: str = Field(min_length=1, max_length=1500)


class IdentityBinding(_StrictModel):
    recognized: bool = False
    character_name: str = ''
    character_tag: str = ''
    source: str = ''
    mapping_source: str = ''
    visual_accuracy_verified: Literal[False] = False

    @model_validator(mode='after')
    def complete_mapping(self):
        if self.recognized and (not self.character_name or not self.character_tag or
                                self.character_tag == 'original_character'):
            raise ValueError('已识别身份需要完整名称与标签；名称映射不是视觉验证')
        return self


class KnowledgeFactRecord(_StrictModel):
    """A frozen stored declaration; none of its labels certify the actual art."""
    value: str = Field(min_length=1)
    record_id: str = Field(min_length=1)
    record_id_origin: Literal['stored_id', 'record_content_sha256']
    record_sha256: str = Field(pattern=r'^[0-9a-f]{64}$')
    source_binding_sha256: str = Field(pattern=r'^[0-9a-f]{64}$')
    raw_record: dict
    source_url: str = Field(min_length=1)
    record_source_url: str
    source_url_origin: Literal['record', 'owner_fallback']
    declared_status: Literal['verified']
    evidence_type: str
    asset_key: str
    evidence_check_scope: Literal['declared_source_presence_only']
    local_original_checked: Literal[False] = False
    review_receipt_checked: Literal[False] = False
    visual_accuracy_verified: Literal[False] = False

    @model_validator(mode='after')
    def declaration_consistent(self):
        records, rejected = declared_visual_fact_records({'fact': [self.raw_record]}, 'fact', self.source_url)
        if rejected or len(records) != 1 or records[0] != self.model_dump(mode='json'):
            raise ValueError('知识条目的值、来源、声明字段或哈希与所存原record不一致；未修补或重新核验视觉')
        return self


class KnowledgeSnapshot(_StrictModel):
    selected: bool = False
    pack: str = ''
    operator: str = ''
    appearance: str = ''
    character_name: str = ''
    character_tag: str = ''
    appearance_name: str = ''
    audited_visual_facts: list[str] = Field(default_factory=list)
    audited_clothing_facts: list[str] = Field(default_factory=list)
    visual_fact_records: list[KnowledgeFactRecord] = Field(default_factory=list)
    clothing_fact_records: list[KnowledgeFactRecord] = Field(default_factory=list)
    evidence_check_scope: Literal['legacy_source_presence_only', 'declared_source_presence_only'] = 'legacy_source_presence_only'
    source_url: str = ''
    rejected_fact_count: int = 0
    identity_accuracy_verified: Literal[False] = False

    @model_validator(mode='after')
    def evidence_values_consistent(self):
        record_groups = ((self.audited_visual_facts, self.visual_fact_records),
                         (self.audited_clothing_facts, self.clothing_fact_records))
        if self.evidence_check_scope == 'legacy_source_presence_only':
            if self.visual_fact_records or self.clothing_fact_records:
                raise ValueError('旧知识快照不能附带新逐条证据却隐藏检查范围')
        else:
            for values, records in record_groups:
                if list(dict.fromkeys(item.value for item in records)) != values:
                    raise ValueError('知识快照的逐条依据与实际注入值不一致，不能遗漏或附加事实')
                identifiers = [item.record_id for item in records]
                if len(identifiers) != len(set(identifiers)):
                    raise ValueError('同一知识字段的record ID重复，无法准确追溯条目')
        if not self.selected and (self.audited_visual_facts or self.audited_clothing_facts or
                                 self.visual_fact_records or self.clothing_fact_records):
            raise ValueError('未选择知识却附带知识事实或条目依据')
        return self


class SourceCondition(_StrictModel):
    id: str
    field: str
    value: str
    entity_ref: str = 'character'
    origin: Origin
    source_quote: str = ''
    source_ref: dict[str, str] = Field(default_factory=dict)
    attributes: dict[str, str] = Field(default_factory=dict)
    status: Literal['active', 'unresolved', 'superseded'] = 'active'
    superseded_by: str | None = None
    provenance_checked: Literal[True] = True
    semantic_correctness_verified: Literal[False] = False
    visual_accuracy_verified: Literal[False] = False


class SourceConflict(_StrictModel):
    kind: Literal['material_attribute', 'identity']
    description: str
    condition_ids: list[str] = Field(default_factory=list)
    attribute: str = ''
    blocking: Literal[True] = True


class SourceConditions(_StrictModel):
    original_text: str
    original_sha256: str
    style_mode: NewbieStyleMode = NewbieStyleMode.STANDARD
    identity_binding: IdentityBinding
    knowledge: KnowledgeSnapshot
    conditions: list[SourceCondition] = Field(default_factory=list)
    objects: list[QuotedUserObject] = Field(default_factory=list)
    relations: list[QuotedUserRelation] = Field(default_factory=list)
    goals: list[SourceGoal] = Field(default_factory=list)
    context: list[SourceContext] = Field(default_factory=list)
    conflicts: list[SourceConflict] = Field(default_factory=list)
    uncertainties: list[str] = Field(default_factory=list)
    semantic_correctness_verified: Literal[False] = False
    identity_accuracy_verified: Literal[False] = False

    @property
    def blocking(self) -> bool:
        return bool(self.conflicts)


def _identifier(value: dict) -> str:
    data = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'))
    return 'src_' + hashlib.sha256(data.encode('utf-8')).hexdigest()[:20]


_COLORS = r'black|white|red|blue|green|brown|gray|grey|silver|golden|gold|yellow|purple|violet|pink|amber|orange|hazel|cyan|teal'
_COLOR_ALIASES = {'grey': 'gray', 'golden': 'gold'}


def _attributes(field: str, value: str) -> dict[str, str]:
    """A deliberately small literal-tag check, not a semantic ontology.

    Only an unambiguous single color immediately attached to a body/garment
    noun is comparable. Mixed descriptions and other properties remain unknown.
    This must not certify translations or invent omitted character attributes.
    """
    result = {}
    if field not in {'appearance', 'clothing', 'count'}:
        return result
    if field == 'count':
        return {'count': ' '.join(value.strip().lower().split())}
    nouns = ('eyes?', 'hair') if field == 'appearance' else ('dress', 'robe', 'uniform', 'bodysuit', 'armor')
    lowered = value.lower()
    if re.search(r'\b(?:not|without|except|instead|no)\b', lowered):
        return result
    if field == 'clothing':
        primary = set(re.findall(r'\b(dress|robe|uniform|bodysuit|armor)\b', lowered))
        if len(primary) == 1:
            result['clothing.primary_type'] = next(iter(primary))
        for noun, group in (('veil', 'headwear'), ('gloves', 'hands'), ('cape', 'outerwear')):
            if re.search(rf'\b{noun}\b', lowered):
                result[f'clothing.{group}.type'] = noun
    for noun in nouns:
        matches = re.findall(rf'\b({_COLORS})\s+({noun})\b', lowered)
        colors = {_COLOR_ALIASES.get(color, color) for color, _ in matches}
        if len(colors) == 1:
            name = 'eyes' if noun == 'eyes?' else noun
            result[f'{field}.{name}.color'] = next(iter(colors))
    return result


def _record(field: str, value: str, *, entity_ref: str = 'character', origin: Origin,
            source_quote: str = '', source_ref: dict[str, str] | None = None) -> SourceCondition:
    if not value.strip():
        raise ValueError('来源条件不能是空白值')
    data = dict(field=field, value=value, entity_ref=entity_ref, origin=origin,
                source_quote=source_quote, source_ref=source_ref or {})
    return SourceCondition(id=_identifier(data), **data, attributes=_attributes(field, value))


def _quote(original: str, quote: str) -> None:
    if not quote.strip() or quote not in original:
        raise ValueError('来源引用必须是完整原句中的非空精确片段；引用不证明语义正确')


def _references(entity_ref: str, field: str, object_ids: set[str]) -> None:
    if entity_ref != 'character' and entity_ref not in object_ids:
        raise ValueError(f'来源条件引用未声明实体：{entity_ref}')
    if entity_ref != 'character' and field not in {'appearance', 'position'}:
        raise ValueError('对象来源条件只支持外形与位置；不能悄悄套入角色字段')


def _reconcile(records: list[SourceCondition]) -> tuple[list[SourceConflict], list[str]]:
    conflicts = []
    uncertainties = []
    # User/professional statements remain separate from knowledge; unspecified
    # named characters can use audited facts without manufacturing new ones.
    for index, left in enumerate(records):
        if left.status == 'superseded':
            continue
        for right in records[index + 1:]:
            if right.status == 'superseded' or (left.field, left.entity_ref) != (right.field, right.entity_ref):
                continue
            if left.value == right.value:
                continue
            shared = set(left.attributes) & set(right.attributes)
            # Garment types can be layered; unlike literal opposing eye/hair
            # colors, a robe/dress pair is an uncertain choice, not a certified
            # contradiction. Keep those alternatives apart without a forced
            # ordinary-user clarification.
            differing = [key for key in shared if left.attributes[key] != right.attributes[key]]
            contradictory = [key for key in differing if key != 'clothing.primary_type']
            if contradictory:
                left.status = right.status = 'unresolved'
                conflicts.append(SourceConflict(kind='material_attribute',
                    attribute=', '.join(sorted(contradictory)), condition_ids=[left.id, right.id],
                    description='两份来源对同一可识别属性给出不同值；没有替换、拼接或声称已解决。'))
            elif (left.origin == 'knowledge_fact') != (right.origin == 'knowledge_fact'):
                if left.attributes and right.attributes and not differing:
                    # Comparable equal or distinct literal attributes, e.g.
                    # silver hair and blue eyes, do not require a user question.
                    continue
                knowledge = left if left.origin == 'knowledge_fact' else right
                knowledge.status = 'unresolved'
                uncertainties.append(f'来源兼容性未确定：{left.id} / {right.id}；暂不注入这一知识补充，未要求用户补专业知识。')
    return conflicts, list(dict.fromkeys(uncertainties))


def assemble_source_conditions(request: NewbieCompileRequest, *,
        identity_binding: IdentityBinding | dict, knowledge: KnowledgeSnapshot | dict,
        user_conditions: Sequence[QuotedUserCondition | dict] = (),
        user_objects: Sequence[QuotedUserObject | dict] = (),
        user_relations: Sequence[QuotedUserRelation | dict] = (),
        goals: Sequence[SourceGoal | dict] = (), context: Sequence[SourceContext | dict] = (),
        professional_overrides: Sequence[ProfessionalSourceEdit | dict] = ()) -> SourceConditions:
    binding = IdentityBinding.model_validate(identity_binding).model_copy(deep=True)
    facts = KnowledgeSnapshot.model_validate(knowledge).model_copy(deep=True)
    if facts.selected != bool(request.knowledge_pack):
        raise ValueError('知识快照与本次资源选择不一致')
    if facts.selected:
        selected = (request.knowledge_pack, request.knowledge_operator, request.knowledge_appearance)
        if selected != (facts.pack, facts.operator, facts.appearance):
            raise ValueError('知识快照不是本次所选角色和外观')
        if not facts.character_name or not facts.character_tag:
            raise ValueError('所选知识缺少角色身份映射')
        if (facts.audited_visual_facts or facts.audited_clothing_facts) and not facts.source_url:
            raise ValueError('知识事实缺少来源地址，不注入无来源事实')
    elif facts.audited_visual_facts or facts.audited_clothing_facts:
        raise ValueError('未选择知识却附带知识事实')
    objects = [QuotedUserObject.model_validate(item).model_copy(deep=True) for item in user_objects]
    object_ids = {item.id for item in objects}
    if len(object_ids) != len(objects) or 'character' in object_ids:
        raise ValueError('用户对象编号重复或占用了character保留引用')
    records = []
    for item in user_conditions:
        condition = QuotedUserCondition.model_validate(item)
        _quote(request.sentence, condition.source_quote)
        ref = condition.entity_ref or 'character'
        _references(ref, condition.field, object_ids)
        records.append(_record(condition.field, condition.value, entity_ref=ref,
            origin='user_quote', source_quote=condition.source_quote))
    for obj in objects:
        _quote(request.sentence, obj.source_quote)
        for field in ('appearance', 'position'):
            for item in getattr(obj, field):
                _quote(request.sentence, item.source_quote)
                records.append(_record(field, item.value, entity_ref=obj.id,
                    origin='user_quote', source_quote=item.source_quote))
    relations = [QuotedUserRelation.model_validate(item).model_copy(deep=True) for item in user_relations]
    for relation in relations:
        _quote(request.sentence, relation.source_quote)
        if relation.subject not in object_ids | {'character'} or relation.object not in object_ids | {'character'}:
            raise ValueError('用户关系只能连接character或声明过的对象ID，不能悄悄更换对象')
    goal_items = [SourceGoal.model_validate(item).model_copy(deep=True) for item in goals]
    context_items = [SourceContext.model_validate(item).model_copy(deep=True) for item in context]
    for item in [*goal_items, *context_items]:
        _quote(request.sentence, item.source_quote)
    if facts.selected:
        reference = {'source_url': facts.source_url, 'pack': facts.pack, 'operator': facts.operator,
                     'appearance': facts.appearance, 'snapshot_status': 'stored_declarations_not_reaudited',
                     'evidence_check_scope': facts.evidence_check_scope,
                     'local_original_checked': 'false', 'review_receipt_checked': 'false',
                     'visual_accuracy_verified': 'false'}
        for field, values, fact_records in (
                ('appearance', facts.audited_visual_facts, facts.visual_fact_records),
                ('clothing', facts.audited_clothing_facts, facts.clothing_fact_records)):
            if facts.evidence_check_scope == 'legacy_source_presence_only':
                records.extend(_record(field, value, origin='knowledge_fact', source_ref=reference) for value in values)
            else:
                for item in fact_records:
                    record_reference = {**reference, **{key: getattr(item, key) for key in (
                        'record_id', 'record_id_origin', 'record_sha256', 'source_binding_sha256', 'source_url',
                        'record_source_url', 'source_url_origin', 'declared_status', 'evidence_type',
                        'asset_key', 'evidence_check_scope')}}
                    records.append(_record(field, item.value, origin='knowledge_fact', source_ref=record_reference))
    records = list({item.id: item for item in records}.values())
    for item in professional_overrides:
        edit = ProfessionalSourceEdit.model_validate(item)
        _references(edit.entity_ref, edit.field, object_ids)
        record = _record(edit.field, edit.value, entity_ref=edit.entity_ref, origin='professional_edit',
                         source_ref={'edit_note': edit.edit_note})
        current = {item.id: item for item in records}
        for identifier in edit.supersedes:
            previous = current.get(identifier)
            if previous is None or previous.status == 'superseded':
                raise ValueError('专业覆盖必须明确指向尚有效的来源记录，不能删掉未知或旧记录')
            if (previous.field, previous.entity_ref) != (edit.field, edit.entity_ref):
                raise ValueError('专业覆盖不能跨字段或跨实体静默删除其他要求')
            previous.status, previous.superseded_by = 'superseded', record.id
        if record.id in current:
            raise ValueError('专业编辑记录重复')
        records.append(record)
    conflicts, uncertainties = _reconcile(records)
    if facts.selected and binding.recognized and binding.character_tag != facts.character_tag:
        conflicts.append(SourceConflict(kind='identity', description='原句身份映射与所选知识角色不同，未自动覆盖任一来源。'))
    if not binding.recognized and not facts.selected:
        uncertainties.append('没有已识别的既有角色映射或已选知识；来源层不猜测性别、服装和外貌。')
    elif not facts.selected:
        uncertainties.append('既有角色仅有名称标签映射；外貌识别仍是未验证的生成模型先验。')
    return SourceConditions(original_text=request.sentence,
        original_sha256=hashlib.sha256(request.sentence.encode('utf-8')).hexdigest(),
        style_mode=request.style_mode,
        identity_binding=binding, knowledge=facts, conditions=records, objects=objects,
        relations=relations, goals=goal_items, context=context_items,
        conflicts=conflicts, uncertainties=uncertainties)


def materialize_source_spec(sources: SourceConditions, base_spec: NewbieSemanticSpec | None = None) -> NewbieSemanticSpec:
    """Adapt active sourced values, never unresolved alternatives, to a spec.

    This is a source-first adapter, not the art-direction merge algorithm. An
    optional base supplies unspecified fields for legacy/professional use;
    sourced fields deliberately win over that base. The current authoring path
    should materialize pure sources first, then add separately identified art
    choices so meaningful compatible elaboration is not erased wholesale.
    Source-to-source reconciliation retains only compatible active values.
    The old destructive apply_knowledge step must not run after this adapter.
    """
    if sources.blocking:
        raise ValueError('来源存在未解决的实质冲突，不能拼接为生成条件')
    if base_spec is not None and base_spec.source_sentence != sources.original_text:
        raise ValueError('基础规格必须保留同一完整原句')
    spec = base_spec.model_copy(deep=True) if base_spec else NewbieSemanticSpec(
        source_sentence=sources.original_text, count='solo')
    spec.style_mode = sources.style_mode
    legacy_character_refs = {spec.character_name, spec.character_tag, 'character'}
    legacy_object_names = {obj.id: obj.name for obj in spec.objects}
    binding, knowledge = sources.identity_binding, sources.knowledge
    named = binding.recognized or knowledge.selected
    if knowledge.selected:
        spec.character_name, spec.character_tag = knowledge.character_name, knowledge.character_tag
        spec.source = binding.source if binding.recognized else knowledge.pack
    elif binding.recognized:
        spec.character_name, spec.character_tag, spec.source = binding.character_name, binding.character_tag, binding.source
    active = [item for item in sources.conditions if item.status == 'active']
    # A named character's unsupported base appearance cannot sneak in merely
    # because no sourced appearance was specified. Original characters remain
    # available for the subsequent replaceable author proposal.
    if named:
        spec.appearance, spec.clothing = [], []
    character_fields = {item.field for item in active if item.entity_ref == 'character'}
    for field in character_fields:
        values = list(dict.fromkeys(item.value for item in active if item.entity_ref == 'character' and item.field == field))
        if field == 'count':
            if len(values) != 1:
                raise ValueError('多份人数条件没有形成唯一当前选择')
            spec.count = values[0]
        else:
            setattr(spec, field, values)
    base_objects = {obj.id: obj for obj in spec.objects}
    for obj in sources.objects:
        current = base_objects.get(obj.id)
        appearance = list(dict.fromkeys(item.value for item in active if item.entity_ref == obj.id and item.field == 'appearance'))
        position = list(dict.fromkeys(item.value for item in active if item.entity_ref == obj.id and item.field == 'position'))
        base_objects[obj.id] = NewbieObjectSpec(id=obj.id, name=obj.name,
            appearance=appearance or (current.appearance if current else []),
            position=position or (current.position if current else []))
    spec.objects = list(base_objects.values())
    user_relations = [NewbieRelationSpec(subject=item.subject, predicate=item.predicate, object=item.object)
                      for item in sources.relations]
    # Sourced relation pairs replace the proposal's competing predicate for that
    # same directed pair; otherwise proposing "holding stone" could survive
    # beside the user's "looking at stone" as an unintended second requirement.
    pairs = {(item.subject, item.object) for item in user_relations}
    character_refs = legacy_character_refs | {spec.character_name, spec.character_tag}
    object_aliases: dict[str, set[str]] = {}
    for obj in spec.objects:
        for alias in (obj.id, obj.name, legacy_object_names.get(obj.id, '')):
            if alias:
                object_aliases.setdefault(alias, set()).add(obj.id)

    def canonical(ref: str) -> str:
        if ref in character_refs:
            return 'character'
        candidates = object_aliases.get(ref, set())
        return next(iter(candidates)) if len(candidates) == 1 else ref

    spec.relations = user_relations + [item for item in spec.relations
        if (canonical(item.subject), canonical(item.object)) not in pairs]
    return NewbieSemanticSpec.model_validate(spec.model_dump(mode='json'))
