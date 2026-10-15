"""Explicit, evidence-bound replacement of an unverified SOURCE interpretation.

This is a pure preparation boundary, not a semantic judge or state transition.
An exact quotation establishes provenance, not that either interpretation is
correct. The caller must atomically recompile the full ART/CreationIntent and
retain the returned old snapshot before making a revision effective.
"""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
import math
from typing import Any, Literal, Sequence

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .authoring_contract import SourceExtraction, parse_source_response, source_relations_with_ids
from .schemas import GenerateRequest, NewbieCompileRequest
from .source_conditions import SourceConditions, _identifier
from .source_continuity import current_source


_COLLECTIONS = ('conditions', 'objects', 'relations', 'goals', 'context',
                'uncertainties', 'conflicts')
SourceCollection = Literal['conditions', 'objects', 'relations', 'goals', 'context',
                           'uncertainties', 'conflicts']


def _json_value(value: Any) -> None:
    """Reject coercible/non-JSON claims, including NaN and non-string keys."""
    if value is None or type(value) in (str, bool, int):
        return
    if type(value) is float and math.isfinite(value):
        return
    if type(value) is list:
        for item in value:
            _json_value(item)
        return
    if type(value) is dict and all(type(key) is str for key in value):
        for item in value.values():
            _json_value(item)
        return
    raise ValueError('修订before/after必须是完整、有限的JSON值，不能隐式转换或隐藏字段')


def _canonical(value: Any) -> str:
    _json_value(value)
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(',', ':'), allow_nan=False)


def _sha(value: Any) -> str:
    return hashlib.sha256(_canonical(value).encode('utf-8')).hexdigest()


class SourceRevisionChange(BaseModel):
    """A checked explanation of actual complete records, not editable metadata.

    add/remove use null on the absent side. replace contains the complete old
    and new record. reorder contains the complete ordered lists of surviving
    identical records (the whole collection when membership is unchanged).
    Derived knowledge status/conflict/uncertainty changes also need coverage.
    """
    model_config = ConfigDict(extra='forbid')
    collection: SourceCollection
    change: Literal['add', 'remove', 'replace', 'reorder']
    before: Any
    after: Any
    source_quote: str = Field(min_length=1, max_length=4000)
    reason: str = Field(min_length=1, max_length=4000)

    @model_validator(mode='after')
    def complete_actual_claim(self):
        _json_value(self.before)
        _json_value(self.after)
        if not self.reason.strip() or not self.source_quote.strip():
            raise ValueError('每项真实修订需要非空原句引用和解释理由')
        if self.change == 'add' and (self.before is not None or self.after is None):
            raise ValueError('新增修订的before必须为null，after必须为完整新条目')
        if self.change == 'remove' and (self.before is None or self.after is not None):
            raise ValueError('删除修订的before必须为完整旧条目，after必须为null')
        if self.change == 'replace' and (self.before is None or self.after is None):
            raise ValueError('替换修订必须包含完整before/after条目')
        if self.change == 'reorder':
            if not isinstance(self.before, list) or not isinstance(self.after, list):
                raise ValueError('顺序修订必须包含完整的存续条目顺序')
            if Counter(map(_canonical, self.before)) != Counter(map(_canonical, self.after)):
                raise ValueError('顺序修订不能夹带新增、删除或改写')
        if _canonical(self.before) == _canonical(self.after):
            raise ValueError('修订不能用未变化的before/after冒充真实更正')
        return self


@dataclass(frozen=True)
class SourceRevisionResult:
    sources: SourceConditions
    receipt: dict
    previous_snapshot: dict


def _record_body(field: str, value: str, entity_ref: str, quote: str) -> dict:
    return {'field': field, 'value': value, 'entity_ref': entity_ref,
            'origin': 'user_quote', 'source_quote': quote, 'source_ref': {}}


def _candidate_records(extraction: SourceExtraction) -> list[dict]:
    records = [_record_body(item.field, item.value, item.entity_ref or 'character',
                            item.source_quote) for item in extraction.conditions]
    for obj in extraction.user_objects:
        for field in ('appearance', 'position'):
            records.extend(_record_body(field, item.value, obj.id, item.source_quote)
                           for item in getattr(obj, field))
    identifiers = [_identifier(record) for record in records]
    if len(set(identifiers)) != len(identifiers):
        raise ValueError('完整SOURCE存在重复条件或条件ID碰撞，不能依赖组装器静默去重')
    relations = [item.model_dump(mode='json') for item in extraction.user_relations]
    if len({_canonical(item) for item in relations}) != len(relations):
        raise ValueError('完整SOURCE存在重复关系，无法形成独立可追溯关系ID')
    if len(set(extraction.uncertainties)) != len(extraction.uncertainties):
        raise ValueError('完整SOURCE存在重复uncertainties，不能静默去重掩盖提案内容')
    if any(not item.strip() for item in extraction.uncertainties):
        raise ValueError('SOURCE uncertainties不能是空白条目')
    return records


def _survivors(items: list, common: Counter) -> list:
    remaining = common.copy()
    result = []
    for item in items:
        key = _canonical(item)
        if remaining[key]:
            result.append(item)
            remaining[key] -= 1
    return result


def actual_source_changes(previous_snapshot: SourceConditions | dict,
                          new_snapshot: SourceConditions | dict) -> list[dict]:
    """Display exact typed record differences, without inventing a correction.

    Removed/added records are not automatically paired as a semantic replace.
    Callers may explicitly explain such a pairing through SourceRevisionChange.
    Reorder describes only unchanged surviving records; every collection's full
    final placement remains available in the complete SOURCE snapshots.
    """
    def validated(snapshot):
        raw = snapshot.model_dump(mode='json') if isinstance(snapshot, SourceConditions) else snapshot
        return SourceConditions.model_validate(deepcopy(raw)).model_dump(mode='json')

    before, after = validated(previous_snapshot), validated(new_snapshot)
    result = []
    for collection in _COLLECTIONS:
        old, new = before[collection], after[collection]
        old_counts, new_counts = Counter(map(_canonical, old)), Counter(map(_canonical, new))
        removed, added = old_counts - new_counts, new_counts - old_counts
        for change, items, remaining in (('remove', old, removed), ('add', new, added)):
            for item in items:
                key = _canonical(item)
                if remaining[key]:
                    result.append({'collection': collection, 'change': change,
                        'before': deepcopy(item) if change == 'remove' else None,
                        'after': deepcopy(item) if change == 'add' else None})
                    remaining[key] -= 1
        common = old_counts & new_counts
        old_order, new_order = _survivors(old, common), _survivors(new, common)
        if _canonical(old_order) != _canonical(new_order):
            result.append({'collection': collection, 'change': 'reorder',
                           'before': deepcopy(old_order), 'after': deepcopy(new_order)})
    return result


def _check_changes(before: dict, after: dict, changes: list[SourceRevisionChange],
                   original_text: str) -> list[dict]:
    """Consume each real before/after difference once, including derived state."""
    actual = []
    for collection in _COLLECTIONS:
        old, new = before[collection], after[collection]
        old_counts, new_counts = Counter(map(_canonical, old)), Counter(map(_canonical, new))
        removed, added = old_counts - new_counts, new_counts - old_counts
        common = old_counts & new_counts
        old_order, new_order = _survivors(old, common), _survivors(new, common)
        reordered = _canonical(old_order) != _canonical(new_order)
        order_explained = False
        declared = [change for change in changes if change.collection == collection]
        for change in declared:
            if change.source_quote not in original_text:
                raise ValueError('修订说明的逐字引用不在同一完整原句中')
            if change.change == 'reorder':
                if (order_explained or not reordered or
                        _canonical(change.before) != _canonical(old_order) or
                        _canonical(change.after) != _canonical(new_order)):
                    raise ValueError(f'{collection}顺序说明未精确覆盖实际存续条目顺序')
                order_explained = True
            else:
                if change.change in {'remove', 'replace'}:
                    key = _canonical(change.before)
                    if not removed[key]:
                        raise ValueError(f'{collection}修订before不是尚未解释的完整实际旧条目')
                    removed[key] -= 1
                if change.change in {'add', 'replace'}:
                    key = _canonical(change.after)
                    if not added[key]:
                        raise ValueError(f'{collection}修订after不是尚未解释的完整实际新条目')
                    added[key] -= 1
            actual.append(change.model_dump(mode='json'))
        if any(removed.values()) or any(added.values()) or (reordered and not order_explained):
            raise ValueError(f'{collection}存在未逐项解释的真实更正、增删或顺序变化')
    if not actual:
        raise ValueError('SOURCE没有真实更正；不能用空diff登记假修订')
    return actual


def reinterpret_source(current_request: GenerateRequest,
                       extraction: SourceExtraction | dict,
                       changes: Sequence[SourceRevisionChange | dict], *,
                       expected_source_sha256: str) -> SourceRevisionResult:
    """Reconstruct a complete unverified SOURCE with explicit actual changes.

    No model receives authority to replace the original, identity mapping,
    selected knowledge payload, or current drawing/style controls. Professional
    source overrides require a separate explicit authority boundary: this
    entry rejects them rather than silently reactivating superseded records.
    The result is not made effective and carries no semantic certification.
    """
    if not isinstance(current_request, GenerateRequest):
        raise TypeError('SOURCE修订需要完整当前GenerateRequest，不接收局部任务片段')
    # Revalidate model instances too; Pydantic otherwise trusts mutated instances.
    request = GenerateRequest.model_validate(deepcopy(current_request.model_dump(mode='json')))
    if request.creation_intent is None:
        raise ValueError('SOURCE修订需要完整当前CreationIntent与原句')
    intent = request.creation_intent
    current = current_source(request.author_proposal, intent.original_text, intent.original_sha256)
    if current is None:
        raise ValueError('没有当前SOURCE解释；不能借历史状态绕开当前来源修订检查')
    if not isinstance(expected_source_sha256, str) or expected_source_sha256 != current.snapshot_sha256:
        raise ValueError('当前SOURCE快照SHA不匹配；修订不能应用于另一个或已经改变的解释')
    old = current.sources
    if any(item.origin == 'professional_edit' or item.status == 'superseded' or
           item.superseded_by is not None for item in old.conditions):
        raise ValueError('当前SOURCE含专业覆盖；模型重新解释入口不能撤销或重建用户授权的覆盖')
    if request.semantic_spec and request.semantic_spec.get('style_mode', old.style_mode) != old.style_mode:
        raise ValueError('当前画面模式与SOURCE不一致；本次解释修订不覆写已选style_mode')
    # Old assembled SOURCE is not new extraction input: it can contain more
    # than twelve uncertainties after adding system-derived items, or a legacy
    # empty goal list which the correction is precisely intended to address.
    old_relation_ids = [item['relation_id'] for item in source_relations_with_ids(old)]
    if len(set(old_relation_ids)) != len(old_relation_ids):
        raise ValueError('当前SOURCE关系ID重复或碰撞，无法准确逐条绑定修订')
    raw_extraction = deepcopy(extraction.model_dump(mode='json', by_alias=True)
                              if isinstance(extraction, SourceExtraction) else extraction)
    revised = SourceExtraction.model_validate(raw_extraction)
    candidate_records = _candidate_records(revised)
    explained = [SourceRevisionChange.model_validate(deepcopy(
        change.model_dump(mode='json') if isinstance(change, SourceRevisionChange) else change))
        for change in changes]
    frozen = {'identity_binding': old.identity_binding.model_dump(mode='json'),
              'knowledge': old.knowledge.model_dump(mode='json')}
    knowledge = old.knowledge
    compile_request = NewbieCompileRequest(sentence=intent.original_text, style_mode=old.style_mode,
        knowledge_pack=knowledge.pack if knowledge.selected else '',
        knowledge_operator=knowledge.operator if knowledge.selected else '',
        knowledge_appearance=knowledge.appearance if knowledge.selected else '')
    sources = parse_source_response(revised.model_dump(mode='json', by_alias=True),
                                    compile_request, deepcopy(frozen))
    # Assembly has a legacy ID-deduplication step. Neither duplicate input nor a
    # hash collision is allowed to disappear behind that implementation detail.
    if knowledge.evidence_check_scope == 'legacy_source_presence_only':
        expected_knowledge_count = (len(set(knowledge.audited_visual_facts)) +
                                    len(set(knowledge.audited_clothing_facts)))
    else:
        expected_knowledge_count = len(knowledge.visual_fact_records) + len(knowledge.clothing_fact_records)
    if (sum(item.origin == 'user_quote' for item in sources.conditions) != len(candidate_records) or
            sum(item.origin == 'knowledge_fact' for item in sources.conditions) != expected_knowledge_count):
        raise ValueError('SOURCE组装丢失条件或发生ID碰撞，未把不完整输出当作完整修订')
    condition_ids = [item.id for item in sources.conditions]
    relation_ids = [item['relation_id'] for item in source_relations_with_ids(sources)]
    if len(set(condition_ids)) != len(condition_ids) or len(set(relation_ids)) != len(relation_ids):
        raise ValueError('新SOURCE条件或关系ID碰撞，无法准确绑定修订依据')
    after = sources.model_dump(mode='json')
    before = old.model_dump(mode='json')
    for key in ('original_text', 'original_sha256', 'style_mode', 'identity_binding', 'knowledge'):
        if _canonical(before[key]) != _canonical(after[key]):
            raise ValueError(f'SOURCE修订越权改写冻结的{key}')
    actual_changes = _check_changes(before, after, explained, intent.original_text)
    checked = current_source({'source_conditions': after,
        'source_conditions_status': 'current_unverified_source_interpretation'},
        intent.original_text, intent.original_sha256)
    if checked is None:  # Defensive: a new interpretation cannot be historical.
        raise ValueError('新SOURCE未形成同一原句的完整当前解释')
    receipt = {
        'schema': 'source-revision/v1',
        'original_text_sha256': intent.original_sha256,
        'previous_source_sha256': current.snapshot_sha256,
        'revised_source_sha256': checked.snapshot_sha256,
        'input_request_sha256': _sha(request.model_dump(mode='json')),
        'complete_extraction': raw_extraction,
        'actual_changes': actual_changes,
        'collection_orders': {key: {'before': deepcopy(before[key]), 'after': deepcopy(after[key])}
                              for key in _COLLECTIONS},
        'frozen_bindings': {'style_mode': before['style_mode'], **deepcopy(frozen)},
        'knowledge_payload_unchanged': True,
        'knowledge_condition_state_recomputed': True,
        'provenance_check_scope': 'exact_original_quotes_and_frozen_bindings_only',
        'semantic_correctness_verified': False,
        'identity_accuracy_verified': False,
        'visual_accuracy_verified': False,
        'state_mutated': False,
        'new_interpretation_effective': False,
        'art_rebinding_required': True,
        'creation_intent_rebinding_required': True,
        'dispatch_performed': False,
    }
    return SourceRevisionResult(sources.model_copy(deep=True), deepcopy(receipt),
                                deepcopy(current.snapshot))
