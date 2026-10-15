"""Track a current SOURCE interpretation without certifying its meaning.

An artistic revision may change realization, but cannot silently erase the
SOURCE records it inherited. This is continuity of an interpretation, not a
claim that its translation, character facts, or resulting pixels are correct.
Explicit SOURCE reinterpretation is not implemented by this helper.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json

from .scene_contract import Evidence, EvidenceKind, SceneContractV3, SourceSpan
from .schemas import NewbieSemanticSpec
from .source_conditions import SourceConditions


def _fingerprint(value) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                    separators=(',', ':')).encode('utf-8')).hexdigest()


@dataclass(frozen=True)
class CurrentSource:
    snapshot: dict
    sources: SourceConditions
    snapshot_sha256: str


def current_source(proposal: dict | None, original_text: str, original_sha256: str) -> CurrentSource | None:
    """Read only an explicitly current or initial source-scoped snapshot.

    Historical/manual native overrides remain historical. Old drafts without
    SOURCE are not inferred to contain frozen commitments.
    """
    if not proposal or 'source_conditions' not in proposal:
        return None
    status = proposal.get('source_conditions_status', '')
    if (status == 'historical_source_interpretation_after_user_revision' or
            proposal.get('condition_status') == 'historical_after_native_override'):
        return None
    if status not in {'', 'current_unverified_source_interpretation'}:
        raise ValueError('SOURCE状态无法确认为当前解释；请重新解释SOURCE，未将历史原件冒充当前条件')
    raw = deepcopy(proposal['source_conditions'])
    sources = SourceConditions.model_validate(raw)
    actual_sha256 = hashlib.sha256(original_text.encode('utf-8')).hexdigest()
    if (original_sha256 != actual_sha256 or sources.original_text != original_text or
            sources.original_sha256 != actual_sha256):
        raise ValueError('当前SOURCE原句或SHA与本次任务不一致；请重新解释SOURCE，未静默替换')
    quotes = [item.source_quote for item in sources.conditions if item.origin == 'user_quote']
    quotes.extend(item.source_quote for item in [*sources.objects, *sources.relations, *sources.goals, *sources.context])
    quotes.extend(item.source_quote for obj in sources.objects for item in [*obj.appearance, *obj.position])
    if any(not quote.strip() or quote not in original_text for quote in quotes):
        raise ValueError('当前SOURCE逐条引用不在完整原句中；请重新解释SOURCE，未认证来源或语义')
    condition_ids = [item.id for item in sources.conditions]
    object_ids = [item.id for item in sources.objects]
    if len(set(condition_ids)) != len(condition_ids) or len(set(object_ids)) != len(object_ids):
        raise ValueError('当前SOURCE编号重复，无法逐条承接；请重新解释SOURCE')
    return CurrentSource(raw, sources, _fingerprint(raw))


def _failure(identifier: str) -> ValueError:
    return ValueError('当前作者方案遗漏或改变SOURCE解释 ' + identifier +
        '；需显式重新解释SOURCE后再重建，未自动补回旧词或认证语义')


def _field_owner(record, character_id: str) -> tuple[str, str]:
    owner = character_id if record.entity_ref == 'character' else record.entity_ref
    return {'count': (character_id, 'gender'), 'framing': ('composition', 'framing'),
        'style': ('style', 'description'), 'exclusions': ('style', 'exclusions'),
        'background': ('environment', 'background'), 'lighting': ('environment', 'lighting')}.get(
            record.field, (owner, record.field))


def _groups(contract: SceneContractV3) -> dict[tuple[str, str], list]:
    result = {}
    for entity in contract.entities:
        for condition in entity.constraints:
            result.setdefault((entity.id, condition.name), []).append(condition)
    for owner, conditions in (('composition', contract.composition.constraints),
                               ('style', contract.style.constraints)):
        for condition in conditions:
            result.setdefault((owner, condition.name), []).append(condition)
    return result


def validate_source_continuity(source: CurrentSource, spec: NewbieSemanticSpec) -> None:
    """Exact, conservative continuity check; no paraphrase/intent equivalence judge."""
    from .contract_builder import contract_from_semantic
    sources = source.sources
    if spec.source_sentence != sources.original_text:
        raise _failure('original_text')
    if sources.blocking:
        raise _failure('unresolved_source_conflicts')
    # Strict parsing is itself only endpoint integrity, not semantic evidence.
    contract = contract_from_semantic(spec, strict_entity_refs=True)
    character = next(item for item in contract.entities if item.kind.value == 'character')
    identity = sources.knowledge if sources.knowledge.selected else sources.identity_binding
    if sources.knowledge.selected or sources.identity_binding.recognized:
        if (spec.character_name, spec.character_tag) != (identity.character_name, identity.character_tag):
            raise _failure('identity_binding')
    objects = {item.id: item for item in spec.objects}
    for obj in sources.objects:
        if obj.id not in objects or objects[obj.id].name != obj.name:
            raise _failure('object:' + obj.id)
    groups = _groups(contract)
    for record in sources.conditions:
        if record.status != 'active':
            continue
        available = [value for item in groups.get(_field_owner(record, character.id), [])
                     for value in item.values]
        if record.value not in available:
            raise _failure(record.id)
    for relation in sources.relations:
        subject = character.id if relation.subject == 'character' else relation.subject
        obj = character.id if relation.object == 'character' else relation.object
        matches = [item for item in contract.relations if
                   (item.subject_ref, item.predicate, item.object_ref) == (subject, relation.predicate, obj)]
        if len(matches) != 1:
            raise _failure('relation:' + _fingerprint(relation.model_dump(mode='json'))[:20])


def validate_source_author_conditions(source: CurrentSource, draft, spec: NewbieSemanticSpec) -> None:
    """Bind goals and identity declarations to the current interpretation.

    A substring of the original cannot authorize arbitrary translated values.
    Exact binding preserves provenance only; it does not certify the SOURCE
    translation or make a mistaken interpretation impossible to revise.
    """
    sources = source.sources
    if [item.model_dump(mode='json') for item in draft.goals] != [
            item.model_dump(mode='json') for item in sources.goals]:
        raise _failure('goals')
    user_bindings = {(item.field, item.value, item.source_quote)
                     for item in sources.conditions
                     if item.status == 'active' and item.origin == 'user_quote'
                     and item.entity_ref == 'character'}
    for evidence in draft.identity_evidence:
        if (evidence.field, evidence.value, evidence.source_quote) not in user_bindings:
            raise _failure('identity_evidence:' + evidence.field)
    if sources.knowledge.selected or sources.identity_binding.recognized:
        for field in ('appearance', 'clothing'):
            allowed = {item.value for item in sources.conditions
                       if item.status == 'active' and item.entity_ref == 'character'
                       and item.field == field}
            if set(getattr(spec, field)) - allowed:
                raise _failure('identity_values:' + field)


def attribute_source_continuity(source: CurrentSource, current: SceneContractV3) -> tuple[SceneContractV3, dict]:
    """Restore value-level provenance, preserving any genuinely reviewed role.

    No unverified SOURCE record gains REQUIREMENT/verified status. Mixed generic
    conditions are split so provenance does not silently certify other values.
    Relations are matched by strict triples, never by their previous list order.
    """
    contract = current.model_copy(deep=True)
    sources = source.sources
    character = next(item for item in contract.entities if item.kind.value == 'character')
    by_value = {}
    for record in sources.conditions:
        if record.status == 'active':
            by_value.setdefault((*_field_owner(record, character.id), record.value), []).append(record)
    mappings = []

    def annotate(owner, conditions):
        result = []
        for condition in conditions:
            sourced = [(value, by_value.get((owner, condition.name, value), [])) for value in condition.values]
            sourced = [(value, records) for value, records in sourced if records]
            if not sourced:
                result.append(condition)
                continue
            other = [value for value in condition.values if not by_value.get((owner, condition.name, value))]
            if other:
                result.append(condition.model_copy(update={'values': other}, deep=True))
            for index, (value, records) in enumerate(sourced, 1):
                identifier = condition.id if len(condition.values) == 1 else f'{condition.id}.source_{index}'
                attributed = condition.model_copy(update={'id': identifier, 'values': [value]}, deep=True)
                for record in records:
                    evidence = Evidence(kind=EvidenceKind.KNOWLEDGE if record.origin == 'knowledge_fact' else EvidenceKind.USER,
                        reference=record.source_quote or record.source_ref.get('source_url', ''), verified=False,
                        note=f'{record.origin}; source record {record.id}; continuity checked, meaning/image unverified')
                    if evidence not in attributed.evidence:
                        attributed.evidence.append(evidence)
                    if attributed.source_span is None and record.source_quote:
                        attributed.source_span = SourceSpan(text=record.source_quote)
                    mappings.append({'source_record_id': record.id, 'record_kind': 'condition',
                        'contract_ref': identifier, 'owner_ref': owner, 'field': condition.name, 'value': value})
                result.append(attributed)
        return result

    for entity in contract.entities:
        entity.constraints = annotate(entity.id, entity.constraints)
    contract.composition.constraints = annotate('composition', contract.composition.constraints)
    contract.style.constraints = annotate('style', contract.style.constraints)
    relations = []
    for relation in sources.relations:
        raw = relation.model_dump(mode='json')
        identifier = 'source_relation_' + _fingerprint({
            'original_text_sha256': sources.original_sha256, 'relation': raw})[:20]
        subject = character.id if relation.subject == 'character' else relation.subject
        obj = character.id if relation.object == 'character' else relation.object
        actual = next(item for item in contract.relations if
                      (item.subject_ref, item.predicate, item.object_ref) == (subject, relation.predicate, obj))
        evidence = Evidence(kind=EvidenceKind.USER, reference=relation.source_quote, verified=False,
            note='Exact source provenance and continuity only; relation interpretation unverified')
        if evidence not in actual.evidence:
            actual.evidence.append(evidence)
        if actual.source_span is None:
            actual.source_span = SourceSpan(text=relation.source_quote)
        relations.append({'relation_id': identifier, **raw})
        mappings.append({'source_record_id': identifier, 'record_kind': 'relation',
            'contract_ref': actual.id, 'subject_ref': actual.subject_ref,
            'predicate': actual.predicate, 'object_ref': actual.object_ref})
    for obj in sources.objects:
        mappings.append({'source_record_id': obj.id, 'record_kind': 'object', 'contract_ref': obj.id})
    active_ids = {item.id for item in sources.conditions if item.status == 'active'}
    mapped_ids = {item['source_record_id'] for item in mappings if item['record_kind'] == 'condition'}
    if active_ids != mapped_ids:
        raise _failure('current_contract_source_bindings')
    contract = SceneContractV3.model_validate(contract.model_dump(mode='json'))
    receipt = {'schema': 'source-continuity/v1', 'source_snapshot_sha256': source.snapshot_sha256,
        'scope': 'exact_current_interpretation_continuity_not_semantic_verification',
        'record_bindings': mappings, 'source_relations': relations,
        'semantic_correctness_verified': False, 'visual_accuracy_verified': False,
        'source_reinterpretation_supported': False}
    return contract, receipt


def preserve_source_metadata(rebuilt: dict, previous: dict, source: CurrentSource, receipt: dict) -> None:
    """Keep SOURCE current, but archive old ART bindings instead of pretending sync."""
    history = deepcopy(previous.get('source_art_history', []))
    art_keys = ('art_protocol', 'art_choices', 'relation_staging_bindings', 'whole_scene_binding', 'narrative_origin')
    prior_art = {key: deepcopy(previous[key]) for key in art_keys if key in previous}
    if prior_art:
        entry = {'author_id': previous.get('id'), 'author_revision': previous.get('revision'),
            'original_sha256': previous.get('original_sha256'),
            'scope': 'historical_art_not_current_realization', 'payload': prior_art,
            'payload_sha256': _fingerprint(prior_art)}
        if entry not in history:
            history.append(entry)
    rebuilt.update(source_conditions=deepcopy(source.snapshot),
        source_conditions_status='current_unverified_source_interpretation',
        source_relations=deepcopy(receipt['source_relations']),
        source_continuity=deepcopy(receipt), source_art_history=history)
