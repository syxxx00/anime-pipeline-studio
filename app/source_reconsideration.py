"""Pure preview and explanation binding for a corrigible SOURCE interpretation.

The server, not the proposing model, computes complete records and their IDs.
This boundary can be used before a GenerateRequest/CreationIntent exists. It
checks integrity and exact provenance only, never the meaning of a proposal.
Nothing here activates a revision, compiles ART, changes a run or dispatches.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Sequence

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .authoring_contract import SourceExtraction, parse_source_response, source_relations_with_ids
from .schemas import NewbieCompileRequest
from .source_conditions import SourceConditions
from .source_continuity import current_source
from .source_revision import (SourceRevisionChange, SourceRevisionResult,
                              _candidate_records, _canonical, _check_changes,
                              _sha, actual_source_changes)


_FROZEN_FIELDS = ('original_text', 'original_sha256', 'style_mode',
                  'identity_binding', 'knowledge')


@dataclass(frozen=True)
class SourceReconsiderationPreview:
    previous_snapshot: dict
    sources: SourceConditions
    actual_changes: list[dict]
    receipt: dict


class SourceChangeAnnotation(BaseModel):
    """Model-supplied explanation of a server-computed change, without IDs."""
    model_config = ConfigDict(extra='forbid')
    change_index: int = Field(ge=0, strict=True)
    source_quote: str = Field(min_length=1, max_length=4000)
    reason: str = Field(min_length=1, max_length=4000)

    @model_validator(mode='after')
    def explicit_explanation(self):
        if not self.source_quote.strip() or not self.reason.strip():
            raise ValueError('每项预览注解需要非空原句引用和理由，服务端不代写')
        return self


def _checked_current(snapshot: SourceConditions | dict):
    raw = deepcopy(snapshot.model_dump(mode='json')
                   if isinstance(snapshot, SourceConditions) else snapshot)
    if not isinstance(raw, dict):
        raise TypeError('SOURCE预览需要完整SourceConditions或CurrentSource.snapshot')
    # Validate finite JSON before Pydantic can coerce or ignore any representation.
    _canonical(raw)
    sources = SourceConditions.model_validate(raw)
    checked = current_source({'source_conditions': raw}, sources.original_text,
                             sources.original_sha256)
    if checked is None:
        raise ValueError('SOURCE预览缺少完整当前来源解释')
    if any(item.origin == 'professional_edit' or item.status == 'superseded' or
           item.superseded_by is not None for item in sources.conditions):
        raise ValueError('当前SOURCE含专业覆盖；模型重看入口不能撤销或重建专业覆盖')
    relation_ids = [item['relation_id'] for item in source_relations_with_ids(sources)]
    if len(set(relation_ids)) != len(relation_ids):
        raise ValueError('当前SOURCE关系ID重复或碰撞，不能准确绑定重看')
    return checked


def preview_source_reconsideration(current_sources: SourceConditions | dict,
                                  extraction: SourceExtraction | dict, *,
                                  expected_source_sha256: str) -> SourceReconsiderationPreview:
    """Assemble the full proposed SOURCE and display its exact unexplained diff.

    Pass CurrentSource.snapshot, not just selected source fields. The expected
    SHA binds the exact raw snapshot, including any legacy omitted defaults.
    Professional overrides are explicitly unsupported, not silently bypassed.
    A no-change proposal remains a preview, not a fabricated SOURCE revision.
    """
    current = _checked_current(current_sources)
    if not isinstance(expected_source_sha256, str) or expected_source_sha256 != current.snapshot_sha256:
        raise ValueError('当前SOURCE快照SHA不匹配，不能重看另一个或已经改变的解释')
    old = current.sources
    raw_extraction = deepcopy(extraction.model_dump(mode='json', by_alias=True)
                              if isinstance(extraction, SourceExtraction) else extraction)
    _canonical(raw_extraction)
    revised = SourceExtraction.model_validate(raw_extraction)
    candidate_records = _candidate_records(revised)
    knowledge = old.knowledge
    frozen = {'identity_binding': old.identity_binding.model_dump(mode='json'),
              'knowledge': knowledge.model_dump(mode='json')}
    # This is an actual SOURCE assembly request, not a fake image request or
    # invented CreationIntent. Its inputs come only from the frozen SOURCE.
    assembly_request = NewbieCompileRequest(sentence=old.original_text, style_mode=old.style_mode,
        knowledge_pack=knowledge.pack if knowledge.selected else '',
        knowledge_operator=knowledge.operator if knowledge.selected else '',
        knowledge_appearance=knowledge.appearance if knowledge.selected else '')
    sources = parse_source_response(revised.model_dump(mode='json', by_alias=True),
                                    assembly_request, deepcopy(frozen))
    if knowledge.evidence_check_scope == 'legacy_source_presence_only':
        expected_knowledge_count = (len(set(knowledge.audited_visual_facts)) +
                                    len(set(knowledge.audited_clothing_facts)))
    else:
        expected_knowledge_count = len(knowledge.visual_fact_records) + len(knowledge.clothing_fact_records)
    if (sum(item.origin == 'user_quote' for item in sources.conditions) != len(candidate_records) or
            sum(item.origin == 'knowledge_fact' for item in sources.conditions) != expected_knowledge_count):
        raise ValueError('SOURCE组装丢失条件或发生ID碰撞，未把不完整输出当作完整预览')
    condition_ids = [item.id for item in sources.conditions]
    relation_ids = [item['relation_id'] for item in source_relations_with_ids(sources)]
    if len(set(condition_ids)) != len(condition_ids) or len(set(relation_ids)) != len(relation_ids):
        raise ValueError('新SOURCE条件或关系ID碰撞，不能准确绑定重看')
    before, after = old.model_dump(mode='json'), sources.model_dump(mode='json')
    for key in _FROZEN_FIELDS:
        if _canonical(before[key]) != _canonical(after[key]):
            raise ValueError(f'SOURCE重看越权改写冻结的{key}')
    checked = current_source({'source_conditions': after}, old.original_text, old.original_sha256)
    if checked is None:
        raise ValueError('新SOURCE未形成同一原句的完整解释')
    changes = actual_source_changes(before, after)
    receipt = {
        'schema': 'source-reconsideration-preview/v1',
        'original_text': old.original_text,
        'original_text_sha256': old.original_sha256,
        'previous_source_sha256': current.snapshot_sha256,
        'revised_source_sha256': checked.snapshot_sha256,
        'proposal_scope_sha256': current.snapshot_sha256,
        'complete_extraction': raw_extraction,
        'actual_changes': deepcopy(changes),
        'outcome': 'changes_require_explanation' if changes else 'no_change',
        'no_change': not bool(changes),
        'frozen_bindings': {key: deepcopy(before[key]) for key in _FROZEN_FIELDS},
        'knowledge_payload_unchanged': True,
        'knowledge_condition_state_recomputed': True,
        'provenance_check_scope': 'exact_original_quotes_and_frozen_bindings_only',
        'semantic_correctness_verified': False,
        'identity_accuracy_verified': False,
        'visual_accuracy_verified': False,
        'state_mutated': False,
        'new_interpretation_effective': False,
        'dispatch_performed': False,
    }
    # Bind the raw old representation, complete new representation, original,
    # raw extraction, unexplained differences and all unverified declarations.
    receipt['preview_sha256'] = _sha({'previous_snapshot': current.snapshot,
        'revised_snapshot': after, 'receipt': receipt})
    return SourceReconsiderationPreview(deepcopy(current.snapshot), sources.model_copy(deep=True),
                                        deepcopy(changes), deepcopy(receipt))


def _validate_preview(preview: SourceReconsiderationPreview, *, expected_preview_sha256: str,
                      expected_source_sha256: str, expected_revised_source_sha256: str):
    if not isinstance(preview, SourceReconsiderationPreview):
        raise TypeError('SOURCE注解必须绑定完整服务端预览，不接收局部diff')
    rebuilt = preview_source_reconsideration(preview.previous_snapshot,
        preview.receipt.get('complete_extraction'), expected_source_sha256=expected_source_sha256)
    if (_canonical(preview.sources.model_dump(mode='json')) != _canonical(rebuilt.sources.model_dump(mode='json')) or
            _canonical(preview.actual_changes) != _canonical(rebuilt.actual_changes) or
            _canonical(preview.receipt) != _canonical(rebuilt.receipt)):
        raise ValueError('完整SOURCE预览已被改变；不能把旧注解绑定到不同的before/after或原件')
    if expected_preview_sha256 != rebuilt.receipt['preview_sha256']:
        raise ValueError('SOURCE预览SHA不匹配，注解不能应用于其他提案')
    if expected_revised_source_sha256 != rebuilt.receipt['revised_source_sha256']:
        raise ValueError('新SOURCE快照SHA不匹配，不能错配修订后的完整解释')
    return rebuilt


def bind_source_reconsideration_annotations(preview: SourceReconsiderationPreview,
        annotations: Sequence[SourceChangeAnnotation | dict], *, expected_preview_sha256: str,
        expected_source_sha256: str, expected_revised_source_sha256: str) -> list[SourceRevisionChange]:
    """Expand indexed annotations into full changes, with exact total coverage.

    Rebuild the preview as well as checking all three hashes: a frozen dataclass
    does not make its nested dicts or Pydantic objects immutable. No default
    quote/reason is supplied. This establishes explanation coverage only.
    """
    checked = _validate_preview(preview, expected_preview_sha256=expected_preview_sha256,
        expected_source_sha256=expected_source_sha256,
        expected_revised_source_sha256=expected_revised_source_sha256)
    # Python mode retains an illegally mutated bool index; JSON serialization
    # of an int field could otherwise silently turn False into 0 before strict
    # validation. Annotation fields themselves are already JSON primitives.
    explained = [SourceChangeAnnotation.model_validate(deepcopy(
        item.model_dump(mode='python') if isinstance(item, SourceChangeAnnotation) else item))
        for item in annotations]
    indices = [item.change_index for item in explained]
    if len(indices) != len(set(indices)) or set(indices) != set(range(len(checked.actual_changes))):
        raise ValueError('change_index必须准确、无重复地覆盖全部真实diff，不能漏掉派生冲突或删除项')
    by_index = {item.change_index: item for item in explained}
    changes = [SourceRevisionChange.model_validate({**deepcopy(change),
        'source_quote': by_index[index].source_quote, 'reason': by_index[index].reason})
        for index, change in enumerate(checked.actual_changes)]
    if changes:
        _check_changes(current_source({'source_conditions': checked.previous_snapshot},
            checked.sources.original_text, checked.sources.original_sha256).sources.model_dump(mode='json'),
            checked.sources.model_dump(mode='json'), changes, checked.sources.original_text)
    return changes


def confirm_source_reconsideration(preview: SourceReconsiderationPreview,
        annotations: Sequence[SourceChangeAnnotation | dict], *, expected_preview_sha256: str,
        expected_source_sha256: str, expected_revised_source_sha256: str) -> SourceRevisionResult:
    """Confirm complete provenance explanations without inventing run state.

    Initial preparation may use this result to construct a fresh complete ART.
    Existing runs must still apply the changes through reinterpret_source and
    their paused/atomic control boundary against the actual current request.
    No-change confirms only that the proposal changes nothing; it is not a
    revision, activation or semantic approval, and does not require rebuilding.
    """
    changes = bind_source_reconsideration_annotations(preview, annotations,
        expected_preview_sha256=expected_preview_sha256,
        expected_source_sha256=expected_source_sha256,
        expected_revised_source_sha256=expected_revised_source_sha256)
    # The preceding binding already rebuilt and checked the complete preview;
    # return separate copies so callers cannot mutate the original audit input.
    receipt = deepcopy(preview.receipt)
    receipt['schema'] = 'source-reconsideration-confirmation/v1'
    receipt['actual_changes'] = [item.model_dump(mode='json') for item in changes]
    receipt['outcome'] = 'explained_unverified_reconsideration' if changes else 'no_change'
    receipt['explanation_coverage_checked'] = True
    receipt['art_reconstruction_required'] = bool(changes)
    receipt['existing_run_control_required'] = True
    return SourceRevisionResult(preview.sources.model_copy(deep=True), receipt,
                                deepcopy(preview.previous_snapshot))
