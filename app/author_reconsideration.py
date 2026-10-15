"""An author can question an interpretation, not silently rewrite its source.

The concern is unverified model data. A later, separately retained SOURCE
response and exact server-side preview are required before any replacement.
Neither a quote nor a complete response certifies understanding or artwork.
"""
from __future__ import annotations

from copy import deepcopy
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from .art_direction import fingerprint
from .authoring_contract import ART_SYSTEM_PROMPT, art_response_schema


class SourceInterpretationConcern(BaseModel):
    model_config = ConfigDict(extra='forbid')
    collection: Literal['conditions', 'objects', 'relations', 'goals', 'context']
    record_index: int = Field(ge=0, strict=True)
    source_quote: str = Field(min_length=1, max_length=4000)
    reason: str = Field(min_length=1, max_length=2500)


class AuthorSourceReconsideration(BaseModel):
    model_config = ConfigDict(extra='forbid')
    art_protocol: Literal['reconsider_source']
    source_snapshot_sha256: str = Field(pattern=r'^[0-9a-f]{64}$')
    reason: str = Field(min_length=1, max_length=2500)
    concerns: list[SourceInterpretationConcern] = Field(min_length=1, max_length=8)


RECONSIDER_ART_SYSTEM = ART_SYSTEM_PROMPT + """

Before making the picture, consider whether the supplied interpretation actually
preserves the original meaning. If a concrete SOURCE misunderstanding prevents
a coherent faithful picture, do not obey it blindly or rewrite it inside art.
Instead return ONLY art_protocol="reconsider_source", the supplied
source_snapshot_sha256, a reason, and one to eight concerns. Each concern names
the collection and zero-based record_index in current_source_snapshot, an exact
original source_quote and a specific reason to revisit that interpretation.
This is a request to investigate, not verified correction or permission to change
identity, selected knowledge, style or the original. Do not request source
changes merely to insert your preferred pose, lighting or other art choices.
If the interpretation is usable, return the complete normal art response.
"""

RECONSIDER_SOURCE_INSTRUCTION = """
The separately retained review_concern reports either an author's concern or
a program-detected conflict in the current interpretation. Neither proves that
the user expressed a conflict: an interpreter may have invented an attribute
from a character name. It is not a user instruction or a correct answer. Reconsider
the entire original in context, including semantic degree, why the scene matters
and the distinction between user content and artistic choices. Return a complete
fresh SOURCE extraction using the same schema, not a patch, record IDs, diffs,
metadata or automatically accepted claims. Frozen identity, knowledge and style
remain program-owned. Preserve unaffected meaning; do not create exclusions
from unknowns or force professional details onto the user.
Do not erase an explicit user choice merely to match selected knowledge. If the
original really requests an incompatible attribute, retain that request and the
resource disagreement. The program, not your explanation, controls adoption.
"""

EXPLAIN_REVISION_ART_SYSTEM = ART_SYSTEM_PROMPT + """

The SOURCE response was reconsidered because of the retained review concern.
The server assembled the proposed interpretation and calculated actual_changes;
the complete before and after are supplied, both semantically unverified.
Return change_annotations and art_choices. For EACH actual change, provide its
change_index exactly once, a verbatim source_quote from original_text and your
specific reason why that change is defensible. Do not invent or edit the diff,
server-generated IDs, hashes, source records or frozen knowledge. A quotation
is provenance, not proof that a value means the same thing. Explain derived
conflict or uncertainty changes as well as direct edits.
If art_is_permitted is false, art_choices MUST be null even when the proposed
snapshot itself has no conflicts. A removed conflict is not permission to erase
a user's content; the separately supplied conflict_transition_review explains
the program's limited adoption gate, not verified semantic understanding.
Otherwise make the COMPLETE normal art_choices from proposed_source_snapshot,
preserving its current goals and identity conditions. Do not revert to the old
interpretation or reuse a stale artistic plan. Artistic additions are proposals,
not facts or user requirements. For a blocking resource/source conflict,
art_choices must be null: do not fabricate an executable scene. For a
nonblocking proposed interpretation, give the complete art response; if you
still find it unusable, you may return a further reconsider_source concern
instead. That closes this bounded attempt, not another automatic retry.
"""


def reconsideration_art_input(data: dict, sources) -> dict:
    raw = sources.model_dump(mode='json')
    return deepcopy(data) | {'current_source_snapshot': raw,
                            'source_snapshot_sha256': fingerprint(raw),
                            'source_reconsideration_is_unverified': True}


def reconsideration_art_schema(sources) -> dict:
    concern = AuthorSourceReconsideration.model_json_schema()
    definitions = concern.pop('$defs', {})
    # Copy a supplied binding; the model never computes or chooses its hash.
    concern['properties']['source_snapshot_sha256']['enum'] = [fingerprint(sources.model_dump(mode='json'))]
    return {'type': 'object', '$defs': definitions, 'anyOf': [art_response_schema(sources), concern]}


def validate_author_concern(raw: dict, sources) -> AuthorSourceReconsideration:
    concern = AuthorSourceReconsideration.model_validate(raw)
    snapshot = sources.model_dump(mode='json')
    if concern.source_snapshot_sha256 != fingerprint(snapshot):
        raise ValueError('作者质疑对应的SOURCE版本已经变化，没有应用旧质疑')
    if not concern.reason.strip():
        raise ValueError('重看来源需要具体原因，不能是空白')
    seen = set()
    for item in concern.concerns:
        if not item.reason.strip() or not item.source_quote.strip():
            raise ValueError('来源质疑必须保留实际引用与具体原因')
        if item.source_quote not in sources.original_text:
            raise ValueError('来源质疑的引用不在完整原句中')
        records = snapshot[item.collection]
        if item.record_index >= len(records):
            raise ValueError('来源质疑引用了不存在的当前条目')
        binding = (item.collection, item.record_index)
        if binding in seen:
            raise ValueError('来源质疑重复引用同一条目，没有隐式去重')
        seen.add(binding)
        record = records[item.record_index]
        if item.collection == 'conditions' and record.get('origin') != 'user_quote':
            raise ValueError('作者不能以来源重解释改写冻结知识或专业覆盖')
    return concern


def revision_art_schema(sources, change_count: int, *, allow_art: bool = True) -> dict:
    from .source_reconsideration import SourceChangeAnnotation
    annotations = SourceChangeAnnotation.model_json_schema()
    annotations['properties']['change_index'].update(minimum=0, maximum=max(0, change_count - 1))
    art = {'type': 'null'} if sources.blocking or not allow_art else reconsideration_art_schema(sources)
    definitions = art.pop('$defs', {})
    return {'$defs': definitions, 'type': 'object', 'additionalProperties': False,
            'required': ['change_annotations', 'art_choices'], 'properties': {
                'change_annotations': {'type': 'array', 'minItems': change_count, 'maxItems': change_count,
                                       'items': annotations},
                'art_choices': art}}


def reported_conflict_concern(sources) -> dict:
    """Report the program's observations without calling them user requirements."""
    raw = sources.model_dump(mode='json')
    records = {item['id']: item for item in raw['conditions']}
    return {'kind': 'program_reported_conflict',
        'source_snapshot_sha256': fingerprint(raw),
        'reason': '核对冲突来自实际描述、资源选择，还是当前解释的无依据补充；不能先要求用户替解释器取舍。',
        'conflict_is_verified_user_requirement': False,
        'reported_conflicts': [{'conflict_index': index, 'conflict': deepcopy(conflict),
            'conditions': [deepcopy(records[key]) for key in conflict['condition_ids'] if key in records]}
            for index, conflict in enumerate(raw['conflicts'])]}


def conflict_transition_review(previous, proposed) -> dict:
    """Separate proposing a revised interpretation from releasing a conflict.

    No general semantic arbiter is wired here yet. Absence from a vocabulary,
    a different field, a complete diff, or a model's explanation cannot prove
    that user content was not discarded. Preserve the candidate for review;
    do not turn that current limitation into the project's final capability.
    """
    reviews = [{'conflict_index': index, 'conflict': conflict.model_dump(mode='json'),
        'automatic_retraction_permitted': False,
        'reason': '新解释可以提出不同理解，但模型理由和词表未命中不能独立核定用户内容；原阻塞不自动解除。'}
        for index, conflict in enumerate(previous.conflicts)]
    return {'scope': 'reported_conflict_reinterpretation_requires_confirmation', 'conflicts': reviews,
        'proposed_interpretation_reports_conflict': proposed.blocking,
        'automatic_adoption_permitted': not bool(previous.conflicts),
        'semantic_correctness_verified': False, 'identity_accuracy_verified': False}
