from __future__ import annotations

from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, Field, model_validator


class EntityKind(StrEnum):
    CHARACTER = "character"
    OBJECT = "object"
    ENVIRONMENT = "environment"
    EFFECT = "effect"


class ConstraintStrength(StrEnum):
    REQUIRED = "required"
    PREFERRED = "preferred"
    FORBIDDEN = "forbidden"


class EvidenceKind(StrEnum):
    USER = "user"
    KNOWLEDGE = "knowledge"
    INFERRED = "inferred"
    DEFAULT = "default"
    LEGACY = "legacy"


class ConstraintRole(StrEnum):
    REQUIREMENT = "requirement"
    SELECTED_DIRECTION = "selected_direction"
    PROPOSAL = "proposal"


class TextIntentMode(StrEnum):
    FORBIDDEN = "forbidden"
    EXACT = "exact"
    CANONICAL = "canonical"
    DECORATIVE_UI = "decorative_ui"


class ClauseStatus(StrEnum):
    MAPPED = "mapped"
    PARTIAL = "partial"
    AMBIGUOUS = "ambiguous"
    UNMAPPED = "unmapped"


class IssueKind(StrEnum):
    MISSING = "missing"
    AMBIGUOUS = "ambiguous"
    CONFLICT = "conflict"
    UNSUPPORTED = "unsupported"


class Evidence(BaseModel):
    kind: EvidenceKind
    reference: str = Field(default="", max_length=1000)
    confidence: float = Field(default=1.0, ge=0, le=1)
    verified: bool = False
    note: str = Field(default="", max_length=1000)


class SourceSpan(BaseModel):
    text: str = Field(min_length=1, max_length=4000)
    start: int | None = Field(default=None, ge=0)
    end: int | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def ordered_offsets(self) -> "SourceSpan":
        if self.start is not None and self.end is not None and self.end < self.start:
            raise ValueError("source span end must not precede start")
        return self


class ConstraintBinding(BaseModel):
    """A reviewed attribution of specific condition values, not an LLM claim.

    Source anchoring and exact value matching are checked by the builder. This
    records input provenance; it does not prove translation or image fidelity.
    """

    values: list[str] = Field(min_length=1, max_length=30)
    intent_role: ConstraintRole
    strength: ConstraintStrength | None = None
    evidence: list[Evidence] = Field(min_length=1, max_length=12)
    source_span: SourceSpan | None = None

    @model_validator(mode="after")
    def supported_attribution(self) -> "ConstraintBinding":
        if self.intent_role != ConstraintRole.PROPOSAL:
            kinds = {EvidenceKind.USER}
            if self.intent_role == ConstraintRole.REQUIREMENT:
                kinds.add(EvidenceKind.KNOWLEDGE)
            if not any(item.kind in kinds and item.verified and item.reference.strip()
                       for item in self.evidence):
                raise ValueError("a requirement or selected direction needs reviewed source evidence")
        return self


def _condition_defaults(value):
    """Do not reinterpret explicit strengths in saved contracts.

    New unreviewed conditions are movable proposals. Verified evidence can
    identify a requirement, while historical strength/preserve values retain
    their serialized meaning until an explicit plan rebuild.
    """
    if not isinstance(value, dict):
        return value
    result = dict(value)
    if result.get("evidence", []) is None:
        return result  # Let the field validator report invalid evidence.
    evidence = [item if isinstance(item, Evidence) else Evidence.model_validate(item)
                for item in result.get("evidence", [])]
    reviewed = any(item.kind in {EvidenceKind.USER, EvidenceKind.KNOWLEDGE}
                   and item.verified and item.reference.strip() for item in evidence)
    role = ConstraintRole(result.get("intent_role", ConstraintRole.REQUIREMENT if reviewed else ConstraintRole.PROPOSAL))
    allowed_kinds = {EvidenceKind.USER} if role == ConstraintRole.SELECTED_DIRECTION else {EvidenceKind.USER, EvidenceKind.KNOWLEDGE}
    if role != ConstraintRole.PROPOSAL and not any(
        item.kind in allowed_kinds and item.verified and item.reference.strip() for item in evidence
    ):
        raise ValueError("a requirement or selected direction needs reviewed source evidence")
    result.setdefault("intent_role", role)
    result.setdefault("strength", ConstraintStrength.REQUIRED if role == ConstraintRole.REQUIREMENT else ConstraintStrength.PREFERRED)
    result.setdefault("preserve_across_stages", role != ConstraintRole.PROPOSAL)
    return result


class VisualConstraint(BaseModel):
    id: str = Field(pattern=r"^[a-zA-Z][a-zA-Z0-9_.-]*$", max_length=120)
    name: str = Field(min_length=1, max_length=160)
    values: list[str] = Field(min_length=1, max_length=30)
    intent_role: ConstraintRole = ConstraintRole.PROPOSAL
    strength: ConstraintStrength = ConstraintStrength.PREFERRED
    preserve_across_stages: bool = False
    evidence: list[Evidence] = Field(default_factory=list, max_length=12)
    source_span: SourceSpan | None = None

    @model_validator(mode="before")
    @classmethod
    def attributed_defaults(cls, value):
        return _condition_defaults(value)


class SceneEntity(BaseModel):
    id: str = Field(pattern=r"^[a-zA-Z][a-zA-Z0-9_-]*$", max_length=80)
    kind: EntityKind
    name: str = Field(min_length=1, max_length=200)
    role: str = Field(default="", max_length=120)
    model_tags: dict[str, list[str]] = Field(default_factory=dict)
    knowledge_ref: str = Field(default="", max_length=500)
    constraints: list[VisualConstraint] = Field(default_factory=list, max_length=80)
    region_hint: list[str] = Field(default_factory=list, max_length=12)

    @model_validator(mode="after")
    def unique_constraint_ids(self) -> "SceneEntity":
        ids = [item.id for item in self.constraints]
        if len(ids) != len(set(ids)):
            raise ValueError(f"duplicate constraint id in entity {self.id}")
        return self


class SceneRelation(BaseModel):
    id: str = Field(pattern=r"^[a-zA-Z][a-zA-Z0-9_.-]*$", max_length=120)
    subject_ref: str = Field(min_length=1, max_length=160)
    predicate: str = Field(min_length=1, max_length=160)
    object_ref: str = Field(min_length=1, max_length=160)
    intent_role: ConstraintRole = ConstraintRole.PROPOSAL
    strength: ConstraintStrength = ConstraintStrength.PREFERRED
    preserve_across_stages: bool = False
    evidence: list[Evidence] = Field(default_factory=list, max_length=12)
    source_span: SourceSpan | None = None

    @model_validator(mode="before")
    @classmethod
    def attributed_defaults(cls, value):
        return _condition_defaults(value)


class CompositionIntent(BaseModel):
    framing: list[str] = Field(default_factory=list, max_length=20)
    camera: list[str] = Field(default_factory=list, max_length=20)
    layout: list[str] = Field(default_factory=list, max_length=30)
    depth: list[str] = Field(default_factory=list, max_length=20)
    preserve_across_stages: bool = False
    constraints: list[VisualConstraint] = Field(default_factory=list, max_length=80)

    @model_validator(mode="after")
    def conditions_reach_composition(self) -> "CompositionIntent":
        for item in self.constraints:
            if item.name not in {"framing", "camera", "layout", "depth"} or any(
                value not in getattr(self, item.name, []) for value in item.values
            ):
                raise ValueError("composition attribution does not match actual composition values")
        if len({item.id for item in self.constraints}) != len(self.constraints):
            raise ValueError("duplicate composition constraint id")
        return self


class StyleIntent(BaseModel):
    description: list[str] = Field(default_factory=list, max_length=30)
    preset_id: str = Field(default="", max_length=120)
    references: list[str] = Field(default_factory=list, max_length=20)
    exclusions: list[str] = Field(default_factory=list, max_length=30)
    constraints: list[VisualConstraint] = Field(default_factory=list, max_length=80)

    @model_validator(mode="after")
    def conditions_reach_style(self) -> "StyleIntent":
        for item in self.constraints:
            if item.name not in {"description", "exclusions"} or any(
                value not in getattr(self, item.name, []) for value in item.values
            ):
                raise ValueError("style attribution does not match actual style values")
        if len({item.id for item in self.constraints}) != len(self.constraints):
            raise ValueError("duplicate style constraint id")
        return self


class TextIntent(BaseModel):
    id: str = Field(pattern=r"^[a-zA-Z][a-zA-Z0-9_.-]*$", max_length=120)
    mode: TextIntentMode
    content: str = Field(default="", max_length=2000)
    language: str = Field(default="", max_length=80)
    target_ref: str = Field(default="scene", max_length=160)
    placement: list[str] = Field(default_factory=list, max_length=12)
    appearance: list[str] = Field(default_factory=list, max_length=12)
    exact_content_required: bool = False
    protect_from_cleanup: bool = False
    evidence: list[Evidence] = Field(default_factory=list, max_length=12)
    source_span: SourceSpan | None = None

    @model_validator(mode="after")
    def content_matches_mode(self) -> "TextIntent":
        if self.mode == TextIntentMode.FORBIDDEN and self.content:
            raise ValueError("forbidden text intent cannot contain requested text")
        if self.mode in {TextIntentMode.EXACT, TextIntentMode.CANONICAL} and not self.content.strip():
            raise ValueError(f"{self.mode.value} text intent requires content")
        if self.mode == TextIntentMode.EXACT:
            self.exact_content_required = True
            self.protect_from_cleanup = True
        if self.mode == TextIntentMode.CANONICAL:
            self.protect_from_cleanup = True
        return self


class ContractIssue(BaseModel):
    id: str = Field(pattern=r"^[a-zA-Z][a-zA-Z0-9_.-]*$", max_length=120)
    kind: IssueKind
    description: str = Field(min_length=1, max_length=2000)
    blocking: bool = True
    source_span: SourceSpan | None = None
    resolution_hint: str = Field(default="", max_length=1000)


class SourceClause(BaseModel):
    id: str = Field(pattern=r"^[a-zA-Z][a-zA-Z0-9_.-]*$", max_length=120)
    span: SourceSpan
    status: ClauseStatus
    blocking: bool = False
    mapped_refs: list[str] = Field(default_factory=list, max_length=40)
    note: str = Field(default="", max_length=1000)


class SceneContractV3(BaseModel):
    version: Literal["3.0"] = "3.0"
    source_text: str = Field(min_length=1, max_length=12_000)
    entities: list[SceneEntity] = Field(min_length=1, max_length=40)
    relations: list[SceneRelation] = Field(default_factory=list, max_length=120)
    composition: CompositionIntent = Field(default_factory=CompositionIntent)
    style: StyleIntent = Field(default_factory=StyleIntent)
    text_intents: list[TextIntent] = Field(default_factory=list, max_length=30)
    global_exclusions: list[str] = Field(default_factory=list, max_length=80)
    clauses: list[SourceClause] = Field(default_factory=list, max_length=100)
    assumptions: list[str] = Field(default_factory=list, max_length=50)
    issues: list[ContractIssue] = Field(default_factory=list, max_length=50)

    @staticmethod
    def _root_ref(reference: str) -> str:
        return reference.split(".", 1)[0]

    @model_validator(mode="after")
    def validate_graph(self) -> "SceneContractV3":
        entity_ids = [item.id for item in self.entities]
        if len(entity_ids) != len(set(entity_ids)):
            raise ValueError("scene entity ids must be unique")
        relation_ids = [item.id for item in self.relations]
        if len(relation_ids) != len(set(relation_ids)):
            raise ValueError("scene relation ids must be unique")
        text_ids = [item.id for item in self.text_intents]
        if len(text_ids) != len(set(text_ids)):
            raise ValueError("text intent ids must be unique")

        known = set(entity_ids)
        for relation in self.relations:
            for reference in (relation.subject_ref, relation.object_ref):
                if self._root_ref(reference) not in known:
                    raise ValueError(f"relation {relation.id} references unknown entity: {reference}")
        for intent in self.text_intents:
            if intent.target_ref != "scene" and self._root_ref(intent.target_ref) not in known:
                raise ValueError(f"text intent {intent.id} references unknown target: {intent.target_ref}")
        return self

    @property
    def blocking_issues(self) -> list[ContractIssue]:
        return [item for item in self.issues if item.blocking]

    @property
    def generation_ready(self) -> bool:
        return not self.blocking_issues and not any(
            clause.blocking and clause.status in {ClauseStatus.AMBIGUOUS, ClauseStatus.UNMAPPED}
            for clause in self.clauses
        )

    def require_generation_ready(self) -> None:
        if not self.generation_ready:
            issue_ids = [item.id for item in self.blocking_issues]
            issue_ids.extend(item.id for item in self.clauses if item.blocking and item.status in {ClauseStatus.AMBIGUOUS, ClauseStatus.UNMAPPED})
            raise ValueError("scene contract is not generation-ready: " + ", ".join(issue_ids))
