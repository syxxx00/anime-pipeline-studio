from __future__ import annotations

import re

from .scene_contract import (
    ClauseStatus,
    CompositionIntent,
    ConstraintBinding,
    ConstraintRole,
    ConstraintStrength,
    EntityKind,
    Evidence,
    EvidenceKind,
    SceneContractV3,
    SceneEntity,
    SceneRelation,
    SourceClause,
    SourceSpan,
    StyleIntent,
    TextIntent,
    TextIntentMode,
    VisualConstraint,
)
from .schemas import NewbieSemanticSpec


def _safe_id(value: str, fallback: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9_-]+", "_", value).strip("_")
    if not cleaned or not cleaned[0].isalpha():
        cleaned = fallback
    return cleaned[:80]


def _constraint(entity_id: str, name: str, values: list[str], evidence: Evidence) -> VisualConstraint | None:
    if not values:
        return None
    return VisualConstraint(
        id=f"{entity_id}.{name}",
        name=name,
        values=values,
        evidence=[evidence],
        intent_role=ConstraintRole.PROPOSAL,
    )


def _binding_span(binding: ConstraintBinding, source: str) -> None:
    span = binding.source_span
    if span is not None:
        if span.text not in source:
            raise ValueError("constraint binding source span is not in the original request")
        start = span.start if span.start is not None else (span.end - len(span.text) if span.end is not None else None)
        end = span.end if span.end is not None else (span.start + len(span.text) if span.start is not None else None)
        if start is not None and source[start:end] != span.text:
            raise ValueError("constraint binding source offsets do not match its quote")
    if binding.intent_role == ConstraintRole.REQUIREMENT and any(
        item.kind == EvidenceKind.USER and item.reference == "source_sentence" and item.verified
        for item in binding.evidence
    ) and span is None:
        raise ValueError("a source-sentence requirement needs its actual source span")


def _bound_constraints(
    entity_id: str, name: str, values: list[str], evidence: Evidence,
    bindings: dict[str, list[ConstraintBinding]], source: str,
) -> list[VisualConstraint]:
    base = f"{entity_id}.{name}"
    attributed = bindings.get(base, [])
    if not values and not attributed:
        return []
    claimed: set[str] = set()
    constraints = []
    for index, binding in enumerate(attributed, 1):
        _binding_span(binding, source)
        if binding.strength == ConstraintStrength.FORBIDDEN and (entity_id == "composition" or (entity_id == "style" and name == "description")):
            raise ValueError(f"binding {base} cannot forbid a value still encoded in a positive composition/style slot")
        if any(value not in values for value in binding.values):
            raise ValueError(f"binding {base} names a value absent from the current conditions")
        if len(set(binding.values)) != len(binding.values) or claimed.intersection(binding.values):
            raise ValueError(f"binding {base} attributes the same value more than once")
        claimed.update(binding.values)
        constraints.append(VisualConstraint(
            id=f"{base}.{binding.intent_role.value}_{index}", name=name,
            values=binding.values, intent_role=binding.intent_role,
            evidence=binding.evidence, source_span=binding.source_span,
            **({"strength": binding.strength} if binding.strength is not None else {}),
        ))
    remaining = [value for value in values if value not in claimed]
    if remaining:
        constraint = _constraint(entity_id, name, remaining, evidence)
        if constraints:
            constraint.id = f"{base}.proposal_{len(constraints) + 1}"
        constraints.append(constraint)
    return constraints


def _relation_ref(value: str, character_id: str, objects: list[SceneEntity], *, subject: bool) -> str:
    normalized = value.casefold().strip().replace(" ", "_")
    if subject:
        if normalized == character_id.casefold() or normalized.startswith(character_id.casefold() + '_('):
            return character_id
        return f"{character_id}.{normalized}"
    for entity in objects:
        candidates = (entity.id.casefold(), entity.name.casefold().replace(" ", "_"))
        for candidate in candidates:
            if normalized.startswith(candidate):
                suffix = normalized[len(candidate):].strip("_")
                return entity.id + (f".{suffix}" if suffix else "")
    return objects[0].id if len(objects) == 1 else normalized


def _text_intents(sentence: str, evidence: Evidence) -> list[TextIntent]:
    lower = sentence.casefold()
    exact_matches = re.findall(r"[“\"]([^”\"]{1,80})[”\"]", sentence)
    if any(token in lower for token in ("写着", "显示文字", "文字为", "text reads")) and exact_matches:
        return [TextIntent(
            id="requested_text",
            mode=TextIntentMode.EXACT,
            content=exact_matches[-1],
            target_ref="scene",
            evidence=[evidence],
            source_span=SourceSpan(text=sentence),
        )]
    if any(token in lower for token in ("抽象界面", "界面色块", "abstract interface", "ui blocks")):
        return [TextIntent(
            id="decorative_ui",
            mode=TextIntentMode.DECORATIVE_UI,
            target_ref="scene",
            evidence=[evidence],
            source_span=SourceSpan(text=sentence),
        )]
    if any(token in lower for token in ("不出现文字", "没有可读文字", "无文字", "no text", "without text", "no readable text")):
        return [TextIntent(
            id="no_text",
            mode=TextIntentMode.FORBIDDEN,
            target_ref="scene",
            evidence=[evidence],
            source_span=SourceSpan(text=sentence),
        )]
    return []


def _canonical_clothing_text(spec: NewbieSemanticSpec, evidence: Evidence, character_id: str) -> list[TextIntent]:
    clothing = " ".join(spec.clothing).lower()
    known = (("rhodes island lettering", "RHODES ISLAND"),)
    return [TextIntent(
        id="canonical_clothing_text", mode=TextIntentMode.CANONICAL,
        content=content, target_ref=f"{character_id}.clothing",
        placement=["on the canonical garment panel"],
        appearance=["match the official costume marking"], evidence=[evidence],
    ) for token, content in known if token in clothing]


def contract_from_semantic(
    spec: NewbieSemanticSpec,
    *,
    compiler: str = "",
    clause_map: list[dict] | None = None,
    constraint_bindings: dict[str, list[ConstraintBinding]] | None = None,
    strict_entity_refs: bool = False,
) -> SceneContractV3:
    """Build current conditions without certifying extractor self-reports.

    Bindings are reviewed, value-level attributions. ``clause_map`` remains a
    limited coverage diagnostic, not evidence upgrading an entire mixed field.
    Preferred proposals still reach every native compiler.
    """
    bindings = {key: [ConstraintBinding.model_validate(item) for item in items]
                for key, items in (constraint_bindings or {}).items()}
    used_binding_refs: set[str] = set()

    def conditions(entity_id: str, name: str, values: list[str], evidence: Evidence) -> list[VisualConstraint]:
        used_binding_refs.add(f"{entity_id}.{name}")
        return _bound_constraints(entity_id, name, values, evidence, bindings, spec.source_sentence)

    user_evidence = Evidence(kind=EvidenceKind.USER, reference="source_sentence", confidence=1.0, verified=True)
    interpreted_evidence = Evidence(kind=EvidenceKind.INFERRED, reference=compiler or 'semantic_spec',
        confidence=0.5, verified=False,
        note='Translated or supplemented semantic content; not an independently verified user statement or canonical character fact.')
    knowledge_evidence = Evidence(
        kind=EvidenceKind.KNOWLEDGE,
        reference=spec.source or "knowledge_source_missing",
        confidence=0.9,
        verified=False,
        note="Knowledge-selected semantic fields; a URL alone does not verify every value. Individual fact evidence must be inspected in the knowledge pack.",
    )
    has_knowledge = "+knowledge" in compiler
    character_id = _safe_id(spec.character_tag.split("_(", 1)[0], "character_1")
    character_constraints = [item for group in (
        conditions(character_id, "gender", [spec.count] if spec.count else [], interpreted_evidence),
        conditions(character_id, "appearance", spec.appearance, knowledge_evidence if has_knowledge else interpreted_evidence),
        conditions(character_id, "clothing", spec.clothing, knowledge_evidence if has_knowledge else interpreted_evidence),
        conditions(character_id, "expression", spec.expression, interpreted_evidence),
        conditions(character_id, "body_action", spec.body_action, interpreted_evidence),
        conditions(character_id, "left_hand", spec.left_hand, interpreted_evidence),
        conditions(character_id, "right_hand", spec.right_hand, interpreted_evidence),
    ) for item in group]
    entities: list[SceneEntity] = [SceneEntity(
        id=character_id,
        kind=EntityKind.CHARACTER,
        name=spec.character_name,
        role="primary_character",
        model_tags={target: [spec.character_tag] for target in ("newbie", "illustrious", "anima", "z_image") if spec.character_tag},
        knowledge_ref=spec.source,
        constraints=character_constraints,
    )]
    object_entities: list[SceneEntity] = []
    for index, obj in enumerate(spec.objects, 1):
        object_id = _safe_id(obj.id, f"object_{index}")
        constraints = [item for group in (
            conditions(object_id, "appearance", obj.appearance, interpreted_evidence),
            conditions(object_id, "position", obj.position, interpreted_evidence),
        ) for item in group]
        object_entities.append(SceneEntity(id=object_id, kind=EntityKind.OBJECT, name=obj.name, constraints=constraints))
    entities.extend(object_entities)
    if spec.background or spec.lighting:
        environment_constraints = [item for group in (
            conditions("environment", "background", spec.background, interpreted_evidence),
            conditions("environment", "lighting", spec.lighting, interpreted_evidence),
        ) for item in group]
        entities.append(SceneEntity(id="environment", kind=EntityKind.ENVIRONMENT, name="scene environment", constraints=environment_constraints))

    relations = []
    def entity_reference(value: str, *, subject: bool) -> str:
        if not strict_entity_refs:
            return _relation_ref(value, character_id, object_entities, subject=subject)
        # The new authoring path declares whole-entity IDs explicitly. Never
        # reinterpret an object subject as a character body-part/field, nor
        # substitute the sole object for an unknown relation endpoint.
        refs = {'character': character_id, character_id: character_id,
                **{obj.id: obj.id for obj in object_entities}}
        if value not in refs:
            raise ValueError('strict relation reference is not a declared entity: ' + value)
        return refs[value]
    for index, item in enumerate(spec.relations, 1):
        relation_id = f"relation_{index}"
        used_binding_refs.add(relation_id)
        relation = SceneRelation(
            id=relation_id,
            subject_ref=entity_reference(item.subject, subject=True),
            predicate=item.predicate,
            object_ref=entity_reference(item.object, subject=False),
            intent_role=ConstraintRole.PROPOSAL,
            evidence=[interpreted_evidence],
        )
        entries = bindings.get(relation_id, [])
        if entries:
            signature = f"{relation.subject_ref} {relation.predicate} {relation.object_ref}"
            if len(entries) != 1 or entries[0].values != [signature]:
                raise ValueError(f"binding {relation_id} does not match the complete current relation")
            binding = entries[0]
            _binding_span(binding, spec.source_sentence)
            relation = SceneRelation(
                **{key: value for key, value in relation.model_dump().items()
                   if key not in {"intent_role", "strength", "preserve_across_stages", "evidence", "source_span"}},
                intent_role=binding.intent_role, evidence=binding.evidence, source_span=binding.source_span,
                **({"strength": binding.strength} if binding.strength is not None else {}),
            )
        relations.append(relation)

    composition_constraints = conditions("composition", "framing", spec.framing, interpreted_evidence)
    style_constraints = conditions("style", "description", spec.style, interpreted_evidence)
    style_constraints += conditions("style", "exclusions", spec.exclusions, interpreted_evidence)
    unknown_refs = set(bindings) - used_binding_refs
    if unknown_refs:
        raise ValueError("unknown constraint binding refs: " + ", ".join(sorted(unknown_refs)))

    clauses: list[SourceClause] = []
    # Do not persist the extractor's unverified self-report as a proven mapping.
    # Import here to avoid a module cycle with response_for().
    from .newbie_compiler import analyze_coverage
    audited_clauses = analyze_coverage(spec, clause_map)["clauses"] if clause_map is not None else []
    for index, item in enumerate(audited_clauses, 1):
        status_value = str(item.get("status", "partial"))
        status = ClauseStatus(status_value if status_value in ClauseStatus._value2member_map_ else "partial")
        clauses.append(SourceClause(
            id=f"clause_{index}",
            span=SourceSpan(text=str(item.get("text") or spec.source_sentence)),
            status=status,
            blocking=False,
            mapped_refs=[str(value) for value in item.get("fields", [])],
            note=str(item.get("reason", "")),
        ))
    if not clauses:
        clauses.append(SourceClause(
            id="clause_1",
            span=SourceSpan(text=spec.source_sentence),
            status=ClauseStatus.PARTIAL,
            mapped_refs=[item.id for item in entities] + [item.id for item in relations],
            note="Migrated semantic specification; no per-clause evidence proves that every source requirement was preserved.",
        ))
    return SceneContractV3(
        source_text=spec.source_sentence,
        entities=entities,
        relations=relations,
        composition=CompositionIntent(framing=spec.framing, constraints=composition_constraints,
            preserve_across_stages=bool(composition_constraints) and all(item.preserve_across_stages for item in composition_constraints)),
        style=StyleIntent(description=spec.style, exclusions=spec.exclusions, constraints=style_constraints),
        text_intents=_text_intents(spec.source_sentence, user_evidence) + _canonical_clothing_text(spec, knowledge_evidence if has_knowledge else interpreted_evidence, character_id),
        clauses=clauses,
    )


def _condition_groups(contract: SceneContractV3) -> dict[str, list[VisualConstraint]]:
    groups: dict[str, list[VisualConstraint]] = {}
    for entity in contract.entities:
        for constraint in entity.constraints:
            groups.setdefault(f"{entity.id}.{constraint.name}", []).append(constraint)
    for owner, constraints in (("composition", contract.composition.constraints), ("style", contract.style.constraints)):
        for constraint in constraints:
            groups.setdefault(f"{owner}.{constraint.name}", []).append(constraint)
    return groups


def _protected(condition: VisualConstraint | SceneRelation) -> bool:
    if condition.intent_role == ConstraintRole.PROPOSAL:
        return False
    kinds = {EvidenceKind.USER}
    if condition.intent_role == ConstraintRole.REQUIREMENT:
        kinds.add(EvidenceKind.KNOWLEDGE)
    return any(item.kind in kinds and item.verified and item.reference.strip() for item in condition.evidence)


def rebuild_current_contract(
    previous: SceneContractV3,
    current_spec: NewbieSemanticSpec,
    *,
    compiler: str = "",
    clause_map: list[dict] | None = None,
    constraint_bindings: dict[str, list[ConstraintBinding]] | None = None,
    replaced_refs: set[str] | None = None,
    strict_entity_refs: bool = False,
) -> SceneContractV3:
    """Replace unverified proposals while checking reviewed commitments.

    Missing/conflicting commitments stop rebuilding; old actions are never
    appended to a contradictory new prompt. Exact value matching is deliberately
    conservative, not a semantic equivalence judge. A selected direction can be
    explicitly reselected via ``replaced_refs``; a user requirement cannot.
    Existing historical objects are not mutated.
    """
    if current_spec.source_sentence != previous.source_text:
        raise ValueError("a plan rebuild cannot replace the original request")
    current = contract_from_semantic(current_spec, compiler=compiler, clause_map=clause_map,
                                     strict_entity_refs=strict_entity_refs)
    replacements = set(replaced_refs or ())
    groups = _condition_groups(current)
    inherited: dict[str, list[ConstraintBinding]] = {}
    selected_refs: set[str] = set()

    def inherit(base: str, condition: VisualConstraint, values: list[str]) -> None:
        if not _protected(condition):
            return
        if condition.intent_role == ConstraintRole.SELECTED_DIRECTION:
            selected_refs.update((base, condition.id))
            if base in replacements or condition.id in replacements:
                return
        available = [value for item in groups.get(base, []) for value in item.values]
        if any(value not in available for value in values):
            raise ValueError(f"current plan changes or omits protected condition {condition.id}")
        inherited.setdefault(base, []).append(ConstraintBinding(
            values=values, intent_role=condition.intent_role,
            evidence=condition.evidence, source_span=condition.source_span,
            strength=condition.strength,
        ))

    # A character switch is not merely a new pose or palette. Stable names/tags
    # are already mandatory in the native compiler even for unreviewed fields.
    for old in previous.entities:
        if old.kind != EntityKind.CHARACTER:
            continue
        new = next((item for item in current.entities if item.id == old.id), None)
        if new is None or new.name != old.name or new.model_tags != old.model_tags:
            raise ValueError(f"current plan changes character identity {old.id}")
    for base, conditions in _condition_groups(previous).items():
        for condition in conditions:
            released = condition.intent_role == ConstraintRole.SELECTED_DIRECTION and (
                base in replacements or condition.id in replacements
            )
            if _protected(condition) and not released:
                root = base.split(".", 1)[0]
                old_entity = next((item for item in previous.entities if item.id == root), None)
                new_entity = next((item for item in current.entities if item.id == root), None)
                if old_entity and (not new_entity or old_entity.name != new_entity.name):
                    raise ValueError(f"current plan changes protected entity identity {root}")
            inherit(base, condition, condition.values)

    for old in previous.relations:
        if not _protected(old):
            continue
        if old.intent_role == ConstraintRole.SELECTED_DIRECTION:
            selected_refs.add(old.id)
            if old.id in replacements:
                continue
        new = next((item for item in current.relations if
                    (item.subject_ref, item.predicate, item.object_ref) ==
                    (old.subject_ref, old.predicate, old.object_ref)), None)
        if new is None:
            raise ValueError(f"current plan changes or omits protected relation {old.id}")
        for reference in (old.subject_ref, old.object_ref):
            root = reference.split(".", 1)[0]
            old_entity = next(item for item in previous.entities if item.id == root)
            new_entity = next(item for item in current.entities if item.id == root)
            if new_entity.name != old_entity.name:
                raise ValueError(f"current plan changes protected relation entity identity {root}")
        inherited[new.id] = [ConstraintBinding(
            values=[f"{new.subject_ref} {new.predicate} {new.object_ref}"],
            intent_role=old.intent_role, evidence=old.evidence, source_span=old.source_span,
            strength=old.strength,
        )]
    unknown_replacements = replacements - selected_refs
    if unknown_replacements:
        raise ValueError("only reviewed selected directions can be reselected: " + ", ".join(sorted(unknown_replacements)))

    for ref, entries in (constraint_bindings or {}).items():
        if ref in inherited:
            raise ValueError(f"new bindings cannot overwrite inherited commitments: {ref}")
        inherited[ref] = entries
    rebuilt = contract_from_semantic(current_spec, compiler=compiler, clause_map=clause_map,
                                    constraint_bindings=inherited, strict_entity_refs=strict_entity_refs)
    for old in previous.text_intents:
        if any(item.verified and item.kind in {EvidenceKind.USER, EvidenceKind.KNOWLEDGE} for item in old.evidence):
            new = next((item for item in rebuilt.text_intents if item.id == old.id), None)
            if new is None or (new.mode, new.content, new.target_ref) != (old.mode, old.content, old.target_ref):
                raise ValueError(f"current plan changes or omits protected text intent {old.id}")
            rebuilt.text_intents = [old.model_copy(deep=True) if item.id == old.id else item for item in rebuilt.text_intents]
    rebuilt.issues = [item.model_copy(deep=True) for item in previous.issues]
    unresolved = [item for item in previous.clauses if item.blocking and item.status in {ClauseStatus.AMBIGUOUS, ClauseStatus.UNMAPPED}]
    unresolved_ids = {item.id for item in unresolved}
    rebuilt.clauses = [item for item in rebuilt.clauses if item.id not in unresolved_ids] + [item.model_copy(deep=True) for item in unresolved]
    return rebuilt
