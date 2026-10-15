from __future__ import annotations

from dataclasses import dataclass, field
from html import unescape
from enum import StrEnum
import re
from xml.etree import ElementTree as ET

from .scene_contract import (
    ConstraintStrength,
    EntityKind,
    SceneContractV3,
    TextIntentMode,
)


class PromptTarget(StrEnum):
    NEWBIE = "newbie"
    ILLUSTRIOUS = "illustrious"
    ANIMA = "anima"
    Z_IMAGE = "z_image"


@dataclass(slots=True)
class CompiledContractPrompt:
    target: PromptTarget
    positive: str
    negative: str
    trace: dict[str, str] = field(default_factory=dict)
    postprocess: list[dict] = field(default_factory=list)
    # Exact compiler entries, not a comma parse of the serialized negative.
    negative_terms: list[str] = field(default_factory=list)

    def missing_required_refs(self, contract: SceneContractV3) -> list[str]:
        required: list[str] = []
        for entity in contract.entities:
            required.append(f"entity:{entity.id}:identity")
            required.extend(
                item.id for item in entity.constraints
                if item.strength == ConstraintStrength.REQUIRED
            )
        required.extend(
            item.id for item in contract.relations
            if item.strength == ConstraintStrength.REQUIRED
        )
        required.extend(item.id for item in contract.text_intents)
        # A trace key is not evidence that its payload reached the prompt.
        positive = self.positive
        if self.target == PromptTarget.NEWBIE:
            try:
                positive = " ".join(ET.fromstring(positive).itertext())
            except ET.ParseError:
                return sorted(set(required))
        def present(reference: str) -> bool:
            channel, _, payload = self.trace.get(reference, "").partition(":")
            if not payload:
                return False
            if channel == "positive":
                return payload in positive
            if channel == "negative":
                return payload in (unescape(self.negative) if self.target == PromptTarget.NEWBIE else self.negative)
            if channel == "postprocess":
                return any(item.get("intent_id") == reference and
                           f"{item.get('operation')}:{item.get('content')}" == payload
                           for item in self.postprocess)
            return False
        return sorted(reference for reference in set(required) if not present(reference))

    def require_complete(self, contract: SceneContractV3) -> None:
        missing = self.missing_required_refs(contract)
        if missing:
            raise ValueError(f"{self.target.value} compiler dropped required contract refs: {', '.join(missing)}")


def _ref_phrase(reference: str, names: dict[str, str]) -> str:
    root, dot, component = reference.partition(".")
    base = names.get(root, root)
    return f"{base} {component.replace('_', ' ')}" if dot else base


def _collect(contract: SceneContractV3, target: PromptTarget) -> tuple[list[str], list[str], dict[str, str], list[dict]]:
    positive: list[str] = []
    negative: list[str] = list(contract.global_exclusions)
    trace: dict[str, str] = {}
    postprocess: list[dict] = []
    names = {entity.id: entity.name for entity in contract.entities}

    for entity in contract.entities:
        tags = entity.model_tags.get(target.value, [])
        identity = ", ".join([entity.name, *tags])
        positive.append(identity)
        trace[f"entity:{entity.id}:identity"] = f"positive:{identity}"
        for constraint in entity.constraints:
            phrase = ", ".join(constraint.values)
            if constraint.strength == ConstraintStrength.FORBIDDEN:
                negative.append(phrase)
                trace[constraint.id] = f"negative:{phrase}"
            else:
                positive.append(phrase)
                trace[constraint.id] = f"positive:{phrase}"
        # AuthorDraft positions reach region_hint, not a semantic constraint.
        # Preserve their entity binding in prose/tag natives as well as in the
        # existing NewBie XML slots. This is text presence, not spatial control
        # or evidence that the proposed position was realized in an image.
        position_name = entity.name
        if sum(name == entity.name for name in names.values()) > 1:
            position_name = f"{entity.name} ({entity.id})"
        for position in entity.region_hint:
            if not position.strip():
                continue
            phrase = f"{position_name} position: {position}"
            if phrase not in positive:
                positive.append(phrase)

    for relation in contract.relations:
        phrase = (
            f"{_ref_phrase(relation.subject_ref, names)} {relation.predicate} "
            f"{_ref_phrase(relation.object_ref, names)}"
        )
        if relation.strength == ConstraintStrength.FORBIDDEN:
            negative.append(phrase)
            trace[relation.id] = f"negative:{phrase}"
        else:
            positive.append(phrase)
            trace[relation.id] = f"positive:{phrase}"

    positive.extend(contract.composition.framing)
    positive.extend(contract.composition.camera)
    positive.extend(contract.composition.layout)
    positive.extend(contract.composition.depth)
    positive.extend(contract.style.description)
    negative.extend(contract.style.exclusions)

    for intent in contract.text_intents:
        if intent.mode == TextIntentMode.FORBIDDEN:
            phrase = "text, letters, words, watermark, signature, logo, pseudo-text, garbled glyphs"
            negative.append(phrase)
            trace[intent.id] = f"negative:{phrase}"
        elif intent.mode == TextIntentMode.DECORATIVE_UI:
            phrase = "abstract interface color blocks with no readable glyphs"
            positive.append(phrase)
            negative.append("readable text, pseudo-text, garbled interface glyphs")
            trace[intent.id] = f"positive:{phrase}"
        else:
            # Exact and canonical text are deterministic post-processing work.
            # Asking a diffusion model to spell it is not accepted as coverage.
            phrase = "clean unobstructed surface reserved for later typography"
            positive.append(phrase)
            operation = {
                "operation": "render_text",
                "intent_id": intent.id,
                "mode": intent.mode.value,
                "content": intent.content,
                "language": intent.language,
                "target_ref": intent.target_ref,
                "placement": intent.placement,
                "appearance": intent.appearance,
                "verify_with_ocr": intent.exact_content_required,
            }
            postprocess.append(operation)
            trace[intent.id] = f"postprocess:render_text:{intent.content}"
    return positive, negative, trace, postprocess


def _neutral_quantity(values: list[str]) -> bool:
    """Recognize quantity-only legacy slots, not infer gender from a name.

    The internal historical field is named gender even when the semantic
    producer supplied only count. Unknown/native descriptions stay intact.
    """
    tokens = [token.strip().lower() for value in values for token in value.split(',') if token.strip()]
    return bool(tokens) and all(token in {'solo', 'duo', 'trio', 'group', 'multiple subjects'}
        or re.fullmatch(r'[0-9]+', token)
        or re.fullmatch(r'(?:[0-9]+|one|two|three|four|five|six|seven|eight|nine|ten)\s*'
            r'(?:subjects?|persons?|people|animals?|cats?|dogs?|figures?)', token)
        for token in tokens)


def _compile_newbie(contract: SceneContractV3, *, scene_narrative: str | None = None) -> CompiledContractPrompt:
    _, negative, trace, postprocess = _collect(contract, PromptTarget.NEWBIE)
    # Use the published NewBie character_N/general_tags structure, not the
    # application's internal contract schema. Keep names/tags and binding prose
    # in actual XML text, where they reach the text encoder.
    root = ET.Element("img")
    groups: dict[str, list[str]] = {key: [] for key in
        ("count", "style", "background", "atmosphere", "quality", "resolution", "artist", "objects", "other")}
    groups['quality'] = ['very_aesthetic', 'masterpiece', 'coherent shapes']
    groups['resolution'] = ['max_high_resolution']
    character_index = 0
    for entity in contract.entities:
        identity = ", ".join([entity.name, *entity.model_tags.get("newbie", [])])
        constraints = [item for item in entity.constraints if item.strength != ConstraintStrength.FORBIDDEN]
        if entity.kind == EntityKind.CHARACTER:
            character_index += 1
            character = ET.SubElement(root, f"character_{character_index}")
            slots = {key: [] for key in ("n", "gender", "appearance", "clothing", "expression", "action", "position")}
            slots["n"].append(identity)
            slots["position"].extend(entity.region_hint)
            aliases = {"identity": "n", "body_action": "action", "left_hand": "action", "right_hand": "action"}
            for item in constraints:
                slot = aliases.get(item.name, item.name)
                phrase = ", ".join(item.values)
                if slot in slots:
                    if slot == "gender":
                        groups["count"].append(phrase)
                        if _neutral_quantity(item.values):
                            # Keep the complete quantity in its native channel;
                            # do not label solo/subjects/species as a sex.
                            continue
                    # Retain laterality even if a professional supplied only a
                    # short value such as "holds phone" in left_hand.
                    if item.name in {"left_hand", "right_hand"}:
                        phrase = f"{item.name.replace('_', ' ')}: {phrase}"
                    slots[slot].append(phrase)
                else:
                    groups["other"].append(f"{entity.name} {item.name}: {phrase}")
            for tag, values in slots.items():
                ET.SubElement(character, tag).text = ", ".join(values)
        elif entity.kind == EntityKind.ENVIRONMENT:
            groups["background"].append(identity)
            for item in constraints:
                slot = "atmosphere" if item.name in {"lighting", "atmosphere"} else "background"
                groups[slot].append(", ".join(item.values))
        else:
            slot = "objects" if entity.kind == EntityKind.OBJECT else "other"
            parts = [identity, *[f"{item.name}: {', '.join(item.values)}" for item in constraints], *entity.region_hint]
            groups[slot].append("; ".join(parts))
    for relation in contract.relations:
        if relation.strength != ConstraintStrength.FORBIDDEN:
            groups["other"].append(trace[relation.id].removeprefix("positive:"))
    groups["other"].extend([
        *contract.composition.framing,
        *contract.composition.camera,
        # A whole picture narrative has a distinct published NewBie channel.
        # Move only the explicitly supplied, exact layout value; do not parse,
        # summarize or remove independent staging conditions from the contract.
        *[value for value in contract.composition.layout if value != scene_narrative],
        *contract.composition.depth,
    ])
    groups["style"].extend(contract.style.description)
    for intent in contract.text_intents:
        if intent.mode == TextIntentMode.DECORATIVE_UI:
            groups["other"].append(trace[intent.id].removeprefix("positive:"))
        elif intent.mode in {TextIntentMode.EXACT, TextIntentMode.CANONICAL}:
            groups["other"].append("clean unobstructed surface reserved for later typography")
    general = ET.SubElement(root, "general_tags")
    for tag, values in groups.items():
        ET.SubElement(general, tag).text = ", ".join(values)
    if scene_narrative:
        ET.SubElement(root, "caption").text = scene_narrative
    xml = ET.tostring(root, encoding="unicode", short_empty_elements=False)
    negative_terms = list(dict.fromkeys(negative))
    return CompiledContractPrompt(PromptTarget.NEWBIE, xml, ", ".join(negative_terms),
                                  trace, postprocess, negative_terms)


def _compile_illustrious(contract: SceneContractV3) -> CompiledContractPrompt:
    positive, negative, trace, postprocess = _collect(contract, PromptTarget.ILLUSTRIOUS)
    tags = ["masterpiece", "best quality", "anime illustration", *positive]
    negative_terms = list(dict.fromkeys(item for item in negative if item))
    return CompiledContractPrompt(
        PromptTarget.ILLUSTRIOUS,
        ", ".join(dict.fromkeys(item for item in tags if item)),
        ", ".join(negative_terms),
        trace,
        postprocess,
        negative_terms,
    )


def _compile_narrative(contract: SceneContractV3, target: PromptTarget) -> CompiledContractPrompt:
    positive, negative, trace, postprocess = _collect(contract, target)
    character_names = [item.name for item in contract.entities if item.kind == EntityKind.CHARACTER]
    opening = f"Create an anime illustration featuring {', '.join(character_names)}."
    body = ". ".join(item.rstrip(". ") for item in positive if item)
    if target == PromptTarget.ANIMA:
        opening = "You are an assistant designed to generate a coherent anime image. " + opening
    negative_terms = list(dict.fromkeys(item for item in negative if item))
    return CompiledContractPrompt(
        target,
        f"{opening} {body}.",
        ", ".join(negative_terms),
        trace,
        postprocess,
        negative_terms,
    )


def compile_contract_english(contract: SceneContractV3) -> CompiledContractPrompt:
    """One shared English condition, without model-specific tags or quality defaults.

    Required constraints are retained, not translated/invented by this renderer.
    The native comparison is a complete adapter condition, not a pure syntax test.
    """
    contract.require_generation_ready()
    plain = contract.model_copy(deep=True)
    for entity in plain.entities:
        entity.model_tags = {}
    positive, negative, trace, postprocess = _collect(plain, PromptTarget.Z_IMAGE)
    result = CompiledContractPrompt(PromptTarget.Z_IMAGE,
        '. '.join(item.rstrip('. ') for item in positive if item) + '.',
        ', '.join(negative), trace, postprocess, list(negative))
    result.require_complete(plain)
    return result


def compile_contract(contract: SceneContractV3, target: PromptTarget | str, *,
                     scene_narrative: str | None = None) -> CompiledContractPrompt:
    contract.require_generation_ready()
    target = PromptTarget(target)
    if target == PromptTarget.NEWBIE:
        result = _compile_newbie(contract, scene_narrative=scene_narrative)
    elif target == PromptTarget.ILLUSTRIOUS:
        result = _compile_illustrious(contract)
    else:
        result = _compile_narrative(contract, target)
    result.require_complete(contract)
    return result
