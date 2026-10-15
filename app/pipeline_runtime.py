from __future__ import annotations

from collections import defaultdict
from typing import Literal

from pydantic import BaseModel, Field

from .pipeline_graph import Defect, ModelRole, repair_role_for
from .reviews import analyze_review


class RepairAction(BaseModel):
    id: str
    defect_ids: list[str]
    role: ModelRole
    operation: Literal["local_inpaint", "component_regenerate", "deterministic_text", "human_mask", "report_only"]
    target_region: list[float] | None = Field(default=None, min_length=4, max_length=4)
    instruction: str
    preserve: list[str] = Field(default_factory=list)
    acceptance: list[str] = Field(default_factory=list)
    max_attempts: int = Field(default=2, ge=1, le=5)
    rollback_on_regression: bool = True
    auto_runnable: bool = False


class RepairPlan(BaseModel):
    version: Literal["3.0"] = "3.0"
    actions: list[RepairAction] = Field(default_factory=list)
    unresolved_defect_ids: list[str] = Field(default_factory=list)
    requires_user_mask: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


FACET_TO_CATEGORY = {
    "identity": "character_identity",
    "face": "character_identity",
    "hands": "anatomy_and_hands",
    "anatomy": "anatomy_and_hands",
    "object": "object_presence",
    "background": "background_and_composition",
    "composition": "background_and_composition",
    "style": "style_match",
    "quality": "technical_quality",
    "semantics": "object_relations",
    "text": "text_intent",
}


def defects_from_review(text: str, *, artwork_id: str = "") -> list[Defect]:
    """Natural language stays authoritative; this is a replaceable routing index."""
    analysis = analyze_review(text)
    problems = analysis.get("problem_excerpts") or []
    if not problems and any(token in text for token in ("需要", "应该", "希望", "改成", "重画", "修")):
        problems = [text.strip()]
    categories = list(dict.fromkeys(FACET_TO_CATEGORY[facet] for facet in analysis.get("facets", []) if facet in FACET_TO_CATEGORY))
    if not categories and problems:
        categories = ["human_review"]
    defects: list[Defect] = []
    for index, category in enumerate(categories, 1):
        excerpt = next((item for item in problems if item), text.strip())
        defects.append(Defect(
            id=f"review_{artwork_id or 'image'}_{index}",
            category=category,
            description=excerpt,
            severity=0.6,
            confidence=0.85 if category != "human_review" else 0.4,
            evidence=[text.strip()],
            suggested_role=repair_role_for(category),
            requires_human=True,
        ))
    return defects


def build_repair_plan(defects: list[Defect], *, locked_refs: list[str] | None = None) -> RepairPlan:
    locked_refs = locked_refs or []
    active = [defect for defect in defects if not defect.resolved]
    grouped: dict[tuple[str, ModelRole], list[Defect]] = defaultdict(list)
    unresolved: list[str] = []
    needs_mask: list[str] = []
    actions: list[RepairAction] = []

    for defect in active:
        role = defect.suggested_role or repair_role_for(defect.category)
        if defect.confidence < 0.6 or role == ModelRole.HUMAN:
            unresolved.append(defect.id)
            continue
        grouped[(defect.category, role)].append(defect)

    for (category, role), items in grouped.items():
        regions = [item.region for item in items if item.region is not None]
        region = regions[0] if len(regions) == len(items) == 1 else None
        if role == ModelRole.DETERMINISTIC and category == "text_intent":
            operation = "deterministic_text"
            # Exact glyph rendering is deterministic only after the user has
            # supplied content, font intent and placement.  A detector warning
            # alone is not enough information to author the text layer.
            auto = False
        elif region is not None:
            operation = "local_inpaint"
            # The installed masked inpaint executor is NewBie-native.  Do not
            # pretend an Anima/Illustrious diagnosis can be repaired by feeding
            # its latent state into an incompatible model.
            auto = role == ModelRole.NEWBIE and not any(item.requires_human for item in items)
        else:
            operation = "human_mask"
            auto = False
            needs_mask.extend(item.id for item in items)
        actions.append(RepairAction(
            id=f"repair_{len(actions) + 1}",
            defect_ids=[item.id for item in items],
            role=role,
            operation=operation,
            target_region=region,
            instruction="；".join(dict.fromkeys(item.description for item in items)),
            preserve=locked_refs,
            acceptance=[f"resolve:{item.id}" for item in items] + ["locked_components_not_regressed"],
            auto_runnable=auto,
        ))

    notes = []
    if needs_mask:
        notes.append("这些问题缺少可靠区域坐标；必须由用户圈选，不能把整图重绘伪装成局部修复。")
    if unresolved:
        notes.append("低置信或纯审美问题保留给人工判断，不自动触发生成。")
    return RepairPlan(actions=actions, unresolved_defect_ids=unresolved, requires_user_mask=needs_mask, notes=notes)
