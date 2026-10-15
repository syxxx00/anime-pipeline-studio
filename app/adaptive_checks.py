from __future__ import annotations

import json

from .pipeline_graph import ArtifactRef, CheckRecord, Defect
from .pipeline_runtime import RepairAction


CATEGORY_BY_CHECK_TOKEN = {
    "scene_relations": "object_relations",
    "iteration_change": "technical_quality",
    "identity": "character_identity",
    "face": "character_identity",
    "hands": "anatomy_and_hands",
    "hand": "anatomy_and_hands",
    "object": "object_relations",
    "composition": "background_and_composition",
    "background": "background_and_composition",
    "aesthetic": "technical_quality",
    "breakdown": "technical_quality",
    "text": "text_intent",
    "style": "style_match",
}


def _first_region(entry: dict) -> list[float] | None:
    details = entry.get("details") if isinstance(entry.get("details"), dict) else {}
    boxes = details.get("boxes") or []
    if not boxes or not isinstance(boxes[0], (list, tuple)) or len(boxes[0]) != 4:
        return None
    x1, y1, x2, y2 = (float(value) for value in boxes[0])
    if x2 <= x1 or y2 <= y1:
        return None
    return [x1, y1, x2 - x1, y2 - y1]


def check_record_from_artifact(artifact: ArtifactRef) -> CheckRecord:
    entries = artifact.metadata.get("checks", []) if isinstance(artifact.metadata, dict) else []
    defects: list[Defect] = []
    if not entries:
        defects.append(Defect(
            id="check_missing", category="human_review",
            description="候选没有可用的 check 报告，不能自动宣称通过。",
            severity=0.5, confidence=0.4, evidence=[], requires_human=True,
        ))
    for index, entry in enumerate(entries, 1):
        status = str(entry.get("status", "")).lower()
        if status not in {"warning", "error", "failed", "unknown"}:
            continue
        name = str(entry.get("name", "check")).lower()
        category = next((value for token, value in CATEGORY_BY_CHECK_TOKEN.items() if token in name), "human_review")
        # Current detectors provide evidence, not a trustworthy autonomous
        # defect verdict.  Even when they provide a box, keep requires_human so
        # a false text/hand hit cannot silently damage an otherwise good image.
        defects.append(Defect(
            id=f"check_{index}", category=category,
            description=str(entry.get("summary") or entry.get("message") or name),
            severity=0.7 if status in {"error", "failed"} else 0.5,
            confidence=0.4 if status == 'unknown' or category == 'human_review' else 0.65,
            region=_first_region(entry), evidence=[json.dumps(entry, ensure_ascii=False)],
            requires_human=True,
        ))
    if not defects:
        defects.append(Defect(
            id='visual_coverage_unverified', category='human_review',
            description='现有标签/定位信号未覆盖完整画面要求；没有警告不代表生成达标。',
            severity=0.5, confidence=0.4, requires_human=True,
        ))
    return CheckRecord(node_id="component_check", artifact_id=artifact.id, passed=False, defects=defects)


def repair_improved(before: CheckRecord | None, after: CheckRecord, action: RepairAction | None) -> bool:
    """Conservative automatic comparison used before accepting a derivative.

    Missing/low-confidence evidence never counts as improvement.  A passing
    recheck is acceptable; otherwise the targeted defect category must become
    strictly less severe without introducing a new severe category.
    """
    if before is None or action is None or before.artifact_id == after.artifact_id:
        return False
    if any(item.requires_human or item.category == "human_review" or item.confidence < 0.6 for item in after.defects if not item.resolved):
        return False
    target_ids = set(action.defect_ids)
    targeted = [item for item in before.defects if item.id in target_ids]
    if not targeted or any(item.requires_human or item.confidence < .6 for item in targeted):
        return False
    if after.passed and not after.defects:
        return True
    categories = {item.category for item in targeted}
    old_score = sum(item.severity for item in targeted)
    new_score = sum(item.severity for item in after.defects if not item.resolved and item.category in categories)
    # A better hand may not pay for a worse face or background, even when that
    # category already existed or the new issue is below a severity threshold.
    protected = {item.category for item in before.defects + after.defects} - categories
    regressed = any(
        sum(d.severity for d in after.defects if not d.resolved and d.category == cat) >
        sum(d.severity for d in before.defects if not d.resolved and d.category == cat)
        for cat in protected
    )
    return new_score < old_score and not regressed
