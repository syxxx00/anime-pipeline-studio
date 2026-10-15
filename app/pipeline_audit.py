from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


PIXEL_BRIDGES = {"anima_newbie", "illustrious_newbie", "z_newbie"}


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _check_map(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    return {item.get("name", ""): item for item in _load_json(path)}


def _source_contains_object(source_prompt: str, objects: list[dict]) -> bool:
    prompt = source_prompt.casefold()
    return all(str(item.get("name", "")).casefold() in prompt for item in objects if item.get("name"))


def audit_request(request_path: Path, expectations: dict[str, dict] | None = None) -> dict:
    request = _load_json(request_path)
    spec = request.get("semantic_spec") or {}
    scene_id = request_path.parent.name
    expected = (expectations or {}).get(scene_id, {})
    pipeline = str(request.get("render_pipeline", ""))
    objects = list(spec.get("objects") or [])
    relations = list(spec.get("relations") or [])
    source_prompt = str(request.get("source_prompt") or request.get("anima_prompt") or "")
    flags: list[str] = []

    if str(request.get("pipeline_version", "")).startswith("2"):
        flags.append("legacy_pipeline_v2")
    if pipeline in PIXEL_BRIDGES:
        flags.append("pixel_bridge")
        if request.get("source_prompt") == request.get("anima_prompt"):
            flags.append("shared_generic_source_prompt")
        if float(request.get("hybrid_denoise", 0.18)) == 0.18:
            flags.append("fixed_bridge_denoise_0_18")
    if expected.get("requires_objects") and not objects:
        flags.append("semantic_objects_missing")
    if expected.get("requires_relations") and not relations:
        flags.append("semantic_relations_missing")
    if objects and source_prompt and not _source_contains_object(source_prompt, objects):
        flags.append("source_prompt_drops_objects")
    if expected.get("text_intent"):
        # Pipeline v2 had no first-class text contract; no_text in quality or
        # negative tags cannot distinguish forbidden, exact, canonical, and UI text.
        flags.append("text_intent_not_structured")

    check_path = request_path.with_name(request_path.name.replace("__request.json", "__check.json"))
    checks = _check_map(check_path)
    warning_checks = sorted(name for name, item in checks.items() if item.get("status") == "warning")
    aesthetic = checks.get("aesthetic", {}).get("details", {}).get("percentile")
    image_path = request_path.with_name(request_path.name.replace("__request.json", "__01.png"))
    return {
        "job_id": request_path.name.rsplit("__", 2)[-2] if "__" in request_path.name else "",
        "character": request_path.parent.parent.name,
        "scene": scene_id,
        "variant": str(request.get("output_label", "")).split("__", 1)[0],
        "pipeline": pipeline,
        "seed": request.get("seed"),
        "image_exists": image_path.exists(),
        "check_exists": check_path.exists(),
        "object_count": len(objects),
        "relation_count": len(relations),
        "contamination": sorted(set(flags)),
        "warning_checks": warning_checks,
        "aesthetic_percentile": aesthetic,
    }


def audit_experiment(
    experiment_root: Path,
    *,
    matrix_config: dict | None = None,
    expectations: dict[str, dict] | None = None,
) -> dict:
    observations = [
        audit_request(path, expectations)
        for path in sorted(experiment_root.rglob("*__request.json"))
    ]
    contamination = Counter(flag for item in observations for flag in item["contamination"])
    by_pipeline: dict[str, dict] = {}
    grouped: dict[str, list[dict]] = defaultdict(list)
    for item in observations:
        grouped[item["pipeline"]].append(item)
    for pipeline, rows in sorted(grouped.items()):
        aesthetic = [float(row["aesthetic_percentile"]) for row in rows if row["aesthetic_percentile"] is not None]
        warnings = Counter(name for row in rows for name in row["warning_checks"])
        by_pipeline[pipeline] = {
            "count": len(rows),
            "average_aesthetic_percentile": round(sum(aesthetic) / len(aesthetic), 3) if aesthetic else None,
            "warning_checks": dict(sorted(warnings.items())),
        }

    planned = None
    planned_scenes: list[str] = []
    if matrix_config:
        planned_scenes = [str(item["id"]) for item in matrix_config.get("texts", [])]
        planned = len(planned_scenes) * len(matrix_config.get("recipes", [])) * len(matrix_config.get("seeds", []))
    observed_scenes = sorted({item["scene"] for item in observations})
    return {
        "schema_version": "pipeline-v2-audit/1",
        "experiment_id": experiment_root.name,
        "planned_request_count": planned,
        "observed_request_count": len(observations),
        "image_count": sum(bool(item["image_exists"]) for item in observations),
        "characters": sorted({item["character"] for item in observations}),
        "observed_scenes": observed_scenes,
        "missing_planned_scenes": sorted(set(planned_scenes) - set(observed_scenes)),
        "contamination_counts": dict(sorted(contamination.items())),
        "by_pipeline": by_pipeline,
        "safe_uses": [
            "broad style tendency",
            "composition tendency with manual review",
            "relative visual novelty of model combinations",
            "seed stability hints",
        ],
        "unsafe_uses": [
            "precise object capability when semantic_objects_missing is present",
            "precise relation capability when semantic_relations_missing is present",
            "native source-model prompt comparison for shared_generic_source_prompt",
            "text rendering capability because Pipeline v2 had no text intent contract",
            "automatic winner selection from aesthetic score alone",
        ],
        "observations": observations,
    }


def render_markdown(report: dict) -> str:
    pipelines = [
        f"| {name} | {item['count']} | {item['average_aesthetic_percentile']} | "
        f"{item['warning_checks'].get('identity', 0)} | {item['warning_checks'].get('composition', 0)} | "
        f"{item['warning_checks'].get('text_scene', 0)} |"
        for name, item in report["by_pipeline"].items()
    ]
    flags = [f"- `{name}`: {count}" for name, count in report["contamination_counts"].items()]
    missing = ", ".join(report["missing_planned_scenes"]) or "none"
    return "\n".join([
        "# Pipeline v2 experiment audit",
        "",
        f"- Experiment: `{report['experiment_id']}`",
        f"- Planned requests: {report['planned_request_count']}",
        f"- Observed requests: {report['observed_request_count']}",
        f"- Images found: {report['image_count']}",
        f"- Missing planned scenes: {missing}",
        "",
        "## Contamination flags",
        "",
        *flags,
        "",
        "## Detector summary (not a user-preference ranking)",
        "",
        "| Pipeline | N | Mean aesthetic percentile | Identity warnings | Composition warnings | Text warnings |",
        "|---|---:|---:|---:|---:|---:|",
        *pipelines,
        "",
        "## Safe uses",
        "",
        *[f"- {item}" for item in report["safe_uses"]],
        "",
        "## Unsafe uses",
        "",
        *[f"- {item}" for item in report["unsafe_uses"]],
        "",
        "The JSON companion keeps per-job contamination flags for future capability routing.",
        "",
    ])
