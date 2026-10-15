from __future__ import annotations

import json
from pathlib import Path

from ..config import PROJECT_ROOT
from ..knowledge import apply_knowledge
from ..newbie_compiler import CompileResult, build_negative, compile_anima_prompt, compile_xml, fallback_semantic
from ..schemas import ExperimentCreateRequest, NewbieStyleMode


PLAN_PATH = PROJECT_ROOT / "config" / "overnight_matrix_v1.json"
KNOWLEDGE_SELECTIONS = {
    "W": ("w", "default"),
    "幽灵鲨": ("specter", "default"),
    "阿米娅": ("amiya_caster", "default"),
    "薇薇安娜": ("viviana", "default"),
    "维什戴尔": ("wisadel", "default"),
    "归溟幽灵鲨": ("specter_unchained", "default"),
}


def load_plan(path: Path = PLAN_PATH) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def plan_summary(plan: dict) -> dict:
    base = len(plan["texts"]) * len(plan["recipes"]) * len(plan["seeds"])
    focused = len(plan["focused_denoise"]["text_ids"]) * len(plan["focused_denoise"]["source_pipelines"]) * len(plan["focused_denoise"]["values"]) * len(plan["seeds"])
    return {
        "name": plan["name"], "notes": plan["notes"],
        "texts": plan["texts"], "recipes": plan["recipes"], "seeds": plan["seeds"],
        "base_items": base, "focused_items": focused, "total_items": base + focused,
        "starts_automatically": False,
    }


def _compiled(text: dict) -> dict:
    mode = NewbieStyleMode(text.get("style_mode", NewbieStyleMode.STANDARD.value))
    result = CompileResult(fallback_semantic(text["sentence"], mode), "deterministic_research", [])
    selection = KNOWLEDGE_SELECTIONS.get(text["character"])
    if selection:
        result = apply_knowledge(result, "arknights", selection[0], selection[1])
    return {
        "xml": compile_xml(result.spec),
        "negative_prompt": build_negative(result.spec),
        "natural": compile_anima_prompt(result.spec),
        "spec": result.spec.model_dump(mode="json"),
    }


def _profile(pipeline: str, compiled: dict) -> dict:
    if pipeline in {"newbie", "anima_newbie", "illustrious_newbie", "z_newbie"}:
        return {"model_id": "newbie", "prompt": compiled["xml"], "steps": 28, "cfg": 4.5,
                "shift": 6, "sampler": "res_multistep", "scheduler": "linear_quadratic",
                "face_detail_enabled": True}
    if pipeline == "anima":
        return {"model_id": "anima_turbo", "prompt": compiled["natural"], "steps": 10, "cfg": 1,
                "shift": 3, "sampler": "euler", "scheduler": "simple", "face_detail_enabled": False}
    if pipeline == "illustrious":
        return {"model_id": "illustrious", "prompt": compiled["natural"], "steps": 28, "cfg": 5,
                "shift": 0, "sampler": "dpmpp_2m_sde", "scheduler": "karras", "face_detail_enabled": False}
    if pipeline == "z_image_turbo":
        return {"model_id": "z_image_turbo", "prompt": compiled["natural"], "steps": 9, "cfg": 1,
                "shift": 3, "sampler": "euler", "scheduler": "simple", "face_detail_enabled": False}
    raise ValueError(f"unsupported pipeline: {pipeline}")


def _request(text: dict, compiled: dict, recipe: dict, seed: int) -> dict:
    pipeline = recipe["pipeline"]
    return {
        **_profile(pipeline, compiled), "pipeline_version": "adaptive", "mode": "character",
        "negative_prompt": compiled["negative_prompt"], "width": 1024, "height": 1024,
        "batch_size": 1, "seed": seed, "cfg_rescale": .9, "render_pipeline": pipeline,
        "anima_prompt": compiled["natural"], "source_prompt": compiled["natural"],
        "semantic_spec": compiled["spec"], "style_preset": recipe.get("style_preset", ""),
        "composition_style_preset": recipe.get("composition_style_preset"),
        "character_style_preset": recipe.get("character_style_preset"),
        "hybrid_denoise": recipe.get("hybrid_denoise", .18), "highres_enabled": False,
        "check_enabled": True, "hand_repair_enabled": False, "text_clean_enabled": False,
        "style_category": "auto", "output_label": f"{recipe['id']}__{text['id']}__seed-{seed}",
    }


def build_boundary_manifest(plan: dict) -> ExperimentCreateRequest:
    compiled = {text["id"]: _compiled(text) for text in plan["texts"]}
    items: list[dict] = []
    for text in plan["texts"]:
        for recipe in plan["recipes"]:
            for seed in plan["seeds"]:
                items.append({"axes": {
                    "model": recipe["id"], "pipeline": recipe["pipeline"],
                    "character": text["character"], "scene": text["id"],
                    "prompt_form": "audited_semantic_contract", "style": recipe["id"],
                    "seed": seed, "boundary": text["boundary"],
                }, "request": _request(text, compiled[text["id"]], recipe, seed)})
    focused = plan["focused_denoise"]
    by_id = {text["id"]: text for text in plan["texts"]}
    for text_id in focused["text_ids"]:
        text = by_id[text_id]
        for pipeline in focused["source_pipelines"]:
            for denoise in focused["values"]:
                recipe = {
                    "id": f"{pipeline}_strength_{round(denoise * 100):02d}", "pipeline": pipeline,
                    "composition_style_preset": focused["composition_style_preset"],
                    "character_style_preset": focused["character_style_preset"], "hybrid_denoise": denoise,
                }
                for seed in plan["seeds"]:
                    items.append({"axes": {
                        "model": recipe["id"], "pipeline": pipeline, "character": text["character"],
                        "scene": text_id, "prompt_form": "audited_semantic_contract",
                        "style": recipe["id"], "seed": seed, "boundary": text["boundary"],
                    }, "request": _request(text, compiled[text_id], recipe, seed)})
    return ExperimentCreateRequest.model_validate({
        "name": plan["name"], "notes": plan["notes"], "external_llm_used": False, "items": items,
    })
