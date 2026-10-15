from __future__ import annotations

from dataclasses import dataclass

from .schemas import GenerateRequest, RenderPipeline
from .routing_guidance import RoutingGuidanceStore


@dataclass(slots=True)
class RouteDecision:
    pipeline: RenderPipeline
    reason: str
    source_target: str


def choose_route(request: GenerateRequest, guidance: dict[str, dict] | None = None) -> RouteDecision:
    spec = request.semantic_spec if isinstance(request.semantic_spec, dict) else {}
    objects = spec.get("objects") or []
    relations = spec.get("relations") or []
    framing = " ".join(str(value) for value in (spec.get("framing") or [])).lower()
    background = spec.get("background") or []
    complex_contact = bool(relations) or len(objects) >= 1
    environment_focused = (
        len(background) >= 4
        or any(token in framing for token in ("wide", "environment", "landscape", "远景", "环境"))
    )
    trigger = "object_relation" if complex_contact else ("environment" if environment_focused else "simple_identity")
    defaults = {
        "object_relation": RouteDecision(RenderPipeline.ILLUSTRIOUS_NEWBIE, "既有启发式默认：道具/关系触发 Illustrious → NewBie。尚未证明此路线更适合这些要求，也不代表模型固定分工。", "illustrious"),
        "environment": RouteDecision(RenderPipeline.HYBRID, "既有启发式默认：环境/远景触发 Anima → NewBie。尚未证明此路线更适合这些要求，也不代表模型固定分工。", "anima"),
        "simple_identity": RouteDecision(RenderPipeline.NEWBIE, "既有启发式默认：当前条件选择 NewBie 单模型。不是同条件对照所得的最优结论。", "newbie"),
    }
    if guidance is None:
        guidance = RoutingGuidanceStore().active_map()
    published = guidance.get(trigger)
    if not published:
        return defaults[trigger]
    pipeline = RenderPipeline(str(published["pipeline"]))
    source = "illustrious" if pipeline == RenderPipeline.ILLUSTRIOUS_NEWBIE else ("anima" if pipeline == RenderPipeline.HYBRID else "newbie")
    evidence = f"实验 {published.get('experiment_id', 'unknown')} / 指导 {published.get('id', 'unknown')}"
    return RouteDecision(pipeline, f"采用你已发布的研究指导（{evidence}）：{published.get('note') or trigger}", source)


def route_request(request: GenerateRequest) -> tuple[GenerateRequest, RouteDecision]:
    if request.render_pipeline != RenderPipeline.ADAPTIVE:
        return request, RouteDecision(request.render_pipeline, "请求显式指定了执行路线。", request.model_id.value)
    decision = choose_route(request)
    native = request.native_prompts or {}
    newbie = native.get("newbie") or {}
    source = native.get(decision.source_target) or {}
    update = {
        "render_pipeline": decision.pipeline,
        "prompt": str(newbie.get("positive") or request.prompt),
        "negative_prompt": str(newbie.get("negative") or request.negative_prompt),
    }
    if decision.pipeline in {RenderPipeline.HYBRID, RenderPipeline.ILLUSTRIOUS_NEWBIE, RenderPipeline.Z_NEWBIE}:
        update["source_prompt"] = str(source.get("positive") or request.source_prompt or request.anima_prompt)
        if not request.use_default_negative_prompt and request.source_negative_prompt is None and "negative" in source:
            update["source_negative_prompt"] = str(source["negative"])
    return request.model_copy(update=update), decision
