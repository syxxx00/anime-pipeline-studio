"""Freeze route, conditioning and execution parameters before queue/preview."""
from .adaptive_router import RouteDecision, route_request
from .config import Settings
from .pipeline_styles import prepare_request_styles
from .schemas import GenerateRequest, RenderPipeline


def execution_parameters(request: GenerateRequest) -> tuple[GenerateRequest, dict]:
    """Preserve old Anima experiments without overriding new user choices.

    This compatibility rule describes historical execution, not recommended
    quality settings. It must run before the request snapshot is persisted.
    """
    if request.parameter_policy != "legacy" or request.render_pipeline != RenderPipeline.ANIMA:
        return request, {}
    baseline = dict(negative_prompt="", steps=10, cfg=1.0, shift=3.0,
                    sampler="euler", scheduler="simple", lora_name=None)
    changes = {key: {"requested": getattr(request, key), "effective": value}
               for key, value in baseline.items() if getattr(request, key) != value}
    return request.model_copy(update=baseline), changes


def prepare_generation_request(request: GenerateRequest, settings: Settings) -> tuple[GenerateRequest, RouteDecision, dict]:
    if request.source_parameters and request.source_parameters.lora_name:
        from .lora_catalog import describe_lora
        root = settings.path_map['loras'].resolve()
        path = (root / request.source_parameters.lora_name).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            raise ValueError('源阶段 LoRA 必须是本地模型目录内的有效文件')
        if request.source_parameters.model_id not in describe_lora(path, root)['compatible_models']:
            raise ValueError('源阶段 LoRA 缺少兼容架构证据，不按文件名猜测兼容性')
    request, decision = route_request(request)
    request = prepare_request_styles(request, settings)
    request, adjustments = execution_parameters(request)
    return request, decision, adjustments
