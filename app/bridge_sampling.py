"""One source-stage parameter resolver shared by preview and execution."""
from .sampling_profiles import PROFILES, SOURCE_PROFILES
from .schemas import GenerateRequest


def source_request(request: GenerateRequest, *, profile: str | None = None) -> GenerateRequest:
    selected = SOURCE_PROFILES.get(request.render_pipeline.value)
    profile = profile or selected
    if profile not in {'anima', 'illustrious', 'z_image_turbo'}:
        raise ValueError('不是受支持的像素桥接源阶段')
    params = dict(PROFILES[profile])
    if request.source_parameters is not None:
        if profile != selected:
            raise ValueError('不能将已指定的源阶段参数应用到另一模型')
        params.update(request.source_parameters.model_dump(exclude={'seed'}))
    seed = request.source_parameters.seed if request.source_parameters else None
    prompt = request.source_prompt or request.anima_prompt
    negative = request.source_negative_prompt
    if negative is None:
        if request.parameter_policy == 'explicit':
            from .model_conditioning import negative_for_target
            channel = 'z_image' if profile == 'z_image_turbo' else profile
            negative = negative_for_target(request.model_dump(mode='json'), (request.native_prompts or {}).get(channel) or {}, profile)
        else:
            negative = '' if profile == 'anima' else request.negative_prompt
    return GenerateRequest(
        **params, render_pipeline=profile, parameter_policy=request.parameter_policy,
        prompt=prompt, anima_prompt=prompt, negative_prompt=negative,
        use_default_negative_prompt=request.use_default_negative_prompt if request.parameter_policy == 'legacy' else False,
        width=request.width, height=request.height, batch_size=1,
        seed=request.seed if seed is None else seed,
        face_detail_enabled=False, check_enabled=False,
    )
