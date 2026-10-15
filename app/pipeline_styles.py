from __future__ import annotations

from .config import Settings
from .schemas import GenerateRequest, PIXEL_BRIDGE_PIPELINES, RenderPipeline
from .style_catalog import find_style, inject_anima_style, inject_newbie_style, inject_tag_style


def prepare_request_styles(request: GenerateRequest, settings: Settings) -> GenerateRequest:
    """Resolve style controls into model-native conditioning before graph build.

    Hybrid graphs intentionally expose two independent style channels. This is
    the first reusable routing boundary for future candidate/selector graphs.
    """
    def snapshot_applied(channel: str, selected: str) -> bool:
        if not request.author_proposal:
            return False
        native = (request.native_prompts or {}).get(channel) or {}
        applied = native.get('style_recipe_applied')
        if not isinstance(applied, dict):
            return False
        if applied.get('id') != selected:
            raise ValueError('当前风格选择与已准备的完整配方不同，请重建当前提案条件')
        # The prepared native channel already contains this exact snapshot.
        # Native editor changes remain deliberate overrides, not a reason to
        # append the live catalog again. Other requests keep legacy behavior.
        return True

    if request.render_pipeline in PIXEL_BRIDGE_PIPELINES:
        composition_id = request.style_preset if request.composition_style_preset is None else request.composition_style_preset
        character_id = request.style_preset if request.character_style_preset is None else request.character_style_preset
        composition = find_style(settings, composition_id) if composition_id else None
        character = find_style(settings, character_id) if character_id else None
        updates: dict[str, str] = {}
        source_prompt = request.source_prompt or request.anima_prompt
        source_channel = {RenderPipeline.HYBRID: 'anima', RenderPipeline.ILLUSTRIOUS_NEWBIE: 'illustrious', RenderPipeline.Z_NEWBIE: 'z_image'}[request.render_pipeline]
        composition_baked = snapshot_applied(source_channel, composition_id)
        character_baked = snapshot_applied('newbie', character_id)
        if composition and not composition_baked:
            if request.render_pipeline == RenderPipeline.HYBRID and composition.mode in {"anima", "both"}:
                source_prompt = inject_anima_style(source_prompt, composition)
            elif request.render_pipeline in {RenderPipeline.ILLUSTRIOUS_NEWBIE, RenderPipeline.Z_NEWBIE}:
                source_prompt = inject_tag_style(source_prompt, composition)
            updates["source_prompt"] = source_prompt
            if request.render_pipeline == RenderPipeline.HYBRID:
                updates["anima_prompt"] = source_prompt
        if character and not character_baked and character.mode in {"newbie", "both"}:
            updates["prompt"] = inject_newbie_style(request.prompt, character)
        return request.model_copy(update=updates) if updates else request

    channel = {'newbie': 'newbie', 'anima': 'anima', 'illustrious': 'illustrious', 'z_image_turbo': 'z_image'}.get(request.render_pipeline.value)
    selected = request.style_preset or (request.image_handoff.requested_style_preset if request.image_handoff else '')
    if channel and snapshot_applied(channel, selected):
        return request
    preset = find_style(settings, request.style_preset) if request.style_preset else None
    if not preset:
        return request
    if request.render_pipeline == RenderPipeline.NEWBIE and preset.mode in {"newbie", "both"}:
        return request.model_copy(update={"prompt": inject_newbie_style(request.prompt, preset)})
    if request.render_pipeline == RenderPipeline.ANIMA and preset.mode in {"anima", "both"}:
        return request.model_copy(update={"prompt": inject_anima_style(request.prompt, preset)})
    if request.render_pipeline in {RenderPipeline.ILLUSTRIOUS, RenderPipeline.Z_IMAGE}:
        return request.model_copy(update={"prompt": inject_tag_style(request.prompt, preset)})
    return request
