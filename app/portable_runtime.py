"""Resolve operator-owned runtime settings; no service startup or downloads."""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from .local_authoring_runtime import (RuntimeConfig, LocalAuthoringRuntime,
    FrozenFileStat, _LOW_HEADROOM_TEXT_MODEL, _BOUNDED_7B_FIRST,
    _BOUNDED_9B_VISUAL_MODEL)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _path(settings, name, default):
    value = getattr(settings, name, '')
    return Path(value).expanduser() if value else Path(default)


def configured_runtime(settings, *, visual=False, model_variant='4b', freeze_visual=True) -> RuntimeConfig:
    config = RuntimeConfig()
    root = _path(settings, 'local_llm_runtime_dir', PROJECT_ROOT / 'runtime')
    defaults = {
        '4b': root / 'models/Qwen3.5-4B-GGUF/Qwen3.5-4B-Q4_K_M.gguf',
        '7b': root / 'models/Qwen2.5-7B-Instruct-GGUF/qwen2.5-7b-instruct-q4_k_m-00001-of-00002.gguf',
        '9b': root / 'models/Qwen3.5-9B-GGUF/Qwen3.5-9B-Q4_K_M.gguf',
    }
    fields = {'4b': 'local_llm_model', '7b': 'local_llm_model_7b', '9b': 'local_llm_model_9b'}
    if model_variant not in fields:
        raise ValueError('Unknown explicit local language-model variant')
    model = _path(settings, fields[model_variant], defaults[model_variant])
    backend = getattr(settings, 'authoring_backend', 'local_gguf')
    if backend not in {'local_gguf', 'openai_compatible'}:
        raise ValueError('Unknown authoring backend; no implicit replacement')
    text_model = getattr(settings, 'authoring_model', '')
    # A blank vision override explicitly means the same selected service
    # model, not a text-only request: unchanged images still travel with it.
    provider_model = (getattr(settings, 'authoring_vision_model', '') or text_model) if visual else text_model
    known = {'4b': _LOW_HEADROOM_TEXT_MODEL, '7b': _BOUNDED_7B_FIRST, '9b': _BOUNDED_9B_VISUAL_MODEL}
    portable = (model.resolve() != known[model_variant].resolve()
        or root.resolve() != (Path('D:/Models/ai_comic/runtime')).resolve()
        or bool(getattr(settings, 'local_llm_server', '')))
    grammar_dir = _path(settings, 'local_llm_grammar_dir', PROJECT_ROOT / 'config/local_grammars')
    config = replace(config, backend=backend, model_path=model, runtime_root=root,
        evidence_root=_path(settings, 'local_llm_evidence_dir', PROJECT_ROOT / 'work/local_authoring_runtime'),
        server_executable=_path(settings, 'local_llm_server', config.server_executable),
        comfy_url=settings.comfy_url, port=int(getattr(settings, 'local_llm_port', 8191)),
        provider_base_url=getattr(settings, 'authoring_base_url', ''),
        provider_model=provider_model,
        provider_api_key_env=getattr(settings, 'authoring_api_key_env', 'AI_COMIC_AUTHORING_API_KEY'),
        provider_visual=visual, portable_local_model=portable,
        native_grammar_directory=grammar_dir if (
            getattr(settings, 'local_llm_grammar_dir', '') or grammar_dir.is_dir()) else None)
    if backend == 'local_gguf' and visual:
        projector = _path(settings, 'local_llm_projector', model.parent / 'mmproj-F16.gguf')
        def binding(path):
            stat = path.stat()
            return FrozenFileStat(path.resolve(), stat.st_size, stat.st_mtime_ns)
        config = replace(config, mmproj_path=projector)
        if freeze_visual:
            config = replace(config, visual_model_stat=binding(model), mmproj_stat=binding(projector))
    return config


def local_runtime_config(settings, purpose='text') -> RuntimeConfig:
    """Path/installation discovery only; no model reads or service requests."""
    if purpose not in {'text', 'vision', '7b', '9b'}:
        raise ValueError('Unknown runtime discovery purpose')
    return configured_runtime(settings, visual=purpose == 'vision',
        model_variant=purpose if purpose in {'7b', '9b'} else '4b', freeze_visual=False)


def runtime_from_config(config: RuntimeConfig):
    if config.backend == 'openai_compatible':
        from .compatible_authoring_runtime import CompatibleAuthoringRuntime
        return CompatibleAuthoringRuntime(config)
    return LocalAuthoringRuntime(config)


def registered_local_profile(config: RuntimeConfig) -> bool:
    """Only historical paths qualify; this is not a capability/quality claim."""
    return config.backend == 'local_gguf' and not config.portable_local_model
