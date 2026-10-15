from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path, PureWindowsPath


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SETTINGS_PATH = PROJECT_ROOT / "config" / "settings.json"
_INFERENCE_PYTHON = "runtime/ComfyUI/.venv/Scripts/python.exe" if os.name == "nt" else "runtime/ComfyUI/.venv/bin/python"
_PATH_FIELDS = (
    "comfy_root", "engine_base_dir", "comfy_python", "model_root", "style_catalog_path", "output_dir",
    "semantic_model_path", "semantic_python", "local_llm_server", "local_llm_model", "local_llm_projector",
    "local_llm_model_7b", "local_llm_model_9b", "local_llm_runtime_dir", "local_llm_evidence_dir",
    "local_llm_grammar_dir", "concept_root",
)
_MODEL_NAMES = (
    "diffusion_model", "vae", "gemma_clip", "jina_clip", "z_diffusion_model", "z_clip", "z_vae",
    "anima_diffusion_model", "anima_clip", "anima_vae", "illustrious_checkpoint",
)


@dataclass(slots=True)
class Settings:
    web_port: int = 8090
    comfy_url: str = "http://127.0.0.1:8190"
    comfy_root: str = "runtime/ComfyUI"
    engine_base_dir: str = "runtime/ComfyUI"
    comfy_python: str = _INFERENCE_PYTHON
    model_root: str = "models"
    diffusion_model: str = "NewBie-image_diffusion_pytorch_model.safetensors"
    vae: str = "NewBie-image_diffusion_pytorch_model.safetensors"
    gemma_clip: str = "gemma3-4b-it.safetensors"
    jina_clip: str = "jina-clip-v2.safetensors"
    z_diffusion_model: str = "z_image_turbo_bf16.safetensors"
    z_clip: str = "qwen_3_4b.safetensors"
    z_vae: str = "ae.safetensors"
    anima_diffusion_model: str = "anima-turbo-v1.1.safetensors"
    anima_clip: str = "qwen_3_06b_base.safetensors"
    anima_vae: str = "qwen_image_vae.safetensors"
    illustrious_checkpoint: str = "Illustrious-XL-v2.0.safetensors"
    style_catalog_path: str = "config/style_presets.json"
    output_dir: str = "data/jobs"
    auto_start_engine: bool = True
    engine_port: int = 8190
    vram_mode: str = "lowvram"
    llm_base_url: str = ""
    llm_model: str = ""
    llm_api_key_env: str = "CHARACTER_LAB_LLM_API_KEY"
    semantic_model_path: str = "runtime/models/Qwen3-1.7B"
    semantic_python: str = _INFERENCE_PYTHON
    authoring_backend: str = "local_gguf"
    authoring_base_url: str = ""
    authoring_model: str = ""
    authoring_vision_model: str = ""
    authoring_api_key_env: str = "AI_COMIC_AUTHORING_API_KEY"
    local_llm_server: str = ""
    local_llm_model: str = ""
    local_llm_projector: str = ""
    local_llm_model_7b: str = ""
    local_llm_model_9b: str = ""
    local_llm_runtime_dir: str = ""
    local_llm_evidence_dir: str = ""
    local_llm_grammar_dir: str = ""
    local_llm_port: int = 8191
    concept_root: str = ""

    def __post_init__(self) -> None:
        self.resolve_paths()
        for name in ("web_port", "engine_port", "local_llm_port"):
            port = getattr(self, name)
            if type(port) is not int or not 1 <= port <= 65535:
                raise ValueError(f"{name} must be an integer port from 1 to 65535")
        if self.authoring_backend not in {"local_gguf", "openai_compatible"}:
            raise ValueError("authoring_backend must be local_gguf or openai_compatible")
        for name in _MODEL_NAMES:
            value = getattr(self, name)
            if (not isinstance(value, str) or not value.strip() or Path(value).is_absolute()
                    or PureWindowsPath(value).drive or ".." in PureWindowsPath(value).parts
                    or ".." in Path(value).parts):
                raise ValueError(f"{name} must be a model-relative filename inside its model directory")

    def resolve_paths(self) -> "Settings":
        """Resolve configured filesystem paths against the install, never the shell CWD.

        Blank optional language/concept paths remain blank for their consumer's local
        defaults. Model names remain relative Comfy loader enums, not absolute paths.
        """
        for name in _PATH_FIELDS:
            value = getattr(self, name)
            if not isinstance(value, (str, Path)):
                raise ValueError(f"{name} must be a filesystem path string")
            if not str(value):
                continue
            path = Path(value).expanduser()
            setattr(self, name, str((path if path.is_absolute() else PROJECT_ROOT / path).resolve()))
        return self

    @property
    def path_map(self) -> dict[str, Path]:
        root = Path(self.model_root)
        return {
            "comfy_root": Path(self.comfy_root),
            "engine_base_dir": Path(self.engine_base_dir),
            "diffusion_model": root / "unet" / self.diffusion_model,
            "vae": root / "vae" / self.vae,
            "gemma_clip": root / "clip" / self.gemma_clip,
            "jina_clip": root / "clip" / self.jina_clip,
            "z_diffusion_model": root / "diffusion_models" / self.z_diffusion_model,
            "z_clip": root / "text_encoders" / self.z_clip,
            "z_clip_alt": root / "clip" / self.z_clip,
            "z_vae": root / "vae" / self.z_vae,
            "anima_diffusion_model": root / "diffusion_models" / self.anima_diffusion_model,
            "anima_clip": root / "text_encoders" / self.anima_clip,
            "anima_vae": root / "vae" / self.anima_vae,
            "illustrious_checkpoint": root / "checkpoints" / self.illustrious_checkpoint,
            "loras": root / "loras",
        }

    def public_dict(self) -> dict:
        data = asdict(self)
        data["llm_ready"] = bool(self.llm_base_url and self.llm_model)
        data["paths"] = {key: str(path) for key, path in self.path_map.items()}
        return data


def load_settings(path: Path = DEFAULT_SETTINGS_PATH) -> Settings:
    path = Path(path)
    if not path.exists():
        return Settings()
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("settings.json must contain a JSON object")
    valid = Settings.__dataclass_fields__.keys()
    return Settings(**{key: value for key, value in raw.items() if key in valid})
