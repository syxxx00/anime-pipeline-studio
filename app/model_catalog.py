from __future__ import annotations

from pathlib import Path

from .config import Settings


def _exists_any(*paths: Path) -> bool:
    return any(path.is_file() for path in paths)


def model_catalog(settings: Settings) -> list[dict[str, object]]:
    paths = settings.path_map
    newbie_missing = [name for name, path in settings.path_map.items() if name in {
        "diffusion_model", "vae", "gemma_clip", "jina_clip"
    } and not path.is_file()]
    z_parts = {
        "transformer": paths["z_diffusion_model"],
        "text_encoder": paths["z_clip"],
        "text_encoder_alt": paths["z_clip_alt"],
        "vae": paths["z_vae"],
    }
    z_missing = []
    if not z_parts["transformer"].is_file():
        z_missing.append(settings.z_diffusion_model)
    if not _exists_any(z_parts["text_encoder"], z_parts["text_encoder_alt"]):
        z_missing.append(settings.z_clip)
    if not z_parts["vae"].is_file():
        z_missing.append(settings.z_vae)
    anima_parts = {
        settings.anima_diffusion_model: settings.path_map["anima_diffusion_model"],
        settings.anima_clip: paths["anima_clip"],
        settings.anima_vae: paths["anima_vae"],
    }
    anima_missing = [name for name, path in anima_parts.items() if not path.is_file() or path.stat().st_size < 100 * 1024 * 1024]
    illustrious_path = settings.path_map["illustrious_checkpoint"]
    illustrious_missing = [] if illustrious_path.is_file() and illustrious_path.stat().st_size > 100 * 1024 * 1024 else [settings.illustrious_checkpoint]
    return [
        {
            "id": "newbie",
            "name": "NewBie Exp0.1",
            "role": "可选图像模型",
            "ready": not newbie_missing,
            "missing": newbie_missing,
            "fit": "支持当前 NewBie 原生条件适配；是否采用由具体作品、兼容条件和比较决定",
            "license": "Newbie-NC-1.0（非商业模型许可；以所用权重原许可证为准）",
        },
        {
            "id": "z_image_turbo",
            "name": "Z-Image Turbo",
            "role": "可选图像模型",
            "ready": not z_missing,
            "missing": z_missing,
            "fit": "支持当前 Z-Image 原生条件适配；不预设固定创作职责",
            "license": "Apache-2.0",
        },
        {
            "id": "anima_turbo",
            "name": "Anima Turbo v1.1 2B",
            "role": "可选图像模型",
            "ready": not anima_missing,
            "missing": anima_missing,
            "fit": "支持当前 Anima 原生条件适配；适用范围需结合具体条件和实际结果",
            "license": "模型及衍生模型有非商业限制；输出使用另查所用权重原始条款",
        },
        {
            "id": "illustrious",
            "name": "Illustrious XL v2.0",
            "role": "可选图像模型",
            "ready": not illustrious_missing,
            "missing": illustrious_missing,
            "fit": "支持当前 SDXL／Illustrious 原生条件适配；不预设固定创作职责",
            "license": "CreativeML OpenRAIL-M（Illustrious XL v2.0；其他权重需另核许可）",
            "license_url": "https://huggingface.co/OnomaAIResearch/Illustrious-XL-v2.0",
        },
        {
            "id": "qwen_image",
            "name": "Qwen-Image",
            "role": "高显存/服务器候选",
            "ready": False,
            "missing": ["当前发行未提供 Qwen-Image 工作流适配器"],
            "fit": "未启用的研究候选，不参与当前默认模型选择",
            "license": "Apache-2.0",
        },
    ]
