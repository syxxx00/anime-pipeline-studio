"""文字清理检测器：调用引擎 venv 子进程跑漫画文字检测（scripts/text_detect.py）。

检测器在 AI 图常见的伪文字（笔画像字但不是字）上比印刷体 OCR 敏感得多；
检测不到文字时返回空结果，调用方据此跳过整个文字清理阶段，管线零开销。
"""
from __future__ import annotations

import asyncio
import json
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

from .config import Settings

DETECT_LOCK = asyncio.Lock()


@dataclass(slots=True)
class TextDetection:
    count: int
    boxes: list[list[int]] = field(default_factory=list)
    scores: list[float] = field(default_factory=list)
    mask: str | None = None
    # 角色身上的文字框：按人物框保护规则从清理中排除（设定内容不能擦），只报告
    excluded: list[dict] = field(default_factory=list)


def _run_detect(settings: Settings, image_path: Path, mask_path: Path,
                exclude_boxes: list[list[float]] | None = None) -> TextDetection:
    script = Path(__file__).resolve().parents[1] / "scripts" / "text_detect.py"
    payload: dict = {"image": str(image_path), "mask_out": str(mask_path)}
    if exclude_boxes:
        payload["exclude_boxes"] = exclude_boxes
    completed = subprocess.run(
        [settings.semantic_python, str(script)],
        input=json.dumps(payload, ensure_ascii=False),
        text=True, encoding="utf-8", capture_output=True, timeout=300, check=True,
    )
    raw = json.loads(completed.stdout.strip().splitlines()[-1])
    return TextDetection(
        count=int(raw.get("count", 0)),
        boxes=raw.get("boxes", []),
        scores=raw.get("scores", []),
        mask=raw.get("mask"),
        excluded=raw.get("excluded", []),
    )


async def detect_text(settings: Settings, image_path: Path, mask_path: Path,
                      exclude_boxes: list[list[float]] | None = None) -> TextDetection:
    """检测图片中的文字区域；GPU 检测与生成共用显卡，串行以防争抢。"""
    async with DETECT_LOCK:
        return await asyncio.to_thread(_run_detect, settings, image_path, mask_path, exclude_boxes)
