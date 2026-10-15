from __future__ import annotations

import hashlib
import shutil
import subprocess
from pathlib import Path
from typing import Literal

from PIL import Image, ImageColor, ImageDraw, ImageFont
from pydantic import BaseModel, Field


class TextLayerSpec(BaseModel):
    content: str = Field(min_length=1, max_length=500)
    x: float = Field(default=0.5, ge=0, le=1)
    y: float = Field(default=0.85, ge=0, le=1)
    width: float = Field(default=0.7, gt=0.02, le=1)
    height: float = Field(default=0.18, gt=0.02, le=1)
    color: str = Field(default="#ffffff", max_length=40)
    stroke_color: str = Field(default="#111111", max_length=40)
    background: str = Field(default="#00000088", max_length=40)
    align: Literal["left", "center", "right"] = "center"
    font_size: int = Field(default=72, ge=8, le=400)


FONT_CANDIDATES = [Path(r"C:\Windows\Fonts\msyh.ttc"), Path(r"C:\Windows\Fonts\msyhbd.ttc"), Path(r"C:\Windows\Fonts\simhei.ttf")]


def _font(size: int):
    for path in FONT_CANDIDATES:
        if path.exists():
            return ImageFont.truetype(str(path), size=size), str(path)
    return ImageFont.load_default(size=size), "Pillow default"


def render_text_layer(source: Path, target: Path, spec: TextLayerSpec) -> dict:
    image = Image.open(source).convert("RGBA")
    width, height = image.size
    box_w, box_h = max(1, int(width * spec.width)), max(1, int(height * spec.height))
    left = max(0, min(width - box_w, int(width * spec.x - box_w / 2)))
    top = max(0, min(height - box_h, int(height * spec.y - box_h / 2)))
    layer = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)
    right, bottom = left + box_w, top + box_h
    # Pillow rectangle endpoints are inclusive. Keep every painted pixel in
    # the declared half-open [left, right) x [top, bottom) region.
    draw.rounded_rectangle((left, top, right - 1, bottom - 1), radius=max(4, min(box_w, box_h) // 10), fill=ImageColor.getcolor(spec.background, "RGBA"))
    font_size = min(spec.font_size, box_h)
    while font_size > 8:
        font, font_path = _font(font_size)
        bbox = draw.multiline_textbbox((0, 0), spec.content, font=font, stroke_width=max(1, font_size // 28), align=spec.align)
        if bbox[2] - bbox[0] <= box_w * 0.9 and bbox[3] - bbox[1] <= box_h * 0.86:
            break
        font_size -= 2
    x = left + {"left": box_w * 0.05, "center": box_w / 2, "right": box_w * 0.95}[spec.align]
    anchor = {"left": "lm", "center": "mm", "right": "rm"}[spec.align]
    draw.multiline_text((x, top + box_h / 2), spec.content, font=font, fill=ImageColor.getcolor(spec.color, "RGBA"), stroke_width=max(1, font_size // 28), stroke_fill=ImageColor.getcolor(spec.stroke_color, "RGBA"), anchor=anchor, align=spec.align)
    composed = Image.alpha_composite(image, layer).convert("RGB")
    composed.save(target, format="PNG")
    alpha_bbox = layer.getchannel("A").getbbox()
    outside_unchanged = True
    if alpha_bbox:
        source_rgb = image.convert("RGB")
        # Deterministic invariant: pixels outside the declared layer rectangle
        # are byte-identical before and after composition.
        regions = (
            (0, 0, width, top), (0, bottom, width, height),
            (0, top, left, bottom), (right, top, width, bottom),
        )
        outside_unchanged = all(source_rgb.crop(box).tobytes() == composed.crop(box).tobytes() for box in regions if box[2] > box[0] and box[3] > box[1])
    ocr = _optional_ocr(target, spec.content)
    return {
        "operation": "deterministic_text_layer", "content": spec.content,
        "box_pixels": [left, top, box_w, box_h], "font": font_path, "font_size": font_size,
        "verification": "deterministic_source_and_pixel_invariants_verified" if alpha_bbox and outside_unchanged else "pixel_invariant_failed",
        "content_sha256": hashlib.sha256(spec.content.encode("utf-8")).hexdigest(),
        "layer_rgba_sha256": hashlib.sha256(layer.tobytes()).hexdigest(),
        "changed_pixel_bbox": list(alpha_bbox) if alpha_bbox else None,
        "outside_declared_box_unchanged": outside_unchanged,
        "ocr": ocr,
    }


def _optional_ocr(image_path: Path, expected: str) -> dict:
    executable = shutil.which("tesseract")
    if not executable:
        return {"status": "unavailable", "engine": "tesseract", "expected": expected, "observed": ""}
    try:
        completed = subprocess.run(
            [executable, str(image_path), "stdout", "-l", "chi_sim+eng", "--psm", "6"],
            capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=30,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return {"status": "error", "engine": "tesseract", "expected": expected, "observed": "", "error": str(exc)}
    observed = completed.stdout.strip()
    normalize = lambda value: "".join(value.split()).casefold()
    return {
        "status": "pass" if normalize(expected) in normalize(observed) else "warning",
        "engine": "tesseract", "expected": expected, "observed": observed,
        "returncode": completed.returncode,
    }
