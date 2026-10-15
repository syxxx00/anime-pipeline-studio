"""Durable programme-made pixels, distinct from generated artwork ancestry.

A record proves which original and construction produced the input pixels. It
does not certify that the construction understood the original or improves art.
"""
from __future__ import annotations

from datetime import UTC, datetime
import hashlib
from io import BytesIO
import json
from pathlib import Path
import struct
from typing import Literal
import uuid

from PIL import Image
from pydantic import BaseModel, Field, model_validator

from .config import Settings


class VisualMaterial(BaseModel):
    model_config = {'extra': 'forbid'}
    id: str = Field(pattern=r'^[0-9a-f]{32}$')
    generation_method: Literal['programme_sketch'] = 'programme_sketch'
    original_text: str = Field(min_length=1, max_length=12000)
    original_sha256: str = Field(pattern=r'^[0-9a-f]{64}$')
    created_at: str
    plan: dict
    file_path: str = Field(max_length=1000)
    record_file: str = Field(max_length=1000)
    png_sha256: str = Field(pattern=r'^[0-9a-f]{64}$')
    pixel_sha256: str = Field(pattern=r'^[0-9a-f]{64}$')
    width: int = Field(ge=1, le=8192)
    height: int = Field(ge=1, le=8192)

    @model_validator(mode='after')
    def original_is_verbatim(self):
        if hashlib.sha256(self.original_text.encode('utf-8')).hexdigest() != self.original_sha256:
            raise ValueError('视觉材料原句校验失败')
        # Only JSON data is saved. Live objects or non-finite numeric values
        # cannot silently become a different construction when reloaded.
        json.dumps(self.plan, ensure_ascii=False, allow_nan=False)
        return self


def material_root(settings: Settings) -> Path:
    return (Path(settings.output_dir).resolve().parent / 'visual_materials').resolve()


def _inspect_png(payload: bytes) -> tuple[int, int, str]:
    with Image.open(BytesIO(payload)) as image:
        if image.format != 'PNG' or getattr(image, 'n_frames', 1) != 1:
            raise ValueError('视觉材料必须是单帧 PNG 原件')
        if image.width * image.height > 40_000_000 or max(image.size) > 8192:
            raise ValueError('视觉材料尺寸过大')
        image.load()
        rgba = image.convert('RGBA')
        digest = hashlib.sha256(struct.pack('>II', *image.size) + b'RGBA\0' + rgba.tobytes()).hexdigest()
        return image.width, image.height, digest


def register_visual_material(png_bytes: bytes, original_text: str, plan: dict,
                             settings: Settings) -> VisualMaterial:
    """Save the exact supplied PNG and its programme construction without a job."""
    width, height, pixel_sha = _inspect_png(png_bytes)
    material_id = uuid.uuid4().hex
    directory = material_root(settings) / material_id
    material = VisualMaterial(
        id=material_id, original_text=original_text,
        original_sha256=hashlib.sha256(original_text.encode('utf-8')).hexdigest(),
        created_at=datetime.now(UTC).isoformat(), plan=plan,
        file_path=str(directory / 'source.png'), record_file=str(directory / 'material.json'),
        png_sha256=hashlib.sha256(png_bytes).hexdigest(), pixel_sha256=pixel_sha,
        width=width, height=height,
    )
    # Each invocation gets a new immutable-by-convention directory, and never
    # replaces a user file or a previous material record.
    directory.mkdir(parents=True, exist_ok=False)
    with Path(material.file_path).open('xb') as handle:
        handle.write(png_bytes)
    with Path(material.record_file).open('x', encoding='utf-8') as handle:
        handle.write(material.model_dump_json(indent=2))
    return material


def load_visual_material(material_id: str, settings: Settings) -> VisualMaterial:
    if len(material_id) != 32 or any(char not in '0123456789abcdef' for char in material_id):
        raise ValueError('视觉材料标识符无效')
    record = material_root(settings) / material_id / 'material.json'
    material = VisualMaterial.model_validate_json(record.read_text(encoding='utf-8'))
    if material.id != material_id:
        raise ValueError('视觉材料记录与标识符不一致')
    verify_visual_material(material, settings)
    return material


def verify_visual_material(material: VisualMaterial, settings: Settings) -> bytes:
    """Verify saved provenance, path confinement, exact PNG and decoded pixels."""
    directory = material_root(settings) / material.id
    source = Path(material.file_path).resolve()
    record = Path(material.record_file).resolve()
    if (source != (directory / 'source.png').resolve()
            or record != (directory / 'material.json').resolve()
            or not source.is_relative_to(material_root(settings))
            or not record.is_relative_to(material_root(settings))):
        raise ValueError('视觉材料来源路径越界或与标识符不一致')
    saved = VisualMaterial.model_validate_json(record.read_text(encoding='utf-8'))
    if saved.model_dump() != material.model_dump():
        raise ValueError('视觉材料记录在登记后变化')
    payload = source.read_bytes()
    if hashlib.sha256(payload).hexdigest() != material.png_sha256:
        raise ValueError('视觉材料 PNG 在登记后变化')
    width, height, pixel_sha = _inspect_png(payload)
    if (width, height, pixel_sha) != (material.width, material.height, material.pixel_sha256):
        raise ValueError('视觉材料尺寸或像素校验失败')
    return payload
