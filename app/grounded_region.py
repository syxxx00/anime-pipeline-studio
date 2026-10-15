"""Bound visual geometry -> inspectable pixel masks, never text-planner guesses."""
from __future__ import annotations

import base64
import hashlib
from io import BytesIO
from typing import Annotated, Literal

from PIL import Image, ImageDraw
from pydantic import BaseModel, Field, model_validator


Pixel = Annotated[int, Field(strict=True, ge=0, le=40000000)]


class RegionLocalizationRequest(BaseModel):
    model_config = {'extra': 'forbid'}
    artifact_id: str = Field(min_length=1, max_length=120, pattern=r'^[A-Za-z0-9][A-Za-z0-9_-]*$')
    instruction: str = Field(min_length=1, max_length=4000)

    @model_validator(mode='after')
    def meaningful(self):
        if not self.instruction.strip():
            raise ValueError('A visual localization requires an explicit modification intent.')
        return self


class PixelRectangle(BaseModel):
    model_config = {'extra': 'forbid'}
    kind: Literal['rectangle']
    x0: Pixel
    y0: Pixel
    x1: Pixel
    y1: Pixel


class PixelPolygon(BaseModel):
    model_config = {'extra': 'forbid'}
    kind: Literal['polygon']
    points: list[tuple[Pixel, Pixel]] = Field(min_length=3, max_length=32)


class RegionLocalization(BaseModel):
    model_config = {'extra': 'forbid'}
    observation_id: str = Field(min_length=1, max_length=80)
    artifact_id: str = Field(min_length=1, max_length=120)
    source_sha256: str = Field(pattern=r'^[0-9a-f]{64}$')
    width: Pixel
    height: Pixel
    status: Literal['located', 'unlocatable']
    regions: list[PixelRectangle | PixelPolygon] = Field(max_length=4)
    evidence: str = Field(min_length=1, max_length=3000)
    uncertainties: list[str] = Field(max_length=8)

    @model_validator(mode='after')
    def status_and_geometry(self):
        if not self.evidence.strip() or any(not value.strip() for value in self.uncertainties):
            raise ValueError('Localization must explain the visible basis or inability to locate.')
        if self.width < 1 or self.height < 1 or self.width * self.height > 40000000:
            raise ValueError('Invalid original image dimensions.')
        if (self.status == 'located') != bool(self.regions):
            raise ValueError('Located replies need geometry; unlocatable replies must carry no geometry.')
        for shape in self.regions:
            if isinstance(shape, PixelRectangle):
                if not (0 <= shape.x0 < shape.x1 <= self.width and 0 <= shape.y0 < shape.y1 <= self.height):
                    raise ValueError('Rectangle must be nonempty and inside the bound original.')
            else:
                points = shape.points
                if (len(set(points)) != len(points)
                        or any(x >= self.width or y >= self.height for x, y in points)):
                    raise ValueError('Polygon vertices must be distinct pixels inside the bound original.')
                area = sum(x * points[(i + 1) % len(points)][1] - y * points[(i + 1) % len(points)][0]
                           for i, (x, y) in enumerate(points))
                if not area or _self_intersects(points):
                    raise ValueError('Polygon must have area and a simple, nonintersecting boundary.')
        return self


def _self_intersects(points):
    def cross(a, b, c):
        return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
    def on(a, b, c):
        return (min(a[0], b[0]) <= c[0] <= max(a[0], b[0])
                and min(a[1], b[1]) <= c[1] <= max(a[1], b[1]))
    for i, a in enumerate(points):
        b = points[(i + 1) % len(points)]
        for j in range(i + 1, len(points)):
            if j == i + 1 or (i == 0 and j == len(points) - 1):
                continue
            c, d = points[j], points[(j + 1) % len(points)]
            ab_c, ab_d, cd_a, cd_b = cross(a, b, c), cross(a, b, d), cross(c, d, a), cross(c, d, b)
            if ((ab_c * ab_d < 0 and cd_a * cd_b < 0)
                    or any(value == 0 and on(start, end, point) for value, start, end, point in (
                        (ab_c, a, b, c), (ab_d, a, b, d), (cd_a, c, d, a), (cd_b, c, d, b)))):
                return True
    return False


LOCALIZATION_SYSTEM = """你收到一张绑定原图及一次局部修改意图。只根据这张实际原图定位需要改变的
可见区域，返回至多四个矩形或简单多边形，使用原图像素坐标，不返回PNG/base64或质量评分。
原点在左上，x向右，y向下；矩形[x0,x1)×[y0,y1)的右下界不含，允许x1=width/y1=height；
多边形3至32个不重复顶点，不重复起点，所有顶点必须落在原图像素内，边不可自交。
圈选范围应回应修改意图；解释你从实际像素看到的定位依据并保留不确定项。
看不清、对象不存在、意图不足以可靠定位或需要猜测时返回status=unlocatable和空regions，说明原因。
原句与意图只说明要寻找什么，不证明图中存在；不得凭文字描述猜坐标，不决定生成或认证效果。
逐值回传给定observation_id、artifact_id、source_sha256、width、height。只返回schema规定JSON。"""


def localization_schema(observation_id, artifact_id, metadata):
    schema = RegionLocalization.model_json_schema()
    bound = {'observation_id': observation_id, 'artifact_id': artifact_id,
             'source_sha256': metadata['sha256'], 'width': metadata['width'], 'height': metadata['height']}
    for key, value in bound.items():
        schema['properties'][key]['enum'] = [value]
    return schema


def bind_localization(raw, observation_id, artifact_id, metadata):
    result = RegionLocalization.model_validate(raw)
    if (result.observation_id != observation_id or result.artifact_id != artifact_id
            or result.source_sha256 != metadata['sha256']
            or (result.width, result.height) != (metadata['width'], metadata['height'])):
        raise ValueError('Visual localization does not match the current observation/artifact/SHA/dimensions.')
    return result


def rasterize_localization(result: RegionLocalization, original: bytes):
    if result.status != 'located':
        raise ValueError('An unlocatable result cannot authorize a mask or edit.')
    if hashlib.sha256(original).hexdigest() != result.source_sha256:
        raise ValueError('Original bytes changed before localization rasterization.')
    with Image.open(BytesIO(original)) as picture:
        if (picture.format != 'PNG' or picture.size != (result.width, result.height)
                or getattr(picture, 'n_frames', 1) != 1):
            raise ValueError('Localization must use the exact bound, single-frame original PNG.')
        picture.load()
        overlay = picture.convert('RGBA')
    mask = Image.new('L', overlay.size, 0)
    draw = ImageDraw.Draw(mask)
    for shape in result.regions:
        if isinstance(shape, PixelRectangle):
            draw.rectangle((shape.x0, shape.y0, shape.x1 - 1, shape.y1 - 1), fill=255)
        else:
            draw.polygon(shape.points, fill=255)
    if not mask.getbbox():
        raise ValueError('Visual geometry produced an empty mask.')
    # Visible color overlay is evidence for inspection, never substituted as input pixels.
    tint = Image.new('RGBA', overlay.size, (255, 40, 90, 0))
    tint.putalpha(mask.point(lambda value: 96 if value else 0))
    overlay = Image.alpha_composite(overlay, tint)
    mask_buffer, overlay_buffer = BytesIO(), BytesIO()
    mask.save(mask_buffer, format='PNG')
    overlay.save(overlay_buffer, format='PNG')
    return mask_buffer.getvalue(), overlay_buffer.getvalue()


def png_data_url(payload):
    return 'data:image/png;base64,' + base64.b64encode(payload).decode('ascii')
