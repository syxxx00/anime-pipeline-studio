"""CPU-only before/after measurements, NOT a noise detector or quality score.

Fixed research matrices remain observational. No metric here can approve,
reject, clean, or repaint an image automatically.
"""
from io import BytesIO
from pathlib import Path
from statistics import fmean

from PIL import Image


def _load(value):
    with Image.open(BytesIO(value) if isinstance(value, bytes) else Path(value)) as im:
        if im.width * im.height > 40_000_000:
            raise ValueError('comparison image exceeds pixel limit')
        size = im.size
        rgb = im.convert('RGB').resize((256, 256), Image.Resampling.LANCZOS)
    def pixels_of(image):
        return list(image.get_flattened_data() if hasattr(image, 'get_flattened_data') else image.getdata())
    pixels = pixels_of(rgb)
    luma = [(r * .2126 + g * .7152 + b * .0722) / 255 for r, g, b in pixels]
    def lap(values):
        return fmean(abs(4 * values[i] - values[i - 1] - values[i + 1] - values[i - 256] - values[i + 256])
                     for y in range(1, 255) for i in range(y * 256 + 1, y * 256 + 255))
    hf = lap(luma)
    chroma = (lap([(r-g)/255 for r, g, b in pixels]) + lap([(b-g)/255 for r, g, b in pixels])) / 2
    low = pixels_of(rgb.resize((64, 64), Image.Resampling.LANCZOS))
    return {'size': size, 'hf': hf, 'chroma_hf': chroma, 'low': low}


def compare_images(candidate, parent, root=None):
    current, previous = _load(candidate), _load(parent)
    origin = _load(root) if root is not None else previous
    def delta(reference):
        regions = []
        for row in range(4):
            for col in range(4):
                indices = [y * 64 + x for y in range(row*16, (row+1)*16) for x in range(col*16, (col+1)*16)]
                difference = fmean(abs(a-b)/255 for i in indices for a, b in zip(current['low'][i], reference['low'][i]))
                regions.append({'region': [col/4, row/4, .25, .25], 'rgb_change': difference})
        return {
            'same_size': current['size'] == reference['size'],
            'hf_ratio': current['hf']/reference['hf'] if reference['hf'] > 1e-8 else None,
            'chroma_hf_ratio': current['chroma_hf']/reference['chroma_hf'] if reference['chroma_hf'] > 1e-8 else None,
            'low_rgb_change': fmean(r['rgb_change'] for r in regions),
            'most_changed_regions': sorted(regions, key=lambda r: r['rgb_change'], reverse=True)[:4],
        }
    return {'version': 1, 'measurement_size': [256, 256], 'low_size': [64, 64],
            'parent': delta(previous), 'root': delta(origin), 'root_available': root is not None,
            'evidence_type': 'pixel_change_only', 'quality_verdict': 'unverified',
            'limitations': ['高频同时包括线条、有效纹理与伪影，不能作为质量分或噪点率',
                           '缩小测量可能漏掉细颗粒；区域变化不证明语义或解剖改变',
                           '不得据此自动接受、拒收或触发重绘']}
