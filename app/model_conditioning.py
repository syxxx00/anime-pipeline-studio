"""Resolve negative conditioning once, including the distinction between absent and empty.

These are existing local baselines, not validated optimal prompts. An explicit
empty negative is a supported experiment condition and must never be replaced.
"""
from .prompting import DEFAULT_NEGATIVE
from xml.etree import ElementTree as ET


NEGATIVE_BASELINES = {
    'newbie': DEFAULT_NEGATIVE,
    'illustrious': 'worst quality, low quality, bad anatomy, bad hands, extra digits, missing digits, text, watermark, signature, jpeg artifacts',
    'anima': 'worst quality, low quality, score_1, score_2, score_3, artist name, blurry, jpeg artifacts, chromatic aberration',
    'z_image_turbo': 'blurry, ugly, bad anatomy, bad hands, text, watermark',
}


def merge_negative(model: str, baseline: str, constraints: str) -> str:
    if not constraints.strip():
        return baseline
    if model == 'newbie' and '<' in baseline:
        root = ET.fromstring('<negative>' + baseline + '</negative>')
        node = root.find('danbooru_tags')
        if node is None:
            node = ET.SubElement(root, 'danbooru_tags')
        node.text = ', '.join(value for value in (node.text or '', constraints) if value.strip())
        return '\n'.join(ET.tostring(child, encoding='unicode', short_empty_elements=False) for child in root)
    return '\n'.join(value for value in (baseline, constraints) if value.strip())


def negative_for_target(parent: dict, channel: dict, target: str, *, override: str | None = None) -> str:
    if override is not None:
        return override
    # A native channel explicitly owns its negative, even when empty.
    if channel.get('negative') is not None:
        return str(channel['negative'])
    model_id = 'anima_turbo' if target == 'anima' else target
    if parent.get('model_id') == model_id:
        text = str(parent.get('negative_prompt') or '')
        if text.strip() or parent.get('use_default_negative_prompt', True) is False:
            return text
    return NEGATIVE_BASELINES[target]
