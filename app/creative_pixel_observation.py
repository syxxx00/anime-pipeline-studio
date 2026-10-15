"""Pure single-full-PNG observation input, without inference or pixel changes.

The caller binds an actual saved artwork and retains its complete source and
purpose in the host. Mechanical declarations do not certify image meaning.
"""
from __future__ import annotations

from copy import deepcopy
import re

from .creative_result_review import CurrentPixelObservation


PIXEL_OBSERVATION_CONTRACT = 'full_pixel_observation_v1'
PIXEL_OBSERVATION_SYSTEM = """Observe the one actual full PNG supplied with its mechanical binding.
Describe visible spatial relationships, shapes, contours, line and color
groups, contact surfaces, support/attachment, overlap and depth order. State
where a visible relation is unclear. Do not infer an intended scene, desired
edit, named character, lore, emotion, identity, gaze endpoint or artistic
purpose from a label. Original text, Source interpretations, author proposals,
feedback and requested outcomes are not supplied to this consumer.
The binding identifies one full unchanged image, not a crop or mask region.
It establishes pixel provenance, not correctness of your interpretation.
Use the frozen response containers: visible_relationships describes the
actual arrangement; potential_value identifies a visible feature worth later
consideration, not fulfillment of a request. problems describes visibly
grounded shape/contact ambiguities or inconsistencies, not unmet author goals.
uncertain records material limits of this image reading. Separate visible
features from uncertain interpretation; do not fill lists to their limits.
A flat drawing, limited palette, cropped framing, stylization or hidden body
part is not inherently faulty. Do not demand realism, extra props or hidden
fingers. Do not invent exact finger counts or contacts when unreadable.
An empty problems list does not mean that anatomy, likeness, attention, emotion
or quality passed inspection. A single image does not prove comparison,
collection completion or successful execution. Reference, purpose and artistic
decisions belong to a later consumer. Return only the supplied JSON.
This is an unverified reading of visible pixels, not approval or certification."""


def _pixel_observation_schema() -> dict:
    # Same four fields and order as the existing main PixelObservation class;
    # no class/grammar relocation or reduced output contract.
    schema = deepcopy(CurrentPixelObservation.model_json_schema())
    schema['title'] = 'PixelObservation'
    schema.pop('description', None)  # The helper class docstring is not in main's frozen schema.
    return schema


PIXEL_OBSERVATION_SCHEMA = _pixel_observation_schema()


def pixel_observation_input(artwork_id: str, image_sha256: str, width: int,
                            height: int, image_bytes_size: int) -> dict:
    """Declare one full-image binding; no I/O, crop, mask or inferred content.

    The runtime verifies these declarations against actual immutable PNG bytes.
    ASCII ID syntax does not authenticate an artwork's existence or identity;
    the host must supply the exact store ID, not a descriptive source label.
    """
    if (not isinstance(artwork_id, str)
            or re.fullmatch(r'[A-Za-z0-9_-]{1,120}', artwork_id, flags=re.ASCII) is None):
        raise ValueError('像素观察需要有界ASCII机械作品编号，不接受角色描述或原句标签。')
    if (not isinstance(image_sha256, str)
            or re.fullmatch(r'[0-9a-f]{64}', image_sha256, flags=re.ASCII) is None):
        raise ValueError('像素观察需要精确小写SHA256，不猜补图像绑定。')
    if any(type(value) is not int or value <= 0 for value in (width, height, image_bytes_size)):
        raise ValueError('全图尺寸和字节长度须为真实正整数，不猜造尺寸或裁片。')
    return {'contract_version': PIXEL_OBSERVATION_CONTRACT,
        'image_order': 'single full original PNG',
        'image_bindings': [{'index': 1, 'role': 'observed', 'kind': 'full_original_png',
            'artwork_id': artwork_id, 'sha256': image_sha256,
            'width': width, 'height': height, 'bytes': image_bytes_size}],
        'visible_scope': 'one full unchanged original PNG; visible reading only, not purpose success, identity or quality certification'}
