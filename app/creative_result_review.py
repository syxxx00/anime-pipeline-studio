"""Prepare two bound originals for an unverified comparison, without inference.

Full conditions and feedback remain in the host's continuation snapshots.
Callers must revalidate those snapshots after awaited review/choice work.
This module never selects an artwork, dispatches a job or certifies a result.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
from io import BytesIO
import os
from pathlib import Path
from typing import Annotated, Literal
import zlib

from PIL import Image
from pydantic import BaseModel, Field, field_validator

from .creative_continuity_context import _digest, _raw_words, _validate_request_original
from .local_authoring_runtime import RuntimeConfig, VisualImageInput
from .region_edit import protected_inputs, verify_region_output
from .schemas import GenerateRequest


class CurrentPixelObservation(BaseModel):
    """Independent shape matching the ordinary PixelObservation consumer."""
    model_config = {'extra': 'forbid', 'strict': True}
    visible_relationships: str = Field(max_length=2000)
    potential_value: str = Field(max_length=1200)
    problems: list[str] = Field(max_length=8)
    uncertain: list[str] = Field(max_length=8)


_Statement = Annotated[str, Field(min_length=1, max_length=1500)]


class ResultComparison(BaseModel):
    model_config = {'extra': 'forbid', 'strict': True}
    current_observation: CurrentPixelObservation
    gains: list[_Statement] = Field(max_length=6)
    losses: list[_Statement] = Field(max_length=6)
    unresolved: list[_Statement] = Field(max_length=6)
    provisional_base: Literal['parent', 'child', 'neither']
    reason: str = Field(min_length=1, max_length=1500)

    @field_validator('reason')
    @classmethod
    def meaningful_reason(cls, value):
        if not value.strip():
            raise ValueError('比较理由不能只有空白；未代写理由。')
        return value


RESULT_COMPARISON_SCHEMA = ResultComparison.model_json_schema()
RESULT_COMPARISON_SYSTEM = """Compare the two actual PNGs: FIRST is parent, SECOND is child.
Use the full original_text, this step's objective and preserve requirements, and
verbatim feedback as context. User words are attributed feedback, not proof of
visible pixels. Describe the child in current_observation. Explain gains, losses
and unresolved questions relative to the parent in natural sentences; no scores.
Separate visible evidence from uncertain interpretation. Do not infer identity,
gaze, emotion or fine carving merely because text requests them. provisional_base
is a tentative parent/child/neither judgment, not user selection or approval.
List bounds are maxima, not quotas; empty lists are valid. Judge changes against
this step's objective, preserve requirements and original intent, not an assumed
need for realism or more detail. Cropping, a limited palette and flat stylization
can be valuable choices rather than defects. Only material, visibly grounded
questions belong in unresolved; do not fill it with irrelevant missing trivia.
The host retains both originals' complete conditions and feedback; this visual
input deliberately excludes old prompts, native text and character explanations
as visible answers. Do not claim those full conditions were sent here. Return
only the supplied JSON. Never repair inputs or self-certify quality."""


LOCAL_RESULT_COMPARISON_SYSTEM = """Compare two exact pixel crops of this local edit:
FIRST is the parent region, SECOND is the child region, using the same original
mask bounding box. The host retains and verifies the full original PNGs; these
two inputs are derived crops, not views of the whole artworks. Describe only
the child crop in current_observation. Compare visible gains, losses and
unresolved questions against this operation's objective and preserve text.
The complete original_text is retained as background, not a requirement that
this local revision fulfill the entire scene or produce another independent
composition. Requested candidate counts belong to the collection, not this ROI.
Feedback is attributed context, not proof of visible pixels. Mechanical outside
mask equality and changed_pixel_bbox establish pixel boundaries, not improvement.
The rectangular crop can include unselected pixels between mask components.
Do not certify whole-image identity, gaze, emotion, composition or quality from
this crop. Do not infer fine detail merely because the instruction requests it.
No scores, automatic approval or user selection. provisional_base is a tentative
parent/child/neither local judgment. List bounds are maxima, not quotas; empty
lists are valid. A limited palette and flat stylization can have value; judge
this actual local purpose, not assumed realism or more detail. State uncertainty
when differences or requested effects are not readable. Return only the supplied
JSON; never repair inputs or self-certify quality."""


LOCAL_VISIBLE_DELTA_CONTRACT = 'local_visible_delta_v1'
LOCAL_VISIBLE_DELTA_SYSTEM = """Observe the two actual exact pixel crops, in the supplied order.
FIRST is the parent region, SECOND is the child region. They use the same
original mask bounding box, without rescaling. Describe only visible changes
between these crops: which lines, shapes, edges, colors, textures or spatial
features are added, removed or changed, and what remains unreadable. Do not
invent a requested edit, desired result, original scene, character identity,
emotion, artist intent or reason for a change. None of those inputs is supplied.
Mechanical bindings and pixel-boundary facts identify these crops; they do not
certify a visual interpretation or whether a change is good. The crop rectangle
may include unselected pixels between mask components. Never describe the full
artwork or infer its quality, gaze, likeness or composition from these crops.
The frozen response schema is reused only as containers: current_observation
describes SECOND as seen; potential_value can describe a visible feature for
later consideration, not usefulness or success. problems contains only visible
ambiguities or inconsistencies, not failed goals. gains means features added
and losses means features removed, NOT improvements and regressions. unresolved
states material limits of this pixel reading. reason explains the observations,
not acceptance or a preferred image. provisional_base is a required legacy
schema placeholder: return neither. Any value in that field has NO authority
to select a reference, keep a child, reject a parent or determine a next action.
Reference and artistic decisions belong to a later purpose-aware consumer.
Do not assume more detail, realism or denser texture is better. Empty lists are
valid; list limits are ceilings, not quotas. Return only the supplied JSON.
This is an unverified visible-delta report, not approval or a quality verdict."""


FULL_VISIBLE_DELTA_CONTRACT = 'full_visible_delta_v1'
FULL_VISIBLE_DELTA_SYSTEM = """Observe the two actual full original PNGs in their supplied order.
FIRST is parent, SECOND is child. Describe the child's visible arrangement and
the visible changes from the parent: positions, body/head directions, shapes,
support/contact, overlap, line/color groups and surfaces, with unreadable parts.
These are complete unchanged images, not mask crops, re-scaled references or
views reconstructed from captions. No original request, desired edit, character
explanation, author proposal or feedback is supplied to this pixel consumer.
Do not infer those purposes, exact gaze endpoints, emotions, identity, biography
or success from labels. Mechanical bindings identify pixels, not their meaning.
Use the frozen containers: current_observation describes SECOND as seen;
potential_value names a visible feature for later consideration, not success;
problems contains only visible ambiguities or inconsistencies. gains means
features added or strengthened and losses means features removed or weakened,
NOT artistic improvement/regression. unresolved states limits of this reading.
reason explains visible differences and uncertainty, not a preferred artwork.
provisional_base is a required legacy placeholder: return neither. That field
has NO authority to choose a reference, adopt a child or determine an operation.
Reference and artistic decisions belong to a later purpose-aware consumer.
A flat drawing, limited palette, crop or hidden body part is not intrinsically
wrong. Do not assume denser texture, more detail or realism is an improvement.
Do not invent exact finger counts, contact or attention when not readable.
Empty lists are valid; bounds are ceilings, not quotas. Return the supplied JSON.
This is a fallible reading of visible pixels, not approval or certification."""


_FULL_BINDING_KEYS = ('index', 'role', 'artwork_id', 'sha256', 'width', 'height',
                      'bytes', 'continuation_snapshot_sha256')


def full_visible_delta_input(data: dict) -> dict:
    """Project known full-pair bindings, retaining task material with the caller.

    No I/O or pixel processing occurs here. prepare_result_pair reads and checks
    the original bytes; the runtime and reference binder revalidate their exact
    image/source identities. Structural bindings do not certify interpretations.
    Existing purpose comparisons are not renamed as neutral returned evidence.
    """
    if (not isinstance(data, dict) or data.get('image_order') != 'first parent; second child'
            or not isinstance(data.get('image_bindings'), list) or len(data['image_bindings']) != 2):
        raise ValueError('Full delta requires two ordered prepared full-original bindings.')
    bindings = []
    for index, (binding, role) in enumerate(zip(data['image_bindings'], ('parent', 'child')), 1):
        if (not isinstance(binding, dict) or set(binding) != set(_FULL_BINDING_KEYS)
                or type(binding['index']) is not int or binding['index'] != index
                or binding['role'] != role or not isinstance(binding['artwork_id'], str)
                or not binding['artwork_id'].strip()):
            raise ValueError('Full delta cannot relabel crops, reorder roles or invent an artwork binding.')
        for key in ('width', 'height', 'bytes'):
            if type(binding[key]) is not int or binding[key] <= 0:
                raise ValueError('Full-original sizes must be exact positive integers.')
        for key in ('sha256', 'continuation_snapshot_sha256'):
            value = binding[key]
            if (not isinstance(value, str) or len(value) != 64
                    or any(char not in '0123456789abcdef' for char in value)):
                raise ValueError('Full originals require exact lower-case SHA bindings.')
        bindings.append(deepcopy(binding))
    if bindings[0]['artwork_id'] == bindings[1]['artwork_id']:
        raise ValueError('Full delta needs two distinct artworks, not the same artwork relabeled.')
    material = {'contract_version': FULL_VISIBLE_DELTA_CONTRACT, 'kind': 'full_original_pair',
        'image_bindings': deepcopy(bindings), 'full_originals_retained_in_host': True,
        'whole_image_quality_verified': False}
    if 'comparison_material' in data and data['comparison_material'] != material:
        raise ValueError('Unknown or changed full-pair material is not upgraded or reconstructed.')
    if 'contract_version' in data and data['contract_version'] != FULL_VISIBLE_DELTA_CONTRACT:
        raise ValueError('Unknown or local delta contract cannot become a full-original reading.')
    return {'contract_version': FULL_VISIBLE_DELTA_CONTRACT,
        'image_order': 'first parent; second child', 'image_bindings': bindings,
        'comparison_material': material,
        'visible_scope': 'two full unchanged original PNGs; unverified visible differences only, not purpose success or reference selection'}


_CROP_BINDING_KEYS = ('index', 'role', 'artwork_id', 'sha256', 'width', 'height', 'bytes',
    'continuation_snapshot_sha256', 'kind', 'full_image_sha256', 'full_image_width',
    'full_image_height', 'full_image_bytes', 'mask_sha256', 'crop_box')


def local_visible_delta_input(data: dict) -> dict:
    """Project a new neutral consumer view without changing the complete host.

    The caller selects/freezes this contract before the new call; this helper
    never upgrades a returned or reserved old comparison. It performs no I/O,
    re-cropping, image transformation, inference or reference selection.
    """
    if not isinstance(data, dict):
        raise ValueError('局部变化视图需要已准备的完整配对材料。')
    material, bindings = data.get('comparison_material'), data.get('image_bindings')
    if (not isinstance(material, dict) or material.get('kind') != 'protected_local_crop_pair'
            or data.get('image_order') != 'first parent crop; second child crop'
            or not isinstance(bindings, list) or len(bindings) != 2
            or _digest(material.get('image_bindings')) != _digest(bindings)):
        raise ValueError('局部变化视图缺少精确有序裁片绑定；未退回全幅或改写标签。')
    if (material.get('coordinates') != 'original_pixel_xyxy_right_bottom_exclusive'
            or material.get('full_originals_retained_in_host') is not True
            or material.get('whole_image_quality_verified') is not False):
        raise ValueError('局部裁片来源或范围声明不一致；未宣称全图已验证。')
    box = material.get('crop_box')
    if (not isinstance(box, list) or len(box) != 4 or any(type(value) is not int for value in box)
            or box[0] < 0 or box[1] < 0 or box[2] <= box[0] or box[3] <= box[1]):
        raise ValueError('裁片边界不是有效的原图整数像素范围。')
    pixel_bindings = []
    for index, (binding, role) in enumerate(zip(bindings, ('parent', 'child')), 1):
        if (not isinstance(binding, dict) or any(key not in binding for key in _CROP_BINDING_KEYS)
                or type(binding['index']) is not int or binding['index'] != index or binding['role'] != role
                or binding['kind'] != 'exact_mask_bbox_crop' or binding['crop_box'] != box
                or binding['mask_sha256'] != material.get('mask_sha256')):
            raise ValueError('局部变化视图的父子顺序、遮罩或裁片身份不一致。')
        for key in ('width', 'height', 'bytes', 'full_image_width', 'full_image_height', 'full_image_bytes'):
            if type(binding[key]) is not int or binding[key] <= 0:
                raise ValueError('裁片绑定的尺寸或字节长度无效。')
        if ((binding['width'], binding['height']) != (box[2] - box[0], box[3] - box[1])
                or box[2] > binding['full_image_width'] or box[3] > binding['full_image_height']):
            raise ValueError('裁片尺寸或边界与原图几何不一致。')
        for key in ('sha256', 'full_image_sha256', 'mask_sha256', 'continuation_snapshot_sha256'):
            value = binding[key]
            if not isinstance(value, str) or len(value) != 64 or any(char not in '0123456789abcdef' for char in value):
                raise ValueError('裁片绑定缺少精确SHA，未以标签替代。')
        pixel_bindings.append({key: deepcopy(binding[key]) for key in _CROP_BINDING_KEYS})
    boundary = material.get('pixel_boundary_verification')
    details = boundary.get('details') if isinstance(boundary, dict) else None
    if (not isinstance(details, dict) or boundary.get('status') not in {'pass', 'warning'}
            or details.get('verification') != 'outside_mask_pixels_verified'
            or details.get('source_sha256') != pixel_bindings[0]['full_image_sha256']
            or details.get('mask_sha256') != material.get('mask_sha256')
            or 'changed_pixel_bbox' not in details):
        raise ValueError('局部变化视图缺少同源像素边界验证；未自证修改有效。')
    changed = details['changed_pixel_bbox']
    if changed is not None and (not isinstance(changed, list) or len(changed) != 4
            or any(type(value) is not int for value in changed)
            or changed[0] < box[0] or changed[1] < box[1] or changed[2] > box[2] or changed[3] > box[3]
            or changed[2] <= changed[0] or changed[3] <= changed[1]):
        raise ValueError('变化像素边界不在已绑定局部范围内。')
    projected = {key: deepcopy(material[key]) for key in ('kind', 'crop_box', 'coordinates',
        'mask_sha256', 'full_originals_retained_in_host', 'whole_image_quality_verified')}
    projected.update(contract_version=LOCAL_VISIBLE_DELTA_CONTRACT, image_bindings=deepcopy(pixel_bindings),
        pixel_boundary_verification={'status': boundary['status'], 'details': {
            key: deepcopy(details[key]) for key in ('verification', 'changed_pixel_bbox', 'source_sha256', 'mask_sha256')}})
    return {'contract_version': LOCAL_VISIBLE_DELTA_CONTRACT,
        'image_order': 'first parent crop; second child crop', 'image_bindings': pixel_bindings,
        'comparison_material': projected,
        'visible_scope': 'two exact original-mask-bbox crops only; rectangle may include unselected pixels; not whole artworks or artistic-purpose evaluation'}


def _checked_snapshot(raw, original_text):
    if not isinstance(raw, dict):
        raise ValueError('结果比较需要完整冻结的 continuation_source 快照。')
    saved = deepcopy(raw)
    try:
        signature = saved.pop('snapshot_sha256')
        digest = hashlib.sha256(original_text.encode('utf-8')).hexdigest()
        if (type(saved['schema_version']) is not int or saved['schema_version'] != 1
                or _digest(saved) != signature or saved['original_text'] != original_text
                or saved['original_sha256'] != digest):
            raise ValueError('结果比较快照或完整原句已变化。')
        job, artwork, feedback = saved['source_job'], saved['artwork'], saved['feedback']
        if (not isinstance(artwork['id'], str) or not artwork['id'] or job['status'] != 'completed'
                or _digest(job['request']) != job['request_sha256']):
            raise ValueError('结果比较缺少确切已完成任务的冻结条件。')
        _validate_request_original(job['request'], original_text, digest)
        words = {'artwork_id': artwork['id'],
                 'artwork_reviews': _raw_words(feedback['artwork_reviews']),
                 'session_feedback': _raw_words(feedback['session_feedback'])}
        if _digest(words) != feedback['original_words_signature']:
            raise ValueError('结果比较的反馈原话签名已变化。')
    except (KeyError, TypeError, AttributeError) as exc:
        raise ValueError('结果比较快照不完整；未猜补条件或反馈。') from exc
    saved['snapshot_sha256'] = signature
    return saved, words


def _stat_identity(value):
    return value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns


def _png_crc(payload):
    if payload[:8] != b'\x89PNG\r\n\x1a\n':
        raise ValueError('比较输入必须是原始 PNG；未转换格式。')
    offset = 8
    while offset + 12 <= len(payload):
        size = int.from_bytes(payload[offset:offset + 4], 'big')
        end = offset + 8 + size
        if end + 4 > len(payload):
            break
        kind = payload[offset + 4:offset + 8]
        crc = int.from_bytes(payload[end:end + 4], 'big')
        if zlib.crc32(payload[offset + 4:end]) & 0xffffffff != crc:
            raise ValueError('原 PNG 的 CRC 校验失败；未修补。')
        offset = end + 4
        if kind == b'IEND':
            if size != 0 or offset != len(payload):
                break
            return
    raise ValueError('原 PNG 不完整或包含尾部数据；未修补。')


def _read_original(snapshot, settings, role, index):
    image, artwork = snapshot['image'], snapshot['artwork']
    raw_path = Path(image['file_path'])
    source = raw_path.resolve()
    limits = RuntimeConfig()
    if (not raw_path.is_absolute() or raw_path.is_symlink() or not source.is_file()
            or not source.is_relative_to(Path(settings.output_dir).resolve())
            or Path(artwork['file_path']).resolve() != source):
        raise ValueError('比较原图路径不存在、越界或与冻结作品不同。')
    before = source.stat()
    if not 0 < before.st_size <= limits.max_image_bytes:
        raise ValueError('比较原图为空或超过标准视觉读取上限；未缩图。')
    with source.open('rb') as stream:
        opened = os.fstat(stream.fileno())
        payload = stream.read(limits.max_image_bytes + 1)
        after_read = os.fstat(stream.fileno())
    after = source.stat()
    if (len(payload) != before.st_size or not
            _stat_identity(before) == _stat_identity(opened) == _stat_identity(after_read) == _stat_identity(after)):
        raise ValueError('比较原图在有界读取期间变化。')
    digest = hashlib.sha256(payload).hexdigest()
    if (type(image['bytes']) is not int or image['bytes'] != len(payload)
            or image['sha256'] != digest or image['format'] != 'PNG'):
        raise ValueError('比较原图字节、SHA 或格式与冻结快照不同。')
    _png_crc(payload)
    try:
        with Image.open(BytesIO(payload)) as picture:
            width, height = picture.size
            if (picture.format != 'PNG' or getattr(picture, 'n_frames', 1) != 1
                    or width * height > limits.max_image_pixels
                    or type(image['width']) is not int or type(image['height']) is not int
                    or (width, height) != (image['width'], image['height'])):
                raise ValueError('比较原图尺寸或单帧边界与冻结快照不一致。')
            picture.verify()
        with Image.open(BytesIO(payload)) as picture:
            picture.load()
    except (OSError, SyntaxError, Image.DecompressionBombError) as exc:
        raise ValueError('比较原 PNG 不能完整解码；未修补或替换。') from exc
    binding = {'index': index, 'role': role, 'artwork_id': artwork['id'], 'sha256': digest,
               'width': width, 'height': height, 'bytes': len(payload),
               'continuation_snapshot_sha256': snapshot['snapshot_sha256']}
    visual = VisualImageInput(image_bytes=payload, expected_sha256=digest, mime_type='image/png',
                              source_ref=f'{role} artwork {artwork["id"]}; sha256={digest}')
    return binding, visual, source, _stat_identity(after)


def prepare_result_pair(parent_snapshot, child_snapshot, settings, *, original_text, objective, preserve):
    """Read FIRST parent / SECOND child originals; no writes or runtime calls."""
    if (not isinstance(original_text, str) or not original_text.strip()
            or not isinstance(objective, str) or not objective.strip() or not isinstance(preserve, str)):
        raise ValueError('比较需要完整原句、明确本次目的及保留要求原文。')
    parent, parent_words = _checked_snapshot(parent_snapshot, original_text)
    child, child_words = _checked_snapshot(child_snapshot, original_text)
    if parent['artwork']['id'] == child['artwork']['id']:
        raise ValueError('父子比较需要两件明确的作品；未重复一图充当后继。')
    first = _read_original(parent, settings, 'parent', 1)
    second = _read_original(child, settings, 'child', 2)
    if any(_stat_identity(item[2].stat()) != item[3] for item in (first, second)):
        raise ValueError('父子原图在配对准备期间变化；请重新冻结。')
    data = {'original_text': original_text,
            'original_sha256': parent['original_sha256'], 'objective': objective, 'preserve': preserve,
            'image_order': 'first parent; second child', 'image_bindings': [first[0], second[0]],
            'parent_feedback': parent_words, 'child_feedback': child_words,
            'feedback_scope': 'verbatim_original_words_and_ids_times_artwork_links_only',
            'conditions_scope': 'complete_conditions_and_feedback_retained_in_host_continuation_snapshots',
            'full_saved_conditions_sent_to_visual_observer': False,
            'comparison_is_unverified_not_user_selection': True}
    return data, (first[1], second[1])


def prepare_local_result_pair(parent_snapshot, child_snapshot, settings, *, original_text, objective, preserve):
    """Derive two exact mask-bbox crops from verified originals, entirely in memory."""
    data, originals = prepare_result_pair(parent_snapshot, child_snapshot, settings,
        original_text=original_text, objective=objective, preserve=preserve)
    request = GenerateRequest.model_validate(child_snapshot['source_job']['request'])
    if (request.operation != 'region_repair'
            or request.parent_artwork_id != parent_snapshot['artwork']['id']
            or request.repair_source_sha256 != originals[0].expected_sha256
            or request.repair_instruction != objective.strip()):
        raise ValueError('局部比较缺少本次真实父图、源SHA或修改目的绑定；未改用全幅比较。')
    source_bytes, mask_bytes = protected_inputs(request, settings)
    if source_bytes != originals[0].image_bytes:
        raise ValueError('局部修改源像素与比较父图不一致；未借用其他原图或遮罩。')
    _png_crc(mask_bytes)
    full_size = (data['image_bindings'][0]['width'], data['image_bindings'][0]['height'])
    try:
        with Image.open(BytesIO(mask_bytes)) as mask:
            if (mask.format != 'PNG' or mask.mode != 'L' or getattr(mask, 'n_frames', 1) != 1
                    or mask.size != full_size):
                raise ValueError('局部比较需要与原图同尺寸的真实规范化单帧遮罩。')
            box = mask.getbbox()
            if box is None:
                raise ValueError('局部比较遮罩为空；未猜测变化区域。')
    except (OSError, SyntaxError, Image.DecompressionBombError) as exc:
        raise ValueError('局部比较遮罩无法完整读取；未重建或替换。') from exc
    # Reuse the execution boundary verifier. Its pass/warning does not judge
    # the quality of the edit or whether it fulfilled the objective.
    boundary = verify_region_output(originals[1].image_bytes, request, settings)
    changed = boundary['details']['changed_pixel_bbox']
    boundary['details']['changed_pixel_bbox'] = list(changed) if changed is not None else None
    limits = RuntimeConfig()
    images, bindings = [], []
    for full, visual in zip(data['image_bindings'], originals):
        with Image.open(BytesIO(visual.image_bytes)) as picture:
            crop = picture.crop(box)
            encoded = BytesIO()
            crop.save(encoded, format='PNG')
            payload = encoded.getvalue()
        if not 0 < len(payload) <= limits.max_image_bytes:
            raise ValueError('局部派生PNG超过标准视觉输入上限；未缩放。')
        digest = hashlib.sha256(payload).hexdigest()
        binding = {**full, 'kind': 'exact_mask_bbox_crop',
            'full_image_sha256': full['sha256'], 'full_image_width': full['width'],
            'full_image_height': full['height'], 'full_image_bytes': full['bytes'],
            'mask_sha256': request.repair_mask_sha256, 'crop_box': list(box),
            'sha256': digest, 'width': crop.width, 'height': crop.height, 'bytes': len(payload)}
        bindings.append(binding)
        images.append(VisualImageInput(image_bytes=payload, expected_sha256=digest, mime_type='image/png',
            source_ref=f'{full["role"]} crop of artwork {full["artwork_id"]}; '
                       f'full_sha256={full["sha256"]}; box={list(box)}; crop_sha256={digest}'))
    material = {'kind': 'protected_local_crop_pair',
        'scope': 'exact_mask_bbox_only_not_whole_image_observation_or_quality',
        'crop_box': list(box), 'coordinates': 'original_pixel_xyxy_right_bottom_exclusive',
        'mask_sha256': request.repair_mask_sha256,
        'pixel_boundary_verification': boundary, 'image_bindings': bindings,
        'full_originals_retained_in_host': True, 'whole_image_quality_verified': False}
    data.update(image_order='first parent crop; second child crop', image_bindings=bindings,
        comparison_material=material,
        current_operation={'operation': request.operation, 'parent_artwork_id': request.parent_artwork_id,
            'scope': 'local_revision_not_an_independent_candidate_or_collection_completion'})
    return data, tuple(images)
