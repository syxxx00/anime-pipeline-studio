"""Bind one internal research PNG and its saved inputs, not a new image source.

No defaults, ranking, source transfer, model calls or copies are created here.
The caller persists this detached snapshot and keeps the current original apart.
Revalidate after awaited work and before using the support in a new submission.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict
import hashlib
from io import BytesIO
import os
from pathlib import Path

from PIL import Image
from pydantic import BaseModel, Field, field_validator

from .creative_continuity_context import _artwork_identity, _digest, _raw_words


class ResearchArtSupport(BaseModel):
    model_config = {'extra': 'forbid', 'strict': True, 'revalidate_instances': 'always'}
    artwork_id: str = Field(min_length=1, max_length=200, pattern=r'^[a-zA-Z0-9_-]+$')
    expected_image_sha256: str = Field(pattern=r'^[0-9a-f]{64}$')
    use_for: str = Field(min_length=1, max_length=2000)

    @field_validator('use_for')
    @classmethod
    def nonblank_use(cls, value):
        if not value.strip():
            raise ValueError('研究支撑用途不能为空；用途原话不作改写。')
        return value


class StaleResearchArtSupportError(ValueError):
    """The selected PNG, saved conditions or original review words changed."""


def _read_png(artwork, job, output_root):
    path = Path(artwork.file_path)
    root = Path(output_root).resolve()
    index = artwork.image_index
    if (not artwork.file_path or not path.is_absolute() or path.is_symlink()
            or not path.resolve().is_relative_to(root) or not path.is_file()
            or type(index) is not int or index < 0 or index >= len(job.files)
            or Path(job.files[index]).resolve() != path.resolve()):
        raise ValueError('研究支撑必须是登记作品自己完成任务的确切输出，且原图在输出目录内。')
    source = path.resolve()
    before = source.stat()
    if not 0 < before.st_size <= 8 * 1024 * 1024:
        raise ValueError('研究原PNG超出单图8MiB上限；未替换缩略图。')
    with source.open('rb') as stream:
        opened = os.fstat(stream.fileno())
        raw = stream.read(8 * 1024 * 1024 + 1)
        after_read = os.fstat(stream.fileno())
    after = source.stat()
    identity = lambda stat: (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns)
    if len(raw) != before.st_size or not identity(before) == identity(opened) == identity(after_read) == identity(after):
        raise StaleResearchArtSupportError('研究原PNG在读取时变化。')
    try:
        with Image.open(BytesIO(raw)) as image:
            if image.format != 'PNG' or getattr(image, 'n_frames', 1) != 1:
                raise ValueError('研究支撑只接受单帧原PNG，不转换格式或抽取动画帧。')
            if image.width * image.height > 16_000_000:
                raise ValueError('研究原PNG超出单图像素上限；未缩小。')
            width, height = image.size
            image.verify()
        with Image.open(BytesIO(raw)) as image:
            image.load()
    except (OSError, SyntaxError, Image.DecompressionBombError) as exc:
        raise ValueError('研究原PNG不能完整解码。') from exc
    return {'file_path': str(source), 'sha256': hashlib.sha256(raw).hexdigest(),
            'bytes': len(raw), 'width': width, 'height': height, 'format': 'PNG',
            'file_identity': {'device': before.st_dev, 'inode': before.st_ino,
                              'mtime_ns': before.st_mtime_ns}}


def _saved_native(request):
    # The builders consume these saved final-target fields. native_prompts and
    # all other stage inputs stay intact in source_job.request, not substituted.
    positive, negative = request.get('prompt'), request.get('negative_prompt')
    policy = request.get('use_default_negative_prompt')
    if not isinstance(positive, str) or not positive.strip() or not isinstance(negative, str):
        raise ValueError('研究源缺少完整保存的原生正负文本；不使用通用默认补造。')
    if policy is not None and type(policy) is not bool:
        raise ValueError('研究源的负面回退条件不明确。')
    if not negative.strip() and policy is not False:
        raise ValueError('研究源负面为空且可能经过默认回退，无法由保存文本核对；不补造。')
    return {'positive': positive, 'negative': negative,
            'positive_source': '/source_job/request/prompt',
            'negative_source': '/source_job/request/negative_prompt',
            'scope': 'saved_final_target_fields_not_independent_workflow_verification',
            'explicit_empty_negative': not negative.strip(),
            'execution_workflow_alignment_verified': False}


def _source_original(request):
    declarations = {}
    for container, field in (('creation_intent', 'original_text'),
                             ('semantic_spec', 'source_sentence'),
                             ('scene_contract', 'source_text')):
        value = request.get(container)
        if value is not None:
            if not isinstance(value, dict):
                raise ValueError('研究源原句容器不是保存的JSON对象。')
            text = value.get(field)
            if text is not None:
                if not isinstance(text, str) or not text.strip():
                    raise ValueError('研究源声明的原句无效；未借用prompt当原句。')
                declarations['/' + container + '/' + field] = text
    texts = list(declarations.values())
    if texts and any(text != texts[0] for text in texts):
        raise ValueError('研究源保存的原句声明互相冲突；未选择性修正旧条件。')
    text = texts[0] if texts else None
    sha = hashlib.sha256(text.encode('utf-8')).hexdigest() if text is not None else None
    for container in ('creation_intent', 'author_proposal'):
        value = request.get(container)
        if value is not None and not isinstance(value, dict):
            raise ValueError('研究源原句SHA容器不是保存的JSON对象。')
        declared_sha = (value or {}).get('original_sha256')
        if text is not None and declared_sha not in (None, sha):
            raise ValueError('研究源原句与保存SHA不一致。')
    return {'available': text is not None, 'original_text': text, 'original_sha256': sha,
            'saved_declarations': declarations,
            'scope': 'historical_source_original_not_current_user_input',
            'missing_was_not_reconstructed': text is None}


def freeze_research_support(selection, artworks, jobs, *, output_root) -> dict:
    """Read exactly one registered work; do not borrow a derivative's ancestor."""
    selection = ResearchArtSupport.model_validate(selection)
    artwork = artworks.get(selection.artwork_id)
    if artwork is None or artwork.id != selection.artwork_id:
        raise ValueError('选中的研究作品不存在或身份不一致。')
    if artwork.source_stage is not None:
        raise ValueError('研究阶段图需要自己的阶段条件；未借用最终图条件。')
    job = jobs.get(artwork.job_id)
    if job is None or job.id != artwork.job_id or job.status != 'completed':
        raise ValueError('研究支撑需有自己已完成的真实任务，不借无任务派生图的祖先条件。')
    if not isinstance(job.request, dict):
        raise ValueError('研究源完整生成条件缺失。')
    raw_request = deepcopy(job.request)
    native = _saved_native(raw_request)
    original = _source_original(raw_request)
    saved_artwork = deepcopy(asdict(artwork))
    reviews = deepcopy(artwork.reviews)
    words = {'artwork_id': artwork.id, 'reviews': _raw_words(reviews)}
    image = _read_png(artwork, job, output_root)
    if image['sha256'] != selection.expected_image_sha256:
        raise StaleResearchArtSupportError('研究原PNG与明确选定的像素SHA不符。')
    snapshot = {'schema_version': 1, 'kind': 'internal_research_art_support',
        'selection': selection.model_dump(mode='json'), 'source_artwork': saved_artwork,
        'source_job': {'id': job.id, 'status': job.status, 'request': raw_request,
                       'request_sha256': _digest(raw_request)},
        'source_original': original, 'source_native': native,
        'source_character_conditions': {key: deepcopy(raw_request[key]) for key in
            ('semantic_spec', 'scene_contract', 'author_proposal', 'lora_name', 'lora_strength')
            if key in raw_request},
        'image': image,
        'feedback': {'artwork_reviews': reviews, 'original_words_signature': _digest(words)},
        'condition_scope': 'own_completed_job',
        'request_scope': 'saved_generation_request_not_independent_workflow_verification',
        'support_scope': 'declared_use_is_correctable_inspiration_not_current_defaults_or_approval',
        'execution_workflow_alignment_verified': False,
        'semantic_understanding_verified': False, 'quality_verified': False,
        'identity_accuracy_verified': False, 'grants_image_handoff': False}
    snapshot['snapshot_sha256'] = _digest(snapshot)
    # Pixel I/O can overlap review/file writes even without an await here.
    current = artworks.get(selection.artwork_id)
    current_job = jobs.get(artwork.job_id)
    if (current is None or current_job is None or current_job.id != job.id
            or current_job.status != 'completed'
            or _artwork_identity(asdict(current)) != _artwork_identity(saved_artwork)
            or _digest(current_job.request) != snapshot['source_job']['request_sha256']
            or _digest({'artwork_id': current.id, 'reviews': _raw_words(current.reviews)})
               != snapshot['feedback']['original_words_signature']):
        raise StaleResearchArtSupportError('研究来源或评价原话在保存中变化。')
    try:
        with Path(image['file_path']).open('rb') as stream:
            # Windows can expose the same timestamp for two rapid same-size
            # writes. Recheck bound bytes, not only metadata; never read beyond
            # the already validated original size plus one sentinel byte.
            final_bytes = stream.read(image['bytes'] + 1)
            opened_stat = os.fstat(stream.fileno())
        final_stat = Path(image['file_path']).stat()
    except OSError as exc:
        raise StaleResearchArtSupportError('研究原PNG在保存完成前消失或不可读。') from exc
    if (len(final_bytes) != image['bytes'] or hashlib.sha256(final_bytes).hexdigest() != image['sha256']
            or final_stat.st_size != image['bytes'] or opened_stat.st_size != image['bytes'] or
            {'device': final_stat.st_dev, 'inode': final_stat.st_ino,
             'mtime_ns': final_stat.st_mtime_ns} != image['file_identity']):
        raise StaleResearchArtSupportError('研究原PNG在保存快照时变化。')
    return snapshot


def validate_research_support(snapshot, artworks, jobs, *, output_root) -> dict:
    """Reject changed support after awaited work; return a detached current receipt."""
    try:
        if not isinstance(snapshot, dict):
            raise StaleResearchArtSupportError('研究支撑快照不是JSON对象。')
        saved = deepcopy(snapshot)
        expected = saved.pop('snapshot_sha256')
        if (saved['schema_version'] != 1 or saved['kind'] != 'internal_research_art_support'
                or _digest(saved) != expected):
            raise StaleResearchArtSupportError('研究支撑快照校验失败。')
        current = freeze_research_support(saved['selection'], artworks, jobs, output_root=output_root)
        pairs = ((_artwork_identity(saved['source_artwork']), _artwork_identity(current['source_artwork'])),
                 (saved['source_job'], current['source_job']), (saved['image'], current['image']),
                 (saved['source_original'], current['source_original']),
                 (saved['source_native'], current['source_native']),
                 (saved['feedback']['original_words_signature'], current['feedback']['original_words_signature']))
        if any(before != after for before, after in pairs):
            raise StaleResearchArtSupportError('研究原图、完整条件或评价原话已变化。')
    except (KeyError, TypeError) as exc:
        raise StaleResearchArtSupportError('研究支撑快照不完整。') from exc
    return current
