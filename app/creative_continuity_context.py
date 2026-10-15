"""Freeze one selected work's saved conditions and feedback for a next choice.

This is an input-identity boundary, not an observer, planner or quality verdict.
It does not dispatch, classify reviews, edit historical jobs, or choose a model.
Callers must persist the returned snapshot, then revalidate it after awaited
observation/choice work and immediately before any successor submission.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict
import hashlib
from io import BytesIO
import json
import os
from pathlib import Path

from PIL import Image

from .region_edit import generation_ancestor
from .schemas import GenerateRequest
from .prose_picture import (ProsePictureChoice, prose_picture_event,
    prose_language_scope, apply_prose_language)
from .creative_request_understanding import (SOURCE_VIEW_VERSIONS, SOURCE_READING_SCOPE,
    _project_request_reading_material)


class StaleContinuationSourceError(ValueError):
    """The selected work, its conditions or original feedback changed."""


def _digest(value: dict) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                    separators=(',', ':'), allow_nan=False).encode('utf-8')).hexdigest()


def _original(text: str, digest: str) -> None:
    if not isinstance(text, str) or not text or hashlib.sha256(text.encode('utf-8')).hexdigest() != digest:
        raise ValueError('创作原句与 SHA 不一致；未用改写文本继续。')


def _validate_request_original(request: dict, text: str, digest: str) -> None:
    # Validate without round-tripping the saved request through defaults: the
    # exact historical negative/native/LoRA/expert fields remain in the snapshot.
    parsed = GenerateRequest.model_validate(request)
    intent = parsed.creation_intent
    if intent is None or intent.original_text != text or intent.original_sha256 != digest:
        raise ValueError('选中作品的生成条件不能核对当前原句；未借另一候选的条件。')
    for container, key in (('semantic_spec', 'source_sentence'), ('scene_contract', 'source_text')):
        value = (request.get(container) or {}).get(key)
        if value is not None and value != text:
            raise ValueError('选中作品的生成上下文含不同原句：' + container)
    proposal = request.get('author_proposal') or {}
    if proposal.get('original_sha256') not in (None, digest):
        raise ValueError('选中作品作者条件的原句 SHA 不一致。')


def _raw_words(reviews) -> list[dict]:
    if not isinstance(reviews, (list, tuple)):
        raise ValueError('评价需要保留完整原文记录，不能用摘要替换。')
    values = []
    for review in reviews:
        if not isinstance(review, dict) or not isinstance(review.get('text'), str):
            raise ValueError('评价原文缺失；未把分类或摘要当成用户原话。')
        value = {key: deepcopy(review[key]) for key in
                 ('id', 'text', 'created_at', 'artwork_id') if key in review}
        # Explicit validation instructions are not human preferences. Preserve
        # their attribution into selection/comparison inputs and signatures;
        # legacy human records without this marker keep their exact old shape.
        if str(review.get('origin', '')).startswith('assistant_validation'):
            value.update({key: deepcopy(review[key]) for key in
                          ('origin', 'scope', 'action_request_id') if key in review})
        values.append(value)
    return values


def _artwork_identity(artwork: dict) -> dict:
    # Reviews have their own verbatim signature. Detector checks and derived
    # review analysis are not new user words and need not invalidate a choice.
    return {key: deepcopy(value) for key, value in artwork.items() if key not in {'reviews', 'checks'}}


def _image(artwork, job, output_root: Path) -> dict:
    path = Path(artwork.file_path)
    if (not artwork.file_path or not path.is_absolute() or path.is_symlink()
            or not path.resolve().is_relative_to(output_root.resolve()) or not path.is_file()):
        raise ValueError('选中作品原图不存在或不在作品输出目录内。')
    source = path.resolve()
    # A derivative without its own job can retain the nearest real ancestor's
    # conditions, but its current pixels must never be confused with that output.
    if artwork.job_id == job.id:
        index = artwork.image_index
        if (type(index) is not int or index < 0 or index >= len(job.files)
                or Path(job.files[index]).resolve() != source):
            raise ValueError('选中作品路径不是其确切生成任务的对应图片。')
    try:
        before = source.stat()
        if not 0 < before.st_size <= 128 * 1024 * 1024:
            raise ValueError('选中原图为空或超过读取上限；未缩小或替换原件。')
        with source.open('rb') as stream:
            opened = os.fstat(stream.fileno())
            data = stream.read(128 * 1024 * 1024 + 1)
            after_read = os.fstat(stream.fileno())
        after = source.stat()
        identity = lambda value: (value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns)
        if len(data) != before.st_size or not identity(before) == identity(opened) == identity(after_read) == identity(after):
            raise StaleContinuationSourceError('选中原图在读取中变化；请重新观察。')
        with Image.open(BytesIO(data)) as image:
            if image.width * image.height > 40_000_000:
                raise ValueError('选中原图超出像素读取上限；未替换为缩略图。')
            width, height, format_name = image.width, image.height, image.format
            image.verify()
    except (OSError, SyntaxError, Image.DecompressionBombError) as exc:
        raise ValueError('选中作品原图不能完整读取。') from exc
    return {'file_path': str(source), 'sha256': hashlib.sha256(data).hexdigest(),
            'bytes': len(data), 'width': width, 'height': height, 'format': format_name}


def freeze_continuation_source(artwork_id: str, artworks, jobs, *, original_text: str,
                               original_sha256: str, output_root: Path,
                               session_feedback=()) -> dict:
    """Return detached saved inputs for this work, never a session's last draft.

    An unbound retained intermediate stage is deliberately unsupported here: its
    containing job's final request must not masquerade as stage-specific conditions.
    Existing stage tools and historical artifacts are not changed by this boundary.
"""
    _original(original_text, original_sha256)
    artwork = artworks.get(artwork_id)
    if artwork is None or artwork.id != artwork_id:
        raise ValueError('选中作品不存在或身份不一致。')
    if artwork.source_stage is not None:
        raise ValueError('保留阶段图需要其实际阶段条件；未把最终图条件当成阶段条件。')
    job = generation_ancestor(artwork, artworks, jobs)
    if job is None or job.status != 'completed':
        raise ValueError('选中作品缺少已完成的真实生成任务条件。')
    ancestry, current, seen = [], artwork, set()
    while current is not None and current.id not in seen:
        if current.source_stage is not None:
            raise ValueError('祖先是保留阶段图，实际阶段条件尚未绑定；未借最终图条件。')
        seen.add(current.id)
        ancestry.append({'artwork_id': current.id, 'job_id': current.job_id,
                         'parent_artwork_id': current.parent_artwork_id})
        if current.job_id == job.id:
            break
        current = artworks.get(current.parent_artwork_id) if current.parent_artwork_id else None
    if not ancestry or ancestry[-1]['job_id'] != job.id:
        raise ValueError('选中作品与实际生成任务的祖先关系无法核对。')
    raw_request = deepcopy(job.request)
    _validate_request_original(raw_request, original_text, original_sha256)
    saved_artwork = deepcopy(asdict(artwork))
    feedback = {'artwork_reviews': deepcopy(artwork.reviews),
                'session_feedback': deepcopy(list(session_feedback))}
    words = {'artwork_id': artwork.id, 'artwork_reviews': _raw_words(feedback['artwork_reviews']),
             'session_feedback': _raw_words(feedback['session_feedback'])}
    feedback['original_words_signature'] = _digest(words)
    snapshot = {'schema_version': 1, 'original_text': original_text,
        'original_sha256': original_sha256, 'artwork': saved_artwork,
        'condition_ancestry': ancestry,
        'condition_scope': 'own_completed_job' if artwork.job_id == job.id else 'ancestor_conditions_for_derivative',
        'source_job': {'id': job.id, 'status': job.status, 'request': raw_request,
                       'request_sha256': _digest(raw_request)},
        'image': _image(artwork, job, Path(output_root)), 'feedback': feedback,
        'request_scope': 'saved_generation_request_not_independent_workflow_verification',
        'execution_workflow_alignment_verified': False,
        'semantic_understanding_verified': False, 'quality_verified': False}
    snapshot['snapshot_sha256'] = _digest(snapshot)
    # Re-fetch mutable stores after pixel I/O. No asynchronous call happened,
    # but review threads/file edits may still have changed the bound inputs.
    current = artworks.get(artwork_id)
    current_job = generation_ancestor(current, artworks, jobs) if current is not None else None
    current_words = {'artwork_id': artwork_id, 'artwork_reviews': _raw_words(current.reviews),
                     'session_feedback': _raw_words(session_feedback)} if current else None
    if (current is None or current_job is None or current_job.id != job.id
            or current_job.status != 'completed' or _digest(current_job.request) != snapshot['source_job']['request_sha256']
            or _artwork_identity(asdict(current)) != _artwork_identity(saved_artwork)
            or current_words is None or _digest(current_words) != feedback['original_words_signature']):
        raise StaleContinuationSourceError('源条件或评价在保存中变化；未使用部分旧条件。')
    return snapshot


def validate_continuation_source(snapshot: dict, artworks, jobs, *, original_text: str,
                                 original_sha256: str, output_root: Path,
                                 session_feedback=()) -> dict:
    """Refuse stale observation/choice inputs before a successor is submitted.

The caller still owns pause, revision, budget, model compatibility and dispatch
protection. This function grants no authority and does not certify artwork.
"""
    try:
        saved = deepcopy(snapshot)
        expected = saved.pop('snapshot_sha256')
        if saved['schema_version'] != 1 or _digest(saved) != expected:
            raise StaleContinuationSourceError('后继源快照校验失败；未派发。')
        if saved['original_text'] != original_text or saved['original_sha256'] != original_sha256:
            raise StaleContinuationSourceError('后继源快照不是当前原句；未派发。')
        current = freeze_continuation_source(saved['artwork']['id'], artworks, jobs,
            original_text=original_text, original_sha256=original_sha256,
            output_root=output_root, session_feedback=session_feedback)
        comparisons = (
            (_artwork_identity(saved['artwork']), _artwork_identity(current['artwork'])),
            (saved['condition_ancestry'], current['condition_ancestry']),
            (saved['condition_scope'], current['condition_scope']),
            (saved['source_job'], current['source_job']), (saved['image'], current['image']),
            (saved['feedback']['original_words_signature'], current['feedback']['original_words_signature']))
        if any(previous != latest for previous, latest in comparisons):
            raise StaleContinuationSourceError('选中原图、确切条件或评价原话已变化；请重新观察和选择。')
    except (KeyError, TypeError) as exc:
        raise StaleContinuationSourceError('后继源快照不完整；未派发。') from exc
    return current


def _host_prose_choice(proposal: dict) -> dict | None:
    """Recognize the service's exact prose transport, not inferred relations.

    Unknown/legacy author conditions retain their complete native consumption.
    The receipt only proves representation identity; it does not certify the
    author's meaning or the picture. Translated narrative remains its own raw
    model return and must match the event actually compiled by the service.
    """
    receipt = proposal.get('prose_picture_receipt') or {}
    if (not isinstance(receipt, dict)
            or receipt.get('kind') != 'host_prose_picture_event_transport'
            or receipt.get('scene_narrative_consumption') != 'verbatim_complete_description'
            or proposal.get('canonical_picture_choice')):
        return None
    try:
        raw = ProsePictureChoice.model_validate(proposal.get('prose_picture_choice'))
        effective = ProsePictureChoice.model_validate(
            proposal.get('translated_prose_picture_choice') or raw.model_dump(mode='json'))
        event = prose_picture_event(effective).model_dump(mode='json')
        if 'language_application' in receipt:
            if (receipt['language_application'] != prose_language_scope(raw)
                    or apply_prose_language(raw, effective.model_dump(mode='json')) != effective):
                return None
        elif raw.count != effective.count:
            return None  # Older quantity-only transport did not translate count.
        if (receipt.get('raw_choice_sha256') != _digest(raw.model_dump(mode='json'))
                or receipt.get('raw_count') != raw.count
                or receipt.get('native_count') != event['visual_spec']['count']
                or proposal.get('picture_event') != event):
            return None
        return effective.model_dump(mode='json')
    except (ValueError, TypeError):
        return None


def continuation_model_material(snapshot: dict, *, for_model_call: bool = False,
                                regenerate_authoring: bool = False,
                                successor_selection: bool = False,
                                current_creation_intent: dict | None = None,
                                current_request_reading_material: dict | None = None) -> dict:
    """Purpose-bound view of one host snapshot, without I/O or text rewriting.

    The full snapshot stays authoritative in the creative-session host record.
    This view is NOT another frozen snapshot and cannot replace revalidation
    against live works, requests, pixels and reviews after an awaited call.
    Known mechanical representations are referenced rather than repeated;
    actual request strings/parameters and unknown request/author fields survive.
    The call view sends only the common binding/full-request reference rather
    than a mechanical reference catalogue. The default view keeps that catalogue
    for host lookup. Neither view rewrites the saved request or its intent.
    A fresh service-compiled regeneration author does not consume the completed
    parent's native text, artistic choices or compilation exclusion policy;
    its full request remains host-bound, not shortened. Selected resources,
    original words and effective parameters still participate in the new task.
    """
    if type(for_model_call) is not bool:
        raise ValueError('后继模型材料的用途标志必须是明确布尔值。')
    if type(regenerate_authoring) is not bool or (regenerate_authoring and not for_model_call):
        raise ValueError('重新构图材料仅用于明确的新作者模型调用。')
    if type(successor_selection) is not bool or (successor_selection and not for_model_call):
        raise ValueError('后继选择材料仅用于明确的决策调用。')
    if current_creation_intent is not None and (not isinstance(current_creation_intent, dict)
            or not for_model_call or not (regenerate_authoring or successor_selection)):
        raise ValueError('当前要求理解只用于明确的新构思或后继决策输入。')
    if current_request_reading_material is not None:
        if (not isinstance(current_request_reading_material, dict) or current_creation_intent is None
                or not for_model_call or not (regenerate_authoring or successor_selection)):
            raise ValueError('当前Source用途视图需要同源意图及明确的新构思或后继决策用途。')
        version = current_request_reading_material.get('model_view_version')
        if version is not None and version not in SOURCE_VIEW_VERSIONS:
            raise ValueError('未知当前Source用途视图；未猜测降级或据此删掉历史材料。')
    if not isinstance(snapshot, dict):
        raise ValueError('后继模型材料需要完整源快照，不能用摘要代替。')
    saved = deepcopy(snapshot)
    try:
        signature = saved.pop('snapshot_sha256')
        if (type(saved['schema_version']) is not int or saved['schema_version'] != 1
                or not isinstance(signature, str) or _digest(saved) != signature):
            raise ValueError('后继模型材料的完整源快照校验失败。')
        text, original_sha = saved['original_text'], saved['original_sha256']
        _original(text, original_sha)
        if current_creation_intent is not None:
            _original(current_creation_intent['original_text'], current_creation_intent['original_sha256'])
            if (current_creation_intent['original_text'] != text
                    or current_creation_intent['original_sha256'] != original_sha):
                raise ValueError('当前要求理解不属于同一完整原句，未合并不同任务的解释。')
        job, artwork, image, feedback = (saved[key] for key in
                                         ('source_job', 'artwork', 'image', 'feedback'))
        if (not isinstance(job['id'], str) or not job['id'] or job['status'] != 'completed'
                or not isinstance(artwork['id'], str) or not artwork['id']
                or not isinstance(image['sha256'], str) or len(image['sha256']) != 64
                or any(char not in '0123456789abcdef' for char in image['sha256'])):
            raise ValueError('后继模型材料缺少确切来源任务、作品或原图 SHA。')
        request = job['request']
        if not isinstance(request, dict) or _digest(request) != job['request_sha256']:
            raise ValueError('后继模型材料的完整生成请求 SHA 不一致。')
        _validate_request_original(request, text, original_sha)
        words = {'artwork_id': artwork['id'],
            'artwork_reviews': _raw_words(feedback['artwork_reviews']),
            'session_feedback': _raw_words(feedback['session_feedback'])}
        if _digest(words) != feedback['original_words_signature']:
            raise ValueError('后继模型材料的评价原话签名不一致。')
        if saved['condition_scope'] not in {'own_completed_job', 'ancestor_conditions_for_derivative'}:
            raise ValueError('后继模型材料的生成条件所属范围不明确。')
    except (KeyError, TypeError, AttributeError) as exc:
        raise ValueError('后继模型材料的源快照不完整；未猜补条件。') from exc

    root = '/continuation_sources/' + signature

    def reference(path, value):
        return {'path': root + path, 'value_sha256': _digest(value)}

    refs = {'binding': {'host_document': 'creative_session_record', 'snapshot_sha256': signature,
            'hash_scope': 'canonical_json_utf8_not_file_bytes', 'sent_as_model_content': False},
            'full_snapshot': reference('', snapshot),
            'full_request': reference('/source_job/request', request),
            'full_feedback': reference('/feedback', feedback),
            'full_artwork_metadata': reference('/artwork', artwork),
            'moved_request_fields': {}, 'moved_author_fields': {},
            'nonselected_native_channels': {}, 'selected_native_traces': {}}
    # Copy the saved dictionary, NOT model_dump: validation must not inject
    # modern defaults or silently lose historical/future unknown fields.
    actual = deepcopy(request)
    for key in ('author_proposal', 'native_prompts', 'semantic_spec', 'scene_contract'):
        if key in actual:
            refs['moved_request_fields'][key] = reference('/source_job/request/' + key, actual.pop(key))

    parsed = GenerateRequest.model_validate(request)
    # These four fields are inputs to bridge_sampling.source_request, and the
    # source/anima texts also participate in check style classification. Do not
    # treat them as generally redundant. Read-only checks leave the generation
    # graph unchanged, but their style classification can consume these texts.
    # Additional/custom pixel stages deliberately keep the full native view.
    explicit_routes = {'newbie': 'newbie', 'illustrious': 'illustrious',
                       'anima': 'anima_turbo', 'z_image_turbo': 'z_image_turbo'}
    single_standard = (request.get('render_pipeline') in explicit_routes
        and request.get('model_id') == explicit_routes[request['render_pipeline']]
        and type(request.get('check_enabled')) is bool
        and parsed.operation in {'generate', 'iteration_generate'}
        and not any((parsed.highres_enabled, parsed.face_detail_enabled,
            parsed.hand_repair_enabled, parsed.text_clean_enabled,
            parsed.regional_conditioning, parsed.image_handoff, parsed.mask_file,
            parsed.repair_source_file, parsed.repair_instruction, parsed.repair_context_prompt,
            parsed.repair_source_sha256, parsed.repair_mask_sha256, parsed.repair_notes)))
    inactive_source = {}
    # checks._style_text_blob reads these native texts only without a scene
    # contract. A saved nonempty contract gives the common report target, so
    # read-only checks do not turn unused bridge channels into pixel inputs.
    if single_standard and (not parsed.check_enabled or parsed.scene_contract):
        for key in ('anima_prompt', 'source_prompt', 'source_negative_prompt', 'source_parameters'):
            if actual.get(key):
                inactive_source[key] = reference('/source_job/request/' + key, actual.pop(key))
    channel = {'newbie': 'newbie', 'illustrious': 'illustrious',
               'anima_turbo': 'anima', 'z_image_turbo': 'z_image'}[parsed.model_id.value]
    native = request.get('native_prompts')
    selected_metadata = None
    if native is None or native == {}:
        channel = None
        native_scope = 'legacy_saved_prompt_and_negative_no_native_channel_declared'
    else:
        selected = native.get(channel) if isinstance(native, dict) else None
        if (not isinstance(selected, dict)
                or not isinstance(selected.get('positive'), str)
                or not isinstance(selected.get('negative'), str)
                or selected['positive'] != request['prompt']
                or selected['negative'] != request.get('negative_prompt', '')
                or selected.get('trace_status') == 'historical_projection_after_native_override'):
            raise ValueError('所选实际模型的完整原生正负条件与保存请求不一致；未取另一通道或改写。')
        native_scope = 'selected_native_exactly_matches_saved_prompt_and_negative_not_execution_verification'
        selected_metadata = deepcopy(selected)
        for key in ('positive', 'negative', 'trace', 'compiled_trace'):
            selected_metadata.pop(key, None)
        refs['selected_native_original'] = reference('/source_job/request/native_prompts/' + channel, selected)
        for key in ('trace', 'compiled_trace'):
            if key in selected:
                refs['selected_native_traces'][key] = reference(
                    '/source_job/request/native_prompts/' + channel + '/' + key, selected[key])
        for name, value in native.items():
            if name != channel:
                escaped = name.replace('~', '~0').replace('/', '~1')
                refs['nonselected_native_channels'][name] = reference(
                    '/source_job/request/native_prompts/' + escaped, value)

    proposal = request.get('author_proposal')
    author = deepcopy(proposal) if proposal is not None else None
    # Default consumers retain non-native reasoning, uncertainties, goals/conflicts,
    # knowledge, identity, full recipes, positions/layout and unknown author fields.
    # These exact known serialized projections remain available in the host.
    mechanical_author_fields = ('runtime_evidence', 'spec', 'picture_event',
        'canonical_picture_choice', 'translated_canonical_picture_choice',
        'picture_projection', 'prose_picture_choice', 'translated_prose_picture_choice',
        'prose_picture_receipt', 'scene_narrative', 'historical_semantic_spec',
        'historical_scene_contract')
    if author is not None:
        for key in mechanical_author_fields:
            if key in author:
                refs['moved_author_fields'][key] = reference(
                    '/source_job/request/author_proposal/' + key, author.pop(key))
    overridden = (proposal or {}).get('condition_status') == 'historical_after_native_override'
    contract = request.get('scene_contract') or {}
    contract_notes = {key: deepcopy(contract[key]) for key in ('assumptions', 'issues') if key in contract}

    material = deepcopy(saved)
    material['snapshot_sha256'] = signature
    material['source_job'].pop('request')
    material['artwork'] = _artwork_identity(artwork)
    material['feedback'] = {'artwork_reviews': words['artwork_reviews'],
        'session_feedback': words['session_feedback'],
        'original_words_signature': feedback['original_words_signature'],
        'scope': 'verbatim_words_ids_times_artwork_links_only_not_derived_review_analysis'}
    material.update(actual_generation_conditions=actual,
        selected_native_channel=channel, selected_native_scope=native_scope,
        selected_native_metadata_unverified=selected_metadata,
        author_context_unverified=author,
        author_context_status='historical_after_native_override_not_current_conditions' if overridden
            else 'source_author_context_unverified_not_actual_native_conditions',
        source_contract_notes_unverified=contract_notes,
        host_references=({'binding': deepcopy(refs['binding']),
                          'full_request': deepcopy(refs['full_request'])} if for_model_call else refs),
        inactive_source_fields=inactive_source,
        inactive_source_scope='only_removed_without_matching_standard_generator_or_report_text_consumers_not_semantic_deletion',
        material_scope='purpose_bound_model_view_not_full_host_snapshot_or_current_pixel_validation',
        full_saved_request_sent_to_model=False,
        live_source_revalidation_required=True)
    declared_prose = _host_prose_choice(proposal or {})
    canonical_parent = bool((proposal or {}).get('canonical_picture_choice')
        and ((proposal or {}).get('picture_projection') or {}).get('kind')
            == 'host_mechanical_projection_of_canonical_author_choice')
    # Only a known, automatically compiled, single-stage parent can
    # take this view. User native overrides, legacy/unknown inputs and pixel
    # consumers keep the complete native material above. The new author's
    # complete choice is compiled separately with the current full settings.
    if ((regenerate_authoring or successor_selection) and single_standard and not overridden
            and not set(request).difference(GenerateRequest.model_fields)
            and (canonical_parent or declared_prose is not None)
            and (selected_metadata or {}).get('trace_status') == 'compiled_payload_presence_only'):
        source_config = (author or {}).get('selected_config')
        if (isinstance(source_config, dict)
                and all(key in actual and actual[key] == value for key, value in source_config.items())):
            author.pop('selected_config')  # Exact values already in effective parent parameters.
        for key in ('prompt', 'negative_prompt', 'highres_scale', 'highres_steps', 'highres_denoise',
                    'face_detail_steps', 'face_detail_denoise', 'hand_detail_steps', 'hand_detail_denoise',
                    'text_clean_steps', 'text_clean_denoise', 'region_repair_steps', 'region_repair_denoise',
                    'hybrid_denoise'):
            actual.pop(key, None)
        material.pop('inactive_source_fields')  # Full host request already binds these non-consumers.
        material['actual_generation_conditions_scope'] = 'effective_parent_parameters_not_new_generation_request'
        material['parent_native_scope'] = 'full_parent_native_in_host_request_not_consumed_by_fresh_regenerate_author'
        material['material_scope'] = 'regenerate_authoring_original_facts_feedback_and_author_context_not_parent_native_replay'
        if regenerate_authoring and not successor_selection:
            # These known fields are choices made for the completed parent,
            # not current knowledge, selected recipes or quoted user conditions.
            # Every original value remains resolvable under the existing
            # full_request reference: author_proposal/<key> or
            # native_prompts/<selected channel>/<key>. This changes this
            # consumer's inputs, not the saved request or its original intent.
            for key in ('visual_logic', 'uncertainties', 'positions', 'layout'):
                author.pop(key, None)
            for key in ('negative_policy', 'compiled_negative_policy'):
                selected_metadata.pop(key, None)
            material['material_scope'] = 'regenerate_authoring_original_facts_feedback_resources_and_effective_parameters_parent_choices_host_bound'
        # The caller explicitly sends this identical source reading once at
        # creation_intent. Its duplicate in the known compiled parent is not
        # another historical judgement. Different/unknown readings remain.
        reading = author.get('request_understanding')
        parent_intent = actual.get('creation_intent')
        parent_model_intent = deepcopy(parent_intent)
        if (isinstance(parent_model_intent, dict)
                and parent_model_intent.get('interpretation_method') == 'owned_runtime_source_reading_unverified'):
            # The current reader consumer view excludes unquoted explanations.
            # Match that known view without rewriting the full saved parent.
            parent_model_intent['interpretation_notes'] = []
        known_parent_reader = (current_creation_intent is not None
                and current_creation_intent.get('interpretation_method') == 'owned_runtime_source_reading_unverified'
                and isinstance(reading, dict)
                and reading.get('scope') == SOURCE_READING_SCOPE
                and reading.get('creation_intent') == parent_intent)
        matched_view = None
        if known_parent_reader and current_request_reading_material is not None:
            # A lightweight original-only intent cannot identify which reader
            # hypotheses were sent. Compare the WHOLE known projection and its
            # host-material digest; different/unknown parent readings survive.
            known_keys = {'creation_intent', 'candidate_request', 'unattributed_model_reading',
                          'scope', 'authority', 'call_index', 'understanding_verified'}
            if set(reading) == known_keys:
                current_block = {'creation_intent': current_creation_intent,
                    'request_reading_material': current_request_reading_material}
                for view_version in SOURCE_VIEW_VERSIONS:
                    try:
                        projected = _project_request_reading_material(reading, version=view_version)
                    except (ValueError, TypeError):
                        break  # Unknown historical producer, not a repair target.
                    projected_scope = deepcopy(projected)
                    projected_intent = projected_scope.pop('creation_intent')
                    if _digest(current_block) == _digest({'creation_intent': projected_intent,
                            'request_reading_material': projected_scope}):
                        matched_view = view_version
                        break
        legacy_identical = (current_request_reading_material is None and known_parent_reader
            and (parent_intent == current_creation_intent or parent_model_intent == current_creation_intent))
        if matched_view is not None or legacy_identical:
            actual.pop('creation_intent')
            reading.pop('creation_intent')
            if matched_view is not None:
                # These exact known Source fields are already represented once
                # in the caller's whole view. Nothing is removed from the host,
                # native strings, differing historical readings or unknown data.
                for key in ('candidate_request', 'unattributed_model_reading', 'scope',
                            'authority', 'call_index', 'understanding_verified'):
                    reading.pop(key, None)
                source_reference = {'model_field': 'request_reading_material',
                    'value_sha256': _digest(current_request_reading_material), 'view_version': matched_view,
                    'scope': 'same_known_source_projection_sent_once_not_semantic_verification'}
                reading['request_reading_material_reference'] = deepcopy(source_reference)
                material['parent_request_reading_material_reference'] = source_reference
            elif current_creation_intent.get('interpretation_notes') == []:
                reading.pop('unattributed_model_reading', None)
                reading['reader_explanations'] = 'Retained in the host source-reading original; not user painting requirements or approvals.'
            reference_once = {'model_field': 'creation_intent',
                'value_sha256': _digest(current_creation_intent),
                'scope': ('known_source_reading_consumer_view_sent_once_original_and_reader_notes_retained_in_host'
                    if parent_intent != current_creation_intent else
                    'identical_source_reading_sent_once_by_caller_not_a_summary_or_new_interpretation')}
            reading['creation_intent_reference'] = deepcopy(reference_once)
            material['parent_creation_intent_reference'] = reference_once
        if successor_selection:
            declared = deepcopy(declared_prose if declared_prose is not None else
                (proposal.get('translated_canonical_picture_choice') or proposal['canonical_picture_choice']))
            material['declared_parent_picture_unverified'] = declared
            if declared_prose is not None:
                material['declared_parent_picture_scope'] = 'complete_prose_not_structurally_extracted_not_pixel_verification'
            # The selector already receives this complete declared picture.
            # Remove only proven identical copies in the author display view;
            # retain differing/unknown author choices and all host originals.
            if isinstance(declared, dict):
                for key in ('visual_logic', 'uncertainties'):
                    if key in author and key in declared and author[key] == declared[key]:
                        author.pop(key)
                expected_positions = ({'character': declared.get('character_position')}
                                      if declared_prose is None else {})
                spec = declared.get('visual_spec')
                objects = spec.get('objects', []) if isinstance(spec, dict) else None
                if isinstance(objects, list) and all(isinstance(obj, dict)
                        and isinstance(obj.get('id'), str) and isinstance(obj.get('position'), list)
                        and all(isinstance(value, str) for value in obj['position']) for obj in objects):
                    expected_positions.update({obj['id']: '; '.join(obj['position'])
                                               for obj in objects if obj['position']})
                    if author.get('positions') == expected_positions:
                        author.pop('positions')
            material['parent_native_scope'] = 'full_parent_native_host_bound_not_pixels_or_new_method_request'
            material['material_scope'] = 'successor_selection_complete_declared_picture_and_parameters_separate_from_pixels'
    return material
