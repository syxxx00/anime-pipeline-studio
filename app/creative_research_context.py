"""Bounded historical artistic evidence, not observations or route rankings.

Only complete named sections from existing local reports are selected. No
inference, image reads, review synthesis, source edits or import-time I/O.
"""
from __future__ import annotations

import hashlib
from copy import deepcopy
from dataclasses import asdict, is_dataclass
import json
from pathlib import Path
import re


MAX_SECTIONS = 4
MAX_ARTISTIC_READS = 2
MAX_REPORT_BYTES = 2 * 1024 * 1024
REPORTS_ROOT = Path(__file__).resolve().parents[1] / 'docs/research/reports'

_REPORTS = {
    'single_hops': ('2026-09-20/实验分析与循环迭代.md',
        '三场景（礁石、W与起爆器、Q版与茶杯），每场景三个seed；本轮输入、采样与源图条件。',
        '直接生成／单跳对照，不是多轮自主迭代；612张含重复条件，未验证严格角色知识或LoRA。'),
    'loops': ('2026-09-26/系统分析.md',
        '三场景、三个seed、1024×1024，固定提示与参数，最多三次整图交接；目标配置不同。',
        '共享源图和前缀，不是684个独立试验；没有关联的网页评价，不证明通用路线或角色准确。'),
    'stages': ('../../EXISTING_EVIDENCE_2026-09-05.md',
        '既有原图与保存阶段复核；特定作品、配方和seed的局部比较。',
        '报告转述的聊天认可不是已保存评分；个例画法价值不等于完整场景、动作或身份合格。'),
}

# The positive cases and their counterexamples travel together. No model is
# permanently assigned an artistic role, and none of these are default routes.
_SELECTIONS = {
    'initial_model_choice': (
        ('single_hops', '礁石'),
        ('single_hops', 'Q 版与茶杯'),
        ('stages', 'f17e29aa90fe'),
        ('stages', '6134483cec18'),
    ),
    'successor_choice': (
        ('single_hops', '5.2 低强度保留错误，高强度未必对准问题'),
        ('single_hops', '5.3 交接方向与源图状态都重要'),
    ),
    'iterated_successor_choice': (
        ('single_hops', '5.3 交接方向与源图状态都重要'),
        ('loops', '4.1 礁石：对累积纹理和画风漂移最敏感'),
    ),
}


def _complete_section(text: str, title: str) -> tuple[str, int, int, str]:
    lines = text.splitlines(keepends=True)
    headings = []
    for index, line in enumerate(lines):
        match = re.match(r'^(#{1,6})\s+(.+?)\s*$', line)
        if match:
            name = re.sub(r'\[([^\]]+)\]\([^)]*\)', r'\1', match[2]).strip()
            headings.append((index, len(match[1]), name))
    matches = [heading for heading in headings if heading[2] == title]
    if len(matches) != 1:
        raise ValueError('选中章节缺失或不唯一：' + title)
    start, level, _ = matches[0]
    end = next((index for index, depth, _ in headings if index > start and depth <= level), len(lines))
    selected = ''.join(lines[start:end])
    if not ''.join(lines[start + 1:end]).strip():
        raise ValueError('选中章节没有正文：' + title)
    return selected, start + 1, end, lines[start].rstrip('\r\n')


def load_creative_research_context(purpose: str, *, reports_root: Path | str | None = None) -> dict:
    """Load at most four complete relevant sections, never truncate source text.

    Conditions/limits are host reading aids grounded in the report's actual
    scope, not new results. Source text, path and hashes remain available for
    dispute. Missing/changed headings are reported, never replaced by summaries.
    """
    if purpose not in _SELECTIONS:
        raise ValueError('研究材料仅用于起点、首个后继或已有比较的后继选择。')
    if len(_SELECTIONS[purpose]) > MAX_SECTIONS:
        raise ValueError('选段数量超过本入口上限；没有截断原文。')
    root = Path(reports_root or REPORTS_ROOT).resolve()
    packet = {'purpose': purpose,
        'scope': 'historical_local_reports_not_current_pixel_observation_or_user_approval',
        'sections': [], 'issues': [], 'max_sections': MAX_SECTIONS,
        'saved_user_reviews_included': False,
        'limitations': [
            '材料用于形成有理由的候选；不是模型排名、固定专家职责、唯一最佳路线或发布默认。',
            '画法／局部收益、相对退化较少与完成当前构想是不同判断；必须看本次真实作品。',
            '报告中的聊天认可不转为保存评分，检测通过或像素指标不代替关系、身份与质量判断。',
        ]}
    documents = {}
    for report, title in _SELECTIONS[purpose]:
        relative, condition, limitation = _REPORTS[report]
        path = (root / relative).resolve()
        if report not in documents:
            try:
                with path.open('rb') as stream:
                    raw = stream.read(MAX_REPORT_BYTES + 1)
                if len(raw) > MAX_REPORT_BYTES:
                    raise ValueError('报告超出完整读取上限，未截断')
                documents[report] = (raw.decode('utf-8-sig'), hashlib.sha256(raw).hexdigest())
            except (OSError, UnicodeError, ValueError) as exc:
                documents[report] = None
                packet['issues'].append(f'{path}: {type(exc).__name__}: {exc}')
        document = documents[report]
        if document is None:
            continue
        try:
            text, start, end, heading = _complete_section(document[0], title)
        except ValueError as exc:
            packet['issues'].append(f'{path}: {exc}')
            continue
        packet['sections'].append({'source': str(path), 'source_sha256': document[1],
            'section': title, 'section_id': hashlib.sha256((relative + '\0' + title).encode()).hexdigest()[:16],
            'heading': heading, 'start_line': start, 'end_line': end, 'text': text,
            'text_sha256': hashlib.sha256(text.encode('utf-8')).hexdigest(),
            'conditions': [condition], 'limits': [limitation],
            'metadata_scope': 'host_reading_aids_from_report_scope_not_new_empirical_conclusions',
            'evidence_kind': 'historical_report_not_current_pixel_observation'})
    return packet


# These are recorded values, not today's schema defaults. Full requests remain
# in host_material; unused compiled targets and compiler receipts do not travel
# to the picture author alongside the actual native text a second time.
_ART_PARAMETER_FIELDS = (
    'model_id', 'render_pipeline', 'pipeline_version', 'parameter_policy', 'mode',
    'width', 'height', 'batch_size', 'steps', 'cfg', 'seed', 'sampler', 'scheduler',
    'shift', 'cfg_rescale', 'weight_dtype', 'lora_name', 'lora_strength',
    'source_parameters', 'style_preset', 'composition_style_preset',
    'character_style_preset', 'hybrid_denoise', 'operation', 'image_handoff',
    'highres_enabled', 'highres_scale', 'highres_steps', 'highres_denoise',
    'face_detail_enabled', 'face_detail_steps', 'face_detail_denoise',
    'hand_repair_enabled', 'hand_detail_steps', 'hand_detail_denoise',
    'text_clean_enabled', 'text_clean_steps', 'text_clean_denoise',
    'mask_file', 'repair_source_file', 'repair_source_sha256', 'repair_mask_sha256',
    'repair_context_policy', 'region_repair_steps', 'region_repair_denoise',
)
_ART_TEXT_FIELDS = (
    'prompt', 'negative_prompt', 'use_default_negative_prompt', 'source_prompt',
    'source_negative_prompt', 'anima_prompt', 'regional_conditioning',
    'repair_instruction', 'repair_context_prompt',
)
_ART_LIMITATIONS = (
    '材料是当前用途的原件线索，不是全局偏好、总榜、固定模型职责或可复制的画面模板。',
    '评论原话的优点与问题及来源一起保留；局部喜欢不代表整图、角色身份或当前任务合格。',
    '文字匹配只说明取材来路，不认证语义相关性、方法适用性或美术收益。',
    '本入口没有看图；路径可达、工作流文本和执行状态不能代替本次真实像素观察。',
)


def _art_digest(value: object) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
        separators=(',', ':')).encode('utf-8')).hexdigest()


def _art_record(value) -> dict:
    if isinstance(value, dict):
        return deepcopy(value)
    if is_dataclass(value):
        return deepcopy(asdict(value))
    raise TypeError('作品与任务材料必须来自已记录的字典或dataclass。')


def _art_literal_match(text: str, term: str) -> bool:
    # In particular, the character W must not match "NewBie", a drive letter,
    # or a filename. No name alias or semantic similarity is invented here.
    if re.fullmatch(r'[A-Za-z0-9_ ]+', term):
        return re.search(r'(?<![A-Za-z0-9_])' + re.escape(term)
            + r'(?![A-Za-z0-9_])', text, re.IGNORECASE) is not None
    return term.casefold() in text.casefold()


def _art_feedback_ids(feedback) -> list[str]:
    entries = feedback if isinstance(feedback, (list, tuple)) else [feedback]
    return [entry['artwork_id'] for entry in entries if isinstance(entry, dict)
        and isinstance(entry.get('artwork_id'), str) and entry['artwork_id']]


def _art_saved_workflow(artwork: dict, job: dict, jobs) -> tuple[dict | None, list[str]]:
    """Read corresponding saved evidence only; never reconstruct a graph."""
    path = Path(artwork['file_path']) if artwork.get('file_path') else None
    if artwork.get('source_stage'):
        stages = getattr(jobs, 'stage_images', None)
        if stages is not None:
            try:
                matches = [item for item in stages.retained(job['id'])
                    if item.get('id') == artwork['source_stage'].get('id')
                    and path is not None and Path(item.get('file_path', '')).resolve() == path.resolve()]
                if len(matches) == 1 and isinstance(matches[0].get('workflow'), dict):
                    record = deepcopy(matches[0])
                    return {'source': 'retained_stage_record', 'scope': 'retained_stage_only_not_final_pipeline',
                        'workflow': record['workflow'], 'stage_record': record,
                        'workflow_sha256': _art_digest(record['workflow']), 'execution_verified': False}, []
            except (OSError, ValueError, KeyError, TypeError) as exc:
                return None, ['阶段原件无法核对：' + str(exc)]
        return None, ['保存阶段没有可核对的同阶段工作流；没有借用最终请求作为该阶段条件。']
    index = artwork.get('image_index')
    files = job.get('files') or []
    if (path is None or type(index) is not int or not 0 <= index < len(files)
            or Path(files[index]).resolve() != path.resolve()):
        return None, ['作品路径未与该任务输出索引对应；没有借用其他作品或父图的条件。']
    base = path.with_name(path.stem.rsplit('__', 1)[0])
    candidates = (base.with_name(base.name + '__workflow.json'), path.parent / 'workflow.json')
    for candidate in candidates:
        if not candidate.is_file():
            continue
        try:
            with candidate.open('rb') as stream:
                raw = stream.read(MAX_REPORT_BYTES + 1)
            if len(raw) > MAX_REPORT_BYTES:
                raise ValueError('工作流超出完整读取上限；未截断或重建')
            graph = json.loads(raw.decode('utf-8-sig'))
            if not isinstance(graph, dict):
                raise ValueError('工作流原件不是JSON对象')
            return {'source': str(candidate.resolve()),
                'scope': 'saved_workflow_record_not_independent_execution_verification',
                'workflow': graph, 'source_sha256': hashlib.sha256(raw).hexdigest(),
                'workflow_sha256': _art_digest(graph), 'execution_verified': False}, []
        except (OSError, UnicodeError, ValueError) as exc:
            return None, ['已保存工作流不可读，未以重建图替代：' + str(exc)]
    return None, ['没有保存工作流；请求文字与参数不认证实际编码或当次执行。']


def _art_workflow_conditions(evidence: dict) -> dict:
    text_inputs, nodes = [], []
    by_text = {}

    def read(graph, stage):
        stages = graph.get('stages')
        if isinstance(stages, dict):
            for stage_id, stage_graph in stages.items():
                if isinstance(stage_graph, dict):
                    read(stage_graph, [*stage, str(stage_id)])
        for node_id, node in graph.items():
            if not isinstance(node, dict) or not isinstance(node.get('class_type'), str):
                continue
            kind, inputs = node['class_type'], node.get('inputs') or {}
            if not isinstance(inputs, dict):
                continue
            if 'TextEncode' in kind:
                for field in ('text', 'text_g', 'text_l'):
                    if field in inputs and isinstance(inputs[field], str):
                        text = inputs[field]
                        if text not in by_text:
                            by_text[text] = len(text_inputs)
                            text_inputs.append({'text': text, 'bindings': []})
                        text_inputs[by_text[text]]['bindings'].append(
                            {'stage': stage, 'node_id': str(node_id), 'class_type': kind, 'field': field})
            elif any(word in kind for word in ('Sampler', 'Guider', 'Loader', 'ModelSampling',
                    'RescaleCFG', 'Latent', 'VAEEncode', 'Conditioning', 'ImageScale', 'LoadImage')):
                nodes.append({'stage': stage, 'node_id': str(node_id), 'class_type': kind,
                    'inputs': deepcopy(inputs)})

    read(evidence['workflow'], [])
    return {'scope': evidence['scope'], 'workflow_source': evidence['source'],
        'workflow_sha256': evidence['workflow_sha256'], 'encoded_texts': text_inputs,
        'conditioning_and_parameter_nodes': nodes, 'execution_verified': False,
        'semantic_verified': False}


def _art_original(artwork: dict, request: dict) -> dict:
    candidates = (
        ('job.request.semantic_spec.source_sentence', (request.get('semantic_spec') or {}).get('source_sentence')),
        ('job.request.creation_intent.original_text', (request.get('creation_intent') or {}).get('original_text')),
        ('artwork.instruction', artwork.get('instruction')),
    )
    source, text = next(((key, value) for key, value in candidates if isinstance(value, str) and value),
        ('not_recorded', ''))
    return {'text': text, 'source': source, 'scope': 'historical_record_not_current_user_requirement',
        'semantic_verified': False}


def _art_material(artwork: dict, jobs, basis: list[dict]) -> tuple[dict, dict]:
    direct_job = jobs.get(artwork['job_id']) if jobs is not None else None
    job = _art_record(direct_job) if direct_job is not None else None
    request = job.get('request') or {} if job else {}
    evidence, limits = _art_saved_workflow(artwork, job, jobs) if job else (None,
        ['该作品没有对应任务请求；没有用父图请求冒充这张图的条件。'])
    index, files = artwork.get('image_index'), (job.get('files') or []) if job else []
    linked = (job is not None and not artwork.get('source_stage') and type(index) is int
        and 0 <= index < len(files) and artwork.get('file_path')
        and Path(files[index]).resolve() == Path(artwork['file_path']).resolve())
    if evidence:
        conditions = _art_workflow_conditions(evidence)
    elif linked:
        proposal = request.get('author_proposal') or {}
        conditions = {'scope': 'saved_request_only_not_actual_encoded_conditions',
            'request_text': {key: deepcopy(request[key]) for key in _ART_TEXT_FIELDS if key in request},
            'recipe_snapshot': deepcopy(proposal.get('recipe_snapshot')),
            'execution_verified': False, 'semantic_verified': False}
    else:
        conditions = {'scope': 'actual_conditions_unavailable', 'execution_verified': False,
            'semantic_verified': False}
    # Review analyses are classifier output, not original natural-language
    # judgements. Keep them verbatim in the host, not as amplified preferences.
    reviews = []
    for review in artwork.get('reviews') or []:
        if not isinstance(review, dict):
            continue
        original = {key: deepcopy(value) for key, value in review.items() if key != 'analysis'}
        reviews.append({'original': original,
            'origin_scope': ('legacy_unmarked_review_not_new_user_approval' if 'origin' not in review
                else 'explicit_user_review' if review['origin'] == 'user'
                else 'assistant_validation_not_user_review' if review['origin'] == 'assistant_validation'
                else 'unknown_explicit_origin_not_user_review')})
    model = {'artwork': {key: deepcopy(artwork[key]) for key in (
            'id', 'origin', 'job_id', 'image_index', 'parent_artwork_id', 'operation',
            'experiment_id', 'experiment_item_id', 'source_stage') if key in artwork},
        'image': {'file_path': artwork.get('file_path', ''), 'image_url': artwork.get('image_url', ''),
            'file_exists': bool(artwork.get('file_path') and Path(artwork['file_path']).is_file()),
            'pixels_observed_by_this_helper': False},
        'selection_basis': deepcopy(basis), 'recorded_instruction': _art_original(artwork, request),
        'conditions': conditions,
        'recorded_request_parameters': ({key: deepcopy(request[key]) for key in _ART_PARAMETER_FIELDS
            if key in request} if linked else {}),
        'reviews': reviews, 'limits': limits}
    if evidence and artwork.get('source_stage'):
        model['limits'].append('这是阶段产物，不是最终图；最终请求参数仅留宿主，不能解释本阶段画法。')
    host = {'artwork': deepcopy(artwork), 'job': job, 'saved_workflow': evidence,
        'selection_basis': deepcopy(basis)}
    host['binding_sha256'] = _art_digest(host)
    return model, host


def load_artistic_material(original_text: str, *, artworks, jobs, purpose: str,
        character_name: str = '', feedback=None, operation: str = '',
        artwork_ids=(), limit: int = 2) -> dict:
    """Link a few existing works, actual conditions and attributed raw reviews.

    Explicit source/feedback references are exclusive: a local correction does
    not get padded with different pictures. Without references, only literal
    current-text/character matches in the existing index are candidates. This
    helper neither reads PNG pixels nor interprets a review as approval of a
    route, user preference, or complete picture. All original records stay in
    host_material, separate from the selected purpose view sent to the author.
    """
    if not isinstance(original_text, str) or not original_text.strip():
        raise ValueError('艺术材料需要完整当前原句；没有用历史原句替代。')
    if not isinstance(purpose, str) or not purpose.strip():
        raise ValueError('艺术材料需要明确当前用途。')
    if type(limit) is not int or not 1 <= limit <= MAX_SECTIONS:
        raise ValueError('艺术材料数量必须为1到4；没有截断选中原件。')
    if not isinstance(character_name, str) or not isinstance(operation, str):
        raise TypeError('当前角色与操作必须为文本。')
    if isinstance(artwork_ids, str):
        raise TypeError('作品引用须为标识列表，而不是单个字符串。')
    explicit_ids = list(artwork_ids)
    if any(not isinstance(value, str) or not value for value in explicit_ids):
        raise ValueError('作品引用包含无效标识。')
    feedback_ids = _art_feedback_ids(feedback)
    ids = list(dict.fromkeys([*explicit_ids, *feedback_ids]))
    query = {'original_text': original_text, 'character_name': character_name,
        'feedback': deepcopy(feedback), 'operation': operation, 'purpose': purpose,
        'artwork_ids': explicit_ids, 'limit': limit}
    model = {'purpose': purpose, 'scope': 'selected_artistic_evidence_not_template_or_user_approval',
        'current_scope': {'original_sha256': hashlib.sha256(original_text.encode('utf-8')).hexdigest(),
            'character_name': character_name, 'operation': operation},
        'selection_scope': 'explicit_artworks_only' if ids else 'literal_text_matches_only',
        'materials': [], 'issues': [], 'limitations': list(_ART_LIMITATIONS)}
    host = {'query': query, 'selected': [], 'selection_scope': model['selection_scope'],
        'issues': [], 'query_policy': 'existing_index_literal_matches_no_sentiment_ranking'}
    candidates = []
    if ids:
        for identifier in ids[:limit]:
            artwork = artworks.get(identifier) if artworks is not None else None
            if artwork is None:
                model['issues'].append({'artwork_id': identifier, 'issue': 'referenced_artwork_not_found'})
                continue
            candidates.append((_art_record(artwork), [{'kind': 'explicit_artwork_id' if identifier in explicit_ids
                else 'current_feedback_artwork_id', 'value': identifier}]))
        host['unselected_explicit_ids'] = ids[limit:]
    elif artworks is not None:
        # This is a bounded query to an existing index, not a new sample store.
        # The index also searches filenames; filter those incidental matches
        # against actual instruction/review text before selecting any work.
        terms = [('current_original_literal', original_text.strip())]
        if character_name.strip() and character_name.strip() != original_text.strip():
            terms.append(('current_character_literal', character_name.strip()))
        seen = set()
        for kind, term in terms:
            for item in artworks.list(query=term, limit=100):
                meaningful = '\n'.join([str(item.get('instruction') or ''),
                    *[str(review.get('text') or '') for review in item.get('reviews') or []
                        if isinstance(review, dict)]])
                if item['id'] in seen or not _art_literal_match(meaningful, term):
                    continue
                candidates.append((deepcopy(item), [{'kind': kind, 'value': term,
                    'semantic_relevance_verified': False}]))
                seen.add(item['id'])
                if len(candidates) >= limit:
                    break
            if len(candidates) >= limit:
                break
        host['index_query_limit_per_term'] = 100
    for artwork, basis in candidates:
        material, original = _art_material(artwork, jobs, basis)
        model['materials'].append(material)
        host['selected'].append(original)
    host['issues'] = deepcopy(model['issues'])
    return {'model_input': model, 'host_material': host}


def _art_packet_materials(packet_model: dict) -> dict[str, dict]:
    """Validate an already selected snapshot, without re-querying its sources."""
    if (not isinstance(packet_model, dict) or not isinstance(packet_model.get('materials'), list)
            or packet_model.get('version') == 'artistic_material_catalog_v1'):
        raise TypeError('艺术材料读取需要已有完整packet_model，不接收目录或外部查询。')
    if not isinstance(packet_model.get('purpose'), str) or not packet_model['purpose'].strip():
        raise ValueError('艺术材料快照缺少当前用途。')
    materials = {}
    for material in packet_model['materials']:
        artwork = material.get('artwork') if isinstance(material, dict) else None
        identifier = artwork.get('id') if isinstance(artwork, dict) else None
        if not isinstance(identifier, str) or not identifier.strip():
            raise ValueError('艺术材料快照含无效作品标识；未猜补来源。')
        if identifier in materials:
            raise ValueError('艺术材料快照含重复作品标识；无法唯一读取原件。')
        materials[identifier] = material
    return materials


def _art_material_ref(packet_sha256: str, material: dict) -> dict:
    return {'artwork_id': material['artwork']['id'],
        'packet_sha256': packet_sha256, 'material_sha256': _art_digest(material),
        'source': 'selected_complete_packet_model_material',
        'scope': 'readable_historical_record_not_current_scene_authorization'}


def artistic_material_catalog(packet_model: dict) -> dict:
    """Expose lookup and method evidence, not automatic old-scene conditions.

    Full native text, historical instructions and recipe snapshots remain in
    packet_model and its host originals. They can be read for an explicit
    current purpose with read_artistic_materials. This pure consumer view does
    not infer transferable style, rank pictures, look at pixels, read paths or
    change the explicit-source consumer; the owner chooses when to use it.
    """
    materials = _art_packet_materials(packet_model)
    packet_sha256 = _art_digest(packet_model)
    catalog = {key: deepcopy(packet_model[key]) for key in (
        'purpose', 'current_scope', 'selection_scope', 'issues', 'limitations') if key in packet_model}
    catalog.update(version='artistic_material_catalog_v1',
        scope='historical_catalog_and_methods_not_old_scene_conditions_or_approval',
        packet_sha256=packet_sha256, materials=[], max_reads=MAX_ARTISTIC_READS)
    for material in materials.values():
        entry = {key: deepcopy(material[key]) for key in (
            'artwork', 'image', 'selection_basis', 'reviews',
            'recorded_request_parameters', 'limits') if key in material}
        original = material.get('recorded_instruction') or {}
        if isinstance(original, dict):
            entry['recorded_instruction'] = {key: deepcopy(original[key]) for key in (
                'source', 'scope', 'semantic_verified') if key in original}
            text = original.get('text')
            if isinstance(text, str):
                entry['recorded_instruction']['text_sha256'] = hashlib.sha256(text.encode('utf-8')).hexdigest()
        conditions = material.get('conditions') or {}
        if not isinstance(conditions, dict):
            raise ValueError('艺术材料条件不是对象；未把损坏快照当可用证据。')
        entry['conditions'] = {key: deepcopy(conditions[key]) for key in (
            'scope', 'workflow_source', 'workflow_sha256',
            'conditioning_and_parameter_nodes', 'execution_verified', 'semantic_verified') if key in conditions}
        entry['conditions']['consumer_scope'] = 'recorded_methods_only_full_native_and_scene_available_by_purpose_read'
        entry['full_material_ref'] = _art_material_ref(packet_sha256, material)
        catalog['materials'].append(entry)
    return catalog


def read_artistic_materials(packet_model: dict, queries) -> dict:
    """Return complete previously catalogued materials for named purposes.

    Queries are {artwork_id, purpose} objects, at most two, with unique IDs.
    Empty queries are a no-op. No missing ID expands the query, opens a path or
    substitutes newer records. The exact returned data can be frozen with its
    packet/material bindings for recovery. Reading does not approve copying
    historical subjects, scenes, interpretation, exclusions or user preference.
    """
    materials = _art_packet_materials(packet_model)
    if not isinstance(queries, (list, tuple)):
        raise TypeError('按需读取须为明确作品与当前用途的列表。')
    if len(queries) > MAX_ARTISTIC_READS:
        raise ValueError('一次按需读取最多两件已有目录原件；没有截断或扩展查询。')
    seen = set()
    for query in queries:
        if not isinstance(query, dict) or set(query) != {'artwork_id', 'purpose'}:
            raise ValueError('按需读取须且仅须给出artwork_id与purpose。')
        identifier, purpose = query['artwork_id'], query['purpose']
        if not isinstance(identifier, str) or identifier not in materials:
            raise ValueError('按需读取的作品不在已提供目录中；未新增查询。')
        if identifier in seen:
            raise ValueError('同次按需读取不能重复作品；未重复附加旧条件。')
        if not isinstance(purpose, str) or not purpose.strip():
            raise ValueError('按需读取需要非空的当前目的，读取本身不是scene条件授权。')
        seen.add(identifier)
    packet_sha256 = _art_digest(packet_model)
    result = {key: deepcopy(packet_model[key]) for key in (
        'purpose', 'current_scope', 'selection_scope', 'issues', 'limitations') if key in packet_model}
    result.update(version='artistic_material_reads_v1',
        scope='complete_historical_material_read_for_current_purpose_not_scene_conditions_or_approval',
        packet_sha256=packet_sha256, queries=deepcopy(list(queries)), materials=[])
    for query in queries:
        material = materials[query['artwork_id']]
        result['materials'].append({'read_purpose': query['purpose'],
            'full_material_ref': _art_material_ref(packet_sha256, material),
            'material': deepcopy(material)})
    return result

