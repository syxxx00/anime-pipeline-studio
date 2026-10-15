"""Explicit, bounded original-text selections for the creative planner.

Construction does no I/O. ``load`` reads only operator-registered documents,
never a path supplied by a model, and never follows their reference links.
Verbatim text delivery and SHA checks do not certify concept understanding,
pixel observation, creative approval, or the correctness of historical claims.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from hashlib import sha256
import json
from pathlib import Path
import re


_PROJECT_ROOT = Path(__file__).resolve().parents[1]
LEGACY_DOCUMENTS = {
    'project': Path('F:/Models2/concept/concepts/ai-comic/CONCEPT.md'),
    'pipeline': Path('F:/Models2/concept/concepts/ai-comic/pipeline/CONCEPT.md'),
    'use': Path('F:/Models2/concept/concepts/ai-comic/SOURCES_AND_USE.md'),
    'method': Path('F:/Models2/concept/concept-workflow/SKILL.md'),
}
REGISTERED_DOCUMENTS = {
    'project': _PROJECT_ROOT / 'resources/concepts/ai-comic/CONCEPT.md',
    'pipeline': _PROJECT_ROOT / 'resources/concepts/ai-comic/pipeline/CONCEPT.md',
    'use': _PROJECT_ROOT / 'resources/concepts/ai-comic/SOURCES_AND_USE.md',
    'method': _PROJECT_ROOT / 'resources/concepts/concept-workflow/SKILL.md',
}

# Operator-owned reading entry, not a task classifier or an exhaustive concept.
# The complete goal section includes the latest corrections maintained within it.
DEFAULT_SELECTIONS = (
    {'document': 'project', 'section': '最终目标与后续工作的依据'},
    {'document': 'project', 'section': '我对整个项目的理解'},
    {'document': 'project', 'section': '多维理解在本项目中的用途'},
    {'document': 'project', 'section': '以项目情境展开嵌套概念'},
    {'document': 'project', 'section': '项目各部分怎样共同参与创作'},
    {'document': 'project', 'section': '用户纠正怎样改变理解'},
    {'document': 'use', 'section': '读取和使用'},
    {'document': 'method', 'section': None},
)

# These are reading purposes for the current execution stage, not a text-task
# classifier, a token-saving ranking or a boundary on the project's concept.
# Every selected member is still delivered as a complete registered original.
PLANNING_INITIAL_SELECTIONS = (
    {'document': 'project', 'section': '最终目标与后续工作的依据'},
    {'document': 'method', 'section': None},
    {'document': 'pipeline', 'section': '从构想发展表现而不是填满画面'},
    {'document': 'pipeline', 'section': '输入承载当前理解而不冻结理解'},
    {'document': 'pipeline', 'section': '来源延续与解释可纠正是两种责任'},
)
PLANNING_RESULT_SELECTIONS = (
    {'document': 'pipeline', 'section': '实际画面怎样改变下一步'},
    {'document': 'pipeline', 'section': '多模型协作与循环怎样发生'},
    {'document': 'pipeline', 'section': 'Check帮助发展和收束作品'},
)

_PLANNING_SELECTION_REASONS = (
    'Keep the complete project goal and latest corrections as the creative purpose, not a completion certificate.',
    'Keep the complete concept working method; its references are not claimed read or understood.',
    'Develop a coherent expressive picture from a brief idea, rather than fill independent fields or add detail mechanically.',
    'Distinguish the original, current interpretation, native conditions and their correctable artistic choices.',
    'Preserve source continuity without treating an unverified interpretation as permanently correct or already revisable.',
    'Use actual intermediate results to reconsider which relationships and expressions to develop.',
    'Choose collaboration and bounded exploration from the current result, not a permanent model chain or guaranteed recovery.',
    'Check the relationships needed for this decision and distinguish unknown, stopping and successful delivery.',
)


class ConceptContextError(ValueError):
    def __init__(self, code: str, message: str, *, document: str | None = None):
        self.code, self.document = code, document
        super().__init__(message)


def _digest(raw: bytes) -> str:
    return sha256(raw).hexdigest()


def _compact_bytes(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':')).encode('utf-8')


def _signature(stat) -> tuple:
    return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns


def _sections(text: str) -> list[dict]:
    """Index approved documents' ATX headings; fenced code is not a heading."""
    lines = text.splitlines(keepends=True)
    headings, ancestors, occurrences = [], [], {}
    fence = None
    for index, line in enumerate(lines):
        value = line.rstrip('\r\n').lstrip('\ufeff') if index == 0 else line.rstrip('\r\n')
        opening = re.match(r'^ {0,3}(`{3,}|~{3,})(.*)$', value)
        if fence:
            if opening and opening[1][0] == fence[0] and len(opening[1]) >= fence[1] and not opening[2].strip():
                fence = None
            continue
        if opening:
            fence = (opening[1][0], len(opening[1]))
            continue
        heading = re.match(r'^ {0,3}(#{1,6})[ \t]+(.+?)[ \t]*$', value)
        if not heading:
            continue
        level, title = len(heading[1]), re.sub(r'[ \t]+#+[ \t]*$', '', heading[2])
        while ancestors and ancestors[-1]['level'] >= level:
            ancestors.pop()
        path = [item['title'] for item in ancestors] + [title]
        key = (level, tuple(path))
        occurrence = occurrences.get(key, 0) + 1
        occurrences[key] = occurrence
        identifier = 'section_' + _digest(_compact_bytes({'path': path, 'level': level, 'occurrence': occurrence}))[:20]
        item = {'section_id': identifier, 'title': title, 'level': level,
                'heading_path': path, 'start_line': index + 1}
        headings.append(item)
        ancestors.append(item)
    for position, item in enumerate(headings):
        end = next((later['start_line'] - 1 for later in headings[position + 1:]
                    if later['level'] <= item['level']), len(lines))
        raw = ''.join(lines[item['start_line'] - 1:end]).encode('utf-8')
        item.update(end_line=end, text_sha256=_digest(raw), text_bytes=len(raw))
    return headings


class CreationConceptContext:
    """Operator config registers paths; callers select document IDs and sections.

    ``documents`` is trusted application/operator configuration, never model or
    API request data. Model selections cannot modify this registered path map.

    ``load(None)`` uses complete default entry sections. ``load([])`` returns
    only document identities and section catalogues. Explicit selections are
    ``{document, section, document_sha256?}``; section is a catalogue ID or an
    exact unique title, and null explicitly selects the whole registered file.
    The optional full-document SHA rejects selection against a changed version.
    """

    def __init__(self, documents: Mapping[str, str | Path] | None = None, *,
                 max_document_bytes: int = 512 * 1024,
                 max_selected_bytes: int = 64 * 1024,
                 max_context_bytes: int = 128 * 1024):
        configured = REGISTERED_DOCUMENTS if documents is None else documents
        if not isinstance(configured, Mapping) or not configured or set(configured) - set(REGISTERED_DOCUMENTS):
            raise ConceptContextError('invalid_registry', '只允许显式配置 project/pipeline/use/method 文档。')
        self.documents = {key: Path(value) for key, value in configured.items()}
        if any(not path.is_absolute() for path in self.documents.values()):
            raise ConceptContextError('invalid_registry', '注册文档必须使用确定的绝对路径；不在构造时读取或解析文件。')
        for limit in (max_document_bytes, max_selected_bytes, max_context_bytes):
            if type(limit) is not int or limit <= 0:
                raise ConceptContextError('invalid_limit', '概念读取边界必须为正整数字节数。')
        self.max_document_bytes = max_document_bytes
        self.max_selected_bytes = max_selected_bytes
        self.max_context_bytes = max_context_bytes

    @classmethod
    def for_settings(cls, settings):
        """Resolve an operator-selected library, never a model-supplied path.

        Only the existing development checkout may prefer its existing F:
        originals. A copied/released project uses its four bundled originals.
        """
        configured = getattr(settings, 'concept_root', '')
        if configured:
            root = Path(configured).expanduser().resolve()
            project_root = root / 'concepts' if (root / 'concepts/ai-comic/CONCEPT.md').is_file() else root
            method_root = root if (root / 'concept-workflow/SKILL.md').is_file() else project_root
            return cls({'project': project_root / 'ai-comic/CONCEPT.md',
                'pipeline': project_root / 'ai-comic/pipeline/CONCEPT.md',
                'use': project_root / 'ai-comic/SOURCES_AND_USE.md',
                'method': method_root / 'concept-workflow/SKILL.md'})
        if (_PROJECT_ROOT.resolve() == Path('D:/Models/ai_comic').resolve()
                and all(path.is_file() for path in LEGACY_DOCUMENTS.values())):
            return cls(LEGACY_DOCUMENTS)
        return cls()

    def _read(self, document: str) -> dict:
        path = self.documents[document]
        try:
            before = path.stat()
            if before.st_size > self.max_document_bytes:
                raise ConceptContextError('document_too_large', f'{document} 整件超限，未读取或截断。', document=document)
            with path.open('rb') as stream:
                raw = stream.read(self.max_document_bytes + 1)
            after = path.stat()
        except OSError as exc:
            raise ConceptContextError('document_unavailable', f'{document} 注册原件不可读；未使用旧摘要或另一文件替代。', document=document) from exc
        if len(raw) > self.max_document_bytes:
            raise ConceptContextError('document_too_large', f'{document} 整件超限，未截断。', document=document)
        if _signature(before) != _signature(after) or len(raw) != after.st_size:
            raise ConceptContextError('document_changed', f'{document} 在读取期间变化，请重新明确读取版本。', document=document)
        try:
            text = raw.decode('utf-8')
        except UnicodeDecodeError as exc:
            raise ConceptContextError('invalid_encoding', f'{document} 不是 UTF-8 原文，未猜测编码或修补。', document=document) from exc
        return {'document': document, 'path': path.as_posix(), 'sha256': _digest(raw),
                'bytes': len(raw), 'line_count': len(text.splitlines()),
                'sections': _sections(text), '_text': text, '_signature': _signature(after)}

    def _selections(self, selections) -> list[dict]:
        values = DEFAULT_SELECTIONS if selections is None else selections
        if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
            raise ConceptContextError('invalid_selection', '选择须为文档ID与完整章节的列表，不能传路径或任务关键词。')
        result = []
        for value in values:
            if (not isinstance(value, Mapping) or set(value) - {'document', 'section', 'document_sha256'} or
                    not {'document', 'section'} <= set(value)):
                raise ConceptContextError('invalid_selection', '章节选择只允许 document、section 与可选 document_sha256。')
            document, section = value['document'], value['section']
            if not isinstance(document, str) or document not in self.documents:
                raise ConceptContextError('unknown_document', '未知或未注册的概念文档；不读取请求中的任意路径。')
            if section is not None and (not isinstance(section, str) or not section.strip()):
                raise ConceptContextError('invalid_selection', 'section 必须是完整标题、目录ID或明确的null整件选择。')
            expected = value.get('document_sha256')
            if expected is not None and (not isinstance(expected, str) or not re.fullmatch(r'[0-9a-fA-F]{64}', expected)):
                raise ConceptContextError('invalid_selection', '预期整件SHA必须是完整SHA256。', document=document)
            result.append(dict(value))
        return result

    def load(self, selections: Sequence[Mapping] | None = None) -> dict:
        requested = self._selections(selections)
        # Catalogues make unselected complete sections discoverable, not read
        # by the model. No linked original, image, sound or network is accessed.
        documents = {key: self._read(key) for key in self.documents}
        selected, occupied = [], {}
        selected_bytes = 0
        for choice in requested:
            document = documents[choice['document']]
            expected = choice.get('document_sha256')
            if expected is not None and expected.lower() != document['sha256']:
                raise ConceptContextError('document_changed', '所选章节所属整件已变化；旧目录不能默认为新版本。', document=document['document'])
            if choice['section'] is None:
                section = {'section_id': None, 'title': None, 'start_line': 1,
                           'end_line': document['line_count'], 'text_sha256': document['sha256'],
                           'text_bytes': document['bytes']}
            else:
                matches = [item for item in document['sections']
                           if choice['section'] in (item['section_id'], item['title'])]
                if not matches:
                    raise ConceptContextError('unknown_section', '完整章节不存在；可先load([])获取当前目录，未近似匹配或截断。', document=document['document'])
                if len(matches) != 1:
                    raise ConceptContextError('ambiguous_section', '章节标题重复；请使用当前目录section_id明确选择。', document=document['document'])
                section = matches[0]
            intervals = occupied.setdefault(document['document'], [])
            span = section['start_line'], section['end_line']
            if any(span[0] <= old[1] and old[0] <= span[1] for old in intervals):
                raise ConceptContextError('overlapping_selection', '同一文档的所选完整章节重叠；未自动去重或覆盖选择。', document=document['document'])
            intervals.append(span)
            text = ''.join(document['_text'].splitlines(keepends=True)[span[0] - 1:span[1]])
            selected_bytes += len(text.encode('utf-8'))
            if selected_bytes > self.max_selected_bytes:
                raise ConceptContextError('selection_too_large', '所选完整原文总量超限；需明确另选章节或调整配置，不自动摘要或截断。')
            selected.append({'document': document['document'], 'document_path': document['path'],
                'document_sha256': document['sha256'], **section, 'text': text,
                'complete_selection': True, 'whole_document': choice['section'] is None,
                'content_kind': 'verbatim_original_text_not_model_summary'})
        result = {'schema_version': 'creation-concept-context-v1',
            'purpose': 'creative_planner_reference_not_pixel_observation',
            'selection_mode': 'default_entry_sections' if selections is None else 'explicit_sections',
            'selected_sections': selected,
            'available_documents': [{key: value for key, value in document.items() if not key.startswith('_')}
                                    for document in documents.values()],
            'selected_text_bytes': selected_bytes,
            'concept_understanding_verified': False, 'reference_pixels_read': False,
            'reading_scope': 'Selected registered text only; not the entire concept or its referenced artifacts.',
            'reference_scope': 'Image/audio/original-file links are retained as text pointers, not opened or observed.',
            'selection_policy': 'Choose complete catalogue sections for the current situation; no keyword-based task routing.',
            'byte_measurement': 'utf8_compact_json',
            'bounds': {'max_document_bytes': self.max_document_bytes,
                       'max_selected_bytes': self.max_selected_bytes,
                       'max_context_bytes': self.max_context_bytes}}
        # Reject changes during this load rather than silently mixing versions.
        for key, original in documents.items():
            current = self._read(key)
            if (current['sha256'], current['_signature']) != (original['sha256'], original['_signature']):
                raise ConceptContextError('document_changed', f'{key} 在本次读取期间变化，未返回混合版本。', document=key)
        result['context_utf8_bytes'] = 0
        for _ in range(4):
            length = len(_compact_bytes(result))
            if result['context_utf8_bytes'] == length:
                break
            result['context_utf8_bytes'] = length
        if result['context_utf8_bytes'] > self.max_context_bytes:
            raise ConceptContextError('context_too_large', '完整目录与所选原文上下文超限；未裁切目录、章节或证据。')
        return result

    def load_for_planning(self, context: Mapping) -> dict:
        """Read complete stage-relevant originals without changing ``load``.

        Presence of current-run artifacts selects the result-dependent reading
        purpose. Neither original-text keywords, model names, reported quality
        nor available tokens select or shorten the sections. The catalogue
        remains complete so the planner can explicitly request further intact
        material. Artifacts establish a stage, not that their pixels were read.
        """
        if (not isinstance(context, Mapping) or 'artifacts' not in context
                or not isinstance(context['artifacts'], list)):
            raise ConceptContextError('invalid_planning_context',
                '规划用途必须依据当前运行显式的artifacts列表，不能猜测阶段或按文本关键词选择。')
        result_dependent = bool(context['artifacts'])
        choices = PLANNING_INITIAL_SELECTIONS + (PLANNING_RESULT_SELECTIONS if result_dependent else ())
        result = self.load([dict(choice) for choice in choices])
        result['planning_purpose'] = ('result_dependent_creative_proposal' if result_dependent
                                      else 'initial_creative_proposal')
        result['planning_selection_basis'] = ('current_run_has_artifacts' if result_dependent
                                              else 'current_run_has_no_artifacts')
        result['planning_selection_reasons'] = [
            {**choice, 'reason': reason}
            for choice, reason in zip(choices, _PLANNING_SELECTION_REASONS)
        ]
        result['planning_scope'] = (
            'Stage-relevant complete registered text, not the entire concept, '
            'not an exhaustive list of creative possibilities, and not pixel observation. '
            'Unselected catalogue entries and linked artifacts are not claimed read; '
            'further complete sections remain explicitly selectable.')
        # Additional scope/selection receipts are part of the actual bounded
        # context, not exempt annotations or a reason to shorten any original.
        result['context_utf8_bytes'] = 0
        for _ in range(4):
            length = len(_compact_bytes(result))
            if result['context_utf8_bytes'] == length:
                break
            result['context_utf8_bytes'] = length
        if result['context_utf8_bytes'] > self.max_context_bytes:
            raise ConceptContextError('context_too_large',
                '完整用途收据、目录与所选原文上下文超限；未裁切目录、章节或证据。')
        return result


def load(selections: Sequence[Mapping] | None = None) -> dict:
    """Explicit convenience read using the project's approved local registry."""
    return CreationConceptContext().load(selections)
