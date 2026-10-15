"""Recoverable, bounded local authoring before any image job is submitted."""
from __future__ import annotations

import asyncio
from copy import deepcopy
import json
import re
import threading
import time
import uuid
from dataclasses import replace
from datetime import datetime, UTC
from pathlib import Path

from pydantic import Field, ValidationError

from .art_direction import (AuthorDraft,
                            authoring_input, compile_author_draft, fingerprint)
from .authoring_contract import (SOURCE_SYSTEM_PROMPT, SOURCE_SCHEMA, ART_SYSTEM_PROMPT,
                                parse_source_response, art_direction_input, art_response_schema, compile_art_choices)
from .config import PROJECT_ROOT, Settings
from .local_authoring_runtime import LocalAuthoringRuntime, RuntimeConfig, apply_start_resource_policy
from .local_authoring_runtime import inference_state
from .portable_runtime import configured_runtime, runtime_from_config
from .schemas import GenerateRequest, NewbieCompileRequest
from .author_reconsideration import (RECONSIDER_ART_SYSTEM, RECONSIDER_SOURCE_INSTRUCTION,
    EXPLAIN_REVISION_ART_SYSTEM, reconsideration_art_input, reconsideration_art_schema,
    revision_art_schema, validate_author_concern, reported_conflict_concern,
    conflict_transition_review)


class PreparationRequest(NewbieCompileRequest):
    config: dict = Field(default_factory=dict)
    request_id: str = Field(pattern=r'^[a-zA-Z0-9_-]{8,80}$')


def preparation_failure_detail(error: Exception, *, returned_output: bool = False) -> str:
    """Explain the failed phase without hiding its original error/evidence."""
    code = getattr(error, 'code', None)
    if code == 'insufficient_start_resources':
        return '当前可用内存或显存不足，构思模型未加载，未提交生图；没有自动重试。'
    if code in {'resource_hard_floor', 'resource_sustained_floor'}:
        return '已触发内存或显存资源保护并释放本次构思进程，未提交生图；没有自动重试。'
    if isinstance(error, ValidationError) or (returned_output and isinstance(error, ValueError)):
        return '构思输出未满足完整格式、原文条件或身份依据要求，未提交生图；原始输出与错误保留供核查，没有自动补造内容。'
    return '本次本地准备未成功，未提交生图；原始错误与可用证据保留，没有回退到简化词或自动重试。'


def typed_generation_config(config: dict) -> dict:
    """Compare execution values in their actual request types, not JSON spelling.

    The raw preparation input remains untouched. Pydantic also normalizes the
    complete nested SourceStageParameters, without rounding numeric values or
    conflating an explicit empty string with null in optional string controls.
    """
    # At this stage only the requested execution configuration exists. The
    # actual target/source conditions are authored later and checked at submit;
    # placeholders validate parameter types and never become model inputs.
    validated = GenerateRequest.model_validate({'source_prompt': 'preparation only',
        **config, 'prompt': 'preparation only'}).model_dump(mode='json')
    return {key: validated[key] for key in config if key in GenerateRequest.model_fields}


class CreationPreparations:
    preparation_timeout_seconds = 600

    def __init__(self, settings: Settings, *, runtime=None, root: Path | None = None, clock=None):
        self.settings = settings
        self.root = root or PROJECT_ROOT / 'data' / 'creation_preparations'
        self.root.mkdir(parents=True, exist_ok=True)
        # Both authoring stages require one JSON document. Use the actual
        # supplied schema; structural constraints do not certify its meaning.
        config = replace(configured_runtime(settings), constrain_json_output=True)
        if config.backend == 'local_gguf':
            config = apply_start_resource_policy(config)
        self.runtime = runtime or runtime_from_config(config)
        self.clock = clock or time.monotonic
        self.records: dict[str, dict] = {}
        self.tasks: dict[str, asyncio.Task] = {}
        self.cancellations: dict[str, threading.Event] = {}
        self._submission_locks: dict[str, asyncio.Lock] = {}
        self._serial = asyncio.Lock()
        for path in self.root.glob('*.json'):
            try:
                record = json.loads(path.read_text(encoding='utf-8'))
                if record.get('status') in {'queued', 'running'}:
                    record.update(status='failed', stage='interrupted',
                                  detail='服务重启中断了准备；没有自动重新推理或提交生图。',
                                  error='preparation_interrupted_by_service_restart')
                    self._save(record)
                self.records[record['id']] = record
            except (OSError, ValueError, KeyError):
                continue

    def _save(self, record: dict):
        record['updated_at'] = datetime.now(UTC).isoformat()
        target = self.root / (record['id'] + '.json')
        temporary = target.with_suffix('.json.tmp')
        temporary.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding='utf-8')
        temporary.replace(target)

    def get(self, identifier: str) -> dict | None:
        record = self.records.get(identifier)
        if record is None:
            return None
        state = inference_state()
        return {**record, 'runtime_state': state if state.get('owner') == identifier else {'busy': False}}

    async def create(self, request: PreparationRequest) -> dict:
        snapshot = request.model_dump(mode='json', exclude={'request_id'})
        digest = fingerprint(snapshot)
        old = next((value for value in self.records.values()
                    if value.get('request_id') == request.request_id), None)
        if old:
            if old['input_fingerprint'] != digest:
                raise ValueError('同一准备请求标识不能用于另一份输入')
            return old
        # Validate complete route/parameters before taking scarce local resources.
        typed_config = typed_generation_config(request.config)
        data = authoring_input(request, request.config, self.settings)
        identifier = uuid.uuid4().hex[:12]
        record = {'id': identifier, 'request_id': request.request_id, 'revision': 1,
                  'input_fingerprint': digest, 'status': 'queued', 'stage': 'queued',
                  'detail': '等待本地推理资源；准备不会提交图片生成。',
                  'input_snapshot': snapshot, 'effective_config': request.config, 'typed_config': typed_config,
                  'authoring_input': data, 'author_proposal': None, 'compiled': None,
                  'stage_evidence': {}, 'source_interpretation': None,
                  'generation_submitted': False, 'iteration_submitted': False, 'error': None}
        self.records[identifier] = record
        self.cancellations[identifier] = threading.Event()
        self._save(record)
        self.tasks[identifier] = asyncio.create_task(self._run(record, request), name=f'authoring-{identifier}')
        return record

    async def _run(self, record: dict, request: PreparationRequest):
        event = self.cancellations[record['id']]
        try:
            async with self._serial:
                request = PreparationRequest.model_validate(deepcopy(record['input_snapshot'])
                    | {'request_id': record['request_id']})
                deadline = self.clock() + self.preparation_timeout_seconds
                frozen_input = deepcopy(record['authoring_input'])
                frozen_input_sha256 = fingerprint(frozen_input)
                async def generate(**kwargs):
                    if event.is_set():
                        raise asyncio.CancelledError()
                    if fingerprint(record['authoring_input']) != frozen_input_sha256:
                        raise ValueError('准备输入在推理过程中改变，未沿用旧解释')
                    remaining = deadline - self.clock()
                    if remaining <= 0:
                        raise TimeoutError('完整准备回合达到600秒上限，未继续推理或派发')
                    result = await asyncio.wait_for(self.runtime.generate_json(**kwargs), timeout=remaining)
                    if self.clock() >= deadline:
                        error = TimeoutError('完整准备回合达到600秒上限，未采纳过期回答')
                        error.evidence_dir, error.raw_runtime_result = result.evidence_dir, result
                        raise error
                    if fingerprint(record['authoring_input']) != frozen_input_sha256:
                        error = ValueError('准备输入在推理过程中改变，未采纳过期回答')
                        error.evidence_dir, error.raw_runtime_result = result.evidence_dir, result
                        raise error
                    return result
                if event.is_set():
                    record.update(status='cancelled', stage='cancelled', detail='准备已取消，未提交生图。')
                    return
                record.update(status='running', stage='source_interpretation',
                              detail='区分原文要求、对象关系与背景缘由；精确引用不代表理解已验证。')
                self._save(record)
                result = await generate(system_prompt=SOURCE_SYSTEM_PROMPT,
                                data=frozen_input, response_schema=SOURCE_SCHEMA, cancel_event=event,
                                owner=record['id'])
                record['source_extraction_raw'] = result.data
                source_evidence = {'directory': str(result.evidence_dir), 'usage': result.usage,
                                   'timings': result.timings, 'status': 'returned'}
                record['stage_evidence']['source_interpretation'] = source_evidence
                self._save(record)
                if event.is_set():
                    source_evidence['status'] = 'cancelled'
                    record.update(status='cancelled', stage='cancelled', detail='准备已取消，未提交生图。')
                    return
                sources = parse_source_response(result.data, request, frozen_input)
                record['source_interpretation'] = sources.model_dump(mode='json')
                source_evidence['status'] = 'validated'
                if sources.blocking:
                    sources, result = await self._reconsider_source(record, request, sources,
                        None, frozen_input, generate, event)
                    if sources.blocking:
                        self._source_review(record, sources)
                        return
                # Same cancellation transaction, separately preserved response/evidence.
                if event.is_set():
                    record.update(status='cancelled', stage='cancelled', detail='准备已取消，未提交生图。')
                    return
                if not record.get('source_reconsideration'):
                    record.update(stage='art_direction',
                        detail='在保留原文条件的基础上提出空间、重要物体形体与光色关系；角色外观不由此阶段编造。')
                    self._save(record)
                    result = await generate(system_prompt=RECONSIDER_ART_SYSTEM,
                        data=reconsideration_art_input(art_direction_input(frozen_input, sources), sources),
                        response_schema=reconsideration_art_schema(sources), cancel_event=event, owner=record['id'])
                    record['author_choices_raw'] = result.data
                    art_evidence = {'directory': str(result.evidence_dir), 'usage': result.usage,
                                    'timings': result.timings, 'status': 'returned'}
                    record['stage_evidence']['art_direction'] = art_evidence
                    self._save(record)
                    if event.is_set():
                        art_evidence['status'] = 'cancelled'
                        record.update(status='cancelled', stage='cancelled', detail='准备已取消，未提交生图。')
                        return
                    if result.data.get('art_protocol') == 'reconsider_source':
                        sources, result = await self._reconsider_source(record, request, sources,
                            result, frozen_input, generate, event)
                        if sources.blocking:
                            self._source_review(record, sources)
                            return
                evidence = {'stages': record['stage_evidence']}
                record.update(runtime_evidence=evidence, unvalidated_draft={
                    'status': 'unvalidated',
                    'parsed_response_file': str(Path(result.evidence_dir) / 'parsed_response.json'),
                    'semantic_correctness_verified': False,
                })
                record.update(stage='compiling', detail='按当前作者提案构建各模型原生正负条件。')
                self._save(record)
                compiled = compile_art_choices(result.data, sources, request, proposal_id=record['id'],
                            runtime_evidence=evidence,
                            recipes=frozen_input['full_style_recipes'], config=request.config)
                if record.get('source_reconsideration'):
                    record['stage_evidence']['source_revision_art']['status'] = 'compiled_unverified'
                    record['current_author_choices_raw'] = deepcopy(result.data)
                else:
                    art_evidence['status'] = 'validated'
                if record.get('source_reconsideration'):
                    record['source_reconsideration']['activation'] = {
                        'current_interpretation_and_art_rebound': True,
                        'source_changed': not record['source_reconsideration']['preview']['receipt']['no_change'],
                        'semantic_correctness_verified': False, 'visual_accuracy_verified': False,
                        'dispatch_performed': False}
                record.update(compiled=compiled, author_proposal=compiled['author_proposal'],
                              source_interpretation=sources.model_dump(mode='json'),
                              status='needs_confirmation' if compiled['professional_review']['blocking'] else 'ready',
                              stage='review', detail='作者提案与模型条件已准备；实际画面仍需检验。')
        except asyncio.CancelledError:
            event.set()
            record.update(status='cancelled', stage='cancelled', detail='准备已停止，未提交生图。')
            raise
        except Exception as exc:
            stage = record.get('stage')
            if stage == 'compiling':
                stage = 'art_direction'
            evidence_dir = getattr(exc, 'evidence_dir', None)
            raw_return = getattr(exc, 'raw_runtime_result', None)
            if raw_return is not None:
                record.setdefault('unadopted_returns', []).append({
                    'stage': stage, 'data': deepcopy(raw_return.data),
                    'directory': str(raw_return.evidence_dir), 'usage': raw_return.usage,
                    'timings': raw_return.timings, 'not_adopted': True,
                    'reason': f'{type(exc).__name__}: {exc}'})
            if evidence_dir is not None:
                record['stage_evidence'][stage] = {'directory': str(evidence_dir),
                    'phase': getattr(exc, 'phase', None), 'error_code': getattr(exc, 'code', None)}
            entry = record['stage_evidence'].setdefault(stage, {})
            entry.update(status='cancelled' if event.is_set() else 'failed', error=f'{type(exc).__name__}: {exc}')
            record['runtime_evidence'] = {'stages': record['stage_evidence']}
            record.update(status='cancelled' if event.is_set() else 'failed',
                          stage='cancelled' if event.is_set() else 'failed',
                          detail='准备已取消，未提交生图。' if event.is_set() else
                              preparation_failure_detail(exc, returned_output=bool(record.get('source_extraction_raw'))),
                          error=f'{type(exc).__name__}: {exc}')
        finally:
            self._save(record)

    @staticmethod
    def _source_review(record, sources):
        source_records = {item.id: item for item in sources.conditions}
        record.update(status='needs_confirmation', stage='review',
            detail='系统已重看原句与资源，但仍不能安全确定解释或取舍；未提交生图。',
            source_review={'blocking': True, 'conflict_is_verified_user_requirement': False,
                'reconsideration_attempted': bool(record.get('source_reconsideration')),
                'questions': [
                {'question': '系统对这一处的解释是否符合你的意思？也可以调整所选角色资源。',
                 'reason': item.description + '\n' + '\n'.join(
                    f'{source_records[key].origin}: {source_records[key].value}'
                    + (f'（原文：{source_records[key].source_quote}）' if source_records[key].source_quote else '')
                    for key in item.condition_ids if key in source_records),
                 'action': '核对描述或资源选择；不用补专业参数。',
                 'condition_ids': item.condition_ids, 'attribute': item.attribute,
                 **({'identity_choices': {
                    'description_identity': sources.identity_binding.model_dump(mode='json'),
                    'knowledge_identity': sources.knowledge.model_dump(mode='json')}} if item.kind == 'identity' else {})}
                for item in sources.conflicts]})

    async def _reconsider_source(self, record, request, sources, initial_art,
                                 frozen_input, generate, event):
        from .source_reconsideration import (preview_source_reconsideration,
                                            confirm_source_reconsideration)
        if initial_art is None:
            concern = reported_conflict_concern(sources)
            trigger = 'reported_conflict'
        else:
            concern = validate_author_concern(initial_art.data, sources).model_dump(mode='json')
            trigger = 'author_concern'
            record['stage_evidence']['art_direction']['status'] = 'source_reconsideration_requested_unverified'
        previous = sources.model_dump(mode='json')
        previous_sha256 = fingerprint(previous)
        entry = {'review_trigger': trigger,
            **({'author_concern_raw': deepcopy(initial_art.data)} if initial_art is not None
               else {'reported_conflict_concern': deepcopy(concern)}),
            'previous_source_interpretation': previous,
            'previous_source_extraction_raw': deepcopy(record['source_extraction_raw']),
            'activation': {'current_interpretation_and_art_rebound': False,
                           'semantic_correctness_verified': False, 'dispatch_performed': False}}
        record['source_reconsideration'] = entry
        record.update(stage='source_reconsideration',
            detail='保留原话与原回答，核对报告的冲突或作者质疑是否来自误解释；尚未要求用户取舍。')
        self._save(record)
        result = await generate(system_prompt=SOURCE_SYSTEM_PROMPT + RECONSIDER_SOURCE_INSTRUCTION,
            data=deepcopy(frozen_input) | {'current_source_snapshot': deepcopy(previous),
                'review_trigger': trigger, 'review_concern': deepcopy(concern), 'concern_is_verified': False},
            response_schema=SOURCE_SCHEMA, cancel_event=event, owner=record['id'])
        entry['source_extraction_raw'] = deepcopy(result.data)
        record['stage_evidence']['source_reconsideration'] = {
            'directory': str(result.evidence_dir), 'usage': result.usage,
            'timings': result.timings, 'status': 'returned'}
        self._save(record)
        if event.is_set():
            raise asyncio.CancelledError()
        preview = preview_source_reconsideration(previous, result.data,
            expected_source_sha256=previous_sha256)
        permission = conflict_transition_review(sources, preview.sources)
        entry['conflict_transition_review'] = permission
        allow_art = permission['automatic_adoption_permitted'] and not preview.sources.blocking
        record['stage_evidence']['source_reconsideration']['status'] = 'preview_assembled_unverified'
        entry['preview'] = {'previous_snapshot': preview.previous_snapshot,
            'proposed_source_snapshot': preview.sources.model_dump(mode='json'),
            'actual_changes': preview.actual_changes, 'receipt': preview.receipt}
        record.update(stage='source_revision_art',
            detail='根据实际解释差异逐项说明理由，并重新提出完整画面方案；没有认证解释正确。')
        self._save(record)
        result = await generate(system_prompt=EXPLAIN_REVISION_ART_SYSTEM,
            data=reconsideration_art_input(art_direction_input(frozen_input, preview.sources), preview.sources)
                | {'previous_source_snapshot': deepcopy(previous),
                   'proposed_source_snapshot': preview.sources.model_dump(mode='json'),
                   'actual_changes': preview.actual_changes,
                   'source_reconsideration_receipt': preview.receipt,
                   'review_trigger': trigger, 'review_concern': deepcopy(concern),
                   'conflict_transition_review': deepcopy(permission), 'art_is_permitted': allow_art},
            response_schema=revision_art_schema(preview.sources, len(preview.actual_changes), allow_art=allow_art),
            cancel_event=event, owner=record['id'])
        entry['revision_art_raw'] = deepcopy(result.data)
        record['stage_evidence']['source_revision_art'] = {
            'directory': str(result.evidence_dir), 'usage': result.usage,
            'timings': result.timings, 'status': 'returned'}
        self._save(record)
        if event.is_set():
            raise asyncio.CancelledError()
        if set(result.data) != {'change_annotations', 'art_choices'}:
            raise ValueError('解释修订必须包含且仅包含逐项变化说明和完整新作者方案')
        confirmed = confirm_source_reconsideration(preview, result.data['change_annotations'],
            expected_preview_sha256=preview.receipt['preview_sha256'],
            expected_source_sha256=previous_sha256,
            expected_revised_source_sha256=fingerprint(preview.sources.model_dump(mode='json')))
        entry['revision_receipt'] = confirmed.receipt
        new_art = result.data['art_choices']
        if not permission['automatic_adoption_permitted']:
            if new_art is not None:
                raise ValueError('解释改写尚未获得自动采纳权限，不能夹带可派发作者方案')
            # The full candidate and its explanation remain in the preview;
            # the old unresolved source, not a silently erased requirement,
            # remains current until the user resolves that interpretation.
            return sources, result
        if confirmed.sources.blocking:
            if new_art is not None:
                raise ValueError('来源冲突未解除，不能夹带可派发作者方案')
            record['source_interpretation'] = confirmed.sources.model_dump(mode='json')
            return confirmed.sources, result
        if not isinstance(new_art, dict):
            raise ValueError('解释修订后缺少完整新作者方案，未使用旧方案补齐')
        if new_art.get('art_protocol') == 'reconsider_source':
            validate_author_concern(new_art, confirmed.sources)
            raise ValueError('一次来源重看后作者仍提出质疑；已保留原件，没有无限重试或派发')
        # A separate envelope is retained verbatim above. Only its explicitly
        # supplied complete art document enters the ordinary compiler.
        from types import SimpleNamespace
        return confirmed.sources, SimpleNamespace(data=deepcopy(new_art), evidence_dir=result.evidence_dir,
            usage=result.usage, timings=result.timings)

    async def cancel(self, identifier: str) -> dict:
        # A cancel cannot race a successful first-job reservation and claim
        # that an already submitted image job never existed.
        async with self._submission_locks.setdefault(identifier, asyncio.Lock()):
            record = self.records[identifier]
            submission = record.get('generation_submissions', {}).get(str(record['revision']), {})
            if submission.get('status') in {'submitting', 'unknown', 'submitted'}:
                record['detail'] = (
                    '本版本已经关联迭代任务或创建结果尚未确认；准备取消不会取消迭代任务。'
                    '请查看关联迭代任务并使用其暂停或取消操作。'
                    if submission.get('kind') == 'iteration' else
                    '本版本已经提交或提交结果尚未确认；准备取消不会取消图片任务。'
                    '请查看关联任务并使用图片任务的取消操作。')
                self._save(record)
                return record
            return self._cancel_preparation(identifier)

    def _cancel_preparation(self, identifier: str) -> dict:
        record = self.records[identifier]
        event = self.cancellations.get(identifier)
        if event and record['status'] in {'queued', 'running'}:
            event.set()
            if record['status'] == 'queued':
                record.update(status='cancelled', stage='cancelled', detail='已取消排队中的准备；不会提交生图。')
            else:
                record['detail'] = '正在停止本地准备并释放其推理进程；不会提交生图。'
        elif record['status'] in {'ready', 'needs_confirmation'}:
            record.update(status='cancelled', stage='cancelled', detail='已放弃这次准备，未提交生图。')
        self._save(record)
        return record

    def _source_candidate(self, record):
        """Rebuild the complete saved candidate; explanations are not approval."""
        from .source_reconsideration import (preview_source_reconsideration,
                                            confirm_source_reconsideration)
        entry = record.get('source_reconsideration') or {}
        saved = entry.get('preview') or {}
        receipt = saved.get('receipt') or {}
        if not saved or not entry.get('revision_art_raw'):
            raise ValueError('没有完整且已解释的来源候选，不能继续空提案')
        current = record.get('source_interpretation')
        if fingerprint(current) != receipt.get('previous_source_sha256'):
            raise ValueError('当前来源与候选的原版本不同，请重新读取，未采用旧候选')
        preview = preview_source_reconsideration(current, entry.get('source_extraction_raw'),
            expected_source_sha256=receipt['previous_source_sha256'])
        rebuilt = {'previous_snapshot': preview.previous_snapshot,
            'proposed_source_snapshot': preview.sources.model_dump(mode='json'),
            'actual_changes': preview.actual_changes, 'receipt': preview.receipt}
        if fingerprint(saved) != fingerprint(rebuilt):
            raise ValueError('完整候选或收据已经变化，未复用原来的采纳权限')
        confirmed = confirm_source_reconsideration(preview,
            entry['revision_art_raw'].get('change_annotations'),
            expected_preview_sha256=receipt['preview_sha256'],
            expected_source_sha256=receipt['previous_source_sha256'],
            expected_revised_source_sha256=receipt['revised_source_sha256'])
        if fingerprint(entry.get('revision_receipt')) != fingerprint(confirmed.receipt):
            raise ValueError('候选解释收据不完整或已变化，不能继续')
        return preview, confirmed

    async def resume_source(self, identifier: str, body: dict) -> dict:
        """User-led interpretation continuation, never an image submission.

        Clarifications are distinct originals, not replacement source quotes.
        Adoption binds exactly the displayed full candidate and authorizes one
        fresh ART attempt only; integrity checks do not certify its meaning.
        """
        record = self.records.get(identifier)
        if record is None:
            raise ValueError('关联准备不存在')
        if not isinstance(body, dict) or set(body) - {
                'action', 'expected_revision', 'candidate_sha256', 'request_id', 'clarification'}:
            raise ValueError('来源接续只接受明确动作、当前版本、候选指纹和请求标识')
        action, request_id = body.get('action'), body.get('request_id')
        if not isinstance(action, str) or action not in {'adopt', 'clarify'} or not isinstance(request_id, str) or not re.fullmatch(
                r'[a-zA-Z0-9_-]{8,80}', request_id):
            raise ValueError('来源接续需要 adopt 或 clarify 及有效请求标识')
        if type(body.get('expected_revision')) is not int:
            raise ValueError('来源接续需要明确的整数准备版本')
        clarification = body.get('clarification')
        if action == 'clarify':
            if not isinstance(clarification, str) or not clarification.strip() or len(clarification) > 4000:
                raise ValueError('请用自然语言说明原句哪处被误理解，最多4000字；不用填写来源JSON')
        elif 'clarification' in body:
            raise ValueError('采用当前候选不能同时夹带未审阅的新纠正')
        digest = fingerprint(body)
        previous = record.get('source_continuation_requests', {}).get(request_id)
        if previous is not None:
            if previous['request_fingerprint'] != digest:
                raise ValueError('同一接续请求标识不能用于另一份纠正或候选')
            return record  # Including failed/unknown attempts: replay never infers again.
        task = self.tasks.get(identifier)
        submission = record.get('generation_submissions', {}).get(str(record['revision']))
        lock = self._submission_locks.get(identifier)
        if (record.get('status') != 'needs_confirmation' or record.get('compiled') is not None
                or task is not None and not task.done() or lock is not None and lock.locked()
                or submission or record.get('generation_submitted') or record.get('iteration_submitted')):
            raise ValueError('只有未派发、无在途操作的来源审阅可以接续')
        if body['expected_revision'] != record['revision']:
            raise ValueError('准备版本已变化，请先重新读取，未接续旧版本')
        preview, confirmed = self._source_candidate(record)
        if body.get('candidate_sha256') != preview.receipt['preview_sha256']:
            raise ValueError('候选指纹已变化，请查看当前解释后重新选择')
        if action == 'adopt' and confirmed.sources.blocking:
            raise ValueError('这份候选仍有来源冲突；请纠正理解或调整原输入和资源后重新准备')
        receipt = {'request_id': request_id, 'request_fingerprint': digest,
            'action': action, 'expected_revision': record['revision'],
            'candidate_sha256': preview.receipt['preview_sha256'],
            'previous_source_sha256': preview.receipt['previous_source_sha256'],
            'status': 'queued', 'model_returns': [], 'dispatch_performed': False,
            'semantic_correctness_verified': False,
            'authority': 'user_explicit_candidate_adoption' if action == 'adopt'
                         else 'user_clarification_request_not_adoption',
            'scope': 'current_source_interpretation_and_fresh_art_only' if action == 'adopt'
                     else 'propose_interpretation_of_unchanged_original_only'}
        if action == 'clarify':
            receipt['clarification'] = {'text': clarification, 'request_id': request_id,
                'original_text_sha256': preview.sources.original_sha256,
                'kind': 'user_clarification_original', 'is_original_source_quote': False,
                'adoption_authorized': False}
        # Save the reservation and user original before scarce resource work.
        before = deepcopy(record)
        record.setdefault('source_continuation_requests', {})[request_id] = receipt
        record.update(status='queued', stage='source_continuation', error=None,
            source_continuation_request_id=request_id,
            detail='等待接续本地构思；本次不会提交图片。')
        try:
            self._save(record)
        except BaseException:
            record.clear()
            record.update(before)
            raise
        event = threading.Event()
        self.cancellations[identifier] = event
        self.tasks[identifier] = asyncio.create_task(self._run_source_continuation(
            record, receipt, preview, event), name=f'source-continuation-{identifier}-{request_id}')
        return record

    async def _run_source_continuation(self, record, receipt, preview, event):
        from .source_reconsideration import (preview_source_reconsideration,
                                            confirm_source_reconsideration)
        initial_revision = record['revision']
        frozen_input = deepcopy(record['authoring_input'])
        input_binding = fingerprint({'authoring_input': frozen_input, 'input_snapshot': record['input_snapshot']})
        candidate_binding = receipt['candidate_sha256']
        complete_candidate_binding = fingerprint(record['source_reconsideration'])
        def still_current():
            if event.is_set():
                raise asyncio.CancelledError()
            if self.clock() >= deadline:
                raise TimeoutError('完整接续回合达到600秒上限，返回未采纳')
            if (record['revision'] != initial_revision
                    or fingerprint({'authoring_input': record['authoring_input'],
                                    'input_snapshot': record['input_snapshot']}) != input_binding
                    or (record.get('source_reconsideration') or {}).get('preview', {}).get(
                        'receipt', {}).get('preview_sha256') != candidate_binding
                    or fingerprint(record.get('source_reconsideration')) != complete_candidate_binding
                    or fingerprint(record['source_interpretation']) != receipt['previous_source_sha256']):
                raise ValueError('原句、资源、版本或候选在接续过程中变化，返回未采纳')
        try:
            async with self._serial:
                deadline = self.clock() + self.preparation_timeout_seconds
                request = PreparationRequest.model_validate(deepcopy(record['input_snapshot'])
                    | {'request_id': record['request_id']})
                # Only originals used to form this candidate accompany it.
                # Failed/later requests remain in history, not hidden new ART authority.
                clarifications = deepcopy(record['source_reconsideration'].get('user_clarifications', []))
                if receipt['action'] == 'clarify':
                    clarifications.append(deepcopy(receipt['clarification']))
                async def generate(**kwargs):
                    still_current()
                    remaining = deadline - self.clock()
                    if remaining <= 0:
                        raise TimeoutError('完整接续回合达到600秒上限，未继续推理')
                    result = await asyncio.wait_for(self.runtime.generate_json(**kwargs), timeout=remaining)
                    receipt['model_returns'].append({'data': deepcopy(result.data),
                        'directory': str(result.evidence_dir), 'usage': result.usage, 'timings': result.timings,
                        'adopted_as_semantically_verified': False})
                    self._save(record)  # Preserve even cancelled/late/invalid replies.
                    if self.clock() >= deadline:
                        raise TimeoutError('完整接续回合达到600秒上限，返回未采纳')
                    still_current()
                    return result
                still_current()
                receipt['status'] = 'running'
                record.update(status='running', stage='source_reconsideration' if receipt['action'] == 'clarify'
                    else 'art_direction', detail='重新解释或构思，未派发图片。')
                self._save(record)
                if receipt['action'] == 'clarify':
                    result = await generate(system_prompt=SOURCE_SYSTEM_PROMPT + RECONSIDER_SOURCE_INSTRUCTION
                        + '\nuser_clarifications保留用户对原句误解释的纠正原话，不是新增画面要求。'
                        '只重解释同一original_text；source_quote仍须来自原句，不能引用澄清冒充原文。'
                        '若纠正实际新增、改变或取消原句要求，请保留原要求并在uncertainties中明确需要修改原输入另准备，'
                        '不能通过这一步静默改要求、角色、资源或风格。',
                        data=deepcopy(frozen_input) | {'current_source_snapshot': deepcopy(preview.previous_snapshot),
                            'candidate_source_snapshot': preview.sources.model_dump(mode='json'),
                            'user_clarifications': deepcopy(clarifications),
                            'review_trigger': 'user_clarification', 'concern_is_verified': False,
                            'clarification_is_adoption': False},
                        response_schema=SOURCE_SCHEMA, cancel_event=event, owner=record['id'])
                    extraction = deepcopy(result.data)
                    revised = preview_source_reconsideration(preview.previous_snapshot, extraction,
                        expected_source_sha256=receipt['previous_source_sha256'])
                    result = await generate(system_prompt=EXPLAIN_REVISION_ART_SYSTEM,
                        data=art_direction_input(frozen_input, revised.sources) | {
                            'previous_source_snapshot': deepcopy(revised.previous_snapshot),
                            'proposed_source_snapshot': revised.sources.model_dump(mode='json'),
                            'actual_changes': deepcopy(revised.actual_changes),
                            'source_reconsideration_receipt': deepcopy(revised.receipt),
                            'user_clarifications': deepcopy(clarifications),
                            'review_trigger': 'user_clarification', 'art_is_permitted': False,
                            'clarification_is_adoption': False},
                        response_schema=revision_art_schema(revised.sources, len(revised.actual_changes), allow_art=False),
                        cancel_event=event, owner=record['id'])
                    if set(result.data) != {'change_annotations', 'art_choices'} or result.data['art_choices'] is not None:
                        raise ValueError('澄清只提出完整候选及变化说明，不能夹带已采纳美术方案')
                    explained = confirm_source_reconsideration(revised, result.data['change_annotations'],
                        expected_preview_sha256=revised.receipt['preview_sha256'],
                        expected_source_sha256=receipt['previous_source_sha256'],
                        expected_revised_source_sha256=revised.receipt['revised_source_sha256'])
                    entry = {'review_trigger': 'user_clarification',
                        'previous_source_interpretation': deepcopy(revised.previous_snapshot),
                        'previous_source_extraction_raw': deepcopy(record.get('source_extraction_raw')),
                        'source_extraction_raw': extraction,
                        'user_clarifications': deepcopy(clarifications),
                        'preview': {'previous_snapshot': revised.previous_snapshot,
                            'proposed_source_snapshot': revised.sources.model_dump(mode='json'),
                            'actual_changes': revised.actual_changes, 'receipt': revised.receipt},
                        'revision_art_raw': deepcopy(result.data), 'revision_receipt': explained.receipt,
                        'activation': {'current_interpretation_and_art_rebound': False,
                            'semantic_correctness_verified': False, 'dispatch_performed': False}}
                    receipt['status'] = 'candidate_returned_unadopted'
                    staged = deepcopy(record)
                    staged.setdefault('source_reconsideration_history', []).append(deepcopy(record['source_reconsideration']))
                    staged.update(source_reconsideration=entry, revision=initial_revision + 1,
                        status='needs_confirmation', stage='review', error=None,
                        detail='已按你的纠正提出新的完整解释，尚未采纳；可核对后继续构思。')
                    still_current()
                    self._save(staged)
                    record.clear()
                    record.update(staged)
                    return
                adoption = {key: deepcopy(receipt[key]) for key in ('request_id', 'candidate_sha256',
                    'previous_source_sha256', 'expected_revision', 'authority', 'scope')}
                adoption.update(original_text_sha256=preview.sources.original_sha256,
                    semantic_correctness_verified=False, identity_accuracy_verified=False,
                    image_dispatch_authorized=False)
                result = await generate(system_prompt=RECONSIDER_ART_SYSTEM
                    + '\nuser_clarifications是当前来源候选形成时的纠正原件，不是新增画面要求、原句引文或身份知识；'
                    'user_source_adoption只允许按这份来源继续构思，不认证语义、身份、画面或任何生成权限。',
                    data=reconsideration_art_input(art_direction_input(frozen_input, preview.sources), preview.sources)
                        | {'user_clarifications': deepcopy(clarifications), 'user_source_adoption': adoption},
                    response_schema=reconsideration_art_schema(preview.sources), cancel_event=event, owner=record['id'])
                if result.data.get('art_protocol') == 'reconsider_source':
                    concern = validate_author_concern(result.data, preview.sources).model_dump(mode='json')
                    receipt.update(status='author_concern_returned_unadopted', author_concern=concern)
                    record.update(status='needs_confirmation', stage='review', error=None,
                        detail='作者仍对解释提出疑问；原候选和原句保留，没有继续循环或提交图片。')
                    return
                record.update(stage='compiling', detail='完整新画面方案编译中，来源尚未切换，未提交图片。')
                self._save(record)
                evidence = {'stages': deepcopy(record.get('stage_evidence', {})),
                    'source_continuation': deepcopy(receipt), 'user_clarifications': deepcopy(clarifications),
                    'user_source_adoption': adoption}
                compiled = compile_art_choices(result.data, preview.sources, request,
                    proposal_id=record['id'], revision=initial_revision + 1, runtime_evidence=evidence,
                    recipes=frozen_input['full_style_recipes'], config=request.config)
                compiled['author_proposal'].update(user_clarifications=deepcopy(clarifications),
                    user_source_adoption=deepcopy(adoption))
                still_current()
                receipt['status'] = 'fresh_art_compiled_unverified'
                staged = deepcopy(record)
                staged.setdefault('source_reconsideration_history', []).append(deepcopy(record['source_reconsideration']))
                staged['source_reconsideration']['activation'] = {
                    'current_interpretation_and_art_rebound': True, 'authority': adoption,
                    'semantic_correctness_verified': False, 'dispatch_performed': False}
                staged.pop('source_review', None)
                staged.update(revision=initial_revision + 1, compiled=compiled,
                    author_proposal=compiled['author_proposal'],
                    source_interpretation=preview.sources.model_dump(mode='json'),
                    current_author_choices_raw=deepcopy(result.data), runtime_evidence=evidence,
                    status='needs_confirmation' if compiled['professional_review']['blocking'] else 'ready',
                    stage='review', error=None, detail='已采用你确认的解释并完整重构画面条件；未提交图片，画面仍待验证。')
                self._save(staged)  # Persist the whole activation before exposing it in memory.
                record.clear()
                record.update(staged)
        except asyncio.CancelledError:
            event.set()
            receipt['status'] = 'cancelled'
            record.update(status='cancelled', stage='cancelled', detail='接续已取消，未切换来源或提交图片。')
            raise
        except Exception as exc:
            receipt.update(status='failed', error=f'{type(exc).__name__}: {exc}')
            if getattr(exc, 'evidence_dir', None) is not None:
                receipt['runtime_error'] = {'directory': str(exc.evidence_dir),
                    'code': getattr(exc, 'code', None), 'phase': getattr(exc, 'phase', None)}
            record.update(status='needs_confirmation', stage='review',
                error=f'{type(exc).__name__}: {exc}',
                detail='接续未成功；用户原话、原候选及返回保留，未切换来源或提交图片；可重新审阅后重试。')
        finally:
            self._save(record)

    def rebuild(self, identifier: str, body: dict) -> dict:
        record = self.records[identifier]
        lock = self._submission_locks.get(identifier)
        submission = record.get('generation_submissions', {}).get(str(record['revision']), {})
        if (lock and lock.locked()) or submission.get('status') in {'submitting', 'unknown'}:
            raise ValueError('本版本提交仍在进行或结果未确认；没有重建或重复派发')
        if record['status'] not in {'ready', 'needs_confirmation'}:
            raise ValueError('只有已完成且未取消的准备可以重建')
        if not record.get('author_proposal') or not record.get('compiled'):
            raise ValueError('来源冲突尚未形成作者提案；请修改原描述或资源选择后重新准备，不能重建空提案')
        if body.get('expected_revision') != record['revision']:
            raise ValueError('准备版本已变化，请先重新读取；没有覆盖较新的修改')
        snapshot = record['input_snapshot']
        request = NewbieCompileRequest.model_validate(snapshot)
        previous = record['author_proposal']
        incoming = body.get('proposal', {})
        if not isinstance(incoming, dict) or any(key not in incoming for key in ('scene_narrative', 'visual_logic', 'positions')):
            raise ValueError('规格修改必须一并审阅当前场景提案与空间安排，不能沿用隐藏旧caption')
        if 'layout' in incoming and not isinstance(incoming['layout'], list):
            raise ValueError('明确修改布局必须给出文本数组；清空用[]，null不是审阅或删除选择')
        draft_data = {key: previous[key] for key in AuthorDraft.model_fields if key in previous}
        draft_data.update({key: value for key, value in incoming.items() if key in AuthorDraft.model_fields})
        if body.get('spec') is not None:
            draft_data['spec'] = body['spec']
        if (previous.get('layout') and body.get('spec') is not None
                and fingerprint(body['spec']) != fingerprint(previous['spec'])
                and 'layout' not in incoming):
            raise ValueError('规格变化时需一并审阅当前布局方案；请明确保留、修改或清空，不沿用隐藏旧布局')
        draft = AuthorDraft.model_validate(draft_data)
        revision = record['revision'] + 1
        compiled = compile_author_draft(draft, request, proposal_id=identifier, revision=revision,
                           runtime_evidence=previous['runtime_evidence'], knowledge=record['authoring_input']['knowledge'],
                           recipes=record['authoring_input']['full_style_recipes'], apply_selected_knowledge=False,
                           config=record['effective_config'],
                           preserve_style_conditions=bool(record.get('source_interpretation')))
        compiled['author_proposal']['author'] = 'user_edited_author_proposal_unverified'
        # An explicit professional revision may change current choices, but never
        # overwrites the source-step original or pretends it was the LLM output.
        if record.get('source_interpretation'):
            compiled['author_proposal']['source_conditions'] = record['source_interpretation']
            compiled['author_proposal']['source_conditions_status'] = 'historical_source_interpretation_after_user_revision'
        compiled['warnings'].append('手工提案与规格一起重建；文本同步不保证语义一致，未自动验证画面。')
        record.update(revision=revision, compiled=compiled, author_proposal=compiled['author_proposal'],
                      status='needs_confirmation' if compiled['professional_review']['blocking'] else 'ready',
                      generation_submitted=False, submitted_job_id=None,
                      iteration_submitted=False, submitted_iteration_run_id=None)
        self._save(record)
        return record

    def _validate_generation(self, request: GenerateRequest) -> dict:
        record = self.records.get(request.preparation_id)
        if record is None:
            raise ValueError('关联准备不存在，未提交图片生成')
        if request.preparation_revision != record.get('revision'):
            raise ValueError('准备版本已变化，请先重新读取；旧版本没有提交生图')
        if record.get('status') != 'ready' or not record.get('compiled'):
            raise ValueError('只有已就绪、无待确认问题且未取消的准备可以提交生图')
        compiled = record['compiled']
        if compiled.get('professional_review', {}).get('blocking'):
            raise ValueError('准备仍有待确认问题，未提交生图')
        proposal = request.author_proposal or {}
        if proposal.get('id') != record['id'] or proposal.get('revision') != record['revision']:
            raise ValueError('提交中的作者提案不是关联准备的当前版本')
        # These two UI annotations describe how the *actual* words are being
        # used; they may not disguise an edited source proposal as this record.
        annotations = {'condition_status', 'scene_semantic_sync_verified'}
        core = lambda value: {key: item for key, item in value.items() if key not in annotations}
        if fingerprint(core(proposal)) != fingerprint(core(record['author_proposal'])):
            raise ValueError('作者提案已变化，请先重建准备，不能沿用旧版本提交')
        if request.creation_intent is None or fingerprint(request.creation_intent.model_dump(mode='json')) != fingerprint(compiled['creation_intent']):
            raise ValueError('提交没有承接本次准备的完整原句与目标')
        manual = proposal.get('condition_status') == 'historical_after_native_override'
        if manual:
            if request.semantic_spec is not None or request.scene_contract is not None:
                raise ValueError('手改原生条件后旧提案仅为历史，不能声明当前规格同步')
            if request.render_pipeline.value == 'adaptive':
                raise ValueError('手改原生条件需要显式路线，不能用历史提案自动选路')
        else:
            if fingerprint(request.semantic_spec) != fingerprint(compiled['spec']) or fingerprint(request.scene_contract) != fingerprint(compiled['contract_v3']):
                raise ValueError('规格或场景已变化，请先重建本次准备')
            if fingerprint(request.native_prompts) != fingerprint(compiled['native_prompts']):
                raise ValueError('原生条件已手改；请明确标记旧提案仅为历史，或重建条件')
        native = request.native_prompts or {}
        if set(native) != set(compiled['native_prompts']):
            raise ValueError('提交缺少本次准备的完整模型条件，未静默补旧词')
        channel = {'anima': 'anima', 'illustrious': 'illustrious', 'z_image_turbo': 'z_image'}.get(request.render_pipeline.value, 'newbie')
        words = native.get(channel, {})
        if request.prompt != words.get('positive') or request.negative_prompt != words.get('negative', ''):
            raise ValueError('实际正负文本与选择的模型通道不一致')
        source_channel = {'anima_newbie': 'anima', 'illustrious_newbie': 'illustrious', 'z_newbie': 'z_image'}.get(request.render_pipeline.value)
        if source_channel:
            source = native[source_channel]
            if (request.source_prompt or request.anima_prompt) != source.get('positive') or request.source_negative_prompt != source.get('negative', ''):
                raise ValueError('实际源阶段正负文本与本次模型通道不一致')
        # A different frame/resource/parameter choice needs a new preparation;
        # it must not silently reuse reasoning authored for another input.
        submitted = request.model_dump(mode='json')
        config = record.get('typed_config')
        if config is None:
            # Persisted ready records created before this typed comparison keep
            # their original evidence; validate their raw values on read.
            config = typed_generation_config(record.get('effective_config', {}))
        if any(fingerprint(submitted.get(key)) != fingerprint(value) for key, value in config.items()):
            raise ValueError('生成配置与准备快照不一致，请重新准备')
        return record

    async def submit_generation(self, request: GenerateRequest, jobs):
        """Associate one public first-job submission with a ready revision.

        This is not machine-wide exactly-once execution. A persisted reservation
        prevents blind redispatch after an unknown response or process crash.
        Internal experiments and iteration continue to call jobs.submit directly.
        """
        if request.preparation_id is None or request.preparation_revision is None:
            raise ValueError('关联准备的提交需要准备 ID 和版本')
        async with self._submission_locks.setdefault(request.preparation_id, asyncio.Lock()):
            record = self._validate_generation(request)
            digest = fingerprint(request.model_dump(mode='json'))
            submissions = record.setdefault('generation_submissions', {})
            key = str(request.preparation_revision)
            previous = submissions.get(key)
            if previous:
                if previous.get('kind', 'generation') != 'generation':
                    raise ValueError('本准备版本已用于迭代提交；不能再创建普通图片任务，请重建或重新准备')
                if previous.get('request_fingerprint') != digest:
                    raise ValueError('本准备版本已有另一份提交输入；不会复用任务或重复生图，请重建或重新准备')
                if previous.get('status') == 'submitted':
                    job = jobs.get(previous.get('job_id'))
                    if job is None:
                        raise ValueError('本版本已关联图片任务，但任务记录暂不可读取；没有重新提交')
                    return job
                if previous.get('status') in {'submitting', 'unknown'}:
                    raise ValueError('本版本提交结果尚未确认；没有自动重复派发，请核对已有任务')
            receipt = {'kind': 'generation', 'revision': request.preparation_revision, 'request_fingerprint': digest,
                       'status': 'submitting', 'job_id': None}
            submissions[key] = receipt
            self._save(record)  # Reserve durably *before* any queue mutation.
            known_ids = {job.id for job in jobs.list()}
            try:
                job = await jobs.submit(request)
            except BaseException as exc:
                accepted = [job for job in jobs.list() if job.id not in known_ids
                            and job.request.get('preparation_id') == request.preparation_id
                            and job.request.get('preparation_revision') == request.preparation_revision]
                if len(accepted) == 1:
                    self._record_generation(record, receipt, accepted[0])
                else:
                    receipt.update(status='rejected' if isinstance(exc, ValueError) and not accepted else 'unknown',
                                   error=f'{type(exc).__name__}: {exc}')
                    self._save(record)
                raise
            self._record_generation(record, receipt, job)
            return job

    async def submit_iteration(self, request: GenerateRequest, controller, allowed_models: list[str],
                               budget, *, autonomous_planning: bool = False):
        """Associate one prepared revision with an iteration, without a first job.

        Ordinary generation and iteration share the same durable reservation
        and lock. The controller alone creates any image jobs later. A replay
        returns the associated run as it is; it never resumes it or changes its
        planning mode. An unknown outcome is not permission to launch again.
        """
        from .iteration_controller import IterationBudget

        if request.preparation_id is None or request.preparation_revision is None:
            raise ValueError('关联准备的迭代提交需要准备 ID 和版本')
        if type(autonomous_planning) is not bool:
            raise ValueError('自主规划必须由本次提交明确选择 true 或 false')
        if (not isinstance(allowed_models, list) or not allowed_models
                or any(not isinstance(model, str) or not model for model in allowed_models)
                or len(set(allowed_models)) != len(allowed_models)):
            raise ValueError('必须提供不重复的可用模型列表')
        budget = IterationBudget.model_validate(
            budget.model_dump(mode='json') if isinstance(budget, IterationBudget) else budget)
        launch_input = {'request': request.model_dump(mode='json'),
                        'allowed_models': list(allowed_models), 'budget': budget.model_dump(mode='json'),
                        'autonomous_planning': autonomous_planning}
        digest = fingerprint(launch_input)
        async with self._submission_locks.setdefault(request.preparation_id, asyncio.Lock()):
            record = self._validate_generation(request)
            submissions = record.setdefault('generation_submissions', {})
            key = str(request.preparation_revision)
            previous = submissions.get(key)
            if previous:
                if previous.get('kind', 'generation') != 'iteration':
                    raise ValueError('本准备版本已用于普通图片提交；不能再创建迭代任务，请重建或重新准备')
                if previous.get('request_fingerprint') != digest:
                    raise ValueError('本准备版本已有另一份迭代输入；不会复用任务或重复创建，请重建或重新准备')
                if previous.get('status') == 'submitted':
                    run = controller.store.runs.get(previous.get('run_id'))
                    if run is None:
                        raise ValueError('本版本已关联迭代任务，但任务记录暂不可读取；没有重新创建')
                    if fingerprint(run.base_request) != fingerprint(launch_input['request']):
                        raise ValueError('关联迭代任务与本版本完整输入不一致；没有复用或重新创建')
                    return run
                if previous.get('status') in {'submitting', 'unknown'}:
                    raise ValueError('本版本迭代创建结果尚未确认；没有自动重复创建，请核对已有任务')
            receipt = {'kind': 'iteration', 'revision': request.preparation_revision,
                       'request_fingerprint': digest, 'status': 'submitting', 'run_id': None,
                       'allowed_models': launch_input['allowed_models'], 'budget': launch_input['budget'],
                       'autonomous_planning': autonomous_planning}
            submissions[key] = receipt
            self._save(record)  # Reserve before controller persistence/dispatch eligibility.
            known_ids = set(controller.store.runs)
            try:
                run = controller.launch(request, list(allowed_models), budget,
                                        autonomous_planning=autonomous_planning)
            except BaseException as exc:
                # IterationStore registers a run only after its durable save.
                # Observe a single matching new record after response loss; do
                # not adopt another preparation's run or relaunch on uncertainty.
                accepted = [run for identifier, run in controller.store.runs.items()
                            if identifier not in known_ids
                            and fingerprint(run.base_request) == fingerprint(launch_input['request'])
                            and run.allowed_models == launch_input['allowed_models']
                            and run.budget.model_dump(mode='json') == launch_input['budget']
                            and getattr(run, 'autonomous_planning', False) == autonomous_planning]
                if len(accepted) == 1:
                    self._record_iteration(record, receipt, accepted[0], controller)
                else:
                    receipt.update(status='rejected' if isinstance(exc, ValueError) and not accepted else 'unknown',
                                   error=f'{type(exc).__name__}: {exc}')
                    self._save(record)
                raise
            self._record_iteration(record, receipt, run, controller)
            return run

    def _record_iteration(self, record: dict, receipt: dict, run, controller) -> None:
        receipt.update(status='submitted', run_id=run.id)
        record.update(iteration_submitted=True, submitted_iteration_run_id=run.id,
                      detail='本版本已关联迭代任务；重复相同提交只返回该任务，不会另建或恢复任务。')
        try:
            self._save(record)
        except Exception as exc:
            # A durable run is not a durable association receipt. Close its
            # dispatch gate before yielding so a lost ack cannot leave an
            # invisible autonomous task generating images. Neither this
            # in-memory marker nor a later replay may claim a committed ack.
            receipt.update(status='unknown', error=f'{type(exc).__name__}: {exc}')
            record.update(iteration_submitted=False,
                          detail='迭代任务已创建，但关联收据保存失败；不会重复创建，请核对关联任务。')
            if run.status == 'running':
                try:
                    controller.pause(run, 'preparation_receipt_storage_failed')
                except Exception as pause_error:
                    receipt['pause_error'] = f'{type(pause_error).__name__}: {pause_error}'
            if run.status != 'running':
                record['detail'] += '该任务已停止后续派发。'
            else:
                record['detail'] += '未能确认停止派发，请使用关联任务的暂停或取消操作。'
            try:
                self._save(record)
            except Exception:
                pass  # The original durable reservation still forbids replay.
            raise

    def _record_generation(self, record: dict, receipt: dict, job) -> None:
        receipt.update(status='submitted', job_id=job.id)
        record.update(generation_submitted=True, submitted_job_id=job.id,
                      detail='本版本已关联图片任务；重复相同提交会返回该任务，不会再次生图。')
        self._save(record)

    async def stop(self):
        for event in self.cancellations.values():
            event.set()
        tasks = [task for task in self.tasks.values() if not task.done()]
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
