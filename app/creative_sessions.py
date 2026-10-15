"""Daily creative sessions sharing existing inference, jobs, works and edit tools.

Restart never replays a job. A returned plan or observation remains unverified;
the original and user feedback are not rewritten into model-authored facts.
"""
from __future__ import annotations

import asyncio
from copy import deepcopy
from dataclasses import asdict
from dataclasses import replace
from datetime import UTC, datetime
import hashlib
import json
import re
from pathlib import Path
import threading
import time
from typing import Literal
import uuid

from pydantic import BaseModel, Field, ValidationError, model_validator

from .config import PROJECT_ROOT
from .creative_support import (SUPPORT_SYSTEM, SupportPlan, CanonicalPlanView, NativeScene, compile_support, compile_picture_support, check_spatial_support,
    render_support, support_input, support_runtime, picture_authoring_input)
from .picture_event import (PictureEvent, PICTURE_EVENT_SCHEMA, CanonicalPictureChoice,
    CANONICAL_PICTURE_SCHEMA, CANONICAL_PICTURE_SYSTEM_PROMPT,
    project_canonical_picture, canonical_projection_receipt, SINGLE_PICTURE_SCOPE,
    canonical_language_scope, apply_canonical_language)
from .region_edit import RegionEditOptions, generation_ancestor, prepare_region_edit, protected_inputs
from .grounded_region import (RegionLocalization, LOCALIZATION_SYSTEM,
    bind_localization, rasterize_localization, png_data_url)
from .image_handoff import prepare_handoff, prepare_visual_material_handoff, verify_handoff_input
from .local_authoring_runtime import LocalAuthoringError, RuntimeConfig, VisualImageInput
from .local_iteration_planner import local_visual_runtime
from .sampling_profiles import PROFILES
from .schemas import GenerateRequest, NewbieCompileRequest
from .visual_material import register_visual_material
from .art_direction import authoring_input
from .creative_continuity_context import (freeze_continuation_source, validate_continuation_source,
    continuation_model_material, _raw_words, _artwork_identity)
from .creative_result_review import (ResultComparison, RESULT_COMPARISON_SYSTEM,
    LOCAL_RESULT_COMPARISON_SYSTEM, prepare_result_pair, prepare_local_result_pair,
    LOCAL_VISIBLE_DELTA_CONTRACT, LOCAL_VISIBLE_DELTA_SYSTEM, local_visible_delta_input,
    FULL_VISIBLE_DELTA_CONTRACT, FULL_VISIBLE_DELTA_SYSTEM, full_visible_delta_input)
from .creative_reference import (ReferenceNextChoice, REFERENCE_CHOICE_SYSTEM,
    REFERENCE_CHOICE_CONTRACT, REFERENCE_PURPOSE_VIEW, REFERENCE_CONTENT_VIEW, bind_reference_context, reference_model_material,
    resolve_reference_choice, reference_choice_system, FULL_REFERENCE_CHOICE_CONTRACT, bind_full_reference_context)
from .creative_reference_content import compact_reference_content
from .creative_art_decision import ART_DECISION_VIEW, ART_DECISION_SYSTEM, art_decision_material
from .creative_pixel_observation import (PIXEL_OBSERVATION_CONTRACT, PIXEL_OBSERVATION_SYSTEM,
    pixel_observation_input)
from .creative_art_support import ResearchArtSupport, freeze_research_support, validate_research_support
from .research_pixel_support import ResearchPixelUse, prepare_research_pixel_handoff
from .relational_picture import (CORE_SYSTEM, FINISH_SYSTEM, RelationalCore, RelationalFinish,
    merge_relational_picture, relational_receipt)
from .prose_picture import (PROSE_PICTURE_SYSTEM, ProsePictureChoice,
    prose_picture_event, prose_picture_receipt, prose_language_scope, apply_prose_language)
from .creative_starting_choice import (StartingModelChoice, STARTING_CHOICE_SYSTEM,
    prepare_starting_choice, apply_starting_choice, compatible_choice_material, ParameterSource,
    _file_evidence, _recipe_compatibility, _lora_compatibility, _node_evidence)
from .creative_picture_decision import (NextChoice, InitialProseDecision, NextProseDecision,
    ReferenceProseDecision, INITIAL_PROSE_DECISION_SYSTEM, NEXT_PROSE_DECISION_SYSTEM,
    REFERENCE_PROSE_DECISION_SYSTEM)
from .creative_research_decision import (ResearchProseDecision,
    RESEARCH_PROSE_DECISION_PROFILE, RESEARCH_PROSE_DECISION_SYSTEM)
from .workflow import build_region_repair_workflow
from .model_catalog import model_catalog
from .creative_research_context import load_creative_research_context
from .creative_cycle_context import cycle_learning_context, cycle_history_view, completed_candidate_context
from .creative_request_understanding import (RequestReading, REQUEST_READING_SYSTEM,
    normalize_request_reading, request_reading_material, request_reading_model_material,
    classify_source_consumer, attach_request_reading)
from .creative_directions import (DirectionPlanProposal, DIRECTION_PLAN_SYSTEM, DIRECTION_COMPARISON_SYSTEM,
    FIXED_PICTURES_CONTRACT, fixed_picture_schema, fixed_picture_profile,
    direction_request, direction_planning_input, allocate_direction_slots,
    direction_assignment as make_direction_assignment)


class CreativeRequest(NewbieCompileRequest):
    model_config = {'extra': 'forbid'}
    request_id: str = Field(pattern=r'^[a-zA-Z0-9_-]{8,80}$')
    # Explicit new QA records only; absent in ordinary/legacy request identity.
    initiator: Literal['assistant_validation'] | None = None
    config: dict = Field(default_factory=dict)
    parameter_sources: dict[Literal['steps', 'cfg', 'sampler', 'scheduler', 'shift', 'cfg_rescale'], ParameterSource] | None = None
    auto_generate: bool = True
    support_mode: Literal['auto', 'text', 'sketch', 'compare'] = 'auto'
    authoring_profile: Literal['local_4b', 'local_4b_unmapped_bounded', 'local_7b', 'local_7b_bounded'] = 'local_4b'
    picture_authoring_backend: Literal['text', 'paired_visual'] = 'text'
    picture_authoring_method: Literal['prose', 'whole', 'relational_steps'] = 'prose'
    # New web requests opt in explicitly. Frozen requests keep their exact
    # author/selector contracts and never acquire new calls on restore.
    joint_picture_decision: bool = False
    # New web requests choose a catalogue/optional-read contract. The default
    # preserves frozen API requests and their actual original model inputs.
    research_mode: Literal['legacy_full', 'catalog'] = 'legacy_full'
    parameter_scope: Literal['initial', 'all'] = 'all'
    understand_request: bool = True
    independent_candidate_planning: bool = True
    # Only new creative records opt into the measured, exact local pair.
    # Legacy records without the field still use the standard factory below.
    visual_profile: Literal['standard', 'bounded_4b_visual_none'] = 'bounded_4b_visual_none'
    research_support: ResearchArtSupport | None = None
    # Seeing a source is not permission to initialise sampling from it. This
    # separate, explicit whole-frame use also works with a text-only author.
    research_pixel_use: ResearchPixelUse | None = None
    max_jobs: int = Field(default=6, ge=1, le=6)
    max_calls: int = Field(default=8, ge=1, le=8)
    max_active_seconds: int = Field(default=3600, ge=60, le=3600)
    result_guided: bool = True

    @model_validator(mode='after')
    def explicit_visual_support(self):
        if self.research_mode == 'catalog' and not self.joint_picture_decision:
            raise ValueError('按需研究读取使用共同画面消费者；其他旧方法不隐式升级。')
        if self.joint_picture_decision and (
                self.support_mode != 'auto' or self.picture_authoring_method != 'prose'
                or self.picture_authoring_backend != 'text'):
            raise ValueError('共同画面决定使用auto完整文字消费者；其他既有方法保持自己的执行合同。')
        if self.research_pixel_use is not None and (self.research_support is None or self.support_mode != 'auto'):
            raise ValueError('研究像素采用需要明确选定原图及auto完整消费者，不自动升级看图用途')
        if self.picture_authoring_backend == 'paired_visual':
            if self.research_support is None or self.support_mode != 'auto':
                raise ValueError('配对视觉构思需显式选择一件带用途研究原图及auto完整消费者')
        elif self.research_support is not None and self.research_pixel_use is None:
            raise ValueError('研究原图需显式选择paired_visual构思，不静默替换文字模型')
        if self.support_mode != 'auto':
            if 'picture_authoring_method' not in self.model_fields_set:
                # Explicit legacy support modes keep their existing consumer.
                object.__setattr__(self, 'picture_authoring_method', 'whole')
            elif self.picture_authoring_method != 'whole':
                raise ValueError('完整描述与关系构思需auto完整消费者，不缩减旧方法对照')
        return self


class CreativeRegionEdit(RegionEditOptions):
    model_config = {'extra': 'forbid'}


class CreativeAction(BaseModel):
    model_config = {'extra': 'forbid'}
    request_id: str = Field(pattern=r'^[a-zA-Z0-9_-]{8,80}$')
    expected_revision: int = Field(ge=1)
    action: Literal['generate', 'continue', 'regenerate', 'pause', 'resume', 'stop', 'select', 'revert', 'prepare_native', 'local_edit', 'complete_comparison']
    artwork_id: str | None = None
    feedback: str = Field(default='', max_length=8000)
    native_override: dict[str, str] | None = None
    max_calls: int | None = Field(default=None, ge=1, le=8)
    region_edit: CreativeRegionEdit | None = None
    initiator: Literal['user', 'assistant_validation'] = 'user'
    comparison_turn_id: str | None = Field(default=None, pattern=r'^[A-Za-z0-9_-]{1,80}$')

    @model_validator(mode='after')
    def validation_is_not_user_feedback(self):
        if self.action == 'complete_comparison':
            if (not self.comparison_turn_id or self.artwork_id is not None or self.feedback
                    or self.native_override is not None or self.region_edit is not None
                    or self.max_calls is not None):
                raise ValueError('补完比较只绑定已保存回合，不评价、选图、覆盖条件或扩大预算。')
            return self
        if self.comparison_turn_id is not None:
            raise ValueError('比较回合标识只用于明确补完比较。')
        if self.initiator == 'assistant_validation':
            if self.action in {'select', 'revert'}:
                if (not self.artwork_id or self.feedback or self.native_override is not None
                        or self.region_edit is not None or self.max_calls is not None):
                    raise ValueError('助手验证选用仅指定版本，不写用户评价、改条件或扩额度。')
            elif self.action == 'resume':
                if (self.artwork_id is not None or self.feedback or self.native_override is not None
                        or self.region_edit is not None or self.max_calls is not None):
                    raise ValueError('助手验证恢复只接续原操作，不改来源、条件或额度。')
            elif self.action in {'pause', 'stop'}:
                if (self.artwork_id is not None or self.feedback or self.native_override is not None
                        or self.region_edit is not None or self.max_calls is not None):
                    raise ValueError('助手验证暂停／停止只控制本任务，不评价、选图、覆盖条件或调整额度。')
            elif self.action in {'continue', 'regenerate'}:
                if self.native_override is not None or self.region_edit is not None or self.max_calls is not None:
                    raise ValueError('助手验证接续／重新构思只提供有来源的验证指令，不覆盖原生条件或扩额度。')
            elif self.action == 'local_edit':
                if (not self.artwork_id or self.region_edit is None or self.native_override is not None
                        or self.max_calls is not None):
                    raise ValueError('助手验证局部操作需要明确原图与遮罩，不代写用户评价或扩额度。')
            elif (self.action not in {'prepare_native', 'generate'} or self.feedback
                    or self.native_override is not None or self.region_edit is not None
                    or self.artwork_id is not None
                    or (self.action == 'generate' and self.max_calls is not None)):
                raise ValueError('助手验证准备／生成不代写用户评价、覆盖、圈选或选图；生成不调整调用额度。')
        return self


class PixelObservation(BaseModel):
    model_config = {'extra': 'forbid'}
    visible_relationships: str = Field(max_length=2000)
    potential_value: str = Field(max_length=1200)
    problems: list[str] = Field(max_length=8)
    uncertain: list[str] = Field(max_length=8)


class CreativeSessions:
    def __init__(self, settings, jobs, artworks, *, root=None, runtime_factory=None,
                 visual_factory=None, clock=None):
        self.settings, self.jobs, self.artworks = settings, jobs, artworks
        self.root = Path(root or PROJECT_ROOT / 'data/creative_sessions')
        self.root.mkdir(parents=True, exist_ok=True)
        self.runtime_factory = runtime_factory
        self.visual_factory = visual_factory
        self.clock = clock or time.monotonic
        self.records, self.tasks, self.events, self.active_started = {}, {}, {}, {}
        self._serial = asyncio.Lock()
        for path in self.root.glob('*.json'):
            try:
                record = json.loads(path.read_text(encoding='utf-8'))
                # This metadata is an execution fallback, not a migration of
                # the historical input or permission to rerun its author/job.
                record['picture_authoring_method'] = record['input_snapshot'].get(
                    'picture_authoring_method', 'whole')
                if record['status'] in {'queued', 'running'}:
                    record.update(status='paused', stage='interrupted',
                        detail='服务重启中断了创作；已有任务和原图保留，没有自动重派。')
                    completion = (record.get('comparison_completions') or {}).get(
                        record.get('last_comparison_completion_request_id'))
                    # Resume the interrupted tail, not the original author/
                    # generation driver. A returned comparison stays reusable;
                    # an entered call with no receipt is still not resubmitted.
                    if (completion and completion.get('status') in {'accepted', 'calling'}
                            and completion.get('original_operation') == record.get('operation')
                            and completion.get('turn_id') == (record.get('result_cycle') or {}).get(
                                'active_turn_id')):
                        completion['status'] = 'paused'
                    self._save(record)
                self.records[record['id']] = record
                self.events[record['id']] = threading.Event()
            except (OSError, ValueError, KeyError):
                continue

    def _save(self, record):
        record['updated_at'] = datetime.now(UTC).isoformat()
        path = self.root / (record['id'] + '.json')
        temporary = path.with_suffix('.json.tmp')
        temporary.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding='utf-8')
        temporary.replace(path)

    @staticmethod
    def _record_request(record):
        snapshot = deepcopy(record['input_snapshot'])
        snapshot.setdefault('picture_authoring_method',
            record.get('picture_authoring_method', 'whole'))
        snapshot.setdefault('understand_request', False)
        snapshot.setdefault('independent_candidate_planning', False)
        return CreativeRequest.model_validate(snapshot)

    def get(self, identifier):
        record = self.records.get(identifier)
        if record is None:
            return None
        value = deepcopy(record)
        task = self.tasks.get(identifier)
        if record['status'] in {'paused', 'failed', 'cancelled', 'ready'} and (task is None or task.done()):
            recovery = self._confirmed_terminal_active(record)
            if recovery is not None:
                self._apply_terminal_cancellation(value, recovery)
                value['terminal_active_job_recovery'] = {
                    **recovery['receipt'], 'scope': 'read_only_view_of_verified_original_cancellation'}
        active = self.jobs.get(value.get('active_job_id') or '')
        value['active_job'] = asdict(active) if active else None
        for candidate in value['candidates']:
            job = self.jobs.get(candidate['job_id'])
            if job:
                candidate.update(status=job.status,
                    image_url=job.images[0] if job.images else None,
                    file_path=job.files[0] if job.files else None,
                    error=job.error)
        used = self._used_budget(record)
        value['remaining_budget'] = {name: record['limits'][name] - used[name]
                                    for name in ('jobs', 'calls')}
        value['remaining_budget']['active_seconds'] = max(0,
            record['limits']['active_seconds'] - used['active_seconds'])
        # Capability/declaration, not permission to bypass this task's state or
        # receipts. Older API workers omit this field, so their frontend must
        # continue using the old cumulative remaining-budget rules.
        try:
            value['next_operation_budget'] = {
                'allows_user_new_round': True, **self._declared_operation_budget(record)}
        except ValueError as exc:
            value['next_operation_budget'] = {'allows_user_new_round': False, 'reason': str(exc)}
        value['saved_prose_preparation'] = self._saved_prose_capability(record)
        value['comparison_completion'] = self._comparison_completion_capability(record)
        return value

    def _confirmed_terminal_active(self, record):
        """Use the original durable cancellation, never backend idle guesses."""
        identifier = record.get('active_job_id')
        if not identifier or record.get('cancel_error'):
            return None
        job = self.jobs.get(identifier)
        verify = getattr(self.jobs, 'closed_cancellation_evidence', None)
        if job is None or job.status != 'cancelled' or job.cancel_error or verify is None:
            return None
        try:
            dispatches = [item for item in record.get('generation_dispatches', [])
                if item.get('job_id') == identifier]
            candidates = [item for item in record['candidates'] if item['job_id'] == identifier]
            if len(dispatches) != 1 or len(candidates) != 1:
                return None
            dispatch = self._generation_dispatch_for_job(record, job)
            if dispatch is not dispatches[0] or candidates[0].get('dispatch_id') != dispatch['id']:
                return None
            evidence = verify(identifier)
            if (evidence['job_id'] != identifier
                    or evidence['request_sha256'] != dispatch['job_request_sha256']):
                return None
            return {'job': job, 'receipt': {'job_id': identifier, 'dispatch_id': dispatch['id'],
                'job_request_sha256': evidence['request_sha256'],
                'engine_receipt_sha256': evidence['record_sha256']}}
        except Exception:
            return None  # Unknown/mismatched evidence keeps the original fence.

    @staticmethod
    def _apply_terminal_cancellation(record, recovery):
        job = recovery['job']
        candidate = next(item for item in record['candidates'] if item['job_id'] == job.id)
        dispatch = next(item for item in record['generation_dispatches']
            if item['id'] == recovery['receipt']['dispatch_id'])
        record['active_job_id'] = None
        candidate.update(status='cancelled', error=job.error)
        dispatch.update(completed=True, status='cancelled')
        pending = record.get('pending_methods')
        if pending and pending[0] == candidate['method']:
            pending.pop(0)

    def _commit_terminal_cancellation(self, record, *, origin):
        recovery = self._confirmed_terminal_active(record)
        if recovery is None:
            return False
        previous = deepcopy(record)
        self._apply_terminal_cancellation(record, recovery)
        record.setdefault('terminal_job_recoveries', []).append({
            **recovery['receipt'], 'origin': origin,
            'scope': 'original_cancelled_dispatch_closed_no_new_submission'})
        try:
            self._save(record)
        except BaseException:
            record.clear()
            record.update(previous)
            raise
        return True

    def _saved_prose_source(self, record):
        """Validate the complete initial author return; never form a new idea."""
        request = self._record_request(record)
        if request.picture_authoring_method == 'whole':
            return self._saved_canonical_source(record)
        if request.picture_authoring_method != 'prose' or request.support_mode != 'auto':
            raise ValueError('只支持有完整原始作者操作的初始prose原稿；其他阶段未恢复。')
        operation = record.get('operation') or {}
        if operation.get('purpose') == 'prepare_native':
            identifier = (operation.get('action') or {}).get('request_id')
            entry = (record.get('native_preparations') or {}).get(identifier) or {}
            operation = entry.get('source_operation') or {}
            if entry.get('source_checkpoint') != record.get('prose_preparation'):
                raise ValueError('保存原稿的原始checkpoint与准备操作来源不一致。')
        if operation != {'purpose': 'initial', 'action': None}:
            raise ValueError('缺少可核对的初始作者操作；未把后继或未知原稿当作可恢复初稿。')
        checkpoint = record.get('prose_preparation') or {}
        context = record.get('picture_authoring_context')
        if not isinstance(context, dict) or not isinstance(checkpoint.get('raw'), dict):
            raise ValueError('缺少完整作者原稿或当时的完整输入材料。')
        author_operation_key = self._request_digest(operation)
        if context.get('direction_assignment') is not None:
            plan = record.get('direction_plan')
            if plan is None or plan['operation_key'] != self._request_digest(operation):
                raise ValueError('保存的独立方向原稿缺少原始集合操作绑定。')
            self._validate_direction_plan(record, plan)
            slot = next((item for item in plan['slots'] if item['id'] == plan.get('active_slot_id')), None)
            if slot is None or context['direction_assignment'] != make_direction_assignment(plan, slot):
                raise ValueError('保存的原稿不是当前已分配的独立方向。')
            author_operation_key = self._request_digest({'operation': operation, 'direction_id': slot['id']})
        raw = ProsePictureChoice.model_validate(checkpoint['raw'])
        if record.get('prose_picture_choice') != checkpoint['raw']:
            raise ValueError('当前原稿与已保存的完整作者返回不一致。')
        calls = record['calls']
        author_index = checkpoint.get('author_call_index')
        expected_author_return = checkpoint['raw']
        if checkpoint.get('joint_decision_contract') is not None:
            self._validate_joint_initial_research(record)
            joint = self._adopted_initial_holder(record)
            joint_model = {InitialProseDecision.__name__: InitialProseDecision,
                ResearchProseDecision.__name__: ResearchProseDecision}.get(joint.get('contract'))
            if joint_model is None:
                raise ValueError('共同原稿的消费者未知；不升级旧合同。')
            joint_schema = joint_model.model_json_schema()
            if (not self._joint_enabled(record)
                    or joint.get('call_index') != author_index
                    or joint.get('input_binding_sha256') != checkpoint.get('joint_decision_input_sha256')
                    or joint.get('prose_checkpoint') != checkpoint
                    or (joint.get('raw') or {}).get('picture') != checkpoint['raw']):
                raise ValueError('共同原稿缺少同操作的完整决定与精确画面投影；未借稿。')
            expected_binding = self._request_digest({'system_prompt': joint.get('system_prompt'),
                'data': joint.get('model_input'), 'schema': joint_schema})
            if (joint.get('response_schema_sha256') != self._request_digest(joint_schema)
                    or joint.get('input_binding_sha256') != expected_binding
                    or (joint.get('model_input') or {}).get('original_text') != request.sentence
                    or type(author_index) is not int or not 0 <= author_index < len(calls)
                    or calls[author_index].get('input_binding_sha256') != expected_binding
                    or calls[author_index].get('response_schema_sha256') != self._request_digest(joint_schema)
                    or calls[author_index].get('joint_picture_contract') != joint_model.__name__):
                raise ValueError('共同原稿的实际输入或调用绑定改变；未借用原返回。')
            joint_model.model_validate(joint['raw'])
            expected_author_return = joint['raw']
        if (type(author_index) is not int or not 0 <= author_index < len(calls)
                or calls[author_index].get('status') != 'returned'
                or calls[author_index].get('purpose') != 'composing'
                or calls[author_index].get('data') != expected_author_return):
            raise ValueError('作者调用缺少对应的完整返回，不能重接或重跑作者。')
        config = deepcopy(request.config)
        if config.get('render_pipeline') == 'adaptive':
            start = record.get('starting_choice') or {}
            if start.get('operation_key') != author_operation_key:
                raise ValueError('初始选路原操作与当前完整原稿不一致。')
            config, receipt = apply_starting_choice(config, start.get('choice'), start['context'])
            if config != start.get('selected_config') or receipt != start.get('receipt'):
                raise ValueError('初始选路配置或回执已变化；不重新选择或合并默认参数。')
            if start.get('choice') is not None:
                index = start.get('call_index')
                start_return = (calls[index].get('data', {}).get('starting_choice')
                    if type(index) is int and 0 <= index < len(calls)
                    and start.get('joint_contract') in {InitialProseDecision.__name__, ResearchProseDecision.__name__}
                    else calls[index].get('data') if type(index) is int and 0 <= index < len(calls) else None)
                if (type(index) is not int or not 0 <= index < len(calls)
                        or calls[index].get('status') != 'returned'
                        or start_return != start['choice']):
                    raise ValueError('初始选择没有同源的完整返回。')
        if (context.get('selected_config') != config
                or checkpoint.get('operation_key') != author_operation_key
                or checkpoint.get('binding_sha256') != self._request_digest({
                    'operation': operation, 'authoring_input': context, 'selected_config': config})):
            raise ValueError('完整作者输入、真实配置或原操作绑定不一致。')
        current = authoring_input(request, config, self.settings)
        for field in ('original_text', 'original_sha256', 'drawing_form', 'identity_binding',
                      'knowledge', 'frame', 'selected_config', 'full_style_recipes'):
            if current.get(field) != context.get(field):
                raise ValueError(f'原句或当前完整创作条件已变化：{field}；未沿用旧原稿。')
        if record.get('preparation_recipe_snapshot') != current['full_style_recipes']:
            raise ValueError('完整配方快照与原稿材料不一致。')
        self._validate_research_support(record)
        target = {'anima_turbo': 'anima'}.get(config.get('model_id', 'newbie'), config.get('model_id', 'newbie'))
        if target not in PROFILES:
            raise ValueError('保存的原稿没有可核对的单模型原生消费者。')
        GenerateRequest.model_validate({**config, 'prompt': 'saved prose configuration inspection only',
            'render_pipeline': target, 'parameter_policy': 'explicit'})
        _recipe_compatibility(config, target, current['full_style_recipes'])
        _lora_compatibility(config, target, self.settings)
        _file_evidence(self.settings, target, model_catalog(self.settings))
        language_checkpoint = checkpoint
        language_index = checkpoint.get('language_call_index')
        if checkpoint.get('translated') is not None:
            if (type(language_index) is not int or not 0 <= language_index < len(calls)
                    or calls[language_index].get('status') != 'returned'
                    or calls[language_index].get('purpose') != 'native_language'
                    or calls[language_index].get('data') != checkpoint['translated']):
                raise ValueError('已存语言稿缺少对应完整语言返回。')
            apply_prose_language(raw, checkpoint['translated'])
        elif (type(language_index) is int and language_index == len(calls)
                and context.get('direction_assignment') is not None):
            # _call records its attempt before awaiting a provider. This
            # known initial direction has only reserved a language index;
            # its original allowance ended before any language call began.
            pass
        elif (type(language_index) is int and 0 <= language_index < len(calls)
                and context.get('direction_assignment') is not None
                and any(entry.get('source_checkpoint') == checkpoint
                    and entry.get('source_operation') == operation
                    and entry.get('source_context_sha256') == self._request_digest(context)
                    and entry.get('calls_before') == language_index
                    and entry.get('language_call_index') == language_index
                    and calls[language_index].get('purpose') == 'native_language'
                    and calls[language_index].get('status') == 'returned'
                    and entry.get('translated') == calls[language_index].get('data')
                    for entry in (record.get('native_preparations') or {}).values())):
            # The old reservation is unchanged. Only the explicit preparation
            # owning this exact source consumed it; its complete return is
            # selected below alongside the other actual preparation receipts.
            pass
        elif language_index is not None:
            if (type(language_index) is not int or not 0 <= language_index < len(calls)
                    or calls[language_index].get('purpose') != 'native_language'
                    or calls[language_index].get('status') != 'failed'
                    or calls[language_index].get('data') is not None):
                raise ValueError('上次语言调用未闭合或原件不足；不能另开语言调用。')
        for entry in (record.get('native_preparations') or {}).values():
            if entry.get('source_checkpoint') != checkpoint or entry.get('language_call_index') is None:
                continue
            index = entry['language_call_index']
            if type(index) is not int or not 0 <= index < len(calls):
                raise ValueError('明确语言准备缺少原调用索引。')
            call = calls[index]
            if call.get('purpose') != 'native_language':
                raise ValueError('明确语言准备的原调用用途不一致。')
            if call.get('status') == 'returned' and entry.get('translated') == call.get('data'):
                apply_prose_language(raw, entry['translated'])
                if language_checkpoint.get('translated') is None or index > language_checkpoint['language_call_index']:
                    language_checkpoint = entry
            elif call.get('status') != 'failed' or call.get('data') is not None:
                raise ValueError('明确语言调用尚未闭合或缺少已存完整回执；不重复调用。')
        sources = prose_language_scope(raw)['field_sources']
        needs_language = ('language_response' in [sources['count'], sources['scene_narrative']]
            or 'language_response' in sources['exclusions'])
        return {'request': request, 'raw': raw, 'checkpoint': checkpoint, 'context': context,
            'config': config, 'operation': operation,
            'language_checkpoint': language_checkpoint,
            'language_new_call_required': needs_language and language_checkpoint.get('translated') is None}

    def _saved_canonical_source(self, record):
        """Reuse a returned initial single whole choice and its returned language.

        This is explicit preparation, not a replay of a failed model call. The
        exact saved model inputs bind the old returns; current source conditions
        still govern compilation. Unknown phases/materials remain unsupported.
        """
        request = self._record_request(record)
        context = record.get('picture_authoring_context')
        operation = record.get('operation') or {}
        if operation.get('purpose') == 'prepare_native':
            entry = (record.get('native_preparations') or {}).get(
                (operation.get('action') or {}).get('request_id')) or {}
            operation = entry.get('source_operation') or {}
        if (request.picture_authoring_method != 'whole' or request.support_mode != 'auto'
                or request.picture_authoring_backend != 'text' or request.research_support is not None
                or request.research_pixel_use is not None
                or operation != {'purpose': 'initial', 'action': None}
                or not isinstance(context, dict) or context.get('direction_assignment') is not None
                or context.get('selected_source') is not None or record.get('direction_plan')
                or request.config.get('render_pipeline') == 'adaptive'):
            raise ValueError('仅支持已返初始单图whole、明确起点与文字原稿；其他阶段不猜补。')
        raw = CanonicalPictureChoice.model_validate(record.get('canonical_picture_choice'))
        raw_value = raw.model_dump(mode='json')
        if record.get('picture_event') != project_canonical_picture(raw).model_dump(mode='json'):
            raise ValueError('保存的完整原稿与原始投影不一致。')
        current = authoring_input(request, request.config, self.settings)
        for field in ('original_text', 'original_sha256', 'drawing_form', 'identity_binding',
                      'knowledge', 'frame', 'selected_config', 'full_style_recipes'):
            if current.get(field) != context.get(field):
                raise ValueError(f'原句或当前完整创作条件已变化：{field}；未沿用旧原稿。')
        if record.get('preparation_recipe_snapshot') != current['full_style_recipes']:
            raise ValueError('完整配方与原稿不一致。')
        if request.understand_request:
            self._reading_checkpoint(record)
        calls = record['calls']
        indexes = [i for i, call in enumerate(calls) if call.get('purpose') == 'composing'
            and call.get('status') == 'returned' and call.get('data') == raw_value]
        if len(indexes) != 1:
            raise ValueError('完整初稿没有唯一可核对的作者返回。')
        author_index = indexes[0]
        language_index = author_index + 1
        if (language_index >= len(calls) or calls[language_index].get('purpose') != 'native_language'
                or calls[language_index].get('status') != 'returned'):
            raise ValueError('没有同一原稿的完整语言返回；不重新调用。')
        packets = []
        for index in (author_index, language_index):
            directory = calls[index].get('directory')
            if not isinstance(directory, str) or not directory:
                raise ValueError('完整返回缺少当次输入原件。')
            packets.append(json.loads((Path(directory) / 'input.json').read_text(encoding='utf-8')))
        author_packet, language_packet = packets
        schema = {**CANONICAL_PICTURE_SCHEMA, 'title': 'CanonicalPictureChoice'}
        if (author_packet.get('schema') != schema
                or author_packet.get('data') != picture_authoring_input(context)
                or language_packet.get('schema') != schema
                or (language_packet.get('data') or {}).get('original_text') != request.sentence
                or (language_packet.get('data') or {}).get('canonical_choice_to_translate') != raw_value
                or (language_packet.get('data') or {}).get('scope')
                    != 'language_conversion_only_not_new_art_or_approval'):
            raise ValueError('作者／语言返回不属于保存的完整输入；不沿用或重新调用。')
        translated = deepcopy(calls[language_index]['data'])
        apply_canonical_language(raw, translated)
        config = deepcopy(request.config)
        target = {'anima_turbo': 'anima'}.get(config.get('model_id', 'newbie'), config.get('model_id', 'newbie'))
        if target not in PROFILES:
            raise ValueError('原稿没有兼容的单模型原生消费者。')
        GenerateRequest.model_validate({**config, 'prompt': 'saved canonical inspection only',
            'render_pipeline': target, 'parameter_policy': 'explicit'})
        _recipe_compatibility(config, target, current['full_style_recipes'])
        _lora_compatibility(config, target, self.settings)
        _file_evidence(self.settings, target, model_catalog(self.settings))
        checkpoint = {'kind': 'saved_canonical_language_return_v1', 'raw': raw_value,
            'author_call_index': author_index, 'language_call_index': language_index,
            'translated': translated, 'binding_sha256': self._request_digest({
                'operation': operation, 'context': context, 'model_inputs': packets, 'config': config})}
        return {'request': request, 'raw': raw, 'checkpoint': checkpoint,
            'context': context, 'config': config, 'operation': operation,
            'language_checkpoint': checkpoint, 'language_new_call_required': False}

    def _saved_prose_capability(self, record):
        used = len(record['calls'])
        result = {'available': False, 'reason': '', 'language_new_call_required': False,
            'minimum_max_calls': max(1, used),
            'suggested_max_calls': max(1, record['limits']['calls']), 'maximum_max_calls': 8}
        try:
            if record.get('plan') is not None:
                raise ValueError('已有完整待审原生方案，按现有审阅流程操作。')
            if record['status'] not in {'failed', 'paused'}:
                raise ValueError('仅已结束失败或暂停的完整初稿可显式准备；在途、已停止或其他阶段不可用。')
            if self.tasks.get(record['id']) and not self.tasks[record['id']].done():
                raise ValueError('当前仍有在途操作。')
            self._require_new_round_boundary(record)
            if record['candidates'] or record.get('generation_request') is not None:
                raise ValueError('该记录不属于尚未派图的初始完整原稿。')
            if self._elapsed(record) >= record['limits']['active_seconds']:
                raise ValueError('累计执行时间已用尽；准备操作不重置时间。')
            source = self._saved_prose_source(record)
            if source['checkpoint'].get('kind') == 'saved_canonical_language_return_v1':
                result['choice_kind'] = 'canonical'
            result['language_new_call_required'] = source['language_new_call_required']
            result['minimum_max_calls'] = max(1, used + int(source['language_new_call_required']))
            result['suggested_max_calls'] = max(record['limits']['calls'], result['minimum_max_calls'])
            if result['minimum_max_calls'] > 8:
                raise ValueError('所需语言调用超过可明确声明的8次上限。')
            result.update(available=True, reason=('复用完整原稿；需一次明确语言调用，不重作者或初选。'
                if source['language_new_call_required'] else '复用完整原稿和已存语言条件，零新模型调用准备，不自动派图。'))
        except (ValueError, KeyError, OSError, TypeError) as exc:
            result['reason'] = str(exc) or '完整原稿的来源条件不足。'
        return result

    @staticmethod
    def _declared_operation_budget(record):
        snapshot = record['input_snapshot']
        fields = {'jobs': ('max_jobs', 1, 6), 'calls': ('max_calls', 1, 8),
                  'active_seconds': ('max_active_seconds', 60, 3600)}
        declared = {}
        for name, (field, minimum, maximum) in fields.items():
            value = snapshot.get(field, CreativeRequest.model_fields[field].default)
            if type(value) is not int or not minimum <= value <= maximum:
                raise ValueError('原创作声明的回合预算无效，没有猜测或扩大额度')
            declared[name] = value
        return declared

    def _used_budget(self, record):
        return {'jobs': len(record['candidates']) + sum(
                    bool(item.get('submission_started') and not item.get('job_id'))
                    for item in record.get('generation_dispatches', [])),
                'calls': len(record['calls']), 'active_seconds': self._elapsed(record)}

    def _require_new_round_boundary(self, record):
        if record['status'] not in {'ready', 'paused', 'failed'}:
            raise ValueError('仅已结束或暂停的创作可明确开始新回合；不会恢复已停止任务')
        if record.get('active_job_id') or record.get('cancel_error') or self.events[record['id']].is_set():
            raise ValueError('旧任务或取消回执尚未闭合，没有开始新回合')
        if self._pending_generation_dispatch(record) is not None:
            raise ValueError('旧生成派发尚未完成回执接续，没有开始新回合')
        edit = record.get('local_edit') or {}
        if (edit.get('submission_uncertain') or (edit.get('submission_started') and not edit.get('job_id'))
                or (edit.get('job_id') and not edit.get('completed'))):
            raise ValueError('旧局部修改回执未确认或尚未闭合，没有开始新回合')
        for candidate in record['candidates']:
            job = self.jobs.get(candidate['job_id'])
            if (job is None or job.status not in {'completed', 'failed', 'cancelled'}
                    or getattr(job, 'cancel_error', None)
                    or any(not item.get('completed') for item in getattr(job, 'engine_dispatches', []))):
                raise ValueError('旧候选的真实任务回执尚未闭合，没有开始新回合')
        self._declared_operation_budget(record)

    def _begin_user_budget_round(self, record, action):
        """A new explicit operation gets one bounded allowance, never a replay.

        All cumulative attempts/time and earlier errors remain. Absolute limits
        are actual already-used resources plus this original declared allowance,
        not another allowance piled on top of unspent older limits.
        """
        used = self._used_budget(record)
        declaration = self._declared_operation_budget(record)
        previous_limits = deepcopy(record['limits'])
        limits = {name: used[name] + declaration[name] for name in declaration}
        record.setdefault('budget_rounds', []).append({
            'id': uuid.uuid4().hex[:12], 'created_at': datetime.now(UTC).isoformat(),
            'origin': 'explicit_user_operation', 'action': action.model_dump(mode='json'),
            'offset': used, 'declaration': declaration, 'previous_limits': previous_limits,
            'limits': deepcopy(limits),
            'previous_operation': deepcopy(record.get('operation')),
            'previous_outcome': {name: deepcopy(record.get(name))
                for name in ('status', 'stage', 'detail', 'error', 'revision')},
            'outcomes': []})
        record['limits'] = limits

    def list(self, *, request_id=None, artwork_id=None, limit=20):
        values = sorted(self.records.values(), key=lambda v: v['created_at'], reverse=True)
        return [self.get(v['id']) for v in values if
            (request_id is None or v['request_id'] == request_id) and
            (artwork_id is None or any(c['artwork_id'] == artwork_id for c in v['candidates']))][:limit]

    async def create(self, request: CreativeRequest):
        snapshot = request.model_dump(mode='json')
        if snapshot['initiator'] is None:
            snapshot.pop('initiator')
        old = next((v for v in self.records.values() if v['request_id'] == request.request_id), None)
        if old:
            if 'result_guided' not in request.model_fields_set and 'result_guided' not in old['input_snapshot']:
                snapshot.pop('result_guided', None)
            # An omitted option must not turn a retry of an old request into a
            # different input just because the default changed on deployment.
            # Explicit changes are still refused; the old record is untouched.
            if 'visual_profile' not in request.model_fields_set:
                if 'visual_profile' in old['input_snapshot']:
                    snapshot['visual_profile'] = old['input_snapshot']['visual_profile']
                else:
                    snapshot.pop('visual_profile', None)
            if 'picture_authoring_method' not in request.model_fields_set:
                if 'picture_authoring_method' in old['input_snapshot']:
                    snapshot['picture_authoring_method'] = old['input_snapshot']['picture_authoring_method']
                else:
                    snapshot.pop('picture_authoring_method', None)
            if 'understand_request' not in request.model_fields_set:
                if 'understand_request' in old['input_snapshot']:
                    snapshot['understand_request'] = old['input_snapshot']['understand_request']
                else:
                    snapshot.pop('understand_request', None)
            if 'independent_candidate_planning' not in request.model_fields_set:
                if 'independent_candidate_planning' in old['input_snapshot']:
                    snapshot['independent_candidate_planning'] = old['input_snapshot']['independent_candidate_planning']
                else:
                    snapshot.pop('independent_candidate_planning', None)
            if ('research_pixel_use' not in request.model_fields_set
                    and 'research_pixel_use' not in old['input_snapshot']):
                snapshot.pop('research_pixel_use', None)
            for field in ('joint_picture_decision', 'parameter_scope', 'research_mode'):
                if field not in request.model_fields_set:
                    if field in old['input_snapshot']:
                        snapshot[field] = old['input_snapshot'][field]
                    else:
                        snapshot.pop(field, None)
            if old['input_snapshot'] != snapshot:
                raise ValueError('同一创作请求标识不能替换输入')
            return self.get(old['id'])
        # Do not spend scarce authoring calls before discovering an unresolved
        # dispatch restored from a previous API process. Existing request-id
        # lookup above stays read-only and never becomes a retry.
        receipt_guard = getattr(self.jobs, '_reject_restored_receipt_fence', None)
        if receipt_guard is not None:
            receipt_guard()
        # Real configuration types are checked before scarce inference.
        config = request.config
        target = str(config.get('model_id', 'newbie'))
        route = {'anima_turbo': 'anima'}.get(target, target)
        if route not in PROFILES:
            raise ValueError('创作起点需选择一个实际模型；后续协作依据图像决定，不预排固定链')
        declared_route = config.get('render_pipeline', route)
        if declared_route not in {route, 'adaptive'} or config.get('source_parameters'):
            raise ValueError('当前创作起点不解释固定桥接／源阶段配置，请选择自动起点或单个起点；原研究与旧路线入口保留')
        if config.get('batch_size', 1) != 1:
            raise ValueError('每个候选单独记录；不使用没有版本身份的批量输出')
        if request.support_mode in {'sketch', 'compare'} and (config.get('highres_enabled') or config.get('face_detail_enabled')):
            raise ValueError('显式草案对照目前是基础单阶段，请明确选择基础质量；不会悄悄关闭高细节或脸部阶段')
        configuration = GenerateRequest.model_validate({**config, 'prompt': 'configuration validation only',
            'model_id': target, 'render_pipeline': route, 'parameter_policy': 'explicit'})
        if request.research_pixel_use is not None and (
                configuration.highres_enabled or configuration.face_detail_enabled
                or configuration.hand_repair_enabled or configuration.text_clean_enabled
                or configuration.regional_conditioning is not None or configuration.image_handoff is not None
                or configuration.mask_file or configuration.repair_source_file
                or configuration.repair_source_sha256 or configuration.repair_mask_sha256
                or configuration.repair_instruction or configuration.repair_context_prompt
                or configuration.repair_notes or configuration.operation != 'generate'
                or configuration.source_parameters is not None):
            raise ValueError('显式研究像素目前只消费完整基础单阶段；不支持的已选阶段未关闭，也未加载构思模型')
        research = (freeze_research_support(request.research_support, self.artworks, self.jobs,
            output_root=Path(self.settings.output_dir)) if request.research_support is not None else None)
        identifier = uuid.uuid4().hex[:12]
        record = {'id': identifier, 'revision': 1, 'request_id': request.request_id,
            'created_at': datetime.now(UTC).isoformat(), 'input_snapshot': snapshot,
            'picture_authoring_method': request.picture_authoring_method,
            'status': 'queued', 'stage': 'queued', 'detail': '等待本地资源形成画面候选。',
            'plan': None, 'native_prompts': {}, 'generation_request': None,
            'candidates': [], 'feedback': [], 'calls': [], 'decisions': [], 'materials': [],
            'selected_artwork_id': None, 'active_job_id': None, 'error': None,
            'action_receipts': {}, 'active_seconds': 0, 'pause_requested': False, 'pending_methods': None,
            'operation': None, 'continuation': None,
            'research_support_snapshot': research,
            'limits': {'jobs': request.max_jobs, 'calls': request.max_calls,
                       'active_seconds': request.max_active_seconds}}
        if request.result_guided:
            record['result_cycle'] = {'status': 'idle', 'cursor_artwork_id': None,
                'active_turn_id': None, 'turns': [], 'stop_reason': None}
        self.records[identifier] = record
        self.events[identifier] = threading.Event()
        self._save(record)
        self._schedule(record, 'initial')
        return self.get(identifier)

    def _schedule(self, record, purpose, action=None):
        old = self.tasks.get(record['id'])
        if old and not old.done():
            raise ValueError('当前创作仍有在途操作，不重复安排')
        record['operation'] = {'purpose': purpose,
            'action': action.model_dump(mode='json') if action else None}
        self._save(record)
        self.tasks[record['id']] = asyncio.create_task(self._run(record, purpose, action),
            name='creative-' + record['id'])

    def _check(self, record):
        if self.events[record['id']].is_set():
            raise asyncio.CancelledError()
        if record['pause_requested']:
            raise InterruptedError('已暂停；保留作品及当前方案，不自动派发下一项。')
        if self._elapsed(record) >= record['limits']['active_seconds']:
            raise InterruptedError('本次累计执行时间预算耗尽，没有新派发。')

    def _elapsed(self, record):
        started = self.active_started.get(record['id'])
        return record['active_seconds'] + (self.clock() - started if started is not None else 0)

    async def _call(self, record, system, data, schema, *, images=None, runtime_profile=None, visual_input_scope=None):
        self._check(record)
        if len(record['calls']) >= record['limits']['calls']:
            raise InterruptedError('本次模型调用预算耗尽，没有自动重试。')
        call = {'purpose': record['stage'], 'status': 'started'}
        joint_schemas = {
            self._request_digest(InitialProseDecision.model_json_schema()): 'local_creative_initial_prose_decision_6703d7894',
            self._request_digest(NextProseDecision.model_json_schema()): 'local_creative_next_prose_decision_6703d7894',
            self._request_digest(ReferenceProseDecision.model_json_schema()): 'local_creative_reference_prose_decision_6703d7894',
            self._request_digest(ResearchProseDecision.model_json_schema()): RESEARCH_PROSE_DECISION_PROFILE,
        }
        joint_profile = joint_schemas.get(self._request_digest(schema))
        if joint_profile:
            call.update(input_binding_sha256=self._request_digest({'system_prompt': system, 'data': data, 'schema': schema}),
                response_schema_sha256=self._request_digest(schema), joint_picture_contract=schema['title'])
        if record['stage'] == 'understanding_request':
            call.update(input_binding_sha256=record['request_reading']['binding_sha256'],
                response_schema_sha256=self._request_digest(schema))
        elif record['stage'] == 'planning_directions':
            call.update(input_binding_sha256=record['direction_plan']['input_binding_sha256'],
                response_schema_sha256=self._request_digest(schema))
        if images and data.get('contract_version') == PIXEL_OBSERVATION_CONTRACT:
            call.update(input_contract_version=PIXEL_OBSERVATION_CONTRACT,
                input_binding_sha256=self._request_digest({'system_prompt': system, 'data': data, 'schema': schema}),
                response_schema_sha256=self._request_digest(schema))
        if images and len(images) == 2 and data.get('contract_version') in {
                FULL_VISIBLE_DELTA_CONTRACT, LOCAL_VISIBLE_DELTA_CONTRACT}:
            call.update(input_contract_version=data['contract_version'],
                input_binding_sha256=self._request_digest({'system_prompt': system, 'data': data, 'schema': schema}),
                response_schema_sha256=self._request_digest(schema))
        record['calls'].append(call)
        self._save(record)
        profile = runtime_profile or record['input_snapshot'].get('authoring_profile', 'local_4b')
        visual_profile = record['input_snapshot'].get('visual_profile', 'standard')
        pair = bool(images and len(images) == 2)
        runtime = ((self.visual_factory() if self.visual_factory else
                    local_visual_runtime(self.settings, profile='standard' if pair else visual_profile)) if images else
                   self.runtime_factory() if self.runtime_factory else support_runtime(self.settings, profile))
        if (joint_profile is not None and not images and self.runtime_factory is None
                and runtime.config.resource_profile == 'bounded_4b_none'
                and runtime.config.load_mode == 'none'):
            # The real initial joint wire measured 10,632 input tokens; its
            # complete 2,400-token author allowance plus reserve needs 13,096.
            # Allocate this consumer's capacity, not a shorter research view or
            # output. Legacy consumers and other model/visual profiles retain
            # their selected configuration and stricter resource thresholds.
            runtime.config = replace(runtime.config, ctx_size=max(runtime.config.ctx_size, 16384))
            call['ctx_size'] = runtime.config.ctx_size
        if pair and self.visual_factory is None:
            # Two unchanged originals do NOT inherit the experimental 3GiB
            # single-image admission. Use the existing standard paired runtime.
            runtime.config = replace(runtime.config, load_mode='none', log_verbosity=3,
                hard_ram_mib=1000, sustained_ram_mib=1500, sustained_ram_seconds=3)
        # Syntax compatibility is independent of the resource/loading profile.
        # Select by the complete consumer schema, never its title alone. The
        # runtime still checks the exact installed DLLs and immutable PNGs.
        native_profile = None
        if images and self.visual_factory is None:
            if len(images) == 1 and schema == PixelObservation.model_json_schema():
                native_profile = 'local_creative_observation_6703d7894'
            elif pair and schema == ResultComparison.model_json_schema():
                native_profile = 'local_creative_result_comparison_6703d7894'
            elif len(images) == 1 and schema == RegionLocalization.model_json_schema():
                native_profile = 'local_creative_region_localization_6703d7894'
        elif not images and self.runtime_factory is None and schema == NextChoice.model_json_schema():
            native_profile = 'local_creative_next_choice_6703d7894'
        elif not images and self.runtime_factory is None and schema == ReferenceNextChoice.model_json_schema():
            native_profile = 'local_creative_reference_next_choice_6703d7894'
        elif not images and self.runtime_factory is None and schema == StartingModelChoice.model_json_schema():
            native_profile = 'local_creative_starting_choice_6703d7894'
        elif not images and self.runtime_factory is None and schema == RequestReading.model_json_schema():
            native_profile = 'local_creative_request_reading_6703d7894'
        elif not images and self.runtime_factory is None and schema == DirectionPlanProposal.model_json_schema():
            native_profile = 'local_creative_direction_plan_6703d7894'
        elif not images and self.runtime_factory is None:
            native_profile = joint_profile
            for picture_count in range(1, 7):
                if schema == fixed_picture_schema(picture_count).model_json_schema():
                    native_profile = fixed_picture_profile(picture_count)
                    break
        if native_profile:
            runtime.config = replace(runtime.config, native_grammar_profile=native_profile)
            call['native_grammar_profile'] = native_profile
        if not images and self.runtime_factory is None and schema in (
                NextChoice.model_json_schema(), ReferenceNextChoice.model_json_schema()):
            # This call selects one operation, not a complete new drawing
            # specification. Reserve its own declared output budget rather
            # than the full author's 2400; input/schema remain unchanged.
            runtime.config = replace(runtime.config, max_output_tokens=1200)
            call['max_output_tokens'] = 1200
        if not images and self.runtime_factory is None and schema == StartingModelChoice.model_json_schema():
            runtime.config = replace(runtime.config, max_output_tokens=900)
            call['max_output_tokens'] = 900
        call['runtime_profile'] = ('paired_4b_visual' if pair else 'single_4b_visual') if images else profile
        if images:
            call['visual_profile'] = visual_input_scope or ('standard_two_originals_none' if pair else visual_profile)
            if visual_input_scope is not None:
                call['visual_input_scope'] = visual_input_scope
            call['image_bindings'] = [{'source_ref': x.source_ref, 'sha256': x.expected_sha256} for x in images]
        self._save(record)
        remaining = record['limits']['active_seconds'] - self._elapsed(record)
        try:
            result = await asyncio.wait_for(runtime.generate_json(system_prompt=system, data=data,
                response_schema=schema, images=images or (), cancel_event=self.events[record['id']],
                owner=record['id'], **({'host_original_text': record['input_snapshot']['sentence']}
                    if (pair and data.get('contract_version') in {LOCAL_VISIBLE_DELTA_CONTRACT, FULL_VISIBLE_DELTA_CONTRACT})
                        or (images and len(images) == 1 and data.get('contract_version') == PIXEL_OBSERVATION_CONTRACT) else {})),
                timeout=max(.01, remaining))
        except BaseException as exc:
            timed_out = isinstance(exc, TimeoutError)
            boundary = exc.__cause__ if timed_out and exc.__cause__ is not None else exc
            directory = getattr(exc, 'evidence_dir', None) or getattr(boundary, 'evidence_dir', None)
            message = ('本次构思超过会话执行时限，未取得完整方案；没有派图或自动重试。'
                       if timed_out else str(exc) or type(exc).__name__)
            error_code = 'session_deadline' if timed_out else getattr(exc, 'code', None)
            call.update(status='cancelled' if isinstance(exc, asyncio.CancelledError) else 'failed',
                        error=message, error_code=error_code,
                        directory=str(directory) if directory else None)
            for field in ('phase', 'worker_error_code', 'worker_error', 'worker_error_type'):
                value = getattr(boundary, field, None)
                if value is not None:
                    call[field] = value
            for field in ('usage', 'timings', 'finish_reason', 'max_output_tokens'):
                value = getattr(boundary, field, None)
                if value is not None:
                    call[field] = deepcopy(value)
            self._save(record)
            if timed_out:
                raise LocalAuthoringError('session_deadline', message, directory) from exc
            raise
        call.update(status='returned', directory=str(result.evidence_dir),
            data=deepcopy(result.data), usage=result.usage, timings=result.timings)
        self._save(record)
        return result

    @staticmethod
    def _request_reading_input(request):
        original = request.sentence
        return {'original_text': original,
            'original_sha256': hashlib.sha256(original.encode('utf-8')).hexdigest()}

    @staticmethod
    def _legacy_request_reading_input(request):
        """Validate saved source material without stripping or rewriting it."""
        original = request.sentence
        return {'original_text': original,
            'original_sha256': hashlib.sha256(original.encode('utf-8')).hexdigest(),
            'drawing_form': request.style_mode.value,
            'identity_selection': {
                'knowledge_pack': request.knowledge_pack,
                'knowledge_operator': request.knowledge_operator,
                'knowledge_appearance': request.knowledge_appearance,
                'scope': 'explicit_resource_selection_not_visual_facts_or_identity_certificate'},
            'frame': {'width': request.config.get('width'), 'height': request.config.get('height')},
            'scope': 'read_original_user_purpose_only_not_model_selection_or_art_direction'}

    def _reading_input_for_checkpoint(self, request, checkpoint):
        version = (checkpoint or {}).get('input_contract_version')
        if checkpoint is None or version == 'original_text_only_v1':
            return self._request_reading_input(request)
        if version is None:
            return self._legacy_request_reading_input(request)
        raise ValueError('未知来源输入合同；不改旧原件或重做读取。')

    def _reading_checkpoint(self, record):
        request = self._record_request(record)
        if not request.understand_request:
            return None  # Historical inputs never acquire a new interpretation.
        checkpoint = record.get('request_reading')
        source = self._reading_input_for_checkpoint(request, checkpoint)
        binding = self._request_digest({'input': source,
            'response_schema': RequestReading.model_json_schema()})
        if (checkpoint is None or checkpoint.get('input') != source
                or checkpoint.get('binding_sha256') != binding
                or checkpoint.get('original_text') != source['original_text']
                or checkpoint.get('original_sha256') != source['original_sha256']):
            raise ValueError('本次要求理解缺少同源原句及选择绑定；没有补猜或借用作者方案。')
        index = checkpoint.get('call_index')
        if type(index) is not int or not 0 <= index < len(record['calls']):
            raise ValueError('要求理解没有可回查的完整调用；没有重跑。')
        call = record['calls'][index]
        if (call.get('purpose') != 'understanding_request' or call.get('status') != 'returned'
                or call.get('input_binding_sha256') != binding
                or call.get('response_schema_sha256') != self._request_digest(RequestReading.model_json_schema())
                or checkpoint.get('raw') != call.get('data')):
            raise ValueError('要求理解与实际完整调用不一致；没有重跑或改写原回应。')
        request_reading_material(checkpoint)
        return checkpoint

    def _with_request_reading(self, record, data, *, saved_input=None):
        checkpoint = self._reading_checkpoint(record)
        version = (classify_source_consumer(saved_input, checkpoint)
                   if saved_input is not None and checkpoint is not None else 'quoted_hypotheses_v2')
        material = request_reading_model_material(checkpoint, version=version)
        if material is not None:
            # Original words remain distinct from fallible reader hypotheses.
            # A bound older consumer keeps its exact original source block;
            # only a fresh consumer adopts the new purpose view.
            scope = deepcopy(material)
            intent = scope.pop('creation_intent')
            data.update(request_reading_material=scope, creation_intent=intent)
        return data

    def _authoring_source_consumer(self, record, assignment, source_context):
        """Find only this author operation's saved input, never a prior draft's view."""
        context = record.get('picture_authoring_context')
        if not isinstance(context, dict) or 'request_reading_material' not in context:
            return None
        if context.get('direction_assignment') != assignment:
            return None
        selected = context.get('selected_source')
        if source_context is not None:
            if (not isinstance(selected, dict)
                    or selected.get('snapshot_sha256') != source_context['snapshot_sha256']):
                return None
        elif selected is not None:
            return None
        operation = record['operation']
        expected = {self._request_digest(operation)}
        if assignment is not None:
            expected.add(self._request_digest({'operation': operation, 'direction_id': assignment['id']}))
        continuation = record.get('continuation') or {}
        if source_context is not None and type(continuation.get('authoring_call_index')) is int:
            expected.add(self._request_digest({'operation': operation,
                'source_snapshot_sha256': source_context['snapshot_sha256'],
                'authoring_call_index': continuation['authoring_call_index']}))
        prose = record.get('prose_preparation') or {}
        if prose.get('operation_key') in expected:
            if (source_context is None
                    or prose.get('operation_key') != self._request_digest(operation)
                    or (type(continuation.get('authoring_call_index')) is int
                        and prose.get('author_call_index') == continuation['authoring_call_index'])):
                return context
        operation_binding = {'purpose': operation['purpose'],
            'action_request_id': (operation.get('action') or {}).get('request_id')}
        if assignment is not None:
            operation_binding['direction_id'] = assignment['id']
        relational = record.get('relational_picture') or {}
        if relational.get('operation_binding') == operation_binding:
            return relational['core_input']
        if assignment is not None:
            slot = next((item for item in (record.get('direction_plan') or {}).get('slots', [])
                         if item['id'] == assignment['id']), None)
            if slot is not None and slot.get('whole_author_checkpoint') is not None:
                return context
        return None

    def _attach_request_reading(self, record, actual):
        actual = attach_request_reading(actual, self._reading_checkpoint(record))
        return self._attach_direction(actual, record.get('active_direction_assignment'))

    @staticmethod
    def _attach_direction(actual, assignment):
        if assignment is None:
            return actual
        proposal = deepcopy(actual.author_proposal or {})
        proposal['direction_assignment'] = deepcopy(assignment)
        return actual.model_copy(update={'author_proposal': proposal})

    async def _ensure_request_reading(self, record, request):
        if not request.understand_request:
            return
        checkpoint = record.get('request_reading')
        source = self._reading_input_for_checkpoint(request, checkpoint)
        binding = self._request_digest({'input': source,
            'response_schema': RequestReading.model_json_schema()})
        if checkpoint is None:
            checkpoint = {'input': deepcopy(source), 'binding_sha256': binding,
                'original_text': source['original_text'], 'original_sha256': source['original_sha256'],
                'input_contract_version': 'original_text_only_v1',
                'call_index': None, 'raw': None, 'creation_intent': None, 'status': 'pending'}
            record['request_reading'] = checkpoint
            self._save(record)
        elif (checkpoint.get('input') != source or checkpoint.get('binding_sha256') != binding):
            raise ValueError('要求理解的原句或明确选择已变化；没有沿用旧解释或重做调用。')
        if checkpoint['raw'] is None:
            index = checkpoint['call_index']
            recovered = None
            if index is not None:
                if type(index) is not int or not 0 <= index <= len(record['calls']):
                    raise ValueError('要求理解的调用预约不完整；没有重跑。')
                if index < len(record['calls']):
                    call = record['calls'][index]
                    if (call.get('purpose') != 'understanding_request'
                            or call.get('status') != 'returned' or 'data' not in call
                            or call.get('input_binding_sha256') != binding
                            or call.get('response_schema_sha256') != self._request_digest(RequestReading.model_json_schema())):
                        raise ValueError('要求理解已用调用但没有同源完整回执；恢复不自动重试。')
                    recovered = deepcopy(call['data'])
            if recovered is None:
                checkpoint['call_index'] = len(record['calls'])
                record.update(stage='understanding_request',
                    detail='理解原句中的关系、情绪与候选用途；这是可纠正解释，不是画面方案或认证。')
                self._save(record)
                self._check(record)
                if len(record['calls']) >= record['limits']['calls']:
                    raise InterruptedError('本次模型调用预算耗尽，没有自动重试。')
                # Understanding is text inference. Starting the image backend
                # here consumes its RAM before any image job needs it. The
                # shared inference lease and runtime queue checks still govern
                # the call; JobManager starts Core when dispatching an image.
                result = await self._call(record, REQUEST_READING_SYSTEM, deepcopy(source),
                    RequestReading.model_json_schema())
                recovered = deepcopy(result.data)
            # Persist the exact returned source draft before quote validation.
            checkpoint['raw'] = deepcopy(recovered)
            self._save(record)
        if checkpoint['creation_intent'] is None:
            checkpoint['creation_intent'] = normalize_request_reading(
                source['original_text'], checkpoint['raw']).model_dump(mode='json')
            checkpoint['status'] = 'returned_unverified'
            self._save(record)
        self._reading_checkpoint(record)

    async def _ensure_direction_plan(self, record):
        request = self._record_request(record)
        if (not request.independent_candidate_planning or not request.understand_request
                or request.support_mode != 'auto' or record['operation']['purpose'] != 'initial'):
            return None
        await self._ensure_request_reading(record, request)
        requested = direction_request(request_reading_material(self._reading_checkpoint(record)))
        if requested is None:
            return None
        plan = record.get('direction_plan')
        max_slots = plan['max_slots'] if plan else min(6, record['limits']['jobs'])
        contract_version = plan.get('contract_version') if plan is not None else FIXED_PICTURES_CONTRACT
        full_input = support_input(request, request.config, settings=self.settings)
        self._with_request_reading(record, full_input, saved_input=plan.get('input') if plan else None)
        planning_input = direction_planning_input(full_input, requested, max_slots, contract_version=contract_version)
        binding = self._request_digest(planning_input)
        operation_key = self._request_digest(record['operation'])
        if plan is None:
            plan = {'requested_count': requested['count'], 'requested_format': requested['format'],
                'source_quote': requested['source_quote'], 'original_sha256': full_input['original_sha256'],
                'operation_key': operation_key, 'input_binding_sha256': binding,
                'input': deepcopy(planning_input), 'max_slots': max_slots,
                'call_index': None, 'raw': None, 'plan_sha256': None, 'slots': [],
                'comparison': None, 'status': 'planning', 'active_slot_id': None,
                'execution_operation_keys': [operation_key]}
            picture_count = min(requested['count'], max_slots, 6)
            plan.update(contract_version=FIXED_PICTURES_CONTRACT,
                picture_count=picture_count, picture_keys=[f'picture_{index}' for index in range(1, picture_count + 1)],
                response_schema_sha256=self._request_digest(fixed_picture_schema(picture_count).model_json_schema()))
            record['direction_plan'] = plan
            self._save(record)
        elif plan['operation_key'] != operation_key or plan['input_binding_sha256'] != binding:
            raise ValueError('集合提案的原句、操作或完整条件已变化；不重规划或借用旧方向。')
        minimum_execution = (2 if request.picture_authoring_method == 'relational_steps' else 1)
        minimum_execution += int(request.config.get('render_pipeline') == 'adaptive')
        if (plan['raw'] is None and plan['call_index'] is None
                and self.get(record['id'])['remaining_budget']['calls'] < 1 + minimum_execution):
            plan.update(status='not_started_budget',
                reason='剩余调用不足以集合规划并执行至少一个方向；不消耗规划调用，保留原单图起点，集合未展开。')
            self._save(record)
            return None
        if plan['raw'] is None:
            schema = self._direction_plan_schema(plan)
            recovered = self._returned_cycle_call(record, plan, 'call_index', schema)
            if recovered is None:
                if contract_version is None:
                    raise ValueError('旧集合没有完整原调用返回；不换消费者合同或使用新职责重规划。')
                plan['call_index'] = len(record['calls'])
                record.update(stage='planning_directions', detail='为独立作品形成集合级表现提案；不把多个方向合进一图。')
                self._save(record)
                result = await self._call(record, DIRECTION_PLAN_SYSTEM, deepcopy(planning_input),
                    schema.model_json_schema())
                recovered = deepcopy(result.data)
            plan['raw'] = deepcopy(recovered)
            self._save(record)
        self._validate_direction_plan(record, plan)
        if plan['plan_sha256'] is None:
            plan['plan_sha256'] = self._request_digest(plan['raw'])
            plan['slots'] = allocate_direction_slots(plan['raw'], plan['requested_count'], max_slots,
                plan['plan_sha256'], contract_version=contract_version)
            plan['status'] = 'planned_unverified'
            self._save(record)
        return plan

    def _direction_plan_schema(self, plan):
        version = plan.get('contract_version')
        if version is None:
            return DirectionPlanProposal
        if version != FIXED_PICTURES_CONTRACT:
            raise ValueError('未知集合消费者合同；没有把旧稿转换为新合同。')
        count = min(plan['requested_count'], plan['max_slots'], 6)
        schema = fixed_picture_schema(count)
        if (type(plan.get('picture_count')) is not int or plan['picture_count'] != count
                or plan.get('picture_keys') != [f'picture_{index}' for index in range(1, count + 1)]
                or plan.get('response_schema_sha256') != self._request_digest(schema.model_json_schema())
                or plan['input']['collection_operation_scope'].get('picture_keys') != plan['picture_keys']
                or plan['input']['collection_operation_scope'].get('contract_version') != version):
            raise ValueError('固定作品数量、键或原schema绑定已变化；未换合同或补作品。')
        return schema

    def _validate_direction_plan(self, record, plan):
        schema = self._direction_plan_schema(plan)
        original = record['input_snapshot']['sentence']
        if plan['original_sha256'] != hashlib.sha256(original.encode('utf-8')).hexdigest():
            raise ValueError('集合提案没有绑定当前完整原句。')
        index = plan.get('call_index')
        if type(index) is not int or not 0 <= index < len(record['calls']):
            raise ValueError('集合提案缺少完整实际调用；没有重规划。')
        call = record['calls'][index]
        if (call.get('status') != 'returned' or call.get('purpose') != 'planning_directions'
                or call.get('data') != plan['raw']
                or call.get('input_binding_sha256') != plan['input_binding_sha256']
                or call.get('response_schema_sha256') != self._request_digest(schema.model_json_schema())
                or self._request_digest(plan['input']) != plan['input_binding_sha256']):
            raise ValueError('集合提案与原始调用或输入不一致；未重做或补写原件。')
        schema.model_validate(plan['raw'])
        if plan['plan_sha256'] is not None:
            if self._request_digest(plan['raw']) != plan['plan_sha256']:
                raise ValueError('集合提案原件哈希改变。')
            expected = allocate_direction_slots(plan['raw'], plan['requested_count'], plan['max_slots'], plan['plan_sha256'],
                contract_version=plan.get('contract_version'))
            identity_fields = ('id', 'ordinal', 'proposal_index', 'proposal') + (
                ('picture_key',) if plan.get('contract_version') == FIXED_PICTURES_CONTRACT else ())
            if len(expected) != len(plan['slots']) or any(
                    any(actual.get(key) != declared[key] for key in identity_fields)
                    for actual, declared in zip(plan['slots'], expected)):
                raise ValueError('已分配方向与集合原件不一致；未填补缺项或改变当前提案。')

    _DIRECTION_DRAFT_FIELDS = ('plan', 'plan_adapter', 'spatial_support', 'native_scene', 'native_user_override',
        'generation_request', 'native_prompts', 'picture_authoring_context', 'preparation_recipe_snapshot',
        'picture_event', 'translated_picture_event', 'canonical_picture_choice', 'translated_canonical_picture_choice',
        'picture_projection', 'prose_picture_choice', 'translated_prose_picture_choice', 'prose_picture_receipt')

    async def _direction_cached_call(self, record, key, system, data, schema, *, images=(), runtime_profile=None):
        """Reuse whole/language returns in their existing collection item only."""
        plan = record.get('direction_plan') or {}
        assignment = record.get('active_direction_assignment')
        slot = next((item for item in plan.get('slots', [])
            if assignment and item['id'] == assignment['id'] == plan.get('active_slot_id')), None)
        if slot is None:
            result = await self._call(record, system, data, schema, images=images, runtime_profile=runtime_profile)
            return deepcopy(result.data)
        binding = self._request_digest({'system': system, 'data': data, 'schema': schema,
            'image_sha256': [item.expected_sha256 for item in images]})
        checkpoint = slot.setdefault(key, {'input_binding_sha256': binding, 'call_index': None, 'raw': None})
        if checkpoint['input_binding_sha256'] != binding:
            raise ValueError('当前独立方向的作者／语言输入已变化；未重做或串用另一方向。')
        index = checkpoint['call_index']
        if index is not None and index < len(record['calls']):
            call = record['calls'][index]
            if call.get('status') != 'returned' or 'data' not in call:
                raise ValueError('当前方向调用未完整返回；恢复不自动重试。')
            if checkpoint['raw'] is not None and checkpoint['raw'] != call['data']:
                raise ValueError('当前方向原稿与实际返回不一致。')
            checkpoint['raw'] = deepcopy(call['data'])
        elif checkpoint['raw'] is not None:
            raise ValueError('当前方向原稿缺少实际调用原件。')
        elif index is not None and index != len(record['calls']):
            raise ValueError('当前方向调用预约不完整；不猜测重跑。')
        if checkpoint['raw'] is None:
            checkpoint['call_index'] = len(record['calls'])
            self._save(record)
            result = await self._call(record, system, data, schema, images=images, runtime_profile=runtime_profile)
            checkpoint['raw'] = deepcopy(result.data)
        self._save(record)
        return deepcopy(checkpoint['raw'])

    async def _drive_directions(self, record, purpose, action):
        plan = record['direction_plan']
        self._validate_direction_plan(record, plan)
        operation_key = self._request_digest(record['operation'])
        if operation_key not in plan['execution_operation_keys']:
            if purpose != 'generate' or action is None or plan['status'] != 'needs_confirmation':
                raise ValueError('集合只能从原操作恢复或当前方向的明确审阅生成接续。')
            plan['execution_operation_keys'].append(operation_key)
        approved_slot = plan.get('active_slot_id') if purpose == 'generate' else None
        review = not record['input_snapshot']['auto_generate']
        plan['status'] = 'running'
        self._save(record)
        for slot in plan['slots']:
            self._check(record)
            if slot['status'] in {'unplanned', 'completed', 'failed', 'cancelled'}:
                continue
            assignment = make_direction_assignment(plan, slot)
            plan['active_slot_id'] = slot['id']
            record['active_direction_assignment'] = deepcopy(assignment)
            if slot.get('job_id'):
                job = self.jobs.get(slot['job_id'])
                if job is None:
                    raise ValueError('独立方向已派发但原回执不可见；不重复作者或派发。')
                await self._finish_job(record, job)
                continue
            if slot.get('generation_request') is None:
                remaining = self.get(record['id'])['remaining_budget']
                minimum = (2 if self._record_request(record).picture_authoring_method == 'relational_steps' else 1)
                minimum += int(record['input_snapshot']['config'].get('render_pipeline') == 'adaptive')
                if remaining['jobs'] <= 0 or remaining['calls'] < minimum:
                    break
                slot['status'] = 'preparing'
                self._save(record)
                await self._prepare(record, direction_assignment=assignment)
                slot.update(status='prepared', generation_request=deepcopy(record['generation_request']),
                    recipe_snapshot=deepcopy(record['preparation_recipe_snapshot']),
                    draft={key: deepcopy(record.get(key)) for key in self._DIRECTION_DRAFT_FIELDS})
                self._save(record)
            else:
                # Explicit native edits apply only to the reviewed current item.
                if approved_slot != slot['id']:
                    record.update(deepcopy(slot['draft']))
                else:
                    current_assignment = (record['generation_request'].get('author_proposal') or {}).get('direction_assignment')
                    if current_assignment != assignment:
                        raise ValueError('待审原生条件不属于当前方向；不批准其他方向或重构。')
                record['active_direction_assignment'] = deepcopy(assignment)
                self._validate_authoring_recipes(record, slot['recipe_snapshot'])
            if review and approved_slot != slot['id']:
                plan['status'] = 'needs_confirmation'
                record.update(status='needs_confirmation', stage='review',
                    detail=f'当前方向{slot["ordinal"]}的完整原生条件待审；其他方向独立保留，未获本次生成批准。')
                self._save(record)
                return True
            if self.get(record['id'])['remaining_budget']['jobs'] <= 0:
                break
            method = 'research_pixels' if record['input_snapshot'].get('research_pixel_use') else 'text'
            try:
                await self._generate(record, method)
            except BaseException:
                if slot.get('job_id'):
                    job = self.jobs.get(slot['job_id'])
                    if job and job.status in {'failed', 'cancelled'}:
                        slot['status'] = job.status
                self._save(record)
                raise
            approved_slot = None
        plan['active_slot_id'] = None
        complete = [slot for slot in plan['slots'] if slot['status'] == 'completed']
        if len(complete) >= 2 and plan['comparison'] is None:
            plan['comparison'] = {'scope': 'independent_directions',
                'pair': {'artwork_ids': [slot['artwork_id'] for slot in complete[:2]],
                    'direction_ids': [slot['id'] for slot in complete[:2]]},
                'call_index': None, 'raw': None, 'status': 'pending'}
            self._save(record)
        comparison = plan['comparison']
        if comparison is not None and comparison['raw'] is None:
            recovered = self._returned_cycle_call(record, comparison, 'call_index', ResultComparison)
            if recovered is None and self.get(record['id'])['remaining_budget']['calls'] > 0:
                sources = [freeze_continuation_source(identifier, self.artworks, self.jobs,
                    original_text=record['input_snapshot']['sentence'], original_sha256=plan['original_sha256'],
                    output_root=Path(self.settings.output_dir), session_feedback=record['feedback'])
                    for identifier in comparison['pair']['artwork_ids']]
                comparison['sources'] = deepcopy(sources)
                data, images = prepare_result_pair(sources[0], sources[1], self.settings,
                    original_text=record['input_snapshot']['sentence'],
                    objective='Compare two independently assigned artistic directions, not whether a child repaired its ancestor.',
                    preserve='The original relations and current explicit conditions, not the first direction\'s camera or layout.')
                self._with_request_reading(record, data)
                data.update(comparison_scope='independent_directions',
                    direction_pair=[make_direction_assignment(plan, slot) for slot in complete[:2]],
                    image_order='FIRST independent direction; SECOND independent direction',
                    image_order_scope='parent/child are FIRST/SECOND image labels here, not ancestry or user selection')
                for index, binding in enumerate(data.get('image_bindings', [])):
                    binding['role'] = 'first_independent_direction' if index == 0 else 'second_independent_direction'
                comparison['call_index'] = len(record['calls'])
                record.update(stage='comparing_directions', detail='比较两个独立方向的实际原图；不虚构父子关系或认证差异达成。')
                self._save(record)
                result = await self._call(record, DIRECTION_COMPARISON_SYSTEM,
                    data, ResultComparison.model_json_schema(), images=images)
                recovered = deepcopy(result.data)
                for source in sources:
                    self._validate_continuation_source(record, source)
            if recovered is not None:
                comparison.update(raw=deepcopy(recovered), status='returned_unverified')
                self._save(record)
        plan['status'] = ('completed_unverified' if len(complete) == plan['requested_count'] else 'partial')
        record['detail'] = (f'已保存{len(complete)}个独立方向候选／请求{plan["requested_count"]}个；'
            '未展开项保留，图像差异与艺术要求未认证。')
        self._save(record)
        return False

    @staticmethod
    def _source_author_choices(source_context):
        proposal = source_context['source_job']['request'].get('author_proposal') or {}
        overridden = proposal.get('condition_status') == 'historical_after_native_override'
        # The canonical source has already been rendered into the complete
        # exact native text sent in continuation_model_material. Do not also
        # duplicate its mechanical PictureEvent in each new model input.
        canonical_rendered = bool(proposal.get('canonical_picture_choice')
            and (proposal.get('picture_projection') or {}).get('kind')
                == 'host_mechanical_projection_of_canonical_author_choice')
        prose_rendered = bool(proposal.get('prose_picture_choice')
            and (proposal.get('prose_picture_receipt') or {}).get('kind')
                == 'host_prose_picture_event_transport')
        rendered = canonical_rendered or prose_rendered
        return {'picture_event': None if overridden or rendered else deepcopy(proposal.get('picture_event')),
                'support_plan': None if overridden or rendered else deepcopy(proposal.get('support_plan')),
                'visual_logic': None if overridden else proposal.get('visual_logic'),
                'native_override_in_effect': overridden,
                'canonical_already_rendered_in_source_native': canonical_rendered and not overridden,
                'prose_already_rendered_in_source_native': prose_rendered and not overridden,
                'status': 'historical_after_native_override' if overridden else 'source_author_choice_unverified'}

    def _validate_continuation_source(self, record, source_context):
        original = record['input_snapshot']['sentence']
        return validate_continuation_source(source_context, self.artworks, self.jobs,
            original_text=original, original_sha256=hashlib.sha256(original.encode('utf-8')).hexdigest(),
            output_root=Path(self.settings.output_dir), session_feedback=record['feedback'])

    def _validate_authoring_recipes(self, record, recipes):
        request = self._record_request(record)
        current = authoring_input(request, request.config, self.settings)['full_style_recipes']
        if current != recipes:
            raise ValueError('本次选中风格配方的完整内容已变化；没有混用旧构思与新配方或派图。')

    def _archive_continuation_source(self, record, source_context):
        identifier = source_context['snapshot_sha256']
        history = record.setdefault('continuation_sources', {})
        previous = history.get(identifier)
        if previous is not None and previous != source_context:
            raise ValueError('后继源历史的同一签名不能替换原件。')
        history[identifier] = deepcopy(source_context)

    def _continuation_source_receipt(self, record, source_context):
        identifier = source_context['snapshot_sha256']
        if (record.get('continuation_sources') or {}).get(identifier) != source_context:
            raise ValueError('后继源原件未在本创作持久历史中保存；未派发。')
        return {'source_snapshot_sha256': identifier,
            'source_artwork_id': source_context['artwork']['id'],
            'source_job_id': source_context['source_job']['id'],
            'source_request_sha256': source_context['source_job']['request_sha256'],
            'image_sha256': source_context['image']['sha256'],
            'original_feedback_signature': source_context['feedback']['original_words_signature'],
            'source_context_ref': {'record_file': str((self.root / (record['id'] + '.json')).resolve()),
                                   'json_pointer': '/continuation_sources/' + identifier},
            'request_scope': source_context['request_scope'],
            'execution_workflow_alignment_verified': False,
            'semantic_understanding_verified': False}

    def _validate_research_support(self, record):
        selected = record['input_snapshot'].get('research_support')
        if selected is None:
            return None
        snapshot = record.get('research_support_snapshot')
        if snapshot is None or snapshot.get('selection') != selected:
            raise ValueError('研究视觉支撑缺少同源快照；未临时借另一件图')
        return validate_research_support(snapshot, self.artworks, self.jobs,
            output_root=Path(self.settings.output_dir))

    def _research_support_view(self, record):
        snapshot = self._validate_research_support(record)
        if snapshot is None:
            return None, ()
        image = snapshot['image']
        # A file can grow after validation; never allocate that unbounded new
        # file before refusing its SHA. The helper has already bound this size.
        with Path(image['file_path']).open('rb') as stream:
            raw = stream.read(image['bytes'] + 1)
        if hashlib.sha256(raw).hexdigest() != image['sha256'] or len(raw) != image['bytes']:
            raise ValueError('实际支撑像素与冻结原图不符；未换缩略图或旧图')
        self._validate_research_support(record)
        # The full saved request is retained, but is deliberately not another
        # new-generation recipe. This is an explicit pixel-material trial, not
        # historical condition transfer or a claim of workflow equivalence.
        material = {'artwork_id': snapshot['selection']['artwork_id'],
            'use_for': snapshot['selection']['use_for'], 'image': deepcopy(image),
            'source_original': deepcopy(snapshot['source_original']),
            'feedback': deepcopy(snapshot['feedback']),
            'current_input_authority': 'current original, identity and full style choices govern the new work',
            'input_scope': 'full pixels, source original and raw reviews; historical request retained in source snapshot, not sent as new conditions',
            'source_snapshot_ref': self._research_support_receipt(record),
            'whole_image_approved': False, 'transfer_as_img2img': False,
            'source_identity_or_intent_verified': False}
        pixel_use = record['input_snapshot'].get('research_pixel_use')
        if pixel_use is not None:
            material['explicit_pixel_use_request'] = deepcopy(pixel_use)
            material['pixel_scope_warning'] = 'whole-frame img2img also carries unapproved subject, layout and marks; use_for is not pixel isolation'
        sees_pixels = record['input_snapshot'].get('picture_authoring_backend') == 'paired_visual'
        material['author_pixel_access'] = 'actual_original_png' if sees_pixels else 'no_image_sent_to_text_author'
        return material, ((VisualImageInput(raw, image['sha256'], source_ref=material['artwork_id']),) if sees_pixels else ())

    def _research_support_receipt(self, record):
        snapshot = record['research_support_snapshot']
        return {'source_snapshot_sha256': snapshot['snapshot_sha256'],
            'artwork_id': snapshot['selection']['artwork_id'],
            'use_for': snapshot['selection']['use_for'], 'image_sha256': snapshot['image']['sha256'],
            'source_context_ref': {'record_file': str((self.root / (record['id'] + '.json')).resolve()),
                                   'json_pointer': '/research_support_snapshot'},
            'execution_workflow_alignment_verified': False, 'quality_verified': False}

    async def _relational_picture(self, record, data, *, images, recipes, source_context,
                                  feedback, observation):
        """A replaceable authoring method, not an automatic quality/check loop."""
        operation = record['operation']
        operation_binding = {'purpose': operation['purpose'],
                             'action_request_id': (operation.get('action') or {}).get('request_id')}
        if data.get('direction_assignment') is not None:
            operation_binding['direction_id'] = data['direction_assignment']['id']
        binding = self._request_digest({'input_snapshot': record['input_snapshot'], 'recipes': recipes,
            'feedback': feedback, 'observation': observation, 'source_context': source_context,
            'research_support': (record.get('research_support_snapshot') or {}).get('snapshot_sha256')})
        entry = record.get('relational_picture')
        if entry and entry.get('operation_binding') == operation_binding:
            if entry.get('input_binding_sha256') != binding:
                raise ValueError('关系构思暂停期间原件或当前选择改变；不续接旧回合。')
        else:
            if entry is not None:
                record.setdefault('relational_picture_history', []).append(deepcopy(entry))
            entry = {'core': None, 'finish': None, 'receipt': None,
                     'core_call': None, 'finish_call': None,
                     'status': 'forming_core', 'source_image_is_not_a_generation_parent': True,
                     'operation_binding': operation_binding, 'input_binding_sha256': binding,
                     'core_input': picture_authoring_input(data)}
            record['relational_picture'] = entry
        record.update(stage='relational_core', detail='共同形成身体、注意对象与空间选择；尚未成为完整生图条件。')
        self._save(record)
        if entry['core'] is None:
            attempt = entry.get('core_attempt_index')
            if attempt is not None and len(record['calls']) > attempt:
                raise ValueError('先前核心调用没有已记录完整回执；恢复不自动重跑。')
            self._check(record)
            entry['core_attempt_index'] = len(record['calls'])
            self._save(record)
            core_result = await self._call(record, CORE_SYSTEM, deepcopy(entry['core_input']),
                                          RelationalCore.model_json_schema(), images=images)
            entry.update(core=deepcopy(core_result.data), core_call=deepcopy(record['calls'][-1]))
            self._save(record)
        self._validate_authoring_recipes(record, recipes)
        self._validate_research_support(record)
        if source_context:
            self._validate_continuation_source(record, source_context)
        try:
            core = RelationalCore.model_validate(entry['core'])
        except ValidationError as exc:
            entry.update(status='core_invalid', validation_errors=exc.errors(
                include_url=False, include_context=False, include_input=False))
            self._save(record)
            raise ValueError('核心画面结构未成立，完整原回应与校验细节保留；未调用后段或生成图片。') from exc
        core_raw = core.model_dump(mode='json')
        finishing_input = deepcopy(entry['core_input'])
        finishing_input.update(relational_core=deepcopy(core_raw),
            core_status='unverified_proposal_that_may_require_revision',
            pixel_access='previous_core_author_only_no_images_in_this_call')
        entry.update(status='forming_finish', finishing_input=deepcopy(finishing_input))
        record.update(stage='relational_finish', detail='检查前一关系选择，补共享环境、光色与画法；可明确要求重构而非强行完成。')
        self._save(record)
        if entry['finish'] is None:
            attempt = entry.get('finish_attempt_index')
            if attempt is not None and len(record['calls']) > attempt:
                raise ValueError('先前后继构思调用没有已记录完整回执；恢复不自动重跑。')
            self._check(record)
            entry['finish_attempt_index'] = len(record['calls'])
            self._save(record)
            finish_result = await self._call(record, FINISH_SYSTEM, deepcopy(finishing_input),
                                            RelationalFinish.model_json_schema())
            entry.update(finish=deepcopy(finish_result.data), finish_call=deepcopy(record['calls'][-1]))
            self._save(record)
        self._validate_authoring_recipes(record, recipes)
        self._validate_research_support(record)
        if source_context:
            self._validate_continuation_source(record, source_context)
        if entry['core'] != core_raw:
            raise ValueError('关系构思核心在后继期间改变；没有拼接旧选择或派图。')
        self._check(record)
        finish = RelationalFinish.model_validate(entry['finish'])
        if finish.core_conflicts:
            entry['status'] = 'core_revision_required'
            self._save(record)
        choice = merge_relational_picture(core, finish)
        entry.update(status='merged_unverified',
            receipt=relational_receipt(core_raw, finish.model_dump(mode='json'), choice.model_dump(mode='json')))
        self._save(record)
        return choice.model_dump(mode='json')

    @staticmethod
    def _joint_enabled(record):
        snapshot = record['input_snapshot']
        return (snapshot.get('joint_picture_decision') is True
            and snapshot.get('support_mode', 'auto') == 'auto'
            and snapshot.get('picture_authoring_method') == 'prose'
            and snapshot.get('picture_authoring_backend', 'text') == 'text')

    def _attach_art_material(self, record, data, *, purpose, feedback=None, source=None):
        from .creative_research_context import load_artistic_material, artistic_material_catalog
        packet = load_artistic_material(record['input_snapshot']['sentence'],
            artworks=self.artworks, jobs=self.jobs, purpose=purpose,
            character_name=(data.get('knowledge', {}).get('character_name')
                or data.get('identity_binding', {}).get('character_name', '')),
            feedback=feedback,
            operation=(record.get('operation') or {}).get('purpose', ''),
            artwork_ids=(source['artwork']['id'],) if source else ())
        if source is not None:
            # The next decision already has one source-purpose view, its raw
            # feedback and actual parameters. Do not reintroduce the previous
            # painting as a second full workflow-prompt example. Original graph
            # conditions remain in this exact packet's host_material and in
            # the execution-bound source, not new composition requirements.
            for material in packet['model_input']['materials']:
                if material['artwork'].get('id') == source['artwork']['id']:
                    material['conditions'] = {
                        'use': 'source_conditions_follow_the_selected_source_purpose_view',
                        'source_artwork_id': source['artwork']['id'],
                        'full_record_location': 'host_artistic_material_originals_and_execution_bound_source',
                        'pixels_observed_by_this_helper': False}
                    material['recorded_request_parameters'] = {
                        'use': 'effective_parameters_in_selected_source_view_not_a_second_configuration'}
        full_packet = deepcopy(packet['model_input'])
        if source is None and record['input_snapshot'].get('research_mode') == 'catalog':
            view = artistic_material_catalog(full_packet)
            signature = self._request_digest(view)
            record.setdefault('artistic_material_packets', {})[signature] = full_packet
        else:
            view = full_packet
        data['artistic_research_material'] = deepcopy(view)
        record.setdefault('artistic_material_originals', {})[self._request_digest(view)] = deepcopy(packet['host_material'])

    def _read_joint_research(self, record, holder, queries):
        """Read only exact catalogue originals already held by this task."""
        from .creative_research_context import artistic_material_catalog, read_artistic_materials
        catalogue = holder['model_input']['artistic_research_material']
        signature = self._request_digest(catalogue)
        packet = (record.get('artistic_material_packets') or {}).get(signature)
        if packet is None or artistic_material_catalog(packet) != catalogue:
            raise ValueError('研究目录与完整原件绑定改变；未重查其他作品或拼补。')
        parsed_queries = [item.model_dump(mode='json') for item in queries]
        result = read_artistic_materials(packet, parsed_queries)
        if holder.get('research_read') is None:
            holder['research_read'] = deepcopy(result)
            self._save(record)
        elif holder['research_read'] != result:
            raise ValueError('已读研究的目的或完整原件改变；没有借另一返回。')
        return result

    @staticmethod
    def _adopted_initial_holder(record):
        holder = record.get('joint_initial_preparation') or {}
        return holder.get('final_decision') or holder

    def _validate_joint_initial_research(self, record):
        holder = record.get('joint_initial_preparation') or {}
        final = holder.get('final_decision')
        if final is None:
            return
        schema = ResearchProseDecision.model_json_schema()
        binding = self._request_digest({'system_prompt': holder.get('system_prompt'),
            'data': holder.get('model_input'), 'schema': schema})
        index = holder.get('call_index')
        calls = record['calls']
        if (holder.get('contract') != ResearchProseDecision.__name__
                or holder.get('input_binding_sha256') != binding
                or holder.get('response_schema_sha256') != self._request_digest(schema)
                or type(index) is not int or not 0 <= index < len(calls)
                or calls[index].get('status') != 'returned' or calls[index].get('data') != holder.get('raw')
                or calls[index].get('input_binding_sha256') != binding
                or calls[index].get('response_schema_sha256') != self._request_digest(schema)
                or holder.get('research_read') is None):
            raise ValueError('原研究问题或完整读取回执改变；未补读或重调。')
        inquiry = ResearchProseDecision.model_validate(holder['raw'])
        if not inquiry.research_queries or inquiry.picture is not None:
            raise ValueError('已采用原稿没有真正研究读取请求；不借后来的原件。')
        read = self._read_joint_research(record, holder, inquiry.research_queries)
        actual_input = final.get('model_input') or final.get('candidate_model_input') or {}
        if (actual_input.get('artistic_research_material') != read
                or actual_input.get('research_read_scope', {}).get('queries') != read['queries']):
            raise ValueError('最终画面输入不是原问题读取的完整材料；未重建。')

    def _successor_configuration_input(self, record, data):
        result = deepcopy(data)
        sources = record['input_snapshot'].get('parameter_sources')
        if record['input_snapshot'].get('parameter_scope', 'all') == 'initial':
            config = result['selected_config']
            target = {'anima_turbo': 'anima'}.get(config.get('model_id', 'newbie'), config.get('model_id', 'newbie'))
            fields = ('steps', 'cfg', 'sampler', 'scheduler', 'shift', 'cfg_rescale')
            defaults = {field: GenerateRequest.model_fields[field].default for field in fields}
            defaults.update({key: value for key, value in PROFILES[target].items() if key in fields})
            config.update(defaults)
            sources = {field: 'model_default' for field in fields}
        return result, sources

    async def _compile_joint_successor(self, record, context, holder, choice, config, *, source, fresh=False):
        picture = ProsePictureChoice.model_validate(holder['raw']['picture'])
        data = support_input(self._record_request(record), config, settings=self.settings,
            feedback={**context['feedback'], 'objective': choice.objective, 'preserve': choice.preserve},
            pixel_observation=None if fresh else context.get('observation'))
        self._with_request_reading(record, data)
        data['joint_method_choice'] = choice.model_dump(mode='json')
        data['current_picture_scope'] = {
            'use': 'execute the exact picture returned with this method, no second composition',
            'feedback': deepcopy(context['feedback']), 'preserve': choice.preserve}
        if not fresh:
            data['selected_source'] = continuation_model_material(source, for_model_call=True,
                successor_selection=True, current_creation_intent=data.get('creation_intent'),
                current_request_reading_material=data.get('request_reading_material'))
        assignment = record.get('active_direction_assignment')
        if assignment is not None:
            data['direction_assignment'] = deepcopy(assignment)
        # These are the same actual decision materials, not a second query or
        # new preferences gathered between the decision and its execution.
        material = holder['model_input'].get('artistic_research_material')
        if material is not None:
            data['artistic_research_material'] = deepcopy(material)
        context['authoring_call_index'] = holder['call_index']
        await self._adopt_joint_picture(record, holder, picture, data, config,
            source_context=None if fresh else source)
        context['authoring_recipe_snapshot'] = deepcopy(data['full_style_recipes'])
        context['request'] = deepcopy(record['generation_request'])
        self._save(record)

    async def _joint_model_call(self, record, holder, model, system, data):
        schema = model.model_json_schema()
        if holder.get('model_input') is None:
            holder.update(model_input=deepcopy(data), system_prompt=system,
                response_schema_sha256=self._request_digest(schema),
                input_binding_sha256=self._request_digest({'system_prompt': system, 'data': data, 'schema': schema}),
                contract=model.__name__, raw=None, call_index=None)
            self._save(record)
        if (holder['contract'] != model.__name__
                or holder['response_schema_sha256'] != self._request_digest(schema)
                or holder['input_binding_sha256'] != self._request_digest({
                    'system_prompt': holder['system_prompt'], 'data': holder['model_input'], 'schema': schema})):
            raise ValueError('共同画面决定的已保存用途或合同改变；未重新调用。')
        if holder.get('raw') is None:
            recovered = self._returned_cycle_call(record, holder, 'call_index', model)
            if recovered is None:
                holder['call_index'] = len(record['calls'])
                self._save(record)
                result = await self._call(record, holder['system_prompt'], holder['model_input'], schema)
                holder['raw'] = deepcopy(result.data)
            else:
                holder['raw'] = deepcopy(record['calls'][holder['call_index']]['data'])
            self._save(record)
        call = record['calls'][holder['call_index']]
        if (call.get('status') != 'returned' or call.get('data') != holder['raw']
                or call.get('input_binding_sha256') != holder['input_binding_sha256']
                or call.get('response_schema_sha256') != holder['response_schema_sha256']):
            raise ValueError('共同决定与实际完整返回不一致；不借用或重做旧调用。')
        return model.model_validate(holder['raw'])

    async def _adopt_joint_picture(self, record, holder, picture, data, selected_config, *, source_context=None):
        """Compile exactly the returned picture; no second author call."""
        raw = picture.model_dump(mode='json')
        request = self._record_request(record)
        context = deepcopy(data)
        context['selected_config'] = deepcopy(selected_config)
        operation_key = self._request_digest(record['operation'])
        if context.get('direction_assignment') is not None:
            operation_key = self._request_digest({'operation': record['operation'],
                'direction_id': context['direction_assignment']['id']})
        binding = self._request_digest({'operation': record['operation'],
            'authoring_input': context, 'selected_config': selected_config})
        checkpoint = holder.get('prose_checkpoint')
        if checkpoint is None:
            checkpoint = {'binding_sha256': binding, 'operation_key': operation_key,
                'raw': raw, 'author_call_index': holder['call_index'],
                'language_call_index': None, 'translated': None,
                'joint_decision_contract': holder['contract'],
                'joint_decision_input_sha256': holder['input_binding_sha256']}
            holder['prose_checkpoint'] = checkpoint
        elif checkpoint['binding_sha256'] != binding or checkpoint['raw'] != raw:
            raise ValueError('已采用的共同画面原稿或执行条件改变；未重构或重调。')
        old = record.get('prose_preparation')
        if old is not None and old is not checkpoint and old != checkpoint:
            record.setdefault('prose_preparation_history', []).append(deepcopy(old))
        record.update(picture_authoring_context=context, prose_preparation=checkpoint,
            prose_picture_choice=raw, translated_prose_picture_choice=None,
            prose_picture_receipt=None, canonical_picture_choice=None,
            translated_canonical_picture_choice=None, picture_projection=None,
            picture_event=prose_picture_event(picture).model_dump(mode='json'), translated_picture_event=None,
            preparation_recipe_snapshot=deepcopy(data['full_style_recipes']))
        record['joint_decision_receipt'] = {
            'contract': holder['contract'], 'decision_call_index': holder['call_index'],
            'decision_sha256': self._request_digest(holder['raw']), 'picture_sha256': self._request_digest(raw),
            'origin': 'system_joint_picture_and_method_unverified'}
        self._save(record)
        self._validate_authoring_recipes(record, data['full_style_recipes'])
        self._validate_research_support(record)
        if source_context:
            self._validate_continuation_source(record, source_context)
        translated = await self._prose_language(record, picture, checkpoint=checkpoint)
        self._validate_authoring_recipes(record, data['full_style_recipes'])
        self._validate_research_support(record)
        if source_context:
            self._validate_continuation_source(record, source_context)
        self._compile_prose_draft(record, translated, request, selected_config, checkpoint=checkpoint,
            research_material=data.get('research_pixel_material'))

    async def _available_node_snapshot(self, record):
        """Inspect a warm backend without starting it for text preparation.

        A cold backend is explicitly node-unverified, not an available-node
        certificate. The image worker starts Core and Core validates the exact
        compiled workflow before accepting its prompt into the real queue.
        """
        from .comfy_client import ComfyClient
        engine = getattr(self.jobs, 'client', None)
        if not isinstance(engine, ComfyClient):
            return None
        health = await engine.health()
        self._check(record)
        if not health.get('running'):
            return None
        import httpx
        async with httpx.AsyncClient(timeout=10, trust_env=False) as client:
            response = await client.get(self.settings.comfy_url.rstrip('/') + '/object_info')
            response.raise_for_status()
            snapshot = response.json()
        self._check(record)
        if not isinstance(snapshot, dict):
            raise ValueError('运行中图像后端的节点能力响应不是字典；未当作兼容证据。')
        return snapshot

    async def _prepare_joint_initial(self, record, request, *, feedback=None, observation=None,
                                     direction_assignment=None):
        operation_key = self._request_digest({'operation': record['operation'],
            'direction_id': (direction_assignment or {}).get('id')})
        holder = record.get('joint_initial_preparation')
        if holder is not None and holder['operation_key'] != operation_key:
            record.setdefault('joint_initial_history', []).append(deepcopy(holder))
            holder = None
        if holder is None:
            data = support_input(request, request.config, settings=self.settings,
                feedback=feedback, pixel_observation=observation)
            self._with_request_reading(record, data)
            if direction_assignment is not None:
                data['direction_assignment'] = deepcopy(direction_assignment)
            self._attach_art_material(record, data, purpose='initial_picture', feedback=feedback)
            research_material, _ = self._research_support_view(record)
            if research_material is not None:
                data['research_pixel_material'] = research_material
            node_snapshot = await self._available_node_snapshot(record)
            selection = prepare_starting_choice(data, self.settings,
                parameter_sources=request.parameter_sources, object_info=node_snapshot)
            model_input = picture_authoring_input(data, joint_decision=True)
            model_input.update(compatible_starts=compatible_choice_material(selection),
                excluded_starts=deepcopy(selection['excluded_starts']),
                starting_choice_required=selection['needs_choice'],
                start_mode=selection['mode'],
                explicit_target={'anima_turbo': 'anima'}.get(request.config.get('model_id', 'newbie'),
                    request.config.get('model_id', 'newbie')) if selection['mode'] == 'explicit_unchanged' else None)
            holder = {'operation_key': operation_key, 'context': selection, 'authoring_input': data,
                'candidate_model_input': deepcopy(model_input),
                'selected_config': None}
            record['joint_initial_preparation'] = holder
            record.update(stage='composing', detail='共同选择具体画面与可执行起点。')
            self._save(record)
        else:
            model_input = holder.get('model_input') or holder.get('candidate_model_input')
            if model_input is None:
                raise ValueError('共同决定原输入缺失；未猜补或重做。')
        record.update(stage='composing', detail='共同选择具体画面与可执行起点。')
        catalogue_mode = record['input_snapshot'].get('research_mode') == 'catalog'
        model = ResearchProseDecision if catalogue_mode else InitialProseDecision
        decision = await self._joint_model_call(record, holder, model,
            RESEARCH_PROSE_DECISION_SYSTEM if catalogue_mode else INITIAL_PROSE_DECISION_SYSTEM, model_input)
        adopted_holder = holder
        if catalogue_mode and decision.research_queries:
            self._check(record)
            read = self._read_joint_research(record, holder, decision.research_queries)
            if holder.get('final_decision') is None:
                final_input = deepcopy(holder['model_input'])
                final_input['artistic_research_material'] = deepcopy(read)
                final_input['research_read_scope'] = {
                    'use': 'complete originals read for these explicit current questions, not inherited scene requirements',
                    'queries': deepcopy(read['queries']), 'additional_reads_available': False}
                holder['final_decision'] = {'candidate_model_input': final_input}
                self._save(record)
            adopted_holder = holder['final_decision']
            decision = await self._joint_model_call(record, adopted_holder, InitialProseDecision,
                INITIAL_PROSE_DECISION_SYSTEM + '\nThe supplied research was read for explicit current questions. '
                'Use it only for those questions; reassess the whole current picture, do not copy a historical '
                'caption, exclusions, biography or pose as requested facts. No further lookup is available.',
                adopted_holder.get('model_input') or adopted_holder['candidate_model_input'])
        model_proposal = (decision.starting_choice.model_dump(mode='json')
            if decision.starting_choice else None)
        explicit_start = holder['context']['mode'] == 'explicit_unchanged'
        # A fixed renderer is already chosen by the user. The joint response
        # may acknowledge it, but that is not another choice to apply. Retain
        # the response without changing it or granting it control of config.
        adopted_choice = None if explicit_start else model_proposal
        selected, receipt = apply_starting_choice(request.config, adopted_choice, holder['context'])
        if explicit_start and model_proposal is not None:
            receipt['unapplied_model_proposal'] = deepcopy(model_proposal)
            receipt['selection_authority'] = 'explicit_user_configuration'
        if holder['selected_config'] is not None and holder['selected_config'] != selected:
            raise ValueError('共同画面决定的采用配置改变；未重选。')
        holder.update(selected_config=deepcopy(selected), receipt=receipt)
        author_operation_key = self._request_digest(record['operation'])
        if direction_assignment is not None:
            author_operation_key = self._request_digest({'operation': record['operation'],
                'direction_id': direction_assignment['id']})
        record['starting_choice'] = {'operation_key': author_operation_key, 'context': holder['context'],
            'choice': adopted_choice,
            'selected_config': deepcopy(selected), 'receipt': receipt, 'call_index': adopted_holder['call_index'],
            'joint_contract': adopted_holder['contract']}
        await self._adopt_joint_picture(record, adopted_holder, decision.picture, holder['authoring_input'], selected)

    async def _prepare(self, record, *, feedback=None, observation=None, source_context=None,
                       successor_operation=None, successor_config=None, cycle_context=None,
                       direction_assignment=None, fresh_reference_rejection=None):
        fresh = fresh_reference_rejection is not None
        if fresh:
            decision, selected = resolve_reference_choice(
                fresh_reference_rejection['context'], fresh_reference_rejection['choice'])
            if (selected is not None or decision.reference_base != 'neither'
                    or decision.operation != 'regenerate' or source_context is not None
                    or successor_operation != 'regenerate'):
                raise ValueError('重新构思必须来自明确的 neither 参照选择；不借被拒图的像素或构图。')
        request = self._record_request(record)
        await self._ensure_request_reading(record, request)
        if direction_assignment is None and source_context is not None:
            direction_assignment = deepcopy((source_context['source_job']['request'].get('author_proposal') or {}).get('direction_assignment'))
        record['active_direction_assignment'] = deepcopy(direction_assignment)
        if (self._joint_enabled(record) and source_context is None and successor_config is None
                and fresh_reference_rejection is None):
            await self._prepare_joint_initial(record, request, feedback=feedback,
                observation=observation, direction_assignment=direction_assignment)
            return
        saved_source_input = self._authoring_source_consumer(record, direction_assignment, source_context)
        research_material, research_images = self._research_support_view(record)
        source_choices = self._source_author_choices(source_context) if source_context else None
        if source_context:
            self._validate_continuation_source(record, source_context)
        selected_config = deepcopy(successor_config) if successor_config is not None else request.config
        # A fresh composition is not a replay of this record's last rendered
        # prose/canonical display. Exact parents are carried by source_context.
        rendered_previous = bool(record.get('prose_picture_choice') or record.get('canonical_picture_choice'))
        previous_plan = None if rendered_previous or fresh else record['plan']
        previous_event = None if rendered_previous or fresh else (
            record.get('translated_picture_event') or record.get('picture_event'))
        if successor_config is not None and source_context is None and not fresh:
            raise ValueError('后继配置必须绑定确切源作品，不用于隐式替换初始用户选择')
        if cycle_context is not None and source_context is None and not fresh:
            raise ValueError('路径比较只用于绑定源图的后继构思，不继承到独立初始构思')
        if source_context is None and selected_config.get('render_pipeline') == 'adaptive':
            initial_input = support_input(request, selected_config, settings=self.settings,
                feedback=feedback, previous_plan=previous_plan, pixel_observation=observation)
            self._with_request_reading(record, initial_input)
            if direction_assignment is not None:
                initial_input['direction_assignment'] = deepcopy(direction_assignment)
            checkpoint = record.get('starting_choice')
            selection_operation = self._request_digest(record.get('operation') or {})
            if direction_assignment is not None:
                selection_operation = self._request_digest({'operation': record['operation'],
                    'direction_id': direction_assignment['id']})
            if checkpoint is not None and checkpoint.get('operation_key') != selection_operation:
                record.setdefault('starting_choice_history', []).append(deepcopy(checkpoint))
                checkpoint = None
            if checkpoint is None:
                node_snapshot = await self._available_node_snapshot(record)
                selection = prepare_starting_choice(initial_input, self.settings,
                    parameter_sources=request.parameter_sources, object_info=node_snapshot)
                checkpoint = {'context': selection, 'choice': None, 'selected_config': None,
                              'operation_key': selection_operation}
                record['starting_choice'] = checkpoint
                self._save(record)
            selection = checkpoint['context']
            if checkpoint['selected_config'] is None:
                if selection['needs_choice']:
                    recovered = self._returned_cycle_call(record, checkpoint, 'call_index', StartingModelChoice)
                    if recovered is None:
                        checkpoint['call_index'] = len(record['calls'])
                        record.update(stage='choosing_start', detail='依据当前构想、真实兼容条件和研究选择起点；不是固定链。')
                        self._save(record)
                        result = await self._call(record, STARTING_CHOICE_SYSTEM,
                            selection['data'], StartingModelChoice.model_json_schema())
                        recovered = StartingModelChoice.model_validate(result.data).model_dump(mode='json')
                    checkpoint['choice'] = recovered
                chosen, receipt = apply_starting_choice(selected_config, checkpoint['choice'], selection)
                checkpoint.update(selected_config=chosen, receipt=receipt)
                self._save(record)
            selected_config = deepcopy(checkpoint['selected_config'])
        record.update(stage='composing', detail='形成可纠正的画面方案与视觉支撑；没有宣称原意已验证。')
        self._save(record)
        data = support_input(request, selected_config, settings=self.settings, feedback=feedback,
            previous_plan=(source_choices['support_plan'] if source_choices else previous_plan),
            pixel_observation=observation)
        self._with_request_reading(record, data, saved_input=saved_source_input)
        if direction_assignment is not None:
            data['direction_assignment'] = deepcopy(direction_assignment)
        data['previous_picture_event_unverified'] = deepcopy(
            source_choices['picture_event'] if source_choices else previous_event)
        if source_context or fresh:
            data['proposed_operation'] = {
                'method': successor_operation, 'target': selected_config.get('model_id'),
                'scope': 'system choice for the next picture, not user quotation or quality approval'}
        if fresh:
            data['rejected_reference_scope'] = {
                'contract_version': REFERENCE_CHOICE_CONTRACT,
                'reference_base': 'neither',
                'source_artwork_ids': {role: snapshot['artwork']['id']
                    for role, snapshot in fresh_reference_rejection['context']['sources'].items()},
                'use': 'Form a fresh composition from the complete current user choices and objective. '
                    'Neither previous image, native text, pose or composition is an adopted draft.'}
        if cycle_context is not None:
            data['cycle_learning_context'] = cycle_history_view(cycle_context)
            data['result_comparison_unverified'] = deepcopy(cycle_context['result_comparison_unverified'])
        if source_context:
            material = continuation_model_material(source_context, for_model_call=True,
                regenerate_authoring=successor_operation == 'regenerate',
                current_creation_intent=(data.get('creation_intent')
                    if successor_operation == 'regenerate' else None),
                current_request_reading_material=(data.get('request_reading_material')
                    if successor_operation == 'regenerate' else None))
            data.update(selected_source=material,
                source_author_choice_status=source_choices['status'],
                selection_authority={'current_input_snapshot': 'current user selections for the new proposal',
                    'selected_source': ('fresh regenerate author: original facts, raw feedback, current observation and effective parent parameters; completed parent native, composition rationale, poses/layout and compiled exclusion policy stay host-bound, not a draft to copy; preserved historical knowledge/recipe declarations are not new user choices'
                        if material.get('parent_native_scope') else
                        'purpose-bound view: complete actual native prompt/negative and effective generation fields, non-native author context and raw feedback; mechanical originals remain host-bound references, not independently verified execution nodes'),
                    'prior_picture_representation': ('completed parent canonical/native retained in host; form a new composition from original facts, observation, objective and preserve, not a parent prompt replay'
                        if material.get('parent_native_scope') else
                        'canonical source already rendered in complete actual native text; do not send mechanical PictureEvent twice' if source_choices['canonical_already_rendered_in_source_native'] else 'retained declared legacy picture choice or historical override boundary'),
                    'source_recipe_snapshot': 'historical evidence only; current complete recipes are above',
                    'native_override': 'actual source prompt/negative remain authoritative for that image; old author choices may be historical'})
            saved_use = (saved_source_input or {}).get('fresh_picture_use')
            if saved_use is not None:
                data['fresh_picture_use'] = deepcopy(saved_use)
            elif (saved_source_input is None and successor_operation == 'regenerate'
                    and material.get('parent_native_scope')
                    == 'full_parent_native_in_host_request_not_consumed_by_fresh_regenerate_author'):
                data['fresh_picture_use'] = {'version': 'fresh_single_picture_v1',
                    'source_artwork_id': source_context['artwork']['id'],
                    'kind': 'new_composition_of_current_reference_not_old_approach_replay',
                    'picture_count': 1}
        recipes = deepcopy(data['full_style_recipes'])
        record['preparation_recipe_snapshot'] = deepcopy(recipes)
        self._save(record)
        if request.support_mode == 'auto':
            # Reuse the existing whole-picture/native consumer. The optional
            # spatial diagnostic schema must not reduce normal art input to
            # count + background + a short caption.
            record['picture_authoring_context'] = deepcopy(data)
            if research_material is not None:
                data['research_pixel_material'] = research_material
                record['picture_authoring_context'] = deepcopy(data)
                record['detail'] = ('以选定研究原图的像素关系支撑完整构思；不默认照搬旧图或旧画风。'
                    if research_images else '形成当前完整画面条件；研究图拟作全幅像素起点，文字作者没有看见该图。')
            self._save(record)
            if request.picture_authoring_method == 'prose':
                await self._prepare_prose(record, request, data, selected_config,
                    images=research_images, research_material=research_material,
                    recipes=recipes, source_context=source_context)
                return
            if request.picture_authoring_method == 'relational_steps':
                raw_choice = await self._relational_picture(record, data, images=research_images,
                    recipes=recipes, source_context=source_context, feedback=feedback, observation=observation)
            else:
                raw_choice = await self._direction_cached_call(record, 'whole_author_checkpoint',
                    '正常创作需要发展完整可画方案。所有绘图词句用英文，visual_logic可中文。'
                    '明确原生count例如1girl, solo，不能把纯数字当作gender；不要凭角色名字编造外观。\n'
                    + ('You can see one internal research image. Use its pixels only for the explicit use_for; '
                       'explain which visible relationships can inform this new proposal and what remains unresolved. '
                       'The source original and reviews are historical material, not observation or current approval. '
                       'Do not claim a carving, gaze or identity is visible just because the current text requests it. '
                       'Current choices govern, not the old pose, framing, costume or style. '
                       'This is paired visual 4B authoring, not an invisible invocation of the text profile.\n'
                       if research_images else '')
                    + ('An explicit whole-frame research pixel use is proposed. You do NOT see its image. '
                       'Its source original, saved comments and declared use_for are not pixel facts or proof of current approval. '
                       'Develop current complete drawable choices, distinguish requested changes from assumed existing layout, '
                       'and retain uncertainty about what the source depicts. Whole-frame img2img can also inherit unwanted content.\n'
                       if research_material is not None and not research_images else '')
                    + CANONICAL_PICTURE_SYSTEM_PROMPT, picture_authoring_input(data),
                    {**CANONICAL_PICTURE_SCHEMA, 'title': 'CanonicalPictureChoice'}, images=research_images)
            self._validate_authoring_recipes(record, recipes)
            self._validate_research_support(record)
            if source_context:
                self._validate_continuation_source(record, source_context)
            record.update(canonical_picture_choice=deepcopy(raw_choice),
                translated_canonical_picture_choice=None, picture_projection=None,
                picture_event=None, translated_picture_event=None,
                prose_picture_choice=None, translated_prose_picture_choice=None,
                prose_picture_receipt=None)
            self._save(record)
            choice = CanonicalPictureChoice.model_validate(raw_choice)
            event = project_canonical_picture(choice)
            record['picture_event'] = event.model_dump(mode='json')
            self._save(record)
            choice = await self._canonical_language(record, choice)
            self._validate_authoring_recipes(record, recipes)
            self._validate_research_support(record)
            if source_context:
                self._validate_continuation_source(record, source_context)
            event = project_canonical_picture(choice)
            if record['translated_canonical_picture_choice'] is not None:
                record['translated_picture_event'] = event.model_dump(mode='json')
            record['picture_projection'] = canonical_projection_receipt(choice,
                original_choice=record['canonical_picture_choice'])
            self._save(record)
            actual = compile_picture_support(event, request, selected_config, self.settings,
                session_id=record['id'], evidence={'author_call': record['calls'][-2]
                    if record['calls'][-1]['purpose'] == 'native_language' else record['calls'][-1],
                    'language_call': record['calls'][-1]
                    if record['calls'][-1]['purpose'] == 'native_language' else None})
            self._tag_canonical_projection(record, actual)
            actual = self._attach_request_reading(record, actual)
            if research_material is not None:
                actual.author_proposal['research_support_receipt'] = self._research_support_receipt(record)
                actual.author_proposal['picture_authoring_backend'] = ('paired_visual' if research_images else 'text_with_explicit_research_pixel_use')
            plan = CanonicalPlanView(scene_text=event.scene_narrative, visual_logic=event.visual_logic,
                count=event.visual_spec.count, framing=event.visual_spec.framing or ['unspecified framing'],
                background=event.visual_spec.background or ['unspecified background'],
                lighting=event.visual_spec.lighting or ['unspecified lighting'],
                elements=[], use_visual_support=False,
                support_reason='完整作者方案由原生消费者承接；当前色块草案不自动采用。',
                uncertainties=event.uncertainties)
            record.update(plan=plan.model_dump(mode='json'),
                plan_adapter='canonical_picture_host_projection_display_view',
                native_prompts=actual.native_prompts, generation_request=actual.model_dump(mode='json'),
                spatial_support=check_spatial_support(plan), native_scene=None, native_user_override=False)
            self._save(record)
            return
        result = await self._call(record, SUPPORT_SYSTEM, data, SupportPlan.model_json_schema())
        self._validate_authoring_recipes(record, recipes)
        if source_context:
            self._validate_continuation_source(record, source_context)
        plan = SupportPlan.model_validate(result.data)
        if selected_config.get('render_pipeline') == 'adaptive' and plan.starting_model:
            if selected_config.get('lora_name') and PROFILES[plan.starting_model]['model_id'] != selected_config.get('model_id'):
                raise ValueError('起点提议与已选 LoRA 兼容性未确认；未静默移除角色／风格条件')
            selected_config = {**selected_config, **PROFILES[plan.starting_model], 'render_pipeline': plan.starting_model}
        actual = compile_support(plan, request, selected_config, self.settings,
            session_id=record['id'], evidence={'directory': str(result.evidence_dir),
                'usage': result.usage, 'timings': result.timings})
        actual = self._attach_request_reading(record, actual)
        record.update(plan=plan.model_dump(mode='json'), native_prompts=actual.native_prompts,
                      generation_request=actual.model_dump(mode='json'),
                      spatial_support=check_spatial_support(plan), native_scene=None, native_user_override=False,
                      picture_event=None, translated_picture_event=None, plan_adapter=None,
                      canonical_picture_choice=None, translated_canonical_picture_choice=None,
                      picture_projection=None, prose_picture_choice=None,
                      translated_prose_picture_choice=None, prose_picture_receipt=None)
        self._save(record)
        await self._ensure_native(record, source_context=source_context, recipe_snapshot=recipes)

    async def _prepare_prose(self, record, request, data, selected_config, *, images,
                             research_material, recipes, source_context):
        checkpoint = record.get('prose_preparation')
        authoring_operation = record['operation']
        binding = self._request_digest({'operation': authoring_operation,
            'authoring_input': data, 'selected_config': selected_config})
        legacy_operation_key = self._request_digest(authoring_operation)
        if source_context is None and data.get('rejected_reference_scope') is not None:
            continuation = record.get('continuation') or {}
            slot = continuation.get('authoring_call_index')
            if (type(slot) is not int or not 0 <= slot <= len(record['calls'])
                    or continuation.get('reference_contract_version') not in {REFERENCE_CHOICE_CONTRACT, FULL_REFERENCE_CHOICE_CONTRACT}
                    or (continuation.get('reference_choice') or {}).get('reference_base') != 'neither'
                    or (continuation.get('reference_choice') or {}).get('operation') != 'regenerate'):
                raise ValueError('全新构思缺少本回合的明确参照拒绝与作者预约；未复用初始作者槽。')
            operation_key = self._request_digest({'operation': authoring_operation,
                'reference_context_sha256': continuation['reference_context']['binding_sha256'],
                'authoring_call_index': slot, 'reference_base': 'neither'})
        elif source_context is None:
            operation_key = legacy_operation_key
            if data.get('direction_assignment') is not None:
                operation_key = self._request_digest({'operation': authoring_operation,
                    'direction_id': data['direction_assignment']['id']})
        else:
            continuation = record.get('continuation') or {}
            authoring_slot = continuation.get('authoring_call_index')
            if (type(authoring_slot) is not int or not 0 <= authoring_slot <= len(record['calls'])
                    or (continuation.get('source_context') or {}).get('snapshot_sha256') != source_context['snapshot_sha256']):
                raise ValueError('后继构思缺少已保存的同源作者操作预约；没有猜补回合或重做调用。')
            operation_key = self._request_digest({'operation': authoring_operation,
                'source_snapshot_sha256': source_context['snapshot_sha256'],
                'authoring_call_index': authoring_slot})
        same_checkpoint = checkpoint is not None and (
            checkpoint.get('operation_key') == operation_key
            or (source_context is not None and checkpoint.get('operation_key') == legacy_operation_key
                and checkpoint.get('binding_sha256') == binding))
        if same_checkpoint:
            if checkpoint.get('binding_sha256') != binding:
                raise ValueError('完整描述暂停期间当前输入或完整配方改变；不重做旧调用。')
        else:
            if checkpoint is not None:
                record.setdefault('prose_preparation_history', []).append(deepcopy(checkpoint))
            checkpoint = {'binding_sha256': binding, 'operation_key': operation_key,
                'raw': None, 'author_call_index': None,
                'language_call_index': None, 'translated': None}
            record['prose_preparation'] = checkpoint
        if checkpoint['raw'] is None:
            attempted = checkpoint['author_call_index']
            if attempted is not None and len(record['calls']) > attempted:
                raise ValueError('完整描述调用没有已记录完整返回；恢复不自动重跑。')
            checkpoint['author_call_index'] = len(record['calls'])
            self._save(record)
            system = PROSE_PICTURE_SYSTEM
            if images:
                system += ('\nYou can see the selected internal research image. Its pixels serve only '
                    'the explicit use_for. Source text and reviews are historical material, not pixel '
                    'facts or current approval. Current choices govern the new complete description; '
                    'retain uncertainty about identity, gaze and other unverified details.')
            elif research_material is not None:
                system += ('\nAn explicit whole-frame research pixel use is proposed, but you do not '
                    'see its image. Source words and use_for are not observations or approval. '
                    'Choose the current complete frame and retain uncertainty about source content.')
            result = await self._call(record, system, picture_authoring_input(data),
                ProsePictureChoice.model_json_schema(), images=images)
            checkpoint['raw'] = deepcopy(result.data)
        record.update(prose_picture_choice=deepcopy(checkpoint['raw']),
            translated_prose_picture_choice=None,
            prose_picture_receipt=None, canonical_picture_choice=None,
            translated_canonical_picture_choice=None, picture_projection=None,
            picture_event=None, translated_picture_event=None)
        # Save the actual return before validation/format adaptation.
        self._save(record)
        self._validate_authoring_recipes(record, recipes)
        self._validate_research_support(record)
        if source_context:
            self._validate_continuation_source(record, source_context)
        choice = ProsePictureChoice.model_validate(checkpoint['raw'])
        record['picture_event'] = prose_picture_event(choice).model_dump(mode='json')
        self._save(record)
        choice = await self._prose_language(record, choice, checkpoint=checkpoint)
        self._validate_authoring_recipes(record, recipes)
        self._validate_research_support(record)
        if source_context:
            self._validate_continuation_source(record, source_context)
        self._compile_prose_draft(record, choice, request, selected_config, checkpoint=checkpoint,
            research_material=research_material, images=images)

    def _compile_prose_draft(self, record, choice, request, selected_config, *, checkpoint,
                             research_material=None, images=()):
        event = prose_picture_event(choice)
        if record.get('translated_prose_picture_choice') is not None:
            record['translated_picture_event'] = event.model_dump(mode='json')
        record['prose_picture_receipt'] = prose_picture_receipt(choice,
            original_choice=record['prose_picture_choice'])
        self._save(record)
        author_call = record['calls'][checkpoint['author_call_index']]
        language_index = checkpoint.get('language_call_index')
        actual = compile_picture_support(event, request, selected_config, self.settings,
            session_id=record['id'], evidence={'author_call': deepcopy(author_call),
                'language_call': deepcopy(record['calls'][language_index])
                    if checkpoint.get('translated') is not None else None})
        self._tag_prose_transport(record, actual)
        actual = self._attach_request_reading(record, actual)
        if research_material is not None:
            actual.author_proposal['research_support_receipt'] = self._research_support_receipt(record)
            actual.author_proposal['picture_authoring_backend'] = ('paired_visual' if images
                else 'text_with_explicit_research_pixel_use')
        plan = CanonicalPlanView(scene_text=event.scene_narrative, visual_logic=event.visual_logic,
            count=event.visual_spec.count,
            framing=['framing not structurally extracted; see complete scene narrative'],
            background=['background not structurally extracted; see complete scene narrative'],
            lighting=['lighting not structurally extracted; see complete scene narrative'],
            elements=[], use_visual_support=False,
            support_reason='完整描述原样进入原生消费者；关系、镜头等未结构化抽取，空结构不表示画面缺少这些内容。',
            uncertainties=event.uncertainties)
        record.update(plan=plan.model_dump(mode='json'), plan_adapter='prose_picture_host_display_view',
            native_prompts=actual.native_prompts, generation_request=actual.model_dump(mode='json'),
            spatial_support=check_spatial_support(plan), native_scene=None, native_user_override=False)
        self._save(record)

    async def _prepare_saved_prose(self, record, action):
        entry = record['native_preparations'][action.request_id]
        source = self._saved_prose_source(record)
        if (entry['source_checkpoint'] != source['checkpoint']
                or entry['source_context_sha256'] != self._request_digest(source['context'])):
            raise ValueError('明确准备操作的原稿材料已变化；没有重跑作者。')
        if source['checkpoint'].get('kind') == 'saved_canonical_language_return_v1':
            self._check(record)
            choice = apply_canonical_language(source['raw'], source['checkpoint']['translated'])
            record['translated_canonical_picture_choice'] = choice.model_dump(mode='json')
            record['canonical_language_application'] = canonical_language_scope(source['raw'])
            event = project_canonical_picture(choice)
            record['translated_picture_event'] = event.model_dump(mode='json')
            record['picture_projection'] = canonical_projection_receipt(choice,
                original_choice=source['checkpoint']['raw'])
            actual = compile_picture_support(event, source['request'], source['config'], self.settings,
                session_id=record['id'], evidence={
                    'author_call': record['calls'][source['checkpoint']['author_call_index']],
                    'language_call': record['calls'][source['checkpoint']['language_call_index']]})
            self._tag_canonical_projection(record, actual)
            actual = self._attach_request_reading(record, actual)
            plan = CanonicalPlanView(scene_text=event.scene_narrative, visual_logic=event.visual_logic,
                count=event.visual_spec.count, framing=event.visual_spec.framing or ['unspecified framing'],
                background=event.visual_spec.background or ['unspecified background'],
                lighting=event.visual_spec.lighting or ['unspecified lighting'],
                elements=[], use_visual_support=False,
                support_reason='复用原完整方案及语言返回，零新调用准备；不认证画面。',
                uncertainties=event.uncertainties)
            record.update(plan=plan.model_dump(mode='json'),
                plan_adapter='canonical_picture_host_projection_display_view',
                native_prompts=actual.native_prompts, generation_request=actual.model_dump(mode='json'),
                spatial_support=check_spatial_support(plan), native_scene=None, native_user_override=False)
            entry.update(status='prepared', actual_new_calls=0,
                applied_choice_scope='same_saved_initial_canonical_choice_language_and_compilation_only')
            actual.author_proposal['saved_canonical_preparation_receipt'] = {
                'origin': action.initiator, 'action_request_id': action.request_id,
                'source_binding_sha256': source['checkpoint']['binding_sha256'],
                'author_call_index': source['checkpoint']['author_call_index'],
                'language_call_index': source['checkpoint']['language_call_index'],
                'new_model_calls': 0, 'semantic_correctness_verified': False}
            record['generation_request']['author_proposal'] = actual.author_proposal
            self._save(record)
            return
        self._check(record)
        record.update(stage='native_language',
            detail='复用已有完整原稿准备原生条件；原作者、初选与失败回执保留，不自动派图。')
        record['translated_prose_picture_choice'] = None
        self._save(record)
        checkpoint = source['language_checkpoint'] if entry['reuse_language_return'] else entry
        choice = await self._prose_language(record, source['raw'], checkpoint=checkpoint)
        self._check(record)
        current = self._saved_prose_source(record)
        if (current['config'] != source['config']
                or current['context'] != source['context']
                or self._pending_generation_dispatch(record) is not None
                or record.get('active_job_id') or record.get('cancel_error')):
            raise ValueError('准备期间完整条件或任务边界改变；没有派图。')
        self._compile_prose_draft(record, choice, source['request'], source['config'],
            checkpoint={**checkpoint, 'author_call_index': source['checkpoint']['author_call_index']},
            research_material=source['context'].get('research_pixel_material'),
            images=source['request'].picture_authoring_backend == 'paired_visual')
        entry.update(status='prepared', actual_new_calls=len(record['calls']) - entry['calls_before'],
            applied_choice_scope='same_saved_initial_author_choice_language_and_compilation_only')
        record['generation_request']['author_proposal']['saved_prose_preparation_receipt'] = {
            'origin': action.initiator, 'action_request_id': action.request_id,
            'source_prose_binding_sha256': source['checkpoint']['binding_sha256'],
            'author_call_index': source['checkpoint']['author_call_index'],
            'language_call_index': checkpoint.get('language_call_index'),
            'new_author_calls': 0, 'new_starting_choice_calls': 0,
            'new_language_calls': entry['actual_new_calls'], 'semantic_correctness_verified': False}
        self._save(record)

    def _tag_prose_transport(self, record, actual):
        if record.get('prose_picture_choice') is not None:
            actual.author_proposal.update(author='local_prose_picture_choice_with_host_transport_unverified',
                prose_picture_choice=deepcopy(record['prose_picture_choice']),
                translated_prose_picture_choice=deepcopy(record.get('translated_prose_picture_choice')),
                prose_picture_receipt=deepcopy(record.get('prose_picture_receipt')),
                scene_narrative_origin='model_authored_complete_prose')

    async def _prose_language(self, record, choice, *, checkpoint):
        raw = choice.model_dump(mode='json')
        scope = prose_language_scope(choice)
        sources = scope['field_sources']
        if ('language_response' not in [sources['count'], sources['scene_narrative']]
                and 'language_response' not in sources['exclusions']):
            return choice
        record.update(stage='native_language', detail='只转换当前完整描述的语言；原稿与数量格式各自保留。')
        self._save(record)
        if checkpoint.get('translated') is None:
            attempted = checkpoint.get('language_call_index')
            if attempted is not None and len(record['calls']) > attempted:
                raise ValueError('完整描述语言转换没有已记录完整返回；恢复不自动重跑。')
            checkpoint['language_call_index'] = len(record['calls'])
            self._save(record)
            result = await self._call(record,
                '仅把language_application_scope注明language_response的原含CJK绘图字段译为英文，返回同一schema。'
                '其余字段保留原文；宿主只消费明确需译的字段，说明和不确定项由宿主从作者原稿承接。仅转换语言，'
                '不增加或删去任何画面选择、关系、动作、限定或似乎语气，不推断性别、物种或外观。'
                '原数字或英文count与已英文scene_narrative不改；中文count仅译原已表明的数量／主体，不猜身份。'
                'exclusions保持原项数和原对应索引；各原英文项不改，不合并或重排。',
                {'original_text': record['input_snapshot']['sentence'], 'prose_choice_to_translate': raw,
                 'language_application_scope': scope,
                 'scope': 'language_conversion_only_not_new_art_or_approval'},
                ProsePictureChoice.model_json_schema(), runtime_profile='local_4b')
            checkpoint['translated'] = deepcopy(result.data)
            self._save(record)
        applied = apply_prose_language(choice, checkpoint['translated'])
        record['translated_prose_picture_choice'] = applied.model_dump(mode='json')
        record['prose_picture_receipt'] = prose_picture_receipt(applied, original_choice=raw)
        self._save(record)
        return applied

    def _tag_canonical_projection(self, record, actual):
        # Old stored PictureEvents keep their original attribution. A new host
        # rendering must not be mislabeled as the model's raw scene narrative.
        if record.get('canonical_picture_choice') is not None:
            actual.author_proposal.update(
                author='local_canonical_picture_choice_with_host_projection_unverified',
                canonical_picture_choice=deepcopy(record['canonical_picture_choice']),
                translated_canonical_picture_choice=deepcopy(record.get('translated_canonical_picture_choice')),
                picture_projection=deepcopy(record['picture_projection']),
                scene_narrative_origin='host_mechanical_projection_not_model_authored_prose')
            if record['input_snapshot'].get('picture_authoring_method', 'whole') == 'relational_steps':
                parts = record['relational_picture']
                actual.author_proposal.update(author='local_relational_picture_choices_with_host_merge_unverified',
                    canonical_choice_origin='host_exact_merge_of_two_model_returns_not_single_model_return',
                    relational_picture={key: deepcopy(parts[key]) for key in ('core', 'finish', 'receipt')},
                    relational_context_ref={'record_file': str((self.root / (record['id'] + '.json')).resolve()),
                        'session_id': record['id'], 'json_pointer': '/relational_picture',
                        'scope': 'full_inputs_and_call_receipts_live_in_owning_session_not_recursive_child_metadata'})

    async def _canonical_language(self, record, choice):
        raw = choice.model_dump(mode='json')
        drawing = {key: value for key, value in raw.items() if key not in {'visual_logic', 'uncertainties'}}
        if not re.search(r'[\u3400-\u9fff\u3040-\u30ff\uac00-\ud7af]', json.dumps(drawing, ensure_ascii=False)):
            return choice
        record.update(stage='native_language', detail='只转换同一份美术选择的语言；叙述和位置表仍由宿主投影。')
        self._save(record)
        value = await self._direction_cached_call(record, 'canonical_language_checkpoint',
            '逐项把所给唯一画面选择译为英文，返回同一schema。只转换语言，不另作美术选择。'
            '保留仿佛/似乎、动作、全部形体、位置与关系，不编造角色衣服或外观。'
            '数组项数、对象id、relation的subject/object及count原样；visual_logic与uncertainties原样。'
            '其余全部词句必须英文。不要另写caption、layout或第二份位置表。',
            {'original_text': record['input_snapshot']['sentence'], 'canonical_choice_to_translate': raw,
             'scope': 'language_conversion_only_not_new_art_or_approval'},
            {**CANONICAL_PICTURE_SCHEMA, 'title': 'CanonicalPictureChoice'}, runtime_profile='local_4b')
        translated = apply_canonical_language(choice, value)
        record['canonical_language_application'] = canonical_language_scope(choice)
        record['translated_canonical_picture_choice'] = translated.model_dump(mode='json')
        self._save(record)
        return translated

    async def _picture_language(self, record, event):
        raw = event.model_dump(mode='json')
        checked = {key: value for key, value in raw.items() if key not in {'visual_logic', 'uncertainties'}}
        if not re.search(r'[\u3400-\u9fff]', json.dumps(checked, ensure_ascii=False)):
            record['translated_picture_event'] = None
            self._save(record)
            return event
        record.update(stage='native_language', detail='转换完整画面条件的语言，保留对象、动作、关系及原方案。')
        self._save(record)
        result = await self._call(record,
            '逐项把所给完整画面方案译为英文，返回同一schema。只转换语言，不发展另一幅图。'
            '保留仿佛/似乎、动作、注意对象、构图、全部形体及关系，不编造角色衣服或外观。'
            '数组项数、positions键、对象id、relation的subject/object及count原样。'
            'visual_logic与uncertainties原样。所有其他文字必须英文，不增加或删除字段。',
            {'original_text': record['input_snapshot']['sentence'], 'picture_event_to_translate': raw,
             'scope': 'language_conversion_only_not_new_art_or_approval'},
            {**PICTURE_EVENT_SCHEMA, 'title': 'PictureEvent'}, runtime_profile='local_4b')
        translated = PictureEvent.model_validate(result.data)
        value = translated.model_dump(mode='json')
        def structure(item, key=''):
            if isinstance(item, dict):
                return {name: structure(child, name) for name, child in item.items()}
            if isinstance(item, list):
                return [structure(child, key) for child in item]
            return item if key in {'id', 'subject', 'object', 'count'} else 'text'
        if (structure(raw) != structure(value)
                or raw['visual_logic'] != value['visual_logic'] or raw['uncertainties'] != value['uncertainties']):
            raise ValueError('完整画面语言转换改变了条件结构、引用、数量或说明；未手工补回')
        record['translated_picture_event'] = value
        self._save(record)
        return translated

    async def _ensure_native(self, record, *, source_context=None, recipe_snapshot=None):
        if record.get('native_scene') or record.get('native_user_override'):
            return
        if record.get('plan_adapter') == 'prose_picture_host_display_view':
            # The complete current prose (and any same-choice language
            # conversion) was compiled before this adapter became ready.
            # Its display markers are not legacy NativeScene drawing fields.
            ProsePictureChoice.model_validate(record.get('translated_prose_picture_choice')
                or record['prose_picture_choice'])
            return
        plan_type = (CanonicalPlanView if record.get('plan_adapter') == 'canonical_picture_host_projection_display_view'
                     else SupportPlan)
        plan = plan_type.model_validate(record['plan'])
        fields = {key: getattr(plan, key) for key in ('scene_text', 'count', 'framing', 'background', 'lighting')}
        if not re.search(r'[\u3400-\u9fff]', json.dumps(fields, ensure_ascii=False)):
            return
        record.update(stage='native_language', detail='转换画面方案的原生语言；不重新创造场景或认证理解。')
        self._save(record)
        result = await self._call(record, '你只负责把所给画面条件逐项译为英文，返回符合schema的JSON。'
            '这不是重新创作：不增加或删去人物、动作、情绪、背景、光线、因果或官方外观。'
            '尤其保留“仿佛／似乎”等未确定语气，不能变成已确定的事实。'
            'count原样；三个数组保持原项数量和顺序。角色英文名称可用宿主提供的tag帮助识别。'
            '原句和选择保留为来源，不是要求你输出中文。所有输出词句除原样数字count外必须英文。',
            {'original_text': record['input_snapshot']['sentence'], 'authored_fields_to_translate': fields,
             'identity_tag': record['generation_request']['semantic_spec']['character_tag'],
             'scope': 'language_conversion_only_not_intent_or_art_approval'},
            NativeScene.model_json_schema(), runtime_profile='local_4b')
        if recipe_snapshot is not None:
            self._validate_authoring_recipes(record, recipe_snapshot)
        if source_context:
            self._validate_continuation_source(record, source_context)
        translated = NativeScene.model_validate(result.data)
        if re.search(r'[\u3400-\u9fff]', json.dumps(translated.model_dump(), ensure_ascii=False)):
            raise ValueError('原生语言转换仍包含未转换的中文，没有静默作为完整英文条件提交')
        if translated.count != plan.count or any(len(getattr(translated, key)) != len(getattr(plan, key))
            for key in ('framing', 'background', 'lighting')):
            raise ValueError('语言转换改变了数量或条件项数；原返回保留，未手工补回')
        prior = GenerateRequest.model_validate(record['generation_request'])
        actual = compile_support(plan, self._record_request(record),
            prior.model_dump(mode='json'), self.settings, session_id=record['id'],
            evidence=record['calls'][0], native_scene=translated)
        actual = self._attach_request_reading(record, actual)
        record.update(native_scene=translated.model_dump(mode='json'), native_prompts=actual.native_prompts,
                      generation_request=actual.model_dump(mode='json'))
        self._save(record)

    @staticmethod
    def _request_digest(value):
        return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
            allow_nan=False, separators=(',', ':')).encode('utf-8')).hexdigest()

    @staticmethod
    def _pending_generation_dispatch(record):
        # Legacy records have no reservation. Do not manufacture a historical
        # receipt or infer that a missing receipt means submission failed.
        return next((item for item in reversed(record.get('generation_dispatches', []))
                     if not item.get('completed')), None)

    async def _generate(self, record, method, *, request=None, artwork=None, denoise=.35,
                        source_context=None, recipe_snapshot=None, region_edit=None):
        self._check(record)
        pending_dispatch = self._pending_generation_dispatch(record)
        if pending_dispatch:
            if pending_dispatch['method'] != method:
                raise ValueError('未完成生成的派发方法不一致，不将原回执改作另一操作')
            if not pending_dispatch.get('job_id'):
                raise ValueError('生成提交回执未确认，不重复派发；请核对原任务')
            prior_job = self.jobs.get(pending_dispatch['job_id'])
            if prior_job is None:
                raise ValueError('生成已提交但原回执暂不可见，不重复派发；请核对原任务')
            if record.get('active_job_id') not in {None, prior_job.id}:
                raise ValueError('生成回执与当前任务不一致，不猜测或重复派发')
            return await self._finish_job(record, prior_job)
        prior_job = self.jobs.get(record.get('active_job_id', ''))
        if record.get('active_job_id') and prior_job is None:
            raise ValueError('已有提交回执对应任务暂不可见，不重派；请核对原任务')
        if prior_job:
            return await self._finish_job(record, prior_job)
        if source_context:
            self._validate_continuation_source(record, source_context)
        self._validate_research_support(record)
        if len(record['candidates']) >= record['limits']['jobs']:
            raise InterruptedError('本次候选预算耗尽，没有新派发。')
        actual = request or GenerateRequest.model_validate(record['generation_request'])
        actual = actual.model_copy(update={'output_label': f'{method}__candidate-{len(record["candidates"])+1}'})
        record.update(stage='generating', detail=f'正在生成 {method} 候选；未认证作品合格。')
        self._save(record)
        workflow, input_bytes, owned_inputs = None, None, []
        if method == 'sketch':
            plan = SupportPlan.model_validate(record['plan'])
            png = render_support(plan, actual.width, actual.height)
            material = register_visual_material(png, actual.creation_intent.original_text,
                plan.model_dump(mode='json'), self.settings)
            record['materials'].append({'id': material.id,
                'image_url': f'/api/creative-materials/{material.id}/image', 'kind': 'programme_sketch',
                'spatial_support': check_spatial_support(plan)})
            self._save(record)
            actual, workflow = prepare_visual_material_handoff(actual, material, self.settings,
                denoise=denoise, materialize=False)
            input_bytes = png
        elif method == 'handoff':
            if artwork is None:
                raise ValueError('交接缺少真实当前作品')
            parent = generation_ancestor(artwork, self.artworks, self.jobs)
            if parent is None:
                raise ValueError('真实作品缺少生成祖先')
            route = actual.render_pipeline.value
            actual, workflow = prepare_handoff(artwork, parent,
                {'target': route, 'denoise': denoise, 'style_preset': actual.style_preset,
                 'resize_to': (actual.width, actual.height),
                 **{key: getattr(actual, key) for key in ('steps', 'cfg', 'sampler', 'scheduler', 'shift',
                     'cfg_rescale', 'weight_dtype', 'lora_name', 'lora_strength', 'seed') if getattr(actual, key) is not None}},
                self.settings, context_request=actual, materialize=False)
            input_bytes = Path(artwork.file_path).read_bytes()
        elif method == 'research_pixels':
            options = record['input_snapshot'].get('research_pixel_use')
            if options is None:
                raise ValueError('研究图只获观察用途，未明确采用全幅像素；没有升级权限或派发')
            snapshot = self._validate_research_support(record)
            if snapshot is None:
                raise ValueError('研究像素缺少绑定的实际来源；未借用其他作品')
            self._validate_authoring_recipes(record, record['preparation_recipe_snapshot'])
            actual, workflow, input_bytes = prepare_research_pixel_handoff(actual, snapshot,
                ResearchPixelUse.model_validate(options), self.settings, materialize=False)
        elif method == 'local_edit':
            if artwork is None or source_context is None or region_edit is None:
                raise ValueError('自动局部修改需要同源原图、完整场景与已定位遮罩；没有猜补。')
            parent = generation_ancestor(artwork, self.artworks, self.jobs)
            actual, workflow, owned_inputs = prepare_region_edit(artwork, parent, region_edit,
                self.settings, context_request=actual)
            actual = self._attach_request_reading(record, actual)
            proposal = deepcopy(actual.author_proposal or {})
            local = record['continuation']['local_edit']
            proposal.update(condition_status='scene_context_with_local_override_unverified',
                automatic_local_edit={'selection_id': local['selection_id'],
                    'source_sha256': local['source_sha256'], 'mask_sha256': local['mask_sha256'],
                    'instruction': region_edit['instruction'],
                    'origin': record['continuation']['feedback'].get('origin', 'user_continuation'),
                    'geometry_accuracy_verified': False, 'quality_verified': False})
            actual = actual.model_copy(update={'author_proposal': proposal,
                'output_group': f'creative_sessions/{record["id"]}',
                'output_label': f'local-edit__candidate-{len(record["candidates"])+1}__from-{artwork.id}'})
        elif method != 'text':
            raise ValueError('未启用的生成方法，不静默转换为文生图')
        # Materialisation may have changed the control state or source files.
        # This check is before any queue call; only this call's new input may
        # be removed when it is certain that nothing has been submitted.
        try:
            self._check(record)
            if len(record['candidates']) >= record['limits']['jobs']:
                raise InterruptedError('本次候选预算耗尽，没有新派发。')
            if input_bytes is not None:
                if hashlib.sha256(input_bytes).hexdigest() != actual.image_handoff.source_sha256:
                    raise ValueError('交接来源在准备后变化，未创建旧图输入或派发')
                input_path = Path(actual.image_handoff.source_file).resolve()
                input_root = (Path(self.settings.engine_base_dir) / 'input').resolve()
                if (input_path.parent != input_root
                        or not re.fullmatch(r'handoff_[0-9a-f]{32}_source\.png', input_path.name)):
                    raise ValueError('准备器未提供本次独占交接输入，不创建或删除共享文件')
                input_root.mkdir(parents=True, exist_ok=True)
                # An exclusive create establishes ownership. A path merely
                # being in the engine input directory does not permit deletion.
                with input_path.open('xb') as handle:
                    owned_inputs.append(input_path)
                    handle.write(input_bytes)
            self._check(record)
            verify_handoff_input(actual, self.settings)
            if method == 'local_edit':
                protected_inputs(actual, self.settings)
            self._validate_research_support(record)
            if method == 'research_pixels':
                self._validate_authoring_recipes(record, record['preparation_recipe_snapshot'])
            if method == 'handoff':
                current_source = Path(artwork.file_path).resolve()
                if hashlib.sha256(current_source.read_bytes()).hexdigest() != actual.image_handoff.source_sha256:
                    raise ValueError('交接原图在准备后变化，未提交旧图条件')
            if source_context:
                self._validate_continuation_source(record, source_context)
                if recipe_snapshot is not None:
                    self._validate_authoring_recipes(record, recipe_snapshot)
                proposal = deepcopy(actual.author_proposal or {})
                proposal['continuation_source_receipt'] = self._continuation_source_receipt(record, source_context)
                update = {'author_proposal': proposal}
                if method == 'text':
                    update['parent_artwork_id'] = source_context['artwork']['id']
                actual = GenerateRequest.model_validate(actual.model_dump(mode='json') | update)
        except BaseException:
            for owned_input in owned_inputs:
                owned_input.unlink(missing_ok=True)
            raise
        submitted_request = actual.model_dump(mode='json')
        if method == 'local_edit':
            record['continuation']['local_edit'].update(prepared_request=deepcopy(submitted_request),
                workflow_sha256=self._request_digest(workflow), inputs=[str(path) for path in owned_inputs])
        operation = record.get('operation') or {}
        dispatch = {'id': uuid.uuid4().hex[:12], 'method': method,
            'purpose': operation.get('purpose'),
            'origin': (operation.get('action') or {}).get('initiator', 'initial_system_operation'),
            'action_request_id': (operation.get('action') or {}).get('request_id'),
            'request': submitted_request, 'request_sha256': self._request_digest(submitted_request),
            'workflow_sha256': self._request_digest(workflow) if workflow is not None else None,
            'source_artwork_id': (actual.image_handoff.source_artwork_id if actual.image_handoff else
                                  actual.parent_artwork_id if source_context else None),
            'source_material_id': actual.image_handoff.source_material_id if actual.image_handoff else None,
            'source_sha256': (actual.image_handoff.source_sha256 if actual.image_handoff else
                             actual.repair_source_sha256 if method == 'local_edit' else None),
            'continuation_source_receipt': deepcopy((actual.author_proposal or {}).get('continuation_source_receipt')),
            'research_pixel_use_receipt': deepcopy((actual.iteration_context or {}).get('research_pixel_use')),
            'submission_started': True, 'submission_uncertain': False,
            'job_id': None, 'completed': False, 'status': 'reserved'}
        assignment = (actual.author_proposal or {}).get('direction_assignment')
        direction_slot = None
        if assignment is not None:
            dispatch['direction_id'] = assignment['id']
            plan = record.get('direction_plan') or {}
            if source_context is None and plan.get('active_slot_id') == assignment['id']:
                direction_slot = next(slot for slot in plan['slots'] if slot['id'] == assignment['id'])
                direction_slot.update(dispatch_id=dispatch['id'], status='dispatching',
                    generation_request=deepcopy(submitted_request))
            dispatch['direction_role'] = 'independent_candidate' if direction_slot is not None else 'revision'
        if method == 'local_edit':
            dispatch['method_origin'] = record['continuation']['feedback'].get('origin', 'user_continuation')
        record.setdefault('generation_dispatches', []).append(dispatch)
        try:
            self._save(record)  # Durable reservation BEFORE the queue await.
        except BaseException:
            # This particular save failure precedes the queue call, so it is
            # known not to have submitted, unlike a failed submission await.
            dispatch.update(submission_started=False, completed=True, status='not_submitted')
            for owned_input in owned_inputs:
                owned_input.unlink(missing_ok=True)
            raise
        try:
            job = (await self.jobs.submit_workflow(actual, workflow) if workflow is not None
                   else await self.jobs.submit(actual))
        except BaseException:
            dispatch.update(submission_uncertain=True, status='submission_uncertain')
            if self.events[record['id']].is_set():
                record['cancel_error'] = '生成提交回执未确认；未能确认后端取消，未重复派发。'
            self._save(record)
            if self.events[record['id']].is_set():
                raise asyncio.CancelledError()
            raise
        # Store the actual receipt before observing. Unknown/failed responses
        # do not trigger a second submission or become a fictional source.
        dispatch.update(job_id=job.id, status='submitted',
            job_request_sha256=self._request_digest(job.request))
        if direction_slot is not None:
            direction_slot.update(job_id=job.id, artwork_id=self.artworks.id_for(job.id, 0), status='submitted')
        if method == 'local_edit':
            record['continuation']['local_edit']['job_id'] = job.id
        candidate = {'job_id': job.id, 'artwork_id': self.artworks.id_for(job.id, 0),
            'method': method, 'reason': (record['input_snapshot']['research_pixel_use']['reason']
                if method == 'research_pixels' else region_edit['instruction']
                if method == 'local_edit' else record['plan']['support_reason']), 'status': job.status,
            'image_url': None, 'file_path': None, 'dispatch_id': dispatch['id']}
        if dispatch['source_artwork_id']:
            candidate['parent_artwork_id'] = dispatch['source_artwork_id']
        if assignment is not None:
            candidate.update(direction_id=assignment['id'], direction_role=dispatch['direction_role'])
        record['candidates'].append(candidate)
        record['active_job_id'] = job.id
        cycle = record.get('result_cycle') or {}
        result_turn = next((item for item in cycle.get('turns', [])
            if item['id'] == cycle.get('active_turn_id')), None)
        fresh_dispatch = bool(method == 'text' and source_context is None and result_turn
            and result_turn.get('continuation') is record.get('continuation')
            and record['continuation'].get('reference_contract_version') in {REFERENCE_CHOICE_CONTRACT, FULL_REFERENCE_CHOICE_CONTRACT}
            and (record['continuation'].get('reference_choice') or {}).get('reference_base') == 'neither'
            and (record['continuation'].get('choice') or {}).get('operation') == 'regenerate')
        if (result_turn and (source_context is not None or fresh_dispatch)
                and cycle.get('operation_key') == self._request_digest(record.get('operation') or {})):
            result_turn['continuation']['job_id'] = job.id
            result_turn['child_artwork_id'] = candidate['artwork_id']
        purpose = (record.get('operation') or {}).get('purpose')
        if purpose == 'continue' and record.get('continuation'):
            record['continuation']['job_id'] = job.id
        elif purpose == 'regenerate':
            record['regeneration_job_id'] = job.id
        self._save(record)
        return await self._finish_job(record, job)

    def _generation_dispatch_for_job(self, record, job):
        dispatch = next((item for item in record.get('generation_dispatches', [])
                         if item.get('job_id') == job.id), None)
        if dispatch and (self._request_digest(dispatch['request']) != dispatch['request_sha256']
                or self._request_digest(job.request) != dispatch['job_request_sha256']):
            raise ValueError('持久生成回执的请求已变化，不采用其他任务或重复派发')
        return dispatch

    async def _finish_job(self, record, job):
        dispatch = self._generation_dispatch_for_job(record, job)
        candidate = next(c for c in record['candidates'] if c['job_id'] == job.id)
        while job.status not in {'completed', 'failed', 'cancelled'}:
            if self.events[record['id']].is_set():
                try:
                    await self.jobs.cancel(job.id)
                except Exception as exc:
                    record['cancel_error'] = str(exc) or type(exc).__name__
                    self._save(record)
                    raise asyncio.CancelledError() from exc
                if job.status in {'completed', 'failed', 'cancelled'}:
                    break  # Persist this actual terminal receipt before stopping.
                record['cancel_error'] = '原任务取消回执未确认，未宣称后端采样已停止。'
                self._save(record)
                raise asyncio.CancelledError()
            if self._elapsed(record) >= record['limits']['active_seconds']:
                await self.jobs.cancel(job.id)
                self._commit_terminal_cancellation(record, origin='session_timeout_cleanup')
                raise InterruptedError('累计执行预算已耗尽，本项目当前任务已请求取消；不派下一项。')
            await asyncio.sleep(.5)
        self._generation_dispatch_for_job(record, job)
        record['active_job_id'] = None
        self.artworks.sync_jobs([job])
        candidate.update(status=job.status, image_url=job.images[0] if job.images else None,
                         file_path=job.files[0] if job.files else None, error=job.error)
        pending = record.get('pending_methods')
        if pending and pending[0] == candidate['method']:
            pending.pop(0)
        if dispatch:
            dispatch.update(completed=True, status=job.status)
        if candidate.get('direction_role') == 'independent_candidate':
            slot = next(item for item in record['direction_plan']['slots']
                if item['id'] == candidate['direction_id'] and item['job_id'] == job.id)
            slot['status'] = job.status
        self._save(record)
        if job.status != 'completed' or not job.images:
            if self.events[record['id']].is_set():
                raise asyncio.CancelledError()
            raise RuntimeError(job.error or '候选没有成功图片；不会自动重派或用失败输出交接')
        self._check(record)
        return self.artworks.get(candidate['artwork_id'])

    async def _observe(self, record, artwork, *, source_context=None, input_contract_version=None):
        self._check(record)
        if source_context:
            self._validate_continuation_source(record, source_context)
        record.update(stage='observing', detail='查看实际原图中的关系、局部价值及问题；不是自动质量批准。')
        self._save(record)
        path = Path(artwork.file_path).resolve()
        if not path.is_relative_to(Path(self.settings.output_dir).resolve()):
            raise ValueError('观察原图不在作品目录')
        # Reject a large file before inference without first allocating all of
        # it. Do not downscale or silently replace the selected original.
        limit = RuntimeConfig().max_image_bytes
        with path.open('rb') as handle:
            raw = handle.read(limit + 1)
        if len(raw) > limit:
            raise ValueError('观察原图超过单图字节上限；没有缩小、替换或调用模型。')
        image = VisualImageInput(raw, hashlib.sha256(raw).hexdigest(), source_ref=artwork.id)
        if source_context and (artwork.id != source_context['artwork']['id']
                or image.expected_sha256 != source_context['image']['sha256']):
            raise ValueError('观察像素不是绑定的所选原图；未继续旧选择。')
        source_choices = self._source_author_choices(source_context) if source_context else None
        data = {'original_text': record['input_snapshot']['sentence'],
            'current_purpose': source_choices['visual_logic'] if source_choices else record['plan']['visual_logic'],
            'artwork_id': artwork.id, 'feedback': record['feedback'],
            'evidence_authority': {
                'pixels': 'only evidence of the current visible arrangement',
                'original_text': 'requested intent, not a description of achieved pixels',
                'current_purpose': 'replaceable author intention, not evidence of achieved pixels',
                'feedback': 'raw attributed judgement/instruction; preserve origin; colour/style value does not cancel a rejected relationship'},
            'review_focus': 'Visible anatomy and contact surfaces; how objects are supported/attached; '
                'inside/outside and depth order; the attention/action relation needed by the original. '
                'Do not demand hidden fingers, realism or unnecessary props. Unclear details stay uncertain.'}
        self._with_request_reading(record, data)
        if source_context:
            data.update(selected_source={'artwork': _artwork_identity(source_context['artwork']),
                'image': deepcopy(source_context['image']), 'snapshot_sha256': source_context['snapshot_sha256'],
                'feedback': {key: _raw_words(value) for key, value in source_context['feedback'].items()
                    if isinstance(value, list)},
                'projection_scope': 'pixel_identity_and_original_feedback_not_intended_native_conditions'},
                        source_author_choice_status=source_choices['status'])
        system = ('Describe the actual image for the current creative purpose. '
            'Do not repeat the intended prompt as observation. Separate visible relationships, local value, '
            'problems and uncertainty. Lists have upper bounds, NOT quotas; an empty list is valid. '
            'Report a problem only when visible evidence supports an execution fault or an unmet relationship '
            'in original_text/current_purpose. Describe where and why it matters; separate an uncertain reading '
            'from a definite fault. A crop, limited palette, stylization, flat drawing, hidden body part or bright '
            'light is NOT inherently defective. Do not impose realism, full-body completeness or extra detail '
            'that the request does not need. Do not invent questions about age, natural versus simulated rain '
            'or other unspecified trivia. Keep only uncertainty that affects the next creative decision. '
            'One image cannot establish comparison to a nonexistent second direction. '
            'Describe the pixels before comparing with intentions. A plausible lighting mood is not evidence '
            'of correct fingers, contact or object support. Examine those visible relations explicitly; '
            'if resolution, occlusion or stylization prevents a reliable judgement, put that limitation in uncertain. '
            'An empty problems list means only no fault was reported, not that these aspects were checked or passed. '
            'No identity, emotion, gaze or quality certification. Return JSON only.')
        reading = None
        if input_contract_version == PIXEL_OBSERVATION_CONTRACT:
            from io import BytesIO
            from PIL import Image
            with Image.open(BytesIO(raw)) as picture:
                width, height = picture.size
            host_data = deepcopy(data)
            data = pixel_observation_input(artwork.id, image.expected_sha256, width, height, len(raw))
            system = PIXEL_OBSERVATION_SYSTEM
            index = len(record['calls'])
            reading = {'input_contract_version': PIXEL_OBSERVATION_CONTRACT,
                'artwork_id': artwork.id, 'source_sha256': image.expected_sha256,
                'source_snapshot_sha256': source_context['snapshot_sha256'] if source_context else None,
                'host_context': host_data, 'model_input': deepcopy(data), 'system_prompt': system,
                'response_schema_sha256': self._request_digest(PixelObservation.model_json_schema()),
                'input_binding_sha256': self._request_digest({'system_prompt': system, 'data': data,
                    'schema': PixelObservation.model_json_schema()}), 'call_index': index}
            record.setdefault('observation_requests', {})[str(index)] = reading
            self._save(record)
        elif input_contract_version is not None:
            raise ValueError('未知全幅观察合同；不猜测改写旧观察。')
        result = await self._call(record, system, data,
            PixelObservation.model_json_schema(), images=(image,))
        if source_context:
            self._validate_continuation_source(record, source_context)
        observation = PixelObservation.model_validate(result.data).model_dump(mode='json')
        record.setdefault('observations', []).append({'artwork_id': artwork.id,
            'source_sha256': image.expected_sha256, 'observation': observation, 'verified': False,
            'coverage': 'single model reading; empty problems does not certify anatomy, contact, spatial relationships or quality',
            **({'input_contract_version': PIXEL_OBSERVATION_CONTRACT,
                'call_index': reading['call_index'], 'input_binding_sha256': reading['input_binding_sha256']}
               if reading else {})})
        self._save(record)
        return observation

    def _edit_context(self, record, artwork_id):
        """Recover the selected version's conditions, never another candidate's.

        Current masked execution is NewBie-only. This is an implementation
        limit disclosed to the user, not a permanent model responsibility.
        """
        artwork = self.artworks.get(artwork_id)
        if artwork is None:
            raise ValueError('圈选原图不存在')
        parent = generation_ancestor(artwork, self.artworks, self.jobs)
        if parent is None or parent.status != 'completed' or not parent.images:
            raise ValueError('圈选需要有成功原图及真实生成条件的候选')
        source = GenerateRequest.model_validate(parent.request)
        if source.creation_intent is None or source.creation_intent.original_text != record['input_snapshot']['sentence']:
            raise ValueError('原图缺少本创作的完整原意关联，没有借用另一候选的条件')
        if source.model_id.value == 'newbie':
            if source.lora_name:
                from .lora_catalog import describe_lora
                root = (Path(self.settings.model_root) / 'loras').resolve()
                path = (root / source.lora_name).resolve()
                if (not path.is_relative_to(root) or not path.is_file()
                        or 'newbie' not in describe_lora(path, root)['compatible_models']):
                    raise ValueError('原图 LoRA 当前缺少可用的 NewBie 兼容证据，没有静默移除')
            return artwork, parent, source
        native = (source.native_prompts or {}).get('newbie') or {}
        if (not str(native.get('positive', '')).strip() or 'negative' not in native
                or native.get('trace_status') == 'historical_projection_after_native_override'):
            raise ValueError('该原图缺少当前有效的完整 NewBie 局部修改条件，没有借用其他模型的文本格式')
        if source.lora_name:
            raise ValueError('当前局部修改为 NewBie；所选 LoRA 跨模型兼容性未确认，没有静默移除')
        context = GenerateRequest.model_validate({**PROFILES['newbie'],
            'model_id': 'newbie', 'render_pipeline': 'newbie', 'parameter_policy': 'explicit',
            'prompt': native['positive'], 'negative_prompt': native['negative'],
            'use_default_negative_prompt': False, 'seed': source.seed,
            'native_prompts': source.native_prompts, 'semantic_spec': source.semantic_spec,
            'scene_contract': source.scene_contract, 'author_proposal': source.author_proposal,
            'creation_intent': source.creation_intent, 'check_enabled': source.check_enabled,
            'face_detail_enabled': False, 'highres_enabled': False})
        return artwork, parent, context

    def _local_edit_capability(self, record, source, compatible_models, node_snapshot):
        capability = {'available': False, 'targets': ['newbie'],
            'source_artwork_id': source['artwork']['id'], 'source_sha256': source['image']['sha256'],
            'source_dimensions': {key: source['image'][key] for key in ('width', 'height')},
            'steps': RegionEditOptions.model_fields['steps'].default,
            'denoise_range': {'minimum': .15, 'maximum': .75},
            'required_calls_after_choice': 2,
            'scope': 'existing_masked_consumer_not_a_permanent_model_role_or_quality_claim'}
        if node_snapshot is None or compatible_models is None or 'newbie' not in compatible_models:
            return {**capability, 'reason': '当前未确认完整NewBie局部消费者的资源、条件及实际节点。'}, None
        try:
            _, _, request = self._edit_context(record, source['artwork']['id'])
            workflow = build_region_repair_workflow(request, self.settings,
                'local-edit-capability-source.png', 'local-edit-capability-mask.png', 'capability-only')
            nodes = _node_evidence(workflow, node_snapshot, deferred_image_files={
                'local-edit-capability-source.png', 'local-edit-capability-mask.png'})
        except (ValueError, OSError) as exc:
            return {**capability, 'reason': str(exc)}, None
        return {**capability, 'available': True, 'reason': '同源完整场景、兼容资源和遮罩执行节点已确认。',
            'nodes': nodes, 'instruction': 'objective is the local change; preserve is not approval',
            'geometry': 'one current original PNG must be visually localized; no text coordinates or masks',
            'quality_verified': False}, request.model_dump(mode='json')

    async def _locate_local_edit(self, record, context, choice):
        source = context['source_context']
        self._validate_continuation_source(record, source)
        entry = context.get('local_edit')
        binding = self._request_digest({'source_snapshot_sha256': source['snapshot_sha256'],
            'choice': choice.model_dump(mode='json'), 'context_request': context['local_edit_context']})
        if entry is None:
            entry = {'binding_sha256': binding, 'source_snapshot_sha256': source['snapshot_sha256'],
                'source_sha256': source['image']['sha256'], 'choice': choice.model_dump(mode='json'),
                'choice_call_index': context.get('choice_call_index'),
                'observation_id': 'creative-region-' + uuid.uuid4().hex[:24],
                'localization_call_index': None, 'raw': None, 'selection_id': None,
                'status': 'pending', 'fallback_choice': None, 'fallback_call_index': None}
            context['local_edit'] = entry
            self._save(record)
        elif entry['binding_sha256'] != binding:
            raise ValueError('局部意图、同源原图或完整场景已变化；没有沿用旧定位或重跑。')
        if entry['raw'] is None:
            recovered = self._returned_cycle_call(record, entry, 'localization_call_index', RegionLocalization)
            if recovered is None:
                entry['localization_call_index'] = len(record['calls'])
                record.update(stage='locating_region', detail='根据当前原PNG定位本次局部意图；不猜坐标或自动判断错手。')
                self._save(record)
                metadata = source['image']
                with Path(metadata['file_path']).open('rb') as stream:
                    raw = stream.read(RuntimeConfig().max_image_bytes + 1)
                if (len(raw) > RuntimeConfig().max_image_bytes
                        or hashlib.sha256(raw).hexdigest() != metadata['sha256']):
                    raise ValueError('定位原PNG已变化或超过原图上限；没有缩图或替换。')
                image = VisualImageInput(raw, metadata['sha256'], source_ref=source['artwork']['id'])
                result = await self._call(record, LOCALIZATION_SYSTEM,
                    {'original_text': record['input_snapshot']['sentence'],
                     'region_request': {'observation_id': entry['observation_id'],
                        'artifact_id': source['artwork']['id'], 'source_sha256': metadata['sha256'],
                        'width': metadata['width'], 'height': metadata['height'],
                        'instruction': choice.objective, 'coordinate_space': 'original_image_pixels'},
                     'feedback': deepcopy(context['feedback']), 'preserve': choice.preserve,
                     'scope': 'actual_pixels_and_attributed_edit_intent_not_parent_prompt_as_observation'},
                    RegionLocalization.model_json_schema(), images=(image,))
                recovered = deepcopy(result.data)
            entry['raw'] = deepcopy(recovered)
            self._save(record)
        self._validate_continuation_source(record, source)
        result = bind_localization(entry['raw'], entry['observation_id'], source['artwork']['id'], source['image'])
        entry['status'] = result.status
        if result.status == 'unlocatable':
            self._save(record)
            return None
        folder = Path(self.settings.output_dir) / 'creative_regions' / record['id']
        if entry['selection_id'] is None:
            original = Path(source['image']['file_path']).read_bytes()
            mask, overlay = rasterize_localization(result, original)
            self._check(record)
            self._validate_continuation_source(record, source)
            selection = 'region-' + uuid.uuid4().hex[:24]
            folder.mkdir(parents=True, exist_ok=True)
            mask_path, overlay_path = folder / (selection + '-mask.png'), folder / (selection + '-overlay.png')
            mask_path.write_bytes(mask)
            overlay_path.write_bytes(overlay)
            entry.update(selection_id=selection, mask_path=str(mask_path.resolve()),
                overlay_path=str(overlay_path.resolve()), mask_sha256=hashlib.sha256(mask).hexdigest(),
                overlay_sha256=hashlib.sha256(overlay).hexdigest(), geometry_accuracy_verified=False)
            for kind, path in (('mask', mask_path), ('overlay', overlay_path)):
                record['materials'].append({'id': selection + '-' + kind,
                    'image_url': '/generated/' + path.relative_to(Path(self.settings.output_dir)).as_posix(),
                    'kind': 'localization_' + kind + '_unverified',
                    'source_artwork_id': source['artwork']['id'], 'source_sha256': source['image']['sha256']})
            self._save(record)
        mask_path = Path(entry['mask_path']).resolve()
        if not mask_path.is_relative_to(folder.resolve()):
            raise ValueError('局部遮罩不属于本创作保存的定位；没有借用外部遮罩。')
        mask = mask_path.read_bytes()
        if hashlib.sha256(mask).hexdigest() != entry['mask_sha256']:
            raise ValueError('保存的局部遮罩已变化；没有重定位或派发。')
        return {'instruction': choice.objective, 'mask_data_url': png_data_url(mask),
            'source_sha256': source['image']['sha256'], 'steps': RegionEditOptions.model_fields['steps'].default,
            'denoise': choice.denoise}

    def _art_choice_input(self, record, holder, data, source, call_key):
        """Freeze only a new purpose view; source/native execution stays separate."""
        if holder.get('art_decision_view_version') is None:
            if holder.get(call_key) is not None:
                return data, ''  # An old reserved call keeps its original contract.
            holder['art_decision_view_version'] = ART_DECISION_VIEW
            holder['art_decision_receipt_policy'] = 'known_host_prepare_receipts_20261013'
            holder['art_decision_full_input'] = deepcopy(data)
            holder['art_decision_full_input_sha256'] = self._request_digest(data)
            projected, receipt = art_decision_material(data, bound_sources={'selected': source},
                include_preparation_receipts=True)
            holder['art_decision_input'] = projected
            holder['art_decision_allocation'] = receipt
            holder['art_decision_system_scope'] = ART_DECISION_SYSTEM
            self._save(record)
        if holder.get('art_decision_view_version') != ART_DECISION_VIEW:
            raise ValueError('未知创作判断用途合同；不升级或重新调用。')
        complete = holder.get('art_decision_full_input')
        if (not isinstance(complete, dict)
                or self._request_digest(complete) != holder.get('art_decision_full_input_sha256')):
            raise ValueError('创作判断的完整原件已变化；不重建或重复调用。')
        policy = holder.get('art_decision_receipt_policy')
        if policy not in (None, 'known_host_prepare_receipts_20261013'):
            raise ValueError('未知准备收据用途策略；不升级已冻结输入。')
        projected, receipt = art_decision_material(complete, bound_sources={'selected': source},
            include_preparation_receipts=policy == 'known_host_prepare_receipts_20261013')
        if projected != holder.get('art_decision_input') or receipt != holder.get('art_decision_allocation'):
            raise ValueError('创作判断与冻结原件不一致；不将用途材料作为生图条件。')
        return deepcopy(holder['art_decision_input']), '\n\n' + holder['art_decision_system_scope']

    def _unavailable_art_choice(self, record, context, raw, available_actions, compatible_models, *, key='choice_rejection'):
        code = ('operation_unavailable_for_reference' if raw['operation'] not in available_actions else
            'target_not_compatible_with_current_conditions' if raw['operation'] not in {'keep', 'stop'}
                and compatible_models is not None and raw['target'] not in compatible_models else None)
        if code is None:
            return False
        context[key] = {'code': code, 'raw_proposal': deepcopy(raw),
            'available_actions': deepcopy(available_actions), 'compatible_models': deepcopy(compatible_models),
            'adopted': False}
        context.update(completed=True, stop_reason='原图与模型原建议已保存；建议的方法或目标不在当前可执行范围，'
            '本轮未采用、未派图或重试。已有作品仍可查看并明确继续。')
        record['detail'] = context['stop_reason']
        self._save(record)
        return True

    async def _local_edit_fallback(self, record, context, data, available_actions):
        entry = context['local_edit']
        if entry['fallback_choice'] is None:
            actions = [item for item in available_actions if item != 'local_edit']
            if self.get(record['id'])['remaining_budget']['calls'] < 3:
                actions = ['keep', 'stop']
            if self.get(record['id'])['remaining_budget']['calls'] < 1:
                context.update(completed=True, stop_reason='无法可靠定位，且剩余调用不足以选择其他方法；未修改原图。')
                self._save(record)
                return None
            recovered = None if self._joint_enabled(record) else self._returned_cycle_call(record, entry, 'fallback_call_index', NextChoice)
            if recovered is None:
                fallback_data = deepcopy(data)
                fallback_data.update(localization_unverified=deepcopy(entry['raw']), available_actions=actions,
                    remaining_budget=self.get(record['id'])['remaining_budget'])
                fallback_data, art_scope = self._art_choice_input(record, entry, fallback_data,
                    context['source_context'], 'fallback_call_index')
                record.update(stage='choosing', detail='定位未成立，选择其他有依据的方法；不重定位或猜遮罩。')
                self._save(record)
                system = ('The requested local edit could not be reliably localized. '
                    'Choose ONE other currently available operation, or stop. Do not guess a mask, retry '
                    'localization, claim the image was edited or invent user approval. ')
                if self._joint_enabled(record):
                    holder = context.setdefault('joint_fallback_decision', {})
                    decision = await self._joint_model_call(record, holder, NextProseDecision,
                        system + NEXT_PROSE_DECISION_SYSTEM + art_scope, fallback_data)
                    entry['fallback_call_index'] = holder['call_index']
                    recovered = decision.choice.model_dump(mode='json')
                else:
                    entry['fallback_call_index'] = len(record['calls'])
                    self._save(record)
                    result = await self._call(record, system + 'Return JSON only.' + art_scope,
                        fallback_data, NextChoice.model_json_schema())
                    recovered = deepcopy(result.data)
            entry['fallback_choice'] = deepcopy(recovered)
            compatible = ((context.get('compatible_model_context') or {}).get('compatible_starts'))
            if self._unavailable_art_choice(record, context, recovered, actions, compatible):
                return None
            context.update(choice=deepcopy(recovered), choice_call_index=entry['fallback_call_index'])
            record['decisions'].append({**deepcopy(recovered), 'based_on_artwork': context['artwork_id'],
                'origin': ('assistant_validation_localization_fallback_not_user_feedback'
                    if context['feedback'].get('origin', '').startswith('assistant_validation')
                    else 'localization_fallback_not_user_feedback')})
            self._save(record)
        return NextChoice.model_validate(entry['fallback_choice'])

    async def _local_edit(self, record, action):
        self._check(record)
        checkpoint = record['local_edit']
        if checkpoint.get('completed'):
            return
        if checkpoint.get('job_id'):
            job = self.jobs.get(checkpoint['job_id'])
            if job is None:
                raise ValueError('局部修改已提交但原回执暂不可见，不重复派发')
            await self._finish_job(record, job)
            checkpoint['completed'] = True
            self._save(record)
            return
        if checkpoint.get('submission_started'):
            raise ValueError('局部修改提交回执未确认，不在恢复时重复派发；请核对原队列')
        if len(record['candidates']) >= record['limits']['jobs']:
            raise InterruptedError('本次候选预算耗尽，没有派发局部修改。')
        artwork, parent, _ = self._edit_context(record, checkpoint['artwork_id'])
        context = GenerateRequest.model_validate(checkpoint['context_request'])
        payload = action.region_edit.model_dump(mode='json')
        payload['source_sha256'] = checkpoint['source_sha256']
        record.update(stage='local_edit', detail='按当前原图圈选修改；完整场景作上下文，圈内以本次修改为准。')
        self._save(record)
        actual, workflow, inputs = prepare_region_edit(artwork, parent, payload, self.settings,
            context_request=context)
        record['active_direction_assignment'] = deepcopy((context.author_proposal or {}).get('direction_assignment'))
        actual = self._attach_request_reading(record, actual)
        try:
            self._check(record)
        except BaseException:
            # No submit was attempted: these unique task-owned inputs are
            # safe to remove, unlike an uncertain queue receipt's inputs.
            for path in inputs:
                path.unlink(missing_ok=True)
            raise
        actual = actual.model_copy(update={
            'output_group': f'creative_sessions/{record["id"]}',
            'output_label': f'local-edit__candidate-{len(record["candidates"])+1}__from-{artwork.id}'})
        # Save the attempt boundary before awaiting the queue. An unknown
        # receipt can never be resumed as a second submission.
        checkpoint.update(submission_started=True, request=actual.model_dump(mode='json'))
        self._save(record)
        try:
            job = await self.jobs.submit_workflow(actual, workflow)
        except BaseException:
            # The queued workflow may already reference these files. Keep the
            # source/mask until the unknown receipt is explicitly resolved.
            checkpoint['submission_uncertain'] = True
            self._save(record)
            raise
        candidate = {'job_id': job.id, 'artwork_id': self.artworks.id_for(job.id, 0),
            'parent_artwork_id': artwork.id, 'method': 'local_edit',
            'reason': action.region_edit.instruction, 'status': job.status,
            'image_url': None, 'file_path': None}
        if (actual.author_proposal or {}).get('direction_assignment') is not None:
            candidate.update(direction_id=actual.author_proposal['direction_assignment']['id'], direction_role='revision')
        record['candidates'].append(candidate)
        record['active_job_id'] = job.id
        checkpoint['job_id'] = job.id
        self._save(record)
        await self._finish_job(record, job)
        checkpoint['completed'] = True
        self._save(record)

    @staticmethod
    def _same_neutral_pixels(old_source, current_source, contract_version):
        if contract_version not in {PIXEL_OBSERVATION_CONTRACT, FULL_VISIBLE_DELTA_CONTRACT}:
            return False
        try:
            return (old_source['artwork']['id'] == current_source['artwork']['id']
                and isinstance(old_source['image']['sha256'], str)
                and all(old_source['image'][key] == current_source['image'][key]
                    for key in ('sha256', 'width', 'height', 'bytes', 'format')))
        except (KeyError, TypeError):
            return False

    @staticmethod
    def _returned_cycle_call(record, context, key, schema):
        index = context.get(key)
        if index is None:
            return None  # Known not to have entered a model call yet.
        if type(index) is not int or index < 0 or index > len(record['calls']):
            raise ValueError('结果回合的调用预约不完整；不猜测或重跑。')
        if index == len(record['calls']):
            return None
        call = record['calls'][index]
        if call.get('status') != 'returned' or 'data' not in call:
            raise ValueError('结果回合已用调用但没有完整回执；不自动重试。')
        if (key == 'comparison_call_index' and schema == ResultComparison
                and (context.get('comparison_material') or {}).get('contract_version') == FULL_VISIBLE_DELTA_CONTRACT):
            model_input = context.get('full_comparison_model_input')
            if (not isinstance(model_input, dict)
                    or model_input != full_visible_delta_input(context['full_comparison_host_input'])
                    or call.get('input_contract_version') != FULL_VISIBLE_DELTA_CONTRACT
                    or call.get('response_schema_sha256') != CreativeSessions._request_digest(schema.model_json_schema())
                    or call.get('input_binding_sha256') != CreativeSessions._request_digest({
                        'system_prompt': context.get('full_comparison_system_prompt'), 'data': model_input,
                        'schema': schema.model_json_schema()})):
                raise ValueError('全幅中性比較没有同一原图和原用途调用绑定；不借旧回应。')
        if (key == 'comparison_call_index' and schema == ResultComparison
                and (context.get('comparison_material') or {}).get('contract_version') == LOCAL_VISIBLE_DELTA_CONTRACT):
            expected_input = local_visible_delta_input(context.get('local_comparison_host_input'))
            model_input = context.get('local_comparison_model_input', expected_input)
            system = context.get('local_comparison_system_prompt', LOCAL_VISIBLE_DELTA_SYSTEM)
            if (not isinstance(model_input, dict) or model_input != expected_input
                    or system != LOCAL_VISIBLE_DELTA_SYSTEM
                    or call.get('input_contract_version') != LOCAL_VISIBLE_DELTA_CONTRACT
                    or call.get('response_schema_sha256') != CreativeSessions._request_digest(schema.model_json_schema())
                    or call.get('input_binding_sha256') != CreativeSessions._request_digest({
                        'system_prompt': system, 'data': model_input, 'schema': schema.model_json_schema()})):
                raise ValueError('局部中性比较没有同一裁片输入与调用绑定；不借旧回应。')
        if key == 'observation_call_index' and schema == PixelObservation:
            version = context.get('observation_contract_version')
            if version == PIXEL_OBSERVATION_CONTRACT:
                reading = (record.get('observation_requests') or {}).get(str(index))
                source = context.get('source_context')
                if not isinstance(source, dict):
                    raise ValueError('全幅中性观察没有当前原图绑定；不借其他图的回应继续。')
                image = source['image']
                expected_input = pixel_observation_input(source['artwork']['id'], image['sha256'],
                    image['width'], image['height'], image['bytes'])
                if (not isinstance(reading, dict) or reading.get('input_contract_version') != version
                        or reading.get('model_input') != expected_input
                        or reading.get('artwork_id') != source['artwork']['id']
                        or reading.get('source_sha256') != image['sha256']
                        or call.get('input_contract_version') != version
                        or call.get('image_bindings') != [{'source_ref': source['artwork']['id'], 'sha256': image['sha256']}]
                        or call.get('response_schema_sha256') != reading.get('response_schema_sha256')
                        or call.get('input_binding_sha256') != reading.get('input_binding_sha256')
                        or reading.get('input_binding_sha256') != CreativeSessions._request_digest({
                            'system_prompt': reading.get('system_prompt'), 'data': reading.get('model_input'),
                            'schema': PixelObservation.model_json_schema()})):
                    raise ValueError('全幅中性观察的原合同或输入绑定不一致；不将旧回应改称新观察。')
            elif version is not None or call.get('input_contract_version') is not None:
                raise ValueError('全幅观察原合同未知或缺失；不以同schema升级旧回应。')
        return schema.model_validate(call['data']).model_dump(mode='json')

    async def _choose_result_reference(self, record, turn, context, data, compatible_models, node_snapshot):
        """One existing purpose call chooses both reference and operation.

        The observation cursor is not an adopted reference. Full source packets
        are frozen before this call; a parent choice swaps every execution-bound
        field, not merely the artwork ID. Explicit user continuations never enter.
        """
        joint = self._joint_enabled(record)
        reference_schema = ReferenceProseDecision if joint else ReferenceNextChoice
        previous = next((item for item in record['result_cycle']['turns']
            if item['id'] == turn.get('reference_from_turn_id')), None)
        if (previous is None or previous.get('phase') != 'reviewed'
                or (previous.get('comparison_material') or {}).get('contract_version')
                    not in {LOCAL_VISIBLE_DELTA_CONTRACT, FULL_VISIBLE_DELTA_CONTRACT}):
            raise ValueError('参照选择没有同源中性变化回执；不补猜或升级旧比较。')
        if context.get('reference_context') is None:
            original = record['input_snapshot']['sentence']
            sources, observations, capabilities = {}, {}, {}
            for role, identifier in (('parent', previous['source_artwork_id']),
                                     ('child', previous['child_artwork_id'])):
                source = freeze_continuation_source(identifier, self.artworks, self.jobs,
                    original_text=original, original_sha256=hashlib.sha256(original.encode()).hexdigest(),
                    output_root=Path(self.settings.output_dir), session_feedback=record['feedback'])
                self._archive_continuation_source(record, source)
                sources[role] = source
                observed = previous.get('source_observation') if role == 'parent' else context['observation']
                observed_source = ((previous.get('continuation') or {}).get('source_context')
                    if role == 'parent' else context['source_context'])
                observation_contract = ((previous.get('continuation') or {}).get('observation_contract_version')
                    if role == 'parent' else context.get('observation_contract_version'))
                if observed is not None and (observed_source == source or self._same_neutral_pixels(
                        observed_source, source, observation_contract)):
                    observations[role] = {'scope': 'full_original_png', 'artwork_id': identifier,
                        'source_sha256': source['image']['sha256'],
                        'source_snapshot_sha256': source['snapshot_sha256'], 'observation': deepcopy(observed)}
                    if observation_contract in {PIXEL_OBSERVATION_CONTRACT, FULL_VISIBLE_DELTA_CONTRACT}:
                        observations[role].update(input_contract_version=observation_contract,
                            observed_source_snapshot_sha256=observed_source['snapshot_sha256'],
                            current_feedback_verified=False,
                            use='same_actual_pixels_unverified_reading_not_latest_feedback_or_quality_review')
                capability, edit_request = self._local_edit_capability(
                    record, source, compatible_models, node_snapshot)
                actions = [item for item in data['available_actions'] if item != 'local_edit']
                if (capability['available'] and edit_request is not None and 'regenerate' in actions
                        and self.get(record['id'])['remaining_budget']['calls'] >= 3):
                    actions.append('local_edit')
                capabilities[role] = {'available_actions': actions,
                    'local_edit_capability': capability, 'local_edit_context': edit_request}
            full_pair = (previous.get('comparison_material') or {}).get('contract_version') == FULL_VISIBLE_DELTA_CONTRACT
            binder = bind_full_reference_context if full_pair else bind_reference_context
            context['reference_context'] = binder(previous, sources['parent'], sources['child'],
                original_text=original, observations=observations)
            context['reference_capabilities'] = capabilities
            context['reference_contract_version'] = FULL_REFERENCE_CHOICE_CONTRACT if full_pair else REFERENCE_CHOICE_CONTRACT
            self._save(record)
        if context.get('reference_contract_version') not in {REFERENCE_CHOICE_CONTRACT, FULL_REFERENCE_CHOICE_CONTRACT}:
            raise ValueError('未知参照选择合同；没有换语法或重做旧选择。')
        bound = context['reference_context']
        for snapshot in bound['sources'].values():
            self._validate_continuation_source(record, snapshot)
        if context.get('reference_selection_applied'):
            choice, selected = resolve_reference_choice(bound, context['reference_choice'])
            if selected is not None and context.get('source_context') != selected:
                raise ValueError('已选参照与后继执行源不一致；没有改源或重新选择。')
            # Later localization fallback and a newly completed full observation
            # belong to this same source. Do not reset either on resume.
            return (context['reference_capabilities'][choice.reference_base]['available_actions']
                if selected is not None else context['reference_neither_capability']['available_actions'])
        purpose_data = deepcopy(data)
        for key in ('selected_source', 'current_plan', 'current_picture_event',
                    'source_author_choice_status', 'pixel_observation_unverified',
                    'local_edit_capability', 'result_comparison_unverified'):
            purpose_data.pop(key, None)
        if (context.get('reference_choice_input') is None
                and context.get('choice_call_index') is None):
            context['reference_model_view_version'] = REFERENCE_CONTENT_VIEW
            context['reference_art_view_version'] = ART_DECISION_VIEW
            context['reference_art_system_scope'] = ART_DECISION_SYSTEM
        purpose_data.update(reference_model_material(bound,
            current_creation_intent=data.get('creation_intent'),
            current_request_reading_material=data.get('request_reading_material'),
            current_user_selections=data['current_user_selections'],
            view_version=context.get('reference_model_view_version')))
        for role, capabilities in context['reference_capabilities'].items():
            purpose_data['reference_options'][role]['method_capabilities'] = {
                key: deepcopy(value) for key, value in capabilities.items() if key != 'local_edit_context'}
        purpose_data['neither_capability'] = {'available_actions': (
            ['regenerate', 'stop'] if 'regenerate' in data['available_actions'] else ['stop']),
            'scope': 'fresh_current_conditions_only_no_rejected_image_base'}
        purpose_data['current_picture_task'] = {'kind': 'one_reference_and_one_operation',
            'maximum_new_pictures_from_this_operation': 1,
            'reference_lineage': {role: {key: deepcopy(assignment[key])
                for key in ('id', 'ordinal', 'picture_key') if key in assignment}
                for role, snapshot in bound['sources'].items()
                for assignment in [(snapshot['source_job']['request'].get('author_proposal') or {}).get('direction_assignment') or {}]},
            'collection_execution': 'Independent directions belong to the host; choosing parent or child '
                'continues that reference, and neither proposes one fresh frame, never a multi-picture objective.'}
        if context.get('reference_choice_input') is None:
            if context.get('reference_model_view_version') == REFERENCE_CONTENT_VIEW:
                # Freeze the complete purpose packet separately from what this
                # consumer reads. The exact bound source requests still drive
                # execution; symbolic content references cannot drive a job.
                context['reference_choice_full_input'] = deepcopy(purpose_data)
                context['reference_choice_full_input_sha256'] = self._request_digest(purpose_data)
                if context.get('reference_art_view_version') == ART_DECISION_VIEW:
                    if context.get('choice_call_index') is None:
                        context['reference_art_receipt_policy'] = 'known_host_prepare_receipts_20261013'
                    purpose_data, art_allocation = art_decision_material(purpose_data,
                        bound_sources=bound['sources'], include_preparation_receipts=(
                            context.get('reference_art_receipt_policy') == 'known_host_prepare_receipts_20261013'))
                    context['reference_choice_art_allocation'] = art_allocation
                purpose_data, allocation = compact_reference_content(purpose_data)
                context['reference_choice_content_allocation'] = allocation
            context['reference_choice_input'] = deepcopy(purpose_data)
            context['reference_choice_schema_sha256'] = self._request_digest(reference_schema.model_json_schema())
            context['reference_choice_input_sha256'] = self._request_digest(purpose_data)
            context['reference_neither_capability'] = deepcopy(purpose_data['neither_capability'])
            self._save(record)
        elif (context['reference_choice_schema_sha256'] != self._request_digest(reference_schema.model_json_schema())
                or context['reference_choice_input_sha256'] != self._request_digest(context['reference_choice_input'])):
            raise ValueError('参照选择的已预约输入或schema改变；不重写或重做选择。')
        if context.get('reference_model_view_version') == REFERENCE_CONTENT_VIEW:
            complete = context.get('reference_choice_full_input')
            if not isinstance(complete, dict) or self._request_digest(complete) != context.get('reference_choice_full_input_sha256'):
                raise ValueError('参照选择的完整用途原件改变；没有重建或重派。')
            expected = complete
            if context.get('reference_art_view_version') == ART_DECISION_VIEW:
                policy = context.get('reference_art_receipt_policy')
                if policy not in (None, 'known_host_prepare_receipts_20261013'):
                    raise ValueError('未知参照准备收据用途策略；不升级已冻结输入。')
                expected, art_allocation = art_decision_material(complete, bound_sources=bound['sources'],
                    include_preparation_receipts=policy == 'known_host_prepare_receipts_20261013')
                if art_allocation != context.get('reference_choice_art_allocation'):
                    raise ValueError('参照创作判断与完整冻结原件不一致；没有代换或重派。')
            elif context.get('reference_art_view_version') is not None:
                raise ValueError('未知参照创作判断合同；没有升级或重新调用。')
            expected, allocation = compact_reference_content(expected)
            if (expected != context['reference_choice_input']
                    or allocation != context.get('reference_choice_content_allocation')):
                raise ValueError('参照判断内容与完整冻结原件不一致；没有代换条件或输入。')
        if context.get('reference_choice') is None:
            recovered = None if joint else self._returned_cycle_call(record, context, 'choice_call_index', ReferenceNextChoice)
            if recovered is None:
                try:
                    art_scope = ('\n\n' + context['reference_art_system_scope']
                        if context.get('reference_art_view_version') == ART_DECISION_VIEW else '')
                    if joint:
                        holder = context.setdefault('joint_reference_decision', {})
                        decision = await self._joint_model_call(record, holder, ReferenceProseDecision,
                            reference_choice_system(context.get('reference_model_view_version'),
                                contract_version=context['reference_contract_version'])
                                + REFERENCE_PROSE_DECISION_SYSTEM + art_scope,
                            context['reference_choice_input'])
                        context['choice_call_index'] = holder['call_index']
                        recovered = decision.choice.model_dump(mode='json')
                    else:
                        context['choice_call_index'] = len(record['calls'])
                        self._save(record)
                        result = await self._call(record, reference_choice_system(
                            context.get('reference_model_view_version'), contract_version=context['reference_contract_version']) + art_scope, context['reference_choice_input'],
                            ReferenceNextChoice.model_json_schema())
                        recovered = ReferenceNextChoice.model_validate(result.data).model_dump(mode='json')
                except LocalAuthoringError as error:
                    if error.code != 'context_budget_exceeded':
                        raise
                    # Both works and the actual delta already exist. A purpose
                    # consumer that cannot read the complete material is a
                    # bounded stop, not permission to trim it, adopt raw visual
                    # base, retry the call or strand those works as a failed job.
                    context.update(completed=True,
                        stop_reason='两版本和局部变化已保存；完整参照决策材料超过当前模型容量，'
                            '没有选参照、重调或派图。可查看两图并明确继续所选版本。')
                    self._save(record)
                    return None
            context['reference_choice'] = recovered
            self._save(record)
        choice, selected = resolve_reference_choice(bound, context['reference_choice'])
        capabilities = (context['reference_capabilities'][choice.reference_base]
            if selected is not None else purpose_data['neither_capability'])
        for snapshot in bound['sources'].values():
            self._validate_continuation_source(record, snapshot)
        rejection = None
        if choice.operation not in capabilities['available_actions']:
            rejection = 'operation_unavailable_for_reference'
        elif (choice.operation not in {'keep', 'stop'} and compatible_models is not None
                and choice.target not in compatible_models):
            rejection = 'target_not_compatible_with_current_conditions'
        if rejection is not None:
            # The returned proposal is an original, not an adopted operation.
            # Keep the pair usable without silently turning the proposal into
            # another action, retrying the model or spending an image budget.
            context['reference_choice_rejection'] = {
                'code': rejection, 'raw_proposal': deepcopy(context['reference_choice']),
                'available_actions': deepcopy(capabilities['available_actions']),
                'compatible_models': deepcopy(compatible_models), 'adopted': False}
            context.update(completed=True,
                stop_reason='两版本与模型原建议已保存；建议的方法或目标不在当前可执行范围，'
                    '没有采用参照、改派图或重试。已有版本仍可查看并明确继续。')
            self._save(record)
            return None
        selected_observation = (bound['observations'][choice.reference_base].get('observation')
            if selected is not None else None)
        if selected is not None:
            selected_packet = bound['observations'][choice.reference_base]
            if context['artwork_id'] != selected['artwork']['id']:
                # A parent's reading did not originate at the child's call.
                context.pop('observation_call_index', None)
            selected_contract = selected_packet.get('input_contract_version')
            if selected_contract in {PIXEL_OBSERVATION_CONTRACT, FULL_VISIBLE_DELTA_CONTRACT} and selected_observation is not None:
                context['observation_contract_version'] = selected_contract
            else:
                # An old full reading must not inherit the new child's marker.
                context.pop('observation_contract_version', None)
            context.update(artwork_id=selected['artwork']['id'], source_context=deepcopy(selected),
                observation=deepcopy(selected_observation),
                local_edit_capability=deepcopy(capabilities['local_edit_capability']),
                local_edit_context=deepcopy(capabilities['local_edit_context']))
            turn['source_artwork_id'] = context['artwork_id']
            record['result_cycle']['cursor_artwork_id'] = context['artwork_id']
        context['choice'] = NextChoice.model_validate({key: value
            for key, value in choice.model_dump(mode='json').items() if key != 'reference_base'}).model_dump(mode='json')
        previous['reference_decision'] = {'contract_version': context['reference_contract_version'],
            'reference_base': choice.reference_base, 'selected_artwork_id': selected['artwork']['id'] if selected else None,
            'selected_source_snapshot_sha256': selected['snapshot_sha256'] if selected else None,
            'choice': deepcopy(context['reference_choice']), 'choice_call_index': context['choice_call_index']}
        if not context.get('reference_decision_recorded'):
            record['decisions'].append({**context['reference_choice'],
                'based_on_artwork': selected['artwork']['id'] if selected else None,
                'origin': 'system_result_cycle', 'reference_turn_id': previous['id']})
            context['reference_decision_recorded'] = True
        context['reference_selection_applied'] = True
        self._save(record)
        return capabilities['available_actions']

    async def _continue(self, record, action, *, result_turn=None):
        context = record.get('continuation')
        if context and context['completed']:
            return
        # Once submitted, recover only that real receipt. A new review or a
        # removed source is not permission to rerun observation/choice/dispatch.
        if context and context.get('job_id'):
            job = self.jobs.get(context['job_id'])
            if job is None:
                raise ValueError('后继已提交但回执暂不可见，未重复派发')
            await self._finish_job(record, job)
            context['completed'] = True
            self._save(record)
            return
        pending = self._pending_generation_dispatch(record)
        if pending and not pending.get('job_id'):
            raise ValueError('后继提交回执未确认，未重新观察、选择或派发')
        autonomous = result_turn is not None
        reference_mode = autonomous and result_turn.get('reference_from_turn_id') is not None
        validation = not autonomous and action.initiator == 'assistant_validation'
        acceptance = record.get('continuation_source_acceptance')
        accepted_source = (acceptance.get('artwork_id')
            if not autonomous and isinstance(acceptance, dict)
                and acceptance.get('action_request_id') == action.request_id
            else record['selected_artwork_id'])
        artwork_id = (context['artwork_id'] if context else result_turn['source_artwork_id'] if autonomous
                      else action.artwork_id or accepted_source)
        if not artwork_id:
            raise ValueError('请选择当前候选再继续；不会猜测你喜欢哪张图')
        artwork = self.artworks.get(artwork_id)
        if artwork is None:
            raise ValueError('当前作品不存在')
        if context is None:
            feedback = ({'artwork_id': artwork_id, 'text': '', 'origin': 'system_result_cycle_not_user_feedback'}
                if autonomous else {'artwork_id': artwork_id, 'text': action.feedback,
                                    'created_at': datetime.now(UTC).isoformat(),
                                    **({'origin': 'assistant_validation_not_user_feedback',
                                        'scope': 'assistant_validation_continuation',
                                        'action_request_id': action.request_id} if validation else {})})
            original = record['input_snapshot']['sentence']
            append_feedback = not autonomous and (not validation or bool(action.feedback))
            source = freeze_continuation_source(artwork_id, self.artworks, self.jobs,
                original_text=original, original_sha256=hashlib.sha256(original.encode('utf-8')).hexdigest(),
                output_root=Path(self.settings.output_dir),
                session_feedback=[*record['feedback'], feedback] if append_feedback else record['feedback'])
            context = {'feedback': feedback, 'artwork_id': artwork_id, 'observation': None,
                       'choice': None, 'request': None, 'completed': False, 'source_context': source}
            self._archive_continuation_source(record, source)
            record['continuation'] = context
            if autonomous:
                result_turn['continuation'] = context
            elif append_feedback:
                record['feedback'].append(feedback)
            self._save(record)
        source = context.get('source_context')
        if source is None:
            raise ValueError('旧后继没有绑定所选版本的完整源条件；请明确重新继续，不猜补旧观察出处。')
        self._validate_continuation_source(record, source)
        record['active_direction_assignment'] = deepcopy((source['source_job']['request'].get('author_proposal') or {}).get('direction_assignment'))
        feedback = context['feedback']
        if context['observation'] is None:
            cached = result_turn.get('source_observation') if autonomous else None
            neutral_cached = (cached and result_turn.get('source_observation_contract_version') in {PIXEL_OBSERVATION_CONTRACT, FULL_VISIBLE_DELTA_CONTRACT}
                and result_turn.get('source_observation_artwork_id') == source['artwork']['id']
                and result_turn.get('source_observation_sha256') == source['image']['sha256'])
            if cached and result_turn.get('cached_source_snapshot_sha256') != source['snapshot_sha256'] and not neutral_cached:
                raise ValueError('缓存观察的原图、条件或评价已变化；不借旧观察继续。')
            recovered = self._returned_cycle_call(record, context, 'observation_call_index', PixelObservation)
            if cached or recovered:
                context['observation'] = deepcopy(cached or recovered)
                if neutral_cached:
                    context['observation_contract_version'] = result_turn['source_observation_contract_version']
            else:
                if context.get('observation_call_index') is None:
                    context['observation_contract_version'] = PIXEL_OBSERVATION_CONTRACT
                context['observation_call_index'] = len(record['calls'])
                self._save(record)
                context['observation'] = await self._observe(record, artwork, source_context=source,
                    input_contract_version=context.get('observation_contract_version'))
            self._save(record)
        observation = context['observation']
        self._validate_continuation_source(record, source)
        self._check(record)
        record.update(stage='choosing', detail='依据当前图与评价选择后继，不预排模型链。')
        available_actions = ['keep', 'regenerate', 'handoff', 'stop']
        if record['input_snapshot']['support_mode'] in {'sketch', 'compare'}:
            available_actions.append('sketch')
        previous_comparison = result_turn.get('previous_comparison') if autonomous else None
        if context.get('cycle_learning_context') is None:
            context['cycle_learning_context'] = cycle_learning_context(record,
                result_turn if autonomous else None,
                record_reference=str((self.root / (record['id'] + '.json')).resolve()))
            self._save(record)
        cycle_context = context['cycle_learning_context']
        research_purpose = ('iterated_successor_choice' if cycle_context.get('attempted_operations')
                            else 'successor_choice')
        if (not reference_mode and previous_comparison
                and previous_comparison.get('provisional_base') == 'neither'):
            available_actions = ['regenerate', 'stop']
        if self.get(record['id'])['remaining_budget']['jobs'] <= 0:
            available_actions = ['keep', 'stop']
        source_choices = self._source_author_choices(source)
        current_inputs = authoring_input(self._record_request(record),
            record['input_snapshot']['config'], self.settings)
        current_inputs, successor_parameter_sources = self._successor_configuration_input(record, current_inputs)
        if context.get('choice_recipe_snapshot') is None:
            context['choice_recipe_snapshot'] = deepcopy(current_inputs['full_style_recipes'])
            self._save(record)
        self._validate_authoring_recipes(record, context['choice_recipe_snapshot'])
        compatible_models, node_snapshot = None, None
        from .comfy_client import ComfyClient
        if isinstance(getattr(self.jobs, 'client', None), ComfyClient):
            if context.get('compatible_model_context') is None:
                node_snapshot = await self._available_node_snapshot(record)
                proposal_input = deepcopy(current_inputs)
                proposal_input['selected_config']['render_pipeline'] = 'adaptive'
                context['compatible_model_context'] = prepare_starting_choice(proposal_input,
                    self.settings, parameter_sources=successor_parameter_sources,
                    object_info=node_snapshot, research_context=load_creative_research_context(research_purpose))
                self._save(record)
            compatible_models = context['compatible_model_context']['compatible_starts']
            if not compatible_models:
                available_actions = ['keep', 'stop']
        if context.get('local_edit_capability') is None:
            capability, edit_context = self._local_edit_capability(record, source, compatible_models, node_snapshot)
            context.update(local_edit_capability=capability, local_edit_context=edit_context)
            self._save(record)
        local_capability = deepcopy(context['local_edit_capability'])
        if (local_capability['available'] and context.get('local_edit_context') is not None
                and self.get(record['id'])['remaining_budget']['calls'] >= 3
                and 'regenerate' in available_actions):
            available_actions.append('local_edit')
        else:
            local_capability.update(available=False,
                reason=local_capability.get('reason') if not local_capability['available'] else
                    '剩余调用不足以明确选择、定位及比较；没有扩额度。')
        data = {'original_text': record['input_snapshot']['sentence'], 'current_plan': source_choices['support_plan'],
            'current_picture_event': source_choices['picture_event'],
            'source_author_choice_status': source_choices['status'],
            'current_user_selections': current_inputs,
            'selection_authority': 'source saved request belongs to the selected image but is not independent workflow verification; current user selections guide a new proposal',
            'pixel_observation_unverified': observation, 'feedback': feedback,
            'evidence_authority': {
                'current_plan_and_picture_event': 'intended parent conditions, not current pixel observations',
                'pixel_observation_unverified': 'fallible reading of this bound image; do not replace it with the intended plan',
                'feedback': 'raw correction/judgement; no invented approval or preservation request',
                'empty_problems': 'no reported fault, never proof of critical-detail coverage or a quality pass'},
            'applicable_research': load_creative_research_context(research_purpose),
            'result_comparison_unverified': deepcopy(cycle_context['result_comparison_unverified']),
            'cycle_learning_context': cycle_history_view(cycle_context),
            'candidate_collection': completed_candidate_context(record),
            'picture_operation_scope': deepcopy(SINGLE_PICTURE_SCOPE),
            'local_edit_capability': local_capability,
            'source_selection_origin': ('system_provisional_reference_not_user_selection' if autonomous else
                'assistant_validation_explicit_source_not_user_selection' if validation else 'user_continuation'),
            'available_actions': available_actions,
            'available_models': list(compatible_models) if compatible_models is not None else list(PROFILES),
            'compatible_model_parameters': (compatible_choice_material(context['compatible_model_context'])
                if compatible_models is not None else None),
            'incompatible_model_reasons': (deepcopy(context['compatible_model_context']['excluded_starts'])
                if compatible_models is not None else None),
            'remaining_budget': self.get(record['id'])['remaining_budget']}
        assignment = record.get('active_direction_assignment') or {}
        data['current_picture_task'] = {'kind': 'one_operation_on_selected_actual_artwork',
            'artwork_id': artwork_id, 'maximum_new_pictures_from_this_operation': 1,
            'direction_lineage': {key: deepcopy(assignment[key]) for key in ('id', 'ordinal', 'picture_key')
                if key in assignment},
            'collection_execution': 'Pending independent directions belong to the host collection executor; '
                'this decision chooses one operation, not delivery of all directions in its objective.'}
        self._with_request_reading(record, data)
        if record.get('active_direction_assignment') is not None:
            data['direction_assignment'] = deepcopy(record['active_direction_assignment'])
        data['selected_source'] = continuation_model_material(source, for_model_call=True,
            successor_selection=True, current_creation_intent=data.get('creation_intent'),
            current_request_reading_material=data.get('request_reading_material'))
        if self._joint_enabled(record):
            self._attach_art_material(record, data, purpose='next_picture', feedback=feedback, source=source)
            data['picture_decision_requirements'] = {
                'picture': 'one complete drawable frame for regenerate or handoff, null for keep/stop/local_edit',
                'current_operation': 'raw feedback and explicit scope govern; prior unfinished items are not automatic edit targets',
                'method_limits': 'handoff inherits all source pixels; masked sampling cannot promise exact line repair',
            }
        if reference_mode:
            available_actions = await self._choose_result_reference(
                record, result_turn, context, data, compatible_models, node_snapshot)
            if available_actions is None:
                return
            source = context['source_context']
            artwork_id = context['artwork_id']
            artwork = self.artworks.get(artwork_id)
            if artwork is None:
                raise ValueError('参照选择的实际作品不可见；未派发。')
            observation = context['observation']
            fresh = context['reference_choice']['reference_base'] == 'neither'
            if (observation is None and not fresh and context['choice']['operation'] not in {'keep', 'stop'}
                    and context.get('joint_reference_decision') is None):
                context['observation_contract_version'] = PIXEL_OBSERVATION_CONTRACT
                context['observation_call_index'] = len(record['calls'])
                self._save(record)
                context['observation'] = await self._observe(record, artwork, source_context=source,
                    input_contract_version=PIXEL_OBSERVATION_CONTRACT)
                observation = context['observation']
            context['cycle_learning_context'] = cycle_learning_context(record, result_turn,
                record_reference=str((self.root / (record['id'] + '.json')).resolve()))
            cycle_context = context['cycle_learning_context']
            record['active_direction_assignment'] = deepcopy(
                (source['source_job']['request'].get('author_proposal') or {}).get('direction_assignment')) if not fresh else None
            # Localization failure still uses the old one-reference chooser.
            # Its material must describe the reference just selected, not the
            # child whose full observation preceded the purpose choice.
            chosen_source_choices = self._source_author_choices(source)
            data.update(current_plan=chosen_source_choices['support_plan'],
                current_picture_event=chosen_source_choices['picture_event'],
                source_author_choice_status=chosen_source_choices['status'],
                pixel_observation_unverified=deepcopy(observation),
                local_edit_capability=deepcopy(context['local_edit_capability']),
                available_actions=available_actions,
                result_comparison_unverified=deepcopy(cycle_context['result_comparison_unverified']),
                cycle_learning_context=cycle_history_view(cycle_context),
                selected_source=continuation_model_material(source, for_model_call=True, successor_selection=True,
                    current_creation_intent=data.get('creation_intent'),
                    current_request_reading_material=data.get('request_reading_material')))
            if record['active_direction_assignment'] is None:
                data.pop('direction_assignment', None)
            else:
                data['direction_assignment'] = deepcopy(record['active_direction_assignment'])
            self._save(record)
        elif context['choice'] is None:
            choice_data, art_scope = self._art_choice_input(record, context, data, source, 'choice_call_index')
            if self._joint_enabled(record):
                holder = context.setdefault('joint_next_decision', {})
                decision = await self._joint_model_call(record, holder, NextProseDecision,
                    NEXT_PROSE_DECISION_SYSTEM + art_scope, choice_data)
                context['choice_call_index'] = holder['call_index']
                context['choice'] = decision.choice.model_dump(mode='json')
                record['decisions'].append({**context['choice'], 'based_on_artwork': artwork_id,
                    'origin': 'system_result_cycle' if autonomous else 'assistant_validation' if validation else 'user_continuation',
                    'joint_decision_call_index': holder['call_index']})
                self._save(record)
            recovered = context['choice'] if self._joint_enabled(record) else self._returned_cycle_call(
                record, context, 'choice_call_index', NextChoice)
            if recovered is None:
                context['choice_call_index'] = len(record['calls'])
                self._save(record)
            result = None if recovered is not None else await self._call(record, 'Choose ONE bounded next operation from actual image observations and '
                'natural-language feedback. Preserve local value, not a fixed model role or chain. keep/stop create no image. '
                'A handoff may damage layout/style; denoise is not a preservation percentage. sketch means new visual '
                'support for changed large shapes, regenerate means a new picture choice. Unreliable observations cannot '
                'approve quality. The parent plan/native conditions describe intent, never override observed layout. '
                'If observations and declarations disagree, do not describe the intended layout as actual pixels. '
                'Critical uncertain anatomy, contacts or scene relations are not cleared by attractive colours. '
                'Use applicable historical findings with their limits to select the operation and model. '
                'A structurally rejected source is not automatically a low-denoise base: consider a fresh full '
                'composition or another compatible method instead of repeating the same model/text without a reason. '
                'Use completed attempts on this same path and the complete latest parent-child comparison to '
                'understand what the last change gained or lost. Reusing a method is allowed when the new '
                'conditions give a reason; returning to a parent does not erase the failed child attempt. '
                'Also consider the completed candidate collection and the original purpose. If distinct '
                'alternatives are requested, one retained image is not that exploration; a useful next '
                'picture may develop a different composition, not just repair or re-seed the same one. '
                'This operation authors ONE composition: its objective must commit to that next direction, '
                'not ask its author to deliver all remaining alternatives in one picture. '
                'local_edit is available only when its explicit capability says so. Its objective is one '
                'complete local change, preferably in English; preserve identifies value to keep, not user '
                'approval. Current masked execution targets NewBie only; that is a consumer limit, not a '
                'permanent model role. Never supply coordinates or a mask. The host must visually locate '
                'the actual PNG first. Do not equate a detected hand with a faulty hand or assume that a '
                'small eye edit can fix a scene-wide attention relationship. '
                'Keep original requirements distinct from your proposed preservation choices; unrequested '
                'parent additions remain replaceable when they obscure the requested action or scene. '
                'Declared framing is not proof that those alternatives appeared in the actual pixels. '
                'Do not choose keep to pretend a requested change already happened. Return JSON only.' + art_scope,
                choice_data, NextChoice.model_json_schema())
            self._validate_continuation_source(record, source)
            self._validate_authoring_recipes(record, context['choice_recipe_snapshot'])
            context['choice'] = recovered or NextChoice.model_validate(result.data).model_dump(mode='json')
            if not self._joint_enabled(record):
                record['decisions'].append({**context['choice'], 'based_on_artwork': artwork_id,
                    'origin': 'system_result_cycle' if autonomous else 'assistant_validation' if validation else 'user_continuation'})
            self._save(record)
        choice = NextChoice.model_validate(context['choice'])
        fresh = bool(reference_mode and context['reference_choice']['reference_base'] == 'neither')
        if self._unavailable_art_choice(record, context, context['choice'], available_actions, compatible_models):
            return
        self._check(record)
        self._validate_continuation_source(record, source)
        if choice.operation == 'local_edit':
            if choice.target != 'newbie' or not .15 <= choice.denoise <= .75:
                raise ValueError('局部选择不符合现有遮罩消费者或强度范围；没有换目标、夹值或重试。')
            payload = await self._locate_local_edit(record, context, choice)
            if payload is not None:
                if self.get(record['id'])['remaining_budget']['calls'] < 1:
                    context.update(completed=True, stop_reason='定位已保存，但调用不足以比较局部后图；未派发。')
                    self._save(record)
                    return
                await self._generate(record, 'local_edit',
                    request=GenerateRequest.model_validate(context['local_edit_context']),
                    artwork=artwork, source_context=source, region_edit=payload)
                context['local_edit']['status'] = 'executed_unverified'
                context['completed'] = True
                self._save(record)
                return
            choice = await self._local_edit_fallback(record, context, data, available_actions)
            if choice is None:
                return
            if choice.operation not in available_actions or choice.operation == 'local_edit':
                raise ValueError('定位失败后的有效选择不在可用方法中；没有重定位或换回局部修改。')
            if (choice.operation not in {'keep', 'stop'} and compatible_models is not None
                    and choice.target not in compatible_models):
                raise ValueError('定位失败后的后继目标不消费当前冻结条件；没有改目标或派图。')
        if choice.operation in {'keep', 'stop'}:
            record['detail'] = choice.reason + '；没有认证合格或自动派图。'
            context['completed'] = True
            self._save(record)
            return
        joint_holder = (context.get('joint_fallback_decision') or context.get('joint_reference_decision')
            or context.get('joint_next_decision')) if self._joint_enabled(record) else None
        if context['request'] is None and joint_holder is not None:
            if (joint_holder.get('raw') or {}).get('picture') is None:
                raise ValueError('共同后继决定没有当前完整画面；未另开作者调用。')
            base_config = current_inputs['selected_config']
            current_target = {'anima_turbo': 'anima'}.get(base_config.get('model_id', 'newbie'),
                base_config.get('model_id', 'newbie'))
            successor_config = (deepcopy(compatible_models[choice.target]['config']) if compatible_models is not None else
                deepcopy(base_config) if choice.target == current_target else
                {**base_config, **PROFILES[choice.target], 'render_pipeline': choice.target})
            await self._compile_joint_successor(record, context, joint_holder, choice, successor_config,
                source=source, fresh=fresh)
        if context['request'] is None:
            if context.get('authoring_call_index', len(record['calls'])) < len(record['calls']):
                raise ValueError('后继构思已用调用但完整编译条件未保存；不自动重做作者回应。')
            context['authoring_call_index'] = len(record['calls'])
            self._save(record)
            current_config = record['input_snapshot']['config']
            current_target = {'anima_turbo': 'anima'}.get(current_config.get('model_id', 'newbie'),
                current_config.get('model_id', 'newbie'))
            successor_config = (deepcopy(compatible_models[choice.target]['config'])
                if compatible_models is not None else deepcopy(current_config) if choice.target == current_target else
                {**current_config, **PROFILES[choice.target], 'render_pipeline': choice.target})
            await self._prepare(record, feedback={**feedback, 'objective': choice.objective, 'preserve': choice.preserve},
                                observation=None if fresh else observation, source_context=None if fresh else source,
                                successor_operation=choice.operation, successor_config=successor_config,
                                cycle_context=cycle_context,
                                fresh_reference_rejection={'context': context['reference_context'],
                                    'choice': context['reference_choice']} if fresh else None)
            context['authoring_recipe_snapshot'] = deepcopy(record['preparation_recipe_snapshot'])
            self._validate_continuation_source(record, source)
            self._validate_authoring_recipes(record, context['authoring_recipe_snapshot'])
            actual = GenerateRequest.model_validate(record['generation_request'])
            if choice.target != actual.render_pipeline.value:
                if actual.lora_name:
                    raise ValueError('后继模型与已选 LoRA 的兼容性尚未确认；未静默移除角色／风格条件')
                target_config = (deepcopy(compatible_models[choice.target]['config'])
                    if compatible_models is not None else
                    {**record['input_snapshot']['config'], **PROFILES[choice.target], 'render_pipeline': choice.target})
                current_event = record.get('translated_picture_event') or record.get('picture_event')
                if current_event:
                    actual = compile_picture_support(PictureEvent.model_validate(current_event),
                        self._record_request(record), target_config,
                        self.settings, session_id=record['id'], evidence=record['calls'][-1])
                    self._tag_canonical_projection(record, actual)
                    self._tag_prose_transport(record, actual)
                else:
                    actual = compile_support(SupportPlan.model_validate(record['plan']),
                        self._record_request(record), target_config,
                        self.settings, session_id=record['id'], evidence=record['calls'][-1],
                        native_scene=NativeScene.model_validate(record['native_scene']) if record.get('native_scene') else None)
            actual = self._attach_request_reading(record, actual)
            context['request'] = actual.model_dump(mode='json')
            record.update(generation_request=deepcopy(context['request']),
                          native_prompts=deepcopy(actual.native_prompts))
            self._save(record)
        actual = GenerateRequest.model_validate(context['request'])
        method = {'regenerate': 'text', 'sketch': 'sketch', 'handoff': 'handoff'}[choice.operation]
        if context.get('authoring_recipe_snapshot') is None:
            raise ValueError('后继请求缺少本次完整配方快照；未把旧请求静默接成新派发。')
        self._validate_continuation_source(record, source)
        self._validate_authoring_recipes(record, context['authoring_recipe_snapshot'])
        self._check(record)
        await self._generate(record, method, request=actual, artwork=artwork, denoise=choice.denoise,
            source_context=None if fresh else source, recipe_snapshot=context['authoring_recipe_snapshot'])
        context['completed'] = True
        self._save(record)

    async def _drive_results(self, record):
        if not record['input_snapshot'].get('result_guided', False):
            return  # Historical sessions are never silently upgraded/replayed.
        cycle = record['result_cycle']
        operation_key = self._request_digest(record.get('operation') or {})
        if cycle.get('operation_key') != operation_key:
            superseded = next((x for x in cycle['turns'] if x['id'] == cycle.get('active_turn_id')), None)
            if superseded:
                superseded['superseded_by_explicit_operation'] = operation_key
            purpose = (record.get('operation') or {}).get('purpose')
            explicit = record.get('continuation') if purpose == 'continue' else None
            if explicit and explicit.get('stop_reason'):
                cycle.update(operation_key=operation_key, status='stopped',
                    cursor_artwork_id=explicit['artwork_id'], active_turn_id=None,
                    stop_reason=explicit['stop_reason'])
                self._save(record)
                return
            if explicit and (explicit.get('choice') or {}).get('operation') in {'keep', 'stop'}:
                cycle.update(operation_key=operation_key, status='stopped',
                    cursor_artwork_id=explicit['artwork_id'], active_turn_id=None,
                    stop_reason=explicit['choice']['reason'])
                self._save(record)
                return  # A no-image user continuation is not permission to reopen the latest unrelated work.
            completed = [c for c in record['candidates'] if c['status'] == 'completed' and c.get('file_path')]
            if not completed:
                return
            cycle.update(operation_key=operation_key, status='running', stop_reason=None,
                cursor_artwork_id=completed[-1]['artwork_id'], active_turn_id=None)
            cycle.pop('previous_comparison', None)
            cycle.pop('pending_reference_turn_id', None)
            cycle.pop('cached_observation', None)
            cycle.pop('cached_source_snapshot_sha256', None)
            for key in ('cached_observation_contract_version', 'cached_observation_artwork_id', 'cached_observation_sha256'):
                cycle.pop(key, None)
            if explicit and explicit.get('job_id'):
                child = next(c for c in completed if c['job_id'] == explicit['job_id'])
                turn = {'id': uuid.uuid4().hex[:12], 'source_artwork_id': explicit['artwork_id'],
                    'operation_key': operation_key,
                    'child_artwork_id': child['artwork_id'], 'phase': 'comparing',
                    'continuation': explicit, 'source_observation': deepcopy(explicit['observation']),
                    'previous_comparison': None}
                cycle['turns'].append(turn)
                cycle['active_turn_id'] = turn['id']
            elif purpose == 'local_edit':
                edit = record['local_edit']
                text = record['input_snapshot']['sentence']
                source = freeze_continuation_source(edit['artwork_id'], self.artworks, self.jobs,
                    original_text=text, original_sha256=hashlib.sha256(text.encode()).hexdigest(),
                    output_root=Path(self.settings.output_dir), session_feedback=record['feedback'])
                if source['image']['sha256'] != edit['source_sha256']:
                    raise ValueError('圈选父图已变化；不使用新父图解释旧遮罩修改。')
                self._archive_continuation_source(record, source)
                child = next(c for c in completed if c['job_id'] == edit['job_id'])
                turn = {'id': uuid.uuid4().hex[:12], 'source_artwork_id': edit['artwork_id'],
                    'operation_key': operation_key,
                    'child_artwork_id': child['artwork_id'], 'phase': 'comparing',
                    'source_observation': None, 'stop_after_comparison': True,
                    'continuation': {'completed': True, 'source_context': source,
                        'choice': {'objective': (record['operation']['action']['region_edit']['instruction']),
                            'preserve': '按明确遮罩修改；圈外保持，不由自动后继扩大为整图改动。'},
                        'origin': ('assistant_validation_region_edit_not_user_feedback'
                            if record['operation']['action'].get('initiator') == 'assistant_validation'
                            else 'explicit_user_region_edit_not_model_choice')}}
                cycle['turns'].append(turn)
                cycle['active_turn_id'] = turn['id']
            self._save(record)
        def stop(reason):
            cycle.update(status='stopped', stop_reason=reason, active_turn_id=None)
            record['detail'] = reason + '；候选全部保留，未认证合格或替用户选图。'
            self._save(record)
        cycle['status'] = 'running'
        while True:
            self._check(record)
            turn = next((x for x in cycle['turns'] if x['id'] == cycle.get('active_turn_id')), None)
            if turn is None:
                remaining = self.get(record['id'])['remaining_budget']
                if remaining['calls'] < (1 if cycle.get('cached_observation') else 2):
                    stop('剩余调用不足以观察和选择，停止新派发')
                    return
                turn = {'id': uuid.uuid4().hex[:12], 'source_artwork_id': cycle['cursor_artwork_id'],
                    'operation_key': operation_key,
                    'child_artwork_id': None, 'phase': 'advancing', 'continuation': None,
                    'source_observation': deepcopy(cycle.get('cached_observation')),
                    'cached_source_snapshot_sha256': cycle.get('cached_source_snapshot_sha256'),
                    'source_observation_contract_version': cycle.get('cached_observation_contract_version'),
                    'source_observation_artwork_id': cycle.get('cached_observation_artwork_id'),
                    'source_observation_sha256': cycle.get('cached_observation_sha256'),
                    'previous_comparison': deepcopy(cycle.get('previous_comparison'))}
                if cycle.get('pending_reference_turn_id') is not None:
                    turn['reference_from_turn_id'] = cycle.pop('pending_reference_turn_id')
                cycle['turns'].append(turn)
                cycle['active_turn_id'] = turn['id']
                record['continuation'] = None
                self._save(record)
            else:
                record['continuation'] = turn['continuation']
            if not turn.get('continuation') or not turn['continuation'].get('completed'):
                await self._continue(record, None, result_turn=turn)
                context = turn['continuation']
                turn['source_observation'] = deepcopy(context['observation'])
                if context.get('observation_contract_version') in {PIXEL_OBSERVATION_CONTRACT, FULL_VISIBLE_DELTA_CONTRACT}:
                    turn.update(source_observation_contract_version=context['observation_contract_version'],
                        source_observation_artwork_id=context['source_context']['artwork']['id'],
                        source_observation_sha256=context['source_context']['image']['sha256'])
                if turn['child_artwork_id'] is None:
                    turn['phase'] = 'no_image'
                    stop(context.get('stop_reason') or context['choice']['reason'])
                    return
                turn['phase'] = 'comparing'
                self._save(record)
            context = turn['continuation']
            self._validate_continuation_source(record, context['source_context'])
            if turn.get('child_source') is None:
                text = record['input_snapshot']['sentence']
                turn['child_source'] = freeze_continuation_source(turn['child_artwork_id'], self.artworks, self.jobs,
                    original_text=text, original_sha256=hashlib.sha256(text.encode()).hexdigest(),
                    output_root=Path(self.settings.output_dir), session_feedback=record['feedback'])
                self._archive_continuation_source(record, turn['child_source'])
                self._save(record)
            self._validate_continuation_source(record, turn['child_source'])
            if turn.get('comparison') is None:
                index = turn.get('comparison_call_index')
                if ((turn.get('comparison_material') or {}).get('contract_version') == LOCAL_VISIBLE_DELTA_CONTRACT
                        and type(index) is int and 0 <= index < len(record['calls'])
                        and record['calls'][index].get('status') == 'returned'):
                    # Recovery must recheck the same mask/crops and mechanical
                    # image labels, not just accept a well-formed response.
                    choice = context['choice']
                    host, pair_images = prepare_local_result_pair(context['source_context'],
                        turn['child_source'], self.settings,
                        original_text=record['input_snapshot']['sentence'],
                        objective=choice['objective'], preserve=choice['preserve'])
                    expected_input = local_visible_delta_input(host)
                    expected_images = [{'source_ref': image.source_ref, 'sha256': image.expected_sha256}
                        for image in pair_images]
                    if (host != turn.get('local_comparison_host_input')
                            or expected_input['comparison_material'] != turn['comparison_material']
                            or record['calls'][index].get('image_bindings') != expected_images):
                        raise ValueError('已返回局部比较的遮罩、裁片或机械来源改变；未借旧回应。')
                recovered = self._returned_cycle_call(record, turn, 'comparison_call_index', ResultComparison)
                if recovered is None:
                    if self.get(record['id'])['remaining_budget']['calls'] <= 0:
                        stop('调用预算已耗尽，子图尚未完成父子比较，不自动采用')
                        return
                    choice = context['choice']
                    local_pair = turn['child_source']['source_job']['request'].get('operation') == 'region_repair'
                    prepare_pair = prepare_local_result_pair if local_pair else prepare_result_pair
                    data, images = prepare_pair(context['source_context'], turn['child_source'], self.settings,
                        original_text=record['input_snapshot']['sentence'], objective=choice['objective'], preserve=choice['preserve'])
                    full_neutral = False
                    if local_pair:
                        version = (turn.get('comparison_material') or {}).get('contract_version')
                        if version not in {None, LOCAL_VISIBLE_DELTA_CONTRACT}:
                            raise ValueError('未知局部比较合同；没有替换旧材料或重调。')
                        # Only an entirely unreserved local call adopts the new
                        # contract. A saved/reserved legacy call keeps its input.
                        neutral = version == LOCAL_VISIBLE_DELTA_CONTRACT or (
                            turn.get('comparison_material') is None
                            and turn.get('comparison_call_index') is None)
                        if neutral:
                            host_input = deepcopy(data)
                            data = local_visible_delta_input(data)
                            if turn.get('local_comparison_host_input') is None:
                                turn['local_comparison_host_input'] = host_input
                            elif turn['local_comparison_host_input'] != host_input:
                                raise ValueError('宿主局部原句、目的或反馈材料已变化；未重调。')
                            if turn.get('local_comparison_model_input') is None:
                                turn['local_comparison_model_input'] = deepcopy(data)
                                turn['local_comparison_system_prompt'] = LOCAL_VISIBLE_DELTA_SYSTEM
                            elif (turn['local_comparison_model_input'] != data
                                    or turn.get('local_comparison_system_prompt') != LOCAL_VISIBLE_DELTA_SYSTEM):
                                raise ValueError('局部中性比较的已预约输入改变；未重调。')
                        material = data['comparison_material']
                        if (turn.get('comparison_material') is not None
                                and self._request_digest(turn['comparison_material']) != self._request_digest(material)):
                            raise ValueError('本次局部比较的原图、遮罩或派生像素绑定已变化；未换材料重调。')
                        turn['comparison_material'] = deepcopy(material)
                    else:
                        version = (turn.get('comparison_material') or {}).get('contract_version')
                        full_neutral = version == FULL_VISIBLE_DELTA_CONTRACT or (
                            version is None and turn.get('comparison_call_index') is None
                            and record['input_snapshot'].get('research_mode') == 'catalog')
                        if full_neutral:
                            host_input = deepcopy(data)
                            data = full_visible_delta_input(data)
                            if turn.get('full_comparison_host_input') is None:
                                turn['full_comparison_host_input'] = host_input
                            elif turn['full_comparison_host_input'] != host_input:
                                raise ValueError('全幅比较的宿主原意或实际目的改变；未重调。')
                            material = data['comparison_material']
                            if turn.get('comparison_material') is not None and turn['comparison_material'] != material:
                                raise ValueError('全幅比较的已预约原图绑定改变；未换图。')
                            turn['comparison_material'] = deepcopy(material)
                            if turn.get('full_comparison_model_input') is None:
                                turn['full_comparison_model_input'] = deepcopy(data)
                                turn['full_comparison_system_prompt'] = FULL_VISIBLE_DELTA_SYSTEM
                            elif turn['full_comparison_model_input'] != data:
                                raise ValueError('全幅中性比较已预约输入改变；未重调。')
                        else:
                            self._with_request_reading(record, data)
                    record.update(stage='comparing', detail=('按原遮罩精确区域记录可见变化及未知；目的选择另有依据，不认证整图。'
                        if local_pair else '记录两张原图的可见差异；用途与参照由后续决定。'
                        if full_neutral else '比较真实父图和子图的收益、损失及未知；不批准质量。'))
                    turn['comparison_call_index'] = len(record['calls'])
                    self._save(record)
                    result = await self._call(record,
                        (LOCAL_VISIBLE_DELTA_SYSTEM if neutral else LOCAL_RESULT_COMPARISON_SYSTEM)
                            if local_pair else turn['full_comparison_system_prompt'] if full_neutral else RESULT_COMPARISON_SYSTEM, data,
                        ResultComparison.model_json_schema(), images=images,
                        **({'visual_input_scope': 'protected_local_crop_pair'} if local_pair else
                            {'visual_input_scope': 'full_original_pair_visible_delta'} if full_neutral else {}))
                    recovered = ResultComparison.model_validate(result.data).model_dump(mode='json')
                self._validate_continuation_source(record, context['source_context'])
                self._validate_continuation_source(record, turn['child_source'])
                turn['comparison'] = recovered
                self._save(record)
            comparison = turn['comparison']
            base = comparison['provisional_base']
            neutral_contract = (turn.get('comparison_material') or {}).get('contract_version')
            neutral = neutral_contract in {LOCAL_VISIBLE_DELTA_CONTRACT, FULL_VISIBLE_DELTA_CONTRACT}
            if neutral:
                # This is only the next full-observation cursor. The purpose
                # call, not raw visual provisional_base, chooses the reference.
                cycle['cursor_artwork_id'] = turn['child_artwork_id']
                cycle['pending_reference_turn_id'] = turn['id']
            else:
                cycle['cursor_artwork_id'] = turn['source_artwork_id'] if base == 'parent' else turn['child_artwork_id']
            if (turn.get('comparison_material') or {}).get('kind') == 'protected_local_crop_pair':
                # A crop reading is never a whole-image PixelObservation. The
                # parent's existing full observation may still be reused; a
                # child cursor must observe its actual full source as usual.
                if not neutral and base == 'parent' and turn.get('source_observation') is not None:
                    cycle['cached_observation'] = deepcopy(turn['source_observation'])
                    cycle['cached_source_snapshot_sha256'] = context['source_context']['snapshot_sha256']
                else:
                    cycle.pop('cached_observation', None)
                    cycle.pop('cached_source_snapshot_sha256', None)
            elif neutral_contract == FULL_VISIBLE_DELTA_CONTRACT:
                cycle['cached_observation'] = deepcopy(comparison['current_observation'])
                cycle['cached_source_snapshot_sha256'] = turn['child_source']['snapshot_sha256']
            else:
                cycle['cached_observation'] = (turn['source_observation'] if base == 'parent'
                    else comparison['current_observation'])
                cycle['cached_source_snapshot_sha256'] = (context['source_context']['snapshot_sha256']
                    if base == 'parent' else turn['child_source']['snapshot_sha256'])
            if neutral_contract == FULL_VISIBLE_DELTA_CONTRACT:
                cycle.update(cached_observation_contract_version=FULL_VISIBLE_DELTA_CONTRACT,
                    cached_observation_artwork_id=turn['child_source']['artwork']['id'],
                    cached_observation_sha256=turn['child_source']['image']['sha256'])
            elif (cycle.get('cached_observation') is not None and base == 'parent'
                    and context.get('observation_contract_version') == PIXEL_OBSERVATION_CONTRACT):
                cycle.update(cached_observation_contract_version=PIXEL_OBSERVATION_CONTRACT,
                    cached_observation_artwork_id=context['source_context']['artwork']['id'],
                    cached_observation_sha256=context['source_context']['image']['sha256'])
            else:
                for key in ('cached_observation_contract_version', 'cached_observation_artwork_id', 'cached_observation_sha256'):
                    cycle.pop(key, None)
            cycle['previous_comparison'] = deepcopy(comparison)
            turn['phase'] = 'reviewed'
            cycle['active_turn_id'] = None
            self._save(record)
            if turn.get('stop_after_comparison'):
                stop('圈选修改的父子比较已保存；不自动扩大到圈外或整图改动')
                return
            if self.get(record['id'])['remaining_budget']['jobs'] <= 0:
                stop('图片预算已耗尽，父子比较保留，停止新派发')
                return

    def _comparison_completion_capability(self, record):
        """A declared tail operation; full source validation occurs on action."""
        unavailable = lambda reason: {'available': False, 'reason': reason}
        if record['status'] not in {'failed', 'paused', 'ready'}:
            return unavailable('当前操作尚未结束。')
        if (record.get('active_job_id') or record.get('cancel_error')
                or self._pending_generation_dispatch(record)):
            return unavailable('仍有待核对的图像派发或取消回执。')
        cycle = record.get('result_cycle') or {}
        turn = next((item for item in cycle.get('turns', [])
            if item['id'] == cycle.get('active_turn_id')), None)
        if not turn or turn.get('phase') != 'comparing' or turn.get('comparison') is not None:
            return unavailable('没有待补完的已保存父子比较。')
        if (turn.get('comparison_material') or {}).get('contract_version') != FULL_VISIBLE_DELTA_CONTRACT:
            return unavailable('此回合没有确切保存的全幅中性比较输入。')
        if not all(isinstance(turn.get(key), dict) for key in (
                'full_comparison_model_input', 'full_comparison_host_input', 'child_source')):
            return unavailable('原比較输入或来源尚未完整保存。')
        if not (turn.get('continuation') or {}).get('completed'):
            return unavailable('原后继派发尚未结束。')
        for identifier in (turn.get('source_artwork_id'), turn.get('child_artwork_id')):
            candidate = next((item for item in record['candidates'] if item['artwork_id'] == identifier), None)
            job = self.jobs.get(candidate['job_id']) if candidate else None
            if not candidate or job is None or job.status != 'completed' or not job.files:
                return unavailable('两张原图必须都已有完成回执。')
        attempts = [entry for entry in (record.get('comparison_completions') or {}).values()
            if entry.get('turn_id') == turn['id']]
        latest = attempts[-1] if attempts else None
        index = latest.get('comparison_call_index') if latest else turn.get('comparison_call_index')
        if index is None:
            mode = 'new_call'  # A saved reservation has not entered _call.
        elif type(index) is not int or not 0 <= index <= len(record['calls']):
            return unavailable('比较调用预约不完整。')
        elif index == len(record['calls']):
            mode = 'new_call'
        else:
            call = record['calls'][index]
            if call.get('status') == 'returned' and isinstance(call.get('data'), dict):
                mode = 'reuse_return'
            elif (call.get('status') == 'failed' and call.get('error_code') == 'invalid_neutral_visual_input'
                    and call.get('directory') is None):
                mode = 'new_call'  # This exact error is raised before reservation/worker.
            else:
                return unavailable('原比较已调用或回执未知，不重复提交。')
        if self._elapsed(record) >= record['limits']['active_seconds']:
            return unavailable('原回合执行时间额度已耗尽。')
        if mode == 'new_call' and len(record['calls']) >= record['limits']['calls']:
            return unavailable('原回合调用额度已耗尽，不借新回合额度。')
        return {'available': True, 'turn_id': turn['id'], 'mode': mode,
            'source_call_index': index, 'maximum_new_calls': int(mode == 'new_call'),
            'new_images': 0, 'resume_existing': bool(latest and latest.get('status') == 'paused'
                and record['status'] == 'paused'),
            'scope': 'saved_pair_comparison_only_no_author_or_generation'}

    def _comparison_completion_inputs(self, record, turn):
        source = turn['continuation']['source_context']
        child = turn['child_source']
        self._validate_continuation_source(record, source)
        self._validate_continuation_source(record, child)
        choice = turn['continuation']['choice']
        host, images = prepare_result_pair(source, child, self.settings,
            original_text=record['input_snapshot']['sentence'],
            objective=choice['objective'], preserve=choice['preserve'])
        data = full_visible_delta_input(host)
        if (host != turn['full_comparison_host_input']
                or data != turn['full_comparison_model_input']
                or data['comparison_material'] != turn['comparison_material']
                or turn['full_comparison_system_prompt'] != FULL_VISIBLE_DELTA_SYSTEM):
            raise ValueError('原比较的输入、原图或合同改变，没有替换后重调。')
        return data, images

    async def _run_comparison_completion(self, record, entry):
        """Complete one existing tail without entering the creative driver."""
        started = None
        try:
            async with self._serial:
                started = self.clock()
                self.active_started[record['id']] = started
                record.update(status='running', stage='comparing', error=None,
                    detail='只补完已保存两图的比较；不重新构思、生图或选用版本。')
                self._check(record)
                if record['operation'] != entry['original_operation']:
                    raise ValueError('原创作操作身份改变，未借另一回合补比较。')
                turn = next(item for item in record['result_cycle']['turns'] if item['id'] == entry['turn_id'])
                data, images = self._comparison_completion_inputs(record, turn)
                if data != entry['full_comparison_model_input']:
                    raise ValueError('本次补比较的已保存输入改变。')
                recovered = self._returned_cycle_call(record, entry, 'comparison_call_index', ResultComparison)
                if recovered is None:
                    entry['comparison_call_index'] = len(record['calls'])
                    entry['status'] = 'calling'
                    self._save(record)
                    result = await self._call(record, entry['full_comparison_system_prompt'], data,
                        ResultComparison.model_json_schema(), images=images,
                        visual_input_scope='full_original_pair_visible_delta')
                    recovered = ResultComparison.model_validate(result.data).model_dump(mode='json')
                self._check(record)
                self._comparison_completion_inputs(record, turn)
                entry.update(status='completed', comparison=deepcopy(recovered))
                turn.update(comparison=deepcopy(recovered), phase='reviewed',
                    comparison_completion_receipt={'request_id': entry['request_id'],
                        'original_comparison_call_index': turn.get('comparison_call_index'),
                        'effective_comparison_call_index': entry['comparison_call_index'],
                        'origin': entry['origin'], 'scope': 'saved_pair_comparison_only'})
                cycle = record['result_cycle']
                # The neutral raw base is not a source choice or art acceptance.
                cycle.update(status='stopped', active_turn_id=None,
                    cursor_artwork_id=turn['child_artwork_id'], pending_reference_turn_id=turn['id'],
                    previous_comparison=deepcopy(recovered),
                    cached_observation=deepcopy(recovered['current_observation']),
                    cached_source_snapshot_sha256=turn['child_source']['snapshot_sha256'],
                    cached_observation_contract_version=FULL_VISIBLE_DELTA_CONTRACT,
                    cached_observation_artwork_id=turn['child_artwork_id'],
                    cached_observation_sha256=turn['child_source']['image']['sha256'],
                    stop_reason='既有两图比较已补完；本动作没有作者、目的选源或新图片派发。')
                record.update(status='ready', stage='ready', detail=cycle['stop_reason'])
        except asyncio.CancelledError:
            record.update(status='cancelled', stage='cancelled', detail='比较已停止；原图和原调用保留。')
            entry['status'] = 'cancelled'
        except InterruptedError as exc:
            record.update(status='paused', stage='paused', detail=str(exc))
            entry['status'] = 'paused'
        except Exception as exc:
            if self.events[record['id']].is_set():
                record.update(status='cancelled', stage='cancelled', error=None,
                    detail='比较已停止；原图和原调用保留。')
                entry['status'] = 'cancelled'
            else:
                record.update(status='failed', stage='comparing', error=str(exc),
                    detail='补比较停在实际失败处；原图、原失败与本次回执保留。')
                entry['status'] = 'failed'
        finally:
            if started is not None:
                record['active_seconds'] += self.clock() - started
            self.active_started.pop(record['id'], None)
            entry.setdefault('outcomes', []).append({
                'status': record['status'], 'error': record.get('error'),
                'used_total': self._used_budget(record), 'recorded_at': datetime.now(UTC).isoformat()})
            record['revision'] += 1
            self._save(record)

    async def _run(self, record, purpose, action=None):
        started = None
        try:
            async with self._serial:
                started = self.clock()
                self.active_started[record['id']] = started
                record.update(status='running', error=None)
                self._check(record)
                pending_dispatch = self._pending_generation_dispatch(record)
                if pending_dispatch and not pending_dispatch.get('job_id'):
                    raise ValueError('生成提交回执未确认，不重复构思或派发；请核对原任务')
                cycle = record.get('result_cycle') or {}
                resuming_cycle = (record['input_snapshot'].get('result_guided', False)
                    and cycle.get('operation_key') == self._request_digest(record['operation'])
                    and cycle.get('status') in {'running', 'paused'})
                direction_plan = (await self._ensure_direction_plan(record) if purpose == 'initial'
                    else record.get('direction_plan') if purpose == 'generate' else None)
                if direction_plan is not None and direction_plan.get('raw') is None:
                    direction_plan = None
                if direction_plan is not None and not resuming_cycle:
                    if await self._drive_directions(record, purpose, action):
                        return  # The current item, not the collection, awaits review.
                    remaining = self.get(record['id'])['remaining_budget']
                    if (direction_plan['status'] == 'completed_unverified'
                            and remaining['jobs'] > 0 and remaining['calls'] >= 2):
                        await self._drive_results(record)
                    record.update(status='ready', stage='ready')
                    return
                if (purpose != 'prepare_native' and record['input_snapshot'].get('result_guided', False)
                        and cycle.get('operation_key') == self._request_digest(record.get('operation') or {})
                        and cycle.get('status') in {'running', 'paused'}):
                    await self._drive_results(record)
                    record.update(status='ready', stage='ready')
                    return
                if purpose == 'initial':
                    if not record['plan']:
                        await self._prepare(record)
                    else:
                        await self._ensure_native(record)
                    if (not record['input_snapshot']['auto_generate']
                            and not record['active_job_id'] and not record['candidates']):
                        record.update(status='needs_confirmation', stage='review',
                            detail='方案及完整原生条件已准备，可审阅修改；未提交图片。')
                        return
                if purpose == 'prepare_native':
                    if action and action.request_id in (record.get('native_preparations') or {}):
                        await self._prepare_saved_prose(record, action)
                    else:
                        await self._ensure_native(record)
                    plan = record.get('direction_plan')
                    if plan is not None and plan.get('active_slot_id'):
                        slot = next(item for item in plan['slots'] if item['id'] == plan['active_slot_id'])
                        expected = make_direction_assignment(plan, slot)
                        if (record['generation_request'].get('author_proposal') or {}).get('direction_assignment') != expected:
                            raise ValueError('语言准备结果不属于当前独立方向。')
                        slot.update(status='prepared', generation_request=deepcopy(record['generation_request']),
                            recipe_snapshot=deepcopy(record['preparation_recipe_snapshot']),
                            draft={key: deepcopy(record.get(key)) for key in self._DIRECTION_DRAFT_FIELDS})
                        plan['status'] = 'needs_confirmation'
                    record.update(status='needs_confirmation', stage='review',
                        detail='原方案和转换后的完整原生条件均保存；请审阅，未自动派图。')
                    return
                if purpose == 'continue':
                    await self._continue(record, action)
                elif purpose == 'local_edit':
                    await self._local_edit(record, action)
                elif purpose == 'regenerate':
                    if not record.get('regeneration_prepared'):
                        feedback = {'text': action.feedback, 'purpose': 'new direction'}
                        if action.initiator == 'assistant_validation':
                            feedback.update(origin='assistant_validation_not_user_feedback',
                                scope='assistant_validation_regeneration', action_request_id=action.request_id)
                        await self._prepare(record, feedback=feedback)
                        record['regeneration_prepared'] = True
                        self._save(record)
                    if record.get('regeneration_job_id'):
                        job = self.jobs.get(record['regeneration_job_id'])
                        if job is None:
                            raise ValueError('重生成已提交但回执暂不可见，未重复派发')
                        await self._finish_job(record, job)
                    else:
                        await self._generate(record, 'text')
                else:
                    await self._ensure_native(record)
                    mode = record['input_snapshot']['support_mode']
                    method = ('research_pixels' if record['input_snapshot'].get('research_pixel_use') else 'text') if mode == 'auto' else mode
                    if mode == 'auto' and record['plan']['use_visual_support']:
                        record['support_limit'] = ('当前低细节色块草案尚无可用作品收益，真实首轮还保留了块状形体；'
                            '不依据模型的使用建议自动采用。草案只保留为显式方法对照，不代表文字路线已合格。')
                    actual = GenerateRequest.model_validate(record['generation_request'])
                    if record.get('pending_methods') is None:
                        record['pending_methods'] = ['text', 'sketch'] if mode == 'compare' else [method]
                        if mode == 'compare' and not check_spatial_support(SupportPlan.model_validate(record['plan']))['available']:
                            record['pending_methods'] = ['text']
                            record['support_limit'] = '空间草案不可用，本次只留下文本候选；不是完成了文字／草案对照。'
                        self._save(record)
                    while record['pending_methods']:
                        await self._generate(record, record['pending_methods'][0])
                keep_detail = (purpose == 'continue' and record.get('continuation')
                    and (record['continuation'].get('stop_reason')
                        or (record['continuation'].get('choice') or {}).get('operation') in {'keep', 'stop'}))
                await self._drive_results(record)
                keep_detail = keep_detail or bool((record.get('result_cycle') or {}).get('stop_reason'))
                record.update(status='ready', stage='ready')
                if not keep_detail:
                    record['detail'] = '候选已保存；请比较与评价。生成成功不代表意图和角色已通过。'
                if (record.get('direction_plan') or {}).get('status') == 'not_started_budget':
                    record['detail'] += '；' + record['direction_plan']['reason']
        except asyncio.CancelledError:
            if record.get('cancel_error'):
                record.update(status='paused', stage='cancellation_uncertain', error=record['cancel_error'],
                    detail='已停止本创作的新派发，但后端取消回执未确认；请核对实际任务，未宣称已停止采样。')
            else:
                record.update(status='cancelled', stage='cancelled', detail='创作已停止；已有原图和反馈保留。')
        except InterruptedError as exc:
            record.update(status='paused', stage='paused', detail=str(exc))
            if record.get('result_cycle'):
                record['result_cycle']['status'] = 'paused'
        except Exception as exc:
            if self.events[record['id']].is_set() and not record.get('cancel_error'):
                record.update(status='cancelled', stage='cancelled', error=None,
                              detail='创作已停止；已有原图和反馈保留。')
            else:
                record.update(status='failed', error=str(exc) or type(exc).__name__,
                    detail='停在实际失败处，原响应与已有作品保留；没有自动重试。')
        finally:
            if record.get('result_cycle') and record['status'] in {'cancelled', 'failed'}:
                record['result_cycle'].update(status='stopped', stop_reason=record['detail'])
            if started is not None:
                record['active_seconds'] += self.clock() - started
            self.active_started.pop(record['id'], None)
            if record.get('budget_rounds'):
                record['budget_rounds'][-1]['outcomes'].append({
                    'recorded_at': datetime.now(UTC).isoformat(),
                    'operation': deepcopy(record.get('operation')),
                    'status': record['status'], 'stage': record['stage'],
                    'error': record.get('error'), 'detail': record.get('detail'),
                    'used_total': self._used_budget(record)})
            preparation = ((record.get('native_preparations') or {}).get(action.request_id)
                if purpose == 'prepare_native' and action is not None else None)
            if preparation is not None:
                if record['status'] in {'failed', 'paused', 'cancelled'}:
                    preparation['status'] = record['status']
                preparation.setdefault('outcomes', []).append({
                    'status': record['status'], 'stage': record['stage'],
                    'error': record.get('error'), 'detail': record['detail'],
                    'used_total': self._used_budget(record)})
            record['revision'] += 1
            self._save(record)

    async def action(self, identifier, action: CreativeAction):
        record = self.records[identifier]
        if (action.initiator == 'assistant_validation' and action.action in {'select', 'revert', 'resume'}
                and record['input_snapshot'].get('initiator') != 'assistant_validation'):
            raise ValueError('助手版本／恢复核验只用于新建时明确署名的验证任务，不改用户或旧任务。')
        new_round_registered = False
        saved_prose_source = None
        snapshot = action.model_dump(mode='json')
        if action.comparison_turn_id is None:
            snapshot.pop('comparison_turn_id')  # Preserve pre-field action identities.
        old = record['action_receipts'].get(action.request_id)
        if old:
            if old['input'] != snapshot:
                raise ValueError('同一操作标识不能用于另一份操作')
            return self.get(identifier)
        if record['revision'] != action.expected_revision:
            raise ValueError('作品状态已变化，请读取当前版本；没有采用旧操作')
        if action.artwork_id and not any(c['artwork_id'] == action.artwork_id for c in record['candidates']):
            raise ValueError('只能操作本创作中的真实候选')
        busy = bool(self.tasks.get(identifier) and not self.tasks[identifier].done())
        if busy and action.action not in {'pause', 'stop'}:
            raise ValueError('当前仍有在途操作，没有重复派发')
        if not busy and record['status'] in {'paused', 'failed', 'cancelled', 'ready'}:
            self._commit_terminal_cancellation(record, origin='explicit_action_preflight')
        if action.region_edit is not None and action.action != 'local_edit':
            raise ValueError('圈选数据只用于明确的局部修改')
        if action.native_override is not None and action.action != 'generate':
            raise ValueError('原生覆盖只用于审阅后的明确生成')
        if action.action in {'generate', 'continue', 'regenerate', 'local_edit'} and record.get('active_job_id'):
            raise ValueError('上次派发尚未完成回执接续，请先恢复并核对原任务，不以新操作绕过')
        pending_dispatch = self._pending_generation_dispatch(record)
        if (pending_dispatch and not pending_dispatch.get('job_id')
                and action.action in {'generate', 'continue', 'regenerate', 'local_edit', 'prepare_native'}):
            raise ValueError('上次生成提交回执未确认，不以新操作绕过或重复派发')
        paused_completion = (record.get('comparison_completions') or {}).get(
            record.get('last_comparison_completion_request_id'))
        resume_completion = (action.action == 'resume' and record['status'] == 'paused'
            and paused_completion is not None and paused_completion.get('status') == 'paused')
        if action.action == 'complete_comparison' or resume_completion:
            capability = self._comparison_completion_capability(record)
            identifier_turn = (paused_completion['turn_id'] if resume_completion else action.comparison_turn_id)
            if not capability['available'] or capability.get('turn_id') != identifier_turn:
                raise ValueError('当前不能补完此比较：' + capability.get('reason', '回合不一致'))
            turn = next(item for item in record['result_cycle']['turns'] if item['id'] == identifier_turn)
            self._comparison_completion_inputs(record, turn)
            rollback = deepcopy(record)
            if resume_completion:
                entry = paused_completion
            else:
                entry = {key: deepcopy(turn[key]) for key in (
                    'full_comparison_host_input', 'full_comparison_model_input',
                    'full_comparison_system_prompt', 'comparison_material')}
                entry.update(request_id=action.request_id, turn_id=identifier_turn,
                    origin=action.initiator, status='accepted',
                    scope='saved_pair_comparison_only_no_author_or_generation',
                    original_operation=deepcopy(record['operation']),
                    original_comparison_call_index=turn.get('comparison_call_index'),
                    comparison_call_index=(capability['source_call_index']
                        if capability['mode'] == 'reuse_return' else None),
                    prior_outcome={field: deepcopy(record.get(field))
                        for field in ('status', 'stage', 'error', 'detail', 'revision')})
                record.setdefault('comparison_completions', {})[action.request_id] = entry
                record['last_comparison_completion_request_id'] = action.request_id
            entry.setdefault('actions', []).append(snapshot)
            record.update(status='queued', stage='comparing', error=None, pause_requested=False,
                detail='等待补完本次已保存比较；不重新构思或生图。')
            record['action_receipts'][action.request_id] = {'input': snapshot, 'status': 'accepted'}
            record['last_action_request_id'] = action.request_id
            record['revision'] += 1
            try:
                self._save(record)
                self.tasks[identifier] = asyncio.create_task(self._run_comparison_completion(record, entry),
                    name='creative-comparison-' + identifier)
            except BaseException:
                record.clear()
                record.update(rollback)
                raise
            return self.get(identifier)
        new_user_round = (action.initiator == 'user'
                          and action.action in {'continue', 'regenerate', 'local_edit'})
        if new_user_round:
            self._require_new_round_boundary(record)
            if action.action == 'continue':
                source_id = action.artwork_id or record.get('selected_artwork_id')
                source = next((item for item in record['candidates']
                               if item['artwork_id'] == source_id), None)
                job = self.jobs.get(source['job_id']) if source else None
                if (not source or job is None or job.status != 'completed' or not job.images
                        or self.artworks.get(source_id) is None):
                    raise ValueError('新接续需要本创作已完成的真实源作品，没有猜测或开始回合')
                original = record['input_snapshot']['sentence']
                freeze_continuation_source(source_id, self.artworks, self.jobs,
                    original_text=original, original_sha256=hashlib.sha256(original.encode('utf-8')).hexdigest(),
                    output_root=Path(self.settings.output_dir), session_feedback=record['feedback'])
        edit_checkpoint = None
        if action.action == 'local_edit':
            if not action.artwork_id or action.region_edit is None:
                raise ValueError('局部修改需要本创作的原图、圈选和自然语言要求')
            if action.feedback and action.feedback != action.region_edit.instruction:
                raise ValueError('局部修改评价与圈选要求不一致，没有猜测采用哪份文本')
            previous_edit = record.get('local_edit') or {}
            if (previous_edit.get('submission_uncertain') or
                    (previous_edit.get('submission_started') and not previous_edit.get('job_id'))):
                raise ValueError('上次局部修改回执未确认，请先核对原任务；不重复提交')
            if previous_edit.get('job_id'):
                previous_job = self.jobs.get(previous_edit['job_id'])
                if previous_job is None or previous_job.status not in {'completed', 'failed', 'cancelled'}:
                    raise ValueError('上次局部修改原回执尚不可核对，不以新操作覆盖或重复派发')
            artwork, parent, context = self._edit_context(record, action.artwork_id)
            checked, _, _ = prepare_region_edit(artwork, parent, action.region_edit.model_dump(mode='json'),
                self.settings, context_request=context, materialize=False)
            edit_checkpoint = {'artwork_id': artwork.id, 'source_sha256': checked.repair_source_sha256,
                'context_request': context.model_dump(mode='json'), 'job_id': None, 'completed': False}
        if action.max_calls is not None:
            if action.action != 'prepare_native' or action.max_calls < len(record['calls']):
                raise ValueError('调用预算调整只用于明确准备原生条件，且不能少于已用预算')
        if action.action == 'prepare_native' and (record['status'] != 'needs_confirmation' or not record['plan']):
            capability = self._saved_prose_capability(record)
            if not capability['available']:
                raise ValueError('完整原稿暂不能准备原生条件：' + capability['reason'])
            limit = action.max_calls if action.max_calls is not None else record['limits']['calls']
            if limit < capability['minimum_max_calls']:
                raise ValueError('新语言调用需要剩余额度或明确声明足够的调用上限；不重作者或借用额度。')
            saved_prose_source = self._saved_prose_source(record)
        stop_resolution_checkpoint = None
        if action.action in {'select', 'revert'}:
            if not action.artwork_id or self.artworks.get(action.artwork_id) is None:
                raise ValueError('请选择已有原图')
            record['selected_artwork_id'] = action.artwork_id
            record['selection_origin'] = action.initiator
        elif action.action == 'pause':
            record['pause_requested'] = True
            record.update(detail='暂停后不再派发下一项；当前已派任务完成后保留结果。')
            if not busy:
                record.update(status='paused', stage='paused')
        elif action.action == 'stop':
            self.events[identifier].set()
            if pending_dispatch and not pending_dispatch.get('job_id'):
                record.update(status='paused', stage='cancellation_uncertain',
                    detail='已停止新派发；生成提交回执未确认，未宣称后端采样已停止。')
                if not busy:
                    record['cancel_error'] = '生成提交回执未确认；请核对原队列，未能确认后端取消。'
            if record['active_job_id']:
                try:
                    stopped_job = await self.jobs.cancel(record['active_job_id'])
                    engine_receipts = getattr(stopped_job, 'engine_dispatches', [])
                    confirmed = (stopped_job is not None
                        and getattr(stopped_job, 'cancel_error', None) is None
                        and ((getattr(stopped_job, 'status', None) == 'cancelled'
                              and getattr(stopped_job, 'cancel_requested', False) is True
                              and all(item.get('completed') is True for item in engine_receipts))
                             or (getattr(stopped_job, 'status', None) in {'failed', 'completed'}
                                 and engine_receipts
                                 and all(item.get('completed') is True for item in engine_receipts))))
                    if confirmed and record.get('cancel_error'):
                        self._generation_dispatch_for_job(record, stopped_job)
                        previous_error = record['cancel_error']
                        record.setdefault('cancel_resolutions', []).append({
                            'job_id': stopped_job.id, 'previous_cancel_error': previous_error,
                            'resolved_by': 'confirmed_job_cancellation',
                            'engine_dispatches': deepcopy(engine_receipts)})
                        record['cancel_error'] = None
                        if record.get('error') == previous_error:
                            record['error'] = None
                except Exception as exc:
                    record['cancel_error'] = str(exc)
            elif (not busy and record.get('active_job_id') is None
                    and record.get('cancel_error') and pending_dispatch is None):
                # The finished original may have legitimately cleared its
                # active pointer before cancellation's display error arrived.
                # Use one complete reservation, not a candidate/note/last-job
                # guess; this does not restore an active task or call Core.
                try:
                    dispatches = record.get('generation_dispatches')
                    if not isinstance(dispatches, list) or len(dispatches) != 1:
                        raise ValueError('没有唯一原生成派发记录，未猜测要核对的任务')
                    dispatch = dispatches[0]
                    if (not isinstance(dispatch, dict) or dispatch.get('completed') is not True
                            or not isinstance(dispatch.get('id'), str) or not dispatch['id']
                            or not isinstance(dispatch.get('method'), str) or not dispatch['method']
                            or dispatch.get('status') != 'cancelled'
                            or dispatch.get('submission_started') is not True
                            or dispatch.get('submission_uncertain') is not False
                            or not isinstance(dispatch.get('job_id'), str) or not dispatch['job_id']
                            or not isinstance(dispatch.get('request'), dict)
                            or not all(isinstance(dispatch.get(key), str)
                                and re.fullmatch(r'[0-9a-f]{64}', dispatch[key])
                                for key in ('request_sha256', 'job_request_sha256'))):
                        raise ValueError('原生成回执不完整或尚未确认，未清除旧错误')
                    original_job = self.jobs.get(dispatch['job_id'])
                    if (original_job is None or original_job.cancel_error is not None
                            or self._generation_dispatch_for_job(record, original_job) is not dispatch):
                        raise ValueError('原任务或完整双请求绑定未确认，未清除旧错误')
                    evidence = self.jobs.closed_cancellation_evidence(original_job.id)
                    if (evidence['job_id'] != dispatch['job_id']
                            or evidence['request_sha256'] != dispatch['job_request_sha256']):
                        raise ValueError('原引擎持久证据与生成回执不符，未清除旧错误')
                except Exception:
                    pass  # Keep the original error; missing proof is not resolution.
                else:
                    stop_resolution_checkpoint = deepcopy(record)
                    previous_error = record['cancel_error']
                    record.setdefault('cancel_resolutions', []).append({
                        'job_id': original_job.id, 'generation_dispatch_id': dispatch['id'],
                        'previous_cancel_error': previous_error,
                        'resolved_by': 'verified_closed_original_generation',
                        'engine_receipt_sha256': evidence['record_sha256'],
                        'engine_dispatches': evidence['engine_dispatches']})
                    record['cancel_error'] = None
                    if record.get('error') == previous_error:
                        record['error'] = None
            if record.get('cancel_error'):
                record.update(status='paused', stage='cancellation_uncertain', error=record['cancel_error'],
                    detail='已停止新派发，但后端取消尚未确认；没有宣称采样已停止。')
            elif not (pending_dispatch and not pending_dispatch.get('job_id')):
                record.update(status='cancelled', stage='cancelled', detail='停止后不再派发，已有作品保留。')
        else:
            if record['status'] == 'cancelled':
                raise ValueError('已停止的创作不会隐式恢复，请明确创建新的构想')
            if record.get('cancel_error'):
                raise ValueError('当前取消回执未确认，不安排新派发')
            if action.action == 'resume' and record['status'] not in {'paused'}:
                raise ValueError('只有暂停的创作可以继续；失败请明确选择重新生成或修正')
            if action.action == 'generate' and record['status'] != 'needs_confirmation':
                raise ValueError('该生成已处理；新方向请明确重新生成，不重派原提交')
            round_checkpoint = deepcopy(record) if new_user_round or saved_prose_source is not None else None
            if new_user_round:
                self._begin_user_budget_round(record, action)
            if saved_prose_source is not None:
                record.setdefault('native_preparations', {})[action.request_id] = {
                    'origin': action.initiator, 'status': 'accepted',
                    'source_operation': deepcopy(saved_prose_source['operation']),
                    'source_checkpoint': deepcopy(saved_prose_source['checkpoint']),
                    'source_context_sha256': self._request_digest(saved_prose_source['context']),
                    'prior_outcome': {field: deepcopy(record.get(field))
                        for field in ('status', 'stage', 'detail', 'error', 'revision', 'operation')},
                    'calls_before': len(record['calls']), 'language_call_index': None,
                    'translated': None,
                    'reuse_language_return': saved_prose_source['language_checkpoint'].get('translated') is not None,
                    'source_refs': {'raw_choice': ('/canonical_picture_choice'
                        if saved_prose_source['checkpoint'].get('kind') == 'saved_canonical_language_return_v1'
                        else '/prose_preparation/raw'),
                        'authoring_input': '/picture_authoring_context',
                        'author_return': f"/calls/{saved_prose_source['checkpoint']['author_call_index']}/data"}}
            if action.max_calls is not None:
                change = {'request_id': action.request_id,
                    'previous_calls': record['limits']['calls'], 'new_calls': action.max_calls}
                if action.initiator == 'assistant_validation' or saved_prose_source is not None:
                    change.update(origin=action.initiator, scope='explicit_prepare_native_call_limit')
                record.setdefault('budget_changes', []).append(change)
                record['limits']['calls'] = action.max_calls
            if action.native_override is not None:
                if action.action != 'generate' or not record['generation_request']:
                    raise ValueError('原生覆盖只用于审阅后的明确生成')
                override = action.native_override
                if set(override) != {'positive', 'negative'} or not override['positive'].strip():
                    raise ValueError('原生覆盖需要明确的正向和负面两字段')
                actual = GenerateRequest.model_validate(record['generation_request'])
                channel = {'anima_turbo': 'anima', 'z_image_turbo': 'z_image'}.get(actual.model_id.value, actual.model_id.value)
                native = deepcopy(actual.native_prompts)
                native[channel]['compiled_trace'] = deepcopy(native[channel].get('trace', []))
                native[channel]['compiled_style_recipe'] = native[channel].pop('style_recipe_applied', None)
                native[channel].update(override, trace=[], trace_status='explicit_user_override_unverified')
                proposal = deepcopy(actual.author_proposal or {})
                proposal.update(condition_status='historical_after_native_override',
                    historical_semantic_spec=deepcopy(actual.semantic_spec),
                    historical_scene_contract=deepcopy(actual.scene_contract))
                for key, value in native.items():
                    if key != channel:
                        value['trace_status'] = 'historical_projection_after_native_override'
                actual = actual.model_copy(update={'prompt': override['positive'], 'negative_prompt': override['negative'],
                    'use_default_negative_prompt': False, 'native_prompts': native, 'style_preset': '',
                    'author_proposal': proposal, 'semantic_spec': None, 'scene_contract': None})
                record.update(generation_request=actual.model_dump(mode='json'), native_prompts=native)
                record['native_user_override'] = True
            if action.action in {'generate', 'regenerate'}:
                record['pending_methods'] = None
            if action.action == 'continue':
                record['continuation'] = None
                # Keep the accepted source separate from the original request
                # identity. Later selection may change the next operation, but
                # not this pending continuation or its feedback after resume.
                record['continuation_source_acceptance'] = {
                    'action_request_id': action.request_id,
                    'artwork_id': action.artwork_id or record.get('selected_artwork_id')}
            if action.action == 'regenerate':
                record['regeneration_prepared'] = False
                record['regeneration_job_id'] = None
                if action.initiator == 'assistant_validation' and action.feedback:
                    record['feedback'].append({'artwork_id': action.artwork_id, 'text': action.feedback,
                        'origin': 'assistant_validation_not_user_feedback',
                        'scope': 'assistant_validation_regeneration', 'action_request_id': action.request_id,
                        'created_at': datetime.now(UTC).isoformat()})
            if action.action == 'local_edit':
                record['local_edit'] = edit_checkpoint
                record['feedback'].append({'artwork_id': action.artwork_id,
                    'text': action.region_edit.instruction,
                    'scope': ('assistant_validation_mask_local_edit' if action.initiator == 'assistant_validation'
                              else 'user_mask_local_edit'),
                    **({'origin': 'assistant_validation_not_user_feedback'}
                       if action.initiator == 'assistant_validation' else {}),
                    'action_request_id': action.request_id,
                    'created_at': datetime.now(UTC).isoformat()})
            record['pause_requested'] = False
            record.update(status='queued', stage='queued')
            purpose = 'continue' if action.action == 'continue' else 'regenerate' if action.action == 'regenerate' else 'generate'
            if action.action == 'prepare_native':
                purpose = 'prepare_native'
            elif action.action == 'local_edit':
                purpose = 'local_edit'
            if action.action == 'resume' and record.get('operation'):
                operation = record['operation']
                purpose = operation['purpose']
                action_for_run = CreativeAction.model_validate(operation['action']) if operation.get('action') else None
            else:
                action_for_run = action
            if not record['plan'] and purpose not in {'continue', 'regenerate', 'local_edit', 'prepare_native'}:
                purpose = 'initial'
            if new_user_round:
                # Persist acceptance in the same first durable state as the
                # new allowance, BEFORE a worker can start. A crash cannot
                # lose this click's identity and open another budget round.
                record['action_receipts'][action.request_id] = {'input': snapshot, 'status': 'accepted'}
                record['last_action_request_id'] = action.request_id
                record['revision'] += 1
            try:
                self._schedule(record, purpose, action_for_run)
            except BaseException:
                if round_checkpoint is not None:
                    record.clear()
                    record.update(round_checkpoint)
                raise
            new_round_registered = new_user_round
        if new_round_registered:
            return self.get(identifier)  # _schedule saved receipt, revision and round together.
        record['action_receipts'][action.request_id] = {'input': snapshot, 'status': 'accepted'}
        record['last_action_request_id'] = action.request_id
        record['revision'] += 1
        try:
            self._save(record)
        except BaseException:
            if stop_resolution_checkpoint is not None:
                record.clear()
                record.update(stop_resolution_checkpoint)
            raise
        return self.get(identifier)

    async def stop(self):
        for identifier, task in self.tasks.items():
            if not task.done():
                self.events[identifier].set()
                active = self.records[identifier].get('active_job_id')
                if active:
                    try:
                        await self.jobs.cancel(active)
                    except Exception as exc:
                        self.records[identifier]['cancel_error'] = str(exc)
                        self._save(self.records[identifier])
                task.cancel()
        await asyncio.gather(*self.tasks.values(), return_exceptions=True)
