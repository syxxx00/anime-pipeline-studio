from __future__ import annotations

import asyncio
from copy import deepcopy
import hashlib
import json
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import httpx

from .checks import CheckContext, CheckEntry, build_report, load_style_rules, resolve_style_category, rules_for, run_detection, text_tag_hints
from .generation_request import execution_parameters, prepare_generation_request
from .comfy_client import ComfyClient, EngineSubmissionRejected, OwnedCancellationUnconfirmed, extract_output_images
from .config import PROJECT_ROOT, Settings
from .job_engine_receipts import EngineReceiptStore, EngineReceiptPersistenceError, validate_exact_inspection, validate_terminal_history
from .schemas import GenerateRequest, PIXEL_BRIDGE_PIPELINES, RenderPipeline
from .pipeline_styles import prepare_request_styles
from .stage_images import StageImages
from .region_edit import protected_inputs, verify_region_output
from .image_handoff import cleanup_handoff_input, verify_handoff_input, verify_handoff_output
from .iteration_evidence import compare_images
from .local_authoring_runtime import INFERENCE_LEASE
from .text_detect import detect_text
from .workflow import build_anima_base_workflow, build_hand_repair_workflow, build_newbie_refine_workflow, build_text_clean_workflow, build_workflow


@dataclass(slots=True)
class Job:
    id: str
    status: str
    created_at: str
    request: dict
    progress: float = 0
    stage: str = "queued"
    images: list[str] = field(default_factory=list)
    files: list[str] = field(default_factory=list)
    timings: dict[str, float] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)
    checks: list[dict] = field(default_factory=list)
    error: str | None = None
    engine_dispatches: list[dict] = field(default_factory=list)
    cancel_requested: bool = False
    cancel_error: str | None = None


class JobManager:
    def __init__(self, settings: Settings, client: ComfyClient) -> None:
        self.settings = settings
        self.client = client
        self.jobs: dict[str, Job] = {}
        self.queue: asyncio.Queue[tuple[str, GenerateRequest, dict | None]] = asyncio.Queue()
        self.worker: asyncio.Task | None = None
        self.active_job_id: str | None = None
        # Exact custom graphs submitted in this process, distinct from both a
        # schema-based preview and a backend execution/persistence receipt.
        self._submitted_workflows: dict[str, dict] = {}
        self._cancel_locks: dict[str, asyncio.Lock] = {}
        self._owned_prompt_jobs: dict[str, str] = {}
        # Only live, checked queue calls can prove that POST has not started.
        # Persisted fields alone never confer this proof after an API restart.
        self._pre_dispatch_receipts: dict[int, dict] = {}
        self.engine_receipts = EngineReceiptStore(PROJECT_ROOT / 'data' / 'engine_receipts')
        self._receipt_workflows: dict[str, list[dict]] = {}
        self._restored_engine_jobs: set[str] = set()
        self._receipt_load_errors: list[str] = []
        self.output_root = Path(settings.output_dir)
        self.output_root.mkdir(parents=True, exist_ok=True)
        self.stage_images = StageImages(self.output_root, Path(settings.engine_base_dir))
        self._load_history()
        self._load_engine_receipts()

    def start(self) -> None:
        if self.worker is None or self.worker.done():
            self.worker = asyncio.create_task(self._run(), name="newbie-generation-worker")

    async def stop(self) -> None:
        if self.worker:
            active = self.jobs.get(self.active_job_id or '')
            if active is not None and self._open_engine_dispatches(active):
                try:
                    await self.cancel(active.id)
                except Exception as exc:
                    active.cancel_error = str(exc) or type(exc).__name__
                    active.stage = 'cancellation_uncertain'
            self.worker.cancel()
            try:
                await self.worker
            except asyncio.CancelledError:
                pass

    async def submit(self, request: GenerateRequest) -> Job:
        self._reject_restored_receipt_fence()
        if request.image_handoff:
            raise ValueError('模型交接必须使用作品 handoff 入口，不能作为普通文生图提交')
        automatic = request.render_pipeline == RenderPipeline.ADAPTIVE
        request, decision, adjustments = prepare_generation_request(request, self.settings)
        job_id = uuid.uuid4().hex[:12]
        job = Job(
            id=job_id,
            status="queued",
            created_at=datetime.now(UTC).isoformat(),
            request=request.model_dump(mode="json"),
        )
        if automatic:
            job.notes.append("自适应路由：" + decision.reason)
            job.request["adaptive_route_reason"] = decision.reason
        if adjustments:
            job.request["parameter_adjustments"] = adjustments
            job.notes.append("旧实验参数策略：已在排队前记录 Anima 历史固定参数；新创作使用显式参数。")
        self.jobs[job_id] = job
        await self.queue.put((job_id, request, None))
        return job

    async def submit_workflow(self, request: GenerateRequest, workflow: dict) -> Job:
        self._reject_restored_receipt_fence()
        request = prepare_request_styles(request, self.settings)
        if not isinstance(workflow, dict):
            raise ValueError('自定义工作流必须是完整JSON对象，不重建或猜测输入。')
        frozen_workflow = deepcopy(workflow)
        workflow_sha = self._workflow_digest(frozen_workflow)
        job_id = uuid.uuid4().hex[:12]
        job = Job(id=job_id, status="queued", created_at=datetime.now(UTC).isoformat(), request=request.model_dump(mode="json"))
        job.notes.extend(request.repair_notes)
        self.jobs[job_id] = job
        snapshot = {'workflow': frozen_workflow, 'workflow_sha256': workflow_sha,
                    'submission_state': 'reserved_before_local_queue',
                    'request_sha256_at_submission': self._workflow_digest(job.request)}
        self._submitted_workflows[job_id] = snapshot
        try:
            # Neither the caller nor a returned evidence dict can mutate the
            # worker graph or the separate exact submitted snapshot.
            await self.queue.put((job_id, request.model_copy(deep=True), deepcopy(frozen_workflow)))
        except BaseException:
            snapshot['submission_state'] = 'local_queue_receipt_unknown'
            raise
        snapshot['submission_state'] = 'local_queue_accepted'
        return job

    @staticmethod
    def _workflow_digest(workflow: dict) -> str:
        return hashlib.sha256(json.dumps(workflow, ensure_ascii=False, sort_keys=True,
            allow_nan=False, separators=(',', ':')).encode('utf-8')).hexdigest()

    def get(self, job_id: str) -> Job | None:
        return self.jobs.get(job_id)

    def list(self) -> list[Job]:
        return sorted(self.jobs.values(), key=lambda job: job.created_at, reverse=True)

    def preview(self, request: GenerateRequest) -> dict:
        request, decision, adjustments = prepare_generation_request(request, self.settings)
        if request.render_pipeline in PIXEL_BRIDGE_PIPELINES:
            source_name, source_request = self._bridge_source(request)
            source_workflow = (
                build_anima_base_workflow(source_request, self.settings, "newbie_lab/preview_source")
                if request.render_pipeline == RenderPipeline.HYBRID
                else build_workflow(source_request, self.settings, "newbie_lab/preview_source")
            )
            workflow = {
                "pipeline": request.render_pipeline.value,
                "bridge": f"{source_name} image -> unload models -> NewBie VAE encode",
                "stages": {
                    "1_source_specialist": source_workflow,
                    "2_newbie_identity_and_face": build_newbie_refine_workflow(request, self.settings, "adaptive_preview_source.png", "newbie_lab/preview"),
                },
            }
        else:
            # The hybrid Anima source builder has separate text/parameters;
            # it is not the single-model preview or execution path.
            workflow = build_workflow(request, self.settings, "newbie_lab/preview")
        return {"request": request.model_dump(mode="json"), "route_reason": decision.reason,
                "parameter_adjustments": adjustments, "workflow": workflow}

    async def cancel(self, job_id: str) -> Job | None:
        job = self.jobs.get(job_id)
        if job is None:
            return job
        async with self._cancel_locks.setdefault(job_id, asyncio.Lock()):
            pending = self._open_engine_dispatches(job)
            if job.status in {'completed', 'failed', 'cancelled'} and not pending:
                if job.status == 'cancelled' and job.cancel_error:
                    self._clear_saved_terminal_cancel_error(job)
                return job
            job.cancel_requested = True
            try:
                if not job.engine_dispatches and job.stage not in {
                        'queued', 'waiting_for_local_resource', 'starting_engine',
                        'waiting_for_unconfirmed_engine_receipt'}:
                    raise OwnedCancellationUnconfirmed('旧在途任务没有已绑定引擎回执，未猜测或全局中断。')
                await self._cancel_open_engine_dispatches_locked(job)
            except Exception as exc:
                job.cancel_error = str(exc) or type(exc).__name__
                job.stage = 'cancellation_uncertain'
                raise
            job.status = job.stage = 'cancelled'
            job.error = job.cancel_error = None
            return job

    @staticmethod
    def _open_engine_dispatches(job: Job) -> list[dict]:
        return [item for item in job.engine_dispatches if not item.get('completed')]

    def _other_open_engine_jobs(self, job: Job) -> list[str]:
        return [other.id for other in self.jobs.values()
                if other.id != job.id and self._open_engine_dispatches(other)] + self._receipt_load_errors

    def _reject_restored_receipt_fence(self) -> None:
        pending = [job_id for job_id in self._restored_engine_jobs
                   if self._open_engine_dispatches(self.jobs[job_id])]
        if pending or self._receipt_load_errors:
            raise OwnedCancellationUnconfirmed('重启前引擎回执仍未确认，未提交新任务或重排旧任务：'
                + '; '.join(pending + self._receipt_load_errors))

    def _persist_engine_receipts(self, job: Job) -> None:
        self.engine_receipts.write(job_id=job.id, created_at=job.created_at,
            backend_url=self.settings.comfy_url.rstrip('/'), request=job.request,
            dispatches=job.engine_dispatches, workflows=self._receipt_workflows.get(job.id, []),
            cancel_requested=job.cancel_requested)

    def _receipt_transition(self, job: Job, dispatch: dict, **changes) -> None:
        previous = deepcopy(dispatch)
        if previous.get('completed') and previous | changes != previous:
            raise EngineReceiptPersistenceError('已持久终止的阶段不可改写或复活。')
        dispatch.update(changes)
        try:
            self._persist_engine_receipts(job)
        except EngineReceiptPersistenceError as exc:
            dispatch.clear()
            dispatch.update(previous)
            # Keep actual late ACK/proofs in memory even when saving fails;
            # never erase their identity or claim a durable terminal result.
            for key in ('prompt_id', 'rejection_response', 'terminal_history', 'cancellation_receipt', 'inspection_receipts'):
                if key in changes:
                    dispatch[key] = deepcopy(changes[key])
            dispatch.update(completed=False, status='receipt_persistence_uncertain',
                            persistence_error=str(exc))
            raise

    def _strict_engine_receipt_binding(self, job: Job, dispatch: dict, *, allow_completed: bool = False) -> tuple[dict, dict]:
        if self.active_job_id == job.id and self.worker is not None and not self.worker.done():
            raise OwnedCancellationUnconfirmed('当前 worker 正在处理原任务，未干涉正常读取或交付。')
        try:
            if self.jobs.get(job.id) is not job:
                raise EngineReceiptPersistenceError('原本地任务对象已变化。')
            record = self.engine_receipts.read(job.id)
            if (record['request_sha256'] != self._workflow_digest(job.request)
                    or record['created_at'] != job.created_at
                    or record['backend_url'] != self.settings.comfy_url.rstrip('/')):
                raise EngineReceiptPersistenceError('原请求／创建记录／后端 URL 与当前任务不符。')
            originals = [{key: value for key, value in item.items() if key != 'workflow'}
                         for item in record['dispatches']]
            if originals != job.engine_dispatches:
                raise EngineReceiptPersistenceError('当前 dispatcher 原件与持久回执不一致。')
            graphs = self._receipt_workflows.get(job.id)
            if graphs != [item['workflow'] for item in record['dispatches']]:
                raise EngineReceiptPersistenceError('当前实际 graph 缓存与持久原件不一致，未重建。')
            if (not any(item is dispatch for item in job.engine_dispatches)
                    or dispatch.get('completed') is not allow_completed):
                raise EngineReceiptPersistenceError('原未完成阶段对象已变化或已结束。')
            prompt_id, owner = dispatch.get('prompt_id'), dispatch.get('client_id')
            instance = dispatch.get('backend_instance_id')
            if (not isinstance(prompt_id, str) or not prompt_id
                    or not isinstance(owner, str) or not owner
                    or not isinstance(instance, str) or str(uuid.UUID(instance)) != instance
                    or dispatch.get('owner_request_sha256') != record['request_sha256']):
                raise EngineReceiptPersistenceError('原 prompt／owner／后端 instance 绑定不完整，未猜补历史。')
            binding = {'owner_client_id': owner, 'workflow_sha256': dispatch['workflow_sha256'],
                       'backend_instance_id': instance}
            return binding, {'record_sha256': record['record_sha256'],
                             'request_sha256': record['request_sha256'],
                             'dispatches_sha256': self._workflow_digest(job.engine_dispatches)}
        except (EngineReceiptPersistenceError, ValueError, TypeError) as exc:
            raise OwnedCancellationUnconfirmed('原任务严格持久回执未确认，未核对／停止其他任务：' + str(exc)) from exc

    def _recheck_engine_receipt_binding(self, job: Job, dispatch: dict, frozen: dict) -> None:
        _, current = self._strict_engine_receipt_binding(job, dispatch)
        if current != frozen:
            raise OwnedCancellationUnconfirmed('等待后原请求／dispatcher／持久原件变化，未静默采纳返回回执。')

    def closed_cancellation_evidence(self, job_id: str) -> dict:
        """Read-only verification of the original closed, cancelled job.

        This neither sends an engine command nor changes its journal/display.
        A session may use it after its active pointer was already cleared;
        a completed bit or the current backend's idle state is not evidence.
        """
        job = self.jobs.get(job_id)
        if (job is None or job.status != 'cancelled' or not job.cancel_requested
                or not job.engine_dispatches or self._open_engine_dispatches(job)):
            raise OwnedCancellationUnconfirmed('没有原任务持久终止依据，未清除旧取消错误。')
        frozen = None
        for dispatch in job.engine_dispatches:
            _, frozen = self._strict_engine_receipt_binding(job, dispatch, allow_completed=True)
            if dispatch['status'] == 'history_returned':
                validate_terminal_history(dispatch, dispatch['terminal_history'])
            elif dispatch['status'] == 'cancelled_or_already_terminal':
                proof = dispatch['cancellation_receipt']
                if (proof.get('ownership_verified') is not True
                        or proof.get('prompt_id') != dispatch['prompt_id']
                        or proof.get('client_id') != dispatch['client_id']
                        or proof.get('absent_from_running_and_pending') is not True
                        or any(key in proof and proof[key] != dispatch[original]
                               for key, original in (('workflow_sha256', 'workflow_sha256'),
                                                     ('instance_id', 'backend_instance_id')))):
                    raise OwnedCancellationUnconfirmed('原取消终止依据未匹配完整绑定，未清除旧错误。')
                # Older same-owner cancellation consumers omitted graph/instance
                # from the verified atomic ACK. The strict journal check above
                # still binds the original request, graph, instance and owner.
                # Match that original owner, not a reloaded client's new ID;
                # absent old proof fields stay absent, never backfilled.
            else:
                raise OwnedCancellationUnconfirmed('当前阶段不具备这条原引擎终止核对依据。')
        return {'job_id': job.id, 'request_sha256': frozen['request_sha256'],
                'record_sha256': frozen['record_sha256'],
                'engine_dispatches': deepcopy(job.engine_dispatches)}

    def _clear_saved_terminal_cancel_error(self, job: Job) -> None:
        # Explicit stop/check may resolve an old display error, but neither
        # an empty engine queue nor a memory-only completed bit proves this.
        self.closed_cancellation_evidence(job.id)
        old_stage, old_error, old_notes = job.stage, job.cancel_error, job.notes[:]
        job.notes.append('原取消错误已由持久终止原件核实收束：' + old_error)
        job.stage, job.cancel_error = 'cancelled', None
        try:
            self._append_history(job)
        except OSError as exc:
            job.stage, job.cancel_error, job.notes = old_stage, old_error, old_notes
            raise EngineReceiptPersistenceError('原终态显示纠正未确认保存，旧错误保留。') from exc

    async def inspect_engine_receipts(self, job_id: str) -> dict | None:
        job = self.jobs.get(job_id)
        if job is None:
            return None
        async with self._cancel_locks.setdefault(job_id, asyncio.Lock()):
            pending = self._open_engine_dispatches(job)
            if not pending:
                if job.status == 'cancelled' and job.cancel_error:
                    self._clear_saved_terminal_cancel_error(job)
                return {'job_id': job.id, 'status': job.status, 'stage': job.stage, 'resolved': True,
                        'dispatches': [], 'message': '本地没有未确认引擎阶段；未自动接图或恢复旧研究。'}
            results = []
            for dispatch in pending:
                binding, frozen = self._strict_engine_receipt_binding(job, dispatch)
                inspect = getattr(self.client, 'inspect_owned', None)
                if inspect is None:
                    raise OwnedCancellationUnconfirmed('引擎客户端没有已核实的原任务原子核对契约。')
                proof = await inspect(dispatch['prompt_id'], **binding)
                self._recheck_engine_receipt_binding(job, dispatch, frozen)
                try:
                    validate_exact_inspection(dispatch, proof)
                except (EngineReceiptPersistenceError, ValueError, TypeError) as exc:
                    raise OwnedCancellationUnconfirmed('原任务检查未证明所述状态：' + str(exc)) from exc
                changes = {'inspection_receipts': deepcopy(dispatch.get('inspection_receipts', [])) + [deepcopy(proof)]}
                if proof['state'] == 'terminal':
                    changes.update(completed=True, status='history_returned', terminal_history=deepcopy(proof['history']))
                self._receipt_transition(job, dispatch, **changes)
                results.append({'prompt_id': dispatch['prompt_id'], 'stage': dispatch['stage'], 'state': proof['state']})
            resolved = not self._open_engine_dispatches(job)
            if resolved:
                job.status = 'failed'
                job.stage = 'engine_terminal_after_restart'
                job.error = '原引擎已结束，本地未交付；未自动接图、重派或恢复旧研究。'
                job.cancel_error = None
                try:
                    self._append_history(job)
                except OSError as exc:
                    job.notes.append(f'已保存确切引擎终止回执，本地展示记录保存未确认：{exc}')
            return {'job_id': job.id, 'status': job.status, 'stage': job.stage, 'resolved': resolved,
                    'dispatches': results, 'message': job.error if resolved else
                    '原任务仍在运行／待处理；仅保存核对原件，未解除阻塞或恢复旧研究。'}

    async def _wait_for_engine_receipt_fence(self, job: Job) -> None:
        # Keep the original local queue item, without reserving another model's
        # lease or assuming that a released local lease made the backend idle.
        # Only an exact completed receipt clears this same-process fence; no
        # source replay, fabricated history, backend poll or old-study resume.
        blockers = self._other_open_engine_jobs(job)
        if blockers:
            job.notes.append('等待原引擎任务回执确认：' + ', '.join(blockers)
                             + '；没有提交本任务或释放全局缓存。')
        while blockers:
            self._stop_before_new_stage(job)
            job.status = 'queued'
            job.stage = 'waiting_for_unconfirmed_engine_receipt'
            await asyncio.sleep(.1)
            blockers = self._other_open_engine_jobs(job)
        self._stop_before_new_stage(job)
        job.stage = 'waiting_for_local_resource'

    async def _cancel_open_engine_dispatches(self, job: Job) -> None:
        # Worker cleanup and explicit stop use the same terminal commit lock.
        # Collect open stages only after acquiring it, never a stale open list.
        async with self._cancel_locks.setdefault(job.id, asyncio.Lock()):
            await self._cancel_open_engine_dispatches_locked(job)

    async def _cancel_open_engine_dispatches_locked(self, job: Job) -> None:
        for dispatch in self._open_engine_dispatches(job):
            if self._pre_dispatch_receipts.get(id(dispatch)) is dispatch:
                self._receipt_transition(job, dispatch, completed=True, submission_started=False,
                                         backend_dispatch_started=False, status='not_submitted')
                self._pre_dispatch_receipts.pop(id(dispatch))
                continue
            prompt_id = dispatch.get('prompt_id')
            if not prompt_id:
                raise OwnedCancellationUnconfirmed('引擎提交回执未确认，未声称取消或猜测其他 prompt。')
            mapped = self._owned_prompt_jobs.get(prompt_id)
            if mapped is not None and mapped != job.id:
                raise OwnedCancellationUnconfirmed('该 prompt 回执未绑定本次本地任务，不取消另一任务或猜补历史。')
            changed_owner = (dispatch.get('client_id') is not None
                             and dispatch['client_id'] != getattr(self.client, '_client_id', None))
            original_binding, frozen = None, None
            if mapped != job.id or changed_owner:
                try:
                    original_binding, frozen = self._strict_engine_receipt_binding(job, dispatch)
                except OwnedCancellationUnconfirmed as exc:
                    prefix = ('当前引擎客户端不是该回执的提交者；' if changed_owner else
                              '该 prompt 回执未绑定本次本地任务；')
                    raise OwnedCancellationUnconfirmed(prefix + str(exc)) from exc
            cancel = getattr(self.client, 'cancel_owned', None)
            if cancel is None:
                raise OwnedCancellationUnconfirmed('引擎客户端没有已核实的按任务原子取消契约。')
            receipt = await cancel(prompt_id, **original_binding) if original_binding else await cancel(prompt_id)
            if original_binding:
                self._recheck_engine_receipt_binding(job, dispatch, frozen)
                if (not isinstance(receipt, dict) or receipt.get('ownership_verified') is not True
                        or receipt.get('client_id') != dispatch['client_id']
                        or receipt.get('workflow_sha256') != dispatch['workflow_sha256']
                        or receipt.get('instance_id') != dispatch['backend_instance_id']):
                    raise OwnedCancellationUnconfirmed('原 owner／graph／后端实例取消回执未匹配。')
            if (not isinstance(receipt, dict) or receipt.get('prompt_id') != prompt_id
                    or receipt.get('absent_from_running_and_pending') is not True):
                raise OwnedCancellationUnconfirmed('未确认所绑定引擎任务已退出运行及待处理队列。')
            self._receipt_transition(job, dispatch, completed=True, status='cancelled_or_already_terminal',
                                     cancellation_receipt=deepcopy(receipt))
            if receipt.get('backend_state') == 'already_finished':
                job.notes.append('停止请求时原引擎任务已完成；没有声称它被中断，仅停止后续派发和迟到交付。')

    def _stop_before_new_stage(self, job: Job) -> None:
        if job.cancel_requested or job.status == 'cancelled':
            if self._open_engine_dispatches(job):
                raise OwnedCancellationUnconfirmed('已停止新阶段，但原引擎任务仍未确认终止。')
            job.status = job.stage = 'cancelled'
            job.cancel_error = None
            raise RuntimeError('任务已取消，没有排入下一阶段。')

    async def _queue_owned(self, job: Job, workflow: dict) -> str:
        self._reject_restored_receipt_fence()
        self._stop_before_new_stage(job)
        blockers = self._other_open_engine_jobs(job)
        if blockers:
            raise OwnedCancellationUnconfirmed('另一任务的引擎回执仍未确认，未向同后端提交新任务：'
                                               + ', '.join(blockers))
        if self._open_engine_dispatches(job):
            raise OwnedCancellationUnconfirmed('原阶段引擎回执未完成，未重复提交或排入另一阶段。')
        workflow = deepcopy(workflow)
        dispatch = {'stage': job.stage, 'prompt_id': None, 'client_id': getattr(self.client, '_client_id', None),
                    'workflow_sha256': self._workflow_digest(workflow),
                    'owner_request_sha256': self._workflow_digest(job.request),
                    'submission_started': False,
                    'backend_dispatch_started': False,
                    'backend_instance_id': None,
                    'completed': False, 'status': 'submitting'}
        job.engine_dispatches.append(dispatch)
        self._receipt_workflows.setdefault(job.id, []).append(deepcopy(workflow))
        try:
            self._persist_engine_receipts(job)  # Durable reservation before any queue await.
        except EngineReceiptPersistenceError as exc:
            dispatch.update(status='receipt_persistence_uncertain', persistence_error=str(exc))
            raise
        checked_queue = getattr(self.client.queue, '__func__', None) is ComfyClient.queue
        if checked_queue:
            self._pre_dispatch_receipts[id(dispatch)] = dispatch

        def close_unsubmitted() -> None:
            self._receipt_transition(job, dispatch, completed=True, submission_started=False,
                                     backend_dispatch_started=False, status='not_submitted')
            self._pre_dispatch_receipts.pop(id(dispatch), None)

        def before_dispatch() -> None:
            if job.cancel_requested or job.status == 'cancelled':
                close_unsubmitted()
                self._stop_before_new_stage(job)
            blockers = self._other_open_engine_jobs(job)
            if blockers:
                close_unsubmitted()
                raise OwnedCancellationUnconfirmed('另一任务的引擎回执仍未确认，未向同后端提交新任务：'
                                                   + ', '.join(blockers))
            if any(item is not dispatch for item in self._open_engine_dispatches(job)):
                close_unsubmitted()
                raise OwnedCancellationUnconfirmed('原阶段引擎回执未完成，未重复提交或排入另一阶段。')
            instance = getattr(self.client, '_dispatch_backend_instance_id', None)
            try:
                instance = instance if isinstance(instance, str) and str(uuid.UUID(instance)) == instance else None
            except ValueError:
                instance = None
            self._receipt_transition(job, dispatch, submission_started=True, backend_dispatch_started=True,
                                     backend_instance_id=instance)
            self._pre_dispatch_receipts.pop(id(dispatch), None)

        try:
            if checked_queue:
                prompt_id = await self.client.queue(workflow, before_dispatch=before_dispatch)
            else:
                # Legacy/fake clients retain their old queue signature. Their
                # awaited queue body is conservatively already in-flight.
                before_dispatch()
                prompt_id = await self.client.queue(workflow)
        except EngineSubmissionRejected as exc:
            self._pre_dispatch_receipts.pop(id(dispatch), None)
            self._receipt_transition(job, dispatch, completed=True, status='rejected_before_queue',
                                     rejection_response=deepcopy(exc.response_evidence))
            if job.cancel_requested:
                # A stop during the POST now has exact no-queue proof. There
                # is no engine prompt to cancel, and no late work to replay.
                job.status = job.stage = 'cancelled'
                job.cancel_error = job.error = None
            raise
        except BaseException:
            if dispatch.get('status') == 'receipt_persistence_uncertain':
                pass  # Preserve the failed checkpoint fence; never replay it.
            elif self._pre_dispatch_receipts.get(id(dispatch)) is dispatch:
                close_unsubmitted()
            elif not dispatch.get('completed'):
                self._receipt_transition(job, dispatch, status='submission_unknown')
            raise
        if not isinstance(prompt_id, str) or not prompt_id:
            self._receipt_transition(job, dispatch, status='submission_unknown')
            raise OwnedCancellationUnconfirmed('引擎提交没有返回确切 prompt_id，未猜补或重复提交。')
        self._owned_prompt_jobs[prompt_id] = job.id
        self._receipt_transition(job, dispatch, prompt_id=prompt_id, status='submitted')
        if job.cancel_requested:
            # A late acknowledgement resolves only this actual prompt. A failed
            # exact cancel leaves the worker observing it, not resubmitting.
            try:
                await self.cancel(job.id)
            except Exception:
                pass
        return prompt_id

    async def _wait_owned(self, job: Job, prompt_id: str) -> dict:
        dispatch = next(item for item in reversed(job.engine_dispatches) if item.get('prompt_id') == prompt_id)
        waiter = asyncio.create_task(self.client.wait(prompt_id))
        try:
            while not waiter.done():
                if job.status == 'cancelled' and dispatch.get('completed'):
                    raise RuntimeError('所绑定引擎任务已退出队列，不再等待或交付迟到输出。')
                await asyncio.wait({waiter}, timeout=.1)
            history = waiter.result()
            # Read the engine outside the lock, but serialize its durable
            # terminal commit with cancellation and explicit inspection. A
            # stop may still be awaiting its exact ACK when history arrives.
            async with self._cancel_locks.setdefault(job.id, asyncio.Lock()):
                if dispatch.get('completed'):
                    self._stop_before_new_stage(job)
                self._receipt_transition(job, dispatch, completed=True, status='history_returned',
                                         terminal_history=deepcopy(history))
                self._stop_before_new_stage(job)
            return history
        except BaseException:
            if not dispatch.get('completed') and dispatch.get('status') != 'receipt_persistence_uncertain':
                self._receipt_transition(job, dispatch, status='wait_unconfirmed')
            raise
        finally:
            if not waiter.done():
                waiter.cancel()
            await asyncio.gather(waiter, return_exceptions=True)

    def first_file(self, job: Job) -> Path | None:
        if job.files:
            return Path(job.files[0])
        if job.images:
            relative = job.images[0].removeprefix("/generated/")
            candidate = (self.output_root / relative).resolve()
            if candidate.is_relative_to(self.output_root.resolve()):
                return candidate
        return None

    def workflow_evidence(self, job: Job) -> dict:
        # Saved evidence must not depend on today's request schema or builders.
        # Otherwise an old job can fail to open, or display a different pipeline.
        workflow = None
        source = "reconstructed"
        scope = 'reconstruction_preview_not_actual_execution'
        submission = None
        image = self.first_file(job)
        candidates = []
        if image is not None:
            base = image.with_name(image.stem.rsplit("__", 1)[0])
            candidates.append(base.with_name(base.name + "__workflow.json"))
            # Early jobs used image_01.png + workflow.json in their own folder.
            candidates.append(image.parent / "workflow.json")
        for workflow_path in candidates:
            if workflow_path.exists():
                workflow = json.loads(workflow_path.read_text(encoding="utf-8"))
                if not isinstance(workflow, dict):
                    raise ValueError('已保存的工作流不是JSON对象，未以重建图替代。')
                source = str(workflow_path)
                scope = 'saved_workflow_record_not_independent_execution_verification'
                break
        if workflow is None and job.id in self._submitted_workflows:
            submitted = self._submitted_workflows[job.id]
            if self._workflow_digest(submitted['workflow']) != submitted['workflow_sha256']:
                raise ValueError('已提交的确切工作流快照校验失败，未以重建图替代。')
            workflow = deepcopy(submitted['workflow'])
            source = 'submitted_custom_workflow_current_process'
            scope = 'exact_custom_graph_at_submission_not_unobserved_later_stages'
            submission = {key: deepcopy(value) for key, value in submitted.items() if key != 'workflow'}
        if workflow is None:
            stages = self.stage_images.retained(job.id)
            if stages:
                workflow = {"stages": {item['stage_id']: item['workflow'] for item in stages},
                            "note": "仅包含已保存的阶段；不表示后续阶段已完成。"}
                source = "retained_stage_records"
                scope = 'saved_partial_stages_not_full_pipeline_completion'
        if workflow is None:
            request, _ = execution_parameters(GenerateRequest.model_validate(job.request))
            if request.render_pipeline in PIXEL_BRIDGE_PIPELINES:
                source_name, source_request = self._bridge_source(request)
                source_workflow = (
                    build_anima_base_workflow(source_request, self.settings, f"reconstructed/{job.id}/source")
                    if request.render_pipeline == RenderPipeline.HYBRID
                    else build_workflow(source_request, self.settings, f"reconstructed/{job.id}/source")
                )
                workflow = {
                    "pipeline": request.render_pipeline.value,
                    "bridge": f"{source_name} image -> NewBie VAE encode (reconstructed)",
                    "stages": {
                        "1_source": source_workflow,
                        "2_newbie": build_newbie_refine_workflow(request, self.settings, "reconstructed_source.png", f"reconstructed/{job.id}/final"),
                    },
                }
            else:
                workflow = build_workflow(request, self.settings, f"reconstructed/{job.id}")
        return {
            "job_id": job.id,
            "status": job.status,
            "source": source,
            "workflow_scope": scope,
            "workflow_sha256": self._workflow_digest(workflow),
            "workflow_hash_scope": 'canonical_json_graph_not_http_body_or_execution_receipt',
            "exact_workflow_available": source != 'reconstructed',
            "actual_workflow_available": (source != 'reconstructed' and
                (submission is None or submission['submission_state'] == 'local_queue_accepted')),
            "execution_verified": False,
            "submitted_workflow_receipt": submission,
            "evidence_note": ('没有保存或本进程提交的确切图；这是按当前构建器重建的预览，不能认证当次实际工作流。'
                              if source == 'reconstructed' else
                              '确切图来源与执行完成分别核对；当前进程快照不提供持久队列恢复，也不认证作品质量。'),
            "request": deepcopy(job.request),
            "semantic_spec": deepcopy(job.request.get("semantic_spec")),
            "workflow": workflow,
            "checks": deepcopy(job.checks),
        }

    def stage_evidence(self, job: Job) -> dict:
        warnings = []
        try:
            evidence = self.workflow_evidence(job)
        except (OSError, ValueError) as exc:
            evidence = None
            warnings.append(f'无法读取工作流记录，未猜测缓存来源：{exc}')
        records = self.stage_images.collect(job, evidence)
        return {'job_id': job.id, 'status': job.status, 'warnings': warnings,
                'images': [self.stage_images.public(record) for record in records]}

    def stage_record(self, job: Job, artifact_id: str) -> dict | None:
        try:
            evidence = self.workflow_evidence(job)
        except (OSError, ValueError):
            evidence = None
        return next((record for record in self.stage_images.collect(job, evidence)
                     if record['id'] == artifact_id or record.get('engine_output', {}).get('source_artifact_id') == artifact_id), None)

    async def _run(self) -> None:
        while True:
            job_id, request, custom_workflow = await self.queue.get()
            job = self.jobs[job_id]
            if job.status == "cancelled":
                idle_lease = None
                try:
                    # A cancelled tail can be the last entry after the actual
                    # image job skipped release for this then-pending item.
                    # It must never restart generation or bypass another
                    # inference owner just to reclaim an idle engine cache.
                    if self.queue.empty() and isinstance(self.client, ComfyClient):
                        idle_lease = await INFERENCE_LEASE.acquire(kind='generation', owner=job_id)
                        if self.queue.empty():
                            self.active_job_id = job_id
                            idle_lease.update(phase="idle_model_release")
                            await self._release_idle_engine_models(job)
                except Exception as exc:
                    detail = str(exc) or type(exc).__name__
                    job.notes.append(f"闲置模型释放未确认：{type(exc).__name__}: {detail}")
                finally:
                    if idle_lease is not None:
                        idle_lease.release()
                    if self.active_job_id == job_id:
                        self.active_job_id = None
                    try:
                        cleanup_note = cleanup_handoff_input(request,self.settings)
                        if cleanup_note:
                            job.notes.append(cleanup_note)
                        self._append_history(job)
                    finally:
                        self.queue.task_done()
                continue
            total_started = time.perf_counter()
            inference_lease = None
            try:
                job.stage = 'waiting_for_local_resource'
                await self._wait_for_engine_receipt_fence(job)
                inference_lease = await INFERENCE_LEASE.acquire(kind='generation', owner=job_id)
                self.active_job_id = job_id
                if job.status == 'cancelled':
                    continue
                job.status = "running"
                job.stage = "starting_engine"
                job.progress = 0.05
                # Queue entries are frozen at submission. Do not reroute or
                # replace professional parameters when execution starts.
                if request.repair_source_file:
                    protected_inputs(request, self.settings)
                verify_handoff_input(request,self.settings)
                if self._other_open_engine_jobs(job) or self._open_engine_dispatches(job):
                    raise OwnedCancellationUnconfirmed('前一任务引擎回执仍未确认，没有启动后端或清理缓存。')
                await self.client.ensure_running()
                self._stop_before_new_stage(job)
                if request.image_handoff:
                    # Pixel handoff does not require keeping the source model
                    # resident while the target model loads.
                    await self.client.free_memory()
                    self._stop_before_new_stage(job)
                if custom_workflow is not None:
                    workflow = custom_workflow
                    job.stage = "queued_in_engine"
                    job.progress = 0.1
                    prompt_id = await self._queue_owned(job, workflow)
                    job.stage = "sampling"
                    job.progress = 0.2
                    history = await self._wait_owned(job, prompt_id)
                elif request.render_pipeline in PIXEL_BRIDGE_PIPELINES:
                    history, workflow = await self._run_pixel_bridge(job, request)
                else:
                    workflow = build_workflow(request, self.settings, f"newbie_lab/{job_id}")
                    job.stage = "queued_in_engine"
                    job.progress = 0.1
                    submit_started = time.perf_counter()
                    prompt_id = await self._queue_owned(job, workflow)
                    job.timings["submit"] = time.perf_counter() - submit_started
                    job.stage = "sampling"
                    job.progress = 0.2
                    sample_started = time.perf_counter()
                    history = await self._wait_owned(job, prompt_id)
                    job.timings["engine"] = time.perf_counter() - sample_started
                if job.status == 'cancelled':
                    raise RuntimeError('任务已取消，不再处理迟到的输出')
                if request.check_enabled:
                    check_request = request
                    if request.parameter_policy == "legacy" and job.request.get("parameter_adjustments"):
                        # The former override applied only to base sampling;
                        # legacy optional check/repair stages used original values.
                        check_request = request.model_copy(update={
                            key: change["requested"] for key, change in job.request["parameter_adjustments"].items()
                        })
                    history, workflow = await self._run_checks(job, check_request, history, workflow)
                output_images = extract_output_images(history)
                if request.image_handoff and len(output_images)!=1:
                    raise ValueError('模型交接必须返回唯一图片')
                if request.repair_source_file and len(output_images) != 1:
                    raise ValueError('局部修改必须返回一张可验证的图片，未保存不完整或混合输出')
                if not output_images:
                    status = history.get("status", {})
                    label = status.get("status_str", "未完成")
                    raise RuntimeError(f"引擎未返回图片（{label}），任务可能已被中断")
                job.stage = "saving"
                job.progress = 0.9
                target_dir = self.output_root / request.output_group if request.output_group else self.output_root / job_id
                target_dir.mkdir(parents=True, exist_ok=True)
                label = request.output_label or f"{request.model_id.value}__{request.mode.value}__seed-{request.seed}"
                for index, image_info in enumerate(output_images, start=1):
                    payload = await self.client.view(
                        image_info["filename"],
                        image_info.get("subfolder", ""),
                        image_info.get("type", "output"),
                    )
                    self._stop_before_new_stage(job)
                    target = target_dir / f"{label}__{job_id}__{index:02d}.png"
                    if request.repair_source_file:
                        job.checks.append(verify_region_output(payload, request, self.settings))
                    if job.status == 'cancelled':
                        raise RuntimeError('任务已取消，不保存迟到的输出')
                    if request.image_handoff:
                        verify_handoff_input(request,self.settings)
                        job.checks.append(verify_handoff_output(payload,request))
                    target.write_bytes(payload)
                    relative = target.relative_to(self.output_root).as_posix()
                    job.images.append(f"/generated/{relative}")
                    job.files.append(str(target.resolve()))
                metadata_name = f"{label}__{job_id}"
                (target_dir / f"{metadata_name}__request.json").write_text(
                    json.dumps(job.request, ensure_ascii=False, indent=2), encoding="utf-8"
                )
                (target_dir / f"{metadata_name}__workflow.json").write_text(
                    json.dumps(workflow, ensure_ascii=False, indent=2), encoding="utf-8"
                )
                if job.checks:
                    (target_dir / f"{metadata_name}__check.json").write_text(
                        json.dumps(job.checks, ensure_ascii=False, indent=2), encoding="utf-8"
                    )
                job.status = "completed"
                job.stage = "completed"
                job.progress = 1
            except Exception as exc:
                if self._open_engine_dispatches(job):
                    try:
                        await self._cancel_open_engine_dispatches(job)
                    except Exception as cancel_exc:
                        job.cancel_error = str(cancel_exc) or type(cancel_exc).__name__
                if job.status == "cancelled":
                    job.stage = "cancelled"
                elif self._open_engine_dispatches(job):
                    job.status = job.stage = 'cancellation_uncertain'
                    job.error = f"{type(exc).__name__}: {exc}" if str(exc) else type(exc).__name__
                else:
                    job.status = "failed"
                    job.stage = "failed"
                    # httpx 超时/断连等异常的 str() 为空，必须带上类型名，否则错误字段空白无法定位
                    job.error = f"{type(exc).__name__}: {exc}" if str(exc) else type(exc).__name__
            finally:
                try:
                    if inference_lease is not None:
                        try:
                            if self.active_job_id == job_id and self.queue.empty():
                                inference_lease.update(phase="idle_model_release")
                                await self._release_idle_engine_models(job)
                        finally:
                            inference_lease.release()
                finally:
                    self.active_job_id = None
                    job.timings["total"] = time.perf_counter() - total_started
                    try:
                        waiting_unsubmitted = (job.status == 'queued'
                            and job.stage == 'waiting_for_unconfirmed_engine_receipt')
                        cleanup_note = (cleanup_handoff_input(request,self.settings)
                            if not self._open_engine_dispatches(job) and not waiting_unsubmitted else
                            '引擎提交或终止回执未确认；交接输入保留，没有删除在途任务来源。')
                        if cleanup_note:
                            job.notes.append(cleanup_note)
                        self._append_history(job)
                    finally:
                        self.queue.task_done()

    async def _release_idle_engine_models(self, job: Job) -> None:
        """Request idle cache release, without treating its ACK as freed RAM.

        The local lease excludes this API's next image/text inference. External
        Comfy clients remain outside that lease: observe their real queue and
        never turn an unavailable or malformed response into an idle queue.
        """
        if self._receipt_load_errors or any(self._open_engine_dispatches(other) for other in self.jobs.values()):
            job.notes.append('闲置模型释放已跳过：仍有未确认的引擎回执，未把后端当作空闲。')
            return
        # Only the project's real engine client has a known HTTP queue/free
        # contract. Alternate/in-memory clients must not accidentally contact
        # an unrelated server from their configured or default URL.
        if not isinstance(self.client, ComfyClient):
            return
        state = INFERENCE_LEASE.state()
        if (not self.queue.empty() or self.active_job_id != job.id
                or state.get("kind") != "generation" or state.get("owner") != job.id):
            return
        started = time.perf_counter()
        try:
            # No ensure_running: cleanup never starts a stopped engine.
            async with asyncio.timeout(10), httpx.AsyncClient(timeout=5, trust_env=False) as client:
                response = await client.get(self.settings.comfy_url.rstrip("/") + "/queue")
                response.raise_for_status()
                observed = response.json()
                if (not isinstance(observed, dict)
                        or not all(isinstance(observed.get(key), list)
                                   for key in ("queue_running", "queue_pending"))):
                    raise ValueError("引擎队列响应无法确认空闲")
                if observed["queue_running"] or observed["queue_pending"]:
                    job.notes.append("闲置模型释放已跳过：引擎仍有运行或待处理任务。")
                    return
                # A new application task can arrive during the queue GET.
                if not self.queue.empty():
                    job.notes.append("闲置模型释放已跳过：应用已有下一项待处理任务。")
                    return
                response = await client.post(self.settings.comfy_url.rstrip("/") + "/free",
                    json={"unload_models": True, "free_memory": True})
                response.raise_for_status()
            job.notes.append("闲置模型释放：已请求 /free；不代表卸载完成或可用内存已回升。")
        except Exception as exc:
            # Cache cleanup cannot invalidate a saved image or replace the
            # actual generation failure. Keep a diagnosable, bounded note.
            detail = str(exc) or type(exc).__name__
            job.notes.append(f"闲置模型释放未确认：{type(exc).__name__}: {detail}")
        finally:
            job.timings["idle_release_request"] = time.perf_counter() - started

    def _bridge_source(self, request: GenerateRequest) -> tuple[str, GenerateRequest]:
        from .bridge_sampling import source_request
        names = {RenderPipeline.HYBRID: 'Anima Turbo v1.1',
                 RenderPipeline.ILLUSTRIOUS_NEWBIE: 'Illustrious XL v2.0', RenderPipeline.Z_NEWBIE: 'Z-Image Turbo'}
        if request.render_pipeline not in names:
            raise ValueError(f"不是像素桥接管线：{request.render_pipeline.value}")
        return names[request.render_pipeline], source_request(request)

    async def _run_pixel_bridge(self, job: Job, request: GenerateRequest) -> tuple[dict, dict]:
        source_name, source_request = self._bridge_source(request)
        if request.render_pipeline == RenderPipeline.HYBRID:
            stage_one = build_anima_base_workflow(source_request, self.settings, f"newbie_lab/{job.id}_source_base")
        else:
            stage_one = build_workflow(source_request, self.settings, f"newbie_lab/{job.id}_source_base")
        job.stage = "source_composition"
        job.progress = 0.12
        started = time.perf_counter()
        prompt_id = await self._queue_owned(job, stage_one)
        anima_history = await self._wait_owned(job, prompt_id)
        job.timings["anima"] = time.perf_counter() - started
        images = extract_output_images(anima_history)
        if not images:
            raise RuntimeError(f"{source_name} 构图阶段没有返回图片")
        image_info = images[0]
        payload = await self.client.view(image_info["filename"], image_info.get("subfolder", ""), image_info.get("type", "output"))
        self._stop_before_new_stage(job)
        self.stage_images.retain(job.id, '1_source_composition', '源阶段输出', source_name,
                                 stage_one, image_info, payload)
        input_dir = Path(self.settings.engine_base_dir) / "input"
        input_dir.mkdir(parents=True, exist_ok=True)
        input_name = f"bridge_{job.id}_source.png"
        input_path = input_dir / input_name
        input_path.write_bytes(payload)
        try:
            self._stop_before_new_stage(job)
            await self.client.free_memory()
            self._stop_before_new_stage(job)
            await asyncio.sleep(1)
            self._stop_before_new_stage(job)
            stage_two = build_newbie_refine_workflow(request, self.settings, input_name, f"newbie_lab/{job.id}")
            job.stage = "newbie_identity"
            job.progress = 0.55
            started = time.perf_counter()
            prompt_id = await self._queue_owned(job, stage_two)
            history = await self._wait_owned(job, prompt_id)
        except Exception:
            # 失败时先掐掉引擎里可能仍排队的阶段二，再删过渡图；
            # 否则 LoadImage 会在我方放弃后继续执行并因找不到输入图而报错
            await self._cancel_open_engine_dispatches(job)
            input_path.unlink(missing_ok=True)
            raise
        input_path.unlink(missing_ok=True)
        for image_info in extract_output_images(history):
            payload = await self.client.view(image_info['filename'], image_info.get('subfolder', ''), image_info.get('type', 'output'))
            self._stop_before_new_stage(job)
            self.stage_images.retain(job.id, '2_newbie_identity_and_face', '后续阶段输出（检查前）', 'NewBie',
                                     stage_two, image_info, payload)
        job.timings["newbie_refine"] = time.perf_counter() - started
        job.timings["engine"] = job.timings["anima"] + job.timings["newbie_refine"]
        return history, {
            "pipeline": request.render_pipeline.value,
            "graph_version": "adaptive",
            "strategy": "specialist_stages",
            "bridge": f"{source_name} image -> unload models -> NewBie VAE encode",
            "roles": {
                "composition": source_name,
                "identity_and_character_style": "NewBie Exp0.1",
                "face_local_detail": "NewBie + FaceDetailer",
                "terminal_check": "CPU anime detectors + rule report",
            },
            "conditioning": {
                "composition_style_preset": request.composition_style_preset,
                "character_style_preset": request.character_style_preset,
                "legacy_style_preset": request.style_preset,
            },
            "stages": {"1_source_composition": stage_one, "2_newbie_identity_and_face": stage_two},
        }

    async def _run_checks(self, job: Job, request: GenerateRequest, history: dict, workflow: dict) -> tuple[dict, dict]:
        """check 环节：生成后全面质检，规则按风格类别区分（data/check_style_rules.json）。

        一次本地子进程跑完检测信号，再按风格规则出报告。check 只报告，
        不再根据“检测到手/文字”自动修改图片；真正的修改必须由用户圈选触发。
        """
        rules_table = load_style_rules()
        category = resolve_style_category(request, rules_table)
        rules = rules_for(rules_table, category)
        ctx = CheckContext(style_category=category, rules=rules)
        ctx.entries.append(CheckEntry("style", "skipped", f"检查规则类别：{category}（来自请求，不是图像风格达标结论）",
                                      {"category": category, "applied_rules": rules, "evidence_type": "configuration"}))

        images = extract_output_images(history)
        if not images:
            job.checks = [asdict(entry) for entry in ctx.entries]
            return history, workflow
        if len(images) > 1:
            ctx.entries.append(CheckEntry("batch", "skipped", "批量输出保留全部图片；逐图自动修复已跳过，避免只保留第一张"))
            job.checks = [asdict(entry) for entry in ctx.entries]
            return history, workflow

        info = images[0]
        payload = await self.client.view(info["filename"], info.get("subfolder", ""), info.get("type", "output"))
        self._stop_before_new_stage(job)
        input_dir = Path(self.settings.engine_base_dir) / "input"
        input_dir.mkdir(parents=True, exist_ok=True)
        src_name = f"check_{job.id}_src.png"
        mask_name = f"check_{job.id}_handmask.png"
        src_path = input_dir / src_name
        mask_path = input_dir / mask_name
        src_path.write_bytes(payload)

        local_checks = ("faces", "hands", "background", "aesthetic", "style_era", "breakdown")
        detection = None
        if any(rules[name].get("enabled", True) for name in local_checks):
            job.stage = "check_detect"
            started = time.perf_counter()
            try:
                detection = await run_detection(
                    self.settings, src_path, mask_path,
                    hand_conf=float(rules["hands"].get("conf", 0.5)),
                    max_hands=int(rules["hands"].get("max_regions", 4)),
                )
                job.timings["check_detect"] = time.perf_counter() - started
            except Exception as exc:
                if job.cancel_requested or job.status == 'cancelled':
                    src_path.unlink(missing_ok=True)
                    mask_path.unlink(missing_ok=True)
                    self._stop_before_new_stage(job)
                job.notes.append(f"check 本地检测失败已跳过：{type(exc).__name__}: {exc}")
                ctx.entries.append(CheckEntry("detection", "error", f"本地检测失败：{type(exc).__name__}"))
        try:
            self._stop_before_new_stage(job)
        except Exception:
            src_path.unlink(missing_ok=True)
            mask_path.unlink(missing_ok=True)
            raise
        ctx.detection = detection
        ctx.entries.extend(build_report(detection, rules, request))
        if request.parent_artwork_id:
            started = time.perf_counter()
            try:
                parent_path, root_path = self._comparison_sources(request.parent_artwork_id)
                evidence = await asyncio.to_thread(compare_images, payload, parent_path, root_path)
                evidence['parent_artwork_id'] = request.parent_artwork_id
                ctx.entries.append(CheckEntry('iteration_change', 'unknown',
                    '已比较上一版与源图的像素变化；这不是质量分，请结合阶段图判断收益与退化', evidence))
            except (OSError, ValueError, KeyError) as exc:
                ctx.entries.append(CheckEntry('iteration_change', 'unknown',
                    f'阶段比较未完成：{type(exc).__name__}；不能判定改进或退化'))
            try:
                self._stop_before_new_stage(job)
            except Exception:
                src_path.unlink(missing_ok=True)
                mask_path.unlink(missing_ok=True)
                raise
            job.timings['check_compare'] = time.perf_counter() - started

        hands_cfg = rules.get("hands", {})
        hands = (detection or {}).get("hands", {})
        if not hands_cfg.get("enabled", True):
            ctx.entries.append(CheckEntry("hands", "skipped", "该风格类别不检查手部"))
        elif detection is None:
            ctx.entries.append(CheckEntry("hands", "error", "本地检测失败，手部检查未执行"))
        elif hands.get("count", 0) > 0:
            ctx.entries.append(CheckEntry("hands", "warning", f"检测到 {hands.get('count', 0)} 处手部；检测器不能判断是否崩坏，仅供人工查看", {"boxes": hands.get("boxes", []), "scores": hands.get("scores", [])}))
        else:
            rejected = hands.get("rejected", [])
            from .check_intent import check_spec
            spec = check_spec(request)
            expects_visible_hands = bool(spec.get("left_hand") or spec.get("right_hand"))
            if expects_visible_hands:
                ctx.entries.append(CheckEntry("hands", "warning", "语义要求可见手部，但检测器没有找到可评估手部区域",
                                              {"rejected": rejected, "expected": {"left": spec.get("left_hand"), "right": spec.get("right_hand")}}))
            else:
                summary = "未检测到可评估手部区域；不能据此宣称手部正常"
                if rejected:
                    summary += f"（另有 {len(rejected)} 个候选框被门控丢弃）"
                ctx.entries.append(CheckEntry("hands", "skipped", summary, {"rejected": rejected}))

        text_cfg = rules.get("text", {})
        if not text_cfg.get("enabled", True):
            ctx.entries.append(CheckEntry("text", "skipped", "该风格类别允许文字存在（如海报），不把文字作为缺陷"))
        else:
            hints = text_tag_hints(detection, float(text_cfg.get("wd14_hint", 0.5))) if detection else []
            if hints:
                top = ", ".join(f"{tag}={prob:.2f}" for tag, prob in hints)
                ctx.entries.append(CheckEntry("text", "warning", f"标签信号提示可能存在文字：{top}；不会自动擦除，请人工确认", {"tags": {tag: prob for tag, prob in hints}}))
            elif detection is None:
                ctx.entries.append(CheckEntry('text', 'unknown', '本地检测未成功，无法判断是否存在文字'))
            else:
                ctx.entries.append(CheckEntry("text", "pass", "现有检测信号未提示明显文字；不会自动修改图片"))

        # 修复 pass 已消费完引擎 input 目录里的过渡文件，统一清理
        src_path.unlink(missing_ok=True)
        mask_path.unlink(missing_ok=True)
        job.checks = [asdict(entry) for entry in ctx.entries]
        return history, workflow

    def _comparison_sources(self, artwork_id: str):
        """Only read existing output records; never accept an arbitrary path."""
        def resolve(value):
            job_id, index = value.rsplit('-', 1)
            job = self.jobs.get(job_id)
            number = int(index)
            if job is None or number < 1 or number > len(job.files):
                raise ValueError('parent output missing')
            path = Path(job.files[number - 1]).resolve()
            if not path.is_relative_to(self.output_root.resolve()) or not path.is_file():
                raise ValueError('parent output outside workspace or missing')
            return job, path
        job, parent = resolve(artwork_id)
        root, seen = parent, {artwork_id}
        while job.request.get('parent_artwork_id'):
            ancestor = job.request['parent_artwork_id']
            if ancestor in seen or len(seen) >= 32:
                raise ValueError('invalid parent chain')
            seen.add(ancestor)
            try:
                job, root = resolve(ancestor)
            except (ValueError, OSError):
                return parent, None  # parent comparison remains useful
        return parent, root

    async def _maybe_hand_repair(self, job: Job, request: GenerateRequest, history: dict, workflow: dict,
                                 src_name: str, mask_name: str, ctx: CheckContext, hands: dict) -> tuple[dict, dict]:
        """手部修复 pass：hand_yolov9c 遮罩 + MaskToSEGS + DetailerForEach 局部重绘。失败降级不毁单。"""
        job.stage = "hand_repair"
        job.progress = max(job.progress, 0.75)
        started = time.perf_counter()
        stage = build_hand_repair_workflow(request, self.settings, src_name, mask_name, f"newbie_lab/{job.id}_handrepair")
        try:
            prompt_id = await self._queue_owned(job, stage)
            repair_history = await self._wait_owned(job, prompt_id)
        except Exception as exc:
            # 引擎里可能还排着这次重绘，先掐掉再降级返回原图
            await self._cancel_open_engine_dispatches(job)
            self._stop_before_new_stage(job)
            job.notes.append(f"手部重绘失败已跳过：{type(exc).__name__}")
            ctx.entries.append(CheckEntry("hands", "error", f"检测到 {hands.get('count', 0)} 处手部但重绘失败已跳过",
                                          {"boxes": hands.get("boxes", []), "scores": hands.get("scores", []),
                                           "error": type(exc).__name__}))
            return history, workflow
        job.timings["hand_repair"] = time.perf_counter() - started
        count = int(hands.get("count", 0))
        job.timings["hand_regions"] = float(count)
        job.notes.append(f"检测到 {count} 处手部区域，已局部重绘修复")
        ctx.entries.append(CheckEntry("hands", "repaired", f"修复 {count} 处手部（hand_yolov9c + DetailerForEach）",
                                      {"boxes": hands.get("boxes", []), "scores": hands.get("scores", []),
                                       "rejected": hands.get("rejected", [])}))
        if "stages" in workflow:
            workflow["stages"][f"{len(workflow['stages']) + 1}_hand_repair"] = stage
        else:
            workflow = {"pipeline": request.render_pipeline.value, "stages": {"1_generate": workflow, "2_hand_repair": stage}}
        workflow["hand_regions_detected"] = count
        return repair_history, workflow

    async def _maybe_text_clean(self, job: Job, request: GenerateRequest, history: dict, workflow: dict,
                                ctx: CheckContext | None = None,
                                exclude_boxes: list[list[float]] | None = None) -> tuple[dict, dict]:
        """文字清理阶段：漫画文字检测器找伪文字 -> 遮罩 -> 引擎内局部重绘。

        检测不到文字时零开销跳过。检测/重绘失败不让整单失败（图本身是好的），
        降级为返回未清理结果并在 timings 里留下标记，避免静默污染对比实验。
        exclude_boxes（人物框，来自 check 检测）内的文字不清理只报告：
        角色身上的文字是角色设定内容，擦除 = 破坏设定。
        """
        if not request.text_clean_enabled:
            return history, workflow
        images = extract_output_images(history)
        if not images:
            return history, workflow
        info = images[0]
        payload = await self.client.view(info["filename"], info.get("subfolder", ""), info.get("type", "output"))
        self._stop_before_new_stage(job)
        input_dir = Path(self.settings.engine_base_dir) / "input"
        input_dir.mkdir(parents=True, exist_ok=True)
        src_name = f"textclean_{job.id}_src.png"
        mask_name = f"textclean_{job.id}_mask.png"
        src_path = input_dir / src_name
        mask_path = input_dir / mask_name
        src_path.write_bytes(payload)
        job.stage = "text_detect"
        try:
            detection = await detect_text(self.settings, src_path, mask_path, exclude_boxes)
        except Exception as exc:
            src_path.unlink(missing_ok=True)
            mask_path.unlink(missing_ok=True)
            self._stop_before_new_stage(job)
            job.notes.append(f"文字检测失败已跳过：{type(exc).__name__}")
            if ctx is not None:
                ctx.entries.append(CheckEntry("text", "error", f"文字检测失败已跳过：{type(exc).__name__}"))
            return history, workflow
        try:
            self._stop_before_new_stage(job)
        except Exception:
            src_path.unlink(missing_ok=True)
            mask_path.unlink(missing_ok=True)
            raise
        if ctx is not None and detection.excluded:
            boxes = [item["box"] for item in detection.excluded]
            ctx.entries.append(CheckEntry(
                "text_on_character", "warning",
                f"角色身上发现 {len(boxes)} 处疑似文字——可能是角色设定内容，未自动清除，请人工确认是否正确渲染",
                {"boxes": boxes, "scores": [item["score"] for item in detection.excluded]}))
        if detection.count == 0 or not detection.mask:
            src_path.unlink(missing_ok=True)
            mask_path.unlink(missing_ok=True)
            if ctx is not None:
                msg = "未检测到伪文字"
                if detection.excluded:
                    msg = "背景无伪文字（角色身上文字按规则保留，见 text_on_character）"
                ctx.entries.append(CheckEntry("text", "pass", msg))
            return history, workflow
        job.stage = "text_clean"
        job.progress = max(job.progress, 0.8)
        started = time.perf_counter()
        stage = build_text_clean_workflow(request, self.settings, src_name, mask_name, f"newbie_lab/{job.id}_textclean")
        try:
            prompt_id = await self._queue_owned(job, stage)
            clean_history = await self._wait_owned(job, prompt_id)
        except Exception as exc:
            # 引擎里可能还排着这次重绘，先掐掉再清理过渡文件
            await self._cancel_open_engine_dispatches(job)
            self._stop_before_new_stage(job)
            src_path.unlink(missing_ok=True)
            mask_path.unlink(missing_ok=True)
            job.notes.append(f"文字重绘失败已跳过：{type(exc).__name__}")
            if ctx is not None:
                ctx.entries.append(CheckEntry("text", "error", f"检测到 {detection.count} 处伪文字但重绘失败已跳过",
                                              {"regions": detection.count, "boxes": detection.boxes,
                                               "error": type(exc).__name__}))
            return history, workflow
        src_path.unlink(missing_ok=True)
        mask_path.unlink(missing_ok=True)
        job.timings["text_clean"] = time.perf_counter() - started
        job.timings["text_regions"] = float(detection.count)
        job.notes.append(f"检测到 {detection.count} 处文字区域，已局部重绘清除")
        if ctx is not None:
            ctx.entries.append(CheckEntry("text", "repaired", f"清除 {detection.count} 处伪文字",
                                          {"regions": detection.count, "boxes": detection.boxes}))
        if "stages" in workflow:
            workflow["stages"][f"{len(workflow['stages']) + 1}_text_clean"] = stage
        else:
            workflow = {"pipeline": request.render_pipeline.value, "stages": {"1_generate": workflow, "2_text_clean": stage}}
        workflow["text_regions_detected"] = detection.count
        return clean_history, workflow

    def _append_history(self, job: Job) -> None:
        path = PROJECT_ROOT / "data" / "history.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(asdict(job), ensure_ascii=False) + "\n")

    def _load_history(self) -> None:
        path = PROJECT_ROOT / "data" / "history.jsonl"
        if not path.exists():
            return
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                raw = json.loads(line)
                job = Job(**raw)
            except (json.JSONDecodeError, TypeError):
                continue
            self.jobs[job.id] = job
            if not job.files:
                path = self.first_file(job)
                if path is not None:
                    job.files = [str(path)]

    def _load_engine_receipts(self) -> None:
        records, self._receipt_load_errors = self.engine_receipts.load()
        for record in records:
            job_id = record['job_id']
            existing = self.jobs.get(job_id)
            if existing and self._workflow_digest(existing.request) != record['request_sha256']:
                self._receipt_load_errors.append(f'{self.engine_receipts.path(job_id)}: 原 history 请求与回执不一致')
                continue
            dispatches = [{key: deepcopy(value) for key, value in item.items() if key != 'workflow'}
                          for item in record['dispatches']]
            self._receipt_workflows[job_id] = [deepcopy(item['workflow']) for item in record['dispatches']]
            if not any(not item['completed'] for item in dispatches):
                # A terminal checkpoint does not resurrect a job or import an
                # output. It can only clear a stale open receipt in history.
                if existing and self._open_engine_dispatches(existing):
                    existing.engine_dispatches = dispatches
                    if existing.status not in {'completed', 'failed', 'cancelled'}:
                        existing.status = 'failed'
                        existing.stage = 'engine_terminal_after_restart'
                        existing.error = '原引擎阶段已结束，本地交付未完成；未自动接图或重排任务。'
                continue
            job = existing or Job(job_id, 'cancellation_uncertain', record['created_at'], deepcopy(record['request']))
            job.engine_dispatches = dispatches
            job.status = job.stage = 'cancellation_uncertain'
            job.cancel_requested = record.get('cancel_requested', False)
            job.cancel_error = '重启前原引擎回执未确认；保留原 owner 和来源，未自动重派或接图。'
            self.jobs[job_id] = job
            self._restored_engine_jobs.add(job_id)
        # Old history with real open receipts still blocks; do not fabricate
        # missing exact graphs or silently upgrade its original client/epoch.
        self._restored_engine_jobs.update(job.id for job in self.jobs.values()
                                         if self._open_engine_dispatches(job))
