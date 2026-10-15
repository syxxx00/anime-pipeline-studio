"""A result-dependent run with explicit, reviewable decisions.

The queue supplies agent/human decisions. It is not a vision model and never
turns a detector score into an autonomous quality judgement.
"""
import asyncio

from fastapi import APIRouter, HTTPException

from .iteration_controller import IterationBudget, IterationDecision, UserSelectionConflict, UserSelectionRequest
from .schemas import GenerateRequest


class DecisionQueue:
    def __init__(self):
        self.pending = {}
        self.pending_feedback_signature = {}
        self.last_feedback_signature = {}
        self.pending_input_fingerprint = {}
        self.last_input_fingerprint = {}
        self.pending_observation_id = {}
        self.last_observation_id = {}

    def require_ready(self):
        return None

    def has_decision(self, run_id):
        return run_id in self.pending

    async def decide(self, context):
        run_id = context['run_id']
        self.last_feedback_signature[run_id] = self.pending_feedback_signature.pop(run_id, None)
        self.last_input_fingerprint[run_id] = self.pending_input_fingerprint.pop(run_id, None)
        self.last_observation_id[run_id] = self.pending_observation_id.pop(run_id, None)
        return self.pending.pop(run_id)


def iteration_router(controller, planner):
    router = APIRouter()

    def require(run_id):
        run = controller.store.runs.get(run_id)
        if run is None:
            raise HTTPException(404, '找不到迭代任务')
        return run

    def snapshot(run):
        return controller.snapshot(run) | {'decision_mode': 'local_result_guided_unverified' if run.autonomous_planning else 'agent_or_human_review',
            'decision_queued': run.id in planner.pending}

    @router.get('/api/iteration-runs')
    async def list_runs():
        return [snapshot(r) for r in sorted(controller.store.runs.values(), key=lambda r: r.created_at, reverse=True)]

    @router.post('/api/iteration-runs')
    async def launch(payload: dict):
        try:
            request = GenerateRequest.model_validate(payload.get('request'))
            if request.creation_intent is None:
                raise ValueError('此入口需要包含原句和任务目标的 creation_intent')
            run = controller.launch(request, payload.get('allowed_models', []),
                IterationBudget.model_validate(payload.get('budget', {})),
                baseline_artwork_id=payload.get('baseline_artwork_id', ''),
                baseline_evidence=payload.get('baseline_evidence', ''),
                seed_artwork_ids=payload.get('seed_artwork_ids', []),
                autonomous_planning=payload.get('autonomous_planning', False))
            return snapshot(run)
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc

    @router.get('/api/iteration-runs/{run_id}')
    async def get_run(run_id: str):
        return snapshot(require(run_id))

    @router.get('/api/iteration-runs/{run_id}/context')
    async def context(run_id: str):
        run = require(run_id)
        return controller._context(run)

    @router.post('/api/iteration-runs/{run_id}/planning')
    async def planning_mode(run_id: str, payload: dict):
        if set(payload) != {'autonomous_planning', 'revision'}:
            raise HTTPException(422, '规划选择必须包含且仅包含模式与当前版本')
        try:
            run = controller.set_planning_mode(require(run_id),
                autonomous_planning=payload['autonomous_planning'], revision=payload['revision'])
            return snapshot(run)
        except ValueError as exc:
            raise HTTPException(409, str(exc)) from exc
        except OSError as exc:
            raise HTTPException(503, '规划模式保存失败；任务仍保持暂停，请重新读取状态') from exc

    @router.get('/api/iteration-runs/{run_id}/source-reinterpretation')
    async def source_reinterpretation_history(run_id: str):
        run = require(run_id)
        return {'revision': run.revision, 'original_text': run.original_text,
                'original_sha256': run.original_sha256, 'history': run.source_revision_history,
                'scope': 'complete_history_not_current_quality_certification'}

    @router.post('/api/iteration-runs/{run_id}/source-reinterpretation')
    async def source_reinterpretation(run_id: str, payload: dict):
        from .source_revision_control import SourceRevisionConflict
        try:
            return snapshot(controller.reinterpret_source(require(run_id), payload))
        except SourceRevisionConflict as exc:
            raise HTTPException(409, str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc
        except OSError as exc:
            raise HTTPException(503, '解释修订未保存；当前解释与暂停状态没有改变') from exc

    @router.post('/api/iteration-runs/{run_id}/decisions')
    async def decision(run_id: str, payload: dict):
        run = require(run_id)
        lock = controller._locks.setdefault(run.id, asyncio.Lock())
        async with lock:
            if (run.status != 'running' or run_id in planner.pending
                    or any(s.status in {'reserved', 'submitted'} for s in run.steps)):
                raise HTTPException(409, '当前不能提交新决策，请读取任务状态')
            if payload.get('revision') != run.revision or payload.get('decision_count') != len(run.decisions):
                raise HTTPException(409, '决策上下文已变化，请重新读取原句、当前版本和检查结果')
            try:
                controller._verify_original(run)
                item = IterationDecision.model_validate(payload.get('decision'))
                if payload.get('observation_id') is not None:
                    if item.observation_id and item.observation_id != payload['observation_id']:
                        raise ValueError('决策与请求的 observation_id 不一致')
                    item = IterationDecision.model_validate(item.model_dump(mode='json') | {
                        'observation_id': payload['observation_id']})
                if item.original_text != run.original_text or item.original_sha256 != run.original_sha256:
                    raise ValueError('决策必须原样携带本任务原句及其校验值')
                known = {a.id for a in controller._artifacts(run)}
                checked = controller.preflight_decision(run, item, known)
                feedback_signature = controller.feedback_signature(run)
                has_feedback = any(a.reviews for a in controller._artifacts(run))
                if ((has_feedback or payload.get('feedback_signature') is not None)
                        and payload.get('feedback_signature') != feedback_signature):
                    raise ValueError('用户评价已变化，请重新读取原话与当前版本后决策')
                controller.validate_decision_observation(run, item)
                input_fingerprint = checked.submission_fingerprint if item.operation == 'local_edit' else None
            except ValueError as exc:
                raise HTTPException(422, str(exc)) from exc
            planner.pending[run_id] = item.model_dump(mode='json')
            planner.pending_feedback_signature[run_id] = feedback_signature
            planner.pending_input_fingerprint[run_id] = input_fingerprint
            planner.pending_observation_id[run_id] = item.observation_id
            run.events.append({'kind': 'decision_queued', 'original_sha256': item.original_sha256,
                               'decision': item.model_dump(mode='json')})
            controller.store.save(run)
        return snapshot(run)

    @router.post('/api/iteration-runs/{run_id}/select')
    async def user_select(run_id: str, payload: UserSelectionRequest):
        # Do not acquire the planner's lock: tick holds it while awaiting a
        # reply. This synchronous control closes the gate before a late reply
        # can dispatch, without touching an already in-flight image job.
        run = require(run_id)
        try:
            controller.select_by_user(run, payload)
            return snapshot(run)
        except UserSelectionConflict as exc:
            raise HTTPException(409, str(exc)) from exc
        except (OSError, ValueError) as exc:
            raise HTTPException(422, str(exc)) from exc

    @router.post('/api/iteration-runs/{run_id}/{action}')
    async def control(run_id: str, action: str):
        run = require(run_id)
        try:
            if action == 'pause':
                controller.pause(run)
            elif action == 'resume':
                controller.resume(run)
            elif action == 'cancel':
                await controller.cancel(run)
            else:
                raise HTTPException(404, '未知操作')
            planner.pending.pop(run.id, None)
            planner.pending_feedback_signature.pop(run.id, None)
            planner.pending_input_fingerprint.pop(run.id, None)
            planner.pending_observation_id.pop(run.id, None)
            return snapshot(run)
        except ValueError as exc:
            raise HTTPException(409, str(exc)) from exc

    return router
