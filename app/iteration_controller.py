"""Result-dependent execution. Fixed research matrices deliberately stay separate.

The planner proposes one operation at a time. Only this executor can submit
jobs; model output is data, never executable code or an arbitrary workflow.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import time
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, model_validator

from .generation_request import prepare_generation_request
from .image_handoff import cleanup_handoff_input, prepare_handoff
from .lora_catalog import describe_lora
from .region_edit import RegionEditOptions, generation_ancestor, prepare_region_edit
from .sampling_profiles import PROFILES
from .schemas import GenerateRequest, NewbieSemanticSpec


class Observation(BaseModel):
    model_config = {'extra': 'forbid'}
    artwork_id: str
    findings: str = Field(max_length=3000)
    uncertainty: str = Field(default='', max_length=2000)


class ExplorationRationale(BaseModel):
    """A bounded hypothesis, not a promise that a later model will repair it."""
    model_config = {'extra': 'forbid'}
    gains: list[str] = Field(min_length=1, max_length=16)
    temporary_losses: list[str] = Field(default_factory=list, max_length=16)
    hypothesis: str = Field(min_length=1, max_length=3000)
    success_criteria: list[str] = Field(min_length=1, max_length=16)
    stop_conditions: list[str] = Field(min_length=1, max_length=16)
    max_followup_jobs: int = Field(default=2, ge=1, le=4)

    @model_validator(mode='after')
    def meaningful(self):
        if not self.hypothesis.strip() or any(not value.strip() for values in (
                self.gains, self.temporary_losses, self.success_criteria, self.stop_conditions) for value in values):
            raise ValueError('探索必须说明具体收益、恢复假说、观察目标与止损条件')
        return self


class CandidateReview(BaseModel):
    """A planner's explicit comparison, not an automatic aesthetic score."""
    model_config = {'extra': 'forbid'}
    artwork_id: str
    baseline_artwork_id: str = ''
    verdict: Literal['baseline', 'retain', 'accept', 'explore', 'reject', 'uncertain']
    requirement_fit: Literal['better', 'same', 'worse', 'unknown'] = 'unknown'
    preservation: Literal['better', 'same', 'worse', 'unknown'] = 'unknown'
    technical_quality: Literal['better', 'same', 'worse', 'unknown'] = 'unknown'
    style_preference: Literal['better', 'same', 'worse', 'unknown'] = 'unknown'
    evidence: str = Field(min_length=1, max_length=5000)
    unresolved_requirements: list[str] = Field(default_factory=list, max_length=32)
    regressions: list[str] = Field(default_factory=list, max_length=32)
    exploration: ExplorationRationale | None = None

    @model_validator(mode='after')
    def evidence_gate(self):
        if not self.evidence.strip():
            raise ValueError('候选比较必须说明证据')
        if self.verdict == 'accept':
            axes = [self.requirement_fit, self.preservation, self.technical_quality, self.style_preference]
            if any(v in {'unknown', 'worse'} for v in axes) or 'better' not in axes or self.regressions or self.unresolved_requirements:
                raise ValueError('接受候选需要明确收益、无已知退化且关键要求已核验；不确定时保留原版本')
        if self.verdict == 'retain':
            axes = [self.requirement_fit, self.preservation, self.technical_quality, self.style_preference]
            if (self.requirement_fit not in {'same', 'better'} or 'better' not in axes
                    or self.preservation not in {'same', 'better'}
                    or self.technical_quality not in {'same', 'better'}
                    or self.style_preference == 'worse' or self.regressions):
                raise ValueError('保留中间版本需要内容、表现或画质的明确收益，且已有关系没有已知退化')
        if self.verdict == 'explore':
            axes = [self.requirement_fit, self.preservation, self.technical_quality, self.style_preference]
            if self.exploration is None or 'better' not in axes:
                raise ValueError('探索中间态需要明确阶段收益及有限的后续验证计划')
            if ('worse' in axes or self.regressions) and not self.exploration.temporary_losses:
                raise ValueError('已知退化必须在探索的暂时损失中明确保留，不能承诺后面自动修好')
        elif self.exploration is not None:
            raise ValueError('探索计划只能附在explore评审上，不能把拒收或交付改叫探索')
        return self


class IterationDecision(BaseModel):
    model_config = {'extra': 'forbid'}
    operation: Literal['generate', 'handoff', 'local_edit', 'rollback', 'finish', 'pause']
    reason: str = Field(min_length=1, max_length=4000)
    observations: list[Observation] = Field(default_factory=list, max_length=16)
    observation_id: str = Field(default='', max_length=80, pattern=r'^$|^[A-Za-z0-9][A-Za-z0-9_-]{0,79}$')
    target: Literal['newbie', 'anima', 'illustrious', 'z_image_turbo'] | None = None
    source_artwork_id: str = ''
    selected_artwork_id: str = ''
    prompt: str | None = Field(default=None, min_length=1, max_length=32000)
    negative_prompt: str | None = Field(default=None, max_length=16000)
    seed: int | None = Field(default=None, ge=0, le=2**63-1)
    denoise: float | None = Field(default=None, gt=0, le=1)
    objective: str = Field(default='', max_length=3000)
    preserve: list[str] = Field(default_factory=list, max_length=32)
    acceptance: list[str] = Field(default_factory=list, max_length=32)
    review: CandidateReview | None = None
    original_text: str = Field(default='', max_length=12000)
    original_sha256: str = ''
    goal_ids: list[str] = Field(default_factory=list, max_length=32)
    semantic_spec: NewbieSemanticSpec | None = None
    author_proposal: dict | None = None
    replace_selected_directions: list[str] = Field(default_factory=list, max_length=32)
    region_edit: RegionEditOptions | None = None

    @model_validator(mode='after')
    def applicable_fields(self):
        if self.operation != 'local_edit' and self.region_edit is not None:
            raise ValueError('区域和局部意图只能用于 local_edit')
        if self.operation in {'finish', 'rollback', 'pause'}:
            if (self.operation != 'pause' and not self.selected_artwork_id) or self.target or self.source_artwork_id or any(
                value is not None for value in (self.prompt, self.negative_prompt, self.seed, self.denoise, self.semantic_spec, self.author_proposal)
            ) or self.replace_selected_directions:
                raise ValueError('finish/rollback 必须选择已有作品；pause 可以在无作品时暂停；三者均不能夹带生成参数')
        else:
            if self.author_proposal is not None and self.semantic_spec is None:
                raise ValueError('重建作者提案必须同时提供完整当前规格')
            if self.replace_selected_directions and self.semantic_spec is None:
                raise ValueError('重新选择表现方向需要完整的当前语义方案')
            if self.semantic_spec is not None and (self.prompt is not None or self.negative_prompt is not None):
                raise ValueError('当前语义方案与手工原生词覆盖不能同时提交，避免记录与真实输入不一致')
            if not self.objective.strip() or not self.acceptance or any(not s.strip() for s in self.acceptance):
                raise ValueError('每次生成必须说明本次目标和验收要求，不能仅指定模型链')
            if not self.target or self.selected_artwork_id:
                raise ValueError('生成操作必须选择目标模型，不能同时宣称结束')
            if self.operation in {'handoff', 'local_edit'}:
                if not self.preserve or any(not s.strip() for s in self.preserve):
                    raise ValueError('交接必须声明哪些已有内容需要保留')
                if not self.source_artwork_id:
                    raise ValueError('图像操作必须明确源作品')
                if self.operation == 'handoff' and self.denoise is None:
                    raise ValueError('handoff 必须明确重绘强度')
                if self.operation == 'local_edit' and (self.target != 'newbie' or self.region_edit is None or self.denoise is not None):
                    raise ValueError('local_edit 当前仅由 NewBie 执行，必须提供 region_edit，重绘参数置于该字段')
                if self.operation == 'local_edit' and self.region_edit.source_sha256 is None:
                    raise ValueError('local_edit 必须提供圈选时所见原图的 source_sha256，不能只凭作品ID套用遮罩')
            elif self.source_artwork_id or self.denoise is not None:
                raise ValueError('generate 不接受图像源或重绘强度')
        return self


class IterationBudget(BaseModel):
    model_config = {'extra': 'forbid'}
    max_jobs: int = Field(default=6, ge=1, le=12)
    max_decisions: int = Field(default=12, ge=1, le=24)
    max_active_seconds: int = Field(default=1800, ge=30, le=28800)
    max_rejected_candidates: int = Field(default=2, ge=1, le=5)


class UserSelectionRequest(BaseModel):
    """Select the viewed version; this is not a quality or goal certification."""
    model_config = {'extra': 'forbid'}
    request_id: str = Field(min_length=8, max_length=128, pattern=r'^[a-zA-Z0-9_-]+$')
    artwork_id: str = Field(min_length=1, max_length=128, pattern=r'^\S+$')
    artwork_sha256: str = Field(pattern=r'^[0-9a-f]{64}$')
    revision: int = Field(ge=0, strict=True)
    decision_count: int = Field(ge=0, strict=True)
    feedback_signature: str = Field(pattern=r'^[0-9a-f]{64}$')


class UserSelectionConflict(ValueError):
    """A selection context changed or dispatch cannot safely be closed yet."""


class IterationStep(BaseModel):
    id: str
    decision: IterationDecision
    status: Literal['reserved', 'submitted', 'completed', 'failed'] = 'reserved'
    job_id: str = ''
    artwork_id: str = ''
    error: str = ''
    exploration_root_artwork_id: str = ''
    submission_fingerprint: str = ''


class ExplorationBranch(BaseModel):
    root_artwork_id: str
    artwork_ids: list[str] = Field(default_factory=list)
    max_followup_jobs: int = Field(ge=1, le=4)
    followup_jobs: int = 0
    origin_incumbent_artwork_id: str = ''

    @property
    def remaining_jobs(self):
        return max(0, self.max_followup_jobs - self.followup_jobs)


class IterationRun(BaseModel):
    id: str
    created_at: float
    updated_at: float
    status: Literal['running', 'paused', 'completed', 'cancelled', 'exhausted'] = 'running'
    base_request: dict
    allowed_models: list[str]
    budget: IterationBudget
    steps: list[IterationStep] = Field(default_factory=list)
    decisions: list[IterationDecision] = Field(default_factory=list)
    decision_attempts: int = 0
    selected_artwork_id: str = ''
    stop_reason: str = ''
    active_seconds: float = 0
    active_since: float | None = None
    revision: int = 0
    events: list[dict] = Field(default_factory=list)
    incumbent_artwork_id: str = ''
    candidate_reviews: list[CandidateReview] = Field(default_factory=list)
    consecutive_rejections: int = 0
    original_text: str = ''
    original_sha256: str = ''
    seed_artwork_ids: list[str] = Field(default_factory=list)
    exploration_artwork_ids: list[str] = Field(default_factory=list)
    exploration_branches: list[ExplorationBranch] = Field(default_factory=list)
    selection_authority: Literal['', 'machine', 'user'] = ''
    user_selection_receipt: dict | None = None
    autonomous_planning: bool = False
    source_revision_history: list[dict] = Field(default_factory=list)


@dataclass(frozen=True)
class DecisionPreflight:
    """Detached post-review state; no review or input has been committed yet."""
    reviewed_run: IterationRun
    review_stop_reason: str = ''
    submission_fingerprint: str = ''


class IterationStore:
    def __init__(self, root: Path):
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)
        self.runs: dict[str, IterationRun] = {}
        self.load_errors: list[str] = []
        for path in self.root.glob('*.json'):
            try:
                run = IterationRun.model_validate_json(path.read_text(encoding='utf-8'))
                if path.stem != run.id:
                    raise ValueError('run id/path mismatch')
                # A saved record is not evidence of a live worker. Never replay
                # submissions automatically after a process restart.
                if run.status == 'running':
                    run.status = 'paused'
                    run.stop_reason = 'service_restart'
                    run.active_since = None
                    run.revision += 1
                    self.save(run)
                self.runs[run.id] = run
            except (OSError, ValueError) as exc:
                self.load_errors.append(f'{path.name}: {type(exc).__name__}')

    def save(self, run: IterationRun):
        if len(run.id) != 12 or any(c not in '0123456789abcdef' for c in run.id):
            raise ValueError('invalid iteration id')
        run.updated_at = time.time()
        path = self.root / f'{run.id}.json'
        temporary = path.with_suffix('.tmp')
        temporary.write_text(run.model_dump_json(indent=2), encoding='utf-8')
        temporary.replace(path)
        self.runs[run.id] = run


def target_request(base: GenerateRequest, target: str) -> GenerateRequest:
    """Own native channel and matching user settings, no artistic model roles."""
    if target not in PROFILES:
        raise ValueError('未知目标模型')
    profile = PROFILES[target]
    channel = 'z_image' if target == 'z_image_turbo' else target
    native = (base.native_prompts or {}).get(channel) or {}
    same = base.model_id.value == profile['model_id']
    positive = native.get('positive') or (base.prompt if same else '')
    if not positive or not positive.strip():
        raise ValueError(f'{target} 缺少原生文本；不能借用其他模型格式')
    from .model_conditioning import negative_for_target
    negative = negative_for_target(base.model_dump(mode='json'), native, target)
    owner = base if same else None
    if owner is None and base.source_parameters and base.source_parameters.model_id == profile['model_id']:
        from .bridge_sampling import source_request
        owner = source_request(base)
    params = dict(profile, seed=base.seed, lora_name=None, lora_strength=.75, cfg_rescale=.9,
                  weight_dtype='fp8_e5m2' if target == 'z_image_turbo' else None)
    if owner:
        for key in ('steps', 'cfg', 'sampler', 'scheduler', 'shift', 'seed', 'cfg_rescale', 'weight_dtype', 'lora_name', 'lora_strength'):
            params[key] = getattr(owner, key)
    return GenerateRequest.model_validate(base.model_dump(mode='json') | params | {
        'parameter_policy': 'explicit', 'render_pipeline': target,
        'prompt': positive, 'negative_prompt': negative, 'use_default_negative_prompt': False,
        'source_parameters': None, 'source_prompt': '', 'source_negative_prompt': None, 'anima_prompt': '',
        'batch_size': 1, 'highres_enabled': False, 'face_detail_enabled': False,
        'hand_repair_enabled': False, 'text_clean_enabled': False,
        'image_handoff': None, 'parent_artwork_id': None, 'operation': 'iteration_generate',
    })


class IterationController:
    def __init__(self, store, jobs, artworks, planner, *, clock=time.time, observations=None):
        self.store, self.jobs, self.artworks, self.planner = store, jobs, artworks, planner
        self.observations = observations
        self.clock = clock
        self.worker = None
        self._locks: dict[str, asyncio.Lock] = {}
        self._tick_tasks: dict[str, asyncio.Task] = {}

    def launch(self, request: GenerateRequest, allowed_models: list[str], budget: IterationBudget,
               *, baseline_artwork_id: str = '', baseline_evidence: str = '', seed_artwork_ids: list[str] | None = None,
               autonomous_planning: bool = False):
        self.planner.require_ready()
        if type(autonomous_planning) is not bool:
            raise ValueError('autonomous_planning 必须为显式布尔值')
        if autonomous_planning and (getattr(self.planner, 'controller', None) is not self or self.observations is None):
            raise ValueError('本服务没有绑定本地结果驱动规划器和原件观察能力')
        if not allowed_models or len(set(allowed_models)) != len(allowed_models):
            raise ValueError('必须提供不重复的可用模型列表')
        if request.parameter_policy != 'explicit' or not request.scene_contract:
            raise ValueError('自主迭代需要已编译的显式请求与场景契约')
        from .scene_contract import SceneContractV3
        if not SceneContractV3.model_validate(request.scene_contract).generation_ready:
            raise ValueError('请先处理意图中的冲突和严重缺失')
        if request.batch_size != 1 or request.highres_enabled or request.face_detail_enabled or request.hand_repair_enabled or request.text_clean_enabled or request.repair_source_file or request.image_handoff:
            raise ValueError('迭代基础请求应为单张且不携带附加修脸、放大或像素输入；局部修改使用审查后的 local_edit 动作')
        for model in allowed_models:
            candidate = target_request(request, model)
            self._validate_resources(candidate)
            prepare_generation_request(candidate, self.jobs.settings)
        seeds = list(dict.fromkeys([*(seed_artwork_ids or []), *([baseline_artwork_id] if baseline_artwork_id else [])]))
        self.artworks.sync_jobs(self.jobs.list())
        if any(self.artworks.get(id) is None for id in seeds):
            raise ValueError('引用的起点作品不存在')
        if baseline_artwork_id and not baseline_evidence.strip():
            raise ValueError('复用起点需要说明观察依据和仍待解决的问题')
        original = (request.creation_intent.original_text if request.creation_intent
                    else str(request.scene_contract.get('source_text', '')))
        now = self.clock()
        run = IterationRun(id=uuid.uuid4().hex[:12], created_at=now, updated_at=now,
                           base_request=request.model_dump(mode='json'), allowed_models=allowed_models,
                           budget=budget, active_since=now, seed_artwork_ids=seeds,
                           original_text=original, original_sha256=hashlib.sha256(original.encode('utf-8')).hexdigest(),
                           autonomous_planning=autonomous_planning)
        self._verify_original(run)
        if baseline_artwork_id:
            run.incumbent_artwork_id = baseline_artwork_id
            run.candidate_reviews.append(CandidateReview(artwork_id=baseline_artwork_id, verdict='baseline',
                evidence=baseline_evidence, unresolved_requirements=[g.id for g in request.creation_intent.goals
                    if g.priority == 'required'] if request.creation_intent else []))
            run.events.append({'kind': 'reused_baseline', 'artwork_id': baseline_artwork_id,
                               'evidence': baseline_evidence, 'original_sha256': run.original_sha256})
        self.store.save(run)
        return run

    def _validate_resources(self, request):
        if request.lora_name:
            root = self.jobs.settings.path_map['loras'].resolve()
            path = (root / request.lora_name).resolve()
            if not path.is_relative_to(root) or not path.is_file() or request.model_id.value not in describe_lora(path, root)['compatible_models']:
                raise ValueError('目标模型 LoRA 缺少兼容架构证据')

    def start(self):
        if self.worker is None or self.worker.done():
            self.worker = asyncio.create_task(self._run(), name='result-dependent-iteration')

    async def stop(self):
        if self.worker:
            self.worker.cancel()
            try:
                await self.worker
            except asyncio.CancelledError:
                pass
        error = None
        try:
            for run in list(self.store.runs.values()):
                if run.status == 'running':
                    try:
                        self.pause(run, 'service_shutdown')
                    except Exception as exc:
                        error = error or exc
        finally:
            tasks = list(self._tick_tasks.values())
            for task in tasks:
                task.cancel()
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)
            self._tick_tasks.clear()
        if error is not None:
            raise error

    async def _run(self):
        while True:
            for run in list(self.store.runs.values()):
                if run.status == 'running':
                    task = self._tick_tasks.get(run.id)
                    if task is None or task.done():
                        if task is not None and not task.cancelled():
                            task.exception()  # Observe, never replay, a finished tick.
                        self._tick_tasks[run.id] = asyncio.create_task(self.tick(run), name='iteration-tick-' + run.id)
            await asyncio.sleep(1)

    def _cancel_planning(self, run):
        cancel = getattr(self.planner, 'cancel_inference', None)
        if cancel is not None:
            cancel(run.id)

    def _account(self, run):
        if run.active_since is not None:
            run.active_seconds += max(0, self.clock() - run.active_since)
            run.active_since = self.clock() if run.status == 'running' else None

    def pause(self, run, reason='user'):
        if run.status != 'running':
            raise ValueError('只能暂停正在执行的任务')
        self._account(run)
        run.status, run.stop_reason, run.active_since = 'paused', reason, None
        run.revision += 1
        self._cancel_planning(run)
        self.store.save(run)
        return run

    def resume(self, run):
        if run.status != 'paused':
            raise ValueError('只能继续已暂停任务')
        # Missing/reserved submissions are reconciled in tick, not replayed.
        self.planner.require_ready()
        run.status, run.stop_reason, run.active_since = 'running', '', self.clock()
        run.revision += 1
        self.store.save(run)
        return run

    def set_planning_mode(self, run, *, autonomous_planning: bool, revision: int):
        """Explicit user choice while dispatch is closed; never resumes a run."""
        if type(autonomous_planning) is not bool or type(revision) is not int:
            raise ValueError('规划模式与版本须为显式布尔值和整数')
        if run.status != 'paused' or revision != run.revision:
            raise ValueError('请先暂停并读取最新版本，再明确选择规划模式')
        if (run.id in getattr(self.planner, 'pending', {})
                or any(step.status in {'reserved', 'submitted'} for step in run.steps)):
            raise ValueError('还有在途提交或待执行决策；先核对当前任务，不能覆盖它们')
        if autonomous_planning and (getattr(self.planner, 'controller', None) is not self or self.observations is None):
            raise ValueError('本服务没有绑定本地结果驱动规划器和原件观察能力')
        if run.autonomous_planning == autonomous_planning:
            return run
        self._cancel_planning(run)
        previous = run.autonomous_planning, run.revision
        run.autonomous_planning = autonomous_planning
        run.revision += 1
        event = {'kind': 'user_planning_mode', 'autonomous_planning': autonomous_planning,
                 'scope': 'future_proposals_only_not_resume_or_quality_certification'}
        run.events.append(event)
        try:
            self.store.save(run)
        except OSError:
            run.autonomous_planning, run.revision = previous
            run.events.pop()
            raise
        return run

    def artwork_sha256(self, artwork):
        """Hash only an existing output original, never an arbitrary supplied path."""
        path = Path(artwork.file_path).resolve()
        if not path.is_relative_to(Path(self.jobs.settings.output_dir).resolve()) or not path.is_file():
            raise ValueError('所选作品原图不存在或不在输出目录内，请重新查看作品')
        digest = hashlib.sha256()
        with path.open('rb') as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b''):
                digest.update(chunk)
        return digest.hexdigest()

    def reinterpret_source(self, run, payload):
        from .source_revision_control import apply_source_revision
        return apply_source_revision(self, run, payload)

    def select_by_user(self, run, selection: UserSelectionRequest):
        """Synchronous dispatch gate; do not wait behind an in-flight planner.

        No await may occur between validation and closing the live gate. A
        planner already awaiting a reply must see the changed status/revision
        before it can reserve or submit another image. Image submissions that
        have already started are instead refused here, never guessed/cancelled.
        """
        payload = selection.model_dump(mode='json')
        fingerprint = hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True).encode('utf-8')).hexdigest()
        receipt = run.user_selection_receipt
        if receipt is not None:
            if receipt.get('request_id') == selection.request_id and receipt.get('request_fingerprint') == fingerprint:
                if (run.status != 'completed' or run.stop_reason != 'user_selected'
                        or run.selection_authority != 'user' or run.selected_artwork_id != receipt.get('artwork_id')):
                    raise UserSelectionConflict('选择收据与任务状态不一致，请核对任务，不能当作成功重读')
                return run  # Replay is an observation of the original durable receipt.
            raise UserSelectionConflict('本任务已保存用户选择；同一请求不能改写，新的创作请另建任务')
        self._verify_original(run)
        artifacts = self._artifacts(run)  # Reconcile known terminal results only.
        referenced = {step.job_id for step in run.steps if step.job_id}
        jobs = self.jobs.list()
        owned = {job.id for job in jobs if job.request.get('output_group') == f'iteration_runs/{run.id}'}
        if (any(step.status in {'reserved', 'submitted'} for step in run.steps)
                or owned - referenced
                or any(job.status not in {'completed', 'failed', 'cancelled'} for job in jobs
                       if job.id in referenced | owned)):
            raise UserSelectionConflict('仍有在途图像或归属未核对的提交；请先暂停/取消，等待结果并核对队列后选择，不会自动重发')
        if selection.revision != run.revision or selection.decision_count != len(run.decisions):
            raise UserSelectionConflict('选择上下文已变化，请重新读取任务、原图与当前评价')
        if selection.feedback_signature != self.feedback_signature(run):
            raise UserSelectionConflict('用户评价已变化，请重新查看原话再选择版本')
        artwork = next((item for item in artifacts if item.id == selection.artwork_id), None)
        if artwork is None:
            raise ValueError('只能选择属于本任务的已有作品')
        if selection.artwork_sha256 != self.artwork_sha256(artwork):
            raise UserSelectionConflict('所见原图像素已变化，请重新打开当前原图后选择')

        receipt = {'request_id': selection.request_id, 'request_fingerprint': fingerprint,
                   'request': payload, 'artwork_id': selection.artwork_id,
                   'artwork_sha256': selection.artwork_sha256, 'original_sha256': run.original_sha256,
                   'selected_at': self.clock(), 'scope': 'version_choice_only_not_quality_certification'}
        previous = (run.status, run.stop_reason, run.selected_artwork_id, run.selection_authority)
        self._account(run)
        run.status, run.stop_reason, run.active_since = 'completed', 'user_selected', None
        run.selected_artwork_id, run.selection_authority = selection.artwork_id, 'user'
        run.user_selection_receipt = receipt
        run.revision += 1
        run.events.append({'kind': 'user_selected', 'receipt': receipt})
        try:
            self.store.save(run)  # Gate is closed and receipt durable before returning.
        except OSError as exc:
            # A failed disk write is not a committed choice. Keep dispatch
            # closed, without exposing an in-memory receipt as a durable ack.
            run.selected_artwork_id, run.selection_authority = previous[2:]
            run.user_selection_receipt = None
            run.events.pop()
            run.events.append({'kind': 'user_selection_storage_failed', 'request_id': selection.request_id})
            run.status, run.stop_reason = (('paused', 'user_selection_storage_failed')
                if previous[0] == 'running' else previous[:2])
            raise UserSelectionConflict('选择收据没有保存成功；后续派发已停止，请重新读取状态后选择') from exc
        finally:
            self._cancel_planning(run)
            for name in ('pending', 'pending_feedback_signature', 'pending_input_fingerprint', 'pending_observation_id'):
                getattr(self.planner, name, {}).pop(run.id, None)
        return run

    async def cancel(self, run):
        if run.status in {'completed', 'exhausted'}:
            raise ValueError('任务已经结束')
        self._account(run)
        run.status, run.stop_reason, run.active_since = 'cancelled', 'user', None
        run.revision += 1
        self._cancel_planning(run)
        self.store.save(run)  # dispatch gate closes before first await
        for step in run.steps:
            if not step.job_id and step.status == 'reserved':
                # A lost submit response does not mean the queue rejected it.
                # Reconcile only one exact owned label; never repost or guess.
                matches = [job for job in self.jobs.list() if
                    job.request.get('output_group') == f'iteration_runs/{run.id}' and
                    job.request.get('output_label') == f'{run.id}__{step.id}']
                if len(matches) == 1:
                    step.job_id, step.status = matches[0].id, 'submitted'
                    self.store.save(run)
                else:
                    run.events.append({'kind': 'cancel_submission_unresolved', 'step_id': step.id,
                                       'message': '提交结果未知；未猜测任务ID、未重发，需核对队列。'})
                    self.store.save(run)
            if step.job_id:
                await self.jobs.cancel(step.job_id)
        return run

    def _reconcile_terminal_steps(self, run):
        """Observe completed jobs even while dispatch is paused; never resubmit."""
        changed = False
        for step in run.steps:
            if step.status != 'submitted' or not step.job_id:
                continue
            job = self.jobs.get(step.job_id)
            if job is None or job.status not in {'completed', 'failed', 'cancelled'}:
                continue
            if job.status == 'completed' and len(job.files) == len(job.images) == 1:
                step.status, step.artwork_id = 'completed', self.artworks.id_for(job.id, 0)
            else:
                step.status, step.error = 'failed', job.error or '生成没有返回一张完整图片'
            changed = True
        if changed:
            self.store.save(run)

    def _artifacts(self, run):
        self._reconcile_terminal_steps(run)
        owned = [self.jobs.get(s.job_id) for s in run.steps if s.job_id]
        self.artworks.sync_jobs([job for job in owned if job is not None and job.status == 'completed'])
        ids = list(dict.fromkeys([*run.seed_artwork_ids, *[s.artwork_id for s in run.steps if s.artwork_id]]))
        return [self.artworks.get(id) for id in ids if self.artworks.get(id)]

    def _verify_original(self, run):
        if not run.original_text:
            return
        intent = GenerateRequest.model_validate(run.base_request).creation_intent
        if hashlib.sha256(run.original_text.encode('utf-8')).hexdigest() != run.original_sha256:
            raise ValueError('任务原句已变化，停止派发以避免逐步改写意图')
        if intent and (intent.original_text != run.original_text or intent.original_sha256 != run.original_sha256):
            raise ValueError('基础请求与任务最初原句不一致')
        if intent:
            for value in ((run.base_request.get('scene_contract') or {}).get('source_text'),
                          (run.base_request.get('semantic_spec') or {}).get('source_sentence')):
                if value is not None and value != run.original_text:
                    raise ValueError('编译上下文丢失了最初原句，不能派发')

    def feedback_signature(self, run):
        """Version the user's words, not our classification of those words."""
        feedback = {a.id: a.reviews for a in self._artifacts(run)}
        return hashlib.sha256(json.dumps(feedback, ensure_ascii=False, sort_keys=True).encode('utf-8')).hexdigest()

    def observation_capability(self):
        available = self.observations is not None
        return {'available': available, 'image_dependent_decisions_require_observation_id': available,
                'original_evidence_only': True, 'quality_verified': False,
                'semantic_understanding_verified': False}

    @staticmethod
    def decision_artwork_ids(decision):
        # Membership/roles follow this actual decision, not all historical seeds
        # or a permanently prescribed root/incumbent/model sequence.
        ids = [decision.source_artwork_id, decision.selected_artwork_id,
               *[item.artwork_id for item in decision.observations]]
        if decision.review is not None:
            ids += [decision.review.artwork_id, decision.review.baseline_artwork_id]
        return list(dict.fromkeys(value for value in ids if value))

    def validate_decision_observation(self, run, decision, *, queued_observation_id=None):
        if queued_observation_id is not None and queued_observation_id != decision.observation_id:
            raise ValueError('排队的观察原件与决策声明不一致，请重新观察')
        references = self.decision_artwork_ids(decision)
        if self.observations is None:
            if decision.observation_id:
                raise ValueError('当前服务未接入观察原件校验，不能把包标识当成已验证')
            return None
        if not decision.observation_id:
            if references:
                raise ValueError('图像相关决策必须携带 observation_id，并包含全部引用原图')
            return None  # A genuinely image-free first generate/pause needs no package.
        manifest = self.observations.validate_still_current(decision.observation_id, self._context(run))
        if not set(references) <= set(manifest['images']):
            raise ValueError('观察包没有实际保存本次决策引用的全部原图，请重新观察')
        return manifest

    @staticmethod
    def _last_review(run, artwork_id):
        return next((r for r in reversed(run.candidate_reviews) if r.artwork_id == artwork_id), None)

    def _refresh_exploration_sources(self, run):
        run.exploration_artwork_ids = list(dict.fromkeys(
            artwork_id for branch in run.exploration_branches if branch.remaining_jobs
            for artwork_id in branch.artwork_ids
            if (review := self._last_review(run, artwork_id)) is not None and review.verdict == 'explore'
        ))

    def _exploration_branch(self, run, artwork_id):
        return next((branch for branch in run.exploration_branches if artwork_id in branch.artwork_ids), None)

    def validate_handoff_source(self, run, decision, known):
        """API preflight and executor use the same post-review source policy."""
        source = run.incumbent_artwork_id
        explored = set(run.exploration_artwork_ids)
        review = decision.review
        if review:
            if review.verdict in {'retain', 'accept', 'baseline'}:
                source = review.artwork_id
            elif review.verdict == 'explore':
                pending = next((s for s in run.steps if s.artwork_id == review.artwork_id), None)
                branch = self._exploration_branch(run, pending.exploration_root_artwork_id) if pending else None
                if branch is None or branch.remaining_jobs:
                    explored.add(review.artwork_id)
        if decision.source_artwork_id not in known or decision.source_artwork_id not in {source, *explored}:
            raise ValueError('交接只能使用已保留版本或有明确收益及剩余预算的探索分支；拒收候选不可续传')

    def _branch_snapshot(self, run):
        return [branch.model_dump(mode='json') | {'remaining_jobs': branch.remaining_jobs,
                'source_artwork_ids': [id for id in branch.artwork_ids if id in run.exploration_artwork_ids]}
                for branch in run.exploration_branches]

    @staticmethod
    def _tool_capabilities():
        # Execution support only. Run authorization/source/review/budget gates
        # still apply, and pixel protection is not an image-quality verdict.
        return {'available_operations': ['generate', 'handoff', 'local_edit', 'rollback', 'finish', 'pause'],
                'local_edit_capability': {'targets': ['newbie'], 'source_sha256_required': True,
                    'outside_mask_protection': True, 'quality_acceptance_automatic': False}}

    def _context(self, run):
        self._verify_original(run)
        artifacts = self._artifacts(run)
        return {
            **self._tool_capabilities(),
            'run_id': run.id, 'revision': run.revision, 'decision_count': len(run.decisions),
            'request': run.base_request, 'observation_capability': self.observation_capability(),
            'original_text': run.original_text, 'original_sha256': run.original_sha256,
            'task_goals': (run.base_request.get('creation_intent') or {}).get('goals', []),
            'allowed_models': run.allowed_models,
            'model_parameters': {m: target_request(GenerateRequest.model_validate(run.base_request), m).model_dump(mode='json') for m in run.allowed_models},
            'artifacts': [asdict(a) for a in artifacts],
            'artifact_inputs': {a.id: self.jobs.get(a.job_id).request for a in artifacts
                                if self.jobs.get(a.job_id) is not None},
            'feedback_signature': self.feedback_signature(run),
            'steps': [s.model_dump(mode='json') for s in run.steps],
            'decisions': [d.model_dump(mode='json') for d in run.decisions],
            'incumbent_artwork_id': run.incumbent_artwork_id,
            'candidate_reviews': [r.model_dump(mode='json') for r in run.candidate_reviews],
            'source_reinterpretation_history': [{
                'receipt': entry['receipt'], 'step_count': entry['step_count'],
                'scope': entry['scope'], 'complete_history_available_via_source_reinterpretation_endpoint': True,
            } for entry in run.source_revision_history],
            'exploration_artwork_ids': run.exploration_artwork_ids,
            'exploration_branches': self._branch_snapshot(run),
            'concept_entries': {
                'project': 'F:/Models2/concept/concepts/ai-comic/CONCEPT.md',
                'pipeline': 'F:/Models2/concept/concepts/ai-comic/pipeline/CONCEPT.md',
                'use': 'F:/Models2/concept/concepts/ai-comic/SOURCES_AND_USE.md',
                'summary_is_not_a_substitute_for_originals': True,
            },
            'comparison_policy': {
                'latest_is_not_best': True, 'compare_to_incumbent_and_root': True,
                'unknown_is_not_pass': True, 'pixel_metrics_are_not_quality_scores': True,
                'require_review_before_next_operation': True,
                'handoff_only_from_retained_or_bounded_exploration': True,
                'exploration_is_not_deliverable': True,
            },
            'remaining_jobs': run.budget.max_jobs - len(run.steps),
            'remaining_decisions': run.budget.max_decisions - run.decision_attempts,
            'remaining_seconds': max(0, run.budget.max_active_seconds - run.active_seconds),
        }

    def _finish(self, run, status, reason):
        self._account(run)
        run.status, run.stop_reason, run.active_since = status, reason, None
        self.store.save(run)

    def _stage_candidate_review(self, run, decision, known):
        """Apply a review to a detached run, without persistence or clock changes."""
        reviewed = {r.artwork_id for r in run.candidate_reviews}
        first_current_step = run.source_revision_history[-1]['step_count'] if run.source_revision_history else 0
        pending = [s.artwork_id for s in run.steps[first_current_step:] if s.artwork_id and s.artwork_id not in reviewed]
        review = decision.review
        if (not pending and review is not None and run.source_revision_history
                and review.artwork_id in known and review.artwork_id not in reviewed):
            # Historical pixels may be re-entered with a current explicit
            # review. Old reviews remain archived, not silently valid again.
            pending = [review.artwork_id]
        if not pending:
            if review is not None:
                raise ValueError('没有待审候选，不能重复或改写既有比较记录')
            return ''
        if len(pending) != 1 or review is None or review.artwork_id != pending[0] or review.artwork_id not in known:
            raise ValueError('必须先审查新候选，再选择下一步；不能跳过旧结果直接继续重绘')
        if review.baseline_artwork_id != run.incumbent_artwork_id:
            raise ValueError('候选必须与保留版本比较，不能自动把最后一张当作最好版本')
        if not run.incumbent_artwork_id:
            if review.verdict != 'baseline':
                raise ValueError('第一张图只能建立待改进基线，不能伪造前后改善')
            run.incumbent_artwork_id = review.artwork_id
        elif review.verdict == 'baseline':
            raise ValueError('已有保留版本，不能用重建基线绕过比较')
        elif review.verdict in {'accept', 'retain'}:
            run.incumbent_artwork_id = review.artwork_id
            run.consecutive_rejections = 0
        elif review.verdict == 'explore':
            step = next((s for s in run.steps[first_current_step:] if s.artwork_id == review.artwork_id), None)
            branch = self._exploration_branch(run, step.exploration_root_artwork_id) if step else None
            if branch is None:
                branch = ExplorationBranch(root_artwork_id=review.artwork_id,
                    artwork_ids=[review.artwork_id], max_followup_jobs=review.exploration.max_followup_jobs,
                    origin_incumbent_artwork_id=run.incumbent_artwork_id)
                run.exploration_branches.append(branch)
            else:
                branch.max_followup_jobs = min(branch.max_followup_jobs,
                    branch.followup_jobs + review.exploration.max_followup_jobs)
                if review.artwork_id not in branch.artwork_ids:
                    branch.artwork_ids.append(review.artwork_id)
            run.consecutive_rejections = 0
        elif review.verdict == 'reject':
            run.consecutive_rejections += 1
        run.candidate_reviews.append(review)
        self._refresh_exploration_sources(run)
        run.events.append({'kind': 'candidate_review', **review.model_dump(mode='json'),
                           'retained_artwork_id': run.incumbent_artwork_id})
        if review.verdict == 'uncertain':
            return 'candidate_comparison_uncertain'
        if run.consecutive_rejections >= run.budget.max_rejected_candidates:
            return 'no_demonstrated_improvement'
        return ''

    def _commit_candidate_review(self, run, reviewed_run):
        if len(reviewed_run.candidate_reviews) == len(run.candidate_reviews):
            return
        # Only review-owned fields are adopted. Decision accounting, controls,
        # submissions and their receipts remain owned by the live executor.
        run.incumbent_artwork_id = reviewed_run.incumbent_artwork_id
        run.candidate_reviews = reviewed_run.candidate_reviews
        run.exploration_artwork_ids = reviewed_run.exploration_artwork_ids
        run.exploration_branches = reviewed_run.exploration_branches
        run.consecutive_rejections = reviewed_run.consecutive_rejections
        run.events.append(reviewed_run.events[-1])
        self.store.save(run)

    def _review_candidate(self, run, decision, known):
        """Commit a standalone review; decision execution uses full preflight."""
        reviewed = run.model_copy(deep=True)
        stop_reason = self._stage_candidate_review(reviewed, decision, known)
        self._commit_candidate_review(run, reviewed)
        if stop_reason:
            self.pause(run, stop_reason)
            return False
        return True

    def preflight_decision(self, run, decision, known=None, *, queued_input_fingerprint=None):
        """Validate review AND its next operation without adopting either.

        Read-only file/format checks may observe terminal artifacts, but never
        persist a candidate review, spend exploration allowance, materialize
        inputs, change the live clock/control state or submit a job.
        """
        self._verify_original(run)
        if run.base_request.get('creation_intent'):
            if decision.original_text != run.original_text or decision.original_sha256 != run.original_sha256:
                raise ValueError('决策没有携带最初原句及其校验值')
            goals = {g['id'] for g in run.base_request['creation_intent']['goals']}
            if decision.operation in {'generate', 'handoff', 'local_edit'} and (not decision.goal_ids or not set(decision.goal_ids) <= goals):
                raise ValueError('每次生成或交接必须关联本次任务中的目标')
        if known is None:
            known = {a.id for a in self._artifacts(run)}
        if any(o.artwork_id not in known for o in decision.observations):
            raise ValueError('检查引用了不属于本次任务的作品')
        reviewed = run.model_copy(deep=True)
        stop_reason = self._stage_candidate_review(reviewed, decision, known)
        if decision.operation == 'rollback':
            retained = {r.artwork_id for r in reviewed.candidate_reviews if r.verdict in {'baseline', 'accept', 'retain'}}
            if decision.selected_artwork_id not in known & retained or decision.selected_artwork_id == reviewed.incumbent_artwork_id:
                raise ValueError('只能回退到此前保留过的不同版本，不能复活拒收候选')
        elif decision.operation == 'pause':
            if decision.selected_artwork_id != reviewed.incumbent_artwork_id:
                raise ValueError('暂停观察只能保留当前已审查版本')
        elif decision.operation == 'finish':
            if decision.selected_artwork_id not in known or decision.selected_artwork_id != reviewed.incumbent_artwork_id:
                raise ValueError('只能结束于已审查的保留版本，不能选择被拒收的候选')
            if reviewed.base_request.get('creation_intent'):
                review = self._last_review(reviewed, reviewed.incumbent_artwork_id)
                if review is None or review.verdict != 'accept' or review.unresolved_requirements:
                    raise ValueError('中间版本尚未完成任务；可以继续加工或暂停请用户查看')
        else:
            if decision.target not in reviewed.allowed_models:
                raise ValueError('决策选择了用户没有启用的模型')
            if len(reviewed.steps) >= reviewed.budget.max_jobs:
                raise ValueError('本任务的生图预算已用完；请先查看已有结果，不能继续派发')
            if decision.operation in {'handoff', 'local_edit'}:
                self.validate_handoff_source(reviewed, decision, known)
            prepared = (self.prepare_local_edit(reviewed, decision, materialize=False)[0]
                        if decision.operation == 'local_edit' else self.prepare_decision_request(reviewed, decision))
            self._validate_resources(prepared)
            if decision.operation == 'handoff':
                self._prepare_handoff(reviewed, decision, prepared, materialize=False)
            elif decision.operation == 'generate':
                prepare_generation_request(prepared, self.jobs.settings)
            fingerprint = self._submission_fingerprint(decision, prepared)
            if decision.operation in {'handoff', 'local_edit'}:
                if (decision.operation == 'local_edit' and queued_input_fingerprint is not None
                        and queued_input_fingerprint != fingerprint):
                    raise ValueError('局部修改源图或实际输入在排队后已变化，请重新查看并提交新决策')
                for old in reviewed.steps:
                    if old.status != 'completed' or old.decision.operation != decision.operation:
                        continue
                    old_fingerprint = old.submission_fingerprint
                    if not old_fingerprint and (old_job := self.jobs.get(old.job_id)) is not None:
                        old_fingerprint = self._submission_fingerprint(old.decision,
                            GenerateRequest.model_validate(old_job.request))
                    if old_fingerprint == fingerprint:
                        raise ValueError('相同源图与相同实际输入已执行；必须实际改变方法，不能重复无效尝试')
            return DecisionPreflight(reviewed, stop_reason, fingerprint)
        return DecisionPreflight(reviewed, stop_reason)

    async def tick(self, run):
        lock = self._locks.setdefault(run.id, asyncio.Lock())
        if lock.locked():
            return
        async with lock:
            if run.status != 'running':
                return
            observed_revision = run.revision
            try:
                await self._tick(run)
            except Exception as exc:
                # One failed planner/tool must not kill all runs. No blind POST
                # retry: preserve the operation and its error for inspection.
                obsolete = run.revision != observed_revision
                run.events.append({'kind': 'discarded_error' if obsolete else 'controller_error',
                                   'message': str(exc)[:3000], 'observed_revision': observed_revision})
                if run.status == 'running' and not obsolete:
                    self.pause(run, 'controller_error')
                else:
                    self.store.save(run)

    async def _tick(self, run):
        self._account(run)
        pending = next((s for s in run.steps if s.status in {'reserved', 'submitted'}), None)
        if pending:
            if not pending.job_id:
                label = f'{run.id}__{pending.id}'
                matches = [j for j in self.jobs.list() if j.request.get('output_group') == f'iteration_runs/{run.id}' and j.request.get('output_label') == label]
                if len(matches) != 1:
                    self.pause(run, 'submission_outcome_unknown')
                    return
                pending.job_id, pending.status = matches[0].id, 'submitted'
                self.store.save(run)
            job = self.jobs.get(pending.job_id)
            if job is None:
                self.pause(run, 'job_record_missing')
                return
            if job.status not in {'completed', 'failed', 'cancelled'}:
                self.store.save(run)
                return  # observation only; never submit a duplicate
            if job.status == 'completed' and len(job.files) == len(job.images) == 1:
                pending.status, pending.artwork_id = 'completed', self.artworks.id_for(job.id, 0)
                self._artifacts(run)
            else:
                pending.status, pending.error = 'failed', job.error or '生成没有返回一张完整图片'
            self.store.save(run)
        if run.active_seconds >= run.budget.max_active_seconds:
            self._finish(run, 'exhausted', 'time_budget')
            return
        if run.decision_attempts >= run.budget.max_decisions:
            self._finish(run, 'exhausted', 'decision_budget')
            return
        if hasattr(self.planner, 'has_decision') and not self.planner.has_decision(run.id):
            return
        revision = run.revision
        context = self._context(run)
        observed_feedback = context.get('feedback_signature')
        run.decision_attempts += 1
        self.store.save(run)
        # Planner timeout bounds observation, not the image job lifetime.
        timeout = getattr(self.planner, 'decision_timeout_seconds', 120)
        if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or not 0 < timeout <= 600:
            raise ValueError('规划等待上限必须在 0 到 600 秒之间')
        remaining = context.get('remaining_seconds', run.budget.max_active_seconds - run.active_seconds)
        if isinstance(remaining, bool) or not isinstance(remaining, (int, float)) or not 0 < remaining <= 28800:
            raise ValueError('规划必须有有效的剩余活动预算')
        timeout = min(timeout, remaining, run.budget.max_active_seconds - run.active_seconds)
        raw = await asyncio.wait_for(self.planner.decide(context), timeout=timeout)
        if run.status != 'running' or run.revision != revision:
            run.events.append({'kind': 'discarded_decision', 'reason': 'control_state_changed'})
            self.store.save(run)
            return
        queued_feedback = getattr(self.planner, 'last_feedback_signature', {}).get(run.id)
        if observed_feedback is not None and (
                self.feedback_signature(run) != observed_feedback or
                queued_feedback is not None and queued_feedback != observed_feedback):
            run.events.append({'kind': 'discarded_decision', 'reason': 'user_feedback_changed'})
            self.pause(run, 'user_feedback_changed')
            return
        self._account(run)
        if run.active_seconds >= run.budget.max_active_seconds:
            self._finish(run, 'exhausted', 'time_budget')
            return
        decision = IterationDecision.model_validate(raw)
        known = {a.id for a in self._artifacts(run)}
        try:
            self.validate_decision_observation(run, decision,
                queued_observation_id=getattr(self.planner, 'last_observation_id', {}).get(run.id))
        except (OSError, ValueError) as exc:
            run.events.append({'kind': 'discarded_decision', 'reason': 'observation_stale',
                               'observation_id': decision.observation_id, 'message': str(exc)[:3000]})
            self.pause(run, 'observation_stale')
            return
        checked = self.preflight_decision(run, decision, known,
            queued_input_fingerprint=getattr(self.planner, 'last_input_fingerprint', {}).get(run.id))
        self._commit_candidate_review(run, checked.reviewed_run)
        if checked.review_stop_reason:
            self.pause(run, checked.review_stop_reason)
            return
        if decision.operation == 'rollback':
            run.events.append({'kind': 'rollback', 'from': run.incumbent_artwork_id,
                               'to': decision.selected_artwork_id, 'reason': decision.reason})
            run.incumbent_artwork_id = decision.selected_artwork_id
            run.decisions.append(decision)
            self.store.save(run)
            return
        if decision.operation == 'pause':
            run.decisions.append(decision)
            self.pause(run, 'awaiting_user_review')
            return
        if decision.operation == 'finish':
            run.decisions.append(decision)
            run.selected_artwork_id = decision.selected_artwork_id
            run.selection_authority = 'machine'
            self._finish(run, 'completed', 'machine_selected_not_user_accepted')
            return
        # Deterministic proposal/format failures must not reserve a job or spend
        # a branch allowance as though a remote submission had taken place.
        run.decisions.append(decision)
        step = IterationStep(id=f'step_{len(run.steps)+1}', decision=decision,
                             submission_fingerprint=checked.submission_fingerprint)
        if decision.operation in {'handoff', 'local_edit'} and decision.source_artwork_id in run.exploration_artwork_ids:
            branch = self._exploration_branch(run, decision.source_artwork_id)
            step.exploration_root_artwork_id = branch.root_artwork_id
            branch.followup_jobs += 1
            self._refresh_exploration_sources(run)
        # Reserve durably BEFORE submitting. A crash here requires reconciliation,
        # not assuming the absence of a response means no job was created.
        run.steps.append(step)
        self.store.save(run)
        try:
            job = await self._submit(run, step)
        except Exception as exc:
            # Submission may have succeeded remotely/in memory. Leave reserved
            # so reconciliation, not automatic retry, decides what happened.
            step.error = str(exc)[:3000]
            self.store.save(run)
            raise
        step.job_id, step.status = job.id, 'submitted'
        self.store.save(run)
        if run.status == 'cancelled':
            await self.jobs.cancel(job.id)

    async def _submit(self, run, step):
        self._verify_original(run)
        decision = step.decision
        request = self.prepare_decision_request(run, decision)
        interpretation_status = ('local_edit_unverified_delta' if decision.operation == 'local_edit' else
                                 'recompiled_current_proposal' if decision.semantic_spec is not None else
                                 'native_override_unverified' if decision.prompt is not None else
                                 'native_negative_override_unverified' if decision.negative_prompt is not None else
                                 'inherited_current_input')
        update = {'output_group': f'iteration_runs/{run.id}', 'output_label': f'{run.id}__{step.id}',
                  'creation_intent': request.creation_intent, 'semantic_spec': request.semantic_spec,
                  'author_proposal': request.author_proposal,
                  'scene_contract': request.scene_contract,
                  'iteration_context': {'run_id': run.id, 'original_text': run.original_text,
                      'original_sha256': run.original_sha256, 'goal_ids': decision.goal_ids,
                      'observation_id': decision.observation_id,
                      'objective': decision.objective, 'preserve': decision.preserve,
                      'acceptance': decision.acceptance, 'reason': decision.reason,
                      'interpretation_status': interpretation_status,
                      'exploration_root_artwork_id': step.exploration_root_artwork_id,
                      'replace_selected_directions': decision.replace_selected_directions,
                      'local_edit': decision.region_edit.model_dump(mode='json', exclude={'mask_data_url'}) if decision.region_edit else None,
                      'research_refs': request.creation_intent.research_refs if request.creation_intent else []}}
        if decision.operation == 'generate':
            request = GenerateRequest.model_validate(request.model_dump(mode='json') | update)
            self._validate_resources(request)
            return await self.jobs.submit(request)
        if decision.operation == 'local_edit':
            edited, workflow, inputs = self.prepare_local_edit(run, decision, materialize=True)
            try:
                if self._submission_fingerprint(decision, edited) != step.submission_fingerprint:
                    raise ValueError('局部修改输入在预检后变化，未派发；请重新查看原图与遮罩')
                edited = GenerateRequest.model_validate(edited.model_dump(mode='json') | update)
                return await self.jobs.submit_workflow(edited, workflow)
            except Exception:
                # Keep protected inputs if submission actually succeeded but its
                # response was lost. The worker/reconciliation owns them then.
                if not any(job.request.get('output_label') == update['output_label'] for job in self.jobs.list()):
                    for path in inputs:
                        path.unlink(missing_ok=True)
                raise
        handoff, workflow = self._prepare_handoff(run, decision, request)
        handoff = GenerateRequest.model_validate(handoff.model_dump(mode='json') | update | {'native_prompts': request.native_prompts})
        try:
            return await self.jobs.submit_workflow(handoff, workflow)
        except Exception:
            # If the queue accepted the request, its worker owns this input.
            if not any(j.request.get('output_label') == update['output_label'] for j in self.jobs.list()):
                cleanup_handoff_input(handoff, self.jobs.settings)
            raise

    def _prepare_handoff(self, run, decision, request, *, materialize=True):
        artwork = self.artworks.get(decision.source_artwork_id)
        parent = self.jobs.get(artwork.job_id) if artwork is not None else None
        if artwork is None or parent is None:
            raise ValueError('交接源作品或其生成记录不存在')
        # Pixels belong to the selected branch; conditions to this proposal.
        # Use the identical read-only preparation in API and executor checks.
        options = {key: getattr(request, key) for key in ('steps', 'cfg', 'sampler', 'scheduler', 'shift', 'seed', 'cfg_rescale', 'weight_dtype', 'lora_name', 'lora_strength', 'style_preset')}
        options.update(target=decision.target, prompt=request.prompt,
                       negative_prompt=request.negative_prompt, denoise=decision.denoise)
        return prepare_handoff(artwork, parent, options, self.jobs.settings,
                               context_request=request, materialize=materialize)

    @staticmethod
    def _submission_fingerprint(decision, request):
        keys = ('prompt', 'negative_prompt', 'steps', 'cfg', 'sampler', 'scheduler',
                'shift', 'seed', 'cfg_rescale', 'weight_dtype', 'lora_name', 'lora_strength', 'style_preset')
        inputs = {key: getattr(request, key) for key in keys}
        inputs.update(operation=decision.operation, target=decision.target,
                      source_artwork_id=decision.source_artwork_id, denoise=decision.denoise)
        if decision.operation == 'local_edit':
            inputs.update(repair_source_sha256=request.repair_source_sha256,
                          repair_mask_sha256=request.repair_mask_sha256,
                          repair_instruction=request.repair_instruction,
                          region_repair_steps=request.region_repair_steps,
                          region_repair_denoise=request.region_repair_denoise)
        if decision.operation != 'handoff':
            inputs.update(width=request.width, height=request.height)
        return hashlib.sha256(json.dumps(inputs, ensure_ascii=False, sort_keys=True).encode('utf-8')).hexdigest()

    def prepare_local_edit(self, run, decision, *, materialize=False):
        current = self.prepare_decision_request(run, decision)
        artwork = self.artworks.get(decision.source_artwork_id)
        parent = generation_ancestor(artwork, self.artworks, self.jobs)
        if artwork is None or parent is None:
            raise ValueError('找不到局部修改源作品的真实像素或生成参数')
        request, workflow, inputs = prepare_region_edit(artwork, parent,
            decision.region_edit.model_dump(mode='json'), self.jobs.settings,
            context_request=current, materialize=materialize)
        if decision.seed is not None:
            request = request.model_copy(update={'seed': decision.seed})
            workflow['12']['inputs']['seed'] = decision.seed
        return request, workflow, inputs

    def prepare_decision_request(self, run, decision):
        """Synchronize the proposal and actual model inputs before dispatch.

        This is source/format protection, not proof of semantic understanding.
        An unstructured native override remains supported, but cannot carry an
        old layout contract as if it described the newly submitted text.
        """
        self._verify_original(run)
        base = GenerateRequest.model_validate(run.base_request)
        content = base
        content_artwork_id = (decision.source_artwork_id if decision.operation in {'handoff', 'local_edit'}
                              else decision.review.artwork_id if decision.review and
                                   decision.review.verdict in {'baseline', 'retain', 'accept'}
                              else run.incumbent_artwork_id)
        if content_artwork_id:
            artwork = self.artworks.get(content_artwork_id)
            job = (generation_ancestor(artwork, self.artworks, self.jobs) if decision.operation == 'local_edit'
                   else self.jobs.get(artwork.job_id) if artwork else None)
            if job and (job.request.get('creation_intent') or {}).get('original_sha256') == run.original_sha256:
                content = GenerateRequest.model_validate(job.request)
        if content is not base:
            if run.source_revision_history:
                from .source_revision_control import source_scope
                if source_scope(content) != source_scope(base):
                    if decision.operation in {'handoff', 'local_edit'} and (
                            decision.semantic_spec is None or decision.author_proposal is None):
                        raise ValueError('旧图像属于另一SOURCE或目标解释；须提交完整当前方案后才可借用像素，未复制旧条件')
                    content = base
            for value in ((content.scene_contract or {}).get('source_text'),
                          (content.semantic_spec or {}).get('source_sentence')):
                if value is not None and value != run.original_text:
                    if decision.semantic_spec is not None or decision.prompt is not None:
                        # Keep the historical pixels/history untouched. Only a
                        # newly explicit proposal may replace stale explanation.
                        content = base
                        break
                    raise ValueError('分支解释没有保留任务原句；请明确重建当前方案，不能冒充已同步的输入')
        working = base.model_copy(update={key: getattr(content, key) for key in
                                          ('semantic_spec', 'scene_contract', 'native_prompts', 'author_proposal')})
        if decision.semantic_spec is not None:
            from .contract_builder import contract_from_semantic, rebuild_current_contract
            from .contract_compilers import PromptTarget, compile_contract
            from .creation_policy import new_creation_negative
            from .newbie_compiler import build_negative
            from .scene_contract import SceneContractV3
            from .model_conditioning import NEGATIVE_BASELINES, merge_negative
            spec = decision.semantic_spec
            if spec.source_sentence != run.original_text:
                raise ValueError('当前表现方案必须保留任务原句，不能用方案摘要替换它')
            previous = working.scene_contract or base.scene_contract
            from .source_continuity import (current_source, validate_source_continuity,
                attribute_source_continuity, preserve_source_metadata, validate_source_author_conditions)
            source = current_source(working.author_proposal, run.original_text, run.original_sha256)
            source_receipt = None
            if source is not None:
                validate_source_continuity(source, spec)
            if previous is None:
                if decision.replace_selected_directions:
                    raise ValueError('没有已确认的表现方案可被替换')
                contract = contract_from_semantic(spec, compiler='iteration_current_proposal',
                    strict_entity_refs=source is not None)
            else:
                contract = rebuild_current_contract(SceneContractV3.model_validate(previous), spec,
                    compiler='iteration_current_proposal',
                    replaced_refs=set(decision.replace_selected_directions),
                    strict_entity_refs=source is not None)
            if source is not None:
                contract, source_receipt = attribute_source_continuity(source, contract)
            native = {}
            for target in PromptTarget:
                compiled = compile_contract(contract, target)
                old_channel = (working.native_prompts or {}).get(target.value) or {}
                old_negative_policy = old_channel.get('negative_policy') or {}
                if not isinstance(old_negative_policy, dict):
                    old_negative_policy = {}
                explicitly_empty = (old_channel.get('negative') == '' and
                    (old_negative_policy.get('scope') != 'model_default_and_current_constraints'
                     or old_channel.get('trace_status') in {
                         'unverified_manual_override', 'unverified_legacy_alias_override',
                         'unverified_iteration_override',
                     }
                     or old_channel.get('negative_policy_status') == 'compiled_policy_not_reapplied_after_prompt_edit'))
                if target == PromptTarget.NEWBIE:
                    baseline = build_negative(spec.model_copy(update={'exclusions': []}))
                    compiled.negative, policy = new_creation_negative(spec, baseline, compiled.negative,
                        compiled.positive, contract_negative_terms=compiled.negative_terms)
                else:
                    baseline_key = 'z_image_turbo' if target == PromptTarget.Z_IMAGE else target.value
                    compiled.negative = merge_negative(baseline_key, NEGATIVE_BASELINES[baseline_key], compiled.negative)
                    policy = {'scope': 'model_default_and_current_constraints',
                              'semantic_correctness_verified': False}
                if explicitly_empty:
                    compiled.negative = ''
                    policy = {'scope': 'explicit_empty_preserved', 'semantic_correctness_verified': False}
                compiled.require_complete(contract)
                native[target.value] = {'positive': compiled.positive, 'negative': compiled.negative,
                    'trace': compiled.trace, 'postprocess': compiled.postprocess,
                    'trace_status': 'compiled_payload_presence_only'}
                native[target.value]['negative_policy'] = policy
            working = working.model_copy(update={'semantic_spec': spec.model_dump(mode='json'),
                'scene_contract': contract.model_dump(mode='json'), 'native_prompts': native})
            if working.author_proposal is not None or decision.author_proposal is not None:
                if decision.author_proposal is None:
                    raise ValueError('当前分支含作者场景提案，改规格时必须一并重建场景/位置，不能悄悄退回旧泛化词')
                from .art_direction import AuthorDraft, compile_author_draft
                from .schemas import NewbieCompileRequest
                old = working.author_proposal or {}
                data = dict(decision.author_proposal)
                data['spec'] = spec.model_dump(mode='json')
                draft = AuthorDraft.model_validate({key: data[key] for key in AuthorDraft.model_fields if key in data})
                if source is not None:
                    validate_source_author_conditions(source, draft, spec)
                rebuilt = compile_author_draft(draft, NewbieCompileRequest(sentence=run.original_text, style_mode=spec.style_mode),
                    proposal_id=old.get('id', run.id), revision=int(old.get('revision', 0)) + 1,
                    knowledge=old.get('knowledge'), recipes=old.get('recipe_snapshot'),
                    runtime_evidence={'previous': old.get('runtime_evidence', {}), 'decision_reason': decision.reason},
                    apply_selected_knowledge=False, config=working.model_dump(mode='json'), contract_override=contract,
                    preserve_style_conditions=source is not None)
                if source is not None:
                    preserve_source_metadata(rebuilt['author_proposal'], old, source, source_receipt)
                for key, channel_data in rebuilt['native_prompts'].items():
                    if native[key]['negative'] == '':
                        channel_data['negative'] = ''
                        channel_data['negative_policy'] = native[key]['negative_policy']
                working = working.model_copy(update={'author_proposal': rebuilt['author_proposal'],
                    'semantic_spec': rebuilt['spec'], 'scene_contract': rebuilt['contract_v3'],
                    'native_prompts': rebuilt['native_prompts']})
        channel = 'z_image' if decision.target == 'z_image_turbo' else decision.target
        native = dict(working.native_prompts or {})
        target_native = dict(native.get(channel) or {})
        if content is not base and decision.semantic_spec is None and decision.prompt is None and not target_native.get('positive'):
            raise ValueError('该分支没有当前目标模型的原生文本；请重建当前方案或提供明确原生词，不能退回首版解释')
        if decision.prompt is not None or decision.negative_prompt is not None:
            target_native.update(compiled_trace=target_native.get('trace', []), trace=[], trace_status='unverified_iteration_override')
            if working.author_proposal:
                working = working.model_copy(update={'author_proposal': {**working.author_proposal,
                    'condition_status': 'historical_after_native_override'}})
        if decision.prompt is not None:
            target_native['positive'] = decision.prompt
            if 'negative' not in target_native:
                target_native['negative'] = ((base.native_prompts or {}).get(channel) or {}).get('negative', '')
            native = {}  # Other channels no longer describe this new proposal.
            working = working.model_copy(update={'semantic_spec': None, 'scene_contract': None})
        if decision.negative_prompt is not None:
            target_native['negative'] = decision.negative_prompt
            # An edited negative no longer represents all of the old exclusions.
            # Original intent survives, but the old full contract is not certified.
            working = working.model_copy(update={'semantic_spec': None, 'scene_contract': None})
        native[channel] = target_native
        working = working.model_copy(update={'native_prompts': native})
        request = target_request(working, decision.target)
        if decision.operation == 'local_edit' and decision.semantic_spec is None and decision.prompt is None:
            # Separate the last repair operation from its underlying scene.
            # Repainting a region must not recursively wrap all prior edit
            # instructions as if they were immutable global scene conditions.
            request = request.model_copy(update={'repair_context_prompt': content.repair_context_prompt,
                'repair_context_policy': content.repair_context_policy})
        return request.model_copy(update={'seed': request.seed if decision.seed is None else decision.seed})

    def snapshot(self, run):
        # Observation may attach terminal job results while dispatch is paused.
        # Capture run fields only after that synchronization, so the first GET
        # cannot show finished jobs beside stale submitted steps.
        artifacts = self._artifacts(run)
        artwork_values = []
        for artwork in artifacts:
            try:
                sha256 = self.artwork_sha256(artwork)
            except (OSError, ValueError):
                sha256 = ''
            artwork_values.append(asdict(artwork) | {'artwork_sha256': sha256})
        return run.model_dump(mode='json') | {
            **self._tool_capabilities(),
            'observation_capability': self.observation_capability(),
            'planning': self.planner.status(run.id) if hasattr(self.planner, 'status') else {'stage': 'manual_review'},
            'planning_mode_capability': {'available': True, 'requires_paused': True,
                'automatic_available': getattr(self.planner, 'controller', None) is self and self.observations is not None,
                'mode_change_resumes_run': False, 'quality_certification': False},
            'artworks': artwork_values,
            'user_selection_capability': {'enabled': True, 'requires_artwork_sha256': True,
                                         'quality_certification': False},
            'graph': self.graph(run),
            'exploration_branches': self._branch_snapshot(run),
            'feedback_signature': self.feedback_signature(run),
            'quality_verdict': '机器选择不等于用户认可；耗尽预算不代表质量达标。',
            'waiting_for_decision': run.status == 'running' and not any(s.status in {'reserved', 'submitted'} for s in run.steps),
            'jobs': [asdict(job) for s in run.steps if s.job_id and (job := self.jobs.get(s.job_id))],
        }

    @staticmethod
    def graph(run):
        nodes = [{'id': 'intent', 'label': '原始意图与约束', 'type': 'contract_gate'}]
        edges = []
        producers = {s.artwork_id: s.id for s in run.steps if s.artwork_id}
        for i, id in enumerate(run.seed_artwork_ids):
            node_id = f'seed_{i}'
            producers[id] = node_id
            nodes.append({'id': node_id, 'label': f'已有候选 {id}', 'type': 'checkpoint'})
            edges.append({'source': 'intent', 'target': node_id, 'condition': 'existing evidence'})
        previous = 'intent'
        for step in run.steps:
            decision_id = f'decide_{step.id}'
            nodes.extend([
                {'id': decision_id, 'label': '检查 / 决策', 'type': 'route', 'config': step.decision.model_dump(mode='json')},
                {'id': step.id, 'label': f'{step.decision.operation} · {step.decision.target}', 'type': 'generate',
                 'config': {'status': step.status, 'job_id': step.job_id, 'artwork_id': step.artwork_id}},
            ])
            edges.extend([{'source': previous, 'target': decision_id, 'condition': 'observed result'},
                          {'source': decision_id, 'target': step.id, 'condition': 'chosen operation'}])
            if step.decision.source_artwork_id:
                edges.append({'source': producers[step.decision.source_artwork_id], 'target': step.id, 'condition': 'pixel input'})
            previous = step.id
        nodes.append({'id': 'result', 'label': run.status, 'type': 'accept', 'config': {'selected_artwork_id': run.selected_artwork_id,
            'incumbent_artwork_id': run.incumbent_artwork_id, 'reason': run.stop_reason,
            'candidate_reviews': [r.model_dump(mode='json') for r in run.candidate_reviews]}})
        edges.append({'source': previous, 'target': 'result', 'condition': 'execution state'})
        if run.selected_artwork_id:
            edges.append({'source': producers[run.selected_artwork_id], 'target': 'result', 'condition': 'selected version'})
        return {'nodes': nodes, 'edges': edges}
