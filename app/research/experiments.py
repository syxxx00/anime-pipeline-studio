from __future__ import annotations

import asyncio
from copy import deepcopy
import json
import logging
import re
import uuid
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from pydantic import BaseModel

from ..comfy_client import ComfyClient
from ..config import PROJECT_ROOT, Settings
from ..jobs import JobManager
from ..reviews import analyze_review
from ..schemas import ExperimentCreateRequest, GenerateRequest
from ..creative_art_support import ResearchArtSupport, freeze_research_support, validate_research_support
from ..creative_continuity_context import _digest
from ..research_pixel_support import ResearchPixelUse, prepare_research_pixel_handoff


logger = logging.getLogger(__name__)

FINAL_JOB_STATES = {"completed", "failed", "cancelled"}
FINAL_ITEM_STATES = {"completed", "failed", "cancelled"}


class ResearchPixelSource(BaseModel):
    model_config = {'extra': 'forbid', 'strict': True, 'revalidate_instances': 'always'}
    selection: ResearchArtSupport
    use: ResearchPixelUse


class ResearchPixelDispatchPaused(InterruptedError):
    """The new explicit consumer was stopped before queue submission."""

_SAFE_SEGMENT = re.compile(r"[^0-9A-Za-z一-鿿._-]+")


def safe_segment(text: str) -> str:
    """路径段清洗：角色名可能带括号等字符（如 w_(arknights)），output_group 校验不允许。"""
    return _SAFE_SEGMENT.sub("_", str(text)).strip("_") or "unknown"


def now_iso() -> str:
    return datetime.now(UTC).isoformat()


@dataclass(slots=True)
class ExperimentItem:
    id: str
    axes: dict
    request: dict
    status: str = "pending"
    job_id: str | None = None
    images: list[str] = field(default_factory=list)
    files: list[str] = field(default_factory=list)
    timings: dict[str, float] = field(default_factory=dict)
    checks: list[dict] = field(default_factory=list)
    error: str | None = None
    cancel_requested: bool = False
    reviews: list[dict] = field(default_factory=list)
    key: str | None = None
    pixel_source: dict | None = None
    research_pixel_source_snapshot: dict | None = None
    research_pixel_dispatch: dict | None = None


@dataclass(slots=True)
class Experiment:
    id: str
    name: str
    state: str
    created_at: str
    updated_at: str
    items: list[ExperimentItem]
    notes: str = ""
    external_llm_used: bool = False
    active_item_id: str | None = None
    design: dict = field(default_factory=dict)


class ExperimentManager:
    """Durable, resumable dispatcher layered on top of the normal job queue."""

    def __init__(
        self,
        settings: Settings,
        jobs: JobManager,
        client: ComfyClient,
        root: Path | None = None,
        *,
        artworks=None,
    ) -> None:
        self.settings = settings
        self.jobs = jobs
        self.client = client
        self.artworks = artworks
        self.root = root or PROJECT_ROOT / "data" / "experiments"
        self.root.mkdir(parents=True, exist_ok=True)
        self.experiments: dict[str, Experiment] = {}
        self.worker: asyncio.Task | None = None
        self._load()

    def start(self) -> None:
        self._reconcile_after_restart()
        if self.worker is None or self.worker.done():
            self.worker = asyncio.create_task(self._run(), name="experiment-dispatcher")

    async def stop(self) -> None:
        if self.worker:
            self.worker.cancel()
            try:
                await self.worker
            except asyncio.CancelledError:
                pass

    def create(self, payload: ExperimentCreateRequest, *, design: dict | None = None) -> dict:
        experiment_id = uuid.uuid4().hex[:12]
        stamp = now_iso()
        name = self._unique_name(payload.name)
        experiment = Experiment(
            id=experiment_id,
            name=name,
            state="paused",
            created_at=stamp,
            updated_at=stamp,
            notes=payload.notes,
            external_llm_used=payload.external_llm_used,
            design=design or {},
            items=[
                ExperimentItem(
                    id=f"{index:04d}-{uuid.uuid4().hex[:6]}",
                    axes=item.axes,
                    request=item.request.model_dump(mode="json"),
                    key=item.key,
                    pixel_source=item.pixel_source.model_dump() if item.pixel_source else None,
                )
                for index, item in enumerate(payload.items, start=1)
            ],
        )
        self._assign_output_paths(experiment)
        self.experiments[experiment_id] = experiment
        self._save(experiment)
        return self.serialize(experiment)

    def _unique_name(self, requested: str) -> str:
        existing = {item.name for item in self.experiments.values()}
        if requested not in existing:
            return requested
        index = 2
        while f"{requested}（{index}）" in existing:
            index += 1
        return f"{requested}（{index}）"

    def get(self, experiment_id: str) -> Experiment | None:
        return self.experiments.get(experiment_id)

    def list(self) -> list[dict]:
        ordered = sorted(self.experiments.values(), key=lambda item: item.created_at, reverse=True)
        return [self.serialize(item, include_items=False) for item in ordered]

    def list_reviews(self, experiment: Experiment) -> list[dict]:
        rows: list[dict] = []
        for item in experiment.items:
            for review in item.reviews:
                image_index = int(review.get("image_index", 0))
                rows.append({
                    "experiment_id": experiment.id,
                    "item_id": item.id,
                    "job_id": item.job_id,
                    "axes": item.axes,
                    "image": item.images[image_index] if image_index < len(item.images) else None,
                    **review,
                })
        return sorted(rows, key=lambda row: row.get("created_at", ""), reverse=True)

    def set_state(self, experiment: Experiment, state: str) -> dict:
        experiment.state = state
        experiment.updated_at = now_iso()
        self._save(experiment)
        return self.serialize(experiment)

    def retry_failed(self, experiment: Experiment) -> dict:
        failed = [item for item in experiment.items if item.status == "failed"]
        if not failed:
            raise ValueError("实验中没有失败项")
        if any(item.research_pixel_dispatch and item.research_pixel_dispatch.get('submission_started')
               for item in failed):
            raise ValueError('研究像素项已经尝试派发；已有或未知回执不通过重试失败项重新排队。')
        active = self._active_item(experiment)
        retained_active_id = active.id if active and active.status == 'running' else None
        for item in failed:
            item.status = "pending"
            item.job_id = None
            item.images = []
            item.files = []
            item.timings = {}
            item.checks = []
            item.error = None
            item.cancel_requested = False
        experiment.active_item_id = retained_active_id
        experiment.state = "running"
        experiment.updated_at = now_iso()
        self._save(experiment)
        return self.serialize(experiment)

    async def interrupt(self, experiment: Experiment) -> dict:
        experiment.state = "paused"
        active = self._active_item(experiment)
        if active:
            active.cancel_requested = True
            if active.research_pixel_dispatch and not active.job_id:
                pass  # No received task ID: never interrupt an unrelated engine job.
            elif hasattr(self.jobs, "cancel"):
                await self.jobs.cancel(active.job_id or "")
            else:  # compatibility with lightweight adapters
                await self.client.interrupt()
        experiment.updated_at = now_iso()
        self._save(experiment)
        return self.serialize(experiment)

    async def cancel(self, experiment: Experiment) -> dict:
        experiment.state = "cancelled"
        for item in experiment.items:
            if item.status == "pending":
                item.status = "cancelled"
        active = self._active_item(experiment)
        if active:
            active.cancel_requested = True
            if active.research_pixel_dispatch and not active.job_id:
                pass
            elif hasattr(self.jobs, "cancel"):
                await self.jobs.cancel(active.job_id or "")
            else:
                await self.client.interrupt()
        experiment.updated_at = now_iso()
        self._save(experiment)
        return self.serialize(experiment)

    async def cancel_item(self, experiment: Experiment, item_id: str) -> dict:
        item = next((value for value in experiment.items if value.id == item_id), None)
        if item is None:
            raise KeyError(item_id)
        if item.status == "pending":
            item.status = "cancelled"
        elif item.status == "running":
            item.cancel_requested = True
            experiment.state = "paused"
            if item.research_pixel_dispatch and not item.job_id:
                pass
            elif hasattr(self.jobs, "cancel"):
                await self.jobs.cancel(item.job_id or "")
            else:
                await self.client.interrupt()
        experiment.updated_at = now_iso()
        self._save(experiment)
        return self.serialize(experiment)

    def add_review(self, experiment: Experiment, item_id: str, text: str, image_index: int = 0) -> dict:
        item = next((value for value in experiment.items if value.id == item_id), None)
        if item is None:
            raise KeyError(item_id)
        if image_index >= max(1, len(item.images)):
            raise ValueError("评价所指向的图片不存在")
        review = {
            "id": uuid.uuid4().hex[:12],
            "text": text.strip(),
            "image_index": image_index,
            "created_at": now_iso(),
            "analysis": analyze_review(text),
        }
        item.reviews.append(review)
        experiment.updated_at = now_iso()
        self._save(experiment)
        return review

    def delete_review(self, experiment: Experiment, item_id: str, review_id: str) -> None:
        item = next((value for value in experiment.items if value.id == item_id), None)
        if item is None:
            raise KeyError(item_id)
        before = len(item.reviews)
        item.reviews = [review for review in item.reviews if review.get("id") != review_id]
        if len(item.reviews) == before:
            raise LookupError(review_id)
        experiment.updated_at = now_iso()
        self._save(experiment)

    async def _run(self) -> None:
        while True:
            for experiment in list(self.experiments.values()):
                try:
                    await self._tick(experiment)
                except Exception:
                    # 单个实验的异常不能弄死整个调度循环（曾有校验失败导致 worker 静默退出、实验假 running）
                    logger.exception("experiment tick failed: %s", experiment.id)
            await asyncio.sleep(0.75)

    async def _tick(self, experiment: Experiment) -> None:
        active = self._active_item(experiment)
        if active:
            if active.research_pixel_dispatch and active.research_pixel_dispatch.get('submission_started'):
                try:
                    job = self._research_pixel_job(active)
                except ValueError as exc:
                    active.status, active.error = 'failed', str(exc)
                    experiment.state, experiment.active_item_id = 'paused', None
                    self._save(experiment)
                    return
            else:
                job = self.jobs.get(active.job_id or "")
            if job and job.status in FINAL_JOB_STATES:
                active.images = list(job.images)
                active.files = list(job.files)
                active.timings = dict(job.timings)
                active.checks = list(job.checks)
                active.error = job.error
                active.status = "cancelled" if active.cancel_requested else job.status
                active.cancel_requested = False
                experiment.active_item_id = None
                experiment.updated_at = now_iso()
                self._finish_if_done(experiment)
                self._save(experiment)
            return
        if experiment.state != "running":
            return
        pending = next((item for item in experiment.items if item.status == "pending"), None)
        if pending is None:
            self._finish_if_done(experiment)
            self._save(experiment)
            return
        try:
            job = await self._submit_item(experiment, pending)
        except ResearchPixelDispatchPaused:
            pending.status = ('cancelled' if experiment.state == 'cancelled' or pending.cancel_requested
                              or pending.status == 'cancelled' else 'pending')
            pending.cancel_requested = False
            if experiment.active_item_id == pending.id:
                experiment.active_item_id = None
            experiment.updated_at = now_iso()
            self._save(experiment)
            return
        except Exception as exc:
            receipt = pending.research_pixel_dispatch
            if receipt and receipt.get('submission_started') and receipt.get('job_id'):
                # The queue returned a real task before saving its receipt
                # failed. Keep ownership and polling; a local persistence
                # failure is not evidence that this task finished or failed.
                try:
                    known = self._research_pixel_job(pending)
                except ValueError:
                    pass  # The ordinary closed failure path below still applies.
                else:
                    pending.job_id = known.id
                    pending.status = 'running'
                    pending.error = f'已知派发回执保存未完成，保留原任务追踪：{exc}'
                    experiment.active_item_id = pending.id
                    if experiment.state != 'cancelled':
                        experiment.state = 'paused'
                    if experiment.state == 'cancelled' or pending.cancel_requested:
                        try:
                            await self.jobs.cancel(known.id)
                        except Exception as cancel_error:
                            pending.error += f'；原任务取消未确认：{cancel_error}'
                    experiment.updated_at = now_iso()
                    self._save(experiment)
                    return
            # 提交校验失败要落为该项失败并继续下一项，否则同一项每 tick 重抛、实验卡死
            pending.status = "failed"
            pending.error = f"提交失败：{exc}"
            if experiment.active_item_id == pending.id:
                experiment.active_item_id = None
            experiment.updated_at = now_iso()
            self._finish_if_done(experiment)
            self._save(experiment)
            return
        pending.job_id = job.id
        if experiment.state == 'cancelled' or pending.cancel_requested or pending.status == 'cancelled':
            pending.status = 'cancelled'
            await self.jobs.cancel(job.id)
            self._save(experiment)
            return
        pending.status = "running"
        experiment.active_item_id = pending.id
        experiment.updated_at = now_iso()
        self._save(experiment)

    async def _submit_item(self, experiment: Experiment, item: ExperimentItem):
        # Receipts precede parsing: item.request may already be the resolved
        # image_handoff request after a pause, crash or late acknowledgement.
        receipt = item.research_pixel_dispatch
        if receipt and receipt.get('submission_started'):
            return self._research_pixel_job(item)
        context = item.request.get('iteration_context') or {}
        declared = isinstance(context, dict) and 'research_pixel_source' in context
        if declared:
            if item.pixel_source or item.request.get('image_handoff'):
                raise ValueError('研究原图声明不能与固定像素源或已有交接混用。')
            return await self._submit_research_pixel_item(experiment, item)
        request = GenerateRequest.model_validate(item.request)
        if not item.pixel_source:
            return await self.jobs.submit(request)
        from types import SimpleNamespace
        from ..image_handoff import cleanup_handoff_input, prepare_handoff
        source = next((x for x in experiment.items if x.key == item.pixel_source['source_key']), None)
        if source is None or source.status != 'completed' or len(source.files) != 1:
            raise ValueError('源实验项未成功返回唯一图片；此交接记失败，不重新生成源图')
        parent_job = self.jobs.get(source.job_id or '')
        if parent_job is None:
            raise ValueError('源生成记录缺失，不能猜测像素来源')
        artwork = SimpleNamespace(id=f'{source.job_id}-01', file_path=source.files[0])
        options = {key: getattr(request, key) for key in ('prompt', 'negative_prompt', 'steps', 'cfg', 'seed', 'sampler', 'scheduler', 'shift', 'cfg_rescale', 'weight_dtype', 'lora_name', 'lora_strength', 'style_preset')}
        options.update(target=request.render_pipeline.value, denoise=item.pixel_source['denoise'])
        resolved, workflow = prepare_handoff(artwork, parent_job, options, self.settings)
        resolved = GenerateRequest.model_validate(resolved.model_dump(mode='json') | {
            'output_group': request.output_group, 'output_label': request.output_label,
            'semantic_spec': request.semantic_spec, 'scene_contract': request.scene_contract, 'native_prompts': request.native_prompts,
        })
        try:
            job = await self.jobs.submit_workflow(resolved, workflow)
        except Exception:
            cleanup_handoff_input(resolved, self.settings)
            raise
        item.request = job.request  # actual source hash, target text and parameters
        # Cancellation during submission must not orphan an untracked job.
        if experiment.state == 'cancelled' or item.cancel_requested or item.status == 'cancelled':
            await self.jobs.cancel(job.id)
        return job

    def _research_pixel_job(self, item):
        receipt = item.research_pixel_dispatch
        identifier = receipt.get('job_id')
        if not identifier:
            raise ValueError('研究像素提交回执未知，不重新消费原图或派发。')
        job = self.jobs.get(identifier)
        if job is None:
            raise ValueError('研究像素已提交但原任务暂不可见，不重新派发。')
        try:
            if (item.job_id not in {None, identifier}
                    or _digest(receipt['current_request']) != receipt['current_request_sha256']
                    or _digest(receipt['actual_request']) != receipt['actual_request_sha256']
                    or _digest(job.request) != receipt['job_request_sha256']):
                raise ValueError('研究像素原回执的任务或条件变化，不采用其他任务。')
        except (KeyError, TypeError) as exc:
            raise ValueError('研究像素原回执缺少完整条件，不猜补或重派。') from exc
        return job

    @staticmethod
    def _check_research_pixel_control(experiment, item):
        if experiment.state != 'running' or item.cancel_requested or item.status == 'cancelled':
            raise ResearchPixelDispatchPaused('实验已暂停或停止；未派发研究像素候选。')

    async def _submit_research_pixel_item(self, experiment, item):
        if self.artworks is None:
            raise ValueError('研究像素消费未绑定真实作品库；不猜测来源。')
        self._check_research_pixel_control(experiment, item)
        original = deepcopy(item.request)
        original_sha = _digest(original)
        request = GenerateRequest.model_validate(original)
        declaration = ResearchPixelSource.model_validate(request.iteration_context['research_pixel_source'])
        snapshot = freeze_research_support(declaration.selection, self.artworks, self.jobs,
            output_root=Path(self.settings.output_dir))
        actual, workflow, raw = prepare_research_pixel_handoff(request, snapshot,
            declaration.use, self.settings, materialize=False)

        def revalidate():
            self._check_research_pixel_control(experiment, item)
            if _digest(item.request) != original_sha:
                raise ValueError('当前完整研究请求或像素声明在准备期间变化，未派发。')
            validate_research_support(snapshot, self.artworks, self.jobs,
                output_root=Path(self.settings.output_dir))

        revalidate()
        from ..image_handoff import verify_handoff_input
        input_root = (Path(self.settings.engine_base_dir) / 'input').resolve()
        path = Path(actual.image_handoff.source_file).resolve()
        if path.parent != input_root or not re.fullmatch(r'handoff_[0-9a-f]{32}_source\.png', path.name):
            raise ValueError('研究消费者没有提供本次独占输入，未创建或删除共享文件。')
        owned, submission_awaited = False, False
        previous = deepcopy(item.research_pixel_dispatch)
        receipt = None
        try:
            input_root.mkdir(parents=True, exist_ok=True)
            with path.open('xb') as stream:
                owned = True
                stream.write(raw)
            verify_handoff_input(actual, self.settings)
            revalidate()
            actual_request = actual.model_dump(mode='json')
            receipt = {'id': uuid.uuid4().hex[:12], 'method': 'explicit_research_whole_pixels',
                'status': 'reserved', 'submission_started': True, 'submission_uncertain': False,
                'job_id': None, 'current_request': original, 'current_request_sha256': original_sha,
                'actual_request': actual_request, 'actual_request_sha256': _digest(actual_request),
                'workflow_sha256': _digest(workflow), 'source_snapshot_sha256': snapshot['snapshot_sha256'],
                'source_artwork_id': declaration.selection.artwork_id,
                'source_sha256': declaration.selection.expected_image_sha256,
                'use': declaration.use.model_dump(mode='json'), 'quality_verified': False}
            if previous:
                receipt['previous_not_submitted'] = previous
            item.research_pixel_source_snapshot = deepcopy(snapshot)
            item.research_pixel_dispatch = receipt
            item.request = deepcopy(actual_request)
            item.status = 'running'
            experiment.active_item_id = item.id
            self._save(experiment)  # Durable BEFORE the potentially accepted queue await.
            self._check_research_pixel_control(experiment, item)
            submission_awaited = True
            job = await self.jobs.submit_workflow(actual, workflow)
        except BaseException:
            if submission_awaited:
                receipt.update(status='submission_uncertain', submission_uncertain=True)
                # The queue may already reference this input: do not remove it.
            else:
                if owned:
                    path.unlink(missing_ok=True)
                if receipt:
                    receipt.update(status='not_submitted', submission_started=False)
                    item.request = original
                    item.status = 'pending'
                    if experiment.active_item_id == item.id:
                        experiment.active_item_id = None
            try:
                self._save(experiment)
            except OSError:
                logger.exception('research pixel failure receipt could not be saved: %s/%s', experiment.id, item.id)
            raise
        receipt.update(status='submitted', job_id=job.id, job_request_sha256=_digest(job.request))
        item.job_id = job.id
        item.request = deepcopy(job.request)
        self._save(experiment)  # Actual receipt is saved before control/observation.
        return job

    def _active_item(self, experiment: Experiment) -> ExperimentItem | None:
        if not experiment.active_item_id:
            return None
        return next((item for item in experiment.items if item.id == experiment.active_item_id), None)

    def _finish_if_done(self, experiment: Experiment) -> None:
        if all(item.status in FINAL_ITEM_STATES for item in experiment.items):
            if experiment.state != "cancelled":
                experiment.state = "completed"

    def serialize(self, experiment: Experiment, include_items: bool = True) -> dict:
        counts = {name: 0 for name in ("pending", "running", "completed", "failed", "cancelled")}
        for item in experiment.items:
            counts[item.status] = counts.get(item.status, 0) + 1
        total = len(experiment.items)
        finished = sum(counts[name] for name in FINAL_ITEM_STATES)
        model_averages: dict[str, list[float]] = {}
        for item in experiment.items:
            elapsed = item.timings.get("total")
            if elapsed and item.status == "completed":
                model_averages.setdefault(str(item.request.get("model_id", "unknown")), []).append(elapsed)
        global_values = [value for values in model_averages.values() for value in values]
        fallback = sum(global_values) / len(global_values) if global_values else None
        eta = 0.0
        unknown = False
        for item in experiment.items:
            if item.status not in {"pending", "running"}:
                continue
            values = model_averages.get(str(item.request.get("model_id", "unknown")), [])
            estimate = sum(values) / len(values) if values else fallback
            if estimate is None:
                unknown = True
            else:
                eta += estimate
        result = {
            "id": experiment.id,
            "name": experiment.name,
            "state": experiment.state,
            "created_at": experiment.created_at,
            "updated_at": experiment.updated_at,
            "notes": experiment.notes,
            "external_llm_used": experiment.external_llm_used,
            "active_item_id": experiment.active_item_id,
            "counts": counts,
            "total": total,
            "finished": finished,
            "progress": finished / total if total else 0,
            "eta_seconds": None if unknown and not global_values else round(eta, 1),
            "review_count": sum(len(item.reviews) for item in experiment.items),
            "design": experiment.design,
            "quality_evidence": {
                "execution_only": True,
                "warning_items": sum(any(c.get('status') in {'warning', 'error', 'failed'} for c in i.checks) for i in experiment.items),
                "unknown_items": sum(not i.checks or any(c.get('status') in {'unknown', 'skipped'} for c in i.checks) for i in experiment.items if i.status == 'completed'),
                "summary": '完成仅表示执行结束；警告与未知证据需单独查看，未宣称质量合格。',
            },
        }
        if include_items:
            result["items"] = [asdict(item) for item in experiment.items]
        return result

    def _save(self, experiment: Experiment) -> None:
        path = self.root / f"{experiment.id}.json"
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(asdict(experiment), ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(path)

    def _load(self) -> None:
        for path in self.root.glob("*.json"):
            try:
                raw = json.loads(path.read_text(encoding="utf-8"))
                raw["items"] = [ExperimentItem(**item) for item in raw["items"]]
                experiment = Experiment(**raw)
            except (OSError, ValueError, TypeError, json.JSONDecodeError, KeyError):
                continue
            self.experiments[experiment.id] = experiment
            if self._assign_output_paths(experiment):
                self._save(experiment)

    def _assign_output_paths(self, experiment: Experiment) -> bool:
        """Give every pending request a human-readable, experiment-scoped destination."""
        changed = False
        for item in experiment.items:
            if item.status not in {"pending", "running"}:
                continue
            if item.research_pixel_dispatch and item.research_pixel_dispatch.get('submission_started'):
                continue  # Never rename a frozen attempted/submitted request on reload.
            axes = item.axes
            # 兼容两套轴命名：设计器用 character_id/scene_id，脚本建仓用 character/scene
            character = safe_segment(axes.get("character_id") or axes.get("character") or "character")
            scene = safe_segment(axes.get("scene_id") or axes.get("scene") or "scene")
            model = safe_segment(axes.get("model") or item.request.get("model_id") or "model")
            prompt_form = safe_segment(axes.get("prompt_form") or "prompt")
            seed = axes.get("seed", item.request.get("seed", 0))
            group = f"experiments/{experiment.id}/{character}/{scene}"
            label = f"{model}__{character}__{scene}__{prompt_form}__seed-{seed}"
            if axes.get('study'):
                label = f"{safe_segment(axes.get('recipe', model))}__{model}__{scene}__seed-{seed}"
            if item.request.get("output_group") != group:
                item.request["output_group"] = group
                changed = True
            if item.request.get("output_label") != label:
                item.request["output_label"] = label
                changed = True
        return changed

    def _reconcile_after_restart(self) -> None:
        for experiment in self.experiments.values():
            changed = False
            for item in experiment.items:
                if item.research_pixel_dispatch and item.research_pixel_dispatch.get('submission_started'):
                    before = asdict(item)
                    try:
                        job = self._research_pixel_job(item)
                    except ValueError as exc:
                        item.status, item.error = 'failed', str(exc)
                    else:
                        item.job_id, item.request = job.id, deepcopy(job.request)
                        item.status = job.status if job.status in FINAL_JOB_STATES else 'running'
                        item.images, item.files = list(job.images), list(job.files)
                        item.timings, item.checks, item.error = dict(job.timings), list(job.checks), job.error
                    item.cancel_requested = False
                    changed = changed or before != asdict(item)
                    if item.status == 'running' and experiment.active_item_id != item.id:
                        changed = True
                    continue
                job = self.jobs.get(item.job_id or "")
                if job and job.status in FINAL_JOB_STATES:
                    if item.images != job.images:
                        item.images = list(job.images)
                        changed = True
                    if item.files != job.files:
                        item.files = list(job.files)
                        changed = True
                if item.status != "running":
                    continue
                if job and job.status in FINAL_JOB_STATES:
                    item.status = job.status
                    item.images = list(job.images)
                    item.files = list(job.files)
                    item.timings = dict(job.timings)
                    item.checks = list(job.checks)
                    item.error = job.error
                else:
                    item.status = "pending"
                    item.job_id = None
                item.cancel_requested = False
                changed = True
            if experiment.state == "running" or changed:
                if experiment.state != 'cancelled':
                    experiment.state = "paused"
                experiment.active_item_id = next((item.id for item in experiment.items
                    if item.research_pixel_dispatch and item.status == 'running'), None)
                experiment.updated_at = now_iso()
                self._save(experiment)
