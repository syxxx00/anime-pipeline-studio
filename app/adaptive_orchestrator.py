from __future__ import annotations

import asyncio
import shutil
from pathlib import Path

from PIL import Image, ImageDraw

from .adaptive_checks import check_record_from_artifact, repair_improved
from .adaptive_runs import AdaptiveRun, AdaptiveRunStore, CANDIDATE_NODES
from .jobs import JobManager
from .pipeline_graph import ArtifactRef
from .schemas import GenerateRequest, ModelProfile, RenderPipeline
from .workflow import build_newbie_refine_workflow, build_region_repair_workflow
from .region_edit import protection_fields, repair_parameters


def _legacy_candidate_request(run: AdaptiveRun, node_id: str) -> GenerateRequest:
    base = GenerateRequest.model_validate(run.base_request)
    native = base.native_prompts or {}
    common = {
        "batch_size": 1, "highres_enabled": False, "face_detail_enabled": False,
        "hand_repair_enabled": False, "text_clean_enabled": False, "check_enabled": True,
        "output_group": f"adaptive_runs/{run.id}/{node_id}",
        "output_label": f"{run.id}__{node_id}", "operation": "adaptive_candidate",
    }
    if node_id == "candidate_newbie":
        prompt = str((native.get("newbie") or {}).get("positive") or base.prompt)
        negative = str((native.get("newbie") or {}).get("negative") or base.negative_prompt)
        update = {**common, "render_pipeline": RenderPipeline.NEWBIE, "model_id": ModelProfile.NEWBIE, "prompt": prompt, "negative_prompt": negative}
    elif node_id == "candidate_anima":
        prompt = str((native.get("anima") or {}).get("positive") or base.source_prompt or base.prompt)
        update = {**common, "render_pipeline": RenderPipeline.ANIMA, "model_id": ModelProfile.ANIMA_TURBO, "prompt": prompt, "negative_prompt": "", "steps": 10, "cfg": 1, "shift": 3, "sampler": "euler", "scheduler": "simple", "lora_name": None}
    elif node_id == "candidate_ill":
        prompt = str((native.get("illustrious") or {}).get("positive") or base.source_prompt or base.prompt)
        negative = str((native.get("illustrious") or {}).get("negative") or base.negative_prompt)
        update = {**common, "render_pipeline": RenderPipeline.ILLUSTRIOUS, "model_id": ModelProfile.ILLUSTRIOUS, "prompt": prompt, "negative_prompt": negative, "steps": 28, "cfg": 5, "shift": 0, "sampler": "dpmpp_2m_sde", "scheduler": "karras", "lora_name": None}
    elif node_id == "fusion_ill_newbie":
        newbie = str((native.get("newbie") or {}).get("positive") or base.prompt)
        source = str((native.get("illustrious") or {}).get("positive") or base.source_prompt or base.prompt)
        # The orchestrator supplies the already-rendered Illustrious candidate as
        # pixels.  This request therefore describes only the NewBie refinement;
        # it must never invoke the older two-stage bridge and render Illustrious
        # a second time.
        update = {
            **common, "render_pipeline": RenderPipeline.NEWBIE,
            "model_id": ModelProfile.NEWBIE, "prompt": newbie,
            "source_prompt": source, "operation": "adaptive_fusion",
        }
    else:
        raise ValueError("unknown candidate node")
    return GenerateRequest.model_validate({**base.model_dump(mode="json"), **update})


def explicit_candidate_request(base: GenerateRequest, node_id: str, run_id: str) -> GenerateRequest:
    """Resolve each candidate's own text/model settings, not an assumed art role.

    Matching main/source settings are reused; otherwise the candidate gets its
    own existing model profile. Source overrides are never carried into a
    single-model request. A missing native channel cannot fall back to other
    models' XML/tags in new explicit creation.
    """
    from .sampling_profiles import PROFILES
    from .bridge_sampling import source_request
    profiles = {'candidate_newbie':'newbie', 'candidate_anima':'anima',
                'candidate_ill':'illustrious', 'fusion_ill_newbie':'newbie'}
    if node_id not in profiles:
        raise ValueError('unknown candidate node')
    profile_id = profiles[node_id]
    profile = PROFILES[profile_id]
    native = base.native_prompts or {}
    channel = native.get(profile_id) or {}
    same_model = base.model_id.value == profile['model_id']
    prompt = str(channel.get('positive') or (base.prompt if same_model else ''))
    if not prompt.strip():
        raise ValueError(f'候选 {node_id} 缺少 {profile_id} 原生提示词，不能用其他模型的格式代替')
    from .model_conditioning import negative_for_target
    negative = negative_for_target(base.model_dump(mode='json'), channel, profile_id)
    parameters = dict(profile, seed=base.seed, lora_name=None, lora_strength=.75, cfg_rescale=.9,
                      weight_dtype='fp8_e5m2' if profile_id == 'z_image_turbo' else None)
    owner = base if same_model else None
    if not owner and base.source_parameters and base.source_parameters.model_id == profile['model_id']:
        owner = source_request(base)
    if owner:
        for key in ('steps','cfg','seed','sampler','scheduler','shift','cfg_rescale','weight_dtype','lora_name','lora_strength'):
            parameters[key] = getattr(owner, key)
    values = base.model_dump(mode='json') | parameters | {
        'render_pipeline':profile_id, 'prompt':prompt, 'negative_prompt':negative,
        'use_default_negative_prompt': False,
        'source_parameters':None, 'source_negative_prompt':None,
        'source_prompt':'', 'anima_prompt':'', 'batch_size':1,
        'highres_enabled':False, 'face_detail_enabled':False,
        'hand_repair_enabled':False, 'text_clean_enabled':False, 'check_enabled':True,
        'output_group':f'adaptive_runs/{run_id}/{node_id}', 'output_label':f'{run_id}__{node_id}',
        'operation':'adaptive_candidate',
    }
    if node_id == 'fusion_ill_newbie':
        # The previously rendered Illustrious image is the input; do not rerender
        # it or copy its seed/sampler into the NewBie sampling stage.
        values.update(operation='adaptive_fusion', source_prompt=str((native.get('illustrious') or {}).get('positive') or ''))
    return GenerateRequest.model_validate(values)


def candidate_request(run: AdaptiveRun, node_id: str) -> GenerateRequest:
    base = GenerateRequest.model_validate(run.base_request)
    if base.parameter_policy == 'legacy':
        return _legacy_candidate_request(run, node_id)
    return explicit_candidate_request(base, node_id, run.id)


class AdaptiveOrchestrator:
    def __init__(self, store: AdaptiveRunStore, jobs: JobManager) -> None:
        self.store = store
        self.jobs = jobs
        self.worker: asyncio.Task | None = None

    def start(self) -> None:
        if self.worker is None or self.worker.done():
            self.worker = asyncio.create_task(self._run(), name="adaptive-pipeline-orchestrator")

    async def stop(self) -> None:
        if self.worker:
            self.worker.cancel()
            try:
                await self.worker
            except asyncio.CancelledError:
                pass

    async def launch(self, contract, request: GenerateRequest) -> AdaptiveRun:
        if request.parameter_policy == 'explicit':
            from .generation_request import prepare_generation_request
            from .lora_catalog import describe_lora
            if request.scene_contract is not None:
                from .scene_contract import SceneContractV3
                if SceneContractV3.model_validate(request.scene_contract) != contract:
                    raise ValueError('协作图契约与候选请求契约不一致，请重新编译后提交')
            prepare_generation_request(request, self.jobs.settings)
            # Fail before persisting a running run if its candidate channels or
            # resource declarations cannot execute. No engine/model is started.
            for node_id in CANDIDATE_NODES:
                candidate = explicit_candidate_request(request, node_id, 'preflight')
                if candidate.lora_name:
                    root = self.jobs.settings.path_map['loras'].resolve()
                    path = (root / candidate.lora_name).resolve()
                    if not path.is_relative_to(root) or not path.is_file() or candidate.model_id.value not in describe_lora(path, root)['compatible_models']:
                        raise ValueError(f'候选 {node_id} 的 LoRA 缺少有效文件或兼容架构证据')
                prepare_generation_request(candidate, self.jobs.settings)
        run = self.store.create(contract)
        self.store.attach_request(run, request.model_dump(mode="json"))
        return run

    async def _run(self) -> None:
        while True:
            for run in self.store.list():
                if run.state.status != "running" or not run.base_request:
                    continue
                await self._tick(run)
            await asyncio.sleep(1)

    async def _tick(self, run: AdaptiveRun) -> None:
        for node_id in list(run.pending_nodes):
            if run.state.status != "running" or run.pause_reason:
                return
            if node_id not in CANDIDATE_NODES:
                continue
            job_id = run.candidate_job_ids.get(node_id)
            if not job_id:
                try:
                    job = await self._submit_candidate(run, node_id)
                except (OSError, RuntimeError, ValueError) as exc:
                    if run.state.status == "running":
                        self.store.fail_candidate(run, node_id, str(exc))
                else:
                    self.store.attach_candidate_job(run, node_id, job.id)
                    if run.state.status == "cancelled":
                        await self.jobs.cancel(job.id)
                continue
            job = self.jobs.get(job_id)
            if job is None:
                self.store.fail_candidate(run, node_id, "candidate job disappeared")
            elif job.status == "completed":
                path = job.files[0] if job.files else ""
                artifact = ArtifactRef(
                    id=f"{job.id}-01", kind="image", path=path, producer_node=node_id,
                    parent_ids=[run.candidates['candidate_ill'].artifact.id] if node_id == 'fusion_ill_newbie' and run.candidates['candidate_ill'].artifact else [],
                    metadata={"job_id": job.id, "image_url": job.images[0] if job.images else "", "checks": job.checks},
                )
                self.store.record_candidate(run, node_id, artifact)
                self._cleanup_fusion_input(run, node_id)
            elif job.status in {"failed", "cancelled"}:
                self.store.fail_candidate(run, node_id, job.error or job.status)
                self._cleanup_fusion_input(run, node_id)
        if run.state.status != "running" or run.pause_reason:
            return
        if run.pending_nodes == ["component_check"]:
            artifact = self._artifact(run, run.selected_artifact_id)
            if artifact is None:
                self.store.block_repair(run, "选中作品记录不存在，无法恢复检查")
            else:
                self.store.record_check(run, check_record_from_artifact(artifact))
        if run.state.status == "running" and run.pending_nodes == ["region_repair"]:
            await self._tick_repair(run)

    def _fusion_input_name(self, run: AdaptiveRun) -> str:
        return f"adaptive_{run.id}_ill_source.png"

    def _cleanup_fusion_input(self, run: AdaptiveRun, node_id: str) -> None:
        if node_id != "fusion_ill_newbie":
            return
        input_path = Path(self.jobs.settings.engine_base_dir) / "input" / self._fusion_input_name(run)
        input_path.unlink(missing_ok=True)

    async def _submit_candidate(self, run: AdaptiveRun, node_id: str):
        request = candidate_request(run, node_id)
        if node_id != "fusion_ill_newbie":
            return await self.jobs.submit(request)

        source_slot = run.candidates["candidate_ill"]
        source = Path(source_slot.artifact.path) if source_slot.artifact and source_slot.artifact.path else None
        if source is None or not source.is_file():
            raise RuntimeError("Illustrious 候选产物不存在，不能执行像素协作")
        request = request.model_copy(update={'parent_artwork_id': source_slot.artifact.id})
        input_dir = Path(self.jobs.settings.engine_base_dir) / "input"
        input_dir.mkdir(parents=True, exist_ok=True)
        input_name = self._fusion_input_name(run)
        shutil.copy2(source, input_dir / input_name)
        workflow = build_newbie_refine_workflow(
            request, self.jobs.settings, input_name,
            f"newbie_lab/adaptive_runs/{run.id}/fusion_ill_newbie",
        )
        return await self.jobs.submit_workflow(request, workflow)

    def _artifact(self, run: AdaptiveRun, artifact_id: str) -> ArtifactRef | None:
        return next((item for item in reversed(run.state.artifacts) if item.id == artifact_id), None)

    def _repair_action(self, run: AdaptiveRun):
        if run.repair_plan is None:
            return None
        if run.repair_action_id:
            return next((item for item in run.repair_plan.actions if item.id == run.repair_action_id), None)
        return next((item for item in run.repair_plan.actions if item.auto_runnable and item.operation == "local_inpaint"), None)

    @staticmethod
    def _region_box(region: list[float], width: int, height: int) -> tuple[int, int, int, int]:
        x, y, w, h = region
        if max(abs(value) for value in region) <= 1.0:
            x, w = x * width, w * width
            y, h = y * height, h * height
        pad = max(8, round(max(w, h) * 0.12))
        left, top = max(0, round(x - pad)), max(0, round(y - pad))
        right, bottom = min(width, round(x + w + pad)), min(height, round(y + h + pad))
        if right <= left or bottom <= top:
            raise ValueError("缺陷区域为空")
        return left, top, right, bottom

    async def _submit_automatic_repair(self, run: AdaptiveRun):
        action = self._repair_action(run)
        if action is None or action.target_region is None:
            raise RuntimeError("没有可安全自动执行的局部修复")
        source = self._artifact(run, run.selected_artifact_id)
        source_path = Path(source.path) if source and source.path else None
        if source_path is None or not source_path.is_file():
            raise RuntimeError("选中作品文件不存在")

        input_dir = Path(self.jobs.settings.engine_base_dir) / "input"
        input_dir.mkdir(parents=True, exist_ok=True)
        stem = f"adaptive_{run.id}_repair_{run.repair_attempts + 1}"
        source_name, mask_name = f"{stem}_source.png", f"{stem}_mask.png"
        source_input, mask_input = input_dir / source_name, input_dir / mask_name
        shutil.copy2(source_path, source_input)
        with Image.open(source_path) as image:
            width, height = image.size
        mask = Image.new("L", (width, height), 0)
        ImageDraw.Draw(mask).rectangle(self._region_box(action.target_region, width, height), fill=255)
        mask.save(mask_input, format="PNG")

        base = GenerateRequest.model_validate(run.base_request)
        parameters, notes = repair_parameters(base.model_dump(mode="json"), self.jobs.settings)
        request = GenerateRequest.model_validate({
            **parameters, 'prompt': action.instruction, 'repair_notes': notes,
            **protection_fields(source_input, mask_input),
            'parameter_policy': 'explicit',
            "model_id": ModelProfile.NEWBIE, "render_pipeline": RenderPipeline.NEWBIE,
            "batch_size": 1, "output_group": f"adaptive_runs/{run.id}/repairs",
            "output_label": f"{run.id}__repair-{run.repair_attempts + 1}",
            "highres_enabled": False, "face_detail_enabled": False,
            "hand_repair_enabled": False, "text_clean_enabled": False,
            "operation": "adaptive_region_repair", "parent_artwork_id": run.selected_artifact_id,
            "repair_instruction": action.instruction,
            "mask_file": str(mask_input.resolve()),
        })
        workflow = build_region_repair_workflow(
            request, self.jobs.settings, source_name, mask_name,
            f"newbie_lab/adaptive_runs/{run.id}/repair_{run.repair_attempts + 1}",
        )
        job = await self.jobs.submit_workflow(request, workflow)
        self.store.attach_repair_job(
            run, job.id, action_id=action.id, origin="automatic",
            input_files=[str(source_input), str(mask_input)],
        )
        return job

    def _cleanup_repair_inputs(self, run: AdaptiveRun) -> None:
        for value in run.repair_input_files:
            Path(value).unlink(missing_ok=True)

    async def _tick_repair(self, run: AdaptiveRun) -> None:
        if run.state.status != "running" or run.pause_reason:
            return
        if not run.repair_job_id:
            try:
                await self._submit_automatic_repair(run)
            except (OSError, RuntimeError, ValueError) as exc:
                if run.state.status == "running":
                    self.store.block_repair(run, str(exc))
            if run.state.status == "cancelled" and run.repair_job_id:
                await self.jobs.cancel(run.repair_job_id)
                self._cleanup_repair_inputs(run)
            return
        job = self.jobs.get(run.repair_job_id)
        if job is None:
            self._cleanup_repair_inputs(run)
            self.store.record_repair(run, None, improved=False)
            return
        if job.status == "completed":
            path = job.files[0] if job.files else ""
            artifact = ArtifactRef(
                id=f"{job.id}-01", kind="image", path=path, producer_node="region_repair",
                parent_ids=[run.selected_artifact_id],
                metadata={"job_id": job.id, "image_url": job.images[0] if job.images else "", "checks": job.checks},
            )
            after = check_record_from_artifact(artifact)
            before = next((item for item in reversed(run.state.checks) if item.artifact_id == run.selected_artifact_id), None)
            action = self._repair_action(run)
            user_mask = run.repair_origin == 'user_mask'
            improved = False if user_mask else repair_improved(before, after, action)
            self._cleanup_repair_inputs(run)
            if user_mask:
                self.store.record_user_repair_candidate(run, artifact, after)
            else:
                self.store.complete_repair(run, artifact, after, improved=improved)
        elif job.status in {"failed", "cancelled"}:
            self._cleanup_repair_inputs(run)
            self.store.record_repair(run, None, improved=False)

    async def cancel(self, run: AdaptiveRun) -> AdaptiveRun:
        # Stop dispatch before the first await. An already in-flight submission
        # is retained and cancelled by _tick when it returns.
        self.store.cancel(run)
        for job_id in list(run.candidate_job_ids.values()):
            await self.jobs.cancel(job_id)
        self._cleanup_fusion_input(run, "fusion_ill_newbie")
        if run.repair_job_id:
            await self.jobs.cancel(run.repair_job_id)
        self._cleanup_repair_inputs(run)
        return run
