from __future__ import annotations

import platform
import subprocess
import json
import uuid
from contextlib import asynccontextmanager
from dataclasses import asdict
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from starlette.concurrency import run_in_threadpool

from .artworks import ArtworkStore
from .research.boundary_matrix import build_boundary_manifest, load_plan as load_boundary_plan, plan_summary as boundary_plan_summary
from .comfy_client import ComfyClient, OwnedCancellationUnconfirmed
from .checks import capability_report
from .config import PROJECT_ROOT, load_settings
from .research.experiments import ExperimentManager
from .research.designer import build_manifest, defaults as designer_defaults, preview as designer_preview
from .adaptive_runs import AdaptiveRunStore
from .adaptive_orchestrator import AdaptiveOrchestrator
from .adaptive_checks import check_record_from_artifact
from .pipeline_graph import build_adaptive_graph, initial_scene_state
from .pipeline_graph import ArtifactRef, CheckRecord, Defect
from .pipeline_runtime import build_repair_plan, defects_from_review
from .scene_contract import EntityKind, SceneContractV3, SceneEntity
from .text_render import TextLayerSpec, render_text_layer
from .jobs import JobManager
from .job_engine_receipts import EngineReceiptPersistenceError
from .stage_api import stage_router
from .region_api import region_router
from .handoff_api import handoff_router
from .iteration_api import iteration_router
from .creation_concept_context import CreationConceptContext
from .local_iteration_planner import LocalIterationPlanner, local_text_runtime, local_visual_runtime
from .iteration_controller import IterationBudget, IterationController, IterationStore
from .iteration_observation import IterationObservations
from .iteration_observation_api import iteration_observation_router
from .creation_preparation import CreationPreparations, PreparationRequest
from .creative_sessions import CreativeSessions, CreativeRequest, CreativeAction
from .visual_material import load_visual_material
from .local_authoring_runtime import inference_state
from .research.controlled_studies import studies_router
from .knowledge import apply_knowledge, list_packs, pack_with_audit
from .model_catalog import model_catalog
from .model_connections import model_connections_router
from .creation_options import creation_options
from .lora_catalog import lora_catalog
from .newbie_compiler import CompileResult, build_negative, compile_sentence, compile_xml, lint, response_for
from .prompt_generator import PromptGeneratorUnavailable, draft_prompt
from .professional_review import build_professional_review
from .research.analysis import analyze_experiment
from .routing_guidance import RoutingGuidanceStore
from .prompting import list_presets, matrix_prompt, natural_matrix_prompt, validate_xml_prompt
from .workflow import build_anima_base_workflow, build_newbie_refine_workflow, build_region_repair_workflow, build_workflow
from .style_catalog import style_catalog_json
from .schemas import ArtworkReviewRequest, DraftPromptRequest, ExperimentCreateRequest, ExperimentDesignRequest, ExperimentReviewRequest, GenerateRequest, JobResponse, MatrixRequest, ModelProfile, NewbieCompileRequest, NewbieRecompileRequest, PIXEL_BRIDGE_PIPELINES, RenderPipeline, TestMode


settings = load_settings()
client = ComfyClient(settings)
jobs = JobManager(settings, client)
artworks = ArtworkStore()
experiments = ExperimentManager(settings, jobs, client, artworks=artworks)
adaptive_runs = AdaptiveRunStore()
adaptive_orchestrator = AdaptiveOrchestrator(adaptive_runs, jobs)
routing_guidance = RoutingGuidanceStore()
iteration_planner = LocalIterationPlanner(concepts=CreationConceptContext.for_settings(settings),
    text_runtime_factory=lambda: local_text_runtime(settings),
    visual_runtime_factory=lambda: local_visual_runtime(settings),
    evidence_root=PROJECT_ROOT / 'work' / 'local_iteration_planning')
iteration_runs = IterationStore(PROJECT_ROOT / 'data' / 'iteration_runs')
iteration_observations = IterationObservations(artworks, jobs, iteration_runs,
    PROJECT_ROOT / 'work' / 'iteration_observations')
iteration_controller = IterationController(iteration_runs, jobs, artworks, iteration_planner,
    observations=iteration_observations)
iteration_planner.bind(iteration_controller)
creation_preparations = CreationPreparations(settings)
creative_sessions = CreativeSessions(settings, jobs, artworks)


def sync_artworks() -> None:
    artworks.sync_jobs(jobs.list())
    artworks.sync_experiments(experiments.experiments.values())


@asynccontextmanager
async def lifespan(_: FastAPI):
    sync_artworks()
    jobs.start()
    experiments.start()
    adaptive_orchestrator.start()
    iteration_controller.start()
    yield
    await creative_sessions.stop()
    await creation_preparations.stop()
    await adaptive_orchestrator.stop()
    await iteration_controller.stop()
    await experiments.stop()
    await jobs.stop()


app = FastAPI(title="Anime Pipeline Studio", version="adaptive", lifespan=lifespan)
app.include_router(model_connections_router(settings))
# The explicit observations POST must precede the legacy /{run_id}/{action} route.
app.include_router(iteration_observation_router(iteration_controller, iteration_observations))
app.include_router(iteration_router(iteration_controller, iteration_planner))
app.include_router(stage_router(jobs, artworks))
app.include_router(studies_router(experiments))
app.include_router(handoff_router(jobs, artworks, settings, require_artwork=lambda value: require_artwork(value)))
app.include_router(region_router(jobs, artworks, settings,
    require_artwork=lambda value: require_artwork(value),
    require_adaptive_run=lambda value: require_adaptive_run(value), adaptive_runs=adaptive_runs))
static_dir = PROJECT_ROOT / "static"


@app.post('/api/creative-sessions')
async def create_creative_session(request: CreativeRequest) -> dict:
    try:
        return await creative_sessions.create(request)
    except OwnedCancellationUnconfirmed as exc:
        raise HTTPException(409, str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc


@app.get('/api/creative-sessions')
async def list_creative_sessions(request_id: str | None = None, artwork_id: str | None = None,
                                 limit: int = 20) -> list[dict]:
    return creative_sessions.list(request_id=request_id, artwork_id=artwork_id, limit=max(1, min(100, limit)))


@app.get('/api/creative-sessions/{identifier}')
async def get_creative_session(identifier: str) -> dict:
    record = creative_sessions.get(identifier)
    if record is None:
        raise HTTPException(404, '创作任务不存在；没有自动重新提交。')
    return record


@app.post('/api/creative-sessions/{identifier}/action')
async def creative_session_action(identifier: str, action: CreativeAction) -> dict:
    if creative_sessions.get(identifier) is None:
        raise HTTPException(404, '创作任务不存在')
    try:
        return await creative_sessions.action(identifier, action)
    except ValueError as exc:
        raise HTTPException(409, str(exc)) from exc


@app.get('/api/creative-materials/{identifier}/image')
async def get_creative_material_image(identifier: str):
    try:
        material = load_visual_material(identifier, settings)
    except (OSError, ValueError) as exc:
        raise HTTPException(404, '中间视觉材料不存在或原件校验失败') from exc
    return FileResponse(material.file_path, media_type='image/png')
generated_dir = Path(settings.output_dir)
app.mount("/assets", StaticFiles(directory=static_dir), name="assets")
app.mount("/generated", StaticFiles(directory=generated_dir), name="generated")


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(static_dir / "index.html")


@app.get('/models')
async def model_connections_page() -> FileResponse:
    return FileResponse(static_dir / 'model-connections.html')


@app.get('/api/app-identity')
async def app_identity() -> dict:
    return {'application': 'anime-pipeline-studio',
            'project_root': str(PROJECT_ROOT.resolve()), 'version': '0.1.0'}


@app.get('/iteration')
async def iteration_page() -> FileResponse:
    return FileResponse(static_dir / 'iteration.html')


@app.get("/pipeline")
async def pipeline() -> FileResponse:
    return FileResponse(static_dir / "pipeline.html")


@app.get("/api/health")
async def health() -> dict:
    paths = {key: path.exists() for key, path in settings.path_map.items()}
    return {
        "app": "ok",
        "engine": await client.health(),
        "paths": paths,
        "platform": platform.platform(),
    }


@app.get("/api/runtime")
async def runtime() -> dict:
    return await client.runtime_status()


@app.get("/api/config")
async def config() -> dict:
    return settings.public_dict()


@app.get("/api/check-capabilities")
async def check_capabilities() -> dict:
    return capability_report(settings)


PIPELINE_DEFS = [
    ("anima_newbie", "Anima → NewBie", "Anima 构图 → 卸载 → NewBie 角色化 → 潜空间放大 → 修脸 → check 环节", False),
    ("newbie", "仅 NewBie", "单模型直接生成 → 潜空间放大 → 修脸 → check 环节", False),
    ("illustrious_newbie", "Illustrious → NewBie", "Illustrious SDXL 构图 → 无损像素桥接 → NewBie 角色化 → check", False),
    ("z_newbie", "Z-Image → NewBie", "Z-Image 自然语言构图 → 无损像素桥接 → NewBie 角色化 → check", False),
    ("anima", "仅 Anima Turbo", "10 步快速直出，适合看构图草稿", False),
    ("illustrious", "Illustrious XL", "SDXL 系二次元第三基线；适合标签、画风和角色知识对照", False),
    ("z_image_turbo", "Z-Image Turbo", "快速通用候选；用于验证自然语言与构图，不作为角色主线", False),
]


def _pipeline_preview_workflow(pipeline_id: str) -> dict:
    request_kwargs: dict = {
        "prompt": "1girl, solo",
        "render_pipeline": RenderPipeline(pipeline_id),
        "highres_enabled": True,
        "face_detail_enabled": True,
    }
    if pipeline_id == "anima":
        request_kwargs.update({"model_id": ModelProfile.ANIMA_TURBO, "sampler": "euler", "scheduler": "simple", "steps": 10, "cfg": 1.0, "shift": 3.0})
    elif pipeline_id == "z_image_turbo":
        request_kwargs.update({"model_id": ModelProfile.Z_IMAGE_TURBO, "sampler": "euler", "scheduler": "simple", "steps": 9, "cfg": 1.0, "shift": 3.0, "face_detail_enabled": False})
    elif pipeline_id == "illustrious":
        request_kwargs.update({"model_id": ModelProfile.ILLUSTRIOUS, "sampler": "dpmpp_2m_sde", "scheduler": "karras", "steps": 28, "cfg": 5.0, "face_detail_enabled": False})
    request = GenerateRequest(anima_prompt="1girl, solo", source_prompt="1girl, solo", **request_kwargs)
    if request.render_pipeline in PIXEL_BRIDGE_PIPELINES:
        source_name = {
            RenderPipeline.HYBRID: "Anima Turbo v1.1",
            RenderPipeline.ILLUSTRIOUS_NEWBIE: "Illustrious XL",
            RenderPipeline.Z_NEWBIE: "Z-Image Turbo",
        }[request.render_pipeline]
        if request.render_pipeline == RenderPipeline.HYBRID:
            source_workflow = build_anima_base_workflow(request, settings, "preview/source")
        else:
            source_request = jobs._bridge_source(request)[1]
            source_workflow = build_workflow(source_request, settings, "preview/source")
        return {
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
            "stages": {
                "1_source_composition": source_workflow,
                "2_newbie_identity_and_detail": build_newbie_refine_workflow(request, settings, "preview_anima.png", "preview/final"),
            },
        }
    return build_workflow(request, settings, "preview/out")


@app.get("/api/pipelines")
async def pipelines() -> list[dict]:
    return [
        {
            "id": pipeline_id,
            "name": name,
            "summary": summary,
            "default": is_default,
            "workflow": _pipeline_preview_workflow(pipeline_id),
        }
        for pipeline_id, name, summary, is_default in PIPELINE_DEFS
    ]


@app.get("/api/presets")
async def presets() -> list[dict[str, str]]:
    return list_presets()


@app.get("/api/models")
async def models() -> list[dict[str, object]]:
    return model_catalog(settings)


@app.get("/api/style-presets")
async def style_presets() -> list[dict[str, str]]:
    return style_catalog_json(settings)


@app.get("/api/loras")
async def loras() -> list[dict[str, object]]:
    return lora_catalog(settings.path_map["loras"])


@app.get("/api/creation-options")
async def get_creation_options() -> dict:
    return creation_options()


@app.post("/api/validate-prompt")
async def validate_prompt(payload: dict[str, str]) -> dict:
    prompt = payload.get("prompt", "")
    return {"valid": not any("解析失败" in item for item in validate_xml_prompt(prompt)), "warnings": validate_xml_prompt(prompt)}


@app.post("/api/draft-prompt")
async def create_prompt_draft(request: DraftPromptRequest) -> dict[str, object]:
    try:
        return await draft_prompt(request, settings)
    except PromptGeneratorUnavailable as exc:
        raise HTTPException(503, str(exc)) from exc


@app.get("/api/newbie-compiler/status")
async def newbie_compiler_status() -> dict:
    model_path = Path(settings.semantic_model_path)
    return {
        "local_model_ready": model_path.exists(),
        "model_path": str(model_path),
        "model_name": model_path.name,
        "fallback_ready": True,
    }


@app.post('/api/creation-preparations')
async def create_preparation(request: PreparationRequest) -> dict:
    try:
        return await creation_preparations.create(request)
    except (KeyError, ValueError) as exc:
        raise HTTPException(409, str(exc)) from exc


@app.get('/api/inference-state')
async def local_inference_state() -> dict:
    # Process-local coordination, not a machine-wide GPU lock. No lease token.
    return {'scope': 'this_api_process_only', **inference_state()}


@app.get('/api/creation-preparations/{identifier}')
async def get_preparation(identifier: str) -> dict:
    record = creation_preparations.get(identifier)
    if record is None:
        raise HTTPException(404, '准备记录不存在')
    return record


@app.post('/api/creation-preparations/{identifier}/cancel')
async def cancel_preparation(identifier: str) -> dict:
    if creation_preparations.get(identifier) is None:
        raise HTTPException(404, '准备记录不存在')
    return await creation_preparations.cancel(identifier)


@app.post('/api/creation-preparations/{identifier}/rebuild')
async def rebuild_preparation(identifier: str, body: dict) -> dict:
    if creation_preparations.get(identifier) is None:
        raise HTTPException(404, '准备记录不存在')
    try:
        return creation_preparations.rebuild(identifier, body)
    except ValueError as exc:
        raise HTTPException(409, str(exc)) from exc


@app.post('/api/creation-preparations/{identifier}/source-review')
async def continue_source_review(identifier: str, body: dict) -> dict:
    if creation_preparations.get(identifier) is None:
        raise HTTPException(404, '准备记录不存在')
    try:
        # This continues interpretation/authoring only; it never queues images.
        return await creation_preparations.resume_source(identifier, body)
    except ValueError as exc:
        raise HTTPException(409, str(exc)) from exc


@app.post('/api/creation-preparations/{identifier}/iteration')
async def submit_prepared_iteration(identifier: str, body: dict) -> dict:
    if creation_preparations.get(identifier) is None:
        raise HTTPException(404, '准备记录不存在')
    try:
        if set(body) - {'request', 'allowed_models', 'budget', 'autonomous_planning'}:
            raise ValueError('存在未知迭代提交字段；没有忽略或降级执行')
        request = GenerateRequest.model_validate(body.get('request'))
        if request.preparation_id != identifier:
            raise ValueError('URL 与完整请求的准备 ID 不一致')
        run = await creation_preparations.submit_iteration(request, iteration_controller,
            body.get('allowed_models', []), IterationBudget.model_validate(body.get('budget', {})),
            autonomous_planning=body.get('autonomous_planning', False))
        return iteration_controller.snapshot(run) | {
            'decision_mode': 'local_result_guided_unverified' if run.autonomous_planning else 'agent_or_human_review',
            'decision_queued': run.id in iteration_planner.pending}
    except (KeyError, ValueError) as exc:
        raise HTTPException(409, str(exc)) from exc


@app.get("/api/knowledge-packs")
async def knowledge_packs() -> list[dict]:
    return list_packs()


@app.get("/api/knowledge-packs/{pack_id}")
async def knowledge_pack(pack_id: str) -> dict:
    try:
        return pack_with_audit(pack_id)
    except KeyError as exc:
        raise HTTPException(404, "知识库不存在") from exc


@app.post("/api/newbie-compiler/compile")
async def compile_newbie_sentence(request: NewbieCompileRequest) -> dict:
    result = await run_in_threadpool(compile_sentence, request.sentence, request.style_mode, settings)
    parsed_character_tag = result.spec.character_tag
    if request.knowledge_pack:
        try:
            result = apply_knowledge(result, request.knowledge_pack, request.knowledge_operator, request.knowledge_appearance)
        except ValueError as exc:
            raise HTTPException(409, str(exc)) from exc
    response = response_for(result)
    response['parsed_character_tag'] = parsed_character_tag
    response["professional_review"] = build_professional_review(
        result,
        parsed_character_tag=parsed_character_tag,
        knowledge_operator=request.knowledge_operator,
    )
    return response


@app.post("/api/newbie-compiler/recompile")
async def recompile_newbie_spec(request: NewbieRecompileRequest) -> dict:
    from .newbie_compiler import recompile_response
    try:
        return recompile_response(request)
    except (KeyError, ValueError) as exc:
        raise HTTPException(409, str(exc)) from exc


@app.post("/api/v3/graph/preview")
async def preview_v3_graph(contract: SceneContractV3) -> dict:
    graph = build_adaptive_graph(contract)
    state = initial_scene_state(contract, graph)
    return {"graph": graph.model_dump(mode="json"), "initial_state": state.model_dump(mode="json")}


@app.get("/api/v3/graph/default")
async def default_v3_graph() -> dict:
    contract = SceneContractV3(
        source_text="自适应动漫创作管线示例",
        entities=[SceneEntity(id="character", kind=EntityKind.CHARACTER, name="selected character")],
    )
    graph = build_adaptive_graph(contract)
    return graph.model_dump(mode="json")


def require_adaptive_run(run_id: str):
    run = adaptive_runs.get(run_id)
    if run is None:
        raise HTTPException(404, "自适应任务不存在")
    return run


@app.get("/api/adaptive-runs")
async def list_adaptive_runs() -> list[dict]:
    return [item.model_dump(mode="json") for item in adaptive_runs.list()]


@app.post("/api/adaptive-runs")
async def create_adaptive_run(contract: SceneContractV3) -> dict:
    return adaptive_runs.create(contract).model_dump(mode="json")


@app.post("/api/adaptive-runs/launch")
async def launch_adaptive_run(payload: dict) -> dict:
    try:
        contract = SceneContractV3.model_validate(payload.get("contract"))
        request = GenerateRequest.model_validate(payload.get("request"))
        run = await adaptive_orchestrator.launch(contract, request)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    return run.model_dump(mode="json")


@app.get("/api/adaptive-runs/{run_id}")
async def get_adaptive_run(run_id: str) -> dict:
    return require_adaptive_run(run_id).model_dump(mode="json")


@app.post("/api/adaptive-runs/{run_id}/candidates/{node_id}")
async def record_adaptive_candidate(run_id: str, node_id: str, artifact: ArtifactRef) -> dict:
    try:
        run = adaptive_runs.record_candidate(require_adaptive_run(run_id), node_id, artifact)
    except ValueError as exc:
        raise HTTPException(409, str(exc)) from exc
    return run.model_dump(mode="json")


@app.post("/api/adaptive-runs/{run_id}/candidates/{node_id}/fail")
async def fail_adaptive_candidate(run_id: str, node_id: str, payload: dict) -> dict:
    try:
        run = adaptive_runs.fail_candidate(require_adaptive_run(run_id), node_id, str(payload.get("error", "candidate failed")))
    except ValueError as exc:
        raise HTTPException(409, str(exc)) from exc
    return run.model_dump(mode="json")


@app.post("/api/adaptive-runs/{run_id}/select")
async def select_adaptive_candidate(run_id: str, payload: dict) -> dict:
    try:
        run = adaptive_runs.select_candidate(
            require_adaptive_run(run_id), str(payload.get("artifact_id", "")),
            [str(item) for item in payload.get("locked_refs", [])],
        )
        artifact = next(item for item in run.state.artifacts if item.id == run.selected_artifact_id)
        run = adaptive_runs.record_check(run, check_record_from_artifact(artifact))
    except ValueError as exc:
        raise HTTPException(409, str(exc)) from exc
    return run.model_dump(mode="json")


@app.post("/api/adaptive-runs/{run_id}/check")
async def check_adaptive_candidate(run_id: str, record: CheckRecord) -> dict:
    try:
        run = adaptive_runs.record_check(require_adaptive_run(run_id), record)
    except ValueError as exc:
        raise HTTPException(409, str(exc)) from exc
    return run.model_dump(mode="json")


@app.post("/api/adaptive-runs/{run_id}/review")
async def review_adaptive_candidate(run_id: str, payload: dict) -> dict:
    run = require_adaptive_run(run_id)
    text = str(payload.get("text", "")).strip()
    artifact_id = str(payload.get("artifact_id") or run.selected_artifact_id)
    if not text or len(text) > 8000:
        raise HTTPException(422, "评价需为 1 至 8000 字")
    if not any(item.id == artifact_id for item in run.state.artifacts):
        raise HTTPException(409, "评价对象不属于此协作任务")
    defects = defects_from_review(text, artwork_id=artifact_id)
    record = CheckRecord(node_id="human_review", artifact_id=artifact_id, passed=False, defects=defects) if defects else None
    # Reviews of any candidate are useful evidence. Only an idle selected
    # version may change the next step; a review must not interrupt a repair.
    can_route = artifact_id == run.selected_artifact_id
    try:
        adaptive_runs._require_user_boundary(run)
    except ValueError:
        can_route = False
    review = artworks.add_review(require_artwork(artifact_id), text)
    adaptive_runs.record_review(run, text, record if can_route else None)
    return {**run.model_dump(mode="json"), "saved_review": review,
            "review_check_updated": bool(can_route and record is not None)}


@app.post("/api/adaptive-runs/{run_id}/repair-result")
async def record_adaptive_repair(run_id: str, payload: dict) -> dict:
    try:
        artifact_data = payload.get("artifact")
        artifact = ArtifactRef.model_validate(artifact_data) if artifact_data else None
        run = adaptive_runs.record_repair(require_adaptive_run(run_id), artifact, improved=bool(payload.get("improved")))
    except ValueError as exc:
        raise HTTPException(409, str(exc)) from exc
    return run.model_dump(mode="json")


@app.post("/api/adaptive-runs/{run_id}/accept")
async def accept_adaptive_run(run_id: str) -> dict:
    try:
        run = adaptive_runs.accept(require_adaptive_run(run_id))
    except ValueError as exc:
        raise HTTPException(409, str(exc)) from exc
    return run.model_dump(mode="json")


@app.post("/api/adaptive-runs/{run_id}/rollback")
async def rollback_adaptive_run(run_id: str) -> dict:
    try:
        run = adaptive_runs.rollback(require_adaptive_run(run_id))
    except ValueError as exc:
        raise HTTPException(409, str(exc)) from exc
    return run.model_dump(mode="json")


@app.post("/api/adaptive-runs/{run_id}/cancel")
async def cancel_adaptive_run(run_id: str) -> dict:
    try:
        return (await adaptive_orchestrator.cancel(require_adaptive_run(run_id))).model_dump(mode="json")
    except ValueError as exc:
        raise HTTPException(409, str(exc)) from exc


@app.post("/api/adaptive-runs/{run_id}/pause")
async def pause_adaptive_run(run_id: str) -> dict:
    try:
        return adaptive_runs.pause(require_adaptive_run(run_id)).model_dump(mode="json")
    except ValueError as exc:
        raise HTTPException(409, str(exc)) from exc


@app.post("/api/adaptive-runs/{run_id}/resume")
async def resume_adaptive_run(run_id: str) -> dict:
    try:
        run = adaptive_runs.resume(require_adaptive_run(run_id))
    except ValueError as exc:
        raise HTTPException(409, str(exc)) from exc
    return run.model_dump(mode="json")


@app.post("/api/v3/repair/plan")
async def preview_v3_repair_plan(payload: dict) -> dict:
    defects = [Defect.model_validate(item) for item in payload.get("defects", [])]
    plan = build_repair_plan(defects, locked_refs=[str(item) for item in payload.get("locked_refs", [])])
    return plan.model_dump(mode="json")


@app.post("/api/v3/reviews/interpret")
async def interpret_v3_review(payload: dict) -> dict:
    text = str(payload.get("text", "")).strip()
    if not text:
        raise HTTPException(422, "评价不能为空")
    defects = defects_from_review(text, artwork_id=str(payload.get("artwork_id", "")))
    return {"defects": [item.model_dump(mode="json") for item in defects]}


@app.post("/api/jobs", response_model=JobResponse)
async def create_job(request: GenerateRequest) -> dict:
    if request.preparation_id is not None:
        try:
            return asdict(await creation_preparations.submit_generation(request, jobs))
        except OwnedCancellationUnconfirmed as exc:
            raise HTTPException(409, str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(409, str(exc)) from exc
    try:
        return asdict(await jobs.submit(request))
    except OwnedCancellationUnconfirmed as exc:
        raise HTTPException(409, str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(422,str(exc)) from exc


@app.post("/api/workflow-preview")
async def preview_workflow(request: GenerateRequest) -> dict:
    return jobs.preview(request)


@app.post("/api/matrix", response_model=list[JobResponse])
async def create_matrix(request: MatrixRequest) -> list[dict]:
    created = []
    for seed in request.seeds:
        for variant in request.variants:
            mode = TestMode.HANDS if variant == "hands" else request.base.mode
            prompt = (
                matrix_prompt(request.base.prompt, variant)
                if request.base.model_id == ModelProfile.NEWBIE
                else natural_matrix_prompt(request.base.prompt, variant)
            )
            item = request.base.model_copy(
                update={
                    "seed": seed,
                    "mode": mode,
                    "prompt": prompt,
                }
            )
            created.append(asdict(await jobs.submit(item)))
    return created


@app.get("/api/jobs", response_model=list[JobResponse])
async def list_jobs(limit: int = 24) -> list[dict]:
    """Return the recent cards needed by the UI, not the complete image history."""
    safe_limit = max(1, min(limit, 100))
    return [asdict(job) for job in jobs.list()[:safe_limit]]


@app.get("/api/artworks")
async def list_artworks(origin: str | None = None, limit: int = 100, offset: int = 0, query: str = "") -> list[dict]:
    if origin not in {None, "creation", "experiment"}:
        raise HTTPException(422, "origin 只能是 creation 或 experiment")
    if offset < 0 or len(query) > 1000:
        raise HTTPException(422, "offset 不能为负，搜索内容不能超过 1000 字")
    sync_artworks()
    return artworks.list(origin=origin, limit=max(1, min(limit, 500)), offset=offset, query=query)


def require_artwork(artwork_id: str):
    sync_artworks()
    artwork = artworks.get(artwork_id)
    if artwork is None:
        raise HTTPException(404, "作品不存在")
    return artwork


@app.get("/api/artworks/{artwork_id}")
async def get_artwork(artwork_id: str) -> dict:
    return artworks.serialize(require_artwork(artwork_id))


@app.get("/api/artworks/{artwork_id}/lineage")
async def get_artwork_lineage(artwork_id: str) -> dict:
    return artworks.lineage(require_artwork(artwork_id))


@app.post("/api/artworks/{artwork_id}/reviews")
async def add_artwork_review(artwork_id: str, request: ArtworkReviewRequest) -> dict:
    if request.image_index != 0:
        raise HTTPException(422, "作品评价不需要指定图片序号")
    return artworks.add_review(require_artwork(artwork_id), request.text, origin=request.origin)


@app.delete("/api/artworks/{artwork_id}/reviews/{review_id}")
async def delete_artwork_review(artwork_id: str, review_id: str) -> dict:
    try:
        artworks.delete_review(require_artwork(artwork_id), review_id)
    except KeyError as exc:
        raise HTTPException(404, "评价不存在") from exc
    return {"deleted": review_id}


@app.post("/api/artworks/{artwork_id}/text-layer")
async def add_artwork_text_layer(artwork_id: str, payload: dict) -> dict:
    try:
        spec = TextLayerSpec.model_validate(payload)
    except ValueError as exc:
        raise HTTPException(422, f"文字图层参数无效：{exc}") from exc
    parent = require_artwork(artwork_id)
    adaptive_run_id = str(payload.get("adaptive_run_id", "")).strip()
    adaptive_run = require_adaptive_run(adaptive_run_id) if adaptive_run_id else None
    intent_id = str(payload.get("intent_id", "")).strip()
    if adaptive_run is not None:
        try:
            adaptive_runs._require_user_boundary(adaptive_run)
        except ValueError as exc:
            raise HTTPException(409, str(exc)) from exc
        if adaptive_run.selected_artifact_id != artwork_id:
            raise HTTPException(409, "文字图层的原图不是协作任务当前选中版本")
        matching = [item.id for item in adaptive_runs.missing_exact_text_intents(adaptive_run) if item.content == spec.content]
        if not intent_id and len(matching) == 1:
            intent_id = matching[0]
        if intent_id not in matching:
            raise HTTPException(409, "文字内容必须与尚未完成的语义契约逐字一致")
    source = Path(parent.file_path).resolve()
    if not source.exists():
        raise HTTPException(404, "原图文件不存在")
    derivative_id = f"{artwork_id}-text-{uuid.uuid4().hex[:8]}"
    target_dir = generated_dir / "edits"
    target = target_dir / f"{derivative_id}.png"
    target_dir.mkdir(parents=True, exist_ok=True)
    manifest = await run_in_threadpool(render_text_layer, source, target, spec)
    if manifest.get("verification") != "deterministic_source_and_pixel_invariants_verified":
        target.unlink(missing_ok=True)
        raise HTTPException(500, "文字图层没有通过像素边界校验，未保存派生版本")
    manifest_path = target.with_suffix(".text.json")
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    relative = target.relative_to(generated_dir).as_posix()
    derivative = artworks.add_derivative(
        artwork_id=derivative_id, parent=parent, image_url=f"/generated/{relative}",
        file_path=str(target.resolve()), operation="deterministic_text_layer", instruction=spec.content,
        checks=[{"name": "精确文字", "status": "pass", "message": "内容由确定性文字图层写入；未要求扩散模型拼写。", "details": manifest}],
    )
    if adaptive_run is not None:
        try:
            adaptive_runs.record_text_artifact(
                adaptive_run,
                ArtifactRef(
                    id=derivative.id, kind="image", path=derivative.file_path,
                    producer_node="text_render", parent_ids=[artwork_id],
                    metadata={"image_url": derivative.image_url, "manifest": manifest},
                ),
                intent_id=intent_id, content=spec.content,
                verification=str(manifest.get("verification", "")),
            )
        except ValueError as exc:
            raise HTTPException(409, str(exc)) from exc
    return artworks.serialize(derivative)


@app.get("/api/jobs/{job_id}", response_model=JobResponse)
async def get_job(job_id: str) -> dict:
    job = jobs.get(job_id)
    if job is None:
        raise HTTPException(404, "任务不存在")
    return asdict(job)


@app.post("/api/jobs/{job_id}/cancel", response_model=JobResponse)
async def cancel_job(job_id: str) -> dict:
    try:
        job = await jobs.cancel(job_id)
    except (OwnedCancellationUnconfirmed, EngineReceiptPersistenceError) as exc:
        # The stop intent remains in the actual job; a rejected proof is not a
        # successful cancellation and must not become an opaque server error.
        raise HTTPException(409, str(exc)) from exc
    if job is None:
        raise HTTPException(404, "任务不存在")
    return asdict(job)


@app.post('/api/jobs/{job_id}/inspect-engine')
async def inspect_job_engine_receipt(job_id: str) -> dict:
    try:
        receipt = await jobs.inspect_engine_receipts(job_id)
    except (OwnedCancellationUnconfirmed, EngineReceiptPersistenceError, ValueError) as exc:
        raise HTTPException(409, str(exc)) from exc
    if receipt is None:
        raise HTTPException(404, '原任务不存在；没有创建或重派任务。')
    return receipt


@app.get("/api/jobs/{job_id}/workflow")
async def get_job_workflow(job_id: str) -> dict:
    job = jobs.get(job_id)
    if job is None:
        raise HTTPException(404, "任务不存在")
    try:
        return jobs.workflow_evidence(job)
    except (OSError, ValueError) as exc:
        raise HTTPException(500, f"无法读取工作流证据：{exc}") from exc


@app.post("/api/jobs/{job_id}/reveal")
async def reveal_job_file(job_id: str) -> dict:
    job = jobs.get(job_id)
    if job is None:
        raise HTTPException(404, "任务不存在")
    target = jobs.first_file(job)
    if target is None or not target.exists():
        raise HTTPException(404, "该任务没有可定位的图片文件")
    if platform.system() != "Windows":
        raise HTTPException(501, "当前系统暂不支持自动打开资源管理器")
    subprocess.Popen(["explorer.exe", f"/select,{target}"])
    return {"path": str(target)}


def require_experiment(experiment_id: str):
    experiment = experiments.get(experiment_id)
    if experiment is None:
        raise HTTPException(404, "实验不存在")
    return experiment


@app.get("/api/experiments")
async def list_experiments() -> list[dict]:
    return experiments.list()


@app.post("/api/experiments")
async def create_experiment(request: ExperimentCreateRequest) -> dict:
    return experiments.create(request)


@app.get("/api/experiment-designer/defaults")
async def get_experiment_designer_defaults() -> dict:
    return designer_defaults()


@app.post("/api/experiment-designer/preview")
async def preview_experiment_design(request: ExperimentDesignRequest) -> dict:
    return designer_preview(request)


@app.post("/api/experiment-designer/create")
async def create_from_experiment_design(request: ExperimentDesignRequest) -> dict:
    try:
        manifest = build_manifest(request)
    except ValueError as exc:
        raise HTTPException(409, str(exc)) from exc
    return experiments.create(manifest)


@app.get("/api/research/boundary-plan")
async def get_boundary_plan() -> dict:
    return boundary_plan_summary(load_boundary_plan())


@app.post("/api/research/boundary-plan/create")
async def create_boundary_plan(payload: dict) -> dict:
    if payload.get("confirmed") is not True:
        raise HTTPException(409, "必须先确认全部测试文本与管线；创建后仍保持暂停")
    try:
        manifest = build_boundary_manifest(load_boundary_plan())
    except (OSError, ValueError) as exc:
        raise HTTPException(422, f"全面边界矩阵无效：{exc}") from exc
    return experiments.create(manifest)


@app.get("/api/experiments/{experiment_id}")
async def get_experiment(experiment_id: str) -> dict:
    return experiments.serialize(require_experiment(experiment_id))


@app.get("/api/experiments/{experiment_id}/reviews")
async def get_experiment_reviews(experiment_id: str) -> list[dict]:
    experiment = require_experiment(experiment_id)
    return experiments.list_reviews(experiment)


@app.get("/api/experiments/{experiment_id}/analysis")
async def get_experiment_analysis(experiment_id: str) -> dict:
    sync_artworks()
    return analyze_experiment(require_experiment(experiment_id), artworks)


@app.get("/api/routing-guidance")
async def list_routing_guidance() -> list[dict]:
    return routing_guidance.list()


@app.post("/api/routing-guidance")
async def publish_routing_guidance(payload: dict) -> dict:
    experiment_id = str(payload.get("experiment_id", ""))
    experiment = require_experiment(experiment_id)
    sync_artworks()
    analysis = analyze_experiment(experiment, artworks)
    if not analysis["summary"]["conclusion_ready"]:
        raise HTTPException(409, "当前聚合仅是描述统计，不能据此发布生产路由；需要带适用范围和同条件对照的结论证据。已有研究和评价不受影响。")
    try:
        return routing_guidance.publish(
            experiment_id=experiment_id, trigger=str(payload.get("trigger", "")),
            pipeline=str(payload.get("pipeline", "")), note=str(payload.get("note", "")),
            evidence={"summary": analysis["summary"], "groups": {"model": analysis["groups"]["model"], "pipeline": analysis["groups"]["pipeline"]}},
        )
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc


@app.delete("/api/routing-guidance/{guidance_id}")
async def deactivate_routing_guidance(guidance_id: str) -> dict:
    try:
        return routing_guidance.deactivate(guidance_id)
    except KeyError as exc:
        raise HTTPException(404, "路由指导不存在") from exc


@app.post("/api/experiments/{experiment_id}/pause")
async def pause_experiment(experiment_id: str) -> dict:
    return experiments.set_state(require_experiment(experiment_id), "paused")


@app.post("/api/experiments/{experiment_id}/resume")
async def resume_experiment(experiment_id: str) -> dict:
    experiment = require_experiment(experiment_id)
    if experiment.state in {"completed", "cancelled"}:
        raise HTTPException(409, "已结束的实验不能恢复")
    return experiments.set_state(experiment, "running")


@app.post("/api/experiments/{experiment_id}/retry-failed")
async def retry_failed_experiment(experiment_id: str) -> dict:
    try:
        return experiments.retry_failed(require_experiment(experiment_id))
    except ValueError as exc:
        raise HTTPException(409, str(exc)) from exc


@app.post("/api/experiments/{experiment_id}/interrupt")
async def interrupt_experiment(experiment_id: str) -> dict:
    return await experiments.interrupt(require_experiment(experiment_id))


@app.post("/api/experiments/{experiment_id}/cancel")
async def cancel_experiment(experiment_id: str) -> dict:
    return await experiments.cancel(require_experiment(experiment_id))


@app.post("/api/experiments/{experiment_id}/items/{item_id}/cancel")
async def cancel_experiment_item(experiment_id: str, item_id: str) -> dict:
    try:
        return await experiments.cancel_item(require_experiment(experiment_id), item_id)
    except KeyError as exc:
        raise HTTPException(404, "实验项不存在") from exc


@app.post("/api/experiments/{experiment_id}/items/{item_id}/reviews")
async def add_experiment_review(experiment_id: str, item_id: str, request: ExperimentReviewRequest) -> dict:
    try:
        return experiments.add_review(require_experiment(experiment_id), item_id, request.text, request.image_index)
    except KeyError as exc:
        raise HTTPException(404, "实验项不存在") from exc
    except ValueError as exc:
        raise HTTPException(409, str(exc)) from exc


@app.delete("/api/experiments/{experiment_id}/items/{item_id}/reviews/{review_id}")
async def delete_experiment_review(experiment_id: str, item_id: str, review_id: str) -> dict:
    try:
        experiments.delete_review(require_experiment(experiment_id), item_id, review_id)
    except KeyError as exc:
        raise HTTPException(404, "实验项不存在") from exc
    except LookupError as exc:
        raise HTTPException(404, "评价不存在") from exc
    return {"deleted": review_id}
