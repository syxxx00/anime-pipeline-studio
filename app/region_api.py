from dataclasses import asdict

from fastapi import APIRouter, HTTPException

from .region_edit import generation_ancestor, prepare_region_edit
from .schemas import JobResponse


def region_router(jobs, artworks, settings, *, require_artwork, require_adaptive_run, adaptive_runs):
    router = APIRouter()

    @router.post('/api/artworks/{artwork_id}/region-repair', response_model=JobResponse)
    async def region_repair_artwork(artwork_id: str, payload: dict):
        artwork = require_artwork(artwork_id)
        parent_job = generation_ancestor(artwork, artworks, jobs)
        if parent_job is None:
            raise HTTPException(409, '找不到此版本或其祖先的生成参数')
        run_id = str(payload.get('adaptive_run_id', '')).strip()
        run = require_adaptive_run(run_id) if run_id else None
        if run is not None and run.selected_artifact_id != artwork_id:
            raise HTTPException(409, '圈选作品不是协作任务当前选中的版本')
        if run is not None:
            try:
                adaptive_runs._require_user_boundary(run)
            except ValueError as exc:
                raise HTTPException(409, str(exc)) from exc
        try:
            request, workflow, inputs = prepare_region_edit(artwork, parent_job, payload, settings)
        except (OSError, ValueError) as exc:
            raise HTTPException(422, f'局部修改参数或图片无效：{exc}') from exc
        try:
            job = await jobs.submit_workflow(request, workflow)
        except Exception:
            for path in inputs:
                path.unlink(missing_ok=True)
            raise
        if run is not None:
            try:
                if run.selected_artifact_id != artwork_id:
                    raise ValueError('协作任务已切换作品，本次修改未接入任务')
                adaptive_runs.attach_repair_job(run, job.id, origin='user_mask', input_files=[str(path) for path in inputs])
            except ValueError as exc:
                await jobs.cancel(job.id)
                for path in inputs:
                    path.unlink(missing_ok=True)
                raise HTTPException(409, str(exc)) from exc
            if run.state.status == 'cancelled':
                await jobs.cancel(job.id)
                for path in inputs:
                    path.unlink(missing_ok=True)
        return asdict(job)

    return router
