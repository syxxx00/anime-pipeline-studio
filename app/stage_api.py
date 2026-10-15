"""Stage browsing is read-only; saving a stage as artwork requires an explicit POST."""
from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import Response


def stage_router(jobs, artworks) -> APIRouter:
    router = APIRouter()

    def require(job_id, artifact_id=None):
        job = jobs.get(job_id)
        if job is None:
            raise HTTPException(404, '任务不存在')
        if artifact_id is None:
            return job, None
        try:
            record = jobs.stage_record(job, artifact_id)
        except (OSError, ValueError, KeyError) as exc:
            raise HTTPException(409, f'阶段证据不可用：{exc}') from exc
        if record is None:
            raise HTTPException(404, '阶段图片不存在')
        return job, record

    @router.get('/api/jobs/{job_id}/stages')
    async def stages(job_id: str):
        job, _ = require(job_id)
        try:
            return jobs.stage_evidence(job)
        except (OSError, ValueError, KeyError) as exc:
            raise HTTPException(409, f'阶段记录读取失败：{exc}') from exc

    @router.get('/api/jobs/{job_id}/stages/{artifact_id}/image')
    async def image(job_id: str, artifact_id: str, download: bool = False):
        _, record = require(job_id, artifact_id)
        try:
            payload = jobs.stage_images.read_bytes(record)
        except (OSError, ValueError) as exc:
            raise HTTPException(409, f'阶段图片不可用：{exc}') from exc
        headers = {'Cache-Control': 'no-store'}
        if download:
            headers['Content-Disposition'] = f'attachment; filename="{job_id}__{artifact_id}.png"'
        return Response(content=payload, media_type='image/png', headers=headers)

    @router.get('/api/jobs/{job_id}/stages/{artifact_id}/workflow')
    async def workflow(job_id: str, artifact_id: str):
        job, record = require(job_id, artifact_id)
        if not record.get('workflow'):
            raise HTTPException(404, '此图片没有单独的阶段工作流，请查看任务完整工作流')
        return {'job_id': job_id, 'status': job.status, 'source': record['evidence'],
                'workflow': record['workflow'], 'warning': record.get('warning', ''),
                'request': job.request}

    @router.post('/api/jobs/{job_id}/stages/{artifact_id}/retain')
    async def retain(job_id: str, artifact_id: str):
        job, record = require(job_id, artifact_id)
        if record['evidence'] == 'delivered':
            raise HTTPException(409, '最终图已是作品，无需另存阶段')
        try:
            payload = jobs.stage_images.read_bytes(record)
            if record['evidence'] == 'cache_match':
                record = jobs.stage_images.retain(
                    job_id, artifact_id, record['label'], record['model'], record['workflow'],
                    {'source_file': record['file_path'], 'source_artifact_id': artifact_id}, payload,
                    evidence='retained_cache_match', warning=record['warning'])
            relative = Path(record['file_path']).relative_to(jobs.stage_images.output_root).as_posix()
            artwork = artworks.add_retained_stage(job, record, f'/generated/{relative}')
            return artworks.serialize(artwork)
        except (OSError, ValueError) as exc:
            raise HTTPException(409, f'无法保留阶段图片：{exc}') from exc

    return router
