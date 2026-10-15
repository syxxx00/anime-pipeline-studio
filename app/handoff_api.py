from dataclasses import asdict

from fastapi import APIRouter, HTTPException

from .image_handoff import cleanup_handoff_input, prepare_handoff
from .region_edit import generation_ancestor


def handoff_router(jobs, artworks, settings, *, require_artwork):
    router = APIRouter()

    @router.post('/api/artworks/{artwork_id}/handoff-preview')
    async def preview_handoff(artwork_id: str, payload: dict):
        request,workflow = prepare(artwork_id,payload,False)
        return {'request':request.model_dump(mode='json'),'workflow':workflow,
                'warning':'仅预览一次像素交接，不代表该模型和强度已通过效果测试。'}

    def prepare(artwork_id, payload, materialize):
        artwork = require_artwork(artwork_id)
        parent = generation_ancestor(artwork,artworks,jobs)
        if parent is None:
            raise HTTPException(409,'找不到当前作品或其祖先的生成文本，请先恢复作品生成记录')
        try:
            return prepare_handoff(artwork,parent,payload,settings,materialize=materialize)
        except (OSError,ValueError) as exc:
            raise HTTPException(422,str(exc)) from exc

    @router.post('/api/artworks/{artwork_id}/handoff')
    async def submit_handoff(artwork_id: str, payload: dict):
        request,workflow = prepare(artwork_id,payload,True)
        try:
            return asdict(await jobs.submit_workflow(request,workflow))
        except Exception:
            cleanup_handoff_input(request,settings)
            raise

    return router
