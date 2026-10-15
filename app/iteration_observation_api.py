"""Explicit original-evidence archives, not observation-model or quality calls."""
from __future__ import annotations

import asyncio
from hashlib import sha256
from io import BytesIO
import json
from typing import Annotated

from fastapi import APIRouter, HTTPException, Response
from PIL import Image
from pydantic import BaseModel, ConfigDict, Field, StrictInt, model_validator
from starlette.concurrency import run_in_threadpool

from .iteration_observation import StaleObservationError, _archive_json, _json_bytes


ObservationID = Annotated[str, Field(strict=True, pattern=r'^[A-Za-z0-9][A-Za-z0-9_-]{0,79}$')]
ArtworkID = Annotated[str, Field(strict=True, pattern=r'^[A-Za-z0-9][A-Za-z0-9_-]{0,119}$')]
Role = Annotated[str, Field(strict=True, pattern=r'^[a-z][a-z0-9_]{0,39}$')]
Digest = Annotated[str, Field(strict=True, pattern=r'^[0-9a-f]{64}$')]
ArtifactList = Annotated[list[ArtworkID], Field(min_length=1, max_length=16)]


class ObservationRequest(BaseModel):
    model_config = ConfigDict(strict=True, extra='forbid')

    observation_id: ObservationID
    revision: Annotated[StrictInt, Field(ge=0)]
    decision_count: Annotated[StrictInt, Field(ge=0)]
    feedback_signature: Digest
    selected_artifacts: Annotated[dict[Role, ArtifactList], Field(max_length=16)]
    expected_image_sha256: Annotated[dict[ArtworkID, Digest], Field(max_length=16)] | None = None

    @model_validator(mode='after')
    def bounded_selected_hashes(self):
        selected = {item for values in self.selected_artifacts.values() for item in values}
        if len(selected) > 16:
            raise ValueError('最多选择 16 个原图，不会自动省略或缩小原件')
        if not set(self.expected_image_sha256 or {}) <= selected:
            raise ValueError('expected_image_sha256 只能绑定本次实际选择的作品 ID')
        return self


_API_RECEIPT = 'api_request.json'
_MAX_API_RECEIPT_BYTES = 64 * 1024
_IMAGE_MIME = {'PNG': 'image/png', 'JPEG': 'image/jpeg', 'WEBP': 'image/webp',
               'GIF': 'image/gif', 'BMP': 'image/bmp', 'TIFF': 'image/tiff'}


def iteration_observation_router(controller, observations):
    """Register before the iteration router's generic POST control route.

    Only an explicit POST freezes originals. Reads are independent of model
    availability and of whether the live run still exists. No run/control lock
    is held while synchronous archive work runs in the thread pool.
    """
    router = APIRouter()
    locks = {}

    def require_observations():
        if observations is None:
            raise HTTPException(503, '原件归档能力尚未接入，不能猜测已经检查或自动创建观察包')

    def current_context(run_id):
        run = controller.store.runs.get(run_id)
        if run is None:
            raise HTTPException(404, '找不到迭代任务')
        try:
            # Detach live references before yielding to archive worker threads.
            return json.loads(_json_bytes(controller._context(run)))
        except (OSError, ValueError) as error:
            raise HTTPException(409, '不能取得完整当前原件上下文：' + str(error)) from error

    def check_request_context(payload, context):
        if any(getattr(payload, key) != context[key]
               for key in ('revision', 'decision_count', 'feedback_signature')):
            raise HTTPException(409, '任务版本、决策数或自然语言评价已变化，请重新读取当前上下文')

    def conflict(observation_id, error, *, retained=False, currentness='not_checked'):
        return HTTPException(409, {'observation_id': observation_id,
            'archive_retained': retained, 'currentness': currentness,
            'error': str(error), 'automatic_retry': False,
            'quality_verified': False, 'semantic_understanding_verified': False})

    def read_archive(observation_id, run_id):
        try:
            folder = observations._folder(observation_id)
        except ValueError as error:
            raise conflict(observation_id, error) from error
        if not folder.exists():
            raise HTTPException(404, '找不到该原件观察包；GET 不会创建它')
        try:
            return observations.read_archive(observation_id, expected_run_id=run_id)
        except ValueError as error:
            raise conflict(observation_id, error, retained=True) from error

    def read_api_receipt(archive):
        manifest = archive['manifest']
        folder = observations._folder(manifest['id'])
        path = folder / _API_RECEIPT
        # A missing request receipt is not proof that an API POST completed.
        if not path.exists() and not path.is_symlink():
            return None
        try:
            receipt = _archive_json(observations._archive_file(folder, _API_RECEIPT, _MAX_API_RECEIPT_BYTES))
            manifest_bytes = observations._archive_file(folder, 'manifest.json', observations.limits.max_json_bytes)
            request = ObservationRequest.model_validate(receipt['request']).model_dump(mode='json')
            if (set(receipt) != {'version', 'observation_id', 'run_id', 'manifest_sha256', 'request_sha256', 'request'}
                    or type(receipt['version']) is not int or receipt['version'] != 1
                    or receipt['observation_id'] != manifest['id'] or receipt['run_id'] != manifest['run_id']
                    or receipt['manifest_sha256'] != sha256(manifest_bytes).hexdigest()
                    or receipt['request_sha256'] != sha256(_json_bytes(request)).hexdigest()
                    or request['observation_id'] != manifest['id']
                    or request['revision'] != manifest['revision']
                    or request['decision_count'] != archive['context']['decision_count']
                    or request['feedback_signature'] != manifest['feedback_signature']
                    or observations._selection(request['selected_artifacts']) != manifest['roles']
                    or any(manifest['images'][key]['sha256'] != value
                           for key, value in (request['expected_image_sha256'] or {}).items())):
                raise ValueError('API 请求收据与已冻结原件不一致')
            return receipt
        except (OSError, KeyError, TypeError, ValueError) as error:
            raise conflict(manifest['id'], error, retained=True) from error

    def checked_archive(observation_id, run_id):
        archive = read_archive(observation_id, run_id)
        receipt = read_api_receipt(archive)
        archive['api_request'] = None if receipt is None else receipt['request']
        archive['api_receipt_verified'] = receipt is not None
        return archive

    def prepare_new(payload, context):
        selected = observations._selection(payload.selected_artifacts)
        artifacts = {item['id']: item for item in context['artifacts']}
        if any(key not in artifacts for values in selected.values() for key in values):
            raise HTTPException(422, '所选作品必须是该迭代任务已有的实际作品 ID')
        for artwork_id, expected in (payload.expected_image_sha256 or {}).items():
            actual, _ = observations._image(artifacts[artwork_id])
            if actual['sha256'] != expected:
                raise conflict(payload.observation_id, '所选原图 SHA 已变化：' + artwork_id)
        manifest = observations.freeze(context, selected, snapshot_id=payload.observation_id,
                                       expected_image_sha256=payload.expected_image_sha256)
        # The freeze itself re-reads originals. Recheck hashes against those
        # exact frozen bytes, not just the earlier source-file read.
        for artwork_id, expected in (payload.expected_image_sha256 or {}).items():
            if manifest['images'][artwork_id]['sha256'] != expected:
                raise conflict(payload.observation_id, '冻结期间所选原图 SHA 已变化：' + artwork_id, retained=True)
        folder = observations._folder(payload.observation_id)
        manifest_bytes = observations._archive_file(folder, 'manifest.json', observations.limits.max_json_bytes)
        request = payload.model_dump(mode='json')
        receipt = {'version': 1, 'observation_id': payload.observation_id, 'run_id': context['run_id'],
            'manifest_sha256': sha256(manifest_bytes).hexdigest(),
            'request_sha256': sha256(_json_bytes(request)).hexdigest(), 'request': request}
        data = _json_bytes(receipt)
        if len(data) > _MAX_API_RECEIPT_BYTES:
            raise ValueError('完整 API 请求收据超过容量；原件归档保留但不猜测 POST 已完成')
        # Exclusive write: an interrupted or unknown receipt is never repaired,
        # replaced or removed automatically.
        with (folder / _API_RECEIPT).open('xb') as stream:
            stream.write(data)
        return checked_archive(payload.observation_id, context['run_id'])

    def urls(archive):
        manifest = archive['manifest']
        archive['image_urls'] = {key: f"/api/iteration-runs/{manifest['run_id']}/observations/{manifest['id']}/images/{key}"
                                 for key in manifest['images']}
        return archive

    async def archive_response(archive):
        # A complete archive can be large. Do not move only disk IO off the
        # loop and then serialize all original requests/prose on that loop.
        body = await run_in_threadpool(_json_bytes, urls(archive))
        return Response(body, media_type='application/json', headers={'Cache-Control': 'no-store'})

    async def check_current(archive):
        run_id, observation_id = archive['manifest']['run_id'], archive['manifest']['id']
        try:
            context = current_context(run_id)
        except HTTPException as error:
            archive['currentness'], archive['current_error'] = 'not_checked', str(error.detail)
            return archive
        try:
            await run_in_threadpool(observations.validate_still_current, observation_id, context)
            # Controls and feedback can run during the threaded read. Check the
            # authoritative revision/count/prose again before returning.
            latest = current_context(run_id)
            if any(latest[key] != context[key] for key in ('revision', 'decision_count', 'feedback_signature')):
                raise StaleObservationError('核验期间任务控制或用户评价发生变化')
            archive['currentness'], archive['current_error'] = 'current', None
        except StaleObservationError as error:
            archive['currentness'], archive['current_error'] = 'stale', str(error)
        except (HTTPException, OSError, ValueError) as error:
            archive['currentness'], archive['current_error'] = 'not_checked', str(getattr(error, 'detail', error))
        return archive

    @router.post('/api/iteration-runs/{run_id}/observations')
    async def create_observation(run_id: str, payload: ObservationRequest):
        require_observations()
        # Only duplicate freeze requests serialize; pause/cancel do not share
        # this lock, nor do unrelated IDs or model dispatches.
        async with locks.setdefault(payload.observation_id, asyncio.Lock()):
            context = current_context(run_id)
            check_request_context(payload, context)
            try:
                folder = observations._folder(payload.observation_id)
                reused = folder.exists()
                if reused:
                    archive = await run_in_threadpool(checked_archive, payload.observation_id, run_id)
                    if archive['api_request'] is None or archive['api_request'] != payload.model_dump(mode='json'):
                        raise conflict(payload.observation_id, '该 ID 已用于另一请求，或原 API 请求收据不完整；不会覆盖', retained=True)
                else:
                    archive = await run_in_threadpool(prepare_new, payload, context)
            except StaleObservationError as error:
                raise conflict(payload.observation_id, error,
                               retained=observations._folder(payload.observation_id).exists()) from error
            except FileExistsError as error:
                raise conflict(payload.observation_id, '该 ID 已存在但归属/收据尚未核验：' + str(error), retained=True) from error
            except OSError as error:
                raise HTTPException(503, {'observation_id': payload.observation_id,
                    'archive_retained': observations._folder(payload.observation_id).exists(),
                    'error': str(error), 'automatic_retry': False}) from error
            except ValueError as error:
                raise HTTPException(422, str(error)) from error
            archive = await check_current(archive)
            if archive['currentness'] != 'current':
                raise conflict(payload.observation_id, archive['current_error'], retained=True,
                               currentness=archive['currentness'])
            # Even a complete frozen package is not permission to dispatch;
            # the decision endpoint/executor independently revalidate it.
            archive['reused'] = reused
            return await archive_response(archive)

    @router.get('/api/iteration-runs/{run_id}/observations/{observation_id}')
    async def get_observation(run_id: str, observation_id: ObservationID):
        require_observations()
        archive = await run_in_threadpool(checked_archive, observation_id, run_id)
        return await archive_response(await check_current(archive))

    def image_data(run_id, observation_id, artwork_id):
        archive = read_archive(observation_id, run_id)
        image = archive['manifest']['images'].get(artwork_id)
        if image is None:
            raise HTTPException(404, '该作品不是此观察包实际选择的原图')
        try:
            data = observations.read_image_bytes(observation_id, expected_run_id=run_id, artwork_id=artwork_id)
            if sha256(data).hexdigest() != image['sha256']:
                raise ValueError('图片与归档中绑定的 SHA 不一致')
            with Image.open(BytesIO(data)) as original:
                if (original.format != image['format'] or original.size != (image['width'], image['height'])
                        or original.format not in _IMAGE_MIME):
                    raise ValueError('归档图像格式或尺寸不能保守验证')
                original.verify()
            return data, _IMAGE_MIME[image['format']], image['sha256']
        except (OSError, SyntaxError, ValueError) as error:
            raise conflict(observation_id, error, retained=True) from error

    @router.get('/api/iteration-runs/{run_id}/observations/{observation_id}/images/{artwork_id}')
    async def get_image(run_id: str, observation_id: ObservationID, artwork_id: ArtworkID):
        require_observations()
        data, mime, digest = await run_in_threadpool(image_data, run_id, observation_id, artwork_id)
        return Response(data, media_type=mime, headers={'Cache-Control': 'no-store',
            'ETag': '"' + digest + '"', 'X-Content-SHA256': digest, 'X-Content-Type-Options': 'nosniff'})

    return router
