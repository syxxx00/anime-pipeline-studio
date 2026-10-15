"""Operator-owned model connections; saving never starts/downloads a model.

The current process retains its startup settings. New settings are persisted
with a revision guard and become active after restart. No model output can
choose a file, execute a command or invoke these operator endpoints.
"""
from __future__ import annotations

import asyncio
from dataclasses import asdict, fields
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import tempfile
import threading

import httpx
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field

from .config import DEFAULT_SETTINGS_PATH, PROJECT_ROOT, Settings, load_settings
from .model_catalog import model_catalog
from .service_endpoints import (validate_authoring_base_url, validate_http_service_url,
    validate_autostart_comfy_url)


GROUPS = [
    ('image_service', '图像引擎', [
        ('comfy_url', 'ComfyUI Core 地址', 'url'),
        ('auto_start_engine', '需要时自动启动本机 Core', 'boolean'),
        ('comfy_root', 'ComfyUI Core 源码目录', 'path'),
        ('comfy_python', 'Core 的 Python 可执行文件', 'path'),
        ('engine_base_dir', 'Core 工作目录', 'path'),
        ('model_root', '图像模型根目录', 'path'),
        ('engine_port', '本机 Core 端口', 'integer'),
        ('web_port', '网页端口', 'integer'),
    ]),
    ('authoring', '理解与观察模型', [
        ('authoring_backend', '接入方式', 'backend'),
        ('authoring_base_url', '兼容服务地址（含 /v1）', 'url_optional'),
        ('authoring_model', '理解与构思模型 ID', 'text'),
        ('authoring_vision_model', '视觉模型 ID（留空使用上面的模型）', 'text'),
        ('authoring_api_key_env', '保存 API Key 的环境变量名称', 'environment'),
        ('local_llm_server', 'llama-server.exe 路径', 'path_optional'),
        ('local_llm_model', '本地 GGUF 模型', 'path_optional'),
        ('local_llm_projector', '与模型配套的视觉 projector', 'path_optional'),
        ('local_llm_model_7b', '可选 7B GGUF／分片首文件', 'path_optional'),
        ('local_llm_model_9b', '可选 9B GGUF', 'path_optional'),
        ('local_llm_runtime_dir', '本地运行器工作目录', 'path_optional'),
        ('local_llm_evidence_dir', '调用记录目录', 'path_optional'),
        ('local_llm_grammar_dir', '登记的运行器语法配置目录', 'path_optional'),
        ('local_llm_port', '本地语言运行器端口', 'integer'),
    ]),
    ('newbie', 'NewBie 适配器', [
        ('diffusion_model', 'unet 下的扩散权重文件', 'model_file'),
        ('vae', 'vae 下的 VAE 文件', 'model_file'),
        ('gemma_clip', 'clip 下的 Gemma 编码器', 'model_file'),
        ('jina_clip', 'clip 下的 Jina 编码器', 'model_file'),
    ]),
    ('anima', 'Anima 适配器', [
        ('anima_diffusion_model', 'diffusion_models 下的权重文件', 'model_file'),
        ('anima_clip', 'text_encoders 下的编码器', 'model_file'),
        ('anima_vae', 'vae 下的 VAE 文件', 'model_file'),
    ]),
    ('illustrious', 'Illustrious／SDXL 适配器', [
        ('illustrious_checkpoint', 'checkpoints 下的兼容 checkpoint', 'model_file'),
    ]),
    ('z_image_turbo', 'Z-Image Turbo 适配器', [
        ('z_diffusion_model', 'diffusion_models 下的权重文件', 'model_file'),
        ('z_clip', 'text_encoders 下的编码器', 'model_file'),
        ('z_vae', 'vae 下的 VAE 文件', 'model_file'),
    ]),
    ('resources', '资源与作品', [
        ('style_catalog_path', '风格配方文件（只读取 styles）', 'path_optional'),
        ('concept_root', '可选概念资源根目录', 'path_optional'),
        ('output_dir', '作品输出目录', 'path'),
    ]),
]
EDITABLE = {key: kind for _, _, members in GROUPS for key, _, kind in members}
_WRITE_LOCK = threading.Lock()


class ConfigurationRevisionConflict(ValueError):
    """A save conflict is distinct from invalid operator input."""


class ConnectionPatch(BaseModel):
    model_config = ConfigDict(extra='forbid')
    values: dict[str, object] = Field(default_factory=dict, max_length=50)


class SaveConnections(ConnectionPatch):
    expected_revision: str = Field(min_length=1, max_length=64)


def _file_revision(path: Path) -> str:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except FileNotFoundError:
        return 'missing'


def _read_document(path: Path) -> dict:
    if not path.exists():
        return {}
    value = json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(value, dict):
        raise ValueError('本地配置必须是 JSON 对象。')
    return value


def _check_url(value: str, key: str):
    try:
        return (validate_authoring_base_url(value) if key == 'authoring_base_url'
                else validate_http_service_url(value))
    except ValueError as exc:
        raise ValueError(f'{key}: {exc}') from exc


def _validated_endpoints(settings: Settings) -> tuple[str, str | None]:
    """Validate the complete saved candidate before files, key reads or HTTP.

    GET may show an older unsafe value so its owner can correct it; an unrelated
    patch cannot bypass validation of that effective value for save or probe.
    """
    core = validate_http_service_url(settings.comfy_url)
    if settings.auto_start_engine:
        core = validate_autostart_comfy_url(settings.comfy_url, settings.engine_port)
    authoring = (validate_authoring_base_url(settings.authoring_base_url)
                 if settings.authoring_base_url else None)
    return core, authoring


def validate_patch(values: dict) -> dict:
    result = {}
    for key, value in values.items():
        if key not in EDITABLE:
            raise ValueError(f'不支持的接入配置字段：{key}')
        kind = EDITABLE[key]
        if kind == 'boolean':
            if type(value) is not bool:
                raise ValueError(f'{key} 必须是布尔值。')
        elif kind == 'integer':
            if type(value) is not int or not 1 <= value <= 65535:
                raise ValueError(f'{key} 必须是 1–65535 的整数端口。')
        else:
            if type(value) is not str or len(value) > 4096 or any(c in value for c in '\x00\r\n'):
                raise ValueError(f'{key} 必须是单行文本。')
            if kind not in {'url', 'url_optional'}:
                value = value.strip()
            if kind == 'backend' and value not in {'local_gguf', 'openai_compatible'}:
                raise ValueError('请选择本地 GGUF 或 OpenAI 兼容服务。')
            if kind == 'environment' and not re.fullmatch(r'[A-Za-z_][A-Za-z0-9_]*', value):
                raise ValueError('API Key 字段只接受环境变量名，不接受密钥本身。')
            if kind in {'url', 'url_optional'} and value:
                value = _check_url(value, key)
            if kind in {'url', 'path', 'model_file'} and not value:
                raise ValueError(f'{key} 不能为空。')
            if kind == 'model_file':
                model_path = PurePosixPath(value.replace('\\', '/'))
                if model_path.is_absolute() or ':' in value or '..' in model_path.parts:
                    raise ValueError(f'{key} 请填写对应模型分类内的相对文件名。')
                value = model_path.as_posix()
        result[key] = value
    return result


def candidate_settings(values: dict, path: Path) -> Settings:
    existing = asdict(load_settings(path))
    existing.update(validate_patch(values))
    candidate = Settings(**existing)
    _validated_endpoints(candidate)
    return candidate


def connection_report(settings: Settings) -> dict:
    """File/configuration inspection only, never a loading or quality claim."""
    _validated_endpoints(settings)
    checks = []
    runtime_readiness = None
    def check(key, ok, message, *, required=True):
        checks.append({'key': key, 'ok': bool(ok), 'required': required, 'message': message})
    catalog = model_catalog(settings)
    enabled_images = [item['id'] for item in catalog if item['ready']]
    check('image_models', bool(enabled_images),
          '文件齐备的适配器：' + ('、'.join(enabled_images) or '无；先准备至少一类图像模型'))
    if settings.auto_start_engine:
        for key, target in [('comfy_python', Path(settings.comfy_python)),
                            ('comfy_root', Path(settings.comfy_root) / 'main.py')]:
            check(key, target.is_file(), f'{key}: {target}')
        check('engine_port', True, '自动启动地址已匹配本机 IPv4 HTTP 端口。')
        check('engine_host', True, '自动启动使用 127.0.0.1；其他已有服务应关闭自动启动。')
    if settings.authoring_backend == 'openai_compatible':
        check('authoring_service', bool(settings.authoring_base_url and settings.authoring_model),
              '填写兼容服务地址与模型 ID；默认循环需要支持图片输入和结构化 JSON 的模型。')
        env_name = settings.authoring_api_key_env
        check('api_key_environment', bool(os.environ.get(env_name)),
              f'{env_name} ' + ('已设置（不返回密钥）' if os.environ.get(env_name) else '未设置；无需鉴权的本地服务可留空'),
              required=False)
    else:
        from .portable_runtime import local_runtime_config
        config = local_runtime_config(settings, purpose='vision')
        for key, target in [('local_llm_server', config.server_executable),
                            ('local_llm_model', config.model_path),
                            ('local_llm_projector', config.mmproj_path)]:
            check(key, target is not None and Path(target).is_file(), f'{key}: {target or "未配置"}')
        from .native_grammar_profile import inspect_local_grammar_runtime
        runtime_readiness = inspect_local_grammar_runtime(
            config.server_executable, profile_root=config.native_grammar_directory)
        compatible = runtime_readiness['configuration_files_compatible']
        unavailable = [item['name'] for item in runtime_readiness['dlls']
                       if item['status'] != 'readable']
        if compatible:
            message = '运行器 DLL 与语法文件配套；未加载 DLL、解析语法或运行模型。'
        elif unavailable:
            message = '运行器缺少或无法读取配套文件：' + '、'.join(unavailable) + '；请准备同一版本的完整运行器。'
        else:
            message = '运行器／语法文件不完整或绑定不匹配；请检查下方运行器路径，登记当前构建或选择正确的语法目录。'
        check('local_runtime_grammar', compatible, message)
    from .creation_concept_context import CreationConceptContext
    try:
        packet = CreationConceptContext.for_settings(settings).load([])
        check('concepts', True, f'概念原件 {len(packet["available_documents"])} 份可读取。')
    except (OSError, ValueError) as exc:
        check('concepts', False, str(exc))
    check('styles', Path(settings.style_catalog_path).is_file(),
          '风格配方存在。' if Path(settings.style_catalog_path).is_file() else '未提供风格配方；仍可使用描述中的画风。', required=False)
    return {'scope': 'configuration_and_files_only_no_models_loaded',
            'configuration_ready': all(item['ok'] for item in checks if item['required']),
            'checks': checks, 'image_adapters': catalog,
            'local_runtime_readiness': runtime_readiness,
            'note': '文件存在不证明架构兼容或推理成功；服务探测也不生成图片。'}


def save_connections(values: dict, expected_revision: str, path: Path = DEFAULT_SETTINGS_PATH) -> str:
    patch = validate_patch(values)
    with _WRITE_LOCK:
        if _file_revision(path) != expected_revision:
            raise ConfigurationRevisionConflict('配置已被其他操作修改，请重新读取后保存。')
        document = _read_document(path)
        document.update(patch)
        # Validate the effective dataclass without changing the live process.
        candidate_settings(patch, path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, temporary = tempfile.mkstemp(prefix='.settings-', suffix='.tmp', dir=path.parent)
        try:
            with os.fdopen(fd, 'w', encoding='utf-8', newline='\n') as handle:
                json.dump(document, handle, ensure_ascii=False, indent=2)
                handle.write('\n')
            if _file_revision(path) != expected_revision:
                raise ConfigurationRevisionConflict('保存期间配置发生变化，未覆盖新配置。')
            os.replace(temporary, path)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
    return _file_revision(path)


async def probe_services(settings: Settings) -> dict:
    """An explicit operator action: GET capabilities, not inference/download."""
    core, authoring = _validated_endpoints(settings)
    results = []
    requests = [('comfy', core + '/object_info', None)]
    if settings.authoring_backend == 'openai_compatible' and authoring:
        key = os.environ.get(settings.authoring_api_key_env, '')
        requests.append(('authoring', authoring + '/models',
                         {'Authorization': 'Bearer ' + key} if key else None))
    async with httpx.AsyncClient(timeout=8, trust_env=False, follow_redirects=False) as client:
        for name, url, headers in requests:
            try:
                response = await client.get(url, headers=headers)
                response.raise_for_status()
                payload = response.json()
                if name == 'comfy':
                    if not isinstance(payload, dict):
                        raise ValueError('object_info 没有返回节点对象。')
                    results.append({'service': name, 'reachable': True, 'node_count': len(payload),
                                    'message': 'Core 节点目录可读取；未加载图像模型。'})
                else:
                    entries = payload.get('data', []) if isinstance(payload, dict) else []
                    ids = [item['id'] for item in entries if isinstance(item, dict) and isinstance(item.get('id'), str)]
                    results.append({'service': name, 'reachable': True, 'model_ids': ids[:200],
                                    'message': '兼容服务模型列表可读取；未进行文字或视觉推理。'})
            except (httpx.HTTPError, ValueError) as exc:
                # Do not return headers, raw service responses, tokens or a
                # remote body's possibly sensitive authentication diagnostic.
                status = getattr(getattr(exc, 'response', None), 'status_code', None)
                results.append({'service': name, 'reachable': False, 'http_status': status,
                                'message': '服务探测失败；检查地址、服务是否启动及环境变量鉴权。'})
    return {'scope': 'explicit_get_only_no_inference_no_autostart', 'services': results}


def _same_origin(request: Request):
    origin = request.headers.get('origin')
    if origin and origin.rstrip('/') != str(request.base_url).rstrip('/'):
        raise HTTPException(403, '模型接入配置只接受本页面的同源操作。')


def model_connections_router(active_settings: Settings, *, settings_path: Path = DEFAULT_SETTINGS_PATH) -> APIRouter:
    router = APIRouter(prefix='/api/model-connections', tags=['model-connections'])
    @router.get('')
    async def read_connections():
        try:
            configured = load_settings(settings_path)
        except (ValueError, OSError) as exc:
            raise HTTPException(422, '本地配置不可读取或无效；请核对 settings.json。') from exc
        members = {field.name for field in fields(Settings)}
        values = {key: value for key, value in asdict(configured).items() if key in EDITABLE and key in members}
        active = {key: value for key, value in asdict(active_settings).items() if key in values}
        return {'revision': _file_revision(settings_path), 'values': values,
                'restart_required': active != values,
                'groups': [{'id': key, 'title': title, 'fields': [
                    {'key': field, 'label': label, 'kind': kind} for field, label, kind in items]}
                    for key, title, items in GROUPS],
                'application': 'anime-pipeline-studio', 'key_values_returned': False}
    @router.post('/validate')
    async def validate_connections(payload: ConnectionPatch, request: Request):
        _same_origin(request)
        try:
            configured = candidate_settings(payload.values, settings_path)
            return await asyncio.to_thread(connection_report, configured)
        except (ValueError, OSError) as exc:
            raise HTTPException(422, str(exc)) from exc
    @router.post('/probe')
    async def probe_connections(payload: ConnectionPatch, request: Request):
        _same_origin(request)
        try:
            return await probe_services(candidate_settings(payload.values, settings_path))
        except (ValueError, OSError) as exc:
            raise HTTPException(422, str(exc)) from exc
    @router.put('')
    async def update_connections(payload: SaveConnections, request: Request):
        _same_origin(request)
        try:
            revision = await asyncio.to_thread(save_connections, payload.values, payload.expected_revision, settings_path)
            return {'saved': True, 'revision': revision, 'restart_required': True,
                    'message': '已保存本机配置。停止后重新启动项目生效；未加载模型或改变当前任务。'}
        except ConfigurationRevisionConflict as exc:
            raise HTTPException(409, str(exc)) from exc
        except (ValueError, OSError) as exc:
            raise HTTPException(422, str(exc)) from exc
    @router.post('/bind-runtime')
    async def bind_runtime(payload: ConnectionPatch, request: Request):
        _same_origin(request)
        try:
            configured = candidate_settings(payload.values, settings_path)
            from .portable_runtime import local_runtime_config
            from .native_grammar_profile import bind_local_grammar_runtime
            config = local_runtime_config(configured)
            destination = Path(configured.local_llm_grammar_dir) if configured.local_llm_grammar_dir else PROJECT_ROOT / 'config/local_grammars'
            receipt = await asyncio.to_thread(bind_local_grammar_runtime, config.server_executable, destination)
            return {'registered': True, 'receipt': receipt,
                    'message': '已检查并登记此运行器的语法兼容配置；未加载 GGUF 或进行推理。'}
        except (ValueError, OSError) as exc:
            raise HTTPException(422, str(exc)) from exc
    return router
