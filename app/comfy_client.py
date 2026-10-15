from __future__ import annotations

import asyncio
import base64
from copy import deepcopy
import json
import hashlib
import os
import re
import subprocess
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import httpx

from .config import PROJECT_ROOT, Settings


class EngineUnavailable(RuntimeError):
    pass


class OwnedCancellationUnconfirmed(RuntimeError):
    """No proof that the bound engine prompt has left running/pending."""


def local_engine_environment(settings: Settings | None = None) -> dict[str, str]:
    """Existing local resources only; ordinary startup cannot enable Hub fetches.

    This is child-process configuration, not a network sandbox or permission to
    install optional node dependencies. Keep the caller's environment intact.
    """
    settings = settings or Settings()
    cache_root = Path(settings.engine_base_dir).parent / "cache"
    hub_root = cache_root / "huggingface"
    env = os.environ.copy()
    env.update(HF_HOME=str(hub_root), HF_HUB_CACHE=str(hub_root / 'hub'),
        HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', HF_HUB_DISABLE_TELEMETRY='1',
        PIP_NO_INDEX='1', PIP_DISABLE_PIP_VERSION_CHECK='1')
    for name, folder in (
        ('TORCH_HOME', 'torch_cache'), ('XDG_CACHE_HOME', 'cache'),
        ('PYTHONPYCACHEPREFIX', 'pycache'), ('TORCHINDUCTOR_CACHE_DIR', 'torchinductor_cache'),
        ('TORCH_EXTENSIONS_DIR', 'torch_extensions'), ('TRITON_CACHE_DIR', 'triton_cache'),
        ('CUDA_CACHE_PATH', 'cuda_cache')):
        env[name] = str(cache_root / folder)
    temporary = cache_root / 'temp'
    env.update(TEMP=str(temporary), TMP=str(temporary))
    return env


def engine_extra_model_paths(settings: Settings) -> dict[str, dict[str, str]]:
    """One source for application model checks and Core model/extension lookup."""
    return {
        "newbie_lab_models": {
            "base_path": str(Path(settings.model_root)), "checkpoints": "checkpoints",
            "diffusion_models": "diffusion_models\nunet", "text_encoders": "text_encoders\nclip",
            "vae": "vae", "loras": "loras", "ultralytics": "ultralytics",
            "ultralytics_bbox": "ultralytics/bbox", "ultralytics_segm": "ultralytics/segm",
        },
        "ai_comic_engine_extensions": {"base_path": str(PROJECT_ROOT), "custom_nodes": "engine_extensions"},
    }


def _write_json_atomic(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    try:
        temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _process_creation_iso(process: subprocess.Popen) -> str | None:
    """Windows process creation time, not the wall clock when a receipt is written."""
    if os.name != 'nt':
        return None
    handle = getattr(process, '_handle', None)
    if handle is None:
        return None
    import ctypes
    from ctypes import wintypes
    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    get_times = kernel.GetProcessTimes
    get_times.argtypes = [wintypes.HANDLE] + [ctypes.POINTER(wintypes.FILETIME)] * 4
    get_times.restype = wintypes.BOOL
    creation, exit_time, system, user = (wintypes.FILETIME() for _ in range(4))
    if not get_times(wintypes.HANDLE(int(handle)), ctypes.byref(creation), ctypes.byref(exit_time),
                     ctypes.byref(system), ctypes.byref(user)):
        return None
    ticks = (creation.dwHighDateTime << 32) | creation.dwLowDateTime
    return datetime.fromtimestamp((ticks - 116444736000000000) / 10_000_000, timezone.utc).isoformat()


class EngineSubmissionRejected(RuntimeError):
    """The inspected Core's complete validation response proves no queue put.

    Only the three top-level validate_prompt rejections qualify. Other HTTP
    failures, partial bodies and responses with any prompt_id remain unknown.
    This is rejection evidence, never a generation or cancellation receipt.
    """

    @classmethod
    def from_response(cls, response: httpx.Response):
        if response.status_code != 400:
            return None
        try:
            body = response.json()
        except (ValueError, UnicodeError):
            return None
        if not isinstance(body, dict) or set(body) != {'error', 'node_errors'}:
            return None
        error = body['error']
        if (not isinstance(error, dict)
                or set(error) != {'type', 'message', 'details', 'extra_info'}
                or not isinstance(error['type'], str)
                or error['type'] not in {'missing_node_type', 'prompt_no_outputs',
                                         'prompt_outputs_failed_validation'}
                or not isinstance(error['message'], str) or not error['message']
                or not isinstance(error['details'], str)
                or not isinstance(error['extra_info'], dict)
                or not isinstance(body['node_errors'], dict)):
            return None
        rejection = cls('引擎在入队前拒绝工作流：' + error['message']
                        + ('；' + error['details'] if error['details'] else ''))
        rejection.response_evidence = {
            'http_status': response.status_code, 'response_json': deepcopy(body),
            'response_body_base64': base64.b64encode(response.content).decode('ascii'),
            'response_text': response.text,
            'evidence_scope': 'inspected_core_validation_rejection_before_prompt_queue_put',
        }
        return rejection


class ComfyClient:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._process: subprocess.Popen | None = None
        self._client_id = str(uuid.uuid4())
        self._dispatch_backend_instance_id: str | None = None

    async def health(self) -> dict:
        try:
            async with httpx.AsyncClient(timeout=5, trust_env=False) as client:
                response = await client.get(f"{self.settings.comfy_url}/system_stats")
                response.raise_for_status()
                return {"running": True, "stats": response.json()}
        except Exception as exc:
            return {"running": False, "reason": str(exc)}

    def _command(self) -> list[str]:
        python = Path(self.settings.comfy_python)
        main = Path(self.settings.comfy_root) / "main.py"
        extra_paths = self._extra_paths_file()
        command = [
            str(python),
            str(main),
            "--base-directory",
            self.settings.engine_base_dir,
            "--listen",
            "127.0.0.1",
            "--port",
            str(self.settings.engine_port),
            "--disable-auto-launch",
            "--extra-model-paths-config",
            str(extra_paths),
        ]
        if self.settings.vram_mode in {"lowvram", "novram"}:
            command.append(f"--{self.settings.vram_mode}")
        # Keep the original engine/node loading policy; explicitly whitelist
        # the project's control extension even when an existing launcher uses
        # --disable-all-custom-nodes. Its path is in extra_model_paths.yaml.
        command.extend(["--whitelist-custom-nodes", "ComfyUI-Newbie-Nodes", "ai_comic_owned_control"])
        return command

    def _extra_paths_file(self) -> Path:
        return PROJECT_ROOT / 'runtime' / 'generated' / 'extra_model_paths.yaml'

    def _write_extra_paths(self) -> Path:
        # JSON is a YAML 1.2 subset, including string escaping. This avoids an
        # installer-only PyYAML dependency and unsafe interpolation of paths.
        path = self._extra_paths_file()
        _write_json_atomic(path, engine_extra_model_paths(self.settings))
        return path

    def _save_process_receipt(self, command: list[str]) -> None:
        if self._process is None:
            return
        started_at = _process_creation_iso(self._process)
        if started_at is None:
            return  # Never issue an ownership receipt with a guessed creation time.
        _write_json_atomic(PROJECT_ROOT / 'data' / 'engine-process.json', {
            'pid': self._process.pid, 'started_at': started_at, 'project_root': str(PROJECT_ROOT),
            'role': 'engine', 'command_line': command, 'engine_port': self.settings.engine_port,
            'comfy_root': self.settings.comfy_root, 'engine_base_dir': self.settings.engine_base_dir,
        })

    async def ensure_running(self, timeout: float = 90) -> None:
        if self.settings.auto_start_engine:
            from .service_endpoints import validate_autostart_comfy_url
            try:
                validate_autostart_comfy_url(self.settings.comfy_url, self.settings.engine_port)
            except ValueError as exc:
                raise EngineUnavailable(str(exc)) from exc
        if (await self.health())["running"]:
            return
        if not self.settings.auto_start_engine:
            raise EngineUnavailable("推理引擎未运行，且自动启动已关闭")
        command = self._command()
        if not Path(command[0]).exists():
            raise EngineUnavailable(f"独立 Python 环境不存在：{command[0]}")
        if not Path(command[1]).exists():
            raise EngineUnavailable(f"独立 ComfyUI Core 不存在：{command[1]}")
        self._write_extra_paths()
        env = local_engine_environment(self.settings)
        Path(env['TEMP']).mkdir(parents=True, exist_ok=True)
        logs = PROJECT_ROOT / "data" / "engine.log"
        logs.parent.mkdir(parents=True, exist_ok=True)
        stream = logs.open("a", encoding="utf-8")
        self._process = subprocess.Popen(
            command,
            cwd=self.settings.comfy_root,
            env=local_engine_environment(self.settings),
            stdout=stream,
            stderr=subprocess.STDOUT,
            creationflags=subprocess.CREATE_NO_WINDOW if hasattr(subprocess, "CREATE_NO_WINDOW") else 0,
        )
        self._save_process_receipt(command)
        started = time.monotonic()
        while time.monotonic() - started < timeout:
            if self._process.poll() is not None:
                raise EngineUnavailable(f"推理引擎启动失败，退出码 {self._process.returncode}")
            if (await self.health())["running"]:
                return
            await asyncio.sleep(1)
        raise EngineUnavailable("推理引擎启动超时，请查看 data/engine.log")

    async def queue(self, workflow: dict, *, before_dispatch=None) -> str:
        await self.ensure_running()
        payload = {"prompt": workflow, "client_id": self._client_id}
        async with httpx.AsyncClient(timeout=30, trust_env=False) as client:
            # Only checked project dispatches need a persistent backend binding.
            # Legacy engines stay usable but are not upgraded into exact
            # cross-restart recovery. Never infer identity from an empty queue.
            instance = await self._read_backend_instance(client) if before_dispatch is not None else None
            self._dispatch_backend_instance_id = instance
            # Synchronous final gate: health/context entry may have yielded to
            # a stop request. No await may separate this gate from POST start.
            if before_dispatch is not None:
                before_dispatch()
            response = await client.post(f"{self.settings.comfy_url}/prompt", json=payload)
            if response.is_error:
                rejection = EngineSubmissionRejected.from_response(response)
                if rejection is not None:
                    raise rejection
                raise RuntimeError(f"工作流提交失败：{response.text}")
            return response.json()["prompt_id"]

    async def _read_backend_instance(self, client) -> str | None:
        try:
            response = await client.get(self.settings.comfy_url.rstrip('/') +
                '/ai-comic/owned-control/capabilities', timeout=3)
            if response.status_code != 200:
                return None
            body = response.json()
            if (not isinstance(body, dict) or body.get('protocol') != 'ai-comic-owned-control-v1'
                    or any(body.get(key) is not True for key in
                        ('owner_checks', 'atomic_queue_lock', 'atomic_graph_owner_checks', 'prompt_inspection'))):
                return None
            instance = body.get('instance_id')
            return instance if isinstance(instance, str) and str(uuid.UUID(instance)) == instance else None
        except (httpx.HTTPError, ValueError):
            return None

    @staticmethod
    def _original_binding(prompt_id, owner_client_id, workflow_sha256, backend_instance_id) -> dict:
        if (not isinstance(prompt_id, str) or not re.fullmatch(r'[a-zA-Z0-9_-]{1,160}', prompt_id)
                or not isinstance(owner_client_id, str) or not 1 <= len(owner_client_id) <= 160
                or not owner_client_id.strip()
                or not isinstance(workflow_sha256, str) or not re.fullmatch(r'[a-f0-9]{64}', workflow_sha256)):
            raise OwnedCancellationUnconfirmed('缺少确切原任务／提交者／工作流绑定，未核对或停止其他任务。')
        try:
            if not isinstance(backend_instance_id, str) or str(uuid.UUID(backend_instance_id)) != backend_instance_id:
                raise ValueError('Invalid original backend instance')
        except (ValueError, AttributeError) as exc:
            raise OwnedCancellationUnconfirmed('原后端实例未绑定，未用当前后端猜补历史。') from exc
        return {'client_id': owner_client_id, 'workflow_sha256': workflow_sha256,
                'instance_id': backend_instance_id}

    async def inspect_owned(self, prompt_id: str, *, owner_client_id: str,
                            workflow_sha256: str, backend_instance_id: str) -> dict:
        """Explicitly read the exact original task; never restart or resume it."""
        binding = self._original_binding(prompt_id, owner_client_id, workflow_sha256, backend_instance_id)
        try:
            async with httpx.AsyncClient(timeout=5, trust_env=False) as client:
                response = await client.post(self.settings.comfy_url.rstrip('/') +
                    '/ai-comic/prompts/' + prompt_id + '/inspect', json=binding)
                response.raise_for_status()
                receipt = response.json()
            if (not isinstance(receipt, dict) or receipt.get('prompt_id') != prompt_id
                    or receipt.get('client_id') != owner_client_id
                    or receipt.get('instance_id') != backend_instance_id
                    or receipt.get('workflow_sha256') != workflow_sha256
                    or receipt.get('ownership_verified') is not True
                    or receipt.get('cancelled') is not False
                    or receipt.get('state') not in {'running', 'pending', 'terminal'}):
                raise ValueError('Exact original inspection receipt did not match')
            history = receipt.get('history')
            if receipt['state'] == 'terminal':
                prompt = history.get('prompt') if isinstance(history, dict) else None
                if (not isinstance(prompt, (list, tuple)) or len(prompt) < 4
                        or prompt[1] != prompt_id or not isinstance(prompt[2], dict)
                        or not isinstance(prompt[3], dict) or prompt[3].get('client_id') != owner_client_id):
                    raise ValueError('Original terminal history is incomplete or mismatched')
                digest = hashlib.sha256(json.dumps(prompt[2], ensure_ascii=False, sort_keys=True,
                    allow_nan=False, separators=(',', ':')).encode('utf-8')).hexdigest()
                if digest != workflow_sha256:
                    raise ValueError('Original terminal workflow has changed')
            elif history is not None:
                raise ValueError('A live prompt cannot be reported as terminal history')
            return deepcopy(receipt)
        except Exception as exc:
            raise OwnedCancellationUnconfirmed('原任务核对未确认；没有重派、接收图片或解除未知状态：'
                + (str(exc) or type(exc).__name__)) from exc

    async def wait(self, prompt_id: str, timeout: float = 3600) -> dict:
        started = time.monotonic()
        failures = 0
        async with httpx.AsyncClient(timeout=15, trust_env=False) as client:
            while time.monotonic() - started < timeout:
                try:
                    response = await client.get(f"{self.settings.comfy_url}/history/{prompt_id}")
                    response.raise_for_status()
                except httpx.HTTPError as exc:
                    # 引擎在换载模型/崩溃重启时会短暂不响应；持续约 3 分钟无响应才判定失败
                    failures += 1
                    if failures >= 90:
                        raise RuntimeError(f"引擎状态查询连续失败（{type(exc).__name__}），引擎可能已崩溃") from exc
                    await asyncio.sleep(2)
                    continue
                if failures:
                    # 曾经断连后恢复 = 引擎重启过；确认任务没有随重启丢失
                    failures = 0
                    queue_response = await client.get(f"{self.settings.comfy_url}/queue")
                    queue_data = queue_response.json()
                    known = [entry[1] for entry in queue_data.get("queue_running", [])]
                    known += [entry[1] for entry in queue_data.get("queue_pending", [])]
                    if prompt_id not in known:
                        raise RuntimeError("引擎崩溃重启后，排队中的任务已丢失；请重新提交该任务")
                data = response.json()
                if prompt_id in data:
                    return data[prompt_id]
                await asyncio.sleep(1)
        raise TimeoutError("生成任务超过一小时仍未完成")

    async def view(self, filename: str, subfolder: str = "", image_type: str = "output") -> bytes:
        params = {"filename": filename, "subfolder": subfolder, "type": image_type}
        async with httpx.AsyncClient(timeout=60, trust_env=False) as client:
            response = await client.get(f"{self.settings.comfy_url}/view", params=params)
            response.raise_for_status()
            return response.content

    async def interrupt(self) -> None:
        """Interrupt the prompt currently executing in the headless engine."""
        async with httpx.AsyncClient(timeout=15, trust_env=False) as client:
            response = await client.post(f"{self.settings.comfy_url}/interrupt")
            response.raise_for_status()

    async def cancel_owned(self, prompt_id: str, *, timeout: float = 8,
                           owner_client_id: str | None = None,
                           workflow_sha256: str | None = None,
                           backend_instance_id: str | None = None) -> dict:
        """Cancel only an acknowledged engine prompt; never global fallback.

        Prompt ids are engine ids, NOT this application's job ids. The local
        owned-control extension checks the submitting client under queue mutex.
        Engines without that endpoint fail closed. A successful
        cancel POST is only dispatch acknowledgement, not proof of completion.
        """
        if not isinstance(prompt_id, str) or not re.fullmatch(r'[a-zA-Z0-9_-]{1,160}', prompt_id):
            raise OwnedCancellationUnconfirmed('缺少有效的已绑定引擎 prompt_id，未取消其他任务。')
        exact = any(value is not None for value in (owner_client_id, workflow_sha256, backend_instance_id))
        binding = self._original_binding(prompt_id, owner_client_id, workflow_sha256, backend_instance_id) if exact else {'client_id': self._client_id}
        owner = binding['client_id']
        try:
            async with asyncio.timeout(timeout), httpx.AsyncClient(timeout=min(5, timeout), trust_env=False) as client:
                while True:
                    # Re-request only this bound owner/id while it remains
                    # queued. This covers the engine's execute-entry flag reset;
                    # it is not a generation resubmission or global interrupt.
                    response = await client.post(self.settings.comfy_url.rstrip('/') + '/ai-comic/prompts/' + prompt_id + '/cancel',
                                                 json=binding)
                    response.raise_for_status()
                    acknowledgement = response.json()
                    if (not isinstance(acknowledgement, dict) or type(acknowledgement.get('cancelled')) is not bool
                            or acknowledgement.get('prompt_id') != prompt_id
                            or acknowledgement.get('client_id') != owner
                            or acknowledgement.get('ownership_verified') is not True
                            or acknowledgement.get('state') not in {'pending_removed', 'interrupt_requested', 'already_finished'}):
                        raise ValueError('原子取消回执格式不完整')
                    if exact and (acknowledgement.get('instance_id') != backend_instance_id
                                  or acknowledgement.get('workflow_sha256') != workflow_sha256):
                        raise ValueError('原后端实例／工作流取消回执未匹配')
                    response = await client.get(self.settings.comfy_url.rstrip('/') + '/queue')
                    response.raise_for_status()
                    queue = response.json()
                    if (not isinstance(queue, dict) or not all(isinstance(queue.get(key), list)
                            for key in ('queue_running', 'queue_pending'))):
                        raise ValueError('无法核对引擎真实运行及待处理队列')
                    entries = queue['queue_running'] + queue['queue_pending']
                    if any(not isinstance(item, (list, tuple)) or len(item) < 2
                           or not isinstance(item[1], str) for item in entries):
                        raise ValueError('引擎队列条目缺少确切 prompt_id')
                    if prompt_id not in {item[1] for item in entries}:
                        return {'prompt_id': prompt_id, 'client_id': owner,
                                'ownership_verified': True, 'cancel_requested': True,
                                'backend_cancelled_acknowledgement': acknowledgement['cancelled'],
                                'backend_state': acknowledgement['state'],
                                'absent_from_running_and_pending': True,
                                **{key: deepcopy(acknowledgement[key])
                                   for key in ('instance_id', 'workflow_sha256')
                                   if key in acknowledgement}}
                    await asyncio.sleep(.1)
        except Exception as exc:
            raise OwnedCancellationUnconfirmed('原任务取消未确认；未全局中断或触碰其他任务：'
                                               + (str(exc) or type(exc).__name__)) from exc

    async def free_memory(self) -> None:
        """Unload the completed stage before loading another large model."""
        async with httpx.AsyncClient(timeout=30, trust_env=False) as client:
            response = await client.post(f"{self.settings.comfy_url}/free", json={"unload_models": True, "free_memory": True})
            response.raise_for_status()

    async def runtime_status(self) -> dict:
        health = await self.health()
        if not health.get("running"):
            return {"running": False, "sampling_percent": 0, "step": 0, "steps": 0, "queue_running": 0, "queue_pending": 0, "log": []}
        try:
            async with httpx.AsyncClient(timeout=5, trust_env=False) as client:
                queue = (await client.get(f"{self.settings.comfy_url}/queue")).json()
        except Exception:
            queue = {"queue_running": [], "queue_pending": []}
        log_path = PROJECT_ROOT / "data" / "engine.log"
        text = ""
        if log_path.exists():
            with log_path.open("rb") as stream:
                stream.seek(max(0, log_path.stat().st_size - 128_000))
                text = stream.read().decode("utf-8", errors="replace")
        segment = text[text.rfind("got prompt") :] if "got prompt" in text else text
        matches = re.findall(r"(\d+)%\|[^\r\n]*?\|\s*(\d+)/(\d+)", segment)
        percent, step, steps = (map(int, matches[-1]) if matches else (0, 0, 0))
        lines = [line.strip() for line in re.split(r"[\r\n]+", segment) if line.strip()][-14:]
        devices = health.get("stats", {}).get("devices", [])
        system = health.get("stats", {}).get("system", {})
        return {
            "running": True,
            "sampling_percent": percent,
            "step": step,
            "steps": steps,
            "queue_running": len(queue.get("queue_running", [])),
            "queue_pending": len(queue.get("queue_pending", [])),
            "device": devices[0] if devices else {},
            "system": {"ram_total": system.get("ram_total"), "ram_free": system.get("ram_free")},
            "log": lines,
        }


def extract_output_images(history: dict) -> list[dict[str, str]]:
    images: list[dict[str, str]] = []
    for node in history.get("outputs", {}).values():
        images.extend(node.get("images", []))
    return images
