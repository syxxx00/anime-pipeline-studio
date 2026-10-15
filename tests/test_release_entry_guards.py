"""Entry guards run before health requests, processes or model loading."""
import asyncio
from pathlib import Path

import pytest

from app.comfy_client import ComfyClient, EngineUnavailable
from app.config import Settings
from app.portable_runtime import local_runtime_config
import app.model_connections as connections
import app.native_grammar_profile as native


@pytest.mark.parametrize('url,port', [
    ('https://127.0.0.1:8190', 8190),
    ('http://[::1]:8190', 8190),
    ('http://localhost:8190', 8190),
    ('http://127.0.0.1:8190/prefix', 8190),
    ('http://127.0.0.1:8199', 8190),
])
def test_invalid_managed_engine_is_rejected_before_health_or_process(url, port, monkeypatch):
    client = ComfyClient(Settings(comfy_url=url, engine_port=port, auto_start_engine=True))
    calls = []
    async def forbidden_health():
        calls.append('network')
        raise AssertionError('Do not contact a wrong managed endpoint')
    monkeypatch.setattr(client, 'health', forbidden_health)
    monkeypatch.setattr('app.comfy_client.subprocess.Popen', lambda *a, **k: calls.append('process'))
    with pytest.raises(EngineUnavailable):
        asyncio.run(client.ensure_running())
    assert calls == []


def test_valid_managed_endpoint_keeps_existing_health_flow(monkeypatch):
    client = ComfyClient(Settings())
    async def healthy():
        return {'running': True}
    monkeypatch.setattr(client, 'health', healthy)
    monkeypatch.setattr('app.comfy_client.subprocess.Popen', lambda *a, **k: pytest.fail('Already running'))
    asyncio.run(client.ensure_running())


@pytest.mark.parametrize('url', ['https://127.0.0.1:8190', 'http://[::1]:8190', 'http://192.0.2.5:8190/prefix'])
def test_external_engine_is_not_subject_to_managed_ipv4_policy(url, monkeypatch):
    client = ComfyClient(Settings(comfy_url=url, auto_start_engine=False))
    async def healthy():
        return {'running': True}
    monkeypatch.setattr(client, 'health', healthy)
    asyncio.run(client.ensure_running())


def test_explicit_missing_grammar_directory_does_not_select_builtin(tmp_path):
    chosen = tmp_path / 'not-yet-registered'
    config = local_runtime_config(Settings(local_llm_grammar_dir=str(chosen)))
    assert config.native_grammar_directory == chosen
    assert not chosen.exists()


def test_default_missing_grammar_directory_can_use_builtin(tmp_path, monkeypatch):
    monkeypatch.setattr('app.portable_runtime.PROJECT_ROOT', tmp_path)
    config = local_runtime_config(Settings(local_llm_grammar_dir=''))
    assert config.native_grammar_directory is None


def _local_report_settings(tmp_path, monkeypatch):
    runner = tmp_path / 'runner/llama-server.exe'
    runner.parent.mkdir()
    runner.write_bytes(b'fake executable never started')
    model = tmp_path / 'model.gguf'
    projector = tmp_path / 'projector.gguf'
    model.write_bytes(b'fake model never read')
    projector.write_bytes(b'fake projector never read')
    monkeypatch.setattr(connections, 'model_catalog', lambda settings: [{'id': 'newbie', 'ready': True}])
    from app.creation_concept_context import CreationConceptContext
    class FakeConcepts:
        def load(self, selections):
            return {'available_documents': [1, 2, 3, 4]}
    monkeypatch.setattr(CreationConceptContext, 'for_settings', classmethod(lambda cls, settings: FakeConcepts()))
    return Settings(auto_start_engine=False, local_llm_server=str(runner), local_llm_model=str(model),
        local_llm_projector=str(projector), local_llm_grammar_dir=str(tmp_path / 'missing-registration'))


def test_local_missing_dlls_no_longer_report_ready_even_with_weight_files(tmp_path, monkeypatch):
    settings = _local_report_settings(tmp_path, monkeypatch)
    def forbidden(*args, **kwargs):
        raise AssertionError('File inspection may not execute native code or contact services')
    monkeypatch.setattr(native, '_native_parse', forbidden)
    monkeypatch.setattr(native.ctypes, 'CDLL', forbidden)
    monkeypatch.setattr(connections.httpx, 'AsyncClient', forbidden)
    result = connections.connection_report(settings)
    checks = {item['key']: item for item in result['checks']}
    assert checks['local_llm_model']['ok'] and checks['local_llm_projector']['ok']
    assert result['configuration_ready'] is False
    assert checks['local_runtime_grammar']['required'] and not checks['local_runtime_grammar']['ok']
    assert 'llama-server-impl.dll' in checks['local_runtime_grammar']['message']
    assert 'llama.dll' in checks['local_runtime_grammar']['message']
    assert result['local_runtime_readiness']['native_dll_loaded'] is False
    assert result['local_runtime_readiness']['profile_source'] == 'registered'


def test_readiness_report_preserves_explicit_directory_and_compatible_result(tmp_path, monkeypatch):
    settings = _local_report_settings(tmp_path, monkeypatch)
    calls = []
    def compatible(server, profile_root=None):
        calls.append((server, profile_root))
        return {'configuration_files_compatible': True, 'dlls': [], 'native_parse_verified': False}
    monkeypatch.setattr(native, 'inspect_local_grammar_runtime', compatible)
    result = connections.connection_report(settings)
    assert result['configuration_ready'] is True
    assert calls[0][1] == Path(settings.local_llm_grammar_dir)
    assert not result['local_runtime_readiness']['native_parse_verified']


def test_compatible_api_does_not_need_local_runtime_registration(tmp_path, monkeypatch):
    settings = _local_report_settings(tmp_path, monkeypatch)
    settings.authoring_backend = 'openai_compatible'
    settings.authoring_base_url = 'https://service.invalid/v1'
    settings.authoring_model = 'user-provided'
    monkeypatch.setattr(native, 'inspect_local_grammar_runtime', lambda *a, **k: pytest.fail('API mode must not inspect DLLs'))
    result = connections.connection_report(settings)
    assert result['configuration_ready'] is True
    assert result['local_runtime_readiness'] is None
