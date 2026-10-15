"""Isolated operator API tests: temporary settings and fake HTTP only."""
import asyncio
import hashlib
import json
from pathlib import Path

import httpx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.config import Settings
import app.model_connections as module


@pytest.mark.parametrize('values', [
    {'web_port': True}, {'engine_port': '8190'}, {'local_llm_port': 65536},
    {'auto_start_engine': 1}, {'auto_start_engine': 'false'}, {'model_root': []},
    {'authoring_backend': 'magic'}, {'unregistered': 'value'}, {'output_dir': ''},
    {'authoring_api_key_env': 'sk-raw-key'}, {'authoring_api_key_env': ''},
    {'authoring_api_key_env': 'VAR\nSECRET'}, {'comfy_root': 'a\x00b'},
    {'diffusion_model': '../outside.safetensors'}, {'z_clip': 'C:\\outside.safetensors'},
    {'anima_clip': '/absolute.safetensors'}, {'z_vae': '\\\\server\\vae.safetensors'},
    {'comfy_url': 'file:///tmp/model'}, {'comfy_url': 'https://user:secret@host.invalid'},
    {'comfy_url': 'https://host.invalid?api_key=secret'}, {'comfy_url': 'http://host.invalid:99999'},
    {'comfy_url': 'http://bad host.invalid'}, {'comfy_url': 'http://bad\thost.invalid'},
    {'authoring_base_url': 'https://host.invalid/v1#secret'},
])
def test_patch_strict_types_model_names_urls_and_environment_names(values):
    with pytest.raises(ValueError):
        module.validate_patch(values)


def test_relative_model_names_and_environment_not_secret_values():
    values = module.validate_patch({'diffusion_model': 'family\\model.safetensors',
        'authoring_api_key_env': 'AI_COMIC_KEY_2', 'local_llm_server': 'runtime/llama/llama-server.exe',
        'authoring_base_url': 'https://compatible.invalid/v1/'})
    assert values['diffusion_model'] == 'family/model.safetensors'
    assert values['authoring_api_key_env'] == 'AI_COMIC_KEY_2'
    assert values['authoring_base_url'] == 'https://compatible.invalid/v1'


def test_save_cas_is_atomic_preserves_unknown_and_never_touches_personal_config(tmp_path):
    path = tmp_path / 'settings.json'
    initial = {'web_port': 8100, 'future_setting': {'keep': [1, '中文']}, 'llm_model': 'legacy'}
    path.write_text(json.dumps(initial), encoding='utf-8')
    personal = module.DEFAULT_SETTINGS_PATH.read_bytes() if module.DEFAULT_SETTINGS_PATH.exists() else None
    revision = module._file_revision(path)
    updated = module.save_connections({'web_port': 8110, 'model_root': 'weights'}, revision, path)
    document = json.loads(path.read_text())
    assert document['future_setting'] == initial['future_setting']
    assert document['llm_model'] == 'legacy'
    assert document['model_root'] == 'weights' and document['web_port'] == 8110
    assert updated == hashlib.sha256(path.read_bytes()).hexdigest() and updated != revision
    with pytest.raises(module.ConfigurationRevisionConflict):
        module.save_connections({'web_port': 8120}, revision, path)
    assert module._file_revision(path) == updated
    assert not list(tmp_path.glob('.settings-*.tmp'))
    if personal is not None:
        assert module.DEFAULT_SETTINGS_PATH.read_bytes() == personal


def test_save_missing_file_and_conflict_during_write_preserve_newer_document(tmp_path, monkeypatch):
    path = tmp_path / 'new/settings.json'
    revision = module.save_connections({'web_port': 8101}, 'missing', path)
    assert json.loads(path.read_text())['web_port'] == 8101
    real_revision = module._file_revision
    calls = 0
    def changing_revision(target):
        nonlocal calls
        calls += 1
        if calls == 2:
            target.write_text('{"web_port":8102,"external_update":true}', encoding='utf-8')
        return real_revision(target)
    monkeypatch.setattr(module, '_file_revision', changing_revision)
    with pytest.raises(module.ConfigurationRevisionConflict):
        module.save_connections({'web_port': 8103}, revision, path)
    assert json.loads(path.read_text()) == {'web_port': 8102, 'external_update': True}
    assert not list(path.parent.glob('.settings-*.tmp'))


@pytest.fixture
def api(tmp_path):
    path = tmp_path / 'settings.json'
    active = Settings(model_root=str(tmp_path / 'weights'))
    app = FastAPI()
    app.include_router(module.model_connections_router(active, settings_path=path))
    with TestClient(app) as client:
        yield client, path, active


def test_read_route_returns_config_and_environment_name_without_secret(api, monkeypatch):
    client, path, active = api
    monkeypatch.setenv('AI_COMIC_AUTHORING_API_KEY', 'sk-never-return-this')
    path.write_text(json.dumps({'authoring_api_key_env': 'AI_COMIC_AUTHORING_API_KEY',
        'future_secret': 'unknown-not-public'}), encoding='utf-8')
    response = client.get('/api/model-connections')
    assert response.status_code == 200
    payload = response.json()
    assert payload['application'] == 'anime-pipeline-studio'
    assert payload['key_values_returned'] is False
    assert payload['values']['authoring_api_key_env'] == 'AI_COMIC_AUTHORING_API_KEY'
    assert 'sk-never-return-this' not in response.text and 'unknown-not-public' not in response.text
    assert any(group['id'] == 'authoring' for group in payload['groups'])


def test_validate_is_only_configuration_inspection_no_start_network_or_inference(api, monkeypatch):
    client, path, active = api
    import subprocess
    def forbidden(*args, **kwargs):
        raise AssertionError('validation must not start models or contact any service')
    monkeypatch.setattr(subprocess, 'Popen', forbidden)
    monkeypatch.setattr(module.httpx, 'AsyncClient', forbidden)
    response = client.post('/api/model-connections/validate', json={'values': {
        'auto_start_engine': False, 'authoring_backend': 'openai_compatible',
        'authoring_base_url': 'https://compatible.invalid/v1', 'authoring_model': 'operator-chosen',
        'concept_root': str(path.parent / 'missing-concepts')}})
    assert response.status_code == 200
    assert response.json()['scope'] == 'configuration_and_files_only_no_models_loaded'
    assert path.exists() is False


def test_put_routes_validation_as_422_and_revision_conflict_as_409(api):
    client, path, active = api
    invalid = client.put('/api/model-connections', json={'expected_revision': 'missing',
        'values': {'engine_port': '8190'}})
    assert invalid.status_code == 422 and not path.exists()
    first = client.put('/api/model-connections', json={'expected_revision': 'missing', 'values': {'web_port': 8111}})
    assert first.status_code == 200 and first.json()['restart_required']
    assert active.web_port == 8090  # Saving is not a live-settings mutation or restart.
    stale = client.put('/api/model-connections', json={'expected_revision': 'missing', 'values': {'web_port': 8112}})
    assert stale.status_code == 409 and json.loads(path.read_text())['web_port'] == 8111


@pytest.mark.parametrize('verb,route,body', [
    ('post', '/validate', {'values': {}}), ('post', '/probe', {'values': {}}),
    ('post', '/bind-runtime', {'values': {}}),
    ('put', '', {'values': {}, 'expected_revision': 'missing'}),
])
def test_operator_write_and_probe_routes_reject_cross_origin(api, verb, route, body):
    client, path, active = api
    response = getattr(client, verb)('/api/model-connections' + route,
        headers={'Origin': 'https://other.invalid'}, json=body)
    assert response.status_code == 403 and not path.exists()


def test_same_origin_and_invalid_body_route_errors_are_explicit(api):
    client, path, active = api
    response = client.put('/api/model-connections', headers={'Origin': 'http://testserver'},
        json={'expected_revision': 'missing', 'values': {'web_port': 8111}})
    assert response.status_code == 200
    assert client.post('/api/model-connections/validate', json={'values': {}, 'extra': 'not allowed'}).status_code == 422
    assert client.put('/api/model-connections', json={'values': {}}).status_code == 422
    path.write_text('{bad json containing private details', encoding='utf-8')
    bad = client.get('/api/model-connections')
    assert bad.status_code == 422 and 'private details' not in bad.text
    path.write_text('[]', encoding='utf-8')
    assert client.get('/api/model-connections').status_code == 422


def test_explicit_probe_only_gets_capabilities_and_keeps_authorization_private(monkeypatch):
    real_async_client = httpx.AsyncClient
    seen = []
    monkeypatch.setenv('TEST_SERVICE_KEY', 'sk-private-sentinel')
    def handler(request):
        seen.append(request)
        if request.url.host == 'comfy.invalid':
            return httpx.Response(200, json={'KSampler': {'input': {}}})
        return httpx.Response(200, json={'data': [{'id': 'chosen-model'}]})
    def fake_client(**kwargs):
        assert kwargs['trust_env'] is False and kwargs['follow_redirects'] is False
        return real_async_client(transport=httpx.MockTransport(handler), **kwargs)
    monkeypatch.setattr(module.httpx, 'AsyncClient', fake_client)
    settings = Settings(comfy_url='http://comfy.invalid:8188', auto_start_engine=False,
        authoring_backend='openai_compatible', authoring_base_url='https://compatible.invalid/v1',
        authoring_model='chosen-model', authoring_api_key_env='TEST_SERVICE_KEY')
    result = asyncio.run(module.probe_services(settings))
    assert [(request.method, request.url.path) for request in seen] == [('GET', '/object_info'), ('GET', '/v1/models')]
    assert 'Authorization' not in seen[0].headers
    assert seen[1].headers['Authorization'] == 'Bearer sk-private-sentinel'
    assert 'sk-private-sentinel' not in json.dumps(result)
    assert result['services'][1]['model_ids'] == ['chosen-model']


def test_failed_probe_hides_auth_response_and_never_follows_redirect(monkeypatch):
    real_async_client = httpx.AsyncClient
    seen = []
    def handler(request):
        seen.append(request)
        if request.url.host == 'comfy.invalid':
            return httpx.Response(307, headers={'Location': 'https://untrusted.invalid/collect'})
        return httpx.Response(401, json={'error': 'private token sk-sensitive-auth-trace'})
    monkeypatch.setattr(module.httpx, 'AsyncClient', lambda **kwargs:
        real_async_client(transport=httpx.MockTransport(handler), **kwargs))
    settings = Settings(comfy_url='http://comfy.invalid', auto_start_engine=False, authoring_backend='openai_compatible',
        authoring_base_url='https://compatible.invalid/v1', authoring_model='chosen')
    result = asyncio.run(module.probe_services(settings))
    assert len(seen) == 2 and all(not service['reachable'] for service in result['services'])
    assert [service['http_status'] for service in result['services']] == [307, 401]
    assert 'sk-sensitive-auth-trace' not in json.dumps(result)
    assert not any(request.url.host == 'untrusted.invalid' for request in seen)


def test_bind_route_only_invokes_explicit_registration_with_selected_paths(api, monkeypatch):
    client, path, active = api
    import app.native_grammar_profile as grammar
    calls = []
    def bind(executable, destination):
        calls.append((executable, destination))
        return {'fixture_registration': True}
    monkeypatch.setattr(grammar, 'bind_local_grammar_runtime', bind)
    runner = path.parent / 'operator-runner.exe'
    destination = path.parent / 'grammar records'
    response = client.post('/api/model-connections/bind-runtime', json={'values': {
        'local_llm_server': str(runner), 'local_llm_grammar_dir': str(destination)}})
    assert response.status_code == 200 and response.json()['registered'] is True
    assert calls == [(runner, destination)]
    assert not path.exists() and not destination.exists()


@pytest.mark.parametrize('route', ['/validate', '/probe', '/bind-runtime'])
def test_operator_routes_reject_invalid_configuration_before_execution(api, route):
    client, path, active = api
    response = client.post('/api/model-connections' + route, json={'values': {'raw_api_key': 'not-allowed'}})
    assert response.status_code == 422 and not path.exists()


@pytest.mark.parametrize('url', [
    'http://remote.invalid/v1', 'http://192.0.2.1:1234/v1',
    'https://user:placeholder@remote.invalid/v1', 'https://remote.invalid/v1?token=placeholder',
    'https://remote.invalid/v1#fragment', ' https://remote.invalid/v1',
    'https://remote.invalid/a b', 'https://remote.invalid/v1/models',
    'https://remote.invalid/v1/chat/completions', 'https://remote.invalid/v1/../admin',
])
def test_authoring_urls_rejected_by_patch_before_persistence(url):
    with pytest.raises(ValueError) as caught:
        module.validate_patch({'authoring_base_url': url})
    assert 'placeholder' not in str(caught.value)


def test_old_remote_http_is_rejected_before_probe_client_or_key_read(monkeypatch):
    settings = Settings(authoring_backend='openai_compatible',
        authoring_base_url='http://authoring.invalid/v1', authoring_model='operator-model')
    calls = []
    def forbidden(*args, **kwargs):
        calls.append(True)
        raise AssertionError('Unsafe authoring URL must be rejected before HTTP or reading any key')
    original_get = module.os.environ.get
    monkeypatch.setattr(module.httpx, 'AsyncClient', forbidden)
    monkeypatch.setattr(module.os.environ, 'get', lambda name, *args:
        forbidden() if name == settings.authoring_api_key_env else original_get(name, *args))
    with pytest.raises(ValueError, match='HTTPS'):
        asyncio.run(module.probe_services(settings))
    assert calls == []


def test_report_rejects_unsafe_effective_url_before_files_or_key_read(monkeypatch):
    settings = Settings(authoring_backend='openai_compatible',
        authoring_base_url='http://authoring.invalid/v1', authoring_model='operator-model')
    calls = []
    def forbidden(*args, **kwargs):
        calls.append(True)
        raise AssertionError('Report must reject unsafe endpoints before resource inspection or key reads')
    original_get = module.os.environ.get
    monkeypatch.setattr(module, 'model_catalog', forbidden)
    monkeypatch.setattr(module.os.environ, 'get', lambda name, *args:
        forbidden() if name == settings.authoring_api_key_env else original_get(name, *args))
    with pytest.raises(ValueError, match='HTTPS'):
        module.connection_report(settings)
    assert calls == []


def test_old_unsafe_setting_can_be_read_but_unrelated_save_and_probe_cannot_bypass(api, monkeypatch):
    client, path, active = api
    path.write_text(json.dumps({'authoring_backend': 'openai_compatible',
        'authoring_base_url': 'http://authoring.invalid/v1', 'authoring_model': 'chosen'}), encoding='utf-8')
    original = path.read_bytes()
    read = client.get('/api/model-connections')
    assert read.status_code == 200
    assert read.json()['values']['authoring_base_url'].startswith('http://')
    def forbidden(*args, **kwargs):
        raise AssertionError('Unsafe saved endpoint must not reach an HTTP client')
    monkeypatch.setattr(module.httpx, 'AsyncClient', forbidden)
    assert client.post('/api/model-connections/probe', json={'values': {}}).status_code == 422
    assert client.post('/api/model-connections/validate', json={'values': {}}).status_code == 422
    save = client.put('/api/model-connections', json={'expected_revision': read.json()['revision'],
        'values': {'web_port': 8110}})
    assert save.status_code == 422 and path.read_bytes() == original
    repaired = client.put('/api/model-connections', json={'expected_revision': read.json()['revision'],
        'values': {'authoring_base_url': 'https://authoring.invalid/v1', 'web_port': 8110}})
    assert repaired.status_code == 200
    assert json.loads(path.read_text())['authoring_base_url'].startswith('https://')


@pytest.mark.parametrize('url', [
    'http://localhost:8190', 'http://[::1]:8190', 'https://127.0.0.1:8190',
    'http://127.0.0.1:8190/prefix', 'http://127.0.0.1:8191',
])
def test_unlaunchable_autostart_addresses_rejected_before_probe_or_key_read(url, monkeypatch):
    settings = Settings()
    settings.comfy_url = url  # Reproduce an old saved/in-memory configuration.
    def forbidden(*args, **kwargs):
        raise AssertionError('An unlaunchable Core endpoint must not cause HTTP or key reads')
    original_get = module.os.environ.get
    monkeypatch.setattr(module.httpx, 'AsyncClient', forbidden)
    monkeypatch.setattr(module.os.environ, 'get', lambda name, *args:
        forbidden() if name == settings.authoring_api_key_env else original_get(name, *args))
    with pytest.raises(ValueError):
        asyncio.run(module.probe_services(settings))
    with pytest.raises(ValueError):
        module.connection_report(settings)


@pytest.mark.parametrize('base', ['http://127.0.0.1:1234/v1', 'http://localhost:11434',
    'http://[::1]:1234/v1', 'https://authoring.invalid/proxy/v1'])
def test_allowed_loopback_and_tls_probes_keep_auth_private(base, monkeypatch):
    real_client = httpx.AsyncClient
    seen = []
    monkeypatch.setenv('REVIEW_FIXTURE_KEY', 'not-an-actual-key')
    def handler(request):
        seen.append(request)
        return httpx.Response(200, json={'data': [{'id': 'chosen'}]})
    monkeypatch.setattr(module.httpx, 'AsyncClient', lambda **kw:
        real_client(transport=httpx.MockTransport(handler), **kw))
    settings = Settings(auto_start_engine=False, comfy_url='http://external-core.invalid:8188',
        authoring_backend='openai_compatible', authoring_base_url=base,
        authoring_model='chosen', authoring_api_key_env='REVIEW_FIXTURE_KEY')
    result = asyncio.run(module.probe_services(settings))
    assert len(seen) == 2 and seen[0].url.scheme == 'http'
    assert seen[1].headers['Authorization'] == 'Bearer not-an-actual-key'
    assert 'not-an-actual-key' not in json.dumps(result)
    assert seen[1].url.path.endswith('/v1/models')
