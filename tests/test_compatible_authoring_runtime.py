"""Offline protocol checks only: MockTransport, no server or model calls."""
import asyncio
import base64
from dataclasses import replace
import hashlib
from io import BytesIO
import json
import threading

import httpx
from PIL import Image
import pytest

import app.compatible_authoring_runtime as compatible
from app.local_authoring_runtime import RuntimeConfig, VisualImageInput, LocalAuthoringError, InferenceLease


SCHEMA = {'type': 'object', 'properties': {'brief': {'type': 'string'}}, 'required': ['brief']}
ORIGINAL = 'A character quietly looks at an engraved stone; the reason is context, not another action.'


@pytest.fixture(autouse=True)
def lease(monkeypatch):
    isolated = InferenceLease()
    monkeypatch.setattr(compatible, 'INFERENCE_LEASE', isolated)
    yield isolated
    assert not isolated.is_busy


def config(tmp_path, **kwargs):
    return replace(RuntimeConfig(), backend='openai_compatible',
        provider_base_url='http://127.0.0.1:1234/v1', provider_model='operator-selected',
        evidence_root=tmp_path / 'evidence', constrain_json_output=True, **kwargs)


def payload(content='{"brief":"unverified proposal"}', finish='stop'):
    return {'choices': [{'finish_reason': finish, 'message': {'content': content}}],
        'usage': {'prompt_tokens': 40, 'completion_tokens': 10}}


def invoke(runtime, **kwargs):
    return asyncio.run(runtime.generate_json(system_prompt='Return the requested JSON only.',
        data={'original_text': ORIGINAL}, response_schema=SCHEMA, **kwargs))


def png(label='fixture'):
    stream = BytesIO()
    Image.new('RGB', (8, 8), '#fedcba').save(stream, format='PNG')
    raw = stream.getvalue()
    return VisualImageInput(raw, hashlib.sha256(raw).hexdigest(), source_ref=label)


def test_full_text_schema_evidence_and_secret_not_logged(tmp_path, monkeypatch):
    sent = []
    monkeypatch.setenv('AI_COMIC_AUTHORING_API_KEY', 'fixture-secret-never-in-evidence')
    def handler(request):
        sent.append(request)
        return httpx.Response(200, json=payload())
    runtime = compatible.CompatibleAuthoringRuntime(config(tmp_path), transport=httpx.MockTransport(handler))
    result = invoke(runtime)
    wire = json.loads(sent[0].content)
    assert json.loads(wire['messages'][1]['content'])['original_text'] == ORIGINAL
    assert wire['response_format']['json_schema']['schema'] == SCHEMA
    assert sent[0].headers['authorization'] == 'Bearer fixture-secret-never-in-evidence'
    assert result.data == {'brief': 'unverified proposal'}
    assert result.usage['prompt_tokens'] == 40
    files = list(result.evidence_dir.glob('*.json'))
    assert {'input.json', 'wire_request.json', 'runtime.json', 'response.json', 'parsed_response.json', 'outcome.json'} <= {p.name for p in files}
    assert all('fixture-secret-never-in-evidence' not in p.read_text() for p in files)
    outcome = json.loads((result.evidence_dir / 'outcome.json').read_text())
    assert outcome['chat_posts'] == 1 and not outcome['external_service_owned']
    assert outcome['context_count_method'] == 'provider_managed_no_local_token_probe'


def test_multiple_original_images_preserve_order_and_bytes(tmp_path):
    sent = []
    def handler(request):
        sent.append(json.loads(request.content))
        return httpx.Response(200, json=payload())
    runtime = compatible.CompatibleAuthoringRuntime(config(tmp_path, provider_visual=True), transport=httpx.MockTransport(handler))
    originals = [png('source'), png('successor')]
    result = invoke(runtime, images=originals)
    images = [entry for entry in sent[0]['messages'][1]['content'] if entry['type'] == 'image_url']
    assert len(images) == 2
    for index, (part, original) in enumerate(zip(images, originals), 1):
        assert base64.b64decode(part['image_url']['url'].split(',', 1)[1]) == original.image_bytes
        assert (result.evidence_dir / f'input_image_{index:02d}.png').read_bytes() == original.image_bytes


@pytest.mark.parametrize('count', [1, 3])
def test_registered_pair_contract_never_drops_or_invents_images(tmp_path, count):
    runtime = compatible.CompatibleAuthoringRuntime(config(tmp_path, provider_visual=True,
        native_grammar_profile='local_creative_result_comparison_6703d7894'))
    with pytest.raises(LocalAuthoringError, match='exact number'):
        invoke(runtime, images=[png()] * count)


def test_bad_image_hash_never_posts(tmp_path):
    image = png()
    runtime = compatible.CompatibleAuthoringRuntime(config(tmp_path, provider_visual=True))
    with pytest.raises(LocalAuthoringError) as caught:
        invoke(runtime, images=[replace(image, expected_sha256='0' * 64)])
    assert caught.value.code == 'image_sha_mismatch'


def test_neutral_pixel_contract_keeps_host_original_out_of_wire(tmp_path):
    from app.creative_pixel_observation import (PIXEL_OBSERVATION_SYSTEM,
        _pixel_observation_schema, pixel_observation_input)
    picture = png('fixture_artwork')
    data = pixel_observation_input('fixture_artwork', picture.expected_sha256, 8, 8, len(picture.image_bytes))
    sent = []
    def handler(request):
        sent.append(json.loads(request.content))
        return httpx.Response(200, json=payload())
    runtime = compatible.CompatibleAuthoringRuntime(config(tmp_path, provider_visual=True,
        native_grammar_profile='local_creative_observation_6703d7894'), transport=httpx.MockTransport(handler))
    result = asyncio.run(runtime.generate_json(system_prompt=PIXEL_OBSERVATION_SYSTEM,
        data=data, response_schema=_pixel_observation_schema(), images=[picture], host_original_text=ORIGINAL))
    assert ORIGINAL not in json.dumps(sent[0])
    recorded = json.loads((result.evidence_dir / 'input.json').read_text())
    assert recorded['host_original_scope']['original_text'] == ORIGINAL
    assert sent[0]['response_format']['json_schema']['schema'] == _pixel_observation_schema()


def test_changed_registered_schema_never_posts(tmp_path):
    runtime = compatible.CompatibleAuthoringRuntime(config(tmp_path,
        native_grammar_profile='local_creative_request_reading_6703d7894'))
    with pytest.raises(LocalAuthoringError) as caught:
        invoke(runtime)
    assert caught.value.code == 'native_grammar_profile_invalid'


@pytest.mark.parametrize('body,error', [
    (payload('{"brief":"a","brief":"b"}'), 'provider_failed'),
    (payload('```json\n{}\n```'), 'provider_failed'),
    (payload('{"brief":"partial"}', 'length'), 'incomplete_output'),
    ({'choices': [{'finish_reason': 'stop', 'message': {'tool_calls': [{}], 'content': '{}'}}]}, 'invalid_output'),
    ({'choices': [{'finish_reason': 'stop', 'message': {'refusal': 'no', 'content': '{}'}}]}, 'invalid_output'),
])
def test_output_remains_strict_no_retry_or_fallback(tmp_path, body, error):
    calls = []
    def handler(request):
        calls.append(request)
        return httpx.Response(200, json=body)
    runtime = compatible.CompatibleAuthoringRuntime(config(tmp_path), transport=httpx.MockTransport(handler))
    with pytest.raises(LocalAuthoringError) as caught:
        invoke(runtime)
    assert caught.value.code == error and len(calls) == 1
    assert (caught.value.evidence_dir / 'response.json').is_file()
    assert not (caught.value.evidence_dir / 'parsed_response.json').exists()


def test_provider_rejecting_schema_is_not_retried_as_json_object(tmp_path):
    calls = []
    def handler(request):
        calls.append(request)
        return httpx.Response(400, json={'error': 'schema unsupported'})
    runtime = compatible.CompatibleAuthoringRuntime(config(tmp_path), transport=httpx.MockTransport(handler))
    with pytest.raises(LocalAuthoringError) as caught:
        invoke(runtime)
    assert caught.value.code == 'provider_http_error' and len(calls) == 1


def test_async_cancellation_aborts_only_own_http_and_releases_lease(tmp_path):
    entered, cancelled = asyncio.Event(), []
    async def handler(request):
        entered.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            cancelled.append(True)
            raise
    runtime = compatible.CompatibleAuthoringRuntime(config(tmp_path), transport=httpx.MockTransport(handler))
    async def scenario():
        task = asyncio.create_task(runtime.generate_json(system_prompt='JSON', data={'original_text': ORIGINAL}, response_schema=SCHEMA))
        await asyncio.wait_for(entered.wait(), 1)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    asyncio.run(scenario())
    assert cancelled
    outcome = json.loads(next((tmp_path / 'evidence').glob('*/outcome.json')).read_text())
    assert outcome['status'] == 'cancelled' and not outcome['external_service_stopped']


def test_external_cancel_keeps_raw_evidence_without_adopting_late_output(tmp_path):
    stop = threading.Event()
    async def handler(request):
        stop.set()
        await asyncio.sleep(.1)
        return httpx.Response(200, json=payload())
    runtime = compatible.CompatibleAuthoringRuntime(config(tmp_path), transport=httpx.MockTransport(handler))
    with pytest.raises(LocalAuthoringError) as caught:
        invoke(runtime, cancel_event=stop)
    assert caught.value.code == 'cancelled'


@pytest.mark.parametrize('url', ['https://user:password@example.com/v1', 'http://remote.example/v1',
    'https://example.com/v1?api_key=secret', 'file:///local/model', '',
    ' https://example.com/v1', 'https://example.com/a b',
    'https://example.com/v1/chat/completions', 'https://example.com/v1/models',
    'https://example.com/v1/../admin', 'https://example.com/v1?', 'https://example.com/v1#'])
def test_invalid_or_insecure_endpoint_fails_without_network(url):
    with pytest.raises(LocalAuthoringError):
        compatible.compatible_chat_url(url)


def test_explicit_remote_https_and_local_service_paths():
    assert compatible.compatible_chat_url('https://operator.example/api/v1') == 'https://operator.example/api/v1/chat/completions'
    assert compatible.compatible_chat_url('http://localhost:11434') == 'http://localhost:11434/v1/chat/completions'


def test_unsafe_provider_never_reads_environment_or_opens_http(tmp_path, monkeypatch):
    runtime = compatible.CompatibleAuthoringRuntime(replace(config(tmp_path),
        provider_base_url='http://authoring.invalid/v1'))
    calls = []
    def forbidden(*args, **kwargs):
        calls.append(True)
        raise AssertionError('Unsafe endpoint must not read a key or create an HTTP client')
    original_get = compatible.os.environ.get
    monkeypatch.setattr(compatible.os.environ, 'get', lambda name, *args:
        forbidden() if name == runtime.config.provider_api_key_env else original_get(name, *args))
    monkeypatch.setattr(compatible.httpx, 'AsyncClient', forbidden)
    with pytest.raises(LocalAuthoringError) as caught:
        invoke(runtime)
    assert caught.value.code == 'invalid_provider_config'
    assert calls == [] and not (tmp_path / 'evidence').exists()
