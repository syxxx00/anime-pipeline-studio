"""Paths and registrations use fake local assets only; never model startup."""
from dataclasses import replace
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from app.config import Settings
from app.creative_support import support_runtime
from app.local_iteration_planner import local_visual_runtime
from app.compatible_authoring_runtime import CompatibleAuthoringRuntime
from app.creation_concept_context import CreationConceptContext
from app.portable_runtime import local_runtime_config
import app.native_grammar_profile as grammar


def test_configured_unknown_gguf_uses_standard_not_historical_4b(tmp_path):
    settings = Settings(local_llm_server=str(tmp_path / 'llama-server.exe'),
        local_llm_model=str(tmp_path / 'custom-model.gguf'), local_llm_runtime_dir=str(tmp_path / 'runtime'))
    config = support_runtime(settings).config
    assert config.model_path == tmp_path / 'custom-model.gguf'
    assert config.resource_profile == 'standard' and config.portable_local_model
    assert config.minimum_start_ram_mib == 4000


def test_compatible_factory_does_not_require_local_weights_or_projector(tmp_path):
    settings = Settings(authoring_backend='openai_compatible', authoring_base_url='http://localhost:1234/v1',
        authoring_model='chosen-text', authoring_vision_model='chosen-vision')
    text = support_runtime(settings)
    vision = local_visual_runtime(settings, profile='bounded_4b_visual_none')
    assert isinstance(text, CompatibleAuthoringRuntime) and isinstance(vision, CompatibleAuthoringRuntime)
    assert text.config.provider_model == 'chosen-text'
    assert vision.config.provider_model == 'chosen-vision' and vision.config.mmproj_path is None
    assert vision.config.resource_profile == 'standard'


def test_blank_vision_override_uses_same_selected_service_model_not_text_only():
    settings = Settings(authoring_backend='openai_compatible', authoring_base_url='http://localhost:1234/v1',
        authoring_model='operator-selected-multimodal', authoring_vision_model='')
    vision = local_visual_runtime(settings)
    assert isinstance(vision, CompatibleAuthoringRuntime)
    assert vision.config.provider_model == 'operator-selected-multimodal'
    assert vision.config.provider_visual is True
    assert vision.config.mmproj_path is None  # The external service owns its encoder.


def test_runtime_discovery_does_not_read_missing_model_or_projector(tmp_path):
    settings = Settings(local_llm_model=str(tmp_path / 'not-present.gguf'),
        local_llm_projector=str(tmp_path / 'not-present-projector.gguf'))
    config = local_runtime_config(settings, purpose='vision')
    assert config.model_path.name == 'not-present.gguf'
    assert config.mmproj_path.name == 'not-present-projector.gguf'
    assert config.visual_model_stat is None and config.mmproj_stat is None


def test_legacy_preparation_entry_uses_operator_selected_compatible_service(tmp_path):
    from app.creation_preparation import CreationPreparations
    settings = Settings(authoring_backend='openai_compatible', authoring_base_url='http://localhost:1234/v1',
        authoring_model='explicit-text', authoring_vision_model='explicit-vision')
    service = CreationPreparations(settings, root=tmp_path / 'records')
    assert isinstance(service.runtime, CompatibleAuthoringRuntime)
    assert service.runtime.config.provider_model == 'explicit-text'
    assert service.runtime.config.constrain_json_output


def test_explicit_concept_root_is_used_without_external_f_library(tmp_path):
    root = tmp_path / 'concepts'
    context = CreationConceptContext.for_settings(SimpleNamespace(concept_root=str(root)))
    assert context.documents['project'] == root / 'ai-comic/CONCEPT.md'
    assert context.documents['method'] == root / 'concept-workflow/SKILL.md'
    assert not any('F:/Models2' in str(p) for p in context.documents.values())


def test_native_registration_keeps_full_grammar_and_rebinds_only_build(tmp_path, monkeypatch):
    source_dir = tmp_path / 'source'
    source_dir.mkdir()
    runner = tmp_path / 'runner' / 'llama-server.exe'
    runner.parent.mkdir()
    runner.write_bytes(b'fake executable never executed')
    runner.with_name('llama-server-impl.dll').write_bytes(b'fake implementation')
    runner.with_name('llama.dll').write_bytes(b'fake parser')
    name, schema = 'local_iteration_6703d7894', {'type': 'object'}
    text = 'root ::= response-format-schema\nresponse-format-schema ::= "{" "}"\n'
    digest = lambda raw: hashlib.sha256(raw).hexdigest()
    asset = {'profile': name, 'schema_sha256': digest(grammar._canonical_json(schema)),
        'grammar_sha256': digest(text.encode()), 'grammar': text,
        'server_impl_sha256': '0' * 64, 'llama_dll_sha256': '0' * 64,
        'source_runtime_directory': 'not-followed', 'transformation': {'length_scope': 'test'}}
    source = source_dir / (name + '.json')
    source.write_text(json.dumps(asset), encoding='utf-8')
    monkeypatch.setattr(grammar, 'PROFILE_REGISTRY', {name: source})
    calls = []
    monkeypatch.setattr(grammar, '_native_parse', lambda *args: calls.append(args))
    output = tmp_path / 'operator-bindings'
    receipt = grammar.bind_local_grammar_runtime(runner, output)
    assert receipt['registered_profiles'] == 1 and not receipt['model_weights_loaded']
    assert not receipt['inference_compatibility_verified']
    bound = json.loads((output / (name + '.json')).read_text())
    assert bound['grammar'] == text and bound['schema_sha256'] == asset['schema_sha256']
    actual, loaded = grammar.load_native_grammar_profile(name, schema, runner, profile_root=output)
    assert actual == text and loaded['native_parse_verified']
    assert json.loads(source.read_text()) == asset
    runner.with_name('llama.dll').write_bytes(b'changed fake parser')
    with pytest.raises(grammar.NativeGrammarProfileError, match='identity'):
        grammar.load_native_grammar_profile(name, schema, runner, profile_root=output)
