"""Temporary bytes and hashes only: never load DLLs, models or consumers."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import socket
import subprocess
import sys

import pytest

import app.native_grammar_profile as module


GRAMMAR = 'root ::= response-format-schema\nresponse-format-schema ::= "{" "}"\n'


def digest(value):
    return hashlib.sha256(value).hexdigest()


def save(path, asset):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asset), encoding='utf-8')


@pytest.fixture
def runtime(tmp_path, monkeypatch):
    executable = tmp_path / 'runner/llama-server.exe'
    executable.parent.mkdir()
    executable.write_bytes(b'never executed')
    dlls = {'server_impl_sha256': executable.with_name('llama-server-impl.dll'),
            'llama_dll_sha256': executable.with_name('llama.dll')}
    for key, path in dlls.items():
        path.write_bytes(('fixture ' + key).encode())
    builtins = tmp_path / 'builtin'
    selected = tmp_path / 'registered'
    assets, registry = {}, {}
    for count in range(3):
        name = f'fixture_profile_{count}'
        assets[name] = {'profile': name, 'schema_sha256': digest(('complete-schema-' + str(count)).encode()),
            'grammar': GRAMMAR, 'grammar_sha256': digest(GRAMMAR.encode()),
            **{key: digest(path.read_bytes()) for key, path in dlls.items()},
            'source_runtime_directory': 'metadata only; never followed', 'transformation': {}}
        registry[name] = builtins / (name + '.json')
        save(registry[name], assets[name])
        save(selected / (name + '.json'), assets[name])
    monkeypatch.setattr(module, 'PROFILE_REGISTRY', registry)
    def forbidden(*args, **kwargs):
        raise AssertionError('readiness must not parse/load native code, start a process, or use network')
    monkeypatch.setattr(module, '_native_parse', forbidden)
    monkeypatch.setattr(module.ctypes, 'CDLL', forbidden)
    monkeypatch.setattr(subprocess, 'Popen', forbidden)
    monkeypatch.setattr(socket, 'create_connection', forbidden)
    original_open = Path.open
    def no_weight_open(path, *args, **kwargs):
        if path.suffix.lower() in {'.gguf', '.safetensors', '.ckpt'}:
            raise AssertionError('readiness must never read model weights')
        return original_open(path, *args, **kwargs)
    monkeypatch.setattr(Path, 'open', no_weight_open)
    return executable, dlls, assets, registry, selected


def test_builtin_matching_build_needs_no_operator_binding_or_native_parse(runtime, monkeypatch):
    executable, dlls, assets, registry, selected = runtime
    hash_calls = []
    original = module._file_sha256
    def record(path):
        hash_calls.append(path)
        return original(path)
    monkeypatch.setattr(module, '_file_sha256', record)
    before = set(sys.modules)
    result = module.inspect_local_grammar_runtime(executable)
    assert result['configuration_files_compatible'] is True
    assert result['profile_source'] == 'builtin' and result['profiles_compatible'] == 3
    assert not result['issues']
    assert not result['native_parse_verified'] and not result['native_dll_loaded']
    assert not result['model_inference_verified']
    assert result['scope'] == 'configuration_files_compatible_native_parse_unverified'
    assert hash_calls == list(dlls.values())  # One hash per DLL, not per profile.
    assert set(sys.modules) == before  # No real consumer imports to obtain schemas.


def test_operator_registered_build_can_match_different_dll_hashes(runtime):
    executable, dlls, assets, registry, selected = runtime
    for path in dlls.values():
        path.write_bytes(b'another supported build')
    for name, asset in assets.items():
        new_asset = deepcopy(asset)
        new_asset.update({key: digest(path.read_bytes()) for key, path in dlls.items()})
        save(selected / (name + '.json'), new_asset)
    result = module.inspect_local_grammar_runtime(executable, selected)
    assert result['configuration_files_compatible'] is True and result['profile_source'] == 'registered'
    builtin_result = module.inspect_local_grammar_runtime(executable)
    assert not builtin_result['configuration_files_compatible']
    assert all(item['code'] == 'runtime_dll_profile_mismatch' for item in builtin_result['issues'])


@pytest.mark.parametrize('missing', ['llama-server-impl.dll', 'llama.dll'])
def test_missing_dll_cannot_be_green(runtime, missing):
    executable, dlls, assets, registry, selected = runtime
    executable.with_name(missing).unlink()
    result = module.inspect_local_grammar_runtime(executable)
    assert not result['configuration_files_compatible']
    assert any(item['name'] == missing and item['code'] == 'runtime_dll_unavailable' for item in result['issues'])


def test_explicit_missing_directory_does_not_fall_back_to_valid_builtin(runtime):
    executable, dlls, assets, registry, selected = runtime
    result = module.inspect_local_grammar_runtime(executable, selected / 'does-not-exist')
    assert result['profile_source'] == 'registered' and not result['configuration_files_compatible']
    assert all(item['code'] == 'selected_profile_unavailable' for item in result['issues'])


def test_explicit_partial_directory_does_not_fill_missing_profile_from_builtin(runtime):
    executable, dlls, assets, registry, selected = runtime
    (selected / 'fixture_profile_1.json').unlink()
    result = module.inspect_local_grammar_runtime(executable, selected)
    assert result['profiles_compatible'] == 2 and not result['configuration_files_compatible']
    assert result['issues'][0]['name'] == 'fixture_profile_1'


@pytest.mark.parametrize('mutation,code', [
    ('identity', 'profile_identity_mismatch'), ('grammar_hash', 'grammar_payload_mismatch'),
    ('schema', 'builtin_contract_mismatch'), ('changed_grammar', 'builtin_contract_mismatch'),
    ('digest_format', 'invalid_digest_binding'), ('root', 'invalid_grammar_root'),
    ('source_metadata', 'invalid_profile_metadata'), ('empty_grammar', 'invalid_grammar_payload'),
    ('invalid_unicode', 'invalid_grammar_payload'),
])
def test_invalid_registered_contract_is_diagnosed_before_native_code(runtime, mutation, code):
    executable, dlls, assets, registry, selected = runtime
    name = 'fixture_profile_0'
    value = deepcopy(assets[name])
    if mutation == 'identity': value['profile'] = 'another_profile'
    elif mutation == 'grammar_hash': value['grammar'] += '\n'
    elif mutation == 'schema': value['schema_sha256'] = '0' * 64
    elif mutation == 'digest_format': value['llama_dll_sha256'] = 'not-a-sha'
    elif mutation == 'source_metadata': value['transformation'] = []
    elif mutation == 'empty_grammar': value['grammar'] = ''
    elif mutation == 'invalid_unicode': value['grammar'] = '\ud800'
    elif mutation in {'root', 'changed_grammar'}:
        value['grammar'] = ('root ::= other\nother ::= "x"\n' if mutation == 'root'
                            else GRAMMAR + 'different ::= "x"\n')
        value['grammar_sha256'] = digest(value['grammar'].encode())
    save(selected / (name + '.json'), value)
    result = module.inspect_local_grammar_runtime(executable, selected)
    assert not result['configuration_files_compatible']
    assert any(item['code'] == code for item in result['issues'])


@pytest.mark.parametrize('content', ['{}', '{"profile":"duplicate","profile":"duplicate"}',
                                     'not json', '{"profile":NaN}', ' ' * (module.MAX_PROFILE_BYTES + 1)],
                         ids=['missing-fields', 'duplicate-key', 'invalid-json', 'nonfinite', 'oversized'])
def test_malformed_or_oversized_registered_json_is_diagnosed(runtime, content):
    executable, dlls, assets, registry, selected = runtime
    (selected / 'fixture_profile_0.json').write_text(content, encoding='utf-8')
    result = module.inspect_local_grammar_runtime(executable, selected)
    assert not result['configuration_files_compatible']
    assert result['issues'][0]['code'] == 'selected_profile_unavailable'


def test_corrupt_builtin_binding_is_not_certified_by_matching_registration(runtime):
    executable, dlls, assets, registry, selected = runtime
    registry['fixture_profile_0'].write_text('{}', encoding='utf-8')
    result = module.inspect_local_grammar_runtime(executable, selected)
    assert not result['configuration_files_compatible']
    assert result['issues'][0]['code'] == 'builtin_profile_unavailable'


@pytest.mark.parametrize('target', [None, 'model.gguf', 'another.exe', 'absent/llama-server.exe'])
def test_invalid_or_missing_executable_returns_diagnostic_without_reading_weights(runtime, target):
    result = module.inspect_local_grammar_runtime(target)
    assert not result['configuration_files_compatible'] and result['issues']
    assert not result['dlls'] and not result['profiles']


def test_symlink_profile_is_not_treated_as_regular_registration(runtime, monkeypatch):
    executable, dlls, assets, registry, selected = runtime
    target = selected / 'fixture_profile_0.json'
    original = Path.is_symlink
    monkeypatch.setattr(Path, 'is_symlink', lambda path: path == target or original(path))
    result = module.inspect_local_grammar_runtime(executable, selected)
    assert not result['configuration_files_compatible']
    assert result['issues'][0]['code'] == 'selected_profile_unavailable'


@pytest.mark.parametrize('target', ['executable', 'dll'])
def test_symlink_runtime_files_are_not_certified(runtime, monkeypatch, target):
    executable, dlls, assets, registry, selected = runtime
    link = executable if target == 'executable' else dlls['llama_dll_sha256']
    original = Path.is_symlink
    monkeypatch.setattr(Path, 'is_symlink', lambda path: path == link or original(path))
    result = module.inspect_local_grammar_runtime(executable)
    assert not result['configuration_files_compatible']
    assert result['issues'][0]['code'] == ('runtime_not_regular_file' if target == 'executable' else 'runtime_dll_unavailable')


def test_registry_cannot_redirect_static_profile_reader_to_model_weights(runtime, monkeypatch):
    executable, dlls, assets, registry, selected = runtime
    changed = dict(registry)
    changed['fixture_profile_0'] = executable.parent / 'never-read.gguf'
    monkeypatch.setattr(module, 'PROFILE_REGISTRY', changed)
    result = module.inspect_local_grammar_runtime(executable)
    assert not result['configuration_files_compatible']
    assert result['issues'][0]['code'] == 'builtin_profile_filename_mismatch'


def test_dll_change_during_single_hash_is_not_certified(runtime, monkeypatch):
    executable, dlls, assets, registry, selected = runtime
    original = module._file_sha256
    def mutate(path):
        result = original(path)
        if path.name == 'llama.dll':
            path.write_bytes(b'changed during one static hash')
        return result
    monkeypatch.setattr(module, '_file_sha256', mutate)
    result = module.inspect_local_grammar_runtime(executable)
    assert not result['configuration_files_compatible']
    assert any(issue['code'] == 'runtime_dll_changed' for issue in result['issues'])
