"""Public builder path/CLI checks; no native DLL, service or model calls."""
import ctypes
import importlib
import json
from pathlib import Path
import re
import subprocess
import sys
from types import ModuleType

import pytest

import app.local_authoring_runtime as runtime
import app.native_grammar_profile as native


ROOT = Path(__file__).resolve().parents[1]
BUILDERS = ('build_creative_review_grammar', 'build_region_planner_grammar')


@pytest.mark.parametrize('name', BUILDERS)
def test_public_builder_source_has_no_personal_user_path(name):
    source = (ROOT / 'scripts' / (name + '.py')).read_text(encoding='utf-8')
    assert not re.search(r'(?i)[a-z]:[\\/]+Users[\\/]', source)
    assert 'ggml.llamacpp_Microsoft.Winget.Source_' not in source


@pytest.mark.parametrize('name', BUILDERS)
def test_import_does_not_discover_read_assets_start_process_or_load_native(name, monkeypatch):
    # Preload normal Python dependencies, then execute the actual script module
    # body under guards. Source reading/compilation happens before those guards.
    importlib.import_module('scripts.' + name)
    path = ROOT / 'scripts' / (name + '.py')
    code = compile(path.read_text(encoding='utf-8'), str(path), 'exec')
    module = ModuleType('_public_import_' + name)
    module.__file__ = str(path)
    monkeypatch.setitem(sys.modules, module.__name__, module)
    calls = []
    def forbid(*args, **kwargs):
        calls.append((args, kwargs))
        raise AssertionError('Import must not inspect assets, discover a runtime or run native code')
    monkeypatch.setattr(runtime, 'discover_llama_server', forbid)
    monkeypatch.setattr(native, '_read_profile', forbid)
    monkeypatch.setattr(native, '_file_sha256', forbid)
    monkeypatch.setattr(native, '_native_parse', forbid)
    monkeypatch.setattr(subprocess, 'Popen', forbid)
    monkeypatch.setattr(ctypes, 'CDLL', forbid)
    monkeypatch.setattr(Path, 'open', forbid)
    monkeypatch.setattr(Path, 'read_bytes', forbid)
    exec(code, module.__dict__)
    assert calls == [] and callable(module.main)


@pytest.mark.parametrize('name', BUILDERS)
def test_default_discovers_installed_runner_only_when_requested(name, tmp_path, monkeypatch):
    module = importlib.import_module('scripts.' + name)
    discovered = tmp_path / 'discovered' / 'llama-server.exe'
    calls = []
    monkeypatch.setattr(runtime, 'discover_llama_server', lambda: calls.append(True) or discovered)
    assert module.runtime_directory() == discovered.parent
    assert calls == [True]
    # Existing readers of the old constant remain lazy and path-compatible.
    assert module.RUNTIME_DIRECTORY == discovered.parent
    assert calls == [True, True]


@pytest.mark.parametrize('name', BUILDERS)
def test_explicit_directory_beats_discovery(name, tmp_path, monkeypatch):
    module = importlib.import_module('scripts.' + name)
    def forbid():
        raise AssertionError('An explicit runtime directory must not be replaced by discovery')
    monkeypatch.setattr(runtime, 'discover_llama_server', forbid)
    chosen = tmp_path / 'chosen-runtime'
    assert module.runtime_directory(chosen) == chosen


@pytest.mark.parametrize('name', BUILDERS)
def test_cli_check_uses_explicit_directory_without_native_execution(name, tmp_path, monkeypatch, capsys):
    module = importlib.import_module('scripts.' + name)
    chosen = tmp_path / 'not-created-or-loaded'
    def forbid():
        raise AssertionError('Explicit CLI must not use discovery')
    monkeypatch.setattr(runtime, 'discover_llama_server', forbid)
    hashes = getattr(module, 'RUNTIME_SHA256', {
        'server_impl_sha256': 'a' * 64, 'llama_dll_sha256': 'b' * 64})
    checked, parsed = [], []
    def candidate(profile):
        return {'profile': profile, 'schema_sha256': 'c' * 64,
            'grammar_sha256': 'd' * 64, 'grammar': 'fixture grammar never parsed natively', **hashes}
    if name == 'build_creative_review_grammar':
        monkeypatch.setattr(module, 'build_asset', lambda profile, schema: candidate(profile))
        monkeypatch.setattr(module, '_read_profile', lambda path: (candidate(path.stem), 'e' * 64))
    else:
        monkeypatch.setattr(module, 'build_asset', lambda schema, previous: candidate(module.PROFILE))
        monkeypatch.setattr(module, '_read_profile', lambda path:
            ({}, module.LEGACY_ASSET_SHA256) if path.stem == module.LEGACY_PROFILE
            else (candidate(module.PROFILE), 'e' * 64))
    def hash_only(path):
        checked.append(path)
        return hashes['llama_dll_sha256' if path.name == 'llama.dll' else 'server_impl_sha256']
    monkeypatch.setattr(module, '_file_sha256', hash_only)
    monkeypatch.setattr(module, '_native_parse', lambda grammar, path: parsed.append(path))
    module.main(['--check', '--runtime-directory', str(chosen)])
    result = json.loads(capsys.readouterr().out)
    assert result and checked and parsed
    assert all(path.parent == chosen for path in checked + parsed)
    assert not chosen.exists()  # No model directory, process or DLL was created.


def test_review_default_print_does_not_even_discover_runtime(monkeypatch, capsys):
    module = importlib.import_module('scripts.build_creative_review_grammar')
    def forbid(*args, **kwargs):
        raise AssertionError('Print-only derivation must not inspect a local runtime')
    monkeypatch.setattr(module, 'runtime_directory', forbid)
    monkeypatch.setattr(module, '_native_parse', forbid)
    monkeypatch.setattr(module, '_file_sha256', forbid)
    monkeypatch.setattr(module, 'build_asset', lambda profile, schema: {'profile': profile})
    module.main([])
    assert json.loads(capsys.readouterr().out)


def test_review_frozen_schema_and_grammar_are_unchanged():
    module = importlib.import_module('scripts.build_creative_review_grammar')
    candidate = module.build_asset(module.NEXT_CHOICE_PROFILE, module.next_choice_schema())
    existing = json.loads((module.ASSET_DIRECTORY / (module.NEXT_CHOICE_PROFILE + '.json')).read_text(encoding='utf-8'))
    assert candidate == existing


def test_region_frozen_schema_and_grammar_are_unchanged():
    module = importlib.import_module('scripts.build_region_planner_grammar')
    from app.local_iteration_planner import proposal_schema
    previous = json.loads((module.ASSET_DIRECTORY / (module.LEGACY_PROFILE + '.json')).read_text(encoding='utf-8'))
    existing = json.loads((module.ASSET_DIRECTORY / (module.PROFILE + '.json')).read_text(encoding='utf-8'))
    assert module.build_asset(proposal_schema(), previous) == existing
