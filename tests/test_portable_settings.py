"""No network, engines or model loads: deployment configuration and saved paths only."""
from dataclasses import fields
import json
from pathlib import Path

import pytest

import app.config as config
import app.comfy_client as comfy
from app.model_catalog import model_catalog


def test_new_install_paths_follow_install_not_cwd(tmp_path, monkeypatch):
    install = tmp_path / 'different install 中文'
    elsewhere = tmp_path / 'elsewhere'
    elsewhere.mkdir()
    monkeypatch.setattr(config, 'PROJECT_ROOT', install)
    monkeypatch.chdir(elsewhere)
    settings = config.Settings()
    assert Path(settings.comfy_root) == install / 'runtime/ComfyUI'
    assert Path(settings.comfy_python).is_relative_to(install)
    assert Path(settings.model_root) == install / 'models'
    assert Path(settings.style_catalog_path) == install / 'config/style_presets.json'
    assert settings.local_llm_server == settings.concept_root == ''
    assert settings.web_port == 8090 and settings.local_llm_port == 8191


def test_json_relative_paths_and_explicit_external_paths(tmp_path, monkeypatch):
    monkeypatch.setattr(config, 'PROJECT_ROOT', tmp_path / 'installation')
    path = tmp_path / 'arbitrary-settings.json'
    external = tmp_path / 'existing models'
    original = {'model_root': str(external), 'engine_base_dir': 'runtime/engine',
                'local_llm_model': 'weights/local.gguf', 'local_llm_grammar_dir': 'config/grammars',
                'diffusion_model': 'another/model.safetensors', 'unknown_legacy_field': 'retained on disk'}
    path.write_text(json.dumps(original), encoding='utf-8')
    settings = config.load_settings(path)
    assert Path(settings.model_root) == external
    assert Path(settings.engine_base_dir) == config.PROJECT_ROOT / 'runtime/engine'
    assert Path(settings.local_llm_model) == config.PROJECT_ROOT / 'weights/local.gguf'
    assert Path(settings.local_llm_grammar_dir) == config.PROJECT_ROOT / 'config/grammars'
    assert settings.diffusion_model == 'another/model.safetensors'
    assert json.loads(path.read_text()) == original


def test_new_settings_fields_are_dataclass_fields():
    names = {field.name for field in fields(config.Settings)}
    assert {'web_port', 'authoring_backend', 'authoring_base_url', 'authoring_model', 'authoring_vision_model',
            'authoring_api_key_env', 'local_llm_server', 'local_llm_model', 'local_llm_projector',
            'local_llm_model_7b', 'local_llm_model_9b', 'local_llm_runtime_dir', 'local_llm_evidence_dir',
            'local_llm_port', 'local_llm_grammar_dir', 'concept_root'} <= names


@pytest.mark.parametrize('update', [{'web_port': 0}, {'engine_port': 65536}, {'local_llm_port': True},
                                   {'diffusion_model': '../outside.safetensors'},
                                   {'z_clip': 'C:/outside.safetensors'}, {'z_vae': 'C:outside.safetensors'},
                                   {'authoring_backend': 'unknown'}])
def test_invalid_port_and_model_name_are_rejected(update):
    with pytest.raises(ValueError):
        config.Settings(**update)


def test_model_catalog_uses_custom_z_filenames_and_legacy_clip_folder(tmp_path):
    settings = config.Settings(model_root=str(tmp_path), z_diffusion_model='custom/z.safetensors',
                               z_clip='custom/q.safetensors', z_vae='custom/v.safetensors')
    for key in ('z_diffusion_model', 'z_clip_alt', 'z_vae'):
        path = settings.path_map[key]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b'fixture')
    entry = next(item for item in model_catalog(settings) if item['id'] == 'z_image_turbo')
    assert entry['ready'] and not entry['missing']
    settings.path_map['z_vae'].unlink()
    assert next(item for item in model_catalog(settings) if item['id'] == 'z_image_turbo')['missing'] == ['custom/v.safetensors']


def test_generated_model_paths_preserve_explicit_model_root_and_current_extensions(tmp_path, monkeypatch):
    monkeypatch.setattr(comfy, 'PROJECT_ROOT', tmp_path / 'copy 中文')
    settings = config.Settings(model_root=str(tmp_path / 'existing weights'), engine_base_dir=str(tmp_path / 'engine data'))
    client = comfy.ComfyClient(settings)
    path = client._write_extra_paths()
    value = json.loads(path.read_text(encoding='utf-8'))
    assert value['newbie_lab_models']['base_path'] == settings.model_root
    assert value['newbie_lab_models']['text_encoders'].splitlines() == ['text_encoders', 'clip']
    assert value['ai_comic_engine_extensions']['base_path'] == str(tmp_path / 'copy 中文')
    assert str(path) in client._command()
    assert not list(path.parent.glob('*.tmp'))


def test_engine_cache_uses_configured_base_and_does_not_mutate_parent(tmp_path, monkeypatch):
    import os
    monkeypatch.setenv('HF_HUB_OFFLINE', '0')
    before = dict(os.environ)
    settings = config.Settings(engine_base_dir=str(tmp_path / 'external runtime/ComfyUI'))
    env = comfy.local_engine_environment(settings)
    root = tmp_path / 'external runtime/cache'
    assert Path(env['HF_HOME']) == root / 'huggingface'
    assert Path(env['TEMP']) == root / 'temp'
    assert env['HF_HUB_OFFLINE'] == env['TRANSFORMERS_OFFLINE'] == env['PIP_NO_INDEX'] == '1'
    assert dict(os.environ) == before


def test_engine_process_receipt_uses_real_creation_time_only(tmp_path, monkeypatch):
    from types import SimpleNamespace
    monkeypatch.setattr(comfy, 'PROJECT_ROOT', tmp_path)
    client = comfy.ComfyClient(config.Settings())
    client._process = SimpleNamespace(pid=123)
    monkeypatch.setattr(comfy, '_process_creation_iso', lambda process: None)
    client._save_process_receipt(['fixture', 'main.py'])
    assert not (tmp_path / 'data/engine-process.json').exists()
    monkeypatch.setattr(comfy, '_process_creation_iso', lambda process: '2026-10-15T12:00:00+00:00')
    client._save_process_receipt(['fixture', 'main.py'])
    receipt = json.loads((tmp_path / 'data/engine-process.json').read_text())
    assert receipt['pid'] == 123 and receipt['started_at'] == '2026-10-15T12:00:00+00:00'
    assert receipt['project_root'] == str(tmp_path) and receipt['role'] == 'engine'
