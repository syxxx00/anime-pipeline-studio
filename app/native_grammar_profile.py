"""Explicit, version-bound GBNF compatibility profiles; never load a model.

Importing this module performs no IO. A profile does not certify understanding,
nor claim every JSON Schema keyword is enforced by the native sampler. The
unchanged schema and downstream Pydantic validation remain authoritative.
"""
from __future__ import annotations

import ctypes
import hashlib
import json
import os
from pathlib import Path
import re


PROFILE_REGISTRY = {
    'local_iteration_6703d7894': Path(__file__).parent / 'grammars' / 'local_iteration_6703d7894.json',
    'local_iteration_region_6703d7894': Path(__file__).parent / 'grammars' / 'local_iteration_region_6703d7894.json',
    'local_creative_observation_6703d7894': Path(__file__).parent / 'grammars' / 'local_creative_observation_6703d7894.json',
    'local_creative_result_comparison_6703d7894': Path(__file__).parent / 'grammars' / 'local_creative_result_comparison_6703d7894.json',
    'local_creative_next_choice_6703d7894': Path(__file__).parent / 'grammars' / 'local_creative_next_choice_6703d7894.json',
    'local_creative_reference_next_choice_6703d7894': Path(__file__).parent / 'grammars' / 'local_creative_reference_next_choice_6703d7894.json',
    'local_creative_starting_choice_6703d7894': Path(__file__).parent / 'grammars' / 'local_creative_starting_choice_6703d7894.json',
    'local_creative_initial_prose_decision_6703d7894': Path(__file__).parent / 'grammars' / 'local_creative_initial_prose_decision_6703d7894.json',
    'local_creative_next_prose_decision_6703d7894': Path(__file__).parent / 'grammars' / 'local_creative_next_prose_decision_6703d7894.json',
    'local_creative_reference_prose_decision_6703d7894': Path(__file__).parent / 'grammars' / 'local_creative_reference_prose_decision_6703d7894.json',
    'local_creative_research_prose_decision_6703d7894': Path(__file__).parent / 'grammars' / 'local_creative_research_prose_decision_6703d7894.json',
    'local_creative_request_reading_6703d7894': Path(__file__).parent / 'grammars' / 'local_creative_request_reading_6703d7894.json',
    'local_creative_region_localization_6703d7894': Path(__file__).parent / 'grammars' / 'local_creative_region_localization_6703d7894.json',
    'local_creative_direction_plan_6703d7894': Path(__file__).parent / 'grammars' / 'local_creative_direction_plan_6703d7894.json',
    'local_creative_fixed_pictures_1_6703d7894': Path(__file__).parent / 'grammars' / 'local_creative_fixed_pictures_1_6703d7894.json',
    'local_creative_fixed_pictures_2_6703d7894': Path(__file__).parent / 'grammars' / 'local_creative_fixed_pictures_2_6703d7894.json',
    'local_creative_fixed_pictures_3_6703d7894': Path(__file__).parent / 'grammars' / 'local_creative_fixed_pictures_3_6703d7894.json',
    'local_creative_fixed_pictures_4_6703d7894': Path(__file__).parent / 'grammars' / 'local_creative_fixed_pictures_4_6703d7894.json',
    'local_creative_fixed_pictures_5_6703d7894': Path(__file__).parent / 'grammars' / 'local_creative_fixed_pictures_5_6703d7894.json',
    'local_creative_fixed_pictures_6_6703d7894': Path(__file__).parent / 'grammars' / 'local_creative_fixed_pictures_6_6703d7894.json',
}
# Syntax permissions only; model/projector and resource admission stay separate.
VISUAL_NATIVE_GRAMMAR_PROFILES = {'local_creative_observation_6703d7894': 1,
                                'local_creative_result_comparison_6703d7894': 2,
                                'local_creative_region_localization_6703d7894': 1}
MAX_PROFILE_BYTES = 64 * 1024
_FIELDS = {'profile', 'schema_sha256', 'grammar_sha256', 'grammar',
    'server_impl_sha256', 'llama_dll_sha256', 'source_runtime_directory', 'transformation'}
_DIGEST = re.compile(r'^[0-9a-f]{64}$')


class NativeGrammarProfileError(ValueError):
    """A requested profile failed closed, without native-schema fallback."""


def _canonical_json(value) -> bytes:
    # Exactly local_authoring_runtime._json, including insertion order.
    return json.dumps(value, ensure_ascii=False, allow_nan=False,
                      separators=(',', ':')).encode('utf-8')


def _unique_pairs(items):
    result = {}
    for key, value in items:
        if key in result:
            raise NativeGrammarProfileError('Duplicate grammar profile JSON key')
        result[key] = value
    return result


def _reject_constant(_):
    raise NativeGrammarProfileError('Non-finite grammar profile JSON value')


def _read_profile(path: Path) -> tuple[dict, str]:
    if path.is_symlink() or not path.is_file():
        raise NativeGrammarProfileError('Registered grammar profile is not a regular local file')
    with path.open('rb') as handle:
        raw = handle.read(MAX_PROFILE_BYTES + 1)
    if not raw or len(raw) > MAX_PROFILE_BYTES:
        raise NativeGrammarProfileError('Grammar profile exceeds its 64 KiB file bound')
    try:
        asset = json.loads(raw.decode('utf-8'), object_pairs_hook=_unique_pairs,
                           parse_constant=_reject_constant)
    except (UnicodeError, ValueError) as exc:
        raise NativeGrammarProfileError('Grammar profile is not strict UTF-8 JSON') from exc
    if not isinstance(asset, dict) or set(asset) != _FIELDS:
        raise NativeGrammarProfileError('Grammar profile has missing or unknown fields')
    return asset, hashlib.sha256(raw).hexdigest()


def _file_sha256(path: Path) -> str:
    if path.is_symlink() or not path.is_file():
        raise NativeGrammarProfileError('Grammar profile requires regular adjacent runtime DLLs')
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _native_parse(grammar: str, dll_path: Path) -> None:
    """Only instantiate/free the parser with NULL vocab, never inference."""
    if os.name != 'nt' or not hasattr(os, 'add_dll_directory'):
        raise NativeGrammarProfileError('This registered native grammar profile requires Windows')
    try:
        with os.add_dll_directory(str(dll_path.parent)):
            library = ctypes.CDLL(str(dll_path), winmode=0x00001100)
            initialize = library.llama_sampler_init_grammar
            initialize.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_char_p]
            initialize.restype = ctypes.c_void_p
            release = library.llama_sampler_free
            release.argtypes = [ctypes.c_void_p]
            release.restype = None
            sampler = initialize(None, grammar.encode('utf-8'), b'root')
            try:
                if not sampler:
                    raise NativeGrammarProfileError('Registered native GBNF did not parse; no fallback')
            finally:
                if sampler:
                    release(sampler)
    except (OSError, AttributeError) as exc:
        raise NativeGrammarProfileError('Native grammar parser is unavailable; no fallback') from exc


def validate_schema_profile(profile, schema) -> dict:
    """Check the complete contract for an external structured-output service.

    This does not load native DLLs, send a request or claim that a service's
    sampler enforces every keyword. Typed caller validation remains required.
    """
    if not isinstance(profile, str) or profile not in PROFILE_REGISTRY:
        raise NativeGrammarProfileError('Unknown registered response schema profile')
    asset, _ = _read_profile(Path(PROFILE_REGISTRY[profile]))
    if (asset['profile'] != profile or not isinstance(schema, dict)
            or hashlib.sha256(_canonical_json(schema)).hexdigest() != asset['schema_sha256']):
        raise NativeGrammarProfileError('Complete response schema does not match its registered profile')
    if hashlib.sha256(asset['grammar'].encode('utf-8')).hexdigest() != asset['grammar_sha256']:
        raise NativeGrammarProfileError('Registered grammar payload was changed')
    return {'profile': profile, 'schema_sha256': asset['schema_sha256'],
        'native_dll_loaded': False, 'provider_schema_enforcement_verified': False}


def _static_profile_problem(asset: dict, profile: str) -> tuple[str, str] | None:
    """File identity checks only; the native parser remains a separate action."""
    if asset['profile'] != profile:
        return 'profile_identity_mismatch', 'Profile identity does not match its registered filename'
    for field in ('schema_sha256', 'grammar_sha256', 'server_impl_sha256', 'llama_dll_sha256'):
        if not isinstance(asset[field], str) or not _DIGEST.fullmatch(asset[field]):
            return 'invalid_digest_binding', 'Profile has an invalid SHA256 binding: ' + field
    grammar = asset['grammar']
    if not isinstance(grammar, str) or not grammar or '\x00' in grammar:
        return 'invalid_grammar_payload', 'Profile grammar must be bounded, nonempty text'
    try:
        grammar_bytes = grammar.encode('utf-8')
    except UnicodeError:
        return 'invalid_grammar_payload', 'Profile grammar is not valid UTF-8 text'
    if len(grammar_bytes) > MAX_PROFILE_BYTES:
        return 'invalid_grammar_payload', 'Profile grammar must be bounded, nonempty text'
    if hashlib.sha256(grammar_bytes).hexdigest() != asset['grammar_sha256']:
        return 'grammar_payload_mismatch', 'Profile grammar payload differs from its declared SHA256'
    roots = [line.strip() for line in grammar.splitlines() if re.match(r'^\s*root\s*::=', line)]
    if roots != ['root ::= response-format-schema']:
        return 'invalid_grammar_root', 'Profile root must be the complete pure-JSON response schema'
    if not isinstance(asset['source_runtime_directory'], str) or not isinstance(asset['transformation'], dict):
        return 'invalid_profile_metadata', 'Profile source/transformation metadata is invalid'
    return None


def inspect_local_grammar_runtime(server_executable, profile_root=None) -> dict:
    """Inspect files and hashes without native parsing or importing consumers.

    A built-in profile is sufficient when its complete bindings match. An
    explicit registration directory is authoritative even when missing or
    invalid: it never silently falls back to the built-in assets. DLL contents
    are hashed once per call, not once for each registered response profile.
    """
    report = {
        'scope': 'configuration_files_compatible_native_parse_unverified',
        'configuration_files_compatible': False, 'native_parse_verified': False,
        'model_inference_verified': False, 'native_dll_loaded': False,
        'profile_source': 'registered' if profile_root is not None else 'builtin',
        'profiles_expected': len(PROFILE_REGISTRY), 'profiles_compatible': 0,
        'dlls': [], 'profiles': [], 'issues': [],
    }

    def issue(component, name, status, code, message):
        value = {'component': component, 'name': name, 'status': status, 'code': code, 'message': message}
        report['issues'].append(value)
        return value

    try:
        executable = Path(server_executable).expanduser()
        report['server_executable'] = str(executable)
        if executable.name.lower() != 'llama-server.exe':
            issue('runtime', 'server_executable', 'invalid', 'invalid_executable_name',
                  'Select the actual llama-server.exe, not a model file or another executable')
            return report
        if executable.is_symlink() or not executable.is_file():
            issue('runtime', 'server_executable', 'missing', 'runtime_not_regular_file',
                  'llama-server.exe must exist as a regular file')
            return report
        executable = executable.resolve(strict=True)
        report['server_executable'] = str(executable)
        selected_root = Path(profile_root).expanduser() if profile_root is not None else None
        report['profile_root'] = str(selected_root) if selected_root is not None else None
    except (OSError, TypeError, ValueError):
        issue('runtime', 'server_executable', 'invalid', 'invalid_runtime_path',
              'Runtime or profile directory path is invalid or unreadable')
        return report

    dll_hashes = {}
    for field, name in (('server_impl_sha256', 'llama-server-impl.dll'), ('llama_dll_sha256', 'llama.dll')):
        path = executable.with_name(name)
        item = {'name': name, 'path': str(path), 'status': 'missing'}
        report['dlls'].append(item)
        try:
            before = path.stat()
            digest = _file_sha256(path)
            after = path.stat()
            if ((before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
                    != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)):
                item['status'] = 'mismatch'
                issue('dll', name, 'mismatch', 'runtime_dll_changed', 'Runtime DLL changed while its hash was inspected')
            else:
                dll_hashes[field] = digest
                item.update(status='readable', sha256=digest)
        except (OSError, ValueError):
            issue('dll', name, 'missing', 'runtime_dll_unavailable',
                  'Required adjacent runtime DLL is missing, unreadable or not a regular file')

    for profile, builtin_path in PROFILE_REGISTRY.items():
        selected_path = selected_root / (profile + '.json') if selected_root is not None else Path(builtin_path)
        item = {'profile': profile, 'path': str(selected_path), 'status': 'invalid'}
        report['profiles'].append(item)
        if Path(builtin_path).name != profile + '.json':
            item.update(code='builtin_profile_filename_mismatch', message='Built-in profile filename does not match its registered identity')
            issue('profile', profile, 'invalid', item['code'], item['message'])
            continue
        try:
            builtin, _ = _read_profile(Path(builtin_path))
            problem = _static_profile_problem(builtin, profile)
            if problem is not None:
                item.update(code='builtin_' + problem[0], message=problem[1])
                issue('profile', profile, 'invalid', item['code'], item['message'])
                continue
        except (OSError, ValueError):
            item.update(code='builtin_profile_unavailable', message='Built-in schema/grammar binding is missing or invalid')
            issue('profile', profile, 'invalid', item['code'], item['message'])
            continue
        try:
            asset, _ = _read_profile(selected_path) if selected_root is not None else (builtin, None)
        except (OSError, ValueError):
            try:
                missing = not selected_path.exists()
            except (OSError, ValueError):
                missing = False
            item.update(status='missing' if missing else 'invalid',
                        code='selected_profile_unavailable', message='Selected profile is missing, unreadable or invalid; no fallback')
            issue('profile', profile, item['status'], item['code'], item['message'])
            continue
        problem = _static_profile_problem(asset, profile)
        if problem is not None:
            item.update(code=problem[0], message=problem[1])
            issue('profile', profile, 'invalid', item['code'], item['message'])
            continue
        if any(asset[field] != builtin[field] for field in ('schema_sha256', 'grammar_sha256')):
            item.update(status='mismatch', code='builtin_contract_mismatch',
                        message='Selected profile does not match this application\'s complete schema and grammar binding')
            issue('profile', profile, item['status'], item['code'], item['message'])
            continue
        if len(dll_hashes) != 2:
            item.update(status='not_checked', code='runtime_dll_unavailable',
                        message='Profile binding cannot be matched until both required DLLs are readable')
            continue
        if any(asset[field] != digest for field, digest in dll_hashes.items()):
            item.update(status='mismatch', code='runtime_dll_profile_mismatch',
                        message='Runtime DLL hashes do not match the selected profile; use matching files or register this build')
            issue('profile', profile, item['status'], item['code'], item['message'])
            continue
        item.update(status='compatible', code='configuration_files_compatible',
                    message='Profile files and DLL hashes match; native parsing has not been checked')
        report['profiles_compatible'] += 1
    report['configuration_files_compatible'] = (not report['issues'] and len(dll_hashes) == 2
        and report['profiles_expected'] > 0 and report['profiles_compatible'] == report['profiles_expected'])
    return report


def load_native_grammar_profile(profile, schema, server_executable, *, profile_root=None) -> tuple[str, dict]:
    """Load only registered assets bound to this exact schema and runtime build."""
    if not isinstance(profile, str) or profile not in PROFILE_REGISTRY:
        raise NativeGrammarProfileError('Unknown native grammar profile; paths are not accepted')
    path = (Path(profile_root) / (profile + '.json') if profile_root is not None
            else Path(PROFILE_REGISTRY[profile]))
    if path.name != profile + '.json':
        raise NativeGrammarProfileError('Registered grammar profile filename does not match its identity')
    try:
        asset, asset_sha256 = _read_profile(path)
        if asset['profile'] != profile:
            raise NativeGrammarProfileError('Grammar asset profile identity mismatch')
        for name in ('schema_sha256', 'grammar_sha256', 'server_impl_sha256', 'llama_dll_sha256'):
            if not isinstance(asset[name], str) or not _DIGEST.fullmatch(asset[name]):
                raise NativeGrammarProfileError('Grammar profile contains an invalid SHA256 binding')
        if not isinstance(schema, dict):
            raise NativeGrammarProfileError('Grammar profile requires the complete schema object')
        schema_sha256 = hashlib.sha256(_canonical_json(schema)).hexdigest()
        if schema_sha256 != asset['schema_sha256']:
            raise NativeGrammarProfileError('Complete schema changed; this grammar profile cannot be reused')
        grammar = asset['grammar']
        if (not isinstance(grammar, str) or not grammar or '\x00' in grammar
                or len(grammar.encode('utf-8')) > MAX_PROFILE_BYTES):
            raise NativeGrammarProfileError('Grammar asset does not contain bounded nonempty GBNF')
        grammar_sha256 = hashlib.sha256(grammar.encode('utf-8')).hexdigest()
        if grammar_sha256 != asset['grammar_sha256']:
            raise NativeGrammarProfileError('Grammar payload SHA256 mismatch')
        roots = [line.strip() for line in grammar.splitlines()
                 if re.match(r'^\s*root\s*::=', line)]
        if roots != ['root ::= response-format-schema']:
            raise NativeGrammarProfileError('Profile root must be the complete pure-JSON response schema')
        if (not isinstance(asset['source_runtime_directory'], str)
                or not isinstance(asset['transformation'], dict)):
            raise NativeGrammarProfileError('Grammar profile lacks its declared source/transformation metadata')
        executable = Path(server_executable)
        if executable.name.lower() != 'llama-server.exe' or executable.is_symlink() or not executable.is_file():
            raise NativeGrammarProfileError('Profile requires an explicit regular local llama-server.exe')
        executable = executable.resolve(strict=True)
        files = {'server_impl_sha256': executable.with_name('llama-server-impl.dll'),
                 'llama_dll_sha256': executable.with_name('llama.dll')}
        for field, dll in files.items():
            if _file_sha256(dll) != asset[field]:
                raise NativeGrammarProfileError('Adjacent native runtime DLL identity does not match profile')
        _native_parse(grammar, files['llama_dll_sha256'])
        # Re-check actual files after loading/parsing instead of assuming that
        # a pre-parse digest still describes a changed installation.
        for field, dll in files.items():
            if _file_sha256(dll) != asset[field]:
                raise NativeGrammarProfileError('Native runtime DLL changed during grammar validation')
    except NativeGrammarProfileError:
        raise
    except (OSError, TypeError, ValueError) as exc:
        raise NativeGrammarProfileError('Requested grammar profile could not be validated; no fallback') from exc
    receipt = {'profile': profile, 'asset_sha256': asset_sha256,
        'schema_sha256': schema_sha256, 'grammar_sha256': grammar_sha256,
        'server_impl_sha256': asset['server_impl_sha256'], 'llama_dll_sha256': asset['llama_dll_sha256'],
        'native_parse_verified': True, 'parser_vocab': None, 'model_weights_loaded': False,
        'root_scope': 'complete_pure_json_response_schema', 'complete_schema_unchanged': True,
        'fallback_used': False, 'native_length_constraints_complete': False,
        'native_length_scope': asset['transformation'].get('length_scope',
            '6703d7894 treats repetition upper bounds above 2000 as unbounded; downstream bounds still required'),
        'pydantic_validation_still_required': True, 'semantic_correctness_verified': False,
        'visual_accuracy_verified': False}
    return grammar, receipt


def bind_local_grammar_runtime(server_executable, output_dir) -> dict:
    """Explicit operator action: bind unchanged grammars to an installed build.

    Native parser initialization/freeing is performed, but no model is loaded,
    executable launched, service contacted or external process stopped. A
    parseable grammar is not an inference/capability compatibility certificate.
    Existing differing registrations are refused instead of overwritten.
    """
    executable = Path(server_executable).expanduser().resolve(strict=True)
    if executable.name.lower() != 'llama-server.exe' or executable.is_symlink() or not executable.is_file():
        raise NativeGrammarProfileError('Select the actual local llama-server.exe explicitly')
    files = {'server_impl_sha256': executable.with_name('llama-server-impl.dll'),
        'llama_dll_sha256': executable.with_name('llama.dll')}
    hashes = {name: _file_sha256(path) for name, path in files.items()}
    output = Path(output_dir).expanduser().resolve()
    prepared = []
    for profile, source in PROFILE_REGISTRY.items():
        asset, _ = _read_profile(Path(source))
        if (asset['profile'] != profile or not isinstance(asset['grammar'], str)
                or hashlib.sha256(asset['grammar'].encode('utf-8')).hexdigest() != asset['grammar_sha256']):
            raise NativeGrammarProfileError('Source grammar asset identity or payload is invalid')
        if not all(isinstance(asset.get(name), str) and _DIGEST.fullmatch(asset[name])
                   for name in ('schema_sha256', 'grammar_sha256')):
            raise NativeGrammarProfileError('Source grammar schema binding is invalid')
        _native_parse(asset['grammar'], files['llama_dll_sha256'])
        asset.update(hashes, source_runtime_directory=str(executable.parent))
        asset['transformation'] = {**asset['transformation'],
            'operator_registered_build': True, 'model_inference_verified': False}
        content = _canonical_json(asset)
        target = output / (profile + '.json')
        if target.exists() and (target.is_symlink() or target.read_bytes() != content):
            raise NativeGrammarProfileError('Existing different grammar registration was not overwritten')
        prepared.append((target, content))
    if hashes != {name: _file_sha256(path) for name, path in files.items()}:
        raise NativeGrammarProfileError('Runtime DLLs changed during registration')
    output.mkdir(parents=True, exist_ok=True)
    for path, content in prepared:
        if not path.exists():
            with path.open('xb') as stream:
                stream.write(content)
    return {'registered_profiles': len(prepared), 'output_dir': str(output),
        'server_executable': str(executable), **hashes, 'native_parse_verified': True,
        'model_weights_loaded': False, 'inference_compatibility_verified': False,
        'downloaded': False, 'external_processes_started_or_stopped': False}
