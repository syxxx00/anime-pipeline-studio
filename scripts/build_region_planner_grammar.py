"""Offline derivation of the region planner GBNF; no model or server is started.

The unchanged rules come from the exact registered legacy asset. This is a
bounded migration, not a general JSON Schema converter: any schema difference
outside the two added request fields, operation and region selection is refused.
Run with --check to verify the committed asset and the installed NULL-vocab
native parser. Without --check, print the candidate JSON asset to stdout.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.native_grammar_profile import (MAX_PROFILE_BYTES, _canonical_json,
    _file_sha256, _native_parse, _read_profile)


LEGACY_PROFILE = 'local_iteration_6703d7894'
PROFILE = 'local_iteration_region_6703d7894'
LEGACY_ASSET_SHA256 = 'b5526f781ac03f81637f0ffa6af3fbe1acfa37f8ce8ca894f26d0000177eca44'
ASSET_DIRECTORY = PROJECT_ROOT / 'app' / 'grammars'
REQUEST_FIELDS = {
    'artifact_id': {'type': 'string', 'minLength': 1, 'maxLength': 120},
    'instruction': {'type': 'string', 'minLength': 1, 'maxLength': 4000},
}
LEGACY_REGION_SCHEMA = {
    'properties': {
        'instruction': {'maxLength': 4000, 'minLength': 1, 'type': 'string'},
        'mask_data_url': {'maxLength': 28000000, 'type': 'string'},
        'steps': {'maximum': 40, 'minimum': 5, 'type': 'integer'},
        'denoise': {'maximum': 0.75, 'minimum': 0.15, 'type': 'number'},
        'source_sha256': {'anyOf': [
            {'pattern': '^[0-9a-f]{64}$', 'type': 'string'}, {'type': 'null'}]},
    },
    'required': ['instruction', 'mask_data_url'],
    'type': 'object',
}


def legacy_schema(schema: dict) -> dict:
    """Reverse only the declared migration, retaining all unrelated schema bytes."""
    result = deepcopy(schema)
    for name, expected in REQUEST_FIELDS.items():
        if result['properties'].pop(name, None) != expected:
            raise ValueError('Unexpected region request field schema: ' + name)
    operations = result['properties']['operation']['enum']
    if not operations or operations.pop() != 'locate_region':
        raise ValueError('Region operation must be the sole appended enum member')
    region = result['$defs']['RegionEditOptions']
    expected_region = deepcopy(LEGACY_REGION_SCHEMA)
    expected_region['properties'].pop('mask_data_url')
    expected_region['properties'].pop('source_sha256')
    expected_region['properties']['region_selection_id'] = {
        'type': 'string', 'minLength': 1, 'maxLength': 80}
    expected_region['required'] = ['instruction', 'region_selection_id']
    expected_region['additionalProperties'] = False
    if region != expected_region:
        raise ValueError('Unexpected region edit schema beyond the declared selection migration')
    result['$defs']['RegionEditOptions'] = deepcopy(LEGACY_REGION_SCHEMA)
    return result


def _literal(value: str) -> str:
    return json.dumps(json.dumps(value, ensure_ascii=False), ensure_ascii=False)


def _stem(prefix: str, name: str) -> str:
    return prefix + '-' + name.replace('_', '-')


def _object_rules(prefix: str, properties: dict, required: list, values: dict) -> dict:
    optional = [name for name in properties if name not in required]
    if not required or not optional:
        raise ValueError('This migration expects required and optional object members')
    rules = {}
    body = '"{" space ' + ' "," space '.join(_stem(prefix, name) + '-kv' for name in required)
    branches = []
    for index, name in enumerate(optional):
        key = _stem(prefix, name)
        branches.append(key + '-kv' + (' ' + key + '-rest' if index + 1 < len(optional) else ''))
        if index + 1 < len(optional):
            following = _stem(prefix, optional[index + 1])
            rules[key + '-rest'] = '( "," space ' + following + '-kv )?' + (
                ' ' + following + '-rest' if index + 2 < len(optional) else '')
    rules[prefix] = body + ' ( "," space ( ' + ' | '.join(branches) + ' ) )? space "}"'
    for name in properties:
        rules[_stem(prefix, name) + '-kv'] = _literal(name) + ' space ":" space ' + values[name]
    return rules


def build_asset(schema: dict, legacy_asset: dict) -> dict:
    """Produce a deterministic candidate after proving the complete schema delta."""
    if legacy_asset['profile'] != LEGACY_PROFILE:
        raise ValueError('The migration requires the registered legacy profile')
    previous_schema = legacy_schema(schema)
    if hashlib.sha256(_canonical_json(previous_schema)).hexdigest() != legacy_asset['schema_sha256']:
        raise ValueError('An unrelated complete schema change cannot reuse legacy grammar rules')
    grammar = legacy_asset['grammar']
    if hashlib.sha256(grammar.encode('utf-8')).hexdigest() != legacy_asset['grammar_sha256']:
        raise ValueError('Legacy grammar digest mismatch')
    rules = {}
    for line in grammar.splitlines():
        name, body = line.split(' ::= ', 1)
        if name in rules:
            raise ValueError('Duplicate legacy GBNF rule')
        rules[name] = body
    prefix = 'response-format-schema'
    values = {name: rules[_stem(prefix, name) + '-kv'].split(' space ":" space ', 1)[1]
              for name in previous_schema['properties']}
    for name, constraints in REQUEST_FIELDS.items():
        key = _stem(prefix, name)
        rules[key] = '"\\\"" char{' + str(constraints['minLength']) + ',' + str(constraints['maxLength']) + '} "\\\""'
        values[name] = key
    rules[prefix + '-operation'] = '(' + ' | '.join(
        _literal(value) for value in schema['properties']['operation']['enum']) + ')'
    rules = {key: value for key, value in rules.items()
             if not (key.startswith(prefix + '-') and key.endswith('-rest'))}
    rules.update(_object_rules(prefix, schema['properties'], schema['required'], values))
    region_prefix = 'ref-defs-RegionEditOptions'
    region = schema['$defs']['RegionEditOptions']
    region_rules = {key: value for key, value in rules.items() if key.startswith(region_prefix + '-')}
    rules = {key: value for key, value in rules.items()
             if key != region_prefix and not key.startswith(region_prefix + '-')}
    values = {}
    for name in ('instruction', 'steps', 'denoise'):
        key = _stem(region_prefix, name)
        values[name] = region_rules[key + '-kv'].split(' space ":" space ', 1)[1]
        if key in region_rules:
            rules[key] = region_rules[key]
    selection = _stem(region_prefix, 'region_selection_id')
    rules[selection] = '"\\\"" char{1,80} "\\\""'
    values['region_selection_id'] = selection
    rules.update(_object_rules(region_prefix, region['properties'], region['required'], values))
    # Reuse every unchanged RHS, including the exact 2000-character workaround.
    grammar = '\n'.join(name + ' ::= ' + rules[name] for name in sorted(rules)) + '\n'
    candidate = {
        'profile': PROFILE,
        'schema_sha256': hashlib.sha256(_canonical_json(schema)).hexdigest(),
        'grammar_sha256': hashlib.sha256(grammar.encode('utf-8')).hexdigest(),
        'grammar': grammar,
        'server_impl_sha256': legacy_asset['server_impl_sha256'],
        'llama_dll_sha256': legacy_asset['llama_dll_sha256'],
        'source_runtime_directory': legacy_asset['source_runtime_directory'],
        'transformation': {
            'parent_profile': LEGACY_PROFILE,
            'parent_asset_sha256': LEGACY_ASSET_SHA256,
            'parent_schema_sha256': legacy_asset['schema_sha256'],
            'unchanged': 'All other complete schema rules and the original exact 2000-character workaround',
            'changed': 'Append locate_region and bounded artifact_id/instruction request fields; region edit accepts a grounded selection ID instead of mask PNG or source digest',
            'method': 'Verified complete schema delta; deterministically rebuild only the two affected object member rules from registered legacy GBNF',
            'length_scope': legacy_asset['transformation']['length_scope'],
            'semantic_correctness_verified': False,
        },
    }
    if len(_canonical_json(candidate)) > MAX_PROFILE_BYTES:
        raise ValueError('Derived grammar profile exceeds the 64 KiB file bound')
    return candidate


def runtime_directory(explicit_directory: Path | None = None) -> Path:
    """Resolve only inside the explicit command, never at import time."""
    if explicit_directory is not None:
        return Path(explicit_directory).expanduser()
    from app.local_authoring_runtime import discover_llama_server
    return discover_llama_server().parent


def __getattr__(name: str):
    # Preserve readers of the old constant without discovery at module import.
    if name == 'RUNTIME_DIRECTORY':
        return runtime_directory()
    raise AttributeError(name)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true', help='Verify the existing asset and native parser; write nothing')
    parser.add_argument('--runtime-directory', type=Path,
                        help='Explicit installed llama.cpp directory; overrides PATH/WinGet discovery')
    args = parser.parse_args(argv)
    from app.local_iteration_planner import proposal_schema
    previous, digest = _read_profile(ASSET_DIRECTORY / (LEGACY_PROFILE + '.json'))
    if digest != LEGACY_ASSET_SHA256:
        raise ValueError('The frozen legacy asset bytes changed')
    asset = build_asset(proposal_schema(), previous)
    directory = runtime_directory(args.runtime_directory)
    for field, filename in (('server_impl_sha256', 'llama-server-impl.dll'), ('llama_dll_sha256', 'llama.dll')):
        if _file_sha256(directory / filename) != asset[field]:
            raise ValueError('Installed runtime identity changed')
    _native_parse(asset['grammar'], directory / 'llama.dll')
    if args.check:
        existing, asset_digest = _read_profile(ASSET_DIRECTORY / (PROFILE + '.json'))
        if existing != asset:
            raise ValueError('Registered region asset differs from the offline derivation')
        print(json.dumps({'profile': PROFILE, 'asset_sha256': asset_digest,
            'schema_sha256': asset['schema_sha256'], 'grammar_sha256': asset['grammar_sha256'],
            'native_parse_verified': True, 'model_weights_loaded': False}, ensure_ascii=False))
    else:
        print(json.dumps(asset, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
