"""Derive three exact text-only picture/method grammars without inference.

Print deterministic assets by default. --check compares saved assets and verifies
the installed NULL-vocab parser after DLL identity checks; it starts no server,
loads no model and writes no file. Legacy assets are not derived or replaced.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.creative_picture_decision import (
    InitialProseDecision, NextProseDecision, ReferenceProseDecision,
    INITIAL_PROSE_DECISION_PROFILE, NEXT_PROSE_DECISION_PROFILE,
    REFERENCE_PROSE_DECISION_PROFILE)
from app.native_grammar_profile import (
    MAX_PROFILE_BYTES, _canonical_json, _file_sha256, _native_parse, _read_profile)
from scripts.build_creative_review_grammar import (
    ASSET_DIRECTORY, LENGTH_SCOPE, RUNTIME_DIRECTORY, RUNTIME_SHA256,
    SOURCE_RUNTIME_DIRECTORY, _grammar, sampler_schema)


SCHEMA_MODELS = {
    INITIAL_PROSE_DECISION_PROFILE: InitialProseDecision,
    NEXT_PROSE_DECISION_PROFILE: NextProseDecision,
    REFERENCE_PROSE_DECISION_PROFILE: ReferenceProseDecision,
}
SCHEMA_SHA256 = {
    INITIAL_PROSE_DECISION_PROFILE: 'a62646730d2d954e4d7f1447e63334a90071a28922a32da2e2963e427334f7fa',
    NEXT_PROSE_DECISION_PROFILE: '71b2cf9167165966b0a3b72c6823d0428767bc44060dfe1205db63a20b760f14',
    REFERENCE_PROSE_DECISION_PROFILE: '74025b4feb1a97d9f69239d6bc0b8d05e62ef973c597d3831d6d117178fc558e',
}


def build_asset(profile: str, schema: dict) -> dict:
    """Compile only the exact committed consumer schema, not arbitrary JSON."""
    if (profile not in SCHEMA_SHA256 or not isinstance(schema, dict)
            or hashlib.sha256(_canonical_json(schema)).hexdigest() != SCHEMA_SHA256[profile]):
        raise ValueError('Unsupported or altered complete picture decision schema')
    grammar = _grammar(sampler_schema(schema), prose_decision=True)
    asset = {
        'profile': profile,
        'schema_sha256': SCHEMA_SHA256[profile],
        'grammar_sha256': hashlib.sha256(grammar.encode('utf-8')).hexdigest(),
        'grammar': grammar,
        **RUNTIME_SHA256,
        'source_runtime_directory': SOURCE_RUNTIME_DIRECTORY,
        'transformation': {
            'changed': 'New independent text-only same-return picture and method consumer; legacy models and assets retain their exact schemas',
            'unchanged': 'Full schema identity, required wrapper fields, complete ProsePictureChoice fields, nested legacy method fields/defaults, nullable choices and all host bounds',
            'method': 'Reuse the bounded review grammar derivation for only these three SHA-bound nested schemas; permit their exact nullable prose/start branches and bounded arrays',
            'length_scope': LENGTH_SCOPE,
            'numeric_scope': 'Optional denoise retains 0 < value <= 1 and host default 0.25; sampler admits decimals without exponents',
            'operation_scope': 'The new sampler excludes sketch without changing nested legacy schema enums; host validators also reject it and require picture for regenerate/handoff and null for keep/stop/local_edit',
            'reference_scope': 'parent/child/neither remain bound reference choices, not user selection; host validates neither operation restrictions, live sources and actual capabilities',
            'input_scope': 'Text-only complete artistic picture plus method decision; no visual permission, resource-admission change or dispatch authority',
            'source_runtime_scope': 'Shared installed DLL diagnostic reference only; derivation is not a picture decision inference receipt and loads no model weights',
            'semantic_correctness_verified': False,
            'visual_accuracy_verified': False,
        },
    }
    if len(_canonical_json(asset)) > MAX_PROFILE_BYTES:
        raise ValueError('Picture decision grammar asset exceeds its registered file bound')
    return asset


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true', help='Verify committed assets and installed zero-model grammar parser')
    parser.add_argument('--profile', choices=tuple(SCHEMA_MODELS), help='Limit derivation/checking to one exact consumer')
    args = parser.parse_args()
    selected = (args.profile,) if args.profile else tuple(SCHEMA_MODELS)
    assets = {profile: build_asset(profile, SCHEMA_MODELS[profile].model_json_schema())
              for profile in selected}
    if not args.check:
        print(json.dumps(assets, ensure_ascii=False, indent=2))
        return
    files = {'server_impl_sha256': RUNTIME_DIRECTORY / 'llama-server-impl.dll',
             'llama_dll_sha256': RUNTIME_DIRECTORY / 'llama.dll'}
    for field, path in files.items():
        if _file_sha256(path) != RUNTIME_SHA256[field]:
            raise ValueError('Installed adjacent runtime DLL identity changed')
    receipts = []
    for profile, candidate in assets.items():
        existing, asset_digest = _read_profile(ASSET_DIRECTORY / (profile + '.json'))
        if existing != candidate:
            raise ValueError('Committed picture decision asset differs from deterministic derivation')
        _native_parse(candidate['grammar'], files['llama_dll_sha256'])
        receipts.append({'profile': profile, 'asset_sha256': asset_digest,
                         'schema_sha256': candidate['schema_sha256'],
                         'grammar_sha256': candidate['grammar_sha256'],
                         'native_parse_verified': True, 'parser_vocab': None,
                         'model_weights_loaded': False})
    for field, path in files.items():
        if _file_sha256(path) != RUNTIME_SHA256[field]:
            raise ValueError('Installed runtime DLL changed during zero-model validation')
    print(json.dumps(receipts, ensure_ascii=False))


if __name__ == '__main__':
    main()
