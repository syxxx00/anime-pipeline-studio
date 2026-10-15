"""Derive the exact optional-read picture grammar without model inference.

The independent asset retains the frozen ProsePictureChoice and initial-model
definitions. --check verifies the committed asset and installed NULL-vocab
parser after DLL identity checks; it loads no weights, starts no service and
writes no file. Default output is the deterministic asset JSON.
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

from app.creative_picture_decision import InitialProseDecision, INITIAL_PROSE_DECISION_PROFILE
from app.creative_research_decision import ResearchProseDecision, RESEARCH_PROSE_DECISION_PROFILE
from app.native_grammar_profile import (
    MAX_PROFILE_BYTES, _canonical_json, _file_sha256, _native_parse, _read_profile)
from scripts.build_creative_picture_decision_grammar import build_asset as build_initial_asset
from scripts.build_creative_review_grammar import (
    ASSET_DIRECTORY, LENGTH_SCOPE, RUNTIME_DIRECTORY, RUNTIME_SHA256,
    SOURCE_RUNTIME_DIRECTORY, _literal)


SCHEMA_SHA256 = '027e60f5f931a8f6a884a845f466561c117cf4b0c920e4670cdd909dca03405f'


def _research_grammar() -> str:
    """Reuse frozen nested definitions, with separate exact read/compose roots.

    This is deliberately not a general schema converter. Query strings and
    array bounds are independently exact; host validation still owns catalogue
    membership, authority and the single-round read limit.
    """
    initial = build_initial_asset(INITIAL_PROSE_DECISION_PROFILE,
                                  InitialProseDecision.model_json_schema())
    rules = dict(line.split(' ::= ', 1) for line in initial['grammar'].splitlines())
    rules = {name: body for name, body in rules.items()
             if name != 'root' and not name.startswith('response-format-schema')}
    rules.update({
        'research-artwork-id': r'"\"" char{1,80} "\""',
        'research-purpose': r'"\"" char{12,800} "\""',
        'research-query': ('"{" space ' + _literal('artwork_id') + ' space ":" space research-artwork-id '
                           '"," space ' + _literal('purpose') + ' space ":" space research-purpose space "}"'),
        'research-query-array': '"[" space research-query ("," space research-query)? space "]"',
        'research-starting-choice': '(ref-defs-StartingModelChoice | "null")',
        'research-compose': ('"{" space ' + _literal('picture') + ' space ":" space ref-defs-ProsePictureChoice '
                             '"," space ' + _literal('starting_choice') + ' space ":" space research-starting-choice '
                             '"," space ' + _literal('research_queries') + ' space ":" space "[" space "]" space "}"'),
        'research-read': ('"{" space ' + _literal('picture') + ' space ":" space "null" '
                          '"," space ' + _literal('starting_choice') + ' space ":" space "null" '
                          '"," space ' + _literal('research_queries') + ' space ":" space research-query-array space "}"'),
        'response-format-schema': '(research-compose | research-read)',
        'root': 'response-format-schema',
    })
    return '\n'.join(name + ' ::= ' + rules[name] for name in sorted(rules)) + '\n'


def build_asset(profile: str, schema: dict) -> dict:
    if (profile != RESEARCH_PROSE_DECISION_PROFILE or not isinstance(schema, dict)
            or hashlib.sha256(_canonical_json(schema)).hexdigest() != SCHEMA_SHA256):
        raise ValueError('Unsupported or altered complete research picture decision schema')
    grammar = _research_grammar()
    asset = {
        'profile': profile,
        'schema_sha256': SCHEMA_SHA256,
        'grammar_sha256': hashlib.sha256(grammar.encode('utf-8')).hexdigest(),
        'grammar': grammar,
        **RUNTIME_SHA256,
        'source_runtime_directory': SOURCE_RUNTIME_DIRECTORY,
        'transformation': {
            'changed': 'New independent optional-read or same-return initial-picture text consumer; no legacy schema or grammar changes',
            'unchanged': 'Complete schema SHA, frozen nested ProsePictureChoice and StartingModelChoice definitions and bounds, exact required wrapper fields and host validation',
            'method': 'Reuse the exact existing InitialProseDecision nested grammar definitions and independently derive two exclusive root branches for this SHA-bound new schema',
            'length_scope': LENGTH_SCOPE,
            'query_scope': 'Sampler retains artwork_id 1..80, purpose 12..800 and read array 1..2; compose has zero queries. Host separately validates actual catalogue IDs and permits at most one read round.',
            'decision_scope': 'compose requires a complete non-null picture with nullable initial choice and an empty query list; read requires null picture, null initial choice and one or two queries. No final image draft is inferred from a read return.',
            'input_scope': 'Text-only current intention and attributed research catalogue; catalogue entries are not current scene instructions, pixel observations or approval. No service, model, material fetch or dispatch authority is granted.',
            'source_runtime_scope': 'Installed DLL diagnostic reference only; NULL-vocab parsing loads no model weights and is not a creative inference receipt',
            'semantic_correctness_verified': False,
            'visual_accuracy_verified': False,
        },
    }
    if len(_canonical_json(asset)) > MAX_PROFILE_BYTES:
        raise ValueError('Research picture decision grammar exceeds its registered file bound')
    return asset


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true', help='Verify committed asset and installed zero-model parser')
    args = parser.parse_args()
    candidate = build_asset(RESEARCH_PROSE_DECISION_PROFILE,
                            ResearchProseDecision.model_json_schema())
    if not args.check:
        print(json.dumps(candidate, ensure_ascii=False, indent=2))
        return
    existing, asset_digest = _read_profile(ASSET_DIRECTORY / (RESEARCH_PROSE_DECISION_PROFILE + '.json'))
    if existing != candidate:
        raise ValueError('Committed research picture asset differs from deterministic derivation')
    files = {'server_impl_sha256': RUNTIME_DIRECTORY / 'llama-server-impl.dll',
             'llama_dll_sha256': RUNTIME_DIRECTORY / 'llama.dll'}
    for field, path in files.items():
        if _file_sha256(path) != RUNTIME_SHA256[field]:
            raise ValueError('Installed adjacent runtime DLL identity changed')
    _native_parse(candidate['grammar'], files['llama_dll_sha256'])
    for field, path in files.items():
        if _file_sha256(path) != RUNTIME_SHA256[field]:
            raise ValueError('Installed runtime DLL changed during zero-model validation')
    print(json.dumps({'profile': RESEARCH_PROSE_DECISION_PROFILE, 'asset_sha256': asset_digest,
                     'schema_sha256': candidate['schema_sha256'], 'grammar_sha256': candidate['grammar_sha256'],
                     'native_parse_verified': True, 'parser_vocab': None,
                     'model_weights_loaded': False}, ensure_ascii=False))


if __name__ == '__main__':
    main()
