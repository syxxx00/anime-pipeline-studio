"""Offline GBNF for exact creative-turn schemas; never load a model.

This is a frozen, bounded derivation, not a general JSON Schema converter.
The two visual-review grammars defer string maxLength to the unchanged schema
and host validation. The text-only NextChoice and ReferenceNextChoice retain
their verified native string bounds. Print candidates by default; --check verifies committed assets,
adjacent installed DLL identities and the NULL-vocab parser.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
from typing import Literal

from pydantic import BaseModel, Field


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.creative_result_review import RESULT_COMPARISON_SCHEMA
from app.creative_request_understanding import RequestReading
from app.creative_directions import DirectionPlanProposal, fixed_picture_profile, fixed_picture_schema
from app.creative_reference import ReferenceNextChoice
from app.grounded_region import RegionLocalization
from app.native_grammar_profile import (
    MAX_PROFILE_BYTES, _canonical_json, _file_sha256, _native_parse, _read_profile)


OBSERVATION_PROFILE = 'local_creative_observation_6703d7894'
COMPARISON_PROFILE = 'local_creative_result_comparison_6703d7894'
NEXT_CHOICE_PROFILE = 'local_creative_next_choice_6703d7894'
REFERENCE_NEXT_CHOICE_PROFILE = 'local_creative_reference_next_choice_6703d7894'
REQUEST_READING_PROFILE = 'local_creative_request_reading_6703d7894'
REGION_LOCALIZATION_PROFILE = 'local_creative_region_localization_6703d7894'
DIRECTION_PLAN_PROFILE = 'local_creative_direction_plan_6703d7894'
FIXED_PICTURE_PROFILES = tuple(fixed_picture_profile(count) for count in range(1, 7))
SCHEMA_SHA256 = {
    OBSERVATION_PROFILE: 'e054e4f13162581b49860f9dfd7c552ae715a21b439446dcd03afc9da859a9bd',
    COMPARISON_PROFILE: '1b140fe080d5606c58a57231a7c8d70babd43ddbf6adab651f251025e3ef9e45',
    NEXT_CHOICE_PROFILE: 'c0704da2539743cee7c337633414c52dddea057fdb5231f97a3664fcbaed498d',
    REFERENCE_NEXT_CHOICE_PROFILE: 'b4f9f92be7fb270e31365afe6f423e3332d6002d632c855f3082697f0b4e0e72',
    REQUEST_READING_PROFILE: '396dbdec1d9b5d864cabe03a9b8abafc687a19d599d6b82faba332ce10f0506a',
    REGION_LOCALIZATION_PROFILE: '12ab9c976c84e09fc0d38ecab8901dd5413a39460fc9f6ea74ebffaa44e6e3f2',
    DIRECTION_PLAN_PROFILE: 'e20522a17b98605f2bbd49bda4251ff4772274716e4dab220278934b9a660e3b',
    'local_creative_fixed_pictures_1_6703d7894': 'ab5fad27e6485102b0b1699b26a4ebbfc479c11f0f03ef94735fa98e2c487549',
    'local_creative_fixed_pictures_2_6703d7894': '4b2ee305e55bc99a140f93f6c61344b6df401af901d3fa36f14e7295410db883',
    'local_creative_fixed_pictures_3_6703d7894': 'bbcd6057b6d99886173331d13260cac8340df13de13f82c32f1b3195907c9ecf',
    'local_creative_fixed_pictures_4_6703d7894': 'de574f9d2975ae8a1bfaaa6adc24e33a0e1396ee1adbc01393447cce81bb2e77',
    'local_creative_fixed_pictures_5_6703d7894': 'feea2859debd79e50e941dce11096fa22dfe97ba3d04959fa44d0a120824eba5',
    'local_creative_fixed_pictures_6_6703d7894': '90b0b45262783c88349bf0f33ef8cdad2b8e2132b4cbdef85d1a2ca5b6922d57',
}
RUNTIME_SHA256 = {
    'server_impl_sha256': '1ebbc51e250a8d1f279c345960911fa3e469e88b812de996bbe07c8efc0771a6',
    'llama_dll_sha256': 'c4dd46b7312abdc20fe729384cb79939721d59d626cc17e682b06b7facf78c12',
}
ASSET_DIRECTORY = PROJECT_ROOT / 'app' / 'grammars'
SOURCE_RUNTIME_DIRECTORY = 'work/local_authoring_runtime/20261012_090018_3f94130bc840'
LENGTH_SCOPE = (
    'Only string maxLength is omitted from sampler GBNF to avoid the installed '
    '6703d7894 repetition parser boundary. String minLength and all array '
    'minItems/maxItems remain enforced by this grammar. The complete original '
    'schema, text system/prompt and host Pydantic retain every string maximum; '
    'overlong responses must be rejected without truncation or repair.')
DENOISE_DECIMAL_GBNF = r'("0." [0-9]* [1-9] [0-9]* | "1" ("." "0"+)?)'


class PixelObservation(BaseModel):
    # Same name and exact fields as the consumer, without importing its session
    # module (which pulls the circular session/fixture dependency graph).
    model_config = {'extra': 'forbid'}
    visible_relationships: str = Field(max_length=2000)
    potential_value: str = Field(max_length=1200)
    problems: list[str] = Field(max_length=8)
    uncertain: list[str] = Field(max_length=8)


class NextChoice(BaseModel):
    # Exact text-only consumer declaration, isolated from session imports.
    model_config = {'extra': 'forbid'}
    operation: Literal['keep', 'regenerate', 'handoff', 'sketch', 'local_edit', 'stop']
    target: Literal['newbie', 'illustrious', 'anima', 'z_image_turbo']
    objective: str = Field(min_length=3, max_length=1200)
    reason: str = Field(min_length=3, max_length=1500)
    preserve: str = Field(max_length=1200)
    denoise: float = Field(default=.25, gt=0, le=1)


def observation_schema() -> dict:
    return PixelObservation.model_json_schema()


def next_choice_schema() -> dict:
    return NextChoice.model_json_schema()


def reference_next_choice_schema() -> dict:
    return ReferenceNextChoice.model_json_schema()


def request_reading_schema() -> dict:
    return RequestReading.model_json_schema()


def region_localization_schema() -> dict:
    return RegionLocalization.model_json_schema()


def direction_plan_schema() -> dict:
    return DirectionPlanProposal.model_json_schema()


def sampler_schema(schema: dict) -> dict:
    """Copy the schema and remove only string maximum lengths."""
    result = deepcopy(schema)

    def visit(value):
        if isinstance(value, dict):
            if value.get('type') == 'string':
                value.pop('maxLength', None)
            for child in value.values():
                visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)

    visit(result)
    return result


def _literal(value: str) -> str:
    return json.dumps(json.dumps(value, ensure_ascii=False), ensure_ascii=False)


def _grammar(schema: dict, *, next_choice: bool = False, reference_next_choice: bool = False,
             prose_decision: bool = False) -> str:
    """Compile only the nodes present in the SHA-bound consumer schemas."""
    text_choice = next_choice or reference_next_choice
    rules = {
        'root': 'response-format-schema',
        'space': r'[ \t\n\r]*',
        'char': r'[^"\\\x00-\x1F] | [\\] (["\\bfnrt] | "u" [0-9a-fA-F]{4})',
    }
    definitions = schema.get('$defs', {})

    def visit(node, name):
        kind = node.get('type')
        metadata = {'title', 'description'}
        if 'anyOf' in node:
            nullable_prose = (prose_decision and set(node) - metadata == {'anyOf'}
                and node['anyOf'] in ([{'$ref': '#/$defs/ProsePictureChoice'}, {'type': 'null'}],
                                      [{'$ref': '#/$defs/StartingModelChoice'}, {'type': 'null'}]))
            if not nullable_prose and (set(node) != {'anyOf'} or node['anyOf'] != [
                    {'$ref': '#/$defs/PixelRectangle'}, {'$ref': '#/$defs/PixelPolygon'}]):
                raise ValueError('Unsupported localization alternatives')
            branches = []
            for index, child in enumerate(node['anyOf']):
                branch = name + '-alternative-' + str(index)
                visit(child, branch)
                branches.append(branch)
            body = '(' + ' | '.join(branches) + ')'
        elif '$ref' in node:
            if set(node) != {'$ref'} or not node['$ref'].startswith('#/$defs/'):
                raise ValueError('Unsupported creative review reference')
            target = node['$ref'].removeprefix('#/$defs/')
            if target not in definitions:
                raise ValueError('Unsupported creative review reference target')
            body = 'ref-defs-' + target
        elif kind == 'object':
            keys = list(node['properties'])
            required = node.get('required')
            optional = keys[len(required):]
            permitted_optional = (([], ['denoise'], ['failure_risks']) if prose_decision
                                  else ((['denoise'],) if text_choice else ([],)))
            if (set(node) - metadata - {'$defs', 'type', 'properties', 'required', 'additionalProperties'}
                    or node.get('additionalProperties') is not False
                    or keys[:len(required)] != required
                    or optional not in permitted_optional):
                raise ValueError('Unsupported creative review object shape')
            members = []
            for key, child in node['properties'].items():
                stem = name + '-' + key.replace('_', '-')
                visit(child, stem)
                rules[stem + '-kv'] = _literal(key) + ' space ":" space ' + stem
                if key in required:
                    members.append(stem + '-kv')
            body = '"{" space ' + ' "," space '.join(members)
            if optional:
                body += ' ( "," space ' + name + '-' + optional[0].replace('_', '-') + '-kv )?'
            body += ' space "}"'
        elif kind == 'array':
            if set(node) - metadata - {'type', 'items', 'prefixItems', 'minItems', 'maxItems'}:
                raise ValueError('Unsupported creative review array constraint')
            minimum, maximum = node.get('minItems', 0), node['maxItems']
            if 'prefixItems' in node:
                if (minimum, maximum) != (2, 2) or len(node['prefixItems']) != 2 or 'items' in node:
                    raise ValueError('Unsupported localization coordinate pair')
                members = []
                for index, child in enumerate(node['prefixItems']):
                    member = name + '-coordinate-' + str(index)
                    visit(child, member)
                    members.append(member)
                body = '"[" space ' + ' "," space '.join(members) + ' space "]"'
            else:
                permitted_bounds = ((0, 4), (0, 6), (1, 6), (0, 8), (0, 30), (0, 60), (3, 32))
                if prose_decision:
                    permitted_bounds += ((0, 3), (0, 12))
                if (minimum, maximum) not in permitted_bounds:
                    raise ValueError('Unsupported creative review array bounds')
                item = name + '-item'
                visit(node['items'], item)
                if minimum:
                    content = item + ' ("," space ' + item + '){' + str(minimum - 1) + ',' + str(maximum - 1) + '}'
                else:
                    content = '(' + item + ' ("," space ' + item + '){0,' + str(maximum - 1) + '})?'
                body = '"[" space ' + content + ' space "]"'
        elif kind == 'string':
            allowed = {'type', 'minLength', 'enum', 'const', 'pattern'} | ({'maxLength'} if text_choice else set())
            if set(node) - metadata - allowed:
                raise ValueError('Unsupported creative review string constraint')
            if 'enum' in node:
                enums = (['keep', 'regenerate', 'handoff', 'sketch', 'local_edit', 'stop'],
                         ['newbie', 'illustrious', 'anima', 'z_image_turbo']) if text_choice else (['parent', 'child', 'neither'],)
                if reference_next_choice:
                    enums += (['parent', 'child', 'neither'],)
                if not text_choice:
                    enums += (['required', 'preferred', 'context'],
                              ['unspecified', 'separate_candidates', 'single_composite'],
                              ['located', 'unlocatable'])
                if prose_decision:
                    enums += (['keep', 'regenerate', 'handoff', 'sketch', 'local_edit', 'stop'],
                              ['newbie', 'illustrious', 'anima', 'z_image_turbo'])
                if node['enum'] not in enums:
                    raise ValueError('Unsupported creative review enum')
                values = [value for value in node['enum'] if not (prose_decision and value == 'sketch')]
                body = '(' + ' | '.join(_literal(value) for value in values) + ')'
            elif 'const' in node:
                if set(node) - metadata != {'type', 'const'} or node['const'] not in ('rectangle', 'polygon'):
                    raise ValueError('Unsupported localization shape kind')
                body = _literal(node['const'])
            elif 'pattern' in node:
                if set(node) - metadata != {'type', 'pattern'} or node['pattern'] != '^[0-9a-f]{64}$':
                    raise ValueError('Unsupported localization string pattern')
                body = r'"\"" [0-9a-f]{64} "\""'
            else:
                minimum = node.get('minLength', 0)
                if text_choice:
                    if minimum not in (0, 3) or node.get('maxLength') not in (1200, 1500):
                        raise ValueError('Unsupported NextChoice string bounds')
                    quote = json.dumps('"')
                    body = quote + ' char{' + str(minimum) + ',' + str(node['maxLength']) + '} ' + quote
                elif prose_decision and minimum in (3, 80):
                    body = r'"\"" char{' + str(minimum) + r'} char* "\""'
                elif minimum == 20:
                    body = r'"\"" char{20} char* "\""'
                elif minimum not in (0, 1):
                    raise ValueError('Unsupported creative review string minimum')
                else:
                    body = r'"\"" char' + ('+' if minimum else '*') + r' "\""'
        elif kind == 'number' and (text_choice or prose_decision):
            if (set(node) - metadata != {'type', 'default', 'exclusiveMinimum', 'maximum'}
                    or node['default'] != .25 or node['exclusiveMinimum'] != 0 or node['maximum'] != 1):
                raise ValueError('Unsupported NextChoice denoise constraints')
            body = DENOISE_DECIMAL_GBNF
        elif kind == 'null' and prose_decision:
            if set(node) - metadata != {'type'}:
                raise ValueError('Unsupported prose nullable constraint')
            body = '"null"'
        elif kind == 'integer':
            if set(node) - metadata != {'type', 'minimum', 'maximum'} or node['minimum'] != 0:
                raise ValueError('Unsupported creative integer constraints')
            if node['maximum'] == 120:
                body = r'([0-9] | [1-9] [0-9] | "1" [0-1] [0-9] | "120")'
            elif node['maximum'] == 40000000:
                body = r'([0-9] | [1-9] [0-9]{1,6} | [1-3] [0-9]{7} | "40000000")'
            else:
                raise ValueError('Unsupported creative integer maximum')
        else:
            raise ValueError('Unsupported creative review JSON type')
        rules[name] = body

    for key, definition in definitions.items():
        visit(definition, 'ref-defs-' + key)
    visit(schema, 'response-format-schema')
    return '\n'.join(name + ' ::= ' + rules[name] for name in sorted(rules)) + '\n'


def build_asset(profile: str, schema: dict) -> dict:
    """Reject any unknown or altered complete schema before derivation."""
    if (not isinstance(profile, str) or profile not in SCHEMA_SHA256 or not isinstance(schema, dict)
            or hashlib.sha256(_canonical_json(schema)).hexdigest() != SCHEMA_SHA256[profile]):
        raise ValueError('Unsupported or altered complete creative review schema')
    next_choice = profile == NEXT_CHOICE_PROFILE
    reference_next_choice = profile == REFERENCE_NEXT_CHOICE_PROFILE
    grammar = _grammar(schema if next_choice or reference_next_choice else sampler_schema(schema),
                       next_choice=next_choice, reference_next_choice=reference_next_choice)
    asset = {
        'profile': profile,
        'schema_sha256': SCHEMA_SHA256[profile],
        'grammar_sha256': hashlib.sha256(grammar.encode('utf-8')).hexdigest(),
        'grammar': grammar,
        **RUNTIME_SHA256,
        'source_runtime_directory': SOURCE_RUNTIME_DIRECTORY,
        'transformation': {
            'changed': 'Defer only sampler string maxLength; retain all complete source schema bytes and SHA binding',
            'unchanged': 'JSON types, exact keys, required members, enum, string minLength and every array minItems/maxItems',
            'method': 'Deterministic pure-JSON GBNF for only two frozen schemas, using schema member order',
            'length_scope': LENGTH_SCOPE,
            'semantic_correctness_verified': False,
            'visual_accuracy_verified': False,
        },
    }
    if next_choice:
        asset['transformation'] = {
            'changed': 'Derive an explicit text-only NextChoice grammar; no source schema constraint changes',
            'unchanged': 'Exact keys, required first five members, optional denoise, both enums, all string minima/maxima, numeric 0 < denoise <= 1 and host default 0.25',
            'method': 'Normal-turn text-only preflight compatibility check for the frozen NextChoice schema; not an observed NextChoice failure and not validation of general automatic schema conversion',
            'length_scope': 'All original NextChoice string bounds retained: objective 3..1200, reason 3..1500, preserve 0..1200; equivalent complete grammar parsed in the exact installed NULL-vocab sampler. Complete schema and host Pydantic validation remain authoritative; reject without repair.',
            'numeric_scope': 'Sampler accepts valid JSON decimal spelling without exponents, restricted to 0 < denoise <= 1; omitting denoise preserves the original host default 0.25. The unchanged host schema retains original numeric bounds and default.',
            'input_scope': 'Text-only ordinary next-turn selection; no visual profile permission or resource changes',
            'source_runtime_scope': 'The source directory identifies diagnostics for this exact installed DLL build; it is not a NextChoice response or failure receipt',
            'semantic_correctness_verified': False,
            'visual_accuracy_verified': False,
        }
    if reference_next_choice:
        asset['transformation'] = {
            'changed': 'Derive an independent text-only ReferenceNextChoice grammar; the old NextChoice schema and asset are unchanged',
            'unchanged': 'Exact keys, required first six members, optional denoise, reference/operation/target enums, all string minima/maxima, numeric 0 < denoise <= 1 and host default 0.25',
            'method': 'Deterministic preflight grammar for the exact frozen ReferenceNextChoice consumer; not a general JSON Schema converter',
            'length_scope': 'All original text-choice bounds retained: objective 3..1200, reason 3..1500, preserve 0..1200. The complete schema and host validation remain authoritative; reject without truncation or repair.',
            'numeric_scope': 'Sampler accepts JSON decimals without exponents, restricted to 0 < denoise <= 1; omitting denoise preserves the unchanged host default 0.25.',
            'input_scope': 'Text-only reference and purpose-operation choice; no visual profile permission or resource-admission changes',
            'reference_scope': 'The sampler preserves the parent/child/neither enum only. Host validation must check reference availability, actual source/condition bindings and that neither permits only regenerate or stop; syntax does not certify purpose, quality or user approval.',
            'source_runtime_scope': 'Shared installed DLL diagnostic reference only, not a reference-choice response or failure receipt; no model weights are loaded by derivation.',
            'semantic_correctness_verified': False,
            'visual_accuracy_verified': False,
        }
    if profile in (REQUEST_READING_PROFILE, REGION_LOCALIZATION_PROFILE):
        asset['source_runtime_directory'] = 'work/local_authoring_runtime/20261013_052709_51e101fad0cd'
        asset['transformation']['method'] = 'Deterministic GBNF for the exact source-reading or localization consumer; fixed integer bounds, polygon/rectangle alternatives, coordinate tuples and array bounds are retained.'
        asset['transformation']['input_scope'] = ('Original-source text reading, not artwork authoring or approval' if profile == REQUEST_READING_PROFILE else 'One unchanged bound original PNG; identity, SHA, dimensions and geometry are revalidated by bind_localization and rasterize_localization')
    if profile == DIRECTION_PLAN_PROFILE:
        asset['transformation']['method'] = 'Deterministic GBNF for the exact collection-direction proposal schema; per-item minimums and 1..6 proposal / 0..6 uncertainty bounds are retained.'
        asset['transformation']['input_scope'] = 'Text-only artistic collection proposals. The host allocates each item separately; schema and plan do not certify understanding, distinct images, quality or user approval.'
        asset['transformation']['source_runtime_scope'] = 'Shared installed DLL diagnostic reference, not a completed direction-plan inference receipt.'
    if profile in FIXED_PICTURE_PROFILES:
        asset['transformation']['method'] = 'Deterministic GBNF for one of six exact SHA-bound fixed-picture schemas; all assigned picture keys, required per-picture title/approach, collection rationale and 0..6 uncertainties are retained.'
        asset['transformation']['input_scope'] = 'Text-only proposals for the host-assigned exact picture count. Fixed keys do not certify one coherent picture per entry, artistic difference, quality, identity or user approval.'
        asset['transformation']['source_runtime_scope'] = 'Shared installed DLL diagnostic reference only; no fixed-picture inference or model weights are loaded by this derivation.'
    if len(_canonical_json(asset)) > MAX_PROFILE_BYTES:
        raise ValueError('Creative review grammar exceeds the 64 KiB asset bound')
    return asset


def runtime_directory(explicit_directory: Path | None = None) -> Path:
    """Resolve only when requested; importing/printing assets does no discovery."""
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
    parser.add_argument('--check', action='store_true', help='Verify committed assets and zero-model native parsers; write nothing')
    parser.add_argument('--write', choices=['next_choice', 'reference_next_choice', 'request_reading', 'region_localization', 'direction_plan', 'fixed_pictures'], action='append',
                        help='Generate only the selected deterministic asset after installed-parser verification')
    parser.add_argument('--runtime-directory', type=Path,
                        help='Explicit installed llama.cpp directory; overrides PATH/WinGet discovery')
    args = parser.parse_args(argv)
    if args.check and args.write:
        parser.error('--check and --write are separate operations')
    assets = {OBSERVATION_PROFILE: build_asset(OBSERVATION_PROFILE, observation_schema()),
              COMPARISON_PROFILE: build_asset(COMPARISON_PROFILE, RESULT_COMPARISON_SCHEMA),
              NEXT_CHOICE_PROFILE: build_asset(NEXT_CHOICE_PROFILE, next_choice_schema()),
              REFERENCE_NEXT_CHOICE_PROFILE: build_asset(REFERENCE_NEXT_CHOICE_PROFILE, reference_next_choice_schema()),
              REQUEST_READING_PROFILE: build_asset(REQUEST_READING_PROFILE, request_reading_schema()),
              REGION_LOCALIZATION_PROFILE: build_asset(REGION_LOCALIZATION_PROFILE, region_localization_schema()),
              DIRECTION_PLAN_PROFILE: build_asset(DIRECTION_PLAN_PROFILE, direction_plan_schema()),
              **{fixed_picture_profile(count): build_asset(fixed_picture_profile(count),
                                                          fixed_picture_schema(count).model_json_schema())
                 for count in range(1, 7)}}
    if not args.check and not args.write:
        print(json.dumps(assets, ensure_ascii=False, indent=2))
        return
    if args.write:
        names = {'next_choice': (NEXT_CHOICE_PROFILE,), 'request_reading': (REQUEST_READING_PROFILE,),
                 'reference_next_choice': (REFERENCE_NEXT_CHOICE_PROFILE,),
                 'region_localization': (REGION_LOCALIZATION_PROFILE,), 'direction_plan': (DIRECTION_PLAN_PROFILE,),
                 'fixed_pictures': FIXED_PICTURE_PROFILES}
        assets = {profile: assets[profile] for name in args.write for profile in names[name]}
    receipts = []
    for profile, candidate in assets.items():
        if args.check:
            existing, asset_digest = _read_profile(ASSET_DIRECTORY / (profile + '.json'))
            if existing != candidate:
                raise ValueError('Committed creative review asset differs from deterministic derivation')
        else:
            asset_digest = None  # Assigned only after every parser check succeeds.
        receipts.append({'profile': profile, 'asset_sha256': asset_digest,
                         'schema_sha256': candidate['schema_sha256'],
                         'grammar_sha256': candidate['grammar_sha256'],
                         'native_parse_verified': True, 'parser_vocab': None,
                         'model_weights_loaded': False})
    directory = runtime_directory(args.runtime_directory)
    files = {'server_impl_sha256': directory / 'llama-server-impl.dll',
             'llama_dll_sha256': directory / 'llama.dll'}
    for field, path in files.items():
        if _file_sha256(path) != RUNTIME_SHA256[field]:
            raise ValueError('Installed adjacent runtime DLL identity changed')
    for candidate in assets.values():
        _native_parse(candidate['grammar'], files['llama_dll_sha256'])
    for field, path in files.items():
        if _file_sha256(path) != RUNTIME_SHA256[field]:
            raise ValueError('Installed runtime DLL changed during zero-model validation')
    if args.write:
        for receipt in receipts:
            profile = receipt['profile']
            destination = ASSET_DIRECTORY / (profile + '.json')
            destination.write_text(json.dumps(assets[profile], ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
            receipt['asset_sha256'] = _read_profile(destination)[1]
    print(json.dumps(receipts, ensure_ascii=False))


if __name__ == '__main__':
    main()
