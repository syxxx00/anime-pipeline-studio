"""Lossless JSON sharing for model input, not a summary or understanding proof.

Only identical serialized subtrees are shared. Inline references are protocol
objects; colliding original objects are escaped as an atomic literal. All
definitions travel with the value. Archives and source files are never changed.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from hashlib import sha256
import json
import math
from typing import Any, Callable


ENCODING = 'lossless-shared-original-json/v1'
REFERENCE = 'original_json_reference'
LITERAL = 'original_json_literal'
_RULE = ('Not a summary: every original value is included. A single-key '
         'original_json_reference refers to definitions[id].value. '
         'original_json_literal contains an untouched original subtree; '
         'do not interpret reference-like objects inside that literal.')


def json_bytes(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, allow_nan=False,
                      separators=(',', ':')).encode('utf-8')


def _validate(value: Any, check: Callable[[], None]) -> None:
    check()
    if type(value) is dict:
        if any(type(key) is not str for key in value):
            raise ValueError('Original JSON object keys must already be strings.')
        for item in value.values():
            _validate(item, check)
    elif type(value) is list:
        for item in value:
            _validate(item, check)
    elif type(value) not in (str, int, float, bool, type(None)):
        raise ValueError('Only original JSON types can be shared; no implicit conversion.')
    elif type(value) is float and not math.isfinite(value):
        raise ValueError('Original JSON must be finite.')


def _reserved(value: Any) -> bool:
    return type(value) is dict and (set(value) == {REFERENCE} or set(value) == {LITERAL})


@dataclass(frozen=True)
class SharedOriginalJson:
    value: Any
    encoding: str
    original_bytes: int
    wire_bytes: int

    @property
    def shared(self) -> bool:
        return self.encoding == ENCODING


def restore_original_json(value: Any, *, encoding: str = 'raw-json') -> Any:
    """Explicit protocol decoding; raw user objects never select their encoding."""
    _validate(value, lambda: None)
    if encoding == 'raw-json':
        return json.loads(json_bytes(value))
    expected = {'encoding', 'reference_rule', 'original_json', 'definitions',
                'original_json_sha256', 'original_json_utf8_bytes', 'is_summary'}
    if (encoding != ENCODING or type(value) is not dict or set(value) != expected
            or value['encoding'] != ENCODING or value['reference_rule'] != _RULE
            or value['is_summary'] is not False or type(value['definitions']) is not dict):
        raise ValueError('Unknown or incomplete shared-original JSON encoding.')
    definitions, resolved, visiting, used = value['definitions'], {}, set(), set()

    def decode(node):
        if type(node) is dict and set(node) == {LITERAL}:
            # Whole subtree is original data, including any nested fake refs.
            return json.loads(json_bytes(node[LITERAL]))
        if type(node) is dict and set(node) == {REFERENCE}:
            identifier = node[REFERENCE]
            if type(identifier) is not str or identifier not in definitions or identifier in visiting:
                raise ValueError('Unknown or cyclic original JSON reference.')
            used.add(identifier)
            if identifier not in resolved:
                entry = definitions[identifier]
                if type(entry) is not dict or set(entry) != {'json_sha256', 'value'}:
                    raise ValueError('Incomplete original JSON definition.')
                visiting.add(identifier)
                original = decode(entry['value'])
                visiting.remove(identifier)
                if sha256(json_bytes(original)).hexdigest() != entry['json_sha256']:
                    raise ValueError('Original JSON definition identity changed.')
                resolved[identifier] = original
            # Repeated uses do not alias mutable reconstructed objects.
            return json.loads(json_bytes(resolved[identifier]))
        if type(node) is dict:
            return {key: decode(item) for key, item in node.items()}
        if type(node) is list:
            return [decode(item) for item in node]
        return node

    original = decode(value['original_json'])
    raw = json_bytes(original)
    if (used != set(definitions) or type(value['original_json_utf8_bytes']) is not int
            or len(raw) != value['original_json_utf8_bytes']
            or sha256(raw).hexdigest() != value['original_json_sha256']):
        raise ValueError('Shared originals do not restore the complete original JSON.')
    return original


def share_original_json(value: Any, *, check_cancel: Callable[[], None] | None = None,
                        min_fragment_bytes: int = 512) -> SharedOriginalJson:
    """Share only repeated subtrees with >=256 bytes actual net saving.

    No gain returns the exact raw JSON representation, without an envelope.
    Selection is conservative rather than a claim of optimal compression.
    The optional positive byte threshold affects candidate discovery only;
    the default, net-saving rule and lossless encoding remain unchanged.
    """
    if type(min_fragment_bytes) is not int or min_fragment_bytes <= 0:
        raise ValueError('min_fragment_bytes must be a positive integer byte count.')
    check = check_cancel or (lambda: None)
    _validate(value, check)
    original = json_bytes(value)
    frozen = json.loads(original)
    counts: Counter[bytes] = Counter()

    def scan(node):
        check()
        raw = json_bytes(node)
        if len(raw) >= min_fragment_bytes:
            counts[raw] += 1
        # A literal is atomic: its nested user markers cannot become protocol.
        if _reserved(node):
            return
        for item in node.values() if type(node) is dict else node if type(node) is list else ():
            scan(item)

    scan(frozen)
    selected = {raw for raw, count in counts.items() if count > 1}
    while selected:
        definitions, identifiers, uses = {}, {}, Counter()

        def encode(node, allow_reference=True):
            check()
            raw = json_bytes(node)
            if allow_reference and raw in selected:
                identifier = identifiers.get(raw)
                if identifier is None:
                    identifier = 'j' + str(len(identifiers) + 1)
                    identifiers[raw] = identifier
                    definitions[identifier] = {'json_sha256': sha256(raw).hexdigest(),
                                               'value': encode(node, False)}
                uses[identifier] += 1
                return {REFERENCE: identifier}
            if _reserved(node):
                return {LITERAL: node}
            if type(node) is dict:
                return {key: encode(item) for key, item in node.items()}
            if type(node) is list:
                return [encode(item) for item in node]
            return node

        root = encode(frozen)
        rejected = set()
        for raw, identifier in identifiers.items():
            entry, count = definitions[identifier], uses[identifier]
            reference_bytes = len(json_bytes({REFERENCE: identifier}))
            definition_bytes = len(json_bytes({identifier: entry})) - 2 + 1
            inline_bytes = count * len(json_bytes(entry['value']))
            if inline_bytes - count * reference_bytes - definition_bytes < 256:
                rejected.add(raw)
        # Selected children hidden inside a shared/literal parent need no defs.
        selected.intersection_update(identifiers)
        if rejected:
            # Remove a larger redundant parent first, then recount its children.
            # Pruning both together would miss a useful shared string which
            # currently appears only once inside that parent's definition.
            selected.remove(max(rejected, key=lambda raw: (len(raw), raw)))
            continue
        envelope = {'encoding': ENCODING, 'reference_rule': _RULE,
                    'original_json': root, 'definitions': definitions,
                    'original_json_sha256': sha256(original).hexdigest(),
                    'original_json_utf8_bytes': len(original), 'is_summary': False}
        encoded = json_bytes(envelope)
        if len(original) - len(encoded) >= 256:
            check()
            if json_bytes(restore_original_json(envelope, encoding=ENCODING)) != original:
                raise ValueError('Shared encoding did not preserve exact original JSON values.')
            check()
            return SharedOriginalJson(envelope, ENCODING, len(original), len(encoded))
        break
    return SharedOriginalJson(frozen, 'raw-json', len(original), len(original))
