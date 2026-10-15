"""Lossless sharing of one complete top-level generation request family.

Only the model-facing copy of ``context.model_parameters`` is encoded. The
complete ``context.request`` is the base; each model carries every differing
top-level value, not a parameter summary or an artistic role. No paths are
resolved and no data is executed. Full originals remain in planner evidence.
Exact restoration does not certify that a model understands the representation.
Domain/schema validation remains the controller's responsibility.
"""
from __future__ import annotations

from hashlib import sha256
import json
import math
from typing import Callable

from .shared_original_json import json_bytes


ENCODING = 'lossless-request-family/v1'
MODELS = frozenset({'newbie', 'illustrious', 'anima', 'z_image_turbo'})
MAX_JSON_DEPTH = 64
MAX_CONTEXT_BYTES = 8 * 1024 * 1024
MAX_REQUEST_KEYS = 128
_RULE = ('Not a summary: context.request is the complete base. For each model in '
         'requests, copy all base fields, then replace only the fields in overrides. '
         'Absent overrides retain their full original values, including null, false '
         'and empty strings. key_order, when present, restores that request order; '
         'otherwise use the base field order. This representation grants no new '
         'model, parameter, file-reading or execution authority.')


def _check_json(value, check, *, depth=0, active=None):
    check()
    if depth > MAX_JSON_DEPTH:
        raise ValueError('Request-family JSON exceeds the bounded nesting depth.')
    if type(value) in (dict, list):
        active = set() if active is None else active
        if id(value) in active:
            raise ValueError('Request-family JSON cannot contain cycles.')
        active.add(id(value))
        try:
            if type(value) is dict:
                if any(type(key) is not str for key in value):
                    raise ValueError('Request-family JSON keys must already be strings.')
                values = value.values()
            else:
                values = value
            for item in values:
                _check_json(item, check, depth=depth + 1, active=active)
        finally:
            active.remove(id(value))
    elif type(value) not in (str, int, float, bool, type(None)):
        raise ValueError('Request-family sharing accepts only original JSON types.')
    elif type(value) is float and not math.isfinite(value):
        raise ValueError('Request-family JSON numbers must be finite.')


def _freeze_context(context, check):
    if type(context) is not dict:
        raise ValueError('Request-family context must be an original JSON object.')
    _check_json(context, check)
    raw = json_bytes(context)
    if len(raw) > MAX_CONTEXT_BYTES:
        raise ValueError('Complete request-family context exceeds its byte bound; not truncated.')
    check()
    return json.loads(raw)


def _base(context):
    base = context.get('request')
    if type(base) is not dict or not 1 <= len(base) <= MAX_REQUEST_KEYS:
        raise ValueError('A complete nonempty bounded context.request is required.')
    return base


def _models(requests):
    if (type(requests) is not dict or not 1 <= len(requests) <= len(MODELS)
            or set(requests) - MODELS):
        raise ValueError('Request family requires one to four registered model names.')


def _digest(value):
    return sha256(json_bytes(value)).hexdigest()


def restore_request_family(encoded_context: dict, *,
                           check_cancel: Callable[[], None] | None = None) -> dict:
    """Restore only this explicitly selected protocol; never interpret user paths."""
    check = check_cancel or (lambda: None)
    restored = _freeze_context(encoded_context, check)
    base = _base(restored)
    family = restored.get('model_parameters')
    expected = {'encoding', 'base_request_field', 'base_request_sha256', 'requests',
                'original_model_parameters_sha256', 'reconstruction_rule', 'is_summary'}
    if (type(family) is not dict or set(family) != expected
            or family['encoding'] != ENCODING or family['base_request_field'] != 'request'
            or family['reconstruction_rule'] != _RULE or family['is_summary'] is not False
            or family['base_request_sha256'] != _digest(base)):
        raise ValueError('Unknown, changed or incomplete request-family encoding.')
    _models(family['requests'])
    requests = {}
    for model, entry in family['requests'].items():
        check()
        if (type(entry) is not dict or set(entry) not in ({'overrides'}, {'overrides', 'key_order'})
                or type(entry['overrides']) is not dict or set(entry['overrides']) - set(base)):
            raise ValueError('Request overrides must replace only existing complete base fields.')
        order = entry.get('key_order', list(base))
        if (type(order) is not list or any(type(key) is not str for key in order)
                or len(order) != len(base) or len(set(order)) != len(order) or set(order) != set(base)):
            raise ValueError('Request key order must preserve the complete base key set exactly.')
        # Reconstruct each request independently: no shared mutable base values.
        values = json.loads(json_bytes(base))
        values.update(json.loads(json_bytes(entry['overrides'])))
        requests[model] = {key: values[key] for key in order}
    if family['original_model_parameters_sha256'] != _digest(requests):
        raise ValueError('Request family does not restore the exact original types, values and order.')
    restored['model_parameters'] = requests
    check()
    return restored


def share_request_family(context: dict, *,
                         check_cancel: Callable[[], None] | None = None) -> dict:
    """Return a lossless model view, without modifying any original object.

    All family members must have the same complete top-level key set. A changed
    nested object/list/string is an entire explicit override, never a partial
    merge. Invalid or mismatched input raises instead of dropping keys.
    """
    check = check_cancel or (lambda: None)
    view = _freeze_context(context, check)
    base = _base(view)
    requests = view.get('model_parameters')
    _models(requests)
    entries = {}
    for model, request in requests.items():
        check()
        if type(request) is not dict or set(request) != set(base):
            raise ValueError('Every complete model request must have exactly the base key set.')
        entry = {'overrides': {key: value for key, value in request.items()
                              if json_bytes(value) != json_bytes(base[key])}}
        if list(request) != list(base):
            entry['key_order'] = list(request)
        entries[model] = entry
    view['model_parameters'] = {'encoding': ENCODING, 'base_request_field': 'request',
        'base_request_sha256': _digest(base), 'requests': entries,
        'original_model_parameters_sha256': _digest(requests),
        'reconstruction_rule': _RULE, 'is_summary': False}
    if json_bytes(restore_request_family(view, check_cancel=check)) != json_bytes(context):
        raise ValueError('Request-family representation did not preserve the complete original context.')
    check()
    return view
