"""Purpose content allocation, not execution, source verification or summarizing.

Only a known local-lineage purpose view is eligible. Complete input remains the
caller's frozen host original; this pure helper reports every moved value and
only shares text when its complete value is exactly equal.
"""
from __future__ import annotations

from collections import Counter
from copy import deepcopy

from .creative_continuity_context import _digest
from .creative_reference import REFERENCE_CHOICE_CONTRACT, REFERENCE_PURPOSE_VIEW
from .creative_result_review import LOCAL_VISIBLE_DELTA_CONTRACT, local_visible_delta_input
from .source_conditions import KnowledgeFactRecord


CONTENT_VERSION = 'local_reference_content_v1'
_HOST_SCOPE = 'host_identity_or_frozen_receipt_not_artistic_judgement_content'
_ARTWORK_METADATA = ('origin', 'job_id', 'image_index', 'image_url', 'file_path',
    'created_at', 'instruction', 'experiment_id', 'experiment_item_id', 'mask_file', 'source_stage')
_RAW_FACT_FIELDS = {'id': 'record_id', 'value': 'value', 'status': 'declared_status',
    'source_url': 'record_source_url', 'evidence_type': 'evidence_type', 'asset_key': 'asset_key'}


def _pointer(path):
    return '/' + '/'.join(str(part).replace('~', '~0').replace('/', '~1') for part in path)


def _contains_text_ref(value):
    if isinstance(value, dict):
        return 'text_ref' in value or any(_contains_text_ref(child) for child in value.values())
    if isinstance(value, list):
        return any(_contains_text_ref(child) for child in value)
    return False


def compact_reference_content(data):
    """Return a model-only view and host-only allocation receipt; never do I/O."""
    receipt = {'version': CONTENT_VERSION, 'status': 'legacy_material_retained',
        'input_sha256': _digest(data), 'moved_to_host': [], 'text_references': [],
        'scope': 'complete_input_retained_by_caller_not_a_drawable_request_or_quality_approval'}
    model = deepcopy(data)
    options = data.get('reference_options') if isinstance(data, dict) else None
    allocation = data.get('purpose_allocation') if isinstance(data, dict) else None
    eligible = (isinstance(data, dict)
        and data.get('contract_version') == REFERENCE_CHOICE_CONTRACT
        and data.get('purpose_view_version') in (REFERENCE_PURPOSE_VIEW, CONTENT_VERSION)
        and isinstance(allocation, dict) and allocation.get('status') == 'known_exact_local_lineage'
        and isinstance(data.get('original_text'), str) and bool(data['original_text'].strip())
        and isinstance(options, dict)
        and all(isinstance(options.get(role), dict)
            and isinstance(options[role].get('source_conditions'), dict) for role in ('parent', 'child'))
        and 'text_pool' not in data and 'content_scope' not in data and not _contains_text_ref(data))
    if not eligible:
        receipt['output_sha256'] = _digest(model)
        return model, receipt

    def move(container, path, keys, scope=_HOST_SCOPE):
        if not isinstance(container, dict):
            return
        for key in keys:
            if key in container:
                value = container.pop(key)
                receipt['moved_to_host'].append({'path': _pointer((*path, key)),
                    'value_sha256': _digest(value), 'scope': scope})

    def model_reference(container, path):
        # Known explicit model links only. Unknown fields or unbound values are
        # retained, never interpreted as host receipts because of a key name.
        if (not isinstance(container, dict) or not {'model_field', 'value_sha256'} <= set(container)
                or set(container) - {'model_field', 'value_sha256', 'view_version'}
                or not isinstance(container['model_field'], str)):
            return
        target = data
        try:
            for key in container['model_field'].split('.'):
                target = target[key]
        except (KeyError, TypeError):
            return
        if container['value_sha256'] == _digest(target):
            move(container, path, ('value_sha256',), 'exact_model_field_link_hash_retained_in_host')

    move(model, (), ('original_sha256', 'reference_context_binding_sha256'))
    for role in ('parent', 'child'):
        prefix = ('reference_options', role)
        option = model['reference_options'][role]
        packet = option['source_conditions']
        path = (*prefix, 'source_conditions')
        move(packet, path, ('snapshot_sha256', 'host_references',
            'inactive_saved_parameter_material', 'historical_execution_material'))
        artwork = packet.get('artwork')
        if isinstance(artwork, dict) and isinstance(artwork.get('id'), str):
            move(artwork, (*path, 'artwork'), _ARTWORK_METADATA)
        image = packet.get('image')
        if (isinstance(image, dict) and all(type(image.get(key)) is int and image[key] > 0
                for key in ('width', 'height'))):
            move(image, (*path, 'image'), ('file_path', 'sha256', 'bytes', 'format'))
        job = packet.get('source_job')
        if isinstance(job, dict) and isinstance(job.get('id'), str) and isinstance(job.get('status'), str):
            move(job, (*path, 'source_job'), ('request_sha256',))
        feedback = packet.get('feedback')
        move(feedback, (*path, 'feedback'), ('original_words_signature',))
        for key in ('declared_author_context_reference', 'inherited_scene_reference', 'feedback'):
            model_reference(packet.get(key), (*path, key))
        for key in ('actual_generation_conditions', 'actual_local_operation'):
            operation = packet.get(key)
            move(operation, (*path, key), ('repair_source_sha256', 'repair_mask_sha256'))
            if isinstance(operation, dict):
                move(operation.get('automatic_local_edit'), (*path, key, 'automatic_local_edit'),
                    ('source_sha256', 'mask_sha256'))
        observation = option.get('full_observation')
        if isinstance(observation, dict) and observation.get('scope') in {'full_original_png', 'missing_full_observation'}:
            move(observation, (*prefix, 'full_observation'), ('source_sha256', 'source_snapshot_sha256'))
            observation.setdefault('role', role)

    material = model.get('comparison_material')
    if isinstance(material, dict) and material.get('contract_version') == LOCAL_VISIBLE_DELTA_CONTRACT:
        try:
            local_visible_delta_input({'image_order': 'first parent crop; second child crop',
                'comparison_material': material, 'image_bindings': material['image_bindings']})
        except (ValueError, KeyError, TypeError):
            pass  # Unknown/malformed scope remains complete rather than guessed.
        else:
            path = ('comparison_material',)
            move(material, path, ('mask_sha256',))
            for index, binding in enumerate(material['image_bindings']):
                move(binding, (*path, 'image_bindings', index), ('sha256', 'bytes',
                    'continuation_snapshot_sha256', 'full_image_sha256', 'full_image_bytes', 'mask_sha256'))
            boundary = material.get('pixel_boundary_verification')
            if isinstance(boundary, dict):
                move(boundary.get('details'), (*path, 'pixel_boundary_verification', 'details'),
                    ('source_sha256', 'mask_sha256'))

    for name in ('creation_intent', 'current_user_selections'):
        move(model.get(name), (name,), ('original_sha256',))
    reading = model.get('request_reading_material')
    move(reading, ('request_reading_material',), ('source_material_sha256',))
    author = model.get('shared_reference_author_context_unverified')
    move(author, ('shared_reference_author_context_unverified',), ('original_sha256', 'semantic_sha256'))
    if isinstance(author, dict):
        for name in ('knowledge', 'identity_binding', 'recipe_snapshot'):
            model_reference(author.get(name), ('shared_reference_author_context_unverified', name))
        understanding = author.get('request_understanding')
        if isinstance(understanding, dict):
            path = ('shared_reference_author_context_unverified', 'request_understanding')
            move(understanding, path, ('source_material_sha256',))
            for name in ('creation_intent_reference', 'request_reading_material_reference'):
                model_reference(understanding.get(name), (*path, name))

    for container_name in ('current_user_selections', 'shared_reference_author_context_unverified'):
        container = model.get(container_name)
        knowledge = container.get('knowledge') if isinstance(container, dict) else None
        if not isinstance(knowledge, dict):
            continue
        for name in ('visual_fact_records', 'clothing_fact_records'):
            records = knowledge.get(name)
            if not isinstance(records, list):
                continue
            for index, record in enumerate(records):
                try:
                    checked = KnowledgeFactRecord.model_validate(record).model_dump(mode='json')
                except (ValueError, TypeError):
                    continue
                raw = record.get('raw_record')
                if (checked != record or not isinstance(raw, dict) or set(raw) - set(_RAW_FACT_FIELDS)
                        or any(raw[key] != record[_RAW_FACT_FIELDS[key]] for key in raw)):
                    continue
                move(record, (container_name, 'knowledge', name, index),
                    ('raw_record', 'record_sha256', 'source_binding_sha256'),
                    'validated_equal_known_declaration_original_and_identity_hashes_retained_in_host')

    # Only exact full strings are pooled, including unknown content. The root
    # original stays a string; matching nested original words link directly to it.
    def strings(value):
        if isinstance(value, str):
            yield value
        elif isinstance(value, dict):
            for child in value.values():
                yield from strings(child)
        elif isinstance(value, list):
            for child in value:
                yield from strings(child)

    counts = Counter(strings(model))
    pool, refs = {}, {}
    def share(value, path=()):
        if isinstance(value, str):
            if path == ('original_text',):
                return value
            if value == model['original_text']:
                reference = 'original_text'
            elif len(value) >= 160 and counts[value] >= 2:
                if value not in refs:
                    key = 'text_' + str(len(pool) + 1)
                    refs[value] = 'text_pool.' + key
                    pool[key] = value
                reference = refs[value]
            else:
                return value
            receipt['text_references'].append({'path': _pointer(path), 'text_ref': reference,
                'value_sha256': _digest(value)})
            return {'text_ref': reference}
        if isinstance(value, dict):
            return {key: share(child, (*path, key)) for key, child in value.items()}
        if isinstance(value, list):
            return [share(child, (*path, index)) for index, child in enumerate(value)]
        return value

    model = share(model)
    if pool:
        model['text_pool'] = pool
    model['content_scope'] = {'version': CONTENT_VERSION,
        'scope': 'purpose_content_only_complete_execution_identity_receipts_retained_in_host_not_quality_approval',
        'text_ref_resolution': 'Resolve {text_ref: original_text} to the complete root original_text; resolve '
            '{text_ref: text_pool.text_N} to that exact complete text_pool value. References preserve the '
            'original field, role and attribution; no summary, truncation or new user requirement.'}
    receipt.update(status='known_content_allocated', output_sha256=_digest(model))
    return model, receipt
