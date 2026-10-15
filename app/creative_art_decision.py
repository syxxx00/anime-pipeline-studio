"""Allocate current artistic decision content, never execute or certify it.

Complete automatic drafts remain in the caller's frozen host input and bound
sources. This view is not a generation request, and cannot resolve pixel bases.
Unknown history, differing Source readings and native overrides stay intact.
"""
from __future__ import annotations

from copy import deepcopy
import re

from .creative_continuity_context import _digest, continuation_model_material
from .creative_reference import _allocate_known_local_purpose, _source_purpose_material
from .creative_request_understanding import QUOTED_HYPOTHESES_VIEW, _project_request_reading_material
from .schemas import GenerateRequest


ART_DECISION_VIEW = 'current_art_decision_v1'
ART_DECISION_SYSTEM = """Make ONE artistic purpose decision for ONE current picture operation.
Use the complete original, current explicit user selections, attributed raw feedback,
actual full observations or explicit missing coverage, and the bounded visible delta.
Assistant-validation reports are attributed QA material, not user preferences or approval.
The host manages requested output counts: do not ask this operation to deliver all
remaining directions, two compositions or extra panels in a single picture.
Known automatic past author drafts and reader hypotheses marked as host-held are
not preservation requirements or evidence of actual pixels. Develop the current
artistic purpose from the original and current feedback; do not replay a past
camera, motif, pose or exclusion merely because a previous author proposed it.
An earlier local instruction was that attempt's scope, not a permanent ban on
changing composition during a new regenerate. Preserve value only when it serves
the current operation and has a reason. Current user native/exclusion overrides
remain explicit conditions, not automatically authored material to discard.
Retained unknown or differing historical interpretations do not certify meaning.
Respect actual available methods, effective parameters, budgets and observation
coverage. A crop delta is not a whole-artwork inspection or an artistic verdict.
No invented user approval, quality certification or automatic selection. Return
only the supplied decision schema; never use this purpose view as image inputs."""

_AUTHOR_KEYS = {'id', 'revision', 'original_sha256', 'scene_narrative', 'visual_logic',
    'positions', 'layout', 'spec', 'semantic_sha256', 'is_user_requirement',
    'semantic_correctness_verified', 'author', 'runtime_evidence', 'uncertainties', 'goals',
    'conflicts', 'identity_evidence', 'knowledge', 'recipe_snapshot', 'spatial_realization_check',
    'original_intent_status', 'selected_config', 'identity_binding', 'picture_event',
    'prose_picture_choice', 'translated_prose_picture_choice', 'prose_picture_receipt',
    'scene_narrative_origin', 'request_understanding', 'author_carrier_original_intent_status',
    'direction_assignment', 'canonical_picture_choice', 'translated_canonical_picture_choice',
    'picture_projection', 'saved_canonical_preparation_receipt', 'saved_prose_preparation_receipt'}
_NATIVE_KEYS = {'positive', 'negative', 'trace', 'compiled_trace', 'postprocess', 'trace_status',
    'negative_policy', 'compiled_negative_policy', 'proposal_id', 'proposal_revision',
    'style_recipe_applied', 'author_choice_session'}

_PREPARATION_RECEIPTS = {
    'saved_canonical_preparation_receipt': {'origin', 'action_request_id', 'source_binding_sha256',
        'author_call_index', 'language_call_index', 'new_model_calls', 'semantic_correctness_verified'},
    'saved_prose_preparation_receipt': {'origin', 'action_request_id', 'source_prose_binding_sha256',
        'author_call_index', 'language_call_index', 'new_author_calls', 'new_starting_choice_calls',
        'new_language_calls', 'semantic_correctness_verified'},
}


def _known_preparation_receipts(author):
    """Recognize only bookkeeping from the two existing explicit preparations.

    This does not validate archived model calls or grant execution permission.
    A receipt with unknown content is not treated as removable host metadata.
    """
    if not isinstance(author, dict):
        return False
    present = set(author) & _PREPARATION_RECEIPTS.keys()
    if len(present) > 1:
        return False  # Existing preparation authors one carrier, not both.
    for key in present:
        value = author[key]
        if not isinstance(value, dict) or set(value) != _PREPARATION_RECEIPTS[key]:
            return False
        identifier = value['action_request_id']
        binding = value.get('source_binding_sha256', value.get('source_prose_binding_sha256'))
        if (not isinstance(value['origin'], str) or value['origin'] not in {'user', 'assistant_validation'}
                or not isinstance(identifier, str) or re.fullmatch(r'[a-zA-Z0-9_-]{8,80}', identifier) is None
                or not isinstance(binding, str) or re.fullmatch(r'[0-9a-f]{64}', binding) is None
                or value['semantic_correctness_verified'] is not False
                or type(value['author_call_index']) is not int or value['author_call_index'] < 0):
            return False
        author_index, language_index = value['author_call_index'], value['language_call_index']
        if key == 'saved_canonical_preparation_receipt':
            if (not isinstance(author.get('canonical_picture_choice'), dict)
                    or type(value['new_model_calls']) is not int or value['new_model_calls'] != 0
                    or type(language_index) is not int or language_index != author_index + 1):
                return False
        else:
            if (not isinstance(author.get('prose_picture_choice'), dict)
                    or any(type(value[name]) is not int or value[name] != 0
                    for name in ('new_author_calls', 'new_starting_choice_calls'))
                    or type(value['new_language_calls']) is not int or value['new_language_calls'] not in {0, 1}
                    or (language_index is not None and
                        (type(language_index) is not int or language_index <= author_index))
                    or (value['new_language_calls'] == 1 and language_index is None)):
                return False
    return True


def _source_reading_matches(source, data):
    reading = source['source_job']['request'].get('author_proposal', {}).get('request_understanding')
    current = data.get('request_reading_material')
    if reading is None:
        return current is None
    if not isinstance(current, dict) or current.get('model_view_version') != QUOTED_HYPOTHESES_VIEW:
        return False
    try:
        expected = _project_request_reading_material(reading, version=QUOTED_HYPOTHESES_VIEW)
        intent = expected.pop('creation_intent')
    except (ValueError, TypeError, KeyError):
        return False
    return _digest({'creation_intent': intent, 'request_reading_material': expected}) == _digest(
        {'creation_intent': data.get('creation_intent'), 'request_reading_material': current})


def _automatic_standard_selector(source, data, *, include_preparation_receipts=False):
    request = source['source_job']['request']
    parsed = GenerateRequest.model_validate(request)
    author, native = request.get('author_proposal'), request.get('native_prompts')
    if (set(request) - set(GenerateRequest.model_fields) or not isinstance(author, dict)
            or set(author) - _AUTHOR_KEYS or not _known_preparation_receipts(author)
            or (set(author) & _PREPARATION_RECEIPTS.keys() and include_preparation_receipts is not True)
            or author.get('condition_status') == 'historical_after_native_override'
            or not isinstance(native, dict) or not native
            or any(not isinstance(channel, dict) or set(channel) - _NATIVE_KEYS for channel in native.values())
            or parsed.source_parameters is not None):
        return None
    # The existing selector recognizer checks complete snapshot/request identity,
    # standard model/route, inactive extra stages, native equality/trace and the
    # known prose/canonical transport. It performs no file or resource access.
    material = continuation_model_material(source, for_model_call=True, successor_selection=True,
        current_creation_intent=data.get('creation_intent'),
        current_request_reading_material=data.get('request_reading_material'))
    if (material.get('material_scope') != 'successor_selection_complete_declared_picture_and_parameters_separate_from_pixels'
            or not material.get('declared_parent_picture_unverified')
            or not _source_reading_matches(source, data)):
        return None
    return material


def _resources_match(author, selections, *, shared=False):
    for name, current_name in (('knowledge', 'knowledge'), ('identity_binding', 'identity_binding'),
                               ('recipe_snapshot', 'full_style_recipes')):
        if name not in author:
            continue
        if current_name not in selections:
            return False
        old = author[name]
        if shared and isinstance(old, dict) and set(old) == {'model_field', 'value_sha256'}:
            if old != {'model_field': 'current_user_selections.' + current_name,
                       'value_sha256': _digest(selections[current_name])}:
                return False
        elif old != selections[current_name]:
            return False
    return True


def art_decision_material(full_data, *, bound_sources=None, include_preparation_receipts=False):
    """Return a current purpose view plus a host-only receipt; never do I/O.

    New callers may explicitly recognize the two known preparation receipts.
    The default retains their old unknown-content treatment so frozen calls do
    not silently acquire a different view when this helper is upgraded.
    """
    model = deepcopy(full_data)
    receipt = {'version': ART_DECISION_VIEW, 'status': 'legacy_material_retained',
        'input_sha256': _digest(full_data), 'moved_to_host': [],
        'reason': 'unknown_or_different_source_or_override_retained',
        'scope': 'host_original_and_execution_sources_retained_not_meaning_or_quality_verification'}

    def legacy(reason):
        receipt.update(reason=reason, output_sha256=_digest(full_data))
        return deepcopy(full_data), receipt

    if (not isinstance(full_data, dict) or not isinstance(bound_sources, dict)
            or set(bound_sources) not in ({'selected'}, {'parent', 'child'})
            or not isinstance(full_data.get('original_text'), str)
            or not isinstance(full_data.get('current_user_selections'), dict)
            or 'art_decision_scope' in full_data):
        return legacy('missing_known_bound_sources_or_reserved_view')
    dual = set(bound_sources) == {'parent', 'child'}
    parent_role = 'parent' if dual else 'selected'
    try:
        parent = bound_sources[parent_role]
        if parent.get('original_text') != full_data['original_text']:
            return legacy('source_original_different')
        parent_selector = _automatic_standard_selector(parent, full_data,
            include_preparation_receipts=include_preparation_receipts)
        if parent_selector is None:
            return legacy('source_is_not_known_automatic_standard_or_source_reading_differs')
        if not dual:
            if _digest(full_data.get('selected_source')) != _digest(parent_selector):
                return legacy('unknown_or_changed_selected_source_view')
            targets = [(('selected_source',), model['selected_source'])]
        else:
            child = bound_sources['child']
            child_author = child['source_job']['request'].get('author_proposal') or {}
            if (not _known_preparation_receipts(child_author)
                    or (set(child_author) & _PREPARATION_RECEIPTS.keys() and include_preparation_receipts is not True)):
                return legacy('unknown_child_preparation_receipt_retained')
            if child.get('original_text') != full_data['original_text'] or not _source_reading_matches(child, full_data):
                return legacy('different_child_source_reading_retained')
            # Validate the child host binding/native before reusing the existing
            # exact modern-local allocation recognizer. No new lineage inferred.
            expected = {'reference_options': {role: {'source_conditions': _source_purpose_material(
                bound_sources[role], full_data.get('creation_intent'), full_data.get('request_reading_material'))}
                for role in ('parent', 'child')}}
            expected = _allocate_known_local_purpose({'sources': bound_sources,
                'comparison_material': full_data['comparison_material']}, expected,
                full_data['current_user_selections'])
            if ((full_data.get('purpose_allocation') or {}).get('status') != 'known_exact_local_lineage'
                    or expected['purpose_allocation']['status'] != 'known_exact_local_lineage'
                    or expected['shared_reference_author_context_unverified'] != full_data.get('shared_reference_author_context_unverified')
                    or any(expected['reference_options'][role]['source_conditions'] !=
                        full_data['reference_options'][role]['source_conditions'] for role in ('parent', 'child'))):
                return legacy('unknown_or_changed_local_lineage_view')
            targets = [(('reference_options', role, 'source_conditions'),
                model['reference_options'][role]['source_conditions']) for role in ('parent', 'child')]
    except (ValueError, TypeError, KeyError, AttributeError):
        return legacy('incomplete_or_unrecognized_source_binding')

    def move(container, path, keys):
        for key in keys:
            if key in container:
                value = container.pop(key)
                pointer = '/' + '/'.join(str(part).replace('~', '~0').replace('/', '~1') for part in (*path, key))
                receipt['moved_to_host'].append({'path': pointer, 'value_sha256': _digest(value),
                    'scope': 'known_old_automatic_art_or_reader_material_held_by_host_not_current_user_requirement'})

    move(model, (), ('current_plan', 'current_picture_event', 'source_author_choice_status'))
    assignment = parent['source_job']['request']['author_proposal'].get('direction_assignment')
    if assignment is not None and model.get('direction_assignment') == assignment:
        move(model, (), ('direction_assignment',))
    reading = model.get('request_reading_material')
    if isinstance(reading, dict):
        move(reading, ('request_reading_material',), ('reader_hypotheses',))
    for path, packet in targets:
        move(packet, path, ('declared_parent_picture_unverified', 'declared_parent_picture_scope',
            'inherited_scene_reference', 'declared_author_context_reference'))
        author = packet.get('author_context_unverified')
        if isinstance(author, dict):
            if _resources_match(author, full_data['current_user_selections']):
                move(packet, path, ('author_context_unverified',))
            else:
                # Different historical knowledge/recipes still have content.
                # Only the proven old automatic composition/Source is moved.
                move(author, (*path, 'author_context_unverified'), ('visual_logic', 'positions', 'layout',
                    'uncertainties', 'goals', 'conflicts', 'direction_assignment', 'request_understanding'))
        metadata = packet.get('selected_native_metadata_unverified')
        if isinstance(metadata, dict):
            move(metadata, (*path, 'selected_native_metadata_unverified'), ('negative_policy', 'compiled_negative_policy'))
    if dual:
        shared = model['shared_reference_author_context_unverified']
        if _resources_match(shared, full_data['current_user_selections'], shared=True):
            move(model, (), ('shared_reference_author_context_unverified',))
        else:
            move(shared, ('shared_reference_author_context_unverified',), ('visual_logic', 'positions', 'layout',
                'uncertainties', 'goals', 'conflicts', 'direction_assignment', 'request_understanding'))
    model['art_decision_scope'] = {'version': ART_DECISION_VIEW,
        'scope': 'current_artistic_purpose_from_original_current_choices_feedback_and_pixels_not_old_automatic_draft_replay',
        'operation_scope': 'one_picture_operation_host_manages_candidate_collection',
        'automatic_history_held_by_host': True, 'meaning_verified': False, 'quality_verified': False}
    receipt.update(status='known_art_decision_allocated', reason='known_automatic_sources_current_content_preserved',
        output_sha256=_digest(model))
    return model, receipt
