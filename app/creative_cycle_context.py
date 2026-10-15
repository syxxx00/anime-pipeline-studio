"""Use completed work on this operation's path, not an invented learning history.

This module only projects saved receipts. It does not observe images, classify
user feedback, select a method, replay calls or certify any artwork. Full older
requests/comparisons remain in the session record; the latest relevant comparison
is passed intact and once to each consumer.
"""
from __future__ import annotations

from copy import deepcopy

from .creative_continuity_context import _digest
from .creative_result_review import LOCAL_VISIBLE_DELTA_CONTRACT
from .creative_reference import REFERENCE_CHOICE_CONTRACT, visible_delta_model_material


_EFFECTIVE_FIELDS = ('model_id', 'render_pipeline', 'steps', 'cfg', 'sampler',
                     'scheduler', 'shift', 'cfg_rescale', 'seed', 'width', 'height',
                     'style_preset', 'composition_style_preset', 'character_style_preset',
                     'lora_name', 'lora_strength', 'operation')


def cycle_learning_context(record: dict, result_turn: dict | None, *, record_reference: str) -> dict:
    """Return at most three verified-receipt path entries, not art verification.

    Explicit independent operations and old turns without an operation identity
    have no inherited path. A comparison is eligible only alongside completed
    continuation/candidate and frozen parent/child identities. No fields are
    inferred from an intended plan or the current model's prose.
    """
    operation_key = _digest(record.get('operation') or {})
    empty = {'scope': 'completed_operations_on_this_bound_path_not_quality_or_user_approval',
             'use': 'Use actual attempted conditions and the full latest comparative gains/losses/unknowns '
                    'to reconsider a next operation and picture. A returned parent does not erase its child attempt; '
                    'repeating a method is not forbidden when changed conditions have a reason.',
             'operation_key': operation_key, 'source_artwork_id': None,
             'attempted_operations': [], 'latest_comparison_ref': None,
             'result_comparison_unverified': None, 'older_completed_turn_count': 0,
             'full_record_reference': record_reference, 'material_status': 'no_bound_path_material'}
    cycle = record.get('result_cycle') or {}
    if (not isinstance(result_turn, dict)
            or cycle.get('operation_key') != operation_key
            or result_turn.get('operation_key') != operation_key):
        return empty
    empty['source_artwork_id'] = result_turn.get('source_artwork_id')
    candidates = {item.get('artwork_id'): item for item in record.get('candidates', [])
                  if isinstance(item, dict) and item.get('status') == 'completed'}
    eligible = []
    # Stop at the current turn: later/other paths cannot become earlier evidence.
    for index, turn in enumerate(cycle.get('turns', [])):
        if turn.get('id') == result_turn.get('id'):
            break
        if turn.get('operation_key') != operation_key or turn.get('phase') != 'reviewed':
            continue
        continuation = turn.get('continuation') or {}
        parent = continuation.get('source_context') or {}
        child = turn.get('child_source') or {}
        comparison = turn.get('comparison')
        candidate = candidates.get(turn.get('child_artwork_id'))
        if (continuation.get('completed') is not True or not isinstance(comparison, dict)
                or comparison.get('provisional_base') not in {'parent', 'child', 'neither'}
                or candidate is None or not candidate.get('file_path')
                or (parent.get('artwork') or {}).get('id') != turn.get('source_artwork_id')
                or (child.get('artwork') or {}).get('id') != turn.get('child_artwork_id')
                or (child.get('source_job') or {}).get('status') != 'completed'
                or (child.get('source_job') or {}).get('id') != candidate.get('job_id')):
            continue
        saved_request = (child.get('source_job') or {}).get('request')
        saved_sha = (child.get('source_job') or {}).get('request_sha256')
        if not isinstance(saved_request, dict) or _digest(saved_request) != saved_sha:
            continue
        comparison_ref = {'record': record_reference, 'pointer': f'/result_cycle/turns/{index}/comparison',
                          'sha256': _digest(comparison)}
        entry = {'turn_id': turn['id'], 'source_artwork_id': turn['source_artwork_id'],
                 'child_artwork_id': turn['child_artwork_id'], 'job_id': candidate['job_id'],
                 'method': candidate.get('method'),
                 'choice': deepcopy(continuation.get('choice')),
                 'effective_execution_parameters': {key: deepcopy(saved_request[key])
                     for key in _EFFECTIVE_FIELDS if key in saved_request},
                 'effective_image_handoff': deepcopy(saved_request.get('image_handoff')),
                 'full_saved_request_ref': {'record': record_reference,
                     'pointer': f'/result_cycle/turns/{index}/child_source/source_job/request',
                     'sha256': saved_sha},
                 'comparison_outcome_unverified': comparison['provisional_base'],
                 'comparison_ref': comparison_ref}
        if (turn.get('comparison_material') or {}).get('contract_version') == LOCAL_VISIBLE_DELTA_CONTRACT:
            entry['comparison_outcome_unverified'] = 'visible_delta_not_reference_selection'
            decision = turn.get('reference_decision')
            if decision is not None:
                if (decision.get('contract_version') != REFERENCE_CHOICE_CONTRACT
                        or decision.get('reference_base') not in {'parent', 'child', 'neither'}):
                    continue
                entry['reference_decision'] = deepcopy(decision)
            comparison = visible_delta_model_material(comparison)
        if turn.get('comparison_material') is not None:
            # Preserve the actual reading extent and precise source/crop
            # bindings; a local observation is not a whole-image judgment.
            entry['comparison_material'] = deepcopy(turn['comparison_material'])
            entry['comparison_material_ref'] = {'record': record_reference,
                'pointer': f'/result_cycle/turns/{index}/comparison_material',
                'sha256': _digest(turn['comparison_material'])}
        eligible.append((entry, comparison))
    if not eligible:
        return empty
    latest_entry, latest_comparison = eligible[-1]
    # A real current cursor must agree with the actual tentative comparison.
    # "neither" remains the existing host's child cursor, not an adopted image.
    neutral = (latest_entry.get('comparison_material') or {}).get('contract_version') == LOCAL_VISIBLE_DELTA_CONTRACT
    if neutral:
        decision = latest_entry.get('reference_decision')
        expected_cursor = ((decision.get('selected_artwork_id') or latest_entry['child_artwork_id'])
            if decision is not None else latest_entry['child_artwork_id'])
        if decision is None and result_turn.get('reference_from_turn_id') != latest_entry['turn_id']:
            empty['material_status'] = 'neutral_reading_has_no_bound_purpose_consumer'
            return empty
    else:
        expected_cursor = (latest_entry['source_artwork_id']
                           if latest_comparison['provisional_base'] == 'parent'
                           else latest_entry['child_artwork_id'])
    if result_turn.get('source_artwork_id') != expected_cursor:
        empty['material_status'] = 'saved_comparison_not_bound_to_current_cursor'
        return empty
    attempts = []
    for entry, comparison in eligible[-3:]:
        entry = deepcopy(entry)
        if entry['turn_id'] != latest_entry['turn_id']:
            # Earlier outcomes carry their complete short reasoning, while the
            # latest full comparison is not repeated inside its attempt entry.
            entry['result_reason_unverified'] = deepcopy(comparison.get('reason'))
        attempts.append(entry)
    return {**empty, 'attempted_operations': attempts,
            'latest_comparison_ref': deepcopy(latest_entry['comparison_ref']),
            'result_comparison_unverified': deepcopy(latest_comparison),
            'older_completed_turn_count': max(0, len(eligible) - 3),
            'material_status': 'bound_completed_receipts_not_independent_pixel_verification'}


def cycle_history_view(context: dict) -> dict:
    """Comparison travels in its existing field, not twice in one model input."""
    if not context.get('attempted_operations'):
        return {'scope': context['scope'], 'attempted_operations': [],
                'material_status': context['material_status']}
    return {key: deepcopy(value) for key, value in context.items()
            if key != 'result_comparison_unverified'}


def completed_candidate_context(record: dict) -> dict:
    """Show the existing alternatives, not only the current provisional base.

    Framing is copied from each image's saved author declaration, never from
    the current (possibly overwritten) plan and never treated as visible fact.
    This is not a diversity score, desired-count parser, or quality approval.
    Full requests remain in their dispatch receipts; six recent candidates are
    enough to expose the normal single-operation image budget without sending
    every prior prompt back to the selector.
    """
    completed = [item for item in record.get('candidates', [])
                 if item.get('status') == 'completed' and item.get('artwork_id')
                 and item.get('file_path')]
    dispatches = {item.get('id'): (index, item)
                  for index, item in enumerate(record.get('generation_dispatches', []))}
    operation = record.get('operation') or {}
    current_action = (operation.get('action') or {}).get('request_id')
    current_count = 0
    recent = []
    for candidate in completed:
        index, dispatch = dispatches.get(candidate.get('dispatch_id'), (None, {}))
        request = dispatch.get('request') or {}
        proposal = request.get('author_proposal') or {}
        belongs = bool(dispatch and dispatch.get('purpose') == operation.get('purpose')
                       and dispatch.get('action_request_id') == current_action)
        current_count += int(belongs)
        declared = None
        if proposal.get('condition_status') != 'historical_after_native_override':
            picture = (proposal.get('translated_canonical_picture_choice')
                       or proposal.get('canonical_picture_choice')
                       or proposal.get('picture_event') or {})
            framing = (picture.get('visual_spec') or {}).get('framing')
            if isinstance(framing, list):
                declared = deepcopy(framing)
        recent.append({'artwork_id': candidate['artwork_id'], 'job_id': candidate.get('job_id'),
            'method': candidate.get('method'), 'parent_artwork_id': candidate.get('parent_artwork_id'),
            'belongs_to_current_operation': belongs, 'model_id': request.get('model_id'),
            'declared_framing_unverified': declared,
            'saved_request_pointer': (f'/generation_dispatches/{index}/request' if index is not None else None)})
    return {'scope': 'completed_candidates_in_this_task_not_verified_compositions_or_user_approval',
            'completed_count': len(completed), 'current_operation_completed_count': current_count,
            'recent_candidates': recent[-6:], 'older_completed_count': max(0, len(recent) - 6),
            'user_selected_artwork_id': record.get('selected_artwork_id'),
            'use': 'The complete original may request exploration or distinct alternatives. '
                   'Consider what candidates actually exist before keep/stop or another correction. '
                   'Different seeds, model names or declared framing do not prove visible diversity; '
                   'the saved declarations are intentions, not additional pixel observations.'}
