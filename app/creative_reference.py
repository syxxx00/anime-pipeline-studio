"""Pure reference-choice material; live validation and execution remain callers' work."""
from __future__ import annotations

from copy import deepcopy
from typing import Literal

from pydantic import BaseModel, Field, model_validator

from .creative_continuity_context import _digest, continuation_model_material
from .creative_request_understanding import (QUOTED_HYPOTHESES_VIEW, SOURCE_READING_SCOPE,
    _project_request_reading_material)
from .creative_result_review import (CurrentPixelObservation, ResultComparison,
    FULL_VISIBLE_DELTA_CONTRACT, LOCAL_VISIBLE_DELTA_CONTRACT, _checked_snapshot,
    full_visible_delta_input, local_visible_delta_input)
from .schemas import GenerateRequest


REFERENCE_CHOICE_CONTRACT = 'local_reference_choice_v1'
FULL_REFERENCE_CHOICE_CONTRACT = 'full_reference_choice_v1'
REFERENCE_PURPOSE_VIEW = 'local_reference_purpose_v2'
REFERENCE_CONTENT_VIEW = 'local_reference_content_v1'


class ReferenceNextChoice(BaseModel):
    # Keep required fields before optional denoise for the frozen native grammar.
    # No import from CreativeSessions: the old NextChoice/schema remains intact.
    model_config = {'extra': 'forbid'}
    reference_base: Literal['parent', 'child', 'neither']
    operation: Literal['keep', 'regenerate', 'handoff', 'sketch', 'local_edit', 'stop']
    target: Literal['newbie', 'illustrious', 'anima', 'z_image_turbo']
    objective: str = Field(min_length=3, max_length=1200)
    reason: str = Field(min_length=3, max_length=1500)
    preserve: str = Field(max_length=1200)
    denoise: float = Field(default=.25, gt=0, le=1)

    @model_validator(mode='after')
    def reference_is_usable_for_operation(self):
        if self.reference_base == 'neither' and self.operation not in {'regenerate', 'stop'}:
            raise ValueError('Neither permits only fresh regenerate or stop, not borrowing either pixel base.')
        return self


REFERENCE_CHOICE_SYSTEM = """Choose a reference and the next operation in one purpose decision.
parent and child refer to the two exact bound works, not approval or user selection.
Use the complete original, current attributed corrections, current requirements,
full recipes and actual method capabilities. The previous_local_operation records
what that earlier edit attempted; it is not new user feedback or the current goal.
visible_delta_unverified is a limited crop reading, not a quality comparison.
features_added and features_removed mean reported visible changes, not gains or
losses in artistic value. The reading can be wrong. Do not infer desired forms
or task success merely because they were requested. A missing full observation
is missing, not a passed inspection; never substitute another work's observation.
Consider what serves the current purpose, what value may be lost and uncertainty.
Return reference_base parent, child or neither together with one available next
operation and a reason. Neither allows only fresh regenerate from complete current
requirements or stop; no image handoff/local edit from either rejected base.
Use only capabilities and parameters bound to the chosen reference. No automatic
quality certification, scores or user approval. Return the supplied JSON only."""


FULL_REFERENCE_CHOICE_SYSTEM = """Choose a reference and the next complete picture operation together.
parent and child are the two exact independently bound full artworks, not an
inherited local scene or user-approved versions. Use the complete original,
current selections, attributed corrections and actual method capabilities.
previous_operation says what the previous attempt tried, not new user feedback
or proof it happened. visible_delta_unverified is a fallible full-image pixel
reading. Features added/removed are visible changes, not artistic gains/losses.
Full observations are separately bound; missing coverage remains missing.
Relate these materials to the current purpose: which visible relations can
support it, which need different choices, and what useful local value may be
lost. Neither textual plans nor richer texture certify intention, identity,
attention, emotion or quality. Consider a materially different feasible way
when the current approach does not support the needed relation.
Choose parent, child or neither with ONE next operation and a reason. Neither
permits only fresh regenerate or stop, not pixel borrowing from either work.
Use each reference's own parameters and conditions; do not assume the child
inherits the parent's composition or declare an unmet relation preserved.
No score, automatic approval or user selection. Return the supplied JSON."""


def reference_choice_system(view_version=None, *, contract_version=None):
    if contract_version not in {None, REFERENCE_CHOICE_CONTRACT, FULL_REFERENCE_CHOICE_CONTRACT}:
        raise ValueError('Unknown reference contract; no inferred local/full purpose scope.')
    system = (FULL_REFERENCE_CHOICE_SYSTEM if contract_version == FULL_REFERENCE_CHOICE_CONTRACT
              else REFERENCE_CHOICE_SYSTEM)
    if view_version != REFERENCE_CONTENT_VIEW:
        return system
    return system + (
        '\nOnly when content_scope.version is local_reference_content_v1, text_ref '
        'points to the exact original_text or text_pool entry in this same input, '
        'never a summary or inaccessible archive. Expand those words when judging; '
        'role IDs retain attribution. Without that content_scope do not reinterpret '
        'an unknown text_ref field. Host SHA/path receipts are not quality evidence '
        'and remain with the executor.')


def visible_delta_model_material(comparison: dict) -> dict:
    """Rename neutral changes, never treating the raw base as a choice."""
    ResultComparison.model_validate(comparison)
    return {'current_region_observation': deepcopy(comparison['current_observation']),
        'features_added': deepcopy(comparison['gains']), 'features_removed': deepcopy(comparison['losses']),
        'unresolved': deepcopy(comparison['unresolved']),
        'reader_explanation_unverified': comparison['reason'], 'verified': False,
        'scope': 'reported_visible_crop_delta_not_artistic_gain_or_reference_selection'}


def full_visible_delta_model_material(comparison: dict) -> dict:
    """A full neutral response is material, never a preferred reference."""
    ResultComparison.model_validate(comparison)
    return {'current_full_observation': deepcopy(comparison['current_observation']),
        'features_added': deepcopy(comparison['gains']), 'features_removed': deepcopy(comparison['losses']),
        'unresolved': deepcopy(comparison['unresolved']),
        'reader_explanation_unverified': comparison['reason'], 'verified': False,
        'scope': 'reported_visible_full_delta_not_artistic_gain_or_reference_selection'}


def bind_full_reference_context(turn, parent_source, child_source, *, original_text, observations=None):
    """Bind independent full sources to an explicitly new neutral pair.

    Does not convert a legacy purpose comparison or crop into neutral evidence,
    infer observations from declarations, or decide which version to execute.
    Live source/byte validation and dispatch remain the caller's responsibility.
    """
    try:
        material = turn['comparison_material']
        if (turn['phase'] != 'reviewed'
                or material.get('contract_version') != FULL_VISIBLE_DELTA_CONTRACT):
            raise ValueError('Full references need an explicitly reviewed full-neutral pair.')
        neutral = full_visible_delta_input({'image_order': 'first parent; second child',
            'comparison_material': material, 'image_bindings': material['image_bindings']})
        if neutral['comparison_material'] != material:
            raise ValueError('Unknown full-pair receipt is not upgraded.')
        full_visible_delta_model_material(turn['comparison'])
        old_sources = {'parent': turn['continuation']['source_context'], 'child': turn['child_source']}
        supplied = {} if observations is None else observations
        if not isinstance(supplied, dict) or set(supplied) - {'parent', 'child'}:
            raise ValueError('Full observations must be attributed to parent/child.')
        sources, visual_sources, scoped_observations = {}, {}, {}
        for role, current, binding in zip(('parent', 'child'), (parent_source, child_source), material['image_bindings']):
            old, _ = _checked_snapshot(old_sources[role], original_text)
            current, _ = _checked_snapshot(current, original_text)
            old_image, image = old['image'], current['image']
            if (old['artwork']['id'] != turn['source_artwork_id' if role == 'parent' else 'child_artwork_id']
                    or current['artwork']['id'] != old['artwork']['id']
                    or current['source_job']['id'] != old['source_job']['id']
                    or current['source_job']['request_sha256'] != old['source_job']['request_sha256']
                    or any(image[key] != old_image[key] for key in ('sha256', 'width', 'height', 'bytes', 'format'))
                    or binding['artwork_id'] != old['artwork']['id']
                    or binding['continuation_snapshot_sha256'] != old['snapshot_sha256']
                    or any(binding[key] != old_image[key] for key in ('sha256', 'width', 'height', 'bytes'))):
                raise ValueError('Full neutral pixels/request do not bind this actual reference.')
            sources[role] = current
            visual_sources[role] = {'artwork_id': old['artwork']['id'], 'source_job_id': old['source_job']['id'],
                'source_request_sha256': old['source_job']['request_sha256'], 'source_sha256': old_image['sha256'],
                'source_snapshot_sha256': old['snapshot_sha256']}
            packet = supplied.get(role)
            if packet is None:
                scoped_observations[role] = {'scope': 'missing_full_observation', 'observation': None,
                    'artwork_id': current['artwork']['id'], 'source_sha256': image['sha256'],
                    'source_snapshot_sha256': current['snapshot_sha256']}
                continue
            if (not isinstance(packet, dict) or packet.get('scope') != 'full_original_png'
                    or packet.get('artwork_id') != current['artwork']['id']
                    or packet.get('source_sha256') != image['sha256']
                    or packet.get('source_snapshot_sha256') != current['snapshot_sha256']
                    or packet.get('input_contract_version') not in
                        {'full_pixel_observation_v1', FULL_VISIBLE_DELTA_CONTRACT}):
                raise ValueError('A full purpose response, crop or different image is not neutral observation.')
            CurrentPixelObservation.model_validate(packet['observation'])
            if packet['input_contract_version'] == FULL_VISIBLE_DELTA_CONTRACT and (
                    role != 'child' or packet['observation'] != turn['comparison']['current_observation']):
                raise ValueError('Full-pair current observation belongs only to its exact second image.')
            scoped_observations[role] = deepcopy(packet)
        context, child_request = turn['continuation'], sources['child']['source_job']['request']
        choice = context['choice']
        if (choice.get('operation') not in {'regenerate', 'handoff', 'sketch'}
                or not isinstance(choice.get('objective'), str) or not choice['objective'].strip()
                or not isinstance(choice.get('preserve'), str)
                or child_request.get('operation') == 'region_repair'):
            raise ValueError('Full references require the actual non-local attempt, not a guessed local scope.')
        bound = {'contract_version': FULL_REFERENCE_CHOICE_CONTRACT, 'source_turn_id': turn['id'],
            'operation_key': turn['operation_key'], 'original_text': original_text,
            'original_sha256': sources['parent']['original_sha256'], 'sources': sources,
            'neutral_visual_sources': visual_sources, 'observations': scoped_observations,
            'comparison_material': deepcopy(material), 'visible_delta_raw': deepcopy(turn['comparison']),
            'previous_operation': {**{key: deepcopy(choice[key]) for key in
                ('operation', 'target', 'objective', 'reason', 'preserve', 'denoise') if key in choice},
                'child_job_operation': child_request.get('operation'),
                'origin': context.get('origin', 'saved_picture_operation_not_user_feedback'),
                'scope': 'actual_previous_full_picture_attempt_not_latest_feedback_or_current_goal'},
            'live_source_revalidation_required': True, 'reference_choice_is_not_user_selection': True}
        bound['binding_sha256'] = _digest(bound)
        return bound
    except (KeyError, TypeError, AttributeError) as exc:
        raise ValueError('Full reference context is incomplete; no source, observation or purpose invented.') from exc


def bind_reference_context(turn, parent_source, child_source, *, original_text, observations=None):
    """Bind current feedback snapshots to a reviewed neutral pair, without I/O."""
    try:
        material = turn['comparison_material']
        if (turn['phase'] != 'reviewed'
                or material.get('contract_version') != LOCAL_VISIBLE_DELTA_CONTRACT):
            raise ValueError('Reference choice requires a reviewed known neutral turn, not a legacy comparison.')
        neutral = local_visible_delta_input({'image_order': 'first parent crop; second child crop',
            'comparison_material': material, 'image_bindings': material['image_bindings']})
        if _digest(neutral['comparison_material']) != _digest(material):
            raise ValueError('Unknown neutral receipt fields; no guessed projection or upgrade.')
        visible_delta_model_material(turn['comparison'])
        sources, visual_sources, scoped_observations = {}, {}, {}
        old_sources = {'parent': turn['continuation']['source_context'], 'child': turn['child_source']}
        supplied = observations or {}
        if not isinstance(supplied, dict) or set(supplied) - {'parent', 'child'}:
            raise ValueError('Full observations must be attributed to parent/child.')
        for role, current, binding in zip(('parent', 'child'), (parent_source, child_source), material['image_bindings']):
            old, _ = _checked_snapshot(old_sources[role], original_text)
            current, _ = _checked_snapshot(current, original_text)
            old_image, image = old['image'], current['image']
            if (old['artwork']['id'] != turn['source_artwork_id' if role == 'parent' else 'child_artwork_id']
                    or current['artwork']['id'] != old['artwork']['id']
                    or current['source_job']['id'] != old['source_job']['id']
                    or current['source_job']['request_sha256'] != old['source_job']['request_sha256']
                    or any(image[key] != old_image[key] for key in ('sha256', 'width', 'height', 'bytes', 'format'))
                    or binding['artwork_id'] != old['artwork']['id']
                    or binding['continuation_snapshot_sha256'] != old['snapshot_sha256']
                    or any(binding['full_image_' + key] != old_image[key] for key in ('sha256', 'width', 'height', 'bytes'))):
                raise ValueError('Neutral pixels/request do not bind the current reference; no replacement or inference.')
            sources[role] = current
            visual_sources[role] = {'artwork_id': old['artwork']['id'], 'source_job_id': old['source_job']['id'],
                'source_request_sha256': old['source_job']['request_sha256'], 'source_sha256': old_image['sha256'],
                'source_snapshot_sha256': old['snapshot_sha256']}
            packet = supplied.get(role)
            if packet is None:
                scoped_observations[role] = {'scope': 'missing_full_observation', 'observation': None,
                    'artwork_id': current['artwork']['id'], 'source_sha256': image['sha256'],
                    'source_snapshot_sha256': current['snapshot_sha256']}
            else:
                if (not isinstance(packet, dict) or packet.get('scope') != 'full_original_png'
                        or packet.get('artwork_id') != current['artwork']['id']
                        or packet.get('source_sha256') != image['sha256']
                        or packet.get('source_snapshot_sha256') != current['snapshot_sha256']):
                    raise ValueError('Observation is not an explicit full reading of this reference snapshot.')
                CurrentPixelObservation.model_validate(packet['observation'])
                scoped_observations[role] = deepcopy(packet)
        context = turn['continuation']
        choice = context['choice']
        if not isinstance(choice['objective'], str) or not isinstance(choice['preserve'], str):
            raise ValueError('The actual previous local objective/preserve is missing; none invented.')
        child_request = sources['child']['source_job']['request']
        if (child_request.get('operation') != 'region_repair'
                or child_request.get('parent_artwork_id') != sources['parent']['artwork']['id']
                or child_request.get('repair_source_sha256') != sources['parent']['image']['sha256']
                or child_request.get('repair_mask_sha256') != material['mask_sha256']
                or child_request.get('repair_instruction') != choice['objective'].strip()):
            raise ValueError('The neutral pair does not bind that actual local operation.')
        bound = {'contract_version': REFERENCE_CHOICE_CONTRACT, 'source_turn_id': turn['id'],
            'operation_key': turn['operation_key'], 'original_text': original_text,
            'original_sha256': sources['parent']['original_sha256'], 'sources': sources,
            'neutral_visual_sources': visual_sources, 'observations': scoped_observations,
            'comparison_material': deepcopy(material), 'visible_delta_raw': deepcopy(turn['comparison']),
            'previous_local_operation': {'objective': choice['objective'], 'preserve': choice['preserve'],
                'origin': context.get('origin', 'saved_local_operation_not_user_feedback'),
                'scope': 'actual_previous_local_attempt_not_latest_feedback_or_current_goal'},
            'live_source_revalidation_required': True, 'reference_choice_is_not_user_selection': True}
        bound['binding_sha256'] = _digest(bound)
        return bound
    except (KeyError, TypeError, AttributeError) as exc:
        raise ValueError('Reference context is incomplete; no source, observation or purpose invented.') from exc


def _checked_reference_context(context):
    saved = deepcopy(context)
    try:
        signature = saved.pop('binding_sha256')
        if (saved['contract_version'] not in {REFERENCE_CHOICE_CONTRACT, FULL_REFERENCE_CHOICE_CONTRACT}
                or _digest(saved) != signature):
            raise ValueError('Reference context binding changed; no replacement or repair.')
        for source in saved['sources'].values():
            _checked_snapshot(source, saved['original_text'])
    except (KeyError, TypeError, AttributeError) as exc:
        raise ValueError('Reference context lacks its complete binding.') from exc
    return saved


def _source_purpose_material(source, current_intent, current_reading):
    material = continuation_model_material(source, for_model_call=True, successor_selection=True,
        current_creation_intent=current_intent, current_request_reading_material=current_reading)
    request = source['source_job']['request']
    reading = (request.get('author_proposal') or {}).get('request_understanding')
    known_keys = {'creation_intent', 'candidate_request', 'unattributed_model_reading',
                  'scope', 'authority', 'call_index', 'understanding_verified'}
    if (not isinstance(reading, dict) or set(reading) != known_keys
            or reading.get('scope') != SOURCE_READING_SCOPE
            or _digest(reading.get('creation_intent')) != _digest(request.get('creation_intent'))):
        material['source_model_view_status'] = 'unknown_or_different_source_material_retained'
        return material
    try:
        projected = _project_request_reading_material(reading, version=QUOTED_HYPOTHESES_VIEW)
    except (ValueError, TypeError):
        material['source_model_view_status'] = 'unknown_source_material_retained'
        return material
    scope = deepcopy(projected)
    intent = scope.pop('creation_intent')
    actual, author = material['actual_generation_conditions'], material['author_context_unverified']
    if (current_intent is not None and current_reading is not None
            and _digest({'creation_intent': intent, 'request_reading_material': scope})
            == _digest({'creation_intent': current_intent, 'request_reading_material': current_reading})):
        actual.pop('creation_intent', None)
        author['request_understanding'] = {'creation_intent_reference': {'model_field': 'creation_intent',
                'value_sha256': _digest(current_intent)},
            'request_reading_material_reference': {'model_field': 'request_reading_material',
                'value_sha256': _digest(current_reading), 'view_version': QUOTED_HYPOTHESES_VIEW},
            'source_material_sha256': projected['source_material_sha256'],
            'scope': 'same_exact_known_source_projection_sent_once_not_meaning_verification'}
    else:
        actual['creation_intent'] = deepcopy(intent)
        author['request_understanding'] = deepcopy(projected)
    material['source_model_view_status'] = 'known_source_quoted_hypotheses_purpose_view'
    return material


def _allocate_known_local_purpose(saved, data, current_user_selections):
    """Allocate only an exact known inherited local scene, not arbitrary history.

    This consumer chooses a reference/method, not a drawable request. Every
    original request and native byte remains in the bound host for execution.
    Unknown fields and changed/overridden author conditions keep the old view.
    """
    parent, child = (saved['sources'][role]['source_job']['request'] for role in ('parent', 'child'))
    parent_material = data['reference_options']['parent']['source_conditions']
    child_material = data['reference_options']['child']['source_conditions']
    author = parent.get('author_proposal')
    child_author = deepcopy(child.get('author_proposal'))
    automatic = (child_author or {}).get('automatic_local_edit')
    if isinstance(automatic, dict) and (
            set(automatic) == {'selection_id', 'source_sha256', 'mask_sha256', 'instruction',
                              'origin', 'geometry_accuracy_verified', 'quality_verified'}
            and isinstance(automatic['selection_id'], str) and bool(automatic['selection_id'])
            and automatic['source_sha256'] == saved['sources']['parent']['image']['sha256']
            and automatic['source_sha256'] == child.get('repair_source_sha256')
            and automatic['mask_sha256'] == saved['comparison_material']['mask_sha256']
            and automatic['mask_sha256'] == child.get('repair_mask_sha256')
            and automatic['instruction'] == child.get('repair_instruction')
            and isinstance(automatic['origin'], str) and bool(automatic['origin'])
            and automatic['geometry_accuracy_verified'] is False and automatic['quality_verified'] is False
            and child_author.get('condition_status') == 'scene_context_with_local_override_unverified'):
        # This known host-owned receipt describes this execution, not a new
        # author scene. Preserve the actual receipt below and compare only the
        # exact inherited art conditions; no unknown delta is normalized away.
        child_author.pop('automatic_local_edit')
        child_author.pop('condition_status')
    known_author_fields = {'id', 'revision', 'original_sha256', 'scene_narrative', 'visual_logic',
        'positions', 'layout', 'spec', 'semantic_sha256', 'is_user_requirement',
        'semantic_correctness_verified', 'author', 'runtime_evidence', 'uncertainties', 'goals',
        'conflicts', 'identity_evidence', 'knowledge', 'recipe_snapshot', 'spatial_realization_check',
        'original_intent_status', 'selected_config', 'identity_binding', 'picture_event',
        'prose_picture_choice', 'translated_prose_picture_choice', 'prose_picture_receipt',
        'scene_narrative_origin', 'request_understanding', 'author_carrier_original_intent_status',
        'direction_assignment', 'canonical_picture_choice', 'translated_canonical_picture_choice',
        'picture_projection'}
    native_fields = {'positive', 'negative', 'trace', 'compiled_trace', 'postprocess', 'trace_status',
        'negative_policy', 'compiled_negative_policy', 'proposal_id', 'proposal_revision',
        'style_recipe_applied', 'author_choice_session'}
    known = (isinstance(author, dict) and not set(author).difference(known_author_fields)
        and author == child_author
        and not set(parent).difference(GenerateRequest.model_fields)
        and not set(child).difference(GenerateRequest.model_fields)
        and bool(parent_material.get('declared_parent_picture_unverified'))
        and parent_material.get('author_context_status') == 'source_author_context_unverified_not_actual_native_conditions'
        and child_material.get('author_context_status') == 'source_author_context_unverified_not_actual_native_conditions'
        and child.get('operation') == 'region_repair' and child.get('model_id') == 'newbie'
        and child.get('render_pipeline') == 'newbie'
        and child.get('repair_context_policy') == 'scene_context_with_local_override'
        and child.get('repair_context_prompt') == parent.get('prompt')
        and not any(child.get(key) for key in ('image_handoff', 'regional_conditioning',
            'highres_enabled', 'face_detail_enabled', 'hand_repair_enabled', 'text_clean_enabled'))
        and (child.get('native_prompts') or {}).get('newbie', {}).get('trace_status')
            == 'local_edit_context_and_override_unverified')
    for request in (parent, child):
        channels = request.get('native_prompts') or {}
        known = known and isinstance(channels, dict) and all(isinstance(item, dict)
            and not set(item).difference(native_fields) for item in channels.values())
    # Full current resources and Source need exact same-purpose identity before
    # an inherited copy can point at them. Different historical facts survive.
    for key in ('knowledge', 'identity_binding'):
        known = known and key in (author or {}) and key in current_user_selections \
            and _digest(author[key]) == _digest(current_user_selections[key])
    for material in (parent_material, child_material):
        reading = (material.get('author_context_unverified') or {}).get('request_understanding') or {}
        known = known and material.get('source_model_view_status') == 'known_source_quoted_hypotheses_purpose_view' \
            and 'request_reading_material_reference' in reading and 'creation_intent_reference' in reading
    if not known:
        data['purpose_allocation'] = {'status': 'legacy_material_retained',
            'scope': 'unknown_different_or_overridden_lineage_not_guessed_or_shortened'}
        return data

    def model_ref(path, value):
        return {'model_field': path, 'value_sha256': _digest(value)}

    shared_author = deepcopy(parent_material['author_context_unverified'])
    for key in ('knowledge', 'identity_binding'):
        shared_author[key] = model_ref('current_user_selections.' + key, current_user_selections[key])
    # A recipe snapshot can be a different historical choice. Keep that one
    # complete, unless it is exactly the current complete recipe value.
    if ('recipe_snapshot' in shared_author and 'full_style_recipes' in current_user_selections
            and shared_author['recipe_snapshot'] == current_user_selections['full_style_recipes']):
        shared_author['recipe_snapshot'] = model_ref('current_user_selections.full_style_recipes',
            current_user_selections['full_style_recipes'])
    shared_author.pop('selected_config', None)  # Past author config is not actual local sampler parameters.
    data['shared_reference_author_context_unverified'] = shared_author
    primitive_parameters = {'model_id', 'render_pipeline', 'mode', 'parameter_policy',
        'width', 'height', 'steps', 'cfg', 'seed', 'sampler', 'scheduler', 'shift', 'cfg_rescale',
        'weight_dtype', 'lora_name', 'lora_strength', 'style_preset', 'composition_style_preset',
        'character_style_preset', 'operation', 'batch_size'}
    # These known routes have no active highres/detail/handoff stages (checked
    # above and by the existing standard-parent contract). Their flags remain
    # explicit; their unused default knobs do not become effective parameters.
    stage_flags = {'highres_enabled', 'face_detail_enabled', 'hand_repair_enabled',
                   'text_clean_enabled', 'check_enabled'}
    region_parameters = {'region_repair_steps', 'region_repair_denoise', 'repair_source_sha256',
                         'repair_mask_sha256', 'parent_artwork_id', 'repair_instruction', 'repair_context_policy'}
    for role, material in (('parent', parent_material), ('child', child_material)):
        source = saved['sources'][role]
        request = source['source_job']['request']
        parameter_keys = primitive_parameters | stage_flags | (region_parameters if role == 'child' else set())
        actual = {key: deepcopy(request[key]) for key in sorted(parameter_keys) if key in request}
        withheld = {}
        for key in ('prompt', 'negative_prompt', 'repair_context_prompt'):
            if key in request:
                withheld[key] = {'value_sha256': _digest(request[key]), 'sent_as_model_content': False}
        packet = {key: deepcopy(material[key]) for key in ('artwork', 'image', 'source_job',
            'snapshot_sha256', 'condition_scope', 'condition_ancestry', 'feedback', 'host_references')}
        packet.update(actual_generation_conditions=actual,
            actual_generation_conditions_scope='effective_saved_parameters_not_new_request_or_quality_evidence',
            inactive_saved_parameter_material={'full_request_sha256': source['source_job']['request_sha256'],
                'host_field': 'sources.' + role + '.source_job.request', 'sent_as_effective_parameters': False,
                'scope': 'inactive_defaults_output_paths_provenance_and_compile_fields_remain_in_complete_host_request'},
            declared_author_context_reference=model_ref('shared_reference_author_context_unverified', shared_author),
            historical_execution_material={'full_request_sha256': source['source_job']['request_sha256'],
                'native_and_compilation_sha256': _digest({key: request.get(key) for key in
                    ('native_prompts', 'semantic_spec', 'scene_contract', 'author_proposal')}),
                'native_fields': withheld, 'sent_as_model_content': False,
                'scope': 'complete_host_execution_original_not_reference_quality_evidence_not_a_drawable_model_view'},
            live_source_revalidation_required=True, full_saved_request_sent_to_model=False)
        if role == 'parent':
            packet['declared_parent_picture_unverified'] = deepcopy(material['declared_parent_picture_unverified'])
            packet['declared_parent_picture_scope'] = material.get('declared_parent_picture_scope',
                'declared_complete_picture_not_pixel_verification')
        else:
            packet['inherited_scene_reference'] = model_ref(
                'reference_options.parent.source_conditions.declared_parent_picture_unverified',
                parent_material['declared_parent_picture_unverified'])
            packet['actual_local_operation'] = {key: deepcopy(request[key]) for key in
                ('repair_instruction', 'repair_context_policy', 'parent_artwork_id',
                 'repair_source_sha256', 'repair_mask_sha256', 'region_repair_steps', 'region_repair_denoise')}
            if automatic is not None:
                packet['actual_local_operation']['automatic_local_edit'] = deepcopy(automatic)
        data['reference_options'][role]['source_conditions'] = packet
    # Repeated original words are referenced only after exact value equality.
    p_feedback = data['reference_options']['parent']['source_conditions']['feedback']
    child_packet = data['reference_options']['child']['source_conditions']
    if child_packet['feedback'] == p_feedback:
        child_packet['feedback'] = model_ref('reference_options.parent.source_conditions.feedback', p_feedback)
    data['purpose_allocation'] = {'status': 'known_exact_local_lineage',
        'scope': 'reference_choice_only_complete_sources_retained_for_execution',
        'complete_native_sent_to_model': False, 'declared_scene_is_not_pixel_evidence': True}
    return data


def reference_model_material(context, *, current_creation_intent=None, current_request_reading_material=None,
                             view_version=None, current_user_selections=None):
    """Return one purpose view; complete sources/delta stay in the bound host."""
    saved = _checked_reference_context(context)
    if (current_creation_intent is None) != (current_request_reading_material is None):
        raise ValueError('Current Source intent and its full purpose scope must travel together.')
    if view_version not in (None, REFERENCE_PURPOSE_VIEW, REFERENCE_CONTENT_VIEW):
        raise ValueError('Unknown reference purpose view; no guessed upgrade or shortening.')
    if view_version in (REFERENCE_PURPOSE_VIEW, REFERENCE_CONTENT_VIEW) and not isinstance(current_user_selections, dict):
        raise ValueError('The new purpose view needs complete explicit current selections.')
    if view_version in (REFERENCE_PURPOSE_VIEW, REFERENCE_CONTENT_VIEW) and any(
            key in current_user_selections and current_user_selections[key] != saved[key]
            for key in ('original_text', 'original_sha256')):
        raise ValueError('Current selections do not belong to the bound complete original.')
    full = saved['contract_version'] == FULL_REFERENCE_CHOICE_CONTRACT
    data = {'contract_version': saved['contract_version'], 'original_text': saved['original_text'],
        'original_sha256': saved['original_sha256'], 'reference_context_binding_sha256': context['binding_sha256'],
        'reference_options': {}, 'visible_delta_unverified': (
            full_visible_delta_model_material(saved['visible_delta_raw']) if full
            else visible_delta_model_material(saved['visible_delta_raw'])),
        'comparison_material': deepcopy(saved['comparison_material']),
        'live_source_revalidation_required': True, 'reference_choice_is_not_user_selection': True}
    previous_key = 'previous_operation' if full else 'previous_local_operation'
    data[previous_key] = deepcopy(saved[previous_key])
    if current_creation_intent is not None:
        data.update(creation_intent=deepcopy(current_creation_intent),
                    request_reading_material=deepcopy(current_request_reading_material))
    for role, source in saved['sources'].items():
        material = _source_purpose_material(source, current_creation_intent, current_request_reading_material)
        data['reference_options'][role] = {'source_conditions': material,
            'full_observation': deepcopy(saved['observations'][role])}
    if view_version in (REFERENCE_PURPOSE_VIEW, REFERENCE_CONTENT_VIEW):
        data['purpose_view_version'] = view_version
        data['current_user_selections'] = deepcopy(current_user_selections)
        if full:
            # Two full drafts can have different scenes, styles or models. The
            # local allocation would wrongly substitute inherited parent art.
            data['purpose_allocation'] = {'status': 'independent_full_sources_retained',
                'scope': 'each_full_source_keeps_its_own_conditions_not_assumed_local_inheritance'}
            return data
        return _allocate_known_local_purpose(saved, data, current_user_selections)
    return data


def resolve_reference_choice(context, raw):
    """Parse and point at the chosen snapshot; caller validates live capabilities."""
    saved = _checked_reference_context(context)
    choice = ReferenceNextChoice.model_validate(raw)
    selected = None if choice.reference_base == 'neither' else deepcopy(saved['sources'][choice.reference_base])
    return choice, selected
