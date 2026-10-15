"""Explicit frozen originals -> existing visual runtime, never a decision maker.

Construction performs no I/O or inference. Only an explicit observe() request
calls the injected runtime. A current package and JSON output do not establish
understanding, image quality, or permission to adopt/dispatch the result.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import io
import json
import threading
from typing import Any, Literal

from PIL import Image

from .grounded_region import (LOCALIZATION_SYSTEM, RegionLocalizationRequest,
                             localization_schema)
from .iteration_observation import StaleObservationError
from .shared_original_json import share_original_json
from .local_authoring_runtime import (
    AuthoringRuntimeResult, LocalAuthoringError, RuntimeConfig, VisualImageInput,
)


class VisualObservationError(ValueError):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


class StaleVisualObservationError(StaleObservationError):
    """Rejected after inference; retain the unadopted raw evidence for inspection."""
    def __init__(self, observation_id: str, raw_runtime_result: AuthoringRuntimeResult,
                 image_bindings: tuple[dict, ...], message: str, *,
                 input_policy: Literal['full_archive', 'pixels_and_original'] = 'full_archive',
                 withheld_archive_fields: tuple[str, ...] = (), input_scope_note: str = ''):
        super().__init__(message)
        self.observation_id = observation_id
        self.raw_runtime_result = raw_runtime_result
        self.image_bindings = image_bindings
        self.not_adopted = True
        self.input_policy = input_policy
        self.withheld_archive_fields = withheld_archive_fields
        self.input_scope_note = input_scope_note


class CancelledVisualObservationError(LocalAuthoringError):
    """A reply arrived, but cancellation revoked its use; preserve the raw reply."""
    def __init__(self, observation_id, raw_runtime_result, image_bindings, *,
                 input_policy, withheld_archive_fields, input_scope_note):
        super().__init__('cancelled', 'Frozen visual observation was cancelled; no result was adopted.',
            raw_runtime_result.evidence_dir)
        self.observation_id = observation_id
        self.raw_runtime_result = raw_runtime_result
        self.image_bindings = image_bindings
        self.not_adopted = True
        self.input_policy = input_policy
        self.withheld_archive_fields = withheld_archive_fields
        self.input_scope_note = input_scope_note


@dataclass(frozen=True)
class VisualObservationResult:
    run_id: str
    observation_id: str
    image_bindings: tuple[dict, ...]
    raw_runtime_result: AuthoringRuntimeResult
    not_adopted: bool = True
    decision_created: bool = False
    quality_verified: bool = False
    semantic_understanding_verified: bool = False
    input_policy: Literal['full_archive', 'pixels_and_original'] = 'full_archive'
    withheld_archive_fields: tuple[str, ...] = ()
    input_scope_note: str = ''


_PIXELS_WITHHELD_ARCHIVE_FIELDS = (
    'context_except_original_text',
    'run',
    'user_reviews',
    'job_requests',
    'manifest_except_original_sha256_and_image_metadata',
    'image_bindings.roles_by_artifact',
)
_PIXELS_INPUT_SCOPE_NOTE = (
    'Only the complete original text/digest and neutral actual-image bindings are sent. '
    'The frozen archive retains user/candidate reviews, interpretations, job prompts and '
    'author choices, but these and descriptive role labels are not sent to this visual call. '
    'The original text can itself influence interpretation; this is not pure/unbiased '
    'observation or verified understanding.'
)


def _json(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, allow_nan=False,
                      separators=(',', ':')).encode('utf-8')


class IterationVisualObserver:
    def __init__(self, controller, observations, runtime):
        self.controller, self.observations, self.runtime = controller, observations, runtime
        self._check_binding()

    def _check_binding(self) -> None:
        if (self.observations is None or self.controller.observations is not self.observations
                or self.controller.store is not self.observations.iteration_store
                or self.controller.jobs is not self.observations.jobs
                or self.controller.artworks is not self.observations.artworks):
            raise VisualObservationError('unbound_observer', 'Use the same explicit controller/stores/observation instance.')
        if not isinstance(getattr(self.runtime, 'config', None), RuntimeConfig):
            raise VisualObservationError('invalid_runtime', 'Inject an existing runtime with explicit RuntimeConfig; no default is chosen.')

    @staticmethod
    def _cancelled(cancel_event) -> None:
        if cancel_event is not None and cancel_event.is_set():
            raise LocalAuthoringError('cancelled', 'Frozen visual observation was cancelled; no result was adopted.')

    def _context_without_sync(self, run_id: str) -> dict:
        """Prevent _context's existing reconcile/sync helpers from writing records."""
        self._check_binding()
        run = self.controller.store.runs.get(run_id)
        if run is None:
            raise VisualObservationError('missing_run', 'The bound current iteration run is missing.')
        if any(step.status in {'reserved', 'submitted'} for step in run.steps):
            raise VisualObservationError('context_needs_reconciliation', 'Observe only already reconciled originals; no in-flight run is changed here.')
        for step in run.steps:
            job = self.controller.jobs.get(step.job_id) if step.job_id else None
            if job is None or job.status != 'completed':
                continue
            for index, image_url in enumerate(job.images):
                artwork = self.controller.artworks.get(self.controller.artworks.id_for(job.id, index))
                file_path = job.files[index] if index < len(job.files) else ''
                if (artwork is None or artwork.image_url != image_url or artwork.file_path != file_path
                        or artwork.checks != list(job.checks)):
                    raise VisualObservationError('context_needs_reconciliation', 'Artwork metadata must already be synchronized; this observer does not write it.')
        return self.controller._context(run)

    @staticmethod
    def _limits(config: RuntimeConfig) -> dict[str, int]:
        ceilings = RuntimeConfig()
        limits = {}
        for name in ('max_images', 'max_image_bytes', 'max_total_image_bytes',
                     'max_image_pixels', 'max_total_image_pixels', 'max_input_bytes'):
            value = getattr(config, name)
            if type(value) is not int or not 0 < value <= getattr(ceilings, name):
                raise VisualObservationError('invalid_capacity', 'Do not raise or disable the existing runtime capacity: ' + name)
            limits[name] = value
        return limits

    def _images(self, archive: dict, config: RuntimeConfig, *,
                input_policy: Literal['full_archive', 'pixels_and_original'] = 'full_archive',
                artifact_ids: tuple[str, ...] | None = None,
                ) -> tuple[tuple[dict, ...], tuple[VisualImageInput, ...]]:
        manifest = archive['manifest']
        roles_by_id, ordered_ids = {}, []
        for role in sorted(manifest['roles']):
            for artwork_id in sorted(manifest['roles'][role]):
                if artwork_id not in roles_by_id:
                    roles_by_id[artwork_id] = []
                    ordered_ids.append(artwork_id)
                roles_by_id[artwork_id].append(role)
        if input_policy == 'pixels_and_original':
            # Do not make the wire order depend on descriptive role names either.
            ordered_ids.sort()
        if artifact_ids is not None:
            if not artifact_ids or not set(artifact_ids) <= set(ordered_ids):
                raise VisualObservationError('image_binding_mismatch', 'Localization requires an actual image in this current packet.')
            ordered_ids = [identifier for identifier in ordered_ids if identifier in artifact_ids]
        limits = self._limits(config)
        groups, group_order = {}, []
        for artwork_id in ordered_ids:
            metadata = manifest['images'][artwork_id]
            if metadata['format'] not in {'PNG', 'JPEG'}:
                raise VisualObservationError('unsupported_image', 'Frozen originals must already be PNG/JPEG; no conversion is substituted.')
            if metadata['sha256'] not in groups:
                groups[metadata['sha256']] = {'metadata': metadata, 'artifact_ids': [], 'roles_by_artifact': {}}
                group_order.append(metadata['sha256'])
            group = groups[metadata['sha256']]
            if any(metadata[key] != group['metadata'][key] for key in ('bytes', 'width', 'height', 'format')):
                raise VisualObservationError('image_binding_mismatch', 'Same-SHA aliases disagree about actual original dimensions/format.')
            group['artifact_ids'].append(artwork_id)
            group['roles_by_artifact'][artwork_id] = roles_by_id[artwork_id]
        if len(groups) > limits['max_images']:
            raise VisualObservationError('image_bound_exceeded', 'Selected originals exceed runtime image count; none were dropped.')
        total_bytes = sum(groups[key]['metadata']['bytes'] for key in group_order)
        total_pixels = sum(groups[key]['metadata']['width'] * groups[key]['metadata']['height'] for key in group_order)
        if (total_bytes > limits['max_total_image_bytes'] or total_pixels > limits['max_total_image_pixels']
                or any(groups[key]['metadata']['bytes'] > limits['max_image_bytes']
                       or groups[key]['metadata']['width'] * groups[key]['metadata']['height'] > limits['max_image_pixels']
                       for key in group_order)):
            raise VisualObservationError('image_bound_exceeded', 'Complete originals exceed runtime byte/pixel limits; no reduced originals are used.')
        if groups and (config.mmproj_path is None or config.constrain_json_output is not True):
            raise VisualObservationError('invalid_visual_runtime', 'Images require an explicitly configured projector and hard JSON grammar.')
        bindings, images = [], []
        for index, key in enumerate(group_order, 1):
            group = groups[key]
            item = group['metadata']
            raw = self.observations.read_image_bytes(manifest['id'], manifest['run_id'], group['artifact_ids'][0])
            if len(raw) != item['bytes'] or hashlib.sha256(raw).hexdigest() != item['sha256']:
                raise VisualObservationError('image_binding_mismatch', 'Actual archived bytes differ from the manifest-bound SHA/size.')
            try:
                with Image.open(io.BytesIO(raw)) as picture:
                    if ((picture.width, picture.height, picture.format) != (item['width'], item['height'], item['format'])
                            or getattr(picture, 'n_frames', 1) != 1):
                        raise VisualObservationError('unsupported_image', 'Original metadata is inconsistent or animated; no substitute is used.')
                    picture.verify()
                with Image.open(io.BytesIO(raw)) as picture:
                    picture.load()
            except VisualObservationError:
                raise
            except Exception as exc:
                raise VisualObservationError('invalid_image', 'Complete original image could not be decoded.') from exc
            mime = {'PNG': 'image/png', 'JPEG': 'image/jpeg'}[item['format']]
            binding = {'wire_index': index, 'artifact_ids': group['artifact_ids'],
                'roles_by_artifact': group['roles_by_artifact'], 'sha256': item['sha256'],
                'width': item['width'], 'height': item['height'], 'mime_type': mime,
                'source_labels_verified': False}
            label_data = {'observation_id': manifest['id'], 'artifact_ids': binding['artifact_ids']}
            if input_policy == 'full_archive':
                label_data['roles_by_artifact'] = binding['roles_by_artifact']
            else:
                # Descriptive selected-role names can contain prior judgements.
                # Keep every actual alias/SHA mapping, but not those role labels,
                # in the explicitly scoped observation input or returned mapping.
                del binding['roles_by_artifact']
            label = _json(label_data).decode('utf-8')
            if len(label.encode('utf-8')) > 4096:
                raise VisualObservationError('image_bound_exceeded', 'Complete role label exceeds the runtime bound; no aliases were removed.')
            bindings.append(binding)
            images.append(VisualImageInput(raw, item['sha256'], mime, source_ref=label))
        return tuple(bindings), tuple(images)

    async def observe(self, run_id: str, observation_id: str, *, system_prompt: str,
                      response_schema: dict, cancel_event: threading.Event | None = None,
                      owner: str | None = None,
                      input_policy: Literal['full_archive', 'pixels_and_original'] = 'full_archive',
                      region_request: RegionLocalizationRequest | None = None,
                      ) -> VisualObservationResult:
        self._cancelled(cancel_event)
        if not isinstance(input_policy, str) or input_policy not in {'full_archive', 'pixels_and_original'}:
            raise VisualObservationError('invalid_input_policy', 'Choose an explicit supported visual input policy; no fallback is selected.')
        context = self._context_without_sync(run_id)
        self.observations.validate_still_current(observation_id, context)
        archive = self.observations.read_archive(observation_id, run_id)
        if region_request is not None:
            if not isinstance(region_request, RegionLocalizationRequest) or input_policy != 'pixels_and_original':
                raise VisualObservationError('invalid_input', 'Localization must use an explicit validated request and pixel/original input scope.')
            metadata = archive['manifest']['images'].get(region_request.artifact_id)
            if metadata is None or metadata['format'] != 'PNG':
                raise VisualObservationError('image_binding_mismatch', 'Localization requires a bound original PNG in this current packet.')
            system_prompt = LOCALIZATION_SYSTEM
            response_schema = localization_schema(observation_id, region_request.artifact_id, metadata)
        config = self.runtime.config
        bindings, images = self._images(archive, config, input_policy=input_policy,
            artifact_ids=(region_request.artifact_id,) if region_request is not None else None)
        withheld_fields, scope_note = (), ''
        if input_policy == 'full_archive':
            data = {'original_text': archive['context']['original_text'],
                'original_sha256': archive['manifest']['original_sha256'], 'observation_id': observation_id,
                'full_archive': archive, 'image_bindings': bindings,
                'concept_paths_are_pointers_not_loaded_content': True,
                'source_labels_are_untrusted': True, 'result_not_adopted': True}
        else:
            data = {'original_text': archive['context']['original_text'],
                'original_sha256': archive['manifest']['original_sha256'], 'observation_id': observation_id,
                'image_bindings': bindings,
                'source_labels_are_untrusted': True, 'result_not_adopted': True}
            withheld_fields = _PIXELS_WITHHELD_ARCHIVE_FIELDS
            scope_note = _PIXELS_INPUT_SCOPE_NOTE
            data.update(input_policy=input_policy, withheld_archive_fields=withheld_fields,
                        input_scope_note=scope_note, frozen_archive_preserved=True)
        if region_request is not None:
            scope_note = _PIXELS_INPUT_SCOPE_NOTE.replace(
                'Only the complete original text/digest and neutral actual-image bindings are sent.',
                'This explicit localization sends the complete original text/digest, neutral actual-image '
                'bindings and requested edit intent; the intent can influence the region selected.')
            data['input_scope_note'] = scope_note
            data['region_request'] = region_request.model_dump(mode='json') | {
                'observation_id': observation_id, 'source_sha256': metadata['sha256'],
                'width': metadata['width'], 'height': metadata['height'],
                'coordinate_space': 'original_image_pixels', 'result_is_unverified_geometry': True}
        if not isinstance(system_prompt, str) or not system_prompt.strip():
            raise VisualObservationError('invalid_input', 'An explicit trusted system prompt is required.')
        if not isinstance(response_schema, dict) or response_schema.get('type') != 'object':
            raise VisualObservationError('invalid_schema', 'An explicit complete object response schema is required.')
        try:
            if input_policy == 'full_archive':
                shared = share_original_json(archive, check_cancel=lambda: self._cancelled(cancel_event))
                if shared.shared:
                    data['full_archive'] = shared.value
                    data['full_archive_encoding'] = shared.encoding
            request = _json({'system_prompt': system_prompt, 'data': data, 'schema': response_schema})
        except (TypeError, ValueError) as exc:
            raise VisualObservationError('invalid_input', 'Complete original input/schema must be finite JSON.') from exc
        if len(request) > self._limits(config)['max_input_bytes']:
            message = ('Full archive/system/schema exceed 128KiB runtime capacity; nothing was summarized or omitted.'
                       if input_policy == 'full_archive' else
                       'Complete original/system/schema/image bindings exceed runtime input capacity; no shortening was applied to the selected input scope.')
            raise VisualObservationError('input_too_large', message)
        frozen = json.loads(request)
        self._cancelled(cancel_event)
        self.observations.validate_still_current(observation_id, self._context_without_sync(run_id))
        if self.runtime.config is not config:
            raise VisualObservationError('runtime_changed', 'The explicit runtime changed before its call; no inference was requested.')
        raw_result = await self.runtime.generate_json(system_prompt=frozen['system_prompt'], data=frozen['data'],
            response_schema=frozen['schema'], images=images, cancel_event=cancel_event,
            owner=owner or 'visual-observation-' + observation_id)
        if cancel_event is not None and cancel_event.is_set():
            raise CancelledVisualObservationError(observation_id, raw_result, bindings,
                input_policy=input_policy, withheld_archive_fields=withheld_fields, input_scope_note=scope_note)
        try:
            if self.runtime.config is not config:
                raise ValueError('The explicitly injected runtime configuration changed during observation.')
            self.observations.validate_still_current(observation_id, self._context_without_sync(run_id))
        except (OSError, ValueError) as exc:
            raise StaleVisualObservationError(observation_id, raw_result, bindings,
                'Original inputs changed before the raw visual result could be returned: ' + str(exc),
                input_policy=input_policy, withheld_archive_fields=withheld_fields,
                input_scope_note=scope_note) from exc
        return VisualObservationResult(run_id, observation_id, bindings, raw_result,
            input_policy=input_policy, withheld_archive_fields=withheld_fields,
            input_scope_note=scope_note)
