"""Explicit, bounded original-artifact input for one iteration observation.

This archive proves which bytes and words were available, not that an observer
understood them or that a picture is good. It does not load models, dispatch jobs,
update user reviews, or create snapshots merely because a context was requested.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from io import BytesIO
import hashlib
import json
import os
from pathlib import Path
import re
from typing import TYPE_CHECKING

from PIL import Image

if TYPE_CHECKING:
    from .artworks import ArtworkStore
    from .iteration_controller import IterationStore
    from .jobs import JobManager


class StaleObservationError(ValueError):
    """Original inputs or the archive changed; another observation is needed."""


@dataclass(frozen=True)
class ObservationLimits:
    max_artifacts: int = 16
    max_image_bytes: int = 64 * 1024 * 1024
    max_total_image_bytes: int = 256 * 1024 * 1024
    max_json_bytes: int = 32 * 1024 * 1024
    max_image_pixels: int = 40_000_000


def _json_bytes(value) -> bytes:
    return json.dumps(value, ensure_ascii=False, allow_nan=False, sort_keys=True,
                      separators=(',', ':')).encode('utf-8')


def _digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _archive_json(data: bytes) -> dict:
    """Archive metadata is an unambiguous finite JSON object, not permissive input."""
    def unique_object(pairs):
        value = {}
        for key, item in pairs:
            if key in value:
                raise ValueError('Duplicate archive JSON key: ' + key)
            value[key] = item
        return value
    def nonfinite(value):
        raise ValueError('Nonfinite archive JSON number: ' + value)
    value = json.loads(data, object_pairs_hook=unique_object, parse_constant=nonfinite)
    if not isinstance(value, dict):
        raise ValueError('Archive metadata must be a JSON object.')
    return value


def _input_identity(state: dict) -> dict:
    """Creative inputs, separate from the complete audit archive.

    Ignore only the known clock/queue accounting fields. All other fields,
    including future additions, remain protected by default. Dispatch status,
    revision, committed decisions, budgets and branch/selection state are not
    clock noise; a fresh identity is not permission to execute.
    """
    identity = json.loads(_json_bytes(state))
    for key in ('events', 'updated_at', 'active_seconds', 'active_since', 'decision_attempts'):
        identity['run'].pop(key, None)
    for key in ('remaining_seconds', 'remaining_decisions'):
        identity['context'].pop(key, None)
    return {'version': 1, 'sha256': _digest(_json_bytes(identity))}


def _stat(value) -> dict:
    return {'bytes': value.st_size, 'mtime_ns': value.st_mtime_ns, 'ctime_ns': value.st_ctime_ns,
            'device': value.st_dev, 'inode': value.st_ino}


def _simple_id(value, *, name: str, maximum: int = 80) -> str:
    if not isinstance(value, str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,' + str(maximum - 1) + '}', value):
        raise ValueError(name + ' must be a simple ASCII identifier, not a path.')
    return value


class IterationObservations:
    """Bind existing stores; construction and validation are read-only.

    `context` must be the controller's current context plus its run revision.
    Membership is checked independently against the bound IterationStore, so
    inserting another run's artwork or a path into that context is not authority.
    `selected_artifacts` maps descriptive roles to lists of existing artwork IDs.
    Multiple roles or IDs may refer to one image; identical bytes are stored once.
    """

    def __init__(self, artworks: ArtworkStore, jobs: JobManager, iteration_store: IterationStore,
                 evidence_root: Path, *, limits: ObservationLimits | None = None):
        self.artworks, self.jobs, self.iteration_store = artworks, jobs, iteration_store
        root = Path(evidence_root)
        if not root.is_absolute() or root.resolve() == Path(root.anchor).resolve():
            raise ValueError('Specify an absolute, dedicated owned evidence_root, not a drive root.')
        self.evidence_root = root.resolve()
        self.output_root = Path(jobs.output_root).resolve()
        protected = [self.output_root, Path(artworks.path).resolve().parent, Path(iteration_store.root).resolve()]
        if any(self.evidence_root.is_relative_to(path) for path in protected):
            raise ValueError('Observation evidence must be outside the original artwork/job/run stores.')
        self.limits = limits or ObservationLimits()
        ceilings = ObservationLimits()
        for name, value in asdict(self.limits).items():
            if type(value) is not int or not 1 <= value <= getattr(ceilings, name):
                raise ValueError('Observation capacity must be positive and no larger than the explicit boundary: ' + name)

    def _folder(self, snapshot_id: str) -> Path:
        value = self.evidence_root / _simple_id(snapshot_id, name='snapshot_id')
        if value.is_symlink() or not value.resolve().is_relative_to(self.evidence_root):
            raise ValueError('Observation directory cannot redirect outside the owned evidence root.')
        return value

    def _selection(self, selected_artifacts: dict) -> dict:
        if not isinstance(selected_artifacts, dict) or len(selected_artifacts) > self.limits.max_artifacts:
            raise ValueError('Select a bounded role-to-artwork-ID mapping.')
        selection, ids = {}, set()
        for role, values in selected_artifacts.items():
            if not isinstance(role, str) or not re.fullmatch(r'[a-z][a-z0-9_]{0,39}', role):
                raise ValueError('Artifact roles must be simple lowercase labels.')
            if not isinstance(values, (list, tuple)) or not values or len(values) > self.limits.max_artifacts:
                raise ValueError('Each selected role needs a bounded nonempty list of artwork IDs.')
            selection[role] = list(dict.fromkeys(_simple_id(value, name='artwork_id', maximum=120) for value in values))
            ids.update(selection[role])
        if len(ids) > self.limits.max_artifacts:
            raise ValueError('Selected artwork count exceeds the explicit observation capacity.')
        return selection

    def _source_state(self, context: dict) -> dict:
        if not isinstance(context, dict):
            raise ValueError('A current controller context is required.')
        required = {'run_id', 'revision', 'request', 'original_text', 'original_sha256', 'artifacts',
            'artifact_inputs', 'feedback_signature', 'steps', 'decisions', 'incumbent_artwork_id', 'candidate_reviews'}
        if not required <= context.keys():
            raise ValueError('Current context is incomplete; include the authoritative run revision.')
        run_id = context['run_id']
        if not isinstance(run_id, str) or not re.fullmatch(r'[0-9a-f]{12}', run_id):
            raise ValueError('Invalid bound iteration run ID.')
        run = self.iteration_store.runs.get(run_id)
        if run is None:
            raise ValueError('Iteration run is missing from the bound store.')
        record = run.model_dump(mode='json')
        if type(context['revision']) is not int or context['revision'] < 0:
            raise ValueError('Context revision must be a nonnegative integer.')
        for key, source_key in [('revision', 'revision'), ('request', 'base_request'), ('original_text', 'original_text'),
                ('original_sha256', 'original_sha256'), ('steps', 'steps'), ('decisions', 'decisions'),
                ('incumbent_artwork_id', 'incumbent_artwork_id'), ('candidate_reviews', 'candidate_reviews')]:
            if context[key] != record[source_key]:
                raise StaleObservationError('Controller context differs from the bound current run: ' + key)
        if (not record['original_text'] or _digest(record['original_text'].encode('utf-8')) != record['original_sha256']):
            raise ValueError('The actual run has a missing or inconsistent original text digest.')
        known = list(dict.fromkeys([*record['seed_artwork_ids'],
            *[step['artwork_id'] for step in record['steps'] if step['artwork_id']]]))
        artifacts, requests = {}, {}
        for artwork_id in known:
            artwork = self.artworks.get(artwork_id)
            if artwork is None:
                raise ValueError('Known run artwork is missing: ' + artwork_id)
            job = self.jobs.get(artwork.job_id)
            if job is None or job.status != 'completed':
                raise ValueError('Known artwork has no completed bound job record: ' + artwork_id)
            artifacts[artwork_id], requests[artwork_id] = asdict(artwork), job.request
        if context['artifacts'] != list(artifacts.values()) or context['artifact_inputs'] != requests:
            raise StaleObservationError('Context artwork metadata or actual job inputs differ from the bound stores.')
        reviews = {artwork_id: item['reviews'] for artwork_id, item in artifacts.items()}
        signature = hashlib.sha256(json.dumps(reviews, ensure_ascii=False, sort_keys=True).encode('utf-8')).hexdigest()
        if context['feedback_signature'] != signature:
            raise StaleObservationError('Context natural-language feedback differs from the current user records.')
        # JSON detaches all live mutable references without normalizing prose.
        state = {'context': context, 'run': record, 'artifacts': artifacts, 'job_requests': requests, 'reviews': reviews}
        data = _json_bytes(state)
        if len(data) > self.limits.max_json_bytes:
            raise ValueError('Complete context/requests/reviews exceed the metadata capacity; nothing was summarized or omitted.')
        return json.loads(data)

    def _image(self, artwork: dict) -> tuple[dict, bytes]:
        path = Path(artwork['file_path'])
        if (not artwork['file_path'] or not path.is_absolute() or path.is_symlink()
                or not path.resolve().is_relative_to(self.output_root) or not path.is_file()):
            raise ValueError('Bound original image is missing or outside the job output root: ' + artwork['id'])
        source = path.resolve()
        job = self.jobs.get(artwork['job_id'])
        if artwork.get('source_stage') is None:
            index = artwork['image_index']
            if (type(index) is not int or index < 0 or index >= len(job.files)
                    or Path(job.files[index]).resolve() != source):
                raise ValueError('Artwork image does not match its completed bound job output: ' + artwork['id'])
        before = source.stat()
        if before.st_size <= 0 or before.st_size > self.limits.max_image_bytes:
            raise ValueError('Original image exceeds the per-image byte capacity or is empty.')
        with source.open('rb') as stream:
            opened = os.fstat(stream.fileno())
            data = stream.read(self.limits.max_image_bytes + 1)
            after_read = os.fstat(stream.fileno())
        after = source.stat()
        # Windows Python exposes creation time through Path.stat().st_ctime,
        # but its descriptor fstat may expose a different time in that field.
        # Compare each API's full before/after record and cross-check stable
        # identity/size/mtime, rather than mistaking that API difference for a write.
        identity = lambda value: {key: item for key, item in _stat(value).items() if key != 'ctime_ns'}
        if (_stat(before) != _stat(after) or _stat(opened) != _stat(after_read)
                or identity(before) != identity(opened) or identity(after_read) != identity(after)
                or len(data) != before.st_size):
            raise StaleObservationError('Original image changed while it was being read.')
        stage_sha = (artwork.get('source_stage') or {}).get('sha256')
        if stage_sha is not None and stage_sha != _digest(data):
            raise StaleObservationError('Retained-stage original no longer matches its bound source SHA.')
        try:
            with Image.open(BytesIO(data)) as image:
                width, height, mode, image_format = image.width, image.height, image.mode, image.format
                if width * height > self.limits.max_image_pixels:
                    raise ValueError('Original image exceeds the pixel capacity; no reduced image was substituted.')
                image.verify()
        except (OSError, SyntaxError) as error:
            raise ValueError('Original artwork is not a complete readable image.') from error
        return {'source_path': str(source), 'source_stat': _stat(before), 'sha256': _digest(data),
            'bytes': len(data), 'width': width, 'height': height, 'mode': mode, 'format': image_format}, data

    def _collect(self, context: dict, selection: dict) -> tuple[dict, dict, dict]:
        state = self._source_state(context)
        ids = list(dict.fromkeys(value for values in selection.values() for value in values))
        if any(value not in state['artifacts'] for value in ids):
            raise ValueError('Selected artifact is not a known ID in this bound iteration run.')
        images, blobs = {}, {}
        for artwork_id in ids:
            images[artwork_id], data = self._image(state['artifacts'][artwork_id])
            blobs.setdefault(images[artwork_id]['sha256'], data)
            if sum(map(len, blobs.values())) > self.limits.max_total_image_bytes:
                raise ValueError('Complete selected originals exceed the total byte capacity; no originals were reduced or omitted.')
        if _input_identity(self._source_state(context)) != _input_identity(state):
            raise StaleObservationError('Bound run, conditions, or user feedback changed while originals were read.')
        return state, images, blobs

    def freeze(self, context: dict, selected_artifacts: dict, *, snapshot_id: str,
               expected_image_sha256: dict[str, str] | None = None) -> dict:
        """Exclusively archive originals, optionally bound to what the caller saw.

        Expected hashes may cover any subset of selected IDs. They bind actual
        archived bytes, not future source currentness or visual understanding.
        """
        folder, selection = self._folder(snapshot_id), self._selection(selected_artifacts)
        if folder.exists():
            raise FileExistsError('Observation ID already exists; it will not be overwritten or silently reused.')
        expected = None
        if expected_image_sha256 is not None:
            selected = {value for values in selection.values() for value in values}
            if (not isinstance(expected_image_sha256, dict) or not set(expected_image_sha256) <= selected
                    or any(not isinstance(value, str) or not re.fullmatch(r'[0-9a-f]{64}', value)
                           for value in expected_image_sha256.values())):
                raise ValueError('Expected image SHA mapping must contain only selected IDs with 64 lowercase hex digits.')
            expected = dict(expected_image_sha256)
        context = json.loads(_json_bytes(context))
        state, images, blobs = self._collect(context, selection)
        if expected is not None:
            for artwork_id, sha256 in expected.items():
                image_sha = images[artwork_id]['sha256']
                if image_sha != sha256 or _digest(blobs[image_sha]) != sha256:
                    raise StaleObservationError('Actual original image differs from the caller expected SHA: ' + artwork_id)
        payloads = {'context.json': _json_bytes(state['context']), 'run.json': _json_bytes(state['run']),
                    'user_reviews.json': _json_bytes(state['reviews'])}
        requests = {}
        extensions = {'PNG': '.png', 'JPEG': '.jpg', 'WEBP': '.webp', 'GIF': '.gif', 'BMP': '.bmp', 'TIFF': '.tiff'}
        for artwork_id in images:
            job_id = state['artifacts'][artwork_id]['job_id']
            _simple_id(job_id, name='job_id', maximum=120)
            filename = 'requests/' + job_id + '.json'
            payloads[filename] = _json_bytes(state['job_requests'][artwork_id])
            requests[artwork_id] = filename
            images[artwork_id]['snapshot_image'] = ('images/' + images[artwork_id]['sha256'] +
                extensions.get(images[artwork_id]['format'], '.original'))
        if sum(map(len, payloads.values())) > self.limits.max_json_bytes:
            raise ValueError('Complete metadata files exceed the observation capacity; no partial archive was created.')
        payloads.update({item['snapshot_image']: blobs[item['sha256']] for item in images.values()})
        manifest = {'version': 1, 'id': snapshot_id, 'run_id': state['run']['id'], 'revision': state['run']['revision'],
            'created_at': datetime.now(UTC).isoformat(), 'status': 'original_evidence_frozen_understanding_unverified',
            'original_sha256': state['run']['original_sha256'], 'feedback_signature': context['feedback_signature'],
            'source_state_sha256': _digest(_json_bytes(state)), 'input_identity': _input_identity(state),
            'roles': selection, 'images': images,
            'job_request_files': requests, 'artifacts_not_materialized': [id for id in state['artifacts'] if id not in images],
            'limits': asdict(self.limits), 'files': {name: {'sha256': _digest(data), 'bytes': len(data)} for name, data in payloads.items()},
            'quality_verified': False, 'semantic_understanding_verified': False, 'identity_accuracy_verified': False,
            'original_images_resized_or_reencoded': False, 'user_prose_normalized_or_classified': False}
        manifest_bytes = _json_bytes(manifest)
        receipt = _json_bytes({'version': 1, 'id': snapshot_id, 'manifest_sha256': _digest(manifest_bytes),
                               'manifest_bytes': len(manifest_bytes)})
        if sum(len(value) for name, value in payloads.items() if name.endswith('.json')) + len(manifest_bytes) + len(receipt) > self.limits.max_json_bytes:
            raise ValueError('Complete metadata and receipts exceed the observation capacity.')
        folder.mkdir(parents=True, exist_ok=False)
        for name, data in payloads.items():
            target = folder / name
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open('xb') as stream:
                stream.write(data)
        # Last file is the completeness receipt. A write failure leaves a partial
        # owned directory for reconciliation; it never deletes original data.
        with (folder / 'manifest.json').open('xb') as stream:
            stream.write(manifest_bytes)
        with (folder / 'receipt.json').open('xb') as stream:
            stream.write(receipt)
        return manifest

    def _archive_file(self, folder: Path, name: str, maximum: int) -> bytes:
        """Only relative manifest-owned paths; never open source paths/URLs."""
        relative = Path(name)
        path = folder / relative
        if (not isinstance(name, str) or not name or relative.is_absolute() or '..' in relative.parts
                or path.is_symlink() or not path.resolve().is_relative_to(folder) or not path.is_file()):
            raise ValueError('Frozen archive file is missing or redirects outside its directory: ' + str(name))
        size = path.stat().st_size
        if not 0 <= size <= maximum:
            raise ValueError('Frozen archive file exceeds its explicit capacity.')
        with path.open('rb') as stream:
            data = stream.read(maximum + 1)
        if len(data) != size or len(data) > maximum:
            raise ValueError('Frozen archive file changed while read or exceeds capacity.')
        return data

    def read_archive(self, snapshot_id: str, expected_run_id: str) -> dict:
        """Return complete verified frozen originals, even if no longer current.

        No bound source image/store is read here. The result expressly does not
        approve current use, quality, identity, or the archived interpretation.
        """
        folder = self._folder(snapshot_id)
        try:
            if not isinstance(expected_run_id, str) or not re.fullmatch(r'[0-9a-f]{12}', expected_run_id):
                raise ValueError('expected_run_id must be an actual iteration identifier.')
            manifest_bytes = self._archive_file(folder, 'manifest.json', self.limits.max_json_bytes)
            receipt_bytes = self._archive_file(folder, 'receipt.json', self.limits.max_json_bytes)
            receipt = _archive_json(receipt_bytes)
            if (set(receipt) != {'version', 'id', 'manifest_bytes', 'manifest_sha256'}
                    or type(receipt['version']) is not int or receipt['version'] != 1 or receipt['id'] != snapshot_id
                    or receipt.get('manifest_bytes') != len(manifest_bytes)
                    or receipt.get('manifest_sha256') != _digest(manifest_bytes)):
                raise ValueError('Completeness receipt or original manifest changed.')
            manifest = _archive_json(manifest_bytes)
            if (manifest.get('id') != snapshot_id or type(manifest.get('version')) is not int
                    or manifest['version'] != 1 or manifest.get('run_id') != expected_run_id):
                raise ValueError('Invalid observation identity/version or expected run ownership.')
            selection = self._selection(manifest['roles'])
            selected = {value for values in selection.values() for value in values}
            if not isinstance(manifest['images'], dict) or set(manifest['images']) != selected:
                raise ValueError('Frozen image IDs differ from the selected comparison roles.')
            if not isinstance(manifest['job_request_files'], dict) or set(manifest['job_request_files']) != selected:
                raise ValueError('Every selected artwork must retain its exact corresponding job request.')
            expected_files = {'context.json', 'run.json', 'user_reviews.json',
                *manifest['job_request_files'].values(), *[item['snapshot_image'] for item in manifest['images'].values()]}
            if not isinstance(manifest['files'], dict) or set(manifest['files']) != expected_files:
                raise ValueError('Archive file table does not bind all and only its complete original inputs.')
            data, image_names = {}, {item['snapshot_image'] for item in manifest['images'].values()}
            total_images = total_json = 0
            for name, metadata in manifest['files'].items():
                image = name in image_names
                original = self._archive_file(folder, name,
                    self.limits.max_image_bytes if image else self.limits.max_json_bytes)
                if (set(metadata) != {'bytes', 'sha256'} or type(metadata['bytes']) is not int
                        or len(original) != metadata['bytes'] or _digest(original) != metadata['sha256']):
                    raise ValueError('Frozen original bytes changed or are missing: ' + name)
                if image:
                    total_images += len(original)
                else:
                    total_json += len(original)
                    data[name] = _archive_json(original)
            if (total_images > self.limits.max_total_image_bytes or
                    total_json + len(manifest_bytes) + len(receipt_bytes) > self.limits.max_json_bytes):
                raise ValueError('Complete archived originals exceed the explicit observation capacity.')
            context, run, reviews = data['context.json'], data['run.json'], data['user_reviews.json']
            if run.get('id') != expected_run_id or context.get('run_id') != expected_run_id:
                raise ValueError('Frozen context/run belongs to another iteration.')
            for key, source_key in [('revision', 'revision'), ('request', 'base_request'), ('original_text', 'original_text'),
                    ('original_sha256', 'original_sha256'), ('steps', 'steps'), ('decisions', 'decisions'),
                    ('incumbent_artwork_id', 'incumbent_artwork_id'), ('candidate_reviews', 'candidate_reviews')]:
                if context[key] != run[source_key]:
                    raise ValueError('Frozen context differs from its original run: ' + key)
            if (not run['original_text'] or _digest(run['original_text'].encode('utf-8')) != run['original_sha256']
                    or manifest['original_sha256'] != run['original_sha256'] or manifest['revision'] != run['revision']):
                raise ValueError('Frozen original text/revision does not match its receipt.')
            artifacts = {item['id']: item for item in context['artifacts']}
            known = list(dict.fromkeys([*run['seed_artwork_ids'],
                *[step['artwork_id'] for step in run['steps'] if step['artwork_id']]]))
            if (len(artifacts) != len(context['artifacts']) or set(artifacts) != set(known)
                    or set(context['artifact_inputs']) != set(known)
                    or manifest['artifacts_not_materialized'] != [id for id in sorted(artifacts) if id not in selected]):
                raise ValueError('Frozen artifact membership is inconsistent with the actual archived run.')
            expected_reviews = {id: item['reviews'] for id, item in artifacts.items()}
            feedback = hashlib.sha256(json.dumps(expected_reviews, ensure_ascii=False, sort_keys=True).encode('utf-8')).hexdigest()
            if reviews != expected_reviews or context['feedback_signature'] != feedback or manifest['feedback_signature'] != feedback:
                raise ValueError('Frozen user prose differs from its feedback receipt.')
            requests = {}
            for artwork_id in selected:
                item, artwork = manifest['images'][artwork_id], artifacts[artwork_id]
                request_file = manifest['job_request_files'][artwork_id]
                if (request_file != 'requests/' + _simple_id(artwork['job_id'], name='job_id', maximum=120) + '.json'
                        or data[request_file] != context['artifact_inputs'][artwork_id]):
                    raise ValueError('Frozen job request is not the selected artwork original.')
                image_file = item['snapshot_image']
                if (not isinstance(item['sha256'], str) or not re.fullmatch(r'[0-9a-f]{64}', item['sha256'])
                        or Path(image_file).parent.as_posix() != 'images'
                        or Path(image_file).stem != item['sha256']
                        or manifest['files'][image_file]['sha256'] != item['sha256']
                        or manifest['files'][image_file]['bytes'] != item['bytes']
                        # Archived source paths are provenance strings, not
                        # locations to resolve/read again. Their originals may
                        # have moved or disappeared without corrupting this archive.
                        or not Path(item['source_path']).is_absolute()
                        or not Path(artwork['file_path']).is_absolute()
                        or type(item['width']) is not int or type(item['height']) is not int
                        or item['width'] < 1 or item['height'] < 1
                        or item['width'] * item['height'] > self.limits.max_image_pixels):
                    raise ValueError('Frozen original image identity/dimensions are not manifest-bound.')
                requests[artwork_id] = data[request_file]
            archived_state = {'context': context, 'run': run, 'artifacts': artifacts,
                'job_requests': context['artifact_inputs'], 'reviews': reviews}
            if manifest['source_state_sha256'] != _digest(_json_bytes(archived_state)):
                raise ValueError('Complete archived source-state audit hash differs from its originals.')
            if 'input_identity' in manifest and manifest['input_identity'] != _input_identity(archived_state):
                raise ValueError('Archived input identity differs from the original protected inputs.')
            return {'manifest': manifest, 'context': context, 'run': run, 'user_reviews': reviews,
                'job_requests': requests, 'archive_integrity_verified': True, 'currentness': 'not_checked'}
        except (OSError, KeyError, TypeError, ValueError) as error:
            raise ValueError('Observation archive is missing, invalid, or unbound: ' + str(error)) from error

    def read_image_bytes(self, snapshot_id: str, expected_run_id: str, artwork_id: str) -> bytes:
        """Only an archived, manifest-bound ID; never a source path or external URL."""
        archive = self.read_archive(snapshot_id, expected_run_id)
        _simple_id(artwork_id, name='artwork_id', maximum=120)
        item = archive['manifest']['images'].get(artwork_id)
        if item is None:
            raise ValueError('Artwork ID is not a selected original in this frozen observation.')
        data = self._archive_file(self._folder(snapshot_id), item['snapshot_image'], self.limits.max_image_bytes)
        if len(data) != item['bytes'] or _digest(data) != item['sha256']:
            raise ValueError('Manifest-bound original image bytes changed after archive verification.')
        return data

    def validate_still_current(self, snapshot_id: str, context: dict) -> dict:
        """Check archive integrity and current protected inputs, not queue clocks."""
        try:
            manifest = self.read_archive(snapshot_id, context['run_id'])['manifest']
            state, images, _ = self._collect(json.loads(_json_bytes(context)), self._selection(manifest['roles']))
            unchanged = (manifest['input_identity'] == _input_identity(state) if 'input_identity' in manifest
                         else manifest['source_state_sha256'] == _digest(_json_bytes(state)))
            # A pre-identity archive keeps its original strict meaning. It is
            # never rewritten/upgraded merely because a caller wants to use it.
            if (not unchanged
                    or manifest['run_id'] != state['run']['id'] or manifest['revision'] != state['run']['revision']):
                raise ValueError('Original context, conditions, revision, or natural-language feedback changed.')
            for artwork_id, current in images.items():
                frozen = {key: value for key, value in manifest['images'][artwork_id].items() if key != 'snapshot_image'}
                if frozen != current:
                    raise ValueError('Original image/path/stat changed: ' + artwork_id)
            return manifest
        except (OSError, KeyError, TypeError, ValueError) as error:
            if isinstance(error, StaleObservationError):
                raise
            raise StaleObservationError('Observation is stale or incomplete: ' + str(error)) from error
