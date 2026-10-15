from __future__ import annotations

import json
import uuid
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Iterable

from .config import PROJECT_ROOT
from .reviews import analyze_review

if TYPE_CHECKING:
    from .research.experiments import Experiment
    from .jobs import Job


def now_iso() -> str:
    return datetime.now(UTC).isoformat()


@dataclass(slots=True)
class Artwork:
    id: str
    origin: str
    job_id: str
    image_index: int
    image_url: str
    file_path: str
    created_at: str
    instruction: str = ""
    experiment_id: str | None = None
    experiment_item_id: str | None = None
    parent_artwork_id: str | None = None
    operation: str = "generate"
    mask_file: str | None = None
    checks: list[dict] = field(default_factory=list)
    reviews: list[dict] = field(default_factory=list)
    source_stage: dict | None = None


class ArtworkStore:
    """Durable user-facing image identity layered over jobs and experiments."""

    def __init__(self, path: Path | None = None) -> None:
        self.path = path or PROJECT_ROOT / "data" / "artworks.json"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.items: dict[str, Artwork] = {}
        self._load()

    @staticmethod
    def id_for(job_id: str, image_index: int) -> str:
        return f"{job_id}-{image_index + 1:02d}"

    def sync_jobs(self, jobs: Iterable["Job"]) -> None:
        changed = False
        for job in jobs:
            request = job.request or {}
            output_group = str(request.get("output_group", ""))
            origin = "experiment" if output_group.replace("\\", "/").startswith("experiments/") else "creation"
            instruction = str((request.get("semantic_spec") or {}).get("source_sentence") or request.get("prompt") or "")
            for index, image_url in enumerate(job.images):
                artwork_id = self.id_for(job.id, index)
                file_path = job.files[index] if index < len(job.files) else ""
                existing = self.items.get(artwork_id)
                if existing is None:
                    self.items[artwork_id] = Artwork(
                        id=artwork_id, origin=origin, job_id=job.id, image_index=index,
                        image_url=image_url, file_path=file_path, created_at=job.created_at,
                        instruction=instruction, checks=list(job.checks),
                        parent_artwork_id=request.get("parent_artwork_id"),
                        operation=str(request.get("operation", "generate")),
                        mask_file=request.get("mask_file"),
                    )
                    changed = True
                else:
                    snapshot = (existing.image_url, existing.file_path, existing.checks)
                    existing.image_url = image_url
                    existing.file_path = file_path
                    existing.checks = list(job.checks)
                    if snapshot != (existing.image_url, existing.file_path, existing.checks):
                        changed = True
        if changed:
            self._save()

    def sync_experiments(self, experiments: Iterable["Experiment"]) -> None:
        changed = False
        for experiment in experiments:
            for item in experiment.items:
                if not item.job_id:
                    continue
                for index, image_url in enumerate(item.images):
                    artwork_id = self.id_for(item.job_id, index)
                    artwork = self.items.get(artwork_id)
                    if artwork is None:
                        artwork = Artwork(
                            id=artwork_id, origin="experiment", job_id=item.job_id, image_index=index,
                            image_url=image_url, file_path=item.files[index] if index < len(item.files) else "",
                            created_at=experiment.created_at,
                            instruction=str((item.request.get("semantic_spec") or {}).get("source_sentence") or item.request.get("prompt") or ""),
                            checks=list(item.checks),
                        )
                        self.items[artwork_id] = artwork
                        changed = True
                    if artwork.origin != "experiment" or artwork.experiment_id != experiment.id or artwork.experiment_item_id != item.id:
                        artwork.origin = "experiment"
                        artwork.experiment_id = experiment.id
                        artwork.experiment_item_id = item.id
                        changed = True
                    known = {review.get("id") for review in artwork.reviews}
                    for review in item.reviews:
                        if review.get("image_index", 0) == index and review.get("id") not in known:
                            artwork.reviews.append({**review, "legacy_experiment_review": True})
                            known.add(review.get("id"))
                            changed = True
        if changed:
            self._save()

    def list(self, *, origin: str | None = None, limit: int = 100, offset: int = 0, query: str = "") -> list[dict]:
        items = self.items.values()
        if origin:
            items = (item for item in items if item.origin == origin)
        needle = query.strip().casefold()
        if needle:
            items = (item for item in items if needle in " ".join([
                item.id, item.job_id, item.file_path, item.instruction, item.operation,
                *[str(review.get("text", "")) for review in item.reviews],
            ]).casefold())
        ordered = sorted(items, key=lambda item: (item.created_at, item.id), reverse=True)
        start = max(0, offset)
        return [asdict(item) for item in ordered[start:start + max(0, limit)]]

    def get(self, artwork_id: str) -> Artwork | None:
        return self.items.get(artwork_id)

    def lineage(self, artwork: Artwork) -> dict:
        ancestors: list[Artwork] = []
        seen = {artwork.id}
        current = artwork
        while current.parent_artwork_id:
            parent = self.items.get(current.parent_artwork_id)
            if parent is None or parent.id in seen:
                break
            ancestors.append(parent)
            seen.add(parent.id)
            current = parent
        ancestors.reverse()
        children = sorted(
            (item for item in self.items.values() if item.parent_artwork_id == artwork.id),
            key=lambda item: item.created_at,
        )
        return {
            "current": self.serialize(artwork),
            "ancestors": [self.serialize(item) for item in ancestors],
            "children": [self.serialize(item) for item in children],
            "root_id": ancestors[0].id if ancestors else artwork.id,
        }

    def add_review(self, artwork: Artwork, text: str, *, origin: str = 'user') -> dict:
        if origin not in ('user', 'assistant_validation'):
            raise ValueError('Artwork review origin must be user or assistant_validation')
        # Keep the legacy user record shape/normalization. An attributed
        # assistant validation is an original report, never a user judgement.
        review = {"id": uuid.uuid4().hex[:12], "text": text if origin == 'assistant_validation' else text.strip(),
            "created_at": now_iso(), "analysis": analyze_review(text)}
        if origin == 'assistant_validation':
            review['origin'] = origin
        artwork.reviews.append(review)
        self._save()
        return review

    def add_derivative(self, *, artwork_id: str, parent: Artwork, image_url: str, file_path: str, operation: str, instruction: str, checks: list[dict] | None = None) -> Artwork:
        artwork = Artwork(
            id=artwork_id, origin="creation", job_id=artwork_id, image_index=0,
            image_url=image_url, file_path=file_path, created_at=now_iso(), instruction=instruction,
            parent_artwork_id=parent.id, operation=operation, checks=checks or [],
        )
        self.items[artwork.id] = artwork
        self._save()
        return artwork

    def add_retained_stage(self, job, record: dict, image_url: str) -> Artwork:
        artwork_id = f"stage-{job.id}-{record['id']}"
        if artwork_id in self.items:
            return self.items[artwork_id]
        artwork = Artwork(
            id=artwork_id, origin='creation', job_id=job.id, image_index=0,
            image_url=image_url, file_path=record['file_path'], created_at=now_iso(),
            instruction=str((job.request.get('semantic_spec') or {}).get('source_sentence') or job.request.get('prompt') or ''),
            operation='retain_stage', source_stage={
                key: record.get(key) for key in ('id', 'stage_id', 'model', 'evidence', 'warning', 'sha256')
            },
        )
        # A source image is not a derivative of the final picture. Do not invent
        # that parent relation, copy its reviews/checks, or mark the experiment won.
        self.items[artwork.id] = artwork
        self._save()
        return artwork

    def delete_review(self, artwork: Artwork, review_id: str) -> None:
        before = len(artwork.reviews)
        artwork.reviews = [review for review in artwork.reviews if review.get("id") != review_id]
        if len(artwork.reviews) == before:
            raise KeyError(review_id)
        self._save()

    @staticmethod
    def serialize(artwork: Artwork) -> dict:
        return asdict(artwork)

    def _save(self) -> None:
        payload = {key: asdict(value) for key, value in self.items.items()}
        temporary = self.path.with_suffix(".tmp")
        temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(self.path)

    def _load(self) -> None:
        if not self.path.exists():
            return
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
            self.items = {key: Artwork(**value) for key, value in payload.items()}
        except (OSError, json.JSONDecodeError, TypeError):
            self.items = {}
