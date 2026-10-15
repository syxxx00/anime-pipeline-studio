from __future__ import annotations

import json
import uuid
from datetime import UTC, datetime
from pathlib import Path

from .config import PROJECT_ROOT


TRIGGERS = {"simple_identity", "environment", "object_relation"}
PIPELINES = {"newbie", "anima_newbie", "illustrious_newbie"}


class RoutingGuidanceStore:
    def __init__(self, path: Path | None = None) -> None:
        self.path = path or PROJECT_ROOT / "data" / "routing_guidance.json"

    def list(self) -> list[dict]:
        if not self.path.exists():
            return []
        try:
            values = json.loads(self.path.read_text(encoding="utf-8"))
            return values if isinstance(values, list) else []
        except (OSError, ValueError, json.JSONDecodeError):
            return []

    def publish(self, *, experiment_id: str, trigger: str, pipeline: str, note: str, evidence: dict) -> dict:
        if trigger not in TRIGGERS:
            raise ValueError("未知路由场景")
        if pipeline not in PIPELINES:
            raise ValueError("未知生产路线")
        values = self.list()
        for item in values:
            if item.get("trigger") == trigger and item.get("active"):
                item["active"] = False
        item = {
            "id": uuid.uuid4().hex[:12], "active": True,
            "created_at": datetime.now(UTC).isoformat(),
            "experiment_id": experiment_id, "trigger": trigger,
            "pipeline": pipeline, "note": note.strip(), "evidence": evidence,
        }
        values.append(item)
        self._save(values)
        return item

    def deactivate(self, guidance_id: str) -> dict:
        values = self.list()
        item = next((value for value in values if value.get("id") == guidance_id), None)
        if item is None:
            raise KeyError(guidance_id)
        item["active"] = False
        self._save(values)
        return item

    def active_map(self) -> dict[str, dict]:
        return {item["trigger"]: item for item in self.list() if item.get("active") and item.get("trigger") in TRIGGERS}

    def _save(self, values: list[dict]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(".tmp")
        temporary.write_text(json.dumps(values, ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(self.path)
