from __future__ import annotations

import json
from pathlib import Path

from .models import PhoenixTracePullResult


class RCAPullStorage:
    def __init__(self, save_dir: Path) -> None:
        self.save_dir = save_dir
        self.save_dir.mkdir(parents=True, exist_ok=True)

    def build_snapshot_path(self, result: PhoenixTracePullResult) -> Path:
        timestamp = result.fetched_at.strftime("%Y%m%dT%H%M%SZ")
        project = result.project_name.replace("/", "_").replace(" ", "_")
        return self.save_dir / f"phoenix-pull-{project}-{timestamp}.json"

    def save(self, result: PhoenixTracePullResult) -> Path:
        path = self.build_snapshot_path(result)
        payload = result.model_dump(mode="json")
        with path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
        return path
