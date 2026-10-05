from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class ModelRegistry:
    """Persist model version metadata for the best model."""

    def __init__(self, base_dir: str | None = None) -> None:
        self.base_dir = Path(base_dir or "outputs/models")
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def save_metadata(self, model_name: str, metrics: dict[str, Any]) -> str:
        data = {"model_name": model_name, "metrics": metrics}
        path = self.base_dir / "model_registry.json"
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        return str(path)
