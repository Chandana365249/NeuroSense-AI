"""File-based model registry: outputs/models/<version>/{model.joblib, metadata.json} + active_model.json.

joblib/pickle files execute code when loaded, so only artifacts written by this registry are loaded,
and each load verifies the SHA-256 recorded at save time.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import joblib


class ModelNotAvailableError(Exception):
    """No trained model has been registered/activated yet."""


class ModelRegistry:
    MODEL_FILE = "model.joblib"
    METADATA_FILE = "metadata.json"
    ACTIVE_FILE = "active_model.json"

    def __init__(self, base_dir: Path) -> None:
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                digest.update(chunk)
        return digest.hexdigest()

    def save(self, version: str, model: Any, metadata: dict[str, Any], activate: bool = True) -> Path:
        directory = self.base_dir / version
        directory.mkdir(parents=True, exist_ok=False)
        model_path = directory / self.MODEL_FILE
        joblib.dump(model, model_path, compress=3)
        metadata = {**metadata, "version": version, "model_path": str(model_path), "sha256": self._sha256(model_path)}
        (directory / self.METADATA_FILE).write_text(json.dumps(metadata, indent=2, default=str), encoding="utf-8")
        if activate:
            self.activate(version)
        return directory

    def activate(self, version: str) -> None:
        if not (self.base_dir / version / self.METADATA_FILE).exists():
            raise ModelNotAvailableError(f"Model version {version} not found")
        (self.base_dir / self.ACTIVE_FILE).write_text(json.dumps({"version": version}), encoding="utf-8")

    def active_version(self) -> str | None:
        path = self.base_dir / self.ACTIVE_FILE
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8")).get("version")

    def metadata(self, version: str) -> dict[str, Any]:
        path = self.base_dir / version / self.METADATA_FILE
        if not path.exists():
            raise ModelNotAvailableError(f"Model version {version} not found")
        return json.loads(path.read_text(encoding="utf-8"))

    def list_versions(self) -> list[dict[str, Any]]:
        items = []
        for meta_path in sorted(self.base_dir.glob(f"*/{self.METADATA_FILE}")):
            items.append(json.loads(meta_path.read_text(encoding="utf-8")))
        return sorted(items, key=lambda m: m.get("trained_at", ""), reverse=True)

    def load(self, version: str | None = None) -> tuple[Any, dict[str, Any]]:
        version = version or self.active_version()
        if version is None:
            raise ModelNotAvailableError("No trained model available. Train a model first.")
        meta = self.metadata(version)
        model_path = self.base_dir / version / self.MODEL_FILE
        if not model_path.exists():
            raise ModelNotAvailableError(f"Model file missing for version {version}")
        if self._sha256(model_path) != meta.get("sha256"):
            raise ModelNotAvailableError(f"Model file for {version} failed its integrity check")
        return joblib.load(model_path), meta
