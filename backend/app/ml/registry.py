"""File-based model registry: outputs/models/<version>/{model file, metadata.json} + active_model.json.

Supported frameworks (metadata["framework"], default "sklearn" for Review 2 artifacts):
- sklearn: model.joblib (joblib/pickle)
- pytorch: model.pt (state_dict + config, loaded with torch.load(weights_only=True))

Pickle-based files execute code when loaded, so only artifacts written by this registry are loaded,
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


MODEL_FILES = {"sklearn": "model.joblib", "pytorch": "model.pt"}


class ModelRegistry:
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

    @staticmethod
    def framework_of(metadata: dict[str, Any]) -> str:
        framework = metadata.get("framework", "sklearn")
        if framework not in MODEL_FILES:
            raise ValueError(f"Unsupported framework: {framework}")
        return framework

    def version_dir(self, version: str) -> Path:
        return self.base_dir / version

    def save(self, version: str, model: Any, metadata: dict[str, Any], activate: bool = True) -> Path:
        framework = self.framework_of(metadata)
        directory = self.version_dir(version)
        directory.mkdir(parents=True, exist_ok=True)
        model_path = directory / MODEL_FILES[framework]
        if model_path.exists() or (directory / self.METADATA_FILE).exists():
            raise FileExistsError(f"Model version {version} already exists")
        if framework == "sklearn":
            joblib.dump(model, model_path, compress=3)
        else:
            model.save(model_path)
        metadata = {**metadata, "framework": framework, "version": version, "model_path": str(model_path),
                    "sha256": self._sha256(model_path)}
        (directory / self.METADATA_FILE).write_text(json.dumps(metadata, indent=2, default=str), encoding="utf-8")
        if activate:
            self.activate(version)
        return directory

    def activate(self, version: str) -> None:
        if not (self.version_dir(version) / self.METADATA_FILE).exists():
            raise ModelNotAvailableError(f"Model version {version} not found")
        (self.base_dir / self.ACTIVE_FILE).write_text(json.dumps({"version": version}), encoding="utf-8")

    def active_version(self) -> str | None:
        path = self.base_dir / self.ACTIVE_FILE
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8")).get("version")

    def metadata(self, version: str) -> dict[str, Any]:
        path = self.version_dir(version) / self.METADATA_FILE
        if not path.exists():
            raise ModelNotAvailableError(f"Model version {version} not found")
        return json.loads(path.read_text(encoding="utf-8"))

    def list_versions(self) -> list[dict[str, Any]]:
        items = []
        for meta_path in sorted(self.base_dir.glob(f"*/{self.METADATA_FILE}")):
            items.append(json.loads(meta_path.read_text(encoding="utf-8")))
        return sorted(items, key=lambda m: m.get("trained_at", ""), reverse=True)

    def latest(self, algorithm: str) -> dict[str, Any] | None:
        return next((m for m in self.list_versions() if m.get("algorithm") == algorithm), None)

    def load(self, version: str | None = None) -> tuple[Any, dict[str, Any]]:
        version = version or self.active_version()
        if version is None:
            raise ModelNotAvailableError("No trained model available. Train a model first.")
        meta = self.metadata(version)
        framework = self.framework_of(meta)
        model_path = self.version_dir(version) / MODEL_FILES[framework]
        if not model_path.exists():
            raise ModelNotAvailableError(f"Model file missing for version {version}")
        if self._sha256(model_path) != meta.get("sha256"):
            raise ModelNotAvailableError(f"Model file for {version} failed its integrity check")
        if framework == "pytorch":
            from app.ml.deep.classifier import CNNClassifier

            return CNNClassifier.load(model_path), meta
        return joblib.load(model_path), meta
