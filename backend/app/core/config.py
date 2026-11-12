from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from app.ml.participants import FROZEN_TEST_PARTICIPANTS, TRUSTED_DEVELOPMENT_PARTICIPANTS

BACKEND_DIR = Path(__file__).resolve().parents[2]
PROJECT_ROOT = BACKEND_DIR.parent


class Settings(BaseSettings):
    """Application configuration, read from environment variables and backend/.env."""

    model_config = SettingsConfigDict(
        env_prefix="NEUROSENSE_",
        env_file=BACKEND_DIR / ".env",
        env_file_encoding="utf-8-sig",
        extra="ignore",
    )

    environment: Literal["development", "test", "production"] = "development"
    log_level: str = "INFO"
    random_seed: int = 42

    # API
    api_prefix: str = "/api"
    cors_origins: list[str] = Field(default_factory=lambda: ["http://localhost:5173", "http://127.0.0.1:5173"])

    # Paths (relative paths are resolved against the project root)
    dataset_dir: Path = Path("Dataset/DEAP")
    output_dir: Path = Path("outputs")
    research_results_dir: Path = Path("docs/results")  # saved, version-controlled research results (read-only)
    # FACED (NEMAR nm000112 v1.1.4) BIDS folder; a separate dataset with its own pipeline and models.
    faced_dataset_dir: Path = Path("Dataset/FACED/v1.1.4")
    # Optional: serve the built frontend (frontend/dist) from the API server, for a single-container deployment.
    frontend_dist_dir: Path | None = None

    # Database
    database_url: str = "postgresql+psycopg://neurosense:neurosense@localhost:5432/neurosense"

    # Auth
    jwt_secret_key: SecretStr
    jwt_algorithm: str = "HS256"
    access_token_expire_minutes: int = 60

    # Signal / preprocessing
    sampling_rate: int = 128
    baseline_seconds: float = 3.0
    segment_seconds: float = 4.0
    n_eeg_channels: int = 32
    artifact_clip_sigma: float = 6.0
    max_clipped_fraction: float = 0.05

    # Labels
    label_strategy: Literal["valence_binary", "arousal_binary", "quadrant"] = "valence_binary"
    valence_threshold: float = 5.0
    arousal_threshold: float = 5.0

    # Participant-aware split (historical, pre-protocol baseline trainer only)
    test_participants: int = 3
    validation_participants: int = 2

    # Evaluation Protocol v1.1: development participants used by protocol-aware candidate training.
    # Defaults to the 12 trusted participants; the frozen final-test group can never be added here.
    development_participants: list[str] = Field(default_factory=lambda: list(TRUSTED_DEVELOPMENT_PARTICIPANTS))

    @field_validator("development_participants")
    @classmethod
    def _no_frozen(cls, value: list[str]) -> list[str]:
        frozen = sorted(p for p in value if p.lower() in FROZEN_TEST_PARTICIPANTS)
        if frozen:
            raise ValueError("development_participants must not include frozen final-test participants")
        return sorted(set(value))

    @field_validator("dataset_dir", "output_dir", "faced_dataset_dir", "research_results_dir")
    @classmethod
    def _resolve(cls, value: Path) -> Path:
        return value if value.is_absolute() else (PROJECT_ROOT / value).resolve()

    @property
    def faced_output_dir(self) -> Path:
        return self.output_dir / "faced"

    @property
    def model_dir(self) -> Path:
        return self.output_dir / "models"

    @property
    def metrics_dir(self) -> Path:
        return self.output_dir / "metrics"

    @property
    def graphs_dir(self) -> Path:
        return self.output_dir / "graphs"

    @property
    def predictions_dir(self) -> Path:
        return self.output_dir / "predictions"

    @property
    def cache_dir(self) -> Path:
        return self.output_dir / "cache"

    @property
    def log_dir(self) -> Path:
        return self.output_dir / "logs"

    def ensure_output_dirs(self) -> None:
        for path in (self.model_dir, self.metrics_dir, self.graphs_dir, self.predictions_dir, self.cache_dir, self.log_dir):
            path.mkdir(parents=True, exist_ok=True)


@lru_cache
def get_settings() -> Settings:
    return Settings()  # type: ignore[call-arg]
