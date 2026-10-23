from __future__ import annotations

import re
from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, EmailStr, Field, field_validator, model_validator

MAX_SAMPLES_PER_CHANNEL = 128 * 120


# ---------- auth ----------
class RegisterRequest(BaseModel):
    # Unknown fields (for example "is_admin" or "role") are rejected, so a signup payload cannot ask for a role.
    model_config = ConfigDict(extra="forbid")

    email: EmailStr
    password: str = Field(min_length=8, max_length=72, description="8-72 characters (bcrypt limit)")
    full_name: str | None = Field(default=None, max_length=200)

    @field_validator("password")
    @classmethod
    def _bcrypt_limit(cls, value: str) -> str:
        if len(value.encode("utf-8")) > 72:
            raise ValueError("Password must be at most 72 bytes")
        return value


class UserOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: int
    email: str
    full_name: str | None
    is_admin: bool
    created_at: datetime


class TokenOut(BaseModel):
    access_token: str
    token_type: Literal["bearer"] = "bearer"
    expires_in: int
    user: UserOut


# ---------- health ----------
class HealthOut(BaseModel):
    """Public liveness check: no configuration, model or dataset details."""
    status: Literal["ok", "degraded"]
    version: str


class DiagnosticsOut(BaseModel):
    """Administrator-only system diagnostics. Never contains connection strings, credentials or secrets."""
    status: Literal["ok", "degraded"]
    version: str
    environment: str
    database: dict[str, Any]
    model: dict[str, Any]
    dataset_available: bool


class ParticipantStructure(BaseModel):
    participant_id: str
    readable: bool
    trials: int | None = None
    channels: int | None = None
    samples: int | None = None


class DatasetSummaryOut(BaseModel):
    """Public, aggregate structure of the working dataset: no paths, file names, sizes or rating statistics."""
    sampling_rate_hz: int
    baseline_seconds: float
    participants_found: int
    participants_valid: int
    total_trials: int
    eeg_channels_used: int
    eeg_channel_names: list[str]
    participants: list[ParticipantStructure]
    authorized_participants: list[str] = Field(
        default=[], description="Configured development participants this server may use (frozen test participants never)")


class ModelSyncOut(BaseModel):
    synced: int
    active_version: str | None


# Text that looks like a credential is refused in an activation reason (the reason is stored in the audit log).
_CREDENTIAL_PATTERNS = (
    re.compile(r"\b(?:password|passwd|pwd|passphrase|secret|api[_-]?key|access[_-]?key|private[_-]?key|token|"
               r"client[_-]?secret|credentials?)\b\s*[:=]", re.IGNORECASE),  # password=..., token: ...
    re.compile(r"\bbearer\s+[A-Za-z0-9._~+/=-]{8,}", re.IGNORECASE),  # Authorization header value
    re.compile(r"\beyJ[A-Za-z0-9_-]{5,}\.[A-Za-z0-9_-]{5,}\.[A-Za-z0-9_-]*"),  # JSON Web Token
    re.compile(r"[a-z][a-z0-9+.-]*://[^\s/:@]+:[^\s/@]+@", re.IGNORECASE),  # URL with user:password@
    re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
    re.compile(r"\b(?:sk|pk|ghp|gho|ghs|xox[abprs])[-_][A-Za-z0-9_-]{16,}"),  # common API key formats
    re.compile(r"\bAKIA[0-9A-Z]{16}\b"),  # cloud access key ID
)


def looks_like_credential(text: str) -> bool:
    return any(p.search(text) for p in _CREDENTIAL_PATTERNS)


class ModelActivateRequest(BaseModel):
    confirm_version: str = Field(min_length=1, max_length=200,
                                 description="Must repeat the version in the URL, as a deliberate confirmation")
    reason: str = Field(description="Why this model is being activated (10-500 characters, stored in the audit log; "
                                    "never include passwords, tokens or other secrets)")

    @field_validator("reason")
    @classmethod
    def _reason(cls, value: str) -> str:
        value = value.strip()
        if not 10 <= len(value) <= 500:
            raise ValueError("Reason must be 10-500 characters after trimming spaces")
        if looks_like_credential(value):
            raise ValueError("Reason appears to contain a credential or secret; remove it and try again")
        return value


class ModelActivationOut(BaseModel):
    version: str
    previous_version: str | None
    changed: bool = Field(description="False when the requested version was already active (nothing was written)")
    active_version: str


# ---------- models / metrics ----------
class TrainingProvenance(BaseModel):
    """Participants behind the stored estimator, from recorded metadata only (never reconstructed)."""
    recorded: bool
    kind: str = Field(description="fixed_split | protocol_all_development | unknown")
    final_fit: str | None = None
    fit_participants: list[str] = []
    selection_participants: list[str] = []
    held_out_participants: list[str] = []
    fit_segments: int | None = None
    fit_trials: int | None = None
    note: str


class ModelSummary(BaseModel):
    version: str
    name: str
    algorithm: str
    label_strategy: str
    class_labels: list[str]
    feature_count: int
    trained_at: datetime
    is_active: bool
    model_type: str | None = None
    framework: str | None = None
    test_metrics: dict[str, float | None] | None = None
    evaluation: dict[str, Any] | None = Field(
        default=None, description="Protocol candidates only: exploratory leave-one-participant-out summary")
    provenance: TrainingProvenance | None = None


class ModelInfoOut(BaseModel):
    available: bool
    message: str | None = None
    version: str | None = None
    name: str | None = None
    algorithm: str | None = None
    algorithm_description: str | None = None
    model_type: str | None = None
    framework: str | None = None
    framework_version: str | None = None
    input_requirements: dict[str, Any] | None = None
    architecture: dict[str, Any] | None = None
    trained_at: datetime | None = None
    label: dict[str, Any] | None = None
    class_labels: list[str] | None = None
    feature_schema: dict[str, Any] | None = None
    preprocessing: dict[str, Any] | None = None
    split: dict[str, Any] | None = None
    test_metrics: dict[str, Any] | None = None
    provenance: TrainingProvenance | None = None
    disclaimer: str | None = None


class FeatureImportanceItem(BaseModel):
    rank: int
    name: str
    importance: float
    channel: str
    family: str
    band: str | None = None


class ImportanceGroup(BaseModel):
    group: str
    label: str
    importance: float


class GlobalImportanceOut(BaseModel):
    available: bool
    reason: str | None = None
    scope: Literal["global"] = "global"
    model_version: str | None = None
    algorithm: str | None = None
    method: str | None = None
    interpretation: str
    limitations: list[str]
    n_features: int | None = None
    n_trees: int | None = None
    uniform_share: float | None = None
    top: list[FeatureImportanceItem] = Field(default_factory=list)
    by_family: list[ImportanceGroup] = Field(default_factory=list)
    by_band: list[ImportanceGroup] = Field(default_factory=list)


class MetricsDatasetOut(BaseModel):
    """Public dataset facts behind the metrics. Allow-listed: never the dataset location or file names."""
    participant_count: int | None = None
    segments: int | None = None
    rejected_segments: int | None = None
    class_balance: dict[str, int | float] | None = None
    fingerprint: str | None = None


class MetricsOut(BaseModel):
    available: bool
    message: str | None = None
    model_version: str | None = None
    label: dict[str, Any] | None = None
    split: dict[str, Any] | None = None
    validation: dict[str, Any] | None = None
    test: dict[str, Any] | None = None
    test_reference_majority: dict[str, Any] | None = None
    cross_validation: dict[str, Any] | None = None
    evaluation: dict[str, Any] | None = Field(
        default=None, description="Protocol-trained models: exploratory leave-one-participant-out results (per participant)")
    protocol: dict[str, Any] | None = None
    dataset: MetricsDatasetOut | None = None
    graphs: list[str] = Field(default_factory=list)


# ---------- training ----------
class TrainRequest(BaseModel):
    run_cross_validation: bool = True


class TrainingStatusOut(BaseModel):
    state: Literal["idle", "running", "succeeded", "failed"]
    stage: str | None = None
    progress: float = 0.0
    started_at: datetime | None = None
    finished_at: datetime | None = None
    model_version: str | None = None
    error: str | None = None
    started_by: int | None = None


# ---------- prediction ----------
class EEGInput(BaseModel):
    signal: list[list[float]] = Field(description="EEG as [channels][samples]; 32 channels in DEAP (Geneva) order, microvolts")
    sampling_rate: int = Field(default=128, description="Must match the model's sampling rate (128 Hz)")
    includes_baseline: bool = Field(default=False, description="True if the first 3 s are a pre-stimulus baseline to drop")
    synthetic_test_signal: bool = Field(
        default=False, description="True for generated test signals (not recorded EEG); stored so the result is labelled")

    @field_validator("signal")
    @classmethod
    def _shape(cls, value: list[list[float]]) -> list[list[float]]:
        if not value:
            raise ValueError("signal must contain at least one channel")
        lengths = {len(ch) for ch in value}
        if len(lengths) != 1:
            raise ValueError("all channels must have the same number of samples")
        if lengths.pop() > MAX_SAMPLES_PER_CHANNEL:
            raise ValueError(f"at most {MAX_SAMPLES_PER_CHANNEL} samples per channel")
        return value


class DEAPReference(BaseModel):
    participant_id: str = Field(pattern=r"^s\d{2}$", examples=["s01"])
    trial_index: int = Field(ge=0, le=39)


class PredictionRequest(BaseModel):
    eeg: EEGInput | None = None
    deap_reference: DEAPReference | None = None
    session_id: int | None = None

    @model_validator(mode="after")
    def _exactly_one(self) -> PredictionRequest:
        if (self.eeg is None) == (self.deap_reference is None):
            raise ValueError("Provide exactly one of 'eeg' or 'deap_reference'")
        return self


class ModelRef(BaseModel):
    version: str
    algorithm: str
    label_strategy: str


class PredictionOut(BaseModel):
    prediction_id: int
    session_id: int | None
    created_at: datetime
    prediction: str
    confidence: float
    probabilities: dict[str, float]
    segments_used: int
    segments_rejected: int
    input_source: str
    source_reference: dict[str, Any] | None = None
    model: ModelRef | None
    disclaimer: str


class PredictionListOut(BaseModel):
    total: int
    items: list[PredictionOut]


# ---------- sessions / reports ----------
class SessionCreate(BaseModel):
    name: str = Field(min_length=1, max_length=200)
    description: str | None = Field(default=None, max_length=2000)


class SessionOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: int
    name: str
    description: str | None
    created_at: datetime
    prediction_count: int = 0


class ReportOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: int
    report_type: str
    title: str
    content: dict[str, Any]
    model_id: int | None
    user_id: int | None
    created_at: datetime


# ---------- dataset ----------
class DemoParticipant(BaseModel):
    participant_id: str
    trial_indices: list[int] = Field(description="Zero-based DEAP trial indices readable on this server")
    model_role: str = Field(description="Recorded role for the active model: fit | selection | held_out | not_used | unknown")


class DemoUnavailable(BaseModel):
    participant_id: str
    reason: str


class DemoTrialsOut(BaseModel):
    model_version: str | None
    held_out_participants: list[str] = Field(
        description="Available participants recorded as held out from the active model's fit and selection")
    trials_per_participant: int | None
    participants: list[DemoParticipant] = []
    unavailable: list[DemoUnavailable] = []
    dataset_available: bool = False
    provenance: TrainingProvenance | None = None
    note: str
