from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, EmailStr, Field, field_validator, model_validator

MAX_SAMPLES_PER_CHANNEL = 128 * 120


# ---------- auth ----------
class RegisterRequest(BaseModel):
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
    status: Literal["ok", "degraded"]
    version: str
    environment: str
    database: dict[str, Any]
    model: dict[str, Any]
    dataset_available: bool


# ---------- models / metrics ----------
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
    disclaimer: str | None = None


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
    dataset: dict[str, Any] | None = None
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
class DemoTrialsOut(BaseModel):
    model_version: str | None
    held_out_participants: list[str]
    trials_per_participant: int
    note: str
