"""ORM entities.

users 1-n sessions (analysis sessions grouping predictions; unrelated to auth tokens)
users 1-n predictions, sessions 1-n predictions, models 1-n predictions
models 1-n metrics, models 1-n reports, users 1-n reports, users 1-n audit_logs
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from sqlalchemy import JSON, Boolean, Float, ForeignKey, Integer, String, Text, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database.base import Base, UTCDateTime, utcnow


class User(Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(primary_key=True)
    email: Mapped[str] = mapped_column(String(320), unique=True, index=True)
    full_name: Mapped[str | None] = mapped_column(String(200))
    hashed_password: Mapped[str] = mapped_column(String(200))
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    is_admin: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime, default=utcnow)
    last_login_at: Mapped[datetime | None] = mapped_column(UTCDateTime)

    sessions: Mapped[list[AnalysisSession]] = relationship(back_populates="user", cascade="all, delete-orphan")
    predictions: Mapped[list[Prediction]] = relationship(back_populates="user", cascade="all, delete-orphan")


class AnalysisSession(Base):
    __tablename__ = "sessions"

    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    name: Mapped[str] = mapped_column(String(200))
    description: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime, default=utcnow)

    user: Mapped[User] = relationship(back_populates="sessions")
    predictions: Mapped[list[Prediction]] = relationship(back_populates="session")


class MLModel(Base):
    __tablename__ = "models"

    id: Mapped[int] = mapped_column(primary_key=True)
    version: Mapped[str] = mapped_column(String(200), unique=True, index=True)
    name: Mapped[str] = mapped_column(String(200))
    algorithm: Mapped[str] = mapped_column(String(100))
    label_strategy: Mapped[str] = mapped_column(String(50))
    class_labels: Mapped[list[str]] = mapped_column(JSON)
    feature_count: Mapped[int] = mapped_column(Integer)
    artifact_path: Mapped[str] = mapped_column(String(500))
    is_active: Mapped[bool] = mapped_column(Boolean, default=False, index=True)
    trained_at: Mapped[datetime] = mapped_column(UTCDateTime)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime, default=utcnow)

    metrics: Mapped[list[Metric]] = relationship(back_populates="model", cascade="all, delete-orphan")


class Metric(Base):
    __tablename__ = "metrics"
    __table_args__ = (UniqueConstraint("model_id", "split", "level", "name", name="uq_metrics_model_split_level_name"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    model_id: Mapped[int] = mapped_column(ForeignKey("models.id", ondelete="CASCADE"), index=True)
    split: Mapped[str] = mapped_column(String(30))  # validation | test | test_reference | cross_validation
    level: Mapped[str] = mapped_column(String(20))  # segment | trial
    name: Mapped[str] = mapped_column(String(50))
    value: Mapped[float | None] = mapped_column(Float)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime, default=utcnow)

    model: Mapped[MLModel] = relationship(back_populates="metrics")


class Prediction(Base):
    __tablename__ = "predictions"

    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    session_id: Mapped[int | None] = mapped_column(ForeignKey("sessions.id", ondelete="SET NULL"), index=True)
    model_id: Mapped[int | None] = mapped_column(ForeignKey("models.id", ondelete="SET NULL"), index=True)
    predicted_label: Mapped[str] = mapped_column(String(100))
    confidence: Mapped[float] = mapped_column(Float)
    probabilities: Mapped[dict[str, float]] = mapped_column(JSON)
    segments_used: Mapped[int] = mapped_column(Integer)
    segments_rejected: Mapped[int] = mapped_column(Integer, default=0)
    input_source: Mapped[str] = mapped_column(String(30))  # upload | deap_reference
    source_reference: Mapped[dict[str, Any] | None] = mapped_column(JSON)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime, default=utcnow, index=True)

    user: Mapped[User] = relationship(back_populates="predictions")
    session: Mapped[AnalysisSession | None] = relationship(back_populates="predictions")
    model: Mapped[MLModel | None] = relationship()


class Report(Base):
    __tablename__ = "reports"

    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[int | None] = mapped_column(ForeignKey("users.id", ondelete="SET NULL"), index=True)
    model_id: Mapped[int | None] = mapped_column(ForeignKey("models.id", ondelete="SET NULL"), index=True)
    report_type: Mapped[str] = mapped_column(String(50))
    title: Mapped[str] = mapped_column(String(300))
    content: Mapped[dict[str, Any]] = mapped_column(JSON)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime, default=utcnow, index=True)


class AuditLog(Base):
    __tablename__ = "audit_logs"

    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[int | None] = mapped_column(ForeignKey("users.id", ondelete="SET NULL"), index=True)
    action: Mapped[str] = mapped_column(String(100), index=True)
    resource_type: Mapped[str | None] = mapped_column(String(50))
    resource_id: Mapped[str | None] = mapped_column(String(100))
    details: Mapped[dict[str, Any] | None] = mapped_column(JSON)
    ip_address: Mapped[str | None] = mapped_column(String(64))
    created_at: Mapped[datetime] = mapped_column(UTCDateTime, default=utcnow, index=True)
