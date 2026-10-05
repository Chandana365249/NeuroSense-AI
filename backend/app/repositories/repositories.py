from __future__ import annotations

from datetime import datetime
from typing import Any

from sqlalchemy import func, select, update
from sqlalchemy.orm import Session

from app.models import AnalysisSession, AuditLog, Metric, MLModel, Prediction, Report, User


class UserRepository:
    def __init__(self, db: Session) -> None:
        self.db = db

    def get(self, user_id: int) -> User | None:
        return self.db.get(User, user_id)

    def get_by_email(self, email: str) -> User | None:
        return self.db.scalar(select(User).where(func.lower(User.email) == email.lower()))

    def count(self) -> int:
        return int(self.db.scalar(select(func.count(User.id))) or 0)

    def create(self, email: str, hashed_password: str, full_name: str | None, is_admin: bool) -> User:
        user = User(email=email.lower(), hashed_password=hashed_password, full_name=full_name, is_admin=is_admin)
        self.db.add(user)
        self.db.flush()
        return user

    def touch_login(self, user: User, when: datetime) -> None:
        user.last_login_at = when
        self.db.flush()


class SessionRepository:
    def __init__(self, db: Session) -> None:
        self.db = db

    def create(self, user_id: int, name: str, description: str | None) -> AnalysisSession:
        item = AnalysisSession(user_id=user_id, name=name, description=description)
        self.db.add(item)
        self.db.flush()
        return item

    def get_for_user(self, session_id: int, user_id: int) -> AnalysisSession | None:
        return self.db.scalar(select(AnalysisSession).where(AnalysisSession.id == session_id, AnalysisSession.user_id == user_id))

    def list_for_user(self, user_id: int, limit: int, offset: int) -> list[AnalysisSession]:
        stmt = (select(AnalysisSession).where(AnalysisSession.user_id == user_id)
                .order_by(AnalysisSession.created_at.desc(), AnalysisSession.id.desc()).limit(limit).offset(offset))
        return list(self.db.scalars(stmt))

    def prediction_counts(self, session_ids: list[int]) -> dict[int, int]:
        if not session_ids:
            return {}
        stmt = (select(Prediction.session_id, func.count(Prediction.id))
                .where(Prediction.session_id.in_(session_ids)).group_by(Prediction.session_id))
        return {sid: int(n) for sid, n in self.db.execute(stmt)}


class ModelRepository:
    def __init__(self, db: Session) -> None:
        self.db = db

    def get_by_version(self, version: str) -> MLModel | None:
        return self.db.scalar(select(MLModel).where(MLModel.version == version))

    def list(self) -> list[MLModel]:
        return list(self.db.scalars(select(MLModel).order_by(MLModel.trained_at.desc())))

    def upsert_from_metadata(self, meta: dict[str, Any]) -> MLModel:
        model = self.get_by_version(meta["version"])
        if model is None:
            model = MLModel(
                version=meta["version"],
                name=meta["name"],
                algorithm=meta["algorithm"],
                label_strategy=meta["label"]["strategy"],
                class_labels=meta["class_labels"],
                feature_count=meta["feature_schema"]["count"],
                artifact_path=meta["model_path"],
                trained_at=datetime.fromisoformat(meta["trained_at"]),
            )
            self.db.add(model)
            self.db.flush()
        return model

    def set_active(self, model: MLModel) -> None:
        self.db.execute(update(MLModel).where(MLModel.id != model.id).values(is_active=False))
        model.is_active = True
        self.db.flush()


class MetricRepository:
    def __init__(self, db: Session) -> None:
        self.db = db

    def replace_for_model(self, model_id: int, rows: list[tuple[str, str, str, float | None]]) -> None:
        for existing in self.db.scalars(select(Metric).where(Metric.model_id == model_id)):
            self.db.delete(existing)
        self.db.flush()
        self.db.add_all(Metric(model_id=model_id, split=s, level=lv, name=n, value=v) for s, lv, n, v in rows)
        self.db.flush()

    def for_model(self, model_id: int) -> list[Metric]:
        return list(self.db.scalars(select(Metric).where(Metric.model_id == model_id).order_by(Metric.split, Metric.level, Metric.name)))


class PredictionRepository:
    def __init__(self, db: Session) -> None:
        self.db = db

    def create(self, **fields: Any) -> Prediction:
        item = Prediction(**fields)
        self.db.add(item)
        self.db.flush()
        return item

    def get_for_user(self, prediction_id: int, user_id: int) -> Prediction | None:
        return self.db.scalar(select(Prediction).where(Prediction.id == prediction_id, Prediction.user_id == user_id))

    def list_for_user(self, user_id: int, limit: int, offset: int, session_id: int | None = None) -> list[Prediction]:
        stmt = select(Prediction).where(Prediction.user_id == user_id)
        if session_id is not None:
            stmt = stmt.where(Prediction.session_id == session_id)
        stmt = stmt.order_by(Prediction.created_at.desc(), Prediction.id.desc()).limit(limit).offset(offset)
        return list(self.db.scalars(stmt))

    def count_for_user(self, user_id: int) -> int:
        return int(self.db.scalar(select(func.count(Prediction.id)).where(Prediction.user_id == user_id)) or 0)


class ReportRepository:
    def __init__(self, db: Session) -> None:
        self.db = db

    def create(self, report_type: str, title: str, content: dict[str, Any], user_id: int | None, model_id: int | None) -> Report:
        item = Report(report_type=report_type, title=title, content=content, user_id=user_id, model_id=model_id)
        self.db.add(item)
        self.db.flush()
        return item

    def list(self, limit: int, offset: int) -> list[Report]:
        return list(self.db.scalars(select(Report).order_by(Report.created_at.desc(), Report.id.desc()).limit(limit).offset(offset)))


class AuditRepository:
    def __init__(self, db: Session) -> None:
        self.db = db

    def add(self, action: str, user_id: int | None = None, resource_type: str | None = None,
            resource_id: str | None = None, details: dict[str, Any] | None = None, ip_address: str | None = None) -> AuditLog:
        item = AuditLog(action=action, user_id=user_id, resource_type=resource_type, resource_id=resource_id,
                        details=details, ip_address=ip_address)
        self.db.add(item)
        self.db.flush()
        return item

    def list(self, limit: int = 100) -> list[AuditLog]:
        return list(self.db.scalars(select(AuditLog).order_by(AuditLog.id.desc()).limit(limit)))
