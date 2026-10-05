from __future__ import annotations

from collections.abc import Iterator

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from app.database.base import Base
from app.repositories import (
    AuditRepository,
    MetricRepository,
    ModelRepository,
    PredictionRepository,
    SessionRepository,
    UserRepository,
)

META = {
    "version": "v1", "name": "m", "algorithm": "random_forest", "label": {"strategy": "valence_binary"},
    "class_labels": ["high_valence", "low_valence"], "feature_schema": {"count": 440},
    "model_path": "/tmp/m.joblib", "trained_at": "2026-10-01T10:00:00+00:00",
}


@pytest.fixture
def db() -> Iterator[Session]:
    engine = create_engine("sqlite://")
    Base.metadata.create_all(engine)
    session = sessionmaker(bind=engine)()
    yield session
    session.close()


def test_user_lookup_is_case_insensitive(db: Session) -> None:
    repo = UserRepository(db)
    user = repo.create("Alice@Example.com", "hash", "Alice", is_admin=False)
    assert user.email == "alice@example.com"
    assert repo.get_by_email("ALICE@example.COM").id == user.id
    assert repo.count() == 1


def test_model_upsert_is_idempotent_and_single_active(db: Session) -> None:
    repo = ModelRepository(db)
    a = repo.upsert_from_metadata(META)
    assert repo.upsert_from_metadata(META).id == a.id
    b = repo.upsert_from_metadata({**META, "version": "v2"})
    repo.set_active(a)
    repo.set_active(b)
    assert [m.version for m in repo.list() if m.is_active] == ["v2"]


def test_metrics_are_replaced_per_model(db: Session) -> None:
    model = ModelRepository(db).upsert_from_metadata(META)
    repo = MetricRepository(db)
    repo.replace_for_model(model.id, [("test", "segment", "accuracy", 0.5), ("test", "segment", "roc_auc", None)])
    repo.replace_for_model(model.id, [("test", "segment", "accuracy", 0.6)])
    rows = repo.for_model(model.id)
    assert [(r.name, r.value) for r in rows] == [("accuracy", 0.6)]


def test_predictions_are_scoped_to_owner_and_session(db: Session) -> None:
    users = UserRepository(db)
    alice, bob = users.create("a@x.io", "h", None, False), users.create("b@x.io", "h", None, False)
    session = SessionRepository(db).create(alice.id, "demo", None)
    repo = PredictionRepository(db)
    common = {"predicted_label": "high_valence", "confidence": 0.6, "probabilities": {"high_valence": 0.6},
              "segments_used": 15, "input_source": "upload"}
    p1 = repo.create(user_id=alice.id, session_id=session.id, **common)
    repo.create(user_id=alice.id, **common)
    repo.create(user_id=bob.id, **common)
    assert repo.count_for_user(alice.id) == 2
    assert [p.id for p in repo.list_for_user(alice.id, 10, 0, session_id=session.id)] == [p1.id]
    assert repo.get_for_user(p1.id, bob.id) is None
    assert SessionRepository(db).prediction_counts([session.id]) == {session.id: 1}
    assert SessionRepository(db).get_for_user(session.id, bob.id) is None


def test_audit_log(db: Session) -> None:
    repo = AuditRepository(db)
    repo.add("auth.login", None, "user", "1", {"k": "v"}, "127.0.0.1")
    assert repo.list()[0].action == "auth.login"
