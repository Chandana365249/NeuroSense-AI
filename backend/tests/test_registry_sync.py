"""Registry synchronisation is a true no-op when stored records already match the model metadata,
and still reconciles them when they differ. Every test uses its own temporary SQLite database (see conftest)."""
from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient
from sqlalchemy import text

from app.database.base import Base
from app.main import create_app
from app.models import Metric
from app.repositories.repositories import MetricRepository, _metrics_match
from tests.conftest import make_settings
from tests.test_api_hardening import WRITE_SQL, captured_sql

SNAPSHOT_SQL = {
    "models": "select id, version, name, algorithm, is_active, trained_at, created_at from models order by id",
    "metrics": "select id, model_id, split, level, name, value, created_at from metrics order by id",
}


def snapshot(app) -> dict[str, list[tuple]]:  # noqa: ANN001
    with app.state.db.engine.connect() as conn:
        return {name: [tuple(r) for r in conn.execute(text(sql))] for name, sql in SNAPSHOT_SQL.items()}


def run_sync(app) -> None:  # noqa: ANN001
    with app.state.db.session() as session:
        app.state.model_service.sync_registry_to_db(session)
        session.commit()


def app_on(db_path: Path, fake_deap_dir: Path, output_dir: Path):  # noqa: ANN201
    app = create_app(make_settings(fake_deap_dir, output_dir, db_path))
    Base.metadata.create_all(app.state.db.engine)
    return app


# ---------- 1 and 4: matching records -> no writes, IDs/timestamps/values and registry preserved ----------
def test_restart_with_matching_records_writes_nothing(tmp_path: Path, fake_deap_dir: Path, trained_output_dir: Path) -> None:
    db_path = tmp_path / "registry.sqlite3"
    first = app_on(db_path, fake_deap_dir, trained_output_dir)
    with TestClient(first):  # first start registers the model and its metrics
        pass
    before = snapshot(first)
    assert before["models"] and before["metrics"]
    first.state.db.dispose()

    second = app_on(db_path, fake_deap_dir, trained_output_dir)  # a restart on the same database
    statements: list[str] = []
    from sqlalchemy import event

    event.listen(second.state.db.engine, "before_cursor_execute",
                 lambda conn, cursor, statement, params, context, many: statements.append(statement))
    with TestClient(second) as client:
        active = client.get("/api/model-info").json()["version"]
    writes = [s for s in statements if WRITE_SQL.match(s)]
    assert any(s.lstrip().upper().startswith("SELECT") for s in statements)  # the sync did read the registry
    assert writes == [], writes
    assert snapshot(second) == before  # same IDs, timestamps, values and active flags
    assert second.state.model_service.registry.active_version() == active
    assert [r[1] for r in before["models"] if r[4]] == [active]


# ---------- 3: repeat-safe ----------
def test_repeated_sync_is_repeat_safe(trained_client: TestClient) -> None:
    app = trained_client.app
    before = snapshot(app)
    for _ in range(3):
        with captured_sql(trained_client) as statements:
            run_sync(app)
        assert [s for s in statements if WRITE_SQL.match(s)] == []
    assert snapshot(app) == before


# ---------- 2: genuine differences are still reconciled ----------
def _metric_rows(app) -> dict[tuple[str, str, str], float | None]:  # noqa: ANN001
    with app.state.db.engine.connect() as conn:
        return {(s, lv, n): v for s, lv, n, v in conn.execute(text("select split, level, name, value from metrics"))}


def test_sync_restores_a_missing_metric(trained_client: TestClient) -> None:
    app = trained_client.app
    expected = _metric_rows(app)
    with app.state.db.engine.begin() as conn:
        conn.execute(text("delete from metrics where id = (select min(id) from metrics)"))
    assert len(_metric_rows(app)) == len(expected) - 1
    with captured_sql(trained_client) as statements:
        run_sync(app)
    assert any(WRITE_SQL.match(s) for s in statements)
    assert _metric_rows(app) == expected


def test_sync_removes_an_unexpected_metric(trained_client: TestClient) -> None:
    app = trained_client.app
    expected = _metric_rows(app)
    with app.state.db.engine.begin() as conn:
        model_id = conn.execute(text("select id from models")).scalar()
        conn.execute(text("insert into metrics (model_id, split, level, name, value, created_at) "
                          "values (:m, 'test', 'segment', 'unexpected_metric', 0.99, CURRENT_TIMESTAMP)"), {"m": model_id})
    run_sync(app)
    assert _metric_rows(app) == expected


def test_sync_corrects_a_changed_value(trained_client: TestClient) -> None:
    app = trained_client.app
    expected = _metric_rows(app)
    key = next(k for k, v in expected.items() if v is not None)
    with app.state.db.engine.begin() as conn:
        conn.execute(text("update metrics set value = value + 0.125 where split = :s and level = :l and name = :n"),
                     {"s": key[0], "l": key[1], "n": key[2]})
    assert _metric_rows(app)[key] != expected[key]
    run_sync(app)
    assert _metric_rows(app) == expected


def test_sync_deactivates_a_wrongly_active_model_without_touching_correct_rows(trained_client: TestClient) -> None:
    app = trained_client.app
    with app.state.db.engine.begin() as conn:
        conn.execute(text("insert into models (version, name, algorithm, label_strategy, class_labels, feature_count, "
                          "artifact_path, trained_at, is_active, created_at) values ('other-version', 'x', 'random_forest', "
                          "'valence_binary', '[]', 1, 'x', CURRENT_TIMESTAMP, 1, CURRENT_TIMESTAMP)"))
    run_sync(app)
    with app.state.db.engine.connect() as conn:
        active = [r[0] for r in conn.execute(text("select version from models where is_active = 1"))]
    assert active == [app.state.model_service.registry.active_version()]


# ---------- comparison rules ----------
def _m(split: str, level: str, name: str, value: float | None) -> Metric:
    return Metric(model_id=1, split=split, level=level, name=name, value=value)


def test_metric_comparison_is_strict() -> None:
    rows = [("test", "segment", "accuracy", 0.5), ("test", "segment", "roc_auc", None), ("test", "trial", "f1_macro", float("nan"))]
    same = [_m(*r) for r in rows]
    assert _metrics_match(same, rows)
    assert not _metrics_match(same[:-1], rows)  # missing
    assert not _metrics_match([*same, _m("test", "segment", "extra", 1.0)], rows)  # unexpected
    assert not _metrics_match([_m("test", "segment", "accuracy", 0.5000001), *same[1:]], rows)  # changed value
    assert not _metrics_match([_m("test", "segment", "accuracy", 0.5), _m("test", "segment", "roc_auc", 0.7), same[2]], rows)  # None vs value
    assert not _metrics_match([*same, _m("test", "segment", "accuracy", 0.5)], rows)  # duplicate key
    assert not _metrics_match([], rows) and _metrics_match([], [])


def test_replace_for_model_reports_whether_it_wrote(trained_client: TestClient) -> None:
    app = trained_client.app
    with app.state.db.session() as session:
        model_id = session.execute(text("select id from models")).scalar()
        rows = [(m.split, m.level, m.name, m.value) for m in MetricRepository(session).for_model(model_id)]
        assert MetricRepository(session).replace_for_model(model_id, rows) is False
        assert MetricRepository(session).replace_for_model(model_id, rows[:-1]) is True
        session.rollback()  # leave this test database as it was
