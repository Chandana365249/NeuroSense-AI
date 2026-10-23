"""Shared fixtures. Synthetic DEAP-format files are created here for fast unit tests only;
production code paths never generate data. Real-data tests are marked `integration`."""

from __future__ import annotations

import pickle
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.core.config import Settings
from app.database.base import Base
from app.main import create_app
from app.ml.training import BaselineTrainer

SR = 128
N_CHANNELS_TOTAL = 40
BASELINE = 3 * SR
SEGMENT = 4 * SR
N_SEGMENTS = 2
N_TRIALS = 6
PARTICIPANTS = [f"s{i:02d}" for i in range(1, 7)]


def synthetic_trial(rng: np.random.Generator, high_valence: bool) -> np.ndarray:
    n = BASELINE + N_SEGMENTS * SEGMENT
    t = np.arange(n) / SR
    data = rng.normal(0, 2.0, size=(N_CHANNELS_TOTAL, n))
    alpha_amp = 6.0 if high_valence else 1.5
    phases = rng.uniform(0, 2 * np.pi, size=(32, 1))
    data[:32] += alpha_amp * np.sin(2 * np.pi * 10 * t + phases)
    data[:32] += 2.0 * np.sin(2 * np.pi * 20 * t)
    data[32:] *= 500
    return data


def write_fake_participant(path: Path, rng: np.random.Generator) -> None:
    labels = np.zeros((N_TRIALS, 4))
    data = np.zeros((N_TRIALS, N_CHANNELS_TOTAL, BASELINE + N_SEGMENTS * SEGMENT))
    for t in range(N_TRIALS):
        high = t % 2 == 0
        labels[t] = [8.0 if high else 2.0, rng.uniform(1, 9), rng.uniform(1, 9), rng.uniform(1, 9)]
        data[t] = synthetic_trial(rng, high)
    with path.open("wb") as fh:
        pickle.dump({"labels": labels, "data": data}, fh, protocol=2)


@pytest.fixture(scope="session")
def fake_deap_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("deap")
    rng = np.random.default_rng(0)
    for pid in PARTICIPANTS:
        write_fake_participant(root / f"{pid}.dat", rng)
    return root


def make_settings(dataset_dir: Path, output_dir: Path, db_path: Path) -> Settings:
    return Settings(
        _env_file=None,  # type: ignore[call-arg]
        environment="test",
        log_level="WARNING",
        jwt_secret_key="test-secret-key-for-unit-tests-only-0123456789",
        dataset_dir=dataset_dir,
        output_dir=output_dir,
        database_url=f"sqlite:///{db_path.as_posix()}",
        test_participants=2,
        validation_participants=1,
        development_participants=PARTICIPANTS,  # the synthetic test files s01..s06 (none of them frozen)
    )


@pytest.fixture(scope="session")
def trained_output_dir(tmp_path_factory: pytest.TempPathFactory, fake_deap_dir: Path) -> Path:
    out = tmp_path_factory.mktemp("trained_outputs")
    settings = make_settings(fake_deap_dir, out, out / "unused.sqlite3")
    BaselineTrainer(settings, run_cross_validation=True, activate=True).run()  # shared fixture: explicitly active
    return out


@pytest.fixture
def settings(tmp_path: Path, fake_deap_dir: Path) -> Settings:
    return make_settings(fake_deap_dir, tmp_path / "outputs", tmp_path / "test.sqlite3")


@pytest.fixture
def trained_settings(tmp_path: Path, fake_deap_dir: Path, trained_output_dir: Path) -> Settings:
    return make_settings(fake_deap_dir, trained_output_dir, tmp_path / "test.sqlite3")


def _client(settings: Settings) -> Iterator[TestClient]:
    app = create_app(settings)
    Base.metadata.create_all(app.state.db.engine)
    with TestClient(app) as client:
        yield client


@pytest.fixture
def client(settings: Settings) -> Iterator[TestClient]:
    yield from _client(settings)


@pytest.fixture
def trained_client(trained_settings: Settings) -> Iterator[TestClient]:
    yield from _client(trained_settings)


ADMIN_EMAIL = "admin@example.com"


def register_and_login(client: TestClient, email: str = ADMIN_EMAIL, password: str = "correct-horse-battery",
                       admin: bool | None = None) -> dict[str, str]:
    """Create an account and return auth headers. The default test administrator is created through the
    command-line bootstrap service (public registration only ever creates standard accounts)."""
    if (email == ADMIN_EMAIL) if admin is None else admin:
        from app.services.auth_service import AuthService

        with client.app.state.db.session() as session:  # type: ignore[attr-defined]
            AuthService(session, client.app.state.settings).bootstrap_admin(email, password, "Test")  # type: ignore[attr-defined]
            session.commit()
    else:
        r = client.post("/api/auth/register", json={"email": email, "password": password, "full_name": "Test"})
        assert r.status_code == 201, r.text
    r = client.post("/api/auth/login", data={"username": email, "password": password})
    assert r.status_code == 200, r.text
    return {"Authorization": f"Bearer {r.json()['access_token']}"}
