"""Deep-learning tests. Small synthetic tensors and synthetic DEAP-format fixtures check that the code
works; they say nothing about real model performance (that comes only from real DEAP runs)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from fastapi.testclient import TestClient  # noqa: E402

from app.core.config import Settings  # noqa: E402
from app.database.base import Base  # noqa: E402
from app.main import create_app  # noqa: E402
from app.ml.deep.classifier import CNNClassifier  # noqa: E402
from app.ml.deep.cnn_model import CNNConfig, EEGCNN, build_cnn, layer_summary, parameter_count  # noqa: E402
from app.ml.deep.data import ChannelNormalizer, EEGSegmentDataset, build_segment_table, encode_labels, make_loader  # noqa: E402
from app.ml.deep.training import CNNTrainer, CNNTrainingConfig  # noqa: E402
from app.ml.factory import make_dataset, make_feature_extractor, make_preprocessor  # noqa: E402
from app.ml.feature_table import build_feature_table  # noqa: E402
from app.ml.registry import ModelNotAvailableError, ModelRegistry  # noqa: E402
from app.services.model_service import ModelService  # noqa: E402
from tests.conftest import make_settings, register_and_login  # noqa: E402


# ---------- model ----------
def test_cnn_is_convolutional_and_small() -> None:
    model = EEGCNN(CNNConfig())
    convs = [m for m in model.modules() if isinstance(m, torch.nn.Conv2d)]
    assert len(convs) == 4
    assert any(m.groups > 1 for m in convs), "depthwise spatial/separable convolutions expected"
    assert any(isinstance(m, torch.nn.AvgPool2d) for m in model.modules())
    assert any(isinstance(m, torch.nn.Dropout) for m in model.modules())
    assert parameter_count(model) == 2130


def test_forward_pass_shape_and_layer_summary() -> None:
    model = EEGCNN(CNNConfig(n_classes=4)).eval()
    out = model(torch.randn(3, 1, 32, 512))
    assert out.shape == (3, 4)
    shapes = {row["stage"]: row["output_shape"] for row in layer_summary(model)}
    assert shapes["spatial"] == [1, 16, 1, 128] and shapes["separable"] == [1, 16, 1, 16]


@pytest.mark.parametrize("shape", [(2, 32, 512), (2, 1, 16, 512), (2, 1, 32, 256), (1, 2, 32, 512)])
def test_input_shape_is_validated(shape: tuple[int, ...]) -> None:
    with pytest.raises(ValueError, match="expects input"):
        EEGCNN(CNNConfig())(torch.zeros(shape))


def test_config_validation() -> None:
    with pytest.raises(ValueError):
        CNNConfig(n_samples=500)
    with pytest.raises(ValueError):
        CNNConfig(dropout=1.0)


def test_seeded_initialization_is_reproducible() -> None:
    a, b, c = build_cnn(CNNConfig(), 1), build_cnn(CNNConfig(), 1), build_cnn(CNNConfig(), 2)
    assert all(torch.equal(x, y) for x, y in zip(a.parameters(), b.parameters(), strict=True))
    assert not all(torch.equal(x, y) for x, y in zip(a.parameters(), c.parameters(), strict=True))


# ---------- data ----------
def test_segment_table_matches_feature_table_rows(settings: Settings) -> None:
    dataset, pre = make_dataset(settings), make_preprocessor(settings)
    seg = build_segment_table(dataset, pre, settings.cache_dir)
    feat = build_feature_table(dataset, pre, make_feature_extractor(settings), settings.cache_dir)
    assert seg.X.shape[1:] == (32, 512) and seg.X.dtype == np.float32
    for attr in ("participant", "trial", "segment", "ratings"):
        assert np.array_equal(getattr(seg, attr), getattr(feat, attr))
    cached = build_segment_table(dataset, pre, settings.cache_dir)
    assert isinstance(cached.X, np.memmap) and np.array_equal(np.asarray(cached.X), np.asarray(seg.X))


def test_normalizer_uses_only_given_rows() -> None:
    X = np.zeros((4, 2, 8), dtype=np.float32)
    X[:2] = 1.0
    X[2:] = 100.0
    norm = ChannelNormalizer.fit(X, np.array([0, 1]))
    assert np.allclose(norm.mean, 1.0)
    restored = ChannelNormalizer.from_dict(norm.to_dict())
    assert np.allclose(restored.apply(X[0]), norm.apply(X[0]))


def test_dataset_and_loader_batches() -> None:
    rng = np.random.default_rng(0)
    X = rng.normal(size=(10, 32, 512)).astype(np.float32)
    codes = encode_labels(np.array(["hi", "lo"] * 5), ["hi", "lo"])
    norm = ChannelNormalizer.fit(X, np.arange(10))
    ds = EEGSegmentDataset(X, np.array([1, 3, 5, 7]), codes, norm)
    x, y = ds[0]
    assert x.shape == (1, 32, 512) and x.dtype == torch.float32 and int(y) == 1
    batches = list(make_loader(ds, batch_size=3, shuffle=True, seed=0))
    assert [b[0].shape[0] for b in batches] == [3, 1]
    again = list(make_loader(ds, batch_size=3, shuffle=True, seed=0))
    assert all(torch.equal(a[1], b[1]) for a, b in zip(batches, again, strict=True))


# ---------- training, checkpoints, registry ----------
@pytest.fixture(scope="module")
def cnn_run(tmp_path_factory: pytest.TempPathFactory, fake_deap_dir: Path) -> tuple[Settings, str]:
    out = tmp_path_factory.mktemp("cnn_outputs")
    settings = make_settings(fake_deap_dir, out, out / "unused.sqlite3")
    cfg = CNNTrainingConfig(max_epochs=4, patience=2, batch_size=8, seed=0)
    return settings, CNNTrainer(settings, training_config=cfg).run().version


def test_training_writes_checkpoints_history_and_metadata(cnn_run: tuple[Settings, str]) -> None:
    settings, version = cnn_run
    run_dir = settings.model_dir / version
    assert version.startswith("cnn_valence_binary_")
    assert (run_dir / "checkpoints" / "best.pt").exists() and (run_dir / "checkpoints" / "last.pt").exists()
    assert (run_dir / "model.pt").exists()
    meta = ModelRegistry(settings.model_dir).metadata(version)
    assert meta["framework"] == "pytorch" and meta["model_type"] == "cnn"
    assert meta["input"]["shape"] == [1, 32, 512]
    assert meta["class_labels"] == ["high_valence", "low_valence"]
    split = meta["split"]
    assert not set(split["test"]) & (set(split["train"]) | set(split["validation"]))
    history = meta["history"]["selection"]
    assert 1 <= len(history) <= 4 and {"train_loss", "val_loss", "val_f1_macro"} <= set(history[0])
    assert 1 <= meta["training_config"]["best_epoch"] <= len(history)
    assert len(meta["history"]["final_fit"]) == meta["training_config"]["best_epoch"]
    assert meta["test"]["segment"]["n_samples"] == split["segments"]["test"]
    for path in meta["artifacts"]["graphs"].values():
        assert path is None or Path(path).exists()
    assert Path(meta["artifacts"]["history"]).exists() and Path(meta["artifacts"]["test_predictions"]).exists()


def test_cnn_is_registered_but_not_activated(cnn_run: tuple[Settings, str]) -> None:
    settings, version = cnn_run
    registry = ModelRegistry(settings.model_dir)
    assert registry.active_version() is None
    assert registry.latest("cnn")["version"] == version


def test_saved_cnn_reloads_with_identical_outputs(cnn_run: tuple[Settings, str]) -> None:
    settings, version = cnn_run
    clf, _ = ModelRegistry(settings.model_dir).load(version)
    assert isinstance(clf, CNNClassifier)
    x = np.random.default_rng(1).normal(size=(3, 32, 512)).astype(np.float32)
    p1 = clf.predict_proba(x)
    clf2, _ = ModelRegistry(settings.model_dir).load(version)
    assert p1.shape == (3, 2) and np.allclose(p1.sum(axis=1), 1, atol=1e-5)
    assert np.allclose(p1, clf2.predict_proba(x))
    with pytest.raises(ValueError, match="segments of shape"):
        clf.predict_proba(np.zeros((1, 16, 512), dtype=np.float32))


def test_tampered_cnn_artifact_is_rejected(tmp_path: Path) -> None:
    clf = CNNClassifier(build_cnn(CNNConfig(), 0), ["a", "b"], ChannelNormalizer(np.zeros(32, np.float32), np.ones(32, np.float32)))
    registry = ModelRegistry(tmp_path)
    registry.save("cnn_v1", clf, {"framework": "pytorch", "trained_at": "2026-01-01T00:00:00+00:00"}, activate=False)
    loaded, _ = registry.load("cnn_v1")
    assert list(loaded.classes_) == ["a", "b"]
    (tmp_path / "cnn_v1" / "model.pt").write_bytes(b"tampered")
    with pytest.raises(ModelNotAvailableError, match="integrity"):
        registry.load("cnn_v1")
    with pytest.raises(FileExistsError):
        registry.save("cnn_v1", clf, {"framework": "pytorch"}, activate=False)


def test_sklearn_artifacts_still_default(trained_output_dir: Path) -> None:
    model, meta = ModelRegistry(trained_output_dir / "models").load()
    assert ModelRegistry.framework_of(meta) == "sklearn" and hasattr(model, "predict_proba")


def test_model_service_predicts_with_active_cnn(cnn_run: tuple[Settings, str], tmp_path: Path) -> None:
    settings, version = cnn_run
    ModelRegistry(settings.model_dir).activate(version)
    try:
        service = ModelService(settings)
        assert service.load_active()
        rng = np.random.default_rng(0)
        result = service.predict_signal(rng.normal(0, 5, size=(32, 128 * 8)), includes_baseline=False)
        assert result.segments_used == 2
        assert set(result.probabilities) == {"high_valence", "low_valence"}
        assert result.label == max(result.probabilities, key=result.probabilities.get)
    finally:
        (settings.model_dir / ModelRegistry.ACTIVE_FILE).unlink()


# ---------- API ----------
def test_api_reports_framework_and_lists_cnn(trained_output_dir: Path, fake_deap_dir: Path, tmp_path: Path,
                                             cnn_run: tuple[Settings, str]) -> None:
    import shutil

    cnn_settings, cnn_version = cnn_run
    out = tmp_path / "outputs"
    shutil.copytree(trained_output_dir, out)
    shutil.copytree(cnn_settings.model_dir / cnn_version, out / "models" / cnn_version)
    settings = make_settings(fake_deap_dir, out, tmp_path / "db.sqlite3")
    app = create_app(settings)
    Base.metadata.create_all(app.state.db.engine)
    with TestClient(app) as client:
        info = client.get("/api/model-info").json()
        assert info["framework"] == "sklearn" and info["model_type"] != "cnn"
        assert info["input_requirements"]["channels"] == 32
        models = {m["version"]: m for m in client.get("/api/models").json()}
        assert models[cnn_version]["framework"] == "pytorch" and models[cnn_version]["model_type"] == "cnn"
        assert models[cnn_version]["is_active"] is False
        assert sum(m["is_active"] for m in models.values()) == 1
        headers = register_and_login(client)
        r = client.post("/api/predict", headers=headers, json={"deap_reference": {"participant_id": "s01", "trial_index": 0}})
        assert r.status_code == 201 and r.json()["model"]["algorithm"] != "cnn"
