"""Robust-evaluation tests: statistics helpers, leakage-safe run planning, and the evaluator's
reproduction of the registered CNN training procedure. Synthetic data only checks that the code
works; real performance numbers come only from real DEAP runs (scripts/evaluate_cnn.py)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from app.ml.evaluation import compute_metrics
from app.ml.splitting import ParticipantSplit
from app.ml.stats import binary_metric_fns, cluster_bootstrap, mean_ci, permutation_test, wilcoxon

CLASSES = ["high_valence", "low_valence"]


def _random_case(seed: int, n: int = 300) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    y = np.asarray(CLASSES)[rng.integers(0, 2, n)]
    p1 = np.round(rng.uniform(size=n), 1)  # coarse values -> ties in ranks and argmax
    return y, np.column_stack([1 - p1, p1])


# ---------- statistics ----------
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_fast_metrics_match_compute_metrics(seed: int) -> None:
    y, proba = _random_case(seed)
    fns, ref = binary_metric_fns(CLASSES), compute_metrics(y, proba, CLASSES)
    for k in ("accuracy", "balanced_accuracy", "f1_macro", "roc_auc"):
        assert fns[k](y, proba) == pytest.approx(ref[k], abs=1e-12)


def test_fast_metrics_single_class_and_one_sided_predictions() -> None:
    y = np.array([CLASSES[0]] * 5)
    proba = np.tile([0.2, 0.8], (5, 1))
    fns, ref = binary_metric_fns(CLASSES), compute_metrics(y, proba, CLASSES)
    assert fns["roc_auc"](y, proba) is None and ref["roc_auc"] is None
    for k in ("accuracy", "balanced_accuracy", "f1_macro"):
        assert fns[k](y, proba) == pytest.approx(ref[k])


def test_cluster_bootstrap_is_seeded_and_brackets_estimate() -> None:
    y, proba = _random_case(3, 240)
    clusters, strata = np.arange(240) // 4, np.arange(240) // 80
    fns = binary_metric_fns(CLASSES)
    a = cluster_bootstrap(y, proba, clusters, strata, fns, n_boot=200, seed=1)
    b = cluster_bootstrap(y, proba, clusters, strata, fns, n_boot=200, seed=1)
    assert a == b
    for k, r in a.items():
        assert r["ci_low"] <= r["estimate"] <= r["ci_high"], k


def test_permutation_test_separates_signal_from_noise() -> None:
    rng = np.random.default_rng(0)
    y = np.asarray(CLASSES)[np.tile([0, 1], 40)]
    strata = np.repeat(np.arange(4), 20)
    perfect = np.column_stack([(y == CLASSES[0]) * 0.9 + 0.05, (y == CLASSES[1]) * 0.9 + 0.05])
    noise = rng.uniform(size=(80, 2))
    fns = {"f1_macro": binary_metric_fns(CLASSES)["f1_macro"]}
    assert permutation_test(y, perfect, strata, fns, n_perm=500)["f1_macro"]["p_value"] < 0.01
    p_noise = permutation_test(y, noise, strata, fns, n_perm=500)["f1_macro"]["p_value"]
    assert 0.05 < p_noise <= 1.0


def test_mean_ci_and_wilcoxon() -> None:
    r = mean_ci([0.4, 0.5, 0.6])
    assert r["mean"] == pytest.approx(0.5) and r["ci_low"] < 0.5 < r["ci_high"] and r["n"] == 3
    assert mean_ci([])["mean"] is None
    assert wilcoxon([0.5, 0.5], 0.5)["p_value"] is None
    assert 0 < wilcoxon([0.6, 0.7, 0.55, 0.65, 0.62, 0.58], 0.5)["p_value"] < 0.05


# ---------- planning / normalization (need torch) ----------
torch = pytest.importorskip("torch")

from app.core.config import Settings  # noqa: E402
from app.ml.deep.robust_evaluation import RobustCNNEvaluator, RunSpec, participant_standardize, plan_runs  # noqa: E402
from app.ml.deep.training import CNNTrainer, CNNTrainingConfig  # noqa: E402
from tests.conftest import make_settings  # noqa: E402


def test_lopo_plan_never_touches_test_participants() -> None:
    split = ParticipantSplit(train=["s02", "s03", "s05", "s07"], validation=["s08", "s09"], test=["s01", "s04"], seed=42)
    specs = plan_runs(split, ["repeated_seeds", "lopo"], ["train_stats"], [42, 43], lopo_seed=42, n_inner_validation=2)
    seeds = [s for s in specs if s.protocol == "repeated_seeds"]
    lopo = [s for s in specs if s.protocol == "lopo"]
    assert [s.seed for s in seeds] == [42, 43] and all(s.evaluate == ("s01", "s04") for s in seeds)
    assert sorted(s.fold for s in lopo) == sorted(split.train + split.validation)
    for s in lopo:
        used = set(s.train) | set(s.validation) | set(s.evaluate)
        assert not used & set(split.test)
        assert s.evaluate == (s.fold,) and len(s.validation) == 2 and len(used) == 6
    assert len({s.run_id for s in specs}) == len(specs)


def test_run_spec_rejects_overlapping_participants() -> None:
    with pytest.raises(ValueError, match="disjoint"):
        RunSpec("lopo", "train_stats", 0, ("s01", "s02"), ("s03",), ("s02",), fold="s02")
    with pytest.raises(ValueError, match="Unknown"):
        RunSpec("kfold", "train_stats", 0, ("s01",), ("s03",), ("s02",))


def test_participant_standardize_uses_only_each_participants_data() -> None:
    rng = np.random.default_rng(0)
    X = rng.normal(5, 3, size=(6, 2, 50)).astype(np.float32)
    participant = np.array(["a", "a", "a", "b", "b", "b"])
    out = participant_standardize(X, participant)
    for pid in ("a", "b"):
        block = out[participant == pid].astype(np.float64)
        assert np.allclose(block.mean(axis=(0, 2)), 0, atol=1e-5) and np.allclose(block.std(axis=(0, 2)), 1, atol=1e-4)
    X2 = X.copy()
    X2[participant == "b"] *= 100
    assert np.array_equal(participant_standardize(X2, participant)[:3], out[:3])


# ---------- evaluator on synthetic DEAP-format data ----------
@pytest.fixture(scope="module")
def eval_settings(tmp_path_factory: pytest.TempPathFactory, fake_deap_dir: Path) -> Settings:
    out = tmp_path_factory.mktemp("robust_outputs")
    return make_settings(fake_deap_dir, out, out / "unused.sqlite3")


CFG = CNNTrainingConfig(max_epochs=3, patience=2, batch_size=8, seed=0)


def test_seed_run_reproduces_registered_training_procedure(eval_settings: Settings) -> None:
    registered = CNNTrainer(eval_settings, training_config=CFG).run().metadata
    evaluator = RobustCNNEvaluator(eval_settings, training_config=CFG)
    spec = evaluator.plan(["repeated_seeds"], ["train_stats"], [0])[0]
    record = evaluator.run(spec)
    assert record["best_epoch"] == registered["training_config"]["best_epoch"]
    assert list(spec.evaluate) == registered["split"]["test"]
    for level in ("segment", "trial"):
        for k in ("accuracy", "f1_macro", "roc_auc", "confusion_matrix"):
            assert record["evaluated"][level][k] == registered["test"][level][k]


def test_run_record_contents(eval_settings: Settings) -> None:
    evaluator = RobustCNNEvaluator(eval_settings, training_config=CFG)
    spec = next(s for s in evaluator.plan(["lopo"], ["per_participant"], [0]) if s.protocol == "lopo")
    record = evaluator.run(spec)
    assert record["run_id"] == spec.run_id and record["normalization"] == "per_participant"
    assert record["participants"]["evaluated"] == [spec.fold]
    assert not set(record["participants"]["evaluated"]) & set(evaluator.split().test)
    n = record["segments"]["evaluated"]
    assert record["evaluated"]["segment"]["n_samples"] == n == len(record["predictions"]["proba"])
    assert set(record["per_participant"]) == {spec.fold}
    assert 1 <= record["best_epoch"] <= record["epochs_run"] <= CFG.max_epochs
    assert len(record["history"]["final_fit"]) == record["best_epoch"]
    assert {"train", "validation"} == set(record["selected_checkpoint"])
    cfg = evaluator.experiment_config()
    assert cfg["dataset"]["fingerprint"] and cfg["architecture"]["parameters"] == 2130


# ---------- report ----------
def test_report_from_runs_with_baseline_comparison(tmp_path: Path, fake_deap_dir: Path) -> None:
    import json

    from app.ml.deep.robust_report import write_report
    from app.ml.training import BaselineTrainer

    settings = make_settings(fake_deap_dir, tmp_path / "out", tmp_path / "unused.sqlite3")
    BaselineTrainer(settings, run_cross_validation=True, activate=True).run()  # active feature model with LOPO folds
    CNNTrainer(settings, training_config=CFG).run()  # registered CNN for the reproduction check
    evaluator = RobustCNNEvaluator(settings, training_config=CFG)
    specs = evaluator.plan(["repeated_seeds", "lopo"], ["train_stats", "per_participant"], [0, 1])
    exp = tmp_path / "exp"
    (exp / "runs").mkdir(parents=True)
    (exp / "experiment.json").write_text(json.dumps({**evaluator.experiment_config(), "seeds": [0, 1]}), encoding="utf-8")
    for spec in specs:
        (exp / "runs" / f"{spec.run_id}.json").write_text(json.dumps(evaluator.run(spec)), encoding="utf-8")

    report = write_report(exp, settings)
    summary = json.loads((exp / "summary.json").read_text(encoding="utf-8"))
    assert summary["n_runs"] == len(specs) == 2 * 2 + 2 * 4
    assert all(summary["comparability_checks"].values())
    assert summary["reproduction_check"]["identical_test_metrics"] is True
    for norm in ("train_stats", "per_participant"):
        assert len(summary["repeated_seeds"][norm]["runs"]) == 2
        assert len(summary["lopo"][norm]["folds"]) == 4 and "paired_vs_random_forest" in summary["lopo"][norm]
        stats = summary["repeated_seeds"][norm]["runs"][0]["statistics"]
        assert 0 < stats["permutation_trial"]["f1_macro"]["p_value"] <= 1
    assert summary["random_forest"]["per_participant"] and summary["random_forest"]["statistics"]
    for path in summary["figures"].values():
        assert Path(path).exists()
    text = report.read_text(encoding="utf-8")
    assert "Repeated seeds" in text and "leave-one-participant-out" in text and "Random Forest" in text
