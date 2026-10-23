"""Model-improvement experiments (Protocol v1.1, exploratory, 12 development participants).

Design (declared here before any result of this script was seen; mirrored in docs/MODEL_IMPROVEMENT_EXPERIMENTS.md §2):

Data: the cached 440-feature table (fingerprint 3c5359083bc90ae5) of the 12 trusted development participants only.
The frozen test participants are asserted absent. Label: high_valence if valence > 5.0 (Decision F).

P1  Participant-independent, nested leave-one-participant-out (12 outer folds).
    Inner selection: grouped 4-fold over the 11 training participants (assignment default_rng([20261021, outer])),
    inner score = mean per-participant trial balanced accuracy. Candidates (7): registered RF, registered XGBoost,
    logistic regression C in {0.001, 0.01, 0.1}, RBF SVM C in {0.1, 1} (scores = logistic(decision_function));
    trial aggregation in {mean, median, vote, mean log-odds}. Ties -> first in declared order.
    Normalization: N0 = training-fold StandardScaler (inductive); N3 = each participant's features z-scored with that
    participant's own unlabelled segments (calibration variant; reported separately, never as zero-shot).
    Seeds 42-44 for RF/XGB (RF reference also 45-46); LR and SVM are deterministic.
    Ablations (RF, mean aggregation, seeds 42-44): feature families and channel regions (FEATURE_SETS).
    Also: continuous valence regression (RF regressor, trial mean thresholded at 5.0), Protocol S1 (borderline 4.5-5.5
    removed from training and scoring), and a training-participant learning curve (3/6/9 of the 11).
P2  Within-participant supervised personalization on the label-free video folds of the earlier experiments
    (lvo.video_folds, seeds 42-46; 32 training videos, 8 test videos). Inner selection: 4 label-free groups of the
    32 training videos (default_rng([20261021, participant, seed, fold])), inner score = trial balanced accuracy over
    the 32 inner out-of-fold trials. Same candidates and aggregations. Ablations, regression, S1 and calibration-size
    curve (8/16/24 videos, the nested subsets of within_person_calibration.lc_subsets) with the registered RF.
P3  Population + personal calibration: train on the other 11 participants' non-test videos plus 0/8/16/32 of the
    person's own rated calibration videos (same subsets as P2); every participant z-scored with its own unlabelled
    segments on the non-test videos; own rows weighted so their total weight equals the other participants' total
    ("equal"), plus unweighted pooling at 32. Registered RF, mean aggregation, seeds 42-44.
LEAK  Demonstration only (invalid by design): within-person segment-level random split, to show the size of the
    inflation that a segment split causes. Never reported as a result.
Primary metric everywhere: trial balanced accuracy per participant, seed-averaged, mean with t 95% CI over 12.
Paired comparisons: participant bootstrap (10,000) CI of the mean difference and Wilcoxon. Three pre-declared
comparisons (Holm): P1 nested-selected vs RF reference; P2 nested-selected vs RF reference; P3 32-video pooled vs 0.

Usage (from backend/):
  python -m scripts.model_improvement --phase baseline|p1|p2|p3|leak|candidate|report
"""

from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse  # noqa: E402
import csv  # noqa: E402
import json  # noqa: E402
from datetime import UTC, datetime  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Any  # noqa: E402

import numpy as np  # noqa: E402
from joblib import Parallel, delayed  # noqa: E402
from scipy import stats as sps  # noqa: E402
from sklearn.ensemble import RandomForestRegressor  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score  # noqa: E402
from sklearn.model_selection import StratifiedKFold  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402
from sklearn.svm import SVC  # noqa: E402

from app.core.config import get_settings  # noqa: E402
from app.ml.deep.robust_evaluation import git_commit  # noqa: E402
from app.ml.models import candidate_models  # noqa: E402
from app.ml.participants import FROZEN_TEST_PARTICIPANTS, PROTOCOL_VERSION  # noqa: E402
from app.ml.stats import mean_ci, wilcoxon  # noqa: E402
from scripts import cross_participant_lvo as lvo  # noqa: E402
from scripts import within_person_calibration as wpc  # noqa: E402
from scripts import within_person_diagnostic as wp  # noqa: E402

PARTICIPANTS = wp.PARTICIPANTS
LABEL = "exploratory (12 participants, Protocol v1.1)"
SEL_SEED = 20261021
SEEDS = (42, 43, 44)
REF_SEEDS = (42, 43, 44, 45, 46)
N_JOBS = 8
BORDERLINE = (4.5, 5.5)
CONFIGS: list[dict[str, Any]] = [
    {"name": "rf", "family": "rf"},
    {"name": "xgb", "family": "xgb"},
    {"name": "lr_C0.001", "family": "lr", "C": 0.001},
    {"name": "lr_C0.01", "family": "lr", "C": 0.01},
    {"name": "lr_C0.1", "family": "lr", "C": 0.1},
    {"name": "svm_C0.1", "family": "svm", "C": 0.1},
    {"name": "svm_C1", "family": "svm", "C": 1.0},
]
DETERMINISTIC = {"lr", "svm"}
AGGS = ("mean", "median", "vote", "logit")
REGIONS = {
    "frontal": ("Fp1", "Fp2", "AF3", "AF4", "F3", "F4", "F7", "F8", "Fz"),
    "central": ("FC1", "FC2", "FC5", "FC6", "C3", "C4", "Cz"),
    "temporal": ("T7", "T8"),
    "parietal": ("CP1", "CP2", "CP5", "CP6", "P3", "P4", "P7", "P8", "Pz"),
    "occipital": ("PO3", "PO4", "O1", "O2", "Oz"),
}
ARTIFACT_PRONE = ("Fp1", "Fp2", "AF3", "AF4", "F7", "F8", "T7", "T8")  # eye (frontopolar) and muscle (temporal) sites
ABLATIONS = ("no_gamma", "theta", "alpha", "beta", "gamma", "non_band", "asymmetry",
             *REGIONS, "no_artifact_prone")
ROOT = Path(__file__).resolve().parents[2]
DOCS_OUT = ROOT / "docs" / "results" / "exploratory" / "model_improvement"


# ---------------------------------------------------------------- data

def feature_sets(names: list[str]) -> dict[str, np.ndarray]:
    def electrodes(n: str) -> tuple[str, ...]:
        if n.startswith("asym_"):
            return tuple(n.split("_")[1].split("-"))
        return (n.split("_")[0],)

    band = {b: np.array([n.endswith("_" + b) for n in names]) for b in ("theta", "alpha", "beta", "gamma")}
    sets = {"full": np.ones(len(names), bool), "no_gamma": ~band["gamma"], **band,
            "non_band": ~np.any(list(band.values()), axis=0),
            "asymmetry": np.array([n.startswith("asym_") for n in names])}
    for r, chans in REGIONS.items():
        sets[r] = np.array([not n.startswith("asym_") and electrodes(n)[0] in chans for n in names])
    sets["no_artifact_prone"] = np.array([not set(electrodes(n)) & set(ARTIFACT_PRONE) for n in names])
    assert sum(int(sets[r].sum()) for r in REGIONS) == 32 * 12, "regions must partition the channel features"
    assert all(sets[k].any() for k in sets)
    return sets


def per_participant_z(X: np.ndarray, p: np.ndarray, stat_rows: np.ndarray) -> np.ndarray:
    """Each participant's features z-scored with the mean/SD of that participant's own `stat_rows` (no labels)."""
    out = np.empty_like(X, dtype=np.float64)
    for q in np.unique(p):
        m = p == q
        ref = X[m & stat_rows]
        sd = ref.std(axis=0)
        out[m] = (X[m] - ref.mean(axis=0)) / np.where(sd > 1e-12, sd, 1.0)
    return out


def load_data() -> dict[str, Any]:
    table, y = wp.load()
    present = set(table.participant.tolist())
    assert present == set(PARTICIPANTS), f"unexpected participants {sorted(present)}"
    assert not present & FROZEN_TEST_PARTICIPANTS, "frozen test participants must never be loaded"
    index = {pid: i for i, pid in enumerate(PARTICIPANTS)}
    p = np.array([index[x] for x in table.participant])
    t = table.trial.astype(int)
    val = table.ratings[:, 0].astype(float)
    yb = (val > 5.0).astype(int)
    assert np.array_equal(yb == 1, y == "high_valence"), "labels must equal the protocol rule"
    for q in range(len(PARTICIPANTS)):
        assert sorted(set(t[p == q].tolist())) == list(range(40)), "every participant must have 40 trials"
    X = table.X.astype(np.float64)
    return {"X": X, "Xn3": per_participant_z(X, p, np.ones(len(p), bool)), "y": yb, "val": val, "p": p, "t": t,
            "border": (val >= BORDERLINE[0]) & (val <= BORDERLINE[1]), "fsets": feature_sets(table.feature_names),
            "names": table.feature_names, "fingerprint": table.fingerprint}


# ---------------------------------------------------------------- models and metrics

def make_model(cfg: dict[str, Any], seed: int) -> Pipeline:
    fam = cfg["family"]
    if fam in ("rf", "xgb"):  # the registered configurations, single-threaded (workers run in parallel)
        est = candidate_models(seed)["random_forest" if fam == "rf" else "xgboost"]().estimator
        return est.set_params(clf__n_jobs=1)
    if fam == "lr":
        return Pipeline([("scaler", StandardScaler()),
                         ("clf", LogisticRegression(C=cfg["C"], class_weight="balanced", max_iter=5000))])
    if fam == "svm":
        return Pipeline([("scaler", StandardScaler()),
                         ("clf", SVC(C=cfg["C"], kernel="rbf", gamma="scale", class_weight="balanced"))])
    if fam == "rf_reg":
        return Pipeline([("scaler", StandardScaler()),
                         ("clf", RandomForestRegressor(n_estimators=300, min_samples_leaf=5, max_features="sqrt",
                                                       random_state=seed, n_jobs=1))])
    raise ValueError(fam)


def threshold(cfg: dict[str, Any]) -> float:
    return 5.0 if cfg["family"] == "rf_reg" else 0.5


def seg_scores(model: Pipeline, cfg: dict[str, Any], X: np.ndarray) -> np.ndarray:
    if cfg["family"] == "svm":
        return 1.0 / (1.0 + np.exp(-model.decision_function(X)))
    if cfg["family"] == "rf_reg":
        return model.predict(X)
    assert list(model.classes_) == [0, 1]
    return model.predict_proba(X)[:, 1]


def aggs_for(cfg: dict[str, Any]) -> tuple[str, ...]:
    return ("mean", "median", "vote") if cfg["family"] == "rf_reg" else AGGS


def trial_outputs(t: np.ndarray, y: np.ndarray, s: np.ndarray, agg: str, thr: float) -> dict[str, list]:
    trials = np.unique(t)
    score, ytr = np.empty(len(trials)), np.empty(len(trials), int)
    for i, tr in enumerate(trials):
        m = t == tr
        v = s[m]
        ytr[i] = y[m][0]
        assert (y[m] == ytr[i]).all()
        if agg == "mean":
            score[i] = v.mean()
        elif agg == "median":
            score[i] = np.median(v)
        elif agg == "vote":
            score[i] = (v > thr).mean()
        else:
            c = np.clip(v, 1e-6, 1 - 1e-6)
            score[i] = np.log(c / (1 - c)).mean()
    cut = {"vote": 0.5, "logit": 0.0}.get(agg, thr)
    return {"trials": trials.tolist(), "y": ytr.tolist(), "score": score.round(6).tolist(),
            "pred": (score > cut).astype(int).tolist()}


def metrics(y: np.ndarray, score: np.ndarray, pred: np.ndarray) -> dict[str, float | None]:
    y, score, pred = np.asarray(y), np.asarray(score), np.asarray(pred)
    both = len(set(y.tolist())) == 2
    return {"balanced_accuracy": float(balanced_accuracy_score(y, pred)) if both else None,
            "f1_macro": float(f1_score(y, pred, average="macro", labels=[0, 1], zero_division=0)),
            "accuracy": float((y == pred).mean()),
            "roc_auc": float(roc_auc_score(y, score)) if both else None,
            "pred_high_share": float(pred.mean()), "true_high_share": float(y.mean()),
            "mean_score": float(score.mean()), "n_trials": int(len(y))}


# ---------------------------------------------------------------- persistence

def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def key(**kw: Any) -> str:
    return "|".join(f"{k}={kw[k]}" for k in sorted(kw))


def run_jobs(path: Path, jobs: list[tuple[str, Any, tuple]], header: dict[str, Any]) -> dict[str, Any]:
    res = load_json(path) or {**header, "results": {}}
    todo = [(k, fn, args) for k, fn, args in jobs if k not in res["results"]]
    print(f"{path.name}: {len(jobs) - len(todo)} done, {len(todo)} to run", flush=True)
    done = 0
    gen = Parallel(n_jobs=N_JOBS, return_as="generator_unordered")(delayed(_call)(k, fn, args) for k, fn, args in todo)
    for k, r in gen:
        res["results"][k] = r
        done += 1
        if done % 8 == 0 or done == len(todo):
            path.write_text(json.dumps(res), encoding="utf-8")
            print(f"  {done}/{len(todo)}", flush=True)
    return res


def _call(k: str, fn: Any, args: tuple) -> tuple[str, Any]:
    return k, fn(*args)


def header(phase: str, D: dict[str, Any]) -> dict[str, Any]:
    return {"label": LABEL, "protocol_version": PROTOCOL_VERSION, "phase": phase, "created_at": datetime.now(UTC).isoformat(),
            "git_commit": git_commit(), "fingerprint": D["fingerprint"], "participants": list(PARTICIPANTS),
            "selection_seed": SEL_SEED, "design": "scripts/model_improvement.py docstring; docs/MODEL_IMPROVEMENT_EXPERIMENTS.md §2"}


# ---------------------------------------------------------------- P1: participant-independent

def inner_groups_p1(o: int) -> list[list[int]]:
    others = [q for q in range(len(PARTICIPANTS)) if q != o]
    perm = np.random.default_rng([SEL_SEED, o]).permutation(others)
    return [sorted(g.tolist()) for g in np.array_split(perm, 4)]


def p1_job(D: dict[str, Any], cfg: dict[str, Any], spec: dict[str, Any], o: int) -> dict[str, Any]:
    X = (D["Xn3"] if spec["norm"] == "N3" else D["X"])[:, D["fsets"][spec["features"]]]
    keep = ~D["border"] if spec.get("s1") else np.ones(len(D["y"]), bool)
    target = D["val"] if cfg["family"] == "rf_reg" else D["y"]
    seed, thr = spec["seed"], threshold(cfg)
    others = [q for q in range(len(PARTICIPANTS)) if q != o]
    if spec.get("n_train"):
        others = sorted(np.random.default_rng([SEL_SEED, o, spec["n_train"], seed]).choice(others, spec["n_train"], replace=False).tolist())

    def fit_score(train_ps: list[int], test_ps: list[int]) -> tuple[np.ndarray, np.ndarray]:
        tr, te = np.isin(D["p"], train_ps) & keep, np.isin(D["p"], test_ps) & keep
        assert not set(D["p"][tr].tolist()) & set(D["p"][te].tolist()), "participant leakage"
        model = make_model(cfg, seed).fit(X[tr], target[tr])
        return te, seg_scores(model, cfg, X[te])

    te, s = fit_score(others, [o])
    outer = {a: trial_outputs(D["t"][te], D["y"][te], s, a, thr) for a in aggs_for(cfg)}
    # majority reference: the most frequent trial class among this fold's training participants
    tr_trials = {(q, tt): D["y"][(D["p"] == q) & (D["t"] == tt) & keep][0]
                 for q in others for tt in np.unique(D["t"][(D["p"] == q) & keep])}
    maj = int(np.mean(list(tr_trials.values())) > 0.5)
    out: dict[str, Any] = {"outer": outer, "majority_class": maj, "train_participants": others}
    if spec.get("inner"):
        inner: dict[str, dict[str, float | None]] = {}
        for g in inner_groups_p1(o):
            te_i, s_i = fit_score([q for q in others if q not in g], g)
            for q in g:
                m = D["p"][te_i] == q
                inner[PARTICIPANTS[q]] = {a: metrics(**{k: np.array(v) for k, v in trial_outputs(
                    D["t"][te_i][m], D["y"][te_i][m], s_i[m], a, thr).items() if k in ("y", "score", "pred")})["balanced_accuracy"]
                    for a in aggs_for(cfg)}
        out["inner"] = inner
    return out


def p1_specs() -> list[tuple[dict[str, Any], dict[str, Any]]]:
    specs = []
    for norm in ("N0", "N3"):
        for cfg in CONFIGS:
            for seed in ((42,) if cfg["family"] in DETERMINISTIC else SEEDS):
                specs.append((cfg, {"study": "main", "norm": norm, "features": "full", "seed": seed, "inner": True}))
    rf = CONFIGS[0]
    for seed in (45, 46):
        specs.append((rf, {"study": "main", "norm": "N0", "features": "full", "seed": seed, "inner": False}))
    for fs in ABLATIONS:
        for seed in SEEDS:
            specs.append((rf, {"study": "ablation", "norm": "N0", "features": fs, "seed": seed}))
    reg = {"name": "rf_reg", "family": "rf_reg"}
    for norm in ("N0", "N3"):
        for seed in SEEDS:
            specs.append((reg, {"study": "regression", "norm": norm, "features": "full", "seed": seed}))
            specs.append((rf, {"study": "s1", "norm": norm, "features": "full", "seed": seed, "s1": True}))
    for n in (3, 6, 9):
        for seed in SEEDS:
            specs.append((rf, {"study": "learning_curve", "norm": "N0", "features": "full", "seed": seed, "n_train": n}))
    return specs


def phase_p1(out: Path) -> None:
    D = load_data()
    jobs = [(key(cfg=cfg["name"], o=o, **spec), p1_job, (D, cfg, spec, o))
            for cfg, spec in p1_specs() for o in range(len(PARTICIPANTS))]
    run_jobs(out / "p1.json", jobs, header("p1", D))


# ---------------------------------------------------------------- P2: within-participant

def p2_job(D: dict[str, Any], cfg: dict[str, Any], spec: dict[str, Any], pi: int, k: int) -> dict[str, Any]:
    rows = D["p"] == pi
    X = D["X"][rows][:, D["fsets"][spec["features"]]]
    t, y, val = D["t"][rows], D["y"][rows], D["val"][rows]
    target = val if cfg["family"] == "rf_reg" else y
    seed, thr = spec["seed"], threshold(cfg)
    border = sorted(set(t[D["border"][rows]].tolist()))
    test_v = lvo.video_folds(seed)[k]
    train_v = sorted(set(range(40)) - set(test_v))
    if spec.get("n_cal"):
        train_v = wpc.lc_subsets(seed, k, train_v)[spec["n_cal"]]
    if spec.get("s1"):
        train_v = [v for v in train_v if v not in border]
        test_v = [v for v in test_v if v not in border]
    if not test_v:
        return {"outer": None, "reason": "no test trials"}

    def fit_score(trv: list[int], tev: list[int]) -> tuple[np.ndarray, np.ndarray] | None:
        tr, te = np.isin(t, trv), np.isin(t, tev)
        assert not set(t[tr].tolist()) & set(t[te].tolist()), "a video is in both train and test"
        if cfg["family"] != "rf_reg" and len(set(y[tr].tolist())) < 2:
            return None
        model = make_model(cfg, seed).fit(X[tr], target[tr])
        return te, seg_scores(model, cfg, X[te])

    r = fit_score(train_v, test_v)
    if r is None:
        return {"outer": None, "reason": "single-class training set"}
    te, s = r
    out: dict[str, Any] = {"outer": {a: trial_outputs(t[te], y[te], s, a, thr) for a in aggs_for(cfg)},
                           "train_videos": train_v, "test_videos": test_v}
    if spec.get("inner"):
        perm = np.random.default_rng([SEL_SEED, pi, seed, k]).permutation(train_v)
        parts: dict[str, list[dict[str, list]]] = {a: [] for a in aggs_for(cfg)}
        valid = True
        for g in np.array_split(perm, 4):
            ri = fit_score([v for v in train_v if v not in g], sorted(g.tolist()))
            if ri is None:
                valid = False
                break
            te_i, s_i = ri
            for a in aggs_for(cfg):
                parts[a].append(trial_outputs(t[te_i], y[te_i], s_i, a, thr))
        out["inner"] = ({a: metrics(np.concatenate([q["y"] for q in parts[a]]), np.concatenate([q["score"] for q in parts[a]]),
                                    np.concatenate([q["pred"] for q in parts[a]]))["balanced_accuracy"] for a in parts}
                        if valid else None)
    return out


def p2_specs() -> list[tuple[dict[str, Any], dict[str, Any]]]:
    specs = []
    for cfg in CONFIGS:
        for seed in ((42, 43, 44) if cfg["family"] in DETERMINISTIC else SEEDS):  # video folds depend on the seed
            specs.append((cfg, {"study": "main", "features": "full", "seed": seed, "inner": True}))
    rf = CONFIGS[0]
    for seed in (45, 46):
        specs.append((rf, {"study": "main", "features": "full", "seed": seed, "inner": False}))
    for fs in ABLATIONS:
        for seed in SEEDS:
            specs.append((rf, {"study": "ablation", "features": fs, "seed": seed}))
    reg = {"name": "rf_reg", "family": "rf_reg"}
    for seed in SEEDS:
        specs.append((reg, {"study": "regression", "features": "full", "seed": seed}))
        specs.append((rf, {"study": "s1", "features": "full", "seed": seed, "s1": True}))
        for n in (8, 16, 24):
            specs.append((rf, {"study": "calibration_size", "features": "full", "seed": seed, "n_cal": n}))
    return specs


def phase_p2(out: Path) -> None:
    D = load_data()
    jobs = [(key(cfg=cfg["name"], pi=pi, k=k, **spec), p2_job, (D, cfg, spec, pi, k))
            for cfg, spec in p2_specs() for pi in range(len(PARTICIPANTS)) for k in range(5)]
    run_jobs(out / "p2.json", jobs, header("p2", D))


# ---------------------------------------------------------------- P3: population + personal calibration

def p3_job(D: dict[str, Any], spec: dict[str, Any], pi: int, k: int) -> dict[str, Any]:
    seed, n = spec["seed"], spec["n_cal"]
    fold = lvo.video_folds(seed)[k]
    train_v = sorted(set(range(40)) - set(fold))
    cal_v = wpc.lc_subsets(seed, k, train_v)[n] if n else []
    X = per_participant_z(D["X"], D["p"], ~np.isin(D["t"], fold))  # unlabelled, non-test videos only
    others = (D["p"] != pi) & ~np.isin(D["t"], fold)
    own = (D["p"] == pi) & np.isin(D["t"], cal_v)
    te = (D["p"] == pi) & np.isin(D["t"], fold)
    assert not np.any(np.isin(D["t"][others | own], fold)), "a test video is in training"
    tr = others | own
    w = np.ones(len(D["y"]))
    if n and spec["weighting"] == "equal":
        w[own] = others.sum() / own.sum()
    cfg = CONFIGS[0]
    model = make_model(cfg, seed).fit(X[tr], D["y"][tr], clf__sample_weight=w[tr])
    s = seg_scores(model, cfg, X[te])
    return {"outer": {"mean": trial_outputs(D["t"][te], D["y"][te], s, "mean", 0.5)}, "cal_videos": cal_v}


def p3_specs() -> list[dict[str, Any]]:
    specs = [{"seed": s, "n_cal": n, "weighting": "equal"} for s in SEEDS for n in (0, 8, 16, 32)]
    return specs + [{"seed": s, "n_cal": 32, "weighting": "none"} for s in SEEDS]


def phase_p3(out: Path) -> None:
    D = load_data()
    jobs = [(key(pi=pi, k=k, **spec), p3_job, (D, spec, pi, k)) for spec in p3_specs()
            for pi in range(len(PARTICIPANTS)) for k in range(5)]
    run_jobs(out / "p3.json", jobs, header("p3", D))


# ---------------------------------------------------------------- leakage demonstration

def leak_job(D: dict[str, Any], pi: int) -> dict[str, Any]:
    rows = D["p"] == pi
    X, y, t = D["X"][rows], D["y"][rows], D["t"][rows]
    s = np.full(len(y), np.nan)
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=42).split(X, y):  # segments, NOT trials: leaky
        model = make_model(CONFIGS[0], 42).fit(X[tr], y[tr])
        s[te] = seg_scores(model, CONFIGS[0], X[te])
    seg = metrics(y, s, (s > 0.5).astype(int))
    tro = trial_outputs(t, y, s, "mean", 0.5)
    return {"segment": seg, "trial": metrics(tro["y"], tro["score"], tro["pred"])}


def phase_leak(out: Path) -> None:
    D = load_data()
    run_jobs(out / "leak.json", [(key(pi=pi), leak_job, (D, pi)) for pi in range(len(PARTICIPANTS))], header("leak", D))


# ---------------------------------------------------------------- baseline reproduction

def phase_baseline(out: Path) -> None:
    import joblib

    s = get_settings()
    version = json.loads((s.model_dir / "active_model.json").read_text(encoding="utf-8"))["version"]
    meta = json.loads((s.model_dir / version / "metadata.json").read_text(encoding="utf-8"))
    rows = list(csv.DictReader((s.predictions_dir / f"{version}_test_predictions.csv").open(encoding="utf-8")))
    split = meta["split"]
    parts = {r: set(split[r]) for r in ("train", "validation", "test")}
    assert not (parts["train"] & parts["test"]) and not (parts["validation"] & parts["test"]) and not (parts["train"] & parts["validation"])
    test_ids = sorted({r["participant"] for r in rows})
    assert set(test_ids) == parts["test"] and not set(test_ids) & FROZEN_TEST_PARTICIPANTS
    y = np.array([r["true_label"] == "high_valence" for r in rows], int)
    ph = np.array([float(r["p_high_valence"]) for r in rows])
    pid = np.array([r["participant"] for r in rows])
    tri = np.array([int(r["trial"]) for r in rows])
    seg_m = metrics(y, ph, (ph > 0.5).astype(int))
    per, pooled = {}, {"y": [], "score": [], "pred": []}
    for q in test_ids:
        m = pid == q
        to = trial_outputs(tri[m], y[m], ph[m], "mean", 0.5)
        per[q] = metrics(to["y"], to["score"], to["pred"])
        for k2 in pooled:
            pooled[k2] += to[k2]
    trial_m = metrics(*(np.array(pooled[k2]) for k2 in ("y", "score", "pred")))
    # re-score the saved model on the cached features of the same rows
    D = load_data()
    model = joblib.load(s.model_dir / version / "model.joblib")
    lookup = {(PARTICIPANTS[a], b, c): i for i, (a, b, c) in enumerate(zip(D["p"], D["t"], wp.load()[0].segment, strict=True))}
    idx = np.array([lookup[(r["participant"], int(r["trial"]), int(r["segment"]))] for r in rows])
    proba = model.predict_proba(D["X"][idx].astype(np.float32))
    col = [str(c) for c in model.classes_].index("high_valence")
    diff = float(np.abs(proba[:, col] - ph).max())
    res = {"label": "historical, pre-protocol fixed split (s01, s09 and s11 are development participants now)",
           "model_version": version, "split": {k2: sorted(v) for k2, v in parts.items()},
           "participants_disjoint": True, "segment_pooled": seg_m, "trial_pooled": trial_m, "trial_per_participant": per,
           "trial_balanced_accuracy_participant_mean": mean_ci([per[q]["balanced_accuracy"] for q in test_ids]),
           "saved_vs_recomputed_max_abs_p_high_difference": diff, "n_segments": len(rows),
           "saved_metrics_file": {k2: meta["test"][k2] for k2 in ("segment", "trial")}}
    (out / "baseline.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(json.dumps({k2: res[k2] for k2 in ("trial_pooled", "segment_pooled", "saved_vs_recomputed_max_abs_p_high_difference")}, indent=1))


# ---------------------------------------------------------------- analysis helpers

def per_participant_from_trials(trials_by_p: dict[str, dict[str, list]]) -> dict[str, dict[str, Any]]:
    return {q: metrics(v["y"], v["score"], v["pred"]) for q, v in trials_by_p.items()}


def seed_average(per_seed: dict[int, dict[str, dict[str, Any]]]) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for q in PARTICIPANTS:
        vals = [per_seed[s][q] for s in per_seed if q in per_seed[s]]
        if not vals:
            continue
        out[q] = {m: float(np.mean([v[m] for v in vals if v[m] is not None]))
                  for m in ("balanced_accuracy", "f1_macro", "accuracy", "roc_auc", "pred_high_share", "true_high_share", "mean_score")
                  if any(v[m] is not None for v in vals)}
    return out


def summarize(per_seed: dict[int, dict[str, dict[str, Any]]]) -> dict[str, Any]:
    avg = seed_average(per_seed)
    summary = {m: mean_ci([avg[q][m] for q in avg if m in avg[q]]) for m in ("balanced_accuracy", "f1_macro", "accuracy", "roc_auc")}
    seed_means = [float(np.mean([v["balanced_accuracy"] for v in per_seed[s].values() if v["balanced_accuracy"] is not None]))
                  for s in per_seed]
    single = [sum(1 for v in per_seed[s].values() if v["pred_high_share"] < 0.05 or v["pred_high_share"] > 0.95) for s in per_seed]
    ps = [avg[q]["mean_score"] for q in avg]
    th = [avg[q]["true_high_share"] for q in avg]
    rho = sps.spearmanr(ps, th).statistic if len(set(ps)) > 1 else None
    return {"n_participants": len(avg), "seeds": sorted(per_seed), "metrics": summary,
            "across_seed_sd_of_mean_ba": float(np.std(seed_means, ddof=1)) if len(seed_means) > 1 else 0.0,
            "seed_mean_ba": dict(zip([str(s) for s in per_seed], seed_means, strict=True)),
            "single_label_participants_mean_over_seeds": float(np.mean(single)),
            "offset_tracking_spearman": None if rho is None else float(rho),
            "per_participant": avg}


def paired(a: dict[str, Any], b: dict[str, Any], metric: str = "balanced_accuracy") -> dict[str, Any]:
    qs = [q for q in PARTICIPANTS if q in a["per_participant"] and q in b["per_participant"]]
    d = np.array([a["per_participant"][q][metric] - b["per_participant"][q][metric] for q in qs])
    rng = np.random.default_rng(0)
    boots = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(10000)]
    return {"n": len(qs), "mean_difference": float(d.mean()), "bootstrap_ci": [float(np.quantile(boots, .025)), float(np.quantile(boots, .975))],
            "wilcoxon_p": wilcoxon(d, 0.0)["p_value"], "n_better": int((d > 0).sum()), "n_worse": int((d < 0).sum())}


def holm(pvals: list[float]) -> list[float]:
    order = np.argsort(pvals)
    adj, running = [0.0] * len(pvals), 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(pvals) - rank) * pvals[i]))
        adj[i] = running
    return adj


def _collect(per: dict[int, dict[str, dict[str, list]]], seed: int, q: str, to: dict[str, list]) -> None:
    acc = per.setdefault(seed, {}).setdefault(q, {"y": [], "score": [], "pred": []})
    for k2 in acc:
        acc[k2] += to[k2]


def finalize(per: dict[int, dict[str, dict[str, list]]]) -> dict[str, Any]:
    return summarize({s: per_participant_from_trials(v) for s, v in per.items()})


# ---------------------------------------------------------------- report

def analyse_p1(res: dict[str, Any]) -> dict[str, Any]:
    R = res["results"]

    def get(cfg: str, o: int, **spec: Any) -> dict[str, Any]:
        return R[key(cfg=cfg, o=o, **spec)]

    def main_spec(norm: str, cfg: dict[str, Any], seed: int) -> dict[str, Any]:
        s = 42 if cfg["family"] in DETERMINISTIC else seed
        return {"study": "main", "norm": norm, "features": "full", "seed": s, "inner": True}

    def fixed(cfg_name: str, agg: str, seeds: tuple, **spec: Any) -> dict[str, Any]:
        per: dict = {}
        for s in seeds:
            for o, q in enumerate(PARTICIPANTS):
                _collect(per, s, q, get(cfg_name, o, seed=s, **spec)["outer"][agg])
        return finalize(per)

    def nested(norm: str, cands: list[dict[str, Any]]) -> tuple[dict[str, Any], dict[str, int]]:
        per: dict = {}
        chosen: dict[str, int] = {}
        for s in SEEDS:
            for o, q in enumerate(PARTICIPANTS):
                best, best_v = None, -1.0
                for cfg in cands:
                    r = get(cfg["name"], o, **main_spec(norm, cfg, s))
                    for a in AGGS:
                        v = float(np.mean([r["inner"][p2][a] for p2 in r["inner"]]))
                        if v > best_v + 1e-12:
                            best, best_v = (cfg, a, r), v
                cfg, a, r = best  # type: ignore[misc]
                chosen[f"{cfg['name']}+{a}"] = chosen.get(f"{cfg['name']}+{a}", 0) + 1
                _collect(per, s, q, r["outer"][a])
        return finalize(per), chosen

    per: dict = {}
    for s in REF_SEEDS:
        for o, q in enumerate(PARTICIPANTS):
            r = get("rf", o, study="main", norm="N0", features="full", seed=s, inner=s in SEEDS)
            _collect(per, s, q, r["outer"]["mean"])
    return {"reference_rf_5seeds": finalize(per)} | _analyse_p1_rest(R, get, fixed, nested)


def _analyse_p1_rest(R: dict[str, Any], get: Any, fixed: Any, nested: Any) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for norm in ("N0", "N3"):
        ref = fixed("rf", "mean", SEEDS, study="main", norm=norm, features="full", inner=True)
        block: dict[str, Any] = {"rf_reference_mean_agg": ref, "families": {}, "aggregation_rf": {}}
        for fam in ("rf", "xgb", "lr", "svm"):
            summ, chosen = nested(norm, [c for c in CONFIGS if c["family"] == fam])
            block["families"][fam] = {"summary": summ, "chosen": chosen, "vs_rf_reference": paired(summ, ref)}
        summ, chosen = nested(norm, CONFIGS)
        block["nested_all"] = {"summary": summ, "chosen": chosen, "vs_rf_reference": paired(summ, ref),
                               "selection_effort": f"{len(CONFIGS)} model configurations x {len(AGGS)} aggregations"}
        for a in AGGS:
            block["aggregation_rf"][a] = fixed("rf", a, SEEDS, study="main", norm=norm, features="full", inner=True)
        # majority reference (trial balanced accuracy is 0.5 by construction)
        per: dict = {}
        for o, q in enumerate(PARTICIPANTS):
            r = get("rf", o, study="main", norm=norm, features="full", seed=42, inner=True)
            to = r["outer"]["mean"]
            _collect(per, 42, q, {"y": to["y"], "score": [float(r["majority_class"])] * len(to["y"]),
                                  "pred": [r["majority_class"]] * len(to["y"])})
        block["majority_reference"] = finalize(per)
        out[norm] = block
    ref0 = out["N0"]["rf_reference_mean_agg"]
    out["N3_vs_N0_rf"] = paired(out["N3"]["rf_reference_mean_agg"], ref0)
    out["ablation"] = {}
    for fs in ABLATIONS:
        s = fixed("rf", "mean", SEEDS, study="ablation", norm="N0", features=fs)
        out["ablation"][fs] = {"summary": s, "vs_full": paired(s, ref0)}
    out["regression"] = {}
    out["s1"] = {}
    for norm in ("N0", "N3"):
        reg = fixed("rf_reg", "mean", SEEDS, study="regression", norm=norm, features="full")
        out["regression"][norm] = {"summary": reg, "vs_classifier": paired(reg, out[norm]["rf_reference_mean_agg"])}
        out["s1"][norm] = fixed("rf", "mean", SEEDS, study="s1", norm=norm, features="full", s1=True)
    out["learning_curve"] = {str(n): fixed("rf", "mean", SEEDS, study="learning_curve", norm="N0", features="full", n_train=n)
                             for n in (3, 6, 9)}
    out["learning_curve"]["11"] = ref0
    return out


def analyse_p2(res: dict[str, Any]) -> dict[str, Any]:
    R = res["results"]

    def get(cfg: str, pi: int, k: int, **spec: Any) -> dict[str, Any]:
        return R[key(cfg=cfg, pi=pi, k=k, **spec)]

    def fixed(cfg_name: str, agg: str, seeds: tuple, **spec: Any) -> dict[str, Any]:
        per: dict = {}
        invalid = 0
        for s in seeds:
            for pi, q in enumerate(PARTICIPANTS):
                rs = [get(cfg_name, pi, k, seed=s, **spec) for k in range(5)]
                if any(r["outer"] is None and r.get("reason") == "single-class training set" for r in rs):
                    invalid += 1
                    continue
                for r in rs:
                    if r["outer"] is not None:
                        _collect(per, s, q, r["outer"][agg])
        summ = finalize(per)
        summ["invalid_participant_seeds"] = invalid
        return summ

    def nested(cands: list[dict[str, Any]]) -> tuple[dict[str, Any], dict[str, int]]:
        per: dict = {}
        chosen: dict[str, int] = {}
        for s in SEEDS:
            for pi, q in enumerate(PARTICIPANTS):
                for k in range(5):
                    best, best_v = None, -1.0
                    for cfg in cands:
                        r = get(cfg["name"], pi, k, study="main", features="full", seed=s, inner=True)
                        if r["inner"] is None:
                            continue
                        for a in AGGS:
                            if r["inner"][a] is not None and r["inner"][a] > best_v + 1e-12:
                                best, best_v = (cfg, a, r), r["inner"][a]
                    if best is None:  # every candidate had a single-class inner fold: fall back to the reference
                        best = (CONFIGS[0], "mean", get("rf", pi, k, study="main", features="full", seed=s, inner=True))
                    cfg, a, r = best
                    chosen[f"{cfg['name']}+{a}"] = chosen.get(f"{cfg['name']}+{a}", 0) + 1
                    _collect(per, s, q, r["outer"][a])
        return finalize(per), chosen

    out: dict[str, Any] = {"reference_rf_5seeds": None}
    per: dict = {}
    for s in REF_SEEDS:
        for pi, q in enumerate(PARTICIPANTS):
            for k in range(5):
                _collect(per, s, q, get("rf", pi, k, study="main", features="full", seed=s, inner=s in SEEDS)["outer"]["mean"])
    out["reference_rf_5seeds"] = finalize(per)
    ref = fixed("rf", "mean", SEEDS, study="main", features="full", inner=True)
    out["rf_reference_mean_agg"] = ref
    out["families"] = {}
    for fam in ("rf", "xgb", "lr", "svm"):
        summ, chosen = nested([c for c in CONFIGS if c["family"] == fam])
        out["families"][fam] = {"summary": summ, "chosen": chosen, "vs_rf_reference": paired(summ, ref)}
    for cfg in CONFIGS:
        out.setdefault("fixed_configs_mean_agg", {})[cfg["name"]] = fixed(cfg["name"], "mean", SEEDS, study="main", features="full", inner=True)
    summ, chosen = nested(CONFIGS)
    out["nested_all"] = {"summary": summ, "chosen": chosen, "vs_rf_reference": paired(summ, ref),
                         "selection_effort": f"{len(CONFIGS)} model configurations x {len(AGGS)} aggregations"}
    out["aggregation_rf"] = {a: fixed("rf", a, SEEDS, study="main", features="full", inner=True) for a in AGGS}
    out["ablation"] = {}
    for fs in ABLATIONS:
        s2 = fixed("rf", "mean", SEEDS, study="ablation", features=fs)
        out["ablation"][fs] = {"summary": s2, "vs_full": paired(s2, ref)}
    reg = fixed("rf_reg", "mean", SEEDS, study="regression", features="full")
    out["regression"] = {"summary": reg, "vs_classifier": paired(reg, ref)}
    out["s1"] = fixed("rf", "mean", SEEDS, study="s1", features="full", s1=True)
    out["calibration_size"] = {str(n): fixed("rf", "mean", SEEDS, study="calibration_size", features="full", n_cal=n)
                               for n in (8, 16, 24)}
    out["calibration_size"]["32"] = ref
    # label-noise diagnostic: reference-model correctness on borderline vs clear trials
    D_val = res["valence"]
    corr_b, corr_c, dist, correct = [], [], [], []
    per_q = {}
    for pi, q in enumerate(PARTICIPANTS):
        b_ok, c_ok = [], []
        for s in SEEDS:
            for k in range(5):
                to = get("rf", pi, k, study="main", features="full", seed=s, inner=True)["outer"]["mean"]
                for tr, yy, pr in zip(to["trials"], to["y"], to["pred"], strict=True):
                    v = D_val[q][tr]
                    ok = int(yy == pr)
                    (b_ok if BORDERLINE[0] <= v <= BORDERLINE[1] else c_ok).append(ok)
                    dist.append(abs(v - 5.0))
                    correct.append(ok)
        per_q[q] = {"borderline_acc": float(np.mean(b_ok)) if b_ok else None, "clear_acc": float(np.mean(c_ok)),
                    "n_borderline_trials": len(b_ok) // len(SEEDS)}
        corr_b += b_ok
        corr_c += c_ok
    rb = sps.pointbiserialr(correct, dist)
    out["label_noise"] = {"pooled_borderline_accuracy": float(np.mean(corr_b)), "pooled_clear_accuracy": float(np.mean(corr_c)),
                          "pointbiserial_distance_from_threshold_vs_correct": {"r": float(rb.statistic), "p": float(rb.pvalue)},
                          "per_participant": per_q, "note": "trial accuracy (not balanced) pooled over seeds 42-44"}
    return out


def analyse_p3(res: dict[str, Any], p2: dict[str, Any], p1: dict[str, Any]) -> dict[str, Any]:
    R = res["results"]
    out: dict[str, Any] = {}
    for n, wt in ((0, "equal"), (8, "equal"), (16, "equal"), (32, "equal"), (32, "none")):
        per: dict = {}
        for s in SEEDS:
            for pi, q in enumerate(PARTICIPANTS):
                for k in range(5):
                    _collect(per, s, q, R[key(pi=pi, k=k, seed=s, n_cal=n, weighting=wt)]["outer"]["mean"])
        out[f"{n}_{wt}"] = finalize(per)
    z = out["0_equal"]
    comp = {}
    for n in (8, 16, 32):
        personal = p2["calibration_size"][str(n)]
        comp[str(n)] = {"pooled_vs_zero_shot": paired(out[f"{n}_equal"], z), "personal_only_vs_zero_shot": paired(personal, z),
                        "pooled_vs_personal_only": paired(out[f"{n}_equal"], personal)}
    comp["32_unweighted_vs_zero_shot"] = paired(out["32_none"], z)
    out["comparisons"] = comp
    return out


def phase_report(out: Path) -> None:
    D = load_data()
    valence = {q: {int(tt): float(D["val"][(D["p"] == i) & (D["t"] == tt)][0]) for tt in range(40)} for i, q in enumerate(PARTICIPANTS)}
    p1 = analyse_p1(load_json(out / "p1.json"))
    p2r = load_json(out / "p2.json")
    p2r["valence"] = valence
    p2 = analyse_p2(p2r)
    p3 = analyse_p3(load_json(out / "p3.json"), p2, p1)
    leak = load_json(out / "leak.json")
    leak_s = {"segment_accuracy": mean_ci([r["segment"]["accuracy"] for r in leak["results"].values()]),
              "segment_balanced_accuracy": mean_ci([r["segment"]["balanced_accuracy"] for r in leak["results"].values()]),
              "trial_balanced_accuracy": mean_ci([r["trial"]["balanced_accuracy"] for r in leak["results"].values()])}
    prim = {"P1_nested_vs_rf": p1["N0"]["nested_all"]["vs_rf_reference"], "P2_nested_vs_rf": p2["nested_all"]["vs_rf_reference"],
            "P3_pooled32_vs_zero_shot": p3["comparisons"]["32"]["pooled_vs_zero_shot"]}
    adj = holm([v["wilcoxon_p"] if v["wilcoxon_p"] is not None else 1.0 for v in prim.values()])
    for (k2, v), a in zip(prim.items(), adj, strict=True):
        v["holm_p"] = a
    summary = {"label": LABEL, "protocol_version": PROTOCOL_VERSION, "created_at": datetime.now(UTC).isoformat(),
               "git_commit": git_commit(), "fingerprint": D["fingerprint"], "participants": list(PARTICIPANTS),
               "baseline": load_json(out / "baseline.json"), "p1": p1, "p2": p2, "p3": p3, "leakage_demo": leak_s,
               "primary_comparisons": prim, "candidate": load_json(out / "candidate" / "metadata.json")}
    (out / "summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    DOCS_OUT.mkdir(parents=True, exist_ok=True)
    (DOCS_OUT / "summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print("wrote", out / "summary.json", "and", DOCS_OUT / "summary.json")


# ---------------------------------------------------------------- candidate artifact

def phase_candidate(out: Path) -> None:
    """Run the same inner selection on all 12 participants and save the selected pipeline as an isolated candidate."""
    import joblib

    D = load_data()
    perm = np.random.default_rng([SEL_SEED, 99]).permutation(len(PARTICIPANTS))
    groups = [sorted(g.tolist()) for g in np.array_split(perm, 4)]
    scores: dict[str, float] = {}
    for cfg in CONFIGS:
        for seed in ((42,) if cfg["family"] in DETERMINISTIC else (42,)):
            ba: dict[str, list[float]] = {a: [] for a in AGGS}
            for g in groups:
                tr, te = ~np.isin(D["p"], g), np.isin(D["p"], g)
                model = make_model(cfg, seed).fit(D["X"][tr], D["y"][tr])
                s = seg_scores(model, cfg, D["X"][te])
                for q in g:
                    m = D["p"][te] == q
                    for a in AGGS:
                        to = trial_outputs(D["t"][te][m], D["y"][te][m], s[m], a, 0.5)
                        ba[a].append(metrics(to["y"], to["score"], to["pred"])["balanced_accuracy"])
            for a in AGGS:
                scores[f"{cfg['name']}+{a}"] = float(np.mean(ba[a]))
            print(cfg["name"], {a: round(scores[f"{cfg['name']}+{a}"], 4) for a in AGGS}, flush=True)
    best = max(scores, key=lambda k2: (scores[k2], -list(scores).index(k2)))
    name, agg = best.split("+")
    cfg = next(c for c in CONFIGS if c["name"] == name)
    model = make_model(cfg, 42).fit(D["X"], D["y"])
    cdir = out / "candidate"
    cdir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, cdir / "model.joblib")
    meta = {"status": "EXPERIMENTAL CANDIDATE - not registered, not active, not evaluated on any independent test",
            "label": LABEL, "selected": {"config": cfg, "aggregation": agg, "seed": 42}, "inner_scores": scores,
            "inner_groups": [[PARTICIPANTS[i] for i in g] for g in groups], "trained_on": list(PARTICIPANTS),
            "normalization": "N0 (StandardScaler fitted on the 12 development participants)",
            "threshold": "0.5 on the aggregated trial score (0 for log-odds)", "fingerprint": D["fingerprint"],
            "created_at": datetime.now(UTC).isoformat(), "git_commit": git_commit(),
            "expected_performance": "see summary.json p1.N0.nested_all (nested LOPO estimate of this selection procedure)"}
    (cdir / "metadata.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    print("candidate:", best, "->", cdir)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=["baseline", "p1", "p2", "p3", "leak", "candidate", "report"], required=True)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    out = args.out or get_settings().output_dir / "experiments" / "model_improvement_20261021"
    out.mkdir(parents=True, exist_ok=True)
    {"baseline": phase_baseline, "p1": phase_p1, "p2": phase_p2, "p3": phase_p3, "leak": phase_leak,
     "candidate": phase_candidate, "report": phase_report}[args.phase](out)


if __name__ == "__main__":
    main()
