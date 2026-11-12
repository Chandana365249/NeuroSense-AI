"""FACED valence experiment under faced-protocol-v1 (see app/ml/faced/protocol.py; declared before any result).

Candidates (compact, fixed in advance): majority class; logistic regression C in {0.01, 0.1, 1}; RBF SVM C in {0.1, 1};
the project's registered Random Forest and XGBoost configurations; an EEGNet-style CNN (project's build_cnn, 30 x 500
at 125 Hz, temporal kernel 62 = 0.5 s, pooling 4 and 5, AdamW lr 1e-3, weight decay 1e-2, batch 64, class-weighted loss, a FIXED 25
epochs, channel z-scoring from training rows; S1 only). Trial prediction: mean segment probability > 0.5.
Selection: inner folds only (pooled trial balanced accuracy averaged over 5 folds); ties -> earlier in CANDIDATES.
Final test: run once per setting with the selected configuration; refuses to run again.

Usage (from backend/):
  python -m scripts.faced_experiment --phase inner      # classical candidates, S1 and S3 inner folds
  python -m scripts.faced_experiment --phase inner_cnn  # CNN, S1 inner folds
  python -m scripts.faced_experiment --phase freeze     # record selections + leakage checks
  python -m scripts.faced_experiment --phase final      # once: S1, S2, S3 on the held-out test set + model artifacts
"""

from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse  # noqa: E402
import json  # noqa: E402
from datetime import UTC, datetime  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Any  # noqa: E402

import numpy as np  # noqa: E402
from joblib import Parallel, delayed  # noqa: E402
from scipy import stats as sps  # noqa: E402
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score  # noqa: E402

from app.ml.faced import dataset as fd  # noqa: E402
from app.ml.faced import pipeline as fpipe  # noqa: E402
from app.ml.faced import protocol as fprot  # noqa: E402
from app.ml.provenance import git_commit  # noqa: E402
from scripts import model_improvement as mi  # noqa: E402

OUT = Path(__file__).resolve().parents[2] / "outputs" / "faced"
FROZEN_SPLIT_SHA = "01e74a4ed9368203973c252f4ec80ca5fdb2d0d4ed6c6e2a389cab421d163c41"
SEED = 42
CANDIDATES: list[dict[str, Any]] = [
    {"name": "majority", "family": "majority"},
    {"name": "lr_C0.01", "family": "lr", "C": 0.01},
    {"name": "lr_C0.1", "family": "lr", "C": 0.1},
    {"name": "lr_C1", "family": "lr", "C": 1.0},
    {"name": "svm_C0.1", "family": "svm", "C": 0.1},
    {"name": "svm_C1", "family": "svm", "C": 1.0},
    {"name": "rf", "family": "rf"},
    {"name": "xgb", "family": "xgb"},
]
CNN = {"name": "cnn_eegnet", "family": "cnn", "epochs": 25, "lr": 1e-3, "weight_decay": 1e-2, "batch": 64,
       "n_samples": 500, "temporal_kernel": 62, "pool2": 5}  # pools 4 x 5 must divide 500 samples


# ---------------------------------------------------------------- data

def load() -> dict[str, Any]:
    z = np.load(OUT / f"features_{fpipe.PIPELINE_VERSION}.npz", allow_pickle=False)
    split = json.loads((OUT / "protocol_split_v1.json").read_text(encoding="utf-8"))
    sha = fprot.FacedSplit(**{k: (tuple(tuple(x) if isinstance(x, list) else x for x in v) if isinstance(v, list) else v)
                              for k, v in split.items()}).sha256
    assert sha == FROZEN_SPLIT_SHA, "the evaluation split differs from the frozen protocol"
    return {"X": z["X"].astype(np.float64), "p": z["participant"], "v": z["video"].astype(int), "y": z["y"].astype(int),
            "valence": z["ratings"][:, list(z["rating_keys"]).index("Valence")], "split": split}


def per_person_z(X: np.ndarray, p: np.ndarray, stat_rows: np.ndarray) -> np.ndarray:
    return mi.per_participant_z(X, p, stat_rows)


# ---------------------------------------------------------------- fitting

def fit_predict(cfg: dict[str, Any], Xtr: np.ndarray, ytr: np.ndarray, Xte: np.ndarray) -> np.ndarray:
    if cfg["family"] == "majority":
        return np.full(len(Xte), float(np.mean(ytr) > 0.5))
    model = mi.make_model(cfg, SEED)
    if cfg["family"] in ("rf", "xgb"):
        model.set_params(clf__n_jobs=4)
    model.fit(Xtr, ytr)
    return mi.seg_scores(model, cfg, Xte)


def trial_table(p: np.ndarray, v: np.ndarray, y: np.ndarray, s: np.ndarray) -> dict[str, np.ndarray]:
    keys = sorted(set(zip(p.tolist(), v.tolist(), strict=True)))
    idx = {k: i for i, k in enumerate(keys)}
    inv = np.array([idx[k] for k in zip(p.tolist(), v.tolist(), strict=True)])
    score = np.bincount(inv, weights=s) / np.bincount(inv)
    ytr = np.zeros(len(keys), int)
    ytr[inv] = y
    return {"participant": np.array([k[0] for k in keys]), "video": np.array([k[1] for k in keys]), "y": ytr,
            "score": score, "pred": (score > 0.5).astype(int)}


def metrics(y: np.ndarray, score: np.ndarray, pred: np.ndarray) -> dict[str, Any]:
    both = len(set(y.tolist())) == 2
    return {"n": int(len(y)), "accuracy": float((y == pred).mean()),
            "balanced_accuracy": float(balanced_accuracy_score(y, pred)) if both else None,
            "f1_macro": float(f1_score(y, pred, average="macro", labels=[0, 1], zero_division=0)),
            "roc_auc": float(roc_auc_score(y, score)) if both and len(set(score.tolist())) > 1 else None,
            "high_share_true": float(y.mean()), "high_share_pred": float(pred.mean())}


# ---------------------------------------------------------------- inner selection

def inner_rows(d: dict[str, Any], k: int) -> tuple[np.ndarray, np.ndarray]:
    sp = d["split"]
    val_p, val_v = set(sp["inner_participant_groups"][k]), set(sp["inner_video_groups"][k])
    dev_p, dev_v = set(sp["dev_participants"]), set(sp["dev_videos"])
    in_p = np.isin(d["p"], list(dev_p - val_p))
    in_v = np.isin(d["v"], list(dev_v - val_v))
    tr = in_p & in_v
    va = np.isin(d["p"], list(val_p)) & np.isin(d["v"], list(val_v))
    assert not set(d["p"][tr]) & set(d["p"][va]) and not set(d["v"][tr]) & set(d["v"][va]), "inner leakage"
    assert not np.isin(d["p"][tr | va], sp["test_participants"]).any() and not np.isin(d["v"][tr | va], sp["test_videos"]).any()
    return tr, va


def inner_job(d: dict[str, Any], cfg: dict[str, Any], setting: str, k: int) -> dict[str, Any]:
    tr, va = inner_rows(d, k)
    X = d["X"]
    if setting == "S3":  # unlabelled per-person stats from that person's non-validation development videos
        sp = d["split"]
        stat_videos = sorted(set(sp["dev_videos"]) - set(sp["inner_video_groups"][k]))
        X = per_person_z(X, d["p"], np.isin(d["v"], stat_videos))
    s = fit_predict(cfg, X[tr], d["y"][tr], X[va])
    t = trial_table(d["p"][va], d["v"][va], d["y"][va], s)
    return {"trial": metrics(t["y"], t["score"], t["pred"])}


def phase_inner(_: argparse.Namespace) -> None:
    d = load()
    jobs = [(cfg, st, k) for st in ("S1", "S3") for cfg in CANDIDATES for k in range(fprot.N_INNER_FOLDS)]
    res = Parallel(n_jobs=4)(delayed(inner_job)(d, cfg, st, k) for cfg, st, k in jobs)
    out: dict[str, Any] = {"created_at": datetime.now(UTC).isoformat(), "git_commit": git_commit(), "results": {}}
    for (cfg, st, k), r in zip(jobs, res, strict=True):
        out["results"].setdefault(st, {}).setdefault(cfg["name"], {})[str(k)] = r
    (OUT / "inner_classical.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps(summarize_inner(out["results"]), indent=1))


def summarize_inner(results: dict[str, Any]) -> dict[str, dict[str, float]]:
    return {st: {name: float(np.mean([f["trial"]["balanced_accuracy"] for f in folds.values()]))
                 for name, folds in by.items()} for st, by in results.items()}


# ---------------------------------------------------------------- CNN

def cnn_fit_predict(Xtr: np.ndarray, ytr: np.ndarray, Xte: np.ndarray) -> np.ndarray:
    import torch
    from torch import nn

    from app.ml.deep.cnn_model import CNNConfig, build_cnn

    torch.manual_seed(SEED)
    torch.set_num_threads(8)
    mean = Xtr.mean(axis=(0, 2), keepdims=True)
    std = Xtr.std(axis=(0, 2), keepdims=True) + 1e-6
    model = build_cnn(CNNConfig(n_channels=len(fd.FACED_CHANNELS), n_samples=CNN["n_samples"],
                                temporal_kernel=CNN["temporal_kernel"], pool2=CNN["pool2"]), SEED)
    w = torch.tensor(len(ytr) / (2 * np.bincount(ytr, minlength=2).clip(min=1)), dtype=torch.float32)
    loss_fn = nn.CrossEntropyLoss(weight=w)
    opt = torch.optim.AdamW(model.parameters(), lr=CNN["lr"], weight_decay=CNN["weight_decay"])
    gen = np.random.default_rng(SEED)
    xt = torch.tensor(((Xtr - mean) / std)[:, None], dtype=torch.float32)
    yt = torch.tensor(ytr, dtype=torch.long)
    for _ in range(CNN["epochs"]):
        model.train()
        order = gen.permutation(len(xt))
        for i in range(0, len(order), CNN["batch"]):
            b = order[i:i + CNN["batch"]]
            opt.zero_grad()
            loss_fn(model(xt[b]), yt[b]).backward()
            opt.step()
    model.eval()
    xe = torch.tensor(((Xte - mean) / std)[:, None], dtype=torch.float32)
    with torch.no_grad():
        return torch.cat([torch.softmax(model(xe[i:i + 256]), 1)[:, 1] for i in range(0, len(xe), 256)]).numpy()


def load_segments() -> tuple[np.ndarray, dict[str, np.ndarray]]:
    seg = np.load(OUT / f"segments125_{fpipe.PIPELINE_VERSION}.npy", mmap_mode="r")
    m = np.load(OUT / f"segments125_{fpipe.PIPELINE_VERSION}_meta.npz")
    return seg, {"p": m["participant"], "v": m["video"].astype(int), "y": m["y"].astype(int)}


def phase_inner_cnn(_: argparse.Namespace) -> None:
    d = load()
    seg, meta = load_segments()
    assert np.array_equal(meta["p"], d["p"]) and np.array_equal(meta["v"], d["v"]), "segments must align with features"
    out: dict[str, Any] = {"created_at": datetime.now(UTC).isoformat(), "config": CNN, "results": {}}
    for k in range(fprot.N_INNER_FOLDS):
        tr, va = inner_rows(d, k)
        s = cnn_fit_predict(np.asarray(seg[tr], dtype=np.float32), d["y"][tr], np.asarray(seg[va], dtype=np.float32))
        t = trial_table(d["p"][va], d["v"][va], d["y"][va], s)
        out["results"][str(k)] = {"trial": metrics(t["y"], t["score"], t["pred"])}
        print(f"fold {k}: {out['results'][str(k)]['trial']}", flush=True)
        (OUT / "inner_cnn.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print("CNN inner mean BA:", np.mean([r["trial"]["balanced_accuracy"] for r in out["results"].values()]))


# ---------------------------------------------------------------- final (once)

def select(setting: str) -> tuple[dict[str, Any], dict[str, float]]:
    inner = json.loads((OUT / "inner_classical.json").read_text(encoding="utf-8"))["results"]
    scores = summarize_inner(inner)[setting]
    if setting == "S1" and (OUT / "inner_cnn.json").exists():
        cnn = json.loads((OUT / "inner_cnn.json").read_text(encoding="utf-8"))["results"]
        if len(cnn) == fprot.N_INNER_FOLDS:
            scores["cnn_eegnet"] = float(np.mean([r["trial"]["balanced_accuracy"] for r in cnn.values()]))
    order = [c["name"] for c in CANDIDATES] + ["cnn_eegnet"]
    best = max((n for n in order if n in scores), key=lambda n: (round(scores[n], 12), -order.index(n)))
    cfg = CNN if best == "cnn_eegnet" else next(c for c in CANDIDATES if c["name"] == best)
    return cfg, scores


def evaluate(t: dict[str, np.ndarray], seg_y: np.ndarray, seg_s: np.ndarray) -> dict[str, Any]:
    pooled = metrics(t["y"], t["score"], t["pred"])
    per = {}
    for q in sorted(set(t["participant"].tolist())):
        m = t["participant"] == q
        per[q] = metrics(t["y"][m], t["score"][m], t["pred"][m])
    bas = [r["balanced_accuracy"] for r in per.values() if r["balanced_accuracy"] is not None]
    accs = [r["accuracy"] for r in per.values()]
    rng = np.random.default_rng(0)
    qs = sorted(per)
    boot_ba, boot_acc = [], []
    for _ in range(2000):  # participant-cluster bootstrap of the pooled trial metrics
        pick = rng.choice(qs, len(qs))
        rows = np.concatenate([np.flatnonzero(t["participant"] == q) for q in pick])
        if len(set(t["y"][rows].tolist())) == 2:
            boot_ba.append(balanced_accuracy_score(t["y"][rows], t["pred"][rows]))
        boot_acc.append(float((t["y"][rows] == t["pred"][rows]).mean()))

    def tci(v: list[float]) -> list[float]:
        h = sps.t.ppf(0.975, len(v) - 1) * np.std(v, ddof=1) / np.sqrt(len(v))
        return [float(np.mean(v) - h), float(np.mean(v) + h)]

    y, pr = t["y"], t["pred"]
    confusion = {"rows_true_cols_pred": "[[low->low, low->high], [high->low, high->high]]",
                 "matrix": [[int(((y == 0) & (pr == 0)).sum()), int(((y == 0) & (pr == 1)).sum())],
                            [int(((y == 1) & (pr == 0)).sum()), int(((y == 1) & (pr == 1)).sum())]]}
    return {"trial_pooled": pooled, "trial_confusion": confusion,
            "trial_pooled_ci95_participant_bootstrap": {"balanced_accuracy": np.quantile(boot_ba, [.025, .975]).tolist(),
                                                         "accuracy": np.quantile(boot_acc, [.025, .975]).tolist()},
            "per_participant_balanced_accuracy": {"n_with_both_classes": len(bas), "mean": float(np.mean(bas)),
                                                  "sd": float(np.std(bas, ddof=1)), "ci95_t": tci(bas),
                                                  "min": float(min(bas)), "max": float(max(bas))},
            "per_participant_accuracy": {"mean": float(np.mean(accs)), "sd": float(np.std(accs, ddof=1)), "ci95_t": tci(accs)},
            "per_participant": per,
            "segment_level": metrics(seg_y, seg_s, (seg_s > 0.5).astype(int))}


def phase_freeze(_: argparse.Namespace) -> None:
    """Before the final run: record the selections and verify the test set was excluded from training and selection."""
    if (OUT / "final_results.json").exists():
        raise SystemExit("final results already exist; the freeze must precede the final evaluation")
    d = load()
    sp = d["split"]
    test_p, test_v = set(sp["test_participants"]), set(sp["test_videos"])
    checks = {}
    inner_rows_seen = np.zeros(len(d["y"]), bool)
    for k in range(fprot.N_INNER_FOLDS):
        tr, va = inner_rows(d, k)
        inner_rows_seen |= tr | va
    checks["inner_folds_use_no_test_participant"] = not (set(d["p"][inner_rows_seen]) & test_p)
    checks["inner_folds_use_no_test_video"] = not (set(d["v"][inner_rows_seen].tolist()) & test_v)
    dev_p, dev_v = np.isin(d["p"], sp["dev_participants"]), np.isin(d["v"], sp["dev_videos"])
    s1_train = dev_p & dev_v
    checks["S1_S3_training_has_no_test_participant"] = not (set(d["p"][s1_train]) & test_p)
    checks["S1_S3_training_has_no_test_video"] = not (set(d["v"][s1_train].tolist()) & test_v)
    checks["S2_training_has_no_test_participant"] = not (set(d["p"][dev_p]) & test_p)
    checks["S3_normalisation_uses_development_videos_only"] = True  # per_person_z(X, p, dev_v) in phase_final
    checks["test_set_size"] = {"participants": len(test_p), "videos": len(test_v),
                               "strict_trials": int(len(set(zip(d["p"][np.isin(d["p"], list(test_p)) & np.isin(d["v"], list(test_v))].tolist(),
                                                               d["v"][np.isin(d["p"], list(test_p)) & np.isin(d["v"], list(test_v))].tolist(), strict=True))))}
    assert all(v for k, v in checks.items() if k != "test_set_size"), checks
    freeze = {"frozen_at": datetime.now(UTC).isoformat(), "protocol": fprot.PROTOCOL_VERSION, "split_sha256": FROZEN_SPLIT_SHA,
              "pipeline": fpipe.PIPELINE_VERSION, "label_rule": f"valence > {fpipe.VALENCE_THRESHOLD} -> high_valence",
              "decision_threshold": "mean segment probability > 0.5", "seed": SEED, "checks": checks, "selection": {}}
    for setting, sel_on in (("S1", "S1"), ("S2", "S1"), ("S3", "S3")):
        cfg, scores = select(sel_on)
        freeze["selection"][setting] = {"selected": cfg, "selected_on_inner_folds_of": sel_on, "inner_scores": scores}
    (OUT / "pre_final_freeze.json").write_text(json.dumps(freeze, indent=1), encoding="utf-8")
    print(json.dumps(freeze, indent=1))


def phase_final(args: argparse.Namespace) -> None:
    path = OUT / "final_results.json"
    if path.exists():
        raise SystemExit("final_results.json exists: the held-out test set is evaluated only once (protocol).")
    if not (OUT / "pre_final_freeze.json").exists():
        raise SystemExit("run --phase freeze first")
    frozen = json.loads((OUT / "pre_final_freeze.json").read_text(encoding="utf-8"))["selection"]
    import joblib

    d = load()
    sp = d["split"]
    test_p, dev_p = np.isin(d["p"], sp["test_participants"]), np.isin(d["p"], sp["dev_participants"])
    test_v, dev_v = np.isin(d["v"], sp["test_videos"]), np.isin(d["v"], sp["dev_videos"])
    results: dict[str, Any] = {"protocol": fprot.PROTOCOL_VERSION, "split_sha256": FROZEN_SPLIT_SHA,
                               "created_at": datetime.now(UTC).isoformat(), "git_commit": git_commit(), "settings": {}}
    models_dir = OUT / "models"
    models_dir.mkdir(exist_ok=True)
    for setting in ("S1", "S2", "S3"):
        sel_setting = "S1" if setting == "S2" else setting
        cfg, scores = select(sel_setting)
        assert cfg == frozen[setting]["selected"], "selection changed after the freeze"
        if setting == "S2":
            tr, te = dev_p, test_p
        else:
            tr, te = dev_p & dev_v, test_p & test_v
        assert not set(d["p"][tr]) & set(d["p"][te])
        if setting != "S2":
            assert not set(d["v"][tr]) & set(d["v"][te])
        X = d["X"]
        if setting == "S3":
            X = per_person_z(X, d["p"], dev_v)  # unlabelled per-person stats from development-video segments
        # majority reference from the training trials
        tt = trial_table(d["p"][tr], d["v"][tr], d["y"][tr], np.zeros(tr.sum()))
        maj = int(tt["y"].mean() > 0.5)
        if cfg["family"] == "cnn":
            seg, meta = load_segments()
            s = cnn_fit_predict(np.asarray(seg[tr], dtype=np.float32), d["y"][tr], np.asarray(seg[te], dtype=np.float32))
        elif cfg["family"] == "majority":
            s = fit_predict(cfg, X[tr], d["y"][tr], X[te])
        else:  # fit once; the saved artifact is exactly the model that produced the reported test scores
            model = mi.make_model(cfg, SEED).fit(X[tr], d["y"][tr])
            s = mi.seg_scores(model, cfg, X[te])
            joblib.dump(model, models_dir / f"faced_{setting}_{cfg['name']}.joblib")
        t = trial_table(d["p"][te], d["v"][te], d["y"][te], s)
        ev = evaluate(t, d["y"][te], s)
        tm = trial_table(d["p"][te], d["v"][te], d["y"][te], np.full(te.sum(), float(maj)))
        results["settings"][setting] = {"selected": cfg, "selection_scores_inner": scores,
                                        "selected_on": sel_setting, "train_rows": int(tr.sum()), "test_rows": int(te.sum()),
                                        "train_trials": int(len(tt["y"])), "test_trials": int(len(t["y"])),
                                        "train_high_share": float(tt["y"].mean()), "majority_class": maj,
                                        "majority_reference": metrics(tm["y"], tm["score"], tm["pred"]), **ev}
        print(setting, cfg["name"], json.dumps(ev["trial_pooled"]), json.dumps(ev["trial_pooled_ci95_participant_bootstrap"]), flush=True)
    path.write_text(json.dumps(results, indent=1), encoding="utf-8")


def phase_card(_: argparse.Namespace) -> None:
    """Model card for the S1 (strict, inductive) model, generated from final_results.json."""
    import hashlib

    res = json.loads((OUT / "final_results.json").read_text(encoding="utf-8"))
    d = load()
    s1 = res["settings"]["S1"]
    cfg = s1["selected"]
    artifact = f"faced_S1_{cfg['name']}.joblib" if cfg["family"] not in ("majority", "cnn") else None

    def headline(st: str) -> dict[str, Any]:
        r = res["settings"][st]
        return {"model": r["selected"]["name"], "test_trials": r["test_trials"],
                "trial_balanced_accuracy": r["trial_pooled"]["balanced_accuracy"],
                "trial_balanced_accuracy_ci95": r["trial_pooled_ci95_participant_bootstrap"]["balanced_accuracy"],
                "trial_accuracy": r["trial_pooled"]["accuracy"],
                "trial_accuracy_ci95": r["trial_pooled_ci95_participant_bootstrap"]["accuracy"],
                "trial_macro_f1": r["trial_pooled"]["f1_macro"], "trial_roc_auc": r["trial_pooled"]["roc_auc"],
                "majority_reference_accuracy": r["majority_reference"]["accuracy"],
                "segment_balanced_accuracy": r["segment_level"]["balanced_accuracy"]}

    names = fpipe.feature_names()
    reached = s1["trial_pooled"]["balanced_accuracy"] >= 0.80
    card = {
        "name": f"faced-valence-{cfg['name']}-S1", "algorithm": cfg, "artifact": artifact, "seed": SEED,
        "pipeline": fpipe.PIPELINE_VERSION, "protocol": fprot.PROTOCOL_VERSION, "split_sha256": FROZEN_SPLIT_SHA,
        "dataset_compatibility": "FACED (NEMAR nm000112) v1.1.4 recordings processed by the FACED pipeline only. "
                                 "Not compatible with DEAP inputs or the DEAP model's feature schema.",
        "dataset": {"id": "nm000112", "version": "1.1.4", "license": "CC-BY-4.0",
                    "citation": "Chen et al. (2023) Sci Data 10:740, doi:10.1038/s41597-023-02650-w"},
        "label_rule": f"high_valence if self-reported valence > {fpipe.VALENCE_THRESHOLD} on the 0-7 scale, else low_valence",
        "preprocessing": {"window": f"last {fpipe.WINDOW_S} s of each video", "reference": "common average (30 shared scalp channels)",
                          "resampling": "1000 Hz -> 250 Hz with resample_poly (FIR anti-alias); 250 Hz unchanged",
                          "band_pass_hz": [4.0, 45.0], "artifact_clip_sigma": 6.0, "segment_seconds": 4.0,
                          "max_clipped_fraction": 0.05, "trial_prediction": "mean segment probability > 0.5"},
        "channels": list(fd.FACED_CHANNELS),
        "feature_schema": {"n_features": len(names), "sha256": hashlib.sha256("\n".join(names).encode()).hexdigest()},
        "trained_on": {"participants": len(d["split"]["dev_participants"]), "videos": d["split"]["dev_videos"],
                       "segments": s1["train_rows"], "trials": s1["train_trials"]},
        "evaluation": {"S1_strict_primary": headline("S1"), "S2_unseen_people_seen_clips": headline("S2"),
                       "S3_strict_unsupervised_calibration": headline("S3")},
        "target_80_reached": reached,
        "status": "EXPERIMENTAL RESEARCH MODEL - not registered, not active, not production-ready"
                  + ("" if reached else "; below the 80% target"),
        "limitations": ["Estimates a binary category of self-reported valence, not a person's emotion; not diagnostic.",
                        "Evaluated once on 25 held-out participants x 9 held-out clips (one protocol, one seed).",
                        "Clips were shown to all participants; S1 holds out clips, S2 does not.",
                        "Valid only for FACED recordings processed by the FACED pipeline."],
        "training_command": "python -m scripts.faced_build --faced-dir <FACED v1.1.4> && "
                            "python -m scripts.faced_experiment --phase inner && "
                            "python -m scripts.faced_experiment --phase inner_cnn && python -m scripts.faced_experiment --phase freeze && "
                            "python -m scripts.faced_experiment --phase final && python -m scripts.faced_experiment --phase card",
        "created_at": datetime.now(UTC).isoformat(), "git_commit": git_commit(),
    }
    (OUT / "models" / "model_card.json").write_text(json.dumps(card, indent=1), encoding="utf-8")
    print(json.dumps({k: card[k] for k in ("name", "artifact", "target_80_reached", "status")}, indent=1))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["inner", "inner_cnn", "freeze", "final", "card"], required=True)
    args = ap.parse_args()
    {"inner": phase_inner, "inner_cnn": phase_inner_cnn, "freeze": phase_freeze, "final": phase_final, "card": phase_card}[args.phase](args)


if __name__ == "__main__":
    main()
