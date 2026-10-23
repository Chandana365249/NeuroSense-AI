"""Per-person EEG covariance alignment (Protocol v1.1, exploratory, 12 development participants).

Design (declared here before any result of this script existed; mirrored in docs/COVARIANCE_ALIGNMENT_EXPERIMENT.md §2):

Data: the cached preprocessed 4 s segments (4-45 Hz, robust clipping; fingerprint 3c5359083bc90ae5), row-for-row
identical to the 440-feature table, of the 12 trusted development participants only. Frozen participants asserted absent.
Label: high_valence if valence > 5.0. Trial prediction: mean segment probability > 0.5.

Representation (fixed, untuned):
  each segment is band-pass filtered (zero-phase Butterworth order 4, within the segment only) into theta 4-8,
  alpha 8-13, beta 13-30 and gamma 30-45 Hz; per band a 32x32 covariance with fixed shrinkage
  C = 0.95 S + 0.05 tr(S)/32 I. A reference R is a log-Euclidean mean expm(mean logm C). Re-centring maps
  C -> R^-1/2 C R^-1/2; the feature vector is the upper triangle of logm of the re-centred matrix (off-diagonal x sqrt 2),
  528 per band, 2,112 in total.

Cross-person, leave-one-participant-and-video-fold-out (the stricter design of the earlier LVO studies):
  for held-out participant p, seed s in 42-44 and video fold k of lvo.video_folds(s) (8 test videos), train on the
  other 11 participants' 32 non-test videos and test on p's 8 test videos. Test videos never enter any training row.
  A0  baseline: 440 features, training-fold StandardScaler (inductive)                 - registered Random Forest
  A1  baseline + per-person z-score of the 440 features on each person's non-test videos (unlabelled)  - RF
  A2  covariance tangent features, ONE global reference = mean of the training rows (no alignment)    - RF
  A3  covariance tangent features, PER-PERSON reference = mean of that person's own non-test-video segments
      (unlabelled; for the held-out person: their 32 non-test videos, never the test videos)            - RF
  A4  A3 features with logistic regression (C = 0.01, standardized, balanced; fixed in advance)        - descriptive
  A1 and A3 need unlabelled EEG of the new person (Protocol §6.2 calibration variant); they are not zero-shot.
Within-person (supervised personalization, the earlier P2 folds): per participant, seed and fold, train on 32 videos,
  test on 8. W0 = 440 features RF; W1 = tangent features re-centred at the person's training-video reference (fit in
  the fold), RF.
Pre-declared comparisons (participant bootstrap 10,000 CI of the paired mean difference, Wilcoxon, Holm over 4):
  primary A3 - A0; secondary A3 - A1, A3 - A2, W1 - W0. A4 and everything else descriptive.

Usage (from backend/):
  python -m scripts.covariance_alignment --phase covariances | cross | within | report
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
from scipy import signal as sps_signal  # noqa: E402

from app.core.config import get_settings  # noqa: E402
from app.ml.deep.robust_evaluation import git_commit  # noqa: E402
from app.ml.participants import FROZEN_TEST_PARTICIPANTS, PROTOCOL_VERSION  # noqa: E402
from scripts import cross_participant_lvo as lvo  # noqa: E402
from scripts import model_improvement as mi  # noqa: E402

PARTICIPANTS, SEEDS, LABEL = mi.PARTICIPANTS, mi.SEEDS, mi.LABEL
BANDS = {"theta": (4.0, 8.0), "alpha": (8.0, 13.0), "beta": (13.0, 30.0), "gamma": (30.0, 45.0)}
SHRINK, FS, N_CH = 0.05, 128, 32
LR = {"name": "lr_C0.01", "family": "lr", "C": 0.01}
RF = mi.CONFIGS[0]
DOCS_OUT = mi.ROOT / "docs" / "results" / "exploratory" / "covariance_alignment"
TRIU = np.triu_indices(N_CH)
OFFDIAG_W = np.where(TRIU[0] == TRIU[1], 1.0, np.sqrt(2.0))


# ---------------------------------------------------------------- SPD helpers

def _eig_fn(C: np.ndarray, fn: Any) -> np.ndarray:
    w, V = np.linalg.eigh(C)
    return (V * fn(w)[..., None, :]) @ np.swapaxes(V, -1, -2)


def logm(C: np.ndarray) -> np.ndarray:
    return _eig_fn(C, np.log)


def expm_sym(L: np.ndarray) -> np.ndarray:
    return _eig_fn(L, np.exp)


def invsqrtm(C: np.ndarray) -> np.ndarray:
    return _eig_fn(C, lambda w: 1.0 / np.sqrt(w))


def tangent(C: np.ndarray, ref_isqrt: np.ndarray) -> np.ndarray:
    """C: (n, bands, 32, 32); ref_isqrt: (n, bands, 32, 32) or broadcastable. Returns (n, bands * 528)."""
    L = logm(ref_isqrt @ C @ ref_isqrt)
    return (L[..., TRIU[0], TRIU[1]] * OFFDIAG_W).reshape(len(C), -1)


def le_mean(Lrows: np.ndarray) -> np.ndarray:
    """Log-Euclidean mean of (n, bands, 32, 32) log-covariances -> (bands, 32, 32)."""
    return expm_sym(Lrows.mean(axis=0))


# ---------------------------------------------------------------- data

def out_dir() -> Path:
    return get_settings().output_dir / "experiments" / "covariance_alignment_20261022"


def phase_covariances(out: Path) -> None:
    from app.ml.deep.data import build_segment_table
    from app.ml.factory import make_dataset, make_preprocessor

    s = get_settings()
    seg = build_segment_table(make_dataset(s), make_preprocessor(s), s.cache_dir)
    D = mi.load_data()
    assert seg.fingerprint == D["fingerprint"], "segments and features must come from the same preprocessing"
    assert not set(seg.participant.tolist()) & FROZEN_TEST_PARTICIPANTS
    index = {pid: i for i, pid in enumerate(PARTICIPANTS)}
    assert np.array_equal(np.array([index[x] for x in seg.participant]), D["p"]) and np.array_equal(seg.trial, D["t"])
    sos = {b: sps_signal.butter(4, f, btype="bandpass", fs=FS, output="sos") for b, f in BANDS.items()}
    C = np.empty((len(D["y"]), len(BANDS), N_CH, N_CH))
    for i0 in range(0, len(C), 500):
        x = np.asarray(seg.X[i0:i0 + 500], dtype=np.float64)
        for j, b in enumerate(BANDS):
            xb = sps_signal.sosfiltfilt(sos[b], x, axis=-1)  # within the segment only
            xb = xb - xb.mean(axis=-1, keepdims=True)
            S = xb @ np.swapaxes(xb, -1, -2) / xb.shape[-1]
            tr = np.trace(S, axis1=-2, axis2=-1)[:, None, None] / N_CH
            C[i0:i0 + len(x), j] = (1 - SHRINK) * S + SHRINK * tr * np.eye(N_CH)
        print(f"  covariances {min(i0 + 500, len(C))}/{len(C)}", flush=True)
    assert np.isfinite(C).all() and (np.linalg.eigvalsh(C) > 0).all(), "covariances must be SPD"
    np.save(out / "covariances.npy", C)
    np.save(out / "log_covariances.npy", logm(C))
    (out / "covariances_meta.json").write_text(json.dumps({
        "fingerprint": D["fingerprint"], "rows": len(C), "bands": BANDS, "shrinkage": SHRINK,
        "created_at": datetime.now(UTC).isoformat()}, indent=1), encoding="utf-8")
    print("saved", out / "covariances.npy")


def load_cov(out: Path) -> tuple[np.ndarray, np.ndarray]:
    return np.load(out / "covariances.npy", mmap_mode="r"), np.load(out / "log_covariances.npy", mmap_mode="r")


def per_person_reference_isqrt(L: np.ndarray, p: np.ndarray, ref_rows: np.ndarray) -> np.ndarray:
    """(rows, bands, 32, 32): for every row, R_q^-1/2 of its participant q, R_q from q's own ref_rows (no labels)."""
    out = np.empty((len(p), len(BANDS), N_CH, N_CH))
    for q in np.unique(p):
        m = p == q
        assert (m & ref_rows).any(), "every participant needs reference rows"
        out[m] = invsqrtm(le_mean(np.asarray(L[m & ref_rows])))
    return out


# ---------------------------------------------------------------- jobs

def cross_job(D: dict[str, Any], out: Path, cond: str, seed: int, pi: int, k: int) -> dict[str, Any]:
    fold = lvo.video_folds(seed)[k]
    nontest = ~np.isin(D["t"], fold)
    train = (D["p"] != pi) & nontest
    test = (D["p"] == pi) & ~nontest
    assert not np.any(D["p"][train] == pi) and not np.any(np.isin(D["t"][train], fold)), "leakage"
    assert set(D["t"][test].tolist()) == set(fold)
    rows = train | test
    if cond == "A0":
        X = D["X"]
    elif cond == "A1":
        X = mi.per_participant_z(D["X"], D["p"], nontest)
    else:
        C, L = load_cov(out)
        Cr = np.asarray(C[rows])
        if cond == "A2":  # one reference from the training rows only, applied to everyone
            iso = np.broadcast_to(invsqrtm(le_mean(np.asarray(L[train]))), Cr.shape)
        else:  # A3/A4: each person re-centred on their own unlabelled non-test videos (all rows, then subset)
            iso = per_person_reference_isqrt(np.asarray(L), D["p"], nontest)[rows]
        X = np.zeros((len(D["y"]), len(BANDS) * len(TRIU[0])))
        X[rows] = tangent(Cr, iso)
    assert np.isfinite(X[train | test]).all(), "features must be finite"
    cfg = LR if cond == "A4" else RF
    model = mi.make_model(cfg, seed).fit(X[train], D["y"][train])
    s = mi.seg_scores(model, cfg, X[test])
    return {"outer": {"mean": mi.trial_outputs(D["t"][test], D["y"][test], s, "mean", 0.5)}}


def within_job(D: dict[str, Any], out: Path, cond: str, seed: int, pi: int, k: int) -> dict[str, Any]:
    rows = D["p"] == pi
    t, y = D["t"][rows], D["y"][rows]
    fold = lvo.video_folds(seed)[k]
    tr, te = ~np.isin(t, fold), np.isin(t, fold)
    assert not set(t[tr].tolist()) & set(t[te].tolist())
    if cond == "W0":
        X = D["X"][rows]
    else:
        C, L = load_cov(out)
        Cp, Lp = np.asarray(C[rows]), np.asarray(L[rows])
        X = tangent(Cp, invsqrtm(le_mean(Lp[tr])))  # reference fitted on the fold's training videos
    assert np.isfinite(X).all(), "features must be finite"
    model = mi.make_model(RF, seed).fit(X[tr], y[tr])
    s = mi.seg_scores(model, RF, X[te])
    return {"outer": {"mean": mi.trial_outputs(t[te], y[te], s, "mean", 0.5)}}


def phase_cross(out: Path) -> None:
    D = mi.load_data()
    jobs = [(mi.key(cond=c, seed=s, pi=pi, k=k), cross_job, (D, out, c, s, pi, k))
            for c in ("A0", "A1", "A2", "A3", "A4") for s in SEEDS for pi in range(len(PARTICIPANTS)) for k in range(5)]
    mi.run_jobs(out / "cross.json", jobs, mi.header("covariance_alignment_cross", D))


def phase_within(out: Path) -> None:
    D = mi.load_data()
    jobs = [(mi.key(cond=c, seed=s, pi=pi, k=k), within_job, (D, out, c, s, pi, k))
            for c in ("W0", "W1") for s in SEEDS for pi in range(len(PARTICIPANTS)) for k in range(5)]
    mi.run_jobs(out / "within.json", jobs, mi.header("covariance_alignment_within", D))


# ---------------------------------------------------------------- report

def _condition(R: dict[str, Any], cond: str) -> dict[str, Any]:
    per: dict = {}
    for s in SEEDS:
        for pi, q in enumerate(PARTICIPANTS):
            for k in range(5):
                mi._collect(per, s, q, R[mi.key(cond=cond, seed=s, pi=pi, k=k)]["outer"]["mean"])
    return mi.finalize(per)


def _sign_flip(d: np.ndarray) -> float:
    import itertools
    obs = abs(d.mean())
    return float(np.mean([abs((d * np.array(sg)).mean()) >= obs - 1e-12 for sg in itertools.product((1, -1), repeat=len(d))]))


def phase_report(out: Path) -> None:
    cross = mi.load_json(out / "cross.json")["results"]
    within = mi.load_json(out / "within.json")["results"]
    conds = {c: _condition(cross, c) for c in ("A0", "A1", "A2", "A3", "A4")} | {c: _condition(within, c) for c in ("W0", "W1")}
    comps = {"A3_vs_A0 (primary)": ("A3", "A0"), "A3_vs_A1": ("A3", "A1"), "A3_vs_A2": ("A3", "A2"), "W1_vs_W0": ("W1", "W0")}
    res: dict[str, Any] = {}
    for name, (a, b) in comps.items():
        r = mi.paired(conds[a], conds[b])
        d = np.array([conds[a]["per_participant"][q]["balanced_accuracy"] - conds[b]["per_participant"][q]["balanced_accuracy"]
                      for q in PARTICIPANTS])
        r["sign_flip_p"] = _sign_flip(d)
        r["macro_f1"] = mi.paired(conds[a], conds[b], "f1_macro")
        r["roc_auc"] = mi.paired(conds[a], conds[b], "roc_auc")
        res[name] = r
    adj = mi.holm([v["wilcoxon_p"] if v["wilcoxon_p"] is not None else 1.0 for v in res.values()])
    for v, a in zip(res.values(), adj, strict=True):
        v["holm_p"] = a
    descriptive = {"A4_vs_A0": mi.paired(conds["A4"], conds["A0"]), "A1_vs_A0": mi.paired(conds["A1"], conds["A0"]),
                   "A2_vs_A0": mi.paired(conds["A2"], conds["A0"])}
    summary = {"label": LABEL, "protocol_version": PROTOCOL_VERSION, "created_at": datetime.now(UTC).isoformat(),
               "git_commit": git_commit(), "participants": list(PARTICIPANTS), "seeds": list(SEEDS),
               "covariances": json.loads((out / "covariances_meta.json").read_text(encoding="utf-8")),
               "conditions": conds, "comparisons": res, "descriptive_comparisons": descriptive}
    (out / "summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    DOCS_OUT.mkdir(parents=True, exist_ok=True)
    (DOCS_OUT / "summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    def f(x: float | None) -> str:
        return "–" if x is None else f"{x:.3f}"

    names = {"A0": "Baseline 440 features (inductive)", "A1": "Baseline + per-person z-score (calibration)",
             "A2": "Covariance tangent, global reference (no alignment)", "A3": "**Covariance tangent, per-person alignment**",
             "A4": "A3 features + logistic regression (descriptive)", "W0": "Within-person: baseline 440 features",
             "W1": "Within-person: covariance tangent, re-centred"}
    lines = ["| Condition | Trial bal. acc. [95% CI] | Macro-F1 | Within-person AUC | Range over people | Single-label people | Seed SD |",
             "|---|---|---|---|---|---|---|"]
    for c, s in conds.items():
        m = s["metrics"]
        lines.append(f"| {c}: {names[c]} | {f(m['balanced_accuracy']['mean'])} [{f(m['balanced_accuracy']['ci_low'])}, "
                     f"{f(m['balanced_accuracy']['ci_high'])}] | {f(m['f1_macro']['mean'])} | {f(m['roc_auc']['mean'])} | "
                     f"{f(m['balanced_accuracy']['min'])}–{f(m['balanced_accuracy']['max'])} | "
                     f"{s['single_label_participants_mean_over_seeds']:.1f} | {s['across_seed_sd_of_mean_ba']:.3f} |")
    lines += ["", "| Comparison | Δ bal. acc. [bootstrap 95% CI] | Wilcoxon p | Sign-flip p | Holm p | Better / worse | Δ macro-F1 [CI] | Δ AUC [CI] |",
              "|---|---|---|---|---|---|---|---|"]
    for n, v in res.items():
        lines.append(f"| {n} | {v['mean_difference']:+.3f} [{v['bootstrap_ci'][0]:+.3f}, {v['bootstrap_ci'][1]:+.3f}] | "
                     f"{f(v['wilcoxon_p'])} | {f(v['sign_flip_p'])} | {f(v['holm_p'])} | {v['n_better']} / {v['n_worse']} | "
                     f"{v['macro_f1']['mean_difference']:+.3f} [{v['macro_f1']['bootstrap_ci'][0]:+.3f}, {v['macro_f1']['bootstrap_ci'][1]:+.3f}] | "
                     f"{v['roc_auc']['mean_difference']:+.3f} [{v['roc_auc']['bootstrap_ci'][0]:+.3f}, {v['roc_auc']['bootstrap_ci'][1]:+.3f}] |")
    for n, v in descriptive.items():
        lines.append(f"| {n} (descriptive) | {v['mean_difference']:+.3f} [{v['bootstrap_ci'][0]:+.3f}, {v['bootstrap_ci'][1]:+.3f}] | "
                     f"{f(v['wilcoxon_p'])} | – | – | {v['n_better']} / {v['n_worse']} | – | – |")
    lines += ["", "| Participant | " + " | ".join(conds) + " |", "|---|" + "---|" * len(conds)]
    for q in PARTICIPANTS:
        lines.append(f"| {q} | " + " | ".join(f(conds[c]["per_participant"][q]["balanced_accuracy"]) for c in conds) + " |")
    (DOCS_OUT / "tables.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=["covariances", "cross", "within", "report"], required=True)
    args = parser.parse_args()
    out = out_dir()
    out.mkdir(parents=True, exist_ok=True)
    {"covariances": phase_covariances, "cross": phase_cross, "within": phase_within, "report": phase_report}[args.phase](out)


if __name__ == "__main__":
    main()
