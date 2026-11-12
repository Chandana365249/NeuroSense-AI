"""EXPLORATORY FACED noise-removal experiment: current preprocessing (A) vs an enhanced cleaning pipeline (B).

Only the 98 development participants of faced-protocol-v1 are processed; the 25 already-evaluated test participants are
never read. Label (unchanged): self-reported valence > 3.5. Classifier (unchanged): the selected FACED model,
logistic regression C=0.01 (StandardScaler fitted on training rows). Evaluation: 5 participant-grouped folds over the 98
development participants (the protocol's inner participant groups), all 28 videos, trial = mean segment probability > 0.5.

Condition A: faced-v2 as used so far (common average reference -> 4-45 Hz band-pass -> 6 MAD-sigma clipping ->
4 s segments, reject > 5% clipped -> 412 features). Features are taken from features_faced-v2.npz.

Condition B (declared in protocol.json after the label-free audit, before any B feature or score existed), per
recording and without labels:
1. bad-channel detection on the 28 trial windows (1-45 Hz, before referencing): robust-amplitude deviation, low
   correlation with the other channels, or near-flat; detected channels are interpolated (spherical splines, standard
   10-20 positions; MNE). At most 6 channels (20%) are interpolated; more is recorded as a failure, not fixed.
2. common average reference over the 30 channels (after interpolation);
3. 50 Hz notch (mains frequency in China, where FACED was recorded), only where the audit shows a line-noise peak;
4. ICA (FastICA, fitted on that recording's 1-45 Hz data) removing ocular components only: components whose
   1-10 Hz time course correlates with Fp1/Fp2 (MNE find_bads_eog, z > 3), at most 3. No muscle or high-frequency
   components are removed and no frequency band is dropped;
5. then exactly condition A's band-pass, clipping, segmentation, rejection and features.
Each step is unsupervised and per recording, so it uses neither labels nor other participants.

Usage (from backend/):
  python -m scripts.faced_cleaning_experiment --phase audit      # label-free data characterisation
  python -m scripts.faced_cleaning_experiment --phase protocol   # declare condition B
  python -m scripts.faced_cleaning_experiment --phase build      # condition B features (dev participants only)
  python -m scripts.faced_cleaning_experiment --phase evaluate   # A vs B on the same folds
"""

from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
from datetime import UTC, datetime  # noqa: E402
from typing import Any  # noqa: E402

import numpy as np  # noqa: E402
from joblib import Parallel, delayed  # noqa: E402
from scipy import signal as sps_signal  # noqa: E402

from app.core.config import get_settings  # noqa: E402
from app.ml.faced import dataset as fd  # noqa: E402
from app.ml.faced import pipeline as fpipe  # noqa: E402
from app.ml.faced import protocol as fprot  # noqa: E402
from app.ml.faced.bdf import read_window  # noqa: E402
from app.ml.provenance import git_commit  # noqa: E402
from scripts import faced_experiment as fe  # noqa: E402

OUT = fe.OUT / "cleaning"
FS = fpipe.TARGET_FS
CLASSIFIER = next(c for c in fe.CANDIDATES if c["name"] == "lr_C0.01")
MAX_BAD = 6
MAX_EOG_COMPONENTS = 3
ICA_REJECT_PTP_UV = 500.0
LINE_HZ = 50.0


def dev_participants() -> list[str]:
    split = json.loads((fe.OUT / "protocol_split_v1.json").read_text(encoding="utf-8"))
    return list(split["dev_participants"])


# ---------------------------------------------------------------- signal access

def raw_windows(part: fd.FacedParticipant) -> list[np.ndarray]:
    """Per trial: (30, (28 + 2) s * 250) microvolts at 250 Hz, NOT re-referenced, per-channel mean removed."""
    fs = part.sampling_rate
    out = []
    for trial in part.trials:
        start = trial.onset_s + trial.duration_s - fpipe.WINDOW_S
        a = int(round((start - fpipe.MARGIN_S) * fs))
        b = a + int(round((fpipe.WINDOW_S + 2 * fpipe.MARGIN_S) * fs))
        x = read_window(part.header, part.channel_indices, a, b)
        x = x - x.mean(axis=1, keepdims=True)
        if fs == 1000.0:
            x = sps_signal.resample_poly(x, up=1, down=4, axis=1)
        out.append(x)
    return out


def band(x: np.ndarray, lo: float, hi: float) -> np.ndarray:
    sos = sps_signal.butter(4, [lo, hi], btype="bandpass", fs=FS, output="sos")
    return sps_signal.sosfiltfilt(sos, x, axis=-1)


def crop(x: np.ndarray) -> np.ndarray:
    m = int(fpipe.MARGIN_S * FS)
    return x[:, m:m + int(fpipe.WINDOW_S * FS)]


# ---------------------------------------------------------------- cleaning steps (unsupervised, per recording)

def detect_bad_channels(windows: list[np.ndarray]) -> dict[str, Any]:
    """PREP-style criteria on 1-45 Hz data after a robust (median-across-channels) reference.

    FACED was recorded with BioSemi, whose CMS reference electrode sits next to CPz, so in the recorded reference the
    neighbouring channels (CP1, CP2, Cz, Pz) are legitimately tiny and weakly correlated. The label-free audit showed
    this; criteria on the recorded reference would mislabel them as bad. Hence: robust reference first, only
    abnormally HIGH amplitude counts, and 'flat' means flat in the recorded data."""
    raw = np.concatenate([crop(band(w, 1.0, 45.0)) for w in windows], axis=1)
    raw_rsd = 1.4826 * np.median(np.abs(raw - np.median(raw, axis=1, keepdims=True)), axis=1)
    x = raw - np.median(raw, axis=0, keepdims=True)
    rsd = 1.4826 * np.median(np.abs(x - np.median(x, axis=1, keepdims=True)), axis=1)
    log_sd = np.log(rsd + 1e-9)
    med, mad = np.median(log_sd), 1.4826 * np.median(np.abs(log_sd - np.median(log_sd)))
    z_amp = (log_sd - med) / max(mad, 1e-9)
    win = FS  # 1 s windows: maximum |correlation| with any other channel
    n = x.shape[1] // win
    xs = x[:, : n * win].reshape(x.shape[0], n, win).transpose(1, 0, 2)
    corr_max = np.empty((n, x.shape[0]))
    for i, seg in enumerate(xs):
        sd = seg.std(axis=1)
        c = np.corrcoef(seg) if (sd > 0).all() else np.zeros((len(seg), len(seg)))
        np.fill_diagonal(c, 0.0)
        corr_max[i] = np.abs(c).max(axis=1)
    low_corr_share = (corr_max < 0.4).mean(axis=0)
    # intermittent high amplitude (bursts a median-based SD ignores, but a common average would spread to all channels)
    wsd = xs.std(axis=2)
    burst_share = (wsd > 5 * np.median(wsd, axis=1, keepdims=True)).mean(axis=0)
    flat = raw_rsd < 0.05
    # Fp1/Fp2 carry blinks (handled by ICA, for which they are the ocular proxies): exempt from the amplitude rule only
    ocular = np.isin(fd.FACED_CHANNELS, ["Fp1", "Fp2"])
    reasons = {"high_amplitude": (z_amp > 5) & ~ocular, "intermittent_high_amplitude": (burst_share > 0.05) & ~ocular,
               "low_correlation": low_corr_share > 0.5, "flat": flat}
    bad = np.any(list(reasons.values()), axis=0)
    return {"bad": [fd.FACED_CHANNELS[i] for i in np.flatnonzero(bad)],
            "reasons": {k: [fd.FACED_CHANNELS[i] for i in np.flatnonzero(m)] for k, m in reasons.items()},
            "robust_sd_uv": rsd.round(3).tolist(), "recorded_reference_robust_sd_uv": raw_rsd.round(3).tolist(),
            "amplitude_z": z_amp.round(2).tolist(), "burst_share": burst_share.round(3).tolist(), "low_correlation_share": low_corr_share.round(3).tolist()}


def line_noise_ratio(windows: list[np.ndarray]) -> np.ndarray:
    """Per channel: PSD at 50 Hz / median PSD over 46-48 and 52-54 Hz (CAR data, Welch 2 s)."""
    x = np.concatenate([crop(w - w.mean(axis=0, keepdims=True)) for w in windows], axis=1)
    f, p = sps_signal.welch(x, fs=FS, nperseg=2 * FS, axis=-1)
    at = p[:, np.argmin(np.abs(f - LINE_HZ))]
    ref = np.median(p[:, ((f >= 46) & (f <= 48)) | ((f >= 52) & (f <= 54))], axis=1)
    return at / ref


def interpolate(windows: list[np.ndarray], bad: list[str]) -> list[np.ndarray]:
    if not bad:
        return windows
    import mne

    info = mne.create_info(list(fd.FACED_CHANNELS), FS, "eeg")
    out = []
    for w in windows:
        raw = mne.io.RawArray(w * 1e-6, info, verbose="error")
        raw.set_montage("standard_1020", verbose="error")
        raw.info["bads"] = list(bad)
        raw.interpolate_bads(reset_bads=True, mode="accurate", verbose="error")
        out.append(raw.get_data() * 1e6)
    return out


def notch(x: np.ndarray) -> np.ndarray:
    b, a = sps_signal.iirnotch(LINE_HZ, Q=30.0, fs=FS)
    return sps_signal.filtfilt(b, a, x, axis=-1)


def ica_ocular(windows: list[np.ndarray], n_bad: int, seed: int) -> tuple[list[np.ndarray], dict[str, Any]]:
    import mne

    info = mne.create_info(list(fd.FACED_CHANNELS), FS, "eeg")
    lengths = [w.shape[1] for w in windows]
    raw = mne.io.RawArray(np.concatenate(windows, axis=1) * 1e-6, info, verbose="error")
    raw.set_montage("standard_1020", verbose="error")
    fit = raw.copy().filter(1.0, 45.0, verbose="error")
    n_comp = len(fd.FACED_CHANNELS) - 1 - n_bad  # rank: CAR removes 1, each interpolated channel removes 1
    ica = mne.preprocessing.ICA(n_components=n_comp, method="fastica", random_state=seed, max_iter="auto", verbose="error")
    ica.fit(fit, reject={"eeg": ICA_REJECT_PTP_UV * 1e-6}, tstep=2.0, verbose="error")  # gross transients excluded from the fit
    idx, scores = ica.find_bads_eog(fit, ch_name=["Fp1", "Fp2"], threshold=3.0, measure="zscore", verbose="error")
    strength = np.max(np.abs(np.atleast_2d(scores)), axis=0)
    idx = sorted(idx, key=lambda i: -strength[i])[:MAX_EOG_COMPONENTS]
    cleaned = ica.apply(raw.copy(), exclude=idx, verbose="error").get_data() * 1e6
    before = fit.get_data() * 1e6
    after = ica.apply(fit.copy(), exclude=idx, verbose="error").get_data() * 1e6
    def rsd(a: np.ndarray) -> np.ndarray:
        return 1.4826 * np.median(np.abs(a - np.median(a, axis=1, keepdims=True)), axis=1)
    var_before, var_after = float(np.var(before)), float(np.var(after))
    parts = np.split(cleaned, np.cumsum(lengths)[:-1], axis=1)
    return parts, {"n_components": n_comp, "removed": [int(i) for i in idx],
                   "variance_removed_share_1_45hz": 1 - var_after / var_before,
                   "robust_sd_ratio_after_before_median": float(np.median(rsd(after) / rsd(before))),
                   "frontal_robust_sd_ratio": float(np.median(rsd(after[:2]) / rsd(before[:2])))}


# ---------------------------------------------------------------- phases

def audit_one(pid: str) -> dict[str, Any]:
    part = fd.load_participant(get_settings().faced_dataset_dir, pid)
    w = raw_windows(part)
    lnr = line_noise_ratio(w)
    bad = detect_bad_channels(w)
    car = [x - x.mean(axis=0, keepdims=True) for x in w]
    # residual mains power that reaches the existing features: gamma (30-45 Hz) power with vs without a 50 Hz notch
    pre = fpipe._preprocessor
    g = []
    for x in car[:6]:
        for y in (x, notch(x)):
            f, p = sps_signal.welch(pre.bandpass_filter(crop(y)), fs=FS, nperseg=FS, axis=-1)
            g.append(np.trapezoid(p[:, (f >= 30) & (f <= 45)], f[(f >= 30) & (f <= 45)], axis=-1))
    g = np.array(g).reshape(-1, 2, len(fd.FACED_CHANNELS))
    gamma_change = float(np.median(1 - g[:, 1] / g[:, 0]))
    # blink indicator: share of 1 s windows where Fp1/Fp2 1-4 Hz peak-to-peak exceeds 5x the posterior channels' median
    fp = [fd.FACED_CHANNELS.index("Fp1"), fd.FACED_CHANNELS.index("Fp2")]
    post = [fd.FACED_CHANNELS.index(c) for c in ("O1", "O2", "Oz", "PO3", "PO4", "Pz")]
    lo = np.concatenate([crop(band(x, 1.0, 4.0)) for x in car], axis=1)
    n = lo.shape[1] // FS
    seg = lo[:, : n * FS].reshape(lo.shape[0], n, FS)
    ptp = seg.max(axis=2) - seg.min(axis=2)
    blink_share = float((ptp[fp].max(axis=0) > 5 * np.median(ptp[post])).mean())
    return {"participant": pid, "layout": part.layout, "fs": part.sampling_rate,
            "line_noise_ratio_median": float(np.median(lnr)), "line_noise_ratio_max": float(np.max(lnr)),
            "gamma_power_removed_by_notch_median": gamma_change, "bad_channels": bad["bad"], "bad_reasons": bad["reasons"],
            "blink_window_share": blink_share}


def phase_audit(_: argparse.Namespace) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = Parallel(n_jobs=4)(delayed(audit_one)(p) for p in dev_participants())
    lnr = np.array([r["line_noise_ratio_median"] for r in rows])
    summary = {
        "participants": len(rows),
        "line_noise_ratio_median_over_participants": float(np.median(lnr)),
        "participants_with_50hz_peak_ratio_gt_3": int((lnr > 3).sum()),
        "gamma_power_removed_by_notch_median": float(np.median([r["gamma_power_removed_by_notch_median"] for r in rows])),
        "participants_with_bad_channels": int(sum(bool(r["bad_channels"]) for r in rows)),
        "bad_channel_counts": {str(k): int(sum(len(r["bad_channels"]) == k for r in rows)) for k in range(0, 31)
                               if any(len(r["bad_channels"]) == k for r in rows)},
        "bad_channel_frequency": {c: int(sum(c in r["bad_channels"] for r in rows)) for c in fd.FACED_CHANNELS
                                  if any(c in r["bad_channels"] for r in rows)},
        "blink_window_share_median": float(np.median([r["blink_window_share"] for r in rows])),
        "blink_window_share_iqr": np.percentile([r["blink_window_share"] for r in rows], [25, 75]).tolist(),
    }
    (OUT / "audit.json").write_text(json.dumps({"created_at": datetime.now(UTC).isoformat(), "summary": summary,
                                                "participants": rows}, indent=1), encoding="utf-8")
    print(json.dumps(summary, indent=1))


def phase_protocol(_: argparse.Namespace) -> None:
    path = OUT / "protocol.json"
    if path.exists():
        raise SystemExit("protocol.json exists; declared once")
    audit = json.loads((OUT / "audit.json").read_text(encoding="utf-8"))["summary"]
    protocol = {
        "declared_at": datetime.now(UTC).isoformat(), "exploratory": True,
        "participants": "98 development participants of faced-protocol-v1 only; test participants never read",
        "label": f"self-reported valence > {fpipe.VALENCE_THRESHOLD} (unchanged)", "videos": "all 28",
        "classifier": CLASSIFIER, "seed": fe.SEED, "folds": "faced-protocol-v1 inner participant groups (5)",
        "primary_metric": "pooled out-of-fold trial balanced accuracy; A vs B on identical folds",
        "condition_A": "faced-v2 features (features_faced-v2.npz), unchanged",
        "condition_B": {"bad_channels": {"computed_on": "1-45 Hz, robust median reference (BioSemi CMS near CPz)", "amplitude_robust_z_gt": 5, "amplitude_rule_exempt": ["Fp1", "Fp2"], "intermittent": "1 s SD > 5x cross-channel median in > 5% of windows", "low_correlation": "max |r| with any other channel < 0.4 in > 50% of 1 s windows",
                                         "flat_recorded_robust_sd_uv_lt": 0.05, "interpolation": "MNE spherical spline, standard_1020",
                                         "max_interpolated": MAX_BAD},
                        "reference": "common average after interpolation",
                        "notch_50hz": "iirnotch Q=30, zero-phase (applied to all recordings)",
                        "ica": {"method": "fastica", "fit": "per recording, 1-45 Hz, the 28 trial windows",
                                "exclude": "find_bads_eog with Fp1/Fp2 proxies, z > 3", "max_components": MAX_EOG_COMPONENTS,
                                "fit_rejects_2s_spans_with_peak_to_peak_uv_gt": ICA_REJECT_PTP_UV,
                                "muscle_components_removed": False},
                        "then": "condition A band-pass, clipping, segmentation, rejection and features"},
        "audit_summary_at_declaration": audit,
        "features_A_sha256": hashlib.sha256((fe.OUT / f"features_{fpipe.PIPELINE_VERSION}.npz").read_bytes()).hexdigest(),
    }
    path.write_text(json.dumps(protocol, indent=1), encoding="utf-8")
    print("protocol sha256", hashlib.sha256(path.read_bytes()).hexdigest())


def build_one(pid: str) -> dict[str, Any]:
    part = fd.load_participant(get_settings().faced_dataset_dir, pid)
    w = raw_windows(part)
    bad = detect_bad_channels(w)
    failed = len(bad["bad"]) > MAX_BAD
    w = interpolate(w, [] if failed else bad["bad"])
    w = [notch(x - x.mean(axis=0, keepdims=True)) for x in w]
    n_bad = 0 if failed else len(bad["bad"])
    w, ica_info = ica_ocular(w, n_bad, fe.SEED)
    X, rows, rejected = [], [], 0
    for trial, x in zip(part.trials, w, strict=True):
        res = fpipe._preprocessor.preprocess(crop(x), has_baseline=False)
        rejected += len(res.rejected_segment_indices)
        if res.segments.shape[0]:
            X.append(fpipe._extractor.extract(res.segments))
            rows += [(trial.video_index, k, trial.ratings["Valence"]) for k in res.kept_segment_indices]
    return {"participant": pid, "X": np.concatenate(X).astype(np.float32), "rows": rows, "rejected": rejected,
            "bad_channels": bad["bad"], "bad_channel_failure": failed, "ica": ica_info}


def phase_build(_: argparse.Namespace) -> None:
    if not (OUT / "protocol.json").exists():
        raise SystemExit("run --phase protocol first")
    dev = dev_participants()
    test = json.loads((fe.OUT / "protocol_split_v1.json").read_text(encoding="utf-8"))["test_participants"]
    assert not set(dev) & set(test)
    res = Parallel(n_jobs=4, verbose=5)(delayed(build_one)(p) for p in dev)
    X = np.concatenate([r["X"] for r in res])
    p = np.concatenate([[r["participant"]] * len(r["rows"]) for r in res])
    v = np.array([row[0] for r in res for row in r["rows"]])
    seg = np.array([row[1] for r in res for row in r["rows"]])
    val = np.array([row[2] for r in res for row in r["rows"]])
    np.savez_compressed(OUT / "features_B.npz", X=X, participant=p, video=v, segment=seg, valence=val,
                        y=(val > fpipe.VALENCE_THRESHOLD).astype(np.int8))
    report = [{k: r[k] for k in ("participant", "rejected", "bad_channels", "bad_channel_failure", "ica")} for r in res]
    (OUT / "build_report.json").write_text(json.dumps({"created_at": datetime.now(UTC).isoformat(), "participants": report},
                                                      indent=1), encoding="utf-8")
    print("segments", len(X), "rejected", sum(r["rejected"] for r in res))


def oof(X: np.ndarray, p: np.ndarray, v: np.ndarray, y: np.ndarray, groups: list[list[str]]) -> dict[str, np.ndarray]:
    s = np.full(len(y), np.nan)
    for g in groups:
        va = np.isin(p, g)
        s[va] = fe.fit_predict(CLASSIFIER, X[~va], y[~va], X[va])
    return {"s": s, "fold": np.array([next(k for k, g in enumerate(groups) if q in g) for q in p])}


def phase_evaluate(_: argparse.Namespace) -> None:
    split = json.loads((fe.OUT / "protocol_split_v1.json").read_text(encoding="utf-8"))
    groups = [list(g) for g in split["inner_participant_groups"]]
    dev = set(split["dev_participants"])
    assert set().union(*map(set, groups)) == dev
    za = np.load(fe.OUT / f"features_{fpipe.PIPELINE_VERSION}.npz")
    ka = np.isin(za["participant"], sorted(dev))
    A = {"X": za["X"][ka].astype(np.float64), "p": za["participant"][ka], "v": za["video"][ka].astype(int), "y": za["y"][ka].astype(int)}
    zb = np.load(OUT / "features_B.npz")
    B = {"X": zb["X"].astype(np.float64), "p": zb["participant"], "v": zb["video"].astype(int), "y": zb["y"].astype(int)}
    out: dict[str, Any] = {"created_at": datetime.now(UTC).isoformat(), "git_commit": git_commit(), "conditions": {}}
    tables = {}
    for name, d in (("A_current", A), ("B_enhanced", B)):
        r = oof(d["X"], d["p"], d["v"], d["y"], groups)
        t = fe.trial_table(d["p"], d["v"], d["y"], r["s"])
        tables[name] = t
        folds = {}
        for k, g in enumerate(groups):
            m = np.isin(t["participant"], g)
            folds[str(k)] = fe.metrics(t["y"][m], t["score"][m], t["pred"][m])
        ev = fe.evaluate(t, d["y"], r["s"])
        ev.pop("per_participant")
        out["conditions"][name] = {"segments": int(len(d["y"])), "trials": int(len(t["y"])),
                                   "trial_high_share": float(t["y"].mean()), "folds": folds,
                                   "fold_mean_balanced_accuracy": float(np.mean([f["balanced_accuracy"] for f in folds.values()])),
                                   **ev}
    # paired comparison on trials present in both conditions
    ka_ = {(a, b): i for i, (a, b) in enumerate(zip(tables["A_current"]["participant"], tables["A_current"]["video"], strict=True))}
    kb_ = {(a, b): i for i, (a, b) in enumerate(zip(tables["B_enhanced"]["participant"], tables["B_enhanced"]["video"], strict=True))}
    common = sorted(set(ka_) & set(kb_))
    ia, ib = np.array([ka_[c] for c in common]), np.array([kb_[c] for c in common])
    y = tables["A_current"]["y"][ia]
    assert np.array_equal(y, tables["B_enhanced"]["y"][ib])
    pa, pb = tables["A_current"]["pred"][ia], tables["B_enhanced"]["pred"][ib]
    part = np.array([c[0] for c in common])
    from sklearn.metrics import balanced_accuracy_score as bas

    rng = np.random.default_rng(0)
    qs = sorted(set(part.tolist()))
    rows_by = {q: np.flatnonzero(part == q) for q in qs}
    diffs = []
    for _ in range(2000):
        rows = np.concatenate([rows_by[q] for q in rng.choice(qs, len(qs))])
        diffs.append(bas(y[rows], pb[rows]) - bas(y[rows], pa[rows]))
    per_a = [bas(y[rows_by[q]], pa[rows_by[q]]) for q in qs if len(set(y[rows_by[q]].tolist())) == 2]
    per_b = [bas(y[rows_by[q]], pb[rows_by[q]]) for q in qs if len(set(y[rows_by[q]].tolist())) == 2]
    from scipy.stats import wilcoxon

    out["paired"] = {"common_trials": len(common), "trials_only_in_A": len(set(ka_) - set(kb_)), "trials_only_in_B": len(set(kb_) - set(ka_)),
                     "balanced_accuracy_A": float(bas(y, pa)), "balanced_accuracy_B": float(bas(y, pb)),
                     "difference_B_minus_A": float(bas(y, pb) - bas(y, pa)),
                     "difference_ci95_participant_bootstrap": np.quantile(diffs, [0.025, 0.975]).tolist(),
                     "per_participant_difference_mean": float(np.mean(np.subtract(per_b, per_a))),
                     "per_participant_wilcoxon_p": float(wilcoxon(per_b, per_a).pvalue),
                     "participants_improved": int(np.sum(np.subtract(per_b, per_a) > 0)),
                     "participants_worse": int(np.sum(np.subtract(per_b, per_a) < 0))}
    (OUT / "results.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    for name, c in out["conditions"].items():
        print(name, c["segments"], c["trials"], json.dumps(c["trial_pooled"]), c["trial_pooled_ci95_participant_bootstrap"],
              round(c["fold_mean_balanced_accuracy"], 4))
    print(json.dumps(out["paired"], indent=1))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["audit", "protocol", "build", "evaluate"], required=True)
    args = ap.parse_args()
    {"audit": phase_audit, "protocol": phase_protocol, "build": phase_build, "evaluate": phase_evaluate}[args.phase](args)


if __name__ == "__main__":
    main()
