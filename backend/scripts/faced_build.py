"""Validate the local FACED v1.1.4 installation and build the FACED feature and segment caches.

For every participant in participants.tsv: parse and validate the BIDS files (channels, sampling rate, units, 28
video spans, ratings), check the microvolt amplitude scale, then extract features (pipeline.trial_features) and
125 Hz segments for the CNN. Nothing here uses the evaluation split.

Usage (from backend/):
  python -m scripts.faced_build --faced-dir <...>/Dataset/FACED/v1.1.4 [--workers 8]
Writes outputs/faced/: validation_report.json, features_faced-v1.npz, segments125_faced-v1.npy (+ _meta.npz).
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed
from scipy import signal as scipy_signal

from app.ml.faced import dataset as fd
from app.ml.faced import pipeline as fpipe

OUT = Path(__file__).resolve().parents[2] / "outputs" / "faced"


def process(root: Path, pid: str, demographics: dict) -> dict:
    try:
        p = fd.load_participant(root, pid, demographics)
    except fd.FacedError as exc:
        return {"participant_id": pid, "ok": False, "error": str(exc)}
    rec = {"participant_id": pid, "ok": True, "sampling_rate": p.sampling_rate, "n_videos": len(p.trials),
           "duration_s": round(p.header.n_samples / p.sampling_rate, 1), "problems": list(p.problems),
           "amplitude_uv": round(fpipe.amplitude_check(p), 3), "trials": []}
    feats, segs = [], []
    for t in p.trials:
        if "Valence" not in t.ratings:
            rec["trials"].append({"video_index": t.video_index, "used": False, "reason": "no valence rating"})
            continue
        if t.duration_s < fpipe.WINDOW_S:
            rec["trials"].append({"video_index": t.video_index, "used": False, "reason": "video shorter than window"})
            continue
        f, kept, rejected = fpipe.trial_features(p, t)
        s = fpipe.trial_segments(p, t)
        assert s.shape[0] == f.shape[0]
        s125 = scipy_signal.resample_poly(s, up=1, down=2, axis=-1).astype(np.float32)
        rec["trials"].append({"video_index": t.video_index, "used": f.shape[0] > 0, "segments": int(f.shape[0]),
                              "rejected_segments": rejected, "valence": t.valence, "duration_s": round(t.duration_s, 3),
                              "ratings": t.ratings, "targeted_emotion": t.targeted_emotion})
        feats.append((t, f, kept, s125))
    rec["_data"] = feats
    return rec


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--faced-dir", type=Path, required=True)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    root = args.faced_dir
    OUT.mkdir(parents=True, exist_ok=True)
    pids = fd.participant_ids(root)
    demo = fd.load_demographics(root)
    on_disk = sorted(d.name for d in root.glob("sub-*") if d.is_dir())
    results = Parallel(n_jobs=args.workers)(delayed(process)(root, pid, demo) for pid in pids)
    X, P, V, S, R, Y, SEG = [], [], [], [], [], [], []
    report_rows = []
    for r in results:
        data = r.pop("_data", [])
        report_rows.append(r)
        for t, f, kept, s125 in data:
            n = f.shape[0]
            X.append(f)
            SEG.append(s125)
            P += [r["participant_id"]] * n
            V += [t.video_index] * n
            S += kept
            R.append(np.repeat(np.array([[t.ratings.get(k, np.nan) for k in fd.RATING_KEYS]]), n, axis=0))
            Y += [fpipe.valence_label(t.valence)] * n
    X, R = np.concatenate(X).astype(np.float32), np.concatenate(R)
    np.savez_compressed(OUT / f"features_{fpipe.PIPELINE_VERSION}.npz", X=X, participant=np.array(P), video=np.array(V),
                        segment=np.array(S), ratings=R, rating_keys=np.array(fd.RATING_KEYS), y=np.array(Y),
                        feature_names=np.array(fpipe.feature_names()))
    np.save(OUT / f"segments125_{fpipe.PIPELINE_VERSION}.npy", np.concatenate(SEG))
    np.savez(OUT / f"segments125_{fpipe.PIPELINE_VERSION}_meta.npz", participant=np.array(P), video=np.array(V), y=np.array(Y))
    ok = [r for r in report_rows if r["ok"]]
    amps = np.array([r["amplitude_uv"] for r in ok])
    med = float(np.median(amps))
    for r in ok:
        if not med / 10 <= r["amplitude_uv"] <= med * 10:
            r["problems"].append(f"amplitude {r['amplitude_uv']} uV is >10x away from the median {med:.2f} uV")
    summary = {
        "dataset": fd.DATASET_ID, "version": fd.DATASET_VERSION, "pipeline": fpipe.PIPELINE_VERSION,
        "created_at": datetime.now(UTC).isoformat(), "participants_declared": len(pids),
        "participant_folders_on_disk": len(on_disk), "participants_valid": len(ok),
        "participants_failed": [(r["participant_id"], r["error"]) for r in report_rows if not r["ok"]],
        "participants_with_problems": {r["participant_id"]: r["problems"] for r in ok if r["problems"]},
        "sampling_rates": {str(k): int(v) for k, v in zip(*np.unique([r["sampling_rate"] for r in ok], return_counts=True), strict=True)},
        "amplitude_uv_median": med, "amplitude_uv_range": [float(amps.min()), float(amps.max())],
        "trials_used": int(sum(t["used"] for r in ok for t in r["trials"])),
        "trials_unused": [(r["participant_id"], t["video_index"], t.get("reason", "all segments rejected"))
                          for r in ok for t in r["trials"] if not t["used"]],
        "segments": int(X.shape[0]), "rejected_segments": int(sum(t.get("rejected_segments", 0) for r in ok for t in r["trials"])),
        "features": int(X.shape[1]), "channels": list(fd.FACED_CHANNELS),
    }
    (OUT / "validation_report.json").write_text(json.dumps({"summary": summary, "participants": report_rows}, indent=1), encoding="utf-8")
    print(json.dumps(summary, indent=1)[:4000])


if __name__ == "__main__":
    main()
