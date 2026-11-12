"""Precompute global (model-level) explanations from each model's own training data. Run from backend/:

    python -m app.xai.build_global              # both models
    python -m app.xai.build_global --only faced

Reference data, never anything held out:
- DEAP Random Forest: the participants in the model's recorded training split (the eight frozen DEAP participants are
  refused before any file is opened). TreeSHAP on every kept segment of every trial, one process per participant
  (about 4 minutes on an 8-core laptop instead of about 20 single-threaded; the result is identical).
- FACED logistic regression: the faced-protocol-v1 development participants on the development clips, read from the
  cached feature table. Rows of the 25 test participants and the 9 test clips are dropped before any value is used.

Writes outputs/cache/xai/global_<dataset>_<model>.json. Nothing is trained or changed.
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import UTC, datetime
from typing import Any

import numpy as np

from app.core.config import get_settings
from app.ml.factory import make_dataset
from app.ml.faced import pipeline as fpipe
from app.ml.participants import is_frozen
from app.services.faced_service import FacedService
from app.services.model_service import ModelService
from app.xai import attribution as A
from app.xai.service import global_path

TOP = 20


def summarise(names: list[str], channels: tuple[str, ...], per_person: dict[str, tuple[np.ndarray, np.ndarray, int]]) -> dict[str, Any]:
    """per_person: participant -> (sum |attribution| per feature, sum attribution per feature, segments)."""
    info = A.feature_info(names)
    abs_all = sum(v[0] for v in per_person.values())
    signed_all = sum(v[1] for v in per_person.values())
    n = sum(v[2] for v in per_person.values())
    mean_abs, mean_signed = abs_all / n, signed_all / n
    person_abs = {q: a / k for q, (a, _, k) in per_person.items()}
    bands = list(A.BAND_LABELS)

    def groups(key: str, kind: str | None = None) -> list[dict[str, Any]]:
        keys = sorted({f[key] for f in info if kind is None or f["kind"] == kind})
        rows = []
        for g in keys:
            idx = [j for j, f in enumerate(info) if f[key] == g and (kind is None or f["kind"] == kind)]
            per = [float(v[idx].sum()) for v in person_abs.values()]
            rows.append({"group": g, "label": A.label_for(g, key), "mean_abs": float(mean_abs[idx].sum()),
                         "mean_signed": float(mean_signed[idx].sum()), "sd_across_participants": float(np.std(per)),
                         "min_participant": float(min(per)), "max_participant": float(max(per))})
        return sorted(rows, key=lambda r: -r["mean_abs"])

    order = np.argsort(mean_abs)[::-1][:TOP]
    band_map = [[float(sum(mean_abs[j] for j, f in enumerate(info) if f["channel"] == c and f["band"] == b and f["kind"] == "channel"))
                 for b in bands] for c in channels]
    return {
        "top_features": [{**info[j], "mean_abs": float(mean_abs[j]), "mean_signed": float(mean_signed[j])} for j in order],
        "by_band": groups("band"), "by_channel": groups("channel", "channel"), "by_pair": groups("channel", "pair"),
        "by_family": groups("family"), "channels": list(channels), "bands": bands, "channel_band_map": band_map,
        "per_participant_by_band": {q: {b: float(sum(a[j] for j, f in enumerate(info) if f["band"] == b)) for b in [*bands, "broadband"]}
                                    for q, a in person_abs.items()},
        "n_segments": int(n), "n_participants": len(per_person),
    }


def _deap_participant(settings: Any, pid: str) -> tuple[str, np.ndarray, np.ndarray, int, str]:
    """TreeSHAP sums over every kept segment of one training participant (runs in its own process)."""
    assert not is_frozen(pid)
    service = ModelService(settings)
    if not service.load_active():
        raise SystemExit("no active DEAP model")
    snap = service.snapshot()
    a = s = 0.0
    k, method = 0, ""
    for record in make_dataset(settings).iter_participants({pid}):
        for t in range(record.n_trials):
            pre = snap.preprocessor.preprocess(record.eeg[t], has_baseline=True)
            if pre.segments.shape[0] == 0:
                continue
            att = A.explain(snap.model, snap.extractor.extract(pre.segments))
            assert att.max_abs_error < A.TOLERANCE
            method = att.method
            a = a + np.abs(att.values).sum(axis=0)
            s = s + att.values.sum(axis=0)
            k += att.values.shape[0]
    return pid, a, s, k, method


def build_deap(settings: Any) -> None:
    from joblib import Parallel, delayed

    service = ModelService(settings)
    if not service.load_active():
        raise SystemExit("no active DEAP model")
    snap = service.snapshot()
    if snap.extractor is None:
        raise SystemExit("the active model works on raw segments; its explanations are per recording (integrated gradients)")
    train = list(snap.meta["split"]["train"])
    assert train and not any(is_frozen(q) for q in train), "a frozen participant is in the training split"
    assert set(train) <= set(settings.development_participants), "training participant not authorised on this server"
    start = time.time()
    rows = Parallel(n_jobs=min(8, len(train)))(delayed(_deap_participant)(settings, q) for q in sorted(train))
    per = {q: (a, s, k) for q, a, s, k, _ in rows}
    method = rows[0][4]
    print(len(per), "participants,", sum(v[2] for v in per.values()), "segments,", round(time.time() - start), "s", flush=True)
    assert sorted(per) == sorted(train)
    out = {"dataset": "DEAP", "model_version": snap.version, "algorithm": snap.meta.get("algorithm"), "method": method,
           "output_kind": "probability", "created_at": datetime.now(UTC).isoformat(),
           "reference": {"description": "The model's own training participants (recorded training split); every kept segment of every trial.",
                         "participants": sorted(per)},
           **summarise(snap.extractor.feature_names, snap.extractor.channel_names, per)}
    write(global_path(settings, "deap", snap.version), out)


def build_faced(settings: Any) -> None:
    import joblib

    faced = FacedService(settings)
    card = faced.card()
    if not card or faced.incompatibility(card):
        raise SystemExit("FACED model not available or incompatible")
    split = json.loads((faced.out / "protocol_split_v1.json").read_text(encoding="utf-8"))
    z = np.load(faced.out / f"features_{fpipe.PIPELINE_VERSION}.npz", allow_pickle=False)
    p, v = z["participant"], z["video"].astype(int)
    keep = np.isin(p, split["dev_participants"]) & np.isin(v, split["dev_videos"])  # test rows are dropped here
    assert not np.isin(p[keep], split["test_participants"]).any() and not np.isin(v[keep], split["test_videos"]).any()
    names = [str(x) for x in z["feature_names"]]
    assert names == fpipe.feature_names(), "feature order differs from the FACED pipeline"
    X = z["X"][keep].astype(np.float64)
    model = joblib.load(faced.out / "models" / card["artifact"])
    att = A.explain(model, X)
    assert att.max_abs_error < A.TOLERANCE
    per = {}
    for q in sorted(set(p[keep].tolist())):
        rows = p[keep] == q
        per[q] = (np.abs(att.values[rows]).sum(axis=0), att.values[rows].sum(axis=0), int(rows.sum()))
    from app.ml.faced.dataset import FACED_CHANNELS

    out = {"dataset": "FACED", "model_version": card["name"], "algorithm": card["algorithm"].get("name"), "method": att.method,
           "output_kind": "log_odds", "created_at": datetime.now(UTC).isoformat(),
           "reference": {"description": "faced-protocol-v1 development participants on the development clips (the model's "
                                        "training data). The 25 test participants and 9 test clips are excluded.",
                         "participants": sorted(per)},
           **summarise(names, FACED_CHANNELS, per)}
    write(global_path(settings, "faced", card["name"]), out)


def write(path: Any, obj: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj, indent=1), encoding="utf-8")
    tmp.replace(path)
    print("wrote", path, flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", choices=["deap", "faced"])
    args = parser.parse_args()
    settings = get_settings()
    if args.only in (None, "faced"):
        build_faced(settings)
    if args.only in (None, "deap"):
        build_deap(settings)


if __name__ == "__main__":
    main()
