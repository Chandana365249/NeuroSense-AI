"""Train and evaluate the affect-dimension models (arousal, dominance, liking) and the confidence calibration.

Run from backend/:  python -m app.affect.build [--only deap|faced]

DEAP: the development participants on this server (never the eight frozen participants), the existing Random Forest
configuration, no tuning. Out-of-person scores come from leave-one-participant-out; the final model per dimension is
fitted on all development participants.
FACED: the faced-protocol-v1 development participants on the development clips only (test participants and test
clips are dropped before any value is used), the FACED logistic regression configuration (C = 0.01), out-of-person
scores from the protocol's five development participant groups.

For every dimension, valence included, it also fits the Platt calibration on the out-of-person trial scores, and
estimates a personalised model (trained on the same person's other rated trials, leave-one-trial-out).

Writes outputs/affect/ (model files and affect_card.json). The active valence model and the FACED valence model are
not changed: valence only gets a calibration.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from datetime import UTC, datetime
from typing import Any

import joblib
import numpy as np
from joblib import Parallel, delayed
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from app.affect import core as C
from app.core.config import get_settings
from app.ml.factory import make_dataset, make_feature_extractor, make_preprocessor
from app.ml.faced import pipeline as fpipe
from app.ml.feature_table import build_feature_table
from app.ml.models import candidate_models
from app.ml.participants import is_frozen

SEED = 42
JOBS = 8


def deap_model(n_jobs: int = 1) -> Pipeline:
    return candidate_models(SEED)["random_forest"]().estimator.set_params(clf__n_jobs=n_jobs)


def faced_model() -> Pipeline:
    return Pipeline([("scaler", StandardScaler()),
                     ("clf", LogisticRegression(C=0.01, class_weight="balanced", max_iter=5000))])


def _fit_score(make: Any, Xtr: np.ndarray, ytr: np.ndarray, Xte: np.ndarray) -> np.ndarray | None:
    if len(set(ytr.tolist())) < 2:
        return None
    model = make().fit(Xtr, ytr)
    assert list(model.classes_) == [0, 1]
    return model.predict_proba(Xte)[:, 1]


def sha256(path: Any) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run_dimension(name: str, make_fold: Any, make_final: Any | None, X: np.ndarray, y_seg: np.ndarray, p: np.ndarray,
                  t: np.ndarray, folds: list[list[str]], out_file: Any | None) -> dict[str, Any]:
    """Out-of-person scores, calibration, a personalised estimate and (optionally) the final model for one dimension."""
    start = time.time()
    oof = np.full(len(y_seg), np.nan)
    jobs = [(g, np.isin(p, g)) for g in folds]
    scores = Parallel(n_jobs=JOBS)(delayed(_fit_score)(make_fold, X[~m], y_seg[~m], X[m]) for _, m in jobs)
    for (g, m), s in zip(jobs, scores, strict=True):
        assert not set(p[~m].tolist()) & set(g), "participant leakage"
        oof[m] = s
    groups = np.char.add(np.char.add(p.astype(str), ":"), t.astype(str))
    keys, raw = C.trial_scores(groups, oof)
    tp = np.array([k.split(":")[0] for k in keys])
    tt = np.array([int(k.split(":")[1]) for k in keys])
    ty = np.array([int(y_seg[groups == k][0]) for k in keys])
    card: dict[str, Any] = {
        "evaluation": C.summarise_scores(tp, ty, raw),
        "calibration": {"method": "Platt scaling on out-of-person trial scores", **C.fit_platt(raw, ty)},
        "out_of_person_scores": {q: {str(int(k)): round(float(s), 6) for k, s in zip(tt[tp == q], raw[tp == q], strict=True)}
                                 for q in sorted(set(tp.tolist()))},
    }

    # Personalised: for each trial, a model trained on the same person's other trials only.
    tasks = [(q, k) for q in sorted(set(tp.tolist())) for k in tt[tp == q]]
    pers = Parallel(n_jobs=JOBS)(delayed(_fit_score)(make_fold, X[(p == q) & (t != k)], y_seg[(p == q) & (t != k)],
                                                     X[(p == q) & (t == k)]) for q, k in tasks)
    ok = [i for i, s in enumerate(pers) if s is not None]
    pp = np.array([tasks[i][0] for i in ok])
    py = np.array([ty[(tp == tasks[i][0]) & (tt == tasks[i][1])][0] for i in ok])
    ps = np.array([float(pers[i].mean()) for i in ok])
    card["personal"] = {"design": "per person, leave-one-trial-out: trained on that person's other rated trials",
                        "trials_skipped_single_label": len(tasks) - len(ok),
                        "evaluation": C.summarise_scores(pp, py, ps),
                        "calibration": {"method": "Platt scaling on personalised out-of-trial scores", **C.fit_platt(ps, py)}}

    if make_final is not None and out_file is not None:
        joblib.dump(make_final().fit(X, y_seg), out_file, compress=3)
        card["model_file"] = out_file.name
        card["sha256"] = sha256(out_file)
    e, pe = card["evaluation"], card["personal"]["evaluation"]
    print(f"  {name}: out-of-person BA {e['balanced_accuracy_per_participant']['mean']:.3f}, AUC {e['pooled_auc']:.3f}, "
          f"ECE {e['ece_raw']:.3f}->{e['ece_calibrated']:.3f}; personal BA {pe['balanced_accuracy_per_participant']['mean']:.3f} "
          f"({round(time.time() - start)} s)", flush=True)
    return card


def build_deap(settings: Any, out: Any) -> dict[str, Any]:
    dev = sorted(settings.development_participants)
    assert dev and not any(is_frozen(q) for q in dev), "a frozen participant is configured as development"
    table = build_feature_table(make_dataset(settings), make_preprocessor(settings), make_feature_extractor(settings),
                                settings.cache_dir)
    keep = np.isin(table.participant, dev)
    assert not any(is_frozen(q) for q in set(table.participant.tolist()))
    X, R, p, t = table.X[keep].astype(np.float64), table.ratings[keep], table.participant[keep], table.trial[keep]
    people = sorted(set(p.tolist()))
    print(f"DEAP: {len(people)} development participants, {len(X)} segments", flush=True)
    dims = {}
    for d in C.DEAP_DIMENSIONS:
        y = (R[:, d.column] > d.threshold).astype(int)
        final = None if d.key == "valence" else (lambda: deap_model(-1))
        dims[d.key] = {"label": d.label, "threshold": d.threshold,
                       "model": "active valence model (unchanged)" if d.key == "valence" else "Random Forest (existing configuration)",
                       **run_dimension(d.key, deap_model, final, X, y, p, t, [[q] for q in people],
                                       None if d.key == "valence" else out / f"deap_{d.key}.joblib")}
    return {"scale": "1-9", "participants": people, "segments": int(len(X)), "feature_count": int(X.shape[1]),
            "folds": "leave-one-participant-out", "feature_fingerprint": table.fingerprint, "dimensions": dims}


def build_faced(settings: Any, out: Any) -> dict[str, Any]:
    from app.services.faced_service import FacedService

    fout = FacedService(settings).out
    split = json.loads((fout / "protocol_split_v1.json").read_text(encoding="utf-8"))
    z = np.load(fout / f"features_{fpipe.PIPELINE_VERSION}.npz", allow_pickle=False)
    names = [str(x) for x in z["feature_names"]]
    assert names == fpipe.feature_names(), "feature order differs from the FACED pipeline"
    p_all, v_all = z["participant"], z["video"].astype(int)
    keep = np.isin(p_all, split["dev_participants"]) & np.isin(v_all, split["dev_videos"])  # test rows dropped here
    assert not np.isin(p_all[keep], split["test_participants"]).any() and not np.isin(v_all[keep], split["test_videos"]).any()
    X, p, v = z["X"][keep].astype(np.float64), p_all[keep], v_all[keep]
    keys = [str(k) for k in z["rating_keys"]]
    folds = [list(g) for g in split["inner_participant_groups"]]
    assert sorted(q for g in folds for q in g) == sorted(set(p.tolist()))
    print(f"FACED: {len(set(p.tolist()))} development participants x {len(set(v.tolist()))} clips, {len(X)} segments", flush=True)
    dims = {}
    for d in C.FACED_DIMENSIONS:
        y = (z["ratings"][keep][:, keys.index(C.FACED_RATING_KEYS[d.key])] > d.threshold).astype(int)
        final = None if d.key == "valence" else faced_model
        dims[d.key] = {"label": d.label, "threshold": d.threshold,
                       "model": "FACED valence model (unchanged)" if d.key == "valence" else "logistic regression (C = 0.01, FACED configuration)",
                       **run_dimension(d.key, faced_model, final, X, y, p, v, folds,
                                       None if d.key == "valence" else out / f"faced_{d.key}.joblib")}
    return {"scale": "0-7", "participants": sorted(set(p.tolist())), "clips": sorted({int(x) for x in v}),
            "segments": int(len(X)), "feature_count": int(X.shape[1]), "pipeline": fpipe.PIPELINE_VERSION,
            "folds": "faced-protocol-v1 development participant groups (5)", "dimensions": dims,
            "excluded": "the 25 test participants and the 9 test clips"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", choices=["deap", "faced"])
    args = parser.parse_args()
    settings = get_settings()
    out = settings.output_dir / "affect"
    out.mkdir(parents=True, exist_ok=True)
    path = out / "affect_card.json"
    card = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    if args.only in (None, "faced"):
        card["faced"] = build_faced(settings, out)
    if args.only in (None, "deap"):
        card["deap"] = build_deap(settings, out)
    card |= {"created_at": datetime.now(UTC).isoformat(), "seed": SEED,
             "note": ("Development participants only. Out-of-person scores are exploratory cross-validation, not an "
                      "independent test. Calibrated probabilities are estimated frequencies, not certainty.")}
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(card, indent=1), encoding="utf-8")
    tmp.replace(path)
    print("wrote", path, flush=True)


if __name__ == "__main__":
    main()
