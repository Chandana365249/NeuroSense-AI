"""Train the frozen FACED configuration on ALL 123 participants x 28 clips (requested after the held-out evaluation).

Same faced-v2 features, label rule and classifier (logistic regression, C = 0.01). No tuning or model selection. The
artifact goes to outputs/faced/all_participants/ and is never registered, activated or used by the application.

With nobody held out, this model has NO independent test score. To estimate performance, the same configuration is
cross-validated on the 123 participants, in two 5-fold designs fixed in this file (seed 20261101):
- unseen people: each validation fold's people are absent from training;
- unseen people and clips: the validation fold's people AND clips are absent from training.
These estimates are EXPLORATORY: the 25 earlier test participants were already examined. They do not replace the
strict held-out 56.6%, which belongs to a different model fit.

Usage (from backend/): python -m scripts.faced_train_all
"""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from typing import Any

import joblib
import numpy as np

from app.ml.provenance import git_commit
from scripts import faced_experiment as fe
from scripts import model_improvement as mi

OUT = fe.OUT / "all_participants"
CFG = next(c for c in fe.CANDIDATES if c["name"] == "lr_C0.01")
CV_SEED = 20261101
FOLDS = 5


def cross_validate(d: dict[str, Any], hold_clips: bool) -> dict[str, Any]:
    rng = np.random.default_rng(CV_SEED)
    people = np.array_split(rng.permutation(sorted(set(d["p"].tolist()))), FOLDS)
    clips = np.array_split(rng.permutation(sorted(set(d["v"].tolist()))), FOLDS)
    scores, seen, per_fold = np.full(len(d["y"]), np.nan), np.zeros(len(d["y"]), bool), []
    for k in range(FOLDS):
        val_p, val_v = np.isin(d["p"], people[k]), np.isin(d["v"], clips[k])
        tr = ~val_p & ~val_v if hold_clips else ~val_p
        va = val_p & val_v if hold_clips else val_p
        assert not set(d["p"][tr]) & set(d["p"][va]), "participant leakage"
        if hold_clips:
            assert not set(d["v"][tr].tolist()) & set(d["v"][va].tolist()), "clip leakage"
        scores[va] = fe.fit_predict(CFG, d["X"][tr], d["y"][tr], d["X"][va])
        seen |= va
        t = fe.trial_table(d["p"][va], d["v"][va], d["y"][va], scores[va])
        per_fold.append(fe.metrics(t["y"], t["score"], t["pred"])["balanced_accuracy"])
    t = fe.trial_table(d["p"][seen], d["v"][seen], d["y"][seen], scores[seen])
    ev = fe.evaluate(t, d["y"][seen], scores[seen])
    ev.pop("per_participant")
    return {"fold_balanced_accuracy": per_fold, "trials": int(len(t["y"])), **ev}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    d = fe.load()
    people, clips = sorted(set(d["p"].tolist())), sorted(set(d["v"].tolist()))
    assert len(people) == 123 and len(clips) == 28
    model = mi.make_model(CFG, fe.SEED).fit(d["X"], d["y"])
    path = OUT / "faced_all123_lr_C0.01.joblib"
    joblib.dump(model, path)
    t = fe.trial_table(d["p"], d["v"], d["y"], mi.seg_scores(model, CFG, d["X"]))
    results = {
        "created_at": datetime.now(UTC).isoformat(), "git_commit": git_commit(), "config": CFG, "seed": fe.SEED,
        "participants": len(people), "clips": len(clips), "segments": int(len(d["y"])), "trials": int(len(t["y"])),
        "artifact_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "status": "EXPERIMENTAL: not registered, not active, not used by the application; no independent test score",
        "training_fit_NOT_a_performance_estimate": fe.metrics(t["y"], t["score"], t["pred"]),
        "exploratory_cv_unseen_people": cross_validate(d, hold_clips=False),
        "exploratory_cv_unseen_people_and_clips": cross_validate(d, hold_clips=True),
    }
    (OUT / "results.json").write_text(json.dumps(results, indent=1), encoding="utf-8")
    for key in ("exploratory_cv_unseen_people", "exploratory_cv_unseen_people_and_clips"):
        r = results[key]
        print(key, r["trials"], json.dumps(r["trial_pooled"]), r["trial_pooled_ci95_participant_bootstrap"],
              "per-participant BA", round(r["per_participant_balanced_accuracy"]["mean"], 4), r["per_participant_balanced_accuracy"]["ci95_t"],
              "folds", [round(x, 3) for x in r["fold_balanced_accuracy"]])
    print("training fit (not performance)", json.dumps(results["training_fit_NOT_a_performance_estimate"]))
    print("artifact", path.name, results["participants"], "participants", results["trials"], "trials")


if __name__ == "__main__":
    main()
