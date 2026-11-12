"""Development-only optimisation experiment (FACED faced-protocol-v1 development people x development clips only).

No FACED test participant, no FACED test clip and no DEAP file is read for computation. The shared FACED cache files
hold every participant, so rows outside development people x development clips are dropped as soon as the participant
and clip columns are read, before any feature, label or score of them is used.

From backend/:
  python -m research.dev_optimisation.run --phase manifest   # split manifest, participant list, presentation orders
  python -m research.dev_optimisation.run --phase classical  # logistic regression: A, B, D (C is computed in report)
  python -m research.dev_optimisation.run --phase neural     # FusionNet erm: A (3 seeds), B, D, H; resumable
  python -m research.dev_optimisation.run --phase report     # metrics, CSVs, plots; no training
"""

from __future__ import annotations

import torch  # noqa: I001  (before the scripts import, which limits BLAS threads)

import argparse
import copy
import csv
import json
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import roc_auc_score

from app.ml.faced import pipeline as fpipe
from app.ml.faced import protocol as fprot
from app.ml.provenance import git_commit
from research.dev_optimisation import methods as M
from research.neurosense_fusion import adapters as ad
from research.neurosense_fusion import model as fm
from research.neurosense_fusion import run as frun
from scripts import faced_calibration_controls as controls
from scripts import faced_experiment as fe
from scripts import model_improvement as mi

ROOT = ad.ROOT
OUT = ROOT / "outputs" / "experiments" / "neurosense_dev_optimisation"
FACED_OUT = ROOT / "outputs" / "faced"
SEED = 42
FOLDS = (0, 1, 2, 3, 4)
LR = {"name": "lr_C0.01", "family": "lr", "C": 0.01}  # the existing S1 classical baseline, unchanged
NEURAL_SEEDS = (42, 43, 44)
EARLY = {"max_epochs": 20, "patience": 5, "stop_group": "inner fold (k + 1) mod 5: its people on its clips"}
NEURAL_ITEMS = [("A", 42), ("B", 42), ("D", 42), ("H", 42), ("A", 43), ("A", 44)]


def write(name: str, obj: Any) -> None:
    path = OUT / name
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1, default=str), encoding="utf-8")
    tmp.replace(path)


def read(name: str) -> Any:
    return json.loads((OUT / name).read_text(encoding="utf-8"))


def load_split() -> dict[str, Any]:
    split = json.loads((FACED_OUT / "protocol_split_v1.json").read_text(encoding="utf-8"))
    sha = fprot.FacedSplit(**{k: (tuple(tuple(x) if isinstance(x, list) else x for x in v) if isinstance(v, list) else v)
                              for k, v in split.items()}).sha256
    assert sha == fe.FROZEN_SPLIT_SHA, "the split differs from faced-protocol-v1"
    return split


def load_dev(features: bool = False, segments: bool = False) -> dict[str, Any]:
    split = load_split()
    z = np.load(FACED_OUT / f"features_{fpipe.PIPELINE_VERSION}.npz", allow_pickle=False)
    p_all, v_all = z["participant"], z["video"].astype(int)
    idx = np.flatnonzero(M.dev_mask(p_all, v_all, split))  # everything else is discarded here
    d: dict[str, Any] = {"split": split, "p": p_all[idx], "v": v_all[idx], "seg": z["segment"][idx].astype(int),
                         "y": z["y"][idx].astype(int)}
    assert set(d["p"].tolist()) == set(split["dev_participants"]) and set(d["v"].tolist()) <= set(split["dev_videos"])
    if features:
        d["X"] = z["X"][idx].astype(np.float64)
    if segments:
        meta = np.load(FACED_OUT / f"segments125_{fpipe.PIPELINE_VERSION}_meta.npz")
        assert np.array_equal(meta["participant"], p_all) and np.array_equal(meta["video"], v_all)
        raw = np.load(FACED_OUT / f"segments125_{fpipe.PIPELINE_VERSION}.npy", mmap_mode="r")
        d["raw"] = raw  # memory-mapped; only rows in d["idx"] are ever indexed
        d["idx"] = idx
    d["order"] = {q: o for q, o in read("presentation_orders.json").items()} if (OUT / "presentation_orders.json").exists() else {}
    return d


def raw_rows(d: dict[str, Any], rows: np.ndarray) -> np.ndarray:
    if "raw_dev" in d:
        return d["raw_dev"][rows]
    return np.asarray(d["raw"][d["idx"][rows]], dtype=np.float32)


# ---------------------------------------------------------------- manifest

def phase_manifest(_: argparse.Namespace) -> None:
    split = load_split()
    z = np.load(FACED_OUT / f"features_{fpipe.PIPELINE_VERSION}.npz", allow_pickle=False)
    p_all, v_all = z["participant"], z["video"].astype(int)
    keep = M.dev_mask(p_all, v_all, split)
    everyone = sorted(set(p_all.tolist()))
    included = sorted(set(p_all[keep].tolist()))
    excluded = {q: "faced-protocol-v1 test participant (examined in earlier work)" for q in split["test_participants"]}
    assert set(included) == set(split["dev_participants"]) and set(included) | set(excluded) == set(everyone)
    orders = {}
    for q in included:
        s = controls.session(q)
        orders[q] = [int(c) for c in s["order"]]  # presentation order from recording timestamps; no rating is read
        assert sorted(orders[q]) == list(range(1, 29)), q
    write("presentation_orders.json", orders)
    groups = [{"fold": k, "validation_people": split["inner_participant_groups"][k], "validation_clips": split["inner_video_groups"][k]}
              for k in FOLDS]
    trials = {q: int(len(set(v_all[keep & (p_all == q)].tolist()))) for q in included}
    write("split_manifest.json", {
        "created_at": datetime.now(UTC).isoformat(), "git_commit": git_commit(), "protocol": split["protocol_version"],
        "split_sha256": fe.FROZEN_SPLIT_SHA, "source": "outputs/faced/protocol_split_v1.json",
        "rows_used": "development participants x development clips only",
        "included_participants": included, "excluded_participants": excluded,
        "development_clips": split["dev_videos"], "excluded_clips": {c: "faced-protocol-v1 test clip" for c in split["test_videos"]},
        "folds": groups, "development_clips_per_participant": trials, "segments_used": int(keep.sum()),
        "deap": "not used: its 12 authorised participants were the external test of the Fusion experiment; the frozen 8 are never read",
        "presentation_order": "from each development participant's own events file timestamps; ratings are not read"})
    with (OUT / "participants.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["participant", "status", "reason", "validation_fold", "development_clips"])
        fold_of = {q: k for k in FOLDS for q in split["inner_participant_groups"][k]}
        for q in everyone:
            if q in excluded:
                w.writerow([q, "excluded", excluded[q], "", ""])
            else:
                w.writerow([q, "included", "faced-protocol-v1 development participant", fold_of[q], trials[q]])
    print(f"{len(included)} included, {len(excluded)} excluded, {int(keep.sum())} segments")


# ---------------------------------------------------------------- classical (logistic regression)

def lr_scores(Xtr: np.ndarray, ytr: np.ndarray, Xte: np.ndarray) -> np.ndarray:
    model = mi.make_model(LR, SEED).fit(Xtr, ytr)
    return mi.seg_scores(model, LR, Xte)


def phase_classical(_: argparse.Namespace) -> None:
    d = load_dev(features=True, segments=True)
    assert d["order"], "run --phase manifest first"
    p, v, y, X = d["p"], d["v"], d["y"], d["X"]
    start = time.time()
    cov = np.empty((len(y), len(M.BANDS), 30, 30))
    for i in range(0, len(y), 1000):
        rows = np.arange(i, min(i + 1000, len(y)))
        cov[rows] = M.band_covariances(raw_rows(d, rows))
    log_cov = M.logm(cov)
    print("covariances", round(time.time() - start), "s", flush=True)
    for k in FOLDS:
        path = OUT / "classical" / f"fold{k}.npz"
        if path.exists():
            continue
        start = time.time()
        tr, va = M.fold_rows(p, v, d["split"], k)
        held = sorted(set(p[va].tolist()))
        out: dict[str, np.ndarray] = {"val_rows": np.flatnonzero(va), "held_rows": np.flatnonzero(np.isin(p, held))}
        # A baseline: unchanged S1 logistic regression
        out["A_held"] = lr_scores(X[tr], y[tr], X[out["held_rows"]])
        # B: each training person z-scored on their own training rows; held-out people batch or online
        Xb = X.copy()
        for q in set(p[tr].tolist()):
            rows = tr & (p == q)
            Xb[rows] = M.standardise(X[rows], *M.moments(X[rows], (0,)))
        pooled = M.moments(X[tr], (0,))
        model = mi.make_model(LR, SEED).fit(Xb[tr], y[tr])
        # D: covariance tangent features; D_global = one reference from training rows (control)
        ref_global = M.invsqrtm(M.log_euclidean_mean(log_cov[tr]))
        Fg = M.tangent_features(cov, ref_global)
        out["D_global_val"] = lr_scores(Fg[tr], y[tr], Fg[va])
        del Fg
        Fd = np.empty((len(y), len(M.BANDS) * 465))
        for q in set(p[tr].tolist()):
            rows = np.flatnonzero(tr & (p == q))
            Fd[rows] = M.tangent_features(cov[rows], M.invsqrtm(M.log_euclidean_mean(log_cov[rows])))
        d_model = mi.make_model(LR, SEED).fit(Fd[tr], y[tr])
        fallbacks = {"B_online": 0, "D_online": 0}
        for mode in ("batch", "online"):
            sb, sd_ = np.empty(int(va.sum())), np.empty(int(va.sum()))
            val_idx = out["val_rows"]
            for q in held:
                person = np.flatnonzero(p == q)
                for c in sorted(set(v[va & (p == q)].tolist())):
                    target = val_idx[(p[val_idx] == q) & (v[val_idx] == c)]
                    pos = np.searchsorted(val_idx, target)
                    ref = M.reference_rows(mode, d["order"][q], c, person, v)
                    if ref is None:
                        fallbacks[f"B_{mode}"] = fallbacks.get(f"B_{mode}", 0) + 1
                        fallbacks[f"D_{mode}"] = fallbacks.get(f"D_{mode}", 0) + 1
                        stats, ref_isqrt = pooled, ref_global
                    else:
                        stats, ref_isqrt = M.moments(X[ref], (0,)), M.invsqrtm(M.log_euclidean_mean(log_cov[ref]))
                    sb[pos] = mi.seg_scores(model, LR, M.standardise(X[target], *stats))
                    sd_[pos] = mi.seg_scores(d_model, LR, M.tangent_features(cov[target], ref_isqrt))
            out[f"B_{mode}_val"], out[f"D_{mode}_val"] = sb, sd_
        del Fd
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, **out, fallback_trials=json.dumps(fallbacks), seconds=time.time() - start)
        print("classical fold", k, round(time.time() - start), "s", fallbacks, flush=True)


# ---------------------------------------------------------------- neural (FusionNet erm)

def fit_early_stopping(x: torch.Tensor, y: np.ndarray, x_stop: torch.Tensor, stop_p: np.ndarray, stop_v: np.ndarray,
                       stop_y: np.ndarray, seed: int) -> tuple[fm.FusionNet, int, list[float]]:
    """The erm training loop of research.neurosense_fusion.run.fit, with one validation pass per epoch on a separate
    group of training people and clips. Keeps the epoch with the best trial balanced accuracy (earliest on ties)."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    net = fm.FusionNet(n_domains=frun.TRAIN["n_domains"])
    target = torch.from_numpy(y.astype(np.float32))
    weight = torch.tensor(float((y == 0).sum() / max((y == 1).sum(), 1)))
    params = [*net.encoder.parameters(), *net.head.parameters(), *net.domain_head.parameters()]
    opt = torch.optim.AdamW(params, lr=frun.TRAIN["lr"], weight_decay=frun.TRAIN["weight_decay"])
    batch = frun.TRAIN["batch"]
    best, best_epoch, best_state, curve = -1.0, 0, None, []
    for epoch in range(EARLY["max_epochs"]):
        net.train()
        order = rng.permutation(len(x))
        for i in range(0, len(order) - batch + 1, batch):
            rows = order[i:i + batch]
            logit, _ = net(x[rows])
            loss = torch.nn.functional.binary_cross_entropy_with_logits(logit, target[rows], pos_weight=weight)
            opt.zero_grad()
            loss.backward()
            opt.step()
        net.eval()
        t = fe.trial_table(stop_p, stop_v, stop_y, frun.scores(net, x_stop))
        ba = float(fe.metrics(t["y"], t["score"], t["pred"])["balanced_accuracy"])
        curve.append(ba)
        if ba > best:
            best, best_epoch, best_state = ba, epoch + 1, copy.deepcopy(net.state_dict())
        elif epoch + 1 - best_epoch >= EARLY["patience"]:
            break
    net.load_state_dict(best_state)
    return net.eval(), best_epoch, curve


def per_person_transform(kind: str, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Statistics of one person's reference segments: B -> per-channel mean and SD after per-segment normalisation;
    D -> R^-1/2 of the mean shrunk spatial covariance of the raw segments."""
    if kind == "B":
        mean, sd = M.moments(ad.normalise_segments(x), (0, 2))
        return mean[:, None], sd[:, None]
    ref = M.shrunk_covariance(x.astype(np.float64)).mean(axis=0)
    return ref, M.invsqrtm(ref)


def apply_transform(kind: str, x: np.ndarray, stats: tuple[np.ndarray, np.ndarray]) -> np.ndarray:
    if kind == "B":
        return M.standardise(ad.normalise_segments(x), *stats).astype(np.float32)
    return ad.normalise_segments(M.euclidean_alignment(x, stats[1]))


def neural_inputs(kind: str, d: dict[str, Any], rows: np.ndarray, stats: dict[str, Any] | None = None) -> np.ndarray:
    x = raw_rows(d, rows)
    if kind in ("A", "H"):
        return ad.normalise_segments(x)
    out = np.empty(x.shape, np.float32)
    p = d["p"][rows]
    for q in set(p.tolist()):
        mine = p == q
        out[mine] = apply_transform(kind, x[mine], stats[q] if stats else per_person_transform(kind, x[mine]))
    return out


def phase_neural(_: argparse.Namespace) -> None:
    d = load_dev(segments=True)
    assert d["order"], "run --phase manifest first"
    d["raw_dev"] = np.asarray(d["raw"][d["idx"]], dtype=np.float32)  # development rows only, read once
    p, v, y, seg = d["p"], d["v"], d["y"], d["seg"]
    for kind, seed in NEURAL_ITEMS:
        for k in FOLDS:
            path = OUT / "neural" / f"{kind}_seed{seed}_fold{k}.npz"
            if path.exists():
                continue
            start = time.time()
            tr, va = M.fold_rows(p, v, d["split"], k)
            held = sorted(set(p[va].tolist()))
            held_rows = np.flatnonzero(np.isin(p, held))
            val_rows = np.flatnonzero(va)
            extra: dict[str, Any] = {}
            if kind == "H":  # choose the epoch count on a split inside the training rows, then retrain on all of them
                j = (k + 1) % len(d["split"]["inner_participant_groups"])
                fit_rows, _ = M.fold_rows(p, v, d["split"], k, exclude=j)
                stop = np.isin(p, d["split"]["inner_participant_groups"][j]) & np.isin(v, d["split"]["inner_video_groups"][j])
                assert not (stop & fit_rows).any() and not (stop & va).any() and stop[tr].sum() == stop.sum()
                _, epoch, curve = fit_early_stopping(torch.from_numpy(neural_inputs("H", d, np.flatnonzero(fit_rows))), y[fit_rows],
                                                     torch.from_numpy(neural_inputs("H", d, np.flatnonzero(stop))),
                                                     p[stop], v[stop], y[stop], seed)
                net = frun.fit({}, torch.from_numpy(neural_inputs("H", d, np.flatnonzero(tr))), y[tr], p[tr], v[tr], seg[tr], seed, epochs=epoch)
                extra = {"best_epoch": epoch, "curve": json.dumps(curve), "fit_segments": int(fit_rows.sum())}
            else:
                xtr = torch.from_numpy(neural_inputs(kind, d, np.flatnonzero(tr)))
                net = frun.fit({}, xtr, y[tr], p[tr], v[tr], seg[tr], seed)
                del xtr
            if kind in ("A", "H"):
                extra["held_scores"] = frun.scores(net, torch.from_numpy(neural_inputs(kind, d, held_rows)))
            else:
                train_stats = {q: per_person_transform(kind, raw_rows(d, np.flatnonzero(tr & (p == q)))) for q in set(p[tr].tolist())}
                fallback = tuple(np.mean([s[i] for s in train_stats.values()], axis=0) for i in range(2))
                if kind == "D":
                    fallback = (fallback[0], M.invsqrtm(fallback[0]))
                fallbacks = 0
                for mode in ("batch", "online"):
                    s = np.empty(len(val_rows))
                    for q in held:
                        person = np.flatnonzero(p == q)
                        person_raw = raw_rows(d, person)
                        for c in sorted(set(v[va & (p == q)].tolist())):
                            target = (p[val_rows] == q) & (v[val_rows] == c)
                            ref = M.reference_rows(mode, d["order"][q], c, person, v)
                            if ref is None:
                                fallbacks += 1
                                stats = fallback
                            else:
                                stats = per_person_transform(kind, person_raw[np.isin(person, ref)])
                            x = apply_transform(kind, raw_rows(d, val_rows[target]), stats)
                            s[target] = frun.scores(net, torch.from_numpy(x))
                    extra[f"{mode}_val_scores"] = s
                extra["online_fallback_trials"] = fallbacks
            path.parent.mkdir(parents=True, exist_ok=True)
            np.savez(path, val_rows=val_rows, held_rows=held_rows, seconds=time.time() - start, **extra)
            print(kind, seed, k, round(time.time() - start), "s", {a: b for a, b in extra.items() if a in ("best_epoch", "online_fallback_trials")}, flush=True)


# ---------------------------------------------------------------- report

def trial_records(d: dict[str, Any], k: int, rows: np.ndarray, s: np.ndarray, threshold: dict[tuple[str, int], float] | None = None) -> list[dict[str, Any]]:
    scores = M.trial_scores(d["p"][rows], d["v"][rows], s)
    label = {(q, int(c)): int(t) for q, c, t in zip(d["p"][rows].tolist(), d["v"][rows].tolist(), d["y"][rows].tolist(), strict=True)}
    out = []
    for key, score in sorted(scores.items()):
        t = 0.5 if threshold is None else threshold[key]
        out.append({"fold": k, "participant": key[0], "clip": key[1], "y": label[key], "score": score - t + 0.5, "pred": int(score > t)})
    return out


def score_thresholds(d: dict[str, Any], k: int, held_rows: np.ndarray, held_scores: np.ndarray, rule: str, mode: str) -> tuple[dict[tuple[str, int], float], int]:
    """Per-person thresholds for the validation trials, from that person's unlabelled trial scores only."""
    tr, va = M.fold_rows(d["p"], d["v"], d["split"], k)
    t_train = M.trial_scores(d["p"][tr], d["v"][tr], d["y"][tr].astype(float))
    high_share = float(np.mean(list(t_train.values())))  # training people's labels only
    person_scores = M.trial_scores(d["p"][held_rows], d["v"][held_rows], held_scores)
    out, fallbacks = {}, 0
    unseen = set(d["split"]["inner_video_groups"][k])
    for (q, c) in sorted(set(zip(d["p"][va].tolist(), d["v"][va].tolist(), strict=True))):
        mine = {cc: s for (qq, cc), s in person_scores.items() if qq == q}
        if mode == "batch":
            ref = np.array(list(mine.values()))
        elif mode == "batch_unseen":  # only clips the model never trained on: the validation clips themselves
            ref = np.array([s for cc, s in mine.items() if cc in unseen])
        else:
            earlier = M.earlier_clips(d["order"][q], c, set(mine))
            ref = np.array([mine[cc] for cc in earlier]) if len(earlier) >= M.MIN_HISTORY else None
        fallbacks += ref is None
        out[(q, int(c))] = M.person_threshold(ref, rule, high_share)
    return out, fallbacks


METHODS: list[dict[str, str]] = []


def describe(family: str, name: str, what: str, uses: str) -> None:
    METHODS.append({"family": family, "method": name, "what": what, "unlabelled_eeg_of_new_person": uses})


BATCH = "all of the person's development-clip EEG, including the clips being predicted"
BATCH_UNSEEN = "the person's EEG on the clips held out from training in this fold only (the clips being predicted)"
ONLINE = "only the person's development clips shown before the clip being predicted (population statistics until 2 exist)"
for fam in ("classical", "neural"):
    describe(fam, "A_baseline", "existing baseline, unchanged", "none")
    describe(fam, "B_batch", "per-person standardisation of inputs", BATCH)
    describe(fam, "B_online", "per-person standardisation of inputs", ONLINE)
    for rule in ("median", "base_rate"):
        describe(fam, f"C_{rule}_batch", f"per-person score threshold ({rule}) on baseline scores", BATCH)
        describe(fam, f"C_{rule}_batch_unseen", f"per-person score threshold ({rule}) on baseline scores", BATCH_UNSEEN)
        describe(fam, f"C_{rule}_online", f"per-person score threshold ({rule}) on baseline scores", ONLINE)
    if fam == "classical":
        describe(fam, "D_global_control", "covariance tangent features, one reference from training rows (no alignment)", "none")
    describe(fam, "D_batch", "per-person covariance alignment", BATCH)
    describe(fam, "D_online", "per-person covariance alignment", ONLINE)
describe("neural", "A_seed43", "baseline with seed 43 (descriptive)", "none")
describe("neural", "A_seed44", "baseline with seed 44 (descriptive)", "none")
describe("neural", "F_ensemble_3_seeds", "mean segment probability of seeds 42, 43, 44", "none")
describe("neural", "H_early_stopping", "epoch count (up to 20, patience 5) chosen on a separate group of training people and clips, "
         "then retrained on the full fold training rows", "none")


def collect(d: dict[str, Any]) -> tuple[dict[tuple[str, str], list[dict[str, Any]]], dict[str, Any]]:
    trials: dict[tuple[str, str], list[dict[str, Any]]] = {}
    info: dict[str, Any] = {"fallback_trials": {}, "seconds": {}, "early_stopping": {}}

    def add(fam: str, name: str, recs: list[dict[str, Any]]) -> None:
        trials.setdefault((fam, name), []).extend(recs)

    for k in FOLDS:
        c = np.load(OUT / "classical" / f"fold{k}.npz")
        val, held = c["val_rows"], c["held_rows"]
        held_val = np.isin(held, val)
        assert np.array_equal(held[held_val], val)
        add("classical", "A_baseline", trial_records(d, k, val, c["A_held"][held_val]))
        for name in ("B_batch", "B_online", "D_batch", "D_online"):
            add("classical", name, trial_records(d, k, val, c[f"{name}_val"]))
        add("classical", "D_global_control", trial_records(d, k, val, c["D_global_val"]))
        info["seconds"][f"classical_fold{k}"] = float(c["seconds"])
        for name, n in json.loads(str(c["fallback_trials"])).items():
            info["fallback_trials"][f"classical_{name}"] = info["fallback_trials"].get(f"classical_{name}", 0) + n
        for fam, held_scores in (("classical", c["A_held"]), ("neural", np.load(OUT / "neural" / f"A_seed42_fold{k}.npz")["held_scores"])):
            for rule in ("median", "base_rate"):
                for mode in ("batch", "batch_unseen", "online"):
                    th, fb = score_thresholds(d, k, held, held_scores, rule, mode)
                    add(fam, f"C_{rule}_{mode}", trial_records(d, k, val, held_scores[held_val], th))
                    info["fallback_trials"][f"{fam}_C_{rule}_{mode}"] = info["fallback_trials"].get(f"{fam}_C_{rule}_{mode}", 0) + fb
        seeds = []
        for seed in NEURAL_SEEDS:
            a = np.load(OUT / "neural" / f"A_seed{seed}_fold{k}.npz")
            assert np.array_equal(a["held_rows"], held) and np.array_equal(a["val_rows"], val)
            seeds.append(a["held_scores"][held_val])
            add("neural", "A_baseline" if seed == 42 else f"A_seed{seed}", trial_records(d, k, val, seeds[-1]))
            info["seconds"][f"neural_A_seed{seed}_fold{k}"] = float(a["seconds"])
        add("neural", "F_ensemble_3_seeds", trial_records(d, k, val, np.mean(seeds, axis=0)))
        for kind in ("B", "D"):
            b = np.load(OUT / "neural" / f"{kind}_seed42_fold{k}.npz")
            for mode in ("batch", "online"):
                add("neural", f"{kind}_{mode}", trial_records(d, k, val, b[f"{mode}_val_scores"]))
            info["fallback_trials"][f"neural_{kind}_online"] = info["fallback_trials"].get(f"neural_{kind}_online", 0) + int(b["online_fallback_trials"])
            info["seconds"][f"neural_{kind}_fold{k}"] = float(b["seconds"])
        h = np.load(OUT / "neural" / f"H_seed42_fold{k}.npz")
        add("neural", "H_early_stopping", trial_records(d, k, val, h["held_scores"][held_val]))
        info["early_stopping"][str(k)] = {"best_epoch": int(h["best_epoch"]), "stop_curve": json.loads(str(h["curve"])), "fit_segments": int(h["fit_segments"])}
        info["seconds"][f"neural_H_fold{k}"] = float(h["seconds"])
    return trials, info


def summarise(recs: list[dict[str, Any]]) -> dict[str, Any]:
    y = np.array([r["y"] for r in recs])
    pred = np.array([r["pred"] for r in recs])
    score = np.array([r["score"] for r in recs])
    out = M.binary_metrics(y, pred)
    out["roc_auc"] = float(roc_auc_score(y, score))
    people = sorted({r["participant"] for r in recs})
    per = {}
    for q in people:
        mine = [r for r in recs if r["participant"] == q]
        yy, pp = np.array([r["y"] for r in mine]), np.array([r["pred"] for r in mine])
        per[q] = {"trials": len(mine), "accuracy": float((yy == pp).mean()), "high_share_true": float(yy.mean()), "high_share_pred": float(pp.mean()),
                  "balanced_accuracy": float(0.5 * ((pp[yy == 1] == 1).mean() + (pp[yy == 0] == 0).mean())) if len(set(yy.tolist())) == 2 else None}
    bas = [x["balanced_accuracy"] for x in per.values() if x["balanced_accuracy"] is not None]
    shares = np.array([x["high_share_pred"] for x in per.values()])
    gap = np.array([abs(x["high_share_pred"] - x["high_share_true"]) for x in per.values()])
    out |= {"participants": len(people), "per_participant_ba_mean": float(np.mean(bas)), "per_participant_ba_n": len(bas),
            "per_participant_accuracy_mean": float(np.mean([x["accuracy"] for x in per.values()])),
            "positive_rate_sd_across_people": float(shares.std()), "positive_rate_abs_gap_mean": float(gap.mean()),
            "people_all_one_label": int(((shares == 0) | (shares == 1)).sum()), "per_participant": per}
    folds = {}
    for k in FOLDS:
        mine = [r for r in recs if r["fold"] == k]
        folds[str(k)] = M.binary_metrics(np.array([r["y"] for r in mine]), np.array([r["pred"] for r in mine]))
    out["folds"] = folds
    out["mean_fold_ba_5"] = float(np.mean([folds[str(k)]["balanced_accuracy"] for k in FOLDS]))
    out["mean_fold_ba_3"] = float(np.mean([folds[str(k)]["balanced_accuracy"] for k in FOLDS if k < 3]))
    return out


def plots(results: dict[str, Any]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colour = {"classical": "#2a78d6", "neural": "#eb6834"}
    ink, muted, surface = "#0b0b0b", "#52514e", "#fcfcfb"
    rows = [(fam, name, r) for fam, by in results.items() for name, r in by.items() if name != "A_baseline"]
    fig, ax = plt.subplots(figsize=(8.5, 0.32 * len(rows) + 1.4), facecolor=surface)
    ax.set_facecolor(surface)
    for i, (fam, name, r) in enumerate(rows):
        lo, hi = r["delta_ba_pp_ci95"]
        ax.plot([lo, hi], [i, i], color=colour[fam], lw=2, solid_capstyle="round")
        ax.plot(r["delta_ba_pp"], i, "o", color=colour[fam], ms=6, mec=surface, mew=2)
    ax.axvline(0, color=muted, lw=1)
    ax.set_yticks(range(len(rows)), [f"{fam[:9]} · {name}" for fam, name, _ in rows], color=ink, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("Change in pooled trial balanced accuracy vs its own baseline (percentage points, 95% CI)", color=ink, fontsize=9)
    ax.set_title("Development cross-validation only: no independent test", color=ink, fontsize=10, loc="left")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis="x", color="#e4e3df", lw=0.6)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(OUT / "plot_delta_vs_baseline.png", dpi=150)
    plt.close(fig)

    names = ["A_baseline", "B_batch", "B_online", "C_base_rate_batch", "C_base_rate_online", "D_batch", "D_online"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True, facecolor=surface)
    rng = np.random.default_rng(0)
    for ax, fam in zip(axes, ("classical", "neural"), strict=True):
        ax.set_facecolor(surface)
        for i, name in enumerate(names):
            per = results[fam][name]["per_participant"]
            shares = np.array([x["high_share_pred"] for x in per.values()])
            ax.scatter(i + rng.uniform(-0.18, 0.18, len(shares)), shares, s=10, color=colour[fam], alpha=0.45, lw=0)
            ax.plot([i - 0.3, i + 0.3], [shares.mean()] * 2, color=ink, lw=2)
        true = np.mean([x["high_share_true"] for x in results[fam]["A_baseline"]["per_participant"].values()])
        ax.axhline(true, color=muted, lw=1, ls="--")
        ax.text(len(names) - 0.5, true + 0.02, "true high share", color=muted, fontsize=8, ha="right")
        ax.set_xticks(range(len(names)), names, rotation=35, ha="right", fontsize=8, color=ink)
        ax.set_title(f"{fam}: share of each person's trials called high", color=ink, fontsize=9, loc="left")
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    axes[0].set_ylabel("Predicted high share per person (dots), mean (bar)", color=ink, fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "plot_positive_rates.png", dpi=150)
    plt.close(fig)


def phase_report(_: argparse.Namespace) -> None:
    d = load_dev()
    trials, info = collect(d)
    results: dict[str, dict[str, Any]] = {"classical": {}, "neural": {}}
    for (fam, name), recs in trials.items():
        results[fam][name] = summarise(recs)
    for fam, by in results.items():
        base = trials[(fam, "A_baseline")]
        keys = [(r["participant"], r["clip"]) for r in base]
        y = np.array([r["y"] for r in base])
        person = np.array([r["participant"] for r in base])
        preds = {}
        for name in by:
            index = {(r["participant"], r["clip"]): r["pred"] for r in trials[(fam, name)]}
            assert set(index) == set(keys), name
            preds[name] = np.array([index[kk] for kk in keys])
        boots = M.bootstrap_ba(person, y, preds)
        for name, r in by.items():
            r["ci95"] = np.quantile(boots[name], [0.025, 0.975]).tolist()
            diff = (boots[name] - boots["A_baseline"]) * 100
            r["delta_ba_pp"] = (r["balanced_accuracy"] - by["A_baseline"]["balanced_accuracy"]) * 100
            r["delta_ba_pp_ci95"] = np.quantile(diff, [0.025, 0.975]).tolist()
    selection = {fam: {"by_5_folds": max(by, key=lambda n: by[n]["mean_fold_ba_5"]), "by_3_folds": max(by, key=lambda n: by[n]["mean_fold_ba_3"])}
                 for fam, by in results.items()}
    reproduction = {
        "classical_baseline_vs_outputs_faced_inner_classical_json": {
            str(k): [round(results["classical"]["A_baseline"]["folds"][str(k)]["balanced_accuracy"], 4),
                     round(json.loads((FACED_OUT / "inner_classical.json").read_text())["results"]["S1"]["lr_C0.01"][str(k)]["trial"]["balanced_accuracy"], 4)]
            for k in FOLDS},
        "neural_baseline_vs_neurosense_fusion_inner": {
            str(k): [round(results["neural"]["A_baseline"]["folds"][str(k)]["balanced_accuracy"], 4),
                     round(json.loads((ad.OUT / "inner" / f"erm_fold{k}.json").read_text())["balanced_accuracy"], 4)]
            for k in FOLDS if k < 3}}
    config = {"created_at": datetime.now(UTC).isoformat(), "git_commit": git_commit(), "seed": SEED, "folds": FOLDS, "classical_model": LR,
              "neural_train": frun.TRAIN, "neural_seeds": NEURAL_SEEDS, "early_stopping": EARLY, "min_history_clips": M.MIN_HISTORY,
              "covariance": {"bands": M.BANDS, "shrinkage": M.SHRINK, "reference": "log-Euclidean mean (classical); arithmetic mean (neural)"},
              "threshold": "0.5 on mean segment probability, except C methods (per-person threshold)", "methods": METHODS}
    write("config.json", config)
    write("results.json", {"note": "DEVELOPMENT CROSS-VALIDATION ONLY. No independent test was run.", "config": config, "selection": selection,
                           "reproduction": reproduction, "info": info, "results": results})
    with (OUT / "comparison.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["family", "method", "unlabelled_eeg_of_new_person", "trials", "balanced_accuracy", "ci_low", "ci_high", "delta_pp",
                    "delta_ci_low", "delta_ci_high", "accuracy", "f1_macro", "roc_auc", "sensitivity", "specificity",
                    "per_participant_ba_mean", "positive_rate_sd", "positive_rate_abs_gap", "people_all_one_label", "mean_fold_ba_3", "mean_fold_ba_5"])
        uses = {(m["family"], m["method"]): m["unlabelled_eeg_of_new_person"] for m in METHODS}
        for fam, by in results.items():
            for name, r in by.items():
                w.writerow([fam, name, uses[(fam, name)], r["n"], *(round(x, 4) for x in (r["balanced_accuracy"], *r["ci95"])),
                            *(round(x, 2) for x in (r["delta_ba_pp"], *r["delta_ba_pp_ci95"])),
                            *(round(r[x], 4) for x in ("accuracy", "f1_macro", "roc_auc", "sensitivity", "specificity", "per_participant_ba_mean",
                                                       "positive_rate_sd_across_people", "positive_rate_abs_gap_mean")),
                            r["people_all_one_label"], round(r["mean_fold_ba_3"], 4), round(r["mean_fold_ba_5"], 4)])
    with (OUT / "per_fold.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["family", "method", "fold", "trials", "balanced_accuracy", "accuracy", "f1_macro", "sensitivity", "specificity", "high_share_true", "high_share_pred"])
        for fam, by in results.items():
            for name, r in by.items():
                for k, x in r["folds"].items():
                    w.writerow([fam, name, k, x["n"], *(round(x[c], 4) for c in ("balanced_accuracy", "accuracy", "f1_macro", "sensitivity", "specificity",
                                                                                "high_share_true", "high_share_pred"))])
    with (OUT / "per_participant.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["family", "method", "participant", "trials", "balanced_accuracy", "accuracy", "high_share_true", "high_share_pred"])
        for fam, by in results.items():
            for name, r in by.items():
                for q, x in r["per_participant"].items():
                    w.writerow([fam, name, q, x["trials"], "" if x["balanced_accuracy"] is None else round(x["balanced_accuracy"], 4),
                                round(x["accuracy"], 4), round(x["high_share_true"], 4), round(x["high_share_pred"], 4)])
    plots(results)
    for fam, by in results.items():
        for name, r in by.items():
            print(f"{fam:9s} {name:22s} BA {r['balanced_accuracy']:.3f} [{r['ci95'][0]:.3f},{r['ci95'][1]:.3f}] "
                  f"d {r['delta_ba_pp']:+.1f} [{r['delta_ba_pp_ci95'][0]:+.1f},{r['delta_ba_pp_ci95'][1]:+.1f}] "
                  f"3f {r['mean_fold_ba_3']:.3f} 5f {r['mean_fold_ba_5']:.3f} posSD {r['positive_rate_sd_across_people']:.3f}")
    print(json.dumps({"selection": selection, "reproduction": reproduction, "fallbacks": info["fallback_trials"], "early": info["early_stopping"]}, indent=1))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", required=True, choices=["manifest", "classical", "neural", "report"])
    args = parser.parse_args()
    {"manifest": phase_manifest, "classical": phase_classical, "neural": phase_neural, "report": phase_report}[args.phase](args)


if __name__ == "__main__":
    main()
