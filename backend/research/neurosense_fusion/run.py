"""NeuroSense Fusion experiment runner. Each phase is resumable and writes to outputs/experiments/neurosense_fusion/.

From backend/:
  python -m research.neurosense_fusion.run --phase check      # adapters, real-data leakage checks, timing; no scores
  python -m research.neurosense_fusion.run --phase inner      # candidates on inner folds (development data only)
  python -m research.neurosense_fusion.run --phase freeze     # record the selection before any test score exists
  python -m research.neurosense_fusion.run --phase final      # zero-shot test on FACED and DEAP, once
  python -m research.neurosense_fusion.run --phase personal   # personalised calibration, once
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import UTC, datetime
from typing import Any

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from app.ml.provenance import git_commit
from research.neurosense_fusion import adapters as ad
from research.neurosense_fusion import model as m
from research.neurosense_fusion import protocol as pr
from scripts import faced_calibration_controls as controls
from scripts import faced_experiment as fe

OUT = ad.OUT
SEED = 42
INNER_FOLDS = (0, 1, 2)
TRAIN = {"epochs": 10, "batch": 128, "lr": 1e-3, "weight_decay": 1e-2, "ssl_epochs": 5, "n_domains": 16,
         "align_weight": 1.0, "dann_weight": 0.1}
CANDIDATES: dict[str, dict[str, Any]] = {
    "erm": {}, "coral": {"align": "coral"}, "mmd": {"align": "mmd"}, "dann": {"dann": True}, "erm_ssl": {"ssl": True},
}
ADAPTER = {"steps": 100, "lr": 0.05, "prior_weight": 0.1}


def write(name: str, obj: Any) -> None:
    path = OUT / name
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj, indent=1, default=str), encoding="utf-8")
    tmp.replace(path)


def read(name: str) -> Any:
    return json.loads((OUT / name).read_text(encoding="utf-8"))


def tensor(segments: np.ndarray, rows: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(ad.normalise_segments(np.asarray(segments[np.flatnonzero(rows)], dtype=np.float32)))


def pretrain(net: m.FusionNet, x: torch.Tensor, p: np.ndarray, key: np.ndarray, seed: int, epochs: int) -> None:
    """Self-supervised: the same clip and segment seen by two different training participants should match. No labels."""
    rng = np.random.default_rng(seed)
    groups: dict[int, np.ndarray] = {}
    for k in np.unique(key):
        rows = np.flatnonzero(key == k)
        if len(set(p[rows].tolist())) > 1:
            groups[int(k)] = rows
    keys = np.array(sorted(groups))
    opt = torch.optim.AdamW([*net.encoder.parameters(), *net.projection.parameters()], lr=TRAIN["lr"], weight_decay=TRAIN["weight_decay"])
    net.train()
    for _ in range(epochs):
        for _ in range(len(x) // TRAIN["batch"]):
            chosen = rng.choice(keys, min(TRAIN["batch"], len(keys)), replace=False)
            pairs = []
            for k in chosen:
                rows = groups[int(k)]
                a = rng.choice(rows)
                pairs.append((a, rng.choice(rows[p[rows] != p[a]])))
            a, b = zip(*pairs, strict=True)
            loss = m.contrastive_loss(net.projection(net.encoder(x[list(a)])), net.projection(net.encoder(x[list(b)])))
            opt.zero_grad()
            loss.backward()
            opt.step()


def fit(cfg: dict[str, Any], x: torch.Tensor, y: np.ndarray, p: np.ndarray, v: np.ndarray, seg: np.ndarray, seed: int,
        epochs: int | None = None) -> m.FusionNet:
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    net = m.FusionNet(n_domains=TRAIN["n_domains"])
    epochs = TRAIN["epochs"] if epochs is None else epochs
    if cfg.get("ssl"):
        pretrain(net, x, p, v.astype(np.int64) * 10 + seg, seed, TRAIN["ssl_epochs"] if epochs else 0)
    people = sorted(set(p.tolist()))
    index = {q: i for i, q in enumerate(people)}
    person = np.array([index[q] for q in p])
    domain = torch.from_numpy(person % TRAIN["n_domains"])
    target = torch.from_numpy(y.astype(np.float32))
    weight = torch.tensor(float((y == 0).sum() / max((y == 1).sum(), 1)))
    params = [*net.encoder.parameters(), *net.head.parameters(), *net.domain_head.parameters()]
    opt = torch.optim.AdamW(params, lr=TRAIN["lr"], weight_decay=TRAIN["weight_decay"])
    net.train()
    for epoch in range(epochs):
        order = rng.permutation(len(x))
        for i in range(0, len(order) - TRAIN["batch"] + 1, TRAIN["batch"]):
            batch = order[i:i + TRAIN["batch"]]
            logit, z = net(x[batch])
            loss = torch.nn.functional.binary_cross_entropy_with_logits(logit, target[batch], pos_weight=weight)
            if cfg.get("align"):
                present = rng.permutation(np.unique(person[batch]))
                left = np.isin(person[batch], present[: len(present) // 2])
                if left.sum() >= 8 and (~left).sum() >= 8:
                    align = m.coral_loss if cfg["align"] == "coral" else m.mmd_loss
                    loss = loss + TRAIN["align_weight"] * align(z[torch.from_numpy(left)], z[torch.from_numpy(~left)])
            if cfg.get("dann"):
                strength = TRAIN["dann_weight"] * (epoch + 1) / epochs
                loss = loss + torch.nn.functional.cross_entropy(net.domain_logits(z, strength), domain[batch])
            opt.zero_grad()
            loss.backward()
            opt.step()
    return net.eval()


@torch.no_grad()
def scores(net: m.FusionNet, x: torch.Tensor) -> np.ndarray:
    return np.concatenate([torch.sigmoid(net(x[i:i + 512])[0]).numpy() for i in range(0, len(x), 512)])


@torch.no_grad()
def latents(net: m.FusionNet, x: torch.Tensor) -> torch.Tensor:
    return torch.cat([net.encoder(x[i:i + 512]) for i in range(0, len(x), 512)])


def full_metrics(t: dict[str, np.ndarray], seg_y: np.ndarray, seg_s: np.ndarray) -> dict[str, Any]:
    ev = fe.evaluate(t, seg_y, seg_s)
    (tn, fp), (fn, tp) = ev["trial_confusion"]["matrix"]
    ev["sensitivity"], ev["specificity"] = tp / max(tp + fn, 1), tn / max(tn + fp, 1)
    ev["class_distribution"] = {"high": int(t["y"].sum()), "low": int((1 - t["y"]).sum())}
    return ev


# ---------------------------------------------------------------- phases

def phase_check(_: argparse.Namespace) -> None:
    faced, deap = ad.load_faced(), ad.load_deap()
    split = json.loads((ad.ROOT / "outputs" / "faced" / "protocol_split_v1.json").read_text(encoding="utf-8"))
    checks: dict[str, Any] = {}
    train, test = pr.final_rows(faced.participant, faced.trial, split)
    pr.check_disjoint(faced.participant, faced.trial, train, test, split)
    for k in INNER_FOLDS:
        a, b = pr.inner_rows(faced.participant, faced.trial, split, k)
        pr.check_disjoint(faced.participant, faced.trial, a, b, split)
    checks["zero_shot_splits_disjoint"] = True
    pr.check_no_frozen(deap.participant)
    checks["deap_participants"] = sorted(set(deap.participant.tolist()))
    checks["deap_excluded_from_training"] = not (set(deap.participant.tolist()) & set(faced.participant.tolist()))
    assert checks["deap_excluded_from_training"]

    ids = list(split["test_participants"])
    sessions = [controls.session(q) for q in ids]
    usable = [s for s in sessions if sorted(len(b) for b in s["blocks"]) == [4] * 7]
    for s in usable:
        for n in pr.CALIBRATION_SIZES:
            c, t = pr.personal_plan(s["order"], s["blocks"], n)
            pr.check_personal_plan(s["order"], s["blocks"], c, t)
    checks["personal_participants_with_clean_blocks"] = {"usable": len(usable), "of": len(sessions)}

    rows = {"segments": {"FACED": faced.segments.shape, "DEAP": deap.segments.shape}, "faced_train_segments": int(train.sum()), "faced_test_segments": int(test.sum())}
    for name, b in (("FACED", faced), ("DEAP", deap)):
        keys = sorted(set(zip(b.participant.tolist(), b.trial.tolist(), strict=True)))
        high = {k: None for k in keys}
        for q, t, label in zip(b.participant.tolist(), b.trial.tolist(), b.y.tolist(), strict=True):
            high[(q, t)] = label
        rows[f"{name}_trials"] = len(keys)
        rows[f"{name}_high_share"] = float(np.mean(list(high.values())))
    x = tensor(faced.segments, train)[:256]
    print("mean abs after normalisation", float(x.abs().mean()), "| DEAP", float(tensor(deap.segments, np.arange(len(deap.y)) < 256).abs().mean()))
    torch.manual_seed(0)
    net = m.FusionNet()
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3)
    start = time.time()
    for _ in range(5):
        logit, _z = net(x[:128])
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logit, torch.zeros(128))
        opt.zero_grad()
        loss.backward()
        opt.step()
    per_batch = (time.time() - start) / 5
    batches = int(train.sum()) // TRAIN["batch"]
    rows["seconds_per_batch"] = per_batch
    rows["estimated_seconds_per_epoch_full_training"] = per_batch * batches
    write("stage1_check.json", {"created_at": datetime.now(UTC).isoformat(), "checks": checks, "data": rows, "compatibility": ad.COMPATIBILITY,
                                "parameters": m.parameter_counts(net), "train_config": TRAIN, "candidates": CANDIDATES, "adapter": ADAPTER, "inner_folds": INNER_FOLDS})
    print(json.dumps({"checks": checks, "data": rows, "parameters": m.parameter_counts(net)}, indent=1, default=str))


def phase_inner(_: argparse.Namespace) -> None:
    faced = ad.load_faced()
    split = json.loads((ad.ROOT / "outputs" / "faced" / "protocol_split_v1.json").read_text(encoding="utf-8"))
    for name, cfg in CANDIDATES.items():
        for k in INNER_FOLDS:
            path = OUT / "inner" / f"{name}_fold{k}.json"
            if path.exists():
                continue
            start = time.time()
            train, val = pr.inner_rows(faced.participant, faced.trial, split, k)
            pr.check_disjoint(faced.participant, faced.trial, train, val, split)
            net = fit(cfg, tensor(faced.segments, train), faced.y[train], faced.participant[train], faced.trial[train], faced.segment[train], SEED)
            s = scores(net, tensor(faced.segments, val))
            t = fe.trial_table(faced.participant[val], faced.trial[val], faced.y[val], s)
            r = fe.metrics(t["y"], t["score"], t["pred"]) | {"seconds": time.time() - start}
            write(f"inner/{name}_fold{k}.json", r)
            print(name, k, round(r["balanced_accuracy"], 4), round(r["roc_auc"], 4), int(r["seconds"]), "s", flush=True)


def phase_freeze(_: argparse.Namespace) -> None:
    if (OUT / "zero_shot_results.json").exists():
        raise SystemExit("the final test already ran")
    table = {name: [read(f"inner/{name}_fold{k}.json")["balanced_accuracy"] for k in INNER_FOLDS] for name in CANDIDATES}
    means = {name: float(np.mean(v)) for name, v in table.items()}
    order = list(CANDIDATES)
    best = max(order, key=lambda n: (round(means[n], 12), -order.index(n)))
    write("freeze.json", {"frozen_at": datetime.now(UTC).isoformat(), "selected": best, "config": CANDIDATES[best], "train": TRAIN, "adapter": ADAPTER, "seed": SEED,
                          "inner_balanced_accuracy": table, "inner_means": means, "selection": "mean trial balanced accuracy over inner folds (FACED development data only)",
                          "threshold": "mean segment probability > 0.5", "git_commit": git_commit()})
    print(best, means)


def trial_output(b: ad.Bundle, rows: np.ndarray, s: np.ndarray) -> dict[str, Any]:
    t = fe.trial_table(b.participant[rows], b.trial[rows], b.y[rows], s)
    ev = full_metrics(t, b.y[rows], s)
    ev["majority_baseline_accuracy"] = float(max(t["y"].mean(), 1 - t["y"].mean()))
    return ev


def final_model(faced: ad.Bundle, split: dict[str, Any], cfg: dict[str, Any]) -> m.FusionNet:
    path = OUT / "model_zero_shot.pt"
    train, _ = pr.final_rows(faced.participant, faced.trial, split)
    if path.exists():
        net = m.FusionNet(n_domains=TRAIN["n_domains"])
        net.load_state_dict(torch.load(path))
        return net.eval()
    net = fit(cfg, tensor(faced.segments, train), faced.y[train], faced.participant[train], faced.trial[train], faced.segment[train], SEED)
    torch.save(net.state_dict(), path)
    return net


def phase_final(_: argparse.Namespace) -> None:
    if (OUT / "zero_shot_results.json").exists():
        raise SystemExit("the final test already ran: it runs once")
    frozen = read("freeze.json")
    faced, deap = ad.load_faced(), ad.load_deap()
    split = json.loads((ad.ROOT / "outputs" / "faced" / "protocol_split_v1.json").read_text(encoding="utf-8"))
    train, test = pr.final_rows(faced.participant, faced.trial, split)
    pr.check_disjoint(faced.participant, faced.trial, train, test, split)
    pr.check_no_frozen(deap.participant)
    start = time.time()
    net = final_model(faced, split, frozen["config"])
    seconds = time.time() - start
    everything = np.ones(len(deap.y), bool)
    results = {"created_at": datetime.now(UTC).isoformat(), "git_commit": git_commit(), "selected": frozen["selected"], "training_seconds": seconds,
               "train": {"participants": len(split["dev_participants"]), "clips": len(split["dev_videos"]), "segments": int(train.sum())},
               "FACED_reused_test": {"note": "previously examined participants and clips, not untouched", **trial_output(faced, test, scores(net, tensor(faced.segments, test)))},
               "DEAP_external": {"note": "external to this model; examined in earlier DEAP work", **trial_output(deap, everything, scores(net, tensor(deap.segments, everything)))}}
    write("zero_shot_results.json", results)
    for key in ("FACED_reused_test", "DEAP_external"):
        print(key, results[key]["trial_pooled"], results[key]["trial_pooled_ci95_participant_bootstrap"], results[key]["sensitivity"], results[key]["specificity"])


def adapt(net: m.FusionNet, z: torch.Tensor, y: torch.Tensor) -> torch.nn.Linear:
    """Personal head: starts at the population head and stays near it. Hyperparameters are fixed in ADAPTER."""
    head = torch.nn.Linear(m.LATENT, 1)
    head.load_state_dict(net.head.state_dict())
    prior = [p.detach().clone() for p in head.parameters()]
    opt = torch.optim.Adam(head.parameters(), lr=ADAPTER["lr"])
    weight = (y == 0).sum() / (y == 1).sum().clamp_min(1)
    for _ in range(ADAPTER["steps"]):
        loss = torch.nn.functional.binary_cross_entropy_with_logits(head(z).squeeze(1), y, pos_weight=weight)
        loss = loss + ADAPTER["prior_weight"] * sum((a - b).pow(2).sum() for a, b in zip(head.parameters(), prior, strict=True))
        opt.zero_grad()
        loss.backward()
        opt.step()
    return head


def summarise(rows: list[tuple[str, int, float, int]]) -> dict[str, Any]:
    """rows: (participant, true label, score, predicted label)."""
    q = np.array([r[0] for r in rows])
    y = np.array([r[1] for r in rows])
    s = np.array([r[2] for r in rows], dtype=float)
    pred = np.array([r[3] for r in rows])
    out = fe.metrics(y, s, pred)
    tp, fn = ((y == 1) & (pred == 1)).sum(), ((y == 1) & (pred == 0)).sum()
    tn, fp = ((y == 0) & (pred == 0)).sum(), ((y == 0) & (pred == 1)).sum()
    people = sorted(set(q.tolist()))
    by = {p: np.flatnonzero(q == p) for p in people}
    rng = np.random.default_rng(0)
    boots = []
    for _ in range(2000):
        pick = np.concatenate([by[p] for p in rng.choice(people, len(people))])
        if len(set(y[pick].tolist())) == 2:
            boots.append(0.5 * ((pred[pick][y[pick] == 1] == 1).mean() + (pred[pick][y[pick] == 0] == 0).mean()))
    per = [0.5 * ((pred[i][y[i] == 1] == 1).mean() + (pred[i][y[i] == 0] == 0).mean()) for i in by.values() if len(set(y[i].tolist())) == 2]
    return out | {"ci95": np.quantile(boots, [0.025, 0.975]).tolist(), "sensitivity": float(tp / max(tp + fn, 1)), "specificity": float(tn / max(tn + fp, 1)),
                  "confusion": [[int(tn), int(fp)], [int(fn), int(tp)]], "participants": len(people), "mean_per_participant_balanced_accuracy": float(np.mean(per)) if per else None}


def phase_personal(_: argparse.Namespace) -> None:
    if (OUT / "personalised_results.json").exists():
        raise SystemExit("the personalised evaluation already ran: it runs once")
    frozen = read("freeze.json")
    faced = ad.load_faced()
    split = json.loads((ad.ROOT / "outputs" / "faced" / "protocol_split_v1.json").read_text(encoding="utf-8"))
    net = final_model(faced, split, frozen["config"])
    sessions = [controls.session(q) for q in split["test_participants"]]
    usable = [s for s in sessions if sorted(len(b) for b in s["blocks"]) == [4] * 7]
    rows: dict[str, list[tuple[str, int, float, int]]] = {}
    fallbacks = dict.fromkeys(pr.CALIBRATION_SIZES, 0)
    for s in usable:
        pid = s["participant"]
        mine = np.flatnonzero(faced.participant == pid)
        vid = faced.trial[mine]
        x = tensor(faced.segments[mine], np.ones(len(mine), bool))
        y = faced.y[mine]
        label = {int(c): int(y[vid == c][0]) for c in set(vid.tolist())}
        z = latents(net, x)
        base = torch.sigmoid(net.head(z).squeeze(1)).detach().numpy()
        for n in pr.CALIBRATION_SIZES:
            calibration, test = pr.personal_plan(s["order"], s["blocks"], n)
            pr.check_personal_plan(s["order"], s["blocks"], calibration, test)
            cal = np.isin(vid, calibration)
            one_class = len({label[c] for c in calibration if c in label}) < 2
            fallbacks[n] += int(one_class)
            head = net.head if one_class else adapt(net, z[torch.from_numpy(cal)], torch.from_numpy(y[cal].astype(np.float32)))
            personal = torch.sigmoid(head(z).squeeze(1)).detach().numpy()
            for c in test:
                if c not in label:
                    continue
                here = vid == c
                nearest = pr.nearest_past_rating(s["order"], calibration, c)
                seen_clip = "strict" if c in split["test_videos"] else "clip_in_training"
                for name, score in (("no_calibration", float(base[here].mean())), ("personalised", float(personal[here].mean())),
                                    ("nearest_in_time", float(label.get(nearest, 0)))):
                    rows.setdefault(f"{name}_{n}", []).append((pid, label[c], score, int(score > 0.5)))
                    if seen_clip == "strict":
                        rows.setdefault(f"{name}_{n}_unseen_clips_only", []).append((pid, label[c], score, int(score > 0.5)))
    results = {"created_at": datetime.now(UTC).isoformat(), "note": "previously examined FACED participants, not untouched; blocks inferred from gaps in the session",
               "participants_used": len(usable), "of": len(sessions), "one_class_calibration_fallbacks": fallbacks,
               "results": {k: summarise(v) for k, v in rows.items()}}
    write("personalised_results.json", results)
    for k, v in results["results"].items():
        print(k, round(v["balanced_accuracy"], 4), [round(x, 3) for x in v["ci95"]], "n", v["n"], "perP", v["mean_per_participant_balanced_accuracy"])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["check", "inner", "freeze", "final", "personal"], required=True)
    args = ap.parse_args()
    torch.set_num_threads(8)
    OUT.mkdir(parents=True, exist_ok=True)
    {"check": phase_check, "inner": phase_inner, "freeze": phase_freeze, "final": phase_final, "personal": phase_personal}[args.phase](args)


if __name__ == "__main__":
    main()
