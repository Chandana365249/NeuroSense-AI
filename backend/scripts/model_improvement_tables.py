"""Markdown tables for docs/MODEL_IMPROVEMENT_EXPERIMENTS.md, generated from the saved summary.json (no re-computation).

Usage (from backend/): python -m scripts.model_improvement_tables > ../docs/results/exploratory/model_improvement/tables.md
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

SUMMARY = Path(__file__).resolve().parents[2] / "docs" / "results" / "exploratory" / "model_improvement" / "summary.json"


def f(x: float | None, d: int = 3) -> str:
    return "–" if x is None else f"{x:.{d}f}"


def ci(m: dict[str, Any]) -> str:
    return f"{f(m['mean'])} [{f(m['ci_low'])}, {f(m['ci_high'])}]"


def row(name: str, s: dict[str, Any], comp: dict[str, Any] | None = None) -> str:
    m = s["metrics"]
    cells = [name, ci(m["balanced_accuracy"]), f(m["f1_macro"]["mean"]), f(m["accuracy"]["mean"]), f(m["roc_auc"]["mean"]),
             f(m["balanced_accuracy"]["min"]) + "–" + f(m["balanced_accuracy"]["max"]),
             f(s["single_label_participants_mean_over_seeds"], 1), f(s["across_seed_sd_of_mean_ba"])]
    if comp is not None:
        cells.append(f"{comp['mean_difference']:+.3f} [{comp['bootstrap_ci'][0]:+.3f}, {comp['bootstrap_ci'][1]:+.3f}]; "
                     f"p={f(comp['wilcoxon_p'])}; {comp['n_better']}/{comp['n']} better")
    return "| " + " | ".join(cells) + " |"


HEAD = ("| Variant | Trial bal. acc. [95% CI] | Macro-F1 | Accuracy | Within-person AUC | Range over people | "
        "Single-label people | Seed SD |")
SEP = "|---|---|---|---|---|---|---|---|"


def table(rows: list[tuple[str, dict[str, Any], dict[str, Any] | None]], comp_title: str | None = None) -> str:
    head, sep = HEAD, SEP
    if comp_title:
        head += f" {comp_title} |"
        sep += "---|"
    return "\n".join([head, sep] + [row(n, s, c) for n, s, c in rows])


def chosen(c: dict[str, int]) -> str:
    total = sum(c.values())
    return ", ".join(f"{k} {v}/{total}" for k, v in sorted(c.items(), key=lambda kv: -kv[1]))


def main() -> None:
    S = json.loads(SUMMARY.read_text(encoding="utf-8"))
    p1, p2, p3 = S["p1"], S["p2"], S["p3"]
    out: list[str] = []
    b = S["baseline"]
    out += ["## Baseline reproduction (historical fixed split)", "",
            "| Level | Accuracy | Balanced acc. | Macro-F1 | ROC AUC | n |", "|---|---|---|---|---|---|"]
    for lvl in ("segment_pooled", "trial_pooled"):
        m = b[lvl]
        out.append(f"| {lvl} | {f(m['accuracy'], 4)} | {f(m['balanced_accuracy'], 4)} | {f(m['f1_macro'], 4)} | {f(m['roc_auc'], 4)} | {m['n_trials']} |")
    out.append("")
    out += ["| Participant | Trial bal. acc. | Trial acc. | Within-person AUC | Predicted-high share | True-high share |", "|---|---|---|---|---|---|"]
    for q, m in b["trial_per_participant"].items():
        out.append(f"| {q} | {f(m['balanced_accuracy'])} | {f(m['accuracy'])} | {f(m['roc_auc'])} | {f(m['pred_high_share'])} | {f(m['true_high_share'])} |")
    mc = b["trial_balanced_accuracy_participant_mean"]
    out += ["", f"Participant mean trial balanced accuracy {f(mc['mean'])}, t 95% CI [{f(mc['ci_low'])}, {f(mc['ci_high'])}] (n = 3). "
            f"Saved model re-scored on cached features: max |Δp| = {b['saved_vs_recomputed_max_abs_p_high_difference']:.1e}.", ""]

    for norm, title in (("N0", "P1 participant-independent (inductive, N0)"), ("N3", "P1 unsupervised calibration (N3, per-person z-score)")):
        blk = p1[norm]
        rows = [("Majority-class reference", blk["majority_reference"], None),
                ("Random Forest reference (registered, mean agg.)", blk["rf_reference_mean_agg"], None)]
        rows += [(f"{fam.upper()} (nested: config + aggregation)", blk["families"][fam]["summary"], blk["families"][fam]["vs_rf_reference"])
                 for fam in ("rf", "xgb", "lr", "svm")]
        rows.append(("**Nested selection over all candidates**", blk["nested_all"]["summary"], blk["nested_all"]["vs_rf_reference"]))
        out += [f"## {title}", "", table(rows, "Δ bal. acc. vs RF [bootstrap CI]; Wilcoxon"), ""]
        out.append("Selections made by the inner loop (outer fold × seed): " + chosen(blk["nested_all"]["chosen"]))
        out.append("")
        out += ["Aggregation (registered RF):", "", table([(a, s, None) for a, s in blk["aggregation_rf"].items()]), ""]
    out += [f"RF 5-seed reference (N0): {ci(p1['reference_rf_5seeds']['metrics']['balanced_accuracy'])}; "
            f"offset tracking ρ = {f(p1['reference_rf_5seeds']['offset_tracking_spearman'])}. N3 − N0 (RF): "
            f"{p1['N3_vs_N0_rf']['mean_difference']:+.3f} {p1['N3_vs_N0_rf']['bootstrap_ci']}, p = {f(p1['N3_vs_N0_rf']['wilcoxon_p'])}.", ""]
    out += ["## P1 feature ablation (RF, N0, mean aggregation, seeds 42–44)", "",
            table([("full (440)", p1["N0"]["rf_reference_mean_agg"], None)] + [(k, v["summary"], v["vs_full"]) for k, v in p1["ablation"].items()],
                  "Δ vs full"), ""]
    out += ["## P1 regression, S1 and learning curve", "",
            table([(f"RF regression → threshold 5.0 ({n})", v["summary"], v["vs_classifier"]) for n, v in p1["regression"].items()], "Δ vs classifier"), "",
            table([(f"S1 borderline excluded ({n})", v, None) for n, v in p1["s1"].items()]), "",
            table([(f"{n} training participants", v, None) for n, v in p1["learning_curve"].items()]), ""]

    rows = [("Random Forest reference (registered, mean agg.)", p2["rf_reference_mean_agg"], None)]
    rows += [(f"{fam.upper()} (nested)", p2["families"][fam]["summary"], p2["families"][fam]["vs_rf_reference"]) for fam in ("rf", "xgb", "lr", "svm")]
    rows.append(("**Nested selection over all candidates**", p2["nested_all"]["summary"], p2["nested_all"]["vs_rf_reference"]))
    out += ["## P2 within-participant (supervised personalization, 32 calibration videos)", "",
            f"RF 5-seed reference: {ci(p2['reference_rf_5seeds']['metrics']['balanced_accuracy'])}.", "",
            table(rows, "Δ vs RF [bootstrap CI]; Wilcoxon"), "",
            "Selections: " + chosen(p2["nested_all"]["chosen"]), "",
            "Fixed configurations (mean aggregation):", "",
            table([(k, v, None) for k, v in p2["fixed_configs_mean_agg"].items()]), "",
            "Aggregation (RF):", "", table([(a, s, None) for a, s in p2["aggregation_rf"].items()]), ""]
    out += ["## P2 feature ablation (RF)", "",
            table([("full (440)", p2["rf_reference_mean_agg"], None)] + [(k, v["summary"], v["vs_full"]) for k, v in p2["ablation"].items()], "Δ vs full"), ""]
    out += ["## P2 regression and S1", "",
            table([("RF regression → threshold 5.0", p2["regression"]["summary"], p2["regression"]["vs_classifier"])], "Δ vs classifier"), "",
            table([("S1 borderline excluded", p2["s1"], None)]), ""]
    ln = p2["label_noise"]
    out += [f"Label noise: pooled accuracy on borderline trials {f(ln['pooled_borderline_accuracy'])} vs clear trials {f(ln['pooled_clear_accuracy'])}; "
            f"point-biserial r(|valence − 5|, correct) = {f(ln['pointbiserial_distance_from_threshold_vs_correct']['r'])}, "
            f"p = {f(ln['pointbiserial_distance_from_threshold_vs_correct']['p'], 4)}.", ""]
    out += ["## P3 calibration (population + personal), RF, seeds 42–44", "",
            table([(f"pooled, {k.replace('_', ' videos, weighting ')}", v, None) for k, v in p3.items() if k != "comparisons"]
                  + [(f"personal only, {n} videos", p2["calibration_size"][n], None) for n in ("8", "16", "24", "32")]), "",
            "| Calibration videos | Pooled − zero-shot | Personal-only − zero-shot | Pooled − personal-only |", "|---|---|---|---|"]
    for n in ("8", "16", "32"):
        c = p3["comparisons"][n]
        cells = [f"{c[k]['mean_difference']:+.3f} [{c[k]['bootstrap_ci'][0]:+.3f}, {c[k]['bootstrap_ci'][1]:+.3f}], p={f(c[k]['wilcoxon_p'])}, {c[k]['n_better']}/12 better"
                 for k in ("pooled_vs_zero_shot", "personal_only_vs_zero_shot", "pooled_vs_personal_only")]
        out.append(f"| {n} | " + " | ".join(cells) + " |")
    out += ["", "## Primary comparisons (Holm over 3)", "", "| Comparison | Δ bal. acc. | Bootstrap CI | Wilcoxon p | Holm p |", "|---|---|---|---|---|"]
    for k, v in S["primary_comparisons"].items():
        out.append(f"| {k} | {v['mean_difference']:+.3f} | [{v['bootstrap_ci'][0]:+.3f}, {v['bootstrap_ci'][1]:+.3f}] | {f(v['wilcoxon_p'])} | {f(v['holm_p'])} |")
    lk = S["leakage_demo"]
    out += ["", f"Leakage demonstration (invalid by design, segment split within person): segment accuracy {ci(lk['segment_accuracy'])}, "
            f"trial balanced accuracy {ci(lk['trial_balanced_accuracy'])}.", ""]
    out += ["## Per-participant trial balanced accuracy (seed-averaged)", "",
            "| Participant | P1 RF (N0) | P1 nested (N0) | P1 RF (N3) | P2 RF | P2 nested | P3 pooled 32 |", "|---|---|---|---|---|---|---|"]
    srcs = [p1["N0"]["rf_reference_mean_agg"], p1["N0"]["nested_all"]["summary"], p1["N3"]["rf_reference_mean_agg"],
            p2["rf_reference_mean_agg"], p2["nested_all"]["summary"], p3["32_equal"]]
    for q in S["participants"]:
        out.append(f"| {q} | " + " | ".join(f(s["per_participant"][q]["balanced_accuracy"]) for s in srcs) + " |")
    print("\n".join(out))


if __name__ == "__main__":
    main()
