"""Robust CNN evaluation on real DEAP: repeated seeds on the fixed split + nested LOPO, two normalizations.

Every run is saved as runs/<run_id>.json inside the experiment directory; existing runs are skipped, so an
interrupted experiment resumes where it stopped. The report (summary.json, REPORT.md, figures) is rebuilt
from the run records at the end, or on its own with --report-only.

Usage (from backend/):
  python -m scripts.evaluate_cnn                                   # full experiment (several hours on CPU)
  python -m scripts.evaluate_cnn --experiment-dir <dir>            # resume
  python -m scripts.evaluate_cnn --experiment-dir <dir> --report-only
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

from app.core.config import get_settings
from app.core.logging import configure_logging
from app.ml.deep.robust_evaluation import NORMALIZATIONS, PROTOCOLS, RobustCNNEvaluator
from app.ml.deep.training import CNNTrainingConfig


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--experiment-dir", type=Path)
    p.add_argument("--protocols", nargs="+", default=list(PROTOCOLS), choices=PROTOCOLS)
    p.add_argument("--normalizations", nargs="+", default=list(NORMALIZATIONS), choices=NORMALIZATIONS)
    p.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44, 45, 46])
    p.add_argument("--max-epochs", type=int, default=60)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--report-only", action="store_true")
    args = p.parse_args()

    settings = get_settings()
    configure_logging("WARNING", settings.log_dir)
    exp_dir = args.experiment_dir or settings.output_dir / "experiments" / f"cnn_robust_{datetime.now(UTC):%Y%m%d-%H%M%S}"
    runs_dir = exp_dir / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)

    if not args.report_only:
        cfg = CNNTrainingConfig(max_epochs=args.max_epochs, patience=args.patience, seed=settings.random_seed)
        evaluator = RobustCNNEvaluator(settings, training_config=cfg)
        specs = evaluator.plan(args.protocols, args.normalizations, args.seeds)
        manifest = exp_dir / "experiment.json"
        if not manifest.exists():
            manifest.write_text(json.dumps({"created_at": datetime.now(UTC).isoformat(), **evaluator.experiment_config(),
                                            "seeds": args.seeds, "protocols": args.protocols,
                                            "runs": [s.run_id for s in specs]}, indent=2), encoding="utf-8")
        for i, spec in enumerate(specs, 1):
            path = runs_dir / f"{spec.run_id}.json"
            if path.exists():
                print(f"[{i}/{len(specs)}] {spec.run_id}: already done", flush=True)
                continue
            print(f"[{i}/{len(specs)}] {spec.run_id}: train {list(spec.train)} | val {list(spec.validation)} | "
                  f"evaluate {list(spec.evaluate)}", flush=True)
            record = evaluator.run(spec, on_epoch=lambda e: print(f"    epoch {e}", flush=True))
            tmp = path.with_suffix(".tmp")
            tmp.write_text(json.dumps(record), encoding="utf-8")
            tmp.replace(path)
            seg = record["evaluated"]["segment"]
            print(f"    -> best epoch {record['best_epoch']}/{record['epochs_run']}, macro-F1 {seg['f1_macro']:.3f}, "
                  f"acc {seg['accuracy']:.3f}, AUC {seg['roc_auc']}, {record['seconds']} s", flush=True)

    from app.ml.deep.robust_report import write_report

    out = write_report(exp_dir, settings)
    print(f"\nReport: {out}")


if __name__ == "__main__":
    main()
