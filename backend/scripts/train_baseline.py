"""Train the baseline model on real DEAP data and print held-out results.

Usage (from backend/):  python -m scripts.train_baseline [--no-cv] [--activate]

The new model is saved as an inactive candidate. It replaces the active model only with --activate (or later through
the administrator endpoint POST /api/models/{version}/activate). A running API server keeps serving the model it
loaded until it is restarted or a model is activated through the API.
"""

from __future__ import annotations

import argparse

from app.core.config import get_settings
from app.core.logging import configure_logging
from app.ml.training import BaselineTrainer


def _fmt(m: dict) -> str:
    auc = "n/a" if m["roc_auc"] is None else f"{m['roc_auc']:.3f}"
    return (f"acc {m['accuracy']:.3f} | bal-acc {m['balanced_accuracy']:.3f} | macro-F1 {m['f1_macro']:.3f} | "
            f"prec {m['precision_macro']:.3f} | rec {m['recall_macro']:.3f} | AUC {auc} | n={m['n_samples']}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-cv", action="store_true", help="skip leave-one-participant-out cross-validation")
    parser.add_argument("--activate", action="store_true",
                        help="make the new model the active model (default: save it as an inactive candidate)")
    args = parser.parse_args()
    settings = get_settings()
    configure_logging("WARNING", settings.log_dir)

    result = BaselineTrainer(settings, run_cross_validation=not args.no_cv, activate=args.activate).run(
        progress=lambda stage, frac: print(f"[{frac:4.0%}] {stage}")
    )
    m = result.metadata
    print(f"\nModel version : {result.version}")
    print("Status        : " + ("ACTIVE (--activate)" if args.activate else "inactive candidate (the active model is unchanged)"))
    print(f"Label         : {m['label']['description']}")
    print(f"Split         : train {m['split']['train']} | val {m['split']['validation']} | test {m['split']['test']}")
    print(f"Segments      : {m['split']['segments']}")
    print("\nValidation participants (model selection):")
    for name, v in m["validation"].items():
        print(f"  {name:<18} {_fmt(v['segment'])}")
    print(f"\nHeld-out TEST participants, selected model = {result.algorithm}")
    print(f"  segment level      {_fmt(m['test']['segment'])}")
    print(f"  trial level        {_fmt(m['test']['trial'])}")
    print(f"  majority reference {_fmt(m['test_reference_majority']['segment'])}")
    print(f"  confusion (segment, rows=true {m['test']['segment']['class_order']}): {m['test']['segment']['confusion_matrix']}")
    cv = m.get("cross_validation")
    if cv:
        print(f"\nLeave-one-participant-out CV ({len(cv['folds'])} folds): "
              f"accuracy {cv['segment_accuracy_mean']:.3f} +/- {cv['segment_accuracy_std']:.3f}, "
              f"macro-F1 {cv['segment_f1_macro_mean']:.3f} +/- {cv['segment_f1_macro_std']:.3f}")
        for f in cv["folds"]:
            print(f"  {f['participant']}: acc {f['segment_accuracy']:.3f}  F1 {f['segment_f1_macro']:.3f}")
    print(f"\nArtifacts: {m['model_path']}")


if __name__ == "__main__":
    main()
