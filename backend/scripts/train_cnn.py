"""Train and evaluate the EEG CNN on real DEAP data (participant-aware, same split as the baseline).
The CNN is registered but NOT activated; the Review 2 Random Forest stays the production model.

Usage (from backend/):  python -m scripts.train_cnn [--max-epochs 60] [--patience 10] [--final-fit train+validation|train]
"""

from __future__ import annotations

import argparse

from app.core.config import get_settings
from app.core.logging import configure_logging
from app.ml.deep.training import CNNTrainer, CNNTrainingConfig


def _fmt(m: dict) -> str:
    auc = "n/a" if m.get("roc_auc") is None else f"{m['roc_auc']:.3f}"
    return (f"acc {m['accuracy']:.3f} | bal-acc {m['balanced_accuracy']:.3f} | macro-F1 {m['f1_macro']:.3f} | "
            f"prec {m['precision_macro']:.3f} | rec {m['recall_macro']:.3f} | AUC {auc} | n={m['n_samples']}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--max-epochs", type=int, default=60)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--learning-rate", type=float, default=1e-3)
    p.add_argument("--final-fit", default="train+validation", choices=["train+validation", "train"])
    args = p.parse_args()
    settings = get_settings()
    configure_logging("WARNING", settings.log_dir)
    cfg = CNNTrainingConfig(max_epochs=args.max_epochs, patience=args.patience, batch_size=args.batch_size,
                            learning_rate=args.learning_rate, final_fit=args.final_fit, seed=settings.random_seed)
    result = CNNTrainer(settings, training_config=cfg).run(progress=lambda stage, f: print(f"[{f:4.0%}] {stage}", flush=True))
    m = result.metadata
    tc = m["training_config"]
    print(f"\nModel version : {result.version}  (registered, not active)")
    print(f"Architecture  : {m['architecture']['parameters']} parameters, input {m['input']['shape']}")
    print(f"Split         : train {m['split']['train']} | val {m['split']['validation']} | test {m['split']['test']}")
    print(f"Epochs        : ran {tc['epochs_run']}, selected epoch {tc['best_epoch']} (validation {tc['selection_metric']} "
          f"{tc['best_validation']['score']:.3f}); final fit = {tc['final_fit']}")
    print(f"Training time : {m['training_seconds']} s")
    print("\nVALIDATION participants (selected epoch, train-only model):")
    print(f"  segment {_fmt(m['validation']['cnn']['segment'])}")
    print(f"  trial   {_fmt(m['validation']['cnn']['trial'])}")
    print("\nTEST participants (final model, evaluated once):")
    print(f"  segment {_fmt(m['test']['segment'])}")
    print(f"  trial   {_fmt(m['test']['trial'])}")
    print(f"  majority reference {_fmt(m['test_reference_majority']['segment'])}")
    print(f"  confusion (rows=true {m['test']['segment']['class_order']}): {m['test']['segment']['confusion_matrix']}")


if __name__ == "__main__":
    main()
