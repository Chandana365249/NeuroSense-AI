"""Train a protocol-aware candidate model (Evaluation Protocol v1.1) and print its exploratory evaluation.

Usage (from backend/):  python -m scripts.train_candidate [--no-cv] [--activate]

- Uses only the development participants (default: the 12 trusted participants); the frozen final-test group is
  never loaded.
- Fixed algorithm (Random Forest, existing configuration), no tuning.
- Exploratory leave-one-participant-out cross-validation, reported per participant at trial level.
- The model is saved as an INACTIVE candidate. It becomes active only with --activate, or later through the
  administrator endpoint POST /api/models/{version}/activate. A running API server keeps its loaded model until
  it is restarted or a model is activated through the API.

The results are exploratory. They are not an independent test result and must not be used to claim that this
model is better than another one.
"""

from __future__ import annotations

import argparse

from app.core.config import get_settings
from app.core.logging import configure_logging
from app.ml.protocol_training import ProtocolTrainer


def _fmt(stats: dict) -> str:
    if not stats or stats.get("mean") is None:
        return "n/a"
    return f"{stats['mean']:.3f} ± {stats['sd_sample']:.3f} (95% CI {stats['ci_low']:.3f}–{stats['ci_high']:.3f}, n={stats['n']})"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-cv", action="store_true", help="skip the leave-one-participant-out evaluation")
    parser.add_argument("--activate", action="store_true",
                        help="make the new model the active model (default: save it as an inactive candidate)")
    args = parser.parse_args()
    settings = get_settings()
    configure_logging("WARNING", settings.log_dir)

    result = ProtocolTrainer(settings, run_cross_validation=not args.no_cv, activate=args.activate).run(
        progress=lambda stage, frac: print(f"[{frac:4.0%}] {stage}"))
    m = result.metadata
    print(f"\nModel version : {result.version}")
    print("Status        : " + ("ACTIVE (--activate)" if args.activate else "inactive candidate (the active model is unchanged)"))
    print(f"Participants  : {', '.join(m['protocol']['development_participants'])}")
    evaluation = m.get("evaluation")
    if evaluation:
        a = evaluation["aggregate"]
        print(f"\n{evaluation['label']}")
        print(f"  trial balanced accuracy  {_fmt(a['trial_balanced_accuracy'])}")
        print(f"  trial macro-F1           {_fmt(a['trial_f1_macro'])}")
        print(f"  within-person AUC        {_fmt(a['within_person_auc'])}")
        print(f"  majority reference (BA)  {_fmt(a['majority_trial_balanced_accuracy'])}")
        print(f"  single-label participants: {evaluation['diagnostics']['single_label_participants']}")
        for f in evaluation["folds"]:
            t = f["trial"]
            print(f"    {f['held_out']}: BA {t['balanced_accuracy']:.3f}  F1 {t['f1_macro']:.3f}  "
                  f"predicted-high share {f['predicted_positive_share_trial']:.2f}")
    print(f"\nLimitations: {m['protocol']['limitations']}")


if __name__ == "__main__":
    main()
