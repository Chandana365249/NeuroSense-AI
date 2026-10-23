"""Public views of stored model and report metadata.

Stored metadata keeps internal details (dataset location, model paths, file names). API responses use these helpers
so that only approved fields reach callers. Inputs are never modified: new structures are returned.
"""
from __future__ import annotations

import re
from typing import Any

# The only dataset facts that API responses may contain.
PUBLIC_DATASET_FIELDS = ("participant_count", "segments", "rejected_segments", "class_balance", "fingerprint")

# Keys whose values are filesystem locations or file listings; dropped wherever they appear.
_PATH_KEYS = frozenset({
    "dir", "dataset_dir", "output_dir", "model_dir", "log_dir", "model_path", "artifacts",
    "path", "paths", "file", "files", "filename", "file_name",
})

PATH_PLACEHOLDER = "[path removed]"
FILE_PLACEHOLDER = "[file removed]"
_ABSOLUTE_PATH = re.compile(
    r"[A-Za-z]:[\\/][^\s\"'<>|]*"            # Windows drive path, e.g. C:\Users\name\...
    r"|\\\\[^\s\"'<>|]+"                      # UNC path, e.g. \\server\share
    r"|(?<![\w.:/])/(?:[\w.~-]+/)+[\w.~-]*"   # POSIX absolute path with at least one directory, e.g. /home/name/x
)
_DATA_FILE = re.compile(r"\b[\w-]+\.(?:dat|bdf|joblib|pt|npz|pkl)\b", re.IGNORECASE)


def public_dataset_facts(dataset: dict[str, Any] | None) -> dict[str, Any]:
    """Allow-listed facts from a metadata `dataset` block; everything else (e.g. its `dir`) stays internal."""
    dataset = dataset if isinstance(dataset, dict) else {}
    participants = dataset.get("participants")
    count = dataset.get("participant_count")
    return {
        "participant_count": len(participants) if isinstance(participants, list) else count if isinstance(count, int) else None,
        "segments": dataset.get("segments"),
        "rejected_segments": dataset.get("rejected_segments"),
        "class_balance": dataset.get("class_balance"),
        "fingerprint": dataset.get("fingerprint"),
    }


def _scrub_text(value: str) -> str:
    return _DATA_FILE.sub(FILE_PLACEHOLDER, _ABSOLUTE_PATH.sub(PATH_PLACEHOLDER, value))


# Public view of a protocol evaluation: summary statistics and per-participant metrics, no file paths.
_EVALUATION_FOLD_FIELDS = ("held_out", "n_train_participants", "n_trials", "trial", "segment", "majority_reference",
                           "predicted_positive_share_trial", "single_label", "few_trials_flag", "within_person_auc")


def public_evaluation(evaluation: dict[str, Any] | None, include_folds: bool = True) -> dict[str, Any] | None:
    if not evaluation:
        return None
    out = {k: evaluation.get(k) for k in ("kind", "protocol_version", "label", "unit_of_analysis", "positive_class",
                                          "aggregate", "diagnostics")}
    out["n_participants"] = len(evaluation.get("folds") or [])
    if include_folds:
        out["folds"] = [{k: f.get(k) for k in _EVALUATION_FOLD_FIELDS} for f in evaluation.get("folds") or []]
    return out


def public_protocol(protocol: dict[str, Any] | None) -> dict[str, Any] | None:
    if not protocol:
        return None
    return {k: protocol.get(k) for k in ("version", "role", "development_participants", "frozen_participants_used",
                                         "algorithm_fixed_in_advance", "search_space", "seeds", "leakage_controls",
                                         "limitations")}


def sanitize_report_content(content: Any) -> Any:
    """A copy of stored report content that is safe to return: approved dataset facts only, no path-type keys and no
    absolute paths or data-file names inside text. Research values (metrics, splits, labels) are kept as they are."""
    if isinstance(content, dict):
        clean: dict[str, Any] = {}
        for key, value in content.items():
            if key in _PATH_KEYS:
                continue
            clean[key] = public_dataset_facts(value) if key == "dataset" and isinstance(value, dict) else sanitize_report_content(value)
        return clean
    if isinstance(content, list):
        return [sanitize_report_content(v) for v in content]
    if isinstance(content, str):
        return _scrub_text(content)
    return content


# ---------- training provenance ----------
# Values of training_config.final_fit written by the trainers in this repository. Provenance is reported only when the
# recorded value is one of these; anything else is "not recorded" rather than reconstructed.
_FIXED_SPLIT_FINAL_FITS = frozenset({"train + validation participants", "train+validation"})
_PROTOCOL_FINAL_FIT = "all development participants"


def _ids(value: Any) -> list[str]:
    return sorted(str(v) for v in value) if isinstance(value, list) else []


def training_provenance(meta: dict[str, Any] | None) -> dict[str, Any] | None:
    """Which participants the stored estimator was fitted on, taken only from the model's recorded metadata.

    `fit_participants` were used to fit the final estimator; `selection_participants` were used to choose between
    candidate algorithms (and are also in `fit_participants` when the final fit includes them); `held_out_participants`
    were used for neither. Counts that the metadata does not record (for example trials of a fixed-split model) are None.
    """
    if not meta:
        return None
    split = meta.get("split") if isinstance(meta.get("split"), dict) else {}
    final_fit = (meta.get("training_config") or {}).get("final_fit")
    train, validation, test = _ids(split.get("train")), _ids(split.get("validation")), _ids(split.get("test"))
    segments = split.get("segments") if isinstance(split.get("segments"), dict) else {}
    unrecorded = {
        "recorded": False, "kind": "unknown", "final_fit": final_fit if isinstance(final_fit, str) else None,
        "fit_participants": [], "selection_participants": [], "held_out_participants": [],
        "fit_segments": None, "fit_trials": None,
        "note": "This model's metadata does not record which participants its final fit used.",
    }
    if final_fit in _FIXED_SPLIT_FINAL_FITS and train:
        seg = [segments.get("train"), segments.get("validation")]
        return {
            "recorded": True, "kind": "fixed_split", "final_fit": final_fit,
            "fit_participants": sorted(train + validation), "selection_participants": validation,
            "held_out_participants": test,
            "fit_segments": sum(seg) if all(isinstance(s, int) for s in seg) else None,
            "fit_trials": None,
            "note": ("Recorded in the model's metadata: fitted on the training and validation participants; the "
                     "validation participants were also used to choose the algorithm. The test participants were not "
                     "used for fitting or selection. Under Protocol v1.1 they are development participants and their "
                     "test results have been reported, so they are no longer an untouched test set."),
        }
    if final_fit == _PROTOCOL_FINAL_FIT and train:
        groups = (meta.get("protocol") or {}).get("trial_groups")
        trials = sum(len(v) for v in groups.values()) if isinstance(groups, dict) else None
        return {
            "recorded": True, "kind": "protocol_all_development", "final_fit": final_fit,
            "fit_participants": train, "selection_participants": [], "held_out_participants": [],
            "fit_segments": segments.get("train") if isinstance(segments.get("train"), int) else None,
            "fit_trials": trials,
            "note": ("Recorded in the model's metadata: fitted on all development participants under Protocol v1.1, "
                     "with no algorithm selection. No participant is held out from this model."),
        }
    return unrecorded
