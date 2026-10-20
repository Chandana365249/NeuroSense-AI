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
