"""Private storage of uploaded EEG signals, so an upload's result can be explained later without re-sending it.

One compressed file per prediction under outputs/uploads/<user id>/<prediction id>.npz. Files are only read through a
prediction that belongs to the requesting user. Synthetic test signals are not stored.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from app.core.config import Settings


def upload_path(settings: Settings, user_id: int, prediction_id: int) -> Path:
    return settings.output_dir / "uploads" / str(int(user_id)) / f"{int(prediction_id)}.npz"


def save_upload(settings: Settings, user_id: int, prediction_id: int, signal: np.ndarray, sampling_rate: int,
                includes_baseline: bool) -> Path:
    path = upload_path(settings, user_id, prediction_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.stem + ".tmp.npz")
    # float64 keeps the stored signal bit-identical to the analysed one, so the result reproduces exactly.
    np.savez_compressed(tmp, signal=np.asarray(signal, dtype=np.float64), sampling_rate=int(sampling_rate),
                        includes_baseline=bool(includes_baseline))
    tmp.replace(path)
    return path


def load_upload(settings: Settings, user_id: int, prediction_id: int) -> dict[str, Any] | None:
    path = upload_path(settings, user_id, prediction_id)
    if not path.exists():
        return None
    with np.load(path, allow_pickle=False) as z:
        return {"signal": z["signal"], "sampling_rate": int(z["sampling_rate"]),
                "includes_baseline": bool(z["includes_baseline"]), "synthetic_test_signal": False}
