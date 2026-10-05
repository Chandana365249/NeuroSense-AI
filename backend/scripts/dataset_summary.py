"""Print and save a structural summary of the local DEAP dataset.

Usage (from backend/):  python -m scripts.dataset_summary
"""

from __future__ import annotations

import json

from app.core.config import get_settings
from app.core.logging import configure_logging
from app.ml.dataset import DEAPDataset


def main() -> None:
    settings = get_settings()
    configure_logging("WARNING")
    settings.ensure_output_dirs()
    dataset = DEAPDataset(settings.dataset_dir, settings.n_eeg_channels, settings.sampling_rate, settings.baseline_seconds)
    summary = dataset.summarize()

    print(f"Dataset directory : {summary.dataset_dir}")
    print(f"Participants      : {summary.participants_valid} valid / {summary.participants_found} found")
    print(f"Total trials      : {summary.total_trials}")
    print(f"Sampling rate     : {summary.sampling_rate_hz} Hz (DEAP preprocessed release, downsampled)")
    print(f"Baseline          : first {summary.baseline_seconds} s of each trial (pre-stimulus)")
    print(f"EEG channels used : {summary.eeg_channels_used} ({', '.join(summary.eeg_channel_names)})")
    print(f"Ratings (labels)  : {', '.join(summary.rating_names)} on a 1-9 scale")
    for name, (lo, hi) in summary.rating_ranges.items():
        print(f"    {name:<9} range {lo:.2f}-{hi:.2f}, mean {summary.rating_means[name]:.2f}")
    print()
    print(f"{'file':<8}{'ok':<5}{'trials':>7}{'channels':>10}{'samples':>9}{'seconds':>9}  labels")
    for f in summary.files:
        if f.ok:
            secs = f.samples / summary.sampling_rate_hz
            print(f"{f.file:<8}{'yes':<5}{f.trials:>7}{f.channels:>10}{f.samples:>9}{secs:>9.1f}  {f.label_shape}")
        else:
            print(f"{f.file:<8}{'NO':<5}  {f.error}")

    out = settings.metrics_dir / "dataset_summary.json"
    out.write_text(json.dumps(summary.to_dict(), indent=2), encoding="utf-8")
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
