"""Run preprocessing on one real DEAP trial and save development figures to outputs/graphs/.

Usage (from backend/):  python -m scripts.preprocessing_report [--participant s01] [--trial 0]
"""

from __future__ import annotations

import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from app.core.config import get_settings
from app.core.logging import configure_logging
from app.ml.dataset import EEG_CHANNEL_NAMES
from app.ml.factory import make_dataset, make_feature_extractor, make_label_config, make_preprocessor
from app.ml.feature_table import build_feature_table

SHOW_CHANNELS = ("Fp1", "F3", "F4", "Cz", "O1", "O2")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--participant", default="s01")
    parser.add_argument("--trial", type=int, default=0)
    args = parser.parse_args()

    settings = get_settings()
    configure_logging("INFO", settings.log_dir)
    settings.ensure_output_dirs()
    dataset = make_dataset(settings)
    pre = make_preprocessor(settings)
    sr = settings.sampling_rate
    out = settings.graphs_dir

    raw, ratings = dataset.load_trial(args.participant, args.trial)
    idx = [EEG_CHANNEL_NAMES.index(c) for c in SHOW_CHANNELS]
    after_baseline = pre.remove_baseline(pre.validate(raw))
    filtered = pre.bandpass_filter(after_baseline)
    cleaned, mask = pre.remove_artifacts(filtered)
    result = pre.preprocess(raw, has_baseline=True)
    title = f"{args.participant} trial {args.trial} (valence {ratings[0]:.2f}, arousal {ratings[1]:.2f})"

    window = slice(0, 10 * sr)
    t = np.arange(window.stop) / sr
    for name, data, label in (("raw", after_baseline, "Raw EEG (baseline removed)"), ("filtered", cleaned, "Band-pass 4-45 Hz + artifact clipping")):
        fig, axes = plt.subplots(len(idx), 1, figsize=(11, 8), sharex=True)
        for ax, ch in zip(axes, idx, strict=True):
            ax.plot(t, data[ch, window], lw=0.6)
            ax.set_ylabel(EEG_CHANNEL_NAMES[ch], rotation=0, labelpad=18)
        axes[-1].set_xlabel("Time (s), first 10 s of stimulus")
        fig.suptitle(f"{label} - {title}")
        fig.tight_layout()
        fig.savefig(out / f"eeg_{name}_example.png", dpi=120)
        plt.close(fig)

    seg_len = pre.config.segment_samples
    ch = EEG_CHANNEL_NAMES.index("Cz")
    fig, ax = plt.subplots(figsize=(11, 3.5))
    full_t = np.arange(cleaned.shape[1]) / sr
    ax.plot(full_t, cleaned[ch], lw=0.4, color="0.3")
    for i in range(result.n_segments_total):
        color = "tab:red" if i in result.rejected_segment_indices else ("tab:blue" if i % 2 == 0 else "tab:green")
        ax.axvspan(i * seg_len / sr, (i + 1) * seg_len / sr, alpha=0.12, color=color)
    ax.set_title(f"Segmentation into {pre.config.segment_seconds:.0f} s windows (Cz) - {result.n_segments_total} segments, "
                 f"{len(result.rejected_segment_indices)} rejected")
    ax.set_xlabel("Time (s)")
    fig.tight_layout()
    fig.savefig(out / "eeg_segmented_example.png", dpi=120)
    plt.close(fig)

    table = build_feature_table(dataset, pre, make_feature_extractor(settings), settings.cache_dir)
    labels = table.labels(make_label_config(settings))
    names = table.feature_names
    show = ["F3_logpow_alpha", "F4_logpow_alpha", "asym_F4-F3_alpha", "Cz_relpow_beta", "O1_spectral_entropy", "Fz_logpow_theta"]
    fig, axes = plt.subplots(2, 3, figsize=(12, 6.5))
    for ax, feat in zip(axes.flat, show, strict=True):
        col = table.X[:, names.index(feat)]
        lo, hi = np.percentile(col, [1, 99])
        bins = np.linspace(lo, hi, 40)
        for cls in sorted(set(labels)):
            ax.hist(col[labels == cls], bins=bins, alpha=0.55, density=True, label=cls)
        ax.set_title(feat, fontsize=10)
    axes.flat[0].legend(fontsize=8)
    fig.suptitle(f"Feature distributions by class, all {table.X.shape[0]} segments, {len(set(table.participant))} participants")
    fig.tight_layout()
    fig.savefig(out / "feature_distributions.png", dpi=120)
    plt.close(fig)

    print(f"Trial shape (channels, samples) : {raw.shape}")
    print(f"After baseline removal          : {after_baseline.shape}")
    print(f"Clipped sample fraction         : {mask.mean():.4%}")
    print(f"Segments kept / total           : {result.segments.shape[0]} / {result.n_segments_total}")
    print(f"Feature table                   : {table.X.shape[0]} segments x {table.X.shape[1]} features")
    print(f"Rejected segments (all data)    : {table.rejected_segments}")
    uniq, counts = np.unique(labels, return_counts=True)
    print(f"Class balance (segments)        : {dict(zip(uniq.tolist(), counts.tolist(), strict=True))}")
    print(f"Figures saved to                : {out}")


if __name__ == "__main__":
    main()
