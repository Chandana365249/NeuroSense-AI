"""Build (or load) the deep-learning segment table from real DEAP data and check it matches the
feature table used by the Review 2 baseline.

Usage (from backend/):  python -m scripts.segment_summary
"""

from __future__ import annotations

import numpy as np

from app.core.config import get_settings
from app.core.logging import configure_logging
from app.ml.deep.data import ChannelNormalizer, build_segment_table
from app.ml.factory import make_dataset, make_feature_extractor, make_label_config, make_preprocessor
from app.ml.feature_table import build_feature_table
from app.ml.splitting import split_participants


def main() -> None:
    settings = get_settings()
    configure_logging("WARNING", settings.log_dir)
    settings.ensure_output_dirs()
    dataset, pre = make_dataset(settings), make_preprocessor(settings)
    seg = build_segment_table(dataset, pre, settings.cache_dir)
    feat = build_feature_table(dataset, pre, make_feature_extractor(settings), settings.cache_dir)

    aligned = (np.array_equal(seg.participant, feat.participant) and np.array_equal(seg.trial, feat.trial)
               and np.array_equal(seg.segment, feat.segment) and np.array_equal(seg.ratings, feat.ratings))
    labels = seg.labels(make_label_config(settings))
    split = split_participants(sorted(set(seg.participant.tolist())), settings.test_participants,
                               settings.validation_participants, settings.random_seed)
    train_rows = np.flatnonzero(split.mask(seg.participant, "train"))
    norm = ChannelNormalizer.fit(seg.X, train_rows)

    print(f"Segment table       : {seg.X.shape} {seg.X.dtype} (rows, channels, samples)")
    print(f"CNN input tensor    : (batch, 1, {seg.shape[0]}, {seg.shape[1]})")
    print(f"Rejected segments   : {seg.rejected_segments}")
    print(f"Aligned with RF rows: {aligned}")
    for part in ("train", "validation", "test"):
        m = split.mask(seg.participant, part)
        uniq, counts = np.unique(labels[m], return_counts=True)
        print(f"  {part:<10} {getattr(split, part)}  segments={int(m.sum())}  {dict(zip(uniq.tolist(), counts.tolist(), strict=True))}")
    print(f"Train channel mean  : {np.round(norm.mean[:4], 3)} ... std {np.round(norm.std[:4], 3)} ...")


if __name__ == "__main__":
    main()
