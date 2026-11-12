"""NeuroSense Fusion research package: labels, splits, leakage guards and model shapes. Synthetic data only."""

from __future__ import annotations

import numpy as np
import pytest

from app.ml.dataset import EEG_CHANNEL_NAMES
from research.neurosense_fusion import adapters as ad
from research.neurosense_fusion import protocol as pr

ORDER = list(range(1, 29))
BLOCKS = [ORDER[i:i + 4] for i in range(0, 28, 4)]
SPLIT = {"dev_participants": ["a", "b", "c"], "test_participants": ["t"], "dev_videos": [1, 2, 3], "test_videos": [9],
         "inner_participant_groups": [["a"]], "inner_video_groups": [[1]]}


def test_labels_use_each_datasets_own_midpoint_and_keep_the_scale_explicit() -> None:
    assert ad.valence_label(3.51, "FACED") == 1 and ad.valence_label(3.5, "FACED") == 0
    assert ad.valence_label(5.01, "DEAP") == 1 and ad.valence_label(5.0, "DEAP") == 0
    assert ad.normalised_rating(0.0, "FACED") == 0 and ad.normalised_rating(9.0, "DEAP") == 1
    with pytest.raises(ValueError):
        ad.valence_label(8.0, "FACED")


def test_all_shared_channels_exist_in_deap_and_keep_their_order() -> None:
    assert len(ad.SHARED_CHANNELS) == 30 and set(ad.SHARED_CHANNELS) <= set(EEG_CHANNEL_NAMES)


def test_a_deap_trial_becomes_the_shared_representation_from_the_named_channels() -> None:
    rng = np.random.default_rng(0)
    t = np.arange(8064) / 128
    eeg = rng.standard_normal((32, 8064))
    eeg[EEG_CHANNEL_NAMES.index("Oz")] += 20 * np.sin(2 * np.pi * 10 * t)
    segments, kept = ad.deap_trial_segments(eeg, ad.fpipe._preprocessor)
    assert segments.shape == (7, 30, 500) and kept == list(range(7))
    assert segments[:, ad.SHARED_CHANNELS.index("Oz")].std() > 3 * segments[:, ad.SHARED_CHANNELS.index("Cz")].std()


def test_normalisation_uses_only_the_segment_itself_and_keeps_the_spatial_pattern() -> None:
    rng = np.random.default_rng(1)
    x = rng.standard_normal((2, 30, 500))
    x[:, 0] *= 5
    alone = ad.normalise_segments(x[:1])
    assert np.array_equal(alone[0], ad.normalise_segments(np.stack([x[0], x[1] * 100]))[0])
    assert alone[0, 0].std() > 3 * alone[0, 1].std()
    assert np.allclose(ad.normalise_segments(x * 50), ad.normalise_segments(x), atol=1e-4)


def test_compatibility_table_never_marks_absent_datasets_as_integrated() -> None:
    for name in ("DREAMER", "AMIGOS", "SEED"):
        assert ad.COMPATIBILITY[name]["status"].startswith("not available")
    assert "never merged" in ad.COMPATIBILITY["SEED"]["labels"]


def test_inner_folds_hold_out_both_people_and_clips() -> None:
    p = np.array(["a", "a", "b", "b", "c", "t"])
    v = np.array([1, 2, 1, 2, 3, 9])
    train, val = pr.inner_rows(p, v, SPLIT, 0)
    assert p[val].tolist() == ["a"] and v[val].tolist() == [1]
    assert "a" not in p[train] and 1 not in v[train] and "t" not in p[train]
    pr.check_disjoint(p, v, train, val, SPLIT)


@pytest.mark.parametrize("row", [("t", 2), ("b", 9)])
def test_leakage_check_catches_a_test_person_or_clip_in_training(row: tuple[str, int]) -> None:
    p = np.array(["b", row[0], "t"])
    v = np.array([2, row[1], 9])
    with pytest.raises(AssertionError):
        pr.check_disjoint(p, v, np.array([True, True, False]), np.array([False, False, True]), SPLIT)


def test_frozen_deap_participants_stop_the_run() -> None:
    pr.check_no_frozen(np.array(["s01", "s25"]))
    with pytest.raises(AssertionError):
        pr.check_no_frozen(np.array(["s01", "s12"]))


@pytest.mark.parametrize(("n", "first_test", "clips"), [(7, 9, 20), (14, 17, 12), (21, 25, 4)])
def test_calibration_comes_first_and_test_blocks_are_whole(n: int, first_test: int, clips: int) -> None:
    calibration, test = pr.personal_plan(ORDER, BLOCKS, n)
    assert calibration == ORDER[:n] and test[0] == first_test and len(test) == clips
    pr.check_personal_plan(ORDER, BLOCKS, calibration, test)


def test_personal_plan_checks_catch_future_ratings_and_split_blocks() -> None:
    with pytest.raises(AssertionError):
        pr.check_personal_plan(ORDER, BLOCKS, [25, 26], [1, 2, 3, 4])
    with pytest.raises(AssertionError):
        pr.check_personal_plan(ORDER, BLOCKS, [1, 2], [3, 4, 5])
    with pytest.raises(AssertionError):
        pr.check_personal_plan(ORDER, BLOCKS, [1, 2], [2, 3, 4])


def test_nearest_rating_baseline_only_looks_backwards() -> None:
    assert pr.nearest_past_rating(ORDER, [1, 2, 3], 9) == 3
    with pytest.raises(AssertionError):
        pr.nearest_past_rating(ORDER, [5, 6], 2)


def test_model_shapes_losses_and_gradient_reversal() -> None:
    torch = pytest.importorskip("torch")
    from research.neurosense_fusion import model as m

    net = m.FusionNet()
    logit, z = net(torch.randn(4, 30, 500))
    assert logit.shape == (4,) and z.shape == (4, m.LATENT)
    counts = m.parameter_counts(net)
    assert counts["total"] < 50_000 and counts["encoder"] > 0
    a, b = torch.randn(16, 8), torch.randn(16, 8)
    assert m.coral_loss(a, a).item() == 0 and m.coral_loss(a, b).item() > 0
    assert abs(m.mmd_loss(a, a).item()) < 1e-6 and m.contrastive_loss(a, b).item() > 0
    x = torch.ones(2, m.LATENT, requires_grad=True)
    net.domain_logits(x, 1.0).sum().backward()
    forward = torch.ones(2, m.LATENT, requires_grad=True)
    net.domain_head(forward).sum().backward()
    assert torch.allclose(x.grad, -forward.grad)
