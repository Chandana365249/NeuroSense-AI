"""FACED evaluation protocol v1 (frozen before any FACED features, labels or model results were computed).

Split (seed 20261026; uses only participant IDs and the published stimulus table, never ratings or EEG):
- Final test participants: the first 25 of numpy.random.default_rng(SEED).permutation(sorted participant IDs).
- Final test videos: one clip per targeted emotion category (9 of 28), drawn with the same generator from each
  category's sorted video indices. Stratifying by the published stimulus category uses no participant rating.
- Development = the other 98 participants x the other 19 videos.

Settings (reported separately, never combined):
- S1 STRICT (primary): train on development participants x development videos; test on test participants x test
  videos. Neither the test people nor the test clips are seen in training.
- S2 UNSEEN PEOPLE, SEEN CLIPS: train on development participants x all 28 videos; test on test participants x all
  28 videos. Clips (other people's responses to them) are seen in training.
- S3 STRICT + UNSUPERVISED CALIBRATION: as S1, but each person's features are z-scored with that person's own
  unlabelled segments from the development videos (requires an unlabelled recording of the new person).

Model selection happens only inside development data: 5 inner folds; fold k validates on development participant
group k x development video group k after training on the other participant groups x the other video groups.
Selection metric: pooled trial balanced accuracy over the fold's validation trials, averaged over folds.
Label: high_valence if self-reported valence > 3.5 on the 0-7 scale (pipeline.VALENCE_THRESHOLD).
Trial prediction: mean segment probability > 0.5. Primary metric: pooled trial balanced accuracy on the S1 test set
with a participant-cluster bootstrap 95% CI; per-participant results and ordinary accuracy are also reported.
The final test is run once per setting, with the configuration chosen by the inner folds.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass

import numpy as np

PROTOCOL_VERSION = "faced-protocol-v1"
SEED = 20261026
N_TEST_PARTICIPANTS = 25
N_INNER_FOLDS = 5


@dataclass(frozen=True)
class FacedSplit:
    protocol_version: str
    seed: int
    test_participants: tuple[str, ...]
    dev_participants: tuple[str, ...]
    test_videos: tuple[int, ...]
    dev_videos: tuple[int, ...]
    inner_participant_groups: tuple[tuple[str, ...], ...]
    inner_video_groups: tuple[tuple[int, ...], ...]

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=1)

    @property
    def sha256(self) -> str:
        return hashlib.sha256(json.dumps(asdict(self), sort_keys=True).encode()).hexdigest()


def make_split(participant_ids: list[str], video_categories: dict[int, str]) -> FacedSplit:
    rng = np.random.default_rng(SEED)
    pids = sorted(participant_ids)
    order = [pids[i] for i in rng.permutation(len(pids))]
    test_p = tuple(sorted(order[:N_TEST_PARTICIPANTS]))
    dev_p = tuple(sorted(order[N_TEST_PARTICIPANTS:]))
    test_v = []
    for category in sorted(set(video_categories.values())):
        vids = sorted(v for v, c in video_categories.items() if c == category)
        test_v.append(vids[int(rng.integers(len(vids)))])
    test_v_t = tuple(sorted(test_v))
    dev_v = tuple(sorted(set(video_categories) - set(test_v_t)))
    p_groups = tuple(tuple(sorted(g.tolist())) for g in np.array_split(np.array(dev_p)[rng.permutation(len(dev_p))], N_INNER_FOLDS))
    v_groups = tuple(tuple(sorted(int(x) for x in g)) for g in np.array_split(np.array(dev_v)[rng.permutation(len(dev_v))], N_INNER_FOLDS))
    split = FacedSplit(PROTOCOL_VERSION, SEED, test_p, dev_p, test_v_t, dev_v, p_groups, v_groups)
    assert not set(test_p) & set(dev_p) and not set(test_v_t) & set(dev_v)
    assert sorted(sum(p_groups, ())) == list(dev_p) and sorted(sum(v_groups, ())) == list(dev_v)
    return split
