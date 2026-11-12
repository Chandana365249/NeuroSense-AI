# FACED per-person calibration experiment (exploratory)

**Result.** Per-person calibration raised trial balanced accuracy for self-reported valence, but less than the headline
number suggests. The best available estimate is **59.5%** (95% CI 56.4–62.1%), measured with whole presentation blocks held
out. The pre-declared analysis reached 64.4%, but a later control showed that part of that gain comes from clips of the
same block appearing in both calibration and test. **The 80% target was not reached.**

Calibrated results apply only to a person who has first rated clips. They do not replace the strict held-out 56.6%, which
measures a new person with no calibration.

## Pre-declared experiment

`backend/scripts/faced_calibration_experiment.py`; the plan was written to `outputs/faced/calibration/protocol.json` before
any model was fitted.

- **Data:** all 123 participants × 28 clips, `faced-v2` features, valence above 3.5, logistic regression C = 0.01.
- **Folds:** 3 seeds; 4 clip folds of 7, dealt evenly across emotion categories; 5 participant groups.
- **Separation:** a person's test clips are never used for their calibration. Population models see neither the tested
  person nor the tested clips.
- **Primary comparison:** D (combined) minus A (no calibration), with 21 rated clips.
- **Rule:** the bootstrap 95% interval must be above 0, and the Holm-corrected Wilcoxon p below 0.05.

| Condition | Rated clips | Pooled balanced accuracy (95% CI) | Mean per participant |
|---|---|---|---|
| A: population, no calibration | 0 | 0.514 (0.495–0.532) | 0.506 |
| B: unlabelled normalisation | 21 | 0.531 (0.516–0.546) | 0.533 |
| C: personal model | 7 | 0.579 (0.560–0.597) | 0.551 |
| C: personal model | 14 | 0.625 (0.604–0.645) | 0.592 |
| C: personal model | 21 | 0.646 (0.624–0.668) | 0.615 |
| D: combined (primary) | 7 | 0.586 (0.570–0.601) | 0.555 |
| D: combined (primary) | 14 | 0.620 (0.601–0.637) | 0.587 |
| D: combined (primary) | 21 | **0.644** (0.624–0.663) | 0.612 |

Primary difference: **+0.130** (95% CI 0.104–0.157), Holm p < 0.001; 96 participants improved and 23 got worse. D at 7 and 14
clips and C at 21 clips also met the rule. B did not: its interval includes 0. In 51 of 1,476 person-fold cases at 21 clips,
the calibration clips held only one class, and the declared fallback was used.

## Control added after seeing the result

`backend/scripts/faced_calibration_controls.py`; results in `outputs/faced/calibration/controls.json`. This control was
**not** pre-declared. It was added to test a specific concern.

FACED presents clips in blocks of related clips, and EEG drifts during a session. The stratified folds above left clips from
the same block on both sides of the split, so a personal model could recognise the part of the session rather than any
response to the clip.

Blocks were inferred per person from the 6 longest gaps between clips. 93% of inferred blocks have the expected 4 clips, and
98% hold a single valence kind. Each condition has 24 calibration clips and 4 test clips.

| Condition | Pooled balanced accuracy (95% CI) | Mean per participant |
|---|---|---|
| Personal model, 4 random clips held out | 0.657 (0.636–0.679) | 0.607 |
| **Personal model, whole block held out** | **0.595** (0.564–0.621) | **0.539** |
| No EEG: rating of the calibration clip nearest in time | 0.563 (0.532–0.593) | 0.507 |
| No EEG: person's most common rating | 0.463 (0.418–0.508) | 0.361 |

## Interpretation

- Holding out whole blocks removes about 6 points of pooled accuracy and about 7 points of per-participant accuracy. **The
  64.4% should not be quoted as the calibrated accuracy.**
- The nearest-in-time baseline uses no EEG at all and reaches 56.3%, so ratings and presentation order alone explain part
  of the remaining result.
- Within a person, the block-held-out score is 53.9%, a small gain on 50.6%. Pooled scores are higher than per-participant
  scores because they also reflect differences between people.
- The baseline (A) and the block control use different folds, so the comparison between them is approximate.

## Limitations

- Exploratory: every participant has been used or examined before.
- The block control was added after the result was seen, and blocks are inferred, imperfectly in about 7% of cases.
- Holding out one block does not remove slower drift across a whole session.
- One classifier and one feature set were used. The 500-resample control intervals are less precise than the
  2,000-resample intervals above.

## Reproduction and safety

From `backend/`:

```bash
python -m scripts.faced_calibration_experiment    # refuses to run twice
python -m scripts.faced_calibration_controls
```

No calibrated model was saved, registered or activated. The application, the existing FACED and DEAP models, the earlier
results and the production database were not changed.
