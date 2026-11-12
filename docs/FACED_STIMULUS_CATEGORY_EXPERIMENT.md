# FACED stimulus-category experiment (exploratory)

**Status: exploratory, finished, below target.** Using EEG, the selected model told the intended positive vs negative
category of known FACED film clips apart for 25 new participants with **55.8% trial-level balanced accuracy** (95% CI
53.0–59.0%). **The 80% target was not reached.** As the protocol required, the experiment stops here. The label rule,
participants, metric and split were not changed afterwards.

## What this task is, and is not

| | This experiment | Earlier FACED evaluation |
|---|---|---|
| Label | the **intended category of the video** (from the dataset's stimulus table) | the **participant's own valence rating** (> 3.5) |
| Question | is this EEG from a positive or a negative clip? | did this person rate the clip as high or low valence? |
| Videos | **known videos**: the same 24 clips appear for training and test participants | strict setting: test clips never seen in training |
| Participants | test participants never used in training | same |

This task does **not** measure prediction of anyone's feelings. It also does **not** measure generalisation to new
videos. A model could succeed here by recognising the clips' content-driven EEG responses (picture, sound, editing), not
emotion.

**Why it is exploratory, not an independent confirmation:** the 25 test participants are the final-test participants of
`faced-protocol-v1`. Their recordings were already examined in the earlier FACED valence evaluation (setting S2 used all
28 of their clips). Confirming any future claim would require a new, untouched test set.

## Protocol (declared before any model was fitted)

Fixed in `outputs/faced/stimulus_category/protocol.json` (SHA-256 `9e61d222…364a18b9`). Script:
`backend/scripts/faced_stimulus_experiment.py`.

1. **Stimulus labels.** Source: `stimuli/stimuli.tsv` `targeted_valence` (Chen et al. 2023, Supplementary Table S1). It
   was cross-checked against every participant's `events.tsv` (`binary_label`, `emotion_label`): **123 of 123
   participants, 0 mismatches**. It also agrees with the dataset README.
   - Positive = 1: videos 17–28 (amusement, inspiration, joy, tenderness; 3 each).
   - Negative = 0: videos 1–12 (anger, disgust, fear, sadness; 3 each).
   - **Excluded:** the 4 neutral videos (13–16). They belong to neither class of a positive-vs-negative task.
   - The classes are balanced by design: 12 vs 12 clips per participant.
2. **Data.** The existing `faced-v2` features, unchanged. The protocol records their SHA-256.
   - Last 28 s of each clip, 30 shared channels, common average reference, 250 Hz.
   - 4–45 Hz band-pass, 4 s segments, 412 band-power, Hjorth and asymmetry features.
   - Every step is computed per trial; nothing is fitted across participants.
3. **Participants.** The `faced-protocol-v1` participant partition is reused unchanged: 98 training / 25 test
   participants (split SHA-256 `01e74a4e…1d163c41`). Reusing it avoids choosing a new split.
   - Videos are not split.
   - No recording or segment from a test participant enters training, scaling, model selection or threshold choice.
   - No per-person normalisation is used, because it would be fitted on the test person's own recordings.
4. **Model selection.** 5 participant-grouped folds over the 98 training participants, using the protocol's existing
   participant groups. Metric: pooled trial balanced accuracy, averaged over folds; ties go to the simpler model.
5. **Candidates (compact, carried over from the earlier FACED comparison):**
   - majority class;
   - logistic regression, C ∈ {0.01, 0.1, 1};
   - the project's Random Forest.
   - No other tuning.
6. **Trial rule.** The mean of the segment probabilities; above **0.5** means positive. This threshold is fixed and was
   not tuned.
7. **Final.** The selected model was fitted on 98 participants × 24 clips (2,352 trials) and evaluated **once** on 25
   participants × 24 clips (600 trials).
   - `--phase freeze` recorded the selection, the protocol hash and these leakage checks before the run. All passed:
     - inner folds contain no test participant;
     - final training contains no test participant;
     - the neutral clips are excluded;
     - labels are the stimulus category.
   - `--phase final` refuses to run twice.

**Presentation order (checked before training).** Clips were shown in category blocks, in an order that varied by
participant. Positive clips were shown earlier on average for 54% of participants; 72 started with a positive block and 51
with a negative one. So, across the group, category is not tied to time in the session.

## Model selection (training participants only)

Pooled trial balanced accuracy in each of the 5 participant-grouped folds:

| Candidate | Folds | Mean |
|---|---|---|
| majority | 0.500 for all five | 0.500 |
| **logistic regression, C = 0.01** | 0.590, 0.552, 0.587, 0.546, 0.579 | **0.571** ← selected |
| logistic regression, C = 0.1 | 0.608, 0.548, 0.565, 0.542, 0.550 | 0.563 |
| logistic regression, C = 1 | 0.598, 0.544, 0.540, 0.533, 0.548 | 0.552 |
| Random Forest | 0.592, 0.577, 0.546, 0.546, 0.572 | 0.567 |

## Result on the 25 held-out participants (run once)

| Metric (trial level, 600 trials) | Value |
|---|---|
| **Balanced accuracy (primary)** | **0.558** (95% CI 0.530–0.590, participant-cluster bootstrap, 2,000 resamples) |
| Accuracy | 0.558 (95% CI 0.530–0.590) |
| Macro-F1 | 0.558 |
| ROC AUC | 0.581 |
| Class balance | 300 positive / 300 negative (50%) |
| Majority baseline | accuracy 0.500, balanced accuracy 0.500 |
| Predicted positive | 51.2% |

Confusion matrix (rows: true category, columns: predicted category):

| | predicted negative | predicted positive |
|---|---|---|
| **negative clip** | 164 | 136 |
| **positive clip** | 129 | 171 |

**Per participant** (24 trials each):

- Balanced accuracy: mean **0.558**, SD 0.077, 95% t-CI 0.527–0.590, range 0.458–0.750.
- 16 of 25 participants are at or below 0.542; the best is sub-063 at 0.750.

| Participant | Balanced accuracy | | Participant | Balanced accuracy |
|---|---|---|---|---|
| sub-000 | 0.667 | | sub-058 | 0.625 |
| sub-001 | 0.542 | | sub-063 | 0.750 |
| sub-003 | 0.708 | | sub-073 | 0.542 |
| sub-006 | 0.542 | | sub-077 | 0.583 |
| sub-011 | 0.500 | | sub-096 | 0.500 |
| sub-013 | 0.583 | | sub-105 | 0.667 |
| sub-016 | 0.500 | | sub-107 | 0.542 |
| sub-024 | 0.500 | | sub-110 | 0.500 |
| sub-028 | 0.542 | | sub-111 | 0.458 |
| sub-032 | 0.500 | | sub-117 | 0.500 |
| sub-044 | 0.583 | | sub-121 | 0.625 |
| sub-046 | 0.500 | | | |
| sub-050 | 0.458 | | | |
| sub-054 | 0.542 | | | |

**Segment level** (separate, not the primary metric; 4,197 segments): balanced accuracy 0.546, ROC AUC 0.566.

**Per clip** (25 test participants each): accuracy ranges from 0.32 (clip 24) to 0.80 (clip 27). No clip category is
reliably recognised.

**Descriptive only:**
- Participants' own valence labels agreed with the clip's intended category in 84.5% of these test trials.
- Scored against self-reported valence, this model's balanced accuracy is 0.539.

## Conclusion

- **The target was not reached.** Trial-level balanced accuracy is **55.8%**; even the upper end of the 95% interval
  (59.0%) is far below 80%.
- The result is above chance: the CI lower bound of 53.0% is above the 50% baseline. It is no stronger than the
  self-reported-valence result (56.6%). Using the intended video category as the label did **not** make the task
  substantially more learnable with these features.
- This is an exploratory result on **known videos** and **new participants**. It does not support claims about new
  videos or about anyone's feelings.

### Observation, not acted on

- Each participant's mean ROC AUC is 0.636 (SD 0.113), higher than their balanced accuracy at the fixed 0.5 threshold.
- Within a person, the model ranks positive clips above negative ones somewhat better than its fixed threshold shows. A
  person-specific offset is the likely reason.
- Using that would need a per-person calibration step decided in advance and an untouched test set. Under this
  protocol it was **not** tried, because changing the threshold or normalisation after seeing the test result is
  exactly what the protocol forbids.

## Limitations

- **Exploratory and not independent:** the test participants' recordings were examined before (see above).
- One protocol, one seed and one held-out run: 25 participants and 600 trials.
- The task can be solved by video-specific EEG responses unrelated to emotion. Even a high score would not show emotion
  recognition.
- Only the last 28 s of each clip and hand-crafted features were used; the neutral clips are excluded.
- Not diagnostic or clinical.

## Safety and integrity

- The research artifact `outputs/faced/stimulus_category/stimulus_category_lr_C0.01.joblib` is Git-ignored, not
  registered, not activated and **not shown in the application**. The app's FACED tab still shows only the earlier
  self-reported-valence model.
- **Unchanged:**
  - the production DEAP model `valence_binary-random_forest-20261008-180558` and its registry (all 70 hashed production
    files match);
  - the FACED valence model files and card;
  - the production PostgreSQL database (read-only checksum snapshot of users, predictions, reports, audit logs,
    sessions and models).
- The eight frozen DEAP participants were not accessed. Nothing was committed or pushed.

## Reproduction

From `backend/`:

```bash
python -m scripts.faced_stimulus_experiment --phase protocol   # verify labels, declare protocol (once)
python -m scripts.faced_stimulus_experiment --phase inner      # 5 participant-grouped development folds
python -m scripts.faced_stimulus_experiment --phase freeze     # record selection + leakage checks
python -m scripts.faced_stimulus_experiment --phase final      # held-out participants, once
```

Outputs: `outputs/faced/stimulus_category/` holds `protocol.json`, `inner.json`, `freeze.json`, `final_results.json` and
the model artifact. Results were generated from commit `f506e27` plus the uncommitted FACED working tree. The seed is 42.
