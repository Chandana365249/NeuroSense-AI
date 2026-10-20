# Cross-Participant, Leave-Video-Out Valence Decoding

**Status: exploratory (12 participants, Protocol v1.1).** This is not a final-test result and not proof of generalization to all new people (Decision C). It follows the within-person diagnostic and stimulus-confound control in `docs/EXPLORATORY_WITHIN_PERSON_DIAGNOSTIC.md`.

> Task A predicts a person's own self-reported valence category. Task B predicts the **consensus** valence category of a video among other participants. Task B is a separate research question: predicting a consensus is **not** equivalent to predicting a person's own emotional experience.

## 1. Research question

Can a model trained on **other participants** and **other videos** predict, for a held-out participant watching held-out videos:

- **A.** that participant's own valence labels?
- **B.** the shared valence-consensus labels of those videos?

## 2. Design (declared before running, 2026-10-11)

**Data**
- Only the 12 trusted DEAP participants (s01, s02, s03, s05, s07, s08, s09, s11, s15, s16, s22, s25), from the configured dataset and the existing feature cache (fingerprint `3c5359083bc90ae5`; 440 features per 4 s segment).
- No third-party data. The 8 frozen test participants are never accessed.
- Trial index *t* is the same video (*t*) for every participant. This was verified from the trusted ratings in the stimulus-confound control: cross-participant correlation 0.443, against a shuffled maximum of 0.060.

**Splits (nested grouping)**
- **Outer loop, participants:** each of the 12 participants *p* is held out once.
- **Video folds:** for each seed, the 40 videos are permuted with `numpy.random.default_rng(seed)` and split into 5 folds of 8. The partition is **not** stratified, so no ratings (including the held-out participant's) are used to build it.
  - The same partition is used for every held-out participant and for both tasks within a seed.
  - Every video is therefore tested exactly once per held-out participant, per seed and per task.
- **One fit per (seed, task, p, video fold *k*):**
  - **Training:** EEG segments of participants ≠ *p* on videos ∉ fold *k*.
  - **Test:** EEG segments of participant *p* on videos ∈ fold *k*.
  - The model never sees any segment of a held-out video, from any participant, and never sees any segment of participant *p*.
- **Size:** 12 participants × 5 video folds × 5 seeds × 2 tasks = 600 fits.

**Targets**
- **Task A, personal:** a training example's label is its own participant's trusted rating, `valence > 5.0` (unchanged rule). The held-out participant's own labels are used **only** to score that participant's predictions.
- **Task B, consensus:**
  - **Training example** (participant *q* ≠ *p*, video *v*): the majority `valence > 5.0` vote of the 10 development participants other than *q* and *p*. The outer held-out participant *p*'s ratings are never used for training labels.
  - **Test target** (participant *p*, video *v* in the held-out fold): the majority vote of the other 11 participants. With 11 voters there are no ties.
  - The test videos are never in training, so their ratings by other participants enter only as test targets, never as training labels.
- **Tie rule (declared now):**
  - With 10 voters, a training consensus is tied when exactly 5 vote high.
  - **Tied (*q*, *v*) training pairs are excluded from training in both tasks.** Both tasks then train on identical EEG examples, and only the labels differ.
  - Feasibility count: 228 of 5,280 participant-video pairs over all outer folds (4.3%), or 12–24 of 440 per held-out participant. Exact counts per fit are recorded and reported.
  - No test target is excluded.

**Model and prediction**
- The registered Random Forest configuration, unchanged (`candidate_models(seed)["random_forest"]`, `n_jobs=-1`). Its scaler is fitted inside the pipeline on training rows only.
- No hyperparameter is tuned.
- Seeds 42–46 set both the video partition and the Random Forest `random_state`.
- **Trial prediction:** the mean of segment probabilities within a (participant, video) trial, then argmax.

**Metrics** (per held-out participant, over its 40 test trials, per seed, then averaged over seeds)
- Trial balanced accuracy and macro-F1 (primary).
- ROC-AUC where both classes are present.
- Predicted-high share, and the true class balance of both targets.

**Statistics** (the participant is the unit of analysis)
- Group mean ± SD across the 12 participants, with a 95% t-CI and a participant bootstrap CI.
- **Three pre-declared tests on trial balanced accuracy,** each an exact sign-flip permutation test over participants (4,096 sign patterns), with Holm correction across the three:
  1. Task A vs 0.5;
  2. Task B vs 0.5;
  3. Task A − Task B (paired).
- Wilcoxon signed-rank p-values are reported alongside.
- **Collapse diagnostics:** the number of participants whose predicted-high share is < 0.05 or > 0.95, and how closely predicted shares track true shares.

**Verification before reporting**
- The code asserts, for every fit, that no training row belongs to *p* or to a held-out video.
- It also asserts that every (*p*, video) test trial is predicted exactly once per seed and task.
- The video partitions, tie counts and configuration are saved.
- One complete participant (all 5 seeds, both tasks) will be recomputed with independently written code.

**Runtime estimate:** about 10 s per fit on battery (about 5,155 training segments), so about 1.7 h on battery and about 40–50 min plugged in. Results are saved after each participant, so the run can resume.

**Code:** `backend/scripts/cross_participant_lvo.py`. **Results:** `outputs/experiments/cross_participant_lvo_rf_20261011/`; the verified copy goes to `docs/results/exploratory/cross_participant_lvo_rf/`.

*(Results are added below only after verification.)*

## 3. Verification (2026-10-11)

- **Run:** all 600 fits completed with no errors. The main phase took about 80 minutes: about 15 minutes per participant on battery and about 4 minutes plugged in.
- **In-code assertions passed for every fit:** no training row belonged to the held-out participant or to a held-out video, and every (participant, video) test trial was predicted exactly once per seed and task.
- **Saved results re-checked:**
  - all 12 participants and 5 seeds are present;
  - every seed's 5 video folds partition the 40 videos;
  - every (participant, seed, task) has exactly 40 trial predictions;
  - no frozen test participant ID appears anywhere in the results.
- **Independent recomputation of one complete participant (s09):** a separately written script rebuilt the votes, ties, splits and targets with plain loops, refitted all 50 models (5 seeds × 5 folds × 2 tasks) and scored them with scikit-learn.
  - Tied pairs: 18, matching the saved count.
  - Personal and consensus targets: identical to the saved ones.
  - Balanced accuracy, macro-F1 and ROC-AUC: equal to the script's values (|Δ| ≤ 1e-9) for all 10 seed × task runs.
- **Ties:** 228 of 5,280 training (participant, video) pairs were excluded over all outer folds (12–24 per held-out participant), exactly as estimated in the feasibility count. No test target was excluded.
- **Holm adjustment** was checked by hand from the raw p-values.
- **Unit tests:** 4 new tests in `backend/tests/test_cross_participant_lvo.py`.

## 4. Results per participant

Trial-level values per held-out participant, averaged over 5 seeds (balanced accuracy ± SD across seeds):
- **Pred. high:** the share of the 40 trials predicted high.
- **True high:** the true high share of that task's target.
- **Single-label seeds:** seeds in which all 40 trials received the same prediction.
- **Agreement:** how often the participant's own label equals the consensus label.

| Participant | Agreement | Ties excl. | **A** BA | A F1 | A AUC | A pred. / true high | A single-label seeds | **B** BA | B F1 | B AUC | B pred. / true high | B single-label seeds |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| s01 | 0.600 | 12 | 0.488 ± 0.026 | 0.329 | 0.472 | 0.97 / 0.475 | 4 | 0.527 ± 0.080 | 0.524 | 0.509 | 0.44 / 0.475 | 0 |
| s02 | 0.775 | 18 | 0.432 ± 0.076 | 0.375 | 0.361 | 0.27 / 0.550 | 0 | 0.479 ± 0.035 | 0.376 | 0.492 | 0.09 / 0.475 | 1 |
| s03 | 0.825 | 24 | 0.504 ± 0.056 | 0.498 | 0.486 | 0.52 / 0.550 | 0 | 0.475 ± 0.094 | 0.378 | 0.514 | 0.10 / 0.475 | 2 |
| s05 | 0.725 | 12 | 0.504 ± 0.016 | 0.396 | 0.522 | 0.98 / 0.600 | 4 | 0.510 ± 0.060 | 0.405 | 0.482 | 0.90 / 0.475 | 1 |
| s07 | 0.750 | 24 | 0.517 ± 0.023 | 0.445 | 0.611 | 0.99 / 0.700 | 5 | 0.545 ± 0.033 | 0.424 | 0.688 | 0.96 / 0.500 | 2 |
| s08 | 0.775 | 24 | 0.482 ± 0.021 | 0.379 | 0.411 | 0.13 / 0.550 | 0 | 0.426 ± 0.021 | 0.295 | 0.374 | 0.08 / 0.525 | 0 |
| s09 | 0.800 | 18 | 0.510 ± 0.049 | 0.488 | 0.527 | 0.66 / 0.500 | 0 | 0.475 ± 0.056 | 0.404 | 0.479 | 0.84 / 0.500 | 0 |
| s11 | 0.775 | 12 | 0.490 ± 0.090 | 0.467 | 0.511 | 0.40 / 0.600 | 0 | 0.453 ± 0.031 | 0.354 | 0.400 | 0.13 / 0.525 | 1 |
| s15 | 0.925 | 24 | 0.485 ± 0.052 | 0.400 | 0.591 | 0.88 / 0.500 | 0 | 0.530 ± 0.100 | 0.525 | 0.515 | 0.51 / 0.525 | 0 |
| s16 | 0.500 | 18 | 0.535 ± 0.056 | 0.435 | 0.666 | 0.81 / 0.375 | 0 | 0.380 ± 0.054 | 0.308 | 0.330 | 0.19 / 0.525 | 0 |
| s22 | 0.850 | 24 | 0.559 ± 0.077 | 0.553 | 0.582 | 0.44 / 0.450 | 0 | 0.485 ± 0.086 | 0.471 | 0.453 | 0.57 / 0.500 | 0 |
| s25 | 0.775 | 18 | 0.500 ± 0.000 | 0.322 | 0.630 | 1.00 / 0.475 | 5 | 0.500 ± 0.000 | 0.333 | 0.578 | 1.00 / 0.500 | 5 |

Figure: `docs/results/exploratory/cross_participant_lvo_rf/cross_participant_lvo.png`.

## 5. Group-level results (participant = unit, n = 12)

| Statistic | Task A: own rating | Task B: consensus | A − B (paired) |
|---|---|---|---|
| Mean trial balanced accuracy (sample SD) | **0.500** (0.031) | **0.482** (0.047) | +0.018 (0.059) |
| 95% t-CI | [0.481, 0.520] | [0.453, 0.512] | [−0.019, 0.056] |
| 95% participant-bootstrap CI | [0.483, 0.517] | [0.456, 0.506] | [−0.011, 0.052] |
| Range | 0.432–0.559 | 0.380–0.545 | −0.047 to +0.154 |
| Mean macro-F1 | 0.424, CI [0.379, 0.468] | 0.400, CI [0.351, 0.448] | +0.024, CI [−0.040, 0.088] |
| Mean ROC-AUC | 0.531, CI [0.473, 0.588] | 0.485, CI [0.425, 0.544] | +0.046, CI [−0.029, 0.122] |
| Participants with A > B (balanced accuracy) | | | 6 of 12 |

**Pre-declared tests on balanced accuracy** (exact sign-flip over participants, 4,096 patterns, two-sided; Holm across the three):

| Test | p (raw) | p (Holm) | Wilcoxon p |
|---|---|---|---|
| 1. Task A vs 0.5 | 0.956 | 0.956 | 0.966 |
| 2. Task B vs 0.5 | 0.218 | 0.653 | 0.365 |
| 3. Task A − Task B | 0.321 | 0.653 | 0.465 |

**Collapse diagnostics:**

| | Task A | Task B |
|---|---|---|
| Participants with seed-averaged predicted-high share < 0.05 or > 0.95 | 4 (s01, s05, s07, s25) | 2 (s07, s25) |
| Participants with at least one single-label seed | 4 | 6 |
| Range of predicted-high share | 0.13–1.00 | 0.08–1.00 |
| Mean \|predicted − true high share\| | 0.30 | 0.31 |

## 6. Interpretation

1. **No detectable transfer to a new person on new videos, for either target.**
   - Task A (own rating) averages 0.500, and Task B (consensus) averages 0.482.
   - Neither differs from chance (Holm p = 0.96 and 0.65), and the CIs are narrow enough to rule out a group-level balanced accuracy above about 0.52.
   - Task A and Task B do not differ (Holm p = 0.65).
2. **This contrasts with the within-person results.**
   - When training and testing on the same person (different videos), both targets were decodable: about 0.62 for own ratings and 0.65 for consensus (§9 of `EXPLORATORY_WITHIN_PERSON_DIAGNOSTIC.md`).
   - So the shared, stimulus-related component found there does **not** transfer through this pipeline to a participant the model has never seen, even for the consensus label that is the same for everyone.
   - The most direct reading is that whatever links EEG features to valence here is **expressed differently in each person's feature space**, so it is usable only after seeing labelled data from that person.
3. **The predictions are dominated by a per-person offset, not by video content.**
   - Predicted-high shares move far from the true shares (mean gap 0.30–0.31). Several participants get nearly all-high or all-low predictions (e.g. s25 is all high for every seed in both tasks; s08 is mostly low).
   - Usually the same direction appears in both tasks. That pattern fits the classifier reacting to the held-out person's overall EEG level rather than to individual trials.
   - This is a hypothesis about the mechanism; the experiment does not test it. It is consistent with the earlier cross-participant results, where RF LOPO accuracy was 0.503 and CNN LOPO balanced accuracy 0.495.
4. **Some ROC-AUC values above 0.6 alongside chance balanced accuracy** (Task A: s07, s16, s25; Task B: s07) mean the trial *ranking* is sometimes informative within a person while the threshold is wrong for that person.
   - Across the group, AUC is not distinguishable from 0.5 (Task A 0.531, CI [0.473, 0.588]).
   - No calibration or per-person thresholding was applied, as required.
5. **Consensus prediction is a separate question.** Task B tests whether EEG can predict how *other people* tend to rate a video. Its result here (chance) says nothing more about a person's own experience than Task A does, and neither result should be read as predicting a new person's emotional state.

## 7. Limitations

- **Small sample:** 12 participants × 40 videos.
  - The CIs exclude moderate effects but not small ones (up to about +0.02 for Task A and +0.01 for Task B).
  - Individual participants' estimates vary by ±0.02–0.10 across seeds (video partitions).
- **Training-set size:** each fit trains on about 11 × 32 = 352 trials (minus ties), from 11 people. A larger or more diverse training group might transfer differently. This is not tested here.
- **Features and model:** only the registered Random Forest with the existing 440 features and no per-participant normalization. No conclusion is drawn about other models, normalization schemes or the CNN.
- **Consensus noise:** training consensus labels come from 10 voters, with ties removed. Test consensus labels come from 11 voters, and close votes (5–6 of 11) are noisy.
- **Unstratified video folds:** some folds have unbalanced targets for some participants, which affects single-fold scores but not the trial-level metrics over all 40 trials.
- **Collapse:** where predictions are all one class, balanced accuracy is fixed at 0.5 and F1 is low, regardless of any ranking information.
- **Status:** exploratory (12 participants, Protocol v1.1). This is not a final-test result, and it makes no claim about the frozen test group or about generalization to all new people.

## 8. What the results support and what remains uncertain

- **Supported (for this sample, pipeline and model):** the registered Random Forest with the current features does not predict either a new person's own valence category or the video's consensus category on unseen videos better than chance. This is the case even though both are predictable within a person.
- **Uncertain:**
  - why the within-person signal does not transfer: per-person feature offsets, differences in the EEG-valence mapping, or too few training people;
  - whether per-participant feature normalization or other models would change this;
  - whether small effects (≤ 0.02) exist.

  These are open questions for future, separately approved work. They were not explored here.

## 9. Reproduce

```powershell
cd backend
..\.venv\Scripts\python -m scripts.cross_participant_lvo --phase main     # ~45 min plugged in, ~2.5 h on battery; resumable
..\.venv\Scripts\python -m scripts.cross_participant_lvo --phase report
..\.venv\Scripts\python -m pytest tests/test_cross_participant_lvo.py
```

**Results:**
- Live run: `outputs/experiments/cross_participant_lvo_rf_20261011/`.
- Verified copy: `docs/results/exploratory/cross_participant_lvo_rf/` (`summary.json`, `main.json`, `cross_participant_lvo.png`).

> **Follow-up (2026-10-12):** participant-specific feature normalization (unsupervised calibration) was tested on the same folds in `docs/EXPLORATORY_PARTICIPANT_NORMALIZATION.md`. It removed the prediction collapse described in §6 point 3, but the primary balanced-accuracy gain was not significant (+0.034, p = 0.098) and ROC-AUC did not change.
