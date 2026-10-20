# Participant-Specific Feature Normalization (Cross-Participant, Leave-Video-Out)

**Status: exploratory (12 participants, Protocol v1.1).** This is not a final-test result and not proof of generalization to all new people (Decision C). It follows `docs/EXPLORATORY_CROSS_PARTICIPANT_LEAVE_VIDEO_OUT.md` (the "LVO experiment").

> **Condition B is an unsupervised-calibration (transductive) condition, not strict zero-shot generalization.** It uses unlabeled EEG from the held-out participant, recorded on videos other than the test videos, to set that participant's feature scale. A real deployment would need a comparable unlabeled calibration recording from each new user.

## 1. Research question

Can unsupervised, participant-specific standardization of the existing 440 features improve the registered Random Forest's ability to predict an **unseen participant's** valence category on **unseen videos**?

The LVO experiment suggested that predictions are dominated by a per-person offset: predicted-high shares far from the true shares, and near single-class predictions for several people. Removing each person's feature location and scale is the most direct test of that hypothesis.

## 2. Design (declared before running, 2026-10-11)

**Data:** the same 12 trusted DEAP participants and feature cache as the LVO experiment (fingerprint `3c5359083bc90ae5`; 440 features per 4 s segment; 7,046 accepted segments). No third-party data is used, and the 8 frozen test participants are never accessed.

**Conditions**
- **A (no participant normalization):** the existing pipeline. These are **exactly the verified LVO results** (`docs/results/exploratory/cross_participant_lvo_rf/main.json`), reused without refitting.
  - A was run with the same code, folds, seeds, training rows and model settings as B below; the only difference is the input features.
  - Reusing A avoids 300 redundant fits.
  - The independent recomputation (below) refits A for one participant, as a reproduction check.
- **B (participant-specific standardization):** identical to A, except that before fitting, every participant's features are z-scored with that participant's own statistics:
  - computed separately for each seed and video fold *k*;
  - each of the 440 features: `(x − mean) / SD`, using the population SD (ddof 0); an SD of 0 is replaced by 1.

**EEG available for normalization statistics (fold *k*):**
- For **every** participant *i* (the 11 training participants **and** the held-out participant *p*): all of *i*'s accepted segments on videos **not** in fold *k*. That is 32 videos, about 5,640 segments in total and 427–480 per participant (feasibility check, seed 42, fold 0).
- **Labels are never used:** the statistics use no valence ratings, personal labels or consensus labels. Training participants' segments on tied-consensus videos (excluded from model training) are still included in their statistics, because exclusion depends on labels and the statistics must not.
- **No held-out video:** EEG from the test videos (fold *k*), from **any** participant, never enters any statistic. Each participant is normalized on the same 32 videos as every other participant in that fold.
- **What the held-out participant contributes:** only unlabeled EEG on the 32 non-test videos. Their labels never enter training, preprocessing, normalization or any choice. This is the transductive element in the box above.
- Segment rejection (MAD clipping, as in the cache) is label-free and unchanged.

**Everything else is identical to the LVO experiment:**
- outer loop over the 12 held-out participants;
- the same seeded video partitions (`default_rng(seed).permutation(40)`, 5 folds of 8, seeds 42–46);
- the same training rows: participants ≠ *p*, videos ∉ *k*, with tied training-consensus pairs excluded in both tasks (228 pairs over all outer folds);
- the same targets and tie rule;
- the registered Random Forest unchanged, including its internal scaler fitted on training rows;
- no tuning;
- trial prediction is the mean of segment probabilities, then argmax.

**Tasks**
- **Primary:** personal valence (own rating, `valence > 5.0`).
- **Secondary control:** video consensus (as in the LVO experiment). Consensus prediction is a separate research question and is not equivalent to predicting a person's own experience.

**Size:** condition B needs 12 participants × 5 folds × 5 seeds × 2 tasks = 600 fits. Condition A is reused.

**Metrics** (per participant, over its 40 test trials, per seed, then averaged over seeds; the participant is the unit of analysis):
- trial balanced accuracy (primary), macro-F1, ROC-AUC;
- predicted-high share versus true share, and the number of single-label seeds.

**Statistics**
- **Primary test (one test, α = 0.05):** personal task, paired B − A trial balanced accuracy over 12 participants, using an exact two-sided sign-flip test (4,096 patterns).
- **Secondary tests (Holm across the three):**
  - personal B vs 0.5;
  - consensus B − A;
  - consensus B vs 0.5.
- **Reported for all:** mean ± SD, 95% t-CI, participant-bootstrap CI (10,000), Wilcoxon p, and the number of participants improved. Macro-F1 and AUC differences are descriptive (CIs only).
- **Collapse diagnostics** for A and B: participants with mean predicted-high share < 0.05 or > 0.95, single-label seeds, and mean |predicted − true share|.

**Verification**
- For every fit, the code asserts:
  - no held-out participant or held-out-video rows in training;
  - every test trial predicted exactly once;
  - normalization statistics drawn only from non-fold videos;
  - the training rows equal the LVO experiment's rows.
- The saved fold partitions must equal the LVO experiment's.
- One complete participant will be recomputed independently, for both conditions and all seeds, with separately written code.

**Feasibility check (no outcome metrics computed):**
- Normalization takes 0.1 s per fold.
- No feature has zero SD for any participant, and all normalized values are finite.
- Per-participant means on the statistics rows are ≤ 3e-5 after normalization.
- One fit has 4,988 training rows and takes 10.7 s on battery.

**Runtime estimate:** 600 fits, so about 1.8 h on battery and about 45 min plugged in. Normalization adds about 0.1 s per fold. The run saves after each participant and can resume.

**Code:** `backend/scripts/participant_normalization.py`. **Results:** `outputs/experiments/participant_normalization_rf_20261011/`; the verified copy goes to `docs/results/exploratory/participant_normalization_rf/`.

*(Results are added below only after verification.)*

## 3. Verification (2026-10-12)

- **Run:**
  - All 600 condition-B fits completed with no errors.
  - The run stopped once overnight, after 6 participants, when the laptop slept. There was no error, and it resumed from the saved file without recomputing finished participants.
- **In-code assertions passed for every fit:**
  - no held-out participant or held-out-video rows in training;
  - normalization statistics only from non-test-fold videos;
  - training-row counts equal to the LVO experiment's;
  - targets and video folds equal to the LVO experiment's;
  - every test trial predicted exactly once.
- **Saved results re-checked:**
  - all 12 participants and 5 seeds are present;
  - every partition covers the 40 videos;
  - every (participant, seed, task) has 40 trial predictions;
  - the held-out participant's normalization used 424–480 unlabeled segments per fold;
  - no frozen test participant ID appears.
- **Independent recomputation of one complete participant (s05):** separately written code refitted all 100 models (2 conditions × 5 seeds × 5 folds × 2 tasks). It used the trusted loader's ratings, plain loops for votes, splits and per-participant normalization, and scikit-learn metrics.
  - Tied pairs (12) and targets match.
  - Balanced accuracy, macro-F1 and ROC-AUC equal the saved values (|Δ| ≤ 1e-9) for all 20 condition × seed × task runs.
  - Refitting condition A from scratch reproduced the LVO results exactly, which confirms that reusing them as condition A is valid.
- **Holm adjustment** was checked by hand.
- **Unit tests:** 3 new tests in `backend/tests/test_participant_normalization.py`.

## 4. Results per participant: primary task (own rating)

Trial-level values, mean over 5 seeds. BA ± SD is across seeds. "Pred. high" is the share of trials predicted high, and "Single" is the number of seeds in which all 40 trials got one prediction.

| Participant | True high | **A** BA | A F1 | A AUC | A pred. high | A single | **B** BA | B F1 | B AUC | B pred. high | B single | **B − A** BA |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| s01 | 0.475 | 0.488 ± 0.026 | 0.329 | 0.472 | 0.97 | 4 | 0.547 ± 0.056 | 0.514 | 0.549 | 0.73 | 0 | +0.059 |
| s02 | 0.550 | 0.432 ± 0.076 | 0.375 | 0.361 | 0.27 | 0 | 0.515 ± 0.023 | 0.494 | 0.568 | 0.75 | 0 | +0.083 |
| s03 | 0.550 | 0.504 ± 0.056 | 0.498 | 0.486 | 0.52 | 0 | 0.530 ± 0.037 | 0.508 | 0.502 | 0.76 | 0 | +0.026 |
| s05 | 0.600 | 0.504 ± 0.016 | 0.396 | 0.522 | 0.98 | 4 | 0.571 ± 0.064 | 0.561 | 0.646 | 0.74 | 0 | +0.067 |
| s07 | 0.700 | 0.517 ± 0.023 | 0.445 | 0.611 | 0.99 | 5 | 0.548 ± 0.034 | 0.530 | 0.573 | 0.55 | 0 | +0.031 |
| s08 | 0.550 | 0.482 ± 0.021 | 0.379 | 0.411 | 0.13 | 0 | 0.529 ± 0.075 | 0.528 | 0.545 | 0.61 | 0 | +0.047 |
| s09 | 0.500 | 0.510 ± 0.049 | 0.488 | 0.527 | 0.66 | 0 | 0.660 ± 0.063 | 0.654 | 0.758 | 0.61 | 0 | +0.150 |
| s11 | 0.600 | 0.490 ± 0.090 | 0.467 | 0.511 | 0.40 | 0 | 0.525 ± 0.038 | 0.524 | 0.463 | 0.58 | 0 | +0.035 |
| s15 | 0.500 | 0.485 ± 0.052 | 0.400 | 0.591 | 0.88 | 0 | 0.555 ± 0.072 | 0.539 | 0.659 | 0.69 | 0 | +0.070 |
| s16 | 0.375 | 0.535 ± 0.056 | 0.435 | 0.666 | 0.81 | 0 | 0.545 ± 0.050 | 0.523 | 0.616 | 0.57 | 0 | +0.010 |
| s22 | 0.450 | 0.559 ± 0.077 | 0.553 | 0.582 | 0.44 | 0 | 0.483 ± 0.026 | 0.473 | 0.444 | 0.59 | 0 | **−0.076** |
| s25 | 0.475 | 0.500 ± 0.000 | 0.322 | 0.630 | 1.00 | 5 | 0.408 ± 0.046 | 0.356 | 0.250 | 0.77 | 0 | **−0.092** |

## 5. Results per participant: secondary control (video consensus)

| Participant | True high | A BA | A AUC | A pred. high | B BA | B AUC | B pred. high | B − A BA |
|---|---|---|---|---|---|---|---|---|
| s01 | 0.475 | 0.527 | 0.509 | 0.44 | 0.418 | 0.415 | 0.36 | −0.109 |
| s02 | 0.475 | 0.479 | 0.492 | 0.09 | 0.515 | 0.535 | 0.50 | +0.036 |
| s03 | 0.475 | 0.475 | 0.514 | 0.10 | 0.471 | 0.428 | 0.52 | −0.004 |
| s05 | 0.475 | 0.510 | 0.482 | 0.90 | 0.596 | 0.620 | 0.52 | +0.086 |
| s07 | 0.500 | 0.545 | 0.688 | 0.96 | 0.555 | 0.603 | 0.62 | +0.010 |
| s08 | 0.525 | 0.426 | 0.374 | 0.08 | 0.513 | 0.498 | 0.54 | +0.087 |
| s09 | 0.500 | 0.475 | 0.479 | 0.84 | 0.635 | 0.691 | 0.50 | +0.160 |
| s11 | 0.525 | 0.453 | 0.400 | 0.13 | 0.487 | 0.473 | 0.66 | +0.034 |
| s15 | 0.525 | 0.530 | 0.515 | 0.51 | 0.493 | 0.432 | 0.53 | −0.037 |
| s16 | 0.525 | 0.380 | 0.330 | 0.19 | 0.421 | 0.362 | 0.49 | +0.041 |
| s22 | 0.500 | 0.485 | 0.453 | 0.57 | 0.405 | 0.334 | 0.56 | −0.080 |
| s25 | 0.500 | 0.500 | 0.578 | 1.00 | 0.515 | 0.447 | 0.66 | +0.015 |

Figure: `docs/results/exploratory/participant_normalization_rf/participant_normalization.png`.

## 6. Group-level results (participant = unit, n = 12)

| Personal task (primary) | A | B | B − A |
|---|---|---|---|
| Balanced accuracy, mean (sample SD) | 0.500 (0.031) | **0.535** (0.058) | **+0.034** (0.066) |
| 95% t-CI | [0.481, 0.520] | [0.498, 0.572] | [−0.008, 0.076] |
| 95% participant-bootstrap CI | [0.483, 0.517] | [0.503, 0.566] | [−0.003, 0.067] |
| Participants improved / worse | | | 10 / 2 |
| Macro-F1, mean [t-CI] | 0.424 [0.379, 0.468] | 0.517 [0.474, 0.560] | +0.093 [0.044, 0.143]; 11 / 1 |
| ROC-AUC, mean [t-CI] | 0.531 [0.473, 0.588] | 0.548 [0.466, 0.629] | +0.017 [−0.089, 0.123]; 7 / 5 |

| Consensus task (secondary) | A | B | B − A |
|---|---|---|---|
| Balanced accuracy, mean [t-CI] | 0.482 [0.453, 0.512] | 0.502 [0.457, 0.547] | +0.020 [−0.027, 0.067]; 8 / 4 |
| Macro-F1 | 0.400 | 0.498 | +0.098 [0.026, 0.169] |
| ROC-AUC | 0.485 | 0.486 | +0.002 [−0.072, 0.076] |

**Pre-declared tests on balanced accuracy** (exact sign-flip over participants, 4,096 patterns, two-sided):

| Test | p (raw) | p (Holm) | Wilcoxon p |
|---|---|---|---|
| **Primary:** personal B − A | **0.098** | (single test) | 0.151 |
| Secondary: personal B vs 0.5 | 0.057 | 0.171 | 0.042 |
| Secondary: consensus B − A | 0.373 | 0.746 | 0.339 |
| Secondary: consensus B vs 0.5 | 0.927 | 0.927 | 0.865 |

The secondary test "personal B vs 0.5" gives Wilcoxon p = 0.042 uncorrected, but its pre-declared test is the sign-flip test. That test is not significant before (0.057) or after (0.171) Holm correction.

**Collapse diagnostics:**

| | Personal A | Personal B | Consensus A | Consensus B |
|---|---|---|---|---|
| Participants with mean predicted-high share < 0.05 or > 0.95 | 4 | **0** | 2 | **0** |
| Participants with any single-label seed (total seeds) | 4 (18) | **0 (0)** | 6 (12) | **0 (0)** |
| Mean \|predicted − true high share\| | 0.30 | 0.16 | 0.31 | 0.06 |
| Mean predicted-high share (true 0.527 for personal, 0.502 for consensus) | — | 0.66 | — | 0.54 |

## 7. Interpretation

1. **The primary test is not significant.**
   - Participant normalization raised personal-task balanced accuracy by +0.034 on average and improved 10 of 12 participants.
   - But the pre-declared paired test gives p = 0.098, and the CI includes 0 ([−0.008, 0.076]).
   - **This experiment does not show that participant normalization improves generalization to unseen people.**
2. **Normalization removed prediction collapse.** No participant or seed is single-class any more, and the gap between predicted and true class shares roughly halves (personal task) or falls to a fifth (consensus task). This supports the LVO experiment's hypothesis that much of the cross-person failure came from per-person feature offsets.
3. **Most of the gain looks like fixing the decision threshold, not better discrimination.**
   - Macro-F1 rises clearly (+0.09, CI excluding 0). Macro-F1 heavily penalizes single-class predictions, so removing collapse raises it almost automatically.
   - ROC-AUC, which measures ranking and does not depend on the threshold, barely changes: +0.017 for the personal task and +0.002 for consensus, both with wide CIs around 0.
   - So the model's ordering of trials from more to less positive did not improve on average. Mainly its operating point moved closer to each person's class balance.
   - Personal-task predictions still lean high: mean predicted-high share 0.66 against a true 0.53.
4. **Effects are uneven.**
   - s09 improved strongly (+0.15 BA, AUC 0.53 → 0.76). s22 and s25 got worse (−0.08 and −0.09); s25's AUC fell to 0.25, so its ranking was inverted.
   - With 12 participants, these individual swings cannot be explained or generalized.
5. **The consensus control does not change.**
   - Consensus balanced accuracy stays at chance with B (0.502; B − A Holm p = 0.75).
   - So normalization does not reveal the shared stimulus-related component seen within a person. The small personal-task gain is therefore not explained by better access to shared video responses.
6. **Condition B is unsupervised calibration, not zero-shot.** Even a modest benefit would require each new user to provide unlabeled EEG from other stimuli before predictions.

## 8. Limitations

- **Small sample:** 12 participants. The primary test had limited power: a true gain of about +0.03 would often be missed, and the observed estimate is compatible with anything from no gain to about +0.08.
- **Exploratory:** under Protocol v1.1 and Decision C, a positive result here would still not be a formal claim. The frozen test group was not used.
- **One normalization scheme:** per-participant z-scoring from 32 videos of unlabeled EEG.
  - Other choices (robust statistics, fewer calibration trials, normalization from rest or baseline segments) were not tried. They were not pre-declared, and trying several would need multiple-comparison control.
  - The number of calibration trials a real user would need is unknown.
- **Shared-video calibration:** the held-out participant's normalization used EEG from the same 32 videos the training participants watched. A real user's calibration recording would differ, which could change the benefit.
- **Model scope:** only the registered Random Forest with no tuning. No conclusion is drawn about the CNN or other models.
- **Reused condition A:** it comes from the LVO run. It was verified to reproduce exactly for s05 but was not refitted for all participants.

## 9. What the results support and what remains uncertain

- **Supported (for this sample, pipeline and model):**
  - participant-specific standardization removes the single-class collapse and makes predicted class shares much more realistic;
  - it does **not** reliably improve balanced accuracy (primary p = 0.098) and does not improve trial ranking (AUC);
  - consensus prediction stays at chance.
- **Uncertain:** whether a small real gain (≤ 0.08) exists, and why some people get worse (s22, s25).

## 10. Recommended next step (for approval; not started)

Normalization fixes the operating point but adds no ranking information. Further tweaks to the Random Forest are therefore unlikely to make it generalize to a new person. Of the options available, I recommend the following:

- **Re-evaluate the existing CNN under the same leave-video-out design, with condition B normalization.**
  - Compare it with the Random Forest results here on identical folds, as an exploratory, paired, pre-declared comparison.
  - Reason: the CNN works on raw multichannel time series rather than fixed band-power features, so it is the one already-built model that could carry different information.
  - It uses no new architecture. Runtime would need a feasibility check first, because CNN fits are much slower than Random Forest fits.
- **Alternative, if compute is a concern:** stop the cross-person modelling line and write up the current evidence. Within-person decoding works; cross-person decoding is at chance; normalization fixes collapse but not discrimination. Then focus the project on a calibrated, within-person framing.

## 11. Reproduce

```powershell
cd backend
..\.venv\Scripts\python -m scripts.participant_normalization --phase main     # ~45 min plugged in; resumable
..\.venv\Scripts\python -m scripts.participant_normalization --phase report
..\.venv\Scripts\python -m pytest tests/test_participant_normalization.py
```

**Results:**
- Live run: `outputs/experiments/participant_normalization_rf_20261011/`.
- Verified copy: `docs/results/exploratory/participant_normalization_rf/` (`summary.json`, `main.json`, `participant_normalization.png`, `run_log.txt`).
