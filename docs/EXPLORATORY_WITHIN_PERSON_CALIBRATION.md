# Supervised Within-Person Calibration on Held-Out Videos

**Status: exploratory (12 participants, Protocol v1.1).** This is not a final-test result, and it makes no claim about people outside this sample (Decision C).

> **Supervised within-person calibration is not zero-shot prediction.** Each model here is trained on **labelled** trials of the same person it is tested on. A real user would first have to watch and rate about 32 videos. These results say nothing about predicting a new, uncalibrated person. That question was tested separately and was at chance (`EXPLORATORY_CROSS_PARTICIPANT_LEAVE_VIDEO_OUT.md`).
>
> **Calibration evidence is not evidence of individual emotional information.** Above-chance within-person prediction can arise from EEG responses that are shared across people and related to the stimulus. Only the disagreement-trial test (S3 below) bears on individual-specific information.

## 1. Research question

Can a model trained on one participant's labelled EEG trials predict **the same participant's** valence ratings for videos held out from training?

Context:
- An earlier diagnostic answered a similar question with label-stratified folds (balanced accuracy 0.625; `EXPLORATORY_WITHIN_PERSON_DIAGNOSTIC.md` §1–9).
- This experiment repeats it on **label-free video folds that are identical to the cross-person experiments**. All conditions can therefore be compared on the same test trials.
- It adds a pre-declared stimulus-confound test, a calibration learning curve and per-participant permutation tests.

## 2. Design (declared before running, 2026-10-12; approved by the project lead)

**Data:** the 12 trusted DEAP participants and the existing feature cache (fingerprint `3c5359083bc90ae5`; 440 features per 4 s segment). No third-party data is used, and the 8 frozen test participants are never accessed. Trial index *t* is video *t* for every participant (verified in the stimulus-confound control).

**Folds:**
- For each seed *s* ∈ {42, 43, 44, 45, 46}: `numpy.random.default_rng(s).permutation(40)` → fold *k* = `order[k::5]`, giving 5 folds of 8 videos.
- These are **identical** to the saved folds of the cross-person LVO and normalization experiments (checked in code).
- Folds use no labels, and all participants share them.
- Every segment of a trial is in its video's fold.

**One fit = (participant *p*, seed *s*, fold *k*, task):**
- **Training:** *p*'s segments on the 32 videos outside fold *k*.
- **Test:** *p*'s segments on the 8 videos in fold *k*.
- **Model:** the registered Random Forest (`candidate_models(s)["random_forest"]`, `random_state = s`), unchanged and untuned. Its scaler is fitted on the training rows only, and there is no feature selection.
- **Leakage:** feature extraction is per segment and label-free. Held-out videos therefore never enter training, scaling, selection or hyperparameter choice.
- **Trial prediction:** the mean of segment probabilities. "Predicted high" is `p_high ≥ p_low`, matching the project's argmax tie rule (ties go to the first class, `high_valence`).
- **Coverage:** each of the 40 trials is predicted exactly once per (participant, seed, task).

**Tasks (identical folds and seeds):**
- **P, primary: own rating.** Label = *p*'s own `valence > 5.0`.
- **C, stimulus control: consensus.**
  - Label for video *v* = the majority `valence > 5.0` vote of the **other 11** participants. With 11 voters there are no ties.
  - It is used as *p*'s training labels on training videos and as the target on test videos.
  - *p*'s own rating is never used in task C.

**Calibration learning curve (personal task; descriptive only):**
- **Nested subsets:** in each (seed *s*, fold *k*), the 32 training videos are ordered by `numpy.random.default_rng([s, k]).permutation(sorted(training videos))`. The first 8, 16, 24 and 32 videos form the nested subsets.
- **Label-free:** the order uses no labels and is shared by all participants. Size 32 is the core P fit.
- **Single-class subsets:** if a subset's **training** labels contain only one class, no model is fitted. That (participant, seed, size) is reported as invalid and excluded from that size's summary, and the count of invalid cases is reported. Test labels are never consulted for this.

**Secondary analysis S1 (Protocol §7, Decision F; personal task; descriptive only):**
- Trials with 4.5 ≤ valence ≤ 5.5 (the participant's own rating) are removed from **both** training and evaluation in every fold. Metrics are computed over the remaining trials.
- The same single-class rule applies.
- *Change from the approved proposal:* the proposal said "excluded from scoring only". The protocol requires exclusion from training too, so the protocol rule is used.

**Metrics** (per participant, over its 40 test trials, per seed, then averaged over seeds; the participant is the unit of analysis):
- trial balanced accuracy (primary), macro-F1, ROC-AUC (where both classes are present);
- predicted-high share versus true share, single-label seeds, and SD across seeds.

**Comparison conditions** (already saved; same test trials, folds and seeds; no refit):
- cross-person condition A (zero-shot): `docs/results/exploratory/cross_participant_lvo_rf/main.json`;
- cross-person condition B (unsupervised normalization): `docs/results/exploratory/participant_normalization_rf/main.json`.

**Statistics** (n = 12 participants):
- **Primary (one test):** P balanced accuracy vs 0.5, exact two-sided sign-flip test over all 4,096 sign patterns.
- **Secondary (Holm across the three):**
  - **S1-cmp.** P − cross-person A balanced accuracy, paired over identical trials: the value of supervised calibration over zero-shot.
  - **S2.** P − C balanced accuracy: is the own rating predicted better than the consensus?
  - **S3. Disagreement test.** For each participant, take the trials where the own label differs from the consensus. Compute the share of those trials where P's **seed-averaged** prediction (mean `p_high` over 5 seeds) matches the own label. Test the shares against 0.5 with an exact sign-flip test over participants. **This is the only test bearing on individual-specific information.**
- **Descriptive:**
  - 95% t-CI and participant-bootstrap CI (10,000 resamples, seed 0) for every mean, with Wilcoxon p-values alongside;
  - P − cross-person B;
  - task C's model on disagreement trials;
  - the learning curve and S1, as above;
  - collapse diagnostics.

**Per-participant permutation test (secondary):**
- Seed 42 folds and RF `random_state = 42`, fixed. Folds do not depend on labels, so they stay identical under permutation.
- **Permutation *i*** (*i* = 0…499) of participant *p* (index in the 12-participant list): the 40 own-rating video labels are permuted with `numpy.random.default_rng([42, p_index, i]).permutation(40)`, and the full 5-fold cross-validation is rerun. Segment labels follow their video.
- **Statistic:** trial balanced accuracy.
- **Observed value:** P's seed-42 balanced accuracy, the same configuration as the null.
- **p-value:** one-sided, (1 + #{null ≥ observed}) / 501.
- **Correction:** Benjamini–Hochberg across the 12 participants.
- **Resumable:** progress is saved every 25 permutations per participant, and completed permutations are never recomputed.

**Interpretation rules (fixed now):**
- **Primary not significant:** within-person calibration with this pipeline is not supported on these folds.
- **Primary significant, but S2 shows no own-rating advantage and S3 is not significantly above 0.5:** calibration works, but the signal is consistent with shared, stimulus-related EEG responses. It is not reported as decoding of the individual's own experience.
- **S3 significantly above 0.5 after Holm:** evidence of individual-specific valence information. It is still exploratory with 12 participants.

**Verification:**
- **In code, every fit:** no test-video segment is in training; folds equal the saved cross-person folds; personal labels equal the trusted ratings rule; consensus labels use only the other 11 participants; every trial is predicted once.
- **After the run:** one complete participant (both tasks, all seeds, learning curve, S1 and all 500 permutations) is recomputed with independently written code.
- **Unit tests** cover fold subsets, permutation reproducibility and the single-class rule.

**Size:**
- Main phase: 12 participants × 5 seeds × 5 folds × (P + C + learning-curve sizes 8/16/24 + S1) = 12 × 25 × 6 = **1,800 fits** of about 90–470 segments each.
- Permutation phase: 12 participants × 500 permutations × 5 folds = **30,000 fits**.

**Code:** `backend/scripts/within_person_calibration.py`. **Results:** `outputs/experiments/within_person_calibration_rf_20261012/`; the verified copy goes to `docs/results/exploratory/within_person_calibration_rf/`.

## 3. Feasibility check (before any outcome metric; training labels only)

- **Learning curve, single-class training subsets:**
  - 4 of 300 (participant, seed, fold) subsets at size 8 contain only one class, so they will be reported as invalid.
  - None at sizes 16 or 24 (and none at 32, where the minimum class is 12 of 40).
- **S1:** 90 borderline trials across the 12 participants (2–13 each). No S1 training fold is single-class, and no test fold is entirely borderline.
- **Timing** (on battery): one fit takes 0.57 s with 32 training videos and 0.49 s with 8. At this data size the Random Forest's fixed per-fit overhead dominates.
- **Runtime estimate:** main phase 1,800 fits ≈ 15–20 min; permutation phase 30,000 fits ≈ 4–5 h on battery, less when plugged in. Both phases resume after interruption.

*(Results are added below only after verification.)*

## 4. Run and verification (2026-10-12)

**Run**
- **Main phase:** 1,800 fits, completed with no errors. It started at 11:01; the laptop slept until about 13:51, after which it took about 33 minutes.
- **Permutation phase:** 30,000 fits, 14:25–23:49, with no errors. Windows standby paused it several times (about 16:00–18:29, 20:37–21:17, 22:39–22:46). The process resumed each time without losing work.
- **Outcome metrics** were first computed in the report phase, after both phases had finished. The run log contains only progress lines.

**In-code assertions passed for every fit:**
- no test-video segment in training;
- folds equal to the saved cross-person folds;
- personal labels equal to the trusted ratings rule;
- consensus labels from the other 11 participants only;
- every trial predicted exactly once per seed and task;
- learning-curve subsets nested, and the 32-video subset equal to the core fit.

**Independent recomputation of one complete participant (s01)**, with separately written code (trusted loader ratings, plain loops, scikit-learn metrics):
- targets and the 2 borderline trials match;
- all 30 metric checks match to within 1e-9 (5 seeds × own rating, consensus, learning-curve sizes 8/16/24, and S1);
- all **500 permutations** were recomputed with **0 mismatches**.

**Statistics re-checked from the raw saved files:**
- all 12 permutation p-values and BH q-values;
- the disagreement shares;
- the primary mean and its exact sign-flip p.

Holm adjustments were checked by hand.

**Integrity:** no frozen test participant ID appears in any result file.

**Learning-curve validity:** as forecast by the feasibility check, 4 (participant, seed) cases at size 8 were invalid (single-class training subsets): s07 × 2, s11 × 1 and s16 × 1. All other sizes and S1 were valid everywhere.

**Unit tests:** 4 new tests in `backend/tests/test_within_person_calibration.py`.

## 5. Results per participant

Trial balanced accuracy, mean over 5 seeds (± SD across seeds where shown):
- **P / C:** within-person own-rating / consensus tasks.
- **A / B:** cross-person zero-shot / unsupervised-calibration conditions, on the same test trials.
- **Agreement:** the share of videos where the own label equals the consensus.
- **Disagreement:** on trials where they differ, how many P's seed-averaged prediction assigns to the **own** label.
- **Perm p / q:** seed-42 permutation test, one-sided, with 500 permutations, before and after BH correction.

| Participant | True high | Agreement | **P** BA | P AUC | P pred. high | **C** BA | C AUC | A BA | B BA | Disagreement: P follows own | Perm p | BH q |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| s01 | 0.475 | 0.600 | 0.511 ± 0.022 | 0.484 | 0.52 | 0.331 ± 0.051 | 0.336 | 0.488 | 0.547 | 11 / 16 | 0.467 | 0.561 |
| s02 | 0.550 | 0.775 | 0.679 ± 0.020 | 0.656 | 0.63 | 0.640 ± 0.101 | 0.720 | 0.432 | 0.515 | 5 / 9 | 0.034 | 0.080 |
| s03 | 0.550 | 0.825 | 0.712 ± 0.045 | 0.749 | 0.56 | 0.798 ± 0.031 | 0.915 | 0.504 | 0.530 | 3 / 7 | **0.002** | **0.024** |
| s05 | 0.600 | 0.725 | 0.640 ± 0.049 | 0.699 | 0.60 | 0.728 ± 0.057 | 0.776 | 0.504 | 0.571 | 4 / 11 | **0.016** | **0.048** |
| s07 | 0.700 | 0.750 | 0.408 ± 0.047 | 0.537 | 0.75 | 0.645 ± 0.033 | 0.701 | 0.517 | 0.548 | 4 / 10 | 0.697 | 0.697 |
| s08 | 0.550 | 0.775 | 0.597 ± 0.055 | 0.609 | 0.59 | 0.617 ± 0.068 | 0.627 | 0.482 | 0.529 | 3 / 9 | 0.040 | 0.080 |
| s09 | 0.500 | 0.800 | 0.630 ± 0.037 | 0.738 | 0.48 | 0.665 ± 0.045 | 0.762 | 0.510 | 0.660 | 3 / 8 | 0.122 | 0.183 |
| s11 | 0.600 | 0.775 | 0.508 ± 0.088 | 0.478 | 0.58 | 0.467 ± 0.085 | 0.435 | 0.490 | 0.525 | 5 / 9 | 0.389 | 0.519 |
| s15 | 0.500 | 0.925 | 0.755 ± 0.045 | 0.850 | 0.51 | 0.859 ± 0.015 | 0.944 | 0.485 | 0.555 | 0 / 3 | **0.004** | **0.024** |
| s16 | 0.375 | 0.500 | 0.663 ± 0.042 | 0.687 | 0.33 | 0.409 ± 0.030 | 0.365 | 0.535 | 0.545 | 13 / 20 | 0.104 | 0.178 |
| s22 | 0.450 | 0.850 | 0.765 ± 0.030 | 0.750 | 0.42 | 0.755 ± 0.027 | 0.832 | 0.559 | 0.483 | 4 / 6 | **0.014** | **0.048** |
| s25 | 0.475 | 0.775 | 0.549 ± 0.047 | 0.535 | 0.48 | 0.595 ± 0.041 | 0.563 | 0.500 | 0.408 | 5 / 9 | 0.525 | 0.573 |

**Learning curve and S1 per participant** (P balanced accuracy, mean over valid seeds; S1 is scored over the non-borderline trials):

| Participant | 8 videos | 16 | 24 | 32 (= P) | S1 (borderline excluded) | S1 trials |
|---|---|---|---|---|---|---|
| s01 | 0.451 | 0.485 | 0.484 | 0.511 | 0.531 | 38 |
| s02 | 0.606 | 0.661 | 0.697 | 0.679 | 0.723 | 31 |
| s03 | 0.595 | 0.698 | 0.705 | 0.712 | 0.794 | 27 |
| s05 | 0.569 | 0.565 | 0.633 | 0.640 | 0.561 | 34 |
| s07 | 0.534 (3 of 5 seeds valid) | 0.488 | 0.458 | 0.408 | 0.458 | 35 |
| s08 | 0.540 | 0.553 | 0.569 | 0.597 | 0.653 | 29 |
| s09 | 0.585 | 0.645 | 0.620 | 0.630 | 0.593 | 30 |
| s11 | 0.430 (4 of 5 valid) | 0.485 | 0.504 | 0.508 | 0.536 | 32 |
| s15 | 0.675 | 0.735 | 0.730 | 0.755 | 0.772 | 35 |
| s16 | 0.572 (4 of 5 valid) | 0.627 | 0.663 | 0.663 | 0.721 | 35 |
| s22 | 0.646 | 0.707 | 0.724 | 0.765 | 0.714 | 33 |
| s25 | 0.544 | 0.547 | 0.591 | 0.549 | 0.528 | 31 |

Figure: `docs/results/exploratory/within_person_calibration_rf/within_person_calibration.png`.

## 6. Group-level results (participant = unit, n = 12)

| Measure | Mean (sample SD) | 95% t-CI | 95% participant-bootstrap CI | Range |
|---|---|---|---|---|
| **P balanced accuracy** | **0.618** (0.108) | [0.550, 0.687] | [0.557, 0.675] | 0.408–0.765 |
| P macro-F1 | 0.617 (0.110) | [0.547, 0.687] | [0.555, 0.675] | |
| P ROC-AUC | 0.648 (0.119) | [0.572, 0.723] | [0.581, 0.712] | 0.478–0.850 |
| C balanced accuracy | 0.626 (0.158) | [0.526, 0.726] | [0.537, 0.709] | 0.331–0.859 |
| C ROC-AUC | 0.665 (0.204) | [0.535, 0.794] | [0.550, 0.772] | |
| **P − cross-person A** | **+0.118** (0.109) | [0.048, 0.187] | [0.056, 0.175] | 11 of 12 better |
| P − cross-person B (descriptive) | +0.083 (0.121) | [0.007, 0.160] | [0.018, 0.148] | 8 of 12 better |
| **P − C** | **−0.008** (0.130) | [−0.090, 0.075] | [−0.075, 0.065] | 5 of 12 better |
| **Disagreement: share where P follows the own label** | **0.464** (0.192) | [0.342, 0.586] | [0.350, 0.561] | pooled 60 / 117 trials |

**Pre-declared tests** (exact sign-flip over 12 participants, 4,096 patterns, two-sided):

| Test | p (raw) | p (Holm) | Wilcoxon p |
|---|---|---|---|
| **Primary:** P vs 0.5 | **0.0044** | (single test) | 0.0034 |
| S1-cmp: P − cross-person A | 0.0044 | **0.013** | 0.0034 |
| S2: P − C | 0.846 | 1.000 | 0.622 |
| S3: disagreement share vs 0.5 | 0.575 | 1.000 | 0.734 |

**Per-participant permutation tests:** 4 of 12 participants are significant after BH correction (q < 0.05): s03, s05, s15 and s22.

**Learning curve** (P balanced accuracy; descriptive, no tests):

| Calibration videos | 8 | 16 | 24 | 32 |
|---|---|---|---|---|
| Mean [95% t-CI] | 0.579 [0.529, 0.629]* | 0.600 [0.542, 0.657] | 0.615 [0.555, 0.675] | 0.618 [0.550, 0.687] |
| Mean ROC-AUC | 0.606* | 0.626 | 0.637 | 0.648 |
| Invalid (participant, seed) cases | 4 | 0 | 0 | 0 |

\*Size 8 covers only the 9 participants with all 5 seeds valid (s07, s11 and s16 are excluded). It is therefore not strictly comparable with the other sizes.

**S1, borderline excluded** (descriptive): balanced accuracy 0.632 [0.561, 0.703] and AUC 0.663 over 12 participants, compared with 0.618 and 0.648 for the primary analysis.

**Collapse:** there are no single-label seeds in either task. The mean |predicted − true high share| is 0.03 for P and 0.04 for C (cross-person condition A: 0.30).

## 7. Interpretation (by the rules fixed in §2)

1. **Supervised within-person calibration works with this pipeline, on held-out videos.**
   - Own-rating balanced accuracy is 0.618, above chance (exact p = 0.004).
   - It is 0.118 higher than zero-shot cross-person prediction on the **same test trials** (Holm p = 0.013; 11 of 12 participants better).
   - Predictions are well calibrated in class share: no collapse, unlike the cross-person models.
   - 4 of 12 participants are individually significant after BH correction.
   - This replicates the earlier label-stratified diagnostic (0.625) on label-free folds.
2. **The pre-declared rule assigns this outcome to "calibration works, but the signal is consistent with shared, stimulus-related EEG responses".**
   - The consensus of other people is predicted just as well as the person's own rating (P − C = −0.008, Holm p = 1.0; consensus is higher for 7 of 12).
   - On trials where the person disagrees with the consensus, the model trained on the person's own labels follows their label only 46% of the time (Holm p = 1.0).
   - **This experiment provides no evidence of individual-specific emotional information.** The within-person result must not be described as decoding the person's own emotional experience.
3. **The best participants agree most with the consensus.**
   - s15 (agreement 0.925), s22 (0.850) and s03 (0.825) have the highest own-rating scores, and their consensus scores are similar or higher (0.859, 0.755 and 0.798 respectively).
   - s16, whose ratings are independent of the consensus (agreement 0.500), reaches 0.663 on their own ratings but 0.409 on consensus. That is the only individual-specific pattern again, and it is not significant (q = 0.18).
   - s07 is below chance on their own ratings (0.408) but above chance on consensus (0.645).
4. **More labelled videos help a little.** The curve rises from about 0.58–0.60 at 8–16 videos to 0.62 at 32, but the CIs overlap heavily and no test was planned. Practical calibration would need **at least 16–32 rated videos** per user, with a modest expected benefit.
5. **Borderline trials:** removing them (S1) changes little (0.632 vs 0.618), so the result is not driven by ambiguous ratings.

## 8. Limitations

- **Stimulus confound not resolved.**
  - Held-out videos stop the model from memorizing a specific video.
  - But it can still learn EEG responses to general stimulus properties (music tempo, brightness, motion) that correlate with valence. The consensus result indicates this kind of shared signal accounts for the effect.
  - The trusted data has no stimulus features, so "shared emotion" cannot be separated from "low-level video properties".
- **Low power for the individual-specific test:** 117 disagreement trials (3–20 per person). A null result here is not evidence that individual information is absent.
- **Small sample:** 12 participants × 40 trials. Per-participant estimates vary by ±0.02–0.09 across seeds, and per-participant CIs are wide.
- **Slow drift:** presentation order is not in the trusted files. Drift that correlates with labels cannot be excluded.
- **Labels:** one 60 s self-report per trial applied to all segments; binary threshold at 5.
- **Learning curve:** size 8 excludes 3 participants; descriptive only.
- **Model scope:** only the registered Random Forest, untuned.
- **Status:** exploratory under Protocol v1.1. No final-test, population or deployment claims, and the frozen test group was not used.

## 9. What the results support and what remains uncertain

- **Supported:**
  - Within this sample, a person-specific Random Forest trained on about 32 of that person's rated videos predicts their binary valence ratings on new videos above chance: about 0.62 balanced accuracy, versus 0.50 zero-shot.
  - It predicts the group consensus equally well.
- **Not supported:**
  - that the calibrated model captures the person's **individual** emotional response beyond what is shared across people;
  - any claim about new users, deployment or clinical relevance.
- **Uncertain:**
  - whether the shared component reflects shared affect or low-level stimulus properties;
  - whether individual-specific information exists but is too weak to detect here;
  - how many calibration videos would be needed outside DEAP.

## 10. Reproduce

```powershell
cd backend
..\.venv\Scripts\python -m scripts.within_person_calibration --phase main          # ~30 min plugged in; resumable
..\.venv\Scripts\python -m scripts.within_person_calibration --phase permutation   # ~5-7 h plugged in; resumable
..\.venv\Scripts\python -m scripts.within_person_calibration --phase report
..\.venv\Scripts\python -m pytest tests/test_within_person_calibration.py
```

**Results:**
- Live run: `outputs/experiments/within_person_calibration_rf_20261012/`.
- Verified copy: `docs/results/exploratory/within_person_calibration_rf/` (`summary.json`, `main.json`, `permutation.json`, `within_person_calibration.png`, `run_log.txt`).
