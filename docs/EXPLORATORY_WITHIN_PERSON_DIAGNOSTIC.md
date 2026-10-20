# Within-Person Valence Diagnostic (Random Forest)

**Status: exploratory (12 participants, Protocol v1.1).** This is a §9 D2 diagnostic. It is **not** a held-out test result, **not** a cross-participant result, and **not** evidence that any model outperforms the Random Forest (Decision C).

> Labels are self-reported valence (> 5.0 on DEAP's 1–9 scale). The results describe what a model can learn about one person's own ratings from that person's EEG. They are not a diagnosis and do not measure emotion directly.

**Files:**
- Results: `docs/results/exploratory/within_person_rf/` (`summary.json`, `main.json`, `permutation.json`, `within_person_ba.png`, `permutation_log.txt`)
- Code: `backend/scripts/within_person_diagnostic.py`
- Tests: `backend/tests/test_within_person_diagnostic.py`

## 1. Question

Does the existing 440-feature EEG pipeline carry **trial-specific** valence information when a model is trained and tested **inside one participant**?

This is a different question from the project's cross-participant evaluations, which ask whether a model trained on some people works on a new person. Those found only chance-level performance (§5).

## 2. Data and design (declared before running)

- **Data:** the 12 trusted DEAP participants from the configured dataset only (s01, s02, s03, s05, s07, s08, s09, s11, s15, s16, s22, s25), via the existing feature cache (fingerprint `3c5359083bc90ae5`, 7,046 segments).
  - No third-party copy, workbook or DENS data was used.
  - The 8 frozen test participants were not touched.
- **Labels:** `valence > 5.0` (primary rule), one label per trial, inherited by its segments.
- **Cross-validation:** per participant, **stratified 5-fold over the 40 trials** (8 test trials per fold). All segments of a trial share its fold. The code asserts that no trial is in both training and test, and that every trial is tested exactly once.
- **Model:** the registered Random Forest configuration, unchanged (`candidate_models(seed)["random_forest"]`, `n_jobs=-1`). Its scaler is fitted on training trials only. No hyperparameter was tuned.
- **Seeds:** 42–46. Each seed sets both the fold assignment and the Random Forest `random_state`.
- **Trial prediction:** the mean of segment probabilities, then argmax.
- **Primary metrics:** trial-level balanced accuracy and macro-F1. ROC-AUC and the predicted-high share are also reported. Per participant, each metric is the mean over 5 seeds.
- **Permutation test:**
  - seed 42, 250 permutations of trial labels within the participant;
  - **the full 5-fold CV is re-run for every permutation** (about 15,000 model fits);
  - one-sided p = (1 + #null ≥ observed) / 251;
  - Benjamini–Hochberg correction across the 12 participants.
- **Group-level permutation test:** compares the mean of the 12 observed seed-42 balanced accuracies against the mean of their null values, pairing permutation indices across participants.
- **Uncertainty:**
  - per participant: 95% trial-bootstrap CI (2,000 resamples within each class) of the seed-averaged prediction;
  - across participants: t-CI and Wilcoxon signed-rank test against 0.5.

**Verification:**
- One participant (s15, seed 42) was recomputed with independently written code. Balanced accuracy, macro-F1 and AUC matched exactly.
- The Benjamini–Hochberg q-values were recomputed independently and matched.
- Unit tests cover trial separation, seeded behaviour and a deliberately leaky splitter (which must fail).
- With `n_jobs=-1`, repeated runs can differ in the last floating-point bit (measured maximum 2.2 × 10⁻¹⁶). Predictions were unaffected.

## 3. Results per participant

| Participant | High / low trials | Borderline trials (4.5–5.5) | Balanced acc. (mean ± SD, 5 seeds) | Macro-F1 | ROC-AUC | Predicted high share | True high share | Seed-averaged balanced acc. [95% CI] | Permutation p (seed 42) | BH q |
|---|---|---|---|---|---|---|---|---|---|---|
| s01 | 19 / 21 | 2 | 0.510 ± 0.053 | 0.509 | 0.495 | 0.49 | 0.48 | 0.528 [0.377, 0.678] | 0.199 | 0.235 |
| s02 | 22 / 18 | 9 | 0.669 ± 0.037 | 0.669 | 0.677 | 0.63 | 0.55 | 0.687 [0.548, 0.821] | 0.028 | 0.084 |
| s03 | 22 / 18 | **13** | 0.706 ± 0.043 | 0.706 | 0.770 | 0.56 | 0.55 | 0.747 [0.619, 0.876] | 0.068 | 0.136 |
| s05 | 24 / 16 | 6 | 0.625 ± 0.038 | 0.626 | 0.728 | 0.68 | 0.60 | 0.656 [0.521, 0.792] | 0.044 | 0.105 |
| s07 | **28 / 12** | 5 | 0.450 ± 0.056 | 0.447 | 0.557 | 0.73 | 0.70 | 0.458 [0.333, 0.595] | 0.626 | 0.626 |
| s08 | 22 / 18 | 11 | 0.612 ± 0.036 | 0.612 | 0.645 | 0.59 | 0.55 | 0.646 [0.495, 0.798] | 0.175 | 0.234 |
| s09 | 20 / 20 | 10 | 0.645 ± 0.048 | 0.644 | 0.750 | 0.48 | 0.50 | 0.650 [0.500, 0.800] | 0.020 | 0.080 |
| s11 | 24 / 16 | 8 | 0.531 ± 0.042 | 0.525 | 0.505 | 0.60 | 0.60 | 0.542 [0.385, 0.698] | 0.215 | 0.235 |
| s15 | 20 / 20 | 5 | **0.760 ± 0.014** | 0.759 | 0.826 | 0.52 | 0.50 | 0.775 [0.650, 0.900] | **0.004** | **0.048** |
| s16 | **15 / 25** | 5 | 0.651 ± 0.050 | 0.653 | 0.694 | 0.35 | 0.38 | 0.633 [0.473, 0.780] | 0.096 | 0.164 |
| s22 | 18 / 22 | 7 | **0.752 ± 0.038** | 0.754 | 0.761 | 0.39 | 0.45 | 0.793 [0.664, 0.917] | **0.008** | **0.048** |
| s25 | 19 / 21 | 9 | 0.594 ± 0.011 | 0.593 | 0.604 | 0.48 | 0.48 | 0.549 [0.393, 0.699] | 0.159 | 0.234 |

Figure: `within_person_ba.png`. It shows each participant's mean ± SD over seeds, the seed-averaged 95% CI, and the 95th percentile of the permutation null.

**Participants whose results are particularly uncertain:**
- **s07 (28 high / 12 low)** is the most imbalanced. It has only about 2–3 low-valence trials per test fold, and it is the only participant below 0.5.
- **s16 (15 high / 25 low)** is the next most imbalanced.
- **Borderline labels:** s03 (13 of 40 trials in 4.5–5.5), s08 (11), s09 (10), s02 and s25 (9 each) have many trials near the threshold, so their labels are noisier.
- No participant falls under the < 5-trials-per-class flag (Decision J); the minimum is 12.

## 4. Group-level results

| Statistic | Value |
|---|---|
| Mean trial balanced accuracy, 12 participants | **0.625** (sample SD 0.094), 95% t-CI [0.566, 0.685] |
| Mean trial macro-F1 | 0.625, 95% CI [0.564, 0.686] |
| Mean trial ROC-AUC | 0.667, 95% CI [0.599, 0.736] |
| Participants with balanced accuracy > 0.5 | 11 of 12 |
| Wilcoxon signed-rank test vs 0.5 (n = 12) | median difference +0.135, p = 0.0024 |
| Group permutation test (seed 42, mean balanced accuracy 0.640) | p = 0.004 (the minimum possible with 250 permutations); null 95th percentile 0.542 |
| Individually significant after BH (q < 0.05) | s15 and s22 (q = 0.048 each) |

**No one-label-per-person collapse.** For every participant, the predicted-high share stays close to the true-high share (largest gap 0.08). This is the opposite of the cross-participant CNN, which gave many people a single label.

## 5. Comparison with cross-participant results (different questions)

| Question | Evaluation | Trial-level result | Source |
|---|---|---|---|
| **Within one person:** train and test on different trials of the same person | 5-fold trial-wise CV, 12 people, this study | balanced accuracy **0.625** (CI 0.566–0.685) | this document |
| **Across people:** train on others, test on a new person | Random Forest LOPO, 9 non-test people (pre-protocol) | accuracy 0.503 (balanced accuracy not recorded) | registered RF metadata |
| **Across people** | CNN LOPO, train_stats, 9 people (pre-protocol) | balanced accuracy 0.495 | `docs/results/review3/robust_cnn/summary.json` |
| **Across people and videos** (added 2026-10-11) | Random Forest, held-out participant × held-out videos, 12 people, 5 seeds | balanced accuracy 0.500 (own rating), 0.482 (consensus) | `docs/EXPLORATORY_CROSS_PARTICIPANT_LEAVE_VIDEO_OUT.md` |

The two rows can't be compared as "better or worse": they answer different questions.
- **Within person:** the model has seen labelled trials from the same person, which a real new user would not provide.
- **Across people:** asks about generalization to unseen people, which is the project's target.

## 6. What the results suggest

1. **The feature pipeline carries some person-specific valence information.**
   - At group level, within-person decoding is above chance: Wilcoxon p = 0.002, group permutation p = 0.004, and 11 of 12 participants above 0.5.
   - So "the 440 features contain no valence-related information at all" is not supported for this sample.
2. **The effect is modest and uneven.**
   - Only 2 of 12 participants are individually significant after correction.
   - s01, s07 and s11 are at or below chance, and s25's 95% CI includes 0.5.
   - With 40 trials, chance alone often produces 0.62–0.67 for a single person (see the permutation null in the figure).
3. **Together with the earlier cross-participant results,** the evidence fits the hypothesis that useful EEG-valence relationships are **largely participant-specific**, rather than shared in a way the current features and models transfer to new people. This is a hypothesis: the diagnostic does not test it directly.

> **Update (2026-10-11):** the stimulus-confound control in §9 does **not** support point 3. The same pipeline predicts the other participants' consensus label for each video at least as well as the person's own rating, and it shows no preference for the person's own rating where the two disagree. The original results in §3–§5 are unchanged; their interpretation is revised in §9.

## 7. What these results cannot establish

- **Stimulus confound.** All participants watched the same 40 music videos, and in DEAP valence ratings depend strongly on the video. A within-person model may therefore pick up EEG responses to **video properties** (visual, audio, music style) that happen to go with the rating, rather than the person's affective state. Train and test always use different videos, which limits but does not remove this.
- **Slow drift.** Trials are stored in video order, not presentation order. Slow changes in the recording (fatigue, electrode drift) could still correlate with labels. The presentation order is not available from the trusted files, so this was not tested.
- **Deployment.** A within-person model needs labelled recordings from each new person. These results say nothing about predicting a new user without such calibration.
- **Small sample.** These are 12 participants and 40 trials each, so individual estimates have 95% CIs about ±0.15 wide.
  - The permutation test used one seed, with 250 permutations. The smallest possible p (0.004) barely clears BH at 12 tests.
  - Results near the significance boundary (s02, s05, s09) are fragile.
- **Labels.** One 60 s self-report per trial, with up to 13 borderline trials per person.
- **Model scope.** Only the registered Random Forest and the existing features were tested. No conclusion is drawn about the CNN or other models.
- **Status.** Exploratory under Protocol v1.1: no formal or independent-test claims.

## 8. Reproduce

```powershell
cd backend
..\.venv\Scripts\python -m scripts.within_person_diagnostic --phase main          # ~4 min plugged in
..\.venv\Scripts\python -m scripts.within_person_diagnostic --phase permutation   # ~3.5 h plugged in; resumable
..\.venv\Scripts\python -m scripts.within_person_diagnostic --phase report
..\.venv\Scripts\python -m pytest tests/test_within_person_diagnostic.py
```

Results are written to `outputs/experiments/within_person_rf_20261010/`. The verified copy is in `docs/results/exploratory/within_person_rf/`.

## 9. Follow-up: stimulus-confound control (2026-10-11)

**Status: exploratory (12 participants, Protocol v1.1).**

**Files:**
- Results: `docs/results/exploratory/stimulus_control_rf/` (`summary.json`, `main.json`, `folds.json`, `stimulus_control.png`, `run_log.txt`)
- Code: `backend/scripts/stimulus_confound_control.py`
- Tests: `backend/tests/test_stimulus_confound_control.py`

### 9.1 Question

Does the pipeline predict a participant's **own** valence rating better than the **consensus** valence of the same video among the other participants?

- If the consensus is predicted as well or better, much of the within-person signal is plausibly shared across people and stimulus-related.
- If the person's own rating is predicted better, that supports individual-specific information.

Neither outcome identifies a mechanism.

### 9.2 Design (declared before running)

- **Data:** the same 12 trusted participants and feature cache (fingerprint `3c5359083bc90ae5`). No third-party data, and the frozen test participants were not touched.
- **Trial = video alignment, verified from the trusted ratings alone:** the mean pairwise Spearman correlation of valence by trial index across the 12 participants is **0.443**, against a shuffled-alignment null with mean 0.000 and maximum 0.060 over 1,000 shuffles (p = 0.001). So trial index *t* refers to the same video for every participant. This agrees with the earlier workbook mapping, which was not used here.
- **Personal label:** own rating, `valence > 5.0`.
- **Consensus label:** for participant *p* and video *t*, the majority of the **other 11** participants' `valence > 5.0` labels (11 voters, so no ties). The participant's own rating is never used. Every participant's consensus labels are balanced (19–21 high out of 40), so no participant was too imbalanced for this task.
- **Folds:** per participant and seed, stratified 5-fold over trials on the **joint** (personal, consensus) label, so neither task is favoured. The folds are saved in `folds.json` and used identically for both tasks.
- **Model and metrics:** seeds 42–46, the registered Random Forest unchanged, trial-level mean of segment probabilities, as in §2.
- **Statistics:**
  - paired per-participant differences (personal − consensus): t-CI, participant bootstrap (10,000), Wilcoxon signed-rank, and an **exact** sign-flip permutation test (all 4,096 sign patterns);
  - personal/consensus agreement and Cohen's κ, with 95% trial-bootstrap CIs;
  - on **disagreement trials** (personal ≠ consensus), the share of the personal-task model's seed-averaged predictions that follow the personal label, with an exact sign-flip test against 0.5.
- **No per-participant retraining permutation test.** The question is a paired comparison, which the exact sign-flip test covers, so per-participant permutation tests for the consensus task (estimated at 3.5 h plugged in) were not run. No family of per-participant tests is reported, so no BH correction was needed.

**Verification:**
- personal and consensus labels were recomputed independently from the trusted loader, and both matched;
- every saved fold set partitions the 40 trials;
- one result (s05, consensus task, seed 43) was recomputed with independently written code using the saved folds, and balanced accuracy, macro-F1 and AUC matched exactly;
- the code asserts trial separation in every fold;
- 5 new unit tests.

**Runtime:** about 12 minutes on battery for the main phase (600 Random Forest fits), and about 1 minute for the report. It saves after each participant, so it can resume.

### 9.3 Results per participant

Trial balanced accuracy, mean ± SD over 5 seeds. "Model follows own / consensus" counts the disagreement trials whose seed-averaged prediction matches each label.

| Participant | Agreement [95% CI] | κ | Personal | Consensus | Personal − consensus | Disagreement trials | Personal-task model follows own / consensus | Consensus-task model follows own / consensus |
|---|---|---|---|---|---|---|---|---|
| s01 | 0.600 [0.45, 0.75] | 0.20 | 0.511 ± 0.064 | 0.395 ± 0.047 | +0.116 | 16 | 10 / 6 | 9 / 7 |
| s02 | 0.775 [0.65, 0.90] | 0.55 | 0.683 ± 0.031 | 0.590 ± 0.028 | +0.093 | 9 | 6 / 3 | 6 / 3 |
| s03 | 0.825 [0.70, 0.93] | 0.65 | 0.736 ± 0.044 | 0.820 ± 0.032 | −0.083 | 7 | 3 / 4 | 2 / 5 |
| s05 | 0.725 [0.60, 0.85] | 0.46 | 0.598 ± 0.009 | 0.774 ± 0.041 | −0.176 | 11 | 2 / 9 | 4 / 7 |
| s07 | 0.750 [0.62, 0.88] | 0.50 | 0.423 ± 0.046 | 0.640 ± 0.063 | −0.217 | 10 | 3 / 7 | 5 / 5 |
| s08 | 0.775 [0.65, 0.90] | 0.55 | 0.582 ± 0.067 | 0.661 ± 0.083 | −0.079 | 9 | 3 / 6 | 4 / 5 |
| s09 | 0.800 [0.68, 0.90] | 0.60 | 0.665 ± 0.029 | 0.685 ± 0.038 | −0.020 | 8 | 3 / 5 | 4 / 4 |
| s11 | 0.775 [0.65, 0.90] | 0.55 | 0.452 ± 0.056 | 0.514 ± 0.070 | −0.061 | 9 | 4 / 5 | 4 / 5 |
| s15 | 0.925 [0.85, 1.00] | 0.85 | 0.775 ± 0.047 | 0.852 ± 0.022 | −0.077 | 3 | 0 / 3 | 0 / 3 |
| s16 | 0.500 [0.35, 0.65] | 0.01 | 0.680 ± 0.016 | 0.454 ± 0.027 | +0.226 | 20 | 13 / 7 | 12 / 8 |
| s22 | 0.850 [0.72, 0.95] | 0.70 | 0.730 ± 0.048 | 0.780 ± 0.033 | −0.050 | 6 | 4 / 2 | 2 / 4 |
| s25 | 0.775 [0.62, 0.90] | 0.55 | 0.595 ± 0.056 | 0.620 ± 0.062 | −0.025 | 9 | 5 / 4 | 4 / 5 |

### 9.4 Group-level results

| Statistic | Value |
|---|---|
| Personal-task balanced accuracy | 0.619, 95% CI [0.547, 0.691] |
| Consensus-task balanced accuracy | **0.649**, 95% CI [0.557, 0.741]; vs 0.5: Wilcoxon p = 0.007 |
| Difference (personal − consensus) | **−0.029**, t-CI [−0.108, 0.049]; participant bootstrap [−0.093, 0.040]; Wilcoxon p = 0.47; exact sign-flip p = 0.42 |
| Participants where personal > consensus | 3 of 12 (s01, s02, s16) |
| Macro-F1 difference | −0.029 (sign-flip p = 0.44) |
| ROC-AUC difference | −0.023 (sign-flip p = 0.55); personal AUC 0.663, consensus AUC 0.686 |
| Personal–consensus label agreement | 0.756, 95% CI [0.685, 0.827]; κ 0.514 [0.374, 0.654] |
| Disagreement trials, personal-task model | follows the **own** label on 56 of 117 pooled trials (0.479); mean per-participant share 0.436, 95% CI [0.302, 0.570]; exact sign-flip p vs 0.5 = 0.32 |
| Agreement vs personal-task balanced accuracy | Spearman ρ = 0.61, p = 0.034 (n = 12) |

Figure: `stimulus_control.png`.

### 9.5 Interpretation

1. **The other participants' consensus is predicted at least as well as the person's own rating.** The paired difference is small and not detectable (−0.029, CI including 0), and consensus scored higher for 9 of 12 participants.
2. **Where a person's rating disagrees with the consensus, the model trained on that person's own ratings does not follow them.** It matches the own label on 48% of pooled disagreement trials, which is not different from 50%.
3. **Personal-task accuracy is higher for people whose ratings agree more with the consensus** (ρ = 0.61). The two individually significant participants in §3, s15 and s22, have the highest agreement (0.925 and 0.850).
4. **Taken together,** the within-person valence decoding in §3–§4 is consistent with a substantial **shared, stimulus-related component**: EEG responses that relate to the video and to how people generally rate it. This sample gives **no evidence for individual-specific valence information** beyond that.
5. **This does not prove the model merely recognizes videos.** The consensus label is itself a shared affective rating, so "stimulus-related" covers both a shared emotional response and low-level properties of the video (visual, audio, music style). This design cannot separate the two.
6. **s16 is the only clear counter-example.** Its own labels are independent of the consensus (κ = 0.01), and it decodes its own ratings (0.680) much better than the consensus (0.454). As a single participant whose personal result was not significant in §3 (q = 0.16), it cannot carry a general conclusion.
7. **Point 3 of §6 is not supported.** The earlier hypothesis that the EEG-valence relationship is largely participant-specific is not supported by this control.

### 9.6 Limitations

- **Small sample:** 12 participants; 3–20 disagreement trials per participant (117 in total), so the disagreement analysis has low power.
- **Fold sensitivity:** changing only the fold stratification moved several personal-task scores compared with §3, for example s11 0.531 → 0.452 and s03 0.706 → 0.736. Single-participant estimates vary by about ±0.08 with fold assignment alone.
- **Consensus from 11 people:** this is noisy, especially for videos with close votes (2–4 such videos per participant had 5–6 of 11 votes high).
- **No stimulus features:** there are no video features in the trusted data, so the low-level stimulus explanation could not be tested directly.
- **Scope:** the Random Forest with the existing features only, and exploratory status, so no formal or independent-test claims.

### 9.7 Reproduce

```powershell
cd backend
..\.venv\Scripts\python -m scripts.stimulus_confound_control --phase main     # ~7 min plugged in; resumable
..\.venv\Scripts\python -m scripts.stimulus_confound_control --phase report
..\.venv\Scripts\python -m pytest tests/test_stimulus_confound_control.py
```

> **Follow-up (2026-10-12):** `docs/EXPLORATORY_WITHIN_PERSON_CALIBRATION.md` repeats this within-person analysis on label-free video folds identical to the cross-person experiments, with a pre-declared stimulus-confound test. The results are own-rating balanced accuracy 0.618 (p = 0.004), consensus 0.626 (P − C Holm p = 1.0) and disagreement-trial share 0.46 (Holm p = 1.0). The conclusion of §9 is unchanged: calibration works, but there is no evidence of individual-specific information.
