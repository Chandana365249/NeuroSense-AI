# NeuroSense AI: Robust CNN Evaluation (Review 3, Phase A)

**Purpose:** find out whether the EEGNet-style CNN's single registered result (`cnn_valence_binary_20261006-045706`) holds up when the training seed, the participants, and the input normalization change.

**Scope:** evaluation only.
- No new architecture was added.
- No hyperparameter was tuned.
- No test or held-out participant influenced any choice.
- The production model and the application are unchanged: the Random Forest stays active, and the CNN stays registered but inactive.

> All numbers below come from real DEAP recordings (12 participants, 7,046 segments, task: binary self-reported valence > 5). They estimate a self-reported rating, not a diagnosis.

Full generated outputs are in `docs/results/review3/robust_cnn/`:
- `REPORT.md`: every table;
- `summary.json`: all metrics, CIs and p-values;
- `figures/`;
- `experiment.json`: the configuration.

The raw per-run records (with per-segment predictions) are in `outputs/experiments/cnn_robust_20261009/runs/`, which is git-ignored.

## 1. What was inspected first

| Item | Where | Finding |
|---|---|---|
| CNN | `app/ml/deep/cnn_model.py` | EEGNet-style, 2,130 parameters; input (1, 32 channels, 512 samples) |
| Input pipeline | `app/ml/deep/data.py` | Same `EEGPreprocessor` segments as the feature models, verified row for row; per-channel z-score with statistics from fitted participants only |
| Training | `app/ml/deep/training.py` | Early stopping on validation macro-F1 (patience 10), then a refit on train + validation for the selected epoch count |
| Evaluation | `CNNTrainer._evaluate`, `app/ml/evaluation.py` | Segment- and trial-level metrics on the held-out test participants, evaluated once |
| Results storage | `outputs/metrics/<version>.json`, `_history.json`, `_test_predictions.csv`, `outputs/graphs/` | Complete for a single run |
| Registry | `ModelRegistry.save(..., activate=False)` | Weights plus metadata with SHA-256; not activated |
| Tests | `tests/test_deep.py` (18) | Shapes, determinism, checkpoints, registry, API; nothing on evaluation robustness |

**Leakage review: no leakage found.**
- Whole participants are kept per split.
- The normalizer and class weights are fitted on training rows only.
- The epoch is selected on validation participants.
- Test participants are evaluated once.
- The 3 test participants were also never used by the Random Forest's leave-one-participant-out (LOPO) cross-validation.

**Evaluation risks found:**
1. One split and one seed, so seed variance was unknown.
2. Only 2 validation participants drive early stopping, which makes it a noisy signal.
3. Only 3 test participants, so between-person variance was unknown.
4. No CNN LOPO cross-validation, unlike the Random Forest (RF).
5. No uncertainty estimates.
6. Training-mode metrics include dropout, so the train/validation gap was not measured cleanly.

## 2. What was added

| File | Purpose |
|---|---|
| `backend/app/ml/stats.py` | Trial-cluster bootstrap CIs, within-participant permutation tests, t-CIs across folds, Wilcoxon signed-rank tests, fast binary metrics (tested equal to `compute_metrics`) |
| `backend/app/ml/deep/robust_evaluation.py` | Run planner and runner: repeated seeds, nested LOPO, two normalizations. Reuses `CNNTrainer`'s own fit and evaluate code, so training is identical to the registered model |
| `backend/app/ml/deep/robust_report.py` | Aggregation, comparability checks, comparison with the registered RF, figures, Markdown report |
| `backend/scripts/evaluate_cnn.py` | Resumable experiment runner (one JSON record per run) |
| `backend/tests/test_robust_evaluation.py` | 13 tests (statistics, leakage-safe planning, exact reproduction of the registered training procedure, end-to-end report) |

**Experiment tracking:** every run records the following.
- architecture config and parameter count;
- seed;
- full training config;
- preprocessing config and segment-cache version;
- dataset fingerprint;
- participant lists per role;
- normalization;
- full per-epoch history;
- epochs run and best epoch;
- eval-mode train/validation metrics of the selected checkpoint;
- held-out metrics at segment, trial and per-participant level;
- majority-class reference;
- per-segment probabilities;
- git commit and timing.

## 3. Protocols

**A. Repeated seeds on the fixed split.**
- **Split:** train s02, s03, s05, s07, s08, s15, s22 · validation s16, s25 · test s01, s09, s11.
- **Seeds:** 42–46. Only the training seed changes (weight initialisation, batch order, dropout).
- **Procedure per seed:** exactly the registered CNN's procedure, with the test participants scored once per seed.
- **Nothing is selected from these test scores.** No "best seed" is reported as the result.

**B. Nested leave-one-participant-out.**
- **Participants:** the 9 non-test participants, the same set as the RF's registered LOPO cross-validation.
- **Per held-out person:** 2 inner validation participants are drawn from the remaining 8 (seeded) and used for early stopping. The model is then refit on all 8 for the selected epoch count and scored once on the held-out person.
- **Test participants:** never touched.

**Normalizations (both use no labels):**
- `train_stats`: per-channel z-score with statistics of the fitted participants. This is the registered CNN's method.
- `per_participant`: each participant's segments are first z-scored per channel with that person's own statistics. For a new person this needs an unlabelled calibration recording.

**Uncertainty:**
- **CIs:** 2,000 bootstrap resamples of whole trials within each participant (95% percentile CI).
- **Permutation tests:** 2,000 permutations of trial labels within each participant (one-sided).
- **Across LOPO folds:** t-based 95% CI and Wilcoxon signed-rank tests (n = 9, low power).

The within-participant permutation asks whether the model separates high- from low-valence **trials of the same person**. A model that gives one label to everyone in a person's data scores p = 1 on macro-F1 under this test, however its pooled accuracy looks.

**Checks, all passed:**
- same split, dataset fingerprint and LOPO participants as the registered RF;
- the RF's recomputed predictions reproduce its registered metrics exactly;
- **the seed-42 / train_stats rerun reproduces the registered CNN exactly** (same best epoch 2, identical metrics and confusion matrix).

## 4. Results A: repeated seeds, held-out test participants (s01, s09, s11)

Mean ± sample SD over 5 seeds. These measure seed sensitivity on the same 3 people, not variability across people.

**Per 4 s segment (n = 1,788):**

| Model | Accuracy | Balanced acc. | Macro-F1 | ROC-AUC |
|---|---|---|---|---|
| CNN, train_stats (5 seeds) | 0.532 ± 0.033 | 0.531 ± 0.026 | 0.466 ± 0.109 | 0.548 ± 0.057 |
| CNN, per_participant (5 seeds) | 0.484 ± 0.021 | 0.500 ± 0.021 | 0.422 ± 0.067 | 0.482 ± 0.047 |
| Random Forest (registered, deterministic) | **0.548** | **0.544** | **0.543** | 0.537 |
| Majority-class reference | 0.525 | 0.500 | 0.344 | 0.500 |

**Per trial (n = 120):**

| Model | Accuracy | Balanced acc. | Macro-F1 | ROC-AUC |
|---|---|---|---|---|
| CNN, train_stats | 0.532 ± 0.035 | 0.530 ± 0.030 | 0.459 ± 0.116 | 0.555 ± 0.078 |
| CNN, per_participant | 0.482 ± 0.030 | 0.498 ± 0.031 | 0.408 ± 0.082 | 0.488 ± 0.062 |
| Random Forest | **0.592** | **0.590** | **0.590** | **0.565** |

**Each seed (segment level):**

| Normalization | Seed | Best / run epochs | Validation macro-F1 | Test acc. | Test macro-F1 | Test AUC | Trial AUC (permutation p) |
|---|---|---|---|---|---|---|---|
| train_stats | 42 | 2 / 12 | 0.558 | 0.542 | 0.525 | 0.577 | 0.590 (0.026) |
| train_stats | 43 | 2 / 12 | 0.548 | 0.532 | 0.378 | 0.574 | 0.605 (0.002) |
| train_stats | 44 | 4 / 14 | 0.536 | 0.554 | 0.554 | 0.572 | 0.578 (0.098) |
| train_stats | 45 | 16 / 26 | 0.528 | 0.475 | 0.322 | 0.445 | 0.417 (0.955) |
| train_stats | 46 | 2 / 12 | 0.584 | 0.556 | 0.553 | 0.571 | 0.585 (0.055) |
| per_participant | 42 | 10 / 20 | 0.532 | 0.477 | 0.350 | 0.449 | 0.448 (0.483) |
| per_participant | 43 | 23 / 33 | 0.475 | 0.475 | 0.351 | 0.446 | 0.445 (0.832) |
| per_participant | 44 | 4 / 14 | 0.536 | 0.521 | 0.494 | 0.561 | 0.596 (0.061) |
| per_participant | 45 | 11 / 21 | 0.554 | 0.476 | 0.452 | 0.482 | 0.482 (0.627) |
| per_participant | 46 | 2 / 12 | 0.541 | 0.471 | 0.460 | 0.470 | 0.469 (0.192) |

**Uncertainty on the test participants** (trial-cluster bootstrap 95% CI; within-participant permutation p):

| Model | Segment macro-F1 [CI] | Trial accuracy [CI] | Trial AUC [CI] | p (trial macro-F1) | p (trial AUC) |
|---|---|---|---|---|---|
| Random Forest | 0.543 [0.485, 0.601] | 0.592 [0.508, 0.683] | 0.565 [0.459, 0.669] | < 0.001 | < 0.001 |
| CNN train_stats, seed 42 (registered) | 0.525 [0.434, 0.615] | 0.542 [0.450, 0.633] | 0.590 [0.487, 0.693] | 1.000 | 0.026 |

The other 9 CNN runs are listed in `robust_cnn/REPORT.md` §3. 2 of 5 train_stats seeds, and 0 of 5 per_participant seeds, reach p < 0.05 for trial AUC. For trial macro-F1, no CNN run reaches p < 0.05.

**Per test participant** (segment level; CNN = mean ± SD over seeds; "pred. high" = share of that person's segments predicted high_valence):

| Participant (self-reported high) | Model | Balanced acc. | Macro-F1 | AUC | Pred. high |
|---|---|---|---|---|---|
| s01 (0.475) | Random Forest | 0.501 | 0.368 | 0.502 | 0.943 |
| | CNN train_stats | 0.501 ± 0.002 | 0.344 ± 0.001 | 0.505 ± 0.048 | 0.196 ± 0.439 |
| | CNN per_participant | 0.511 ± 0.009 | 0.423 ± 0.045 | 0.524 ± 0.028 | 0.370 ± 0.411 |
| s09 (0.503) | Random Forest | **0.643** | **0.641** | **0.696** | 0.431 |
| | CNN train_stats | 0.521 ± 0.022 | 0.425 ± 0.099 | 0.569 ± 0.064 | 0.591 ± 0.424 |
| | CNN per_participant | 0.505 ± 0.011 | 0.357 ± 0.037 | 0.454 ± 0.101 | 0.026 ± 0.032 |
| s11 (0.598) | Random Forest | 0.551 | 0.523 | 0.574 | 0.372 |
| | CNN train_stats | 0.500 ± 0.000 | 0.359 ± 0.040 | 0.527 ± 0.047 | 0.797 ± 0.446 |
| | CNN per_participant | 0.501 ± 0.006 | 0.340 ± 0.091 | 0.512 ± 0.026 | 0.150 ± 0.302 |

**Train / validation / test consistency** (segment macro-F1 of the selected checkpoint in eval mode, mean ± SD over seeds):

| Normalization | Train | Validation | Test | Selected epoch |
|---|---|---|---|---|
| train_stats | 0.588 ± 0.026 | 0.551 ± 0.022 | 0.466 ± 0.109 | 5.2 ± 6.1 |
| per_participant | 0.649 ± 0.068 | 0.527 ± 0.030 | 0.422 ± 0.067 | 10.0 ± 8.2 |

Figures: `figures/seed_distribution.png`, `figures/per_participant.png`, `figures/learning_curves.png`.

## 5. Results B: nested leave-one-participant-out (9 non-test participants)

Mean ± sample SD over 9 folds; 95% t-CI.

| Model | Segment accuracy [95% CI] | Balanced acc. | Macro-F1 | ROC-AUC | Trial accuracy |
|---|---|---|---|---|---|
| CNN, train_stats | 0.503 ± 0.080 [0.441, 0.565] | 0.494 ± 0.023 | 0.383 ± 0.065 | 0.506 ± 0.102 | 0.503 ± 0.085 |
| CNN, per_participant | 0.552 ± 0.074 [0.495, 0.609] | 0.505 ± 0.016 | 0.408 ± 0.051 | 0.550 ± 0.136 | 0.561 ± 0.074 |
| Random Forest (registered LOPO) | 0.510 ± 0.091 [0.440, 0.580] | not recorded | **0.416 ± 0.053** | 0.538 ± 0.085 | 0.503 ± 0.100 |
| Majority-class reference | 0.528 ± 0.093 | 0.500 | 0.343 ± 0.040 | 0.500 | |

The Random Forest's ± 0.091 is the sample SD. The ± 0.086 quoted in Review 2 is the population SD of the same 9 folds.

**Per fold** (segment accuracy / macro-F1 / AUC / selected epoch):

| Held-out | Inner validation | CNN train_stats | CNN per_participant | RF acc. / macro-F1 | Majority acc. |
|---|---|---|---|---|---|
| s02 | s08, s15 | 0.486 / 0.483 / 0.468 / 1 | 0.443 / 0.347 / 0.406 / 1 | 0.460 / 0.387 | 0.551 |
| s03 | s08, s15 | 0.449 / 0.310 / 0.567 / 7 | 0.544 / 0.412 / 0.551 / 1 | 0.474 / 0.465 | 0.551 |
| s05 | s08, s15 | 0.472 / 0.449 / 0.553 / 1 | 0.600 / 0.383 / 0.547 / 9 | 0.590 / 0.438 | 0.602 |
| s07 | s08, s15 | 0.700 / 0.412 / 0.633 / 13 | 0.693 / 0.508 / 0.669 / 19 | 0.700 / 0.417 | 0.700 |
| s08 | s07, s15 | 0.433 / 0.344 / 0.492 / 5 | 0.477 / 0.458 / 0.533 / 26 | 0.449 / 0.391 | 0.548 |
| s15 | s07, s08 | 0.501 / 0.354 / 0.561 / 13 | 0.519 / 0.403 / 0.743 / 31 | 0.570 / 0.486 | 0.501 |
| s16 | s07, s08 | 0.513 / 0.441 / 0.376 / 9 | 0.600 / 0.387 / 0.343 / 7 | 0.413 / 0.373 | 0.376 |
| s22 | s07, s08 | 0.446 / 0.309 / 0.322 / 5 | 0.569 / 0.425 / 0.701 / 6 | 0.463 / 0.463 | 0.446 |
| s25 | s07, s08 | 0.525 / 0.344 / 0.581 / 29 | 0.525 / 0.351 / 0.454 / 4 | 0.475 / 0.322 | 0.475 |

**Wilcoxon signed-rank tests across folds** (n = 9, two-sided):

| Comparison | Median difference | p |
|---|---|---|
| CNN train_stats: balanced accuracy vs 0.5 | 0.000 | 0.625 |
| CNN train_stats: accuracy vs majority reference | 0.000 | 0.562 |
| CNN train_stats vs RF: accuracy / macro-F1 | −0.015 / −0.005 | 0.844 / 0.496 |
| CNN per_participant: balanced accuracy vs 0.5 | 0.000 | 0.203 |
| CNN per_participant: accuracy vs majority reference | −0.002 | 0.734 |
| CNN per_participant vs RF: accuracy / macro-F1 | +0.029 / −0.038 | 0.164 / 0.734 |

Figure: `figures/lopo_folds.png`.

## 6. Findings

1. **The CNN does not beat the Random Forest under any protocol.**
   - On the held-out test participants, the RF (macro-F1 0.543, trial accuracy 0.592) is above the CNN's 5-seed mean with either normalization (0.466 and 0.422; trial accuracy 0.532 and 0.482).
   - In LOPO, the CNN's macro-F1 (0.383 and 0.408) is at or below the RF's 0.416.
   - No CNN-vs-RF difference is statistically detectable with 9 folds.
2. **Neither model shows reliable cross-subject valence decoding.**
   - In LOPO, CNN balanced accuracy is 0.494 ± 0.023 (train_stats) and 0.505 ± 0.016 (per_participant). Neither differs from 0.5 (p = 0.63 and 0.20), and neither beats the majority-class reference's accuracy (p = 0.56 and 0.73).
   - The RF's LOPO accuracy (0.510) is also below the majority reference's (0.528).
   - The RF is the only model whose test-set predictions separate trials **within** a person beyond chance (within-participant permutation p < 0.001 for trial macro-F1 and AUC). That signal comes mainly from s09 (balanced accuracy 0.643, AUC 0.696).
3. **The registered CNN result depends strongly on the seed.**
   - Across 5 seeds with identical data and settings, test macro-F1 ranged from 0.322 to 0.554 (SD 0.109), and segment AUC from 0.445 to 0.577.
   - The registered seed 42 (0.525) is one of the better seeds. The 5-seed mean (0.466) is the more honest estimate.
4. **The CNN mostly assigns one label per person.**
   - With train_stats, per-participant balanced accuracy is 0.500–0.521 on every test person.
   - The share of a person's segments predicted high_valence swings between about 0 and 1 from seed to seed (SD ≈ 0.42–0.45).
   - So most of the CNN's pooled accuracy comes from which label it happens to give each whole person, not from separating that person's trials.
5. **Per-participant normalization did not fix this.**
   - On the fixed split it was worse than train_stats: macro-F1 0.422 vs 0.466, AUC 0.482 vs 0.548, and 0 of 5 seeds with p < 0.05 for trial AUC (vs 2 of 5).
   - In LOPO it raised mean accuracy (0.552 vs 0.503) and AUC (0.550 vs 0.506). But balanced accuracy stayed at 0.505 and accuracy did not beat the majority reference, so the gain is mostly from leaning towards the more common class, not better discrimination.
   - It also predicted low_valence for most segments on the test split (summed confusion matrix `[[856, 3839], [774, 3471]]`).
6. **Overfitting is visible, and model selection is noisy.**
   - Training loss falls steadily while validation loss rises after about epoch 2 in most runs (`figures/learning_curves.png`).
   - Eval-mode train macro-F1 (0.588 / 0.649) exceeds validation (0.551 / 0.527), which exceeds test (0.466 / 0.422).
   - Validation macro-F1 at the selected epoch (0.475–0.584 across the 10 seed runs) did not reliably predict test macro-F1 (0.322–0.554). For example, train_stats seed 45 had validation 0.528 but test 0.322, and per_participant seed 45 had the second-highest validation score (0.554) but test 0.452.
   - Selected epochs range from 1 to 31.
7. **Some ranking signal exists but does not hold across people.**
   - Several CNN runs and folds reach AUC 0.6–0.74 (for example LOPO s15 per_participant 0.743, s22 0.701), but others fall well below 0.5 (s16: 0.376 and 0.343; s22 train_stats 0.322).
   - Per-person AUC below 0.5 means the learned direction is reversed for that person.

**Consequences for the project:**
- **Production:** the Random Forest remains the active model. The CNN stays registered but inactive, and no CNN variant from this phase is registered.
- **Reporting:** report the CNN as **0.466 ± 0.109 test macro-F1 over 5 seeds** (LOPO 0.383 ± 0.065), not the single 0.525 run.
- **Next models:** later architectures (LSTM, CNN-LSTM, TCN) should be judged with this same protocol (seeds plus nested LOPO plus within-participant tests). A single split cannot tell them apart.

## 7. Limitations

- **Few people.** There are 12 participants, with 3 test people and 9 LOPO folds. The bootstrap CIs cover trials of the evaluated people only, not the population, and with n = 9 the Wilcoxon tests have little power.
- **Noisy model selection.** Early stopping relies on 2 validation participants, and validation macro-F1 is a poor guide to test performance (§6).
- **Little variety in inner validation.** In LOPO, the inner validation pair is drawn with one fixed seeded permutation, so it is {s08, s15} for 4 folds, {s07, s08} for 4 folds and {s07, s15} for 1 fold.
- **One seed for LOPO.** Each LOPO fold used seed 42 only. Given the seed spread in §4, fold-level CNN numbers carry seed noise as well as person-to-person differences.
- **Untuned.** The CNN uses EEGNet defaults. Untuned is not the same as "cannot work", but tuning on 2 validation participants would mostly fit noise.
- **Coarse labels.** Every 4 s segment inherits its trial's single 60 s self-report, and the threshold of 5 is a project convention, not a ground truth.
- **Calibration assumption.** Per-participant normalization assumes an unlabelled calibration recording per person. It is evaluated here, not deployed.
- **CPU only.** All 28 runs ran on CPU (about 30 s per epoch), which limited how many seeds and folds were feasible.
- **Comparability caveat for the RF.** The RF's LOPO trains on all 8 remaining participants with no early stopping. The CNN uses 6 for training and 2 for early stopping, then refits on all 8 for the selected epoch count. Both see the same data, but not in exactly the same way.

## 8. Reproduce

```powershell
cd backend
..\.venv\Scripts\python -m scripts.evaluate_cnn --experiment-dir ..\outputs\experiments\cnn_robust_20261009   # resumable; ~3-4 h CPU
..\.venv\Scripts\python -m scripts.evaluate_cnn --experiment-dir ..\outputs\experiments\cnn_robust_20261009 --report-only
..\.venv\Scripts\python -m pytest                    # unit + API + ML tests
..\.venv\Scripts\python -m pytest -m integration     # real-DEAP tests
```
