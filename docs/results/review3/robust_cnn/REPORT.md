# Robust CNN evaluation (2026-10-08)

Task: Binary valence: high_valence if self-reported valence > 5.0 (1-9 scale), else low_valence  
Model: EEGNet-style CNN, 2130 parameters  
Dataset: 12 DEAP participants, 7046 segments (fingerprint 3c5359083bc90ae5)  
Split: train s02, s03, s05, s07, s08, s15, s22 · validation s16, s25 · test s01, s09, s11  
Seeds: [42, 43, 44, 45, 46] · code commit 755345c · pytorch 2.14.1+cpu

All numbers are generated from the run records; nothing was selected using test or held-out participants.

## Comparability and reproduction checks

- same split as random forest: **True**
- same dataset fingerprint: **True**
- same lopo participants: **True**
- random forest predictions reproduce registered metrics: **True**
- seed 42 rerun reproduces registered CNN `cnn_valence_binary_20261006-045706` exactly: **True** (best epoch 2 vs 2)

## 1. Repeated seeds, fixed split (held-out test participants)

Mean ± sample SD over seeds with a t-based 95% CI. All seeds are scored on the same 3 test participants, so this shows sensitivity to the training seed, not variability across people.

### Test, per 4 s segment

| Model | Accuracy | Balanced acc. | Macro-F1 | ROC-AUC | 95% CI of macro-F1 |
|---|---|---|---|---|---|
| CNN, training-set normalization (5 seeds) | 0.532 ± 0.033 | 0.531 ± 0.026 | 0.466 ± 0.109 | 0.548 ± 0.057 | [0.331, 0.601] |
| CNN, per-participant normalization (5 seeds) | 0.484 ± 0.021 | 0.500 ± 0.021 | 0.422 ± 0.067 | 0.482 ± 0.047 | [0.339, 0.504] |
| Random Forest (registered, deterministic) | 0.548 | 0.544 | 0.543 | 0.537 | see §3 |
| Majority-class reference | 0.525 | 0.500 | 0.344 | 0.500 | |

### Test, per trial

| Model | Accuracy | Balanced acc. | Macro-F1 | ROC-AUC | 95% CI of macro-F1 |
|---|---|---|---|---|---|
| CNN, training-set normalization (5 seeds) | 0.532 ± 0.035 | 0.530 ± 0.030 | 0.459 ± 0.116 | 0.555 ± 0.078 | [0.315, 0.603] |
| CNN, per-participant normalization (5 seeds) | 0.482 ± 0.030 | 0.498 ± 0.031 | 0.408 ± 0.082 | 0.488 ± 0.062 | [0.307, 0.510] |
| Random Forest (registered, deterministic) | 0.592 | 0.590 | 0.590 | 0.565 | see §3 |
| Majority-class reference | 0.525 | 0.500 | 0.344 | 0.500 | |

### Each seed

| Normalization | Seed | Best epoch / run | Val. macro-F1 (selection) | Test seg. acc. | Test seg. macro-F1 | Test seg. AUC | Test trial acc. | Test trial AUC | Trial AUC perm. p |
|---|---|---|---|---|---|---|---|---|---|
| train_stats | 42 | 2 / 12 | 0.558 | 0.542 | 0.525 | 0.577 | 0.542 | 0.590 | 0.026 |
| train_stats | 43 | 2 / 12 | 0.548 | 0.532 | 0.378 | 0.574 | 0.525 | 0.605 | 0.002 |
| train_stats | 44 | 4 / 14 | 0.536 | 0.554 | 0.554 | 0.572 | 0.567 | 0.578 | 0.098 |
| train_stats | 45 | 16 / 26 | 0.528 | 0.475 | 0.322 | 0.445 | 0.475 | 0.417 | 0.955 |
| train_stats | 46 | 2 / 12 | 0.584 | 0.556 | 0.553 | 0.571 | 0.550 | 0.585 | 0.055 |
| per_participant | 42 | 10 / 20 | 0.532 | 0.477 | 0.350 | 0.449 | 0.475 | 0.448 | 0.483 |
| per_participant | 43 | 23 / 33 | 0.475 | 0.475 | 0.351 | 0.446 | 0.475 | 0.445 | 0.832 |
| per_participant | 44 | 4 / 14 | 0.536 | 0.521 | 0.494 | 0.561 | 0.533 | 0.596 | 0.061 |
| per_participant | 45 | 11 / 21 | 0.554 | 0.476 | 0.452 | 0.482 | 0.467 | 0.482 | 0.627 |
| per_participant | 46 | 2 / 12 | 0.541 | 0.471 | 0.460 | 0.470 | 0.458 | 0.469 | 0.192 |

### Train / validation / test consistency (segment macro-F1, selected checkpoint in eval mode)

| Normalization | Train | Validation | Test (final refit) | Best epoch |
|---|---|---|---|---|
| train_stats | 0.588 ± 0.026 | 0.551 ± 0.022 | 0.466 ± 0.109 | 5.200 ± 6.099 |
| per_participant | 0.649 ± 0.068 | 0.527 ± 0.030 | 0.422 ± 0.067 | 10.000 ± 8.216 |

### Per test participant (segment level, mean ± SD over seeds)

'Predicted high_valence' is the share of the participant's segments assigned to high_valence; a value of 0 or 1 means the model gave every segment of that person the same label.

| Participant | Model | Accuracy | Balanced acc. | Macro-F1 | ROC-AUC | Predicted high_valence | Self-reported high_valence |
|---|---|---|---|---|---|---|---|
| s01 | Random Forest | 0.478 | 0.501 | 0.368 | 0.502 | 0.943 | 0.475 |
| s01 | CNN, training-set normalization | 0.516 ± 0.020 | 0.501 ± 0.002 | 0.344 ± 0.001 | 0.505 ± 0.048 | 0.196 ± 0.439 | 0.475 |
| s01 | CNN, per-participant normalization | 0.518 ± 0.015 | 0.511 ± 0.009 | 0.423 ± 0.045 | 0.524 ± 0.028 | 0.370 ± 0.411 | 0.475 |
| s09 | Random Forest | 0.643 | 0.643 | 0.641 | 0.696 | 0.431 | 0.503 |
| s09 | CNN, training-set normalization | 0.522 ± 0.022 | 0.521 ± 0.022 | 0.425 ± 0.099 | 0.569 ± 0.064 | 0.591 ± 0.424 | 0.503 |
| s09 | CNN, per-participant normalization | 0.501 ± 0.012 | 0.505 ± 0.011 | 0.357 ± 0.037 | 0.454 ± 0.101 | 0.026 ± 0.032 | 0.503 |
| s11 | Random Forest | 0.524 | 0.551 | 0.523 | 0.574 | 0.372 | 0.598 |
| s11 | CNN, training-set normalization | 0.558 ± 0.087 | 0.500 ± 0.000 | 0.359 ± 0.040 | 0.527 ± 0.047 | 0.797 ± 0.446 | 0.598 |
| s11 | CNN, per-participant normalization | 0.432 ± 0.061 | 0.501 ± 0.006 | 0.340 ± 0.091 | 0.512 ± 0.026 | 0.150 ± 0.302 | 0.598 |

### Summed segment confusion matrices over seeds

Rows = self-reported, columns = predicted, order high_valence, low_valence.

- train_stats: `[[2610, 2085], [2100, 2145]]`
- per_participant: `[[856, 3839], [774, 3471]]`

## 2. Nested leave-one-participant-out (9 non-test participants)

For each held-out participant, early stopping uses 2 inner validation participants from the other 8; the model is refit on all 8 for the selected epoch count and scored once on the held-out person. The 3 test participants are never used. Mean ± sample SD over the 9 folds, t-based 95% CI.

| Model | Seg. accuracy | Seg. balanced acc. | Seg. macro-F1 | Seg. ROC-AUC | Trial accuracy | 95% CI seg. accuracy |
|---|---|---|---|---|---|---|
| CNN, training-set normalization | 0.503 ± 0.080 | 0.494 ± 0.023 | 0.383 ± 0.065 | 0.506 ± 0.102 | 0.503 ± 0.085 | [0.441, 0.565] |
| CNN, per-participant normalization | 0.552 ± 0.074 | 0.505 ± 0.016 | 0.408 ± 0.051 | 0.550 ± 0.136 | 0.561 ± 0.074 | [0.495, 0.609] |
| Random Forest (registered CV, no inner validation needed) | 0.510 ± 0.091 | n/a | 0.416 ± 0.053 | 0.538 ± 0.085 | 0.503 ± 0.100 | [0.440, 0.580] |
| Majority-class reference | 0.528 ± 0.093 | 0.500 ± 0.000 | 0.343 ± 0.040 | 0.500 ± 0.000 | | |

### Per fold (segment level)

| Held-out | Inner validation | train_stats acc. / F1 / AUC / best ep. | per_participant acc. / F1 / AUC / best ep. | RF acc. / F1 | Majority acc. |
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

### Tests across folds (Wilcoxon signed-rank, n = 9, two-sided; low power)

| Comparison | Median difference | p |
|---|---|---|
| train_stats: balanced accuracy vs 0.5 | 0.000 | 0.625 |
| train_stats: accuracy vs majority reference | 0.000 | 0.562 |
| train_stats: accuracy vs Random Forest | -0.015 | 0.844 |
| train_stats: macro-F1 vs Random Forest | -0.005 | 0.496 |
| per_participant: balanced accuracy vs 0.5 | 0.000 | 0.203 |
| per_participant: accuracy vs majority reference | -0.002 | 0.734 |
| per_participant: accuracy vs Random Forest | 0.029 | 0.164 |
| per_participant: macro-F1 vs Random Forest | -0.038 | 0.734 |

## 3. Uncertainty on the held-out test participants

95% CIs from resampling whole trials within each participant; p-values from permuting trial labels within each participant (trial level). With 3 test people these describe uncertainty over their trials only.

| Model | Seg. macro-F1 [95% CI] | Trial accuracy [95% CI] | Trial ROC-AUC [95% CI] | p (trial macro-F1) | p (trial AUC) |
|---|---|---|---|---|---|
| Random Forest `valence_binary-random_forest-20261008-180558` | 0.543 [0.485, 0.601] | 0.592 [0.508, 0.683] | 0.565 [0.459, 0.669] | < 0.001 | < 0.001 |
| CNN, training-set normalization, seed 42 | 0.525 [0.434, 0.615] | 0.542 [0.450, 0.633] | 0.590 [0.487, 0.693] | 1.000 | 0.026 |
| CNN, training-set normalization, seed 43 | 0.378 [0.335, 0.417] | 0.525 [0.433, 0.617] | 0.605 [0.498, 0.705] | 1.000 | 0.002 |
| CNN, training-set normalization, seed 44 | 0.554 [0.480, 0.633] | 0.567 [0.483, 0.658] | 0.578 [0.479, 0.685] | 0.226 | 0.098 |
| CNN, training-set normalization, seed 45 | 0.322 [0.278, 0.361] | 0.475 [0.383, 0.567] | 0.417 [0.320, 0.524] | 1.000 | 0.955 |
| CNN, training-set normalization, seed 46 | 0.553 [0.474, 0.628] | 0.550 [0.458, 0.642] | 0.585 [0.480, 0.686] | 0.493 | 0.055 |
| CNN, per-participant normalization, seed 42 | 0.350 [0.309, 0.393] | 0.475 [0.392, 0.567] | 0.448 [0.350, 0.550] | 1.000 | 0.483 |
| CNN, per-participant normalization, seed 43 | 0.351 [0.307, 0.394] | 0.475 [0.383, 0.567] | 0.445 [0.343, 0.546] | 1.000 | 0.832 |
| CNN, per-participant normalization, seed 44 | 0.494 [0.420, 0.566] | 0.533 [0.442, 0.625] | 0.596 [0.492, 0.698] | 0.408 | 0.061 |
| CNN, per-participant normalization, seed 45 | 0.452 [0.379, 0.521] | 0.467 [0.375, 0.550] | 0.482 [0.379, 0.587] | 0.562 | 0.627 |
| CNN, per-participant normalization, seed 46 | 0.460 [0.385, 0.536] | 0.458 [0.375, 0.550] | 0.469 [0.365, 0.571] | 1.000 | 0.192 |

## Figures

- `learning_curves.png`: learning curves
- `seed_distribution.png`: seed distribution
- `per_participant.png`: per participant
- `lopo_folds.png`: lopo folds
