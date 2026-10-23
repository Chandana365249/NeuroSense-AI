## Baseline reproduction (historical fixed split)

| Level | Accuracy | Balanced acc. | Macro-F1 | ROC AUC | n |
|---|---|---|---|---|---|
| segment_pooled | 0.5481 | 0.5441 | 0.5428 | 0.5365 | 1788 |
| trial_pooled | 0.5917 | 0.5902 | 0.5903 | 0.5647 | 120 |

| Participant | Trial bal. acc. | Trial acc. | Within-person AUC | Predicted-high share | True-high share |
|---|---|---|---|---|---|
| s01 | 0.500 | 0.475 | 0.509 | 1.000 | 0.475 |
| s09 | 0.750 | 0.750 | 0.787 | 0.350 | 0.500 |
| s11 | 0.604 | 0.550 | 0.643 | 0.250 | 0.600 |

Participant mean trial balanced accuracy 0.618, t 95% CI [0.306, 0.930] (n = 3). Saved model re-scored on cached features: max |Δp| = 5.0e-05.

## P1 participant-independent (inductive, N0)

| Variant | Trial bal. acc. [95% CI] | Macro-F1 | Accuracy | Within-person AUC | Range over people | Single-label people | Seed SD | Δ bal. acc. vs RF [bootstrap CI]; Wilcoxon |
|---|---|---|---|---|---|---|---|---|
| Majority-class reference | 0.500 [0.500, 0.500] | 0.343 | 0.527 | 0.500 | 0.500–0.500 | 12.0 | 0.000 |
| Random Forest reference (registered, mean agg.) | 0.533 [0.490, 0.576] | 0.429 | 0.535 | 0.606 | 0.442–0.667 | 5.7 | 0.011 |
| RF (nested: config + aggregation) | 0.532 [0.489, 0.575] | 0.427 | 0.535 | 0.590 | 0.458–0.658 | 5.3 | 0.017 | -0.001 [-0.007, +0.005]; p=0.641; 4/12 better |
| XGB (nested: config + aggregation) | 0.521 [0.488, 0.555] | 0.418 | 0.531 | 0.578 | 0.445–0.664 | 6.7 | 0.007 | -0.012 [-0.057, +0.042]; p=0.322; 3/12 better |
| LR (nested: config + aggregation) | 0.538 [0.498, 0.579] | 0.435 | 0.558 | 0.600 | 0.482–0.700 | 6.0 | 0.000 | +0.005 [-0.045, +0.054]; p=0.770; 5/12 better |
| SVM (nested: config + aggregation) | 0.539 [0.501, 0.577] | 0.428 | 0.542 | 0.624 | 0.495–0.675 | 5.0 | 0.000 | +0.006 [-0.029, +0.039]; p=0.520; 7/12 better |
| **Nested selection over all candidates** | 0.541 [0.504, 0.579] | 0.429 | 0.535 | 0.595 | 0.495–0.675 | 4.7 | 0.002 | +0.008 [-0.024, +0.039]; p=0.465; 7/12 better |

Selections made by the inner loop (outer fold × seed): svm_C0.1+mean 8/36, svm_C0.1+vote 6/36, svm_C0.1+median 6/36, svm_C1+mean 3/36, svm_C1+vote 3/36, svm_C1+median 3/36, lr_C0.01+median 3/36, lr_C0.001+mean 2/36, xgb+median 1/36, rf+vote 1/36

Aggregation (registered RF):

| Variant | Trial bal. acc. [95% CI] | Macro-F1 | Accuracy | Within-person AUC | Range over people | Single-label people | Seed SD |
|---|---|---|---|---|---|---|---|
| mean | 0.533 [0.490, 0.576] | 0.429 | 0.535 | 0.606 | 0.442–0.667 | 5.7 | 0.011 |
| median | 0.526 [0.483, 0.569] | 0.422 | 0.530 | 0.608 | 0.454–0.650 | 5.0 | 0.010 |
| vote | 0.527 [0.481, 0.572] | 0.421 | 0.530 | 0.568 | 0.443–0.650 | 5.0 | 0.011 |
| logit | 0.533 [0.490, 0.576] | 0.429 | 0.535 | 0.606 | 0.442–0.667 | 5.7 | 0.011 |

## P1 unsupervised calibration (N3, per-person z-score)

| Variant | Trial bal. acc. [95% CI] | Macro-F1 | Accuracy | Within-person AUC | Range over people | Single-label people | Seed SD | Δ bal. acc. vs RF [bootstrap CI]; Wilcoxon |
|---|---|---|---|---|---|---|---|---|
| Majority-class reference | 0.500 [0.500, 0.500] | 0.343 | 0.527 | 0.500 | 0.500–0.500 | 12.0 | 0.000 |
| Random Forest reference (registered, mean agg.) | 0.575 [0.525, 0.624] | 0.556 | 0.569 | 0.610 | 0.414–0.683 | 0.0 | 0.008 |
| RF (nested: config + aggregation) | 0.574 [0.523, 0.625] | 0.556 | 0.569 | 0.608 | 0.396–0.683 | 0.0 | 0.008 | -0.000 [-0.007, +0.006]; p=1.000; 4/12 better |
| XGB (nested: config + aggregation) | 0.563 [0.512, 0.613] | 0.534 | 0.560 | 0.588 | 0.425–0.681 | 0.0 | 0.009 | -0.012 [-0.063, +0.042]; p=0.791; 5/12 better |
| LR (nested: config + aggregation) | 0.560 [0.498, 0.622] | 0.554 | 0.558 | 0.593 | 0.398–0.725 | 0.0 | 0.000 | -0.014 [-0.065, +0.041]; p=0.622; 4/12 better |
| SVM (nested: config + aggregation) | 0.525 [0.454, 0.597] | 0.513 | 0.517 | 0.539 | 0.365–0.700 | 0.0 | 0.000 | -0.049 [-0.104, -0.003]; p=0.105; 4/12 better |
| **Nested selection over all candidates** | 0.543 [0.494, 0.593] | 0.535 | 0.540 | 0.563 | 0.398–0.625 | 0.0 | 0.003 | -0.031 [-0.051, -0.009]; p=0.026; 2/12 better |

Selections made by the inner loop (outer fold × seed): lr_C0.001+logit 7/36, rf+mean 6/36, lr_C0.001+mean 6/36, svm_C0.1+vote 3/36, lr_C0.001+vote 3/36, svm_C0.1+mean 3/36, xgb+logit 2/36, xgb+median 2/36, rf+logit 2/36, xgb+vote 1/36, rf+vote 1/36

Aggregation (registered RF):

| Variant | Trial bal. acc. [95% CI] | Macro-F1 | Accuracy | Within-person AUC | Range over people | Single-label people | Seed SD |
|---|---|---|---|---|---|---|---|
| mean | 0.575 [0.525, 0.624] | 0.556 | 0.569 | 0.610 | 0.414–0.683 | 0.0 | 0.008 |
| median | 0.575 [0.523, 0.626] | 0.554 | 0.567 | 0.607 | 0.396–0.683 | 0.0 | 0.003 |
| vote | 0.572 [0.521, 0.624] | 0.552 | 0.564 | 0.601 | 0.396–0.683 | 0.0 | 0.002 |
| logit | 0.575 [0.525, 0.624] | 0.556 | 0.569 | 0.610 | 0.414–0.683 | 0.0 | 0.008 |

RF 5-seed reference (N0): 0.528 [0.489, 0.568]; offset tracking ρ = -0.120. N3 − N0 (RF): +0.042 [-0.013452975478468925, 0.09240042815371764], p = 0.204.

## P1 feature ablation (RF, N0, mean aggregation, seeds 42–44)

| Variant | Trial bal. acc. [95% CI] | Macro-F1 | Accuracy | Within-person AUC | Range over people | Single-label people | Seed SD | Δ vs full |
|---|---|---|---|---|---|---|---|---|
| full (440) | 0.533 [0.490, 0.576] | 0.429 | 0.535 | 0.606 | 0.442–0.667 | 5.7 | 0.011 |
| no_gamma | 0.526 [0.477, 0.575] | 0.448 | 0.540 | 0.600 | 0.384–0.692 | 3.0 | 0.021 | -0.007 [-0.032, +0.016]; p=0.770; 6/12 better |
| theta | 0.503 [0.475, 0.531] | 0.385 | 0.514 | 0.546 | 0.431–0.597 | 7.3 | 0.009 | -0.030 [-0.071, +0.010]; p=0.240; 3/12 better |
| alpha | 0.519 [0.482, 0.557] | 0.425 | 0.508 | 0.584 | 0.443–0.669 | 4.7 | 0.007 | -0.014 [-0.056, +0.029]; p=0.413; 3/12 better |
| beta | 0.543 [0.497, 0.589] | 0.457 | 0.561 | 0.592 | 0.455–0.692 | 4.7 | 0.021 | +0.010 [-0.018, +0.050]; p=1.000; 6/12 better |
| gamma | 0.518 [0.468, 0.569] | 0.408 | 0.524 | 0.573 | 0.375–0.683 | 7.0 | 0.016 | -0.015 [-0.054, +0.035]; p=0.301; 3/12 better |
| non_band | 0.527 [0.484, 0.570] | 0.447 | 0.539 | 0.611 | 0.385–0.635 | 3.3 | 0.009 | -0.006 [-0.037, +0.025]; p=0.770; 5/12 better |
| asymmetry | 0.499 [0.457, 0.541] | 0.361 | 0.472 | 0.500 | 0.326–0.617 | 7.7 | 0.005 | -0.034 [-0.074, +0.005]; p=0.105; 2/12 better |
| frontal | 0.532 [0.479, 0.585] | 0.469 | 0.551 | 0.595 | 0.410–0.720 | 4.0 | 0.001 | -0.001 [-0.047, +0.060]; p=0.375; 3/12 better |
| central | 0.540 [0.483, 0.596] | 0.457 | 0.553 | 0.588 | 0.375–0.667 | 5.3 | 0.010 | +0.007 [-0.027, +0.043]; p=1.000; 5/12 better |
| temporal | 0.525 [0.479, 0.571] | 0.438 | 0.517 | 0.564 | 0.392–0.682 | 4.0 | 0.003 | -0.008 [-0.062, +0.053]; p=0.770; 5/12 better |
| parietal | 0.485 [0.434, 0.537] | 0.362 | 0.466 | 0.512 | 0.331–0.669 | 7.3 | 0.008 | -0.047 [-0.102, -0.004]; p=0.105; 3/12 better |
| occipital | 0.530 [0.506, 0.554] | 0.447 | 0.559 | 0.546 | 0.494–0.627 | 4.0 | 0.009 | -0.003 [-0.044, +0.036]; p=0.970; 7/12 better |
| no_artifact_prone | 0.518 [0.485, 0.551] | 0.399 | 0.512 | 0.597 | 0.424–0.625 | 6.0 | 0.023 | -0.015 [-0.037, +0.008]; p=0.247; 5/12 better |

## P1 regression, S1 and learning curve

| Variant | Trial bal. acc. [95% CI] | Macro-F1 | Accuracy | Within-person AUC | Range over people | Single-label people | Seed SD | Δ vs classifier |
|---|---|---|---|---|---|---|---|---|
| RF regression → threshold 5.0 (N0) | 0.499 [0.489, 0.508] | 0.352 | 0.504 | 0.531 | 0.459–0.528 | 10.0 | 0.008 | -0.034 [-0.071, -0.002]; p=0.074; 2/12 better |
| RF regression → threshold 5.0 (N3) | 0.531 [0.509, 0.554] | 0.456 | 0.552 | 0.597 | 0.469–0.583 | 1.7 | 0.006 | -0.043 [-0.074, -0.012]; p=0.067; 4/12 better |

| Variant | Trial bal. acc. [95% CI] | Macro-F1 | Accuracy | Within-person AUC | Range over people | Single-label people | Seed SD |
|---|---|---|---|---|---|---|---|
| S1 borderline excluded (N0) | 0.531 [0.510, 0.552] | 0.410 | 0.517 | 0.601 | 0.500–0.611 | 6.0 | 0.013 |
| S1 borderline excluded (N3) | 0.591 [0.531, 0.652] | 0.574 | 0.587 | 0.629 | 0.381–0.703 | 0.0 | 0.017 |

| Variant | Trial bal. acc. [95% CI] | Macro-F1 | Accuracy | Within-person AUC | Range over people | Single-label people | Seed SD |
|---|---|---|---|---|---|---|---|
| 3 training participants | 0.516 [0.503, 0.528] | 0.389 | 0.532 | 0.550 | 0.496–0.564 | 8.0 | 0.012 |
| 6 training participants | 0.519 [0.485, 0.553] | 0.410 | 0.528 | 0.570 | 0.395–0.597 | 7.7 | 0.015 |
| 9 training participants | 0.524 [0.493, 0.555] | 0.414 | 0.535 | 0.608 | 0.486–0.667 | 6.3 | 0.009 |
| 11 training participants | 0.533 [0.490, 0.576] | 0.429 | 0.535 | 0.606 | 0.442–0.667 | 5.7 | 0.011 |

## P2 within-participant (supervised personalization, 32 calibration videos)

RF 5-seed reference: 0.618 [0.550, 0.687].

| Variant | Trial bal. acc. [95% CI] | Macro-F1 | Accuracy | Within-person AUC | Range over people | Single-label people | Seed SD | Δ vs RF [bootstrap CI]; Wilcoxon |
|---|---|---|---|---|---|---|---|---|
| Random Forest reference (registered, mean agg.) | 0.618 [0.548, 0.688] | 0.617 | 0.635 | 0.644 | 0.413–0.758 | 0.0 | 0.012 |
| RF (nested) | 0.613 [0.544, 0.683] | 0.613 | 0.631 | 0.642 | 0.413–0.750 | 0.0 | 0.019 | -0.004 [-0.009, +0.000]; p=0.137; 2/12 better |
| XGB (nested) | 0.588 [0.517, 0.659] | 0.584 | 0.615 | 0.615 | 0.425–0.758 | 0.0 | 0.006 | -0.030 [-0.055, -0.009]; p=0.042; 3/12 better |
| LR (nested) | 0.634 [0.568, 0.699] | 0.632 | 0.643 | 0.655 | 0.465–0.800 | 0.0 | 0.012 | +0.016 [-0.010, +0.044]; p=0.520; 7/12 better |
| SVM (nested) | 0.635 [0.574, 0.696] | 0.627 | 0.633 | 0.669 | 0.443–0.750 | 0.0 | 0.007 | +0.017 [-0.016, +0.063]; p=0.835; 7/12 better |
| **Nested selection over all candidates** | 0.620 [0.559, 0.681] | 0.618 | 0.626 | 0.645 | 0.452–0.747 | 0.0 | 0.016 | +0.002 [-0.022, +0.031]; p=0.622; 4/12 better |

Selections: lr_C0.001+mean 21/180, svm_C0.1+mean 20/180, lr_C0.001+median 18/180, svm_C0.1+median 15/180, rf+mean 15/180, lr_C0.01+median 13/180, lr_C0.1+mean 12/180, rf+median 8/180, lr_C0.01+mean 7/180, lr_C0.1+median 6/180, lr_C0.01+vote 5/180, xgb+mean 5/180, xgb+median 5/180, svm_C1+median 5/180, lr_C0.1+logit 4/180, svm_C1+mean 4/180, lr_C0.01+logit 3/180, lr_C0.1+vote 3/180, lr_C0.001+logit 2/180, svm_C0.1+vote 2/180, lr_C0.001+vote 2/180, xgb+vote 2/180, xgb+logit 2/180, rf+vote 1/180

Fixed configurations (mean aggregation):

| Variant | Trial bal. acc. [95% CI] | Macro-F1 | Accuracy | Within-person AUC | Range over people | Single-label people | Seed SD |
|---|---|---|---|---|---|---|---|
| rf | 0.618 [0.548, 0.688] | 0.617 | 0.635 | 0.644 | 0.413–0.758 | 0.0 | 0.012 |
| xgb | 0.584 [0.510, 0.657] | 0.579 | 0.612 | 0.617 | 0.416–0.767 | 0.0 | 0.011 |
| lr_C0.001 | 0.642 [0.576, 0.709] | 0.641 | 0.650 | 0.666 | 0.482–0.817 | 0.0 | 0.013 |
| lr_C0.01 | 0.620 [0.552, 0.687] | 0.618 | 0.631 | 0.652 | 0.438–0.808 | 0.0 | 0.020 |
| lr_C0.1 | 0.614 [0.539, 0.688] | 0.612 | 0.628 | 0.646 | 0.409–0.817 | 0.0 | 0.017 |
| svm_C0.1 | 0.640 [0.581, 0.698] | 0.627 | 0.636 | 0.672 | 0.458–0.758 | 0.0 | 0.021 |
| svm_C1 | 0.630 [0.560, 0.699] | 0.627 | 0.640 | 0.656 | 0.468–0.767 | 0.0 | 0.006 |

Aggregation (RF):

| Variant | Trial bal. acc. [95% CI] | Macro-F1 | Accuracy | Within-person AUC | Range over people | Single-label people | Seed SD |
|---|---|---|---|---|---|---|---|
| mean | 0.618 [0.548, 0.688] | 0.617 | 0.635 | 0.644 | 0.413–0.758 | 0.0 | 0.012 |
| median | 0.618 [0.549, 0.688] | 0.618 | 0.635 | 0.639 | 0.427–0.767 | 0.0 | 0.015 |
| vote | 0.617 [0.548, 0.687] | 0.616 | 0.633 | 0.639 | 0.427–0.767 | 0.0 | 0.014 |
| logit | 0.612 [0.544, 0.680] | 0.611 | 0.631 | 0.645 | 0.419–0.750 | 0.0 | 0.010 |

## P2 feature ablation (RF)

| Variant | Trial bal. acc. [95% CI] | Macro-F1 | Accuracy | Within-person AUC | Range over people | Single-label people | Seed SD | Δ vs full |
|---|---|---|---|---|---|---|---|---|
| full (440) | 0.618 [0.548, 0.688] | 0.617 | 0.635 | 0.644 | 0.413–0.758 | 0.0 | 0.012 |
| no_gamma | 0.614 [0.558, 0.670] | 0.613 | 0.633 | 0.640 | 0.458–0.742 | 0.0 | 0.022 | -0.004 [-0.021, +0.014]; p=0.691; 6/12 better |
| theta | 0.590 [0.531, 0.649] | 0.588 | 0.610 | 0.616 | 0.385–0.717 | 0.0 | 0.021 | -0.028 [-0.062, +0.009]; p=0.204; 4/12 better |
| alpha | 0.579 [0.524, 0.635] | 0.576 | 0.601 | 0.624 | 0.493–0.750 | 0.0 | 0.009 | -0.039 [-0.083, +0.007]; p=0.151; 3/12 better |
| beta | 0.600 [0.532, 0.668] | 0.598 | 0.618 | 0.633 | 0.405–0.767 | 0.0 | 0.020 | -0.018 [-0.034, +0.001]; p=0.042; 2/12 better |
| gamma | 0.620 [0.548, 0.692] | 0.620 | 0.636 | 0.646 | 0.448–0.775 | 0.0 | 0.012 | +0.002 [-0.014, +0.018]; p=0.970; 7/12 better |
| non_band | 0.616 [0.559, 0.674] | 0.615 | 0.634 | 0.637 | 0.466–0.758 | 0.0 | 0.019 | -0.002 [-0.025, +0.020]; p=1.000; 5/12 better |
| asymmetry | 0.616 [0.533, 0.700] | 0.616 | 0.635 | 0.638 | 0.387–0.792 | 0.0 | 0.022 | -0.001 [-0.029, +0.024]; p=0.791; 6/12 better |
| frontal | 0.590 [0.526, 0.654] | 0.589 | 0.608 | 0.616 | 0.406–0.717 | 0.0 | 0.006 | -0.028 [-0.052, -0.005]; p=0.077; 3/12 better |
| central | 0.612 [0.544, 0.680] | 0.611 | 0.630 | 0.628 | 0.389–0.800 | 0.0 | 0.032 | -0.006 [-0.048, +0.036]; p=0.791; 6/12 better |
| temporal | 0.573 [0.522, 0.623] | 0.570 | 0.582 | 0.593 | 0.440–0.677 | 0.0 | 0.015 | -0.045 [-0.087, -0.001]; p=0.129; 4/12 better |
| parietal | 0.595 [0.531, 0.658] | 0.591 | 0.611 | 0.621 | 0.372–0.717 | 0.0 | 0.017 | -0.023 [-0.056, +0.011]; p=0.266; 5/12 better |
| occipital | 0.609 [0.533, 0.685] | 0.606 | 0.623 | 0.632 | 0.406–0.783 | 0.0 | 0.009 | -0.009 [-0.061, +0.041]; p=0.985; 7/12 better |
| no_artifact_prone | 0.609 [0.536, 0.682] | 0.608 | 0.629 | 0.649 | 0.406–0.758 | 0.0 | 0.007 | -0.009 [-0.034, +0.013]; p=0.713; 4/12 better |

## P2 regression and S1

| Variant | Trial bal. acc. [95% CI] | Macro-F1 | Accuracy | Within-person AUC | Range over people | Single-label people | Seed SD | Δ vs classifier |
|---|---|---|---|---|---|---|---|---|
| RF regression → threshold 5.0 | 0.564 [0.497, 0.631] | 0.536 | 0.603 | 0.647 | 0.462–0.775 | 0.3 | 0.022 | -0.054 [-0.103, -0.008]; p=0.176; 5/12 better |

| Variant | Trial bal. acc. [95% CI] | Macro-F1 | Accuracy | Within-person AUC | Range over people | Single-label people | Seed SD |
|---|---|---|---|---|---|---|---|
| S1 borderline excluded | 0.635 [0.567, 0.703] | 0.635 | 0.660 | 0.661 | 0.487–0.798 | 0.0 | 0.033 |

Label noise: pooled accuracy on borderline trials 0.589 vs clear trials 0.646; point-biserial r(|valence − 5|, correct) = 0.074, p = 0.0049.

## P3 calibration (population + personal), RF, seeds 42–44

| Variant | Trial bal. acc. [95% CI] | Macro-F1 | Accuracy | Within-person AUC | Range over people | Single-label people | Seed SD |
|---|---|---|---|---|---|---|---|
| pooled, 0 videos, weighting equal | 0.535 [0.492, 0.578] | 0.521 | 0.535 | 0.549 | 0.412–0.667 | 0.0 | 0.005 |
| pooled, 8 videos, weighting equal | 0.576 [0.532, 0.620] | 0.569 | 0.579 | 0.603 | 0.479–0.725 | 0.0 | 0.023 |
| pooled, 16 videos, weighting equal | 0.594 [0.530, 0.659] | 0.592 | 0.608 | 0.618 | 0.420–0.742 | 0.0 | 0.022 |
| pooled, 32 videos, weighting equal | 0.601 [0.527, 0.675] | 0.600 | 0.622 | 0.635 | 0.389–0.767 | 0.0 | 0.019 |
| pooled, 32 videos, weighting none | 0.590 [0.512, 0.668] | 0.587 | 0.612 | 0.620 | 0.351–0.783 | 0.0 | 0.018 |
| personal only, 8 videos | 0.562 [0.518, 0.606] | 0.554 | 0.581 | 0.584 | 0.451–0.667 | 0.0 | 0.023 |
| personal only, 16 videos | 0.601 [0.542, 0.659] | 0.597 | 0.621 | 0.614 | 0.444–0.733 | 0.0 | 0.023 |
| personal only, 24 videos | 0.603 [0.538, 0.667] | 0.600 | 0.619 | 0.627 | 0.434–0.708 | 0.0 | 0.023 |
| personal only, 32 videos | 0.618 [0.548, 0.688] | 0.617 | 0.635 | 0.644 | 0.413–0.758 | 0.0 | 0.012 |

| Calibration videos | Pooled − zero-shot | Personal-only − zero-shot | Pooled − personal-only |
|---|---|---|---|
| 8 | +0.041 [-0.009, +0.097], p=0.457, 7/12 better | +0.027 [-0.026, +0.084], p=0.677, 6/12 better | +0.014 [-0.005, +0.033], p=0.233, 7/12 better |
| 16 | +0.059 [-0.009, +0.126], p=0.064, 10/12 better | +0.066 [+0.002, +0.132], p=0.129, 7/12 better | -0.007 [-0.018, +0.004], p=0.520, 6/12 better |
| 32 | +0.066 [-0.009, +0.140], p=0.129, 8/12 better | +0.083 [+0.011, +0.155], p=0.077, 8/12 better | -0.017 [-0.027, -0.006], p=0.023, 3/12 better |

## Primary comparisons (Holm over 3)

| Comparison | Δ bal. acc. | Bootstrap CI | Wilcoxon p | Holm p |
|---|---|---|---|---|
| P1_nested_vs_rf | +0.008 | [-0.024, +0.039] | 0.465 | 0.930 |
| P2_nested_vs_rf | +0.002 | [-0.022, +0.031] | 0.622 | 0.930 |
| P3_pooled32_vs_zero_shot | +0.066 | [-0.009, +0.140] | 0.129 | 0.388 |

Leakage demonstration (invalid by design, segment split within person): segment accuracy 0.731 [0.697, 0.764], trial balanced accuracy 0.805 [0.776, 0.834].

## Per-participant trial balanced accuracy (seed-averaged)

| Participant | P1 RF (N0) | P1 nested (N0) | P1 RF (N3) | P2 RF | P2 nested | P3 pooled 32 |
|---|---|---|---|---|---|---|
| s01 | 0.500 | 0.529 | 0.580 | 0.501 | 0.452 | 0.508 |
| s02 | 0.476 | 0.596 | 0.517 | 0.666 | 0.622 | 0.658 |
| s03 | 0.508 | 0.500 | 0.572 | 0.705 | 0.743 | 0.705 |
| s05 | 0.500 | 0.542 | 0.632 | 0.656 | 0.653 | 0.611 |
| s07 | 0.528 | 0.500 | 0.661 | 0.413 | 0.528 | 0.389 |
| s08 | 0.486 | 0.495 | 0.652 | 0.636 | 0.599 | 0.609 |
| s09 | 0.667 | 0.675 | 0.683 | 0.650 | 0.625 | 0.658 |
| s11 | 0.559 | 0.507 | 0.545 | 0.472 | 0.542 | 0.438 |
| s15 | 0.592 | 0.625 | 0.575 | 0.758 | 0.733 | 0.767 |
| s16 | 0.442 | 0.500 | 0.582 | 0.671 | 0.669 | 0.631 |
| s22 | 0.638 | 0.527 | 0.484 | 0.746 | 0.747 | 0.722 |
| s25 | 0.500 | 0.500 | 0.414 | 0.541 | 0.526 | 0.516 |
