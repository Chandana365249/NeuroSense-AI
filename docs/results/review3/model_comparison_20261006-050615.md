# Model comparison (20261006-050615)

Task: Binary valence: high_valence if self-reported valence > 5.0 (1-9 scale), else low_valence  
Test participants: s01, s09, s11 (1788 segments, 120 trials)

## Test, per 4 s segment

| Model | Accuracy | Balanced acc. | Macro F1 | ROC-AUC |
|---|---|---|---|---|
| Random Forest | 0.548 | 0.544 | 0.543 | 0.537 |
| SVM (RBF) | 0.498 | 0.486 | 0.460 | 0.433 |
| XGBoost | 0.500 | 0.499 | 0.499 | 0.478 |
| CNN (EEGNet-style) | 0.542 | 0.534 | 0.525 | 0.577 |
| Majority-class reference | 0.525 | 0.500 | 0.344 | 0.500 |

## Test, per trial

| Model | Accuracy | Balanced acc. | Macro F1 | ROC-AUC |
|---|---|---|---|---|
| Random Forest | 0.592 | 0.590 | 0.590 | 0.565 |
| SVM (RBF) | 0.475 | 0.459 | 0.406 | 0.421 |
| XGBoost | 0.517 | 0.515 | 0.515 | 0.471 |
| CNN (EEGNet-style) | 0.542 | 0.533 | 0.524 | 0.590 |
| Majority-class reference | 0.525 | 0.500 | 0.344 | 0.500 |

## Validation participants, per segment (models fit on train only)

| Model | Accuracy | Balanced acc. | Macro F1 | ROC-AUC |
|---|---|---|---|---|
| Random Forest | 0.462 | 0.515 | 0.420 | 0.536 |
| SVM (RBF) | 0.425 | 0.500 | 0.298 | 0.488 |
| XGBoost | 0.457 | 0.520 | 0.385 | 0.622 |
| CNN (EEGNet-style) | 0.559 | 0.562 | 0.558 | 0.582 |
| Majority-class reference | 0.425 | 0.500 | 0.298 | 0.500 |
