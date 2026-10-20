# NeuroSense AI: Review 3 Progress (first ~30%)

**Scope of this milestone:**
- a deep-learning foundation;
- a real CNN;
- participant-aware evaluation;
- model-registry integration.

LSTM, CNN-LSTM, Transformer, XAI and deployment are deliberately **not** started.

> **Update (Phase A, robust CNN evaluation):** the single-run CNN numbers below were re-examined with 5 seeds, nested leave-one-participant-out cross-validation, two normalizations, bootstrap CIs and permutation tests. See [REVIEW3_CNN_ROBUST_EVALUATION.md](REVIEW3_CNN_ROBUST_EVALUATION.md).
> - The registered run is reproduced exactly, but it is one of the better seeds: 5-seed test macro-F1 is 0.466 ± 0.109.
> - In LOPO the CNN is at chance (balanced accuracy 0.494 ± 0.023).

**Baseline:** `review2.1`. Work is on the branch `review3/deep-learning-foundation`.

> This is a research prototype. All numbers below come from real DEAP recordings and participants the model never saw during training. They estimate self-reported valence and are not a diagnosis.

## 1. Why Review 3 introduces deep learning

The Review 2 baseline uses 440 hand-crafted features (band powers, asymmetry, Hjorth parameters, entropy). That limits it to what those features capture. A convolutional network can instead learn its own temporal (frequency) and spatial (electrode) filters directly from the preprocessed EEG. It also gives a natural base for the planned CNN-LSTM and for XAI methods that work on the raw signal.

## 2. Limitations of the Review 2 baseline

- **Near chance across people.** Held-out test accuracy is 0.548 (macro-F1 0.543), and leave-one-participant-out cross-validation gives 0.51 ± 0.09.
- **Fixed features.** The features are designed by hand and every segment is summarized the same way; nothing is learned from the waveform.
- **Small, varied data.** There are only 12 participants, with strong differences between them (see the two-peaked feature distributions in Review 2).

## 3. Why a CNN first

- A compact CNN is the standard deep baseline for EEG classification and the simplest real deep model.
- Its convolutional front end is reused unchanged by a later CNN-LSTM.
- It trains in minutes on a CPU, so the evaluation protocol could be kept identical without compromises.

## 4. CNN architecture (`backend/app/ml/deep/cnn_model.py`)

`EEGCNN`, based on EEGNet (Lawhern et al., 2018, *J. Neural Eng.* 15:056013). It was chosen as a well-known small baseline, **not** as an optimal design.

| stage | layers | output shape | parameters |
|---|---|---|---|
| input | one z-scored 4 s segment | (1, 32, 512) | 0 |
| temporal | zero-pad + Conv2d(1→8, kernel 1×64) + BatchNorm | (8, 32, 512) | 528 |
| spatial | depthwise Conv2d(8→16, kernel 32×1, groups=8) + BatchNorm + ELU + AvgPool(1×4) + Dropout(0.5) | (16, 1, 128) | 544 |
| separable | depthwise Conv2d(16, 1×16) + pointwise Conv2d(16→16) + BatchNorm + ELU + AvgPool(1×8) + Dropout(0.5) | (16, 1, 16) | 544 |
| classifier | Flatten + Linear(256→2) | 2 logits | 514 |
| **total** | | | **2,130** |

**Output classes:** `high_valence`, `low_valence`, in the same order as the baseline.

**Hyperparameters:** all of them live in `CNNConfig` (filters, kernel sizes, pooling, dropout). The input shape is validated and weight initialization is seeded.

The old `EEGCNN` was not a CNN: it was three `Linear` layers on a 9-value feature vector, with no convolution, no pooling and no access to the signal. It has been replaced. The unused `lstm_model.py` is unchanged, as it is out of scope for this milestone.

## 5. Input representation (`backend/app/ml/deep/data.py`)

- **Source:** the same `EEGPreprocessor` output used for the baseline features (3 s baseline removed, 4–45 Hz zero-phase band-pass, robust clipping, non-overlapping 4 s windows, artifact-heavy windows rejected).
- **Example:** one segment = 32 EEG channels (DEAP Geneva order) × 512 samples (128 Hz), float32. The tensor is (batch, 1, 32, 512): channels as rows, time as columns.
- **Rows:** 7,046 segments (154 of 7,200 rejected). These are verified identical, row for row (participant, trial, segment, ratings), to the Random Forest feature table.
- **Normalization:** per-channel mean and standard deviation computed **only from the participants the model is fitted on**, then stored with the model.
- **Labels:** binary valence, high if self-reported valence > 5 on the 1–9 scale. This is an operational label, unchanged from Review 2. Classes are encoded as sorted class names.
- **Plumbing:** `EEGSegmentDataset` and a seeded `DataLoader`. Segments are cached once as a memory-mapped `.npy` file (about 440 MB, git-ignored).

## 6. Training strategy (`backend/app/ml/deep/training.py`)

| setting | value |
|---|---|
| framework | PyTorch 2.14.1 (CPU build), deterministic algorithms, 4 threads, seed 42 |
| optimizer | AdamW, learning rate 1e-3, weight decay 1e-2, batch 64 |
| loss | cross-entropy with balanced class weights |
| selection | validation macro-F1 each epoch (ties broken by lower validation loss), early stopping with patience 10, max 60 epochs |
| checkpoints | `checkpoints/best.pt` and `last.pt` in the model folder |
| final model | retrained from the same seed on train + validation participants for the selected number of epochs, mirroring the baseline's refit |
| registry | `outputs/models/cnn_valence_binary_<timestamp>/model.pt` (weights-only) plus `metadata.json` with SHA-256, input shape, normalization, architecture, split, history and metrics. **Registered but not activated.** |

**Real run:** `cnn_valence_binary_20261006-045706`.
- **Epochs:** 12 run; epoch 2 selected (validation macro-F1 0.558); early stopping after 10 epochs with no improvement.
- **Final fit:** 2 epochs on train + validation.
- **Time:** 449 s on CPU.

From the curves (`docs/results/review3/*_loss.png`, `*_metrics.png`):
- training loss falls steadily (0.69 → 0.61) and training macro-F1 rises to about 0.64;
- validation loss stays noisy and higher;
- after epoch 2, validation macro-F1 drops to about 0.37 because the model mostly predicts one class for the validation participants.

So the network learns patterns specific to the training participants that do not transfer to new people.

## 7. Participant-aware evaluation (unchanged from Review 2)

Participants are split with `split_participants(seed 42)`; every participant's segments stay in one split.

| split | participants | segments |
|---|---|---|
| train | s02, s03, s05, s07, s08, s15, s22 | 4,073 |
| validation | s16, s25 | 1,185 |
| test (held out) | s01, s09, s11 | 1,788 (120 trials) |

The test participants were used only for the single final evaluation. No architecture, hyperparameter or epoch choice was based on them.

## 8. Results (real DEAP, held-out test participants)

The CNN on test participants, evaluated once:

| level | accuracy | balanced acc. | precision | recall | macro-F1 | ROC-AUC | n |
|---|---|---|---|---|---|---|---|
| segment | 0.542 | 0.534 | 0.538 | 0.534 | 0.525 | 0.577 | 1788 |
| trial (segment probabilities averaged) | 0.542 | 0.533 | 0.538 | 0.533 | 0.524 | 0.590 | 120 |

Segment confusion matrix (rows = self-reported, columns = predicted, order high/low): `[[654, 285], [534, 315]]`. The model leans towards predicting high valence.

The CNN on validation participants (train-only model, selected epoch) reached macro-F1 0.558. This is a **validation** score used for model selection, not a final result.

## 9. Comparison with the baseline

Produced by `scripts/compare_models.py`. Before comparing, the script checks that the split, label rule, dataset fingerprint and test set are identical.
- **Random Forest and CNN:** values come from their registered artifacts.
- **SVM and XGBoost:** refit with their unchanged Review 2 settings on train + validation and scored once. These test values are for reporting only.

**Test participants, per 4 s segment (n = 1788):**

| model | accuracy | balanced acc. | macro-F1 | ROC-AUC |
|---|---|---|---|---|
| Random Forest (Review 2 production) | **0.548** | **0.544** | **0.543** | 0.537 |
| SVM (RBF) | 0.498 | 0.486 | 0.460 | 0.433 |
| XGBoost | 0.500 | 0.499 | 0.499 | 0.478 |
| **CNN (EEGNet-style)** | 0.542 | 0.534 | 0.525 | **0.577** |
| majority-class reference | 0.525 | 0.500 | 0.344 | 0.500 |

**Test participants, per trial (n = 120):**

| model | accuracy | balanced acc. | macro-F1 | ROC-AUC |
|---|---|---|---|---|
| Random Forest | **0.592** | **0.590** | **0.590** | 0.565 |
| SVM (RBF) | 0.475 | 0.459 | 0.406 | 0.421 |
| XGBoost | 0.517 | 0.515 | 0.514 | 0.471 |
| CNN | 0.542 | 0.533 | 0.524 | **0.590** |

**Validation participants, per segment (fit on train only), used for selection:**

| model | Random Forest | SVM | XGBoost | CNN | majority |
|---|---|---|---|---|---|
| macro-F1 | 0.420 | 0.298 | 0.385 | 0.558 | 0.298 |

**Interpretation:**
- The CNN does **not** beat the Random Forest on held-out participants. Its accuracy is similar and its macro-F1 is lower.
- It has the best ranking quality (ROC-AUC 0.577 per segment, 0.590 per trial), which suggests its probability scores carry some signal but its decision threshold is skewed towards high valence.
- Its clear lead on the 2 validation participants did not carry over to the 3 test participants. With this few people per split, rankings between models are unstable.
- All models are within a few points of chance. The Random Forest stays the active production model.

The full outputs are in `docs/results/review3/` (`model_comparison_*.md`, `.json`, `.png`).

## 10. Limitations

- **Small data:** 12 of 32 DEAP participants; 2 validation and 3 test participants, so the score differences between models are within the expected noise.
- **Single split, no repeats:** sections 6–9 describe one split and one seed. This has since been addressed in [REVIEW3_CNN_ROBUST_EVALUATION.md](REVIEW3_CNN_ROBUST_EVALUATION.md).
- **Quick early stop:** stopping at epoch 2 shows validation macro-F1 is a noisy signal with only 2 validation participants.
- **No subject adaptation:** there is no per-participant normalization or domain adaptation, which is the most likely cause of the poor transfer.
- **Not tuned:** hyperparameters are EEGNet-style defaults and were deliberately left untuned.
- **Not used for predictions yet:** the CNN is registered and can be served (`ModelService` routes segment-input models), but it is not the active model. The prediction API therefore still uses the Random Forest.
- **CPU only:** training ran on CPU. The NVIDIA T500 (2 GB) is not used.

## 11. What remains (next phases of Review 3)

1. **Evaluation robustness:** done (Phase A). Repeated seeds, nested LOPO, CIs and permutation/Wilcoxon tests; see [REVIEW3_CNN_ROBUST_EVALUATION.md](REVIEW3_CNN_ROBUST_EVALUATION.md).
2. **Better input representation:**
   - per-participant normalization: evaluated in Phase A, no improvement in discrimination;
   - possibly arousal and quadrant targets.
3. **CNN-LSTM:** reuse `EEGCNN`'s convolutional front end over sequences of segments within a trial.
4. **LSTM / Transformer** comparisons under the same protocol.
5. **Explainable AI:**
   - saliency maps and integrated gradients for the CNN;
   - SHAP and permutation importance for the Random Forest;
   - scalp topographies.
6. **Dashboard:** model comparison view and model selection for predictions; then final integration, deployment and testing.

## Reproduce

```powershell
cd backend
..\.venv\Scripts\python -m pip install -r requirements-dl.txt
..\.venv\Scripts\python -m scripts.segment_summary      # build/verify segment table (real DEAP)
..\.venv\Scripts\python -m scripts.train_cnn            # train + evaluate CNN (~8 min CPU)
..\.venv\Scripts\python -m scripts.compare_models       # RF / SVM / XGBoost / CNN table
..\.venv\Scripts\python -m pytest                       # 104 tests (91 at this milestone + 13 robust-evaluation tests)
..\.venv\Scripts\python -m pytest -m integration        # 2 real-DEAP tests
```
