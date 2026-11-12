# NeuroSense AI — development-only optimisation experiment

> Versioned copy of `outputs/experiments/neurosense_dev_optimisation/NEUROSENSE_DEV_OPTIMISATION_REPORT.md`.
> "This folder" below means `outputs/experiments/neurosense_dev_optimisation/`, which holds the CSVs, JSON and
> plots and is not committed to git.

> **Development cross-validation only. No new independent test has been conducted.**
> The numbers below come from inner folds of FACED development data. They are not test results and must not be
> compared with, or presented as, the earlier FACED test or DEAP figures. No method approaches 80%.

**Result: negative.**
- **Classical model:** none of the unsupervised adaptations improved cross-person valence prediction. Every
  adaptation scored below the unchanged baseline.
- **Neural model:** the largest gain was +1.7 percentage points (95% CI −3.2 to +6.6), for online per-person input
  standardisation. That is within the run-to-run noise of retraining the same model.
- **Per-person score thresholds:** these made results worse; two variants were significantly worse.

Nothing here justifies an external-data experiment for these methods (§10).

Code: `backend/research/dev_optimisation/` (`methods.py`, `run.py`); tests: `backend/tests/test_dev_optimisation.py`.
All outputs are in this folder. No model was registered, activated or deployed. No production file, database row,
existing model or existing report was changed.

---

## 1. Data and split

| Item | Value |
|---|---|
| Split | `faced-protocol-v1` (`outputs/faced/protocol_split_v1.json`), SHA-256 `01e74a4e…1d163c41`, asserted at load |
| Included | the **98 development participants** on the **19 development clips** (2, 3, 4, 5, 8, 9, 10, 11, 13, 15, 16, 17, 19, 20, 21, 24, 25, 27, 28) |
| Excluded participants | the **25 protocol test participants** (sub-000, 001, 003, 006, 011, 013, 016, 024, 028, 032, 044, 046, 050, 054, 058, 063, 073, 077, 096, 105, 107, 110, 111, 117, 121): examined in earlier work |
| Excluded clips | the **9 protocol test clips** (1, 6, 7, 12, 14, 18, 22, 23, 26), for every participant |
| DEAP | not used at all: its 12 authorised participants were the Fusion external test; the 8 frozen participants are never read |
| Segments used | 12,994 (4 s, 30 channels, 125 Hz) |
| Validation trials | 373 (each development person is held out once, on 3 or 4 clips) |

`split_manifest.json` and `participants.csv` list every included and excluded ID with the reason.

The split was established without ambiguity from the manifest. No guessing was needed.

**One caveat on "load".** The shared FACED cache files hold all 123 participants in one array. The code reads the
participant and clip columns, keeps only development people on development clips, and discards every other row
before any feature, label or score is computed (`load_dev`, asserted).

**Presentation order** for the online variants comes from each development participant's own recording
timestamps; no rating is read for it (`presentation_orders.json`).

## 2. Folds

These are the protocol's own five inner folds. Fold *k* holds out about 20 development people **and** 3–4
development clips; it trains on the other people's data for the other clips only.

| Fold | Held-out clips | Validation trials | True high share |
|---|---|---|---|
| 0 | 5, 10, 24, 28 | 80 | 44% |
| 1 | 9, 11, 13, 25 | 80 | 31% |
| 2 | 3, 4, 8, 16 | 80 | 28% |
| 3 | 2, 19, 21, 27 | 76 | 63% |
| 4 | 15, 17, 20 | 57 | 75% |

## 3. Methods

Both baselines run unchanged:
- **classical:** the existing S1 logistic regression (C = 0.01, class-balanced, standard-scaled, 440 features);
- **neural:** the Fusion `erm` network (10 epochs, seed 42).

All settings below were fixed before any result.

| ID | Method | Models | Unlabelled EEG of the held-out person used before prediction |
|---|---|---|---|
| A | Baseline, unchanged | both | none |
| B | Per-person standardisation: classical features z-scored per person; neural input channels standardised per person. Training people use their own training rows | both | batch or online (see below) |
| C | Per-person score threshold on the **baseline** scores. *median*: half the person's trials are called high. *base_rate*: the training people's high share is called high. No held-out labels | both | batch, batch-unseen or online |
| D | Covariance alignment. Classical: band covariance tangent features (θ α β γ, shrinkage 0.05), re-centred at each person's log-Euclidean mean, as in `docs/COVARIANCE_ALIGNMENT_EXPERIMENT.md`. Neural: Euclidean alignment of the signals (R^−½ x). References for training people come from their training rows | both | batch or online |
| D control | Same covariance features with **one** reference from the training rows (no per-person alignment) | classical | none |
| F | Ensemble: mean segment probability of seeds 42, 43 and 44 (each seed also shown) | neural | none |
| H | Early stopping: the number of epochs (up to 20, patience 5) is chosen on a separate group of training people on separate training clips (inner fold *k + 1*), then the model is retrained on the full fold training rows for that many epochs | neural | none |

Logistic regression has no epochs and no random seed, so F and H apply to the neural model only.

**Adaptation modes (E).** The held-out person's emotion ratings are never used.
- **batch:** statistics from all of that person's development-clip EEG (19 clips), including the clips being
  predicted. A prediction session would need the whole recording first.
- **batch-unseen** (C only): only the person's clips held out from this fold's training, which are the clips being
  predicted. Added before any result, because the smoke test showed that the baseline's scores on clips it trained
  on (other people's ratings) are distributed differently from its scores on unseen clips.
- **online:** only the person's development clips shown **before** the clip being predicted. Until 2 earlier clips
  exist, the training people's population statistics are used. This happened for 45 of the 373 trials.

## 4. Evaluation

| Item | Setting |
|---|---|
| Unit | trial = one person × one clip; trial score = mean segment probability; high if above 0.5 (or the per-person threshold for C) |
| Primary metric | pooled trial balanced accuracy over all 5 folds (373 trials, 98 people) |
| Also reported | accuracy, macro-F1, ROC AUC, sensitivity, specificity, mean per-participant balanced accuracy (65 people with both classes in their validation trials), per-participant predicted-high share |
| CIs | participant-clustered bootstrap, 2,000 resamples, the same resamples for every method, so differences from the baseline are paired |
| Multiplicity | 27 conditions, no correction applied. Read the CIs as descriptive |

## 5. Results — all conditions

BA = trial balanced accuracy; Δ = change from the same model family's baseline in percentage points (pp).

### Classical (logistic regression)

| Method | BA [95% CI] | Δ pp [95% CI] | Acc | Macro-F1 | AUC | Sens / Spec | Per-person BA | Pred-high SD |
|---|---|---|---|---|---|---|---|---|
| **A baseline** | **55.2 [49.4, 60.9]** | — | 55.2 | 0.552 | 0.546 | 55.5 / 55.0 | 52.6 | 0.405 |
| B batch | 51.4 [46.0, 56.7] | −3.8 [−11.6, +3.9] | 51.7 | 0.514 | 0.515 | 46.8 / 56.0 | 49.1 | 0.273 |
| B online | 49.3 [43.9, 54.6] | −5.9 [−13.8, +2.3] | 49.6 | 0.493 | 0.496 | 45.7 / 53.0 | 50.3 | 0.270 |
| C median, batch | 50.8 [46.1, 55.5] | −4.5 [−10.7, +1.8] | 50.9 | 0.508 | 0.527 | 48.6 / 53.0 | 50.0 | 0.260 |
| C median, batch-unseen | 46.7 [42.4, 51.1] | **−8.5 [−14.8, −2.4]** | 46.9 | 0.467 | 0.507 | 43.9 / 49.5 | 46.9 | 0.066 |
| C median, online | 50.9 [46.1, 56.0] | −4.3 [−10.2, +2.2] | 50.9 | 0.509 | 0.530 | 50.9 / 51.0 | 48.3 | 0.282 |
| C base rate, batch | 49.3 [44.5, 54.4] | −6.0 [−12.6, +0.4] | 49.6 | 0.493 | 0.495 | 45.1 / 53.5 | 49.4 | 0.261 |
| C base rate, batch-unseen | 46.7 [42.4, 51.1] | **−8.5 [−14.8, −2.4]** | 46.9 | 0.467 | 0.488 | 43.9 / 49.5 | 46.9 | 0.066 |
| C base rate, online | 50.2 [45.1, 55.3] | −5.1 [−11.3, +1.2] | 50.4 | 0.502 | 0.512 | 46.8 / 53.5 | 49.1 | 0.279 |
| D batch | 50.8 [45.6, 55.8] | −4.5 [−12.7, +3.3] | 51.2 | 0.507 | 0.506 | 44.5 / 57.0 | 52.4 | 0.265 |
| D online | 51.9 [46.9, 56.9] | −3.3 [−11.4, +5.0] | 52.3 | 0.519 | 0.519 | 46.8 / 57.0 | 53.6 | 0.287 |
| D control, global reference | 55.4 [50.1, 60.5] | +0.1 [−6.7, +7.0] | 55.8 | 0.553 | 0.574 | 49.7 / 61.0 | 51.9 | 0.399 |

### Neural (FusionNet `erm`)

| Method | BA [95% CI] | Δ pp [95% CI] | Acc | Macro-F1 | AUC | Sens / Spec | Per-person BA | Pred-high SD |
|---|---|---|---|---|---|---|---|---|
| **A baseline (seed 42)** | **55.5 [50.3, 60.4]** | — | 56.3 | 0.553 | 0.565 | 44.5 / 66.5 | 52.2 | 0.384 |
| A seed 43 | 55.0 [49.7, 60.1] | −0.5 [−5.6, +4.8] | 53.6 | 0.526 | 0.565 | 73.4 / 36.5 | 51.4 | 0.385 |
| A seed 44 | 56.4 [50.6, 62.0] | +0.9 [−4.6, +6.3] | 55.2 | 0.545 | 0.605 | 72.8 / 40.0 | 50.4 | 0.386 |
| F ensemble, 3 seeds | 56.7 [51.1, 62.0] | +1.2 [−2.7, +5.3] | 56.0 | 0.559 | 0.592 | 66.5 / 47.0 | 53.3 | 0.399 |
| B batch | 55.6 [49.4, 61.5] | +0.1 [−5.0, +5.2] | 55.2 | 0.552 | 0.580 | 60.1 / 51.0 | 49.2 | 0.422 |
| B online | 57.2 [51.2, 62.7] | +1.7 [−3.2, +6.6] | 56.8 | 0.568 | 0.585 | 61.8 / 52.5 | 51.3 | 0.407 |
| C median, batch | 51.9 [46.6, 56.9] | −3.6 [−9.0, +2.2] | 52.0 | 0.518 | 0.536 | 49.7 / 54.0 | 51.0 | 0.250 |
| C median, batch-unseen | 47.8 [43.0, 52.5] | **−7.7 [−13.8, −1.4]** | 48.0 | 0.478 | 0.497 | 45.1 / 50.5 | 49.4 | 0.066 |
| C median, online | 51.9 [46.7, 56.9] | −3.6 [−8.6, +1.7] | 51.7 | 0.517 | 0.525 | 53.8 / 50.0 | 51.5 | 0.269 |
| C base rate, batch | 50.1 [44.8, 55.4] | −5.4 [−11.0, +0.7] | 50.4 | 0.501 | 0.505 | 46.2 / 54.0 | 52.4 | 0.238 |
| C base rate, batch-unseen | 47.8 [43.0, 52.5] | **−7.7 [−13.8, −1.4]** | 48.0 | 0.478 | 0.479 | 45.1 / 50.5 | 49.4 | 0.066 |
| C base rate, online | 52.2 [47.0, 57.4] | −3.3 [−8.3, +2.0] | 52.3 | 0.522 | 0.510 | 50.9 / 53.5 | 53.2 | 0.279 |
| D batch | 55.0 [50.2, 59.8] | −0.5 [−6.4, +5.3] | 54.4 | 0.543 | 0.575 | 63.6 / 46.5 | 50.5 | 0.366 |
| D online | 55.2 [50.1, 59.9] | −0.3 [−5.9, +5.6] | 54.7 | 0.546 | 0.572 | 62.4 / 48.0 | 50.4 | 0.371 |
| H early stopping | 54.3 [48.9, 59.6] | −1.2 [−6.0, +3.5] | 53.4 | 0.530 | 0.554 | 67.1 / 41.5 | 51.4 | 0.400 |

Every row's interval overlaps the baseline's. The only differences whose CI excludes zero are the four
batch-unseen C variants, and all four are **worse**. Full rows are in `comparison.csv`; per-fold figures are in
`per_fold.csv`; per-person figures are in `per_participant.csv`. `plot_delta_vs_baseline.png` shows every Δ with
its CI.

## 6. Does score adjustment fix the shifting threshold?

It narrows the spread of per-person positive rates, but accuracy gets worse. The spread was not the main error.

**Per-person predicted-high share.** Most people have only 3–4 validation trials, so these shares come in coarse
steps. See `plot_positive_rates.png`.

| Model | Spread across people (SD), baseline | Best C variant | Mean gap to the person's true share, baseline | Best C variant |
|---|---|---|---|---|
| Classical | 0.405 | 0.066 (batch-unseen) | 0.381 | 0.279 |
| Neural | 0.384 | 0.066 (batch-unseen) | 0.379 | 0.279 |

**Per fold, predicted vs true high share (%):**

| | Fold 0 | Fold 1 | Fold 2 | Fold 3 | Fold 4 |
|---|---|---|---|---|---|
| True | 44 | 31 | 28 | 63 | 75 |
| Neural baseline | 52 | 34 | 21 | 45 | 42 |
| Neural C base rate, batch | 31 | 51 | 49 | 47 | 54 |
| Classical baseline | 50 | 44 | 49 | 61 | 46 |
| Classical C base rate, batch | 36 | 48 | 52 | 49 | 44 |

**Why it fails.**
- **The true high share depends on the clips, not just the person.** It ranges from 28% to 75% across folds
  because each fold holds out different clips, and some clips are mostly rated pleasant while others are mostly
  rated unpleasant.
- **A per-person threshold assumes the opposite.** It assumes each person has a fixed base rate, so it forces
  every fold towards about 45–50% high and erases the clip-driven variation the baseline partly tracked. The most
  extreme case is the batch-unseen median, which calls exactly half of a person's 3–4 clips high whatever the
  clips are. It was significantly worse.
- **The model barely ranks one person's trials.** The pooled AUC of the adjusted scores is about 0.48–0.54 (it
  measures how well the model orders a person's own trials once their personal offset is removed). So moving the
  threshold cannot recover much.

**Correction to an earlier statement.** In the Fusion experiment, I attributed the per-fold swings in predicted
high share (21–54%) to a threshold that shifts between groups of people. Part of those swings is real, because
the clip groups differ in their true high share (28–75%). The shift between people is real but is not the
bottleneck.

## 7. Batch vs online adaptation (E)

| Model | Method | Batch | Online | Online − batch |
|---|---|---|---|---|
| Classical | B | 51.4 | 49.3 | −2.1 |
| Classical | D | 50.8 | 51.9 | +1.1 |
| Classical | C median / base rate | 50.8 / 49.3 | 50.9 / 50.2 | +0.1 / +0.9 |
| Neural | B | 55.6 | 57.2 | +1.6 |
| Neural | D | 55.0 | 55.2 | +0.2 |
| Neural | C median / base rate | 51.9 / 50.1 | 51.9 / 52.2 | 0.0 / +2.1 |

- **Neither mode is consistently better.** Every difference between batch and online is within ±2.1 pp, inside
  the CIs.
- **Online is the deployable form.** It uses only EEG already recorded before each prediction, and it lost
  nothing measurable compared with batch.
- **Neither mode helped the classical model.** Both stay below the classical baseline.

## 8. Seeds, ensemble and reproducibility (F)

| Item | Result |
|---|---|
| Single neural seeds | 55.5, 55.0 and 56.4: a 1.4 pp spread in BA |
| Decision behaviour across seeds | changes completely: sensitivity 44.5% (seed 42) vs 73.4% and 72.8% (seeds 43, 44) |
| 3-seed ensemble | 56.7%, Δ +1.2 pp [−2.7, +5.3]; AUC 0.592 vs 0.565; still within noise |
| Retraining the same seed 42 | not bit-for-bit reproducible on this CPU. Fold 0: 54.1% here vs 53.0% in the Fusion run. Fold 1: 57.5% vs 59.5%. Fold 2: identical (57.3%) |

**Classical reproduction.** The classical baseline reproduces `outputs/faced/inner_classical.json` exactly on all
5 folds (48.7, 56.0, 63.4, 52.7, 56.6).

**Noise.** Retraining the identical model therefore moves fold scores by up to 2 pp. That is the same size as
every neural "gain" in this study.

## 9. Three vs five folds (G)

| Model | Method | Mean fold BA, 3 folds | Mean fold BA, 5 folds | Pooled BA, 5 folds |
|---|---|---|---|---|
| Classical | A baseline | 56.0 | 55.5 | 55.2 |
| Neural | A baseline | 56.3 | 53.4 | 55.5 |
| Neural | D batch | **57.8 (highest)** | 51.6 | 55.0 |
| Neural | B online | 56.7 | 51.3 | 57.2 |
| Neural | F ensemble | 56.7 | 53.5 | 56.7 |
| Neural | H early stopping | 53.5 | 51.1 | 54.3 |

- **3-fold selection is unreliable.** On folds 0–2, neural covariance alignment looks best, at +1.5 pp over the
  baseline. Folds 3–4 reverse that. The Fusion experiment selected on 3 folds.
- **5-fold selection:** the classical baseline remains first. For the neural model, the highest mean fold score
  belongs to a single extra seed (seed 43, 53.8), which is not a method.
- **Conclusion:** no method is selected as an improvement, and 5 folds should be the minimum from now on.

## 10. Early stopping (H)

| Item | Result |
|---|---|
| Chosen epochs | 2, 3, 3, 7, 6 (baseline: fixed 10) |
| Stop-group curves | noisy; between 39% and 64% from epoch to epoch, on 57–80 trials per group |
| BA | 54.3%, Δ −1.2 pp [−6.0, +3.5] |
| Side effect | shorter training raised sensitivity and lowered specificity (67.1 / 41.5) |

With validation groups this small, epoch selection is mostly noise. Early stopping is **not** recommended in this
form.

## 11. Leakage controls and safeguards

| Check | How | Result |
|---|---|---|
| Only development people on development clips | `dev_mask` asserts that no test person or test clip remains; manifest checks | passed |
| Validation people and clips never in training | `fold_rows` asserts for every fold | passed |
| Early-stopping group inside training, separate from validation | asserted per fold | passed |
| Statistics for training people from training rows only | code path; references are built per training person from their training rows | by construction |
| No held-out ratings in adaptation | B, C and D read only EEG or model scores. The C base rate comes from training labels | by construction; unit tests |
| Online uses only earlier clips | `reference_rows` / `earlier_clips`; unit tests | passed |
| Bugs found before any result | 1. matrix exponential clamped negative eigenvalues (unit test) 2. early stopping trained on 55% of rows (redesigned) 3. C thresholds used clips the model trained on (batch-unseen added) | fixed before results |
| Production unchanged | 70 production file hashes match; saved FACED models unchanged; database snapshot unchanged; Fusion results and earlier reports untouched | passed |
| No old test set used | FACED test people and clips and DEAP were never scored | passed |
| Git | nothing committed or pushed | — |

Tests: 7 new (`test_dev_optimisation.py`) and 15 Fusion tests pass.

## 12. Limitations

- **Small validation sets:** 373 trials, 3–4 per person. Per-person rates are coarse, and CIs are wide (about
  ±5–6 pp for BA, ±5–8 pp for differences).
- **No multiplicity correction:** 27 conditions were run. If anything, the CIs overstate how promising the best row is.
- **One seed for B, D and H:** only the baseline had 3 seeds, so seed noise (about ±1–2 pp) is not averaged out for
  the adaptations.
- **Few earlier clips for online adaptation:** the online variants have at most 18 earlier development clips per
  person; test clips are excluded by rule.
- **One development dataset:** the conclusions apply to FACED valence binarised at 3.5.

## 13. Conclusion: is anything worth an external-data experiment?

**No.**
- **Classical model:** every unsupervised adaptation scored below the existing baseline.
- **Neural model:**
  - The best results are +1.7 pp (online per-person standardisation) and +1.2 pp (3-seed ensemble), with CIs
    spanning about −3 to +7.
  - Retraining the identical model moves results by up to 2 pp.
  - Per-person threshold adjustment is counter-productive, because the class balance depends on the clips.
- **Recommended:** if new external data is obtained, keep 5-fold development selection and a 3-seed ensemble as
  cheap defaults (it does no harm). The more promising use of that data is the **labelled personal calibration**
  route, which is a separate question that this study did not test.
