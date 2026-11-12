# NeuroSense Fusion — research report

> Versioned copy of `outputs/experiments/neurosense_fusion/NEUROSENSE_FUSION_REPORT.md`. "This folder" below means
> `outputs/experiments/neurosense_fusion/`, which holds the result files and is not committed to git.

Status: **experimental, not integrated, target not achieved.** Neither capability reached the 80% research target.
Nothing in the application imports this code. The production models, the database and the existing reports are unchanged.

| Capability | Pre-declared primary result (trial balanced accuracy) | Target | Outcome |
|---|---|---|---|
| A. Zero-shot, unseen people (FACED) | **57.3%** (95% CI 51.3–63.6) | 80% | not achieved |
| A. Zero-shot, external dataset (DEAP) | **49.1%** (95% CI 47.6–50.0) | 80% | not achieved (chance) |
| B. Personalised, 21 calibration clips (FACED) | **68.3%** (95% CI 57.3–79.0) | 80% | not achieved |

Following the brief, the experiment stops here. The test sets were evaluated once and nothing was retuned afterwards.

Code: `backend/research/neurosense_fusion/` (`adapters.py`, `protocol.py`, `model.py`, `run.py`); tests: `backend/tests/test_fusion.py`.
Results: this folder (`stage1_check.json`, `inner/`, `freeze.json`, `model_zero_shot.pt`, `zero_shot_results.json`, `personalised_results.json`).

---

## 1. Architecture

```
FACED (250/1000 Hz, 32 ch)          DEAP release (128 Hz, 32 EEG ch)
   │ FACED adapter                      │ DEAP adapter (12 authorised people; frozen 8 never opened)
   │ 30 shared channels by name, common average reference, last 28 s,
   │ 4–45 Hz, 125 Hz, 7 × 4 s segments, per-segment normalisation
   └───────────────┬────────────────────┘
                   ▼  (batch, 30 channels, 500 samples)
   Encoder (27,920 parameters)
     temporal conv, 16 filters × 0.5 s ─► depthwise spatial conv, 32 maps ─► avg pool ×4      EEGNet-style
     3 dilated residual temporal blocks (dilation 1, 2, 4)                                  TCN
     mean + log-variance over time ─► linear                                                shared latent (64)
                   │
     ┌─────────────┼──────────────────────┬────────────────────────────┐
     ▼             ▼                      ▼                            ▼
  valence head   CORAL / MMD loss       gradient reversal ─►         projection head
  (65)           between participant-   domain head (1,040)          (6,240), contrastive
     │           disjoint halves        DANN candidate only          pretraining only
     │           of each batch
     ▼
  trial score = mean of segment probabilities; high if > 0.5
     │
     └─► personal calibration adapter: a copy of the valence head, fitted on one
         person's calibration clips, kept close to the population head
```

Parameter counts: encoder 27,920; valence head 65; domain head 1,040; projection head 6,240; **total 35,265**.
No Transformer was used.

Candidates, all with the same encoder, data and training settings:

| Name | What it adds |
|---|---|
| `erm` | nothing: plain supervised training (baseline) |
| `coral` | CORAL covariance alignment between participant-disjoint halves of each batch |
| `mmd` | multi-bandwidth RBF MMD alignment, same halves |
| `dann` | domain-adversarial head on 16 participant groups, ramped reversal weight up to 0.1 |
| `erm_ssl` | 5 epochs of contrastive pretraining first (no labels): the same clip and segment from two different people is a positive pair |

Fixed settings, chosen before any results: 10 epochs, batch 128, AdamW learning rate 1e-3, weight decay 1e-2, seed 42,
class-balanced loss. Adapter: 100 Adam steps, learning rate 0.05, pull towards the population head weighted 0.1.

## 2. Dataset compatibility

| Dataset | Status | Role here | People | Rate | Channels | Labels |
|---|---|---|---|---|---|---|
| FACED v1.1.4 | integrated | training, selection, reused test | 123 | 250/1000 Hz → 125 Hz | 32 recorded, 30 shared | own valence 0–7; intended category kept separately |
| DEAP | integrated | external evaluation only | 12 authorised; 8 frozen never read | 128 Hz release → 125 Hz | 32 EEG, 30 shared (AF3, AF4 unused) | own valence 1–9 |
| DREAMER | not available: not on this computer, must be requested from its authors | candidate external test | 23 (published) | 128 Hz (published) | 14-channel Emotiv: most shared channels absent | own valence 1–5 |
| AMIGOS | not available: access by request | optional | 40 (published) | 128 Hz (published) | 14-channel Emotiv: most shared channels absent | own valence 1–9 |
| SEED | not available: access by application | optional unlabelled pretraining | 15 (published) | 200 Hz release (published) | 62 channels: covers the shared channels | intended film category only: never merged with self-reported valence |

**Label mapping (documented before use):** each rating is scaled to 0–1 on its own dataset's scale (FACED 0–7, DEAP 1–9).
High valence means above that scale's midpoint, and a rating exactly at the midpoint counts as low.
FACED: 43.3% of trials are high; DEAP: 52.7%.
The intended stimulus category is stored separately and never used as a label.

## 3. Participants and clips

The split is the existing `faced-protocol-v1` (seed 20261026), reused unchanged.

**FACED development (98 people, 19 clips 2, 3, 4, 5, 8, 9, 10, 11, 13, 15, 16, 17, 19, 20, 21, 24, 25, 27, 28)** —
used for training and model selection:
sub-002, 004, 005, 007, 008, 009, 010, 012, 014, 015, 017, 018, 019, 020, 021, 022, 023, 025, 026, 027, 029, 030, 031,
033, 034, 035, 036, 037, 038, 039, 040, 041, 042, 043, 045, 047, 048, 049, 051, 052, 053, 055, 056, 057, 059, 060, 061,
062, 064, 065, 066, 067, 068, 069, 070, 071, 072, 074, 075, 076, 078, 079, 080, 081, 082, 083, 084, 085, 086, 087, 088,
089, 090, 091, 092, 093, 094, 095, 097, 098, 099, 100, 101, 102, 103, 104, 106, 108, 109, 112, 113, 114, 115, 116, 118,
119, 120, 122.

**Validation for model selection** used the first 3 of the protocol's 5 inner folds. Each fold holds out one group of
about 20 development people *and* one group of 4 development clips. Training uses only the remaining people on the
remaining clips:

| Fold | Held-out people | Held-out clips |
|---|---|---|
| 0 | sub-012, 014, 018, 034, 036, 037, 040, 047, 062, 065, 066, 074, 075, 081, 084, 089, 100, 112, 113, 116 | 5, 10, 24, 28 |
| 1 | sub-005, 008, 017, 025, 027, 030, 033, 035, 039, 042, 059, 064, 068, 070, 078, 085, 091, 092, 106, 118 | 9, 11, 13, 25 |
| 2 | sub-004, 009, 022, 026, 043, 051, 067, 072, 076, 086, 088, 090, 097, 098, 099, 101, 102, 103, 115, 122 | 3, 4, 8, 16 |

**FACED test (25 people × 9 clips 1, 6, 7, 12, 14, 18, 22, 23, 26):** sub-000, 001, 003, 006, 011, 013, 016, 024, 028,
032, 044, 046, 050, 054, 058, 063, 073, 077, 096, 105, 107, 110, 111, 117, 121.
**These people and clips were examined in earlier FACED work, so this is a reused test, not an untouched one.**

**DEAP external (12 people, all 40 trials each):** s01, s02, s03, s05, s07, s08, s09, s11, s15, s16, s22, s25.
These were examined in earlier DEAP work. The frozen participants s12, s17, s18, s20, s21, s23, s29 and s30 were never read.

**Personalised evaluation (21 of the 25 FACED test people):** sub-000, 001, 003, 006, 011, 013, 016, 024, 028, 032, 044,
046, 050, 054, 058, 063, 077, 096, 107, 111, 117. sub-073, 105, 110 and 121 were excluded before any scoring,
because their seven presentation blocks of four clips could not be recovered from gaps in the recording.

## 4. Evaluation protocol

**Zero-shot (A).**
1. Train every candidate on each of the 3 inner folds and score the held-out people on the held-out clips.
2. Select the highest mean trial balanced accuracy, ties going to the earlier candidate in the list, and write `freeze.json`.
3. Train the selected candidate once on all 98 development people × 19 clips.
4. Score the FACED test and DEAP once. The decision threshold is fixed at 0.5 and is never tuned.

**Personalised (B), on the frozen zero-shot model.**
For each person and each calibration size n = 7, 14, 21:
- **Calibration:** the first n clips that person watched.
- **Test:** every later presentation block that contains no calibration clip, kept whole.

Three methods are scored on the same test clips:
- **No calibration:** the frozen population head.
- **Personalised:** the adapter fitted on that person's calibration clips. When all calibration clips have the same label,
  it falls back to the population head; this happened for 7 people at n = 7, 1 at n = 14 and 0 at n = 21.
- **Nearest-in-time rating:** the label of the most recent calibration clip, with no EEG used.

Each size is also reported for the test clips the base model never trained on (`unseen clips only`).

**Metrics:** pooled trial balanced accuracy (primary), accuracy, macro-F1, ROC AUC, sensitivity, specificity,
the confusion matrix, class distribution, and the mean per-participant balanced accuracy. The 95% CIs come from
a participant-clustered bootstrap with 2,000 resamples.

## 5. Leakage checks

Each check is enforced in code, not only described.

| Check | How | Result |
|---|---|---|
| No test person or test clip in training (zero-shot and every inner fold) | `check_disjoint` asserts before each fit | passed |
| Inner validation people and clips excluded from that fold's training | `inner_rows` + `check_disjoint` | passed |
| Frozen DEAP participants never read | ID check before any file is opened; `check_no_frozen` on loaded data | passed |
| DEAP never in training | asserted in `stage1_check.json` | passed |
| No cross-recording statistics | preprocessing and normalisation use only the segment itself (unit test) | passed |
| No segments from one recording in both training and testing | every split is by person and by clip | passed |
| Model choice independent of the test | `freeze.json` written before the test; selection uses only inner folds | passed |
| Test run once | `final` and `personal` refuse to run again once results exist | passed |
| Threshold not tuned | fixed 0.5 | passed |
| Calibration precedes test; test blocks whole; no future ratings | `check_personal_plan` asserts per person and size | passed |
| Nearest-in-time baseline looks only backwards | `nearest_past_rating` asserts | passed |
| Production untouched | 70 production file hashes match; saved FACED models unchanged; database row counts and checksums show only one sign-in on 23 Oct, before this work | passed |

Tests: `test_fusion.py` has 15 tests; the full backend suite passes, 285 passed and 2 deselected.

## 6. Measured results

### 6.1 Model selection (FACED development data only)

| Candidate | Fold 0 | Fold 1 | Fold 2 | Mean | Time per fold |
|---|---|---|---|---|---|
| **`erm` (selected)** | 53.0% | 59.5% | 57.3% | **56.6%** | ~290 s |
| `coral` | 54.0% | 54.9% | 58.4% | 55.8% | ~295 s |
| `erm_ssl` | 55.4% | 56.9% | 53.9% | 55.4% | 1,029–1,385 s |
| `dann` | 50.2% | 52.5% | 55.6% | 52.8% | ~285 s |
| `mmd` | 50.0% | 50.0% | 50.0% | 50.0% | 286–343 s |

- **`mmd` failed:** it predicted high for every validation trial in all three folds. It was not retuned.
- **Fold size:** each fold has only 80 trials, so the 0.8–1.2 point gaps between the top three are not meaningful.
- **Alignment and pretraining:** neither beat the plain baseline.

### 6.2 Zero-shot (A)

| Measure | FACED reused test | DEAP external |
|---|---|---|
| Trials (high / low) | 225 (100 / 125) | 480 (253 / 227) |
| **Balanced accuracy** | **57.3%** (51.3–63.6) | **49.1%** (47.6–50.0) |
| Accuracy (majority baseline) | 58.7% (55.6%) | 46.7% (52.7%) |
| Macro-F1 | 0.572 | 0.355 |
| ROC AUC | 0.569 | 0.504 |
| Sensitivity / specificity | 45.0% / 69.6% | 4.7% / 93.4% |
| Confusion [[low→low, low→high], [high→low, high→high]] | [[87, 38], [55, 45]] | [[212, 15], [241, 12]] |
| Mean per-participant balanced accuracy | 54.8% (range 12.5–87.5%) | 49.5% (range 42.5–52.5%) |

- **FACED:** the result is in line with the earlier strict logistic-regression result, 56.6% (51.8–61.2). The new
  architecture gives no measurable gain.
- **DEAP:** the model called 5.6% of trials high, so it does not transfer to a different headset, population and
  stimulus set. This matches the earlier DEAP zero-shot result of 50.0%.

### 6.3 Personalised (B), 21 FACED test people

| Calibration clips | Test trials | No calibration | **Personalised** | Nearest-in-time rating |
|---|---|---|---|---|
| 7 | 420 | 59.8% (54.5–64.2) | **59.6%** (54.6–64.3) | 48.1% (42.2–53.7) |
| 14 | 252 | 61.5% (55.8–66.9) | **58.7%** (51.7–64.2) | 52.7% (44.3–60.4) |
| 21 | 84 | 58.4% (46.4–71.2) | **68.3%** (57.3–79.0) | 57.3% (40.4–74.3) |

Further metrics for the personalised model:

| Clips | Accuracy | Macro-F1 | ROC AUC | Sensitivity / specificity | Confusion | Mean per-participant BA |
|---|---|---|---|---|---|---|
| 7 | 60.5% | 0.597 | 0.651 | 54% / 66% | [[157, 82], [84, 97]] | 55.0% |
| 14 | 59.5% | 0.587 | 0.650 | 52% / 65% | [[93, 49], [53, 57]] | 57.9% |
| 21 | 69.0% | 0.679 | 0.765 | 53% / 84% | [[37, 7], [19, 21]] | 73.1% |

Test clips the base model never trained on:

| Clips | Test trials | No calibration | Personalised | Nearest-in-time |
|---|---|---|---|---|
| 7 | 134 | 64.0% (57.5–70.0) | 65.3% (58.5–71.5) | 45.0% (37.5–52.2) |
| 14 | 83 | 64.1% (56.6–71.3) | 59.6% (50.6–68.0) | 59.0% (48.5–68.5) |
| 21 | 24 (16 people) | 61.1% (43.1–76.3) | **80.0% (65.6–93.8)** | 60.0% (34.4–80.4) |

**What this shows:**
- **7 and 14 clips:** calibration did not help. Personalised was equal to or below no calibration.
- **21 clips:** calibration gave a 10-point gain over no calibration on the same 84 trials, and the ROC AUC rose from
  0.645 to 0.765. This is the only sign of a personal benefit, but it rests on one held-out block of 4 clips per
  person, and the confidence interval reaches down to 57%.
- **The 80.0% figure is not evidence the target was met.** It is a subgroup of the 21-clip test containing 24 trials
  from 16 people and only 9 low trials. Its interval runs from 65.6% to 93.8%, its accuracy is 75.0%, and the people
  were examined in earlier work. Its per-participant figure of 100% covers only the few people whose remaining unseen
  clips contained both classes. The pre-declared primary result is 68.3%.
- **Comparison with the earlier calibration study:** the earlier simple model's block-held-out control scored 59.5%,
  but its calibration and test sets were built differently, so the two figures are not directly comparable.
- **Nearest-in-time baseline:** its per-participant figure is exactly 50% by construction, because it gives each
  person one constant answer.

## 7. Training requirements and runtime

Hardware: CPU only, 8 cores, no GPU; PyTorch 2.14.1+cpu. The FACED segment array is 24,045 × 30 × 500 float32
(1.44 GB) and is memory-mapped.

| Step | Command (from `backend/`) | Measured time |
|---|---|---|
| Stage 1 checks and DEAP cache | `python -m research.neurosense_fusion.run --phase check` | not timed (the DEAP cache is built once and reused) |
| Model selection, 15 fits | `--phase inner` (resumable) | 2 h 3 min |
| Freeze | `--phase freeze` | < 1 s |
| Final training and zero-shot test | `--phase final` (runs once) | 508 s training + scoring |
| Personalised evaluation | `--phase personal` (runs once) | about 1 min |

## 8. Limitations

- **Reused test data:** the FACED test participants and DEAP participants were examined earlier, so neither test is
  untouched. No independent result exists.
- **One source dataset:** DREAMER, AMIGOS and SEED are not on this computer. "Domain" alignment was therefore trained
  only across participants within FACED, not across datasets, and SEED pretraining was not possible.
- **Small personalised test:** at 21 calibration clips the test is 84 trials, 4 per person. A different block order or
  person sample could move the result by about ±10 points.
- **Inferred block structure:** FACED presentation blocks were inferred from gaps in the recording. Four people could
  not be used.
- **No personalised DEAP evaluation:** the DEAP preprocessed release has no presentation order.
- **Shallow selection:** only 3 of the 5 inner folds were used, each with 80 trials.
- **Fixed design:** one seed, one training length and one set of hyperparameters. This avoided test-driven tuning but
  means `mmd` and `dann` may have been under-tuned. That cannot now be fixed against this test.
- **Labels:** self-reported valence binarised at the midpoint is noisy, and 43% of FACED trials are high.
  The results are about this binary label only, not about emotions generally, and carry no clinical meaning.

## 9. Independent test plan (Stage 5)

No valid independent test is possible with the data on this computer. Any future claim of 80% needs:
1. **New data never examined:** new participants recorded with the FACED protocol and the presentation order logged,
   or DREAMER/AMIGOS if access is granted. DREAMER and AMIGOS use a 14-channel headset, so a 14-channel variant of the
   encoder would have to be trained and selected *before* opening them.
2. **Pre-registration:** model, threshold, calibration sizes, primary metric and analysis plan written down and
   hashed before the data is opened.
3. **Enough people:** at least 30 new people for a participant-clustered CI narrow enough (about ±6 points) to tell 80%
   from 70%. This is a planning estimate, not a power calculation.
4. **One evaluation:** run once; report whatever results.

## 10. Frontend integration proposal

**Proposal: do not integrate a prediction feature.** Neither capability met its target, so there is nothing to
activate. If the work is shown at all, it should be a read-only research card on the Model Evidence page, which reads
`zero_shot_results.json` and `personalised_results.json`, labels the work "experimental — not used for predictions",
and shows the primary results with their confidence intervals beside the existing models. No new prediction endpoint,
model registration or database change is proposed.

This proposal has not been implemented.

## 11. Next steps

1. Request DREAMER and AMIGOS access, and plan a pre-registered 14-channel evaluation.
2. Plan fresh recordings with the presentation order logged, so calibration can be tested on untouched people.
3. Any new modelling idea starts again from development data only. The current test sets cannot be used to select
   or tune anything.
