# NeuroSense AI: Per-Person EEG Covariance Alignment Experiment

**Status:** exploratory (12 participants, Protocol v1.1). There is no final-test result.
- **Data:** the frozen test participants were never loaded, and the unapproved Kaggle copy was not used.
- **Production:** the active model, its pointer and files, the production database and the protocol were not changed (§7).
- **Run:** 22–23 October 2026.
- **Code:** `backend/scripts/covariance_alignment.py`.
- **Results:** `docs/results/exploratory/covariance_alignment/` (`summary.json`, `tables.md`).

> **Result: negative.**
> - **New person, new videos:** per-person covariance alignment scored **0.499 [0.458, 0.539]** trial balanced accuracy, against **0.493 [0.470, 0.516]** for the existing baseline. The difference is +0.006 [−0.025, +0.038]; Holm p = 1.0.
> - **Simpler normalization was better:** per-person z-scoring of the existing features (0.535) **outperformed** alignment.
> - **Within a person:** aligned covariance features did not beat the existing features (0.607 vs 0.618).
> - **Recommendation:** alignment does **not** merit further work on these 12 participants. The reasons are in §6.

---

## 1. Motivation

The model-improvement study (`docs/MODEL_IMPROVEMENT_EXPERIMENTS.md`) found that cross-person prediction fails mainly through **per-person offsets**:
- **Collapse:** about 5 of 12 people receive one label for nearly every trial.
- **Ranking without calibration:** within-person AUC is above 0.5, but the decision threshold sits in the wrong place for each person.

**Per-person covariance re-centring** ("Riemannian alignment") is the standard EEG transfer-learning method aimed at exactly this problem:
- each person's spatial covariance structure is moved to a common origin before a classifier sees it;
- it is the representation proposed in Protocol §9 D3 (N4) and D4.

## 2. Methodology (declared in the script docstring before any result existed)

### 2.1 Data

- **Same data as all earlier experiments:** the cached preprocessed 4 s segments of the 12 development participants. These come from the same pipeline (4–45 Hz band-pass, robust clipping) as the 440-feature table and match it row for row: 7,046 segments, fingerprint `3c5359083bc90ae5`. The code asserts that participant, trial and fingerprint agree and that no frozen participant is present.
- **Label:** `high_valence` if valence > 5.0.
- **Trial prediction:** mean segment probability > 0.5.

### 2.2 Covariance representation (fixed; nothing tuned)

| Step | Setting |
|---|---|
| Bands | θ 4–8, α 8–13, β 13–30, γ 30–45 Hz. Zero-phase Butterworth (order 4), filtered **within each segment only**. |
| Covariance | One 32×32 matrix per band and segment, with fixed shrinkage C = 0.95 S + 0.05 · tr(S)/32 · I. All matrices verified symmetric positive definite. |
| Reference | Log-Euclidean mean expm(mean logm C) |
| Re-centring (alignment) | C → R^−½ C R^−½ |
| Features | Upper triangle of logm(re-centred C), off-diagonal terms × √2: 528 per band, **2,112** per segment |

**Mechanical check of the alignment (all 7,046 segments):**
- **Without alignment** (one global reference), the average spread **between** people's mean feature vectors (18.1) was more than twice the spread **within** a person (7.7). Person identity dominated the representation.
- **With per-person alignment,** the between-person spread fell to **0.6**, while within-person spread was unchanged (8.0).
- **Conclusion:** the transform did what it is designed to do.

### 2.3 Evaluation: identical folds for every condition

**Cross-person (new person on new videos):**
- **Folds:** for each held-out participant, each seed (42–44) and each of the 5 label-free video folds (`cross_participant_lvo.video_folds`, 8 test videos), train on the **other 11 participants' 32 non-test videos** and test on the held-out person's 8 test videos.
- **Size:** 12 × 3 × 5 = 180 fits per condition.
- **Test videos never enter training,** for any participant.

| Condition | Features | Normalization / alignment | Model |
|---|---|---|---|
| **A0 baseline** | Existing 440 | StandardScaler fitted on training rows (inductive) | Registered Random Forest |
| A1 | Existing 440 | Each person z-scored on their own unlabelled non-test videos | Registered Random Forest |
| A2 control | Covariance tangent | **One** reference from the training rows only, applied to everyone (no alignment) | Registered Random Forest |
| **A3 alignment** | Covariance tangent | **Each person re-centred on their own unlabelled non-test-video segments** | Registered Random Forest |
| A4 | A3 features | As A3 | Logistic regression, C = 0.01, fixed in advance (descriptive only) |

**Within-person:**
- **Folds:** the same video folds and seeds; train on the person's 32 videos, test on 8.
- **W0:** the 440 features with the Random Forest.
- **W1:** tangent features re-centred at the person's **training-video** reference, fitted inside the fold, with the Random Forest.

**Assumption about unlabelled data (A1, A3, A4):**
- **What these conditions use:** the new person's **unlabelled** EEG from the 32 videos that are not being tested. They never use labels, and never the test videos.
- **What that makes them:** the Protocol §6.2 *unsupervised calibration* variant. They are **not zero-shot**, and a product would need an unlabelled recording session from each new user.

**Leakage controls asserted in code:**
- the held-out person and the test videos are never in training;
- every test video is predicted exactly once;
- every feature is finite.

During the smoke test, before any results were produced, this check caught a bug: the held-out person's reference had been computed from an empty row set. The bug was fixed before the run.

### 2.4 Statistics

- **Primary metric:** trial balanced accuracy per participant, averaged over seeds; mean with t 95% CI over the 12 participants. Macro-F1 and within-person ROC AUC are also reported.
- **Comparisons (declared in advance):**
  - **primary** A3 − A0;
  - **secondary** A3 − A1, A3 − A2 and W1 − W0.
- **Tests:** participant-bootstrap 95% CI of the paired mean difference (10,000 resamples), Wilcoxon signed-rank and an exact sign-flip test, **Holm-corrected across the four**. Everything else is descriptive.
- **No tuning:** one run per condition, no repeated tuning, nothing changed after results were seen.

## 3. Results

### 3.1 Conditions

| Condition | Trial bal. acc. [95% CI] | Macro-F1 | Within-person AUC | Range over people | Single-label people | Seed SD |
|---|---|---|---|---|---|---|
| A0 baseline, 440 features (inductive) | 0.493 [0.470, 0.516] | 0.419 | 0.524 | 0.423–0.550 | 3.0 | 0.015 |
| A1 baseline + per-person z-score | **0.535 [0.492, 0.578]** | 0.521 | 0.549 | 0.412–0.667 | 0.0 | 0.005 |
| A2 covariance, global reference | 0.506 [0.471, 0.541] | 0.423 | 0.509 | 0.429–0.633 | 3.3 | 0.032 |
| **A3 covariance, per-person alignment** | **0.499 [0.458, 0.539]** | 0.466 | 0.529 | 0.403–0.592 | 0.0 | 0.023 |
| A4 aligned covariance + logistic regression | 0.509 [0.444, 0.573] | 0.494 | 0.528 | 0.372–0.676 | 0.0 | 0.017 |
| W0 within-person, 440 features | **0.618 [0.548, 0.688]** | 0.617 | 0.644 | 0.413–0.758 | 0.0 | 0.012 |
| W1 within-person, aligned covariance | 0.607 [0.532, 0.681] | 0.604 | 0.649 | 0.440–0.776 | 0.0 | 0.011 |

**Replication checks:**
- A1 = **0.535 [0.492, 0.578]** and W0 = **0.618** exactly reproduce the earlier unsupervised-calibration and personalized results.
- A0 (0.493) agrees with the earlier leave-video-out baseline (0.500), which used slightly different training rows.

### 3.2 Paired comparisons (pre-declared, Holm over 4)

| Comparison | Δ bal. acc. [bootstrap 95% CI] | Wilcoxon p | Sign-flip p | Holm p | Better / worse | Δ macro-F1 [CI] | Δ AUC [CI] |
|---|---|---|---|---|---|---|---|
| **A3 − A0 (primary)** | **+0.006 [−0.025, +0.038]** | 0.62 | 0.72 | **1.00** | 8 / 4 | +0.047 [+0.002, +0.090] | +0.005 [−0.047, +0.052] |
| A3 − A1 | −0.036 [−0.070, −0.003] | 0.09 | 0.07 | 0.37 | 4 / 8 | −0.055 [−0.096, −0.012] | −0.020 [−0.072, +0.038] |
| A3 − A2 (effect of alignment itself) | −0.007 [−0.042, +0.025] | 0.97 | 0.69 | 1.00 | 7 / 5 | +0.042 [−0.028, +0.109] | +0.021 [−0.018, +0.064] |
| W1 − W0 | −0.011 [−0.038, +0.014] | 0.41 | 0.44 | 1.00 | 6 / 6 | −0.013 [−0.041, +0.012] | +0.005 [−0.015, +0.024] |

**Descriptive comparisons (not corrected):**

| Comparison | Δ bal. acc. [CI] | Better / worse |
|---|---|---|
| A1 − A0 | +0.042 [+0.003, +0.076] | 9 / 3 |
| A2 − A0 | +0.013 [−0.017, +0.043] | 7 / 5 |
| A4 − A0 | +0.016 [−0.035, +0.066] | 7 / 5 |

### 3.3 Participant-level results (trial balanced accuracy, seed-averaged)

| Participant | A0 | A1 | A2 | **A3** | A4 | W0 | W1 |
|---|---|---|---|---|---|---|---|
| s01 | 0.471 | 0.510 | 0.487 | 0.501 | 0.391 | 0.501 | 0.440 |
| s02 | 0.423 | 0.535 | 0.491 | 0.431 | 0.381 | 0.666 | 0.690 |
| s03 | 0.475 | 0.536 | 0.508 | 0.591 | 0.514 | 0.705 | 0.729 |
| s05 | 0.531 | 0.597 | 0.517 | 0.510 | 0.479 | 0.656 | 0.559 |
| s07 | 0.514 | 0.562 | 0.528 | 0.573 | 0.528 | 0.413 | 0.454 |
| s08 | 0.462 | 0.566 | 0.441 | 0.487 | 0.526 | 0.636 | 0.569 |
| s09 | 0.550 | 0.667 | 0.542 | 0.592 | 0.675 | 0.650 | 0.675 |
| s11 | 0.458 | 0.538 | 0.448 | 0.403 | 0.497 | 0.472 | 0.510 |
| s15 | 0.492 | 0.483 | 0.517 | 0.517 | 0.583 | 0.758 | 0.750 |
| s16 | 0.504 | 0.567 | 0.633 | 0.509 | 0.676 | 0.671 | 0.627 |
| s22 | 0.529 | 0.445 | 0.429 | 0.431 | 0.372 | 0.746 | 0.776 |
| s25 | 0.500 | 0.412 | 0.529 | 0.439 | 0.481 | 0.541 | 0.500 |

**Consistency across participants:**
- **A3 is not consistent.** It beats the baseline for 8 of 12 people, but the gains are small and offset by losses: s05 −0.021, s11 −0.055, s22 −0.098, s25 −0.061. The largest gain is s03 (+0.116).
- **A3 is mostly near chance.** It scores ≥ 0.55 for only 3 people (s03, s07, s09), with no clear group pattern.
- **Within-person is mixed.** Aligned covariance helps 6 people and hurts 6, mostly by small amounts; s05 is the largest loss (−0.097).

## 4. Interpretation

1. **Alignment removed the collapse.** No person received a single label under A3 (0 vs 3 for the baseline), and macro-F1 rose by +0.047, CI just above 0.
   - This is the same mechanical effect per-person z-scoring achieves. Macro-F1 penalizes collapse heavily, so the gain is partly mechanical.
2. **It did not improve discrimination.**
   - Trial balanced accuracy (+0.006) and within-person AUC (+0.005) are unchanged.
   - Aligned covariance features rank a new person's trials no better than chance (AUC 0.529).
3. **Alignment itself added nothing.** On the same covariance features, A3 vs A2 (global reference) is −0.007.
4. **Simple feature z-scoring was better.**
   - A1, using the *same* unlabelled data as A3, reached 0.535.
   - A3 was lower by 0.036 (bootstrap CI excludes 0, but Holm p = 0.37).
5. **The person-specific signal is not in the covariance structure.**
   - **What the 18.1 → 0.6 spread means:** most of what distinguishes people in covariance space is a person-level offset, and alignment removes it as intended.
   - **What's left:** after removing it, the remaining within-person variation carries no transferable valence information in this sample.
   - **Within-person evidence:** inside a person, covariance features (W1, 0.607) carry about as much information as the existing band-power features (W0, 0.618). So the representation is not useless, but it adds nothing.

### 4.1 Why the 49.3% baseline is not comparable with the earlier 59.2%

Both numbers come from the same type of Random Forest, but they are **different measurements**:

| | 59.2% (active model, historical) | 49.3% (A0 here) |
|---|---|---|
| Metric | Trial **accuracy** (not balanced for class mix) | Trial **balanced** accuracy |
| Aggregation | Pooled over 120 trials | Mean of 12 per-participant values |
| Who is tested | 3 fixed people (s01, s09, s11), one split, one seed | Every one of the 12 people × 5 video folds × 3 seeds |
| Videos in training | All 40, including the videos being tested (seen from other people) | Only the 32 non-test videos; test videos are never in training |
| Status | Historical; these 3 people have been examined repeatedly. The participant-level CI is [0.31, 0.93]. | Exploratory, Protocol v1.1 |

**What this means:**
- **Not a regression.** The 49.3% does not mean the model got worse. It answers a harder, more honest question: a new person watching new material, scored with a metric that a one-label guess cannot inflate.
- **Results here, in context.** Results in this document are comparable only with each other (identical folds) and with the earlier leave-video-out studies.
- **Consolidated comparison:** see `RESEARCH_FINDINGS_AND_NEXT_STEPS.md` §9c.

## 5. Limitations

- **Small, non-independent sample:**
  - with 12 participants and no independent test, intervals are about ±0.04 wide around 0.5;
  - an effect smaller than about +0.03 cannot be ruled out;
  - nothing here says anything about people outside this sample.
- **Only one configuration was tested:**
  - the bands, shrinkage, reference (log-Euclidean rather than an iterative Riemannian mean), the Random Forest and the tangent-space classifier were all fixed in advance;
  - other variants (e.g. Riemannian minimum-distance classifiers, more bands, longer windows) were deliberately not searched, to avoid tuning toward a favourable result.
- **Calibration assumption:** A3 assumes about 32 videos of unlabelled EEG from each new person. Less data would make the reference noisier.
- **Same stimuli for everyone:** all participants watched the same 40 videos. Videos were held out, but an effect specific to this stimulus set cannot be excluded.
- **Run interruptions:** the run was interrupted twice by session restarts and resumed from saved per-job results. Every job is deterministic given its seed, so resuming does not change results.

## 6. Recommendation

**Do not pursue per-person covariance alignment further on the current 12-participant data.**

- **The pre-declared primary comparison is null:** +0.006, CI [−0.025, +0.038].
- **A simpler method was better:** z-scoring the existing features does better with the same unlabelled data.
- **It adds nothing within a person either.**
- **What would change this verdict:** re-evaluation once Decision I is resolved and the official DEAP release adds development participants, run as part of a pre-declared comparison against **A1** (not A0), since A1 is now the stronger reference.

**Effect on the product:** none.
- The active Random Forest stays active.
- No candidate was produced or activated.
- Personalized calibration with the person's own rated videos (about 0.62) remains the only setting reliably above chance (see `MODEL_IMPROVEMENT_EXPERIMENTS.md`).
- No 90% claim is supported by any result here.

## 7. Verification

| Check | Result |
|---|---|
| Production model files, metrics, predictions, graphs, protocol, `participants.py` | 70 files, **SHA-256 identical** before and after |
| Active model pointer | Unchanged: `valence_binary-random_forest-20261008-180558` |
| Production PostgreSQL (read-only snapshot) | Users, predictions, reports, audit logs, sessions, models and metrics: **all unchanged**; no write transaction |
| Frozen participants / Kaggle copy | Never loaded / not used |
| Backend tests | `python -m pytest -q`: **241 passed** |
| Outputs | `outputs/experiments/covariance_alignment_20261022/` (git-ignored; 441 MB including cached covariance matrices) and `docs/results/exploratory/covariance_alignment/` |
| Commits, pushes, deployments | None |

## 8. Reproducing

Run from `backend/` with the virtual environment active. Phases resume and skip finished jobs.

| Step | Command | Runtime (8 workers, this machine) |
|---|---|---|
| Covariances (one-off, about 35 s) | `python -m scripts.covariance_alignment --phase covariances` | ~35 s |
| Within-person (W0, W1) | `python -m scripts.covariance_alignment --phase within` | ~6 min, 360 fits |
| Cross-person (A0–A4) | `python -m scripts.covariance_alignment --phase cross` | ~2.5–3 h, 900 fits |
| Report (writes `summary.json` and `tables.md`) | `PYTHONIOENCODING=utf-8 python -m scripts.covariance_alignment --phase report` | seconds |
