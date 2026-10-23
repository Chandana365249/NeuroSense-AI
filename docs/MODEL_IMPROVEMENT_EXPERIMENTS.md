# NeuroSense AI: Model-Improvement Experiments

**Status:** all results here are **exploratory (12 participants, Protocol v1.1)**.

- **No independent test.** There is no final-test result. The eight frozen test participants were never loaded; the code asserts their absence.
- **No formal claim.** Under Decision C, nothing here supports a formal claim that any model outperforms the Random Forest, or any claim about people outside this sample.
- **Production untouched.** The active model, its pointer, its files, the production database and the evaluation protocol were not changed (§10).
- **Run:** 21–22 October 2026. Code: `backend/scripts/model_improvement.py`. Tables: `backend/scripts/model_improvement_tables.py`. Results: `docs/results/exploratory/model_improvement/`.

> **Short answer.**
> - **New, uncalibrated people:** no model reached a level that is clearly above chance. The best inductive estimate is a trial balanced accuracy of about **0.53–0.54**.
> - **Personal calibration:** a model trained on 32 of a person's own rated videos reaches about **0.62** on that person's other videos. 16 rated videos give about **0.60**.
> - **No 90%.** No valid protocol came close to 90%. The highest upper confidence bound of any valid mean was 0.71.
> - **Where 80% comes from.** A **deliberately leaky** segment-level split reaches 0.80 trial balanced accuracy. It shows how such figures arise in DEAP papers, and it is not a valid result.

---

## 1. Question

**Starting point:**
- The active Random Forest scores **0.592 trial accuracy** on s01, s09 and s11. Those three participants have been examined repeatedly, so this is not an independent test.
- Cross-validation on its training participants gave **0.510 ± 0.086** segment accuracy.

**The question:** what performance can genuinely be achieved on the development data, **without leakage**, and what limits it?

**The 90% aspiration** was treated as a question to test, not as a target.

## 2. Data and evaluation protocol (declared before running)

The design was fixed in the docstring of `backend/scripts/model_improvement.py` before any of its results existed. Outcome metrics were only computed in the final `report` phase.

| Item | Setting |
|---|---|
| Participants | The 12 trusted development participants: s01, s02, s03, s05, s07, s08, s09, s11, s15, s16, s22, s25. They come from the cached feature table, and the code asserts that no frozen participant is present. |
| Features | The existing 440 features per 4 s segment, fingerprint `3c5359083bc90ae5`, with 7,046 segments. **No preprocessing was changed.** |
| Label | `high_valence` if valence > 5.0 (Decision F), the same for every segment of a trial |
| Unit of analysis | **Trial.** Segment scores are aggregated into one trial score, and the trial is predicted high if that score is > 0.5 (> 5.0 for regression, > 0 for log-odds). |
| Primary metric | Trial **balanced accuracy** per participant, averaged over seeds, then the mean over the 12 participants with a t-based 95% CI |
| Secondary metrics | Macro-F1, accuracy, within-person ROC AUC, the number of single-label participants (predicted-high share < 0.05 or > 0.95), and across-seed SD |
| Paired comparisons | Participant bootstrap (10,000 resamples) 95% CI of the mean difference, plus a Wilcoxon signed-rank test |
| Pre-declared primary comparisons | Holm-corrected across three: (P1) nested selection vs RF; (P2) nested selection vs RF; (P3) 32-video pooled calibration vs zero-shot. **Everything else is exploratory.** |
| Seeds | 42–44 for every stochastic model; 42–46 for the Random Forest reference. Logistic regression and SVM are deterministic. |

### 2.1 Three separate protocols (never mixed in one headline)

| Protocol | Question | Design |
|---|---|---|
| **P1 participant-independent** | Does a model work for a **new person** with no calibration? | **Nested** leave-one-participant-out (12 outer folds). Model and aggregation are chosen inside each outer fold by grouped 4-fold CV over its 11 training participants. **N0** (inductive): scaling fitted on training participants only. **N3** (unsupervised calibration): each participant z-scored with their own **unlabelled** segments; reported separately and never as zero-shot. |
| **P2 within-participant (supervised personalization)** | With a person's own **rated** videos, can the model predict that person's ratings of **other** videos? | Per participant: label-free video folds (8 test videos, 32 training videos), identical to the earlier calibration study. Model and aggregation are chosen inside the 32 training videos by 4 label-free inner groups. **Split by video, never by segment.** |
| **P3 population + personal calibration** | Do 8, 16 or 32 personal ratings help a population model? | Train on the other 11 participants' non-test videos plus 0, 8, 16 or 32 of the person's rated videos. Same folds and calibration subsets as P2. Every participant is z-scored on their own unlabelled non-test videos, and own data is weighted to equal the others' total. |

### 2.2 Candidate set (compact by design)

| Choice | Options |
|---|---|
| Models (7 configurations) | Registered Random Forest; registered XGBoost; logistic regression with C ∈ {0.001, 0.01, 0.1}; RBF SVM with C ∈ {0.1, 1} (scores = logistic(decision function)). All with balanced class weights. |
| Trial aggregation (4) | Mean probability (the protocol rule), median, segment majority vote, mean log-odds |
| **Selection effort** | 7 × 4 = 28 combinations, chosen **only** on inner folds |
| Not done | No large hyperparameter search, no new features or preprocessing, no deep models, no threshold tuning |

### 2.3 Leakage controls (asserted in code)

| Protocol | Control |
|---|---|
| P1 | Training and evaluation participants are disjoint in every outer and inner fit. Scaling and model fitting see only the fit's training rows. |
| P2 and P3 | A video is never in both training and test, and every test video is predicted exactly once. Calibration subsets are drawn without labels. |
| P3 | Test videos are excluded from **all** training participants. Normalization uses no labels. |
| S1 | Borderline trials (4.5 ≤ valence ≤ 5.5) are removed from **both** training and scoring (Protocol §7). |

## 3. Baseline reproduction

The active model's historical numbers were re-derived from its saved predictions file. The saved `model.joblib` was then re-scored on the cached features of the same 1,788 segments; the maximum difference is |Δp| = 5.0e-05, which is only the 4-decimal rounding in the CSV. The train, validation and test participant sets are disjoint.

| Level | Accuracy | Balanced acc. | Macro-F1 | ROC AUC | n |
|---|---|---|---|---|---|
| Segment (pooled) | 0.548 | 0.544 | 0.543 | 0.537 | 1,788 segments |
| Trial (pooled) | **0.592** | 0.590 | 0.590 | 0.565 | 120 trials |

**Per participant** (trial balanced accuracy):

| Participant | Trial balanced accuracy | Note |
|---|---|---|
| s01 | 0.500 | Predicted "high" for **every** trial |
| s09 | 0.750 | |
| s11 | 0.604 | |

- **Mean:** 0.618, t 95% CI **[0.306, 0.930]**.
- **Reading:** with 3 people this interval is uninformative, and these participants are no longer an independent test (Protocol §1).

## 4. P1 — participant-independent results

### 4.1 Inductive (N0): a new person, no calibration

| Variant | Trial bal. acc. [95% CI] | Macro-F1 | Within-person AUC | Single-label people | Δ vs RF [bootstrap CI] |
|---|---|---|---|---|---|
| Majority-class reference | 0.500 | 0.343 | 0.500 | 12 | — |
| **Random Forest reference** | **0.533 [0.490, 0.576]** | 0.429 | 0.606 | 5.7 | — |
| RF, nested | 0.532 [0.489, 0.575] | 0.427 | 0.590 | 5.3 | −0.001 [−0.007, +0.005] |
| XGBoost, nested | 0.521 [0.488, 0.555] | 0.418 | 0.578 | 6.7 | −0.012 [−0.057, +0.042] |
| Logistic regression, nested | 0.538 [0.498, 0.579] | 0.435 | 0.600 | 6.0 | +0.005 [−0.045, +0.054] |
| SVM, nested | 0.539 [0.501, 0.577] | 0.428 | 0.624 | 5.0 | +0.006 [−0.029, +0.039] |
| **Nested selection over all 28** | **0.541 [0.504, 0.579]** | 0.429 | 0.595 | 4.7 | +0.008 [−0.024, +0.039], Holm p = 0.93 |

The Random Forest reference with 5 seeds gives 0.528 [0.489, 0.568].

**Reading:**
- **No model helps.** No model or selection procedure differs from the Random Forest, and every interval sits close to 0.5.
- **Collapse persists.** About 5 of 12 people receive one label for nearly all their trials.
- **Ranking without calibration:** within-person AUC (0.59–0.62) is somewhat above 0.5. The models rank a person's trials a little better than chance, but they place the decision threshold wrongly for each person.

### 4.2 Unsupervised calibration (N3): the new person's unlabelled EEG is used for normalization

| Variant | Trial bal. acc. [95% CI] | Macro-F1 | Single-label people | Δ vs RF (N3) |
|---|---|---|---|---|
| Random Forest (N3) | **0.575 [0.525, 0.624]** | 0.556 | 0 | — |
| Logistic regression, nested | 0.560 [0.498, 0.622] | 0.554 | 0 | −0.014 [−0.065, +0.041] |
| XGBoost, nested | 0.563 [0.512, 0.613] | 0.534 | 0 | −0.012 [−0.063, +0.042] |
| SVM, nested | 0.525 [0.454, 0.597] | 0.513 | 0 | −0.049 [−0.104, −0.003] |
| Nested selection over all 28 | 0.543 [0.494, 0.593] | 0.535 | 0 | **−0.031 [−0.051, −0.009]**, p = 0.026 |

**Reading:**
- **Collapse removed.** Per-person z-scoring removes single-label collapse completely (0 participants).
- **Improvement over N0 is not established.** RF N3 − N0 = +0.042 [−0.013, +0.092], p = 0.20.
- **Inner-loop selection hurt.** Choosing among 28 options with 11 people *lowered* performance relative to simply keeping the Random Forest. This is a concrete sign that the development set is too small to support model selection.

**Important caveat:**
- **Same videos in training.** In leave-one-participant-out, other people's EEG for the **same 40 music videos** is in the training set. Earlier work found the within-person signal to be largely shared and stimulus-related (`RESEARCH_FINDINGS_AND_NEXT_STEPS.md` §6, §10), so this design can benefit from recognising the stimulus.
- **The stricter estimate is lower.** When test videos are also held out (P3 with 0 calibration videos, next section), the same normalized Random Forest scores **0.535 [0.492, 0.578]**. That exactly replicates the earlier normalization study (0.535).
- **Which estimate to use:** 0.535 is the realistic estimate for a new person watching new material.

### 4.3 Trial aggregation (Random Forest)

| Normalization | Mean | Median | Vote | Log-odds |
|---|---|---|---|---|
| N0 | 0.533 | 0.526 | 0.527 | 0.533 |
| N3 | 0.575 | 0.575 | 0.572 | 0.575 |

**Reading:** aggregation does not matter. The protocol's mean-probability rule is kept.

### 4.4 Feature ablation (Random Forest, N0)

Trial balanced accuracy, with Δ vs the full set and its bootstrap CI.

**Full feature set (440 features):** 0.533.

**Frequency bands and feature families:**

| Feature set | Bal. acc. | Δ vs full |
|---|---|---|
| Without gamma | 0.526 | −0.007 [−0.032, +0.016] |
| θ only | 0.503 | −0.030 [−0.071, +0.010] |
| α only | 0.519 | −0.014 [−0.056, +0.029] |
| β only | 0.543 | +0.010 [−0.018, +0.050] |
| γ only | 0.518 | −0.015 [−0.054, +0.035] |
| Non-band (entropy, Hjorth, variance) | 0.527 | −0.006 [−0.037, +0.025] |
| Asymmetry only | 0.499 | −0.034 [−0.074, +0.005] |

**Channel regions:**

| Feature set | Bal. acc. | Δ vs full |
|---|---|---|
| Frontal | 0.532 | −0.001 [−0.047, +0.060] |
| Central | 0.540 | +0.007 [−0.027, +0.043] |
| Temporal | 0.525 | −0.008 [−0.062, +0.053] |
| Parietal | 0.485 | −0.047 [−0.102, −0.004] |
| Occipital | 0.530 | −0.003 [−0.044, +0.036] |
| Without artifact-prone sites (Fp1/2, AF3/4, F7/8, T7/8) | 0.518 | −0.015 [−0.037, +0.008] |

**Reading:** no subset improves on the full set. Every variant stays near chance.

### 4.5 Regression, borderline exclusion and data quantity

| Variant | Trial bal. acc. [95% CI] |
|---|---|
| RF regression of valence, threshold 5.0 (N0) | 0.499 [0.489, 0.508] (10 of 12 single-label) |
| RF regression of valence, threshold 5.0 (N3) | 0.531 [0.509, 0.554] (−0.043 vs the N3 classifier) |
| S1, borderline excluded (N0) | 0.531 [0.510, 0.552] |
| S1, borderline excluded (N3) | 0.591 [0.531, 0.652]. *A different, easier question (Protocol §7); not a primary-task result.* |
| 3 / 6 / 9 / 11 training participants (N0) | 0.516 / 0.519 / 0.524 / 0.533 |

**Reading:**
- **Regression:** continuous valence regression followed by thresholding was **worse** than direct classification.
- **More training people:** the curve rises only slightly. Within this range there is no sign of a steep gain from more people, but 11 people is far too few to rule one out.

## 5. P2 — within-participant supervised personalization

32 rated calibration videos per person; each person's 8 held-out videos are tested.

| Variant | Trial bal. acc. [95% CI] | Macro-F1 | Within-person AUC | Range over people | Δ vs RF [bootstrap CI] |
|---|---|---|---|---|---|
| **Random Forest reference** | **0.618 [0.548, 0.688]** | 0.617 | 0.644 | 0.41–0.76 | — |
| RF, nested | 0.613 [0.544, 0.683] | 0.613 | 0.642 | 0.41–0.75 | −0.004 [−0.009, +0.000] |
| XGBoost, nested | 0.588 [0.517, 0.659] | 0.584 | 0.615 | 0.43–0.76 | −0.030 [−0.055, −0.009] |
| Logistic regression, nested | 0.634 [0.568, 0.699] | 0.632 | 0.655 | 0.47–0.80 | +0.016 [−0.010, +0.044] |
| SVM, nested | 0.635 [0.574, 0.696] | 0.627 | 0.669 | 0.44–0.75 | +0.017 [−0.016, +0.063] |
| **Nested selection over all 28** | **0.620 [0.559, 0.681]** | 0.618 | 0.645 | 0.45–0.75 | +0.002 [−0.022, +0.031], Holm p = 0.93 |

**Replication:**
- The Random Forest reference with 5 seeds gives **0.618 [0.550, 0.687]**. This is identical to the earlier calibration study's 0.618 [0.550, 0.687].
- So the pipeline here reproduces the earlier result exactly.

**Fixed configurations, chosen after the fact:**

| Configuration | Bal. acc. | Note |
|---|---|---|
| Logistic regression, C = 0.001 | 0.642 [0.576, 0.709] | Highest of the 7 |
| SVM, C = 0.1 | 0.640 [0.581, 0.698] | |
| Random Forest | 0.618 | |
| XGBoost | 0.584 | Lowest of the 7 |

These are the best of 7 configurations picked *after seeing* outer results, so they are **optimistic**. The honest estimate of "choose the best of these" is the nested 0.620.

**Calibration size (Random Forest, personal data only):**

| Rated calibration videos | 8 | 16 | 24 | 32 |
|---|---|---|---|---|
| Bal. acc. | 0.562 | 0.601 | 0.603 | 0.618 |

**Other variants (Random Forest, 32 videos):**

| Variant | Result |
|---|---|
| Regression, then threshold 5.0 | 0.564, **worse** (−0.054 [−0.103, −0.008]) |
| S1, borderline excluded | 0.635. *Easier question; not the primary task.* |
| Aggregation | 0.612–0.618; no difference |

**Ablation:** trial balanced accuracy and Δ vs the full set (0.618).

| Feature set | Bal. acc. | Δ vs full |
|---|---|---|
| Without gamma | 0.614 | −0.004 |
| θ only | 0.590 | −0.028 |
| α only | 0.579 | −0.039 |
| β only | 0.600 | −0.018 |
| γ only | **0.620** | +0.002 |
| Non-band | 0.616 | −0.002 |
| Asymmetry only | 0.616 | −0.001 |
| Frontal | 0.590 | −0.028 |
| Central | 0.612 | −0.006 |
| Temporal | **0.573** | −0.045 [−0.087, −0.001] |
| Parietal | 0.595 | −0.023 |
| Occipital | 0.609 | −0.009 |
| Without artifact-prone sites | 0.609 | −0.009 |

**Reading:**
- **Redundant information.** Gamma alone matches the full set, yet removing gamma costs nothing. The information is spread across bands.
- **Artifacts are not the explanation.** The signal is not concentrated at the frontopolar and temporal sites where eye and muscle artifacts are strongest:
  - temporal-only is the weakest set;
  - dropping those sites leaves 0.609.
- **But muscle can't be fully excluded.** Gamma is also where muscle activity shows. That gamma alone suffices means some contribution from muscle cannot be ruled out with these features.

## 6. P3 — population model plus personal calibration

Random Forest, per-person unsupervised normalization; test videos are excluded from all training data.

| Rated calibration videos | Pooled (others + own) | Personal only (P2) | Pooled − zero-shot | Pooled − personal-only |
|---|---|---|---|---|
| 0 (zero-shot on new videos) | **0.535 [0.492, 0.578]** | — | — | — |
| 8 | 0.576 [0.532, 0.620] | 0.562 | +0.041 [−0.009, +0.097], 7/12 better | +0.014 [−0.005, +0.033] |
| 16 | 0.594 [0.530, 0.659] | 0.601 | +0.059 [−0.009, +0.126], 10/12 better | −0.007 [−0.018, +0.004] |
| 32 | 0.601 [0.527, 0.675] | **0.618** | +0.066 [−0.009, +0.140], 8/12 better; Holm p = 0.39 | **−0.017 [−0.027, −0.006]**, p = 0.023 |
| 32, unweighted pooling | 0.590 [0.512, 0.668] | — | — | — |

**Reading:**
- **Gains with calibration:** personal ratings raise balanced accuracy from 0.535 to about 0.60–0.62.
- **Not formally significant.** The pre-declared pooled-32 comparison does not survive correction: its bootstrap CI touches 0 and Holm p = 0.39. Personal-only 32 vs zero-shot (+0.083 [+0.011, +0.155]) is descriptive.
- **Pooling helps only at small calibration sets.** Adding other people's data helps slightly at 8 videos and **hurts** at 32. With enough personal ratings, the person's own model is better.
- **Who improves:**
  - **Helped by calibration:** 7–10 of 12, depending on the number of videos.
  - **Never reliably helped:** s07, s11 and s25 stay at or below about 0.54 in every personalized condition.

## 7. Diagnosing the performance ceiling

| Candidate cause | Test performed | Finding |
|---|---|---|
| **Participant variability / per-person offsets** | Single-label counts, N3 normalization, within-person AUC | **Major factor for new people.** N0 collapses ~5/12 people onto one label, while within-person AUC is 0.59–0.62: trials are ranked somewhat, but the person's threshold is wrong. Per-person normalization removes the collapse but gains only +0.04 (n.s.). |
| **Weak generalizable signal** | P1 every model; P3 at 0 videos | **Major factor.** On new videos and new people every approach is at 0.50–0.54. |
| **Small dataset** | Learning curve over 3–11 people; nested-selection behaviour | **Contributes.** The curve rises slightly (0.516 → 0.533). Inner selection with 11 people chose worse models than a fixed default under N3. Too few people to tune anything. |
| **Noisy labels near the threshold** | S1; accuracy vs \|valence − 5\| | **Contributes modestly.** Within-person accuracy is 0.589 on borderline trials vs 0.646 on clear trials; r = 0.074, p = 0.005. Removing borderline trials moves P2 from 0.618 to 0.635: real but small, and a different task. |
| **Muscle or eye artifacts** | Gamma ablation; excluding artifact-prone sites | **No evidence it drives results;** not excluded. Removing gamma or artifact-prone sites changes little, but gamma alone carries as much within-person information as all features. |
| **Feature extraction** | 14 family and region subsets | **No subset helps.** Bands are largely redundant. Better *representations* (e.g. covariance / Riemannian, Protocol D4) were not tested here. |
| **Normalization** | N0 vs N3 | Removes collapse; small, non-significant gain. |
| **Model choice** | RF, XGBoost, LR, SVM with nested selection | **Not the bottleneck.** All within ±0.02 of each other except XGBoost, which was slightly worse. |
| **Trial aggregation** | Mean, median, vote, log-odds | **No effect.** |
| **Continuous target** | RF regression, then threshold | **Worse** than classification in all three settings. |

**Overall:**
- **Not the model.** The ceiling is not mainly the choice of classifier or a fixable pipeline detail.
- **The data limits it.** The current features carry a weak, partly person-specific and partly stimulus-shared valence signal. Cross-person transfer fails, mostly because of per-person offsets, and 12 people are too few to learn or select around that.

## 8. Is 90% achievable? Leakage demonstration

| Setting | Valid? | Trial balanced accuracy |
|---|---|---|
| Best valid participant-independent mean (nested, N0) | Yes | 0.541 [0.504, 0.579] |
| Best valid personalized mean (nested, P2) | Yes | 0.620 [0.559, 0.681] |
| Best post-hoc personalized configuration (LR, C = 0.001) | Optimistic | 0.642; upper CI 0.709 |
| Best single person, seed-averaged, valid | Yes, but cherry-picked | 0.758 (s15, RF) |
| **Random 5-fold split of segments within a person** | **No — leaky by design** | **0.805 [0.776, 0.834]**; segment accuracy 0.731 |

**90% was not achieved under any valid protocol, and no valid confidence interval reaches it.**

**How the leaky number arises:**
- **Mechanism.** If 4 s segments of the same 60 s video are allowed into both training and test, neighbouring near-duplicate segments let the model recognise the *recording*, not the valence.
- **Size of the effect.** That alone lifts the score to 0.80, without any change to the model or features.
- **Implication.** High DEAP accuracies reported with segment-level splits should be read with this in mind.

## 9. Best supported model

These are the predefined criteria:
- leakage-free nested estimates;
- the trial balanced-accuracy primary metric;
- no chasing of a single number.

| Use case | Best defensible choice | Estimated trial balanced accuracy | Status |
|---|---|---|---|
| New, uncalibrated user | **Keep the registered Random Forest.** No alternative is distinguishable from it. | ~0.53 [0.49, 0.58]; near chance | Not useful for prediction |
| New user, unlabelled recording available | RF with per-person normalization | 0.535 on new videos; 0.575 when the same videos appear in training | Removes collapse; not reliable |
| **User provides ~16–32 rated videos** | **Per-person model** (RF, or nested selection; LR and SVM were not significantly better) | **~0.60–0.62** [~0.55, ~0.69] | The only setting reliably above chance in this sample |

**Isolated candidate artifact:**
- **What was run.** The P1 inner-selection procedure, run on all 12 participants, chose **logistic regression (C = 0.001) with mean aggregation**, with an inner score of 0.535.
- **What was saved.** It was refit on all 12 participants and saved to `outputs/experiments/model_improvement_20261021/candidate/` (`model.joblib`, `metadata.json`).
- **Its honest expected performance** is the nested estimate of that procedure: **0.541 [0.504, 0.579]**. That is **not** better than the active Random Forest.
- **Status.** It is **not registered, not activated and not recommended for activation.** It exists only so the selection result is reproducible.

**Personalized models** are per person, so no single artifact applies. They are reproduced by the P2 phase.

## 10. Verification and boundaries

| Check | Result |
|---|---|
| Active model pointer | Still `valence_binary-random_forest-20261008-180558` |
| Production artifacts | `outputs/models`, `outputs/metrics`, `outputs/predictions`, `outputs/graphs`, the protocol and `participants.py`: **70 files, SHA-256 identical** before and after |
| Production PostgreSQL | Read-only snapshot before and after: users, predictions, reports, audit logs, sessions, models and metrics **all unchanged**; no write transaction |
| Frozen test participants | Never loaded. `load_data()` asserts their absence. The feature cache contains only the 12 development participants. |
| Unapproved Kaggle copy | Not used |
| Historical results | Not overwritten. New outputs are only in `outputs/experiments/model_improvement_20261021/` (git-ignored) and `docs/results/exploratory/model_improvement/`. |
| Backend tests | `python -m pytest -q`: **241 passed** |
| Commits, pushes, deployments | **None** |

## 11. Limitations

- **Small sample.** With 12 participants, intervals are wide (about ±0.06–0.07 for personalized means), and nested model selection is itself unreliable (§4.2).
- **No independent test.** The final 8-person group is frozen and blocked on Decision I.
- **Stimulus overlap.** Every DEAP participant watched the same 40 videos. Personalized and leave-one-participant-out results may partly reflect shared, stimulus-related responses (`RESEARCH_FINDINGS_AND_NEXT_STEPS.md` §6). Results on held-out videos (P2, P3) are the more conservative estimates.
- **Fixed features.** Only the existing 440 handcrafted features were tested. Covariance or Riemannian representations, other segment lengths and deep models were out of scope.
- **Muscle not ruled out.** A contribution from muscle (EMG) to the gamma-band within-person signal cannot be ruled out.
- **Same seeds, different meanings.** Seeds 42–44 were used for exploratory variants and 42–46 for the reference, per Protocol §4. Seed SDs were small (≤ 0.033).
- **Run interruptions.** The run was interrupted twice by session restarts and resumed from saved per-job results. Every job is deterministic given its seed, so a resumed run produces the same results as an uninterrupted one. The only code change during the run was the save frequency (every 24 jobs to every 8), which does not affect results.

## 12. Recommended next steps

1. **Resolve Decision I and obtain the official DEAP release.**
   - **Why:** it is the only route to more development participants (24), a meaningful learning curve and an independent test.
   - **What it means for model choice:** nothing here justifies more cross-person model tuning on 12 people.
2. **If the product needs predictions now, frame it as personalized calibration.**
   - **How:** the user rates about 16–32 videos, and a per-person model is fitted.
   - **How to present it:** as an exploratory estimate of about 0.6 balanced accuracy, never as emotion detection.
3. **Done on 2026-10-23, negative:** the highest-value representation experiment was per-person covariance alignment with Riemannian tangent-space features (Protocol D3 N4 / D4). See `COVARIANCE_ALIGNMENT_EXPERIMENT.md`: 0.499 vs baseline 0.493, below simple z-scoring. Model experimentation on these 12 participants is now frozen. The original recommendation follows.
   - **Why:** it targets the main failure mode found here, per-person offsets.
   - **How:** it should be pre-declared and run on the full development set.
4. **Test artifact robustness explicitly,** for example with EMG-sensitive features or ICA-cleaned data, before attributing the within-person gamma signal to valence.
5. **Do not change the active model.** No candidate passes the Protocol §10 criteria.

## 13. Reproducing

All commands run from `backend/`, with the virtual environment active. The design is in the script docstring.

| Command | What it does | Runtime on this machine (8 workers) |
|---|---|---|
| `python -m scripts.model_improvement --phase baseline` | Re-derives and re-scores the active model's historical results | ~1 min |
| `python -m scripts.model_improvement --phase leak` | Leakage demonstration | ~1 min |
| `python -m scripts.model_improvement --phase p2` | 4,620 within-person fits | ~40 min |
| `python -m scripts.model_improvement --phase p3` | 900 pooled calibration fits | ~1.5 h |
| `python -m scripts.model_improvement --phase p1` | 1,008 nested leave-one-participant-out jobs | ~2 h |
| `python -m scripts.model_improvement --phase candidate` | Inner selection on all 12 participants; writes the isolated candidate | ~5 min |
| `python -m scripts.model_improvement --phase report` | Writes `summary.json` | ~1 min |
| `PYTHONIOENCODING=utf-8 python -m scripts.model_improvement_tables > ../docs/results/exploratory/model_improvement/tables.md` | Generates every table in this document from `summary.json` | seconds |

- **Resuming:** phases can be resumed; finished jobs are skipped.
- **Full tables:** all tables, including per-participant values for every protocol, are in `docs/results/exploratory/model_improvement/tables.md`.
