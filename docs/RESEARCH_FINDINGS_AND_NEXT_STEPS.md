# NeuroSense AI: Research Findings and Next Steps

**Status:** all findings below are **exploratory (12 participants, Protocol v1.1)**.

- There are no final-test results. The frozen final test group has never been used. Its evaluation is **BLOCKED** until Decision I is resolved.
- Under Decision C, nothing here supports a formal claim that any model outperforms the Random Forest, or any claim about people outside this sample.
- Compiled on 2026-10-13 from the reports and saved result files listed in §13. **Updated 2026-10-23** with:
  - the model-improvement experiments (§9a);
  - the covariance-alignment experiment (§9b);
  - a note on why headline numbers from different protocols cannot be compared (§9c).
- **Model experimentation on the current 12 participants is frozen** (2026-10-23). No further model, feature or tuning experiments will be run on them; the next step is data (§11).

> **What this project measures.** NeuroSense AI estimates a binary category of **self-reported valence**. That is a person's own rating of how pleasant a music video felt, given once after each 60 s trial. It does not detect emotions, measure a person's exact emotional experience, or provide any diagnosis.

## 1. Research objective

**Goal:** estimate from EEG whether a person rated a stimulus as relatively pleasant or unpleasant.

**Data:** DEAP `data_preprocessed_python`.
- 32 EEG channels, 128 Hz, 40 one-minute music-video trials per participant.
- After each trial the participant rated valence on a 1–9 scale.

**Label:**
- **Rule (Decision F):** `high_valence` if valence > 5.0, otherwise `low_valence`.
- **Granularity:** one label per trial, inherited by every 4 s segment of that trial.
- **Borderline trials:** 90 of the 480 trials (19%) lie in the borderline band 4.5–5.5. They are analysed separately only as Protocol secondary analysis S1.

**Pipeline:**
- 4–45 Hz band-pass, per-trial robust (MAD) clipping and segment rejection, 4 s segments.
- **Features:** 440 per segment, fingerprint `3c5359083bc90ae5`. There are 7,046 accepted segments; 154 were rejected.
- **Trial prediction:** in all exploratory experiments, the mean of segment probabilities, then argmax.

**Evaluation questions** (distinguished throughout):

| Question | Meaning |
|---|---|
| **Participant-independent ("zero-shot")** | Train on other people, predict a new person. This is the original product goal. |
| **Unsupervised calibration** | As above, but the new person's **unlabelled** EEG may be used to normalize features. |
| **Supervised within-person calibration** | Train on a person's own **rated** trials and predict that person's ratings for other videos. |

## 2. Dataset limitations

| Item | Status | Source |
|---|---|---|
| Trusted data | **12 participants** (s01, s02, s03, s05, s07, s08, s09, s11, s15, s16, s22, s25), 480 trials. All results in this document use only these files. | Protocol §1; all exploratory reports |
| Trial-to-video alignment | Trial index *t* is the same video for every participant. Verified from the trusted ratings alone: mean pairwise Spearman ρ 0.443, against a shuffled maximum of 0.060 over 1,000 shuffles. | `EXPLORATORY_WITHIN_PERSON_DIAGNOSTIC.md` §9.2 |
| The other 20 DEAP participants | The only copy available is a **third-party (Kaggle) upload**. Its EEG for our 12 participants is identical to the trusted files, but **on about 30% of trials both valence and arousal were replaced by 9 − x**. Its `participant_ratings.xls` matches our 12 trusted participants on all 1,920 rating values. | Protocol §6.4 |
| Integrity check of the 20 (Decision K) | Completed; **all groups passed** (aggregate results in §2.1). A pass shows consistency only, not authenticity or permission to use the data. | Protocol §6.4, Decision K; §2.1 below |
| Licensing and authenticity | The third-party copy is **not authorized** for use and has **not** been integrated. DEAP is distributed under an end-user licence agreement through the official channel (Queen Mary University of London). | Protocol §8.1 |
| **Decision I (data access and storage)** | **PENDING.** Not decided: who signs the DEAP EULA, who downloads the data, where files are stored, and whether the duplicate local copy is removed. Until it is decided: no signing, downloading, moving or deleting, and no configuration change. | Protocol §12 |
| Frozen final test group | **s12, s17, s18, s20, s21, s23, s29, s30** (seed 20261010), frozen before any of their ratings or signals were examined. Never used. Their usable labels depend on Decision I. | Protocol §8.2 freeze record |
| Other datasets | DENS was assessed for feasibility earlier but not adopted. **No DENS data was used.** | Exploratory reports, data statements |

**Consequences:**
- **Small sample:** every estimate rests on 12 people, with per-person intervals about ±0.15 wide.
- **No independent test:** none is possible until Decision I is resolved.

### 2.1 Kaggle DEAP copy: integrity-check results (aggregate only)

These results were reported earlier and are restated here without rerunning any check or rereading any file. Per the protocol rules (§6.4, §8.2), only aggregate results are given for the frozen test participants: no per-participant results, rating values, class balances, signal statistics or transformation patterns.

| Check | Result |
|---|---|
| Files present in the candidate copy | All 32 participant files (`s01.dat`–`s32.dat`) |
| EEG, 12 overlapping (trusted) participants | Identical to the trusted files |
| `participant_ratings.xls` vs the 12 trusted participants | Matches all four rating fields (valence, arousal, dominance, liking) under the `Experiment_id` mapping (`.dat` trial *t* ↔ `Experiment_id` *t* + 1) |
| Consistency check, trusted controls | **12 / 12 passed** |
| Consistency check, additional development participants | **12 / 12 passed** |
| Consistency check, frozen test participants | **8 / 8 passed** |
| Failure counts in every group | **0** structural, **0** dominance/liking, **0** valence/arousal-pair, **0** range |

**What this does and does not establish:**
- The Kaggle `.dat` labels may contain the documented `9 − x` transformation of valence and arousal. The workbook and the `.dat` files are **consistent** with each other under that known alteration.
- **Consistency is not authenticity.** A consistent edit to both files would also pass.
- **Unverified data:** the EEG and ratings of the 20 additional participants have **not** been independently verified against any official source.
- **Authorization unresolved:** Decision I is still pending.
- **Nothing reconstructed or integrated:** no reconstructed dataset has been created, and nothing from this copy is used in any experiment or in the application.

## 3. Baseline comparison (Random Forest, SVM, XGBoost, CNN)

These are **pre-protocol** results (before Protocol v1.0).
- **Fixed split:** train s02, s03, s05, s07, s08, s15, s22; validation s16, s25; test s01, s09, s11.
- Under Decision B, s01, s09 and s11 are now development participants, so these test numbers are **historical**.
- Each model was trained once on the fixed split. Only the CNN was later re-run with 5 seeds (§4).

**Held-out test participants (s01, s09, s11), single run:**

| Model | Segment acc. | Segment bal. acc. | Segment macro-F1 | Segment AUC | Trial acc. | Trial bal. acc. | Trial macro-F1 | Trial AUC |
|---|---|---|---|---|---|---|---|---|
| Random Forest | 0.548 | 0.544 | 0.543 | 0.537 | 0.592 | 0.590 | 0.590 | 0.565 |
| SVM (RBF) | 0.498 | 0.486 | 0.460 | 0.433 | 0.475 | 0.459 | 0.406 | 0.421 |
| XGBoost | 0.500 | 0.499 | 0.499 | 0.478 | 0.517 | 0.515 | 0.515* | 0.471 |
| CNN (EEGNet-style, seed 42) | 0.542 | 0.534 | 0.525 | 0.577 | 0.542 | 0.533 | 0.524 | 0.590 |
| Majority-class reference | 0.525 | 0.500 | 0.344 | 0.500 | 0.525 | 0.500 | 0.344 | 0.500 |

\*The saved value is 0.5145, shown as 0.515 in `model_comparison_20261006-050615.md` and as 0.514 in `REVIEW3_PROGRESS_30.md`. It is a rounding difference; the saved JSON is the source.

**Participant-independent (leave-one-participant-out) results, 9 non-test participants:**

| Model | Level | Metric | Value | Source and notes |
|---|---|---|---|---|
| Random Forest (registered) | segment | accuracy | 0.510 ± 0.086 (population SD) = ± 0.091 (sample SD) | `REVIEW2_REPORT.md`; `REVIEW3_CNN_ROBUST_EVALUATION.md` §5. **Same folds, two SD conventions.** |
| Random Forest | segment | macro-F1 / AUC | 0.416 / 0.538 | as above |
| Random Forest | **trial** | accuracy | 0.503 ± 0.100 | robust CNN report §5. The "0.503" quoted in the within-person report §5 is this **trial-level** accuracy, not the 0.510 segment-level figure. |
| Random Forest | — | balanced accuracy | **not recorded** for this pre-protocol LOPO | — |
| CNN, train_stats normalization | segment | balanced acc. / macro-F1 / AUC | 0.494 ± 0.023 / 0.383 ± 0.065 / 0.506 ± 0.102 | robust CNN report §5 (seed 42 only) |
| CNN, per-participant normalization | segment | balanced acc. / macro-F1 / AUC | 0.505 ± 0.016 / 0.408 ± 0.051 / 0.550 ± 0.136 | as above |
| Majority-class reference | segment | accuracy | 0.528 ± 0.093 | as above |
| SVM, XGBoost | — | — | **No participant-independent CV result recorded** | Their only results are the fixed-split values above. |

**Reading:**
- On the 3 historical test participants, the Random Forest scored highest. But its within-participant signal came mainly from s09 (robust CNN report §6).
- Under leave-one-participant-out, **no model was distinguishable from chance or from the majority-class reference**.

## 4. Robust CNN evaluation (pre-protocol; `REVIEW3_CNN_ROBUST_EVALUATION.md`)

- **Seed sensitivity:**
  - Same data and settings, 5 seeds, same 3 test participants. Test segment macro-F1 was **0.466 ± 0.109** (range 0.322–0.554), and segment AUC ranged 0.445–0.577.
  - The registered seed-42 result (0.525) is one of the better seeds.
  - Selected epochs ranged from 1 to 31, and validation macro-F1 did not predict test macro-F1.
- **Cross-participant:**
  - Leave-one-participant-out balanced accuracy was 0.494 ± 0.023 (train_stats) and 0.505 ± 0.016 (per-participant normalization).
  - Neither differs from 0.5 (Wilcoxon p = 0.63 and 0.20) or from the majority reference.
  - No CNN-vs-RF difference is detectable with 9 folds.
- **Single-class prediction:**
  - Per test participant, CNN balanced accuracy was 0.500–0.521.
  - The share of a person's segments predicted "high" swung between about 0 and 1 across seeds (SD ≈ 0.42–0.45).
  - The CNN mostly assigned **one label per person**. Its pooled accuracy came from which label each whole person received, not from separating that person's trials.
- **Per-participant normalization did not fix this.** It was worse on the fixed split (macro-F1 0.422 vs 0.466) and only moved leave-one-participant-out accuracy toward the majority class.
- **Status:** the CNN remains registered but **inactive**, and the Random Forest remains the active model. A proposed CNN leave-video-out evaluation (1,200 fits, about 70–110 CPU hours) was **not run**, by decision of 2026-10-12.

## 5. Within-person diagnostic (`EXPLORATORY_WITHIN_PERSON_DIAGNOSTIC.md` §1–8)

**Design:** the registered Random Forest, trained and tested **inside one participant**.
- Stratified 5-fold over the 40 trials; seeds 42–46; trial-level metrics.
- Permutation tests: 250 permutations, seed 42, with BH correction.

**Results:**

| Measure | Value |
|---|---|
| Mean trial balanced accuracy | **0.625** (SD 0.094), 95% t-CI [0.566, 0.685] |
| Trial macro-F1 / ROC-AUC | 0.625 / 0.667 |
| Wilcoxon p vs 0.5 | 0.0024 (11 of 12 participants above 0.5) |
| Group permutation p | 0.004, the minimum possible with 250 permutations |
| Individually significant after BH | 2 of 12 (s15, s22) |

**Limitations:**
- **Folds use labels:** they are stratified on the labels.
- **Stimulus confound:** it is not controlled (addressed in §6).
- **Drift and power:**
  - slow drift cannot be tested, because presentation order is not in the trusted files;
  - 40 trials per person give wide intervals;
  - the permutation resolution of 250 limits individual tests.

## 6. Stimulus-consensus control (`EXPLORATORY_WITHIN_PERSON_DIAGNOSTIC.md` §9)

**Design:** within-person, as in §5, but folds are jointly stratified on (own label, consensus label). Consensus is the majority `valence > 5.0` vote of the **other 11** participants for the same video.

**Results:**

| Measure | Value |
|---|---|
| Own-rating balanced accuracy | 0.619, CI [0.547, 0.691] |
| Consensus balanced accuracy | **0.649**, CI [0.557, 0.741] |
| Own − consensus | −0.029, t-CI [−0.108, 0.049]; exact sign-flip p = 0.42; consensus higher for 9 of 12 |
| Disagreement trials (own ≠ consensus) | The own-rating model followed the own label on 56 of 117 pooled trials (0.479). Mean per-participant share 0.436, sign-flip p = 0.32. |
| Own/consensus label agreement | 0.756; κ = 0.514 |

**Interpretation:** the within-person signal is consistent with a substantial **shared, stimulus-related** component.

**Limitation:** the consensus label is itself a shared affective rating. This design **cannot separate a shared emotional response from low-level stimulus properties** (music, visuals, tempo). The trusted data contains no stimulus features to test this.

## 7. Cross-participant, leave-video-out (`EXPLORATORY_CROSS_PARTICIPANT_LEAVE_VIDEO_OUT.md`)

**Design:**
- Hold out one participant **and** one fold of 8 videos. Train on the other 11 participants' remaining videos only.
- Seeded, label-free video folds (seeds 42–46), 600 fits.
- Consensus training labels exclude the example's own participant and the held-out participant. 228 tied training pairs (4.3%) were excluded in both tasks.

**Results:**

| Target | Mean trial balanced accuracy | 95% t-CI | Holm p vs 0.5 |
|---|---|---|---|
| Own rating (A) | **0.500** | [0.481, 0.520] | 0.96 |
| Consensus (B) | **0.482** | [0.453, 0.512] | 0.65 |
| A − B | +0.018 | [−0.019, 0.056] | 0.65 |

- **Both targets stayed near chance** for an unseen person on unseen videos.
- **Prediction collapse:**
  - 4 participants (own rating) and 2 participants (consensus) received near-single-class predictions (mean predicted-high share < 0.05 or > 0.95);
  - the mean gap between predicted and true high share was 0.30–0.31.

## 8. Unsupervised participant normalization (`EXPLORATORY_PARTICIPANT_NORMALIZATION.md`)

**Design:**
- The same folds, seeds, training rows and model as §7.
- Each participant's 440 features were z-scored using that participant's **unlabelled** EEG on the 32 non-test videos.
- This is an **unsupervised-calibration (transductive) condition, not zero-shot**.

**Results, own-rating task (primary):**

| Measure | A (none) | B (normalized) | B − A |
|---|---|---|---|
| Trial balanced accuracy | 0.500 | 0.535 | **+0.034**, t-CI [−0.008, 0.076], bootstrap [−0.003, 0.067]; **primary exact p = 0.098**; 10 of 12 improved |
| Macro-F1 | 0.424 | 0.517 | +0.093, CI [0.044, 0.143] |
| ROC-AUC | 0.531 | 0.548 | +0.017, CI [−0.089, 0.123] |
| Participants with collapsed predictions | 4 | **0** | — |
| Single-label seeds | 18 | **0** | — |
| Mean \|predicted − true high share\| | 0.30 | 0.16 | — |

**Consensus control:** 0.482 → 0.502 (B − A Holm p = 0.75).

**Interpretation:**
- Normalization **removed single-class collapse** and improved macro-F1. Macro-F1 heavily penalizes collapse, so this gain is partly mechanical.
- It did **not** produce a significant balanced-accuracy gain.
- It did **not** improve discrimination: ROC-AUC barely changed.
- Predictions still leaned high (mean predicted-high share 0.66 vs 0.53 true).

## 9. Supervised personalized calibration (`EXPLORATORY_WITHIN_PERSON_CALIBRATION.md`)

**Design:**
- Within one participant, train on 32 rated videos and test on 8 held-out videos.
- **Label-free video folds identical to §7–§8**, so all conditions share the same test trials.
- Seeds 42–46, both tasks.
- Nested learning curve with 8/16/24/32 calibration videos.
- 500 label permutations per participant with BH correction; all statistical tests declared before running.

**Results:**

| Measure | Value |
|---|---|
| **Own-rating trial balanced accuracy** | **0.618**, t-CI [0.550, 0.687]; primary exact p = 0.004 |
| Own-rating macro-F1 / ROC-AUC | 0.617 / 0.648 [0.572, 0.723] |
| vs zero-shot cross-person (§7 A), same trials | **+0.118**, CI [0.048, 0.187]; Holm p = 0.013; 11 of 12 better |
| vs unsupervised calibration (§8 B), descriptive | +0.083, CI [0.007, 0.160] |
| **Consensus balanced accuracy (same folds)** | **0.626**; own − consensus −0.008, CI [−0.090, 0.075]; Holm p = 1.0 |
| **Disagreement trials** | own-rating model followed the own label on 60 of 117 (mean share 0.464); Holm p = 1.0 |
| Individually significant (BH q < 0.05) | 4 of 12: s03, s05, s15, s22 |
| Collapse | none; mean \|predicted − true share\| 0.03 |
| S1, borderline excluded from training and scoring | 0.632 [0.561, 0.703] |

**Learning curve** (descriptive, no tests):

| Calibration videos | 8 | 16 | 24 | 32 |
|---|---|---|---|---|
| Balanced accuracy [95% t-CI] | 0.579 [0.529, 0.629]* | 0.600 [0.542, 0.657] | 0.615 [0.555, 0.675] | 0.618 [0.550, 0.687] |

\*9 participants only: 4 (participant, seed) cases had single-class training subsets.

**Three estimates of the same within-person quantity:** 0.625 (§5, label-stratified folds), 0.619 (§6, jointly stratified folds) and 0.618 (§9, label-free video folds). They come from **different fold schemes** and are not identical runs. Their agreement is a replication, not one result reported three times.

**Statistical limitations:**
- only 117 disagreement trials (3–20 per person), so the individual-specific test has low power;
- per-participant estimates vary by ±0.02–0.09 across seeds;
- 12 participants;
- the learning curve has no tests.

## 9a. Model-improvement experiments (`MODEL_IMPROVEMENT_EXPERIMENTS.md`, 2026-10-22)

**Design:**
- **Protocols:** nested leave-one-participant-out, participant-independent; within-person video folds; population plus personal calibration.
- **Candidates (selected only inside training folds):** RF, XGBoost, logistic regression ×3, RBF SVM ×2, each with 4 trial-aggregation rules.
- **Also tested:** feature-family and region ablations, continuous-valence regression, borderline exclusion (S1), and a training-set-size curve.
- **Seeds:** 42–44.

| Finding | Value |
|---|---|
| Participant-independent, inductive (nested selection over 28 options) | 0.541 [0.504, 0.579]; RF 0.533. Δ +0.008 [−0.024, +0.039], Holm p = 0.93 |
| Model choice | RF, LR, SVM within ±0.02 of each other; XGBoost slightly worse. **Not the bottleneck.** |
| Aggregation, ablations, regression | No improvement; regression worse in every setting |
| Supervised personalization (32 videos), nested | 0.620 [0.559, 0.681]; RF 0.618 (identical to §9) |
| Calibration size, RF (personal-only / pooled) | 8 videos: 0.562 / 0.576; 16: 0.601 / 0.594; 32: 0.618 / 0.601. Zero-shot on new videos: 0.535. |
| Ratings close to 5 | Accuracy 0.589 on borderline vs 0.646 on clear trials (r = 0.074); a modest contributor |
| Leakage demonstration (random **segment** split, invalid) | 0.805, which shows how inflated DEAP figures arise |

**Conclusions:**
- **No model reached 90%** under any valid protocol; the highest upper CI bound was 0.71.
- **No candidate was activated.** An isolated, unregistered candidate (logistic regression; nested estimate 0.541) exists only for reproducibility.

## 9b. Per-person covariance alignment (`COVARIANCE_ALIGNMENT_EXPERIMENT.md`, 2026-10-23)

**Design:**
- **Features:** 4-band 32×32 covariance tangent-space features, 2,112 per segment.
- **Alignment:** each person re-centred on their own **unlabelled** EEG from non-test videos (unsupervised calibration).
- **Folds:** leave one participant **and** 8 videos out, seeds 42–44. These are identical folds for all conditions.

| Condition | Trial balanced accuracy [95% CI] |
|---|---|
| A0 baseline, 440 features (inductive) | 0.493 [0.470, 0.516] |
| A1 baseline + per-person z-score | 0.535 [0.492, 0.578] |
| A2 covariance, no alignment | 0.506 [0.471, 0.541] |
| **A3 covariance, per-person alignment** | **0.499 [0.458, 0.539]** |
| Within-person: 440 features / aligned covariance | 0.618 / 0.607 |

**Result: negative.**
- **Primary comparison (A3 − A0):** +0.006 [−0.025, +0.038], Holm p = 1.0.
- **Simpler normalization was better:** A3 was below simple z-scoring (A1) by 0.036 [−0.070, −0.003].
- **What alignment did:** it removed the between-person offset (18.1 → 0.6) and single-label collapse (3 → 0 people), but did not improve discrimination (within-person AUC 0.53).
- **Recommendation:** not to be pursued further on 12 participants.

## 9c. Why the headline numbers are not interchangeable

| Number | Metric | Who is tested | Training data | Why it differs |
|---|---|---|---|---|
| **59.2%** (active model, Review 3) | Trial **accuracy**, **pooled** over 120 trials | 3 fixed people (s01, s09, s11), one split, one seed | 9 fixed people, **all 40 videos**, including the same videos later tested | Not balanced for class mix; driven by s09 (0.75), while s01 got one label for every trial (0.50). The participant-level CI is [0.31, 0.93] (n = 3). These participants have since been examined repeatedly, so the result is historical, not an independent test. |
| **53.3%** (model-improvement P1, RF) | Trial **balanced accuracy**, mean of 12 people | Every person once (leave-one-participant-out), 3 seeds | Other 11 people, **all 40 videos** (the test videos' stimuli are in training) | Covers all 12 people, not 3 favourable ones |
| **49.3%** (covariance experiment A0) | Trial **balanced accuracy**, mean of 12 people | Every person × 5 video folds, 3 seeds | Other 11 people, **only the 32 non-test videos** | Strictest design: neither the person nor the video has been seen, so any stimulus-recognition advantage is removed |

**What follows:**
- **None of these is a better or worse "version" of the same model.** They answer different questions with different metrics. They must never be placed in one table as if comparable, or used to suggest that a model got worse or better.
- **The consistent message:** for a new person, every valid participant-level estimate is close to 0.5.

## 10. Scientific interpretation

**Supported by the evidence (this sample, this pipeline, Random Forest):**
1. **No zero-shot transfer.** Zero-shot prediction of a new person's valence category is at chance. This holds for the Random Forest and the CNN under leave-one-participant-out, and for the Random Forest when videos are also held out (0.500).
2. **Collapse comes from per-person offsets.** Cross-person failures show strong per-person prediction collapse. Unsupervised per-person normalization removes the collapse but does not demonstrably improve discrimination.
3. **Supervised calibration works.** A model trained on about 32 of a person's own rated videos predicts that person's binary ratings on **new videos** above chance (about 0.62 balanced accuracy, versus 0.50 zero-shot on the same trials). This was replicated across three fold schemes.
4. **The calibrated signal is shared.** It predicts the **other participants' consensus** for each video as well as the person's own rating. Where the person disagrees with the consensus, the model does not follow them.
5. **The bottleneck is not the model or the representation** (§9a, §9b).
   - **What didn't help:** four model families with nested selection, four aggregation rules, 14 feature subsets, regression, and per-person covariance alignment.
   - **What that means:** none changed the near-chance cross-person result. With 12 participants, model selection itself is unreliable.

**Hypotheses (not established):**
- **H1.** The within-person signal reflects EEG responses shared across people and related to the stimulus. These could be shared affect, low-level audiovisual properties, or both, and the current data cannot separate them.
- **H2.** Per-person feature offsets are the main reason cross-person models collapse. This is consistent with §7–§8 but not tested directly.
- **H3.** Individual-specific valence information may exist but be too weak to detect with 117 disagreement trials.

**Not supported, and must not be claimed:**
- that any model detects a person's exact or individual emotional experience;
- that any model works for new, uncalibrated users;
- that EEG **cannot** contain individual-specific emotional information (the null results here are low-powered and limited to one feature set and model);
- any product or clinical performance figure.

## 11. Recommended direction

| Option | Merits | Limitations |
|---|---|---|
| **A. Personalized calibration as the research framing** | The only setting with above-chance evidence, replicated across three fold schemes, with no collapse and well-matched class shares. It needs no new model. | Needs about 16–32 **rated** videos per user. The accuracy is modest (about 0.62). The signal is not shown to be individual. Results are based on 12 people with no independent test. |
| **B. Obtain an independently verified full DEAP dataset (Decision I)** | Required for **any** formal claim. It unlocks the frozen 8-person test group and adds 12 development participants. It roughly triples the disagreement trials for the individual-specific question (an estimate scaled from 117 trials over 12 people). It also removes the dependence on an altered third-party copy. | It is administrative (EULA, signatory, storage) and takes time. It doesn't improve any model by itself, and it needs a re-verification step when the data arrives. |
| **C. Develop the research interface** | Makes the existing pipeline and verified findings usable and presentable. It can display results honestly: a "calibrated, exploratory" mode and an explanation of what valence labels mean. It can proceed without new training. | It risks implying performance that hasn't been shown. The interface must not present uncalibrated predictions as emotion detection, must not show "Review 1/2/3" labels, and must not change the active model or API behaviour without a separate decision. |

**Recommendation:**
1. **Prioritize B (resolve Decision I and obtain the official DEAP release).** It is the only route to the independent evidence that every further claim depends on.
2. **In parallel, a scoped version of C** is reasonable: honest presentation of the existing results and the calibration framing, with no new models and no performance claims.
3. **Keep A as the working hypothesis.** Re-test it on the full development set before building product features around it.
4. **Model experimentation is frozen on the current 12 participants** (2026-10-23; §9a–§9b).
   - **What is kept:** the active Random Forest.
   - **What is not done:** no experimental candidate is registered or activated.
   - **When work resumes:** only after B, as a pre-declared comparison against the stronger references found here:
     - per-person z-scoring for new users (0.535 on new videos);
     - the personal Random Forest for calibrated users (0.618).
   - **How personalized calibration is described:** exploratory, at about 0.6 balanced accuracy, and never as emotion detection or a clinical tool.

## 12. Decision criteria before new models or product-performance claims

**Before starting any new model or architecture:**
1. **Data:** Decision I resolved, the official DEAP release obtained, and its integrity verified against the trusted 12 files (as done for the Kaggle copy).
2. **Motivation:** a specific, pre-declared hypothesis that the current evidence motivates. For example, a model that improves **ranking** (ROC-AUC) within a person, or an adequately powered test of individual-specific information.
3. **Design written first:** folds, seeds (42–46), metrics, tests and multiple-comparison handling, following Protocol v1.1.
4. **Feasibility:** a runtime and feasibility check, with a cost proportionate to the expected information gain.

**Before claiming one model is better than the Random Forest:** all seven criteria of Protocol §10 on development cross-validation:
- the same protocol;
- a paired participant-bootstrap CI for the balanced-accuracy difference entirely above 0;
- macro-F1 non-inferiority within 0.02;
- a balanced-accuracy CI lower bound above 0.5;
- no increase in person-level collapse;
- seed stability;
- disclosed selection effort.

Then **one** evaluation on the frozen test group, reported as is.

**Before any product-performance claim:**
- the claim must name the setting (zero-shot, unsupervised calibration or supervised calibration) and the metric (trial balanced accuracy, with CIs);
- it must rest on the frozen test group, evaluated once;
- for supervised calibration, it must state the number of rated calibration videos required;
- **any claim of individual-specific emotion estimation** additionally requires a significant, pre-declared disagreement-trial test (or an equivalent control) on adequately powered data;
- no claim may describe the output as a person's exact emotional experience, a diagnosis or a clinical measure.

## 13. Sources (all preserved, unchanged)

| Topic | Report | Saved results |
|---|---|---|
| Protocol, decisions, frozen group | `docs/EVALUATION_PROTOCOL.md` (v1.1) | — |
| Baselines | `docs/REVIEW2_REPORT.md`, `docs/REVIEW3_PROGRESS_30.md` | `docs/results/review3/model_comparison_20261006-050615.json` |
| Robust CNN | `docs/REVIEW3_CNN_ROBUST_EVALUATION.md` | `docs/results/review3/robust_cnn/` |
| Within-person diagnostic and stimulus control | `docs/EXPLORATORY_WITHIN_PERSON_DIAGNOSTIC.md` | `docs/results/exploratory/within_person_rf/`, `stimulus_control_rf/` |
| Cross-participant leave-video-out | `docs/EXPLORATORY_CROSS_PARTICIPANT_LEAVE_VIDEO_OUT.md` | `docs/results/exploratory/cross_participant_lvo_rf/` |
| Unsupervised normalization | `docs/EXPLORATORY_PARTICIPANT_NORMALIZATION.md` | `docs/results/exploratory/participant_normalization_rf/` |
| Supervised calibration | `docs/EXPLORATORY_WITHIN_PERSON_CALIBRATION.md` | `docs/results/exploratory/within_person_calibration_rf/` |
| Model-improvement experiments (added 2026-10-22) | `docs/MODEL_IMPROVEMENT_EXPERIMENTS.md` | `docs/results/exploratory/model_improvement/` (script `backend/scripts/model_improvement.py`) |
| Covariance alignment (added 2026-10-23) | `docs/COVARIANCE_ALIGNMENT_EXPERIMENT.md` | `docs/results/exploratory/covariance_alignment/` (script `backend/scripts/covariance_alignment.py`) |

**Inconsistencies identified while compiling (not reconciled, only explained):**
1. **XGBoost trial macro-F1:** 0.515 vs 0.514 between two documents; this is rounding of the saved 0.5145.
2. **Random Forest leave-one-participant-out accuracy:** "0.510 ± 0.086" vs "± 0.091" uses population vs sample SD on the same folds. "0.503" is the trial-level accuracy, not the segment-level 0.510.
3. **Within-person balanced accuracy:** 0.625 / 0.619 / 0.618 come from three different fold schemes (§9).
4. **Kaggle integrity check:** the aggregate Decision K outcome is now recorded in §2.1. Per-participant results for the frozen test group are deliberately not listed.
