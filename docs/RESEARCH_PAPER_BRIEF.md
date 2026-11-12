# Research-Paper Brief: NeuroSense AI

**Purpose of this document.** It gives a writing assistant everything needed to draft a research paper on this project: the question, data, methods, every result with its exact number, the limits on what may be claimed, and a suggested structure.

**Every number below comes from saved result files in the project repository** (paths in §12).

**Instructions for the writer:**
- Do not invent, round up, or "improve" any number.
- Do not add results that are not listed here.
- If something is missing, write `[TODO: …]` instead of guessing.
- Verify every literature reference before citing it.

---

## 1. One-paragraph summary (abstract material)

**Question.** We studied whether binary **self-reported valence** can be estimated from EEG with a classical machine-learning pipeline. Valence is a person's own 1–9 rating of how pleasant a one-minute music video felt, split at 5.

**Data and evaluation.**
- **Data:** 12 participants of the DEAP dataset (480 trials).
- **Evaluation:** a pre-registered-style protocol. Whole participants, and in the strictest design whole videos, are held out. Eight further participants are frozen as a future independent test group.
- **Three settings:** participant-independent (zero-shot), unsupervised per-person calibration, and supervised personal calibration.

**Results.**

| Setting | Trial balanced accuracy |
|---|---|
| Zero-shot, new person | Near chance in every valid design: **0.493–0.541** |
| Supervised calibration (32 of the person's own rated videos) | **0.618** [0.548, 0.688] |

- **No model change helped the zero-shot result.** Not four model families with nested selection, not aggregation rules, not 14 feature subsets, not valence regression, and not per-person Riemannian covariance alignment.
- **Leakage inflates results.** A deliberately leaky segment-level split reaches **0.805**, which shows how leakage can explain the high accuracies often reported on DEAP.

**Conclusion.** With 12 participants, the limitation is the data (per-person offsets, too few people, noisy labels), not the classifier. Honest evaluation matters more than model choice.

---

## 2. Suggested titles

1. *Honest Evaluation of EEG-Based Valence Estimation: Why Cross-Person Accuracy Stays Near Chance on DEAP*
2. *Leakage, Personalization and the Limits of EEG Valence Decoding: A Protocol-Driven Study on DEAP*
3. *Beyond Inflated Accuracies: A Leakage-Free Evaluation of EEG Emotion Recognition with Personal Calibration*

## 3. Research questions and contributions

**Research questions:**

| | Question |
|---|---|
| RQ1 | How well does a standard EEG pipeline predict a **new person's** binary self-reported valence under leakage-free evaluation? |
| RQ2 | Do model choice, feature subsets, trial aggregation, regression or per-person covariance alignment improve this? |
| RQ3 | How much does **supervised personal calibration** help, and how many rated videos does it need? |
| RQ4 | How much do common evaluation shortcuts (segment-level splits) inflate accuracy? |

**Contributions:**
1. **An explicit, frozen evaluation protocol** for EEG valence decoding:
   - participant- and video-level hold-out;
   - trial-level balanced accuracy as the primary metric;
   - per-participant confidence intervals and paired tests;
   - a frozen test group selected before inspection.
2. **A systematic negative result.** Zero-shot valence decoding stays at chance across model families, features, aggregation, regression and covariance alignment.
3. **Evidence that personal calibration works modestly** (~0.62), and that its signal appears **shared across people and stimulus-related**, not individual-specific.
4. **A direct measurement of leakage inflation** on the same data (0.805 vs ~0.5).
5. **Reproducible, open scripts,** with saved per-condition result files.

## 4. Data

| Item | Value |
|---|---|
| Dataset | DEAP, `data_preprocessed_python` release (Koelstra et al., 2012) |
| Participants used | 12: s01, s02, s03, s05, s07, s08, s09, s11, s15, s16, s22, s25 (the locally available, trusted files) |
| Per participant | 40 one-minute music-video trials, 32 EEG channels at 128 Hz, plus a 3 s pre-trial baseline. 8 peripheral (body) channels exist but were **not used**. |
| Label | `high_valence` if self-reported valence > 5.0 (1–9 scale), else `low_valence`. One label per trial. |
| Class balance | Varies by person, from 37.5% to 70% high-valence. This is why **balanced** accuracy is the primary metric. |
| Borderline trials | 90 of 480 trials (19%) have valence in [4.5, 5.5] |
| Segments | 4 s non-overlapping segments: **7,046 kept, 154 rejected** |
| Frozen test group | s12, s17, s18, s20, s21, s23, s29, s30, drawn with seed 20261010 **before** any of their data was inspected. **Never used**: evaluation is blocked because the only available copy of these participants is an unauthorised third-party upload with altered labels, and official data access is pending. |
| Trial-to-video alignment | Trial *t* is the same video for every participant (verified from the ratings; mean pairwise Spearman ρ 0.443 vs ≤ 0.060 under shuffling) |

## 5. Methods

### 5.1 Preprocessing (per trial; no information crosses trials or participants)
1. Remove the 3 s pre-stimulus baseline.
2. 4–45 Hz zero-phase Butterworth band-pass, order 4.
3. Per-channel robust clipping at median ± 6 × MAD-sigma.
4. Non-overlapping 4 s segments. Reject a segment if more than 5% of its samples were clipped.

### 5.2 Features (440 per segment)
- **Per channel (32 × 12 = 384):**
  - log band power and relative band power in θ (4–8), α (8–13), β (13–30) and γ (30–45 Hz);
  - spectral entropy;
  - Hjorth mobility and complexity;
  - log variance.
- **Hemispheric asymmetry:** 14 electrode pairs × 4 bands = **56**.
- **Covariance experiment only:** 4-band 32×32 covariances (shrinkage 0.05) mapped to tangent space, giving **2,112 features**.

### 5.3 Models
| Model | Configuration |
|---|---|
| Random Forest (reference; the deployed model) | StandardScaler + 300 trees, min_samples_leaf = 5, max_features = √, balanced class weights |
| Others compared | XGBoost (300 trees, depth 4, lr 0.05); logistic regression (C ∈ {0.001, 0.01, 0.1}); RBF SVM (C ∈ {0.1, 1}); an EEGNet-style CNN (earlier study) |
| Trial prediction | Mean of segment probabilities, threshold 0.5. Median, majority vote and mean log-odds were also tested. |

### 5.4 Evaluation protocol (frozen before the experiments)
- **Unit of analysis:** the **participant**. Report per-participant trial-level metrics, then the mean with a t-based 95% CI over the 12.
- **Primary metric:** trial **balanced accuracy**. Secondary: macro-F1, accuracy, within-person ROC-AUC.
- **Mandatory diagnostics:**
  - predicted-class share;
  - the number of "single-label" participants (more than 95% of their trials given one label);
  - within-person AUC.
- **Leakage controls, asserted in code:**
  - whole participants held out;
  - videos held out where stated;
  - all scaling, selection and normalization fitted inside training folds only;
  - within-person splits **by video, never by segment**.
- **Statistics:**
  - paired per-participant differences with a participant bootstrap (10,000 resamples) 95% CI, plus Wilcoxon signed-rank and exact sign-flip tests;
  - Holm correction across pre-declared comparisons;
  - seeds 42–46 (exploratory variants 42–44), always averaged within participant first.
- **Labelling:** all results are "exploratory (12 participants)". No independent-test claim is made.

### 5.5 Evaluation settings (never mix these in one headline)
| Setting | What is held out | What the model may use from the new person |
|---|---|---|
| **Zero-shot, participant-independent (LOPO)** | The whole person | Nothing |
| **Zero-shot, person + video held out (LVO)** | The person **and** 8 of 40 videos | Nothing |
| **Unsupervised calibration** | As above | The person's **unlabelled** EEG on non-test videos (per-person z-scoring or covariance re-centring) |
| **Supervised personal calibration** | 8 of the person's videos | 8–32 of the person's **own rated** videos |

## 6. Results

### 6.1 Historical baseline (pre-protocol; context only)
| Item | Value |
|---|---|
| Split | Fixed: train 7 participants, validation 2 (s16, s25), test 3 (s01, s09, s11). Single seed. |
| Random Forest on the 3 test participants | **0.592** trial accuracy, 0.590 balanced accuracy, macro-F1 0.590, AUC 0.565 (120 trials); segment accuracy 0.548 |
| Per participant (trial balanced accuracy) | s01 0.500 (one label for every trial); s09 0.750; s11 0.604 |
| Participant-level mean | 0.618, 95% CI **[0.306, 0.930]** (n = 3) |
| Other models on the same split (trial balanced accuracy) | SVM 0.459, XGBoost 0.515, CNN 0.533, majority 0.500 |
| Pre-protocol LOPO (9 participants) | Segment accuracy 0.510 ± 0.086 |
| Robust CNN (5 seeds) | Test segment macro-F1 0.466 ± 0.109; LOPO balanced accuracy 0.494 ± 0.023 (not different from 0.5); predictions collapse to one label per person |

**Important:** this 59.2% is **not comparable** with the later results (§6.6).

### 6.2 Zero-shot (participant-independent) results
| Design | Model / variant | Trial balanced accuracy [95% CI] |
|---|---|---|
| LOPO | Majority-class reference | 0.500 |
| LOPO | Random Forest (seeds 42–44) | **0.533** [0.490, 0.576]; within-person AUC 0.606; ~5.7 of 12 participants single-label |
| LOPO | RF, 5 seeds | 0.528 [0.489, 0.568] |
| LOPO | XGBoost, nested | 0.521 [0.488, 0.555] |
| LOPO | Logistic regression, nested | 0.538 [0.498, 0.579] |
| LOPO | SVM, nested | 0.539 [0.501, 0.577] |
| LOPO | **Nested selection over all 28 model × aggregation options** | **0.541** [0.504, 0.579]; vs RF +0.008 [−0.024, +0.039], Holm p = 0.93 |
| LVO (person + videos held out), earlier study | Random Forest | 0.500 [0.481, 0.520] |
| LVO, covariance study | Random Forest (A0) | **0.493** [0.470, 0.516] |

**Other zero-shot findings:**
- **Trial aggregation (mean, median, vote, log-odds):** 0.526–0.533; no effect.
- **Feature ablation (14 subsets: bands, families, regions, without artifact-prone sites):** none better than the full set. Parietal-only was worse (0.485); β-only 0.543 (n.s.).
- **Continuous valence regression, then threshold at 5:** 0.499, worse than classification.
- **Training-set size (3 / 6 / 9 / 11 people):** 0.516 / 0.519 / 0.524 / 0.533. Slightly rising; too few people to know the trend.

### 6.3 Unsupervised per-person calibration
| Design | Variant | Trial balanced accuracy [95% CI] |
|---|---|---|
| LOPO | RF + per-person z-score | 0.575 [0.525, 0.624]; collapse removed (0 single-label people). The test videos' stimuli are in training (other people's recordings). |
| LOPO | Nested selection + z-score | 0.543; **worse** than plain RF (−0.031 [−0.051, −0.009]): model selection with 11 people is unreliable |
| LVO | RF + per-person z-score | **0.535** [0.492, 0.578]; replicated exactly in two separate experiments |
| LVO | **Per-person Riemannian covariance alignment (A3)** | **0.499** [0.458, 0.539] |
| LVO | Covariance, no alignment (A2) | 0.506 [0.471, 0.541] |
| LVO | Aligned covariance + logistic regression (A4) | 0.509 [0.444, 0.573] |

**Covariance alignment (pre-declared comparisons, Holm over 4):**

| Comparison | Difference [95% CI] | Note |
|---|---|---|
| A3 − A0 (primary) | **+0.006** [−0.025, +0.038] | Holm p = 1.0; 8 better / 4 worse |
| A3 − z-score | −0.036 [−0.070, −0.003] | |
| A3 − A2 (alignment itself) | −0.007 | |
| Macro-F1, A3 − A0 | +0.047 [+0.002, +0.090] | Mechanical: collapse removed |
| Within-person AUC, A3 − A0 | +0.005 | No change |

- **Mechanistic finding:** before alignment, the between-person spread of mean covariance features (18.1) was more than twice the within-person spread (7.7). Alignment reduced it to 0.6 while keeping within-person variation (8.0), yet discrimination did not improve.
- **Interpretation:** person identity dominates the representation. After removing it, what remains does not transfer valence information across people.

### 6.4 Supervised personal calibration (within-person, split by video)
| Variant | Trial balanced accuracy [95% CI] |
|---|---|
| Random Forest, 32 own rated videos (5 seeds) | **0.618** [0.550, 0.687]; replicated across three fold schemes (0.625, 0.619, 0.618) |
| Nested selection over 28 options | 0.620 [0.559, 0.681]; vs RF +0.002, n.s. |
| Logistic regression / SVM (nested) | 0.634 / 0.635 (n.s. vs RF) |
| XGBoost (nested) | 0.588 (worse, −0.030) |
| Aligned covariance features (W1) | 0.607 [0.532, 0.681] (vs 0.618, n.s.) |
| Borderline trials (4.5–5.5) excluded from training and scoring | 0.635; an easier, different task |
| Valence regression, then threshold | 0.564 (worse) |

**Number of rated calibration videos:**

| Calibration videos | 8 | 16 | 24 | 32 |
|---|---|---|---|---|
| Personal model only | 0.562 | 0.601 | 0.603 | 0.618 |
| Pooled (others + own, equal weight) | 0.576 | 0.594 | — | 0.601 |

The zero-shot reference on the same videos is 0.535.

**Pooling other people's data helps slightly at 8 videos and hurts at 32** (−0.017 [−0.027, −0.006]).

**Who benefits:**
- **Helped by calibration:** 7–10 of 12 participants, depending on the number of videos.
- **Never reliably helped:** s07, s11, s25.

**Earlier statistical tests:**
- **Above chance:** within-person 0.618, primary exact p = 0.004; 4 of 12 participants individually significant after BH correction.
- **Against zero-shot on the same trials:** +0.118 [0.048, 0.187], Holm p = 0.013.

**Stimulus-consensus control:**
- **Consensus as well as own rating:** the calibrated model predicts the **other 11 participants' consensus** label for each video (0.626) as well as the person's own rating (0.618).
- **Disagreement trials:** on trials where the person disagrees with the consensus, it follows the person only 60 of 117 times (n.s.).
- **Reading:** the signal appears **shared and stimulus-related**, not individual-specific.

### 6.5 Diagnostics of the performance ceiling
| Factor | Evidence | Verdict |
|---|---|---|
| Per-person offsets | ~5/12 single-label people under zero-shot; within-person AUC 0.59–0.62; between-person covariance spread larger than within-person | **Major** |
| Weak generalizable signal | Every zero-shot design at 0.49–0.54 | **Major** |
| Small dataset | Flat-ish learning curve; nested selection unreliable with 11 people | Contributes |
| Noisy labels near 5 | Accuracy 0.589 on borderline vs 0.646 on clear trials; r = 0.074, p = 0.005 | Modest |
| Eye/muscle artifacts | Removing γ or artifact-prone sites changes little; γ alone matches all features within person | No evidence it drives results; **not excluded** |
| Model choice, aggregation, features | All within ±0.02–0.03 | Not the bottleneck |

### 6.6 Leakage demonstration and comparability
| Setting | Valid? | Trial balanced accuracy |
|---|---|---|
| Random 5-fold split of **segments** within a person | **No** (neighbouring 4 s pieces of one recording in both train and test) | **0.805** [0.776, 0.834]; segment accuracy 0.731 |
| The same task split **by video** | Yes | 0.618 |

**Why the headline numbers are not interchangeable:**

| Number | Metric | Who is tested | Training data |
|---|---|---|---|
| 59.2% | Pooled trial **accuracy** | 3 fixed people | 9 people; test videos' stimuli in training |
| 53.3% | Mean per-person **balanced** accuracy | All 12, LOPO | Test videos' stimuli in training |
| 49.3% | Mean per-person **balanced** accuracy | All 12, LVO | Neither the person nor the video in training |

**These must not be presented as a model getting worse or better.**

### 6.7 Explainability (active Random Forest)
- **Method:** global impurity-based feature importance only; no per-prediction explanations.
- **Finding:** gamma-band features carry about **31%** of total importance.
- **Caution:** gamma is prone to muscle (EMG) contamination, so this is reported as a model property, not evidence of a neural valence signal.

## 7. Claims: allowed and not allowed

**Supported (state with the CIs above):**
- **Zero-shot valence decoding** on these 12 participants is at chance under leakage-free evaluation.
- **Model choice, aggregation, feature subsets, regression and covariance alignment** did not change this.
- **Supervised personal calibration** gives modest above-chance accuracy (~0.60–0.62), with a signal that appears shared and stimulus-related.
- **Segment-level splitting** inflates accuracy dramatically (0.805 vs ~0.5–0.62).

**Not supported (must not be claimed):**
- 90% or any high accuracy;
- generalization to new or uncalibrated people;
- detection of a person's actual or exact emotion;
- individual-specific emotion estimation;
- any clinical, diagnostic or product performance;
- any claim about people outside the 12 participants;
- any independent-test result (none exists).

**Required framing:**
- "Exploratory, 12 participants"; "self-reported valence", not "emotion".
- Personalization is described as exploratory.

## 8. Limitations (for the paper's Limitations section)

- **Small sample, no independent test:**
  - 12 participants; CI half-widths are about ±0.04 (zero-shot) and ±0.07 (personalized);
  - no independent test (the frozen group is blocked on data access).
- **Labels:**
  - one self-report per 60 s trial;
  - binary threshold at 5;
  - 19% borderline trials.
- **Same stimuli for everyone:** all participants watched the same 40 videos, so a stimulus-specific effect cannot be fully excluded.
- **Fixed features and models:**
  - handcrafted features only (plus one covariance representation);
  - deep models were evaluated only in the earlier CNN study;
  - no pretrained EEG foundation models.
- **Artifacts:** possible EMG contribution in the gamma band was not fully ruled out (no ICA).
- **Preprocessing:** the 4 Hz high-pass removes delta-band information.
- **Exploratory status:** results are exploratory, and selection effort is disclosed (28 configurations).

## 9. Future work (for the Discussion)

1. **More participants:**
   - **Official DEAP:** obtain the official full release (32 participants) and evaluate once on the frozen 8-person group.
   - **FACED (123 healthy participants, 28 clips, valence self-reports):**
     - a separate dataset with its own frozen test group;
     - realistic expectation ~55–65% for new-person own-rating valence;
     - this is an estimate, **not a result**.
2. **Personal calibration as the application mode,** with fewer required ratings.
3. **Additional modalities:** peripheral physiological signals (skin conductance, heart, respiration) as a pre-declared experiment.
4. **Pretrained EEG models** fine-tuned on larger datasets; artifact-robust preprocessing (ICA); an explicit EMG control.

## 10. Suggested paper structure

1. **Introduction:** EEG emotion recognition; inflated accuracies in the literature; the need for honest evaluation; the RQs and contributions.
2. **Related work:** DEAP and similar datasets; cross-subject EEG emotion recognition; transfer learning and alignment; data leakage in EEG evaluation. [TODO: verified citations]
3. **Data:** §4.
4. **Methods:** §5. A pipeline diagram is recommended.
5. **Evaluation protocol:** §5.4–5.5, emphasising the pre-declared design and the frozen test group.
6. **Results:** in this order:
   1. the zero-shot results (§6.2–6.3);
   2. personalization (§6.4);
   3. diagnostics (§6.5);
   4. the leakage demonstration (§6.6).
7. **Discussion:** why the data, not the model, is the bottleneck; shared vs individual signal; implications for evaluation practice.
8. **Limitations:** §8.
9. **Conclusion and future work:** §9.
10. **Reproducibility statement:** scripts, seeds, protocol version, result files (§12).

**Suggested figures and tables:**
- **Fig. 1:** the pipeline.
- **Fig. 2:** the evaluation settings (zero-shot LOPO vs LVO vs calibration).
- **Table 1:** zero-shot results.
- **Table 2:** calibration results and the calibration-size curve.
- **Fig. 3:** per-participant balanced accuracy across settings, with chance at 0.5. The data is in `tables.md`.
- **Table 3:** the leakage demonstration and the comparability table.
- **Existing image files:** `docs/results/exploratory/*/*.png`.

## 11. Ethics and data statement (draft text)

- **Data source:** DEAP is used under its end-user licence; no data is redistributed.
- **Participant protection:**
  - all analysis used only the 12 locally available participants;
  - the 8 frozen participants were never inspected;
  - an unauthorised third-party copy of the remaining participants was **not used**.
- **No new participants:** no new human data was collected.
- **Framing of outputs:** the system estimates **self-reported valence** categories for research. It is not a diagnostic or clinical tool.

## 12. Source files (in the project repository)

| Topic | Document | Saved results |
|---|---|---|
| Evaluation protocol | `docs/EVALUATION_PROTOCOL.md` | — |
| Consolidated findings | `docs/RESEARCH_FINDINGS_AND_NEXT_STEPS.md` | — |
| Historical baseline | `docs/REVIEW2_REPORT.md`, `docs/REVIEW3_PROGRESS_30.md` | `docs/results/review3/` |
| Robust CNN | `docs/REVIEW3_CNN_ROBUST_EVALUATION.md` | `docs/results/review3/robust_cnn/` |
| Within-person, stimulus control, LVO, normalization, calibration | `docs/EXPLORATORY_*.md` | `docs/results/exploratory/*/` |
| Model-improvement experiments | `docs/MODEL_IMPROVEMENT_EXPERIMENTS.md` | `docs/results/exploratory/model_improvement/` |
| Covariance alignment | `docs/COVARIANCE_ALIGNMENT_EXPERIMENT.md` | `docs/results/exploratory/covariance_alignment/` |
| Code | `backend/app/` (pipeline); `backend/scripts/` (experiments) | — |

## 13. References to verify before citing

- Koelstra, S., et al. (2012). DEAP: A Database for Emotion Analysis Using Physiological Signals. *IEEE Transactions on Affective Computing*, 3(1).
- Chen, J., et al. (2023). A large finer-grained affective computing EEG dataset (FACED). *Scientific Data*.
- Zanini, P., et al. (2018). Transfer learning: a Riemannian geometry framework with applications to brain–computer interfaces. *IEEE Transactions on Biomedical Engineering*.
- He, H., & Wu, D. (2020). Transfer learning for brain–computer interfaces: a Euclidean space data alignment approach. *IEEE Transactions on Biomedical Engineering*.
- Lawhern, V. J., et al. (2018). EEGNet: a compact convolutional neural network for EEG-based brain–computer interfaces. *Journal of Neural Engineering*.
- [TODO: papers on data leakage and segment-level splitting in EEG emotion recognition. Verify before citing.]
