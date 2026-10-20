# NeuroSense AI: Evaluation Protocol

| | |
|---|---|
| **Version** | 1.1 (2026-10-10). Amends v1.0 by adding one narrowly scoped label-integrity check (§6.4, Decision K). All other rules are unchanged; see §11 for the change log. |
| **Status** | **Approved and frozen** for decisions A–H, J and K (approved 2026-10-10). **Decision I (data access and storage) is pending** (§12). Nothing in §8.1 that depends on Decision I may be carried out until it is confirmed. |
| **Applies to** | Every model developed or compared from now on: Random Forest, SVM, XGBoost, CNN, LSTM/GRU, CNN-LSTM, TCN, and later models. |
| **Task** | Binary self-reported valence from EEG: `high_valence` if valence > 5.0 on DEAP's 1–9 scale, else `low_valence`. |

This protocol fixes **how** models are evaluated before any further experiment is run. That way, the evaluation rules cannot be adjusted after seeing results. It is based on two earlier reports:
- the robust CNN evaluation (`docs/REVIEW3_CNN_ROBUST_EVALUATION.md`);
- the read-only project audit (October 2026).

It promises no particular accuracy. Its purpose is that a reported improvement is real, and a reported failure is reported as one.

Results produced under this protocol must cite its version (for example, "evaluated under Protocol v1.0"). Results produced before it (Review 2 and the 28-run CNN evaluation) remain valid as historical results, but they are **not** protocol results (see §11).

---

## 1. Participant roles

| Role | Who | Use |
|---|---|---|
| **Development participants** | Every available participant **except** the final test group. Until new data arrives, this is all 12 current participants, including s01, s09 and s11 (see below). | All model development, selection, tuning and comparison, through cross-validation (§2). |
| **Final test group** | A fresh group drawn only from newly acquired participants (§8). It does not exist yet. | One final evaluation of models that have already been accepted on development data (§10). |

**Why s01, s09 and s11 stop being a test group.** These three participants have been evaluated by the Random Forest, SVM, XGBoost, the registered CNN and 10 robust-evaluation runs. No model was *selected* on them, but their results are now known and have shaped how the project is understood. They can no longer act as an independent test. Under this protocol they become development participants, and the historical "fixed-split test" results stay labelled as historical.

**Interim period (12 participants only; Decision C).**
- Cross-validation over the 12 current participants is allowed for **exploratory work only**.
- Every interim result is labelled *exploratory (12 participants, Protocol v1.0)*.
- No interim result may be described as an independent or held-out test result.
- Interim results may be used for debugging, investigating the pipeline and exploring model improvements. They **must not** support a formal claim that a new model outperforms the Random Forest.
- Formal model comparisons under the §10 acceptance criteria wait until the full dataset is available and the development protocol (§2) can be followed on the full development set (§8).
- The final 8-participant test group (§8.2) stays untouched until the selected model is ready for final evaluation.

---

## 2. Primary evaluation method

### 2.1 Outer loop: leave-one-participant-out (LOPO) over development participants

1. For each development participant *p*: train on all other development participants, then evaluate once on *p*.
2. Each participant is held out exactly once. Every held-out participant contributes one value per metric.
3. **The participant is the unit of analysis.** Aggregate results are means across held-out participants, not pooled over segments.

**Permitted fallback: grouped K-fold** (K ≥ 8, whole participants per fold).
- Use it only when LOPO is computationally infeasible for a model class, for example deep models on CPU with 24+ development participants.
- The choice must be declared **before** the experiment, and the same folds must be used for every model in that comparison.
- Metrics are still computed per held-out participant.

### 2.2 Inner loop: model selection inside each outer fold

Every choice that depends on data is made **inside** the outer fold, using only that fold's training participants:
- the early-stopping epoch;
- hyperparameters;
- the decision threshold, if any;
- normalization statistics;
- class weights;
- feature selection.

**Inner validation for models with early stopping:**
- Use **3** inner validation participants (instead of 2) when at least 15 participants are available for that fold's training; otherwise use 2.
- Rotate them by a seeded rule that changes across folds, so that no small set of people decides every fold. This addresses audit finding M4.

**After selection, refit** on all of the fold's training participants for the selected setting (for example, the selected number of epochs). This is the same refit rule as now.

### 2.3 Fold definition file

- Folds, inner-validation assignments and seeds are written to one versioned **fold definition file** (planned: `protocol/folds_v1.json`), together with its SHA-256.
- Every experiment records the hash of the fold file it used.
- **Models are comparable only if they used the same fold file.**
- The file is generated once from sorted participant IDs and a recorded seed. It is regenerated only when the participant set changes, and that requires a protocol version bump (§11).

---

## 3. Metrics

### 3.1 Primary metrics, per held-out participant

| Metric | Level | Why |
|---|---|---|
| **Balanced accuracy** | **Trial** | Corrects for each participant's own class balance, which ranges from 37.5% to 70% high-valence among the current 12. A model that gives everyone the majority label scores 0.5. |
| **Macro-F1** | **Trial** | Penalizes collapsing onto one class. Comparable with earlier reports. |

**Trial-level prediction rule:**
- Average the segment probabilities within the trial, then take the argmax (equivalent to a 0.5 threshold).
- No threshold may be tuned on held-out participants.

**Aggregate:** the mean ± SD across held-out participants, with confidence intervals as in §5.

### 3.2 Secondary metrics

These are always reported, but they never decide acceptance on their own:
- trial-level accuracy and ROC-AUC;
- segment-level accuracy, balanced accuracy, macro-F1 and ROC-AUC;
- pooled metrics over all held-out trials;
- the majority-class reference: the most frequent class among that fold's training participants.

### 3.3 Mandatory diagnostics for every model and fold

| Diagnostic | Definition | What it detects |
|---|---|---|
| **Predicted-class share** | Fraction of a participant's trials (and segments) predicted `high_valence`. | Collapsing onto one label for a whole person. |
| **Single-label participants** | Number of held-out participants whose predicted-class share is < 0.05 or > 0.95. | The pattern found in the robust CNN evaluation. |
| **Within-person AUC** | Trial-level ROC-AUC computed inside each held-out participant. | Whether the model ranks one person's trials, independent of an overall offset. |
| **Within/between variance** | The SD of p(high) within each participant, compared with the SD of per-participant mean p(high). | Whether the output is dominated by a per-person offset. |
| **Base-rate tracking** | Spearman ρ, across training participants, between each person's mean predicted p(high) and their true high-valence share. | Whether the model learns "what kind of person this is" instead of trial content. The registered CNN showed ρ = 0.83. |

### 3.4 Participants with too few trials of one class

**Rule:** if a held-out participant has fewer than **5** trials of either class:
- their balanced accuracy and within-person AUC are reported but flagged;
- the aggregate is reported both with and without them.

All current participants have at least 12 trials of each class.

---

## 4. Fair comparison between models

Two models are compared only if all of the following are identical, and the experiment record checks them:

1. **Data:** the same participant set, dataset fingerprint, preprocessing configuration and segment table (the same rows: the feature table and segment table already match row for row).
2. **Labels:** the same rule (`valence > 5.0`) and the same treatment of borderline trials (§7).
3. **Folds:** the same fold definition file, including inner-validation assignments.
4. **Seeds:** the same seed set for **every** stochastic model, including the Random Forest. Primary model comparisons use **42, 43, 44, 45, 46** (Decision D). Exploratory variants in §9 may use fewer, at least 3, taken from the start of this list.
   - Every seed is reported. There is no best-seed reporting.
   - Results are first averaged over seeds within each held-out participant, and that seed-averaged value is used for comparison. Seeds are not independent observations and are never counted as extra samples.
5. **Metrics and aggregation:** the code path in this document, with the same trial aggregation and the same 0.5 threshold.
6. **Selection effort is disclosed.** Each model reports:
   - its search space;
   - the number of configurations tried;
   - whether any setting was fixed in advance.

   Tuning is only ever inner-fold. A model tuned over 50 configurations is not compared silently against a model run with defaults.
7. **References always included:** the majority-class reference and the Random Forest baseline, re-run under the same folds and seeds.

Items 4 and 7 close audit finding M3: the Random Forest was previously run with one seed and had no balanced accuracy in its LOPO.

---

## 5. Uncertainty: confidence intervals and permutation tests

### 5.1 Across participants (primary uncertainty)

- **Per model:** mean ± SD over held-out participants, with a t-based 95% CI (n = number of development participants).
- **Between two models:**
  1. Compute paired per-participant differences of the seed-averaged primary metric.
  2. Report a **participant-level bootstrap 95% CI of the mean difference** (10,000 resamples of participants).
  3. Report a **Wilcoxon signed-rank test** on the paired differences.
- **Interpretation:** these intervals describe how results vary across people. They are the relevant uncertainty for the claim "works on new people".
  - With 12 participants they are wide. With about 24 development participants they become usable, but are still modest.

### 5.2 Within one participant

- **Within-participant permutation test** for each held-out participant:
  - Permute that participant's trial labels 2,000 times, keeping predictions fixed.
  - The test statistics are trial-level balanced accuracy and within-person AUC.
  - The p-value is one-sided: (1 + count of permutations ≥ observed) / 2,001.
- **Group-level summary:**
  - the number of participants with p < 0.05 after **Benjamini–Hochberg** correction across participants;
  - a sign-flip test across participants of mean within-person AUC − 0.5.
- **Interpretation:**
  - This tests whether the model separates **trials of the same person** beyond what that person's label distribution allows.
  - It does **not** test cross-person generalization; §5.1 does that.
  - A model that gives each person one label scores p ≈ 1 here, however good its pooled accuracy looks.
- **Trial-cluster bootstrap CIs:** resample trials within each participant. These may be reported for a single held-out group (for example, the final test group in §10), and should be described as uncertainty over trials of those people only.

### 5.3 Multiple comparisons and pre-specification

- **One primary comparison per study,** declared before running. Example: "new model vs Random Forest, trial balanced accuracy".
- **Everything else is exploratory.** Secondary comparisons are labelled exploratory and are not used to claim improvement.
- **Every variant tried is reported,** including failed ones (§9).

---

## 6. Preventing leakage and contamination

### 6.1 Participant leakage

- **Whole participants only.** A participant's segments and trials are never split between training and evaluation in a cross-subject analysis.
- **Checked automatically.** Each experiment asserts that its train, inner-validation and evaluation participant sets are disjoint, and records them. The existing `RunSpec` check does this already, and it becomes mandatory for every model.
- **Within-person diagnostics (§9, D2)** split by **trial**, never by segment. Neighbouring 4 s segments of one trial are almost duplicates.

### 6.2 Preprocessing leakage

| Operation | Allowed scope |
|---|---|
| Baseline removal, band-pass filtering, artifact clipping (statistics from the **same trial only**), segmentation | Per trial. No information crosses trials or participants. This is the current behaviour. |
| Feature scaling, channel normalization, PCA or feature selection, class weights, decision thresholds, any learned transform | Fitted on the fold's **training participants only**, then applied unchanged to inner-validation and held-out participants. |
| Unsupervised per-participant normalization or alignment (for example, per-person z-scoring or covariance re-centring) | Allowed only as a **declared calibration variant**. It must use **no labels**, it assumes an unlabelled recording of each new person is available, and it is reported separately from fully inductive models. |
| Anything using held-out participants' **labels** | Never, outside final metric computation. |

Caches are keyed by the dataset fingerprint and the preprocessing configuration. A changed preprocessing configuration therefore can never silently reuse old features.

### 6.3 Test-set contamination

1. **Lock first.** The final test group is chosen by a recorded random procedure (§8) **before** any of its EEG or ratings are inspected.
2. **Integrity checks only.** Before the final evaluation, the test files may only be checked for file count, SHA-256, array shapes and finite values, plus the label-integrity check defined in §6.4 (added in v1.1). No rating summaries, class balances, spectra or plots.
3. **Code guard (planned).** Development scripts will refuse to load test participants. Loading them will require an explicit final-evaluation command that writes a timestamped, append-only log entry.
4. **Final evaluation is run once per accepted model and reported as is.** If a model is changed afterwards in response to its test result, the test group counts as contaminated. That must be documented, and later claims revert to development-only evidence.
5. **No decisions from historical test results.** Results already seen on s01, s09 and s11 (for example, "per-participant normalization was worse on that split") must not be used to choose future designs. Only development cross-validation evidence counts.

### 6.4 Label-integrity check for the 20 additional participants (v1.1, Decision K)

**Background.**
- The only available copy of the 20 additional participants is a third-party (Kaggle) upload.
- Its `.dat` files hold EEG identical to our 12 trusted files, but on about 30% of trials both valence and arousal were replaced by 9 − x.
- The same upload contains `participant_ratings.xls`. For the 12 trusted participants, it matches our ratings exactly on all four ratings (1,920 of 1,920 values), using the mapping `.dat` trial *t* ↔ `Experiment_id` *t* + 1.
- That comparison did not read any of the 20 additional participants' ratings.

**Scope.** One automated check, run once, on all 20 additional participants, including the 8 frozen test participants. For each participant it verifies:

1. **Structure:** the workbook has exactly 40 rows for the participant, and `Experiment_id` is a permutation of 1–40.
2. **Dominance and liking:** the workbook values equal the `.dat` values for all 40 trials (mapping by `Experiment_id`; tolerance 1e-9).
3. **Valence/arousal pairs:** in each trial, the `.dat` valence/arousal pair equals either the workbook pair, or 9 − the workbook pair, with **the same choice for both ratings**.
4. **Range:** all four workbook ratings lie within 1–9.

**Output allowed:** for each participant, its role (development or test), PASS/FAIL, and counts of failures for checks 1–4.

**Output not allowed:**
- rating values;
- the number or positions of reversed trials (reversal patterns);
- class balances;
- EEG signal statistics;
- any model-performance information.

**Implementation constraints:**
- read-only;
- the workbook is opened read-only with macros disabled;
- values are passed in memory and never written to disk or displayed;
- nothing is constructed, integrated or changed.

**Interpretation.**
- **A PASS is consistency only.** It means the workbook and the `.dat` file agree with each other under the known alteration.
- **It does not prove authenticity or authorization.** A consistent edit to both files would still pass, and the check gives no permission to use the third-party copy (Decision I remains pending).
- **Not done here:** building any reconstructed dataset needs separate approval.

---

## 7. Label handling and borderline ratings

- **Primary analysis (unchanged):** `high_valence` if valence > 5.0, else `low_valence`. A rating of exactly 5.0 counts as low. Every segment inherits its trial's label.
- **Size of the issue:** among the current 12 participants, **90 of 480 trials (19%)** have valence in [4.5, 5.5]. These labels are likely the noisiest.
- **Secondary analysis S1, borderline excluded:** reported separately, and it never replaces the primary result.
  - **Exclusion rule:** trials with 4.5 ≤ valence ≤ 5.5 are removed from **both** training and evaluation in every fold. The band is fixed now, before any S1 result is seen.
  - **Report:** the number of trials removed per participant, and the resulting per-participant class balance.
  - **Same everything else:** the same folds, seeds, metrics and acceptance reporting as the primary analysis.
  - **Interpretation:** S1 asks whether the model separates clearly positive from clearly negative ratings. It is an easier and different question. An S1 improvement is not an improvement on the primary task.
- **Not part of this protocol:**
  - per-participant thresholds (for example, median splits);
  - continuous regression;
  - arousal or quadrant targets.

  Each of these is a different task, and if adopted it needs its own protocol section.

---

## 8. Data acquisition and the final test group

### 8.1 Acquiring the missing 20 DEAP participants

**Missing:** s04, s06, s10, s12, s13, s14, s17–s21, s23, s24 and s26–s32.

> **Decision I is pending.** The team has not yet decided:
> - who signs the DEAP agreement;
> - who downloads the data;
> - where the files are stored;
> - whether the duplicate local copy is removed.
>
> Steps 1–3 below describe what must happen, not who does it. Nobody signs, downloads, moves or deletes dataset files until Decision I is confirmed. The current configuration and both existing copies of the 12 files stay as they are.

1. **Official channel only.** Request access through the official DEAP dataset page (Queen Mary University of London). DEAP is released under an end-user licence agreement (EULA), which must be completed and returned as instructed on that page. *Who signs: pending Decision I.*
2. **Download** the `data_preprocessed_python` release (files `s01.dat`–`s32.dat`), the same release the current 12 files come from. *Who downloads: pending Decision I.*
3. **Storage.** *Pending Decision I.* Requirements for whatever location is chosen:
   - outside git (`Dataset/` is already git-ignored);
   - never redistributed or committed.

   Options to decide as part of Decision I:
   - whether to avoid a synced cloud folder;
   - where `NEUROSENSE_DATASET_DIR` should point;
   - whether to remove the duplicate copy so the code and documentation refer to one place (audit finding M7).
4. **Verify,** without looking at test-group content (§6.3):
   - all 32 files are present;
   - SHA-256 of every file is recorded in a dataset manifest;
   - the 12 existing files are byte-identical to the current copies (confirms the same release);
   - shape (40, 40, 8064) and finite values for every file.
5. **Lock the final test group (§8.2) before the next step.**
6. **Development participants only:** run the full structure, rating and class-balance audit.

### 8.2 Reserving a fresh final test group (approved, Decision G)

**Drawn only from the 20 newly acquired participants.** They have never been seen. The 12 current participants remain development participants.

**Approved size (Decision G): 8 participants** (25% of 32), which leaves **24 development participants** (the current 12 plus 12 new).

| Test size | Development size | Approx. 95% CI half-width of mean trial accuracy* | Trade-off |
|---|---|---|---|
| 6 | 26 | ±0.09 | More development data, but a wide, fragile test estimate. One unusual person moves it a lot. |
| **8** | **24** | **±0.075** | **Balanced.** Doubles development data compared with now, and gives a test estimate about 1.5× tighter than 3 people. |
| 10 | 22 | ±0.065 | Somewhat tighter test estimate, but less data for development and selection. |

\* Indicative t-interval assuming the between-person SD of about 0.09 seen in the current LOPO results. The real value is unknown until measured.

**Selection procedure** (approved; not yet run, so no seed is chosen and no IDs are selected):
1. Fix a selection seed, and record it in this document **before** any new participant's data is inspected. Inspection includes ratings, EEG, spectra, plots and class balance.
2. Sort the IDs of the newly acquired participants, permute them with `numpy.random.default_rng(seed)`, and take the first 8.
3. **No stratification by class balance or rating statistics.** Stratifying would require looking at the test participants' labels.
4. **Freeze the seed and the 8 IDs** in this document (date, seed, IDs, procedure) **before** any data is inspected. After freezing, the group cannot be redrawn or swapped. Only the integrity checks in §6.3 are allowed on its files.
5. Only then may the remaining new participants be audited and added to the development set.

**If fewer than 20 participants can be obtained:** reserve about 25% of the **new** participants, with a minimum of 5. Below 5 new participants, keep cross-validation as the only evidence and make no held-out test claim.

---

## 9. Development-only investigation: participant-specific bias and representations

Everything in this section uses **development participants and the frozen folds only**. The final test group is not touched.

**D1. Bias diagnostics as standard.**
- Report all §3.3 diagnostics for the Random Forest and the current CNN under Protocol v1.0. This sets the reference levels for single-label participants, within-person AUC and base-rate tracking.

**D2. Within-person diagnostic.**
- For each participant separately: grouped 5-fold cross-validation over that person's **40 trials**, split by trial.
- Purpose: separate "the pipeline carries no valence information" from "the information does not transfer between people".
- Labelled as a within-subject diagnostic. It is never reported as, or compared with, cross-subject performance.

**D3. Normalization variants.** Each is declared as either **inductive** (needs nothing from the new person) or **calibration** (needs that person's unlabelled EEG):

| Variant | Type |
|---|---|
| N0: current method, channel z-score from training participants | inductive |
| N1: per-segment channel z-score | inductive |
| N2: per-participant channel z-score (already tried on the old split; re-evaluated here under LOPO) | calibration |
| N3: per-participant feature standardization for the Random Forest (never tested) | calibration |
| N4: per-participant covariance re-centring ("Euclidean alignment") | calibration |

**D4. Representation variants** (candidates; the final list is declared before running):
- band-power or differential-entropy maps per channel for the CNN;
- covariance or tangent-space features for linear and tree models.

**D5. Procedure for D3 and D4:**
- Each variant runs on the frozen folds, with **at least 3 seeds** (5 for a finalist).
- Reported with the primary metrics **and** the §3.3 diagnostics.
- A variant is preferred only by development cross-validation under §10's criteria.
- Every variant tried is listed in the results, including failures.

**D6. Rating-style bias.**
- Report each development participant's high-valence share and mean rating next to the model's per-person mean prediction.
- The goal is lower base-rate tracking (§3.3). Matching base rates is not a goal.

D2–D4 are investigations. None of them changes the served model unless it passes §10.

---

## 10. Criteria for accepting a new model or architecture

A new model (call it **M**) is accepted as an improvement over the current reference (**R**, initially the Random Forest re-run under this protocol) only if **all** of the following hold on **development cross-validation**:

1. **Same protocol.** Same fold file, seeds, data fingerprint, label rule and metrics as R. All folds and all seeds are reported.
2. **Primary metric better.** The paired per-participant difference in seed-averaged **trial balanced accuracy** (M − R) has a participant-bootstrap **95% CI entirely above 0**. The Wilcoxon p-value is reported alongside.
3. **No loss on macro-F1.** The lower bound of the 95% CI of the paired difference in trial macro-F1 is above **−0.02** (non-inferiority margin).
4. **Above chance.** M's mean trial balanced accuracy has a 95% CI lower bound **above 0.5**, and M beats the majority-class reference on trial balanced accuracy.
5. **No increase in person-level collapse:**
   - the number of single-label participants is not greater than R's;
   - base-rate tracking ρ is not higher than R's.

   Both are reported with values.
6. **Seed stability.** The improvement holds for the seed-averaged result. The across-seed SD of M's mean trial balanced accuracy is reported, and the decision does not rest on any single seed.
7. **Selection effort disclosed** (§4.6).

**Then, final evaluation:**
- **Run once.** An accepted model is evaluated once on the locked final test group, and the result is reported as is, with trial-cluster CIs and within-participant tests.
- **Test results don't re-decide acceptance.** If the test result disagrees with development, that disagreement is the finding and is reported. The model is not tuned further against the test group.
- **Production is a separate decision.** A model may be activated only if it has passed §10 and its final test result has been reviewed. The application, API, registry and active model are unchanged until then.

**Outcomes (Decision H):**
- **Outperforms the Random Forest:** only if criteria 1–7 are all met.
- **Inconclusive:** if criteria 1–7 are not all met. Report M as *inconclusive* (for example, "no detectable difference from the Random Forest under Protocol v1.0") or as *exploratory*. Show the measured values for every criterion, including the ones that failed.
- **Thresholds are fixed.** The criteria in this section (the 0 bound, the 0.02 macro-F1 margin, the 0.5 chance bound, the collapse and base-rate conditions) must not be changed after any result has been observed. A change requires a new protocol version (§11), applies only to experiments that have not yet been run, and cannot be used to re-label an existing result.

A model that fails these criteria is still a valid, reportable result.

---

## 11. Versioning and changes

- **Frozen.** v1.0 is frozen for decisions A–H and J as of 2026-10-10. Recording Decision I, and later the test-group seed and IDs (§8.2), completes v1.0. These are logged in §12 and do not change any evaluation rule.
- **Change log:**

  | Version | Date | Change | Reason |
  |---|---|---|---|
  | 1.0 | 2026-10-10 | Initial protocol (decisions A–H and J approved; I pending; test group frozen) | — |
  | 1.1 | 2026-10-10 | Added §6.4 and Decision K: a count-only label-integrity check on the 20 additional participants. §6.3 item 2 now refers to it. | The only available copy of these participants has altered valence and arousal ratings, and their labels must be checked for consistency before any decision on reconstruction. This changes no participant role, fold, metric, seed, label rule, borderline band or acceptance criterion, and it does not alter the frozen test group. |
- **How to change it.** Any change to participant roles, folds, metrics, seeds, the label rule, the borderline band or the acceptance criteria:
  - requires a new version number and a written reason;
  - must be decided **before** the experiments it affects.
- **Every result records:**
  - the protocol version;
  - the fold-file hash;
  - the dataset fingerprint;
  - the git commit.
- **Pre-protocol results stay as they are:** Review 2 baselines, the fixed-split model comparison, and the 28-run robust CNN evaluation. They are labelled *pre-protocol* when cited.

---

## 12. Decision log

| # | Decision | Status | Rule |
|---|---|---|---|
| A | Primary method | **Approved 2026-10-10** | Nested LOPO on development participants. Grouped K-fold (K ≥ 8) only as a documented fallback for costly models, declared in advance (§2.1). |
| B | Status of s01, s09, s11 | **Approved 2026-10-10** | Development participants from now on. Their earlier test results are kept as historical, pre-protocol results (§1, §11). |
| C | Interim period | **Approved 2026-10-10** | Cross-validation on the current 12 participants for exploratory work only, labelled "exploratory (12 participants, Protocol v1.0)". No independent-test claims and no formal claim of outperforming the Random Forest; formal §10 comparisons wait for the full dataset (§1). |
| D | Seeds | **Approved 2026-10-10** | 42–46 for primary model comparisons, including the Random Forest (§4). |
| E | Inner validation | **Approved 2026-10-10** | 3 rotating participants when ≥ 15 training participants are available, otherwise 2 (§2.2). |
| F | Labels and borderline trials | **Approved 2026-10-10** | `valence > 5.0` is the primary rule. Excluding 4.5 ≤ valence ≤ 5.5 is reported only as the separate secondary analysis S1 (§7). |
| G | Final test group | **Approved 2026-10-10; FROZEN 2026-10-10** | 8 randomly selected newly acquired participants. Seed and IDs recorded and frozen before their data is inspected. No stratification (§8.2). See the freeze record below. |
| H | Acceptance criteria | **Approved 2026-10-10** | Criteria as in §10. If they are not met, the outcome is reported as inconclusive or exploratory. Thresholds are never changed after results are observed. |
| I | Data access and storage | **PENDING** | Not decided: who signs the DEAP EULA, who downloads the data, where the files are stored, and whether the duplicate local copy is removed. Until confirmed: no signing, downloading, moving or deleting of dataset files, and no configuration change (§8.1). |
| J | Few-trials rule | **Approved 2026-10-10** | Flag participants with fewer than 5 trials of either class, and report aggregates with and without them (§3.4). |
| K | Label-integrity check (v1.1) | **Approved 2026-10-10** | One count-only consistency check of `participant_ratings.xls` against the Kaggle `.dat` labels for all 20 additional participants, as defined in §6.4. PASS/FAIL and failure counts only; no values or patterns. Integrity only; not authorization. |

**Final test group: freeze record (Decision G)**

| Field | Value |
|---|---|
| Frozen on | 2026-10-10, before any rating values, EEG signal values, plots or class balances of the 20 additional participants were examined |
| Candidate pool | s04, s06, s10, s12, s13, s14, s17, s18, s19, s20, s21, s23, s24, s26, s27, s28, s29, s30, s31, s32 (the 20 participants missing from the original 12) |
| Seed | `20261010` |
| Procedure | Sort the 20 IDs, `numpy.random.default_rng(20261010).permutation(20)`, take the first 8 (numpy 2.5.3). No stratification. |
| **Final test group (8)** | **s12, s17, s18, s20, s21, s23, s29, s30** |
| New development participants (12) | s04, s06, s10, s13, s14, s19, s24, s26, s27, s28, s31, s32 |
| Seen before the freeze (allowed under §6.3) | File presence and size, SHA-256, pickle structure, array shapes and dtypes, finite-value checks, and a pass/fail check of whether ratings lie within 1–9. No rating values, class balances or signal statistics were displayed. |
| Data source status | The only copy of these participants is a third-party Kaggle upload whose valence and arousal ratings are altered (see the dataset validation report of 2026-10-10). Their usable labels are not yet established, and Decision I (data access) is pending. |

The group is frozen: it cannot be redrawn, swapped or extended. Until final evaluation, only the integrity checks in §6.3 (including the §6.4 check, from v1.1) are allowed on these 8 participants.

**Still to record:**
- Decision I, once the team confirms it.
