# NeuroSense AI: Project Report

**EEG-based estimation of self-reported valence using machine learning, deep learning and explainable AI**

| | |
|---|---|
| Type | Academic research prototype (B.Tech capstone), local single-computer application |
| Report date | 2026-10-20; results updated 2026-10-23 (§6.3, §6.4) |
| Evidence base | The repository code, the automated tests, and the saved results and reports in `docs/` (sources in §10). Every number comes from a saved result file. |
| Status of results | Historical fixed-split results (pre-protocol) and **exploratory** results under Evaluation Protocol v1.1 on 12 participants. **No independent final-test result exists yet.** |

---

## 1. Problem and scope

The project asks whether a person's **self-reported valence** can be estimated from EEG. Valence here is the person's own 1–9 rating of how pleasant a one-minute music video felt, collapsed to two categories: above 5 ("higher") and 5 or below ("lower").

The system is a research tool, not a medical or psychological instrument. It does not detect exact emotions, diagnose any condition, or claim to work on new, uncalibrated people. The research findings (§6) show why those claims would be unsupported.

## 2. Data

- **Source:** DEAP `data_preprocessed_python`: 32 EEG channels at 128 Hz and 40 one-minute trials per participant, each followed by valence, arousal, dominance and liking ratings.
- **Trusted local data:** 12 participants (s01, s02, s03, s05, s07, s08, s09, s11, s15, s16, s22, s25), 480 trials. All development and evaluation uses only these.
- **Frozen final-test group:** s12, s17, s18, s20, s21, s23, s29, s30. Frozen before any of their data was examined, and never used. Development code refuses to load them: there is a loader guard and a settings validation.
- **Data-access decision (Decision I):** pending. The only available copy of the other DEAP participants is an unauthorized third-party upload with altered labels on about 30 % of trials. It is not used.

## 3. System architecture

| Layer | Implementation |
|---|---|
| Data | `DEAPDataset`: restricted unpickler (numpy reconstruction only), per-file validation, frozen-participant guard |
| Signal processing | `EEGPreprocessor`: 3 s baseline removal, 4–45 Hz zero-phase Butterworth band-pass, per-trial robust clipping (median ± 6 MAD), non-overlapping 4 s segments, rejection of segments with >5 % clipped samples |
| Features | `FeatureExtractor`: 440 per segment. Per channel: log and relative band power (theta, alpha, beta, gamma), spectral entropy, Hjorth mobility and complexity, log variance. Plus 56 hemispheric asymmetry features. Cached by a dataset/configuration fingerprint |
| Models | Random Forest (active), RBF SVM, XGBoost, and an EEGNet-style CNN (PyTorch, registered and inactive). Scaling sits inside each pipeline |
| Training | `ProtocolTrainer` (Protocol v1.1, current) and `BaselineTrainer` (historical fixed split) |
| Registry | `outputs/models/<version>`: artifact + metadata with SHA-256; an atomic active pointer |
| Backend | FastAPI, SQLAlchemy 2, PostgreSQL (Alembic migrations), JWT authentication with bcrypt, role checks on the server, audit log |
| Frontend | React 18 + Vite + Chart.js. Pages: Overview, Analyze EEG, History, Model & Evidence, Model registry, Dataset, System health, Training (admin) |
| Operations | Windows start/stop/status scripts (loopback only), GitHub Actions CI |

**Communication:** the browser calls `/api/*`, which Vite proxies to FastAPI on `127.0.0.1:8000`. The backend authorizes every request itself; the interface only hides controls for convenience.

## 4. Methodology

### 4.1 Labels and evaluation unit
- **Label:** `high_valence` if valence > 5.0, else `low_valence`. One label per trial, inherited by its segments.
- **Trial prediction:** the mean of segment probabilities, then argmax (0.5 threshold).
- **Primary metrics:** trial-level balanced accuracy and macro-F1, computed **per held-out participant**. The participant is the unit of analysis.

### 4.2 Leakage control
- **Whole participants only:** a participant's segments never appear in both training and evaluation.
- **Per-trial preprocessing:** baseline, filtering and clipping use statistics from the same trial only.
- **Learned transforms inside the fold:** scaling is fitted on the fold's training participants only.
- **Protocol candidates:**
  - no model selection or tuning (the algorithm is fixed in advance);
  - each fold asserts that the held-out person and their trial (video) groups are absent from training.

### 4.3 Evaluation Protocol v1.1 (summary of `docs/EVALUATION_PROTOCOL.md`)
- **Development participants:** all 12 trusted participants, evaluated by leave-one-participant-out (LOPO).
- **Reporting:** mean ± SD with t-based CIs across participants, plus a majority-class reference.
- **Mandatory diagnostics:** predicted-class share, single-label participants, within-person AUC.
- **Labelling:** interim results on 12 participants are "exploratory" and may not support claims that one model outperforms another.
- **Final evaluation:** one evaluation of an accepted model on the frozen group, after Decision I.

## 5. What was implemented in the final completion phase

1. **Secure administrator bootstrap.**
   - Public registration always creates standard accounts, and role fields in a signup payload are rejected.
   - The first administrator comes only from a local command, which refuses if an administrator already exists and never promotes an existing account.
   - There is an audited password-recovery mode for an existing administrator.
2. **Protocol-aware candidate training.**
   - Uses development participants only, with the frozen guard enforced.
   - Fixed Random Forest, exploratory LOPO, trial-level per-participant metrics with diagnostics.
   - Full split and trial-group tracking, explicit limitations in the metadata, and an inactive candidate by default.
3. **Explainability.** Global impurity-based feature importance for the active Random Forest, with schema-alignment checks, an explicit unavailable state for other models, and stated limitations. It is not presented as a per-prediction explanation.
4. **Integration.**
   - The API, registry, metrics view and activation dialog all handle protocol candidates.
   - The Analyze page explains when no held-out participant exists, and offers a clearly labelled synthetic test signal.
5. **Quality and operations.**
   - Frontend unit tests, and a GitHub Actions workflow (backend, deep-learning and frontend jobs; migration check).
   - A fixed frontend dependency conflict (`react-chartjs-2` 4 → 5 for Chart.js 4).
   - Windows start/stop/status scripts, and updated documentation.

These build on the earlier hardening:
- public/admin endpoint separation and response sanitisation;
- a read-only registry listing and a no-op startup sync;
- audited activation with rollback;
- snapshot-consistent predictions;
- a loopback-only development server.

## 6. Experiments and results (existing evidence only)

### 6.1 Historical fixed split (pre-protocol)

Test participants s01, s09, s11:

| Model | Trial balanced accuracy | Trial macro-F1 | Segment macro-F1 |
|---|---|---|---|
| Random Forest | 0.590 | 0.590 | 0.543 |
| SVM (RBF) | 0.459 | 0.406 | 0.460 |
| XGBoost | 0.515 | 0.515 | 0.499 |
| CNN (seed 42) | 0.533 | 0.524 | 0.525 |
| Majority reference | 0.500 | 0.344 | 0.344 |

These participants are now development participants, so these are historical numbers only.

The robust CNN evaluation (5 seeds, nested LOPO) found:
- test segment macro-F1 of **0.466 ± 0.109**;
- LOPO balanced accuracy of 0.494 ± 0.023, not different from 0.5;
- prediction collapse to one label per person.

### 6.2 Exploratory studies under Protocol v1.1 (12 participants)

| Setting | Trial balanced accuracy | Reading |
|---|---|---|
| Zero-shot, new person and unseen videos | **0.500** [0.481, 0.520] | At chance |
| Unsupervised per-person normalization | 0.535 (+0.034, p = 0.098) | Removes collapse; no significant gain |
| Supervised within-person calibration (32 rated videos) | **0.618** [0.550, 0.687] | Above chance (p = 0.004) |
| Same calibration, predicting the other participants' consensus | 0.626 | As good as the person's own rating |

### 6.3 Model-improvement and covariance-alignment experiments (2026-10-22/23)

**Model-improvement study** (`docs/MODEL_IMPROVEMENT_EXPERIMENTS.md`):
- **Design:**
  - nested leave-one-participant-out with in-fold selection over RF, XGBoost, logistic regression and SVM and over 4 trial-aggregation rules;
  - feature ablations, regression, and calibration-size curves.
- **Participant-independent:** 0.541 [0.504, 0.579] vs the RF's 0.533. No model choice, feature subset or aggregation rule helped.
- **Personalized calibration (32 rated videos):** 0.620. With 16 videos, about 0.60.
- **No valid protocol came near 90%.** A deliberately leaky segment-level split scored 0.805, which illustrates how inflated figures arise.

**Covariance alignment** (`docs/COVARIANCE_ALIGNMENT_EXPERIMENT.md`):
- **What it is:** per-person Riemannian re-centring of 4-band covariance features.
- **Result, new person on new videos:** 0.499 vs the baseline's 0.493 (Δ +0.006 [−0.025, +0.038]). That is a negative result, below simple per-person z-scoring (0.535).

**Model experimentation on the 12 participants is frozen.** The active Random Forest is unchanged, and no experimental candidate is registered or activated.

**Comparability.** The historical 59.2% and these figures are different measurements:
- 59.2% is pooled trial *accuracy* on 3 fixed people, with the test videos' stimuli in training.
- 53.3% and 49.3% are per-participant *balanced* accuracy over all 12 people, the latter with the test videos excluded from training.

They must not be compared as if one model improved or degraded. See `RESEARCH_FINDINGS_AND_NEXT_STEPS.md` §9c.

### 6.4 Interpretation
- **Supported:**
  - no zero-shot transfer to new people (every valid estimate is near 0.5);
  - collapse driven by per-person offsets;
  - above-chance within-person calibration (exploratory, about 0.6), whose signal appears shared across people and related to the stimulus;
  - the classifier and feature representation are not the bottleneck.
- **Not supported:**
  - reliable emotion recognition;
  - generalization to new users;
  - individual-specific emotion estimation;
  - any 90% figure;
  - any clinical or product performance figure.

### 6.5 Explainability output (active model)

The active Random Forest's global importance is concentrated in gamma-band features (about 31 % of total importance by band). Because gamma-band EEG is easily contaminated by muscle and eye-movement activity, this is reported as a property of the model, not as evidence of a neural valence signal.

## 7. Testing and verification

- **Backend:** pytest suite on synthetic DEAP-format fixtures and temporary SQLite databases. It covers:
  - authentication and roles;
  - administrator bootstrap;
  - endpoint authorization;
  - response sanitising;
  - registry sync and activation with rollback;
  - snapshot consistency under concurrent activation;
  - protocol training (fold disjointness, frozen guard, pointer untouched);
  - explainability.
- **Integration:** 2 read-only tests on the real DEAP files.
- **Frontend:** unit tests of the activation rules, display helpers and synthetic-signal generator; production build.
- **Browser:** end-to-end checks of the Stage D workflows and the earlier regression suite, run against isolated backends (scratch SQLite, temporary registries, test-only models).
- **Data safety:** read-only checks before and after confirm the real PostgreSQL database and model registry were unchanged by testing.

Exact commands and pass counts: [FINAL_COMPLETION_REPORT.md](FINAL_COMPLETION_REPORT.md).

## 8. Limitations

- 12 participants; no independent test; single-seed protocol candidates.
- One self-report per 60 s trial; the 4 Hz high-pass removes delta information.
- Global importance only; impurity importance is biased and is not causal.
- Uploads must be in DEAP layout; there is no resampling.
- The application is designed and secured for local, single-computer use. It is not a hardened multi-user deployment: tokens are kept in `sessionStorage`, and training state lives in memory.

## 9. Future work

1. **Resolve Decision I.**
   - Obtain and verify the official DEAP release.
   - Run the frozen-group evaluation once. It is **BLOCKED** until then.
2. **Resume model work only on the full development set,** as five-seed pre-declared comparisons with a versioned fold-definition file. The references are per-person z-scoring (new users) and the personal Random Forest (calibrated users).
3. **Develop supervised calibration as an explicit, exploratory mode.**
4. **Explainability:**
   - permutation importance on held-out participants;
   - per-prediction attribution only if it can be validated;
   - an explicit test of whether muscle (EMG) activity drives the gamma-band signal.

## 10. Sources

| Topic | Document |
|---|---|
| Protocol and decisions | `docs/EVALUATION_PROTOCOL.md` |
| Consolidated findings | `docs/RESEARCH_FINDINGS_AND_NEXT_STEPS.md` |
| Baselines | `docs/REVIEW2_REPORT.md`, `docs/REVIEW3_PROGRESS_30.md` |
| Robust CNN | `docs/REVIEW3_CNN_ROBUST_EVALUATION.md` |
| Exploratory studies | `docs/EXPLORATORY_*.md`, `docs/results/exploratory/` |
| Model improvement and covariance alignment | `docs/MODEL_IMPROVEMENT_EXPERIMENTS.md`, `docs/COVARIANCE_ALIGNMENT_EXPERIMENT.md` |
| Completion status | `docs/FINAL_COMPLETION_REPORT.md` |
