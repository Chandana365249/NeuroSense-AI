# NeuroSense AI

**EEG-based estimation of self-reported valence using machine learning, deep learning and explainable AI.** This is an academic research prototype that runs locally on one computer.

NeuroSense AI loads EEG recordings from the DEAP dataset, preprocesses them, extracts spectral features and trains a classifier. It evaluates models only on participants they have never seen, and serves predictions, evaluation results, global feature importance and history through an authenticated FastAPI backend with a React dashboard.

> **Not a medical device.** Outputs estimate a *self-reported* valence category (higher or lower than the scale midpoint) from EEG features. They are not a diagnosis of any condition, they do not detect a person's exact emotions, and EEG does not "read thoughts". Cross-participant estimates are near chance (section 7).

**Contents:** [1 Architecture](#1-architecture) · [2 Dataset](#2-dataset-and-data-boundaries) · [3 Labels](#3-labeling-strategy) · [4 Preprocessing](#4-preprocessing-appppreprocessingeeg_preprocessorpy) · [5 Features](#5-feature-extraction-apppreprocessingfeaturespy) · [6 Training](#6-model-training) · [7 Results](#7-evaluation-and-current-results) · [8 Explainability](#8-explainability-global-feature-importance) · [9 Database](#9-database-schema) · [10 API](#10-api) · [11 Authentication](#11-authentication-and-administrator-bootstrap) · [12 Configuration](#12-configuration) · [13 Running locally](#13-running-locally-windows) · [14 Tests and CI](#14-tests-and-continuous-integration) · [15 Troubleshooting](#15-troubleshooting) · [16 Limitations](#16-current-limitations) · [17 Future work](#17-future-work)

---

## 1. Architecture

```
DEAP .dat files ─► DEAPDataset (restricted unpickler, validation, frozen-participant guard)
                    └► EEGPreprocessor (baseline removal, 4–45 Hz filter, artifact clipping, 4 s segments)
                        └► FeatureExtractor (440 features / segment, cached by fingerprint)
                            ├► ProtocolTrainer  (Evaluation Protocol v1.1: development participants only,
                            │                    fixed Random Forest, exploratory leave-one-participant-out)
                            └► BaselineTrainer  (historical, pre-protocol fixed split; kept for reproduction)
                                └► ModelRegistry (outputs/models/<version>, SHA-256, atomic active pointer)

Browser ─► React (Vite, 127.0.0.1:5173) ─► /api/* proxied to FastAPI (127.0.0.1:8000)
FastAPI ─ routes (thin) ─► services ─► repositories ─► SQLAlchemy ─► PostgreSQL (tests: temporary SQLite)
          ├─ ModelService: active model in memory; each request uses one immutable snapshot
          ├─ TrainingService: one background job; result is always an inactive candidate
          └─ Explainability: global importance of the active Random Forest
```

**Request flow.** The React app calls relative `/api/...` URLs. In development, Vite proxies them to the backend on `127.0.0.1:8000`. The backend authenticates the request (JWT bearer token), checks the role on the server, runs the service, and returns JSON in a fixed shape (`{"error": {"code", "message", "details?", "request_id"}}` on errors). Hiding a link in the UI never grants or removes access.

```
backend/
  app/
    main.py                 app factory, lifespan (load active model, no-op registry sync), routers, static graphs
    core/                   settings (env-driven), JSON logging, bcrypt/JWT, error handlers
    database/               declarative base, UTC datetimes, session dependency
    models/                 ORM entities (users, sessions, models, metrics, predictions, reports, audit_logs)
    schemas/                Pydantic request/response models
    repositories/           all database queries
    services/               auth (+ admin bootstrap), model (snapshots, activation), prediction, training,
                            explainability, public_metadata (response sanitising)
    api/                    dependencies (current user, admin) and route modules
    ml/                     dataset, participants (protocol roles), labels, splitting, feature table, models,
                            protocol_training, training (historical), evaluation, stats, registry; deep/ (CNN)
    preprocessing/          EEGPreprocessor and FeatureExtractor
  alembic/                  migrations
  scripts/                  create_admin, train_candidate, train_baseline (historical), dataset_summary, reports
  tests/                    pytest suite (synthetic DEAP-format fixtures, temporary SQLite)
frontend/
  src/                      pages, components, API client, pure logic modules (activationRules, synthetic, format)
  tests/                    unit tests (Node's built-in test runner)
scripts/                    start.ps1, stop.ps1, status.ps1 (Windows, loopback only)
.github/workflows/ci.yml    CI: backend, deep-learning and frontend jobs
docs/                       protocol, research findings, exploratory reports, project report, completion report
outputs/                    models, metrics, graphs, predictions, logs, feature cache (generated, not committed)
```

## 2. Dataset and data boundaries

[DEAP](https://www.eecs.qmul.ac.uk/mmv/datasets/deap/), "data_preprocessed_python" release. Each participant file `sNN.dat` is a Python 2 pickle:

| key | shape | content |
|---|---|---|
| `data` | (40 trials, 40 channels, 8064 samples) | 32 EEG channels (Geneva order) + 8 peripheral; 128 Hz; 63 s = 3 s pre-trial baseline + 60 s music video |
| `labels` | (40 trials, 4) | self-assessment ratings: valence, arousal, dominance, liking, on a 1–9 scale |

**What this project may use.**

| Group | Participants | Use |
|---|---|---|
| Trusted development participants | s01, s02, s03, s05, s07, s08, s09, s11, s15, s16, s22, s25 (480 trials) | All development, training and exploratory evaluation |
| Frozen final-test group | s12, s17, s18, s20, s21, s23, s29, s30 | **Never** loaded by development code. Reserved for one final evaluation (blocked, see below) |
| Other DEAP participants | — | Not used. The only available copy is an unauthorized third-party (Kaggle) upload with altered labels; it is not integrated |

These roles are enforced in code (`app/ml/participants.py`): the dataset loader skips and refuses frozen participant files, and the settings refuse a development list that contains any of them.

The loader (`app/ml/dataset.py`):
- uses a **restricted unpickler** that only allows numpy array reconstruction, so a tampered file cannot execute code;
- validates every file (dimensions, matching trial counts, ≥32 channels, finite values, ratings within 1–9);
- reports corrupt or missing files instead of crashing.

The DEAP data is licensed separately and is **never committed** to the repository, used in CI, or included in documents.

**Blocked: data access (Decision I).** Who signs the DEAP end-user licence agreement, who downloads the official release, where it is stored and whether the duplicate local copy is removed are still undecided (`docs/EVALUATION_PROTOCOL.md` §12). Until then the frozen group cannot be evaluated. What unlocks it is listed in [docs/FINAL_COMPLETION_REPORT.md](docs/FINAL_COMPLETION_REPORT.md#blocked-items-and-the-shortest-next-step).

**Dataset limitations**
- Only 12 participants, so every estimate is statistically fragile.
- One self-report per 60 s video; all segments of a trial inherit that label.
- The DEAP preprocessed release is high-passed at 4 Hz with EOG removed, so no delta-band information.
- A small, non-representative sample.

## 3. Labeling strategy

`app/ml/labels.py`, configured with `NEUROSENSE_LABEL_STRATEGY` and the thresholds:

| strategy | classes | rule |
|---|---|---|
| `valence_binary` (default, the only one evaluated) | `low_valence`, `high_valence` | high if valence **>** 5.0; a rating exactly at the threshold counts as low |
| `arousal_binary` | `low_arousal`, `high_arousal` | same rule on arousal (not evaluated; no arousal claim is made) |
| `quadrant` | `negative/positive` × `low/high_arousal` | both thresholds (not evaluated) |

The threshold of 5 is the scale midpoint. It is an **operational project choice, not a psychological ground truth**. The interface shows the classes as "Higher self-reported valence" and "Lower self-reported valence". With the 12 participants, segment-level class balance is high_valence 3714 vs low_valence 3332.

## 4. Preprocessing (`app/preprocessing/eeg_preprocessor.py`)

| step | what | why |
|---|---|---|
| validate | (32, samples), finite, not flat, ≥ one segment long | reject unusable input early |
| remove baseline | drop the first 3 s | it is pre-stimulus, not a response to the video |
| band-pass | 4th-order Butterworth, 4–45 Hz, zero-phase (`sosfiltfilt`) | matches DEAP's band and makes arbitrary API input comparable |
| artifacts | per-channel clipping at median ± 6·(1.4826·MAD), statistics from the same trial | limits spikes without leaking information across trials |
| segment | non-overlapping 4 s windows (15 per trial) | more examples; each inherits the trial label |
| reject | drop segments with >5 % clipped samples | 154 of 7,200 real segments (2.1 %) rejected |
| scaling | `StandardScaler` inside the model pipeline | fitted on the training participants of each fold only |

All preprocessing is per trial, so no information crosses trials or participants. Figures: `python -m scripts.preprocessing_report`.

## 5. Feature extraction (`app/preprocessing/features.py`)

440 features per 4 s segment:
- **per channel (32 × 12):** log10 absolute band power in theta (4–8 Hz), alpha (8–13), beta (13–30) and gamma (30–45) from a Welch PSD; relative band power for the same bands; normalized spectral entropy; Hjorth mobility and complexity; log variance;
- **hemispheric asymmetry (14 left/right pairs × 4 bands):** log power(right) − log power(left).

The feature table is cached in `outputs/cache/` under a fingerprint of the files and the preprocessing settings, so a changed configuration never reuses stale features.

## 6. Model training

### 6.1 Protocol-aware candidate training (current; `app/ml/protocol_training.py`)

Used by the web **Train candidate** button (`POST /api/train`) and `python -m scripts.train_candidate`. It follows Evaluation Protocol v1.1:

1. **Data:** only the development participants (default: the 12 trusted ones). Frozen participants can never be loaded.
2. **Model fixed in advance:** Random Forest (300 trees, `min_samples_leaf=5`, balanced class weights) behind a `StandardScaler`. No tuning and no model selection, so no selection step can leak information.
3. **Exploratory leave-one-participant-out (LOPO) evaluation:** for each participant, train on everyone else and evaluate once on that person. Every fold asserts that the held-out person is not in training and that no trial (video) group appears on both sides.
4. **Metrics per held-out participant at trial level** (segment probabilities averaged within a trial, 0.5 threshold): balanced accuracy, macro-F1, accuracy, within-person ROC-AUC. Also the majority-class reference from the fold's training participants, and the mandatory diagnostics (predicted-class share, single-label participants, few-trials flag). Aggregates are mean ± SD with a t-based 95 % CI across participants.
5. **Final model:** refitted on all development participants and saved as an **inactive candidate**.
6. **Metadata:** split strategy, participants and trial groups per participant, label rule, feature schema, seed, leakage controls and limitations, plus per-fold results. LOPO trial predictions go to `outputs/predictions/<version>_lopo_trial_predictions.csv`.

These results are labelled **exploratory (development participants only, Protocol v1.1)**. They are not an independent test result and must not be used to claim one model beats another (single seed; the protocol requires seeds 42–46 and the §10 criteria for that).

### 6.2 Historical baseline training (`app/ml/training.py`)

`python -m scripts.train_baseline` reproduces the Review 2 pipeline: a fixed participant split (train s02, s03, s05, s07, s08, s15, s22; validation s16, s25; test s01, s09, s11), candidate selection by validation macro-F1 among Random Forest, RBF SVM and XGBoost, then refit and one test evaluation. Under the protocol, s01, s09 and s11 are now development participants, so these numbers are **historical**. The active model was produced this way.

### 6.3 Model lifecycle: candidates vs the active model

- **Training never replaces the active model.** Every new version, from the API or a script, is saved and registered as an **inactive candidate**. The active model, its pointer (`outputs/models/active_model.json`) and the predictions it serves stay as they were.
- **Activation is explicit and audited.** An administrator calls `POST /api/models/{version}/activate`, or uses **Activate…** in the Model registry. They must retype the exact version and give a 10–500 character reason, which is stored in the audit log; text that looks like a credential is refused. The server checks the SHA-256, the feature schema and the class labels, and runs one neutral test prediction before switching. It then updates, together: the pointer (atomically, via a temporary file and `os.replace`), the in-memory model, the database `is_active` flag and the audit entry. If any step fails, it restores the previous state.
- **Rollback:** activate the previous version the same way.
- **Script option:** `--activate` on either training script activates the new model directly. A running server keeps its loaded model until restart.
- **Predictions are consistent.** Each prediction takes one snapshot of the model and its metadata. A concurrent activation cannot mix one model's output with another model's version or labels.

## 7. Evaluation and current results

Metrics (`app/ml/evaluation.py`): accuracy, balanced accuracy, macro/weighted precision, recall and F1, confusion matrix and ROC-AUC, reported per 4 s segment and per trial.

**Historical fixed-split result, model `valence_binary-random_forest-20261005-182352`, test participants s01, s09, s11.** The active model, `valence_binary-random_forest-20261008-180558`, reproduces these metrics exactly.

| | accuracy | balanced acc. | macro-F1 | ROC-AUC | n |
|---|---|---|---|---|---|
| Random Forest, segment | 0.548 | 0.544 | 0.543 | 0.537 | 1788 |
| Random Forest, trial | 0.592 | 0.590 | 0.590 | 0.565 | 120 |
| Majority-class reference, segment | 0.525 | 0.500 | 0.344 | 0.500 | 1788 |

Pre-protocol leave-one-participant-out CV (9 participants): accuracy **0.510 ± 0.086**, macro-F1 **0.416 ± 0.050**.

**EEGNet-style CNN, robust evaluation** ([docs/REVIEW3_CNN_ROBUST_EVALUATION.md](docs/REVIEW3_CNN_ROBUST_EVALUATION.md)): test segment macro-F1 **0.466 ± 0.109** over 5 seeds vs the Random Forest's 0.543; LOPO balanced accuracy 0.494 ± 0.023. The CNN does not outperform the Random Forest and stays inactive.

**Exploratory studies under Protocol v1.1** (12 participants; [docs/RESEARCH_FINDINGS_AND_NEXT_STEPS.md](docs/RESEARCH_FINDINGS_AND_NEXT_STEPS.md)):

| Setting | Trial balanced accuracy |
|---|---|
| Zero-shot: new person, unseen videos (leave-video-out) | **0.500**, 95 % CI [0.481, 0.520] |
| Unsupervised per-person normalization | 0.535 (difference not significant; removes single-label collapse) |
| Supervised within-person calibration (32 rated videos) | **0.618**, CI [0.550, 0.687] |
| Nested model selection (RF, XGBoost, LR, SVM; 4 aggregation rules), participant-independent | 0.541, CI [0.504, 0.579]; no improvement over the RF ([docs/MODEL_IMPROVEMENT_EXPERIMENTS.md](docs/MODEL_IMPROVEMENT_EXPERIMENTS.md)) |
| Per-person covariance alignment, new person and unseen videos | 0.499, CI [0.458, 0.539], vs baseline 0.493; negative result ([docs/COVARIANCE_ALIGNMENT_EXPERIMENT.md](docs/COVARIANCE_ALIGNMENT_EXPERIMENT.md)) |

**These numbers are not interchangeable with the 0.592 above.**
- **The 0.592:** pooled trial *accuracy* on three fixed people, with the test videos' stimuli in training.
- **The table's numbers:** per-participant *balanced* accuracy over all 12 people, and the leave-video-out rows also hold out the test videos.
- **Details:** [docs/RESEARCH_FINDINGS_AND_NEXT_STEPS.md §9c](docs/RESEARCH_FINDINGS_AND_NEXT_STEPS.md).

Model experimentation on these 12 participants is frozen: the active Random Forest stays, and no experimental candidate is registered or activated.

**Interpretation.** Zero-shot prediction of a new person's valence category is at chance. A model calibrated on a person's own rated videos does better, but it predicts the other participants' consensus for each video equally well, so the signal looks shared and stimulus-related rather than individual. Nothing here supports clinical use, reliable emotion recognition, cross-person generalization or individual-level explanation. Many published DEAP accuracies far above these come from random segment-level splits, which leak each participant's data between train and test.

## 8. Explainability: global feature importance

`GET /api/explain/global` and the **Global feature importance** card on Model & Evidence.

- **What it is:** the active Random Forest's impurity-based importance (mean decrease in Gini impurity, averaged over its 300 trees), read from the trained model. It covers the top features plus totals by frequency band and by feature type, and names the model version.
- **When it is shown:** only when the model demonstrably aligns with its feature schema (a scikit-learn pipeline ending in a `RandomForestClassifier`, with exactly one value per named feature). For a CNN, another algorithm or a schema mismatch, the API and UI say it is unavailable and why.
- **Global vs individual.** Global importance describes the whole model across its training data. It is **not** an explanation of any single prediction, and the app shows no per-prediction explanation. It is not evidence about what a person felt, and not causal.
- **Limitations shown with it:**
  - impurity importance can favour high-cardinality features;
  - correlated features share importance;
  - importance can reflect person-specific signal;
  - high gamma-band importance can come from muscle (EMG) and eye-movement contamination rather than neural activity.

## 9. Database schema

PostgreSQL via SQLAlchemy 2, with Alembic migrations in `backend/alembic/versions` (the CI workflow runs `alembic upgrade head` and `alembic check` on a temporary SQLite database; both pass locally).

| table | key columns | relations |
|---|---|---|
| `users` | email (unique), hashed_password (bcrypt), is_admin, is_active, last_login_at | 1-n sessions, predictions |
| `sessions` | name, description (analysis sessions that group predictions; not auth sessions) | n-1 users, 1-n predictions |
| `models` | version (unique), algorithm, label_strategy, class_labels, feature_count, artifact_path, is_active, trained_at | 1-n metrics |
| `metrics` | split (validation / test / test_reference / cross_validation / lopo_protocol), level, name, value | n-1 models |
| `predictions` | predicted_label, confidence (model probability), probabilities (JSON), segments used/rejected, input_source, source_reference | n-1 users, sessions, models |
| `reports` | report_type, title, content (JSON) | n-1 users, models |
| `audit_logs` | action, resource_type / id, details, ip_address | n-1 users |

All timestamps are stored and returned in UTC.

## 10. API

Interactive docs: **http://127.0.0.1:8000/api/docs** (ReDoc at `/api/redoc`).

| method | path | auth | purpose |
|---|---|---|---|
| GET | `/api/health` | – | public liveness: `status` and API `version` only |
| GET | `/api/admin/diagnostics` | admin | database, active-model and dataset status, environment (never URLs, credentials or secrets) |
| POST | `/api/auth/register` | – | create a **standard** account; unknown fields such as `is_admin` are rejected (422) |
| POST | `/api/auth/login` | – | OAuth2 password form (`username` = email) → JWT |
| GET | `/api/auth/me` | user | current user |
| POST | `/api/train` | admin | start protocol-aware training in the background (202); the result is an **inactive candidate** |
| GET | `/api/training/status` | admin | state, stage, progress, candidate version; failures show a generic message (details in the server log only) |
| GET | `/api/model-info` | – | active model metadata (`available: false` if none), including `provenance`: the participants its final fit used, taken only from recorded metadata (`recorded: false` when not recorded) |
| GET | `/api/models` | – | registered versions (read-only), with historical test metrics or a protocol candidate's exploratory summary, and each model's recorded `provenance` |
| POST | `/api/models/sync` | admin | idempotent registry → database sync (same as at startup; never changes the active model) |
| POST | `/api/models/{version}/activate` | admin | explicit, audited activation; body `{"confirm_version", "reason"}`. 404 unknown, 409 busy, 422 invalid request or model, 500 failed switch (previous model stays active). Already active: no writes, `changed: false` |
| GET | `/api/metrics` | – | evaluation of the active model: historical fixed-split metrics, or the protocol LOPO evaluation; dataset facts limited to approved aggregate fields |
| GET | `/api/explain/global?top=20` | – | global feature importance of the active Random Forest (1–100 features), or an explicit unavailable state |
| GET | `/api/dataset/summary` | – | aggregate dataset structure and the configured `authorized_participants`; no paths, file names or rating statistics |
| GET | `/api/admin/dataset/summary` | admin | full dataset diagnostics |
| GET | `/api/dataset/demo-trials` | – | every authorized, readable development participant (never a frozen one) with its actual trial indices and its recorded `model_role` for the active model (`fit`, `selection`, `held_out`, `not_used`, or `unknown` when provenance is unrecorded); input availability is independent of the evaluation split |
| POST | `/api/predict` | user | predict and store; `{"deap_reference": {...}}` or `{"eeg": {"signal": [[...]], "sampling_rate": 128}}` (add `"synthetic_test_signal": true` for generated test input; it is stored and shown as a synthetic test, never as an estimate) |
| GET | `/api/predictions[/{id}]` | user | own prediction history |
| POST/GET | `/api/sessions` | user | analysis sessions |
| GET | `/api/reports` | admin | training reports, sanitised (no filesystem paths); stored reports are unchanged |
| GET | `/api/graphs/<file>` | – | evaluation figures |

**Prediction pipeline:**
1. Validate the input: 32 channels, 128 Hz, at most 120 s.
2. Take one model snapshot.
3. Preprocess with that model's saved configuration.
4. Extract features and predict per segment.
5. Average the segment probabilities.
6. Store the prediction and write an audit entry.

"Model probability" is the model's raw probability, not a calibrated confidence.

## 11. Authentication and administrator bootstrap

- Passwords are hashed with bcrypt (12 rounds); plain-text passwords are never stored or logged.
- Access tokens are HS256 JWTs signed with `NEUROSENSE_JWT_SECRET_KEY`. The app refuses to start without it. Default lifetime: 60 min.
- `get_current_user` protects user routes and `get_admin_user` protects administrator routes, on the server.
- Unknown-email logins are checked against a dummy hash, which reduces user enumeration through timing.

**Every web registration creates a standard account**, including the first one in an empty database. A signup payload cannot request a role: unknown fields are rejected.

**Creating the first administrator** (command line, on this computer only):

```powershell
cd backend
..\.venv\Scripts\python -m scripts.create_admin --email admin@your-org.example --full-name "Research Admin"
```

- The password is asked for twice and never shown or logged. For automation, `--password-stdin` reads it from standard input.
- The command **refuses** if any administrator already exists, so it can never create a second one. It also refuses if the email already belongs to an account, so it never promotes existing accounts.
- It writes an `auth.admin_bootstrap` audit entry.
- Database errors are reported without connection details.

**Recovering administrator access:**

```powershell
..\.venv\Scripts\python -m scripts.create_admin --email admin@your-org.example --reset-password
```

This sets a new password for an existing administrator only, and is audited as `auth.admin_password_reset`. It creates no accounts.

Existing accounts and roles in your database are not changed by this behaviour.

**Audit log.** Training failures are recorded as `training.failed` with a safe description (no exception text). Entries written before that change may still hold raw exception text; they stay server-side and were not rewritten.

## 12. Configuration

All settings come from environment variables prefixed `NEUROSENSE_` or from `backend/.env` (see [`backend/.env.example`](backend/.env.example)). Never commit `backend/.env`.

| setting | covers |
|---|---|
| `DATASET_DIR`, `OUTPUT_DIR` | data and output locations (relative paths resolve from the project root) |
| `DATABASE_URL` | database connection (URL-encode special characters in the password) |
| `JWT_SECRET_KEY`, `ACCESS_TOKEN_EXPIRE_MINUTES` | authentication |
| `API_PREFIX`, `CORS_ORIGINS` | API (CORS allows only the local frontend origins) |
| `LOG_LEVEL`, `RANDOM_SEED` | logging, reproducibility |
| `LABEL_STRATEGY`, `VALENCE_THRESHOLD`, `AROUSAL_THRESHOLD` | labels |
| `DEVELOPMENT_PARTICIPANTS` | protocol training participants (default: the 12 trusted; frozen IDs are refused) |
| `TEST_PARTICIPANTS`, `VALIDATION_PARTICIPANTS` | historical baseline split sizes |
| `SAMPLING_RATE`, `SEGMENT_SECONDS` | signal processing |

## 13. Running locally (Windows)

**Prerequisites:** Python 3.13, Node.js 20+ (CI uses 22), PostgreSQL (tested with 18), and the DEAP files for the 12 trusted participants (only needed for DEAP-trial analyses and training).

**One-time setup:**

```powershell
python -m venv .venv
.venv\Scripts\python -m pip install -r backend\requirements.txt
copy backend\.env.example backend\.env     # set DATASET_DIR, DATABASE_URL and a JWT secret (see below)
cd backend
..\.venv\Scripts\alembic upgrade head       # create the schema
..\.venv\Scripts\python -m scripts.create_admin --email admin@your-org.example
cd ..\frontend
npm ci
```

- **JWT secret:** generate one with `.venv\Scripts\python -c "import secrets; print(secrets.token_urlsafe(48))"`.
- **PostgreSQL:** create the database first (`CREATE DATABASE neurosense_ai;`). Then set `NEUROSENSE_DATABASE_URL=postgresql+psycopg://postgres:<password>@localhost:5432/neurosense_ai`.

**Start, check and stop** (from the project root):

```powershell
powershell -ExecutionPolicy Bypass -File scripts\start.ps1    # checks prerequisites, starts both, waits for health
powershell -ExecutionPolicy Bypass -File scripts\status.ps1   # ports, loopback binding, health, active model
powershell -ExecutionPolicy Bypass -File scripts\stop.ps1     # stops only the processes start.ps1 started
```

`start.ps1` reports missing dependencies, an incomplete `backend\.env` (it checks keys, never values), an unreachable database, or busy ports. It never starts training. Logs and PID files go to `.run\` (not committed). `-BackendOnly` and `-FrontendOnly` are available.

| Address | |
|---|---|
| http://127.0.0.1:5173 | web app |
| http://127.0.0.1:8000/api/docs | Swagger API docs |
| http://127.0.0.1:8000/api/health | health check |

Both servers bind to **127.0.0.1 only**, so nothing is reachable from other machines.

**Manual start** (alternative):
- backend: `cd backend; ..\.venv\Scripts\python -m uvicorn app.main:create_app --factory --host 127.0.0.1 --port 8000`;
- frontend: `cd frontend; npm run dev`.

**Training and evaluation commands** (from `backend/`):

```powershell
..\.venv\Scripts\python -m scripts.dataset_summary        # verify the dataset files
..\.venv\Scripts\python -m scripts.train_candidate        # protocol-aware candidate (inactive); add --no-cv to skip LOPO
..\.venv\Scripts\python -m scripts.train_baseline         # historical fixed-split reproduction (inactive)
..\.venv\Scripts\python -m scripts.train_cnn              # CNN (needs requirements-dl.txt); registered, not activated
```

**Without the DEAP files:** the app still runs. Analyze EEG accepts uploaded JSON, and **Download a synthetic test file** creates a clearly labelled generated signal (noise plus sine rhythms, not EEG) for trying the upload path. Its result is not a meaningful valence estimate.

## 14. Tests and continuous integration

```powershell
cd backend
..\.venv\Scripts\python -m pytest -rs                    # all unit/API/ML tests (synthetic data, temporary SQLite)
..\.venv\Scripts\python -m pytest -m integration         # 2 read-only tests on the real DEAP files
cd ..\frontend
npm test                                                 # frontend unit tests
npm run build                                            # production build
```

- Tests generate small synthetic DEAP-format files in temporary folders and use temporary SQLite databases. They never touch PostgreSQL, the real model registry or the dataset.
- Deep-learning tests skip with a stated reason if `torch` is not installed.
- Current counts are recorded in [docs/FINAL_COMPLETION_REPORT.md](docs/FINAL_COMPLETION_REPORT.md).

**CI** (`.github/workflows/ci.yml`, on push and pull request, read-only permissions, no deployment, no artifacts):

| Job | What it runs |
|---|---|
| backend | `pip install`, `alembic upgrade head` + `alembic check` on temporary SQLite, `pytest -rs` |
| backend-deep-learning | CPU-only torch; CNN and robust-evaluation tests |
| frontend | `npm ci` (lockfile), `npm test`, `npm run build` |

Real-dataset tests are deselected in CI, and the job prints a notice saying so.

## 15. Troubleshooting

| Problem | What to do |
|---|---|
| `start.ps1`: "database is not reachable" | Start the PostgreSQL service. Check `NEUROSENSE_DATABASE_URL` (host, port, database name, URL-encoded password). Then run `alembic upgrade head`. |
| App refuses to start: JWT secret missing | Set `NEUROSENSE_JWT_SECRET_KEY` in `backend\.env`. |
| "Port 8000/5173 is already in use" | Run `scripts\status.ps1` to see the owner. Stop the old instance with `scripts\stop.ps1`, or end the process shown. |
| Model & Evidence says "No trained model" | No active model is loaded. Train a candidate (`scripts.train_candidate`) and activate it as an administrator, or restore `outputs/models/active_model.json`. |
| Model fails to load ("integrity check") | The artifact differs from its recorded SHA-256. Do not edit model files; retrain or restore the original. |
| Dataset page: "Dataset not available on this server" | `NEUROSENSE_DATASET_DIR` does not point to the DEAP files. Uploads and the synthetic test file still work. |
| Cannot sign in as administrator | Public sign-up never creates administrators. Use `scripts.create_admin`, or `--reset-password` for recovery. |
| `npm ci` fails | Use Node 20+, and delete `frontend\node_modules` before retrying. |

## 16. Current limitations

- Cross-participant estimates are at chance (section 7). The served model is a research baseline, not an emotion recognizer.
- 12 participants; no independent test until Decision I is resolved.
- Protocol candidate training uses one seed. Formal comparisons need five seeds and the protocol §10 criteria.
- Explainability is global only; there are no per-prediction explanations.
- Uploaded EEG must already be in DEAP layout (32 channels, Geneva order, 128 Hz, microvolts); no resampling or montage mapping.
- Training status is held in memory, with one job at a time; a restart loses in-progress status (completed models persist).
- `pickle`/`joblib` artifacts are trusted only because the registry writes them and verifies their SHA-256.
- The JWT is kept in `sessionStorage`. That is acceptable for a local prototype, but not hardened against XSS.
- Automated tests use SQLite. PostgreSQL is covered by migrations and the live health and diagnostics checks.

## 17. Future work

- **Data (highest priority):**
  - resolve Decision I and obtain the official DEAP release;
  - verify it against the trusted files;
  - run the one-time frozen-group evaluation under the protocol.
- **Evaluation:** five-seed protocol runs with the fold definition file (protocol §2.3), and paired comparisons against the Random Forest.
- **Calibration framing:** the supervised within-person calibration setting as an explicit, exploratory product mode.
- **Explainability:** permutation importance on held-out participants, and topographic maps. Per-prediction attribution only if it can be validated.
- **Product:** arousal targets once evaluated; EDF/CSV upload with resampling.
