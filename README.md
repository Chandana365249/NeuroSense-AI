# NeuroSense AI

**Emotion Recognition from EEG Brain Signals using Deep Learning and Explainable AI.** This is an academic research prototype.

NeuroSense AI loads EEG recordings from the public DEAP dataset, preprocesses them, extracts spectral features and trains a classifier. It then evaluates the classifier on participants it has never seen and serves predictions, metrics and history through an authenticated FastAPI backend with a React dashboard.

> **Not a medical device.** Outputs estimate a *self-reported* affect category (for example, low vs. high valence) from EEG features. They are not a diagnosis of depression, anxiety, ADHD or any other condition, and EEG does not "read thoughts".

---

## 1. Architecture

```
DEAP .dat files ─► DEAPDataset (safe loader, validation)
                    └► EEGPreprocessor (baseline removal, 4–45 Hz filter, artifact clipping, 4 s segments)
                        └► FeatureExtractor (440 features / segment)
                            └► labels (configurable thresholds) + participant-aware split
                                └► BaselineTrainer (RF / SVM / XGBoost vs majority reference)
                                    └► evaluation (segment + trial level, LOPO CV, figures)
                                        └► ModelRegistry (outputs/models/<version>, SHA-256)

FastAPI  ─ routes (thin) ─► services ─► repositories ─► SQLAlchemy ─► PostgreSQL 18 (automated tests use temporary SQLite)
          └─ ModelService keeps the active model in memory (loaded at startup; replaced only by an explicit administrator activation)
React (Vite) ─► /api/* (proxied to :8000)
```

```
backend/
  app/
    main.py                 FastAPI app factory, lifespan (model loading), routers, static graphs
    core/                   config (env-driven settings), JSON logging, security (bcrypt/JWT), error handlers
    database/               Declarative base, UTC datetime type, engine/session dependency
    models/                 ORM entities (users, sessions, models, metrics, predictions, reports, audit_logs)
    schemas/                Pydantic request/response models
    repositories/           all database queries
    services/               auth, model, prediction, training (background thread)
    api/                    dependencies (current user, admin) and route modules
    middleware/             request logging with request IDs
    ml/                     dataset loader, labels, splitting, feature table, models, training, evaluation, registry
      deep/                 Review 3: segment Dataset/DataLoader, EEGNet-style CNN, CNN trainer (PyTorch)
    preprocessing/          EEGPreprocessor and FeatureExtractor
  alembic/                  migrations
  scripts/                  dataset_summary, preprocessing_report, train_baseline
  tests/                    pytest suite (unit + integration)
frontend/                   React + Vite dashboard
outputs/                    models, metrics, graphs, predictions, logs, feature cache (generated)
docs/                       Review 2 report and screenshots
```

## 2. Dataset

[DEAP](https://www.eecs.qmul.ac.uk/mmv/datasets/deap/), "data_preprocessed_python" release. Each participant file `sNN.dat` is a Python 2 pickle containing:

| key | shape | content |
|---|---|---|
| `data` | (40 trials, 40 channels, 8064 samples) | 32 EEG channels (Geneva order) + 8 peripheral; 128 Hz; 63 s = 3 s pre-trial baseline + 60 s music video |
| `labels` | (40 trials, 4) | self-assessment ratings: valence, arousal, dominance, liking, on a 1–9 scale |

The local copy used here has **12 of the 32 participants** (s01, s02, s03, s05, s07, s08, s09, s11, s15, s16, s22, s25), which is **480 trials**. You can verify this with `python -m scripts.dataset_summary`.

The loader (`app/ml/dataset.py`) does the following:
- uses a **restricted unpickler** that only allows numpy array reconstruction, so a tampered file cannot execute code;
- validates every file's structure (dimensions, matching trial counts, ≥32 channels, finite values, ratings within 1–9);
- reports corrupt or missing files instead of crashing.

The DEAP data is licensed separately and is **not committed** to the repository.

**Dataset limitations**
- Only 12 participants are available, so test results rest on 3 people and are statistically fragile.
- Labels are one self-report per 60 s video. All segments of a trial inherit that label, even though the actual experience varies over the minute.
- The DEAP preprocessed release is already high-passed at 4 Hz with EOG removed, so delta-band information is not available.
- The participants are a small, non-representative sample.

## 3. Labeling strategy

`app/ml/labels.py`, configured with `NEUROSENSE_LABEL_STRATEGY` and the thresholds:

| strategy | classes | rule |
|---|---|---|
| `valence_binary` (default) | `low_valence`, `high_valence` | high if valence **>** threshold (default 5.0); a rating exactly at the threshold counts as low |
| `arousal_binary` | `low_arousal`, `high_arousal` | same rule on arousal |
| `quadrant` | `negative/positive` × `low/high_arousal` | both thresholds |

The threshold of 5 is the scale midpoint, a common convention in DEAP studies. It is an **operational project choice, not a psychological ground truth**.

With the 12 local participants, segment-level class balance is high_valence 3714 vs low_valence 3332.

## 4. Preprocessing (`app/preprocessing/eeg_preprocessor.py`)

| step | what | why |
|---|---|---|
| validate | (32, samples), finite, not flat, ≥ one segment long | reject unusable input early |
| remove baseline | drop the first 3 s | it is pre-stimulus, not a response to the video |
| band-pass | 4th-order Butterworth, 4–45 Hz, zero-phase (`sosfiltfilt`) | matches DEAP's band and makes arbitrary API input comparable |
| artifacts | per-channel clipping at median ± 6·(1.4826·MAD) | limits spikes without creating the steps that zeroing caused |
| segment | non-overlapping 4 s windows (15 per trial) | more training examples; each inherits the trial label |
| reject | drop segments with >5 % clipped samples | 154 of 7,200 real segments (2.1 %) rejected |
| normalize | per-channel z-score (available for future deep models) | not applied for the baseline, because it would erase absolute band power; features are standardized by a scaler fit on training participants only |

Development figures generated from real data by `python -m scripts.preprocessing_report`: `outputs/graphs/eeg_raw_example.png`, `eeg_filtered_example.png`, `eeg_segmented_example.png`, `feature_distributions.png`.

## 5. Feature extraction (`app/preprocessing/features.py`)

There are 440 features per 4 s segment:

- **per channel (32 × 12):**
  - log10 absolute band power in theta (4–8 Hz), alpha (8–13), beta (13–30) and gamma (30–45), from a Welch PSD;
  - relative band power for the same four bands;
  - normalized spectral entropy;
  - Hjorth mobility and complexity;
  - log variance.
- **hemispheric asymmetry (14 left/right pairs × 4 bands):** log power(right) − log power(left). Frontal alpha asymmetry is studied in relation to valence, but the effect is small and inconsistent in the literature.

The feature table is cached in `outputs/cache/` using a fingerprint of the files and the preprocessing settings.

## 6. Model training (`app/ml/training.py`)

1. **Participant-aware split** (`app/ml/splitting.py`). The sorted participant IDs are shuffled with `numpy.random.default_rng(42)`: the first 3 become **test**, the next 2 **validation**, and the remaining 7 **train**. All segments of a participant stay in one split, so there is no subject leakage.
   Current split: train s02, s03, s05, s07, s08, s15, s22 · validation s16, s25 · test s01, s09, s11.
2. **Candidates.** Each is a `StandardScaler` + classifier pipeline, fitted on training data only:
   - majority-class reference (chance level);
   - Random Forest;
   - RBF SVM with calibrated probabilities;
   - XGBoost.

   All are wrapped in `EncodedLabelClassifier`, so the class mapping always comes from the fitted model.
3. **Selection** by validation macro-F1. The majority reference is never selected.
4. **Final fit** on train + validation participants, then a single evaluation on the test participants.
5. **Leave-one-participant-out CV** over the 9 non-test participants, to show how much results vary from person to person.
6. **Saved artifacts:**
   - `outputs/models/<version>/model.joblib` and `metadata.json`, containing the feature schema, class labels, preprocessing and label configuration, split, all metrics and SHA-256;
   - `active_model.json`, the pointer to the active version. Training does **not** change it: a new version is saved as an inactive candidate (see *Model lifecycle* below);
   - `outputs/metrics/<version>.json`;
   - `outputs/predictions/<version>_test_predictions.csv`;
   - figures in `outputs/graphs/`.

Training is reproducible: re-running gives identical metrics.

## 7. Evaluation and current results

Metrics (`app/ml/evaluation.py`): accuracy, balanced accuracy, macro/weighted precision, recall and F1, confusion matrix, and ROC-AUC. Binary ROC-AUC uses the positive-class probability; multiclass uses one-vs-rest macro averaging. ROC-AUC is reported as undefined when the true labels contain only one class. Results are reported **per 4 s segment** and **per trial** (segment probabilities averaged within the trial).

**Real result, model `valence_binary-random_forest-20261005-182352`, test participants s01, s09, s11 (never seen in training or selection).** The currently active retrain, `valence_binary-random_forest-20261008-180558`, reproduces these metrics exactly.

| | accuracy | balanced acc. | macro-F1 | ROC-AUC | n |
|---|---|---|---|---|---|
| Random Forest, segment | 0.548 | 0.544 | 0.543 | 0.537 | 1788 |
| Random Forest, trial | 0.592 | 0.590 | 0.590 | 0.565 | 120 |
| Majority-class reference, segment | 0.525 | 0.500 | 0.344 | 0.500 | 1788 |

Leave-one-participant-out CV (9 participants): accuracy **0.510 ± 0.086**, macro-F1 **0.416 ± 0.050**. Per-participant accuracy ranges from 0.41 to 0.70.

**EEGNet-style CNN, robust evaluation** (5 seeds plus nested leave-one-participant-out; see [docs/REVIEW3_CNN_ROBUST_EVALUATION.md](docs/REVIEW3_CNN_ROBUST_EVALUATION.md)):
- **Test participants:** segment macro-F1 **0.466 ± 0.109** over 5 seeds (range 0.322–0.554), vs the Random Forest's 0.543.
- **LOPO:** accuracy 0.503 ± 0.080, macro-F1 0.383 ± 0.065, balanced accuracy 0.494 ± 0.023.
- **Normalization:** per-participant normalization did not improve discrimination.
- **Status:** the CNN does not outperform the Random Forest and stays inactive.

On validation participants, the SVM collapsed to the majority class (macro-F1 0.298) and XGBoost reached macro-F1 0.385.

**Interpretation.** Cross-subject valence recognition from these hand-crafted features is only slightly above chance, and it varies a lot between people. This is consistent with the difficulty of subject-independent evaluation reported for DEAP. Many published DEAP accuracies far above this come from random segment-level splits, which leak information about each participant between train and test. Those numbers are not comparable to the participant-aware results here.

## 8. Database schema

PostgreSQL via SQLAlchemy 2, with Alembic migrations in `backend/alembic/versions`.

| table | key columns | relations |
|---|---|---|
| `users` | email (unique), hashed_password (bcrypt), is_admin, is_active, last_login_at | 1-n sessions, predictions |
| `sessions` | name, description (analysis sessions that group predictions; not auth sessions) | n-1 users, 1-n predictions |
| `models` | version (unique), algorithm, label_strategy, class_labels, feature_count, artifact_path, is_active, trained_at | 1-n metrics |
| `metrics` | split (validation / test / test_reference / cross_validation), level (segment / trial), name, value | n-1 models |
| `predictions` | predicted_label, confidence, probabilities (JSON), segments_used / rejected, input_source, source_reference (JSON) | n-1 users, sessions, models |
| `reports` | report_type, title, content (JSON) | n-1 users, models |
| `audit_logs` | action, resource_type / id, details, ip_address | n-1 users |

All timestamps are stored and returned in UTC.

## 9. API

Interactive OpenAPI docs are at **http://localhost:8000/api/docs** (ReDoc at `/api/redoc`). Errors always use the same shape: `{"error": {"code", "message", "details?", "request_id"}}`.

| method | path | auth | purpose |
|---|---|---|---|
| GET | `/api/health` | – | public liveness check: `status` (`ok`/`degraded`) and API `version` only |
| GET | `/api/admin/diagnostics` | admin | database connection and engine, active-model status, dataset availability, environment (never URLs, credentials or secrets) |
| POST | `/api/auth/register` | – | create account (**the first account becomes administrator**) |
| POST | `/api/auth/login` | – | OAuth2 password form (`username` = email) → JWT |
| GET | `/api/auth/me` | user | current user |
| POST | `/api/train` | admin | start training in the background (202). The result is registered as an **inactive candidate**; the active model does not change |
| GET | `/api/training/status` | admin | state, stage, progress, resulting version; a failed run reports a generic message (details are in the server log only) |
| GET | `/api/model-info` | – | active model metadata (`available: false` if none) |
| GET | `/api/models` | – | registered model versions (read-only; never writes to the database) |
| POST | `/api/models/sync` | admin | synchronise saved model versions into the database registry (the same idempotent sync that runs at startup; does not change the active model) |
| POST | `/api/models/{version}/activate` | admin | make a registered version the active model. Body: `{"confirm_version": "<same version>", "reason": "<10-500 characters>"}`. Validates integrity, feature schema and a test prediction first; 404 unknown version, 409 while training or another activation runs, 422 invalid request or model, 500 if the switch fails (the previous model stays active). Audited as `model.activate`. Activating the version that is already active writes nothing (`changed: false`) |
| GET | `/api/metrics` | – | held-out, validation, CV and reference metrics, plus graph URLs; dataset facts limited to participant count, segment and rejected-segment counts, class balance and fingerprint (no dataset location or file names) |
| GET | `/api/dataset/summary` | – | aggregate structure of the working dataset (counts, channels, sampling rate, per-participant shape); no paths, file names or rating statistics |
| GET | `/api/admin/dataset/summary` | admin | full dataset diagnostics, including location, file names, sizes and rating statistics |
| GET | `/api/dataset/demo-trials` | – | held-out participants for honest demo predictions |
| POST | `/api/predict` | user | predict and store; body: `{"deap_reference": {"participant_id": "s01", "trial_index": 0}}` **or** `{"eeg": {"signal": [[...32 channels...]], "sampling_rate": 128, "includes_baseline": false}}`, optional `session_id` |
| GET | `/api/predictions[/{id}]` | user | own prediction history |
| POST/GET | `/api/sessions` | user | create or list analysis sessions |
| GET | `/api/reports` | admin | training reports (administrator only); responses are sanitised (dataset facts limited to participant count, segment and rejected-segment counts, class balance and fingerprint; no filesystem paths or file names). Stored reports are unchanged. |
| GET | `/api/graphs/<file>` | – | evaluation figures |

**Prediction pipeline:**
1. validate the input (32 channels, 128 Hz, ≤120 s);
2. preprocess, using the *trained model's* saved configuration;
3. extract features;
4. predict per segment with the loaded model;
5. average the probabilities, taking the class order from `model.classes_`;
6. store in `predictions` and write an audit log entry;
7. return the prediction, confidence, probabilities, model version, timestamp and prediction ID.

For DEAP references, the response also includes the participant's self-reported rating and which split that participant belonged to.

## 10. Authentication

- Passwords are hashed with bcrypt (12 rounds). Plain-text passwords are never stored or logged.
- Access tokens are JWT (HS256) signed with `NEUROSENSE_JWT_SECRET_KEY` from the environment. The app refuses to start without it. Default lifetime is 60 min.
- `get_current_user` protects user routes and `get_admin_user` protects training.
- Unknown-email logins are checked against a dummy hash, which reduces user enumeration through response timing.

### Deployment and security consideration: first account becomes administrator

`POST /api/auth/register` makes the **first account registered in an empty database an administrator** (`backend/app/services/auth_service.py`, `is_admin=is_first`). Every later account is a standard user. This is intentional bootstrap behaviour and has not been changed. On a fresh deployment, though, whoever registers first gets administrator rights. These include `POST /api/train` (trains a new candidate model), `POST /api/models/{version}/activate` (changes the model that serves predictions), `POST /api/models/sync`, the training reports (`GET /api/reports`), and the detailed system and dataset diagnostics (`/api/admin/diagnostics`, `/api/admin/dataset/summary`).

Required precaution for every fresh deployment or new, empty database:

1. Before the application is reachable by anyone else (public network, shared host, tunnel or reverse proxy), register the intended administrator account yourself.
2. Verify it: sign in, call `GET /api/auth/me` and confirm `"is_admin": true` for that email address. Also confirm there are no other accounts in the `users` table.
3. Only then expose the application to users.

If the order goes wrong, an unintended account can hold administrator rights. Review the `users` table and correct `is_admin` before continuing.

**Audit log and training failures.** A failed training run is recorded in the `audit_logs` table as `training.failed`. The record holds a safe description, the stage, the progress and the exception type, and the full exception goes to the server log only. Entries written before this change may still contain raw exception text, which can include server paths. They stay server-side: no API endpoint returns audit-log entries. They have not been rewritten or deleted.

Hiding administration links in the frontend does **not** restrict access. The server-side check (`get_admin_user`) is what protects administrator routes. Any account the backend marks as administrator can call those routes directly, whatever the interface shows.

## 11. Configuration

All settings come from environment variables prefixed `NEUROSENSE_` or from `backend/.env`. See [`backend/.env.example`](backend/.env.example).

| setting | covers |
|---|---|
| `DATASET_DIR`, `OUTPUT_DIR` | data and output locations (relative paths resolve from the project root) |
| `DATABASE_URL` | database connection |
| `JWT_SECRET_KEY`, `ACCESS_TOKEN_EXPIRE_MINUTES` | authentication |
| `API_PREFIX`, `CORS_ORIGINS` | API |
| `LOG_LEVEL` | logging |
| `RANDOM_SEED` | reproducibility |
| `LABEL_STRATEGY`, `VALENCE_THRESHOLD`, `AROUSAL_THRESHOLD` | labels |
| `TEST_PARTICIPANTS`, `VALIDATION_PARTICIPANTS` | split sizes |
| `SAMPLING_RATE`, `SEGMENT_SECONDS` | signal processing |

Never commit `backend/.env`.

## 12. Running locally (Windows, Python 3.13, Node 20+)

```powershell
# backend
python -m venv .venv
.venv\Scripts\python -m pip install -r backend\requirements.txt
copy backend\.env.example backend\.env      # then set DATASET_DIR, DATABASE_URL and a JWT secret
cd backend
..\.venv\Scripts\alembic upgrade head        # create the schema
..\.venv\Scripts\python -m uvicorn app.main:create_app --factory --port 8000

# frontend (second terminal)
cd frontend
npm install
npm run dev                                  # http://localhost:5173 (proxies /api to :8000)
```

**PostgreSQL.** The project runs on a local PostgreSQL 18 server with this setup:

| setting | value |
|---|---|
| host | `localhost` |
| port | `5432` |
| user | `postgres` |
| database | `neurosense_ai` |

1. Create the database (as the `postgres` user):

   ```sql
   CREATE DATABASE neurosense_ai;
   ```

2. Set the URL in `backend/.env`:

   ```
   NEUROSENSE_DATABASE_URL=postgresql+psycopg://postgres:<password>@localhost:5432/neurosense_ai
   ```

   URL-encode special characters in the password (`@` → `%40`, `:` → `%3A`, `/` → `%2F`, `#` → `%23`, `%` → `%25`).

3. Run `alembic upgrade head`.

4. Check `GET /api/health`. It should report `"status": "ok"`. Then, signed in as the administrator, check `GET /api/admin/diagnostics`. It should report `"database": {"connected": true, "dialect": "postgresql"}`.

## 13. Tests

```powershell
cd backend
..\.venv\Scripts\python -m pytest            # 104 unit/API/ML tests incl. 18 deep-learning + 13 robust-evaluation tests (torch parts skipped without torch)
..\.venv\Scripts\python -m pytest -m integration   # 2 tests on the real DEAP files
```

The unit tests generate small synthetic files in DEAP format inside a temporary folder. That data exists only in tests; production code has no path that generates or accepts synthetic training data.

## 14. Training and predicting

```powershell
cd backend
..\.venv\Scripts\python -m scripts.dataset_summary        # verify the dataset
..\.venv\Scripts\python -m scripts.preprocessing_report   # preprocessing figures
..\.venv\Scripts\python -m scripts.train_baseline         # train + evaluate (~75 s incl. CV; add --no-cv); saved as an inactive candidate
```

You can also train through the API (`POST /api/train` as admin).

**Model lifecycle.** Training never replaces the active model. Every new version (script or API) is saved and registered as an **inactive candidate**; the active model, its pointer (`outputs/models/active_model.json`) and the predictions it serves stay as they were. A candidate becomes active only through an explicit step:

- `POST /api/models/{version}/activate` as an administrator, with the version repeated in `confirm_version` and a `reason` (10-500 characters, stored in the audit log; never include passwords, tokens or other secrets; text that looks like a credential is refused). The server checks the artifact's SHA-256, the feature schema and the class labels, and runs one neutral test prediction before switching. The pointer is written atomically (temporary file + `os.replace`); the pointer, the in-memory model, the database `is_active` flag and the audit entry change together, and any failure restores the previous state. To roll back, activate the previous version the same way.
- Or `python -m scripts.train_baseline --activate` to activate the newly trained model directly. A running API server keeps serving the model it loaded until it is restarted.

The candidate's metrics come from the same participant-aware split as before (not the frozen evaluation protocol), so activating a candidate is not evidence of better performance. There is no activation control in the web interface yet.

**Review 3 CNN** (needs `pip install -r backend/requirements-dl.txt`):

```powershell
..\.venv\Scripts\python -m scripts.train_cnn        # EEGNet-style CNN, same participant split; registered, not activated
..\.venv\Scripts\python -m scripts.compare_models   # Random Forest / SVM / XGBoost / CNN on the same test participants
..\.venv\Scripts\python -m scripts.evaluate_cnn     # robust CNN evaluation: 5 seeds + nested LOPO, 2 normalizations (resumable, ~3-4 h CPU)
```

The Random Forest remains the active model. For CNN details and results, see [docs/REVIEW3_PROGRESS_30.md](docs/REVIEW3_PROGRESS_30.md).

To predict, use the dashboard's Prediction card (a held-out participant and trial, or an uploaded EEG JSON) or `POST /api/predict`.

## 15. Current limitations

- Cross-subject accuracy is near chance (see section 7). The current model is a pipeline baseline, not a usable emotion recognizer.
- Only 12 participants, and a 3-participant test set, so the estimates have high variance.
- No per-participant calibration in the served models. Per-participant normalization was evaluated for the CNN and did not improve cross-subject discrimination.
- Uploaded EEG must already be in DEAP layout: 32 channels in Geneva order, 128 Hz, microvolts. There is no resampling or montage mapping.
- Training state is held in memory, with one job at a time. A restart loses in-progress status (completed models persist).
- `pickle`/`joblib` artifacts are trusted only because the registry writes them and verifies their SHA-256.
- The JWT is stored in `sessionStorage` in the browser. That is acceptable for a prototype, but not hardened against XSS.
- The application runs on PostgreSQL, but the automated tests use isolated temporary SQLite databases. PostgreSQL is checked through migrations and the live health and administrator diagnostics endpoints, not by the test suite.

## 16. Future work (Review 3)

- **Deep models:**
  - done (Review 3, first phase): EEGNet-style CNN on z-scored segments; see `docs/REVIEW3_PROGRESS_30.md`;
  - next: CNN-LSTM and LSTM on segment sequences, then a Transformer over channel × time tokens.
- **Better evaluation:**
  - done: repeated seeds, nested LOPO, trial-cluster bootstrap CIs, within-participant permutation tests and Wilcoxon tests for the CNN (`scripts/evaluate_cnn.py`);
  - domain adaptation;
  - full 32-participant DEAP;
  - within-subject results reported separately.
- **Explainable AI:** permutation importance and SHAP on band-power features, plus topographic maps of feature contributions.
- **Product:**
  - arousal and quadrant targets in the UI;
  - EDF/CSV upload with resampling;
  - a complete dashboard and deployment.
