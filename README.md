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
          └─ ModelService keeps the active model in memory (loaded once, reloaded after training)
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
      deep/                 CNN/LSTM prototypes (not used yet, see Future work)
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
   - `active_model.json`;
   - `outputs/metrics/<version>.json`;
   - `outputs/predictions/<version>_test_predictions.csv`;
   - figures in `outputs/graphs/`.

Training is reproducible: re-running gives identical metrics.

## 7. Evaluation and current results

Metrics (`app/ml/evaluation.py`): accuracy, balanced accuracy, macro/weighted precision, recall and F1, confusion matrix, and ROC-AUC. Binary ROC-AUC uses the positive-class probability; multiclass uses one-vs-rest macro averaging. ROC-AUC is reported as undefined when the true labels contain only one class. Results are reported **per 4 s segment** and **per trial** (segment probabilities averaged within the trial).

**Real result, model `valence_binary-random_forest-20261005-182352`, test participants s01, s09, s11 (never seen in training or selection):**

| | accuracy | balanced acc. | macro-F1 | ROC-AUC | n |
|---|---|---|---|---|---|
| Random Forest, segment | 0.548 | 0.544 | 0.543 | 0.537 | 1788 |
| Random Forest, trial | 0.592 | 0.590 | 0.590 | 0.565 | 120 |
| Majority-class reference, segment | 0.525 | 0.500 | 0.344 | 0.500 | 1788 |

Leave-one-participant-out CV (9 participants): accuracy **0.510 ± 0.086**, macro-F1 **0.416 ± 0.050**. Per-participant accuracy ranges from 0.41 to 0.70.

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
| GET | `/api/health` | – | backend, database, model and dataset status |
| POST | `/api/auth/register` | – | create account (**the first account becomes administrator**) |
| POST | `/api/auth/login` | – | OAuth2 password form (`username` = email) → JWT |
| GET | `/api/auth/me` | user | current user |
| POST | `/api/train` | admin | start training in the background (202) |
| GET | `/api/training/status` | – | state, stage, progress, resulting version |
| GET | `/api/model-info` | – | active model metadata (`available: false` if none) |
| GET | `/api/models` | – | registered model versions |
| GET | `/api/metrics` | – | held-out, validation, CV and reference metrics, plus graph URLs |
| GET | `/api/dataset/summary` | – | structure of the local DEAP files |
| GET | `/api/dataset/demo-trials` | – | held-out participants for honest demo predictions |
| POST | `/api/predict` | user | predict and store; body: `{"deap_reference": {"participant_id": "s01", "trial_index": 0}}` **or** `{"eeg": {"signal": [[...32 channels...]], "sampling_rate": 128, "includes_baseline": false}}`, optional `session_id` |
| GET | `/api/predictions[/{id}]` | user | own prediction history |
| POST/GET | `/api/sessions` | user | create or list analysis sessions |
| GET | `/api/reports` | user | training reports |
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

4. Check `GET /api/health`. It should report `"database": {"connected": true, "dialect": "postgresql"}`.

## 13. Tests

```powershell
cd backend
..\.venv\Scripts\python -m pytest            # 73 unit/API tests (synthetic DEAP-format fixtures, ~30 s)
..\.venv\Scripts\python -m pytest -m integration   # 2 tests on the real DEAP files
```

The unit tests generate small synthetic files in DEAP format inside a temporary folder. That data exists only in tests; production code has no path that generates or accepts synthetic training data.

## 14. Training and predicting

```powershell
cd backend
..\.venv\Scripts\python -m scripts.dataset_summary        # verify the dataset
..\.venv\Scripts\python -m scripts.preprocessing_report   # preprocessing figures
..\.venv\Scripts\python -m scripts.train_baseline         # train + evaluate (~75 s incl. CV; add --no-cv)
```

You can also train through the API (`POST /api/train` as admin) or with the dashboard's **Train on DEAP** button.

To predict, use the dashboard's Prediction card (a held-out participant and trial, or an uploaded EEG JSON) or `POST /api/predict`.

## 15. Current limitations

- Cross-subject accuracy is near chance (see section 7). The current model is a pipeline baseline, not a usable emotion recognizer.
- Only 12 participants, and a 3-participant test set, so the estimates have high variance.
- No per-participant calibration or feature normalization yet.
- Uploaded EEG must already be in DEAP layout: 32 channels in Geneva order, 128 Hz, microvolts. There is no resampling or montage mapping.
- Training state is held in memory, with one job at a time. A restart loses in-progress status (completed models persist).
- `pickle`/`joblib` artifacts are trusted only because the registry writes them and verifies their SHA-256.
- The JWT is stored in `sessionStorage` in the browser. That is acceptable for a prototype, but not hardened against XSS.
- The application runs on PostgreSQL, but the automated tests use isolated temporary SQLite databases. PostgreSQL is checked through migrations and the live `/api/health` check, not by the test suite.

## 16. Future work (Review 3)

- **Deep models:**
  - CNN on raw or z-scored segments, and LSTM on segment sequences;
  - a Transformer over channel × time tokens;
  - `app/ml/deep/` currently holds placeholder MLP/LSTM modules (`torch` is in `requirements-dl.txt`).
- **Better evaluation:**
  - per-participant feature normalization and domain adaptation;
  - full 32-participant DEAP;
  - permutation tests against chance;
  - within-subject results reported separately.
- **Explainable AI:** permutation importance and SHAP on band-power features, plus topographic maps of feature contributions.
- **Product:**
  - arousal and quadrant targets in the UI;
  - EDF/CSV upload with resampling;
  - a complete dashboard and deployment.
