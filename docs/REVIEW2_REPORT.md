# NeuroSense AI: Review 2 Technical Progress Report

*Emotion Recognition from EEG Brain Signals using Deep Learning and Explainable AI.* This is an academic research prototype, not a diagnostic system.

## 1. Starting point (before this phase, tag `review1-baseline`)

**What existed:**
- a FastAPI app with 5 endpoints;
- a single-channel preprocessor;
- a trainer that used **random noise with random labels**;
- unconnected CNN/LSTM and evaluation modules;
- a React page that called non-proxied paths.

**Known defects:**
- an invalid `roc_auc_score` argument that crashed `/train`;
- XGBoost failing on string labels;
- a hard-coded probability label order;
- NaN entropy from negative amplitudes;
- `np.trapz` no longer existing in numpy 2.x;
- a corrupted notebook;
- unused configuration;
- no database, authentication or tests beyond one;
- no git history.

## 2. Implemented in this phase

| area | result |
|---|---|
| Foundation | git history (one commit per phase), env-driven settings (`.env.example`), JSON logging, `backend/app` layered package |
| Data | safe DEAP loader with structural validation and corrupt-file reporting; dataset summary script |
| Labels | binary valence (default), binary arousal, four quadrants; configurable thresholds, documented as operational labels |
| Preprocessing | 32-channel EEG: baseline removal, 4–45 Hz zero-phase filter, robust artifact clipping, 4 s segments, segment rejection |
| Features | 440 per segment: band powers, relative powers, spectral entropy, Hjorth, hemispheric asymmetry |
| Evaluation design | participant-aware train/validation/test split (seed 42), segment and trial metrics, leave-one-participant-out CV, majority-class reference |
| Models | majority reference, Random Forest, calibrated SVM, XGBoost; selection on validation; registry with metadata and SHA-256 |
| Database | 7 tables (users, sessions, models, metrics, predictions, reports, audit_logs); Alembic migration |
| Auth | registration, bcrypt hashing, JWT login, current-user and admin dependencies |
| API | 16 endpoints under `/api`; Pydantic schemas; consistent error JSON; OpenAPI at `/api/docs` |
| Frontend | existing React app wired to the real API; shows "Not available" instead of invented values |
| Tests | 73 unit/API tests and 2 real-data integration tests, all passing |

## 3. What works end to end

The full chain runs on real data:

1. 12 DEAP files are loaded and validated.
2. 480 trials become 7,200 segments, of which 154 are rejected for artifacts, leaving 7,046 × 440 features.
3. Participants are split into train / validation / test.
4. Candidate models are trained and compared, and the best is selected.
5. The selected model is evaluated on held-out participants.
6. The model and its metadata are saved.
7. The API loads the model at startup.
8. An authenticated `POST /api/predict` stores the result in the database.
9. The history is visible in the dashboard.

Training can also be started from the API or the UI. The new model is hot-reloaded and a training report is stored. This was verified over HTTP and in a headless browser (`docs/screenshots/`).

## 4. Real model results

Binary valence (self-reported valence > 5), model `valence_binary-random_forest-20261005-182352`.

**Split:**
- train: s02, s03, s05, s07, s08, s15, s22
- validation: s16, s25
- test: s01, s09, s11

| held-out test participants | accuracy | balanced acc. | macro-F1 | ROC-AUC | n |
|---|---|---|---|---|---|
| Random Forest, per 4 s segment | 0.548 | 0.544 | 0.543 | 0.537 | 1788 |
| Random Forest, per trial | 0.592 | 0.590 | 0.590 | 0.565 | 120 |
| Majority-class reference | 0.525 | 0.500 | 0.344 | 0.500 | 1788 |

| validation participants (model selection) | macro-F1 | ROC-AUC |
|---|---|---|
| Random Forest (selected) | 0.420 | 0.536 |
| XGBoost | 0.385 | 0.622 |
| SVM (RBF, calibrated) | 0.298 (always predicts the majority class) | 0.488 |
| Majority reference | 0.298 | 0.500 |

Leave-one-participant-out CV (9 participants): accuracy **0.510 ± 0.086**, macro-F1 0.416 ± 0.050. Per-participant accuracy ranges from 0.41 to 0.70.

**Honest reading.** The pipeline is correct and reproducible, but cross-subject valence recognition with these features is close to chance and varies strongly between people. This is the expected difficulty of subject-independent DEAP evaluation. Higher published figures usually come from random segment splits that leak participant identity, and they are not comparable.

Figures are in `docs/results/` (confusion matrices, ROC, model comparison, preprocessing examples, feature distributions). The full metrics are in `docs/results/<version>.json`.

## 5. Backend APIs

| | endpoints |
|---|---|
| Health | `GET /api/health` |
| Auth | `POST /api/auth/register`, `POST /api/auth/login`, `GET /api/auth/me` |
| Training | `POST /api/train` (admin), `GET /api/training/status` |
| Models and metrics | `GET /api/model-info`, `GET /api/models`, `GET /api/metrics` |
| Dataset | `GET /api/dataset/summary`, `GET /api/dataset/demo-trials` |
| Prediction | `POST /api/predict` |
| History | `GET /api/predictions`, `GET /api/predictions/{id}`, `GET`/`POST /api/sessions` |
| Reports | `GET /api/reports` |
| Figures | `GET /api/graphs/<file>` |

## 6. Database entities

| table | holds |
|---|---|
| `users` | accounts |
| `sessions` | analysis sessions that group predictions |
| `models` | registered model versions, with the active flag |
| `metrics` | one row per model, split, level and metric |
| `predictions` | stored prediction results |
| `reports` | JSON training reports |
| `audit_logs` | register, login, failed login, prediction and training events |

Foreign keys link predictions to users, sessions and models; metrics and reports to models; and audit logs to users. Migration `98de36d96d48` was tested with upgrade, downgrade and upgrade again, and `alembic check` reports no drift.

**PostgreSQL (verified).**
- The backend runs on local PostgreSQL 18 (`postgres@localhost:5432/neurosense_ai`, psycopg 3 driver).
- `alembic upgrade head` created all 7 tables plus `alembic_version`, and `alembic check` reports no drift.
- The running API's `/api/health` reports `{"connected": true, "dialect": "postgresql"}`.
- At startup the backend registers the active model and its 46 metric rows in PostgreSQL.
- The automated tests use isolated temporary SQLite databases by design.

## 7. Authentication

- bcrypt password hashes.
- HS256 JWTs signed with a secret from the environment.
- The first registered account becomes administrator, and only administrators can train.
- Users can only see their own predictions and sessions; this is tested.
- Passwords and tokens never appear in logs; this was checked against the log file.

## 8. Tests

`pytest`: **73 passed**. `pytest -m integration` on the real DEAP files: **2 passed**.

**Coverage:**
- loader, including a block on malicious pickles;
- preprocessing;
- features, labels and splitting, including leakage checks;
- models and evaluation;
- registry tamper detection;
- training;
- repositories;
- authentication;
- prediction, metrics, history and reports API.

**Regression tests for earlier bugs:**
- the AUC arguments;
- XGBoost string labels;
- class order;
- NaN entropy;
- timezone-aware timestamps.

## 9. Demo commands

```powershell
cd backend
..\.venv\Scripts\python -m scripts.dataset_summary
..\.venv\Scripts\python -m scripts.preprocessing_report
..\.venv\Scripts\python -m scripts.train_baseline
..\.venv\Scripts\python -m pytest ; ..\.venv\Scripts\python -m pytest -m integration
..\.venv\Scripts\python -m uvicorn app.main:create_app --factory --port 8000   # then open /api/docs
cd ..\frontend ; npm run dev                                                    # http://localhost:5173
```

Screenshots: `docs/screenshots/1_signed_out.png`, `docs/screenshots/2_signed_in_prediction.png`.

## 10. Acceptance criteria status

| criterion | status |
|---|---|
| Real DEAP files discovered and loaded; labels extracted | ✅ |
| Preprocessing on real EEG; participant-aware split | ✅ |
| Baseline trained on real data; held-out test metrics | ✅ |
| Model and metadata saved | ✅ |
| FastAPI starts; OpenAPI docs work | ✅ |
| PostgreSQL connection works | ✅ Local PostgreSQL 18 (`neurosense_ai`); `/api/health` reports dialect `postgresql` |
| Alembic migrations work | ✅ `alembic upgrade head` run against PostgreSQL; all 7 tables created; `alembic check` shows no drift |
| Registration, login, JWT protection | ✅ |
| Prediction API, metrics API, stored history | ✅ |
| Frontend consumes real APIs; no fake values | ✅ |
| Tests pass | ✅ |
| No random data or labels in production training | ✅ The demo-data path was removed. |
| Configuration not hard-coded; README; API docs | ✅ |

## 11. Remaining work for Review 3

1. Per-participant feature normalization, all 32 DEAP participants, and permutation tests against chance.
2. CNN, LSTM and Transformer models on segments, evaluated with the same participant-aware protocol so the comparison with the baseline is fair.
3. Explainable AI: permutation importance and SHAP on band-power features, plus scalp topographies.
4. Complete dashboard (arousal and quadrant targets, EDF/CSV upload with resampling) and deployment.

**Suggested next milestone.** A deep-learning model that beats the 0.51 ± 0.09 leave-one-participant-out baseline under the same protocol, with XAI explanations shown in the dashboard.
