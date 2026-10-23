# NeuroSense AI: Final Completion Report

| | |
|---|---|
| Date | 2026-10-20; final delivery 2026-10-23 (§14) |
| Starting point | branch `review3/deep-learning-foundation`, commit `8af920f` (same as `main` on GitHub), clean working tree |
| State of this work | §1–§13 describe the completion phase; **§14 is the final delivery.** The work was committed and pushed (no force push) after the final audit. Nothing was deployed. |
| Scope | Everything in the completion brief that could be done safely without the pending data-access decision (Decision I) |

## 1. What was completed

| Area | Result |
|---|---|
| Security: administrator bootstrap | Public registration always creates standard accounts, and role fields in signup are rejected. A local `create_admin` command creates the first administrator: it refuses if one exists, never promotes existing accounts, and has an audited password-recovery mode. |
| Protocol-aware training | New `ProtocolTrainer` (Evaluation Protocol v1.1): development participants only, frozen-participant guard, fixed Random Forest, exploratory leave-one-participant-out with trial-level per-participant metrics and diagnostics, split and trial-group tracking, inactive by default. |
| Explainability | Global feature importance for the active Random Forest (API and UI), with schema-alignment checks, an honest unavailable state and stated limitations. No per-prediction explanation is claimed. |
| Frontend/backend integration | Protocol candidates handled in the registry, metrics, Model & Evidence and activation dialog. The Analyze page explains when there is no held-out participant, and offers a clearly labelled synthetic test file. Training text updated. |
| Tests | New backend tests (auth/bootstrap, protocol training, explainability), the first frontend unit tests, and updated browser test suites. |
| CI | `.github/workflows/ci.yml`: backend, deep-learning and frontend jobs; migration check; no dataset; no artifacts; read-only permissions. |
| Dependency fix | `react-chartjs-2` 4 → 5. Version 4 declares Chart.js 3 as a peer, which made a clean `npm ci` fail with Chart.js 4. |
| Local operations | `scripts/start.ps1`, `stop.ps1`, `status.ps1`: prerequisite checks, loopback only, stop only what was started. |
| Documentation | README rewritten for setup, run, test, CI, troubleshooting, data boundaries, the candidate/active distinction and explainability. New `docs/PROJECT_REPORT.md`; this report. |

## 2. Important files

**New**
- `backend/app/ml/participants.py`: protocol roles (12 trusted development participants, 8 frozen).
- `backend/app/ml/protocol_training.py`: protocol-aware candidate training.
- `backend/app/services/explainability.py`: global feature importance.
- `backend/scripts/create_admin.py`: administrator bootstrap and recovery.
- `backend/scripts/train_candidate.py`: command-line protocol training.
- `backend/tests/test_auth_bootstrap.py`, `test_protocol_training.py`, `test_explainability.py`.
- `frontend/src/components/GlobalImportance.jsx`, `frontend/src/activationRules.js`, `frontend/src/synthetic.js`, `frontend/tests/logic.test.mjs`.
- `.github/workflows/ci.yml`, `scripts/start.ps1`, `scripts/stop.ps1`, `scripts/status.ps1`.
- `docs/PROJECT_REPORT.md`, `docs/FINAL_COMPLETION_REPORT.md`.

**Changed (main ones)**
- `backend/app/services/auth_service.py`: no first-account administrator; bootstrap and recovery services.
- `backend/app/schemas/schemas.py`: signup rejects unknown fields; schemas for importance and protocol evaluation.
- `backend/app/ml/dataset.py`, `backend/app/core/config.py`: frozen-participant guard; `DEVELOPMENT_PARTICIPANTS` setting.
- `backend/app/services/model_service.py`: `metric_rows`, which keeps the exact historical rows and adds protocol summary rows.
- `backend/app/services/training_service.py`: the API trains with `ProtocolTrainer`.
- `backend/app/api/routes/ml.py`: `GET /api/explain/global`; protocol evaluation in `/models` and `/metrics`.
- `backend/app/services/public_metadata.py`: public views of the protocol evaluation (no paths).
- `backend/scripts/train_baseline.py`: labelled as the historical pre-protocol reproduction.
- `frontend/src/pages/*`, `frontend/src/components/*`, `frontend/package.json`: integration and UI changes described above.
- `README.md`, `.gitignore` (`.run/`), `backend/.env.example` (documented protocol setting).

No database migration was needed: `alembic check` reports the models and migrations in sync.

## 3. Security fixes

1. **First-account administrator removed.** `POST /api/auth/register` creates `is_admin=False` for every account, including the first.
2. **No role escalation through signup.** Payloads containing `is_admin`, `role` or any other unknown field are rejected with 422, and no account is created.
3. **Explicit, safe bootstrap.**
   - `python -m scripts.create_admin` refuses when an administrator exists and never promotes an existing account.
   - The password is read without echo and never printed.
   - Database errors are reported without connection details.
   - Recovery (`--reset-password`) works only for an existing administrator and is audited.
4. **Frozen participants cannot be loaded** by development code: loader guard plus settings validation.
5. **Existing hardening re-verified live:**
   - every administrator endpoint returns 401 without a valid token, and activation checks authorization before reading the body;
   - public responses contain no paths, file names or account names.

Existing accounts in your database were **not** changed. Your database still has its one account, which keeps the administrator role it already had.

## 4. Training and evaluation-protocol changes

- **Data:** the API **Train candidate** button and `scripts.train_candidate` use only the development participants. These default to the 12 trusted participants; frozen IDs are refused in code.
- **Model:** a fixed Random Forest (no tuning or selection).
- **Evaluation:** exploratory LOPO.
  - Whole participants per fold; every fold asserts that participants and trial groups are disjoint.
  - Scaling is fitted inside the fold.
  - Per-participant trial-level balanced accuracy, macro-F1 and within-person AUC; a majority reference from the fold's training participants.
  - Predicted-class share, single-label and few-trials diagnostics.
- **Metadata:** split strategy, participants, trial groups, seed, leakage controls and the limitations text. Results are labelled "exploratory (development participants only, Protocol v1.1)", never as an independent test.
- **Inactive by default.** Training never touches the active pointer; tests prove the pointer is byte-identical afterwards.
- **Fails safely:** fewer than 3 development participants gives a clear error and saves nothing.
- **Historical trainer:** still available for reproducing the historical fixed split, and labelled as such.
- **No real training was run.** Protocol training was exercised only on synthetic DEAP-format data in temporary directories.

## 5. Explainability actually implemented

`GET /api/explain/global?top=N` (1–100), and the **Global feature importance** card on Model & Evidence.

- **What it shows:** impurity-based importance read from the active Random Forest. It covers the top features, totals by band and by feature type, and the model version and tree count.
- **When it is produced:** only when the model is a scikit-learn pipeline ending in a `RandomForestClassifier`, with exactly one value per named feature matching the recorded schema. Otherwise it reports `available: false` with the reason (CNN, other algorithm, schema mismatch, no model).
- **Verified on the real active model** (read-only artifact load): 440 features, 300 trees. Gamma-band features carry about 31 % of total importance, so the UI warns that gamma features are prone to muscle/EMG contamination.
- **Not implemented, deliberately:** per-prediction explanations. None is shown, and the UI states that global importance does not explain a prediction or reveal what a person felt.

## 6. Tests: exact commands and results

| # | Command (directory) | Kind | Result |
|---|---|---|---|
| 1 | `..\.venv\Scripts\python -m pytest -q -rs` (`backend`) | Unit + API + ML, synthetic data, temporary SQLite | **224 passed**, 0 failed, 0 skipped, 2 deselected (integration) |
| 2 | `..\.venv\Scripts\python -m pytest -m integration` (`backend`) | Read-only tests on the real DEAP files | **2 passed** |
| 3 | `npm test` (`frontend`) | Frontend unit tests | **8 passed**, 0 failed |
| 4 | `npm run build` (`frontend`) | Production build | **OK** |
| 5 | Stage D browser suite (headless Edge, isolated backend: scratch SQLite, temporary registry, synthetic data, test-only models) | Browser end-to-end | **61 / 61 passed** |
| 6 | Earlier UI regression suite (isolated backend, scratch SQLite, scratch copy of the registry) | Browser end-to-end | **144 / 144 passed** |
| 7 | `alembic upgrade head` + `alembic check` on a temporary SQLite database | Migrations | **OK**, no drift |
| 8 | Clean-copy `npm ci` + `npm test` + `npm run build` (simulates the CI frontend job) | CI validation | **OK** (failed before the dependency fix) |
| 9 | YAML parse of `ci.yml` (jobs, steps, permissions, no artifact uploads) | CI validation | **OK** |
| 10 | Live read-only smoke on the real local app (public GETs; unauthenticated admin calls) | Real local endpoints | **21 / 21 passed**, plus `/api/explain/global` via the frontend proxy |
| 11 | Signed-out browser smoke of the real app (desktop and 375 px; non-GET requests blocked) | Real local UI | **51 / 51 passed** |

**Failures found and fixed during this phase:**
- **My own changes broke three existing tests:**
  - an error-message wording change (reverted);
  - a lifecycle test that trained without the evaluation and then expected metric rows (now runs with the evaluation on);
  - an exact-shape `/metrics` test (extended for the two new optional fields).
- **Pre-existing `npm ci` failure:** caused by the peer-dependency conflict; fixed by the upgrade.

The browser suites live in the session's scratch area and are not part of the repository. The repository's automated tests are #1–#4, and CI runs #1, #3, #4 and #7.

## 7. CI

- **File:** `.github/workflows/ci.yml`, on push and pull request, `contents: read`.
- **Jobs:**
  - `backend`: pip install; a random throwaway JWT secret per run; `alembic upgrade head` and `alembic check` on temporary SQLite; `pytest -rs`; an explicit notice that integration tests are not run.
  - `backend-deep-learning`: CPU-only torch; CNN and robust-evaluation tests.
  - `frontend`: `npm ci`, `npm test`, `npm run build`.
- **Data safety:** no PostgreSQL, no DEAP data, no artifact uploads, no deployment.
- **Not run on GitHub yet.** That needs a push, which this phase was not allowed to do. Locally validated: YAML structure, the migration commands, the full backend suite, and a clean-install run of the frontend job.
- **Unverified locally:** the `backend-deep-learning` job's CPU wheel download (`torch==2.14.1` from the PyTorch CPU index). If that exact version is missing there, that job will fail visibly and the pin or index needs adjusting.

## 8. Local startup

From the project root:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\start.ps1    # checks prerequisites, starts backend + frontend, waits for health
powershell -ExecutionPolicy Bypass -File scripts\status.ps1   # ports, 127.0.0.1-only binding, health, active model
powershell -ExecutionPolicy Bypass -File scripts\stop.ps1     # stops only what start.ps1 started
```

| Address | |
|---|---|
| http://127.0.0.1:5173 | app |
| http://127.0.0.1:8000/api/docs | Swagger |
| http://127.0.0.1:8000/api/health | health |

- **Verified:** `start.ps1` found the database reachable, both servers became healthy, and `status.ps1` reported both listening on **127.0.0.1 only**, serving `valence_binary-random_forest-20261008-180558`.
- **Current state:** **the app was left running** (started by `start.ps1`). Stop it with `scripts\stop.ps1`.

## 9. Real PostgreSQL data and model registry

Read-only snapshots (PostgreSQL `READ ONLY`, `REPEATABLE READ` transactions; counts and checksums only) were taken before any change in this phase and again after all testing, including starting the real app.

| Check | Before | After |
|---|---|---|
| models / metrics rows | 6 / 265 | 6 / 265 (all columns identical) |
| model+metric checksum | `8109a7517d6cbb9d` | `8109a7517d6cbb9d` |
| users, predictions, reports, audit_logs, sessions | 1, 13, 4, 27, 0 | identical rows and checksums |
| active model (database and pointer) | `valence_binary-random_forest-20261008-180558` | unchanged |
| model registry files (16) digest | `cb15c834a0caedb2` | `cb15c834a0caedb2` |
| write transaction during the checks | none | none |

Before starting the real backend, a read-only comparison confirmed that all 265 stored metric rows exactly match what the current code would write. The startup sync therefore stayed a no-op, and the after-snapshot confirms it.

The live checks included one rejected signup request carrying `is_admin`. It failed validation (422) and created nothing; the unchanged `users` and `audit_logs` checksums confirm this.

**Real PostgreSQL data was left unchanged.**

## 10. Skipped, not run, or limited

- **No skipped tests** in the local backend run (torch is installed locally). In CI's `backend` job the torch tests skip with a stated reason, and they run in the separate deep-learning job.
- **Not verified against the real database:** authenticated administrator workflows (signing in, training a real candidate, activating a model). They write audit records or change models, which the safety rules forbid. They are verified only in isolated environments (#1, #5, #6).
- **No real training on DEAP** in this phase, by rule.
- **No browser test suite in the repository.** The browser suites ran from scratch harnesses outside it.

## 11. Blocked items and the shortest next step

| Item | Status | Why | Shortest next step |
|---|---|---|---|
| Final independent evaluation on the frozen group (s12, s17, s18, s20, s21, s23, s29, s30) | **BLOCKED** | Decision I (DEAP data access and storage) is pending. The only copy of those participants is an unauthorized third-party upload with altered labels. | The team decides who signs the DEAP EULA, and that person requests the official `data_preprocessed_python` release through the official channel. Then: store it at an agreed path, verify the trusted 12 files against it (SHA-256, shapes, ratings) as in protocol §6.4, run the 5-seed protocol comparison on development data, and **once** run a logged final-evaluation command on the frozen group (protocol §6.3, §10). |
| DEAP-trial waveform previews in the UI | **BLOCKED** (deferred) | Serving raw licensed EEG depends on Decision I. | Revisit after Decision I; uploads and the synthetic file preview work now. |
| CI run on GitHub | **NOT RUN** | Pushing was out of scope for this phase. | Commit and push when you choose; check the Actions tab, especially the deep-learning job's torch download. |
| Live administrator workflow on the real database | **NOT RUN** by rule | It would write real audit/model records. | When you want it: sign in with your existing administrator account (no bootstrap needed), train a candidate on the Training page, review it in the Model registry, and decide whether to activate it. Activation is reversible by activating the previous version. |

## 12. Final checklist

| Requirement | Status |
|---|---|
| Audit of branch, commit, tree and code | PASS |
| Registration never creates administrators; role fields rejected | PASS |
| Safe bootstrap command (refuses a second admin, never promotes) and recovery | PASS |
| Administrator checks enforced on the backend (401/403 tests, live 401 checks) | PASS |
| Protocol-aware, group-safe, inactive-by-default candidate training | PASS |
| Frozen participants unusable by development code | PASS |
| Training never changes the active pointer (tested) | PASS |
| Global feature importance with honest unavailable states | PASS |
| No fabricated per-prediction explanations | PASS |
| Frontend/backend contracts aligned; pages build and render | PASS |
| Synthetic sample clearly labelled; no licensed data in repo, tests or CI | PASS |
| Backend suite | PASS (224/224) |
| Integration tests (real data, read-only) | PASS (2/2) |
| Frontend unit tests and build | PASS (8/8, build OK) |
| Browser end-to-end (isolated) | PASS (61/61, 144/144) |
| CI workflow added and validated locally | PASS (not yet run on GitHub) |
| Start/stop/status scripts; loopback-only verified | PASS |
| README, project report and completion report | PASS |
| Real PostgreSQL data, active model and registry unchanged | PASS |
| No secrets, datasets, databases or model files tracked | PASS |
| Final independent (frozen-group) evaluation | BLOCKED (Decision I) |
| DEAP waveform previews | BLOCKED (Decision I) |
| Live administrator workflows on the real database | NOT RUN (safety rule; verified in isolation) |
| Committed or pushed | NOT DONE (by instruction) |

## 13. Follow-up: UI and data-flow corrections (2026-10-20)

**Why Analyze EEG offered only s01, s09 and s11.** `GET /dataset/demo-trials` returned the active model's recorded `split.test`. The active Random Forest (`valence_binary-random_forest-20261008-180558`) records `final_fit: "train + validation participants"`. It was fitted on s02, s03, s05, s07, s08, s15, s16 and s22, and on s25, which was also used with s16 to choose the algorithm. The trainer refits on exactly `train | validation`. All 12 trusted files are readable with 40 trials each, so the limit was by design and correct. The "not used to train the active model" statement was accurate. Since Protocol v1.1, however, these three participants are development participants and their test results have been reported, so they are no longer an untouched test set. The UI now says so.

**What changed**
- **Backend:**
  - `training_provenance()` reports a model's fit, selection and held-out participants only from its recorded `final_fit`; anything else is `recorded: false`. It is returned by `/model-info` and `/models`.
  - `/dataset/demo-trials` now offers only participants that are recorded as held out, authorized, never frozen, and readable. It returns their real trial indices and a note when none can be offered.
  - `/dataset/summary` returns the authorized participant list.
  - Uploads can carry `synthetic_test_signal`.
  - `GET /predictions?session_id=` now counts only that session.
- **Frontend:**
  - Analyze EEG is now one flow: choose the input (DEAP, your file or synthetic test), then run it, then read the result.
  - Trials are chosen from a list labelled "Trial n (index n-1)".
  - Uploads are checked in the browser against the model's own requirements before anything is sent.
  - Synthetic results are labelled everywhere.
  - Navigation is primary (Overview, Analyze EEG, History), then a collapsible Research tools group, then Administration (administrators only).
  - Overview, Model & Evidence, Registry and Dataset read provenance and the authorized list from the API; failed requests show errors instead of fallback values.

**Tests:**

| Suite | Result |
|---|---|
| Backend | 239 passed, 2 deselected; new: `test_demo_trials_provenance.py`, `test_history_consistency.py` |
| Frontend unit | 22/22 (`tests/analysis.test.mjs`) |
| Production build | OK |
| Browser, on isolated SQLite (scratch harnesses outside the repo): workflow | 61/61 |
| Browser: model lifecycle | 61/61 |
| Browser: regression (ported to the new UI) | 147/147 |
| Live signed-out, GET-only smoke | 6/6 |

**Database and registry:** unchanged by this work; read-only snapshots before and after the backend restart are identical. The only database changes since the earlier baseline are one sign-in and one analysis at 21:18 that day, made through the live app before this work began.

## 14. Final delivery (2026-10-23)

### 14.1 Scope of the final phase

| Area | Result |
|---|---|
| Authentication | **Standalone sign-in.** A signed-out visitor sees only the sign-in card, with no shell, navigation or model data. Return-to-page, create account and session-expiry notice all work, and an already signed-in user is not asked again. User and administrator endpoints refuse requests without a token (401) or without the administrator role (403). |
| Analyze EEG selector | **All 12 trusted development participants**, with their real trial indices: s01, s02, s03, s05, s07, s08, s09, s11, s15, s16, s22, s25. Each carries its recorded role for the active model (`fit`, `selection`, `held_out`). Frozen participants are never listed, and analysing one returns 404. Training participants are labelled *in-sample demonstration*; s01, s09 and s11 are labelled *held out, but not an untouched test*. |
| Overview | Heading, one primary action, accepted data, recent analyses, the near-chance limitation and a small service indicator. Model internals now live only on Model & Evidence and Model registry. |
| Model experiments | Model-improvement study and covariance-alignment study (`MODEL_IMPROVEMENT_EXPERIMENTS.md`, `COVARIANCE_ALIGNMENT_EXPERIMENT.md`). **Both are exploratory, and neither improved cross-person performance.** Model experimentation on the 12 participants is now **frozen**. |
| Documentation | Research findings §9a–§9c (including why 59.2% trial accuracy and 49.3% balanced accuracy are not comparable), the project report, the README results section, and this report. **The evaluation protocol is unchanged.** |

### 14.2 Research conclusions carried into the delivery

- **Cross-person generalization remains near chance.**
  - Nested model selection: 0.541 [0.504, 0.579].
  - Strict leave-person-and-video-out: 0.493 baseline; 0.499 with covariance alignment.
- **Personalized calibration is the only setting above chance, and it is exploratory:** 0.618 [0.548, 0.688] with 32 of a person's own rated videos, about 0.60 with 16.
- **No 90% result exists under any valid protocol.** No clinical usefulness is claimed.
- **The active Random Forest is unchanged.** No experimental candidate was registered or activated. The one isolated candidate artifact lives in the git-ignored `outputs/experiments/` directory.

### 14.3 Final verification

| # | Check | Result |
|---|---|---|
| 1 | `python -m pytest -q` (backend; synthetic data, temporary SQLite) | **241 passed** |
| 2 | `npm test` (frontend unit tests) | **36 passed**, 0 failed |
| 3 | `npx vite build` (production build) | **OK** |
| 4 | Browser acceptance suite (headless Edge, isolated SQLite backend + copied registry; desktop and 375 px) | **37 / 37**: standalone sign-in on every route, auth errors, deep links, session expiry, 401/403, exactly 12 participants, no frozen ID, frozen 404, role notices, Overview, mobile |
| 5 | Browser workflow suite (same isolated stack) | **82 / 82** |
| 6 | Live app on 127.0.0.1:5173, signed out (GET-only) | **5 / 5**; the live `demo-trials` response lists the 12 participants with 40 trials each |
| 7 | `git diff --check` | Clean (line-ending notices only) |
| 8 | Production model files, metrics, predictions, graphs, protocol, `participants.py` (SHA-256) | **70 files identical** before and after all experiments and checks |
| 9 | Active model pointer | `valence_binary-random_forest-20261008-180558`, unchanged |
| 10 | Real PostgreSQL (read-only snapshots) | Users, predictions, reports, audit logs, sessions, models and metrics **unchanged**; no write transaction |
| 11 | Secret and data audit of everything committed | No secrets, `.env`, datasets, ratings, databases, caches, models or local paths. The 441 MB covariance cache and all `outputs/experiments/` stay git-ignored. |

The browser suites are scratch harnesses outside the repository. The repository's automated checks are #1–#3, and CI runs them.

### 14.4 Remaining blockers

| Item | Status |
|---|---|
| Final independent evaluation on the frozen group (s12, s17, s18, s20, s21, s23, s29, s30) | **BLOCKED.** Decision I (official DEAP data access) is pending. The next step is unchanged (§11). |
| More development participants; resuming model work | **BLOCKED** on the same decision. Resume only as pre-declared comparisons against per-person z-scoring (new users) and the personal Random Forest (calibrated users). |
| DEAP waveform previews | **BLOCKED** (Decision I) |
| Live administrator workflows on the real database | **NOT RUN** by rule; verified in isolated environments |
| Public research endpoints (model info, metrics, dataset summary, participant list) | Readable without sign-in by design, to preserve the existing API contracts. The UI hides them until sign-in. |

### 14.5 Final checklist

| Requirement | Status |
|---|---|
| Active Random Forest and production artifacts preserved; no candidate registered or activated | PASS |
| Experiment documents, findings and project report accurate; protocols and metrics kept separate | PASS |
| Cross-person near chance stated; personalization exploratory; no 90% or clinical claims | PASS |
| Scripts and concise result summaries retained; caches and data not committed | PASS |
| Standalone sign-in | PASS (browser 37/37) |
| Analyze EEG lists exactly the 12 trusted participants and excludes the frozen 8 | PASS (API and browser, isolated and live) |
| Backend tests, frontend tests, build, `git diff --check` | PASS |
| PostgreSQL, active pointer and model files preserved | PASS |
| Frozen-participant evaluation | **BLOCKED** (Decision I) |
| Commit and push (no force); GitHub Actions verified | Commit and run IDs are reported with the delivery message |
