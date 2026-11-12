# NeuroSense AI — Explainable AI (XAI)

Every non-synthetic analysis now answers **"Why did the model make this prediction?"**. The answer is computed from
the trained model that made the prediction and the exact EEG features it received. It is not a static list.

Code: `backend/app/xai/` (`attribution.py`, `service.py`, `build_global.py`) and `backend/app/api/routes/xai.py`.
Frontend: `TrialInsights.jsx`, `GlobalXai.jsx`, `ScoreGauge.jsx` and `xai.js`.
Tests: `backend/tests/test_xai.py` and `frontend/tests/xai.test.mjs`.

## 1. What was inspected

| Item | Finding (verified in code and on the saved artifacts) |
|---|---|
| Active production model | `valence_binary-random_forest-20261008-180558`: `EncodedLabelClassifier` around `StandardScaler` → `RandomForestClassifier` (300 trees, `min_samples_leaf=5`, balanced class weights). Classes `['high_valence', 'low_valence']`, so column 0 is high valence |
| FACED model (experimental, never activated) | `faced-valence-lr_C0.01-S1`: `StandardScaler` → `LogisticRegression(C=0.01, class_weight='balanced')`. Classes `[0, 1]`, where 1 = high valence |
| Features | `app/preprocessing/features.py`, one row per 4 s segment, 12 per channel: log band power and relative band power in θ 4–8, α 8–13, β 13–30 and γ 30–45 Hz; spectral entropy; Hjorth mobility and complexity; log variance. Plus right−left log-power asymmetry per channel pair and band. **No delta band**: the data is high-passed at 4 Hz |
| Feature counts | DEAP: 32 channels × 12 + 14 pairs × 4 = **440**. FACED: 30 channels × 12 + 13 pairs × 4 = **412** |
| Channel and band mapping | Parsed from the extractor's own feature names (`{channel}_{family}_{band}`, `asym_{right}-{left}_{band}`). No electrode position is used or invented |
| Trial prediction | Mean of the segment probabilities (`ModelSnapshot.predict_signal`, `FacedService.predict`) |
| XAI dependencies | `shap` was not installed. Added `shap==0.54.1` (with `pandas==3.1.0`) to the project virtual environment and `requirements.txt` |

## 2. Methods

| Model | Technique | What adds up exactly |
|---|---|---|
| Random Forest (DEAP) | **TreeSHAP**: `shap.TreeExplainer`, exact, tree-path-dependent, on the forest's high-valence probability. The scaler transforms each feature on its own, so the attribution of a scaled feature is the attribution of the named raw feature | For each segment: base value + Σ SHAP = model probability (error about 1e-14) |
| Logistic regression (FACED) | **Exact linear attribution**: contribution = coefficient × (x − training mean) / training SD; baseline = intercept. This is linear SHAP with the scaler's training mean as reference | For each segment: intercept + Σ contributions = decision score (log-odds) |
| XGBoost | **TreeSHAP** on the booster's margin (log-odds of higher valence) | For each segment: base value + Σ SHAP = log-odds (error below 1e-4; XGBoost computes in float32) |
| CNN (raw EEG segments) | **Integrated gradients**: 128 steps from an all-zero normalised input, i.e. each channel at its training mean, on log(p_high / p_low). Summed per channel and per second. No band attribution is claimed, because the network has no band features | Baseline + Σ attributions = log-odds, up to a measured integration error (must stay below 0.02; about 2e-5 on the saved CNN) |
| Any other model (for example an SVM) | **Not supported**: the API says so and gives no explanation | — |

The CNN and XGBoost paths are used only if such a model is the active one. The active model is the Random Forest.

**Trial level.** The trial output is the mean over segments, so the trial attribution is the mean of the segment
attributions. This is an exact identity.

**Grouping.**
- **By frequency band:** every feature has exactly one band. Spectral entropy, Hjorth parameters and variance are
  "broadband".
- **By channel:** single-channel features only.
- **Asymmetry pairs:** reported separately.
- **By feature family.**
- **Channel × band:** a map of log-band-power contributions.

## 3. How an explanation is tied to a result

`GET /api/predictions/{id}/explanation` (or `POST` for an upload that re-sends its signal):

1. **Load the input.** Re-read the same recording, or take the re-sent upload, and check its shape against the stored
   record.
2. **Check the model.** Load the model recorded for the result; if it is no longer the active or current version, refuse.
3. **Recompute the features** with the same preprocessing and feature extractor.
4. **Recompute the probability** and compare it with the stored one. If they differ by more than 1e-6, refuse.
5. **Explain** and check that the attributions reconstruct the model output, again to within 1e-6.

Refusals are explicit (`available: false` with a reason). They cover:
- synthetic test signals, which are never explained;
- uploads analysed before uploads were stored, whose signal was not re-sent. Since this version, each real upload is
  stored privately (`outputs/uploads/<user>/<prediction>.npz`, readable only through its owner's result) and explained
  later without re-sending; synthetic signals are never stored;
- a model that has changed;
- a result that cannot be reproduced;
- an unsupported algorithm.

Nothing is written to the database.

## 4. Global (model-level) explanations

`python -m app.xai.build_global` precomputes the mean absolute attribution on each model's **own training data**:

| Model | Reference data |
|---|---|
| DEAP | Recorded training split s02, s03, s05, s07, s08, s15, s22. Every kept segment is used, about 4,000 segments, with one process per participant: a few minutes on an 8-core laptop instead of about 20. The result is the same. The frozen DEAP participants are refused before any file is opened |
| FACED | The 98 faced-protocol-v1 development participants on the 19 development clips (12,994 segments). Rows of the 25 test participants and the 9 test clips are dropped before use |

The result goes to `outputs/cache/xai/` and is served by `GET /api/explain/global/{deap|faced}`. It includes:
- top features;
- totals by band, channel, asymmetry pair and feature family;
- a channel × band map;
- per-participant variability: SD and range across participants.

## 5. In the app

**Analyze EEG, Result details and FACED results:**
- **Valence gauge:** the calibrated probability of higher valence. It falls back to the labelled raw model score when
  no calibration is available.
- **Estimated emotion card:** valence, arousal, dominance (DEAP only) and liking, each with:
  - the model decision and the calibrated probability;
  - the participant's own rating;
  - the out-of-person accuracy, with a "no better than chance" label where it applies;
  - an optional personalised estimate.

  See `docs/AFFECT_MODELS_AND_CALIBRATION.md`.
- **Per-trial graphs:**
  - valence score per 4 s segment;
  - EEG traces of every channel, for any chosen segment;
  - relative band power by channel and band.
- **"Why did the model make this prediction?":**
  - a verified sum: base value + contributions = the saved result;
  - top features pushing toward higher or lower valence;
  - contribution by band and by channel;
  - a channel × band contribution map;
  - band contributions per segment;
  - caveats.

**Model & Evidence.** An "Explainable AI" panel for each dataset shows:
- the most influential features, bands and channels;
- the channel × band map;
- variability across participants.

**Removed from the main workflow:** accuracy figures on Overview and the long accuracy text on Analyze and result
pages. Measured accuracy remains on the Model & Evidence and Dataset research pages.

## 6. Scientific limits

- **Model attribution, not causation:** a highly weighted channel is not where an emotion comes from.
- **Not probabilities:** SHAP values and log-odds contributions are not probabilities or confidence. The calibrated
  probabilities are a separate output (`docs/AFFECT_MODELS_AND_CALIBRATION.md`).
- **Possible artefacts:** gamma-band and frontal features are prone to muscle and eye-movement artefacts.
- **Weak model, weak explanation:** the explanations describe models that are only slightly better than chance for new
  people (see `docs/FINAL_RESEARCH_SUMMARY.md`). A clear explanation does not make an estimate correct.
- **Arousal, dominance and liking** now have their own models (`docs/AFFECT_MODELS_AND_CALIBRATION.md`). The DEAP ones
  are at chance level for new people, and the app labels them so.
- **No clinical validity.**

## 7. Real example and verification

These come from an end-to-end run on an isolated server (scratch database, real models, real recordings read-only),
3 November 2026.

### DEAP

**Input:** participant s01, trial 1 (index 0), through the active Random Forest. The participant is in the model's
held-out split, but was examined in earlier development.

| Item | Value |
|---|---|
| Saved result | higher valence, valence score 0.55928263852495 |
| Recomputed by the explainer | 0.55928263852495, identical |
| Base value (average model score on training data) | 0.49957 |
| Base + Σ SHAP | 0.55928, reconstruction error 1.0e-14 per segment (15 segments) and 3.3e-16 for the trial |
| Top features toward higher valence | Fp1 log beta power +0.94 pts, Fp1 log variance +0.83, Fp1 log gamma power +0.64, P8 log gamma power +0.57, P4−P3 gamma asymmetry +0.44 |
| Top features toward lower valence | FC6 log gamma power −0.54, FC6 log variance −0.42, PO4−PO3 alpha asymmetry −0.39, FC5 log gamma power −0.37, O2 log variance −0.36 |
| By band | broadband +2.03, gamma +1.49, theta +1.16, alpha +0.90, beta +0.38 pts |
| By channel | Fp1 +3.34, P8 +1.47, FC6 −1.38, FC5 −0.92 pts |
| Participant's own ratings (1–9) | valence 7.71, arousal 7.6, dominance 6.9, liking 7.83 |

Fp1 is a frontal-pole electrode, where eye movements are strongest, and gamma power is easily contaminated by muscle.
This example shows why the explanation must not be read as the brain's valence signal.

### FACED

**Input:** development participant sub-002, video 3, through the FACED logistic regression.

| Item | Value |
|---|---|
| Saved result | valence score 0.64072, recomputed identically |
| Intercept | −0.0430 |
| Intercept + Σ contributions | 0.5971 log-odds (the mean of the segment log-odds), reconstruction error 8.9e-16 |
| By band (log-odds) | beta −0.661, broadband +0.558, gamma +0.548, theta +0.150, alpha +0.045 |

The saved score is the mean of the segment probabilities, which is not the same as the sigmoid of the mean log-odds.
The app therefore states the log-odds sum and the saved score separately.

### Global

| Model | Reference data | Bands by mean absolute contribution (SD across participants) |
|---|---|---|
| DEAP | 4,073 segments from the 7 training participants, matching the training-segment count in the model's metadata | gamma 14.75 (1.67), beta 9.45 (2.01), broadband 8.00 (1.01), alpha 5.04 (0.35), theta 4.08 (0.23) pts |
| FACED | 12,994 segments from 98 development participants | broadband first, then gamma and beta |

## 8. Tests

| Suite | Result | What it checks |
|---|---|---|
| `backend/tests/test_xai.py` | 11 tests | feature order (440 / 412); TreeSHAP and linear additivity; class direction; scaling and intercept; reproducibility; refusal of unsupported models; API verification against the stored result; unchanged stored prediction; synthetic signals refused; uploads need the same signal; a tampered stored result refused; ownership; FACED log-odds; global reference data excludes test participants and clips (FACED) and uses only the training split (DEAP) |
| `frontend/tests/xai.test.mjs` | 6 tests | display rules: synthetic results not explained, units and signs, labels, rating order including "Not collected in FACED", colours, the verification sentence |
| Full backend suite | 303 passed | — |
| Frontend suite | 54 passed | — |
| Browser check | 15 of 15 | isolated server, real models: DEAP and FACED explanations verified; synthetic signals refused; global panels shown; Overview without accuracy; no horizontal overflow at 375 px; no JavaScript errors |
