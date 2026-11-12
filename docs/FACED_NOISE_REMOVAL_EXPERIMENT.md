# FACED noise-removal experiment (exploratory)

**Result: negative.** An enhanced cleaning pipeline did **not** improve FACED self-reported valence classification.

| Condition | Trial-level balanced accuracy |
|---|---|
| A: current preprocessing | **0.508** |
| B: enhanced (bad-channel interpolation + 50 Hz notch + ocular ICA) | **0.498** |

The paired difference B − A is −0.010 (95% CI −0.042 to +0.024). Both conditions are close to chance. Cleaning is
**not** a route to the 80% target with this feature set and classifier, and no change to the FACED pipeline is
recommended.

The comparison used only the 98 development participants. The 25 already-evaluated test participants were never read.

---

## 1. Audit of the existing preprocessing (verified in code)

Sources: `app/ml/faced/pipeline.py`, `app/preprocessing/eeg_preprocessor.py`, `app/preprocessing/features.py` and
`scripts/faced_experiment.py`.

| Operation | Status in the current pipeline (`faced-v2`) |
|---|---|
| Channel selection | **Applied.** The 30 scalp channels shared by both recording layouts (`FACED_CHANNELS`). Mastoids A1/A2 (layout A) and HEOR/HEOL (layout B) are not used. |
| Re-referencing | **Applied.** Common average over the 30 channels, computed **before** any cleaning, so it includes any faulty channel. |
| Resampling | **Applied.** 1000 Hz recordings → 250 Hz with `scipy.signal.resample_poly` (FIR anti-alias filter); 250 Hz recordings are unchanged. |
| Band-pass filtering | **Applied.** 4–45 Hz Butterworth, order 4, zero-phase (`sosfiltfilt`). |
| Line-noise removal | **Not applied** as a separate step (no notch). Only the 45 Hz low-pass edge attenuates 50 Hz. |
| Bad-channel handling | **Not applied.** No detection or interpolation. |
| Eye-blink handling | **Not applied** as a separate step. The 4 Hz high-pass removes the slowest part of blinks; there is no ICA or regression. |
| Muscle-artifact handling | **Not applied** as a separate step. The 30–45 Hz gamma band is kept as features. |
| Artifact clipping | **Applied.** Each channel is clipped at its median ± 6 robust SD (MAD-based), per trial. |
| Segmentation and rejection | **Applied.** Last 28 s of each clip → 7 × 4 s segments; a segment is rejected if more than 5% of its samples were clipped. Across all 123 participants: 63 of 24,108 segments rejected. |
| Normalisation | **Applied.** `StandardScaler` inside the model pipeline, fitted on training rows only. No per-person normalisation in the primary model. |
| Feature extraction | **Applied.** 412 features: per channel, log and relative power in theta, alpha, beta and gamma (Welch, 1 s Hann windows), spectral entropy, Hjorth mobility and complexity, log variance; plus 13 hemispheric asymmetry pairs × 4 bands. |
| Trial prediction | Mean of the segment probabilities, compared with 0.5. |

### Label-free audit of the data (98 development participants; `outputs/faced/cleaning/audit.json`)

These measurements use no labels. They were used to decide which cleaning steps are justified.

- **Line noise:** a 50 Hz mains peak is present. Its power is a median 4.2× that of the neighbouring frequencies, and
  above 3× for 59 of 98 participants. FACED was recorded in China, where mains is 50 Hz.
  - However, a 50 Hz notch removes only about **0.8%** of the 30–45 Hz power that the features measure. The 4–45 Hz
    band-pass already suppresses most of it.
  - A notch was therefore justified as good practice, but was expected to change little.
- **Eye artifacts:** large 1–4 Hz frontal (Fp1/Fp2) deflections, more than 5× the posterior channels, occur in a median
  26% of 1-second windows (interquartile range 17–37%). Ocular artifacts are common, which justifies ocular ICA.
- **Bad channels:** 4 of 98 participants have clearly abnormal channels (details in section 2).
- **Recording reference (an important finding):** the system is **BioSemi** (`eeg.json`), which records relative to its
  CMS electrode, placed on the cap next to CPz.
  - In the recorded data, the channels nearest CMS (CP1, CP2, Cz, Pz) are therefore legitimately tiny (about 0.2–0.6 µV
    against a median of about 6 µV) and weakly correlated with the others.
  - A first detector run on the recorded reference flagged 25 participants, almost all for these channels. Interpolating
    them would have **replaced valid EEG**.
  - That detector was corrected before any condition-B feature was computed. The first audit is kept as
    `audit_v0_recorded_reference.json`.

---

## 2. Conditions compared

**A: current pipeline.** The `faced-v2` features as used by the existing FACED model, unchanged
(`features_faced-v2.npz`, development participants only).

**B: enhanced cleaning** (`scripts/faced_cleaning_experiment.py`, declared in `outputs/faced/cleaning/protocol.json`
before any B feature or score existed). Every step is unsupervised and computed per recording from that participant's 28
trial windows, so it uses neither labels nor other participants.

1. **Bad-channel detection** on 1–45 Hz data after a robust (median-across-channels) reference. A channel is bad if any
   of these holds:
   - its robust amplitude is more than 5 robust SD above the other channels;
   - its 1 s SD exceeds 5× the cross-channel median in more than 5% of windows (intermittent bursts);
   - its maximum |r| with any other channel is below 0.4 in more than 50% of 1 s windows;
   - it is flat in the recorded data (robust SD < 0.05 µV).

   Fp1/Fp2 are exempt from the two amplitude rules only: their high amplitude is blinks, which step 4 handles, and they
   are the ocular reference for ICA.
2. **Interpolation** of bad channels by spherical splines on standard 10–20 positions (MNE), with at most 6 per
   recording.
3. **Common average reference** after interpolation, then a **50 Hz notch** (IIR, Q = 30, zero-phase).
4. **Ocular ICA:**
   - FastICA, fitted on that recording's 1–45 Hz data, excluding 2 s spans with peak-to-peak amplitude above 500 µV.
   - The rank is set for the average reference and for interpolated channels.
   - Only components whose 1–10 Hz time course correlates with Fp1/Fp2 (MNE `find_bads_eog`, z > 3) are removed, at
     most 3 per recording.
   - **No muscle or high-frequency components were removed and no frequency band was dropped.**
5. Exactly condition A's band-pass, clipping, segmentation, rejection and 412 features.

**Changes to B made before evaluation, all on label-free evidence:**
- *v0 → v1 (detector computed on the recorded reference):* superseded because of the BioSemi CMS reference effect above.
- *v1 → v2 (no intermittent-burst rule, ICA fitted on all data):* superseded after a smoke test on one participant, with
  no labels involved. In sub-030, PO4 had bursts reaching about 150 mV that the median-based amplitude ignored.
  - The common average spread those bursts to every channel.
  - ICA's "ocular" components then captured the spread burst and *raised* the cleaned signal's robust amplitude 1.5×.
  - v2 added the intermittent-burst rule and the 500 µV ICA-fit exclusion, after which ICA behaved as intended.
- v1 is kept as `protocol_v1_superseded.json`. No B feature or classification score existed before v2 was declared.

### What B did to the data (98 development participants)

| | A: current | B: enhanced |
|---|---|---|
| Segments kept | 19,148 | 19,178 |
| Segments rejected | 60 (0.31%) | 30 (0.16%) |
| Trials kept | 2,744 (all) | 2,744 (all) |
| Class balance (trials, high valence) | 43.3% | 43.3% (unchanged) |
| Participants with interpolated channels | — | 4: sub-030 (F8 flat; PO4 and Oz bursts); sub-064 (Fz); sub-101 (FC2); sub-103 (T7, T8 low correlation) |
| Recordings over the interpolation limit | — | 0 |
| ICA components removed per recording | — | 0 for 3, 1 for 52, 2 for 38, 3 for 5 |
| Frontal (Fp1/Fp2) robust amplitude after/before ICA | — | median 0.52 (range 0.30–1.00) |
| All-channel robust amplitude after/before ICA, per recording | — | median 0.89 (range 0.70–1.00) |

**Could useful signal have been removed?** Possibly, at least at frontal channels.
- Ocular components were picked by correlation with Fp1/Fp2. These channels also carry genuine frontal EEG, including
  the frontal activity that has been linked to valence. Halving their robust amplitude certainly removed blink activity,
  and may also have removed some of that signal.
- Elsewhere, about 11% of robust amplitude was removed, which suggests ocular spread was taken out without wholesale loss.
- Rejection fell rather than rose, because interpolation and ICA reduced the clipping that triggers it. B discarded
  *less* data, not more.

---

## 3. Evaluation

- **Participants:** the 98 development participants of `faced-protocol-v1` only. The 25 test participants, which have
  already been evaluated, were never read, so this is not a test-set result.
- **Folds:** 5 participant-grouped folds, using the protocol's existing inner participant groups. A validation fold's
  participants are never in its training data. All 28 clips are used, so clips are shared across folds.
- **Label (unchanged):** self-reported valence > 3.5.
- **Classifier (unchanged):** the selected FACED model, logistic regression with C = 0.01, with the scaler fitted on
  training rows only.
- **Trial rule:** mean segment probability > 0.5.
- **Leakage:** no labels are used by any cleaning step, and nothing is fitted across participants. There is no feature
  selection.
- **Same for both conditions:** folds, classifier, label and metric.

### Results (out-of-fold, 2,744 trials, 98 participants)

| Metric | A: current | B: enhanced |
|---|---|---|
| **Trial balanced accuracy (pooled)** | **0.508** (95% CI 0.480–0.536) | **0.498** (95% CI 0.472–0.526) |
| Trial balanced accuracy per fold | 0.493, 0.520, 0.529, 0.530, 0.467 (mean 0.508) | 0.470, 0.544, 0.519, 0.492, 0.463 (mean 0.497) |
| Accuracy | 0.507 (95% CI 0.477–0.536) | 0.497 (95% CI 0.469–0.525) |
| Macro-F1 | 0.505 | 0.495 |
| ROC AUC | 0.518 | 0.505 |
| Confusion [[low→low, low→high], [high→low, high→high]] | [[777, 779], [574, 614]] | [[757, 799], [582, 606]] |
| Per-participant balanced accuracy (94 with both classes) | mean 0.518, SD 0.083, 95% CI 0.501–0.535 | mean 0.504, SD 0.087, 95% CI 0.486–0.522 |
| Segment-level balanced accuracy / AUC | 0.509 / 0.517 | 0.502 / 0.505 |
| Majority baseline (always low) | accuracy 0.567, balanced accuracy 0.500 | same |

CIs for the pooled metrics are participant-cluster bootstraps with 2,000 resamples.

**Paired comparison** (same 2,744 trials):
- B − A balanced accuracy = **−0.010**, 95% CI −0.042 to +0.024 (paired participant bootstrap).
- Per participant: mean difference −0.015; 29 participants improved and 48 got worse (Wilcoxon p = 0.12).

**There is no evidence that the enhanced cleaning helps. If anything, the point estimate is slightly worse.**

### Context

In this participant-grouped setting with all 28 clips, even condition A is near chance (0.508). The earlier inner folds,
which held out both participants and clips, gave 0.555 for the same classifier. The two numbers come from different fold
designs and are not directly comparable; the A-vs-B comparison here is the one that matters. Both results agree that
little valence information is reachable with these features across people.

---

## 4. Limitations

- One cleaning variant (B) was compared with the current pipeline. Other choices (ICLabel-style classification, ASR,
  regression on the layout-B HEOG channels, muscle removal) were not tested. ICLabel and HEOG regression would also need
  information that only some recordings have, or extra dependencies.
- Ocular components were chosen with Fp1/Fp2 as proxies, because there is no vertical EOG channel. Some frontal EEG may
  have been removed with them.
- B was revised twice before evaluation, for documented, label-free reasons (section 2).
- One classifier and one feature set were used; the result says nothing about raw-signal deep models.
- The evaluation uses development participants only. It is exploratory and gives no independent test-set estimate.
- MNE 1.13.2 is needed for this experiment only. It is pinned in `backend/requirements-research.txt`, not in
  `requirements.txt`, because the application does not use it. numpy, scipy and scikit-learn versions were unchanged.

## 5. Recommended next step

- **Do not adopt condition B.** It adds complexity, a new dependency and per-recording ICA, with no measurable benefit.
- **Keep `faced-v2` and the existing FACED model unchanged.**
- Two audit findings are worth carrying forward as data-quality notes:
  - the BioSemi CMS reference makes CP1/CP2/Cz/Pz look tiny in the recorded reference, so any future channel check must
    re-reference first;
  - a few recordings have bursty channels (sub-030 PO4/Oz, sub-064 Fz, sub-101 FC2) or a faulty T7/T8 pair (sub-103).
- Accuracy is limited by how weakly these EEG features relate to self-reported valence across people, not by residual
  noise.
- Any further attempt should change the **question or representation**, not the cleaning. Options:
  - within-person calibration with a few labelled trials, declared in advance;
  - a different target, such as arousal or discrete emotions;
  - learned representations with a properly held-out evaluation.
- Any such attempt needs a new, untouched test set.

## 6. Reproduction

From `backend/`, with `NEUROSENSE_FACED_DATASET_DIR` set:

```bash
python -m pip install -r requirements-research.txt                 # research-only dependency: mne==1.13.2
python -m scripts.faced_cleaning_experiment --phase audit          # label-free characterisation
python -m scripts.faced_cleaning_experiment --phase protocol       # declare condition B
python -m scripts.faced_cleaning_experiment --phase build          # condition B features, development participants (~30 min, 4 workers)
python -m scripts.faced_cleaning_experiment --phase evaluate       # A vs B on identical folds
```

- Outputs (Git-ignored) are in `outputs/faced/cleaning/`: `audit.json`, `audit_v0_recorded_reference.json`,
  `audit_v1.json`, `protocol.json` (SHA-256 `249a4b6e…0b14006`), `protocol_v1_superseded.json`, `features_B.npz`,
  `build_report.json` and `results.json`.
- Seed: 42. Results were generated from commit `f506e27` plus the uncommitted FACED working tree.

## 7. Tests, safety and integrity

**Tests (2026-10-26, after MNE was installed):**
- Backend `python -m pytest`: **262 passed**. This run also included the 2 real-DEAP integration tests, which read only
  the 12 authorised DEAP files in the configured folder; none of the frozen participants is present there.
- Frontend `npm test`: **40 / 40 passed**.
- `git diff --check`: clean.

- **Unchanged:**
  - the active DEAP model `valence_binary-random_forest-20261008-180558` and the 70 hashed production files;
  - the existing FACED model files, model card and earlier result files (timestamps unchanged);
  - the production PostgreSQL database (read-only checksum snapshot of all tables).
- No new model was saved or activated.
- The eight frozen DEAP participants were not accessed. Nothing was committed or pushed.
