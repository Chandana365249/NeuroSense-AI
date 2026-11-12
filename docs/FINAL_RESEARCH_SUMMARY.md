# NeuroSense AI — final research summary

**What the system estimates:** a binary category of **self-reported valence**: whether a person later rated a video
clip as more or less pleasant than the midpoint of the rating scale. It does not detect emotions, measure exact
feelings or support any clinical use.

**Bottom line:** under honest evaluation, EEG predicts a new person's valence rating at about **55–57% balanced
accuracy**, a little above chance (50%). Every attempt to push this higher landed in the same range. The 80% target
was **not** reached, and no result in this project is claimed to reach it.

---

## 1. All results in one place

Primary metric: trial-level balanced accuracy with a participant-clustered 95% CI.

### Evaluated on held-out test people

| Study | Dataset | Setting | Model | Result | Status |
|---|---|---|---|---|---|
| Baseline (active app model) | DEAP, 3 test people (s01, s09, s11) | fixed held-out split | Random Forest | 59.0% (120 trials, no CI) | small test; frozen |
| CNN robust evaluation | DEAP, 12 people | leave-one-person-out | EEGNet-style CNN | 49.4% ± 2.3 | below the Random Forest; inactive |
| FACED main study | FACED, 25 unseen people × 9 unseen clips | strict, evaluated once | logistic regression | **56.6% (51.8–61.2)** | primary FACED result |
| FACED, clips seen in training | FACED | unseen people, seen clips | logistic regression | 54.3% (50.6–57.6) | secondary |
| Stimulus category (different task) | FACED | intended category of known clips | logistic regression | 55.8% (53.0–59.0) | exploratory |
| Fusion deep model, zero-shot | FACED, same 25 test people (reused) | strict, evaluated once | FusionNet (35k parameters) | **57.3% (51.3–63.6)** | no gain over logistic regression |
| Fusion, external dataset | DEAP, 12 people | trained on FACED only | FusionNet | **49.1% (47.6–50.0)** | chance: does not transfer |

### Development cross-validation and calibration studies (not independent tests)

| Study | Dataset | Setting | Model | Result | Status |
|---|---|---|---|---|---|
| Noise removal | FACED development | enhanced cleaning vs standard | logistic regression | 49.8% vs 50.8% | negative |
| All-123-people model | FACED, cross-validation | unseen people / unseen people and clips | logistic regression | 53.4% / 49.3% | not a test result |
| Personal calibration | FACED, 25 test people (reused) | 21 rated clips, whole blocks held out | logistic regression | 59.5% (56.4–62.1) | best honest calibration figure |
| Fusion, personalised | FACED, 21 test people (reused) | 21 rated clips, last block tested | FusionNet + personal head | 68.3% (57.3–79.0) | 84 trials; wide CI; not independent |
| Development optimisation | FACED, 98 development people | 5-fold, 6 method families, 27 conditions | logistic regression and FusionNet | best +1.7 pp (−3.2 to +6.6) | negative |

Sources: `README.md`, `docs/FACED_INTEGRATION_AND_EVALUATION.md`, `docs/FACED_STIMULUS_CATEGORY_EXPERIMENT.md`,
`docs/FACED_NOISE_REMOVAL_EXPERIMENT.md`, `docs/FACED_CALIBRATION_EXPERIMENT.md`,
`docs/REVIEW3_CNN_ROBUST_EVALUATION.md`, `docs/NEUROSENSE_FUSION_REPORT.md`,
`docs/NEUROSENSE_DEV_OPTIMISATION_REPORT.md`.

## 2. What is established, and what is not

**Established:**
1. **Cross-person ceiling:** with these datasets and labels, a model trained on other people predicts a new
   person's valence category at about 55–57%. Two very different models agree (logistic regression 56.6%, deep
   network 57.3%).
2. **No dataset transfer:** a model trained on FACED does not carry over to DEAP (49.1%). Headset, population and
   stimuli all differ.
3. **What didn't help:** extra cleaning, per-person standardisation, covariance alignment, per-person score
   thresholds, early stopping and seed ensembles. None produced a reliable gain.
4. **Calibration helps somewhat:** a person's own ratings raise accuracy to about 60% in the most careful
   measurement.

**Not established:**
- **80% accuracy**, in any setting.
- **The personalised 68.3% (Fusion):** it rests on 4 test clips per person, from people examined before. It is a
  hypothesis for a future study, not a finding.
- **Individual emotion recognition, or any clinical meaning.**

## 3. Why accuracy stops near 55–57%

1. **The label is noisy.** It is one self-rating per clip, cut at the scale midpoint, so near-midpoint ratings
   flip easily.
2. **The class balance is driven by the clips.** Across folds of different clips, the share of "pleasant" trials
   ranged from 28% to 75%. Methods that give each person a fixed decision rate removed that real signal and made
   results worse.
3. **People's EEG differs more than their ratings do.** Within one person, the model barely ranks pleasant above
   unpleasant clips once that person's offset is removed: the AUC is about 0.5.
4. **The data is small for deep learning:** 98 training people. A larger model did not help.
5. **Why published papers report 80–90%.** Many DEAP papers use random segment-level splits, which put
   4-second pieces of the same recording in both training and testing. On this project's data, a deliberately
   leaky split reproduces about 80% (`docs/MODEL_IMPROVEMENT_EXPERIMENTS.md`). Every result above avoids that leak.

## 4. How the evaluation was kept honest

- **Held-out data:** people and clips were held out together; models never saw the test people or their clips.
- **One-shot testing:** models and thresholds were chosen on development folds only, fixed before testing, and
  tested once. Test runs refuse to repeat.
- **Protected participants:** 8 DEAP participants (s12, s17, s18, s20, s21, s23, s29, s30) were never read.
- **Leakage checks:** checks are written as code assertions with unit tests, and they caught several bugs before
  any result was produced.
- **Negative results reported:** the most important conclusion is how good the models are *not*.

## 5. Future work

1. **Personal calibration on new people.** This is the only route with a measured gain. Record 30 or more new
   participants who each rate 40–60 clips over more than one session, with the presentation order logged.
   Register the analysis before recording and test once.
2. **An independent external dataset.** Request DREAMER or AMIGOS from their authors. They use a 14-channel
   headset, so a 14-channel model would have to be built and selected before opening them.
3. **Richer labels.** Use continuous ratings during or after each clip, analysed as regression rather than
   high/low.
4. **Defaults for any future modelling:** 5-fold development selection (3 folds proved unreliable) and a small
   seed ensemble.

## 6. Application status

| Item | Status |
|---|---|
| Active model | the DEAP Random Forest, unchanged |
| FACED models | both are experimental, not registered and cannot be activated |
| Fusion model | research only; not used by the app |
| Application | runs on 127.0.0.1 only, behind sign-in, and labels every result as a research estimate of self-reported valence |
