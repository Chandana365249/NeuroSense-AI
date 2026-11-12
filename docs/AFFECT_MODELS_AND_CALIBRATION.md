# Arousal, dominance and liking models, calibrated probabilities and personalised estimates

Code: `backend/app/affect/` (`core.py`, `build.py`, `service.py`). Build: `python -m app.affect.build` from `backend/`.
Tests: `backend/tests/test_affect.py`. Artifacts: `outputs/affect/`, which is not committed.

The active DEAP valence model and the FACED valence model are **unchanged**. Valence only gets a calibration. The new
models are separate files, are not registered in the model registry and do not change the active-model pointer.

## 1. What was built

| Dataset | Dimension | Model | Training data |
|---|---|---|---|
| DEAP (1–9, high = rating > 5) | arousal, dominance, liking | existing Random Forest configuration, no tuning | the 12 development participants (7,046 segments). The 8 frozen participants are never read |
| FACED (0–7, high = rating > 3.5) | arousal, liking (dominance was not collected) | FACED logistic regression configuration (C = 0.01) | the 98 development participants × 19 development clips (12,994 segments). The 25 test participants and 9 test clips are dropped before any value is used |

## 2. How it was evaluated

**Out-of-person accuracy.**
- **DEAP:** leave-one-participant-out cross-validation.
- **FACED:** the five development participant groups fixed by faced-protocol-v1.
- A person's trials are scored only by a model that never saw that person.
- Balanced accuracy is computed per person, then averaged, with a t-based 95% CI across people.

**Calibration.**
- **Method:** Platt scaling of the out-of-person trial scores, so that "60%" means trials given that score were rated
  high about 60% of the time.
- **Error figures:** the calibration error (ECE) figures are nested. The calibration that scores a person is fitted
  without that person.
- **Safeguard:** if the fitted slope is negative, meaning higher scores go with fewer high ratings, the calibrated
  probability falls back to the base rate instead of being inverted.

**Personalised models.**
- One model per dimension, trained only on the same person's other rated trials and tested on the left-out trial
  (leave-one-trial-out).
- The app does the same on request ("Personalise to this participant"). It never trains on the trial being
  estimated.

**What these figures are not.** They are exploratory cross-validation on development participants, not an
independent held-out test.

## 3. Results

Balanced accuracy per person, mean with 95% CI.

| Dataset | Dimension | Unseen people | Better than chance? | Personalised |
|---|---|---|---|---|
| DEAP | valence (calibration only) | 52.2% (48.6–55.8) | no | **62.1% (54.8–69.5)** |
| DEAP | arousal | 51.4% (47.7–55.2) | no | 56.7% (51.1–62.2) |
| DEAP | dominance | 50.9% (45.7–56.2) | no | 56.5% (47.5–65.5) |
| DEAP | liking | 49.9% (49.6–50.1), AUC 0.42 | no | 51.4% (45.4–57.4) |
| FACED | valence (calibration only) | 53.4% (51.3–55.5) | yes, slightly | **59.5% (56.5–62.4)** |
| FACED | arousal | 54.3% (52.1–56.5) | yes, slightly | 54.5% (51.7–57.3) |
| FACED | liking | 53.2% (51.2–55.2) | yes, slightly | 56.7% (53.5–60.0) |

Calibration error (ECE, out-of-person scores):

| | Raw | Calibrated |
|---|---|---|
| FACED valence | 0.109 | 0.013 |
| FACED arousal | 0.127 | 0.009 |
| FACED liking | 0.091 | 0.008 |
| DEAP valence | 0.034 | 0.040 |
| DEAP arousal | 0.038 | 0.043 |
| DEAP dominance | 0.046 | 0.038 |
| DEAP liking | 0.126 | 0.129 (base rate only: the score has no usable direction for new people, and how often people rate videos high varies from person to person) |

DEAP Random Forest scores were already close to calibrated, so Platt scaling changes little there.

## 4. What this means

- **Strongest lever: personalisation.** Personalised valence models are clearly better than chance in both datasets
  (62% DEAP, 60% FACED). This matches the earlier within-person study (0.618).
- **DEAP arousal, dominance and liking models are at chance level for new people.** The app shows their estimates but
  labels each one "no better than chance on new people". For DEAP liking, the calibrated probability is the base rate.
- **FACED models are slightly but reliably better than chance** (53–54%).
- **Calibration makes the probabilities honest; it does not make them confident.** Calibrated probabilities of weak
  models stay close to the base rate.

## 5. In the app

The "Estimated emotion for this trial" card shows, for valence, arousal, dominance and liking:
- the model's decision;
- the calibrated probability of a higher rating;
- the participant's own rating, with whether they match;
- the out-of-person accuracy with its CI;
- a "no better than chance" label where the CI includes 50%.

The valence gauge shows the calibrated probability. For a participant the models were trained on, the estimate uses
the stored score from the cross-validation fold where that person was held out, so no estimate comes from a model that
saw the person.
