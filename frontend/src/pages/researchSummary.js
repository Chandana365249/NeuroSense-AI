// STATIC research summary shown on Model & Evidence. This is not live experiment data.
// Source of truth: docs/RESEARCH_FINDINGS_AND_NEXT_STEPS.md (compiled 2026-10-13), sections 7-10.
// When that document changes, update this file by hand; do not add numbers that are not in it.

export const SUMMARY_SOURCE = {
  document: 'docs/RESEARCH_FINDINGS_AND_NEXT_STEPS.md',
  compiled: '2026-10-13',
  status: 'Exploratory findings: 12 trusted DEAP participants, Protocol v1.1. No independent final-test results yet.',
};

export const FINDINGS = [
  {
    id: 'zero-shot',
    title: 'Predicting a new person: near chance',
    tone: 'negative',
    body: 'When both the person and the videos were held out from training, trial balanced accuracy was 0.500 '
      + '(95% CI 0.481–0.520) for the person’s own rating and 0.482 (0.453–0.512) for the other participants’ '
      + 'consensus rating. Earlier leave-one-participant-out evaluations of the Random Forest and the CNN were also near chance.',
  },
  {
    id: 'normalization',
    title: 'Unsupervised per-person normalization: no demonstrated improvement',
    tone: 'neutral',
    body: 'Standardizing each person’s features with their own unlabelled EEG raised balanced accuracy from 0.500 to 0.535, '
      + 'but the change (+0.034, 95% CI −0.008 to 0.076, p = 0.098) was not significant. It removed single-category '
      + 'predictions (4 participants to 0), but ROC-AUC barely changed (+0.017), so ranking of trials did not improve.',
  },
  {
    id: 'calibration',
    title: 'Supervised within-person calibration: above chance',
    tone: 'positive',
    body: 'A model trained on about 32 of a person’s own rated videos reached 0.618 trial balanced accuracy '
      + '(95% CI 0.550–0.687) on that person’s other videos, 0.118 higher than predicting a new person on the same trials. '
      + 'With fewer rated videos it was lower: about 0.58 with 8 (9 participants), 0.60 with 16 and 0.615 with 24.',
  },
  {
    id: 'consensus',
    title: 'The calibrated signal looks shared, not individual',
    tone: 'negative',
    body: 'On the same videos, the calibrated model predicted the other participants’ consensus rating about as well '
      + '(0.626; difference −0.008, 95% CI −0.090 to 0.075). Where a person disagreed with the consensus, the model followed '
      + 'the person’s own rating on 60 of 117 trials, which is not different from chance. Individual-specific emotional '
      + 'information has therefore not been established. The results also do not show that EEG cannot contain it: that test had low power.',
  },
  {
    id: 'not-available',
    title: 'Not available',
    tone: 'neutral',
    body: 'There is no validated arousal model and no explainability (XAI) implementation. The interface shows neither.',
  },
];

export const INTERPRETATION = 'NeuroSense AI estimates a binary category of self-reported valence. It does not detect a '
  + 'person’s exact emotional experience and is not a diagnostic tool. Estimates for new, uncalibrated users should be '
  + 'treated as experimental.';
