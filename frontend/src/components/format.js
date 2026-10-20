// Display helpers. User-facing wording follows the approved scientific labels.

export const NA = 'Not available';

export const CLASS_LABELS = {
  high_valence: 'Higher self-reported valence',
  low_valence: 'Lower self-reported valence',
};

export const ALGORITHM_LABELS = {
  random_forest: 'Random Forest',
  cnn: 'CNN (EEGNet-style)',
  svm_rbf: 'SVM (RBF)',
  xgboost: 'XGBoost',
};

export const classLabel = (c) => (c ? CLASS_LABELS[c] || c : NA);
export const algorithmLabel = (a) => (a ? ALGORITHM_LABELS[a] || a : NA);
export const pct = (v) => (typeof v === 'number' ? `${(v * 100).toFixed(1)}%` : NA);
export const num = (v, d = 3) => (typeof v === 'number' ? v.toFixed(d) : NA);
export const when = (iso) => (iso ? new Date(iso).toLocaleString() : NA);

export function sourceLabel(p) {
  if (p?.input_source === 'deap_reference' && p.source_reference) {
    return `DEAP ${p.source_reference.participant_id}, trial ${p.source_reference.trial_index}`;
  }
  return 'Uploaded EEG file';
}

export const VALENCE_EXPLANATION =
  'Valence is how pleasant a participant rated a music video, on a 1–9 scale, once after each one-minute trial. '
  + 'Ratings above 5 are classed as higher self-reported valence; 5 or below as lower self-reported valence.';

export const RELIABILITY_NOTE =
  'Research prototype: cross-participant estimates were near chance in our exploratory evaluations. '
  + "Treat predictions as experimental, not as a reliable measurement of an individual's emotional experience.";
