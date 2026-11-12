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

export { sourceText as sourceLabel } from '../analysis.js';

/** Category as shown in lists. Synthetic test inputs are never presented as an estimate. */
export function predictionLabel(p) {
  if (p?.input_source === 'upload' && p.source_reference?.synthetic_test_signal) return 'Synthetic test (not an estimate)';
  return classLabel(p?.prediction);
}

export const VALENCE_EXPLANATION =
  'Valence is how pleasant a participant rated a music video, on a 1–9 scale, once after each one-minute trial. '
  + 'Ratings above 5 are classed as higher self-reported valence; 5 or below as lower self-reported valence.';

// ---------- global feature importance ----------
const BAND_NAMES = { theta: 'theta', alpha: 'alpha', beta: 'beta', gamma: 'gamma' };
const FAMILY_NAMES = {
  logpow: 'log band power', relpow: 'relative band power', spectral_entropy: 'spectral entropy',
  hjorth_mobility: 'Hjorth mobility', hjorth_complexity: 'Hjorth complexity', log_variance: 'log variance',
  asym: 'right–left asymmetry',
};

/** Plain-language description of a feature, e.g. "O2 · log band power · gamma". */
export function featureLabel(f) {
  return [f.channel, FAMILY_NAMES[f.family] || f.family, f.band ? BAND_NAMES[f.band] || f.band : null].filter(Boolean).join(' · ');
}
