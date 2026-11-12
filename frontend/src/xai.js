// Display rules for model explanations. Values come from the server, which computes them from the model itself.
import { featureLabel as describeFeature } from './components/format.js';

export const BAND = { theta: 'Theta 4–8 Hz', alpha: 'Alpha 8–13 Hz', beta: 'Beta 13–30 Hz', gamma: 'Gamma 30–45 Hz', broadband: 'Broadband' };
export const RATING_ORDER = ['Valence', 'Arousal', 'Dominance', 'Liking'];

/** True when a result may have an explanation. Synthetic test signals never do. */
export const explainable = (p) => Boolean(p) && !(p.input_source === 'upload' && p.source_reference?.synthetic_test_signal === true);

/** e.g. "O2 · log band power · gamma"; whole-signal features have no band. */
export const featureLabel = (f) => describeFeature({ ...f, band: f.band === 'broadband' ? null : f.band });

/** Contribution with its unit: points of the model's high-valence score (Random Forest) or log-odds (logistic regression). */
export function fmtContribution(v, kind) {
  const sign = v > 0 ? '+' : v < 0 ? '−' : '';
  return kind === 'probability' ? `${sign}${Math.abs(v * 100).toFixed(2)} pts` : `${sign}${Math.abs(v).toFixed(3)}`;
}

export const unitText = (kind) => (kind === 'probability'
  ? 'points of the model’s higher-valence score'
  : 'log-odds of higher valence');

/** Rows for the ratings panel, always in the same order; a rating the dataset did not collect says so. */
export function ratingRows(ratings, dataset) {
  const by = Object.fromEntries((ratings || []).map((r) => [r.name, r]));
  const names = dataset === 'FACED' ? [...RATING_ORDER, 'Familiarity'] : RATING_ORDER;
  return names.filter((n) => by[n] || n !== 'Familiarity').map((name) => {
    const r = by[name];
    if (!r) return { name, missing: dataset === 'FACED' && name === 'Dominance' ? 'Not collected in FACED' : 'Not available for this input' };
    const [lo, hi] = r.scale;
    return { name, value: r.value, scale: r.scale, share: Math.max(0, Math.min(1, (r.value - lo) / (hi - lo))) };
  });
}

/** Diverging colour: teal toward higher valence, blue toward lower, transparent at zero. */
export function divergingColor(v, max) {
  const a = max > 0 ? Math.min(1, Math.abs(v) / max) : 0;
  return v >= 0 ? `rgba(0, 184, 176, ${(0.08 + 0.85 * a).toFixed(3)})` : `rgba(77, 124, 254, ${(0.08 + 0.85 * a).toFixed(3)})`;
}

/** Sequential colour for a value scaled within its column (min..max -> light..strong teal). */
export function sequentialColor(v, min, max) {
  const a = max > min ? (v - min) / (max - min) : 0.5;
  return `rgba(0, 184, 176, ${(0.06 + 0.88 * Math.max(0, Math.min(1, a))).toFixed(3)})`;
}

export const AFFECT_ORDER = ['valence', 'arousal', 'dominance', 'liking'];
const AFFECT_LABEL = { valence: 'Valence', arousal: 'Arousal', dominance: 'Dominance', liking: 'Liking' };
const AFFECT_MEANING = { valence: 'pleasant', arousal: 'activated', dominance: 'in control', liking: 'liked' };

/** Rows for the estimates card, always in the same order; a dimension the dataset did not collect says so. */
export function affectRows(affect) {
  const by = Object.fromEntries((affect?.dimensions || []).map((d) => [d.key, d]));
  return AFFECT_ORDER.map((key) => {
    const d = by[key];
    if (!d) return { key, label: AFFECT_LABEL[key], missing: affect?.dataset === 'FACED' && key === 'dominance' ? 'Not collected in FACED' : 'No model available' };
    const v = verdict(d.predicted, d.probability_high); // an uncertain call is neither a match nor a miss
    return { ...d, meaning: AFFECT_MEANING[key], agrees: d.own_label && (v === 'higher' || v === 'lower') ? d.own_label === v : null };
  });
}

/** 'higher' | 'lower' | 'uncertain' | null. Uncertain when the model's decision and its calibrated probability point
 * different ways: the model leans one way, but trials with such scores were rated that way less than half the time. */
export function verdict(predicted, probabilityHigh) {
  if (!predicted) return null;
  if (typeof probabilityHigh !== 'number') return predicted;
  return (predicted === 'higher') === (probabilityHigh >= 0.5) ? predicted : 'uncertain';
}

export const pct = (x, digits = 0) => (typeof x === 'number' && Number.isFinite(x) ? `${(x * 100).toFixed(digits)}%` : '—');

/** The verification sentence: base value + contributions = the model output, as the server measured it. */
export function verificationText(e) {
  const total = e.trial_output - e.base_value;
  if (e.output_kind === 'probability') {
    return `Average model score ${(e.base_value * 100).toFixed(1)}% ${total >= 0 ? '+' : '−'} ${Math.abs(total * 100).toFixed(1)} pts from this recording's features `
      + `= ${(e.trial_output * 100).toFixed(1)}%, the saved result.`;
  }
  const base = e.method?.startsWith('Integrated gradients') ? 'Flat-signal baseline' : 'Intercept';
  return `${base} ${e.base_value.toFixed(3)} ${total >= 0 ? '+' : '−'} ${Math.abs(total).toFixed(3)} from this recording's features `
    + `= ${e.trial_output.toFixed(3)} log-odds (mean over segments); the saved score is ${(e.probability_high_valence * 100).toFixed(1)}%.`;
}

/** Highest and lowest segment of the per-segment score, with readable time windows. */
export function segmentExtremes(view) {
  const p = view?.segment_probability_high || [];
  if (!p.length) return null;
  const at = (i) => ({ i, p: p[i], label: `${view.kept_segments[i] * view.segment_seconds}–${(view.kept_segments[i] + 1) * view.segment_seconds} s` });
  let hi = 0;
  let lo = 0;
  p.forEach((x, i) => { if (x > p[hi]) hi = i; if (x < p[lo]) lo = i; });
  return { high: at(hi), low: at(lo) };
}

/** The cell of a channel x band table with the largest value (or largest magnitude when `signed`). */
export function strongestCell(channels, bands, matrix, signed = false) {
  let best = null;
  (matrix || []).forEach((row, i) => row.forEach((v, j) => {
    if (!Number.isFinite(v)) return;
    const score = signed ? Math.abs(v) : v;
    if (!best || score > best.score) best = { channel: channels[i], band: bands[j], value: v, score };
  }));
  return best;
}

/** The band with the largest total absolute contribution across segments. */
export function dominantBand(segmentBands) {
  if (!segmentBands?.values?.length) return null;
  const totals = segmentBands.bands.map((_, j) => segmentBands.values.reduce((a, r) => a + Math.abs(r[j]), 0));
  const j = totals.indexOf(Math.max(...totals));
  return segmentBands.bands[j];
}
