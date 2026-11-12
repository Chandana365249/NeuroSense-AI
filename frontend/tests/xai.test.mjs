import assert from 'node:assert/strict';
import { test } from 'node:test';

import { affectRows, dominantBand, segmentExtremes, strongestCell, verdict, divergingColor, explainable, featureLabel, fmtContribution, pct, ratingRows, verificationText } from '../src/xai.js';

test('synthetic test signals are never explained; real inputs are', () => {
  assert.equal(explainable({ input_source: 'upload', source_reference: { synthetic_test_signal: true } }), false);
  assert.equal(explainable({ input_source: 'upload', source_reference: {} }), true);
  assert.equal(explainable({ input_source: 'deap_reference', source_reference: {} }), true);
  assert.equal(explainable(null), false);
});

test('contributions keep their sign and unit, and are never shown as a percentage probability', () => {
  assert.equal(fmtContribution(0.0123, 'probability'), '+1.23 pts');
  assert.equal(fmtContribution(-0.004, 'probability'), '−0.40 pts');
  assert.equal(fmtContribution(-0.25, 'log_odds'), '−0.250');
  assert.ok(!fmtContribution(0.5, 'probability').includes('%'));
});

test('feature labels come from the feature name parts; broadband features have no band', () => {
  assert.equal(featureLabel({ channel: 'O2', family: 'logpow', band: 'gamma' }), 'O2 · log band power · gamma');
  assert.equal(featureLabel({ channel: 'Cz', family: 'hjorth_mobility', band: 'broadband' }), 'Cz · Hjorth mobility');
});

test('ratings keep a fixed order and say when a dataset did not collect one', () => {
  const deap = ratingRows([{ name: 'Valence', value: 7, scale: [1, 9] }, { name: 'Arousal', value: 3, scale: [1, 9] },
    { name: 'Dominance', value: 5, scale: [1, 9] }, { name: 'Liking', value: 9, scale: [1, 9] }], 'DEAP');
  assert.deepEqual(deap.map((r) => r.name), ['Valence', 'Arousal', 'Dominance', 'Liking']);
  assert.equal(deap[0].share, 0.75);
  const faced = ratingRows([{ name: 'Valence', value: 3.5, scale: [0, 7] }, { name: 'Liking', value: 7, scale: [0, 7] }], 'FACED');
  assert.equal(faced.find((r) => r.name === 'Dominance').missing, 'Not collected in FACED');
  assert.ok(!faced.some((r) => r.name === 'Familiarity'));
});

test('colours follow the sign of a contribution', () => {
  assert.match(divergingColor(0.2, 0.2), /^rgba\(0, 184, 176/);
  assert.match(divergingColor(-0.2, 0.2), /^rgba\(77, 124, 254/);
});

test('the verification sentence reports the server-measured sum', () => {
  const text = verificationText({ output_kind: 'probability', base_value: 0.5, trial_output: 0.559, probability_high_valence: 0.559 });
  assert.equal(text, 'Average model score 50.0% + 5.9 pts from this recording\'s features = 55.9%, the saved result.');
});

test('affect rows keep a fixed order and say when FACED did not collect dominance', () => {
  const dim = (key, predicted, own) => ({ key, label: key, predicted, own_label: own, probability_high: predicted === 'lower' ? 0.4 : 0.6, evaluation: {} });
  const rows = affectRows({ dataset: 'FACED', dimensions: [dim('liking', 'lower', 'lower'), dim('valence', 'higher', 'lower'), dim('arousal', 'higher', null)] });
  assert.deepEqual(rows.map((r) => r.key), ['valence', 'arousal', 'dominance', 'liking']);
  assert.equal(rows[2].missing, 'Not collected in FACED');
  assert.equal(rows[0].agrees, false);
  assert.equal(rows[1].agrees, null); // no own rating (e.g. an upload)
  assert.equal(rows[3].agrees, true);
  assert.equal(affectRows({ dataset: 'DEAP', dimensions: [] })[2].missing, 'No model available');
  assert.equal(affectRows({ dataset: 'DEAP', dimensions: [dim('liking', null, 'higher')] })[3].agrees, null); // no usable signal: no verdict
});

test('percent formatting', () => {
  assert.equal(pct(undefined), '—');
  assert.equal(pct(0.546), '55%');
});

test('a decision that its calibrated probability contradicts is shown as uncertain', () => {
  assert.equal(verdict('higher', 0.47), 'uncertain');
  assert.equal(verdict('lower', 0.62), 'uncertain');
  assert.equal(verdict('higher', 0.55), 'higher');
  assert.equal(verdict('lower', 0.3), 'lower');
  assert.equal(verdict(null, 0.71), null);
});

test('an uncertain call is neither a match nor a miss', () => {
  const rows = affectRows({ dataset: 'FACED', dimensions: [{ key: 'valence', predicted: 'higher', probability_high: 0.47, own_label: 'lower', evaluation: {} }] });
  assert.equal(rows[0].agrees, null);
});

test('flashcard headlines: segment extremes, strongest cell and dominant band', () => {
  const view = { segment_probability_high: [0.5, 0.7, 0.3], kept_segments: [0, 2, 3], segment_seconds: 4 };
  const x = segmentExtremes(view);
  assert.deepEqual([x.high.i, x.high.label, x.low.i, x.low.label], [1, '8–12 s', 2, '12–16 s']);
  assert.equal(segmentExtremes({}), null);
  const cell = strongestCell(['Fp1', 'O1'], ['theta', 'alpha'], [[0.1, -0.9], [0.5, NaN]], true);
  assert.deepEqual([cell.channel, cell.band, cell.value], ['Fp1', 'alpha', -0.9]);
  assert.equal(strongestCell(['Fp1', 'O1'], ['theta', 'alpha'], [[0.1, -0.9], [0.5, NaN]]).channel, 'O1');
  assert.equal(dominantBand({ bands: ['theta', 'gamma'], values: [[0.1, -0.5], [0.2, 0.4]] }), 'gamma');
  assert.equal(dominantBand(null), null);
});
