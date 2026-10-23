// Analyze EEG data flow and model-information wording. Run with: npm test
import assert from 'node:assert/strict';
import { test } from 'node:test';

import { demoOptions, resultContext, roleNotice, sourceText, trialLabel, validateUpload } from '../src/analysis.js';
import { authFailureMessage, localErrors } from '../src/authRules.js';
import { predictionLabel, sourceLabel } from '../src/components/format.js';
import { evaluationStatus, heldOutSummary, modelStatusLabel, NOT_RECORDED, trainingDataSummary } from '../src/modelInfo.js';
import { makeSyntheticSignal } from '../src/synthetic.js';

const REQ = { channels: 32, sampling_rate_hz: 128, minimum_seconds: 4 };
const FIXED = {
  recorded: true, kind: 'fixed_split', fit_participants: ['s02', 's03', 's16'], selection_participants: ['s16'],
  held_out_participants: ['s01', 's09'], fit_segments: 5258, fit_trials: null, note: 'recorded',
};
const TRUSTED = ['s01', 's02', 's03', 's05', 's07', 's08', 's09', 's11', 's15', 's16', 's22', 's25'];
const ROLES = { s01: 'held_out', s09: 'held_out', s11: 'held_out', s16: 'selection', s25: 'selection' };
const demoResponse = (overrides = {}) => ({
  model_version: 'v1', held_out_participants: ['s01', 's09', 's11'], trials_per_participant: 40,
  participants: TRUSTED.map((id) => ({ participant_id: id, trial_indices: Array.from({ length: 40 }, (_, i) => i), model_role: ROLES[id] || 'fit' })),
  unavailable: [], dataset_available: true, provenance: FIXED, note: 'Authorized participants readable on this server.', ...overrides,
});
const signal = (channels, samples, value = 1) => Array.from({ length: channels }, () => Array(samples).fill(value));

// ---------- participant dropdown ----------
test('dropdown: all 12 trusted participants from the API, with their trials and recorded roles', () => {
  const o = demoOptions({ status: 'ready', data: demoResponse() });
  assert.equal(o.status, 'ready');
  assert.deepEqual(o.participants.map((p) => p.id), TRUSTED);
  assert.ok(o.participants.every((p) => p.trials.length === 40 && p.trials[0] === 0 && p.trials[39] === 39));
  assert.equal(o.participants.find((p) => p.id === 's02').role, 'fit');
  assert.equal(o.participants.find((p) => p.id === 's09').role, 'held_out');
});

test('dropdown: the list is not limited to the model\'s held-out participants', () => {
  const o = demoOptions({ status: 'ready', data: demoResponse() });
  assert.ok(o.participants.length > demoResponse().held_out_participants.length);
});

test('dropdown: participants without readable trials are not offered', () => {
  const o = demoOptions({ status: 'ready', data: demoResponse({ participants: [{ participant_id: 's01', trial_indices: [], model_role: 'held_out' }] }) });
  assert.equal(o.status, 'none');
  assert.deepEqual(o.participants, []);
});

test('dropdown: an API failure is reported, never replaced by fallback participants', () => {
  const o = demoOptions({ status: 'error', error: 'Request failed (500)' });
  assert.equal(o.status, 'error');
  assert.deepEqual(o.participants, []);
  assert.match(o.message, /could not be loaded/);
  assert.equal(demoOptions({ status: 'loading' }).status, 'loading');
  assert.equal(demoOptions(undefined).participants.length, 0);
});

test('dropdown: a missing role is shown as unknown, never guessed', () => {
  const o = demoOptions({ status: 'ready', data: demoResponse({ participants: [{ participant_id: 's02', trial_indices: [0] }] }) });
  assert.equal(o.participants[0].role, 'unknown');
});

// ---------- evaluation status per participant ----------
test('status: training participants are in-sample demonstrations', () => {
  for (const role of ['fit', 'selection', 'train', 'validation']) {
    const n = roleNotice(role);
    assert.equal(n.kind, 'in_sample');
    assert.equal(n.text, 'This participant contributed to model training. Treat this prediction as an in-sample demonstration, not an independent test.');
  }
});

test('status: held-out participants are not an untouched final test', () => {
  for (const role of ['held_out', 'test']) {
    const n = roleNotice(role);
    assert.equal(n.kind, 'held_out');
    assert.equal(n.text, 'This participant was held out from model fitting, but has already been used in development analyses. This is not an untouched final test.');
  }
});

test('status: unrecorded roles say so and never claim independence', () => {
  for (const role of [undefined, 'unknown', 'something-else']) {
    const n = roleNotice(role);
    assert.equal(n.kind, 'unknown');
    assert.doesNotMatch(n.text, /is an independent test|untouched final test\./);
  }
  assert.equal(roleNotice('not_used').kind, 'not_used');
  assert.equal(heldOutSummary({ recorded: false }), null);
  assert.equal(heldOutSummary(FIXED), 's01, s09');
});

test('provenance: training data counts come from provenance only, unknown counts say so', () => {
  assert.equal(trainingDataSummary(FIXED), '3 participants · 5,258 4-second EEG segments · trial count not recorded');
  assert.equal(trainingDataSummary({ ...FIXED, fit_trials: 120 }), '3 participants · 5,258 4-second EEG segments · 120 trials');
  assert.equal(trainingDataSummary({ recorded: false }), `${NOT_RECORDED} in this model's metadata`);
  assert.equal(trainingDataSummary(undefined), `${NOT_RECORDED} in this model's metadata`);
});

// ---------- trial indices ----------
test('trials: one label everywhere, 1-based number with the stored 0-based index', () => {
  assert.equal(trialLabel(0), 'Trial 1 (index 0)');
  assert.equal(trialLabel(39), 'Trial 40 (index 39)');
  const p = { input_source: 'deap_reference', source_reference: { participant_id: 's09', trial_index: 6 } };
  assert.equal(sourceText(p), 'DEAP s09 · Trial 7 (index 6)');
  assert.equal(sourceLabel(p), sourceText(p)); // history, recent list and result use the same text
});

// ---------- upload validation ----------
test('upload: valid file passes with the model rate assumed only when absent', () => {
  const r = validateUpload({ signal: signal(32, 512) }, REQ, 3);
  assert.equal(r.eeg.sampling_rate, 128);
  assert.equal(r.rateGiven, false);
  assert.equal(r.seconds, 4);
  assert.equal(r.eeg.synthetic_test_signal, undefined);
  assert.equal(validateUpload(signal(32, 512), REQ).eeg.signal.length, 32); // bare array form
});

test('upload: clear errors for wrong shape, rate, values and length', () => {
  assert.throws(() => validateUpload({ data: [] }, REQ), /list of channels/);
  assert.throws(() => validateUpload({ signal: signal(14, 512) }, REQ), /14 channels; the model needs exactly 32/);
  assert.throws(() => validateUpload({ signal: [...signal(31, 512), Array(500).fill(1)] }, REQ), /same number of samples/);
  assert.throws(() => validateUpload({ signal: signal(32, 512, NaN) }, REQ), /must be a number/);
  assert.throws(() => validateUpload({ signal: signal(32, 512), sampling_rate: 256 }, REQ), /256 Hz; the model needs 128 Hz/);
  assert.throws(() => validateUpload({ signal: signal(32, 400) }, REQ), /at least 4 s is needed/);
  assert.throws(() => validateUpload({ signal: signal(32, 600), includes_baseline: true }, REQ, 3), /at least 7 s is needed \(including the 3 s baseline\)/);
});

// ---------- synthetic vs real ----------
test('synthetic: generated file is valid input and is flagged for the server', () => {
  const r = validateUpload(makeSyntheticSignal({ seconds: 8 }), REQ);
  assert.equal(r.synthetic, true);
  assert.equal(r.eeg.synthetic_test_signal, true);
});

test('result context: synthetic, held-out, in-sample and unknown are distinguished', () => {
  const synth = resultContext({ input_source: 'upload', source_reference: { synthetic_test_signal: true } });
  assert.equal(synth.kind, 'synthetic');
  assert.match(synth.summary, /not an emotion estimate/);
  assert.equal(sourceText({ input_source: 'upload', source_reference: { synthetic_test_signal: true } }), 'Synthetic test signal');
  assert.equal(resultContext({ input_source: 'upload', source_reference: { channels: 32 } }).kind, 'upload');
  const deap = (split) => resultContext({ input_source: 'deap_reference', source_reference: { participant_id: 's01', participant_split: split } });
  assert.equal(deap('test').fairness, 'held_out');
  assert.match(deap('test').notice.text, /not an untouched final test/);
  assert.equal(deap('train').fairness, 'trained');
  assert.match(deap('train').notice.text, /in-sample demonstration/);
  assert.equal(deap('validation').fairness, 'trained');
  assert.equal(deap('not_used').fairness, 'unknown');
  assert.equal(deap(undefined).fairness, 'unknown');
});

// ---------- model labels ----------
test('models: active, candidate and inactive labels', () => {
  assert.equal(modelStatusLabel({ is_active: true, evaluation: {} }), 'Active');
  assert.equal(modelStatusLabel({ is_active: false, evaluation: { protocol_version: '1.1' } }), 'Candidate (inactive)');
  assert.equal(modelStatusLabel({ is_active: false, provenance: { kind: 'protocol_all_development' } }), 'Candidate (inactive)');
  assert.equal(modelStatusLabel({ is_active: false, provenance: FIXED }), 'Inactive');
});

test('models: evaluation status says exploratory, historical or none', () => {
  assert.equal(evaluationStatus({ evaluation: { protocol_version: '1.1', n_participants: 12 } }).label, 'Exploratory');
  assert.match(evaluationStatus({ evaluation: { protocol_version: '1.1', n_participants: 12 } }).detail, /participant-independent/);
  assert.equal(evaluationStatus({ test_metrics: { balanced_accuracy: 0.52 } }).label, 'Historical');
  assert.equal(evaluationStatus({ test_metrics: { balanced_accuracy: null } }).label, 'No evaluation recorded');
  assert.equal(evaluationStatus({}).label, 'No evaluation recorded');
});

test('lists: synthetic results are never shown as a valence category', () => {
  assert.equal(predictionLabel({ input_source: 'upload', prediction: 'low_valence', source_reference: { synthetic_test_signal: true } }),
    'Synthetic test (not an estimate)');
  assert.equal(predictionLabel({ input_source: 'upload', prediction: 'low_valence', source_reference: {} }), 'Lower self-reported valence');
  assert.equal(predictionLabel({ input_source: 'deap_reference', prediction: 'high_valence' }), 'Higher self-reported valence');
});

// ---------- sign-in form rules ----------
test('sign-in: missing or malformed fields are caught before any request', () => {
  assert.deepEqual(localErrors('login', '', ''), { email: 'Enter your email address.', password: 'Enter your password.' });
  assert.match(localErrors('login', 'not-an-email', 'x').email, /valid email address/);
  assert.deepEqual(localErrors('login', 'a@example.com', 'x'), {}); // a short password is the server's call at sign-in
  assert.equal(localErrors('register', 'a@example.com', 'short').password, 'Use at least 8 characters.');
});

test('sign-in: failures get understandable messages', () => {
  assert.match(authFailureMessage({ status: 401, message: 'Incorrect email or password' }, 'login'), /Incorrect email or password/);
  assert.match(authFailureMessage({ status: 0 }, 'login'), /unreachable/);
  assert.match(authFailureMessage({ status: 409 }, 'register'), /already exists/);
  assert.match(authFailureMessage({ status: 429 }, 'login'), /Too many attempts/);
});
