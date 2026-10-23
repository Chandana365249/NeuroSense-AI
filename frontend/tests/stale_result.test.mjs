// Regression tests for stale results on Analyze EEG: an outcome is shown only for the selection that produced it.
import assert from 'node:assert/strict';
import { test } from 'node:test';

import { isSynthetic, NO_RESULT_MESSAGE, selectionKey, SYNTHETIC_DONE_MESSAGE, visibleOutcome } from '../src/analysis.js';

const deap = (participant, trial) => selectionKey('deap', { deap: { participant, trial } });
const synth = (id) => selectionKey('synthetic', { synthetic: { id } });
const upload = (id) => selectionKey('upload', { upload: { id } });
const P = { prediction: 'low_valence', probabilities: { high_valence: 0.491, low_valence: 0.509 } };
const resultFor = (key) => ({ result: { key, prediction: P }, error: null, pending: null });

test('keys: each source, participant, trial, file and generated signal is a distinct selection', () => {
  const keys = [deap('s01', '0'), deap('s01', '1'), deap('s09', '0'), synth(1), synth(2), upload(1), upload(2)];
  assert.equal(new Set(keys).size, keys.length);
  assert.equal(deap('s01', '0'), deap('s01', '0'));
  assert.equal(selectionKey('deap', { deap: { participant: '', trial: '' } }), 'deap:none');
  assert.equal(selectionKey('upload', {}), 'upload:none');
  assert.equal(selectionKey('synthetic', {}), 'synthetic:none');
});

test('synthetic -> DEAP: the synthetic result is not shown for the DEAP selection', () => {
  const state = resultFor(synth(1));
  assert.equal(visibleOutcome(state, synth(1)).kind, 'result');
  assert.equal(visibleOutcome(state, deap('s01', '0')).kind, 'empty');
});

test('DEAP -> synthetic: the DEAP result is not shown for the synthetic selection', () => {
  const state = resultFor(deap('s01', '0'));
  assert.equal(visibleOutcome(state, synth(3)).kind, 'empty');
  assert.equal(visibleOutcome(state, selectionKey('synthetic', {})).kind, 'empty');
  assert.equal(visibleOutcome(state, upload(1)).kind, 'empty');
});

test('changing participant or trial hides the earlier result', () => {
  const state = resultFor(deap('s01', '0'));
  assert.equal(visibleOutcome(state, deap('s01', '0')).kind, 'result');
  assert.equal(visibleOutcome(state, deap('s01', '1')).kind, 'empty');
  assert.equal(visibleOutcome(state, deap('s09', '0')).kind, 'empty');
});

test('a new upload or a newly generated synthetic signal does not inherit the previous result', () => {
  assert.equal(visibleOutcome(resultFor(upload(1)), upload(2)).kind, 'empty');
  assert.equal(visibleOutcome(resultFor(synth(1)), synth(2)).kind, 'empty');
});

test('a failed analysis shows its error, never a result; errors of other selections are hidden', () => {
  const failed = { result: null, error: { key: deap('s01', '2'), message: 'Internal error' }, pending: null };
  assert.deepEqual(visibleOutcome(failed, deap('s01', '2')), { kind: 'error', message: 'Internal error' });
  assert.equal(visibleOutcome(failed, deap('s01', '3')).kind, 'empty');
  // Even if a stale result were still held, an error for the current selection wins over it.
  const both = { result: { key: deap('s01', '2'), prediction: P }, error: { key: deap('s01', '2'), message: 'x' }, pending: null };
  assert.equal(visibleOutcome(both, deap('s01', '2')).kind, 'error');
});

test('loading is shown only for the selection being analysed; nothing else meanwhile', () => {
  const pending = { result: null, error: null, pending: deap('s09', '6') };
  assert.equal(visibleOutcome(pending, deap('s09', '6')).kind, 'loading');
  assert.equal(visibleOutcome(pending, synth(1)).kind, 'empty');
});

test('a result appears only after success for the current selection', () => {
  assert.equal(visibleOutcome({ result: null, error: null, pending: null }, deap('s01', '0')).kind, 'empty');
  const done = visibleOutcome(resultFor(deap('s01', '0')), deap('s01', '0'));
  assert.equal(done.kind, 'result');
  assert.equal(done.prediction, P);
  assert.equal(visibleOutcome(undefined, deap('s01', '0')).kind, 'empty');
});

test('messages: neutral empty state and neutral synthetic status', () => {
  assert.equal(NO_RESULT_MESSAGE, 'No result for this selection yet. Choose your input and click Analyze.');
  assert.equal(SYNTHETIC_DONE_MESSAGE, 'Synthetic pipeline test completed. This output is not an emotion estimate.');
  assert.doesNotMatch(SYNTHETIC_DONE_MESSAGE, /valence/i);
  assert.equal(isSynthetic({ input_source: 'upload', source_reference: { synthetic_test_signal: true } }), true);
  assert.equal(isSynthetic({ input_source: 'upload', source_reference: {} }), false);
  assert.equal(isSynthetic({ input_source: 'deap_reference', source_reference: { participant_id: 's01' } }), false);
});
