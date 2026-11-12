// Unit tests for the frontend's pure logic. Run with: npm test  (Node's built-in test runner, no dependencies).
import assert from 'node:assert/strict';
import { test } from 'node:test';

import { failureMessage, fieldErrorsFrom, REASON_MAX, REASON_MIN, validate } from '../src/activationRules.js';
import { classLabel, featureLabel, NA, num, pct } from '../src/components/format.js';
import { makeSyntheticSignal, SYNTHETIC_FILE_NAME } from '../src/synthetic.js';

// ---------- activation dialog rules ----------
test('activation: exact version confirmation is required', () => {
  const reason = 'A valid audit reason';
  assert.deepEqual(validate('v1', 'v1', reason), {});
  assert.match(validate('v1', '', reason).confirm, /exactly as shown/);
  assert.match(validate('v1', 'V1', reason).confirm, /does not match/);
  assert.match(validate('v1', 'v1 ', reason).confirm, /does not match/);
});

test('activation: reason must be 10-500 characters after trimming', () => {
  assert.equal(REASON_MIN, 10);
  assert.equal(REASON_MAX, 500);
  assert.match(validate('v', 'v', '   123456789   ').reason, /at least 10 characters \(9 after trimming/);
  assert.equal(validate('v', 'v', '  1234567890  ').reason, undefined);
  assert.equal(validate('v', 'v', 'x'.repeat(500)).reason, undefined);
  assert.match(validate('v', 'v', 'x'.repeat(501)).reason, /at most 500 characters/);
});

test('activation: server field errors map to fields without echoing values', () => {
  const err = { details: [
    { loc: ['body', 'reason'], msg: 'Value error, Reason appears to contain a credential or secret; remove it and try again' },
    { loc: ['body', 'confirm_version'], msg: 'String should have at least 1 character' },
  ] };
  assert.deepEqual(fieldErrorsFrom(err), {
    reason: 'Reason appears to contain a credential or secret; remove it and try again',
    confirm: 'String should have at least 1 character',
  });
  assert.deepEqual(fieldErrorsFrom({}), {});
});

test('activation: every failure status gets an explicit, safe message', () => {
  assert.match(failureMessage({ status: 403 }).body, /Nothing was changed/);
  assert.match(failureMessage({ status: 404 }).title, /not found/);
  assert.match(failureMessage({ status: 409 }).body, /training run or another activation/);
  assert.match(failureMessage({ status: 0 }).body, /Reload the registry/);
  for (const status of [500, 502, 503]) {
    assert.match(failureMessage({ status }).body, /previous model should remain active/);
  }
  const known = failureMessage({ status: 422, message: 'The model file failed its integrity check; it was not activated.' });
  assert.match(known.body, /integrity check/);
  assert.match(known.body, /current model remains active/);
  const unknown = failureMessage({ status: 422, message: 'Traceback: C:\\secret\\path.py line 3' });
  assert.doesNotMatch(unknown.body, /Traceback|secret|path/); // unknown server text is never shown
});

// ---------- display helpers ----------
test('format: approved labels and missing values', () => {
  assert.equal(classLabel('high_valence'), 'Higher self-reported valence');
  assert.equal(classLabel('low_valence'), 'Lower self-reported valence');
  assert.equal(pct(0.5234), '52.3%');
  assert.equal(num(0.5), '0.500');
  assert.equal(pct(null), NA);
  assert.equal(num(undefined), NA);
});

test('format: feature labels', () => {
  assert.equal(featureLabel({ channel: 'O2', family: 'logpow', band: 'gamma' }), 'O2 · log band power · gamma');
  assert.equal(featureLabel({ channel: 'Fp2-Fp1', family: 'asym', band: 'theta' }), 'Fp2-Fp1 · right–left asymmetry · theta');
  assert.equal(featureLabel({ channel: 'Cz', family: 'hjorth_mobility', band: null }), 'Cz · Hjorth mobility');
});

// ---------- synthetic test signal ----------
test('synthetic signal: shape, labelling and determinism', () => {
  const a = makeSyntheticSignal({ seconds: 5 });
  const b = makeSyntheticSignal({ seconds: 5 });
  assert.equal(a.synthetic_test_fixture, true);
  assert.match(a.note, /not recorded EEG, not DEAP data/);
  assert.match(SYNTHETIC_FILE_NAME, /NOT_REAL_EEG/);
  assert.equal(a.signal.length, 32);
  assert.ok(a.signal.every((row) => row.length === 640 && row.every(Number.isFinite)));
  assert.deepEqual(a, b);
  assert.notDeepEqual(makeSyntheticSignal({ seconds: 5, seed: 1 }).signal[0], a.signal[0]);
});
