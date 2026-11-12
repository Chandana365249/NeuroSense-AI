import assert from 'node:assert/strict';
import test from 'node:test';
import { ciText, facedResultKey, filterParticipants, pct1, visibleFacedResult } from '../src/faced.js';

const rows = [
  { participant_id: 'sub-000', label: 'Participant 000' },
  { participant_id: 'sub-012', label: 'Participant 012' },
  { participant_id: 'sub-120', label: 'Participant 120' },
];

test('participant search matches the original ID, the number and the readable label', () => {
  assert.deepEqual(filterParticipants(rows, '').map((r) => r.participant_id), ['sub-000', 'sub-012', 'sub-120']);
  assert.deepEqual(filterParticipants(rows, 'sub-012').map((r) => r.participant_id), ['sub-012']);
  assert.deepEqual(filterParticipants(rows, '12').map((r) => r.participant_id), ['sub-012', 'sub-120']);
  assert.deepEqual(filterParticipants(rows, 'participant 120').map((r) => r.participant_id), ['sub-120']);
  assert.deepEqual(filterParticipants(rows, 'xyz'), []);
});

test('a FACED result key needs participant, video and model', () => {
  assert.equal(facedResultKey('sub-001', 3, 'faced-valence-lr'), 'faced|sub-001|3|faced-valence-lr');
  assert.equal(facedResultKey('sub-001', null, 'm'), null);
  assert.equal(facedResultKey('sub-001', 3, null), null);
});

test('a result is shown only for the selection it was computed for', () => {
  const outcome = { key: facedResultKey('sub-001', 3, 'm'), status: 'done' };
  assert.equal(visibleFacedResult(outcome, facedResultKey('sub-001', 3, 'm')), outcome);
  assert.equal(visibleFacedResult(outcome, facedResultKey('sub-001', 4, 'm')), null); // other video
  assert.equal(visibleFacedResult(outcome, facedResultKey('sub-002', 3, 'm')), null); // other participant
  assert.equal(visibleFacedResult(outcome, facedResultKey('sub-001', 3, 'other-model')), null); // other model
  assert.equal(visibleFacedResult(outcome, null), null);
});

test('percentages and intervals are formatted without inventing values', () => {
  assert.equal(pct1(0.6234), '62.3%');
  assert.equal(pct1(null), 'n/a');
  assert.equal(ciText([0.5, 0.7]), '50.0% to 70.0%');
  assert.equal(ciText(undefined), 'n/a');
});
