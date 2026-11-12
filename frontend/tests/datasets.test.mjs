import assert from 'node:assert/strict';
import { test } from 'node:test';

import { datasetKind, resultContext, sourceText } from '../src/analysis.js';
import { predictionLabel } from '../src/components/format.js';
import { datasetPath, deapParticipantFrom, facedSelectionFrom, filterDeapParticipants, probabilityNote, resolveDataset } from '../src/datasets.js';
import { parseHash } from '../src/router.js';

test('the URL decides the dataset, then the stored choice, then DEAP', () => {
  assert.equal(resolveDataset('faced', 'deap'), 'faced');
  assert.equal(resolveDataset(null, 'faced'), 'faced');
  assert.equal(resolveDataset(null, null), 'deap');
  assert.equal(resolveDataset('other', 'also-other'), 'deap');
});

test('a dataset link round-trips through the router', () => {
  const { path, query } = parseHash(`#${datasetPath('/analyze', 'faced', { participant: 'sub-012', video: 7 })}`);
  assert.equal(path, '/analyze');
  assert.equal(query.get('dataset'), 'faced');
  assert.deepEqual(facedSelectionFrom(query), { participant: 'sub-012', video: 7 });
});

test('malformed selections from a link are ignored', () => {
  const q = (s) => new URLSearchParams(s);
  assert.deepEqual(facedSelectionFrom(q('participant=../x&video=99')), { participant: null, video: null });
  assert.deepEqual(facedSelectionFrom(q('participant=s01&video=0')), { participant: null, video: null });
  assert.equal(deapParticipantFrom(q('participant=sub-001')), '');
  assert.equal(deapParticipantFrom(q('participant=s07')), 's07');
});

test('DEAP search finds a participant by ID or number', () => {
  const rows = [{ id: 's01' }, { id: 's07' }, { id: 's11' }];
  assert.deepEqual(filterDeapParticipants(rows, 's07'), [{ id: 's07' }]);
  assert.deepEqual(filterDeapParticipants(rows, '7'), [{ id: 's07' }]);
  assert.deepEqual(filterDeapParticipants(rows, '07'), [{ id: 's07' }]);
  assert.deepEqual(filterDeapParticipants(rows, ''), rows);
  assert.deepEqual(filterDeapParticipants(rows, 'zzz'), []);
  assert.deepEqual(filterDeapParticipants([{ participant_id: 's11' }], '11', (r) => r.participant_id), [{ participant_id: 's11' }]);
});

test('a probability near 50% is never presented as a decision', () => {
  assert.match(probabilityNote(0.513), /undecided/);
  assert.match(probabilityNote(0.42), /slightly/);
  assert.match(probabilityNote(0.9), /does not mean/);
  assert.equal(probabilityNote(null), null);
});

const FACED_RECORD = { input_source: 'faced_reference', prediction: 'high_valence',
  source_reference: { dataset: 'FACED', participant_id: 'sub-012', video_index: 7, evaluation_status: 'strict_held_out', evaluation_text: 'Strict held-out test.' } };
const SYNTHETIC_RECORD = { input_source: 'upload', prediction: 'high_valence', source_reference: { synthetic_test_signal: true } };

test('every stored record is one of four kinds, and FACED is never taken for an upload', () => {
  assert.equal(datasetKind({ input_source: 'deap_reference', source_reference: {} }), 'deap');
  assert.equal(datasetKind(FACED_RECORD), 'faced');
  assert.equal(datasetKind(SYNTHETIC_RECORD), 'synthetic');
  assert.equal(datasetKind({ input_source: 'upload', source_reference: {} }), 'upload');
});

test('a FACED record names its participant and video, and carries its evaluation status', () => {
  assert.equal(sourceText(FACED_RECORD), 'FACED sub-012 · Video 7');
  const ctx = resultContext(FACED_RECORD);
  assert.equal(ctx.kind, 'faced');
  assert.equal(ctx.notice.title, 'Strict held-out');
  assert.equal(ctx.notice.text, 'Strict held-out test.');
  assert.match(resultContext({ input_source: 'faced_reference', source_reference: {} }).notice.title, /not recorded/);
});

test('a synthetic record is never shown as an estimate', () => {
  assert.equal(predictionLabel(SYNTHETIC_RECORD), 'Synthetic test (not an estimate)');
  assert.equal(resultContext(SYNTHETIC_RECORD).kind, 'synthetic');
  assert.notEqual(predictionLabel(FACED_RECORD), 'Synthetic test (not an estimate)');
});
