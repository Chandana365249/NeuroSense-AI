// Pure logic for the Analyze EEG workflow. Every participant, trial and requirement shown comes from the API
// (GET /dataset/demo-trials and GET /model-info); nothing here substitutes values when the API has none.
import { STATUS_SHORT, STATUS_TONE } from './faced.js';

/** Human trial label. DEAP stores trials by zero-based index; people count from 1, so both are shown. */
export const trialLabel = (index) => `Trial ${index + 1} (index ${index})`;

/**
 * Evaluation status of one participant for the active model, from its recorded role (GET /dataset/demo-trials
 * `model_role`, or a stored result's `participant_split`). Never inferred: unrecorded roles say so.
 */
const ROLE_NOTICES = {
  in_sample: {
    tone: 'warning', title: 'In-sample demonstration',
    text: 'This participant contributed to model training. Treat this prediction as an in-sample demonstration, not an independent test.',
  },
  held_out: {
    tone: 'info', title: 'Held out, but not an untouched test',
    text: 'This participant was held out from model fitting, but has already been used in development analyses. This is not an untouched final test.',
  },
  not_used: {
    tone: 'info', title: 'Not in the recorded training groups',
    text: "The active model's recorded metadata does not list this participant in its training, selection or test groups.",
  },
  unknown: {
    tone: 'info', title: 'Training role not recorded',
    text: 'The active model does not record whether this participant was used in its training, so this result cannot be treated as an independent test.',
  },
};
const ROLE_KIND = { fit: 'in_sample', selection: 'in_sample', train: 'in_sample', validation: 'in_sample',
  held_out: 'held_out', test: 'held_out', not_used: 'not_used' };

/** { kind, tone, title, text } for a recorded role; anything unrecognised is 'unknown'. */
export function roleNotice(role) {
  const kind = ROLE_KIND[role] || 'unknown';
  return { kind, ...ROLE_NOTICES[kind] };
}

/**
 * What the DEAP picker can offer, from the demo-trials request state: every authorized participant readable on the
 * server, with their actual trial indices and recorded role for the active model.
 * demo: { status: 'loading' | 'ready' | 'error', data?, error? }
 * Returns { status: 'loading' | 'error' | 'ready' | 'none', participants: [{ id, trials, role }], message }
 */
export function demoOptions(demo) {
  if (!demo || demo.status === 'loading') return { status: 'loading', participants: [], message: null };
  if (demo.status === 'error') {
    return {
      status: 'error', participants: [],
      message: 'The list of DEAP participants could not be loaded from the server, so none can be offered. '
        + 'You can still upload a file or try the synthetic test signal.',
    };
  }
  const data = demo.data || {};
  const participants = (data.participants || [])
    .filter((p) => Array.isArray(p.trial_indices) && p.trial_indices.length)
    .map((p) => ({ id: p.participant_id, trials: p.trial_indices, role: p.model_role || 'unknown' }));
  return { status: participants.length ? 'ready' : 'none', participants, message: data.note || null };
}

/**
 * Validates a parsed upload against the active model's input requirements (from /model-info).
 * Returns { eeg, synthetic, rateGiven, seconds } or throws an Error with a plain-language message.
 */
export function validateUpload(parsed, requirements, baselineSeconds = 0) {
  const signal = Array.isArray(parsed) ? parsed : parsed?.signal;
  if (!Array.isArray(signal) || !signal.length || !signal.every(Array.isArray)) {
    throw new Error('The file must hold the EEG as a list of channels, each a list of numbers: '
      + '{"signal": [[...], [...], ...]} or just [[...], [...], ...].');
  }
  const channels = requirements?.channels;
  const rate = requirements?.sampling_rate_hz;
  if (typeof channels === 'number' && signal.length !== channels) {
    throw new Error(`The file has ${signal.length} channel${signal.length === 1 ? '' : 's'}; the model needs exactly `
      + `${channels} (DEAP channel order).`);
  }
  const length = signal[0].length;
  if (!signal.every((ch) => ch.length === length)) throw new Error('Every channel must have the same number of samples.');
  if (!signal.every((ch) => ch.every((v) => typeof v === 'number' && Number.isFinite(v)))) {
    throw new Error('Every sample must be a number (no text, empty values, NaN or Infinity).');
  }
  const rateGiven = typeof parsed?.sampling_rate === 'number';
  if (rateGiven && typeof rate === 'number' && parsed.sampling_rate !== rate) {
    throw new Error(`The file says it was sampled at ${parsed.sampling_rate} Hz; the model needs ${rate} Hz. `
      + 'Resampling is not supported, so please resample the recording first.');
  }
  const includesBaseline = parsed?.includes_baseline === true;
  const minSeconds = (requirements?.minimum_seconds || 0) + (includesBaseline ? baselineSeconds : 0);
  if (typeof rate === 'number' && length < Math.ceil(minSeconds * rate)) {
    throw new Error(`The recording is ${(length / rate).toFixed(1)} s long; at least ${minSeconds} s is needed`
      + `${includesBaseline ? ` (including the ${baselineSeconds} s baseline)` : ''}.`);
  }
  const synthetic = parsed?.synthetic_test_fixture === true;
  return {
    eeg: {
      signal,
      sampling_rate: rateGiven ? parsed.sampling_rate : rate,
      includes_baseline: includesBaseline,
      ...(synthetic ? { synthetic_test_signal: true } : {}),
    },
    synthetic,
    rateGiven,
    seconds: typeof rate === 'number' ? length / rate : null,
  };
}

/** Which kind of record a stored prediction is. Only these four exist, and they are never mixed up in lists. */
export function datasetKind(p) {
  if (p?.input_source === 'deap_reference') return 'deap';
  if (p?.input_source === 'faced_reference') return 'faced';
  return p?.source_reference?.synthetic_test_signal ? 'synthetic' : 'upload';
}

export const DATASET_LABELS = { deap: 'DEAP', faced: 'FACED', upload: 'Uploaded file', synthetic: 'Synthetic test' };

/**
 * How to read a stored result. kind: 'synthetic' | 'deap' | 'faced' | 'upload'.
 * fairness (DEAP only): 'unseen' | 'trained' | 'unknown', from the participant's recorded role for that model.
 */
export function resultContext(prediction) {
  const ref = prediction?.source_reference || {};
  if (prediction?.input_source === 'upload' && ref.synthetic_test_signal) {
    return { kind: 'synthetic', fairness: null, notice: null,
      summary: 'Synthetic test signal: generated noise, not recorded EEG. This output is not an emotion estimate.' };
  }
  if (prediction?.input_source === 'deap_reference') {
    const notice = roleNotice(ref.participant_split);
    const fairness = { in_sample: 'trained', held_out: 'held_out' }[notice.kind] || 'unknown';
    return { kind: 'deap', fairness, notice, summary: notice.text };
  }
  if (prediction?.input_source === 'faced_reference') {
    const status = ref.evaluation_status;
    const notice = { tone: STATUS_TONE[status] === 'ok' ? 'info' : 'warning', title: STATUS_SHORT[status] || 'Evaluation status not recorded',
      text: ref.evaluation_text || 'This record does not state how the recording relates to the model\'s training data.' };
    return { kind: 'faced', fairness: null, notice, summary: notice.text };
  }
  return { kind: 'upload', fairness: null, notice: null, summary: 'Your uploaded recording.' };
}

/** One-line source label used in lists (recent analyses, history). */
export function sourceText(p) {
  const ref = p?.source_reference;
  if (p?.input_source === 'deap_reference' && ref) return `DEAP ${ref.participant_id} · ${trialLabel(ref.trial_index)}`;
  if (p?.input_source === 'faced_reference' && ref) return `FACED ${ref.participant_id} · Video ${ref.video_index}`;
  if (p?.input_source === 'upload' && ref?.synthetic_test_signal) return 'Synthetic test signal';
  return 'Uploaded EEG file';
}

/**
 * Identity of the current input selection. A result, error or loading state is shown only while its key equals the
 * key of the current selection, so nothing produced for one input can appear next to another.
 * source: 'deap' | 'upload' | 'synthetic'; deap: { participant, trial }; upload / synthetic: { id } of the staged input.
 */
export function selectionKey(source, { deap, upload, synthetic } = {}) {
  if (source === 'deap') return deap?.participant && deap?.trial !== '' && deap?.trial != null ? `deap:${deap.participant}:${deap.trial}` : 'deap:none';
  if (source === 'upload') return upload ? `upload:${upload.id}` : 'upload:none';
  if (source === 'synthetic') return synthetic ? `synthetic:${synthetic.id}` : 'synthetic:none';
  return 'none';
}

/**
 * What the Result card shows for the current selection.
 * state: { result: { key, prediction } | null, error: { key, message } | null, pending: key | null }
 * Returns { kind: 'loading' | 'error' | 'result' | 'empty', prediction?, message? }.
 */
export function visibleOutcome(state, currentKey) {
  if (state?.pending && state.pending === currentKey) return { kind: 'loading' };
  if (state?.error && state.error.key === currentKey) return { kind: 'error', message: state.error.message };
  if (state?.result && state.result.key === currentKey) return { kind: 'result', prediction: state.result.prediction, eeg: state.result.eeg || null };
  return { kind: 'empty' };
}

export const NO_RESULT_MESSAGE = 'No result for this selection yet. Choose your input and click Analyze.';
export const SYNTHETIC_DONE_MESSAGE = 'Synthetic pipeline test completed. This output is not an emotion estimate.';

/** True for a stored result of the app's generated synthetic test signal. */
export const isSynthetic = (p) => p?.input_source === 'upload' && p?.source_reference?.synthetic_test_signal === true;

/** Counts for the summary tiles: totals per input type and the time of the latest analysis. */
export function dashboardSummary(items) {
  const counts = { deap: 0, faced: 0, upload: 0, synthetic: 0 };
  let latest = null;
  for (const p of items) {
    counts[datasetKind(p)] += 1;
    const d = new Date(p.created_at);
    if (!Number.isNaN(d.getTime()) && (!latest || d > latest)) latest = d;
  }
  const real = items.filter((p) => !isSynthetic(p));
  const higher = real.filter((p) => p.prediction === 'high_valence').length;
  return { total: items.length, counts, latest, real: real.length, higher, lower: real.length - higher };
}
