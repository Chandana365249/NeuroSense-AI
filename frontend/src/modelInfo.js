// Plain-language descriptions of a model's recorded training provenance and evaluation status.
// Input is the `provenance` object from GET /model-info or GET /models; missing records stay "not recorded".

export const NOT_RECORDED = 'Not recorded';

const plural = (n, one, many) => `${n.toLocaleString()} ${n === 1 ? one : many}`;

/** Short "trained on" line, e.g. "9 participants · 5,258 EEG segments · trials not recorded". */
export function trainingDataSummary(prov) {
  if (!prov || !prov.recorded) return `${NOT_RECORDED} in this model's metadata`;
  const parts = [plural(prov.fit_participants.length, 'participant', 'participants')];
  parts.push(typeof prov.fit_segments === 'number' ? plural(prov.fit_segments, '4-second EEG segment', '4-second EEG segments')
    : 'segments not recorded');
  parts.push(typeof prov.fit_trials === 'number' ? plural(prov.fit_trials, 'trial', 'trials') : 'trial count not recorded');
  return parts.join(' · ');
}

/** Participants never used for fitting or selection, as recorded; null when unknown. */
export function heldOutSummary(prov) {
  if (!prov || !prov.recorded) return null;
  if (!prov.held_out_participants.length) return 'None (trained on every development participant)';
  return prov.held_out_participants.join(', ');
}

/**
 * Status of a model's evaluation numbers. `model` is a /models row or /model-info + /metrics data with
 * `evaluation` (protocol LOPO summary) and/or `test_metrics` (historical fixed split).
 * Returns { label, detail } where label is "Exploratory", "Historical" or "No evaluation recorded".
 */
export function evaluationStatus(model) {
  if (model?.evaluation) {
    return {
      label: 'Exploratory',
      detail: `Protocol v${model.evaluation.protocol_version}, leave-one-participant-out on ${model.evaluation.n_participants} `
        + 'participants (participant-independent: each person is scored by a model that never saw them). '
        + 'Not a final test result.',
    };
  }
  const tm = model?.test_metrics;
  if (tm && Object.values(tm).some((v) => v != null)) {
    return {
      label: 'Historical',
      detail: 'Pre-protocol fixed split, recorded at training time (participant-independent: test participants were '
        + 'not used for training). Those participants are now used for development, so this is not current evidence.',
    };
  }
  return { label: 'No evaluation recorded', detail: 'The registry holds no evaluation for this model.' };
}

/** Registry status label: the active model, a protocol candidate awaiting activation, or an older inactive model. */
export function modelStatusLabel(model) {
  if (model?.is_active) return 'Active';
  if (model?.evaluation || model?.provenance?.kind === 'protocol_all_development') return 'Candidate (inactive)';
  return 'Inactive';
}
