// Pure helpers for the FACED explorer (unit-tested in tests/faced.test.mjs).

/** Case-insensitive search over the original ID (sub-012), the number (012, 12) and the readable label. */
export function filterParticipants(rows, query) {
  const q = String(query || '').trim().toLowerCase();
  if (!q) return rows;
  const digits = q.replace(/^sub-?/, '').replace(/^participant\s*/, '');
  return rows.filter((r) => {
    const num = r.participant_id.split('-').pop();
    return r.participant_id.toLowerCase().includes(q) || r.label.toLowerCase().includes(q)
      || (/^\d+$/.test(digits) && Number(num) === Number(digits));
  });
}

/** A FACED result belongs to exactly one (participant, video, model) selection. */
export function facedResultKey(participantId, videoIndex, modelName) {
  if (!participantId || !videoIndex || !modelName) return null;
  return `faced|${participantId}|${videoIndex}|${modelName}`;
}

/** Only show a result whose key equals the current selection; anything else is stale and hidden. */
export function visibleFacedResult(outcome, currentKey) {
  return outcome && currentKey && outcome.key === currentKey ? outcome : null;
}

export const ROLE_TEXT = {
  final_test: 'Final test participant',
  development: 'Development participant',
  unassigned: 'Not assigned',
};

export const STATUS_SHORT = {
  strict_held_out: 'Strict held-out',
  unseen_person_seen_clip: 'Unseen person, clip seen in training',
  seen_person_unseen_clip: 'Person seen in training, clip held out',
  in_sample: 'In-sample (training data)',
};

export const STATUS_TONE = {
  strict_held_out: 'ok', unseen_person_seen_clip: 'neutral', seen_person_unseen_clip: 'neutral', in_sample: 'warn',
};

export function pct1(x) {
  return typeof x === 'number' ? `${(x * 100).toFixed(1)}%` : 'n/a';
}

export function ciText(ci) {
  return Array.isArray(ci) && ci.length === 2 ? `${pct1(ci[0])} to ${pct1(ci[1])}` : 'n/a';
}
