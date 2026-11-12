// The dataset choice shared by Analyze, Dataset and Model & Evidence (unit-tested in tests/datasets.test.mjs).
// DEAP and FACED have separate pipelines and models; a page shows exactly one of them at a time.

export const DATASETS = [
  { id: 'deap', label: 'DEAP (active model)' },
  { id: 'faced', label: 'FACED (separate research dataset)' },
];

const STORAGE_KEY = 'neurosense_dataset_tab';
const isDataset = (id) => DATASETS.some((d) => d.id === id);

/** The URL wins, so links are exact; then the last choice in this tab; DEAP is the default. */
export function resolveDataset(fromQuery, stored) {
  if (isDataset(fromQuery)) return fromQuery;
  return isDataset(stored) ? stored : 'deap';
}

export function readStoredDataset() {
  try { return sessionStorage.getItem(STORAGE_KEY); } catch { return null; }
}

export function storeDataset(id) {
  try { sessionStorage.setItem(STORAGE_KEY, id); } catch { /* storage unavailable; the URL still carries the choice */ }
}

export function datasetPath(path, id, extra = {}) {
  const query = new URLSearchParams({ dataset: id, ...extra });
  return `${path}?${query}`;
}

/** A FACED selection taken from a link; anything malformed is ignored rather than sent to the server. */
export function facedSelectionFrom(query) {
  const participant = query.get('participant');
  const video = Number(query.get('video'));
  return {
    participant: /^sub-\d{3}$/.test(participant || '') ? participant : null,
    video: Number.isInteger(video) && video >= 1 && video <= 28 ? video : null,
  };
}

export function deapParticipantFrom(query) {
  const participant = query.get('participant');
  return /^s\d{2}$/.test(participant || '') ? participant : '';
}

/** Case-insensitive search over DEAP IDs: s01, 01 and 1 all find the same participant. */
export function filterDeapParticipants(rows, query, idOf = (r) => r.id) {
  const q = String(query || '').trim().toLowerCase();
  if (!q) return rows;
  const digits = q.replace(/^s/, '');
  return rows.filter((r) => {
    const id = idOf(r).toLowerCase();
    return id.includes(q) || (/^\d+$/.test(digits) && Number(id.slice(1)) === Number(digits));
  });
}

/** A short verdict for a probability, so a value near 50% is never read as a decision. */
export function probabilityNote(p) {
  if (typeof p !== 'number') return null;
  const distance = Math.abs(p - 0.5);
  if (distance < 0.05) return 'This is essentially undecided: the probability is close to 50%.';
  if (distance < 0.15) return 'This leans one way only slightly.';
  return 'Even a clearer probability does not mean the estimate is likely to be correct.';
}
