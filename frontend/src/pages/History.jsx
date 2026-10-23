import { lazy, Suspense, useCallback, useEffect, useRef, useState } from 'react';
import { api } from '../api';
import Card from '../components/Card';
import LoadingBlock from '../components/LoadingBlock';
import Notice from '../components/Notice';
import PageHeader from '../components/PageHeader';
import RequireAuth from '../components/RequireAuth';
import { classLabel, predictionLabel, pct, sourceLabel, when } from '../components/format';
import { isSynthetic } from '../analysis';
import { href } from '../router';

const PAGE_SIZE = 20;
const FETCH_SIZE = 200;       // the API's maximum page size
const MAX_CHART_RECORDS = 5000;
const HistoryCharts = lazy(() => import('../components/HistoryCharts'));

/** Fetches every page of the user's history (deduplicated by id) for the charts. */
async function fetchAllHistory(token) {
  const byId = new Map();
  let total = 0;
  for (let offset = 0; offset < MAX_CHART_RECORDS; offset += FETCH_SIZE) {
    const data = await api.predictions(token, { limit: FETCH_SIZE, offset });
    total = data.total;
    for (const p of data.items) byId.set(p.prediction_id, p);
    if (!data.items.length || offset + FETCH_SIZE >= total) break;
  }
  return { items: [...byId.values()], total };
}

function HistoryInsights({ auth }) {
  const [state, setState] = useState({ status: 'loading' });
  const latest = useRef(0); // only the most recent load may update the state
  const load = useCallback(async () => {
    const run = ++latest.current;
    setState({ status: 'loading' });
    try {
      const data = await fetchAllHistory(auth.token);
      if (run === latest.current) setState({ status: 'ready', ...data });
    } catch (err) {
      if (run !== latest.current) return;
      if (err.status === 401) auth.expire();
      else setState({ status: 'error', error: err.status === 0 ? 'The service is unreachable.' : 'Your history could not be summarised.' });
    }
  }, [auth.token, auth.expire]);
  useEffect(() => {
    load();
    return () => { latest.current += 1; };
  }, [load]);

  if (state.status === 'loading') return <LoadingBlock label="Summarising your history…" />;
  if (state.status === 'error') {
    return <Notice tone="error" title="Could not draw the charts" action={<button type="button" className="btn btn-secondary btn-small" onClick={load}>Retry</button>}>{state.error}</Notice>;
  }
  if (!state.items.length) return <p className="muted empty">Charts appear once you have saved analyses.</p>;
  const partial = state.items.length < state.total;
  return (
    <>
      <p className="muted">
        {partial
          ? `Based on your ${state.items.length} most recent analyses (of ${state.total}).`
          : `Based on all ${state.total} of your saved ${state.total === 1 ? 'analysis' : 'analyses'}, across every page of the table.`}
        {' '}These are records of the analyses you ran. They are not research results or evidence of model performance.
      </p>
      <Suspense fallback={<LoadingBlock label="Loading charts…" />}>
        <HistoryCharts items={state.items} />
      </Suspense>
    </>
  );
}

function HistoryTable({ auth }) {
  const [page, setPage] = useState(0);
  const [state, setState] = useState({ status: 'loading' });

  const load = useCallback(async () => {
    setState((s) => ({ ...s, status: 'loading' }));
    try {
      const data = await api.predictions(auth.token, { limit: PAGE_SIZE, offset: page * PAGE_SIZE });
      setState({ status: 'ready', data, page });
    } catch (err) {
      if (err.status === 401) auth.expire();
      else setState({ status: 'error', error: err.status === 0 ? 'The service is unreachable.' : err.message });
    }
  }, [auth.token, auth.expire, page]);

  useEffect(() => { load(); }, [load]);

  if (state.status === 'loading' && !state.data) return <LoadingBlock label="Loading your history…" />;
  if (state.status === 'error') {
    return <Notice tone="error" title="Could not load your history" action={<button type="button" className="btn btn-secondary btn-small" onClick={load}>Retry</button>}>{state.error}</Notice>;
  }
  const { items, total } = state.data;
  const shown = state.page ?? 0; // the page whose rows are on screen (`page` may still be loading)
  if (!total) {
    return <p className="muted empty">No analyses yet. <a href={href('/analyze')}>Analyze an EEG recording</a> to start your history.</p>;
  }
  const pages = Math.max(1, Math.ceil(total / PAGE_SIZE));
  const first = shown * PAGE_SIZE + 1;
  const last = Math.min(total, first + items.length - 1);
  // aria-disabled rather than disabled, so keyboard focus stays on the button while pages change.
  const canPrev = shown > 0 && state.status !== 'loading';
  const canNext = shown + 1 < pages && state.status !== 'loading';

  return (
    <>
      <div className="table-wrap" aria-busy={state.status === 'loading'}>
        <table className="table history-table">
          <caption className="visually-hidden">Your analyses, newest first. Showing {first}–{last} of {total}.</caption>
          <thead>
            <tr>
              <th scope="col">#</th>
              <th scope="col">Date</th>
              <th scope="col">Input</th>
              <th scope="col">Estimated category</th>
              <th scope="col">Model probability</th>
              <th scope="col">Self-reported category</th>
              <th scope="col"><span className="visually-hidden">Details</span></th>
            </tr>
          </thead>
          <tbody>
            {items.map((p) => (
              <tr key={p.prediction_id}>
                <td className="nowrap">{p.prediction_id}</td>
                <td>{when(p.created_at)}</td>
                <td>{sourceLabel(p)}</td>
                <td>{predictionLabel(p)}</td>
                <td className="nowrap">{isSynthetic(p) ? 'Not shown (synthetic)' : pct(p.confidence)}</td>
                <td>{p.source_reference?.self_reported_label ? classLabel(p.source_reference.self_reported_label) : '—'}</td>
                <td className="nowrap"><a href={href(`/results/${p.prediction_id}`)}>View<span className="visually-hidden"> prediction {p.prediction_id}</span></a></td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <nav className="pager" aria-label="History pages">
        <button type="button" className="btn btn-secondary btn-small" onClick={() => { if (canPrev) setPage((p) => p - 1); }}
          aria-disabled={!canPrev || undefined}>Previous</button>
        <span className="muted small" aria-live="polite">{first}–{last} of {total} · page {shown + 1} of {pages}</span>
        <button type="button" className="btn btn-secondary btn-small" onClick={() => { if (canNext) setPage((p) => p + 1); }}
          aria-disabled={!canNext || undefined}>Next</button>
      </nav>
      <p className="meta small">Model probability is the model&apos;s probability for its estimated category, not a measure of accuracy.</p>
    </>
  );
}

export default function History({ auth }) {
  return (
    <>
      <PageHeader title="History" subtitle="Your saved analyses. Open one to see its full result." />
      <RequireAuth auth={auth} reason="Sign in to see your analysis history.">
        <Card><HistoryTable auth={auth} /></Card>
        <Card title="Your history in charts" titleId="history-charts"><HistoryInsights auth={auth} /></Card>
      </RequireAuth>
    </>
  );
}
