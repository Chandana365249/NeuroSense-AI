import { lazy, Suspense, useCallback, useEffect, useMemo, useState } from 'react';
import { dashboardSummary, datasetKind, DATASET_LABELS, isSynthetic } from '../analysis';
import Card from '../components/Card';
import LoadingBlock from '../components/LoadingBlock';
import Notice from '../components/Notice';
import PageHeader from '../components/PageHeader';
import { pct, predictionLabel, sourceLabel, when } from '../components/format';
import { href } from '../router';
import { fetchAllHistory } from './History';

const DashboardCharts = lazy(() => import('../components/DashboardCharts'));
const ValenceTrend = lazy(() => import('../components/ValenceTrend'));
const RECENT = 10;

function Tile({ label, value, hint }) {
  return (
    <div className="kpi">
      <span className="kpi-label">{label}</span>
      <span className="kpi-value">{value}</span>
      {hint ? <span className="kpi-hint">{hint}</span> : null}
    </div>
  );
}

function RecentTable({ items }) {
  const rows = [...items].sort((a, b) => String(b.created_at).localeCompare(String(a.created_at))).slice(0, RECENT);
  return (
    <div className="table-wrap" role="region" aria-labelledby="dash-recent" tabIndex={0}>
      <table className="table dash-table">
        <caption className="visually-hidden">Your {rows.length} most recent analyses</caption>
        <thead>
          <tr>
            <th scope="col">Result</th><th scope="col">Date</th><th scope="col">Input</th><th scope="col">Dataset</th>
            <th scope="col">Estimate</th><th scope="col">Valence score</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((p) => (
            <tr key={p.prediction_id}>
              <th scope="row"><a href={href(`/results/${p.prediction_id}`)}>#{p.prediction_id}</a></th>
              <td className="nowrap">{when(p.created_at)}</td>
              <td>{sourceLabel(p)}</td>
              <td><span className={`tag dataset-tag dataset-${datasetKind(p)}`}>{DATASET_LABELS[datasetKind(p)]}</span></td>
              <td>{isSynthetic(p) ? <span className="muted">Not an estimate</span>
                : <span className={`gauge-chip ${p.prediction === 'high_valence' ? '' : 'is-low'}`}>{predictionLabel(p)}</span>}</td>
              <td className="nowrap">{isSynthetic(p) ? '—' : pct(p.probabilities?.high_valence)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

export default function Dashboard({ auth }) {
  const [state, setState] = useState({ status: 'loading' });
  const filters = useMemo(() => ({}), []);
  const load = useCallback(async () => {
    setState({ status: 'loading' });
    try {
      setState({ status: 'ready', ...(await fetchAllHistory(auth.token, filters)) });
    } catch (err) {
      if (err.status === 401) auth.expire();
      else setState({ status: 'error', error: err.status === 0 ? 'The service is unreachable.' : 'Your analyses could not be loaded.' });
    }
  }, [auth.token, auth.expire, filters]);
  useEffect(() => { load(); }, [load]);

  const header = (
    <PageHeader title="Dashboard" subtitle="Your analyses at a glance.">
      <a className="btn btn-secondary btn-small" href={href('/history')}>Full history</a>
      <a className="btn btn-primary btn-small" href={href('/analyze')}>New analysis</a>
    </PageHeader>
  );
  if (state.status === 'loading') return <>{header}<LoadingBlock label="Loading your analyses…" /></>;
  if (state.status === 'error') {
    return (
      <>
        {header}
        <Notice tone="error" title="Could not load your analyses" action={<button type="button" className="btn btn-secondary btn-small" onClick={load}>Retry</button>}>{state.error}</Notice>
      </>
    );
  }
  if (!state.items.length) {
    return (
      <>
        {header}
        <Card>
          <div className="empty-state">
            <p><strong>No analyses yet.</strong></p>
            <p className="muted">Run your first analysis and it will appear here.</p>
            <a className="btn btn-primary" href={href('/analyze')}>Analyze EEG</a>
          </div>
        </Card>
      </>
    );
  }
  const s = dashboardSummary(state.items);
  return (
    <>
      {header}
      <section className="kpi-grid" aria-label="Summary">
        <Tile label="Total analyses" value={s.total} hint={s.counts.synthetic ? `${s.counts.synthetic} synthetic tests included` : null} />
        <Tile label="DEAP recordings" value={s.counts.deap} />
        <Tile label="FACED recordings" value={s.counts.faced} />
        <Tile label="Uploaded files" value={s.counts.upload} />
        <Tile label="Higher / lower" value={`${s.higher} / ${s.lower}`} hint="valence estimates" />
        <Tile label="Last analysis" value={s.latest ? s.latest.toLocaleDateString(undefined, { month: 'short', day: 'numeric' }) : '—'}
          hint={s.latest ? s.latest.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }) : null} />
      </section>

      <div className="dash-row">
        <Card title="Valence score of your recent analyses" titleId="dash-trend">
          <Suspense fallback={<LoadingBlock label="Loading chart…" lines={2} />}><ValenceTrend items={state.items} /></Suspense>
        </Card>
        <Card title="Overview" titleId="dash-split">
          <Suspense fallback={<LoadingBlock label="Loading charts…" lines={2} />}><DashboardCharts summary={s} /></Suspense>
        </Card>
      </div>

      <Card title="Recent analyses" titleId="dash-recent" actions={<a href={href('/history')} className="small">View all</a>}>
        <RecentTable items={state.items} />
      </Card>
    </>
  );
}
