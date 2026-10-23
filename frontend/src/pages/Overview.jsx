import { useEffect, useState } from 'react';
import { api } from '../api';
import { isSynthetic } from '../analysis';
import Card from '../components/Card';
import LoadingBlock from '../components/LoadingBlock';
import Notice from '../components/Notice';
import { pct, predictionLabel, sourceLabel, when } from '../components/format';
import { href } from '../router';

function RecentAnalyses({ auth, reloadKey }) {
  const [state, setState] = useState({ status: 'loading' });

  useEffect(() => {
    if (!auth.token) return undefined;
    let cancelled = false;
    setState({ status: 'loading' });
    api.predictions(auth.token, { limit: 5 })
      .then((data) => { if (!cancelled) setState({ status: 'ready', data }); })
      .catch((err) => {
        if (cancelled) return;
        if (err.status === 401) auth.expire();
        else setState({ status: 'error', error: err.status === 0 ? 'The service is unreachable.' : err.message });
      });
    return () => { cancelled = true; };
  }, [auth.token, auth.expire, reloadKey]);

  if (state.status === 'loading') return <LoadingBlock label="Loading recent analyses…" lines={2} />;
  if (state.status === 'error') return <Notice tone="error" title="Could not load recent analyses">{state.error}</Notice>;
  if (!state.data.items.length) {
    return (
      <div className="empty-state">
        <p><strong>No analyses yet.</strong></p>
        <p className="muted">Your results will appear here after your first analysis.</p>
      </div>
    );
  }
  return (
    <ul className="recent-list">
      {state.data.items.map((p) => (
        <li key={p.prediction_id}>
          <a href={href(`/results/${p.prediction_id}`)}>
            <span className="recent-class">{predictionLabel(p)}{isSynthetic(p) ? null : <span className="muted small"> · {pct(p.confidence)}</span>}</span>
            <span className="recent-meta">#{p.prediction_id} · {sourceLabel(p)} · {when(p.created_at)}</span>
          </a>
        </li>
      ))}
    </ul>
  );
}

/** A small, unobtrusive service indicator; details live on System health (administrators). */
function ServiceIndicator({ publicData }) {
  const { status, health } = publicData;
  const [tone, text] = status === 'loading' ? ['neutral', 'Checking service…']
    : status === 'error' ? ['error', 'Service unavailable']
      : health?.status === 'ok' ? ['ok', 'Service available'] : ['warn', 'Service limited'];
  return <span className={`service-dot service-${tone}`} role="status"><span aria-hidden="true" />{text}</span>;
}

export default function Overview({ auth, publicData }) {
  const name = auth.user?.full_name?.trim();
  return (
    <>
      <header className="page-header overview-hero">
        <div>
          <p className="eyebrow">{name ? `Welcome, ${name}` : 'Welcome'}</p>
          <h1 id="page-title" tabIndex={-1}>NeuroSense AI</h1>
          <p className="page-subtitle">
            Estimate whether an EEG recording resembles moments people rated as more or less pleasant.
          </p>
        </div>
        <ServiceIndicator publicData={publicData} />
      </header>

      {publicData.status === 'error' ? (
        <Notice tone="error" title="The service is unreachable" action={<button type="button" className="btn btn-secondary btn-small" onClick={publicData.reload}>Retry</button>}>
          Analyses and history are unavailable until the connection is restored.
        </Notice>
      ) : null}

      <section className="card start-card" aria-labelledby="start-analysis">
        <div className="start-copy">
          <h2 id="start-analysis">Start an analysis</h2>
          <p className="muted">Choose an EEG recording, run the analysis, and read the result in a few seconds.</p>
        </div>
        <a className="btn btn-primary btn-large" href={href('/analyze')}>Analyze EEG</a>
      </section>

      <div className="grid-2">
        <Card title="Recent analyses" titleId="recent-analyses" actions={<a href={href('/history')} className="small">View all</a>}>
          <RecentAnalyses auth={auth} reloadKey={publicData.loadedAt} />
        </Card>

        <div className="stack">
          <Card title="What you can analyze" titleId="accepted-data">
            <ul className="check-list">
              <li><strong>DEAP research recordings</strong> from the authorized participants on this server.</li>
              <li><strong>Your own EEG file</strong>: JSON with 32 channels at 128 samples per second, at least 4 seconds.</li>
              <li><strong>A synthetic test signal</strong> to check that everything works. It is not real EEG.</li>
            </ul>
          </Card>
          <Card title="Please keep in mind" titleId="reliability" className="limits-card">
            <p>
              This is a research prototype. For people it had not seen before, its estimates were
              <strong> close to chance</strong> in our exploratory tests. It does not detect emotions or diagnose anything.
            </p>
            <p className="meta"><a href={href('/model')}>How the model was evaluated</a></p>
          </Card>
        </div>
      </div>
    </>
  );
}
