import { useEffect, useState } from 'react';
import { api } from '../api';
import Card from '../components/Card';
import LoadingBlock from '../components/LoadingBlock';
import Notice from '../components/Notice';
import PageHeader from '../components/PageHeader';
import StatusPill from '../components/StatusPill';
import { algorithmLabel, classLabel, sourceLabel, VALENCE_EXPLANATION, when } from '../components/format';
import { href } from '../router';

function RecentAnalyses({ auth }) {
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
        else setState({ status: 'error', error: err.message });
      });
    return () => { cancelled = true; };
  }, [auth.token, auth.expire]);

  if (!auth.user) {
    return <p className="muted">{auth.status === 'checking' ? 'Checking your session…' : <>
      <a href={href('/sign-in?next=/overview')}>Sign in</a> to see your recent analyses.</>}</p>;
  }
  if (state.status === 'loading') return <LoadingBlock label="Loading recent analyses…" lines={2} />;
  if (state.status === 'error') return <Notice tone="error" title="Could not load recent analyses">{state.error}</Notice>;
  if (!state.data.items.length) {
    return <p className="muted">No analyses yet. <a href={href('/analyze')}>Analyze an EEG recording</a> to get started.</p>;
  }
  return (
    <ul className="recent-list">
      {state.data.items.map((p) => (
        <li key={p.prediction_id}>
          <a href={href(`/results/${p.prediction_id}`)}>
            <span className="recent-class">{classLabel(p.prediction)}</span>
            <span className="recent-meta">{sourceLabel(p)} · {when(p.created_at)}</span>
          </a>
        </li>
      ))}
    </ul>
  );
}

export default function Overview({ auth, publicData }) {
  const { status, health, modelInfo } = publicData;
  const serviceTone = status === 'error' ? 'error' : health?.status === 'ok' ? 'ok' : health ? 'warn' : 'neutral';
  const serviceText = status === 'loading' ? 'Checking…' : status === 'error' ? 'Unavailable'
    : health?.status === 'ok' ? 'Available' : 'Limited';
  const modelReady = modelInfo?.available;

  return (
    <>
      <PageHeader title="Overview" subtitle="EEG-based estimation of self-reported valence, for research use." />

      <section className="status-row" aria-label="Service status">
        <div className="status-item"><span className="status-label">Service</span><StatusPill tone={serviceTone}>{serviceText}</StatusPill></div>
        <div className="status-item">
          <span className="status-label">Active model</span>
          {status === 'loading' ? <StatusPill>Checking…</StatusPill>
            : modelReady ? <StatusPill tone="ok">{algorithmLabel(modelInfo.algorithm)}</StatusPill>
              : <StatusPill tone="warn">No trained model</StatusPill>}
        </div>
      </section>
      {status === 'error' ? (
        <Notice tone="error" title="The service is unreachable" action={<button type="button" className="btn btn-secondary btn-small" onClick={publicData.reload}>Retry</button>}>
          Analyses and history are unavailable until the connection is restored.
        </Notice>
      ) : null}

      <div className="grid-2">
        <Card title="What this estimates" titleId="what-estimates">
          <p>{VALENCE_EXPLANATION}</p>
          <p className="muted">
            NeuroSense AI estimates which of these two categories a recording most resembles. It is a research
            prototype: it does not detect a person&apos;s exact emotions and it does not diagnose any medical or
            psychological condition.
          </p>
          <div className="button-row">
            <a className="btn btn-primary" href={href('/analyze')}>Analyze EEG</a>
            <a className="btn btn-secondary" href={href('/model')}>See the evidence</a>
          </div>
        </Card>

        <Card title="Recent analyses" titleId="recent-analyses" actions={auth.user ? <a href={href('/history')} className="small">View all</a> : null}>
          <RecentAnalyses auth={auth} />
        </Card>
      </div>

      <Card title="How reliable are the estimates?" titleId="reliability">
        <p>
          In exploratory evaluations on 12 participants, estimates for people the model had never seen were
          <strong> close to chance</strong>. Models trained on a person&apos;s own rated recordings did better, but
          predicted the group&apos;s typical rating of each video just as well as the person&apos;s own rating.
        </p>
        <p><a href={href('/model')}>Read the research summary and model details</a></p>
      </Card>
    </>
  );
}
