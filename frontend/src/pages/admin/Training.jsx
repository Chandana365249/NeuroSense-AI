import { useCallback, useEffect, useRef, useState } from 'react';
import { api } from '../../api';
import Card from '../../components/Card';
import LoadingBlock from '../../components/LoadingBlock';
import Notice from '../../components/Notice';
import PageHeader from '../../components/PageHeader';
import RequireAuth from '../../components/RequireAuth';
import StatusPill from '../../components/StatusPill';
import { pct, when } from '../../components/format';
import { href } from '../../router';

const STATE_TONE = { idle: 'neutral', running: 'warn', succeeded: 'ok', failed: 'error' };
const POLL_MS = 3000;

function startErrorMessage(err) {
  if (err.status === 0) return 'The service is unreachable. Training was not started.';
  if (err.status === 409) return 'A training run or a model activation is already in progress. The status below has been refreshed; try again after it finishes.';
  if (err.status === 503) return 'The dataset is not available on this server, so training cannot start.';
  return 'Training could not be started. Try again later or check the server logs.';
}

function Outcome({ t }) {
  if (t.state === 'succeeded' && t.model_version) {
    return (
      <Notice tone="info" title="Candidate registered (inactive)" announce={false}>
        <span className="mono">{t.model_version}</span> was saved as an inactive candidate. The active model has not
        changed. Its evaluation is exploratory (leave-one-participant-out on the development participants), not an
        independent test, so a higher score is not evidence that it is better.
        Review it in the <a href={href('/models')}>Model registry</a>; activation is a separate administrator step.
      </Notice>
    );
  }
  if (t.state === 'failed') {
    return (
      <Notice tone="error" title="Training failed" announce={false}>
        {t.error || 'Training failed.'} The active model is unchanged. You can start a new run.
      </Notice>
    );
  }
  return null;
}

// Every training endpoint is administrator-only on the server; the token is always sent.
function TrainingPanel({ auth }) {
  const [state, setState] = useState({ status: 'loading' });
  const [pollError, setPollError] = useState(null);
  const [runCv, setRunCv] = useState(true);
  const [submitting, setSubmitting] = useState(false);
  const [startError, setStartError] = useState(null);
  const [announcement, setAnnouncement] = useState('');
  const submittingRef = useRef(false);
  const previous = useRef(null);

  const load = useCallback(async () => {
    try {
      const data = await api.trainingStatus(auth.token);
      setPollError(null);
      setState({ status: 'ready', data });
    } catch (err) {
      if (err.status === 401) { auth.expire(); return; }
      if (err.status === 403) { setState({ status: 'error', forbidden: true }); return; }
      const message = err.status === 0 ? 'The service is unreachable.' : 'The training status could not be loaded.';
      // Keep the last known status (and keep polling) if a refresh fails; otherwise show the error.
      setState((s) => (s.data ? s : { status: 'error', error: message }));
      setPollError(message);
    }
  }, [auth.token, auth.expire]);

  useEffect(() => { load(); }, [load]);
  const running = state.data?.state === 'running';
  useEffect(() => {
    if (!running) return undefined;
    const id = setInterval(load, POLL_MS);
    return () => clearInterval(id);
  }, [running, load]);

  // Announce the end of a run that this page watched.
  useEffect(() => {
    const now = state.data?.state;
    if (previous.current === 'running' && now === 'succeeded') {
      setAnnouncement('Training finished. The new model was registered as an inactive candidate; the active model is unchanged.');
    } else if (previous.current === 'running' && now === 'failed') {
      setAnnouncement('Training failed. The active model is unchanged.');
    }
    previous.current = now;
  }, [state.data?.state]);

  async function start() {
    if (submittingRef.current || running) return; // no duplicate submissions
    submittingRef.current = true;
    setSubmitting(true);
    setStartError(null);
    try {
      const data = await api.train(auth.token, runCv);
      setState({ status: 'ready', data });
      setAnnouncement('Candidate training started.');
    } catch (err) {
      if (err.status === 401) { auth.expire(); return; }
      if (err.status === 403) { setState({ status: 'error', forbidden: true }); return; }
      setStartError(startErrorMessage(err));
      if (err.status === 409) load();
    } finally {
      submittingRef.current = false;
      setSubmitting(false);
    }
  }

  if (state.status === 'loading') return <LoadingBlock label="Loading training status…" />;
  if (state.status === 'error' && state.forbidden) {
    return <Notice tone="warning" title="Administrator access required">The server did not allow this account to use training.</Notice>;
  }
  if (state.status === 'error') {
    return <Notice tone="error" title="Could not load training status" action={<button type="button" className="btn btn-secondary btn-small" onClick={load}>Retry</button>}>{state.error}</Notice>;
  }
  const t = state.data;
  const blocked = submitting || running;
  return (
    <div className="grid-2 training-grid">
      <Card title="Train a candidate model" titleId="train-candidate">
        <p>
          Trains a Random Forest under Evaluation Protocol v1.1 on the development participants only (the frozen
          final-test group is never loaded) and saves the result as an <strong>inactive candidate</strong>. It does not
          replace the active model and is never activated automatically. Activating a model is a separate, audited
          administrator action in the <a href={href('/models')}>Model registry</a>.
        </p>
        <div className="form-section">
          <label className="check" htmlFor="train-cv">
            <input id="train-cv" type="checkbox" checked={runCv} onChange={(e) => setRunCv(e.target.checked)} disabled={blocked} />
            Run the exploratory leave-one-participant-out evaluation (slower)
          </label>
          <button type="button" className="btn btn-primary" onClick={start} aria-disabled={blocked || undefined}
            aria-describedby="train-note">
            {submitting ? 'Starting…' : running ? 'Training in progress…' : 'Train candidate'}
          </button>
          <span id="train-note" className="field-hint">
            {running ? 'A run is in progress; a new run can start after it finishes.' : 'One run at a time. Progress is shown alongside.'}
          </span>
        </div>
        {startError ? <Notice tone="error" title="Training was not started">{startError}</Notice> : null}
      </Card>

      <Card title="Training status" titleId="training-status">
        <dl className="kv">
          <dt>State</dt><dd><StatusPill tone={STATE_TONE[t.state] || 'neutral'}>{t.state}</StatusPill></dd>
          <dt>Stage</dt><dd>{t.stage || '—'}</dd>
          <dt>Progress</dt><dd>{pct(t.progress)}</dd>
          <dt>Started</dt><dd>{when(t.started_at)}</dd>
          <dt>Finished</dt><dd>{when(t.finished_at)}</dd>
          {t.model_version ? <><dt>Candidate version</dt><dd className="mono">{t.model_version}</dd></> : null}
        </dl>
        {running ? (
          <div className="field training-progress">
            <label htmlFor="training-progress">Training progress</label>
            <progress id="training-progress" max="1" value={t.progress || 0}>{pct(t.progress)}</progress>
          </div>
        ) : null}
        {pollError ? (
          <Notice tone="warning" title="Status could not be refreshed" action={<button type="button" className="btn btn-secondary btn-small" onClick={load}>Retry</button>}>
            {pollError} Showing the last known status{running ? '; refreshing again shortly' : ''}.
          </Notice>
        ) : null}
        <Outcome t={t} />
      </Card>
      <p className="visually-hidden" role="status">{announcement}</p>
    </div>
  );
}

export default function Training({ auth }) {
  return (
    <>
      <PageHeader title="Training" subtitle="Research administration · train candidate models without changing the active model." />
      <RequireAuth auth={auth} admin reason="Sign in with an administrator account to use training.">
        <TrainingPanel auth={auth} />
      </RequireAuth>
    </>
  );
}
