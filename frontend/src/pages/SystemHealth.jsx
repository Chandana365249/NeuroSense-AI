import { useCallback, useEffect, useState } from 'react';
import { api } from '../api';
import Card from '../components/Card';
import LoadingBlock from '../components/LoadingBlock';
import Notice from '../components/Notice';
import PageHeader from '../components/PageHeader';
import RequireAuth from '../components/RequireAuth';
import StatusPill from '../components/StatusPill';
import { NA } from '../components/format';

const DIALECTS = { postgresql: 'PostgreSQL', sqlite: 'SQLite' };

function Check({ label, ok, okText, failText, detail }) {
  return (
    <li className="health-check">
      <span className="health-label">{label}</span>
      <StatusPill tone={ok ? 'ok' : 'error'}>{ok ? okText : failText}</StatusPill>
      {detail ? <span className="health-detail">{detail}</span> : null}
    </li>
  );
}

// Detailed diagnostics come from GET /admin/diagnostics, which the server only answers for administrators.
function Diagnostics({ auth }) {
  const [state, setState] = useState({ status: 'loading' });

  const load = useCallback(async () => {
    setState((s) => ({ ...s, status: 'loading' }));
    const started = performance.now();
    try {
      const data = await api.adminDiagnostics(auth.token);
      setState({ status: 'ready', data, ms: Math.round(performance.now() - started), checkedAt: new Date() });
    } catch (err) {
      if (err.status === 401) { auth.expire(); return; }
      setState({ status: 'error', unreachable: err.status === 0, forbidden: err.status === 403, httpStatus: err.status, checkedAt: new Date() });
    }
  }, [auth.token, auth.expire]);
  useEffect(() => { load(); }, [load]);

  const busy = state.status === 'loading';
  const h = state.data;

  return (
    <>
      <div className="page-header-actions">
        <button type="button" className="btn btn-secondary btn-small" onClick={() => { if (!busy) load(); }} aria-disabled={busy || undefined}>
          {busy ? 'Checking…' : 'Check again'}
        </button>
      </div>

      <Card title="Current status" titleId="health-status">
        <div aria-busy={busy}>
          {busy && !h ? <LoadingBlock label="Checking system health…" /> : null}
          {state.status === 'error' ? (
            state.forbidden ? (
              <Notice tone="warning" title="Administrator access required">
                The server did not allow this account to read system diagnostics.
              </Notice>
            ) : (
              <Notice tone="error" title={state.unreachable ? 'The API is unreachable' : 'The health check failed'}>
                {state.unreachable
                  ? 'The web interface could not reach the API service. The service may be stopped or restarting.'
                  : `The API answered with an error (HTTP ${state.httpStatus}).`}
                {' '}Checked {state.checkedAt.toLocaleTimeString()}.
              </Notice>
            )
          ) : null}
          {state.status !== 'error' && h ? (
            <>
              <div className="health-overall" role="status">
                <StatusPill tone={h.status === 'ok' ? 'ok' : 'warn'}>{h.status === 'ok' ? 'All core services responding' : 'Degraded'}</StatusPill>
                <span className="muted small">
                  {busy ? 'Checking again…' : `Checked ${state.checkedAt.toLocaleTimeString()} · response in ${state.ms} ms`}
                </span>
              </div>
              <ul className="health-list">
                <Check label="API service" ok okText="Responding" detail={`Version ${h.version || NA}`} />
                <Check label="Database" ok={!!h.database?.connected} okText="Connected" failText="Not connected"
                  detail={DIALECTS[h.database?.dialect] || h.database?.dialect || null} />
                <Check label="Active model" ok={!!h.model?.loaded} okText="Loaded" failText="Not loaded"
                  detail={h.model?.version ? <span className="mono">{h.model.version}</span> : null} />
                <Check label="Dataset files" ok={!!h.dataset_available} okText="Found" failText="Not found"
                  detail="DEAP participant files in the server's dataset location" />
              </ul>
              <dl className="kv kv-compact">
                <dt>Environment</dt><dd>{h.environment || NA}</dd>
              </dl>
            </>
          ) : null}
        </div>
      </Card>

      <Notice tone="info" title="About this page">
        These values come from the administrator-only diagnostics endpoint; the server rejects other accounts. It reports
        status only, never credentials, connection strings or tokens. A healthy result does not show that the system is
        secure.
      </Notice>
    </>
  );
}

export default function SystemHealth({ auth }) {
  return (
    <>
      <PageHeader title="System health" subtitle="Research administration · technical diagnostics: the API, database, active model and dataset files." />
      <RequireAuth auth={auth} admin reason="Sign in with an administrator account to view system diagnostics.">
        <Diagnostics auth={auth} />
      </RequireAuth>
    </>
  );
}
