import { useCallback, useEffect, useRef, useState } from 'react';
import { api } from '../api';
import ActivationDialog from '../components/ActivationDialog';
import Breakable from '../components/Breakable';
import Card from '../components/Card';
import LoadingBlock from '../components/LoadingBlock';
import Notice from '../components/Notice';
import PageHeader from '../components/PageHeader';
import { datasetPath } from '../datasets';
import StatusPill from '../components/StatusPill';
import { algorithmLabel, NA, when } from '../components/format';
import { evaluationStatus, modelStatusLabel, trainingDataSummary } from '../modelInfo';
import { href } from '../router';

const FRAMEWORK_LABELS = { sklearn: 'scikit-learn', pytorch: 'PyTorch', torch: 'PyTorch' };

/** Active model first, then newest first. */
function ordered(models) {
  return [...models].sort((a, b) => (Number(b.is_active) - Number(a.is_active))
    || String(b.trained_at).localeCompare(String(a.trained_at)));
}

function RegistryTable({ state, load, canActivate, onActivate }) {
  if (state.status === 'loading') return <LoadingBlock label="Loading registered models…" />;
  if (state.status === 'error') {
    return <Notice tone="error" title="Could not load the model registry" action={<button type="button" className="btn btn-secondary btn-small" onClick={load}>Retry</button>}>{state.error}</Notice>;
  }
  const { models } = state;
  if (!models.length) return <p className="muted empty">No models are registered on this server.</p>;
  const active = models.filter((m) => m.is_active);

  return (
    <>
      <p className="registry-count">
        {models.length} registered {models.length === 1 ? 'model' : 'models'} ·{' '}
        {active.length === 1 ? <>active: <span className="mono ident"><Breakable text={active[0].version} /></span></> : active.length ? `${active.length} marked active` : 'no active model'}
      </p>
      {/* Focusable so keyboard users can scroll the table sideways on narrow screens. */}
      <div className="table-wrap" role="region" aria-labelledby="registered-models" tabIndex={0}>
        <table className="table registry-table">
          <caption className="visually-hidden">Registered model versions, active model first.</caption>
          <thead>
            <tr>
              <th scope="col">Status</th>
              <th scope="col">Model</th>
              <th scope="col">Version</th>
              <th scope="col">Trained</th>
              {canActivate ? <th scope="col">Actions</th> : null}
            </tr>
          </thead>
          <tbody>
            {models.map((m) => (
              <tr key={m.version} className={m.is_active ? 'row-active' : undefined}>
                <td className="nowrap">
                  <StatusPill tone={m.is_active ? 'ok' : modelStatusLabel(m).startsWith('Candidate') ? 'warn' : undefined}>{modelStatusLabel(m)}</StatusPill>
                </td>
                <td className="model-cell">
                  {algorithmLabel(m.model_type || m.algorithm)}
                  <span className="cell-sub">{FRAMEWORK_LABELS[m.framework] || m.framework || NA}{m.feature_count ? ` · ${m.feature_count} features` : ''}</span>
                  <span className="cell-sub">Trained on: {trainingDataSummary(m.provenance)}</span>
                  <span className="cell-sub">Evaluation: {evaluationStatus(m).label}</span>
                </td>
                <td className="mono small ident version-cell"><Breakable text={m.version} /></td>
                <td className="trained-cell">{when(m.trained_at)}</td>
                {canActivate ? (
                  <td className="nowrap">
                    {m.is_active ? <span className="muted small">Active model</span> : (
                      <button type="button" className="btn btn-secondary btn-small" aria-label={`Activate ${m.version}`}
                        onClick={(e) => onActivate(m, e.currentTarget)}>Activate…</button>
                    )}
                  </td>
                ) : null}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <p className="meta small">
        “Candidate (inactive)” marks a model that has not been activated; it is never used for analyses until an
        administrator activates it.
      </p>
    </>
  );
}

function ActivationResult({ result, resultRef }) {
  if (!result) return null;
  return (
    <div ref={resultRef} tabIndex={-1} className="focus-target">
      {result.changed ? (
        <Notice tone="info" title="Model activated">
          <span className="mono">{result.version}</span> is now the active model
          {result.previous_version ? <> (previously <span className="mono">{result.previous_version}</span>)</> : null}.
          New predictions use it; earlier results keep the model they were made with. To roll back, activate the previous
          version the same way.
        </Notice>
      ) : (
        <Notice tone="info" title="No change">
          <span className="mono">{result.version}</span> was already the active model. Nothing was changed.
        </Notice>
      )}
    </div>
  );
}

export default function ModelRegistry({ auth, publicData }) {
  const [state, setState] = useState({ status: 'loading' });
  const [dialog, setDialog] = useState(null); // { model, trigger }
  const [result, setResult] = useState(null);
  const resultRef = useRef(null);
  const headingRef = useRef(null);
  // Shown for administrator accounts only. This is a usability choice: the server authorizes every activation itself.
  const canActivate = Boolean(auth?.user?.is_admin);

  const load = useCallback(async () => {
    setState({ status: 'loading' });
    try {
      setState({ status: 'ready', models: ordered(await api.models()) });
    } catch (err) {
      setState({ status: 'error', error: err.status === 0 ? 'The service is unreachable.' : 'The model registry could not be loaded.' });
    }
  }, []);
  useEffect(() => { load(); }, [load]);

  const openDialog = useCallback((model, trigger) => {
    setResult(null);
    setDialog({ model, trigger });
  }, []);

  const cancel = useCallback(() => {
    const trigger = dialog?.trigger;
    setDialog(null);
    requestAnimationFrame(() => (trigger?.isConnected ? trigger : headingRef.current)?.focus());
  }, [dialog]);

  const activated = useCallback(async (res) => {
    setDialog(null);
    setResult(res);
    // Registry data is refreshed only after a successful activation.
    await load();
    publicData?.reload?.();
    requestAnimationFrame(() => resultRef.current?.focus());
  }, [load, publicData]);

  const activeVersion = state.models?.find((m) => m.is_active)?.version || null;

  return (
    <>
      <PageHeader title="Model registry" subtitle="Every model version registered on this server, and which one is active." />
      <p className="muted small">
        This registry holds DEAP models only. The FACED research model is kept separately and can never be activated here.{' '}
        <a href={href(datasetPath('/model', 'faced'))}>About the FACED model</a>
      </p>
      <ActivationResult result={result} resultRef={resultRef} />
      <Card title="Registered models" titleId="registered-models" titleRef={headingRef}>
        <RegistryTable state={state} load={load} canActivate={canActivate} onActivate={openDialog} />
      </Card>
      <p className="meta small">
        {canActivate
          ? 'As an administrator you can activate an inactive model. Training never activates a model by itself; the server checks the model and your authorization before switching, and records each activation in the audit log.'
          : 'This page is read-only. The active model is managed by research administrators.'}
      </p>
      {dialog ? (
        <ActivationDialog model={dialog.model} activeVersion={activeVersion} auth={auth} onCancel={cancel} onActivated={activated} />
      ) : null}
    </>
  );
}
