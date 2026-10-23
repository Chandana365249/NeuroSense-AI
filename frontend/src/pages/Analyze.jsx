import { useEffect, useId, useRef, useState } from 'react';
import { api } from '../api';
import { demoOptions, isSynthetic, NO_RESULT_MESSAGE, roleNotice, selectionKey, SYNTHETIC_DONE_MESSAGE, trialLabel,
  validateUpload, visibleOutcome } from '../analysis';
import Card from '../components/Card';
import EegPreview from '../components/EegPreview';
import LoadingBlock from '../components/LoadingBlock';
import Notice from '../components/Notice';
import PageHeader from '../components/PageHeader';
import RequireAuth from '../components/RequireAuth';
import ResultPanel from '../components/ResultPanel';
import { classLabel, predictionLabel, pct, sourceLabel, when } from '../components/format';
import { href } from '../router';
import { makeSyntheticSignal, SYNTHETIC_FILE_NAME } from '../synthetic';

// Each staged upload or generated signal gets its own id, so a new file or a new signal is a new selection.
let stagedInputs = 0;
const nextStagedId = () => { stagedInputs += 1; return stagedInputs; };

const SOURCES = [
  { id: 'deap', label: 'DEAP recording', hint: 'Recorded research EEG' },
  { id: 'upload', label: 'Your EEG file', hint: 'Upload a JSON file' },
  { id: 'synthetic', label: 'Synthetic test', hint: 'Generated, not real EEG' },
];

function SourcePicker({ value, onChange, name }) {
  const refs = useRef({});
  // Arrow keys move between options, as in a native radio group.
  const onKeyDown = (event) => {
    const i = SOURCES.findIndex((s) => s.id === value);
    const step = { ArrowRight: 1, ArrowDown: 1, ArrowLeft: -1, ArrowUp: -1 }[event.key];
    if (!step) return;
    event.preventDefault();
    const next = SOURCES[(i + step + SOURCES.length) % SOURCES.length].id;
    onChange(next);
    refs.current[next]?.focus();
  };
  return (
    <div className="source-picker" role="radiogroup" aria-label={name} onKeyDown={onKeyDown}>
      {SOURCES.map((s) => (
        <button key={s.id} type="button" role="radio" aria-checked={value === s.id} tabIndex={value === s.id ? 0 : -1}
          ref={(el) => { refs.current[s.id] = el; }} className={`source-option ${value === s.id ? 'is-selected' : ''}`}
          onClick={() => onChange(s.id)}>
          <span className="source-label">{s.label}</span>
          <span className="source-hint">{s.hint}</span>
        </button>
      ))}
    </div>
  );
}

function DeapPanel({ publicData, busy, onRun, ids, selection, onSelect }) {
  const options = demoOptions(publicData.demo);
  const { participant, trial } = selection;
  const current = options.participants.find((p) => p.id === participant) || options.participants[0];

  // Resolve the default (first participant, first available trial) into the shared selection.
  useEffect(() => {
    if (!current) return;
    const p = current.id;
    const t = current.trials.includes(Number(trial)) && trial !== '' ? trial : String(current.trials[0]);
    if (p !== participant || t !== trial) onSelect({ participant: p, trial: t });
  }, [current, participant, trial, onSelect]);

  if (options.status === 'loading') return <LoadingBlock label="Loading available DEAP participants…" lines={2} />;
  if (options.status === 'error') {
    return (
      <Notice tone="error" title="DEAP participants unavailable"
        action={<button type="button" className="btn btn-secondary btn-small" onClick={publicData.reload}>Retry</button>}>
        {options.message}
      </Notice>
    );
  }
  if (options.status === 'none') {
    return (
      <Notice tone="info" title="No DEAP participant can be offered">
        <p>{options.message}</p>
        <p>Choose “Your EEG file” or “Synthetic test” instead.</p>
      </Notice>
    );
  }
  const trialIndex = Number(trial);
  const valid = current && current.trials.includes(trialIndex);
  const status = current ? roleNotice(current.role) : null;
  const submit = (event) => {
    event.preventDefault();
    if (busy || !valid) return;
    onRun({ deap_reference: { participant_id: current.id, trial_index: trialIndex } },
      selectionKey('deap', { deap: { participant: current.id, trial: String(trialIndex) } }));
  };
  return (
    <form onSubmit={submit} className="form-section" aria-describedby={`${ids}-deap-help`}>
      <p id={`${ids}-deap-help`} className="field-hint">
        One-minute EEG recordings from the DEAP research study ({options.participants.length} authorized participants
        on this server). Each person rated how pleasant each music video felt, so you can compare the estimate with
        their own rating.
      </p>
      <div className="field-row">
        <div className="field">
          <label htmlFor={`${ids}-participant`}>Participant ({options.participants.length} available)</label>
          <select id={`${ids}-participant`} value={current?.id || ''} onChange={(e) => onSelect({ participant: e.target.value, trial })}>
            {options.participants.map((p) => <option key={p.id} value={p.id}>{p.id}</option>)}
          </select>
        </div>
        <div className="field">
          <label htmlFor={`${ids}-trial`}>Trial ({current?.trials.length ?? 0} available)</label>
          <select id={`${ids}-trial`} value={trial} onChange={(e) => onSelect({ participant: current.id, trial: e.target.value })}>
            {(current?.trials || []).map((t) => <option key={t} value={t}>{trialLabel(t)}</option>)}
          </select>
        </div>
      </div>
      {status ? (
        <div className={`role-note role-${status.kind}`} role="note" aria-live="polite">
          <strong>{status.title}.</strong> {status.text}
        </div>
      ) : null}
      <button type="submit" className="btn btn-primary" disabled={!valid} aria-disabled={busy || undefined}>
        {busy ? 'Analyzing…' : `Analyze ${current?.id ?? ''} · ${valid ? trialLabel(trialIndex) : ''}`}
      </button>
    </form>
  );
}

function UploadPanel({ publicData, busy, onRun, onStage, staged, ids }) {
  const info = publicData.modelInfo;
  const req = info?.input_requirements;
  const baseline = info?.preprocessing?.baseline_seconds ?? 0;
  const [error, setError] = useState(null);

  const choose = async (event) => {
    const file = event.target.files?.[0];
    event.target.value = '';
    if (!file || busy) return;
    try {
      const parsed = JSON.parse(await file.text());
      const checked = validateUpload(parsed, req, baseline);
      setError(null);
      onStage({ ...checked, fileName: file.name, id: nextStagedId() });
    } catch (err) {
      onStage(null);
      setError(err instanceof SyntaxError ? 'The file is not valid JSON. Save it as a .json file and try again.' : err.message);
    }
  };

  return (
    <div className="form-section">
      <ul className="format-list" id={`${ids}-format`}>
        <li><strong>{req?.channels ?? '—'} EEG channels</strong> in DEAP (Geneva) order</li>
        <li><strong>{req?.sampling_rate_hz ?? '—'} samples per second</strong>, values in microvolts</li>
        <li><strong>At least {req?.minimum_seconds ?? '—'} seconds</strong> long (at most 120 s)</li>
      </ul>
      <details className="format-details">
        <summary>File format details</summary>
        <p className="field-hint">
          A JSON file containing {'{"signal": [[channel 1 samples…], [channel 2 samples…], …]}'}. Optional:
          {' "sampling_rate"'} (must be {req?.sampling_rate_hz ?? 'the model rate'}) and {'"includes_baseline": true'} if the
          first {baseline} s are a pre-stimulus baseline to drop.
        </p>
      </details>
      <label className="file-drop" htmlFor={`${ids}-file`}>
        <span>{staged ? 'Choose a different file' : 'Choose a JSON file'}</span>
        <input id={`${ids}-file`} type="file" accept=".json,application/json" onChange={choose}
          aria-disabled={busy || undefined} aria-describedby={`${ids}-format`}
          aria-invalid={error ? true : undefined} />
      </label>
      {error ? <p className="form-error" role="alert">{error}</p> : null}
      {staged ? (
        <p className="selection-chip">
          Selected: <strong>{staged.fileName}</strong> · {staged.eeg.signal.length} channels
          {staged.seconds != null ? ` · ${staged.seconds.toFixed(1)} s` : ''}
          {staged.synthetic ? ' · synthetic test file' : ''}
        </p>
      ) : null}
      <button type="button" className="btn btn-primary" disabled={!staged} aria-disabled={busy || undefined}
        onClick={() => { if (!busy && staged) onRun({ eeg: staged.eeg }, selectionKey('upload', { upload: staged })); }}>
        {busy ? 'Analyzing…' : 'Analyze this file'}
      </button>
    </div>
  );
}

function SyntheticPanel({ publicData, busy, onRun, onStage, onAnnounce }) {
  const info = publicData.modelInfo;
  const make = () => validateUpload(makeSyntheticSignal({
    channels: info?.input_requirements?.channels ?? 32, rate: info?.input_requirements?.sampling_rate_hz ?? 128,
  }), info?.input_requirements, info?.preprocessing?.baseline_seconds ?? 0);

  const run = () => {
    if (busy) return;
    const staged = { ...make(), fileName: 'Synthetic test signal', id: nextStagedId() };
    onStage(staged);
    onRun({ eeg: staged.eeg }, selectionKey('synthetic', { synthetic: staged }));
  };
  // Generated in the browser; nothing is sent to the server until the user analyzes it.
  const download = () => {
    const blob = new Blob([JSON.stringify(makeSyntheticSignal())], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = SYNTHETIC_FILE_NAME;
    document.body.appendChild(a);
    a.click();
    a.remove();
    setTimeout(() => URL.revokeObjectURL(url), 0);
    onAnnounce('Synthetic test file downloaded. It is generated noise, not real EEG.');
  };
  return (
    <div className="form-section">
      <p className="field-hint">
        Generates 60 seconds of random noise and sine waves in your browser and runs it through the analysis. Use it to
        check that the pipeline works when you have no EEG file. It is not real EEG, so the output is not an emotion
        estimate, and it is saved in your history as “Synthetic test signal”.
      </p>
      <div className="button-row tight">
        <button type="button" className="btn btn-primary" onClick={run} aria-disabled={busy || undefined}>
          {busy ? 'Analyzing…' : 'Analyze a synthetic test signal'}
        </button>
        <button type="button" className="btn btn-ghost btn-small" onClick={download}>Download the file instead</button>
      </div>
    </div>
  );
}

function RecentList({ recent, warning, onRetry }) {
  return (
    <>
      {warning ? (
        <p className="muted small recent-stale">
          {warning === 'load' ? 'Recent analyses could not be loaded.' : 'This list may not include your latest analysis.'}{' '}
          <button type="button" className="btn btn-secondary btn-small" onClick={onRetry}>Retry</button>
        </p>
      ) : null}
      {!recent ? (warning ? null : <LoadingBlock label="Loading recent analyses…" lines={2} />)
        : !recent.items.length ? <p className="muted">No analyses yet.</p> : (
          <ul className="recent-list">
            {recent.items.map((p) => (
              <li key={p.prediction_id}>
                <a href={href(`/results/${p.prediction_id}`)}>
                  <span className="recent-class">{predictionLabel(p)}{isSynthetic(p) ? null : <span className="muted small"> · {pct(p.confidence)}</span>}</span>
                  <span className="recent-meta">{sourceLabel(p)} · {when(p.created_at)}</span>
                </a>
              </li>
            ))}
          </ul>
        )}
    </>
  );
}

function AnalyzeWorkspace({ auth, publicData }) {
  const ids = useId();
  const [source, setSource] = useState(null);
  const [busy, setBusy] = useState(false);
  const busyRef = useRef(false); // blocks a second submission before React re-renders the disabled buttons
  // Every outcome carries the key of the selection that produced it; see selectionKey / visibleOutcome.
  const [outcome, setOutcome] = useState({ result: null, error: null, pending: null });
  const [deapSelection, setDeapSelection] = useState({ participant: '', trial: '' });
  const [uploaded, setUploaded] = useState(null); // validated file chosen by the user
  const [generated, setGenerated] = useState(null); // synthetic signal generated in the browser
  const [recent, setRecent] = useState(null);
  // null | 'load' (the list could not be loaded) | 'refresh' (a new result was saved but the list is stale)
  const [recentWarning, setRecentWarning] = useState(null);
  const [announcement, setAnnouncement] = useState('');
  const resultHeading = useRef(null);
  // DEAP is the default: when no participant can be offered, its panel explains why and points to the alternatives.
  const activeSource = source || 'deap';
  const currentKey = selectionKey(activeSource, { deap: deapSelection, upload: uploaded, synthetic: generated });
  const shown = visibleOutcome(outcome, currentKey);

  // A different source, participant, trial, file or synthetic signal: drop results and errors of the old selection.
  // The screen-reader announcement belongs to the old outcome too, unless the new selection is being analysed.
  useEffect(() => {
    setOutcome((o) => (o.result || o.error ? { ...o, result: null, error: null } : o));
    if (!busyRef.current) setAnnouncement('');
  }, [currentKey]);

  const refreshRecent = async ({ afterPrediction = false, manual = false } = {}) => {
    try {
      setRecent(await api.predictions(auth.token, { limit: 5 }));
      setRecentWarning(null);
      if (manual) setAnnouncement('Recent analyses refreshed.');
    } catch (err) {
      if (err.status === 401) auth.expire();
      else {
        setRecentWarning(afterPrediction ? 'refresh' : 'load');
        if (afterPrediction) setAnnouncement('This result was saved, but your recent analyses could not be refreshed.');
      }
    }
  };
  useEffect(() => { refreshRecent(); }, [auth.token]); // eslint-disable-line react-hooks/exhaustive-deps

  // Keyboard and screen-reader users land on the outcome; the summary is announced once.
  const finish = (message) => {
    setAnnouncement(message);
    setTimeout(() => resultHeading.current?.focus(), 0);
  };

  // `key` identifies the selection being analysed; the outcome is shown only while that selection is current.
  const run = async (payload, key) => {
    if (busyRef.current) return;
    busyRef.current = true;
    setBusy(true);
    setOutcome({ result: null, error: null, pending: key }); // nothing from an earlier analysis stays on screen
    setAnnouncement('Analyzing the recording…');
    let result;
    try {
      result = await api.predict(auth.token, payload);
    } catch (err) {
      busyRef.current = false;
      setBusy(false);
      if (err.status === 401) { setOutcome({ result: null, error: null, pending: null }); auth.expire(); return; }
      setOutcome({ result: null, pending: null, error: { key,
        message: err.status === 0 ? 'The service is unreachable, so nothing was analysed or saved. Please try again shortly.' : err.message } });
      finish(''); // the error notice is announced immediately (role="alert")
      return;
    }
    // The prediction succeeded and is stored; a failed list refresh must not hide it.
    busyRef.current = false;
    setBusy(false);
    setOutcome({ result: { key, prediction: result }, error: null, pending: null });
    finish(isSynthetic(result) ? SYNTHETIC_DONE_MESSAGE
      : `Analysis complete: ${classLabel(result.prediction)}, model probability ${pct(result.confidence)}.`);
    refreshRecent({ afterPrediction: true });
  };

  if (publicData.status === 'loading') return <LoadingBlock label="Loading the active model…" />;
  if (publicData.status === 'error') {
    return (
      <Notice tone="error" title="The service is unreachable"
        action={<button type="button" className="btn btn-secondary btn-small" onClick={publicData.reload}>Retry</button>}>
        Analyses are unavailable until the connection is restored.
      </Notice>
    );
  }
  if (publicData.modelInfoError) {
    return (
      <Notice tone="error" title="The active model could not be loaded"
        action={<button type="button" className="btn btn-secondary btn-small" onClick={publicData.reload}>Retry</button>}>
        {publicData.modelInfoError}
      </Notice>
    );
  }
  if (!publicData.modelInfo?.available) {
    return <Notice tone="warning" title="No trained model is available">{publicData.modelInfo?.message || 'Analyses are unavailable.'}</Notice>;
  }

  const panelProps = { publicData, busy, onRun: run, ids };
  const preview = activeSource === 'upload' ? uploaded : activeSource === 'synthetic' ? generated : null;
  return (
    <>
      <div className="analyze-grid">
        <div className="stack">
          <Card title="1. Choose an input" titleId="choose-input">
            <SourcePicker value={activeSource} onChange={setSource} name="Input source" />
            <div className="source-panel">
              {activeSource === 'deap' ? <DeapPanel {...panelProps} selection={deapSelection} onSelect={setDeapSelection} /> : null}
              {activeSource === 'upload' ? <UploadPanel {...panelProps} staged={uploaded} onStage={setUploaded} /> : null}
              {activeSource === 'synthetic' ? <SyntheticPanel {...panelProps} onStage={setGenerated} onAnnounce={setAnnouncement} /> : null}
            </div>
          </Card>
          <Card title="Recent analyses" titleId="analyze-recent" actions={<a href={href('/history')} className="small">View history</a>}>
            <RecentList recent={recent} warning={recentWarning} onRetry={() => refreshRecent({ manual: true })} />
          </Card>
        </div>

        <Card title="2. Result" titleId="result" titleRef={resultHeading} className="result-card">
          <p className="visually-hidden" role="status">{announcement}</p>
          <div aria-busy={shown.kind === 'loading'} data-outcome={shown.kind}>
            {shown.kind === 'loading' ? <LoadingBlock label="Analyzing the recording…" announce={false} /> : null}
            {shown.kind === 'error' ? <Notice tone="error" title="The analysis could not be completed">{shown.message}</Notice> : null}
            {shown.kind === 'result' && recentWarning === 'refresh' ? (
              <Notice tone="warning" announce={false}>
                This result was saved, but your recent analyses could not be refreshed. It will appear in your history.
              </Notice>
            ) : null}
            {shown.kind === 'result' ? <ResultPanel prediction={shown.prediction} showDetailLink /> : null}
            {shown.kind === 'empty' ? (
              <div className="empty-state">
                <p><strong>{NO_RESULT_MESSAGE}</strong></p>
                <p className="muted">Each result is saved to your history.</p>
              </div>
            ) : null}
          </div>
        </Card>
      </div>
      {preview ? (
        <Card title={preview.synthetic ? 'Synthetic signal preview' : 'Uploaded file preview'} titleId="upload-preview" className="preview-card">
          {preview.synthetic ? (
            <Notice tone="warning" title="Synthetic test signal" announce={false}>
              Generated test signal, not recorded EEG. Its result is not a meaningful valence estimate.
            </Notice>
          ) : null}
          <EegPreview key={preview.fileName + preview.eeg.signal[0].length} eeg={preview.eeg} fileName={preview.fileName} rateGiven={preview.rateGiven} />
        </Card>
      ) : null}
    </>
  );
}

export default function Analyze({ auth, publicData }) {
  return (
    <>
      <PageHeader title="Analyze EEG" subtitle="Choose an EEG input, run the analysis, and read the result." />
      <div className="intro-strip">
        <p>
          The model estimates whether a one-minute EEG recording looks more like trials people rated as
          <strong> more pleasant</strong> (higher self-reported valence) or <strong>less pleasant</strong> (lower), on
          the 1–9 scale used in the DEAP study.
        </p>
        <p className="muted small">
          Research prototype: for people it had not seen, estimates were close to chance. It does not detect emotions or
          diagnose anything. <a href={href('/model')}>Read the evidence</a>
        </p>
      </div>
      <RequireAuth auth={auth} reason="Sign in to analyze EEG recordings. Your analyses are saved to your history.">
        <AnalyzeWorkspace auth={auth} publicData={publicData} />
      </RequireAuth>
    </>
  );
}
