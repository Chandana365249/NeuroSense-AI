import { useEffect, useId, useRef, useState } from 'react';
import { api } from '../api';
import Card from '../components/Card';
import EegPreview from '../components/EegPreview';
import LoadingBlock from '../components/LoadingBlock';
import Notice from '../components/Notice';
import PageHeader from '../components/PageHeader';
import RequireAuth from '../components/RequireAuth';
import ResultPanel from '../components/ResultPanel';
import { classLabel, pct, RELIABILITY_NOTE, sourceLabel, VALENCE_EXPLANATION, when } from '../components/format';
import { href } from '../router';

function parseUpload(parsed) {
  const signal = Array.isArray(parsed) ? parsed : parsed?.signal;
  if (!Array.isArray(signal) || !signal.length || !Array.isArray(signal[0])) {
    throw new Error('The file must contain a [channels][samples] array, or {"signal": [[...], ...]}.');
  }
  return {
    signal,
    sampling_rate: parsed.sampling_rate || 128,
    includes_baseline: !!parsed.includes_baseline,
  };
}

function AnalyzeWorkspace({ auth, publicData }) {
  const ids = useId();
  const participants = publicData.demo?.held_out_participants || [];
  const [participant, setParticipant] = useState('');
  const [trial, setTrial] = useState('0');
  const [busy, setBusy] = useState(false);
  const [prediction, setPrediction] = useState(null);
  const [error, setError] = useState(null);
  const [recent, setRecent] = useState(null);
  // null | 'load' (the list could not be loaded) | 'refresh' (a new result was saved but the list is stale)
  const [recentWarning, setRecentWarning] = useState(null);
  const [announcement, setAnnouncement] = useState('');
  const [preview, setPreview] = useState(null); // { eeg, fileName, rateGiven } of the last uploaded file
  const resultHeading = useRef(null);

  useEffect(() => {
    if (!participant && participants.length) setParticipant(participants[0]);
  }, [participant, participants]);

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

  // Keyboard and screen-reader users land on the outcome; the summary is announced once.
  const finish = (message) => {
    setAnnouncement(message);
    setTimeout(() => resultHeading.current?.focus(), 0);
  };

  useEffect(() => { refreshRecent(); }, [auth.token]); // eslint-disable-line react-hooks/exhaustive-deps

  const showError = (message) => {
    setPrediction(null);
    setError(message);
    finish(''); // the error notice is announced immediately (role="alert")
  };

  const run = async (payload) => {
    setBusy(true);
    setError(null);
    setAnnouncement('Analyzing the recording…');
    let result;
    try {
      result = await api.predict(auth.token, payload);
    } catch (err) {
      setBusy(false);
      if (err.status === 401) { auth.expire(); return; }
      showError(err.status === 0 ? 'The service is unreachable. Please try again shortly.' : err.message);
      return;
    }
    // The prediction succeeded and is stored; a failed list refresh must not hide it.
    setPrediction(result);
    setBusy(false);
    finish(`Analysis complete: ${classLabel(result.prediction)}, model probability ${pct(result.confidence)}.`);
    refreshRecent({ afterPrediction: true });
  };

  const trialNumber = Number(trial);
  const trialValid = trial !== '' && Number.isInteger(trialNumber) && trialNumber >= 0 && trialNumber <= 39;

  const submitDeap = (event) => {
    event.preventDefault();
    if (busy || !participant || !trialValid) return;
    setPreview(null); // the preview describes an uploaded file, not a DEAP trial
    run({ deap_reference: { participant_id: participant, trial_index: trialNumber } });
  };

  const handleUpload = async (event) => {
    const file = event.target.files?.[0];
    event.target.value = '';
    if (!file || busy) return;
    let eeg;
    let rateGiven;
    try {
      const parsed = JSON.parse(await file.text());
      eeg = parseUpload(parsed);
      rateGiven = typeof parsed?.sampling_rate === 'number';
    } catch (err) {
      setPreview(null);
      showError(err instanceof SyntaxError ? 'The file is not valid JSON.' : err.message);
      return;
    }
    setPreview({ eeg, fileName: file.name, rateGiven });
    run({ eeg });
  };

  if (publicData.status === 'loading') return <LoadingBlock label="Loading the active model…" />;
  if (publicData.status === 'error') {
    return <Notice tone="error" title="The service is unreachable">Analyses are unavailable until the connection is restored.</Notice>;
  }
  if (!publicData.modelInfo?.available) {
    return <Notice tone="warning" title="No trained model is available">{publicData.modelInfo?.message || 'Analyses are unavailable.'}</Notice>;
  }

  return (
    <>
      <div className="analyze-grid">
        <div className="stack">
          <Card title="Choose an input" titleId="choose-input">
            <form onSubmit={submitDeap} className="form-section" aria-describedby={`${ids}-deap-help`}>
              <h3 className="subhead">DEAP recording</h3>
              <p id={`${ids}-deap-help`} className="field-hint">
                Participants listed here were not used to train the active model. Each has 40 one-minute trials (0–39).
              </p>
              <div className="field-row">
                <div className="field">
                  <label htmlFor={`${ids}-participant`}>Participant</label>
                  <select id={`${ids}-participant`} value={participant} onChange={(e) => setParticipant(e.target.value)} disabled={!participants.length}>
                    {participants.map((p) => <option key={p} value={p}>{p}</option>)}
                  </select>
                </div>
                <div className="field">
                  <label htmlFor={`${ids}-trial`}>Trial</label>
                  <input id={`${ids}-trial`} type="number" inputMode="numeric" min="0" max="39" step="1" value={trial}
                    onChange={(e) => setTrial(e.target.value)} aria-invalid={!trialValid}
                    aria-describedby={!trialValid ? `${ids}-trial-error` : undefined} />
                </div>
              </div>
              {!trialValid ? <p id={`${ids}-trial-error`} className="form-error">Enter a whole number from 0 to 39.</p> : null}
              <button type="submit" className="btn btn-primary" disabled={!participant || !trialValid} aria-disabled={busy || undefined}>
                {busy ? 'Analyzing…' : 'Analyze DEAP trial'}
              </button>
            </form>

            <div className="divider" role="presentation"><span>or</span></div>

            <div className="form-section">
              <h3 className="subhead">Upload an EEG file</h3>
              <label className="file-drop" htmlFor={`${ids}-file`}>
                <span>Choose a JSON file</span>
                <input id={`${ids}-file`} type="file" accept=".json,application/json" onChange={handleUpload}
                  aria-disabled={busy || undefined} aria-describedby={`${ids}-file-help`} />
              </label>
              <p id={`${ids}-file-help`} className="field-hint">
                32 channels in DEAP order × samples at 128 Hz, as {'{"signal": [[...], ...]}'}. Set
                {' "includes_baseline": true'} if the first 3 s are a pre-stimulus baseline.
              </p>
            </div>
          </Card>

          <Card title="Recent analyses" titleId="analyze-recent" actions={<a href={href('/history')} className="small">View history</a>}>
            {recentWarning ? (
              <p className="muted small recent-stale">
                {recentWarning === 'load' ? 'Recent analyses could not be loaded.' : 'This list may not include your latest analysis.'}{' '}
                <button type="button" className="btn btn-secondary btn-small" onClick={() => refreshRecent({ manual: true })}>Retry</button>
              </p>
            ) : null}
            {!recent ? (recentWarning ? null : <LoadingBlock label="Loading recent analyses…" lines={2} />)
              : !recent.items.length ? <p className="muted">No analyses yet.</p> : (
                <ul className="recent-list">
                  {recent.items.map((p) => (
                    <li key={p.prediction_id}>
                      <a href={href(`/results/${p.prediction_id}`)}>
                        <span className="recent-class">{classLabel(p.prediction)}</span>
                        <span className="recent-meta">{sourceLabel(p)} · {when(p.created_at)}</span>
                      </a>
                    </li>
                  ))}
                </ul>
              )}
          </Card>
        </div>

        <Card title="Result" titleId="result" titleRef={resultHeading} className="result-card">
          <p className="visually-hidden" role="status">{announcement}</p>
          <div aria-busy={busy}>
            {busy ? <LoadingBlock label="Analyzing the recording…" announce={false} /> : null}
            {!busy && error ? <Notice tone="error" title="The analysis could not be completed">{error}</Notice> : null}
            {!busy && !error && prediction && recentWarning === 'refresh' ? (
              <Notice tone="warning" announce={false}>
                This result was saved, but your recent analyses could not be refreshed. It will appear in your history.
              </Notice>
            ) : null}
            {!busy && !error && prediction ? <ResultPanel prediction={prediction} showDetailLink /> : null}
            {!busy && !error && !prediction ? (
              <p className="muted empty">Choose a DEAP trial or upload a file to see an estimate here.</p>
            ) : null}
          </div>
        </Card>
      </div>
      {preview ? (
        <Card title="Uploaded file preview" titleId="upload-preview" className="preview-card">
          <EegPreview key={preview.fileName + preview.eeg.signal.length} eeg={preview.eeg} fileName={preview.fileName} rateGiven={preview.rateGiven} />
        </Card>
      ) : null}
    </>
  );
}

export default function Analyze({ auth, publicData }) {
  return (
    <>
      <PageHeader title="Analyze EEG" subtitle="Estimate the self-reported valence category of one EEG recording." />
      <Notice tone="warning" title="Reliability">{RELIABILITY_NOTE}</Notice>
      <p className="explainer">{VALENCE_EXPLANATION}</p>
      <RequireAuth auth={auth} reason="Sign in to analyze EEG recordings. Your analyses are saved to your history.">
        <AnalyzeWorkspace auth={auth} publicData={publicData} />
      </RequireAuth>
    </>
  );
}
