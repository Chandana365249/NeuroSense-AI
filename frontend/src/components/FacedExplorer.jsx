import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { api } from '../api';
import { datasetPath } from '../datasets';
import {
  ROLE_TEXT, STATUS_SHORT, STATUS_TONE, facedResultKey, filterParticipants, visibleFacedResult,
} from '../faced';
import { href, navigate } from '../router';
import Card from './Card';
import { PipelineStrip } from './DatasetTabs';
import LoadingBlock from './LoadingBlock';
import Notice from './Notice';
import Stat from './Stat';
import StatusPill from './StatusPill';


export function useLoad(fn, deps, auth) {
  const [state, setState] = useState({ status: 'loading' });
  const load = useCallback(async () => {
    setState({ status: 'loading' });
    try {
      setState({ status: 'ready', data: await fn() });
    } catch (err) {
      if (err?.status === 401) auth?.expire();
      else setState({ status: 'error', error: err });
    }
  }, deps); // eslint-disable-line react-hooks/exhaustive-deps
  useEffect(() => { load(); }, [load]);
  return [state, load];
}

export function errorText(err) {
  if (err?.status === 0) return 'The service is unreachable.';
  if (err?.status === 503) return err.message || 'FACED is not available on this server.';
  return err?.message || 'The request failed.';
}

function retryButton(onClick) {
  return <button type="button" className="btn btn-secondary btn-small" onClick={onClick}>Retry</button>;
}

/** Waits for the FACED status and refuses to show a workflow for a dataset this server has not validated. */
function FacedGate({ state, reload, children }) {
  if (state.status === 'loading') return <LoadingBlock label="Checking the FACED installation…" />;
  if (state.status === 'error') return <Notice tone="error" title="Could not load FACED" action={retryButton(reload)}>{errorText(state.error)}</Notice>;
  const st = state.data;
  if (!st.validated) {
    return (
      <Notice tone="warning" title="FACED is not available on this server">
        {st.installed ? 'The files are present but have not been validated yet.' : 'The FACED files were not found in the configured location.'}
      </Notice>
    );
  }
  return children(st);
}

function ModelSummary({ model }) {
  if (!model?.available) {
    return (
      <Notice tone="warning" title="No FACED model available">
        {model?.message || 'A FACED model has not been trained and evaluated on this server.'} The dataset can still be
        browsed, and the DEAP model is never used in its place.
      </Notice>
    );
  }
  return (
    <div className="faced-model">
      <div className="dataset-status">
        <StatusPill tone="warn">Experimental research model</StatusPill>
        <span className="muted small">{model.name} · not the active DEAP model · FACED recordings only</span>
      </div>
      <p className="muted small">{model.dataset_compatibility}</p>
      <p className="meta"><a href={href(datasetPath('/model', 'faced'))}>About the FACED model</a></p>
    </div>
  );
}

function ParticipantDirectory({ listState, reload, selected, onSelect }) {
  const [query, setQuery] = useState('');
  const rows = useMemo(() => (listState.status === 'ready' ? filterParticipants(listState.data.participants, query) : []), [listState, query]);
  return (
    <>
      <label className="field">
        <span>Search participants</span>
        <input type="search" value={query} onChange={(e) => setQuery(e.target.value)} placeholder="e.g. 012 or sub-012" />
      </label>
      {listState.status === 'loading' ? <LoadingBlock label="Loading participants…" lines={2} /> : null}
      {listState.status === 'error' ? <Notice tone="error" title="Could not load participants" action={retryButton(reload)}>{errorText(listState.error)}</Notice> : null}
      {listState.status === 'ready' ? (
        <>
          <p className="muted small" aria-live="polite">{rows.length} of {listState.data.count} participants</p>
          <ul className="faced-list" aria-label="FACED participants">
            {rows.map((r) => (
              <li key={r.participant_id}>
                <button type="button" className={`faced-item${selected === r.participant_id ? ' selected' : ''}`}
                  aria-pressed={selected === r.participant_id} onClick={() => onSelect(r.participant_id)}>
                  <span>{r.label}</span>
                  <span className="muted small">{r.participant_id} · {ROLE_TEXT[r.role]} · {r.usable_trials} trials</span>
                </button>
              </li>
            ))}
          </ul>
        </>
      ) : null}
    </>
  );
}

function useTrials(token, participant, auth) {
  const [trials, setTrials] = useState({ status: 'idle' });
  useEffect(() => {
    if (!participant) { setTrials({ status: 'idle' }); return undefined; }
    let cancelled = false;
    setTrials({ status: 'loading' });
    api.facedTrials(token, participant)
      .then((data) => { if (!cancelled) setTrials({ status: 'ready', data }); })
      .catch((error) => {
        if (cancelled) return;
        if (error?.status === 401) auth?.expire();
        else setTrials({ status: 'error', error });
      });
    return () => { cancelled = true; };
  }, [participant, token]); // eslint-disable-line react-hooks/exhaustive-deps
  return trials;
}

function PipelineSteps() {
  return (
    <ol className="pipeline-steps">
      <li><strong>Window:</strong> the last 28 seconds of each video.</li>
      <li><strong>Channels:</strong> the 30 scalp channels shared by both recording layouts, matched by name.</li>
      <li><strong>Reference:</strong> common average across those channels.</li>
      <li><strong>Sampling rate:</strong> 1000 Hz recordings are resampled to 250 Hz; 250 Hz recordings are used as recorded.</li>
      <li><strong>Filter and artifacts:</strong> 4–45 Hz band-pass, robust clipping of extreme values, 4-second segments, heavily clipped segments rejected.</li>
      <li><strong>Features:</strong> 412 per segment (band power, spectral entropy, Hjorth, log variance, hemispheric asymmetry).</li>
      <li><strong>Trial estimate:</strong> the mean segment probability, compared with 0.5.</li>
    </ol>
  );
}

/** Dataset page: what FACED is on this server, and a read-only directory of participants and recordings. */
export function FacedDatasetPanel({ auth }) {
  const token = auth?.token;
  const [statusState, reloadStatus] = useLoad(() => api.facedStatus(token), [token], auth);
  const [listState, reloadList] = useLoad(() => api.facedParticipants(token), [token], auth);
  const [participant, setParticipant] = useState(null);
  const trials = useTrials(token, participant, auth);

  return (
    <FacedGate state={statusState} reload={reloadStatus}>
      {(st) => (
        <>
          <Card title="FACED (NEMAR nm000112, version 1.1.4)" titleId="faced-about">
            <p>
              Healthy adults watched 28 emotional film clips while 32-channel EEG was recorded. After each clip they rated it,
              including <strong>valence</strong> on a 0–7 scale. Here, a clip rated above 3.5 counts as higher valence.
            </p>
            <div className="stats-grid stats-grid-wide">
              <Stat label="Participants validated on this server" value={st.participants_valid}
                hint={`${st.participants_declared} listed by the dataset${st.participants_failed ? `; ${st.participants_failed} failed validation` : ''}`} />
              <Stat label="Usable trials" value={st.trials_used} hint={`${st.segments} segments of 4 s`} />
              <Stat label="Recording rates" value={Object.entries(st.sampling_rates).map(([hz, n]) => `${Number(hz)} Hz × ${n}`).join(', ')} hint={`resampled to ${st.target_sampling_rate} Hz`} />
              <Stat label="EEG channels used" value={st.channels.length} hint="scalp channels shared by both recording layouts (32 recorded)" />
            </div>
            <p className="muted small">{st.citation} Licence {st.license}.</p>
          </Card>

          <div className="grid-2 faced-grid">
            <Card title={`FACED processing pipeline (${st.pipeline})`} titleId="faced-pipeline"><PipelineSteps /></Card>
            <Card title="FACED model and evaluation" titleId="faced-model"><ModelSummary model={st.model} /></Card>
          </div>

          <div className="grid-2 faced-grid">
            <Card title="Participants" titleId="faced-participants">
              <ParticipantDirectory listState={listState} reload={reloadList} selected={participant} onSelect={setParticipant} />
            </Card>
            <Card title="Recordings and trials" titleId="faced-trials">
              {!participant ? <p className="muted">Choose a participant to see their recording and video trials.</p> : null}
              {trials.status === 'loading' ? <LoadingBlock label="Loading trials…" lines={2} /> : null}
              {trials.status === 'error' ? <Notice tone="error" title="Could not load trials">{errorText(trials.error)}</Notice> : null}
              {trials.status === 'ready' ? (
                <>
                  <p className="small muted">{trials.data.label} ({trials.data.participant_id}) · one continuous recording at {trials.data.sampling_rate} Hz · {trials.data.trials.length} video trials</p>
                  <div className="table-wrap faced-trial-table">
                    <table className="table">
                      <caption className="visually-hidden">Video trials of {trials.data.label}</caption>
                      <thead><tr><th scope="col">Video</th><th scope="col">Own valence</th><th scope="col">Evaluation</th><th scope="col"><span className="visually-hidden">Action</span></th></tr></thead>
                      <tbody>
                        {trials.data.trials.map((t) => (
                          <tr key={t.video_index}>
                            <th scope="row">{t.video_index}: {t.title || 'untitled'} <span className="muted small">({t.targeted_emotion})</span></th>
                            <td>{typeof t.self_reported_valence === 'number' ? t.self_reported_valence.toFixed(2) : 'n/a'}</td>
                            <td>{STATUS_SHORT[t.evaluation_status] || 'Not evaluated'}</td>
                            <td>{t.usable ? <a href={href(datasetPath('/analyze', 'faced', { participant: participant, video: t.video_index }))}>Analyze</a> : 'Not usable'}</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </>
              ) : null}
            </Card>
          </div>
        </>
      )}
    </FacedGate>
  );
}

/** Status of the analysis; a finished analysis opens its own result page. */
function TrialResult({ outcome }) {
  if (!outcome) return null;
  if (outcome.status === 'running') return <LoadingBlock label="Running the FACED pipeline and model on this recording…" lines={2} />;
  if (outcome.status === 'error') return <Notice tone="error" title="Analysis failed">{errorText(outcome.error)}</Notice>;
  return <LoadingBlock label="Opening the result…" lines={1} />;
}

export function FacedAnalyze({ auth, initial }) {
  const token = auth?.token;
  const [statusState, reloadStatus] = useLoad(() => api.facedStatus(token), [token], auth);
  const [listState, reloadList] = useLoad(() => api.facedParticipants(token), [token], auth);
  const [selection, setSelection] = useState({ participant: initial.participant, video: initial.participant ? initial.video : null });
  const { participant, video } = selection;
  const trials = useTrials(token, participant, auth);
  const [outcome, setOutcome] = useState(null);
  const latest = useRef(null);
  const resultHeading = useRef(null);

  // A link to another participant or video replaces the current selection.
  useEffect(() => {
    if (initial.participant) setSelection({ participant: initial.participant, video: initial.video });
  }, [initial.participant, initial.video]);

  const model = statusState.status === 'ready' ? statusState.data.model : null;
  const modelName = model?.available ? model.name : null;
  const key = facedResultKey(participant, video, modelName);
  const shown = visibleFacedResult(outcome, key);

  // Any change of participant, video or model clears the previous result.
  useEffect(() => { setOutcome(null); latest.current = key; }, [key]);

  async function analyze() {
    if (!key) return;
    const requestKey = key;
    setOutcome({ key: requestKey, status: 'running' });
    try {
      const data = await api.facedPredict(token, participant, video);
      if (latest.current === requestKey) {
        setOutcome({ key: requestKey, status: 'done', data });
        navigate(`/results/${data.prediction_id}`); // every result opens on its own page
      }
    } catch (error) {
      if (error?.status === 401) { auth?.expire(); return; }
      if (latest.current === requestKey) setOutcome({ key: requestKey, status: 'error', error });
    }
  }

  const selectedTrial = trials.status === 'ready' ? trials.data.trials.find((t) => t.video_index === video) : null;

  return (
    <FacedGate state={statusState} reload={reloadStatus}>
      {(st) => (
        <>
          <div className="intro-strip">
            <p>
              The FACED model estimates whether a trial looks more like clips people rated as <strong>higher</strong> or
              <strong> lower valence</strong>, on the 0–7 scale used in the FACED study.
            </p>
            <p className="muted small">Research prototype. It does not detect emotions or diagnose anything.</p>
          </div>
          <PipelineStrip dataset={`FACED ${st.version}`} pipeline={`${st.pipeline} (30 channels, 250 Hz, 412 features)`}
            model={modelName || 'None available'}
            note="This recording is processed only by the FACED pipeline and model. If either is unavailable the analysis stops; the DEAP model is never used instead." />
          {!model?.available ? <ModelSummary model={model} /> : null}
          <div className="analyze-grid">
            <div className="stack">
              <Card title="1. Choose a participant" titleId="faced-participants">
                <ParticipantDirectory listState={listState} reload={reloadList} selected={participant}
                  onSelect={(p) => setSelection({ participant: p, video: null })} />
              </Card>
            </div>
            <div className="stack">
              <Card title="2. Choose a recording" titleId="faced-trials">
                {!participant ? <p className="muted">Choose a participant to see their recording and video trials.</p> : null}
                {trials.status === 'loading' ? <LoadingBlock label="Loading trials…" lines={2} /> : null}
                {trials.status === 'error' ? <Notice tone="error" title="Could not load trials">{errorText(trials.error)}</Notice> : null}
                {trials.status === 'ready' ? (
                  <>
                    <p className="small muted">{trials.data.label} ({trials.data.participant_id}) · one continuous recording at {trials.data.sampling_rate} Hz · {trials.data.trials.length} video trials</p>
                    <label className="field">
                      <span>Video trial</span>
                      <select value={video ?? ''} onChange={(e) => setSelection({ participant, video: e.target.value ? Number(e.target.value) : null })}>
                        <option value="">Choose a video…</option>
                        {trials.data.trials.map((t) => (
                          <option key={t.video_index} value={t.video_index} disabled={!t.usable}>
                            Video {t.video_index}: {t.title || 'untitled'} ({t.targeted_emotion}){t.usable ? '' : ' (not usable)'}
                          </option>
                        ))}
                      </select>
                    </label>
                    {selectedTrial ? (
                      <p className="small">
                        <StatusPill tone={STATUS_TONE[selectedTrial.evaluation_status] || 'neutral'}>{STATUS_SHORT[selectedTrial.evaluation_status] || 'Not evaluated'}</StatusPill>{' '}
                        {selectedTrial.evaluation_text}
                      </p>
                    ) : null}
                    <button type="button" className="btn btn-primary" onClick={analyze} disabled={!key || shown?.status === 'running'}>
                      {shown?.status === 'running' ? 'Analyzing…' : 'Analyze this trial with the FACED model'}
                    </button>
                    {!modelName ? <p className="muted small">No FACED model is available, so trials cannot be analyzed.</p> : null}
                  </>
                ) : null}
              </Card>
              <Card title="3. Run the analysis" titleId="faced-result-card" titleRef={resultHeading} className="result-card">
                <div aria-live="polite">
                  {shown ? <TrialResult outcome={shown} />
                    : <p className="muted">Choose a participant and a video, then click Analyze. The result opens on its own page and is saved to your history.</p>}
                </div>
              </Card>
            </div>
          </div>
        </>
      )}
    </FacedGate>
  );
}
