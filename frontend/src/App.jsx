import { useCallback, useEffect, useState } from 'react';
import { api } from './api';

const NA = 'Not available';
const TOKEN_KEY = 'neurosense_token';

const pct = (v) => (typeof v === 'number' ? `${(v * 100).toFixed(1)}%` : NA);
const num = (v, d = 3) => (typeof v === 'number' ? v.toFixed(d) : NA);
const when = (iso) => (iso ? new Date(iso).toLocaleString() : NA);

function readToken() {
  try {
    return sessionStorage.getItem(TOKEN_KEY);
  } catch {
    return null;
  }
}

function saveToken(token) {
  try {
    if (token) sessionStorage.setItem(TOKEN_KEY, token);
    else sessionStorage.removeItem(TOKEN_KEY);
  } catch {
    /* storage unavailable; token stays in memory only */
  }
}

function Card({ id, title, children, className = '' }) {
  return (
    <section id={id} className={`card ${className}`}>
      <h3>{title}</h3>
      {children}
    </section>
  );
}

function Stat({ label, value, hint }) {
  return (
    <div className="stat">
      <span className="stat-label">{label}</span>
      <span className="stat-value">{value}</span>
      {hint ? <span className="stat-hint">{hint}</span> : null}
    </div>
  );
}

function AuthPanel({ user, onAuth, onLogout }) {
  const [mode, setMode] = useState('login');
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [fullName, setFullName] = useState('');
  const [error, setError] = useState(null);
  const [busy, setBusy] = useState(false);

  if (user) {
    return (
      <div className="auth-row">
        <span>
          Signed in as <strong>{user.email}</strong> {user.is_admin ? <span className="badge">admin</span> : null}
        </span>
        <button className="secondary" onClick={onLogout}>Sign out</button>
      </div>
    );
  }

  const submit = async (event) => {
    event.preventDefault();
    setBusy(true);
    setError(null);
    try {
      if (mode === 'register') await api.register(email, password, fullName);
      const { access_token: token } = await api.login(email, password);
      onAuth(token);
    } catch (err) {
      setError(err.message);
    } finally {
      setBusy(false);
    }
  };

  return (
    <form className="auth-form" onSubmit={submit}>
      <div className="tabs">
        <button type="button" className={mode === 'login' ? '' : 'secondary'} onClick={() => setMode('login')}>Sign in</button>
        <button type="button" className={mode === 'register' ? '' : 'secondary'} onClick={() => setMode('register')}>Register</button>
      </div>
      {mode === 'register' ? (
        <input placeholder="Full name (optional)" value={fullName} onChange={(e) => setFullName(e.target.value)} />
      ) : null}
      <input type="email" placeholder="Email" value={email} onChange={(e) => setEmail(e.target.value)} required />
      <input type="password" placeholder="Password (min 8 characters)" value={password}
        onChange={(e) => setPassword(e.target.value)} minLength={8} required />
      <button type="submit" disabled={busy}>{busy ? 'Please wait...' : mode === 'login' ? 'Sign in' : 'Create account'}</button>
      {error ? <p className="error">{error}</p> : null}
    </form>
  );
}

function App() {
  const [token, setToken] = useState(readToken);
  const [user, setUser] = useState(null);
  const [health, setHealth] = useState(null);
  const [healthError, setHealthError] = useState(null);
  const [modelInfo, setModelInfo] = useState(null);
  const [metrics, setMetrics] = useState(null);
  const [training, setTraining] = useState(null);
  const [demo, setDemo] = useState(null);
  const [history, setHistory] = useState(null);
  const [prediction, setPrediction] = useState(null);
  const [predictError, setPredictError] = useState(null);
  const [busy, setBusy] = useState(false);
  const [participant, setParticipant] = useState('');
  const [trial, setTrial] = useState(0);
  const [uploadError, setUploadError] = useState(null);
  const [runCv, setRunCv] = useState(true);

  const loadPublic = useCallback(async () => {
    try {
      setHealth(await api.health());
      setHealthError(null);
    } catch (err) {
      setHealth(null);
      setHealthError(err.message);
      return;
    }
    const [info, m, status, trials] = await Promise.allSettled([
      api.modelInfo(), api.metrics(), api.trainingStatus(), api.demoTrials(),
    ]);
    setModelInfo(info.status === 'fulfilled' ? info.value : null);
    setMetrics(m.status === 'fulfilled' ? m.value : null);
    setTraining(status.status === 'fulfilled' ? status.value : null);
    if (trials.status === 'fulfilled') {
      setDemo(trials.value);
      setParticipant((p) => p || trials.value.held_out_participants[0] || '');
    }
  }, []);

  const loadPrivate = useCallback(async () => {
    if (!token) {
      setUser(null);
      setHistory(null);
      return;
    }
    try {
      setUser(await api.me(token));
      setHistory(await api.predictions(token));
    } catch (err) {
      if (err.status === 401) {
        saveToken(null);
        setToken(null);
      }
    }
  }, [token]);

  useEffect(() => { loadPublic(); }, [loadPublic]);
  useEffect(() => { loadPrivate(); }, [loadPrivate]);

  useEffect(() => {
    if (training?.state !== 'running') return undefined;
    const id = setInterval(async () => {
      try {
        const status = await api.trainingStatus();
        setTraining(status);
        if (status.state !== 'running') loadPublic();
      } catch {
        /* keep polling */
      }
    }, 3000);
    return () => clearInterval(id);
  }, [training?.state, loadPublic]);

  const handleAuth = (newToken) => {
    saveToken(newToken);
    setToken(newToken);
  };

  const handleLogout = () => {
    saveToken(null);
    setToken(null);
    setPrediction(null);
  };

  const handleTrain = async () => {
    try {
      setTraining(await api.train(token, runCv));
    } catch (err) {
      setTraining({ state: 'failed', error: err.message });
    }
  };

  const runPrediction = async (payload) => {
    setBusy(true);
    setPredictError(null);
    try {
      setPrediction(await api.predict(token, payload));
      setHistory(await api.predictions(token));
    } catch (err) {
      setPrediction(null);
      setPredictError(err.message);
    } finally {
      setBusy(false);
    }
  };

  const handleUpload = async (event) => {
    const file = event.target.files?.[0];
    event.target.value = '';
    setUploadError(null);
    if (!file) return;
    try {
      const parsed = JSON.parse(await file.text());
      const signal = Array.isArray(parsed) ? parsed : parsed.signal;
      if (!Array.isArray(signal) || !Array.isArray(signal[0])) {
        throw new Error('File must contain a [channels][samples] array, or {"signal": [[...]]}');
      }
      await runPrediction({ eeg: { signal, sampling_rate: parsed.sampling_rate || 128, includes_baseline: !!parsed.includes_baseline } });
    } catch (err) {
      setUploadError(err.message);
    }
  };

  const backendOk = health?.status === 'ok';
  const modelReady = modelInfo?.available;
  const test = metrics?.available ? metrics.test : null;
  const reference = metrics?.available ? metrics.test_reference_majority : null;
  const cv = metrics?.available ? metrics.cross_validation : null;

  return (
    <div className="app-shell">
      <aside className="sidebar">
        <h1>NeuroSense AI</h1>
        <p>Emotion-related state estimation from EEG (DEAP research prototype)</p>
        <nav>
          <a href="#status">Status</a>
          <a href="#account">Account</a>
          <a href="#model">Model &amp; training</a>
          <a href="#metrics">Evaluation</a>
          <a href="#predict">Prediction</a>
          <a href="#history">History</a>
        </nav>
      </aside>

      <main className="main-content">
        <section className="hero-card" id="status">
          <div>
            <h2>EEG emotion recognition: Review 2 backend</h2>
            <p className="muted">
              Real DEAP recordings → preprocessing → features → participant-aware evaluation → API.
              Outputs are estimates of self-reported affect, not a diagnosis.
            </p>
          </div>
        </section>

        <section className="stats-grid">
          <Stat label="Backend" value={health ? (backendOk ? 'Online' : 'Degraded') : healthError || 'Checking...'} />
          <Stat label="Database" value={health ? `${health.database.connected ? 'Connected' : 'Down'} (${health.database.dialect})` : NA} />
          <Stat label="Dataset" value={health ? (health.dataset_available ? 'Available' : 'Missing') : NA} />
          <Stat label="Active model" value={modelReady ? modelInfo.algorithm : NA} hint={modelReady ? modelInfo.version : null} />
        </section>

        <Card id="account" title="Account">
          {health ? <AuthPanel user={user} onAuth={handleAuth} onLogout={handleLogout} /> : <p>{NA}: backend offline</p>}
        </Card>

        <div className="two-col">
          <Card id="model" title="Active model">
            {modelReady ? (
              <dl className="kv">
                <dt>Version</dt><dd>{modelInfo.version}</dd>
                <dt>Algorithm</dt><dd>{modelInfo.algorithm_description}</dd>
                <dt>Target</dt><dd>{modelInfo.label?.description}</dd>
                <dt>Classes</dt><dd>{modelInfo.class_labels?.join(', ')}</dd>
                <dt>Features</dt><dd>{modelInfo.feature_schema?.count}</dd>
                <dt>Trained</dt><dd>{when(modelInfo.trained_at)}</dd>
                <dt>Train participants</dt><dd>{modelInfo.split?.train?.join(', ')}</dd>
                <dt>Validation</dt><dd>{modelInfo.split?.validation?.join(', ')}</dd>
                <dt>Test (held out)</dt><dd>{modelInfo.split?.test?.join(', ')}</dd>
              </dl>
            ) : (
              <p>{modelInfo?.message || NA}</p>
            )}
          </Card>

          <Card title="Training">
            <dl className="kv">
              <dt>State</dt><dd>{training?.state || NA}</dd>
              <dt>Stage</dt><dd>{training?.stage || '—'}</dd>
              <dt>Progress</dt><dd>{training ? pct(training.progress) : NA}</dd>
              <dt>Last finished</dt><dd>{when(training?.finished_at)}</dd>
              {training?.error ? (<><dt>Error</dt><dd className="error">{training.error}</dd></>) : null}
            </dl>
            {training?.state === 'running' ? (
              <progress max="1" value={training.progress || 0} />
            ) : user?.is_admin ? (
              <div className="row">
                <label className="check">
                  <input type="checkbox" checked={runCv} onChange={(e) => setRunCv(e.target.checked)} />
                  leave-one-participant-out CV
                </label>
                <button onClick={handleTrain}>Train on DEAP</button>
              </div>
            ) : (
              <p className="muted">{user ? 'Only administrators can start training.' : 'Sign in as an administrator to train.'}</p>
            )}
          </Card>
        </div>

        <Card id="metrics" title="Evaluation on held-out participants">
          {test ? (
            <>
              <table className="table">
                <thead>
                  <tr><th></th><th>Accuracy</th><th>Balanced acc.</th><th>Macro F1</th><th>ROC-AUC</th><th>n</th></tr>
                </thead>
                <tbody>
                  {[['Model, per 4 s segment', test.segment], ['Model, per trial (averaged)', test.trial],
                    ['Majority-class reference', reference?.segment]].map(([name, m]) => (
                    <tr key={name}>
                      <td>{name}</td><td>{pct(m?.accuracy)}</td><td>{pct(m?.balanced_accuracy)}</td>
                      <td>{num(m?.f1_macro)}</td><td>{num(m?.roc_auc)}</td><td>{m?.n_samples ?? NA}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
              <p className="muted">
                Leave-one-participant-out CV: {cv ? `${pct(cv.segment_accuracy_mean)} ± ${pct(cv.segment_accuracy_std)} accuracy over ${cv.folds.length} participants` : NA}.
                Chance level for this binary task is about 50%.
              </p>
              <div className="graphs">
                {metrics.graphs.map((src) => <img key={src} src={src} alt={src.split('/').pop()} />)}
              </div>
            </>
          ) : (
            <p>{metrics?.message || NA}</p>
          )}
        </Card>

        <Card id="predict" title="Prediction">
          {!user ? (
            <p className="muted">Sign in to run predictions.</p>
          ) : !modelReady ? (
            <p>{NA}: no trained model.</p>
          ) : (
            <>
              <div className="row">
                <label>
                  Held-out participant
                  <select value={participant} onChange={(e) => setParticipant(e.target.value)}>
                    {(demo?.held_out_participants || []).map((p) => <option key={p} value={p}>{p}</option>)}
                  </select>
                </label>
                <label>
                  Trial (0-39)
                  <input type="number" min="0" max="39" value={trial} onChange={(e) => setTrial(Number(e.target.value))} />
                </label>
                <button disabled={busy || !participant}
                  onClick={() => runPrediction({ deap_reference: { participant_id: participant, trial_index: trial } })}>
                  {busy ? 'Predicting...' : 'Predict DEAP trial'}
                </button>
                <label className="file">
                  or upload EEG JSON
                  <input type="file" accept=".json,application/json" onChange={handleUpload} disabled={busy} />
                </label>
              </div>
              <p className="muted">Upload format: 32 channels in DEAP order × samples at 128 Hz, e.g. {'{"signal": [[...], ...]}'}.</p>
              {uploadError ? <p className="error">{uploadError}</p> : null}
              {predictError ? <p className="error">{predictError}</p> : null}
              {prediction ? (
                <div className="result">
                  <div className="stats-grid">
                    <Stat label="Predicted class" value={prediction.prediction} />
                    <Stat label="Confidence" value={pct(prediction.confidence)} />
                    <Stat label="Segments used" value={`${prediction.segments_used} (${prediction.segments_rejected} rejected)`} />
                    <Stat label="Self-reported label" value={prediction.source_reference?.self_reported_label || NA}
                      hint={prediction.source_reference?.participant_split ? `participant split: ${prediction.source_reference.participant_split}` : null} />
                  </div>
                  <ul className="bars">
                    {Object.entries(prediction.probabilities).map(([label, p]) => (
                      <li key={label}>
                        <span>{label}</span>
                        <span className="bar"><span style={{ width: `${p * 100}%` }} /></span>
                        <span>{pct(p)}</span>
                      </li>
                    ))}
                  </ul>
                  <p className="muted">#{prediction.prediction_id} · {when(prediction.created_at)} · model {prediction.model?.version}</p>
                  <p className="muted small">{prediction.disclaimer}</p>
                </div>
              ) : null}
            </>
          )}
        </Card>

        <Card id="history" title="Prediction history">
          {!user ? (
            <p className="muted">Sign in to see your history.</p>
          ) : !history?.items?.length ? (
            <p>{NA}: no predictions yet.</p>
          ) : (
            <table className="table">
              <thead>
                <tr><th>#</th><th>When</th><th>Source</th><th>Prediction</th><th>Confidence</th><th>Self-reported</th></tr>
              </thead>
              <tbody>
                {history.items.map((p) => (
                  <tr key={p.prediction_id}>
                    <td>{p.prediction_id}</td>
                    <td>{when(p.created_at)}</td>
                    <td>{p.input_source === 'deap_reference'
                      ? `${p.source_reference.participant_id} / trial ${p.source_reference.trial_index}` : 'upload'}</td>
                    <td>{p.prediction}</td>
                    <td>{pct(p.confidence)}</td>
                    <td>{p.source_reference?.self_reported_label || '—'}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </Card>
      </main>
    </div>
  );
}

export default App;
