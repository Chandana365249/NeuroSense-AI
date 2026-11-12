import { useState } from 'react';
import { api } from '../api';
import { affectRows, pct, verdict } from '../xai';

function ProbBar({ p, label }) {
  const v = Math.max(0, Math.min(1, p));
  return (
    <span className="prob-track" role="img" aria-label={`${label}: ${pct(v)} calibrated probability of higher`}>
      <span className={`prob-fill ${v >= 0.5 ? '' : 'is-low'}`} style={{ width: `${v * 100}%` }} />
      <span className="prob-mid" />
    </span>
  );
}

/** The model's decision; null when its score has no usable direction (the probability is then the base rate). */
function Decision({ value, p }) {
  const v = verdict(value, p);
  if (v === 'uncertain') return <span className="tag" title={`The model leaned ${value}, but calibrated against real ratings it is too weak to call`}>Uncertain</span>;
  if (!value) return <span className="tag" title="The score carries no usable direction, so the probability shown is how often such trials were rated higher">No usable signal</span>;
  return <span className={`gauge-chip ${value === 'higher' ? '' : 'is-low'}`}>{value === 'higher' ? 'Higher' : 'Lower'}</span>;
}

function Row({ r, personal }) {
  if (r.missing) {
    return (
      <li className="affect-row is-missing">
        <span className="affect-name">{r.label}</span>
        <span className="muted">{r.missing}</span>
      </li>
    );
  }
  const weak = !r.evaluation?.better_than_chance;
  return (
    <li className={`affect-row ${weak ? 'is-chance' : ''}`}>
      <div className="affect-head">
        <span className="affect-name">{r.label}</span>
        <Decision value={r.predicted} p={r.probability_high} />
        {weak ? <span className="tag tag-warn" title="This estimate is not reliable for people the model has not seen">Low reliability</span> : null}
        <span className="affect-value">{pct(r.probability_high)}<span className="muted"> likely higher</span></span>
      </div>
      <ProbBar p={r.probability_high} label={r.label} />
      <div className="affect-meta">
        {r.own_rating !== null && r.own_rating !== undefined ? (
          <span>
            Participant rated <strong>{r.own_rating}</strong> ({r.own_label})
            {r.agrees === true ? <span className="tag tag-teal">matches</span> : null}
            {r.agrees === false ? <span className="tag">differs</span> : null}
          </span>
        ) : null}
      </div>
      {personal ? (
        personal.available ? (
          <div className="affect-personal">
            <span className="affect-sub">Personalised</span>
            <Decision value={personal.predicted} p={personal.probability_high} />
            <span>{pct(personal.probability_high)} likely higher</span>
            {personal.evaluation?.better_than_chance ? null : <span className="tag tag-warn">Low reliability</span>}
            <span className="muted">trained on {personal.training_trials} of this person&apos;s other rated trials</span>
          </div>
        ) : <div className="affect-personal muted">Personalised: {personal.reason}</div>
      ) : null}
    </li>
  );
}

/** Calibrated estimates of all four dimensions for one trial, with the participant's own ratings and an optional
 * personalised estimate trained only on the same person's other rated trials. */
export default function AffectPanel({ auth, predictionId, affect }) {
  const [personal, setPersonal] = useState({ status: 'idle' });
  const rows = affectRows(affect);
  const byKey = personal.status === 'ok' && personal.data.available
    ? Object.fromEntries(personal.data.dimensions.map((d) => [d.key, d])) : {};
  const run = () => {
    setPersonal({ status: 'loading' });
    api.personalEstimate(auth.token, predictionId)
      .then((data) => setPersonal({ status: 'ok', data }))
      .catch((err) => {
        if (err.status === 401) { auth.expire(); return; }
        setPersonal({ status: 'error', message: err.message });
      });
  };
  return (
    <div className="affect">
      <ul className="affect-list">
        {rows.map((r) => <Row key={r.key} r={r} personal={personal.status === 'ok' ? (byKey[r.key] || (personal.data.available ? null : { available: false, reason: personal.data.reason })) : null} />)}
      </ul>
      {affect.personal_available ? (
        <div className="affect-actions">
          <button type="button" className="btn btn-secondary btn-small" onClick={run} disabled={personal.status === 'loading'}>
            {personal.status === 'loading' ? 'Training personal models…' : personal.status === 'ok' ? 'Personalised' : 'Personalise to this participant'}
          </button>
          <span className="meta small">Trains one model per dimension on this person&apos;s other rated trials only, never on this trial.</span>
          {personal.status === 'error' ? <span className="meta small">Could not personalise: {personal.message}</span> : null}
        </div>
      ) : null}
      <p className="meta small">
        Probabilities are calibrated, so 60% means trials like this one were rated higher about 6 times in 10. Estimates of
        self-reported ratings, not a diagnosis.
      </p>
    </div>
  );
}
