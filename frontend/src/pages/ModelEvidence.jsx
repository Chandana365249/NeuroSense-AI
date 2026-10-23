import { useCallback, useEffect, useState } from 'react';
import { api } from '../api';
import Breakable from '../components/Breakable';
import Card from '../components/Card';
import GlobalImportance from '../components/GlobalImportance';
import LoadingBlock from '../components/LoadingBlock';
import Notice from '../components/Notice';
import PageHeader from '../components/PageHeader';
import { algorithmLabel, classLabel, NA, num, pct, when } from '../components/format';
import { heldOutSummary, NOT_RECORDED, trainingDataSummary } from '../modelInfo';
import { FINDINGS, INTERPRETATION, SUMMARY_SOURCE } from './researchSummary';

const GRAPH_CAPTIONS = [
  [/_cm_segment\.png$/, 'Confusion matrix, per 4 s segment (historical test split)'],
  [/_cm_trial\.png$/, 'Confusion matrix, per trial (historical test split)'],
  [/_roc\.png$/, 'ROC curve (historical test split)'],
  [/_validation_comparison\.png$/, 'Candidate models on the 2 validation participants (model selection)'],
  [/model_comparison.*\.png$/, 'Model comparison on the historical test split'],
];
const caption = (src) => (GRAPH_CAPTIONS.find(([re]) => re.test(src)) || [null, 'Evaluation figure'])[1];

function ActiveModel({ publicData }) {
  const info = publicData.modelInfo;
  if (publicData.status === 'loading') return <LoadingBlock label="Loading the active model…" />;
  if (publicData.status === 'error') return <Notice tone="error" title="The service is unreachable">Model details are unavailable.</Notice>;
  if (publicData.modelInfoError) return <Notice tone="error" title="Could not load the active model">{publicData.modelInfoError}</Notice>;
  if (!info?.available) return <Notice tone="warning" title="No trained model">{info?.message || NA}</Notice>;
  const prov = info.provenance;
  return (
    <>
      <dl className="kv kv-model">
        <dt>Model</dt><dd>{algorithmLabel(info.algorithm)}</dd>
        <dt>Description</dt><dd>{info.algorithm_description || NA}</dd>
        <dt>Estimates</dt><dd>{(info.class_labels || []).map(classLabel).join(' or ') || NA}</dd>
        <dt>Label rule</dt>
        <dd>
          {typeof info.label?.valence_threshold === 'number'
            ? `Self-reported valence above ${info.label.valence_threshold} on the 1–9 scale is “higher”; otherwise “lower”.`
            : info.label?.description || NA}
        </dd>
        <dt>Input features</dt><dd>{info.feature_schema?.count ?? NA} per 4 s EEG segment</dd>
        <dt>Version</dt><dd className="mono ident"><Breakable text={info.version} /></dd>
        <dt>Trained</dt><dd>{when(info.trained_at)}</dd>
        <dt>Training data</dt><dd>{trainingDataSummary(prov)}</dd>
        {prov?.recorded ? <><dt>Trained on</dt><dd>{prov.fit_participants.join(', ')}</dd></> : null}
        {prov?.selection_participants?.length ? (
          <><dt>Used to choose the algorithm</dt><dd>{prov.selection_participants.join(', ')} (also in the training data)</dd></>
        ) : null}
        <dt>Not used for training</dt><dd>{heldOutSummary(prov) ?? NOT_RECORDED}</dd>
      </dl>
      <p className="provenance-note">{prov?.note || "This model's metadata does not record its training participants."}</p>
    </>
  );
}

function HistoricalEvaluation() {
  const [state, setState] = useState({ status: 'loading' });
  const load = useCallback(async () => {
    setState({ status: 'loading' });
    try {
      setState({ status: 'ready', data: await api.metrics() });
    } catch (err) {
      setState({ status: 'error', error: err.status === 0 ? 'The service is unreachable.' : err.message });
    }
  }, []);
  useEffect(() => { load(); }, [load]);

  if (state.status === 'loading') return <LoadingBlock label="Loading evaluation results…" />;
  if (state.status === 'error') {
    return <Notice tone="error" title="Could not load evaluation results" action={<button type="button" className="btn btn-secondary btn-small" onClick={load}>Retry</button>}>{state.error}</Notice>;
  }
  const m = state.data;
  if (!m.available) return <Notice tone="warning" title="No evaluation results">{m.message || NA}</Notice>;
  if (m.evaluation) return <ProtocolEvaluation evaluation={m.evaluation} protocol={m.protocol} />;
  const testIds = m.split?.test || [];
  const cv = m.cross_validation;
  const rows = [
    ['Model, per 4 s segment', m.test?.segment],
    ['Model, per trial', m.test?.trial],
    ['Always predicting the most common category (reference)', m.test_reference_majority?.segment],
  ];
  return (
    <>
      <p className="badge-line"><span className="tag">Historical</span> <span className="tag">Participant-independent</span> <span className="tag">Pre-protocol</span></p>
      <Notice tone="info" title={`Pre-protocol results, ${testIds.length} participants`}>
        These come from one train/test split made before the current evaluation protocol; the test participants were not
        used for training, so the scores are participant-independent. The {testIds.length} test
        participants ({testIds.join(', ')}) are now used for development, so these numbers are historical and are not
        independent-test evidence. Chance level for this two-category task is 50% balanced accuracy.
      </Notice>
      <div className="table-wrap">
        <table className="table">
          <caption className="visually-hidden">Historical test results of the active model</caption>
          <thead>
            <tr>
              <th scope="col">Evaluation</th><th scope="col">Accuracy</th><th scope="col">Balanced accuracy</th>
              <th scope="col">Macro-F1</th><th scope="col">ROC-AUC</th><th scope="col">n</th>
            </tr>
          </thead>
          <tbody>
            {rows.map(([name, r]) => (
              <tr key={name}>
                <th scope="row">{name}</th>
                <td>{pct(r?.accuracy)}</td><td>{pct(r?.balanced_accuracy)}</td><td>{num(r?.f1_macro)}</td>
                <td>{num(r?.roc_auc)}</td><td>{r?.n_samples ?? NA}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      {cv ? (
        <p className="meta">
          Leave-one-participant-out cross-validation ({cv.folds?.length ?? NA} participants, pre-protocol): segment accuracy
          {' '}{pct(cv.segment_accuracy_mean)} ± {pct(cv.segment_accuracy_std)}, macro-F1 {num(cv.segment_f1_macro_mean)}
          {' '}± {num(cv.segment_f1_macro_std)} (mean ± SD across participants).
        </p>
      ) : null}
      {m.graphs?.length ? (
        <div className="figure-grid">
          {m.graphs.map((src) => (
            <figure key={src} className="figure">
              <img src={src} alt={caption(src)} loading="lazy" />
              <figcaption>{caption(src)}</figcaption>
            </figure>
          ))}
        </div>
      ) : null}
    </>
  );
}

/** Exploratory leave-one-participant-out results of a protocol-trained model (no fixed test split exists). */
function ProtocolEvaluation({ evaluation, protocol }) {
  const a = evaluation.aggregate || {};
  const stat = (x, f = pct) => (x?.mean == null ? NA : `${f(x.mean)} ± ${f(x.sd_sample)}`);
  return (
    <>
      <p className="badge-line"><span className="tag tag-teal">Exploratory</span> <span className="tag">Participant-independent</span> <span className="tag">Protocol v{evaluation.protocol_version}</span></p>
      <Notice tone="info" title={`Exploratory results, Protocol v${evaluation.protocol_version}, ${evaluation.n_participants} participants`}>
        Leave-one-participant-out cross-validation on the development participants: each person was predicted by a model
        trained on everyone else. This is {evaluation.label}; it is not an independent test result. Chance level is 50%
        balanced accuracy.
      </Notice>
      <dl className="kv kv-model">
        <dt>Trial balanced accuracy</dt><dd>{stat(a.trial_balanced_accuracy)} (mean ± SD across participants)</dd>
        <dt>Trial macro-F1</dt><dd>{stat(a.trial_f1_macro, num)}</dd>
        <dt>Within-person AUC</dt><dd>{stat(a.within_person_auc, num)}</dd>
        <dt>Majority reference</dt><dd>{stat(a.majority_trial_balanced_accuracy)} balanced accuracy</dd>
        <dt>Single-label participants</dt><dd>{evaluation.diagnostics?.single_label_participants ?? NA} (predicted one category for ≥95% of their trials)</dd>
      </dl>
      {evaluation.folds?.length ? (
        <div className="table-wrap">
          <table className="table">
            <caption className="visually-hidden">Per-participant exploratory results</caption>
            <thead>
              <tr>
                <th scope="col">Held-out participant</th><th scope="col">Trial balanced accuracy</th><th scope="col">Trial macro-F1</th>
                <th scope="col">Within-person AUC</th><th scope="col">Predicted “higher” share</th><th scope="col">Trials</th>
              </tr>
            </thead>
            <tbody>
              {evaluation.folds.map((f) => (
                <tr key={f.held_out}>
                  <th scope="row">{f.held_out}{f.few_trials_flag ? ' (few trials of one class)' : ''}</th>
                  <td>{pct(f.trial?.balanced_accuracy)}</td><td>{num(f.trial?.f1_macro)}</td><td>{num(f.within_person_auc)}</td>
                  <td>{pct(f.predicted_positive_share_trial)}</td><td>{f.n_trials}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      ) : null}
      {protocol?.limitations ? <p className="meta">{protocol.limitations}</p> : null}
    </>
  );
}

export default function ModelEvidence({ publicData }) {
  return (
    <>
      <PageHeader title="Model & Evidence" subtitle="What the active model is, how it was evaluated, and what our research found." />

      <div className="grid-2">
        <Card title="Active model" titleId="active-model"><ActiveModel publicData={publicData} /></Card>
        <Card title="What the output means" titleId="output-meaning">
          <p>{INTERPRETATION}</p>
          <p className="muted">
            “Model probability” is the model’s probability for each category. Given the evaluation results below, a high
            model probability does not mean the estimate is likely to be correct.
          </p>
        </Card>
      </div>

      <Card title="Research summary" titleId="research-summary">
        <p className="badge-line"><span className="tag tag-teal">Exploratory</span> <span className="tag">12 participants</span> <span className="tag">Protocol v1.1</span></p>
        <p className="muted">{SUMMARY_SOURCE.status}</p>
        <ul className="findings">
          {FINDINGS.map((f) => (
            <li key={f.id} className={`finding finding-${f.tone}`}>
              <h3>{f.title}</h3>
              <p>{f.body}</p>
            </li>
          ))}
        </ul>
        <p className="meta small">
          Static summary, not live experiment data. Source: <span className="mono">{SUMMARY_SOURCE.document}</span>
          {' '}(compiled {SUMMARY_SOURCE.compiled}). Metrics are trial-level balanced accuracy unless stated.
        </p>
      </Card>

      <Card title="Evaluation of the active model" titleId="historical-evaluation">
        <HistoricalEvaluation />
      </Card>

      <Card title="Global feature importance" titleId="global-importance">
        <GlobalImportance />
      </Card>
    </>
  );
}
