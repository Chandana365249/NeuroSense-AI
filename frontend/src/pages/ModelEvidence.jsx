import { useCallback, useEffect, useState } from 'react';
import { api } from '../api';
import Breakable from '../components/Breakable';
import Card from '../components/Card';
import LoadingBlock from '../components/LoadingBlock';
import Notice from '../components/Notice';
import PageHeader from '../components/PageHeader';
import { algorithmLabel, classLabel, NA, num, pct, when } from '../components/format';
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
  if (!info?.available) return <Notice tone="warning" title="No trained model">{info?.message || NA}</Notice>;
  const trained = [...(info.split?.train || []), ...(info.split?.validation || [])];
  return (
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
      <dt>Participants used for training</dt><dd>{trained.join(', ') || NA}</dd>
      <dt>Participants not used for training</dt><dd>{(info.split?.test || []).join(', ') || NA}</dd>
    </dl>
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
  const testIds = m.split?.test || [];
  const cv = m.cross_validation;
  const rows = [
    ['Model, per 4 s segment', m.test?.segment],
    ['Model, per trial', m.test?.trial],
    ['Always predicting the most common category (reference)', m.test_reference_majority?.segment],
  ];
  return (
    <>
      <Notice tone="info" title={`Pre-protocol results, ${testIds.length} participants`}>
        These come from one train/test split made before the current evaluation protocol. The {testIds.length} test
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

      <Card title="Historical evaluation of the active model" titleId="historical-evaluation">
        <HistoricalEvaluation />
      </Card>
    </>
  );
}
