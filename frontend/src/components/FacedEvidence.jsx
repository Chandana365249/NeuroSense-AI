import { api } from '../api';
import Card from './Card';
import { errorText, useLoad } from './FacedExplorer';
import LoadingBlock from './LoadingBlock';
import Notice from './Notice';
import StatusPill from './StatusPill';

const MODEL_NAMES = {
  'lr_C0.01': 'Logistic regression, C = 0.01', 'lr_C0.1': 'Logistic regression, C = 0.1', lr_C1: 'Logistic regression, C = 1',
  'svm_C0.1': 'SVM, C = 0.1', svm_C1: 'SVM, C = 1', rf: 'Random Forest', xgb: 'XGBoost', cnn_eegnet: 'EEGNet CNN',
};
const modelName = (id) => MODEL_NAMES[id] || id;

/** The FACED research model: what it is and what its output means. */
export default function FacedEvidence({ auth }) {
  const token = auth?.token;
  const [statusState, reload] = useLoad(() => api.facedStatus(token), [token], auth);

  if (statusState.status === 'loading') return <LoadingBlock label="Loading the FACED model…" />;
  if (statusState.status === 'error') {
    return <Notice tone="error" title="Could not load the FACED model" action={<button type="button" className="btn btn-secondary btn-small" onClick={reload}>Retry</button>}>{errorText(statusState.error)}</Notice>;
  }
  const model = statusState.data.model;
  return (
    <>
      <div className="grid-2">
        <Card title="FACED research model" titleId="faced-model">
          {model?.available ? (
            <>
              <div className="dataset-status"><StatusPill tone="warn">Experimental research model</StatusPill></div>
              <dl className="kv kv-model">
                <dt>Model</dt><dd>{modelName(model.algorithm?.name || model.algorithm)}</dd>
                <dt>Version</dt><dd className="mono ident">{model.name}</dd>
                <dt>Estimates</dt><dd>Higher or lower self-reported valence, arousal and liking</dd>
                <dt>Label rule</dt><dd>{statusState.data.label_rule}</dd>
                <dt>Pipeline</dt><dd>{model.pipeline}</dd>
                <dt>Trained on</dt><dd>{model.trained_on.participants} participants, {model.trained_on.videos.length} clips, {model.trained_on.trials} trials</dd>
              </dl>
              <p className="provenance-note">{model.dataset_compatibility}</p>
            </>
          ) : <Notice tone="warning" title="No usable FACED model">{model?.message || 'The FACED model is not available.'}</Notice>}
        </Card>
        <Card title="What the output means" titleId="output-meaning">
          <p>
            The FACED model estimates a binary category of the person&apos;s own ratings. It does not detect a
            person&apos;s exact emotional experience and is not a diagnostic tool.
          </p>
          <p className="muted">
            This model is separate from the DEAP models and cannot process DEAP recordings. DEAP and FACED estimates come
            from different recordings and protocols, so they are not interchangeable.
          </p>
        </Card>
      </div>
      {model?.limitations?.length ? (
        <Card title="Limitations" titleId="faced-limitations" className="limits-card">
          <ul className="read-list">{model.limitations.map((l) => <li key={l}>{l}</li>)}</ul>
        </Card>
      ) : null}
    </>
  );
}
