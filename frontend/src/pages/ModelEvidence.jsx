import { lazy, Suspense } from 'react';
import Breakable from '../components/Breakable';
import Card from '../components/Card';
import DatasetTabs, { useDataset } from '../components/DatasetTabs';
import FacedEvidence from '../components/FacedEvidence';
import LoadingBlock from '../components/LoadingBlock';
import Notice from '../components/Notice';
import PageHeader from '../components/PageHeader';
import { algorithmLabel, classLabel, NA, when } from '../components/format';
import { heldOutSummary, NOT_RECORDED, trainingDataSummary } from '../modelInfo';
import { DEAP_LIMITATIONS, INTERPRETATION } from './researchSummary';

const GlobalXai = lazy(() => import('../components/GlobalXai'));

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
        <dt>Not used for training</dt><dd>{heldOutSummary(prov) ?? NOT_RECORDED}</dd>
      </dl>
      <p className="provenance-note">{prov?.note || "This model's metadata does not record its training participants."}</p>
    </>
  );
}

export default function ModelEvidence({ auth, publicData, route }) {
  const [dataset, choose] = useDataset(route);
  return (
    <>
      <PageHeader title="Models" subtitle="What each dataset's model is, and which EEG features drive its estimates." />
      <DatasetTabs value={dataset} onChange={choose} />
      {dataset === 'faced' ? (
        <div role="tabpanel" id="panel-faced" aria-labelledby="tab-faced">
          <FacedEvidence auth={auth} />
          <Card title="Explainable AI" titleId="xai-global-faced" className="xai-card"><Suspense fallback={<LoadingBlock label="Loading charts…" />}><GlobalXai auth={auth} dataset="faced" /></Suspense></Card>
        </div>
      ) : (
        <div role="tabpanel" id="panel-deap" aria-labelledby="tab-deap">
          <div className="grid-2">
            <Card title="Active model" titleId="active-model"><ActiveModel publicData={publicData} /></Card>
            <Card title="What the output means" titleId="output-meaning">
              <p>{INTERPRETATION}</p>
              <p className="muted">
                Each estimate comes with a calibrated probability and an explanation of the EEG features that drove it.
              </p>
            </Card>
          </div>
          <Card title="Explainable AI" titleId="xai-global-deap" className="xai-card">
            <Suspense fallback={<LoadingBlock label="Loading charts…" />}><GlobalXai auth={auth} dataset="deap" /></Suspense>
          </Card>
          <Card title="Limitations" titleId="deap-limitations" className="limits-card">
            <ul className="read-list">{DEAP_LIMITATIONS.map((l) => <li key={l}>{l}</li>)}</ul>
          </Card>
        </div>
      )}
    </>
  );
}
