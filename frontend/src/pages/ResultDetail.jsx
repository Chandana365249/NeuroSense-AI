import { lazy, Suspense, useEffect, useState } from 'react';
import { api } from '../api';
import Card from '../components/Card';
import LoadingBlock from '../components/LoadingBlock';
import Notice from '../components/Notice';
import PageHeader from '../components/PageHeader';
import RequireAuth from '../components/RequireAuth';
import ResultPanel from '../components/ResultPanel';
import { datasetPath } from '../datasets';
import { href } from '../router';
import { explainable } from '../xai';

const TrialInsights = lazy(() => import('../components/TrialInsights'));

function Detail({ auth, id, onLoaded }) {
  const [state, setState] = useState({ status: 'loading' });
  const [calibrated, setCalibrated] = useState(null);

  useEffect(() => {
    let cancelled = false;
    setState({ status: 'loading' });
    api.prediction(auth.token, id)
      .then((data) => { if (!cancelled) { setState({ status: 'ready', data }); onLoaded(data); } })
      .catch((err) => {
        if (cancelled) return;
        if (err.status === 401) auth.expire();
        else setState({ status: 'error', notFound: err.status === 404 || err.status === 422, error: err.message });
      });
    return () => { cancelled = true; };
  }, [auth.token, auth.expire, id]);

  if (state.status === 'loading') return <LoadingBlock label="Loading the result…" />;
  if (state.status === 'error') {
    return state.notFound
      ? <Notice tone="warning" title="Result not found">There is no saved analysis #{id} in your history.</Notice>
      : <Notice tone="error" title="Could not load the result">{state.error}</Notice>;
  }
  return (
    <>
      <Card>
        <ResultPanel prediction={state.data} calibrated={calibrated} />
      </Card>
      {explainable(state.data) ? (
        <Suspense fallback={<LoadingBlock label="Loading charts…" />}>
          <TrialInsights auth={auth} predictionId={state.data.prediction_id} inputSource={state.data.input_source} onAffect={setCalibrated} />
        </Suspense>
      ) : null}
    </>
  );
}

export default function ResultDetail({ auth, id }) {
  const [dataset, setDataset] = useState('');
  return (
    <>
      <PageHeader title={`Result #${id}`} subtitle="One saved analysis.">
        <a className="btn btn-secondary btn-small" href={href('/history')}>Back to history</a>
        <a className="btn btn-primary btn-small" href={href(dataset ? datasetPath('/analyze', dataset) : '/analyze')}>New analysis</a>
      </PageHeader>
      <RequireAuth auth={auth} reason="Sign in to view this result.">
        <Detail auth={auth} id={id} onLoaded={(p) => { if (p.input_source === 'faced_reference') setDataset('faced'); }} />
      </RequireAuth>
    </>
  );
}
