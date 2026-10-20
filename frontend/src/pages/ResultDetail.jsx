import { useEffect, useState } from 'react';
import { api } from '../api';
import Card from '../components/Card';
import LoadingBlock from '../components/LoadingBlock';
import Notice from '../components/Notice';
import PageHeader from '../components/PageHeader';
import RequireAuth from '../components/RequireAuth';
import ResultPanel from '../components/ResultPanel';
import { RELIABILITY_NOTE } from '../components/format';
import { href } from '../router';

function Detail({ auth, id }) {
  const [state, setState] = useState({ status: 'loading' });

  useEffect(() => {
    let cancelled = false;
    setState({ status: 'loading' });
    api.prediction(auth.token, id)
      .then((data) => { if (!cancelled) setState({ status: 'ready', data }); })
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
    <Card>
      <ResultPanel prediction={state.data} />
      {state.data.input_source === 'upload' ? (
        <p className="meta small">The uploaded signal itself is not stored, so its waveform preview is only shown at upload time.</p>
      ) : null}
    </Card>
  );
}

export default function ResultDetail({ auth, id }) {
  return (
    <>
      <PageHeader title={`Result #${id}`} subtitle="One saved analysis.">
        <a className="btn btn-secondary btn-small" href={href('/history')}>Back to history</a>
        <a className="btn btn-primary btn-small" href={href('/analyze')}>New analysis</a>
      </PageHeader>
      <Notice tone="warning" title="Reliability">{RELIABILITY_NOTE}</Notice>
      <RequireAuth auth={auth} reason="Sign in to view this result.">
        <Detail auth={auth} id={id} />
      </RequireAuth>
    </>
  );
}
