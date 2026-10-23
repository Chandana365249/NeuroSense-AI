import { useCallback, useEffect, useState } from 'react';
import { api } from '../api';
import Breakable from './Breakable';
import LoadingBlock from './LoadingBlock';
import Notice from './Notice';
import { featureLabel, pct } from './format';

function Bars({ items, label, caption }) {
  const max = Math.max(...items.map((it) => it.value)) || 1; // bars are relative to the largest item in this table
  return (
    <div className="table-wrap">
      <table className="table importance-table">
        <caption className="visually-hidden">{caption}</caption>
        <thead>
          <tr><th scope="col">{label}</th><th scope="col">Share of total importance</th></tr>
        </thead>
        <tbody>
          {items.map((it) => (
            <tr key={it.key}>
              <th scope="row">{it.name}{it.sub ? <span className="cell-sub mono ident"><Breakable text={it.sub} /></span> : null}</th>
              <td>
                <div className="importance-cell">
                  <span className="importance-bar" aria-hidden="true"><span style={{ width: `${Math.max(2, (it.value / max) * 100)}%` }} /></span>
                  <span className="importance-value">{pct(it.value)}</span>
                </div>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

export default function GlobalImportance() {
  const [state, setState] = useState({ status: 'loading' });
  const load = useCallback(async () => {
    setState({ status: 'loading' });
    try {
      setState({ status: 'ready', data: await api.explainGlobal(15) });
    } catch (err) {
      setState({ status: 'error', error: err.status === 0 ? 'The service is unreachable.' : 'Feature importance could not be loaded.' });
    }
  }, []);
  useEffect(() => { load(); }, [load]);

  if (state.status === 'loading') return <LoadingBlock label="Loading feature importance…" />;
  if (state.status === 'error') {
    return <Notice tone="error" title="Could not load feature importance" action={<button type="button" className="btn btn-secondary btn-small" onClick={load}>Retry</button>}>{state.error}</Notice>;
  }
  const d = state.data;
  if (!d.available) {
    return (
      <Notice tone="info" title="Global feature importance is not available for the active model">
        {d.reason} Analyses still work; only this model-level summary is unavailable.
      </Notice>
    );
  }
  return (
    <>
      <p>
        Which of the {d.n_features} input features the active model (<span className="mono ident"><Breakable text={d.model_version} /></span>)
        relied on most, across all of its training data.
      </p>
      <Notice tone="warning" title="Model-level summary, not an explanation of a prediction" announce={false}>
        {d.interpretation} An evenly spread model would give each feature {pct(d.uniform_share)}.
      </Notice>
      <div className="grid-2 importance-grid">
        <section aria-labelledby="importance-top">
          <h3 id="importance-top" className="subhead">Top {d.top.length} features</h3>
          <Bars caption={`Top ${d.top.length} features by global importance`} label="Feature"
            items={d.top.map((t) => ({ key: t.name, name: `${t.rank}. ${featureLabel(t)}`, sub: t.name, value: t.importance }))} />
        </section>
        <section aria-labelledby="importance-bands">
          <h3 id="importance-bands" className="subhead">By frequency band</h3>
          <Bars caption="Global importance summed by frequency band" label="Band"
            items={d.by_band.map((b) => ({ key: b.group, name: b.label, value: b.importance }))} />
          <h3 className="subhead">By feature type</h3>
          <Bars caption="Global importance summed by feature type" label="Feature type"
            items={d.by_family.map((b) => ({ key: b.group, name: b.label, value: b.importance }))} />
        </section>
      </div>
      <h3 className="subhead">Limitations</h3>
      <ul className="limitations">
        {d.limitations.map((l) => <li key={l}>{l}</li>)}
      </ul>
      <p className="meta small">Method: {d.method} ({d.n_trees} trees). Computed from the trained model; nothing is estimated per recording.</p>
    </>
  );
}
