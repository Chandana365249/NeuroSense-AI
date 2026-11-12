import { useEffect, useRef } from 'react';
import { DATASETS, datasetPath, readStoredDataset, resolveDataset, storeDataset } from '../datasets';
import { navigate } from '../router';

/** The selected dataset of a page. The choice lives in the URL, so it survives reloads and can be linked. */
export function useDataset(route) {
  const dataset = resolveDataset(route.query.get('dataset'), readStoredDataset());
  useEffect(() => { storeDataset(dataset); }, [dataset]);
  const choose = (id) => {
    storeDataset(id);
    navigate(datasetPath(route.path, id), { replace: true }); // drops any participant carried by the old choice
  };
  return [dataset, choose];
}

export default function DatasetTabs({ value, onChange }) {
  const refs = useRef({});
  const onKeyDown = (event) => {
    const step = { ArrowRight: 1, ArrowLeft: -1 }[event.key];
    if (!step) return;
    event.preventDefault();
    const i = DATASETS.findIndex((d) => d.id === value);
    const next = DATASETS[(i + step + DATASETS.length) % DATASETS.length].id;
    onChange(next);
    refs.current[next]?.focus();
  };
  return (
    <div className="tabs" role="tablist" aria-label="Dataset" onKeyDown={onKeyDown}>
      {DATASETS.map((d) => (
        <button key={d.id} type="button" role="tab" id={`tab-${d.id}`} aria-selected={value === d.id}
          aria-controls={`panel-${d.id}`} tabIndex={value === d.id ? 0 : -1}
          ref={(el) => { refs.current[d.id] = el; }}
          className={`tab${value === d.id ? ' active' : ''}`} onClick={() => onChange(d.id)}>
          {d.label}
        </button>
      ))}
    </div>
  );
}

/** Dataset, pipeline and model of the current page, stated before anything is run. */
export function PipelineStrip({ dataset, pipeline, model, note }) {
  return (
    <div className="pipeline-strip">
      <dl aria-label="Dataset, pipeline and model in use">
        <div><dt>Dataset</dt><dd>{dataset}</dd></div>
        <div><dt>Pipeline</dt><dd>{pipeline}</dd></div>
        <div><dt>Model</dt><dd>{model}</dd></div>
      </dl>
      {note ? <p className="muted small">{note}</p> : null}
    </div>
  );
}
