import Notice from '../components/Notice';
import { datasetPath } from '../datasets';
import { href } from '../router';

const OPTIONS = [
  { title: 'DEAP recording', text: 'Analyse a trial from the DEAP research recordings.', to: datasetPath('/analyze', 'deap'), tag: 'Dataset' },
  { title: 'FACED recording', text: 'Analyse a film clip from the FACED research recordings.', to: datasetPath('/analyze', 'faced'), tag: 'Dataset' },
  { title: 'Your EEG file', text: 'Upload a JSON recording: 32 channels at 128 samples per second.', to: datasetPath('/analyze', 'deap', { source: 'upload' }), tag: 'Upload' },
];

/** A small, unobtrusive service indicator; details live on System health (administrators). */
function ServiceIndicator({ publicData }) {
  const { status, health } = publicData;
  const [tone, text] = status === 'loading' ? ['neutral', 'Checking service…']
    : status === 'error' ? ['error', 'Service unavailable']
      : health?.status === 'ok' ? ['ok', 'Service available'] : ['warn', 'Service limited'];
  return <span className={`service-dot service-${tone}`} role="status"><span aria-hidden="true" />{text}</span>;
}

export default function Overview({ auth, publicData }) {
  const name = auth.user?.full_name?.trim();
  return (
    <>
      <header className="page-header overview-hero">
        <div>
          <p className="eyebrow">{name ? `Welcome, ${name}` : 'Welcome'}</p>
          <h1 id="page-title" tabIndex={-1}>NeuroSense AI</h1>
          <p className="page-subtitle">Estimate valence, arousal, dominance and liking from EEG, with an explanation for every result.</p>
        </div>
        <ServiceIndicator publicData={publicData} />
      </header>

      {publicData.status === 'error' ? (
        <Notice tone="error" title="The service is unreachable" action={<button type="button" className="btn btn-secondary btn-small" onClick={publicData.reload}>Retry</button>}>
          Analyses are unavailable until the connection is restored.
        </Notice>
      ) : null}

      <section className="card start-panel" aria-labelledby="start-analysis">
        <div className="start-head">
          <div>
            <h2 id="start-analysis">Start an analysis</h2>
            <p className="muted">Choose a recording, run the analysis, and read the result with its explanation.</p>
          </div>
          <a className="btn btn-primary btn-large" href={href('/analyze')}>Analyze EEG</a>
        </div>
        <ul className="start-options">
          {OPTIONS.map((o) => (
            <li key={o.title}>
              <a className="start-option" href={href(o.to)}>
                <span className="tag">{o.tag}</span>
                <span className="start-option-title">{o.title}</span>
                <span className="start-option-text">{o.text}</span>
                <span className="start-option-go" aria-hidden="true">→</span>
              </a>
            </li>
          ))}
        </ul>
        <p className="meta small">Research prototype: estimates self-reported ratings and is not a diagnostic tool.</p>
      </section>
    </>
  );
}
