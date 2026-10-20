import { classLabel, pct } from './format';

export default function ProbabilityBars({ probabilities }) {
  const entries = Object.entries(probabilities || {});
  if (!entries.length) return null;
  return (
    <ul className="prob-bars" aria-label="Model probability for each category">
      {entries.map(([label, p]) => (
        <li key={label}>
          <span className="prob-label">{classLabel(label)}</span>
          <span className="prob-track" aria-hidden="true">
            <span className={`prob-fill ${label === 'low_valence' ? 'prob-fill-blue' : ''}`} style={{ width: `${Math.max(0, Math.min(1, p)) * 100}%` }} />
          </span>
          <span className="prob-value">{pct(p)}</span>
        </li>
      ))}
    </ul>
  );
}
