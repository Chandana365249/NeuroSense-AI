import { verdict } from '../xai';
import { classLabel } from './format';

/** Half-circle gauge of the probability of higher valence (0–100%). With `calibrated` (the server's calibrated
 * valence estimate) it shows the calibrated probability; otherwise the model's raw score, labelled as such. */
export default function ScoreGauge({ score, label, calibrated }) {
  const isCal = typeof calibrated?.probability_high === 'number';
  const p = Math.max(0, Math.min(1, Number(isCal ? calibrated.probability_high : score) || 0));
  const r = 80;
  const angle = Math.PI * (1 - p);
  const x = 100 + r * Math.cos(angle);
  const y = 100 - r * Math.sin(angle);
  const high = label === 'high_valence';
  const uncertain = isCal && verdict(high ? 'higher' : 'lower', calibrated.probability_high) === 'uncertain';
  const caption = isCal ? 'Calibrated probability of higher valence' : 'Model score for higher valence (not calibrated)';
  return (
    <figure className="gauge" aria-label={`${caption}: ${(p * 100).toFixed(1)} percent, estimate ${classLabel(label)}`}>
      <svg viewBox="0 0 200 116" role="img" aria-hidden="true">
        <path d="M 20 100 A 80 80 0 0 1 180 100" className="gauge-track" />
        {p > 0 ? <path d={`M 20 100 A 80 80 0 0 1 ${x.toFixed(2)} ${y.toFixed(2)}`} className={`gauge-fill ${p >= 0.5 ? '' : 'is-low'}`} /> : null}
        <line x1="100" y1="16" x2="100" y2="30" className="gauge-mid" />
      </svg>
      <div className="gauge-value">{(p * 100).toFixed(1)}<span>%</span></div>
      <figcaption>
        {uncertain
          ? <span className="tag" title={`The model leaned toward ${classLabel(label).toLowerCase()}, but its calibrated probability says it is too weak to call`}>Uncertain · model leaned {high ? 'higher' : 'lower'}</span>
          : <span className={`gauge-chip ${high ? '' : 'is-low'}`}>{classLabel(label)}</span>}
        <span className="gauge-caption">{caption}</span>
      </figcaption>
    </figure>
  );
}
