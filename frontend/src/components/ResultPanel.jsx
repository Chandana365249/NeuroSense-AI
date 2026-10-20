import { href } from '../router';
import { classLabel, sourceLabel, when } from './format';
import ProbabilityBars from './ProbabilityBars';
import Stat from './Stat';

/** One prediction, as returned by POST /predict or GET /predictions/{id}. */
export default function ResultPanel({ prediction, showDetailLink = false }) {
  const ref = prediction.source_reference || {};
  const selfValence = ref.self_reported_ratings?.valence;
  const isDeap = prediction.input_source === 'deap_reference';
  return (
    <div className="result">
      <div className="result-headline">
        <span className="result-kicker">Estimated category</span>
        <span className={`result-class ${prediction.prediction === 'low_valence' ? 'is-low' : 'is-high'}`}>
          {classLabel(prediction.prediction)}
        </span>
      </div>

      <h3 className="subhead">Model probability</h3>
      <ProbabilityBars probabilities={prediction.probabilities} />

      <div className="stats-grid">
        <Stat label="Input" value={sourceLabel(prediction)} />
        <Stat label="Segments used" value={prediction.segments_used} hint={`${prediction.segments_rejected} rejected by quality checks`} />
        {isDeap ? (
          <Stat label="Self-reported category" value={classLabel(ref.self_reported_label)}
            hint={typeof selfValence === 'number' ? `Valence rating ${selfValence} on a 1–9 scale` : null} />
        ) : null}
        {isDeap && ref.self_reported_label ? (
          <Stat label="Matches self-report" value={ref.self_reported_label === prediction.prediction ? 'Yes' : 'No'} />
        ) : null}
      </div>

      <p className="meta">
        Prediction #{prediction.prediction_id} · {when(prediction.created_at)} · model {prediction.model?.version || 'unknown'}
        {showDetailLink ? <> · <a href={href(`/results/${prediction.prediction_id}`)}>Open result details</a></> : null}
      </p>
      <p className="meta small">{prediction.disclaimer}</p>
    </div>
  );
}
