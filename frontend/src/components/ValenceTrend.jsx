// Line chart of the valence score of the user's most recent real analyses (synthetic tests left out).
import { CategoryScale, Chart as ChartJS, Filler, Legend, LinearScale, LineElement, PointElement, Tooltip } from 'chart.js';
import { Line } from 'react-chartjs-2';
import { datasetKind, DATASET_LABELS, isSynthetic } from '../analysis';

ChartJS.register(CategoryScale, LinearScale, LineElement, PointElement, Filler, Tooltip, Legend);

const LAST = 30;
const INK = '#A9AEB8';
const GRID = 'rgba(244, 241, 234, 0.08)';
const POINT = { deap: '#00B8B0', faced: '#9085e9', upload: '#c98500' };

export default function ValenceTrend({ items }) {
  const rows = items.filter((p) => !isSynthetic(p) && typeof p.probabilities?.high_valence === 'number')
    .sort((a, b) => String(a.created_at).localeCompare(String(b.created_at))).slice(-LAST);
  if (!rows.length) return <p className="muted">Only synthetic tests so far; real analyses will appear here.</p>;
  const labels = rows.map((p) => `#${p.prediction_id}`);
  const data = {
    labels,
    datasets: [
      { label: 'Valence score', data: rows.map((p) => p.probabilities.high_valence * 100), borderColor: '#00B8B0',
        backgroundColor: 'rgba(0,184,176,0.12)', fill: true, tension: 0.25, borderWidth: 2, pointRadius: 5,
        pointBackgroundColor: rows.map((p) => POINT[datasetKind(p)]), pointBorderColor: '#191D24' },
      { label: '50%', data: labels.map(() => 50), borderColor: 'rgba(169,174,184,0.5)', borderDash: [4, 4], pointRadius: 0, borderWidth: 1 },
    ],
  };
  const options = {
    responsive: true, maintainAspectRatio: false, animation: false,
    plugins: {
      legend: { display: false },
      tooltip: { filter: (c) => c.datasetIndex === 0, callbacks: {
        title: (c) => `Result ${labels[c[0].dataIndex]} · ${DATASET_LABELS[datasetKind(rows[c[0].dataIndex])]}`,
        label: (c) => `Valence score ${c.parsed.y.toFixed(1)}%`,
      } },
    },
    scales: {
      x: { ticks: { color: INK, font: { size: 11 } }, grid: { display: false } },
      y: { min: 0, max: 100, ticks: { color: INK, callback: (v) => `${v}%` }, grid: { color: GRID } },
    },
  };
  return (
    <>
      <div className="chart-box"><Line data={data} options={options} role="img" aria-label={`Valence score of your last ${rows.length} analyses`} /></div>
      <div className="legend-row" aria-hidden="true">
        {Object.entries(POINT).map(([k, c]) => <span key={k}><i className="sw" style={{ background: c }} /> {DATASET_LABELS[k]}</span>)}
        <span className="muted">Each point is one analysis, oldest to newest.</span>
      </div>
    </>
  );
}
