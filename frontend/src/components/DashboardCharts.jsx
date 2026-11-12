// Two compact dashboard charts: the split of estimates and the number of analyses per input type.
import { ArcElement, BarElement, CategoryScale, Chart as ChartJS, Legend, LinearScale, Tooltip } from 'chart.js';
import { Bar, Doughnut } from 'react-chartjs-2';
import { DATASET_LABELS } from '../analysis';

ChartJS.register(ArcElement, BarElement, CategoryScale, LinearScale, Tooltip, Legend);

const INK = '#A9AEB8';
const GRID = 'rgba(244, 241, 234, 0.08)';
const COLORS = { deap: '#00B8B0', faced: '#9085e9', upload: '#c98500', synthetic: '#5b6270' };

export default function DashboardCharts({ summary }) {
  const kinds = ['deap', 'faced', 'upload', 'synthetic'];
  const split = {
    labels: ['Higher valence', 'Lower valence'],
    datasets: [{ data: [summary.higher, summary.lower], backgroundColor: ['#00B8B0', '#4D7CFE'], borderColor: '#191D24', borderWidth: 3 }],
  };
  const byKind = {
    labels: kinds.map((k) => DATASET_LABELS[k]),
    datasets: [{ data: kinds.map((k) => summary.counts[k]), backgroundColor: kinds.map((k) => COLORS[k]), borderRadius: 6, barThickness: 28 }],
  };
  return (
    <div className="dash-charts">
      <figure className="dash-chart">
        <figcaption className="chart-title">Estimates</figcaption>
        <div className="chart-box chart-box-small">
          <Doughnut data={split} role="img" aria-label={`${summary.higher} higher and ${summary.lower} lower valence estimates`}
            options={{ responsive: true, maintainAspectRatio: false, animation: false, cutout: '62%',
              plugins: { legend: { position: 'bottom', labels: { color: INK, boxWidth: 12 } } } }} />
        </div>
        <p className="meta small">Real recordings only; synthetic tests are not estimates.</p>
      </figure>
      <figure className="dash-chart">
        <figcaption className="chart-title">Analyses by input</figcaption>
        <div className="chart-box chart-box-small">
          <Bar data={byKind} role="img" aria-label={kinds.map((k) => `${DATASET_LABELS[k]}: ${summary.counts[k]}`).join(', ')}
            options={{ responsive: true, maintainAspectRatio: false, animation: false, plugins: { legend: { display: false } },
              scales: { x: { ticks: { color: INK }, grid: { display: false } }, y: { beginAtZero: true, ticks: { color: INK, precision: 0 }, grid: { color: GRID } } } }} />
        </div>
      </figure>
    </div>
  );
}
