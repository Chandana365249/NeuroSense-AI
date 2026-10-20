// Charts of the signed-in user's saved analyses. Loaded lazily so Chart.js is only downloaded on History.
import { BarElement, CategoryScale, Chart as ChartJS, Legend, LinearScale, Tooltip } from 'chart.js';
import { Bar } from 'react-chartjs-2';
import { CLASS_LABELS } from './format';

ChartJS.register(CategoryScale, LinearScale, BarElement, Tooltip, Legend);

const CATEGORIES = ['high_valence', 'low_valence'];
const SOURCES = [['deap_reference', 'DEAP trials'], ['upload', 'Uploaded files']];
const PROB_BINS = [[0.5, 0.6], [0.6, 0.7], [0.7, 0.8], [0.8, 0.9], [0.9, 1.0001]];
const DAY_WINDOW = 30;

function tokens() {
  const css = getComputedStyle(document.documentElement);
  const v = (name, fallback) => css.getPropertyValue(name).trim() || fallback;
  return { teal: v('--teal', '#00B8B0'), blue: v('--blue', '#4D7CFE'), muted: v('--muted', '#A9AEB8'), grid: 'rgba(244, 241, 234, 0.09)' };
}

const reducedMotion = () => window.matchMedia?.('(prefers-reduced-motion: reduce)').matches;

function baseOptions(t, { xTitle, yTitle, stacked = false, legend = true }) {
  const axis = (title) => ({
    stacked,
    title: { display: !!title, text: title, color: t.muted },
    ticks: { color: t.muted, precision: 0 },
    grid: { color: t.grid },
  });
  return {
    responsive: true,
    maintainAspectRatio: false,
    animation: reducedMotion() ? false : { duration: 300 },
    plugins: { legend: { display: legend, labels: { color: t.muted } }, tooltip: { intersect: false, mode: 'index' } },
    scales: { x: { ...axis(xTitle), grid: { display: false } }, y: { ...axis(yTitle), beginAtZero: true } },
  };
}

const dayKey = (d) => `${d.getFullYear()}-${String(d.getMonth() + 1).padStart(2, '0')}-${String(d.getDate()).padStart(2, '0')}`;

export function summarise(items) {
  const byCategory = Object.fromEntries(SOURCES.map(([s]) => [s, { high_valence: 0, low_valence: 0 }]));
  const bins = PROB_BINS.map(() => 0);
  let belowHalf = 0;
  const days = new Map();
  let latest = null;
  for (const p of items) {
    const src = p.input_source === 'deap_reference' ? 'deap_reference' : 'upload';
    if (byCategory[src][p.prediction] !== undefined) byCategory[src][p.prediction] += 1;
    const i = PROB_BINS.findIndex(([lo, hi]) => p.confidence >= lo && p.confidence < hi);
    if (i >= 0) bins[i] += 1; else if (typeof p.confidence === 'number') belowHalf += 1;
    const d = new Date(p.created_at);
    if (!Number.isNaN(d.getTime())) {
      days.set(dayKey(d), (days.get(dayKey(d)) || 0) + 1);
      if (!latest || d > latest) latest = d;
    }
  }
  const dayLabels = [];
  if (latest) {
    for (let k = DAY_WINDOW - 1; k >= 0; k -= 1) {
      const d = new Date(latest.getFullYear(), latest.getMonth(), latest.getDate() - k);
      dayLabels.push(dayKey(d));
    }
  }
  const dayCounts = dayLabels.map((k) => days.get(k) || 0);
  const inWindow = dayCounts.reduce((a, b) => a + b, 0);
  return { byCategory, bins, belowHalf, dayLabels, dayCounts, outsideWindow: items.length - inWindow };
}

function DataTable({ caption, head, rows }) {
  return (
    <details className="chart-data">
      <summary>Show the data as a table</summary>
      <div className="table-wrap">
        <table className="table table-compact">
          <caption className="visually-hidden">{caption}</caption>
          <thead><tr>{head.map((h) => <th key={h} scope="col">{h}</th>)}</tr></thead>
          <tbody>{rows.map((r) => <tr key={r[0]}>{r.map((c, i) => (i === 0 ? <th key={i} scope="row">{c}</th> : <td key={i}>{c}</td>))}</tr>)}</tbody>
        </table>
      </div>
    </details>
  );
}

function ChartFigure({ id, title, description, children, table }) {
  return (
    <figure className="chart-figure" aria-labelledby={`${id}-title`}>
      <figcaption>
        <span id={`${id}-title`} className="chart-title">{title}</span>
        <span className="chart-desc">{description}</span>
      </figcaption>
      <div className="chart-box">{children}</div>
      {table}
    </figure>
  );
}

export default function HistoryCharts({ items }) {
  const t = tokens();
  const s = summarise(items);
  const sourcesPresent = SOURCES.filter(([src]) => CATEGORIES.some((c) => s.byCategory[src][c] > 0));
  const catColors = { high_valence: t.teal, low_valence: t.blue };
  const binLabels = PROB_BINS.map(([lo, hi]) => `${Math.round(lo * 100)}–${Math.min(100, Math.round(hi * 100))}%`);
  const shortDay = (k) => new Date(`${k}T00:00:00`).toLocaleDateString(undefined, { month: 'short', day: 'numeric' });
  const catText = sourcesPresent.map(([src, label]) => `${label}: ${CATEGORIES.map((c) => `${s.byCategory[src][c]} ${CLASS_LABELS[c].toLowerCase()}`).join(', ')}`).join('; ');

  return (
    <div className="charts-grid">
      <ChartFigure id="chart-category" title="Estimated category, by input"
        description="How many of your analyses the model placed in each category."
        table={<DataTable caption="Estimated category by input" head={['Input', ...CATEGORIES.map((c) => CLASS_LABELS[c])]}
          rows={sourcesPresent.map(([src, label]) => [label, ...CATEGORIES.map((c) => s.byCategory[src][c])])} />}>
        <Bar role="img" aria-label={`Bar chart. ${catText}.`}
          data={{
            labels: sourcesPresent.map(([, label]) => label),
            datasets: CATEGORIES.map((c) => ({ label: CLASS_LABELS[c], data: sourcesPresent.map(([src]) => s.byCategory[src][c]), backgroundColor: catColors[c], borderRadius: 4 })),
          }}
          options={baseOptions(t, { yTitle: 'Analyses' })} />
      </ChartFigure>

      <ChartFigure id="chart-probability" title="Model probability of the estimated category"
        description="Model probability is the model's own output, not a measure of accuracy."
        table={<DataTable caption="Model probability distribution" head={['Model probability', 'Analyses']}
          rows={[...binLabels.map((l, i) => [l, s.bins[i]]), ...(s.belowHalf ? [['Below 50%', s.belowHalf]] : [])]} />}>
        <Bar role="img" aria-label={`Histogram. ${binLabels.map((l, i) => `${l}: ${s.bins[i]}`).join(', ')}.`}
          data={{ labels: binLabels, datasets: [{ label: 'Analyses', data: s.bins, backgroundColor: t.teal, borderRadius: 4 }] }}
          options={baseOptions(t, { xTitle: 'Model probability', yTitle: 'Analyses', legend: false })} />
      </ChartFigure>

      <ChartFigure id="chart-days" title={`Analyses per day (${DAY_WINDOW} days to your latest)`}
        description={s.outsideWindow ? `${s.outsideWindow} earlier ${s.outsideWindow === 1 ? 'analysis is' : 'analyses are'} outside this window.` : 'All your analyses fall in this window.'}
        table={<DataTable caption="Analyses per day" head={['Date', 'Analyses']}
          rows={s.dayLabels.map((k, i) => [k, s.dayCounts[i]]).filter((r) => r[1] > 0)} />}>
        <Bar role="img" aria-label={`Bar chart of analyses per day. ${s.dayLabels.map((k, i) => [k, s.dayCounts[i]]).filter((r) => r[1] > 0).map(([k, n]) => `${shortDay(k)}: ${n}`).join(', ')}.`}
          data={{ labels: s.dayLabels.map(shortDay), datasets: [{ label: 'Analyses', data: s.dayCounts, backgroundColor: t.blue, borderRadius: 3 }] }}
          options={baseOptions(t, { yTitle: 'Analyses', legend: false })} />
      </ChartFigure>
    </div>
  );
}
