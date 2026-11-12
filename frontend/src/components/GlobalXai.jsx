import { BarElement, CategoryScale, Chart as ChartJS, LinearScale, Tooltip } from 'chart.js';
import { Bar } from 'react-chartjs-2';
import { api } from '../api';
import { BAND, featureLabel, sequentialColor, unitText } from '../xai';
import { useLoad } from './FacedExplorer';
import LoadingBlock from './LoadingBlock';
import Notice from './Notice';

ChartJS.register(CategoryScale, LinearScale, BarElement, Tooltip);
ChartJS.defaults.animation = false; // static figures: no redraw animation on resize

const INK = '#A9AEB8';
const GRID = 'rgba(244, 241, 234, 0.08)';
const scaleOf = (kind) => (kind === 'probability' ? 100 : 1);
const unitOf = (kind) => (kind === 'probability' ? ' pts' : '');

function Bars({ rows, kind, label }) {
  const k = scaleOf(kind);
  const data = { labels: rows.map((r) => r.label), datasets: [{ data: rows.map((r) => r.value * k), backgroundColor: '#00B8B0', borderRadius: 4, barThickness: 14 }] };
  const options = { indexAxis: 'y', responsive: true, maintainAspectRatio: false,
    plugins: { legend: { display: false }, tooltip: { callbacks: { label: (c) => `${c.parsed.x.toFixed(kind === 'probability' ? 2 : 3)}${unitOf(kind)}${rows[c.dataIndex].extra || ''}` } } },
    scales: { x: { beginAtZero: true, ticks: { color: INK }, grid: { color: GRID } }, y: { ticks: { color: '#F4F1EA', font: { size: 11 }, autoSkip: false }, grid: { display: false } } } };
  return <div className="chart-box" style={{ height: `${rows.length * 26 + 40}px` }}><Bar data={data} options={options} role="img" aria-label={label} /></div>;
}

/** Model-level explanation: mean absolute attribution over the model's own training data, read from the server. */
export default function GlobalXai({ auth, dataset }) {
  const [state, reload] = useLoad(() => api.globalExplanation(auth.token, dataset), [auth.token, dataset], auth);
  if (state.status === 'loading') return <LoadingBlock label="Loading the model explanation…" lines={2} />;
  if (state.status === 'error') {
    return <Notice tone="error" title="The explanation could not be loaded" action={<button type="button" className="btn btn-secondary btn-small" onClick={reload}>Retry</button>}>{state.error?.message}</Notice>;
  }
  const g = state.data;
  if (!g.available) return <Notice tone="info" title="Not available yet" announce={false}>{g.reason}</Notice>;
  const kind = g.output_kind;
  const k = scaleOf(kind);
  const spread = (r) => ` (participants ${(r.min_participant * k).toFixed(2)}–${(r.max_participant * k).toFixed(2)})`;
  const cols = g.bands.map((_, j) => g.channel_band_map.map((r) => r[j]));
  return (
    <div className="global-xai">
      <p>
        How strongly each input moved this model&apos;s output, averaged over {g.n_segments.toLocaleString()} segments from {g.n_participants} participants
        of the model&apos;s own training data. Measured in mean absolute {unitText(kind)}.
      </p>
      <div className="insights-grid">
        <section className="panel">
          <h3>Most influential EEG features</h3>
          <Bars rows={g.top_features.slice(0, 15).map((f) => ({ label: featureLabel(f), value: f.mean_abs }))} kind={kind} label="Most influential features" />
        </section>
        <div className="stack">
          <section className="panel">
            <h3>Frequency bands</h3>
            <Bars rows={g.by_band.map((r) => ({ label: r.label, value: r.mean_abs, extra: spread(r) }))} kind={kind} label="Influence of each frequency band" />
          </section>
          <section className="panel">
            <h3>Channels (top 12)</h3>
            <Bars rows={g.by_channel.slice(0, 12).map((r) => ({ label: r.group, value: r.mean_abs, extra: spread(r) }))} kind={kind} label="Influence of each channel" />
          </section>
        </div>
        <section className="panel">
          <h3>Channel × band</h3>
          <div className="heatmap-grid" style={{ gridTemplateColumns: `52px repeat(${g.bands.length}, minmax(0, 1fr))` }}>
            <span />
            {g.bands.map((b) => <span key={b} className="heatmap-head">{BAND[b].split(' ')[0]}</span>)}
            {g.channels.map((ch, i) => (
              <div key={ch} style={{ display: 'contents' }}>
                <span className="heatmap-ch">{ch}</span>
                {g.bands.map((b, j) => (
                  <span key={b} className="heatmap-cell" title={`${ch} · ${BAND[b]}: ${(g.channel_band_map[i][j] * k).toFixed(3)}${unitOf(kind)}`}
                    style={{ background: sequentialColor(g.channel_band_map[i][j], Math.min(...cols[j]), Math.max(...cols[j])) }} />
                ))}
              </div>
            ))}
          </div>
          <p className="meta small">Single-channel band features only; colour scaled within each band.</p>
        </section>
        <section className="panel">
          <h3>Variability across participants</h3>
          <div className="table-wrap">
            <table className="table">
              <thead><tr><th scope="col">Band</th><th scope="col">Mean</th><th scope="col">SD across participants</th><th scope="col">Range</th></tr></thead>
              <tbody>
                {g.by_band.map((r) => (
                  <tr key={r.group}>
                    <th scope="row">{r.label}</th><td>{(r.mean_abs * k).toFixed(2)}</td><td>{(r.sd_across_participants * k).toFixed(2)}</td>
                    <td>{(r.min_participant * k).toFixed(2)}–{(r.max_participant * k).toFixed(2)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </section>
      </div>
      <p className="meta small">
        {g.dataset} · model {g.model_version} · {g.method}. Reference data: {g.reference.description} Influence on the model is not
        evidence about where emotions arise in the brain, and it has no clinical meaning.
      </p>
    </div>
  );
}
