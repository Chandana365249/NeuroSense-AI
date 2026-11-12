import { useEffect, useMemo, useState } from 'react';
import { BarElement, CategoryScale, Chart as ChartJS, Filler, Legend, LinearScale, LineElement, PointElement, Tooltip } from 'chart.js';
import { Bar, Line } from 'react-chartjs-2';
import { api } from '../api';
import { BAND, divergingColor, dominantBand, featureLabel, fmtContribution, ratingRows, segmentExtremes, sequentialColor, strongestCell, unitText, verificationText } from '../xai';
import AffectPanel from './AffectPanel';
import Card from './Card';
import LoadingBlock from './LoadingBlock';
import Notice from './Notice';

ChartJS.register(CategoryScale, LinearScale, BarElement, LineElement, PointElement, Filler, Tooltip, Legend);
ChartJS.defaults.animation = false; // static figures: no redraw animation on resize

const INK = '#A9AEB8';
const GRID = 'rgba(244, 241, 234, 0.08)';
const TEAL = '#00B8B0';
const BLUE = '#4D7CFE';
// Fixed categorical order (theta, alpha, beta, gamma, broadband), dark-surface steps of the reference palette.
const BAND_COLORS = { theta: '#3987e5', alpha: '#d95926', beta: '#199e70', gamma: '#c98500', broadband: '#9085e9' };

const axes = (x = {}, y = {}) => ({
  x: { ticks: { color: INK, font: { size: 11 } }, grid: { color: GRID }, ...x },
  y: { ticks: { color: INK, font: { size: 11 } }, grid: { color: GRID }, ...y },
});

function Ratings({ rows }) {
  return (
    <ul className="rating-list">
      {rows.map((r) => (
        <li key={r.name}>
          <span className="rating-name">{r.name}</span>
          {r.missing ? <span className="rating-missing">{r.missing}</span> : (
            <>
              <span className="rating-track" aria-hidden="true"><span className="rating-fill" style={{ width: `${r.share * 100}%` }} /></span>
              <span className="rating-value">{r.value}<span className="muted"> / {r.scale[1]}</span></span>
            </>
          )}
        </li>
      ))}
    </ul>
  );
}

function SegmentScores({ view, onPick }) {
  const s = view.segment_seconds;
  const labels = view.kept_segments.map((i) => `${i * s}–${(i + 1) * s} s`);
  const data = {
    labels,
    datasets: [
      { label: 'Valence score', data: view.segment_probability_high.map((p) => p * 100), borderColor: TEAL, backgroundColor: 'rgba(0,184,176,0.15)',
        fill: true, tension: 0.25, pointRadius: 4, borderWidth: 2 },
      { label: '50% line', data: labels.map(() => 50), borderColor: 'rgba(169,174,184,0.5)', borderDash: [4, 4], pointRadius: 0, borderWidth: 1 },
    ],
  };
  const options = { responsive: true, maintainAspectRatio: false,
    onClick: (_, els) => { if (onPick && els.length) onPick(els[0].index); },
    onHover: (ev, els) => { if (ev.native?.target) ev.native.target.style.cursor = onPick && els.length ? 'pointer' : 'default'; },
    plugins: { legend: { display: false },
      tooltip: { callbacks: { label: (c) => (c.datasetIndex ? '' : `Valence score ${c.parsed.y.toFixed(1)}%`),
        footer: () => (onPick ? 'Click to view this segment’s EEG' : '') } } },
  scales: axes({}, { min: 0, max: 100, ticks: { color: INK, callback: (v) => `${v}%` } }) };
  return <div className="chart-box"><Line data={data} options={options} aria-label="Valence score for each 4-second segment" role="img" /></div>;
}

function EegTraces({ signal, channels, segment, onSegment }) {
  const [own, setOwn] = useState(0);
  const k = segment ?? own;
  const setK = onSegment ?? setOwn;
  const seg = signal.segments[Math.min(k, signal.segments.length - 1)];
  const rowH = 22;
  const W = 1000;
  const left = 46;
  const n = seg.data[0].length;
  const flat = seg.data.flat().map(Math.abs).sort((a, b) => a - b);
  const p95 = flat[Math.floor(flat.length * 0.95)] || 1;
  const scale = (rowH * 0.45) / p95;
  const H = channels.length * rowH + 22;
  const seconds = n / signal.sampling_rate;
  return (
    <div>
      <div className="seg-picker" role="group" aria-label="Choose a segment">
        {signal.segments.map((s, i) => (
          <button key={s.index} type="button" className={`chip ${i === k ? 'is-active' : ''}`} aria-pressed={i === k} onClick={() => setK(i)}>
            Segment {s.index + 1}
          </button>
        ))}
      </div>
      <div className="eeg-box">
        <svg viewBox={`0 0 ${W} ${H}`} preserveAspectRatio="none" role="img" aria-label={`EEG traces of ${channels.length} channels, segment ${seg.index + 1}`}>
          {seg.data.map((row, c) => {
            const y0 = c * rowH + rowH / 2;
            const d = row.map((v, i) => `${i ? 'L' : 'M'}${(left + (i / (n - 1)) * (W - left - 6)).toFixed(1)} ${(y0 - v * scale).toFixed(1)}`).join(' ');
            return (
              <g key={channels[c]}>
                <text x="2" y={y0 + 4} className="eeg-label">{channels[c]}</text>
                <path d={d} className="eeg-line" />
              </g>
            );
          })}
          {[0, 0.25, 0.5, 0.75, 1].map((f) => (
            <text key={f} x={left + f * (W - left - 6)} y={H - 4} className="eeg-label" textAnchor={f === 0 ? 'start' : f === 1 ? 'end' : 'middle'}>
              {(f * seconds).toFixed(1)} s
            </text>
          ))}
        </svg>
      </div>
      <p className="meta small">{signal.unit}; shown at {signal.sampling_rate.toFixed(0)} samples per second; each row scaled the same.</p>
    </div>
  );
}

/** channels x bands grid. mode 'column' colours each band within its own range; 'diverging' centres on zero. */
function Heatmap({ channels, bands, matrix, mode, format, caption }) {
  const cols = bands.map((_, j) => matrix.map((r) => r[j]).filter(Number.isFinite));
  const ranges = cols.map((c) => [Math.min(...c), Math.max(...c)]);
  const maxAbs = Math.max(...matrix.flat().filter(Number.isFinite).map(Math.abs), 0);
  return (
    <figure className="heatmap">
      <div className="heatmap-grid" style={{ gridTemplateColumns: `52px repeat(${bands.length}, minmax(0, 1fr))` }}>
        <span />
        {bands.map((b) => <span key={b} className="heatmap-head">{(BAND[b] || b).split(' ')[0]}</span>)}
        {channels.map((ch, i) => (
          <div key={ch} className="heatmap-row" style={{ display: 'contents' }}>
            <span className="heatmap-ch">{ch}</span>
            {bands.map((b, j) => {
              const v = matrix[i][j];
              const bg = !Number.isFinite(v) ? 'transparent' : mode === 'diverging' ? divergingColor(v, maxAbs) : sequentialColor(v, ...ranges[j]);
              return <span key={b} className="heatmap-cell" style={{ background: bg }} title={`${ch} · ${BAND[b] || b}: ${format(v)}`} />;
            })}
          </div>
        ))}
      </div>
      <figcaption className="meta small">{caption}</figcaption>
    </figure>
  );
}

function SignedBars({ items, kind, label }) {
  const max = Math.max(...items.map((i) => Math.abs(i.value)), 1e-12);
  const data = {
    labels: items.map((i) => i.label),
    datasets: [{ data: items.map((i) => (kind === 'probability' ? i.value * 100 : i.value)), borderRadius: 4,
      backgroundColor: items.map((i) => (i.value >= 0 ? TEAL : BLUE)), barThickness: 14 }],
  };
  const options = { indexAxis: 'y', responsive: true, maintainAspectRatio: false,
    plugins: { legend: { display: false }, tooltip: { callbacks: { label: (c) => fmtContribution(items[c.dataIndex].value, kind) } } },
    scales: axes({ ticks: { color: INK, callback: (v) => (kind === 'probability' ? `${v} pts` : v) }, suggestedMin: -max * (kind === 'probability' ? 100 : 1), suggestedMax: max * (kind === 'probability' ? 100 : 1) },
      { ticks: { color: '#F4F1EA', font: { size: 11 }, autoSkip: false }, grid: { display: false } }) };
  return <div className="chart-box" style={{ height: `${items.length * 26 + 40}px` }}><Bar data={data} options={options} role="img" aria-label={label} /></div>;
}

function SegmentBands({ e }) {
  const v = e.trial_view;
  const s = v.segment_seconds;
  const bands = v.segment_band_contribution.bands;
  const data = {
    labels: v.kept_segments.map((i) => `${i * s}–${(i + 1) * s} s`),
    datasets: bands.map((b, j) => ({ label: BAND[b] || b, data: v.segment_band_contribution.values.map((r) => (e.output_kind === 'probability' ? r[j] * 100 : r[j])),
      backgroundColor: BAND_COLORS[b], borderColor: '#191D24', borderWidth: 1, stack: 'all' })),
  };
  const options = { responsive: true, maintainAspectRatio: false, plugins: { legend: { labels: { color: INK, boxWidth: 12 } } },
    scales: axes({ stacked: true }, { stacked: true, ticks: { color: INK, callback: (x) => (e.output_kind === 'probability' ? `${x} pts` : x) } }) };
  return <div className="chart-box"><Bar data={data} options={options} role="img" aria-label="Contribution of each frequency band in each segment" /></div>;
}

function TimeProfile({ e }) {
  const t = e.by_second;
  const n = t.values.length ? t.values[0].length : 0;
  const mean = Array.from({ length: n }, (_, k) => t.values.reduce((a, r) => a + r[k], 0) / t.values.length);
  const data = { labels: mean.map((_, k) => `${k}–${k + 1} s`),
    datasets: [{ data: mean, backgroundColor: mean.map((v) => (v >= 0 ? TEAL : BLUE)), borderRadius: 4 }] };
  const options = { responsive: true, maintainAspectRatio: false, plugins: { legend: { display: false },
    tooltip: { callbacks: { label: (c) => fmtContribution(c.parsed.y, 'log_odds') } } }, scales: axes() };
  return <div className="chart-box"><Bar data={data} options={options} role="img" aria-label="Contribution of each second within a segment, averaged over segments" /></div>;
}

function SignLegend({ kind }) {
  return (
    <div className="legend-row" aria-hidden="true">
      <span><i className="sw sw-teal" /> toward higher valence</span>
      <span><i className="sw sw-blue" /> toward lower valence</span>
      <span className="muted">Units: {unitText(kind)}</span>
    </div>
  );
}

/** One flashcard: a question with its answer on the front; the interactive chart on the back. */
function Flashcard({ id, icon, title, answer, detail, flipped, expanded, onFlip, onExpand, children }) {
  return (
    <article className={`flashcard ${flipped ? 'is-flipped' : ''} ${expanded ? 'is-expanded' : ''}`} aria-labelledby={`${id}-t`}>
      {flipped ? (
        <div className="flashcard-face flashcard-back" key="back">
          <header className="flashcard-head">
            <h3 id={`${id}-t`}>{title}</h3>
            <div className="flashcard-actions">
              <button type="button" className="btn btn-ghost btn-small" onClick={onExpand} aria-pressed={expanded}>{expanded ? 'Collapse' : 'Expand'}</button>
              <button type="button" className="btn btn-secondary btn-small" onClick={onFlip}>Summary</button>
            </div>
          </header>
          <div className="flashcard-body">{children}</div>
        </div>
      ) : (
        <button type="button" className="flashcard-face flashcard-front" key="front" onClick={onFlip} aria-label={`${title}. ${answer}. Show the chart.`}>
          <span className="flashcard-icon" aria-hidden="true">{icon}</span>
          <span id={`${id}-t`} className="flashcard-q">{title}</span>
          <span className="flashcard-a">{answer}</span>
          {detail ? <span className="flashcard-d">{detail}</span> : null}
          <span className="flashcard-hint" aria-hidden="true">Show the chart ↻</span>
        </button>
      )}
    </article>
  );
}

const firstWord = (band) => (BAND[band] || band).split(' ')[0];

/** Every graph of a result as a deck of flashcards. Click a card to flip it; expand it to full width. */
function Deck({ e }) {
  const v = e.trial_view;
  const kind = e.output_kind;
  const fmt = (x) => fmtContribution(x, kind);
  const [flipped, setFlipped] = useState(() => new Set());
  const [expanded, setExpanded] = useState(() => new Set());
  const [segment, setSegment] = useState(0);
  const toggler = (setSet) => (id) => setSet((old) => {
    const next = new Set(old);
    if (next.has(id)) next.delete(id); else next.add(id);
    return next;
  });
  const flip = toggler(setFlipped);
  const expand = toggler(setExpanded);

  const ext = segmentExtremes(v);
  const power = strongestCell(e.channels, e.bands, v.relative_band_power);
  const topUp = e.top_positive?.[0];
  const topDown = e.top_negative?.[0];
  const band = e.by_band?.[0];
  const channel = e.by_channel?.[0];
  const cell = e.contribution_map ? strongestCell(e.channels, e.bands, e.contribution_map, true) : null;
  const domBand = dominantBand(v.segment_band_contribution);
  const top = [...(e.top_positive || []).slice(0, 8), ...(e.top_negative || []).slice(0, 8).reverse()]
    .filter((f) => f.contribution !== 0).map((f) => ({ label: featureLabel(f), value: f.contribution }));
  const bandRows = (e.by_band || []).map((g) => ({ label: g.label, value: g.contribution }));
  const chanRows = (e.by_channel || []).slice(0, 12).map((g) => ({ label: g.group, value: g.contribution }));
  const towardHigh = e.trial_output >= e.base_value;
  const direction = (x) => `Pushed toward ${x >= 0 ? 'higher' : 'lower'} valence`;
  const openSegment = (i) => {
    if (!v.signal) return;
    setSegment(i);
    setFlipped((old) => new Set(old).add('fc-eeg'));
  };

  const cards = [
    {
      id: 'fc-time', icon: '∿', title: 'How did the score change over the recording?',
      answer: ext ? `Highest at ${ext.high.label} (${(ext.high.p * 100).toFixed(0)}%)` : 'No segments',
      detail: ext ? `Lowest at ${ext.low.label} (${(ext.low.p * 100).toFixed(0)}%) · ${v.kept_segments.length} segments` : null,
      body: (
        <>
          <SegmentScores view={v} onPick={v.signal ? openSegment : undefined} />
          <p className="meta small">Model output for each {v.segment_seconds}-second segment; the result is their average.{v.signal ? ' Click a point to open that segment’s EEG.' : ''}</p>
        </>
      ),
    },
    // No raw signal when the server holds only the saved features (FACED online): the card is left out.
    v.signal ? {
      id: 'fc-eeg', icon: '≋', title: 'What does the EEG look like?',
      answer: `${e.channels.length} channels · ${v.kept_segments.length} clean segments`,
      detail: 'Pick any segment to see every channel',
      body: <EegTraces signal={v.signal} channels={e.channels} segment={segment} onSegment={setSegment} />,
    } : null,
    {
      id: 'fc-power', icon: '▦', title: 'Where is the brain activity strongest?',
      answer: power ? `${firstWord(power.band)} at ${power.channel}` : '—',
      detail: power ? `${(power.value * 100).toFixed(0)}% of that channel’s 4–45 Hz power` : null,
      body: (
        <Heatmap channels={e.channels} bands={e.bands} matrix={v.relative_band_power} mode="column"
          format={(x) => `${(x * 100).toFixed(1)}% of 4–45 Hz power`}
          caption="Relative band power, averaged over segments; colour scaled within each band. Hover a cell for its value." />
      ),
    },
    top.length ? {
      id: 'fc-features', icon: '★', title: 'Which EEG features drove the estimate?',
      answer: topUp ? `${featureLabel(topUp)} (${fmt(topUp.contribution)})` : '—',
      detail: topDown ? `Strongest the other way: ${featureLabel(topDown)} (${fmt(topDown.contribution)})` : null,
      body: <><SignLegend kind={kind} /><SignedBars items={top} kind={kind} label="Features that pushed the estimate up or down" /></>,
    } : null,
    bandRows.length ? {
      id: 'fc-bands', icon: '≡', title: 'Which frequency band mattered most?',
      answer: band ? `${band.label.split(' (')[0]} (${fmt(band.contribution)})` : '—',
      detail: band ? direction(band.contribution) : null,
      body: <><SignLegend kind={kind} /><SignedBars items={bandRows} kind={kind} label="Contribution of each frequency band" /></>,
    } : null,
    {
      id: 'fc-channels', icon: '◉', title: 'Which electrodes mattered most?',
      answer: channel ? `${channel.group} (${fmt(channel.contribution)})` : '—',
      detail: channel ? direction(channel.contribution) : null,
      body: <><SignLegend kind={kind} /><SignedBars items={chanRows} kind={kind} label="Contribution of each EEG channel" /></>,
    },
    cell ? {
      id: 'fc-map', icon: '▤', title: 'Which electrode and band together?',
      answer: `${cell.channel} · ${firstWord(cell.band)}`,
      detail: `${fmt(cell.value)} toward ${cell.value >= 0 ? 'higher' : 'lower'} valence`,
      body: (
        <Heatmap channels={e.channels} bands={e.bands} matrix={e.contribution_map} mode="diverging" format={fmt}
          caption="Log band power features. Teal pushed toward higher valence, blue toward lower. Hover a cell for its value." />
      ),
    } : null,
    v.segment_band_contribution ? {
      id: 'fc-bandtime', icon: '▥', title: 'How did each band contribute over time?',
      answer: domBand ? `${firstWord(domBand)} contributed most` : '—',
      detail: e.by_pair?.length ? `Strongest asymmetry: ${e.by_pair[0].group} (${fmt(e.by_pair[0].contribution)})` : null,
      body: <SegmentBands e={e} />,
    } : null,
    e.by_second ? {
      id: 'fc-seconds', icon: '⏱', title: 'Which moments within a segment mattered?',
      answer: 'Contribution per second', detail: 'This network reads the raw signal, not frequency bands',
      body: <TimeProfile e={e} />,
    } : null,
    {
      id: 'fc-verify', icon: '✓', title: 'Can this explanation be trusted?',
      answer: 'Verified against the saved result',
      detail: `The contributions add up to the model output (error ${e.verification.max_segment_reconstruction_error.toExponential(0)})`,
      body: (
        <>
          <p className="xai-lead">
            The model estimated <strong>{e.predicted_label === 'high_valence' ? 'higher valence' : 'lower valence'}</strong>; overall, this
            recording&apos;s features moved it {towardHigh ? 'toward higher' : 'toward lower'} valence than its average.
          </p>
          <p className="xai-check"><span className="tag tag-teal">Verified</span> {verificationText(e)}</p>
          <p className="small">{e.meaning}</p>
          <ul className="limitations">{e.caveats.map((c) => <li key={c}>{c}</li>)}</ul>
          <p className="meta small">{e.dataset} · model {e.model_version} · {e.method}</p>
        </>
      ),
    },
  ].filter(Boolean);

  const allFlipped = cards.every((c) => flipped.has(c.id));
  return (
    <section className="deck-section" aria-labelledby="deck-title">
      <div className="deck-head">
        <div>
          <h2 id="deck-title">Explore this result</h2>
          <p className="muted small">Each card answers one question. Click a card to see its interactive chart, and hover the chart for exact values.</p>
        </div>
        <button type="button" className="btn btn-secondary btn-small"
          onClick={() => setFlipped(allFlipped ? new Set() : new Set(cards.map((c) => c.id)))}>
          {allFlipped ? 'Show all summaries' : 'Show all charts'}
        </button>
      </div>
      <div className="deck">
        {cards.map((c) => (
          <Flashcard key={c.id} id={c.id} icon={c.icon} title={c.title} answer={c.answer} detail={c.detail}
            flipped={flipped.has(c.id)} expanded={expanded.has(c.id)} onFlip={() => flip(c.id)} onExpand={() => expand(c.id)}>
            {c.body}
          </Flashcard>
        ))}
      </div>
    </section>
  );
}

/** Estimates and an interactive deck of every graph for one saved result. Synthetic results are never passed in. */
export default function TrialInsights({ auth, predictionId, inputSource, eeg, onAffect }) {
  const [state, setState] = useState({ status: 'loading' });
  useEffect(() => {
    let cancelled = false;
    setState({ status: 'loading' });
    const call = inputSource === 'upload' && eeg ? api.explainUpload(auth.token, predictionId, eeg) : api.explanation(auth.token, predictionId);
    call.then((data) => { if (!cancelled) setState({ status: 'ok', data }); })
      .catch((err) => {
        if (err.status === 401) { auth.expire(); return; }
        if (!cancelled) setState({ status: 'error', message: err.message });
      });
    return () => { cancelled = true; };
  }, [predictionId, auth.token]); // eslint-disable-line react-hooks/exhaustive-deps

  const e = state.data;
  useEffect(() => {
    if (onAffect) onAffect(e?.available ? e.affect?.dimensions.find((d) => d.key === 'valence') || null : null);
  }, [e]); // eslint-disable-line react-hooks/exhaustive-deps
  const rows = useMemo(() => (e?.available ? ratingRows(e.ratings, e.dataset) : []), [e]);
  const title = 'Why did the model make this prediction?';
  if (state.status === 'loading') {
    return <Card title={title} titleId="xai-title"><LoadingBlock label="Computing the estimates and the explanation from the model and this recording…" lines={3} /></Card>;
  }
  if (state.status === 'error') return <Card title={title} titleId="xai-title"><Notice tone="error" title="The explanation could not be loaded">{state.message}</Notice></Card>;
  if (!e.available) {
    return <Card title={title} titleId="xai-title"><Notice tone="info" title="No explanation for this result" announce={false}>{e.reason}</Notice></Card>;
  }
  return (
    <div className="insights">
      {e.affect ? (
        <Card title="Estimated emotion for this trial" titleId="affect-title" className="affect-card">
          <AffectPanel auth={auth} predictionId={predictionId} affect={e.affect} />
        </Card>
      ) : (
        <Card title="Participant’s ratings for this trial" titleId="ratings-title">
          {e.ratings.length ? <Ratings rows={rows} /> : <p className="muted">No ratings exist for an uploaded recording.</p>}
          <p className="meta small">The participant&apos;s own ratings from the {e.dataset} dataset.</p>
        </Card>
      )}
      <Deck e={e} />
    </div>
  );
}
