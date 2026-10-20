import { useEffect, useId, useMemo, useRef, useState } from 'react';
import { DEAP_CHANNELS } from './channels';

const GROUP_SIZE = 8;
const WINDOW_SECONDS = 10;
const BASELINE_SECONDS = 3; // the backend drops this much when the file says it includes a baseline

/** Shape check of the uploaded [channels][samples] array, before anything is drawn. */
export function inspectSignal(signal) {
  const lengths = signal.map((ch) => (Array.isArray(ch) ? ch.length : -1));
  const min = Math.min(...lengths);
  const max = Math.max(...lengths);
  let nonFinite = 0;
  if (min > 0) for (const ch of signal) for (const v of ch) if (typeof v !== 'number' || !Number.isFinite(v)) nonFinite += 1;
  return { channels: signal.length, samples: max, rectangular: min === max && min > 0, minLength: min, nonFinite };
}

function channelStats(ch) {
  let lo = Infinity; let hi = -Infinity; let sum = 0; let n = 0;
  for (const v of ch) if (Number.isFinite(v)) { if (v < lo) lo = v; if (v > hi) hi = v; sum += v; n += 1; }
  return n ? { min: lo, max: hi, mean: sum / n } : { min: NaN, max: NaN, mean: NaN };
}

// Median and a robust amplitude (98th percentile of |x - median|) from an evenly spaced sample of the window.
function robust(ch, start, end) {
  const step = Math.max(1, Math.floor((end - start) / 2000));
  const vals = [];
  for (let i = start; i < end; i += step) if (Number.isFinite(ch[i])) vals.push(ch[i]);
  if (!vals.length) return { median: 0, amp: 0 };
  vals.sort((a, b) => a - b);
  const median = vals[Math.floor(vals.length / 2)];
  const dev = vals.map((v) => Math.abs(v - median)).sort((a, b) => a - b);
  return { median, amp: dev[Math.min(dev.length - 1, Math.floor(dev.length * 0.98))] };
}

const fmt = (v) => (Number.isFinite(v) ? (Math.abs(v) >= 100 ? v.toFixed(0) : v.toFixed(1)) : '—');
const niceStep = (seconds) => [0.5, 1, 2, 5, 10, 15, 30, 60].find((s) => seconds / s <= 8) || 120;

export default function EegPreview({ eeg, fileName, rateGiven }) {
  const ids = useId();
  const info = useMemo(() => inspectSignal(eeg.signal), [eeg.signal]);
  const rate = eeg.sampling_rate;
  const names = info.channels === DEAP_CHANNELS.length ? DEAP_CHANNELS : null;
  const label = (i) => (names ? names[i] : `Ch ${i + 1}`);
  const groups = Math.ceil(info.channels / GROUP_SIZE);
  const duration = info.samples / rate;
  const windows = Math.ceil(duration / WINDOW_SECONDS);
  const [group, setGroup] = useState('0');
  // Long recordings open on the first 10 s, which is readable; the whole recording is one selection away.
  const [win, setWin] = useState(() => (info.samples / rate > WINDOW_SECONDS ? '0' : 'full'));
  const canvasRef = useRef(null);
  const boxRef = useRef(null);
  const [scaleNote, setScaleNote] = useState('');

  const shown = useMemo(() => {
    if (group === 'all') return Array.from({ length: info.channels }, (_, i) => i);
    const g = Number(group);
    return Array.from({ length: Math.min(GROUP_SIZE, info.channels - g * GROUP_SIZE) }, (_, i) => g * GROUP_SIZE + i);
  }, [group, info.channels]);
  const start = win === 'full' ? 0 : Math.min(info.samples, Number(win) * WINDOW_SECONDS * rate);
  const end = win === 'full' ? info.samples : Math.min(info.samples, start + WINDOW_SECONDS * rate);

  useEffect(() => {
    if (!info.rectangular) return undefined;
    const canvas = canvasRef.current;
    const box = boxRef.current;
    const draw = () => {
      const css = getComputedStyle(document.documentElement);
      const color = (n, f) => css.getPropertyValue(n).trim() || f;
      const rowH = group === 'all' ? 22 : 44;
      const labelW = 46; const axisH = 24; const pad = 8;
      const width = Math.max(240, box.clientWidth);
      const height = shown.length * rowH + axisH;
      const dpr = window.devicePixelRatio || 1;
      canvas.width = Math.round(width * dpr); canvas.height = Math.round(height * dpr);
      canvas.style.width = `${width}px`; canvas.style.height = `${height}px`;
      const ctx = canvas.getContext('2d');
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      ctx.clearRect(0, 0, width, height);
      const plotX = labelW; const plotW = width - labelW - pad; const n = end - start;
      const xOfSample = (i) => plotX + ((i - start) / n) * plotW;

      if (eeg.includes_baseline) { // shade the pre-stimulus baseline the backend removes
        const b = Math.min(end, BASELINE_SECONDS * rate);
        if (b > start) {
          ctx.fillStyle = 'rgba(242, 193, 78, 0.10)';
          ctx.fillRect(plotX, 0, xOfSample(b) - plotX, shown.length * rowH);
        }
      }

      const stats = shown.map((c) => robust(eeg.signal[c], start, end));
      const amp = Math.max(...stats.map((s) => s.amp)) || 1; // one common scale for all shown channels
      const half = rowH * 0.45; // `amp` is drawn at 45% of the row height from the centre line
      const edge = amp * (rowH / 2) / half; // file units from the centre line to the row edge
      let clipped = false;
      ctx.font = '12px system-ui, sans-serif'; ctx.textBaseline = 'middle';
      shown.forEach((c, r) => {
        const top = r * rowH; const mid = top + rowH / 2;
        ctx.strokeStyle = 'rgba(244, 241, 234, 0.07)'; ctx.lineWidth = 1;
        ctx.beginPath(); ctx.moveTo(plotX, mid); ctx.lineTo(plotX + plotW, mid); ctx.stroke();
        ctx.fillStyle = color('--muted', '#A9AEB8'); ctx.fillText(label(c), 4, mid);
        ctx.save(); ctx.beginPath(); ctx.rect(plotX, top, plotW, rowH); ctx.clip();
        ctx.strokeStyle = r % 2 ? color('--blue', '#4D7CFE') : color('--teal', '#00B8B0'); ctx.lineWidth = 1;
        ctx.beginPath();
        const ch = eeg.signal[c]; const { median } = stats[r];
        const y = (v) => mid - ((v - median) / amp) * half;
        let started = false;
        for (let px = 0; px < plotW; px += 1) { // min/max envelope of the samples behind each pixel column
          const s0 = start + Math.floor((px / plotW) * n);
          const s1 = Math.max(s0 + 1, start + Math.floor(((px + 1) / plotW) * n));
          let lo = Infinity; let hi = -Infinity;
          for (let i = s0; i < s1 && i < end; i += 1) { const v = ch[i]; if (Number.isFinite(v)) { if (v < lo) lo = v; if (v > hi) hi = v; } }
          if (lo === Infinity) { started = false; continue; } // gap for non-numeric values
          if (Math.abs(lo - median) > edge || Math.abs(hi - median) > edge) clipped = true;
          const x = plotX + px + 0.5;
          if (!started) { ctx.moveTo(x, y(hi)); started = true; } else ctx.lineTo(x, y(hi));
          ctx.lineTo(x, y(lo));
        }
        ctx.stroke(); ctx.restore();
      });

      const axisY = shown.length * rowH;
      ctx.strokeStyle = 'rgba(244, 241, 234, 0.18)';
      ctx.beginPath(); ctx.moveTo(plotX, axisY + 0.5); ctx.lineTo(plotX + plotW, axisY + 0.5); ctx.stroke();
      const t0 = start / rate; const t1 = end / rate; const step = niceStep(t1 - t0);
      ctx.fillStyle = color('--muted', '#A9AEB8'); ctx.textBaseline = 'top'; ctx.textAlign = 'center';
      for (let t = Math.ceil(t0 / step) * step; t <= t1 + 1e-9; t += step) {
        const x = plotX + ((t - t0) / (t1 - t0)) * plotW;
        ctx.fillRect(x, axisY, 1, 4);
        ctx.fillText(`${Number(t.toFixed(1))} s`, Math.min(Math.max(x, plotX + 12), plotX + plotW - 12), axisY + 6);
      }
      ctx.textAlign = 'start';
      setScaleNote(`Common vertical scale: the distance from a channel's centre line to its row edge is about ${fmt(edge)} file units.`
        + `${clipped ? ' Larger excursions are cut off at the row edges.' : ''}`);
    };
    draw();
    const ro = new ResizeObserver(draw);
    ro.observe(box);
    return () => ro.disconnect();
  }, [eeg, info.rectangular, shown, start, end, group, rate]); // eslint-disable-line react-hooks/exhaustive-deps

  const perChannel = useMemo(() => (info.rectangular ? shown.map((c) => ({ c, ...channelStats(eeg.signal[c]) })) : []), [eeg.signal, info.rectangular, shown]);

  if (!info.rectangular) {
    return (
      <p className="muted">
        A preview is not available: {info.minLength < 0 ? 'some channels are not arrays of numbers' : `the channels have different lengths (${info.minLength}–${info.samples} samples)`}.
      </p>
    );
  }
  const range = shown.length ? `${label(shown[0])}–${label(shown[shown.length - 1])}` : '';
  return (
    <div className="eeg-preview">
      <p className="eeg-summary">
        <span className="mono small">{fileName}</span> · {info.channels} {info.channels === 1 ? 'channel' : 'channels'} × {info.samples} samples
        {' '}· {Number(duration.toFixed(2))} s at {rate} Hz{rateGiven ? '' : ' (no sampling rate in the file; 128 Hz assumed, as for the analysis)'}
        {info.nonFinite ? ` · ${info.nonFinite} non-numeric values shown as gaps` : ''}
      </p>
      <div className="field-row eeg-controls">
        <div className="field">
          <label htmlFor={`${ids}-group`}>Channels</label>
          <select id={`${ids}-group`} value={group} onChange={(e) => setGroup(e.target.value)}>
            {Array.from({ length: groups }, (_, g) => {
              const a = g * GROUP_SIZE; const b = Math.min(info.channels, a + GROUP_SIZE) - 1;
              return <option key={g} value={String(g)}>{`${a + 1}–${b + 1} (${label(a)}–${label(b)})`}</option>;
            })}
            {groups > 1 ? <option value="all">All {info.channels} channels</option> : null}
          </select>
        </div>
        <div className="field">
          <label htmlFor={`${ids}-window`}>Time window</label>
          <select id={`${ids}-window`} value={win} onChange={(e) => setWin(e.target.value)}>
            <option value="full">Whole recording (0–{Number(duration.toFixed(1))} s)</option>
            {windows > 1 ? Array.from({ length: windows }, (_, k) => (
              <option key={k} value={String(k)}>{`${k * WINDOW_SECONDS}–${Math.min(duration, (k + 1) * WINDOW_SECONDS).toFixed(0)} s`}</option>
            )) : null}
          </select>
        </div>
      </div>
      <div ref={boxRef} className="eeg-canvas-box">
        <canvas ref={canvasRef} role="img"
          aria-label={`Waveform preview of ${shown.length} channels (${range}) from ${Number((start / rate).toFixed(1))} to ${Number((end / rate).toFixed(1))} seconds of the uploaded file.`} />
      </div>
      <p className="meta small">
        Drawn in your browser from the file you selected; nothing is added or smoothed. Each trace is centred on its
        median, and each pixel column spans the minimum and maximum of the samples it covers. {scaleNote}
        {eeg.includes_baseline ? ` The shaded first ${BASELINE_SECONDS} s are the pre-stimulus baseline, which is removed before analysis.` : ''}
        {names ? ' Channel names assume DEAP (Geneva) order, as the analysis does.' : ''}
      </p>
      <details className="chart-data">
        <summary>Show channel values as a table</summary>
        <div className="table-wrap">
          <table className="table table-compact">
            <caption className="visually-hidden">Minimum, maximum and mean of each shown channel over the whole file, in file units</caption>
            <thead><tr><th scope="col">Channel</th><th scope="col">Minimum</th><th scope="col">Maximum</th><th scope="col">Mean</th></tr></thead>
            <tbody>
              {perChannel.map((r) => <tr key={r.c}><th scope="row">{label(r.c)}</th><td>{fmt(r.min)}</td><td>{fmt(r.max)}</td><td>{fmt(r.mean)}</td></tr>)}
            </tbody>
          </table>
        </div>
      </details>
    </div>
  );
}
