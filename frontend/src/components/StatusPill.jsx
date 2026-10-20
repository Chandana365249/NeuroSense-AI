// tone: 'ok' | 'warn' | 'error' | 'neutral'. The text always states the status; colour is only a secondary cue.
export default function StatusPill({ tone = 'neutral', children }) {
  return (
    <span className={`pill pill-${tone}`}>
      <span className="pill-dot" aria-hidden="true" />
      {children}
    </span>
  );
}
