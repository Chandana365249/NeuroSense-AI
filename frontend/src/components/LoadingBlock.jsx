export default function LoadingBlock({ label = 'Loading…', lines = 3, announce = true }) {
  return (
    <div className="loading" role={announce ? 'status' : undefined}>
      <span className="visually-hidden">{label}</span>
      {Array.from({ length: lines }, (_, i) => (
        <span key={i} className="skeleton" aria-hidden="true" style={{ width: `${90 - i * 15}%` }} />
      ))}
    </div>
  );
}
