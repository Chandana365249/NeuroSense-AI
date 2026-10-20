// titleRef makes the heading a programmatic focus target (e.g. after an analysis finishes).
export default function Card({ title, titleId, titleRef, actions, children, className = '' }) {
  return (
    <section className={`card ${className}`} aria-labelledby={title ? titleId : undefined}>
      {title || actions ? (
        <div className="card-head">
          {title ? <h2 id={titleId} ref={titleRef} tabIndex={titleRef ? -1 : undefined}>{title}</h2> : null}
          {actions ? <div className="card-actions">{actions}</div> : null}
        </div>
      ) : null}
      {children}
    </section>
  );
}
