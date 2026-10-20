// Each page's h1 is focusable so the layout can move focus to it after navigation (screen-reader friendly).
export default function PageHeader({ title, subtitle, children }) {
  return (
    <header className="page-header">
      <div>
        <h1 id="page-title" tabIndex={-1}>{title}</h1>
        {subtitle ? <p className="page-subtitle">{subtitle}</p> : null}
      </div>
      {children ? <div className="page-header-actions">{children}</div> : null}
    </header>
  );
}
