// tone: 'info' | 'warning' | 'error'. Errors are announced immediately (role=alert); others politely.
// announce={false} when the surrounding page announces the outcome itself.
export default function Notice({ tone = 'info', title, children, action, announce = true }) {
  const role = !announce ? undefined : tone === 'error' ? 'alert' : 'status';
  return (
    <div className={`notice notice-${tone}`} role={role}>
      <div>
        {title ? <strong className="notice-title">{title}</strong> : null}
        {children ? <div className="notice-body">{children}</div> : null}
      </div>
      {action ? <div className="notice-action">{action}</div> : null}
    </div>
  );
}
