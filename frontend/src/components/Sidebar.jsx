import { useEffect, useState } from 'react';
import { href } from '../router';

// Primary workflow: what most users need.
const PRIMARY = [
  { path: '/overview', label: 'Overview' },
  { path: '/analyze', label: 'Analyze EEG' },
  { path: '/dashboard', label: 'Dashboard' },
  { path: '/history', label: 'History' },
];

// Secondary research tools: read-only evidence and reference pages.
const RESEARCH = [
  { path: '/model', label: 'Models' },
  { path: '/dataset', label: 'Dataset' },
  { path: '/models', label: 'Model registry' },
];

// Shown to administrators only (a convenience; the server authorizes every request itself).
const ADMIN = [
  { path: '/admin/training', label: 'Training' },
  { path: '/system', label: 'System health' },
];

function NavList({ items, current }) {
  return (
    <ul className="nav-list">
      {items.map((item) => {
        const active = current === item.path || (item.path === '/history' && current.startsWith('/results/'));
        return (
          <li key={item.path}>
            <a href={href(item.path)} className={active ? 'nav-link active' : 'nav-link'} aria-current={active ? 'page' : undefined}>
              {item.label}
            </a>
          </li>
        );
      })}
    </ul>
  );
}

/** A collapsible group; it opens automatically while one of its pages is shown. */
function NavGroup({ id, title, items, current, admin = false }) {
  const containsCurrent = items.some((i) => i.path === current);
  const [open, setOpen] = useState(containsCurrent);
  useEffect(() => { if (containsCurrent) setOpen(true); }, [containsCurrent]);
  return (
    <div className={`nav-section ${admin ? 'nav-section-admin' : ''}`}>
      <button type="button" className="nav-group-toggle" aria-expanded={open} aria-controls={id} onClick={() => setOpen((o) => !o)}>
        <span>{title}</span><span className="nav-caret" aria-hidden="true">{open ? '−' : '+'}</span>
      </button>
      <div id={id} hidden={!open}><NavList items={items} current={current} /></div>
    </div>
  );
}

export default function Sidebar({ id, open, currentPath, auth }) {
  return (
    <aside id={id} className={`sidebar ${open ? 'open' : ''}`}>
      <div className="brand">
        <span className="brand-mark" aria-hidden="true" />
        <div>
          <span className="brand-name">NeuroSense AI</span>
          <span className="brand-tag">EEG emotion estimation with explainable AI</span>
        </div>
      </div>

      <nav aria-label="Main">
        <NavList items={PRIMARY} current={currentPath} />
        <NavGroup id="nav-research" title="Research tools" items={RESEARCH} current={currentPath} />
        {auth.user?.is_admin ? (
          <NavGroup id="nav-admin" title="Administration" items={ADMIN} current={currentPath} admin />
        ) : null}
      </nav>

      <div className="account">
        {auth.user ? (
          <>
            <span className="account-email" title={auth.user.email}>{auth.user.email}</span>
            {auth.user.is_admin ? <span className="tag">Administrator</span> : null}
            <button type="button" className="btn btn-ghost btn-small" onClick={auth.signOut}>Sign out</button>
          </>
        ) : auth.status === 'checking' ? (
          <span className="muted small">Checking session…</span>
        ) : (
          <a className="btn btn-secondary btn-small" href={href('/sign-in')}>Sign in</a>
        )}
      </div>
    </aside>
  );
}
