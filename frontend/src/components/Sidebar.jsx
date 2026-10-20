import { href } from '../router';

const WORKSPACE = [
  { path: '/overview', label: 'Overview' },
  { path: '/analyze', label: 'Analyze EEG' },
  { path: '/history', label: 'History' },
  { path: '/model', label: 'Model & Evidence' },
];

const REFERENCE = [
  { path: '/models', label: 'Model registry' },
  { path: '/dataset', label: 'Dataset' },
];

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

export default function Sidebar({ id, open, currentPath, auth }) {
  return (
    <aside id={id} className={`sidebar ${open ? 'open' : ''}`}>
      <div className="brand">
        <span className="brand-mark" aria-hidden="true" />
        <div>
          <span className="brand-name">NeuroSense AI</span>
          <span className="brand-tag">Self-reported valence estimation from EEG · research prototype</span>
        </div>
      </div>

      <nav aria-label="Main">
        <p className="nav-group" id="nav-workspace">Research workspace</p>
        <NavList items={WORKSPACE} current={currentPath} />
        <p className="nav-group nav-group-sep" id="nav-reference">Reference</p>
        <NavList items={REFERENCE} current={currentPath} />
        {auth.user?.is_admin ? (
          <>
            <p className="nav-group nav-group-admin" id="nav-admin">Research administration</p>
            <NavList items={ADMIN} current={currentPath} />
          </>
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
