import { useEffect, useRef, useState } from 'react';
import Sidebar from './Sidebar';

export default function Layout({ route, auth, children }) {
  const [menuOpen, setMenuOpen] = useState(false);
  const mainRef = useRef(null);
  const lastPath = useRef(null);

  // After client-side navigation (not on first load, and not on React's dev double-run of effects):
  // close the mobile menu and move focus to the new page heading.
  useEffect(() => {
    const previous = lastPath.current;
    lastPath.current = route.path;
    if (previous === null || previous === route.path) return;
    setMenuOpen(false);
    const heading = document.getElementById('page-title');
    (heading || mainRef.current)?.focus();
    window.scrollTo(0, 0);
  }, [route.path]);

  // Signing in or out replaces the focused form or button; put focus back on the page heading.
  const userId = auth.user?.id ?? null;
  const lastUser = useRef(userId);
  useEffect(() => {
    if (lastUser.current === userId) return;
    lastUser.current = userId;
    const active = document.activeElement;
    if (!active || active === document.body) (document.getElementById('page-title') || mainRef.current)?.focus();
  }, [userId]);

  const skipToMain = (event) => {
    event.preventDefault(); // a plain "#main" link would change the hash route
    mainRef.current?.focus();
  };

  return (
    <div className="app-shell">
      <a href="#main" className="skip-link" onClick={skipToMain}>Skip to main content</a>
      <header className="topbar">
        <span className="brand-name">NeuroSense AI</span>
        <button type="button" className="btn btn-ghost btn-small" aria-expanded={menuOpen} aria-controls="sidebar"
          onClick={() => setMenuOpen((o) => !o)}>
          {menuOpen ? 'Close menu' : 'Menu'}
        </button>
      </header>
      <Sidebar id="sidebar" open={menuOpen} currentPath={route.path} auth={auth} />
      <main id="main" ref={mainRef} tabIndex={-1} className="main">
        {auth.notice ? <div className="session-notice" role="status">{auth.notice}</div> : null}
        {children}
      </main>
    </div>
  );
}
