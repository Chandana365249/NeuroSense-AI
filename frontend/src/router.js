import { useEffect, useState } from 'react';

// Minimal hash router: #/path?query. Direct links, reloads and browser back/forward all go through
// the hashchange event, so no history API or extra dependency is needed.

export const DEFAULT_PATH = '/overview';

export function parseHash(hash) {
  const raw = (hash || '').replace(/^#/, '');
  const [pathPart, query = ''] = raw.split('?');
  const path = `/${pathPart.replace(/^\/+|\/+$/g, '')}`;
  return { path: path === '/' ? DEFAULT_PATH : path, query: new URLSearchParams(query) };
}

export function useRoute() {
  const [route, setRoute] = useState(() => parseHash(window.location.hash));
  useEffect(() => {
    const onChange = () => setRoute(parseHash(window.location.hash));
    window.addEventListener('hashchange', onChange);
    return () => window.removeEventListener('hashchange', onChange);
  }, []);
  return route;
}

export const href = (path) => `#${path}`;

/** `replace` swaps the current history entry (used for redirects, so Back does not loop). */
export function navigate(path, { replace = false } = {}) {
  if (replace) window.location.replace(`#${path}`);
  else window.location.hash = path;
}

/** Match '/results/:id' against '/results/12' -> { id: '12' }; null when it does not match. */
export function matchPath(pattern, path) {
  const a = pattern.split('/').filter(Boolean);
  const b = path.split('/').filter(Boolean);
  if (a.length !== b.length) return null;
  const params = {};
  for (let i = 0; i < a.length; i += 1) {
    if (a[i].startsWith(':')) params[a[i].slice(1)] = decodeURIComponent(b[i]);
    else if (a[i] !== b[i]) return null;
  }
  return params;
}
