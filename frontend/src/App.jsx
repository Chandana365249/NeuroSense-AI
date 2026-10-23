import { useCallback, useEffect, useMemo, useState } from 'react';
import { api } from './api';
import AuthScreen from './components/AuthScreen';
import Layout from './components/Layout';
import Analyze from './pages/Analyze';
import History from './pages/History';
import Dataset from './pages/Dataset';
import ModelEvidence from './pages/ModelEvidence';
import ModelRegistry from './pages/ModelRegistry';
import NotFound from './pages/NotFound';
import Overview from './pages/Overview';
import ResultDetail from './pages/ResultDetail';
import { safeNext } from './pages/SignIn';
import SystemHealth from './pages/SystemHealth';
import Training from './pages/admin/Training';
import { matchPath, navigate, useRoute } from './router';

const TOKEN_KEY = 'neurosense_token';
const SESSION_ENDED = 'Your session has ended. Please sign in again.';

function readToken() {
  try {
    return sessionStorage.getItem(TOKEN_KEY);
  } catch {
    return null;
  }
}

function saveToken(token) {
  try {
    if (token) sessionStorage.setItem(TOKEN_KEY, token);
    else sessionStorage.removeItem(TOKEN_KEY);
  } catch {
    /* storage unavailable; token stays in memory only */
  }
}

/** Session state. status: 'checking' | 'signed-in' | 'signed-out' | 'error' (service unreachable). */
function useAuth() {
  const [token, setToken] = useState(readToken);
  const [user, setUser] = useState(null);
  const [status, setStatus] = useState(token ? 'checking' : 'signed-out');
  const [error, setError] = useState(null);
  const [notice, setNotice] = useState(null);
  const [attempt, setAttempt] = useState(0); // bumped by retry() to re-check the session

  useEffect(() => {
    if (!token) {
      setUser(null);
      setStatus('signed-out');
      return undefined;
    }
    let cancelled = false;
    setStatus('checking');
    api.me(token)
      .then((u) => {
        if (cancelled) return;
        setUser(u);
        setError(null);
        setStatus('signed-in');
      })
      .catch((err) => {
        if (cancelled) return;
        setUser(null);
        if (err.status === 401) {
          saveToken(null);
          setToken(null);
          setNotice(SESSION_ENDED);
        } else {
          setError(err.status === 0 ? 'The service is unreachable.' : err.message);
          setStatus('error');
        }
      });
    return () => { cancelled = true; };
  }, [token, attempt]);

  const signIn = useCallback((newToken) => {
    saveToken(newToken);
    setNotice(null);
    setToken(newToken);
  }, []);
  const signOut = useCallback(() => {
    saveToken(null);
    setNotice(null);
    setToken(null);
  }, []);
  const expire = useCallback(() => {
    saveToken(null);
    setToken(null);
    setNotice(SESSION_ENDED);
  }, []);

  const retry = useCallback(() => setAttempt((n) => n + 1), []);

  return useMemo(() => ({ token, user, status, error, notice, signIn, signOut, expire, retry }),
    [token, user, status, error, notice, signIn, signOut, expire, retry]);
}

/** Public data used by several pages. status: 'loading' | 'ready' | 'error' (service unreachable). */
function usePublicData() {
  const [state, setState] = useState({ status: 'loading' });
  const load = useCallback(async () => {
    setState((s) => ({ ...s, status: 'loading' }));
    let health;
    try {
      health = await api.health();
    } catch (err) {
      setState({ status: 'error', error: err.message });
      return;
    }
    const [info, demo] = await Promise.allSettled([api.modelInfo(), api.demoTrials()]);
    // A failed request is kept as an error, never replaced by stand-in values.
    setState({
      status: 'ready',
      loadedAt: Date.now(), // lets pages refresh their own data when the user retries
      health,
      modelInfo: info.status === 'fulfilled' ? info.value : null,
      modelInfoError: info.status === 'rejected' ? info.reason?.message || 'The active model could not be loaded.' : null,
      demo: demo.status === 'fulfilled' ? { status: 'ready', data: demo.value }
        : { status: 'error', error: demo.reason?.message || 'Request failed' },
    });
  }, []);
  useEffect(() => { load(); }, [load]);
  return useMemo(() => ({ ...state, reload: load }), [state, load]);
}

function renderPage(route, auth, publicData) {
  const { path } = route;
  if (path === '/overview') return <Overview auth={auth} publicData={publicData} />;
  if (path === '/analyze') return <Analyze auth={auth} publicData={publicData} />;
  if (path === '/history') return <History auth={auth} />;
  if (path === '/model') return <ModelEvidence publicData={publicData} />;
  if (path === '/models') return <ModelRegistry auth={auth} publicData={publicData} />;
  if (path === '/dataset') return <Dataset />;
  if (path === '/system') return <SystemHealth auth={auth} />;
  if (path === '/admin/training') return <Training auth={auth} />;
  const result = matchPath('/results/:id', path);
  if (result && /^\d+$/.test(result.id)) return <ResultDetail key={result.id} auth={auth} id={result.id} />;
  return <NotFound path={path} />;
}

const TITLES = {
  '/overview': 'Overview', '/analyze': 'Analyze EEG', '/history': 'History', '/model': 'Model & Evidence',
  '/models': 'Model registry', '/dataset': 'Dataset', '/system': 'System health',
  '/admin/training': 'Training', '/sign-in': 'Sign in',
};

export default function App() {
  const route = useRoute();
  const auth = useAuth();
  const publicData = usePublicData();

  const signedIn = auth.status === 'signed-in';
  useEffect(() => {
    const page = !signedIn ? 'Sign in'
      : TITLES[route.path] || (route.path.startsWith('/results/') ? 'Result' : 'NeuroSense AI');
    document.title = page === 'NeuroSense AI' ? page : `${page} · NeuroSense AI`;
  }, [route.path, signedIn]);

  // After signing in on the sign-in route, continue to the requested page (internal routes only).
  useEffect(() => {
    if (signedIn && route.path === '/sign-in') navigate(safeNext(route.query.get('next')), { replace: true });
  }, [signedIn, route]);

  // Until a session is verified, only the standalone sign-in screen is rendered: no application shell, navigation or
  // data pages. The backend independently requires a valid token for every user-data endpoint.
  if (!signedIn) {
    const next = route.path === '/sign-in' ? route.query.get('next') : `${route.path}${route.query.toString() ? `?${route.query}` : ''}`;
    return <AuthScreen auth={auth} next={next} />;
  }
  if (route.path === '/sign-in') return null; // redirecting (effect above)

  return (
    <Layout route={route} auth={auth}>
      {renderPage(route, auth, publicData)}
    </Layout>
  );
}
