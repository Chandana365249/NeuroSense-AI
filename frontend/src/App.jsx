import { useCallback, useEffect, useMemo, useState } from 'react';
import { api } from './api';
import Layout from './components/Layout';
import Analyze from './pages/Analyze';
import History from './pages/History';
import Dataset from './pages/Dataset';
import ModelEvidence from './pages/ModelEvidence';
import ModelRegistry from './pages/ModelRegistry';
import NotFound from './pages/NotFound';
import Overview from './pages/Overview';
import ResultDetail from './pages/ResultDetail';
import SignIn from './pages/SignIn';
import SystemHealth from './pages/SystemHealth';
import Training from './pages/admin/Training';
import { matchPath, useRoute } from './router';

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
  }, [token]);

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

  return useMemo(() => ({ token, user, status, error, notice, signIn, signOut, expire }),
    [token, user, status, error, notice, signIn, signOut, expire]);
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
    setState({
      status: 'ready',
      health,
      modelInfo: info.status === 'fulfilled' ? info.value : null,
      demo: demo.status === 'fulfilled' ? demo.value : null,
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
  if (path === '/sign-in') return <SignIn auth={auth} next={route.query.get('next')} />;
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

  useEffect(() => {
    const page = TITLES[route.path] || (route.path.startsWith('/results/') ? 'Result' : 'NeuroSense AI');
    document.title = page === 'NeuroSense AI' ? page : `${page} · NeuroSense AI`;
  }, [route.path]);

  return (
    <Layout route={route} auth={auth}>
      {renderPage(route, auth, publicData)}
    </Layout>
  );
}
