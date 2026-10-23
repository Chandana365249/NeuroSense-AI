import { useEffect, useRef, useState } from 'react';
import AuthForm from './AuthForm';

/**
 * Standalone first screen for visitors without a verified session. It renders no application shell, navigation or
 * data. After sign-in the app shows the page that was requested (the URL is kept), or Overview by default.
 */
export default function AuthScreen({ auth, next }) {
  const [mode, setMode] = useState('login');
  const heading = useRef(null);

  // Land keyboard and screen-reader users on the heading once the form is shown.
  useEffect(() => {
    if (auth.status === 'signed-out') heading.current?.focus();
  }, [auth.status]);

  let body;
  if (auth.status === 'checking') {
    body = (
      <div className="auth-state" role="status">
        <span className="spinner" aria-hidden="true" />
        <p>Checking your session…</p>
      </div>
    );
  } else if (auth.status === 'error') {
    body = (
      <div className="auth-state">
        <p className="form-error" role="alert">Could not verify your session. {auth.error || 'The service is unreachable.'}</p>
        <button type="button" className="btn btn-primary btn-block" onClick={auth.retry}>Try again</button>
        <button type="button" className="btn btn-ghost btn-block" onClick={auth.signOut}>Sign in with a different account</button>
      </div>
    );
  } else {
    body = (
      <>
        <h1 id="page-title" ref={heading} tabIndex={-1}>{mode === 'login' ? 'Welcome back' : 'Create your account'}</h1>
        <p className="auth-lead">
          {mode === 'login'
            ? 'Sign in to analyze EEG recordings and explore your research history.'
            : 'Create an account to analyze EEG recordings and keep a history of your analyses.'}
        </p>
        {auth.notice ? <div className="session-notice" role="status">{auth.notice}</div> : null}
        <AuthForm onAuth={auth.signIn} onModeChange={setMode} />
      </>
    );
  }

  return (
    <div className="auth-screen">
      <main id="main" className="auth-main" data-next={next || undefined}>
        <div className="auth-card">
          <div className="auth-brand">
            <span className="brand-mark" aria-hidden="true" />
            <span className="auth-brand-name">NeuroSense AI</span>
          </div>
          {body}
        </div>
        <p className="auth-foot">
          Research prototype for estimating self-reported valence from EEG. Not a diagnostic tool.
        </p>
      </main>
    </div>
  );
}
