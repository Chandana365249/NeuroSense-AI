import { useId, useRef, useState } from 'react';
import { api } from '../api';
import { authFailureMessage, localErrors } from '../authRules';

const FIELD_ORDER = ['full_name', 'email', 'password'];

// Plain-language message for one backend validation error; raw validator text is never shown.
function fieldMessage(field, type) {
  if (type === 'missing') return 'This field is required.';
  if (field === 'email') return 'Enter a valid email address, such as name@example.com. Reserved or test-only domains are not accepted.';
  if (field === 'password') {
    if (type === 'string_too_short') return 'Use at least 8 characters.';
    return 'Use at most 72 characters (72 bytes).';
  }
  if (field === 'full_name') return 'Use at most 200 characters.';
  return null;
}

/** Maps a 422 response's details to {field: message}; empty when it names no known field. */
function fieldErrorsFrom(err) {
  const result = {};
  for (const item of err?.details || []) {
    const loc = Array.isArray(item?.loc) ? item.loc : [];
    const raw = loc[0] === 'body' ? loc[1] : loc[loc.length - 1];
    const field = raw === 'username' ? 'email' : raw; // the login form sends the email as "username"
    if (!FIELD_ORDER.includes(field) || result[field]) continue;
    const message = fieldMessage(field, item.type);
    if (message) result[field] = message;
  }
  return result;
}

/**
 * Sign-in / registration form. Registration then signs in, as before. `onModeChange` lets a surrounding page follow
 * the mode (for its heading); `initialMode` is 'login' or 'register'.
 */
export default function AuthForm({ onAuth, onModeChange, initialMode = 'login' }) {
  const [mode, setMode] = useState(initialMode);
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [fullName, setFullName] = useState('');
  const [error, setError] = useState(null);
  const [fieldErrors, setFieldErrors] = useState({});
  const [busy, setBusy] = useState(false);
  const id = useId();
  const inputs = { full_name: useRef(null), email: useRef(null), password: useRef(null) };

  const switchMode = (next) => {
    setMode(next); setError(null); setFieldErrors({});
    onModeChange?.(next);
    setTimeout(() => inputs[next === 'register' ? 'full_name' : 'email'].current?.focus(), 0);
  };

  const showFieldErrors = (fields) => {
    const first = FIELD_ORDER.find((f) => fields[f]);
    setFieldErrors(fields);
    setError(Object.keys(fields).length > 1 ? 'Please correct the highlighted fields.' : 'Please correct the highlighted field.');
    setTimeout(() => inputs[first].current?.focus(), 0);
  };

  const submit = async (event) => {
    event.preventDefault();
    if (busy) return;
    const local = localErrors(mode, email, password);
    if (Object.keys(local).length) { showFieldErrors(local); return; }
    setBusy(true);
    setError(null);
    setFieldErrors({});
    try {
      if (mode === 'register') await api.register(email.trim(), password, fullName);
      const { access_token: token } = await api.login(email.trim(), password);
      onAuth(token);
    } catch (err) {
      const fields = err.status === 422 ? fieldErrorsFrom(err) : {};
      if (FIELD_ORDER.some((f) => fields[f])) showFieldErrors(fields);
      else setError(authFailureMessage(err, mode));
    } finally {
      setBusy(false);
    }
  };

  // Props linking an input to its hint and (if any) its error message.
  const fieldProps = (field, hintId) => {
    const errorId = fieldErrors[field] ? `${id}-${field}-error` : null;
    const describedBy = [hintId, errorId].filter(Boolean).join(' ') || undefined;
    return { ref: inputs[field], 'aria-invalid': fieldErrors[field] ? true : undefined, 'aria-describedby': describedBy };
  };
  const fieldError = (field) => (fieldErrors[field]
    ? <span id={`${id}-${field}-error`} className="form-error field-error">{fieldErrors[field]}</span> : null);
  const login = mode === 'login';

  return (
    <form className="auth-form" onSubmit={submit} noValidate aria-label={login ? 'Sign in' : 'Create account'}>
      {!login ? (
        <div className="field">
          <label htmlFor={`${id}-name`}>Full name <span className="optional">(optional)</span></label>
          <input id={`${id}-name`} autoComplete="name" value={fullName} onChange={(e) => setFullName(e.target.value)}
            maxLength={200} {...fieldProps('full_name')} />
          {fieldError('full_name')}
        </div>
      ) : null}
      <div className="field">
        <label htmlFor={`${id}-email`}>Email</label>
        <input id={`${id}-email`} type="email" autoComplete="email" inputMode="email" value={email}
          onChange={(e) => setEmail(e.target.value)} required {...fieldProps('email')} />
        {fieldError('email')}
      </div>
      <div className="field">
        <label htmlFor={`${id}-password`}>Password</label>
        <input id={`${id}-password`} type="password" autoComplete={login ? 'current-password' : 'new-password'}
          value={password} onChange={(e) => setPassword(e.target.value)} minLength={8} maxLength={72} required
          {...fieldProps('password', login ? undefined : `${id}-password-hint`)} />
        {!login ? <span id={`${id}-password-hint`} className="field-hint">8–72 characters</span> : null}
        {fieldError('password')}
      </div>
      <div aria-live="assertive">{error ? <p className="form-error auth-error" role="alert">{error}</p> : null}</div>
      <button type="submit" className="btn btn-primary btn-block" disabled={busy} aria-busy={busy || undefined}>
        {busy ? (login ? 'Signing in…' : 'Creating account…') : login ? 'Sign in' : 'Create account'}
      </button>
      <p className="auth-switch">
        {login ? 'New to NeuroSense AI?' : 'Already have an account?'}{' '}
        <button type="button" className="link-button" onClick={() => switchMode(login ? 'register' : 'login')} disabled={busy}>
          {login ? 'Create an account' : 'Sign in'}
        </button>
      </p>
    </form>
  );
}
