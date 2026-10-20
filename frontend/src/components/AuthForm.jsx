import { useId, useRef, useState } from 'react';
import { api } from '../api';

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

/** Sign-in / registration form. Registration then signs in, as before. */
export default function AuthForm({ onAuth }) {
  const [mode, setMode] = useState('login');
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [fullName, setFullName] = useState('');
  const [error, setError] = useState(null);
  const [fieldErrors, setFieldErrors] = useState({});
  const [busy, setBusy] = useState(false);
  const id = useId();
  const inputs = { full_name: useRef(null), email: useRef(null), password: useRef(null) };

  const switchMode = (next) => { setMode(next); setError(null); setFieldErrors({}); };

  const submit = async (event) => {
    event.preventDefault();
    if (busy) return;
    setBusy(true);
    setError(null);
    setFieldErrors({});
    try {
      if (mode === 'register') await api.register(email, password, fullName);
      const { access_token: token } = await api.login(email, password);
      onAuth(token);
    } catch (err) {
      const fields = err.status === 422 ? fieldErrorsFrom(err) : {};
      const first = FIELD_ORDER.find((f) => fields[f]);
      if (first) {
        setFieldErrors(fields);
        setError(Object.keys(fields).length > 1 ? 'Please correct the highlighted fields.' : 'Please correct the highlighted field.');
        setTimeout(() => inputs[first].current?.focus(), 0);
      } else {
        setError(err.status === 0 ? 'The service is unreachable. Please try again shortly.'
          : err.status === 422 ? 'Some details could not be accepted. Please check the form and try again.'
            : err.message);
      }
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

  return (
    <form className="auth-form" onSubmit={submit} noValidate={false}>
      <div className="segmented" role="group" aria-label="Account action">
        <button type="button" aria-pressed={mode === 'login'} className={mode === 'login' ? 'seg-active' : ''}
          onClick={() => switchMode('login')}>Sign in</button>
        <button type="button" aria-pressed={mode === 'register'} className={mode === 'register' ? 'seg-active' : ''}
          onClick={() => switchMode('register')}>Create account</button>
      </div>
      {mode === 'register' ? (
        <div className="field">
          <label htmlFor={`${id}-name`}>Full name <span className="optional">(optional)</span></label>
          <input id={`${id}-name`} autoComplete="name" value={fullName} onChange={(e) => setFullName(e.target.value)}
            maxLength={200} {...fieldProps('full_name')} />
          {fieldError('full_name')}
        </div>
      ) : null}
      <div className="field">
        <label htmlFor={`${id}-email`}>Email</label>
        <input id={`${id}-email`} type="email" autoComplete="email" value={email} onChange={(e) => setEmail(e.target.value)} required
          {...fieldProps('email')} />
        {fieldError('email')}
      </div>
      <div className="field">
        <label htmlFor={`${id}-password`}>Password</label>
        <input id={`${id}-password`} type="password" autoComplete={mode === 'login' ? 'current-password' : 'new-password'}
          value={password} onChange={(e) => setPassword(e.target.value)} minLength={8} maxLength={72} required
          {...fieldProps('password', `${id}-password-hint`)} />
        <span id={`${id}-password-hint`} className="field-hint">8–72 characters</span>
        {fieldError('password')}
      </div>
      <button type="submit" className="btn btn-primary" aria-disabled={busy || undefined}>
        {busy ? 'Please wait…' : mode === 'login' ? 'Sign in' : 'Create account'}
      </button>
      <div aria-live="assertive">{error ? <p className="form-error">{error}</p> : null}</div>
    </form>
  );
}
