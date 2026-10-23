// Sign-in and registration form rules (pure, unit-tested). The server validates everything again.

/** Checks done in the browser before any request; the server still validates everything. */
export function localErrors(mode, email, password) {
  const errors = {};
  if (!email.trim()) errors.email = 'Enter your email address.';
  else if (!/^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(email.trim())) errors.email = 'Enter a valid email address, such as name@example.com.';
  if (!password) errors.password = 'Enter your password.';
  else if (mode === 'register' && password.length < 8) errors.password = 'Use at least 8 characters.';
  return errors;
}

/** Plain-language message for a failed request that names no field. */
export function authFailureMessage(err, mode) {
  if (err?.status === 0) return 'The service is unreachable. Please try again shortly.';
  if (err?.status === 401) return mode === 'login' ? 'Incorrect email or password. Please check them and try again.' : err.message;
  if (err?.status === 409) return 'An account with this email already exists. Sign in instead.';
  if (err?.status === 429) return 'Too many attempts. Please wait a moment and try again.';
  if (err?.status === 422) return 'Some details could not be accepted. Please check the form and try again.';
  return err?.message || 'Something went wrong. Please try again.';
}
