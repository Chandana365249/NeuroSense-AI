// Activation-dialog rules (pure functions, unit-tested in frontend/tests). The server validates everything again.

export const REASON_MIN = 10;
export const REASON_MAX = 500;

/** Client-side checks mirror the server's rules; the server validates again and is the real boundary. */
export function validate(version, confirm, reason) {
  const errors = {};
  if (confirm !== version) {
    errors.confirm = confirm ? 'This does not match the version exactly.' : 'Type the version exactly as shown above.';
  }
  const n = reason.trim().length;
  if (n < REASON_MIN) errors.reason = `Enter at least ${REASON_MIN} characters (${n} after trimming spaces).`;
  else if (n > REASON_MAX) errors.reason = `Use at most ${REASON_MAX} characters (${n} after trimming spaces).`;
  return errors;
}

/** Field messages from a 422 validation response (never the submitted values). */
export function fieldErrorsFrom(err) {
  const errors = {};
  for (const d of err.details || []) {
    const field = d.loc?.[d.loc.length - 1];
    const msg = String(d.msg || '').replace(/^Value error, /, '');
    if (field === 'reason') errors.reason = msg || 'The reason was not accepted.';
    if (field === 'confirm_version') errors.confirm = msg || 'The confirmation was not accepted.';
  }
  return errors;
}

// Server messages are only shown when they are one of the known, value-free activation messages.
export const SAFE_SERVER_MESSAGE = /(it was not activated\.|confirm_version must repeat the version|appears to contain a credential or secret)/;

export function failureMessage(err) {
  if (err.status === 0) {
    return { title: 'The service could not be reached',
      body: 'The request may not have reached the server. Reload the registry to see which model is active.' };
  }
  if (err.status === 403) {
    return { title: 'Administrator access required', body: 'The server did not accept this account as an administrator. Nothing was changed.' };
  }
  if (err.status === 404) {
    return { title: 'Model version not found',
      body: 'This version is no longer registered on the server. Nothing was changed. Close this dialog and reload the registry.' };
  }
  if (err.status === 409) {
    return { title: 'Activation is not possible right now',
      body: 'A training run or another activation is in progress. Nothing was changed; try again after it finishes.' };
  }
  if (err.status === 422) {
    const detail = SAFE_SERVER_MESSAGE.test(err.message) ? `${err.message} ` : 'The server rejected this model as invalid for activation. ';
    return { title: 'The model was not activated', body: `${detail}The current model remains active.` };
  }
  return { title: 'Activation failed',
    body: 'The server reported an error while switching models. When activation fails the server restores the previous state, '
      + 'so the previous model should remain active. Reload the registry to confirm.' };
}
