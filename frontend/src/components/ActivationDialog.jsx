import { useEffect, useId, useRef, useState } from 'react';
import { api } from '../api';
import Notice from './Notice';
import { failureMessage, fieldErrorsFrom, REASON_MAX, REASON_MIN, validate } from '../activationRules';
import { heldOutSummary, NOT_RECORDED, trainingDataSummary } from '../modelInfo';
import { algorithmLabel, evaluationSummary, NA, num, pct, when } from './format';

export { REASON_MAX, REASON_MIN };

/**
 * Confirmation dialog for activating a registered model. The caller renders it only for administrator accounts;
 * the server enforces authorization independently. The reason stays in component state and the request body only.
 */
export default function ActivationDialog({ model, activeVersion, auth, onCancel, onActivated }) {
  const dialogRef = useRef(null);
  const confirmRef = useRef(null);
  const reasonRef = useRef(null);
  const busyRef = useRef(false);
  const id = useId();
  const [confirm, setConfirm] = useState('');
  const [reason, setReason] = useState('');
  const [submitted, setSubmitted] = useState(false);
  const [serverErrors, setServerErrors] = useState({});
  const [busy, setBusy] = useState(false);
  const [failure, setFailure] = useState(null);

  useEffect(() => {
    const dialog = dialogRef.current;
    if (dialog && !dialog.open) dialog.showModal();
    confirmRef.current?.focus();
  }, []);

  useEffect(() => {
    const dialog = dialogRef.current;
    const onDialogCancel = (e) => {
      e.preventDefault(); // Escape: close through React state (and never while a request is running)
      if (!busyRef.current) onCancel();
    };
    dialog?.addEventListener('cancel', onDialogCancel);
    return () => dialog?.removeEventListener('cancel', onDialogCancel);
  }, [onCancel]);

  const errors = submitted ? { ...validate(model.version, confirm, reason), ...serverErrors } : serverErrors;
  const trimmed = reason.trim().length;
  const m = model.test_metrics || {};

  async function submit(event) {
    event.preventDefault();
    if (busyRef.current) return;
    const clientErrors = validate(model.version, confirm, reason);
    setSubmitted(true);
    setServerErrors({});
    setFailure(null);
    if (Object.keys(clientErrors).length) {
      (clientErrors.confirm ? confirmRef : reasonRef).current?.focus();
      return;
    }
    busyRef.current = true;
    setBusy(true);
    try {
      const result = await api.activateModel(auth.token, model.version, { confirmVersion: confirm, reason: reason.trim() });
      busyRef.current = false;
      onActivated(result);
      return;
    } catch (err) {
      busyRef.current = false;
      setBusy(false);
      if (err.status === 401) { onCancel(); auth.expire(); return; }
      const fieldErrors = err.code === 'validation_error' ? fieldErrorsFrom(err) : {};
      if (Object.keys(fieldErrors).length) {
        setServerErrors(fieldErrors);
        setFailure({ title: 'Please correct the highlighted field', body: 'Nothing was changed.' });
        (fieldErrors.confirm ? confirmRef : reasonRef).current?.focus();
      } else {
        setFailure(failureMessage(err));
      }
    }
  }

  const describe = (field, hint) => [hint, errors[field] ? `${id}-${field}-error` : null].filter(Boolean).join(' ');

  return (
    <dialog ref={dialogRef} className="dialog" aria-labelledby={`${id}-title`} aria-describedby={`${id}-intro`}>
      <form onSubmit={submit} noValidate>
        <h2 id={`${id}-title`}>Activate model</h2>
        <p id={`${id}-intro`}>
          Activation replaces the model that serves all new predictions. The server checks the model&apos;s integrity,
          feature schema and class labels before switching, and records the change and your reason in the audit log.
        </p>
        <dl className="kv kv-compact">
          <dt>Version to activate</dt><dd className="mono">{model.version}</dd>
          <dt>Currently active</dt><dd className="mono">{activeVersion || 'No active model'}</dd>
          <dt>Model</dt><dd>{algorithmLabel(model.model_type || model.algorithm)}</dd>
          <dt>Trained</dt><dd>{when(model.trained_at)}</dd>
          <dt>Training data</dt><dd>{trainingDataSummary(model.provenance)}</dd>
          <dt>Held out</dt><dd>{heldOutSummary(model.provenance) ?? NOT_RECORDED}</dd>
        </dl>

        <section className="dialog-section" aria-labelledby={`${id}-hist`}>
          <h3 id={`${id}-hist`}>Historical pre-protocol metrics</h3>
          <p className="small muted">
            Recorded when this model was trained, on its own pre-protocol test split. They are not comparable across
            models, and a higher historical score is not evidence that this model is better than the active one.
          </p>
          <dl className="kv kv-compact">
            <dt>Balanced accuracy</dt><dd>{m.balanced_accuracy == null ? NA : pct(m.balanced_accuracy)}</dd>
            <dt>Macro-F1</dt><dd>{m.f1_macro == null ? NA : num(m.f1_macro)}</dd>
            <dt>ROC-AUC</dt><dd>{m.roc_auc == null ? NA : num(m.roc_auc)}</dd>
          </dl>
          {evaluationSummary(model.evaluation) ? (
            <p className="small">{evaluationSummary(model.evaluation)}. Exploratory, development participants only; not an
              independent test result.</p>
          ) : null}
        </section>

        <div className="field">
          <label htmlFor={`${id}-confirm`}>Type the version to confirm</label>
          <input id={`${id}-confirm`} ref={confirmRef} value={confirm} onChange={(e) => setConfirm(e.target.value)}
            autoComplete="off" autoCapitalize="off" spellCheck={false} className="mono-input"
            aria-invalid={errors.confirm ? 'true' : undefined} aria-describedby={describe('confirm', `${id}-confirm-hint`)} />
          <span id={`${id}-confirm-hint`} className="field-hint">Must match <span className="mono">{model.version}</span> exactly.</span>
          {errors.confirm ? <span id={`${id}-confirm-error`} className="form-error field-error">{errors.confirm}</span> : null}
        </div>

        <div className="field">
          <label htmlFor={`${id}-reason`}>Reason for the audit log</label>
          <textarea id={`${id}-reason`} ref={reasonRef} value={reason} onChange={(e) => setReason(e.target.value)} rows={3}
            autoComplete="off" aria-invalid={errors.reason ? 'true' : undefined}
            aria-describedby={describe('reason', `${id}-reason-hint ${id}-reason-count`)} />
          <span id={`${id}-reason-hint`} className="field-hint">
            {REASON_MIN}–{REASON_MAX} characters. Stored in the audit log; never include passwords, tokens or other secrets.
          </span>
          <span id={`${id}-reason-count`} className={`field-hint${trimmed > REASON_MAX ? ' over-limit' : ''}`}>
            {trimmed} / {REASON_MAX} characters
          </span>
          {errors.reason ? <span id={`${id}-reason-error`} className="form-error field-error">{errors.reason}</span> : null}
        </div>

        {failure ? <Notice tone="error" title={failure.title}>{failure.body}</Notice> : null}
        <p className="visually-hidden" role="status">{busy ? 'Activating model…' : ''}</p>

        <div className="dialog-actions">
          <button type="button" className="btn btn-secondary" onClick={() => { if (!busyRef.current) onCancel(); }}
            aria-disabled={busy || undefined}>Cancel</button>
          <button type="submit" className="btn btn-primary" aria-disabled={busy || undefined}>
            {busy ? 'Activating…' : 'Activate model'}
          </button>
        </div>
      </form>
    </dialog>
  );
}
