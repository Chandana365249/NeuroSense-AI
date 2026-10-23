const BASE = '/api';

export class ApiError extends Error {
  constructor(status, message, { code = null, details = null } = {}) {
    super(message);
    this.status = status;
    this.code = code;
    // Structured field errors ([{loc, msg, type}]) are only kept for request-validation failures.
    this.details = code === 'validation_error' && Array.isArray(details) ? details : null;
  }
}

async function request(path, { token, method = 'GET', json, form } = {}) {
  const headers = {};
  let body;
  if (token) headers.Authorization = `Bearer ${token}`;
  if (json !== undefined) {
    headers['Content-Type'] = 'application/json';
    body = JSON.stringify(json);
  } else if (form) {
    body = new URLSearchParams(form);
  }
  let response;
  try {
    response = await fetch(`${BASE}${path}`, { method, headers, body });
  } catch {
    throw new ApiError(0, 'Backend unreachable');
  }
  const data = await response.json().catch(() => null);
  if (!response.ok) {
    throw new ApiError(response.status, data?.error?.message || `Request failed (${response.status})`,
      { code: data?.error?.code, details: data?.error?.details });
  }
  return data;
}

export const api = {
  health: () => request('/health'),
  register: (email, password, fullName) =>
    request('/auth/register', { method: 'POST', json: { email, password, full_name: fullName || null } }),
  login: (email, password) => request('/auth/login', { method: 'POST', form: { username: email, password } }),
  me: (token) => request('/auth/me', { token }),
  modelInfo: () => request('/model-info'),
  metrics: () => request('/metrics'),
  trainingStatus: (token) => request('/training/status', { token }),
  train: (token, runCrossValidation) =>
    request('/train', { token, method: 'POST', json: { run_cross_validation: runCrossValidation } }),
  demoTrials: () => request('/dataset/demo-trials'),
  datasetSummary: () => request('/dataset/summary'),
  models: () => request('/models'),
  explainGlobal: (top = 15) => request(`/explain/global?top=${encodeURIComponent(top)}`),
  // The confirmation and the audit reason travel only in the request body (never in the URL).
  activateModel: (token, version, { confirmVersion, reason }) =>
    request(`/models/${encodeURIComponent(version)}/activate`, {
      token, method: 'POST', json: { confirm_version: confirmVersion, reason },
    }),
  adminDiagnostics: (token) => request('/admin/diagnostics', { token }),
  predict: (token, payload) => request('/predict', { token, method: 'POST', json: payload }),
  predictions: (token, { limit = 20, offset = 0 } = {}) =>
    request(`/predictions?limit=${limit}&offset=${offset}`, { token }),
  prediction: (token, id) => request(`/predictions/${encodeURIComponent(id)}`, { token }),
};
