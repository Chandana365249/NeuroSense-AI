const BASE = '/api';

export class ApiError extends Error {
  constructor(status, message) {
    super(message);
    this.status = status;
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
    throw new ApiError(response.status, data?.error?.message || `Request failed (${response.status})`);
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
  trainingStatus: () => request('/training/status'),
  train: (token, runCrossValidation) =>
    request('/train', { token, method: 'POST', json: { run_cross_validation: runCrossValidation } }),
  demoTrials: () => request('/dataset/demo-trials'),
  predict: (token, payload) => request('/predict', { token, method: 'POST', json: payload }),
  predictions: (token) => request('/predictions?limit=20', { token }),
};
