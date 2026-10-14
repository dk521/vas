const API_BASE = import.meta.env.VITE_API_URL || '';

async function request(path, { token, body, isForm = false, method = 'POST' } = {}) {
  const response = await fetch(`${API_BASE}${path}`, {
    method,
    headers: {
      ...(token ? { Authorization: `Bearer ${token}` } : {}),
      ...(!isForm && body ? { 'Content-Type': 'application/json' } : {}),
    },
    body: body ? (isForm ? body : JSON.stringify(body)) : undefined,
  });

  const data = await response.json().catch(() => ({ detail: 'The server returned an invalid response.' }));
  if (!response.ok) {
    const message = typeof data === 'object'
      ? Object.values(data).flat().join(' ') || data.detail
      : String(data);
    throw new Error(message || `Request failed (${response.status})`);
  }
  return data;
}

export const api = {
  register: (body) => request('/api/accounts/register/', { body }),
  login: (body) => request('/api/accounts/login/', { body }),
  enrollmentStatus: (token) => request('/api/voice/enrollment-status/', { token, method: 'GET' }),
  enroll: (audioFile, token, { promptType, pangramId } = {}) => {
    const body = new FormData();
    body.append('audio_file', audioFile, audioFile.name);
    body.append('prompt_type', promptType);
    if (pangramId != null) body.append('pangram_id', pangramId);
    return request('/api/voice/enroll/', { token, body, isForm: true });
  },
  pangrams: (token, count = 1) => request(`/api/voice/pangrams/?count=${count}`, { token, method: 'GET' }),
  authenticate: (audioFile, token) => {
    const body = new FormData();
    body.append('audio_file', audioFile, audioFile.name);
    return request('/api/voice/authenticate/', { token, body, isForm: true });
  },
};
