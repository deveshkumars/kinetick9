const API_BASE = '/api';

export async function getHealth() {
  const res = await fetch(`${API_BASE}/health`);
  if (!res.ok) throw new Error('Health check failed');
  return res.json();
}

export async function getDogState() {
  const res = await fetch(`${API_BASE}/dog/state`);
  if (!res.ok) throw new Error('Failed to fetch dog state');
  return res.json();
}

export async function walkTo(x, y) {
  const res = await fetch(`${API_BASE}/dog/walk`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ x, y }),
  });
  if (!res.ok) {
    const data = await res.json().catch(() => ({}));
    throw new Error(data.error || 'Failed to send walk command');
  }
  return res.json();
}

/**
 * Text-to-speech via ElevenLabs (backend proxy). Returns a promise that resolves when playback starts.
 * Rejects if TTS is not configured or the request fails.
 */
/**
 * Transcribe audio via ElevenLabs Speech-to-Text. Returns { text }.
 */
export async function transcribe(audioBlob) {
  const form = new FormData()
  form.append('file', audioBlob, 'audio.webm')
  const res = await fetch(`${API_BASE}/stt`, { method: 'POST', body: form })
  if (!res.ok) {
    const d = await res.json().catch(() => ({}))
    throw new Error(d.error || `Transcription failed (${res.status})`)
  }
  return res.json()
}

/**
 * Convert a video blob (e.g. from recording) to another format. Returns the converted blob.
 * Backend must have ffmpeg for mp4. format: 'mp4' | 'webm'
 */
export async function convertVideoToFormat(videoBlob, format) {
  const form = new FormData()
  form.append('file', videoBlob)
  const res = await fetch(`${API_BASE}/video/convert?format=${encodeURIComponent(format)}`, {
    method: 'POST',
    body: form,
  })
  if (!res.ok) {
    const d = await res.json().catch(() => ({}))
    throw new Error(d.error || `Conversion failed (${res.status})`)
  }
  return res.blob()
}

/**
 * Run the quadruped simulation walking forward or backward and return the rendered MP4 video as a Blob
 * along with the robot's final position.
 * Uses vector geometry: the robot's heading quaternion determines a waypoint along its facing direction.
 * @param {'forward'|'backward'} direction
 * @param {{ distance?: number, nSteps?: number, renderEvery?: number }} opts
 * @returns {Promise<{ blob: Blob, finalX: number, finalY: number }>}
 */
export async function simulateDirectionWalkVideo(direction, { distance, nSteps, renderEvery } = {}) {
  const body = { direction }
  if (distance != null) body.distance = distance
  if (nSteps != null) body.n_steps = nSteps
  if (renderEvery != null) body.render_every = renderEvery
  const res = await fetch(`${API_BASE}/dog/walk/direction`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  })
  if (!res.ok) {
    const d = await res.json().catch(() => ({}))
    throw new Error(d.error || `Direction walk simulation failed (${res.status})`)
  }
  const finalX = parseFloat(res.headers.get('X-Final-X')) || 0
  const finalY = parseFloat(res.headers.get('X-Final-Y')) || 0
  const blob = await res.blob()
  return { blob, finalX, finalY }
}

/**
 * Run the quadruped simulation to chase the red ball and return the video blob + final position.
 * @param {{ ballX?: number, ballY?: number, nSteps?: number, renderEvery?: number }} opts
 * @returns {Promise<{ blob: Blob, finalX: number, finalY: number }>}
 */
export async function simulateBallWalkVideo({ ballX, ballY, nSteps, renderEvery } = {}) {
  const body = {}
  if (ballX != null) body.ball_x = ballX
  if (ballY != null) body.ball_y = ballY
  if (nSteps != null) body.n_steps = nSteps
  if (renderEvery != null) body.render_every = renderEvery
  const res = await fetch(`${API_BASE}/dog/walk/ball`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  })
  if (!res.ok) {
    const d = await res.json().catch(() => ({}))
    throw new Error(d.error || `Ball walk simulation failed (${res.status})`)
  }
  const finalX = parseFloat(res.headers.get('X-Final-X')) || 0
  const finalY = parseFloat(res.headers.get('X-Final-Y')) || 0
  const blob = await res.blob()
  return { blob, finalX, finalY }
}

/**
 * Run the quadruped simulation to walk toward (x, y) and return the rendered MP4 video as a Blob.
 * The simulation can take 30-60+ seconds on the first call (JAX JIT compilation).
 */
export async function simulateWalkVideo(x, y, { nSteps, renderEvery } = {}) {
  const body = { x, y }
  if (nSteps != null) body.n_steps = nSteps
  if (renderEvery != null) body.render_every = renderEvery
  const res = await fetch(`${API_BASE}/dog/walk/video`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  })
  if (!res.ok) {
    const d = await res.json().catch(() => ({}))
    throw new Error(d.error || `Simulation failed (${res.status})`)
  }
  return res.blob()
}

/** Fetch TTS audio as a blob (no playback). Use for mixing into video. */
export async function getTtsBlob(text) {
  const res = await fetch(`${API_BASE}/tts`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ text }),
  })
  if (!res.ok) {
    const d = await res.json().catch(() => ({}))
    throw new Error(d.error || d.detail || 'TTS failed')
  }
  return res.blob()
}

export function speak(text) {
  return fetch(`${API_BASE}/tts`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ text }),
  }).then(async (res) => {
    if (!res.ok) {
      const d = await res.json().catch(() => ({}))
      const raw = d.detail ?? d.error ?? `TTS failed (${res.status})`
      const msg = typeof raw === 'string' ? raw : (raw?.message ?? JSON.stringify(raw))
      return Promise.reject(new Error(msg))
    }
    return res.blob()
  }).then((blob) => {
    const url = URL.createObjectURL(blob)
    const audio = new Audio(url)
    const cleanup = () => URL.revokeObjectURL(url)
    audio.addEventListener('ended', cleanup)
    audio.addEventListener('error', cleanup)
    return audio.play().then(() => {}).catch((playErr) => {
      cleanup()
      if (playErr.name === 'NotAllowedError') {
        return Promise.reject(new Error('Browser blocked sound. Click the speaker icon to play.'))
      }
      return Promise.reject(playErr)
    })
  })
}
