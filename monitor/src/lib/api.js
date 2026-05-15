const BASE = ''

// Sidecar runs on :11436 — independent of ov_server, survives restarts.
// Use window.location.hostname so the browser reaches the server's sidecar,
// not localhost on the user's own machine.
const SIDECAR_BASE =
  typeof window !== 'undefined'
    ? `${window.location.protocol}//${window.location.hostname}:11436`
    : 'http://localhost:11436'

export async function fetchSidecar() {
  const r = await fetch(`${SIDECAR_BASE}/metrics`)
  if (!r.ok) throw new Error(`sidecar /metrics ${r.status}`)
  return r.json()
}

export async function fetchHealth() {
  const r = await fetch(`${BASE}/health`)
  if (!r.ok) throw new Error(`/health ${r.status}`)
  return r.json()
}

export async function fetchSystem() {
  const r = await fetch(`${BASE}/monitor/api/system`)
  if (!r.ok) throw new Error(`/monitor/api/system ${r.status}`)
  return r.json()
}

export async function fetchMetrics(metric, minutes = 60) {
  const r = await fetch(`${BASE}/monitor/api/metrics?metric=${metric}&minutes=${minutes}`)
  if (!r.ok) throw new Error(`/monitor/api/metrics ${r.status}`)
  return r.json()
}

export async function switchProfile(name) {
  await fetch(`${BASE}/admin/profile`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ profile: name }),
  })
}

export async function switchScope(scope) {
  await fetch(`${BASE}/admin/scope`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ scope }),
  })
}

export async function fetchModelUsage(hours = 24) {
  const r = await fetch(`${BASE}/monitor/api/model-usage?hours=${hours}`)
  if (!r.ok) throw new Error(`/monitor/api/model-usage ${r.status}`)
  return r.json()
}

export async function fetchProfilerStatus() {
  const r = await fetch(`${BASE}/admin/profile-models`)
  if (!r.ok) throw new Error(`/admin/profile-models ${r.status}`)
  return r.json()
}

export async function triggerProfiling() {
  await fetch(`${BASE}/admin/profile-models`, { method: 'POST' })
}

export async function fetchModels() {
  const r = await fetch(`${BASE}/v1/models`)
  if (!r.ok) throw new Error(`/v1/models ${r.status}`)
  const data = await r.json()
  return data.data ?? []
}

export async function fetchVramProfiles() {
  const r = await fetch(`${BASE}/monitor/api/vram-profiles`)
  if (!r.ok) throw new Error(`/monitor/api/vram-profiles ${r.status}`)
  return r.json()
}

export async function fetchAvailableModels() {
  const r = await fetch(`${BASE}/v1/models`)
  if (!r.ok) throw new Error(`/v1/models ${r.status}`)
  const data = await r.json()
  // Strip the Auto routing entry — the dropdown has its own AUTO option
  return (data.data ?? []).filter(m => m.id !== 'Auto')
}

export async function loadModel(modelId) {
  await fetch(`${BASE}/admin/load-model`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ model_id: modelId }),
  })
}
