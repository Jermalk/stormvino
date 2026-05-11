const BASE = ''

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

export async function fetchProfilerStatus() {
  const r = await fetch(`${BASE}/admin/profile-models`)
  if (!r.ok) throw new Error(`/admin/profile-models ${r.status}`)
  return r.json()
}

export async function triggerProfiling() {
  await fetch(`${BASE}/admin/profile-models`, { method: 'POST' })
}
