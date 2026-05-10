/**
 * API client — all fetch calls go here.
 * /health        — live server state (polled every 2s)
 * /monitor/api/metrics — historical time-series from Postgres
 */

const BASE = ''   // same origin; vite proxy handles dev

export async function fetchHealth() {
  const r = await fetch(`${BASE}/health`)
  if (!r.ok) throw new Error(`/health ${r.status}`)
  return r.json()
}

/**
 * Fetch aggregated metric rows from Postgres.
 * @param {string} metric  - 'tok_per_sec' | 'elapsed_sec' | 'vram_free_gb'
 * @param {number} minutes - lookback window
 * @returns {Promise<{ts: number[], values: number[]}>}
 */
export async function fetchMetrics(metric, minutes = 60) {
  const r = await fetch(`${BASE}/monitor/api/metrics?metric=${metric}&minutes=${minutes}`)
  if (!r.ok) throw new Error(`/monitor/api/metrics ${r.status}`)
  return r.json()
}

/**
 * Fetch per-model usage summary from Postgres.
 * @param {number} hours
 * @returns {Promise<Array<{model_id, requests, avg_tok_per_sec, total_tokens}>>}
 */
export async function fetchModelUsage(hours = 24) {
  const r = await fetch(`${BASE}/monitor/api/model-usage?hours=${hours}`)
  if (!r.ok) throw new Error(`/monitor/api/model-usage ${r.status}`)
  return r.json()
}
