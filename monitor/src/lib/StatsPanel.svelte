<script>
  /** @type {{ health: import('./api.js').Health | null }} */
  let { health } = $props()

  function fmt(n, dec = 1) { return n == null ? '—' : n.toFixed(dec) }
  function pct(used, total) { return total ? ((used / total) * 100).toFixed(1) : '—' }
</script>

<section class="stats-panel">
  <h2>Live</h2>

  {#if health}
    <div class="grid">
      <div class="stat">
        <span class="label">Status</span>
        <span class="value" class:ok={health.status === 'ok'}>{health.status}</span>
      </div>
      <div class="stat">
        <span class="label">Active requests</span>
        <span class="value">{health.active_requests ?? 0}</span>
      </div>
      <div class="stat">
        <span class="label">Last model</span>
        <span class="value mono">{health.last_model || '—'}</span>
      </div>
      <div class="stat">
        <span class="label">tok/s</span>
        <span class="value">{fmt(health.last_tok_per_sec)}</span>
      </div>
      <div class="stat">
        <span class="label">Last elapsed</span>
        <span class="value">{fmt(health.last_elapsed_sec)}s</span>
      </div>
      <div class="stat">
        <span class="label">RAM used</span>
        <span class="value">{fmt(health.ram_used_pct, 0)}%</span>
      </div>
      <div class="stat">
        <span class="label">Total requests</span>
        <span class="value">{health.total_requests ?? 0}</span>
      </div>
      <div class="stat">
        <span class="label">Total tokens</span>
        <span class="value">{(health.total_tokens ?? 0).toLocaleString()}</span>
      </div>
    </div>
  {:else}
    <p class="offline">Server unreachable</p>
  {/if}
</section>

<style>
  .stats-panel { padding: 1rem; }
  h2 { margin: 0 0 .75rem; font-size: 1rem; text-transform: uppercase; letter-spacing: .08em; opacity: .6; }
  .grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(160px, 1fr)); gap: .5rem; }
  .stat { background: var(--card); border-radius: 6px; padding: .6rem .8rem; }
  .label { display: block; font-size: .7rem; opacity: .55; margin-bottom: .2rem; }
  .value { font-size: 1.1rem; font-weight: 600; }
  .value.ok { color: var(--green); }
  .mono { font-family: monospace; font-size: .9rem; }
  .offline { color: var(--red); }
</style>
