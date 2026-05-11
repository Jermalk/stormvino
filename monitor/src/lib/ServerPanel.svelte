<script>
  let { health } = $props()
  const fmt = (n, d=1) => n == null ? '—' : Number(n).toFixed(d)

  const busy    = $derived(health?.busy ?? false)
  const busySec = $derived(health?.busy_for_sec ?? 0)
  const rd      = $derived(health?.last_routing_decision ?? null)
  const scope   = $derived(health?.provider_scope ?? 'local')
</script>

<section class="panel">
  <h2>Server</h2>
  {#if !health}
    <p class="offline">OFFLINE</p>
  {:else}
    <table>
      <tbody>
        <tr>
          <td>Status</td>
          <td>
            {#if busy}
              <span class="busy">BUSY {busySec.toFixed(0)}s</span>
            {:else}
              <span class="online">ONLINE</span>
            {/if}
          </td>
        </tr>
        <tr><td>Active reqs</td><td>{health.active_requests ?? 0}</td></tr>

        <tr class="spacer"><td colspan="2"></td></tr>

        <tr>
          <td>LLM</td>
          <td class="mono">
            {#if (health.loaded_models ?? []).length}
              {health.loaded_models.join(', ')}
            {:else}<span class="dim">none</span>{/if}
          </td>
        </tr>
        <tr>
          <td>VLM</td>
          <td class="mono">
            {#if (health.loaded_vlm_models ?? []).length}
              {health.loaded_vlm_models.join(', ')}
            {:else}<span class="dim">none</span>{/if}
          </td>
        </tr>
        <tr>
          <td>KV cache</td>
          <td class="hl">{fmt(health.kv_cache_size_gb)} GB</td>
        </tr>
        <tr>
          <td>Embeddings</td>
          <td>{health.embedding_loaded ? '✓' : '—'}</td>
        </tr>
        <tr>
          <td>Image model</td>
          <td class="mono">
            {#if health.image_model_loaded}{health.image_model_id}{:else}<span class="dim">—</span>{/if}
          </td>
        </tr>
        <tr>
          <td>STT</td>
          <td>{health.stt_model_loaded ? '✓' : '—'}</td>
        </tr>
        <tr>
          <td>Scope</td>
          <td class:hl={scope !== 'local'}>{scope}</td>
        </tr>

        <tr class="spacer"><td colspan="2"></td></tr>

        {#if rd}
          <tr>
            <td>Last route</td>
            <td class="mono route">
              {rd.task_class ?? '—'} → {rd.model ?? '—'}
              <span class="dim">[{rd.strategy ?? ''}{rd.confidence != null ? ` {(rd.confidence*100).toFixed(0)}%` : ''}{rd.latency_ms != null ? ` {rd.latency_ms}ms` : ''}]</span>
            </td>
          </tr>
        {/if}

        {#if health.last_model}
          <tr><td>Last model</td><td class="mono">{health.last_model}</td></tr>
          <tr><td>Throughput</td><td>{fmt(health.last_tok_per_sec)} tok/s · {health.last_tokens ?? 0} tok · {fmt(health.last_elapsed_sec)}s</td></tr>
          <tr><td>Last req</td><td class="dim">{health.last_request_at ?? ''}</td></tr>
        {/if}

        <tr class="spacer"><td colspan="2"></td></tr>

        <tr><td>Total reqs</td><td>{(health.total_requests ?? 0).toLocaleString()}</td></tr>
        <tr><td>Total tokens</td><td>{(health.total_tokens ?? 0).toLocaleString()}</td></tr>
      </tbody>
    </table>
  {/if}
</section>

<style>
  .panel { padding: .75rem 1rem; height: 100%; }
  h2 { font-size: .7rem; text-transform: uppercase; letter-spacing: .08em; opacity: .45; margin-bottom: .6rem; }
  table { width: 100%; border-collapse: collapse; font-size: .82rem; }
  td { padding: .15rem 0; vertical-align: top; }
  td:first-child { color: var(--text); opacity: .5; width: 7rem; flex-shrink: 0; white-space: nowrap; padding-right: .5rem; }
  .spacer td { height: .4rem; }
  .online { color: var(--green); font-weight: 700; }
  .busy   { color: var(--yellow); font-weight: 700; }
  .offline { color: var(--red); font-weight: 700; }
  .dim    { opacity: .35; }
  .hl     { color: var(--blue); }
  .mono   { font-family: monospace; font-size: .78rem; word-break: break-all; }
  .route  { font-size: .75rem; }
</style>
