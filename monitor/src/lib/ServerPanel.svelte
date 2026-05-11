<script>
  let { health } = $props()
  const fmt = (n, d=1) => n == null ? '—' : Number(n).toFixed(d)

  const busy    = $derived(health?.busy ?? false)
  const busySec = $derived(health?.busy_for_sec ?? 0)
  const rd      = $derived(health?.last_routing_decision ?? null)
  const scope   = $derived(health?.provider_scope ?? 'local')

  let rdExpanded = $state(false)
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
          <tr class="route-row" onclick={() => rdExpanded = !rdExpanded} title="Click to expand routing detail">
            <td>Last route</td>
            <td class="mono route">
              {rd.task_class ?? '—'} → {rd.model ?? '—'}
              <span class="dim">[{rd.strategy ?? ''}{rd.confidence != null ? ` ${(rd.confidence*100).toFixed(0)}%` : ''}{rd.latency_ms != null ? ` ${rd.latency_ms}ms` : ''}]</span>
              <span class="expand-caret">{rdExpanded ? '▲' : '▼'}</span>
            </td>
          </tr>
          {#if rdExpanded}
            <tr class="route-detail">
              <td colspan="2">
                <table class="rd-table">
                  <tbody>
                    <tr><td>Task class</td><td>{rd.task_class ?? '—'}</td></tr>
                    <tr><td>Strategy</td><td class:hl-strategy={true}>{rd.strategy ?? '—'}</td></tr>
                    <tr><td>Confidence</td><td>{rd.confidence != null ? `${(rd.confidence*100).toFixed(1)}%` : '—'}</td></tr>
                    <tr><td>Latency</td><td>{rd.latency_ms != null ? `${rd.latency_ms} ms` : '—'}</td></tr>
                    <tr><td>Model</td><td class="mono-sm">{rd.model ?? '—'}</td></tr>
                    {#if rd.cloud_directive}
                      <tr><td>Directive</td><td class="hl-cloud">#cloud override</td></tr>
                    {/if}
                  </tbody>
                </table>
              </td>
            </tr>
          {/if}
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
  .mono     { font-family: monospace; font-size: .78rem; word-break: break-all; }
  .mono-sm  { font-family: monospace; font-size: .72rem; }
  .route    { font-size: .75rem; }

  .route-row { cursor: pointer; }
  .route-row:hover td { opacity: .85; }
  .expand-caret { margin-left: .3rem; opacity: .4; font-size: .65rem; }

  .route-detail td { padding-top: 0; padding-bottom: .4rem; }
  .rd-table { width: 100%; border-collapse: collapse; font-size: .75rem; }
  .rd-table td { padding: .1rem 0; }
  .rd-table td:first-child { opacity: .45; width: 6rem; }
  .rd-table tr:first-child td { padding-top: .25rem; }
  .hl-cloud { color: var(--yellow); }
</style>
