<script>
  import { onMount, onDestroy } from 'svelte'
  import { fetchVramProfiles } from './api.js'

  let rows  = $state([])
  let timer = null

  const shortId = id => (id ?? '—').replace(/-int4-ov|-int8-ov|-fp16-ov|-int4|-int8/g, '')
  const fmt2    = v  => v == null ? '—' : Number(v).toFixed(2)
  const fmt1    = v  => v == null ? '—' : Number(v).toFixed(1)

  onMount(() => { load(); timer = setInterval(load, 60_000) })
  onDestroy(() => clearInterval(timer))

  async function load() {
    try { rows = await fetchVramProfiles() }
    catch { /* keep stale */ }
  }

  // Group rows by model_id for display
  const grouped = $derived(
    rows.reduce((acc, r) => {
      ;(acc[r.model_id] ??= []).push(r)
      return acc
    }, {})
  )
</script>

<section class="panel">
  <h2>VRAM profiles</h2>

  {#if !rows.length}
    <p class="dim">no measurements yet — run the profiler</p>
  {:else}
    <table>
      <thead>
        <tr>
          <th>Model</th>
          <th class="right">KV GB</th>
          <th class="right">VRAM GB</th>
          <th class="right">Load s</th>
        </tr>
      </thead>
      <tbody>
        {#each Object.entries(grouped) as [modelId, entries]}
          {#each entries as row, i}
            <tr>
              <td class="name">{i === 0 ? shortId(modelId) : ''}</td>
              <td class="right dim">{fmt2(row.kv_cache_gb)}</td>
              <td class="right hl">{fmt2(row.vram_gb)}</td>
              <td class="right dim">{fmt1(row.load_time_s)}</td>
            </tr>
          {/each}
        {/each}
      </tbody>
    </table>
  {/if}
</section>

<style>
  .panel { padding: .75rem 1rem; }
  h2 {
    font-size: .7rem; text-transform: uppercase; letter-spacing: .08em;
    opacity: .45; margin-bottom: .5rem;
  }
  p.dim { font-size: .8rem; opacity: .4; }

  table { width: 100%; border-collapse: collapse; font-size: .78rem; }
  th {
    font-weight: 400; opacity: .4; padding-bottom: .3rem;
    font-size: .68rem; text-transform: uppercase; letter-spacing: .05em;
  }
  td { padding: .18rem 0; vertical-align: middle; }
  tr:not(:last-child) td { border-bottom: 1px solid #ffffff07; }

  .name { font-family: monospace; font-size: .75rem; }
  .right { text-align: right; padding-left: .5rem; font-variant-numeric: tabular-nums; }
  .hl  { color: var(--blue); }
  .dim { opacity: .4; }
</style>
