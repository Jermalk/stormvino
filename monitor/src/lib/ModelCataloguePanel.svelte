<script>
  import { onMount, onDestroy } from 'svelte'
  import { fetchModels, fetchVramProfiles, triggerProfiling } from './api.js'

  let { profiler = null } = $props()

  let models     = $state([])
  let profiles   = $state([])
  let timer      = null
  let triggering = $state(false)

  const shortId = id => (id ?? '—').replace(/-int4-ov|-int8-ov|-fp16-ov|-int4|-int8/g, '')
  const fmt2    = v  => v == null ? '—' : Number(v).toFixed(2)

  onMount(() => { load(); timer = setInterval(load, 30_000) })
  onDestroy(() => clearInterval(timer))

  async function load() {
    try { [models, profiles] = await Promise.all([fetchModels(), fetchVramProfiles()]) }
    catch { /* keep stale */ }
  }

  // model_id → {kv_cache_gb, vram_gb, load_time_s}
  const profileMap = $derived(Object.fromEntries(profiles.map(p => [p.model_id, p])))

  const measured   = $derived(profiler?.vram_measured ?? {})
  const running    = $derived(profiler?.running       ?? false)
  const current    = $derived(profiler?.current       ?? null)
  const unmeasured = $derived([
    ...(profiler?.unmeasured_llms ?? []),
    ...(profiler?.unmeasured_vlms ?? []),
  ])

  const localModels = $derived(models.filter(m => m.provider === 'loc' && m.id !== 'Auto'))
  const cloudModels = $derived(models.filter(m => m.provider !== 'loc' && m.id !== 'Auto'))

  async function trigger() {
    if (triggering || running) return
    triggering = true
    await triggerProfiling().catch(() => {})
    await new Promise(r => setTimeout(r, 800))
    triggering = false
  }
</script>

<section class="panel">
  <div class="toolbar">
    <h2>Model Catalogue</h2>

    {#if profiler}
      <div class="profiler">
        {#if running}
          <span class="badge running">RUNNING</span>
          {#if current}<span class="current">→ {shortId(current)}</span>{/if}
        {:else if unmeasured.length}
          <span class="badge warn">{unmeasured.length} unmeasured</span>
        {:else}
          <span class="badge ok">IDLE · all measured</span>
        {/if}
        <button class="run-btn" onclick={trigger} disabled={triggering || running}>
          {running || triggering ? '…' : '▷ profile'}
        </button>
      </div>
    {/if}
  </div>

  {#if !models.length}
    <p class="dim">loading…</p>
  {:else}
    <table>
      <thead>
        <tr>
          <th>Model</th>
          <th class="center">Tier</th>
          <th class="center">Status</th>
          <th class="right">VRAM GB</th>
          <th class="right">KV GB</th>
          <th class="right">Load s</th>
        </tr>
      </thead>
      <tbody>
        {#if localModels.length}
          <tr class="group-header"><td colspan="6">Local · GPU.1</td></tr>
          {#each localModels as m}
            {@const p    = profileMap[m.id]}
            {@const vram = p?.vram_gb     ?? measured[m.id] ?? null}
            {@const kv   = p?.kv_cache_gb ?? null}
            {@const ls   = p?.load_time_s ?? null}
            {@const busy = running && current === m.id}
            <tr class:loaded={m.loaded}>
              <td class="name">{shortId(m.id)}</td>
              <td class="center"><span class="badge tier-{m.tier ?? 'fast'}">{m.tier ?? '—'}</span></td>
              <td class="center">
                {#if busy}
                  <span class="dot measuring" title="profiling…">◌</span>
                {:else if m.loaded}
                  <span class="dot loaded-dot" title="in VRAM">●</span>
                {:else}
                  <span class="dot idle-dot">○</span>
                {/if}
              </td>
              <td class="right" class:hl={vram != null} class:dim={vram == null}>{fmt2(vram)}</td>
              <td class="right dim">{kv != null ? fmt2(kv) : '—'}</td>
              <td class="right dim">{ls != null ? Number(ls).toFixed(1) : '—'}</td>
            </tr>
          {/each}
        {/if}

        {#if cloudModels.length}
          <tr class="group-header"><td colspan="6">Cloud · OVH</td></tr>
          {#each cloudModels as m}
            <tr>
              <td class="name">{m.id}</td>
              <td class="center"><span class="badge tier-{m.tier ?? 'auto'}">{m.tier ?? '—'}</span></td>
              <td class="center dim">—</td>
              <td class="right dim">cloud</td>
              <td class="right dim">—</td>
              <td class="right dim">—</td>
            </tr>
          {/each}
        {/if}
      </tbody>
    </table>
  {/if}
</section>

<style>
  .panel { padding: .75rem 1rem; }

  .toolbar {
    display: flex; align-items: center; gap: .75rem;
    margin-bottom: .6rem; flex-wrap: wrap;
  }
  h2 {
    font-size: .7rem; text-transform: uppercase; letter-spacing: .08em;
    opacity: .45; white-space: nowrap;
  }

  .profiler {
    display: flex; align-items: center; gap: .5rem;
    margin-left: auto; flex-wrap: wrap;
  }
  .current { font-family: monospace; font-size: .78rem; opacity: .8; }

  .run-btn {
    background: transparent; border: 1px solid #ffffff18;
    border-radius: 4px; padding: .15rem .6rem;
    font-size: .75rem; cursor: pointer; color: inherit; opacity: .6;
    transition: opacity .15s, border-color .15s, color .15s;
  }
  .run-btn:hover:not(:disabled) { opacity: 1; border-color: var(--blue); color: var(--blue); }
  .run-btn:disabled { cursor: not-allowed; opacity: .3; }

  p.dim { font-size: .8rem; opacity: .4; }

  table { width: 100%; border-collapse: collapse; font-size: .8rem; }
  th {
    font-weight: 400; opacity: .4; padding-bottom: .35rem;
    font-size: .68rem; text-transform: uppercase; letter-spacing: .05em;
    border-bottom: 1px solid #ffffff0a;
  }
  td { padding: .25rem 0; vertical-align: middle; }
  tr:not(.group-header):not(:last-child) td { border-bottom: 1px solid #ffffff07; }

  .group-header td {
    font-size: .65rem; text-transform: uppercase; letter-spacing: .08em;
    opacity: .3; padding-top: .6rem; padding-bottom: .2rem;
  }

  .name { font-family: monospace; font-size: .78rem; }
  tr.loaded .name { color: var(--green); }

  .center { text-align: center; }
  .right  { text-align: right; padding-left: .75rem; font-variant-numeric: tabular-nums; }
  .hl     { color: var(--blue); }
  .dim    { opacity: .35; }

  .dot { font-size: .72rem; }
  .loaded-dot  { color: var(--green); }
  .idle-dot    { opacity: .2; }
  .measuring   { color: var(--yellow); animation: spin 1.2s linear infinite; display: inline-block; }
  @keyframes spin { to { transform: rotate(360deg); } }

  .badge {
    display: inline-block; font-size: .65rem; padding: .1rem .35rem;
    border-radius: 3px; font-weight: 600; letter-spacing: .04em; text-transform: uppercase;
  }
  .running  { background: var(--yellow)33; color: var(--yellow); }
  .warn     { background: var(--yellow)22; color: var(--yellow); }
  .ok       { background: var(--green)18;  color: var(--green);  opacity: .7; }

  .tier-fast     { background: #4ef1a018; color: var(--green); }
  .tier-balanced { background: #4e9af118; color: var(--blue); }
  .tier-best     { background: #9b6ef318; color: var(--purple); }
  .tier-auto     { background: #ffffff10; color: var(--text); }
</style>
