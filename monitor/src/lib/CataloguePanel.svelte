<script>
  import { onMount, onDestroy } from 'svelte'
  import { fetchModels } from './api.js'

  let { profiler = null } = $props()

  let models = $state([])
  let timer  = null

  const shortId = id => (id ?? '—').replace(/-int4-ov|-int8-ov|-fp16-ov|-int4|-int8/g, '')
  const fmtVram = v => v == null ? '' : `${Number(v).toFixed(1)} GB`

  onMount(() => { load(); timer = setInterval(load, 30_000) })
  onDestroy(() => clearInterval(timer))

  async function load() {
    try { models = await fetchModels() }
    catch { /* keep stale */ }
  }

  const vramMeasured = $derived(profiler?.vram_measured ?? {})

  // split into local vs cloud, skip the Auto pseudo-model
  const localModels = $derived(models.filter(m => m.provider === 'loc' && m.id !== 'Auto'))
  const cloudModels = $derived(models.filter(m => m.provider !== 'loc' && m.id !== 'Auto'))
</script>

<section class="panel">
  <h2>Model catalogue</h2>

  {#if !models.length}
    <p class="dim">loading…</p>
  {:else}
    <table>
      <thead>
        <tr>
          <th>Model</th>
          <th class="center">Tier</th>
          <th class="center">Status</th>
          <th class="right">VRAM</th>
        </tr>
      </thead>
      <tbody>
        {#if localModels.length}
          <tr class="group-header"><td colspan="4">Local · GPU.1</td></tr>
          {#each localModels as m}
            <tr class:loaded={m.loaded}>
              <td class="name">{shortId(m.id)}</td>
              <td class="center"><span class="badge tier-{m.tier ?? 'fast'}">{m.tier ?? '—'}</span></td>
              <td class="center">
                {#if m.loaded}
                  <span class="dot loaded-dot" title="loaded in VRAM">●</span>
                {:else}
                  <span class="dot idle-dot" title="on disk">○</span>
                {/if}
              </td>
              <td class="right vram-cell">
                {#if vramMeasured[m.id] != null}
                  <span class="hl">{fmtVram(vramMeasured[m.id])}</span>
                {:else}
                  <span class="dim">—</span>
                {/if}
              </td>
            </tr>
          {/each}
        {/if}

        {#if cloudModels.length}
          <tr class="group-header"><td colspan="4">Cloud · OVH</td></tr>
          {#each cloudModels as m}
            <tr>
              <td class="name">{m.id}</td>
              <td class="center"><span class="badge tier-{m.tier ?? 'fast'}">{m.tier ?? '—'}</span></td>
              <td class="center"><span class="dim">—</span></td>
              <td class="right"><span class="dim">cloud</span></td>
            </tr>
          {/each}
        {/if}
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

  table { width: 100%; border-collapse: collapse; font-size: .8rem; }
  th {
    font-weight: 400; opacity: .4; padding-bottom: .3rem;
    font-size: .68rem; text-transform: uppercase; letter-spacing: .05em;
  }
  td { padding: .2rem 0; vertical-align: middle; }
  tr:not(.group-header):not(:last-child) td { border-bottom: 1px solid #ffffff07; }

  .group-header td {
    font-size: .65rem; text-transform: uppercase; letter-spacing: .08em;
    opacity: .3; padding-top: .5rem; padding-bottom: .2rem;
  }

  .name { font-family: monospace; font-size: .77rem; }
  tr.loaded .name { color: var(--green); }

  .center { text-align: center; }
  .right  { text-align: right; }
  .hl     { color: var(--blue); font-variant-numeric: tabular-nums; }
  .dim    { opacity: .35; }
  .vram-cell { font-variant-numeric: tabular-nums; font-size: .77rem; }

  .dot { font-size: .7rem; }
  .loaded-dot { color: var(--green); }
  .idle-dot   { opacity: .2; }

  .badge {
    display: inline-block; font-size: .65rem; padding: .1rem .35rem;
    border-radius: 3px; font-weight: 600; letter-spacing: .04em; text-transform: uppercase;
  }
  .tier-fast     { background: #4ef1a018; color: var(--green); }
  .tier-balanced { background: #4e9af118; color: var(--blue); }
  .tier-best     { background: #9b6ef318; color: var(--purple); }
  .tier-auto     { background: #ffffff10; color: var(--text); }
</style>
