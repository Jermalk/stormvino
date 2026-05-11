<script>
  import { fetchModelUsage } from './api.js'

  const HOURS = [
    { h: 1,   label: '1h'  },
    { h: 6,   label: '6h'  },
    { h: 24,  label: '24h' },
    { h: 168, label: '7d'  },
  ]

  let activeHours = $state(HOURS[2])
  let rows        = $state([])
  let loading     = $state(false)
  let timer       = null

  const shortId = id => (id ?? '—').replace(/-int4-ov|-int8-ov|-fp16-ov|-int4|-int8/g, '')
  const fmt1    = v  => v == null ? '—' : Number(v).toFixed(1)
  const fmt2    = v  => v == null ? '—' : Number(v).toFixed(2)

  import { onMount, onDestroy } from 'svelte'

  async function load() {
    loading = true
    try { rows = await fetchModelUsage(activeHours.h) }
    catch { rows = [] }
    loading = false
  }

  onMount(() => { load(); timer = setInterval(load, 30_000) })
  onDestroy(() => clearInterval(timer))

  function select(h) { activeHours = h; load() }

  const maxReqs = $derived(rows.length ? Math.max(...rows.map(r => r.requests)) : 1)
</script>

<section class="panel">
  <div class="toolbar">
    <h2>Model usage</h2>
    <div class="tabs">
      {#each HOURS as h}
        <button class:active={activeHours.h === h.h} onclick={() => select(h)}>{h.label}</button>
      {/each}
    </div>
  </div>

  {#if loading && !rows.length}
    <p class="dim">loading…</p>
  {:else if !rows.length}
    <p class="dim">no requests in this period</p>
  {:else}
    <table>
      <thead>
        <tr>
          <th>Model</th>
          <th class="right">reqs</th>
          <th class="right">tok/s</th>
          <th class="right">tot tok</th>
          <th class="right">avg s</th>
        </tr>
      </thead>
      <tbody>
        {#each rows as r}
          <tr>
            <td class="model-cell">
              <div class="bar-bg">
                <div class="bar-fill" style="width:{(r.requests / maxReqs * 100).toFixed(1)}%"></div>
              </div>
              <span class="model-name">{shortId(r.model_id)}</span>
            </td>
            <td class="right num">{r.requests.toLocaleString()}</td>
            <td class="right num hl">{fmt1(r.avg_tok_per_sec)}</td>
            <td class="right num">{r.total_tokens.toLocaleString()}</td>
            <td class="right num dim">{fmt2(r.avg_elapsed_sec)}</td>
          </tr>
        {/each}
      </tbody>
    </table>
  {/if}
</section>

<style>
  .panel { padding: .75rem 1rem; }
  .toolbar { display: flex; align-items: center; gap: .5rem; margin-bottom: .5rem; }
  h2 {
    font-size: .7rem; text-transform: uppercase; letter-spacing: .08em;
    opacity: .45; white-space: nowrap;
  }
  .tabs { display: flex; gap: .2rem; margin-left: auto; }
  button {
    background: var(--card); border: 1px solid #ffffff0a;
    border-radius: 4px; padding: .2rem .55rem;
    cursor: pointer; font-size: .75rem; color: inherit; opacity: .55;
  }
  button:hover  { opacity: .85; }
  button.active { opacity: 1; background: #ffffff12; border-color: #ffffff20; }

  table { width: 100%; border-collapse: collapse; font-size: .78rem; }
  th {
    font-weight: 400; opacity: .4; padding-bottom: .3rem;
    font-size: .7rem; text-transform: uppercase; letter-spacing: .05em;
  }
  td { padding: .2rem 0; vertical-align: middle; }
  tr:not(:last-child) td { border-bottom: 1px solid #ffffff07; }

  .model-cell { position: relative; padding-right: .5rem; min-width: 120px; }
  .bar-bg {
    position: absolute; inset: 2px 0;
    background: #ffffff06; border-radius: 2px; overflow: hidden;
  }
  .bar-fill { height: 100%; background: #4e9af122; border-radius: 2px; }
  .model-name { position: relative; font-family: monospace; font-size: .74rem; }

  .right  { text-align: right; padding-left: .6rem; }
  .num    { font-variant-numeric: tabular-nums; }
  .hl     { color: var(--blue); }
  .dim    { opacity: .4; }
  p.dim   { font-size: .8rem; }
</style>
