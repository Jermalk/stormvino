<script>
  import { onMount, onDestroy } from 'svelte'
  import { fetchHealth, fetchSystem, fetchProfilerStatus } from './lib/api.js'
  import VramBar       from './lib/VramBar.svelte'
  import ServerPanel   from './lib/ServerPanel.svelte'
  import GpuPanel      from './lib/GpuPanel.svelte'
  import ProfilesPanel from './lib/ProfilesPanel.svelte'
  import ProfilerPanel from './lib/ProfilerPanel.svelte'
  import SystemPanel   from './lib/SystemPanel.svelte'
  import Charts          from './lib/Charts.svelte'
  import ModelUsage      from './lib/ModelUsage.svelte'
  import CataloguePanel  from './lib/CataloguePanel.svelte'

  let health   = $state(null)
  let sys      = $state(null)
  let profiler = $state(null)
  let error    = $state(null)
  let timers   = []

  async function pollHealth() {
    try { health = await fetchHealth(); error = null }
    catch (e) { error = e.message; health = null }
  }

  async function pollSystem() {
    try { sys = await fetchSystem() }
    catch { sys = null }
  }

  async function pollProfiler() {
    try { profiler = await fetchProfilerStatus() }
    catch { profiler = null }
  }

  onMount(() => {
    pollHealth(); pollSystem(); pollProfiler()
    timers.push(setInterval(pollHealth,   1000))
    timers.push(setInterval(pollSystem,   3000))
    timers.push(setInterval(pollProfiler, 4000))
  })
  onDestroy(() => timers.forEach(clearInterval))
</script>

<div class="layout">
  <header>
    <span class="title">SVP</span>
    <span class="host">EnvyStorm · localhost:11435</span>
    {#if error}
      <span class="error">{error}</span>
    {:else if health}
      <span class="heartbeat" title="connected">●</span>
    {/if}
    <span class="clock">{new Date().toLocaleTimeString()}</span>
  </header>

  <!-- Row 1: VRAM bar -->
  <div class="vram-row">
    <VramBar {health} />
  </div>

  <!-- Row 2: Server (40%) · Arc B60 (30%) · Profiles (30%) -->
  <div class="row tri">
    <div class="cell"><ServerPanel {health} /></div>
    <div class="cell"><GpuPanel {sys} /></div>
    <div class="cell"><ProfilesPanel {health} /></div>
  </div>

  <!-- Row 3: CPU + Memory (40%) · VRAM Profiler (30%) · Model usage (30%) -->
  <div class="row tri">
    <div class="cell"><SystemPanel {sys} /></div>
    <div class="cell"><ProfilerPanel {profiler} /></div>
    <div class="cell"><ModelUsage /></div>
  </div>

  <!-- Row 4: Model catalogue (100%) -->
  <div class="row full">
    <CataloguePanel {profiler} />
  </div>

  <!-- Row 5: History chart (100%) -->
  <div class="row full">
    <Charts />
  </div>
</div>

<style>
  :global(*) { box-sizing: border-box; margin: 0; padding: 0; }
  :global(:root) {
    --bg:     #0f1117;
    --card:   #161922;
    --border: #ffffff0c;
    --blue:   #4e9af1;
    --purple: #9b6ef3;
    --green:  #4ef1a0;
    --yellow: #f7c44e;
    --red:    #f1544e;
    --text:   #c8ccd8;
  }
  :global(body) {
    background: var(--bg); color: var(--text);
    font-family: system-ui, sans-serif; font-size: 14px;
  }

  .layout { display: flex; flex-direction: column; min-height: 100vh; }

  header {
    display: flex; align-items: center; gap: .75rem;
    padding: .5rem 1rem; background: var(--card);
    border-bottom: 1px solid var(--border);
    position: sticky; top: 0; z-index: 10; font-size: .82rem;
  }
  .title     { font-weight: 700; letter-spacing: .04em; font-size: .9rem; }
  .host      { opacity: .35; font-size: .75rem; }
  .heartbeat { color: var(--green); animation: pulse 2s infinite; }
  .error     { color: var(--red); }
  .clock     { margin-left: auto; opacity: .35; font-size: .75rem; font-variant-numeric: tabular-nums; }
  @keyframes pulse { 0%,100%{opacity:1} 50%{opacity:.25} }

  .vram-row { background: var(--card); }

  /* Rows: border-top separates every row from the one above */
  .row { display: grid; min-width: 0; border-top: 1px solid var(--border); }
  .tri  { grid-template-columns: 4fr 3fr 3fr; }
  .full { background: var(--card); }

  /* Cells: right border on all but the last in a row */
  .cell { background: var(--bg); min-width: 0; overflow: hidden; border-right: 1px solid var(--border); }
  .cell:last-child { border-right: none; }

  /* Responsive: collapse to 1 column below 900px */
  @media (max-width: 900px) {
    .tri { grid-template-columns: 1fr; }
    .cell { border-right: none; border-bottom: 1px solid var(--border); }
    .cell:last-child { border-bottom: none; }
  }
</style>
