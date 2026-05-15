<script>
  import { onMount, onDestroy } from 'svelte'
  import { fetchSidecar, fetchProfilerStatus } from './lib/api.js'
  import VramBar       from './lib/VramBar.svelte'
  import ServerPanel   from './lib/ServerPanel.svelte'
  import GpuPanel      from './lib/GpuPanel.svelte'
  import ProfilesPanel from './lib/ProfilesPanel.svelte'
  import SystemPanel   from './lib/SystemPanel.svelte'
  import Charts               from './lib/Charts.svelte'
  import ModelUsage           from './lib/ModelUsage.svelte'
  import ModelCataloguePanel  from './lib/ModelCataloguePanel.svelte'

  // Sidecar is the primary data source: health + system + live VRAM.
  let metrics        = $state(null)
  let profiler       = $state(null)
  let error          = $state(null)
  let timers         = []
  let clockStr       = $state(new Date().toLocaleTimeString())
  let loadingOverride = $state(false)

  // Set immediately when the user triggers a loading action (profile switch, restart).
  // Clears automatically once health reports stable — all loading flags gone and
  // startup_loading is false. Stays true while server is offline (health == null).
  function onActionStart() { loadingOverride = true }

  $effect(() => {
    if (
      loadingOverride &&
      health != null &&
      !health.loading_model_id &&
      !health.profile_switching &&
      !health.startup_loading
    ) {
      loadingOverride = false
    }
  })

  // Derived views used by child panels (same shape as before).
  const health   = $derived(metrics?.server_health ?? null)
  const sys      = $derived(metrics?.system        ?? null)
  const vramLive = $derived(metrics?.vram_live      ?? null)
  const serverUp = $derived(metrics?.server_up      ?? false)

  async function pollSidecar() {
    try { metrics = await fetchSidecar(); error = null }
    catch (e) { error = e.message }
  }

  async function pollProfiler() {
    try { profiler = await fetchProfilerStatus() }
    catch { profiler = null }
  }

  onMount(() => {
    pollSidecar(); pollProfiler()
    timers.push(setInterval(pollSidecar,   1000))
    timers.push(setInterval(pollProfiler,  4000))
    timers.push(setInterval(() => { clockStr = new Date().toLocaleTimeString() }, 1000))
  })
  onDestroy(() => timers.forEach(clearInterval))
</script>

<div class="layout">
  <header>
    <span class="title">SVP</span>
    <span class="host">EnvyStorm · localhost:11435</span>
    {#if error}
      <span class="error">{error}</span>
    {:else if serverUp}
      <span class="heartbeat" title="server connected">●</span>
    {:else if metrics}
      <span class="server-down" title="server offline">⊘</span>
    {/if}
    <span class="clock">{clockStr}</span>
  </header>

  <!-- Row 1: VRAM bar -->
  <div class="vram-row">
    <VramBar {health} {vramLive} overrideLoading={loadingOverride} />
  </div>

  <!-- Row 2: Server (40%) · Arc B60 (30%) · Profiles (30%) -->
  <div class="row tri">
    <div class="cell"><ServerPanel {health} /></div>
    <div class="cell"><GpuPanel {sys} /></div>
    <div class="cell"><ProfilesPanel {health} {onActionStart} /></div>
  </div>

  <!-- Row 3: CPU + Memory (50%) · Model usage (50%) -->
  <div class="row bi">
    <div class="cell"><SystemPanel {sys} /></div>
    <div class="cell"><ModelUsage /></div>
  </div>

  <!-- Row 4: Unified model catalogue + profiler (100%) -->
  <div class="row full">
    <ModelCataloguePanel {profiler} />
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
  .heartbeat   { color: var(--green); animation: pulse 2s infinite; }
  .server-down { color: var(--red); opacity: .7; }
  .error       { color: var(--red); }
  .clock     { margin-left: auto; opacity: .35; font-size: .75rem; font-variant-numeric: tabular-nums; }
  @keyframes pulse { 0%,100%{opacity:1} 50%{opacity:.25} }

  .vram-row { background: var(--card); }

  /* Rows: border-top separates every row from the one above */
  .row { display: grid; min-width: 0; border-top: 1px solid var(--border); }
  .tri  { grid-template-columns: 4fr 3fr 3fr; }
  .bi   { grid-template-columns: 3fr 7fr; }
  .full { background: var(--card); }

  /* Cells: right border on all but the last in a row */
  .cell { background: var(--bg); min-width: 0; overflow: hidden; border-right: 1px solid var(--border); }
  .cell:last-child { border-right: none; }

  /* Responsive: collapse to 1 column below 900px */
  @media (max-width: 900px) {
    .tri, .bi { grid-template-columns: 1fr; }
    .cell { border-right: none; border-bottom: 1px solid var(--border); }
    .cell:last-child { border-bottom: none; }
  }

  /* Narrow screens: strip header to dot + title only */
  @media (max-width: 640px) {
    .host, .clock { display: none; }
  }
</style>
