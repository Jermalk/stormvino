<script>
  import { onMount, onDestroy } from 'svelte'
  import { fetchHealth, fetchSystem } from './lib/api.js'
  import VramBar      from './lib/VramBar.svelte'
  import ServerPanel  from './lib/ServerPanel.svelte'
  import GpuPanel     from './lib/GpuPanel.svelte'
  import ProfilesPanel from './lib/ProfilesPanel.svelte'
  import SystemPanel  from './lib/SystemPanel.svelte'
  import Charts       from './lib/Charts.svelte'

  let health = $state(null)
  let sys    = $state(null)
  let error  = $state(null)
  let timers = []

  async function pollHealth() {
    try { health = await fetchHealth(); error = null }
    catch (e) { error = e.message; health = null }
  }

  async function pollSystem() {
    try { sys = await fetchSystem() }
    catch { sys = null }
  }

  onMount(() => {
    pollHealth(); pollSystem()
    timers.push(setInterval(pollHealth,  2000))
    timers.push(setInterval(pollSystem,  3000))
  })
  onDestroy(() => timers.forEach(clearInterval))
</script>

<div class="layout">
  <header>
    <span class="title">ov_monitor</span>
    <span class="host">EnvyStorm · localhost:11435</span>
    {#if error}
      <span class="error">{error}</span>
    {:else if health}
      <span class="heartbeat" title="connected">●</span>
    {/if}
    <span class="clock">{new Date().toLocaleTimeString()}</span>
  </header>

  <div class="vram-row">
    <VramBar {health} />
  </div>

  <div class="main-grid">
    <div class="cell border-right">
      <ServerPanel {health} />
    </div>
    <div class="cell">
      <GpuPanel {sys} />
    </div>
    <div class="cell border-right border-top">
      <ProfilesPanel {health} />
    </div>
    <div class="cell border-top">
      <SystemPanel {sys} />
    </div>
  </div>

  <div class="charts-row">
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

  .vram-row {
    border-bottom: 1px solid var(--border);
    background: var(--card);
  }

  .main-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    flex: 1;
  }

  .cell { background: var(--bg); }
  .border-right  { border-right:  1px solid var(--border); }
  .border-top    { border-top:    1px solid var(--border); }

  .charts-row {
    border-top: 1px solid var(--border);
    background: var(--card);
  }

  @media (max-width: 640px) {
    .main-grid { grid-template-columns: 1fr; }
    .border-right { border-right: none; }
  }
</style>
