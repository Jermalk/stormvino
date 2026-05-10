<script>
  import { onMount, onDestroy } from 'svelte'
  import { fetchHealth } from './lib/api.js'
  import StatsPanel  from './lib/StatsPanel.svelte'
  import VramBar     from './lib/VramBar.svelte'
  import ModelsPanel from './lib/ModelsPanel.svelte'
  import Charts      from './lib/Charts.svelte'

  let health = $state(null)
  let error  = $state(null)
  let timer

  async function poll() {
    try {
      health = await fetchHealth()
      error  = null
    } catch (e) {
      error  = e.message
      health = null
    }
  }

  onMount(() => {
    poll()
    timer = setInterval(poll, 2000)
  })

  onDestroy(() => clearInterval(timer))
</script>

<div class="layout">
  <header>
    <span class="title">ov_monitor</span>
    {#if error}
      <span class="error">{error}</span>
    {:else if health}
      <span class="heartbeat">●</span>
    {/if}
  </header>

  <main>
    <VramBar     {health} />
    <ModelsPanel {health} />
    <StatsPanel  {health} />
    <Charts />
  </main>
</div>

<style>
  :global(*) { box-sizing: border-box; margin: 0; padding: 0; }
  :global(:root) {
    --bg:     #0f1117;
    --card:   #1a1d27;
    --blue:   #4e9af1;
    --purple: #9b6ef3;
    --green:  #4ef1a0;
    --red:    #f1544e;
    --text:   #d0d4e0;
  }
  :global(body) { background: var(--bg); color: var(--text); font-family: system-ui, sans-serif; font-size: 14px; }

  .layout { display: flex; flex-direction: column; min-height: 100vh; }

  header {
    display: flex; align-items: center; gap: 1rem;
    padding: .6rem 1rem;
    background: var(--card);
    border-bottom: 1px solid #ffffff10;
    position: sticky; top: 0; z-index: 10;
  }
  .title     { font-weight: 700; letter-spacing: .05em; }
  .heartbeat { color: var(--green); animation: pulse 1.5s infinite; }
  .error     { color: var(--red); font-size: .8rem; }
  @keyframes pulse { 0%,100%{opacity:1} 50%{opacity:.3} }

  main { display: flex; flex-direction: column; gap: 1px; background: #ffffff08; flex: 1; }
  main > :global(*) { background: var(--bg); }
</style>
