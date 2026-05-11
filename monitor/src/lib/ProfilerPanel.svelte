<script>
  import { triggerProfiling } from './api.js'

  let { profiler } = $props()

  const running     = $derived(profiler?.running   ?? false)
  const current     = $derived(profiler?.current   ?? null)
  const pendingLlm  = $derived(profiler?.pending_llms ?? [])
  const pendingVlm  = $derived(profiler?.pending_vlms ?? [])
  const profiled    = $derived(profiler?.profiled   ?? [])
  const measured    = $derived(profiler?.vram_measured ?? {})
  const unmeasLlm   = $derived(profiler?.unmeasured_llms ?? [])
  const unmeasVlm   = $derived(profiler?.unmeasured_vlms ?? [])
  const unmeasured  = $derived([...unmeasLlm, ...unmeasVlm])

  const shortId = id => id.replace(/-int4-ov|-int8-ov|-fp16-ov|-int4|-int8/g, '')

  let triggering = $state(false)
  async function trigger() {
    if (triggering || running) return
    triggering = true
    await triggerProfiling().catch(() => {})
    await new Promise(r => setTimeout(r, 800))
    triggering = false
  }
</script>

<section class="panel">
  <h2>VRAM Profiler</h2>

  {#if !profiler}
    <p class="dim">unavailable</p>
  {:else}
    <div class="status-row">
      {#if running}
        <span class="badge running">RUNNING</span>
        {#if current}
          <span class="current">→ {shortId(current)}</span>
        {/if}
      {:else if unmeasured.length}
        <span class="badge idle-warn">IDLE</span>
        <span class="warn">{unmeasured.length} model{unmeasured.length > 1 ? 's' : ''} unmeasured</span>
      {:else}
        <span class="badge idle-ok">IDLE</span>
        <span class="dim">all measured</span>
      {/if}
      <button
        class="run-btn"
        onclick={trigger}
        disabled={triggering || running}
        title="Trigger VRAM profiling run"
      >
        {running ? '…' : triggering ? '…' : '▷ run'}
      </button>
    </div>

    {#if running && (pendingLlm.length + pendingVlm.length)}
      <div class="queue dim">
        pending: {[...pendingLlm, ...pendingVlm].map(shortId).join(', ')}
      </div>
    {/if}

    {#if profiled.length}
      <div class="queue">
        profiled this run: <span class="hl">{profiled.map(shortId).join(', ')}</span>
      </div>
    {/if}

    {#if unmeasured.length && !running}
      <div class="queue warn">
        missing: {unmeasured.map(shortId).join(', ')}
      </div>
    {/if}

    {#if Object.keys(measured).length}
      <table class="meas">
        <thead>
          <tr><th>Model</th><th class="right">VRAM</th></tr>
        </thead>
        <tbody>
          {#each Object.entries(measured).sort() as [id, gb]}
            <tr>
              <td class="mono">{shortId(id)}</td>
              <td class="right hl">{Number(gb).toFixed(2)} GB</td>
            </tr>
          {/each}
        </tbody>
      </table>
    {/if}
  {/if}
</section>

<style>
  .panel { padding: .75rem 1rem; }
  h2 { font-size: .7rem; text-transform: uppercase; letter-spacing: .08em; opacity: .45; margin-bottom: .6rem; }

  .status-row { display: flex; align-items: center; gap: .5rem; margin-bottom: .4rem; flex-wrap: wrap; }
  .badge {
    font-size: .68rem; font-weight: 700; letter-spacing: .06em;
    padding: .1rem .4rem; border-radius: 3px;
  }
  .running  { background: var(--yellow)33; color: var(--yellow); }
  .idle-ok  { background: var(--green)22;  color: var(--green); }
  .idle-warn{ background: var(--yellow)22; color: var(--yellow); }
  .current { font-size: .8rem; font-family: monospace; opacity: .8; }
  .warn    { font-size: .78rem; color: var(--yellow); }
  .dim     { opacity: .4; font-size: .78rem; }
  .hl      { color: var(--blue); }

  .run-btn {
    margin-left: auto;
    background: transparent; border: 1px solid #ffffff18;
    border-radius: 4px; padding: .15rem .55rem;
    font-size: .75rem; cursor: pointer; color: inherit; opacity: .6;
    transition: opacity .15s, border-color .15s, color .15s;
  }
  .run-btn:hover:not(:disabled) { opacity: 1; border-color: var(--blue); color: var(--blue); }
  .run-btn:disabled { cursor: not-allowed; opacity: .3; }

  .queue { font-size: .75rem; margin-bottom: .3rem; opacity: .6; }

  .meas { width: 100%; border-collapse: collapse; font-size: .78rem; margin-top: .5rem; }
  .meas th { opacity: .4; font-weight: 400; padding-bottom: .2rem; }
  .meas td { padding: .1rem 0; }
  .meas tr:not(:last-child) td { border-bottom: 1px solid #ffffff08; }
  .right { text-align: right; }
  .mono  { font-family: monospace; font-size: .75rem; }
</style>
