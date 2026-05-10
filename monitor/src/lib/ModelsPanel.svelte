<script>
  /** @type {{ health: object | null }} */
  let { health } = $props()

  const llms  = $derived(health?.loaded_models     ?? [])
  const vlms  = $derived(health?.loaded_vlm_models ?? [])
  const imageLoaded = $derived(health?.image_model_loaded ?? false)
  const imageId     = $derived(health?.image_model_id     ?? '')
  const sttLoaded   = $derived(health?.stt_model_loaded   ?? false)
  const profile     = $derived(health?.active_profile     ?? '—')
  const routing     = $derived(health?.last_routing_decision ?? null)
</script>

<section class="models-panel">
  <h2>Models</h2>

  <div class="row">
    <span class="badge profile">{profile}</span>
    {#each llms as m}
      <span class="badge llm">{m}</span>
    {/each}
    {#each vlms as m}
      <span class="badge vlm">{m}</span>
    {/each}
    {#if imageLoaded}
      <span class="badge img">img:{imageId}</span>
    {/if}
    {#if sttLoaded}
      <span class="badge stt">stt</span>
    {/if}
  </div>

  {#if routing}
    <div class="routing">
      <span class="label">Last route</span>
      <span class="mono">{routing.task_class ?? '—'}</span>
      <span class="sep">→</span>
      <span class="mono">{routing.model ?? '—'}</span>
      <span class="conf">({((routing.confidence ?? 0) * 100).toFixed(0)}% · {routing.strategy ?? ''})</span>
    </div>
  {/if}
</section>

<style>
  .models-panel { padding: 1rem; }
  h2 { margin: 0 0 .5rem; font-size: 1rem; text-transform: uppercase; letter-spacing: .08em; opacity: .6; }
  .row { display: flex; flex-wrap: wrap; gap: .4rem; margin-bottom: .5rem; }
  .badge { border-radius: 4px; padding: .2rem .5rem; font-size: .75rem; font-family: monospace; }
  .profile { background: #333; }
  .llm     { background: #1a3a5c; color: #7ec8f7; }
  .vlm     { background: #2e1a5c; color: #c07ef7; }
  .img     { background: #1a4c2e; color: #7ef7a0; }
  .stt     { background: #4c3a1a; color: #f7c47e; }
  .routing { font-size: .8rem; display: flex; align-items: center; gap: .4rem; opacity: .75; }
  .label   { opacity: .55; }
  .mono    { font-family: monospace; }
  .sep     { opacity: .4; }
  .conf    { opacity: .5; font-size: .7rem; }
</style>
