<script>
  import { onMount } from 'svelte'
  import { switchProfile, switchScope, fetchAvailableModels, loadModel } from './api.js'

  let { health, onActionStart } = $props()

  const PROFILES = [
    { name: 'fast',      label: 'Fast'      },
    { name: 'precise',   label: 'Precise'   },
    { name: 'laborious', label: 'Laborious' },
  ]
  const SCOPES = ['local', 'local+ovh', 'all']

  const active    = $derived(health?.active_profile    ?? '')
  const switching = $derived(health?.profile_switching ?? false)
  const scope     = $derived(health?.provider_scope    ?? 'local')
  const profCfg   = $derived(health?.profiles_config   ?? {})

  function pref(name)   { return profCfg[name]?.model_preference ?? '—' }
  function think(name)  { return profCfg[name]?.thinking ?? false }
  function maxtok(name) {
    const n = profCfg[name]?.max_new_tokens
    if (!n) return '—'
    return n >= 1000 ? `${Math.round(n / 1000)}K` : `${n}`
  }

  let busy       = $state(false)
  let restarting = $state(false)

  // Model selection
  let availableLlms = $state([])
  let selectedModel = $state('auto')

  onMount(async () => {
    try {
      const data = await fetchAvailableModels()
      availableLlms = data.llm ?? []
    } catch { /* server not ready yet — leave list empty */ }
  })

  async function setProfile(name) {
    if (busy || restarting) return
    onActionStart()
    busy = true
    await switchProfile(name).catch(() => {})
    busy = false
  }

  async function cycleScope() {
    if (busy || restarting) return
    busy = true
    const next = SCOPES[(SCOPES.indexOf(scope) + 1) % SCOPES.length]
    await switchScope(next).catch(() => {})
    busy = false
  }

  async function handleModelSelect() {
    if (busy || restarting) return
    onActionStart()
    busy = true
    await loadModel(selectedModel).catch(() => {})
    busy = false
  }

  async function restart() {
    if (restarting) return
    onActionStart()
    restarting = true
    await fetch('/maintenance/restart', { method: 'POST' }).catch(() => {})
    // Wait for server to go down, then poll until it's back
    await new Promise(r => setTimeout(r, 3000))
    while (restarting) {
      try {
        const r = await fetch('/health')
        if (r.ok) { restarting = false; break }
      } catch { /* still down */ }
      await new Promise(r => setTimeout(r, 1000))
    }
  }
</script>

<section class="panel">
  <h2>Profiles</h2>
  <div class="cards">
    {#each PROFILES as p}
      <button
        class="card"
        class:active={p.name === active}
        class:switching={p.name === active && switching}
        onclick={() => setProfile(p.name)}
        disabled={busy || restarting}
      >
        <span class="dot"></span>
        <div class="info">
          <span class="pname">{p.label}</span>
          <span class="pdesc">
            {pref(p.name)} · {think(p.name) ? 'think' : 'no think'} · {maxtok(p.name)} tok
          </span>
        </div>
      </button>
    {/each}
  </div>

  <div class="controls">
    <div class="scope-row">
      <span class="slabel">Scope</span>
      <button class="pill" onclick={cycleScope} disabled={busy || restarting}>
        {scope}
      </button>
    </div>
    <button
      class="restart-btn"
      class:active={restarting}
      onclick={restart}
      disabled={restarting}
      title="Graceful restart via systemd"
    >
      <span class="restart-icon">↺</span><span class="restart-text">{restarting ? 'restarting…' : 'restart'}</span>
    </button>
  </div>

  <div class="model-row">
    <span class="slabel">Model</span>
    <select
      class="model-sel"
      bind:value={selectedModel}
      onchange={handleModelSelect}
      disabled={busy || restarting}
    >
      <option value="auto">AUTO</option>
      {#each availableLlms as m}
        <option value={m}>{m.replace(/-int4-ov|-int8-ov|-fp16-ov/g, '')}</option>
      {/each}
    </select>
  </div>
</section>

<style>
  .panel { padding: .75rem 1rem; }
  h2 { font-size: .7rem; text-transform: uppercase; letter-spacing: .08em; opacity: .45; margin-bottom: .6rem; }
  .cards { display: flex; flex-direction: column; gap: .3rem; margin-bottom: .7rem; }
  .card {
    display: flex; align-items: center; gap: .6rem;
    background: var(--card); border: 1px solid #ffffff0a;
    border-radius: 6px; padding: .45rem .7rem;
    cursor: pointer; text-align: left; color: inherit; transition: border-color .15s;
  }
  .card:hover:not(:disabled) { border-color: #ffffff25; }
  .card.active   { border-color: var(--blue); }
  .card.switching { border-color: var(--yellow); }
  .card:disabled { opacity: .5; cursor: not-allowed; }
  .dot {
    width: 8px; height: 8px; border-radius: 50%; flex-shrink: 0;
    background: #ffffff20; transition: background .2s;
  }
  .card.active .dot    { background: var(--blue); box-shadow: 0 0 6px var(--blue)55; }
  .card.switching .dot { background: var(--yellow); }
  .info { display: flex; flex-direction: column; }
  .pname { font-size: .85rem; font-weight: 600; }
  .pdesc { font-size: .7rem; opacity: .45; margin-top: .05rem; }
  .controls { display: flex; align-items: center; gap: .5rem; flex-wrap: wrap; }
  .scope-row { display: flex; align-items: center; gap: .4rem; flex: 1; }
  .slabel { font-size: .75rem; opacity: .45; white-space: nowrap; }
  .pill {
    background: var(--card); border: 1px solid #ffffff15;
    border-radius: 4px; padding: .32rem .9rem;
    font-size: .92rem; cursor: pointer; color: inherit;
  }
  .pill:hover:not(:disabled) { border-color: var(--blue); color: var(--blue); }
  .pill:disabled { opacity: .4; cursor: not-allowed; }
  .restart-btn {
    display: flex; align-items: center; gap: .3rem;
    background: transparent; border: 1px solid #ffffff15;
    border-radius: 4px; padding: .32rem .9rem;
    font-size: .92rem; cursor: pointer; color: inherit;
    transition: border-color .15s;
  }
  .restart-btn:hover:not(:disabled) { border-color: #ffffff30; }
  .restart-btn.active { border-color: var(--yellow); }
  .restart-btn:disabled { cursor: not-allowed; }
  .restart-icon { color: var(--red); font-size: 1rem; line-height: 1; }
  .restart-text { opacity: .45; }
  .restart-btn.active .restart-icon { color: var(--yellow); }
  .restart-btn.active .restart-text { opacity: 1; color: var(--yellow); }

  .model-row {
    display: flex; align-items: center; gap: .4rem; margin-top: .5rem;
  }
  .model-sel {
    flex: 1;
    background: var(--card); border: 1px solid #ffffff15;
    border-radius: 4px; padding: .3rem .4rem;
    font-size: .82rem; color: inherit; cursor: pointer;
    appearance: auto;
  }
  .model-sel:hover:not(:disabled) { border-color: var(--blue); }
  .model-sel:disabled { opacity: .4; cursor: not-allowed; }
</style>
