<script>
  import { switchProfile, switchScope } from './api.js'

  let { health } = $props()

  const PROFILES = [
    { name: 'fast',      label: 'Fast'      },
    { name: 'precise',   label: 'Precise'   },
    { name: 'laborious', label: 'Laborious' },
  ]
  const SCOPES = ['local', 'local+ovh', 'all']

  const active       = $derived(health?.active_profile    ?? '')
  const switching    = $derived(health?.profile_switching ?? false)
  const scope        = $derived(health?.provider_scope    ?? 'local')
  const profCfg      = $derived(health?.profiles_config   ?? {})
  const loadingModel = $derived(health?.loading_model_id  ?? null)

  const shortModel = id => (id ?? '').replace(/-int4-ov|-int8-ov|-fp16-ov|-int4|-int8/g, '')

  function pref(name)   { return profCfg[name]?.model_preference ?? '—' }
  function think(name)  { return profCfg[name]?.thinking ?? false }
  function maxtok(name) {
    const n = profCfg[name]?.max_new_tokens
    if (!n) return '—'
    return n >= 1000 ? `${Math.round(n / 1000)}K` : `${n}`
  }

  let busy        = $state(false)
  let restarting  = $state(false)

  async function setProfile(name) {
    if (busy || restarting) return
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

  async function restart() {
    if (restarting) return
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

  {#if switching}
    <div class="loading-bar">
      <div class="shimmer"></div>
      <span class="loading-text">
        {#if loadingModel}
          Loading {shortModel(loadingModel)}…
        {:else}
          Switching profile…
        {/if}
      </span>
    </div>
  {/if}

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
      {restarting ? 'restarting…' : '↺ restart'}
    </button>
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
    border-radius: 4px; padding: .2rem .6rem;
    font-size: .8rem; cursor: pointer; color: inherit;
  }
  .pill:hover:not(:disabled) { border-color: var(--blue); color: var(--blue); }
  .pill:disabled { opacity: .4; cursor: not-allowed; }
  .restart-btn {
    background: transparent; border: 1px solid #ffffff15;
    border-radius: 4px; padding: .2rem .6rem;
    font-size: .78rem; cursor: pointer; color: inherit; opacity: .5;
    transition: opacity .15s, border-color .15s, color .15s;
  }
  .restart-btn:hover:not(:disabled) { opacity: 1; border-color: var(--red); color: var(--red); }
  .restart-btn.active { opacity: 1; border-color: var(--yellow); color: var(--yellow); }
  .restart-btn:disabled { cursor: not-allowed; }

  .loading-bar {
    position: relative; overflow: hidden;
    background: #f7c44e0a; border: 1px solid #f7c44e22;
    border-radius: 5px; padding: .35rem .7rem;
    margin-bottom: .55rem;
    display: flex; align-items: center; gap: .5rem;
  }
  .shimmer {
    position: absolute; inset: 0;
    background: linear-gradient(90deg, transparent 0%, #f7c44e18 50%, transparent 100%);
    background-size: 200% 100%;
    animation: shimmer 1.4s ease-in-out infinite;
  }
  .loading-text {
    position: relative; font-size: .76rem; color: var(--yellow);
    font-family: monospace; white-space: nowrap; overflow: hidden;
    text-overflow: ellipsis;
  }
  @keyframes shimmer {
    0%   { background-position: -200% 0; }
    100% { background-position:  200% 0; }
  }
</style>
