<script>
  import { switchProfile, switchScope } from './api.js'

  let { health } = $props()

  const PROFILES = [
    { name: 'fast',      label: 'Fast',      desc: '14b · KV 8GB · local' },
    { name: 'precise',   label: 'Precise',   desc: '14b · KV 8GB · think' },
    { name: 'laborious', label: 'Laborious', desc: '14b · KV 8GB · deep'  },
  ]
  const SCOPES = ['local', 'local+ovh', 'all']

  const active    = $derived(health?.active_profile ?? '')
  const switching = $derived(health?.profile_switching ?? false)
  const scope     = $derived(health?.provider_scope ?? 'local')

  let busy = $state(false)

  async function setProfile(name) {
    if (busy) return
    busy = true
    await switchProfile(name).catch(() => {})
    busy = false
  }

  async function cycleScope() {
    if (busy) return
    busy = true
    const next = SCOPES[(SCOPES.indexOf(scope) + 1) % SCOPES.length]
    await switchScope(next).catch(() => {})
    busy = false
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
        disabled={busy}
      >
        <span class="dot"></span>
        <div class="info">
          <span class="pname">{p.label}</span>
          <span class="pdesc">{p.desc}</span>
        </div>
      </button>
    {/each}
  </div>

  <div class="scope-row">
    <span class="slabel">Scope</span>
    <button class="scope-btn" onclick={cycleScope} disabled={busy}>
      {scope}
    </button>
  </div>
</section>

<style>
  .panel { padding: .75rem 1rem; }
  h2 { font-size: .7rem; text-transform: uppercase; letter-spacing: .08em; opacity: .45; margin-bottom: .6rem; }
  .cards { display: flex; flex-direction: column; gap: .3rem; margin-bottom: .6rem; }
  .card {
    display: flex; align-items: center; gap: .6rem;
    background: var(--card); border: 1px solid #ffffff0a;
    border-radius: 6px; padding: .45rem .7rem;
    cursor: pointer; text-align: left; color: inherit; transition: border-color .15s;
  }
  .card:hover:not(:disabled) { border-color: #ffffff25; }
  .card.active { border-color: var(--blue); }
  .card.switching { border-color: var(--yellow); }
  .dot {
    width: 8px; height: 8px; border-radius: 50%; flex-shrink: 0;
    background: #ffffff20;
  }
  .card.active .dot    { background: var(--blue); box-shadow: 0 0 6px var(--blue); }
  .card.switching .dot { background: var(--yellow); }
  .info { display: flex; flex-direction: column; }
  .pname { font-size: .85rem; font-weight: 600; }
  .pdesc { font-size: .7rem; opacity: .45; }
  .scope-row { display: flex; align-items: center; gap: .5rem; }
  .slabel { font-size: .75rem; opacity: .45; }
  .scope-btn {
    background: var(--card); border: 1px solid #ffffff15;
    border-radius: 4px; padding: .2rem .6rem;
    font-size: .8rem; cursor: pointer; color: inherit;
  }
  .scope-btn:hover:not(:disabled) { border-color: var(--blue); color: var(--blue); }
</style>
