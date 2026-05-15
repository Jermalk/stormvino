<script>
  const COLORS = ['#4e9af1', '#9b6ef3', '#4ef1a0', '#f7c44e', '#f1544e']

  let { health, vramLive } = $props()

  // Use sidecar's live VRAM for total/used; fall back to health if sidecar not yet ready.
  const totalGb = $derived(health?.vram_total_gb ?? vramLive?.total_gb ?? 22.71)

  // Live server-process VRAM from sidecar — grows during model load.
  const liveServerGb = $derived(vramLive?.by_proc?.['ov_server'] ?? 0)

  // Per-model coloured segments from server health (registered models only).
  const segments = $derived.by(() => {
    if (!health) return []
    if (!totalGb) return []
    const vlmIds = new Set(health.loaded_vlm_models ?? [])
    const kvGb   = health.kv_cache_size_gb ?? 0
    const aKvGb  = health.assessor_kv_cache_size_gb ?? kvGb
    const alloc  = health.vram_allocated_gb ?? {}
    return Object.entries(alloc).map(([id, allocGb], i) => {
      const isVlm     = vlmIds.has(id)
      const thisKv    = id === '_assessor' ? aKvGb : kvGb
      const weightsGb = isVlm ? allocGb : Math.max(0, allocGb - thisKv)
      const kvSegGb   = isVlm ? 0 : thisKv
      const color     = COLORS[i % COLORS.length]
      return { id, weightsGb, kvSegGb, color, isVlm,
               wPct: weightsGb / totalGb * 100,
               kPct: kvSegGb   / totalGb * 100 }
    })
  })

  // Allocated by OV (what the server knows about).
  const allocatedSum = $derived(
    Object.values(health?.vram_allocated_gb ?? {}).reduce((s, v) => s + v, 0)
  )

  // Loading gap: VRAM the server process holds beyond what's registered.
  // Grows in real-time as a model loads. OV runtime overhead is ~0.9 GB,
  // so threshold at 1.2 GB to avoid false animation at idle.
  const loadingGb  = $derived(Math.max(0, liveServerGb - allocatedSum))

  // Canonical loading flag — explicit server signals take priority.
  // loadingGb > 1.2 is the belt-and-suspenders fallback for the brief gap
  // before the server sets loading_model_id (e.g. task scheduled but not yet run).
  const isLoading = $derived(
    !!(health?.loading_model_id) ||
    !!(health?.profile_switching) ||
    !!(health?.startup_loading) ||
    loadingGb > 1.2
  )

  const usedGb = $derived(liveServerGb || allocatedSum)
  const freeGb = $derived(Math.max(0, totalGb - usedGb))
  const pct    = $derived(totalGb ? usedGb / totalGb * 100 : 0)
  const over   = $derived(pct > 100)

  const loadingLabel = $derived(health?.loading_model_id
    ? health.loading_model_id.replace(/-int4-ov|-int8-ov|-fp16-ov|-int4|-int8/g, '')
    : null)
</script>

<section class="vram-section">
  <div class="header">
    <span class="title">VRAM</span>
    <span class="nums">{usedGb.toFixed(1)} / {totalGb.toFixed(1)} GB</span>
    <span class="pct" class:over>{Math.min(pct, 100).toFixed(1)}%{over ? '!' : ''}</span>
  </div>

  <div class="bar-track">
    {#if isLoading}
      <div class="seg loading-seg" style="width:100%">
        <div class="load-shimmer"></div>
      </div>
    {:else}
      {#each segments as s}
        <div class="seg weights" style="width:{s.wPct}%; background:{s.color}" title="{s.id} weights {s.weightsGb.toFixed(1)}GB"></div>
        {#if s.kPct > 0}
          <div class="seg kv" style="width:{s.kPct}%; background:{s.color}66" title="{s.id} KV {s.kvSegGb.toFixed(1)}GB"></div>
        {/if}
      {/each}
      <div class="seg free" style="flex:1" title="free {freeGb.toFixed(1)}GB"></div>
    {/if}
  </div>

  <div class="legend">
    {#if isLoading}
      <span class="leg-item loading-item">
        <span class="dot loading-dot"></span>
        <span class="leg-label">
          {loadingLabel ? `Loading ${loadingLabel}…` : (health?.profile_switching ? 'Switching profile…' : 'Initializing…')}
        </span>
        {#if loadingGb > 0.5}
          <span class="leg-detail">{loadingGb.toFixed(1)}GB</span>
        {/if}
      </span>
    {:else}
      {#each segments as s}
        <span class="leg-item">
          <span class="dot" style="background:{s.color}"></span>
          <span class="leg-label">{s.id.replace(/-int4-ov|-int8-ov|-fp16-ov/g, '')}</span>
          <span class="leg-detail">
            {s.isVlm ? `${s.weightsGb.toFixed(1)}GB` : `${s.weightsGb.toFixed(1)}+${s.kvSegGb.toFixed(1)}GB`}
          </span>
        </span>
      {/each}
      <span class="leg-item">
        <span class="dot free-dot"></span>
        <span class="leg-label">free</span>
        <span class="leg-detail">{freeGb.toFixed(1)}GB</span>
      </span>
    {/if}
  </div>
</section>

<style>
  .vram-section { padding: .75rem 1rem; }
  .header { display: flex; align-items: baseline; gap: .6rem; margin-bottom: .4rem; }
  .title  { font-size: .7rem; text-transform: uppercase; letter-spacing: .08em; opacity: .5; }
  .nums   { font-size: .85rem; font-weight: 600; }
  .pct    { font-size: .85rem; font-weight: 700; color: var(--green); margin-left: auto; }
  .pct.over { color: var(--red); }
  .bar-track { height: 14px; background: #ffffff0e; border-radius: 7px; overflow: hidden; display: flex; }
  .seg { height: 100%; transition: width .4s; }
  .free { background: transparent; }

  .loading-seg { position: relative; overflow: hidden; transition: width .5s; }
  .load-shimmer {
    position: absolute; inset: 0;
    background: repeating-linear-gradient(90deg, transparent 0%, #f7c44e22 40%, #f7c44e44 50%, #f7c44e22 60%, transparent 100%);
    background-size: 200% 100%;
    animation: vram-shimmer 1.4s ease-in-out infinite;
  }
  @keyframes vram-shimmer {
    0%   { background-position: 200% 0; }
    100% { background-position: -200% 0; }
  }

  .legend { display: flex; flex-wrap: wrap; gap: .35rem .9rem; margin-top: .5rem; }
  .leg-item { display: flex; align-items: center; gap: .3rem; font-size: .82rem; }
  .dot { width: 9px; height: 9px; border-radius: 2px; flex-shrink: 0; }
  .free-dot { background: #ffffff20; }
  .leg-label { opacity: .8; }
  .leg-detail { opacity: .5; font-size: .76rem; }

  .loading-item .leg-label { color: var(--yellow); opacity: 1; animation: leg-pulse 1.4s ease-in-out infinite; }
  .loading-dot { background: var(--yellow); }
  @keyframes leg-pulse { 0%,100% { opacity:.6 } 50% { opacity:1 } }

  /* Narrow: hide legend, GB numbers in header are enough */
  @media (max-width: 640px) {
    .legend { display: none; }
  }
</style>
