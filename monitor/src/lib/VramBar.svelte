<script>
  /** @type {{ health: object | null }} */
  let { health } = $props()

  const segments = $derived.by(() => {
    if (!health) return []
    const total = health.vram_total_gb ?? 0
    if (!total) return []
    // Build segments from loaded_models + loaded_vlm_models + image/stt
    const segs = []
    for (const [id, info] of Object.entries(health.loaded_models_detail ?? {})) {
      segs.push({ label: id, gb: info.vram_gb ?? 0, color: 'var(--blue)' })
    }
    for (const [id, info] of Object.entries(health.loaded_vlm_detail ?? {})) {
      segs.push({ label: id, gb: info.vram_gb ?? 0, color: 'var(--purple)' })
    }
    return segs
  })

  const usedGb  = $derived(health?.vram_used_gb  ?? 0)
  const totalGb = $derived(health?.vram_total_gb ?? 0)
  const pct     = $derived(totalGb ? (usedGb / totalGb * 100) : 0)
  const over    = $derived(pct > 100)
</script>

<section class="vram-panel">
  <h2>VRAM — {usedGb.toFixed(1)} / {totalGb.toFixed(1)} GB
    <span class="pct" class:over>{Math.min(pct, 100).toFixed(1)}%{over ? '!' : ''}</span>
  </h2>

  <div class="bar-track">
    {#each segments as seg}
      <div class="seg"
        style="width:{Math.min(seg.gb / totalGb * 100, 100)}%; background:{seg.color}"
        title="{seg.label} — {seg.gb.toFixed(1)}GB"
      ></div>
    {/each}
    <!-- fallback: single bar when no detail -->
    {#if segments.length === 0 && totalGb}
      <div class="seg" style="width:{Math.min(pct,100)}%; background:var(--blue)"></div>
    {/if}
  </div>

  {#if segments.length}
    <ul class="legend">
      {#each segments as seg}
        <li><span class="dot" style="background:{seg.color}"></span>{seg.label} {seg.gb.toFixed(1)}GB</li>
      {/each}
    </ul>
  {/if}
</section>

<style>
  .vram-panel { padding: 1rem; }
  h2 { margin: 0 0 .5rem; font-size: 1rem; display: flex; align-items: baseline; gap: .5rem; }
  h2 { text-transform: uppercase; letter-spacing: .08em; opacity: .6; }
  .pct { font-weight: 700; opacity: 1; }
  .pct.over { color: var(--red); }
  .bar-track { height: 18px; background: var(--card); border-radius: 9px; overflow: hidden; display: flex; }
  .seg { height: 100%; transition: width .4s; }
  .legend { list-style: none; padding: 0; margin: .4rem 0 0; display: flex; gap: 1rem; flex-wrap: wrap; font-size: .75rem; }
  .dot { display: inline-block; width: 8px; height: 8px; border-radius: 50%; margin-right: 4px; }
</style>
