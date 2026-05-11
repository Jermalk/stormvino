<script>
  import BarMeter from './BarMeter.svelte'

  let { sys } = $props()
  const gpu = $derived(sys?.gpu ?? {})

  const ENGINE_LABELS = {
    rcs:  'Render  (rcs)',
    ccs:  'Compute (ccs)',
    vcs:  'Video   (vcs)',
    vecs: 'VideoEnh(vecs)',
    bcs:  'Blitter (bcs)',
  }

  const engines = $derived(
    Object.entries(ENGINE_LABELS)
      .filter(([k]) => k in (gpu.engine_pct ?? {}))
      .map(([k, label]) => ({ key: k, label, pct: gpu.engine_pct[k] }))
  )

  const vramUsedMib  = $derived(gpu.vram_used_mib  ?? 0)
  const vramTotalMib = $derived(gpu.vram_total_mib ?? 24480)
  const vramPct      = $derived(vramTotalMib ? vramUsedMib / vramTotalMib * 100 : 0)

  function tempColor(t) {
    return t == null ? 'inherit' : t < 70 ? 'var(--green)' : t < 85 ? 'var(--yellow)' : 'var(--red)'
  }
  function fanColor(rpm) {
    return rpm == null ? 'inherit' : rpm < 1500 ? 'var(--green)' : rpm < 2500 ? 'var(--yellow)' : 'var(--red)'
  }
</script>

<section class="panel">
  <h2>Intel Arc B60</h2>
  {#if !sys}
    <p class="dim">waiting…</p>
  {:else}
    {#if engines.length}
      <div class="group">
        {#each engines as e}
          <div class="row">
            <span class="label">{e.label}</span>
            <BarMeter pct={e.pct} />
          </div>
        {/each}
      </div>
    {/if}

    <div class="group">
      <div class="row">
        <span class="label">VRAM raw</span>
        <BarMeter pct={vramPct} color="var(--blue)" />
      </div>
      <div class="row sub">
        <span class="label"></span>
        <span>{vramUsedMib.toLocaleString()} / {vramTotalMib.toLocaleString()} MiB</span>
      </div>
    </div>

    <div class="group specs">
      {#if gpu.temp_gt_c != null}
        <div class="spec"><span class="label">GT temp</span>
          <span style="color:{tempColor(gpu.temp_gt_c)}">{gpu.temp_gt_c} °C</span>
        </div>
      {/if}
      {#if gpu.temp_mem_c != null}
        <div class="spec"><span class="label">VRAM temp</span>
          <span style="color:{tempColor(gpu.temp_mem_c)}">{gpu.temp_mem_c} °C</span>
        </div>
      {/if}
      {#if gpu.fan_rpm != null}
        <div class="spec"><span class="label">Fan</span>
          <span style="color:{fanColor(gpu.fan_rpm)}">{gpu.fan_rpm} RPM</span>
        </div>
      {/if}
      {#if gpu.power_w != null}
        <div class="spec"><span class="label">Power</span>
          <span>{gpu.power_w} W{gpu.power_cap_w ? ` / ${gpu.power_cap_w} W cap` : ''}</span>
        </div>
      {/if}
    </div>
  {/if}
</section>

<style>
  .panel { padding: .75rem 1rem; height: 100%; }
  h2 { font-size: .7rem; text-transform: uppercase; letter-spacing: .08em; opacity: .45; margin-bottom: .6rem; }
  .dim { opacity: .35; font-size: .82rem; }
  .group { margin-bottom: .6rem; }
  .row  { display: flex; align-items: center; gap: .5rem; margin-bottom: .2rem; font-size: .82rem; }
  .row.sub { opacity: .5; font-size: .75rem; }
  .label { width: 9rem; flex-shrink: 0; opacity: .5; font-size: .78rem; white-space: nowrap; }
  .specs { display: grid; grid-template-columns: 1fr 1fr; gap: .2rem .5rem; }
  .spec  { display: flex; flex-direction: column; font-size: .82rem; }
  .spec .label { width: auto; font-size: .68rem; margin-bottom: .05rem; }
</style>
