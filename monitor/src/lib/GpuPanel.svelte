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

  const vramUsedMib  = $derived(gpu.vram_used_mib  ?? 0)
  const vramTotalMib = $derived(gpu.vram_total_mib ?? 24480)
  const vramPct      = $derived(vramTotalMib ? vramUsedMib / vramTotalMib * 100 : 0)

  function tempColor(t) {
    return t == null ? 'var(--text)' : t < 70 ? 'var(--green)' : t < 85 ? 'var(--yellow)' : 'var(--red)'
  }
  function fanColor(rpm) {
    return rpm == null ? 'var(--text)' : rpm < 1500 ? 'var(--green)' : rpm < 2500 ? 'var(--yellow)' : 'var(--red)'
  }

</script>

<section class="panel">
  <h2>Intel Arc B60</h2>
  {#if !sys}
    <p class="dim">waiting…</p>
  {:else}
    <div class="group">
      {#each Object.entries(ENGINE_LABELS) as [k, label]}
        <div class="row">
          <span class="label">{label}</span>
          <BarMeter pct={gpu.engine_pct?.[k] ?? 0} />
        </div>
      {/each}
    </div>

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

    <div class="group">
      <div class="row">
        <span class="label">GT temp</span>
        <BarMeter pct={gpu.temp_gt_c != null ? gpu.temp_gt_c / 105 * 100 : 0}
                  color={tempColor(gpu.temp_gt_c)}
                  label={gpu.temp_gt_c != null ? `${gpu.temp_gt_c} °C` : '—'} />
      </div>
      <div class="row">
        <span class="label">VRAM temp</span>
        <BarMeter pct={gpu.temp_mem_c != null ? gpu.temp_mem_c / 105 * 100 : 0}
                  color={tempColor(gpu.temp_mem_c)}
                  label={gpu.temp_mem_c != null ? `${gpu.temp_mem_c} °C` : '—'} />
      </div>
      <div class="row">
        <span class="label">Fan</span>
        <BarMeter pct={gpu.fan_rpm != null ? Math.min(gpu.fan_rpm / 3500 * 100, 100) : 0}
                  color={fanColor(gpu.fan_rpm)}
                  label={gpu.fan_rpm != null ? `${gpu.fan_rpm} RPM` : '—'} />
      </div>
      <div class="row">
        <span class="label">Power</span>
        <BarMeter pct={gpu.power_w != null ? Math.min(gpu.power_w / (gpu.power_cap_w ?? 225) * 100, 100) : 0}
                  label={gpu.power_w != null ? `${gpu.power_w} W${gpu.power_cap_w ? ` / ${gpu.power_cap_w} W` : ''}` : '—'} />
      </div>
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
</style>
