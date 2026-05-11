<script>
  import BarMeter from './BarMeter.svelte'

  let { sys } = $props()
  const cpu = $derived(sys?.cpu ?? null)
  const mem = $derived(sys?.memory ?? null)

  const topTemps = $derived(
    Object.entries(cpu?.temps ?? {})
      .sort(([,a],[,b]) => b - a)
      .slice(0, 4)
  )

  function tempColor(t) {
    return t < 70 ? 'var(--green)' : t < 90 ? 'var(--yellow)' : 'var(--red)'
  }
</script>

<section class="panel">
  <h2>CPU &amp; Memory</h2>
  {#if !sys}
    <p class="dim">waiting…</p>
  {:else}
    <div class="group">
      <div class="row"><span class="label">CPU overall</span><BarMeter pct={cpu?.percent ?? 0} /></div>
      {#if (cpu?.per_core ?? []).length}
        <div class="cores">
          {#each cpu.per_core as pct, i}
            <div class="core-cell" title="Core {i}: {pct.toFixed(0)}%">
              <div class="core-bar" style="height:{pct}%; background:{pct < 60 ? 'var(--green)' : pct < 85 ? 'var(--yellow)' : 'var(--red)'}"></div>
            </div>
          {/each}
        </div>
      {/if}
      {#if cpu?.freq_ghz}
        <div class="row sub"><span class="label">Freq</span><span>{cpu.freq_ghz} GHz (max {cpu.freq_max_ghz})</span></div>
      {/if}
      {#if cpu?.load_avg?.length}
        <div class="row sub"><span class="label">Load avg</span>
          <span>{cpu.load_avg.map(v => v.toFixed(2)).join('  ')} (1/5/15m)</span>
        </div>
      {/if}
      {#each topTemps as [label, t]}
        <div class="row sub">
          <span class="label">{label.slice(0,14)}</span>
          <span style="color:{tempColor(t)}">{t.toFixed(0)} °C</span>
        </div>
      {/each}
    </div>

    {#if mem}
      <div class="group">
        <div class="row"><span class="label">RAM</span><BarMeter pct={mem.ram_pct} /></div>
        <div class="row sub"><span class="label"></span><span>{mem.ram_used_gb} / {mem.ram_total_gb} GB · {mem.ram_avail_gb} GB free</span></div>
        <div class="row"><span class="label">Swap</span><BarMeter pct={mem.swap_pct} /></div>
        <div class="row sub"><span class="label"></span><span>{mem.swap_used_gb} / {mem.swap_total_gb} GB</span></div>
      </div>
    {/if}
  {/if}
</section>

<style>
  .panel { padding: .75rem 1rem; height: 100%; }
  h2 { font-size: .7rem; text-transform: uppercase; letter-spacing: .08em; opacity: .45; margin-bottom: .6rem; }
  .dim { opacity: .35; font-size: .82rem; }
  .group { margin-bottom: .7rem; }
  .row { display: flex; align-items: center; gap: .5rem; margin-bottom: .25rem; font-size: .82rem; }
  .row.sub { opacity: .6; font-size: .75rem; }
  .label { width: 7rem; flex-shrink: 0; opacity: .5; font-size: .78rem; white-space: nowrap; }
  .cores {
    display: flex; gap: 2px; align-items: flex-end;
    height: 28px; margin: .3rem 0 .3rem 7.5rem;
    background: #ffffff08; border-radius: 3px; padding: 2px; overflow: hidden;
  }
  .core-cell { flex: 0 0 7px; height: 100%; display: flex; align-items: flex-end; }
  .core-bar  { width: 100%; border-radius: 1px; min-height: 2px; transition: height .4s; }
</style>
