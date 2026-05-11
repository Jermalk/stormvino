<script>
  /** @type {{ pct: number, color?: string }} */
  let { pct = 0, color = null } = $props()
  const clamped = $derived(Math.max(0, Math.min(100, pct)))
  const auto = $derived(!color ? (pct < 60 ? 'var(--green)' : pct < 85 ? 'var(--yellow)' : 'var(--red)') : color)
</script>

<div class="bar-wrap">
  <div class="bar-track">
    <div class="bar-fill" style="width:{clamped}%; background:{auto}"></div>
  </div>
  <span class="pct" style="color:{auto}">{pct.toFixed(0)}%</span>
</div>

<style>
  .bar-wrap  { display: flex; align-items: center; gap: .4rem; }
  .bar-track { flex: 1; height: 10px; background: #ffffff12; border-radius: 5px; overflow: hidden; }
  .bar-fill  { height: 100%; border-radius: 5px; transition: width .4s; }
  .pct       { font-size: .75rem; min-width: 3rem; text-align: right; font-variant-numeric: tabular-nums; }
</style>
