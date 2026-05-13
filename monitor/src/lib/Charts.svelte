<script>
  import { onMount, onDestroy } from 'svelte'
  import uPlot from 'uplot'
  import 'uplot/dist/uPlot.min.css'
  import { fetchMetrics } from './api.js'

  const METRICS = [
    { key: 'tok_per_sec',       label: 'tok/s',      unit: 'tok/s', color: '#4e9af1' },
    { key: 'elapsed_sec',       label: 'duration',   unit: 's',     color: '#f1a14e' },
    { key: 'completion_tokens', label: 'out tokens', unit: 'tok',   color: '#4ef1a0' },
    { key: 'prompt_tokens',     label: 'in tokens',  unit: 'tok',   color: '#9b6ef3' },
    { key: 'vram_used_gb',      label: 'VRAM',       unit: 'GB',    color: '#f7c44e' },
    { key: 'ram_used_pct',      label: 'RAM %',      unit: '%',     color: '#f1544e' },
  ]
  const RANGES = [
    { minutes: 15,   label: '15m' },
    { minutes: 60,   label: '1h'  },
    { minutes: 360,  label: '6h'  },
    { minutes: 1440, label: '24h' },
  ]

  let activeMetric = $state(METRICS[0])
  let activeRange  = $state(RANGES[1])
  let container    = $state(null)
  let plot         = null
  let timer        = null
  let isEmpty      = $state(true)
  let loading      = $state(false)
  let loadSeq      = 0

  async function load() {
    if (!container) return
    const seq = ++loadSeq
    loading = true
    try {
      const data = await fetchMetrics(activeMetric.key, activeRange.minutes)
      if (seq !== loadSeq) return  // stale — a newer load is in flight
      loading = false
      isEmpty = !data.ts?.length
      if (plot) { plot.destroy(); plot = null }
      if (!data.ts?.length) return

      const hasDual = data.model_counts?.length > 0
      const uData   = hasDual
        ? [data.ts, data.values, data.model_counts]
        : [data.ts, data.values]

      const opts = {
        width:  container.clientWidth || 600,
        height: 180,
        scales: hasDual
          ? { x: { time: true }, y: { auto: true }, y2: { auto: true } }
          : { x: { time: true }, y: { auto: true } },
        series: hasDual
          ? [
              {},
              {
                label:  activeMetric.unit,
                stroke: activeMetric.color,
                width:  1.5,
                fill:   activeMetric.color + '18',
                scale:  'y',
                points: { show: false },
              },
              {
                label:  'models',
                stroke: '#4ef1a070',
                width:  1,
                scale:  'y2',
                paths:  uPlot.paths.stepped({ align: -1 }),
                points: { show: false },
              },
            ]
          : [
              {},
              {
                label:  activeMetric.unit,
                stroke: activeMetric.color,
                width:  1.5,
                fill:   activeMetric.color + '18',
                points: { show: data.ts.length < 60 },
              },
            ],
        axes: hasDual
          ? [
              { stroke: '#ffffff40', ticks: { stroke: '#ffffff15' }, grid: { stroke: '#ffffff08' } },
              { size: 52, stroke: '#ffffff40', ticks: { stroke: '#ffffff15' }, grid: { stroke: '#ffffff08' } },
              { scale: 'y2', side: 1, size: 28, stroke: '#4ef1a060',
                ticks: { show: false }, grid: { show: false },
                values: (_, vals) => vals.map(v => v != null ? String(Math.round(v)) : '') },
            ]
          : [
              { stroke: '#ffffff40', ticks: { stroke: '#ffffff15' }, grid: { stroke: '#ffffff08' } },
              { size: 52, stroke: '#ffffff40', ticks: { stroke: '#ffffff15' }, grid: { stroke: '#ffffff08' } },
            ],
        cursor: { stroke: '#ffffff30', width: 1 },
      }
      plot = new uPlot(opts, uData, container)
    } catch {
      loading = false
    }
  }

  function resize() {
    if (plot && container) plot.setSize({ width: container.clientWidth, height: 180 })
  }

  let ro
  onMount(() => {
    load()
    timer = setInterval(load, 30_000)
    ro = new ResizeObserver(resize)
    if (container) ro.observe(container)
  })
  onDestroy(() => {
    clearInterval(timer)
    ro?.disconnect()
    plot?.destroy()
  })

  function selectMetric(m) { activeMetric = m; load() }
  function selectRange(r)  { activeRange  = r; load() }
</script>

<section class="charts-panel">
  <div class="toolbar">
    <h2>History</h2>
    <div class="tabs">
      {#each METRICS as m}
        <button class:active={activeMetric.key === m.key} onclick={() => selectMetric(m)}>
          {m.label}
        </button>
      {/each}
    </div>
    <div class="tabs range-tabs">
      {#each RANGES as r}
        <button class:active={activeRange.minutes === r.minutes} onclick={() => selectRange(r)}>
          {r.label}
        </button>
      {/each}
    </div>
  </div>

  <div bind:this={container} class="chart-container">
    {#if loading}
      <div class="overlay dim">loading…</div>
    {:else if isEmpty}
      <div class="overlay dim">no data for this period</div>
    {/if}
  </div>
</section>

<style>
  .charts-panel { padding: .75rem 1rem; }
  .toolbar {
    display: flex; align-items: center; gap: .5rem;
    flex-wrap: wrap; margin-bottom: .5rem;
  }
  h2 {
    font-size: .7rem; text-transform: uppercase; letter-spacing: .08em;
    opacity: .45; white-space: nowrap; margin-right: .25rem;
  }
  .tabs { display: flex; gap: .2rem; flex-wrap: wrap; }
  .range-tabs { margin-left: auto; }
  button {
    background: var(--card); border: 1px solid #ffffff0a;
    border-radius: 4px; padding: .2rem .55rem;
    cursor: pointer; font-size: .75rem; color: inherit; opacity: .55;
    transition: opacity .12s, background .12s;
  }
  button:hover  { opacity: .85; }
  button.active { opacity: 1; background: #ffffff12; border-color: #ffffff20; }
  .chart-container { min-height: 200px; position: relative; }
  .overlay {
    position: absolute; inset: 0;
    display: flex; align-items: center; justify-content: center;
    font-size: .8rem; pointer-events: none;
  }
  .dim { opacity: .3; }
  :global(.uplot) { width: 100% !important; }
</style>
