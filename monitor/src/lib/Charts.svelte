<script>
  import { onMount, onDestroy } from 'svelte'
  import uPlot from 'uplot'
  import 'uplot/dist/uPlot.min.css'
  import { fetchMetrics } from './api.js'

  let container
  let plot
  let timer

  const METRICS = [
    { key: 'tok_per_sec', label: 'tok/s',       color: '#4e9af1' },
    { key: 'elapsed_sec', label: 'elapsed (s)',  color: '#f1a14e' },
  ]
  let activeMetric = $state(METRICS[0])

  async function load() {
    const data = await fetchMetrics(activeMetric.key, 60).catch(() => null)
    if (!data || !data.ts?.length) return

    const opts = {
      width:  container.clientWidth,
      height: 200,
      scales: { x: { time: true }, y: { auto: true } },
      series: [
        {},
        { label: activeMetric.label, stroke: activeMetric.color, width: 2, fill: activeMetric.color + '22' },
      ],
      axes: [{ }, { size: 50 }],
    }

    if (plot) { plot.destroy(); plot = null }
    plot = new uPlot(opts, [data.ts, data.values], container)
  }

  onMount(() => {
    load()
    timer = setInterval(load, 30_000)
  })

  onDestroy(() => {
    clearInterval(timer)
    plot?.destroy()
  })
</script>

<section class="charts-panel">
  <div class="toolbar">
    <h2>History (60 min)</h2>
    <div class="tabs">
      {#each METRICS as m}
        <button class:active={activeMetric.key === m.key} onclick={() => { activeMetric = m; load() }}>
          {m.label}
        </button>
      {/each}
    </div>
  </div>
  <div bind:this={container} class="chart-container"></div>
</section>

<style>
  .charts-panel { padding: 1rem; }
  .toolbar { display: flex; align-items: center; gap: 1rem; margin-bottom: .5rem; }
  h2 { margin: 0; font-size: 1rem; text-transform: uppercase; letter-spacing: .08em; opacity: .6; }
  .tabs { display: flex; gap: .25rem; }
  button { background: var(--card); border: none; border-radius: 4px; padding: .25rem .6rem; cursor: pointer; font-size: .8rem; color: inherit; opacity: .6; }
  button.active { opacity: 1; background: var(--blue); color: #fff; }
  .chart-container { min-height: 200px; }
</style>
