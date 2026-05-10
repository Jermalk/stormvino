import { defineConfig } from 'vite'
import { svelte } from '@sveltejs/vite-plugin-svelte'

export default defineConfig({
  plugins: [svelte()],
  base: '/monitor/',
  build: {
    outDir: 'dist',
    emptyOutDir: true,
  },
  server: {
    proxy: {
      '/health':          'http://localhost:11435',
      '/monitor/api':     'http://localhost:11435',
      '/v1/models':       'http://localhost:11435',
    }
  }
})
