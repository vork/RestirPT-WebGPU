// Only for the stock LightUSDWorkerLoader / LightUSDLoader modes of the spike (run-spike.ts pass B):
//   npx vite validation/spikes/usd --port 5190 --config validation/spikes/usd/vite.lightusd-stock.config.ts
// The 'direct' mode (own worker over lightusd_next.js) runs with Vite's default config (no config file).
import { fileURLToPath } from 'node:url';
import { defineConfig } from 'vite';

export default defineConfig({
  root: fileURLToPath(new URL('.', import.meta.url)),
  // Pre-bundling moves LightUSDWorkerLoader.js into .vite/deps, which breaks its
  // new URL('./LightUSDWorker.js', import.meta.url) and the emscripten wasm lookups.
  optimizeDeps: { exclude: ['lightusd'] },
  resolve: { alias: { fzstd: fileURLToPath(new URL('./fzstd-stub.ts', import.meta.url)) } },
});
