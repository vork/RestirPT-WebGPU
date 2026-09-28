import { defineConfig } from 'vitest/config';
import { playwright } from '@vitest/browser-playwright';

// Three lanes (plan §1.1): CPU tests, dawn.node fast pre-check, and Chrome 154 (authoritative for gates).
export default defineConfig({
  test: {
    projects: [
      {
        test: { name: 'cpu', environment: 'node', include: ['tests/**/*.test.ts'] },
      },
      {
        test: {
          name: 'node-dawn',
          environment: 'node',
          include: ['validation/gpu-tests/**/*.gpu.test.ts'],
          testTimeout: 120_000,
          // dawn.node keeps the process alive while a GPU object is referenced; one worker avoids contention.
          pool: 'forks',
          maxWorkers: 1,
        },
      },
      {
        // Pre-bundle the loader dependencies (reached from Workers / dynamic imports) so a cold cache does not make
        // Vite reload the page in the middle of a test ("Vite unexpectedly reloaded a test").
        optimizeDeps: {
          include: [
            '@gltf-transform/core', '@gltf-transform/extensions', '@gltf-transform/functions', 'meshoptimizer', 'meshoptimizer/decoder',
            'draco3dgltf', 'three', 'three/examples/jsm/loaders/EXRLoader.js',
          ],
        },
        test: {
          name: 'chrome',
          include: ['validation/gpu-tests/**/*.gpu.test.ts'],
          testTimeout: 120_000,
          browser: {
            enabled: true,
            headless: true,
            provider: playwright({
              // No --enable-unsafe-webgpu: headless Chrome 154 exposes Metal WebGPU without it (M0 smoke).
              launchOptions: { channel: 'chrome' },
            }),
            instances: [{ browser: 'chromium' }],
          },
        },
      },
    ],
  },
});
