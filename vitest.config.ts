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
        test: {
          name: 'chrome',
          include: ['validation/gpu-tests/**/*.gpu.test.ts'],
          testTimeout: 120_000,
          browser: {
            enabled: true,
            headless: true,
            provider: playwright({
              launchOptions: { channel: 'chrome', args: ['--enable-unsafe-webgpu'] },
            }),
            instances: [{ browser: 'chromium' }],
          },
        },
      },
    ],
  },
});
