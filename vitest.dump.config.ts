// Local (untracked): the chrome lane with Dawn's dump_shaders / disable_symbol_renaming toggles and Chrome logging to
// stderr (surfaced by DEBUG=pw:browser*), to capture the generated MSL.
import { defineConfig } from 'vitest/config';
import { playwright } from '@vitest/browser-playwright';
export default defineConfig({
  cacheDir: '.vite',
  test: {
    projects: [{
      test: {
        name: 'chrome', include: ['validation/gpu-tests/**/*.gpu.test.ts'], testTimeout: 120_000,
        browser: {
          enabled: true, headless: true, instances: [{ browser: 'chromium' }],
          provider: playwright({ launchOptions: { channel: 'chrome', args: ['--enable-dawn-features=dump_shaders,disable_symbol_renaming', '--enable-logging=stderr', '--v=0'] } }),
        },
      },
    }],
  },
});
