import { realpathSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { defineConfig } from 'vite';
import { referenceEndpointPlugin } from './validation/harness/reference-endpoint.ts';
import { harnessUploadPlugin } from './validation/harness/upload-middleware.ts';

const root = fileURLToPath(new URL('.', import.meta.url));
// Worktrees symlink node_modules into the main checkout: allow its real path too, or Worker-side `?url` imports
// (mikktspace wasm) are refused by the dev server's fs.allow check.
const nodeModulesReal = realpathSync(`${root}node_modules`);

// Multi-page: the app shell and the validation harness page (plan §4.1). The upload middleware is dev-only.
// The dev server's root is the repo, so validation/assets/** (Cornell, downloaded Sponza + HDRIs) are served as-is:
// the app loads /validation/assets/cornell/cornell.glb by default and accepts ?scene=/validation/assets/...&env=...
// Loader dependencies are pre-bundled up front: they are first reached from Workers / dynamic imports, which the
// dependency scanner does not see, and a late discovery reloads the page mid-run (Playwright smoke, Chrome lane).
export default defineConfig({
  // Local, not node_modules/.vite: worktrees symlink node_modules to the main checkout, and a shared deps cache that
  // another checkout re-optimizes reloads pages mid-run.
  cacheDir: '.vite',
  server: { port: 5173, strictPort: false, fs: { allow: [root, nodeModulesReal] } },
  optimizeDeps: {
    include: [
      '@gltf-transform/core', '@gltf-transform/extensions', '@gltf-transform/functions', 'meshoptimizer/decoder', 'draco3dgltf',
      'three', 'three/examples/jsm/loaders/EXRLoader.js', 'tweakpane',
    ],
  },
  // Both plugins are dev-only (apply: 'serve'); /api/reference spawns headless Blender for "Export for Cycles".
  plugins: [harnessUploadPlugin(`${root}validation/out`), referenceEndpointPlugin({ root })],
  build: {
    target: 'es2023',
    rolldownOptions: {
      input: {
        main: `${root}index.html`,
        harness: `${root}validation/harness/harness.html`,
      },
    },
  },
});
