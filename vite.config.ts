import { fileURLToPath } from 'node:url';
import { defineConfig } from 'vite';
import { harnessUploadPlugin } from './validation/harness/upload-middleware.ts';

const root = fileURLToPath(new URL('.', import.meta.url));

// Multi-page: the app shell and the validation harness page (plan §4.1). The upload middleware is dev-only.
// The dev server's root is the repo, so validation/assets/** (Cornell, downloaded Sponza + HDRIs) are served as-is:
// the app loads /validation/assets/cornell/cornell.glb by default and accepts ?scene=/validation/assets/...&env=...
// Loader dependencies are pre-bundled up front: they are first reached from Workers / dynamic imports, which the
// dependency scanner does not see, and a late discovery reloads the page mid-run (Playwright smoke, Chrome lane).
export default defineConfig({
  server: { port: 5173, strictPort: false },
  optimizeDeps: {
    include: [
      '@gltf-transform/core', '@gltf-transform/extensions', '@gltf-transform/functions', 'meshoptimizer/decoder', 'draco3dgltf',
      'three', 'three/examples/jsm/loaders/EXRLoader.js', 'tweakpane',
    ],
  },
  plugins: [harnessUploadPlugin(`${root}validation/out`)],
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
