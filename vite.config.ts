import { fileURLToPath } from 'node:url';
import { defineConfig } from 'vite';
import { harnessUploadPlugin } from './validation/harness/upload-middleware.ts';

const root = fileURLToPath(new URL('.', import.meta.url));

// Multi-page: the app shell and the validation harness page (plan §4.1). The upload middleware is dev-only.
export default defineConfig({
  server: { port: 5173, strictPort: false },
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
