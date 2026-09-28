// Dev-only Vite plugin: lets harness pages write results into validation/out/<run>/ (plan §4.1, M0).
//   POST /__harness/upload?run=<id>&name=<file>  raw body -> validation/out/<run>/<name>   (204)
//   POST /__harness/log?run=<id>                  JSON body -> appended line in validation/out/<run>/log.jsonl (204)
import { createWriteStream } from 'node:fs';
import { appendFile, mkdir, rename } from 'node:fs/promises';
import path from 'node:path';
import { pipeline } from 'node:stream/promises';
import type { IncomingMessage, ServerResponse } from 'node:http';
import type { Plugin } from 'vite';

const SAFE = /^[A-Za-z0-9._-]{1,128}$/;
const MAX_LOG_BYTES = 1 << 20;

/** Returns the basename if it is a safe single path component, otherwise null. */
export function sanitizeComponent(s: string | null): string | null {
  if (!s) return null;
  const b = path.basename(s);
  if (b !== s || !SAFE.test(b) || b === '.' || b === '..') return null;
  return b;
}

async function readBody(req: IncomingMessage, limit: number): Promise<Buffer> {
  const chunks: Buffer[] = [];
  let n = 0;
  for await (const c of req) {
    n += (c as Buffer).length;
    if (n > limit) throw new Error(`body exceeds ${limit} bytes`);
    chunks.push(c as Buffer);
  }
  return Buffer.concat(chunks);
}

function fail(res: ServerResponse, code: number, msg: string): void {
  res.statusCode = code;
  res.setHeader('content-type', 'text/plain');
  res.end(msg);
}

export function harnessUploadPlugin(outRoot: string): Plugin {
  return {
    name: 'restir-harness',
    apply: 'serve',
    configureServer(server) {
      server.middlewares.use('/__harness', (req, res, next) => {
        const url = new URL(req.url ?? '/', 'http://localhost');
        const route = url.pathname.replace(/\/+$/, '');
        if (route !== '/upload' && route !== '/log') return next();
        if (req.method !== 'POST') return fail(res, 405, 'POST only');
        const run = sanitizeComponent(url.searchParams.get('run'));
        if (!run) return fail(res, 400, 'bad run id');
        const name = route === '/upload' ? sanitizeComponent(url.searchParams.get('name')) : 'log.jsonl';
        if (!name) return fail(res, 400, 'bad file name');
        const dir = path.join(outRoot, run);

        void (async () => {
          await mkdir(dir, { recursive: true });
          if (route === '/upload') {
            const dst = path.join(dir, name);
            const tmp = `${dst}.part`;
            // Stream to a temp file and rename so readers never see a partial upload.
            await pipeline(req, createWriteStream(tmp));
            await rename(tmp, dst);
          } else {
            const body = await readBody(req, MAX_LOG_BYTES);
            const parsed: unknown = JSON.parse(body.toString('utf8'));
            await appendFile(path.join(dir, name), JSON.stringify({ t: Date.now(), ...(parsed as object) }) + '\n');
          }
          res.statusCode = 204;
          res.end();
        })().catch((e: unknown) => {
          if (!res.headersSent) fail(res, 500, String(e));
        });
      });
    },
  };
}
