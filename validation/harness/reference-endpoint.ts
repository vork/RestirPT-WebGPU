// Dev-only Vite plugin: "Export for Cycles" render endpoint (plan §4.1 reference-endpoint.ts, §5 M3a).
//
//   POST /api/reference   content-type: application/json
//     { packageDir: "validation/out/<run>", spp: 16, seeds: "0..3" | [0,1,2,3], frames: "all" | [0, 5],
//       maxBounces?: 3, device?: "GPU" | "CPU", force?: false }
//   → 200 application/x-ndjson, one JSON object per line, streamed while Blender runs:
//       {"type":"start","cmd":[...],"lockHeld":false}
//       {"type":"lock","message":"..."}                         (another job holds the GPU lock, gpu-lock.ts)
//       {"type":"progress","done":1,"total":8,"file":"f0000_s000.exr","seconds":0.42}
//       {"type":"log","line":"..."}
//       {"type":"result","ok":true,"dir":"validation/out/refs-app/<name>-<key16>","cacheHit":false,
//        "exrs":[{"frame":0,"seed":0,"path":"validation/out/...exr","url":"/validation/out/...exr"}],"manifest":{...}}
//     or {"type":"error","message":"..."} as the last line.
//
// It spawns `Blender -b --factory-startup --python-exit-code 1 -P validation/blender/render_reference.py -- ...`.
// GPU lock (plan §7.5 Orchestration; gpu-lock.ts): render_reference.py takes the GPU lock itself around the renders;
// this process never holds it (a harness that holds it while waiting on this endpoint would deadlock — release it
// first, as tests/editor/e2e-editor.ts does). The app suspends its own frame loop while the request is in flight.
//
// Safety: registered only by `vite serve` (apply: 'serve'; production builds never contain it), loopback clients
// only, JSON content type required (no simple cross-site form posts), Origin (when present) must be a loopback
// origin, packageDir must resolve inside validation/out or validation/scenes and contain scene.json, one render at a
// time, bounded spp/seeds/frames, and the child is killed when the client disconnects.
import { spawn, type ChildProcess } from 'node:child_process';
import { existsSync, readFileSync } from 'node:fs';
import path from 'node:path';
import type { IncomingMessage, ServerResponse } from 'node:http';
import type { Plugin } from 'vite';
import { GPU_LOCK, gpuLockHolders } from './gpu-lock.ts';

export { GPU_LOCK };

export const BLENDER_DEFAULT = '/Applications/Blender.app/Contents/MacOS/Blender';
const MAX_BODY = 64 * 1024;
const PACKAGE_ROOTS = ['validation/out', 'validation/scenes'];

export interface ReferenceRequest {
  packageDir: string;
  spp: number;
  seeds: string | number[];
  frames?: 'all' | number[] | string;
  maxBounces?: number;
  device?: 'GPU' | 'CPU';
  force?: boolean;
}

export interface ReferenceOptions {
  /** Repository root (the dev server root). */
  root: string;
  /** Where render_reference.py writes (<outDir>/<name>-<key16>/). Default validation/out/refs-app. */
  outDir?: string;
  blender?: string;
  /** Spawn override (tests). */
  spawnFn?: typeof spawn;
}

export class ReferenceRequestError extends Error {}

const LOOPBACK = new Set(['127.0.0.1', '::1', '::ffff:127.0.0.1']);

export function isLoopbackAddress(a: string | undefined): boolean { return !!a && LOOPBACK.has(a); }

export function isLoopbackOrigin(origin: string | undefined): boolean {
  if (!origin) return true; // same-origin fetches from some browsers omit it; the JSON content type still blocks forms
  try {
    const u = new URL(origin);
    return (u.protocol === 'http:' || u.protocol === 'https:') && ['localhost', '127.0.0.1', '[::1]'].includes(u.hostname);
  } catch { return false; }
}

/** "0..3" / "0,2,5" / [0, 1] → validated render_reference.py --seeds string. */
export function normalizeSeeds(s: string | number[]): string {
  const list: number[] = [];
  const parts = Array.isArray(s) ? s.map(String) : String(s).split(',');
  for (const part of parts) {
    const p = part.trim();
    const m = /^(\d{1,3})\.\.(\d{1,3})$/.exec(p);
    if (m) { for (let i = Number(m[1]); i <= Number(m[2]); i++) list.push(i); }
    else if (/^\d{1,3}$/.test(p)) list.push(Number(p));
    else if (p) throw new ReferenceRequestError(`bad seeds '${p}'`);
  }
  if (!list.length || list.length > 64 || new Set(list).size !== list.length) throw new ReferenceRequestError('seeds: 1..64 distinct values in 0..999');
  return list.join(',');
}

export function normalizeFrames(f: ReferenceRequest['frames']): string {
  if (f === undefined || f === 'all') return 'all';
  const list = Array.isArray(f) ? f : String(f).split(',').map((x) => x.trim()).filter(Boolean).map(Number);
  if (!list.length || list.length > 1000 || list.some((k) => !Number.isInteger(k) || k < 0 || k > 1e6)) throw new ReferenceRequestError('frames: "all" or up to 1000 integers ≥ 0');
  return list.join(',');
}

/** Resolve and check a package directory (relative to the root, inside PACKAGE_ROOTS, with a scene.json). */
export function resolvePackageDir(root: string, dir: string): string {
  if (typeof dir !== 'string' || !dir || dir.includes('\0')) throw new ReferenceRequestError('packageDir missing');
  const rel = dir.replace(/^\/+/, '');
  const abs = path.resolve(root, rel);
  const ok = PACKAGE_ROOTS.some((r) => { const base = path.resolve(root, r) + path.sep; return abs.startsWith(base); });
  if (!ok) throw new ReferenceRequestError(`packageDir must be inside ${PACKAGE_ROOTS.join(' or ')}`);
  if (!existsSync(path.join(abs, 'scene.json'))) throw new ReferenceRequestError(`${rel}/scene.json not found`);
  return abs;
}

export function validateRequest(root: string, body: unknown): { args: string[]; pkg: string; total: (framesInPackage: number) => number; seeds: string; frames: string } {
  const r = body as ReferenceRequest;
  if (!r || typeof r !== 'object') throw new ReferenceRequestError('JSON object body required');
  const pkg = resolvePackageDir(root, r.packageDir);
  if (!Number.isInteger(r.spp) || r.spp < 1 || r.spp > 65536) throw new ReferenceRequestError('spp must be an integer in 1..65536');
  const seeds = normalizeSeeds(r.seeds ?? '0');
  const frames = normalizeFrames(r.frames);
  if (r.maxBounces !== undefined && (!Number.isInteger(r.maxBounces) || r.maxBounces < 0 || r.maxBounces > 1024)) throw new ReferenceRequestError('maxBounces must be an integer in 0..1024');
  if (r.device !== undefined && r.device !== 'GPU' && r.device !== 'CPU') throw new ReferenceRequestError('device must be GPU or CPU');
  const args = ['--package', pkg, '--spp', String(r.spp), '--seeds', seeds, '--frames', frames];
  if (r.maxBounces !== undefined) args.push('--max-bounces', String(r.maxBounces));
  if (r.device) args.push('--device', r.device);
  if (r.force) args.push('--force');
  const nSeeds = seeds.split(',').length;
  return { args, pkg, seeds, frames, total: (nf) => nSeeds * (frames === 'all' ? Math.max(1, nf) : frames.split(',').length) };
}

async function readJson(req: IncomingMessage): Promise<unknown> {
  const chunks: Buffer[] = [];
  let n = 0;
  for await (const c of req) {
    n += (c as Buffer).length;
    if (n > MAX_BODY) throw new ReferenceRequestError('body too large');
    chunks.push(c as Buffer);
  }
  try { return JSON.parse(Buffer.concat(chunks).toString('utf8')); } catch { throw new ReferenceRequestError('body is not JSON'); }
}

function fail(res: ServerResponse, code: number, msg: string): void {
  res.statusCode = code;
  res.setHeader('content-type', 'text/plain');
  res.end(msg);
}

export function referenceEndpointPlugin(opts: ReferenceOptions): Plugin {
  const root = opts.root;
  const outDir = path.resolve(root, opts.outDir ?? 'validation/out/refs-app');
  const blender = opts.blender ?? process.env.BLENDER ?? BLENDER_DEFAULT;
  const spawnFn = opts.spawnFn ?? spawn;
  let busy: ChildProcess | undefined;
  let inFlight = false; // set synchronously: a second request in the same tick is refused too

  return {
    name: 'restir-reference-endpoint',
    apply: 'serve',
    configureServer(server) {
      server.middlewares.use('/api/reference', (req, res, next) => {
        const route = new URL(req.url ?? '/', 'http://localhost').pathname.replace(/\/+$/, '');
        if (route !== '' && route !== '/') return next();
        if (req.method === 'GET') {
          res.setHeader('content-type', 'application/json');
          res.end(JSON.stringify({ available: existsSync(blender), blender, busy: inFlight, lockHeld: existsSync(GPU_LOCK) }));
          return;
        }
        if (req.method !== 'POST') return fail(res, 405, 'POST only');
        if (!isLoopbackAddress(req.socket.remoteAddress)) return fail(res, 403, 'loopback clients only');
        if (!isLoopbackOrigin(req.headers.origin)) return fail(res, 403, 'cross-origin request refused');
        if (!(req.headers['content-type'] ?? '').startsWith('application/json')) return fail(res, 415, 'application/json required');
        if (inFlight) return fail(res, 409, 'a reference render is already running');
        inFlight = true;
        void (async () => {
          let v: ReturnType<typeof validateRequest>;
          try { v = validateRequest(root, await readJson(req)); } catch (e) {
            inFlight = false;
            return fail(res, e instanceof ReferenceRequestError ? 400 : 500, e instanceof Error ? e.message : String(e));
          }
          if (!existsSync(blender)) { inFlight = false; return fail(res, 503, `Blender not found at ${blender} (set BLENDER)`); }
          let nFrames = 1;
          try { nFrames = (JSON.parse(readFileSync(path.join(v.pkg, 'scene.json'), 'utf8')) as { frames?: unknown[] }).frames?.length ?? 1; } catch { /* checked by Blender */ }
          const total = v.total(nFrames);
          res.statusCode = 200;
          res.setHeader('content-type', 'application/x-ndjson');
          res.setHeader('cache-control', 'no-store');
          const send = (o: object) => { if (!res.writableEnded) res.write(`${JSON.stringify(o)}\n`); };
          const cmd = ['-b', '--factory-startup', '--python-exit-code', '1', '-P', path.join(root, 'validation/blender/render_reference.py'), '--',
            ...v.args, '--out', outDir];
          const lockHeld = existsSync(GPU_LOCK);
          send({ type: 'start', cmd: [blender, ...cmd], total, lockHeld });
          if (lockHeld) {
            let by = '';
            try { by = gpuLockHolders(GPU_LOCK)?.map((h) => h.name).join(', ') ?? ''; } catch { /* unreadable: generic message */ }
            by ||= 'another job';
            send({ type: 'lock', message: `${GPU_LOCK} is held by ${by}; Blender waits for it (polls every 5 s)` });
          }
          const child = spawnFn(blender, cmd, { cwd: root, stdio: ['ignore', 'pipe', 'pipe'] });
          busy = child;
          let done = 0;
          let result: { dir: string; cache_hit: boolean; key: string; renders: number } | undefined;
          const tail: string[] = [];
          const onLine = (line: string) => {
            if (!line) return;
            tail.push(line);
            if (tail.length > 40) tail.shift();
            const r = /^\[render_reference\] RESULT (.*)$/.exec(line);
            if (r) { try { result = JSON.parse(r[1]); } catch { /* reported below */ } return; }
            const p = /^\[render_reference\] (f\d{4}_s\d{3}\.exr) seed=\d+ ([\d.]+) s/.exec(line);
            if (p) { done++; send({ type: 'progress', done, total, file: p[1], seconds: Number(p[2]) }); return; }
            if (line.startsWith('[render_reference]')) send({ type: /GPU lock/.test(line) ? 'lock' : 'log', line, message: line });
          };
          const pipeLines = (s: NodeJS.ReadableStream) => {
            let buf = '';
            s.setEncoding('utf8');
            s.on('data', (d: string) => { buf += d; let i; while ((i = buf.indexOf('\n')) >= 0) { onLine(buf.slice(0, i).trimEnd()); buf = buf.slice(i + 1); } });
            s.on('end', () => { if (buf) onLine(buf.trimEnd()); });
          };
          pipeLines(child.stdout!);
          pipeLines(child.stderr!);
          const onClose = () => { if (busy === child && child.exitCode === null) child.kill('SIGTERM'); };
          res.on('close', onClose);
          child.on('error', (e) => { send({ type: 'error', message: `spawn failed: ${e.message}` }); });
          child.on('close', (code) => {
            busy = undefined;
            inFlight = false;
            res.off('close', onClose);
            if (code === 0 && result) {
              const dir = result.dir;
              let manifest: { renders?: { frame: number; seed: number; file: string; mean?: number[] }[] } | undefined;
              try { manifest = JSON.parse(readFileSync(path.join(dir, 'manifest.json'), 'utf8')); } catch { /* */ }
              const rel = path.relative(root, dir).split(path.sep).join('/');
              const exrs = (manifest?.renders ?? []).map((r) => ({ frame: r.frame, seed: r.seed, path: `${rel}/${r.file}`, url: `/${rel}/${r.file}`, mean: r.mean }));
              send({ type: 'result', ok: true, dir: rel, cacheHit: result.cache_hit, key: result.key, exrs, manifest });
            } else {
              send({ type: 'error', message: `Blender exited with ${code}${result ? '' : ' (no RESULT line)'}`, tail });
            }
            res.end();
          });
        })().catch((e: unknown) => { busy = undefined; inFlight = false; if (!res.headersSent) fail(res, 500, String(e)); else res.end(); });
      });
    },
  };
}
