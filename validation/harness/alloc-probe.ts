// M0 allocation probe (plan §1.9, §5 M0): can Chrome hold the validation-layout buffers of an ensemble atlas
// (E members x 256² px x ~540 B/px) on the target Mac? Every buffer is created inside 'out-of-memory' and
// 'validation' error scopes, and residency is proven by writing a pattern at its start and end and reading it back.
import { readBuffer } from '../../src/core/gpu/readback.ts';

/** Bytes per pixel of the validation layout, split the way the renderer will split its arenas (sum = 540). */
export const VALIDATION_LAYOUT: readonly { name: string; bpp: number }[] = [
  { name: 'reservoirCur', bpp: 128 },
  { name: 'reservoirPrev', bpp: 128 },
  { name: 'slot0', bpp: 32 },
  { name: 'slot1', bpp: 32 },
  { name: 'slot2', bpp: 32 },
  { name: 'gbufferCur', bpp: 32 },
  { name: 'gbufferPrev', bpp: 32 },
  { name: 'suffixScratch', bpp: 48 },
  { name: 'accum', bpp: 16 },
  { name: 'debugOut', bpp: 16 },
  { name: 'queueItems', bpp: 16 },
  { name: 'rngMisc', bpp: 28 },
];

export interface AllocProbeOptions {
  members: number;
  /** Member tile side in pixels (default 256). */
  tile?: number;
}

export interface AllocBufferResult {
  name: string;
  bytes: number;
  ok: boolean;
  created: boolean;
  verified: boolean;
  error?: string;
}

export interface AllocProbeReport {
  ok: boolean;
  members: number;
  pixels: number;
  bytesPerPixel: number;
  requestedBytes: number;
  allocatedBytes: number;
  maxBufferSize: number;
  buffers: AllocBufferResult[];
  deviceLost?: string;
  allocMs: number;
  verifyMs: number;
  totalMs: number;
}

const PATTERN_BYTES = 16;

function pattern(seed: number): Uint32Array {
  const p = new Uint32Array(PATTERN_BYTES / 4);
  for (let i = 0; i < p.length; i++) p[i] = (Math.imul(seed + 1, 0x9e3779b1) ^ Math.imul(i + 1, 0x85ebca6b)) >>> 0;
  return p;
}

export async function allocProbe(device: GPUDevice, opts: AllocProbeOptions): Promise<AllocProbeReport> {
  const t0 = performance.now();
  const tile = opts.tile ?? 256;
  const pixels = opts.members * tile * tile;
  const maxBufferSize = device.limits.maxBufferSize;
  let lost: string | undefined;
  void device.lost.then((i) => { lost = `${i.reason}: ${i.message}`; });

  // Plan every logical buffer, split into parts that each fit maxBufferSize (4-byte aligned).
  const plan: { name: string; bytes: number }[] = [];
  for (const { name, bpp } of VALIDATION_LAYOUT) {
    const total = pixels * bpp;
    const cap = Math.floor(maxBufferSize / bpp) * bpp;
    const parts = Math.ceil(total / cap);
    for (let p = 0; p < parts; p++) {
      const bytes = Math.min(cap, total - p * cap);
      plan.push({ name: parts > 1 ? `${name}#${p}` : name, bytes });
    }
  }

  const results: AllocBufferResult[] = [];
  const live: { buf: GPUBuffer; res: AllocBufferResult; seed: number }[] = [];
  try {
    for (const [i, { name, bytes }] of plan.entries()) {
      const res: AllocBufferResult = { name, bytes, ok: false, created: false, verified: false };
      results.push(res);
      device.pushErrorScope('out-of-memory');
      device.pushErrorScope('validation');
      const buf = device.createBuffer({
        label: `probe-${name}`,
        size: bytes,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
      });
      const valErr = await device.popErrorScope();
      const oomErr = await device.popErrorScope();
      if (valErr || oomErr) {
        res.error = `${oomErr ? 'out-of-memory' : 'validation'}: ${(oomErr ?? valErr)!.message}`;
        buf.destroy();
        continue;
      }
      res.created = true;
      live.push({ buf, res, seed: i });
      // Touch both ends so the allocation is committed (Metal may defer residency until first use).
      device.queue.writeBuffer(buf, 0, pattern(2 * i));
      device.queue.writeBuffer(buf, bytes - PATTERN_BYTES, pattern(2 * i + 1));
    }
    const tAlloc = performance.now();

    for (const { buf, res, seed } of live) {
      device.pushErrorScope('out-of-memory');
      device.pushErrorScope('validation');
      try {
        const head = new Uint32Array(await readBuffer(device, buf, PATTERN_BYTES, 0));
        const tail = new Uint32Array(await readBuffer(device, buf, PATTERN_BYTES, res.bytes - PATTERN_BYTES));
        const want0 = pattern(2 * seed), want1 = pattern(2 * seed + 1);
        res.verified = head.every((v, k) => v === want0[k]) && tail.every((v, k) => v === want1[k]);
        if (!res.verified) res.error = 'readback mismatch';
      } catch (e) {
        res.error = `readback: ${e instanceof Error ? e.message : String(e)}`;
      }
      const valErr = await device.popErrorScope();
      const oomErr = await device.popErrorScope();
      if (valErr || oomErr) res.error = `${oomErr ? 'out-of-memory' : 'validation'} during verify: ${(oomErr ?? valErr)!.message}`;
      res.ok = res.created && res.verified && !valErr && !oomErr;
    }
    const tVerify = performance.now();
    await device.queue.onSubmittedWorkDone();

    const bpp = VALIDATION_LAYOUT.reduce((s, b) => s + b.bpp, 0);
    return {
      ok: results.every((r) => r.ok) && lost === undefined,
      members: opts.members,
      pixels,
      bytesPerPixel: bpp,
      requestedBytes: pixels * bpp,
      allocatedBytes: results.filter((r) => r.created).reduce((s, r) => s + r.bytes, 0),
      maxBufferSize,
      buffers: results,
      deviceLost: lost,
      allocMs: tAlloc - t0,
      verifyMs: tVerify - tAlloc,
      totalMs: performance.now() - t0,
    };
  } finally {
    for (const { buf } of live) buf.destroy();
  }
}
