// T1 RNG (validation-harness T1; math.md#rng-layout) and T13b jitter i.i.d. (plan §1.2), both GPU lanes, on the
// production common/rng-path.wgsl + frame jitter stream:
//   - determinism: two dispatches bitwise identical, and GPU ≡ a CPU pcg3d/pcg4d mirror bit for bit (so both lanes agree);
//   - 64-bit PathSeed (M3c): pixels whose 32-bit jitter seed (initSeed.x) collides still get different path dims (a
//     32-bit initSeed made every path one of 2^32 dimension vectors: a fixed-quadrature bias, math.md#rng-layout);
//   - χ² uniformity: 1D (100 bins) and 2D (32×32) at 10^7 draws, Šidák-style strict critical values;
//   - Pearson |r| < 4/√N across slots of one vertex, across vertices, across frames (t, t+1) and across pixels;
//   - T13b: jitter uniform over the pixel, independent across samples/frames and of the path stream; negative controls
//     (a fixed jitter, a jitter reusing a path dim) are detected by the same statistics.
import { afterAll, describe, expect, it } from 'vitest';
import { composeWgsl, createCheckedShaderModule } from '../../src/core/gpu/wgsl-composer.ts';
import { readBuffer } from '../../src/core/gpu/readback.ts';
import { shaderSources } from '../../src/core/shaders/index.ts';
import { getTestGpu, releaseTestGpu } from './device-factory.ts';

afterAll(releaseTestGpu);

const KERNEL = /* wgsl */ `
#include "common/rng-path.wgsl"
struct P { n: u32, mode: u32, runSeed: u32, t0: u32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read_write> outv: array<u32>;
@group(0) @binding(2) var<storage, read_write> hist: array<atomic<u32>>;

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3u, @builtin(local_invocation_index) li: u32, @builtin(num_workgroups) nw: vec3u) {
  let i = (wg.y * nw.x + wg.x) * 256u + li;
  if (i >= p.n) { return; }
  let seed = path_init_seed(p.runSeed, 0u, p.t0, i);
  if (p.mode == 0u) {
    for (var s = 0u; s < 10u; s++) { outv[16u * i + s] = path_hash(seed, 1u, s); }
    outv[16u * i + 10u] = path_hash(seed, 2u, SLOT_H1);                                   // next vertex
    let seedNext = path_init_seed(p.runSeed, 0u, p.t0 + 1u, i);
    outv[16u * i + 11u] = path_hash(seedNext, 1u, SLOT_H1);                               // next frame
    let j = pcg3d(vec3u(seed.x, 0u, STREAM_JITTER));                                       // rand2(seed.x, 0, JITTER)
    outv[16u * i + 12u] = j.x;
    outv[16u * i + 13u] = j.y;
    outv[16u * i + 14u] = pcg3d(vec3u(seedNext.x, 0u, STREAM_JITTER)).x;                  // jitter, next frame
    outv[16u * i + 15u] = seed.x;
    return;
  }
  if (p.mode == 1u) {
    let u = path_u01(seed, 1u + (i & 3u), SLOT_H1 + (i >> 2u) % 9u);
    atomicAdd(&hist[min(u32(u * 100.0), 99u)], 1u);
    return;
  }
  let u = vec2f(path_u01(seed, 1u, SLOT_H1), path_u01(seed, 1u, SLOT_H2));
  atomicAdd(&hist[min(u32(u.y * 32.0), 31u) * 32u + min(u32(u.x * 32.0), 31u)], 1u);
}
`;

// CPU mirror of rng.wgsl pcg3d (u32 arithmetic).
function pcg3d(x: number, y: number, z: number): [number, number, number] {
  let a = (Math.imul(x, 1664525) + 1013904223) >>> 0, b = (Math.imul(y, 1664525) + 1013904223) >>> 0, c = (Math.imul(z, 1664525) + 1013904223) >>> 0;
  a = (a + Math.imul(b, c)) >>> 0; b = (b + Math.imul(c, a)) >>> 0; c = (c + Math.imul(a, b)) >>> 0;
  a ^= a >>> 16; b ^= b >>> 16; c ^= c >>> 16;
  a = (a + Math.imul(b, c)) >>> 0; b = (b + Math.imul(c, a)) >>> 0; c = (c + Math.imul(a, b)) >>> 0;
  return [a >>> 0, b >>> 0, c >>> 0];
}
// CPU mirror of rng.wgsl pcg4d.
function pcg4d(x: number, y: number, z: number, w: number): [number, number, number, number] {
  let a = (Math.imul(x, 1664525) + 1013904223) >>> 0, b = (Math.imul(y, 1664525) + 1013904223) >>> 0;
  let c = (Math.imul(z, 1664525) + 1013904223) >>> 0, d = (Math.imul(w, 1664525) + 1013904223) >>> 0;
  a = (a + Math.imul(b, d)) >>> 0; b = (b + Math.imul(c, a)) >>> 0; c = (c + Math.imul(a, b)) >>> 0; d = (d + Math.imul(b, c)) >>> 0;
  a ^= a >>> 16; b ^= b >>> 16; c ^= c >>> 16; d ^= d >>> 16;
  a = (a + Math.imul(b, d)) >>> 0; b = (b + Math.imul(c, a)) >>> 0; c = (c + Math.imul(a, b)) >>> 0; d = (d + Math.imul(b, c)) >>> 0;
  return [a >>> 0, b >>> 0, c >>> 0, d >>> 0];
}
const STREAM_PATH = 0x9e3779b9, STREAM_JITTER = 0xc2b2ae35;
const u01 = (h: number): number => (h >>> 8) / 16777216;

async function dispatch(mode: number, n: number, runSeed: number, t0: number): Promise<Uint32Array> {
  const { device, features, wgslLanguageFeatures } = await getTestGpu();
  const shader = composeWgsl('tests/rng-t1.wgsl', { sources: { ...shaderSources, 'tests/rng-t1.wgsl': KERNEL }, features, wgslLanguageFeatures });
  const module = await createCheckedShaderModule(device, shader, 'rng-t1');
  const pipeline = await device.createComputePipelineAsync({ layout: 'auto', compute: { module, entryPoint: 'main' } });
  const params = device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
  device.queue.writeBuffer(params, 0, new Uint32Array([n, mode, runSeed, t0]));
  const outBytes = mode === 0 ? n * 64 : 16;
  const out = device.createBuffer({ size: outBytes, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
  const hist = device.createBuffer({ size: 1024 * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
  const bg = device.createBindGroup({ layout: pipeline.getBindGroupLayout(0), entries: [
    { binding: 0, resource: { buffer: params } }, { binding: 1, resource: { buffer: out } }, { binding: 2, resource: { buffer: hist } }] });
  const groups = Math.ceil(n / 256);
  const gx = Math.min(groups, 65535), gy = Math.ceil(groups / 65535);
  const enc = device.createCommandEncoder();
  const pass = enc.beginComputePass();
  pass.setPipeline(pipeline); pass.setBindGroup(0, bg); pass.dispatchWorkgroups(gx, gy); pass.end();
  device.queue.submit([enc.finish()]);
  const r = new Uint32Array(await readBuffer(device, mode === 0 ? out : hist, mode === 0 ? outBytes : 4096));
  params.destroy(); out.destroy(); hist.destroy();
  return r;
}

function pearson(a: ArrayLike<number>, b: ArrayLike<number>): number {
  const n = a.length;
  let sa = 0, sb = 0;
  for (let i = 0; i < n; i++) { sa += a[i]; sb += b[i]; }
  const ma = sa / n, mb = sb / n;
  let cab = 0, caa = 0, cbb = 0;
  for (let i = 0; i < n; i++) { const x = a[i] - ma, y = b[i] - mb; cab += x * y; caa += x * x; cbb += y * y; }
  return cab / Math.sqrt(caa * cbb);
}
/** Wilson–Hilferty χ²_k upper quantile at z (z = 3.72 ⇔ p ≈ 1e-4). */
const chi2Crit = (k: number, z = 3.72): number => k * (1 - 2 / (9 * k) + z * Math.sqrt(2 / (9 * k))) ** 3;

describe('T1 path RNG (common/rng-path.wgsl)', () => {
  const N = 1 << 20;

  it('bitwise deterministic across dispatches and equal to the CPU pcg3d mirror (lane-independent)', async () => {
    const a = await dispatch(0, N, 0xdeadbeef, 5);
    const b = await dispatch(0, N, 0xdeadbeef, 5);
    let diff = 0;
    for (let i = 0; i < a.length; i++) if (a[i] !== b[i]) diff++;
    expect(diff).toBe(0);
    let bad = 0;
    for (let i = 0; i < N; i += 997) {
      const [sx, sy] = pcg3d(0xdeadbeef, 5, i);
      if (a[16 * i + 15] !== sx) bad++;
      for (let s = 0; s < 10; s++) if (a[16 * i + s] !== pcg4d(sx, sy, 16 + s, STREAM_PATH)[0]) bad++;
      if (a[16 * i + 10] !== pcg4d(sx, sy, 32 + 1, STREAM_PATH)[0]) bad++;
      if (a[16 * i + 12] !== pcg3d(sx, 0, STREAM_JITTER)[0]) bad++;
    }
    expect(bad).toBe(0);
  });

  it('64-bit PathSeed: pixels with colliding 32-bit initSeed.x still draw different path dims', async () => {
    const a = await dispatch(0, N, 0x1234567, 3);
    const seen = new Map<number, number>();
    let collisions = 0, sameDims = 0;
    for (let i = 0; i < N; i++) {
      const x = a[16 * i + 15];
      const j = seen.get(x);
      if (j === undefined) { seen.set(x, i); continue; }
      collisions++;
      let eq = true;
      for (let s = 0; s < 10; s++) if (a[16 * i + s] !== a[16 * j + s]) eq = false;
      if (eq) sameDims++;
    }
    // ~N²/2^33 = 128 expected birthday collisions of the 32-bit word; with a 32-bit seed every one would repeat its path.
    expect(collisions).toBeGreaterThan(50);
    expect(sameDims).toBe(0);
  });

  it('χ² uniformity: 1D 100 bins and 2D 32×32 at 10^7 draws', async () => {
    const n = 10_000_000;
    const h1 = await dispatch(1, n, 12345, 0);
    const e1 = n / 100;
    let c1 = 0;
    for (let i = 0; i < 100; i++) c1 += (h1[i] - e1) ** 2 / e1;
    expect(c1).toBeLessThan(chi2Crit(99));
    const h2 = await dispatch(2, n, 999, 7);
    const e2 = n / 1024;
    let c2 = 0;
    for (let i = 0; i < 1024; i++) c2 += (h2[i] - e2) ** 2 / e2;
    expect(c2).toBeLessThan(chi2Crit(1023));
  });

  it('Pearson |r| < 4/√N across slots, vertices, frames and pixels', async () => {
    const a = await dispatch(0, N, 777, 11);
    const col = (k: number) => Float64Array.from({ length: N }, (_, i) => u01(a[16 * i + k]));
    const cols = Array.from({ length: 12 }, (_, k) => col(k));
    const lim = 4 / Math.sqrt(N);
    for (let s = 0; s < 10; s++) for (let t = s + 1; t < 12; t++) {
      expect(Math.abs(pearson(cols[s], cols[t])), `slot ${s} vs col ${t}`).toBeLessThan(lim);
    }
    // neighbouring pixels (i, i+1)
    const h1 = cols[1];
    expect(Math.abs(pearson(h1.subarray(0, N - 1), h1.subarray(1)))).toBeLessThan(lim);
  });
});

describe('T13b jitter (i.i.d. per run and sample; separate from the path stream)', () => {
  const N = 1 << 20;

  it('uniform over the pixel (χ² 8×8), independent across samples and of the path dims; negative controls detected', async () => {
    const a = await dispatch(0, N, 4242, 3);
    const jx = Float64Array.from({ length: N }, (_, i) => u01(a[16 * i + 12]));
    const jy = Float64Array.from({ length: N }, (_, i) => u01(a[16 * i + 13]));
    const jxNext = Float64Array.from({ length: N }, (_, i) => u01(a[16 * i + 14]));
    const hist = new Array(64).fill(0);
    for (let i = 0; i < N; i++) hist[8 * Math.floor(jy[i] * 8) + Math.floor(jx[i] * 8)]++;
    const e = N / 64;
    const c = hist.reduce((s, x) => s + (x - e) ** 2 / e, 0);
    expect(c).toBeLessThan(chi2Crit(63));
    const lim = 4 / Math.sqrt(N);
    expect(Math.abs(pearson(jx, jy))).toBeLessThan(lim);
    expect(Math.abs(pearson(jx, jxNext))).toBeLessThan(lim);             // next sample / frame
    for (let s = 0; s < 10; s++) expect(Math.abs(pearson(jx, Float64Array.from({ length: N }, (_, i) => u01(a[16 * i + s]))))).toBeLessThan(lim);
    // negative control 1: a fixed jitter (JITTER_NONE-like) fails the uniformity χ²
    const fixed = new Array(64).fill(0); fixed[8 * 4 + 4] = N;
    expect(fixed.reduce((s, x) => s + (x - e) ** 2 / e, 0)).toBeGreaterThan(chi2Crit(63));
    // negative control 2: a jitter drawn from the path stream (slot u_h1) is caught by the independence test
    const bad = Float64Array.from({ length: N }, (_, i) => u01(a[16 * i + 1]));
    expect(Math.abs(pearson(bad, Float64Array.from({ length: N }, (_, i) => u01(a[16 * i + 1]))))).toBeGreaterThan(lim);
    // negative control 3: a jitter that repeats across samples (seeded without t) is caught across frames
    expect(Math.abs(pearson(jx, jx))).toBeGreaterThan(lim);
  });
});
