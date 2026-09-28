// T12 (plan §5 M1, §7.4): BVH2 traversal vs an f64 CPU reference, watertightness, self-intersection, zero
// overflow / iteration-cap counters, and throughput at 1080p-equivalent ray counts. Runs in both GPU lanes.
import { afterAll, describe, expect, it } from 'vitest';
import { getTestGpu, lane, releaseTestGpu } from './device-factory.ts';
import { composeWgsl, createCheckedShaderModule } from '../../src/core/gpu/wgsl-composer.ts';
import { shaderSources } from '../../src/core/shaders/index.ts';
import { readBuffer } from '../../src/core/gpu/readback.ts';
import type { GpuContext } from '../../src/core/gpu/device.ts';
import { buildBvh } from '../../src/core/bvh/sah-builder.ts';
import { BVH_MISS, uploadBvh, type BvhData, type BvhGpuBuffers } from '../../src/core/bvh/layout.ts';
import { bruteAny, bruteClosest, bvhTrace64, intersectTri64 } from '../../src/core/bvh/cpu-trace.ts';
import { icosphere, loadGltfMesh, meshBounds, proceduralScene, randomDir, rng, type Mesh } from '../../tests/bvh/fixtures.ts';

const N_RAYS = 1_000_000;
const W1080 = 1920, H1080 = 1080;

const KERNEL = /* wgsl */ `
#include "geom/visible.wgsl"
#include "common/rng.wgsl"

@group(0) @binding(0) var<storage, read> bvh_nodes: array<vec4f>;
@group(0) @binding(1) var<storage, read> bvh_tris: array<vec4f>;
@group(0) @binding(2) var<storage, read> inp: array<vec4f>;
@group(0) @binding(3) var<storage, read_write> outp: array<vec4u>;
@group(0) @binding(4) var<storage, read_write> ctr: array<atomic<u32>, 8>;
struct Params { n: u32, seed: u32, width: u32, height: u32, a: vec4f, b: vec4f, c: vec4f, d: vec4f, e: vec4f }
@group(0) @binding(5) var<uniform> P: Params;

// ctr: 0 steps, 1 box tests, 2 tri tests, 3 flags (bit0 overflow, bit1 itercap), 4 max stack depth
fn flush_stats() {
  _ = &inp; // every entry point statically uses all bindings (one 'auto' layout shape)
  let s = bvh_stats();
#if BVH_STATS
  atomicAdd(&ctr[0], s.x); atomicAdd(&ctr[1], s.y); atomicAdd(&ctr[2], s.z);
  atomicMax(&ctr[4], (s.w >> 8u) & 0xffu);
#endif
  if ((s.w & 3u) != 0u) { atomicOr(&ctr[3], s.w & 3u); }
}

// stride 2: [o | tmax_any] [d | 0]
@compute @workgroup_size(64) fn closest_any(@builtin(global_invocation_id) gid: vec3u) {
  let i = gid.x + gid.y * 65535u * 64u;
  if (i >= P.n) { return; }
  bvh_stats_reset();
  let r0 = inp[2u * i];
  let r1 = inp[2u * i + 1u];
  let h = trace_closest(r0.xyz, r1.xyz, FLT_MAX);
  let anyPrim = bvh_trace(r0.xyz, r1.xyz, r0.w, true, BVH_MISS, BVH_MISS).primId; // = trace_any, keeping the prim
  outp[i] = vec4u(bitcast<u32>(h.t), h.primId, anyPrim, bitcast<u32>(h.u));
  flush_stats();
}

// stride 4: [p0 | primId] [p1 | u] [p2 | v] [dir | side]; point from barycentrics, offset_ray, closest hit.
@compute @workgroup_size(64) fn spawn(@builtin(global_invocation_id) gid: vec3u) {
  let i = gid.x;
  if (i >= P.n) { return; }
  bvh_stats_reset();
  let p0 = inp[4u * i];
  let p1 = inp[4u * i + 1u];
  let p2 = inp[4u * i + 2u];
  let dd = inp[4u * i + 3u];
  let p = tri_point(p0.xyz, p1.xyz, p2.xyz, p1.w, p2.w);
  let ng = tri_geom_normal(p0.xyz, p1.xyz, p2.xyz) * dd.w;
  // P.width = 1: negative control without the offset (the test must be able to see self-hits).
  let o = select(offset_ray(p, ng), p, P.width == 1u);
  let h = trace_closest(o, dd.xyz, FLT_MAX);
  outp[i] = vec4u(bitcast<u32>(h.t), h.primId, bitcast<u32>(p0.w), 0u);
  flush_stats();
}

// stride 8: [a0 | primA] [a1 | ua] [a2 | va] [b0 | primB] [b1 | ub] [b2 | vb] [mode, 0, 0, 0] [dir | 0]
@compute @workgroup_size(64) fn vis(@builtin(global_invocation_id) gid: vec3u) {
  let i = gid.x;
  if (i >= P.n) { return; }
  bvh_stats_reset();
  let k = 8u * i;
  let a0 = inp[k]; let a1 = inp[k + 1u]; let a2 = inp[k + 2u];
  let b0 = inp[k + 3u]; let b1 = inp[k + 4u]; let b2 = inp[k + 5u];
  let pa = tri_point(a0.xyz, a1.xyz, a2.xyz, a1.w, a2.w);
  let na = tri_geom_normal(a0.xyz, a1.xyz, a2.xyz);
  var res = false;
  if (inp[k + 6u].x > 0.5) {
    res = visibleInf(pa, na, bitcast<u32>(a0.w), inp[k + 7u].xyz);
  } else {
    let pb = tri_point(b0.xyz, b1.xyz, b2.xyz, b1.w, b2.w);
    let nb = tri_geom_normal(b0.xyz, b1.xyz, b2.xyz);
    res = visible(pa, na, bitcast<u32>(a0.w), pb, nb, bitcast<u32>(b0.w));
  }
  outp[i] = vec4u(select(0u, 1u, res), 0u, 0u, 0u);
  flush_stats();
}

// Throughput kernels. P.a = eye | tanHalfFov, P.b = forward | aspect, P.c = right, P.d = up (camera);
// random kernels: P.a / P.b = box min / max for origins.
fn cam_dir(x: u32, y: u32) -> vec3f {
  let sx = (2.0 * (f32(x) + 0.5) / f32(P.width) - 1.0) * P.a.w * P.b.w;
  let sy = (1.0 - 2.0 * (f32(y) + 0.5) / f32(P.height)) * P.a.w;
  return normalize(P.b.xyz + sx * P.c.xyz + sy * P.d.xyz);
}
fn sphere_dir(u: vec2f) -> vec3f {
  let z = 1.0 - 2.0 * u.x;
  let r = sqrt(max(0.0, 1.0 - z * z));
  let phi = TWO_PI * u.y;
  return vec3f(r * cos(phi), r * sin(phi), z);
}
@compute @workgroup_size(8, 8) fn perf_primary(@builtin(global_invocation_id) gid: vec3u) {
  if (gid.x >= P.width || gid.y >= P.height) { return; }
  let h = trace_closest(P.a.xyz, cam_dir(gid.x, gid.y), FLT_MAX);
  outp[gid.y * P.width + gid.x] = vec4u(h.primId, bitcast<u32>(h.t), 0u, 0u);
  flush_stats();
}
@compute @workgroup_size(8, 8) fn perf_secondary(@builtin(global_invocation_id) gid: vec3u) {
  if (gid.x >= P.width || gid.y >= P.height) { return; }
  let pix = gid.y * P.width + gid.x;
  let d0 = cam_dir(gid.x, gid.y);
  let t0 = bitcast<f32>(outp[pix].y);
  var prim = BVH_MISS;
  if (outp[pix].x != BVH_MISS) {
    // Hemisphere around −d0 from the primary hit (incoherent, diffuse-like; no normals in this test).
    let p = P.a.xyz + d0 * (t0 * 0.9999);
    var d = sphere_dir(rand2(pix, P.seed, STREAM_PATH));
    d = select(d, -d, dot(d, d0) > 0.0);
    prim = trace_closest(p, d, FLT_MAX).primId;
  }
  outp[pix].z = prim;
  flush_stats();
}
@compute @workgroup_size(8, 8) fn perf_shadow(@builtin(global_invocation_id) gid: vec3u) {
  if (gid.x >= P.width || gid.y >= P.height) { return; }
  let pix = gid.y * P.width + gid.x;
  let d0 = cam_dir(gid.x, gid.y);
  var occ = 0u;
  if (outp[pix].x != BVH_MISS) {
    // Shadow segment from the primary hit to one point light at P.e.xyz.
    let p = P.a.xyz + d0 * (bitcast<f32>(outp[pix].y) * 0.9999);
    let l = P.e.xyz;
    occ = select(0u, 1u, trace_any(p, l - p, 0.9999));
  }
  outp[pix].w = occ;
  flush_stats();
}
@compute @workgroup_size(64) fn perf_random(@builtin(global_invocation_id) gid: vec3u) {
  let i = gid.x + gid.y * 65535u * 64u;
  if (i >= P.n) { return; }
  let r = rand2(i, P.seed, STREAM_PATH);
  let o = mix(P.a.xyz, P.b.xyz, vec3f(r, rand1(i, P.seed + 1u, STREAM_PATH)));
  let d = sphere_dir(rand2(i, P.seed + 2u, STREAM_PATH));
  outp[i].x = trace_closest(o, d, FLT_MAX).primId;
  flush_stats();
}
`;

// Alpha hook + self-declared bindings (traverse.wgsl BVH_DECLARE_BINDINGS): primIds with bit 0 set are cut out.
const ALPHA_KERNEL = /* wgsl */ `
#include "bvh/traverse.wgsl"
@group(0) @binding(2) var<storage, read> inp: array<vec4f>;
@group(0) @binding(3) var<storage, read_write> outp: array<vec4u>;
fn alpha_pass(primId: u32, u: f32, v: f32) -> bool { return (primId & 1u) == 0u; }
@compute @workgroup_size(64) fn main(@builtin(global_invocation_id) gid: vec3u) {
  let i = gid.x;
  if (i >= arrayLength(&outp)) { return; }
  let r0 = inp[2u * i];
  let r1 = inp[2u * i + 1u];
  let h = trace_closest(r0.xyz, r1.xyz, FLT_MAX);
  let a = bvh_trace(r0.xyz, r1.xyz, r0.w, true, BVH_MISS, BVH_MISS).primId;
  outp[i] = vec4u(bitcast<u32>(h.t), h.primId, a, bvh_stats().w);
}
`;

interface Variant { watertight: boolean; stats: boolean }
interface RunResult { out: Uint32Array; ctr: Uint32Array; gpuMs: number }

const pipelines = new Map<string, Promise<GPUComputePipeline>>();
function getPipeline(ctx: GpuContext, v: Variant, entry: string): Promise<GPUComputePipeline> {
  const key = `${v.watertight}|${v.stats}|${entry}`;
  let p = pipelines.get(key);
  if (!p) {
    p = (async () => {
      const shader = composeWgsl('tests/bvh_t12.wgsl', {
        sources: { ...shaderSources, 'tests/bvh_t12.wgsl': KERNEL },
        defines: { WATERTIGHT: v.watertight, BVH_STATS: v.stats },
        features: ctx.features, wgslLanguageFeatures: ctx.wgslLanguageFeatures,
      });
      const module = await createCheckedShaderModule(ctx.device, shader, `bvh_t12.${key}`);
      return ctx.device.createComputePipelineAsync({ layout: 'auto', compute: { module, entryPoint: entry } });
    })();
    pipelines.set(key, p);
  }
  return p;
}

function storage(device: GPUDevice, data: ArrayBufferView | number, label: string): GPUBuffer {
  const size = Math.max(16, Math.ceil((typeof data === 'number' ? data : data.byteLength) / 16) * 16);
  const buf = device.createBuffer({ size, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST, mappedAtCreation: typeof data !== 'number', label });
  if (typeof data !== 'number') { new Uint8Array(buf.getMappedRange()).set(new Uint8Array(data.buffer, data.byteOffset, data.byteLength)); buf.unmap(); }
  return buf;
}

/** Dispatch one entry point; `outBuf` may be passed to chain kernels (perf_secondary reads perf_primary's output). */
async function run(
  ctx: GpuContext, v: Variant, entry: string, bvh: BvhGpuBuffers, input: Float32Array | null, n: number,
  opts: { params?: Float32Array; outBuf?: GPUBuffer; grid?: [number, number]; reps?: number; readOut?: boolean } = {},
): Promise<RunResult & { outBuf: GPUBuffer }> {
  const { device } = ctx;
  const pipeline = await getPipeline(ctx, v, entry);
  device.pushErrorScope('validation');
  const inBuf = storage(device, input ?? new Float32Array(4), 'inp');
  const outBuf = opts.outBuf ?? storage(device, n * 16, 'outp');
  const ctrBuf = storage(device, new Uint32Array(8), 'ctr');
  const params = opts.params ?? new Float32Array(24);
  const pu = new Uint32Array(params.buffer);
  pu[0] = n;
  const uni = device.createBuffer({ size: 96, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
  device.queue.writeBuffer(uni, 0, params);
  const bg = device.createBindGroup({ layout: pipeline.getBindGroupLayout(0), entries: [
    { binding: 0, resource: { buffer: bvh.nodes } }, { binding: 1, resource: { buffer: bvh.tris } },
    { binding: 2, resource: { buffer: inBuf } }, { binding: 3, resource: { buffer: outBuf } },
    { binding: 4, resource: { buffer: ctrBuf } }, { binding: 5, resource: { buffer: uni } }] });
  const groups = opts.grid ?? [Math.min(65535, Math.ceil(n / 64)), Math.ceil(Math.ceil(n / 64) / 65535)];
  const hasTs = ctx.features.has('timestamp-query');
  const qs = hasTs ? device.createQuerySet({ type: 'timestamp', count: 2 }) : undefined;
  const qBuf = hasTs ? device.createBuffer({ size: 16, usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC }) : undefined;
  const times: number[] = [];
  for (let rep = 0; rep < (opts.reps ?? 1); rep++) {
    const enc = device.createCommandEncoder();
    const pass = enc.beginComputePass(qs ? { timestampWrites: { querySet: qs, beginningOfPassWriteIndex: 0, endOfPassWriteIndex: 1 } } : {});
    pass.setPipeline(pipeline); pass.setBindGroup(0, bg); pass.dispatchWorkgroups(groups[0], groups[1]); pass.end();
    if (qs && qBuf) enc.resolveQuerySet(qs, 0, 2, qBuf, 0);
    const t0 = performance.now();
    device.queue.submit([enc.finish()]);
    await device.queue.onSubmittedWorkDone();
    const wall = performance.now() - t0;
    if (qBuf) {
      const ts = new BigUint64Array(await readBuffer(device, qBuf, 16));
      const ms = Number(ts[1] - ts[0]) / 1e6;
      times.push(ms > 0 ? ms : wall);
    } else times.push(wall);
  }
  const err = await device.popErrorScope();
  if (err) throw new Error(`WebGPU validation error in ${entry}: ${err.message}`);
  const out = opts.readOut === false ? new Uint32Array(0) : new Uint32Array(await readBuffer(device, outBuf, n * 16));
  const ctr = new Uint32Array(await readBuffer(device, ctrBuf, 32));
  inBuf.destroy(); ctrBuf.destroy(); uni.destroy(); qs?.destroy(); qBuf?.destroy();
  if (!opts.outBuf && opts.readOut !== false) outBuf.destroy();
  times.sort((a, b) => a - b);
  return { out, ctr, gpuMs: times[Math.floor(times.length / 2)], outBuf };
}

const VARIANTS: Variant[] = [{ watertight: false, stats: true }, { watertight: true, stats: true }];
const vname = (v: Variant) => (v.watertight ? 'woop' : 'mt');

// ---------------------------------------------------------------------------------------------------------------
// (1) Random rays vs the f64 CPU reference.

function makeRays(m: Mesh, n: number, seed: number): Float32Array {
  const r = rng(seed), b = meshBounds(m);
  const diag = Math.hypot(b.max[0] - b.min[0], b.max[1] - b.min[1], b.max[2] - b.min[2]);
  const rays = new Float32Array(n * 8);
  const nTri = m.indices.length / 3;
  for (let i = 0; i < n; i++) {
    const o = [0, 1, 2].map((k) => b.min[k] + (b.max[k] - b.min[k]) * (1.2 * r() - 0.1));
    let d: number[];
    if (i % 10 === 0) {
      // Aim at a point on a random triangle edge (stress shared edges / ties).
      const p = Math.floor(r() * nTri), e = Math.floor(r() * 3), s = r();
      const va = m.indices[3 * p + e] * 3, vb = m.indices[3 * p + ((e + 1) % 3)] * 3;
      const tgt = [0, 1, 2].map((k) => m.positions[va + k] + s * (m.positions[vb + k] - m.positions[va + k]));
      d = [tgt[0] - o[0], tgt[1] - o[1], tgt[2] - o[2]];
      const l = Math.hypot(d[0], d[1], d[2]) || 1;
      d = d.map((x) => x / l);
    } else d = randomDir(r);
    rays.set([o[0], o[1], o[2], r() * diag, d[0], d[1], d[2], 0], i * 8);
  }
  return rays;
}

interface Ref { prim: Uint32Array; t: Float64Array; occ: Uint8Array }
function cpuReference(m: Mesh, bvh: BvhData, rays: Float32Array, n: number): Ref {
  const ref: Ref = { prim: new Uint32Array(n), t: new Float64Array(n), occ: new Uint8Array(n) };
  const o = [0, 0, 0], d = [0, 0, 0];
  for (let i = 0; i < n; i++) {
    o[0] = rays[8 * i]; o[1] = rays[8 * i + 1]; o[2] = rays[8 * i + 2];
    d[0] = rays[8 * i + 4]; d[1] = rays[8 * i + 5]; d[2] = rays[8 * i + 6];
    const h = bvhTrace64(bvh, m.positions, m.indices, o, d);
    ref.prim[i] = h.primId; ref.t[i] = h.t;
    // Any-hit within tmax is decided by the closest hit (t < tmax), which also gives the ambiguity margin.
    ref.occ[i] = h.primId !== BVH_MISS && h.t < rays[8 * i + 3] ? 1 : 0;
  }
  return ref;
}

const TIE_REL = 1e-6;   // plan T12: ties within 1e-6·t
const ULP_K = 16;       // f32 precision band: 16 ulps of the largest coordinate involved
const uvTmp = new Float64Array(2);

/** f64 ray–plane hit of `prim`, the signed distance of that point to the triangle boundary (> 0 inside, < 0
 *  outside; min_i b_i·h_i with h_i the altitude onto edge i), and |cos| between the ray and the plane normal. */
function edgeDistance(m: Mesh, o: number[], d: number[], prim: number): { t: number; dist: number; cos: number } {
  const t = intersectTri64(o[0], o[1], o[2], d[0], d[1], d[2], m.positions, m.indices, prim, uvTmp, Infinity);
  if (!Number.isFinite(t)) return { t: NaN, dist: -Infinity, cos: 1 };
  const V = [0, 1, 2].map((j) => { const a = m.indices[3 * prim + j] * 3; return [m.positions[a], m.positions[a + 1], m.positions[a + 2]]; });
  const len = (a: number[], b: number[]) => Math.hypot(a[0] - b[0], a[1] - b[1], a[2] - b[2]);
  const e1 = V[1].map((x, k) => x - V[0][k]), e2 = V[2].map((x, k) => x - V[0][k]);
  const nx = e1[1] * e2[2] - e1[2] * e2[1], ny = e1[2] * e2[0] - e1[0] * e2[2], nz = e1[0] * e2[1] - e1[1] * e2[0];
  const area2 = Math.hypot(nx, ny, nz);
  const b = [1 - uvTmp[0] - uvTmp[1], uvTmp[0], uvTmp[1]];
  const h = [area2 / len(V[1], V[2]), area2 / len(V[2], V[0]), area2 / len(V[0], V[1])];
  const cos = Math.abs(nx * d[0] + ny * d[1] + nz * d[2]) / (area2 * Math.hypot(d[0], d[1], d[2]));
  return { t, dist: Math.min(b[0] * h[0], b[1] * h[1], b[2] * h[2]), cos };
}

/**
 * Classify GPU vs f64 reference. A mismatch is a `tie` (other prim at the same t, plan: within 1e-6·t), `precision`
 * (the deciding triangle is within 16 f32 ulps of its boundary, scaled by 1/|cos| for grazing rays: an f32
 * intersector may legitimately decide either way), or `unexplained` (a real bug; must be 0). Rays with index % 10 == 0 are aimed at edges and reported apart.
 */
function compare(m: Mesh, rays: Float32Array, ref: Ref, gpu: Uint32Array, n: number) {
  const gpuF = new Float32Array(gpu.buffer, gpu.byteOffset, gpu.length);
  const c = { n, match: 0, tie: 0, tieUlp: 0, precisionRandom: 0, precisionAimed: 0, unexplained: 0, anyMatch: 0, anyPrecisionRandom: 0, anyPrecisionAimed: 0, anyUnexplained: 0, maxRelT: 0, examples: [] as string[] };
  const o = [0, 0, 0], d = [0, 0, 0];
  for (let i = 0; i < n; i++) {
    const gp = gpu[4 * i + 1], rp = ref.prim[i], aimed = i % 10 === 0;
    o[0] = rays[8 * i]; o[1] = rays[8 * i + 1]; o[2] = rays[8 * i + 2];
    d[0] = rays[8 * i + 4]; d[1] = rays[8 * i + 5]; d[2] = rays[8 * i + 6];
    const tmax = rays[8 * i + 3];
    const oMax = Math.max(Math.abs(o[0]), Math.abs(o[1]), Math.abs(o[2]));
    // In-plane f32 position error grows as 1/|cos| at grazing incidence.
    const delta = (t: number, cos = 1) => (ULP_K * 2 ** -23 * Math.max(1, oMax + (Number.isFinite(t) ? t : 0))) / Math.max(cos, 1e-6);
    const R = rp !== BVH_MISS ? edgeDistance(m, o, d, rp) : { t: Infinity, dist: Infinity, cos: 1 };
    const refNearEdge = rp !== BVH_MISS && R.dist < delta(ref.t[i], R.cos);
    let closestOk = true;
    if (gp === rp) {
      c.match++;
      if (rp !== BVH_MISS) c.maxRelT = Math.max(c.maxRelT, Math.abs(gpuF[4 * i] - ref.t[i]) / ref.t[i]);
    } else {
      const G = gp !== BVH_MISS ? edgeDistance(m, o, d, gp) : { t: NaN, dist: -Infinity, cos: 1 };
      const dg = delta(G.t, G.cos);
      const dt = Math.abs(G.t - ref.t[i]);
      if (gp !== BVH_MISS && rp !== BVH_MISS && dt <= TIE_REL * ref.t[i] && G.dist >= -dg) c.tie++;
      // Two surfaces closer in t than f32 can resolve at this |o| (coplanar decals / z-fighting): tie at ulp level.
      else if (gp !== BVH_MISS && rp !== BVH_MISS && dt <= delta(ref.t[i]) && G.dist >= -dg) c.tieUlp++;
      else if ((gp !== BVH_MISS && G.t > 0 && G.dist >= -dg && G.dist < dg && (rp === BVH_MISS || G.t <= ref.t[i] * (1 + TIE_REL))) // GPU hit a triangle at its boundary
        || (refNearEdge && (gp === BVH_MISS || (G.t >= ref.t[i] * (1 - TIE_REL) && G.dist >= -dg)))) { // GPU slipped past a boundary hit
        if (aimed) c.precisionAimed++; else c.precisionRandom++;
      } else {
        closestOk = false; c.unexplained++;
        if (c.examples.length < 5) c.examples.push(`ray ${i}: gpu ${gp} t=${gpuF[4 * i]} (f64 t=${G.t} edge=${G.dist}) cpu ${rp} t=${ref.t[i]} edge=${R.dist}`);
      }
    }
    // Any-hit: GPU reports the prim it stopped at.
    const ga = gpu[4 * i + 2];
    const gocc = ga !== BVH_MISS ? 1 : 0;
    if (gocc === ref.occ[i]) { c.anyMatch++; continue; }
    let amb = false;
    if (gocc) { const A = edgeDistance(m, o, d, ga); const da = delta(A.t, A.cos); amb = A.t > 0 && A.dist >= -da && (A.dist < da || A.t >= tmax * (1 - 1e-5)); }
    else amb = refNearEdge || Math.abs(ref.t[i] - tmax) <= 1e-5 * tmax || !closestOk;
    if (amb) { if (aimed) c.anyPrecisionAimed++; else c.anyPrecisionRandom++; }
    else { c.anyUnexplained++; if (c.examples.length < 8) c.examples.push(`any ${i}: gpu ${ga} cpu occ ${ref.occ[i]} tclosest=${ref.t[i]} tmax=${tmax}`); }
  }
  return c;
}

// ---------------------------------------------------------------------------------------------------------------

function spawnInputs(m: Mesh, n: number, seed: number, filter?: (prim: number) => boolean) {
  const r = rng(seed);
  const nTri = m.indices.length / 3;
  const input = new Float32Array(n * 16), u32 = new Uint32Array(input.buffer);
  let k = 0, guard = 0;
  while (k < n && guard++ < n * 20) {
    const p = Math.floor(r() * nTri);
    if (filter && !filter(p)) continue;
    const V = [0, 1, 2].map((j) => { const a = m.indices[3 * p + j] * 3; return [m.positions[a], m.positions[a + 1], m.positions[a + 2]]; });
    const e1 = V[1].map((x, j) => x - V[0][j]), e2 = V[2].map((x, j) => x - V[0][j]);
    const ng = [e1[1] * e2[2] - e1[2] * e2[1], e1[2] * e2[0] - e1[0] * e2[2], e1[0] * e2[1] - e1[1] * e2[0]];
    const area2 = Math.hypot(ng[0], ng[1], ng[2]);
    const maxE = Math.max(Math.hypot(e1[0], e1[1], e1[2]), Math.hypot(e2[0], e2[1], e2[2]), Math.hypot(V[2][0] - V[1][0], V[2][1] - V[1][1], V[2][2] - V[1][2]));
    if (!(area2 > 0) || area2 / (maxE * maxE) < 1e-3) continue; // skip slivers: their f32 normal is ill-conditioned
    const nn = ng.map((x) => x / area2);
    let su = r(), sv = r();
    if (su + sv > 1) { su = 1 - su; sv = 1 - sv; }
    const side = r() < 0.5 ? 1 : -1;
    const theta = Math.exp(Math.log(1e-6) + r() * (Math.log(0.3) - Math.log(1e-6)));
    const t1 = e1.map((x) => x / Math.hypot(e1[0], e1[1], e1[2]));
    const t2 = [nn[1] * t1[2] - nn[2] * t1[1], nn[2] * t1[0] - nn[0] * t1[2], nn[0] * t1[1] - nn[1] * t1[0]];
    const phi = 2 * Math.PI * r();
    const dir = [0, 1, 2].map((j) => Math.cos(theta) * (Math.cos(phi) * t1[j] + Math.sin(phi) * t2[j]) + Math.sin(theta) * side * nn[j]);
    const o = k * 16;
    input.set([...V[0], 0, ...V[1], su, ...V[2], sv, ...dir, side], o);
    u32[o + 3] = p;
    k++;
  }
  return { input, n: k };
}

function visInputs(m: Mesh, pairs: [number, number, number, number, number, number][], infDirs?: number[][]) {
  const input = new Float32Array(pairs.length * 32), u32 = new Uint32Array(input.buffer);
  pairs.forEach(([pa, ua, va, pb, ub, vb], i) => {
    const o = i * 32;
    for (const [slot, p, u, v] of [[0, pa, ua, va], [3, pb, ub, vb]] as const) {
      for (let j = 0; j < 3; j++) { const a = m.indices[3 * p + j] * 3; input.set([m.positions[a], m.positions[a + 1], m.positions[a + 2]], o + (slot + j) * 4); }
      u32[o + slot * 4 + 3] = p; input[o + (slot + 1) * 4 + 3] = u; input[o + (slot + 2) * 4 + 3] = v;
    }
    if (infDirs) { input[o + 24] = 1; input.set(infDirs[i], o + 28); }
  });
  return input;
}

function pointOn(m: Mesh, p: number, u: number, v: number): number[] {
  const V = [0, 1, 2].map((j) => m.indices[3 * p + j] * 3);
  return [0, 1, 2].map((k) => m.positions[V[0] + k] + u * (m.positions[V[1] + k] - m.positions[V[0] + k]) + v * (m.positions[V[2] + k] - m.positions[V[0] + k]));
}

// ---------------------------------------------------------------------------------------------------------------

describe(`T12 BVH traversal (${lane()})`, () => {
  afterAll(releaseTestGpu);
  const report: Record<string, unknown> = { lane: lane() };
  afterAll(() => console.log('T12_REPORT', JSON.stringify(report)));

  const scenes: { name: string; mesh: () => Promise<Mesh | undefined> }[] = [
    { name: 'procedural', mesh: async () => proceduralScene(7) },
    { name: 'sponza', mesh: () => loadGltfMesh() },
  ];

  for (const sc of scenes) {
    it(`(1) ${sc.name}: 10^6 random rays, closest primId + any-hit identical to the f64 reference (MT and Woop); (4) counters 0`, async () => {
      const m = await sc.mesh();
      if (!m) { console.warn(`${sc.name} not present; skipping`); report[`${sc.name}.skipped`] = true; return; }
      const ctx = await getTestGpu();
      const bvh = buildBvh(m.positions, m.indices);
      const rays = makeRays(m, N_RAYS, 1234);
      const t0 = performance.now();
      const ref = cpuReference(m, bvh, rays, N_RAYS);
      report[`${sc.name}.cpuRefMs`] = Math.round(performance.now() - t0);
      // Tie the f64 BVH reference to true brute force on a subset.
      const nb = sc.name === 'sponza' ? 100 : 2000;
      for (let i = 0; i < nb; i++) {
        const o = [rays[8 * i], rays[8 * i + 1], rays[8 * i + 2]], d = [rays[8 * i + 4], rays[8 * i + 5], rays[8 * i + 6]];
        expect(bruteClosest(m.positions, m.indices, o, d).primId).toBe(ref.prim[i]);
        expect(bruteAny(m.positions, m.indices, o, d, rays[8 * i + 3]) ? 1 : 0).toBe(ref.occ[i]);
      }
      for (const v of VARIANTS) {
        const bufs = uploadBvh(ctx.device, bvh, { watertight: v.watertight });
        const res = await run(ctx, v, 'closest_any', bufs, rays, N_RAYS);
        const c = compare(m, rays, ref, res.out, N_RAYS);
        const key = `${sc.name}.${vname(v)}`;
        report[key] = { ...c, steps: res.ctr[0] / N_RAYS / 2, boxTests: res.ctr[1] / N_RAYS / 2, triTests: res.ctr[2] / N_RAYS / 2, flags: res.ctr[3], maxStack: res.ctr[4], gpuMs: res.gpuMs };
        console.log('T12', lane(), key, JSON.stringify(report[key]));
        expect(res.ctr[3], 'overflow/itercap flags').toBe(0);
        expect(c.unexplained, c.examples.join('\n')).toBe(0);
        expect(c.anyUnexplained, c.examples.join('\n')).toBe(0);
        expect(c.precisionRandom + c.anyPrecisionRandom).toBeLessThan(N_RAYS * 1e-4);
        bufs.nodes.destroy(); bufs.tris.destroy();
      }
    }, 900_000);
  }

  it('(2) watertightness: 10^6 rays from inside a closed icosphere never escape with WATERTIGHT (MT: reported)', async () => {
    const ctx = await getTestGpu();
    const center = [3.7, -1.2, 5.1], radius = 2.3;
    const m = icosphere(5, radius, center); // 20480 triangles
    const bvh = buildBvh(m.positions, m.indices);
    const r = rng(99);
    const nv = m.positions.length / 3;
    const mk = (fromCenter: boolean) => {
      const rays = new Float32Array(N_RAYS * 8);
      for (let i = 0; i < N_RAYS; i++) {
        let o = center;
        if (!fromCenter) { const q = randomDir(r), s = 0.9 * radius * Math.cbrt(r()); o = [center[0] + s * q[0], center[1] + s * q[1], center[2] + s * q[2]]; }
        let d: number[];
        const kind = i % 10;
        if (kind === 0) {
          // Exactly toward a vertex: the case where non-watertight tests leak.
          const a = Math.floor(r() * nv) * 3;
          d = [0, 1, 2].map((k) => m.positions[a + k] - o[k]);
        } else if (kind === 1) {
          // Toward a point on a triangle edge.
          const p = Math.floor(r() * (m.indices.length / 3)), e = Math.floor(r() * 3), s = r();
          const va = m.indices[3 * p + e] * 3, vb = m.indices[3 * p + ((e + 1) % 3)] * 3;
          d = [0, 1, 2].map((k) => m.positions[va + k] + s * (m.positions[vb + k] - m.positions[va + k]) - o[k]);
        } else d = randomDir(r);
        rays.set([o[0], o[1], o[2], 1e30, d[0], d[1], d[2], 0], i * 8);
      }
      return rays;
    };
    for (const [label, fromCenter] of [['center', true], ['interior', false]] as const) {
      const rays = mk(fromCenter);
      for (const v of VARIANTS) {
        const bufs = uploadBvh(ctx.device, bvh, { watertight: v.watertight });
        const res = await run(ctx, v, 'closest_any', bufs, rays, N_RAYS);
        let miss = 0, anyMiss = 0;
        const byKind = { towardVertex: 0, towardEdge: 0, random: 0 };
        for (let i = 0; i < N_RAYS; i++) {
          if (res.out[4 * i + 1] === BVH_MISS) { miss++; byKind[i % 10 === 0 ? 'towardVertex' : i % 10 === 1 ? 'towardEdge' : 'random']++; }
          if (res.out[4 * i + 2] === BVH_MISS) anyMiss++;
        }
        const key = `watertight.${label}.${vname(v)}`;
        report[key] = { misses: miss, missesByRayKind: byKind, anyMisses: anyMiss, flags: res.ctr[3] };
        console.log('T12', lane(), key, JSON.stringify(report[key]));
        expect(res.ctr[3]).toBe(0);
        if (v.watertight) { expect(miss).toBe(0); expect(anyMiss).toBe(0); }
        bufs.nodes.destroy(); bufs.tris.destroy();
      }
    }
  }, 600_000);

  it('(3) self-intersection: offset_ray spawns at grazing angles never re-hit the origin triangle; visible()/visibleInf() consistent', async () => {
    const ctx = await getTestGpu();
    // Scene V: tessellated floor (flat, many coplanar neighbours) + a floating icosphere, off-origin.
    const floor = proceduralFloor();
    const sph = icosphere(4, 1.0, [12.3 + 1, 4.5 + 2.5, -7.8 - 1]);
    const m = concat(floor, sph);
    const nFloor = floor.indices.length / 3;
    const proc = proceduralScene(7);
    const sponza = await loadGltfMesh();
    const targets: [string, Mesh, ((p: number) => boolean) | undefined][] = [['sceneV', m, undefined], ['procedural', proc, undefined]];
    if (sponza) targets.push(['sponza', sponza, undefined]);
    for (const [name, mesh] of targets) {
      const bvh = buildBvh(mesh.positions, mesh.indices);
      const { input, n } = spawnInputs(mesh, 1 << 20, 5);
      const u32 = new Uint32Array(input.buffer);
      for (const v of VARIANTS) {
        const bufs = uploadBvh(ctx.device, bvh, { watertight: v.watertight });
        const res = await run(ctx, v, 'spawn', bufs, input, n);
        let self = 0, floorHits = 0;
        const ex: string[] = [];
        for (let i = 0; i < n; i++) {
          const hit = res.out[4 * i + 1], origin = u32[i * 16 + 3];
          if (hit === origin) { self++; if (ex.length < 5) ex.push(`spawn ${i} prim ${origin} t=${new Float32Array(res.out.buffer)[4 * i]} side=${input[i * 16 + 15]}`); }
          if (name === 'sceneV' && origin < nFloor && hit < nFloor) floorHits++; // floor is one plane: any floor hit is false
        }
        const key = `self.${name}.${vname(v)}`;
        report[key] = { n, selfHits: self, floorPlaneHits: name === 'sceneV' ? floorHits : undefined, flags: res.ctr[3] };
        console.log('T12', lane(), key, JSON.stringify(report[key]));
        expect(self, ex.join('\n')).toBe(0);
        expect(floorHits).toBe(0);
        expect(res.ctr[3]).toBe(0);
        if (name === 'procedural' && !v.watertight) {
          // Negative control: without offset_ray the same spawns must self-intersect (the test has power).
          const params = new Float32Array(24); new Uint32Array(params.buffer)[2] = 1;
          const ctl = await run(ctx, v, 'spawn', bufs, input, n, { params });
          let selfCtl = 0;
          for (let i = 0; i < n; i++) if (ctl.out[4 * i + 1] === u32[i * 16 + 3]) selfCtl++;
          report['self.procedural.mt.noOffsetControl'] = selfCtl;
          expect(selfCtl).toBeGreaterThan(0);
        }
        bufs.nodes.destroy(); bufs.tris.destroy();
      }
    }

    // visible(): floor–floor pairs (segment in the plane) and sphere chords must be visible; floor→through-sphere
    // pairs and visibleInf compared with the f64 reference (ambiguous contacts skipped).
    const bvh = buildBvh(m.positions, m.indices);
    const r = rng(77);
    const nS = sph.indices.length / 3;
    const bary = () => { let u = r(), w = r(); if (u + w > 1) { u = 1 - u; w = 1 - w; } return [u, w]; };
    const pairs: [number, number, number, number, number, number][] = [];
    const expect1: number[] = [];
    const NP = 1 << 18;
    for (let i = 0; i < NP; i++) {
      const kind = i % 3;
      let pa: number, pb: number;
      if (kind === 0) { pa = Math.floor(r() * nFloor); pb = Math.floor(r() * nFloor); }
      else if (kind === 1) { pa = nFloor + Math.floor(r() * nS); pb = nFloor + Math.floor(r() * nS); if (pa === pb) pb = nFloor + ((pb - nFloor + 1) % nS); }
      else { pa = Math.floor(r() * nFloor); pb = nFloor + Math.floor(r() * nS); }
      const [ua, va] = bary(), [ub, vb] = bary();
      pairs.push([pa, ua, va, pb, ub, vb]);
      if (kind < 2) expect1.push(1);
      else {
        const a = pointOn(m, pa, ua, va), b = pointOn(m, pb, ub, vb);
        const d = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
        const h = bvhTrace64(bvh, m.positions, m.indices, a, d, 1, false, pa, pb);
        const hs = bvhTrace64(bvh, m.positions, m.indices, a, d, Infinity, false, pa, pb);
        // Ambiguous: an occluder within 1e-3 of either endpoint (grazing contact with the sphere) → -1.
        expect1.push(h.primId === BVH_MISS ? (hs.primId !== BVH_MISS && hs.t < 1 + 1e-3 ? -1 : 1) : h.t > 1 - 1e-3 || h.t < 1e-3 ? -1 : 0);
      }
    }
    const infDirs: number[][] = [], infPairs: [number, number, number, number, number, number][] = [], infExp: number[] = [];
    for (let i = 0; i < NP; i++) {
      const pa = Math.floor(r() * nFloor), [ua, va] = bary();
      const d = randomDir(r); d[1] = Math.abs(d[1]) * (i % 2 ? 1 : 1e-3); // upward, half of them grazing
      const l = Math.hypot(d[0], d[1], d[2]); const dir = d.map((x) => Math.fround(x / l));
      infPairs.push([pa, ua, va, pa, ua, va]); infDirs.push(dir);
      const h = bvhTrace64(bvh, m.positions, m.indices, pointOn(m, pa, ua, va), dir, Infinity, true, pa);
      infExp.push(h.primId === BVH_MISS ? 1 : h.primId < nFloor || h.t < 1e-3 ? -1 : 0);
    }
    for (const v of VARIANTS) {
      const bufs = uploadBvh(ctx.device, bvh, { watertight: v.watertight });
      for (const [label, inp, exp] of [['segment', visInputs(m, pairs), expect1], ['inf', visInputs(m, infPairs, infDirs), infExp]] as const) {
        const res = await run(ctx, v, 'vis', bufs, inp, exp.length);
        let bad = 0, amb = 0, visCount = 0;
        const perKind = [0, 0, 0];
        for (let i = 0; i < exp.length; i++) {
          const g = res.out[4 * i];
          visCount += g;
          if (exp[i] < 0) { amb++; continue; }
          if (g !== exp[i]) { bad++; perKind[label === 'segment' ? i % 3 : 0]++; }
        }
        const key = `visible.${label}.${vname(v)}`;
        report[key] = { n: exp.length, mismatches: bad, perKind, ambiguousSkipped: amb, visibleFraction: visCount / exp.length, flags: res.ctr[3] };
        console.log('T12', lane(), key, JSON.stringify(report[key]));
        expect(bad).toBe(0);
        expect(res.ctr[3]).toBe(0);
      }
      bufs.nodes.destroy(); bufs.tris.destroy();
    }
  }, 600_000);

  it('(5) alpha_pass hook (CUSTOM_ALPHA) and BVH_DECLARE_BINDINGS: rejected prims are skipped by closest and any-hit', async () => {
    const ctx = await getTestGpu();
    const m = proceduralScene(7);
    const bvh = buildBvh(m.positions, m.indices);
    // Reference: the same scene with only even primIds (the hook rejects odd ones).
    const even = { positions: m.positions, indices: new Uint32Array(Array.from({ length: Math.ceil(m.indices.length / 6) }, (_, k) => [m.indices[6 * k], m.indices[6 * k + 1], m.indices[6 * k + 2]]).flat()) };
    const bvhEven = buildBvh(even.positions, even.indices);
    const n = 1 << 18;
    const rays = makeRays(m, n, 4321);
    const ref = cpuReference(even, bvhEven, rays, n);
    for (let i = 0; i < n; i++) if (ref.prim[i] !== BVH_MISS) ref.prim[i] *= 2;
    for (const watertight of [false, true]) {
      const shader = composeWgsl('tests/bvh_alpha.wgsl', {
        sources: { ...shaderSources, 'tests/bvh_alpha.wgsl': ALPHA_KERNEL },
        defines: { WATERTIGHT: watertight, CUSTOM_ALPHA: 1, BVH_DECLARE_BINDINGS: 1, BVH_GROUP: 0, BVH_BINDING_NODES: 0, BVH_BINDING_TRIS: 1 },
        features: ctx.features, wgslLanguageFeatures: ctx.wgslLanguageFeatures,
      });
      const module = await createCheckedShaderModule(ctx.device, shader, 'bvh_alpha');
      const pipeline = await ctx.device.createComputePipelineAsync({ layout: 'auto', compute: { module, entryPoint: 'main' } });
      const bufs = uploadBvh(ctx.device, bvh, { watertight });
      const inBuf = storage(ctx.device, rays, 'inp'), outBuf = storage(ctx.device, n * 16, 'outp');
      ctx.device.pushErrorScope('validation');
      const bg = ctx.device.createBindGroup({ layout: pipeline.getBindGroupLayout(0), entries: [
        { binding: 0, resource: { buffer: bufs.nodes } }, { binding: 1, resource: { buffer: bufs.tris } },
        { binding: 2, resource: { buffer: inBuf } }, { binding: 3, resource: { buffer: outBuf } }] });
      const enc = ctx.device.createCommandEncoder();
      const pass = enc.beginComputePass(); pass.setPipeline(pipeline); pass.setBindGroup(0, bg); pass.dispatchWorkgroups(n / 64); pass.end();
      ctx.device.queue.submit([enc.finish()]);
      const err = await ctx.device.popErrorScope();
      expect(err?.message).toBeUndefined();
      const out = new Uint32Array(await readBuffer(ctx.device, outBuf, n * 16));
      let odd = 0, flags = 0;
      for (let i = 0; i < n; i++) flags |= out[4 * i + 3] & 3;
      for (let i = 0; i < n; i++) if ((out[4 * i + 1] !== BVH_MISS && out[4 * i + 1] & 1) || (out[4 * i + 2] !== BVH_MISS && out[4 * i + 2] & 1)) odd++;
      const c = compare(m, rays, ref, out, n);
      const key = `alpha.${watertight ? 'woop' : 'mt'}`;
      report[key] = { oddPrimHits: odd, unexplained: c.unexplained, anyUnexplained: c.anyUnexplained, match: c.match, tie: c.tie, flags };
      console.log('T12', lane(), key, JSON.stringify(report[key]));
      expect(odd).toBe(0);
      expect(flags).toBe(0);
      expect(c.unexplained, c.examples.join('\n')).toBe(0);
      expect(c.anyUnexplained, c.examples.join('\n')).toBe(0);
      inBuf.destroy(); outBuf.destroy(); bufs.nodes.destroy(); bufs.tris.destroy();
    }
  }, 600_000);

  it('throughput at 1080p-equivalent ray counts (Mrays/s)', async () => {
    const ctx = await getTestGpu();
    const m = (await loadGltfMesh()) ?? proceduralScene(7);
    const name = m.indices.length > 100_000 ? 'sponza' : 'procedural';
    const bvh = buildBvh(m.positions, m.indices);
    const b = meshBounds(m);
    const c = [0, 1, 2].map((k) => 0.5 * (b.min[k] + b.max[k]));
    const ext = [0, 1, 2].map((k) => b.max[k] - b.min[k]);
    // Camera inside the scene looking along the long axis (+X for Sponza), vfov 60°.
    const eye = [c[0] - 0.35 * ext[0], b.min[1] + 0.25 * ext[1], c[2]];
    const light = [c[0], b.min[1] + 0.8 * ext[1], c[2]];
    const params = new Float32Array(24), pu = new Uint32Array(params.buffer);
    pu[1] = 17; pu[2] = W1080; pu[3] = H1080;
    params.set([eye[0], eye[1], eye[2], Math.tan(Math.PI / 6), 1, 0, 0, W1080 / H1080, 0, 0, 1, 0, 0, 1, 0, 0, light[0], light[1], light[2], 0], 4);
    const rnd = new Float32Array(24), ru = new Uint32Array(rnd.buffer);
    ru[1] = 5;
    rnd.set([b.min[0] + 0.1 * ext[0], b.min[1] + 0.05 * ext[1], b.min[2] + 0.1 * ext[2], 0, b.max[0] - 0.1 * ext[0], b.max[1] - 0.3 * ext[1], b.max[2] - 0.1 * ext[2], 0], 4);
    const N = W1080 * H1080;
    const grid: [number, number] = [W1080 / 8, H1080 / 8];
    for (const watertight of [false, true]) {
      const v: Variant = { watertight, stats: false };
      const bufs = uploadBvh(ctx.device, bvh, { watertight });
      const prim = await run(ctx, v, 'perf_primary', bufs, null, N, { params, grid, reps: 7, readOut: false });
      const sec = await run(ctx, v, 'perf_secondary', bufs, null, N, { params, grid, reps: 7, outBuf: prim.outBuf, readOut: false });
      const sh = await run(ctx, v, 'perf_shadow', bufs, null, N, { params, grid, reps: 7, outBuf: prim.outBuf });
      const rand = await run(ctx, v, 'perf_random', bufs, null, N, { params: rnd, reps: 7, readOut: false });
      prim.outBuf.destroy(); rand.outBuf.destroy();
      let hits = 0;
      for (let i = 0; i < N; i++) if (sh.out[4 * i] !== BVH_MISS) hits++;
      const mr = (ms: number) => +(N / ms / 1e3).toFixed(1);
      const key = `perf.${name}.${watertight ? 'woop' : 'mt'}`;
      report[key] = {
        rays: N, primaryHitFraction: +(hits / N).toFixed(3),
        primary: { ms: +prim.gpuMs.toFixed(2), mrays: mr(prim.gpuMs) }, secondaryHemisphere: { ms: +sec.gpuMs.toFixed(2), mrays: mr(sec.gpuMs) },
        shadowAny: { ms: +sh.gpuMs.toFixed(2), mrays: mr(sh.gpuMs) }, randomIncoherent: { ms: +rand.gpuMs.toFixed(2), mrays: mr(rand.gpuMs) },
        flags: prim.ctr[3] | sec.ctr[3] | sh.ctr[3] | rand.ctr[3], timer: ctx.features.has('timestamp-query') ? 'timestamp' : 'wall',
      };
      console.log('T12', lane(), key, JSON.stringify(report[key]));
      expect(prim.ctr[3] | sec.ctr[3] | sh.ctr[3] | rand.ctr[3]).toBe(0);
      bufs.nodes.destroy(); bufs.tris.destroy();
    }
  }, 600_000);

  it.runIf(lane() === 'chrome')('builds in a Worker with transfer lists, bit-identical to the synchronous build', async () => {
    const { buildBvhInWorker, terminateBvhWorker } = await import('../../src/core/bvh/build-in-worker.ts');
    const m = proceduralScene(3);
    const a = buildBvh(m.positions, m.indices);
    const b = await buildBvhInWorker(m.positions, m.indices);
    terminateBvhWorker();
    expect(m.positions.length).toBeGreaterThan(0); // inputs were copied, not detached
    expect(new Uint32Array(b.nodes.buffer)).toEqual(new Uint32Array(a.nodes.buffer));
    expect(new Uint32Array(b.trisW.buffer)).toEqual(new Uint32Array(a.trisW.buffer));
    expect(b.primOrder).toEqual(a.primOrder);
  });
});

function proceduralFloor(): Mesh {
  // 96×96 indexed grid at y = 4.5 (off-origin, like the procedural scene), 18432 triangles.
  const n = 96, pos: number[] = [], idx: number[] = [];
  for (let j = 0; j <= n; j++) for (let i = 0; i <= n; i++) pos.push(12.3 - 10 + (20 * i) / n, 4.5, -7.8 - 10 + (20 * j) / n);
  for (let j = 0; j < n; j++) for (let i = 0; i < n; i++) { const a = j * (n + 1) + i; idx.push(a, a + n + 2, a + 1, a, a + n + 1, a + n + 2); }
  return { positions: new Float32Array(pos), indices: new Uint32Array(idx) };
}

function concat(a: Mesh, b: Mesh): Mesh {
  const positions = new Float32Array(a.positions.length + b.positions.length);
  positions.set(a.positions); positions.set(b.positions, a.positions.length);
  const off = a.positions.length / 3;
  const indices = new Uint32Array(a.indices.length + b.indices.length);
  indices.set(a.indices); for (let i = 0; i < b.indices.length; i++) indices[a.indices.length + i] = b.indices[i] + off;
  return { positions, indices };
}
