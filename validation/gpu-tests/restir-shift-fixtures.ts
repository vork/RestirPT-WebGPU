// WP-B helpers for the shift GPU tests (restir-shift.gpu.test.ts; restir-api.md §1.2, §6.1–§6.2): custom test
// pipelines on top of RestirKernel (G0/G1 of the kernel, a test G2 of storage buffers), f32 bit helpers.
import { readBuffer } from '../../src/core/gpu/readback.ts';
import type { Defines } from '../../src/core/gpu/wgsl-composer.ts';
import type { RestirKernel } from '../../src/core/render/restir/kernel.ts';
import type { RsDispatchCpu } from '../../src/core/render/restir/layout.ts';
import { storageBuffer } from './restir-fixtures.ts';

const f32b = new Float32Array(1);
const u32b = new Uint32Array(f32b.buffer);
export const bitsToF32 = (u: number): number => { u32b[0] = u; return f32b[0]; };
export const f32ToBits = (x: number): number => { f32b[0] = x; return u32b[0]; };
export const toF32 = (x: number): number => Math.fround(x);

export interface TestPipeline {
  pipeline: GPUComputePipeline;
  layout: GPUBindGroupLayout;
  /** Bind `buffers` (G2 bindings 0…n−1, storage) and dispatch [x, y] workgroups with the kernel's G0 / G1. */
  run(buffers: GPUBuffer[], work: [number, number], d?: Partial<RsDispatchCpu>, textures?: GPUTextureView[]): Promise<void>;
}

/**
 * Compile a test pipeline from an in-memory WGSL source: G2 = `nBuffers` read-write storage buffers (bindings 0…),
 * then `textures` sampled textures (uint | unfilterable-float) at the following bindings. Uses the kernel's G0 (frame,
 * env, RestirParams, lights, records, RsDispatch) and G1 (scene). No debug group (DEBUG_NO_BINDINGS).
 */
export async function testPipeline(k: RestirKernel, name: string, src: string, entry: string, nBuffers: number,
  defines: Defines = {}, textures: GPUTextureSampleType[] = [], extraSources: Record<string, string> = {}): Promise<TestPipeline> {
  const c = GPUShaderStage.COMPUTE;
  const entries: GPUBindGroupLayoutEntry[] = [];
  for (let i = 0; i < nBuffers; i++) entries.push({ binding: i, visibility: c, buffer: { type: 'storage' } });
  textures.forEach((t, j) => entries.push({ binding: nBuffers + j, visibility: c, texture: { sampleType: t } }));
  const layout = k.device.createBindGroupLayout({ label: `${name}-g2`, entries });
  const file = `${name}.wgsl`;
  const pipeline = await k.compile(file, entry, k.customDefines(defines, true), k.customLayout(layout, true), name, { ...extraSources, [file]: src });
  return {
    pipeline, layout,
    async run(buffers, work, d = {}, texViews = []) {
      const g2 = k.device.createBindGroup({
        layout,
        entries: [...buffers.map((b, i) => ({ binding: i, resource: { buffer: b } })), ...texViews.map((v, j) => ({ binding: nBuffers + j, resource: v }))],
      });
      k.beginSubmit();
      const enc = k.device.createCommandEncoder({ label: name });
      k.encodeCustom(enc, pipeline, g2, d, work, true);
      k.device.queue.submit([enc.finish()]);
      await k.device.queue.onSubmittedWorkDone();
    },
  };
}

export async function readU32(device: GPUDevice, b: GPUBuffer, bytes = b.size): Promise<Uint32Array> {
  return new Uint32Array(await readBuffer(device, b, bytes));
}
export { storageBuffer };

// ------------------------------------------------------------------------------------------------ T3 round-trip kernel

/** Case bins of the T3 statistics (restir-api.md §6.1 T3-4 list), × 2 for k = 2 (∅: d = 2) vs k > 2 (∅: d > 2). */
export const T3_CASES = ['a-delta', 'a-area', 'a-tri', 'a-sun', 'f-env', 'b', 'b-env', 'c-tri', 'c-env', 'd', 'e', 'deep-nee', 'deep-bsdf', 'none-tri', 'none-env',
  // M6 (restir-m6-api.md §4 T3-M6): Mode-B analytic crossings (deep crossings count in deep-bsdf); bins 30–35 use the
  // skip bits 16–19 of RsDispatch.flags (passId holds bins 0–31)
  'none-ana', 'd-ana', 'c-ana'] as const;
export const T3_NBINS = T3_CASES.length * 2;
/** M6 extra counters after the class table: per category (ℓ_{k−1} = G_R, ℓ_k = G_R, side flip at x_k, rc segment crossing a
 *  cutout card) {trials, rtOk, logic} (restir-m6-api.md §1.5, §4). */
export const T3_X_CATS = ['grKm1', 'grK', 'sideFlip', 'alphaCard'] as const;
export const T3_X_WORDS = 3 * T3_X_CATS.length;
export const T3_BIN_NAMES = T3_CASES.flatMap((c) => [`${c}/k2`, `${c}/k>2`]);
/** Per-bin stats words. */
export const T3S = {
  trials: 0, fwdOk: 1, rtOk: 2, logic: 3, fp: 4, visZero: 5, sigLogic: 6, sigFp: 7, jViol: 8, fViol: 9, skipped: 10,
  fwdDefined: 11, invDefined: 12, fCond: 13, words: 48, fwdCodes: 16, invCodes: 32,
} as const;
export const T3_HIST_BINS = 64;
/** Global regions after the log2 J histogram: chain (T3-5) counters and per-endpoint-type (U5) counters. */
export const T3_CHAIN = { trials: 0, both: 1, jViol: 2, defMismatch: 3, fViol: 4, platform: 5 } as const;
export const T3_EP_WORDS = 3;   // per endpoint type LT_* (8): fwdOk, rtOk, logic
/** Path-class table (dense PSS sweep): key = flags bits 0–21 | (ℓ₁ | ℓ₂ << 3 | ℓ₃ << 6) << 22, count. */
export const T3_CLASS_SLOTS = 4096;
/** Dense sweep: destination texels live in the dual buffer at this word offset (8 words each: vbuf, geo bits). */
export const DENSE_OFF = 64;
export const T3_VIOL_WORDS = 16;
export const T3_VIOL_CAP = 4096;
/** Violation kinds (viol record word 1 high byte). */
export const T3V = { invUndefined: 1, invZero: 2, sig: 3, J: 4, F: 5, selfUndefined: 6, selfJ: 7, selfF: 8, selfSig: 9, selfZero: 10, platform: 20 } as const;

/**
 * T3-0 / T3-1 / T4 kernel. One thread per trial = (atlas pixel p, dumped candidate di, partner j):
 *   mode 0 (self, T3-0): T_{p→p}(x̄) must be defined, σ equal (replayed prefix ids + lobes = the base's), |log2 J| < 2⁻¹⁶,
 *          |F(ȳ)/F(x̄) − 1| < 1e-4 per channel.
 *   mode 1 (round trip, T3-1/T4): q = p + a disk offset of 1…30 px; ȳ = T_{p→q}(x̄); if defined with F > 0, write back
 *          (F ← FJ/J, jDen ← J·jDen) and x̄' = T_{q→p}(ȳ): defined, σ(x̄') = σ(x̄), |ln(J·J')| < 1e-4, F(x̄') ≈ F(x̄).
 * Both shifts go through ONE shift_hybrid call site (loop), RS_REPLAY = 1, RS_SHIFT_TRACE = 1 (replayed prefix).
 * Failures of the inverse are classified FP-BOUNDARY (|margin| < 2⁻¹⁶ of the deciding pair, or a replay divergence
 * at a triangle edge: barycentric distance < 1e-6) or LOGIC (everything else), gap-rc §10.3.
 * G2: 0 candDump, 1 stats (atomic), 2 violations, 3 dual trace records; 4 rsVbuf, 5 rsGeo. RsDispatch: treeBase = first trial,
 * treeCount = trials, round = mode (0 self, 1 round trip, 2 T5 census, 3 T3-5 chain), flags = partners per candidate
 * (bit 31: T5 side B), passId = bin skip mask, rowBase = dual trace stride, rowEnd = T5 offset δ (i16 × 2)
 * (0 = off; every rowBase-th trial records both shifts for the f64 dual).
 */
/** Rank-1 lattice over the first 12 path dims (BSDF slots 0–3 of vertices 1–3; Korobov a = 154303, N = 2²⁰) with a
 *  per-pair Cranley–Patterson shift; every other dim from the path hash. WGSL, shared by both dense kernels. */
export const DENSE_LATTICE_WGSL = `
fn dense_lattice(seed: vec2u, vertex: u32, slot: u32, n: u32, pairSeed: u32) -> u32 {
  if (vertex < 1u || vertex > 3u || slot > 3u) { return path_hash(seed, vertex, slot); }
  let dim = (vertex - 1u) * 4u + slot;
  var g = 1u;
  for (var i = 0u; i < dim; i++) { g = (g * 154303u) & 0xFFFFFu; }
  let x = (n * g) & 0xFFFFFu;
  return (x << 12u) + pcg4d(vec4u(pairSeed, dim, 0x51ed270bu, 0x2c1b3c6du)).x;
}
`;

/** Dense PSS sweep, path-tree pass: every atlas pixel is the SAME primary hit (texel in the arena words 0–7) with lattice
 *  point treeBase + ai; tree 0 of the production pathtree_run, candidate dump on. G2: 0 resOut, 1 arena, 4 candDump. */
export const DENSE_INIT_WGSL = `
#include "restir/frame.wgsl"
#include "restir/reservoir.wgsl"
#include "restir/queue.wgsl"
fn rs_vbuf(px: vec2u) -> vec4u { return vec4u(rsArena.words[0], rsArena.words[1], rsArena.words[2], rsArena.words[3]); }
fn rs_geo(px: vec2u) -> vec4f { return bitcast<vec4f>(vec4u(rsArena.words[4], rsArena.words[5], rsArena.words[6], rsArena.words[7])); }
var<private> latIdx: u32;
fn rs_rng_override(seed: vec2u, vertex: u32, slot: u32) -> u32 { return dense_lattice(seed, vertex, slot, latIdx, rsArena.words[8]); }
${DENSE_LATTICE_WGSL}
#include "path/pathtree.wgsl"
@compute @workgroup_size(8, 8, 1)
fn dense_initial(@builtin(global_invocation_id) gid: vec3u) {
  let p = rs_pix(vec2u(gid.x, gid.y + rsDispatch.rowBase));
  if (!p.valid) { return; }
  latIdx = rsDispatch.treeBase + p.ai;
  pathtree_run(p, rs_frame_key(p.member, rs_t(), p.localIdx), 0u, 1u, true, true);
}
`;

/** Dual (T3-D) trace records: word 0 of the buffer = count; record r at 4 + r·T3_DUAL_WORDS (layout: decodeDualRecord). */
export const T3_DUAL_WORDS = 80;
export const T3_DUAL_CAP = 131072;
/** An axis-aligned-in-its-frame card (alpha fixture): centre, unit axes u / v with half sizes, unit normal. */
export interface T3Card { c: [number, number, number]; u: [number, number, number]; v: [number, number, number]; hu: number; hv: number }
const v4 = (a: number[], w: number) => `vec4f(${a.map((x) => x.toFixed(7)).join(', ')}, ${w.toFixed(7)})`;
function cardsWgsl(cards: T3Card[]): string {
  const n = cards.length;
  const items = cards.flatMap((k) => {
    const nrmv = [k.u[1] * k.v[2] - k.u[2] * k.v[1], k.u[2] * k.v[0] - k.u[0] * k.v[2], k.u[0] * k.v[1] - k.u[1] * k.v[0]];
    return [v4(k.c, k.hu), v4(k.u, k.hv), v4(nrmv, 0)];
  });
  if (!items.length) items.push(v4([0, 0, 0], 0));
  return `const T3_NCARDS: u32 = ${n}u;\nconst T3_CARDS = array<vec4f, ${items.length}>(${items.join(', ')});`;
}

export const T3_WGSL_BODY = (cards: T3Card[] = []) => `
#include "restir/shift.wgsl"
${DENSE_LATTICE_WGSL}
@group(2) @binding(0) var<storage, read_write> dump: array<u32>;
@group(2) @binding(1) var<storage, read_write> stats: array<atomic<u32>>;
@group(2) @binding(2) var<storage, read_write> viol: array<atomic<u32>>;

${cardsWgsl(cards)}
const NB: u32 = ${T3_NBINS}u;
const SW: u32 = ${T3S.words}u;
const CH0: u32 = ${T3_NBINS * T3S.words + T3_HIST_BINS}u;
fn stg(w: u32) { atomicAdd(&stats[CH0 + w], 1u); }
fn stEp(ep: u32, w: u32) { atomicAdd(&stats[CH0 + 8u + ep * ${T3_EP_WORDS}u + w], 1u); }
const CL0: u32 = CH0 + 8u + 8u * ${T3_EP_WORDS}u;
fn stClass(key: u32) {
  var i = pcg3d(vec3u(key, 0x2545f491u, 7u)).x % ${T3_CLASS_SLOTS}u;
  for (var n = 0u; n < 64u; n++) {
    let r = atomicCompareExchangeWeak(&stats[CL0 + 2u * i], 0u, key);
    if (r.exchanged || r.old_value == key) { atomicAdd(&stats[CL0 + 2u * i + 1u], 1u); return; }
    if (r.old_value != 0u) { i = (i + 1u) % ${T3_CLASS_SLOTS}u; }
  }
}
#if T3_DENSE
// Dense PSS sweep (restir-api.md §6.1): destinations from the dual buffer (0 = the source pixel, 1… partners); the
// first 12 path dims (BSDF slots 0–3 of vertices 1–3) from a rank-1 lattice point (RS_RNG_OVERRIDE, the same function
// as the dense path-tree pass).
fn shift_dst_load(px: vec2u) -> ShiftDst {
  let o = ${DENSE_OFF}u + 8u * px.x;
  var d: ShiftDst;
  d.prim = atomicLoad(&dual[o]);
  d.valid = d.prim != 0xFFFFFFFFu;
  d.bary = bitcast<vec2f>(vec2u(atomicLoad(&dual[o + 1u]), atomicLoad(&dual[o + 2u])));
  d.thr = bitcast<f32>(atomicLoad(&dual[o + 7u]));
  d.camPos = rs_cam_pos();
  return d;
}
var<private> latIdx: u32;
fn rs_rng_override(seed: vec2u, vertex: u32, slot: u32) -> u32 {
  return dense_lattice(seed, vertex, slot, latIdx, atomicLoad(&dual[${DENSE_OFF - 1}u]));
}
#endif
@group(2) @binding(3) var<storage, read_write> dual: array<atomic<u32>>;
const DW: u32 = ${T3_DUAL_WORDS}u;
var<private> trPrim: array<u32, 8>;
var<private> trLobe: array<u32, 8>;
var<private> trEdge: array<f32, 8>;
var<private> trCos: f32;
var<private> trSlot: u32;      // dual record word offset (0 = not recording)
var<private> trNP: u32;
fn dw(i: u32, v: u32) { atomicStore(&dual[trSlot + i], v); }
fn rs_trace_vertex(b: u32, prim: u32, u: f32, v: f32, lobe: u32) {
  if (b < 8u) { trPrim[b] = prim; trLobe[b] = lobe; trEdge[b] = min(min(u, v), 1.0 - u - v); }
  if (trSlot != 0u && b <= 7u) { let o = 26u + 4u * (b - 1u); dw(o, prim); dw(o + 1u, bitcast<u32>(u)); dw(o + 2u, bitcast<u32>(v)); dw(o + 3u, lobe); }
}
fn rs_trace_pair(j: u32, ok: bool, margin: f32, term: u32) {
  if (trSlot != 0u && trNP < 10u) { dw(54u + 2u * trNP, j | (select(0u, 1u, ok) << 4u) | (term << 5u)); dw(55u + 2u * trNP, bitcast<u32>(margin)); }
  trNP++;
}
fn rs_trace_light(dir: vec3f, pos: vec3f, inf: bool) {
  if (trSlot != 0u) {
    dw(16u, bitcast<u32>(dir.x)); dw(17u, bitcast<u32>(dir.y)); dw(18u, bitcast<u32>(dir.z));
    dw(19u, bitcast<u32>(pos.x)); dw(20u, bitcast<u32>(pos.y)); dw(21u, bitcast<u32>(pos.z)); dw(22u, select(0u, 1u, inf));
  }
}
fn rs_trace_escape(dir: vec3f) {
  if (trSlot != 0u) { dw(23u, bitcast<u32>(dir.x)); dw(24u, bitcast<u32>(dir.y)); dw(25u, bitcast<u32>(dir.z)); }
}
var<private> trDist: f32;
var<private> trEdgeK: f32;
fn rs_trace_recon(cosY: f32, cosK: f32, dist: f32, edgeK: f32) { trCos = min(cosY, cosK); trDist = dist; trEdgeK = edgeK; }
// M6 hooks (Mode-B crossing point for the dual; glass side and alpha-card segment counters, RS_M6_TRACE)
fn rs_trace_cross(n: vec3f, z: vec3f) {
  if (trSlot != 0u) {
    dw(16u, bitcast<u32>(n.x)); dw(17u, bitcast<u32>(n.y)); dw(18u, bitcast<u32>(n.z));
    dw(19u, bitcast<u32>(z.x)); dw(20u, bitcast<u32>(z.y)); dw(21u, bitcast<u32>(z.z)); dw(22u, 2u);
  }
}
var<private> trSide: u32;
var<private> trSeg: bool;
fn rs_trace_side(back: bool) { trSide = select(1u, 2u, back); }
fn t3_seg_card(a: vec3f, b: vec3f) -> bool {
  for (var i = 0u; i < T3_NCARDS; i++) {
    let c = T3_CARDS[3u * i]; let u = T3_CARDS[3u * i + 1u]; let n = T3_CARDS[3u * i + 2u].xyz;
    let da = dot(a - c.xyz, n); let db = dot(b - c.xyz, n);
    if (da * db >= 0.0) { continue; }
    let pnt = a + (b - a) * (da / (da - db));
    let v = cross(n, u.xyz);
    if (abs(dot(pnt - c.xyz, u.xyz)) <= c.w && abs(dot(pnt - c.xyz, v)) <= u.w) { return true; }
  }
  return false;
}
fn rs_trace_seg(a: vec3f, b: vec3f, valid: bool) { trSeg = valid && t3_seg_card(a, b); }
const X0: u32 = CL0 + 2u * ${T3_CLASS_SLOTS}u;
fn stX(mask: u32, w: u32) { for (var c = 0u; c < ${T3_X_CATS.length}u; c++) { if (((mask >> c) & 1u) != 0u) { atomicAdd(&stats[X0 + 3u * c + w], 1u); } } }

fn t3_case(f: u32) -> u32 {
  let d = rf_d(f); let k = rf_k(f); let tech = rf_tech(f); let ep = rf_ep(f);
  var c = 12u;
  if (tech == RS_TECH_BSDF_ANALYTIC) {
    if (k == 0u) { c = 15u; } else if (k == d) { c = 16u; } else if (k + 1u == d) { c = 17u; }
    return 2u * c + select(0u, 1u, select(k > 2u, d > 2u, k == 0u));
  }
  if (k == 0u) { c = select(13u, 14u, tech == RS_TECH_BSDF_ENV); }
  else if (tech == RS_TECH_NEE && k == d) {
    if (ep == LT_POINT || ep == LT_SPOT) { c = 0u; } else if (ep == LT_RECT || ep == LT_DISK) { c = 1u; }
    else if (ep == LT_TRI) { c = 2u; } else if (ep == LT_SUN) { c = 3u; } else { c = 4u; }
  } else if (tech == RS_TECH_NEE && k == d - 1u) { c = select(5u, 6u, ep == LT_ENV); }
  else if (k == d - 1u) { c = select(7u, 8u, tech == RS_TECH_BSDF_ENV); }
  else if (k == d) { c = select(9u, 10u, tech == RS_TECH_BSDF_ENV); }
  else { c = select(12u, 11u, tech == RS_TECH_NEE); }
  let hi = select(k > 2u, d > 2u, k == 0u);
  return 2u * c + select(0u, 1u, hi);
}
fn st(bin: u32, w: u32) { atomicAdd(&stats[bin * SW + w], 1u); }
fn undefined_sc(sc: u32) -> bool { return sc != SC_OK && sc != SC_ZERO && sc != SC_OCCLUDED && sc != SC_NONFINITE; }
fn margin_of(code: u32) -> f32 { return unpack2x16float(code >> 16u).x; }
fn t3_viol(kind: u32, bin: u32, trial: u32, c0: u32, c1: u32, sig: u32, edge: f32, J0: f32, J1: f32, flags: u32, ai: u32, q: u32, fr: f32) {
  let i = atomicAdd(&viol[0], 1u);
  if (i >= ${T3_VIOL_CAP}u) { return; }
  let o = 4u + i * ${T3_VIOL_WORDS}u;
  atomicStore(&viol[o], kind | (bin << 8u)); atomicStore(&viol[o + 1u], trial); atomicStore(&viol[o + 2u], c0);
  atomicStore(&viol[o + 3u], c1); atomicStore(&viol[o + 4u], sig); atomicStore(&viol[o + 5u], bitcast<u32>(edge));
  atomicStore(&viol[o + 6u], bitcast<u32>(J0)); atomicStore(&viol[o + 7u], bitcast<u32>(J1)); atomicStore(&viol[o + 8u], flags);
  atomicStore(&viol[o + 9u], ai); atomicStore(&viol[o + 10u], q); atomicStore(&viol[o + 11u], bitcast<u32>(fr));
  atomicStore(&viol[o + 12u], rsDispatch.t); atomicStore(&viol[o + 13u], trial);
}

/// σ check of the replayed prefix against the base's dumped ids / lobes: 0 = equal, else (b | kind << 8), kind 1 prim,
/// 2 lobe; *edge = barycentric edge distance of the diverging hit.
fn t3_sig(base: u32, f: u32, edge: ptr<function, f32>) -> u32 {
  let d = rf_d(f); let k = rf_k(f);
  if (k != 0u && k <= 2u) { return 0u; }
  let nB = select(k - 2u, d - 1u, k == 0u);
  let hist = dump[base + 23u];
  for (var b = 1u; b <= min(nB, 7u); b++) {
    let lb = (hist >> (4u * (b - 1u))) & 0xFu;
    if (trLobe[b] != lb) { *edge = trEdge[b]; return b | (2u << 8u); }
    if (b + 1u <= d - 1u && trPrim[b] != dump[base + 40u + b]) { *edge = trEdge[b]; return b | (1u << 8u); }
  }
  return 0u;
}

@compute @workgroup_size(64)
fn t3_main(@builtin(global_invocation_id) gid: vec3u) {
  let idx = gid.y * 65535u * 64u + gid.x;
  if (idx >= rsDispatch.treeCount) { return; }
  let trial = rsDispatch.treeBase + idx;
  let NP = max(rsDispatch.flags & 0xFFFFu, 1u);
  let W = rsParams.atlasSize.x; let H = rsParams.atlasSize.y; let P = W * H;
  let ai = trial / (RS_DUMP_CAP * NP);
  let rem = trial % (RS_DUMP_CAP * NP);
  let di = rem / NP; let j = rem % NP;
  if (ai >= P || di >= dump[P * RS_DUMP_CAP * RS_DUMP_WORDS + ai]) { return; }
  let base = (ai * RS_DUMP_CAP + di) * RS_DUMP_WORDS;
  var src: ShiftSrc;
  src.flags = dump[base + 6u];
  src.empty = res_empty(src.flags);
  src.seed = vec2u(dump[base + 4u], dump[base + 5u]);
  src.F = bitcast<vec3f>(vec3u(dump[base + 1u], dump[base + 2u], dump[base + 3u]));
  src.rc = vec3u(dump[base + 8u], dump[base + 9u], dump[base + 10u]);
  src.jDen = bitcast<f32>(dump[base + 11u]);
  src.rcWi = bitcast<vec3f>(vec3u(dump[base + 12u], dump[base + 13u], dump[base + 14u]));
  src.aux = bitcast<f32>(dump[base + 15u]);
  src.rcRad = bitcast<vec3f>(vec3u(dump[base + 16u], dump[base + 17u], dump[base + 18u]));
  src.end = vec3u(dump[base + 20u], dump[base + 21u], dump[base + 22u]);
  let f = src.flags;
  let bin = t3_case(f);
  let skipBit = select((rsDispatch.passId >> (bin & 31u)) & 1u, (rsDispatch.flags >> (16u + (bin & 15u))) & 1u, bin >= 32u);
  // bins already at their target (skip mask); M6 keep mask (RsDispatch.flags bits 20–22): trials whose stored path has
  // the selected glass event still run in a closed bin, so the extra counters keep accumulating after the standard bins
  // reach their targets (a selection of which round trips run; every check is unchanged). Bit 0: ℓ_{k−1} = G_R; bit 1:
  // ℓ_k = G_R; bit 2: x_k (ℓ_k ∈ {G_R, G_T, NEE}) on a transmissive material, where side flips happen (NEE at x_k: the
  // light sample is re-evaluated from the other side).
  if (skipBit != 0u) {
    let km = (rsDispatch.flags >> 20u) & 7u;
    let lk = rf_lk(f);
    var keepX = (select(0u, 1u, rf_lkm1(f) == LOBE_GR) | select(0u, 2u, lk == LOBE_GR)) & km;
    if ((km & 4u) != 0u && rf_k(f) != 0u && (lk == LOBE_GR || lk == LOBE_GT || lk == LOBE_NEE) && src.rc.x < arrayLength(&sceneTris)) {
      let mt = sceneMaterials[tri_material(sceneTris[src.rc.x])];
      if (mt.transmission > 0.0 || (mt.flags & (MAT_GLASS_NODE | MAT_REFRACTION_NODE)) != 0u) { keepX |= 4u; }
    }
    if (keepX == 0u) { return; }
  }
#if T3_DENSE
  latIdx = atomicLoad(&dual[${DENSE_OFF - 2}u]) + ai;
  let p = vec2u(0u);
  if (j == 0u) {
    let hh = dump[base + 23u];
    let lob = (hh & 7u) | (((hh >> 4u) & 7u) << 3u) | (((hh >> 8u) & 7u) << 6u);
    stClass((f & 0x3FFFFFu) | (lob << 22u));
  }
#else
  let p = vec2u(ai % W, ai / W);
#endif
  let mode = rsDispatch.round;
  var q = p;
  var q2 = p;
  let h = pcg4d(vec4u(ai, di, j, 0x5bd1e995u ^ rsDispatch.t));
#if T3_DENSE
  if (mode == 1u) { q = vec2u(1u + j, 0u); }
#else
  if (mode == 1u || mode == 3u) {
    let r = 1.0 + 29.0 * sqrt(u32_to_unit(h.x));
    let ph = TWO_PI * u32_to_unit(h.y);
    let qi = vec2i(p) + vec2i(round(r * vec2f(cos(ph), sin(ph))));
    if (any(qi < vec2i(0)) || qi.x >= i32(W) || qi.y >= i32(H) || all(vec2u(qi) == p)) { st(bin, ${T3S.skipped}u); return; }
    q = vec2u(qi);
  }
#endif
  if (mode == 2u) {                                    // T5: fixed offset δ (rowEnd = i16 dx | i16 dy << 16), side B: −δ
    var dl = vec2i(i32(rsDispatch.rowEnd << 16u) >> 16u, i32(rsDispatch.rowEnd) >> 16u);
    if ((rsDispatch.flags & 0x80000000u) != 0u) { dl = -dl; }
    let qi = vec2i(p) + dl;
    if (any(qi < vec2i(0)) || qi.x >= i32(W) || qi.y >= i32(H)) { return; }
    q = vec2u(qi);
  }
  if (mode == 3u) {                                    // T3-5: q2 = q + another disk offset
    let r = 1.0 + 29.0 * sqrt(u32_to_unit(h.z));
    let ph = TWO_PI * u32_to_unit(h.w);
    let qi = vec2i(q) + vec2i(round(r * vec2f(cos(ph), sin(ph))));
    if (any(qi < vec2i(0)) || qi.x >= i32(W) || qi.y >= i32(H) || all(vec2u(qi) == q)) { return; }
    q2 = vec2u(qi);
    if (!shift_dst_load(q2).valid) { return; }
  }
  let dq = shift_dst_load(q);
  if (!dq.valid) { st(bin, ${T3S.skipped}u); return; }
  let dp = shift_dst_load(p);
  let dq2 = shift_dst_load(q2);
  let ep = rf_ep(f);
  st(bin, ${T3S.trials}u);
  let F0 = src.F;
  // sentinels (a forward result that was never assigned must not read as "OK, J = 0"; SC_OK == 0)
  var J0 = -1.0;
  var c0 = 0xFFFFFFFFu;
  var nThen = 0u;                                      // executions of the ps == 0 / ps != 0 branches (PLATFORM check)
  var nElse = 0u;
  var nPass = 1u;
  if (mode == 1u) { nPass = 2u; } else if (mode == 3u) { nPass = 3u; }
  var J1 = 0.0;
  var Fc = vec3f(0.0);
  var s = src;
  var dst = dq;
  var side0 = 0u;
  var xmask = 0u;
  if (rf_lkm1(f) == 2u) { xmask |= 1u; }
  if (rf_lk(f) == 2u) { xmask |= 2u; }
  for (var ps = 0u; ps < nPass; ps++) {
    for (var b = 0u; b < 8u; b++) { trPrim[b] = 0xFFFFFFFEu; trLobe[b] = 0xFFu; trEdge[b] = 1.0; }
    trCos = 1.0; trDist = 0.0; trEdgeK = 1.0;
    trSlot = 0u;
    trNP = 0u;
    if (rsDispatch.rowBase != 0u && (trial % rsDispatch.rowBase) == 0u) {
      let r = atomicAdd(&dual[0], 1u);
      if (r < ${T3_DUAL_CAP}u) { trSlot = 4u + r * DW; }
    }
    trSide = 0u; trSeg = false;
    let o = shift_hybrid(s, dst);
    let oc = o.code;                                   // captured right after the call, before any branch on ps
    if (ps == 0u) { side0 = trSide; if (trSeg) { xmask |= 8u; } }
    let oJ = o.J;
    let sc = rs_slot_code_sc(oc);
    if (mode == 2u) {                                  // T5 census values: side A J·h(F_q(ȳ)), side B h(F_p(x̄))
      let ok = sc == SC_OK;
      let sideB = (rsDispatch.flags & 0x80000000u) != 0u;
      var v1 = 0.0;
      var vh = 0.0;
      if (ok) {
        let Fq = luminance(select(o.FJ / o.J, src.F, sideB));
        let hh = Fq / (1.0 + Fq);
        v1 = select(o.J, 1.0, sideB);
        vh = v1 * hh;
      }
      let ob = 4u + 3u * idx;
      atomicStore(&dual[ob], bitcast<u32>(v1)); atomicStore(&dual[ob + 1u], bitcast<u32>(vh));
      atomicStore(&dual[ob + 2u], bin | 0x200u | select(0u, 0x100u, ok));
      return;
    }
    if (mode == 3u) {                                  // T3-5: J_{p→q}·J_{q→q2} vs J_{p→q2}, same offset path at q2
      if (ps == 0u) {
        if (sc != SC_OK) { return; }
        J0 = o.J; s.F = o.FJ / o.J; s.jDen = o.J * s.jDen; dst = dq2;
        continue;
      }
      if (ps == 1u) {
        stg(${T3_CHAIN.trials}u);
        J1 = select(0.0, o.J, sc == SC_OK); Fc = select(vec3f(0.0), o.FJ / o.J, sc == SC_OK); c0 = o.code;
        s = src; dst = dq2;
        continue;
      }
      let okA = J1 > 0.0;
      let okB = sc == SC_OK;
      if (okA != okB) {
        let mA = margin_of(c0); let mB = margin_of(o.code);
        if (!(abs(mA) < 1.52587890625e-5 || abs(mB) < 1.52587890625e-5)) {
          stg(${T3_CHAIN.defMismatch}u);
          t3_viol(11u, bin, trial, c0, o.code, 0u, 0.0, J0 * J1, o.J, f, ai, q2.y * W + q2.x, 0.0);
        }
        return;
      }
      if (!okA) { return; }
      stg(${T3_CHAIN.both}u);
      if (!(abs(log(J0 * J1 / o.J)) < 1e-4)) { stg(${T3_CHAIN.jViol}u); t3_viol(12u, bin, trial, c0, o.code, 0u, 0.0, J0 * J1, o.J, f, ai, q2.y * W + q2.x, 0.0); }
      let Fb = o.FJ / o.J;
      let dF = abs(Fc - Fb) / max(Fb, vec3f(1e-30));
      if (!(max(max(dF.x, dF.y), dF.z) < 1e-4)) { stg(${T3_CHAIN.fViol}u); t3_viol(13u, bin, trial, c0, o.code, 0u, 0.0, J0 * J1, o.J, f, ai, q2.y * W + q2.x, max(max(dF.x, dF.y), dF.z)); }
      return;
    }
    if (trSlot != 0u) {
      dw(0u, o.code); dw(1u, s.flags); dw(2u, bitcast<u32>(dst.thr)); dw(3u, dst.prim); dw(4u, bitcast<u32>(dst.bary.x)); dw(5u, bitcast<u32>(dst.bary.y));
      dw(6u, s.rc.x); dw(7u, s.rc.y); dw(8u, s.rc.z);
      dw(9u, bitcast<u32>(s.rcWi.x)); dw(10u, bitcast<u32>(s.rcWi.y)); dw(11u, bitcast<u32>(s.rcWi.z));
      dw(12u, s.end.x); dw(13u, s.end.y); dw(14u, s.end.z);
      dw(15u, min(trNP, 10u) | (ps << 8u) | (bin << 16u) | (mode << 24u));
      dw(74u, bitcast<u32>(dst.camPos.x)); dw(75u, bitcast<u32>(dst.camPos.y)); dw(76u, bitcast<u32>(dst.camPos.z));
      dw(77u, bitcast<u32>(o.J)); dw(78u, bitcast<u32>(s.jDen));
    }
    let last = ps + 1u == nPass;
    var edge = 1.0;
    if (ps == 0u) {
      nThen++;
      J0 = oJ; c0 = oc;
      atomicAdd(&stats[bin * SW + ${T3S.fwdCodes}u + sc], 1u);
      if (!undefined_sc(sc)) { st(bin, ${T3S.fwdDefined}u); }
      if (sc == SC_OK) {
        st(bin, ${T3S.fwdOk}u);
        stEp(ep, 0u);
        let lb = clamp(i32(floor(log2(o.J) * 2.0)) + 32, 0, 63);
        atomicAdd(&stats[NB * SW + u32(lb)], 1u);
      }
    } else {
      nElse++;
      atomicAdd(&stats[bin * SW + ${T3S.invCodes}u + sc], 1u);
      if (!undefined_sc(sc)) { st(bin, ${T3S.invDefined}u); }
    }
    if (last) {
      if (mode == 1u && side0 != 0u && trSide != 0u && side0 != trSide) { xmask |= 4u; }
      stX(xmask, 0u);
      // PLATFORM: the branch on the uniform-per-thread loop counter must have run exactly once per pass, ps == 0 once
      if (nThen != 1u || nThen + nElse != ps + 1u || c0 == 0xFFFFFFFFu) {
        stg(${T3_CHAIN.platform}u);
        t3_viol(${T3V.platform}u, bin, trial, c0, oc, ps | (nPass << 4u) | (mode << 8u) | (nThen << 12u) | (nElse << 16u), 0.0, J0, oJ, f, ai, q.y * W + q.x, 0.0);
        return;
      }
      let sig = t3_sig(base, f, &edge);
      let selfM = mode == 0u;
      if (undefined_sc(sc)) {
        // classify: a predicate flip with |margin| < 2⁻¹⁶, or a replay divergence at a triangle edge, is FP-BOUNDARY
        let m = margin_of(o.code);
        let isPair = sc == SC_O1 || sc == SC_O2 || sc == SC_O3;
        let fp = (isPair && abs(m) < 1.52587890625e-5) || (sig != 0u && edge < 1e-6);
        if (fp) { st(bin, ${T3S.fp}u); } else {
          st(bin, ${T3S.logic}u);
          stX(xmask, 2u);
          stEp(ep, 2u);
          t3_viol(select(${T3V.invUndefined}u, ${T3V.selfUndefined}u, selfM), bin, trial, c0, o.code, sig, edge, J0, o.J, f, ai, q.y * W + q.x, m);
        }
        return;
      }
      if (sc != SC_OK) {
        st(bin, ${T3S.visZero}u);
        stX(xmask, 2u);
        // diagnostics: J0 slot ← |cos| of the reconnection segment (min over both ends), edge ← bary edge distance of x_k,
        // fr ← segment length
        t3_viol(select(${T3V.invZero}u, ${T3V.selfZero}u, selfM), bin, trial, c0, o.code, sig, trEdgeK, trCos, o.J, f, ai, q.y * W + q.x, trDist);
        return;
      }
      st(bin, ${T3S.rtOk}u);
      stX(xmask, 1u);
      stEp(ep, 1u);
      if (sig != 0u) {
        if (edge < 1e-6) { st(bin, ${T3S.sigFp}u); } else {
          st(bin, ${T3S.sigLogic}u);
          stX(xmask, 2u);
          t3_viol(select(${T3V.sig}u, ${T3V.selfSig}u, selfM), bin, trial, c0, o.code, sig, edge, J0, o.J, f, ai, q.y * W + q.x, 0.0);
        }
      }
      let F1 = o.FJ / o.J;
      // per-channel relative difference; channels with F0 = 0 must stay 0 (no NaN is ever formed: relaxed math)
      let dF = abs(F1 - F0) / max(F0, vec3f(1e-30));
      let fr = max(max(dF.x, dF.y), dF.z);
      let frOk = fr < 1e-4;
      let lumOk = abs(luminance(F1) / luminance(F0) - 1.0) < 1e-4;
      if (!(frOk && lumOk)) {
        // diagnostic: grazing cosine of the reconnection directions (|ng·ω'| at y_{k−1}, |ng·ω_k| at x_k)
        let gz = trCos;
        // F is one function of the stored vertices (D3 / Changelog B-2): no tolerance beyond 1e-4 (gz: diagnostic)
        st(bin, ${T3S.fViol}u);
        stX(xmask, 2u);
        t3_viol(select(${T3V.F}u, ${T3V.selfF}u, selfM), bin, trial, c0, o.code, sig, gz, J0, o.J, f, ai, q.y * W + q.x, fr);
      }
      var jBad = false;
      if (selfM) { jBad = !(abs(log2(o.J)) < 1.52587890625e-5); }
      else { jBad = !(abs(log(J0 * o.J)) < 1e-4); }
      if (jBad) {
        st(bin, ${T3S.jViol}u);
        stX(xmask, 2u);
        t3_viol(select(${T3V.J}u, ${T3V.selfJ}u, selfM), bin, trial, c0, o.code, sig, edge, J0, o.J, f, ai, q.y * W + q.x, 0.0);
      }
      return;
    }
    if (sc != SC_OK) { return; }
    // write-back of the selected shifted sample (math.md#jacobian): F ← FJ/J, jDen ← J·jDen
    s.F = o.FJ / o.J;
    s.jDen = o.J * s.jDen;
    dst = dp;
  }
}
`;
/** The T3 kernel without cards (M4 fixtures). */
export const T3_WGSL = T3_WGSL_BODY();

export interface T3Stats { bins: Record<string, Record<string, number>>; fwdCodes: Record<string, number[]>; invCodes: Record<string, number[]>; hist: number[]; chain: Record<string, number>; ep: { fwdOk: number; rtOk: number; logic: number }[]; classes: { key: number; count: number }[] }
export function decodeT3Stats(w: Uint32Array): T3Stats {
  const bins: Record<string, Record<string, number>> = {};
  const fwdCodes: Record<string, number[]> = {}, invCodes: Record<string, number[]> = {};
  T3_BIN_NAMES.forEach((n, b) => {
    const o = b * T3S.words;
    bins[n] = Object.fromEntries(Object.entries(T3S).filter(([k]) => !['words', 'fwdCodes', 'invCodes'].includes(k)).map(([k, i]) => [k, w[o + i]]));
    fwdCodes[n] = Array.from(w.subarray(o + T3S.fwdCodes, o + T3S.fwdCodes + 16));
    invCodes[n] = Array.from(w.subarray(o + T3S.invCodes, o + T3S.invCodes + 16));
  });
  const h0 = T3_NBINS * T3S.words;
  const c0 = h0 + T3_HIST_BINS;
  const chain = Object.fromEntries(Object.entries(T3_CHAIN).map(([k, i]) => [k, w[c0 + i]]));
  const ep = Array.from({ length: 8 }, (_, e) => ({ fwdOk: w[c0 + 8 + e * T3_EP_WORDS], rtOk: w[c0 + 9 + e * T3_EP_WORDS], logic: w[c0 + 10 + e * T3_EP_WORDS] }));
  const cl0 = c0 + 8 + 8 * T3_EP_WORDS;
  const classes: { key: number; count: number }[] = [];
  for (let i = 0; i < T3_CLASS_SLOTS; i++) if (w[cl0 + 2 * i]) classes.push({ key: w[cl0 + 2 * i], count: w[cl0 + 2 * i + 1] });
  return { bins, fwdCodes, invCodes, hist: Array.from(w.subarray(h0, h0 + T3_HIST_BINS)), chain, ep, classes };
}
export const T3_STATS_WORDS = T3_NBINS * T3S.words + T3_HIST_BINS + 8 + 8 * T3_EP_WORDS + 2 * T3_CLASS_SLOTS + T3_X_WORDS;
export function decodeT3Extra(w: Uint32Array): Record<string, { trials: number; rtOk: number; logic: number }> {
  const x0 = T3_NBINS * T3S.words + T3_HIST_BINS + 8 + 8 * T3_EP_WORDS + 2 * T3_CLASS_SLOTS;
  return Object.fromEntries(T3_X_CATS.map((c, i) => [c, { trials: w[x0 + 3 * i], rtOk: w[x0 + 3 * i + 1], logic: w[x0 + 3 * i + 2] }]));
}
