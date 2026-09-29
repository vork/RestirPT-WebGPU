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
  defines: Defines = {}, textures: GPUTextureSampleType[] = []): Promise<TestPipeline> {
  const c = GPUShaderStage.COMPUTE;
  const entries: GPUBindGroupLayoutEntry[] = [];
  for (let i = 0; i < nBuffers; i++) entries.push({ binding: i, visibility: c, buffer: { type: 'storage' } });
  textures.forEach((t, j) => entries.push({ binding: nBuffers + j, visibility: c, texture: { sampleType: t } }));
  const layout = k.device.createBindGroupLayout({ label: `${name}-g2`, entries });
  const file = `${name}.wgsl`;
  const pipeline = await k.compile(file, entry, k.customDefines(defines, true), k.customLayout(layout, true), name, { [file]: src });
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
export const T3_CASES = ['a-delta', 'a-area', 'a-tri', 'a-sun', 'f-env', 'b', 'b-env', 'c-tri', 'c-env', 'd', 'e', 'deep-nee', 'deep-bsdf', 'none-tri', 'none-env'] as const;
export const T3_NBINS = T3_CASES.length * 2;
export const T3_BIN_NAMES = T3_CASES.flatMap((c) => [`${c}/k2`, `${c}/k>2`]);
/** Per-bin stats words. */
export const T3S = {
  trials: 0, fwdOk: 1, rtOk: 2, logic: 3, fp: 4, visZero: 5, sigLogic: 6, sigFp: 7, jViol: 8, fViol: 9, skipped: 10,
  fwdDefined: 11, invDefined: 12, fCond: 13, words: 48, fwdCodes: 16, invCodes: 32,
} as const;
export const T3_HIST_BINS = 64;
export const T3_VIOL_WORDS = 16;
export const T3_VIOL_CAP = 4096;
/** Violation kinds (viol record word 1 high byte). */
export const T3V = { invUndefined: 1, invZero: 2, sig: 3, J: 4, F: 5, selfUndefined: 6, selfJ: 7, selfF: 8, selfSig: 9, selfZero: 10 } as const;

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
 * treeCount = trials, round = mode, flags = partners per candidate, passId = bin skip mask, rowBase = dual trace stride
 * (0 = off; every rowBase-th trial records both shifts for the f64 dual).
 */
/** Dual (T3-D) trace records: word 0 of the buffer = count; record r at 4 + r·T3_DUAL_WORDS (layout: decodeDualRecord). */
export const T3_DUAL_WORDS = 80;
export const T3_DUAL_CAP = 131072;
export const T3_WGSL = `
#include "restir/shift.wgsl"
@group(2) @binding(0) var<storage, read_write> dump: array<u32>;
@group(2) @binding(1) var<storage, read_write> stats: array<atomic<u32>>;
@group(2) @binding(2) var<storage, read_write> viol: array<atomic<u32>>;

const NB: u32 = ${T3_NBINS}u;
const SW: u32 = ${T3S.words}u;
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
fn rs_trace_recon(cosY: f32, cosK: f32) { trCos = min(cosY, cosK); }

fn t3_case(f: u32) -> u32 {
  let d = rf_d(f); let k = rf_k(f); let tech = rf_tech(f); let ep = rf_ep(f);
  var c = 12u;
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
  let NP = max(rsDispatch.flags, 1u);
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
  if (((rsDispatch.passId >> bin) & 1u) != 0u) { return; }     // bins already at their target (skip mask)
  let p = vec2u(ai % W, ai / W);
  let mode = rsDispatch.round;
  var q = p;
  if (mode == 1u) {
    let h = pcg4d(vec4u(ai, di, j, 0x5bd1e995u ^ rsDispatch.t));
    let r = 1.0 + 29.0 * sqrt(u32_to_unit(h.x));
    let ph = TWO_PI * u32_to_unit(h.y);
    let qi = vec2i(p) + vec2i(round(r * vec2f(cos(ph), sin(ph))));
    if (any(qi < vec2i(0)) || qi.x >= i32(W) || qi.y >= i32(H) || all(vec2u(qi) == p)) { st(bin, ${T3S.skipped}u); return; }
    q = vec2u(qi);
  }
  let dq = shift_dst_load(q);
  if (!dq.valid) { st(bin, ${T3S.skipped}u); return; }
  let dp = shift_dst_load(p);
  st(bin, ${T3S.trials}u);
  let F0 = src.F;
  var J0 = 0.0;
  var c0 = 0u;
  let nPass = select(1u, 2u, mode == 1u);
  var s = src;
  var dst = dq;
  for (var ps = 0u; ps < nPass; ps++) {
    for (var b = 0u; b < 8u; b++) { trPrim[b] = 0xFFFFFFFEu; trLobe[b] = 0xFFu; trEdge[b] = 1.0; }
    trCos = 1.0;
    trSlot = 0u;
    trNP = 0u;
    if (rsDispatch.rowBase != 0u && (trial % rsDispatch.rowBase) == 0u) {
      let r = atomicAdd(&dual[0], 1u);
      if (r < ${T3_DUAL_CAP}u) { trSlot = 4u + r * DW; }
    }
    let o = shift_hybrid(s, dst);
    let sc = rs_slot_code_sc(o.code);
    if (trSlot != 0u) {
      dw(0u, o.code); dw(1u, s.flags); dw(2u, bitcast<u32>(dst.thr)); dw(3u, dst.prim); dw(4u, bitcast<u32>(dst.bary.x)); dw(5u, bitcast<u32>(dst.bary.y));
      dw(6u, s.rc.x); dw(7u, s.rc.y); dw(8u, s.rc.z);
      dw(9u, bitcast<u32>(s.rcWi.x)); dw(10u, bitcast<u32>(s.rcWi.y)); dw(11u, bitcast<u32>(s.rcWi.z));
      dw(12u, s.end.x); dw(13u, s.end.y); dw(14u, s.end.z);
      dw(15u, min(trNP, 10u) | (ps << 8u) | (bin << 16u) | (mode << 24u));
      dw(74u, bitcast<u32>(dst.camPos.x)); dw(75u, bitcast<u32>(dst.camPos.y)); dw(76u, bitcast<u32>(dst.camPos.z));
    }
    let last = ps + 1u == nPass;
    var edge = 1.0;
    if (ps == 0u) {
      J0 = o.J; c0 = o.code;
      atomicAdd(&stats[bin * SW + ${T3S.fwdCodes}u + sc], 1u);
      if (!undefined_sc(sc)) { st(bin, ${T3S.fwdDefined}u); }
      if (sc == SC_OK) {
        st(bin, ${T3S.fwdOk}u);
        let lb = clamp(i32(floor(log2(o.J) * 2.0)) + 32, 0, 63);
        atomicAdd(&stats[NB * SW + u32(lb)], 1u);
      }
    } else {
      atomicAdd(&stats[bin * SW + ${T3S.invCodes}u + sc], 1u);
      if (!undefined_sc(sc)) { st(bin, ${T3S.invDefined}u); }
    }
    if (last) {
      let sig = t3_sig(base, f, &edge);
      let selfM = mode == 0u;
      if (undefined_sc(sc)) {
        // classify: a predicate flip with |margin| < 2⁻¹⁶, or a replay divergence at a triangle edge, is FP-BOUNDARY
        let m = margin_of(o.code);
        let isPair = sc == SC_O1 || sc == SC_O2 || sc == SC_O3;
        let fp = (isPair && abs(m) < 1.52587890625e-5) || (sig != 0u && edge < 1e-6);
        if (fp) { st(bin, ${T3S.fp}u); } else {
          st(bin, ${T3S.logic}u);
          t3_viol(select(${T3V.invUndefined}u, ${T3V.selfUndefined}u, selfM), bin, trial, c0, o.code, sig, edge, J0, o.J, f, ai, q.y * W + q.x, m);
        }
        return;
      }
      if (sc != SC_OK) {
        st(bin, ${T3S.visZero}u);
        t3_viol(select(${T3V.invZero}u, ${T3V.selfZero}u, selfM), bin, trial, c0, o.code, sig, edge, J0, o.J, f, ai, q.y * W + q.x, 0.0);
        return;
      }
      st(bin, ${T3S.rtOk}u);
      if (sig != 0u) {
        if (edge < 1e-6) { st(bin, ${T3S.sigFp}u); } else {
          st(bin, ${T3S.sigLogic}u);
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
        // The base F uses the sampler's weight f/p at the SAMPLED direction (D3 allows it), the shift f/p at the
        // position-derived one; they differ by the ray-offset angle (~ulp·256/t) over |cos|: explained if fr·|cos| < 1e-3.
        if (fr * gz < 1e-3) { st(bin, ${T3S.fCond}u); } else {
          st(bin, ${T3S.fViol}u);
          t3_viol(select(${T3V.F}u, ${T3V.selfF}u, selfM), bin, trial, c0, o.code, sig, gz, J0, o.J, f, ai, q.y * W + q.x, fr);
        }
      }
      var jBad = false;
      if (selfM) { jBad = !(abs(log2(o.J)) < 1.52587890625e-5); }
      else { jBad = !(abs(log(J0 * o.J)) < 1e-4); }
      if (jBad) {
        st(bin, ${T3S.jViol}u);
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

export interface T3Stats { bins: Record<string, Record<string, number>>; fwdCodes: Record<string, number[]>; invCodes: Record<string, number[]>; hist: number[] }
export function decodeT3Stats(w: Uint32Array): T3Stats {
  const bins: Record<string, Record<string, number>> = {};
  const fwdCodes: Record<string, number[]> = {}, invCodes: Record<string, number[]> = {};
  T3_BIN_NAMES.forEach((n, b) => {
    const o = b * T3S.words;
    bins[n] = Object.fromEntries(Object.entries(T3S).filter(([k]) => !['words', 'fwdCodes', 'invCodes'].includes(k)).map(([k, i]) => [k, w[o + i]]));
    fwdCodes[n] = Array.from(w.subarray(o + T3S.fwdCodes, o + T3S.fwdCodes + 16));
    invCodes[n] = Array.from(w.subarray(o + T3S.invCodes, o + T3S.invCodes + 16));
  });
  return { bins, fwdCodes, invCodes, hist: Array.from(w.subarray(T3_NBINS * T3S.words, T3_NBINS * T3S.words + T3_HIST_BINS)) };
}
export const T3_STATS_WORDS = T3_NBINS * T3S.words + T3_HIST_BINS;
