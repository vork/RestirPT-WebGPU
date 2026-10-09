// Fixtures of the temporal GPU tests (restir-temporal-api.md §6.1; T-B): camera paths in the box scene, a chain driver
// (advance() + frameUnits per frame, one submit per frame, TD26), tState / reservoir readback and CPU decoders.
// OWNER T-B.
// perf2 (perf2-api.md): rigs built with restirRig / testPerfFlags() carry VITE_PERF_FLAGS, and the custom check
// pipelines below compose with k.customDefines() (the kernel's perf flags) — T3 round trips with flags forced on.
import { readBuffer } from '../../src/core/gpu/readback.ts';
import type { EnvParamsCpu } from '../../src/core/render/env-gpu.ts';
import type { CameraState } from '../../src/core/render/frame-uniforms.ts';
import type { RestirCounters, RestirKernel } from '../../src/core/render/restir/kernel.ts';
import { RES_WORDS, RW, SC_NAMES, decodeTStateLocal, type TStateRecord } from '../../src/core/render/restir/layout.ts';
import type { LightData } from '../../src/core/scene/types.ts';
import { boxCamera, type RestirRig } from './restir-fixtures.ts';

/** boxCamera translated by (dx, dy, dz) and yawed by `yaw` radians about +Y (column-major camToWorld). */
export function movedCamera(dx: number, dy: number, dz: number, yaw = 0, yfov?: number): CameraState {
  const c = boxCamera();
  const m = c.camToWorld.slice();
  const cs = Math.cos(yaw), sn = Math.sin(yaw);
  for (const col of [0, 4, 8]) {                          // rotate the basis columns about +Y
    const x = m[col], z = m[col + 2];
    m[col] = cs * x + sn * z;
    m[col + 2] = -sn * x + cs * z;
  }
  m[12] += dx; m[13] += dy; m[14] += dz;
  return { camToWorld: m, yfov: yfov ?? c.yfov };
}

export interface ChainFrame { t: number; camera: CameraState; lights: LightData[]; reset?: boolean; env?: { params: EnvParamsCpu; mapId: string } }
export interface FrameResult { t: number; histValid: boolean; flags: number; counters: RestirCounters }

/** Run frames through RestirKernel.advance + frameUnits (one submit per frame). `after(f, res)` runs after each frame's
 *  GPU work (readbacks); counters are read with reset per frame. */
export async function runChain(rig: RestirRig, frames: ChainFrame[], after?: (f: ChainFrame, r: FrameResult) => Promise<void> | void, keep = true): Promise<FrameResult[]> {
  const k = rig.kernel, device = rig.g.device;
  const clear = device.createCommandEncoder();
  clear.clearBuffer(rig.accum); clear.clearBuffer(rig.counters); clear.clearBuffer(k.resources.arena, 0, 256);
  device.queue.submit([clear.finish()]);
  const out: FrameResult[] = [];
  for (const f of frames) {
    const adv = k.advance({ t: f.t, camera: f.camera, lights: f.lights, reset: f.reset, env: f.env });
    k.beginSubmit();
    const enc = device.createCommandEncoder({ label: `chain-${f.t}` });
    for (const u of k.frameUnits(f.t, { accum: rig.accum, counters: rig.counters })) u.encode(enc);
    device.queue.submit([enc.finish()]);
    await device.queue.onSubmittedWorkDone();
    const r = { t: f.t, histValid: adv.histValid, flags: adv.flags, counters: await k.readCounters(true) };
    if (keep) out.push(r);
    await after?.(f, r);
  }
  return out;
}

/** Temporal state of the last frame: tState per pixel, the history (res[h]) and output (res[w]) reservoirs. */
export interface TemporalSnapshot {
  P: number; NS: number; h: number; w: number;
  ts: (ai: number) => TStateRecord;
  hist: Uint32Array; out: Uint32Array; histF: Float32Array; outF: Float32Array;
  rec: (buf: Uint32Array, ai: number, word: number) => number;
  recF: (buf: Float32Array, ai: number, word: number) => number;
}

export async function temporalSnapshot(k: RestirKernel): Promise<TemporalSnapshot> {
  const res = k.resources;
  const P = res.pixels, NS = res.alloc.slots;
  const words = await k.readTemporalState();
  const h = k.historyIndex(), w = k.resBase();
  const hist = h >= 0 ? await k.readReservoirs(h as 0 | 1) : new Uint32Array(P * RES_WORDS);
  const out = await k.readReservoirs(w as 0 | 1);
  return {
    P, NS, h, w, hist, out,
    histF: new Float32Array(hist.buffer), outF: new Float32Array(out.buffer),
    ts: (ai) => decodeTStateLocal(words, P, NS, ai),
    rec: (buf, ai, word) => buf[ai * RES_WORDS + word],
    recF: (buf, ai, word) => buf[ai * RES_WORDS + word],
  };
}

export const lum = (r: number, g: number, b: number) => 0.2126 * r + 0.7152 * g + 0.0722 * b;
/** lum F of record ai of a reservoir buffer (as f32 words). */
export const recLumF = (f: Float32Array, ai: number) => lum(f[ai * RES_WORDS + RW.F], f[ai * RES_WORDS + RW.F + 1], f[ai * RES_WORDS + RW.F + 2]);

/** Read an arbitrary GPU buffer range as u32. */
export async function readU32(device: GPUDevice, b: GPUBuffer, bytes = b.size): Promise<Uint32Array> {
  return new Uint32Array(await readBuffer(device, b, bytes));
}

/** Lights of frame t for the light-change tests: light `id` translated by t·dp and its power scaled by `power(t)`. */
export function animatedLights(base: LightData[], t: number, o: { id: number; dp?: [number, number, number]; power?: (t: number) => number }[]): LightData[] {
  return base.map((l) => {
    const a = o.find((x) => x.id === l.id);
    if (!a) return l;
    const m = Array.from(l.matrix as ArrayLike<number>);
    if (a.dp) { m[12] += t * a.dp[0]; m[13] += t * a.dp[1]; m[14] += t * a.dp[2]; }
    return { ...l, matrix: Float32Array.from(m), power: l.power * (a.power ? a.power(t) : 1) };
  });
}

/** FNV-1a over u32 words (hex). */
export function hashU32(u: Uint32Array): string {
  let h = 0x811c9dc5;
  for (let i = 0; i < u.length; i++) {
    let x = u[i];
    for (let k = 0; k < 4; k++) { h ^= x & 0xff; h = Math.imul(h, 0x01000193) >>> 0; x >>>= 8; }
  }
  return h.toString(16).padStart(8, '0');
}

// ------------------------------------------------------------------------------------------------ T3-2 GPU harness

/** Case letters of the M4 T3 bins (restir-shift.gpu.test.ts t3_case) × {k = 2, k > 2} (∅: d = 2, d > 2). */
export const T32_CASES = ['a-delta', 'a-area', 'a-tri', 'a-sun', 'f-env', 'b', 'b-env', 'c-tri', 'c-env', 'd', 'e', 'deep-nee', 'deep-bsdf', '∅-tri', '∅-env',
  '∅-ana', 'd-ana', 'c-ana'] as const;   // M6 Mode-B crossings (restir-m6-api.md MD8; deep crossings count in deep-bsdf)
export const T32_BINS = T32_CASES.length * 2;
/** The M6 crossing cases: candidates with technique BSDF_ANALYTIC exist only in the RS_MODE_B variant (light mode B / A′,
 *  pathtree.wgsl pt_cross_candidates), so in light mode A these bins are structurally empty (Changelog M6-13). */
export const T32_MODEB_CASES: readonly string[] = ['∅-ana', 'd-ana', 'c-ana'];
export const t32ModeBBin = (binName: string): boolean => T32_MODEB_CASES.includes(binName.split('/')[0]);
const BW = 40;                       // words per bin: 0 trials 1 fwdOk 2 rtOk 3 cand 4 fp(margin) 5 fBad 6 jBad 7 noInv 8..23 fwd SC 24..39 inv SC
const CAND0 = 2048, CAND_CAP = 4096, CW = 13;   // candidates: [ai, frame, bin, invCode, fwdJ, invJ, bits(eF), bits(eJ), minEdge bits, recheckCode, done, kind, X_p flags]
export const T32_STATS_WORDS = CAND0 + 1 + CAND_CAP * CW;

const T32_CHECK_WGSL = `
#include "restir/tframe.wgsl"
@group(2) @binding(1) var<storage, read> resH: array<vec4u>;
@group(2) @binding(2) var<storage, read> resW: array<vec4u>;
@group(2) @binding(3) var<storage, read_write> st: array<atomic<u32>>;
fn t32_case(f: u32) -> u32 {
  let d = f & 0xFu; let k = (f >> 4u) & 0xFu; let tech = (f >> 8u) & 3u; let ep = (f >> 10u) & 7u;
  var c = 12u;
  if (tech == RS_TECH_BSDF_ANALYTIC) {
    if (k == 0u) { c = 15u; } else if (k == d) { c = 16u; } else if (k + 1u == d) { c = 17u; }
    return 2u * c + select(0u, 1u, select(k > 2u, d > 2u, k == 0u));
  }
  if (k == 0u) { c = select(13u, 14u, tech == RS_TECH_BSDF_ENV); }
  else if (tech == RS_TECH_NEE && k == d) {
    if (ep == 1u || ep == 2u) { c = 0u; } else if (ep == 3u || ep == 4u) { c = 1u; } else if (ep == 0u) { c = 2u; } else if (ep == 5u) { c = 3u; } else { c = 4u; }
  } else if (tech == RS_TECH_NEE && k + 1u == d) { c = select(5u, 6u, ep == 7u); }
  else if (k + 1u == d) { c = select(7u, 8u, tech == RS_TECH_BSDF_ENV); }
  else if (k == d) { c = select(9u, 10u, tech == RS_TECH_BSDF_ENV); }
  else { c = select(12u, 11u, tech == RS_TECH_NEE); }
  let hi = select(k > 2u, d > 2u, k == 0u);
  return 2u * c + select(0u, 1u, hi);
}
fn bump(bin: u32, w: u32) { atomicAdd(&st[bin * ${BW}u + w], 1u); }
fn relE(a: f32, b: f32) -> f32 { if (a == 0.0 && b == 0.0) { return 0.0; } return abs(a - b) / max(max(abs(a), abs(b)), 1e-30); }
fn cand(ai: u32, bin: u32, kind: u32, invCode: u32, fj: u32, ij: u32, eF: f32, eJ: f32, xf: u32) {
  let i = atomicAdd(&st[${CAND0}u], 1u);
  if (i >= ${CAND_CAP}u) { return; }
  let o = ${CAND0 + 1}u + i * ${CW}u;
  atomicStore(&st[o], ai); atomicStore(&st[o + 1u], rsDispatch.t); atomicStore(&st[o + 2u], bin); atomicStore(&st[o + 3u], invCode);
  atomicStore(&st[o + 4u], fj); atomicStore(&st[o + 5u], ij); atomicStore(&st[o + 6u], bitcast<u32>(eF)); atomicStore(&st[o + 7u], bitcast<u32>(eJ));
  atomicStore(&st[o + 8u], 0x7f7fffffu); atomicStore(&st[o + 9u], 0xFFFFFFFFu); atomicStore(&st[o + 10u], 0u); atomicStore(&st[o + 11u], kind); atomicStore(&st[o + 12u], xf);
}
@compute @workgroup_size(64) fn check(@builtin(global_invocation_id) gid: vec3u) {
  let ai = gid.y * 65535u * 64u + gid.x;
  if (ai >= rs_atlas_pixels()) { return; }
  let flags = ts_load(ai, TSW_FLAGS);
  if ((flags & TS_QVALID) == 0u) { return; }
  let qP = ts_load(ai, TSW_QPRIME);
  let f = resH[qP * 10u + 1u].z;
  if ((f & 0xFu) == 0u) { return; }
  let bin = t32_case(f);
  bump(bin, 0u);
  let fc = ts_load(ai, TSW_FWDCODE);
  bump(bin, 8u + (fc & 15u));
  if ((fc & 0xFFu) != SC_OK) { return; }
  bump(bin, 1u);
  let fj = ts_load(ai, TSW_FWDJ);
  if ((flags & TS_INV_DONE) == 0u) { bump(bin, 7u); cand(ai, bin, 1u, 0u, fj, 0u, 0.0, 0.0, f); return; }
  let ic = ts_load(ai, TSW_INVCODE);
  bump(bin, 24u + (ic & 15u));
  let ij = ts_load(ai, TSW_INVJ);
  if ((ic & 0xFFu) != SC_OK) {
    let m = unpack2x16float(ic >> 16u).x;
    let sc = ic & 0xFFu;
    if (abs(m) < 1.52587890625e-5 && sc != SC_ZERO && sc != SC_OCCLUDED) { bump(bin, 4u); return; }
    bump(bin, 3u); cand(ai, bin, 2u, ic, fj, ij, 0.0, 0.0, f); return;
  }
  let Fx = bitcast<vec3f>(resH[qP * 10u].yzw);
  let Fi = vec3f(ts_loadf(ai, TSW_INVF), ts_loadf(ai, TSW_INVF + 1u), ts_loadf(ai, TSW_INVF + 2u));
  let eF = max(relE(Fi.x, Fx.x), max(relE(Fi.y, Fx.y), relE(Fi.z, Fx.z)));
  let eJ = abs(log(bitcast<f32>(fj) * bitcast<f32>(ij)));
  if (eF > 1e-4) { bump(bin, 5u); }
  if (eJ > 1e-4) { bump(bin, 6u); }
  if (eF > 1e-4 || eJ > 1e-4) { bump(bin, 3u); cand(ai, bin, 3u, ic, fj, ij, eF, eJ, f); return; }
  bump(bin, 2u);
}`;

/** The inverse of every new candidate re-run with the replay trace (M4 B-4 replay-edge rule): minimum barycentric edge
 *  distance over the replayed hits (the base path's vertex ids are not in the reservoir, so the diverging hit cannot be
 *  singled out as M4's dump-based σ check does: the minimum over all replayed hits is used, which is permissive only for
 *  paths that pass within 1e-6 of an edge somewhere). The re-run code must reproduce T4's code word (else PLATFORM). */
const T32_RECHECK_WGSL = `
var<private> trEdge: f32;
fn rs_trace_vertex(b: u32, prim: u32, u: f32, v: f32, lobe: u32) { if (prim != 0xFFFFFFFFu) { trEdge = min(trEdge, min(min(u, v), 1.0 - u - v)); } }
fn rs_trace_pair(j: u32, ok: bool, margin: f32, term: u32) { }
fn rs_trace_light(dir: vec3f, pos: vec3f, inf: bool) { }
fn rs_trace_escape(dir: vec3f) { }
fn rs_trace_recon(cosY: f32, cosK: f32, dist: f32, edgeK: f32) { }
fn rs_trace_cross(n: vec3f, z: vec3f) { }
#include "restir/tshift.wgsl"
@group(2) @binding(6) var<storage, read_write> st: array<atomic<u32>>;
@compute @workgroup_size(64) fn recheck(@builtin(global_invocation_id) gid: vec3u) {
  let i = gid.x;
  if (i >= min(atomicLoad(&st[${CAND0}u]), ${CAND_CAP}u)) { return; }
  let o = ${CAND0 + 1}u + i * ${CW}u;
  if (atomicLoad(&st[o + 10u]) != 0u || atomicLoad(&st[o + 1u]) != rsDispatch.t) { return; }
  let q = atomicLoad(&st[o]);
  trEdge = 1e30;
  var code = 0xFFFFFFFFu;
  let o2 = temporal_shift(tsrc_load(q, 0u, SFX_INV, RS_FS_PREV), tdst_prev(ts_load(q, TSW_QPRIME)));
  code = o2.code;
  atomicStore(&st[o + 8u], bitcast<u32>(trEdge));
  atomicStore(&st[o + 9u], code);
  atomicStore(&st[o + 10u], 1u);
}`;

export interface T32Bin { name: string; trials: number; fwdOk: number; rtOk: number; logic: number; fp: number; fBad: number; jBad: number; noInv: number; fwd: Record<string, number>; inv: Record<string, number> }
export interface T32Result { bins: T32Bin[]; total: { trials: number; fwdOk: number; rtOk: number; logic: number; fp: number; edgeFp: number }; logicSamples: string[]; platform: number; candOverflow: number }

/** GPU round-trip checks of test frames (after a frame with forced s = p in robust mode): scene-free check kernel over
 *  every pixel (bins, codes, F / J reciprocity, margin FP) + a traced re-run of the candidate inverses (edge FP). */
export class T32Harness {
  private constructor(readonly k: RestirKernel, readonly stats: GPUBuffer, private readonly check: GPUComputePipeline, private readonly recheck: GPUComputePipeline,
    private readonly gl: { a: GPUBindGroupLayout; b: GPUBindGroupLayout }) {}

  static async create(k: RestirKernel, extraSources: Record<string, string> = {}): Promise<T32Harness> {
    const d = k.device;
    const C = GPUShaderStage.COMPUTE;
    const buf = (t: GPUBufferBindingType): GPUBindGroupLayoutEntry['buffer'] => ({ type: t });
    const a = d.createBindGroupLayout({ entries: [buf('read-only-storage'), buf('read-only-storage'), buf('read-only-storage'), buf('storage')].map((b, binding) => ({ binding, visibility: C, buffer: b })) });
    const tex = (s: GPUTextureSampleType): GPUBindGroupLayoutEntry => ({ binding: 0, visibility: C, texture: { sampleType: s } });
    const b = d.createBindGroupLayout({ entries: [
      { binding: 0, visibility: C, buffer: buf('read-only-storage') }, { binding: 1, visibility: C, buffer: buf('read-only-storage') },
      { ...tex('uint'), binding: 2 }, { ...tex('unfilterable-float'), binding: 3 }, { ...tex('uint'), binding: 4 }, { ...tex('unfilterable-float'), binding: 5 },
      { binding: 6, visibility: C, buffer: buf('storage') },
    ] });
    const src = { ...extraSources, 't32-check.wgsl': T32_CHECK_WGSL, 't32-recheck.wgsl': T32_RECHECK_WGSL };
    const check = await k.compile('t32-check.wgsl', 'check', k.customDefines({ RS_ARENA_BINDING: '0u', RS_TEMPORAL: 1 }, false), k.customLayout(a, false), 't32-check', src);
    const recheck = await k.compile('t32-recheck.wgsl', 'recheck', k.customDefines({
      RS_TEMPORAL: 1, RS_RES_IN_BINDING: '0u', RS_ARENA_BINDING: '1u', RS_VBUF_BINDING: '2u', RS_GEO_BINDING: '3u', RS_VBUF_PREV_BINDING: '4u', RS_GEO_PREV_BINDING: '5u',
      RS_REPLAY: 1, RS_SHIFT_TRACE: 1,
    }, true), k.customLayout(b, true), 't32-recheck', src);
    const stats = d.createBuffer({ label: 't32-stats', size: 4 * T32_STATS_WORDS, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
    const enc = d.createCommandEncoder(); enc.clearBuffer(stats); d.queue.submit([enc.finish()]);
    return new T32Harness(k, stats, check, recheck, { a, b });
  }

  /** Check the frame just rendered (t = its RsDispatch t; h/w/parity of that frame). */
  frame(t: number): void {
    const k = this.k, d = k.device, res = k.resources;
    const h = k.historyIndex(), w = k.resBase();
    if (h < 0) return;
    const ga = d.createBindGroup({ layout: this.gl.a, entries: [res.arena, res.res[h], res.res[w], this.stats].map((b, binding) => ({ binding, resource: { buffer: b } })) });
    const v = res.views;
    const gb = d.createBindGroup({ layout: this.gl.b, entries: [
      { binding: 0, resource: { buffer: res.res[w] } }, { binding: 1, resource: { buffer: res.arena } },
      { binding: 2, resource: v.vbuf }, { binding: 3, resource: v.geo }, { binding: 4, resource: v.vbufPrev }, { binding: 5, resource: v.geoPrev },
      { binding: 6, resource: { buffer: this.stats } },
    ] });
    k.beginSubmit();
    const enc = d.createCommandEncoder({ label: `t32-${t}` });
    const g = Math.ceil(res.pixels / 64);
    k.encodeCustom(enc, this.check, ga, { t }, [Math.min(g, 65535), Math.ceil(g / 65535)], false);
    k.encodeCustom(enc, this.recheck, gb, { t }, [CAND_CAP / 64, 1], true);
    d.queue.submit([enc.finish()]);
  }

  async result(): Promise<T32Result> {
    const u = new Uint32Array(await readBuffer(this.k.device, this.stats, 4 * T32_STATS_WORDS));
    const f = new Float32Array(u.buffer);
    const bins: T32Bin[] = [];
    const nCand = Math.min(u[CAND0], CAND_CAP);
    const edgeFpPerBin = new Array(T32_BINS).fill(0);
    const logicSamples: string[] = [];
    let platform = 0;
    for (let i = 0; i < nCand; i++) {
      const o = CAND0 + 1 + i * CW;
      const [ai, t, bin, invCode, fj, ij] = u.subarray(o, o + 6);
      const edge = f[o + 8], recode = u[o + 9], kind = u[o + 11];
      if (u[o + 10] !== 1 || (kind !== 1 && recode !== invCode)) platform++;          // re-run missing / not reproducing T4
      if (edge < 1e-6) { edgeFpPerBin[bin]++; continue; }
      if (logicSamples.length < 24) logicSamples.push(`t${t} ai${ai} ${T32_CASES[bin >> 1]}/${bin & 1 ? 'k>2' : 'k2'} kind ${kind} inv ${SC_NAMES[invCode & 15]} fwdJ ${u2f(fj)} invJ ${u2f(ij)} eF ${f[o + 6]} eJ ${f[o + 7]} minEdge ${edge}`);
    }
    const tot = { trials: 0, fwdOk: 0, rtOk: 0, logic: 0, fp: 0, edgeFp: 0 };
    for (let b = 0; b < T32_BINS; b++) {
      const o = b * BW;
      const codes = (base: number) => Object.fromEntries(SC_NAMES.map((n, i) => [n, u[o + base + i]]).filter(([, x]) => x));
      const bb: T32Bin = {
        name: `${T32_CASES[b >> 1]}/${b & 1 ? 'k>2' : 'k2'}`, trials: u[o], fwdOk: u[o + 1], rtOk: u[o + 2],
        logic: u[o + 3] - edgeFpPerBin[b], fp: u[o + 4] + edgeFpPerBin[b], fBad: u[o + 5], jBad: u[o + 6], noInv: u[o + 7], fwd: codes(8), inv: codes(24),
      };
      bins.push(bb);
      tot.trials += bb.trials; tot.fwdOk += bb.fwdOk; tot.rtOk += bb.rtOk; tot.logic += bb.logic; tot.fp += bb.fp; tot.edgeFp += edgeFpPerBin[b];
    }
    return { bins, total: tot, logicSamples, platform, candOverflow: Math.max(0, u[CAND0] - CAND_CAP) };
  }

  destroy(): void { this.stats.destroy(); }
}
const u2f = (x: number) => new Float32Array(new Uint32Array([x]).buffer)[0];
