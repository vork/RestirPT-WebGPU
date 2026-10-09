// ReSTIR PT paired spatial reuse, queues, MIS and ensemble mode (WP-C, restir-api.md §6.1): T3-3/M4, T6(a), T6(b),
// T7, T17, U-MIS-1, U-ENS-1, U-ENS-2, U-OFF-1 and the replay-predicate agreement (Changelog C3); M5 (T-D): T3-3-boost
// (restir-temporal-api.md TD21, §6.1). Chrome lane authoritative; dawn.node is a pre-check.
import { afterAll, describe, expect, it } from 'vitest';
import { readBuffer } from '../../src/core/gpu/readback.ts';
import { EnsembleCollector, decodeEnsStats, ensStatsLayout } from '../../src/core/render/restir/ensemble.ts';
import { RS_WGSL_CONSTS as K, RES_WORDS, RW, TS_CONSTS, TS_WORDS, TSW, arenaWords, rfPack, tStateAosToSoa } from '../../src/core/render/restir/layout.ts';
import { readNpz, writeNpz } from '../../src/core/render/restir/npz.ts';
import { gaussLayer, pairLayer, pairPartner, pairTransform } from '../../src/core/render/restir/pairing.ts';
import { GAUSS_PAIR_SIZES, PAIR_TEX_SIZES } from '../../src/core/render/restir/presets.ts';
import { ensStatsFloats } from '../../src/core/render/restir/resources.ts';
import { queueArgs } from '../../src/core/render/restir/stage-spatial.ts';
import { misWeightsAt } from '../../tests/restir/mis-ref.ts';
import { releaseTestGpu } from './device-factory.ts';
import { bitFixtureScene, readTexture4, restirRig, storageBuffer, type RestirRig } from './restir-fixtures.ts';

afterAll(releaseTestGpu);

const lum = (r: number, g: number, b: number) => 0.2126 * r + 0.7152 * g + 0.0722 * b;
const C = GPUShaderStage.COMPUTE;
const SB = (binding: number): GPUBindGroupLayoutEntry => ({ binding, visibility: C, buffer: { type: 'storage' } });
const TEXU = (binding: number): GPUBindGroupLayoutEntry => ({ binding, visibility: C, texture: { sampleType: 'uint' } });
const TEXF = (binding: number): GPUBindGroupLayoutEntry => ({ binding, visibility: C, texture: { sampleType: 'unfilterable-float' } });
const TEXP = (binding: number): GPUBindGroupLayoutEntry => ({ binding, visibility: C, texture: { sampleType: 'sint', viewDimension: '2d-array' } });
const u32View = (f: Float32Array) => new Uint32Array(f.buffer, f.byteOffset, f.length);
/** Bitwise equality of two word arrays. */
function sameBits(a: Uint32Array, b: Uint32Array): boolean {
  if (a.length !== b.length) return false;
  for (let i = 0; i < a.length; i++) if (a[i] !== b[i]) return false;
  return true;
}

/** Arena words (after the 64-word header) of the rig's current allocation. */
async function arenaBody(rig: RestirRig): Promise<Uint32Array> {
  const res = rig.kernel.resources;
  return new Uint32Array(await readBuffer(rig.g.device, res.arena, res.arena.size)).subarray(64);
}

/** Compile a test pipeline (no debug group) from an in-memory entry file. */
async function testPipeline(rig: RestirRig, src: string, entry: string, g2: GPUBindGroupLayoutEntry[], defines: Record<string, string | number | boolean>, scene = false) {
  const k = rig.kernel;
  const layout = rig.g.device.createBindGroupLayout({ entries: g2 });
  const pipeline = await k.compile('wpc-test.wgsl', entry, k.customDefines(defines, scene), k.customLayout(layout, scene), `wpc-${entry}`, { 'wpc-test.wgsl': src });
  return { pipeline, layout };
}

function submit(rig: RestirRig, f: (enc: GPUCommandEncoder) => void): Promise<undefined> {
  const d = rig.g.device;
  rig.kernel.beginSubmit();
  const enc = d.createCommandEncoder();
  f(enc);
  d.queue.submit([enc.finish()]);
  return d.queue.onSubmittedWorkDone();
}

// ------------------------------------------------------------------------------------------------ end to end

describe('paired spatial reuse end to end', () => {
  for (const [preset, scene] of [['offline', 'x_quads'], ['offline', 'c0s'], ['criteria2022', 'x_quads'], ['interactive', 'x_quads']] as const) {
  it(`${preset} on ${scene}: finite, no PENDING left, no slot mismatch, no overflow; U-MIS-1 lum(rsShade) = Σw per pixel`, async () => {
    const rig = await restirRig(bitFixtureScene(scene), 64, 48, { preset, settings: { maxBounces: 3, trees: preset === 'interactive' ? 1 : 4 } });
    const r = await rig.frames(2);
    const a = r.arena;
    console.log(`[spatial e2e ${preset}/${scene}] rsc ${JSON.stringify(a.rsc)} codes ${a.codes} f_r ${a.fr.toFixed(3)} q0 ${JSON.stringify(a.queues[0])}`);
    expect(r.counters).toEqual([0, 0, 0, 0]);
    expect(a.rsc.pendingLeft).toBe(0);
    expect(a.rsc.slotMismatch).toBe(0);
    expect(a.rsc.shiftNonFinite).toBe(0);
    expect(a.rsc.wNonFinite).toBe(0);
    expect(a.queues[0].overflow).toBe(0);
    expect(a.rsc.accepted).toBeGreaterThan(0);
    expect(a.codes[K.SC_PENDING]).toBe(0);
    // U-MIS-1: Σw = W_out · lum(F_out) (wScale = 1) vs lum(rsShade) of the last round
    const resv = await rig.kernel.readReservoirs('final');
    const f = new Float32Array(resv.buffer);
    const shade = new Float32Array((await readTexture4(rig.g.device, rig.kernel.resources.shade)).buffer);
    let worst = 0, n = 0;
    for (let i = 0; i < 64 * 48; i++) {
      const o = i * RES_WORDS;
      const sw = f[o + RW.W] * lum(f[o + RW.F], f[o + RW.F + 1], f[o + RW.F + 2]);
      const ls = lum(shade[4 * i], shade[4 * i + 1], shade[4 * i + 2]);
      if (sw === 0 && ls === 0) continue;
      n++;
      worst = Math.max(worst, Math.abs(ls - sw) / Math.max(sw, 1e-30));
    }
    console.log(`[U-MIS-1 ${preset}/${scene}] ${n} pixels, max |lum(L) − Σw| / Σw = ${worst.toExponential(3)}`);
    expect(n).toBeGreaterThan(1000);
    expect(worst).toBeLessThan(1e-6);
    rig.destroy();
  });
  }
});

// ------------------------------------------------------------------------------------------------ U-OFF-1

describe('U-OFF-1: offline mode determinism and split dispatch', () => {
  it('two runs bitwise identical; tree chunks {8×4} × row bands of 16 give the same reservoirs and image', async () => {
    const scene = bitFixtureScene('x_quads');
    const W = 64, H = 48;
    const run = async (rowBand: number, treeChunk: number) => {
      const rig = await restirRig(scene, W, H, { preset: 'offline', settings: { maxBounces: 3 } });
      rig.kernel.rowBand = rowBand;
      rig.kernel.treeChunk = treeChunk;
      const r = await rig.frames(2);
      const resv = await rig.kernel.readReservoirs('final');
      rig.destroy();
      return { r, resv };
    };
    const a = await run(0, 0), b = await run(0, 0), c = await run(16, 8);
    console.log(`[U-OFF-1] f_r = ${a.r.arena.fr.toFixed(4)} (queued ${a.r.arena.rsc.queued} / accepted ${a.r.arena.rsc.accepted}), selected shifted ${a.r.arena.rsc.selectedShifted}`);
    for (const x of [a, b, c]) {
      expect(x.r.arena.rsc.pendingLeft).toBe(0);
      expect(x.r.arena.rsc.slotMismatch).toBe(0);
      expect(x.r.counters).toEqual([0, 0, 0, 0]);
    }
    expect(sameBits(b.resv, a.resv)).toBe(true);
    expect(sameBits(u32View(b.r.mean), u32View(a.r.mean))).toBe(true);
    expect(sameBits(c.resv, a.resv)).toBe(true);
    expect(sameBits(u32View(c.r.mean), u32View(a.r.mean))).toBe(true);
  });
});

// ------------------------------------------------------------------------------------------------ T3-3/M4

const T33_SRC = `
#include "restir/frame.wgsl"
#include "restir/pairing.wgsl"
@group(2) @binding(3) var<storage, read_write> t33: array<atomic<u32>>;
// t33: 0 codes-seen mask, 1 paired, 2 non-reciprocal, 3 A(p,q) != A(q,p), 4 accepted, 5 |d| > R, 6 cross-member,
//      16… partner of (t = rsDispatch.t, round 0, slot 0) per pixel (0xFFFFFFFF = none)
@compute @workgroup_size(8, 8, 1)
fn t33_main(@builtin(global_invocation_id) gid: vec3u) {
  let p = rs_pix(gid.xy);
  if (!p.valid) { return; }
  let R = i32(rsDispatch.treeBase);
  var codes = 0u; var paired = 0u; var nonrec = 0u; var asym = 0u; var acc = 0u; var far = 0u;
  for (var ti = 0u; ti < rsDispatch.treeCount; ti++) {
    let t = rsDispatch.t + ti;
    for (var r = 0u; r < 2u; r++) {
      for (var s = 0u; s < rsParams.numSlots; s++) {
        codes |= 1u << pair_transform(p.member, t, r, s).code;
        let pr = pair_partner(p.local, p.member, t, r, s);
        if (ti == 0u && r == 0u && s == 0u) {
          atomicStore(&t33[16u + p.ai], select(0xFFFFFFFFu, pr.partner.y * rsParams.memberSize.x + pr.partner.x, pr.valid));
        }
        if (!pr.valid) { continue; }
        paired++;
        let back = pair_partner(pr.partner, p.member, t, r, s);
        if (!back.valid || back.partner.x != p.local.x || back.partner.y != p.local.y) { nonrec++; }
        let d = vec2i(pr.partner) - vec2i(p.local);
        if (d.x * d.x + d.y * d.y > R * R) { far++; }
        let q = pair_atlas_px(p, pr.partner);
        let apq = pair_A0(rs_vbuf(p.px), rs_geo(p.px), rs_vbuf(q), rs_geo(q));
        let aqp = pair_A0(rs_vbuf(q), rs_geo(q), rs_vbuf(p.px), rs_geo(p.px));
        if (apq != aqp) { asym++; }
        if (apq) { acc++; }
      }
    }
  }
  atomicOr(&t33[0], codes);
  atomicAdd(&t33[1], paired); atomicAdd(&t33[2], nonrec); atomicAdd(&t33[3], asym); atomicAdd(&t33[4], acc); atomicAdd(&t33[5], far);
}
`;

/** T3-3 on one resolution with the M4 disk maps (T3-3/M4) or the M6 σ = 16 Gaussian maps (T3-3/M6, restir-m6-api.md §4). */
async function t33Case(W: number, H: number, pairing: 'disk' | 'gauss'): Promise<void> {
      const rig = await restirRig(bitFixtureScene('x_quads'), W, H, { preset: 'offline', settings: { maxBounces: 1, trees: 1, rounds: 1, slots: 6, pairing } });
      const k = rig.kernel, res = k.resources, dev = rig.g.device;
      const fr = await rig.frames(1);
      expect(fr.arena.rsc.slotMismatch).toBe(0);
      expect(fr.arena.rsc.pendingLeft).toBe(0);
      // (1) production slots of round 0 of frame 0 vs the TS mirror of the transform and maps
      const body = await arenaBody(rig);
      const gauss = pairing === 'gauss';
      const R = gauss ? 127 : k.settings.diskRadius, NS = 6;   // Gaussian maps: |d| ≤ 127 per axis (T14)
      const SIZES = gauss ? GAUSS_PAIR_SIZES : PAIR_TEX_SIZES;
      const layers = SIZES.slice(0, NS).map((w, s) => (gauss ? gaussLayer(w, 16, s) : pairLayer(w, R, s)));
      let bad = 0, accepted = 0, noPartnerAccepted = 0;
      for (let s = 0; s < NS; s++) {
        const tr = pairTransform(11, 0, 0, 0, s, SIZES[s]);
        for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
          const ai = y * W + x;
          const jw = body[4 * (ai * NS + s) + 3];
          const q = pairPartner(layers[s], tr, [x, y], W, H);
          if (!q) { if (jw !== K.JW_NOT_ACCEPTED) noPartnerAccepted++; continue; }
          const jq = body[4 * ((q[1] * W + q[0]) * NS + s) + 3];
          if ((jw === K.JW_NOT_ACCEPTED) !== (jq === K.JW_NOT_ACCEPTED)) bad++;
          if (jw !== K.JW_NOT_ACCEPTED) accepted++;
        }
      }
      // (2) custom kernel: involution, A symmetry, 8 codes, partner = TS mirror
      const out = storageBuffer(dev, (16 + W * H) * 4);
      const { pipeline, layout } = await testPipeline(rig, T33_SRC, 't33_main', [TEXU(0), TEXF(1), TEXP(2), SB(3)],
        { RS_VBUF_BINDING: '0u', RS_GEO_BINDING: '1u', RS_PAIRTEX_BINDING: '2u' });
      const g2 = dev.createBindGroup({ layout, entries: [
        { binding: 0, resource: res.views.vbuf }, { binding: 1, resource: res.views.geo }, { binding: 2, resource: res.views.pair }, { binding: 3, resource: { buffer: out } }] });
      const NT = 16;
      await submit(rig, (enc) => k.encodeCustom(enc, pipeline, g2, { t: 0, treeCount: NT, treeBase: R }, [Math.ceil(W / 8), Math.ceil(H / 8)], false));
      const o = new Uint32Array(await readBuffer(dev, out, (16 + W * H) * 4));
      const tr0 = pairTransform(11, 0, 0, 0, 0, SIZES[0]);
      let mirrorBad = 0;
      for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
        const q = pairPartner(layers[0], tr0, [x, y], W, H);
        if (o[16 + y * W + x] !== (q ? q[1] * W + q[0] : 0xFFFFFFFF)) mirrorBad++;
      }
      console.log(`[T3-3/${gauss ? 'M6 gauss' : 'M4'} ${W}×${H}] slots: accepted ${accepted}, mismatched ${bad}, accepted-without-partner ${noPartnerAccepted}; kernel: codes 0x${o[0].toString(16)} paired ${o[1]} non-reciprocal ${o[2]} A-asymmetric ${o[3]} A-accepted ${o[4]} |d|>R ${o[5]}; mirror mismatches ${mirrorBad}`);
      expect(bad).toBe(0);
      expect(noPartnerAccepted).toBe(0);
      expect(accepted).toBeGreaterThan(W * H);
      expect(o[0]).toBe(0xFF);
      expect(o[1]).toBeGreaterThan(W * H * NT * 2 * NS * 0.9);
      expect(o[2]).toBe(0);
      expect(o[3]).toBe(0);
      expect(o[5]).toBe(0);
      expect(mirrorBad).toBe(0);
      out.destroy();
      rig.destroy();
}

describe('T3-3/M4: paired-acceptance symmetry', () => {
  for (const [W, H] of [[960, 540], [1024, 1024]] as const) {
    it(`${W}×${H}: partner(partner(p)) = p, A(p,q) = A(q,p) bitwise over 8 dihedral codes; slot acceptedness equal on both sides`, () => t33Case(W, H, 'disk'));
  }
});

describe('T3-3/M6: paired-acceptance symmetry with the σ = 16 Gaussian maps', () => {
  for (const [W, H] of [[960, 540], [1024, 1024]] as const) {
    it(`${W}×${H} gauss: partner(partner(p)) = p, A(p,q) = A(q,p) bitwise, acceptedness equal on both sides, RSC_SLOT_MISMATCH = 0`, () => t33Case(W, H, 'gauss'));
  }
});

// ------------------------------------------------------------------------------------------------ T17

const T17_SRC = `
#include "restir/frame.wgsl"
#include "restir/queue.wgsl"
@group(2) @binding(1) var<storage, read_write> visit: array<atomic<u32>>;
@compute @workgroup_size(64)
fn produce(@builtin(global_invocation_id) gid: vec3u, @builtin(num_workgroups) nwg: vec3u) {
  let i = gid.y * nwg.x * 64u + gid.x;
  if (i < rsDispatch.treeCount) { queue_append(0u, i); }
}
@compute @workgroup_size(64)
fn consume(@builtin(workgroup_id) wid: vec3u, @builtin(num_workgroups) nwg: vec3u, @builtin(local_invocation_index) li: u32) {
  let i = queue_item(0u, wid, nwg, li);
  if (i == 0xFFFFFFFFu) { return; }
  atomicAdd(&visit[arena_word(arena_item_word(i))], 1u);
}
@compute @workgroup_size(64)
fn args_math(@builtin(global_invocation_id) gid: vec3u) {
  if (gid.x >= 8u) { return; }
  let ns = array<u32, 8>(0u, 1u, 64u, 65u, 65535u * 64u, 65535u * 64u + 1u, 65536u * 64u + 1u, 5u * 1048576u);
  let a = queue_args(ns[gid.x]);
  atomicStore(&visit[3u * gid.x], a.x); atomicStore(&visit[3u * gid.x + 1u], a.y); atomicStore(&visit[3u * gid.x + 2u], a.z);
}
`;

describe('T17: queue overflow and 2D indirect args (> 4.19 M items)', () => {
  it('5·2²⁰ items: each processed exactly once through rs_args + 2D indirect; capacity − 1 / + 1: overflow exactly when counter > capacity', async () => {
    const W = 1024, H = 1024, NS = 5;
    const rig = await restirRig(bitFixtureScene('c0c'), W, H, { settings: { slots: NS, maxBounces: 0 } });
    const k = rig.kernel, res = k.resources, dev = rig.g.device;
    const cap = W * H * NS;
    const argsP = await k.pipeline('rs_args');
    const entries = [SB(0), SB(1)];
    const defs = { RS_ARENA_BINDING: '0u', RS_ARENA_RW: true };
    const prod = await testPipeline(rig, T17_SRC, 'produce', entries, defs);
    const cons = await testPipeline(rig, T17_SRC, 'consume', entries, defs);
    const math = await testPipeline(rig, T17_SRC, 'args_math', entries, defs);
    const visit = storageBuffer(dev, (cap + 1) * 4);
    const g2 = dev.createBindGroup({ layout: prod.layout, entries: [{ binding: 0, resource: { buffer: res.arena } }, { binding: 1, resource: { buffer: visit } }] });
    // args math on the GPU (queue_args) = TS queueArgs
    await submit(rig, (enc) => { enc.clearBuffer(visit); k.encodeCustom(enc, math.pipeline, g2, {}, [1, 1], false); });
    const am = new Uint32Array(await readBuffer(dev, visit, 24 * 4));
    [0, 1, 64, 65, 65535 * 64, 65535 * 64 + 1, 65536 * 64 + 1, 5 * 2 ** 20].forEach((n, i) => expect(Array.from(am.subarray(3 * i, 3 * i + 3)), `n=${n}`).toEqual(queueArgs(n)));
    for (const n of [cap, cap - 1, cap + 1]) {
      const [gx, gy] = queueArgs(n);
      await submit(rig, (enc) => {
        enc.clearBuffer(res.arena, 0, 256);
        enc.clearBuffer(visit);
        k.encodeCustom(enc, prod.pipeline, g2, { treeCount: n }, [gx, gy], false);
        k.encodePass(enc, 'rs_args', argsP, res.g2('rs_args'), {}, [1, 1]);
        k.encodePass(enc, 'rs_args', cons.pipeline, g2, {}, { indirect: res.args, offset: 0 });
      });
      const v = new Uint32Array(await readBuffer(dev, visit, (cap + 1) * 4));
      const hdr = new Uint32Array(await readBuffer(dev, res.arena, 16));
      const args = new Uint32Array(await readBuffer(dev, res.args, 16));
      let once = 0, more = 0, zeroBelowN = 0;
      for (let i = 0; i <= cap; i++) { if (v[i] === 1) once++; else if (v[i] > 1) more++; else if (i < Math.min(n, cap)) zeroBelowN++; }
      console.log(`[T17] n=${n} cap=${cap}: header ${Array.from(hdr)} args ${Array.from(args.subarray(0, 3))} once=${once} more=${more} missing(<n)=${zeroBelowN}`);
      expect(hdr[0]).toBe(n);
      expect(hdr[1]).toBe(Math.min(n, cap));
      expect(hdr[2]).toBe(cap);
      expect(hdr[3]).toBe(n > cap ? 1 : 0);
      expect(Array.from(args.subarray(0, 3))).toEqual(queueArgs(Math.min(n, cap)));
      expect(more).toBe(0);
      expect(once).toBe(Math.min(n, cap));
      if (n <= cap) expect(zeroBelowN).toBe(0);
    }
    visit.destroy();
    rig.destroy();
  });
});

// ------------------------------------------------------------------------------------------------ T6(a) GPU

const T6A_SRC = `
#include "restir/mis.wgsl"
#include "common/rng.wgsl"
@group(2) @binding(0) var<storage, read_write> t6: array<f32>;
const T6_REC: u32 = 24u;
fn t6u(i: u32, d: u32) -> f32 { return u32_to_unit(pcg4d(vec4u(i, d, 0x7e57u, 0x6au)).x); }
fn t6conf(i: u32, d: u32) -> f32 {
  let cs = array<f32, 7>(1.0, 2.0, 3.7, 7.0, 20.0, 49.0, 343.0);
  return cs[min(u32(t6u(i, d) * 7.0), 6u)];
}
fn t6val(i: u32, d: u32, pZero: f32) -> f32 { return select(exp(8.0 * (t6u(i, d + 1u) - 0.5)), 0.0, t6u(i, d) < pZero); }
@compute @workgroup_size(64)
fn t6a(@builtin(global_invocation_id) gid: vec3u, @builtin(num_workgroups) nwg: vec3u) {
  let i = gid.y * nwg.x * 64u + gid.x;
  let k = 1u + min(u32(t6u(i, 0u) * 6.0), 5u);
  let cc = t6conf(i, 1u);
  let pcY = t6val(i, 2u, 0.15);
  var c: array<f32, 6>; var pf: array<f32, 6>;
  for (var j = 0u; j < k; j++) { c[j] = t6conf(i, 10u + j); pf[j] = t6val(i, 20u + 2u * j, 0.25); }
  let a = cc / f32(k);
  var s = 1.0;
  for (var j = 0u; j < k; j++) { s += mis_canonical_term(a, pcY, c[j], pf[j]); }
  let mc = s / f32(k + 1u);
  var sum = mc;
  let o = i * T6_REC;
  for (var j = 0u; j < 6u; j++) {
    var mj = 0.0;
    if (j < k) { mj = mis_partner_weight(a, c[j], pf[j], pcY, k); }
    sum += mj;
    t6[o + 4u + j] = c[j]; t6[o + 10u + j] = pf[j]; t6[o + 16u + j] = mj;
  }
  t6[o] = f32(k); t6[o + 1u] = cc; t6[o + 2u] = pcY; t6[o + 3u] = mc; t6[o + 22u] = sum - 1.0;
}
`;

describe('T6(a) GPU: production mis_* functions', () => {
  it('|Σm − 1| < 1e-5 at arbitrary y (X_c or Y_j) for 2²⁰ random configs; m values agree with the f64 mis-ref to 1e-6', async () => {
    const rig = await restirRig(bitFixtureScene('c0c'), 8, 8, {});
    const dev = rig.g.device, N = 1 << 20;
    const buf = storageBuffer(dev, N * 24 * 4);
    const { pipeline, layout } = await testPipeline(rig, T6A_SRC, 't6a', [SB(0)], {});
    const g2 = dev.createBindGroup({ layout, entries: [{ binding: 0, resource: { buffer: buf } }] });
    await submit(rig, (enc) => rig.kernel.encodeCustom(enc, pipeline, g2, {}, [4096, 4], false));
    const f = new Float32Array(await readBuffer(dev, buf, N * 24 * 4));
    let worstSum = 0, worstRel = 0, nz = 0;
    for (let i = 0; i < N; i++) {
      const o = 24 * i, k = f[o];
      worstSum = Math.max(worstSum, Math.abs(f[o + 22]));
      const c = Array.from(f.subarray(o + 4, o + 4 + k)), pf = Array.from(f.subarray(o + 10, o + 10 + k));
      const ref = misWeightsAt(f[o + 1], c, f[o + 2], pf);
      const cmp = (g: number, r: number) => { if (r > 1e-6) worstRel = Math.max(worstRel, Math.abs(g - r) / r); else if (Math.abs(g - r) > 1e-12) nz++; };
      cmp(f[o + 3], ref.mc);
      for (let j = 0; j < k; j++) cmp(f[o + 16 + j], ref.mj[j]);
    }
    console.log(`[T6(a) GPU] 2^20 configs: max |Σm − 1| = ${worstSum.toExponential(3)}, max rel |m − m_f64| = ${worstRel.toExponential(3)}, tiny-value abs misses ${nz}`);
    expect(worstSum).toBeLessThan(1e-5);
    expect(worstRel).toBeLessThan(1e-6);
    expect(nz).toBe(0);
    buf.destroy();
    rig.destroy();
  });
});

// ------------------------------------------------------------------------------------------------ U-ENS-1, U-ENS-2

describe('U-ENS-1 / U-ENS-2: ensemble atlas', () => {
  it('member m of an E = 4 atlas ≡ sequential run of member m (reservoirs and radiance, bitwise); ensStats/ensPixel = f64 reduction; npz round trip', async () => {
    const scene = bitFixtureScene('x_quads');
    const W = 128, H = 128, E = 4;
    const settings = { maxBounces: 2, trees: 4 };
    const ens = await restirRig(scene, W, H, { preset: 'offline', settings, members: E, seed: 4002 });
    const re = await ens.frames(1, 3);
    expect(re.counters).toEqual([0, 0, 0, 0]);
    expect(re.arena.rsc.pendingLeft + re.arena.rsc.slotMismatch).toBe(0);
    const a = ens.kernel.resources.alloc;
    const atlasRes = await ens.kernel.readReservoirs('final');
    const atlasFrame = new Float32Array((await readTexture4(ens.g.device, ens.kernel.resources.frameTex)).buffer);
    let resBad = 0, frameBad = 0;
    for (let m = 0; m < E; m++) {
      const seq = await restirRig(scene, W, H, { preset: 'offline', settings, memberBase: m, seed: 4002 });
      await seq.frames(1, 3);
      const sr = await seq.kernel.readReservoirs('final');
      const sf = new Float32Array((await readTexture4(seq.g.device, seq.kernel.resources.frameTex)).buffer);
      const ox = (m % a.memberCols) * W, oy = Math.floor(m / a.memberCols) * H;
      for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
        const ai = (oy + y) * a.atlasW + ox + x, li = y * W + x;
        for (let w = 0; w < RES_WORDS; w++) if (atlasRes[ai * RES_WORDS + w] !== sr[li * RES_WORDS + w]) { resBad++; break; }
        for (let c = 0; c < 3; c++) if (!Object.is(atlasFrame[4 * ai + c], sf[4 * li + c])) { frameBad++; break; }
      }
      seq.destroy();
    }
    console.log(`[U-ENS-1] E=${E} ${W}×${H}: reservoir mismatches ${resBad}, radiance mismatches ${frameBad}`);
    expect(resBad).toBe(0);
    expect(frameBad).toBe(0);

    // U-ENS-2: ensStats / ensPixel vs f64 reduction of the read-back atlas
    const dev = ens.g.device, res = ens.kernel.resources;
    const raw = new Float32Array(await readBuffer(dev, res.ensStats, ensStatsFloats(E, W, H) * 4));
    const st = decodeEnsStats(raw, E, W, H);
    const px = new Float32Array(await readBuffer(dev, res.ensPixel, W * H * 32));
    const L = ensStatsLayout(E, W, H);
    let worst = 0;
    const rel = (g: number, r: number) => { worst = Math.max(worst, Math.abs(g - r) / Math.max(Math.abs(r), 1e-3)); };
    for (let m = 0; m < E; m++) {
      const ox = (m % a.memberCols) * W, oy = Math.floor(m / a.memberCols) * H;
      const v = (x: number, y: number, c: number) => atlasFrame[4 * ((oy + y) * a.atlasW + ox + x) + c];
      for (const tl of L.tiles) for (let ty = 0; ty < tl.th; ty++) for (let tx = 0; tx < tl.tw; tx++) for (let c = 0; c < 3; c++) {
        let s = 0;
        for (let y = ty * tl.level; y < Math.min(H, (ty + 1) * tl.level); y++) for (let x = tx * tl.level; x < Math.min(W, (tx + 1) * tl.level); x++) s += v(x, y, c);
        rel(st.tiles[tl.level].data[3 * ((m * tl.th + ty) * tl.tw + tx) + c], s);
      }
      for (let c = 0; c < 3; c++) { let s = 0; for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) s += v(x, y, c); rel(st.global[3 * m + c], s); }
    }
    for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) for (let c = 0; c < 3; c++) {
      let s = 0, q = 0;
      for (let m = 0; m < E; m++) { const val = atlasFrame[4 * ((Math.floor(m / a.memberCols) * H + y) * a.atlasW + (m % a.memberCols) * W + x) + c]; s += val; q += val * val; }
      rel(px[8 * (y * W + x) + c], s);
      rel(px[8 * (y * W + x) + 4 + c], q);
    }
    console.log(`[U-ENS-2] max rel error of ensStats / ensPixel vs f64 = ${worst.toExponential(3)}`);
    expect(worst).toBeLessThan(1e-6);
    // npz round trip of the collected frame
    const col = new EnsembleCollector(E, W, H, 4002);
    col.addFrameStats(st, 3);
    col.addPixelMoments(px);
    const z = readNpz(writeNpz(col.npzArrays()));
    expect(z.get('tiles16')!.shape).toEqual([E, H / 16, W / 16, 3]);
    expect(Array.from(z.get('global')!.data as Float64Array)).toEqual(Array.from(st.global));
    expect(col.seeds).toEqual(['4002:3:0', '4002:3:1', '4002:3:2', '4002:3:3']);
    ens.destroy();
  });
});

// ------------------------------------------------------------------------------------------------ replay predicate (C3)

const PRED_SRC = `
#include "restir/shift.wgsl"
@group(2) @binding(0) var<storage, read_write> pr: array<u32>;
fn pa_needs_replay(flags: u32) -> bool {
  let k = rf_k(flags);
  return !res_empty(flags) && (k > 2u || k == 0u);
}
@compute @workgroup_size(64)
fn pred(@builtin(global_invocation_id) gid: vec3u) {
  if (gid.x >= 4096u) { return; }
  let f = gid.x | ((gid.x >> 8u) << 16u);    // d (bits 0–3), k (4–7), tech (8–9), ep (10–11 → 10–12), forced/bg
  pr[gid.x] = select(0u, 1u, res_needs_replay(f)) | (select(0u, 2u, pa_needs_replay(f)));
}
`;

describe('Changelog C3: pa_needs_replay (pair-accept) ≡ res_needs_replay (shift.wgsl)', () => {
  it('agree on all (d, k, technique, endpoint, forced, bg) flag combinations', async () => {
    const rig = await restirRig(bitFixtureScene('c0c'), 8, 8, {});
    const dev = rig.g.device;
    // the pair-accept copy must be textually the same predicate: extract it from the pass source
    const { shaderSources } = await import('../../src/core/shaders/index.ts');
    const pa = /fn pa_needs_replay[\s\S]*?\n}\n/.exec(shaderSources['passes/restir/pair-accept.wgsl'])![0];
    const src = PRED_SRC.replace(/fn pa_needs_replay[\s\S]*?\n}\n/, pa);
    const buf = storageBuffer(dev, 4096 * 4);
    const { pipeline, layout } = await testPipeline(rig, src, 'pred', [SB(0)], {}, true);
    const g2 = dev.createBindGroup({ layout, entries: [{ binding: 0, resource: { buffer: buf } }] });
    await submit(rig, (enc) => rig.kernel.encodeCustom(enc, pipeline, g2, {}, [64, 1], true));
    const v = new Uint32Array(await readBuffer(dev, buf, 4096 * 4));
    let bad = 0, replay = 0;
    for (let i = 0; i < 4096; i++) { if (v[i] !== 0 && v[i] !== 3) bad++; if (v[i] === 3) replay++; }
    expect(rfPack({ d: 3, k: 3 }) & 0xFF).toBe(0x33);
    console.log(`[C3] 4096 flag words: disagreements ${bad}, needs-replay ${replay}`);
    expect(bad).toBe(0);
    buf.destroy();
    rig.destroy();
  });
});

// ------------------------------------------------------------------------------------------------ T7 GRIS toys

// 1D toys (validation-harness T7): K = 4 "pixels" on [0, 1) with integrands f_i(x) = 1 + ½ sin 2π(x + ψ_i) (optional
// hole of width 0.3 = partial support), initial RIS with 4 uniform candidates (non-uniform canonical sampling), then
// R rounds of fully connected paired reuse (k = 3) with the PRODUCTION mis_canonical_term / mis_partner_weight /
// ris_update. Shifts go through base coordinates g_i: translation g_i(x) = fract(x + φ_i) (|J| = 1) or power warp
// g_i(x) = x^γ_i (|J| = g_i'(x)/g_j'(y)); T_{i→j} is undefined where g_i(x) falls in a pair-specific base interval of
// width 0.2 (symmetric, so T_{j→i} is undefined on the image). Output of pixel 0: (Y, W_Y).
// Negative controls: T7_NEG = 1 constant m = 1/(k+1); T7_NEG = 2 Jacobian omitted.
const T7_SRC = `
#include "restir/mis.wgsl"
#include "common/rng.wgsl"
@group(2) @binding(0) var<storage, read_write> t7out: array<vec2f>;
const T7K: u32 = 4u;
const T7_MODE: u32 = u32($T7_MODE);
const T7_UNDEF: bool = bool($T7_UNDEF);
const T7_PARTIAL: bool = bool($T7_PARTIAL);
const T7_NEG: u32 = u32($T7_NEG);
const T7_ROUNDS: u32 = u32($T7_ROUNDS);
fn t7_phi(i: u32) -> f32 { let a = array<f32, 4>(0.0, 0.17, 0.55, 0.81); return a[i]; }
fn t7_gam(i: u32) -> f32 { let a = array<f32, 4>(1.0, 0.6, 1.7, 2.5); return a[i]; }
fn t7_psi(i: u32) -> f32 { let a = array<f32, 4>(0.1, 0.4, 0.7, 0.25); return a[i]; }
fn t7_hole(i: u32) -> f32 { let a = array<f32, 4>(0.3, 0.6, 0.1, 0.45); return a[i]; }
fn t7_f(i: u32, x: f32) -> f32 {
  if (T7_PARTIAL && x >= t7_hole(i) && x < t7_hole(i) + 0.3) { return 0.0; }
  return 1.0 + 0.5 * sin(6.283185307179586 * (x + t7_psi(i)));
}
fn t7_g(i: u32, x: f32) -> f32 { if (T7_MODE == 0u) { return fract(x + t7_phi(i)); } return pow(x, t7_gam(i)); }
fn t7_ginv(i: u32, u: f32) -> f32 { if (T7_MODE == 0u) { return fract(u - t7_phi(i)); } return pow(u, 1.0 / t7_gam(i)); }
fn t7_gd(i: u32, x: f32) -> f32 { if (T7_MODE == 0u) { return 1.0; } return t7_gam(i) * pow(x, t7_gam(i) - 1.0); }
/// Shift T_{i→j}(x) → (y, |J|); J = 0: undefined.
fn t7_T(i: u32, j: u32, x: f32) -> vec2f {
  let u = t7_g(i, x);
  if (T7_UNDEF) {
    let lo = fract(0.13 * f32(min(i, j) * 4u + max(i, j)) + 0.07);
    if (fract(u - lo) < 0.2) { return vec2f(0.0); }
  }
  let y = t7_ginv(j, u);
  if (!(y > 0.0 && y < 1.0)) { return vec2f(0.0); }
  return vec2f(y, t7_gd(i, x) / t7_gd(j, y));
}
var<private> t7ctr: u32;
var<private> t7key: vec2u;
fn t7_rnd() -> f32 { t7ctr += 1u; return (f32(pcg4d(vec4u(t7key.x, t7key.y, t7ctr, 0x77u)).x >> 8u) + 0.5) * (1.0 / 16777216.0); }

@compute @workgroup_size(64)
fn t7(@builtin(global_invocation_id) gid: vec3u, @builtin(num_workgroups) nwg: vec3u) {
  let tid = gid.y * nwg.x * 64u + gid.x;
  t7key = vec2u(tid, rsDispatch.t);
  t7ctr = 0u;
  var X: array<f32, 4>; var W: array<f32, 4>; var Cf: array<f32, 4>;
  for (var i = 0u; i < T7K; i++) {
    var wSum = 0.0; var sel = -1.0;
    for (var m = 0u; m < 4u; m++) {
      let x = t7_rnd();
      if (ris_update(&wSum, t7_f(i, x), t7_rnd())) { sel = x; }
    }
    X[i] = sel; W[i] = select(0.0, wSum / (4.0 * t7_f(i, sel)), wSum > 0.0); Cf[i] = 1.0;
  }
  for (var r = 0u; r < T7_ROUNDS; r++) {
    var X2: array<f32, 4>; var W2: array<f32, 4>; var C2: array<f32, 4>;
    for (var c = 0u; c < T7K; c++) {
      let k = T7K - 1u;
      let a = Cf[c] / f32(k);
      let hasC = X[c] >= 0.0;
      let pc = select(0.0, t7_f(c, X[c]), hasC);
      var s = 1.0; var cout = Cf[c];
      for (var j = 0u; j < T7K; j++) {
        if (j == c) { continue; }
        var H = 0.0;
        if (hasC) {
          let t = t7_T(c, j, X[c]);
          if (t.y > 0.0) { H = t7_f(j, t.x) * select(t.y, 1.0, T7_NEG == 2u); }
        }
        s += mis_canonical_term(a, pc, Cf[j], H);
        cout += Cf[j];
      }
      let mc = select(s / f32(k + 1u), 1.0 / f32(k + 1u), T7_NEG == 1u);
      var wSum = 0.0; var selY = X[c]; var selP = pc;
      if (ris_update(&wSum, mc * pc * W[c], t7_rnd())) { selY = X[c]; selP = pc; }
      for (var j = 0u; j < T7K; j++) {
        if (j == c) { continue; }
        let u = t7_rnd();
        if (X[j] < 0.0) { continue; }
        let t = t7_T(j, c, X[j]);
        if (!(t.y > 0.0)) { continue; }
        let fy = t7_f(c, t.x);
        let G = fy * select(t.y, 1.0, T7_NEG == 2u);
        if (!(G > 0.0)) { continue; }
        let mj = select(mis_partner_weight(a, Cf[j], t7_f(j, X[j]), G, k), 1.0 / f32(k + 1u), T7_NEG == 1u);
        if (ris_update(&wSum, mj * G * W[j], u)) { selY = t.x; selP = fy; }
      }
      X2[c] = select(-1.0, selY, wSum > 0.0);
      W2[c] = select(0.0, wSum / selP, wSum > 0.0);
      C2[c] = cout;
    }
    X = X2; W = W2; Cf = C2;
  }
  t7out[tid] = vec2f(X[0], W[0]);
}
`;

interface ToyCfg { name: string; mode: 0 | 1; undef: boolean; partial: boolean; neg: 0 | 1 | 2; rounds: number; chunks: number; expectPass: boolean }

/** ∫ f_0 over [lo, hi) (f64, analytic), with the optional hole [0.3, 0.6). */
function toyIntegral(lo: number, hi: number, partial: boolean): number {
  const F = (x: number) => x - (0.5 / (2 * Math.PI)) * Math.cos(2 * Math.PI * (x + 0.1));
  const seg = (a: number, b: number) => (b > a ? F(b) - F(a) : 0);
  if (!partial) return seg(lo, hi);
  return seg(lo, Math.min(hi, 0.3)) + seg(Math.max(lo, 0.6), hi);
}
const toyLen = (lo: number, hi: number, partial: boolean) => (hi - lo) - (partial ? Math.max(0, Math.min(hi, 0.6) - Math.max(lo, 0.3)) : 0);

describe('T7: GRIS toy integrals with the production RIS/MIS WGSL', () => {
  const cfgs: ToyCfg[] = [
    { name: 'translation, 1 round', mode: 0, undef: false, partial: false, neg: 0, rounds: 1, chunks: 12, expectPass: true },
    { name: 'translation + undefined regions, 2 rounds', mode: 0, undef: true, partial: false, neg: 0, rounds: 2, chunks: 12, expectPass: true },
    { name: 'power warp, 2 rounds', mode: 1, undef: false, partial: false, neg: 0, rounds: 2, chunks: 12, expectPass: true },
    { name: 'power warp + undefined + partial support, 2 rounds', mode: 1, undef: true, partial: true, neg: 0, rounds: 2, chunks: 12, expectPass: true },
    { name: 'NEG m = 1/M, translation + undefined', mode: 0, undef: true, partial: false, neg: 1, rounds: 1, chunks: 2, expectPass: false },
    { name: 'NEG m = 1/M, translation + partial support', mode: 0, undef: false, partial: true, neg: 1, rounds: 1, chunks: 2, expectPass: false },
    { name: 'NEG Jacobian omitted, power warp', mode: 1, undef: false, partial: false, neg: 2, rounds: 1, chunks: 2, expectPass: false },
  ];
  it('E[f(Y)W] = ∫f (z-test, 10⁸ trials) and the 64-bin W-histogram identity; negative controls fail', async () => {
    const rig = await restirRig(bitFixtureScene('c0c'), 8, 8, {});
    const dev = rig.g.device;
    const N = 1 << 23;
    const buf = storageBuffer(dev, N * 8);
    const results: string[] = [];
    for (const cfg of cfgs) {
      const { pipeline, layout } = await testPipeline(rig, T7_SRC, 't7', [SB(0)],
        { T7_MODE: cfg.mode, T7_UNDEF: cfg.undef ? 1 : 0, T7_PARTIAL: cfg.partial ? 1 : 0, T7_NEG: cfg.neg, T7_ROUNDS: cfg.rounds });
      const g2 = dev.createBindGroup({ layout, entries: [{ binding: 0, resource: { buffer: buf } }] });
      let s1 = 0, s2 = 0, n = 0;
      const b1 = new Float64Array(64), b2 = new Float64Array(64);
      for (let ch = 0; ch < cfg.chunks; ch++) {
        await submit(rig, (enc) => rig.kernel.encodeCustom(enc, pipeline, g2, { t: 1000 + ch }, [N / 64 / 16, 16], false));
        const v = new Float32Array(await readBuffer(dev, buf, N * 8));
        for (let i = 0; i < N; i++) {
          const y = v[2 * i], w = v[2 * i + 1];
          n++;
          if (!(w > 0) || y < 0) continue;
          const fy = cfg.partial && y >= 0.3 && y < 0.6 ? 0 : 1 + 0.5 * Math.sin(2 * Math.PI * (y + 0.1));
          const e = fy * w;
          s1 += e; s2 += e * e;
          const b = Math.min(63, Math.floor(y * 64));
          b1[b] += w; b2[b] += w * w;
        }
      }
      const I = toyIntegral(0, 1, cfg.partial);
      const mean = s1 / n, se = Math.sqrt(Math.max(s2 / n - mean * mean, 0) / n);
      const z = (mean - I) / se;
      let zBin = 0;
      for (let b = 0; b < 64; b++) {
        const m = b1[b] / n, sb = Math.sqrt(Math.max(b2[b] / n - m * m, 0) / n);
        const ref = toyLen(b / 64, (b + 1) / 64, cfg.partial);
        zBin = Math.max(zBin, Math.abs(m - ref) / Math.max(sb, 1e-12));
      }
      const pass = Math.abs(z) < 4.5 && zBin < 4.5;
      results.push(`${cfg.name}: N=${n.toExponential(2)} E[fW]=${mean.toFixed(6)} (∫f=${I.toFixed(6)}, rel ${((mean - I) / I).toExponential(2)}) z=${z.toFixed(2)} max|z_bin|=${zBin.toFixed(2)} → ${pass ? 'pass' : 'FAIL'}`);
      expect(pass, results.at(-1)).toBe(cfg.expectPass);
    }
    console.log(`[T7]\n  ${results.join('\n  ')}`);
    buf.destroy();
    rig.destroy();
  }, 600_000);
});

// ------------------------------------------------------------------------------------------------ T6(b)

// For every VALID partner slot (j, s) of canonical c (Y = T_{j→c}(X_j), G = F_c(Y)·J, J): rebuild the record that
// the write-back would produce (resIn[j] with F = G/J, jDen = J·jDen_j), shift it back into j with the production
// shift_hybrid (replay compiled in) and compare a = p̂_j(X_j)/J_{j→c} (the neighbour-weight input) with
// b = p̂_j(T_{c→j}(Y))·J_{c→j}(Y) = lum(FJ_back) (the canonical-weight input).
const T6B_SRC = `
#include "restir/frame.wgsl"
#include "restir/reservoir.wgsl"
#include "restir/queue.wgsl"
#include "restir/pairing.wgsl"
#include "restir/shift.wgsl"
@group(2) @binding(5) var<storage, read_write> t6b: array<atomic<u32>>;
// 0 tested, 1 back-shift not OK, 2 |a − b| > 1e-3·max, 3 max rel (f32 bits, atomicMax), 4 |a − b| > 1e-5·max
@compute @workgroup_size(8, 8, 1)
fn t6b_main(@builtin(global_invocation_id) gid: vec3u) {
  let p = rs_pix(gid.xy);
  if (!p.valid) { return; }
  var tested = 0u; var undef = 0u; var bad = 0u; var bad5 = 0u; var worst = 0.0;
  for (var s = 0u; s < rsParams.numSlots; s++) {
    let pr = pair_partner(p.local, p.member, rs_t(), 0u, s);
    if (!pr.valid) { continue; }
    let jpx = pair_atlas_px(p, pr.partner);
    let j = pair_atlas_index(jpx);
    let g = arena_slot(j, s);
    if (!jw_valid(g.w)) { continue; }
    let G = bitcast<vec3f>(g.xyz);
    let J = bitcast<f32>(g.w);
    var src = shift_src_load(j);
    let a = luminance(src.F) / J;
    src.F = G / J;
    src.jDen = J * src.jDen;
    let o = shift_hybrid(src, shift_dst_load(jpx));
    tested++;
    if (rs_slot_code_sc(o.code) != SC_OK) { undef++; continue; }
    let b = luminance(o.FJ);
    let rel = abs(a - b) / max(max(a, b), 1e-30);
    worst = max(worst, rel);
    if (rel > 1e-3) {
      bad++;
      let slotIdx = atomicAdd(&t6b[5], 1u);
      if (slotIdx < 8u) {
        let o8 = 8u + 8u * slotIdx;
        atomicStore(&t6b[o8], src.flags); atomicStore(&t6b[o8 + 1u], bitcast<u32>(J)); atomicStore(&t6b[o8 + 2u], bitcast<u32>(o.J));
        atomicStore(&t6b[o8 + 3u], bitcast<u32>(a)); atomicStore(&t6b[o8 + 4u], bitcast<u32>(b)); atomicStore(&t6b[o8 + 5u], src.rc.x);
        atomicStore(&t6b[o8 + 6u], bitcast<u32>(src.jDen)); atomicStore(&t6b[o8 + 7u], bitcast<u32>(rel));
      }
    }
    if (rel > 1e-5) { bad5++; }
  }
  atomicAdd(&t6b[0], tested); atomicAdd(&t6b[1], undef); atomicAdd(&t6b[2], bad); atomicAdd(&t6b[4], bad5);
  atomicMax(&t6b[3], bitcast<u32>(worst));
}
`;

describe('T6(b): stored vs recomputed p̂_{←j} (real shifts)', () => {
  for (const name of ['x_quads', 'c0s'] as const) {
    it(`${name}: p̂_j(X_j)/J_{j→c} = p̂_j(T_{c→j}(Y_j))·J_{c→j}(Y_j) within 1e-3`, async () => {
      const rig = await restirRig(bitFixtureScene(name), 128, 96, { preset: 'offline', settings: { maxBounces: 3, trees: 4, rounds: 1 } });
      const k = rig.kernel, res = k.resources, dev = rig.g.device;
      await rig.frames(1);
      const out = storageBuffer(dev, 80 * 4);
      const { pipeline, layout } = await testPipeline(rig, T6B_SRC, 't6b_main', [
        { binding: 0, visibility: C, buffer: { type: 'read-only-storage' } }, SB(1), TEXU(2), TEXF(3), TEXP(4), SB(5)],
      { RS_RES_IN_BINDING: '0u', RS_ARENA_BINDING: '1u', RS_ARENA_RW: true, RS_VBUF_BINDING: '2u', RS_GEO_BINDING: '3u', RS_PAIRTEX_BINDING: '4u', RS_REPLAY: 1 }, true);
      const g2 = dev.createBindGroup({ layout, entries: [
        { binding: 0, resource: { buffer: res.res[0] } }, { binding: 1, resource: { buffer: res.arena } }, { binding: 2, resource: res.views.vbuf },
        { binding: 3, resource: res.views.geo }, { binding: 4, resource: res.views.pair }, { binding: 5, resource: { buffer: out } }] });
      await submit(rig, (enc) => k.encodeCustom(enc, pipeline, g2, { t: 0 }, [16, 12], true));
      const o = new Uint32Array(await readBuffer(dev, out, 80 * 4));
      const of = new Float32Array(o.buffer);
      for (let i = 0; i < Math.min(8, o[5]); i++) {
        const b8 = 8 + 8 * i, fl = o[b8];
        console.log(`[T6(b) ${name}] outlier d=${fl & 15} k=${(fl >> 4) & 15} tech=${(fl >> 8) & 3} ep=${(fl >> 10) & 7} lk-1=${(fl >> 14) & 7} lk=${(fl >> 18) & 7} J=${of[b8 + 1]} Jback=${of[b8 + 2]} J·Jback=${of[b8 + 1] * of[b8 + 2]} a=${of[b8 + 3]} b=${of[b8 + 4]} rc=${o[b8 + 5].toString(16)} rel=${of[b8 + 7]}`);
      }
      const worst = new Float32Array(o.buffer, 12, 1)[0];
      console.log(`[T6(b) ${name}] tested ${o[0]}, back-shift not OK ${o[1]}, |a−b| > 1e-3·max: ${o[2]}, > 1e-5·max: ${o[4]}, max rel ${worst.toExponential(3)}`);
      expect(o[0]).toBeGreaterThan(1000);
      expect(o[2]).toBe(0);
      expect(o[1] / o[0]).toBeLessThan(1e-3);
      out.destroy();
      rig.destroy();
    });
  }
});

// ------------------------------------------------------------------------------------------------ app-path regression

// App smoke bug (M4, Changelog C8): RestirFramePass.encode bracketed the ReSTIR passes with two compute passes carrying
// the app's frame timestamp writes. In some page loads (not the first page of a fresh browser, so a fresh test page
// cannot reproduce the GPU-side symptom) Chrome 154 / Metal then lost the frame's ReSTIR effects with no WebGPU error:
// arena header 0 with q0 capacity kept (both clearBuffer ranges as if applied last), view 460 unwritten, image ≈ L1,
// GPU time halved. Plain Cornell and Cornell + HDRI failed alike; frames encoded without ReSTIR timestamp writes never
// failed (measured in failing pages: manual frames, app frames with the timestamp ring disabled, timestamps on the M1
// primary pass only). Guard: the pass must not put timestampWrites on any pass it encodes, and must render the same
// result with and without a timestamp argument; driven like the renderer: 540p, env + point + rect.
describe('interactive RestirFramePass with frame timestamps (app path, 540p, env + analytic lights)', () => {
  it('the spatial stage runs every frame: accepted > 0, q0 capacity set, SC histogram non-empty, same result as without timestamps', async () => {
    const { FrameUniformBuffer, JITTER_IID, FRAME_RESET_HISTORY } = await import('../../src/core/render/frame-uniforms.ts');
    const { RestirKernel } = await import('../../src/core/render/restir/kernel.ts');
    const { restirSettings, numSlotsOf } = await import('../../src/core/render/restir/presets.ts');
    const NS = numSlotsOf(restirSettings('interactive'));   // M5: slots 3 + boost 3 (temporal on; TD21, Q7)
    const { gpuScene, boxScene, boxCamera, light } = await import('./restir-fixtures.ts');
    const { lightMatrixToward } = await import('./pt-fixtures.ts');
    const { synthEnvData } = await import('./env-fixtures.ts');
    const down = (p: [number, number, number]) => lightMatrixToward([0, -1, 0], p);
    const scene = boxScene([light({ id: 1, type: 'point', power: 30, matrix: down([0.2, 1.7, -0.5]) }),
      light({ id: 2, type: 'rect', power: 40, sizeX: 0.5, sizeY: 0.3, matrix: down([0, 1.95, -0.9]) })], { env: synthEnvData(128, 64) });
    const g = await gpuScene(scene);
    const dev = g.device;
    const W = 960, H = 540;
    const hasTs = g.features.has('timestamp-query');
    const qs = hasTs ? dev.createQuerySet({ type: 'timestamp', count: 8 }) : undefined;
    const results: { ts: boolean; acc: number[]; cap: number[]; codes: number[]; mean: number }[] = [];
    for (const useTs of [false, true]) {
      if (useTs && !qs) continue;
      const pass = await RestirKernel.interactive(dev, g.gpu, g.env, 'rgba16float', { settings: restirSettings('interactive', { maxBounces: 3 }), features: g.features, wgslLanguageFeatures: g.wgslLanguageFeatures });
      const color = dev.createTexture({ size: [W, H], format: 'rgba16float', usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.COPY_SRC });
      const fu = new FrameUniformBuffer(dev);
      pass.setTargets({ width: W, height: H, color, frameUniforms: fu.buffer });
      const k = pass.kernel;
      const r = { ts: useTs, acc: [] as number[], cap: [] as number[], codes: [] as number[], mean: 0 };
      dev.pushErrorScope('validation');
      for (let f = 0; f < 4; f++) {
        fu.write({ camera: boxCamera(), prevCamera: boxCamera(), width: W, height: H, frameIndex: f, seedIndex: f, runSeed: 7, flags: f === 0 ? FRAME_RESET_HISTORY : 0,
          jitterMode: JITTER_IID, jitter: [0.5, 0.5], origin: g.gpu.origin, exposure: 1, time: 0, dt: 0, sceneDiag: 1 });
        const enc = dev.createCommandEncoder();
        const bcp = enc.beginComputePass.bind(enc);
        let tsPasses = 0;
        enc.beginComputePass = (d?: GPUComputePassDescriptor) => { if (d?.timestampWrites) tsPasses++; return bcp(d); };
        enc.clearBuffer(k.resources.arena, 0, 256);
        expect(pass.encode(enc, { advanced: true, accumulate: true }, qs ? { querySet: qs, beginningOfPassWriteIndex: 0, endOfPassWriteIndex: 1 } : undefined)).toBe(true);
        expect(tsPasses, 'compute passes with timestampWrites').toBe(0);
        dev.queue.submit([enc.finish()]);
        const c = await k.readCounters(false);
        r.acc.push(c.rsc.accepted); r.cap.push(c.queues[0].capacity); r.codes.push(c.codes.reduce((a, b) => a + b, 0));
      }
      const err = await dev.popErrorScope();
      expect(err?.message ?? '').toBe('');
      const frame = new Float32Array((await readTexture4(dev, k.resources.frameTex)).buffer);
      let m = 0; for (let i = 0; i < W * H; i++) m += frame[4 * i] + frame[4 * i + 1] + frame[4 * i + 2];
      r.mean = m / (3 * W * H);
      results.push(r);
      pass.destroy(); color.destroy(); fu.destroy();
    }
    console.log(`[app-path] timestamp-query ${hasTs}: ${JSON.stringify(results)}`);
    for (const r of results) {
      for (let f = 0; f < 4; f++) {
        expect(r.acc[f], `ts ${r.ts} frame ${f} accepted`).toBeGreaterThan(W * H * 0.5);
        expect(r.cap[f], `ts ${r.ts} frame ${f} q0 capacity`).toBe(W * H * NS);
        expect(r.codes[f], `ts ${r.ts} frame ${f} SC histogram`).toBe(W * H * NS);
      }
    }
    if (results.length === 2) expect(results[1].mean).toBe(results[0].mean);
    qs?.destroy(); g.destroy();
  });
});

// ------------------------------------------------------------------------------------------------ T3-3-boost (M5, T-D)

describe('T3-3-boost: reciprocal disocclusion boost (restir-temporal-api.md TD21, §3.7; T-D)', () => {
  it('symmetric, only pairs with a disoccluded member, no slot mismatch; no dis ⇒ no boost pair; ordinary slots unchanged', async () => {
    const W = 64, H = 48, P = W * H, t = 7, BOOST0 = 3;
    const rig = await restirRig(bitFixtureScene('x_quads'), W, H, { preset: 'full', settings: { maxBounces: 3, boostSlots: 3 } });
    const k = rig.kernel;
    const NS = k.resources.alloc.slots;
    expect(NS, 'numSlots = slots 3 + boost 3').toBe(6);
    const aw = arenaWords(P, NS);
    const R = k.settings.diskRadius;
    // The frame is built without advance(): a reset frame without temporal units, so the synthetic tState flags
    // (TS_DISOCC or TS_QVALID) written here are what rs_pair_accept reads.
    const run = async (dis: (ai: number) => boolean) => {
      const words = new Uint32Array(TS_WORDS * P);
      for (let ai = 0; ai < P; ai++) words[TS_WORDS * ai + TSW.flags] = dis(ai) ? TS_CONSTS.TS_DISOCC : TS_CONSTS.TS_QVALID;
      // perf2 WP-6 (RS_TSTATE_SOA forced): the GPU tState region is word-major
      rig.g.device.queue.writeBuffer(k.resources.arena, 256 + 4 * aw.tState, k.perfFlags.RS_TSTATE_SOA ? tStateAosToSoa(words, P) : words);
      const r = await rig.frames(1, t);
      const body = await arenaBody(rig);
      const acc = (ai: number, s: number) => body[aw.slots + 4 * (ai * NS + s) + 3] >>> 0 !== K.JW_NOT_ACCEPTED;
      return { r, acc };
    };
    const partner = (ai: number, s: number): number | undefined => {
      const layer = pairLayer(PAIR_TEX_SIZES[s], R, s);
      const q = pairPartner(layer, pairTransform(11, 0, t, 0, s, PAIR_TEX_SIZES[s]), [ai % W, Math.floor(ai / W)], W, H);
      return q ? q[1] * W + q[0] : undefined;
    };
    const r0 = rng32(17);
    const disR = new Uint8Array(P).map(() => (r0() < 0.2 ? 1 : 0));
    const all = await run(() => true), none = await run(() => false), rnd = await run((ai) => disR[ai] === 1);
    let boostAll = 0, boostRnd = 0, bad = 0;
    const msgs: string[] = [];
    for (let ai = 0; ai < P; ai++) {
      for (let s = 0; s < NS; s++) {
        const q = partner(ai, s);
        for (const [name, x] of [['all', all], ['none', none], ['rnd', rnd]] as const) {
          if (q === undefined) { if (x.acc(ai, s)) { bad++; msgs.push(`${name} ${ai}/${s}: accepted without partner`); } continue; }
          if (x.acc(ai, s) !== x.acc(q, s)) { bad++; if (msgs.length < 6) msgs.push(`${name} ${ai}/${s}: asymmetric`); }
        }
        if (s < BOOST0) {
          if (all.acc(ai, s) !== none.acc(ai, s) || all.acc(ai, s) !== rnd.acc(ai, s)) { bad++; if (msgs.length < 6) msgs.push(`ordinary slot ${ai}/${s} changed`); }
          continue;
        }
        if (none.acc(ai, s)) { bad++; if (msgs.length < 6) msgs.push(`none: boost ${ai}/${s} accepted`); }
        const wantRnd = all.acc(ai, s) && q !== undefined && (disR[ai] === 1 || disR[q] === 1);
        if (rnd.acc(ai, s) !== wantRnd) { bad++; if (msgs.length < 6) msgs.push(`rnd ${ai}/${s}: ${rnd.acc(ai, s)} ≠ ${wantRnd}`); }
        if (all.acc(ai, s)) boostAll++;
        if (rnd.acc(ai, s)) boostRnd++;
      }
    }
    for (const x of [all, none, rnd]) {
      expect(x.r.arena.rsc.slotMismatch, 'RSC_SLOT_MISMATCH').toBe(0);
      expect(x.r.arena.rsc.pendingLeft, 'RSC_PENDING_LEFT').toBe(0);
      expect(x.r.arena.queues[0].overflow).toBe(0);
      expect(x.r.counters).toEqual([0, 0, 0, 0]);
    }
    console.log(`[T3-3-boost] accepted boost slots: all-disoccluded ${boostAll}, 20% disoccluded ${boostRnd}, none 0; accepted total ${all.r.arena.rsc.accepted}/${none.r.arena.rsc.accepted}`);
    expect(msgs, msgs.join('\n')).toEqual([]);
    expect(bad).toBe(0);
    expect(boostAll).toBeGreaterThan(100);
    expect(boostRnd).toBeGreaterThan(0);
    expect(boostRnd).toBeLessThan(boostAll);
    expect(all.r.arena.rsc.accepted).toBeGreaterThan(none.r.arena.rsc.accepted);
    rig.destroy();
  });
});

function rng32(seed: number): () => number {
  let a = seed >>> 0;
  return () => { a = (a + 0x6D2B79F5) >>> 0; let x = a; x = Math.imul(x ^ (x >>> 15), x | 1); x ^= x + Math.imul(x ^ (x >>> 7), x | 61); return ((x ^ (x >>> 14)) >>> 0) / 4294967296; };
}
