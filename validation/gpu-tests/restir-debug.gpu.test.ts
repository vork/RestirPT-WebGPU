// ReSTIR debug views and the pixel inspector (WP-D, restir-api.md §2.11, §6.1 U-DBG-1…3; Changelog D1–D4).
//   U-DBG-1  every view id writes the expected code/value at known pixels:
//            (a) hook views 400–410 / 460–470 from a synthetic hook-caller pipeline with known records (known answer,
//                also the file's canary), incl. the stage-tap rule (initial / spatial / final = last writer);
//            (b) shift views 420–446 of rs_debug_views from a synthetic shift arena;
//            (c) real frames: reservoir views ≡ the read-back reservoirs, shift views ≡ the read-back arena, thr ≡ rsGeo.
//   U-DBG-2  views 461 (Σm − 1) and 462 ((lum L − Σw)/Σw) are 0 to 1e-6 on an offline frame.
//   U-DBG-3  the probe dump decodes to the reservoirs read back (bitwise), and the slot records to the arena.
//   Debug off: validation pipelines (no debug group) compile every hook to an empty body; with debug resources the
//   finalize output is bitwise identical with views / probe on and off (the debug-enabled pipelines may differ from
//   the validation pipelines by f32 contraction only, ≤ 1e-5 rel).
// M5 (T-D; restir-temporal-api.md §2.11, §6.1, Changelog D-1):
//   U-TD-1   views 480–497 write the expected codes / values at known pixels: (a) a synthetic T1/T3 hook caller over
//            synthetic tState / sfxOut / res[h] / res[w] records and RsTemporal flags (known answer, the M5 canary), incl.
//            the tap "after temporal" of the reservoir views and phase A skipping the Q_i pixels; (b) view 497 (boost
//            mask) from a synthetic arena; (c) real temporal frames: views ≡ tState / sfxOut / res read back.
//   U-TD-2   view 493 (forward vs stored p̂) = 0 on static identity frames (jitter off, static camera and lights).
//   U-TD-3   probe tags 73–78 decode to the tState / sfxOut words read back (synthetic and real).
// Chrome lane authoritative; dawn.node is a pre-check.
import { afterAll, describe, expect, it } from 'vitest';
import { readBuffer } from '../../src/core/gpu/readback.ts';
import {
  BUILTIN_VIEWS, DEBUG_BUFFER_LAYOUT, DebugResources, DebugViewRegistry, defaultDebugSettings, type DebugSettings,
} from '../../src/core/render/debug-views.ts';
import { JITTER_IID, JITTER_NONE } from '../../src/core/render/frame-uniforms.ts';
import { parseDebugHeader, type ProbeFrame } from '../../src/core/render/probe.ts';
import {
  RESTIR_VIEWS, RS_VIEW_T, RS_VIEW_THR, RestirDebugPass, decodeJWord, decodeRestirProbe, f16ToF32, shiftedPolylines,
} from '../../src/core/render/restir/debug.ts';
import { RestirKernel } from '../../src/core/render/restir/kernel.ts';
import {
  RES_WORDS, RS_VIEW, RS_WGSL_CONSTS as K, RW, TS_CONSTS as TS, TSW, arenaWords, decodeReservoir, decodeSfxLocal, decodeTStateLocal,
  packRsTemporal, pathClass, rfPack, rfUnpack, sfxWord, tsWord,
} from '../../src/core/render/restir/layout.ts';
import { numSlotsOf, restirSettings, type RestirPresetName, type RestirSettings } from '../../src/core/render/restir/presets.ts';
import { RS_PASSES, restirCommonDefines } from '../../src/core/render/restir/resources.ts';
import { composeWgsl } from '../../src/core/gpu/wgsl-composer.ts';
import { shaderSources } from '../../src/core/shaders/index.ts';
import { releaseTestGpu } from './device-factory.ts';
import { allLightsScene, bitFixtureScene, boxCamera, gpuScene, readTexture4, restirRig, storageBuffer, type GpuScene } from './restir-fixtures.ts';

afterAll(releaseTestGpu);

const NONE = 0xFFFFFFFF;
const TAP = { final: 0, initial: 1, spatial: 3 } as const;
const f32 = (u: number) => new Float32Array(Uint32Array.of(u).buffer)[0];
const bits = (x: number) => new Uint32Array(Float32Array.of(x).buffer)[0];
const lum = (F: ArrayLike<number>) => 0.2126 * F[0] + 0.7152 * F[1] + 0.0722 * F[2];
function rng(seed: number): () => number {
  let a = seed >>> 0;
  return () => { a = (a + 0x6D2B79F5) >>> 0; let t = a; t = Math.imul(t ^ (t >>> 15), t | 1); t ^= t + Math.imul(t ^ (t >>> 7), t | 61); return ((t ^ (t >>> 14)) >>> 0) / 4294967296; };
}
/** f32 product as the GPU computes it. */
const fmul = (a: number, b: number) => Math.fround(Math.fround(a) * Math.fround(b));
const relErr = (a: number, b: number) => (a === b ? 0 : Math.abs(a - b) / Math.max(Math.abs(b), 1e-30));

interface Frame { aov: Uint32Array; aovF: Float32Array; probe: ProbeFrame }

interface DebugRig {
  g: GpuScene; kernel: RestirKernel; debug: DebugResources; pass: RestirDebugPass; W: number; H: number;
  accum: GPUBuffer; counters: GPUBuffer;
  /** One frame t with the given debug settings (optionally only the rs_debug_views pass, on the current buffers). */
  frame(t: number, s: Partial<DebugSettings>, o?: { onlyViews?: number; custom?: (enc: GPUCommandEncoder) => void }): Promise<Frame>;
  destroy(): void;
}

async function debugRig(W: number, H: number, preset: RestirPresetName, over: Partial<RestirSettings> = {}): Promise<DebugRig> {
  const g = await gpuScene(bitFixtureScene('x_quads'));
  const device = g.device;
  const debug = new DebugResources(device, new DebugViewRegistry([...BUILTIN_VIEWS, ...RESTIR_VIEWS]));
  await debug.init();
  debug.resize(W, H);
  const kernel = await RestirKernel.create(device, g.gpu, g.env, {
    settings: restirSettings(preset, over), debug, features: g.features, wgslLanguageFeatures: g.wgslLanguageFeatures,
  });
  kernel.setView({ camera: boxCamera(), width: W, height: H, runSeed: 11, jitterMode: JITTER_IID });
  await kernel.prepare();
  const pass = await RestirDebugPass.create(kernel, debug);
  const accum = device.createBuffer({ size: W * H * 16, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
  const counters = device.createBuffer({ size: 16, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
  return {
    g, kernel, debug, pass, W, H, accum, counters,
    async frame(t, s, o = {}) {
      const ds = { ...defaultDebugSettings(), ...s };
      debug.update(ds, t);
      const enc = device.createCommandEncoder({ label: `dbg-frame-${t}` });
      debug.beginFrame(enc);
      kernel.beginSubmit();
      if (o.custom) o.custom(enc);
      else if (o.onlyViews !== undefined) pass.encodeViews(enc, { rounds: o.onlyViews, t });
      else {
        pass.encodeBegin(enc);
        for (const u of kernel.frameUnits(t, { accum, counters })) u.encode(enc);
        pass.encodeViews(enc, { rounds: kernel.lastRounds, t });
      }
      device.queue.submit([enc.finish()]);
      const L = DEBUG_BUFFER_LAYOUT;
      const aovBuf = await readBuffer(device, debug.buffer, W * H * 16, L.aovOffset);
      const head = await readBuffer(device, debug.buffer, L.headerBytes, 0);
      return { aov: new Uint32Array(aovBuf), aovF: new Float32Array(aovBuf), probe: parseDebugHeader(head, t) };
    },
    destroy() { kernel.destroy(); debug.destroy(); accum.destroy(); counters.destroy(); g.destroy(); },
  };
}

// ------------------------------------------------------------------------------------------------ synthetic records

function synthReservoirs(P: number, seed: number): Uint32Array {
  const r = rng(seed);
  const out = new Uint32Array(P * RES_WORDS);
  const f = new Float32Array(out.buffer);
  const eps = [0, 1, 2, 3, 4, 5, 7];
  for (let i = 0; i < P; i++) {
    const o = i * RES_WORDS;
    const d = Math.floor(r() * 16), k = d === 0 ? 0 : Math.floor(r() * (d + 1));
    const flags = rfPack({
      d, k, tech: Math.floor(r() * 4), ep: eps[Math.floor(r() * eps.length)], isDelta: r() < 0.3, lkm1: Math.floor(r() * 6), dkm1: r() < 0.2,
      lk: Math.floor(r() * 6), dk: r() < 0.2, forced: r() < 0.2, bg: r() < 0.1,
    });
    const W = r() < 0.15 ? 0 : 10 ** (4 * r() - 2);
    f[o + RW.W] = W;
    for (let c = 0; c < 3; c++) f[o + RW.F + c] = r() * 3;
    out[o + RW.seed] = i; out[o + RW.seed + 1] = 7;
    out[o + RW.flags] = flags;
    f[o + RW.c] = 1 + Math.floor(r() * 20);
    f[o + RW.jDen] = 1;
    f[o + RW.kMargin] = 8 * r() - 4;
    out[o + RW.endpointId] = NONE;
  }
  return out;
}

/** Expected AOV words of reservoir view `id` for record `i` of `res` (restir-views.wgsl rsdbg_reservoir). */
function expectedReservoirView(id: number, res: Uint32Array, i: number): { code?: number; v?: number[] } {
  const x = decodeReservoir(res, i);
  const none = x.bg || x.d === 0;
  switch (id) {
    case RS_VIEW.c: return { v: [x.c] };
    case RS_VIEW.W: return { v: [x.W] };
    case RS_VIEW.phat: return { v: [lum(x.F)] };
    case RS_VIEW.FW: return { v: x.W === 0 ? [0, 0, 0] : x.F.map((c) => c * x.W) };
    case RS_VIEW.d: return { code: x.bg ? NONE : x.d };
    case RS_VIEW.k: return { code: none ? NONE : x.k };
    case RS_VIEW.tech: return { code: none ? NONE : x.tech };
    case RS_VIEW.endpoint: return { code: none ? NONE : x.ep };
    case RS_VIEW.lobes: return { code: none ? NONE : x.lkm1 * 16 + x.lk + ((x.dkm1 ? 1 : 0) | (x.dk ? 2 : 0)) * 256 };
    case RS_VIEW.class: { const c = pathClass(x); return { code: c < 0 ? NONE : c }; }
    case RS_VIEW.kMargin: return { v: [x.kMargin] };
  }
  throw new Error(`not a reservoir view ${id}`);
}

function checkReservoirView(id: number, fr: Frame, res: Uint32Array, pixels: number[], label: string): number {
  let bad = 0;
  const msgs: string[] = [];
  for (const i of pixels) {
    const e = expectedReservoirView(id, res, i);
    if (e.code !== undefined) {
      if (fr.aov[4 * i] >>> 0 !== e.code >>> 0) { bad++; if (msgs.length < 4) msgs.push(`px ${i}: ${fr.aov[4 * i]} ≠ ${e.code}`); }
    } else {
      for (let c = 0; c < e.v!.length; c++) {
        const got = fr.aovF[4 * i + c];
        if (!(relErr(got, e.v![c]) <= 1e-6 || Math.abs(got - e.v![c]) <= 1e-30)) { bad++; if (msgs.length < 4) msgs.push(`px ${i}.${c}: ${got} ≠ ${e.v![c]}`); }
      }
    }
  }
  expect(bad, `${label} view ${id}: ${msgs.join('; ')}`).toBe(0);
  return pixels.length;
}

const RES_IDS = [RS_VIEW.c, RS_VIEW.W, RS_VIEW.phat, RS_VIEW.FW, RS_VIEW.d, RS_VIEW.k, RS_VIEW.tech, RS_VIEW.endpoint, RS_VIEW.lobes, RS_VIEW.class, RS_VIEW.kMargin];

// A synthetic caller of every hook: record ai of resOut is the pixel's reservoir (tap INITIAL on even ai, SPATIAL on
// odd ai); rsdbg_mis with pixel-dependent values; one candidate / vertex / slot event (probe records only).
const HOOK_CALLER = `
#include "restir/frame.wgsl"
#include "restir/reservoir.wgsl"
#include "debug/restir-views.wgsl"
@compute @workgroup_size(8, 8, 1)
fn t_hooks(@builtin(global_invocation_id) gid: vec3u) {
  let p = rs_pix(vec2u(gid.x, gid.y + rsDispatch.rowBase));
  if (!p.valid) { return; }
  rsdbg_reservoir(p.px, p.ai, select(1u, 3u, (p.ai & 1u) == 1u));
  let x = f32(p.ai);
  rsdbg_mis(p.px, p.ai % 7u, x * 1e-3, x * 1e-7, -x * 1e-7, 0xFFu, x * 1e-3, x, p.ai % 5u);
  for (var s = 0u; s < 6u; s++) { rsdbg_mis(p.px, 0u, 0.0, 0.0, 0.0, s, f32(s) + x * 1e-4, 2.0 + f32(s), 0u); }
  rsdbg_candidate(p.px, 5u, 1u, 3u, 0.25, 0.5, (2u << 20u) | (3u << 12u) | 1u, true);
  rsdbg_vertex(p.px, 9u, 2u, vec3f(1.0, 2.0, 3.0), 0x9u);
  rsdbg_slot(p.px, 4u, 0x3C001208u, 1.5, true);   // O2, term R, pair 1, margin 1.0 (f16)
}`;

describe('U-DBG-1 (a): hook views 400–410 and 460–470 (known answer)', () => {
  it('every hook view writes the expected value; taps initial / spatial / final; probe records decode', async () => {
    const W = 24, H = 16, P = W * H;
    const rig = await debugRig(W, H, 'initial');
    const { device } = rig.g;
    const res = synthReservoirs(P, 7);
    const resBuf = storageBuffer(device, res, 'synth-res');
    const g2l = device.createBindGroupLayout({ entries: [{ binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }] });
    const layout = device.createPipelineLayout({ bindGroupLayouts: [rig.kernel.layouts.g0, rig.kernel.layouts.empty, g2l, rig.kernel.layouts.g3] });
    const pl = await rig.kernel.compile('tests/rsdbg-hooks.wgsl', 't_hooks', { ...restirCommonDefines(undefined, true), RS_RES_OUT_BINDING: '0u' }, layout, 't_hooks',
      { 'tests/rsdbg-hooks.wgsl': HOOK_CALLER });
    const g2 = device.createBindGroup({ layout: g2l, entries: [{ binding: 0, resource: { buffer: resBuf } }] });
    const run = (s: Partial<DebugSettings>) => rig.frame(0, s, {
      custom: (enc) => rig.kernel.encodePass(enc, 'rs_args', pl, g2, { rowBase: 0, rowEnd: H }, rig.kernel.perPixelWorkgroups(0, H)),
    });
    const all = Array.from({ length: P }, (_, i) => i);
    const even = all.filter((i) => i % 2 === 0), odd = all.filter((i) => i % 2 === 1);
    let n = 0;
    for (const id of RES_IDS) {
      n += checkReservoirView(id, await run({ mode: id, tap: TAP.final }), res, all, 'tap final');
    }
    // stage taps: initial = even pixels only, spatial = odd pixels only, the other half stays cleared (0)
    for (const [tap, on, off] of [[TAP.initial, even, odd], [TAP.spatial, odd, even]] as const) {
      const fr = await run({ mode: RS_VIEW.d, tap });
      checkReservoirView(RS_VIEW.d, fr, res, [...on], `tap ${tap}`);
      expect(off.filter((i) => fr.aov[4 * i] !== 0).length, `tap ${tap}: untapped pixels written`).toBe(0);
    }
    // MIS views
    const want: Record<number, (i: number) => number> = {
      [RS_VIEW.misMc]: (i) => fmul(i, 1e-3), [RS_VIEW.misSumM]: (i) => fmul(i, 1e-7), [RS_VIEW.misLumL]: (i) => fmul(-i, 1e-7),
    };
    for (const [id, fn] of Object.entries(want)) {
      const fr = await run({ mode: Number(id) });
      expect(all.filter((i) => relErr(fr.aovF[4 * i], fn(i)) > 1e-6).length, `view ${id}`).toBe(0);
      n++;
    }
    for (let s = 0; s < 6; s++) {
      const fr = await run({ mode: RS_VIEW.misMj + s });
      expect(all.filter((i) => relErr(fr.aovF[4 * i], Math.fround(s + i * 1e-4)) > 1e-6).length, `view m_j[${s}]`).toBe(0);
    }
    for (const [id, fn] of [[RS_VIEW.misK, (i: number) => i % 7], [RS_VIEW.misSel, (i: number) => i % 5]] as const) {
      const fr = await run({ mode: id });
      expect(all.filter((i) => fr.aov[4 * i] !== fn(i)).length, `view ${id}`).toBe(0);
    }
    // probe: one reservoir block (header + 10 planes) whatever the tap, MIS, candidate, vertex, slot event
    const probePx: [number, number] = [5, 3];
    const ai = probePx[1] * W + probePx[0];
    const fr = await run({ mode: 0, probeEnabled: true, probePixel: probePx, tap: TAP.initial });
    const d = decodeRestirProbe(fr.probe.records);
    expect(d.reservoirs.length).toBe(1);
    expect(d.reservoirs[0].ai).toBe(ai);
    expect(d.reservoirs[0].tap).toBe(ai % 2 === 1 ? TAP.spatial : TAP.initial);
    expect(Array.from(d.reservoirs[0].words)).toEqual(Array.from(res.subarray(ai * RES_WORDS, (ai + 1) * RES_WORDS)));
    expect(d.mis.canonical).toEqual({ k: ai % 7, sel: ai % 5, mc: fmul(ai, 1e-3), sumM: fmul(ai, 1e-7), lumRel: fmul(-ai, 1e-7), wc: ai });
    expect(d.mis.partners.map((p) => p.s)).toEqual([0, 1, 2, 3, 4, 5]);
    expect(d.mis.partners[3].w).toBe(5);
    expect(d.candidates).toEqual([{ d: 5, tech: 1, k: 3, selected: true, w: 0.25, lumF: 0.5, counter: (2 << 20) | (3 << 12) | 1, tree: 2, B: 3, slotInVertex: 1 }]);
    expect(d.vertices).toEqual([{ path: 9, b: 2, lobe: 1, delta: true, pos: [1, 2, 3] }]);
    expect(d.slotEvents.length).toBe(1);
    expect(d.slotEvents[0]).toMatchObject({ s: 4, J: 1.5, replayed: true, code: { sc: 8, term: 2, pair: 1 } });
    expect(fr.probe.counters[1], 'probe overflow').toBe(0);
    console.log(`[U-DBG-1a] ${n} view checks over ${P} px; probe ${fr.probe.records.length} records`);
    resBuf.destroy();
    rig.destroy();
  });
});

// ------------------------------------------------------------------------------------------------ U-DBG-1 (b)

describe('U-DBG-1 (b): shift views 420–446 from a synthetic arena (rs_debug_views)', () => {
  it('codes, log2 J, term|pair, margin, replay / accept masks, thr; probe slot records', async () => {
    const W = 24, H = 16, P = W * H, NS = 4;
    const rig = await debugRig(W, H, 'initial', { slots: NS });
    const { device } = rig.g;
    await rig.frame(0, {});           // rs_primary fills rsGeo (thr)
    const res = synthReservoirs(P, 11);
    const r = rng(5);
    const aw = arenaWords(P, NS);
    const words = new Uint32Array(5 * P * NS + P * NS);
    const scs = [K.SC_OK, K.SC_NOT_ACCEPTED, K.SC_EMPTY_SRC, K.SC_O0_MISS, K.SC_O1, K.SC_O2, K.SC_OCCLUDED, K.SC_J_INVALID, K.SC_O0_SUPPORT];
    const marginHalves = [0x0000, 0x3C00, 0xBC00, 0x4200, 0xC500];   // 0, 1, −1, 3, −5 as f16
    for (let ai = 0; ai < P; ai++) for (let s = 0; s < NS; s++) {
      const sc = scs[Math.floor(r() * scs.length)];
      const term = sc === K.SC_O2 ? 1 + Math.floor(r() * 5) : 0, pair = sc === K.SC_O1 ? 2 + Math.floor(r() * 3) : 0;
      const mh = marginHalves[Math.floor(r() * marginHalves.length)];
      const code = (sc | (term << 8) | (pair << 12) | (mh << 16)) >>> 0;
      const J = 2 ** (8 * r() - 4);
      const Jw = sc === K.SC_OK ? bits(J) : sc === K.SC_NOT_ACCEPTED ? K.JW_NOT_ACCEPTED : K.JW_FAILED;
      const w = aw.slots + 4 * (ai * NS + s);
      words[w] = bits(0.5); words[w + 1] = bits(0.25); words[w + 2] = bits(0.125); words[w + 3] = Jw;
      words[aw.codes + ai * NS + s] = code;
    }
    device.queue.writeBuffer(rig.kernel.resources.res[0], 0, res);
    device.queue.writeBuffer(rig.kernel.resources.arena, 0, new Uint32Array(64));
    device.queue.writeBuffer(rig.kernel.resources.arena, 256, words);
    const geo = new Float32Array((await readTexture4(device, rig.kernel.resources.geo)).buffer);
    const all = Array.from({ length: P }, (_, i) => i);
    const code = (ai: number, s: number) => words[aw.codes + ai * NS + s];
    const jw = (ai: number, s: number) => words[aw.slots + 4 * (ai * NS + s) + 3];
    const needsReplay = (ai: number) => { const f = rfUnpack(res[ai * RES_WORDS + RW.flags]); return f.d !== 0 && (f.k > 2 || f.k === 0); };
    const bad: string[] = [];
    const cmp = (id: number, want: (ai: number) => number, isCode: boolean) => async () => {
      const fr = await rig.frame(0, { mode: id }, { onlyViews: 1 });
      for (const ai of all) {
        const got = isCode ? fr.aov[4 * ai] >>> 0 : fr.aovF[4 * ai];
        const w = want(ai);
        if (isCode ? got !== w >>> 0 : relErr(got, w) > 1e-6 && Math.abs(got - w) > 1e-30) { bad.push(`view ${id} px ${ai}: ${got} ≠ ${w}`); break; }
      }
    };
    for (let s = 0; s < 6; s++) {
      await cmp(RS_VIEW.shiftCode + s, (ai) => (s < NS ? code(ai, s) & 0xFF : NONE), true)();
      await cmp(RS_VIEW.shiftTerm + s, (ai) => (s < NS ? ((code(ai, s) >>> 8) & 0xF) | (((code(ai, s) >>> 12) & 0xF) << 4) : NONE), true)();
      if (s < NS) {
        await cmp(RS_VIEW.shiftLogJ + s, (ai) => ((code(ai, s) & 0xFF) === K.SC_OK ? Math.log2(f32(jw(ai, s))) : 0), false)();
        await cmp(RS_VIEW.shiftMargin + s, (ai) => f16ToF32(code(ai, s) >>> 16), false)();
      }
    }
    const acc = (ai: number) => { let m = 0; for (let s = 0; s < NS; s++) if (jw(ai, s) >>> 0 !== K.JW_NOT_ACCEPTED) m |= 1 << s; return m; };
    const rep = (ai: number) => { let m = 0; for (let s = 0; s < NS; s++) if (jw(ai, s) >>> 0 !== K.JW_NOT_ACCEPTED && (code(ai, s) & 0xFF) !== K.SC_EMPTY_SRC && needsReplay(ai)) m |= 1 << s; return m; };
    await cmp(RS_VIEW.acceptMask, acc, true)();
    await cmp(RS_VIEW.replayMask, rep, true)();
    await cmp(RS_VIEW_THR, (ai) => geo[4 * ai + 3], false)();
    expect(bad, bad.join('\n')).toEqual([]);
    // rounds = 0 (no spatial stage this frame): shift views stay untouched (fill = NONE for code views)
    const fr0 = await rig.frame(0, { mode: RS_VIEW.shiftCode }, { onlyViews: 0 });
    expect(all.filter((i) => fr0.aov[4 * i] !== 0).length, 'rounds 0 writes nothing').toBe(0);
    // probe: tag 71 per slot mirrors the arena; tag 72 (incoming) whenever the pairing module gives a partner
    const px: [number, number] = [7, 9];
    const ai = px[1] * W + px[0];
    const fp = await rig.frame(0, { mode: 0, probeEnabled: true, probePixel: px }, { onlyViews: 1 });
    const d = decodeRestirProbe(fp.probe.records);
    expect(d.slots.map((s) => s.s)).toEqual([0, 1, 2, 3]);
    for (const s of d.slots) {
      expect(s.code.sc).toBe(code(ai, s.s) & 0xFF);
      expect(s.j).toEqual(decodeJWord(jw(ai, s.s)));
      expect(s.queued).toBe(((rep(ai) >> s.s) & 1) === 1);
      if (s.partnerAi !== undefined) expect(s.incoming?.code.sc).toBe(code(s.partnerAi, s.s) & 0xFF);
    }
    console.log(`[U-DBG-1b] shift views ok on ${P} px × ${NS} slots; probe partners ${d.slots.filter((s) => s.partnerAi !== undefined).length}/${NS} (0 here: this kernel never uploaded pairing maps)`);
    rig.destroy();
  });
});

// ------------------------------------------------------------------------------------------------ U-DBG-1 (c), U-DBG-2, U-DBG-3

describe('U-DBG-1 (c) / U-DBG-3: real frames', () => {
  const W = 48, H = 32, P = W * H, t = 3;
  const all = Array.from({ length: P }, (_, i) => i);

  it('initial tap: reservoir views ≡ read-back res[0]; thr ≡ rsGeo; shift views ≡ arena; probe dump ≡ res[0]', async () => {
    const rig = await debugRig(W, H, 'interactive', { maxBounces: 3 });
    const views = async (id: number, tap: number) => rig.frame(t, { mode: id, tap });
    await rig.frame(t, {});          // the reservoir buffers are identical in every frame t (views never change them)
    const res0 = await rig.kernel.readReservoirs(0);
    const rounds = rig.kernel.lastRounds;
    let nonEmpty = 0;
    for (let i = 0; i < P; i++) if (rfUnpack(res0[i * RES_WORDS + RW.flags]).d !== 0) nonEmpty++;
    for (const id of RES_IDS) checkReservoirView(id, await views(id, TAP.initial), res0, all, 'real initial');
    if (rounds === 0) for (const id of RES_IDS) checkReservoirView(id, await views(id, TAP.final), res0, all, 'real final (no spatial)');
    const geo = new Float32Array((await readTexture4(rig.g.device, rig.kernel.resources.geo)).buffer);
    const thr = await views(RS_VIEW_THR, 0);
    expect(all.filter((ai) => thr.aovF[4 * ai] !== geo[4 * ai + 3]).length, 'thr view').toBe(0);
    let accepted = 0;
    if (rounds > 0) {
      const NS = numSlotsOf(rig.kernel.settings);            // M5: + boost slots (interactive preset)
      const arena = new Uint32Array(await readBuffer(rig.g.device, rig.kernel.resources.arena, 256 + 24 * P * NS));
      const aw = arenaWords(P, NS);
      for (let s = 0; s < NS; s++) {
        const fr = await views(RS_VIEW.shiftCode + s, 0);
        expect(all.filter((ai) => fr.aov[4 * ai] !== (arena[64 + aw.codes + ai * NS + s] & 0xFF)).length, `shift code slot ${s}`).toBe(0);
        accepted += all.filter((ai) => arena[64 + aw.slots + 4 * (ai * NS + s) + 3] >>> 0 !== K.JW_NOT_ACCEPTED).length;
      }
    }
    // U-DBG-3: probe at a hit pixel near the centre
    const px: [number, number] = [W >> 1, H >> 1];
    const ai = px[1] * W + px[0];
    const fp = await rig.frame(t, { mode: 0, probeEnabled: true, probePixel: px });
    const d = decodeRestirProbe(fp.probe.records);
    const init = d.reservoirs.find((r) => r.tap === TAP.initial);
    expect(init, 'initial reservoir record').toBeDefined();
    expect(Array.from(init!.words)).toEqual(Array.from(res0.subarray(ai * RES_WORDS, (ai + 1) * RES_WORDS)));
    if (d.candidates.length && init!.rec.d > 0) expect(d.candidates.some((c) => c.selected && c.counter === init!.rec.selId), 'selected candidate recorded').toBe(true);
    if (rounds > 0) {
      expect(d.slots.length, 'slot records of the probe pixel').toBe(numSlotsOf(rig.kernel.settings));
      // every slot with a partner has its incoming record; accepted slots (both pixels hit, A0) have the shifted-path
      // anchors (the partner's / own primary hit)
      for (const s of d.slots.filter((x) => x.partnerAi !== undefined)) expect(s.incoming, `slot ${s.s} incoming`).toBeDefined();
      for (const s of d.slots.filter((x) => x.partnerAi !== undefined && x.j.status !== 'NOT_ACCEPTED')) {
        expect(d.paths.get(16 + s.s)?.[0]?.b, `slot ${s.s} p→partner anchor`).toBe(1);
        expect(d.paths.get(24 + s.s)?.[0]?.b, `slot ${s.s} partner→p anchor`).toBe(1);
      }
      expect(shiftedPolylines(d, [0, 0, 0]).length, 'overlay polylines').toBeGreaterThan(0);
    }
    expect(fp.probe.counters[1], 'probe overflow').toBe(0);
    console.log(`[U-DBG-1c/3] rounds ${rounds}, non-empty reservoirs ${nonEmpty}/${P}, accepted slots ${accepted}; probe: ${d.reservoirs.length} reservoirs, `
      + `${d.candidates.length} candidates, ${d.vertices.length} vertices in ${d.paths.size} paths, ${d.slots.length} slots `
      + `(${d.slots.filter((x) => x.partnerAi !== undefined).length} with partner), ${d.slotEvents.length} slot events, MIS ${d.mis.canonical ? `k ${d.mis.canonical.k}` : 'none'}`);
    rig.destroy();
  });

  it('spatial / final taps: the final-round resample reports its reservoir (rsdbg_reservoir, DBG_TAP_SPATIAL)', async () => {
    const rig = await debugRig(W, H, 'interactive', { maxBounces: 3 });
    await rig.frame(t, {});
    const rounds = rig.kernel.lastRounds;
    if (rounds === 0) { console.log('[U-DBG-1c spatial] no spatial stage ran (stub): nothing to check'); rig.destroy(); return; }
    const resF = await rig.kernel.readReservoirs('final');
    const probe = await rig.frame(t, { mode: 0, probeEnabled: true, probePixel: [W >> 1, H >> 1] });
    const spat = decodeRestirProbe(probe.probe.records).reservoirs.find((r) => r.tap === TAP.spatial);
    expect(spat, 'rs_spatial_resample must call rsdbg_reservoir(px, ai, DBG_TAP_SPATIAL) in the final round (restir-api.md §2.11)').toBeDefined();
    const ai = (H >> 1) * W + (W >> 1);
    expect(Array.from(spat!.words)).toEqual(Array.from(resF.subarray(ai * RES_WORDS, (ai + 1) * RES_WORDS)));
    for (const id of RES_IDS) {
      checkReservoirView(id, await rig.frame(t, { mode: id, tap: TAP.spatial }), resF, all, 'real spatial');
      checkReservoirView(id, await rig.frame(t, { mode: id, tap: TAP.final }), resF, all, 'real final');
    }
    rig.destroy();
  });

  it('U-DBG-2: Σm − 1 (461) and (lum L − Σw)/Σw (462) are 0 to 1e-6 on an offline frame', async () => {
    const rig = await debugRig(W, H, 'offline', { maxBounces: 3, trees: 4 });
    const worst: Record<number, number> = {};
    for (const id of [RS_VIEW.misSumM, RS_VIEW.misLumL]) {
      const fr = await rig.frame(1, { mode: id });
      let m = 0;
      for (let i = 0; i < P; i++) m = Math.max(m, Math.abs(fr.aovF[4 * i]));
      worst[id] = m;
      expect(m, `view ${id}`).toBeLessThanOrEqual(1e-6);
    }
    const kv = await rig.frame(1, { mode: RS_VIEW.misK });
    let withPartners = 0;
    for (let i = 0; i < P; i++) if (kv.aov[4 * i] > 0 && kv.aov[4 * i] !== NONE) withPartners++;
    console.log(`[U-DBG-2] max |461| ${worst[RS_VIEW.misSumM]}, max |462| ${worst[RS_VIEW.misLumL]}; pixels with k > 0: ${withPartners}/${P}`
      + `${withPartners === 0 ? ' (VACUOUS: no spatial MIS records yet)' : ''}; rounds ${rig.kernel.lastRounds}`);
    rig.destroy();
  });
});

describe('arena HUD counters on an env + analytic-light scene (the app smoke\'s HDRI case)', () => {
  it('the spatial stage runs and counts (accepted > 0, SC histogram non-empty) with env NEE and analytic lights', async () => {
    const W = 48, H = 32;
    const rig = await restirRig(allLightsScene(), W, H, { preset: 'interactive', settings: { maxBounces: 3 } });
    const r = await rig.frames(2, 0);
    const hist = r.arena.codes.reduce((a, b) => a + b, 0);
    console.log(`[env+lights] rounds ${rig.kernel.lastRounds} accepted ${r.arena.rsc.accepted} queued ${r.arena.rsc.queued} SC total ${hist} finalize ${r.counters}`);
    expect(r.arena.rsc.accepted, 'accepted slots').toBeGreaterThan(0);
    expect(hist, 'SC histogram').toBeGreaterThan(0);
    rig.destroy();
  });
});

describe('debug off: validation unaffected', () => {
  it('validation pipelines (no G3) compile every hook to an empty body', async () => {
    const rig = await restirRig(bitFixtureScene('c0c'), 16, 16, { preset: 'initial' });
    const names = ['rs_primary', 'rs_initial', 'rs_pair_accept', 'rs_spatial_replay', 'rs_spatial_shift', 'rs_spatial_resample', 'rs_finalize'] as const;
    const hooks = ['rsdbg_reservoir', 'rsdbg_candidate', 'rsdbg_vertex', 'rsdbg_slot', 'rsdbg_accept', 'rsdbg_mis'];
    const bad: string[] = [];
    for (const n of names) {
      const code = composeWgsl(RS_PASSES[n].file, { sources: shaderSources, defines: rig.kernel.defines(n) }).code;
      for (const h of hooks) {
        const m = new RegExp(`fn ${h}\\([^)]*\\)\\s*\\{([^}]*)\\}`).exec(code);
        if (m && m[1].replace(/\/\/.*$/gm, '').trim() !== '') bad.push(`${n}: ${h} has a body`);
      }
    }
    rig.destroy();
    expect(bad).toEqual([]);
  });

  it('finalize output: views / probe on ≡ off (same pipelines, bitwise); debug pipelines ≈ validation pipelines', async () => {
    const W2 = 32, H2 = 24, n = 3;
    const ref = await restirRig(bitFixtureScene('x_quads'), W2, H2, { preset: 'interactive', settings: { maxBounces: 3 } });
    const a = await ref.frames(n, 0);
    ref.destroy();
    const rig = await debugRig(W2, H2, 'interactive', { maxBounces: 3 });
    const run = async (s: Partial<DebugSettings>) => {
      const device = rig.g.device;
      const clear = device.createCommandEncoder();
      clear.clearBuffer(rig.accum); clear.clearBuffer(rig.counters);
      device.queue.submit([clear.finish()]);
      for (let f = 0; f < n; f++) await rig.frame(f, s);
      return new Float32Array(await readBuffer(device, rig.accum, W2 * H2 * 16));
    };
    const off = await run({ mode: 0 });
    const on = await run({ mode: RS_VIEW.W, tap: TAP.initial, probeEnabled: true, probePixel: [W2 >> 1, H2 >> 1] });
    const onShift = await run({ mode: RS_VIEW.shiftCode, probeEnabled: true, probePixel: [3, 4] });
    const onMis = await run({ mode: RS_VIEW.misMc, probeEnabled: true, probePixel: [10, 5] });
    const diff = (x: Float32Array, y: Float32Array) => { let k = 0; for (let i = 0; i < x.length; i++) if (bits(x[i]) !== bits(y[i])) k++; return k; };
    expect(diff(on, off), 'reservoir view + probe on ≡ off').toBe(0);
    expect(diff(onShift, off), 'shift view + probe on ≡ off').toBe(0);
    expect(diff(onMis, off), 'MIS view + probe on ≡ off').toBe(0);
    // A kernel with debug resources compiles other pipelines (hooks present): Metal may contract FMAs differently, so
    // the image may differ from the validation kernel's by f32 rounding only (validation never binds debug resources).
    const m = new Float32Array(W2 * H2 * 3);
    for (let i = 0; i < W2 * H2; i++) for (let c = 0; c < 3; c++) m[3 * i + c] = off[4 * i + c] / n;
    let words = 0, mx = 0;
    for (let i = 0; i < m.length; i++) if (bits(m[i]) !== bits(a.mean[i])) { words++; mx = Math.max(mx, relErr(m[i], a.mean[i])); }
    console.log(`[debug-off] debug-kernel vs validation kernel: ${words}/${m.length} words differ, max rel ${mx}`);
    expect(mx, 'debug pipelines vs validation pipelines').toBeLessThanOrEqual(1e-5);
    rig.destroy();
  });
});

// ================================================================================================ M5 (T-D)

const SCS = [K.SC_OK, K.SC_EMPTY_SRC, K.SC_O0_MISS, K.SC_O0_LIGHT, K.SC_O1, K.SC_O2, K.SC_OCCLUDED, K.SC_ZERO, K.SC_PENDING];
const TF_SYNTH = K.TF_HIST_VALID | K.TF_REFRESH | K.TF_LIGHTS_SAME | K.TF_ENV_MOVED;
const GEN = 5;
const jwValid = (w: number) => w >>> 0 !== K.JW_FAILED && w >>> 0 < 0x7f800000;
const rel = (a: number, b: number) => { const m = Math.max(Math.abs(a), Math.abs(b)); return m > 0 ? (a - b) / m : 0; };
const log2pos = (x: number) => (x > 0 && x < 3e38 ? Math.log2(x) : 0);

/** Mirrors of restir-views.wgsl (view codes of §2.11). */
function qvalidCode(f: number): number {
  if (f & TS.TS_BG) return 0;
  if (f & TS.TS_QVALID) return f & TS.TS_PICK_RING ? 2 : 1;
  return f & TS.TS_NO_HIST ? 4 : 3;
}
function refreshClass(tf: number, st: number, gen: number, frameGen = GEN): number {
  if (!(tf & K.TF_REFRESH) || gen !== frameGen || !(st & TS.SXS_DONE)) return 0;
  if (st & TS.SXS_E2) return 4;
  if (st & TS.SXS_UNDEF) return 3;
  if (st & TS.SXS_ZERO) return 5;
  return st & TS.SXS_RAY ? 2 : 1;
}
function selCode(f: number): number {
  if (!(f & TS.TS_QVALID)) return 0;
  if (f & TS.TS_EMPTY_OUT) return 3;
  return f & TS.TS_SEL_P ? 2 : 1;
}

interface TSynth { words: Uint32Array; resIn: Uint32Array; resOut: Uint32Array; base: number }

/** Synthetic tState + sfxOut (arena words[] from tState on) and history / current reservoirs for P pixels. */
function synthTemporal(P: number, NS: number, seed: number): TSynth {
  const r = rng(seed);
  const aw = arenaWords(P, NS);
  const words = new Uint32Array(aw.end - aw.tState);
  const f = new Float32Array(words.buffer);
  const tw = (ai: number, w: number) => tsWord(P, NS, ai, w) - aw.tState;
  const sw = (d: number, ai: number, w: number) => sfxWord(P, NS, d, ai, w) - aw.tState;
  const statuses = [0, TS.SXS_DONE, TS.SXS_DONE | TS.SXS_RAY | TS.SXS_DEEP, TS.SXS_DONE | TS.SXS_UNDEF, TS.SXS_DONE | TS.SXS_UNDEF | TS.SXS_E2,
    TS.SXS_DONE | TS.SXS_ZERO | TS.SXS_DEEP, TS.SXS_DONE | TS.SXS_N1 | TS.SXS_VIS | TS.SXS_RAY, TS.SXS_DONE | TS.SXS_B1];
  const qvalidVariants = [
    TS.TS_SEL_P | TS.TS_FINAL, TS.TS_EMPTY_OUT | TS.TS_FINAL, TS.TS_SEL_C | TS.TS_INV_QUEUED, TS.TS_SEL_P | TS.TS_INV_QUEUED | TS.TS_ROBUST,
    TS.TS_INV_QUEUED, TS.TS_SEL_P | TS.TS_FINAL | TS.TS_FWD_QUEUED,
  ];
  for (let ai = 0; ai < P; ai++) {
    const cat = Math.floor(r() * 6);
    let flags = 0, qP = 0xFFFFFFFF;
    if (cat === 0) flags = TS.TS_BG;
    else if (cat === 1) flags = TS.TS_DISOCC | TS.TS_NO_HIST;
    else if (cat === 2) flags = TS.TS_DISOCC;
    else {
      flags = TS.TS_QVALID | (r() < 0.3 ? TS.TS_PICK_RING : 0) | qvalidVariants[Math.floor(r() * qvalidVariants.length)];
      qP = Math.floor(r() * P);
    }
    words[tw(ai, TSW.flags)] = flags;
    words[tw(ai, TSW.qPrime)] = qP;
    for (let c = 0; c < 3; c++) { f[tw(ai, TSW.fwdF + c)] = r() * 2; f[tw(ai, TSW.invF + c)] = r() * 2; }
    const jc = r();
    words[tw(ai, TSW.fwdJ)] = jc < 0.6 ? bits(2 ** (6 * r() - 3)) : jc < 0.8 ? K.JW_FAILED : K.JW_PENDING;
    words[tw(ai, TSW.invJ)] = r() < 0.7 ? bits(2 ** (6 * r() - 3)) : K.JW_FAILED;
    const cPrev = 1 + Math.floor(r() * 60);
    f[tw(ai, TSW.cPrev)] = cPrev; f[tw(ai, TSW.cP)] = Math.min(20, cPrev);
    words[tw(ai, TSW.fwdCode)] = (SCS[Math.floor(r() * SCS.length)] | (Math.floor(r() * 5) << 8)) >>> 0;
    words[tw(ai, TSW.invCode)] = (SCS[Math.floor(r() * SCS.length)] | (Math.floor(r() * 5) << 12)) >>> 0;
    f[tw(ai, TSW.jP)] = r() < 0.1 ? 0 : 2 ** (2 * r() - 1);
    const zeroW = r() < 0.1;
    f[tw(ai, TSW.wc)] = zeroW ? 0 : r() * 3; f[tw(ai, TSW.wp)] = zeroW ? 0 : r() * 3;
    f[tw(ai, TSW.piStored)] = r() * 4; f[tw(ai, TSW.piRecomp)] = r() < 0.3 ? f[tw(ai, TSW.piStored)] * (1 + 1e-4 * (r() - 0.5)) : r() * 4;
    words[tw(ai, TSW.xpEntry)] = Math.floor(r() * 4);
    for (const d of [0, 1]) {
      for (let c = 0; c < 3; c++) f[sw(d, ai, c)] = r();
      f[sw(d, ai, 3)] = r();
      words[sw(d, ai, 4)] = statuses[Math.floor(r() * statuses.length)];
      const e = r();
      words[sw(d, ai, 5)] = e < 0.3 ? 0xFFFFFFFF : e < 0.6 ? K.RC_NONE : 0;
      f[sw(d, ai, 6)] = 2 ** (r() - 0.5);
      words[sw(d, ai, 7)] = r() < 0.8 ? GEN : GEN - 1;
    }
  }
  return { words, resIn: synthReservoirs(P, seed + 1), resOut: synthReservoirs(P, seed + 2), base: aw.tState };
}

/** Expected value of temporal view `id` at pixel ai (mirror of rsdbg_temporal / rsdbg_tpick). */
function expectedTemporalView(id: number, x: TSynth, P: number, NS: number, ai: number, tf: number,
  frameGen = GEN, lcb: (entryCur: number) => number = () => 0): { code?: number; v?: number[] } {
  const t = decodeTStateLocal(x.words, P, NS, ai);
  const qvalid = (t.flags & TS.TS_QVALID) !== 0, invq = (t.flags & TS.TS_INV_QUEUED) !== 0;
  const fwdOk = qvalid && jwValid(t.fwdJ);
  const out = decodeReservoir(x.resOut, ai);
  const qp = Math.min(t.qPrime, P - 1);
  const hist = decodeReservoir(x.resIn, qp);
  switch (id) {
    case RS_VIEW_T.qvalid: return { code: qvalidCode(t.flags) };
    case RS_VIEW_T.refreshFwd: { const s = decodeSfxLocal(x.words, P, NS, 0, qp); return { code: qvalid ? refreshClass(tf, s.status, s.gen, frameGen) : 0 }; }
    case RS_VIEW_T.refreshInv: { const s = decodeSfxLocal(x.words, P, NS, 1, ai); return { code: invq ? refreshClass(tf, s.status, s.gen, frameGen) : 0 }; }
    case RS_VIEW_T.fwdCode: return { code: qvalid ? t.fwdCode & 0xFF : NONE };
    case RS_VIEW_T.invCode: return { code: invq ? t.invCode & 0xFF : NONE };
    case RS_VIEW_T.logJ: return { v: [fwdOk ? log2pos(f32(t.fwdJ)) : 0] };
    case RS_VIEW_T.logJP: return { v: [fwdOk ? log2pos(t.jP) : 0] };
    case RS_VIEW_T.pic: return { v: [lum(out.F)] };
    case RS_VIEW_T.pip: return { v: [t.flags & TS.TS_SEL_P ? t.piStored : invq ? t.piRecomp : 0] };
    case RS_VIEW_T.cprev: return { v: [qvalid ? t.cPrev : 0] };
    case RS_VIEW_T.cout: return { v: [out.c] };
    case RS_VIEW_T.sel: return { code: selCode(t.flags) };
    case RS_VIEW_T.phatRel: return { v: [fwdOk ? rel(lum(t.fwdF), lum(hist.F)) : 0] };
    case RS_VIEW_T.robust: return { v: [t.flags & TS.TS_ROBUST ? rel(t.piRecomp, t.piStored) : 0] };
    case RS_VIEW_T.wp: return { v: [t.wc + t.wp > 0 ? t.wp / (t.wc + t.wp) : 0] };
    case RS_VIEW_T.lightsChanged: {
      if (!qvalid || !(tf & K.TF_REFRESH) || hist.d === 0) return { code: 0 };
      if (hist.tech === K.RS_TECH_BSDF_ENV) return { code: (tf & K.TF_ENV_MOVED ? 1 : 0) | (tf & K.TF_ENV_RADIO ? 2 : 0) };
      if (hist.tech !== K.RS_TECH_NEE) return { code: 0 };
      const s = decodeSfxLocal(x.words, P, NS, 0, qp);
      if (s.gen !== frameGen || s.status & TS.SXS_UNDEF || s.entryTo === 0xFFFFFFFF) return { code: 4 };
      const b = lcb(s.entryTo);
      return { code: (b & K.LCB_MOVED ? 1 : 0) | (b & K.LCB_RADIO ? 2 : 0) };     // synthetic: TF_LIGHTS_SAME ⇒ 0
    }
  }
  throw new Error(`not a temporal view ${id}`);
}

// Synthetic T1 / T3 caller: rsdbg_temporal phase A for every pixel (skips the Q_i pixels), phase B for the Q_i pixels;
// rsdbg_tpick with s′ = local + (0.25·(ai mod 5), −0.5), no projection (−1, −1) for ai mod 7 = 0.
const THOOK_CALLER = `
#include "restir/frame.wgsl"
#include "restir/reservoir.wgsl"
#include "restir/tframe.wgsl"
#include "debug/restir-views.wgsl"
@compute @workgroup_size(8, 8, 1)
fn t_thooks(@builtin(global_invocation_id) gid: vec3u) {
  let p = rs_pix(vec2u(gid.x, gid.y + rsDispatch.rowBase));
  if (!p.valid) { return; }
  let sp = select(vec2f(p.local) + vec2f(0.25 * f32(p.ai % 5u), -0.5), vec2f(-1.0), p.ai % 7u == 0u);
  rsdbg_tpick(p.px, sp, p.ai % 10u, (p.ai & 1u) == 1u);
  rsdbg_temporal(p.px, p.ai, 0u);
  if ((ts_load(p.ai, TSW_FLAGS) & TS_INV_QUEUED) != 0u) { rsdbg_temporal(p.px, p.ai, 1u); }
}`;

describe('U-TD-1 (a) / U-TD-3: temporal views 480–496 and probe tags 73–78 (synthetic, known answer)', () => {
  it('every temporal view writes the expected value; tap "after temporal"; phase A skips Q_i pixels; probe decodes', async () => {
    const W = 24, H = 16, P = W * H;
    const rig = await debugRig(W, H, 'temporal');
    const { device } = rig.g;
    const NS = rig.kernel.resources.alloc.slots;
    const x = synthTemporal(P, NS, 21);
    device.queue.writeBuffer(rig.kernel.resources.arena, 256 + 4 * x.base, x.words);
    device.queue.writeBuffer(rig.kernel.rsTemporal, 0, packRsTemporal({
      flags: TF_SYNTH, histFrames: 3, frameGen: GEN, prevGen: GEN - 1, envPrev: new Uint32Array(8), gens: [1, 2, 3, 4], gensPrev: [0, 1, 2, 3], configHash: 7,
    }));
    const inBuf = storageBuffer(device, x.resIn, 'synth-res-h'), outBuf = storageBuffer(device, x.resOut, 'synth-res-w');
    const c = GPUShaderStage.COMPUTE;
    const g2l = device.createBindGroupLayout({ entries: [0, 1, 2].map((binding) => ({ binding, visibility: c, buffer: { type: binding === 0 ? 'read-only-storage' as const : 'storage' as const } })) });
    const layout = device.createPipelineLayout({ bindGroupLayouts: [rig.kernel.layouts.g0, rig.kernel.layouts.empty, g2l, rig.kernel.layouts.g3] });
    const pl = await rig.kernel.compile('tests/rsdbg-thooks.wgsl', 't_thooks', {
      ...restirCommonDefines(undefined, true), RS_TEMPORAL: 1, RS_RES_IN_BINDING: '0u', RS_RES_OUT_BINDING: '1u', RS_ARENA_BINDING: '2u', RS_ARENA_RW: true,
    }, layout, 't_thooks', { 'tests/rsdbg-thooks.wgsl': THOOK_CALLER });
    const g2 = device.createBindGroup({ layout: g2l, entries: [{ binding: 0, resource: { buffer: inBuf } }, { binding: 1, resource: { buffer: outBuf } }, { binding: 2, resource: { buffer: rig.kernel.resources.arena } }] });
    const run = (s: Partial<DebugSettings>) => rig.frame(0, s, {
      custom: (enc) => rig.kernel.encodePass(enc, 'rs_args', pl, g2, { rowBase: 0, rowEnd: H }, rig.kernel.perPixelWorkgroups(0, H)),
    });
    const all = Array.from({ length: P }, (_, i) => i);
    const bad: string[] = [];
    for (let id: number = RS_VIEW_T.qvalid; id <= RS_VIEW_T.lightsChanged; id++) {
      if (id === RS_VIEW_T.motion) continue;
      const fr = await run({ mode: id });
      let nb = 0;
      for (const ai of all) {
        const e = expectedTemporalView(id, x, P, NS, ai, TF_SYNTH);
        if (e.code !== undefined) { if (fr.aov[4 * ai] >>> 0 !== e.code >>> 0) { nb++; if (nb < 3) bad.push(`view ${id} px ${ai}: ${fr.aov[4 * ai]} ≠ ${e.code}`); } continue; }
        const got = fr.aovF[4 * ai], w = e.v![0];
        const absTol = id === RS_VIEW_T.phatRel || id === RS_VIEW_T.robust ? 1e-6 : id === RS_VIEW_T.logJ || id === RS_VIEW_T.logJP ? 1e-5 : 0;
        if (!(relErr(got, w) <= 1e-6 || Math.abs(got - w) <= Math.max(absTol, 1e-30))) { nb++; if (nb < 3) bad.push(`view ${id} px ${ai}: ${got} ≠ ${w}`); }
      }
    }
    // 481 motion (s′ − q)
    const fm = await run({ mode: RS_VIEW_T.motion });
    for (const ai of all) {
      const want = ai % 7 === 0 ? [0, 0, 0] : [0.25 * (ai % 5), -0.5, 0];
      if ([0, 1, 2].some((k) => fm.aovF[4 * ai + k] !== Math.fround(want[k]))) { bad.push(`motion px ${ai}: ${Array.from(fm.aovF.subarray(4 * ai, 4 * ai + 3))} ≠ ${want}`); break; }
    }
    expect(bad, bad.join('\n')).toEqual([]);
    // reservoir views at the tap "after temporal" (2) = res[w]; the initial / spatial taps stay unwritten by T3
    for (const id of RES_IDS) checkReservoirView(id, await run({ mode: id, tap: 2 }), x.resOut, all, 'tap temporal');
    const fi = await run({ mode: RS_VIEW.d, tap: TAP.initial });
    expect(all.filter((i) => fi.aov[4 * i] !== 0).length, 'tap initial: T3 writes nothing').toBe(0);
    // probe at a Q_i pixel with a q′ (every tag): one header (phase A skipped), fwd, inv, 2 refresh, select, pick
    const probeAi = all.find((ai) => { const t = decodeTStateLocal(x.words, P, NS, ai); return (t.flags & TS.TS_INV_QUEUED) && (t.flags & TS.TS_QVALID); })!;
    const px: [number, number] = [probeAi % W, Math.floor(probeAi / W)];
    const fp = await run({ mode: 0, probeEnabled: true, probePixel: px });
    const d = decodeRestirProbe(fp.probe.records);
    const t = decodeTStateLocal(x.words, P, NS, probeAi);
    const tp = d.temporal!;
    expect(fp.probe.records.filter((r) => r.tag === 73).length, 'one header (phase A skipped the Q_i pixel)').toBe(1);
    expect(d.reservoirs.map((r) => r.tap)).toEqual([2]);
    expect(Array.from(d.reservoirs[0].words)).toEqual(Array.from(x.resOut.subarray(probeAi * RES_WORDS, (probeAi + 1) * RES_WORDS)));
    expect([tp.ai, tp.qPrime, tp.cPrev, tp.flags]).toEqual([probeAi, t.qPrime, t.cPrev, t.flags]);
    expect(tp.forward).toMatchObject({ code: { sc: t.fwdCode & 0xFF }, Jp: jwValid(t.fwdJ) ? f32(t.fwdJ) : 0, JP: t.jP });
    expect(relErr(tp.forward!.lumF, lum(t.fwdF))).toBeLessThanOrEqual(1e-6);
    expect(tp.inverse).toMatchObject({ code: { sc: t.invCode & 0xFF }, Jinv: jwValid(t.invJ) ? f32(t.invJ) : 0, piP: t.piRecomp });
    expect(tp.select).toMatchObject({ wc: t.wc, wp: t.wp, sel: selCode(t.flags), phase: 1 });
    expect(tp.refresh.map((r) => [r.dir, r.fromPass])).toEqual([['fwd', false], ['inv', false]]);
    const sf = decodeSfxLocal(x.words, P, NS, 0, t.qPrime), si = decodeSfxLocal(x.words, P, NS, 1, probeAi);
    expect(tp.refresh.map((r) => [r.status, r.entryTo, r.aux])).toEqual([[sf.status, sf.entryTo, sf.aux], [si.status, si.entryTo, si.aux]]);
    expect(tp.pick).toEqual({ sp: (probeAi % 7 === 0 ? [-1, -1] : [px[0] + 0.25 * (probeAi % 5), px[1] - 0.5]), tap: probeAi % 10, valid: (probeAi & 1) === 1 });
    expect(fp.probe.counters[1], 'probe overflow').toBe(0);
    console.log(`[U-TD-1a/3] temporal views 480–496 ok on ${P} px; probe pixel ${probeAi}: ${fp.probe.records.length} records, flags ${tp.flagNames.join('|')}`);
    inBuf.destroy(); outBuf.destroy();
    rig.destroy();
  });
});

describe('U-TD-1 (b): view 497 s.boost from a synthetic arena (rs_debug_views)', () => {
  it('bit s − slots of every accepted boost slot; 0 without the boost', async () => {
    const W = 24, H = 16, P = W * H;
    const rig = await debugRig(W, H, 'full', { boostSlots: 2 });
    const { device } = rig.g;
    const NS = rig.kernel.resources.alloc.slots;
    expect(NS, 'numSlots = slots + boostSlots').toBe(5);
    await rig.frame(0, {});
    const r = rng(3);
    const aw = arenaWords(P, NS);
    const words = new Uint32Array(6 * P * NS);
    for (let ai = 0; ai < P; ai++) for (let s = 0; s < NS; s++) {
      const u = r();
      words[aw.slots + 4 * (ai * NS + s) + 3] = u < 0.4 ? K.JW_NOT_ACCEPTED : u < 0.7 ? K.JW_FAILED : bits(1.5);
      words[aw.codes + ai * NS + s] = u < 0.4 ? K.SC_NOT_ACCEPTED : K.SC_OK;
    }
    device.queue.writeBuffer(rig.kernel.resources.arena, 256, words);
    const fr = await rig.frame(0, { mode: RS_VIEW_T.boost }, { onlyViews: 1 });
    let bad = 0, set = 0;
    for (let ai = 0; ai < P; ai++) {
      let m = 0;
      for (let s = 3; s < NS; s++) if (words[aw.slots + 4 * (ai * NS + s) + 3] >>> 0 !== K.JW_NOT_ACCEPTED) m |= 1 << (s - 3);
      if (fr.aov[4 * ai] !== m) bad++;
      if (m) set++;
    }
    expect(bad, 'boost mask').toBe(0);
    expect(set).toBeGreaterThan(0);
    rig.destroy();
  });
});

// ------------------------------------------------------------------------------------------------ U-TD real frames

/** One advanced temporal frame (validation API: advance + frameUnits) with the debug passes, then the AOV, probe,
 *  tState / sfxOut and the post-temporal buffer res[w] of that same frame. */
async function temporalFrame(rig: DebugRig, t: number, cam: { camToWorld: number[]; yfov: number }, s: Partial<DebugSettings>) {
  const k = rig.kernel;
  const adv = k.advance({ t, camera: cam, lights: rig.g.gpu.scene.lights });
  const fr = await rig.frame(t, s, {
    custom: (enc) => {
      rig.pass.encodeBegin(enc);
      for (const u of k.frameUnits(t, { accum: rig.accum, counters: rig.counters })) u.encode(enc);
      rig.pass.encodeViews(enc, { rounds: k.lastRounds, t });
    },
  });
  const tw = await k.readTemporalState();
  const w = k.resBase();
  const resW = await k.readReservoirs(w as 0 | 1);
  return { fr, adv, tw, resW, rounds: k.lastRounds };
}
const camAt = (dx: number): { camToWorld: number[]; yfov: number } => { const c = boxCamera(); c.camToWorld[12] += dx; return c; };

describe('U-TD-1 (c) / U-TD-3: temporal views on real frames ≡ tState / res[w] read back', () => {
  it('moving camera (full preset): 480, 484, 486, 490, 491, 492, 495 and the tap "after temporal"; probe tags ≡ tState', async () => {
    const W = 48, H = 32, P = W * H;
    const rig = await debugRig(W, H, 'full', { maxBounces: 3 });
    const NS = rig.kernel.resources.alloc.slots;
    const all = Array.from({ length: P }, (_, i) => i);
    let t = 1;
    for (; t <= 3; t++) await temporalFrame(rig, t, camAt(0.004 * t), {});
    const bad: string[] = [];
    let qvalid = 0, fwdOk = 0, selP = 0;
    const ids = [RS_VIEW_T.qvalid, RS_VIEW_T.fwdCode, RS_VIEW_T.logJ, RS_VIEW_T.cprev, RS_VIEW_T.cout, RS_VIEW_T.sel, RS_VIEW_T.wp, RS_VIEW_T.pic];
    for (const id of ids) {
      const { fr, adv, tw, resW } = await temporalFrame(rig, t, camAt(0.004 * t), { mode: id });
      t++;
      expect(adv.histValid, `frame ${t - 1}: ${adv.reasons}`).toBe(true);
      let nb = 0;
      for (const ai of all) {
        const ts = decodeTStateLocal(tw, P, NS, ai);
        const out = decodeReservoir(resW, ai);
        const q = (ts.flags & TS.TS_QVALID) !== 0;
        if (id === RS_VIEW_T.qvalid) { qvalid += q ? 1 : 0; fwdOk += q && jwValid(ts.fwdJ) ? 1 : 0; selP += ts.flags & TS.TS_SEL_P ? 1 : 0; }
        let want: number, isCode = true;
        switch (id) {
          case RS_VIEW_T.qvalid: want = qvalidCode(ts.flags); break;
          case RS_VIEW_T.fwdCode: want = ts.flags & TS.TS_BG ? NONE : q ? ts.fwdCode & 0xFF : NONE; break;
          case RS_VIEW_T.sel: want = ts.flags & TS.TS_BG ? NONE : selCode(ts.flags); break;
          case RS_VIEW_T.logJ: isCode = false; want = q && jwValid(ts.fwdJ) ? log2pos(f32(ts.fwdJ)) : 0; break;
          case RS_VIEW_T.cprev: isCode = false; want = q ? ts.cPrev : 0; break;
          case RS_VIEW_T.cout: isCode = false; want = ts.flags & TS.TS_BG ? 0 : out.c; break;
          case RS_VIEW_T.wp: isCode = false; want = ts.wc + ts.wp > 0 ? ts.wp / (ts.wc + ts.wp) : 0; break;
          default: isCode = false; want = ts.flags & TS.TS_BG ? 0 : lum(out.F); break;
        }
        const got = isCode ? fr.aov[4 * ai] >>> 0 : fr.aovF[4 * ai];
        const ok = isCode ? got === want >>> 0 : relErr(got, want) <= 1e-6 || Math.abs(got - want) <= (id === RS_VIEW_T.logJ ? 1e-5 : 1e-30);
        if (!ok && nb++ < 2) bad.push(`view ${id} px ${ai}: ${got} ≠ ${want} (flags ${ts.flags})`);
      }
    }
    // reservoir views at the tap "after temporal" ≡ res[w] (hit pixels; T3 skips background pixels)
    for (const id of [RS_VIEW.c, RS_VIEW.W, RS_VIEW.d]) {
      const { fr, tw, resW } = await temporalFrame(rig, t, camAt(0.004 * t), { mode: id, tap: 2 });
      t++;
      const hit = all.filter((ai) => !(decodeTStateLocal(tw, P, NS, ai).flags & TS.TS_BG));
      checkReservoirView(id, fr, resW, hit, 'real tap temporal');
    }
    expect(bad, bad.join('\n')).toEqual([]);
    // U-TD-3 (real): probe at a hit pixel with a q′
    const px: [number, number] = [W >> 1, H >> 1];
    const pr = await temporalFrame(rig, t, camAt(0.004 * t), { mode: 0, probeEnabled: true, probePixel: px });
    const ai = px[1] * W + px[0];
    const ts = decodeTStateLocal(pr.tw, P, NS, ai);
    const d = decodeRestirProbe(pr.fr.probe.records);
    const tp = d.temporal;
    expect(tp, 'temporal probe records').toBeDefined();
    expect([tp!.ai, tp!.flags, tp!.cPrev]).toEqual([ai, ts.flags, ts.cPrev]);
    expect(tp!.qPrime).toBe(ts.qPrime === NONE ? undefined : ts.qPrime);
    if (ts.flags & TS.TS_QVALID) {
      expect(tp!.forward!.code.sc).toBe(ts.fwdCode & 0xFF);
      expect(tp!.select).toMatchObject({ wc: ts.wc, wp: ts.wp, sel: selCode(ts.flags) });
      expect(tp!.pick?.valid).toBe(true);
      expect(d.reservoirs.some((r) => r.tap === 2)).toBe(true);
      expect(d.paths.get(23)?.[0]?.b, 'forward path starts at y₁').toBe(1);
    }
    console.log(`[U-TD-1c/3 real] q′ valid ${qvalid}/${P}, forward OK ${fwdOk}, s = p ${selP}; probe ${pr.fr.probe.records.length} records, flags ${tp?.flagNames.join('|')}`);
    expect(qvalid, 'q′ valid pixels (vacuous without T-A A1 temporal_pixel)').toBeGreaterThan(P / 4);
    expect(fwdOk, 'forward OK (vacuous without T-B B1)').toBeGreaterThan(0);
    rig.destroy();
  });
});

describe('U-TD-2: view 493 (forward vs stored p̂) = 0 on static identity frames', () => {
  it('jitter off, static camera and lights (temporal preset): |493| ≤ 1e-5 wherever the forward shift is valid', async () => {
    const W = 48, H = 32, P = W * H;
    const rig = await debugRig(W, H, 'temporal', { maxBounces: 3 });
    rig.kernel.setView({ camera: boxCamera(), width: W, height: H, runSeed: 11, jitterMode: JITTER_NONE, jitter: [0.5, 0.5] });
    const NS = rig.kernel.resources.alloc.slots;
    for (let t = 1; t <= 3; t++) await temporalFrame(rig, t, boxCamera(), {});
    const { fr, tw, adv } = await temporalFrame(rig, 4, boxCamera(), { mode: RS_VIEW_T.phatRel });
    expect(adv.histValid).toBe(true);
    let n = 0, worst = 0;
    for (let ai = 0; ai < P; ai++) {
      const ts = decodeTStateLocal(tw, P, NS, ai);
      if (!(ts.flags & TS.TS_QVALID) || !jwValid(ts.fwdJ)) continue;
      n++;
      worst = Math.max(worst, Math.abs(fr.aovF[4 * ai]));
    }
    const m = await temporalFrame(rig, 5, boxCamera(), { mode: RS_VIEW_T.motion });
    let motion = 0;
    for (let ai = 0; ai < P; ai++) motion = Math.max(motion, Math.abs(m.fr.aovF[4 * ai]), Math.abs(m.fr.aovF[4 * ai + 1]));
    console.log(`[U-TD-2] ${n}/${P} pixels with a valid forward shift, max |493| ${worst}; max |481 motion| ${motion}`);
    expect(n, 'valid forward shifts (vacuous before T-B B1)').toBeGreaterThan(P / 4);
    expect(worst).toBeLessThanOrEqual(1e-5);
    expect(motion, 'static camera, jitter off: s′ − q = 0 (A-7; f32 re-projection)').toBeLessThan(1e-4);
    rig.destroy();
  });
});

describe('U-TD-1 (d): views 482, 483, 493, 494, 496 on real frames with light changes (robust mode)', () => {
  it('moving + brightening lights, moving camera, temporalCheck robust (rounds 0: res[h] still readable): views ≡ tState / sfxOut / res read back', async () => {
    const W = 48, H = 32, P = W * H;
    const rig = await debugRig(W, H, 'temporal', { maxBounces: 3, temporalCheck: 'robust' });
    const k = rig.kernel;
    const NS = k.resources.alloc.slots;
    const base = rig.g.gpu.scene.lights;
    const lightsAt = (t: number) => base.map((l) => {
      const m = Float32Array.from(l.matrix);
      if (l.id === 1) m[12] += 0.03 * t;                                   // point light moves (LCB_MOVED)
      return { ...l, matrix: m, power: l.id === 3 ? l.power * (1 + 0.5 * (t % 2)) : l.power };   // rect power steps (RADIO)
    });
    const frame = async (t: number, s: Partial<DebugSettings>) => {
      const adv = k.advance({ t, camera: camAt(0.003 * t), lights: lightsAt(t) });
      const h = k.historyIndex();
      const fr = await rig.frame(t, s, {
        custom: (enc) => {
          rig.pass.encodeBegin(enc);
          for (const u of k.frameUnits(t, { accum: rig.accum, counters: rig.counters })) u.encode(enc);
          rig.pass.encodeViews(enc, { rounds: k.lastRounds, t });
        },
      });
      expect(k.lastRounds, 'temporal preset: no spatial round (res[h] intact)').toBe(0);
      const tw = await k.readTemporalState();
      const resOut = await k.readReservoirs(k.resBase() as 0 | 1);
      const resIn = h < 0 ? resOut : await k.readReservoirs(h as 0 | 1);   // h = −1: first frame (no history)
      const st = k.lights.state, slot = st.curSlot;
      const lcb = (e: number) => {
        if (e === slot.envEntry) return (adv.flags & K.TF_ENV_MOVED ? K.LCB_MOVED : 0) | (adv.flags & K.TF_ENV_RADIO ? K.LCB_RADIO : 0);
        if (e < slot.nAnalytic) return adv.flags & K.TF_LIGHTS_SAME ? 0 : st.records[slot.lightOff + 28 * e + 26];
        return 0;
      };
      return { fr, adv, x: { words: tw, resIn, resOut, base: 0 } as TSynth, lcb };
    };
    let t = 1;
    for (; t <= 3; t++) await frame(t, {});
    const bad: string[] = [];
    const stats: Record<number, Record<string, number>> = {};
    for (const id of [RS_VIEW_T.refreshFwd, RS_VIEW_T.refreshInv, RS_VIEW_T.phatRel, RS_VIEW_T.robust, RS_VIEW_T.lightsChanged]) {
      const { fr, adv, x, lcb } = await frame(t, { mode: id });
      t++;
      expect(adv.histValid, adv.reasons.join(',')).toBe(true);
      expect(adv.flags & K.TF_REFRESH, 'refresh frame').toBe(K.TF_REFRESH);
      const hist: Record<string, number> = {};
      let nb = 0;
      for (let ai = 0; ai < P; ai++) {
        const ts = decodeTStateLocal(x.words, P, NS, ai);
        if (ts.flags & TS.TS_BG) continue;                                // T3 skips background (fill / cleared AOV)
        const e = expectedTemporalView(id, x, P, NS, ai, adv.flags, adv.temporal.frameGen, lcb);
        const got = e.code !== undefined ? fr.aov[4 * ai] >>> 0 : fr.aovF[4 * ai];
        const want = e.code !== undefined ? e.code >>> 0 : e.v![0];
        const ok = e.code !== undefined ? got === want : Math.abs(got - want) <= 1e-6 || relErr(got, want) <= 1e-5;
        const key = e.code !== undefined ? String(want) : want === 0 ? '0' : '≠0';
        hist[key] = (hist[key] ?? 0) + 1;
        if (!ok && nb++ < 3) bad.push(`view ${id} px ${ai}: ${got} ≠ ${want} (flags ${ts.flags})`);
      }
      stats[id] = hist;
    }
    console.log(`[U-TD-1d] value histograms (hit pixels): ${JSON.stringify(stats)}`);
    expect(bad, bad.join('\n')).toEqual([]);
    // non-vacuous: refreshed records of both directions, robust comparisons, light-change bits
    const nz = (id: number, keys: string[]) => keys.reduce((a, kk) => a + (stats[id][kk] ?? 0), 0);
    expect(nz(RS_VIEW_T.refreshFwd, ['1', '2', '3', '4', '5']), 'fwd refresh classes').toBeGreaterThan(0);
    expect(nz(RS_VIEW_T.refreshInv, ['1', '2', '3', '4', '5']), 'inv refresh classes').toBeGreaterThan(0);
    expect(nz(RS_VIEW_T.robust, ['≠0']) + (stats[RS_VIEW_T.robust]['0'] ?? 0)).toBeGreaterThan(0);
    expect(nz(RS_VIEW_T.lightsChanged, ['1', '2', '3', '4', '5', '6', '7']), 'lightsChanged bits').toBeGreaterThan(0);
    rig.destroy();
  });
});

describe('interactive RestirFramePass: the finalize bind-group cache stays bounded (Changelog D-4)', () => {
  it('N advanced + held frames reuse one colour view: the group cache size is constant after the first frames', async () => {
    const g = await gpuScene(bitFixtureScene('x_quads'));
    const dev = g.device;
    const { FrameUniformBuffer } = await import('../../src/core/render/frame-uniforms.ts');
    const W = 64, H = 48;
    const pass = await RestirKernel.interactive(dev, g.gpu, g.env, 'rgba16float', { settings: restirSettings('interactive', { maxBounces: 3 }), features: g.features, wgslLanguageFeatures: g.wgslLanguageFeatures });
    const color = dev.createTexture({ size: [W, H], format: 'rgba16float', usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.COPY_SRC });
    const fu = new FrameUniformBuffer(dev);
    pass.setTargets({ width: W, height: H, color, frameUniforms: fu.buffer });
    const k = pass.kernel;
    const groups = () => (k.resources as unknown as { groups: Map<string, GPUBindGroup> }).groups.size;
    const sizes: number[] = [];
    for (let f = 0; f < 24; f++) {
      fu.write({ camera: boxCamera(), prevCamera: boxCamera(), width: W, height: H, frameIndex: f, seedIndex: f, runSeed: 7, flags: 0,
        jitterMode: JITTER_IID, jitter: [0.5, 0.5], origin: g.gpu.origin, exposure: 1, time: 0, dt: 0, sceneDiag: 1 });
      const enc = dev.createCommandEncoder();
      if (f % 4 === 3) pass.encodeHold(enc, { accumulate: true });
      else { k.advanceInteractive(fu.buffer, { reset: false }); pass.encode(enc, { advanced: true, accumulate: true }); }
      dev.queue.submit([enc.finish()]);
      await dev.queue.onSubmittedWorkDone();
      sizes.push(groups());
    }
    console.log(`[bind-group cache] sizes ${sizes.join(',')}`);
    expect(sizes[23], 'no growth after the warm-up frames').toBe(sizes[7]);
    pass.destroy(); color.destroy(); fu.destroy(); g.destroy();
  });
});

// ------------------------------------------------------------------------------------------------ M6 views 471–479

import { RS_VIEW_M6 } from '../../src/core/render/restir/debug.ts';
import { gaussLayer, pairPartner, pairTransform } from '../../src/core/render/restir/pairing.ts';
import { GAUSS_PAIR_SIZES } from '../../src/core/render/restir/presets.ts';

describe('U-PAIR-VIEW / U-DUP-VIEW (restir-m6-api.md §1.1, §1.6): views 471–479', () => {
  it('pairing offset (hue = angle, value = length/48) of slot 0 and reciprocity (0 / 1, never 2) with the σ = 16 maps', async () => {
    const W = 64, H = 64;
    const rig = await debugRig(W, H, 'offline', { trees: 1, rounds: 1, slots: 3, pairing: 'gauss' });
    const t = 5;
    const off = await rig.frame(t, { mode: RS_VIEW_M6.pairOffset });
    const layer = gaussLayer(GAUSS_PAIR_SIZES[0], 16, 0);
    const tr = pairTransform(11, 0, t, 0, 0, layer.W);
    let checked = 0, bad = 0;
    for (let y = 0; y < H; y += 5) for (let x = 0; x < W; x += 5) {
      const q = pairPartner(layer, tr, [x, y], W, H);
      const i = 4 * (y * W + x);
      const c = [off.aovF[i], off.aovF[i + 1], off.aovF[i + 2]];
      if (!q) { if (c.some((v) => v !== 0)) bad++; continue; }
      const dx = q[0] - x, dy = q[1] - y;
      const v = Math.min(Math.hypot(dx, dy) / 48, 1);
      if (Math.abs(Math.max(...c) - v) > 1e-4) bad++;
      checked++;
    }
    expect(checked).toBeGreaterThan(50);
    expect(bad).toBe(0);
    const rec = await rig.frame(t, { mode: RS_VIEW_M6.pairRecip });
    const hist = [0, 0, 0];
    for (let i = 0; i < W * H; i++) { const c = rec.aov[4 * i]; if (c <= 2) hist[c]++; }
    expect(hist[2]).toBe(0);
    expect(hist[0]).toBeGreaterThan(0);
    rig.destroy();
  });

  it('duplication count (478) equals a CPU count over the final reservoirs; cap (479) = cCap − (cCap − 1)·(n/288)^0.1', async () => {
    const W = 48, H = 40;
    const rig = await debugRig(W, H, 'full', { dupmap: true, cCap: 20 });
    const k = rig.kernel;
    let last: Frame | undefined, capF: Frame | undefined;
    for (let t = 0; t < 4; t++) {
      k.advance({ t, camera: boxCamera(), lights: rig.g.gpu.scene.lights });
      last = await rig.frame(t, { mode: RS_VIEW_M6.dupCount });
    }
    const res = await k.readReservoirs('final');
    const seed = (i: number) => [res[40 * i + 4], res[40 * i + 5]];
    const ok = (i: number) => (res[40 * i + 6] & 0xF) !== 0 && (res[40 * i + 6] & 0x2000000) === 0;
    let bad = 0, nonzero = 0;
    for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
      const i = y * W + x;
      let n = 0;
      if (ok(i)) {
        const s = seed(i);
        for (let dy = -8; dy <= 8; dy++) for (let dx = -8; dx <= 8; dx++) {
          if (!dx && !dy) continue;
          const xx = x + dx, yy = y + dy;
          if (xx < 0 || yy < 0 || xx >= W || yy >= H) continue;
          const j = yy * W + xx;
          if (ok(j) && seed(j)[0] === s[0] && seed(j)[1] === s[1]) n++;
        }
      }
      if (n > 0) nonzero++;
      if (last!.aovF[4 * i] !== n) bad++;
    }
    console.log(`[U-DUP-VIEW] pixels with duplicates: ${nonzero} / ${W * H}`);
    expect(bad).toBe(0);
    expect(nonzero).toBeGreaterThan(0);
    k.advance({ t: 4, camera: boxCamera(), lights: rig.g.gpu.scene.lights });
    capF = await rig.frame(4, { mode: RS_VIEW_M6.dupCap });
    expect(capF.aovF[0]).toBeGreaterThanOrEqual(1);
    expect(capF.aovF[0]).toBeLessThanOrEqual(20);
    rig.destroy();
  });
});
