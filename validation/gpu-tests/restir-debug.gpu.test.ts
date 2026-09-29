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
// Chrome lane authoritative; dawn.node is a pre-check.
import { afterAll, describe, expect, it } from 'vitest';
import { readBuffer } from '../../src/core/gpu/readback.ts';
import {
  BUILTIN_VIEWS, DEBUG_BUFFER_LAYOUT, DebugResources, DebugViewRegistry, defaultDebugSettings, type DebugSettings,
} from '../../src/core/render/debug-views.ts';
import { JITTER_IID } from '../../src/core/render/frame-uniforms.ts';
import { parseDebugHeader, type ProbeFrame } from '../../src/core/render/probe.ts';
import { RESTIR_VIEWS, RS_VIEW_THR, RestirDebugPass, decodeJWord, decodeRestirProbe, f16ToF32, shiftedPolylines } from '../../src/core/render/restir/debug.ts';
import { RestirKernel } from '../../src/core/render/restir/kernel.ts';
import {
  RES_WORDS, RS_VIEW, RS_WGSL_CONSTS as K, RW, arenaWords, decodeReservoir, pathClass, rfPack, rfUnpack,
} from '../../src/core/render/restir/layout.ts';
import { restirSettings, type RestirPresetName, type RestirSettings } from '../../src/core/render/restir/presets.ts';
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
      const NS = rig.kernel.settings.slots;
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
      expect(d.slots.length, 'slot records of the probe pixel').toBe(rig.kernel.settings.slots);
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
