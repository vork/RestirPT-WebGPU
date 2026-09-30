// M5 temporal passes T1–T4 (T-B, restir-temporal-api.md §3.4, §3.6, §6.1). Chrome lane authoritative.
//   canary        known-answer kernel on the temporal G0 layout (Metal Q1)
//   T6(a) GPU     production tmis_* (WGSL, f32) ≡ the f64 reference tests/restir/tmis-ref.ts to 1e-6
//   U-TQ-1 GPU    Q_f / Q_i: every item processed once, overflow exactly when counter > P, rs_args queue selection
//   §9.3-5        jitter off, static camera and lights: T = identity (q′ = q, J = 1, F_t(Y_p) = F_p^st), π_p(X_c) = p̂(X_c),
//                 the temporal W_Y equals the GRIS formula (f64 from the tState words), c = 1 + c_p
//   B1 camera     camera translation / rotation / zoom chains: T6(b) robust mode (stored route ≡ π_p(Y_p) recomputed by
//                 E_{t−1}), the W_Y formula on every finalised pixel, counters (non-finite, pending, overflow) 0
import { afterAll, describe, expect, it } from 'vitest';
import { RestirKernel } from '../../src/core/render/restir/kernel.ts';
import { RES_WORDS, RW, RS_WGSL_CONSTS as K, SC_NAMES, TS_CONSTS as T, arenaWords } from '../../src/core/render/restir/layout.ts';
import { restirSettings } from '../../src/core/render/restir/presets.ts';
import { JITTER_IID, JITTER_NONE, type CameraState, type JitterMode } from '../../src/core/render/frame-uniforms.ts';
import { releaseTestGpu } from './device-factory.ts';
import { allLightsScene, bitFixtureScene, boxCamera, gpuScene, restirRig, storageBuffer } from './restir-fixtures.ts';
import { shaderSources } from '../../src/core/shaders/index.ts';
import type { LightData } from '../../src/core/scene/types.ts';
import { lightMatrixToward } from './pt-fixtures.ts';
import { animatedLights, hashU32, movedCamera, readU32, recLumF, runChain, temporalSnapshot, type ChainFrame, type TemporalSnapshot } from './restir-temporal-fixtures.ts';
import { tmisContribW, tmisTalbotMc, tmisTalbotMp } from '../../tests/restir/tmis-ref.ts';

afterAll(releaseTestGpu);

const f32 = (x: number) => Math.fround(x);
const u2f = (u: number) => new Float32Array(new Uint32Array([u]).buffer)[0];
const relErr = (a: number, b: number) => Math.abs(a - b) / Math.max(Math.abs(a), Math.abs(b), 1e-30);
const scOf = (code: number) => code & 0xff;

// ------------------------------------------------------------------------------------------------ canary + T6(a)

describe('canary + T6(a) GPU: tmis_* (WGSL f32) ≡ f64 reference', () => {
  it('known answer, then 2^16 random inputs incl. zeros, capped c_p, denormal-free extremes', async () => {
    const g = await gpuScene(bitFixtureScene('c0c'));
    const k = await RestirKernel.create(g.device, g.gpu, g.env, { settings: restirSettings('temporal'), features: g.features, wgslLanguageFeatures: g.wgslLanguageFeatures });
    k.setView({ camera: boxCamera(), width: 8, height: 8, runSeed: 1 });
    const device = g.device;
    const N = 1 << 16;
    const inp = new Float32Array(N * 8);
    let s = 12345;
    const rnd = () => { s = (Math.imul(s, 1664525) + 1013904223) >>> 0; return s / 4294967296; };
    const piv = () => (rnd() < 0.15 ? 0 : f32(10 ** (-6 + 10 * rnd())));
    for (let i = 0; i < N; i++) {
      const cP = rnd() < 0.3 ? 20 : Math.floor(21 * rnd());
      inp.set([piv(), piv() + 1e-12, piv(), 1, cP, piv() + 1e-12, piv(), 0], 8 * i);
    }
    const bIn = storageBuffer(device, inp), bOut = storageBuffer(device, N * 16);
    const g2l = device.createBindGroupLayout({ entries: [0, 1].map((binding) => ({ binding, visibility: GPUShaderStage.COMPUTE, buffer: { type: binding === 0 ? 'read-only-storage' : 'storage' } as GPUBufferBindingLayout })) });
    const src = `
#include "restir/tmis.wgsl"
@group(2) @binding(0) var<storage, read> inp: array<f32>;
@group(2) @binding(1) var<storage, read_write> outv: array<vec4f>;
@compute @workgroup_size(64) fn main(@builtin(global_invocation_id) gid: vec3u) {
  let i = gid.y * 65535u * 64u + gid.x;
  if (i >= ${N}u) { return; }
  let b = 8u * i;
  let piSel = inp[b]; let piC = inp[b + 1u]; let piP = inp[b + 2u]; let cC = inp[b + 3u]; let cP = inp[b + 4u]; let wSum = inp[b + 5u];
  let ph = inp[b + 6u];
  if (i == 0u) { outv[0] = vec4f(bitcast<f32>(0xC0FFEEu), tmis_contrib_W(2.0, 2.0, 1.0, 1.0, 2.0, 8.0), tmis_talbot_mc(1.0, 3.0, 1.0, 1.0), tmis_talbot_mp(1.0, 0.0, 0.0, 0.0)); return; }
  outv[i] = vec4f(tmis_contrib_W(piSel, piC, piP, cC, cP, wSum), tmis_talbot_mc(cC, ph, cP, piP), tmis_talbot_mp(cC, ph, cP, piP), 0.0);
}`;
    const pl = await k.compile('tmis-test.wgsl', 'main', k.customDefines({}, false), k.customLayout(g2l, false), 'tmis-test', { 'tmis-test.wgsl': src });
    const bg = device.createBindGroup({ layout: g2l, entries: [{ binding: 0, resource: { buffer: bIn } }, { binding: 1, resource: { buffer: bOut } }] });
    k.beginSubmit();
    const enc = device.createCommandEncoder();
    k.encodeCustom(enc, pl, bg, {}, [N / 64, 1], false);
    device.queue.submit([enc.finish()]);
    const o = new Float32Array((await readU32(device, bOut)).buffer);
    expect(new Uint32Array(o.buffer)[0]).toBe(0xC0FFEE);
    expect(o[1]).toBeCloseTo((2 / (2 + 2)) * (8 / 2), 6);
    expect(o[2]).toBeCloseTo(0.75, 6);
    expect(o[3]).toBe(0);
    let worst = 0;
    for (let i = 1; i < N; i++) {
      const [piSel, piC, piP, cC, cP, wSum, ph] = inp.subarray(8 * i, 8 * i + 7);
      const ref = [tmisContribW(piSel, piC, piP, cC, cP, wSum), tmisTalbotMc(cC, ph, cP, piP), tmisTalbotMp(cC, ph, cP, piP)];
      for (let j = 0; j < 3; j++) {
        const e = relErr(o[4 * i + j], ref[j]);
        if (!(Number.isFinite(ref[j]) && Math.abs(ref[j]) < 1e30)) continue;
        worst = Math.max(worst, e);
        if (e > 1e-6) expect.fail(`i=${i} fn=${j} gpu ${o[4 * i + j]} ref ${ref[j]} (in ${Array.from(inp.subarray(8 * i, 8 * i + 7))})`);
      }
    }
    console.log(`[T6(a) GPU] worst rel err ${worst.toExponential(2)} over ${N} × 3`);
    bIn.destroy(); bOut.destroy(); k.destroy(); g.destroy();
  });
});

// ------------------------------------------------------------------------------------------------ U-TQ-1 GPU

describe('U-TQ-1 GPU: temporal queues', () => {
  it('Q_f / Q_i appends (RS_TEMPORAL queue_append), rs_args per queue, every item consumed once, overflow iff counter > P', async () => {
    const g = await gpuScene(bitFixtureScene('c0c'));
    const k = await RestirKernel.create(g.device, g.gpu, g.env, { settings: restirSettings('temporal'), features: g.features, wgslLanguageFeatures: g.wgslLanguageFeatures });
    const W = 16, H = 8, P = W * H;
    k.setView({ camera: boxCamera(), width: W, height: H, runSeed: 1 });
    await k.pipeline('rs_args');
    const device = g.device, res = k.resources;
    const NS = res.alloc.slots;
    const hits = storageBuffer(device, 4 * 2 * P);
    const g2l = device.createBindGroupLayout({ entries: [0, 1].map((binding) => ({ binding, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } as GPUBufferBindingLayout })) });
    // producer: thread i appends item word (i << 3) to q1 for i < nF and to q2 for i < nI; consumer (indirect): marks
    // hits[q][item ai]; a q0 item must never appear in q1/q2's regions (the q0 counter is left at 0).
    const src = `
#include "restir/tframe.wgsl"
@group(2) @binding(1) var<storage, read_write> hits: array<atomic<u32>>;
@compute @workgroup_size(64) fn produce(@builtin(global_invocation_id) gid: vec3u) {
  let i = gid.x;
  if (i < rsDispatch.treeBase) { queue_append(RS_Q_FWD, queue_item_word(i % ${P}u, 0u)); }
  if (i < rsDispatch.treeCount) { queue_append(RS_Q_INV, queue_item_word(i % ${P}u, 0u)); }
}
@compute @workgroup_size(64) fn consume(@builtin(workgroup_id) wid: vec3u, @builtin(num_workgroups) nwg: vec3u, @builtin(local_invocation_index) lid: u32) {
  let q = (rsDispatch.flags >> RSD_QUEUE_SHIFT) & 3u;
  let i = queue_item_chunk(q, wid, nwg, lid, 0u, 0u);
  if (i == 0xFFFFFFFFu) { return; }
  let ai = queue_item_ai(arena_word(arena_item_word(queue_item_base(q) + i)));
  atomicAdd(&hits[(q - 1u) * ${P}u + ai], 1u);
}`;
    const defs = k.customDefines({ RS_ARENA_BINDING: '0u', RS_ARENA_RW: true, RS_TEMPORAL: 1 }, false);
    const layout = k.customLayout(g2l, false);
    const prod = await k.compile('tq.wgsl', 'produce', defs, layout, 'tq-produce', { 'tq.wgsl': src });
    const cons = await k.compile('tq.wgsl', 'consume', defs, layout, 'tq-consume', { 'tq.wgsl': src });
    const bg = device.createBindGroup({ layout: g2l, entries: [{ binding: 0, resource: { buffer: res.arena } }, { binding: 1, resource: { buffer: hits } }] });
    for (const [nF, nI] of [[0, 1], [1, 0], [64, 65], [P, P], [P + 7, 3 * P]]) {
      const enc = device.createCommandEncoder();
      enc.clearBuffer(res.arena);
      enc.clearBuffer(hits);
      k.beginSubmit();
      k.encodeCustom(enc, prod, bg, { treeBase: nF, treeCount: nI }, [Math.ceil(Math.max(nF, nI, 1) / 64), 1], false);
      for (const q of [K.RS_Q_FWD, K.RS_Q_INV]) {
        k.encodePass(enc, 'rs_args', k.pipelineSync('rs_args'), res.g2('rs_args'), { flags: q << K.RSD_QUEUE_SHIFT }, [1, 1]);
        const off = k.dispatchSlot({ flags: q << K.RSD_QUEUE_SHIFT });
        const pass = enc.beginComputePass();
        pass.setPipeline(cons);
        pass.setBindGroup(0, (k as unknown as { ensureG0(): GPUBindGroup }).ensureG0(), [off]);
        pass.setBindGroup(1, device.createBindGroup({ layout: k.layouts.empty, entries: [] }));
        pass.setBindGroup(2, bg);
        pass.setBindGroup(3, device.createBindGroup({ layout: k.layouts.empty, entries: [] }));
        pass.dispatchWorkgroupsIndirect(res.args, 16 * q);
        pass.end();
      }
      device.queue.submit([enc.finish()]);
      const c = await k.readCounters(true);
      const h = await readU32(device, hits);
      const args = await readU32(device, res.args, 64);
      for (const [q, n] of [[K.RS_Q_FWD, nF], [K.RS_Q_INV, nI]] as const) {
        const hq = c.queues[q];
        expect(hq.counter, `q${q} n=${n}`).toBe(n);
        expect(hq.n).toBe(Math.min(n, P));
        expect(hq.capacity).toBe(P);
        expect(hq.overflow).toBe(n > P ? 1 : 0);
        const g64 = Math.ceil(Math.min(n, P) / 64);
        expect(Array.from(args.subarray(4 * q, 4 * q + 3))).toEqual([g64, 1, 1]);
        // every stored item consumed exactly once: min(n, P) items in total; which appends win the P slots is up to the
        // atomics, so per pixel the count is bounded by that pixel's appends (appends i carry ai = i mod P)
        const appends = new Uint32Array(P);
        for (let i = 0; i < n; i++) appends[i % P]++;
        const hq2 = Array.from(h.subarray((q - 1) * P, q * P));
        expect(hq2.reduce((x, y) => x + y, 0)).toBe(Math.min(n, P));
        expect(hq2.every((x, ai) => x <= appends[ai])).toBe(true);
        if (n <= P) expect(hq2).toEqual(Array.from(appends));
      }
      expect(c.queues[0].counter).toBe(0);
      expect(c.queues[0].capacity).toBe(0);                // rs_args(q0) not run here
      expect(arenaWords(P, NS).items + 2 * P).toBeLessThanOrEqual(arenaWords(P, NS).tState);
    }
    hits.destroy(); k.destroy(); g.destroy();
  });
});

// ------------------------------------------------------------------------------------------------ helpers

/** W_Y the GRIS formula predicts for finalised pixel ai (contribution MIS), from the tState words and the reservoirs. */
function expectedW(s: TemporalSnapshot, ai: number): number | undefined {
  const r = s.ts(ai);
  if (!(r.flags & T.TS_QVALID) || (r.flags & T.TS_EMPTY_OUT)) return undefined;
  const wSum = r.wc + r.wp;
  if (r.flags & T.TS_SEL_P) {
    const piC = recLumF(s.outF, ai);                     // = lum F_t(Y_p) (written back)
    const Jp = u2f(r.fwdJ);
    const piP = recLumF(s.histF, r.qPrime) / Jp;
    return tmisContribW(piP, piC, piP, 1, r.cP, wSum);
  }
  if (r.flags & T.TS_SEL_C) {
    const piC = recLumF(s.outF, ai);
    return tmisContribW(piC, piC, r.piRecomp, 1, r.cP, wSum);
  }
  return undefined;
}

// ------------------------------------------------------------------------------------------------ §9.3-5 identity

describe('§9.3-5: jitter off, static camera and lights ⇒ the temporal shift is the identity', () => {
  it('q′ = q, J_p = 1, F_t(Y_p) = F_p^st (≤ 1e-6), π_p(X_c) = p̂(X_c), W_Y = GRIS formula, c = 1 + c_p', async () => {
    const scene = bitFixtureScene('x_quads');
    const rig = await restirRig(scene, 64, 64, { preset: 'temporal', settings: { maxBounces: 3 }, jitterMode: JITTER_NONE, seed: 31 });
    const frames: ChainFrame[] = Array.from({ length: 4 }, (_, t) => ({ t, camera: boxCamera(), lights: scene.lights }));
    const stats = { q: 0, fwdOk: 0, jBad: 0, fBad: 0, fWorst: 0, jWorst: 0, piWorst: 0, piBad: 0, piN: 0, wWorst: 0, wBad: 0, wN: 0, cBad: 0, notSelf: 0 };
    const codes = new Map<string, number>();
    const res = await runChain(rig, frames, async (f, r) => {
      if (f.t < 3) return;
      expect(r.histValid).toBe(true);
      const s = await temporalSnapshot(rig.kernel);
      for (let ai = 0; ai < s.P; ai++) {
        const ts = s.ts(ai);
        if (!(ts.flags & T.TS_QVALID)) continue;
        stats.q++;
        if (ts.qPrime !== ai) stats.notSelf++;
        const sc = scOf(ts.fwdCode);
        codes.set(SC_NAMES[sc], (codes.get(SC_NAMES[sc]) ?? 0) + 1);
        if (ts.fwdJ !== K.JW_FAILED) {
          stats.fwdOk++;
          const J = u2f(ts.fwdJ);
          const e = Math.abs(J - 1);
          stats.jWorst = Math.max(stats.jWorst, e);
          if (e > 1e-6) stats.jBad++;
          const Fh = [0, 1, 2].map((c) => s.recF(s.histF, ts.qPrime, RW.F + c));
          const ef = Math.max(...[0, 1, 2].map((c) => relErr(ts.fwdF[c], Fh[c])));
          stats.fWorst = Math.max(stats.fWorst, ef);
          if (ef > 1e-6) stats.fBad++;
        }
        if ((ts.flags & T.TS_SEL_C) && (ts.flags & T.TS_INV_DONE)) {
          stats.piN++;
          const e = relErr(ts.piRecomp, recLumF(s.outF, ai));
          stats.piWorst = Math.max(stats.piWorst, e);
          if (e > 1e-6) stats.piBad++;
        }
        const Wexp = expectedW(s, ai);
        if (Wexp !== undefined) {
          stats.wN++;
          const e = relErr(s.recF(s.outF, ai, RW.W), Wexp);
          stats.wWorst = Math.max(stats.wWorst, e);
          if (e > 1e-5) stats.wBad++;
          if (s.recF(s.outF, ai, RW.c) !== f32(1 + ts.cP)) stats.cBad++;
        }
      }
    });
    console.log(`[§9.3-5] ${JSON.stringify(stats)} fwd codes ${JSON.stringify(Object.fromEntries(codes))}`);
    for (const r of res) {
      expect(r.counters.rsc.tNonFinite, `t=${r.t}`).toBe(0);
      expect(r.counters.rsc.tPendingLeft, `t=${r.t}`).toBe(0);
      expect(r.counters.queues[1].overflow + r.counters.queues[2].overflow).toBe(0);
    }
    expect(stats.q).toBeGreaterThan(64 * 64 * 0.5);
    expect(stats.notSelf).toBe(0);
    expect(stats.fwdOk).toBeGreaterThan(stats.q * 0.5);
    expect(stats.jBad).toBe(0);
    expect(stats.fBad).toBe(0);
    expect(stats.piBad).toBe(0);
    expect(stats.wBad).toBe(0);
    expect(stats.cBad).toBe(0);
    rig.destroy();
  });
});

// ------------------------------------------------------------------------------------------------ B1: camera motion

interface CamCase { name: string; cam: (t: number) => CameraState }
const CAM_CASES: CamCase[] = [
  { name: 'translate', cam: (t) => movedCamera(0.015 * t, 0.004 * t, 0, 0) },
  { name: 'translate-fast', cam: (t) => movedCamera(0.08 * t, -0.02 * t, 0, 0) },
  { name: 'rotate', cam: (t) => movedCamera(0, 0, 0, 0.006 * t) },
  { name: 'rotate-fast', cam: (t) => movedCamera(0, 0, 0, 0.03 * t) },
  { name: 'zoom-in', cam: (t) => movedCamera(0, 0, -0.03 * t, 0) },
  { name: 'zoom-out', cam: (t) => movedCamera(0, 0, 0.03 * t, 0, (55 - 0.3 * t) * Math.PI / 180) },
];

describe('B1 camera-only motion: robust mode (T6(b) camera part), W_Y formula, counters', () => {
  for (const c of CAM_CASES) {
    it(c.name, async () => {
      const scene = bitFixtureScene('x_quads');
      const rig = await restirRig(scene, 96, 96, { preset: 'temporal', settings: { maxBounces: 3, temporalCheck: 'robust' }, jitterMode: JITTER_IID, seed: 41 });
      const frames: ChainFrame[] = Array.from({ length: 8 }, (_, t) => ({ t, camera: c.cam(t), lights: scene.lights }));
      const st = { selP: 0, mism: 0, robustN: 0, robustBad: 0, worst: 0, wN: 0, wBad: 0, wWorst: 0, q: 0, disocc: 0, ring: 0 };
      const bad: string[] = [];
      const res = await runChain(rig, frames, async (f, r) => {
        if (f.t === 0) return;
        expect(r.histValid).toBe(true);
        st.selP += r.counters.rsc.tSelP;
        st.mism += r.counters.rsc.tRobustMismatch;
        st.q += r.counters.rsc.tQvalid;
        st.disocc += r.counters.rsc.tDisocc;
        const s = await temporalSnapshot(rig.kernel);
        for (let ai = 0; ai < s.P; ai++) {
          const ts = s.ts(ai);
          if (ts.flags & T.TS_PICK_RING) st.ring++;
          if ((ts.flags & T.TS_ROBUST) && (ts.flags & T.TS_INV_DONE)) {
            st.robustN++;
            const e = scOf(ts.invCode) === K.SC_OK ? relErr(ts.piStored, ts.piRecomp) : 1;
            st.worst = Math.max(st.worst, e);
            if (e > 1e-3) { st.robustBad++; if (bad.length < 8) bad.push(`t${f.t} ai${ai} q'${ts.qPrime} stored ${ts.piStored} recomp ${ts.piRecomp} inv ${SC_NAMES[scOf(ts.invCode)]} fwd ${SC_NAMES[scOf(ts.fwdCode)]} flags ${rig.kernel.resources ? s.rec(s.hist, ts.qPrime, RW.flags).toString(16) : ''}`); }
          }
          const Wexp = expectedW(s, ai);
          if (Wexp !== undefined && !(ts.flags & T.TS_ROBUST)) {
            st.wN++;
            const e = relErr(s.recF(s.outF, ai, RW.W), Wexp);
            st.wWorst = Math.max(st.wWorst, e);
            if (e > 1e-5) st.wBad++;
          }
        }
      });
      console.log(`[B1 ${c.name}] ${JSON.stringify(st)}${bad.length ? '\n  ' + bad.join('\n  ') : ''}`);
      for (const r of res) {
        const x = r.counters.rsc;
        expect([x.tNonFinite, x.tPendingLeft, x.wNonFinite, x.shiftNonFinite, x.candNonFinite], `t=${r.t}`).toEqual([0, 0, 0, 0, 0]);
        expect(r.counters.queues[1].overflow + r.counters.queues[2].overflow).toBe(0);
      }
      expect(st.q).toBeGreaterThan(0);
      expect(st.robustN).toBeGreaterThan(100);
      expect(st.robustBad / st.robustN).toBeLessThanOrEqual(1e-3);
      expect(st.wBad).toBe(0);
      rig.destroy();
    });
  }
});

// ------------------------------------------------------------------------------------------------ B1: unbiasedness smoke

/** Per-member mean over frames 1…T−1 and hit pixels of lum(F)·W of the final reservoirs (rounds 0: the estimate). */
async function memberMeans(preset: 'temporal' | 'initial', E: number, W: number, H: number, T: number, cam: (t: number) => CameraState, settings: Record<string, unknown> = {}): Promise<number[]> {
  const scene = bitFixtureScene('x_quads');
  const rig = await restirRig(scene, W, H, { preset, settings: { maxBounces: 3, ...settings }, members: E, jitterMode: JITTER_IID, seed: 7012 });
  const a = rig.kernel.resources.alloc;
  const sums = new Float64Array(E);
  await runChain(rig, Array.from({ length: T }, (_, t) => ({ t, camera: cam(t), lights: scene.lights })), async (f) => {
    if (f.t === 0) return;
    const r = new Float32Array((await rig.kernel.readReservoirs('final')).buffer);
    for (let y = 0; y < a.atlasH; y++) for (let x = 0; x < a.atlasW; x++) {
      const ai = y * a.atlasW + x, m = Math.floor(y / H) * a.memberCols + Math.floor(x / W);
      if (m >= E) continue;
      const o = ai * RES_WORDS;
      sums[m] += (0.2126 * r[o + RW.F] + 0.7152 * r[o + RW.F + 1] + 0.0722 * r[o + RW.F + 2]) * r[o + RW.W];
    }
  });
  rig.destroy();
  return Array.from(sums, (x) => x / ((T - 1) * W * H));
}
const meanVar = (x: number[]) => { const m = x.reduce((a, b) => a + b, 0) / x.length; return { m, v: x.reduce((a, b) => a + (b - m) ** 2, 0) / (x.length - 1) }; };

describe('B1 unbiasedness smoke: temporal chains ≡ canonical-only in expectation (camera motion)', () => {
  for (const [name, settings] of [['contribution', {}], ['recompute', { temporalCheck: 'recompute' }], ['talbot', { temporalMis: 'talbot' }]] as const) {
    it(name, async () => {
      const E = 32, W = 48, H = 48, T = 24;
      const cam = (t: number) => movedCamera(0.04 * t, 0, -0.01 * t, 0.01 * t);
      const tm = await memberMeans('temporal', E, W, H, T, cam, settings);
      const off = await memberMeans('initial', E, W, H, T, cam);
      const a = meanVar(tm), b = meanVar(off);
      const z = (a.m - b.m) / Math.sqrt(a.v / E + b.v / E);
      console.log(`[B1 smoke ${name}] temporal ${a.m.toFixed(6)} ± ${Math.sqrt(a.v / E).toExponential(2)}, off ${b.m.toFixed(6)} ± ${Math.sqrt(b.v / E).toExponential(2)}, rel ${((a.m - b.m) / b.m).toExponential(2)}, z ${z.toFixed(2)}`);
      expect(Math.abs(z)).toBeLessThan(4);
    });
  }
});

// ------------------------------------------------------------------------------------------------ T3-2 / T4-t (camera)

/** M4 T3 case bins (restir-shift.gpu.test.ts t3_case): letter by (d, k, technique, endpoint), × k = 2 | k > 2. */
export function t3CaseName(flags: number): string {
  const d = flags & 0xf, k = (flags >>> 4) & 0xf, tech = (flags >>> 8) & 3, ep = (flags >>> 10) & 7;
  let c: string;
  if (k === 0) c = tech === K.RS_TECH_BSDF_ENV ? '∅-env' : '∅-tri';
  else if (tech === K.RS_TECH_NEE && k === d) c = ep === 1 || ep === 2 ? 'a-delta' : ep === 3 || ep === 4 ? 'a-area' : ep === 0 ? 'a-tri' : ep === 5 ? 'a-sun' : 'f-env';
  else if (tech === K.RS_TECH_NEE && k === d - 1) c = ep === 7 ? 'b-env' : 'b';
  else if (k === d - 1) c = tech === K.RS_TECH_BSDF_ENV ? 'c-env' : 'c-tri';
  else if (k === d) c = tech === K.RS_TECH_BSDF_ENV ? 'e' : 'd';
  else c = tech === K.RS_TECH_NEE ? 'deep-nee' : 'deep-bsdf';
  const hi = k === 0 ? d > 2 : k > 2;
  return `${c}/${hi ? 'k>2' : 'k2'}`;
}
const halfToF = (h: number) => {
  const s = h & 0x8000 ? -1 : 1, e = (h >>> 10) & 0x1f, m = h & 0x3ff;
  if (e === 0) return s * m * 2 ** -24;
  if (e === 31) return m ? NaN : s * Infinity;
  return s * (1 + m / 1024) * 2 ** (e - 15);
};
const marginOf = (code: number) => halfToF(code >>> 16);

export interface RoundTripStats { trials: number; fwdOk: number; rtOk: number; logic: number; fp: number; fBad: number; jBad: number; codes: Record<string, number> }

/** Round trips of one test frame (forced s = p, robust mode): T⁻¹(T(X_p)) against X_p for every pixel with a valid q′
 *  and a defined forward shift. LOGIC: an inverse that is undefined / ZERO / OCCLUDED with a decisive margin, or
 *  F / J reciprocity > 1e-4; FP-BOUNDARY: the deciding pair's |margin| < 2⁻¹⁶ (M4 B-4). */
export function roundTrips(s: TemporalSnapshot, bins: Map<string, RoundTripStats>, viol: string[]): void {
  for (let ai = 0; ai < s.P; ai++) {
    const ts = s.ts(ai);
    if (!(ts.flags & T.TS_QVALID)) continue;
    const fl = s.rec(s.hist, ts.qPrime, RW.flags);
    if ((fl & 0xf) === 0) continue;                        // empty X_p
    const name = t3CaseName(fl);
    let b = bins.get(name);
    if (!b) { b = { trials: 0, fwdOk: 0, rtOk: 0, logic: 0, fp: 0, fBad: 0, jBad: 0, codes: {} }; bins.set(name, b); }
    b.trials++;
    const fsc = scOf(ts.fwdCode);
    b.codes[`f:${SC_NAMES[fsc]}`] = (b.codes[`f:${SC_NAMES[fsc]}`] ?? 0) + 1;
    if (fsc !== K.SC_OK) continue;
    b.fwdOk++;
    if (!(ts.flags & T.TS_INV_DONE)) { b.logic++; viol.push(`${name} ai${ai}: forward OK, no inverse`); continue; }
    const isc = scOf(ts.invCode);
    b.codes[`i:${SC_NAMES[isc]}`] = (b.codes[`i:${SC_NAMES[isc]}`] ?? 0) + 1;
    if (isc !== K.SC_OK) {
      const m = marginOf(ts.invCode);
      if (Math.abs(m) < 2 ** -16 && isc !== K.SC_ZERO && isc !== K.SC_OCCLUDED) b.fp++;
      else { b.logic++; if (viol.length < 24) viol.push(`${name} ai${ai} q'${ts.qPrime}: inverse ${SC_NAMES[isc]} pair ${(ts.invCode >>> 12) & 0xf} margin ${m} (fwd J ${u2f(ts.fwdJ)})`); }
      continue;
    }
    const Fx = [0, 1, 2].map((c) => s.recF(s.histF, ts.qPrime, RW.F + c));
    const eF = Math.max(...[0, 1, 2].map((c) => (Fx[c] === 0 && ts.invF[c] === 0 ? 0 : relErr(ts.invF[c], Fx[c]))));
    const eJ = Math.abs(Math.log(u2f(ts.fwdJ) * u2f(ts.invJ)));
    if (eF > 1e-4) b.fBad++;
    if (eJ > 1e-4) b.jBad++;
    if (eF > 1e-4 || eJ > 1e-4) { b.logic++; if (viol.length < 24) viol.push(`${name} ai${ai}: F rel ${eF.toExponential(2)} |log JJ⁻¹| ${eJ.toExponential(2)}`); continue; }
    b.rtOk++;
  }
}

/** t-select with every defined forward shift selected (TSEL_TRACE_FORCE_P, Changelog B-7). */
export function tselectTraceSource(): Record<string, string> {
  const src = shaderSources['passes/restir/t-select.wgsl'];
  const out = src.replace('const TSEL_TRACE_FORCE_P: bool = false;', 'const TSEL_TRACE_FORCE_P: bool = true;');
  if (out === src) throw new Error('t-select.wgsl: TSEL_TRACE_FORCE_P not found');
  return { 'passes/restir/t-select.wgsl': out };
}

export function reportBins(tag: string, bins: Map<string, RoundTripStats>): { trials: number; logic: number; fp: number; rtOk: number } {
  const tot = { trials: 0, logic: 0, fp: 0, rtOk: 0 };
  const lines = [...bins.entries()].sort(([a], [b]) => (a < b ? -1 : 1)).map(([n, b]) => {
    tot.trials += b.trials; tot.logic += b.logic; tot.fp += b.fp; tot.rtOk += b.rtOk;
    return `  ${n.padEnd(14)} trials=${b.trials} fwdOk=${b.fwdOk} rtOk=${b.rtOk} LOGIC=${b.logic} FP=${b.fp} F=${b.fBad} J=${b.jBad} ${JSON.stringify(b.codes)}`;
  });
  console.log(`[${tag}] ${JSON.stringify(tot)}\n${lines.join('\n')}`);
  return tot;
}

const T32_PAIRS = Number(import.meta.env?.VITE_T32_PAIRS ?? 12);

describe('T3-2 / T4-t (camera part): production round trips T⁻¹(T(X_p)) with forced s = p (robust mode)', () => {
  const cases: [string, (t: number) => CameraState, JitterMode][] = [
    ['translate', (t) => movedCamera(0.05 * t, 0.01 * t, 0, 0), JITTER_IID],
    ['rotate', (t) => movedCamera(0, 0, 0, 0.02 * t), JITTER_IID],
    ['zoom-in', (t) => movedCamera(0, 0, -0.04 * t, 0), JITTER_IID],
    ['zoom-out+fov', (t) => movedCamera(0, 0, 0.04 * t, 0, (55 + 0.5 * (t % 7)) * Math.PI / 180), JITTER_IID],
    ['translate, jitter off', (t) => movedCamera(0.05 * t, 0.01 * t, 0, 0), JITTER_NONE],
  ];
  for (const [name, cam, jitter] of cases) {
    it(name, async () => {
      const scene = allLightsScene();
      const rig = await restirRig(scene, 128, 128, { preset: 'temporal', settings: { maxBounces: 4, temporalCheck: 'robust' }, jitterMode: jitter, seed: 3201, extraSources: tselectTraceSource() });
      const bins = new Map<string, RoundTripStats>();
      const viol: string[] = [];
      const frames: ChainFrame[] = [];
      for (let i = 0; i < 2 * T32_PAIRS; i++) frames.push({ t: i, camera: cam(i), lights: scene.lights, reset: i % 2 === 0 });
      const res = await runChain(rig, frames, async (f, r) => {
        if (f.reset) return;
        expect(r.histValid).toBe(true);
        roundTrips(await temporalSnapshot(rig.kernel), bins, viol);
      });
      const tot = reportBins(`T3-2 ${name}`, bins);
      if (viol.length) console.log(viol.join('\n'));
      for (const r of res) expect([r.counters.rsc.tNonFinite, r.counters.rsc.tPendingLeft], `t=${r.t}`).toEqual([0, 0]);
      expect(tot.rtOk).toBeGreaterThan(1000);
      expect(tot.logic).toBe(0);
      expect(tot.fp / Math.max(tot.trials, 1)).toBeLessThanOrEqual(1e-5);
      rig.destroy();
    });
  }
});

// ------------------------------------------------------------------------------------------------ U-TR-2, U-TE-1

/** 16-frame script on allLightsScene: camera drift, point light 1 moving, spot 2 intensity steps, env rotation / tint. */
function scriptFrames(scene: ReturnType<typeof allLightsScene>, n: number): ChainFrame[] {
  return Array.from({ length: n }, (_, t) => ({
    t, camera: movedCamera(0.03 * t, 0, -0.01 * t, 0.004 * t),
    lights: animatedLights(scene.lights, t, [{ id: 1, dp: [0.03, 0, 0.02] }, { id: 2, power: (u) => (u >= 5 && u < 11 ? 2 : 1) }]),
    env: { params: { rotationZ: 0.02 * t, strength: t >= 9 ? 1.5 : 1, tint: [1, 1, 1], visibleToCamera: true }, mapId: 'env' },
  }));
}

describe('U-TR-2: a temporal chain with light and env changes is bitwise independent of row bands and item chunks', () => {
  it('16 frames, full preset (temporal + spatial), rowBand 0 vs 16 vs 40', async () => {
    const scene = allLightsScene();
    const hashes: string[][] = [];
    for (const band of [0, 16, 40]) {
      const rig = await restirRig(scene, 64, 64, { preset: 'full', settings: { maxBounces: 3 }, seed: 5301 });
      rig.kernel.rowBand = band;
      const h: string[] = [];
      await runChain(rig, scriptFrames(scene, 16), async (f) => {
        h.push(hashU32(await rig.kernel.readReservoirs('final')) + ':' + hashU32(await rig.kernel.readTemporalState()));
      });
      hashes.push(h);
      rig.destroy();
    }
    console.log(`[U-TR-2] ${hashes.map((h) => h[h.length - 1]).join(' ')}`);
    expect(hashes[1]).toEqual(hashes[0]);
    expect(hashes[2]).toEqual(hashes[0]);
  });
});

describe('U-TE-1: atlas member m of an E = 4 temporal ensemble ≡ the sequential chain with memberBase m', () => {
  it('8 frames with camera, light and env changes; reservoirs of every member bitwise', async () => {
    const scene = allLightsScene();
    const W = 32, H = 32, E = 4;
    const ens = await restirRig(scene, W, H, { preset: 'full', settings: { maxBounces: 3 }, members: E, seed: 5302 });
    const frames = scriptFrames(scene, 8);
    await runChain(ens, frames);
    const a = ens.kernel.resources.alloc;
    const ensRes = await ens.kernel.readReservoirs('final');
    ens.destroy();
    let mism = 0;
    for (let m = 0; m < E; m++) {
      const seq = await restirRig(scene, W, H, { preset: 'full', settings: { maxBounces: 3 }, memberBase: m, seed: 5302 });
      await runChain(seq, frames);
      const sr = await seq.kernel.readReservoirs('final');
      const ox = (m % a.memberCols) * W, oy = Math.floor(m / a.memberCols) * H;
      for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
        const ea = ((oy + y) * a.atlasW + ox + x) * RES_WORDS, sa = (y * W + x) * RES_WORDS;
        for (let w = 0; w < RES_WORDS; w++) if (ensRes[ea + w] !== sr[sa + w]) { mism++; break; }
      }
      seq.destroy();
    }
    console.log(`[U-TE-1] E=${E} ${W}×${H}: reservoir mismatches ${mism}`);
    expect(mism).toBe(0);
  });
});

// ------------------------------------------------------------------------------------------------ T3-2 / T4-t (lights)

type LightScript = (base: LightData[], t: number) => LightData[];
const withLight = (base: LightData[], id: number, f: (l: LightData) => LightData | undefined): LightData[] =>
  base.flatMap((l) => (l.id === id ? (f(l) ? [f(l)!] : []) : [l]));
const LIGHT_CASES: [string, LightScript, ((t: number) => { rotationZ: number; strength: number; tint: [number, number, number] }) | undefined][] = [
  ['moving point + rect', (b, t) => animatedLights(b, t, [{ id: 1, dp: [0.04, 0, 0.03] }, { id: 3, dp: [0.02, -0.01, 0] }]), undefined],
  ['resized + rotated rect', (b, t) => withLight(b, 3, (l) => ({ ...l, sizeX: (l.sizeX ?? 0.5) * (1 + 0.1 * t), matrix: lightMatrixToward([Math.sin(0.08 * t), -0.3, Math.cos(0.08 * t)], [0.3, 1.3, -1.9]) })), undefined],
  ['rotating spot', (b, t) => withLight(b, 2, (l) => ({ ...l, matrix: lightMatrixToward([0.2 + 0.05 * t, -1, -0.3], [-0.5, 1.8, 0.2]) })), undefined],
  ['intensity steps', (b, t) => animatedLights(b, t, [{ id: 2, power: (u) => (u % 2 ? 2 : 1) }, { id: 4, power: (u) => (u % 2 ? 0.5 : 1) }]), undefined],
  ['add / remove', (b, t) => (t % 2 ? withLight(b, 4, () => undefined) : withLight(b, 1, () => undefined)), undefined],
  ['env rotation + strength', (b) => b, (t) => ({ rotationZ: 0.05 * t, strength: t % 2 ? 1.5 : 1, tint: [1, 1, 1] })],
];

describe('T3-2 / T4-t (light part): production round trips under light and env changes (refresh C1/C2)', () => {
  for (const [name, lights, env] of LIGHT_CASES) {
    it(name, async () => {
      const scene = allLightsScene();
      const rig = await restirRig(scene, 128, 128, { preset: 'temporal', settings: { maxBounces: 4, temporalCheck: 'robust' }, jitterMode: JITTER_IID, seed: 3202, extraSources: tselectTraceSource() });
      const bins = new Map<string, RoundTripStats>();
      const viol: string[] = [];
      const frames: ChainFrame[] = [];
      for (let i = 0; i < 2 * T32_PAIRS; i++) {
        frames.push({
          t: i, camera: movedCamera(0.02 * i, 0, 0, 0), lights: lights(scene.lights, i), reset: i % 2 === 0,
          env: env ? { params: { ...env(i), visibleToCamera: true }, mapId: 'env' } : undefined,
        });
      }
      let refreshFrames = 0, classUndef = 0, lightUndef = 0;
      const res = await runChain(rig, frames, async (f, r) => {
        if (f.reset) return;
        expect(r.histValid).toBe(true);
        if (r.flags & K.TF_REFRESH) refreshFrames++;
        classUndef += r.counters.rsc.tClassUndef;
        lightUndef += r.counters.rsc.tLightUndef;
        roundTrips(await temporalSnapshot(rig.kernel), bins, viol);
      });
      const tot = reportBins(`T3-2 lights ${name}`, bins);
      console.log(`[T3-2 lights ${name}] refresh frames ${refreshFrames}, §9.3-6′ class-change (undefined) ${classUndef} = ${(classUndef / Math.max(tot.trials, 1)).toExponential(2)} per trial, light-undefined ${lightUndef}`);
      if (viol.length) console.log(viol.join('\n'));
      for (const r of res) expect([r.counters.rsc.tNonFinite, r.counters.rsc.tPendingLeft], `t=${r.t}`).toEqual([0, 0]);
      expect(refreshFrames).toBe(T32_PAIRS);
      expect(tot.rtOk).toBeGreaterThan(1000);
      expect(tot.logic).toBe(0);
      expect(tot.fp / Math.max(tot.trials, 1)).toBeLessThanOrEqual(1e-5);
      rig.destroy();
    });
  }
});
