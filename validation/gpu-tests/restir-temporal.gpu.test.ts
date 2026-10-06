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
import { RES_WORDS, RW, RS_WGSL_CONSTS as K, SC_NAMES, TS_CONSTS as T, arenaWords, decodeSfxLocal } from '../../src/core/render/restir/layout.ts';
import { restirSettings, type RestirSettings } from '../../src/core/render/restir/presets.ts';
import { ChainRunner, type ChainSpec } from '../../src/core/render/restir/chain-runner.ts';
import { BASE_ENV_MAP, fetchScenePackage, resolvePackageFrame } from '../../src/core/scene/scene-package.ts';
import { SceneGpu } from '../../src/core/render/scene-gpu.ts';
import { createEnvResources, destroyEnvResources, writeEnvParams } from '../../src/core/render/env-gpu.ts';
import { JITTER_IID, JITTER_NONE, computeRenderOrigin, type CameraState, type JitterMode } from '../../src/core/render/frame-uniforms.ts';
import { releaseTestGpu } from './device-factory.ts';
import { allLightsScene, bitFixtureScene, boxCamera, gpuScene, light, readTexture4, restirRig, storageBuffer } from './restir-fixtures.ts';
import { shaderSources } from '../../src/core/shaders/index.ts';
import type { LightData } from '../../src/core/scene/types.ts';
import { lightMatrixToward, material, quadScene } from './pt-fixtures.ts';
import { evalLocal, type V1Params } from '../../tests/material/bsdf-ref.ts';
import { recentrePositions } from '../../src/core/render/scene-gpu.ts';
import { T32Harness, animatedLights, hashU32, movedCamera, readU32, recLumF, runChain, temporalSnapshot, type ChainFrame, type FrameResult, type T32Result, type TemporalSnapshot } from './restir-temporal-fixtures.ts';
import type { SceneData } from '../../src/core/scene/types.ts';
import { loadHdri, synthEnvData } from './env-fixtures.ts';
import { t3M6Scene } from '../scenes/m6-fixtures.ts';
import { T3_ENV_ID, t3Scene } from '../scenes/make-m4.ts';
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


// ------------------------------------------------------------------------------------------------ T3-2 / T4-t

/** t-select with every defined forward shift selected (TSEL_TRACE_FORCE_P, Changelog B-7). */
export function tselectTraceSource(): Record<string, string> {
  const src = shaderSources['passes/restir/t-select.wgsl'];
  const out = src.replace('const TSEL_TRACE_FORCE_P: bool = false;', 'const TSEL_TRACE_FORCE_P: bool = true;');
  if (out === src) throw new Error('t-select.wgsl: TSEL_TRACE_FORCE_P not found');
  return { 'passes/restir/t-select.wgsl': out };
}

type LightScript = (base: LightData[], t: number) => LightData[];
type EnvScript = (t: number) => { rotationZ: number; strength: number; tint: [number, number, number] };
const withLight = (base: LightData[], id: number, f: (l: LightData) => LightData | undefined): LightData[] =>
  base.flatMap((l) => (l.id === id ? (f(l) ? [f(l)!] : []) : [l]));

/** Pairs (reset frame, test frame) per case and resolution. The M5 gate raises the `rare` cases (t3_rare_256: every
 *  bin populated) with VITE_T32_RARE_PAIRS / VITE_T32_RARE_RES so every bin of the camera, light and env change types
 *  reaches ≥ 10⁶ round trips (B-9; budget: restir-temporal-api.md B-11). */
const T32_PAIRS = Number(import.meta.env?.VITE_T32_PAIRS ?? 12);
const T32_RES = Number(import.meta.env?.VITE_T32_RES ?? 128);
const T32_RARE_PAIRS = Number(import.meta.env?.VITE_T32_RARE_PAIRS ?? T32_PAIRS);
const T32_RARE_RES = Number(import.meta.env?.VITE_T32_RARE_RES ?? T32_RES);
const T32_MIN_BIN = Number(import.meta.env?.VITE_T32_MIN_BIN ?? 0);

interface T32Case { name: string; scene: 'all' | 'rare' | 'modeb'; cam: (base: CameraState, t: number) => CameraState; jitter?: JitterMode; lights?: LightScript; env?: EnvScript; lightClass?: 'zero' | 'positive' }
const moved = (base: CameraState, dx: number, dy: number, dz: number, yaw = 0, yfov?: number): CameraState => {
  const m = Array.from(base.camToWorld as ArrayLike<number>);
  const cs = Math.cos(yaw), sn = Math.sin(yaw);
  for (const col of [0, 4, 8]) { const x = m[col], z = m[col + 2]; m[col] = cs * x + sn * z; m[col + 2] = -sn * x + cs * z; }
  m[12] += dx; m[13] += dy; m[14] += dz;
  return { camToWorld: m, yfov: yfov ?? base.yfov };
};
const T32_CASES_LIST: T32Case[] = [
  // camera part (frame selector with TF_LIGHTS_SAME, previous camera / V-buffer)
  { name: 'translate', scene: 'all', cam: (c, t) => moved(c, 0.05 * t, 0.01 * t, 0), lightClass: 'zero' },
  { name: 'rotate', scene: 'all', cam: (c, t) => moved(c, 0, 0, 0, 0.02 * t), lightClass: 'zero' },
  { name: 'zoom-in', scene: 'all', cam: (c, t) => moved(c, 0, 0, -0.04 * t), lightClass: 'zero' },
  { name: 'zoom-out+fov', scene: 'all', cam: (c, t) => moved(c, 0, 0, 0.04 * t, 0, c.yfov * (1 + 0.01 * (t % 7))), lightClass: 'zero' },
  { name: 'translate, jitter off', scene: 'all', cam: (c, t) => moved(c, 0.05 * t, 0.01 * t, 0), jitter: JITTER_NONE, lightClass: 'zero' },
  { name: 'rare bins: translate', scene: 'rare', cam: (c, t) => moved(c, 0.03 * t, 0, 0.01 * t), lightClass: 'zero' },
  // light / env part (refresh, entry renumbering, J_P)
  { name: 'moving point + rect', scene: 'all', cam: (c, t) => moved(c, 0.02 * t, 0, 0), lights: (b, t) => animatedLights(b, t, [{ id: 1, dp: [0.04, 0, 0.03] }, { id: 3, dp: [0.02, -0.01, 0] }]), lightClass: 'zero' },
  { name: 'resized + rotated rect', scene: 'all', cam: (c, t) => moved(c, 0.02 * t, 0, 0), lights: (b, t) => withLight(b, 3, (l) => ({ ...l, sizeX: (l.sizeX ?? 0.5) * (1 + 0.1 * t), matrix: lightMatrixToward([Math.sin(0.08 * t), -0.3, Math.cos(0.08 * t)], [0.3, 1.3, -1.9]) })), lightClass: 'zero' },
  { name: 'rotating spot', scene: 'all', cam: (c, t) => moved(c, 0.02 * t, 0, 0), lights: (b, t) => withLight(b, 2, (l) => ({ ...l, matrix: lightMatrixToward([0.2 + 0.05 * t, -1, -0.3], [-0.5, 1.8, 0.2]) })), lightClass: 'zero' },
  { name: 'intensity steps', scene: 'all', cam: (c, t) => moved(c, 0.02 * t, 0, 0), lights: (b, t) => animatedLights(b, t, [{ id: 2, power: (u) => (u % 2 ? 2 : 1) }, { id: 4, power: (u) => (u % 2 ? 0.5 : 1) }]), lightClass: 'zero' },
  { name: 'add / remove', scene: 'all', cam: (c, t) => moved(c, 0.02 * t, 0, 0), lights: (b, t) => (t % 2 ? withLight(b, 4, () => undefined) : withLight(b, 1, () => undefined)), lightClass: 'positive' },
  { name: 'env rotation + strength', scene: 'all', cam: (c, t) => moved(c, 0.02 * t, 0, 0), env: (t) => ({ rotationZ: 0.05 * t, strength: t % 2 ? 1.5 : 1, tint: [1, 1, 1] }), lightClass: 'zero' },
  { name: 'rare bins: add / remove + intensity', scene: 'rare', cam: (c, t) => moved(c, 0.02 * t, 0, 0), lights: (b: LightData[], t: number) => (t % 2 ? b.slice(1) : b.map((l) => ({ ...l, power: l.power * 1.5 }))), lightClass: 'positive' },
  // M6 (restir-m6-api.md MD8, R16): Mode-B temporal on t3_modeb_rare_256 (crossing entries renumbered, refresh of deep /
  // B1 crossing ends with a ray iff the light moved, (d-ana) points carried rigidly, ∅ crossings replayed under t−1)
  { name: 'Mode B: camera translate', scene: 'modeb', cam: (c, t) => moved(c, 0.03 * t, 0, 0.01 * t), lightClass: 'zero' },
  { name: 'Mode B: moving + rotating crossing lights', scene: 'modeb', cam: (c, t) => moved(c, 0.02 * t, 0, 0), lights: (b, t) => animatedLights(withLight(b, 6, (l) => ({ ...l, matrix: lightMatrixToward([Math.sin(0.06 * t), -0.1, Math.cos(0.06 * t)], [0.05 + 0.01 * t, 0.95, -0.55]) })), t, [{ id: 7, dp: [0.02, 0, -0.01] }, { id: 8, dp: [0, 0.01, 0.01] }]), lightClass: 'zero' },
  { name: 'Mode B: add / remove + intensity (crossing lights)', scene: 'modeb', cam: (c, t) => moved(c, 0.02 * t, 0, 0), lights: (b: LightData[], t: number) => (t % 2 ? b.filter((l) => l.id !== 7) : b.map((l) => (l.id === 6 ? { ...l, power: l.power * 1.5 } : l))), lightClass: 'positive' },
  { name: 'rare bins: moving lights + env', scene: 'rare', cam: (c, t) => moved(c, 0.02 * t, 0, 0), lights: (b: LightData[], t: number) => b.map((l) => animatedLights([l], t, [{ id: l.id, dp: [0.02, 0, 0.01], power: (u) => (u % 3 ? 1 : 1.5) }])[0]), env: (t) => ({ rotationZ: 0.03 * t, strength: 1, tint: [1, 1, 1] }), lightClass: 'zero' },
];

async function t32Scene(which: 'all' | 'rare' | 'modeb'): Promise<{ scene: SceneData; cam: CameraState; maxBounces: number }> {
  if (which === 'all') return { scene: allLightsScene(), cam: boxCamera(), maxBounces: 4 };
  const env = (await loadHdri(`${T3_ENV_ID}_1k.hdr`)) ?? synthEnvData(256, 128);
  if (which === 'modeb') {
    const m = t3M6Scene('t3_modeb_rare_256', env);
    return { scene: m.scene, cam: { camToWorld: m.camera.matrix, yfov: m.camera.yfov }, maxBounces: m.maxBounces };
  }
  const t = t3Scene('t3_rare_256', env);
  return { scene: t.scene, cam: { camToWorld: t.camera.matrix, yfov: t.camera.yfov }, maxBounces: t.maxBounces };
}

export async function runT32(c: T32Case, pairs = T32_PAIRS, W = T32_RES): Promise<{ r: T32Result; ms: number; lightClass: number; refreshFrames: number; frames: FrameResult[] }> {
  const s = await t32Scene(c.scene);
  const rig = await restirRig(s.scene, W, W, { preset: 'temporal', settings: { maxBounces: s.maxBounces, temporalCheck: 'robust' }, jitterMode: c.jitter ?? JITTER_IID, cam: { camToWorld: Array.from(s.cam.camToWorld as ArrayLike<number>), yfov: s.cam.yfov }, seed: 3201, extraSources: tselectTraceSource(), lightMode: c.scene === 'modeb' ? 'B' : 'A' });
  const h = await T32Harness.create(rig.kernel);
  const frames: ChainFrame[] = [];
  for (let i = 0; i < 2 * pairs; i++) {
    // camera and light scripts are periodic (40 frames; a wrap falls between a test frame and the next reset frame), so
    // long gate runs keep the view inside the scene
    const ic = i % 40;
    frames.push({ t: i, camera: c.cam(s.cam, ic), lights: c.lights ? c.lights(s.scene.lights, ic) : s.scene.lights, reset: i % 2 === 0,
      env: c.env ? { params: { ...c.env(i), visibleToCamera: true }, mapId: 'env' } : undefined });
  }
  let lightClass = 0, refreshFrames = 0;
  const bad: FrameResult[] = [];
  const t0 = performance.now();
  await runChain(rig, frames, (f, r) => {
    lightClass += r.counters.rsc.tLightClass;
    if (r.counters.rsc.tNonFinite || r.counters.rsc.tPendingLeft) bad.push(r);
    if (f.reset) return;
    if (r.flags & K.TF_REFRESH) refreshFrames++;
    h.frame(f.t);
  }, false);
  const ms = performance.now() - t0;
  const r = await h.result();
  h.destroy(); rig.destroy();
  return { r, ms, lightClass, refreshFrames, frames: bad };
}

function reportT32(tag: string, x: Awaited<ReturnType<typeof runT32>>, pairs: number): void {
  const lines = x.r.bins.filter((b) => b.trials).map((b) => `  ${b.name.padEnd(14)} trials=${b.trials} fwdOk=${b.fwdOk} rtOk=${b.rtOk} LOGIC=${b.logic} FP=${b.fp} F=${b.fBad} J=${b.jBad} fwd=${JSON.stringify(b.fwd)} inv=${JSON.stringify(b.inv)}`);
  console.log(`[T3-2 ${tag}] ${JSON.stringify(x.r.total)} platform ${x.r.platform} lightClass ${x.lightClass} refresh ${x.refreshFrames}/${pairs} ${(x.ms / pairs).toFixed(1)} ms/pair\n${lines.join('\n')}${x.r.logicSamples.length ? '\n  LOGIC: ' + x.r.logicSamples.join('\n  LOGIC: ') : ''}`);
}

describe('T3-2 / T4-t: production round trips T⁻¹(T(X_p)) with forced s = p (robust mode), camera and light / env changes', () => {
  for (const c of T32_CASES_LIST) {
    it(c.name, async () => {
      const rare = c.scene !== 'all';
      const pairs = rare ? T32_RARE_PAIRS : T32_PAIRS;
      const x = await runT32(c, pairs, rare ? T32_RARE_RES : T32_RES);
      reportT32(c.name, x, pairs);
      if (c.scene === 'rare' && T32_MIN_BIN > 0) for (const b of x.r.bins) expect(b.rtOk, b.name).toBeGreaterThanOrEqual(T32_MIN_BIN);
      if (c.scene === 'modeb' && T32_MIN_BIN > 0) {
        for (const b of x.r.bins.filter((bb) => /ana/.test(bb.name))) if (b.trials) expect(b.rtOk, b.name).toBeGreaterThanOrEqual(T32_MIN_BIN / 10);
      }
      for (const r of x.frames) expect([r.counters.rsc.tNonFinite, r.counters.rsc.tPendingLeft], `t=${r.t}`).toEqual([0, 0]);
      if (c.lights || c.env) expect(x.refreshFrames).toBe(pairs);
      if (c.lightClass === 'zero') expect(x.lightClass).toBe(0);
      if (c.lightClass === 'positive') expect(x.lightClass).toBeGreaterThan(0);
      expect(x.r.candOverflow).toBe(0);
      expect(x.r.platform).toBe(0);
      expect(x.r.total.rtOk).toBeGreaterThan(1000);
      expect(x.r.total.logic).toBe(0);
      expect(x.r.total.fp / Math.max(x.r.total.trials, 1)).toBeLessThanOrEqual(1e-5);
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


// ------------------------------------------------------------------------------------------------ T6(b): ixs sequences

const IXS = ['ixs_a_point_256', 'ixs_b_area_256', 'ixs_c_spot_b0_256', 'ixs_c_spot_b03_256', 'ixs_d_camera_256', 'ixs_d0_jitter_256',
  'ixs_d_glossy_256', 'ixs_e_addremove_256', 'ixs_e_half_256', 'ixs_f_combined_256', 'ixs_g_sun_256', 'ixs_n4_twolights_256'];
const IXS_ENV = ['ixs_h_envrot_256', 'ixs_i_envradio_256', 'ixs_j_envcombo_256', 'ixs_k_envswap_256'];   // validation/out/m5/scenes
const T6B_MEMBERS = Number(import.meta.env?.VITE_T6B_MEMBERS ?? 2);

interface RobustFrame { t: number; histValid: boolean; flags: number; robust: number; mism: number; selP: number }

/** One robust-mode chain batch (E members, `full` preset) through a sequence package with T-E's ChainRunner; per frame
 *  the robust checks (= temporal selections, RSC_T_SEL_P) and RSC_T_ROBUST_MISMATCH from the cumulative counters. */
async function robustSequence(pkgName: string, o: { tPlant?: RestirSettings['tPlant']; frames?: number } = {}): Promise<{ frames: RobustFrame[]; testFrames: number[] }> {
  const base = IXS_ENV.includes(pkgName) ? '/validation/out/m5/scenes/' : '/validation/scenes/';
  const p = await fetchScenePackage(`${base}${pkgName}/`);
  const seq = p.sequence!;
  const { device, features, wgslLanguageFeatures } = await (await import('./device-factory.ts')).getTestGpu();
  const origin = computeRenderOrigin(p.scene.bounds, p.scene.quant);
  const gpu = await SceneGpu.create(device, p.scene, origin, { textureMode: 'validation', watertight: true, features, wgslLanguageFeatures });
  const f0 = resolvePackageFrame(p, 0);
  let envMapId = f0.env?.mapId ?? BASE_ENV_MAP;
  let env = await createEnvResources(device, f0.env?.map ?? p.scene.env);
  const envs = [env];
  if (f0.env) writeEnvParams(device, env, f0.env.params);
  const settings = restirSettings('full', { maxBounces: p.render.maxBounces ?? 3, temporalCheck: 'robust', ...(o.tPlant ? { tPlant: o.tPlant } : {}) });
  const k = await RestirKernel.create(device, gpu, env, { settings, lightMode: 'A', env: { nee: (p.json.env?.sampling ?? 'AUTOMATIC') !== 'NONE' }, features, wgslLanguageFeatures });
  const W = p.render.width, H = p.render.height;
  k.setView({ camera: { camToWorld: Array.from(f0.camera.camToWorld), yfov: f0.camera.yfov }, width: W, height: H, runSeed: 7002, members: T6B_MEMBERS, jitterMode: JITTER_IID });
  await k.prepare();
  const nFrames = o.frames ?? Math.max(...seq.testFrames) + 1;
  const frames: RobustFrame[] = [];
  let last = { selP: 0, mism: 0 };
  const runner = new ChainRunner(k, { runSeed: 7002 });
  const spec: ChainSpec = {
    frames: nFrames, testFrames: [],
    state: (t) => {
      const r = resolvePackageFrame(p, t);
      return { t, camera: { camToWorld: Array.from(r.camera.camToWorld), yfov: r.camera.yfov }, lights: r.lights, ...(r.env ? { env: { params: r.env.params, mapId: r.env.mapId } } : {}) };
    },
    beforeFrame: async (t) => {
      const r = resolvePackageFrame(p, t);
      if (r.env && r.env.mapId !== envMapId) {
        env = await createEnvResources(device, r.env.map);
        envs.push(env);
        writeEnvParams(device, env, r.env.params);
        k.setEnvironment(env);
        envMapId = r.env.mapId;
      }
    },
    afterFrame: async (t, kk) => {
      const c = await kk.readCounters(false);
      const adv = kk.currentAdvance!;
      frames.push({ t, histValid: adv.histValid, flags: adv.flags, robust: c.rsc.tSelP - last.selP, mism: c.rsc.tRobustMismatch - last.mism, selP: c.rsc.tSelP });
      last = { selP: c.rsc.tSelP, mism: c.rsc.tRobustMismatch };
    },
  };
  await runner.runBatch(spec, 0, 0, () => undefined);
  k.destroy(); gpu.destroy();
  for (const e of envs) destroyEnvResources(e);
  return { frames, testFrames: seq.testFrames };
}

describe('T6(b): robust mode (stored route ≡ π_p(Y_p) recomputed by E_{t−1}) on every ixs sequence at its test frames', () => {
  for (const pkg of [...IXS, ...IXS_ENV]) {
    it(pkg, async () => {
      const { frames, testFrames } = await robustSequence(pkg);
      const at = frames.filter((f) => testFrames.includes(f.t));
      const robust = at.reduce((a, f) => a + f.robust, 0), mism = at.reduce((a, f) => a + f.mism, 0);
      const allR = frames.reduce((a, f) => a + f.robust, 0), allM = frames.reduce((a, f) => a + f.mism, 0);
      console.log(`[T6(b) ${pkg}] test frames ${testFrames.join(',')}: robust ${robust} mismatch ${mism}; all frames: robust ${allR} mismatch ${allM}; per test frame ${JSON.stringify(at.map((f) => [f.t, f.robust, f.mism]))}`);
      expect(robust).toBeGreaterThan(0);
      expect(mism / robust).toBeLessThanOrEqual(1e-5);
      expect(allM / Math.max(allR, 1)).toBeLessThanOrEqual(1e-5);
    });
  }
});

describe('T6(b) planted: N1-mixed fails exactly on the light-change frames; N3 fails after a move (pending T-C, B-10)', () => {
  it('N1-mixed on ixs_e_addremove: mismatches iff the frame changes lights (8, 14, 20), 0 on every other history frame', async () => {
    const { frames } = await robustSequence('ixs_e_addremove_256', { tPlant: { n1Mixed: true }, frames: 24 });
    const changed = frames.filter((f) => f.histValid && !(f.flags & K.TF_LIGHTS_SAME));
    const same = frames.filter((f) => f.histValid && (f.flags & K.TF_LIGHTS_SAME));
    console.log(`[T6(b) N1-mixed] changed ${JSON.stringify(changed.map((f) => [f.t, f.robust, f.mism]))}; unchanged mismatches ${same.reduce((a, f) => a + f.mism, 0)} of ${same.reduce((a, f) => a + f.robust, 0)}`);
    expect(changed.map((f) => f.t)).toEqual([8, 14, 20]);
    for (const f of changed) expect(f.mism, `t=${f.t}`).toBeGreaterThan(0.01 * f.robust);
    expect(same.reduce((a, f) => a + f.mism, 0)).toBe(0);
  });
});

// ------------------------------------------------------------------------------------------------ T6(c)

/** Provenance hooks (T6(c) part i, restir-temporal-api.md §6.1): every frame-state accessor ORs a (kind, frame) bit into
 *  a private mask (kind 0 camera, 1 G-buffer, 2 light slot, 3 env record; frame bit 1 = t−1); the temporal passes store
 *  the mask of their shift call in tState word xpEntry (forward: bits 0–7, inverse: bits 16–23). Test-only sources. */
function provenanceSources(): Record<string, string> {
  const sub = (file: string, pairs: [string, string][]) => {
    let s = shaderSources[file];
    for (const [a, b] of pairs) { if (!s.includes(a)) throw new Error(`${file}: '${a}' not found`); s = s.replace(a, b); }
    return s;
  };
  const P = (k: number, prev: string) => `prov(${k}u, ${prev});`;
  return {
    'restir/frame.wgsl': sub('restir/frame.wgsl', [
      ['#include "restir/types.wgsl"', '#include "restir/types.wgsl"\nvar<private> provMask: u32;\nfn prov(k: u32, prev: bool) { provMask |= 1u << (2u * k + select(0u, 1u, prev)); }'],
      ['fn rs_vbuf(px: vec2u) -> vec4u { return', `fn rs_vbuf(px: vec2u) -> vec4u { ${P(1, 'false')} return`],
      ['fn rs_geo(px: vec2u) -> vec4f { return', `fn rs_geo(px: vec2u) -> vec4f { ${P(1, 'false')} return`],
      ['fn rs_vbuf_prev(px: vec2u) -> vec4u { return', `fn rs_vbuf_prev(px: vec2u) -> vec4u { ${P(1, 'true')} return`],
      ['fn rs_geo_prev(px: vec2u) -> vec4f { return', `fn rs_geo_prev(px: vec2u) -> vec4f { ${P(1, 'true')} return`],
      ['fn rs_cam_pos() -> vec3f { return', `fn rs_cam_pos() -> vec3f { ${P(0, 'false')} return`],
    ]),
    'restir/tframe.wgsl': sub('restir/tframe.wgsl', [
      ['fn lf_slot(fs: u32) -> LightSlot {', `fn lf_slot(fs: u32) -> LightSlot {\n  ${P(2, 'fs == RS_FS_PREV')}`],
      ['fn lf_env(fs: u32) -> EnvParams {', `fn lf_env(fs: u32) -> EnvParams {\n  ${P(3, 'fs == RS_FS_PREV')}`],
      ['fn lf_cam_pos(fs: u32) -> vec3f {', `fn lf_cam_pos(fs: u32) -> vec3f {\n  ${P(0, 'fs == RS_FS_PREV')}`],
    ]),
    'passes/restir/t-classify.wgsl': sub('passes/restir/t-classify.wgsl', [
      ['    let o = temporal_shift(src, tdst_cur(p));', '    provMask = 0u;\n    let o = temporal_shift(src, tdst_cur(p));\n    ts_store(p.ai, TSW_XPENTRY, provMask);'],
    ]),
    'passes/restir/t-forward.wgsl': sub('passes/restir/t-forward.wgsl', [
      ['  let o = temporal_shift(', '  provMask = 0u;\n  let o = temporal_shift('],
      ['  ts_store_fwd(q, o);', '  ts_store_fwd(q, o);\n  ts_store(q, TSW_XPENTRY, provMask);'],
    ]),
    'passes/restir/t-inverse.wgsl': sub('passes/restir/t-inverse.wgsl', [
      ['  let o = temporal_shift(src, tdst_prev(qP));', '  provMask = 0u;\n  let o = temporal_shift(src, tdst_prev(qP));\n  ts_store(q, TSW_XPENTRY, (ts_load(q, TSW_XPENTRY) & 0xFFFFu) | (provMask << 16u));'],
    ]),
  };
}

/** A V1-only box (Lambert + one glossy V1 wall) with point, spot and rect lights: the f64 cross-evaluator scene. */
const XE_MATS: V1Params[] = [
  { model: 0, diffuse: [0.6, 0.6, 0.6], glossy: [0, 0, 0], roughness: 0.5, mix: 0 },
  { model: 0, diffuse: [0.5, 0.45, 0.4], glossy: [0.8, 0.8, 0.8], roughness: 0.3, mix: 0.5 },
  { model: 0, diffuse: [0.2, 0.5, 0.3], glossy: [0, 0, 0], roughness: 0.5, mix: 0 },
  { model: 0, diffuse: [0.7, 0.3, 0.2], glossy: [0, 0, 0], roughness: 0.5, mix: 0 },
];
function xeScene(t: number): SceneData {
  const mats = XE_MATS.map((m) => material({ baseColorFactor: [...m.diffuse, 1], v1: { diffuse: m.diffuse, glossy: m.glossy, roughness: m.roughness, mix: m.mix } }));
  const quads = [
    { p: [[-1.5, 0, 1], [1.5, 0, 1], [1.5, 0, -2], [-1.5, 0, -2]], mat: 0 },
    { p: [[-1.5, 0, -2], [1.5, 0, -2], [1.5, 2, -2], [-1.5, 2, -2]], mat: 1 },
    { p: [[-1.5, 0, 1], [-1.5, 0, -2], [-1.5, 2, -2], [-1.5, 2, 1]], mat: 2 },
    { p: [[1.5, 0, -2], [1.5, 0, 1], [1.5, 2, 1], [1.5, 2, -2]], mat: 3 },
  ];
  const down = (p: [number, number, number]) => lightMatrixToward([0, -1, 0], p);
  return quadScene(quads, mats, [
    light({ id: 1, type: 'point', power: 30, matrix: down([0.6 - 0.05 * t, 1.6, -0.3]) }),
    light({ id: 2, type: 'spot', power: 50, spotSize: 1.0, spotBlend: 0.2, matrix: lightMatrixToward([0.2, -1, -0.3], [-0.5, 1.8, 0.2]) }),
    light({ id: 3, type: 'rect', power: t % 2 ? 60 : 40, sizeX: 0.5, sizeY: 0.3, matrix: lightMatrixToward([0, -0.3, 1], [0.3, 1.3, -1.9]) }),
  ]);
}

/** f64 F_{t−1} of a class-L path (d = 2, forced NEE on an analytic light) from explicit vertices: previous camera c,
 *  y₁′ (ids of the previous V-buffer), the light point of the translated entry under the frame-(t−1) records and pmf.
 *  Mode A analytic: ω1 = 1. Written from math.md#measure / #units-lights, independent of the WGSL. */
function xeEvalF(o: { cam: number[]; pos: Float32Array; idx: Uint32Array; triMat: Uint32Array; rec: Uint32Array; slot: { lightOff: number; pmfOff: number }; prim: number; bu: number; bv: number; entry: number; eu: number; ev: number }): { F: [number, number, number]; kappa: number } | undefined {
  const f = new Float32Array(o.rec.buffer, o.rec.byteOffset, o.rec.length);
  const P = (i: number) => [o.pos[3 * i], o.pos[3 * i + 1], o.pos[3 * i + 2]];
  const [a, b, c] = [0, 1, 2].map((j) => P(o.idx[3 * o.prim + j]));
  const w = 1 - o.bu - o.bv;
  const y = [0, 1, 2].map((j) => w * a[j] + o.bu * b[j] + o.bv * c[j]);
  const e1 = [0, 1, 2].map((j) => b[j] - a[j]), e2 = [0, 1, 2].map((j) => c[j] - a[j]);
  let ng = [e1[1] * e2[2] - e1[2] * e2[1], e1[2] * e2[0] - e1[0] * e2[2], e1[0] * e2[1] - e1[1] * e2[0]];
  const nrm = (v: number[]) => { const l = Math.hypot(v[0], v[1], v[2]); return v.map((x) => x / l); };
  const dt = (u: number[], v: number[]) => u[0] * v[0] + u[1] * v[1] + u[2] * v[2];
  ng = nrm(ng);
  const V = nrm([0, 1, 2].map((j) => o.cam[j] - y[j]));
  if (dt(ng, V) < 0) ng = ng.map((x) => -x);
  const L0 = o.slot.lightOff + o.entry * 28;
  const rv = (i: number) => [f[L0 + i], f[L0 + i + 1], f[L0 + i + 2]];
  const kind = o.rec[L0 + 3], lpos = rv(0), axisU = rv(4), axisV = rv(8), normal = rv(12), emit = rv(16);
  const halfU = f[L0 + 7], halfV = f[L0 + 11], cosHalf = f[L0 + 20], spotSmooth = f[L0 + 21], spreadNorm = f[L0 + 22], tanHalf = f[L0 + 23], invArea = f[L0 + 25];
  const pmf = f[o.slot.pmfOff + o.entry];
  let z = lpos;
  if (kind === 3) z = [0, 1, 2].map((j) => lpos[j] + (2 * o.eu - 1) * halfU * axisU[j] + (2 * o.ev - 1) * halfV * axisV[j]);
  else if (kind !== 1 && kind !== 2) return undefined;
  const d = [0, 1, 2].map((j) => z[j] - y[j]);
  const r2 = dt(d, d), r = Math.sqrt(r2);
  const L = d.map((x) => x / r);
  let lam: number[], q: number;
  // f32 conditioning of the evaluation (as M4 U-11): relative sensitivity to ulp-level position / direction errors
  let kappa = 1 / Math.max(Math.abs(dt(ng, L)), 1e-12) + 1 / Math.max(Math.abs(dt(ng, V)), 1e-12);
  if (kind === 1) { lam = emit.map((x) => x / r2); q = pmf; }
  else if (kind === 2) {
    const ct = dt(L.map((x) => -x), normal);
    let S: number;
    if (spotSmooth < 0) S = ct > cosHalf ? 1 : 0;
    else {
      const s = (ct - cosHalf) * spotSmooth; S = s <= 0 ? 0 : s >= 1 ? 1 : s * s * (3 - 2 * s);
      if (s > 0 && s < 1) kappa += (6 * s * (1 - s) * spotSmooth) / Math.max(S, 1e-12);   // |d ln S / d cosθ'|
    }
    lam = emit.map((x) => (x * S) / r2); q = pmf;
  } else {
    const cz = dt(L.map((x) => -x), normal);
    if (!(cz > 0)) return { F: [0, 0, 0], kappa };
    kappa += 1 / cz;
    const spread = spreadNorm < 0 ? 1 : Math.max((tanHalf - Math.sqrt(Math.max(1 - cz * cz, 0)) / cz) * spreadNorm, 0);
    lam = emit.map((x) => (x * spread * cz) / r2); q = pmf * invArea;
  }
  // local frame at y (N = ng)
  const tA = Math.abs(ng[0]) > 0.9 ? [0, 1, 0] : [1, 0, 0];
  const T1 = nrm([ng[1] * tA[2] - ng[2] * tA[1], ng[2] * tA[0] - ng[0] * tA[2], ng[0] * tA[1] - ng[1] * tA[0]]);
  const B1 = [ng[1] * T1[2] - ng[2] * T1[1], ng[2] * T1[0] - ng[0] * T1[2], ng[0] * T1[1] - ng[1] * T1[0]];
  const loc = (v: number[]): [number, number, number] => [dt(v, T1), dt(v, B1), dt(v, ng)];
  const ev = evalLocal(XE_MATS[o.triMat[o.prim]], loc(V), loc(L));
  return { F: [0, 1, 2].map((j) => ((ev.fD[j] + ev.fS[j]) * lam[j]) / q) as [number, number, number], kappa };
}

describe('T6(c): previous-state provenance and the f64 cross-evaluator', () => {
  it('(i) provenance: forward shifts read only frame-t state, inverse shifts only frame t−1 state; gensPrev(t) = gens(t−1)', async () => {
    const scene = allLightsScene();
    const rig = await restirRig(scene, 64, 64, { preset: 'temporal', settings: { maxBounces: 3 }, seed: 6101, extraSources: provenanceSources() });
    const frames: ChainFrame[] = Array.from({ length: 8 }, (_, t) => ({ t, camera: movedCamera(0.03 * t, 0, 0, 0.01 * t), lights: animatedLights(scene.lights, t, [{ id: 1, dp: [0.03, 0, 0] }, { id: 3, power: (u) => (u % 2 ? 1.5 : 1) }]) }));
    const gens: number[][] = [];
    const st = { fwd: 0, inv: 0, fwdBad: 0, invBad: 0, fwdLight: 0, invLight: 0 };
    const bad: string[] = [];
    const CUR = 0x55, PREV = 0xaa;
    await runChain(rig, frames, async (f, r) => {
      const a = rig.kernel.currentAdvance!;
      gens.push([...a.temporal.gens]);
      if (f.t > 0) expect(a.temporal.gensPrev, `t=${f.t}`).toEqual(gens[f.t - 1]);
      if (!r.histValid) return;
      const s = await temporalSnapshot(rig.kernel);
      for (let ai = 0; ai < s.P; ai++) {
        const ts = s.ts(ai);
        if (ts.flags & T.TS_FWD_DONE) {
          if (scOf(ts.fwdCode) === K.SC_EMPTY_SRC || scOf(ts.fwdCode) === K.SC_O0_LIGHT) continue;
          const m = ts.xpEntry & 0xff;
          st.fwd++;
          if ((m & PREV) || !(m & 0x1) || !(m & 0x4)) { st.fwdBad++; if (bad.length < 8) bad.push(`t${f.t} ai${ai} fwd mask ${m.toString(2)}`); }
          if (m & 0x10) st.fwdLight++;
        }
        if (ts.flags & T.TS_INV_DONE) {
          const m = (ts.xpEntry >>> 16) & 0xff;
          st.inv++;
          if ((m & CUR) || !(m & 0x2) || !(m & 0x8)) { st.invBad++; if (bad.length < 8) bad.push(`t${f.t} ai${ai} inv mask ${m.toString(2)}`); }
          if (m & 0x20) st.invLight++;
        }
      }
    });
    console.log(`[T6(c)-i] ${JSON.stringify(st)}${bad.length ? '\n  ' + bad.join('\n  ') : ''}`);
    expect(st.fwd).toBeGreaterThan(1000);
    expect(st.inv).toBeGreaterThan(1000);
    expect(st.fwdLight).toBeGreaterThan(0);
    expect(st.invLight).toBeGreaterThan(0);
    expect(st.fwdBad).toBe(0);
    expect(st.invBad).toBe(0);
    rig.destroy();
  });

  it('(ii) cross-evaluator: T⁻¹(X_c) as explicit vertices, f64 under S_{t−1} ≡ π_p(X_c)/J_inv = F_{t−1} (1e-4), ≥ 10⁵ canonicals', async () => {
    const W = 192;
    const frames: ChainFrame[] = Array.from({ length: 100 }, (_, t) => ({ t, camera: movedCamera(0.02 * t, 0, 0, 0.005 * t), lights: xeScene(t).lights }));
    const rig = await restirRig(xeScene(0), W, W, { preset: 'temporal', settings: { maxBounces: 1, temporalMis: 'talbot' }, seed: 6102 });
    const g = rig.kernel.lights.state as unknown as { records: Uint32Array; prevSlot: { lightOff: number; pmfOff: number }; translate(e: number, f: 'prev' | 'cur', t: 'prev' | 'cur', same?: boolean): number };
    const origin = rig.g.gpu.origin as unknown as number[];
    const pos = recentrePositions(rig.g.gpu.scene.geometry.positions, rig.g.gpu.origin);
    const idx = rig.g.gpu.scene.geometry.indices, triMat = rig.g.gpu.scene.geometry.triMaterial;
    const st = { n: 0, bad: 0, fp: 0, worst: 0, skipped: 0 };
    const bad: string[] = [];
    await runChain(rig, frames, async (f, r) => {
      if (!r.histValid) return;
      const k = rig.kernel, res = k.resources;
      const s = await temporalSnapshot(k);
      const vbPrev = await readTexture4(rig.g.device, res.vbufPrev);
      const words = await k.readTemporalState();
      const cm = frames[f.t - 1].camera.camToWorld as ArrayLike<number>;
      const cam = [cm[12] - origin[0], cm[13] - origin[1], cm[14] - origin[2]];
      const same = !!(r.flags & K.TF_LIGHTS_SAME);
      for (let ai = 0; ai < s.P; ai++) {
        const ts = s.ts(ai);
        if (!(ts.flags & T.TS_SEL_C) || !(ts.flags & T.TS_INV_DONE) || scOf(ts.invCode) !== K.SC_OK) continue;
        const fl = s.rec(s.out, ai, RW.flags);
        if ((fl & 0xf) !== 2 || ((fl >>> 4) & 0xf) !== 2 || ((fl >>> 8) & 3) !== K.RS_TECH_NEE) { st.skipped++; continue; }
        const eCur = s.rec(s.out, ai, RW.end) & K.RC_ENTRY_MASK;
        const sfx = decodeSfxLocal(words, s.P, s.NS, 1, ai);
        const ePrev = r.flags & K.TF_REFRESH ? sfx.entryTo & K.RC_ENTRY_MASK : g.translate(eCur, 'cur', 'prev', same);
        const qp = ts.qPrime;
        const xe = xeEvalF({ cam, pos, idx, triMat, rec: g.records, slot: g.prevSlot, prim: vbPrev[4 * qp], bu: u2f(vbPrev[4 * qp + 1]), bv: u2f(vbPrev[4 * qp + 2]), entry: ePrev, eu: u2f(s.rec(s.out, ai, RW.end + 1)), ev: u2f(s.rec(s.out, ai, RW.end + 2)) });
        if (!xe) { st.skipped++; continue; }
        const F = xe.F;
        st.n++;
        const e = Math.max(...[0, 1, 2].map((c) => (F[c] === 0 && ts.invF[c] === 0 ? 0 : relErr(F[c], ts.invF[c]))));
        const tol = Math.max(1e-4, 64 * 2 ** -23 * xe.kappa);   // 1e-4, or the f32 conditioning (M4 U-11 rule)
        st.worst = Math.max(st.worst, e);
        if (e > 1e-4) st.fp++;
        if (e > tol) { st.bad++; if (bad.length < 8) bad.push(`t${f.t} ai${ai} f64 ${F.map((x) => x.toExponential(5))} gpu ${ts.invF.map((x) => x.toExponential(5))} κ ${xe.kappa.toExponential(2)}`); }
        void (ts.piRecomp / u2f(ts.invJ));                 // π_p(X_c)/J_inv = lum(invF) by construction (ts_store_inv)
      }
    });
    console.log(`[T6(c)-ii] ${JSON.stringify(st)}${bad.length ? '\n  ' + bad.join('\n  ') : ''}`);
    expect(st.n).toBeGreaterThanOrEqual(1e5);
    expect(st.bad).toBe(0);
    rig.destroy();
  });
});

// ------------------------------------------------------------------------------------------------ U8 plant activity

/** Final reservoirs + mean image of a short `full` chain (camera drift, light script) with the given plant settings. */
async function plantChain(scene: SceneData, maxBounces: number, lights: (t: number) => LightData[], settings: Partial<RestirSettings>): Promise<{ res: string; mean: number }> {
  const rig = await restirRig(scene, 64, 64, { preset: 'full', settings: { maxBounces, ...settings }, seed: 7301 });
  let mean = 0;
  await runChain(rig, Array.from({ length: 8 }, (_, t) => ({ t, camera: movedCamera(0.02 * t, 0, 0), lights: lights(t) })), async (f) => {
    if (f.t !== 7) return;
    const r = new Float32Array((await rig.kernel.readReservoirs('final')).buffer);
    for (let i = 0; i < 64 * 64; i++) mean += (0.2126 * r[i * RES_WORDS + RW.F] + 0.7152 * r[i * RES_WORDS + RW.F + 1] + 0.0722 * r[i * RES_WORDS + RW.F + 2]) * r[i * RES_WORDS + RW.W];
  });
  const res = hashU32(await rig.kernel.readReservoirs('final'));
  rig.destroy();
  return { res, mean: mean / (64 * 64) };
}

describe('U8 plant activity (gate finding 2026-10-01): where U8-3 and U8-2t can act', () => {
  it('U8-3 (no p_k): bitwise inert with maxBounces 1 and only an analytic light (no case (c) / deep paths); active with deep paths', async () => {
    const scene = bitFixtureScene('c0e');
    const L = () => scene.lights;
    const b1 = [await plantChain(scene, 1, L, {}), await plantChain(scene, 1, L, { plant: { u8NoPk: true } })];
    const b3 = [await plantChain(scene, 3, L, {}), await plantChain(scene, 3, L, { plant: { u8NoPk: true } })];
    console.log(`[U8-3] b1 ${JSON.stringify(b1)} b3 ${JSON.stringify(b3)}`);
    expect(b1[1].res).toBe(b1[0].res);
    expect(b3[1].res).not.toBe(b3[0].res);
  });
  it('U8-2t (stale aux): bitwise inert with only analytic lights (Mode A ω1 ≡ 1, aux unused); active with emissive triangles / env NEE', async () => {
    const box = bitFixtureScene('c0e');
    const pw = (s: SceneData) => (t: number) => animatedLights(s.lights, t, s.lights.map((l) => ({ id: l.id, power: (u: number) => (u % 2 ? 1.5 : 1) })));
    const rect = [await plantChain(box, 3, pw(box), {}), await plantChain(box, 3, pw(box), { tPlant: { u8StaleAux: true } })];
    const all = allLightsScene();
    const allL = [await plantChain(all, 3, pw(all), {}), await plantChain(all, 3, pw(all), { tPlant: { u8StaleAux: true } })];
    console.log(`[U8-2t] rect-only ${JSON.stringify(rect)} emissive + env ${JSON.stringify(allL)}`);
    expect(rect[1].res).toBe(rect[0].res);
    expect(allL[1].res).not.toBe(allL[0].res);
  });
});
