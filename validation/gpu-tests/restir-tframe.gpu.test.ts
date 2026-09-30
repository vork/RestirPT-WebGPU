// M5 temporal frame state (T-A, restir-temporal-api.md §6.1): canary, U-M4-BITS (temporal off ≡ the pre-M5 build),
// U-BIND-1 (GPU part: the real merged scene layout), the P0 compile smoke of every temporal pipeline, and the P0 role /
// parity plumbing (temporal on + advance() with the stub stage ≡ temporal off). A1 adds U-TL-1, U-TV-1, U-TH-1,
// U-TP-1 and the `_s` evaluator bit test. Chrome lane authoritative.
import { afterAll, describe, expect, it } from 'vitest';
import { readBuffer } from '../../src/core/gpu/readback.ts';
import { createEnvResources, destroyEnvResources } from '../../src/core/render/env-gpu.ts';
import { JITTER_IID, JITTER_NONE, recentre, rigidInverse } from '../../src/core/render/frame-uniforms.ts';
import type { RestirFrameState } from '../../src/core/render/restir/frame-state.ts';
import type { LightData, SceneData } from '../../src/core/scene/types.ts';
import { BUILTIN_VIEWS, DebugResources, DebugViewRegistry } from '../../src/core/render/debug-views.ts';
import { RESTIR_VIEWS } from '../../src/core/render/restir/debug.ts';
import { RestirKernel } from '../../src/core/render/restir/kernel.ts';
import { RS_WGSL_CONSTS as K } from '../../src/core/render/restir/layout.ts';
import { restirSettings } from '../../src/core/render/restir/presets.ts';
import { RS_PASSES, TEMPORAL_PASSES, g2LayoutEntries, type RsPassName } from '../../src/core/render/restir/resources.ts';
import { fetchScenePackage } from '../../src/core/scene/scene-package.ts';
import { releaseTestGpu } from './device-factory.ts';
import { allLightsScene, bitFixtureScene, boxCamera, gpuScene, hashF32, readTexture4, restirRig, storageBuffer } from './restir-fixtures.ts';
import { BITS_CASES, bitsCase } from './restir-tframe-bits.ts';

afterAll(releaseTestGpu);

// ------------------------------------------------------------------------------------------------ canary (Q1)

describe('canary: a known-answer kernel on the temporal G0 layout', () => {
  it('lt_translate / lf_slot / ts_word on a fresh kernel (TF_* = 0): identity translation, cur slot, tState offsets', async () => {
    const g = await gpuScene(bitFixtureScene('x_quads'));
    const k = await RestirKernel.create(g.device, g.gpu, g.env, { settings: restirSettings('temporal'), features: g.features, wgslLanguageFeatures: g.wgslLanguageFeatures });
    k.setView({ camera: boxCamera(), width: 8, height: 8, runSeed: 1 });
    const device = g.device;
    const out = storageBuffer(device, 64 * 4);
    const g2l = device.createBindGroupLayout({ entries: [
      { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
      { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
    ] });
    const src = `
#include "restir/tframe.wgsl"
@group(2) @binding(1) var<storage, read_write> outv: array<u32, 64>;
@compute @workgroup_size(1) fn main() {
  outv[0] = 0xC0FFEEu;
  outv[1] = lt_translate(2u, RS_FS_PREV, RS_FS_CUR);
  outv[2] = lf_slot(RS_FS_PREV).nAnalytic;
  outv[3] = ts_word(3u, 5u);
  outv[4] = arena_tbase();
  outv[5] = rs_ns_alloc();
  outv[6] = queue_item_base(RS_Q_INV);
  outv[7] = queue_capacity_q(RS_Q_FWD);
  outv[8] = bitcast<u32>(rsParams.cCap);
  outv[9] = rsTemporal.flags;
  outv[10] = lt_translate(LIGHT_NONE, RS_FS_PREV, RS_FS_CUR);
}`;
    const pl = await k.compile('canary.wgsl', 'main', k.customDefines({ RS_ARENA_BINDING: '0u', RS_ARENA_RW: true }, false), k.customLayout(g2l, false), 'tframe-canary', { 'canary.wgsl': src });
    const bg = device.createBindGroup({ layout: g2l, entries: [{ binding: 0, resource: { buffer: k.resources.arena } }, { binding: 1, resource: { buffer: out } }] });
    k.beginSubmit();
    const enc = device.createCommandEncoder();
    k.encodeCustom(enc, pl, bg, {}, [1, 1], false);
    device.queue.submit([enc.finish()]);
    const u = new Uint32Array(await readBuffer(device, out, 64 * 4));
    const P = 64;
    expect(u[0]).toBe(0xC0FFEE);
    expect(u[1]).toBe(2);                                   // curToPrev/prevToCur of an unchanged light set: identity
    expect(u[2]).toBe(5);                                   // x_quads: 5 analytic lights
    expect(u[3]).toBe(6 * P * 3 + 20 * 3 + 5);
    expect(u[4]).toBe(6 * P * 3);
    expect(u[5]).toBe(3);
    expect(u[6]).toBe(P);
    expect(u[7]).toBe(P);
    expect(new Float32Array(u.buffer)[8]).toBe(20);
    expect(u[9]).toBe(0);
    expect(u[10]).toBe(0xFFFFFFFF);
    out.destroy(); k.destroy(); g.destroy();
  });
});

// ------------------------------------------------------------------------------------------------ U-M4-BITS (P0 part)

// Recorded on the pre-M5 build (main e251b43, Chrome 154 / Metal, 2026-09-30) with restir-tframe-bits.ts unchanged.
const GOLDEN: Record<string, Record<string, string>> = {
  'i-3.1': { res0: '8aeadda4', res1: '113c9dc5', final: '8aeadda4', image: 'bc8e8f67', counters: '0,0,0,0', arena: '0,0,0,0,0' },
  'i-3.2': { res0: 'dd6406e5', res1: 'bdb2645b', final: 'dd6406e5', image: '4fd15bbf', counters: '0,0,0,0', arena: '111334,1283,23054,3732,0' },
  'i-interactive': { res0: '8aeadda4', res1: '95ea379d', final: '95ea379d', image: '7e1727f5', counters: '0,0,0,0', arena: '24276,265,5360,4943,0' },
  'xq-3.1': { res0: '908a3e68', res1: '25249dc5', final: '908a3e68', image: 'efe76362', counters: '0,0,0,0', arena: '0,0,0,0,0' },
  'xq-3.2': { res0: '5f25862a', res1: '92d06299', final: '92d06299', image: '1e0fb8ad', counters: '0,0,0,0', arena: '31792,3263,4787,535,0' },
  'xq-interactive': { res0: 'daf2a634', res1: 'e9f010ff', final: 'e9f010ff', image: '1801361c', counters: '0,0,0,0', arena: '1660,121,383,1121,0' },
};

describe('U-M4-BITS: temporal off ≡ the pre-M5 build (reservoirs, final reservoirs, finalize output, counters)', () => {
  for (const c of BITS_CASES) {
    it(`${c.name} (preset ${c.preset} as shipped, and temporal: false explicitly)`, async () => {
      const asShipped = await bitsCase(c);
      const off = await bitsCase(c, { temporal: false });
      console.log(`[U-M4-BITS] ${c.name} ${JSON.stringify(asShipped)}`);
      expect(asShipped).toEqual(GOLDEN[c.name]);
      expect(off).toEqual(GOLDEN[c.name]);
    });
  }
});

describe('roles / parity with temporal on (TD2, §2.6): h/w rotation, G-buffer parity; reset-every-frame ≡ the M4 golden', () => {
  for (const c of BITS_CASES.filter((x) => x.scene === 'x_quads' && x.preset !== 'interactive')) {
    it(c.name, async () => {
      const scene = bitFixtureScene('x_quads');
      for (const resetEveryFrame of [false, true]) {
        const rig = await restirRig(scene, 64, 64, { preset: c.preset, settings: { ...c.settings, temporal: true } as never, seed: 23 });
        const k = rig.kernel, device = rig.g.device;
        await k.prepare();
        expect(k.resources.alloc.temporal).toBe(true);
        const clear = device.createCommandEncoder();
        clear.clearBuffer(rig.accum); clear.clearBuffer(rig.counters); clear.clearBuffer(k.resources.arena, 0, 256);
        device.queue.submit([clear.finish()]);
        const parities: number[] = [];
        let lastFinal = 0;
        for (let f = 0; f < c.frames; f++) {
          const t = 5 + f;
          const adv = k.advance({ t, camera: boxCamera(), lights: scene.lights, reset: resetEveryFrame });
          const hist = f > 0 && !resetEveryFrame;
          expect(adv.histValid, `frame ${f}: ${adv.reasons.join(',')}`).toBe(hist);
          expect(adv.flags & K.TF_HIST_VALID).toBe(hist ? K.TF_HIST_VALID : 0);
          parities.push(k.resources.parity);
          expect(k.historyIndex()).toBe(hist ? lastFinal : -1);
          expect(k.resBase()).toBe(hist ? 1 - lastFinal : 0);   // TD2 role rotation
          k.beginSubmit();
          const enc = device.createCommandEncoder();
          const units = k.frameUnits(t, { accum: rig.accum, counters: rig.counters });
          expect(k.currentAdvance).toBe(adv);                    // readable while / after building (A-10)
          for (const u of units) u.encode(enc);
          device.queue.submit([enc.finish()]);
          await device.queue.onSubmittedWorkDone();
          lastFinal = k.finalResIndex();
          expect(lastFinal).toBe((k.resBase() + k.lastRounds) % 2);
        }
        expect(parities).toEqual([1, 0].slice(0, c.frames));  // the G-buffer parity flips per advanced frame
        const r = await k.readCounters(false);
        expect(r.rsc.tNonFinite + r.rsc.tPendingLeft + r.rsc.pendingLeft + r.rsc.slotMismatch).toBe(0);
        if (resetEveryFrame) {                                  // no history ⇒ canonical + spatial only ≡ M4 bitwise
          const sum = new Float32Array(await readBuffer(device, rig.accum, 64 * 64 * 16));
          const mean = new Float32Array(64 * 64 * 3);
          for (let i = 0; i < 64 * 64; i++) for (let ch = 0; ch < 3; ch++) mean[3 * i + ch] = sum[4 * i + ch] / c.frames;
          expect(hashF32(mean)).toBe(GOLDEN[c.name].image);
          expect(hashF32(new Float32Array((await k.readReservoirs('final')).buffer))).toBe(GOLDEN[c.name].final);
        }
        rig.destroy();
      }
    });
  }
});

// ------------------------------------------------------------------------------------------------ U-BIND-1 (GPU)

describe('U-BIND-1 (GPU part): the merged scene layout has ≤ 5 storage buffers; every pass layout ≤ 9', () => {
  it('allLightsScene (textured? no) and cornell_i_512 (package v2); every RS_PASSES layout with debug', async () => {
    const pkg = await fetchScenePackage('/validation/scenes/cornell_i_512/');
    for (const scene of [allLightsScene(), pkg.scene]) {
      const g = await gpuScene(scene);
      const sceneSB = g.gpu.layoutEntries().filter((e) => e.buffer && e.buffer.type !== 'uniform').length;
      expect(sceneSB).toBeLessThanOrEqual(5);
      for (const name of Object.keys(RS_PASSES) as RsPassName[]) {
        const d = RS_PASSES[name];
        const g2 = g2LayoutEntries(name, 'rgba16float').filter((e) => e.buffer && e.buffer.type !== 'uniform').length;
        const total = 1 + (d.scene ? sceneSB : 0) + g2 + (d.debug ? 1 : 0);
        expect(total, name).toBeLessThanOrEqual(9);
      }
      g.destroy();
    }
  });
});

// ------------------------------------------------------------------------------------------------ compile smoke

describe('P0 compile smoke: every temporal / refresh pipeline (validation and debug variants)', () => {
  it('RestirKernel with the full preset compiles T1–T4, refresh fwd/inv and rs_args; debug variants too', async () => {
    const g = await gpuScene(allLightsScene());
    const device = g.device;
    const k = await RestirKernel.create(device, g.gpu, g.env, { settings: restirSettings('full'), features: g.features, wgslLanguageFeatures: g.wgslLanguageFeatures });
    k.setView({ camera: boxCamera(), width: 16, height: 16, runSeed: 1 });
    await k.prepare();                                     // TemporalStage.prepare compiles every temporal pipeline
    for (const n of TEMPORAL_PASSES) await k.pipeline(n);
    // bind groups of every temporal pass exist for both roles and both parities
    for (const n of TEMPORAL_PASSES) for (const idx of [0, 1]) for (const par of [0, 1]) {
      k.resources.parity = par;
      expect(k.resources.g2(n, idx)).toBeTruthy();
    }
    k.resources.parity = 0;
    k.destroy();
    const debug = new DebugResources(device, new DebugViewRegistry([...BUILTIN_VIEWS, ...RESTIR_VIEWS]));
    const kd = await RestirKernel.create(device, g.gpu, g.env, { settings: restirSettings('full'), debug, features: g.features, wgslLanguageFeatures: g.wgslLanguageFeatures });
    kd.setView({ camera: boxCamera(), width: 16, height: 16, runSeed: 1 });
    await kd.prepare();
    kd.destroy(); debug.destroy(); g.destroy();
  });
});

// ================================================================================================ A1

/** A temporal kernel on a scene; frame(t, camera, lights?) = advance + frameUnits + submit. */
async function temporalRig(scene: SceneData, W: number, H: number, o: { preset?: 'temporal' | 'full'; members?: number; jitterMode?: number; cam?: { camToWorld: number[]; yfov: number } } = {}) {
  const rig = await restirRig(scene, W, H, { preset: o.preset ?? 'temporal', members: o.members, jitterMode: o.jitterMode as never, seed: 31, cam: o.cam });
  const k = rig.kernel, device = rig.g.device;
  return {
    rig, k, device,
    async frame(t: number, camera: { camToWorld: number[]; yfov: number }, lights: LightData[] = scene.lights, extra: (enc: GPUCommandEncoder) => void = () => { }, env?: RestirFrameState['env']) {
      const adv = k.advance({ t, camera, lights, env });
      k.beginSubmit();
      const enc = device.createCommandEncoder();
      for (const u of k.frameUnits(t, { accum: rig.accum, counters: rig.counters })) u.encode(enc);
      extra(enc);
      device.queue.submit([enc.finish()]);
      await device.queue.onSubmittedWorkDone();
      return adv;
    },
  };
}

const bgl = (device: GPUDevice, kinds: ('tex-u' | 'tex-f' | 'rw')[]) => device.createBindGroupLayout({
  entries: kinds.map((k, binding): GPUBindGroupLayoutEntry => (k === 'rw'
    ? { binding, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }
    : { binding, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: k === 'tex-u' ? 'uint' : 'unfilterable-float' } })),
});

// ------------------------------------------------------------------------------------------------ `_s` ≡ wrappers

const S_BITS_WGSL = `
#include "restir/endpoint.wgsl"
#include "common/rng.wgsl"
@group(2) @binding(0) var<storage, read_write> cnt: array<atomic<u32>, 16>;
fn rnd(i: u32, k: u32) -> f32 { return u32_to_unit(pcg4d(vec4u(i, k, 0x5e11u, 0x77u)).x); }
fn ls_eq(a: LightSample, b: LightSample) -> bool {
  return a.valid == b.valid && all(bitcast<vec3u>(a.dir) == bitcast<vec3u>(b.dir)) && all(bitcast<vec3u>(a.Lambda) == bitcast<vec3u>(b.Lambda))
    && bitcast<u32>(a.q) == bitcast<u32>(b.q) && bitcast<u32>(a.p1) == bitcast<u32>(b.p1) && bitcast<u32>(a.dist) == bitcast<u32>(b.dist)
    && all(bitcast<vec3u>(a.pos) == bitcast<vec3u>(b.pos)) && a.entry == b.entry && a.prim == b.prim;
}
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3u, @builtin(num_workgroups) nwg: vec3u) {
  let i = gid.x + gid.y * nwg.x * 64u;
  if (i >= 1048576u) { return; }
  let slot = lightsParams.cur;
  let er = envParams;
  let x = vec3f(-1.45, 0.02, -1.95) + vec3f(2.9, 1.9, 2.9) * vec3f(rnd(i, 0u), rnd(i, 1u), rnd(i, 2u));
  let z = vec3f(-1.45, 0.02, -1.95) + vec3f(2.9, 1.9, 2.9) * vec3f(rnd(i, 3u), rnd(i, 4u), rnd(i, 5u));
  let d = normalize(vec3f(rnd(i, 6u), rnd(i, 7u), rnd(i, 8u)) * 2.0 - 1.0);
  let hv = pcg4d(vec4u(i, 17u, 3u, 9u));
  let ep = nee_draw(slot, hv.x, hv.y, pcg4d(vec4u(i, 18u, 4u, 10u)).xyz);
  atomicAdd(&cnt[0], select(1u, 0u, ls_eq(nee_eval(x, ep), nee_eval_s(x, ep, slot, er))));
  atomicAdd(&cnt[1], select(1u, 0u, nee_endpoint_id(ep) == nee_endpoint_id_s(ep, slot)));
  let prim = hv.z % max(lightsParams.triCount * 2u + 2u, 1u);
  let ng = normalize(vec3f(rnd(i, 9u), rnd(i, 10u), rnd(i, 11u)) - 0.5);
  atomicAdd(&cnt[2], select(1u, 0u, bitcast<u32>(tri_light_p1(x, z, ng, prim)) == bitcast<u32>(tri_light_p1_s(x, z, ng, prim, slot))));
  let ae = hv.w % max(slot.nAnalytic, 1u);
  atomicAdd(&cnt[3], select(1u, 0u, slot.nAnalytic == 0u || bitcast<u32>(analytic_area_p1(x, ae, z)) == bitcast<u32>(analytic_area_p1_s(x, ae, z, slot))));
  atomicAdd(&cnt[4], select(1u, 0u, bitcast<u32>(p1Env(d)) == bitcast<u32>(p1Env_s(d, slot, er))));
  let p2 = rnd(i, 12u) * 4.0;
  let B = 1u + (hv.x & 3u);
  let ad = (hv.y & 7u) == 0u;
  atomicAdd(&cnt[5], select(1u, 0u, bitcast<u32>(env_bsdf_mis_weight(d, p2, B, ad)) == bitcast<u32>(env_bsdf_mis_weight_s(d, p2, B, ad, slot, er))));
  let uv = vec2f(rnd(i, 13u), rnd(i, 14u));
  atomicAdd(&cnt[6], select(1u, 0u, all(bitcast<vec3u>(envRadiance(uv)) == bitcast<vec3u>(envRadiance_s(uv, er)))));
  if (slot.envEntry != LIGHT_NONE) {
    let ci = hv.z % env_Hm();
    let cj = hv.w % env_Wm();
    atomicAdd(&cnt[7], select(1u, 0u, ls_eq(env_light_sample_cell(slot, slot.envEntry, ci, cj, hv.x), env_light_sample_cell_s(slot, er, slot.envEntry, ci, cj, hv.x))));
  }
  atomicAdd(&cnt[15], 1u);
}`;

describe('U-M4-BITS (A1 part): every `_s` evaluator ≡ its wrapper bitwise (10⁶ random inputs each)', () => {
  it('nee_eval, nee_endpoint_id, tri_light_p1, analytic_area_p1, p1Env, env_bsdf_mis_weight, envRadiance, env_light_sample_cell', async () => {
    const g = await gpuScene(allLightsScene());
    const device = g.device;
    const k = await RestirKernel.create(device, g.gpu, g.env, { settings: restirSettings('initial'), features: g.features, wgslLanguageFeatures: g.wgslLanguageFeatures });
    k.setView({ camera: boxCamera(), width: 8, height: 8, runSeed: 1 });
    const cnt = storageBuffer(device, 64);
    const l = bgl(device, ['rw']);
    const pl = await k.compile('sbits.wgsl', 'main', k.customDefines({}, true), k.customLayout(l, true), 'tframe-sbits', { 'sbits.wgsl': S_BITS_WGSL });
    const bg = device.createBindGroup({ layout: l, entries: [{ binding: 0, resource: { buffer: cnt } }] });
    k.beginSubmit();
    const enc = device.createCommandEncoder();
    k.encodeCustom(enc, pl, bg, {}, [1024, 16], true);
    device.queue.submit([enc.finish()]);
    const c = new Uint32Array(await readBuffer(device, cnt, 64));
    console.log(`[U-M4-BITS _s] mismatches ${Array.from(c.subarray(0, 8)).join(',')} over ${c[15]}`);
    expect(c[15]).toBe(1048576);
    expect(Array.from(c.subarray(0, 8))).toEqual([0, 0, 0, 0, 0, 0, 0, 0]);
    cnt.destroy(); k.destroy(); g.destroy();
  });
});

// ------------------------------------------------------------------------------------------------ U-TL-1 (GPU)

const LT_WGSL = `
#include "restir/tframe.wgsl"
@group(2) @binding(0) var<storage, read_write> outv: array<u32, 512>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let e = gid.x;
  if (e >= 64u) { return; }
  outv[e] = lt_translate(e, RS_FS_PREV, RS_FS_CUR);
  outv[64u + e] = lt_translate(e, RS_FS_CUR, RS_FS_PREV);
  outv[128u + e] = bitcast<u32>(lt_pmf(e, RS_FS_PREV));
  outv[192u + e] = bitcast<u32>(lt_pmf(e, RS_FS_CUR));
  outv[256u + e] = lt_change_bits(e);
  if (e == 0u) {
    outv[320u] = lf_slot(RS_FS_PREV).nAnalytic; outv[321u] = lf_slot(RS_FS_CUR).nAnalytic;
    outv[322u] = lf_slot(RS_FS_PREV).envEntry; outv[323u] = lf_slot(RS_FS_CUR).envEntry; outv[324u] = rsTemporal.flags;
  }
}`;

describe('U-TL-1 (GPU): WGSL lt_translate / lt_pmf / lt_change_bits ≡ the CPU light state across nA changes', () => {
  it('remove light 1, add light 4 (other type), move light 2, env strength change; then an unchanged frame (TF_LIGHTS_SAME)', async () => {
    const scene = allLightsScene();
    const T = await temporalRig(scene, 16, 16);
    const { k, device } = T;
    const cam = boxCamera();
    const out = storageBuffer(device, 512 * 4);
    const l = bgl(device, ['rw']);
    const pl = await k.compile('lt.wgsl', 'main', k.customDefines({}, false), k.customLayout(l, false), 'tframe-lt', { 'lt.wgsl': LT_WGSL });
    const bg = device.createBindGroup({ layout: l, entries: [{ binding: 0, resource: { buffer: out } }] });
    const run = async () => {
      k.beginSubmit();
      const enc = device.createCommandEncoder();
      k.encodeCustom(enc, pl, bg, {}, [1, 1], false);
      device.queue.submit([enc.finish()]);
      return new Uint32Array(await readBuffer(device, out, 512 * 4));
    };
    const env0 = { params: { rotationZ: 0, strength: 1, tint: [1, 1, 1] as [number, number, number], visibleToCamera: true }, mapId: 'env' };
    await T.frame(0, cam, scene.lights, undefined, env0);
    const edited: LightData[] = [
      ...scene.lights.filter((x) => x.id !== 1).map((x) => (x.id === 2 ? { ...x, matrix: new Float32Array([...Array.from(x.matrix).slice(0, 12), 0.1, 1.7, 0.1, 1]) } : x)),
      { ...scene.lights[0], id: 9, type: 'disk', sizeX: 0.3 },
    ];
    const adv = await T.frame(1, cam, edited, undefined, { ...env0, params: { ...env0.params, strength: 2 } });
    expect(adv.histValid).toBe(true);
    expect(adv.commit.same).toBe(false);
    expect(adv.commit.removed).toEqual([1]);
    expect(adv.commit.added).toEqual([9]);
    expect(adv.flags & (K.TF_REFRESH | K.TF_ENV_RADIO | K.TF_PMF_CHANGED | K.TF_LIGHT_MOVED)).toBe(K.TF_REFRESH | K.TF_ENV_RADIO | K.TF_PMF_CHANGED | K.TF_LIGHT_MOVED);
    const st = k.lights.state;
    const check = (u: Uint32Array, same: boolean) => {
      const prev = st.prevSlot, cur = st.curSlot;
      const f = new Float32Array(u.buffer);
      expect(u[320]).toBe(same ? cur.nAnalytic : prev.nAnalytic);
      expect(u[321]).toBe(cur.nAnalytic);
      let n = 0;
      for (let e = 0; e < (same ? cur : prev).nEntries; e++, n++) {
        expect(u[e], `prev→cur ${e}`).toBe(st.translate(e, 'prev', 'cur', same));
        expect(f[128 + e]).toBe(st.pmfOf(same ? cur : prev, e));
      }
      for (let e = 0; e < cur.nEntries; e++, n++) {
        expect(u[64 + e], `cur→prev ${e}`).toBe(st.translate(e, 'cur', 'prev', same));
        expect(f[192 + e]).toBe(st.pmfOf(cur, e));
        const bits = same ? 0 : e < cur.nAnalytic ? st.records[cur.lightOff + 28 * e + 26] : e === cur.envEntry ? ((adv.flags & K.TF_ENV_MOVED ? 1 : 0) | (adv.flags & K.TF_ENV_RADIO ? 2 : 0)) : 0;
        if (!same) expect(u[256 + e], `bits ${e}`).toBe(bits);
      }
      return n;
    };
    const n1 = check(await run(), false);
    // the moved light 2 (entry of id 2 in cur) has MOVED, the added one ADDED
    const idsCur = st.slots[st.cur]!.ids;
    const u1 = await run();
    expect(u1[256 + idsCur.indexOf(2)] & 1).toBe(1);
    expect(u1[256 + idsCur.indexOf(9)]).toBe(4);
    const adv2 = await T.frame(2, cam, edited, undefined, { ...env0, params: { ...env0.params, strength: 2 } });
    expect(adv2.commit.same).toBe(true);
    expect(adv2.flags & (K.TF_LIGHTS_SAME | K.TF_REFRESH)).toBe(K.TF_LIGHTS_SAME);
    const n2 = check(await run(), true);
    console.log(`[U-TL-1 GPU] entries compared: ${n1} (changed frame), ${n2} (unchanged frame)`);
    out.destroy(); T.rig.destroy();
  });
});

// ------------------------------------------------------------------------------------------------ U-TV-1, U-TH-1

describe('U-TV-1: the previous G-buffer at t ≡ the G-buffer of t−1 bitwise; envPrev ≡ the previous packed env words', () => {
  it('3 frames with camera motion and env rotation / tint edits', async () => {
    const scene = allLightsScene();
    const T = await temporalRig(scene, 32, 24);
    const { k, device } = T;
    const cam0 = boxCamera();
    const cams = [0, 1, 2].map((f) => ({ ...cam0, camToWorld: cam0.camToWorld.map((v, i) => (i === 12 ? v + 0.02 * f : v)) }));
    const envs = [0, 1, 2].map((f) => ({ params: { rotationZ: 0.1 * f, strength: 1, tint: [1, 1 - 0.1 * f, 1] as [number, number, number], visibleToCamera: true }, mapId: 'env' }));
    let prevV: Uint32Array | undefined, prevG: Uint32Array | undefined, prevEnv: Uint32Array | undefined;
    for (let f = 0; f < 3; f++) {
      const adv = await T.frame(f, cams[f], scene.lights, undefined, envs[f]);
      const tu = new Uint32Array(await readBuffer(device, k.rsTemporal, 128));
      if (f > 0) {
        expect(adv.histValid).toBe(true);
        expect(await readTexture4(device, k.resources.vbufPrev)).toEqual(prevV);
        expect(await readTexture4(device, k.resources.geoPrev)).toEqual(prevG);
        expect(Array.from(tu.subarray(4, 12))).toEqual(Array.from(prevEnv!));
        expect(adv.flags & (K.TF_ENV_MOVED | K.TF_ENV_RADIO | K.TF_CAM_SAME)).toBe(K.TF_ENV_MOVED | K.TF_ENV_RADIO);
        expect(tu[0]).toBe(adv.flags);
        expect(tu[2]).toBe(tu[3] + 1);                               // frameGen = prevGen + 1
        expect(Array.from(tu.subarray(16, 20))).toEqual(adv.temporal.gensPrev);
      } else {
        expect(Array.from(tu.subarray(4, 12))).toEqual(Array.from(new Uint32Array(k.envParamsWords())));   // reset: envPrev = cur
      }
      prevV = await readTexture4(device, k.resources.vbuf);
      prevG = await readTexture4(device, k.resources.geo);
      prevEnv = new Uint32Array(k.envParamsWords());
      expect(prevEnv[0]).toBe(new Uint32Array(Float32Array.of(Math.cos(0.1 * f)).buffer)[0]);
    }
    T.rig.destroy();
  });
});

describe('U-TH-1 (synthetic, T-E repeats it on ixs_k frame 20): an env map swap resets history for exactly one frame', () => {
  it('frames 0…3, setEnvironment before frame 2: TF_HIST_VALID 0 / RSC_T_QVALID 0 at 2, valid again at 3', async () => {
    const scene = allLightsScene();
    const T = await temporalRig(scene, 32, 24);
    const { k, device } = T;
    const cam = boxCamera();
    const t1 = await k.pipeline('rs_t_classify');
    const classify = (t: number) => (enc: GPUCommandEncoder) => {
      const h = k.historyIndex();
      const a = k.resources.alloc;
      k.encodePass(enc, 'rs_t_classify', t1, k.resources.g2('rs_t_classify', h < 0 ? 0 : h), { t, passId: K.RS_PASS_T_CLASSIFY, rowBase: 0, rowEnd: a.atlasH }, k.perPixelWorkgroups(0, a.atlasH));
    };
    const env2 = await createEnvResources(device, { ...scene.env!, name: 'swap' });
    const res: { valid: boolean; qvalid: number; reasons: string[] }[] = [];
    for (let f = 0; f < 4; f++) {
      if (f === 2) k.setEnvironment(env2);
      await device.queue.onSubmittedWorkDone();
      const clr = device.createCommandEncoder(); clr.clearBuffer(k.resources.arena, 0, 256); device.queue.submit([clr.finish()]);
      const adv = await T.frame(f, cam, scene.lights, classify(f));
      const c = await k.readCounters(true);
      res.push({ valid: adv.histValid, qvalid: c.rsc.tQvalid, reasons: adv.reasons });
    }
    console.log(`[U-TH-1] ${JSON.stringify(res)}`);
    expect(res.map((r) => r.valid)).toEqual([false, true, false, true]);
    expect(res[2].reasons).toContain('env-map');
    expect(res[2].qvalid).toBe(0);
    expect(res[1].qvalid).toBeGreaterThan(0);
    expect(res[3].qvalid).toBeGreaterThan(0);
    destroyEnvResources(env2);
    T.rig.destroy();
  });
});

// ------------------------------------------------------------------------------------------------ U-TP-1

const TP_WGSL = `
#include "restir/tpick.wgsl"
@group(2) @binding(4) var<storage, read_write> outv: array<u32>;
@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let p = rs_pix(gid.xy);
  if (!p.valid) { return; }
  let pk = temporal_pixel(p, vec2u(0u));
  let h = tpick_hash(p);
  let o = 12u * p.ai;
  let vb = rs_vbuf(p.px);
  var x1 = vec3f(0.0);
  if (vb.x != 0xFFFFFFFFu) { x1 = vertex_from_ids(vb.x, bitcast<f32>(vb.y), bitcast<f32>(vb.z), rs_cam_pos()).pos; }
  outv[o] = select(0u, 1u, pk.valid) | (pk.tap << 1u) | ((h.z & 7u) << 5u) | select(0u, 256u, vb.x == 0xFFFFFFFFu);
  outv[o + 1u] = pk.ai;
  outv[o + 2u] = bitcast<u32>(pk.sp.x); outv[o + 3u] = bitcast<u32>(pk.sp.y);
  outv[o + 4u] = bitcast<u32>(u32_to_unit(h.x)); outv[o + 5u] = bitcast<u32>(u32_to_unit(h.y));
  outv[o + 6u] = bitcast<u32>(x1.x); outv[o + 7u] = bitcast<u32>(x1.y); outv[o + 8u] = bitcast<u32>(x1.z);
  outv[o + 9u] = bitcast<u32>(length(x1 - lf_cam_pos(RS_FS_PREV)));
  outv[o + 10u] = p.member; outv[o + 11u] = 0u;
}`;
const RING = [[1, 0], [1, 1], [0, 1], [-1, 1], [-1, 0], [-1, -1], [0, -1], [1, -1]];

describe('U-TP-1: temporal_pixel (q′)', () => {
  const W = 48, H = 32, E = 4;
  const base = boxCamera();
  const yaw = (deg: number) => {
    const a = deg * Math.PI / 180, c = Math.cos(a), s = Math.sin(a);
    const m = base.camToWorld.slice();
    for (const col of [0, 1, 2]) { const x = m[4 * col], z = m[4 * col + 2]; m[4 * col] = c * x + s * z; m[4 * col + 2] = -s * x + c * z; }
    return { ...base, camToWorld: m };
  };
  const cases: [string, { camToWorld: number[]; yfov: number }, number][] = [
    ['static, jitter off', base, JITTER_NONE],
    ['translation', { ...base, camToWorld: base.camToWorld.map((v, i) => (i === 12 ? v + 0.05 : i === 14 ? v - 0.1 : v)) }, JITTER_IID],
    ['rotation 2°', yaw(2), JITTER_IID],
    ['zoom in (fov 55° → 50°)', { ...base, yfov: 50 * Math.PI / 180 }, JITTER_IID],
    ['zoom out (fov 55° → 60°)', { ...base, yfov: 60 * Math.PI / 180 }, JITTER_IID],
  ];
  for (const [name, cam1, jm] of cases) {
    it(`${name}: back-projection ≡ CPU, c₀ = ⌊sp + ξ⌋, taps / validity ≡ the rule, same member only`, async () => {
      const scene = bitFixtureScene('x_quads');
      const T = await temporalRig(scene, W, H, { members: E, jitterMode: jm });
      const { k, device } = T;
      const res = k.resources, a = res.alloc;
      const P = a.atlasW * a.atlasH;
      const out = storageBuffer(device, P * 48);
      const l = bgl(device, ['tex-u', 'tex-f', 'tex-u', 'tex-f', 'rw']);
      const defs = k.customDefines({ RS_VBUF_BINDING: '0u', RS_GEO_BINDING: '1u', RS_VBUF_PREV_BINDING: '2u', RS_GEO_PREV_BINDING: '3u' }, true);
      const pl = await k.compile('tp.wgsl', 'main', defs, k.customLayout(l, true), 'tframe-tp', { 'tp.wgsl': TP_WGSL });
      await T.frame(0, base);
      const prevV = await readTexture4(device, res.vbuf), prevG = new Float32Array((await readTexture4(device, res.geo)).buffer);
      const adv = await T.frame(1, cam1, scene.lights, (enc) => {
        const bg = device.createBindGroup({ layout: l, entries: [res.views.vbuf, res.views.geo, res.views.vbufPrev, res.views.geoPrev, { buffer: out }].map((r, binding) => ({ binding, resource: r as GPUBindingResource })) });
        k.encodeCustom(enc, pl, bg, { t: 1, rowBase: 0, rowEnd: a.atlasH }, k.perPixelWorkgroups(0, a.atlasH), true);
      });
      expect(adv.histValid).toBe(true);
      const curG = new Float32Array((await readTexture4(device, res.geo)).buffer);
      const u = new Uint32Array(await readBuffer(device, out, P * 48));
      const f = new Float32Array(u.buffer);
      // CPU back-projection with the previous (recentred, f64) camera.
      const c2w = recentre(base.camToWorld, k.scene.origin), w2c = rigidInverse(c2w);
      const tanY = Math.tan(base.yfov / 2), aspect = W / H;
      const prevPos = [c2w[12], c2w[13], c2w[14]];
      let hits = 0, valid = 0, same = 0, spBad = 0, ruleBad = 0, borderline = 0, memberBad = 0;
      for (let ai = 0; ai < P; ai++) {
        const o = 12 * ai;
        const px = ai % a.atlasW, py = Math.floor(ai / a.atlasW);
        if (u[o] & 256) { expect(u[o] & 1).toBe(0); continue; }
        hits++;
        const x = [f[o + 6], f[o + 7], f[o + 8]];
        const pc = [0, 1, 2].map((r) => w2c[r] * x[0] + w2c[4 + r] * x[1] + w2c[8 + r] * x[2] + w2c[12 + r]);
        const spCpu = [((pc[0] / (-pc[2] * tanY * aspect)) + 1) * 0.5 * W - 0.5, H - ((pc[1] / (-pc[2] * tanY)) + 1) * 0.5 * H - 0.5];
        const sp = [f[o + 2], f[o + 3]];
        if (-pc[2] >= 1e-6 && (Math.abs(sp[0] - spCpu[0]) > 2e-3 || Math.abs(sp[1] - spCpu[1]) > 2e-3)) spBad++;
        const xi = [f[o + 4], f[o + 5]], rot = (u[o] >>> 5) & 7;
        const c0 = [Math.floor(Math.fround(sp[0] + xi[0])), Math.floor(Math.fround(sp[1] + xi[1]))];
        const mx = Math.floor(px / W), my = Math.floor(py / H);
        const dist = f[o + 9];
        const ng = [curG[4 * ai], curG[4 * ai + 1], curG[4 * ai + 2]];
        // replay the tap rule on the CPU
        let expTap = 9, expAi = -1, near = false;
        for (let tap = 0; tap < 9 && expTap === 9; tap++) {
          const c = tap === 0 ? c0 : [c0[0] + RING[(tap - 1 + rot) & 7][0], c0[1] + RING[(tap - 1 + rot) & 7][1]];
          if (c[0] < 0 || c[1] < 0 || c[0] >= W || c[1] >= H) continue;
          const bi = (my * H + c[1]) * a.atlasW + mx * W + c[0];
          if (prevV[4 * bi] === 0xFFFFFFFF) continue;
          const d = ng[0] * prevG[4 * bi] + ng[1] * prevG[4 * bi + 1] + ng[2] * prevG[4 * bi + 2];
          const zp = new Float32Array(Uint32Array.of(prevV[4 * bi + 3]).buffer)[0];
          if (Math.abs(d - 0.5) < 1e-5 || Math.abs(Math.abs(dist - zp) - 0.1 * zp) < 1e-5 * zp) near = true;
          if (!(d >= 0.5) || !(Math.abs(dist - zp) <= 0.1 * zp)) continue;
          expTap = tap; expAi = bi;
        }
        const gotValid = (u[o] & 1) === 1, gotTap = (u[o] >>> 1) & 15;
        if (gotValid) {
          valid++;
          const qa = u[o + 1], qx = qa % a.atlasW, qy = Math.floor(qa / a.atlasW);
          if (Math.floor(qx / W) !== mx || Math.floor(qy / H) !== my) memberBad++;
          if (qa === ai) same++;
        }
        if (gotTap !== expTap || (gotValid && u[o + 1] !== expAi)) { if (near) borderline++; else ruleBad++; }
        expect(gotTap === 9).toBe(!gotValid);
        const cp = [0, 1, 2].map((r) => x[r] - prevPos[r]);
        expect(Math.abs(dist - Math.hypot(cp[0], cp[1], cp[2]))).toBeLessThan(1e-4 * dist + 1e-6);
      }
      console.log(`[U-TP-1] ${name}: hits ${hits}, valid ${valid}, q′ = q ${same}, sp mismatches ${spBad}, rule mismatches ${ruleBad} (+${borderline} at a threshold), off-member ${memberBad}`);
      expect(spBad).toBe(0);
      expect(ruleBad).toBe(0);
      expect(borderline).toBeLessThanOrEqual(Math.ceil(1e-3 * hits));
      expect(memberBad).toBe(0);
      if (name.startsWith('static')) expect(same).toBeGreaterThanOrEqual(Math.floor(0.999 * hits));
      else expect(valid).toBeGreaterThan(0.5 * hits);
      out.destroy(); T.rig.destroy();
    });
  }
});
