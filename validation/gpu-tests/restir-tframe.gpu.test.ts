// M5 temporal frame state (T-A, restir-temporal-api.md §6.1): canary, U-M4-BITS (temporal off ≡ the pre-M5 build),
// U-BIND-1 (GPU part: the real merged scene layout), the P0 compile smoke of every temporal pipeline, and the P0 role /
// parity plumbing (temporal on + advance() with the stub stage ≡ temporal off). A1 adds U-TL-1, U-TV-1, U-TH-1,
// U-TP-1 and the `_s` evaluator bit test. Chrome lane authoritative.
import { afterAll, describe, expect, it } from 'vitest';
import { readBuffer } from '../../src/core/gpu/readback.ts';
import { BUILTIN_VIEWS, DebugResources, DebugViewRegistry } from '../../src/core/render/debug-views.ts';
import { RESTIR_VIEWS } from '../../src/core/render/restir/debug.ts';
import { RestirKernel } from '../../src/core/render/restir/kernel.ts';
import { RS_WGSL_CONSTS as K } from '../../src/core/render/restir/layout.ts';
import { restirSettings } from '../../src/core/render/restir/presets.ts';
import { RS_PASSES, TEMPORAL_PASSES, g2LayoutEntries, type RsPassName } from '../../src/core/render/restir/resources.ts';
import { fetchScenePackage } from '../../src/core/scene/scene-package.ts';
import { releaseTestGpu } from './device-factory.ts';
import { allLightsScene, bitFixtureScene, boxCamera, gpuScene, hashF32, restirRig, storageBuffer } from './restir-fixtures.ts';
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

describe('P0 roles / parity: temporal on + advance() per frame with the stub stage ≡ temporal off', () => {
  for (const c of BITS_CASES.filter((x) => x.scene === 'x_quads' && x.preset !== 'interactive')) {
    it(c.name, async () => {
      const scene = bitFixtureScene('x_quads');
      const rig = await restirRig(scene, 64, 64, { preset: c.preset, settings: { ...c.settings, temporal: true } as never, seed: 23 });
      const k = rig.kernel, device = rig.g.device;
      await k.prepare();
      expect(k.resources.alloc.temporal).toBe(true);
      const clear = device.createCommandEncoder();
      clear.clearBuffer(rig.accum); clear.clearBuffer(rig.counters); clear.clearBuffer(k.resources.arena, 0, 256);
      device.queue.submit([clear.finish()]);
      const parities: number[] = [];
      for (let f = 0; f < c.frames; f++) {
        const t = 5 + f;
        const adv = k.advance({ t, camera: boxCamera(), lights: scene.lights });
        expect(adv.histValid).toBe(false);                   // P0 stub: every frame is a reset
        expect(adv.flags & K.TF_HIST_VALID).toBe(0);
        parities.push(k.resources.parity);
        expect(k.resBase()).toBe(0);
        k.beginSubmit();
        const enc = device.createCommandEncoder();
        for (const u of k.frameUnits(t, { accum: rig.accum, counters: rig.counters })) u.encode(enc);
        device.queue.submit([enc.finish()]);
        await device.queue.onSubmittedWorkDone();
      }
      expect(parities).toEqual([1, 0].slice(0, c.frames));  // the G-buffer parity flips per advanced frame
      const sum = new Float32Array(await readBuffer(device, rig.accum, 64 * 64 * 16));
      const mean = new Float32Array(64 * 64 * 3);
      for (let i = 0; i < 64 * 64; i++) for (let ch = 0; ch < 3; ch++) mean[3 * i + ch] = sum[4 * i + ch] / c.frames;
      const fin = await k.readReservoirs('final');
      expect(hashF32(mean)).toBe(GOLDEN[c.name].image);
      expect(hashF32(new Float32Array(fin.buffer))).toBe(GOLDEN[c.name].final);
      rig.destroy();
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
