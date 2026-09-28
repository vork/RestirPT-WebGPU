// T9 light sampling (gap-light U1/U2, validation-harness T9a–e) on the production WGSL (lights/measure.wgsl and the
// modules it includes), both GPU lanes, through a probe kernel:
//   (a) GPU alias selection ≡ CPU aliasSample bit-for-bit, and χ² against the REALIZED pmf;
//       rect / disk / triangle position samplers are uniform in area (χ² in the samplers' own parameters);
//   (b) E[Λ·cosθ_x/q] at a point = the analytic irradiance (rect polygon formula, disk on axis, triangle polygon);
//       q, p1 = q·r²/|cosθ_z| consistent with the record area and the realized pmf;
//   (c) NEE/BSDF MIS partition of unity: ω1 (NEE-time code) + ω2 (hit-time code, p1 recomputed at x) = 1 for emissive
//       triangles (every mode) and Mode-B analytic area lights (reserved for M3b); Mode A / delta lights: ω1 ≡ 1 by
//       flag even with p2 > 0 (negative control: the balance formula with the delta pmf would give ω1 < 1);
//   (e) the realized pmf read back from the GPU records sums to 1.
import { afterAll, describe, expect, it } from 'vitest';
import { buildBvh } from '../../src/core/bvh/sah-builder.ts';
import { composeWgsl, createCheckedShaderModule } from '../../src/core/gpu/wgsl-composer.ts';
import { readBuffer } from '../../src/core/gpu/readback.ts';
import { aliasSample } from '../../src/core/render/alias.ts';
import { computeRenderOrigin } from '../../src/core/render/frame-uniforms.ts';
import { LIGHT_REC, LIGHT_REC_WORDS, LightsGpu, LP_MODE_A } from '../../src/core/render/lights-gpu.ts';
import { SceneGpu, recentrePositions } from '../../src/core/render/scene-gpu.ts';
import { shaderSources } from '../../src/core/shaders/index.ts';
import type { LightData, SceneData } from '../../src/core/scene/types.ts';
import { getTestGpu, releaseTestGpu } from './device-factory.ts';
import { dot, lambert, lightMatrixToward, material, norm, polygonIrradiance, quadScene, sub, type V3 } from './pt-fixtures.ts';

afterAll(releaseTestGpu);

const PROBE = /* wgsl */ `
#include "common/rng.wgsl"
#include "lights/measure.wgsl"

struct ProbeParams { n: u32, mode: u32, seed: u32, lpFlags: u32, x: vec3f, p2scale: f32, nx: vec3f, pad: f32 }
@group(2) @binding(0) var<uniform> pp: ProbeParams;
@group(2) @binding(1) var<storage, read_write> outv: array<vec4f>;
@group(2) @binding(2) var<storage, read> hashes: array<u32>;

@compute @workgroup_size(64)
fn probe(@builtin(global_invocation_id) gid: vec3u) {
  let i = gid.x;
  if (i >= pp.n) { return; }
  if (pp.mode == 0u) {                       // alias selection for given hashes
    var slot = lightsParams.cur;
    outv[i] = vec4f(f32(alias_sample(slot, hashes[2u * i], hashes[2u * i + 1u])), 0.0, 0.0, 0.0);
    return;
  }
  let h = pcg3d(vec3u(pp.seed, i, 77u));
  let u = vec2f(u32_to_unit(h.y), u32_to_unit(h.z));
  let ls = light_sample(pp.x, h.x, pcg3d(vec3u(pp.seed, i, 78u)).x, u);
  let cosX = max(dot(pp.nx, ls.dir), 0.0);
  let p2 = cosX * INV_PI * pp.p2scale;
  let w1 = mis_w1(ls, p2, 1u);
  var p1hit = 0.0;
  if (ls.kind == LT_TRI) { p1hit = tri_light_p1(pp.x, ls.pos, tri_normal(ls.prim), ls.prim); }
  else if (ls.kind == LT_RECT || ls.kind == LT_DISK) { p1hit = analytic_area_p1(pp.x, ls.entry, ls.pos); }
  let w2 = mis_w2(p1hit, p2, 1u);
  outv[4u * i] = vec4f(ls.pos, bitcast<f32>(ls.entry));
  outv[4u * i + 1u] = vec4f(select(vec3f(0.0), ls.Lambda * cosX / ls.q, ls.valid), ls.p1);
  outv[4u * i + 2u] = vec4f(ls.q, ls.cosZ, select(ls.dist, -1.0, ls.isInf), bitcast<f32>(ls.kind | select(0u, 256u, ls.valid) | select(0u, 512u, ls.isDelta)));
  outv[4u * i + 3u] = vec4f(w1, w2, p1hit, p2);
}
`;

interface Rec { pos: V3; axisU: V3; axisV: V3; halfU: number; halfV: number }
interface Probe { gpu: SceneGpu; lights: LightsGpu; rec(i: number): Rec; run(mode: number, n: number, opts?: { x?: V3; nx?: V3; seed?: number; p2scale?: number; hashes?: Uint32Array }): Promise<Float32Array>; destroy(): void }

async function probeRig(scene: SceneData, lightMode: 'A' | 'B' = 'A'): Promise<Probe> {
  const { device, features, wgslLanguageFeatures } = await getTestGpu();
  const origin = computeRenderOrigin(scene.bounds);
  const gpu = await SceneGpu.create(device, scene, origin, {
    textureMode: 'validation', watertight: true, buildBvh: async (p, i) => buildBvh(p, i, { mt: true, woop: true }), features, wgslLanguageFeatures,
  });
  const lights = new LightsGpu(device, scene, origin, recentrePositions(scene.geometry.positions, origin), { lightMode });
  const c = GPUShaderStage.COMPUTE;
  const g0 = device.createBindGroupLayout({ entries: [
    { binding: 5, visibility: c, buffer: { type: 'uniform' } }, { binding: 6, visibility: c, buffer: { type: 'read-only-storage' } }] });
  const g1 = device.createBindGroupLayout({ entries: gpu.layoutEntries(c) });
  const g2 = device.createBindGroupLayout({ entries: [
    { binding: 0, visibility: c, buffer: { type: 'uniform' } }, { binding: 1, visibility: c, buffer: { type: 'storage' } },
    { binding: 2, visibility: c, buffer: { type: 'read-only-storage' } }] });
  const shader = composeWgsl('tests/lights-probe.wgsl', {
    sources: { ...shaderSources, 'tests/lights-probe.wgsl': PROBE },
    defines: { ...gpu.defines(1), LIGHTS_GROUP: 0, LIGHTS_BINDING: 5 }, features, wgslLanguageFeatures,
  });
  const module = await createCheckedShaderModule(device, shader, 'lights-probe');
  const pipeline = await device.createComputePipelineAsync({ layout: device.createPipelineLayout({ bindGroupLayouts: [g0, g1, g2] }), compute: { module, entryPoint: 'probe' } });
  const bg0 = device.createBindGroup({ layout: g0, entries: [{ binding: 5, resource: { buffer: lights.params } }, { binding: 6, resource: { buffer: lights.records } }] });
  const bg1 = device.createBindGroup({ layout: g1, entries: gpu.bindGroupEntries() });
  const params = device.createBuffer({ size: 48, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
  const toInternal = (p: V3): V3 => [p[0] - origin[0], p[1] - origin[1], p[2] - origin[2]];
  return {
    gpu, lights,
    rec(i) {
      const st = lights.state, b = st.curSlot.lightOff + i * LIGHT_REC_WORDS;
      const f = new Float32Array(st.records.buffer);
      const v = (w: number): V3 => [f[b + w], f[b + w + 1], f[b + w + 2]];
      const p = v(LIGHT_REC.pos);
      return { pos: [p[0] + origin[0], p[1] + origin[1], p[2] + origin[2]], axisU: v(LIGHT_REC.axisU), axisV: v(LIGHT_REC.axisV), halfU: f[b + LIGHT_REC.halfU], halfV: f[b + LIGHT_REC.halfV] };
    },
    async run(mode, n, o = {}) {
      const perSample = mode === 0 ? 1 : 4;
      const out = device.createBuffer({ size: n * perSample * 16, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
      const hs = o.hashes ?? new Uint32Array(2);
      const hb = device.createBuffer({ size: Math.max(16, hs.byteLength), usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
      device.queue.writeBuffer(hb, 0, hs);
      const ab = new ArrayBuffer(48), u = new Uint32Array(ab), f = new Float32Array(ab);
      u[0] = n; u[1] = mode; u[2] = o.seed ?? 1; f.set(toInternal(o.x ?? [0, 0, 0]), 4); f[7] = o.p2scale ?? 1; f.set(o.nx ?? [0, 1, 0], 8);
      device.queue.writeBuffer(params, 0, ab);
      const bg2 = device.createBindGroup({ layout: g2, entries: [{ binding: 0, resource: { buffer: params } }, { binding: 1, resource: { buffer: out } }, { binding: 2, resource: { buffer: hb } }] });
      const enc = device.createCommandEncoder();
      const pass = enc.beginComputePass();
      pass.setPipeline(pipeline); pass.setBindGroup(0, bg0); pass.setBindGroup(1, bg1); pass.setBindGroup(2, bg2);
      pass.dispatchWorkgroups(Math.ceil(n / 64));
      pass.end();
      device.queue.submit([enc.finish()]);
      const r = new Float32Array(await readBuffer(device, out, n * perSample * 16));
      out.destroy(); hb.destroy();
      // positions back to world
      if (mode !== 0) for (let i = 0; i < n; i++) for (let k = 0; k < 3; k++) r[16 * i + k] += origin[k];
      return r;
    },
    destroy() { lights.destroy(); gpu.destroy(); params.destroy(); },
  };
}

const light = (o: Partial<LightData> & Pick<LightData, 'type' | 'id'>): LightData =>
  ({ name: 'l', color: [1, 1, 1], power: 10, exposure: 0, matrix: lightMatrixToward([0, -1, 0], [0, 1, 0]), visibleToCamera: false, ...o });

/** χ² statistic with expected counts e (pooled below 5 into one cell); returns [stat, dof]. */
function chi2(obs: number[], exp: number[]): [number, number] {
  let s = 0, dof = -1, po = 0, pe = 0;
  for (let i = 0; i < obs.length; i++) {
    if (exp[i] < 5) { po += obs[i]; pe += exp[i]; continue; }
    s += (obs[i] - exp[i]) ** 2 / exp[i]; dof++;
  }
  if (pe > 0) { s += (po - pe) ** 2 / pe; dof++; }
  return [s, dof];
}
/** Wilson–Hilferty upper quantile of χ²_k at p ≈ 1e-4 (z = 3.72; the per-test α, well inside Šidák 0.01 over this file). */
const chi2Crit = (k: number): number => k * (1 - 2 / (9 * k) + 3.72 * Math.sqrt(2 / (9 * k))) ** 3;

// Scene: ground + an emissive quad (two triangles, prims 2, 3) at y = 1.2 over x ∈ [1, 1.6]; analytic lights below.
function scene(lights: LightData[]): SceneData {
  const Le: V3 = [3, 2, 1];
  return quadScene([
    { p: [[-5, 0, 5], [5, 0, 5], [5, 0, -5], [-5, 0, -5]], mat: 0 },
    { p: [[1, 1.2, 0.3], [1.6, 1.2, 0.3], [1.6, 1.2, -0.3], [1, 1.2, -0.3]], mat: 1 },
  ], [lambert(0.5), material({ emissiveFactor: Le, v1: { diffuse: [0, 0, 0], glossy: [0, 0, 0], roughness: 0.5, mix: 0 } })], lights);
}

const RECT = light({ id: 5, type: 'rect', power: 20, sizeX: 0.8, sizeY: 0.5, matrix: lightMatrixToward([0, -1, 0], [-1, 1, 0.2]) });
const DISK = light({ id: 9, type: 'disk', power: 8, sizeX: 0.6, matrix: lightMatrixToward(norm([0.2, -1, 0.1]), [0.2, 1.4, -1]) });
const POINT = light({ id: 12, type: 'point', power: 30, matrix: lightMatrixToward([0, -1, 0], [0.5, 2, 1]) });
const SPOT = light({ id: 13, type: 'spot', power: 30, spotSize: Math.PI / 3, spotBlend: 0.3, matrix: lightMatrixToward([0, -1, 0], [-0.5, 2, 1]) });
const SUN = light({ id: 20, type: 'sun', power: 2, matrix: lightMatrixToward(norm([0.3, -1, 0.2]), [0, 5, 0]) });

describe('T9 light sampling (lights/measure.wgsl)', () => {
  it('(a) GPU alias selection ≡ CPU aliasSample for the same hashes; realized pmf from GPU records sums to 1 (e)', async () => {
    const r = await probeRig(scene([RECT, DISK, POINT, SPOT, SUN]));
    const t = r.lights.state.table!;
    const n = 1 << 16;
    let s = 0x1234567;
    const hashes = new Uint32Array(2 * n).map(() => (s = (Math.imul(s, 1664525) + 1013904223) >>> 0));
    const got = await r.run(0, n, { hashes });
    let mismatch = 0;
    for (let i = 0; i < n; i++) if (got[4 * i] !== aliasSample(t, hashes[2 * i], hashes[2 * i + 1])) mismatch++;
    expect(mismatch).toBe(0);
    const { device } = await getTestGpu();
    const rec = new Float32Array(await readBuffer(device, r.lights.records, r.lights.records.size));
    const slot = r.lights.state.curSlot;
    let sum = 0;
    for (let e = 0; e < slot.nEntries; e++) sum += rec[slot.pmfOff + e];
    expect(Math.abs(sum - 1)).toBeLessThan(1e-6);
    r.destroy();
  });

  it('(a) selection χ² vs realized pmf and area-uniform samplers (rect, disk, triangle) χ²', async () => {
    const r = await probeRig(scene([RECT, DISK, POINT, SPOT, SUN]));
    const n = 1 << 20;
    const out = await r.run(1, n, { x: [0, 0, 0], seed: 3 });
    const slot = r.lights.state.curSlot;
    const rectRec = r.rec(0), diskRec = r.rec(1);
    expect(rectRec.halfU * rectRec.halfV).toBeCloseTo(0.4 * 0.25, 6);
    const counts = new Array(slot.nEntries).fill(0);
    const rect: number[] = new Array(256).fill(0), disk: number[] = new Array(128).fill(0), tri: number[] = new Array(256).fill(0);
    let nRect = 0, nDisk = 0, nTri = 0;
    const tris = r.gpu.scene.geometry;
    for (let i = 0; i < n; i++) {
      const e = new Uint32Array(out.buffer, (16 * i + 3) * 4, 1)[0];
      counts[e]++;
      const p: V3 = [out[16 * i], out[16 * i + 1], out[16 * i + 2]];
      if (e === 0) { // RECT (id 5 is the smallest id): light-local (u, v) uniform
        const L = rectRec, d = sub(p, L.pos);
        const a = dot(d, L.axisU) / L.halfU, b = dot(d, L.axisV) / L.halfV;
        const bi = Math.min(15, Math.floor((a + 1) * 8)), bj = Math.min(15, Math.floor((b + 1) * 8));
        rect[16 * bi + bj]++; nRect++;
      } else if (e === 1) { // DISK: uniform in area ⇔ (ρ², φ) uniform
        const L = diskRec, d = sub(p, L.pos);
        const x = dot(d, L.axisU) / L.halfU, y = dot(d, L.axisV) / L.halfV;
        const rr = Math.min(0.999999, x * x + y * y), phi = Math.atan2(y, x) + Math.PI;
        disk[16 * Math.floor(rr * 8) + Math.min(15, Math.floor(phi / (2 * Math.PI) * 16))]++; nDisk++;
      } else if (e >= slot.nAnalytic) { // triangle entries: invert the barycentric warp (s = (1−b0)², w = b2/(1−b0))
        const prim = r.lights.state.tris.primIds[e - slot.nAnalytic];
        const ix = [tris.indices[3 * prim], tris.indices[3 * prim + 1], tris.indices[3 * prim + 2]];
        const P = ix.map((k) => [tris.positions[3 * k], tris.positions[3 * k + 1], tris.positions[3 * k + 2]] as V3);
        const e1 = sub(P[1], P[0]), e2 = sub(P[2], P[0]), q = sub(p, P[0]);
        const d00 = dot(e1, e1), d01 = dot(e1, e2), d11 = dot(e2, e2), d20 = dot(q, e1), d21 = dot(q, e2);
        const den = d00 * d11 - d01 * d01;
        const b1 = (d11 * d20 - d01 * d21) / den, b2 = (d00 * d21 - d01 * d20) / den;
        const sq = b1 + b2; // = 1 − b0 = √u1
        const s2 = Math.min(0.999999, sq * sq), w = Math.min(0.999999, Math.max(0, b2 / sq));
        tri[16 * Math.floor(s2 * 16) + Math.floor(w * 16)]++; nTri++;
      }
    }
    const pmf = Array.from({ length: slot.nEntries }, (_, e) => r.lights.state.pmf(e));
    const [cs, cd] = chi2(counts, pmf.map((p) => p * n));
    expect(cs, `selection χ² (dof ${cd})`).toBeLessThan(chi2Crit(cd));
    for (const [name, h, m] of [['rect', rect, nRect], ['disk', disk, nDisk], ['triangle', tri, nTri]] as const) {
      expect(m, name).toBeGreaterThan(5000);
      const [st, dof] = chi2(h, h.map(() => m / h.length));
      expect(st, `${name} χ² (dof ${dof}, n ${m})`).toBeLessThan(chi2Crit(dof));
    }
    r.destroy();
  });

  it('(b) E[Λ·cosθ_x/q] = analytic irradiance for rect, disk (on axis) and the emissive quad; p1 = q r²/|cosθ_z|', async () => {
    const x: V3 = [-0.9, 0, 0.1];
    const cases: { lights: LightData[]; want: (x: V3) => number; label: string }[] = [
      { label: 'rect', lights: [RECT], want: (p) => 20 / (Math.PI * 0.4) * polygonIrradiance(p, [0, 1, 0], [[-1.25, 1, -0.2], [-0.75, 1, -0.2], [-0.75, 1, 0.6], [-1.25, 1, 0.6]]) },
      { label: 'disk', lights: [light({ id: 1, type: 'disk', power: 8, sizeX: 0.6, matrix: lightMatrixToward([0, -1, 0], [-0.9, 1.3, 0.1]) })],
        want: () => { const Le = 8 / (Math.PI * Math.PI / 4 * 0.36); return Math.PI * Le * 0.09 / (0.09 + 1.69); } },
      // L_e.r = 3 (the red channel is checked)
      { label: 'emissive quad', lights: [], want: (p) => 3 * polygonIrradiance(p, [0, 1, 0], [[1, 1.2, 0.3], [1.6, 1.2, 0.3], [1.6, 1.2, -0.3], [1, 1.2, -0.3]]) },
    ];
    for (const c of cases) {
      const r = await probeRig(scene(c.lights));
      const n = 1 << 20;
      const out = await r.run(1, n, { x, seed: 9 });
      // restrict to this light's entries: the emissive quad is an entry in every case; only sum the target light
      const slot = r.lights.state.curSlot;
      let sum = 0, sum2 = 0, bad = 0;
      for (let i = 0; i < n; i++) {
        const e = new Uint32Array(out.buffer, (16 * i + 3) * 4, 1)[0];
        const isTarget = c.label === 'emissive quad' ? e >= slot.nAnalytic : e < slot.nAnalytic;
        const v = isTarget ? out[16 * i + 4] : 0; // red channel (L_e.r = 3 for the quad)
        sum += v; sum2 += v * v;
        const q = out[16 * i + 8], cosZ = out[16 * i + 9], dist = out[16 * i + 10], p1 = out[16 * i + 7];
        if (cosZ > 0 && Math.abs(p1 - q * dist * dist / cosZ) > 1e-5 * p1) bad++;
      }
      const mean = sum / n, se = Math.sqrt((sum2 / n - mean * mean) / n);
      const want = c.want(x);
      expect(bad, `${c.label}: p1 consistency`).toBe(0);
      expect(Math.abs(mean - want), `${c.label}: ${mean} vs ${want} (se ${se})`).toBeLessThan(5 * se);
      expect(Math.abs(mean - want) / want, c.label).toBeLessThan(3e-3);
      r.destroy();
    }
  });

  it('(c) MIS partition ω1 + ω2 = 1 (emissive triangles in Mode A and B, analytic area lights in Mode B); ω1 ≡ 1 for delta / Mode A', async () => {
    for (const mode of ['A', 'B'] as const) {
      const r = await probeRig(scene([RECT, DISK, POINT, SPOT, SUN]), mode);
      expect(r.lights.state.paramsBytes().byteLength).toBeGreaterThan(0);
      const n = 1 << 18;
      for (const p2scale of [1, 0.05, 20]) {
        const out = await r.run(1, n, { x: [0.1, 0, -0.2], seed: 17 + p2scale, p2scale });
        let worst = 0, checked = 0, deltaOk = 0, deltaN = 0, areaA = 0;
        for (let i = 0; i < n; i++) {
          const flags = new Uint32Array(out.buffer, (16 * i + 11) * 4, 1)[0];
          const kind = flags & 255, valid = (flags & 256) !== 0, isDelta = (flags & 512) !== 0;
          if (!valid) continue;
          const w1 = out[16 * i + 12], w2 = out[16 * i + 13], p1hit = out[16 * i + 14], p2 = out[16 * i + 15], p1 = out[16 * i + 7];
          if (isDelta) { deltaN++; if (w1 === 1) deltaOk++; continue; }
          if ((kind === 3 || kind === 4) && mode === 'A') { if (w1 === 1) areaA++; else areaA -= 1e9; continue; }
          if (!(p1 > 0) || out[16 * i + 9] <= 0) continue;
          expect(Math.abs(p1hit - p1) / p1).toBeLessThan(2e-5); // hit-time p1 recomputed at x = NEE-time p1
          worst = Math.max(worst, Math.abs(w1 + w2 - 1));
          if (p2 > 0) expect(w1).toBeLessThan(1); // genuinely two-technique
          checked++;
        }
        expect(checked).toBeGreaterThan(1000);
        expect(worst, `mode ${mode} p2×${p2scale}`).toBeLessThan(1e-5);
        expect(deltaOk).toBe(deltaN);
        expect(deltaN).toBeGreaterThan(1000);
        if (mode === 'A') expect(areaA).toBeGreaterThan(1000);
        // negative control: the balance formula with the delta light's pmf (a mass, not a density) is not 1
        const pmfPoint = r.lights.state.pmf(2);
        expect(pmfPoint / (pmfPoint + 0.3)).toBeLessThan(1);
      }
      expect((new Uint32Array(r.lights.state.paramsBytes())[27] & LP_MODE_A) !== 0).toBe(mode === 'A');
      r.destroy();
    }
  });
});
