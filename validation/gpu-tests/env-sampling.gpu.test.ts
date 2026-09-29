// Env importance sampling on the GPU (plan §7.4 M3c; env §5.1), production WGSL (lights/env-sample.wgsl), both lanes:
//   ENV-U3  realized-pdf identity (every NEE sample lies in its own cell, so p1Env(uv) ≡ pdfUV[i,j]; uv ∈ [0,1)²) and a
//           GPU χ² of the sampled cells against the REALIZED pdf (cells merged to ≥ 50 expected), studio_small_09 1k
//           and a synthetic high-contrast map: p ≥ 1e-3 in ≥ 99/100 seeds; the direction round trip
//           cell(envUV(envDir(uv))) ≠ (i,j) rate is recorded (accepted residual, env §2.4);
//   ENV-U4  support: L_env(ω) > 0 ⇒ p1Env(ω) > 0 outside the pole cap, over random + seam + pole + pole-wrap
//           directions, on floored maps incl. a single-bright-texel map (0 violations); the no-floor table of a map with
//           a 1e-9 region has violations (the test can fail);
//   ENV-U6  NEE/BSDF partition at fixed shading points (Lambert, GGX α ∈ {0.2, 0.5}) with pmf[ENV] ∈ {1, 0.3}:
//           E[ω1·NEE_ENV] + E[ω2·BSDF_ENV] = ∫ f_cos L_env dω (f64 quadrature of the GPU-exact 8-bit bilinear map),
//           |z| ≤ 4 with a relative SE ≤ 3e-5 (so a 1e-4 bias is resolved); both parts must be positive (both
//           techniques contribute).
import { afterAll, describe, expect, it } from 'vitest';
import { buildBvh } from '../../src/core/bvh/sah-builder.ts';
import { composeWgsl, createCheckedShaderModule } from '../../src/core/gpu/wgsl-composer.ts';
import { readBuffer } from '../../src/core/gpu/readback.ts';
import { createEnvResources, destroyEnvResources, envBindGroupEntries, envDefines, envImportanceFor, type EnvGpuResources } from '../../src/core/render/env-gpu.ts';
import { computeRenderOrigin } from '../../src/core/render/frame-uniforms.ts';
import { LightsGpu, LUT_RECORDS_BASE } from '../../src/core/render/lights-gpu.ts';
import { lutDefines } from '../../src/core/render/luts/lut-layout.ts';
import { applyEnvLighting } from '../../src/core/render/pt-kernel.ts';
import { SceneGpu, recentrePositions } from '../../src/core/render/scene-gpu.ts';
import { envPowerProxy, type EnvImportance } from '../../src/core/scene/env/env-importance.ts';
import { shaderSources } from '../../src/core/shaders/index.ts';
import type { EnvironmentData, LightData, MaterialData, SceneData } from '../../src/core/scene/types.ts';
import { chi2Merged } from '../../src/core/render/chi2.ts';
import { evalLocal, type MatParams } from '../../tests/material/bsdf-ref.ts';
import { getTestGpu, lane, releaseTestGpu } from './device-factory.ts';
import { envDir, envLookup, envToBlender, envUV, loadHdri, sunTexelEnv, synthEnvData, type V3 } from './env-fixtures.ts';
import { material, quadScene } from './pt-fixtures.ts';

afterAll(releaseTestGpu);

const PROBE = /* wgsl */ `
#include "common/rng-path.wgsl"
#include "lights/env-sample.wgsl"
#include "material/material-eval.wgsl"

struct ProbeParams { n: u32, seed: u32, K: u32, mat: u32, V: vec3f, pad: f32 }
@group(2) @binding(0) var<uniform> pp: ProbeParams;
@group(2) @binding(1) var<storage, read_write> hist: array<atomic<u32>>;
@group(2) @binding(2) var<storage, read_write> io: array<vec4f>;
@group(2) @binding(3) var<storage, read_write> cnt: array<atomic<u32>, 8>;

@compute @workgroup_size(256)
fn sample_hist(@builtin(global_invocation_id) gid: vec3u) {
  let t = gid.x;
  if (t >= pp.n) { return; }
  let W = env_Wm();
  var offCell = 0u; var outRange = 0u; var roundTrip = 0u; var poleCap = 0u;
  for (var k = 0u; k < pp.K; k++) {
    let h = pcg3d(vec3u(pp.seed, t, k));
    let c = env_sample_cell(h.x, h.y, h.z);
    atomicAdd(&hist[c.i * W + c.j], 1u);
    let cc = env_cell_of(c.uv);
    if (cc.x != c.j || cc.y != c.i) { offCell++; }
    if (!(c.uv.x >= 0.0 && c.uv.x < 1.0 && c.uv.y >= 0.0 && c.uv.y < 1.0)) { outRange++; }
    let rt = env_cell_of(envUV(envDir(c.uv, envParams.cg, envParams.sg), envParams.cg, envParams.sg));
    if (rt.x != c.j || rt.y != c.i) { roundTrip++; }
    if (c.s < ENV_POLE_CAP) { poleCap++; }
  }
  if (offCell != 0u) { atomicAdd(&cnt[0], offCell); }
  if (outRange != 0u) { atomicAdd(&cnt[1], outRange); }
  if (roundTrip != 0u) { atomicAdd(&cnt[2], roundTrip); }
  if (poleCap != 0u) { atomicAdd(&cnt[3], poleCap); }
}

@compute @workgroup_size(64)
fn support(@builtin(global_invocation_id) gid: vec3u) {
  let i = gid.x;
  if (i >= pp.n) { return; }
  let d = io[i].xyz;
  let L = envRadiance(envUV(d, envParams.cg, envParams.sg));
  let b = envToBlender(d, envParams.cg, envParams.sg);
  io[i] = vec4f(max(max(L.r, L.g), L.b), p1Env(d), length(b.xy), 0.0);
}

@compute @workgroup_size(256)
fn mis_partition(@builtin(global_invocation_id) gid: vec3u) {
  let t = gid.x;
  if (t >= pp.n) { return; }
  let ng = vec3f(0.0, 1.0, 0.0);
  let m = material_eval_record(sceneMaterials[pp.mat], vec2f(0.0), vec4f(1.0), ng, ng, pp.V);
  var sN = vec3f(0.0);
  var sB = vec3f(0.0);
  for (var k = 0u; k < pp.K; k++) {
    let seed = pcg3d(vec3u(pp.seed, t, k)).xy;
    let ls = nee_sample(vec3f(0.0), path_hash(seed, 1u, SLOT_SEL), path_hash(seed, 1u, SLOT_SEL2),
                        vec3u(path_hash(seed, 1u, SLOT_L0), path_hash(seed, 1u, SLOT_L1), path_hash(seed, 1u, SLOT_L2)));
    if (ls.valid && ls.kind == LT_ENV) {
      let ev = bsdf_eval(m, pp.V, ls.dir);
      sN += (nee_mis_w1(ls, ev.pdf_marginal, 1u) / ls.q) * ev.f_cos * ls.Lambda;
    }
    let bs = bsdf_sample(m, pp.V, path_bsdf_u4(seed, 1u));
    if (bs.valid) {
      let w2 = env_bsdf_mis_weight(bs.L, bs.pdf_marginal, 1u, bs.is_delta);
      sB += w2 * bs.weight * envRadiance(envUV(bs.L, envParams.cg, envParams.sg));
    }
  }
  io[2u * t] = vec4f(sN, 0.0);
  io[2u * t + 1u] = vec4f(sB, 0.0);
}
`;

interface Rig {
  device: GPUDevice;
  env: EnvGpuResources;
  lights: LightsGpu;
  table: EnvImportance;
  run(entry: 'sample_hist' | 'support' | 'mis_partition', p: { n: number; seed?: number; K?: number; mat?: number; V?: V3 }, wg: number): Promise<void>;
  hist: GPUBuffer; io: GPUBuffer; cnt: GPUBuffer;
  destroy(): void;
}

const V1 = (name: string, o: Partial<NonNullable<MaterialData['v1']>>): MaterialData =>
  material({ name, v1: { diffuse: [0.8, 0.8, 0.8], glossy: [0, 0, 0], roughness: 0.5, mix: 0, ...o } });

/** Materials of the partition scene: 0 Lambert ρ 0.8, 1 V1 GGX α 0.2 (r = √0.2), 2 V1 GGX α 0.5 (F ≡ 1). */
const MATS: { mat: MaterialData; ref: MatParams }[] = [
  { mat: V1('lambert', {}), ref: { model: 0, diffuse: [0.8, 0.8, 0.8], glossy: [0, 0, 0], roughness: 0.5, mix: 0 } },
  { mat: V1('ggx0.2', { glossy: [1, 1, 1], roughness: Math.sqrt(0.2), mix: 1 }), ref: { model: 0, diffuse: [0.8, 0.8, 0.8], glossy: [1, 1, 1], roughness: Math.sqrt(0.2), mix: 1 } },
  { mat: V1('ggx0.5', { glossy: [1, 1, 1], roughness: Math.sqrt(0.5), mix: 1 }), ref: { model: 0, diffuse: [0.8, 0.8, 0.8], glossy: [1, 1, 1], roughness: Math.sqrt(0.5), mix: 1 } },
];

async function rig(envData: EnvironmentData, o: { lights?: LightData[]; floors?: boolean; histCells?: number; ioVec4?: number } = {}): Promise<Rig> {
  const { device, features, wgslLanguageFeatures } = await getTestGpu();
  const S = 2;
  const scene: SceneData = { ...quadScene([{ p: [[-S, 0, S], [S, 0, S], [S, 0, -S], [-S, 0, -S]], mat: 0 }], MATS.map((m) => m.mat), o.lights ?? []), env: envData };
  const origin = computeRenderOrigin(scene.bounds);
  const gpu = await SceneGpu.create(device, scene, origin, {
    textureMode: 'validation', watertight: true, buildBvh: async (p, i) => buildBvh(p, i, { mt: true, woop: true }), features, wgslLanguageFeatures,
  });
  const env = await createEnvResources(device, envData, 'env-sampling');
  const lights = new LightsGpu(device, scene, origin, recentrePositions(scene.geometry.positions, origin), { lightMode: 'A' });
  applyEnvLighting(lights, env, { floors: o.floors });
  const table = envImportanceFor(env, { floors: o.floors })!;
  const c = GPUShaderStage.COMPUTE;
  const g0 = device.createBindGroupLayout({ entries: [
    { binding: 1, visibility: c, buffer: { type: 'uniform' } }, { binding: 2, visibility: c, texture: { sampleType: 'float' } }, { binding: 3, visibility: c, sampler: { type: 'filtering' } },
    { binding: 5, visibility: c, buffer: { type: 'uniform' } }, { binding: 6, visibility: c, buffer: { type: 'read-only-storage' } }] });
  const g1 = device.createBindGroupLayout({ entries: gpu.layoutEntries(c) });
  const g2 = device.createBindGroupLayout({ entries: [
    { binding: 0, visibility: c, buffer: { type: 'uniform' } }, { binding: 1, visibility: c, buffer: { type: 'storage' } },
    { binding: 2, visibility: c, buffer: { type: 'storage' } }, { binding: 3, visibility: c, buffer: { type: 'storage' } }] });
  const shader = composeWgsl('tests/env-sampling-probe.wgsl', {
    sources: { ...shaderSources, 'tests/env-sampling-probe.wgsl': PROBE },
    defines: { ...gpu.defines(1), ...envDefines(0, 1), LIGHTS_GROUP: 0, LIGHTS_BINDING: 5, ...lutDefines({ base: LUT_RECORDS_BASE, recordsKind: 'u32' }) },
    features, wgslLanguageFeatures,
  });
  const module = await createCheckedShaderModule(device, shader, 'env-sampling-probe');
  const layout = device.createPipelineLayout({ bindGroupLayouts: [g0, g1, g2] });
  const pipes = new Map<string, GPUComputePipeline>();
  for (const e of ['sample_hist', 'support', 'mis_partition']) pipes.set(e, await device.createComputePipelineAsync({ layout, compute: { module, entryPoint: e } }));
  const prm = device.createBuffer({ size: 32, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
  const hist = device.createBuffer({ size: 4 * Math.max(4, o.histCells ?? table.Wm * table.Hm), usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
  const io = device.createBuffer({ size: 16 * Math.max(4, o.ioVec4 ?? 4), usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
  const cnt = device.createBuffer({ size: 32, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
  const bg1 = device.createBindGroup({ layout: g1, entries: gpu.bindGroupEntries() });
  const bg2 = device.createBindGroup({ layout: g2, entries: [
    { binding: 0, resource: { buffer: prm } }, { binding: 1, resource: { buffer: hist } }, { binding: 2, resource: { buffer: io } }, { binding: 3, resource: { buffer: cnt } }] });
  return {
    device, env, lights, table, hist, io, cnt,
    async run(entry, p, wg) {
      const bg0 = device.createBindGroup({ layout: g0, entries: [...envBindGroupEntries(env, 1), { binding: 5, resource: { buffer: lights.params } }, { binding: 6, resource: { buffer: lights.records } }] });
      const buf = new ArrayBuffer(32);
      const u = new Uint32Array(buf), f = new Float32Array(buf);
      u.set([p.n, p.seed ?? 0, p.K ?? 1, p.mat ?? 0]);
      f.set(p.V ?? [0, 1, 0], 4);
      device.queue.writeBuffer(prm, 0, buf);
      const enc = device.createCommandEncoder();
      const pass = enc.beginComputePass();
      pass.setPipeline(pipes.get(entry)!);
      pass.setBindGroup(0, bg0); pass.setBindGroup(1, bg1); pass.setBindGroup(2, bg2);
      const groups = Math.ceil(p.n / wg);
      pass.dispatchWorkgroups(Math.min(groups, 65535), Math.ceil(groups / 65535));
      pass.end();
      device.queue.submit([enc.finish()]);
      await device.queue.onSubmittedWorkDone();
    },
    destroy() { prm.destroy(); hist.destroy(); io.destroy(); cnt.destroy(); lights.destroy(); gpu.destroy(); destroyEnvResources(env); },
  };
}

describe('env importance sampling on the GPU (M3c)', () => {
  it('ENV-U3: realized-pdf identity + GPU χ² vs the realized pdf (≥ 99/100 seeds at p ≥ 1e-3)', async () => {
    const studio = await loadHdri('studio_small_09_1k.hdr');
    const maps: { env: EnvironmentData; seeds: number; N: number }[] = [
      { env: synthEnvData(128, 64), seeds: 100, N: 2 ** 24 },
      ...(studio ? [{ env: studio, seeds: 100, N: 2 ** 26 }] : []),
    ];
    if (!studio) console.warn('ENV-U3: studio_small_09_1k.hdr missing (validation/assets/fetch_hdris.ts); synthetic map only');
    const report: Record<string, unknown>[] = [];
    for (const m of maps) {
      const r = await rig(m.env);
      const cells = r.table.Wm * r.table.Hm;
      const threads = 1 << 16, K = m.N / threads;
      const exp = new Float64Array(cells);
      for (let k = 0; k < cells; k++) exp[k] = (m.N * r.table.pdfUV[k]) / cells;
      const ps: number[] = [];
      const counters = [0, 0, 0, 0];
      const t0 = performance.now();
      for (let seed = 0; seed < m.seeds; seed++) {
        const enc = r.device.createCommandEncoder();
        enc.clearBuffer(r.hist); enc.clearBuffer(r.cnt);
        r.device.queue.submit([enc.finish()]);
        await r.run('sample_hist', { n: threads, seed: 1000 + seed, K }, 256);
        const h = new Uint32Array(await readBuffer(r.device, r.hist, 4 * cells));
        const cn = new Uint32Array(await readBuffer(r.device, r.cnt, 32));
        for (let k = 0; k < 4; k++) counters[k] += cn[k];
        let tot = 0;
        for (let k = 0; k < cells; k++) tot += h[k];
        expect(tot).toBe(m.N);
        ps.push(chi2Merged(h, exp, 50).p);
      }
      const ms = performance.now() - t0;
      const pass = ps.filter((p) => p >= 1e-3).length;
      const total = m.N * m.seeds;
      report.push({ map: m.env.name, grid: [r.table.Wm, r.table.Hm], samplesPerSeed: m.N, seeds: m.seeds, pass, minP: Math.min(...ps),
        offCell: counters[0], outOfRange: counters[1], roundTripCellMismatchRate: counters[2] / total, poleCapRate: counters[3] / total, ms: Math.round(ms) });
      expect(counters[0], 'NEE uv outside its sampled cell (cell clamp)').toBe(0);
      expect(counters[1], 'NEE uv outside [0,1)²').toBe(0);
      expect(counters[2] / total, 'direction round-trip cell mismatch rate (accepted residual)').toBeLessThan(1e-4);
      expect(pass, `${m.env.name}: χ² p ≥ 1e-3 in ${pass}/${m.seeds} seeds`).toBeGreaterThanOrEqual(Math.ceil(0.99 * m.seeds));
      r.destroy();
    }
    console.log('ENV-U3', lane(), JSON.stringify(report));
  }, 600_000);

  it('ENV-U4: L_env(ω) > 0 ⇒ p1Env(ω) > 0 outside the pole cap (random, seam, poles, pole-wrap rows)', async () => {
    const dirsFor = (W: number, H: number, g: number): V3[] => {
      const out: V3[] = [];
      let s = 12345;
      const rnd = () => { s = (Math.imul(s, 1664525) + 1013904223) >>> 0; return s / 4294967296; };
      for (let i = 0; i < 200_000; i++) {             // uniform sphere
        const z = 2 * rnd() - 1, ph = 2 * Math.PI * rnd(), rr = Math.sqrt(1 - z * z);
        out.push([rr * Math.cos(ph), z, rr * Math.sin(ph)]);
      }
      for (let i = 0; i < 20_000; i++) {              // seam (u → 0/1), pole caps, pole-wrap rows (half a texel)
        const k = i % 4;
        const u = k === 0 ? (rnd() - 0.5) * 1e-3 : rnd();
        const v = k === 1 ? rnd() * 0.5 / H : k === 2 ? 1 - rnd() * 0.5 / H : k === 3 ? (rnd() < 0.5 ? rnd() * 1e-5 : 1 - rnd() * 1e-5) : rnd();
        out.push(envDir(((u % 1) + 1) % 1, v, g));
      }
      out.push(envDir(0, 0.5, g), envDir(1, 0.5, g), [0, 1, 0], [0, -1, 0], envDir(0.5 / W, 0.25 / H, g));
      return out;
    };
    const cases: { name: string; env: EnvironmentData; floors?: boolean; expectViolations: boolean }[] = [
      { name: 'synthetic', env: synthEnvData(128, 64), expectViolations: false },
      { name: 'sun texel', env: sunTexelEnv(512, 256, 100, 191), expectViolations: false },
      { name: 'sun texel on seam', env: sunTexelEnv(512, 256, 0, 191), expectViolations: false },
      { name: 'sun texel in the top row', env: sunTexelEnv(512, 256, 300, 255), expectViolations: false },
    ];
    // Negative check: a 1e-9 region without floors rounds to q = 0 (support hole the test must see).
    const dim = synthEnvData(128, 64, 'dim-region');
    for (let r = 0; r < 20; r++) for (let c = 0; c < 128; c++) dim.texels.set([1e-9, 1e-9, 1e-9, 1], 4 * (r * 128 + c));
    cases.push({ name: 'dim region, no floors', env: dim, floors: false, expectViolations: true });
    const report: Record<string, unknown>[] = [];
    for (const cs of cases) {
      for (const g of [0, 0.7]) {
        const env = { ...cs.env, rotationZ: g };
        const dirs = dirsFor(env.width, env.height, g);
        const r = await rig(env, { floors: cs.floors, ioVec4: dirs.length });
        const data = new Float32Array(dirs.length * 4);
        dirs.forEach((d, i) => data.set([...d, 0], 4 * i));
        r.device.queue.writeBuffer(r.io, 0, data);
        await r.run('support', { n: dirs.length }, 64);
        const out = new Float32Array(await readBuffer(r.device, r.io, dirs.length * 16));
        let viol = 0, lit = 0, capped = 0;
        for (let i = 0; i < dirs.length; i++) {
          const L = out[4 * i], p1 = out[4 * i + 1], s = out[4 * i + 2];
          if (!(L > 0)) continue;
          lit++;
          if (s < 1e-6) { capped++; continue; }
          if (!(p1 > 0)) viol++;
        }
        report.push({ map: cs.name, gamma: g, dirs: dirs.length, lit, poleCapped: capped, violations: viol });
        if (cs.expectViolations) expect(viol, `${cs.name}: expected support holes`).toBeGreaterThan(0);
        else expect(viol, `${cs.name} γ ${g}: L > 0 but p1Env = 0`).toBe(0);
        r.destroy();
      }
    }
    console.log('ENV-U4', lane(), JSON.stringify(report));
  }, 300_000);

  it('ENV-U6: E[ω1·NEE_ENV] + E[ω2·BSDF_ENV] = ∫ f_cos L_env dω at pmf[ENV] ∈ {1, 0.3} (rel ≤ 1e-4, z ≤ 4)', async () => {
    const env = synthEnvData(128, 64, 'u6');
    env.rotationZ = 0.4;
    const Vs: V3[] = [[0, 1, 0], [Math.sin(1.0), Math.cos(1.0), 0]];     // normal incidence and 57° (world, N = +Y)
    // f64 quadrature: ∫ f_cos(V, ω) L(ω) dω over (u, v), dω = 2π² sin(πv) du dv, ss×ss midpoints per texel.
    const quad = (ref: MatParams, V: V3): V3 => {
      const ss = 12, W = env.width, H = env.height;
      const acc: V3 = [0, 0, 0];
      const Vl: V3 = [V[0], V[2], V[1]];                                   // local frame: N = +Y → +z (isotropic)
      for (let r = 0; r < H * ss; r++) {
        const v = (r + 0.5) / (H * ss);
        const w = 2 * Math.PI * Math.PI * Math.sin(Math.PI * v) / (W * ss * H * ss);
        for (let c = 0; c < W * ss; c++) {
          const u = (c + 0.5) / (W * ss);
          const d = envDir(u, v, env.rotationZ);
          if (d[1] <= 0) continue;
          const e = evalLocal(ref, Vl, [d[0], d[2], d[1]]);
          const fc = e.fD[0] + e.fS[0];
          if (!(fc > 0)) continue;
          const [uu, vv] = envUV(d, env.rotationZ);
          const L = envLookup(env, uu, vv, 8);
          for (let k = 0; k < 3; k++) acc[k] += w * (e.fD[k] + e.fS[k]) * L[k];
        }
      }
      return acc;
    };
    const report: Record<string, unknown>[] = [];
    for (const pEnvTarget of [1, 0.3]) {
      // pmf[ENV] = 0.3: a point light under the plane takes 70% of the NEE samples and contributes nothing here.
      const tmp = await rig(env);
      const phiEnv = envPowerProxy(tmp.table, 1, [1, 1, 1], tmp.lights.state.sceneRadius);
      tmp.destroy();
      const lights: LightData[] = pEnvTarget === 1 ? [] : [{
        id: 0, name: 'below', type: 'point', color: [1, 1, 1], power: phiEnv * 0.7 / 0.3, exposure: 0,
        matrix: new Float32Array([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, -1, 0, 1]), visibleToCamera: false,
      }];
      // Sized so that the relative SE of the sum is ≤ 3e-5 (a 1e-4 bias is then a > 3σ effect).
      const threads = 1 << 16, K = 256, reps = pEnvTarget === 1 ? 128 : 384;
      const r = await rig(env, { lights, ioVec4: 2 * threads });
      const pEnv = r.lights.state.envPmf();
      expect(Math.abs(pEnv - pEnvTarget)).toBeLessThan(1e-3);
      for (let mi = 0; mi < MATS.length; mi++) {
        for (const V of Vs) {
          const want = quad(MATS[mi].ref, V);
          const vals: number[][] = [];
          for (let rep = 0; rep < reps; rep++) {
            await r.run('mis_partition', { n: threads, seed: 777 + 131 * rep + 7919 * mi, K, mat: mi, V }, 256);
            const o = new Float32Array(await readBuffer(r.device, r.io, 2 * threads * 16));
            const sN = [0, 0, 0], sB = [0, 0, 0];
            for (let t = 0; t < threads; t++) for (let k = 0; k < 3; k++) { sN[k] += o[8 * t + k]; sB[k] += o[8 * t + 4 + k]; }
            const n = threads * K;
            vals.push([...sN.map((x) => x / n), ...sB.map((x) => x / n)]);
          }
          const mean = vals[0].map((_, j) => vals.reduce((a, v) => a + v[j], 0) / reps);
          const se = mean.map((m, j) => Math.sqrt(vals.reduce((a, v) => a + (v[j] - m) ** 2, 0) / (reps - 1) / reps));
          for (let k = 0; k < 3; k++) {
            const got = mean[k] + mean[3 + k];
            const s = Math.hypot(se[k], se[3 + k]);
            const rel = (got - want[k]) / want[k];
            const z = (got - want[k]) / s;
            report.push({ pEnv: Number(pEnv.toFixed(4)), mat: MATS[mi].mat.name, V: V.map((x) => Number(x.toFixed(3))), ch: k, want: want[k], nee: mean[k], bsdf: mean[3 + k], rel: Number(rel.toExponential(2)), z: Number(z.toFixed(2)), relSE: Number((s / want[k]).toExponential(2)) });
            expect(mean[k], 'NEE part > 0').toBeGreaterThan(0);
            expect(mean[3 + k], 'BSDF part > 0').toBeGreaterThan(0);
            expect(s / want[k], 'powered: relative SE ≤ 3e-5').toBeLessThanOrEqual(3e-5);
            expect(Math.abs(z), `p ${pEnvTarget} ${MATS[mi].mat.name} V ${V} ch ${k}: rel ${rel}, z ${z}`).toBeLessThanOrEqual(4);
            expect(Math.abs(rel), `p ${pEnvTarget} ${MATS[mi].mat.name} ch ${k}: rel ${rel}`).toBeLessThanOrEqual(1e-4 + 4 * s / want[k]);
          }
        }
      }
      r.destroy();
    }
    console.log('ENV-U6', lane(), JSON.stringify(report));
  }, 900_000);
});

void envToBlender;
