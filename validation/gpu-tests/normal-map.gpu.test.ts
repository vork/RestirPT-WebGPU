// M7 normal maps on the GPU (docs/decisions/m7-api.md §6.1), Chrome lane authoritative:
//   U-NM-1  scene_surface (NORMAL_MAP) at random (primId, u, v) vs the f64 Cycles Normal Map node on the CPU-decoded
//           vertices / tangents and the nearest texel: smooth + flat faces, strength 0.4 / 1 / 1.7, mirrored UVs (w = −1),
//           backfacing hits (N inverted), a material without a normal map (ns = nsm = the smooth normal), both vertex
//           formats (Q oct-15 tangents, lossless f32).
//   U-NM-2  BSDF bump-shadowing (bsdf.wgsl bsdf_bump_ok): for random (V, L) and mapped / smooth / geometric normals the
//           NEE f and the diffuse lobe are the no-bump values times 1[bump ok], the glossy / glass lobes are unchanged,
//           bsdf_query agrees with bsdf_eval, and bsdf_sample equals the no-bump sampler except that diffuse samples
//           failing the test are rejected.
//   U-NM-3  PT closed form: a flat Lambert plane with a constant-tilt normal map (θ = 40°) under a constant env, b = 1:
//           L = ρ(1 + cos θ')/2 (the hemispheres of N and Ng intersect; θ' = the Cycles-strength tilt), at strength 1 and 0.5.
import { afterAll, describe, expect, it } from 'vitest';
import { getTestGpu, lane, releaseTestGpu } from './device-factory.ts';
import { composeWgsl, createCheckedShaderModule } from '../../src/core/gpu/wgsl-composer.ts';
import { readBuffer } from '../../src/core/gpu/readback.ts';
import { shaderSources } from '../../src/core/shaders/index.ts';
import { buildBvh } from '../../src/core/bvh/sah-builder.ts';
import { computeRenderOrigin } from '../../src/core/render/frame-uniforms.ts';
import { SceneGpu, recentrePositions } from '../../src/core/render/scene-gpu.ts';
import { createLutBuffer, lutDefines } from '../../src/core/render/luts/lut-layout.ts';
import { quantizeScene } from '../../src/core/scene/quantize.ts';
import { loadMikkTSpace } from '../../src/core/scene/gltf-loader.ts';
import { sceneTangents } from '../../src/core/scene/tangents.ts';
import { TRI_FLAT, type MaterialData, type SceneData, type TextureData } from '../../src/core/scene/types.ts';
import { MeshBuilder, principled, texture } from '../scenes/kit-core.ts';
import { normalMapCycles, smoothSheet, smoothSphere, tiltMap } from '../scenes/m7-kit.ts';
import { batchStats, lookDown, ptRig, quadScene } from './pt-fixtures.ts';
import { rng } from '../../tests/bvh/fixtures.ts';

const SURF = /* wgsl */ `
#include "scene/scene-data.wgsl"
@group(1) @binding(0) var<storage, read> probes: array<vec4f>;
@group(1) @binding(1) var<storage, read_write> outS: array<vec4f>;
@group(1) @binding(2) var<uniform> np: vec4u;
@compute @workgroup_size(64) fn surfaces(@builtin(global_invocation_id) gid: vec3u) {
  let i = gid.x;
  if (i >= np.x) { return; }
  let q = probes[i];
  let prim = bitcast<u32>(q.x);
  var d = vec3f(0.0);
  if (q.w > 0.5) { d = scene_surface(prim, q.y, q.z, vec3f(0.0)).ng; }   // a ray arriving from behind
  let s = scene_surface(prim, q.y, q.z, d);
  outS[4u * i] = vec4f(s.ns, f32(s.matId));
  outS[4u * i + 1u] = vec4f(s.nsm, select(0.0, 1.0, s.backfacing));
  outS[4u * i + 2u] = vec4f(s.ng, bitcast<f32>(s.triFlags));
  outS[4u * i + 3u] = vec4f(s.uv, 0.0, 0.0);
}`;

const ENC = (n: number[]) => n.map((x) => (x + 1) / 2 * 255);

/** Random unit vectors in the +z hemisphere tilted ≤ 50°, encoded as a nearest-filtered RGBA8 normal map. */
function randomNormalMap(size: number, seed: number): TextureData {
  const R = rng(seed);
  return texture(`rnd${seed}`, size, size, () => {
    const th = 0.87 * Math.sqrt(R()), ph = 2 * Math.PI * R();
    const e = ENC([Math.sin(th) * Math.cos(ph), Math.sin(th) * Math.sin(ph), Math.cos(th)]);
    return [e[0], e[1], e[2], 255];
  }, { filter: 'nearest' });
}

/** Smooth sphere (nm, s = 1), flat quad (nm, s = 0.4), mirrored-u quad (nm, s = 1.7), smooth sheet (nm, s = 1), plain sphere. */
async function nmScene(mode: 'quantized' | 'lossless'): Promise<SceneData> {
  const textures = [randomNormalMap(8, 5), randomNormalMap(16, 6)];
  const mats: MaterialData[] = [
    principled('nm1', { normalTexture: { texture: 0, texCoord: 0, scale: 1 } }),
    principled('nm04', { normalTexture: { texture: 1, texCoord: 0, scale: 0.4 } }),
    principled('nm17', { normalTexture: { texture: 0, texCoord: 0, scale: 1.7 } }),
    principled('plain'),
  ];
  const mb = new MeshBuilder();
  smoothSphere(mb, [0, 1, 0], 0.8, 10, 20, 0, { uvScale: [2, 1] });
  mb.quad([-3, 0, 1], [-1, 0, 1], [-1, 0, -1], [-3, 0, -1], 1, [1.5, 1.5]);
  const base = mb.pos.length / 3;
  mb.quad([1, 0, 1], [3, 0, 1], [3, 0, -1], [1, 0, -1], 2);
  for (let k = 0; k < 4; k++) mb.uv[2 * (base + k)] = 1 - mb.uv[2 * (base + k)];   // mirrored u ⇒ MikkTSpace w = −1
  smoothSheet(mb, -3, 3, 2, 5, 0.2, 0.25, 1.3, 1.1, 18, 0, { uvScale: [3, 1.5] });
  smoothSphere(mb, [0, 1, -3], 0.6, 6, 12, 3);
  const g = mb.build(mats);
  const raw: SceneData = { name: `nm-${mode}`, geometry: g, materials: mats, textures, lights: [], cameras: [], bounds: { min: [-3, 0, -4], max: [3, 2, 5] }, warnings: [] };
  const q = quantizeScene(raw, { mode }).scene;
  return sceneTangents(q, await loadMikkTSpace(), { quantized: mode === 'quantized', flatFaceNormals: true }).scene;
}

describe(`M7 normal maps (${lane()})`, () => {
  afterAll(releaseTestGpu);
  const report: Record<string, unknown> = { lane: lane() };
  afterAll(() => console.log('NM_REPORT', JSON.stringify(report)));

  for (const mode of ['quantized', 'lossless'] as const) {
    it(`U-NM-1 (${mode}): scene_surface ns = the Cycles Normal Map node (f64), nsm = the unmapped normal`, async () => {
      const scene = await nmScene(mode);
      const { device, features, wgslLanguageFeatures } = await getTestGpu();
      const O = computeRenderOrigin(scene.bounds, scene.quant);
      const gpu = await SceneGpu.create(device, scene, O, { textureMode: 'validation', watertight: true, buildBvh: async (p, i) => buildBvh(p, i, { mt: false, woop: true }), features, wgslLanguageFeatures });
      expect(gpu.defines(0).NORMAL_MAP).toBe(true);
      const g = scene.geometry, nT = g.indices.length / 3;
      const rec = recentrePositions(g.positions, O);
      const NP = 50_000, R = rng(91);
      const probes = new Float32Array(NP * 4), pu = new Uint32Array(probes.buffer);
      for (let i = 0; i < NP; i++) {
        let u = R(), v = R();
        if (u + v > 1) { u = 1 - u; v = 1 - v; }
        pu[4 * i] = Math.floor(R() * nT); probes[4 * i + 1] = u; probes[4 * i + 2] = v; probes[4 * i + 3] = R() < 0.25 ? 1 : 0;
      }
      const shader = composeWgsl('tests/nm.wgsl', { sources: { ...shaderSources, 'tests/nm.wgsl': SURF }, defines: gpu.defines(0), features, wgslLanguageFeatures });
      const module = await createCheckedShaderModule(device, shader, 'nm');
      const l0 = device.createBindGroupLayout({ entries: gpu.layoutEntries(GPUShaderStage.COMPUTE) });
      const C = GPUShaderStage.COMPUTE;
      const l1 = device.createBindGroupLayout({ entries: [
        { binding: 0, visibility: C, buffer: { type: 'read-only-storage' } }, { binding: 1, visibility: C, buffer: { type: 'storage' } }, { binding: 2, visibility: C, buffer: { type: 'uniform' } }] });
      const pipe = await device.createComputePipelineAsync({ layout: device.createPipelineLayout({ bindGroupLayouts: [l0, l1] }), compute: { module, entryPoint: 'surfaces' } });
      const mk = (data: ArrayBufferView | number, usage: number) => {
        const size = Math.max(16, Math.ceil((typeof data === 'number' ? data : data.byteLength) / 16) * 16);
        const b = device.createBuffer({ size, usage: usage | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST, mappedAtCreation: typeof data !== 'number' });
        if (typeof data !== 'number') { new Uint8Array(b.getMappedRange()).set(new Uint8Array(data.buffer, data.byteOffset, data.byteLength)); b.unmap(); }
        return b;
      };
      const bP = mk(probes, GPUBufferUsage.STORAGE), bO = mk(NP * 64, GPUBufferUsage.STORAGE), bN = mk(Uint32Array.from([NP, 0, 0, 0]), GPUBufferUsage.UNIFORM);
      const enc = device.createCommandEncoder();
      const pass = enc.beginComputePass();
      pass.setPipeline(pipe);
      pass.setBindGroup(0, device.createBindGroup({ layout: l0, entries: gpu.bindGroupEntries() }));
      pass.setBindGroup(1, device.createBindGroup({ layout: l1, entries: [bP, bO, bN].map((b, binding) => ({ binding, resource: { buffer: b } })) }));
      pass.dispatchWorkgroups(Math.ceil(NP / 64));
      pass.end();
      device.queue.submit([enc.finish()]);
      const out = new Float32Array(await readBuffer(device, bO, NP * 64));
      const outU = new Uint32Array(out.buffer);
      let nsErr = 0, nsmErr = 0, plainBad = 0, compared = 0, skipped = 0, backfacing = 0, mirrored = 0, fallbacks = 0;
      for (let i = 0; i < NP; i++) {
        const t = pu[4 * i], u = probes[4 * i + 1], v = probes[4 * i + 2], w = 1 - u - v;
        const vs = [0, 1, 2].map((k) => g.indices[3 * t + k]);
        const W3 = [w, u, v];
        const mi = g.triMaterial[t], mat = scene.materials[mi];
        const back = out[16 * i + 7] > 0.5;
        const flat = (outU[16 * i + 11] & TRI_FLAT) !== 0;
        // ng from the recentred positions (winding), the smooth normal (un-normalised), tangent, sign
        const e1 = [0, 1, 2].map((k) => rec[3 * vs[1] + k] - rec[3 * vs[0] + k]), e2 = [0, 1, 2].map((k) => rec[3 * vs[2] + k] - rec[3 * vs[0] + k]);
        const ngW0 = [e1[1] * e2[2] - e1[2] * e2[1], e1[2] * e2[0] - e1[0] * e2[2], e1[0] * e2[1] - e1[1] * e2[0]];
        const ngl = Math.hypot(...ngW0), ngW = ngW0.map((x) => x / ngl);
        const nU = flat ? ngW : [0, 1, 2].map((k) => W3.reduce((s, x, j) => s + x * g.normals[3 * vs[j] + k], 0));
        const nUl = Math.hypot(...nU);
        const nsm = flat ? ngW : nU.map((x) => x / nUl);
        const sgnF = back ? -1 : 1;
        for (let k = 0; k < 3; k++) nsmErr = Math.max(nsmErr, Math.abs(out[16 * i + 4 + k] - sgnF * nsm[k]));
        if (!mat.normalTexture) {
          for (let k = 0; k < 3; k++) if (outU[16 * i + k] !== outU[16 * i + 4 + k]) plainBad++;
          continue;
        }
        const T = [0, 1, 2].map((k) => W3.reduce((s, x, j) => s + x * g.tangents[4 * vs[j] + k], 0));
        const sg = W3.reduce((s, x, j) => s + x * g.tangents[4 * vs[j] + 3], 0);
        if (sg < 0) mirrored++;
        // nearest texel at the GPU's uv (skip probes within 1e-3 texel of a texel edge)
        const tex = scene.textures[mat.normalTexture.texture];
        const fx = out[16 * i + 12] * tex.width, fy = out[16 * i + 13] * tex.height;
        const dx = fx - Math.floor(fx), dy = fy - Math.floor(fy);
        if (Math.min(dx, 1 - dx, dy, 1 - dy) < 1e-3) { skipped++; continue; }
        const xi = ((Math.floor(fx) % tex.width) + tex.width) % tex.width, yi = ((Math.floor(fy) % tex.height) + tex.height) % tex.height;
        const rgb = [0, 1, 2].map((k) => tex.pixels[4 * (yi * tex.width + xi) + k] / 255);
        const N = normalMapCycles(rgb, mat.normalTexture.scale, T, sg, nU);
        const want = N ? N.map((x) => sgnF * x) : nsm.map((x) => sgnF * x);
        if (!N) fallbacks++;
        if (back) backfacing++;
        compared++;
        for (let k = 0; k < 3; k++) nsErr = Math.max(nsErr, Math.abs(out[16 * i + k] - want[k]));
      }
      const r = { mode, probes: NP, compared, skipped, backfacing, mirrored, fallbacks, nsErr, nsmErr, plainBad, tangentWords: gpu.vertexArena.bytes.tangent };
      report[`U-NM-1-${mode}`] = r;
      console.log('U-NM-1', lane(), JSON.stringify(r));
      expect(compared).toBeGreaterThan(20_000);
      expect(backfacing).toBeGreaterThan(3000);
      expect(mirrored).toBeGreaterThan(50);              // 2 of ~1300 triangles (the mirrored quad)
      expect(nsErr).toBeLessThan(2e-5);
      expect(nsmErr).toBeLessThan(2e-5);
      expect(plainBad).toBe(0);
      for (const b of [bP, bO, bN]) b.destroy();
      gpu.destroy();
    }, 300_000);
  }

  it('U-NM-2: bump-shadowing term: NEE / diffuse × 1[bump ok], glossy / glass unchanged, diffuse samples rejected', async () => {
    const { device, features, wgslLanguageFeatures } = await getTestGpu();
    const lut = createLutBuffer(device);
    const KER = /* wgsl */ `
#include "material/bsdf.wgsl"
#include "common/rng.wgsl"
@group(0) @binding(0) var<storage, read_write> cnt: array<atomic<u32>, 16>;
@group(0) @binding(1) var<storage, read> records: array<u32>;
fn unit(h: u32) -> f32 { return u32_to_unit(h); }
fn sph(a: f32, b: f32) -> vec3f { let z = 2.0 * a - 1.0; let r = sqrt(max(0.0, 1.0 - z * z)); return vec3f(r * cos(6.2831853 * b), r * sin(6.2831853 * b), z); }
fn eqv(a: vec3f, b: vec3f) -> bool { return all(bitcast<vec3u>(a) == bitcast<vec3u>(b)); }
@compute @workgroup_size(64) fn k(@builtin(global_invocation_id) gid: vec3u) {
  let h = pcg3d(vec3u(gid.x, 17u, 3u));
  let h2 = pcg3d(vec3u(gid.x, 91u, 5u));
  let ng = vec3f(0.0, 0.0, 1.0);
  // smooth normal within 25° of ng, mapped normal within 55° of the smooth one
  let nsm = normalize(ng + 0.47 * sph(unit(h.x), unit(h.y)) * vec3f(1.0, 1.0, 0.0));
  let ns = normalize(nsm + 1.4 * sph(unit(h.z), unit(h2.x)));
  var V = sph(unit(h2.y), unit(h2.z));
  if (dot(V, ng) < 0.0) { V = -V; }
  let L = sph(unit(pcg3d(vec3u(gid.x, 7u, 1u)).x), unit(pcg3d(vec3u(gid.x, 7u, 2u)).y));
  var m: MatEval;
  let model = gid.x % 3u;
  m.model = select(BSDF_MODEL_V2, BSDF_MODEL_V1, model == 1u);
  if (model == 2u) { m.model = BSDF_MODEL_GLASS; }
  m.base_color = vec3f(0.8, 0.6, 0.4);
  m.metallic = select(0.0, 0.5, model == 1u);
  m.roughness = 0.45;
  m.ior = 1.45;
  m.specular_level = 0.5;
  m.specular_tint = vec3f(1.0);
  m.transmission = select(0.0, 0.7, model == 2u);
  m.ns = ns; m.ng = ng; m.nsm = ns;
  m.flags = bsdf_flags(m, V);
  var mb = m;
  mb.nsm = nsm;
  let ok = !(dot(nsm, L) * dot(nsm, ns) * dot(ns, L) < 0.0);
  if (!ok) { atomicAdd(&cnt[0], 1u); }
  let e0 = bsdf_eval(m, V, L);
  let e1 = bsdf_eval(mb, V, L);
  let want = select(vec3f(0.0), e0.f_cos, ok);
  if (!eqv(e1.f_cos, want) || bitcast<u32>(e1.pdf_marginal) != bitcast<u32>(e0.pdf_marginal)) { atomicAdd(&cnt[1], 1u); }
  for (var lobe = 0u; lobe <= 4u; lobe++) {
    let q0 = bsdf_query(m, V, L, lobe);
    let q1 = bsdf_query(mb, V, L, lobe);
    let bumped = lobe == LOBE_D || lobe == LOBE_NEE;
    let wl = select(q0.f_lobe, select(vec3f(0.0), q0.f_lobe, ok), bumped);
    if (!eqv(q1.f_lobe, wl) || bitcast<u32>(q1.p_joint) != bitcast<u32>(q0.p_joint) || q1.supp != q0.supp) { atomicAdd(&cnt[2], 1u); }
    if (!eqv(q1.f_all, e1.f_cos)) { atomicAdd(&cnt[3], 1u); }
    if (lobe == LOBE_S && any(q1.f_lobe != q0.f_lobe)) { atomicAdd(&cnt[4], 1u); }
  }
  let u4 = vec4f(unit(h.y ^ 0x9e37u), unit(h.z ^ 0x51u), unit(h2.x ^ 0x77u), unit(h2.y ^ 0x13u));
  let s0 = bsdf_sample(m, V, u4);
  let s1 = bsdf_sample(mb, V, u4);
  if (s0.valid) {
    let okS = !(dot(nsm, s0.L) * dot(nsm, ns) * dot(ns, s0.L) < 0.0);
    if (s0.lobe == LOBE_D && !okS) {
      atomicAdd(&cnt[5], 1u);
      if (s1.valid) { atomicAdd(&cnt[6], 1u); }
    } else {
      if (!s1.valid || !eqv(s1.L, s0.L) || !eqv(s1.weight, s0.weight) || s1.lobe != s0.lobe) { atomicAdd(&cnt[7], 1u); }
      if (!okS && s0.lobe != LOBE_D) { atomicAdd(&cnt[8], 1u); }
    }
  } else if (s1.valid) { atomicAdd(&cnt[9], 1u); }
}`;
    const defines = { ...lutDefines({ base: 0, recordsKind: 'u32' }), NORMAL_MAP: true };
    const shader = composeWgsl('tests/bump.wgsl', { sources: { ...shaderSources, 'tests/bump.wgsl': KER }, defines, features, wgslLanguageFeatures });
    const module = await createCheckedShaderModule(device, shader, 'bump');
    const pipe = await device.createComputePipelineAsync({ layout: 'auto', compute: { module, entryPoint: 'k' } });
    const cnt = device.createBuffer({ size: 64, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
    const enc = device.createCommandEncoder();
    const pass = enc.beginComputePass();
    pass.setPipeline(pipe);
    pass.setBindGroup(0, device.createBindGroup({ layout: pipe.getBindGroupLayout(0), entries: [{ binding: 0, resource: { buffer: cnt } }, { binding: 1, resource: { buffer: lut } }] }));
    const N = 64 * 16384;
    pass.dispatchWorkgroups(N / 64);
    pass.end();
    device.queue.submit([enc.finish()]);
    const c = Array.from(new Uint32Array(await readBuffer(device, cnt, 64)));
    const r = { pairs: N, bumpRejected: c[0], evalMismatch: c[1], queryMismatch: c[2], fAllMismatch: c[3], glossyChanged: c[4], diffuseInBad: c[5], diffuseAccepted: c[6],
      sampleMismatch: c[7], glossyInBad: c[8], spurious: c[9] };
    report['U-NM-2'] = r;
    console.log('U-NM-2', lane(), JSON.stringify(r));
    expect(r.bumpRejected).toBeGreaterThan(N / 50);
    expect(r.diffuseInBad).toBeGreaterThan(1000);
    expect(r.glossyInBad).toBeGreaterThan(100);
    for (const k of ['evalMismatch', 'queryMismatch', 'fAllMismatch', 'glossyChanged', 'diffuseAccepted', 'sampleMismatch', 'spurious'] as const) expect(r[k], k).toBe(0);
    cnt.destroy(); lut.destroy();
  }, 120_000);

  for (const strength of [1, 0.5]) {
    it(`U-NM-3: tilted-normal Lambert plane in a constant env = ρ(1 + cos θ')/2 (θ = 40°, strength ${strength})`, async () => {
      const th = 40 * Math.PI / 180;
      const tex = tiltMap('tilt40', th, 0.3);
      // Cycles strength: c.xy·s, c.z = mix(1, c.z, s)
      const cz = 1 + (Math.cos(th) - 1) * Math.min(1, strength), cxy = strength * Math.sin(th);
      const thEff = Math.atan2(cxy, cz);
      const rho = 0.5;
      const mat = principled('lambert-nm', { baseColorFactor: [rho, rho, rho, 1], specularFactor: 0, roughnessFactor: 1, normalTexture: { texture: 0, texCoord: 0, scale: strength } });
      const S = 50;
      const q = quadScene([{ p: [[-S, 0, S], [S, 0, S], [S, 0, -S], [-S, 0, -S]], mat: 0 }], [mat]);
      // uv over the quad (any non-degenerate map: the tilt is constant)
      q.geometry.uv0.set([0, 1, 1, 1, 1, 0, 0, 0]);
      const W = 64, H = 64;
      const env = { name: 'const', width: 32, height: 16, texels: new Float32Array(32 * 16 * 4).fill(1), strength: 1, tint: [1, 1, 1] as [number, number, number], rotationZ: 0, visibleToCamera: true };
      const scene0: SceneData = { ...q, textures: [tex], env };
      const scene = sceneTangents(quantizeScene({ ...scene0, quant: undefined }, { mode: 'quantized' }).scene, await loadMikkTSpace(), { quantized: true, flatFaceNormals: true }).scene;
      const rig = await ptRig(scene, W, H, { camToWorld: lookDown([0, 2, 0]), yfov: 0.4 }, { maxBounces: 1 });
      const st = await batchStats(rig, 16, 64, (m) => {
        let s = 0;
        for (let i = 0; i < W * H; i++) s += m[3 * i];
        return [s / (W * H)];
      });
      rig.destroy();
      const want = rho * (1 + Math.cos(thEff)) / 2;
      const z = (st.mean[0] - want) / st.se[0];
      const r = { strength, thetaEffDeg: thEff * 180 / Math.PI, mean: st.mean[0], se: st.se[0], want, rel: st.mean[0] / want - 1, z, counters: st.counters };
      report[`U-NM-3-s${strength}`] = r;
      console.log('U-NM-3', lane(), JSON.stringify(r));
      expect(Math.abs(z)).toBeLessThan(4.5);
      expect(Math.abs(r.rel)).toBeLessThan(2e-3);
      expect(st.counters[0]).toBe(0);
    }, 300_000);
  }
});
