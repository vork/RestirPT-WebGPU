// Quantized vertex format on the GPU (docs/decisions/data-formats.md §B0, §D P1), both lanes (Chrome = Metal with
// relaxed / fast math): the WGSL decoders (scene-data.wgsl vq_*) vs the TS mirror (gpu/vertex-format.ts).
//   - positions and UVs (dyadic: integer × 2^k) must decode BIT-IDENTICALLY to the CPU values (recentred f32),
//   - oct normals and COLOR_0 (exact products q·f32(1/(2^b − 1)), then normalize) within 4 ulp,
//   - scene_surface at 10^5 random (primId, u, v) vs an f64 evaluation on the dequantized vertices; TRI_FLAT ⇒ ns ≡ ng.
import { afterAll, describe, expect, it } from 'vitest';
import { getTestGpu, lane, releaseTestGpu } from './device-factory.ts';
import { composeWgsl, createCheckedShaderModule } from '../../src/core/gpu/wgsl-composer.ts';
import { readBuffer } from '../../src/core/gpu/readback.ts';
import { shaderSources } from '../../src/core/shaders/index.ts';
import { buildBvh } from '../../src/core/bvh/sah-builder.ts';
import { computeRenderOrigin } from '../../src/core/render/frame-uniforms.ts';
import { SceneGpu, recentrePositions } from '../../src/core/render/scene-gpu.ts';
import { VERTEX_FORMAT_Q, f32bits } from '../../src/core/gpu/vertex-format.ts';
import { quantizeScene, vertexLatticeGroups } from '../../src/core/scene/quantize.ts';
import { TRI_FLAT, type MaterialData, type SceneData, type TextureData } from '../../src/core/scene/types.ts';
import { loadPackage, loadSponzaQuantized } from './quant-fixtures.ts';
import { rng } from '../../tests/bvh/fixtures.ts';

const KERNEL = /* wgsl */ `
#include "scene/scene-data.wgsl"
struct TP { nVerts: u32, nProbes: u32, pad0: u32, pad1: u32 }
@group(1) @binding(0) var<storage, read> vmat: array<u32>;
@group(1) @binding(1) var<storage, read_write> outV: array<vec4u>;
@group(1) @binding(2) var<storage, read> probes: array<vec4f>;
@group(1) @binding(3) var<storage, read_write> outS: array<vec4u>;
@group(1) @binding(4) var<uniform> tp: TP;

@compute @workgroup_size(64) fn verts(@builtin(global_invocation_id) gid: vec3u) {
  let i = gid.x + gid.y * 65535u * 64u;
  if (i >= tp.nVerts) { return; }
  let p = scene_vertex_pos(i);
  let n = scene_vertex_normal(i);
  var uv = vec2f(0.0);
  if (vmat[i] != 0xFFFFFFFFu) { uv = scene_vertex_uv(i, vmat[i]); }
  let c = scene_vertex_color(i);
  outV[3u * i] = vec4u(bitcast<vec3u>(p), bitcast<u32>(uv.x));
  outV[3u * i + 1u] = vec4u(bitcast<vec3u>(n), bitcast<u32>(uv.y));
  outV[3u * i + 2u] = bitcast<vec4u>(c);
}

@compute @workgroup_size(64) fn surfaces(@builtin(global_invocation_id) gid: vec3u) {
  let i = gid.x;
  if (i >= tp.nProbes) { return; }
  let q = probes[i];
  let s = scene_surface(bitcast<u32>(q.x), q.y, q.z, vec3f(0.0, 0.0, 0.0));
  outS[4u * i] = vec4u(bitcast<vec3u>(s.pos), bitcast<u32>(s.uv.x));
  outS[4u * i + 1u] = vec4u(bitcast<vec3u>(s.ng), bitcast<u32>(s.uv.y));
  outS[4u * i + 2u] = vec4u(bitcast<vec3u>(s.ns), s.triFlags);
  outS[4u * i + 3u] = bitcast<vec4u>(s.color);
}`;

/** |a − b| in units of the f32 ulp at max(|a|, |b|) (0 when bit-identical; +0 / −0 count as equal). */
function ulps(a: number, b: number): number {
  if (a === b) return 0;
  const m = Math.max(Math.abs(a), Math.abs(b));
  const e = Math.max(Math.floor(Math.log2(m)), -126);
  return Math.abs(a - b) / 2 ** (e - 23);
}

const MAT: MaterialData = {
  name: 'm', model: 'principled', baseColorFactor: [0.8, 0.8, 0.8, 1], metallicFactor: 0, roughnessFactor: 0.5, emissiveFactor: [0, 0, 0],
  emissiveStrength: 1, ior: 1.5, specularFactor: 1, specularColorFactor: [1, 1, 1], transmissionFactor: 0, alphaMode: 'OPAQUE', alphaCutoff: 0.5, doubleSided: true,
};

/** Smooth random blobs with COLOR_0 (rgba16), a lattice-UV material and a wide-UV material (16k² texture). */
function synthetic(unorm8: boolean): SceneData {
  const R = rng(unorm8 ? 31 : 32);
  const P: number[] = [], N: number[] = [], U: number[] = [], C: number[] = [], I: number[] = [], M: number[] = [];
  for (let t = 0; t < 3000; t++) {
    const c = [R() * 12 - 6 + 1e3, R() * 3, R() * 12 - 6];
    for (let k = 0; k < 3; k++) {
      I.push(P.length / 3);
      P.push(...c.map((x) => x + R() - 0.5));
      const z = 2 * R() - 1, ph = 2 * Math.PI * R(), r = Math.sqrt(1 - z * z);
      N.push(r * Math.cos(ph), r * Math.sin(ph), z);
      U.push(R() * 6 - 3, R() * 4);
      C.push(...[R(), R(), R(), R()].map((x) => (unorm8 ? Math.fround(Math.round(x * 255) / 255) : x)));
    }
    M.push(t % 2);
  }
  const tex = (w: number): TextureData => ({ name: 't', width: w, height: w, pixels: new Uint8Array(w * w * 4), wrapS: 'repeat', wrapT: 'repeat', filter: 'linear' });
  const nV = P.length / 3;
  return quantizeScene({
    name: unorm8 ? 'synthetic-rgba8' : 'synthetic-rgba16',
    geometry: { positions: Float32Array.from(P), normals: Float32Array.from(N), tangents: new Float32Array(nV * 4), uv0: Float32Array.from(U), color0: Float32Array.from(C),
      indices: Uint32Array.from(I), triMaterial: Uint32Array.from(M), triFlags: new Uint32Array(M.length) },
    materials: [{ ...MAT, baseColorTexture: { texture: 0, texCoord: 0 } }, { ...MAT, name: 'wide', baseColorTexture: { texture: 1, texCoord: 0 } }],
    textures: [tex(64), tex(16384)], lights: [], cameras: [], bounds: { min: [0, 0, 0], max: [0, 0, 0] }, warnings: [],
  }).scene;
}

describe(`quantized vertex format: GPU decode = CPU mirror (${lane()})`, () => {
  afterAll(releaseTestGpu);
  const report: Record<string, unknown> = { lane: lane() };
  afterAll(() => console.log('VQ_REPORT', JSON.stringify(report)));

  const cases: { name: string; scene: () => Promise<SceneData | undefined> }[] = [
    { name: 'xi_contact', scene: () => loadPackage('xi_contact_512') },
    { name: 'vii_textured', scene: () => loadPackage('vii_textured_512') },
    { name: 'synthetic-rgba8', scene: async () => synthetic(true) },
    { name: 'synthetic-rgba16', scene: async () => synthetic(false) },
    { name: 'sponza', scene: () => loadSponzaQuantized() },
  ];

  for (const c of cases) {
    it(`${c.name}: positions / UVs bit-exact, normals / colours ≤ 4 ulp, scene_surface vs f64`, async () => {
      const scene = await c.scene();
      if (!scene) { console.warn(`${c.name} not present; skipping`); report[`${c.name}.skipped`] = true; return; }
      const { device, features, wgslLanguageFeatures } = await getTestGpu();
      const O = computeRenderOrigin(scene.bounds, scene.quant);
      const gpu = await SceneGpu.create(device, scene, O, {
        textureMode: 'validation', watertight: true, buildBvh: async (p, i) => buildBvh(p, i, { mt: false, woop: true }), features, wgslLanguageFeatures,
      });
      expect(gpu.vertexArena.format).toBe(VERTEX_FORMAT_Q);
      const g = scene.geometry;
      const nV = g.positions.length / 3, nT = g.indices.length / 3;
      const rec = recentrePositions(g.positions, O);
      const groups = vertexLatticeGroups(g, scene.quant!.uv);
      const vmat = Uint32Array.from(groups, (m) => (m < 0 ? 0xffffffff : m));
      // probes: random (primId, u, v)
      const NP = 100_000, R = rng(77);
      const probes = new Float32Array(NP * 4), pu = new Uint32Array(probes.buffer);
      for (let i = 0; i < NP; i++) {
        let u = R(), v = R();
        if (u + v > 1) { u = 1 - u; v = 1 - v; }
        pu[4 * i] = Math.floor(R() * nT); probes[4 * i + 1] = u; probes[4 * i + 2] = v;
      }
      const shader = composeWgsl('tests/vq.wgsl', { sources: { ...shaderSources, 'tests/vq.wgsl': KERNEL }, defines: gpu.defines(0), features, wgslLanguageFeatures });
      const module = await createCheckedShaderModule(device, shader, 'vq');
      const l0 = device.createBindGroupLayout({ entries: gpu.layoutEntries(GPUShaderStage.COMPUTE) });
      const ro = (binding: number): GPUBindGroupLayoutEntry => ({ binding, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } });
      const rw = (binding: number): GPUBindGroupLayoutEntry => ({ binding, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } });
      const l1 = device.createBindGroupLayout({ entries: [ro(0), rw(1), ro(2), rw(3), { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } }] });
      const layout = device.createPipelineLayout({ bindGroupLayouts: [l0, l1] });
      const [pv, ps] = await Promise.all(['verts', 'surfaces'].map((entryPoint) => device.createComputePipelineAsync({ layout, compute: { module, entryPoint } })));
      const buf = (data: ArrayBufferView | number, usage: number) => {
        const size = Math.max(16, Math.ceil((typeof data === 'number' ? data : data.byteLength) / 16) * 16);
        const b = device.createBuffer({ size, usage: usage | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST, mappedAtCreation: typeof data !== 'number' });
        if (typeof data !== 'number') { new Uint8Array(b.getMappedRange()).set(new Uint8Array(data.buffer, data.byteOffset, data.byteLength)); b.unmap(); }
        return b;
      };
      const bVmat = buf(vmat, GPUBufferUsage.STORAGE), bOutV = buf(nV * 48, GPUBufferUsage.STORAGE), bProbes = buf(probes, GPUBufferUsage.STORAGE);
      const bOutS = buf(NP * 64, GPUBufferUsage.STORAGE), bTp = buf(Uint32Array.from([nV, NP, 0, 0]), GPUBufferUsage.UNIFORM);
      const bg0 = device.createBindGroup({ layout: l0, entries: gpu.bindGroupEntries() });
      const bg1 = device.createBindGroup({ layout: l1, entries: [bVmat, bOutV, bProbes, bOutS, bTp].map((b, binding) => ({ binding, resource: { buffer: b } })) });
      const enc = device.createCommandEncoder();
      const pass = enc.beginComputePass();
      pass.setBindGroup(0, bg0); pass.setBindGroup(1, bg1);
      pass.setPipeline(pv);
      const wg = Math.ceil(nV / 64);
      pass.dispatchWorkgroups(Math.min(wg, 65535), Math.ceil(wg / 65535));
      pass.setPipeline(ps); pass.dispatchWorkgroups(Math.ceil(NP / 64));
      pass.end();
      device.queue.submit([enc.finish()]);
      const outV = new Uint32Array(await readBuffer(device, bOutV, nV * 48));
      const outS = new Uint32Array(await readBuffer(device, bOutS, NP * 64));
      const fV = new Float32Array(outV.buffer), fS = new Float32Array(outS.buffer);
      let posBad = 0, uvBad = 0, nMax = 0, cMax = 0;
      const ex: string[] = [];
      for (let v = 0; v < nV; v++) {
        for (let k = 0; k < 3; k++) if (outV[12 * v + k] !== f32bits(rec[3 * v + k])) { posBad++; if (ex.length < 4) ex.push(`pos ${v}.${k}: gpu ${fV[12 * v + k]} cpu ${rec[3 * v + k]}`); }
        if (groups[v] >= 0) {
          if (outV[12 * v + 3] !== f32bits(g.uv0[2 * v])) { uvBad++; if (ex.length < 8) ex.push(`uv ${v}.u: gpu ${fV[12 * v + 3]} cpu ${g.uv0[2 * v]}`); }
          if (outV[12 * v + 7] !== f32bits(g.uv0[2 * v + 1])) { uvBad++; if (ex.length < 8) ex.push(`uv ${v}.v: gpu ${fV[12 * v + 7]} cpu ${g.uv0[2 * v + 1]}`); }
        }
        for (let k = 0; k < 3; k++) nMax = Math.max(nMax, ulps(fV[12 * v + 4 + k], g.normals[3 * v + k]));
        for (let k = 0; k < 4; k++) cMax = Math.max(cMax, ulps(fV[12 * v + 8 + k], g.color0 ? g.color0[4 * v + k] : 1));
      }
      // scene_surface vs f64 on the dequantized (recentred) vertices
      let posErr = 0, uvErr = 0, ngErr = 0, flatMismatch = 0, flatProbes = 0;
      for (let i = 0; i < NP; i++) {
        const t = pu[4 * i], u = probes[4 * i + 1], v = probes[4 * i + 2], w = 1 - u - v;
        const [a, b, cc] = [0, 1, 2].map((k) => g.indices[3 * t + k]);
        const p = [0, 1, 2].map((k) => w * rec[3 * a + k] + u * rec[3 * b + k] + v * rec[3 * cc + k]);
        const scale = Math.max(1, ...[a, b, cc].flatMap((x) => [0, 1, 2].map((k) => Math.abs(rec[3 * x + k]))));
        for (let k = 0; k < 3; k++) posErr = Math.max(posErr, Math.abs(fS[16 * i + k] - p[k]) / scale);
        const uv = [0, 1].map((k) => w * g.uv0[2 * a + k] + u * g.uv0[2 * b + k] + v * g.uv0[2 * cc + k]);
        const uvScale = Math.max(1, ...[a, b, cc].flatMap((x) => [Math.abs(g.uv0[2 * x]), Math.abs(g.uv0[2 * x + 1])]));
        uvErr = Math.max(uvErr, Math.abs(fS[16 * i + 3] - uv[0]) / uvScale, Math.abs(fS[16 * i + 7] - uv[1]) / uvScale);
        const e1 = [0, 1, 2].map((k) => rec[3 * b + k] - rec[3 * a + k]), e2 = [0, 1, 2].map((k) => rec[3 * cc + k] - rec[3 * a + k]);
        const n = [e1[1] * e2[2] - e1[2] * e2[1], e1[2] * e2[0] - e1[0] * e2[2], e1[0] * e2[1] - e1[1] * e2[0]];
        const nl = Math.hypot(n[0], n[1], n[2]);
        const area = nl / Math.max(Math.hypot(...e1) * Math.hypot(...e2), 1e-30);
        if (area > 1e-3) ngErr = Math.max(ngErr, ...[0, 1, 2].map((k) => Math.abs(fS[16 * i + 4 + k] - n[k] / nl))); // skip slivers
        if (outS[16 * i + 11] & TRI_FLAT) {
          flatProbes++;
          for (let k = 0; k < 3; k++) if (outS[16 * i + 8 + k] !== outS[16 * i + 4 + k]) flatMismatch++;
        }
      }
      const r = { vertices: nV, triangles: nT, wide: gpu.vertexArena.wideCount, colorFormat: scene.quant!.color, posBad, uvBad, normalMaxUlp: nMax, colorMaxUlp: cMax,
        surface: { probes: NP, posRelErr: posErr, uvRelErr: uvErr, ngErr, flatProbes, flatMismatch } };
      report[c.name] = r;
      console.log('VQ', lane(), c.name, JSON.stringify(r));
      expect(posBad, ex.join('\n')).toBe(0);
      expect(uvBad, ex.join('\n')).toBe(0);
      expect(nMax).toBeLessThanOrEqual(4);
      expect(cMax).toBeLessThanOrEqual(4);
      expect(posErr).toBeLessThan(4e-7);
      expect(uvErr).toBeLessThan(4e-7);
      expect(ngErr).toBeLessThan(1e-4);
      expect(flatMismatch).toBe(0);
      for (const b of [bVmat, bOutV, bProbes, bOutS, bTp]) b.destroy();
      gpu.destroy();
    }, 300_000);
  }
});
