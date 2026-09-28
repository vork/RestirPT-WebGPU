// Texture path probes (plan §1.6; gap-bsdf §9 U-T1): sRGB decode AFTER filtering, wrap modes, nearest,
// KHR_texture_transform, interactive linear-space mips, budget downscale. Chrome lane additionally runs the
// browser glTF path (Worker + createImageBitmap decode + MikkTSpace wasm).
import { afterAll, describe, expect, it } from 'vitest';
import { getTestGpu, lane, releaseTestGpu } from './device-factory.ts';
import { composeWgsl, createCheckedShaderModule } from '../../src/core/gpu/wgsl-composer.ts';
import { readBuffer } from '../../src/core/gpu/readback.ts';
import { shaderSources } from '../../src/core/shaders/index.ts';
import { createGpuTextures, packTexSlot, type GpuTextures, type TexSlot } from '../../src/core/render/textures-gpu.ts';
import type { MaterialData, TextureData, WrapMode } from '../../src/core/scene/types.ts';
// Static imports so Vite's dependency scan sees @gltf-transform before the browser test starts (avoids a reload).
import { Document, WebIO } from '@gltf-transform/core';
import { loadScene } from '../../src/core/scene/load-scene.ts';

const PROBE = `
#include "material/textures.wgsl"
struct Query { slot: TexSlot, uv: vec2f, lod: f32, pad: f32 }
@group(1) @binding(0) var<storage, read> queries: array<Query>;
@group(1) @binding(1) var<storage, read_write> results: array<vec4f>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  if (gid.x >= arrayLength(&queries)) { return; }
  let q = queries[gid.x];
  results[gid.x] = tex_sample_lod(q.slot, q.uv, q.lod);
}`;

interface Query { slot: TexSlot | null; uv: [number, number]; lod?: number }

async function probe(set: GpuTextures, queries: Query[]): Promise<Float32Array> {
  const { device, features, wgslLanguageFeatures } = await getTestGpu();
  const shader = composeWgsl('tests/tex_probe.wgsl', {
    sources: { ...shaderSources, 'tests/tex_probe.wgsl': PROBE }, defines: set.defines(0, 0), features, wgslLanguageFeatures,
  });
  const module = await createCheckedShaderModule(device, shader, 'tex-probe');
  const bgl0 = device.createBindGroupLayout({ entries: set.layoutEntries(0) });
  const bgl1 = device.createBindGroupLayout({ entries: [
    { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
    { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
  ] });
  const pipeline = await device.createComputePipelineAsync({ layout: device.createPipelineLayout({ bindGroupLayouts: [bgl0, bgl1] }), compute: { module, entryPoint: 'main' } });
  const qbytes = new ArrayBuffer(queries.length * 48);
  const dv = new DataView(qbytes);
  queries.forEach((q, i) => {
    packTexSlot(q.slot, dv, i * 48);
    dv.setFloat32(i * 48 + 32, q.uv[0], true); dv.setFloat32(i * 48 + 36, q.uv[1], true); dv.setFloat32(i * 48 + 40, q.lod ?? 0, true);
  });
  const qbuf = device.createBuffer({ size: qbytes.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
  device.queue.writeBuffer(qbuf, 0, qbytes);
  const rbuf = device.createBuffer({ size: queries.length * 16, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
  const enc = device.createCommandEncoder();
  const pass = enc.beginComputePass();
  pass.setPipeline(pipeline);
  pass.setBindGroup(0, device.createBindGroup({ layout: bgl0, entries: set.bindGroupEntries(0) }));
  pass.setBindGroup(1, device.createBindGroup({ layout: bgl1, entries: [{ binding: 0, resource: { buffer: qbuf } }, { binding: 1, resource: { buffer: rbuf } }] }));
  pass.dispatchWorkgroups(Math.ceil(queries.length / 64));
  pass.end();
  device.queue.submit([enc.finish()]);
  const out = new Float32Array(await readBuffer(device, rbuf, queries.length * 16));
  qbuf.destroy(); rbuf.destroy();
  return out;
}

const srgbToLinear = (c: number) => (c < 0.04045 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4);
const linearToSrgb = (c: number) => (c < 0.0031308 ? c * 12.92 : 1.055 * c ** (1 / 2.4) - 0.055);
const tex = (name: string, width: number, height: number, pixels: Uint8Array, wrap: WrapMode = 'clamp-to-edge', filter: 'linear' | 'nearest' = 'linear'): TextureData =>
  ({ name, width, height, pixels, wrapS: wrap, wrapT: wrap, filter });
const mat = (over: Partial<MaterialData>): MaterialData => ({
  name: 'm', baseColorFactor: [1, 1, 1, 1], metallicFactor: 0, roughnessFactor: 1, emissiveFactor: [0, 0, 0], emissiveStrength: 1, ior: 1.5,
  specularFactor: 1, specularColorFactor: [1, 1, 1], transmissionFactor: 0, alphaMode: 'OPAQUE', alphaCutoff: 0.5, doubleSided: true, model: 'principled', ...over,
});
const ref = (texture: number, transform?: [number, number, number, number, number, number]) => ({ texture, texCoord: 0, ...(transform ? { transform } : {}) });

describe(`textures (${lane()})`, () => {
  afterAll(releaseTestGpu);

  it('validation path: sRGB decode after bilinear filtering (U-T1), wrap modes, nearest, transform, invalid slot', async () => {
    const { device } = await getTestGpu();
    const px = new Uint8Array([0, 0, 0, 255, 255, 255, 255, 255]); // 2x1: black | white
    const textures = [
      tex('clamp', 2, 1, px, 'clamp-to-edge'),
      tex('repeat', 2, 1, px, 'repeat'),
      tex('mirror', 2, 1, px, 'mirror-repeat'),
      tex('nearest', 2, 1, px, 'clamp-to-edge', 'nearest'),
      tex('other', 4, 2, new Uint8Array(32).fill(255)),
    ];
    const set = await createGpuTextures(device, { textures, materials: [] }, { mode: 'validation' });
    expect(set.arrays).toHaveLength(2);            // one array per distinct (w, h)
    expect(set.arrays[0].depthOrArrayLayers).toBe(1); // the four 2x1 TextureData share one pixel buffer → one layer
    expect(set.arrays[0].mipLevelCount).toBe(1);
    expect(set.samplers).toHaveLength(4);
    const S = (i: number, srgb = true, xf?: [number, number, number, number, number, number]) => set.slot(ref(i, xf), srgb);
    const v = 0.5;
    const cases: [Query, number, number][] = [
      [{ slot: S(0), uv: [0.5, v] }, srgbToLinear(0.5), 1e-3],          // 0.21404, not 0.5
      [{ slot: S(0, false), uv: [0.5, v] }, 0.5, 2e-3],                  // linear data: plain filtered byte
      [{ slot: S(0), uv: [0.0, v] }, 0, 1e-6],                           // clamp
      [{ slot: S(0), uv: [1.0, v] }, 1, 1e-6],
      [{ slot: S(0), uv: [1.25, v] }, 1, 1e-6],
      [{ slot: S(1), uv: [0.0, v] }, srgbToLinear(0.5), 1e-3],          // repeat: blends texel 1 and texel 0
      [{ slot: S(1), uv: [1.25, v] }, 0, 1e-6],
      [{ slot: S(2), uv: [0.0, v] }, 0, 1e-6],                           // mirror-repeat
      [{ slot: S(2), uv: [1.25, v] }, 1, 1e-6],
      [{ slot: S(3), uv: [0.49, v] }, 0, 1e-6],                          // nearest
      [{ slot: S(3), uv: [0.51, v] }, 1, 1e-6],
      [{ slot: S(0, true, [0, 0, 0.5, 0, 0, 0.5]), uv: [0.9, 0.1] }, srgbToLinear(0.5), 1e-3], // transform → (0.5, 0.5)
      [{ slot: S(0, true, [2, 0, 0, 0, 1, 0]), uv: [0.25, v] }, srgbToLinear(0.5), 1e-3],     // u' = 2u
      [{ slot: null, uv: [0.5, v] }, 1, 0],                              // absent texture → 1
      [{ slot: S(4), uv: [0.3, 0.7] }, 1, 1e-6],                         // second array
    ];
    const out = await probe(set, cases.map((c) => c[0]));
    const got = cases.map((_, i) => out[i * 4]);
    console.log('TEX_PROBE', lane(), JSON.stringify(got.map((x) => +x.toFixed(6))));
    cases.forEach(([, want, tol], i) => expect(Math.abs(got[i] - want), `case ${i}: got ${got[i]} want ${want}`).toBeLessThanOrEqual(tol));
    for (let i = 0; i < cases.length; i++) expect(out[i * 4 + 3]).toBeCloseTo(1, 6); // alpha untouched
    expect(Math.abs(got[0] - 0.5)).toBeGreaterThan(0.25); // the -srgb hardware-decode result would be 0.5
    set.destroy();
  });

  it('binding extremes: empty set (no bindings) and 16 arrays × 8 samplers compile and sample', async () => {
    const { device } = await getTestGpu();
    const empty = await createGpuTextures(device, { textures: [], materials: [] }, { mode: 'validation' });
    expect(empty.layoutEntries(0)).toHaveLength(0);
    expect(Array.from(await probe(empty, [{ slot: null, uv: [0.5, 0.5] }]))).toEqual([1, 1, 1, 1]);
    const combos: [WrapMode, WrapMode, 'linear' | 'nearest'][] = [
      ['repeat', 'repeat', 'linear'], ['clamp-to-edge', 'clamp-to-edge', 'linear'], ['mirror-repeat', 'mirror-repeat', 'linear'],
      ['repeat', 'repeat', 'nearest'], ['clamp-to-edge', 'clamp-to-edge', 'nearest'], ['mirror-repeat', 'mirror-repeat', 'nearest'],
      ['repeat', 'clamp-to-edge', 'linear'], ['clamp-to-edge', 'repeat', 'nearest']];
    const textures = Array.from({ length: 16 }, (_, i) => {
      const w = i + 1, px = new Uint8Array(w * 4);
      for (let k = 0; k < w; k++) px.set([i * 16, 255 - i * 16, 0, 255], k * 4);
      const [ws, wt, f] = combos[i % 8];
      return { ...tex(`t${i}`, w, 1, px, ws, f), wrapT: wt };
    });
    const full = await createGpuTextures(device, { textures, materials: [] }, { mode: 'validation' });
    expect(full.arrays).toHaveLength(16);
    expect(full.samplers).toHaveLength(8);
    const extra = [...textures, { ...tex('t16', 1, 1, new Uint8Array(4)), wrapT: 'mirror-repeat' as WrapMode }];
    await expect(createGpuTextures(device, { textures: extra, materials: [] }, { mode: 'validation' })).rejects.toThrow(/binding budget/);
    const out = await probe(full, textures.map((_, i) => ({ slot: full.slot(ref(i), false), uv: [0.5, 0.5] as [number, number] })));
    textures.forEach((_, i) => { expect(out[i * 4]).toBeCloseTo((i * 16) / 255, 5); expect(out[i * 4 + 1]).toBeCloseTo((255 - i * 16) / 255, 5); });
    full.destroy();
  });

  it('validation path rejects more than 16 distinct sizes', async () => {
    const { device } = await getTestGpu();
    const textures = Array.from({ length: 17 }, (_, i) => tex(`t${i}`, i + 1, 1, new Uint8Array((i + 1) * 4)));
    await expect(createGpuTextures(device, { textures, materials: [] }, { mode: 'validation' })).rejects.toThrow(/distinct sizes/);
  });

  it('interactive path: buckets, linear-space mips of sRGB data, resampling, budget downscale', async () => {
    const { device, features, wgslLanguageFeatures } = await getTestGpu();
    const checker = () => {
      const p = new Uint8Array(256 * 256 * 4);
      for (let y = 0; y < 256; y++) for (let x = 0; x < 256; x++) { const c = (x + y) & 1 ? 255 : 0; p.set([c, c, c, 255], (y * 256 + x) * 4); }
      return p;
    };
    const solid = new Uint8Array(300 * 200 * 4);
    for (let i = 0; i < solid.length; i += 4) solid.set([200, 100, 50, 255], i);
    const textures = [tex('srgbChecker', 256, 256, checker(), 'repeat'), tex('linChecker', 256, 256, checker(), 'repeat'), tex('solid', 300, 200, solid, 'repeat')];
    const materials = [mat({ baseColorTexture: ref(0), metallicRoughnessTexture: ref(1) }), mat({ emissiveTexture: ref(2) })];
    const set = await createGpuTextures(device, { textures, materials }, { mode: 'interactive', features, wgslLanguageFeatures });
    expect(set.arrays.map((a) => [a.width, a.depthOrArrayLayers, a.mipLevelCount])).toEqual([[256, 2, 9], [512, 1, 10]]);
    const q: Query[] = [
      { slot: set.slot(ref(0), true), uv: [0.5, 0.5], lod: 8 },   // 1x1 mip of an sRGB checker
      { slot: set.slot(ref(1), false), uv: [0.5, 0.5], lod: 8 },  // 1x1 mip of a linear checker
      { slot: set.slot(ref(0), true), uv: [0.5, 0.5], lod: 1 },   // 128² mip: every texel is a 2x2 box of 0/255
      { slot: set.slot(ref(2), true), uv: [0.3, 0.6], lod: 0 },   // resampled 300x200 → 512²
      { slot: set.slot(ref(2), true), uv: [0.3, 0.6], lod: 9 },
    ];
    const out = await probe(set, q);
    console.log('TEX_MIPS', lane(), JSON.stringify(Array.from(out).map((x) => +x.toFixed(5))));
    const q8 = 1 / 255;
    // Linear-space average of black/white = 0.5 (re-encoded to 187/188); the naive byte average would decode to 0.214.
    expect(Math.abs(out[0] - 0.5)).toBeLessThan(srgbToLinear(linearToSrgb(0.5) + q8) - 0.5 + 1e-4);
    expect(Math.abs(out[4] - 0.5)).toBeLessThanOrEqual(q8 + 1e-4);
    expect(Math.abs(out[8] - 0.5)).toBeLessThan(0.01);
    for (const o of [12, 16]) {
      expect(out[o]).toBeCloseTo(srgbToLinear(200 / 255), 3);
      expect(out[o + 1]).toBeCloseTo(srgbToLinear(100 / 255), 3);
      expect(out[o + 2]).toBeCloseTo(srgbToLinear(50 / 255), 3);
    }
    set.destroy();

    // Budget: 512² with mips ≈ 1.33 MiB does not fit in 400 KiB → one halving to 256 (≈ 341 KiB).
    const small = await createGpuTextures(device, { textures: [tex('solid', 300, 200, solid)], materials: [mat({ baseColorTexture: ref(0) })] },
      { mode: 'interactive', budgetBytes: 400 * 1024, features, wgslLanguageFeatures });
    expect(small.arrays[0].width).toBe(256);
    expect(small.warnings.some((w) => /budget/.test(w))).toBe(true);
    const o2 = await probe(small, [{ slot: small.slot(ref(0), true), uv: [0.5, 0.5], lod: 0 }]);
    expect(o2[0]).toBeCloseTo(srgbToLinear(200 / 255), 3);
    small.destroy();
  });

  it.runIf(lane() === 'chrome')('browser glTF path: Worker + createImageBitmap decode + MikkTSpace', async () => {
    const cornell = await loadScene('/validation/assets/cornell/cornell.glb');
    expect(cornell.scene.geometry.indices.length / 3).toBe(34);
    expect(cornell.scene.cameras).toHaveLength(1);

    // Textured, normal-mapped quad built in the page; the PNG comes from OffscreenCanvas (α = 255 everywhere).
    const w = 4, h = 2, px = new Uint8ClampedArray(w * h * 4);
    for (let i = 0; i < w * h; i++) px.set([i * 30, 255 - i * 20, (i * 77) & 255, 255], i * 4);
    const cv = new OffscreenCanvas(w, h);
    cv.getContext('2d')!.putImageData(new ImageData(px, w, h), 0, 0);
    const png = new Uint8Array(await (await cv.convertToBlob({ type: 'image/png' })).arrayBuffer());
    const doc = new Document();
    const buf = doc.createBuffer();
    const acc = (type: 'VEC2' | 'VEC3', a: number[]) => doc.createAccessor().setType(type).setArray(new Float32Array(a)).setBuffer(buf);
    const t = doc.createTexture('img').setImage(png).setMimeType('image/png');
    const m = doc.createMaterial('m').setBaseColorTexture(t).setNormalTexture(t);
    const prim = doc.createPrimitive().setMaterial(m).setAttribute('POSITION', acc('VEC3', [0, 0, 0, 1, 0, 0, 0, 1, 0]))
      .setAttribute('NORMAL', acc('VEC3', [0, 0, 1, 0, 0, 1, 0, 0, 1])).setAttribute('TEXCOORD_0', acc('VEC2', [0, 0, 1, 0, 0, 1]));
    const scene = doc.createScene();
    scene.addChild(doc.createNode().setMesh(doc.createMesh().addPrimitive(prim)));
    doc.getRoot().setDefaultScene(scene);
    const glb = await new WebIO().writeBinary(doc);
    const r = await loadScene(glb);
    const td = r.scene.textures[0];
    expect([td.width, td.height]).toEqual([w, h]);
    expect(Array.from(td.pixels)).toEqual(Array.from(px));
    const tg = r.scene.geometry.tangents;
    expect([tg[0], tg[1], tg[2], tg[3]]).toEqual([1, 0, 0, -1].map((x) => expect.closeTo(x, 5)));
    // And the decoded pixels go through the validation texture path.
    const { device } = await getTestGpu();
    const set = await createGpuTextures(device, r.scene, { mode: 'validation' });
    const o = await probe(set, [{ slot: set.slot(r.scene.materials[0].baseColorTexture, true), uv: [0.125, 0.25] }]);
    expect(o[0]).toBeCloseTo(srgbToLinear(px[0] / 255), 4);
    expect(o[1]).toBeCloseTo(srgbToLinear(px[1] / 255), 4);
    set.destroy();
  });
  it.runIf(lane() === 'chrome')('browser Sponza: Worker load with image decode, validation + interactive texture sets', async () => {
    const url = '/validation/assets/downloaded/sponza/Sponza.gltf';
    if (!(await fetch(url, { method: 'HEAD' })).ok) { console.log('SPONZA_BROWSER skipped (asset missing)'); return; }
    const t0 = performance.now();
    const r = await loadScene(url);
    const tLoad = performance.now() - t0;
    const s = r.scene;
    expect(s.textures.every((t) => t.pixels.length === t.width * t.height * 4)).toBe(true);
    let masked = 0;
    for (const f of s.geometry.triFlags) if (f & 1) masked++;
    expect(masked).toBeGreaterThan(0);
    const { device, features, wgslLanguageFeatures } = await getTestGpu();
    const t1 = performance.now();
    const val = await createGpuTextures(device, s, { mode: 'validation' });
    await device.queue.onSubmittedWorkDone();
    const t2 = performance.now();
    const int = await createGpuTextures(device, s, { mode: 'interactive', features, wgslLanguageFeatures });
    const t3 = performance.now();
    console.log('SPONZA_BROWSER', JSON.stringify({ loadMs: Math.round(tLoad), stats: r.stats.ms, tris: s.geometry.indices.length / 3, textures: s.textures.length,
      maskedTris: masked, validation: { ms: Math.round(t2 - t1), arrays: val.arrays.map((a) => `${a.width}x${a.height}x${a.depthOrArrayLayers}`), MiB: +(val.bytes / 2 ** 20).toFixed(1) },
      interactive: { ms: Math.round(t3 - t2), arrays: int.arrays.map((a) => `${a.width}x${a.depthOrArrayLayers}`), MiB: +(int.bytes / 2 ** 20).toFixed(1) }, warnings: s.warnings }));
    expect(val.arrays.length).toBeLessThanOrEqual(16);
    val.destroy();
    int.destroy();
  }, 120_000);
  it.runIf(lane() === 'chrome')('browser glTF path: meshopt + draco decoders in the Worker', async () => {
    const { ALL_EXTENSIONS } = await import('@gltf-transform/extensions');
    const { meshopt, draco } = await import('@gltf-transform/functions');
    const { MeshoptEncoder } = await import('meshoptimizer');
    await MeshoptEncoder.ready;
    // @ts-ignore draco3dgltf ships no type declarations
    const draco3d = (await import('draco3dgltf')).default ?? (await import('draco3dgltf'));
    const encWasm = (await import('draco3dgltf/draco_encoder.wasm?url')).default;
    const dracoEncoder = await draco3d.createEncoderModule({ wasmBinary: await (await fetch(encWasm)).arrayBuffer() });
    const build = () => {
      const doc = new Document();
      const buf = doc.createBuffer();
      const p: number[] = [], ix: number[] = [];
      for (let y = 0; y <= 8; y++) for (let x = 0; x <= 8; x++) p.push(x / 8, y / 8, 0);
      for (let y = 0; y < 8; y++) for (let x = 0; x < 8; x++) { const a = y * 9 + x; ix.push(a, a + 1, a + 10, a, a + 10, a + 9); }
      const prim = doc.createPrimitive().setAttribute('POSITION', doc.createAccessor().setType('VEC3').setArray(new Float32Array(p)).setBuffer(buf))
        .setAttribute('NORMAL', doc.createAccessor().setType('VEC3').setArray(new Float32Array(p.map((_, i) => (i % 3 === 2 ? 1 : 0)))).setBuffer(buf))
        .setIndices(doc.createAccessor().setType('SCALAR').setArray(new Uint32Array(ix)).setBuffer(buf));
      const sc = doc.createScene();
      sc.addChild(doc.createNode().setMesh(doc.createMesh().addPrimitive(prim)));
      doc.getRoot().setDefaultScene(sc);
      return doc;
    };
    const io = new WebIO().registerExtensions(ALL_EXTENSIONS).registerDependencies({ 'meshopt.encoder': MeshoptEncoder, 'draco3d.encoder': dracoEncoder });
    const d1 = build();
    await d1.transform(meshopt({ encoder: MeshoptEncoder }));
    const d2 = build();
    await d2.transform(draco());
    for (const [d, ext] of [[d1, 'EXT_meshopt_compression'], [d2, 'KHR_draco_mesh_compression']] as const) {
      const r = await loadScene(await io.writeBinary(d));
      expect(r.stats.extensionsUsed).toContain(ext);
      expect(r.scene.geometry.indices.length / 3).toBe(128);
      r.scene.bounds.max.forEach((x, k) => expect(x).toBeCloseTo([1, 1, 0][k], 2));
    }
  });
});
