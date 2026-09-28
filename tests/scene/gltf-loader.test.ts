import { readFileSync } from 'node:fs';
import { describe, expect, it } from 'vitest';
import { TextureInfo } from '@gltf-transform/core';
import { KHRTextureTransform } from '@gltf-transform/extensions';
import { loadGltf, textureTransformMatrix } from '../../src/core/scene/gltf-loader.ts';
import { TRI_ALPHA_MASK, TRI_EMISSIVE, TRI_FLIPPED, type SceneData } from '../../src/core/scene/types.ts';
import { encodePng, newDoc, toGlb } from './helpers.ts';

const root = new URL('../../', import.meta.url);
const cornellGlb = new Uint8Array(readFileSync(new URL('validation/assets/cornell/cornell.glb', root)));
const meta = JSON.parse(readFileSync(new URL('validation/assets/cornell/cornell.meta.json', root), 'utf8'));

const triNormal = (s: SceneData, t: number) => {
  const p = s.geometry.positions, i = s.geometry.indices;
  const v = [0, 1, 2].map((c) => [p[i[3 * t + c] * 3], p[i[3 * t + c] * 3 + 1], p[i[3 * t + c] * 3 + 2]]);
  const a = v[1].map((x, k) => x - v[0][k]), b = v[2].map((x, k) => x - v[0][k]);
  const n = [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
  const l = Math.hypot(...n);
  return n.map((x) => x / l);
};

describe('glTF loader: Cornell GLB (Node, no image decode)', () => {
  it('matches cornell.meta.json: counts, materials, bounds, camera, no punctual lights', async () => {
    const { scene: s, stats } = await loadGltf({ kind: 'glb', bytes: cornellGlb, name: 'cornell.glb' });
    const g = s.geometry;
    expect(g.indices.length / 3).toBe(34); // 5 walls × 2 + 2 boxes × 12
    expect(stats.flatten.droppedDegenerate).toBe(0);
    expect(g.triMaterial.length).toBe(34);
    expect(Math.max(...g.indices)).toBeLessThan(g.positions.length / 3);
    expect(s.materials.map((m) => m.name)).toEqual(['white', 'red', 'green']);
    for (const m of s.materials) {
      expect(m.baseColorFactor.slice(0, 3)).toEqual(meta.reflectance[m.name].map((x: number) => expect.closeTo(x, 6)));
      expect(m.metallicFactor).toBe(0);
      expect(m.roughnessFactor).toBe(1);
      expect(m.specularFactor).toBe(0);
      expect(m.model).toBe('principled');
    }
    const H = 0.555 / 2;
    s.bounds.min.forEach((x, k) => expect(x).toBeCloseTo([-H, 0, -H][k], 5));
    s.bounds.max.forEach((x, k) => expect(x).toBeCloseTo([H, 0.555, H][k], 5));
    expect(s.cameras).toHaveLength(1);
    const cam = s.cameras[0];
    expect(cam.yfov).toBeCloseTo((meta.camera.vfov_deg * Math.PI) / 180, 5);
    expect([cam.matrix[12], cam.matrix[13], cam.matrix[14]]).toEqual(meta.camera.position.map((x: number) => expect.closeTo(x, 5)));
    expect([-cam.matrix[8], -cam.matrix[9], -cam.matrix[10]]).toEqual(meta.camera.forward.map((x: number) => expect.closeTo(x, 6)));
    expect([cam.matrix[4], cam.matrix[5], cam.matrix[6]]).toEqual(meta.camera.up.map((x: number) => expect.closeTo(x, 6)));
    expect(s.lights).toHaveLength(0); // glTF exporter drops AREA lights (meta.notes)
    // flat-shaded walls face the interior; vertex normals agree with the geometric normal
    const floorTris = [...Array(34).keys()].filter((t) => g.triMaterial[t] === 0 && triNormal(s, t)[1] > 0.99);
    expect(floorTris.length).toBeGreaterThanOrEqual(2);
    for (let t = 0; t < 34; t++) {
      const n = triNormal(s, t);
      const vi = g.indices[3 * t];
      const dot = n[0] * g.normals[vi * 3] + n[1] * g.normals[vi * 3 + 1] + n[2] * g.normals[vi * 3 + 2];
      expect(dot).toBeGreaterThan(0.999);
      expect(g.triFlags[t]).toBe(0);
    }
    expect(g.tangents.every((x) => x === 0)).toBe(true); // no normal maps
  });
});

describe('KHR_texture_transform matrix', () => {
  it('matches the Blender importer (Mapping node in v-flipped space) and three.js GLTFLoader', () => {
    // Blender: texture_transform_gltf_to_blender → offset_b = (ox + sy·sinθ, 1 − oy − sy·cosθ), rot θ, scale s; the
    // Mapping node (POINT) computes out_b = offset_b + R_z(θ)·(s ⊙ uv_b) with uv_b = (u, 1 − v); glTF v' = 1 − out_b.y.
    let seed = 1;
    const rnd = () => ((seed = (seed * 16807) % 2147483647) / 2147483647) * 4 - 2;
    for (let k = 0; k < 50; k++) {
      const o: [number, number] = [rnd(), rnd()], th = rnd() * 2, sc: [number, number] = [rnd(), rnd()], u = rnd(), v = rnd();
      const c = Math.cos(th), s = Math.sin(th);
      const ob = [o[0] + sc[1] * s, 1 - o[1] - sc[1] * c];
      const pb = [sc[0] * u, sc[1] * (1 - v)];
      const outB = [ob[0] + c * pb[0] - s * pb[1], ob[1] + s * pb[0] + c * pb[1]];
      const blender = [outB[0], 1 - outB[1]];
      const X = textureTransformMatrix(o, th, sc);
      expect([X[0] * u + X[1] * v + X[2], X[3] * u + X[4] * v + X[5]]).toEqual(blender.map((x) => expect.closeTo(x, 10)));
      // three.js GLTFLoader extendTexture(): row-major [sx·c, sy·s, ox; −sx·s, sy·c, oy]
      expect(X).toEqual([sc[0] * c, sc[1] * s, o[0], -sc[0] * s, sc[1] * c, o[1]].map((x) => expect.closeTo(x, 12)));
    }
  });
});

describe('glTF loader: synthetic documents', () => {
  it('mirrored node flips winding (TRI_FLIPPED), normals use the inverse-transpose, flat normals when NORMAL is missing', async () => {
    const { doc, f32, u32, scene } = newDoc();
    const pos = f32('VEC3', [0, 0, 0, 1, 0, 0, 0, 1, 0]); // CCW in XY, front = +Z
    const flatMesh = doc.createMesh('flat').addPrimitive(doc.createPrimitive().setAttribute('POSITION', pos).setIndices(u32([0, 1, 2])));
    const n = Math.SQRT1_2;
    const smoothMesh = doc.createMesh('smooth').addPrimitive(doc.createPrimitive()
      .setAttribute('POSITION', pos).setAttribute('NORMAL', f32('VEC3', [n, 0, n, n, 0, n, n, 0, n])).setIndices(u32([0, 1, 2])));
    scene.addChild(doc.createNode('id').setMesh(flatMesh));
    scene.addChild(doc.createNode('mirror').setMesh(flatMesh).setScale([-1, 1, 1]).setTranslation([5, 0, 0]));
    scene.addChild(doc.createNode('stretch').setMesh(smoothMesh).setScale([2, 1, 1]));
    scene.addChild(doc.createNode('stretchMirror').setMesh(smoothMesh).setScale([2, 1, -1]));
    const { scene: s, stats } = await loadGltf({ kind: 'glb', bytes: await toGlb(doc) });
    const g = s.geometry;
    expect(g.indices.length / 3).toBe(4);
    expect(stats.flatten.flippedTriangles).toBe(2);
    expect([...g.triFlags]).toEqual([0, TRI_FLIPPED, 0, TRI_FLIPPED]);
    // Geometric normal of the flipped triangle still points to +Z (front face preserved under mirroring).
    expect(triNormal(s, 0)).toEqual([0, 0, 1].map((x) => expect.closeTo(x, 6)));
    expect(triNormal(s, 1)).toEqual([0, 0, 1].map((x) => expect.closeTo(x, 6)));
    // mirrored vertex positions: x → 5 − x
    const xs = [...g.indices.subarray(3, 6)].map((i) => g.positions[i * 3]).sort();
    expect(xs).toEqual([4, 5, 5]);
    // flat normals for the NORMAL-less mesh
    for (const t of [0, 1]) for (let c = 0; c < 3; c++) expect(g.normals[g.indices[3 * t + c] * 3 + 2]).toBeCloseTo(1, 6);
    // scale (2,1,1): n' ∝ M^-T n = (0.5, 0, 1)·n → normalize(0.5, 0, 1)
    const e = [0.5, 0, 1].map((x) => x / Math.hypot(0.5, 1));
    const vi = g.indices[6];
    expect([g.normals[vi * 3], g.normals[vi * 3 + 1], g.normals[vi * 3 + 2]]).toEqual(e.map((x) => expect.closeTo(x, 6)));
    // scale (2,1,−1): n' ∝ (0.5, 0, −1); winding flipped so the geometric normal is −Z and agrees with n'
    const vj = g.indices[9];
    expect([g.normals[vj * 3], g.normals[vj * 3 + 1], g.normals[vj * 3 + 2]]).toEqual([e[0], 0, -e[2]].map((x) => expect.closeTo(x, 6)));
    expect(triNormal(s, 3)[2]).toBeCloseTo(-1, 6);
    expect(s.bounds.min).toEqual([0, 0, 0].map((x) => expect.closeTo(x, 6)));
    expect(s.bounds.max).toEqual([5, 1, 0].map((x) => expect.closeTo(x, 6)));
  });

  it('drops degenerate triangles with a warning (dense primIds), handles strips and fans', async () => {
    const { doc, f32, u32, scene } = newDoc();
    const pos = f32('VEC3', [0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 0]);
    const list = doc.createPrimitive().setAttribute('POSITION', pos).setIndices(u32([0, 1, 2, 0, 0, 1, 1, 3, 2]));
    const strip = doc.createPrimitive().setAttribute('POSITION', pos).setIndices(u32([0, 1, 2, 3])).setMode(5);
    const fan = doc.createPrimitive().setAttribute('POSITION', pos).setIndices(u32([0, 1, 3, 2])).setMode(6);
    const pts = doc.createPrimitive().setAttribute('POSITION', pos).setMode(0);
    scene.addChild(doc.createNode().setMesh(doc.createMesh().addPrimitive(list).addPrimitive(strip).addPrimitive(fan).addPrimitive(pts)));
    const { scene: s, stats } = await loadGltf({ kind: 'glb', bytes: await toGlb(doc) });
    expect(stats.flatten.droppedDegenerate).toBe(1);
    expect(s.geometry.indices.length / 3).toBe(2 + 2 + 2);
    expect(s.warnings.some((w) => /zero-area/.test(w))).toBe(true);
    expect(s.warnings.some((w) => /non-triangle/.test(w))).toBe(true);
    // every kept triangle is CCW seen from +Z (glTF strip/fan winding rules)
    for (let t = 0; t < 6; t++) expect(triNormal(s, t)[2]).toBeCloseTo(1, 6);
  });

  it('maps materials (extensions, texture transform, BLEND→MASK, alpha/emissive flags, MikkTSpace tangents)', async () => {
    const { doc, f32, u32, scene } = newDoc();
    const tt = doc.createExtension(KHRTextureTransform);
    const ext = await import('@gltf-transform/extensions');
    const es = doc.createExtension(ext.KHRMaterialsEmissiveStrength);
    const iorX = doc.createExtension(ext.KHRMaterialsIOR);
    const specX = doc.createExtension(ext.KHRMaterialsSpecular);
    const trX = doc.createExtension(ext.KHRMaterialsTransmission);
    const opaquePng = encodePng(2, 2, new Uint8Array(16).fill(255));
    const cutPng = encodePng(2, 1, new Uint8Array([255, 255, 255, 255, 255, 255, 255, 0]));
    const texOpaque = doc.createTexture('opaque').setImage(opaquePng).setMimeType('image/png');
    const texCut = doc.createTexture('cut').setImage(cutPng).setMimeType('image/png');
    const texKtx = doc.createTexture('ktx').setImage(new Uint8Array([0xab, 0x4b, 0x54, 0x58])).setMimeType('image/ktx2');

    const mA = doc.createMaterial('A').setAlphaMode('MASK').setBaseColorTexture(texOpaque).setNormalTexture(texOpaque).setNormalScale(0.5)
      .setExtension('KHR_materials_emissive_strength', es.createEmissiveStrength().setEmissiveStrength(3))
      .setEmissiveFactor([1, 0.5, 0])
      .setExtension('KHR_materials_ior', iorX.createIOR().setIOR(1.33))
      .setExtension('KHR_materials_specular', specX.createSpecular().setSpecularFactor(0.25).setSpecularColorFactor([1, 0.5, 0.25]))
      .setExtension('KHR_materials_transmission', trX.createTransmission().setTransmissionFactor(0.75));
    mA.getBaseColorTextureInfo()!.setWrapS(TextureInfo.WrapMode.MIRRORED_REPEAT).setWrapT(TextureInfo.WrapMode.CLAMP_TO_EDGE);
    mA.getBaseColorTextureInfo()!.setExtension('KHR_texture_transform', tt.createTransform().setOffset([0.25, 0.5]).setRotation(Math.PI / 2).setScale([2, 3]));
    const mB = doc.createMaterial('B').setAlphaMode('BLEND').setBaseColorTexture(texCut);
    const mC = doc.createMaterial('C').setAlphaMode('MASK').setBaseColorTexture(texKtx);
    mB.getBaseColorTextureInfo()!.setMagFilter(TextureInfo.MagFilter.NEAREST);

    const pos = f32('VEC3', [0, 0, 0, 1, 0, 0, 0, 1, 0]);
    const nrm = f32('VEC3', [0, 0, 1, 0, 0, 1, 0, 0, 1]);
    const uv = f32('VEC2', [0, 0, 1, 0, 0, 1]);
    const prim = (m: typeof mA | null, color?: number[]) => {
      const p = doc.createPrimitive().setAttribute('POSITION', pos).setAttribute('NORMAL', nrm).setAttribute('TEXCOORD_0', uv).setIndices(u32([0, 1, 2]));
      if (m) p.setMaterial(m);
      if (color) p.setAttribute('COLOR_0', f32('VEC4', color));
      return p;
    };
    const mesh = doc.createMesh().addPrimitive(prim(mA)).addPrimitive(prim(mB)).addPrimitive(prim(mC)).addPrimitive(prim(null))
      .addPrimitive(prim(null, [1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1]));
    scene.addChild(doc.createNode().setMesh(mesh));
    const glb = await toGlb(doc);

    // Injected decoder (Node): return known pixels per image.
    const pixelsOf = new Map<string, { width: number; height: number; pixels: Uint8Array }>([
      ['opaque', { width: 2, height: 2, pixels: new Uint8Array(16).fill(255) }],
      ['cut', { width: 2, height: 1, pixels: new Uint8Array([255, 255, 255, 255, 255, 255, 255, 0]) }],
    ]);
    const { scene: s } = await loadGltf({ kind: 'glb', bytes: glb }, {
      decodeImage: async (bytes, mime, name) => (mime === 'image/png' ? pixelsOf.get(name) ?? null : null),
    });
    const [A, B, C, D, E] = s.materials;
    expect(s.materials.map((m) => m.name)).toEqual(['A', 'B', 'C', '__default', '__default_vertex_color']);
    expect(A.emissiveStrength).toBe(3);
    expect(A.ior).toBeCloseTo(1.33, 6);
    expect(A.specularFactor).toBe(0.25);
    expect(A.specularColorFactor).toEqual([1, 0.5, 0.25]);
    expect(A.transmissionFactor).toBe(0.75);
    expect(A.normalTexture?.scale).toBe(0.5);
    expect(A.baseColorTexture?.transform).toEqual(textureTransformMatrix([0.25, 0.5], Math.PI / 2, [2, 3]).map((x) => expect.closeTo(x, 12)));
    // rotation π/2: u' = s·sy·v + ox = 3v + 0.25; v' = −s·sx·u + oy = −2u + 0.5
    const X = A.baseColorTexture!.transform!;
    const apply = (u: number, v: number) => [X[0] * u + X[1] * v + X[2], X[3] * u + X[4] * v + X[5]];
    expect(apply(1, 0)).toEqual([expect.closeTo(0.25, 12), expect.closeTo(-1.5, 12)]);
    expect(apply(0, 1)).toEqual([expect.closeTo(3.25, 12), expect.closeTo(0.5, 12)]);
    const tA = s.textures[A.baseColorTexture!.texture];
    expect([tA.wrapS, tA.wrapT, tA.filter]).toEqual(['mirror-repeat', 'clamp-to-edge', 'linear']);
    // normal map uses the same image with default sampler → a second TextureData sharing the pixel buffer
    const tN = s.textures[A.normalTexture!.texture];
    expect(A.normalTexture!.texture).not.toBe(A.baseColorTexture!.texture);
    expect(tN.pixels).toBe(tA.pixels);
    expect(B.alphaMode).toBe('MASK');
    expect(B.alphaCutoff).toBe(0.5);
    expect(s.textures[B.baseColorTexture!.texture].filter).toBe('nearest');
    expect(C.baseColorTexture).toBeUndefined(); // KTX2 dropped
    expect(s.warnings.some((w) => /KTX2/.test(w))).toBe(true);
    expect(s.warnings.some((w) => /BLEND/.test(w))).toBe(true);
    expect(D.baseColorFactor).toEqual([0.8, 0.8, 0.8, 1]);
    expect(D.roughnessFactor).toBe(0.5);
    expect(E.baseColorFactor).toEqual([1, 1, 1, 1]);

    const f = s.geometry.triFlags;
    expect(f[0]).toBe(TRI_EMISSIVE);          // MASK with opaque texture and factor 1 → no any-hit
    expect(f[1]).toBe(TRI_ALPHA_MASK);        // BLEND→MASK with a transparent texel
    expect(f[2]).toBe(0);                     // MASK, KTX2 texture dropped → α = factor.a = 1 (constant)
    expect(f[3]).toBe(0);
    expect(f[4]).toBe(0);
    expect(s.geometry.color0).toBeDefined();
    // tangents: only material A (normal map + uv) → unit xyz along +u = +X, w = ±1
    const tg = s.geometry.tangents, idx = s.geometry.indices;
    for (let c = 0; c < 3; c++) {
      const v = idx[c];
      expect([tg[v * 4], tg[v * 4 + 1], tg[v * 4 + 2]]).toEqual([1, 0, 0].map((x) => expect.closeTo(x, 5)));
      // MikkTSpace on glTF uvs gives w = +1 here (∂p/∂v = +Y = N×T); glTF/Blender convention negates it: bitangent
      // N×T·w = −Y = ∂p/∂v_blender with v_blender = 1 − v.
      expect(tg[v * 4 + 3]).toBe(-1);
    }
    for (let c = 3; c < 15; c++) expect(tg[idx[c] * 4 + 3]).toBe(0);
  });

  it('decodes EXT_meshopt_compression and KHR_draco_mesh_compression', async () => {
    const { meshopt, draco } = await import('@gltf-transform/functions');
    const { MeshoptEncoder } = await import('meshoptimizer');
    await MeshoptEncoder.ready;
    const build = () => {
      const { doc, f32, u32, scene } = newDoc();
      const p: number[] = [], ix: number[] = [];
      for (let y = 0; y <= 8; y++) for (let x = 0; x <= 8; x++) p.push(x / 8, y / 8, 0);
      for (let y = 0; y < 8; y++) for (let x = 0; x < 8; x++) { const a = y * 9 + x; ix.push(a, a + 1, a + 10, a, a + 10, a + 9); }
      const n = p.map((_, i) => (i % 3 === 2 ? 1 : 0));
      scene.addChild(doc.createNode().setMesh(doc.createMesh().addPrimitive(doc.createPrimitive()
        .setAttribute('POSITION', f32('VEC3', p)).setAttribute('NORMAL', f32('VEC3', n)).setIndices(u32(ix)))));
      return doc;
    };
    const d1 = build();
    await d1.transform(meshopt({ encoder: MeshoptEncoder }));
    const glb1 = await toGlb(d1, { 'meshopt.encoder': MeshoptEncoder });
    const r1 = await loadGltf({ kind: 'glb', bytes: glb1 });
    expect(r1.stats.extensionsUsed).toContain('EXT_meshopt_compression');
    expect(r1.scene.geometry.indices.length / 3).toBe(128);
    r1.scene.bounds.max.forEach((x, k) => expect(x).toBeCloseTo([1, 1, 0][k], 3));

    // @ts-ignore draco3dgltf ships no type declarations
    const draco3d = (await import('draco3dgltf')).default;
    const d2 = build();
    await d2.transform(draco());
    const glb2 = await toGlb(d2, { 'draco3d.encoder': await draco3d.createEncoderModule() });
    const r2 = await loadGltf({ kind: 'glb', bytes: glb2 });
    expect(r2.stats.extensionsUsed).toContain('KHR_draco_mesh_compression');
    expect(r2.scene.geometry.indices.length / 3).toBe(128);
    r2.scene.bounds.max.forEach((x, k) => expect(x).toBeCloseTo([1, 1, 0][k], 2));
  });
});
