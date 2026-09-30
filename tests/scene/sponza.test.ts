import { existsSync, readdirSync, readFileSync } from 'node:fs';
import { describe, expect, it } from 'vitest';
import { loadGltf } from '../../src/core/scene/gltf-loader.ts';
import { TRI_ALPHA_MASK } from '../../src/core/scene/types.ts';
import { computeRenderOrigin } from '../../src/core/render/frame-uniforms.ts';
import { recentrePositions } from '../../src/core/render/scene-gpu.ts';

// Khronos Sponza (gitignored; re-fetch with validation/blender/fetch_sponza.py). Geometry only in Node.
const dir = new URL('../../validation/assets/downloaded/sponza/', import.meta.url);
const have = existsSync(new URL('Sponza.gltf', dir));

describe.skipIf(!have)('glTF loader: Sponza (Node, geometry only)', () => {
  it('loads 262k triangles with MASK foliage and MikkTSpace tangents', async () => {
    const json = JSON.parse(readFileSync(new URL('Sponza.gltf', dir), 'utf8'));
    const resources: Record<string, Uint8Array> = {};
    for (const f of readdirSync(dir)) if (f !== 'Sponza.gltf' && f !== 'manifest.json') resources[f] = new Uint8Array(readFileSync(new URL(f, dir)));
    const t0 = performance.now();
    const { scene: s, stats } = await loadGltf({ kind: 'gltf', json, resources, name: 'Sponza.gltf' });
    const ms = performance.now() - t0;
    console.log('SPONZA_LOAD', JSON.stringify({ ms: Math.round(ms), ...stats.ms, flatten: stats.flatten, textures: s.textures.length, warnings: s.warnings }));
    const { uv, ...q } = stats.quantize;
    console.log('SPONZA_QUANT', JSON.stringify(q), JSON.stringify(uv.map((u) => [u.material, u.ku, u.kv, u.wide, +u.worstTexel.toFixed(4)])));
    const g = s.geometry;
    const nt = g.indices.length / 3;
    expect(nt + stats.flatten.droppedDegenerate + stats.flatten.droppedNonFinite).toBe(262267);
    expect(nt).toBeGreaterThan(260000);
    expect(s.materials).toHaveLength(25);
    expect(s.textures.length).toBeGreaterThan(60);
    const nv = g.positions.length / 3;
    let maxIdx = 0;
    for (const i of g.indices) if (i > maxIdx) maxIdx = i;
    expect(maxIdx).toBeLessThan(nv);
    // every referenced vertex has a unit normal; normal-mapped vertices have unit tangents with w = ±1
    let badN = 0, tanVerts = 0, badT = 0;
    for (const v of g.indices) {
      const l = Math.hypot(g.normals[v * 3], g.normals[v * 3 + 1], g.normals[v * 3 + 2]);
      if (Math.abs(l - 1) > 1e-4) badN++;
      const w = g.tangents[v * 4 + 3];
      if (w !== 0) {
        tanVerts++;
        const lt = Math.hypot(g.tangents[v * 4], g.tangents[v * 4 + 1], g.tangents[v * 4 + 2]);
        if (Math.abs(Math.abs(w) - 1) > 0 || Math.abs(lt - 1) > 1e-3) badT++;
      }
    }
    expect(badN).toBe(0);
    expect(tanVerts).toBeGreaterThan(0.5 * g.indices.length);
    expect(badT).toBe(0);
    // MASK materials with a base-colour texture whose alpha is unknown (not decoded) → any-hit flag set
    const masked = s.materials.map((m, i) => (m.alphaMode === 'MASK' ? i : -1)).filter((i) => i >= 0);
    expect(masked.length).toBe(3);
    let flagged = 0;
    for (let t = 0; t < nt; t++) if (g.triFlags[t] & TRI_ALPHA_MASK) { flagged++; expect(masked).toContain(g.triMaterial[t]); }
    expect(flagged).toBeGreaterThan(0);
    const ext = s.bounds.max.map((x, k) => x - s.bounds.min[k]);
    expect(Math.max(...ext)).toBeGreaterThan(20); // ~30 m long atrium
    expect(ms).toBeLessThan(30_000);
    // data-formats.md §B1/§B3/§B5 on Sponza: k = −16 (15.3 µm), worst 13.2 µm, oct errors, wide materials 5/6/7,
    // the re-weld (786,783 → ≈ 198k), and U-Q2: recentred coordinates and MT edges exact in f32 (872,829 → 0).
    expect(stats.quantize.posLog2).toBe(-16);
    expect(stats.quantize.maxPosErr).toBeLessThanOrEqual(13.3e-6);
    expect(stats.quantize.maxNormalErrDeg).toBeLessThanOrEqual(0.0025);
    expect(stats.quantize.maxTangentErrDeg).toBeLessThanOrEqual(0.005);
    expect(stats.quantize.wideMaterials).toEqual([5, 6, 7]);
    expect(stats.quantize.droppedDegenerate).toBe(0);
    expect(nv).toBeLessThan(200_000);
    const O = computeRenderOrigin(s.bounds, s.quant);
    const r = recentrePositions(g.positions, O);
    let inexact = 0;
    for (let i = 0; i < r.length; i++) if (r[i] !== g.positions[i] - O[i % 3]) inexact++;
    for (let t = 0; t < g.indices.length; t += 3) for (const c of [1, 2]) for (let k = 0; k < 3; k++) {
      const d = r[3 * g.indices[t + c] + k] - r[3 * g.indices[t] + k];
      if (Math.fround(d) !== d) inexact++;
    }
    expect(inexact).toBe(0);
  }, 60_000);
});
