// U-M7-ARENA (docs/decisions/m7-api.md §2, M7 Gate 0): companion of U-M7-BITS on the data side. For committed packages
// WITHOUT normal maps, readScenePackage (M7: + withSceneTangents) and packVertexArena (M7: + optional tangent section)
// produce exactly the bytes of the M6 build (main b5b0f5d): geometry buffers, materials and the vertex-arena words. The
// digests were recorded by running this probe against `git archive b5b0f5d src` (validation/out is not needed).
import { createHash } from 'node:crypto';
import { readdirSync, readFileSync } from 'node:fs';
import path from 'node:path';
import { describe, expect, it } from 'vitest';
import { packVertexArena } from '../../src/core/gpu/vertex-format.ts';
import { readScenePackage } from '../../src/core/scene/scene-package.ts';

const GOLDEN: Record<string, string> = {
  cornell_i_512: '06485d15364cbd2f', v_glossy_v1_512: '9f8b8b1d0b5a9ea2', xii_alpha_foliage_512: '532bf5f1967b8849',
  iv_emissive_mesh_512: '83dced3c6e6b7d17', g8_cornell_glass_512: 'b4166bce90eec7e8',
};

async function digest(dir: string): Promise<string> {
  const files = new Map(readdirSync(dir).map((n) => [n, new Uint8Array(readFileSync(path.join(dir, n)))]));
  const { scene } = await readScenePackage(files);
  const g = scene.geometry;
  const h = createHash('sha256');
  for (const k of ['positions', 'normals', 'uv0', 'indices', 'triMaterial', 'triFlags'] as const) {
    const a = g[k] as ArrayBufferView | undefined;
    if (a) h.update(new Uint8Array(a.buffer, a.byteOffset, a.byteLength));
  }
  h.update(JSON.stringify(scene.materials));
  const origin = [0.125, -0.25, 0.5];
  const rc = new Float32Array(g.positions.length);
  for (let i = 0; i < rc.length; i++) rc[i] = g.positions[i] - origin[i % 3];
  const w = packVertexArena(g, rc, scene.quant, origin).words;
  h.update(new Uint8Array(w.buffer, w.byteOffset, w.byteLength));
  return h.digest('hex').slice(0, 16);
}

describe('U-M7-ARENA: packages without normal maps load and pack to the M6 bytes', () => {
  for (const [pkg, want] of Object.entries(GOLDEN)) {
    it(pkg, async () => {
      expect(await digest(path.join('validation/scenes', pkg))).toBe(want);
    });
  }
});
