// M7 T3 fixtures (docs/decisions/m7-api.md §6.1 T3-M7; browser-safe, built in the Chrome test like make-m4.ts's t3Scene):
//   t3_smooth_256  t3_cases_256 + smooth-shaded objects (shared vertices, analytic normals, m7-kit.ts): a V1 GGX r 0.2
//                  sphere, a Principled plastic sphere, a metal torus and an OPEN wavy sheet (Ng·L ≤ 0 < Ns·L reachable);
//                  reconnections land on smooth triangles (counter smoothRc).
//   t3_nm_256      t3_smooth_256 with normal maps on those objects and on the floor / back wall (Principled, planar UVs):
//                  reconnection vertices on normal-mapped materials (counter nmRc), the bump-shadowing term on every shift.
// Tangents: Blender-semantics MikkTSpace on the quantized fixture (tangents.ts).
import type { EnvironmentData } from '../../src/core/scene/types.ts';
import { loadMikkTSpace } from '../../src/core/scene/gltf-loader.ts';
import { sceneTangents } from '../../src/core/scene/tangents.ts';
import { T3_MAT, principled, t3Scene, v1, type T3Scene } from './make-m4.ts';
import { bumpsMap, smoothSheet, smoothSphere, smoothTorus, tilesMap, wavesMap } from './m7-kit.ts';

export type T3M7Variant = 't3_smooth_256' | 't3_nm_256';

export async function t3M7Scene(variant: T3M7Variant, env?: EnvironmentData): Promise<T3Scene> {
  const nm = variant === 't3_nm_256';
  const t = t3Scene('t3_cases_256', env, (b) => {
    const tx = b.textures.length;
    if (nm) b.textures.push(tilesMap('tiles', 128, 4), bumpsMap('bumps', 128, 6, 0.8), wavesMap('waves', 128, 5));
    const N = (i: number, s: number) => (nm ? { normalTexture: { texture: tx + i, texCoord: 0, scale: s } } : {});
    const m0 = b.materials.length;
    b.materials.push(
      nm ? principled('glossy_nm', { baseColorFactor: [0.9, 0.9, 0.9, 1], metallicFactor: 1, roughnessFactor: 0.45, ...N(1, 1) })
        : v1('glossy_smooth', { diffuse: [0, 0, 0], glossy: [0.9, 0.9, 0.9], roughness: 0.2, mix: 1 }),
      principled('plastic_smooth', { baseColorFactor: [0.2, 0.4, 0.8, 1], roughnessFactor: 0.3, ...N(0, 1) }),
      principled('metal_torus', { baseColorFactor: [0.95, 0.7, 0.4, 1], metallicFactor: 1, roughnessFactor: 0.3, ...N(2, 0.8) }),
      principled('sheet', { baseColorFactor: [0.7, 0.7, 0.5, 1], roughnessFactor: 0.4, ...N(1, 1.3) }),
    );
    smoothSphere(b.mesh, [-0.45, 0.62, 0.45], 0.2, 16, 32, m0, { uvScale: [3, 1.5] });
    smoothSphere(b.mesh, [0.4, 0.6, 0.55], 0.22, 16, 32, m0 + 1, { uvScale: [2, 1] });
    smoothTorus(b.mesh, [0.45, 0.06, 0.45], 0.2, 0.06, 32, 12, m0 + 2, { uvScale: [4, 1] });
    smoothSheet(b.mesh, -1.3, -0.3, -1.4, -0.6, 1.25, 0.05, 6, 5, 16, m0 + 3, { uvScale: [2, 1.6] });
    if (nm) {
      // floor and back wall: normal-mapped Principled with planar UVs (their quads are the mesh's first 12 vertices)
      b.materials[T3_MAT.floor] = principled('floor_nm', { baseColorFactor: [0.7, 0.7, 0.7, 1], roughnessFactor: 0.5, ...N(0, 1) });
      b.materials[T3_MAT.back] = principled('back_nm', { baseColorFactor: [0.6, 0.6, 0.5, 1], roughnessFactor: 0.6, ...N(1, 0.6) });
      for (let v = 0; v < 12; v++) {
        const x = b.mesh.pos[3 * v], y = b.mesh.pos[3 * v + 1], z = b.mesh.pos[3 * v + 2];
        b.mesh.uv[2 * v] = 0.8 * x;
        b.mesh.uv[2 * v + 1] = v < 6 ? 0.8 * z : -0.8 * y;
      }
    }
  });
  const scene = nm ? sceneTangents(t.scene, await loadMikkTSpace(), { quantized: true, flatFaceNormals: true }).scene : t.scene;
  scene.name = variant;
  return { ...t, scene, notes: `${variant}: t3_cases_256 + smooth objects${nm ? ' + normal maps (floor, back wall, objects)' : ''} (m7-api.md §6.1)` };
}
