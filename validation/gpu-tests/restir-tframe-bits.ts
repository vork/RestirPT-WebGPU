// U-M4-BITS rig (restir-temporal-api.md §6.1, T-A): renders the M4 rungs with temporal OFF on (i) 128² (cornell_i_512
// at 128²) and (x) quads 64² and hashes every reservoir buffer, the final reservoirs and the finalize output. The same
// file runs unchanged on the pre-M5 build (it uses only the M4 kernel API), which recorded the golden hashes in
// restir-tframe.gpu.test.ts; `extra` settings let the M5 build add `temporal: false` explicitly.
import { fetchScenePackage } from '../../src/core/scene/scene-package.ts';
import type { SceneData } from '../../src/core/scene/types.ts';
import { bitFixtureScene, boxCamera, hashF32, restirRig } from './restir-fixtures.ts';

export interface BitsCase { name: string; scene: 'i' | 'x_quads'; preset: 'initial' | 'offline' | 'interactive'; settings: Record<string, unknown>; frames: number }

export const BITS_CASES: BitsCase[] = [
  { name: 'i-3.1', scene: 'i', preset: 'initial', settings: { maxBounces: 3 }, frames: 2 },
  { name: 'i-3.2', scene: 'i', preset: 'offline', settings: { maxBounces: 3, trees: 2, rounds: 2, slots: 3 }, frames: 2 },
  { name: 'i-interactive', scene: 'i', preset: 'interactive', settings: { maxBounces: 3 }, frames: 2 },
  { name: 'xq-3.1', scene: 'x_quads', preset: 'initial', settings: { maxBounces: 3 }, frames: 2 },
  { name: 'xq-3.2', scene: 'x_quads', preset: 'offline', settings: { maxBounces: 3, trees: 2, rounds: 3, slots: 6 }, frames: 2 },
  { name: 'xq-interactive', scene: 'x_quads', preset: 'interactive', settings: { maxBounces: 3 }, frames: 2 },
];

const u32Hash = (u: Uint32Array) => hashF32(new Float32Array(u.buffer, u.byteOffset, u.length));

/** Hashes of one case: res[0], res[1], final reservoirs, finalize mean image, counters (finalize + arena RSC). */
export async function bitsCase(c: BitsCase, extra: Record<string, unknown> = {}): Promise<Record<string, string>> {
  let scene: SceneData, cam = boxCamera(), W = 64, H = 64;
  if (c.scene === 'i') {
    const pkg = await fetchScenePackage('/validation/scenes/cornell_i_512/');
    scene = pkg.scene;
    cam = { camToWorld: Array.from(pkg.camera.matrix), yfov: pkg.camera.yfov };
    W = 128; H = 128;
  } else {
    scene = bitFixtureScene('x_quads');
  }
  const rig = await restirRig(scene, W, H, { preset: c.preset, settings: { ...c.settings, ...extra } as never, cam, seed: 23 });
  const r = await rig.frames(c.frames, 5);
  const out: Record<string, string> = {
    res0: u32Hash(await rig.kernel.readReservoirs(0)),
    res1: u32Hash(await rig.kernel.readReservoirs(1)),
    final: u32Hash(await rig.kernel.readReservoirs('final')),
    image: hashF32(r.mean),
    counters: r.counters.join(','),
    arena: [r.arena.rsc.accepted, r.arena.rsc.queued, r.arena.rsc.selectedShifted, r.arena.rsc.emptyCanon, r.arena.rsc.pendingLeft].join(','),
  };
  rig.destroy();
  return out;
}
