// perf2 WP-2d (docs/decisions/perf2-plan.md §2 WP-2 step 2d; perf2-api.md): RS_RIS_PREPASS — the RIS-NEE selection at
// x₁ runs as the lean pre-pass rs_ris_nee and reaches rs_initial through the pixel's RP_DIAG plane.
//   U-WP2D-DIAG  the diagnostic of the plan: rs_initial compiled with RS_RIS_PREPASS_DIAG (tests only) evaluates the
//                inline ris_nee_select next to the pre-pass record at every B = 1 NEE site and counts (arena header words
//                40–43, free while no spatial / temporal pass runs): compared sites, endpoint (entry, a, b) or mult
//                mismatches (must be 0), W bit mismatches and the max relative |ΔW| (must be ≤ 1e-5). Scenes: the
//                all-lights fixture (Mode A and B), the normal-mapped HDRI package m7_nm_env_256, Sponza-lite (CWBVH, MT,
//                textures, SoA), an ensemble (E = 4) and the U8-10 plant (the stored multiplicity).
//   U-WP2D-SCHED the kernel runs the pre-pass only with RIS-NEE, trees = 1, no candidate dump and no test text; its
//                variant key separates the two texts.
import { afterAll, describe, expect, it } from 'vitest';
import { releaseTestGpu } from './device-factory.ts';
import type { SceneData } from '../../src/core/scene/types.ts';
import type { RestirPresetName, RestirSettings } from '../../src/core/render/restir/presets.ts';
import type { LightMode } from '../../src/core/render/lights-gpu.ts';
import { fetchScenePackage } from '../../src/core/scene/scene-package.ts';
import { allLightsScene, boxCamera, restirRig, testPerfFlags, type GpuSceneOptions } from './restir-fixtures.ts';
import { normalizePerfFlags } from '../../src/core/render/restir/perf-flags.ts';
import { m8BitsScene } from './m8-bits.ts';

afterAll(async () => { await releaseTestGpu(); });

type Scene = 'all' | 'nm_env' | 'sponza_lite';
async function sceneOf(s: Scene): Promise<{ scene: SceneData; cam: { camToWorld: number[]; yfov: number } }> {
  if (s === 'all') return { scene: allLightsScene(), cam: boxCamera() };
  if (s === 'sponza_lite') return m8BitsScene('sponza_lite');
  const p = await fetchScenePackage('/validation/out/m7/scenes/m7_nm_env_256/');
  return { scene: p.scene, cam: { camToWorld: Array.from(p.camera.matrix), yfov: p.camera.yfov } };
}

interface DiagCase {
  name: string; scene: Scene; lightMode: LightMode; settings: Partial<RestirSettings>; size?: [number, number]; members?: number;
  gpu?: GpuSceneOptions; resLayout?: 'aos' | 'soa'; frames?: number;
}
const RIS: Partial<RestirSettings> = { risNee: true, risM: 32, maxBounces: 3 };
const CASES: DiagCase[] = [
  { name: 'all-lights, Mode A', scene: 'all', lightMode: 'A', settings: RIS },
  { name: 'all-lights, Mode B', scene: 'all', lightMode: 'B', settings: RIS },
  { name: 'all-lights, Mode B, RR, risM 8', scene: 'all', lightMode: 'B', settings: { ...RIS, rr: true, rrMinBounces: 1, risM: 8 } },
  { name: 'all-lights, Mode A, ensemble E = 4', scene: 'all', lightMode: 'A', settings: RIS, members: 4, size: [48, 48] },
  { name: 'all-lights, Mode A, U8-10 plant (stored multiplicity)', scene: 'all', lightMode: 'A', settings: { ...RIS, plant: { u8TilePmf: true } } },
  { name: 'm7_nm_env_256 (HDRI, normal maps)', scene: 'nm_env', lightMode: 'A', settings: RIS },
  { name: 'Sponza-lite (CWBVH, MT, textures, SoA)', scene: 'sponza_lite', lightMode: 'B', settings: RIS, size: [160, 90],
    gpu: { bvhKind: 'cwbvh', watertight: false }, resLayout: 'soa' },
];

describe('U-WP2D-DIAG: the pre-pass selection ≡ the inline ris_nee_select (endpoint exact, W within 1e-5)', () => {
  for (const c of CASES) {
    it(c.name, async () => {
      const { scene, cam } = await sceneOf(c.scene);
      const [W, H] = c.size ?? [64, 64];
      const rig = await restirRig(scene, W, H, {
        preset: 'initial' as RestirPresetName, settings: c.settings, lightMode: c.lightMode, cam, members: c.members, gpu: c.gpu, resLayout: c.resLayout,
        modeBNeedsAreaLights: c.scene === 'sponza_lite', perfFlags: { ...normalizePerfFlags(testPerfFlags()), RS_RIS_HOIST: 1, RS_NEE_SITE: 1, RS_RIS_PREPASS: 1 }, initialDefines: { RS_RIS_PREPASS_DIAG: 1 }, seed: 23,
      });
      expect(rig.kernel.risPrepassActive()).toBe(true);
      expect(rig.kernel.frameUnits(0, { accum: rig.accum, counters: rig.counters }).some((u) => u.label.startsWith('rs_ris_nee'))).toBe(true);
      const r = await rig.frames(c.frames ?? 4, 3);
      const raw = r.arena.raw;
      const d = { sites: raw[40], epMismatch: raw[41], wBitMismatch: raw[42], maxRelW: new Float32Array(new Uint32Array([raw[43]]).buffer)[0] };
      console.log(`[U-WP2D-DIAG] ${c.name}: ${JSON.stringify(d)}`);
      rig.destroy();
      expect(d.sites).toBeGreaterThan(0);
      expect(d.epMismatch).toBe(0);
      expect(d.maxRelW).toBeLessThanOrEqual(1e-5);
    }, 300_000);
  }
});

describe('U-WP2D-SCHED: when the kernel runs rs_ris_nee', () => {
  it('RIS-NEE, trees = 1, no dump / test text; variant keys differ', async () => {
    const base = { preset: 'initial' as RestirPresetName, settings: RIS, perfFlags: 'RS_RIS_PREPASS' };
    const on = await restirRig(allLightsScene(), 16, 16, base);
    expect(on.kernel.risPrepassActive()).toBe(true);
    expect(on.kernel.defines('rs_initial').RS_RIS_PREPASS).toBe(1);
    expect(on.kernel.defines('rs_spatial_shift').RS_RIS_PREPASS).toBeUndefined();
    expect(on.kernel.customDefines({}).RS_RIS_PREPASS).toBeUndefined();
    const keyOn = on.kernel.variantKey();
    on.kernel.setSettings({ trees: 2 });
    expect(on.kernel.risPrepassActive()).toBe(false);
    expect(on.kernel.defines('rs_initial').RS_RIS_PREPASS).toBeUndefined();
    expect(on.kernel.variantKey()).not.toBe(keyOn);
    expect(on.kernel.isPrepared()).toBe(false);
    await on.kernel.prepare();
    expect(on.kernel.frameUnits(0, { accum: on.accum, counters: on.counters }).some((u) => u.label.startsWith('rs_ris_nee'))).toBe(false);
    on.destroy();
    const noRis = await restirRig(allLightsScene(), 16, 16, { ...base, settings: { ...RIS, risNee: false } });
    expect(noRis.kernel.risPrepassActive()).toBe(false);
    noRis.destroy();
    const dump = await restirRig(allLightsScene(), 16, 16, { ...base, dumpCandidates: true });
    expect(dump.kernel.risPrepassActive()).toBe(false);
    dump.destroy();
    const test = await restirRig(allLightsScene(), 16, 16, { ...base, initialDefines: { RS_PT_DIRECTIONS: 1 } });
    expect(test.kernel.risPrepassActive()).toBe(false);
    test.destroy();
  }, 300_000);
});
