// M7 performance at 960×540, interactive (docs/decisions/m7-api.md §8; coordinator: "performance impact at 540p
// interactive (report only)"). The same scene with and without its normal maps (NORMAL_MAP build vs the M6 text): the
// shipped interactive preset (every M6 feature, light mode B) and the reference PT frame kernel (1 spp). Each work unit is
// submitted on its own and timed by wall clock around onSubmittedWorkDone (Q3: no timestampWrites around ReSTIR passes);
// an empty submit's round trip is subtracted. Scenes: m7_nm_smooth_256 (normal maps on smooth + flat geometry, rect + point)
// with / without its normal textures, and cornell_i_512 (flat, the M6 baseline). Reported, not gating.
//   VITE_M7_PERF_FRAMES (default 24 timed frames after 8 warm-up frames)
import { describe, expect, it } from 'vitest';
import { RestirKernel } from '../../src/core/render/restir/kernel.ts';
import { restirSettings } from '../../src/core/render/restir/presets.ts';
import { JITTER_IID } from '../../src/core/render/frame-uniforms.ts';
import { fetchScenePackage } from '../../src/core/scene/scene-package.ts';
import type { SceneData } from '../../src/core/scene/types.ts';
import { gpuScene } from './restir-fixtures.ts';

const FRAMES = Number(import.meta.env?.VITE_M7_PERF_FRAMES ?? 24);
const WARM = 8;
const W = 960, H = 540;
const SCENES: { name: string; url: string; strip?: boolean }[] = [
  { name: 'm7_nm_smooth (normal maps)', url: '/validation/out/m7/scenes/m7_nm_smooth_256/' },
  { name: 'm7_nm_smooth without normal maps', url: '/validation/out/m7/scenes/m7_nm_smooth_256/', strip: true },
  { name: 'cornell_i (flat)', url: '/validation/scenes/cornell_i_512/' },
];
const kindOf = (label: string) => label.replace(/[-:#\s].*$/, '');

describe('M7 perf: ms per pass at 960×540, interactive (reported)', () => {
  it('normal maps on / off, and the flat Cornell baseline', async () => {
    const out: Record<string, Record<string, number>> = {};
    for (const sc of SCENES) {
      let pkg;
      try { pkg = await fetchScenePackage(sc.url); } catch (e) { console.warn(`[perf] ${sc.name}: package missing (${(e as Error).message})`); continue; }
      const scene: SceneData = sc.strip ? { ...pkg.scene, materials: pkg.scene.materials.map((m) => ({ ...m, normalTexture: undefined })) } : pkg.scene;
      const g = await gpuScene(scene);
      const device = g.device;
      const nm = !!g.gpu.defines(1).NORMAL_MAP;
      const accum = device.createBuffer({ size: W * H * 16, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
      const counters = device.createBuffer({ size: 16, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
      const rt: number[] = [];
      for (let i = 0; i < 64; i++) { const t0 = performance.now(); device.queue.submit([device.createCommandEncoder().finish()]); await device.queue.onSubmittedWorkDone(); rt.push(performance.now() - t0); }
      rt.sort((a, b) => a - b);
      const base = rt[rt.length >> 1];
      const cam0 = { camToWorld: Array.from(pkg.camera.matrix), yfov: pkg.camera.yfov };
      const settings = restirSettings('interactive', { maxBounces: pkg.render.maxBounces ?? 3 });
      const kernel = await RestirKernel.create(device, g.gpu, g.env, { settings, lightMode: 'B', features: g.features, wgslLanguageFeatures: g.wgslLanguageFeatures });
      kernel.setView({ camera: cam0, width: W, height: H, runSeed: 7, jitterMode: JITTER_IID });
      await kernel.prepare();
      const per = new Map<string, number[]>();
      for (let t = 0; t < WARM + FRAMES; t++) {
        const m = cam0.camToWorld.slice();
        m[12] += 0.002 * t * m[0]; m[13] += 0.002 * t * m[1]; m[14] += 0.002 * t * m[2];
        kernel.advance({ t, camera: { camToWorld: m, yfov: cam0.yfov }, lights: scene.lights });
        kernel.beginSubmit();
        const sums = new Map<string, number>();
        for (const u of kernel.frameUnits(t, { accum, counters })) {
          const enc = device.createCommandEncoder();
          u.encode(enc);
          const t0 = performance.now();
          device.queue.submit([enc.finish()]);
          await device.queue.onSubmittedWorkDone();
          const k = kindOf(u.label);
          sums.set(k, (sums.get(k) ?? 0) + Math.max(0, performance.now() - t0 - base));
        }
        if (t >= WARM) for (const [k, v] of sums) per.set(k, [...(per.get(k) ?? []), v]);
      }
      const med = (a: number[]) => { const s = [...a].sort((x, y) => x - y); return s[s.length >> 1]; };
      const row: Record<string, number> = { NORMAL_MAP: nm ? 1 : 0 };
      let total = 0;
      for (const [k, v] of per) { row[k] = Number(med(v).toFixed(3)); total += med(v); }
      row.total = Number(total.toFixed(3));
      out[sc.name] = row;
      expect(per.size).toBeGreaterThan(0);
      kernel.destroy(); accum.destroy(); counters.destroy(); g.destroy();
    }
    console.log(`[M7 perf ${W}x${H} interactive, Mode B]\n${JSON.stringify(out, null, 1)}`);
  }, 1_800_000);
});
