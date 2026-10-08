// M6 performance at 960×540, interactive configuration (restir-m6-api.md §6 R2; coordinator: "perf impact at 540p
// interactive, ms per pass"). Reported, not gating. Each work unit of a temporal frame is submitted on its own and timed
// by wall clock around queue.onSubmittedWorkDone() (Q3: no timestampWrites around ReSTIR passes); an empty submit's
// round trip is measured and subtracted. Configurations: the M5 interactive configuration (M6 features off), each M6
// feature alone, all of them (the shipped interactive preset), and all of them in Mode B; scenes: (i) Cornell (rect
// light) and m6_crossings_B_256 (point / spot / rect / disk / sun / emissive + 3 crossing lights) at 960×540.
//   VITE_M6_PERF_FRAMES (default 24 timed frames after 8 warm-up frames)
import { describe, expect, it } from 'vitest';
import { RestirKernel } from '../../src/core/render/restir/kernel.ts';
import { M6_OFF, restirSettings, type RestirSettings } from '../../src/core/render/restir/presets.ts';
import { JITTER_IID } from '../../src/core/render/frame-uniforms.ts';
import { fetchScenePackage } from '../../src/core/scene/scene-package.ts';
import type { LightMode } from '../../src/core/render/lights-gpu.ts';
import { gpuScene } from './restir-fixtures.ts';

const FRAMES = Number(import.meta.env?.VITE_M6_PERF_FRAMES ?? 24);
const WARM = 8;
const W = 960, H = 540;

interface Cfg { name: string; settings: Partial<RestirSettings>; mode: LightMode }
const CFGS: Cfg[] = [
  { name: 'M5 interactive (M6 off)', settings: { ...M6_OFF }, mode: 'A' },
  { name: '+ σ 16 pairing', settings: { ...M6_OFF, pairing: 'gauss' }, mode: 'A' },
  { name: '+ RIS-NEE (M 32)', settings: { ...M6_OFF, risNee: true }, mode: 'A' },
  { name: '+ dual MVs', settings: { ...M6_OFF, dualMv: true }, mode: 'A' },
  { name: '+ duplication map', settings: { ...M6_OFF, dupmap: true }, mode: 'A' },
  { name: 'M6 interactive (all)', settings: {}, mode: 'A' },
  { name: 'M6 interactive (all), Mode B', settings: {}, mode: 'B' },
];
const SCENES = [
  { name: 'cornell_i', url: '/validation/scenes/cornell_i_512/' },
  { name: 'm6_crossings', url: '/validation/out/m6/scenes/m6_crossings_B_256/' },
];

const kindOf = (label: string) => label.replace(/[-:#\s].*$/, '');

describe('M6 perf: ms per pass at 960×540, interactive (reported)', () => {
  for (const sc of SCENES) {
    it(sc.name, async () => {
      let pkg;
      try { pkg = await fetchScenePackage(sc.url); } catch (e) { console.warn(`[perf] ${sc.name}: package missing (${(e as Error).message})`); return; }
      const g = await gpuScene(pkg.scene);
      const device = g.device;
      const accum = device.createBuffer({ size: W * H * 16, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
      const counters = device.createBuffer({ size: 16, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
      // empty-submit round trip (subtracted from every unit)
      const rt: number[] = [];
      for (let i = 0; i < 64; i++) { const t0 = performance.now(); device.queue.submit([device.createCommandEncoder().finish()]); await device.queue.onSubmittedWorkDone(); rt.push(performance.now() - t0); }
      rt.sort((a, b) => a - b);
      const base = rt[rt.length >> 1];
      const report: Record<string, Record<string, number>> = {};
      const cam0 = { camToWorld: Array.from(pkg.camera.matrix), yfov: pkg.camera.yfov };
      for (const c of CFGS) {
        const settings = restirSettings('interactive', { maxBounces: pkg.render.maxBounces ?? 3, ...c.settings });
        const kernel = await RestirKernel.create(device, g.gpu, g.env, { settings, lightMode: c.mode, features: g.features, wgslLanguageFeatures: g.wgslLanguageFeatures });
        kernel.setView({ camera: cam0, width: W, height: H, runSeed: 7, jitterMode: JITTER_IID });
        await kernel.prepare();
        const per = new Map<string, number[]>();
        let nonEmpty = 0;
        for (let t = 0; t < WARM + FRAMES; t++) {
          // a slow camera pan (temporal reuse, dual-MV taps and disocclusions are exercised)
          const m = cam0.camToWorld.slice();
          m[12] += 0.004 * t * m[0]; m[13] += 0.004 * t * m[1]; m[14] += 0.004 * t * m[2];   // pan along the camera's x axis
          kernel.advance({ t, camera: { camToWorld: m, yfov: cam0.yfov }, lights: pkg.scene.lights });
          kernel.beginSubmit();
          const sums = new Map<string, number>();
          for (const u of kernel.frameUnits(t, { accum, counters })) {
            const enc = device.createCommandEncoder();
            u.encode(enc);
            const t0 = performance.now();
            device.queue.submit([enc.finish()]);
            await device.queue.onSubmittedWorkDone();
            const ms = Math.max(0, performance.now() - t0 - base);
            const k = kindOf(u.label);
            sums.set(k, (sums.get(k) ?? 0) + ms);
          }
          if (t >= WARM) { for (const [k, v] of sums) per.set(k, [...(per.get(k) ?? []), v]); nonEmpty++; }
        }
        const med = (a: number[]) => { const s = [...a].sort((x, y) => x - y); return s[s.length >> 1]; };
        const row: Record<string, number> = {};
        let total = 0;
        for (const [k, v] of per) { row[k] = Number(med(v).toFixed(3)); total += med(v); }
        row.total = Number(total.toFixed(3));
        report[c.name] = row;
        expect(nonEmpty).toBe(FRAMES);
        kernel.destroy();
      }
      console.log(`[M6 perf ${sc.name} ${W}x${H}] empty-submit round trip ${base.toFixed(3)} ms (subtracted)\n${JSON.stringify(report, null, 1)}`);
      accum.destroy(); counters.destroy(); g.destroy();
    }, 1_800_000);
  }
});
