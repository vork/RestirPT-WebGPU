// perf2 WP-Q (docs/decisions/perf2-plan.md, equal-quality harness), browser side: export a run-perf.ts PERF_SCENES glTF
// setup (Sponza + HDRI) as a scene package, so the denoiser runner (denoise-run.ts) and the PT reference (run-batches.ts)
// render exactly the scene that run-perf.ts times. The camera and the warm point light are perf-run.ts renderPerf's
// glTF "autoSetup" (camera along the long axis at 20 % height, 55° vfov; one 2000 W point light at 25 % height) — kept
// numerically identical here (perf-run.ts is WP-0's file; no refactor across packages). The env is loaded in
// 'validation' mode as perf-run.ts does. Uploads validation/out/<run>/{scene.json, geometry.bin, tex_*.png, env.exr}.
import { exportAndUpload } from './export-package.ts';
import { loadScene } from '../../src/core/scene/load-scene.ts';
import { loadEnvironment } from '../../src/core/scene/env/load-env.ts';
import { ensureLightStore } from '../../src/core/scene/light-store.ts';
import type { LightMode } from '../../src/core/scene/scene-package.ts';
import type { LightData, SceneData } from '../../src/core/scene/types.ts';

export interface ExportPerfSceneOptions {
  run: string;
  /** glTF URL (PERF_SCENES[..].scene). */
  scene: string;
  /** HDRI URL (PERF_SCENES[..].env). */
  env?: string;
  width: number; height: number; maxBounces: number;
  lightMode: LightMode;
  name?: string;
}

/** perf-run.ts lookAt (column-major camToWorld, −z forward, +y up). */
function lookAt(eye: number[], tgt: number[]): number[] {
  const f = [tgt[0] - eye[0], tgt[1] - eye[1], tgt[2] - eye[2]];
  const n = Math.hypot(...f); for (let i = 0; i < 3; i++) f[i] /= n;
  const up = [0, 1, 0];
  const r = [f[1] * up[2] - f[2] * up[1], f[2] * up[0] - f[0] * up[2], f[0] * up[1] - f[1] * up[0]];
  const rn = Math.hypot(...r); for (let i = 0; i < 3; i++) r[i] /= rn;
  const u = [r[1] * f[2] - r[2] * f[1], r[2] * f[0] - r[0] * f[2], r[0] * f[1] - r[1] * f[0]];
  return [r[0], r[1], r[2], 0, u[0], u[1], u[2], 0, -f[0], -f[1], -f[2], 0, eye[0], eye[1], eye[2], 1];
}

export async function exportPerfScene(o: ExportPerfSceneOptions): Promise<{ files: string[]; bytes: number; sha256: string; triangles: number; lights: number; camera: number[] }> {
  let scene: SceneData = (await loadScene(o.scene)).scene;
  const b = scene.bounds, e = [0, 1, 2].map((i) => b.max[i] - b.min[i]), c = [0, 1, 2].map((i) => 0.5 * (b.min[i] + b.max[i]));
  const long = e[0] >= e[2] ? 0 : 2;
  const eye = [...c], tgt = [...c];
  eye[1] = tgt[1] = b.min[1] + 0.2 * e[1];
  eye[long] = c[long] + 0.35 * e[long]; tgt[long] = c[long] - 0.35 * e[long];
  const camToWorld = lookAt(eye, tgt);
  const store = ensureLightStore(scene);
  store.add({ type: 'point', power: 2000, color: [1, 0.85, 0.7], matrix: new Float32Array([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, c[0], b.min[1] + 0.25 * e[1], c[2], 1]) });
  scene = { ...scene, lights: store.list().map((l) => ({ ...l })) as LightData[] };
  if (o.env) scene = { ...scene, env: (await loadEnvironment(o.env, { mode: 'validation' })).env };
  const r = await exportAndUpload(scene, {
    camera: { matrix: camToWorld, yfov: (55 * Math.PI) / 180 }, render: { width: o.width, height: o.height, maxBounces: o.maxBounces },
    lightMode: o.lightMode, name: o.name ?? o.run, source: { uri: `${o.scene}${o.env ? ` + ${o.env}` : ''} (perf-run.ts autoSetup; validation/harness/perf-scene-export.ts)` },
  }, o.run);
  return { files: r.files, bytes: r.bytes, sha256: r.sha256, triangles: scene.geometry.indices.length / 3, lights: scene.lights.length, camera: camToWorld };
}
