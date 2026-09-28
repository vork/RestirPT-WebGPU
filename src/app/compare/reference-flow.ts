// "Export for Cycles" → reference → compare (plan §5 M3a, dev only): export the edited scene (LightStore lights,
// current camera at the internal resolution, env params, resolved animation frames) as a scene package through the
// harness upload middleware, POST /api/reference (validation/harness/reference-endpoint.ts) which runs headless
// Blender, stream its progress, then load the EXRs of the requested frame into the compare view.
// The app's frame loop is suspended while Blender renders (plan §7.5: never overlap GPU jobs).
import { allFrames, exportFrames, type Animation } from '../../core/scene/animation.ts';
import type { LightStore } from '../../core/scene/light-store.ts';
import type { PackageFrame } from '../../core/scene/scene-package.ts';
import type { App } from '../app.ts';
import { DEG } from '../camera-math.ts';
import type { TimelinePlayer } from '../timeline/player.ts';
import type { CompareView } from './compare-view.ts';
import { fetchExr } from './images.ts';

export interface ReferenceConfig {
  spp: number;
  /** render_reference.py seeds, e.g. "0..3". */
  seeds: string;
  maxBounces: number;
  lightMode: 'A' | 'B';
  /** 'current' = the timeline's current frame only (default), 'all' = every frame of the animation. */
  frames: 'current' | 'all';
  /** Output size; 0 = the app's internal resolution (needed for pixel-exact comparison). */
  width: number;
  height: number;
  device: 'GPU' | 'CPU';
}

export const defaultReferenceConfig = (): ReferenceConfig => ({ spp: 16, seeds: '0..1', maxBounces: 3, lightMode: 'A', frames: 'current', width: 0, height: 0, device: 'GPU' });

export interface ReferenceProgress { stage: 'export' | 'render' | 'load' | 'done' | 'error'; message: string; done?: number; total?: number }

export interface ReferenceResult {
  packageDir: string;
  refDir: string;
  cacheHit: boolean;
  frame: number;
  exrs: { frame: number; seed: number; path: string; url: string }[];
}

interface Deps { app: App; store: LightStore; anim: Animation; player: TimelinePlayer; compare: CompareView }

type StreamMsg =
  | { type: 'start'; total: number; lockHeld: boolean }
  | { type: 'lock'; message: string }
  | { type: 'progress'; done: number; total: number; file: string; seconds: number }
  | { type: 'log'; line: string }
  | { type: 'result'; ok: true; dir: string; cacheHit: boolean; exrs: ReferenceResult['exrs'] }
  | { type: 'error'; message: string; tail?: string[] };

/** Export the current editor state as a scene package; returns the package dir (validation/out/<run>). */
export async function exportPackage(d: Deps, cfg: ReferenceConfig, run: string): Promise<{ dir: string; frame: number; frames?: PackageFrame[] }> {
  const { app, store, anim, player } = d;
  const scene = app.scene;
  if (!scene) throw new Error('no scene loaded');
  const { exportAndUpload } = await import('../../../validation/harness/export-package.ts');
  const p = app.envParams;
  const env = app.env ? {
    ...app.env, strength: p.strength, rotationZ: p.rotationDeg * DEG, tint: [p.tint.r, p.tint.g, p.tint.b] as [number, number, number], visibleToCamera: p.visibleToCamera,
  } : undefined;
  const cam = app.camera;
  let frames: PackageFrame[] | undefined;
  let frame = 0;
  if (!anim.isEmpty) {
    // pin the timeline to an exact frame so our image and Cycles' frame resolve identically (t = k / fps)
    if (player.mode !== 'validation') player.setMode('validation');
    player.pause();
    frame = player.frame;
    const base = {
      camera: { position: [...cam.position] as [number, number, number], quaternion: [...cam.quaternion] as [number, number, number, number], yfov: cam.yfov },
      lights: store.list(),
      env: env ? { rotationZ: env.rotationZ, strength: env.strength } : undefined,
    };
    frames = exportFrames(anim, { base, frames: cfg.frames === 'all' ? allFrames(anim) : [frame] });
  }
  const w = cfg.width > 0 ? cfg.width : app.targets.width;
  const h = cfg.height > 0 ? cfg.height : app.targets.height;
  const res = await exportAndUpload({ ...scene, lights: [...store.list()], env }, {
    camera: { matrix: cam.camToWorld(), yfov: cam.yfov, znear: 1e-4 },
    render: { width: w, height: h, maxBounces: cfg.maxBounces },
    lightMode: cfg.lightMode,
    frames,
    source: { uri: scene.name },
  }, run);
  return { dir: res.dir, frame, frames };
}

/** Full flow. Progress is reported through `onProgress`; resolves with the loaded reference. */
export async function exportAndRenderReference(d: Deps, cfg: ReferenceConfig, onProgress: (p: ReferenceProgress) => void, signal?: AbortSignal): Promise<ReferenceResult> {
  const { app, compare } = d;
  onProgress({ stage: 'export', message: 'exporting scene package...' });
  const run = `ref-${new Date().toISOString().replace(/[-:]/g, '').replace(/\..*/, '').replace('T', '-')}-${Math.random().toString(36).slice(2, 6)}`;
  const pkg = await exportPackage(d, cfg, run);
  onProgress({ stage: 'render', message: `package ${pkg.dir}; starting Blender...` });
  let result: Extract<StreamMsg, { type: 'result' }> | undefined;
  let error: string | undefined;
  app.suspended = true; // the GPU belongs to Blender now
  try {
    const r = await fetch('/api/reference', {
      method: 'POST', signal, headers: { 'content-type': 'application/json' },
      body: JSON.stringify({ packageDir: pkg.dir, spp: cfg.spp, seeds: cfg.seeds, frames: cfg.frames === 'all' || !pkg.frames ? 'all' : [pkg.frame], maxBounces: cfg.maxBounces, device: cfg.device }),
    });
    if (!r.ok || !r.body) throw new Error(`/api/reference: HTTP ${r.status} ${await r.text()}`);
    const reader = r.body.pipeThrough(new TextDecoderStream()).getReader();
    let buf = '';
    for (;;) {
      const { value, done } = await reader.read();
      if (value) buf += value;
      let i;
      while ((i = buf.indexOf('\n')) >= 0) {
        const line = buf.slice(0, i).trim();
        buf = buf.slice(i + 1);
        if (!line) continue;
        const m = JSON.parse(line) as StreamMsg;
        if (m.type === 'start') onProgress({ stage: 'render', message: m.lockHeld ? 'waiting for the GPU lock...' : 'rendering...', done: 0, total: m.total });
        else if (m.type === 'lock') onProgress({ stage: 'render', message: m.message });
        else if (m.type === 'progress') onProgress({ stage: 'render', message: `${m.file} (${m.seconds.toFixed(2)} s)`, done: m.done, total: m.total });
        else if (m.type === 'result') result = m;
        else if (m.type === 'error') error = `${m.message}${m.tail ? `\n${m.tail.slice(-6).join('\n')}` : ''}`;
      }
      if (done) break;
    }
  } finally {
    app.suspended = false;
  }
  if (error || !result) throw new Error(error ?? 'no result from /api/reference');
  const exrs = result.exrs.filter((e) => e.frame === pkg.frame);
  onProgress({ stage: 'load', message: `${result.cacheHit ? 'cache hit' : 'rendered'}: loading ${exrs.length} EXR(s) of frame ${pkg.frame}...` });
  const imgs = await Promise.all(exrs.sort((a, b) => a.seed - b.seed).map((e) => fetchExr(e.url)));
  compare.setReference(imgs, `${result.dir.split('/').pop()} f${pkg.frame} ${cfg.spp} spp`);
  app.resetHistory();
  onProgress({ stage: 'done', message: `loaded ${imgs.length} seed(s) from ${result.dir}` });
  return { packageDir: pkg.dir, refDir: result.dir, cacheHit: result.cacheHit, frame: pkg.frame, exrs: result.exrs };
}
