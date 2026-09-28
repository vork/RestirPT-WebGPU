// Animation (src/core/scene/animation.ts): channel interpolation (lerp, step, slerp along the shortest path),
// presets, the validation clock (src/app/timeline/player.ts), and the frames[] export with CONSTANT semantics in the
// scene-bridge.md format (round-tripped through exportScenePackage / readScenePackage).
import { readFileSync } from 'node:fs';
import { describe, expect, it } from 'vitest';
import {
  Animation, AnimationError, FrameExportError, allFrames, evalChannel, exportFrames, frameTime, lightTarget, matrixToPoseQ, poseToMatrix,
  presetBob, presetOrbit, presetSweep, quatAxisAngle, quatRotate, slerpShortest, type BaseState, type Quat,
} from '../../src/core/scene/animation.ts';
import { isRigid, exportScenePackage, readScenePackage } from '../../src/core/scene/scene-package.ts';
import { loadGltf } from '../../src/core/scene/gltf-loader.ts';
import type { LightData } from '../../src/core/scene/types.ts';
import { TimelinePlayer } from '../../src/app/timeline/player.ts';

const close = (a: ArrayLike<number>, b: ArrayLike<number>, eps = 1e-12) => { for (let i = 0; i < b.length; i++) expect(Math.abs(a[i] - b[i]), `index ${i}: ${a[i]} vs ${b[i]}`).toBeLessThan(eps); };
const T = (x: number, y: number, z: number) => new Float32Array([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, x, y, z, 1]);
const L = (o: Partial<LightData>): LightData => ({ id: 0, name: 'l', type: 'point', color: [1, 1, 1], power: 10, exposure: 0, matrix: T(0, 1, 0), visibleToCamera: false, ...o });
const angleBetween = (a: readonly number[], b: readonly number[]) => 2 * Math.acos(Math.min(1, Math.abs(a[0] * b[0] + a[1] * b[1] + a[2] * b[2] + a[3] * b[3])));

describe('channel interpolation', () => {
  it('linear, step (left-closed hold) and clamping outside the keys', () => {
    const keys = [{ t: 1, v: [0], interp: 'linear' as const }, { t: 3, v: [10], interp: 'step' as const }, { t: 4, v: [20], interp: 'linear' as const }];
    expect(evalChannel(keys, 'power', 0)).toEqual([0]);
    expect(evalChannel(keys, 'power', 2)).toEqual([5]);
    expect(evalChannel(keys, 'power', 2.999)[0]).toBeCloseTo(9.995, 10);
    expect(evalChannel(keys, 'power', 3)).toEqual([10]);
    expect(evalChannel(keys, 'power', 3.999)).toEqual([10]); // step holds until the next key...
    expect(evalChannel(keys, 'power', 4)).toEqual([20]);     // ...which applies exactly at its time
    expect(evalChannel(keys, 'power', 99)).toEqual([20]);
  });

  it('slerp takes the shortest path (q and −q are the same rotation) and stays unit length', () => {
    const a = quatAxisAngle([0, 1, 0], 0.1);
    const b = quatAxisAngle([0, 1, 0], 0.5);
    const negB = b.map((x) => -x);
    for (const bb of [b, negB]) {
      const m = slerpShortest(a, bb, 0.5);
      expect(Math.hypot(...m)).toBeCloseTo(1, 14);
      expect(angleBetween(m, quatAxisAngle([0, 1, 0], 0.3))).toBeLessThan(1e-7);
    }
    // 170° apart through the short side: halfway = 85°, not 180 + 85
    const c = quatAxisAngle([0, 0, 1], 170 * Math.PI / 180);
    const h = slerpShortest([0, 0, 0, 1], c.map((x) => -x), 0.5);
    expect(angleBetween(h, quatAxisAngle([0, 0, 1], 85 * Math.PI / 180))).toBeLessThan(1e-7);
    // constant angular velocity
    const q = [0.25, 0.5, 0.75].map((t) => slerpShortest(a, b, t));
    expect(angleBetween(a, q[0])).toBeCloseTo(0.1, 9);
    expect(angleBetween(a, q[2])).toBeCloseTo(0.3, 9);
    // nearly identical quaternions (lerp branch) stay normalized
    const n = slerpShortest(a, quatAxisAngle([0, 1, 0], 0.1001), 0.3);
    expect(Math.hypot(...n)).toBeCloseTo(1, 14);
  });

  it('pose ↔ matrix round trip and rigid output', () => {
    const q = slerpShortest(quatAxisAngle([1, 2, 3], 1.1), quatAxisAngle([-1, 0, 2], 2.2), 0.37);
    const m = poseToMatrix([1, -2, 3], q);
    expect(isRigid(m, 1e-12)).toBe(true);
    const p = matrixToPoseQ(m);
    close(p.position, [1, -2, 3]);
    expect(angleBetween(p.quaternion, q)).toBeLessThan(1e-7);
  });
});

describe('Animation tracks', () => {
  it('evaluates camera, light and env channels; unanimated channels keep the base values bit-exactly', () => {
    const a = new Animation();
    const id = 5;
    const base: BaseState = {
      camera: { position: [0, 1, 5], quaternion: [0, 0, 0, 1], yfov: 0.7 },
      lights: [L({ id, matrix: new Float32Array([0, 0, -1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 2, 3, 1]), power: 7, color: [1, 0.5, 0.25] })],
      env: { rotationZ: 0.1, strength: 2 },
    };
    a.setKey(lightTarget(id), 'position', 0, [0, 0, 0]);
    a.setKey(lightTarget(id), 'position', 2, [2, 4, 6]);
    a.setKey(lightTarget(id), 'power', 0, [1]);
    a.setKey(lightTarget(id), 'power', 2, [3], 'step');
    a.setKey('camera', 'yfov', 0, [0.5]);
    a.setKey('camera', 'yfov', 1, [0.9]);
    a.setKey('env', 'rotationZ', 0, [0]);
    a.setKey('env', 'rotationZ', 4, [Math.PI]);
    const s = a.evaluate(1, base);
    const l = s.lights.get(id)!;
    close([l.matrix[12], l.matrix[13], l.matrix[14]], [1, 2, 3], 1e-7);
    for (let i = 0; i < 12; i++) expect(l.matrix[i]).toBe(base.lights[0].matrix[i]); // rotation untouched
    expect(l.power).toBe(2);
    expect(l.color).toEqual([1, 0.5, 0.25]);
    expect(l.animated.sort()).toEqual(['position', 'power']);
    expect(s.camera!.yfov).toBeCloseTo(0.9, 14);
    close(s.camera!.position, [0, 1, 5]);
    expect(s.env).toEqual({ rotationZ: Math.PI / 4, strength: 2 });
    expect(a.duration).toBe(10); // keys inside the default duration do not shrink it
    a.setKey('camera', 'yfov', 12, [0.9]);
    expect(a.duration).toBe(12);
    expect(a.keyTimes()).toEqual([0, 1, 2, 4, 12]);
    expect(a.keyTimes('camera')).toEqual([0, 1, 12]);
  });

  it('setKey replaces keys at the same time, validates values; removeKey drops empty tracks', () => {
    const a = new Animation();
    a.setKey('camera', 'yfov', 1, [0.5]);
    a.setKey('camera', 'yfov', 1 + 1e-12, [0.6]);
    expect(a.track('camera')!.channels.yfov).toHaveLength(1);
    expect(a.track('camera')!.channels.yfov![0].v).toEqual([0.6]);
    expect(() => a.setKey('camera', 'power', 0, [1])).toThrow(AnimationError);
    expect(() => a.setKey(lightTarget(1), 'power', 0, [-1])).toThrow(AnimationError);
    expect(() => a.setKey(lightTarget(1), 'position', 0, [1, 2])).toThrow(AnimationError);
    expect(() => a.setKey(lightTarget(1), 'position', -1, [1, 2, 3])).toThrow(AnimationError);
    expect(a.removeKey('camera', 1)).toBe(1);
    expect(a.isEmpty).toBe(true);
    expect(a.targets).toEqual([]);
  });

  it('presets: orbit keeps radius/height and aims at the centre; bob; sweep rotates about world Y', () => {
    const o = presetOrbit({ start: [3, 1, 0], center: [0, 0, 0], period: 8, turns: 1, aim: true, samplesPerTurn: 8 });
    expect(o.position).toHaveLength(9);
    for (const k of o.position!) { expect(Math.hypot(k.v[0], k.v[2])).toBeCloseTo(3, 12); expect(k.v[1]).toBe(1); }
    close(o.position![8].v, [3, 1, 0], 1e-12);
    expect(o.position![8].t).toBe(8);
    for (const [i, k] of o.rotation!.entries()) {
      const p = o.position![i].v;
      const fwd = quatRotate(k.v, [0, 0, -1]);
      const to = [-p[0], -p[1], -p[2]];
      const n = Math.hypot(...to);
      close(fwd, to.map((x) => x / n), 1e-12);
    }
    const b = presetBob({ start: [1, 2, 3], amplitude: 0.5, period: 2, samplesPerPeriod: 4 });
    expect(b.position!.map((k) => Math.round(k.v[1] * 1e9) / 1e9)).toEqual([2, 2.5, 2, 1.5, 2]);
    const q0: Quat = [0, 0, 0, 1];
    const sw = presetSweep({ start: q0, angle: Math.PI / 6, period: 4, samplesPerPeriod: 4 });
    expect(angleBetween(sw.rotation![1].v, quatAxisAngle([0, 1, 0], Math.PI / 6))).toBeLessThan(1e-9);
    expect(angleBetween(sw.rotation![3].v, quatAxisAngle([0, 1, 0], -Math.PI / 6))).toBeLessThan(1e-9);
  });

  it('JSON round trip is exact', () => {
    const a = new Animation();
    a.setSettings({ fps: 30, duration: 5, loop: false });
    a.setKey(lightTarget(65536 + 3), 'rotation', 0.1, quatAxisAngle([1, 1, 0], 0.3));
    a.setKey(lightTarget(65536 + 3), 'color', 1 / 3, [0.1, 0.2, 0.30000000000000004], 'step');
    a.setKey('env', 'strength', 2, [Math.PI]);
    const j = JSON.parse(JSON.stringify(a.toJSON()));
    const b = Animation.fromJSON(j);
    expect(JSON.stringify(b.toJSON())).toBe(JSON.stringify(a.toJSON()));
    expect(b.fps).toBe(30);
    expect(b.loop).toBe(false);
  });
});

describe('timeline clock', () => {
  it('validation mode steps exactly one frame per advanced app frame; t = k / fps', () => {
    const a = new Animation();
    a.setSettings({ fps: 24, duration: 1, loop: true });
    const p = new TimelinePlayer(a);
    p.setMode('validation');
    p.play();
    expect(p.tick(0.5, false)).toBe(false); // paused app frame: no advance
    for (let i = 0; i < 5; i++) p.tick(123, true); // wall-clock dt is ignored
    expect(p.frame).toBe(5);
    expect(p.time).toBe(frameTime(5, 24));
    for (let i = 0; i < 20; i++) p.tick(0, true);
    expect(p.frame).toBe(0); // 24 frames + frame 0..24 inclusive → wraps after frame 24
    p.seek(0.51);
    expect(p.frame).toBe(12);
    p.setMode('interactive');
    expect(p.time).toBe(0.5);
    p.tick(0.25, true);
    expect(p.time).toBeCloseTo(0.75, 12);
    p.tick(0.5, true);
    expect(p.time).toBeCloseTo(0.25, 12); // looped
  });
});

describe('frames export (scene-bridge.md, CONSTANT semantics)', () => {
  const base = (): BaseState => ({
    camera: { position: [0, 1, 5], quaternion: [0, 0, 0, 1], yfov: 0.6 },
    lights: [L({ id: 2, power: 5 }), L({ id: 70000, type: 'spot', spotSize: 1, spotBlend: 0.2, matrix: T(1, 2, 3), power: 9 })],
    env: { rotationZ: 0.2, strength: 1.5 },
  });

  it('frame k holds the state at t = k / fps; step keys switch exactly at their frame', () => {
    const a = new Animation();
    a.setSettings({ fps: 24, duration: 2 });
    a.setKey(lightTarget(2), 'power', 0, [5], 'step');
    a.setKey(lightTarget(2), 'power', 1, [50], 'step');
    a.setKey(lightTarget(70000), 'position', 0, [1, 2, 3]);
    a.setKey(lightTarget(70000), 'position', 2, [3, 2, 3]);
    a.setKey(lightTarget(70000), 'rotation', 0, [0, 0, 0, 1]);
    a.setKey(lightTarget(70000), 'rotation', 2, quatAxisAngle([0, 1, 0], 1));
    a.setKey('camera', 'position', 0, [0, 1, 5]);
    a.setKey('camera', 'position', 2, [0, 1, 3]);
    a.setKey('env', 'rotationZ', 0, [0]);
    a.setKey('env', 'rotationZ', 2, [1]);
    const frames = exportFrames(a, { base: base() });
    expect(frames.map((f) => f.frame)).toEqual(allFrames(a));
    expect(frames).toHaveLength(49);
    expect(frames[23].lights!['2'].power).toBe(5);
    expect(frames[24].lights!['2'].power).toBe(50);
    for (const f of frames) {
      const t = f.frame / 24;
      // exact equality with an independent evaluation at t = k / fps (the renderer's validation clock)
      const s = a.evaluate(t, base());
      expect(f.lights!['70000'].matrix).toEqual(Array.from(s.lights.get(70000)!.matrix));
      expect(f.camera!.matrix).toEqual(Array.from(s.camera!.matrix));
      expect(f.env).toEqual({ rotationZ: s.env!.rotationZ, strength: 1.5 });
      // contract shape: string ids, 16-number column-major rigid matrices, f32-exact light matrices
      expect(Object.keys(f.lights!).sort()).toEqual(['2', '70000']);
      for (const l of Object.values(f.lights!)) {
        expect(l.matrix).toHaveLength(16);
        expect(Array.from(Float32Array.from(l.matrix!))).toEqual(l.matrix);
        expect(isRigid(l.matrix!)).toBe(true);
      }
      expect(isRigid(f.camera!.matrix)).toBe(true);
    }
    close(Array.from(frames[24].lights!['70000'].matrix!).slice(12, 15), [2, 2, 3], 1e-6);
    // selected frames only
    expect(exportFrames(a, { base: base(), frames: [3, 7] }).map((f) => f.frame)).toEqual([3, 7]);
    expect(() => exportFrames(a, { base: base(), frames: [3, 3] })).toThrow(FrameExportError);
  });

  it('refuses what the bridge cannot represent: animated colour, unknown lights, env tracks without env', () => {
    const a = new Animation();
    a.setKey(lightTarget(2), 'color', 0, [1, 1, 1]);
    a.setKey(lightTarget(2), 'color', 1, [1, 0, 0]);
    expect(() => exportFrames(a, { base: base(), frames: [12] })).toThrow(/colour/);
    expect(exportFrames(a, { base: base(), frames: [0] })[0].lights!['2']).toBeDefined(); // unchanged colour at frame 0 is fine
    const b = new Animation();
    b.setKey(lightTarget(99), 'power', 0, [1]);
    expect(() => exportFrames(b, { base: base() })).toThrow(/unknown light/);
    const c = new Animation();
    c.setKey('env', 'strength', 0, [1]);
    expect(() => exportFrames(c, { base: { ...base(), env: undefined } })).toThrow(/environment/);
  });

  it('frames survive exportScenePackage / readScenePackage exactly (the Blender bridge input)', async () => {
    const bytes = new Uint8Array(readFileSync('validation/assets/cornell/cornell.glb'));
    const scene = (await loadGltf({ kind: 'glb', bytes, name: 'cornell.glb' }, { tangents: false })).scene;
    const bl = base();
    scene.lights = [...bl.lights];
    const a = new Animation();
    a.setSettings({ fps: 10, duration: 1 });
    a.setKey(lightTarget(2), 'position', 0, [0, 0.3, 0]);
    a.setKey(lightTarget(2), 'position', 1, [0.1, 0.4, -0.2]);
    a.setKey('camera', 'yfov', 0, [0.6]);
    a.setKey('camera', 'yfov', 1, [0.8]);
    const frames = exportFrames(a, { base: { ...bl, env: undefined } });
    const pkg = await exportScenePackage(scene, {
      camera: { matrix: poseToMatrix([0, 1, 5], [0, 0, 0, 1]), yfov: 0.6 }, render: { width: 64, height: 64, maxBounces: 3 }, lightMode: 'A', frames,
    });
    const back = await readScenePackage(pkg.files);
    expect(back.frames).toHaveLength(11);
    for (const [i, f] of frames.entries()) {
      const g = back.frames![i];
      expect(g.frame).toBe(f.frame);
      expect(g.camera!.matrix).toEqual(f.camera!.matrix);
      expect(g.camera!.yfov).toBe(f.camera!.yfov);
      expect(g.lights!['2'].matrix).toEqual(f.lights!['2'].matrix);
      expect(g.lights!['2'].power).toBe(5);
    }
  });
});
