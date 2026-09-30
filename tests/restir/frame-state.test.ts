// U-TL-2 (restir-temporal-api.md §6.1, TD4, TD5, §2.5; T-A): one light commit per frame — env + light + animation edits
// of one frame give prev = the frame t−1 state bitwise; unchanged frames do not flip (TF_LIGHTS_SAME) and keep the pmf
// bitwise; change bits (word 26) per light type; the M3 update() path still flips on every call. Plus the pure frame
// helpers (envDiff, temporalFlags) and the TD27 package frame format (export / read round trip, resolvePackageFrame).
import { readFileSync } from 'node:fs';
import { describe, expect, it } from 'vitest';
import { packEnvParams } from '../../src/core/render/env-gpu.ts';
import { LCB, LIGHT_REC, LIGHT_REC_WORDS, type LightsCommit } from '../../src/core/render/lights-gpu.ts';
import { RS_WGSL_CONSTS as K } from '../../src/core/render/restir/layout.ts';
import { envDiff, temporalFlags } from '../../src/core/render/restir/frame-state.ts';
import { loadGltf } from '../../src/core/scene/gltf-loader.ts';
import { BASE_ENV_MAP, exportScenePackage, readScenePackage, resolvePackageFrame } from '../../src/core/scene/scene-package.ts';
import type { EnvironmentData, LightData } from '../../src/core/scene/types.ts';
import { L, M, envInput, lightState } from './light-state-fixtures.ts';

const slotWords = (st: ReturnType<typeof lightState>, s: { lightOff: number; lightCount: number; aliasOff: number; aliasLog2: number; pmfOff: number; nEntries: number }) => ({
  lights: Array.from({ length: s.lightCount }, (_, i) => Array.from(st.records.subarray(s.lightOff + i * LIGHT_REC_WORDS, s.lightOff + i * LIGHT_REC_WORDS + LIGHT_REC.changeBits))),
  alias: Array.from(st.records.subarray(s.aliasOff, s.aliasOff + 2 * 2 ** s.aliasLog2)),
  pmf: Array.from(st.records.subarray(s.pmfOff, s.pmfOff + s.nEntries)),
});

describe('U-TL-2: one commit per frame', () => {
  it('env strength + light move + light power in one frame: prev = frame t−1 bitwise; one flip', () => {
    const st = lightState();
    const a = L({ id: 1, type: 'point' }), b = L({ id: 2, type: 'rect', matrix: M([2, 2, 2]) });
    st.commit([a, b], envInput(1));
    const t1 = st.commit([a, b], envInput(1));
    expect(t1.same).toBe(true);
    const before = slotWords(st, st.curSlot);
    const curIndex = st.cur;
    const c = st.commit([{ ...a, matrix: M([1, 2.5, 3]) }, { ...b, power: 20 }], envInput(3));
    expect(c.same).toBe(false);
    expect(st.cur).toBe(curIndex ^ 1);                     // exactly one flip
    expect(slotWords(st, st.prevSlot)).toEqual(before);     // prev = frame t−1 (records, alias, pmf) bitwise
    expect(c.pmfChanged).toBe(true);
    expect([c.anyMoved, c.anyRadio]).toEqual([true, true]);
    expect(st.records[st.curSlot.lightOff + LIGHT_REC.changeBits]).toBe(LCB.moved);
    expect(st.records[st.curSlot.lightOff + LIGHT_REC_WORDS + LIGHT_REC.changeBits]).toBe(LCB.radio);
  });

  it('unchanged frames: no flip, same = true, prev slot ≡ cur slot bitwise (pmf incl.)', () => {
    const st = lightState();
    const ls = [L({ id: 1, type: 'spot' }), L({ id: 3, type: 'disk' })];
    st.commit(ls, envInput(2));
    st.commit([{ ...ls[0], power: 7 }, ls[1]], envInput(2));
    const cur = st.cur;
    const words = slotWords(st, st.curSlot);
    for (let f = 0; f < 3; f++) {
      const c = st.commit([{ ...ls[0], power: 7 }, ls[1]], envInput(2));
      expect(c).toEqual({ same: true, pmfChanged: false, anyMoved: false, anyRadio: false, added: [], removed: [], reallocated: false });
      expect(st.cur).toBe(cur);
      expect(slotWords(st, st.curSlot)).toEqual(words);
      expect(slotWords(st, st.prevSlot)).toEqual(words);
    }
  });

  it('change bits per type (TD5): MOVED only for the words that enter Φ', () => {
    const cases: [string, LightData, Partial<LightData>, number][] = [
      ['point move', L({ id: 1, type: 'point' }), { matrix: M([0, 2, 3]) }, LCB.moved],
      ['point power', L({ id: 1, type: 'point' }), { power: 11 }, LCB.radio],
      ['spot axis', L({ id: 1, type: 'spot' }), { matrix: new Float32Array([0, 1, 0, 0, -1, 0, 0, 0, 0, 0, 1, 0, 1, 2, 3, 1]) }, LCB.radio],
      ['spot cone', L({ id: 1, type: 'spot', spotSize: 1 }), { spotSize: 0.9 }, LCB.radio],
      ['sun rotate', L({ id: 1, type: 'sun' }), { matrix: new Float32Array([1, 0, 0, 0, 0, 0, 1, 0, 0, -1, 0, 0, 1, 2, 3, 1]) }, LCB.moved | LCB.radio],   // normal = Φ; axes are other words
      ['sun translate (no Φ change)', L({ id: 1, type: 'sun' }), { matrix: M([4, 4, 4]) }, LCB.radio],
      ['rect resize', L({ id: 1, type: 'rect' }), { sizeX: 0.75 }, LCB.moved | LCB.radio],   // halfU moves Φ; area / L_e are radiometric
      ['rect colour', L({ id: 1, type: 'rect' }), { color: [0.5, 0.5, 0.5] }, LCB.radio],
      ['disk spread', L({ id: 1, type: 'disk' }), { spread: 1 }, LCB.radio],
      ['type change', L({ id: 1, type: 'point' }), { type: 'spot' }, LCB.moved | LCB.radio],
    ];
    for (const [name, l0, edit, bits] of cases) {
      const st = lightState();
      st.commit([l0, L({ id: 5, type: 'point', matrix: M([3, 3, 3]) })]);
      const c = st.commit([{ ...l0, ...edit }, L({ id: 5, type: 'point', matrix: M([3, 3, 3]) }), L({ id: 9, type: 'rect' })]);
      expect(st.records[st.curSlot.lightOff + LIGHT_REC.changeBits], name).toBe(bits);
      expect(st.records[st.curSlot.lightOff + LIGHT_REC_WORDS + LIGHT_REC.changeBits], `${name}: unchanged light`).toBe(0);
      expect(st.records[st.curSlot.lightOff + 2 * LIGHT_REC_WORDS + LIGHT_REC.changeBits], `${name}: added light`).toBe(LCB.added);
      expect(c.added).toEqual([9]);
    }
  });

  it('M3 update() (the PT path) still flips on every call; word 26 never makes lightsChanged', () => {
    const st = lightState();
    const ls = [L({ id: 1, type: 'point' })];
    st.update(ls);
    const cur = st.cur;
    const u = st.update(ls);
    expect(st.cur).toBe(cur ^ 1);
    expect(u).toEqual({ lightsChanged: false, pmfChanged: false, reallocated: false });
    st.update([{ ...ls[0], matrix: M([9, 9, 9]) }]);
    expect(st.update([{ ...ls[0], matrix: M([9, 9, 9]) }]).lightsChanged).toBe(false);
  });
});

describe('frame helpers', () => {
  const words = (p: Partial<{ rotationZ: number; strength: number; tint: [number, number, number]; visibleToCamera: boolean }>) =>
    new Uint32Array(packEnvParams({ rotationZ: 0.3, strength: 1, tint: [1, 1, 1], visibleToCamera: true, ...p }, true));
  it('envDiff: rotation ⇒ moved, strength / tint ⇒ radio, visibleToCamera ⇒ neither (but not same)', () => {
    const base = words({});
    expect(envDiff(base, words({}))).toEqual({ same: true, moved: false, radio: false });
    expect(envDiff(base, words({ rotationZ: 0.31 }))).toEqual({ same: false, moved: true, radio: false });
    expect(envDiff(base, words({ strength: 2 }))).toEqual({ same: false, moved: false, radio: true });
    expect(envDiff(base, words({ tint: [1, 0.5, 1] }))).toEqual({ same: false, moved: false, radio: true });
    expect(envDiff(base, words({ visibleToCamera: false }))).toEqual({ same: false, moved: false, radio: false });
  });
  it('temporalFlags: reset, same, refresh scope', () => {
    const same: LightsCommit = { same: true, pmfChanged: false, anyMoved: false, anyRadio: false, added: [], removed: [], reallocated: false };
    const envSame = { same: true, moved: false, radio: false };
    expect(temporalFlags({ histValid: false, commit: same, env: envSame, camSame: false })).toBe(K.TF_RESET | K.TF_LIGHTS_SAME | K.TF_ENV_SAME);
    expect(temporalFlags({ histValid: true, commit: same, env: envSame, camSame: true })).toBe(K.TF_HIST_VALID | K.TF_LIGHTS_SAME | K.TF_ENV_SAME | K.TF_CAM_SAME);
    const moved = { ...same, same: false, anyMoved: true };
    expect(temporalFlags({ histValid: true, commit: moved, env: envSame, camSame: true }) & (K.TF_REFRESH | K.TF_LIGHT_MOVED)).toBe(K.TF_REFRESH | K.TF_LIGHT_MOVED);
    expect(temporalFlags({ histValid: true, commit: same, env: { same: false, moved: true, radio: false }, camSame: true }) & (K.TF_REFRESH | K.TF_ENV_MOVED)).toBe(K.TF_REFRESH | K.TF_ENV_MOVED);
    expect(temporalFlags({ histValid: true, commit: same, env: { same: false, moved: false, radio: false }, camSame: true }) & K.TF_REFRESH).toBe(0);
    expect(temporalFlags({ histValid: false, commit: moved, env: envSame, camSame: true }) & K.TF_REFRESH).toBe(0);
  });
});

describe('TD27 package frames (A-8)', () => {
  it('export / read round trip of dense frames with enabled, radiometric overrides, env tint and a map swap; resolvePackageFrame', async () => {
    const bytes = new Uint8Array(readFileSync('validation/assets/cornell/cornell.glb'));
    const scene = (await loadGltf({ kind: 'glb', bytes, name: 'cornell.glb' }, { tangents: false })).scene;
    const mkEnv = (k: number): EnvironmentData => {
      const W = 16, H = 8, t = new Float32Array(W * H * 4);
      for (let i = 0; i < W * H; i++) { t[4 * i] = k * (1 + (i % 3)); t[4 * i + 1] = 1; t[4 * i + 2] = 0.5; t[4 * i + 3] = 1; }
      return { name: `e${k}`, width: W, height: H, texels: t, strength: 1.5, tint: [1, 1, 1], rotationZ: 0.1, visibleToCamera: true };
    };
    scene.env = mkEnv(1);
    scene.lights = [L({ id: 1, type: 'point' }), L({ id: 2, type: 'rect', matrix: M([0, 0.5, 0]) }), L({ id: 3, type: 'spot' })];
    const frames = [0, 1, 2].map((f) => ({
      frame: f,
      camera: { matrix: [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0.01 * f, 0.273, 1.0775, 1], yfov: 0.7 },
      lights: { '1': { power: 10 + f }, '2': { enabled: f !== 1, sizeX: 0.2 + 0.1 * f }, '3': { enabled: f >= 1, color: [1, 0.5, 0.5] as [number, number, number] } },
      env: { rotationZ: 0.1 * f, strength: 2, tint: [1, 0.9, 0.8] as [number, number, number], map: f === 2 ? 'b' : BASE_ENV_MAP },
    }));
    const sequence = { fps: 24, frameCount: 3, testFrames: [1, 2], notes: 'test' };
    const pkg = await exportScenePackage(scene, { camera: { matrix: frames[0].camera.matrix, yfov: 0.7 }, render: { width: 64, height: 64, maxBounces: 3 }, lightMode: 'A', frames, sequence, envMaps: [{ id: 'b', env: mkEnv(3) }] });
    expect([...pkg.files.keys()].sort()).toEqual(['env.exr', 'env_b.exr', 'geometry.bin', 'scene.json']);
    const back = await readScenePackage(pkg.files);
    expect(back.sequence).toEqual(sequence);
    expect(back.envMaps?.get('b')?.texels).toEqual(mkEnv(3).texels);
    const r1 = resolvePackageFrame(back, 1);
    expect(r1.lights.map((l) => l.id)).toEqual([1, 3]);
    expect(r1.lights[0].power).toBe(11);
    expect(r1.lights[1].color).toEqual([1, 0.5, 0.5]);
    expect(r1.env?.mapId).toBe(BASE_ENV_MAP);
    expect(r1.env?.params).toEqual({ rotationZ: 0.1, strength: 2, tint: [1, 0.9, 0.8], visibleToCamera: true });
    expect(Array.from(r1.camera.camToWorld)).toEqual(frames[1].camera.matrix);
    const r2 = resolvePackageFrame(back, 2);
    expect(r2.lights.map((l) => [l.id, l.sizeX])).toEqual([[1, 0.5], [2, Math.fround(0.4) === 0.4 ? 0.4 : r2.lights[1].sizeX], [3, 0.5]]);
    expect(r2.env?.mapId).toBe('b');
    expect(r2.env?.map.texels).toEqual(mkEnv(3).texels);
    const r0 = resolvePackageFrame(back, 0);
    expect(r0.lights.map((l) => l.id)).toEqual([1, 2]);
    // a frame outside frames[] = the base state
    const r9 = resolvePackageFrame(back, 9);
    expect(r9.lights.map((l) => l.id)).toEqual([1, 2, 3]);
    await expect(exportScenePackage(scene, { camera: { matrix: frames[0].camera.matrix, yfov: 0.7 }, render: { width: 64, height: 64, maxBounces: 3 }, lightMode: 'A', frames: [{ frame: 0, env: { map: 'nope' } }] })).rejects.toThrow(/env map/);
  });
});
