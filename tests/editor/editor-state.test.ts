// Editor scene.json (src/app/editor/editor-state.ts): lossless round trip of lights (+ id allocator), tracks, camera
// and env parameters (plan §5 M3a exit "scene.json round-trips losslessly"), and restoring it into a LightStore.
import { describe, expect, it } from 'vitest';
import { Animation, lightTarget, quatAxisAngle } from '../../src/core/scene/animation.ts';
import { LightStore } from '../../src/core/scene/light-store.ts';
import { EditorStateError, parseEditorState, serializeEditorState, type EditorState } from '../../src/app/editor/editor-state.ts';
import { frameFromEmission } from '../../src/app/editor/placement.ts';

function state(): { s: EditorState; store: LightStore } {
  const store = new LightStore();
  const a = store.add({ type: 'spot', name: 'key "spot" ✓', power: 123.456789, color: [1, 0.1 + 0.2, 1 / 3], exposure: -0.7, spotSize: 0.7, spotBlend: 0.15, matrix: frameFromEmission([0.3, -0.9, 0.1], [0.1 + 0.2, 1e-7, -2.5]) });
  const b = store.add({ type: 'rect', sizeX: 0.13, sizeY: 0.105, spread: 1.2345, visibleToCamera: true, matrix: frameFromEmission([0, -1, 0], [0, 0.55, 0]) });
  store.remove(a.id);
  const c = store.add({ type: 'disk', sizeX: Math.PI / 10 }); // reuses a's slot with generation 1
  store.add({ type: 'sun', power: 3.3 });
  const anim = new Animation();
  anim.setSettings({ fps: 30, duration: 7.5, loop: false });
  anim.setKey(lightTarget(b.id), 'position', 0, [0, 0.55, 0]);
  anim.setKey(lightTarget(b.id), 'position', 1 / 3, [0.1, 0.55 + 1e-12, 0.2], 'step');
  anim.setKey(lightTarget(c.id), 'rotation', 2, quatAxisAngle([1, 2, 3], 0.123456789));
  anim.setKey('camera', 'yfov', 0, [0.6981317007977318]);
  anim.setKey('env', 'strength', 4, [2.5]);
  const s: EditorState = {
    source: { name: 'cornell' },
    lights: [...store.list()],
    lightIds: store.idState(),
    camera: { position: [0.1 + 0.2, 1.0000000000000002, 5], quaternion: quatAxisAngle([0.2, 1, 0], 0.3), yfov: 0.6981317007977318 },
    env: { url: '/validation/assets/x.hdr', strength: 1.7, rotationZ: -0.5235987755982988, tint: [1, 0.9, 0.8], visibleToCamera: false },
    timeline: { time: 1.2345, mode: 'validation', frame: 37 },
    animation: anim.toJSON(),
  };
  return { s, store };
}

describe('editor scene.json', () => {
  it('round-trips losslessly (text and values, f32 matrices bit-exact)', () => {
    const { s } = state();
    const text = serializeEditorState(s);
    const back = parseEditorState(text);
    expect(serializeEditorState(back)).toBe(text);
    expect(back.lights.map((l) => l.id)).toEqual(s.lights.map((l) => l.id).sort((a, b) => a - b));
    for (const l of s.lights) {
      const m = back.lights.find((x) => x.id === l.id)!;
      expect(m.matrix).toBeInstanceOf(Float32Array);
      expect(new Uint32Array(m.matrix.buffer)).toEqual(new Uint32Array(Float32Array.from(l.matrix).buffer));
      expect({ ...m, matrix: undefined }).toEqual({ ...l, matrix: undefined });
    }
    expect(back.camera).toEqual(s.camera);
    expect(back.env).toEqual(s.env);
    expect(back.timeline).toEqual(s.timeline);
    expect(back.animation).toEqual(s.animation);
    expect(back.lightIds).toEqual(s.lightIds);
    expect(Animation.fromJSON(back.animation).toJSON()).toEqual(s.animation);
  });

  it('restores into a LightStore with the same ids and allocator state (no id reuse after load)', () => {
    const { s, store } = state();
    const back = parseEditorState(serializeEditorState(s));
    const fresh = new LightStore();
    fresh.add({ type: 'point' }); // replaced by the load
    fresh.replaceAll(back.lights, back.lightIds);
    expect(fresh.list().map((l) => l.id)).toEqual(store.list().map((l) => l.id));
    const n1 = fresh.add({ type: 'point' }).id, n2 = store.add({ type: 'point' }).id;
    expect(n1).toBe(n2); // the allocator continues exactly as the saved one would
    expect(s.lights.some((l) => l.id === n1)).toBe(false);
  });

  it('rejects malformed documents (non-f32 matrices, scaled matrices, duplicate ids, wrong format)', () => {
    const { s } = state();
    const j = JSON.parse(serializeEditorState(s));
    const bad1 = structuredClone(j); bad1.lights[0].matrix[12] = 0.1; // 0.1 is not an f32 value
    expect(() => parseEditorState(bad1)).toThrow(/not an f32/);
    const bad2 = structuredClone(j); bad2.lights[0].matrix[0] = 2; bad2.lights[0].matrix[5] = 2; bad2.lights[0].matrix[10] = 2;
    expect(() => parseEditorState(bad2)).toThrow(EditorStateError);
    const bad3 = structuredClone(j); bad3.lights[1].id = bad3.lights[0].id;
    expect(() => parseEditorState(bad3)).toThrow(/duplicate/);
    expect(() => parseEditorState({ ...j, format: 'restir-scene-package' })).toThrow(/not an editor scene/);
  });
});
