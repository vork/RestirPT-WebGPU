// LightStore (src/core/scene/light-store.ts): stable ids (free list + generation), validation (rigid matrices only),
// change events {moved, radiometric, added, removed, typeChanged} and batching, the scene.lights mirror, emissive
// meshes; UndoStack commands (src/app/editor/undo.ts) restore exact states and ids.
import { describe, expect, it } from 'vitest';
import { Animation, lightTarget } from '../../src/core/scene/animation.ts';
import {
  LightStore, LightValidationError, ensureLightStore, lightGeneration, lightSlot, lightStoreOf, makeLightId, type LightChangeBatch,
} from '../../src/core/scene/light-store.ts';
import { TRI_EMISSIVE, type LightData, type SceneData } from '../../src/core/scene/types.ts';
import {
  UndoStack, addLightCommand, changeTypeCommand, removeLightCommand, trackCommand, updateLightCommand,
} from '../../src/app/editor/undo.ts';

const T = (x: number, y: number, z: number) => new Float32Array([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, x, y, z, 1]);
const point = (o: Partial<LightData> = {}): LightData => ({ id: 0, name: 'p', type: 'point', color: [1, 1, 1], power: 10, exposure: 0, matrix: T(0, 1, 0), visibleToCamera: false, ...o });

function record(s: LightStore): LightChangeBatch[] {
  const out: LightChangeBatch[] = [];
  s.onChange((b) => out.push(b));
  return out;
}

function emptyScene(lights: LightData[] = []): SceneData {
  return {
    name: 't', geometry: {
      positions: new Float32Array([0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 1, 0, 1, 1]), normals: new Float32Array(18), tangents: new Float32Array(24),
      uv0: new Float32Array(12), indices: new Uint32Array([0, 1, 2, 3, 4, 5]), triMaterial: new Uint32Array([0, 1]), triFlags: new Uint32Array([0, TRI_EMISSIVE]),
    },
    materials: [
      { name: 'white', baseColorFactor: [1, 1, 1, 1], metallicFactor: 0, roughnessFactor: 1, emissiveFactor: [0, 0, 0], emissiveStrength: 1, ior: 1.5, specularFactor: 1, specularColorFactor: [1, 1, 1], transmissionFactor: 0, alphaMode: 'OPAQUE', alphaCutoff: 0.5, doubleSided: true, model: 'principled' },
      { name: 'lamp', baseColorFactor: [1, 1, 1, 1], metallicFactor: 0, roughnessFactor: 1, emissiveFactor: [1, 0.5, 0.25], emissiveStrength: 4, ior: 1.5, specularFactor: 1, specularColorFactor: [1, 1, 1], transmissionFactor: 0, alphaMode: 'OPAQUE', alphaCutoff: 0.5, doubleSided: true, model: 'principled' },
    ],
    textures: [], lights, cameras: [], bounds: { min: [0, 0, 0], max: [1, 1, 1] }, warnings: [],
  };
}

describe('LightStore ids', () => {
  it('encodes slot + generation and never reissues an id', () => {
    expect(makeLightId(5, 3)).toBe((3 << 16) | 5);
    expect(lightSlot(makeLightId(5, 3))).toBe(5);
    expect(lightGeneration(makeLightId(5, 3))).toBe(3);
    const s = new LightStore();
    const a = s.add({ type: 'point' }).id;
    const b = s.add({ type: 'spot' }).id;
    expect([a, b]).toEqual([0, 1]);
    s.remove(a);
    const c = s.add({ type: 'rect' }).id;
    expect(lightSlot(c)).toBe(0);           // slot recycled through the free list
    expect(lightGeneration(c)).toBe(1);     // ...with a new generation: a different id
    expect(c).not.toBe(a);
    const issued = new Set<number>([a, b, c]);
    for (let i = 0; i < 50; i++) {
      const id = s.add({ type: 'point' }).id;
      expect(issued.has(id)).toBe(false);
      issued.add(id);
      if (i % 3 === 0) s.remove(id);
    }
    expect(s.slotCapacity).toBeLessThan(40); // compact: removed slots are reused
    expect(s.list().map((l) => l.id)).toEqual([...s.list().map((l) => l.id)].sort((x, y) => x - y));
  });

  it('keeps file-light ids when valid and unique, reassigns the rest; mirrors into scene.lights', () => {
    const scene = emptyScene([point({ id: 3, name: 'a' }), point({ id: 3, name: 'dup' }), point({ id: 1 << 20, name: 'big' })]);
    const s = ensureLightStore(scene);
    expect(ensureLightStore(scene)).toBe(s);
    expect(lightStoreOf(scene)).toBe(s);
    const byName = Object.fromEntries(s.list().map((l) => [l.name, l.id]));
    expect(byName.a).toBe(3);
    expect(new Set(Object.values(byName)).size).toBe(3);
    expect(scene.lights.map((l) => l.id)).toEqual(s.list().map((l) => l.id));
    const id = s.add({ type: 'sun', power: 2 }).id;
    expect(scene.lights.some((l) => l.id === id)).toBe(true);
    s.remove(id);
    expect(scene.lights.some((l) => l.id === id)).toBe(false);
  });

  it('restore() re-inserts a removed light with its original id; a used slot is refused', () => {
    const s = new LightStore();
    const l = s.add({ type: 'spot', name: 'k', power: 5 });
    const snap = { ...l, matrix: Float32Array.from(l.matrix), color: [...l.color] as [number, number, number] };
    s.remove(l.id);
    expect(s.restore(snap).id).toBe(l.id);
    expect(() => s.restore(snap)).toThrow(LightValidationError);
    const next = s.add({ type: 'point' }).id;
    expect(next).not.toBe(l.id);
  });
});

describe('LightStore validation', () => {
  it('accepts rigid matrices only (no scale, shear, mirror)', () => {
    const s = new LightStore();
    const scaled = T(0, 0, 0); scaled[0] = 2;
    const mirror = T(0, 0, 0); mirror[0] = -1;
    const shear = T(0, 0, 0); shear[4] = 0.3;
    for (const m of [scaled, mirror, shear]) expect(() => s.add({ type: 'point', matrix: m })).toThrow(/scale\/shear\/mirror/);
    const id = s.add({ type: 'point' }).id;
    expect(() => s.update(id, { matrix: scaled })).toThrow(LightValidationError);
    expect(s.get(id)!.matrix[0]).toBe(1); // unchanged after a rejected update
    const c = Math.cos(0.3), sn = Math.sin(0.3);
    expect(() => s.update(id, { matrix: new Float32Array([c, sn, 0, 0, -sn, c, 0, 0, 0, 0, 1, 0, 1, 2, 3, 1]) })).not.toThrow();
  });

  it('checks type-specific fields and radiometry', () => {
    const s = new LightStore();
    expect(() => s.add({ type: 'point', power: -1 })).toThrow(/power/);
    expect(() => s.add({ type: 'point', color: [1, Number.NaN, 0] })).toThrow(/colour/);
    expect(() => s.add({ type: 'spot', spotSize: 4 })).toThrow(/spotSize/);
    expect(() => s.add({ type: 'spot', spotBlend: 1.5 })).toThrow(/spotBlend/);
    expect(() => s.add({ type: 'rect', sizeX: 0 })).toThrow(/sizeX/);
    expect(() => s.add({ type: 'rect', spread: 0 })).toThrow(/spread/);
    const pt = s.add({ type: 'point', visibleToCamera: true });
    expect(pt.visibleToCamera).toBe(false); // only area lights can be camera-visible
    expect(pt.spotSize).toBeUndefined();
    const d = s.add({ type: 'disk', sizeX: 0.5 });
    expect(d.sizeY).toBeUndefined();
    expect(() => s.update(d.id, { sizeY: 0.7 })).toThrow(/circular/);
    expect(() => s.update(d.id, { type: 'rect' } as never)).toThrow(/changeType/);
  });
});

describe('LightStore events', () => {
  it('classifies moved / radiometric, flags pmf changes, skips no-ops', () => {
    const s = new LightStore();
    const ev = record(s);
    const id = s.add({ type: 'spot' }).id;
    expect(ev.at(-1)!.events.map((e) => e.kind)).toEqual(['added']);
    expect(ev.at(-1)!.idsChanged && ev.at(-1)!.pmfChanged).toBe(true);
    s.update(id, { matrix: T(1, 2, 3) });
    let b = ev.at(-1)!;
    expect(b.events.map((e) => e.kind)).toEqual(['moved']);
    expect(b.pmfChanged).toBe(false);
    expect(b.idsChanged).toBe(false);
    expect(b.events[0].before!.matrix[12]).toBe(0);
    expect(b.events[0].light!.matrix[12]).toBe(1);
    s.update(id, { spotBlend: 0.3 });
    b = ev.at(-1)!;
    expect(b.events.map((e) => e.kind)).toEqual(['radiometric']);
    expect(b.pmfChanged).toBe(false); // shape only: the power alias table is unchanged
    s.update(id, { power: 42, matrix: T(0, 0, 0) });
    b = ev.at(-1)!;
    expect(b.events.map((e) => e.kind).sort()).toEqual(['moved', 'radiometric']);
    expect(b.pmfChanged).toBe(true);
    const n = ev.length;
    s.update(id, { power: 42 });
    expect(ev.length).toBe(n); // no-op edit: no event, no version bump
    const v = s.version;
    s.remove(id);
    expect(ev.at(-1)!.events[0]).toMatchObject({ kind: 'removed', id });
    expect(s.version).toBe(v + 1);
  });

  it('batches many edits into one notification (animation step)', () => {
    const s = new LightStore();
    const ids = [0, 1, 2].map(() => s.add({ type: 'point' }).id);
    const ev = record(s);
    s.batch('animation', () => { for (const [i, id] of ids.entries()) s.update(id, { matrix: T(i + 1, 0, 0), power: 1 + i }, 'animation'); });
    expect(ev.length).toBe(1);
    expect(ev[0].source).toBe('animation');
    expect(ev[0].events.length).toBe(ids.length * 2);
  });

  it('type change = remove + add with a NEW id, one typeChanged event; revertType restores the old id', () => {
    const s = new LightStore();
    const a = s.add({ type: 'rect', sizeX: 0.3, sizeY: 0.2, power: 7, name: 'panel', matrix: T(1, 1, 1) });
    const ev = record(s);
    const d = s.changeType(a.id, 'disk');
    expect(d.id).not.toBe(a.id);
    expect(s.has(a.id)).toBe(false);
    expect(d).toMatchObject({ type: 'disk', power: 7, name: 'panel', sizeX: 0.3 });
    expect(d.matrix[12]).toBe(1);
    expect(ev.length).toBe(1);
    expect(ev[0].events).toHaveLength(1);
    expect(ev[0].events[0]).toMatchObject({ kind: 'typeChanged', id: d.id, prevId: a.id });
    expect(ev[0].idsChanged).toBe(true);
    const back = s.revertType(d.id, a);
    expect(back.id).toBe(a.id);
    expect(back.type).toBe('rect');
    expect(s.size).toBe(1);
  });
});

describe('emissive meshes', () => {
  it('lists emissive materials as static entries', () => {
    const em = LightStore.emissiveMeshes(emptyScene());
    expect(em).toHaveLength(1);
    expect(em[0]).toMatchObject({ materialIndex: 1, name: 'lamp', triangles: 1 });
    expect(em[0].radiance).toEqual([4, 2, 1]);
    expect(em[0].area).toBeCloseTo(0.5, 6);
  });
});

describe('undo/redo commands', () => {
  it('add / update / remove / type change round-trip exactly, with the same ids on redo', () => {
    const s = new LightStore();
    const anim = new Animation();
    const u = new UndoStack();
    const add = addLightCommand(s, { type: 'spot', power: 3, name: 's' });
    u.push(add);
    const id = add.id()!;
    const snap = () => JSON.stringify(s.list().map((l) => ({ ...l, matrix: Array.from(l.matrix) })));
    const s0 = snap();
    u.push(updateLightCommand(s, id, { matrix: Float32Array.from(s.get(id)!.matrix) }, { matrix: T(4, 5, 6) }, 'move'));
    u.push(updateLightCommand(s, id, { power: 3 }, { power: 9 }));
    anim.setKey(lightTarget(id), 'power', 0, [3]);
    anim.setKey(lightTarget(id), 'power', 1, [9]);
    const s2 = snap();
    u.push(changeTypeCommand(s, anim, id, 'point'));
    const newId = s.list()[0].id;
    expect(newId).not.toBe(id);
    expect(anim.hasTrack(lightTarget(newId))).toBe(true); // the track follows the light
    expect(anim.hasTrack(lightTarget(id))).toBe(false);
    u.push(removeLightCommand(s, anim, newId));
    expect(s.size).toBe(0);
    expect(anim.isEmpty).toBe(true);
    u.undo(); // un-delete
    expect(s.list()[0].id).toBe(newId);
    expect(anim.hasTrack(lightTarget(newId))).toBe(true);
    u.undo(); // un-retype
    expect(snap()).toBe(s2);
    expect(anim.hasTrack(lightTarget(id))).toBe(true);
    u.undo(); u.undo();
    expect(snap()).toBe(s0);
    u.undo();
    expect(s.size).toBe(0);
    expect(u.canUndo).toBe(false);
    // redo everything: same ids as the first time
    u.redo();
    expect(s.list()[0].id).toBe(id);
    u.redo(); u.redo();
    expect(snap()).toBe(s2);
    u.redo();
    expect(s.list()[0].id).toBe(newId);
    u.redo();
    expect(s.size).toBe(0);
    expect(u.canRedo).toBe(false);
  });

  it('a new command clears the redo branch; track commands restore key sets', () => {
    const s = new LightStore();
    const anim = new Animation();
    const u = new UndoStack();
    u.push(addLightCommand(s, { type: 'point' }));
    u.undo();
    expect(u.canRedo).toBe(true);
    u.push(addLightCommand(s, { type: 'sun' }));
    expect(u.canRedo).toBe(false);
    u.push(trackCommand(anim, 'camera', 'key', () => anim.setKey('camera', 'yfov', 0.5, [0.7])));
    expect(anim.keyTimes('camera')).toEqual([0.5]);
    u.undo();
    expect(anim.isEmpty).toBe(true);
    u.redo();
    expect(anim.keyTimes('camera')).toEqual([0.5]);
  });
});
