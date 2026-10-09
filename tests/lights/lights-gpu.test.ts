// Light packing (lights-gpu.ts, emissive-tris.ts; math.md#units-lights, #light-selection): Cycles-exact emission per
// type, spot/spread parameters and guards, power proxies, emissive-triangle entries, record layout, cur/prev slots,
// stable-id maps, deterministic alias rebuild only on power/set changes (pmf_t ≡ pmf_{t−1} bitwise otherwise).
import { describe, expect, it } from 'vitest';
import { collectEmissiveTriangles, meanTextureRgb, NO_ENTRY, srgbToLinear } from '../../src/core/render/emissive-tris.ts';
import {
  LIGHT_REC, LIGHT_REC_WORDS, LightsState, LT, LF_DELTA, LF_VISIBLE_CAMERA, lightEmission, lightPowerProxy, packLightRecord, spotParams,
  spotProfile,
} from '../../src/core/render/lights-gpu.ts';
import { LUT_LAYOUT } from '../../src/core/render/luts/lut-layout.ts';
import type { LightData, MaterialData, SceneData } from '../../src/core/scene/types.ts';
import { TRI_EMISSIVE } from '../../src/core/scene/types.ts';

const M = (p: number[]): Float32Array => new Float32Array([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, ...p, 1]);

function L(o: Partial<LightData> & Pick<LightData, 'type' | 'id'>): LightData {
  return { name: 'l', color: [1, 0.5, 0.25], power: 10, exposure: 1, matrix: M([1, 2, 3]), visibleToCamera: false, ...o };
}

function mat(o: Partial<MaterialData> = {}): MaterialData {
  return {
    name: 'm', baseColorFactor: [1, 1, 1, 1], metallicFactor: 0, roughnessFactor: 1, emissiveFactor: [0, 0, 0], emissiveStrength: 1, ior: 1.5,
    specularFactor: 1, specularColorFactor: [1, 1, 1], transmissionFactor: 0, alphaMode: 'OPAQUE', alphaCutoff: 0.5, doubleSided: true, model: 'v1',
    v1: { diffuse: [0.5, 0.5, 0.5], glossy: [0, 0, 0], roughness: 0.5, mix: 0 }, ...o,
  };
}

/** Two triangles: prim 0 plain (area 0.5), prim 1 emissive (area 2), plus prim 2 emissive with emission_sampling NONE. */
function scene(lights: LightData[] = []): SceneData {
  const positions = Float32Array.from([0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 2, 0, 1, 0, 2, 1, 5, 5, 5, 6, 5, 5, 5, 6, 5]);
  return {
    name: 's',
    geometry: {
      positions, normals: new Float32Array(27), tangents: new Float32Array(36), uv0: new Float32Array(18),
      indices: Uint32Array.from([0, 1, 2, 3, 4, 5, 6, 7, 8]), triMaterial: Uint32Array.from([0, 1, 2]), triFlags: Uint32Array.from([0, TRI_EMISSIVE, TRI_EMISSIVE]),
    },
    materials: [mat(), mat({ emissiveFactor: [2, 1, 0.5], emissiveStrength: 3 }), { ...mat({ emissiveFactor: [1, 1, 1] }), emissionSampling: 'NONE' } as MaterialData],
    textures: [], lights, cameras: [], bounds: { min: [0, 0, 0], max: [6, 6, 5] }, warnings: [],
  };
}

describe('light emission (math.md#units-lights)', () => {
  it('point/spot I = Φ/(4π), area L = Φ/(πA) (disk A = π/4·d²), sun E = Φ; Φ = color·P·2^exposure', () => {
    const P = 10 * 2;
    expect(lightEmission(L({ id: 0, type: 'point' })).emit[0]).toBeCloseTo(P / (4 * Math.PI), 12);
    expect(lightEmission(L({ id: 0, type: 'spot' })).emit[1]).toBeCloseTo(0.5 * P / (4 * Math.PI), 12);
    const r = lightEmission(L({ id: 0, type: 'rect', sizeX: 2, sizeY: 0.5 }));
    expect(r.area).toBe(1);
    expect(r.emit[0]).toBeCloseTo(P / Math.PI, 12);
    const d = lightEmission(L({ id: 0, type: 'disk', sizeX: 2 }));
    expect(d.area).toBeCloseTo(Math.PI, 12);
    expect(d.emit[2]).toBeCloseTo(0.25 * P / (Math.PI * Math.PI), 12);
    expect(lightEmission(L({ id: 0, type: 'sun' })).emit).toEqual([P, 0.5 * P, 0.25 * P]);
  });

  it('spot profile: cos-domain smoothstep, blend = 0 hard step guard, not cone-normalised', () => {
    const l = L({ id: 0, type: 'spot', spotSize: Math.PI / 3, spotBlend: 0.2 });
    const { cosHalf, smooth } = spotParams(l);
    expect(cosHalf).toBeCloseTo(Math.cos(Math.PI / 6), 14);
    expect(smooth).toBeCloseTo(1 / ((1 - cosHalf) * 0.2), 10);
    expect(spotProfile(1, cosHalf, smooth)).toBe(1);
    expect(spotProfile(cosHalf, cosHalf, smooth)).toBe(0);
    const mid = cosHalf + 0.5 * (1 - cosHalf) * 0.2;
    expect(spotProfile(mid, cosHalf, smooth)).toBeCloseTo(0.5, 10);
    const hard = spotParams(L({ id: 0, type: 'spot', spotSize: Math.PI / 3, spotBlend: 0 }));
    expect(hard.smooth).toBe(-1);
    expect(spotProfile(cosHalf + 1e-7, hard.cosHalf, hard.smooth)).toBe(1);
    expect(spotProfile(cosHalf - 1e-7, hard.cosHalf, hard.smooth)).toBe(0);
  });

  it('power proxies (variance-only): point = lum Φ, rect = lum(L_e)πA = lum Φ, sun ∝ R_s²', () => {
    const lumPhi = 20 * (0.2126 + 0.7152 * 0.5 + 0.0722 * 0.25);
    expect(lightPowerProxy(L({ id: 0, type: 'point' }), 1)).toBeCloseTo(lumPhi, 10);
    expect(lightPowerProxy(L({ id: 0, type: 'rect', sizeX: 3, sizeY: 2 }), 1)).toBeCloseTo(lumPhi, 10);
    expect(lightPowerProxy(L({ id: 0, type: 'sun' }), 2)).toBeCloseTo(4 * lightPowerProxy(L({ id: 0, type: 'sun' }), 1), 10);
    const spot = lightPowerProxy(L({ id: 0, type: 'spot', spotSize: Math.PI, spotBlend: 0 }), 1); // hemisphere cone
    expect(spot).toBeCloseTo(lumPhi / 2, 10);
  });

  it('record layout: frame axes, a_L = −Z_obj, recentring, flags, spread normalisation', () => {
    const buf = new ArrayBuffer(LIGHT_REC_WORDS * 4);
    const dv = new DataView(buf);
    const rot = new Float32Array([0, 1, 0, 0, -1, 0, 0, 0, 0, 0, 1, 0, 4, 5, 6, 1]); // X_obj = +Y, Y_obj = −X
    packLightRecord(L({ id: 42, type: 'rect', sizeX: 2, sizeY: 1, spread: Math.PI / 2, matrix: rot, visibleToCamera: true }), [1, 1, 1], dv, 0);
    const f = new Float32Array(buf), u = new Uint32Array(buf);
    expect(Array.from(f.subarray(LIGHT_REC.pos, LIGHT_REC.pos + 3))).toEqual([3, 4, 5]);
    expect(u[LIGHT_REC.type]).toBe(LT.rect);
    expect(Array.from(f.subarray(LIGHT_REC.axisU, LIGHT_REC.axisU + 3))).toEqual([0, 1, 0]);
    expect(f[LIGHT_REC.halfU]).toBe(1);
    expect(f[LIGHT_REC.halfV]).toBe(0.5);
    expect(Array.from(f.subarray(LIGHT_REC.normal, LIGHT_REC.normal + 3)).map((x) => x + 0)).toEqual([0, 0, -1]);
    expect(u[LIGHT_REC.flags]).toBe(LF_VISIBLE_CAMERA);
    expect(u[LIGHT_REC.stableId]).toBe(42);
    const a = Math.PI / 4;
    expect(f[LIGHT_REC.spreadNorm]).toBeCloseTo(1 / (Math.tan(a) - a), 5);
    expect(f[LIGHT_REC.tanHalfSpread]).toBeCloseTo(1, 6);
    expect(f[LIGHT_REC.invArea]).toBeCloseTo(0.5, 7);
    packLightRecord(L({ id: 1, type: 'sun', visibleToCamera: true }), [0, 0, 0], dv, 0);
    expect(u[LIGHT_REC.flags]).toBe(LF_DELTA); // only area lights are camera-visible
  });
});

describe('emissive triangles', () => {
  it('entries in primId order, area, avg radiance, power proxy, emission_sampling NONE excluded, primId map', () => {
    const t = collectEmissiveTriangles(scene());
    expect(Array.from(t.primIds)).toEqual([1]);
    // prim 1: v0 = (0,0,1), v1 = (2,0,1), v2 = (0,2,1): area 2
    expect(t.areas[0]).toBeCloseTo(2, 6);
    expect(Array.from(t.avgRadiance)).toEqual([6, 3, 1.5]);
    expect(t.power[0]).toBeCloseTo((0.2126 * 6 + 0.7152 * 3 + 0.0722 * 1.5) * 2 * Math.PI * 2, 5);
    expect(Array.from(t.primToEntry)).toEqual([NO_ENTRY, 0, NO_ENTRY]);
  });

  it('textured emission uses the decoded whole-texture mean (sRGB EOTF)', () => {
    expect(srgbToLinear(0)).toBe(0);
    expect(srgbToLinear(1)).toBeCloseTo(1, 12);
    expect(srgbToLinear(0.04)).toBeCloseTo(0.04 / 12.92, 12);
    const px = new Uint8Array([255, 0, 128, 255, 0, 0, 128, 255]);
    const m = meanTextureRgb({ name: 't', width: 2, height: 1, pixels: px, wrapS: 'repeat', wrapT: 'repeat', filter: 'linear' });
    expect(m[0]).toBeCloseTo(0.5, 12);
    expect(m[1]).toBe(0);
    expect(m[2]).toBeCloseTo(srgbToLinear(128 / 255), 12);
  });
});

describe('LightsState: records, slots, maps, deterministic pmf', () => {
  const lights = [L({ id: 7, type: 'point', power: 5 }), L({ id: 2, type: 'rect', sizeX: 1, sizeY: 1 }), L({ id: 65538, type: 'sun', power: 0.1 })];
  const make = () => { const sc = scene(lights); return new LightsState(sc, [0, 0, 0], sc.geometry.positions); };

  it('LUT block first, stable-id order, alias over analytic + emissive, realized pmf sums to 1', () => {
    const s = make();
    s.update(lights);
    expect(s.triOff).toBe(LUT_LAYOUT.floats);
    const slot = s.curSlot;
    expect(slot.nAnalytic).toBe(3);
    expect(slot.nEntries).toBe(4);
    const ids = [0, 1, 2].map((i) => s.records[slot.lightOff + i * LIGHT_REC_WORDS + LIGHT_REC.stableId]);
    expect(ids).toEqual([2, 7, 65538]);
    let sum = 0;
    for (let e = 0; e < slot.nEntries; e++) sum += s.pmf(e);
    expect(sum).toBeCloseTo(1, 6);
    expect(s.records[s.triOff]).toBe(1); // emissive entry 0 = prim 1
    expect(s.records[s.primMapOff + 1]).toBe(0);
    expect(s.records[s.primMapOff + 2]).toBe(NO_ENTRY);
    const p = new Uint32Array(s.paramsBytes());
    expect(p[0]).toBe(slot.lightOff);
    expect(p[12]).toBe(slot.lightOff); // first frame: prev = cur
  });

  it('rigid motion flips the slot, keeps the pmf bitwise, maps ids; a power change rebuilds; add/remove maps', () => {
    const s = make();
    s.update(lights);
    const pmf0 = Array.from({ length: 4 }, (_, e) => s.pmf(e));
    const slot0 = s.cur;
    const moved = lights.map((l) => (l.id === 7 ? { ...l, matrix: M([9, 9, 9]) } : l));
    const u1 = s.update(moved);
    expect(s.cur).toBe(slot0 ^ 1);
    expect(u1).toMatchObject({ lightsChanged: true, pmfChanged: false, reallocated: false });
    expect(Array.from({ length: 4 }, (_, e) => s.pmf(e))).toEqual(pmf0);
    const c2p = s.curSlot.curToPrevOff;
    expect([0, 1, 2].map((i) => s.records[c2p + i])).toEqual([0, 1, 2]);
    const same = s.update(moved);
    expect(same).toMatchObject({ lightsChanged: false, pmfChanged: false });
    const brighter = moved.map((l) => (l.id === 7 ? { ...l, power: 50 } : l));
    const u2 = s.update(brighter);
    expect(u2.pmfChanged).toBe(true);
    expect(s.pmf(1)).toBeGreaterThan(pmf0[1]);
    // remove id 2, add id 3: cur [3, 7, 65538] vs prev [2, 7, 65538]
    const next = [L({ id: 3, type: 'point' }), ...brighter.filter((l) => l.id !== 2)];
    const u3 = s.update(next);
    expect(u3).toMatchObject({ lightsChanged: true, pmfChanged: true });
    const sl = s.curSlot;
    expect([0, 1, 2].map((i) => s.records[sl.curToPrevOff + i])).toEqual([NO_ENTRY, 1, 2]);
    expect([0, 1, 2].map((i) => s.records[sl.prevToCurOff + i])).toEqual([NO_ENTRY, 1, 2]);
    const pp = new Uint32Array(s.paramsBytes());
    expect(pp[12]).not.toBe(pp[0]); // prev slot differs from cur
  });

  it('perf2 WP-4a (RS_LIGHT_REC4): records and alias pairs 16 B-aligned; rect/disk index list per slot in the uniform', () => {
    const s = make();
    const disk = L({ id: 9, type: 'disk', sizeX: 1 });
    const steps = [lights, [...lights, disk], lights.filter((l) => l.type !== 'rect')];
    for (const set of steps) {
      s.update(set);
      for (const sl of [s.curSlot, s.prevSlot]) {
        expect(sl.lightOff % 4).toBe(0);
        expect(sl.aliasOff % 4).toBe(0);
      }
      expect(s.totalWords % 4).toBe(0);
      const sl = s.curSlot;
      const kinds = Array.from({ length: sl.lightCount }, (_, i) => s.records[sl.lightOff + i * LIGHT_REC_WORDS + LIGHT_REC.type]);
      const want = kinds.flatMap((k, i) => (k === LT.rect || k === LT.disk ? [i] : []));
      expect(Array.from(s.records.subarray(sl.areaOff, sl.areaOff + sl.areaCount))).toEqual(want);
      expect(sl.areaOff).toBeGreaterThanOrEqual(sl.prevToCurOff + s.capLights);
      expect(sl.areaOff + s.capLights).toBeLessThanOrEqual(s.slotBase[s.cur] + s.slotWords);
      const p = new Uint32Array(s.paramsBytes());
      expect([p[10], p[11]]).toEqual([sl.areaOff, sl.areaCount]);
      expect([p[22], p[23]]).toEqual([s.prevSlot.areaOff, s.prevSlot.areaCount]);
    }
    expect(s.curSlot.areaCount).toBe(0);
  });

  it('identical inputs give bitwise-identical records (deterministic rebuild)', () => {
    const a = make(), b = make();
    a.update(lights); b.update([...lights].reverse());
    expect(a.records).toEqual(b.records);
  });

  it('grows capacity (reallocation) when many lights are added', () => {
    const s = make();
    s.update(lights);
    const many = Array.from({ length: 40 }, (_, i) => L({ id: 100 + i, type: 'point', power: 1 + i }));
    const u = s.update(many);
    expect(u.reallocated).toBe(true);
    expect(s.curSlot.nAnalytic).toBe(40);
    expect(s.curSlot.nEntries).toBe(41);
  });
});
