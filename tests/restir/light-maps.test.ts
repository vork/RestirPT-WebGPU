// U-TL-1 (CPU part, restir-temporal-api.md §6.1, §9.3-1; T-A): random add / remove / reorder / type-change sequences of
// the analytic lights (with emissive triangles and an env entry): curToPrev[prevToCur[j]] = j for survivors, NO_ENTRY
// exactly for added / removed lights; LightsState.translate (the CPU mirror of tframe.wgsl lt_translate) maps every
// entry kind across nA changes, and the round trip prev → cur → prev is the identity on survivors. The WGSL
// lt_translate ≡ CPU check is in restir-tframe.gpu.test.ts.
import { describe, expect, it } from 'vitest';
import { NO_ENTRY } from '../../src/core/render/emissive-tris.ts';
import type { LightData } from '../../src/core/scene/types.ts';
import { L, M, envInput, lightState, rng } from './light-state-fixtures.ts';

const TYPES: LightData['type'][] = ['point', 'spot', 'rect', 'disk', 'sun'];

describe('U-TL-1: id maps and entry translation under random light-set edits', () => {
  it('200 random frames × 4 seeds: maps are inverse on survivors, NONE for added / removed; every entry kind translates', () => {
    for (const seed of [1, 2, 3, 4]) {
      const r = rng(seed);
      const st = lightState();
      let nextId = 1;
      let lights: LightData[] = [];
      const env = seed % 2 === 0 ? envInput(2) : undefined;
      st.commit(lights, env);
      for (let f = 0; f < 200; f++) {
        const prevIds = st.slots[st.cur]!.ids.slice();
        // edit: remove some, add some, retype some (a type change keeps the id: MOVED | RADIO), move some, reorder
        lights = lights.filter(() => r() > 0.15);
        const nAdd = Math.floor(r() * 3);
        for (let i = 0; i < nAdd; i++) lights.push(L({ id: nextId++, type: TYPES[Math.floor(r() * TYPES.length)], power: 1 + 10 * r() }));
        lights = lights.map((l) => (r() < 0.1 ? { ...l, type: TYPES[Math.floor(r() * TYPES.length)] } : r() < 0.2 ? { ...l, matrix: M([r(), r(), r()]) } : l));
        lights.sort(() => r() - 0.5);
        const c = st.commit(lights, env);
        const cur = st.curSlot, prev = st.prevSlot;
        const curIds = st.slots[st.cur]!.ids;
        if (c.same) {
          expect(curIds).toEqual(prevIds);
          continue;
        }
        if (c.reallocated) continue;                       // capacity growth: a reset (config hash), maps restart
        expect(prev.nAnalytic).toBe(prevIds.length);
        const survivors = prevIds.filter((id) => curIds.includes(id));
        expect(c.removed.sort((a, b) => a - b)).toEqual(prevIds.filter((id) => !curIds.includes(id)).sort((a, b) => a - b));
        expect(c.added.sort((a, b) => a - b)).toEqual(curIds.filter((id) => !prevIds.includes(id)).sort((a, b) => a - b));
        for (let j = 0; j < prev.nAnalytic; j++) {
          const i = st.translate(j, 'prev', 'cur');
          if (survivors.includes(prevIds[j])) {
            expect(curIds[i]).toBe(prevIds[j]);
            expect(st.translate(i, 'cur', 'prev')).toBe(j);
          } else {
            expect(i).toBe(NO_ENTRY);
          }
        }
        for (let i = 0; i < cur.nAnalytic; i++) {
          const j = st.translate(i, 'cur', 'prev');
          if (!prevIds.includes(curIds[i])) expect(j).toBe(NO_ENTRY);
        }
        // triangles: offset by nA_to − nA_from; env: the target's env entry
        const nT = st.tris.primIds.length;
        for (let t = 0; t < nT; t++) {
          expect(st.translate(prev.nAnalytic + t, 'prev', 'cur')).toBe(cur.nAnalytic + t);
          expect(st.translate(cur.nAnalytic + t, 'cur', 'prev')).toBe(prev.nAnalytic + t);
        }
        if (env && cur.nEntries > 0 && prev.nEntries > 0) {
          expect(st.translate(prev.envEntry, 'prev', 'cur')).toBe(cur.envEntry);
          expect(st.translate(cur.envEntry, 'cur', 'prev')).toBe(prev.envEntry);
        }
        expect(st.translate(NO_ENTRY, 'prev', 'cur')).toBe(NO_ENTRY);
        expect(st.translate(3, 'prev', 'cur', true)).toBe(3);
      }
    }
  });
});
