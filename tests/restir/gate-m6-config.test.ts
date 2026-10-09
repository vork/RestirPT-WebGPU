// M6 gate configuration (restir-m6-api.md §4–§5; validation/harness/gate-m6.ts): unit lists per rung, package light
// modes, seeds disjoint from M4 / M5, the M6 T16 assertions (sequential and chains), Gate 5 pairs, plant predictions.
import { existsSync, readFileSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { describe, expect, it } from 'vitest';
import { restirSettings } from '../../src/core/render/restir/presets.ts';
import { SEEDS as M4_SEEDS } from '../../validation/harness/gate-m4.ts';
import { SEEDS as M5_SEEDS, t16ChainProblems } from '../../validation/harness/gate-m5.ts';
import { ALL_PARTS, CHAIN_UNITS, DECISION_CHAIN_UNITS, M6_PKGS, PARTS, PLANTS_M6, SEEDS, SEQ_UNITS, T3M6_RUNS, nUnits, pkgDirM6, t16M6 } from '../../validation/harness/gate-m6.ts';

const ROOT = fileURLToPath(new URL('../../', import.meta.url));

describe('M6 unit lists (§5.1)', () => {
  it('rung 3.7: σ 16 on (i) + (v) V1, RR on (i) + (iv), RIS-NEE on (x), (iv), (xiv) overcast + rect, C0r, all on (i) + (v) V2, cached M4 PT; chains full-m6 + dual MVs', () => {
    const r37 = SEQ_UNITS.filter((u) => u.rung === '3.7');
    const tags = r37.map((u) => u.id).sort();
    expect(tags).toEqual([
      'c0r_irradiance_256@3.7-ris', 'cornell_i_512@3.7-all', 'cornell_i_512@3.7-gauss', 'cornell_i_512@3.7-rr', 'iv_emissive_mesh_512@3.7-ris', 'iv_emissive_mesh_512@3.7-rr',
      'v_glossy_v1_512@3.7-gauss', 'v_glossy_v2_512@3.7-all', 'x_many_lights_512@3.7-ris', 'xiv_overcast_rect_b3_512@3.7-ris',
    ]);
    for (const u of r37) expect(u.pt, u.id).toBe('m4');
    expect(CHAIN_UNITS.filter((u) => u.rung === '3.7').map((u) => `${u.pkg}:${u.preset}`)).toEqual(['m5s_cornell_i:full-m6', 'ixs_d_camera_256:full']);
  });
  it('rung 3.9: 21 glass units (C0h–C0k, G1 ×2, G3–G10, (vi) A / B, (vi-B), (xiv) glass), offline-m6; 3.10: (xii) + x10_foliage; 3.11: 4 sequential + 2 chain units in Mode B', () => {
    expect(SEQ_UNITS.filter((u) => u.rung === '3.9')).toHaveLength(21);
    for (const u of SEQ_UNITS.filter((x) => x.rung !== '3.7')) expect(u.preset, u.id).toBe('offline-m6');
    expect(SEQ_UNITS.filter((u) => u.rung === '3.10').map((u) => u.pkg)).toEqual(['xii_alpha_foliage_512', 'x10_foliage_256']);
    const r311 = [...SEQ_UNITS.filter((u) => u.rung === '3.11'), ...CHAIN_UNITS.filter((u) => u.rung === '3.11')];
    expect(r311).toHaveLength(6);
    for (const u of r311) expect(u.lightMode, u.id).toBe('B');
  });
  it('every package exists and declares the unit\'s light mode (the PT reads it from the package)', () => {
    for (const u of [...SEQ_UNITS, ...CHAIN_UNITS]) {
      const f = path.join(ROOT, M6_PKGS.includes(u.pkg) || !existsSync(path.join(ROOT, `validation/scenes/${u.pkg}`)) ? pkgDirM6(u.pkg) : `validation/scenes/${u.pkg}`, 'scene.json');
      if (!existsSync(f)) { expect(M6_PKGS.includes(u.pkg) || /^xiv_|^c0r_|^c0q_/.test(u.pkg), `${u.pkg}: generated package`).toBe(true); continue; }
      const j = JSON.parse(readFileSync(f, 'utf8'));
      expect(j.lightMode ?? 'A', u.id).toBe(u.lightMode);
    }
  });
  it('Gate 5: 3 scenes × c_cap {5, 20} × duplication map {on, off}, interactive; T16 biased only with the map on', () => {
    const g5 = CHAIN_UNITS.filter((u) => u.gate5);
    expect(g5).toHaveLength(12);
    for (const u of g5) {
      expect(u.preset).toBe('interactive');
      expect(u.t16?.biased === 'dupmap').toBe(u.gate5!.dupmap);
      expect(u.t16?.rr).toBe(true);
      expect(u.t16?.cCap).toBe(u.gate5!.cCap);
    }
    expect(new Set(g5.map((u) => u.gate5!.scene))).toEqual(new Set(['m5s_cornell_i', 'm5s_glossy_v1', 'ixs_d_camera_256']));
  });
  it('perf2 decisions (opt-in part): interactive c_cap 5 at the app defaults, map off, unbiased T16, Mode B on m6_crossings_B; outside nUnits()', () => {
    expect(PARTS).not.toContain('decisions');
    expect(ALL_PARTS).toContain('decisions');
    expect(DECISION_CHAIN_UNITS.map((u) => `${u.pkg}:${u.lightMode}:${u.kind}`)).toEqual(['m6_crossings_B_256:B:static', 'm5s_cornell_i:A:static', 'ixs_d_camera_256:A:dyn']);
    for (const u of DECISION_CHAIN_UNITS) {
      expect(u.preset).toBe('interactive');
      expect(JSON.parse(u.extra[1])).toEqual({ cCap: 5, rrMinBounces: 2, dupmap: false });
      expect(u.t16?.biased).toBeUndefined();
      expect(u.t16?.cCap).toBe(5);
      expect(u.gate5).toBeUndefined();
      expect(CHAIN_UNITS).not.toContain(u);
    }
  });
  it('plants: U8-4 (M_foot +), U8-7 (global −), U8-8 (global +), U8-10 (global +, M_hl +) with derivations; T3-M6 runs', () => {
    const pr = Object.fromEntries(PLANTS_M6.map((p) => [p.id, p.predict.map((x) => `${x.region}${x.sign}`).join(' ')]));
    expect(pr).toEqual({ 'U8-4': 'M_foot+ globaldetect', 'U8-7': 'global-', 'U8-8': 'global+', 'U8-10': 'global+ M_hl+' });
    for (const p of PLANTS_M6) expect(p.derivation.length).toBeGreaterThan(20);
    expect(T3M6_RUNS).toHaveLength(6);
    expect(nUnits()).toBeGreaterThan(4 * 40);
  });
  it('seeds disjoint from M4 and M5', () => {
    const m6 = new Set<number>(Object.entries(SEEDS).filter(([k]) => k !== 'revisedOffset').map(([, v]) => v));
    for (const v of [...Object.values(M4_SEEDS), ...Object.values(M5_SEEDS)]) expect(m6.has(v as number), String(v)).toBe(false);
  });
});

describe('M6 T16 (§5.4)', () => {
  const pt = { ok: true, kernel: 'pt', width: 64, height: 64, config: { jitter: 'iid-per-run', maxBounces: 3, lightMode: 'B', rr: false, scene: 'abc', env: 'none' } };
  const rs = (o: { lightMode?: string; m6?: Record<string, unknown> } = {}) => {
    const s = restirSettings('offline-m6');
    return {
      ok: true, kernel: 'restir', width: 64, height: 64, members: 1,
      config: { jitter: 'iid-per-run', scene: 'abc', env: 'none' },
      t16: { validationModeUnbiased: true, plantsNamed: [], internalScale: 1, denoiser: 'none', upscaler: 'none', readback: 'linear accumulation', jitterMode: 'iid-per-run',
        maxBounces: 3, lightMode: o.lightMode ?? 'B', spatialRoundsExecuted: 3,
        m6: { pairing: s.pairing, pairSigma: 16, risNee: s.risNee, risM: 32, dualMv: false, dupmap: false, rr: false, ...o.m6 } },
    };
  };
  const expectS = restirSettings('offline-m6');
  it('passes on a matching Mode-B offline-m6 run', () => {
    expect(t16M6(rs(), pt, { rounds: 3, lightMode: 'B', expect: expectS })).toEqual([]);
  });
  it('flags a light-mode mismatch, a feature that differs from the unit, and the duplication map', () => {
    expect(t16M6(rs({ lightMode: 'A' }), pt, { rounds: 3, lightMode: 'B', expect: expectS }).join(' ')).toMatch(/light mode/);
    expect(t16M6(rs({ m6: { risNee: false } }), pt, { rounds: 3, lightMode: 'B', expect: expectS }).join(' ')).toMatch(/risNee/);
    expect(t16M6(rs({ m6: { dupmap: true } }), pt, { rounds: 3, lightMode: 'B', expect: expectS }).join(' ')).toMatch(/duplication map/);
  });
  it('chains: the light mode, c_cap, RR and the named biased feature are asserted', () => {
    const cm = { kernel: 'restir', kind: 'chains', ok: true, width: 64, height: 64, members: 16, config: { scene: 'abc', env: 'none' },
      t16: { validationModeUnbiased: false, plantsNamed: [], internalScale: 1, denoiser: 'none', upscaler: 'none', readback: 'linear rows', jitterMode: 'iid-per-run', maxBounces: 3,
        lightMode: 'A', rr: true, temporal: true, cCap: 5, spatialRoundsExecuted: 1, historyPattern: [false, true], resetFrames: [0], m6: { dupmap: true } } };
    const ptA = { ...pt, config: { ...pt.config, lightMode: 'A', frame: null } };
    expect(t16ChainProblems(cm, ptA, { rounds: 1, staticScene: true, cCap: 5, rr: true, biased: 'dupmap' })).toEqual([]);
    expect(t16ChainProblems(cm, ptA, { rounds: 1, staticScene: true, cCap: 5, rr: true }).join(' ')).toMatch(/not unbiased|duplication map/);
    expect(t16ChainProblems(cm, ptA, { rounds: 1, staticScene: true, rr: true, biased: 'dupmap' }).join(' ')).toMatch(/cCap 5 != 20/);
  });
});
