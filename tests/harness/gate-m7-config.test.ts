// M7 gate configuration (validation/harness/gate-m7.ts; m7-api.md §6): the unit lists, tiers and plant ids are consistent
// with the generators and the recorded predictions.
import { describe, expect, it } from 'vitest';
import { CHAIN_UNITS, E2E_PKGS, M7_PKGS, MODEL_APPROX_CHECKS, nUnits, PLANT_IDS, SEEDS, SEQ_UNITS, STAGE_A_UNITS, TANGENT_PKGS } from '../../validation/harness/gate-m7.ts';
import { E2E_UNITS } from '../../validation/scenes/make-m7-e2e.ts';

describe('M7 gate configuration', () => {
  it('E2E packages = make-m7-e2e.ts units, all in the e2e part and model-approximate', () => {
    expect(new Set(E2E_PKGS)).toEqual(new Set(E2E_UNITS.map((u) => u.name)));
    const e2e = STAGE_A_UNITS.filter((u) => u.part === 'e2e');
    expect(e2e.map((u) => u.pkg).sort()).toEqual([...E2E_PKGS].sort());
    expect(e2e.every((u) => u.tier === 'model-approximate')).toBe(true);
  });
  it('Stage A: references >= 4096 spp (D4), K = B = 16; flat normal maps and E2E-HDR are tight', () => {
    for (const u of STAGE_A_UNITS) { expect(u.cyclesSpp).toBeGreaterThanOrEqual(4096); expect(u.K).toBe(16); expect(u.B).toBe(16); }
    for (const p of ['m7_nm_flat_256', 'm7_xivlite_hdr_256', 'm7_xivlite_exr_256']) expect(STAGE_A_UNITS.find((u) => u.pkg === p)!.tier).toBe('tight');
    expect(STAGE_A_UNITS.filter((u) => u.hdr).map((u) => u.hdr).sort()).toEqual(['exr', 'hdr']);
  });
  it('the model-approximate tier only switches off the Δ = 0 rejection checks (TOST and min_replicates stay gating)', () => {
    expect(Object.keys(MODEL_APPROX_CHECKS).sort()).toEqual(['chi2_red', 'ks_ad', 'mean_t', 'numeric_tiles', 'sidak_tiles']);
    expect(Object.values(MODEL_APPROX_CHECKS).every((v) => v === false)).toBe(true);
  });
  it('rung 3.8 covers every M7 Stage-B package; the chain unit is static full-m6 t ∈ {1, 24}', () => {
    for (const p of ['m7_smooth_256', 'm7_nm_flat_256', 'm7_nm_smooth_256', 'm7_nm_env_256', 'm7_nm_smooth_B_256']) expect(SEQ_UNITS.some((u) => u.pkg === p)).toBe(true);
    for (const u of SEQ_UNITS) expect(M7_PKGS).toContain(u.pkg);
    expect(SEQ_UNITS.find((u) => u.pkg === 'm7_nm_smooth_B_256')!.lightMode).toBe('B');
    expect(CHAIN_UNITS[0]).toMatchObject({ kind: 'static', preset: 'full-m6', testFrames: [1, 24] });
  });
  it('plants, seeds, tangent packages, n_units', () => {
    expect(PLANT_IDS).toEqual(['A-NM-sign', 'A-NM-strength', 'B-NM-sign', 'B-NM-strength', 'B-SM-J']);
    const s = Object.values(SEEDS).flat();
    expect(new Set(s).size).toBeGreaterThan(10);
    expect(SEEDS.ours).not.toBe(SEEDS.oursRerun);
    for (const p of TANGENT_PKGS) expect([...M7_PKGS, ...E2E_PKGS]).toContain(p);
    expect(nUnits()).toBe(4 * (STAGE_A_UNITS.length + 1 + SEQ_UNITS.length + 2 + PLANT_IDS.length + 2));
  });
});
