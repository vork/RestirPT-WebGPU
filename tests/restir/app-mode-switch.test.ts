import { describe, expect, it } from 'vitest';
import { restirAppSettings, RESTIR_APP_MODES, type RestirAppMode } from '../../src/core/render/renderer.ts';
import { DEFAULT_RESTIR_SETTINGS, validateSettings } from '../../src/core/render/restir/presets.ts';

// The kernel merges setSettings() onto its current settings; every app mode must therefore be complete and valid
// after any previous mode (regression: interactive / 2022-criteria → Offline threw "slots + boostSlots = 9 > 6").
const MODES = Object.keys(RESTIR_APP_MODES) as RestirAppMode[];

describe('app ReSTIR mode switches', () => {
  for (const from of MODES) for (const to of MODES) {
    it(`${from} → ${to}`, () => {
      const merged = { ...DEFAULT_RESTIR_SETTINGS, ...restirAppSettings(from, 3), ...restirAppSettings(to, 3) };
      expect(() => validateSettings(merged)).not.toThrow();
      expect(merged).toEqual({ ...DEFAULT_RESTIR_SETTINGS, ...restirAppSettings(to, 3) });
    });
  }
});

it('Potato bounds work without overwriting the other modes or explicit feature overrides', () => {
  const p = restirAppSettings('potato', 5, true);
  expect(p).toMatchObject({ maxBounces: 1, temporal: false, slots: 1, rounds: 1, boostSlots: 0,
    risM: 4, rrMinBounces: 1, dupmap: false, dualMv: false });
  expect(restirAppSettings('potato', 0).maxBounces).toBe(0);
  expect(restirAppSettings('interactive', 5)).toMatchObject({ maxBounces: 5, temporal: true, slots: 3, boostSlots: 3, risM: 32 });
  expect(restirAppSettings('potato', 3, true, { rrMinBounces: 2 }).rrMinBounces).toBe(2);
});
