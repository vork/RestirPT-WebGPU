import { describe, expect, it } from 'vitest';
import { restirAppSettings, type RestirAppMode } from '../../src/core/render/renderer.ts';
import { DEFAULT_RESTIR_SETTINGS, validateSettings } from '../../src/core/render/restir/presets.ts';

// The kernel merges setSettings() onto its current settings; every app mode must therefore be complete and valid
// after any previous mode (regression: interactive / 2022-criteria → Offline threw "slots + boostSlots = 9 > 6").
const MODES: RestirAppMode[] = ['interactive', 'unbiased', 'criteria2022', 'offline', 'initial'];

describe('app ReSTIR mode switches', () => {
  for (const from of MODES) for (const to of MODES) {
    it(`${from} → ${to}`, () => {
      const merged = { ...DEFAULT_RESTIR_SETTINGS, ...restirAppSettings(from, 3), ...restirAppSettings(to, 3) };
      expect(() => validateSettings(merged)).not.toThrow();
      expect(merged).toEqual({ ...DEFAULT_RESTIR_SETTINGS, ...restirAppSettings(to, 3) });
    });
  }
});
