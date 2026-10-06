// U-TL-3 (restir-temporal-api.md §2.10, TD19; T-A): every input of the config hash changes it; none of the non-reset
// edits (light edits, animation, camera motion, env rotation / strength / tint, visibleToCamera) is an input.
import { describe, expect, it } from 'vitest';
import { configHash, type ConfigHashInput } from '../../src/core/render/restir/frame-state.ts';
import { restirFlags, restirSettings, tModeOf, tPlantsOf } from '../../src/core/render/restir/presets.ts';

const s = restirSettings('full');
const base: ConfigHashInput = {
  sceneGen: 1, atlas: [1024, 1024], member: [256, 256], members: 16, memberBase: 0, lightMode: 'A', lightsLayout: 0,
  envMapGen: 0, envMapId: 'env', importanceKey: '{"cap":4096}', envNee: true, settings: { restir: s, env: {} },
  flags: restirFlags(s), tMode: tModeOf(s), tPlants: tPlantsOf(s), jitterMode: 1, misM: 1,
};

describe('U-TL-3: config hash', () => {
  it('is deterministic and key-order independent', () => {
    expect(configHash(base)).toBe(configHash({ ...base }));
    const reordered = Object.fromEntries(Object.entries(base).reverse()) as unknown as ConfigHashInput;
    expect(configHash(reordered)).toBe(configHash(base));
  });
  it('every hashed input changes it (§2.10)', () => {
    const edits: [string, Partial<ConfigHashInput>][] = [
      ['scene', { sceneGen: 2 }], ['atlas', { atlas: [1024, 512] }], ['member size', { member: [128, 256] }], ['E', { members: 8 }],
      ['memberBase', { memberBase: 16 }], ['light mode', { lightMode: 'B' }], ['lights reallocated', { lightsLayout: 1 }],
      ['env map gen', { envMapGen: 1 }], ['env map id', { envMapId: 'b' }], ['importance key', { importanceKey: '{"cap":1024}' }],
      ['env NEE', { envNee: false }], ['jitter mode', { jitterMode: 0 }], ['M(B)', { misM: 32 }],
      ['flags', { flags: base.flags | 4 }], ['tMode', { tMode: 1 }], ['tPlants', { tPlants: 2 }],
      ['env options', { settings: { restir: s, env: { misPower: true } } }],
    ];
    const settingEdits: Partial<typeof s>[] = [
      { maxBounces: 4 }, { criteria: '2022' }, { tau: 1e-4 }, { alphaMin: 0.3 }, { cCap: 10 }, { temporalMis: 'talbot' }, { temporalCheck: 'robust' },
      { refresh: 'e2' }, { tPlant: { n1Mixed: true } }, { plant: { wScale: 1.003 } }, { plant: { u8T2: true } }, { slots: 2 }, { boostSlots: 1 },
      { rr: true }, { trees: 2 }, { rounds: 2 }, { diskRadius: 10 },
    ];
    for (const e of settingEdits) edits.push([`settings ${JSON.stringify(e)}`, { settings: { restir: { ...s, ...e }, env: {} } }]);
    const h0 = configHash(base);
    const seen = new Set([h0]);
    for (const [name, e] of edits) {
      const h = configHash({ ...base, ...e });
      expect(h, name).not.toBe(h0);
      seen.add(h);
    }
    expect(seen.size).toBe(edits.length + 1);
  });
  it('non-reset edits are not inputs (lights, camera, env rotation / strength / tint, visibleToCamera, t)', () => {
    const keys = Object.keys(base).sort();
    expect(keys).toEqual(['atlas', 'envMapGen', 'envMapId', 'envNee', 'flags', 'importanceKey', 'jitterMode', 'lightMode', 'lightsLayout', 'member', 'memberBase', 'members', 'misM', 'sceneGen', 'settings', 'tMode', 'tPlants']);
    const settingsKeys = JSON.stringify(base.settings);
    for (const k of ['rotationZ', 'strength', 'tint', 'visibleToCamera', 'camera', 'lights']) expect(settingsKeys).not.toContain(`"${k}"`);
  });
});
