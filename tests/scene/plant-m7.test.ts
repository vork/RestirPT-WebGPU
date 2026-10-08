// M7 plant predictions (docs/decisions/m7-api.md §5): the derivation lives in validation/scenes/m7-plants.ts; this test
// pins the recorded signs (a change must be a deliberate re-derivation BEFORE any planted render is measured).
import { describe, expect, it } from 'vitest';
import { m7PlantPredictions } from '../../validation/scenes/m7-plants.ts';

describe('M7 plant predictions (derived before measurement)', () => {
  it('sign / strength plants: the panel direct-irradiance change has a definite sign', () => {
    const p = m7PlantPredictions();
    console.log('M7_PLANT_PREDICTIONS', JSON.stringify(p));
    // the recorded predictions (m7-api.md §5.1): a change here must be a deliberate re-derivation
    expect(p.sign.sign).toBe('-');
    expect(p.strength.sign).toBe('-');
  });
});
