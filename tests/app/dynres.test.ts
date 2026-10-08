// M8 dynamic resolution controller (src/app/dynres.ts; docs/decisions/m8-perf.md §9).
import { describe, expect, it } from 'vitest';
import { DYNRES_LEVELS, DynamicResolution, gpuBusyMs } from '../../src/app/dynres.ts';
import { internalResolution } from '../../src/core/render/present.ts';

/** A frame-time model: ms = base · (scale)² (pixel count) — the controller sees it with ±3 % noise. */
function simulate(base: number, frames: number, c = new DynamicResolution()): { levels: number[]; changes: number } {
  const levels: number[] = [];
  let changes = 0, seed = 7;
  for (let f = 0; f < frames; f++) {
    seed = (seed * 1103515245 + 12345) >>> 0;
    const noise = 1 + 0.06 * ((seed / 2 ** 32) - 0.5);
    if (c.update(base * c.scale * c.scale * noise)) changes++;
    levels.push(c.level);
  }
  return { levels, changes };
}

describe('dynamic resolution (M8)', () => {
  it('stays at full resolution under the target', () => {
    const r = simulate(20, 600);
    expect(r.changes).toBe(0);
    expect(r.levels.at(-1)).toBe(0);
  });
  it('steps down until the frame time is under the target, then holds (no oscillation)', () => {
    const r = simulate(80, 2000);   // Sponza-like: 80 ms at 100 % → 0.625 (31 ms) is the first level ≤ 33 · 1.08
    const final = r.levels.at(-1)!;
    expect(80 * DYNRES_LEVELS[final] ** 2).toBeLessThanOrEqual(33 * 1.08);
    expect(r.changes).toBe(final);   // monotone: one change per level, never back up
  });
  it('steps back up when the load drops', () => {
    const c = new DynamicResolution();
    simulate(80, 2000, c);
    const down = c.level;
    simulate(12, 2000, c);
    expect(c.level).toBeLessThan(down);
    expect(c.level).toBe(0);
  });
  it('ignores non-finite samples and the frames right after a change', () => {
    const c = new DynamicResolution({ minFrames: 1 });
    expect(c.update(Number.NaN)).toBe(false);
    expect(c.update(-1)).toBe(false);
    for (let i = 0; i < 3; i++) expect(c.update(1000)).toBe(false);
    expect(c.update(1000)).toBe(true);
    for (let i = 0; i < 3; i++) expect(c.update(1000)).toBe(false);
  });
  it('GPU busy time and scaled internal resolutions', () => {
    expect(gpuBusyMs(10, 40, undefined)).toBe(30);
    expect(gpuBusyMs(10, 40, 25)).toBe(15);
    expect(internalResolution('540p', 1920, 1080)).toEqual([960, 540]);
    expect(internalResolution('540p', 1920, 1080, 0.5)).toEqual([480, 272]);
    expect(internalResolution('720p', 1920, 1080, 0.75)).toEqual([968, 544]);
  });
});
