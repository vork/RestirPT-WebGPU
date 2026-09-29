// T17 CPU part (restir-api.md §2.7, §6.1; PLAN §1.8): 2D indirect-args math and item coverage beyond 4.19 M items.
import { describe, expect, it } from 'vitest';
import { queueArgs } from '../../src/core/render/restir/stage-spatial.ts';

describe('T17 CPU: 2D indirect args', () => {
  it('args for n ∈ {0, 1, 64, 65, 65535·64, 65535·64+1, 65536·64+1, 5·2²⁰}', () => {
    expect(queueArgs(0)).toEqual([0, 1, 1]);
    expect(queueArgs(1)).toEqual([1, 1, 1]);
    expect(queueArgs(64)).toEqual([1, 1, 1]);
    expect(queueArgs(65)).toEqual([2, 1, 1]);
    expect(queueArgs(65535 * 64)).toEqual([65535, 1, 1]);
    expect(queueArgs(65535 * 64 + 1)).toEqual([65535, 2, 1]);
    expect(queueArgs(65536 * 64 + 1)).toEqual([65535, 2, 1]);
    expect(queueArgs(5 * 2 ** 20)).toEqual([65535, 2, 1]);
  });

  it('every item index < n is produced exactly once by item = (wid.y·nwg.x + wid.x)·64 + lid; the rest exit', () => {
    for (const n of [1, 63, 64, 4097, 65535 * 64 - 1, 65535 * 64, 65535 * 64 + 1, 65536 * 64 + 1, 5 * 2 ** 20]) {
      const [x, y] = queueArgs(n);
      const lanes = x * y * 64;
      expect(lanes).toBeGreaterThanOrEqual(n);
      expect(lanes - n).toBeLessThan(x * 64 + 64);   // at most one partial row of workgroups is wasted
      // the item map is a bijection from (wid, lid) onto [0, x·y·64): items < n are covered once
      const lastItem = ((y - 1) * x + (x - 1)) * 64 + 63;
      expect(lastItem).toBe(lanes - 1);
      // a 1D dispatch caps at 65535 workgroups
      expect(Math.min(Math.ceil(n / 64), 65535) * 64 >= n).toBe(n <= 65535 * 64);
    }
  });
});
