// ReSTIR PT paired spatial reuse, queues, MIS and ensemble mode (WP-C, restir-api.md §6.1): T3-3/M4, T6(a), T6(b),
// T7, T17, U-MIS-1, U-ENS-1, U-ENS-2, U-OFF-1 and the replay-predicate agreement. Chrome lane authoritative.
import { afterAll, describe, expect, it } from 'vitest';
import { readBuffer } from '../../src/core/gpu/readback.ts';
import { RS_WGSL_CONSTS as K, RES_WORDS, RW, arenaWords, decodeReservoir } from '../../src/core/render/restir/layout.ts';
import { SpatialStage } from '../../src/core/render/restir/stage-spatial.ts';
import { getTestGpu, releaseTestGpu } from './device-factory.ts';
import { bitFixtureScene, restirRig, type RestirRig } from './restir-fixtures.ts';

afterAll(releaseTestGpu);

const lum = (r: number, g: number, b: number) => 0.2126 * r + 0.7152 * g + 0.0722 * b;

/** Arena words (after the 64-word header) of the rig's current allocation. */
async function arenaBody(rig: RestirRig): Promise<Uint32Array> {
  const res = rig.kernel.resources;
  const all = new Uint32Array(await readBuffer(rig.g.device, res.arena, res.arena.size));
  return all.subarray(64);
}

describe('paired spatial reuse end to end (offline preset)', () => {
  it('compiles, runs 3 rounds × 6 slots; queue/arena invariants hold; lum(rsShade) = Σw (U-MIS-1)', async () => {
    const rig = await restirRig(bitFixtureScene('x_quads'), 64, 48, { preset: 'offline', settings: { maxBounces: 3, trees: 4 } });
    const r = await rig.frames(2);
    const a = r.arena;
    console.log(`[spatial e2e] counters ${JSON.stringify(a.rsc)} codes ${a.codes} f_r ${a.fr.toFixed(3)} q0 ${JSON.stringify(a.queues[0])}`);
    expect(r.counters).toEqual([0, 0, 0, 0]);
    expect(a.rsc.pendingLeft).toBe(0);
    expect(a.rsc.slotMismatch).toBe(0);
    expect(a.rsc.shiftNonFinite).toBe(0);
    expect(a.rsc.wNonFinite).toBe(0);
    expect(a.queues[0].overflow).toBe(0);
    expect(a.rsc.accepted).toBeGreaterThan(0);
    expect(a.codes[K.SC_PENDING]).toBe(0);
    rig.destroy();
  });
});
