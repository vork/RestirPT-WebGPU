// perf2 WP-5 (docs/decisions/perf2-plan.md §2 WP-5): same-run A/B of the spatial-reuse flags against the flag-free
// kernel, frame by frame — final reservoirs, the frame image, the finalize counters and the WHOLE arena header except
// the WP-5 words (q3 header 12–15, boost gate 30): every RSC_* counter, the SC histogram and the q0 header.
// Complements U-M8-BITS (goldens) with paths the goldens do not force:
//   - static frames, where the boost gate can close (logged: frames with the gate word 0 must occur on 'all' and
//     'sponza_lite', otherwise the gated path was not exercised);
//   - row bands of 16 (several pair_accept / replay / dense-shift chunks per round);
//   - an E = 2 ensemble (memberCount 2: the pairing table and rs_pix fall back to the per-thread path);
//   - non-advanced (reset) frames with boost slots (RSD_BOOST_OPEN).
// Flags: RS_DENSE_SLOTS, RS_BOOST_GATE, RS_MIS_TRIM, RS_PAIR_TABLE, each alone and all together.
import { describe, expect, it } from 'vitest';
import { readBuffer } from '../../src/core/gpu/readback.ts';
import { INTERACTIVE_PINNED, type RestirPresetName, type RestirSettings } from '../../src/core/render/restir/presets.ts';
import { WP5_CONSTS } from '../../src/core/render/restir/layout.ts';
import type { LightMode } from '../../src/core/render/lights-gpu.ts';
import { hashF32, restirRig } from './restir-fixtures.ts';
import { m8BitsScene, type M8BitsScene } from './m8-bits.ts';
import type { GpuSceneOptions } from './restir-fixtures.ts';

interface Case {
  name: string; scene: M8BitsScene; preset: RestirPresetName; settings: Partial<RestirSettings>; lightMode: LightMode;
  size: [number, number]; staticFrames: number; movingFrames: number; advance: boolean; rowBand?: number; members?: number;
  gpu?: GpuSceneOptions; kernel?: { resLayout?: 'aos' | 'soa'; modeBNeedsAreaLights?: boolean }; expectGateClosed?: boolean;
}
const INTERACTIVE: Partial<RestirSettings> = { maxBounces: 3, ...INTERACTIVE_PINNED };
const CASES: Case[] = [
  { name: 'all-interactive-B', scene: 'all', preset: 'interactive', settings: INTERACTIVE, lightMode: 'B', size: [64, 64], staticFrames: 6, movingFrames: 3, advance: true, expectGateClosed: true },
  { name: 'all-interactive-B-bands16', scene: 'all', preset: 'interactive', settings: INTERACTIVE, lightMode: 'B', size: [64, 64], staticFrames: 4, movingFrames: 2, advance: true, rowBand: 16 },
  { name: 'sponza-lite-interactive', scene: 'sponza_lite', preset: 'interactive', settings: INTERACTIVE, lightMode: 'B', size: [160, 90], staticFrames: 6, movingFrames: 2, advance: true,
    gpu: { bvhKind: 'cwbvh', watertight: false }, kernel: { resLayout: 'soa', modeBNeedsAreaLights: true } },
  { name: 'all-full-boost-reset', scene: 'all', preset: 'full', settings: { maxBounces: 3, boostSlots: 3 }, lightMode: 'B', size: [64, 48], staticFrames: 2, movingFrames: 0, advance: false },
  { name: 'all-offline-m6-E2', scene: 'all', preset: 'offline-m6', settings: { maxBounces: 3, trees: 2 }, lightMode: 'B', size: [48, 32], staticFrames: 1, movingFrames: 0, advance: false, members: 2 },
];
const FLAG_SETS = ['RS_DENSE_SLOTS', 'RS_DENSE_SLOTS,RS_MIS_TRIM', 'RS_BOOST_GATE', 'RS_MIS_TRIM', 'RS_PAIR_TABLE', 'RS_DENSE_SLOTS,RS_BOOST_GATE,RS_MIS_TRIM,RS_PAIR_TABLE'];
const WP5_HDR = new Set([12, 13, 14, 15, WP5_CONSTS.RS_HDR_BOOST_GATE]);

function yawed(cam: { camToWorld: number[]; yfov: number }, a: number): { camToWorld: number[]; yfov: number } {
  const m = cam.camToWorld.slice();
  const c = Math.cos(a), s = Math.sin(a);
  for (const col of [0, 4, 8]) { const x = m[col], z = m[col + 2]; m[col] = c * x + s * z; m[col + 2] = -s * x + c * z; }
  return { camToWorld: m, yfov: cam.yfov };
}

interface FrameOut { hash: string; gate: number; q3: number; q0: number; accepted: number }

async function runCase(c: Case, flags: string): Promise<FrameOut[]> {
  const { scene, cam: cam0 } = await m8BitsScene(c.scene);
  const [W, H] = c.size;
  const rig = await restirRig(scene, W, H, {
    preset: c.preset, settings: c.settings, seed: 31, lightMode: c.lightMode, cam: cam0, resLayout: c.kernel?.resLayout,
    modeBNeedsAreaLights: c.kernel?.modeBNeedsAreaLights, gpu: c.gpu, perfFlags: flags, members: c.members,
  });
  const k = rig.kernel, device = rig.g.device;
  if (c.rowBand) k.rowBand = c.rowBand;
  await k.prepare();
  const out: FrameOut[] = [];
  let cam = cam0;
  const n = c.staticFrames + c.movingFrames;
  for (let f = 0; f < n; f++) {
    const t = 3 + f;
    if (f >= c.staticFrames) cam = yawed(cam, 0.01);
    const clear = device.createCommandEncoder();
    clear.clearBuffer(rig.accum); clear.clearBuffer(rig.counters);
    device.queue.submit([clear.finish()]);
    if (c.advance && k.settings.temporal) k.advance({ t, camera: cam, lights: scene.lights });
    k.beginSubmit();
    const enc = device.createCommandEncoder();
    for (const u of k.frameUnits(t, { accum: rig.accum, counters: rig.counters })) u.encode(enc);
    device.queue.submit([enc.finish()]);
    await device.queue.onSubmittedWorkDone();
    const img = new Float32Array(await readBuffer(device, rig.accum, W * H * 16));
    const fin = await k.readReservoirs('final');
    const cnt = Array.from(new Uint32Array(await readBuffer(device, rig.counters, 16)));
    const ar = await k.readCounters(true);
    const hdr = Array.from(ar.raw).map((v, i) => (WP5_HDR.has(i) ? 0 : v));
    out.push({
      hash: `${hashF32(new Float32Array(fin.buffer))}:${hashF32(img)}:${cnt.join(',')}:${hdr.join(',')}`,
      gate: ar.raw[WP5_CONSTS.RS_HDR_BOOST_GATE], q3: ar.raw[4 * WP5_CONSTS.RS_Q_DENSE], q0: ar.raw[0], accepted: ar.rsc.accepted,
    });
  }
  rig.destroy();
  return out;
}

describe('WP-5 spatial flags: same-run A/B against the flag-free kernel', () => {
  for (const c of CASES) {
    it(`${c.name}: every flag set bitwise equal to no flags (reservoirs, image, counters, arena header)`, async () => {
      const base = await runCase(c, '');
      const msgs: string[] = [];
      for (const fs of FLAG_SETS) {
        const got = await runCase(c, fs);
        const gateClosed = got.filter((g) => fs.includes('RS_BOOST_GATE') && c.advance && g.gate === 0).length;
        console.log(`[wp5] ${c.name} ${fs}: q3 ${got.map((g) => g.q3).join('/')} q0 ${got.map((g) => g.q0).join('/')} gate ${got.map((g) => g.gate).join('')} (closed ${gateClosed}) accepted ${got.map((g) => g.accepted).join('/')}`);
        for (let f = 0; f < base.length; f++) if (got[f].hash !== base[f].hash) msgs.push(`${c.name} ${fs} frame ${f}: ${got[f].hash.slice(0, 40)} ≠ ${base[f].hash.slice(0, 40)}`);
        if (fs.includes('RS_DENSE_SLOTS')) expect(got.some((g) => g.q3 > 0), 'q3 used').toBe(true);
        if (c.expectGateClosed && fs.includes('RS_BOOST_GATE')) expect(gateClosed, 'frames with a closed boost gate').toBeGreaterThan(0);
      }
      expect(msgs, msgs.join('\n')).toEqual([]);
    }, 600_000);
  }
});
