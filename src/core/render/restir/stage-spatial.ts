// Paired spatial reuse stage (restir-api.md §4.1, §3.8–§3.9, §2.6–§2.8; PLAN §3 step 5, §1.8 queues). OWNER WP-C.
// Per round r (reads res[(w + r) % 2], writes res[(w + r + 1) % 2]; w = k.resBase(), 0 with temporal off — M5 TD2):
//   clear queue-0 {counter, n} (8 B; overflow stays sticky) → rs_pair_accept (row bands) → per replay chunk:
//   rs_args (1 thread) + rs_spatial_replay (2D indirect over the chunk; one chunk per row band, Changelog C7) →
//   rs_spatial_shift (row bands) → rs_spatial_resample (row bands; RSD_FINAL_ROUND on the last round writes rsShade).
// The stage emits all rounds or none (Changelog A6). Every stage of a round covers the whole atlas before the next
// stage starts (units are emitted in order and the runner keeps unit order), because pair_accept writes the slots of
// both pixels of a pair and the shift/resample passes read partners in other bands.
// The pairing texture (uniform-disk involution maps of radius settings.diskRadius, pairing.ts) is uploaded whenever the
// kernel's resources or the radius change.
import type { RestirKernel, RestirStage, WorkUnit } from './kernel.ts';
import { RS_WGSL_CONSTS as K } from './layout.ts';
import { uploadPairTex } from './pairing.ts';
import { PAIR_TEX_SIZES } from './presets.ts';
import type { RsPassName } from './resources.ts';

export const SPATIAL_PASSES: readonly RsPassName[] = ['rs_pair_accept', 'rs_args', 'rs_spatial_replay', 'rs_spatial_shift', 'rs_spatial_resample'];

/** 2D indirect args of n queue items (§2.7, mirror of queue.wgsl queue_args): (min(g, 65535), ceil(g / 65535), 1). */
export function queueArgs(n: number): [number, number, number] {
  const g = Math.ceil(n / K.RS_WG);
  return [Math.min(g, 65535), Math.max(Math.ceil(g / 65535), 1), 1];
}

export class SpatialStage implements RestirStage {
  private readonly uploaded = new WeakMap<GPUTexture, number>();

  async prepare(k: RestirKernel): Promise<void> {
    await Promise.all(SPATIAL_PASSES.map((n) => k.pipeline(n)));
  }

  /** Make sure the kernel's pairing texture holds the maps of the current disk radius. */
  ensurePairTex(k: RestirKernel): void {
    const tex = k.resources.pairTex;
    const R = k.settings.diskRadius;
    if (this.uploaded.get(tex) === R) return;
    uploadPairTex(k.device, tex, R, PAIR_TEX_SIZES);
    this.uploaded.set(tex, R);
  }

  frameUnits(k: RestirKernel, t: number): WorkUnit[] {
    const s = k.settings;
    if (s.rounds <= 0) return [];
    this.ensurePairTex(k);
    const res = k.resources;
    const a = res.alloc;
    const bands = k.rowBands();
    const pl = (n: RsPassName) => k.pipelineSync(n);
    const accept = pl('rs_pair_accept'), args = pl('rs_args'), replay = pl('rs_spatial_replay'), shift = pl('rs_spatial_shift'), resample = pl('rs_spatial_resample');
    const units: WorkUnit[] = [];
    const slotWork = (r0: number, r1: number) => a.atlasW * (r1 - r0) * s.slots;
    for (let r = 0; r < s.rounds; r++) {
      const inIdx = (k.resBase() + r) % 2;
      const final = r === s.rounds - 1;
      const d = { t, passId: K.RS_PASS_SPATIAL + r, round: r };
      bands.forEach(([r0, r1], bi) => units.push({
        label: `rs_pair_accept[${r}][${r0}]`, costHint: slotWork(r0, r1),
        encode: (enc) => {
          if (bi === 0) enc.clearBuffer(res.arena, 0, 8);
          k.encodePass(enc, 'rs_pair_accept', accept, res.g2('rs_pair_accept', inIdx), { ...d, rowBase: r0, rowEnd: r1 }, k.perPixelWorkgroups(r0, r1));
        },
      }));
      // Replay: one chunk per row band (Changelog C7) so the runner can split it across submits like the per-pixel
      // passes; with a single band the whole queue is one dispatch. Chunk c covers items [c·chunk, (c+1)·chunk).
      const chunk = bands.length > 1 ? Math.max(1, Math.ceil((a.atlasW * a.atlasH * s.slots) / bands.length)) : 0;
      bands.forEach(([r0, r1], ci) => units.push({
        label: `rs_spatial_replay[${r}][${ci}]`, costHint: slotWork(r0, r1) * (s.maxBounces + 1),
        encode: (enc) => {
          const dc = { ...d, treeBase: ci * chunk, treeCount: chunk };
          k.encodePass(enc, 'rs_args', args, res.g2('rs_args'), dc, [1, 1]);
          k.encodePass(enc, 'rs_spatial_replay', replay, res.g2('rs_spatial_replay', inIdx), dc, { indirect: res.args, offset: 0 });
        },
      }));
      for (const [r0, r1] of bands) {
        units.push({
          label: `rs_spatial_shift[${r}][${r0}]`, costHint: slotWork(r0, r1) * 2,
          encode: (enc) => k.encodePass(enc, 'rs_spatial_shift', shift, res.g2('rs_spatial_shift', inIdx), { ...d, rowBase: r0, rowEnd: r1 }, k.perPixelWorkgroups(r0, r1)),
        });
      }
      for (const [r0, r1] of bands) {
        units.push({
          label: `rs_spatial_resample[${r}][${r0}]`, costHint: slotWork(r0, r1),
          encode: (enc) => k.encodePass(enc, 'rs_spatial_resample', resample, res.g2('rs_spatial_resample', inIdx),
            { ...d, flags: final ? K.RSD_FINAL_ROUND : 0, rowBase: r0, rowEnd: r1 }, k.perPixelWorkgroups(r0, r1)),
        });
      }
    }
    return units;
  }

  destroy(): void { }
}
