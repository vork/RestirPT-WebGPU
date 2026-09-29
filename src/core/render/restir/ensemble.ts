// Ensemble atlas statistics (restir-api.md §2.10, D14/D15; PLAN §3 "Ensemble mode"). OWNER WP-C.
// EnsembleStage: after rs_finalize wrote rsFrame, rs_ensemble_stats (per (member, 16² tile) tree reduction + per-pixel
// Σx / Σx² over members into ensPixel) and rs_ensemble_reduce (32² / 64² tiles, global, mask) — both entry points of
// passes/restir/ensemble-stats.wgsl with the rs_ensemble_stats bindings. Deterministic fixed-order reductions.
// EnsembleCollector: host side (f64). Per frame it reads ensStats and appends one row per member (run id
// r = t·E + m, seed label `${runSeed}:${t}:${m}`); per batch it reads ensPixel into the f64 per-pixel sums and clears
// it; npzArrays() gives the arrays of ensemble.npz (compare.py README "Ensemble .npz format").
import { readBuffer } from '../../gpu/readback.ts';
import type { RestirKernel, RestirStage, WorkUnit } from './kernel.ts';
import type { NpzArray } from './npz.ts';
import { ENS_LEVELS, RS_PASSES, ensStatsFloats } from './resources.ts';

export const ENS_MASK_REGIONS = 16;

export class EnsembleStage implements RestirStage {
  private reduce: GPUComputePipeline | undefined;

  async prepare(k: RestirKernel): Promise<void> {
    const d = RS_PASSES.rs_ensemble_stats;
    const [, reduce] = await Promise.all([
      k.pipeline('rs_ensemble_stats'),
      k.compile(d.file, 'rs_ensemble_reduce', k.defines('rs_ensemble_stats'), k.pipelineLayout('rs_ensemble_stats'), 'rs_ensemble_reduce'),
    ]);
    this.reduce = reduce;
  }

  frameUnits(k: RestirKernel, t: number): WorkUnit[] {
    if (!this.reduce) throw new Error('EnsembleStage: prepare() first (RestirKernel.prepare after setView with members > 1)');
    const res = k.resources;
    const a = res.alloc;
    const stats = k.pipelineSync('rs_ensemble_stats');
    const reduce = this.reduce;
    const g2 = res.g2('rs_ensemble_stats');
    const tw = Math.ceil(a.memberW / 16), th = Math.ceil(a.memberH / 16);
    const entries = ensReduceEntries(a.members, a.memberW, a.memberH);
    return [{
      label: 'rs_ensemble_stats', costHint: a.atlasW * a.atlasH,
      encode: (enc) => {
        k.encodePass(enc, 'rs_ensemble_stats', stats, g2, { t }, [tw, th * a.members]);
        k.encodePass(enc, 'rs_ensemble_stats', reduce, g2, { t }, [Math.ceil(entries / 64), 1]);
      },
    }];
  }

  destroy(): void { }
}

/** Entries of the second reduction pass: E·(n32 + n64 + 1 + 16). */
export function ensReduceEntries(E: number, W: number, H: number): number {
  const n = (l: number) => Math.ceil(W / l) * Math.ceil(H / l);
  return E * (n(32) + n(64) + 1 + ENS_MASK_REGIONS);
}

/** Offsets (in vec4 units) of the parts of ensStats (§2.10). */
export function ensStatsLayout(E: number, W: number, H: number) {
  const tiles = ENS_LEVELS.map((l) => ({ level: l, th: Math.ceil(H / l), tw: Math.ceil(W / l) }));
  let o = 0;
  const off: Record<number, number> = {};
  for (const t of tiles) { off[t.level] = o; o += E * t.th * t.tw; }
  const global = o;
  const mask = global + E;
  return { tiles, off, global, mask, total: mask + E * ENS_MASK_REGIONS };
}

export interface EnsembleFrameStats {
  /** Per level: [m][ty][tx][rgb] (Float32Array of E·th·tw·3). */
  tiles: Record<number, { th: number; tw: number; data: Float32Array }>;
  global: Float32Array;   // [m][rgb]
  raw: Float32Array;
}

/** Split an ensStats read-back (vec4 f32) into per-level RGB arrays. */
export function decodeEnsStats(raw: Float32Array, E: number, W: number, H: number): EnsembleFrameStats {
  const L = ensStatsLayout(E, W, H);
  const rgb = (o: number, n: number) => {
    const out = new Float32Array(n * 3);
    for (let i = 0; i < n; i++) for (let c = 0; c < 3; c++) out[3 * i + c] = raw[4 * (o + i) + c];
    return out;
  };
  const tiles: EnsembleFrameStats['tiles'] = {};
  for (const t of L.tiles) tiles[t.level] = { th: t.th, tw: t.tw, data: rgb(L.off[t.level], E * t.th * t.tw) };
  return { tiles, global: rgb(L.global, E), raw };
}

export async function readEnsStats(k: RestirKernel): Promise<EnsembleFrameStats> {
  const res = k.resources, a = res.alloc;
  const raw = new Float32Array(await readBuffer(k.device, res.ensStats, ensStatsFloats(a.members, a.memberW, a.memberH) * 4));
  return decodeEnsStats(raw, a.members, a.memberW, a.memberH);
}

/** Host-side f64 aggregation of ensemble frames and batches → ensemble.npz arrays. */
export class EnsembleCollector {
  readonly rows: { tiles: Record<number, Float64Array>; global: Float64Array }[] = [];
  readonly seeds: string[] = [];
  readonly pixelSum: Float64Array;
  readonly pixelSumSq: Float64Array;

  constructor(readonly E: number, readonly W: number, readonly H: number, readonly runSeed: number) {
    this.pixelSum = new Float64Array(W * H * 3);
    this.pixelSumSq = new Float64Array(W * H * 3);
  }

  /** Append the E per-member rows of frame t (call after the frame's submit, before the next frame's stats pass). */
  async addFrame(k: RestirKernel, t: number): Promise<void> {
    this.addFrameStats(await readEnsStats(k), t);
  }
  addFrameStats(st: EnsembleFrameStats, t: number): void {
    for (let m = 0; m < this.E; m++) {
      const tiles: Record<number, Float64Array> = {};
      for (const l of ENS_LEVELS) {
        const { th, tw, data } = st.tiles[l];
        tiles[l] = Float64Array.from(data.subarray(m * th * tw * 3, (m + 1) * th * tw * 3));
      }
      this.rows.push({ tiles, global: Float64Array.from(st.global.subarray(3 * m, 3 * m + 3)) });
      this.seeds.push(`${this.runSeed}:${t}:${m}`);
    }
  }

  /** Read ensPixel (Σx, Σx² of this batch's frames and members) into the f64 sums and clear it. */
  async addBatchPixels(k: RestirKernel): Promise<void> {
    const res = k.resources;
    const px = new Float32Array(await readBuffer(k.device, res.ensPixel, this.W * this.H * 32));
    this.addPixelMoments(px);
    const enc = k.device.createCommandEncoder({ label: 'rs-ens-pixel-clear' });
    enc.clearBuffer(res.ensPixel);
    k.device.queue.submit([enc.finish()]);
  }
  addPixelMoments(px: Float32Array): void {
    for (let i = 0; i < this.W * this.H; i++) for (let c = 0; c < 3; c++) {
      this.pixelSum[3 * i + c] += px[8 * i + c];
      this.pixelSumSq[3 * i + c] += px[8 * i + 4 + c];
    }
  }

  /** Arrays of ensemble.npz: tiles16/32/64 (R, Th, Tw, 3), global (R, 3), pixel_sum / pixel_sumsq (H, W, 3), count,
   *  channels, height, width (float64; no masks in M4). */
  npzArrays(): NpzArray[] {
    const R = this.rows.length;
    const out: NpzArray[] = [];
    for (const l of ENS_LEVELS) {
      const th = Math.ceil(this.H / l), tw = Math.ceil(this.W / l);
      const a = new Float64Array(R * th * tw * 3);
      this.rows.forEach((r, i) => a.set(r.tiles[l], i * th * tw * 3));
      out.push({ name: `tiles${l}`, shape: [R, th, tw, 3], dtype: '<f8', data: a });
    }
    const g = new Float64Array(R * 3);
    this.rows.forEach((r, i) => g.set(r.global, 3 * i));
    out.push({ name: 'global', shape: [R, 3], dtype: '<f8', data: g });
    out.push({ name: 'pixel_sum', shape: [this.H, this.W, 3], dtype: '<f8', data: this.pixelSum });
    out.push({ name: 'pixel_sumsq', shape: [this.H, this.W, 3], dtype: '<f8', data: this.pixelSumSq });
    out.push({ name: 'count', shape: [], dtype: '<i8', data: BigInt64Array.from([BigInt(R)]) });
    out.push({ name: 'channels', shape: [3], dtype: '<U', data: ['R', 'G', 'B'] });
    out.push({ name: 'height', shape: [], dtype: '<i8', data: BigInt64Array.from([BigInt(this.H)]) });
    out.push({ name: 'width', shape: [], dtype: '<i8', data: BigInt64Array.from([BigInt(this.W)]) });
    return out;
  }
}
