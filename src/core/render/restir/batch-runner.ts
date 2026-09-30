// ReSTIR batch runner for Stage-B validation renders (restir-api.md §4.4, §2.10, §6.3; OWNER WP-E).
// WorkUnit → submit packing through BatchAccumulator.runBatchUnits (target 50 ms, budget 100 ms, hard cap 200 ms; the
// ms per costHint of every unit kind is learned from onSubmittedWorkDone), per-batch readback of the linear
// accumulation, the arena counters (RSC_*, queue headers, f_r) per batch, the adaptive tree chunk / row band, and the
// ensemble collector (ensStats rows per (frame, member), ensPixel per batch) that feeds ensemble.npz.
//
// Sequential validation (D14): t = run-global frame index, batch b covers frames [b·F, (b+1)·F), member 0; the batch
// mean is Σ_frames L / F (rs_finalize accumulates L = L1 + estimate per frame). Chunking never changes the result
// (U-RIS-2: tree chunks and row bands are bitwise invariant).
import { BatchAccumulator, unitKind, type BatchResult, type BatchWorkUnit, type SubmitBudget, type SubmitRecord, DEFAULT_SUBMIT_BUDGET } from '../batch-accumulator.ts';
import { readBuffer } from '../../gpu/readback.ts';
import { RSC, SC_NAMES, type RscName } from './layout.ts';
import type { RestirCounters, RestirKernel } from './kernel.ts';
import { ENS_LEVELS, ensStatsFloats } from './resources.ts';

/** PT-layout counters of rs_finalize (restir-api.md §4.4) + the arena's BVH counters. */
export interface RestirBatchCounters { nonFinite: number; bvhOverflow: number; bvhItercap: number; negative: number }

export interface RestirRunTotals {
  counters: RestirBatchCounters;
  /** Σ over batches of every RSC_* counter. */
  rsc: Record<RscName, number>;
  /** Σ SC_* histogram (header words 32–47), by name. */
  codes: Record<string, number>;
  /** Max over batches of each queue's overflow flag and counter (restir-api.md §2.6: overflow must be 0). */
  queueOverflow: number[];
  queueMaxCounter: number[];
  /** Replay fraction QUEUED / ACCEPTED over the whole run (PLAN §1.10, logged). */
  fr: number;
  submits: { total: number; maxMs: number; overBudget: number; overHardCap: number; perBatch: number[] };
  /** Chunking used per batch (row band 0 = whole atlas; tree chunk 0 = all S trees). */
  chunking: { rowBand: number; treeChunk: number }[];
  /** Learned ms per costHint per unit kind (last). */
  rates: Record<string, number>;
  /** Wall ms of the discarded chunking probe frame. */
  probeMs: number;
}

export interface RestirBatchRunnerOptions {
  budget?: Partial<SubmitBudget>;
  /** Frames per batch (sizes the ensemble staging buffer). */
  framesPerBatch: number;
  /** Batches (sizes the ensemble host rows). */
  batches: number;
  /** Probe frame index for the chunking calibration (discarded output; must lie outside the run's frames). */
  probeFrame?: number;
}

const zeroCounters = (): RestirBatchCounters => ({ nonFinite: 0, bvhOverflow: 0, bvhItercap: 0, negative: 0 });

/** Tree chunk / row band from learned unit rates (restir-api.md §4.4 "start from costHint and adapt"). Pure. */
export function chooseChunking(o: {
  rateInitial: number; atlasW: number; atlasH: number; maxBounces: number; trees: number; unitTargetMs: number;
}): { rowBand: number; treeChunk: number } {
  const perRowTree = o.rateInitial * o.atlasW * (o.maxBounces + 1);   // ms of one tree over one row
  if (!(perRowTree > 0)) return { rowBand: 0, treeChunk: 0 };
  const fullOneTree = perRowTree * o.atlasH;
  if (fullOneTree <= o.unitTargetMs) {
    const tc = Math.max(1, Math.min(o.trees, Math.floor(o.unitTargetMs / fullOneTree)));
    return { rowBand: 0, treeChunk: tc >= o.trees ? 0 : tc };
  }
  const rows = Math.max(8, Math.floor(o.unitTargetMs / perRowTree / 8) * 8);
  return { rowBand: rows >= o.atlasH ? 0 : rows, treeChunk: o.trees > 1 ? 1 : 0 };
}

/**
 * Ensemble collector (restir-api.md §2.10): after every frame a unit copies ensStats into a per-batch staging slot;
 * per batch the slots and ensPixel are read back into f64 host rows (run id r = t·E + m, seed label
 * `${runSeed}:${t}:${m}`) and ensPixel is cleared. Masks are not used in M4 (M = 0).
 */
export class EnsembleCollector {
  readonly E: number; readonly W: number; readonly H: number;
  readonly statsFloats: number;
  /** Floats of the levels + global part (masks excluded). */
  readonly usedFloats: number;
  readonly levels: { l: number; th: number; tw: number; offset: number }[];
  readonly globalOffset: number;
  readonly rows: number;
  readonly tiles: Float64Array[];
  readonly global: Float64Array;
  readonly pixelSum: Float64Array;
  readonly pixelSumSq: Float64Array;
  readonly seeds: string[] = [];
  private readonly staging: GPUBuffer;
  private filled = 0;
  private readonly frameIds: number[] = [];

  constructor(readonly device: GPUDevice, readonly kernel: RestirKernel, readonly framesPerBatch: number, batches: number, readonly runSeed: number) {
    const a = kernel.resources.alloc;
    this.E = a.members; this.W = a.memberW; this.H = a.memberH;
    this.statsFloats = ensStatsFloats(this.E, this.W, this.H);
    let off = 0;
    this.levels = ENS_LEVELS.map((l) => {
      const th = Math.ceil(this.H / l), tw = Math.ceil(this.W / l);
      const lv = { l, th, tw, offset: off };
      off += this.E * th * tw * 4;
      return lv;
    });
    this.globalOffset = off;
    this.usedFloats = off + this.E * 4;
    this.rows = framesPerBatch * batches * this.E;
    this.tiles = this.levels.map((lv) => new Float64Array(this.rows * lv.th * lv.tw * 3));
    this.global = new Float64Array(this.rows * 3);
    this.pixelSum = new Float64Array(this.W * this.H * 3);
    this.pixelSumSq = new Float64Array(this.W * this.H * 3);
    const bytes = this.usedFloats * 4 * framesPerBatch;
    if (bytes > 1 << 30) throw new Error(`EnsembleCollector: staging ${bytes} B > 1 GiB; lower the frames per batch`);
    this.staging = device.createBuffer({ label: 'rs-ens-staging', size: Math.max(16, bytes), usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC });
  }

  /** The unit appended after frame t's kernel units (copies this frame's ensStats into its staging slot). */
  copyUnit(t: number): BatchWorkUnit {
    const slot = this.filled++;
    if (slot >= this.framesPerBatch) throw new Error('EnsembleCollector: more frames than framesPerBatch in one batch');
    this.frameIds.push(t);
    const res = this.kernel.resources;
    return {
      label: `ens_copy[${t}]`, costHint: this.usedFloats,
      encode: (enc) => enc.copyBufferToBuffer(res.ensStats, 0, this.staging, slot * this.usedFloats * 4, this.usedFloats * 4),
    };
  }

  /** After a batch: decode the staged frames into host rows, add ensPixel to the f64 pixel sums, clear ensPixel. */
  async collectBatch(): Promise<void> {
    const res = this.kernel.resources;
    const n = this.filled;
    const f = new Float32Array(await readBuffer(this.device, this.staging, Math.max(4, n * this.usedFloats * 4)));
    for (let s = 0; s < n; s++) {
      const t = this.frameIds[s];
      const base = s * this.usedFloats;
      for (let m = 0; m < this.E; m++) {
        const r = this.seeds.length;
        if (r >= this.rows) throw new Error('EnsembleCollector: more rows than allocated');
        this.seeds.push(`${this.runSeed}:${t}:${m}`);
        this.levels.forEach((lv, li) => {
          const dst = this.tiles[li];
          for (let ty = 0; ty < lv.th; ty++) {
            for (let tx = 0; tx < lv.tw; tx++) {
              const src = base + lv.offset + ((m * lv.th + ty) * lv.tw + tx) * 4;
              const o = ((r * lv.th + ty) * lv.tw + tx) * 3;
              dst[o] = f[src]; dst[o + 1] = f[src + 1]; dst[o + 2] = f[src + 2];
            }
          }
        });
        const g = base + this.globalOffset + m * 4;
        this.global[r * 3] = f[g]; this.global[r * 3 + 1] = f[g + 1]; this.global[r * 3 + 2] = f[g + 2];
      }
    }
    const px = new Float32Array(await readBuffer(this.device, res.ensPixel, this.W * this.H * 32));
    for (let i = 0; i < this.W * this.H; i++) {
      for (let c = 0; c < 3; c++) {
        this.pixelSum[3 * i + c] += px[8 * i + c];
        this.pixelSumSq[3 * i + c] += px[8 * i + 4 + c];
      }
    }
    const enc = this.device.createCommandEncoder({ label: 'rs-ens-pixel-clear' });
    enc.clearBuffer(res.ensPixel);
    this.device.queue.submit([enc.finish()]);
    this.filled = 0;
    this.frameIds.length = 0;
  }

  get count(): number { return this.seeds.length; }

  destroy(): void { this.staging.destroy(); }
}

/** Drives a RestirKernel through BatchAccumulator.runBatchUnits (sequential or ensemble). */
export class RestirBatchRunner {
  readonly acc: BatchAccumulator;
  readonly totals: RestirRunTotals;
  readonly ensemble: EnsembleCollector | undefined;
  private readonly rates = new Map<string, number>();
  private readonly budget: SubmitBudget;
  private probed = false;

  constructor(readonly kernel: RestirKernel, readonly o: RestirBatchRunnerOptions & { runSeed: number }) {
    const a = kernel.resources.alloc;
    this.budget = { ...DEFAULT_SUBMIT_BUDGET, ...o.budget };
    // the accumulator is member-sized (sequential runs are E = 1; ensemble runs do not accumulate, RSF_ENSEMBLE)
    this.acc = new BatchAccumulator(kernel.device, a.memberW, a.memberH, o.budget);
    this.ensemble = a.members > 1 ? new EnsembleCollector(kernel.device, kernel, o.framesPerBatch, o.batches, o.runSeed) : undefined;
    this.totals = {
      counters: zeroCounters(),
      rsc: Object.fromEntries(Object.keys(RSC).map((k) => [k, 0])) as Record<RscName, number>,
      codes: Object.fromEntries(SC_NAMES.map((k) => [k, 0])),
      queueOverflow: [0, 0, 0, 0], queueMaxCounter: [0, 0, 0, 0], fr: 0,
      submits: { total: 0, maxMs: 0, overBudget: 0, overHardCap: 0, perBatch: [] },
      chunking: [], rates: {}, probeMs: 0,
    };
  }

  /**
   * Chunking probe: one discarded frame (scratch accum/counters, frame index `probeFrame`) with 1 tree per unit and
   * 128-row bands, run unit by unit to learn the ms per costHint of rs_initial; then choose the tree chunk and row band
   * so a unit is ≤ ½ the submit target. Resets the arena counters afterwards.
   */
  async probe(): Promise<void> {
    const k = this.kernel, d = k.device, a = k.resources.alloc;
    const t0 = performance.now();
    const accum = d.createBuffer({ label: 'rs-probe-accum', size: Math.max(16, a.memberW * a.memberH * 16), usage: GPUBufferUsage.STORAGE });
    const counters = d.createBuffer({ label: 'rs-probe-counters', size: 16, usage: GPUBufferUsage.STORAGE });
    const saved = { rowBand: k.rowBand, treeChunk: k.treeChunk };
    k.rowBand = Math.min(128, a.atlasH); k.treeChunk = 1;
    try {
      const units = k.frameUnits(this.o.probeFrame ?? 0x7ffffff0, { accum, counters });
      for (const u of units) {
        k.beginSubmit();
        const enc = d.createCommandEncoder({ label: 'rs-probe' });
        u.encode(enc);
        const ts = performance.now();
        d.queue.submit([enc.finish()]);
        await d.queue.onSubmittedWorkDone();
        const ms = performance.now() - ts;
        const kind = unitKind(u.label);
        const r = ms / Math.max(u.costHint, 1);
        this.rates.set(kind, Math.max(this.rates.get(kind) ?? 0, r));   // worst unit of the kind (conservative)
      }
    } finally {
      k.rowBand = saved.rowBand; k.treeChunk = saved.treeChunk;
      accum.destroy(); counters.destroy();
    }
    const ri = this.rates.get('rs_initial');
    if (ri !== undefined) {
      const c = chooseChunking({ rateInitial: ri, atlasW: a.atlasW, atlasH: a.atlasH, maxBounces: k.settings.maxBounces, trees: k.settings.trees, unitTargetMs: 0.5 * this.budget.targetMs });
      k.rowBand = c.rowBand; k.treeChunk = c.treeChunk;
    }
    await k.readCounters(true);
    if (this.ensemble) {
      const enc = d.createCommandEncoder({ label: 'rs-probe-ens-clear' });
      enc.clearBuffer(k.resources.ensPixel);
      d.queue.submit([enc.finish()]);
    }
    this.totals.probeMs = performance.now() - t0;
    this.probed = true;
  }

  /** Batch `index` of `frames` frames; returns the batch result (mean = Σ L / frames) and adds the counters. */
  async runBatch(frames: number, index: number): Promise<BatchResult & { restir: RestirCounters }> {
    if (!this.probed) await this.probe();
    const k = this.kernel;
    this.totals.chunking.push({ rowBand: k.rowBand, treeChunk: k.treeChunk });
    const src = {
      beginSubmit: () => k.beginSubmit(),
      frameUnits: (t: number): BatchWorkUnit[] => {
        const u: BatchWorkUnit[] = k.frameUnits(t, { accum: this.acc.accum, counters: this.acc.counters, ensemble: !!this.ensemble });
        if (this.ensemble) u.push(this.ensemble.copyUnit(t));
        return u;
      },
    };
    const records: SubmitRecord[] = [];
    const r = await this.acc.runBatchUnits(src, frames, index, { rates: this.rates, onSubmit: (s) => records.push(s) });
    const rc = await k.readCounters(true);
    if (this.ensemble) await this.ensemble.collectBatch();
    this.addCounters(r, rc);
    // adapt for the next batch: a submit above the budget halves the row band / tree chunk (results are invariant)
    if (r.maxSubmitMs > this.budget.budgetMs) {
      const a = k.resources.alloc;
      const rows = k.rowBand > 0 ? k.rowBand : a.atlasH;
      if (k.treeChunk > 1 || (k.treeChunk === 0 && k.settings.trees > 1)) k.treeChunk = Math.max(1, Math.floor((k.treeChunk || k.settings.trees) / 2));
      else k.rowBand = Math.max(8, Math.floor(rows / 2 / 8) * 8);
    }
    this.totals.rates = Object.fromEntries([...this.rates].map(([kk, v]) => [kk, Number(v.toPrecision(4))]));
    return { ...r, restir: rc };
  }

  private addCounters(r: BatchResult, rc: RestirCounters): void {
    const T = this.totals;
    T.counters.nonFinite += r.counters[0];
    T.counters.negative += r.counters[3];
    T.counters.bvhOverflow += r.counters[1] + rc.rsc.bvhOverflow;
    T.counters.bvhItercap += r.counters[2] + rc.rsc.bvhItercap;
    for (const key of Object.keys(RSC) as RscName[]) T.rsc[key] += rc.rsc[key];
    rc.codes.forEach((v, i) => { T.codes[SC_NAMES[i]] += v; });
    rc.queues.forEach((q, i) => { T.queueOverflow[i] = Math.max(T.queueOverflow[i], q.overflow); T.queueMaxCounter[i] = Math.max(T.queueMaxCounter[i], q.counter); });
    T.fr = T.rsc.accepted > 0 ? T.rsc.queued / T.rsc.accepted : 0;
    T.submits.total += r.submits; T.submits.maxMs = Math.max(T.submits.maxMs, r.maxSubmitMs);
    T.submits.overBudget += r.overBudget; T.submits.overHardCap += r.overHardCap; T.submits.perBatch.push(r.submits);
  }

  /**
   * T15 / §2.6 invariants of the run so far: NaN/Inf = 0, negatives = 0, BVH overflow/itercap = 0, the RSC_* error
   * counters = 0, every queue overflow = 0, BASE_JDEN_INVALID / nCand ≤ 1e-5 (nCand ≈ candidates, not tracked here:
   * reported), no submit above the hard cap.
   */
  invariantErrors(): string[] {
    const T = this.totals, e: string[] = [];
    if (T.counters.nonFinite) e.push(`${T.counters.nonFinite} NaN/Inf radiance values (T15)`);
    if (T.counters.negative) e.push(`${T.counters.negative} negative radiance values (T15)`);
    if (T.counters.bvhOverflow || T.counters.bvhItercap) e.push(`BVH overflow ${T.counters.bvhOverflow}, iteration cap ${T.counters.bvhItercap}`);
    for (const key of ['candNonFinite', 'shiftNonFinite', 'pendingLeft', 'slotMismatch', 'wNonFinite'] as const) {
      if (T.rsc[key]) e.push(`RSC ${key} = ${T.rsc[key]} (must be 0, restir-api.md §2.6)`);
    }
    T.queueOverflow.forEach((v, q) => { if (v) e.push(`queue ${q} overflow (counter ${T.queueMaxCounter[q]})`); });
    if (T.submits.overHardCap) e.push(`${T.submits.overHardCap} submits above the ${this.budget.hardCapMs} ms hard cap`);
    return e;
  }

  destroy(): void {
    this.acc.destroy();
    this.ensemble?.destroy();
  }
}
