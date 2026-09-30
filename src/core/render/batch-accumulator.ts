// Batch accumulation for validation renders (plan §1.8 "Submit budget", §7.3 replicates).
// A batch of `spp` samples per pixel is rendered as sub-submits, each one dispatch of k samples over a row band,
// sized so every queue.submit stays within the GPU-time budget (target 50 ms, budget ≤ 100 ms, hard cap 200 ms).
// Samples accumulate in f32 on the GPU (a batch is small, ≤ a few hundred spp); after each batch the sum is read back,
// divided by spp in f64 (the batch mean, one replicate of the statistic) and added to a Float64 running total.
// Sample indices are run-global: batch b covers samples [b·spp, (b+1)·spp), so batches are i.i.d. under one run seed.
// runBatchUnits (M4, restir-api.md §4.4, WP-E): the same batch contract for kernels that describe a frame as a list of
// work units (ReSTIR): units of consecutive frames are packed into submits by a learned ms/costHint model.
import { readBuffer } from '../gpu/readback.ts';

export interface SampleDispatch { sampleBase: number; sampleCount: number; rowBase: number; rows: number }

/** Encode ONE dispatch into `encoder`, accumulating into `accum` (vec4f per pixel, rgb = sum) and `counters`. */
export type SampleEncoder = (encoder: GPUCommandEncoder, d: SampleDispatch, accum: GPUBuffer, counters: GPUBuffer) => void;

export interface SubmitBudget {
  /** Target GPU time per submit (ms). */
  targetMs: number;
  /** Plan budget per submit (ms); exceeding it is recorded. */
  budgetMs: number;
  /** Hard cap (ms); exceeding it is an error in validation runs. */
  hardCapMs: number;
  /** Upper bound on samples per dispatch. */
  maxSamplesPerDispatch: number;
}

export const DEFAULT_SUBMIT_BUDGET: SubmitBudget = { targetMs: 50, budgetMs: 100, hardCapMs: 200, maxSamplesPerDispatch: 256 };

/** A unit of GPU work of one frame (structurally = restir/kernel.ts WorkUnit). */
export interface BatchWorkUnit { label: string; costHint: number; encode(enc: GPUCommandEncoder): void }

/** A frame-graph kernel as the unit packer sees it (restir-api.md §4.4). */
export interface UnitSource {
  /** Work units of run-global frame t (encoded in order; units of consecutive frames may share a submit). */
  frameUnits(t: number): BatchWorkUnit[];
  /** Called before the first unit of every submit is encoded (resets the kernel's RsDispatch ring). */
  beginSubmit(): void;
}

export interface UnitPackOptions {
  /** Upper bound on units per submit (the RsDispatch ring holds 512 slots; a unit may take several). Default 64. */
  maxUnitsPerSubmit?: number;
  /** Known ms per costHint per unit kind (carried across batches by the caller; updated in place). */
  rates?: Map<string, number>;
  /** Per-submit record (labels, estimate, measured wall ms) for diagnostics / adaptation. */
  onSubmit?: (s: SubmitRecord) => void;
}

export interface SubmitRecord { units: string[]; kinds: string[]; costs: number[]; estMs: number; ms: number }

/** Unit kind = label up to the first '[' / '(' / digit-free suffix (e.g. 'rs_initial[0+32][0]' → 'rs_initial'). */
export function unitKind(label: string): string {
  const m = /^[^[(:]+/.exec(label);
  return (m ? m[0] : label).trim();
}

/**
 * Greedy submit packing (pure; tested in the cpu lane): units are appended to the open submit while the estimated
 * GPU time (Σ rate(kind)·costHint) stays within `targetMs` and the unit count within `maxUnits`. A unit of an unknown
 * kind is estimated at `targetMs`, so it runs alone (or with known cheap units before it) until measured.
 */
export class UnitPacker {
  private open: { units: BatchWorkUnit[]; est: number } | undefined;
  constructor(readonly targetMs: number, readonly maxUnits: number, readonly rates: Map<string, number>) {}

  estimate(u: BatchWorkUnit): number {
    const r = this.rates.get(unitKind(u.label));
    return r === undefined ? this.targetMs : r * Math.max(u.costHint, 0);
  }

  /** Add a unit; returns the units of a submit that must be flushed BEFORE this unit (or undefined). */
  push(u: BatchWorkUnit): BatchWorkUnit[] | undefined {
    const c = this.estimate(u);
    let flushed: BatchWorkUnit[] | undefined;
    if (this.open && this.open.units.length > 0 && (this.open.est + c > this.targetMs || this.open.units.length >= this.maxUnits)) {
      flushed = this.open.units;
      this.open = undefined;
    }
    this.open ??= { units: [], est: 0 };
    this.open.units.push(u);
    this.open.est += c;
    return flushed;
  }

  /** Units of the open submit (and close it). */
  drain(): BatchWorkUnit[] | undefined {
    const u = this.open?.units;
    this.open = undefined;
    return u && u.length ? u : undefined;
  }

  /**
   * Learn from a measured submit: with exactly one unknown kind its rate is (ms − known estimate)/its cost; with none,
   * every kind in the submit is scaled by √(ms/est) (clamped to [¼, 4]) so shared submits converge without oscillating.
   */
  learn(units: BatchWorkUnit[], ms: number): void {
    const unknown = new Map<string, number>();
    let known = 0;
    for (const u of units) {
      const k = unitKind(u.label), r = this.rates.get(k);
      if (r === undefined) unknown.set(k, (unknown.get(k) ?? 0) + Math.max(u.costHint, 0));
      else known += r * Math.max(u.costHint, 0);
    }
    if (unknown.size === 1) {
      const [[k, cost]] = [...unknown];
      if (cost > 0) this.rates.set(k, Math.max(ms - known, 0.05 * ms) / cost);
      return;
    }
    if (unknown.size > 1 || !(known > 0)) return;
    const f = Math.sqrt(Math.min(4, Math.max(0.25, ms / known)));
    for (const k of new Set(units.map((u) => unitKind(u.label)))) this.rates.set(k, this.rates.get(k)! * f);
  }
}

export interface BatchResult {
  index: number;
  spp: number;
  /** Batch mean, RGB float32, row 0 = top (encodePFM-ready). */
  mean: Float32Array;
  /** Raw GPU counters of this batch (kernel-defined; emission: NaN/Inf samples, BVH overflow, BVH itercap, -). */
  counters: number[];
  submits: number;
  /** Per-submit wall time until onSubmittedWorkDone (upper bound of GPU time), ms. */
  submitMs: number[];
  maxSubmitMs: number;
  overBudget: number;
  overHardCap: number;
  wallMs: number;
}

export class BatchAccumulator {
  readonly accum: GPUBuffer;
  readonly counters: GPUBuffer;
  /** Σ over finished batches of the batch means (f64, RGB, row 0 = top). */
  readonly total: Float64Array;
  batches = 0;
  /** Adaptive work unit (samples per dispatch, rows per dispatch), carried across batches. */
  private k = 1;
  private rows: number;
  private readonly budget: SubmitBudget;

  constructor(readonly device: GPUDevice, readonly width: number, readonly height: number, budget: Partial<SubmitBudget> = {}) {
    this.budget = { ...DEFAULT_SUBMIT_BUDGET, ...budget };
    const n = width * height;
    this.accum = device.createBuffer({ label: 'batch-accum', size: n * 16, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
    this.counters = device.createBuffer({ label: 'batch-counters', size: 16, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
    this.total = new Float64Array(n * 3);
    this.rows = height;
  }

  /** Render batch `index` (`spp` samples per pixel) and read it back. */
  async runBatch(encode: SampleEncoder, spp: number, index = this.batches): Promise<BatchResult> {
    if (!(spp >= 1 && Number.isInteger(spp))) throw new Error(`bad spp ${spp}`);
    const t0 = performance.now();
    const { device, height: H } = this;
    const clear = device.createCommandEncoder({ label: 'batch-clear' });
    clear.clearBuffer(this.accum);
    clear.clearBuffer(this.counters);
    device.queue.submit([clear.finish()]);

    const submitMs: number[] = [];
    let done = 0;
    while (done < spp) {
      const k = Math.min(this.k, spp - done);
      const rowsPer = this.rows;
      let worst = 0;
      for (let r0 = 0; r0 < H; r0 += rowsPer) {
        const rows = Math.min(rowsPer, H - r0);
        const enc = device.createCommandEncoder({ label: `batch-${index}` });
        encode(enc, { sampleBase: index * spp + done, sampleCount: k, rowBase: r0, rows }, this.accum, this.counters);
        const ts = performance.now();
        device.queue.submit([enc.finish()]);
        await device.queue.onSubmittedWorkDone();
        const ms = performance.now() - ts;
        submitMs.push(ms);
        worst = Math.max(worst, ms);
      }
      done += k;
      this.adapt(worst, k, rowsPer);
    }

    return this.finishBatch(index, spp, submitMs, t0);
  }

  /**
   * Render batch `index` of a frame-graph kernel (restir-api.md §4.4): clears accum and counters, runs `frames` frames
   * t = index·frames + i, packs their work units into submits (target 50 ms, budget 100 ms, hard cap 200 ms; the ms per
   * costHint of every unit kind is learned from onSubmittedWorkDone and carried in `o.rates`), calls
   * `src.beginSubmit()` before each submit, and reads back like runBatch (batch mean = accumulated sum / frames).
   */
  async runBatchUnits(src: UnitSource, frames: number, index = this.batches, o: UnitPackOptions = {}): Promise<BatchResult> {
    if (!(frames >= 1 && Number.isInteger(frames))) throw new Error(`bad frames ${frames}`);
    const t0 = performance.now();
    const { device } = this;
    const clear = device.createCommandEncoder({ label: 'batch-clear' });
    clear.clearBuffer(this.accum);
    clear.clearBuffer(this.counters);
    device.queue.submit([clear.finish()]);

    const packer = new UnitPacker(this.budget.targetMs, o.maxUnitsPerSubmit ?? 64, o.rates ?? new Map());
    const submitMs: number[] = [];
    const submit = async (units: BatchWorkUnit[]) => {
      src.beginSubmit();
      const enc = device.createCommandEncoder({ label: `batch-${index}-units` });
      for (const u of units) u.encode(enc);
      const estMs = units.reduce((s, u) => s + packer.estimate(u), 0);
      const ts = performance.now();
      device.queue.submit([enc.finish()]);
      await device.queue.onSubmittedWorkDone();
      const ms = performance.now() - ts;
      submitMs.push(ms);
      o.onSubmit?.({ units: units.map((u) => u.label), kinds: units.map((u) => unitKind(u.label)), costs: units.map((u) => u.costHint), estMs, ms });
      packer.learn(units, ms);
    };
    for (let i = 0; i < frames; i++) {
      for (const u of src.frameUnits(index * frames + i)) {
        const full = packer.push(u);
        if (full) await submit(full);
      }
    }
    const rest = packer.drain();
    if (rest) await submit(rest);
    return this.finishBatch(index, frames, submitMs, t0);
  }

  /** Read back accum / counters of a finished batch: mean = sum / n (f64 division, stored f32), running total. */
  private async finishBatch(index: number, n: number, submitMs: number[], t0: number): Promise<BatchResult> {
    const { device, width: W, height: H } = this;
    const bytes = await readBuffer(device, this.accum, W * H * 16);
    const counters = Array.from(new Uint32Array(await readBuffer(device, this.counters, 16)));
    const sum = new Float32Array(bytes);
    const mean = new Float32Array(W * H * 3);
    for (let i = 0; i < W * H; i++) {
      for (let c = 0; c < 3; c++) {
        const m = sum[4 * i + c] / n; // f64 division, stored f32
        mean[3 * i + c] = m;
        this.total[3 * i + c] += m;
      }
    }
    this.batches++;
    const maxSubmitMs = submitMs.length ? Math.max(...submitMs) : 0;
    return {
      index, spp: n, mean, counters, submits: submitMs.length, submitMs, maxSubmitMs,
      overBudget: submitMs.filter((x) => x > this.budget.budgetMs).length,
      overHardCap: submitMs.filter((x) => x > this.budget.hardCapMs).length,
      wallMs: performance.now() - t0,
    };
  }

  /** Mean over all finished batches (f64 → f32), RGB, row 0 = top. */
  overallMean(): Float32Array {
    const out = new Float32Array(this.total.length);
    for (let i = 0; i < out.length; i++) out[i] = this.total[i] / Math.max(1, this.batches);
    return out;
  }

  /** Resize the work unit from the slowest submit of the last pass: samples first, row bands only at k = 1. */
  private adapt(worstMs: number, k: number, rows: number): void {
    const { targetMs, maxSamplesPerDispatch } = this.budget;
    const perUnit = Math.max(worstMs, 0.05) / (k * rows); // ms per (sample · row), wall-clock upper bound
    const units = targetMs / perUnit;                        // (samples · rows) that fit the target
    if (units >= this.height) {
      this.rows = this.height;
      this.k = Math.max(1, Math.min(maxSamplesPerDispatch, Math.floor(units / this.height), 2 * k));
    } else {
      this.k = 1;
      this.rows = Math.max(8, Math.floor(units / 8) * 8);
    }
  }

  destroy(): void { this.accum.destroy(); this.counters.destroy(); }
}
