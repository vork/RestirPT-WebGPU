// Batch accumulation for validation renders (plan §1.8 "Submit budget", §7.3 replicates).
// A batch of `spp` samples per pixel is rendered as sub-submits, each one dispatch of k samples over a row band,
// sized so every queue.submit stays within the GPU-time budget (target 50 ms, budget ≤ 100 ms, hard cap 200 ms).
// Samples accumulate in f32 on the GPU (a batch is small, ≤ a few hundred spp); after each batch the sum is read back,
// divided by spp in f64 (the batch mean, one replicate of the statistic) and added to a Float64 running total.
// Sample indices are run-global: batch b covers samples [b·spp, (b+1)·spp), so batches are i.i.d. under one run seed.
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
    const { device, width: W, height: H } = this;
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

    const bytes = await readBuffer(device, this.accum, W * H * 16);
    const counters = Array.from(new Uint32Array(await readBuffer(device, this.counters, 16)));
    const sum = new Float32Array(bytes);
    const mean = new Float32Array(W * H * 3);
    for (let i = 0; i < W * H; i++) {
      for (let c = 0; c < 3; c++) {
        const m = sum[4 * i + c] / spp; // f64 division, stored f32
        mean[3 * i + c] = m;
        this.total[3 * i + c] += m;
      }
    }
    this.batches++;
    const maxSubmitMs = Math.max(...submitMs);
    return {
      index, spp, mean, counters, submits: submitMs.length, submitMs, maxSubmitMs,
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
