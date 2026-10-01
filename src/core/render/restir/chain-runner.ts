// Temporal validation chains (restir-temporal-api.md §3.8, §4.4 TD26, §5, §6.4, TD25; PLAN §7.3 dynamic statistics).
// OWNER T-E.
//
// A chain is one atlas member followed through the frames 0…T−1 of a scene script from a history reset at t = 0
// (TD25: chain id c = memberBase + atlas member, RsDispatch.t = t). One BATCH renders E chains side by side in the
// ensemble atlas (members = E); R chains of a unit = R/E batches with memberBase = chainBase + b·E. Every frame:
//   spec.beforeFrame(t) (env map swaps) → kernel.advance(state_t) → kernel.frameUnits(t) (+ harness units) →
//   submits packed by BatchAccumulator's UnitPacker and CLOSED at the frame boundary (TD26: advance() writes per-frame
//   uniforms and light slots with queue.writeBuffer, which would run ahead of an earlier frame's unsubmitted units).
// Harness units (not kernel passes; labels start with a stable kind, restir-api.md E1):
//   chain_copy[t]  test frames: copyTextureToBuffer of the atlas rsFrame (linear L = L1 + estimate, rgba32f) into a
//                  MAP_READ slot; after the batch the host reduces each member image in f64 into the per-chain rows of
//                  ensemble.npz (tiles 16/32/64, global, mask regions, per-pixel Σx / Σx² over chains).
//   chain_accum[t] rung 3.5: adds rsFrame into an atlas-sized f32 accumulator over the frames [from, to] (standalone
//                  8×8 pass); after the batch its member images / L are reduced exactly like a test frame ("avg").
// The host reduction replaces the ensStats readback of M4 (restir-api.md E3) for chains: masks change per test frame
// and the per-pixel moments must be per frame, not per batch (restir-temporal-api.md Changelog E-1).
import { UnitPacker, unitKind, DEFAULT_SUBMIT_BUDGET, type BatchWorkUnit, type SubmitBudget } from '../batch-accumulator.ts';
import { chooseChunking } from './batch-runner.ts';
import { RSC, SC_NAMES, TS_WORDS, TSW, type RscName } from './layout.ts';
import type { RestirCounters, RestirKernel, WorkUnit } from './kernel.ts';
import type { RestirAdvance, RestirFrameState } from './frame-state.ts';
import { ENS_LEVELS } from './resources.ts';

/** Region masks of one test frame (member-local, row 0 = top): bit i of bits[y·W + x] ⇔ the pixel is in region i. */
export interface ChainMasks { names: string[]; bits: Uint16Array }

export interface ChainSpec {
  /** Frames per chain (0…frames−1). */
  frames: number;
  /** Frames whose per-chain images are reduced (ensemble rows). */
  testFrames: number[];
  /** Scene state of frame t (lights, camera, env params / map id). t = 0 is forced to a reset. */
  state(t: number): RestirFrameState;
  /** Called before advance(t) (e.g. RestirKernel.setEnvironment on an env map swap). */
  beforeFrame?(t: number): Promise<void> | void;
  /** Region masks of test frame t (≤ 16 regions). */
  masks?(t: number): ChainMasks | undefined;
  /** Rung 3.5: per-chain mean over frames [from, to] (inclusive), reduced as frame "avg". */
  average?: { from: number; to: number };
  /** Called after frame t's GPU work completed (every submit of the frame done; before advance(t + 1)): per-frame
   *  readback of tState / reservoirs / counters (T-B's T6(b) robust-mode checks). Unset: no behaviour change. */
  afterFrame?(t: number, k: RestirKernel): Promise<void> | void;
}

/** Per-chain rows of one reduced frame (one batch = E rows). Sums, not means (compare.py ensemble.npz format). */
export interface ChainFrameRows {
  frame: number | 'avg';
  E: number; W: number; H: number;
  /** tiles[l]: E·th·tw·3 tile SUMS (level l ∈ 16, 32, 64; edge tiles partial). */
  tiles: Record<number, { th: number; tw: number; data: Float64Array }>;
  /** E·3 image sums. */
  global: Float64Array;
  /** E·M·3 mask-region sums, the region pixel counts and names (M = 0 without masks). */
  masks: Float64Array; maskPixels: number[]; maskNames: string[];
  /** Σ over the E members of x and x² per member-local pixel (W·H·3). */
  pixelSum: Float64Array; pixelSumSq: Float64Array;
  /** Non-finite values replaced by 0 in the reduction (finalize already zeroes and counts them; must be 0). */
  nonFinite: number;
}

/** Per-frame record of a chain batch (T16: resets exactly where the script demands them). */
export interface ChainFrameRecord { t: number; histValid: boolean; flags: number; reasons: string[]; temporalUnits: number; units: number; submits: number; ms: number }

export interface ChainTotals {
  batches: number; chains: number; frames: number;
  counters: { nonFinite: number; negative: number };
  rsc: Record<RscName, number>;
  codes: Record<string, number>;
  queueOverflow: number[]; queueMaxCounter: number[];
  submits: { total: number; maxMs: number; overBudget: number; overHardCap: number };
  /** Frame records of the first batch (every batch runs the same script; resets are checked on all). */
  frameRecords: ChainFrameRecord[];
  /** Frames whose history validity differed from batch 0's (must stay empty: the script decides resets). */
  resetMismatches: string[];
  rates: Record<string, number>;
  probeMs: number;
  batchMs: number[];
}

const zeroRsc = () => Object.fromEntries(Object.keys(RSC).map((k) => [k, 0])) as Record<RscName, number>;

/**
 * Reduce the atlas image (rgba f32, row-major, atlas row 0 = top) into per-member rows (pure; cpu-tested).
 * Member m occupies the tile ((m mod cols)·W, ⌊m/cols⌋·H).
 */
export function reduceAtlas(img: Float32Array, o: { atlasW: number; W: number; H: number; E: number; memberCols: number; frame: number | 'avg'; scale?: number; masks?: ChainMasks }): ChainFrameRows {
  const { W, H, E, memberCols, atlasW } = o;
  const s = o.scale ?? 1;
  const tiles: ChainFrameRows['tiles'] = {};
  for (const l of ENS_LEVELS) { const th = Math.ceil(H / l), tw = Math.ceil(W / l); tiles[l] = { th, tw, data: new Float64Array(E * th * tw * 3) }; }
  const M = o.masks?.names.length ?? 0;
  if (M > 16) throw new Error(`reduceAtlas: ${M} mask regions > 16`);
  const global = new Float64Array(E * 3), masks = new Float64Array(E * M * 3);
  const maskPixels = new Array<number>(M).fill(0);
  if (o.masks) {
    if (o.masks.bits.length !== W * H) throw new Error(`reduceAtlas: mask size ${o.masks.bits.length} != ${W}x${H}`);
    for (let i = 0; i < W * H; i++) for (let r = 0; r < M; r++) if (o.masks.bits[i] & (1 << r)) maskPixels[r]++;
  }
  const pixelSum = new Float64Array(W * H * 3), pixelSumSq = new Float64Array(W * H * 3);
  const t16 = tiles[16], t32 = tiles[32], t64 = tiles[64];
  let nonFinite = 0;
  for (let m = 0; m < E; m++) {
    const ox = (m % memberCols) * W, oy = Math.floor(m / memberCols) * H;
    let g0 = 0, g1 = 0, g2 = 0;
    for (let y = 0; y < H; y++) {
      const row = (oy + y) * atlasW + ox;
      const r16 = (m * t16.th + (y >> 4)) * t16.tw, r32 = (m * t32.th + (y >> 5)) * t32.tw, r64 = (m * t64.th + (y >> 6)) * t64.tw;
      for (let x = 0; x < W; x++) {
        const src = 4 * (row + x);
        let r = img[src] * s, g = img[src + 1] * s, b = img[src + 2] * s;
        if (!Number.isFinite(r) || !Number.isFinite(g) || !Number.isFinite(b)) { nonFinite++; r = 0; g = 0; b = 0; }
        g0 += r; g1 += g; g2 += b;
        let k = 3 * (r16 + (x >> 4)); t16.data[k] += r; t16.data[k + 1] += g; t16.data[k + 2] += b;
        k = 3 * (r32 + (x >> 5)); t32.data[k] += r; t32.data[k + 1] += g; t32.data[k + 2] += b;
        k = 3 * (r64 + (x >> 6)); t64.data[k] += r; t64.data[k + 1] += g; t64.data[k + 2] += b;
        const p = 3 * (y * W + x);
        pixelSum[p] += r; pixelSum[p + 1] += g; pixelSum[p + 2] += b;
        pixelSumSq[p] += r * r; pixelSumSq[p + 1] += g * g; pixelSumSq[p + 2] += b * b;
        if (M) {
          const bits = o.masks!.bits[y * W + x];
          if (bits) for (let q = 0; q < M; q++) if (bits & (1 << q)) { const o3 = 3 * (m * M + q); masks[o3] += r; masks[o3 + 1] += g; masks[o3 + 2] += b; }
        }
      }
    }
    global[3 * m] = g0; global[3 * m + 1] = g1; global[3 * m + 2] = g2;
  }
  return { frame: o.frame, E, W, H, tiles, global, masks, maskPixels, maskNames: o.masks?.names ?? [], pixelSum, pixelSumSq, nonFinite };
}

/** Rows of one reduced frame across batches → the ensemble.npz arrays (restir-api.md §2.10 / compare.py README). */
export class ChainFrameCollector {
  readonly tiles: Record<number, Float64Array[]> = { 16: [], 32: [], 64: [] };
  readonly global: Float64Array[] = [];
  readonly masks: Float64Array[] = [];
  readonly seeds: string[] = [];
  pixelSum: Float64Array | undefined;
  pixelSumSq: Float64Array | undefined;
  shape: { W: number; H: number; tiles: Record<number, { th: number; tw: number }>; maskNames: string[]; maskPixels: number[] } | undefined;
  nonFinite = 0;

  add(r: ChainFrameRows, chainIds: number[], runSeed: number): void {
    if (chainIds.length !== r.E) throw new Error('ChainFrameCollector: chain ids / rows mismatch');
    const tshape = Object.fromEntries(ENS_LEVELS.map((l) => [l, { th: r.tiles[l].th, tw: r.tiles[l].tw }]));
    if (!this.shape) this.shape = { W: r.W, H: r.H, tiles: tshape, maskNames: r.maskNames, maskPixels: r.maskPixels };
    else if (this.shape.W !== r.W || this.shape.H !== r.H || this.shape.maskNames.join('|') !== r.maskNames.join('|')) throw new Error('ChainFrameCollector: shape changed between batches');
    for (const l of ENS_LEVELS) this.tiles[l].push(r.tiles[l].data);
    this.global.push(r.global);
    this.masks.push(r.masks);
    for (const c of chainIds) this.seeds.push(`${runSeed}:c${c}`);
    if (!this.pixelSum) { this.pixelSum = new Float64Array(r.pixelSum.length); this.pixelSumSq = new Float64Array(r.pixelSum.length); }
    for (let i = 0; i < r.pixelSum.length; i++) { this.pixelSum[i] += r.pixelSum[i]; this.pixelSumSq![i] += r.pixelSumSq[i]; }
    this.nonFinite += r.nonFinite;
  }

  get count(): number { return this.seeds.length; }

  /** ensemble.npz arrays (float64 sums; `masks` only with M > 0). */
  npzArrays(): { name: string; shape: number[]; dtype: '<f8' | '<i8' | '<U'; data: Float64Array | BigInt64Array | string[] }[] {
    const s = this.shape;
    if (!s) throw new Error('ChainFrameCollector: no rows');
    const R = this.count;
    const cat = (parts: Float64Array[]) => { const n = parts.reduce((a, p) => a + p.length, 0); const o = new Float64Array(n); let k = 0; for (const p of parts) { o.set(p, k); k += p.length; } return o; };
    const out: ReturnType<ChainFrameCollector['npzArrays']> = [];
    for (const l of ENS_LEVELS) out.push({ name: `tiles${l}`, shape: [R, s.tiles[l].th, s.tiles[l].tw, 3], dtype: '<f8', data: cat(this.tiles[l]) });
    out.push({ name: 'global', shape: [R, 3], dtype: '<f8', data: cat(this.global) });
    const M = s.maskNames.length;
    if (M) {
      out.push({ name: 'masks', shape: [R, M, 3], dtype: '<f8', data: cat(this.masks) });
      out.push({ name: 'mask_pixels', shape: [M], dtype: '<f8', data: Float64Array.from(s.maskPixels) });
      out.push({ name: 'mask_names', shape: [M], dtype: '<U', data: s.maskNames });
    }
    out.push({ name: 'pixel_sum', shape: [s.H, s.W, 3], dtype: '<f8', data: this.pixelSum! });
    out.push({ name: 'pixel_sumsq', shape: [s.H, s.W, 3], dtype: '<f8', data: this.pixelSumSq! });
    out.push({ name: 'count', shape: [], dtype: '<i8', data: BigInt64Array.from([BigInt(R)]) });
    out.push({ name: 'channels', shape: [3], dtype: '<U', data: ['R', 'G', 'B'] });
    out.push({ name: 'height', shape: [], dtype: '<i8', data: BigInt64Array.from([BigInt(s.H)]) });
    out.push({ name: 'width', shape: [], dtype: '<i8', data: BigInt64Array.from([BigInt(s.W)]) });
    return out;
  }
}

const ACCUM_WGSL = /* wgsl */ `
@group(0) @binding(0) var src: texture_2d<f32>;
@group(0) @binding(1) var<storage, read_write> acc: array<vec4f>;
@group(0) @binding(2) var<uniform> dims: vec4u;
@compute @workgroup_size(8, 8, 1)
fn chain_accum(@builtin(global_invocation_id) g: vec3u) {
  if (g.x >= dims.x || g.y >= dims.y) { return; }
  let i = g.y * dims.x + g.x;
  acc[i] = acc[i] + textureLoad(src, vec2i(g.xy), 0);
}`;

export interface ChainRunnerOptions {
  runSeed: number;
  budget?: Partial<SubmitBudget>;
  /** Discarded probe frame (outside every chain's frame range; not advanced ⇒ a reset frame, restir-api.md E2). */
  probeFrame?: number;
  /** Counters that are expected by construction in this run (e.g. RSC_T_PENDING_LEFT under the N1-mixed plant, which
   *  skips rs_refresh_inv: §3.5; Changelog E-15); reported, not errors. */
  expectedCounters?: RscName[];
}

/**
 * Runs chain batches on a kernel whose view is an ensemble atlas (members = E, temporal settings). The caller sets the
 * view once (camera irrelevant: advance() writes it per frame); runBatch sets memberBase = chainBase + b·E.
 */
export class ChainRunner {
  readonly totals: ChainTotals;
  private readonly rates = new Map<string, number>();
  private readonly budget: SubmitBudget;
  private probed = false;
  private readonly accum: GPUBuffer;
  private readonly counters: GPUBuffer;
  private accumPipe: GPUComputePipeline | undefined;
  private accumBuf: GPUBuffer | undefined;
  private accumBind: GPUBindGroup | undefined;
  private dimsBuf: GPUBuffer | undefined;

  constructor(readonly kernel: RestirKernel, readonly o: ChainRunnerOptions) {
    const d = kernel.device;
    this.budget = { ...DEFAULT_SUBMIT_BUDGET, ...o.budget };
    this.accum = d.createBuffer({ label: 'chain-accum-unused', size: 16, usage: GPUBufferUsage.STORAGE });
    this.counters = d.createBuffer({ label: 'chain-counters', size: 16, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
    this.totals = {
      batches: 0, chains: 0, frames: 0, counters: { nonFinite: 0, negative: 0 }, rsc: zeroRsc(),
      codes: Object.fromEntries(SC_NAMES.map((k) => [k, 0])), queueOverflow: [0, 0, 0, 0], queueMaxCounter: [0, 0, 0, 0],
      submits: { total: 0, maxMs: 0, overBudget: 0, overHardCap: 0 }, frameRecords: [], resetMismatches: [], rates: {}, probeMs: 0, batchMs: [],
    };
  }

  private get alloc() { return this.kernel.resources.alloc; }

  /** Chunking probe (restir-api.md E2): one discarded, non-advanced frame unit by unit → tree chunk / row band. */
  async probe(): Promise<void> {
    const k = this.kernel, d = k.device, a = this.alloc;
    const t0 = performance.now();
    const saved = { rowBand: k.rowBand, treeChunk: k.treeChunk };
    k.rowBand = Math.min(128, a.atlasH); k.treeChunk = 1;
    try {
      for (const u of k.frameUnits(this.o.probeFrame ?? 0x7ffffff0, { accum: this.accum, counters: this.counters, ensemble: a.members > 1 })) {
        k.beginSubmit();
        const enc = d.createCommandEncoder({ label: 'chain-probe' });
        u.encode(enc);
        const ts = performance.now();
        d.queue.submit([enc.finish()]);
        await d.queue.onSubmittedWorkDone();
        const kind = unitKind(u.label);
        this.rates.set(kind, Math.max(this.rates.get(kind) ?? 0, (performance.now() - ts) / Math.max(u.costHint, 1)));
      }
    } finally { k.rowBand = saved.rowBand; k.treeChunk = saved.treeChunk; }
    const ri = this.rates.get('rs_initial');
    if (ri !== undefined) {
      const c = chooseChunking({ rateInitial: ri, atlasW: a.atlasW, atlasH: a.atlasH, maxBounces: k.settings.maxBounces, trees: k.settings.trees, unitTargetMs: 0.5 * this.budget.targetMs });
      k.rowBand = c.rowBand; k.treeChunk = c.treeChunk;
    }
    await k.readCounters(true);
    await this.clearCounters();
    this.totals.probeMs = performance.now() - t0;
    this.probed = true;
  }

  private async clearCounters(): Promise<number[]> {
    const d = this.kernel.device;
    const st = d.createBuffer({ size: 16, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    const enc = d.createCommandEncoder({ label: 'chain-counters' });
    enc.copyBufferToBuffer(this.counters, 0, st, 0, 16);
    enc.clearBuffer(this.counters);
    d.queue.submit([enc.finish()]);
    await st.mapAsync(GPUMapMode.READ);
    const c = Array.from(new Uint32Array(st.getMappedRange()));
    st.unmap(); st.destroy();
    return c;
  }

  private async ensureAccum(): Promise<void> {
    if (this.accumPipe) return;
    const d = this.kernel.device, a = this.alloc;
    const module = d.createShaderModule({ label: 'chain-accum', code: ACCUM_WGSL });
    this.accumPipe = await d.createComputePipelineAsync({ label: 'chain_accum', layout: 'auto', compute: { module, entryPoint: 'chain_accum' } });
    this.accumBuf = d.createBuffer({ label: 'chain-accum', size: a.atlasW * a.atlasH * 16, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
    this.dimsBuf = d.createBuffer({ label: 'chain-accum-dims', size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    d.queue.writeBuffer(this.dimsBuf, 0, Uint32Array.from([a.atlasW, a.atlasH, 0, 0]));
    this.accumBind = d.createBindGroup({ layout: this.accumPipe.getBindGroupLayout(0), entries: [
      { binding: 0, resource: this.kernel.resources.frameTex.createView() }, { binding: 1, resource: { buffer: this.accumBuf } }, { binding: 2, resource: { buffer: this.dimsBuf } },
    ] });
  }

  /**
   * Batch b: chains chainBase + b·E + m (m < E) through frames 0…spec.frames−1; `onFrame` receives the reduced rows of
   * every test frame (and of "avg" for rung 3.5) with the chain ids of the rows.
   */
  async runBatch(spec: ChainSpec, b: number, chainBase: number, onFrame: (rows: ChainFrameRows, chainIds: number[]) => void): Promise<ChainFrameRecord[]> {
    const k = this.kernel, d = k.device;
    const view = k.currentView;
    if (!view) throw new Error('ChainRunner: kernel.setView() first');
    const a0 = this.alloc;
    const E = a0.members;
    k.setView({ ...view, memberBase: chainBase + b * E });   // same allocation; frame state invalidated ⇒ t = 0 resets
    if (!this.probed) await this.probe();
    const a = this.alloc;
    const tBatch = performance.now();
    const tests = [...new Set(spec.testFrames)].filter((t) => t < spec.frames).sort((x, y) => x - y);
    const bytes = a.atlasW * a.atlasH * 16;
    const slots = new Map<number, GPUBuffer>();
    for (const t of tests) slots.set(t, d.createBuffer({ label: `chain-copy-f${t}`, size: bytes, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST }));
    const avg = spec.average;
    if (avg) {
      if (!(avg.from >= 0 && avg.to < spec.frames && avg.from <= avg.to)) throw new Error(`ChainRunner: average window ${avg.from}..${avg.to} outside 0..${spec.frames - 1}`);
      await this.ensureAccum();
      const enc = d.createCommandEncoder({ label: 'chain-accum-clear' });
      enc.clearBuffer(this.accumBuf!);
      d.queue.submit([enc.finish()]);
    }
    { // the ensemble stage's ensPixel is not used by chains; keep it from growing across batches
      const enc = d.createCommandEncoder({ label: 'chain-ens-clear' });
      enc.clearBuffer(k.resources.ensPixel);
      d.queue.submit([enc.finish()]);
    }
    const packer = new UnitPacker(this.budget.targetMs, 64, this.rates);
    const records: ChainFrameRecord[] = [];
    const T = this.totals;
    const submit = async (units: BatchWorkUnit[]): Promise<number> => {
      k.beginSubmit();
      const enc = d.createCommandEncoder({ label: `chain-b${b}` });
      for (const u of units) u.encode(enc);
      const ts = performance.now();
      d.queue.submit([enc.finish()]);
      await d.queue.onSubmittedWorkDone();
      const ms = performance.now() - ts;
      packer.learn(units, ms);
      T.submits.total++; T.submits.maxMs = Math.max(T.submits.maxMs, ms);
      if (ms > this.budget.budgetMs) T.submits.overBudget++;
      if (ms > this.budget.hardCapMs) T.submits.overHardCap++;
      return ms;
    };
    for (let t = 0; t < spec.frames; t++) {
      await spec.beforeFrame?.(t);
      const st = spec.state(t);
      const adv: RestirAdvance = k.advance(t === 0 ? { ...st, t, reset: true } : { ...st, t });
      const units: WorkUnit[] = k.frameUnits(t, { accum: this.accum, counters: this.counters, ensemble: E > 1 });
      const temporalUnits = units.filter((u) => /^rs_(t_|refresh_)/.test(u.label)).length;
      const slot = slots.get(t);
      if (slot) {
        units.push({
          label: `chain_copy[${t}]`, costHint: a.atlasW * a.atlasH,
          encode: (enc) => enc.copyTextureToBuffer({ texture: k.resources.frameTex }, { buffer: slot, bytesPerRow: a.atlasW * 16, rowsPerImage: a.atlasH }, [a.atlasW, a.atlasH]),
        });
      }
      if (avg && t >= avg.from && t <= avg.to) {
        units.push({
          label: `chain_accum[${t}]`, costHint: a.atlasW * a.atlasH,
          encode: (enc) => {
            const p = enc.beginComputePass({ label: 'chain_accum' });
            p.setPipeline(this.accumPipe!); p.setBindGroup(0, this.accumBind!);
            p.dispatchWorkgroups(Math.ceil(a.atlasW / 8), Math.ceil(a.atlasH / 8)); p.end();
          },
        });
      }
      let submits = 0, ms = 0;
      for (const u of units) { const full = packer.push(u); if (full) { ms += await submit(full); submits++; } }
      const rest = packer.drain();                          // TD26: a submit never spans a frame boundary
      if (rest) { ms += await submit(rest); submits++; }
      records.push({ t, histValid: adv.histValid, flags: adv.flags, reasons: adv.reasons, temporalUnits, units: units.length, submits, ms });
      await spec.afterFrame?.(t, k);
    }
    // ---- readback + counters
    const rc: RestirCounters = await k.readCounters(true);
    const fc = await this.clearCounters();
    T.counters.nonFinite += fc[0]; T.counters.negative += fc[3];
    for (const key of Object.keys(RSC) as RscName[]) T.rsc[key] += rc.rsc[key];
    rc.codes.forEach((v, i) => { T.codes[SC_NAMES[i]] += v; });
    rc.queues.forEach((q, i) => { T.queueOverflow[i] = Math.max(T.queueOverflow[i], q.overflow); T.queueMaxCounter[i] = Math.max(T.queueMaxCounter[i], q.counter); });
    const chainIds = Array.from({ length: E }, (_, m) => chainBase + b * E + m);
    const red = { atlasW: a.atlasW, W: a.memberW, H: a.memberH, E, memberCols: a.memberCols };
    for (const t of tests) {
      const buf = slots.get(t)!;
      await buf.mapAsync(GPUMapMode.READ);
      const img = new Float32Array(buf.getMappedRange());
      onFrame(reduceAtlas(img, { ...red, frame: t, masks: spec.masks?.(t) }), chainIds);
      buf.unmap(); buf.destroy();
    }
    if (avg) {
      const st = d.createBuffer({ size: bytes, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
      const enc = d.createCommandEncoder({ label: 'chain-accum-read' });
      enc.copyBufferToBuffer(this.accumBuf!, 0, st, 0, bytes);
      d.queue.submit([enc.finish()]);
      await st.mapAsync(GPUMapMode.READ);
      onFrame(reduceAtlas(new Float32Array(st.getMappedRange()), { ...red, frame: 'avg', scale: 1 / (avg.to - avg.from + 1) }), chainIds);
      st.unmap(); st.destroy();
    }
    // T16: every batch runs the same script, so its reset pattern must equal batch 0's
    if (!T.frameRecords.length) T.frameRecords = records;
    else records.forEach((r, i) => { if (T.frameRecords[i] && r.histValid !== T.frameRecords[i].histValid) T.resetMismatches.push(`batch ${b} frame ${r.t}: histValid ${r.histValid} (batch 0: ${T.frameRecords[i].histValid})`); });
    T.batches++; T.chains += E; T.frames += spec.frames;
    T.batchMs.push(performance.now() - tBatch);
    T.rates = Object.fromEntries([...this.rates].map(([kk, v]) => [kk, Number(v.toPrecision(4))]));
    return records;
  }

  /**
   * Harness mode for M_disocc (restir-temporal-api.md §6.4): the production T1 flags of the LAST frame run (tState
   * word `flags` per atlas pixel; the view must be E = 1 with jitter off). Returns 1 where T1 found no valid q′ on a hit
   * pixel (TS_DISOCC without TS_NO_HIST), 0 elsewhere (member-local, row 0 = top).
   */
  async disocclusionMask(): Promise<Uint8Array> {
    const k = this.kernel, a = this.alloc;
    if (a.members !== 1) throw new Error('disocclusionMask: E = 1 view required');
    const words = await k.readTemporalState();
    const out = new Uint8Array(a.memberW * a.memberH);
    const TS_DISOCC = 2, TS_NO_HIST = 512, TS_BG = 16384;
    for (let y = 0; y < a.memberH; y++) for (let x = 0; x < a.memberW; x++) {
      const f = words[TS_WORDS * (y * a.atlasW + x) + TSW.flags];
      out[y * a.memberW + x] = (f & TS_DISOCC) && !(f & TS_NO_HIST) && !(f & TS_BG) ? 1 : 0;
    }
    return out;
  }

  /** T15 / §2.6 / §2.1 invariants of the chain run so far (all RSC error counters incl. the temporal ones, q0–q2). */
  invariantErrors(): string[] {
    const T = this.totals, e: string[] = [];
    if (T.counters.nonFinite) e.push(`${T.counters.nonFinite} NaN/Inf radiance values (T15)`);
    if (T.counters.negative) e.push(`${T.counters.negative} negative radiance values (T15)`);
    if (T.rsc.bvhOverflow || T.rsc.bvhItercap) e.push(`BVH overflow ${T.rsc.bvhOverflow}, iteration cap ${T.rsc.bvhItercap}`);
    for (const key of ['candNonFinite', 'shiftNonFinite', 'pendingLeft', 'slotMismatch', 'wNonFinite', 'tNonFinite', 'tPendingLeft'] as const) {
      if (T.rsc[key] && !this.o.expectedCounters?.includes(key)) e.push(`RSC ${key} = ${T.rsc[key]} (must be 0)`);
    }
    T.queueOverflow.forEach((v, q) => { if (v) e.push(`queue ${q} overflow (counter ${T.queueMaxCounter[q]})`); });
    if (T.submits.overHardCap) e.push(`${T.submits.overHardCap} submits above the ${this.budget.hardCapMs} ms hard cap`);
    if (T.resetMismatches.length) e.push(`history resets differ between batches: ${T.resetMismatches.slice(0, 4).join('; ')}`);
    return e;
  }

  destroy(): void {
    this.accum.destroy(); this.counters.destroy();
    this.accumBuf?.destroy(); this.dimsBuf?.destroy();
  }
}
