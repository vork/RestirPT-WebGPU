// A-SVGF-lite denoiser (PLAN §5 M5.5; docs/decisions/denoiser.md). App-only: built by the interactive Renderer, never by
// a validation runner (DN4: every live instance is in render/denoise/registry.ts, which the harness checks for T16).
// Frame (advanced): [dn_gradient → dn_grad_filter] (ReSTIR temporal) → [copy colour → dnInput] (PT) → dn_temporal →
// dn_variance → dn_atrous × N (the last writes the colour target). Every pass is a pure function of its inputs and the
// histories ping-pong (parity flips only on advanced frames), so a held (paused) frame re-encodes the same plan and the
// HUD timing re-runs it in a separate submit with timestamp writes (DN9; platform-lanes.md Q3).
import { composeWgsl, createCheckedShaderModule } from '../../gpu/wgsl-composer.ts';
import { shaderSources } from '../../shaders/index.ts';
import { registerDenoiser, unregisterDenoiser } from './registry.ts';
import {
  DENOISER_DEFAULTS, DN_ITER_SIZE, DN_PARAMS_SIZE, DNF, atrousPlan, dnTiles, packDnParams, type DenoiserSettings,
} from './layout.ts';

const modules = import.meta.glob('./shaders/*.wgsl', { query: '?raw', import: 'default', eager: true }) as Record<string, string>;
/** The denoiser's WGSL under the keys 'denoise/<file>' (composed with the shared shader sources). */
export const denoiseSources: Record<string, string> = Object.fromEntries(
  Object.entries(modules).map(([k, v]) => [`denoise/${k.replace(/^\.\/shaders\//, '')}`, v]),
);

export type DenoiseKind = 'restir' | 'pt';

export interface DenoiseRestirInput {
  arena: GPUBuffer;
  /** res[w] (the temporal output while ≤ 1 spatial round ran) and res[final] of this frame. */
  resW: GPUBuffer;
  resFinal: GPUBuffer;
  /** Global arena word of tState[0] (64 + 6·P·NS_alloc). */
  tsBase: number;
  /** The gradient passes run (temporal frame with valid history, contribution MIS). */
  gradient: boolean;
  /** The change bits open the λ gate (a light / env change between t−1 and t). */
  lightingChanged: boolean;
  /** res[w] still holds the temporal output (≤ 1 executed spatial round). */
  inverse: boolean;
}

export interface DenoiseFrame {
  kind: DenoiseKind;
  /** The frame advanced (else a held frame: the last plan is re-encoded, DN9). */
  advanced: boolean;
  /** Full history reset (denoiser.md §6). */
  reset: boolean;
  frameUniforms: GPUBuffer;
  /** M1 G-buffer (GBufTexel array; renderer.ts). */
  gbuf: GPUBuffer;
  /** ReSTIR: rsFrame (L1 + estimate) and rsL1. PT: undefined (the colour target holds the 1-spp sample; it is copied). */
  radiance?: GPUTexture;
  l1?: GPUTexture;
  /** Output (and, for the PT, input) colour target. */
  colour: GPUTexture;
  restir?: DenoiseRestirInput;
  debugGroup: GPUBindGroup;
}

export interface DenoiserTiming { totalMs: number; passes: { name: string; ms: number }[] }

interface PlanPass { name: string; pipeline: GPUComputePipeline; g1: GPUBindGroup; wg: [number, number] }

interface Targets {
  w: number; h: number;
  hist: [GPUTexture, GPUTexture]; mom: [GPUTexture, GPUTexture]; alb: [GPUTexture, GPUTexture]; l1: [GPUTexture, GPUTexture]; taa: [GPUTexture, GPUTexture]; out: GPUTexture; lumG: GPUTexture; geo: [GPUTexture, GPUTexture]; atrous: [GPUTexture, GPUTexture];
  gradTile: GPUTexture; gradTile2: GPUTexture; lambda: GPUTexture; input?: GPUTexture;
}

const C = 0x4;   // GPUShaderStage.COMPUTE (no WebGPU global at import time: the CPU tests import the renderer)
const tex = (sampleType: GPUTextureSampleType = 'unfilterable-float'): GPUBindGroupLayoutEntry['texture'] => ({ sampleType });
const st = (format: GPUTextureFormat): GPUBindGroupLayoutEntry['storageTexture'] => ({ access: 'write-only', format });
const ro: GPUBindGroupLayoutEntry['buffer'] = { type: 'read-only-storage' };
const entries = (list: Omit<GPUBindGroupLayoutEntry, 'binding' | 'visibility'>[]): GPUBindGroupLayoutEntry[] => list.map((e, binding) => ({ binding, visibility: C, ...e }));

let nextId = 1;
const ids = new WeakMap<object, number>();
const views = new WeakMap<GPUTexture, GPUTextureView>();
/** One default view per texture (bind groups are cached by texture identity; no new view objects per frame). */
const view = (t: GPUTexture): GPUTextureView => { let x = views.get(t); if (!x) { x = t.createView(); views.set(t, x); } return x; };
const oid = (o: object): number => { let i = ids.get(o); if (i === undefined) { i = nextId++; ids.set(o, i); } return i; };

export class Denoiser {
  readonly settings: DenoiserSettings = { ...DENOISER_DEFAULTS };
  /** Active kind of the last encoded frame (a change resets the history). */
  kind: DenoiseKind | undefined;
  /** Frames encoded since the last reset (HUD) and the reasons of the last reset. */
  framesSinceReset = 0;
  /** Timing history (ms) of the separate timing submits. */
  readonly timings: DenoiserTiming[] = [];
  private t: Targets | undefined;
  private cur = 0;
  private needsReset = true;
  private plan: PlanPass[] = [];
  private planG0: GPUBindGroup | undefined;
  private planDebug: GPUBindGroup | undefined;
  private lastFlags = 0;
  /** Frames since the last lighting change (λ gate open, or an accumulation restart without a gradient). */
  private sinceChange = 0xffff;
  private readonly params: GPUBuffer;
  private readonly iterBufs: GPUBuffer[];
  private readonly g0Layout: GPUBindGroupLayout;
  private readonly layouts: Record<'gradient' | 'gradFilter' | 'temporal' | 'variance' | 'atrous' | 'resolve', GPUBindGroupLayout>;
  private readonly empty: GPUBindGroupLayout;
  private readonly emptyGroup: GPUBindGroup;
  private readonly pipelines = new Map<string, GPUComputePipeline>();
  private readonly dummyTex: GPUTexture;
  private readonly dummyBuf: GPUBuffer;
  private readonly groups = new Map<string, GPUBindGroup>();
  private iterKey = '';
  private timingBusy = false;
  private querySet: GPUQuerySet | undefined;

  private constructor(readonly device: GPUDevice, readonly debugLayout: GPUBindGroupLayout, readonly colorFormat: GPUTextureFormat,
    private readonly o: { features?: Set<string>; wgslLanguageFeatures?: Set<string> }) {
    this.params = device.createBuffer({ label: 'dn-params', size: DN_PARAMS_SIZE, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    this.iterBufs = Array.from({ length: 7 }, (_, i) => device.createBuffer({ label: `dn-iter${i}`, size: DN_ITER_SIZE, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST }));
    this.g0Layout = device.createBindGroupLayout({ label: 'dn-g0', entries: entries([{ buffer: { type: 'uniform' } }, { buffer: { type: 'uniform', minBindingSize: DN_PARAMS_SIZE } }]) });
    this.empty = device.createBindGroupLayout({ label: 'dn-empty', entries: [] });
    this.emptyGroup = device.createBindGroup({ label: 'dn-empty', layout: this.empty, entries: [] });
    this.layouts = {
      gradient: device.createBindGroupLayout({ label: 'dn-gradient', entries: entries([{ buffer: ro }, { buffer: ro }, { texture: tex() }, { storageTexture: st('rgba32float') }, { texture: tex() }, { texture: tex() }, { texture: tex() }, { texture: tex() }, { storageTexture: st('rgba32float') }]) }),
      gradFilter: device.createBindGroupLayout({ label: 'dn-grad-filter', entries: entries([{ texture: tex() }, { storageTexture: st('r32float') }, { texture: tex() }]) }),
      temporal: device.createBindGroupLayout({
        label: 'dn-temporal',
        entries: entries([{ buffer: ro }, { texture: tex() }, { texture: tex() }, { texture: tex('uint') }, { texture: tex() }, { texture: tex() }, { texture: tex() }, { buffer: ro },
          { storageTexture: st('rgba16float') }, { storageTexture: st('rgba16float') }, { storageTexture: st('rgba16float') }, { storageTexture: st('rg32uint') },
          { texture: tex() }, { storageTexture: st('rgba32float') }, { texture: tex() }, { storageTexture: st('rgba32float') }]),
      }),
      variance: device.createBindGroupLayout({ label: 'dn-variance', entries: entries([{ texture: tex() }, { texture: tex() }, { texture: tex('uint') }, { storageTexture: st('rgba16float') }, { storageTexture: st('r32float') }]) }),
      atrous: device.createBindGroupLayout({
        label: `dn-atrous-${colorFormat}`,
        entries: entries([{ texture: tex() }, { texture: tex('uint') }, { buffer: { type: 'uniform', minBindingSize: DN_ITER_SIZE } }, { storageTexture: st('rgba16float') },
          { storageTexture: st('rgba16float') }, { storageTexture: st('rgba16float') }, { texture: tex() }, { texture: tex() }, { texture: tex() }, { texture: tex() }, { texture: tex() }, { texture: tex() }]),
      }),
      resolve: device.createBindGroupLayout({
        label: `dn-resolve-${colorFormat}`,
        entries: entries([{ texture: tex() }, { texture: tex() }, { texture: tex('uint') }, { texture: tex('uint') }, { buffer: ro }, { texture: tex() }, { texture: tex() },
          { storageTexture: st('rgba32float') }, { storageTexture: st(colorFormat) }]),
      }),
    };
    this.dummyTex = device.createTexture({ label: 'dn-zero', size: [1, 1], format: 'rgba32float', usage: GPUTextureUsage.TEXTURE_BINDING });
    this.dummyBuf = device.createBuffer({ label: 'dn-zero', size: 256, usage: GPUBufferUsage.STORAGE });
    registerDenoiser(this, () => `A-SVGF-lite (${this.kind ?? 'idle'}, ${this.settings.iterations} iterations)`);
  }

  static async create(device: GPUDevice, o: { debugLayout: GPUBindGroupLayout; colorFormat: GPUTextureFormat; features?: Set<string>; wgslLanguageFeatures?: Set<string> }): Promise<Denoiser> {
    const d = new Denoiser(device, o.debugLayout, o.colorFormat, o);
    await d.compile();
    return d;
  }

  private async compile(): Promise<void> {
    const sources = { ...shaderSources, ...denoiseSources };
    const mk = async (key: string, file: string, entry: string, layout: GPUBindGroupLayout, defines: Record<string, boolean | string>) => {
      const shader = composeWgsl(file, { sources, defines, features: this.o.features, wgslLanguageFeatures: this.o.wgslLanguageFeatures });
      const module = await createCheckedShaderModule(this.device, shader, key);
      const pipe = await this.device.createComputePipelineAsync({
        label: key, layout: this.device.createPipelineLayout({ label: key, bindGroupLayouts: [this.g0Layout, layout, this.empty, this.debugLayout] }),
        compute: { module, entryPoint: entry },
      });
      this.pipelines.set(key, pipe);
    };
    await Promise.all([
      mk('dn_gradient', 'denoise/dn-gradient.wgsl', 'dn_gradient', this.layouts.gradient, { DN_GRADIENT: true }),
      mk('dn_grad_filter', 'denoise/dn-gradient.wgsl', 'dn_grad_filter', this.layouts.gradFilter, { DN_GRAD_FILTER: true }),
      mk('dn_temporal', 'denoise/dn-temporal.wgsl', 'dn_temporal', this.layouts.temporal, {}),
      mk('dn_variance', 'denoise/dn-filter.wgsl', 'dn_variance', this.layouts.variance, { DN_VARIANCE: true }),
      mk('dn_atrous', 'denoise/dn-filter.wgsl', 'dn_atrous', this.layouts.atrous, { DN_ATROUS: true }),
      mk('dn_resolve', 'denoise/dn-resolve.wgsl', 'dn_resolve', this.layouts.resolve, { COLOR_FORMAT: this.colorFormat }),
    ]);
  }

  /** Settings change; `reset` when the meaning of the history changes (α_min, demodulation-relevant parameters). */
  setSettings(s: Partial<DenoiserSettings>): void {
    const resetKeys: (keyof DenoiserSettings)[] = ['alphaMin', 'nMax', 'resolve'];
    if (resetKeys.some((k) => s[k] !== undefined && s[k] !== this.settings[k])) this.needsReset = true;
    Object.assign(this.settings, s);
  }

  /** Force a full history reset on the next advanced frame. */
  reset(): void { this.needsReset = true; }

  /** (Re)allocate the per-pixel buffers (formats: denoiser.md §2). */
  resize(w: number, h: number): void {
    if (this.t && this.t.w === w && this.t.h === h) return;
    this.destroyTargets();
    const d = this.device;
    const S = GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_SRC;
    const mk = (label: string, format: GPUTextureFormat, size: [number, number] = [w, h]) => d.createTexture({ label, size, format, usage: S });
    const pair = (label: string, format: GPUTextureFormat): [GPUTexture, GPUTexture] => [mk(`${label}0`, format), mk(`${label}1`, format)];
    const [tx, ty] = dnTiles(w, h);
    this.t = {
      w, h, hist: pair('dn-hist', 'rgba16float'), mom: pair('dn-mom', 'rgba16float'), alb: pair('dn-alb', 'rgba32float'), l1: pair('dn-l1', 'rgba32float'), taa: pair('dn-taa', 'rgba32float'), out: mk('dn-out', 'rgba16float'), lumG: mk('dn-lumg', 'r32float'), geo: pair('dn-geo', 'rg32uint'), atrous: pair('dn-atrous', 'rgba16float'),
      gradTile: mk('dn-grad-tile', 'rgba32float', [tx, ty]), gradTile2: mk('dn-grad-tile2', 'rgba32float', [tx, ty]), lambda: mk('dn-lambda', 'r32float', [tx, ty]),
    };
    this.groups.clear();
    this.plan = [];
    this.needsReset = true;
  }

  private pt(w: number, h: number, fmt: GPUTextureFormat): GPUTexture {
    const t = this.t!;
    if (!t.input || t.input.format !== fmt) {
      t.input?.destroy();
      t.input = this.device.createTexture({ label: 'dn-input', size: [w, h], format: fmt, usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST });
    }
    return t.input;
  }

  private group(key: string, layout: GPUBindGroupLayout, res: GPUBindingResource[]): GPUBindGroup {
    let g = this.groups.get(key);
    if (!g) {
      if (this.groups.size > 128) this.groups.clear();
      g = this.device.createBindGroup({ label: key, layout, entries: res.map((resource, binding) => ({ binding, resource })) });
      this.groups.set(key, g);
    }
    return g;
  }

  private writeIterations(): void {
    const key = `${this.settings.iterations}`;
    if (key === this.iterKey) return;
    this.iterKey = key;
    atrousPlan(this.settings.iterations).forEach(([iter, step, flags], i) => {
      this.device.queue.writeBuffer(this.iterBufs[i], 0, new Uint32Array([iter, step, flags, 0]));
    });
  }

  /** Encode the denoiser for one frame (after the ReSTIR finalize or the PT pass). Returns false when not ready. */
  encode(enc: GPUCommandEncoder, f: DenoiseFrame): boolean {
    const t = this.t;
    if (!t || t.w !== f.colour.width || t.h !== f.colour.height) return false;
    if (!f.advanced && this.plan.length && this.planG0 && f.debugGroup === this.planDebug) {
      this.encodePlan(enc, this.planG0, f.debugGroup);   // held frame (DN9): same parity, same inputs, same outputs
      return true;
    }
    if (f.kind === 'pt') {
      const input = this.pt(t.w, t.h, f.colour.format);
      enc.copyTextureToTexture({ texture: f.colour }, { texture: input }, [t.w, t.h]);
    }
    if (this.kind !== f.kind) { this.kind = f.kind; this.needsReset = true; }
    if (f.advanced || !this.plan.length) this.cur = 1 - this.cur;
    const cur = this.cur, prev = 1 - cur;
    const reset = f.reset || this.needsReset;
    this.needsReset = false;
    this.framesSinceReset = reset ? 0 : this.framesSinceReset + 1;
    const r = f.kind === 'restir' ? f.restir : undefined;
    let flags = 0;
    if (reset) flags |= DNF.RESET;
    if (f.kind === 'restir') flags |= DNF.HAS_L1;
    if (!this.settings.resolve) flags |= DNF.NO_RESOLVE;
    if (r) flags |= DNF.FW;
    if (r?.gradient && !reset) {
      flags |= DNF.GRADIENT;
      if (r.lightingChanged) flags |= DNF.LAMBDA;
      if (this.settings.gradientOnCamera) flags |= DNF.LAMBDA_CAM;
      if (r.inverse) flags |= DNF.INVERSE;
    }
    this.lastFlags = flags;
    this.sinceChange = (flags & DNF.LAMBDA) || reset ? 0 : Math.min(this.sinceChange + 1, 0xffff);
    if (this.settings.guide && this.settings.resolve && this.sinceChange >= 8) flags |= DNF.GUIDE;
    this.device.queue.writeBuffer(this.params, 0, packDnParams({ width: t.w, height: t.h, flags, settings: this.settings, tsBase: r?.tsBase ?? 0, resPlanes: 10, sinceChange: this.sinceChange }));
    this.writeIterations();

    const v = view;
    const radiance = f.kind === 'pt' ? t.input! : f.radiance!;
    const l1 = f.l1 ?? this.dummyTex;
    const wg: [number, number] = [Math.ceil(t.w / 8), Math.ceil(t.h / 8)];
    const [tx, ty] = dnTiles(t.w, t.h);
    const plan: PlanPass[] = [];
    if (flags & DNF.GRADIENT) {
      plan.push({ name: 'dn_gradient', pipeline: this.pipelines.get('dn_gradient')!, wg,
        g1: this.group(`grad:${oid(r!.arena)}:${oid(r!.resW)}:${prev}:${oid(radiance)}:${oid(l1)}`, this.layouts.gradient,
          [{ buffer: r!.arena }, { buffer: r!.resW }, v(t.mom[prev]), v(t.gradTile), v(radiance), v(l1), v(t.hist[prev]), v(t.alb[prev]), v(t.gradTile2)]) });
      plan.push({ name: 'dn_grad_filter', pipeline: this.pipelines.get('dn_grad_filter')!, wg: [Math.ceil(tx / 8), Math.ceil(ty / 8)],
        g1: this.group('gradFilter', this.layouts.gradFilter, [v(t.gradTile), v(t.lambda), v(t.gradTile2)]) });
    }
    plan.push({ name: 'dn_temporal', pipeline: this.pipelines.get('dn_temporal')!, wg,
      g1: this.group(`temporal:${oid(f.gbuf)}:${oid(radiance)}:${oid(l1)}:${r ? oid(r.resFinal) : 0}:${cur}:${(flags & DNF.GRADIENT) ? 1 : 0}`, this.layouts.temporal, [
        { buffer: f.gbuf }, v(radiance), v(l1), v(t.geo[prev]), v(t.hist[prev]), v(t.mom[prev]), (flags & DNF.GRADIENT) ? v(t.lambda) : v(this.dummyTex),
        { buffer: r?.resFinal ?? this.dummyBuf }, v(t.atrous[0]), v(t.hist[cur]), v(t.mom[cur]), v(t.geo[cur]), v(t.alb[prev]), v(t.alb[cur]), v(t.l1[prev]), v(t.l1[cur]),
      ]) });
    plan.push({ name: 'dn_variance', pipeline: this.pipelines.get('dn_variance')!, wg,
      g1: this.group(`variance:${cur}`, this.layouts.variance, [v(t.atrous[0]), v(t.mom[cur]), v(t.geo[cur]), v(t.atrous[1]), v(t.lumG)]) });
    const its = atrousPlan(this.settings.iterations);
    its.forEach(([iter], i) => {
      const src = (i + 1) % 2, dst = i % 2;   // dn_variance wrote atrous[1]: iteration 0 reads 1 and writes 0, …
      plan.push({ name: `dn_atrous${iter}`, pipeline: this.pipelines.get('dn_atrous')!, wg,
        g1: this.group(`atrous:${i}:${cur}:${oid(f.colour)}:${oid(radiance)}:${oid(l1)}`, this.layouts.atrous, [
          v(t.atrous[src]), v(t.geo[cur]), { buffer: this.iterBufs[i] }, v(t.atrous[dst]), v(t.hist[cur]), v(t.out), v(radiance), v(t.l1[cur]), v(t.alb[cur]), v(t.taa[prev]), v(t.mom[cur]), v(t.lumG),
        ]) });
    });
    plan.push({ name: 'dn_resolve', pipeline: this.pipelines.get('dn_resolve')!, wg,
      g1: this.group(`resolve:${cur}:${oid(f.colour)}:${oid(f.gbuf)}:${(flags & DNF.GRADIENT) ? 1 : 0}`, this.layouts.resolve, [
        v(t.out), v(t.taa[prev]), v(t.geo[cur]), v(t.geo[prev]), { buffer: f.gbuf }, (flags & DNF.GRADIENT) ? v(t.lambda) : v(this.dummyTex), v(t.mom[cur]), v(t.taa[cur]), v(f.colour),
      ]) });
    this.plan = plan;
    this.planG0 = this.group(`g0:${oid(f.frameUniforms)}`, this.g0Layout, [{ buffer: f.frameUniforms }, { buffer: this.params }]);
    this.planDebug = f.debugGroup;
    this.encodePlan(enc, this.planG0, f.debugGroup);
    return true;
  }

  private encodePlan(enc: GPUCommandEncoder, g0: GPUBindGroup, dbg: GPUBindGroup, ts?: GPUQuerySet): void {
    this.plan.forEach((p, i) => {
      const pass = enc.beginComputePass({ label: p.name, ...(ts ? { timestampWrites: { querySet: ts, beginningOfPassWriteIndex: 2 * i, endOfPassWriteIndex: 2 * i + 1 } } : {}) });
      pass.setPipeline(p.pipeline);
      pass.setBindGroup(0, g0);
      pass.setBindGroup(1, p.g1);
      pass.setBindGroup(2, this.emptyGroup);
      pass.setBindGroup(3, dbg);
      pass.dispatchWorkgroups(p.wg[0], p.wg[1]);
      pass.end();
    });
  }

  /** Pass names of the last frame's plan. */
  get passNames(): string[] { return this.plan.map((p) => p.name); }
  get flags(): number { return this.lastFlags; }

  /**
   * GPU time of the last frame's denoiser passes (DN9, Q3): re-encodes the plan in a SEPARATE submit with a timestamp
   * pair per pass. Call right after the frame's queue.submit (before the next frame writes any uniform): the re-run
   * reads the same inputs and rewrites the same outputs. `iterations` back-to-back re-runs (≥ 1) are averaged.
   */
  async time(iterations = 1): Promise<DenoiserTiming | undefined> {
    if (this.timingBusy || !this.plan.length || !this.planG0 || !this.planDebug || !this.device.features.has('timestamp-query')) return undefined;
    this.timingBusy = true;
    try {
      const n = this.plan.length;
      const count = 2 * n * iterations;
      if (!this.querySet || this.querySet.count < count) { this.querySet?.destroy(); this.querySet = this.device.createQuerySet({ label: 'dn-timing', type: 'timestamp', count }); }
      const res = this.device.createBuffer({ size: count * 8, usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC });
      const read = this.device.createBuffer({ size: count * 8, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
      const enc = this.device.createCommandEncoder({ label: 'dn-timing' });
      const qs = this.querySet;
      for (let k = 0; k < iterations; k++) {
        this.plan.forEach((p, i) => {
          const j = k * n + i;
          const pass = enc.beginComputePass({ label: `${p.name}-timing`, timestampWrites: { querySet: qs, beginningOfPassWriteIndex: 2 * j, endOfPassWriteIndex: 2 * j + 1 } });
          pass.setPipeline(p.pipeline);
          pass.setBindGroup(0, this.planG0!);
          pass.setBindGroup(1, p.g1);
          pass.setBindGroup(2, this.emptyGroup);
          pass.setBindGroup(3, this.planDebug!);
          pass.dispatchWorkgroups(p.wg[0], p.wg[1]);
          pass.end();
        });
      }
      enc.resolveQuerySet(qs, 0, count, res, 0);
      enc.copyBufferToBuffer(res, 0, read, 0, count * 8);
      this.device.queue.submit([enc.finish()]);
      await read.mapAsync(GPUMapMode.READ);
      const q = new BigInt64Array(read.getMappedRange().slice(0));
      read.unmap();
      res.destroy(); read.destroy();
      const passes = this.plan.map((p, i) => {
        let s = 0;
        for (let k = 0; k < iterations; k++) { const j = k * n + i; s += Number(q[2 * j + 1] - q[2 * j]) / 1e6; }
        return { name: p.name, ms: s / iterations };
      });
      let total = 0;
      for (let k = 0; k < iterations; k++) total += Number(q[2 * (k * n + n - 1) + 1] - q[2 * k * n]) / 1e6;
      const r = { totalMs: total / iterations, passes };
      this.timings.push(r);
      if (this.timings.length > 64) this.timings.shift();
      return r;
    } finally {
      this.timingBusy = false;
    }
  }

  /** Mean of the last ≤ 32 timing submits (HUD). */
  timingAverage(): DenoiserTiming | undefined {
    const t = this.timings.slice(-32);
    if (!t.length) return undefined;
    const names = t[t.length - 1].passes.map((p) => p.name);
    return {
      totalMs: t.reduce((a, x) => a + x.totalMs, 0) / t.length,
      passes: names.map((name) => { const v = t.map((x) => x.passes.find((p) => p.name === name)?.ms).filter((x): x is number => x !== undefined); return { name, ms: v.reduce((a, b) => a + b, 0) / Math.max(1, v.length) }; }),
    };
  }

  /** The per-pixel textures (tests, readback). */
  get textures(): Readonly<Targets> | undefined { return this.t; }
  /** Parity of the current frame's outputs. */
  get parity(): number { return this.cur; }

  private destroyTargets(): void {
    const t = this.t;
    if (!t) return;
    for (const x of [...t.hist, ...t.mom, ...t.alb, ...t.l1, ...t.taa, t.out, t.lumG, ...t.geo, ...t.atrous, t.gradTile, t.gradTile2, t.lambda]) x.destroy();
    t.input?.destroy();
    this.t = undefined;
  }

  destroy(): void {
    unregisterDenoiser(this);
    this.destroyTargets();
    this.params.destroy();
    for (const b of this.iterBufs) b.destroy();
    this.dummyTex.destroy();
    this.dummyBuf.destroy();
    this.querySet?.destroy();
    this.groups.clear();
  }
}
