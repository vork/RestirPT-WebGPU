// Debug framework, CPU side (plan §6). Mirror of src/core/shaders/debug/debug-common.wgsl.
// - A registry of views {id, label, source pass, kind}. Ids are stable numbers shared with WGSL constants; views
//   registered later (M3+) only need a new id, since debugMode/debugTap are uniforms (no recompiles, plan §1.8).
// - DebugResources: DebugParams uniform, the DebugBuffer (counters | probe records | AOV plane), debugOut texture,
//   the group-3 bind group used by every pass, and the resolve pass (AOV -> false colour -> debugOut).
import { composeWgsl, createCheckedShaderModule } from '../gpu/wgsl-composer.ts';
import { shaderSources } from '../shaders/index.ts';

export type DebugKind = 'scalar' | 'vec3' | 'code';
export type Colormap = 'viridis' | 'turbo' | 'signed' | 'grey';

export interface DebugViewDef {
  /** Stable id (matches a WGSL const). 0 is reserved for 'beauty'. */
  id: number;
  /** Unique key, e.g. 'gbuffer.depth'. */
  key: string;
  label: string;
  /** UI group, e.g. 'G-buffer'. */
  group: string;
  /** Pass that writes the AOV (informational; shown in the UI). */
  source: string;
  kind: DebugKind;
  range?: [number, number];
  log?: boolean;
  colormap?: Colormap;
  abs?: boolean;
  /** True if the value depends on debugTap (stage taps). */
  tapped?: boolean;
  description?: string;
}

export const KIND_CODE: Record<DebugKind, number> = { scalar: 0, vec3: 1, code: 2 };
export const COLORMAP_CODE: Record<Colormap, number> = { viridis: 0, turbo: 1, signed: 2, grey: 3 };

export const DBGF = { LOG: 1, CMAP_SHIFT: 1, PROBE: 8, SPLIT: 16, ABS: 32, NONFINITE: 64 } as const;

export const DEBUG_TAPS = [
  { id: 0, label: 'final' },
  { id: 1, label: 'after initial' },
  { id: 2, label: 'after temporal' },
  { id: 3, label: 'after spatial' },
  { id: 4, label: 'denoised' },
] as const;

/** Built-in view ids (debug-common.wgsl DBG_*). Ranges: 1-99 test, 100 G-buffer, 200 BVH, 300 env, 400+ later. */
export const DBG = {
  OFF: 0,
  TEST_UV: 1, TEST_DEPTH: 2, TEST_NORMAL: 3, TEST_CELL: 4,
  GB_ALBEDO: 100, GB_NS: 101, GB_NG: 102, GB_DEPTH: 103, GB_PRIM: 104, GB_MATERIAL: 105, GB_UV: 106, GB_THR: 107,
  GB_MOTION: 108, GB_BARY: 109,
  BVH_STEPS: 200, BVH_BOX: 201, BVH_TRI: 202, BVH_STACK: 203, BVH_FLAGS: 204,
  ENV_GRID: 300, ENV_BGMASK: 301,
} as const;

/** Counter slots (DBGC_*). */
export const DBGC = {
  PROBE_COUNT: 0, PROBE_OVERFLOW: 1, NAN: 2, INF: 3, BVH_OVERFLOW: 4, BVH_ITERCAP: 5, QUEUE_OVERFLOW: 6, NEGATIVE: 7,
} as const;

export const DEBUG_BUFFER_LAYOUT = {
  counterCount: 16,
  probeOffset: 64,
  probeStride: 32,
  probeCapacity: 256,
  aovOffset: 64 + 256 * 32,
  headerBytes: 64 + 256 * 32,
  aovStride: 16,
} as const;

export const DEBUG_PARAMS_SIZE = 48;

/** Views of M1 (plan §6 table). Producers write them with debug_write*(pixel, DBG_*, v). */
export const BUILTIN_VIEWS: DebugViewDef[] = [
  { id: DBG.TEST_UV, key: 'test.uv', label: 'Test: pixel UV', group: 'Test', source: 'test-pattern', kind: 'vec3', range: [0, 1] },
  { id: DBG.TEST_DEPTH, key: 'test.depth', label: 'Test: ray t', group: 'Test', source: 'test-pattern', kind: 'scalar', range: [0.1, 100], log: true, colormap: 'turbo' },
  { id: DBG.TEST_NORMAL, key: 'test.normal', label: 'Test: normal', group: 'Test', source: 'test-pattern', kind: 'vec3', range: [-1, 1] },
  { id: DBG.TEST_CELL, key: 'test.cell', label: 'Test: grid cell id', group: 'Test', source: 'test-pattern', kind: 'code' },
  { id: DBG.GB_ALBEDO, key: 'gbuffer.albedo', label: 'Albedo', group: 'G-buffer', source: 'primary', kind: 'vec3', range: [0, 1] },
  { id: DBG.GB_NS, key: 'gbuffer.ns', label: 'Shading normal Ns', group: 'G-buffer', source: 'primary', kind: 'vec3', range: [-1, 1] },
  { id: DBG.GB_NG, key: 'gbuffer.ng', label: 'Geometric normal Ng', group: 'G-buffer', source: 'primary', kind: 'vec3', range: [-1, 1] },
  { id: DBG.GB_DEPTH, key: 'gbuffer.depth', label: 'Depth (view z)', group: 'G-buffer', source: 'primary', kind: 'scalar', range: [0.1, 100], log: true, colormap: 'turbo' },
  { id: DBG.GB_PRIM, key: 'gbuffer.primId', label: 'primId', group: 'G-buffer', source: 'primary', kind: 'code' },
  { id: DBG.GB_MATERIAL, key: 'gbuffer.material', label: 'Material id', group: 'G-buffer', source: 'primary', kind: 'code' },
  { id: DBG.GB_UV, key: 'gbuffer.uv', label: 'UV0 (fract)', group: 'G-buffer', source: 'primary', kind: 'vec3', range: [0, 1] },
  { id: DBG.GB_THR, key: 'gbuffer.thr', label: 'thr (footprint)', group: 'G-buffer', source: 'primary', kind: 'scalar', range: [1e-4, 1], log: true },
  { id: DBG.GB_MOTION, key: 'gbuffer.motion', label: 'Motion vectors (px)', group: 'G-buffer', source: 'primary', kind: 'vec3', range: [-8, 8], colormap: 'signed' },
  { id: DBG.GB_BARY, key: 'gbuffer.bary', label: 'Barycentrics', group: 'G-buffer', source: 'primary', kind: 'vec3', range: [0, 1] },
  { id: DBG.BVH_STEPS, key: 'bvh.steps', label: 'Traversal steps', group: 'BVH', source: 'primary', kind: 'scalar', range: [0, 200], colormap: 'turbo' },
  { id: DBG.BVH_BOX, key: 'bvh.box', label: 'Box tests', group: 'BVH', source: 'primary', kind: 'scalar', range: [0, 400], colormap: 'turbo' },
  { id: DBG.BVH_TRI, key: 'bvh.tri', label: 'Triangle tests', group: 'BVH', source: 'primary', kind: 'scalar', range: [0, 100], colormap: 'turbo' },
  { id: DBG.BVH_STACK, key: 'bvh.stack', label: 'Max stack depth', group: 'BVH', source: 'primary', kind: 'scalar', range: [0, 32], colormap: 'turbo' },
  { id: DBG.BVH_FLAGS, key: 'bvh.flags', label: 'Overflow / itercap flags', group: 'BVH', source: 'primary', kind: 'code' },
  { id: DBG.ENV_GRID, key: 'env.grid', label: 'Env orientation grid', group: 'Env', source: 'primary', kind: 'vec3', range: [0, 1] },
  { id: DBG.ENV_BGMASK, key: 'env.bgmask', label: 'Background mask', group: 'Env', source: 'primary', kind: 'code' },
];

export class DebugViewRegistry {
  private readonly views = new Map<number, DebugViewDef>();
  private readonly listeners = new Set<() => void>();

  constructor(initial: DebugViewDef[] = BUILTIN_VIEWS) { for (const v of initial) this.register(v); }

  register(def: DebugViewDef): DebugViewDef {
    if (!Number.isInteger(def.id) || def.id <= 0 || def.id > 0xffff) throw new Error(`debug view '${def.key}': bad id ${def.id}`);
    const clash = this.views.get(def.id);
    if (clash) throw new Error(`debug view id ${def.id} already used by '${clash.key}'`);
    for (const v of this.views.values()) if (v.key === def.key) throw new Error(`debug view key '${def.key}' already registered`);
    if (def.range && !(def.range[0] <= def.range[1])) throw new Error(`debug view '${def.key}': bad range`);
    this.views.set(def.id, { ...def });
    this.emit();
    return def;
  }

  unregister(id: number): void { if (this.views.delete(id)) this.emit(); }
  get(id: number): DebugViewDef | undefined { return this.views.get(id); }
  byKey(key: string): DebugViewDef | undefined { for (const v of this.views.values()) if (v.key === key) return v; return undefined; }
  list(): DebugViewDef[] { return [...this.views.values()].sort((a, b) => a.id - b.id); }
  onChange(cb: () => void): () => void { this.listeners.add(cb); return () => this.listeners.delete(cb); }
  private emit(): void { for (const cb of this.listeners) cb(); }
}

export interface DebugSettings {
  mode: number;
  tap: number;
  rangeMin: number;
  rangeMax: number;
  log: boolean;
  colormap: Colormap;
  abs: boolean;
  highlightNonFinite: boolean;
  probeEnabled: boolean;
  probePixel: [number, number];
  split: boolean;
  splitPos: number;
}

export function defaultDebugSettings(): DebugSettings {
  return {
    mode: 0, tap: 0, rangeMin: 0, rangeMax: 1, log: false, colormap: 'viridis', abs: false,
    highlightNonFinite: true, probeEnabled: false, probePixel: [0, 0], split: false, splitPos: 0.5,
  };
}

/** Apply a view's defaults (range, log, colormap, abs) to the settings when the user selects it. */
export function applyViewDefaults(s: DebugSettings, v: DebugViewDef | undefined): void {
  if (!v) return;
  s.rangeMin = v.range?.[0] ?? 0;
  s.rangeMax = v.range?.[1] ?? 1;
  s.log = v.log ?? false;
  s.colormap = v.colormap ?? 'viridis';
  s.abs = v.abs ?? false;
}

export function debugFlags(s: DebugSettings): number {
  let f = (COLORMAP_CODE[s.colormap] & 3) << DBGF.CMAP_SHIFT;
  if (s.log) f |= DBGF.LOG;
  if (s.probeEnabled) f |= DBGF.PROBE;
  if (s.split) f |= DBGF.SPLIT;
  if (s.abs) f |= DBGF.ABS;
  if (s.highlightNonFinite) f |= DBGF.NONFINITE;
  return f >>> 0;
}

/** Pack DebugParams (48 B, see debug-common.wgsl). */
export function packDebugParams(s: DebugSettings, kind: DebugKind, frame: number, size: [number, number], out = new ArrayBuffer(DEBUG_PARAMS_SIZE)): ArrayBuffer {
  const u32 = new Uint32Array(out);
  const f32 = new Float32Array(out);
  u32[0] = s.mode >>> 0;
  u32[1] = s.tap >>> 0;
  u32[2] = s.probePixel[0] >>> 0;
  u32[3] = s.probePixel[1] >>> 0;
  f32[4] = s.rangeMin;
  f32[5] = s.rangeMax;
  u32[6] = debugFlags(s);
  u32[7] = frame >>> 0;
  u32[8] = KIND_CODE[kind];
  f32[9] = s.splitPos;
  u32[10] = size[0];
  u32[11] = size[1];
  return out;
}

/** GPU side of the debug framework. Group 3 = {0: DebugParams, 1: DebugBuffer} for every pass. */
export class DebugResources {
  readonly params: GPUBuffer;
  /** Group-3 layout every pass includes in its pipeline layout. */
  readonly layout: GPUBindGroupLayout;
  private readonly resolveLayout: GPUBindGroupLayout;
  private resolvePipeline: GPUComputePipeline | undefined;
  buffer!: GPUBuffer;
  debugOut!: GPUTexture;
  bindGroup!: GPUBindGroup;
  private resolveBindGroup!: GPUBindGroup;
  width = 0;
  height = 0;
  private readonly scratch = new ArrayBuffer(DEBUG_PARAMS_SIZE);
  /** Last settings written (read by the probe panel and the blit). */
  settings: DebugSettings = defaultDebugSettings();

  constructor(private readonly device: GPUDevice, readonly registry = new DebugViewRegistry()) {
    this.params = device.createBuffer({ label: 'debug-params', size: DEBUG_PARAMS_SIZE, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    const vis = GPUShaderStage.COMPUTE | GPUShaderStage.FRAGMENT;
    const common: GPUBindGroupLayoutEntry[] = [
      { binding: 0, visibility: vis, buffer: { type: 'uniform' } },
      { binding: 1, visibility: vis, buffer: { type: 'storage' } },
    ];
    this.layout = device.createBindGroupLayout({ label: 'debug-g3', entries: common });
    this.resolveLayout = device.createBindGroupLayout({
      label: 'debug-g3-resolve',
      entries: [...common, { binding: 2, visibility: GPUShaderStage.COMPUTE, storageTexture: { access: 'write-only', format: 'rgba32float' } }],
    });
  }

  async init(): Promise<void> {
    const shader = composeWgsl('post/resolve.wgsl', { sources: shaderSources, defines: { RESOLVE_DEBUG_ENTRY: true, DEBUG_OUT_BINDING: true } });
    const module = await createCheckedShaderModule(this.device, shader, 'debug-resolve');
    const empty = this.device.createBindGroupLayout({ label: 'empty', entries: [] });
    this.resolvePipeline = await this.device.createComputePipelineAsync({
      label: 'debug-resolve',
      layout: this.device.createPipelineLayout({ bindGroupLayouts: [empty, empty, empty, this.resolveLayout] }),
      compute: { module, entryPoint: 'debug_resolve' },
    });
    this.emptyGroup = this.device.createBindGroup({ layout: empty, entries: [] });
  }
  private emptyGroup: GPUBindGroup | undefined;

  /** (Re)allocate the AOV plane and debugOut for the internal resolution. Invalidates `bindGroup`. */
  resize(width: number, height: number): void {
    if (width === this.width && height === this.height && this.buffer) return;
    this.buffer?.destroy();
    this.debugOut?.destroy();
    this.width = width;
    this.height = height;
    this.buffer = this.device.createBuffer({
      label: 'debug-buffer',
      size: DEBUG_BUFFER_LAYOUT.aovOffset + width * height * DEBUG_BUFFER_LAYOUT.aovStride,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
    });
    this.debugOut = this.device.createTexture({
      label: 'debugOut',
      size: [width, height],
      format: 'rgba32float',
      usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_SRC,
    });
    const entries: GPUBindGroupEntry[] = [
      { binding: 0, resource: { buffer: this.params } },
      { binding: 1, resource: { buffer: this.buffer } },
    ];
    this.bindGroup = this.device.createBindGroup({ label: 'debug-g3', layout: this.layout, entries });
    this.resolveBindGroup = this.device.createBindGroup({
      label: 'debug-g3-resolve',
      layout: this.resolveLayout,
      entries: [...entries, { binding: 2, resource: this.debugOut.createView() }],
    });
  }

  activeView(): DebugViewDef | undefined { return this.registry.get(this.settings.mode); }

  update(s: DebugSettings, frame: number): void {
    this.settings = s;
    const kind = this.activeView()?.kind ?? 'scalar';
    this.device.queue.writeBuffer(this.params, 0, packDebugParams(s, kind, frame, [this.width, this.height], this.scratch));
  }

  /** Clear per-frame counters/probe records; clear the AOV plane when a view is active (stale values otherwise). */
  beginFrame(encoder: GPUCommandEncoder): void {
    encoder.clearBuffer(this.buffer, 0, DEBUG_BUFFER_LAYOUT.headerBytes);
    if (this.settings.mode !== 0) encoder.clearBuffer(this.buffer, DEBUG_BUFFER_LAYOUT.aovOffset);
  }

  /** AOV -> false colour -> debugOut. Runs only when a view is active (or the probe needs the AOV value). */
  encodeResolve(encoder: GPUCommandEncoder, timestampWrites?: GPUComputePassTimestampWrites): void {
    if (!this.resolvePipeline || this.settings.mode === 0) return;
    const pass = encoder.beginComputePass({ label: 'debug-resolve', timestampWrites });
    pass.setPipeline(this.resolvePipeline);
    pass.setBindGroup(0, this.emptyGroup!);
    pass.setBindGroup(1, this.emptyGroup!);
    pass.setBindGroup(2, this.emptyGroup!);
    pass.setBindGroup(3, this.resolveBindGroup);
    pass.dispatchWorkgroups(Math.ceil(this.width / 8), Math.ceil(this.height / 8));
    pass.end();
  }

  destroy(): void {
    this.buffer?.destroy();
    this.debugOut?.destroy();
    this.params.destroy();
  }
}
