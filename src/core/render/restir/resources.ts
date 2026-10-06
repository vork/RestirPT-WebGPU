// Allocation of every M4 ReSTIR buffer and texture, the bind group layouts G0–G3, the per-pass G2 layouts / defines and
// the ping-pong bind groups (restir-api.md §2.5–§2.10, §2.13, §4.2). One owner: WP-A (P0). Other WPs consume:
//   RS_PASSES[name]            file, entry point, scene/debug groups, G2 layout entries and binding defines of a pass
//   restirPassDefines(name)    the per-pass binding defines ('<n>u' strings: binding 0 is truthy in #if, Changelog A4)
//   RestirResources            buffers/textures + g2(name, variant) bind groups (variant = index of the input reservoir)
// M5 (restir-temporal-api.md §2.2, §2.6, §4.2; P0 by T-A): the temporal / refresh passes, G0 binding 8 (RsTemporal),
// the ping-ponged ReSTIR G-buffer rsVbuf[2] / rsGeo[2] (parity g; `vbuf`/`geo`/`views.vbuf`/`views.geo` are the CUR
// parity; `vbufPrev`/`geoPrev` the previous frame's) and the arena extension (tState, sfxOut).
import type { Defines } from '../../gpu/wgsl-composer.ts';
import { envDefines } from '../env-gpu.ts';
import { LUT_RECORDS_BASE } from '../lights-gpu.ts';
import { lutDefines } from '../luts/lut-layout.ts';
import {
  ARENA_HDR_BYTES, RES_BYTES, RESTIR_PARAMS_SIZE, RS_DISPATCH_RING, RS_DISPATCH_SIZE, RS_DISPATCH_STRIDE, arenaBytesM6, dumpBytes,
} from './layout.ts';

export type RsPassName =
  | 'rs_primary' | 'rs_initial' | 'rs_initial_dump' | 'rs_pair_accept' | 'rs_args' | 'rs_spatial_replay' | 'rs_spatial_shift'
  | 'rs_spatial_resample' | 'rs_finalize' | 'rs_finalize_frame' | 'rs_ensemble_stats'
  // M5 temporal / refresh passes (restir-temporal-api.md §4.2)
  | 'rs_refresh_fwd' | 'rs_refresh_inv' | 'rs_t_classify' | 'rs_t_forward' | 'rs_t_select' | 'rs_t_inverse'
  // M6 passes (restir-m6-api.md §3)
  | 'rs_light_tiles' | 'rs_dupmap';

type G2Kind =
  | { k: 'ro' } | { k: 'rw'; min?: number } | { k: 'st'; format: GPUTextureFormat } | { k: 'tex'; sampleType: GPUTextureSampleType; dim?: GPUTextureViewDimension }
  | { k: 'colour' };

export interface RsPassDef {
  file: string;
  entry: string;
  /** G1 = the scene (SceneGpu.layoutEntries) or an empty layout. */
  scene: boolean;
  /** G3 = DebugResources.layout when the kernel has debug resources (else empty + DEBUG_NO_BINDINGS). */
  debug: boolean;
  /** G2 bindings in order (§4.2). */
  g2: G2Kind[];
  /** Pass-specific defines (binding defines are '<n>u' strings). */
  defines: Defines;
}

const b = (n: number) => `${n}u`;
const RO: G2Kind = { k: 'ro' };
const RW: G2Kind = { k: 'rw' };
const TEX_U: G2Kind = { k: 'tex', sampleType: 'uint' };
const TEX_F: G2Kind = { k: 'tex', sampleType: 'unfilterable-float' };
const TEX_PAIR: G2Kind = { k: 'tex', sampleType: 'sint', dim: '2d-array' };
const ST_F: G2Kind = { k: 'st', format: 'rgba32float' };
const TEMPORAL: Defines = { RS_TEMPORAL: 1 };
/** T1 / T2: resIn = res[h] · arena · cur and prev G-buffer (§4.2). */
const T_BINDINGS_IN: Defines = {
  RS_RES_IN_BINDING: b(0), RS_ARENA_BINDING: b(1), RS_ARENA_RW: true, RS_VBUF_BINDING: b(2), RS_GEO_BINDING: b(3),
  RS_VBUF_PREV_BINDING: b(4), RS_GEO_PREV_BINDING: b(5),
};

/** The M4 pass table (§4.2). WP-C's files may not exist yet; the kernel compiles a pass only when asked. */
export const RS_PASSES: Record<RsPassName, RsPassDef> = {
  rs_primary: {
    file: 'passes/restir/primary.wgsl', entry: 'rs_primary', scene: true, debug: true,
    g2: [{ k: 'st', format: 'rgba32uint' }, ST_F, ST_F, RW],
    defines: { RS_VBUF_W_BINDING: b(0), RS_GEO_W_BINDING: b(1), RS_L1_W_BINDING: b(2), RS_ARENA_BINDING: b(3), RS_ARENA_RW: true },
  },
  rs_initial: {
    file: 'passes/restir/initial.wgsl', entry: 'rs_initial', scene: true, debug: true,
    g2: [RW, RW, TEX_U, TEX_F],
    defines: { RS_RES_OUT_BINDING: b(0), RS_ARENA_BINDING: b(1), RS_ARENA_RW: true, RS_VBUF_BINDING: b(2), RS_GEO_BINDING: b(3) },
  },
  rs_initial_dump: {
    file: 'passes/restir/initial.wgsl', entry: 'rs_initial', scene: true, debug: false,
    g2: [RW, RW, TEX_U, TEX_F, RW],
    defines: { RS_RES_OUT_BINDING: b(0), RS_ARENA_BINDING: b(1), RS_ARENA_RW: true, RS_VBUF_BINDING: b(2), RS_GEO_BINDING: b(3), RS_DUMP_CANDIDATES: true },
  },
  rs_pair_accept: {
    file: 'passes/restir/pair-accept.wgsl', entry: 'rs_pair_accept', scene: false, debug: true,
    g2: [RO, RW, TEX_U, TEX_F, TEX_PAIR],
    defines: { RS_RES_IN_BINDING: b(0), RS_ARENA_BINDING: b(1), RS_ARENA_RW: true, RS_VBUF_BINDING: b(2), RS_GEO_BINDING: b(3), RS_PAIRTEX_BINDING: b(4) },
  },
  rs_args: {
    file: 'passes/restir/args.wgsl', entry: 'rs_args', scene: false, debug: true,
    g2: [RW, RW],   // Changelog C1: rs_args writes hdr.n / capacity / overflow
    defines: { RS_ARENA_BINDING: b(0), RS_ARENA_RW: true, RS_ARGS_BINDING: b(1) },
  },
  rs_spatial_replay: {
    file: 'passes/restir/spatial-replay.wgsl', entry: 'rs_spatial_replay', scene: true, debug: true,
    g2: [RO, RW, TEX_U, TEX_F, TEX_PAIR],
    defines: { RS_RES_IN_BINDING: b(0), RS_ARENA_BINDING: b(1), RS_ARENA_RW: true, RS_VBUF_BINDING: b(2), RS_GEO_BINDING: b(3), RS_PAIRTEX_BINDING: b(4), RS_REPLAY: 1 },
  },
  rs_spatial_shift: {
    file: 'passes/restir/spatial-shift.wgsl', entry: 'rs_spatial_shift', scene: true, debug: true,
    g2: [RO, RW, TEX_U, TEX_F, TEX_PAIR],
    defines: { RS_RES_IN_BINDING: b(0), RS_ARENA_BINDING: b(1), RS_ARENA_RW: true, RS_VBUF_BINDING: b(2), RS_GEO_BINDING: b(3), RS_PAIRTEX_BINDING: b(4), RS_REPLAY: 0 },
  },
  rs_spatial_resample: {
    file: 'passes/restir/spatial-resample.wgsl', entry: 'rs_spatial_resample', scene: false, debug: true,
    g2: [RO, RW, RW, TEX_PAIR, ST_F, TEX_U],   // Changelog C1: the resample counts RSC_PENDING_LEFT etc.
    defines: { RS_RES_IN_BINDING: b(0), RS_RES_OUT_BINDING: b(1), RS_ARENA_BINDING: b(2), RS_ARENA_RW: true, RS_PAIRTEX_BINDING: b(3), RS_SHADE_W_BINDING: b(4), RS_VBUF_BINDING: b(5) },
  },
  rs_finalize: {
    file: 'passes/restir/finalize.wgsl', entry: 'rs_finalize', scene: false, debug: true,
    g2: [RO, RW, { k: 'rw', min: 16 }, TEX_F, TEX_F, ST_F, TEX_U],
    defines: { RS_RES_IN_BINDING: b(0), RS_L1_BINDING: b(3), RS_SHADE_BINDING: b(4), RS_FRAME_W_BINDING: b(5), RS_VBUF_BINDING: b(6), RS_INTERACTIVE: false },
  },
  rs_finalize_frame: {
    file: 'passes/restir/finalize.wgsl', entry: 'rs_finalize_frame', scene: false, debug: true,
    g2: [RO, RW, { k: 'rw', min: 16 }, TEX_F, TEX_F, ST_F, TEX_U, { k: 'colour' }],
    defines: { RS_RES_IN_BINDING: b(0), RS_L1_BINDING: b(3), RS_SHADE_BINDING: b(4), RS_FRAME_W_BINDING: b(5), RS_VBUF_BINDING: b(6), RS_INTERACTIVE: true },
  },
  rs_ensemble_stats: {
    file: 'passes/restir/ensemble-stats.wgsl', entry: 'rs_ensemble_stats', scene: false, debug: true,
    g2: [RW, RW, TEX_F, TEX_U],
    defines: { RS_ENS_STATS_BINDING: b(0), RS_ENS_PIXEL_BINDING: b(1), RS_FRAME_BINDING: b(2), RS_MASK_BINDING: b(3) },
  },
  // ---- M5 (restir-temporal-api.md §4.2; RS_TEMPORAL on every temporal / refresh pass) ----
  rs_refresh_fwd: {
    file: 'passes/restir/refresh.wgsl', entry: 'rs_refresh_fwd', scene: true, debug: true,
    g2: [RO, RW, TEX_U, TEX_F],
    defines: { ...TEMPORAL, RS_RES_IN_BINDING: b(0), RS_ARENA_BINDING: b(1), RS_ARENA_RW: true, RS_VBUF_BINDING: b(2), RS_GEO_BINDING: b(3) },
  },
  rs_refresh_inv: {
    file: 'passes/restir/refresh.wgsl', entry: 'rs_refresh_inv', scene: true, debug: true,
    g2: [RO, RW, TEX_U, TEX_F],
    defines: { ...TEMPORAL, RS_RES_IN_BINDING: b(0), RS_ARENA_BINDING: b(1), RS_ARENA_RW: true, RS_VBUF_BINDING: b(2), RS_GEO_BINDING: b(3) },
  },
  rs_t_classify: {
    file: 'passes/restir/t-classify.wgsl', entry: 'rs_t_classify', scene: true, debug: true,
    g2: [RO, RW, TEX_U, TEX_F, TEX_U, TEX_F],
    defines: { ...TEMPORAL, ...T_BINDINGS_IN, RS_REPLAY: 0 },
  },
  rs_t_forward: {
    file: 'passes/restir/t-forward.wgsl', entry: 'rs_t_forward', scene: true, debug: true,
    g2: [RO, RW, TEX_U, TEX_F, TEX_U, TEX_F],
    defines: { ...TEMPORAL, ...T_BINDINGS_IN, RS_REPLAY: 1 },
  },
  rs_t_select: {
    file: 'passes/restir/t-select.wgsl', entry: 'rs_t_select', scene: false, debug: true,
    g2: [RO, RW, RW, TEX_U],
    defines: { ...TEMPORAL, RS_RES_IN_BINDING: b(0), RS_RES_OUT_BINDING: b(1), RS_ARENA_BINDING: b(2), RS_ARENA_RW: true, RS_VBUF_BINDING: b(3) },
  },
  rs_t_inverse: {
    file: 'passes/restir/t-inverse.wgsl', entry: 'rs_t_inverse', scene: true, debug: true,
    g2: [RW, RW, TEX_U, TEX_F, TEX_U, TEX_F],
    defines: {
      ...TEMPORAL, RS_RES_OUT_BINDING: b(0), RS_ARENA_BINDING: b(1), RS_ARENA_RW: true, RS_VBUF_BINDING: b(2), RS_GEO_BINDING: b(3),
      RS_VBUF_PREV_BINDING: b(4), RS_GEO_PREV_BINDING: b(5), RS_REPLAY: 1,
    },
  },
  // ---- M6 (restir-m6-api.md §3) ----
  rs_light_tiles: {
    file: 'passes/restir/light-tiles.wgsl', entry: 'rs_light_tiles', scene: false, debug: false,
    g2: [RW],
    defines: { RS_ARENA_BINDING: b(0), RS_ARENA_RW: true },
  },
  rs_dupmap: {
    file: 'passes/restir/dupmap.wgsl', entry: 'rs_dupmap', scene: false, debug: false,
    g2: [RO, RW],
    defines: { RS_RES_IN_BINDING: b(0), RS_ARENA_BINDING: b(1), RS_ARENA_RW: true },
  },
};

/** The M5 temporal / refresh passes (compile smoke, U-BIND-1). */
export const TEMPORAL_PASSES: readonly RsPassName[] = ['rs_refresh_fwd', 'rs_refresh_inv', 'rs_t_classify', 'rs_t_forward', 'rs_t_select', 'rs_t_inverse'];

/** Per-pass binding defines (plus the pass's own switches). */
export function restirPassDefines(name: RsPassName): Defines { return { ...RS_PASSES[name].defines }; }

/** Every composer define of a ReSTIR pipeline: scene (SceneGpu.defines(1), passes with G1 = scene), env in G0 at 1,
 *  lights in G0 at 5/6, the BSDF LUTs in `records`, RS_MODE_B = 0, DEBUG_NO_BINDINGS without G3, the pass bindings. */
export function restirDefines(name: RsPassName, o: { sceneDefines?: Defines; debug: boolean; extra?: Defines }): Defines {
  const d = RS_PASSES[name];
  if (d.scene && !o.sceneDefines) throw new Error(`${name}: scene defines required`);
  return { ...restirCommonDefines(d.scene ? o.sceneDefines : undefined, d.debug && o.debug), ...restirPassDefines(name), ...(o.extra ?? {}) };
}

/** The defines every ReSTIR module needs, without pass bindings (custom / test pipelines). */
export function restirCommonDefines(sceneDefines: Defines | undefined, debug: boolean): Defines {
  return {
    ...(sceneDefines ?? {}),
    ...envDefines(0, G0_BINDING.env), LIGHTS_GROUP: 0, LIGHTS_BINDING: G0_BINDING.lights,
    ...lutDefines({ base: LUT_RECORDS_BASE, recordsKind: 'u32' }),
    RS_MODE_B: 0, DEBUG_NO_BINDINGS: !debug,
  };
}

/** G2 layout entries of a pass; `colorFormat` is required for the interactive finalize. */
export function g2LayoutEntries(name: RsPassName, colorFormat?: GPUTextureFormat): GPUBindGroupLayoutEntry[] {
  const c = GPUShaderStage.COMPUTE;
  return RS_PASSES[name].g2.map((g, binding): GPUBindGroupLayoutEntry => {
    switch (g.k) {
      case 'ro': return { binding, visibility: c, buffer: { type: 'read-only-storage' } };
      case 'rw': return { binding, visibility: c, buffer: { type: 'storage', ...(g.min ? { minBindingSize: g.min } : {}) } };
      case 'st': return { binding, visibility: c, storageTexture: { access: 'write-only', format: g.format } };
      case 'tex': return { binding, visibility: c, texture: { sampleType: g.sampleType, viewDimension: g.dim ?? '2d' } };
      case 'colour':
        if (!colorFormat) throw new Error(`${name}: colour format required`);
        return { binding, visibility: c, storageTexture: { access: 'write-only', format: colorFormat } };
    }
  });
}

/** Storage buffers a pipeline layout of this pass counts (G0 records + G1 scene 5 + G2 + G3 debug 1), §4.2. */
export function storageBufferCount(name: RsPassName, debug: boolean): number {
  const d = RS_PASSES[name];
  return 1 + (d.scene ? 5 : 0) + d.g2.filter((g) => g.k === 'ro' || g.k === 'rw').length + (debug && d.debug ? 1 : 0);
}

export const G0_BINDING = { frame: 0, env: 1, params: 4, lights: 5, records: 6, dispatch: 7, temporal: 8 } as const;

/** Sizes of an allocation: atlas, member tile, E, NS (= NS_alloc of the arena, layout.ts nsAlloc), temporal (M5: the arena
 *  extension and the second G-buffer pair). */
export interface RestirAllocation {
  atlasW: number; atlasH: number; memberW: number; memberH: number; members: number; memberCols: number; slots: number; dump: boolean;
  temporal?: boolean;
  /** M6 arena region (restir-m6-api.md §2.3): duplication counts (dupmap) and light tiles of `tileMembers` members (RIS). */
  m6?: { dup: boolean; tileMembers: number };
}

export const ENS_LEVELS = [16, 32, 64] as const;
/** ensStats floats (§2.10): per level, member and tile a vec4; then global[m]; then mask[m][16]. */
export function ensStatsFloats(E: number, W: number, H: number, maskRegions = 16): number {
  let n = 0;
  for (const l of ENS_LEVELS) n += E * Math.ceil(H / l) * Math.ceil(W / l) * 4;
  return n + E * 4 + E * maskRegions * 4;
}

/**
 * Every M4 buffer / texture of one atlas allocation. Textures are atlas-sized; `resA`/`resB` are the ping-pong
 * reservoirs (res[0] = A is the initial output). The pairing texture (256² × 8 rg8sint) starts at zero (no partner)
 * and is filled by WP-C's pairing.ts through `pairTex`.
 */
export class RestirResources {
  readonly alloc: RestirAllocation;
  readonly res: [GPUBuffer, GPUBuffer];
  readonly arena: GPUBuffer;
  readonly args: GPUBuffer;
  /** Ping-ponged ReSTIR G-buffer (M5 §2.6): [0] and [1]; without temporal both entries are the same texture. */
  readonly vbufs: [GPUTexture, GPUTexture];
  readonly geos: [GPUTexture, GPUTexture];
  /** G-buffer parity g of the current frame (rs_primary writes [g]; the kernel flips it in advance()). */
  parity = 0;
  readonly l1: GPUTexture;
  readonly shade: GPUTexture;
  readonly frameTex: GPUTexture;
  readonly pairTex: GPUTexture;
  readonly maskTex: GPUTexture;
  readonly ensStats: GPUBuffer;
  readonly ensPixel: GPUBuffer;
  readonly candDump: GPUBuffer | undefined;
  /** Views; `vbuf`/`geo` are the current parity's, `vbufPrev`/`geoPrev` the previous frame's (1 − g). */
  readonly views: {
    readonly vbuf: GPUTextureView; readonly geo: GPUTextureView; readonly vbufPrev: GPUTextureView; readonly geoPrev: GPUTextureView;
    l1: GPUTextureView; shade: GPUTextureView; frame: GPUTextureView; pair: GPUTextureView; mask: GPUTextureView;
  };
  private readonly vbufViews: [GPUTextureView, GPUTextureView];
  private readonly geoViews: [GPUTextureView, GPUTextureView];
  private groups = new Map<string, GPUBindGroup>();

  constructor(readonly device: GPUDevice, alloc: RestirAllocation, private readonly layouts: (name: RsPassName) => GPUBindGroupLayout) {
    this.alloc = alloc;
    const { atlasW: W, atlasH: H, slots } = alloc;
    const P = W * H;
    const S = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST;
    const buf = (label: string, size: number, usage = S) => device.createBuffer({ label, size: Math.max(16, Math.ceil(size / 16) * 16), usage });
    this.res = [buf('rs-resA', P * RES_BYTES), buf('rs-resB', P * RES_BYTES)];
    this.arena = buf('rs-arena', arenaBytesM6(P, slots, !!alloc.temporal, alloc.m6 ?? { dup: false, tileMembers: 0 }));
    this.args = buf('rs-args', 64, S | GPUBufferUsage.INDIRECT);
    const tex = (label: string, format: GPUTextureFormat, storage = true) => device.createTexture({
      label, size: [W, H], format,
      usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_SRC | (storage ? GPUTextureUsage.STORAGE_BINDING : GPUTextureUsage.COPY_DST),
    });
    const vb0 = tex('rs-vbuf0', 'rgba32uint'), geo0 = tex('rs-geo0', 'rgba32float');
    this.vbufs = alloc.temporal ? [vb0, tex('rs-vbuf1', 'rgba32uint')] : [vb0, vb0];
    this.geos = alloc.temporal ? [geo0, tex('rs-geo1', 'rgba32float')] : [geo0, geo0];
    this.l1 = tex('rs-l1', 'rgba32float');
    this.shade = tex('rs-shade', 'rgba32float');
    this.frameTex = tex('rs-frame', 'rgba32float');
    this.pairTex = device.createTexture({ label: 'rs-pairtex', size: [256, 256, 8], format: 'rg8sint', usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST });
    this.maskTex = device.createTexture({ label: 'rs-mask', size: [1, 1], format: 'r8uint', usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST });
    const E = alloc.members;
    this.ensStats = buf('rs-ens-stats', E > 1 ? ensStatsFloats(E, alloc.memberW, alloc.memberH) * 4 : 16);
    this.ensPixel = buf('rs-ens-pixel', E > 1 ? alloc.memberW * alloc.memberH * 32 : 16);
    this.candDump = alloc.dump ? buf('rs-cand-dump', dumpBytes(P)) : undefined;
    const v0 = this.vbufs[0].createView(), g0 = this.geos[0].createView();
    this.vbufViews = alloc.temporal ? [v0, this.vbufs[1].createView()] : [v0, v0];
    this.geoViews = alloc.temporal ? [g0, this.geos[1].createView()] : [g0, g0];
    // eslint-disable-next-line @typescript-eslint/no-this-alias
    const self = this;
    this.views = {
      get vbuf() { return self.vbufViews[self.parity]; }, get geo() { return self.geoViews[self.parity]; },
      get vbufPrev() { return self.vbufViews[1 - self.parity]; }, get geoPrev() { return self.geoViews[1 - self.parity]; },
      l1: this.l1.createView(), shade: this.shade.createView(),
      frame: this.frameTex.createView(), pair: this.pairTex.createView({ dimension: '2d-array' }), mask: this.maskTex.createView(),
    };
  }

  /** Current / previous parity G-buffer textures (M4 code sees `vbuf` / `geo` = the current frame's). */
  get vbuf(): GPUTexture { return this.vbufs[this.parity]; }
  get geo(): GPUTexture { return this.geos[this.parity]; }
  get vbufPrev(): GPUTexture { return this.vbufs[1 - this.parity]; }
  get geoPrev(): GPUTexture { return this.geos[1 - this.parity]; }

  get pixels(): number { return this.alloc.atlasW * this.alloc.atlasH; }

  /** Bytes of the arena header (queue headers + counters). */
  static readonly arenaHeaderBytes = ARENA_HDR_BYTES;

  private group(key: string, name: RsPassName, resources: GPUBindingResource[]): GPUBindGroup {
    let g = this.groups.get(key);
    if (!g) {
      g = this.device.createBindGroup({ label: key, layout: this.layouts(name), entries: resources.map((resource, binding) => ({ binding, resource })) });
      this.groups.set(key, g);
    }
    return g;
  }

  /**
   * G2 bind group of a pass. `inIdx` = index (0 = A, 1 = B) of the input reservoir for the ping-pong passes (the
   * resample writes the other one; finalize reads res[inIdx]). Finalize needs `accum` / `counters` (and `colour` for
   * the interactive variant); they are part of the cache key.
   */
  g2(name: RsPassName, inIdx = 0, ext: { accum?: GPUBuffer; counters?: GPUBuffer; colour?: GPUTextureView } = {}): GPUBindGroup {
    const v = this.views;
    const rin = { buffer: this.res[inIdx] }, rout = { buffer: this.res[1 - inIdx] };
    const arena = { buffer: this.arena };
    // Every key carries the G-buffer parity (M5 §2.6: groups cached per (pass, reservoir index, parity)).
    const g = `:g${this.parity}`;
    switch (name) {
      case 'rs_primary': return this.group(name + g, name, [v.vbuf, v.geo, v.l1, arena]);
      // rs_initial writes res[inIdx] (= res[w], TD2; 0 with temporal off).
      case 'rs_initial': return this.group(`${name}:${inIdx}${g}`, name, [rin, arena, v.vbuf, v.geo]);
      case 'rs_initial_dump':
        if (!this.candDump) throw new Error('rs_initial_dump: allocate with dump = true');
        return this.group(`${name}:${inIdx}${g}`, name, [rin, arena, v.vbuf, v.geo, { buffer: this.candDump }]);
      case 'rs_pair_accept': case 'rs_spatial_replay': case 'rs_spatial_shift':
        return this.group(`${name}:${inIdx}${g}`, name, [rin, arena, v.vbuf, v.geo, v.pair]);
      case 'rs_args': return this.group(name, name, [arena, { buffer: this.args }]);
      case 'rs_spatial_resample': return this.group(`${name}:${inIdx}${g}`, name, [rin, rout, arena, v.pair, v.shade, v.vbuf]);
      case 'rs_finalize': case 'rs_finalize_frame': {
        if (!ext.accum || !ext.counters) throw new Error(`${name}: accum and counters required`);
        const base: GPUBindingResource[] = [rin, { buffer: ext.accum }, { buffer: ext.counters, size: 16 }, v.l1, v.shade, v.frame, v.vbuf];
        if (name === 'rs_finalize_frame') {
          if (!ext.colour) throw new Error('rs_finalize_frame: colour target view required');
          base.push(ext.colour);
        }
        return this.group(`${name}:${inIdx}${g}:${objId(ext.accum)}:${objId(ext.counters)}:${ext.colour ? objId(ext.colour) : ''}`, name, base);
      }
      case 'rs_ensemble_stats': return this.group(name, name, [{ buffer: this.ensStats }, { buffer: this.ensPixel }, v.frame, v.mask]);
      // ---- M5 (§4.2). inIdx = h for the passes reading the history (refresh fwd, T1, T2; T3: resIn = res[h], resOut =
      // res[1 − h] = res[w]); inIdx = w for refresh inv (resIn = res[w]) and T4 (resOut = res[w]).
      case 'rs_refresh_fwd': case 'rs_refresh_inv':
        this.needTemporal(name);
        return this.group(`${name}:${inIdx}${g}`, name, [rin, arena, v.vbuf, v.geo]);
      case 'rs_t_classify': case 'rs_t_forward':
        this.needTemporal(name);
        return this.group(`${name}:${inIdx}${g}`, name, [rin, arena, v.vbuf, v.geo, v.vbufPrev, v.geoPrev]);
      case 'rs_t_select':
        this.needTemporal(name);
        return this.group(`${name}:${inIdx}${g}`, name, [rin, rout, arena, v.vbuf]);
      case 'rs_t_inverse':
        this.needTemporal(name);
        return this.group(`${name}:${inIdx}${g}`, name, [rin, arena, v.vbuf, v.geo, v.vbufPrev, v.geoPrev]);
      // ---- M6 (restir-m6-api.md §3): tiles write the arena only; the duplication map reads res[inIdx] = the final buffer
      case 'rs_light_tiles': return this.group(name, name, [arena]);
      case 'rs_dupmap': return this.group(`${name}:${inIdx}`, name, [rin, arena]);
    }
  }

  private needTemporal(name: RsPassName): void {
    if (!this.alloc.temporal) throw new Error(`${name}: allocate with temporal = true (settings.temporal)`);
  }

  /** Drop cached bind groups that reference external buffers (accum/counters/colour changed). */
  forgetExternalGroups(): void {
    for (const k of [...this.groups.keys()]) if (k.startsWith('rs_finalize')) this.groups.delete(k);
  }

  destroy(): void {
    for (const x of [...this.res, this.arena, this.args, this.ensStats, this.ensPixel]) x.destroy();
    this.candDump?.destroy();
    for (const t of new Set([...this.vbufs, ...this.geos, this.l1, this.shade, this.frameTex, this.pairTex, this.maskTex])) t.destroy();
    this.groups.clear();
  }
}

const ids = new WeakMap<object, number>();
let nextId = 1;
function objId(b: object): number {
  let i = ids.get(b);
  if (i === undefined) { i = nextId++; ids.set(b, i); }
  return i;
}

/** Uniform buffers shared by every pass of a kernel: RestirParams and the RsDispatch ring (§2.9, D17). */
export function createUniforms(device: GPUDevice): { params: GPUBuffer; ring: GPUBuffer } {
  return {
    params: device.createBuffer({ label: 'rs-params', size: RESTIR_PARAMS_SIZE, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST }),
    ring: device.createBuffer({ label: 'rs-dispatch-ring', size: RS_DISPATCH_RING * RS_DISPATCH_STRIDE, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST }),
  };
}
export { RS_DISPATCH_SIZE };
