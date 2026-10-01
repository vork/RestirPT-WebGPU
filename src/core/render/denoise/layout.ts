// A-SVGF-lite denoiser: constants, uniform packing, debug views, the mode table (docs/decisions/denoiser.md).
// Mirror of shaders/dn-common.wgsl (DnParams, DNF_*, DNV_*) and of the tState / slot-code constants the gradient pass
// reads (restir/tframe.wgsl, restir/types.wgsl; checked against render/restir/layout.ts by tests/denoise).
import type { DebugViewDef } from '../debug-views.ts';

export const DN_PARAMS_SIZE = 64;
export const DN_ITER_SIZE = 16;
/** DnParams.flags (dn-common.wgsl DNF_*). */
export const DNF = { RESET: 1, LAMBDA: 2, HAS_L1: 4, FW: 8, INVERSE: 16, GRADIENT: 32, LAMBDA_CAM: 64 } as const;
/** DnIter.flags (dn-filter.wgsl DNI_*). */
export const DNI = { FEEDBACK: 1, FINAL: 2, COPY: 4 } as const;
/** Reprojection outcome codes (view 526; dn-temporal.wgsl DN_REPROJ_*). */
export const DN_REPROJ = { BG: 0, FULL: 1, PARTIAL: 2, RING: 3, NONE: 4, RESET: 5 } as const;
export const DN_REPROJ_NAMES = ['background', 'bilinear (4 taps)', 'bilinear (partial)', 'ring fallback', 'disoccluded', 'reset'];

/** Mirror of the tState words / flags and slot codes read by dn_gradient (checked against restir/layout.ts). */
export const DN_TSTATE = {
  TSW_QPRIME: 8, TSW_CP: 9, TSW_FWDCODE: 10, TSW_FLAGS: 11, TSW_WC: 13, TSW_WP: 14, TSW_INVCODE: 15, TSW_PIRECOMP: 17, TS_WORDS: 20,
  TS_QVALID: 1, TS_SEL_C: 32, TS_INV_DONE: 128, TS_EMPTY_OUT: 256,
  SC_OK: 0, SC_O0_LIGHT: 6, SC_OCCLUDED: 10, SC_ZERO: 11,
} as const;

export interface DenoiserSettings {
  /** à-trous iterations N (0 = temporal accumulation only). */
  iterations: number;
  /** α_min of the colour and moment blends (SVGF 0.2). */
  alphaMin: number;
  /** Ramp of the gradient: λ′ = clamp((λ − λ0)/(λ1 − λ0), 0, 1). */
  lambda0: number;
  lambda1: number;
  sigmaZ: number;
  sigmaN: number;
  sigmaL: number;
  /** History length cap. */
  nMax: number;
  /** Also use λ on camera-only frames (view-dependent glossy changes; default off, denoiser.md §5). */
  gradientOnCamera: boolean;
}

export const DENOISER_DEFAULTS: Readonly<DenoiserSettings> = {
  iterations: 5, alphaMin: 0.2, lambda0: 0.03, lambda1: 0.15, sigmaZ: 1, sigmaN: 128, sigmaL: 4, nMax: 64, gradientOnCamera: false,
};
export const DN_MAX_ITERATIONS = 6;

export interface DnParamsCpu {
  width: number; height: number; flags: number; settings: DenoiserSettings; tsBase: number; resPlanes: number;
}

export const dnTiles = (w: number, h: number): [number, number] => [Math.ceil(w / 8), Math.ceil(h / 8)];

/** Pack DnParams (64 B, dn-common.wgsl). */
export function packDnParams(o: DnParamsCpu, out = new ArrayBuffer(DN_PARAMS_SIZE)): ArrayBuffer {
  const u = new Uint32Array(out), f = new Float32Array(out);
  const [tx, ty] = dnTiles(o.width, o.height);
  u[0] = o.width; u[1] = o.height; u[2] = tx; u[3] = ty;
  u[4] = o.flags >>> 0;
  f[5] = o.settings.nMax; f[6] = o.settings.alphaMin; f[7] = o.settings.lambda0; f[8] = o.settings.lambda1;
  f[9] = o.settings.sigmaZ; f[10] = o.settings.sigmaN; f[11] = o.settings.sigmaL;
  u[12] = o.tsBase >>> 0; u[13] = o.resPlanes >>> 0; u[14] = 0; u[15] = 0;
  return out;
}

/** The à-trous iteration list for N iterations: [iter, step, flags] (N = 0: one copy pass that writes the output). */
export function atrousPlan(n: number): [number, number, number][] {
  if (n <= 0) return [[0, 0, DNI.COPY | DNI.FEEDBACK | DNI.FINAL]];
  return Array.from({ length: n }, (_, i) => [i, 1 << i, (i === 0 ? DNI.FEEDBACK : 0) | (i === n - 1 ? DNI.FINAL : 0)] as [number, number, number]);
}

// ------------------------------------------------------------------------------------------------ app modes (§8)

export type DenoiseRenderMode = 'pt' | 'albedo' | 'restir';
export type DenoiseRestirMode = 'interactive' | 'unbiased' | 'criteria2022' | 'offline' | 'initial';

/** Whether the denoiser may run (denoiser.md §8): never in ReSTIR-unbiased (the validation-mode preset) or albedo. */
export function denoiserAllowed(render: DenoiseRenderMode, restir: DenoiseRestirMode): boolean {
  if (render === 'albedo') return false;
  if (render === 'restir' && restir === 'unbiased') return false;
  return true;
}
/** Default toggle state of a mode: on in ReSTIR-interactive, off elsewhere. */
export function denoiserDefault(render: DenoiseRenderMode, restir: DenoiseRestirMode): boolean {
  return render === 'restir' && restir === 'interactive';
}
/** Key of the per-mode toggle memory. */
export const denoiseModeKey = (render: DenoiseRenderMode, restir: DenoiseRestirMode): string => (render === 'restir' ? `restir:${restir}` : render);

// ------------------------------------------------------------------------------------------------ debug views (§9)

export const DN_VIEW = {
  variance: 520, history: 521, alpha: 522, lambda: 523, demod: 524, integrated: 525, reproj: 526, pairs: 527, level0: 530,
} as const;
const G = 'Denoiser';
export const DENOISER_VIEWS: DebugViewDef[] = [
  { id: DN_VIEW.variance, key: 'dn.variance', label: 'variance (input of à-trous 0)', group: G, source: 'dn_variance', kind: 'scalar', range: [1e-6, 10], log: true, colormap: 'turbo' },
  { id: DN_VIEW.history, key: 'dn.history', label: 'history length n', group: G, source: 'dn_temporal', kind: 'scalar', range: [0, 32], colormap: 'turbo' },
  { id: DN_VIEW.alpha, key: 'dn.alpha', label: 'α (colour blend)', group: G, source: 'dn_temporal', kind: 'scalar', range: [0, 1], colormap: 'viridis' },
  { id: DN_VIEW.lambda, key: 'dn.lambda', label: 'temporal gradient λ (ReSTIR)', group: G, source: 'dn_temporal', kind: 'scalar', range: [0, 0.5], colormap: 'turbo' },
  { id: DN_VIEW.demod, key: 'dn.demod', label: 'demodulated input', group: G, source: 'dn_temporal', kind: 'vec3', range: [0, 1] },
  { id: DN_VIEW.integrated, key: 'dn.integrated', label: 'temporally integrated (demodulated)', group: G, source: 'dn_temporal', kind: 'vec3', range: [0, 1] },
  { id: DN_VIEW.reproj, key: 'dn.reproj', label: 'reprojection (bg / 4 taps / partial / ring / disoccluded / reset)', group: G, source: 'dn_temporal', kind: 'code' },
  { id: DN_VIEW.pairs, key: 'dn.pairs', label: 'gradient pairs (bit 0 forward, bit 1 inverse)', group: G, source: 'dn_gradient', kind: 'code' },
  ...Array.from({ length: DN_MAX_ITERATIONS }, (_, i): DebugViewDef => ({
    id: DN_VIEW.level0 + i, key: `dn.level[${i}]`, label: `à-trous level ${i} output (demodulated, step ${1 << i})`, group: G, source: 'dn_atrous', kind: 'vec3', range: [0, 1],
  })),
];
