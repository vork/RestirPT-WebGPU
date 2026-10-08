// A-SVGF-lite denoiser: constants, uniform packing, debug views, the mode table (docs/decisions/denoiser.md).
// Mirror of shaders/dn-common.wgsl (DnParams, DNF_*, DNV_*) and of the tState / slot-code constants the gradient pass
// reads (restir/tframe.wgsl, restir/types.wgsl; checked against render/restir/layout.ts by tests/denoise).
import type { DebugViewDef } from '../debug-views.ts';

export const DN_PARAMS_SIZE = 112;
export const DN_ITER_SIZE = 16;
/** DnParams.taaFlags (dn-common.wgsl DNT_*; Changelog DN-16). */
export const DNT = { DILATE: 1, CUBIC: 2 } as const;
/** DnParams.flags (dn-common.wgsl DNF_*). */
export const DNF = { RESET: 1, LAMBDA: 2, HAS_L1: 4, FW: 8, INVERSE: 16, GRADIENT: 32, LAMBDA_CAM: 64, NO_RESOLVE: 128, GUIDE: 256 } as const;
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
  /** σ_a of the albedo edge stop on the accumulated demodulation factor (Changelog DN-5; 0 = off). */
  sigmaA: number;
  /** K: the edge stops use min(1, K·α/(2 − α)) × the sample variance (the integrated colour's, with K frames per
   *  independent sample; Changelog DN-4). 0 = the sample variance (SVGF). */
  varCorr: number;
  /** Temporal resolve of the output (Changelog DN-6): accumulate the denoised output (static: the progressive mean of
   *  the denoised frames, up to nMaxT; in motion / under lighting changes: ≤ 8 frames, variance-clipped). */
  resolve: boolean;
  nMaxT: number;
  /** DN-12: the à-trous luminance stop applies from this colour-history length on (young histories: geometric and
   *  albedo stops only, which are independent of the noisy values: mean-preserving). */
  lumMinN: number;
  /** DN-13: radius of the luminance-guide prefilter (0 = each tap's own value, as SVGF). */
  lumPre: number;
  /** DN-10: tile radius of the inverse gradient family's window (forward: 3×3 tiles). */
  invRadius: number;
  /** DN-9: the à-trous luminance stop uses the converged previous output once camera and lighting are static. */
  guide: boolean;
  /** History length cap. */
  nMax: number;
  /** Also use λ on camera-only frames (view-dependent glossy changes; default off, denoiser.md §5). */
  gradientOnCamera: boolean;
  /** Changelog DN-16: history cap of the output resolve on static-camera frames within 8 frames of a lighting change
   *  (variance-clipped; the gradient's λ′ still cuts it). */
  taaLightMax: number;
  /** Changelog DN-16: history cap of the output resolve while the camera moves (variance-clipped). */
  taaCamMax: number;
  /** Changelog DN-16: in motion, reproject with the motion vector of the closest hit in the 3×3 neighbourhood. */
  taaDilate: boolean;
  /** Changelog DN-16: in motion, fetch the output history with a Catmull-Rom (4×4) filter instead of bilinear. */
  taaCubic: boolean;
  /** Changelog DN-16: variance-clipping width γ (YCoCg μ ± γσ of the current 3×3) in the lighting window (static
   *  camera) and in motion; 0 = no clipping. */
  taaGammaLight: number;
  taaGammaCam: number;
  /** M8 P-6 (m8-perf.md §7): the step-1 à-trous level reads its taps from a workgroup-memory tile (bitwise the texture
   *  path, U-DN-3c). */
  atrousTile: boolean;
}

export const DENOISER_DEFAULTS: Readonly<DenoiserSettings> = {
  iterations: 4, alphaMin: 0.2, lambda0: 0.03, lambda1: 0.15, sigmaZ: 1, sigmaN: 128, sigmaL: 4, sigmaA: 0.05, varCorr: 3, resolve: true, nMaxT: 1024, guide: true, invRadius: 3, lumMinN: 4, lumPre: 1, nMax: 64, gradientOnCamera: false,
  taaLightMax: 32, taaCamMax: 8, taaDilate: true, taaCubic: true, taaGammaLight: 4, taaGammaCam: 1, atrousTile: true,
};
export const DN_MAX_ITERATIONS = 6;

export interface DnParamsCpu {
  width: number; height: number; flags: number; settings: DenoiserSettings; tsBase: number; resPlanes: number;
  /** Frames since the last lighting change (0 = this frame; Changelog DN-6). */
  sinceChange?: number;
}

export const dnTiles = (w: number, h: number): [number, number] => [Math.ceil(w / 8), Math.ceil(h / 8)];

/** Pack DnParams (112 B, dn-common.wgsl). */
export function packDnParams(o: DnParamsCpu, out = new ArrayBuffer(DN_PARAMS_SIZE)): ArrayBuffer {
  const u = new Uint32Array(out), f = new Float32Array(out);
  const [tx, ty] = dnTiles(o.width, o.height);
  u[0] = o.width; u[1] = o.height; u[2] = tx; u[3] = ty;
  u[4] = o.flags >>> 0;
  f[5] = o.settings.nMax; f[6] = o.settings.alphaMin; f[7] = o.settings.lambda0; f[8] = o.settings.lambda1;
  f[9] = o.settings.sigmaZ; f[10] = o.settings.sigmaN; f[11] = o.settings.sigmaL;
  u[12] = o.tsBase >>> 0; u[13] = o.resPlanes >>> 0; f[14] = o.settings.sigmaA; f[15] = o.settings.varCorr; f[16] = o.settings.nMaxT; u[17] = Math.min(o.sinceChange ?? 0xffff, 0xffff); u[18] = o.settings.invRadius; f[19] = o.settings.lumMinN; u[20] = o.settings.lumPre;
  f[21] = o.settings.taaLightMax; f[22] = o.settings.taaCamMax; u[23] = (o.settings.taaDilate ? DNT.DILATE : 0) | (o.settings.taaCubic ? DNT.CUBIC : 0);
  f[24] = o.settings.taaGammaLight; f[25] = o.settings.taaGammaCam;
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
  variance: 520, history: 521, alpha: 522, lambda: 523, demod: 524, integrated: 525, reproj: 526, pairs: 527, taaN: 528, level0: 530, demodFactor: 540, albedoAccum: 541, demodCheck: 542,
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
  { id: DN_VIEW.taaN, key: 'dn.resolveN', label: 'output resolve history length n_t', group: G, source: 'dn_resolve', kind: 'scalar', range: [0, 64], colormap: 'turbo' },
  { id: DN_VIEW.pairs, key: 'dn.pairs', label: 'gradient pairs (bit 0 forward, bit 1 inverse)', group: G, source: 'dn_gradient', kind: 'code' },
  ...Array.from({ length: DN_MAX_ITERATIONS }, (_, i): DebugViewDef => ({
    id: DN_VIEW.level0 + i, key: `dn.level[${i}]`, label: `à-trous level ${i} output (demodulated, step ${1 << i})`, group: G, source: 'dn_atrous', kind: 'vec3', range: [0, 1],
  })),
  { id: DN_VIEW.demodFactor, key: 'dn.demodFactor', label: 'a′ of this frame (demodulation factor)', group: G, source: 'dn_temporal', kind: 'vec3', range: [0, 1] },
  { id: DN_VIEW.albedoAccum, key: 'dn.albedoAccum', label: 'ā (accumulated a′, remodulates the output)', group: G, source: 'dn_temporal', kind: 'vec3', range: [0, 1] },
  { id: DN_VIEW.demodCheck, key: 'dn.demodCheck', label: 'lum(c·a′)/lum(L − L1) (must be 1)', group: G, source: 'dn_temporal', kind: 'scalar', range: [0.99, 1.01], colormap: 'signed' },
];
