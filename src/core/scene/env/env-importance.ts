// Environment importance tables (plan §1.4b "Importance sampling", §2 rule 15; math.md#env-sampling, env §2.4).
// Pure f64 TypeScript with a fixed loop order (deterministic: identical texels + options → identical bits). Built in
// a Worker by the app (env-importance.worker.ts) and inline by validation batches and tests.
//
// Grid: W_m = the largest power of two ≤ min(W, cap), W_m ≥ 4, H_m = W_m/2; row i (bottom-up in v) covers
// v ∈ [i/H_m, (i+1)/H_m), column j covers u ∈ [j/W_m, (j+1)/W_m).
// Cell weights (strength and tint EXCLUDED, so they never change the table):
//   a[r][c] = (|R|+|G|+|B|)/3 of the raw texels
//   B       = 3×3 kernel k = (1/8, 6/8, 1/8) ⊗ k over a, u periodic, v periodic (row −1 ≡ row H−1: Cycles' repeat
//             quirk). B is the exact integral of the bilinear reconstruction over a texel footprint.
//   B̂_ij    = area-weighted mean of B over the cell (texel footprints overlapping the cell; the block mean when W is a
//             multiple of W_m)
//   s̄_i     = (H_m/π)(cos(π i/H_m) − cos(π(i+1)/H_m))     (row mean of sinθ = sin(πv))
//   w_ij    = s̄_i·B̂_ij, then the defensive floors (2⁻¹⁰ of the all-cell mean, of the row mean, and a marginal floor
//             2⁻¹⁰·mean(R) on the row totals), which make every realized cell probability positive.
// Two-level integer alias (rows, then columns within the row), u16 thresholds, packed entry = (alias << 16) | q.
// pdfUV[i·W_m + j] = P_row(i)·P_col(j|i)·W_m·H_m with P from the STORED INTEGERS (alias.ts realizedPmf), never from w.
// Also: the per-channel env mean M = (1/4π)Σ texel·ΔΩ_r at native resolution (for Φ_env = 4π²R_s²·strength·lum(tint⊙M),
// math.md#light-selection) and, for diagnostics, the normalized target density (w_ij / mean w).
import { buildAliasTable, realizedPmf } from '../../render/alias.ts';

export const ENV_IMPORTANCE_CAP_VALIDATION = 4096;
export const ENV_IMPORTANCE_CAP_INTERACTIVE = 2048;
export const ENV_FLOOR = 2 ** -10;
const LUM = [0.2126, 0.7152, 0.0722];

export interface EnvImportanceOptions {
  /** Upper bound of W_m (default 4096; the app uses 2048). The "importance resolution" setting. */
  cap?: number;
  /** Defensive floors (default true). false = the "no floors" negative control (still unbiased: BSDF covers p1 = 0). */
  floors?: boolean;
  /**
   * Planted bias (validation only): store the Cycles-style TARGET density (texel-centre avg(|rgb|) × row-centre sinθ,
   * normalized) as pdfUV while sampling with the kernel + floor alias (env §5.3 "pdf from target weights").
   */
  plantPdfFromTargets?: boolean;
}

export interface EnvImportance {
  /** Source texture size. */
  width: number;
  height: number;
  /** Importance grid. */
  Wm: number;
  Hm: number;
  log2W: number;
  log2H: number;
  /** Row alias entries (alias << 16 | q), H_m. */
  rowAlias: Uint32Array;
  /** Column alias entries per row (alias << 16 | q), H_m × W_m, row-major. */
  colAlias: Uint32Array;
  /** Realized density in (u, v) per cell (f32), H_m × W_m, row-major; ∫ pdfUV du dv = 1. */
  pdfUV: Float32Array;
  /** Normalized target density w_ij / mean(w) after the floors (diagnostics: realized/target ratio). */
  targetUV: Float32Array;
  /** (1/4π)·Σ texel_c·ΔΩ_r per channel at native resolution (untinted, strength 1). */
  meanRgb: [number, number, number];
  options: Required<EnvImportanceOptions>;
  /** Build time in ms (informational). */
  buildMs: number;
}

/** Importance grid width: the largest power of two ≤ min(W, cap), at least 4 (math.md#env-sampling). */
export function importanceWidth(width: number, cap = ENV_IMPORTANCE_CAP_VALIDATION): number {
  const lim = Math.max(4, Math.min(width, cap));
  let w = 4;
  while (w * 2 <= lim) w *= 2;
  if (w > 65536) throw new Error(`env importance width ${w} > 2^16`);
  return w;
}

/**
 * Overlap weights of source texels [0, n) onto m equal cells over [0, 1): for cell k the list of (texel, fraction of
 * the cell covered by that texel). Separable per axis.
 */
function overlaps(n: number, m: number): { idx: Int32Array; wt: Float64Array; start: Int32Array } {
  const idx: number[] = [], wt: number[] = [], start = new Int32Array(m + 1);
  for (let k = 0; k < m; k++) {
    start[k] = idx.length;
    const a = (k * n) / m, b = ((k + 1) * n) / m;       // cell in texel units
    for (let t = Math.floor(a); t < Math.ceil(b); t++) {
      const lo = Math.max(a, t), hi = Math.min(b, t + 1);
      if (hi > lo) { idx.push(t); wt.push((hi - lo) / (b - a)); }
    }
  }
  start[m] = idx.length;
  return { idx: Int32Array.from(idx), wt: Float64Array.from(wt), start };
}

/** Build the tables from RGBA float32 texels, rows BOTTOM-UP (EnvironmentData.texels). */
export function buildEnvImportance(texels: Float32Array, width: number, height: number, opts: EnvImportanceOptions = {}): EnvImportance {
  const t0 = typeof performance !== 'undefined' ? performance.now() : 0;
  const options: Required<EnvImportanceOptions> = { cap: opts.cap ?? ENV_IMPORTANCE_CAP_VALIDATION, floors: opts.floors ?? true, plantPdfFromTargets: opts.plantPdfFromTargets ?? false };
  if (texels.length !== width * height * 4) throw new Error(`env texels ${texels.length} != ${width}x${height}x4`);
  const W = width, H = height;
  const Wm = importanceWidth(W, options.cap), Hm = Wm / 2;
  const log2W = Math.round(Math.log2(Wm)), log2H = log2W - 1;

  // a = avg |rgb|; env mean per channel (native resolution, exact solid angle per row).
  const a = new Float64Array(W * H);
  const meanRgb: [number, number, number] = [0, 0, 0];
  for (let r = 0; r < H; r++) {
    const dOmega = (2 * Math.PI / W) * (Math.cos(Math.PI * r / H) - Math.cos(Math.PI * (r + 1) / H));   // rows bottom-up: θ = π − πv
    const rs = [0, 0, 0];
    for (let c = 0; c < W; c++) {
      const o = 4 * (r * W + c);
      const R = texels[o], G = texels[o + 1], B = texels[o + 2];
      a[r * W + c] = (Math.abs(R) + Math.abs(G) + Math.abs(B)) / 3;
      rs[0] += R; rs[1] += G; rs[2] += B;
    }
    for (let k = 0; k < 3; k++) meanRgb[k] += rs[k] * dOmega;
  }
  for (let k = 0; k < 3; k++) meanRgb[k] /= 4 * Math.PI;

  // Bilinear-integral kernel, horizontal then vertical, periodic in both u and v.
  const bh = new Float64Array(W * H);
  for (let r = 0; r < H; r++) {
    const o = r * W;
    for (let c = 0; c < W; c++) {
      const l = c === 0 ? W - 1 : c - 1, rr = c === W - 1 ? 0 : c + 1;
      bh[o + c] = 0.125 * a[o + l] + 0.75 * a[o + c] + 0.125 * a[o + rr];
    }
  }
  const Bk = new Float64Array(W * H);
  for (let r = 0; r < H; r++) {
    const dn = (r === 0 ? H - 1 : r - 1) * W, up = (r === H - 1 ? 0 : r + 1) * W, o = r * W;
    for (let c = 0; c < W; c++) Bk[o + c] = 0.125 * bh[dn + c] + 0.75 * bh[o + c] + 0.125 * bh[up + c];
  }

  // Cell means (separable area-weighted overlap) × row mean sinθ.
  const ou = overlaps(W, Wm), ov = overlaps(H, Hm);
  const colMean = new Float64Array(H * Wm);                   // per texel row, per cell column
  for (let r = 0; r < H; r++) {
    for (let j = 0; j < Wm; j++) {
      let s = 0;
      for (let k = ou.start[j]; k < ou.start[j + 1]; k++) s += ou.wt[k] * Bk[r * W + ou.idx[k]];
      colMean[r * Wm + j] = s;
    }
  }
  const w = new Float64Array(Hm * Wm);
  for (let i = 0; i < Hm; i++) {
    const sbar = (Hm / Math.PI) * (Math.cos(Math.PI * i / Hm) - Math.cos(Math.PI * (i + 1) / Hm));
    for (let j = 0; j < Wm; j++) {
      let s = 0;
      for (let k = ov.start[i]; k < ov.start[i + 1]; k++) s += ov.wt[k] * colMean[ov.idx[k] * Wm + j];
      w[i * Wm + j] = sbar * s;
    }
  }

  // Defensive floors (math.md#env-sampling "floors").
  const R = new Float64Array(Hm);
  if (options.floors) {
    let tot = 0;
    for (let k = 0; k < w.length; k++) tot += w[k];
    const fAll = ENV_FLOOR * (tot / w.length);
    for (let k = 0; k < w.length; k++) if (w[k] < fAll) w[k] = fAll;
  }
  for (let i = 0; i < Hm; i++) {
    let s = 0;
    for (let j = 0; j < Wm; j++) s += w[i * Wm + j];
    if (options.floors) {
      const fRow = ENV_FLOOR * s / Wm;
      s = 0;
      for (let j = 0; j < Wm; j++) { const k = i * Wm + j; if (w[k] < fRow) w[k] = fRow; s += w[k]; }
    }
    R[i] = s;
  }
  if (options.floors) {
    let mR = 0;
    for (let i = 0; i < Hm; i++) mR += R[i];
    mR /= Hm;
    for (let i = 0; i < Hm; i++) if (R[i] < ENV_FLOOR * mR) R[i] = ENV_FLOOR * mR;
  }

  // Two-level integer alias and the realized density.
  const rowT = buildAliasTable(R, log2H, { qMinReal: 0 });
  if (!rowT) throw new Error('env importance: all-zero environment (nothing to sample)');
  if (rowT.n !== Hm) throw new Error(`env importance: row table size ${rowT.n} != ${Hm}`);
  const rowAlias = new Uint32Array(Hm);
  for (let i = 0; i < Hm; i++) rowAlias[i] = ((rowT.alias[i] << 16) | rowT.q[i]) >>> 0;
  const rowP = realizedPmf(rowT.q, rowT.alias, Hm);
  const colAlias = new Uint32Array(Hm * Wm);
  const pdfUV = new Float32Array(Hm * Wm);
  const rowW = new Float64Array(Wm);
  const cells = Wm * Hm;
  for (let i = 0; i < Hm; i++) {
    for (let j = 0; j < Wm; j++) rowW[j] = w[i * Wm + j];
    const ct = buildAliasTable(rowW, log2W, { qMinReal: 0 });
    if (!ct) {                                                  // an all-zero row (no floors only): never selected
      for (let j = 0; j < Wm; j++) colAlias[i * Wm + j] = ((j << 16) | 0xffff) >>> 0;
      continue;
    }
    const colP = realizedPmf(ct.q, ct.alias, Wm);
    for (let j = 0; j < Wm; j++) {
      colAlias[i * Wm + j] = ((ct.alias[j] << 16) | ct.q[j]) >>> 0;
      pdfUV[i * Wm + j] = rowP[i] * colP[j] * cells;
    }
  }

  // Target density (floored weights, normalized) and, for the planted bias, the Cycles-style texel-centre target.
  let wSum = 0;
  for (let k = 0; k < cells; k++) wSum += w[k];
  const targetUV = new Float32Array(cells);
  for (let k = 0; k < cells; k++) targetUV[k] = (w[k] * cells) / wSum;
  if (options.plantPdfFromTargets) {
    const f = new Float64Array(cells);
    let fs = 0;
    for (let i = 0; i < Hm; i++) {
      const st = Math.sin(Math.PI * (i + 0.5) / Hm);
      const r = Math.min(H - 1, Math.floor(((i + 0.5) / Hm) * H));
      for (let j = 0; j < Wm; j++) {
        const c = Math.min(W - 1, Math.floor(((j + 0.5) / Wm) * W));
        f[i * Wm + j] = a[r * W + c] * st;
        fs += f[i * Wm + j];
      }
    }
    for (let k = 0; k < cells; k++) pdfUV[k] = (f[k] * cells) / fs;
  }
  const buildMs = typeof performance !== 'undefined' ? performance.now() - t0 : 0;
  return { width, height, Wm, Hm, log2W, log2H, rowAlias, colAlias, pdfUV, targetUV, meanRgb, options, buildMs };
}

/** Env mean radiance L̄_env = strength·lum(tint ⊙ M) (math.md#light-selection). */
export function envMeanLuminance(t: Pick<EnvImportance, 'meanRgb'>, strength: number, tint: readonly number[]): number {
  return strength * (LUM[0] * tint[0] * t.meanRgb[0] + LUM[1] * tint[1] * t.meanRgb[1] + LUM[2] * tint[2] * t.meanRgb[2]);
}

/** Power proxy Φ_env = 4π²·R_s²·L̄_env (math.md#light-selection). */
export function envPowerProxy(t: Pick<EnvImportance, 'meanRgb'>, strength: number, tint: readonly number[], sceneRadius: number): number {
  return 4 * Math.PI * Math.PI * sceneRadius * sceneRadius * Math.max(0, envMeanLuminance(t, strength, tint));
}

/** Realized alias probabilities of one packed table (alias << 16 | q) of n = 2^m entries, exact rationals in f64. */
export function realizedFromPacked(entries: Uint32Array, offset: number, n: number): Float64Array {
  const q = new Uint32Array(n), al = new Uint32Array(n);
  for (let b = 0; b < n; b++) { const e = entries[offset + b]; q[b] = e & 0xffff; al[b] = e >>> 16; }
  return realizedPmf(q, al, n);
}

/** CPU mirror of the WGSL env sampler (lights/env-sample.wgsl env_sample_cell): (i, j) for hashes h0, h1. */
export function envSampleCell(t: Pick<EnvImportance, 'rowAlias' | 'colAlias' | 'Wm' | 'log2W' | 'log2H'>, h0: number, h1: number): [number, number] {
  const i0 = (h0 >>> 0) >>> (32 - t.log2H);
  const er = t.rowAlias[i0];
  const i = (h0 & 0xffff) < (er & 0xffff) ? i0 : er >>> 16;
  const j0 = (h1 >>> 0) >>> (32 - t.log2W);
  const ec = t.colAlias[i * t.Wm + j0];
  const j = (h1 & 0xffff) < (ec & 0xffff) ? j0 : ec >>> 16;
  return [i, j];
}

/** Bytes of the GPU tables (row alias + column alias + pdf) for memory readouts. */
export const envImportanceBytes = (t: Pick<EnvImportance, 'Wm' | 'Hm'>): number => 4 * (t.Hm + 2 * t.Wm * t.Hm);
