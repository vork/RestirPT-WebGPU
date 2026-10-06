// Pairing maps for paired spatial reuse (restir-api.md §2.8, D9; math.md#paired-mis "pairing textures" [M4 addition]).
// OWNER WP-C.
//
// CPU side of the M4 uniform-disk involution maps and a bit-exact TS mirror of the per-(t, round, member, slot)
// dihedral transform of shaders/restir/pairing.wgsl (tests, inspector):
//   * generatePairLayer(W, R, layer): a W×W torus involution of partner deltas d ∈ [−R, R]², d = (0, 0) = no partner.
//     Deterministic greedy random matching: a PCG32 stream seeded by (layer, R, W); Fisher–Yates order of the W²
//     texels; each still-unmatched texel a draws up to 64 integer offsets d uniformly from {0 < dx² + dy² ≤ R²} and
//     is matched with b = (a + d) mod W if b is unmatched (delta[a] = d, delta[b] = −d). Leftovers keep (0, 0).
//   * pairTexData(R, sizes): the rg8sint 256 × 256 × 8 texture contents (texels outside W_s are 0).
//   * pairTransform / pairPartner: h = pcg4d(runSeed ^ m·φ, t, (r << 8) | s, STREAM_PAIRING), dihedral code h.x & 7,
//     offset (h.y % W_s, h.z % W_s); q = (M·p + o) mod W_s, partner = p + Mᵀ·d(q), valid iff d ≠ 0 and inside the tile.
// The M6 σ = 16 Gaussian maps replace the layers without kernel changes (same format and transform):
//   * generateGaussLayer(W, σ, layer) (restir-m6-api.md MD3, math.md#pairing-textures): link index L = (y·W + x) >> 1, n_σ
//     shuffles of independent random permutations of every 2×2 block (odd passes: block grid offset by (1, 1) on the
//     torus), then the two texels of each link are partners with d = wrap(b − a), d(b) = −d(a) (explicit negation).
//   * nSigma(σ): the corrected Eq. 3 (enh-verify C1), 128 at σ = 16.
import { PAIR_TEX_SIZES, GAUSS_PAIR_SIZES } from './presets.ts';
import { RS_WGSL_CONSTS as K } from './layout.ts';

export const PAIR_TEX_DIM = 256;
export const PAIR_TEX_LAYERS = 8;
/** Match attempts per unmatched texel (§2.8). */
export const PAIR_ATTEMPTS = 64;

// ------------------------------------------------------------------------------------------------ PCG32 (64-bit LCG)

/** PCG32 (O'Neill, XSH-RR) with the 64-bit state held as two u32 words (no BigInt on the hot path). */
export class Pcg32 {
  private hi = 0; private lo = 0;
  private readonly incHi: number; private readonly incLo: number;
  /** pcg32_srandom(initstate, initseq) with 64-bit arguments given as (hi, lo) u32 pairs. */
  constructor(stateHi: number, stateLo: number, seqHi: number, seqLo: number) {
    // inc = (initseq << 1) | 1
    this.incHi = ((seqHi << 1) | (seqLo >>> 31)) >>> 0;
    this.incLo = ((seqLo << 1) | 1) >>> 0;
    this.step();
    this.add(stateHi >>> 0, stateLo >>> 0);
    this.step();
  }
  private add(hi: number, lo: number): void {
    const l = this.lo + lo;
    this.lo = l >>> 0;
    this.hi = (this.hi + hi + (l > 0xFFFFFFFF ? 1 : 0)) >>> 0;
  }
  /** state = state · 6364136223846793005 + inc (mod 2⁶⁴). */
  private step(): void {
    const MH = 0x5851F42D, ML = 0x4C957F2D;
    const a0 = this.lo & 0xFFFF, a1 = this.lo >>> 16, b0 = ML & 0xFFFF, b1 = ML >>> 16;
    const p00 = a0 * b0, p01 = a0 * b1, p10 = a1 * b0, p11 = a1 * b1;
    const mid = (p00 >>> 16) + (p01 & 0xFFFF) + (p10 & 0xFFFF);
    const lo = (((mid & 0xFFFF) << 16) | (p00 & 0xFFFF)) >>> 0;
    let hi = (p11 + (p01 >>> 16) + (p10 >>> 16) + (mid >>> 16)) >>> 0;
    hi = (hi + Math.imul(this.hi, ML) + Math.imul(this.lo, MH)) >>> 0;
    this.hi = hi; this.lo = lo;
    this.add(this.incHi, this.incLo);
  }
  /** Next u32. */
  next(): number {
    const hi = this.hi, lo = this.lo;
    this.step();
    // xorshifted = ((old >> 18) ^ old) >> 27, truncated to 32 bits; rot = old >> 59
    const s18hi = hi >>> 18, s18lo = ((lo >>> 18) | (hi << 14)) >>> 0;
    const xhi = (s18hi ^ hi) >>> 0, xlo = (s18lo ^ lo) >>> 0;
    const xs = ((xlo >>> 27) | (xhi << 5)) >>> 0;
    const rot = hi >>> 27;
    return ((xs >>> rot) | (xs << ((32 - rot) & 31))) >>> 0;
  }
  /** Uniform integer in [0, n) (rejection of the biased low range, as pcg32_boundedrand). */
  bounded(n: number): number {
    const threshold = (0x100000000 - n) % n;
    for (;;) {
      const r = this.next();
      if (r >= threshold) return r % n;
    }
  }
}

/** The generator stream of one layer: state (0x9e3779b9 ^ layer, R << 16 | W), sequence (0, STREAM_PAIRING + layer). */
export function pairRng(layer: number, R: number, W: number): Pcg32 {
  return new Pcg32((0x9e3779b9 ^ layer) >>> 0, ((R << 16) | W) >>> 0, 0, (K.STREAM_PAIRING + layer) >>> 0);
}

// ------------------------------------------------------------------------------------------------ generator

export interface PairLayer {
  /** R: disk radius (M4 maps) or 0 for the Gaussian maps (then `sigma` is set). */
  W: number; R: number; layer: number; sigma?: number;
  /** Partner deltas (dx, dy) per texel, index 2·(y·W + x). */
  delta: Int8Array;
  unmatched: number;
}

/** Integer offset drawn uniformly from {0 < dx² + dy² ≤ R²}. */
function diskOffset(rng: Pcg32, R: number): [number, number] {
  const n = 2 * R + 1, R2 = R * R;
  for (;;) {
    const dx = rng.bounded(n) - R, dy = rng.bounded(n) - R;
    const r2 = dx * dx + dy * dy;
    if (r2 > 0 && r2 <= R2) return [dx, dy];
  }
}

/** One W×W torus involution layer (§2.8). W even, 2R + 1 ≤ W, R ≤ 127. */
export function generatePairLayer(W: number, R: number, layer: number): PairLayer {
  if (!(W > 0 && W <= 254 && W % 2 === 0)) throw new Error(`pairing: W_s = ${W} must be even and ≤ 254`);
  if (!(R >= 1 && R <= 127 && 2 * R + 1 <= W)) throw new Error(`pairing: R = ${R} must be in [1, 127] with 2R + 1 ≤ W_s`);
  const rng = pairRng(layer, R, W);
  const N = W * W;
  const order = new Uint32Array(N);
  for (let i = 0; i < N; i++) order[i] = i;
  for (let i = N - 1; i > 0; i--) {
    const j = rng.bounded(i + 1);
    const t = order[i]; order[i] = order[j]; order[j] = t;
  }
  const matched = new Uint8Array(N);
  const delta = new Int8Array(2 * N);
  for (let i = 0; i < N; i++) {
    const a = order[i];
    if (matched[a]) continue;
    const ax = a % W, ay = (a - ax) / W;
    for (let k = 0; k < PAIR_ATTEMPTS; k++) {
      const [dx, dy] = diskOffset(rng, R);
      const bx = (ax + dx + W) % W, by = (ay + dy + W) % W;
      const b = by * W + bx;
      if (b === a || matched[b]) continue;
      matched[a] = 1; matched[b] = 1;
      delta[2 * a] = dx; delta[2 * a + 1] = dy;
      delta[2 * b] = -dx; delta[2 * b + 1] = -dy;
      break;
    }
  }
  // A texel whose own attempts failed may still have been matched later as some other texel's b.
  let unmatched = 0;
  for (let a = 0; a < N; a++) unmatched += 1 - matched[a];
  return { W, R, layer, delta, unmatched };
}

const layerCache = new Map<string, PairLayer>();
/** Cached generatePairLayer (the maps depend only on (W, R, layer)). */
export function pairLayer(W: number, R: number, layer: number): PairLayer {
  const key = `${W}:${R}:${layer}`;
  let l = layerCache.get(key);
  if (!l) { l = generatePairLayer(W, R, layer); layerCache.set(key, l); }
  return l;
}

/** Layers of the given logical sizes (0 = unused layer, all zeros). */
export function pairLayers(R: number, sizes: readonly number[] = PAIR_TEX_SIZES): (PairLayer | undefined)[] {
  return sizes.slice(0, PAIR_TEX_LAYERS).map((W, s) => (W > 0 ? pairLayer(W, R, s) : undefined));
}

// ------------------------------------------------------------------------------------------------ Gaussian maps (M6)

/** Corrected shuffle count n_σ = ⌊σ²/2 + 1.46/σ − 1.76/σ² + 0.656/σ³ + 0.5⌋ (math.md#dupmap; 128 at σ = 16). */
export function nSigma(sigma: number): number {
  return Math.floor(sigma * sigma / 2 + 1.46 / sigma - 1.76 / (sigma * sigma) + 0.656 / (sigma * sigma * sigma) + 0.5);
}

/** The 24 permutations of 4 (index = the uniform draw). */
const PERMS4: readonly (readonly [number, number, number, number])[] = (() => {
  const out: [number, number, number, number][] = [];
  for (let a = 0; a < 4; a++) for (let b = 0; b < 4; b++) for (let c = 0; c < 4; c++) for (let d = 0; d < 4; d++) {
    if (a !== b && a !== c && a !== d && b !== c && b !== d && c !== d) out.push([a, b, c, d]);
  }
  return out;
})();

/** Generator stream of a Gaussian layer: state (0x7f4a7c15 ^ layer, ⌊16σ⌋ << 16 | W), sequence (0, STREAM_PAIRING + 256 + layer). */
export function gaussPairRng(layer: number, sigma: number, W: number): Pcg32 {
  return new Pcg32((0x7f4a7c15 ^ layer) >>> 0, ((Math.floor(16 * sigma) << 16) | W) >>> 0, 0, (K.STREAM_PAIRING + 256 + layer) >>> 0);
}

const wrapDelta = (v: number, W: number) => (v > W / 2 ? v - W : v < -W / 2 ? v + W : v);

/** One W×W Gaussian reciprocal layer (MD3): every texel has a partner, |d| ≤ W/2 ≤ 127. W even, ≤ 254. */
export function generateGaussLayer(W: number, sigma: number, layer: number): PairLayer {
  if (!(W > 0 && W <= 254 && W % 2 === 0)) throw new Error(`pairing: W_s = ${W} must be even and ≤ 254`);
  if (!(sigma >= 0.8 && sigma <= 40)) throw new Error(`pairing: σ = ${sigma} must be in [0.8, 40]`);
  const rng = gaussPairRng(layer, sigma, W);
  const N = W * W, H2 = W / 2;
  const L = new Uint32Array(N);
  for (let i = 0; i < N; i++) L[i] = i >>> 1;
  const n = nSigma(sigma);
  const idx = [0, 0, 0, 0], val = [0, 0, 0, 0];
  for (let s = 0; s < n; s++) {
    const o = s & 1;
    for (let by = 0; by < H2; by++) {
      const y0 = (2 * by + o) % W, y1 = (2 * by + 1 + o) % W;
      for (let bx = 0; bx < H2; bx++) {
        const x0 = (2 * bx + o) % W, x1 = (2 * bx + 1 + o) % W;
        idx[0] = y0 * W + x0; idx[1] = y0 * W + x1; idx[2] = y1 * W + x0; idx[3] = y1 * W + x1;
        const p = PERMS4[rng.bounded(24)];
        for (let k = 0; k < 4; k++) val[k] = L[idx[k]];
        for (let k = 0; k < 4; k++) L[idx[k]] = val[p[k]];
      }
    }
  }
  const first = new Int32Array(N / 2).fill(-1);
  const delta = new Int8Array(2 * N);
  let paired = 0;
  for (let i = 0; i < N; i++) {
    const l = L[i];
    if (first[l] < 0) { first[l] = i; continue; }
    const a = first[l], ax = a % W, ay = (a - ax) / W, bx = i % W, by = (i - bx) / W;
    const dx = wrapDelta(bx - ax, W), dy = wrapDelta(by - ay, W);
    delta[2 * a] = dx; delta[2 * a + 1] = dy;
    delta[2 * i] = -dx; delta[2 * i + 1] = -dy;
    paired += 2;
  }
  return { W, R: 0, sigma, layer, delta, unmatched: N - paired };
}

const gaussCache = new Map<string, PairLayer>();
/** Cached generateGaussLayer. */
export function gaussLayer(W: number, sigma: number, layer: number): PairLayer {
  const key = `${W}:${sigma}:${layer}`;
  let l = gaussCache.get(key);
  if (!l) { l = generateGaussLayer(W, sigma, layer); gaussCache.set(key, l); }
  return l;
}

/** Gaussian layers of the given logical sizes (0 = unused). */
export function gaussLayers(sigma: number, sizes: readonly number[] = GAUSS_PAIR_SIZES): (PairLayer | undefined)[] {
  return sizes.slice(0, PAIR_TEX_LAYERS).map((W, s) => (W > 0 ? gaussLayer(W, sigma, s) : undefined));
}

/** Per-axis standard deviations and the partner statistics of a layer (T14). */
export function layerDeltaStats(l: PairLayer): { sx: number; sy: number; maxAbs: number; zero: number; n: number } {
  let sxx = 0, syy = 0, maxAbs = 0, zero = 0, n = 0;
  for (let i = 0; i < l.delta.length; i += 2) {
    const dx = l.delta[i], dy = l.delta[i + 1];
    if (dx === 0 && dy === 0) { zero++; continue; }
    sxx += dx * dx; syy += dy * dy; n++;
    maxAbs = Math.max(maxAbs, Math.abs(dx), Math.abs(dy));
  }
  return { sx: Math.sqrt(sxx / n), sy: Math.sqrt(syy / n), maxAbs, zero, n };
}

/** rg8sint texture contents (256 × 256 × 8, row 0 = texel y 0): layer s holds its W_s × W_s map, zeros elsewhere. */
export function pairTexData(R: number, sizes: readonly number[] = PAIR_TEX_SIZES): Int8Array {
  return layersTexData(pairLayers(R, sizes));
}

/** rg8sint texture contents of any layer list (M4 disk or M6 Gaussian). */
export function layersTexData(layers: (PairLayer | undefined)[]): Int8Array {
  const D = PAIR_TEX_DIM;
  const out = new Int8Array(D * D * PAIR_TEX_LAYERS * 2);
  layers.forEach((l, s) => {
    if (!l) return;
    for (let y = 0; y < l.W; y++) {
      out.set(l.delta.subarray(2 * y * l.W, 2 * (y + 1) * l.W), 2 * (s * D * D + y * D));
    }
  });
  return out;
}

/** Upload the maps of radius R into the kernel's pairing texture (256 × 256 × 8 rg8sint). */
export function uploadPairTex(device: GPUDevice, tex: GPUTexture, R: number, sizes: readonly number[] = PAIR_TEX_SIZES): void {
  uploadLayers(device, tex, pairLayers(R, sizes));
}

/** Upload the M6 Gaussian maps of std σ (sizes GAUSS_PAIR_SIZES unless given). */
export function uploadGaussPairTex(device: GPUDevice, tex: GPUTexture, sigma: number, sizes: readonly number[] = GAUSS_PAIR_SIZES): void {
  uploadLayers(device, tex, gaussLayers(sigma, sizes));
}

function uploadLayers(device: GPUDevice, tex: GPUTexture, layers: (PairLayer | undefined)[]): void {
  const data = layersTexData(layers);
  const D = PAIR_TEX_DIM;
  device.queue.writeTexture({ texture: tex }, data.buffer as ArrayBuffer, { offset: data.byteOffset, bytesPerRow: D * 2, rowsPerImage: D }, [D, D, PAIR_TEX_LAYERS]);
}

// ------------------------------------------------------------------------------------------------ transform mirror

/** pcg4d (Jarzynski & Olano), bit-exact mirror of common/rng.wgsl. */
export function pcg4d(x: number, y: number, z: number, w: number): [number, number, number, number] {
  let v0 = (Math.imul(x >>> 0, 1664525) + 1013904223) >>> 0;
  let v1 = (Math.imul(y >>> 0, 1664525) + 1013904223) >>> 0;
  let v2 = (Math.imul(z >>> 0, 1664525) + 1013904223) >>> 0;
  let v3 = (Math.imul(w >>> 0, 1664525) + 1013904223) >>> 0;
  v0 = (v0 + Math.imul(v1, v3)) >>> 0; v1 = (v1 + Math.imul(v2, v0)) >>> 0; v2 = (v2 + Math.imul(v0, v1)) >>> 0; v3 = (v3 + Math.imul(v1, v2)) >>> 0;
  v0 = (v0 ^ (v0 >>> 16)) >>> 0; v1 = (v1 ^ (v1 >>> 16)) >>> 0; v2 = (v2 ^ (v2 >>> 16)) >>> 0; v3 = (v3 ^ (v3 >>> 16)) >>> 0;
  v0 = (v0 + Math.imul(v1, v3)) >>> 0; v1 = (v1 + Math.imul(v2, v0)) >>> 0; v2 = (v2 + Math.imul(v0, v1)) >>> 0; v3 = (v3 + Math.imul(v1, v2)) >>> 0;
  return [v0, v1, v2, v3];
}

/** Dihedral code (bit0 swap axes, bit1 negate x, bit2 negate y, applied in that order) — pairing.wgsl dihedral_apply. */
export function dihedralApply(code: number, v: [number, number]): [number, number] {
  let x = v[0], y = v[1];
  if (code & 1) { const t = x; x = y; y = t; }
  if (code & 2) x = -x;
  if (code & 4) y = -y;
  return [x, y];
}
/** Mᵀ = M⁻¹. */
export function dihedralApplyT(code: number, v: [number, number]): [number, number] {
  let x = v[0], y = v[1];
  if (code & 4) y = -y;
  if (code & 2) x = -x;
  if (code & 1) { const t = x; x = y; y = t; }
  return [x, y];
}

export interface PairTransform { code: number; ox: number; oy: number }
/** Transform of (frame t, round r, member m, slot s) for a layer of logical size W_s (§2.8). */
export function pairTransform(runSeed: number, member: number, t: number, round: number, slot: number, Ws: number): PairTransform {
  const h = pcg4d((runSeed ^ Math.imul(member, 0x9e3779b9)) >>> 0, t >>> 0, ((round << 8) | slot) >>> 0, K.STREAM_PAIRING);
  return { code: h[0] & 7, ox: h[1] % Ws, oy: h[2] % Ws };
}

const pmod = (a: number, n: number) => ((a % n) + n) % n;

/** Partner of member-local pixel p (null = no partner / off-tile), with the layer's deltas. */
export function pairPartner(layer: PairLayer, tr: PairTransform, p: [number, number], memberW: number, memberH: number): [number, number] | null {
  const W = layer.W;
  const mp = dihedralApply(tr.code, p);
  const qx = pmod(mp[0] + tr.ox, W), qy = pmod(mp[1] + tr.oy, W);
  const i = 2 * (qy * W + qx);
  const dx = layer.delta[i], dy = layer.delta[i + 1];
  if (dx === 0 && dy === 0) return null;
  const d = dihedralApplyT(tr.code, [dx, dy]);
  const x = p[0] + d[0], y = p[1] + d[1];
  if (x < 0 || y < 0 || x >= memberW || y >= memberH) return null;
  return [x, y];
}

// ------------------------------------------------------------------------------------------------ statistics (T14-M4)

/** Exact radial CDF of the lattice-uniform offset distribution {0 < |d|² ≤ R²}: sorted distinct radii² and CDF values. */
export function latticeRadialCdf(R: number): { r2: number[]; cdf: number[] } {
  const counts = new Map<number, number>();
  let n = 0;
  for (let dy = -R; dy <= R; dy++) for (let dx = -R; dx <= R; dx++) {
    const r2 = dx * dx + dy * dy;
    if (r2 > 0 && r2 <= R * R) { counts.set(r2, (counts.get(r2) ?? 0) + 1); n++; }
  }
  const r2 = [...counts.keys()].sort((a, b) => a - b);
  let acc = 0;
  const cdf = r2.map((k) => (acc += counts.get(k)!) / n);
  return { r2, cdf };
}

/**
 * KS distance between the radial distribution of a layer's matched deltas (each pair counted once per texel) and
 * (a) the lattice-uniform reference (exact target of the generator) and (b) the continuous uniform-disk CDF r²/R²
 * (informational: lattice discretisation alone gives ≈ 0.015 at R = 10).
 */
export function radialKs(layer: PairLayer): { lattice: number; continuous: number; samples: number } {
  const R = layer.R;
  const hist = new Map<number, number>();
  let n = 0;
  for (let i = 0; i < layer.delta.length; i += 2) {
    const dx = layer.delta[i], dy = layer.delta[i + 1];
    if (dx === 0 && dy === 0) continue;
    const r2 = dx * dx + dy * dy;
    hist.set(r2, (hist.get(r2) ?? 0) + 1);
    n++;
  }
  const ref = latticeRadialCdf(R);
  let acc = 0, ksL = 0, ksC = 0, prevEmp = 0;
  for (let i = 0; i < ref.r2.length; i++) {
    const r2 = ref.r2[i];
    acc += hist.get(r2) ?? 0;
    const emp = acc / n;
    ksL = Math.max(ksL, Math.abs(emp - ref.cdf[i]));
    const c = r2 / (R * R);
    ksC = Math.max(ksC, Math.abs(emp - c), Math.abs(prevEmp - c));
    prevEmp = emp;
  }
  return { lattice: ksL, continuous: ksC, samples: n };
}
