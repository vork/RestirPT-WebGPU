// Reference-compare statistics and display mapping (plan §6 M3a "Compare: Cycles EXR split/flip; relative error;
// t-map"; §7.3 "uncertainty comes from replicates of the statistic, never summed per-pixel variances").
// Images are RGBA float32, rows TOP-DOWN (row 0 = top, as displayed), linear Rec.709 radiance.
// Pure functions, unit-tested in tests/editor/compare-math.test.ts.

export interface RgbaImage { width: number; height: number; data: Float32Array }

export const lum = (r: number, g: number, b: number): number => 0.2126 * r + 0.7152 * g + 0.0722 * b;

export function sameSize(a: RgbaImage, b: RgbaImage): boolean { return a.width === b.width && a.height === b.height; }

/** Per-pixel mean of replicates (all the same size). */
export function meanImage(imgs: readonly RgbaImage[]): RgbaImage {
  if (!imgs.length) throw new Error('meanImage: no images');
  const { width, height } = imgs[0];
  const out = new Float32Array(width * height * 4);
  for (const im of imgs) {
    if (!sameSize(im, imgs[0])) throw new Error('meanImage: size mismatch');
    for (let i = 0; i < out.length; i++) out[i] += im.data[i];
  }
  for (let i = 0; i < out.length; i++) out[i] /= imgs.length;
  return { width, height, data: out };
}

/** Nearest-neighbour resample (only used to display mismatched sizes; statistics require equal sizes). */
export function resampleNearest(im: RgbaImage, w: number, h: number): RgbaImage {
  if (im.width === w && im.height === h) return im;
  const out = new Float32Array(w * h * 4);
  for (let y = 0; y < h; y++) {
    const sy = Math.min(im.height - 1, Math.floor(((y + 0.5) * im.height) / h));
    for (let x = 0; x < w; x++) {
      const sx = Math.min(im.width - 1, Math.floor(((x + 0.5) * im.width) / w));
      out.set(im.data.subarray((sy * im.width + sx) * 4, (sy * im.width + sx) * 4 + 4), (y * w + x) * 4);
    }
  }
  return { width: w, height: h, data: out };
}

/** |Y_ours − Y_ref| / (Y_ref + eps) per pixel (luminance). */
export function relativeError(ours: RgbaImage, ref: RgbaImage, eps = 1e-3): Float32Array {
  if (!sameSize(ours, ref)) throw new Error('relativeError: size mismatch');
  const n = ours.width * ours.height;
  const out = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    const a = lum(ours.data[4 * i], ours.data[4 * i + 1], ours.data[4 * i + 2]);
    const b = lum(ref.data[4 * i], ref.data[4 * i + 1], ref.data[4 * i + 2]);
    out[i] = Math.abs(a - b) / (Math.abs(b) + eps);
  }
  return out;
}

/** Per-replicate tile means of luminance: [rep][tile]. */
export function tileMeans(imgs: readonly RgbaImage[], tile: number): { tx: number; ty: number; means: Float64Array[] } {
  const { width, height } = imgs[0];
  const tx = Math.ceil(width / tile), ty = Math.ceil(height / tile);
  const means = imgs.map((im) => {
    if (!sameSize(im, imgs[0])) throw new Error('tileMeans: size mismatch');
    const s = new Float64Array(tx * ty), c = new Float64Array(tx * ty);
    for (let y = 0; y < height; y++) for (let x = 0; x < width; x++) {
      const i = (y * width + x) * 4;
      const k = Math.floor(y / tile) * tx + Math.floor(x / tile);
      s[k] += lum(im.data[i], im.data[i + 1], im.data[i + 2]);
      c[k]++;
    }
    for (let k = 0; k < s.length; k++) s[k] /= c[k];
    return s;
  });
  return { tx, ty, means };
}

export interface WelchResult { t: number; df: number; diff: number; se: number }

/** Welch t between replicate samples a and b (either may have one replicate: its variance is then taken as 0). */
export function welch(a: ArrayLike<number>, b: ArrayLike<number>): WelchResult {
  const m = (x: ArrayLike<number>) => { let s = 0; for (let i = 0; i < x.length; i++) s += x[i]; return s / x.length; };
  const v = (x: ArrayLike<number>, mu: number) => { if (x.length < 2) return 0; let s = 0; for (let i = 0; i < x.length; i++) s += (x[i] - mu) ** 2; return s / (x.length - 1); };
  const ma = m(a), mb = m(b);
  const va = v(a, ma) / a.length, vb = v(b, mb) / b.length;
  const se = Math.sqrt(va + vb);
  const diff = ma - mb;
  const num = (va + vb) ** 2;
  const den = (a.length > 1 ? va * va / (a.length - 1) : 0) + (b.length > 1 ? vb * vb / (b.length - 1) : 0);
  const df = den > 0 ? num / den : Infinity;
  const t = se > 0 ? diff / se : (diff === 0 ? 0 : Math.sign(diff) * Infinity);
  return { t, df, diff, se };
}

export interface TMap { tx: number; ty: number; tile: number; t: Float32Array; valid: boolean; note: string }

/** Tile t-map: Welch t per tile on per-replicate tile means (ours batches vs Cycles seeds). */
export function tileTMap(ours: readonly RgbaImage[], ref: readonly RgbaImage[], tile = 16): TMap {
  if (!ours.length || !ref.length) throw new Error('tileTMap: need images');
  if (!sameSize(ours[0], ref[0])) throw new Error('tileTMap: size mismatch');
  const A = tileMeans(ours, tile), B = tileMeans(ref, tile);
  const n = A.tx * A.ty;
  const t = new Float32Array(n);
  const a = new Float64Array(ours.length), b = new Float64Array(ref.length);
  for (let k = 0; k < n; k++) {
    for (let r = 0; r < ours.length; r++) a[r] = A.means[r][k];
    for (let r = 0; r < ref.length; r++) b[r] = B.means[r][k];
    t[k] = welch(a, b).t;
  }
  const valid = ours.length + ref.length >= 3 && (ours.length > 1 || ref.length > 1);
  const note = ours.length > 1 && ref.length > 1 ? `Welch t, ${ours.length} batches vs ${ref.length} seeds`
    : ref.length > 1 ? `t with the Cycles seed variance only (${ref.length} seeds; capture ≥ 2 batches for ours)`
      : ours.length > 1 ? `t with our batch variance only (${ours.length} batches; render ≥ 2 Cycles seeds)`
        : 'no replicates: t undefined (need ≥ 2 seeds or batches)';
  return { tx: A.tx, ty: A.ty, tile, t, valid, note };
}

export interface CompareSummary {
  /** Mean luminance and per-channel means. */
  ours: [number, number, number, number];
  ref: [number, number, number, number];
  /** ours / ref − 1 per channel (r, g, b, Y). */
  relDiff: [number, number, number, number];
  /** Global Welch t on per-replicate image-mean luminance. */
  globalT: WelchResult;
  meanRelErr: number;
  /** Fraction of tiles with |t| > 3.3 (≈ 1e-3 two-sided). */
  tilesOver: number;
}

function imageMeans(im: RgbaImage): [number, number, number, number] {
  let r = 0, g = 0, b = 0;
  const n = im.width * im.height;
  for (let i = 0; i < n; i++) { r += im.data[4 * i]; g += im.data[4 * i + 1]; b += im.data[4 * i + 2]; }
  return [r / n, g / n, b / n, lum(r / n, g / n, b / n)];
}

export function summarize(ours: readonly RgbaImage[], ref: readonly RgbaImage[], tile = 16): CompareSummary {
  const mo = meanImage(ours), mr = meanImage(ref);
  const o = imageMeans(mo), r = imageMeans(mr);
  const rel = relativeError(mo, mr);
  let s = 0;
  for (const x of rel) s += x;
  const tm = tileTMap(ours, ref, tile);
  let over = 0;
  for (const x of tm.t) if (Math.abs(x) > 3.3) over++;
  return {
    ours: o, ref: r,
    relDiff: [0, 1, 2, 3].map((i) => (r[i] !== 0 ? o[i] / r[i] - 1 : 0)) as [number, number, number, number],
    globalT: welch(ours.map((im) => imageMeans(im)[3]), ref.map((im) => imageMeans(im)[3])),
    meanRelErr: s / rel.length,
    tilesOver: tm.valid ? over / tm.t.length : NaN,
  };
}

// ---- display ------------------------------------------------------------------------------------------------------

/** Blender 'Standard' view transform: exposure, clamp, sRGB OETF → 8 bit. */
export function srgb8(x: number): number {
  if (!(x > 0)) return 0; // also NaN
  const c = Math.min(1, x);
  const v = c <= 0.0031308 ? 12.92 * c : 1.055 * c ** (1 / 2.4) - 0.055;
  return Math.round(v * 255);
}

export function toneMap(im: RgbaImage, exposureEV = 0, out = new Uint8ClampedArray(im.width * im.height * 4)): Uint8ClampedArray {
  const k = 2 ** exposureEV;
  for (let i = 0; i < im.width * im.height; i++) {
    out[4 * i] = srgb8(im.data[4 * i] * k);
    out[4 * i + 1] = srgb8(im.data[4 * i + 1] * k);
    out[4 * i + 2] = srgb8(im.data[4 * i + 2] * k);
    out[4 * i + 3] = 255;
  }
  return out;
}

/** Sequential map for [0, max] (black → purple → orange → yellow), NaN → magenta. */
export function heat(v: number, max: number): [number, number, number] {
  if (!Number.isFinite(v)) return [255, 0, 255];
  const x = Math.min(1, Math.max(0, v / max));
  const stops: [number, number, number][] = [[0, 0, 4], [81, 18, 124], [183, 55, 121], [252, 137, 97], [252, 253, 191]];
  const f = x * (stops.length - 1);
  const i = Math.min(stops.length - 2, Math.floor(f));
  const u = f - i;
  return [0, 1, 2].map((c) => Math.round(stops[i][c] + (stops[i + 1][c] - stops[i][c]) * u)) as [number, number, number];
}

/** Diverging map for t in [−lim, lim] (blue ← white → red), non-finite → magenta. */
export function diverging(t: number, lim = 4): [number, number, number] {
  if (!Number.isFinite(t)) return [255, 0, 255];
  const x = Math.max(-1, Math.min(1, t / lim));
  if (x >= 0) return [255, Math.round(255 * (1 - x)), Math.round(255 * (1 - x))];
  return [Math.round(255 * (1 + x)), Math.round(255 * (1 + x)), 255];
}

/** Decode IEEE half floats (rgba16float readback). */
export function halfToFloat(h: number): number {
  const s = h & 0x8000 ? -1 : 1;
  const e = (h >> 10) & 0x1f;
  const f = h & 0x3ff;
  if (e === 0) return s * 2 ** -14 * (f / 1024);
  if (e === 31) return f ? NaN : s * Infinity;
  return s * 2 ** (e - 15) * (1 + f / 1024);
}
