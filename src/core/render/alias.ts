// Integer alias table (Vose) with u16 thresholds and the REALIZED pmf (plan §1.4 "Selection"; math.md#light-selection,
// math.md#env-sampling "Two-level integer alias", open item 5).
//
// Layout: n = 2^m entries (m ≥ 1), padded with zero-weight entries that always alias to real entries. Each entry is
// (q, alias) with q ∈ [0, 65535]. Sampling from one u32 hash h (WGSL select.wgsl alias_sample):
//   bucket b = h >> (32 − m)                                  (exactly uniform over the n buckets)
//   t       = (m ≤ 16 ? h : h2) & 0xFFFF                     (h2 = second hash, slot u_sel2, only when m > 16)
//   entry   = (t < q_b) ? b : alias_b
// so P(keep b) = q_b/65536 exactly and the realized probability of entry k is the exact rational
//   P(k) = [ Σ_b ( [b == k]·q_b + [alias_b == k]·(65536 − q_b) ) ] / (65536·n).
// That P (computed here in f64 from the stored integers, stored as f32) is the pmf used EVERYWHERE (NEE q, p1, ω1, ω2,
// J_P); the target weights are never used as a pmf.
//
// Construction (f64, deterministic: fixed index order, identical float ops for identical inputs):
//   p_k = w_k·n/Σw; small bucket s (p_s < 1): q_s = clamp(round(p_s·65536), 1, 65535) for w_s > 0 (0 for padding),
//   alias_s = the current large bucket l, and l is debited by EXACTLY (65536 − q_s)/65536 (the quantized mass it
//   donates), so quantization errors never accumulate. Leftover buckets: alias = self, q = 65535.
// Deviation from math.md#env-sampling (documented): real entries with w > 0 get q ≥ 1 (clamp lower bound 1 instead of
// 0), so every positive-weight light keeps a positive realized pmf (≥ 1/(65536·n)); otherwise a very dim delta light
// could be rounded to pmf 0, which would drop its contribution (NEE is its only technique).

export const ALIAS_ONE = 65536;
export const ALIAS_Q_MAX = 65535;

export interface AliasTable {
  /** log2 of the padded size n (≥ 1). */
  log2n: number;
  /** n = 2^log2n. */
  n: number;
  /** Number of real (unpadded) entries. */
  count: number;
  /** Thresholds q_b (u16 values). */
  q: Uint32Array;
  /** Alias targets. */
  alias: Uint32Array;
  /** Realized pmf per real entry (exact rational evaluated in f64). */
  pmf: Float64Array;
}

/** Smallest m ≥ 1 with 2^m ≥ count. */
export function aliasLog2(count: number): number {
  let m = 1;
  while ((1 << m) < count) m++;
  if (m > 30) throw new Error(`alias table too large (${count} entries)`);
  return m;
}

/**
 * Build the integer alias table for non-negative weights. Returns undefined when Σw = 0 (nothing can be sampled).
 * `minLog2` pads further (e.g. to keep the table size fixed across rebuilds).
 */
export function buildAliasTable(weights: ArrayLike<number>, minLog2 = 1): AliasTable | undefined {
  const count = weights.length;
  let sum = 0;
  for (let i = 0; i < count; i++) {
    const w = weights[i];
    if (!(w >= 0) || !Number.isFinite(w)) throw new Error(`alias weight ${i} is ${w} (must be finite and ≥ 0)`);
    sum += w;
  }
  if (!(sum > 0)) return undefined;
  const log2n = Math.max(minLog2, aliasLog2(count));
  const n = 2 ** log2n;
  const p = new Float64Array(n);
  for (let i = 0; i < count; i++) p[i] = (weights[i] * n) / sum;
  const q = new Uint32Array(n);
  const alias = new Uint32Array(n);
  // Work lists in index order (stacks popped from the end; the order is fixed, hence deterministic).
  const small: number[] = [];
  const large: number[] = [];
  for (let i = n - 1; i >= 0; i--) (p[i] < 1 ? small : large).push(i);
  while (small.length && large.length) {
    const s = small.pop()!;
    const l = large[large.length - 1];
    const real = s < count && weights[s] > 0;
    const qs = real ? Math.min(ALIAS_Q_MAX, Math.max(1, Math.round(p[s] * ALIAS_ONE))) : 0;
    q[s] = qs;
    alias[s] = l;
    p[l] -= (ALIAS_ONE - qs) / ALIAS_ONE; // debit exactly the donated (quantized) mass
    if (p[l] < 1) { large.pop(); small.push(l); }
  }
  // Leftovers (numerically ≈ 1): keep themselves with q = 65535, alias = self (P(keep) + P(alias) = 1). A zero-weight
  // leftover (possible only through quantization drift) must never be selectable: q = 0, alias = the heaviest entry.
  let heaviest = 0;
  for (let i = 1; i < count; i++) if (weights[i] > weights[heaviest]) heaviest = i;
  for (const i of [...small, ...large]) {
    const real = i < count && weights[i] > 0;
    q[i] = real ? ALIAS_Q_MAX : 0;
    alias[i] = real ? i : heaviest;
  }
  const pmf = realizedPmf(q, alias, count);
  return { log2n, n, count, q, alias, pmf };
}

/** Realized pmf from the stored integers (exact: integer counts, one f64 division). */
export function realizedPmf(q: Uint32Array, alias: Uint32Array, count: number): Float64Array {
  const n = q.length;
  const num = new Float64Array(n); // integer-valued, < 2^53
  for (let b = 0; b < n; b++) {
    num[b] += q[b];
    num[alias[b]] += ALIAS_ONE - q[b];
  }
  const pmf = new Float64Array(count);
  const den = ALIAS_ONE * n;
  for (let k = 0; k < count; k++) pmf[k] = num[k] / den;
  // Padding entries must receive no mass.
  for (let k = count; k < n; k++) if (num[k] !== 0) throw new Error(`alias: padding entry ${k} has realized mass ${num[k]}`);
  return pmf;
}

/** CPU mirror of WGSL alias_sample (select.wgsl): entry index for hashes h (bucket + threshold) and h2 (m > 16). */
export function aliasSample(t: Pick<AliasTable, 'log2n' | 'q' | 'alias'>, h: number, h2 = 0): number {
  const b = (h >>> 0) >>> (32 - t.log2n);
  const thr = (t.log2n > 16 ? h2 : h) & 0xffff;
  return thr < t.q[b] ? b : t.alias[b];
}

/** Interleaved (q, alias) u32 pairs as uploaded to the GPU records buffer. */
export function packAliasEntries(t: AliasTable): Uint32Array {
  const out = new Uint32Array(2 * t.n);
  for (let i = 0; i < t.n; i++) { out[2 * i] = t.q[i]; out[2 * i + 1] = t.alias[i]; }
  return out;
}
