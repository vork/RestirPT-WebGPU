// perf2 perf-flag registry (docs/decisions/perf2-api.md; perf2-plan.md §0 rule 2, WP-0 step 1).
//
// Every perf2 optimisation ships behind an interactive-only composer define. The defines are named here, once, so the
// packages working in parallel never collide in kernel.ts. A flag that is off is ABSENT from the define set (never
// `NAME: 0`): the composer treats an undefined identifier in `#if` as false, so validation text (no flags) stays
// byte-identical to the pre-perf2 text — U-M7-BITS holds by construction (tests/scene/m7-wgsl-bits.test.ts asserts no
// registry key in any validation define set).
//
// Flow: RestirKernelOptions.perfFlags → RestirKernel.perfDefines() → every ReSTIR pipeline (kernel.defines(),
// customDefines(): T3 test pipelines, the debug views) and the renderer's M1 primary pass; part of variantKey(), so a
// change recompiles (prepare()) and resets the temporal history. RestirKernel.interactive (the app) defaults to
// RELEASE_PERF_FLAGS; RestirKernel.create (every validation caller) defaults to none.
//
// Adding a flag (a package): the name is already reserved below. When the WGSL lands, set `landed: true` (and
// `release: true` once the package's keep/drop evidence is accepted and it should be on in the app). A name not in the
// registry is rejected by normalizePerfFlags (typos never silently compile the off text).

/** Result class of a flag (perf2-plan.md §0 rule 3): 'bitwise' flags must reproduce the Anchor goldens with the flag
 *  on; 'unbiased' / 'biased' flags change results (Anchor goldens are recorded with them off; Shipped goldens with every
 *  release flag on). */
export type PerfFlagClass = 'bitwise' | 'unbiased' | 'biased';

export interface PerfFlagDef {
  /** Owning work package (perf2-plan.md §2). */
  wp: string;
  cls: PerfFlagClass;
  /** Hash parts a 'bitwise' flag is allowed to change (documented exceptions): 'P9' = the reservoir diagnostic plane
   *  (P9-excluded hash mode), 'counters' = BVH / candidate counters. */
  except?: readonly ('P9' | 'counters')[];
  /** The WGSL behind the define exists (false: reserved name; forcing it compiles the unchanged text). */
  landed: boolean;
  /** On by default in the app (RestirKernel.interactive). Release flags of class unbiased / biased move the Shipped
   *  goldens. Biased flags are never released by default (user decision D6: a toggle, off in every validation path). */
  release: boolean;
  what: string;
}

const reserved = (wp: string, cls: PerfFlagClass, what: string, except?: PerfFlagDef['except']): PerfFlagDef =>
  ({ wp, cls, what, landed: false, release: false, ...(except ? { except } : {}) });

/** The registry. Order is documentation only (keys are sorted wherever they form a key / define set). */
export const PERF_FLAGS = {
  RS_VIS_MERGE: reserved('WP-1', 'bitwise', 'one trace_any_ex call site in shift_finish_vis (forced-inf, forced-local, ENV and reconnect merged via selects)'),
  RS_RIS_HOIST: reserved('WP-2a', 'bitwise', 'bsdf_prepare hoisted out of the RIS candidate loop; dead nee_draw at B=1 skipped'),
  RS_NEE_SITE: reserved('WP-2b', 'bitwise', 'rs_initial: one NEE visibility site (visible + visibleInf + k=B retest) and one post-hit retest site'),
  RS_LAST_ANYHIT: reserved('WP-2c', 'bitwise', 'last continuation ray as any-hit (Mode-A text, no TRI_EMISSIVE)', ['counters']),
  RS_RIS_PREPASS: reserved('WP-2d', 'unbiased', 'lean RIS pre-pass rs_ris_nee (interactive, trees=1); record in the pixel\'s RP_DIAG plane'),
  RS_ENV_WRAP: { ...reserved('WP-4a', 'bitwise', 'envTexel select-wrap instead of emulated i32 modulos (lights/env.wgsl)'), landed: true },
  RS_ENV_PRESAMPLE: reserved('WP-4b', 'unbiased', 'env-presampled light tiles inside the RIS pre-pass (adds correlation)'),
  CW_TRI_BUDGET: reserved('WP-3', 'bitwise', 'CWBVH K=2 triangle budget'),
  BVH_ALPHA_BIT: reserved('WP-3', 'bitwise', 'alpha bit in the MT triangle record; CUSTOM_ALPHA compiled out without MASK'),
  BVH2_PRIV_STACK: reserved('WP-3', 'bitwise', 'module-private traversal stack (BVH2 only; never on CWBVH)'),
  RS_DENSE_SLOTS: reserved('WP-5', 'bitwise', 'dense spatial slot-item queue, write-back trimming'),
  RS_BOOST_GATE: reserved('WP-5', 'bitwise', 'boost-slot gating'),
  RS_TSEL_FOLD: reserved('WP-6', 'bitwise', 'T4 split into non-replay / replay pipelines, res[w] read-only, phase B folded into T4'),
  RS_TSTATE_SOA: reserved('WP-6', 'bitwise', 'tState plane-major'),
  RS_AGG_COUNTERS: reserved('WP-6', 'bitwise', 'subgroup-aggregated arena counter atomics'),
  RS_NO_PLANTS: reserved('WP-9b', 'bitwise', 'U8-*, TP_* and 2022-criteria plant code compiled out'),
  RS_NO_DIAG: reserved('WP-9b', 'bitwise', 'P9 diagnostic writes / copies skipped (allocation kept)', ['P9']),
  MAT_VARIANTS: reserved('WP-8', 'bitwise', 'material compile variants (models present in the scene only), TEX_XFORM, branch-free LUT lerp'),
  RS_PRIMARY_EXT: reserved('WP-7e', 'unbiased', 'rs_primary reads the M1 V-buffer (bits(t) in vbuf.w) instead of tracing again (ulp edge ties)'),
  RS_HALF_RATE: reserved('D6', 'biased', 'half-rate path trees (history-age selection); interactive toggle only, never in validation'),
} as const satisfies Record<string, PerfFlagDef>;

export type PerfFlagName = keyof typeof PERF_FLAGS;
/** Normalised flag set: name → define value (1 for a plain switch; integers for parametrised flags, e.g. a budget). */
export type PerfFlags = Partial<Record<PerfFlagName, number>>;
/** Accepted input: a normalised set, a name list, or a spec string 'A,B=2' (CLI --kernel-flags, VITE_PERF_FLAGS). */
export type PerfFlagsInput = PerfFlags | Partial<Record<PerfFlagName, number | boolean>> | readonly string[] | string | undefined;

export const PERF_FLAG_NAMES = Object.keys(PERF_FLAGS) as PerfFlagName[];
export const isPerfFlagName = (n: string): n is PerfFlagName => Object.prototype.hasOwnProperty.call(PERF_FLAGS, n);

/** Flags on in the app by default (RestirKernel.interactive). Empty until a package's flag is accepted. */
export const RELEASE_PERF_FLAGS: PerfFlags = Object.fromEntries(
  PERF_FLAG_NAMES.filter((n) => (PERF_FLAGS[n] as PerfFlagDef).release).map((n) => [n, 1])) as PerfFlags;

/** Parse / validate / normalise: unknown names throw, false / 0 entries are dropped, keys sorted. */
export function normalizePerfFlags(input: PerfFlagsInput): PerfFlags {
  if (input === undefined) return {};
  const entries: [string, number | boolean][] = [];
  if (typeof input === 'string') {
    for (const tok of input.split(/[\s,]+/).filter(Boolean)) {
      const [n, v] = tok.split('=');
      entries.push([n, v === undefined ? 1 : Number(v)]);
    }
  } else if (Array.isArray(input)) {
    for (const n of input as readonly string[]) entries.push([n, 1]);
  } else {
    entries.push(...Object.entries(input as Record<string, number | boolean>));
  }
  const out: Record<string, number> = {};
  for (const [n, v0] of entries) {
    if (!isPerfFlagName(n)) throw new Error(`unknown perf flag '${n}' (registry: src/core/render/restir/perf-flags.ts)`);
    const v = v0 === true ? 1 : v0 === false ? 0 : v0;
    if (!Number.isInteger(v) || v < 0) throw new Error(`perf flag ${n}=${String(v0)}: a non-negative integer required`);
    if (v === 0) delete out[n]; else out[n] = v;
  }
  return Object.fromEntries(Object.keys(out).sort().map((k) => [k, out[k]])) as PerfFlags;
}

/** Composer defines of a flag set (absent keys = off; empty object for no flags). */
export function perfFlagDefines(input: PerfFlagsInput): Record<string, number> {
  return { ...normalizePerfFlags(input) } as Record<string, number>;
}

/** Stable key of a flag set ('' for none; 'A,B=2' otherwise): cache keys, config hashes, report labels. */
export function perfFlagsKey(input: PerfFlagsInput): string {
  const f = normalizePerfFlags(input);
  return Object.entries(f).map(([k, v]) => (v === 1 ? k : `${k}=${v}`)).join(',');
}

/** Flags of the set whose class changes results (unbiased or biased). */
export function resultsChangingFlags(input: PerfFlagsInput): PerfFlagName[] {
  return (Object.keys(normalizePerfFlags(input)) as PerfFlagName[]).filter((n) => PERF_FLAGS[n].cls !== 'bitwise');
}
/** Flags of the set of class 'biased' (never allowed in a validation run without an explicit opt-in). */
export function biasedFlags(input: PerfFlagsInput): PerfFlagName[] {
  return (Object.keys(normalizePerfFlags(input)) as PerfFlagName[]).filter((n) => PERF_FLAGS[n].cls === 'biased');
}
/** Flags of the set that are reserved names only (no WGSL yet: forcing them has no effect). */
export function unlandedFlags(input: PerfFlagsInput): PerfFlagName[] {
  return (Object.keys(normalizePerfFlags(input)) as PerfFlagName[]).filter((n) => !(PERF_FLAGS[n] as PerfFlagDef).landed);
}
/** The bitwise subset of a set (the flags the Anchor goldens must hold with). */
export function bitwiseSubset(input: PerfFlagsInput): PerfFlags {
  const f = normalizePerfFlags(input);
  return Object.fromEntries(Object.entries(f).filter(([n]) => PERF_FLAGS[n as PerfFlagName].cls === 'bitwise')) as PerfFlags;
}

/** Validation runners (restir-chain-run / restir-batch-run): normalise the forced flags, refuse biased ones unless
 *  allowed (user decision D6: off in every validation path), warn about reserved names without WGSL. */
export function checkValidationPerfFlags(input: PerfFlagsInput, allowBiased: boolean): { flags: PerfFlags; key: string; biased: PerfFlagName[] } {
  const flags = normalizePerfFlags(input);
  const biased = biasedFlags(flags);
  if (biased.length && !allowBiased) throw new Error(`perf flags ${biased.join(', ')} are biased: refused in a validation run (allowBiasedFlags / --allow-biased-flags to override)`);
  const un = unlandedFlags(flags);
  if (un.length) console.warn(`[perf-flags] ${un.join(', ')}: reserved names without WGSL yet (no effect)`);
  return { flags, key: perfFlagsKey(flags), biased };
}
