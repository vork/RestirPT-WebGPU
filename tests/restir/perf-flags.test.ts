// perf2 WP-0 (docs/decisions/perf2-api.md): the perf-flag registry, its parsing / keys, the pinned interactive knobs
// and the app-default hook of the user decisions. CPU only; the GPU side (flags reach every pipeline, Anchor / Shipped
// goldens) is m8-bits.gpu.test.ts, the validation-text side U-M7-BITS (m7-wgsl-bits.test.ts).
import { describe, expect, it } from 'vitest';
import {
  PERF_FLAGS, PERF_FLAG_NAMES, RELEASE_PERF_FLAGS, bitwiseSubset, biasedFlags, checkValidationPerfFlags, normalizePerfFlags,
  perfFlagDefines, perfFlagsKey, resultsChangingFlags, unlandedFlags, type PerfFlagDef,
} from '../../src/core/render/restir/perf-flags.ts';
import { INTERACTIVE_PINNED, RESTIR_PRESETS, restirSettings } from '../../src/core/render/restir/presets.ts';
import { INTERACTIVE_APP_DEFAULTS, restirAppSettings } from '../../src/core/render/renderer.ts';
import { APP_DEFAULTS_RESTIR, DECISION_CONFIGS, abbaJobs } from '../../validation/harness/run-perf.ts';
import { APP_DEFAULTS_M6 } from '../../validation/harness/gate-m6.ts';
import { shaderSources } from '../../src/core/shaders/index.ts';
import { denoiseSources } from '../../src/core/render/denoise/denoiser.ts';

/** perf2-plan.md WP-0 step 1 (+ RS_HALF_RATE, user decision D6). */
const RESERVED = ['RS_VIS_MERGE', 'RS_RIS_HOIST', 'RS_NEE_SITE', 'RS_LAST_ANYHIT', 'RS_RIS_PREPASS', 'RS_ENV_WRAP', 'RS_ENV_PRESAMPLE',
  'CW_TRI_BUDGET', 'BVH_ALPHA_BIT', 'BVH2_PRIV_STACK', 'RS_DENSE_SLOTS', 'RS_BOOST_GATE', 'RS_TSEL_FOLD', 'RS_TSTATE_SOA', 'RS_AGG_COUNTERS',
  'RS_NO_PLANTS', 'RS_NO_DIAG', 'MAT_VARIANTS', 'RS_PRIMARY_EXT', 'RS_HALF_RATE',
  'RS_REFRESH_VIS',  // WP-1's optional refresh merge (perf2-plan.md WP-1 "separate flag")
  'BVH_CONST_LOOPS', 'CW_EXP_OR',   // WP-3c (constant-bound traversal loops) and WP-3e (exponent-OR decode), added by WP-3
  // added by WP-5 (separately measurable parts of #6 / #16)
  'RS_MIS_TRIM', 'RS_PAIR_TABLE',
  // WP-7 a–d, f (app-only passes)
  'DN_GRAD_SKIP', 'RS_DUPMAP_S64', 'PRIM_SKIP_BEAUTY', 'RS_SKIP_DISPLAY', 'GBUF_48', 'DN_ZGRAD_TEX', 'DN_COLOUR_EARLY'];

describe('perf-flag registry', () => {
  it('reserves every WP-0 name, RS_HALF_RATE and the WP-7 names', () => {
    expect([...PERF_FLAG_NAMES].sort()).toEqual([...RESERVED].sort());
  });
  it('classes: results-changing flags are the plan\'s unbiased items and the biased D6 toggle', () => {
    expect(resultsChangingFlags(PERF_FLAG_NAMES).sort()).toEqual(['RS_ENV_PRESAMPLE', 'RS_HALF_RATE', 'RS_PRIMARY_EXT', 'RS_RIS_PREPASS']);
    expect(biasedFlags(PERF_FLAG_NAMES)).toEqual(['RS_HALF_RATE']);
  });
  it('no biased flag is a release flag (D6: a toggle, off by default and in every validation path)', () => {
    for (const n of PERF_FLAG_NAMES) if (PERF_FLAGS[n].cls === 'biased') expect((PERF_FLAGS[n] as PerfFlagDef).release).toBe(false);
    expect(biasedFlags(RELEASE_PERF_FLAGS)).toEqual([]);
  });
  it('a release flag has landed', () => {
    for (const n of Object.keys(RELEASE_PERF_FLAGS)) expect(unlandedFlags([n])).toEqual([]);
  });
  it('every flag a shader #if references is marked landed', () => {
    const used = new Set<string>();
    for (const src of [...Object.values(shaderSources), ...Object.values(denoiseSources)]) {
      for (const line of src.split('\n')) {
        const m = /^\s*#(?:if|elif)\b(.*)$/.exec(line);
        if (!m) continue;
        for (const tok of m[1].match(/[A-Za-z_]\w*/g) ?? []) if ((PERF_FLAG_NAMES as string[]).includes(tok)) used.add(tok);
      }
    }
    expect([...used].filter((n) => unlandedFlags([n]).length > 0), 'set landed: true in perf-flags.ts when the WGSL lands').toEqual([]);
  });
});

describe('perf-flag parsing and keys', () => {
  it('normalises strings, lists and objects; sorted; off entries dropped', () => {
    expect(normalizePerfFlags(undefined)).toEqual({});
    expect(normalizePerfFlags('')).toEqual({});
    expect(normalizePerfFlags('RS_VIS_MERGE')).toEqual({ RS_VIS_MERGE: 1 });
    expect(normalizePerfFlags('RS_VIS_MERGE, CW_TRI_BUDGET=2')).toEqual({ CW_TRI_BUDGET: 2, RS_VIS_MERGE: 1 });
    expect(Object.keys(normalizePerfFlags('RS_VIS_MERGE,CW_TRI_BUDGET'))).toEqual(['CW_TRI_BUDGET', 'RS_VIS_MERGE']);
    expect(normalizePerfFlags(['RS_RIS_HOIST'])).toEqual({ RS_RIS_HOIST: 1 });
    expect(normalizePerfFlags({ RS_RIS_HOIST: true, RS_VIS_MERGE: false, RS_NO_DIAG: 0 })).toEqual({ RS_RIS_HOIST: 1 });
  });
  it('rejects unknown names and bad values', () => {
    expect(() => normalizePerfFlags('RS_VIS_MERG')).toThrow(/unknown perf flag/);
    expect(() => normalizePerfFlags('RS_VIS_MERGE=x')).toThrow(/non-negative integer/);
    expect(() => normalizePerfFlags({ RS_VIS_MERGE: -1 })).toThrow();
  });
  it('defines: absent keys for no flags (the validation text), 1 / value otherwise', () => {
    expect(perfFlagDefines(undefined)).toEqual({});
    expect(perfFlagDefines({})).toEqual({});
    expect(perfFlagDefines('CW_TRI_BUDGET=2,RS_VIS_MERGE')).toEqual({ CW_TRI_BUDGET: 2, RS_VIS_MERGE: 1 });
  });
  it('key: empty for no flags, stable and order-independent', () => {
    expect(perfFlagsKey(undefined)).toBe('');
    expect(perfFlagsKey('RS_VIS_MERGE,CW_TRI_BUDGET=2')).toBe('CW_TRI_BUDGET=2,RS_VIS_MERGE');
    expect(perfFlagsKey(['CW_TRI_BUDGET', 'RS_VIS_MERGE'])).toBe(perfFlagsKey('RS_VIS_MERGE,CW_TRI_BUDGET'));
    expect(normalizePerfFlags(perfFlagsKey('RS_VIS_MERGE,CW_TRI_BUDGET=2'))).toEqual(normalizePerfFlags('RS_VIS_MERGE,CW_TRI_BUDGET=2'));
  });
  it('bitwise subset (the Anchor run of forced flags)', () => {
    expect(bitwiseSubset('RS_VIS_MERGE,RS_RIS_PREPASS,RS_HALF_RATE')).toEqual({ RS_VIS_MERGE: 1 });
  });
  it('validation runners refuse biased flags unless allowed', () => {
    expect(() => checkValidationPerfFlags('RS_HALF_RATE', false)).toThrow(/biased/);
    expect(checkValidationPerfFlags('RS_HALF_RATE', true).biased).toEqual(['RS_HALF_RATE']);
    expect(checkValidationPerfFlags(undefined, false)).toEqual({ flags: {}, key: '', biased: [] });
  });
});

describe('pinned interactive knobs and app defaults (WP-0 step 2, user decisions)', () => {
  it('INTERACTIVE_PINNED equals the interactive preset (pinning changes no bits)', () => {
    const s = restirSettings('interactive');
    for (const [k, v] of Object.entries(INTERACTIVE_PINNED)) expect(s[k as keyof typeof s], k).toBe(v);
  });
  it('presets are unchanged by app defaults; interactive app settings = preset ⊕ INTERACTIVE_APP_DEFAULTS', () => {
    expect(RESTIR_PRESETS.interactive.dupmap).toBe(true);
    const app = restirAppSettings('interactive', 3);
    for (const [k, v] of Object.entries(INTERACTIVE_APP_DEFAULTS)) expect(app[k as keyof typeof app], k).toEqual(v);
    // perf2 §5: D1 (RR after bounce 2) and D3 (duplication map off) are the app defaults; the preset keeps 3 / on
    expect(INTERACTIVE_APP_DEFAULTS).toEqual({ rrMinBounces: 2, dupmap: false });
    expect(app.dupmap).toBe(false);
    expect(app.rrMinBounces).toBe(2);
    expect(app.rr).toBe(true);
    expect(RESTIR_PRESETS.interactive.rrMinBounces).toBe(3);
    // only the interactive mode gets them
    expect(restirAppSettings('criteria2022', 3).dupmap).toBe(true);
    expect(restirAppSettings('criteria2022', 3).rrMinBounces).toBe(3);
  });
  it('the pins of the bits rigs / perf baselines are the preset, not the app defaults (goldens hold)', () => {
    expect(INTERACTIVE_PINNED).toEqual({ slots: 3, risM: 32, rrMinBounces: 3, dupmap: true });
  });
  it('the harness copies of the app defaults (run-perf --abba-restir app-defaults, gate-m6 --part decisions) equal them', () => {
    expect(APP_DEFAULTS_RESTIR).toEqual(INTERACTIVE_APP_DEFAULTS);
    expect(APP_DEFAULTS_M6).toEqual(INTERACTIVE_APP_DEFAULTS);
    expect(DECISION_CONFIGS.find((d) => d.id === 'D1+D3 appDefaults')?.restir).toEqual(INTERACTIVE_APP_DEFAULTS);
    const j = abbaJobs('', ['cornell'], ['540p'], 1, {}, { role: 'appDefaults', settings: APP_DEFAULTS_RESTIR });
    expect(j.map((x) => x.label!.split(' ')[2])).toEqual(['base', 'appDefaults', 'appDefaults', 'base', 'base', 'appDefaults', 'appDefaults', 'base']);
    expect(j[0].restir).toEqual(INTERACTIVE_PINNED);
    expect(j[1].restir).toEqual({ ...INTERACTIVE_PINNED, ...INTERACTIVE_APP_DEFAULTS });
    // both roles run the same (release) flag set: a variant never drops the released flags
    expect(j[1].perfFlags).toEqual(j[0].perfFlags);
    const f = abbaJobs('RS_TSEL_FOLD', ['cornell'], ['540p'], 1);
    expect(f[1].perfFlags).toEqual({ ...normalizePerfFlags(RELEASE_PERF_FLAGS), RS_TSEL_FOLD: 1 });
    expect(f[0].perfFlags).toBeUndefined();
  });
  it('RR start can be overridden per session (panel)', () => {
    expect(restirAppSettings('interactive', 3, true, { rrMinBounces: 3 }).rrMinBounces).toBe(3);
  });
  it('the duplication map can be switched off per session (feature override over the app defaults)', () => {
    expect(restirAppSettings('interactive', 3, true, { dupmap: false }).dupmap).toBe(false);
    expect(restirAppSettings('interactive', 3, true, { dupmap: true }).dupmap).toBe(true);
  });
});
