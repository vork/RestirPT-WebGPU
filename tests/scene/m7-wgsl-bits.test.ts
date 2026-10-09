// U-M7-BITS (docs/decisions/m7-api.md §2, M7 Gate 0): a scene WITHOUT normal maps (flat or smooth shaded) composes every
// pipeline to exactly the text of the M6 build (main b5b0f5d), comments and blank lines removed, whitespace collapsed.
// Identical WGSL ⇒ identical pipelines ⇒ every M4 / M5 / M6 Stage-B result, U-M4-BITS / U-M5-BITS golden and cached PT
// reference stays bitwise valid. Normal maps reach the code only through the composer define NORMAL_MAP (set by
// SceneGpu.defines() iff the scene has a normal-mapped material with tangents). The goldens were recorded on the
// unmodified b5b0f5d tree with M7_WGSL_RECORD=1. Entries: the PT (batch / frame / probe / every glass plant), the primary
// pass, the emission kernel, env-debug, every ReSTIR pass (debug on / off) under the M6 variant sets, and the ReSTIR
// debug views; scene defines for both vertex formats and with / without material textures.
import { createHash } from 'node:crypto';
import { readFileSync, writeFileSync } from 'node:fs';
import { describe, expect, it } from 'vitest';
import { composeWgsl, type Defines } from '../../src/core/gpu/wgsl-composer.ts';
import { shaderSources } from '../../src/core/shaders/index.ts';
import { RS_PASSES, restirCommonDefines, restirDefines, type RsPassName } from '../../src/core/render/restir/resources.ts';
import { envDefines } from '../../src/core/render/env-gpu.ts';
import { lutDefines } from '../../src/core/render/luts/lut-layout.ts';
import { LUT_RECORDS_BASE } from '../../src/core/render/lights-gpu.ts';
import { PERF_FLAG_NAMES } from '../../src/core/render/restir/perf-flags.ts';

const RECORD = process.env.M7_WGSL_RECORD === '1';
const FEATURES = new Set(['subgroups', 'shader-f16', 'timestamp-query', 'float32-filterable', 'texture-formats-tier2', 'texture-formats-tier1']);
const LANG = new Set(['immediate_address_space', 'linear_indexing', 'readonly_and_readwrite_storage_textures', 'pointer_composite_access']);

/** Scene defines as SceneGpu.defines() writes them (no NORMAL_MAP key: the pre-M7 shape). */
const sceneDefines = (vf: 0 | 1, tex: boolean, group = 1): Defines => ({
  SCENE_GROUP: group, BVH_DECLARE_BINDINGS: true, BVH_GROUP: group, BVH_BINDING_NODES: 0, BVH_BINDING_TRIS: 1, WATERTIGHT: true,
  CUSTOM_ALPHA: true, VERTEX_FORMAT: vf, TEX_GROUP: group, TEX_BINDING_BASE: 8, TEX_ARRAYS: tex ? 3 : 0, TEX_SAMPLERS: tex ? 2 : 0,
});
const SCENES: Record<string, Defines> = { q: sceneDefines(1, false), qtex: sceneDefines(1, true), f32tex: sceneDefines(0, true) };

const norm = (code: string) => code.split('\n').map((l) => l.replace(/\/\/.*$/, '').replace(/\s+/g, ' ').trim()).filter((l) => l.length > 0).join('\n');
const h = (s: string) => createHash('sha256').update(s).digest('hex').slice(0, 16);
/** perf2 (perf2-api.md, V-BIT item 1): every define set of the validation texts is collected; none may carry a perf flag. */
const seenDefineKeys = new Set<string>();
const compose = (file: string, defines: Defines) => {
  for (const k of Object.keys(defines)) seenDefineKeys.add(k);
  return h(norm(composeWgsl(file, { sources: shaderSources, defines, features: FEATURES, wgslLanguageFeatures: LANG }).code));
};

const M6_VARIANTS: Record<string, Defines> = {
  off: {}, ris: { RS_RIS_NEE: 1 }, modeB: { RS_MODE_B: 1 }, temporal: { RS_DUAL_MV: 1, RS_DUPMAP: 1 }, t2: { RS_PLANT_T2: 1 },
  all: { RS_RIS_NEE: 1, RS_MODE_B: 1, RS_DUAL_MV: 1, RS_DUPMAP: 1 },
};

export function m7Hashes(extraScene: Defines = {}): Record<string, string> {
  const out: Record<string, string> = {};
  for (const [sk, sd0] of Object.entries(SCENES)) {
    const sd = { ...sd0, ...extraScene };
    const ptBase = { ...sd, ...envDefines(0, 1), LIGHTS_GROUP: 0, LIGHTS_BINDING: 5, ...lutDefines({ base: LUT_RECORDS_BASE, recordsKind: 'u32' }), ENV_PLANT: 0, ENV_MIS_POWER: false };
    for (const [tag, d] of Object.entries<Defines>({
      batch: { PT_INTERACTIVE: false, PT_PROBE: false, GLASS_PLANT: 0 },
      frame: { PT_INTERACTIVE: true, PT_PROBE: false, GLASS_PLANT: 0, COLOR_FORMAT: 'rgba16float' },
      probe: { PT_INTERACTIVE: false, PT_PROBE: true, GLASS_PLANT: 0 },
      plant5: { PT_INTERACTIVE: false, PT_PROBE: false, GLASS_PLANT: 5 },
      envplant: { PT_INTERACTIVE: false, PT_PROBE: false, GLASS_PLANT: 0, ENV_PLANT: 2, ENV_MIS_POWER: true },
    })) out[`${sk}:pt:${tag}`] = compose('passes/pt.wgsl', { ...ptBase, ...d });
    out[`${sk}:primary`] = compose('passes/primary.wgsl', { COLOR_FORMAT: 'rgba16float', BVH_STATS: false, ...sd, ...envDefines(0, 1) });
    out[`${sk}:primary:stats`] = compose('passes/primary.wgsl', { COLOR_FORMAT: 'rgba16float', BVH_STATS: true, ...sd, ...envDefines(0, 1) });
    out[`${sk}:emission`] = compose('passes/emission.wgsl', { ...sd, ...envDefines(0, 1) });
    out[`${sk}:env-debug`] = compose('passes/env-debug.wgsl', { ...sd, ...envDefines(0, 1), LIGHTS_GROUP: 0, LIGHTS_BINDING: 5, ...lutDefines({ base: LUT_RECORDS_BASE, recordsKind: 'u32' }) });
    for (const name of Object.keys(RS_PASSES) as RsPassName[]) {
      if (!RS_PASSES[name].scene) continue;
      for (const [vk, vd] of Object.entries(M6_VARIANTS)) for (const debug of [false, true]) {
        const d = restirDefines(name, { sceneDefines: sd, debug, extra: { RS_RIS_NEE: 0, RS_MODE_B: 0, RS_DUAL_MV: 0, RS_DUPMAP: 0, RS_PLANT_T2: 0, ...vd } });
        out[`${sk}:${name}:${vk}${debug ? ':debug' : ''}`] = compose(RS_PASSES[name].file, d);
      }
    }
    out[`${sk}:rs_debug_views`] = compose('passes/restir/debug.wgsl', {
      ...restirCommonDefines(sd, true), RS_RIS_NEE: 0, RS_MODE_B: 0, RS_DUAL_MV: 0, RS_DUPMAP: 0, RS_PLANT_T2: 0,
      RS_ARENA_BINDING: '1u', RS_ARENA_RW: false, RS_VBUF_BINDING: '2u', RS_GEO_BINDING: '3u', RS_PAIRTEX_BINDING: '4u', RS_VBUF_PREV_BINDING: '5u',
    });
  }
  return out;
}

/** sha256 over the sorted (key, hash) list: one golden for the whole table (the per-entry table is printed on failure). */
const digest = (t: Record<string, string>) => h(Object.keys(t).sort().map((k) => `${k}=${t[k]}`).join('\n'));

// Recorded on b5b0f5d (M6 merge) with M7_WGSL_RECORD=1 (per-entry table: m7-wgsl-golden.json next to this file).
const GOLDEN_DIGEST = '6c1051b1d1e545c9';
const GOLDEN_COUNT = 390;
const GOLDEN_FILE = new URL('./m7-wgsl-golden.json', import.meta.url);

describe('U-M7-BITS: scenes without normal maps compose to the M6 WGSL', () => {
  it('every PT / primary / emission / env-debug / ReSTIR pipeline text equals the b5b0f5d build', () => {
    const t = m7Hashes();
    if (RECORD) {
      console.log(`GOLDEN_DIGEST = '${digest(t)}'; GOLDEN_COUNT = ${Object.keys(t).length}`);
      writeFileSync(GOLDEN_FILE, `${JSON.stringify(t, null, 1)}\n`);
    }
    const g = JSON.parse(readFileSync(GOLDEN_FILE, 'utf8')) as Record<string, string>;
    expect(digest(g)).toBe(GOLDEN_DIGEST);
    const changed = Object.keys(g).filter((k) => g[k] !== t[k]);
    expect(changed, `entries whose text changed: ${changed.slice(0, 20).join(', ')}`).toEqual([]);
    expect(Object.keys(t).length).toBe(GOLDEN_COUNT);
    expect(digest(t)).toBe(GOLDEN_DIGEST);
  });
  it('perf2: no perf-flag key in any validation define set (flags reach only interactive / forced kernels)', () => {
    seenDefineKeys.clear();
    m7Hashes();
    expect(seenDefineKeys.size).toBeGreaterThan(20);
    expect([...seenDefineKeys].filter((k) => (PERF_FLAG_NAMES as string[]).includes(k))).toEqual([]);
  });
  it('NORMAL_MAP: false is the same text as no NORMAL_MAP key', () => {
    expect(digest(m7Hashes({ NORMAL_MAP: false }))).toBe(digest(m7Hashes()));
  });
});
