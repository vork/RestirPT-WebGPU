// U-WGSL-BITS (restir-m6-api.md D2, §7): with every M6 feature off, the composed WGSL of every ReSTIR pass and of the
// reference PT is the M5 build's text (comments and blank lines removed, whitespace collapsed), so the compiled
// pipelines — and therefore every M4/M5 Stage-B result and every cached PT reference — are unchanged. The goldens were
// recorded on main f5c23fd (M6_WGSL_RECORD=1 prints them). An M6 feature reaches the code only through its own composer
// define (RS_RIS_NEE, RS_MODE_B, RS_DUAL_MV, RS_DUPMAP) or its own pass, never through a default pipeline.
import { createHash } from 'node:crypto';
import { describe, expect, it } from 'vitest';
import { composeWgsl, type Defines } from '../../src/core/gpu/wgsl-composer.ts';
import { shaderSources } from '../../src/core/shaders/index.ts';
import { RS_PASSES, restirDefines, type RsPassName } from '../../src/core/render/restir/resources.ts';
import { envDefines } from '../../src/core/render/env-gpu.ts';
import { lutDefines } from '../../src/core/render/luts/lut-layout.ts';
import { LUT_RECORDS_BASE } from '../../src/core/render/lights-gpu.ts';

const RECORD = process.env.M6_WGSL_RECORD === '1';
const FEATURES = new Set(['subgroups', 'shader-f16', 'timestamp-query', 'float32-filterable', 'texture-formats-tier2', 'texture-formats-tier1']);
const LANG = new Set(['immediate_address_space', 'linear_indexing', 'readonly_and_readwrite_storage_textures', 'pointer_composite_access']);
const SCENE: Defines = {
  SCENE_GROUP: 1, BVH_DECLARE_BINDINGS: true, BVH_GROUP: 1, BVH_BINDING_NODES: 0, BVH_BINDING_TRIS: 1, WATERTIGHT: true,
  CUSTOM_ALPHA: true, VERTEX_FORMAT: 1, TEX_GROUP: 1, TEX_BINDING_BASE: 5, TEX_ARRAYS: 2, TEX_SAMPLERS: 1,
};

/** Comment-free, whitespace-normalised composed text. */
export function normWgsl(code: string): string {
  return code.split('\n').map((l) => l.replace(/\/\/.*$/, '').replace(/\s+/g, ' ').trim()).filter((l) => l.length > 0).join('\n');
}
const h = (s: string) => createHash('sha256').update(s).digest('hex').slice(0, 16);

export function m5PassHashes(extra: Defines = {}): Record<string, string> {
  const out: Record<string, string> = {};
  for (const name of Object.keys(RS_PASSES) as RsPassName[]) {
    if (!(name in M5_PASSES_SET)) continue;
    for (const debug of [false, true]) {
      const d = restirDefines(name, { sceneDefines: SCENE, debug, extra });
      if (name === 'rs_finalize_frame') d.COLOR_FORMAT = 'rgba16float';
      const code = composeWgsl(RS_PASSES[name].file, { sources: shaderSources, defines: d, features: FEATURES, wgslLanguageFeatures: LANG }).code;
      out[`${name}${debug ? ':debug' : ''}`] = h(normWgsl(code));
    }
  }
  for (const interactive of [false, true]) {
    const code = composeWgsl('passes/pt.wgsl', {
      sources: shaderSources, features: FEATURES, wgslLanguageFeatures: LANG,
      defines: {
        ...SCENE, ...envDefines(0, 1), LIGHTS_GROUP: 0, LIGHTS_BINDING: 5, ...lutDefines({ base: LUT_RECORDS_BASE, recordsKind: 'u32' }),
        PT_INTERACTIVE: interactive, PT_PROBE: false, GLASS_PLANT: 0, ...(interactive ? { COLOR_FORMAT: 'rgba16float' } : {}), ENV_PLANT: 0, ENV_MIS_POWER: false,
      },
    }).code;
    out[`pt${interactive ? ':frame' : ':batch'}`] = h(normWgsl(code));
  }
  return out;
}

/** The passes of the M5 build (M6 adds its own passes, which have no golden). */
const M5_PASSES_SET: Record<string, true> = Object.fromEntries([
  'rs_primary', 'rs_initial', 'rs_initial_dump', 'rs_pair_accept', 'rs_args', 'rs_spatial_replay', 'rs_spatial_shift', 'rs_spatial_resample',
  'rs_finalize', 'rs_finalize_frame', 'rs_ensemble_stats', 'rs_refresh_fwd', 'rs_refresh_inv', 'rs_t_classify', 'rs_t_forward', 'rs_t_select', 'rs_t_inverse',
].map((n) => [n, true]));

// Recorded on main f5c23fd (M5 build) with M6_WGSL_RECORD=1. Re-recorded for restir-m6-api.md Changelog M6-10 (incoming
// direction after a delta event in path/pathtree.wgsl + path/replay.wgsl): rs_initial(_dump), rs_spatial_replay,
// rs_spatial_shift, rs_t_classify, rs_t_forward, rs_t_inverse. Every other entry (incl. the PT) is still the M5 text.
const GOLDEN: Record<string, string> = {
  'rs_primary': '57e56f165a915491',
  'rs_primary:debug': '57e56f165a915491',
  'rs_initial': '3684883566148440',
  'rs_initial:debug': 'bfdaed93482b8a09',
  'rs_initial_dump': '35016b17948e56f6',
  'rs_initial_dump:debug': '35016b17948e56f6',
  'rs_pair_accept': '5d33240916ef67b4',
  'rs_pair_accept:debug': '5d33240916ef67b4',
  'rs_args': '151307927572ba68',
  'rs_args:debug': '151307927572ba68',
  'rs_spatial_replay': '519651afe929fa58',
  'rs_spatial_replay:debug': '3b83bdcb562f73b1',
  'rs_spatial_shift': '607a41daea9c3dc4',
  'rs_spatial_shift:debug': '5d77b9796584fd8b',
  'rs_spatial_resample': '93731d7f202731d5',
  'rs_spatial_resample:debug': 'b0a0d616dc6f430f',
  'rs_finalize': '9c1257d4fb3a33ba',
  'rs_finalize:debug': '9c1257d4fb3a33ba',
  'rs_finalize_frame': 'be1f3c84d953b3cc',
  'rs_finalize_frame:debug': 'be1f3c84d953b3cc',
  'rs_ensemble_stats': 'e9259f58ecc142b4',
  'rs_ensemble_stats:debug': 'e9259f58ecc142b4',
  'rs_refresh_fwd': '352d25be032e15db',
  'rs_refresh_fwd:debug': 'f87bd780a588bb9c',
  'rs_refresh_inv': '352d25be032e15db',
  'rs_refresh_inv:debug': 'f87bd780a588bb9c',
  'rs_t_classify': '6404220d51a91db4',
  'rs_t_classify:debug': '98433380f137e249',
  'rs_t_forward': 'b8507663049a75f7',
  'rs_t_forward:debug': 'adc026a91dd88a1a',
  'rs_t_select': 'e17316b5c5bd7213',
  'rs_t_select:debug': '164310b2dd991fe0',
  'rs_t_inverse': 'ca08376725d533a7',
  'rs_t_inverse:debug': '556521c453edfe86',
  'pt:batch': '7c0f2ee93bbb4746',
  'pt:frame': '2e9ffceff8e48809',
};

describe('U-WGSL-BITS: M6 features off ⇒ every composed ReSTIR pass and the PT are the M5 text', () => {
  it('normalised composed WGSL hashes equal the M5 goldens', () => {
    const got = m5PassHashes();
    if (RECORD) { console.log(`[U-WGSL-BITS] ${JSON.stringify(got)}`); return; }
    expect(got).toEqual(GOLDEN);
  });
});
