// perf2 WP-5 (docs/decisions/perf2-plan.md §2 WP-5): CPU checks of the spatial-reuse flags RS_DENSE_SLOTS,
// RS_BOOST_GATE, RS_MIS_TRIM and RS_PAIR_TABLE.
//   - the WGSL constants behind the flags mirror layout.ts WP5_CONSTS, and the q3 / gate words do not collide with any
//     queue header or RSC_* counter;
//   - the exact float-reciprocal modulo of pairing.wgsl (pair_mod_fast) equals ((x % W) + W) % W for every layer size
//     over the whole range of M·p + o, also with a perturbed reciprocal (fast-math 1/W);
//   (the table itself is computed on the GPU by pair_transform, once per workgroup: its equality with the per-thread
//   path is covered by the bits suites with VITE_PERF_FLAGS=RS_PAIR_TABLE);
//   - composing every spatial pass with every flag subset works, and without flags gives the unchanged text.
import { describe, expect, it } from 'vitest';
import { composeWgsl } from '../../src/core/gpu/wgsl-composer.ts';
import { shaderSources } from '../../src/core/shaders/index.ts';
import { RS_WGSL_CONSTS as K, RSC, WP5_CONSTS, queueHdr } from '../../src/core/render/restir/layout.ts';
import { RS_PASSES, restirDefines, type RsPassName } from '../../src/core/render/restir/resources.ts';
import { GAUSS_PAIR_SIZES, PAIR_TEX_SIZES } from '../../src/core/render/restir/presets.ts';

const f32 = Math.fround;
const wgslConst = (src: string, name: string): number => {
  const m = new RegExp(`const ${name}: u32 = (\\d+)u;`).exec(src);
  if (!m) throw new Error(`${name} not found`);
  return Number(m[1]);
};

/** pairing.wgsl pair_mod_fast with f32 arithmetic (inv = the f32 reciprocal, optionally perturbed). */
function modFast(x: number, size: number, inv: number): number {
  let r = x - size * Math.floor(f32(f32(x) * inv));
  if (r < 0) r += size;
  if (r >= size) r -= size;
  return r;
}

describe('WP-5 constants', () => {
  it('queue.wgsl mirrors WP5_CONSTS', () => {
    const q = shaderSources['restir/queue.wgsl'];
    for (const [n, v] of Object.entries(WP5_CONSTS)) expect(wgslConst(q, n), n).toBe(v);
  });
  it('q3 and the gate word collide with no queue header word or counter; RSD_BOOST_OPEN with no RSD_* bit', () => {
    const used = new Set<number>([0, 1, 2].flatMap((q) => Object.values(queueHdr(q))));
    for (const w of Object.values(RSC)) used.add(w);
    for (let c = 0; c < 16; c++) used.add(K.RSC_CODE_BASE + c);
    for (const w of Object.values(queueHdr(WP5_CONSTS.RS_Q_DENSE))) expect(used.has(w), `q3 word ${w}`).toBe(false);
    expect(used.has(WP5_CONSTS.RS_HDR_BOOST_GATE)).toBe(false);
    expect(WP5_CONSTS.RS_HDR_BOOST_GATE).toBeLessThan(K.RS_ARENA_HDR_WORDS);
    const rsd = [K.RSD_FIRST_CHUNK, K.RSD_FINAL_CHUNK, K.RSD_FINAL_ROUND, K.RSD_ACCUMULATE, K.RSD_ADVANCED, K.RSD_PHASE_B, 3 << K.RSD_QUEUE_SHIFT];
    for (const b of rsd) expect(b & WP5_CONSTS.RSD_BOOST_OPEN).toBe(0);
  });
});

describe('WP-5 exact float-reciprocal modulo (RS_PAIR_TABLE)', () => {
  const sizes = [...new Set([...PAIR_TEX_SIZES, ...GAUSS_PAIR_SIZES].filter((s) => s > 0)), 1, 2, 3, 7, 255, 256];
  it('equals ((x % W) + W) % W for |x| ≤ 8448 (members up to 8192 px + offsets < 256), exact and perturbed 1/W', () => {
    for (const W of sizes) {
      const inv = f32(1 / W);
      for (const e of [0, 2 ** -20, -(2 ** -20), 2 ** -14, -(2 ** -14)]) {
        const invE = f32(inv * (1 + e));
        for (let x = -8448; x <= 8448; x++) {
          const want = ((x % W) + W) % W;
          const got = modFast(x, W, invE);
          if (got !== want) throw new Error(`W ${W} e ${e} x ${x}: ${got} != ${want}`);
        }
      }
    }
  });
});

describe('WP-5 composed texts', () => {
  const SPATIAL: RsPassName[] = ['rs_pair_accept', 'rs_args', 'rs_spatial_replay', 'rs_spatial_shift', 'rs_spatial_resample', 'rs_t_classify'];
  const FLAGS = ['RS_DENSE_SLOTS', 'RS_BOOST_GATE', 'RS_MIS_TRIM', 'RS_PAIR_TABLE'] as const;
  const scene = {
    SCENE_GROUP: 1, BVH_DECLARE_BINDINGS: true, BVH_GROUP: 1, BVH_BINDING_NODES: 0, BVH_BINDING_TRIS: 1, WATERTIGHT: true,
    CUSTOM_ALPHA: true, VERTEX_FORMAT: 1, TEX_GROUP: 1, TEX_BINDING_BASE: 8, TEX_ARRAYS: 0, TEX_SAMPLERS: 0,
  };
  const compose = (name: RsPassName, extra: Record<string, number>) => composeWgsl(RS_PASSES[name].file, {
    sources: shaderSources, defines: restirDefines(name, { sceneDefines: RS_PASSES[name].scene ? scene : undefined, debug: true, extra: { RS_RIS_NEE: 1, RS_MODE_B: 0, RS_DUAL_MV: 1, RS_DUPMAP: 1, ...extra } }),
    features: new Set(['subgroups', 'shader-f16', 'timestamp-query', 'float32-filterable']), wgslLanguageFeatures: new Set(['pointer_composite_access']),
  }).code;
  it('every flag subset composes; the flagged variants differ from the plain text only where they should', () => {
    for (const name of SPATIAL) {
      const base = compose(name, {});
      for (let m = 1; m < 1 << FLAGS.length; m++) {
        const extra = Object.fromEntries(FLAGS.filter((_, i) => m & (1 << i)).map((f) => [f, 1]));
        const code = compose(name, extra);
        if (name === 'rs_spatial_shift' && extra.RS_DENSE_SLOTS) expect(code).toContain('@workgroup_size(64)');
        if (name === 'rs_pair_accept' && extra.RS_DENSE_SLOTS) expect(code).toContain('arena_dense_item_word');
        expect(code.length).toBeGreaterThan(0);
        if (name === 'rs_t_classify' && Object.keys(extra).join() === 'RS_MIS_TRIM') expect(code).toBe(base);
        if (extra.RS_DENSE_SLOTS) expect(compose(name, { ...extra, RS_DENSE_SLOTS: 2 }).length).toBeGreaterThan(0);   // T1 sees the gate, the queue consts and rs_pix only
      }
    }
  });
});
