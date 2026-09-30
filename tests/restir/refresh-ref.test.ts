// Suffix refresh, CPU part (T-C, restir-temporal-api.md §3.2, §3.5, §4.1, §6.1 "§9.3-2"): the class derivation mirror,
// the discrete PSS change of variables of the light-selection dimension with J_P (exact, f64) and the symmetric
// undefined predicate over random light-set edits (add / remove / reorder / intensity), and the refresh unit builders
// (gating by TF_REFRESH / TF_HIST_VALID / history / N1-mixed, row bands, Q_i item chunks, queue selection, args offset).
import { describe, expect, it } from 'vitest';
import { PATH_CLASS_NAMES, RS_WGSL_CONSTS as K, pathClass } from '../../src/core/render/restir/layout.ts';
import { refreshClass, refreshEntry, refreshFwdUnits, refreshInvUnits } from '../../src/core/render/restir/refresh.ts';
import type { RestirKernel } from '../../src/core/render/restir/kernel.ts';

const NONE = 0xFFFFFFFF;

function rng(seed: number): () => number {
  let s = seed >>> 0;
  return () => { s = (Math.imul(s ^ (s >>> 15), 0x2c1b3c6d) + 0x9e3779b9) >>> 0; s ^= s >>> 13; return (s >>> 0) / 2 ** 32; };
}

describe('refreshClass ≡ layout.ts pathClass (math.md#reservoir-fields)', () => {
  it('every (d, k, tech) of a non-empty record', () => {
    for (let d = 1; d <= 15; d++) for (let k = 0; k <= d; k++) for (let tech = 0; tech < 4; tech++) {
      expect(refreshClass(d, k, tech), `${d} ${k} ${tech}`).toBe(pathClass({ d, k, tech }));
    }
    expect(PATH_CLASS_NAMES[refreshClass(5, 2, K.RS_TECH_NEE)]).toBe('D-NEE');
  });
});

/** A frame's light set: stable ids in alias order, weights → realized pmf; + a static triangle block and an env entry. */
interface Frame { ids: number[]; w: number[]; nTri: number; env: boolean; wTri: number; wEnv: number }
function pmfOf(f: Frame): number[] {
  const all = [...f.w, ...Array(f.nTri).fill(f.wTri), ...(f.env ? [f.wEnv] : [])];
  const s = all.reduce((a, b) => a + b, 0);
  return all.map((x) => x / s);
}
/** lt_translate of restir-temporal-api.md §3.2 on CPU: analytic by stable id, triangles by offset, env to env. */
function translate(from: Frame, to: Frame, e: number): number {
  const nA = from.ids.length, nT = from.nTri;
  if (from.env && e === nA + nT) return to.env ? to.ids.length + to.nTri : NONE;
  if (e < nA) { const j = to.ids.indexOf(from.ids[e]); return j < 0 ? NONE : j; }
  return to.ids.length + (e - nA);
}
function editFrame(f: Frame, r: () => number, nextId: { v: number }): Frame {
  const ids = [...f.ids], w = [...f.w];
  if (r() < 0.5 && ids.length > 1) { const i = Math.floor(r() * ids.length); ids.splice(i, 1); w.splice(i, 1); }        // remove
  if (r() < 0.5) { const i = Math.floor(r() * (ids.length + 1)); ids.splice(i, 0, nextId.v++); w.splice(i, 0, 0.5 + r()); } // add
  if (r() < 0.3 && ids.length > 1) { const i = Math.floor(r() * ids.length), j = Math.floor(r() * ids.length); [ids[i], ids[j]] = [ids[j], ids[i]]; [w[i], w[j]] = [w[j], w[i]]; } // reorder
  for (let i = 0; i < w.length; i++) if (r() < 0.3) w[i] *= r() < 0.5 ? 2 : 0.5;                                         // intensity
  if (r() < 0.1 && w.length) w[Math.floor(r() * w.length)] = 0;                                                          // pmf 0
  return { ...f, ids, w, wEnv: r() < 0.5 ? f.wEnv * (0.5 + r()) : f.wEnv };
}

describe('§9.3-2 (b), discrete part: Σ pmf_{t−1}(e)·h(T e)·J_P·1_D = Σ pmf_t(e′)·h(e′)·1_I, and the undefined predicate is symmetric', () => {
  it('500 random edit sequences (add, remove, reorder, intensity steps, pmf 0, env strength) with triangles and env', () => {
    const r = rng(7);
    let checked = 0;
    for (let seq = 0; seq < 500; seq++) {
      const nextId = { v: 100 };
      let prev: Frame = { ids: [1, 2, 3, 4].slice(0, 1 + Math.floor(r() * 4)), w: [1, 2, 0.5, 3].map((x) => x * (0.5 + r())), nTri: Math.floor(r() * 3), env: r() < 0.7, wTri: 0.2 + r(), wEnv: 1 + r() };
      prev.w = prev.w.slice(0, prev.ids.length);
      for (let step = 0; step < 4; step++) {
        const cur = editFrame(prev, r, nextId);
        const pp = pmfOf(prev), pc = pmfOf(cur);
        const idOf = (f: Frame, e: number) => (e < f.ids.length ? f.ids[e] : e < f.ids.length + f.nTri ? 1000 + e - f.ids.length : 2000);
        const h = (e: number) => 1 + ((idOf(cur, e) * 2654435761) >>> 0) / 2 ** 32;          // any function of the frame-t light
        let lhs = 0, rhs = 0;
        for (let e = 0; e < pp.length; e++) {
          const fwd = refreshEntry(e, (x) => translate(prev, cur, x), (x) => pp[x], (x) => pc[x]);
          if (pp[e] > 0 && !fwd.undef) lhs += pp[e] * h(fwd.eTo) * (pc[fwd.eTo] / pp[e]);
          // symmetry: e ∈ D (forward defined) ⇔ T e ∈ I (inverse of T e defined and maps back to e)
          if (pp[e] > 0) {
            const back = fwd.eTo === NONE ? undefined : refreshEntry(fwd.eTo, (x) => translate(cur, prev, x), (x) => pc[x], (x) => pp[x]);
            expect(!fwd.undef, `seq ${seq} e ${e}`).toBe(back !== undefined && !back.undef && back.eTo === e);
            if (back && !back.undef) expect(Math.abs(Math.log(fwd.jp * back.jp))).toBeLessThan(1e-6);   // J_P·J_P⁻¹ = 1 (f32)
          }
        }
        for (let e = 0; e < pc.length; e++) {
          const inv = refreshEntry(e, (x) => translate(cur, prev, x), (x) => pc[x], (x) => pp[x]);
          if (pc[e] > 0 && !inv.undef) rhs += pc[e] * h(e);
        }
        expect(Math.abs(lhs - rhs)).toBeLessThan(1e-12 * Math.max(1, rhs));
        checked++;
        prev = cur;
      }
    }
    expect(checked).toBe(2000);
  });

  it('plants: TP_NO_JP ⇒ J_P = 1; TP_NO_JP_ENV ⇒ J_P = 1 for env endpoints only; missing / zero-pmf entries undefined', () => {
    const tr = (e: number) => (e === 3 ? NONE : e);
    const pf = (e: number) => [0.25, 0.25, 0.5, 0.1][e], pt = (e: number) => [0.5, 0, 0.5, 0.2][e];
    expect(refreshEntry(0, tr, pf, pt)).toEqual({ eTo: 0, jp: 2, undef: false });
    expect(refreshEntry(1, tr, pf, pt).undef).toBe(true);                                   // pmf_to = 0
    expect(refreshEntry(3, tr, pf, pt)).toEqual({ eTo: NONE, jp: 1, undef: true });          // removed
    expect(refreshEntry(NONE, tr, pf, pt).undef).toBe(true);
    expect(refreshEntry(0, tr, pf, pt, { noJP: true }).jp).toBe(1);
    expect(refreshEntry(0, tr, pf, pt, { noJPEnv: true }).jp).toBe(2);
    expect(refreshEntry(0, tr, pf, pt, { noJPEnv: true, isEnv: true }).jp).toBe(1);
  });
});

// ------------------------------------------------------------------------------------------------ unit builders

interface Enc { name: string; d: Record<string, number>; work: unknown }
function fakeKernel(o: { h: number; w: number; flags?: number; rowBand: number; atlas: [number, number]; n1Mixed?: boolean }): { k: RestirKernel; calls: Enc[] } {
  const calls: Enc[] = [];
  const [W, H] = o.atlas;
  const args = { label: 'args' };
  const k = {
    historyIndex: () => o.h, resBase: () => o.w,
    currentAdvance: o.flags === undefined ? undefined : { flags: o.flags },
    settings: { tPlant: { n1Mixed: o.n1Mixed } },
    resources: { alloc: { atlasW: W, atlasH: H }, args, g2: (name: string, idx = 0) => ({ name, idx }) },
    pipelineSync: (n: string) => ({ n }),
    rowBands: () => { const out: [number, number][] = []; for (let r = 0; r < H; r += o.rowBand) out.push([r, Math.min(H, r + o.rowBand)]); return out; },
    perPixelWorkgroups: (r0: number, r1: number) => [Math.ceil(W / 8), Math.ceil((r1 - r0) / 8)],
    encodePass: (_enc: unknown, name: string, _pl: unknown, g2: { idx: number }, d: Record<string, number>, work: unknown) => calls.push({ name, d: { ...d, g2: g2.idx }, work }),
  } as unknown as RestirKernel;
  return { k, calls };
}

describe('refresh unit builders (restir-temporal-api.md §4.1, §4.4)', () => {
  const REF = K.TF_REFRESH | K.TF_HIST_VALID;
  it('gating: history, TF_REFRESH, TF_HIST_VALID (fwd), N1-mixed (inv); unknown flags ⇒ emitted (the passes check)', () => {
    expect(refreshFwdUnits(fakeKernel({ h: -1, w: 0, flags: REF, rowBand: 16, atlas: [16, 16] }).k, 3)).toEqual([]);
    expect(refreshFwdUnits(fakeKernel({ h: 0, w: 1, flags: K.TF_HIST_VALID, rowBand: 16, atlas: [16, 16] }).k, 3)).toEqual([]);
    expect(refreshFwdUnits(fakeKernel({ h: 0, w: 1, flags: K.TF_REFRESH, rowBand: 16, atlas: [16, 16] }).k, 3)).toEqual([]);
    expect(refreshFwdUnits(fakeKernel({ h: 0, w: 1, flags: REF, rowBand: 16, atlas: [16, 16] }).k, 3)).toHaveLength(1);
    expect(refreshFwdUnits(fakeKernel({ h: 0, w: 1, rowBand: 16, atlas: [16, 16] }).k, 3)).toHaveLength(1);
    expect(refreshFwdUnits(fakeKernel({ h: 0, w: 1, flags: 0, rowBand: 16, atlas: [16, 16] }).k, 3, REF)).toHaveLength(1);   // explicit flags win
    expect(refreshInvUnits(fakeKernel({ h: 0, w: 1, flags: K.TF_HIST_VALID, rowBand: 16, atlas: [16, 16] }).k, 3)).toEqual([]);
    expect(refreshInvUnits(fakeKernel({ h: 0, w: 1, flags: REF, rowBand: 16, atlas: [16, 16], n1Mixed: true }).k, 3)).toEqual([]);
    expect(refreshInvUnits(fakeKernel({ h: 0, w: 1, flags: REF, rowBand: 16, atlas: [16, 16] }).k, 3)).toHaveLength(1);
  });

  it('fwd: one unit per row band over res[h]; inv: rs_args(q2) + indirect rs_refresh_inv per item chunk over res[w]', () => {
    const { k, calls } = fakeKernel({ h: 1, w: 0, flags: REF, rowBand: 8, atlas: [32, 24] });
    const fwd = refreshFwdUnits(k, 9), inv = refreshInvUnits(k, 9);
    expect(fwd.map((u) => u.label)).toEqual(['rs_refresh_fwd[0]', 'rs_refresh_fwd[8]', 'rs_refresh_fwd[16]']);
    expect(inv.map((u) => u.label)).toEqual(['rs_refresh_inv[0]', 'rs_refresh_inv[1]', 'rs_refresh_inv[2]']);
    expect(fwd[0].costHint).toBe(32 * 8 * 2);
    for (const u of [...fwd, ...inv]) u.encode({} as GPUCommandEncoder);
    const f = calls.filter((c) => c.name === 'rs_refresh_fwd');
    expect(f.map((c) => [c.d.rowBase, c.d.rowEnd, c.d.passId, c.d.g2, c.d.t])).toEqual([[0, 8, K.RS_PASS_T_REFRESH_FWD, 1, 9], [8, 16, K.RS_PASS_T_REFRESH_FWD, 1, 9], [16, 24, K.RS_PASS_T_REFRESH_FWD, 1, 9]]);
    const a = calls.filter((c) => c.name === 'rs_args'), i = calls.filter((c) => c.name === 'rs_refresh_inv');
    const P = 32 * 24, chunk = P / 3;
    expect(a.map((c) => [c.d.treeBase, c.d.treeCount, c.d.flags >>> K.RSD_QUEUE_SHIFT])).toEqual([[0, chunk, K.RS_Q_INV], [chunk, chunk, K.RS_Q_INV], [2 * chunk, chunk, K.RS_Q_INV]]);
    expect(i.map((c) => [c.d.treeBase, c.d.treeCount, c.d.g2, c.d.passId])).toEqual([[0, chunk, 0, K.RS_PASS_T_REFRESH_INV], [chunk, chunk, 0, K.RS_PASS_T_REFRESH_INV], [2 * chunk, chunk, 0, K.RS_PASS_T_REFRESH_INV]]);
    expect(i.every((c) => (c.work as { offset: number }).offset === 16 * K.RS_Q_INV)).toBe(true);
    // one band ⇒ one unbounded chunk (count 0 = the whole queue)
    const one = fakeKernel({ h: 1, w: 0, flags: REF, rowBand: 64, atlas: [32, 24] });
    for (const u of refreshInvUnits(one.k, 1)) u.encode({} as GPUCommandEncoder);
    expect(one.calls.filter((c) => c.name === 'rs_args').map((c) => [c.d.treeBase, c.d.treeCount])).toEqual([[0, 0]]);
  });
});
