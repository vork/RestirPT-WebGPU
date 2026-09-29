// U-ENS-2 CPU part (restir-api.md §2.10, §6.1): ensemble.npz written by npz.ts reads back in TS and through numpy +
// compare.py's loader (stats.replicates_from_sums) with the expected shapes and values.
import { spawnSync } from 'node:child_process';
import { existsSync, mkdtempSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { describe, expect, it } from 'vitest';
import { crc32, readNpz, writeNpz } from '../../src/core/render/restir/npz.ts';
import { EnsembleCollector, decodeEnsStats, ensStatsLayout } from '../../src/core/render/restir/ensemble.ts';

const root = resolve(dirname(fileURLToPath(import.meta.url)), '../..');
const python = join(root, 'validation/.venv/bin/python');
const hasVenv = existsSync(python);

/** A synthetic 2-frame, E = 3 ensemble of 40×24 member images and the collector that reduces it (as the GPU would). */
function synthetic(): { col: EnsembleCollector; imgs: Float64Array[] } {
  const E = 3, W = 40, H = 24;
  const col = new EnsembleCollector(E, W, H, 4002);
  const imgs: Float64Array[] = [];
  let seed = 1;
  const rnd = () => { seed = (Math.imul(seed, 1664525) + 1013904223) >>> 0; return seed / 2 ** 32; };
  const px = new Float32Array(W * H * 8);
  for (let t = 0; t < 2; t++) {
    const L = ensStatsLayout(E, W, H);
    const raw = new Float32Array(L.total * 4);
    for (let m = 0; m < E; m++) {
      const img = new Float64Array(W * H * 3);
      for (let i = 0; i < img.length; i++) img[i] = Math.fround(rnd() * 3);
      imgs.push(img);
      for (const tl of L.tiles) {
        for (let ty = 0; ty < tl.th; ty++) for (let tx = 0; tx < tl.tw; tx++) for (let c = 0; c < 3; c++) {
          let s = 0;
          for (let y = ty * tl.level; y < Math.min(H, (ty + 1) * tl.level); y++) for (let x = tx * tl.level; x < Math.min(W, (tx + 1) * tl.level); x++) s += img[3 * (y * W + x) + c];
          raw[4 * (L.off[tl.level] + (m * tl.th + ty) * tl.tw + tx) + c] = s;
        }
      }
      for (let c = 0; c < 3; c++) { let s = 0; for (let i = 0; i < W * H; i++) s += img[3 * i + c]; raw[4 * (L.global + m) + c] = s; }
      for (let i = 0; i < W * H; i++) for (let c = 0; c < 3; c++) { const v = img[3 * i + c]; px[8 * i + c] += v; px[8 * i + 4 + c] += v * v; }
    }
    col.addFrameStats(decodeEnsStats(raw, E, W, H), t);
  }
  col.addPixelMoments(px);
  return { col, imgs };
}

describe('npz writer', () => {
  it('crc32 reference ("123456789" → cbf43926)', () => {
    expect(crc32(new TextEncoder().encode('123456789'))).toBe(0xcbf43926);
  });

  it('TS round trip: shapes, dtypes, values, strings, scalars', () => {
    const { col } = synthetic();
    const z = readNpz(writeNpz(col.npzArrays()));
    expect([...z.keys()].sort()).toEqual(['channels', 'count', 'global', 'height', 'pixel_sum', 'pixel_sumsq', 'tiles16', 'tiles32', 'tiles64', 'width'].sort());
    expect(z.get('tiles16')!.shape).toEqual([6, 2, 3, 3]);
    expect(z.get('tiles64')!.shape).toEqual([6, 1, 1, 3]);
    expect(z.get('global')!.shape).toEqual([6, 3]);
    expect(z.get('pixel_sum')!.shape).toEqual([24, 40, 3]);
    expect(z.get('channels')!.data).toEqual(['R', 'G', 'B']);
    expect(Number((z.get('count')!.data as BigInt64Array)[0])).toBe(6);
    expect(z.get('count')!.shape).toEqual([]);
    expect(col.seeds).toEqual(['4002:0:0', '4002:0:1', '4002:0:2', '4002:1:0', '4002:1:1', '4002:1:2']);
  });

  describe.skipIf(!hasVenv)('numpy + compare.py loader (validation/.venv)', () => {
    it('np.load(allow_pickle=False) and stats.replicates_from_sums read it: means per tile/global, per-pixel moments', () => {
      const { col, imgs } = synthetic();
      const dir = mkdtempSync(join(tmpdir(), 'npz-'));
      const path = join(dir, 'ensemble.npz');
      writeFileSync(path, writeNpz(col.npzArrays()));
      const script = `
import json, sys
import numpy as np
sys.path.insert(0, ${JSON.stringify(join(root, 'validation/tools'))})
import stats as S
with np.load(${JSON.stringify(path)}, allow_pickle=False) as z:
    d = {k: z[k] for k in z.files}
r = S.replicates_from_sums(d, ("Y", "R", "G", "B"))
print(json.dumps({"count": r.count, "h": r.height, "w": r.width, "glob": r.glob.tolist(), "t16": r.tiles[16][:, 0, 0, :].tolist(),
  "ps": r.pixel_sum[3, 5, :].tolist(), "pq": r.pixel_sumsq[3, 5, 1:].tolist(), "dt": str(d["tiles16"].dtype), "ch": d["channels"].tolist()}))
`;
      const res = spawnSync(python, ['-c', script], { encoding: 'utf8' });
      expect(res.status, res.stderr).toBe(0);
      const o = JSON.parse(res.stdout.trim().split('\n').pop()!);
      expect(o.count).toBe(6);
      expect([o.h, o.w]).toEqual([24, 40]);
      expect(o.dt).toBe('float64');
      expect(o.ch).toEqual(['R', 'G', 'B']);
      const W = 40, H = 24;
      for (let r = 0; r < 6; r++) {
        const img = imgs[r];
        for (let c = 0; c < 3; c++) {
          let s = 0, t = 0;
          for (let i = 0; i < W * H; i++) s += img[3 * i + c];
          for (let y = 0; y < 16; y++) for (let x = 0; x < 16; x++) t += img[3 * (y * W + x) + c];
          expect(o.glob[r][1 + c]).toBeCloseTo(s / (W * H), 5);
          expect(o.t16[r][1 + c]).toBeCloseTo(t / 256, 5);
        }
        const y = 0.2126 * o.glob[r][1] + 0.7152 * o.glob[r][2] + 0.0722 * o.glob[r][3];
        expect(o.glob[r][0]).toBeCloseTo(y, 9);
      }
      for (let c = 0; c < 3; c++) {
        let s = 0, q = 0;
        for (const img of imgs) { const v = img[3 * (3 * W + 5) + c]; s += v; q += v * v; }
        expect(o.ps[1 + c]).toBeCloseTo(s, 4);
        expect(o.pq[c]).toBeCloseTo(q, 3);
      }
    });
  });
});
