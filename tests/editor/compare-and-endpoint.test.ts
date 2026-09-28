// Compare-view statistics (src/app/compare/compare-math.ts) and the /api/reference request validation
// (validation/harness/reference-endpoint.ts): seeds/frames normalisation, package-dir confinement, loopback checks.
import { mkdtempSync, mkdirSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import path from 'node:path';
import { describe, expect, it } from 'vitest';
import {
  diverging, halfToFloat, heat, meanImage, relativeError, srgb8, summarize, tileTMap, welch, type RgbaImage,
} from '../../src/app/compare/compare-math.ts';
import {
  ReferenceRequestError, isLoopbackAddress, isLoopbackOrigin, normalizeFrames, normalizeSeeds, resolvePackageDir, validateRequest,
} from '../../validation/harness/reference-endpoint.ts';

const img = (w: number, h: number, f: (x: number, y: number) => number): RgbaImage => {
  const d = new Float32Array(w * h * 4);
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) { const v = f(x, y); d.set([v, v, v, 1], (y * w + x) * 4); }
  return { width: w, height: h, data: d };
};

describe('compare statistics', () => {
  it('Welch t matches a hand computation; zero variance on one side is allowed', () => {
    const r = welch([1, 2, 3, 4], [2, 4, 6]);
    // means 2.5 / 4, var 1.6667 / 4 → se² = 1.6667/4 + 4/3
    expect(r.diff).toBeCloseTo(-1.5, 12);
    expect(r.se).toBeCloseTo(Math.sqrt(1.6666666666666667 / 4 + 4 / 3), 12);
    expect(r.t).toBeCloseTo(-1.5 / r.se, 12);
    const df = (1.6666666666666667 / 4 + 4 / 3) ** 2 / ((1.6666666666666667 / 4) ** 2 / 3 + (4 / 3) ** 2 / 2);
    expect(r.df).toBeCloseTo(df, 10);
    const one = welch([5], [4, 6]);
    expect(one.t).toBeCloseTo(0, 12);
    expect(welch([1], [1]).t).toBe(0);
  });

  it('relative error, tile t-map and summary detect a planted +5% region', () => {
    const W = 64, H = 32;
    const noise = (s: number) => (x: number, y: number) => 1 + 0.02 * Math.sin(1.7 * x + 3.1 * y + s);
    const ref = [0, 1, 2, 3].map((s) => img(W, H, noise(s)));
    const ours = [4, 5, 6].map((s) => img(W, H, (x, y) => noise(s)(x, y) * (x < 16 && y < 16 ? 1.05 : 1)));
    const e = relativeError(meanImage(ours), meanImage(ref));
    let inR = 0, outR = 0;
    for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) { if (x < 16 && y < 16) inR += e[y * W + x]; else outR += e[y * W + x]; }
    expect(inR / 256).toBeGreaterThan(0.03);
    expect(outR / (W * H - 256)).toBeLessThan(0.02);
    const tm = tileTMap(ours, ref, 16);
    expect([tm.tx, tm.ty]).toEqual([4, 2]);
    expect(tm.valid).toBe(true);
    expect(tm.t[0]).toBeGreaterThan(5);
    for (let k = 1; k < 8; k++) expect(Math.abs(tm.t[k])).toBeLessThan(5);
    const s = summarize(ours, ref, 16);
    expect(s.relDiff[3]).toBeGreaterThan(0.005);
    expect(s.tilesOver).toBeGreaterThan(0);
    expect(tileTMap([ours[0]], [ref[0]], 16).valid).toBe(false);
  });

  it('display helpers: sRGB OETF, colour maps, half floats', () => {
    expect(srgb8(0)).toBe(0);
    expect(srgb8(1)).toBe(255);
    expect(srgb8(0.18)).toBe(118);
    expect(srgb8(Number.NaN)).toBe(0);
    expect(heat(Number.NaN, 1)).toEqual([255, 0, 255]);
    expect(diverging(0)).toEqual([255, 255, 255]);
    expect(diverging(10)).toEqual([255, 0, 0]);
    expect(diverging(-10)).toEqual([0, 0, 255]);
    expect(halfToFloat(0x3c00)).toBe(1);
    expect(halfToFloat(0xc000)).toBe(-2);
    expect(halfToFloat(0x0001)).toBe(2 ** -24);
    expect(halfToFloat(0x7c00)).toBe(Infinity);
    expect(Number.isNaN(halfToFloat(0x7e00))).toBe(true);
  });
});

describe('/api/reference request validation', () => {
  const root = mkdtempSync(path.join(tmpdir(), 'refep-'));
  mkdirSync(path.join(root, 'validation/out/pkg1'), { recursive: true });
  writeFileSync(path.join(root, 'validation/out/pkg1/scene.json'), '{}');
  mkdirSync(path.join(root, 'validation/out/nojson'), { recursive: true });
  mkdirSync(path.join(root, 'elsewhere'), { recursive: true });
  writeFileSync(path.join(root, 'elsewhere/scene.json'), '{}');

  it('normalises seeds and frames', () => {
    expect(normalizeSeeds('0..3')).toBe('0,1,2,3');
    expect(normalizeSeeds('1, 5,7..8')).toBe('1,5,7,8');
    expect(normalizeSeeds([2, 4])).toBe('2,4');
    for (const bad of ['', 'a', '0;rm -rf', '1,1', '0..999']) expect(() => normalizeSeeds(bad)).toThrow(ReferenceRequestError);
    expect(normalizeFrames(undefined)).toBe('all');
    expect(normalizeFrames('all')).toBe('all');
    expect(normalizeFrames([0, 12])).toBe('0,12');
    expect(() => normalizeFrames([-1])).toThrow(ReferenceRequestError);
    expect(() => normalizeFrames('1,x')).toThrow(ReferenceRequestError);
  });

  it('confines packageDir to validation/out or validation/scenes and requires scene.json', () => {
    expect(resolvePackageDir(root, 'validation/out/pkg1')).toBe(path.join(root, 'validation/out/pkg1'));
    expect(resolvePackageDir(root, '/validation/out/pkg1')).toBe(path.join(root, 'validation/out/pkg1'));
    for (const bad of ['validation/out/../../elsewhere', 'elsewhere', '/etc', 'validation/out', 'validation/out/nojson', '']) {
      expect(() => resolvePackageDir(root, bad), bad).toThrow(ReferenceRequestError);
    }
  });

  it('builds the render_reference.py argument list and bounds the numbers', () => {
    const v = validateRequest(root, { packageDir: 'validation/out/pkg1', spp: 4, seeds: '0..1', frames: [3], maxBounces: 2 });
    expect(v.args).toEqual(['--package', path.join(root, 'validation/out/pkg1'), '--spp', '4', '--seeds', '0,1', '--frames', '3', '--max-bounces', '2']);
    expect(v.total(10)).toBe(2);
    expect(validateRequest(root, { packageDir: 'validation/out/pkg1', spp: 1, seeds: '0..2' }).total(5)).toBe(15);
    for (const bad of [{ spp: 0 }, { spp: 1.5 }, { spp: 1e6 }, { spp: 4, maxBounces: -1 }, { spp: 4, device: 'TPU' }]) {
      expect(() => validateRequest(root, { packageDir: 'validation/out/pkg1', seeds: '0', ...bad })).toThrow(ReferenceRequestError);
    }
  });

  it('accepts loopback clients/origins only', () => {
    for (const a of ['127.0.0.1', '::1', '::ffff:127.0.0.1']) expect(isLoopbackAddress(a)).toBe(true);
    for (const a of ['192.168.1.2', '10.0.0.1', undefined]) expect(isLoopbackAddress(a)).toBe(false);
    expect(isLoopbackOrigin(undefined)).toBe(true);
    expect(isLoopbackOrigin('http://localhost:5173')).toBe(true);
    expect(isLoopbackOrigin('http://127.0.0.1:61234')).toBe(true);
    expect(isLoopbackOrigin('https://evil.example')).toBe(false);
    expect(isLoopbackOrigin('http://localhost.evil.example')).toBe(false);
    expect(isLoopbackOrigin('null')).toBe(false);
  });
});
