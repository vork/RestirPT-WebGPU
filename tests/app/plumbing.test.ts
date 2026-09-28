import { describe, expect, it } from 'vitest';
import { classifyFile, sourcesFromFiles, sourcesFromQuery } from '../../src/app/loader.ts';
import { DEBUG_BUFFER_LAYOUT } from '../../src/core/render/debug-views.ts';
import { overlayViewProj } from '../../src/core/render/overlay.ts';
import { internalResolution } from '../../src/core/render/present.ts';
import { parseDebugHeader, pickPixel } from '../../src/core/render/probe.ts';

describe('internal resolution (DPR independent)', () => {
  it('540p at the canvas aspect: height 540, width rounded to a multiple of 8', () => {
    expect(internalResolution('540p', 1920, 1080)).toEqual([960, 540]);
    expect(internalResolution('540p', 3840, 2160)).toEqual([960, 540]); // DPR 2 backing store: same internal res
    expect(internalResolution('720p', 2560, 1080)).toEqual([1704, 720]);
    expect(internalResolution('1080p', 1000, 1000)).toEqual([1080, 1080]);
    expect(internalResolution('native', 1234, 567)).toEqual([1234, 567]);
    const [w] = internalResolution('540p', 1437, 811);
    expect(w % 8).toBe(0);
  });
});

describe('probe', () => {
  it('maps CSS clicks to internal pixels (row 0 = top)', () => {
    expect(pickPixel(0, 0, 800, 450, 960, 540)).toEqual([0, 0]);
    expect(pickPixel(799.9, 449.9, 800, 450, 960, 540)).toEqual([959, 539]);
    expect(pickPixel(400, 225, 800, 450, 960, 540)).toEqual([480, 270]);
    expect(pickPixel(-5, 1e6, 800, 450, 960, 540)).toEqual([0, 539]);
  });
  it('parses the DebugBuffer header (counters + records sorted by seq, capacity clamp)', () => {
    const L = DEBUG_BUFFER_LAYOUT;
    const buf = new ArrayBuffer(L.headerBytes);
    const u = new Uint32Array(buf);
    const f = new Float32Array(buf);
    u[0] = 2; u[2] = 5; // count, NaN
    const put = (slot: number, seq: number, tag: number, v: number) => {
      const b = (L.probeOffset + slot * L.probeStride) / 4;
      u[b] = 3; u[b + 1] = 4; u[b + 2] = tag; u[b + 3] = seq; f[b + 4] = v;
    };
    put(0, 1, 16, 2.5);
    put(1, 0, 1, 1.5);
    const p = parseDebugHeader(buf, 9);
    expect(p.counters[2]).toBe(5);
    expect(p.records.map((r) => r.value[0])).toEqual([1.5, 2.5]);
    expect(p.records[0].pixel).toEqual([3, 4]);
    u[0] = 100000;
    expect(parseDebugHeader(buf, 9).records.length).toBe(L.probeCapacity);
  });
});

describe('overlay projection', () => {
  it('reversed infinite perspective: z_ndc = near / depth, image centre on the axis', () => {
    const I = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1];
    const vp = overlayViewProj(I, Math.PI / 2, 2, 0.01);
    const clip = (p: number[]) => [0, 1, 2, 3].map((r) => vp[r] * p[0] + vp[4 + r] * p[1] + vp[8 + r] * p[2] + vp[12 + r]);
    const c = clip([0, 0, -4]);
    expect(c[0] / c[3]).toBeCloseTo(0, 12);
    expect(c[2] / c[3]).toBeCloseTo(0.01 / 4, 12);
    const top = clip([0, 4, -4]); // tan(45°) = 1 -> top edge
    expect(top[1] / top[3]).toBeCloseTo(1, 12);
    const right = clip([8, 0, -4]); // aspect 2
    expect(right[0] / right[3]).toBeCloseTo(1, 12);
  });
});

describe('loader plumbing', () => {
  const file = (name: string) => new File([new Uint8Array(1)], name);
  it('classifies files', () => {
    expect(classifyFile('Sponza.gltf')).toBe('scene');
    expect(classifyFile('x.USDZ')).toBe('scene');
    expect(classifyFile('sky.exr')).toBe('env');
    expect(classifyFile('Sponza.bin')).toBe('aux');
    expect(classifyFile('notes.txt')).toBe('unknown');
  });
  it('groups a dropped glTF with its buffers/textures and picks an env', () => {
    const s = sourcesFromFiles([file('a.bin'), file('Sponza.gltf'), file('t.png'), file('sky.hdr'), file('b.glb'), file('x.txt')]);
    expect(s.scene?.kind === 'files' && s.scene.main.name).toBe('Sponza.gltf');
    expect(s.scene?.kind === 'files' && s.scene.files.length).toBe(6);
    expect(s.env?.name).toBe('sky.hdr');
    expect(s.ignored).toEqual(['b.glb', 'x.txt']);
  });
  it('parses ?scene=&env= relative to the page and rejects other protocols', () => {
    const s = sourcesFromQuery('?scene=assets/cornell.glb&env=javascript:alert(1)', 'http://localhost:5173/index.html');
    expect(s.scene).toEqual({ kind: 'url', url: 'http://localhost:5173/assets/cornell.glb', name: 'cornell.glb' });
    expect(s.env).toBeUndefined();
    expect(s.ignored.length).toBe(1);
  });
});
