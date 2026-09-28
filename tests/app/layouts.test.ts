// CPU check that the TS packers match the WGSL struct layouts (WGSL host-shareable layout rules, computed from the
// shader source), plus composition of every app/debug shader through the real composer.
import { describe, expect, it } from 'vitest';
import { composeWgsl } from '../../src/core/gpu/wgsl-composer.ts';
import { DEBUG_BUFFER_LAYOUT, DEBUG_PARAMS_SIZE, DebugViewRegistry, BUILTIN_VIEWS, DBG, packDebugParams, defaultDebugSettings, debugFlags } from '../../src/core/render/debug-views.ts';
import { FRAME_LAYOUT, packFrameUniforms, recentre, rigidInverse, r2Jitter, FRAME_CAMERA_MOVED } from '../../src/core/render/frame-uniforms.ts';
import { shaderSources } from '../../src/core/shaders/index.ts';
import testPattern from '../../src/app/test-pattern.wgsl?raw';

type Layout = { size: number; align: number; offsets: Record<string, number> };

function structLayouts(src: string): Map<string, Layout> {
  const code = src.replace(/\/\/.*$/gm, '');
  const out = new Map<string, Layout>();
  const scalar = (t: string): { size: number; align: number } | undefined => {
    t = t.trim();
    if (/^(f32|u32|i32|atomic<u32>|atomic<i32>)$/.test(t)) return { size: 4, align: 4 };
    let m = /^vec([234])(?:f|u|i|<\w+>)$/.exec(t);
    if (m) { const n = Number(m[1]); return n === 2 ? { size: 8, align: 8 } : n === 3 ? { size: 12, align: 16 } : { size: 16, align: 16 }; }
    if (/^mat4x4(f|<f32>)$/.test(t)) return { size: 64, align: 16 };
    m = /^array<(.+),\s*(\d+)>$/.exec(t);
    if (m) { const e = scalar(m[1])!; const stride = Math.ceil(e.size / e.align) * e.align; return { size: stride * Number(m[2]), align: e.align }; }
    m = /^array<(.+)>$/.exec(t);
    if (m) { const e = scalar(m[1])!; return { size: 0, align: e.align }; }
    const s = out.get(t);
    return s ? { size: s.size, align: s.align } : undefined;
  };
  for (const m of code.matchAll(/struct\s+(\w+)\s*\{([^}]*)\}/g)) {
    let off = 0;
    let align = 1;
    const offsets: Record<string, number> = {};
    const fields: string[] = [];
    let depth = 0;
    let cur = '';
    for (const ch of m[2]) {
      if (ch === '<') depth++;
      if (ch === '>') depth--;
      if (ch === ',' && depth === 0) { fields.push(cur); cur = ''; } else cur += ch;
    }
    fields.push(cur);
    for (const f of fields.map((s) => s.trim()).filter(Boolean)) {
      const fm = /^(?:@\w+(?:\([^)]*\))?\s*)*(\w+)\s*:\s*(.+)$/.exec(f);
      if (!fm) continue;
      const t = scalar(fm[2]);
      if (!t) throw new Error(`unknown type ${fm[2]} in ${m[1]}`);
      off = Math.ceil(off / t.align) * t.align;
      offsets[fm[1]] = off;
      off += t.size;
      align = Math.max(align, t.align);
    }
    out.set(m[1], { size: Math.ceil(off / align) * align, align, offsets });
  }
  return out;
}

const compose = (entry: string, defines = {}) =>
  composeWgsl(entry, { sources: { ...shaderSources, 'app/test-pattern.wgsl': testPattern }, defines }).code;

describe('WGSL <-> TS layouts', () => {
  it('FrameUniforms matches FRAME_LAYOUT', () => {
    const L = structLayouts(compose('common/frame.wgsl'));
    const cam = L.get('CameraFrame')!;
    const fu = L.get('FrameUniforms')!;
    expect(cam.size).toBe(FRAME_LAYOUT.cameraSize);
    expect(fu.size).toBe(FRAME_LAYOUT.size);
    for (const k of ['cam', 'prevCam', 'resolution', 'invResolution', 'frameIndex', 'seedIndex', 'runSeed', 'flags', 'jitterMode', 'jitter', 'origin', 'exposure', 'time', 'dt', 'sceneDiag'] as const) {
      expect(fu.offsets[k], k).toBe(FRAME_LAYOUT[k]);
    }
  });

  it('DebugParams / DebugBuffer match debug-views.ts', () => {
    const L = structLayouts(compose('debug/debug-common.wgsl'));
    const dp = L.get('DebugParams')!;
    expect(dp.size).toBe(DEBUG_PARAMS_SIZE);
    expect(dp.offsets).toEqual({ mode: 0, tap: 4, probePixel: 8, rangeMin: 16, rangeMax: 20, flags: 24, frame: 28, kind: 32, split: 36, size: 40 });
    const rec = L.get('ProbeRecord')!;
    expect(rec.size).toBe(DEBUG_BUFFER_LAYOUT.probeStride);
    const db = L.get('DebugBuffer')!;
    expect(db.offsets.probe).toBe(DEBUG_BUFFER_LAYOUT.probeOffset);
    expect(db.offsets.aov).toBe(DEBUG_BUFFER_LAYOUT.aovOffset);
  });

  it('BlitParams and OverlayParams sizes match the packers (48 B, 160 B)', () => {
    expect(structLayouts(compose('post/blit.wgsl')).get('BlitParams')!.size).toBe(48);
    const o = structLayouts(compose('debug/overlay.wgsl')).get('OverlayParams')!;
    expect(o.size).toBe(160);
    expect(o.offsets).toMatchObject({ srcSize: 128, dstSize: 136, depthKind: 144, hiddenAlpha: 148, depthEps: 152, hasDepth: 156 });
  });

  it('every app/debug shader composes (all defines resolved)', () => {
    expect(() => compose('post/blit.wgsl')).not.toThrow();
    expect(() => compose('post/resolve.wgsl', { RESOLVE_DEBUG_ENTRY: true, DEBUG_OUT_BINDING: true })).not.toThrow();
    expect(() => compose('debug/overlay.wgsl')).not.toThrow();
    expect(() => compose('app/test-pattern.wgsl', { COLOR_FORMAT: 'rgba32float' })).not.toThrow();
    expect(compose('post/resolve.wgsl')).not.toContain('fn debug_resolve');
  });

  it('WGSL DBG_* ids match the TS registry', () => {
    const code = compose('debug/debug-common.wgsl');
    const ids = new Map([...code.matchAll(/const DBG_(\w+): u32 = (\d+)u;/g)].map((m) => [m[1], Number(m[2])]));
    for (const [k, v] of Object.entries(DBG)) expect(ids.get(k), k).toBe(v);
  });
});

describe('frame uniforms packer', () => {
  it('recentres in f64 and stores the rigid inverse', () => {
    const m = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1000.5, 2.25, -3000.125, 1];
    const buf = packFrameUniforms({
      camera: { camToWorld: m, yfov: 0.7 }, prevCamera: { camToWorld: m, yfov: 0.7 },
      width: 960, height: 540, frameIndex: 7, seedIndex: 9, runSeed: 0xdeadbeef, flags: 0, jitterMode: 1,
      jitter: [0.5, 0.5], origin: [1000, 2, -3000], exposure: 1, time: 0, dt: 0, sceneDiag: 10,
    });
    const f = new Float32Array(buf);
    const u = new Uint32Array(buf);
    expect([...f.slice(12, 15)]).toEqual([0.5, 0.25, -0.125]);
    expect([...f.slice(16 + 12, 16 + 15)]).toEqual([-0.5, -0.25, 0.125]);
    expect(f[33]).toBeCloseTo(Math.tan(0.35), 6);
    expect(f[34]).toBeCloseTo(960 / 540, 6);
    expect(u[FRAME_LAYOUT.resolution / 4]).toBe(960);
    expect(u[FRAME_LAYOUT.runSeed / 4]).toBe(0xdeadbeef);
    expect(u[FRAME_LAYOUT.flags / 4] & FRAME_CAMERA_MOVED).toBe(0);
  });
  it('rigidInverse inverts, recentre strips scale', () => {
    const c = Math.cos(0.3), s = Math.sin(0.3);
    const m = [c, 0, -s, 0, 0, 1, 0, 0, s, 0, c, 0, 1, 2, 3, 1];
    const inv = rigidInverse(m);
    // inv * (m * p) = p for p = (0.1, 0.2, 0.3)
    const p = [0.1, 0.2, 0.3];
    const mp = [0, 1, 2].map((r) => m[r] * p[0] + m[4 + r] * p[1] + m[8 + r] * p[2] + m[12 + r]);
    const back = [0, 1, 2].map((r) => inv[r] * mp[0] + inv[4 + r] * mp[1] + inv[8 + r] * mp[2] + inv[12 + r]);
    back.forEach((x, i) => expect(x).toBeCloseTo(p[i], 12));
    const scaled = m.map((x, i) => (i < 12 ? x * 2 : x));
    const r = recentre(scaled, [1, 2, 3]);
    expect(Math.hypot(r[0], r[1], r[2])).toBeCloseTo(1, 12);
    expect([...r.slice(12, 15)]).toEqual([0, 0, 0]);
  });
  it('R2 jitter stays in [0,1)² and depends on the run seed', () => {
    for (let i = 0; i < 100; i++) for (const v of r2Jitter(i, 12345)) { expect(v).toBeGreaterThanOrEqual(0); expect(v).toBeLessThan(1); }
    expect(r2Jitter(3, 1)).not.toEqual(r2Jitter(3, 2));
  });
});

describe('debug registry', () => {
  it('rejects id/key collisions and bad ids; notifies listeners', () => {
    const r = new DebugViewRegistry();
    expect(r.list().length).toBe(BUILTIN_VIEWS.length);
    let n = 0;
    r.onChange(() => n++);
    r.register({ id: 500, key: 'reservoir.W', label: 'W', group: 'Reservoir', source: 'spatial', kind: 'scalar', tapped: true });
    expect(n).toBe(1);
    expect(() => r.register({ id: 500, key: 'x', label: 'x', group: 'g', source: 's', kind: 'scalar' })).toThrow(/already used/);
    expect(() => r.register({ id: 501, key: 'reservoir.W', label: 'x', group: 'g', source: 's', kind: 'scalar' })).toThrow(/key/);
    expect(() => r.register({ id: 0, key: 'z', label: 'x', group: 'g', source: 's', kind: 'scalar' })).toThrow(/bad id/);
    expect(r.byKey('reservoir.W')?.id).toBe(500);
  });
  it('packs DebugParams flags', () => {
    const s = defaultDebugSettings();
    s.mode = 103; s.tap = 2; s.probePixel = [5, 7]; s.log = true; s.colormap = 'turbo'; s.probeEnabled = true;
    const u = new Uint32Array(packDebugParams(s, 'code', 42, [960, 540]));
    expect([u[0], u[1], u[2], u[3], u[7], u[8], u[10], u[11]]).toEqual([103, 2, 5, 7, 42, 2, 960, 540]);
    expect(debugFlags(s)).toBe(1 | (1 << 1) | 8 | 64);
  });
});
