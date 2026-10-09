// perf2 WP-7g (perf2-plan.md §2 WP-7g): the present's tonemap-once path (post/tonemap.wgsl + the 9-tap bicubic and
// the 1:1 fast path of blit.wgsl) against the M8 per-tap blit: every canvas pixel within 1 LSB (8-bit target), for every
// view transform and filter, at 1:1 and at non-integer up- and downscales, with the debug split view and a non-finite
// texel highlighted. App only (the present never runs in validation).
import { afterAll, describe, expect, it } from 'vitest';
import { getTestGpu, releaseTestGpu } from './device-factory.ts';
import { Presenter, type PresentSettings, type Tonemap, type UpscaleFilter } from '../../src/core/render/present.ts';

afterAll(async () => { await releaseTestGpu(); });

const SW = 96, SH = 54;

describe('U-WP7-PRESENT: tonemap once ≡ the per-tap blit within 1 LSB', () => {
  it('every view transform × filter × scale; split debug view; highlighted non-finite texel', async () => {
    const { device } = await getTestGpu();
    const canvas = document.createElement('canvas');
    canvas.width = 8; canvas.height = 8;
    const pr = new Presenter(device, canvas);
    await pr.init();
    // HDR test image: smooth gradients, hard edges, fireflies (up to 2^8), a NaN texel
    const px = new Float32Array(SW * SH * 4);
    let s = 99;
    const rnd = () => { s = (Math.imul(s, 1103515245) + 12345) >>> 0; return s / 2 ** 32; };
    for (let y = 0; y < SH; y++) for (let x = 0; x < SW; x++) {
      const i = 4 * (y * SW + x);
      const edge = (x > 40 && x < 60) || (y > 20 && y < 24) ? 4 : 0.05;
      px[i] = edge * (x / SW) * 2; px[i + 1] = edge * (y / SH); px[i + 2] = 0.3 * rnd();
      if (rnd() < 0.02) px[i] = px[i + 1] = px[i + 2] = 2 ** (8 * rnd());
      px[i + 3] = 1;
    }
    px[4 * (10 * SW + 10)] = NaN;
    const half = new Uint16Array(px.length);
    const f16 = new Float16Array(half.buffer);
    for (let i = 0; i < px.length; i++) f16[i] = px[i];
    const colour = device.createTexture({ size: [SW, SH], format: 'rgba16float', usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST });
    device.queue.writeTexture({ texture: colour }, half, { bytesPerRow: SW * 8 }, [SW, SH]);
    const dbgTex = device.createTexture({ size: [SW, SH], format: 'rgba32float', usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST });
    const dpx = new Float32Array(SW * SH * 4).map((_, i) => ((i * 7919) % 101) / 100);
    device.queue.writeTexture({ texture: dbgTex }, dpx, { bytesPerRow: SW * 16 }, [SW, SH]);

    const render = async (dw: number, dh: number, st: PresentSettings, split: boolean): Promise<Uint8Array> => {
      const target = device.createTexture({ size: [dw, dh], format: pr.format, usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC });
      const enc = device.createCommandEncoder();
      pr.encode(enc, { color: colour, debugOut: dbgTex }, st, { active: split, split, splitPos: 0.37, probe: false, probePixel: [0, 0] }, undefined, undefined, { target });
      const bpr = Math.ceil((dw * 4) / 256) * 256;
      const buf = device.createBuffer({ size: bpr * dh, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
      enc.copyTextureToBuffer({ texture: target }, { buffer: buf, bytesPerRow: bpr }, [dw, dh]);
      device.queue.submit([enc.finish()]);
      await buf.mapAsync(GPUMapMode.READ);
      const src = new Uint8Array(buf.getMappedRange());
      const out = new Uint8Array(dw * dh * 4);
      for (let y = 0; y < dh; y++) out.set(src.subarray(y * bpr, y * bpr + dw * 4), y * dw * 4);
      buf.unmap(); buf.destroy(); target.destroy();
      return out;
    };

    const sizes: [number, number][] = [[SW, SH], [229, 131], [61, 37]];
    const report: string[] = [];
    for (const tonemap of ['standard', 'agx', 'aces', 'raw'] as Tonemap[]) for (const filter of ['nearest', 'bilinear', 'bicubic'] as UpscaleFilter[]) {
      for (const [dw, dh] of sizes) for (const split of [false, true]) {
        const st: PresentSettings = { exposureEV: 0.5, tonemap, filter, highlightNonFinite: true };
        const a = await render(dw, dh, { ...st, pretone: false }, split);
        const b = await render(dw, dh, { ...st, pretone: true }, split);
        let max = 0, n1 = 0;
        for (let i = 0; i < a.length; i++) { const d = Math.abs(a[i] - b[i]); if (d > max) max = d; if (d === 1) n1++; }
        report.push(`${tonemap}/${filter}/${dw}x${dh}${split ? '/split' : ''}: max ${max} LSB, ${n1} channels at 1`);
        expect(max, `${tonemap} ${filter} ${dw}x${dh} split ${split}`).toBeLessThanOrEqual(1);
        if (dw === SW && dh === SH) expect(max, `1:1 ${tonemap} ${filter}`).toBeLessThanOrEqual(1);
      }
    }
    console.log(`[U-WP7-PRESENT]\n${report.join('\n')}`);
    pr.destroy(); colour.destroy(); dbgTex.destroy();
  });
});
