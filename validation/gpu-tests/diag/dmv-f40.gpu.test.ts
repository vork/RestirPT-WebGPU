// Diagnostic (not part of any gate; skipped unless VITE_DMV_DIAG is set): where does the per-chain variance of the
// dual-MV rung-3.7 unit ixs_d_camera_256@3.7-dualmv-f40 come from? Runs `full` (+ dualMv) chains on ix-d through frame
// VITE_DMV_FRAME (default 40) and, at that frame, reads the production tState and the final radiance of every member.
// Per pixel category (no q′ / standard q′ / dual q′) × selection (c / p / empty): count, ΣL, ΣL² (L = luminance);
// the per-chain M_disocc sums split by category; and every pixel with L > VITE_DMV_THRESH (default 5) as an outlier
// record. VITE_DMV_OLD=1 / VITE_DMV_CDUAL=c patch the dual-MV confidence cap (see below). node-dawn lane only (writes VITE_DMV_OUT, a JSON file). Run under the GPU lock:
//   VITE_DMV_DIAG=1 VITE_DMV_OUT=/path/out.json VITE_DMV_SEED=840101 VITE_DMV_BATCHES=16 VITE_DMV_DUAL=1 \
//   npx tsx validation/harness/with-gpu-lock.ts dmv-diag -- npx vitest run --project node-dawn validation/gpu-tests/diag/dmv-f40.gpu.test.ts
import { describe, it } from 'vitest';
import { readFileSync, writeFileSync } from 'node:fs';
import path from 'node:path';
import { readBuffer } from '../../../src/core/gpu/readback.ts';
import { createEnvResources } from '../../../src/core/render/env-gpu.ts';
import { computeRenderOrigin, JITTER_IID } from '../../../src/core/render/frame-uniforms.ts';
import { ChainRunner, type ChainSpec } from '../../../src/core/render/restir/chain-runner.ts';
import { RestirKernel } from '../../../src/core/render/restir/kernel.ts';
import { TS_WORDS, TSW } from '../../../src/core/render/restir/layout.ts';
import { restirSettings, type RestirSettings } from '../../../src/core/render/restir/presets.ts';
import { SceneGpu } from '../../../src/core/render/scene-gpu.ts';
import { fetchScenePackage, resolvePackageFrame } from '../../../src/core/scene/scene-package.ts';
import type { LightMode } from '../../../src/core/render/lights-gpu.ts';
import { getTestGpu } from '../device-factory.ts';
import { shaderSources } from '../../../src/core/shaders/index.ts';

const env = (import.meta as unknown as { env: Record<string, string | undefined> }).env;
const ON = !!env.VITE_DMV_DIAG;
const ROOT = process.cwd();

const TS_QVALID = 1, TS_DISOCC = 2, TS_SEL_P = 16, TS_SEL_C = 32, TS_EMPTY_OUT = 256, TS_BG = 16384, TS_DUAL_PICK = 65536;

describe.skipIf(!ON)('diag: dual-MV variance at ix-d f40', () => {
  it('chains', async () => {
    // VITE_DMV_CDUAL=c: the dual-MV confidence cap DMV_C_CAP (tpick.wgsl) set to c; VITE_DMV_OLD=1: no cap (the
    // pre-DMV-1 estimator, c_p = min(cCap, c_prev) on dual picks too)
    const cdual = env.VITE_DMV_OLD ? 1e9 : env.VITE_DMV_CDUAL ? Number(env.VITE_DMV_CDUAL) : undefined;
    if (cdual !== undefined) {
      const k = 'restir/tpick.wgsl', old = 'const DMV_C_CAP: f32 = 1.0;';
      if (!shaderSources[k].includes(old)) throw new Error('DMV_C_CAP patch');
      shaderSources[k] = shaderSources[k].replace(old, `const DMV_C_CAP: f32 = ${cdual.toExponential()};`);
    }
    const ctx = await getTestGpu();
    const { device, features, wgslLanguageFeatures } = ctx;
    const pkgDir = env.VITE_DMV_PKG ?? 'validation/scenes/ixs_d_camera_256';
    const p = await fetchScenePackage(`${pkgDir}/`, async (u) => new Uint8Array(readFileSync(path.join(ROOT, u))));
    const W = p.render.width, H = p.render.height, E = 16;
    const frame = Number(env.VITE_DMV_FRAME ?? 40);
    const seed = Number(env.VITE_DMV_SEED ?? 840101);
    const batches = Number(env.VITE_DMV_BATCHES ?? 4);
    const batchOffset = Number(env.VITE_DMV_BATCH_OFFSET ?? 0);
    const thresh = Number(env.VITE_DMV_THRESH ?? 5);
    const extra = env.VITE_DMV_SETTINGS ? JSON.parse(env.VITE_DMV_SETTINGS) as Partial<RestirSettings> : {};
    const settings = restirSettings('full', { maxBounces: p.render.maxBounces ?? 3, dualMv: env.VITE_DMV_DUAL !== '0', ...extra });
    const masksDir = path.join(ROOT, `validation/out/m5/masks/ixs_d_camera_256/gate/f${frame}`);
    let maskBits: Uint16Array | undefined;
    try { const b = readFileSync(path.join(masksDir, 'masks.bin')); maskBits = new Uint16Array(b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength)); } catch { /* no masks */ }

    const origin = computeRenderOrigin(p.scene.bounds, p.scene.quant);
    const gpu = await SceneGpu.create(device, p.scene, origin, { textureMode: 'validation', watertight: true, features, wgslLanguageFeatures });
    const envRes = await createEnvResources(device, p.scene.env);
    const kernel = await RestirKernel.create(device, gpu, envRes, { settings, lightMode: (p.lightMode ?? 'A') as LightMode, env: { nee: false }, features, wgslLanguageFeatures });
    const f0 = resolvePackageFrame(p, 0);
    kernel.setView({ camera: { camToWorld: Array.from(f0.camera.camToWorld), yfov: f0.camera.yfov }, width: W, height: H, runSeed: seed >>> 0, members: E, jitterMode: JITTER_IID, jitter: [0.5, 0.5] });
    await kernel.prepare();
    const runner = new ChainRunner(kernel, { runSeed: seed >>> 0 });
    const a = kernel.resources.alloc;

    // category index: cat = pick (0 none, 1 standard, 2 dual) * 4 + sel (0 none, 1 c, 2 p, 3 empty)
    const NC = 12;
    const cnt = new Float64Array(NC), sL = new Float64Array(NC), sL2 = new Float64Array(NC);
    const cntD = new Float64Array(NC), sLD = new Float64Array(NC), sL2D = new Float64Array(NC);   // inside M_disocc
    const chainRows: number[][] = [];   // per chain: M_disocc ΣL per pick class (3), count per pick class (3)
    const outliers: number[][] = [];
    const invStats = Array.from({ length: 3 }, () => new Array<number>(16).fill(0));   // per pick class: T4 code (INV_DONE pixels)
    const fwdStats = Array.from({ length: 3 }, () => new Array<number>(16).fill(0));   // per pick class: forward code (QVALID)
    const transfer = [0, 0, 0];   // per pick class: Σ L over s = c pixels whose π_p(X_c) = 0 while w̃_p > 0 (the Σw̃ hand-over)
    const pickMap = new Float64Array(W * H * 3);   // per member-local pixel: count of no / std / dual picks
    const lumByPick = new Float64Array(W * H * 3);
    const resolved = new Map<number, ReturnType<typeof resolvePackageFrame>>();
    const frameOf = (t: number) => { let r = resolved.get(t); if (!r) { r = resolvePackageFrame(p, t); resolved.set(t, r); } return r; };
    let curBatch = 0;
    const spec: ChainSpec = {
      frames: frame + 1, testFrames: [],
      state: (t) => { const r = frameOf(t); return { t, camera: { camToWorld: Array.from(r.camera.camToWorld), yfov: r.camera.yfov }, lights: r.lights }; },
      afterFrame: async (t, k) => {
        if (t !== frame) return;
        const ts = await k.readTemporalState();
        const bytes = a.atlasW * a.atlasH * 16;
        const st = device.createBuffer({ size: bytes, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC });
        const enc = device.createCommandEncoder();
        enc.copyTextureToBuffer({ texture: k.resources.frameTex }, { buffer: st, bytesPerRow: a.atlasW * 16, rowsPerImage: a.atlasH }, [a.atlasW, a.atlasH]);
        device.queue.submit([enc.finish()]);
        const img = new Float32Array(await readBuffer(device, st, bytes));
        st.destroy();
        const tf = new Float32Array(ts.buffer, ts.byteOffset, ts.length);
        for (let m = 0; m < E; m++) {
          const ox = (m % a.memberCols) * W, oy = Math.floor(m / a.memberCols) * H;
          const row = [0, 0, 0, 0, 0, 0];
          for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
            const ai = (oy + y) * a.atlasW + ox + x;
            const L = 0.2126 * img[4 * ai] + 0.7152 * img[4 * ai + 1] + 0.0722 * img[4 * ai + 2];
            const o = TS_WORDS * ai;
            const f = ts[o + TSW.flags];
            if (f & TS_BG) continue;
            const pick = (f & TS_QVALID) ? ((f & TS_DUAL_PICK) ? 2 : 1) : 0;
            const sel = (f & TS_EMPTY_OUT) ? 3 : (f & TS_SEL_P) ? 2 : (f & TS_SEL_C) ? 1 : 0;
            const c = pick * 4 + sel;
            cnt[c]++; sL[c] += L; sL2[c] += L * L;
            if (f & TS_QVALID) fwdStats[pick][ts[o + TSW.fwdCode] & 0xff]++;
            if (f & 128) invStats[pick][ts[o + TSW.invCode] & 0xff]++;
            if ((f & TS_SEL_C) && (f & 128) && tf[o + TSW.piRecomp] === 0 && tf[o + TSW.wp] > 0) transfer[pick] += L;
            const li = y * W + x;
            pickMap[3 * li + pick]++; lumByPick[3 * li + pick] += L;
            const inD = maskBits ? (maskBits[li] & 1) !== 0 : false;
            if (inD) { cntD[c]++; sLD[c] += L; sL2D[c] += L * L; row[pick] += L; row[3 + pick]++; }
            if (L > thresh) {
              const qP = ts[o + TSW.qPrime];
              const qx = qP === 0xFFFFFFFF ? -1 : (qP % a.atlasW) - ox, qy = qP === 0xFFFFFFFF ? -1 : Math.floor(qP / a.atlasW) - oy;
              outliers.push([curBatch * E + m, x, y, L, f, tf[o + TSW.cP], tf[o + TSW.cPrev], tf[o + TSW.wc], tf[o + TSW.wp], tf[o + TSW.fwdJ], tf[o + TSW.jP],
                tf[o + TSW.piStored], tf[o + TSW.piRecomp], ts[o + TSW.invCode], ts[o + TSW.fwdCode], qx, qy, inD ? 1 : 0,
                tf[o + TSW.fwdF], tf[o + TSW.fwdF + 1], tf[o + TSW.fwdF + 2]]);
            }
          }
          chainRows.push(row);
        }
      },
    };
    const t0 = performance.now();
    for (let b = batchOffset; b < batchOffset + batches; b++) {
      curBatch = b;
      await runner.runBatch(spec, b, 0, () => undefined);
    }
    const out = {
      seed, frame, batches, batchOffset, E, settings, thresh, ms: performance.now() - t0,
      cats: { cnt: [...cnt], sL: [...sL], sL2: [...sL2] }, catsDisocc: { cnt: [...cntD], sL: [...sLD], sL2: [...sL2D] },
      chainRows, outliers, invStats, fwdStats, transfer, rsc: runner.totals.rsc, errors: runner.invariantErrors(),
      outlierFields: ['chain', 'x', 'y', 'L', 'flags', 'cP', 'cPrev', 'wc', 'wp', 'fwdJ', 'jP', 'piStored', 'piRecomp', 'invCode', 'fwdCode', 'qx', 'qy', 'inDisocc', 'fwdF.r', 'fwdF.g', 'fwdF.b'],
    };
    writeFileSync(env.VITE_DMV_OUT ?? path.join(ROOT, 'dmv-diag.json'), JSON.stringify(out));
    if (env.VITE_DMV_MAPS) writeFileSync(env.VITE_DMV_MAPS, Buffer.from(new Float64Array([...pickMap, ...lumByPick]).buffer));
    console.log(`[dmv-diag] ${batches * E} chains in ${((performance.now() - t0) / 1000).toFixed(1)} s, outliers ${outliers.length}, errors ${out.errors.length}`);
    runner.destroy(); kernel.destroy(); gpu.destroy();
  }, 3_600_000);
});
