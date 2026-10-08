// M8 (docs/decisions/m8-perf.md §11): validation mode uploads the env in the smallest exact format
// (ENV_COMPACT_IN_VALIDATION). That is bitwise neutral only while no shader reads texEnv through the hardware filter
// (rgb9e5 / f16 filtering is lower precision, platform-lanes.md Q4): every lookup must be envTexel's textureLoad.
import { describe, expect, it } from 'vitest';
import { readdirSync, readFileSync, statSync } from 'node:fs';
import path from 'node:path';
import { ENV_COMPACT_IN_VALIDATION } from '../../src/core/render/env-gpu.ts';

function wgslFiles(dir: string): string[] {
  return readdirSync(dir).flatMap((n) => { const p = path.join(dir, n); return statSync(p).isDirectory() ? wgslFiles(p) : p.endsWith('.wgsl') ? [p] : []; });
}

describe('env compact formats in validation (M8)', () => {
  it('no shader samples texEnv / sEnv with the hardware filter', () => {
    const offenders: string[] = [];
    for (const f of [...wgslFiles('src/core/shaders'), ...wgslFiles('src/core/render/denoise/shaders')]) {
      const src = readFileSync(f, 'utf8');
      for (const m of src.matchAll(/textureSample\w*\s*\(\s*texEnv|textureGather\w*\s*\([^)]*texEnv|\bsEnv\b(?!:)/g)) {
        const line = src.slice(0, m.index).split('\n').length;
        const text = src.split('\n')[line - 1];
        if (/^\s*(\/\/|@group)/.test(text) || /\b_\s*=\s*sEnv\b/.test(text)) continue;   // comments, the binding declaration, the phony keep-alive
        offenders.push(`${f}:${line}: ${text.trim()}`);
      }
    }
    expect(offenders).toEqual([]);
    expect(ENV_COMPACT_IN_VALIDATION).toBe(true);
  });
});
