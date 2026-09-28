// M0 IO orientation test (plan §5): TS PFM encoder -> Python reader must agree on row 0 = top.
import { spawnSync } from 'node:child_process';
import { existsSync, mkdirSync, writeFileSync } from 'node:fs';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { describe, expect, it } from 'vitest';
import { encodePFM, orientationPattern, type FloatImage } from '../src/core/io/pfm.ts';

const root = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const python = join(root, 'validation/.venv/bin/python');
const script = join(root, 'validation/tools/orientation_check.py');
const outDir = join(root, 'validation/out/test');
const hasVenv = existsSync(python);
if (!hasVenv) console.warn(`[pfm-python-roundtrip] skipped: ${python} missing; create it per validation/requirements.txt`);

function check(img: FloatImage, name: string) {
  mkdirSync(outDir, { recursive: true });
  const path = join(outDir, name);
  writeFileSync(path, encodePFM(img));
  return spawnSync(python, [script, 'pfm', path], { encoding: 'utf8' });
}

function flipV(img: FloatImage): FloatImage {
  const { width: w, height: h, channels: c, data } = img;
  const out = new Float32Array(data.length);
  for (let r = 0; r < h; r++) out.set(data.subarray((h - 1 - r) * w * c, (h - r) * w * c), r * w * c);
  return { ...img, data: out };
}

describe.skipIf(!hasVenv)('pfm python round trip (validation/.venv)', () => {
  it('python reads the TS-encoded orientation pattern with row 0 = top', () => {
    const res = check(orientationPattern(64, 48), 'orientation.pfm');
    expect(res.status, res.stdout + res.stderr).toBe(0);
    expect(res.stdout).toContain('OK');
  });

  it('negative control: a vertically flipped pattern fails', () => {
    const res = check(flipV(orientationPattern(64, 48)), 'orientation-flipped.pfm');
    expect(res.status, res.stdout + res.stderr).toBe(1);
    expect(res.stdout).toContain('vertically flipped');
  });
});
