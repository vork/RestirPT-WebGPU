// M7 U-TAN-B (docs/decisions/m7-api.md §6.2): our package tangents (readScenePackage → tangents.ts: whole-mesh MikkTSpace,
// Rust port, oct 2 × 15 + sign) vs Blender 5.2.2's own MikkTSpace (Mesh.calc_tangents on the package mesh built by
// build_scene.py), per corner, on every normal-mapped package given. Gate: bitangent sign equal on every corner whose
// material has a normal map; tangent angle ≤ 0.01° at the 99.9th percentile and ≤ 0.05° max (oct-15 snap ≤ 0.0049°
// + the two MikkTSpace implementations); the CPU-only check needs no GPU (no lock).
//   npx tsx validation/harness/m7-tangent-check.ts [--out DIR] PKG_DIR...
import { spawnSync } from 'node:child_process';
import { mkdirSync, readFileSync, readdirSync, writeFileSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { readScenePackage } from '../../src/core/scene/scene-package.ts';

const ROOT = fileURLToPath(new URL('../../', import.meta.url));
const BLENDER = process.env.BLENDER ?? '/Applications/Blender.app/Contents/MacOS/Blender';
export const TAN_GATE = { maxDeg: 0.05, p999Deg: 0.01 } as const;

export interface TangentCheck { pkg: string; corners: number; nmCorners: number; signMismatch: number; maxDeg: number; p999Deg: number; meanDeg: number; ok: boolean; worst?: string }

export async function tangentCheck(pkgDir: string, outDir: string): Promise<TangentCheck> {
  const abs = path.resolve(ROOT, pkgDir);
  const bin = path.join(outDir, `${path.basename(abs)}.tan.bin`);
  const r = spawnSync(BLENDER, ['-b', '--factory-startup', '--python-exit-code', '1', '-P', path.join(ROOT, 'validation/blender/tangent_dump.py'), '--', '--package', abs, '--out', bin], { encoding: 'utf8', maxBuffer: 1 << 26 });
  if (r.status !== 0) throw new Error(`tangent_dump.py failed: ${(r.stdout ?? '') + (r.stderr ?? '')}`.slice(-800));
  const raw = readFileSync(bin);
  const bl = new Float32Array(raw.buffer, raw.byteOffset, raw.byteLength / 4);
  const files = new Map(readdirSync(abs).map((n) => [n, new Uint8Array(readFileSync(path.join(abs, n)))]));
  const { scene } = await readScenePackage(files);
  const g = scene.geometry;
  const nC = g.indices.length;
  if (bl.length !== nC * 4) throw new Error(`${pkgDir}: Blender has ${bl.length / 4} loops, the package ${nC} corners`);
  const angles: number[] = [];
  let signMismatch = 0, nm = 0, worst = '';
  let maxDeg = 0;
  for (let c = 0; c < nC; c++) {
    if (!scene.materials[g.triMaterial[Math.floor(c / 3)]].normalTexture) continue;
    nm++;
    const v = g.indices[c];
    const a = [g.tangents[4 * v], g.tangents[4 * v + 1], g.tangents[4 * v + 2]], b = [bl[4 * c], bl[4 * c + 1], bl[4 * c + 2]];
    if (Math.sign(g.tangents[4 * v + 3]) !== Math.sign(bl[4 * c + 3])) signMismatch++;
    const cr = Math.hypot(a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]);
    const d = Math.atan2(cr, a[0] * b[0] + a[1] * b[1] + a[2] * b[2]) * 180 / Math.PI;
    angles.push(d);
    if (d > maxDeg) { maxDeg = d; worst = `corner ${c} (prim ${Math.floor(c / 3)}): ours (${a.map((x) => x.toFixed(6))}) blender (${b.map((x) => x.toFixed(6))})`; }
  }
  angles.sort((x, y) => x - y);
  const p999Deg = angles.length ? angles[Math.min(angles.length - 1, Math.floor(0.999 * angles.length))] : 0;
  const meanDeg = angles.reduce((s, x) => s + x, 0) / Math.max(1, angles.length);
  const ok = nm > 0 && signMismatch === 0 && maxDeg <= TAN_GATE.maxDeg && p999Deg <= TAN_GATE.p999Deg;
  return { pkg: pkgDir, corners: nC, nmCorners: nm, signMismatch, maxDeg, p999Deg, meanDeg, ok, worst };
}

if (process.argv[1] && import.meta.url === new URL(`file://${path.resolve(process.argv[1])}`).href) {
  const args = process.argv.slice(2);
  const oi = args.indexOf('--out');
  const out = oi >= 0 ? path.resolve(args[oi + 1]) : path.join(ROOT, 'validation/out/m7/tangent-check');
  const pkgs = args.filter((x, i) => !(i === oi || i === oi + 1));
  mkdirSync(out, { recursive: true });
  (async () => {
    const res: TangentCheck[] = [];
    for (const p of pkgs) {
      const r = await tangentCheck(p, out);
      res.push(r);
      console.log(`${r.ok ? 'PASS' : 'FAIL'}  U-TAN-B ${p}: ${r.nmCorners} normal-mapped corners, sign mismatches ${r.signMismatch}, max ${r.maxDeg.toExponential(3)}°, p99.9 ${r.p999Deg.toExponential(3)}°, mean ${r.meanDeg.toExponential(3)}°${r.ok ? '' : `; worst ${r.worst}`}`);
    }
    writeFileSync(path.join(out, 'report.json'), `${JSON.stringify({ gate: TAN_GATE, results: res }, null, 1)}\n`);
    process.exit(res.every((r) => r.ok) ? 0 : 1);
  })().catch((e) => { console.error(e); process.exit(2); });
}
