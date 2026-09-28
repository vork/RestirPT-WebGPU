// Browser-side scene-package export: build the package (src/core/scene/scene-package.ts) and POST its files to the
// dev upload middleware, which writes validation/out/<run>/<file> (upload-middleware.ts). Used by the harness page
// (window.__harness.exportPackage) and the app's dev-only "Export for Cycles" button.
import { sha256Hex } from '../../src/core/io/zlib.ts';
import { exportScenePackage, type ExportScenePackageOptions, type ScenePackage } from '../../src/core/scene/scene-package.ts';
import type { SceneData } from '../../src/core/scene/types.ts';

export async function uploadFile(run: string, name: string, bytes: Uint8Array): Promise<void> {
  const q = new URLSearchParams({ run, name });
  const r = await fetch(`/__harness/upload?${q}`, { method: 'POST', body: bytes as Uint8Array<ArrayBuffer>, headers: { 'content-type': 'application/octet-stream' } });
  if (r.status !== 204) throw new Error(`upload ${name}: HTTP ${r.status} ${await r.text()}`);
}

/** Same definition as validation/blender/build_scene.py package_sha256: sha256 of "name\0sha256(file)\n" sorted by name. */
export async function packageSha256(files: Map<string, Uint8Array>): Promise<{ sha256: string; files: Record<string, string> }> {
  const per: Record<string, string> = {};
  for (const [k, v] of files) per[k] = await sha256Hex(v);
  const text = Object.keys(per).sort().map((k) => `${k}\0${per[k]}\n`).join('');
  return { sha256: await sha256Hex(new TextEncoder().encode(text)), files: per };
}

export interface ExportedPackage { run: string; dir: string; files: string[]; bytes: number; sha256: string; pkg: ScenePackage }

/** Export `scene` and upload every package file under validation/out/<run>/. */
export async function exportAndUpload(scene: SceneData, opts: ExportScenePackageOptions, run: string): Promise<ExportedPackage> {
  const pkg = await exportScenePackage(scene, opts);
  // scene.json last: a reader that sees it can rely on the other files being complete
  const names = [...pkg.files.keys()].sort((a, b) => (a === 'scene.json' ? 1 : b === 'scene.json' ? -1 : a < b ? -1 : 1));
  for (const n of names) await uploadFile(run, n, pkg.files.get(n)!);
  const bytes = names.reduce((s, n) => s + pkg.files.get(n)!.length, 0);
  return { run, dir: `validation/out/${run}`, files: names, bytes, sha256: (await packageSha256(pkg.files)).sha256, pkg };
}
