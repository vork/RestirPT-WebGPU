// Fetch the CC0 Poly Haven 1k HDRIs used by validation (plan §7.2, ENV-U1) with pinned SHA-256.
//
//   npx tsx validation/assets/fetch_hdris.ts [--out validation/assets/downloaded/hdri] [--pin]
//
// Files land in validation/assets/downloaded/hdri/ (gitignored). Hashes are pinned in validation/assets/hdris.json:
// a missing entry is recorded on first download (TOFU); an existing entry must match or the script fails.
// --pin rewrites the pins from the current downloads (only after a deliberate upstream change).
// License: CC0 (https://polyhaven.com/license).
import { createHash } from 'node:crypto';
import { existsSync, mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { dirname, join, relative, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

export const HDRI_IDS = ['overcast_soil_puresky', 'studio_small_09', 'kloofendal_48d_partly_cloudy_puresky'] as const;
export type HdriId = (typeof HDRI_IDS)[number];
export type HdriExt = 'hdr' | 'exr';

export interface HdriPin { id: string; ext: HdriExt; file: string; url: string; bytes: number; sha256: string }
export interface HdriManifest { source: string; license: string; resolution: string; files: HdriPin[] }

const here = dirname(fileURLToPath(import.meta.url));
export const ROOT = resolve(here, '../..');
export const MANIFEST_PATH = join(here, 'hdris.json');
export const DEFAULT_OUT = join(here, 'downloaded/hdri');

export const hdriUrl = (id: string, ext: HdriExt) => `https://dl.polyhaven.org/file/ph-assets/HDRIs/${ext}/1k/${id}_1k.${ext}`;
export const hdriPath = (id: string, ext: HdriExt, out = DEFAULT_OUT) => join(out, `${id}_1k.${ext}`);

const sha256 = (b: Uint8Array) => createHash('sha256').update(b).digest('hex');

export function readManifest(): HdriManifest {
  if (existsSync(MANIFEST_PATH)) return JSON.parse(readFileSync(MANIFEST_PATH, 'utf8')) as HdriManifest;
  return { source: 'https://polyhaven.com', license: 'CC0', resolution: '1k', files: [] };
}

/** True if the file exists and matches its pinned hash (used by tests to decide whether to skip). */
export function verifyLocal(id: string, ext: HdriExt, out = DEFAULT_OUT): boolean {
  const p = hdriPath(id, ext, out);
  const pin = readManifest().files.find((f) => f.id === id && f.ext === ext);
  return !!pin && existsSync(p) && sha256(readFileSync(p)) === pin.sha256;
}

async function download(url: string): Promise<Uint8Array> {
  const res = await fetch(url, { headers: { 'User-Agent': 'restirpt-validation' } });
  if (!res.ok) throw new Error(`${url}: HTTP ${res.status}`);
  return new Uint8Array(await res.arrayBuffer());
}

async function main(): Promise<number> {
  const args = process.argv.slice(2);
  const outIdx = args.indexOf('--out');
  const out = outIdx >= 0 ? resolve(args[outIdx + 1]) : DEFAULT_OUT;
  const repin = args.includes('--pin');
  mkdirSync(out, { recursive: true });
  const manifest = readManifest();
  let failures = 0;
  let changed = false;
  for (const id of HDRI_IDS) {
    for (const ext of ['hdr', 'exr'] as const) {
      const url = hdriUrl(id, ext);
      const path = hdriPath(id, ext, out);
      const rel = relative(ROOT, path);
      const pin = manifest.files.find((f) => f.id === id && f.ext === ext);
      let data: Uint8Array | undefined = existsSync(path) ? new Uint8Array(readFileSync(path)) : undefined;
      if (data && pin && !repin && sha256(data) === pin.sha256) { console.log(`ok (cached)  ${rel}`); continue; }
      data = await download(url);
      const hash = sha256(data);
      if (pin && !repin && hash !== pin.sha256) {
        console.error(`HASH MISMATCH ${rel}: got ${hash}, pinned ${pin.sha256} (upstream changed? rerun with --pin deliberately)`);
        failures++;
        continue;
      }
      writeFileSync(path, data);
      const entry: HdriPin = { id, ext, file: `${id}_1k.${ext}`, url, bytes: data.length, sha256: hash };
      if (!pin) { manifest.files.push(entry); changed = true; console.log(`pinned      ${rel} ${hash}`); }
      else if (repin) { Object.assign(pin, entry); changed = true; console.log(`re-pinned   ${rel} ${hash}`); }
      else console.log(`ok          ${rel}`);
    }
  }
  if (changed) {
    manifest.files.sort((a, b) => a.id.localeCompare(b.id) || a.ext.localeCompare(b.ext));
    writeFileSync(MANIFEST_PATH, JSON.stringify(manifest, null, 2) + '\n');
  }
  return failures ? 1 : 0;
}

if (process.argv[1] && resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  main().then((code) => process.exit(code), (e) => { console.error(e); process.exit(1); });
}
