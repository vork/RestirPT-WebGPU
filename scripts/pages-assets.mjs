// Copies the scenes the GitHub Pages build serves into dist/ under the same paths the dev server uses, so
// ?scene=validation/assets/... and ?env=... work unchanged below the Pages base. Sponza and the HDRI are not in git:
// CI fetches them first (validation/blender/fetch_sponza.py, validation/assets/fetch_hdris.ts).
import { cpSync, existsSync, mkdirSync } from 'node:fs';
import { dirname } from 'node:path';

const ASSETS = [
  'validation/assets/cornell/cornell.glb',
  'validation/assets/cornell/cornell_point_spot.glb',
  'validation/assets/downloaded/sponza',
  'validation/assets/downloaded/hdri/kloofendal_48d_partly_cloudy_puresky_1k.hdr',
];
let missing = 0;
for (const a of ASSETS) {
  if (!existsSync(a)) { console.warn(`pages-assets: missing ${a} (skipped)`); missing++; continue; }
  mkdirSync(dirname(`dist/${a}`), { recursive: true });
  cpSync(a, `dist/${a}`, { recursive: true });
  console.log(`pages-assets: ${a}`);
}
if (missing && process.env.CI) process.exit(1);
