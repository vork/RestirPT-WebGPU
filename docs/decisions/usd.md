# USD loading: LightUSD rc4 spike result (M0)

Plan §5 M0 "LightUSD rc4 spike", §1.2 USD light rules, `docs/research/scene-io.md` §5–6.
Run 2026-09-28: `lightusd@1.0.0-rc4`, HeadlessChrome/154 via Playwright `channel: 'chrome'`, Apple M5 Pro,
Blender 5.1.2 exporter, reference = OpenUSD `pxr` 0.25.8 (Blender's bundled Python).

## Decision: GO, with conditions

Use **LightUSD rc4, `next` backend (`lightusd_next.wasm`, `RenderStream`), called from our own module Worker**.
Put `UsdSceneSource` in front of it, and have the adapter apply the six workarounds listed under
"Adapter contract" below.

The conditions:

- **Do not use the stock JS loaders** (`LightUSDLoader`, `LightUSDWorkerLoader`, `NextRenderSceneAdapter`). In rc4 they are broken or awkward for us (findings 1–3).
- **Pin the exact version.** `package.json` currently has `^1.0.0-rc4`, which allows rc5 or 1.0.0. Change it to exact `1.0.0-rc4` (`npm install --save-exact`). The spike harness is the regression test for any upgrade.
- Fallback order:
  - **Fallback-0 (in-package):** the `legacy` backend, `lightusd.wasm`, 6.9 MB. Use it only if `next` regresses. It has complete light and material fields, but no PointInstancer, no instanceable prims through the worker, and snorm8 normals.
  - **Fallback-1 (three.js `USDLoader`, r186) is rejected.** It fails on every USDC/USDZ file Blender writes. It loses spot lights, turns DiskLight into a square rect, converts colours as sRGB, and ignores `exposure`, `normalize` and colour temperature. See the table.
  - **Fallback-2 (offline Blender conversion)** stays valid. It is also what the E2E-USD Cycles reference uses anyway (§7.2).

With the adapter, the result matched the pxr dump on every compared field: all light, draw and PointInstancer fields in all 7 test files, and material fields except 4 material records per instanced Blender file (condition 4).

## What was run

| Item | Path |
|---|---|
| Blender test scene. Two boxes (one with 2 materials → GeomSubset), plane, linked duplicate, collection instances (`use_instancing=True`), POINT r=0, POINT r=0.25 + exposure 1, SPOT 60°/0.15, AREA RECT 1×0.5, AREA DISK (normalize off), SUN 2°. Exported to .usda/.usdc/.usdz, plus a Y-up .usda. Also writes `*.truth.json`, and `perf.usdc` (655k tris) with `--perf-out`. | `validation/spikes/usd/make_usd.py` |
| Hand-authored test file (+ .usdc/.usdz via pxr). upAxis Y, metersPerUnit 0.01, RectLight normalize=1 with exposure 1, DiskLight normalize=0, SphereLight r=0.5 with colorTemperature 3000, spot with treatAsPoint + ShapingAPI angle/softness/focus/focusTint, spot r=5, treatAsPoint with radius left at default, DistantLight 0.53°, GeomSubset partition, `xformOp:transform`/`orient`/parent scale, PointInstancer with 3 instances (quath orientations, non-uniform scale, rotated and translated parent) | `validation/assets/usd-spike/spike_hand.usda` |
| pxr reference dump (UsdImaging conventions: instance proxies expanded, class and PointInstancer-prototype subtrees not drawn) | `validation/spikes/usd/pxr_dump.py` |
| Spike page. Modes: `direct` (own worker), `next` and `legacy` (stock `LightUSDWorkerLoader`), `three` | `validation/spikes/usd/{index.html,spike.ts,usd-direct.worker.ts,normalize.ts,usda-scan.ts}` |
| Driver. Pass A: `npx vite validation/spikes/usd --port 5190`, default config, modes direct+three. Pass B: `--config vite.lightusd-stock.config.ts`, modes next+legacy. Then compares against pxr. | `npx tsx validation/spikes/usd/run-spike.ts` |
| Output: per-loader dumps, `compare.json`, `summary.json` | `validation/out/usd-spike/` |

## Evidence

### Field coverage vs pxr

Counts are summed over the 7 small files (+ `perf.usdc`) as ok/mismatch/missing. The `+adapter` column is the `next` backend with the adapter rules below applied.

| Field | next (`direct`) raw | next + adapter | legacy (stock worker) | three r186 |
|---|---|---|---|---|
| files loaded | 8/8 | 8/8 | 8/8 | **3/8** (every USDC/USDZ fails: `double3 xformOp:translate` → "Unsupported scalar type 35", then a `makeTranslation` TypeError) |
| stage upAxis / metersPerUnit | ok | ok | ok | ok (applied to the root group) |
| root-layer `doc` ("Blender v5.1.2") | **not exposed** | ok (layer scan) | not exposed | — |
| light type | 46/0/0 | 46/0/0 | 46/0/0 | 16/3/0 (DiskLight → square RectAreaLight) |
| intensity, color, width, height | all ok | all ok | all ok | ok (raw `inputs:intensity`) |
| exposure, normalize, enableColorTemperature, colorTemperature | all ok | all ok | all ok | **missing** |
| radius | 36/0/**10** (dropped on `spot`) | 46/0/0 | 46/0/0 | missing |
| treatAsPoint | **missing** on all SphereLights | ok (layer scan) | **missing** | missing |
| DistantLight angle | ok (degrees) | ok | ok | missing |
| shaping cone angle / softness / focus / focusTint | ok. **The cone angle comes back as `angle` in radians** | ok | ok. The ShapingAPI-applied flag is not exposed (defaults 90/0/0 are always reported) | **all missing**: every spot becomes a PointLight (it reads `shaping:cone:angle` without the `inputs:` prefix) |
| light world transform | 46/0/0 | 46/0/0 | 46/0/0 | 17/2/0 (DirectionalLight position rewritten; three ignores rotation for spot/directional) |
| draws matched (instance proxies expanded) | 34 of 38. Misses every instanceable instance except the first, and adds the `class` prim's mesh and the PointInstancer prototype meshes as extra draws | **38/38**, no extra draws | 30 of 38. No instanceable instances, draws the `class` prim, no PointInstancer | 10 of 38 |
| triangles / world / local bbox per draw | ok | ok | ok | ok (when loaded) |
| GeomSubset material assignment (tris per material) | ok for matched draws | 38/38 | 30/30 | 0/10 (colours sRGB→linear converted, e.g. 0.8 → 0.604; unauthored roughness 1.0 instead of 0.5) |
| PreviewSurface diffuse/emissive/metallic/roughness/opacity | ok | ok | ok | converted (see above) |
| opacityThreshold | only when > 0 | ok | ok | — |
| ior, clearcoat, clearcoatRoughness, useSpecularWorkflow, specularColor | **missing** | 27/0/4 (the 4 misses are materials composed by reference inside instance prototypes) | ok | — |
| Blender's non-standard `specular` | missing | ok for root-layer shaders | missing | — |
| PointInstancer instances | 3/3 present. The transform is **instancer-relative, with a transposed 3×3** | 3/3 world ok | **not supported** (prototypes drawn in place) | not supported |
| cameras | exposed (not compared) | — | exposed | exposed |

### Sizes, timings, platform

| | next (`lightusd_next.wasm`) | legacy (`lightusd.wasm`) |
|---|---|---|
| wasm | 1,672,421 B (gzip 677 kB, `.zst` 559 kB) | 6,942,962 B (`.zst` 1.45 MB) |
| JS glue | 72 kB (our production worker chunk incl. scanner: 48 kB) | 80 kB, plus `LightUSDLoader.js`/`LightUSDWorker.js` (and a second copy of three.js via the stock loader) |
| wasm init in the worker | 8 ms | 52 ms |
| first result (worker spawn + init + Vite dev transforms) | 257 ms (dev). Production build, Blender .usdz: **33 ms** total in the worker | — |
| small files (6–23 kB) | 2–10 ms end-to-end | 4–52 ms |
| `perf.usdc` (3.9 MB, 8 meshes, 655k tris → 1.97M unwelded verts) | begin 46 ms + extract 191 ms, plus **layer scan 354 ms** = 599 ms | 176 ms |

- **Threads:** no `SharedArrayBuffer`/pthreads in the glue.
- **Isolation:** the page ran with `crossOriginIsolated === false`, and so did a production build served by a plain `python -m http.server`. **No COOP/COEP needed.**
- **License:** Apache-2.0 OR MIT (the npm metadata). The `LICENSE` file ships Apache-2.0 text only.
- **Dependencies:** the package declares a `three >= 0.178` dependency. We do not need it with the direct path.

## Findings (rc4 bugs and quirks, all reproduced by the harness)

1. **The stock `LightUSDWorkerLoader` with `backend: 'next'` always fails.** The error is "Invalid value used as weak map key". `LightUSDWorker.js:612` runs `deepCloneNode` over `node.children`, which are numeric node ids on `next` → `WeakMap.set(1, …)`. The stock worker's default is `legacy`, which works. That path also drops `getInstance()` data, so instanceable prims vanish.
2. **Importing `LightUSDLoader.js` in any form breaks Vite.**
   - It does an undeclared `import('fzstd')` (`LightUSDLoader.js:1655`). The dev server gives HTTP 500 from import-analysis, and `vite build` fails ("Rolldown failed to resolve import fzstd").
   - Dep pre-bundling breaks `new URL('./LightUSDWorker.js', import.meta.url)`.
   - The stock path needs `optimizeDeps.exclude: ['lightusd']` + a `fzstd` alias (`vite.lightusd-stock.config.ts`).
   - It also pulls in a second three.js instance ("Multiple instances of Three.js").
   - The direct worker needs **no Vite config**, in dev or in build.
3. **`next`: `RenderStream.getNode(i).dataId` is not an index into `getMesh()`.**
   - With instancing present it drifts. On the Blender file, 5 of 7 mesh nodes pointed at the wrong mesh.
   - LightUSD's own three builder says to match renderables to nodes **by prim path** (`NextRenderSceneUtils.js:1160-1166`).
4. **`next`: instanceable prims are not instanced.**
   - The shared mesh is emitted once, under the *first* instance's path. The other instances have mesh nodes but no mesh.
   - `class` prims (Blender writes `class "prototypes"` under `use_instancing=True`) are emitted as real meshes at the origin.
   - The RenderStream options `setFlattenRenderTree`/`setMeshMerge`/`setMaterialDedup` change neither behaviour.
5. **`next` PointInstancer:**
   - `getPointInstanceDraw(i).transform` is relative to the PointInstancer prim (no parent transforms).
   - Its **3×3 (orientation·scale) is transposed**; translation is correct.
   - Prototype meshes are also emitted as normal meshes at their authored place.
6. **`next` lights:**
   - `treatAsPoint` is not exposed (also not by `legacy`).
   - `radius` is omitted for `type: 'spot'` (SphereLight + ShapingAPI).
   - The spot cone is `angle` in **radians**, while DistantLight `angle` is in **degrees** (the angular diameter, as authored).
   - The `shapingConeAngle` key is absent.
   - Types are `sphere | spot | rect | disk | directional` (+ cylinder/dome/geometry per `LIGHT_TYPE` map).
7. **`next` materials drop `ior`, `clearcoat`, `clearcoatRoughness`, `useSpecularWorkflow` and `specularColor`.**
   - `previewSurface` in `materialXJson` carries only diffuseColor/emissiveColor/metallic/roughness/opacity/normal.
   - `opacityThreshold` is only present when > 0.
   - `ior` matters for our glass.
8. **`next` metadata has no root-layer `doc`/comment.** The Blender-quirk detection (scene-io §6) needs it.
9. **The USDA-text workaround is expensive on big binary files.**
   - `NextUSDZConverterNative.exportAsUSDA()` works (usda/usdc/usdz) and exposes everything that findings 6–8 drop.
   - It cost 354 ms on the 3.9 MB `perf.usdc`, 60 % of the load.
   - Mesh extraction is 191 ms there, because meshes come back unwelded when normals are faceVarying (1.97M verts for 328k authored points).
10. **Coordinates and UVs come back raw.**
    - Points are prim-local and `worldMatrix` is USD row-major. Translation is at 12..14 = column-major for column vectors, so a straight copy works.
    - No up-axis or metersPerUnit conversion is applied.
    - UVs are **not** flipped (`primvars:st` as authored), so the adapter applies `v = 1 − v`.
    - Winding is preserved.
    - Normals are computed when not authored.
    - `setBuildVertexIndices(true)` welds identical corners (Plane: 6 → 4 verts).

### Blender 5.1.2 USD export quirks (in addition to scene-io §6)

1. **`inputs:enableColorTemperature = 1` is written on every light, even with `use_temperature = False`.**
   - `colorTemperature` is left unauthored, so it takes the default 6500 K.
   - Blender's own `wm.usd_import` honours it. The re-imported lights get `use_temperature=True` with `temperature_color = (1.0426, 0.9841, 1.0353)`, which is **not white**.
   - pxr `BlackbodyTemperatureAsRgb(6500)` = (1.0433, 0.9836, 1.0346).
   - Because the Cycles reference is rendered from the re-imported file (§8.5), the adapter **must apply the colour temperature** (spec behaviour). For exactness, use Blender's blackbody normalisation. It differs from pxr by < 0.1 %.
2. Sun: `angle` = half the true diameter, and `intensity = energy/4` (confirmed; the round trip restores 2° and 3 W).
3. `use_instancing=True` exports instanceable references to a `class "prototypes"`. Given findings 3–4, **export validation scenes with `use_instancing=False`** (the default) until M7.

### rc4 vs the `dev` binding claimed in scene-io §5.1

That report said the `dev` `getLight()` (`web/binding.cc:4285-4455`) exposes: type, color, intensity, exposure, normalize, colorTemperature, radius, width, height, angle, shaping cone angle/softness/focus, and world transform.

- In rc4, the **legacy** `LightUSDLoaderNative.getLight()` has all of these. It also has `direction`, `position`, `length`, IES fields, shadow fields and `enableColorTemperature`, and reports the shaping angle in degrees.
- The **next** `RenderStream.getLight()` diverges:
  - no `shapingConeAngle` (the spot cone is in `angle`, radians);
  - no `radius` on spot;
  - `width`/`height` only on rect;
  - `radius` only on sphere/disk.
- **Neither backend has `treatAsPoint`**, and neither did the dev list.
- `dev` itself was not re-checked in this spike.

## Adapter contract (`src/core/scene/usd/`, M1)

One module Worker per load. Replies use transfer lists. Exact calls:

```ts
import createLightUSDNext from 'lightusd/lightusd_next.js';      // exports map allows the deep import
import wasmUrl from 'lightusd/lightusd_next.wasm?url';            // Vite asset URL: works in dev and build, no config
const native = await createLightUSDNext({ locateFile: (p) => (p.endsWith('.wasm') ? wasmUrl : p) });

const rs = new native.RenderStream();
rs.setBuildVertexIndices(true);                                    // weld corners (verify per asset; default off)
// .usdz: stored-zip entries; begin() takes the ROOT LAYER bytes (first .usd/.usda/.usdc entry),
// other USD layers via rs.provideAsset(name, bytes) BEFORE begin().
const r = rs.begin(rootLayerBytes);                                // {success, meshCount, nodeCount, lightCount, ...}
if (!r.success) throw new Error(r.error ?? rs.error());
const meta = rs.getSceneMetadata();                                // {upAxis, metersPerUnit, timeCodesPerSecond, ...}
rs.getNode(i)             // i < rs.nodeCount(): {primPath, type:'xform'|'mesh'|'sphereLight'|..., visible, localMatrix, worldMatrix, children:number[], parentId}
rs.getMesh(i)             // i < rs.meshCount(): {primPath, points/indices/normals/uv0: {ptr,length,dtype}, submeshes:[{start,count,materialIndex}],
                          //   material, materials[], worldMatrix, doubleSided}; COPY views out of native.HEAPU8 before the next call
rs.getLight(i)            // i < rs.lightCount(): {primPath, type, intensity, exposure, color, normalize, enableColorTemperature,
                          //   colorTemperature, radius?, width?, height?, angle?, shapingConeSoftness, shapingFocus, shapingFocusTint, transform}
rs.getPointInstancer(i)   // {primPath, prototypePaths, drawStart, drawCount, instanceCount}
rs.getPointInstanceDraw(i)// {pointInstancerId, prototypeIndex, meshPath, materialPath, transform}
rs.getCamera(i); rs.getUnsupportedRenderables(); rs.getStats();
rs.end(); rs.delete();
// Missing-field workaround (finding 9). For .usda, scan the file text directly instead:
const conv = new native.NextUSDZConverterNative(); conv.loadFromBinary(bytes, name); const usda = conv.exportAsUSDA(); conv.delete();
```

The adapter rules. All of them are implemented and verified in `normalize.ts` (`normalizeNext(..., adapter = true)`) and `usda-scan.ts`:

1. **Renderables.**
   - Iterate the `mesh` nodes. Match `getMesh()` entries by `primPath`, **never** by `dataId`.
   - For a node without a mesh under an `instanceable` prim, reuse the mesh at the same path suffix under another instance of the same reference target. Use the node's `worldMatrix`.
   - Skip subtrees of `class` prims and of `PointInstancer.prototypePaths`.
2. **PointInstancers.** world = transpose3×3(`draw.transform`) · world(instancer node) (row-vector convention).
   **Amended (M7, m7-api.md M7-3):** the per-instance transform is rebuilt from the authored `positions` /
   `orientations` (half quaternions, NOT normalised, as pxr) / `scales` / `protoIndices` of the root-layer scan; rc4
   normalises the quaternion (3e-5 bbox error vs pxr).
3. **Lights.**
   - `spot` → SphereLight + shaping, with `coneAngle = angle·180/π`.
   - `directional` → DistantLight, with `angle` in degrees (the diameter).
   - `treatAsPoint` and spot `radius` come from the layer scan. The schema defaults are false / 0.5.
   - Then apply the plan §1.2 v1 rules: sphere → point with r := 0 and preserved I. The DistantLight Blender quirk applies when `doc` starts with `Blender v`.
   - Colour temperature applies when `enableColorTemperature` is set.
4. **Materials.** Use PreviewSurface from `mesh.material` / `mesh.materials[submesh.materialIndex]`. Patch `ior`, `clearcoat*`, `useSpecularWorkflow`, `specularColor`, Blender `specular` and `opacityThreshold` from the scan (shader prim → parent Material), and use the spec defaults otherwise.
   **M7 additions (m7-api.md §3.1, M7-4):** UsdUVTexture networks from the scan with Blender 5.2's importer semantics
   (diffuse / roughness + metallic / normal / emissive / opacity-of-the-diffuse-texture; one wrap = wrapS; normal
   strength 1); a mesh bound inside an instance prototype (rc4 reports no material) takes its `material:binding` and
   every PreviewSurface constant from the scan. In Blender-compatible mode (`blenderCompat`, the E2E-USD gate) the
   Blender-exporter `specular` input is not read (Blender's importer ignores it, M7-7).
5. **Frames.** Apply upAxis Z → R_x(−90°) and metersPerUnit scaling once, in the adapter. Apply `v = 1 − v`.
6. **Scan scope.**
   - The scan reads the root layer only (no composition). Log a warning when the file has other layers: a USDZ with several USD entries, or `subLayers`/external `references`/`payload` in the root layer.
   - Run it lazily: only when the scene has SphereLights or materials, or Blender-quirk detection is needed.
   - For `.usda`, decode the bytes instead of calling `exportAsUSDA()`.

## Follow-ups

- Change the dependency to the exact pin `1.0.0-rc4`. Owner: package.json.
- File upstream issues against LightUSD for findings 1, 2, 3–5, 6–8. Re-run `run-spike.ts` on every version bump.
- Re-run the harness on the M1 `make_cornell.py` USD export and on a Kitchen_set / PointInstancer asset (non-commercial test only). The harness currently reads files from `validation/assets/usd-spike/` (+ `validation/out/usd-spike/perf/`).
- Not covered by this spike: textures (UsdUVTexture), variants, multi-file references and payloads, `visibility`/`purpose` pruning, subdivision, `orientation = leftHanded`, cameras vs pxr.
- **M7 status.** UsdUVTexture (single root layer + texture assets resolved next to the file / inside a USDZ), instanceable
  prims and PointInstancers (incl. bindings inside prototypes) and cameras are covered and checked against pxr per draw /
  material / light / camera by (viii-L) (`validation/harness/m7-loader-fidelity.ts`, 7 files) and rendered against
  Blender's stock importer by E2E-USD (5 files). Still out of scope: variants, payloads / multi-layer composition,
  `visibility` / `purpose` pruning, subdivision, `UsdTransform2d`, non-`st` primvars.
