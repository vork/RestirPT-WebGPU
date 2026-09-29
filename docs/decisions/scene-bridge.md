# Scene bridge format (scene package v1)

A **scene package** is the exact scene our renderer draws, exported so that Blender can rebuild it for Cycles
(plan §5 M2, §7.5). It is written by `src/core/scene/scene-package.ts` (browser or Node) and read by
`validation/blender/build_scene.py`. One package is a directory:

```
<name>/scene.json      metadata (below)
<name>/geometry.bin    little-endian binary blobs referenced by scene.json "buffers"
<name>/tex_<i>.png     textures, lossless RGBA8 PNG (only when the scene has textures)
<name>/env.exr         the exact GPU env texel array (float32 ZIP EXR, rows TOP-DOWN as EXR convention)
```

## Frame and units

- **Everything is in the glTF canonical frame**: right-handed, +Y up, metres, **un-recentred** world coordinates
  (plan §1.2). `build_scene.py` applies `C = R_x(+90°)` exactly once to meshes, cameras and lights.
- Light units and parameters are Blender's (plan §1.4, `math.md#units-lights`).
- Matrices are column-major 4×4 float arrays (glTF/WebGPU convention).

## scene.json

```jsonc
{
  "format": "restir-scene-package", "version": 1,
  "name": "cornell",
  "source": { "uri": "validation/assets/cornell/cornell.glb", "sha256": "..." },      // informational
  "buffers": {                                  // byte ranges in geometry.bin (offset, length in bytes, dtype)
    "positions": { "offset": 0, "length": 408, "dtype": "f32", "components": 3 },
    "normals":   { "offset": 408, "length": 408, "dtype": "f32", "components": 3 },
    "uv0":       { ..., "dtype": "f32", "components": 2 },            // glTF UV convention (origin top-left)
    "color0":    { ..., "dtype": "f32", "components": 4 },            // optional
    "indices":   { ..., "dtype": "u32", "components": 3 },            // one entry per triangle, primId order
    "triMaterial": { ..., "dtype": "u32", "components": 1 }
  },
  "flatShaded": true,                           // Blender: shade_flat on all faces; else custom split normals
  "materials": [ {
      "name": "white", "model": "v1" | "principled",
      // model v1 (validation BSDF, plan §1.5): Diffuse BSDF + Glossy BSDF(GGX) mixed by 'mix' (0 = pure diffuse)
      "v1": { "diffuse": [r,g,b], "glossy": [r,g,b], "roughness": r, "mix": m },
      // model principled: MaterialData fields (baseColorFactor, metallicFactor, roughnessFactor, ior,
      //   specularFactor, specularColorFactor, transmissionFactor, emissive*, alphaMode, alphaCutoff)
      //   + texture refs {"texture": i, "texCoord": 0, "transform": [a,b,c,d,e,f]}
      "emission": { "color": [r,g,b], "strength": s },          // L_e = color*strength, two-sided
      "emissionSampling": "FRONT_BACK" | "NONE"
  } ],
  "textures": [ { "file": "tex_0.png", "wrapS": "repeat", "wrapT": "repeat", "filter": "linear" } ],
  "lights": [ {                                  // LightData (types.ts); ids are stable
      "id": 0, "name": "key", "type": "point"|"spot"|"rect"|"disk"|"sun",
      "color": [r,g,b], "power": P, "exposure": 0, "matrix": [16 floats],
      "spotSize": rad, "spotBlend": b, "sizeX": m, "sizeY": m, "spread": rad, "visibleToCamera": false
  } ],
  "lightMode": "A" | "B",                        // A: per-light MIS off in Blender (plan §1.4)
  "camera": { "matrix": [16], "yfov": rad, "znear": 1e-4 },
  "env": null | { "file": "env.exr", "strength": s, "tint": [r,g,b], "rotationZ": rad,
                  "visibleToCamera": true, "sampling": "AUTOMATIC" | "NONE",
                  "blenderWorld": "texture" | "constant",   // optional (M3c C0q); default "texture"
                  "sha256": "<hex>" },   // ENV-U9: SHA-256 of env.exr's pixels as written = little-endian float32,
                                          // RGBA interleaved, rows TOP-DOWN (Blender: foreach_get rows are bottom-up → flip)
  "render": { "width": 512, "height": 512, "maxBounces": 3 },
  "cycles": { "use_light_tree": false },          // optional reference-setting override (only this key; cycles-deviations D4)
  "frames": [                                    // optional: resolved per-frame states for animations
    { "frame": 0, "camera": { "matrix": [16], "yfov": rad }, "lights": { "<id>": { "matrix": [16], "power": P } },
      "env": { "rotationZ": rad, "strength": s } }
  ]
}
```

## Rules

- `primId` order in `indices` is the renderer's stable order. Blender faces are created in the same order, so
  per-face debugging can map primId → Blender face index directly.
- Normals: if `flatShaded`, Blender shades flat (geometric normals). Otherwise vertex normals are applied as
  custom split normals, matching our interpolated shading normals.
- The env EXR is written from the exact float32 texels our GPU samples. Blender asserts the image-pixel hash
  (ENV-U9, `math.md#env-mapping`).
- Anything the bridge cannot represent exactly (e.g. KTX2 textures, BLEND alpha) is a hard error, not a warning.
- **env.sampling** is the env-NEE switch on both sides: Blender `world.cycles.sampling_method` and our env alias entry
  (`NONE` = no env NEE, BSDF escapes with ω2 = 1; math.md#env-sampling).
- **env.blenderWorld = "constant"** (M3c C0q only): env.exr must be one constant colour c. Blender builds a plain
  Background node with Color = c·tint and the package strength instead of the texture graph, so Cycles has no background
  light (a constant world is not spatially varying) and samples the env by BSDF only. Our renderer still samples the
  constant texture. Frames with env overrides are refused for this variant.

