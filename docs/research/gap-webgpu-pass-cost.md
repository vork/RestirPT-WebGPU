# Gap-fill: WebGPU per-pass cost, binding and memory budget for ReSTIR PT Enhanced on an Apple M5 Pro

> Research report (gap-fill task `webgpu-pass-cost-budget`) for engineers implementing ReSTIR PT (Lin et al. 2022 GRIS + Lin/Kettunen/Wyman 2026 "Enhanced") in WGSL.
>
> **Where this file lives.** The orchestrator asked for `.../scratchpad/research/gap-webgpu-pass-cost-budget.md`. This session ran in **plan mode**, where the only writable file is this plan file, so the report is here. Copy it to the research path if needed. No files were written anywhere else. All benchmark code ran in-page in the Claude desktop built-in browser and was not saved to disk. The load-bearing WGSL and JS is reproduced in §10.
>
> **Tags.**
> - **[M]** = measured on this Mac in this session.
> - **[M-plat]** = measured earlier by the platform report.
> - **[SRC]** = stated by a cited source.
> - **[I]** / **[INFERENCE]** = my own reasoning or extrapolation.
> - **[UNVERIFIED]** = plausible but not checked.
>
> Mock kernels are *cost models*, not the renderer. §1.4 lists exactly what they contain and what they leave out.

---

## 0. TL;DR and decision

**Headline numbers** [M]. Sponza, 262,267 triangles, Apple M5 Pro, Dawn/Metal. All are GPU medians. Frame totals were cross-checked against wall-clock time (±2%).

| Full Enhanced-style frame (3 bounces + NEE, RR, temporal fwd+inv, paired spatial N=3, 10% of pairs need replay) | BVH2 | CWBVH (this session's port) |
|---|---|---|
| 1920×1080 | **132.5 ms** (30% replay: 171.0) | ≈110 [I] |
| 1280×720 | **60.6 ms** (30% replay: 76.0) | **50.8 ms** |
| 960×540 | **34.4 ms** (30% replay: 42.7) | **28.2 ms** |
| Paper reference: RTX 5880 Ada, hardware RT, 1080p (Enhanced Table 1, "All") | 15.53 ms [SRC] | |

The mock at 1080p is **~8.5× slower** than the paper's hardware-RT number. The platform report's extrapolated "45–65 ms at 720p" (its §5.7) is **confirmed at low replay fractions** (59–61 ms). It is exceeded when 30% of pairs need replay (76 ms).

**Lean configurations (measured whole frames, CWBVH, 10% replay):**

| Config | 1280×720 | 960×540 |
|---|---|---|
| Max 2 bounces, RR, temporal, spatial N=2 | 41.2 ms | 23.5 ms |
| Max 2 bounces, RR, temporal, spatial N=1 | 37.4 ms | 20.9 ms |
| … N=1, plus moving lights (suffix re-trace fwd+inv, NEE-only suffix) | 43.1 ms | 24.6 ms |
| … N=1, plus moving lights (suffix re-trace fwd+inv, 1-bounce suffix) | 54.9 ms | 31.4 ms |

**Unit costs at 720p, BVH2** [M]. These make up the cost model in §7.

| Operation | Cost |
|---|---|
| Incoherent reconnection-visibility ray | **≈5.0 ns/ray (~190–212 Mrays/s)**. Neighbour distance hardly matters (2 px ≈ 30 px), because x₂ is already scattered. |
| Reconnection-only shift (64 B reservoir load, rc-vertex rebuild, 2 GGX+Lambert evals, 1 visibility ray, 16 B slot write) | **≈4.65 ns per pair** |
| Compacted prefix replay (1–3 bounces, mean 2) + fused reconnection | **≈21 ns per compacted pair** |
| Initial path tree (primary + 3 bounces + NEE at every vertex, RR) | ≈28 ns/px |
| Suffix re-trace, per direction | 4.8 ns/px (NEE only), 12.3 ns/px (1 bounce + NEE), 19 ns/px (2 bounces + NEE) |

**Answers to the open design questions:**

1. **Go / no-go.**
   - **GO for interactive rates at a 960×540 internal resolution** with upscaling to the canvas: the full Enhanced graph with CWBVH runs at 28 ms (35 fps) with static lights and 25–31 ms (N=1) with moving lights.
   - **720p is an "interactive 15–27 fps" quality mode** (37–51 ms), not 30 fps.
   - **1080p real time is NO-GO**; use it for progressive and validation rendering only.
2. **Max bounces:** 2 by default (NEE at x₁, x₂, x₃). Max 3 as a quality option (+3.5–4 ms at 720p with CWBVH). Keep RR in initial sampling (Enhanced §6.2.4): it cuts the initial pass by 1.4× (37.0 → 25.7 ms at 720p).
3. **Neighbour count:** paired spatial reuse with **N=3 at 540p** (fits), **N=1–2 at 720p**. Each extra paired neighbour costs one reconnection shift per pixel plus its share of replays (10% replay):
   - measured N=2 vs N=1 with CWBVH: **+3.8 ms at 720p, +2.6 ms at 540p**;
   - with BVH2 ≈ +6.2 ms at 720p [I, from the unit costs in §7].
4. **Fused vs separate retrace:**
   - **Prefix replay: separate, compacted, indirect pass.** It is 1.39–1.40× faster than replaying inline in the shift kernel, at both 720p (29.2 vs 40.9 ms) and 540p (17.6 vs 24.5 ms), matching the paper's 1.39×.
   - **Fuse the reconnection into that replay pass.** Pairs then need no replay state (0 B instead of 32 B per pair).
   - **Suffix re-trace for moving lights: separate pass.** Fusing it into the shift kernel was 3–4% *slower*. Run it only on light-change frames, and compact it when possible.
5. **CWBVH is recommended but not required** for the 540p go decision.
   - The port gives **1.32–1.57× on any-hit / visibility / shift / NEE-suffix passes** and 1.05–1.21× on closest-hit-heavy passes: **1.19–1.22× per frame**, not the paper's 2×.
   - It is needed to get near 30 fps at 720p, and even then it is not sufficient.
6. **Compile time.**
   - Traversal call-site count is **not** the problem: 1 vs 2 sites compile in the same time (61 KB: 421 vs 422 ms; 124 KB: 920 vs 922 ms). Each extra traversal copy costs only ≈20–50 ms.
   - The cost is **inlined copies of the material/BSDF library**: ≈70 ms per copy with 48 KB of material code, ≈160 ms per copy with 110 KB.
   - **Do not force a single traversal call site with a state-machine loop.** It is **39% slower at runtime** (64.4 vs 46.4 ms).
7. **Bindings.**
   - A merged-buffer plan fits every pass in **≤ 8 of 10 storage buffers, ≤ 2 of 8 storage textures, 4 bind groups** (§4).
   - A single read-write "arena" binding has no performance cost (0.79 vs 0.85 ms) [M].
   - Binding disjoint ranges of one buffer read-only *and* read-write in the same pass is a **validation error** [M], so arenas must be split by write-set.
8. **Memory** (production layout, 2 reservoir buffers + 3 slots, excluding scene/textures/denoiser): **273 MB at 720p and 614 MB at 1080p**. With 3 reservoir buffers: 332 MB and 747 MB. Ensemble validation mode with 64 members × 256² (4.19 M px) needs **≈1.31 GB** (§5).

---

## 1. Method

### 1.1 Environment [M]

- **Where the measurements ran.** The Claude desktop built-in browser, Chromium **152.0.7977.130** (UA `... Claude/2.9939.2 Chrome/152.0.7977.130`), on origin `https://webgpureport.org` (secure context, CORS fetch allowed).
  - This is the same Dawn/Metal stack as the installed Chrome 154 [INFERENCE, as in the platform report §1]. The user's Chrome profile was not touched.
- **Adapter.** Vendor `apple`, architecture `metal-3`, features include `timestamp-query`, `subgroups`, `shader-f16`, `texture-formats-tier2`.
- **Limits requested at adapter maximum:**
  - `maxStorageBufferBindingSize` and `maxBufferSize`: 4,294,967,292
  - `maxStorageBuffersPerShaderStage`: 10
  - `maxStorageTexturesPerShaderStage`: 8
  - `maxComputeWorkgroupStorageSize`: 32 KiB
  - `maxBindGroups`: 4 (hard)
- **WGSL language features:** identical to the platform report's list. No `buffer_view` (that is Chrome 153+).

### 1.2 Scene and BVH [M]

- **Scene.** Khronos glTF-Sample-Assets Sponza (`Models/Sponza/glTF/Sponza.gltf` + `.bin`, fetched from raw.githubusercontent.com in 831 ms): **262,267 triangles, 192,496 vertices, 25 materials**. Bounds x ∈ [−15.37, 14.40], y ∈ [−1.01, 11.44], z ∈ [−9.46, 8.84].
- **BVH2** for all BVH2 numbers:
  - inline JS binned SAH, 16 bins, maximum leaf size 4, C_trav = C_isect = 1;
  - **SAH cost 76.70, 131,364 interior nodes.** This is identical to the platform report's tree, so its throughput numbers carry over;
  - build 322 ms;
  - converted to the Aila–Laine layout: 8.41 MB of 64 B nodes, 12.59 MB of 48 B `(v0|primId, e1, e2)` triangles;
  - the traversal is the platform report's §4.3 kernel, returning a `Hit` struct instead of a bit-cast float.
- **CWBVH**, built in this session (§2.7, §10.2):
  - a BVH2 rebuilt with **maximum leaf size 3** (371 ms);
  - collapsed greedily to 8-wide by largest child surface area, with octant-ordered child slots, then encoded in the Ylitie et al. 2017 / tinybvh 80 B layout (165 ms);
  - result: **43,306 nodes, 3.46 MB**; maximum 20 triangles referenced per node.
  - Correctness: **99.991% of primary and 99.985% of secondary hits return the same triangle as BVH2** at 1280×720. The rest are coplanar or tie cases (maximum position difference 0.42 m on a handful of z-fighting pixels).
- **Attribute data** for shading and rc-vertex rebuild:
  - `prims` (16 B: i0, i1, i2, material), 4.2 MB;
  - `verts` (16 B: position + octahedral normal), 3.1 MB;
  - `mats` (2 × vec4 per material).
  - Scene storage total ≈ 28.3 MB (BVH2) or 23.4 MB (CWBVH).
- **Camera and lights.**
  - Camera at (−11, 1.5, −0.4), looking along +x with 0.05 rad pitch down, 60° vertical FOV.
  - 100% of primary rays hit (mean depth 5.8 m); 98.1% of one-bounce diffuse rays hit.
  - Lights: 8 point lights in a uniform array.

### 1.3 Timing [M]

- Each measured stage is **one compute pass with `timestampWrites`**. Apple GPUs only sample at pass boundaries (platform §2.3).
- **Median of 6–10 repetitions after 2 warm-ups.** Each repetition is its own command buffer and submit, and the frame index is bumped every repetition so neighbour choices change.
- Timestamps are **quantized to 65.536 µs**, so passes shorter than that show 0 (e.g. `args`, `lighttiles`).
- **Run-to-run variation is about ±5–8%.** For example, the spatial shift at 720p ranged 11.1–12.9 ms across runs; outliers stay in the min/max but not in the median.
- **Whole-frame wall clock:** 30 frames were submitted back to back, one submit per frame, followed by `onSubmittedWorkDone()`.

| Config | Sum of GPU passes | Wall clock per frame |
|---|---|---|
| 720p, CWBVH, 2 bounces, N=1 | 37.7 ms | 38.3 ms |
| 540p, CWBVH, 3 bounces, N=3 | 29.6 ms | 29.2 ms |
| 540p, CWBVH, 2 bounces, N=1 | 21.2 ms | 21.7 ms |

  - **Conclusion: 14–16 compute passes per frame add no measurable bubbles.** The sum of pass times is a valid frame-time predictor.
  - CPU encode time was below timer resolution (< 0.1 ms per frame).

### 1.4 What the mock kernels contain (and omit)

All kernels are 8×8 workgroups, one thread per pixel, except replay (64-wide 1D, indirect) and the duplication map (16×16).

The common code (about 7 KB of WGSL, §10.1) contains:
- the platform BVH2 traversal (32-entry private stack);
- Wächter–Binder ray offsets;
- attribute fetch through `prims` → 3 × `verts` (interpolated position and octahedral normal, geometric normal);
- a GGX (Trowbridge–Reitz) + Lambert mixture BSDF: eval (f·cos, pdf) and sample (lobe select, GGX D-sampling or cosine sampling);
- a counter-based PCG hash RNG, `rnd(seed, dim)`.

| Kernel | What it does | Rays per item |
|---|---|---|
| `gbuf` (test-data generator) | Primary hit x₁ plus one BSDF bounce to x₂. Writes a 64 B/px G-buffer (x₁ and x₂ position, normal, prim, material) and a **synthetic 64 B reservoir (4 × vec4u)**: W, flags (rcLen), seed, rcPrim = x₂ prim, bary unorm16×2, cached-Jacobian triple, octahedral ω_k, f16-packed F and L_o. rcLen = 2 for (100 − r)% of pixels and 3, 4 or 5 uniformly for r% (**replay fraction r = 0, 10 or 30%**) | 2 closest |
| **(a) `vis`** | Per pixel, NS neighbours q in a disk of radius 1–30 px (sqrt-uniform radius). Segment from offset(x₁(p)) to offset(x₂(q)). The ray is skipped if either endpoint faces away (as a real shift would fail before tracing). Any-hit, t ∈ [0, 1], unnormalized direction | NS any-hit |
| **(c) `shift`** (reconnection pre-pass) | Per pixel p: load its own 64 B reservoir, rebuild x_k from (prim, bary) via `prims`/`verts`. Then for each of NS partners q (disk 1–30 px; 0–2 px for temporal): if rcLen ≤ 2, y_{k−1} = x₁(q); evaluate the BSDF at y_{k−1} toward x_k and at x_k from −ω′ toward ω_k (2 GGX+Lambert evals); compute G and the Jacobian ratio and a dual-footprint compare (ALU only); trace the visibility ray; write a 16 B slot (F·J as f16×3, J as f32, flags). Pairs with rcLen > 2 are left to the replay pass | ≤ NS any-hit |
| **(b) `classify`** | Per pixel × NS: needs-replay = rcLen > 2. **Subgroup-ballot append** (platform §5.6 code) of `(p<<2)\|s` into a queue buffer (atomic counter at offset 0) | 0 |
| `args` | 1 thread: `args = (ceil(count/64), 1, 1)` | 0 |
| **(b) `replay`** (indirect) | One thread per compacted pair: replay rcLen − 2 bounces (1–3) from y₁ = x₁(q) with the source seed (BSDF sample, eval, closest-hit trace, attribute fetch, footprint compare). **FUSE = true** then reconnects to x_k (as `shift`) and writes the 16 B slot. **FUSE = false** writes 32 B of replay state instead | 1–3 closest + ≤ 1 any-hit |
| **(d) `suffix`** | Per pixel: rebuild x_k, trace NB bounces from x_k along stored ω_k (with BSDF sampling after the first), then NEE (light select + shadow ray) at the final vertex. NB = 0 means NEE only (an NEE-final path's light moved) | NB closest + 1 any-hit |
| `init` (initial path tree) | Primary ray, then for b = 0..MAXB: NEE with RIS over NC' = max(1, NC/(b+1)²) light candidates (target without visibility) + 1 shadow ray; streaming RIS with a separate resampling stream; optional RR for b ≥ 1 (q = clamp(lum(thp), 0.05, 1), folded into the source pdf); BSDF sample → closest hit → attribute fetch → footprint/roughness rc-vertex test. Writes 64 B G-buffer and 64 B reservoir | 1 + ≤ MAXB closest + ≤ MAXB+1 any-hit |
| `resample` | Pairwise-MIS-shaped arithmetic over its own reservoir, NS partner reservoirs (the 32 B actually needed), its own NS slots and the partners' slots; streaming selection; re-reads the selected 64 B reservoir; writes 64 B + an RGBA32F colour (vector weights). NS = 1 stands in for the temporal Talbot resample | 0 |
| `dup` | Enhanced §5 duplication map: 16×16 workgroup, 32×32 `u32` seed tile (4 KiB LDS) loaded from reservoir plane a, 289 compares per pixel, u32 count out | 0 |
| `resolve` | accum += colour (RGBA32F buffer), Reinhard tonemap, `pack4x8unorm` | 0 |
| `tiles` | 128 × 1024 presampled light records (32 B each) | 0 |

**Omissions** (see §8 for their likely impact):
- no TLAS or instancing;
- no alpha-tested any-hit;
- no emissive-triangle or environment light sampling;
- no lobe bookkeeping or full invertibility checks;
- a simplified Jacobian and MIS (arithmetic cost only);
- partners come from a hash, not the pairing textures (similar locality);
- no texture sampling, except one test (§2.9);
- no denoiser or upscaler.

---

## 2. Measurements

### 2.1 (a) Incoherent reconnection-visibility rays [M]

x₁(p) → x₂(q), q at distance 1–30 px, any-hit, BVH2:

| Resolution | Rays per pixel (after facing test) | Time | Throughput | Occluded |
|---|---|---|---|---|
| 1280×720, 1 neighbour | 0.822 | **3.80 ms** | 199 Mrays/s | 9.8% |
| 1280×720, 3 neighbours | 2.468 | **11.93 ms** | 191 Mrays/s | 9.8% |
| 1280×720, 3 neighbours, radius 0–2 px (temporal-like) | 2.731 | 11.86 ms | 212 Mrays/s | 3.7% |
| 1280×720, 3 neighbours, radius 1–8 px | 2.628 | 12.32 ms | 197 Mrays/s | 5.9% |
| 1280×720, 3 neighbours, **CWBVH** | 2.468 | **9.11 ms** | ≈251 Mrays/s (1.32×) | |
| 960×540, 1 neighbour | | 2.10 ms | | |
| 960×540, 3 neighbours | | 7.14 ms | | |
| 960×540, 3 neighbours, CWBVH | | 5.24 ms (1.36×) | | |

Findings:
- Reconnection rays behave exactly like diffuse secondary rays (platform: ~192 Mrays/s), not like coherent shadow rays (platform: ~610 Mrays/s to a single point light).
- **Neighbour distance does not buy coherence**: the endpoints x₂(q) are already scattered by the diffuse bounce.
- About 90% of traced segments are *visible*, so they traverse the whole segment (no early out).
- 18% of candidate pairs fail the facing test before any ray is traced. A real shift would also fail on BSDF zero or footprint rejection.

### 2.2 (b) Compacted prefix replay (subgroup-ballot append + `dispatchWorkgroupsIndirect`) [M]

Spatial, N=3, 1280×720, BVH2, 30% replay fraction:

| Pass | Median | Note |
|---|---|---|
| shift (reconnection pairs only) | 11.27 ms | 70% of pairs active |
| classify (ballot append, 3 per pixel) | 0.39–0.79 ms | 813,066 items (29.4% of 2.76 M pairs) |
| args | < 0.07 ms | |
| **replay + fused reconnection (indirect)** | **17.11 ms** | **21.0 ns/item** |
| replay only (writes 32 B state) | 13.17 ms | the reconnection part is therefore ≈ 3.9 ms (≈ 4.8 ns/item) |
| **Total, compacted** | **≈ 29.2 ms** | |
| **Inline alternative:** shift with replay inside the same kernel (warp-divergent) | **40.89 ms** | **1.40× slower** |
| 960×540, compacted (7.14 + 0.26 + 10.16) vs inline | 17.6 vs 24.5 ms | **1.39×** |

- The paper reports 9.73 → 6.99 ms (**1.39×**) for spatial from compaction (Enhanced Table 1; enhanced-paper report §6.3, §8.2). **The WebGPU gain matches the hardware-RT gain.**

**Replay-fraction sweep, 720p, BVH2** (the queue count scales linearly with the fraction):

| Replay fraction | Spatial N=3: shift + classify + replay | Spatial total | Temporal: shift N=2 + classify + replay + resample | Temporal total |
|---|---|---|---|---|
| 0% | 12.85 + 0.59 + 0 | **13.4 ms** | 8.32 + 0.20 + 0 + 0.52 | **9.0 ms** |
| 10% (271 k / 181 k items) | 12.52 + 0.20 + 6.10 | **18.8 ms** | 8.65 + 0.26 + 3.87 + 0.72 | **13.5 ms** |
| 30% (812 k / 542 k items) | 11.08 + 0.59 + 17.11 | **28.8 ms** | 7.27 + 0.46 + 11.01 + 0.85 | **19.6 ms** |

- Per-item replay cost is constant at **20.3–22.5 ns** (mean 2 bounces + reconnection) for both the temporal and spatial queues.
- The shift pass barely shrinks when pairs move to the replay queue, because idle lanes stay in their warps.

### 2.3 (c) Mock shift kernel [M]

It loads the 64 B reservoir, rebuilds the rc vertex from (prim, bary), does 2 GGX+Lambert evals and 1 visibility ray, and writes a 16 B slot.

| Config | Median | Per pair |
|---|---|---|
| 720p, N=1, radius 1–30 | 3.93 ms (another run: 3.60) | ≈ 4.3 ns |
| 720p, N=2, radius 0–2 (temporal fwd+inv) | 7.27–8.65 ms | ≈ 4.3 ns |
| 720p, N=3, 0% replay (all pairs reconnect) | **12.85 ms** | **4.65 ns** |
| 720p, N=3, CWBVH, 10% replay | 9.11–9.18 ms (BVH2 at 10%: 12.58–12.85) | ≈ 1.37–1.40× |
| 540p, N=1 / N=3 (30% replay) | 2.29 / 7.01 ms | |
| 1080p, N=3 (10% replay) | 28.57 ms | |

Slot outcome at 720p with 30% replay, N=3: 60.3% visible, 11.2% occluded, 28.5% no ray (backfacing or failed).

- Loading the reservoir and rebuilding x_k once per pixel amortizes over the N partners. This is the paired-reuse structure, where each pixel shifts *its own* path into each partner.
- The per-shift cost is essentially the visibility ray plus about 0.5 ns of loads and evals.

### 2.4 (d) Suffix re-trace for moving lights [M]

Per pixel: rebuild x_k, re-trace from x_k along stored ω_k, NEE at the final vertex.

| Config | 1280×720 BVH2 | 1280×720 CWBVH | 960×540 BVH2 | 960×540 CWBVH |
|---|---|---|---|---|
| NB=0 (NEE only: NEE-final path, light moved) | **4.46 ms** (5.05 in another run) | **3.08–3.21 ms** (1.57×) | 3.41 ms | 1.77–1.84 ms |
| NB=1 (1 bounce + NEE) | **11.34 ms** | **8.85–9.04 ms** (1.29×) | 6.95 ms | 4.85–5.05 ms |
| NB=2 (2 bounces + NEE) | **17.50 ms** | – | 10.03 ms | – |

**Fused vs separate** (720p, BVH2, NB=1 suffix fused into the temporal shift kernel):

| Shift N | Fused | Separate (shift + suffix) | Verdict |
|---|---|---|---|
| N=1 | 15.53 ms | 3.60 + 11.34 = 14.94 ms | fused 4% slower |
| N=2 | 19.66 ms | 7.67 + 11.34 = 19.01 ms | fused 3% slower |

**Keep the suffix re-trace a separate pass.**

**Cost in a moving-light frame** [I from the above]. Exact temporal MIS needs the forward suffix under S_t and the inverse (canonical) suffix under S_{t−1} (restir-practice report §3.6), i.e. **2 × suffix**:

| Suffix | 720p BVH2 | 720p CWBVH | 540p CWBVH |
|---|---|---|---|
| NEE-only | +8.9 ms | +6.2 ms | +3.6 ms |
| 1-bounce | +22.7 ms | +18.0 ms | +9.9 ms |

The measured frames E and F in §3.2 confirm this.

### 2.5 Initial path tree [M]

All at 1280×720; 540p in brackets where measured.

| Variant | BVH2 | CWBVH |
|---|---|---|
| MAXB=3, NEE at every vertex (8 lights, 1 candidate), no RR | 37.03 ms (540p: 22.02) | 31.06 ms (1.21×) |
| MAXB=3, RR at b ≥ 1 | **25.69 ms** (540p: 14.29–15.07) | **23.92 ms** (540p: 13.50) |
| MAXB=3, NEE RIS with 32 / 8 / 3 / 2 candidates (light-tile style) | 39.58 ms | |
| MAXB=3, RIS 32 candidates + RR | 27.26 ms | |
| MAXB=2, no RR | 26.41 ms | |
| MAXB=2, RR | | **19.92–20.19 ms** (540p: 11.34–11.53) |
| MAXB=3, RR, 1920×1080 | 56.95 ms | |

- RR in initial sampling only gives **1.44×** here. The paper reports 12.56 → 5.21 ms for its initial pass (enhanced-paper report §6.5), with an unknown path length.
- RIS NEE with 32 candidates costs only +2.6 ms at 720p (ALU plus a uniform-array light read), because only the selected light is shadow-tested.
- Cross-check against the platform report:
  - Primary + one diffuse bounce (`gbuf`) = 6.49–7.21 ms at 720p, against the platform's 14.29 ms at 1080p × 0.444 = 6.35 ms. The extra cost is attribute fetch, GGX, and 128 B/px of writes.
  - The platform's "3 bounces + NEE = 77.3 ms at 1080p" scales to 34.4 ms at 720p; this mock measures 37.0 ms.

### 2.6 Small passes [M]

1280×720, BVH2 data. Other resolutions from the frame runs.

| Pass | 720p | 540p | 1080p |
|---|---|---|---|
| Spatial resample N=3 (+ RGB vector weights) | 0.79–0.98 ms | 0.39–0.52 ms | 2.03 ms |
| Temporal resample (N=1 stand-in) | 0.52–0.92 ms | 0.39–0.46 ms | 1.44–1.77 ms |
| Duplication map 17×17 (4 KiB LDS tile) | 0.33 ms | 0.20 ms | 0.72 ms |
| Resolve (accumulate + tonemap) | 0.13–0.20 ms | 0.07 ms | 0.33–0.39 ms |
| Light tiles 128 × 1024 | < 0.07 ms | < 0.07 ms | < 0.07 ms |
| classify, per queue | 0.20–0.66 ms | 0.13–0.20 ms | 0.52–0.98 ms |
| args | < 0.07 ms | | |

These passes are bandwidth-bound and together cost about 2–3 ms at 720p. **The resample passes are not a concern; ray tracing is.**

### 2.7 CWBVH vs BVH2 on the same passes [M]

1280×720, 10% replay unless noted:

| Pass | BVH2 | CWBVH | Speed-up |
|---|---|---|---|
| gbuf (coherent primary + 1 diffuse bounce, closest hit) | 7.21 ms | 6.88 ms | 1.05× |
| (a) visibility, 3 per pixel (any hit) | 12.06 ms | 9.11 ms | **1.32×** |
| init MAXB=3, RR | 26.15 ms | 23.92 ms | 1.09× |
| init MAXB=3, no RR | 37.62 ms | 31.06 ms | 1.21× |
| suffix NEE-only | 5.05 ms | 3.21 ms | **1.57×** |
| suffix NB=1 | 11.40 ms | 8.85 ms | 1.29× |
| spatial shift N=3 | 12.71 ms | 9.11 ms | **1.40×** |
| spatial replay + rc (10% / 30%) | 6.03 / 16.91 ms | 4.98 / 15.47 ms | 1.21× / 1.09× |
| temporal shift N=2 (10% / 30%) | 8.06 / 8.00 ms | 5.90 / 5.64 ms | **1.37× / 1.42×** |
| temporal replay + rc (10% / 30%) | 3.80 / 11.67 ms | 3.21 / 10.22 ms | 1.18× / 1.14× |
| Whole frame 720p / 540p (§3) | 60.6 / 34.4 ms | 50.8 / 28.2 ms | **1.19× / 1.22×** |

- The gains concentrate in **any-hit segment rays** (shift, visibility, NEE-suffix). Closest-hit bounce rays gain little.
- The Ylitie paper's 1.9–2.1× (platform §4.2) was measured on NVIDIA. Unified memory and large caches on the M5 make BVH2 node fetches less costly [INFERENCE].
- Not tried here (possible further gains, [UNVERIFIED]):
  - SAH-optimal collapse (Ylitie's dynamic programming) instead of greedy area collapse;
  - compressed triangles (Baldwin–Weber / tinybvh `CWBVH_COMPRESSED_TRIS`);
  - triangle postponing;
  - a shared-memory traversal stack;
  - f16 node math.
- Implementation cost was ≈ 110 lines of WGSL plus ≈ 120 lines of JS. It was correct on the first compile (§10.2).

### 2.8 (e) Pipeline compile time for 60–120 KB shift/replay modules [M]

**Generated module.** Common code (traversal, attributes, BSDF helpers) plus the reservoir/`RS` helpers, plus a **material library** of M distinct GGX/sheen/clearcoat-ish evaluation and sampling functions dispatched by `switch` (`evalAll`, `sampleAll`), plus replay, reconnect and an entry point.
- Cold compiles are salted with `const SALT` used in seeding. `createShaderModule` + `getCompilationInfo` (Tint front end) took 9–19 ms. Everything else is `createComputePipelineAsync` (Tint → MSL → Metal).

Variants:
- **E1:** single traversal call site (state-machine loop), 1 sample + 3 eval call sites.
- **E2:** normal code. `replayX()` (traverse, sample, eval) plus `reconnectX()` (2 evals, traverse): 2 traversal sites, 1 sample + 3 eval sites.
- **E2t:** E2 plus 2 extra any-hit traversal sites (4 traversal copies, same material copies).
- **E2e:** E2 with both reconnection evals routed through one `evalAll` call (1 sample + 2 eval sites).
- **E4:** naive temporal (2 × replay + 2 × reconnect): 4 traversal sites, 2 sample + 6 eval sites.

| Module | E1 | E2 | E2t | E2e | E4 |
|---|---|---|---|---|---|
| 61–63 KB (M=36) | 409–422 ms | 421–445 ms | 457–467 ms | **345–373 ms** | 735–793 ms |
| 124–126 KB (M=80) | 922–943 ms | 920–945 ms | 1022–1039 ms | **771–774 ms** | 1624–1639 ms |

**Cost model** [I, fitted to the table]:

`pipeline_ms ≈ base + c_mat · (#inlined material-switch copies) + c_trav · (#traversal copies)`

| Material library size | c_mat per copy | c_trav per copy |
|---|---|---|
| ≈ 48 KB | ≈ 70–80 ms | ≈ 20 ms |
| ≈ 110 KB | ≈ 160–170 ms | ≈ 50 ms |

- **C2-Renderer's compile-time explosion** (reference-code report §2.4: "randomReplay at multiple call sites") is the *replay function's material code being inlined repeatedly*, not traversal.
- **Runtime of the variants** (M=36, inline replay for all pairs, 30% replay, 720p):

| Variant | Runtime |
|---|---|
| E2 | 46.40 ms |
| E2e | 46.40 ms |
| **E1 (single traversal site via state machine)** | **64.36 ms (+39%)** |

  **Never trade runtime for one traversal call site.**
- **Parallel compile**, 61 KB E2 modules via `Promise.all` of `createComputePipelineAsync`:

| Pipelines in parallel | Wall time | Per-pipeline times | Effective parallelism |
|---|---|---|---|
| 1 | 490 ms | 490 ms | – |
| 3 | 564 ms | 501–564 ms | – |
| 6 | **723 ms** | 597–721 ms | **≈ 4×** |

- **Cache.**
  - Identical source and constants recompile in **0 ms** (in-session cache).
  - A new `override` value (`NS: 2`) costs a **full compile (412 ms)**, confirming platform §5.3.
- **The real mock kernels** (7–11 KB each) compiled in 11–87 ms. Kernels with traversal took 46–87 ms; kernels without took 11–23 ms.

### 2.9 Other checks [M]

- **Texture sampling.** Three `textureSampleLevel(texture_2d_array, LOD 0)` fetches per material lookup (1024² × 8 layers rgba8, 32 MB), at hashed UVs:

| Pass | Untextured | Textured | Change |
|---|---|---|---|
| init (CWBVH, MAXB=2, RR) | 19.99 ms | 20.38 ms | +2% |
| spatial shift N=3 | 9.24 ms | 10.36 ms | +12% |

  Real scenes with hundreds of MB of textures will miss caches more [UNVERIFIED magnitude].
- **Merged "arena" binding** (resample N=3, 720p):

| Binding scheme | Time |
|---|---|
| One `read_write` binding holding res, slots, reso and colour at uniform offsets | 0.786 ms |
| Separate read-only + read-write bindings | 0.852 ms |

  **No penalty** for the arena.
- **Aliasing rule.** Binding disjoint sub-ranges of the *same* `GPUBuffer` as `read-only-storage` and `storage` in one dispatch fails with: "usage (Storage(read-write)|Storage(read-only)) includes writable usage and another usage in the same synchronization scope". Usage is tracked per buffer, not per range.

---

## 3. Per-pass ms table for the Enhanced pass graph

### 3.1 Main table (10% of pairs need replay, static lights)

All values in ms. **M** = measured in a whole-frame run; **I** = inferred. The RTX column is the paper's Table 1 "All" row (4-scene average, 1920×1080, hardware RT) [SRC].

| # | Pass (Enhanced graph) | 720p BVH2 | 720p CWBVH | 540p BVH2 | 540p CWBVH | 1080p BVH2 | RTX 5880 Ada, 1080p |
|---|---|---|---|---|---|---|---|
| 1 | Light tiles 128 × 1024 | < 0.07 M | < 0.07 M | < 0.07 M | < 0.07 M | 0.07 M | (in "others") |
| 2 | **Initial path tree**: primary + ≤ 3 bounces, NEE at every vertex, RR at initial sampling only | **26.02 M** | **23.92 M** | **14.68 M** | **13.50 M** | **56.95 M** | 6.77 |
| 3a | Temporal: shift fwd + inv (2 reconnection shifts per pixel) | 8.65 M | 6.42 M | 5.31 M | 3.28 M | 19.20 M | |
| 3b | Temporal: classify + args (ballot append) | 0.59 M | 0.26 M | 0.13 M | 0.13 M | 0.52 M | |
| 3c | Temporal: replay + fused reconnection (indirect, 10% of pairs) | 3.87 M | 3.21 M | 2.29 M | 1.90 M | 8.85 M | |
| 3d | Temporal: resample (Talbot, 2 inputs) | 0.79 M | 0.85 M | 0.39 M | 0.46 M | 1.77 M | |
| | **Temporal subtotal** | **13.9** | **10.7** | **8.1** | **5.8** | **30.3** | 3.20 |
| 4a | Spatial pre-pass: shifts to N=3 paired partners | 12.85 M | 9.18 M | 7.34 M | 5.37 M | 28.57 M | |
| 4b | Spatial: classify + args | 0.26 M | 0.26 M | 0.20 M | 0.13 M | 0.59 M | |
| 4c | Spatial: replay + fused reconnection (indirect, 10%) | 6.23 M | 5.24 M | 3.34 M | 2.75 M | 12.78 M | |
| 5 | Spatial resample N=3 + vector-weight shading | 0.85 M | 0.98 M | 0.46 M | 0.39 M | 2.03 M | |
| | **Spatial subtotal** | **20.2** | **15.7** | **11.3** | **8.6** | **44.0** | 3.73 |
| 6 | Duplication map (17×17 / 288) | 0.33 M | 0.33 M | 0.20 M | 0.20 M | 0.72 M | |
| 7 | Resolve (accumulate, tonemap) | 0.20 M | 0.13 M | 0.07 M | 0.07 M | 0.39 M | |
| | Others subtotal (1 + 6 + 7) | ≈ 0.5 | ≈ 0.5 | ≈ 0.3 | ≈ 0.3 | ≈ 1.2 | 1.83 |
| | **Total per frame** | **60.6 M** | **50.8 M** | **34.4 M** | **28.2 M** | **132.5 M** | **15.53** |
| | Same frame at **30% replay** | 76.0 M | ≈ 67 I | 42.7 M | ≈ 37 I | 171.0 M | |
| | + moving lights, NEE-only suffix fwd+inv | +8.9 I (2 × 4.46 M) | +6.2 M | +6.8 I (2 × 3.41 M) | +3.6 M | ≈ +20 I | |
| | + moving lights, 1-bounce suffix fwd+inv | +22.7 I (2 × 11.34 M) | +18.0 M | +13.9 I (2 × 6.95 M) | +9.9 M | ≈ +50 I | |
| | Not included: denoiser / upscaler | ≈ 2–4 [UNVERIFIED] | | ≈ 1.5–3 [UNVERIFIED] | | | (NRD / DLSS in the paper) |

- The "M" frame totals come from whole-frame runs.
- The CWBVH 30%-replay totals are **[I]**: the BVH2 30% frame scaled by the measured CWBVH factors of each pass. They were not measured as whole frames.

### 3.2 Lean whole-frame configurations (CWBVH, 10% replay; every row measured) [M]

| Config | 720p | 540p |
|---|---|---|
| C: MAXB=2, RR, temporal, spatial N=2 | 41.2 ms (initial 20.1, T 10.6, S 10.0, other 0.5) | 23.5 ms |
| D: MAXB=2, RR, temporal, spatial N=1 | **37.4 ms** (wall 38.3) | **20.9 ms** (wall 21.7) |
| E: D + moving-light suffix fwd+inv, NEE-only | 43.1 ms | 24.6 ms |
| F: D + moving-light suffix fwd+inv, 1 bounce | 54.9 ms | 31.4 ms |

### 3.3 Where the time goes (720p, BVH2, full config)

- Initial: 43%.
- Temporal: 23%, of which 62% is the two shift visibility rays.
- Spatial: 33%, of which 64% is the three shift visibility rays.
- Everything else: < 1%.

Versus the RTX 5880 Ada at 1080p:

| Stage | This mock at 1080p | Paper | Ratio |
|---|---|---|---|
| Initial | 56.95 ms | 6.77 ms | 8.4× |
| Temporal | 30.3 ms | 3.2 ms | 9.5× |
| Spatial | 44.0 ms | 3.73 ms | 11.8× |

The gap is widest for the shift passes. Their reconnection rays are exactly the incoherent any-hit rays that hardware RT accelerates most [INFERENCE].

---

## 4. Binding plan: ≤ 10 storage buffers, ≤ 8 storage textures, ≤ 4 bind groups

### 4.1 Rules that constrain the plan

- **Explicit layouts count in full.** `maxStorageBuffersPerShaderStage` (10) is checked against the **pipeline layout** (every `COMPUTE`-visible storage entry in all bind group layouts), not against what the shader uses [SRC: WebGPU spec, pipeline-layout validation].
  - With a shared, explicit group-0 layout, group 0's storage buffers count against **every** pass that uses it.
  - `layout: 'auto'` counts only what is used, but auto layouts cannot share bind groups across pipelines [SRC].
  - **Use explicit layouts and keep group 0 small.** Give tiny passes (`args`, `resolve`) a pipeline layout whose group 0 is an empty layout.
- **Aliasing** [M]: one `GPUBuffer` must not be bound both read-only and writable in one dispatch, even on disjoint ranges.
  - **Arena buffers must be partitioned by write-set.** Anything a pass writes must sit in a buffer that the pass binds exactly once, as `read_write`.
  - Binding a whole arena `read_write` costs nothing measurable [M, §2.9].
- **Keep per-pixel images the neighbours only read in sampled textures** (`textureLoad` on `texture_2d<u32>` or `rgba16float`). They count against the 48 sampled-texture budget, not the 10 storage buffers:
  - V-buffer, previous V-buffer;
  - motion and dual motion;
  - duplication map;
  - pairing textures.
- **Per-dispatch constants** (slot index t, fwd/inv flag, pairing-texture transform, pass id) go in **immediates**. `var<immediate>` ≤ 64 B works in compute [M-plat §2.4].
- Frame data goes in 1 uniform (≤ 64 KiB).

### 4.2 Physical buffers (merged scheme)

| Buffer | Contents | Written by (GPU) | Size at 720p / 1080p |
|---|---|---|---|
| **B1 `bvh`** | TLAS + all BLAS nodes (BVH2 or CWBVH), one array with per-BLAS offsets | – (CPU upload; TLAS via `writeBuffer`) | Sponza: 8.4 MB (BVH2) / 3.5 MB (CWBVH) |
| **B2 `tris`** | Leaf-ordered triangle records (v0\|primId, e1, e2) | – | 12.6 MB |
| **B3 `records`** (read-only arena, `array<vec4u>`; section offsets in the frame uniform; `buffer_view` in Chrome 153+) | prims (i0, i1, i2, material), vertex attributes (position, octahedral normal, uv), instances (cur + prev 3×4), materials, lights cur + prev, cur↔prev light maps, emissive-triangle alias table, env CDF | – (CPU only) | scene-dependent, ~10–40 MB |
| **B4 `lightTiles`** | 128 × 1024 × 32 B | presample pass only | 4.2 MB |
| **B5 `reservoirs`** (arena: region A = current/temporal, region B = previous/spatial-out; 4 planes of vec4u, 8×8-tiled order) | 2 × 64 B/px | initial (A), temporal resample (A in place), spatial resample (B) | 118 / 265 MB |
| **B6 `slots`** | N × 16 B/px shift results (temporal uses 2 of them) | shift, replay | 44 / 100 MB (N=3) |
| **B7 `queue`** | Counter at offset 0 + items (N × 4 B/px capacity) | classify / shift (append) | 11 / 25 MB |
| **B8 `args`** | 3 × u32 indirect arguments (`STORAGE \| INDIRECT`) | args pass | 16 B |
| **B9 `suffixOut`** | 2 × 8 B/px (fwd and inv L_o, f16) | suffix passes | 15 / 33 MB |
| **B10 `debug`** | Counter + records for a probe pixel | any | ~1 MB |
| (texture) `vbuf[2]` | rgba32uint: inst, prim, bary unorm16×2, linear depth; + rg32uint octahedral normal + material → **2 × 24 B/px** | primary pass (storage write) | 44 / 100 MB |
| (texture) `motion` | rgba16float (MV + dual MV) | primary pass | 7 / 17 MB |
| (texture) `dupMap` | r32uint (or r8unorm) | duplication-map pass | 4 / 8 MB |
| (texture) `color` | rgba16float or rgba32float shading output (vector weights) | spatial resample | 15 / 33 MB (rgba32f) |
| (texture) `accum` | rgba32float `read_write` (tier 2) | resolve | 15 / 33 MB |
| (texture) pairing ×3 | rg8sint 254² / 230² / 210² | CPU | < 0.3 MB |
| (texture) material arrays ≤ 8, env map | see platform report §3 | CPU | scene |

### 4.3 Bind groups

| Group | Contents | Notes |
|---|---|---|
| **G0 "scene + frame"** (explicit, static per scene) | uniform `Frame`; SB B1, B2, B3, B4 (all read-only); sampled material arrays; env map; samplers | **4 SB**. The presample pass uses a G0 *variant* with a dummy at B4, because B4 is its write target |
| **G1 "ReSTIR state"** (ping-pong variants prebuilt; one layout per pass family) | B5 (ro or rw), B6 (ro or rw), B7 (ro or rw); sampled `vbuf[cur]`, `vbuf[prev]`, `motion`, `dupMap`, pairing textures | ≤ 3 SB |
| **G2 "pass outputs"** | storage textures (`vbufOut`, `motionOut`, `dupOut`, `color`, `accum`, canvas `bgra8unorm`); B8 or B9 when needed | ≤ 2 ST |
| **G3 "debug"** | B10 | 1 SB |

### 4.4 Per-pass binding counts

SB = storage buffers, ST = storage textures.

| Pass | G0 SB | G1 SB | G2 | G3 | **Total SB** | **ST** | Sampled textures (≤ 48) |
|---|---|---|---|---|---|---|---|
| presample_lights | 4 (dummy at B4) | – | B4 rw | debug | **6** | 0 | 0 |
| primary V-buffer (compute; or raster + `primitive-index`) | 4 | – | – | debug | **5** | 2 (`vbufOut`, `motionOut`) | material arrays (alpha test) |
| initial path tree | 4 | B5 rw | – | debug | **6** | 0 | vbuf[cur], materials, env |
| temporal shift (reconnection pairs + fused ballot append) | 4 | B5 ro, B6 rw, B7 rw | – | debug | **8** | 0 | vbuf[cur], vbuf[prev], motion, dupMap, materials |
| args | (empty G0) | B7 ro | B8 rw | – | **2** | 0 | 0 |
| temporal replay + reconnect (indirect) | 4 | B5 ro, B6 rw, B7 ro | – | debug | **8** | 0 | vbufs, materials |
| suffix re-trace fwd / inv (light-change frames only) | 4 | B5 ro | B9 rw | debug | **7** | 0 | vbufs, materials |
| temporal resample (Talbot) | 4 | B5 rw (in place), B6 ro | B9 ro | debug | **8** | 0 | dupMap |
| spatial shift (+ append) | 4 | B5 ro, B6 rw, B7 rw | – | debug | **8** | 0 | vbuf, pairing ×3, materials |
| spatial replay + reconnect (indirect) | 4 | B5 ro, B6 rw, B7 ro | – | debug | **8** | 0 | vbufs, materials |
| spatial resample + shade | 4 | B5 rw (read A, write B), B6 ro | – | debug | **7** | 1 (`color`) | vbuf, pairing ×3 |
| duplication map | (empty G0) | B5 ro | – | debug | **2** | 1 (`dupOut`) | 0 |
| resolve / present | (empty G0) | – | – | debug | **1** | 2 (`accum` rw, canvas) | `color` |
| ensemble statistics (validation mode) | (empty G0) | – | Welford buffer rw | debug | **2** | 0 | `color` |

- **Maximum 8 of 10 storage buffers, 2 of 8 storage textures, 4 of 4 bind groups.** There are 2 SB of headroom, for example to split `records` if the offset bookkeeping gets unwieldy, or for an ensemble-mode buffer.
- **Without the `records` arena**, prims, vertices, instances ×2, materials, lights ×2, light maps, light CDF and env CDF would be **≥ 10 extra buffers**. The arena is mandatory.
- **Temporal resample binds B5 `read_write`** because it reads region B (previous) and writes region A in place. That is legal only because both regions live in one buffer bound once.
  - If A and B were separate buffers, it would need 2 bindings: still ≤ 10.
- **Fusing classify into the shift kernel** (as EvanLuo42's port does, reference-code report §2.3) removes one pass. The append must stay in **subgroup-uniform control flow**: no early `return` before `append()`. The mock's classify pass shows the pattern (§10.3).

---

## 5. Memory

### 5.1 Screen-resolution buffers (production layout from §4)

Pixel counts: P₇₂₀ = 921,600; P₁₀₈₀ = 2,073,600. MB = 10⁶ B.

| Item | B/px | 720p (MB) | 1080p (MB) | Notes |
|---|---|---|---|---|
| Reservoirs, **2 buffers** (A current/temporal in place, B previous/spatial-out) | 128 | 117.96 | 265.42 | Enhanced's double buffering: "2 × 64 = 128 B/px, 265 MB at 1080p" [SRC, enhanced-paper report §6.2] |
| (alternative) 3 reservoir buffers (init, temporal-out, previous) | 192 | 176.95 | 398.13 | Only if temporal resample cannot run in place |
| Pairing / shift slots, N=3 × 16 B (F·J f16×3, J f32, flags); temporal reuses 2 of them | 48 | 44.24 | 99.53 | N=2 → 32 B; N=1 → 16 B |
| Replay queue (counter + N × 4 B items) | 12 | 11.06 | 24.88 | Shared by the temporal and spatial queues |
| **Replay state, fused replay + reconnect (recommended)** | 0 | 0 | 0 | Measured faster (§2.2) |
| (alternative) replay state for separate replay → reconnect passes: 32 B × N pairs (Code22 `ReconnectionData` compressed) | 96 | 88.47 | 199.07 | Code22 uses 256 B/px (reference-code report §1.3) |
| V-buffer current + **previous frame** (2 × 24 B) | 48 | 44.24 | 99.53 | Previous is needed for the inverse temporal shift |
| Motion + dual motion vectors (rgba16float) | 8 | 7.37 | 16.59 | |
| Duplication map (r32uint; r8 = 1 B) | 4 | 3.69 | 8.29 | |
| Suffix re-trace output (fwd + inv, f16×3 each, padded) | 16 | 14.75 | 33.18 | Moving-light frames |
| Shading colour (rgba32f) | 16 | 14.75 | 33.18 | rgba16f → 8 B |
| Accumulation (rgba32f `read_write`) | 16 | 14.75 | 33.18 | Progressive / validation |
| **Total, recommended (2 reservoirs, fused replay)** | **296** | **272.8** | **613.8** | |
| Total with 3 reservoir buffers | 360 | 331.8 | 746.5 | |
| Total with 3 reservoirs + separate replay state | 456 | 420.2 | 945.6 | |

### 5.2 Resolution-independent buffers

| Item | Size | Source / notes |
|---|---|---|
| Light tiles 128 × 1024 × 32 B | 4.19 MB | Can be packed to 16 B → 2.1 MB |
| Lights current + previous (1 k lights × 64 B × 2) + cur↔prev maps | ≈ 0.14 MB | Previous copy required for exact temporal MIS (restir-practice report §3.3) |
| Scene, Sponza, BVH2 | 28.3 MB | nodes 8.4 + tris 12.6 + prims 4.2 + verts 3.1 [M] |
| Scene, Sponza, CWBVH | 23.4 MB | [M] |
| Previous TLAS / instances (only for moving geometry) | ≈ size of TLAS + 96 B per instance | Not needed for static geometry |
| Material textures | 0.3–1.1 GB | Platform report §3.2 |

### 5.3 Ensemble validation mode (validation-harness report §3.6)

E independent chains render in one dispatch.

| Item | B/px |
|---|---|
| Reservoirs 2 × 64 | 128 |
| Slots N=3 | 48 |
| Queue | 12 |
| V-buffer cur + prev | 48 |
| Motion | 8 |
| Duplication map | 4 |
| Colour | 16 |
| **Welford mean + M2 (2 × rgba32f)** | 32 |
| Suffix output | 16 |
| **Total** | **312** |

| Ensemble | Pixels | Memory | Largest single buffer (reservoir arena) |
|---|---|---|---|
| E = 64 × 256² | 4,194,304 | **1.31 GB** | 537 MB → needs the raised `maxStorageBufferBindingSize` |
| E = 16 × 256² (or 4 × 512²) | 1,048,576 | 327 MB | |
| E = 64 × 128² | 1,048,576 | 327 MB | |

- **Time** [I]: 4.19 M px is 4.55× the 720p pixel count, so one ensemble frame takes ≈ 4.55 × 51–61 ms ≈ **230–280 ms**.
  - Split it into several submits (one per pass family or per member group) to stay well under the GPU watchdog (platform report §7).
  - R = 1024 runs × T = 60 frames at E = 64 takes 16 batches × 60 × 0.25 s ≈ **4 min per test sequence** [I].
- **Limits.** Every buffer here stays under 4 GiB − 4 once the adapter maximum is requested. On a unified-memory Mac, 1.3 GB of GPU buffers is fine [UNVERIFIED: Chrome imposes no per-tab GPU memory cap below this; watch `device.lost` with reason `destroyed` or OOM errors].

---

## 6. Go / no-go recommendation

### 6.1 Verdict [INFERENCE from §2–§3 measurements]

| Target | Verdict | Evidence |
|---|---|---|
| 1920×1080 real time | **NO-GO** | 132–171 ms per frame (BVH2), ≈ 110 ms (CWBVH, [I]) |
| 1280×720 at ≥ 30 fps | **NO-GO** | Leanest measured config (CWBVH, 2 bounces, N=1) = 37.4 ms. The initial path tree alone is 20 ms |
| 1280×720 "interactive quality" 15–27 fps | **GO** | 37–51 ms (CWBVH); 43–55 ms with moving lights |
| **960×540 internal + upscale to canvas, ≥ 30 fps** | **GO (default)** | Full Enhanced graph (3 bounces, N=3, CWBVH) = 28.2 ms; 2 bounces, N=1 = 20.9 ms; moving lights 24.6–31.4 ms |
| Progressive / validation (any resolution) | **GO** | Batched submits; temporal off (spatial-only accumulation per GRIS §6.5) or ensemble mode |

### 6.2 Recommended defaults for interactive mode

1. **Internal resolution:** 960×540 with temporal upscaling to the canvas (a Web-RTRT FSR2 port or TAA-U; platform report §5.7). Offer a 720p quality toggle, and a dynamic-resolution controller that targets 33 ms using the §7 cost model and the live queue counts.
2. **Max bounces:** **2** by default (MAXB=2: NEE at x₁, x₂, x₃); 3 as the quality setting.
   - Keep **RR at initial sampling only**: 1.44× on the initial pass [M]. It is also the paper's largest single win.
   - NEE RIS with light-tile candidates (32 at x₁) costs only +2.6 ms at 720p. Use it when there are many lights; with ≤ 8 user lights, plain power-based selection is enough.
3. **Neighbours:** paired spatial reuse (Enhanced §3) with **N=3 at 540p** (8.6 ms spatial with CWBVH) and **N=1–2 at 720p**.
   - Paired reuse is essential: unpaired pairwise MIS would double the shift count, because each shift is about one visibility ray [M].
4. **Temporal:** on, with 2 shifts per pixel (forward + inverse). Use the Talbot or contribution-MIS form (restir-practice report §3.1): both cost 2 shifts.
5. **Prefix replay:** a **separate, compacted, indirect pass with the reconnection fused in** (1.39–1.40× over inline; 0 B of replay state).
   - Classify can be fused into the reconnection-shift kernel as a subgroup-ballot append.
   - **Instrument the queue count every frame.** Frame time moves by ≈ 21 ns × (compacted pairs): at 720p that is ≈ 5.8 ms per 10% of spatial N=3 pairs.
6. **Suffix re-trace (moving lights):** a **separate pass, run only on light-change frames**.
   - Prefer **forced NEE-light reconnection plus storing (lightId, uv)** (Enhanced §6.2.3; restir-practice report §3.5). NEE-final paths then need only the 1-ray NEE-only update (3.1 ms per direction at 720p with CWBVH), or no re-trace at all if the light-local sample is re-evaluated in the shift.
   - Reserve the 1–2-bounce suffix re-trace (9–17.5 ms per direction at 720p) for paths whose suffix passes through a light change. Compact these (same ballot/indirect pattern), or amortize ReSTIR-GI-style.
7. **CWBVH:** **recommended (v1.5), not required for the 540p go.**
   - It gives 1.19–1.22× per frame, 1.3–1.6× on the any-hit-heavy shift and NEE passes, and 0.41× the node memory.
   - It is cheap to build: 165 ms collapse + encode on top of the BVH2 build [M].
   - It *is* required to get close to 30 fps at 720p, and even there it falls short.
   - Build BVH2 with a maximum leaf size of 3 so leaves map onto CWBVH leaf slots.
8. **Compile:**
   - Create all pipelines at startup with `createComputePipelineAsync` **in parallel**: ≈ 4× concurrency, so ~12–15 pipelines of 10–60 KB compile in ~1–2 s [I from §2.8].
   - Keep **one copy of the material-switch per kernel where possible**: route the two reconnection BSDF evals through one call site, as in E2e (−17% compile time, same runtime).
   - Do **not** collapse traversal into one site.
   - Keep `override` sets few, since each is a full compile.
9. **Validation-mode configuration** (independent of interactive defaults):
   - f32 everywhere, the 128 B uncompressed reservoir;
   - duplication map off (biased, Enhanced §5);
   - progressive accumulation without temporal reuse, or ensemble mode with ≤ 16 × 256² members when memory or time is tight.

---

## 7. Cost model for re-budgeting

Fitted from §2 at 720p; all unit costs are **[M]**. At 540p multiply unit costs by ≈ 1.05–1.1: pixel-count scaling is slightly sub-linear (e.g. visibility 7.14 ms at 540p vs 11.93 ms at 720p, ratio 0.60 against a pixel ratio of 0.5625).

`T_frame ≈ P · [ t_init(MAXB, RR) + (2 + N) · t_shift + (2 + N) · f_r · t_rep + t_small ] + L · 2 · P · t_suffix(NB)`

Symbols:
- P = pixel count.
- N = spatial partners.
- f_r = fraction of pairs whose rc vertex is beyond x₂ or absent (needs replay). Unknown for real scenes until measured.
- L = 1 on light-change frames, else 0.

Unit costs (ns per pixel, per pair or per item):

| Term | BVH2 | CWBVH |
|---|---|---|
| t_init, 3 bounces, RR | 28 | 26 |
| t_init, 3 bounces, no RR | 40 | 34 |
| t_init, 2 bounces, RR | ≈ 22 [I] | 22 |
| t_shift (per pair; shift pass, idle lanes included) | 4.65 | ≈ 3.3 |
| t_rep (per compacted pair, mean 2 bounces + reconnection) | 21 | 19 |
| t_suffix, NEE only | 4.8 | 3.4 |
| t_suffix, NB=1 | 12.3 | 9.7 |
| t_suffix, NB=2 | 19 | ≈ 15 [I] |
| t_small (2 resamples, dup map, resolve, classify ×2, args ×2) | ≈ 2.5 | ≈ 2.5 |

**Check** (720p, BVH2, N=3, f_r = 0.1):

`921,600 × (28 + 5·4.65 + 5·0.1·21 + 2.5) ns = 921,600 × 64.25 ns ≈ 59.2 ms` (measured: 59.2–60.6 ms).

**30 fps budget at 540p (518,400 px): 33.3 ms ⇒ ≤ 64 ns/px.**
- The full config (CWBVH, 3 bounces, RR, N=3, f_r = 0.1) needs 26 + 5·3.3 + 5·0.1·19 + 2.5 ≈ 54.5 ns/px → 28.2 ms ✓.
- It stays within budget up to f_r ≈ 0.2, **or** with moving lights of up to NEE-only suffixes (+6.8 ns/px).

---

## 8. Caveats, risks and open questions

1. **Replay fraction f_r is the biggest unknown.**
   - The mock uses synthetic 0/10/30%. The real value depends on the Enhanced footprint criterion (c = 0.02, α_min = 0.2) and scene glossiness.
   - In mostly diffuse scenes like Sponza it is probably ≤ 10%; glossy interiors can be far higher [UNVERIFIED].
   - **Measure it in the real renderer** from the queue counters and feed it to the §7 model and the dynamic-resolution controller.
2. **Not modelled:**
   - TLAS / instance traversal: extra transform and a sentinel stack entry per BLAS entry;
   - alpha-tested any-hit (Sponza has alpha-masked foliage, which makes shadow and visibility rays more expensive);
   - emissive-triangle area lights hit by BSDF rays, and env-map sampling;
   - exact lobe bookkeeping and invertibility checks during replay (adds ALU; may terminate replays earlier);
   - large texture working sets (the single test showed +2% / +12%).

   Expect **+10–25%** on the ray-bound passes from these combined [UNVERIFIED].
3. **Neighbour pattern.** The hash-based uniform-disk partners (1–30 px) stand in for pairing textures (Gaussian σ = 16 px). Reservoir-fetch locality is similar, and visibility cost was insensitive to radius [M, §2.1].
4. **Timestamp quantization** (65.5 µs) makes sub-0.1 ms passes read as 0 or 0.066 ms. It is irrelevant to the budget.
5. **Thermal behaviour.** Measurements are short bursts, not sustained minutes. M5 Pro sustained GPU clocks may be lower [UNVERIFIED].
6. **Chromium 152 vs Chrome 154.** Chrome 154 adds `buffer_view`, which would make the `records` arena typed and cleaner. The traversal performance should be identical (same Tint/Metal path) [INFERENCE].
7. **CWBVH.** Only a straightforward port was tested. A better collapse, compressed triangles or stack tweaks might add more [UNVERIFIED]. The measured gain is below the 2× often quoted, so budget only 1.2× per frame.
8. **Compile.**
   - The material library in the compile test is synthetic, with distinct constants per material.
   - Real Principled/OpenPBR code with texture fetches is larger per material.
   - Metal compiler inlining heuristics may change with macOS updates [UNVERIFIED].
   - The in-session cache (0 ms repeat) says nothing about cross-session cache persistence (platform report §5.3: Dawn blob cache).
9. **Moving lights make exact temporal MIS expensive.** Both suffix directions under S_t and S_{t−1} are required for exact results (restir-practice report §3.4–3.6). Any cheaper approximation must be flagged as biased in validation.
10. **Denoiser and upscaler were not measured.** SVGF-class filters and an FSR2-like upscaler plausibly add 2–5 ms at 540p–720p [UNVERIFIED]. Reserve 3–5 ms of the 33 ms budget.

---

## 9. Sources

**Earlier reports** (under `/Users/mark.boss/.claude/plans/`):

| Report | File | Sections used |
|---|---|---|
| Platform | `do-a-deep-dive-shimmering-ritchie-agent-a64ceeae6903be301.md` | §0 key numbers; §2.1 limits; §2.4 immediates; §4.1 throughput (600 / 192 / 610 / 212 Mrays/s; 77.3 ms at 1080p); §4.2 CWBVH claim; §4.3 BVH2 traversal kernel reused here; §5.1–5.7 pass graph, bindings, compile times, reservoir layout, compaction code, the 45–65 ms estimate; §7 timestamps; §8.3 ray offsets; §8.5 RNG |
| Reference code | `...-a3b89657b22d15f34.md` | §1.3 `ReconnectionData` 40 B / `PixelReconnectionData` 256 B; §1.11 item 7 (≈ 900 MB at 1080p in Code22); §2.3 EvanLuo42 pass structure (shift → args → indirect replay → resample); §2.4 C2-Renderer compile-time lesson; §3 WGSL porting pitfalls; §4.1–4.4 recommended passes, reservoir packing, RNG, shift skeleton |
| Enhanced paper | `...-a91ab1d861819b276.md` | §2.6 paired pre-pass / resample split and slot memory; §6.2 64 B reservoir, 265 MB at 1080p; §6.3 compaction gains; §7 pass table; §8.2 Table 1 (15.53 ms total; 6.77 / 3.20 / 3.73 / 1.83); §8.3–8.4 per-scene and resolution scaling |
| ReSTIR practice | `...-a668fe13fb61070e8.md` | §3.1 temporal MIS forms; §3.3–3.6 previous-frame state and suffix re-trace cost; §3.11 recommended temporal pipeline |
| Validation harness | `...-a26c6df2c3f5e4af0.md` | §3.6 ensemble mode (E = 64 × 256², 537 MB for reservoirs) |

**Papers:**
- Lin, Kettunen, Wyman 2026, *ReSTIR PT Enhanced*: local `scratchpad/restirpt_enhanced_2026.pdf`, Table 1 and §§3, 5, 6.2 (via the enhanced-paper report).
- Lin et al. 2022, *GRIS*: `scratchpad/gris_sig22.pdf` §6.5, §8.
- Ylitie, Karras, Laine 2017, *Efficient Incoherent Ray Traversal on GPUs Through Compressed Wide BVHs* (CWBVH layout and traversal). The WGSL port in §10.2 follows the structure of jbikker/tinybvh (MIT) `BVH8_CWBVH` and `tiny8.comp`, re-derived from memory. Byte layout: n0 = origin + exponents + imask; n1 = child base, triangle base, meta ×8; n2–n4 = quantized lo/hi.

**Scene:** https://raw.githubusercontent.com/KhronosGroup/glTF-Sample-Assets/main/Models/Sponza/glTF/Sponza.gltf and `Sponza.bin`.

**WebGPU spec:** buffer usage scopes (writable usage conflicts) and pipeline-layout storage-buffer limits. The aliasing behaviour was confirmed by a validation error in this session.

---

## 10. Reproduction material (load-bearing code, verified this session)

### 10.1 Harness shape (JS, in page)

```js
// one compute pass per stage, timestampWrites -> resolve -> map; median of reps
B.time = async (steps, reps=8, warm=2, W, H, opt={}) => {
  for (let r = 0; r < warm + reps; r++) {
    B.setUni(W, H, 100 + r, opt);                       // bump frame index -> new neighbours
    const enc = dev.createCommandEncoder();
    steps.forEach((st, i) => st.run(enc, {querySet: B.qs, beginningOfPassWriteIndex: 2*i, endOfPassWriteIndex: 2*i+1}));
    enc.resolveQuerySet(B.qs, 0, 2*steps.length, B.qbuf, 0); enc.copyBufferToBuffer(B.qbuf, 0, B.rbuf, 0, 16*steps.length);
    dev.queue.submit([enc.finish()]);
    await B.rbuf.mapAsync(GPUMapMode.READ, 0, 16*steps.length); /* read BigUint64 pairs, unmap, keep (end-begin)/1e6 */ }
};
// compacted replay step pair:
//  {run: enc => { enc.clearBuffer(queue, 0, 16); pass(classify) }}, {run: pass(args,[1])}, {run: pass(replay, null, argsBuffer)}  // dispatchWorkgroupsIndirect
```

Uniform: `struct Uni { camPos, camFwd, camRight, camUp: vec4f, res: vec4u (W, H, frame, nLights), p: vec4f (rmax, rmin, footprintThr, replayFootprintThr), q: vec4u (replayPct, arena offsets...), lights: array<vec4f, 8> }`.

G-buffer miss sentinel: `const GMISS: u32 = 0x7f7fffffu;`. `bitcast<f32>(0xffffffffu)` is a compile error (constant NaN) [M].

### 10.2 CWBVH: encoder (JS) and traversal (WGSL), verified against BVH2 (99.99% identical hits)

**Encoder** (per BVH8 node, BFS order so that the internal children of a node are contiguous):

```text
children = [left, right] of the BVH2 node; while (<8 and any interior child): replace the interior child with the largest surface area by its two children
lo/hi = union of the child boxes; e[a] = ceil(log2((hi[a]-lo[a])/255)) as signed int8 (bump until 2^e*255 >= extent); scale = 2^e
slot assignment: cost(child, s) = dot(sign_s, childCentroid - nodeCentroid), sign_s = (s&4?-1:1, s&2?-1:1, s&1?-1:1); greedy lowest cost first
per slot: qlo = floor((clo-lo)/scale), qhi = ceil((chi-lo)/scale), clamped to [0, 255]
  interior child: imask |= 1<<s; meta = 0x20 | (24 + s); node index = baseChild + (#interior children in earlier slots)
  leaf child (<=3 tris): meta = (((1<<n)-1) << 5) | (triOffset relative to baseTri); its triangles appended to the node's triangle block
words (little endian, 20 x u32): [p.x, p.y, p.z, e.x|e.y<<8|e.z<<16|imask<<24, baseChild, baseTri, meta0..3, meta4..7,
  qlo_x0..3, qlo_x4..7, qlo_y0..3, qlo_y4..7, qlo_z0..3, qlo_z4..7, qhi_x0..3, qhi_x4..7, qhi_y0..3, qhi_y4..7, qhi_z0..3, qhi_z4..7]
triangles re-emitted in CWBVH order as (v0|primId, e1, e2), 48 B
```

**Traversal** (replaces `traverse` in the common code; `cwn: array<vec4u>` at `@group(0) @binding(1)`):

```wgsl
var<private> cstack: array<vec2u, 32>;
fn ext8(x: u32, i: u32) -> u32 { return (x >> (8u * i)) & 0xffu; }
fn traverse(O: vec3f, D: vec3f, tmaxIn: f32, anyHit: bool) -> Hit {
  let rD = vec3f(safe_rcp(D.x), safe_rcp(D.y), safe_rcp(D.z));
  var h = Hit(tmaxIn, 0.0, 0.0, MISS);
  let oct = select(0u, 4u, D.x < 0.0) | select(0u, 2u, D.y < 0.0) | select(0u, 1u, D.z < 0.0);
  let octinv = (7u - oct) * 0x01010101u;
  var ng = vec2u(0u, 0x80000000u); var tg = vec2u(0u, 0u); var sp = 0u;
  loop {
    if (ng.y > 0x00ffffffu) {
      let hits = ng.y; let imask = ng.y & 0xffu;
      let cbi = firstLeadingBit(hits);
      ng.y &= ~(1u << cbi);
      if (ng.y > 0x00ffffffu) { if (sp < 32u) { cstack[sp] = ng; sp += 1u; } }
      let slot = (cbi - 24u) ^ (octinv & 0xffu);
      let rel = countOneBits(imask & ~(0xffffffffu << slot));
      let ni = (ng.x + rel) * 5u;
      let n0 = cwn[ni]; let n1 = cwn[ni + 1u]; let n2 = cwn[ni + 2u]; let n3 = cwn[ni + 3u]; let n4 = cwn[ni + 4u];
      let e = vec3i(bitcast<i32>(n0.w << 24u) >> 24u, bitcast<i32>(n0.w << 16u) >> 24u, bitcast<i32>(n0.w << 8u) >> 24u);
      let adj = vec3f(bitcast<f32>(u32(e.x + 127) << 23u), bitcast<f32>(u32(e.y + 127) << 23u), bitcast<f32>(u32(e.z + 127) << 23u)) * rD;
      let orig = (bitcast<vec3f>(n0.xyz) - O) * rD;
      var hitmask = 0u;
      for (var hf = 0u; hf < 2u; hf++) {
        let second = hf == 1u;
        let meta4 = select(n1.z, n1.w, second);
        let isInner4 = (meta4 & (meta4 << 1u)) & 0x10101010u;
        let innerMask4 = ((isInner4 << 3u) >> 7u) * 0xffu;
        let bitIndex4 = (meta4 ^ (octinv & innerMask4)) & 0x1f1f1f1fu;
        let childBits4 = (meta4 >> 5u) & 0x07070707u;
        let lox = select(n2.x, n2.y, second); let hix = select(n3.z, n3.w, second);
        let loy = select(n2.z, n2.w, second); let hiy = select(n4.x, n4.y, second);
        let loz = select(n3.x, n3.y, second); let hiz = select(n4.z, n4.w, second);
        let qlx = select(lox, hix, rD.x < 0.0); let qhx = select(hix, lox, rD.x < 0.0);
        let qly = select(loy, hiy, rD.y < 0.0); let qhy = select(hiy, loy, rD.y < 0.0);
        let qlz = select(loz, hiz, rD.z < 0.0); let qhz = select(hiz, loz, rD.z < 0.0);
        for (var i = 0u; i < 4u; i++) {
          let tmn = vec3f(f32(ext8(qlx, i)), f32(ext8(qly, i)), f32(ext8(qlz, i))) * adj + orig;
          let tmx = vec3f(f32(ext8(qhx, i)), f32(ext8(qhy, i)), f32(ext8(qhz, i))) * adj + orig;
          let cmin = max(max(max(tmn.x, tmn.y), tmn.z), 0.0);
          let cmax = min(min(min(tmx.x, tmx.y), tmx.z), h.t);
          if (cmin <= cmax) { hitmask |= ext8(childBits4, i) << ext8(bitIndex4, i); }
        }
      }
      ng = vec2u(n1.x, (hitmask & 0xff000000u) | (n0.w >> 24u));
      tg = vec2u(n1.y, hitmask & 0x00ffffffu);
    } else { tg = ng; ng = vec2u(0u); }
    loop {
      if (tg.y == 0u) { break; }
      let ti = firstLeadingBit(tg.y); tg.y &= ~(1u << ti);
      let v = (tg.x + ti) * 3u;
      let v0 = tris[v]; let e1 = tris[v + 1u].xyz; let e2 = tris[v + 2u].xyz;
      let hv = cross(D, e2); let a = dot(e1, hv); let f = 1.0 / a;
      let s = O - v0.xyz; let bu = f * dot(s, hv); let qv = cross(s, e1);
      let bv = f * dot(D, qv); let d = f * dot(e2, qv);
      if (bu >= 0.0 && bv >= 0.0 && bu + bv <= 1.0 && d > 0.0 && d < h.t) {
        h = Hit(d, bu, bv, tg.x + ti); if (anyHit) { return h; } }
    }
    if (ng.y <= 0x00ffffffu) { if (sp > 0u) { sp -= 1u; ng = cstack[sp]; } else { break; } }
  }
  return h;
}
```

Notes on the traversal:
- The 32-entry group stack silently drops pushes on overflow, as tinybvh's does. Add a debug counter in debug builds.
- The quantized boxes are conservative, since lo is floored and hi is ceiled.

### 10.3 Shift-kernel structure used for (c), with the replay pass (b) as the indirect variant

```wgsl
// per pixel p: own reservoir -> x_k rebuilt once; N partners; rcLen<=2 reconnect inline, else leave to the compacted replay pass
let r = loadRes(p);                                   // 4 x vec4u = 64 B
let xk = loadPrim(r.rcPrim, r.bary);                  // prims[prim] -> 3 x verts -> pos, n, ng, material
for (var s = 0u; s < NS; s++) {
  let q = pixOf(neighbor(g.xy, pcg(p * 3u + s + u.res.z * 0x9e37u), u.p.y, u.p.x));
  if (rcLen <= 2u) { slots[p * NS + s] = reconnect(replay(q, r.seed, 0u), xk, r); }   // 2 evals + 1 any-hit ray + 16 B write
}
// classify (separate or fused; must be in subgroup-uniform control flow):
for (var s = 0u; s < NS; s++) { append(inb && rcLen > 2u, (p << 2u) | s); }   // ballot + exclusive add + 1 atomicAdd per subgroup
// replay pass: @workgroup_size(64), dispatchWorkgroupsIndirect; item -> (p, s); replay(q, seed, rcLen-2) then reconnect(...) -> slots[p*NS+s]
```
