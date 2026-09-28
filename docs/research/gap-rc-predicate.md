# Gap report `rc-predicate-invertibility`: one normative reconnection predicate, offset-path invertibility checks, and paired-reuse rules for a WebGPU ReSTIR PT (Enhanced) renderer

> **Where this report lives.** The orchestrator asked for
> `/private/tmp/claude-502/-Users-mark-boss-Dev-WebGPURestirPT/4e489933-ea73-491f-b7b7-6d0aab1ffce1/scratchpad/research/gap-rc-predicate-invertibility.md`.
> This agent ran in **plan mode**, where the only writable file is this plan file, so the report is here.
> Copy it verbatim to the research path if needed. Nothing was written to the scratchpad or the project directory.
> No repositories were cloned. All code was streamed read-only to stdout (`curl … | cat -n`).
>
> **Tags.**
> - **[SRC]** a source states it; the location is given.
> - **[CODE]** a reference implementation does it; `file:line` is given.
> - **[INFERENCE]** my own derivation or design decision.
> - **[UNVERIFIED]** plausible but not confirmed.
>
> **Citation keys.**
> - **P-** Enhanced paper (Lin, Kettunen, Wyman 2026). Page images `scratchpad/pages/enhanced/p-NN.png`, read directly: p-04 (§2.3, Eq. 2), p-06 to p-08 (§4, Eqs. 4–6, fn. 4–6), p-11 (§6.2.3–6.2.4), p-12, p-14.
> - **S-** Enhanced supplemental. Read from the harness's cached text extraction `~/.claude/projects/-Users-mark-boss-Dev-WebGPURestirPT/4e489933-ea73-491f-b7b7-6d0aab1ffce1/tool-results/bkusnhdet.txt`, lines 1–400 (§1–§6).
> - **G-** GRIS / ReSTIR PT 2022. Page images `scratchpad/pages/gris/p-13.png` (§7.1–7.5), `p-14.png` (§7.5–7.6, §8).
> - **F-** Falcor reference `DQLin/ReSTIR_PT` at commit `8d12332228eb64bc234e27c6f7e0913a926285ab`, `Source/RenderPasses/ReSTIRPTPass/` (BSD-3).
> - **E-** `EvanLuo42/ReSTIR-PT-Enhanced` HEAD (fetched 2026-09-28), `Source/RenderPasses/ReSTIRPTPass/` (BSD-3).
> - **R-** `NVIDIA-RTX/RTXDI-Library` main, `Include/Rtxdi/PT/`. **Proprietary license.** Only semantics are described below; no code is reproduced.
> - Prior reports (all under `/Users/mark.boss/.claude/plans/`, prefix `do-a-deep-dive-shimmering-ritchie-agent-`):
>   - **EP** enhanced-paper `a91ab1d861819b276`, and **EV** its verification `acb243dbea3ce2a29`
>   - **GM** gris-math `a7f08dec9968f0324`, and **GV** its verification `a147516e5d2fb9558`
>   - **RC** reference-code `a3b89657b22d15f34`
>   - **VH** validation-harness `a26c6df2c3f5e4af0`
>   - **WP** webgpu-platform `a64ceeae6903be301`
>   - **CR** critique `a40267d04641507fa`
>   - **CC** cycles-conventions `a2d6cc1a462b20a80`

---

## 0. TL;DR: the normative decisions

1. **Reconnection is a property of a candidate path in its own domain.**
   - Definition: `k*(x̄; D) = min{ k : P_k(x̄; D) }` over the candidate's own pair list.
   - If the list is empty and x̄ ends in an NEE-sampled light vertex, then `k* = d` (forced, P-§6.2.3).
   - Otherwise `k* = ∅`, which means full random replay with J = 1.
   - **A shift T_{S→D}(x̄) = ȳ is defined only if `k*(ȳ; D) = k*(x̄; S)`** (G-§7.4). Every other rule below exists to make this computable and bit-stable.
2. **One pair predicate** `P_k = D_k ∧ R_k ∧ F_k ∧ I_k` for the pair (x_{k−1}, x_k), k ≥ 2:
   - **D_k (delta).** The lobe used at x_{k−1} is non-delta. If the path continues from x_k by BSDF sampling, that lobe is non-delta too.
   - **R_k (roughness).** Perceptual roughness of the lobe used at x_{k−1} satisfies α(x_{k−1}, ℓ_{k−1}) ≥ α_min = 0.2. Lambert counts as 1, delta as 0.
   - **F_k (ray footprint).** `t²/(p̄_{k−1}(ω_{k−1})·|cos θ^g_{x_k}|) ≥ thr(D)`. It is +∞ for the environment.
   - **I_k (inverse footprint).** `t²/(p̄_k(ω_k)·|cos θ^g_{x_{k−1}}|) ≥ thr(D)`. It is skipped when x_k is a light or environment vertex, or when x_k's BSDF is a single Lambert lobe.
   - `thr(D) = τ·R²_pri(D)`, with `τ = c/100 = 2·10⁻⁴` and `R²_pri(D) = ‖x₁−x₀‖²·4π/|⟨n^g_{x₁}, ω̂_{x₁→x₀}⟩|` of **that domain**.
   - p̄ is the **marginal** (all-lobe) solid-angle sampling pdf, conditioned on the path's own incoming direction.
   - Every cosine uses the **geometric** normal and is taken as an absolute value.
3. **NEE-final candidates.** p̄_k for the pair (x_{d−2}, x_{d−1}) is the all-lobe **BSDF** sampling pdf of the NEE direction at x_{d−1}. This is the same number the NEE MIS weight uses.
   - In the Jacobian, the p_k factor for an NEE suffix is 1.
   - If no pair passes, reconnect to the NEE light vertex (forced, no test).
4. **No lobe revocation.** A pair (x_{k−1}, x_k) is decided only once the event that leaves x_k in *that candidate* is known:
   - the BSDF sample, for continuing and emitter-hit candidates;
   - the NEE direction, for the NEE candidate at x_k.
   - An NEE candidate's rc index must never depend on the BSDF sample drawn at the same vertex afterwards.
5. **k = 2 (rc = x₂, Falcor's `rcVertexLength == 1`) is not special.** The pair (y₁, x₂) uses the same P_k with the **copied** lobe ℓ₁ at y₁. Falcor's `hasRoughComponent(sd, 1.0)` is a lobe-existence test belonging to its two-vertex rule. It is replaced by "the joint pdf of the copied lobe at y₁ is > 0" plus R_k.
6. **Offset-path checks for base rc k:**
   - **(O1)** every replayed pair j < k fails P_j. The last pre-rc pair (y_{k−2}, y_{k−1}) is tested with the **reconnection** event at y_{k−1}: the copied lobe, direction ω' toward x_k, and p̄^y(ω'). For forced NEE, the event at y_{d−1} is the NEE event toward the shared light vertex.
   - **(O2)** (y_{k−1}, x_k) passes P_k with **recomputed** p̄^y_{k−1}(ω') and p̄^y_k(ω_k | from y_{k−1}), and with the destination's thr.
   - **(O3)** if k = ∅, no pair of ȳ passes, including the terminal emitter or environment pair, and ȳ ends with the same technique at the same length.
   - R_pri comes from the **destination** domain: its own jittered primary hit and camera, and the **previous** frame's camera and V-buffer for temporal inverse shifts.
   - Store `thr` per pixel per frame in the G-buffer and keep the previous frame's copy.
7. **Thresholds are fixed** (τ = 2·10⁻⁴, α_min = 0.2) by default. RTXDI-style jitter is allowed only if it is a deterministic function of the path's own replay seed.
8. **Paired reuse.**
   - **Acceptance.** `A(p,q) = A₀(G[min(p,q)], G[max(p,q)])` is evaluated once per pair in canonical order and depends only on the G-buffer. S2 never re-evaluates it; it reads slot status codes. `NOT_ACCEPTED ≠ FAILED`.
   - **MIS.** Defensive pairwise MIS (GRIS Eq. 38, |R| = 1) with confidences and `a_c = c_c/|S_c|`. It is computable from slot `F·J` alone, plus the partner's `c` and `p̂`; see §7.3.
   - **On selecting a shifted sample:** `F ← (F·J)/J` and `jDen ← J·jDen_partner`, i.e. the offset path's own product `p^y_{k−1}(ω',ℓ)·G(y_{k−1}→x_k)·p^y_k(ω_k,ℓ_k)` with **lobe-joint** pdfs.
9. **Bit stability under Chrome's relaxed Metal math.**
   - Base-path and replay predicates cannot be *guaranteed* bit-identical across pipelines. FMA contraction and reassociation are measured to be on (WP §8.1).
   - The design therefore follows a **same-formula principle**. All predicate inputs are recomputed through one shared WGSL module from IDs, stored direction bits and stored per-pixel `thr`, in the base path too.
   - Comparisons are done on integer bit patterns.
   - The tests classify violations by predicate margin: logic bugs must be **zero**; FP-boundary flips are reported, must be rare, and must vanish in a single-pipeline or `strictMath` validation build.
10. **New finding.** EvanLuo42's hybrid shift **never re-evaluates the reconnection gate on the offset path**. There is no O1 and no O2 (`HybridShift.slang:72-110, 232-377`; the gate is used only in `PathSampling.slang`). It is therefore not guaranteed bijective, which is expected bias [INFERENCE, HIGH confidence the check is absent]. Do not copy its shift.

---

## 1. Index conventions (the #1 source of off-by-one bugs)

| Concept | This report / paper (P-§2.3) | Falcor (F-) | RTXDI (R-) | EvanLuo (E-) |
|---|---|---|---|---|
| Camera | x₀ | – | – | – |
| Primary hit | x₁ | `path.length = 0` | bounce depth 1 (`initialBounceDepth = 2` means the first *traced* bounce hits x₂) | surface vertex 1 |
| rc vertex | x_k, k ≥ 2 | `rcVertexLength = k−1` (so 1 means x₂) | `rcVertexLength = k` (so 2 means x₂) | `reconnectionVertex = k` |
| Last vertex | x_d (light, emitter or env) | `pathLength = d−1` = index of the last *scattering* vertex; NEE/escape is implicit | `pathLength = d` (the light vertex is counted) | `getPathLength` (techniqueKey bits 12–15) |
| "Needs replay" | k > 2 or k = ∅ | `rcVertexLength > 1` | `rcVertexLength > 2` | `reconnectionVertex > 2 or invalid` (`HybridShift.slang:473-480`) |

Notation used below:
- ℓ_i is the lobe used at x_i (1 ≤ i ≤ d−1); ℓ_{d−1} = 0 means NEE (BSDF evaluated with all lobes, S-§1).
- ω_i = (x_{i+1}−x_i)/‖·‖.
- `p_i(ω, ℓ)` is the **joint** pdf at x_i of drawing (ω, ℓ) given incoming −ω_{i−1}: `P(ℓ | x_i, −ω_{i−1})·p(ω | ℓ, x_i, −ω_{i−1})`.
- `p̄_i(ω) = Σ_ℓ p_i(ω, ℓ)` is the **marginal**.
- n^g is the unoriented geometric (face) normal.
- `G(a→b) = |⟨n^g_b, (a−b)/‖a−b‖⟩| / ‖a−b‖²`, with the cosine at the **receiving** vertex b. This follows G-Eq.52, F-`Shift.slang:452` and EV-C8. P-p.4 and S-§1 print "normal at y_{k−1}", which is a wording slip.

---

## 2. What the sources actually say or do (evidence digest)

### 2.1 Paper and supplement

- **P-§2.3 (p-04) [SRC].** The base path precomputes the rc vertex as the first pair (x_{k−1}, x_k) meeting the criteria. The 2022 criteria were min(α_{x_{k−1}}, α_{x_k}) ≥ α_min and ‖x_k−x_{k−1}‖ ≥ d_min. Invertibility is checked when connecting y_{k−1} → x_k.
  - Eq. 2 is the hybrid PSS Jacobian; p^x_k := 1 for k = d, and J = 1 without reconnection.
- **P-§4 Eq. 5 (p-07) [SRC].**
  ```
  min( (p^x_{k−1}(ω_{k−1})·G(x_{k−1}→x_k))⁻¹ , (p^x_k(ω_k)·G(x_k→x_{k−1}))⁻¹ ) ≥ (c/100)·‖x₀−x₁‖² / (⟨n_{x₁}, x̂₁x₀⟩/(4π))
  ```
  - c = 0.02, and additionally α_{x_{k−1}} ≥ α_min.
  - §4.1: the bound on ‖Δx_k‖ is stated through the primary footprint R^x̄_pri.
- **P-§4.2 (p-08) [SRC].** The roughness test at x_{k−1} is kept for parallax, curvature and very low roughness, and it "also handles reconnection to environment lights" (Δx_k is unbounded).
  - Footnote 5: sampling-pdf reciprocity is exact for NDF sampling and approximate for VNDF.
  - **Footnote 6: for diffuse or emissive x_k the inverse-footprint test is skipped.**
- **P-§7.2 (p-14) [SRC].** For the single-vertex threshold the authors use ReSTIR PT's default α = 0.2. In Falcor that value is `linearRoughness`, i.e. **perceptual** roughness (GV-C9).
- **P-§6.2.3 (p-11) [SRC].** Replaying NEE-ended paths would need light sampling, so the NEE light vertex is forced to be the rc vertex when no earlier reconnection exists.
- **P-§6.2.4 (p-11), S-§6 [SRC].** Russian roulette is removed from replay and only changes the initial sample's PSS source pdf. The rc predicate must not depend on RR.
- **S-§1 [SRC].** The Jacobian (S-Eq.4) uses the **joint** pdf p(ω_k, ℓ_k | x_k, −ω_{k−1}). Random replay is the identity in PSS.
- **S-§2 [SRC].** S-Eq.6 defines R^x̄_pri and bounds |x₁−y₁| symmetrically by c₁R^x̄_pri and by c₁R^ȳ_pri.
  - S-Eq.18 states the sufficient condition **on ȳ with R^ȳ_pri**.
  - S-Eq.19 derives the x̄ condition "by shift invertibility".
  - So the authors' own derivation evaluates the criterion **in each path's own domain with that domain's R_pri**. This is the source basis for the per-domain R_pri rule of §6.3.
- **S-§3 [SRC].** Footprints use the **marginal** pdf over all lobes (the conditional pdf is "ideal", with negligible difference: FLIP 0.171 vs 0.169). The lobe index is preserved through reconnection.
- **S-§4, S-Eq.26 [SRC].** The PDF proxy for black-box materials is printed as `1/(p^x_{k−1})² ≥ α_min`. See EP §3.6 for the scale doubt; RTXDI uses `pdf ≤ 1/α²`.
- **G-§7.4 (p-13) [SRC].** Reconnection "must happen at this vertex, or it does not happen". If the offset path disagrees on the earliest possible reconnection vertex, the shift is undefined.
- **G-§7.5 (p-14) [SRC].** NEE-sampled vertices count as rough if *at least one* lobe is rough. All light vertices are rough. A reconnection copies the lobe index from the base vertex.

### 2.2 Falcor 2022 (the only reference with complete invertibility checks, but with the old criteria)

- **rc test on arrival** at x_{L+1} (`PathTracer.slang:1157-1193`) [CODE]:
  `canConnect = L ≥ 1 ∧ L < rcLen ∧ far ∧ curRough ∧ lastRough`.
  - With separate lobes, `curRough = hasRoughComponent(sd, 0.2)`, i.e. *any* rough lobe.
  - NEE at x_{L+1} (`:1266-1353`) is streamed with that rc vertex **before** the BSDF sample.
- **Lobe veto** (`:1402-1425`) [CODE]. After sampling at the tentative rc vertex, `seenAsConnectible = !specularBounce || linearRoughness > 0.2`.
  - If false, the rc is revoked (`rcVertexLength = 15`) for all later candidates.
  - The NEE candidate already streamed keeps rc = this vertex. The rule is therefore per-candidate in effect.
- **Replay invertibility** (`:1194-1261`, `:1407-1415`) [CODE].
  - At intermediate replayed vertices the shift is invalidated if the pair would connect: after sampling, `seenAsConnectible ∧ lastRough ∧ far`.
  - At y_{k−1}, if the previous vertex was rough, y_{k−1} would itself become connectible (it will be a "diffuse bounce"), so the shift is invalid.
  - y_{k−1} must be able to produce the copied lobe class: specular group needs `linearRoughness > 0.2`, otherwise `hasRoughComponent(sd, 1.0)` ("has a diffuse lobe").
  - An emitter hit that would be rc-able during replay also makes the shift non-invertible (`:1093-1118`).
- **k = 2 (Falcor `rcVertexLength == 1`)** (`Shift.slang:135-153`) [CODE]. There is no replay (Tp = 1). The destination primary must pass the copied-lobe-class test above, else Tp = 0.
- **Reconnection** (`Shift.slang:434-470, 550-571`) [CODE].
  - A delta event before or after the rc vertex fails (`:438-442`).
  - G uses `faceN` at x_k (`:452`).
  - Near-field distance test on the offset (`:454-458`).
  - The cached Jacobian is overwritten with the destination values (`:568-569`), which is C4 done right.

### 2.3 RTXDI 3.x (semantics only; proprietary)

**`PT/PathReconnectibility.hlsli`** [CODE, paraphrased]
- The primary footprint is computed from the **surface being resampled into**, i.e. the destination. When shifting into the previous frame it uses the **previous camera position**, with a correction for the camera-relative translation.
- The footprint threshold is the primary footprint times `c²`, where c is **Gaussian-jittered** (mean 0.02, relative σ 0.2). The pdf threshold is `1/α²` with α jittered (mean 0.1, relative σ 0.01).
- Both jitters are drawn from the **replay** random stream, so base and offset draw identical thresholds.
- The ray footprint is `1/(G·pdf)` with `G = |n·v|/t²`, and 0 for delta.

**`PT/InitialSamplingPathTracerContext.hlsli`** [CODE, paraphrased]
- **Order.** After tracing to x_j:
  - "last vertex far" means the forward footprint exceeds the threshold (strict >);
  - "last vertex rough" means the pdf of the event at x_{j−1} is at most the pdf threshold and not delta.
- **Before tracing from x_j**, once ω_j is sampled: "current vertex rough for connection" means the inverse footprint, using the **previous** vertex's normal and the sampled pdf at x_j, exceeds the threshold. If current rough ∧ last rough ∧ last far, then rc = x_j. This is decided after the BSDF sample, so no revocation is needed.
- **`RecordNeeLightSample`**, three cases:
  1. an earlier rc exists;
  2. otherwise x_j is connectible, tested with the **BSDF pdf of the NEE direction** as the inverse-footprint pdf;
  3. otherwise the NEE light vertex is the rc (forced, no test).
- **`RecordEmissiveLightSample`** (BSDF-hit emitter): rc = the emitter if there is no earlier rc and the last vertex is far. There is **no roughness test** (EV-O5).

**`PT/HybridShiftPathTracerContext.hlsli`** [CODE, paraphrased]
- **Init** computes the thresholds from the **destination primary surface** and the replay RNG.
- **At every replayed vertex before y_{k−1}:** last rough ∧ last far ∧ inverse footprint > thr makes the shift invalid.
- **At y_{k−1}:**
  - If the base has an rc and the pre-rc pair (y_{k−2}, y_{k−1}) is rough and far, it stores the partial term `t²/|cos at y_{k−2}|`. The pdf is not yet known, because it needs the direction toward x_k.
  - If the base has no rc (emitter ending) and the last vertex is far, the shift is non-invertible ("case 2").

**`PT/HybridShift.hlsli` `ValidateInvertibilityCondition`** [CODE, paraphrased]
- Forward footprint from y_{k−1} with a **recomputed** BSDF pdf toward x_k.
- Pre-rc check: stored partial term / pdf(ω') > thr makes the shift invalid.
- Pdf-roughness at y_{k−1} for mid-path rc only.
- Inverse footprint at x_k with the **stored base pdf** `rcWiPdf`, *not* recomputed. The code comment calls this an approximation, because RTXDI stores no material reference.
- **For the NEE light** (`UpdateReconnectionForRTXDIConnectedLight`): J is the ratio of solid-angle light pdfs, plus the pre-rc check with the BSDF pdf toward the light.

**Consequence for us [INFERENCE].** RTXDI's offset check is structurally what we want: earlier pairs fail, the pre-rc pair is tested with the reconnection direction, and the rc pair passes. But its inverse footprint at x_k is not recomputed, so its shift is only approximately bijective. **Our spec requires recomputing p̄^y_k**, which needs x_k's material (Falcor-style reservoir with IDs), not RTXDI's position-only reservoir.

### 2.4 EvanLuo42 (unofficial Enhanced; BSD-3), read in full for the relevant files

**Gate** (`Common/EnhancedPolicy.slang:117-149`) [CODE]
- `min(1/(pFrom·G_fwd), 1/(pAt·G_inv)) ≥ footprintScale·primaryFootprint`, with footprintScale 2·10⁻⁴ (`Params.slang:101`).
- The inverse term is skipped on request.
- Roughness uses `sampledLobeAlpha`: delta 0, diffuse 1, otherwise the **material GGX α = roughness²** (`SurfaceLoad.slang:104`), compared with 0.2.
  - That is perceptual ≥ 0.447, **not** the paper's perceptual 0.2 [INFERENCE: unit mismatch].
- `geometryTerm` (`:211-218`) is **one-sided**, `max(dot(n, −Δ), 0)`, with oriented normals.
- `primaryRayFootprint` (`:220-226`) uses the face normal, |cos| floored at 1e-6.

**Base path** (`Path/PathSampling.slang`) [CODE]
- `cacheInternalReconnection` (`:113-161`) evaluates the pair (x_{b−1}, x_b) **after** sampling at x_b (`:570-578`). It requires "exact single lobe" and non-delta at both vertices, so marginal = joint; skipInverse when x_b's sampled lobe is diffuse.
- NEE at x_b is streamed **before** the BSDF sample (`:510-526`) and never considers the pair (x_{b−1}, x_b). Its rc is either the cached earlier rc or the light vertex: the gate is tried with `pFrom = light pdf`, and the light vertex is otherwise forced (`:199-247`). So in effect: **"NEE-final x_{d−1} is never an rc vertex"**.
- BSDF-hit emitters are tested with the gate (skipInverse, roughness of the lobe at x_{d−1}) (`:249-287`). Environment escapes get no rc (full replay).
- NEE, RR and GRIS use separate non-replayed RNG streams (`:451-456`). This is good.

**Shift** (`Common/HybridShift.slang`) [CODE]
- `replayPrefix` (`:72-110`) only checks that each replayed lobe equals the stored lobe (`:62-64`).
- `evalReconnect` (`:232-377`) checks exact-single-lobe at y_{k−1} and x_k, visibility and J = new/old density.
- **Neither calls the gate.** A grep of all 28 files finds `passesEnhancedReconnectionGate` only in `PathSampling.slang` (`:138, :217, :264`).
- Neither O1 nor O2 is enforced, so the shift is not guaranteed bijective: expected bias at shift boundaries [INFERENCE, HIGH].

**Pairing and resampling** [CODE]
- `SpatialShift.cs.slang:36-63`: the **lower-index pixel computes both shifts** and evaluates compatibility **once** for the pair. That makes acceptance symmetric by construction.
- `Surface.slang:27-56`: normal ≥ 0.9, and `|d_a−d_b| ≤ 0.1·min(d_a,d_b)`, which is symmetric.
- `Pairing.slang:45-67`: verifies partner(partner(p)) = p.
- `SpatialResample.cs.slang:127`: `applyShiftResult` copies the shifted `reconnectDensity`, so **C4 is handled**. The neighbour MIS weight has swapped arguments (`:115-121`, RC §2.3).

---

## 3. The normative predicate

### 3.1 Inputs, and where each comes from (identical provenance in the base path and in every replay)

| Input | Definition | Provenance rule [INFERENCE: "same-formula principle", §8] |
|---|---|---|
| `pos(v)`, `n^g(v)` | World position and unit geometric normal | `vertexFromIds(inst, prim, bary)` with object→world transform. **Never `o + t·d`**, never the hit `t`. Same function everywhere. |
| t² | ‖x_k − x_{k−1}‖² | `dot(Δ,Δ)` of reconstructed positions |
| Incoming direction at v | `normalize(pos(prev) − pos(v))` | From positions. Not the ray-direction variable, even in the base path. |
| Outgoing direction at x_k (continuing) | ω_k | The **sampled** direction bits in the base. The **stored `rcWi` bits** in shifts. Store uncompressed in validation builds; if octahedral-compressed, the base must also use `decode(encode(ω_k))` in the predicate and the Jacobian cache. |
| NEE direction at x_{d−1} | `normalize(pos(light vertex) − pos(x_{d−1}))` | Light vertex reconstructed from (lightId, uv) or emitter (inst, prim, bary) |
| p̄_v(ω) | Marginal solid-angle BSDF sampling pdf over all lobes, including view-dependent lobe-selection probabilities | `bsdfPdfMarginal(mat(v), wiFromPositions, ω)`. In the base, **re-evaluate** it instead of using the sampler's returned pdf. |
| p_v(ω, ℓ) | Joint pdf (Jacobian only) | `bsdfPdfLobe(mat, wi, ω, ℓ)·P(ℓ | wi)` |
| α(v, ℓ) | Perceptual roughness of lobe ℓ (Blender/glTF "Roughness", GGX α = r²); Lambert 1; delta/singular 0 (Cycles singular if α_x·α_y ≤ 2·10⁻¹⁰, CC §2.7) | Material eval at **LOD 0** (no ray-cone LOD), so the value depends only on (v, ℓ) |
| `diffuseOnly(v)` | Material is exactly one Lambert lobe with selection probability 1 | Material flag. **Not** "the sampled lobe is diffuse" on a multi-lobe material (EV-O8). |
| `thr(D)` | τ·R²_pri(D) | Computed **once** per pixel per frame at primary-hit time and stored as f32 in the G-buffer. The previous frame's copy is kept. Every consumer reads the stored bits. |
| τ | c/100 = 2·10⁻⁴ | **Runtime uniform**, not a WGSL `override`, so no pipeline constant-folds it (§8) |
| α_min | 0.2 perceptual | Runtime uniform |

`R²_pri(D) = ‖pos(x₁^D) − x₀^D‖²·4π / max(|⟨n^g_{x₁^D}, normalize(x₀^D − pos(x₁^D))⟩|, 10⁻⁶)` (P-Eq.5 RHS; S-Eq.6).
- x₀^D is the camera position of domain D's frame.
- x₁^D is D's **jittered** primary hit (CR-X1).
- The floor 10⁻⁶ is my choice (E- uses the same). It is harmless because it is a function of D only.

### 3.2 Events

A **surface vertex** carries the event that leaves it *within the candidate path being classified*:

| Event kind | lobe | delta | α used by R | p̄ used by F (when this vertex is x_{k−1}) or by I (when it is x_k) |
|---|---|---|---|---|
| `EV_BSDF` (continuing, or BSDF ray that hits an emitter or escapes) | sampled ℓ | lobe delta? | α(v, ℓ) | p̄_v(ω_sampled) |
| `EV_NEE` (last scattering vertex of an NEE candidate) | 0 (all lobes) | – | never used as x_{k−1} | p̄_v(ω_NEE): the all-lobe BSDF sampling pdf of the NEE direction (the p₂ of the NEE MIS weight) |
| `EV_RECONNECT` (y_{k−1} on an offset path) | ℓ_{k−1} **copied** from the base | base's (always false here) | α(y_{k−1}, ℓ_{k−1}) | p̄^y_{k−1}(ω'), with ω' = normalize(x_k − y_{k−1}) |
| `EV_RECONNECT_NEE` (y_{d−1} of a forced-NEE offset) | 0 | – | – | p̄^y_{d−1}(normalize(x_d − y_{d−1})) |

### 3.3 The pair predicate `P_k` (pair (a, b) = (x_{k−1}, x_k), k ≥ 2)

```
P_k(a, e_a, b, e_b, thr) =
   D:  e_a.delta == 0                                             // delta lobe ⇒ footprint 0
   R:  e_a.alpha ≥ α_min                                           // single-vertex roughness at x_{k−1} (P-Eq.5 + §4.2)
   F:  kind(b) == ENV  ?  true                                     // rayFP = +∞ (P-§4.2)
                       :  t²/(e_a.p̄ · |cos θ^g_b|) ≥ thr           // ray footprint, cosine at the RECEIVING vertex b
   I:  kind(b) ∈ {LIGHT, ENV} or diffuseOnly(b) ? true             // fn. 6: emissive/diffuse x_k skip
                       :  e_b.delta == 0 ∧ t²/(e_b.p̄ · |cos θ^g_a|) ≥ thr   // inverse footprint, cosine at a
```

Here `|cos θ^g_b| = |⟨n^g_b, Δ⟩|/t` and `|cos θ^g_a| = |⟨n^g_a, Δ⟩|/t`, with Δ = pos(b) − pos(a).

Guards:
- t² > 10⁻¹², p̄ > 0 and finite, cos > 0. **A failed guard means the pair fails.**
- The one exception is ENV, whose rayFP is +∞ by definition.

### 3.4 Candidate rule: which pairs a candidate owns, and forcing

For a candidate path x̄ of length d with technique t ∈ {NEE, BSDF-emitter, BSDF-env}, the ordered pair list is:

| Pair index k | Pair | e_{x_{k−1}} | e_{x_k} | Kind of x_k |
|---|---|---|---|---|
| 2 … d−2 | shared prefix pairs | EV_BSDF | EV_BSDF | SURFACE |
| d−1 (NEE candidate) | (x_{d−2}, x_{d−1}) | EV_BSDF | **EV_NEE** | SURFACE |
| d−1 (BSDF-ending candidate) | (x_{d−2}, x_{d−1}) | EV_BSDF | EV_BSDF (the sampled ray toward the emitter or env) | SURFACE; identical to the continuing path's pair d−1 |
| d (BSDF-hit emitter / pass-through analytic light) | (x_{d−1}, x_d) | EV_BSDF | – | LIGHT |
| d (BSDF escape) | (x_{d−1}, env) | EV_BSDF | – | ENV |
| d (NEE light vertex) | – | – | – | **forced, never tested** |

Then:

```
k*(x̄; D) = min{ k ∈ list, 2 ≤ k ≤ K_max : P_k }        (K_max optional replay-length cap, default ∞, same in all domains)
if none:   k* = d  if t == NEE  (forced, P-§6.2.3)
           k* = ∅  otherwise    (full random replay, J = 1)
```

**Well-definedness requirement [INFERENCE, critical].** k* must be a function of the candidate path's own vertices, lobes and directions only.
- The NEE candidate at x_b does **not** contain the BSDF direction ω_b sampled later at x_b.
- So its rc index must not depend on it, not even through "lazy" updates of an already-streamed candidate.
- Conversely, a BSDF-emitter candidate at x_{b+1} *does* contain ω_b, so it inherits the pair-b decision.

### 3.5 Case table by vertex kind (the deliverable)

Each entry below gives, in order:
- **rayFP** (when x_k is this kind, from x_{k−1}) and **invFP** at x_k;
- the pdf and the normals;
- delta handling;
- the Jacobian factors (joint pdfs), for completeness.

**(a) BSDF-sampled surface vertex x_k, path continues (EV_BSDF at x_k)**
- **rayFP:** `t²/(p̄_{k−1}(ω_{k−1})·|⟨n^g_{x_k},Δ̂⟩|)`.
- **invFP:** `t²/(p̄_k(ω_k | −ω_{k−1})·|⟨n^g_{x_{k−1}},Δ̂⟩|)`. Skipped if `diffuseOnly(x_k)`.
- **Pdf / normal:** marginal p̄ (S-§3), geometric normals, cosines at the receiving vertex.
- **Delta:** fail if ℓ_{k−1} or ℓ_k is delta.
- **Jacobian:** `p_{k−1}(ω,ℓ_{k−1})·G(·→x_k)·p_k(ω_k,ℓ_k)`, joint.

**(b) NEE light vertex x_d (point, spot, area light or emissive triangle sampled by NEE)**
- **rayFP / invFP:** **none; forced** when no earlier pair of the candidate passes (P-§6.2.3).
- **Pre-rc pair (x_{d−2}, x_{d−1}):** invFP uses the **BSDF all-lobe pdf of the NEE direction** at x_{d−1} (RTXDI case 2).
- **Pdf / normal:** – (light pdfs are irrelevant to the predicate).
- **Delta:** –
- **Jacobian:** 1 for area-measure or delta-position lights with shading-point-independent selection. Otherwise the ratio of area↔solid-angle conversions (Gap 1).

**(c) BSDF-hit emissive triangle x_d (terminal)**
- **rayFP:** `t²/(p̄_{d−1}(ω_{d−1})·|⟨n^g_{tri},Δ̂⟩|)`, with the triangle's geometric normal and |cos|. One-sided emission is handled by Le = 0, not by the predicate.
- **invFP:** skipped (fn. 6).
- **Pdf / normal:** marginal at x_{d−1}.
- **Delta:** fail if ℓ_{d−1} is delta.
- **Jacobian:** `p_{d−1}·G(·→x_d)`, with p_d := 1.

**(d) Pass-through analytic light (Cycles MIS mode B, CC §2.8)**
- **As the terminal vertex z of an emitter-hit candidate:** same as (c), with n = the light's normal at z. The light vertex is stored as (lightId, uv). Reconnection visibility **ignores analytic lights**.
- **As a mid-path event:** **not a vertex.** Pairs of the continuing path use the full segment x_i → x_{i+1} behind the light.
- **Pdf / normal:** as (c).
- **Delta:** as (c).
- **Jacobian:** as (c).

**(e) Environment, BSDF escape**
- **rayFP:** **+∞**.
- **invFP:** skipped.
- **R:** α(x_{d−1}, ℓ_{d−1}) ≥ α_min is the **only** guard (P-§4.2).
- **Delta:** fail if ℓ_{d−1} is delta.
- **Jacobian:** direction copy: `p^y_{d−1}(ω,ℓ)/p^x_{d−1}(ω,ℓ)`, with G := 1.

**(f) Environment, NEE**
- Forced, as (b).
- **Jacobian:** 1 (the direction comes from shading-point-independent env sampling).

**(g) NEE-final surface vertex x_{d−1} as rc (EV_NEE at x_k)**
- **rayFP:** as (a), with p̄_{d−2}.
- **invFP:** `t²/(p̄_{d−1}(ω_NEE | −ω_{d−2})·|⟨n^g_{x_{d−2}},Δ̂⟩|)`. Skipped if `diffuseOnly`.
- **Pdf / normal:** marginal p̄ of the **NEE direction** under the BSDF sampler.
- **Delta:** fail if ℓ_{d−2} is delta. x_{d−1} has a non-delta lobe because NEE happened there.
- **Jacobian:** `p_{d−2}·G(·→x_{d−1})`, with **p_k := 1**: the NEE direction's density does not depend on the incoming direction. The integrand still changes through f(x_{d−1}) and the NEE MIS weight.

**(h) k = 2 (rc = x₂, i.e. x_{k−1} = x₁)**
- Same P_k. At y₁ the lobe ℓ₁ is **copied**. It needs the joint pdf `p^y₁(ω',ℓ₁) > 0` (else J = 0, so the shift fails) and `α(y₁,ℓ₁) ≥ 0.2`.
- No earlier pairs exist.
- Jacobian as (a), (c), (e) or (g), depending on x₂'s kind.

**Answers to the specific open items:**

1. **p_k for NEE-terminated paths.**
   - Predicate: p̄_BSDF(ω_NEE | x_{d−1}, −ω_{d−2}), all lobes. This follows RTXDI `RecordNeeLightSample` and its offset counterpart.
   - Jacobian: 1, as in F-`Shift.slang` (`isRcVertexNEE` excludes the p₂ ratio).
   - **Alternative (EvanLuo):** never let x_{d−1} be the rc for NEE candidates.
   - Both are unbiased if applied identically in base and offset, since P is only a domain restriction.
   - I recommend the RTXDI/Falcor behaviour [INFERENCE]. It reconnects diffuse-ish x_{d−1} vertices *before* the light, so the shared suffix x_{d−1}→x_d needs **no shadow ray** in the shift. Forced light reconnection needs one more replayed bounce plus a shadow ray.
2. **k = 1 in Falcor's indexing (rc = x₂).** See (h). Falcor's `hasRoughComponent(sd, 1.0)` (`Shift.slang:146`, `PathTracer.slang:1232`) means "y₁ can produce the copied *diffuse-group* lobe". In a true lobe-indexed space this becomes "the joint pdf of the copied lobe is positive" (feasibility) plus R_k. No special threshold.
3. **Emitter and pass-through light vertices.** See (c) and (d).
   - The paper's roughness test at x_{d−1} applies. RTXDI drops it for emitter hits (EV-O5); either is unbiased if consistent. **Choose the paper's.**
   - BSDF-hit emitters are **not** forced (the paper forces only NEE light vertices).
   - Mode A (NEE-only analytic lights) never produces (d).
4. **Forced NEE reconnection.** k* = d whenever every pair of the NEE candidate fails. That includes pair d−1 with the NEE event. The offset side is covered by O1 in §6.1.
5. **Delta lobes.** A delta lobe at x_{k−1}, or at x_k when the path continues from x_k, fails the pair. A delta light (point or spot) is only ever a forced NEE vertex. Cycles singular GGX (α_xα_y ≤ 2·10⁻¹⁰) counts as delta.
6. **Single-vertex roughness.** At x_{k−1} only, with the lobe used there:
   - the sampled lobe on a base path;
   - the replayed lobe at replayed offset vertices;
   - the **copied** ℓ_{k−1} at y_{k−1}.
   Perceptual roughness 0.2 is Falcor's `linearRoughness` and the glTF/Blender convention. EvanLuo's GGX-α 0.2 is a different, stricter rule.
7. **Lobe revocation.** Replaced by *deferred, per-candidate evaluation* (§3.4). Falcor's revocation (`PathTracer.slang:1416-1423`) is the two-vertex-rule version of the same thing. In our predicate, the lobe sampled at x_k enters through D (delta) and through invFP's p̄_k(ω_k).

### 3.6 WGSL sketch of the predicate module (`rc.wgsl`, the only place these formulas exist)

```wgsl
// Integer-bit helpers: robust under Chrome/Metal relaxed math (WP §8.1: `x != x` is folded away)
fn posFinite(x: f32) -> bool { let b = bitcast<i32>(x); return b > 0 && b < 0x7f800000; } // 0 < x < +inf
// For non-negative finite floats, IEEE order == signed-integer order of the bit patterns.
fn geqPos(a: f32, b: f32) -> bool { return bitcast<i32>(a) >= bitcast<i32>(b); }

const K_SURFACE: u32 = 0u; const K_LIGHT: u32 = 1u; const K_ENV: u32 = 2u;

struct RcVertex { pos: vec3f, ng: vec3f, kind: u32, diffuseOnly: u32 };
struct RcEvent  { lobe: u32, delta: u32, alpha: f32, pMarg: f32 };
struct RcResult { pass: bool, margin: f32 };  // margin = min over sub-tests of log2(value/threshold) or (α-α_min); debug only

fn rcPairTest(a: RcVertex, ea: RcEvent, b: RcVertex, eb: RcEvent, thr: f32, alphaMin: f32) -> RcResult {
  var r = RcResult(false, 0.0);
  if (ea.delta != 0u) { return r; }
  r.margin = ea.alpha - alphaMin;
  if (!(ea.alpha >= alphaMin)) { return r; }                  // alpha is a plain material value, never NaN
  if (b.kind == K_ENV) { r.pass = true; return r; }          // rayFP = +inf, inverse skipped
  let dlt = b.pos - a.pos;
  let t2  = dot(dlt, dlt);
  if (!posFinite(t2) || t2 < 1e-12) { return r; }
  let t   = sqrt(t2);
  let cb  = abs(dot(b.ng, dlt)) / t;                          // cos at receiving vertex x_k
  if (!posFinite(cb) || !posFinite(ea.pMarg)) { return r; }
  let rayFP = t2 / (ea.pMarg * cb);
  if (!posFinite(rayFP) || !geqPos(rayFP, thr)) { r.margin = min(r.margin, log2(rayFP / thr)); return r; }
  r.margin = min(r.margin, log2(rayFP / thr));
  if (b.kind == K_LIGHT || b.diffuseOnly != 0u) { r.pass = true; return r; }   // fn. 6
  if (eb.delta != 0u) { return r; }
  let ca = abs(dot(a.ng, dlt)) / t;                           // cos at x_{k-1}
  if (!posFinite(ca) || !posFinite(eb.pMarg)) { return r; }
  let invFP = t2 / (eb.pMarg * ca);
  r.margin = min(r.margin, log2(invFP / thr));
  r.pass = posFinite(invFP) && geqPos(invFP, thr);
  return r;
}

// Stored ONCE per pixel per frame at primary time; every consumer reads the stored f32 bits.
fn primaryThreshold(camPos: vec3f, x1: vec3f, ng1: vec3f, tau: f32) -> f32 {
  let v = camPos - x1; let d2 = dot(v, v);
  let c = max(abs(dot(ng1, v)) * inverseSqrt(d2), 1e-6);
  return tau * (d2 * 12.566370614359172 / c);
}
```

---

## 4. Base-path construction (initial sampling), pseudo-code

The rc index is evaluated **per candidate** and streamed with the candidate (RIS over disjoint path-tree candidates, GM §5.4).

```
thr    = gbuf.thr[pixel]            // this domain, this frame
x1     = vertexFromIds(vbuf[pixel]) // jittered primary
treeRc = NONE; treeCache = {}       // first passing SHARED pair (x_{b-1}, x_b) with EV_BSDF at x_b
prevV  = none; prevE = none
for b in 1 .. maxLen-1:                                   // current scattering vertex x_b
  curV = rcVertex(x_b)
  wi_b = (b == 1) ? normalize(camPos - pos(x_b)) : normalize(pos(prevV) - pos(x_b))
  // (1) NEE candidate (x_0..x_b, L), d = b+1. Randoms from the NEE stream (never replayed).
  if nonDeltaLobes(x_b):
     ls   = sampleLight(neeStream(pixel, frame, b))
     eNee = RcEvent(0, 0, -, bsdfPdfMarginal(x_b, wi_b, normalize(pos(ls) - pos(x_b))))
     k = treeRc                                           // shared pairs 2..b-1 only
     if k == NONE && b >= 2 && b <= K_MAX && rcPairTest(prevV, prevE, curV, eNee, thr).pass: k = b   // case (g)
     if k == NONE: k = b + 1                              // forced (case b)
     streamCandidate(NEE, d=b+1, k, jDen(k, NEE), ...)    // jDen per §3.5; rcWi = NEE dir if k == b
  // (2) RR: initial sampling only, separate stream; changes only the UCW (S-§6). Never influences k.
  // (3) BSDF sample at x_b with the REPLAY stream (the only replayed randoms)
  (ell_b, w_b) = sampleBsdf(x_b, wi_b, replayStream(seed, b))
  eB = RcEvent(ell_b, isDelta(ell_b), alpha(x_b, ell_b), bsdfPdfMarginal(x_b, wi_b, w_b))   // re-evaluated, §8
  // (4) shared pair (x_{b-1}, x_b) is now complete
  if treeRc == NONE && b >= 2 && b <= K_MAX && rcPairTest(prevV, prevE, curV, eB, thr).pass:
     treeRc = b; treeCache = { rcIds(x_b), rcWi = w_b (exact bits), ell_{b-1}, ell_b,
                               jDen = pJoint_{b-1}(dir x_{b-1}->x_b, ell_{b-1}) * G(x_{b-1}->x_b) * pJoint_b(w_b, ell_b) }
  // (5) trace
  hit = trace(offsetRay(pos(x_b), ng(x_b), w_b), w_b)    // analytic lights are NOT in the BVH
  for each analytic light crossed by the segment (mode B):              // pass-through, case (d)
     k = treeRc; if k == NONE && b+1 <= K_MAX && rcPairTest(curV, eB, lightVertex(z), -, thr).pass: k = b+1
     streamCandidate(BSDF_EMITTER, d=b+1, k, ...)          // k may be NONE -> full replay
  if miss:  k = treeRc; if k == NONE && b+1 <= K_MAX && rcPairTest(curV, eB, ENV, -, thr).pass: k = b+1
            streamCandidate(BSDF_ENV, d=b+1, k, ...); break
  if emissive(hit): k = treeRc; if k == NONE && b+1 <= K_MAX && rcPairTest(curV, eB, lightVertex(hit), -, thr).pass: k = b+1
                    streamCandidate(BSDF_EMITTER, d=b+1, k, ...)
  prevV = curV; prevE = eB; x_{b+1} = vertexFromIds(hit)
```

Notes:
- The RIS stream must be a separate RNG, so selection never perturbs the replay stream (F- `PathBuilder.sg`, RC §1.4). This pseudo-code uses per-dimension counter-based randoms (WP §8.5).
- **Record the replayed lobes ℓ₁…ℓ_{k−1}** in the reservoir, e.g. 8 bits × 8 (E-`packLobe`). Two uses:
  - the T3 signature;
  - optional "replay lobe must equal stored lobe" (E-`HybridShift.slang:62-64`). That is a symmetric domain restriction, so it is unbiased [INFERENCE], but it is not required.

---

## 5. What the reservoir must carry for the predicate and the Jacobian

| Field | Why |
|---|---|
| `initSeed` | Replay stream. Also the path identity for optional τ jitter (§6.4) and for the duplication map. |
| d, technique, k (or ∅), light kind (surface / analytic area / delta / env) | Pair list and case selection |
| rc vertex IDs (inst, prim, bary) or (lightId, uv), plus ℓ_{k−1}, ℓ_k | Rebuild x_k and its material; copy lobes |
| `rcWi` (ω_k), exact f32×3 in validation builds | invFP at x_k and the p_k Jacobian factor |
| `jDen` = the base's own product of joint factors (§3.5) | Jacobian denominator. **Must be replaced on selection** (§7.4). |
| Light-vertex data for NEE (lightId, uv), NEE candidate count M, light pdf if solid-angle | Gap 1 (not needed by the predicate) |
| Prefix lobes ℓ₁…ℓ_{k−2} (optional) | Tests, and optional lobe-equality restriction |

---

## 6. Offset-path checks (spatial, paired, temporal forward and inverse)

### 6.1 The conditions

Given base x̄ ∈ Ω_S with rc index k, and destination domain D (primary y₁, `thr_D`), the shift T_{S→D} is **defined** iff all of the following hold:

- **(O0) Structure.**
  - The replay from y₁ with x̄'s seed reaches y_{k−1} without a miss.
  - For k = ∅, it ends at the same index d with the same technique (emitter vs env hit).
  - The copied-lobe joint pdfs are positive: `p^y_{k−1}(ω',ℓ_{k−1}) > 0`, and for case (a) `p^y_k(ω_k,ℓ_k) > 0`.
  - f ≠ 0.
  - Visibility of y_{k−1} ↔ x_k, where the shadow ray ignores analytic lights and skips the primitives of both endpoints (§9).
- **(O1) No earlier pair passes.** For every 2 ≤ j ≤ min(k−1, K_max), `P_j(ȳ; thr_D)` is **false**, where:
  - for j ≤ k−2, the pair (y_{j−1}, y_j) uses EV_BSDF events from the **replayed** samples;
  - for j = k−1, the pair (y_{k−2}, y_{k−1}) uses `e_{y_{k−1}}` = **EV_RECONNECT**: copied ℓ_{k−1}, ω' toward x_k, p̄^y_{k−1}(ω'). For forced NEE it is EV_RECONNECT_NEE toward the shared light vertex x_d.
  - This is exactly RTXDI's `checkPreRcInverseGeoTerm`. It is the check most implementations forget.
- **(O2) The rc pair passes** (skip only for a forced NEE light): `P_k(y_{k−1}, EV_RECONNECT, x_k, e_{x_k}; thr_D)` is **true**, where:
  - R uses α(y_{k−1}, ℓ_{k−1});
  - F uses p̄^y_{k−1}(ω') and |cos| at x_k;
  - I uses **recomputed** p̄^y_k(ω_k | from y_{k−1}), i.e. EV_BSDF with the stored ω_k for case (a), or EV_NEE for case (g), with |cos| at y_{k−1};
  - I is skipped for LIGHT, ENV or diffuseOnly.
- **(O3) k = ∅.** For every 2 ≤ j ≤ min(d, K_max), `P_j(ȳ)` is false, **including the terminal pair** (y_{d−1}, y_d) of kind LIGHT or ENV (RTXDI "case 2"; Falcor `PathTracer.slang:1104-1107`).

**Claim [INFERENCE, proof sketch].** O0–O3 ⇔ `k*(ȳ; D) = k`.
- ȳ shares x_k…x_d with x̄, so pairs j > k never matter to the minimum.
- Pairs j < k are exactly O1, and pair k is O2. For forced NEE, "all j < d fail" is exactly the forcing condition.
- Symmetric case: `T_{D→S}(ȳ)` replays the same ū from x₁ and gets back x_{1..k−1}. The conditions O1/O2 evaluated on x̄ in S are exactly the conditions that made k*(x̄; S) = k in the base.
- So `T_{D→S}∘T_{S→D} = id` on the domain, and both maps have the same domain condition, which gives bijectivity (G-Def.4.2, G-§7.1).
- The PSS Jacobian of this map is S-Eq.4, and random replay contributes 1 (S-§1).

### 6.2 Split across the replay and reconnection passes (paired pre-pass with compaction)

```
ReplayRecord (per (pixel, slot) or temporal pair), written by the replay pass:
  yPrevIds: (inst, prim, bary f32x2)   // y_{k-1}
  wiPrev:   vec3f                      // incoming direction at y_{k-1} = normalize(pos(y_{k-2}) - pos(y_{k-1})) (exact bits)
  thp:      vec3f                      // replayed prefix throughput (PSS)
  preV:     (pos, ng of y_{k-2}) or (preIds)   // to evaluate the last pre-rc pair once ω' is known
  preE:     RcEvent at y_{k-2}         // replayed lobe, delta, alpha, pMarg
  status:   OK | FAIL_EARLY_PAIR(j) | FAIL_MISS | FAIL_TECHNIQUE
```

- **Replay pass.** It replays y₂…y_{k−1}. After each replayed BSDF sample at y_j (2 ≤ j ≤ k−2) it evaluates `rcPairTest(y_{j−1}, e_{j−1}, y_j, e_j, thr_D)`; a pass means FAIL_EARLY_PAIR.
- **Shift (reconnection) pass.** It has x_k from IDs, so it evaluates the last pre-rc pair with EV_RECONNECT (O1 at j = k−1), then O2, then visibility, then F, J and `jNum`.
- **Alternative.** The replay pass can also rebuild x_k and do both. Either is fine as long as **both call the same `rcPairTest`** and read the same `thr_D` bits.
- Pairs with k = 2 skip the replay pass: y_{k−1} = y₁, so only O2 applies (there is no pre-rc pair).

### 6.3 R_pri per domain (the "which camera, which primary" rule)

| Shift | Destination domain D | x₀^D | x₁^D (and n^g) | thr source |
|---|---|---|---|---|
| Initial (canonical) in pixel p, frame t | (p, t) | camera(t) | p's jittered primary at t | `gbuf_t.thr[p]` |
| Spatial / paired p → q | (q, t) | camera(t) | q's jittered primary at t | `gbuf_t.thr[q]` |
| Temporal forward (p', t−1) → (p, t) | (p, t) | camera(t) | p's primary at t | `gbuf_t.thr[p]` |
| **Temporal inverse** (p, t) → (p', t−1), for MIS | (p', t−1) | **camera(t−1)** | **previous V-buffer** hit of p', with transforms of t−1 for moving instances | `gbuf_{t−1}.thr[p']` |

Sources:
- S-Eq.18/19: each path's condition uses its own R_pri.
- R-`PathReconnectibility.hlsli`: the destination surface, with the previous camera for previous-frame shifts.

Consequences:
- Keep last frame's `thr` plane (4 B/px), camera position and V-buffer.
- With per-frame jitter (CR-X1), a static-camera temporal shift is *not* the identity; it is a real shift with its own R_pri.
- EvanLuo's "identity shortcut" (`TemporalShift.cs.slang:129-139`) uses 10⁻⁶ tolerances. Use it only if the primary IDs, bary bits, camera position and `thr` bits are all identical; then k* is identical.

### 6.4 Fixed or jittered thresholds? Decision: **fixed by default**

- **Correctness.** Any τ that is a deterministic function of (path, domain) keeps T bijective. The paper uses fixed c = 0.02 and does not mention jitter [SRC P-§4, S-§2].
- **RTXDI jitters** c and the pdf-roughness α with a Gaussian drawn from the path's **replay** stream, so the base and every offset draw the same value [CODE R-].
  - The purpose is not documented. I assume it smooths the spatial on/off seam of reconnection behaviour [INFERENCE].
- **If jitter is enabled:**
  - τ(ū) = τ₀·exp(σ·z(ū)) or a clamped Gaussian;
  - z is drawn from a **dedicated counter-based dimension of the path seed**, the same in every domain and every frame;
  - **never** from pixel, frame or resampling RNG. Those would change τ between base and offset, which is non-bijective and therefore bias.
  - Store nothing extra: recompute τ(ū) from `initSeed` wherever `thr_D` is formed as `τ(ū)·R²_pri(D)`. In that case store R²_pri, not thr, in the G-buffer.
- **Units trap.** RTXDI's `minConnectionFootprint = 0.02` is squared, giving 4·10⁻⁴·R²_pri, i.e. twice the paper's 2·10⁻⁴ (EV-C11). EvanLuo's `footprintScale = 2·10⁻⁴` equals the paper.

---

## 7. Paired spatial reuse: symmetric acceptance, MIS from shared slot data, and the cached-Jacobian update

### 7.1 Symmetric acceptance (fixes EV-C3)

```
A(p, q) = partner_t(p) == q  ∧  partner_t(q) == p          // involution check (E-Pairing.slang:45-67)
        ∧ A0(G[min(p,q)], G[max(p,q)])                     // canonical argument order
A0(a, b) = a.valid ∧ b.valid ∧ dot(a.ng, b.ng) ≥ 0.5 ∧ |a.z − b.z| ≤ 0.1·min(a.z, b.z)   // z = camera distance, frame t
```

Rules:
- **A depends only on the G-buffer, never on reservoir contents.** That keeps it sample-independent, hence unbiased (GM §7.5).
- The mathematically symmetric form alone is fragile. It is symmetric in IEEE arithmetic only when both calls compile identically. The canonical order makes it robust to future asymmetric terms (e.g. view-dependent material tests) and to per-pipeline code-generation differences.
- The thresholds 0.5 (Falcor) or 0.9 (EvanLuo) are variance-only choices [INFERENCE].
- **S2 never re-evaluates A.** Each S1 thread writes the slot status. Both slots of a pair then agree by construction; assert it in debug builds.

**Slot status codes**, packed into the J word with integer compares (`bitcast<u32>`):
- `VALID`: J is finite and > 0.
- `FAILED`: `J bits == 0`. The pair is accepted, but the shift is undefined or p̂ = 0.
- `NOT_ACCEPTED`: `bits == 0xFFFFFFFF`. No partner, not reciprocal, off-screen, or A false.

**`FAILED` partners belong to S_c. `NOT_ACCEPTED` partners do not.** Treating "slot not computed" as "failed" (or the reverse) breaks the partition of unity. That is the EV-C3 energy loss.

### 7.2 S1 (shift pre-pass), per (pixel p, slot t)

```
q = partner_t(p); if !A(p,q): slot[t][p] = NOT_ACCEPTED; return
X = resIn[p]                                   // immutable post-temporal buffer (S2 must read the SAME buffer)
if X empty: slot[t][p] = FAILED; return
(ok, Fy, J) = hybridShift(X, domain(q, t))     // O0–O3 with thr_q, §6
slot[t][p] = ok ? (Fy*J, J) : FAILED           // Fy*J is RGB; J > 0
```

### 7.3 The chosen pairwise MIS (defensive pairwise, GRIS Eq. 38, |R| = 1, confidence-weighted), computable from slots

For pixel c:
- S_c = {partners with status ∈ {VALID, FAILED}} and `k = |S_c|`.
- `a = c_c / max(k, 1)`, using the post-temporal confidences.
- For partner j ∈ S_c the data are:
  - `H_j = slot[t][c].FJ = F_j(Z_j)·J_{c→j}` (0 if FAILED), where Z_j = T_{c→j}(X_c);
  - `G_j = slot[t][j].FJ = F_c(Y_j)·J_{j→c}` and `J_j = slot[t][j].J`, where Y_j = T_{j→c}(X_j);
  - partner confidence c_j and `p̂_j(X_j) = lum(F_j(X_j))`, taken from `resIn[j]`.
- `p̂_c(X_c) = lum(F_c(X_c))`.

```
m_c(X_c) = 1/(k+1) · [ 1 + Σ_{j∈S_c}  a·p̂_c(X_c) / ( a·p̂_c(X_c) + c_j·lum(H_j) ) ]              (pair term := 1 if denominator = 0)
m_j(Y_j) = 1/(k+1) ·   c_j·p̂_j(X_j) / ( c_j·p̂_j(X_j) + a·lum(G_j) )                              (:= 0 if FAILED or denominator = 0)
w⃗_c = m_c · F_c(X_c) · W_c          w⃗_j = m_j · G_j · W_j          w_i = lum(w⃗_i)   (p̂ = lum is linear)
select Y ∝ w_i;  W_out = Σ w_i / p̂_c(Y);  p̂_c(Y_j) = lum(G_j)/J_j;  shading colour = Σ w⃗_i  (P-§6.3)
c_out = c_c + Σ_{j∈S_c} c_j                                            (FAILED partners still count, as in Falcor)
```

**Derivation [INFERENCE, checked algebra].**
- GRIS Eq. 38 with |R| = 1, M = k+1, and p̂ → c_c·p̂_c, p̂_{←j} → c_j·p̂_{←j} (G-§6.2), gives:
  `m_j(y) = (1/(k+1))·c_j p̂_{←j}(y)/(c_j p̂_{←j}(y) + (c_c/k) p̂_c(y))`
  `m_c(y) = (1/(k+1))[1 + Σ_j (c_c/k)p̂_c/(…)]`.
  - The (c_c/k) comes from α = M/|R| − 1 = k (GM §3.4).
- For the neighbour sample, `p̂_{←j}(Y_j) = p̂_j(X_j)/J_{j→c}` and `p̂_c(Y_j) = lum(G_j)/J_{j→c}`. Multiplying numerator and denominator by J_{j→c} gives the J-free form above.
- For the canonical, `p̂_{←j}(X_c) = p̂_j(Z_j)·J_{c→j} = lum(H_j)`.
- **Partition of unity.** For any y, `m_c + Σ_j m_j = (1/(k+1))·(1 + Σ_j 1) = 1`, and this holds for any positive constant in place of k (EV-C2).
- **Why `k = |S_c|` (valid count) instead of Falcor's nominal N.** It is the GRIS-faithful α = M/|R|−1, which keeps the Thm A.4 bound `w_i ≤ C_i/|R|` intact. Falcor's N is also unbiased but is a different estimator (CR-X16). The same k must be used inside m_c and all m_j of pixel c; it is.
- **Do not use EvanLuo's neighbour weight.** Its arguments are swapped (`SpatialResample.cs.slang:115-121`; RC §2.3), so it is not a partition of unity.

### 7.4 On selecting a shifted sample: what to overwrite (fixes EV-C4)

When Y_j is selected into pixel c:

```
out           = resIn[j]                         // suffix, seeds, k, d, technique, rc IDs, ℓ_{k-1}, ℓ_k, rcWi, radiance, light id/uv
out.F         = G_j / J_j                        // integrand in c's domain; p̂ = lum(out.F)
out.jDen      = J_j · resIn[j].jDen              // = p^y_{k-1}(ω',ℓ_{k-1})·G(y_{k-1}→x_k)·p^y_k(ω_k,ℓ_k)  (JOINT pdfs, EV-O2)
                                                 //   (or store jNum from S1 directly; only rounding differs)
out.lightPdf  = recomputed at y_{k-1}  if the light pdf is solid-angle or selection is shading-point dependent (GV-C2)
out.prefixLobes = replayed ℓ^y_1..ℓ^y_{k-2}      // only if you store prefix lobes and replay may change them
out.W = W_out;  out.c = c_out
```

- For k = ∅, and for forced NEE with area-measure light sampling, J = 1 and `jDen` is unused.
- Reservoirs must be ping-ponged. S2 writes `resOut` and reads only `resIn`, and so does S1. Reading a partner reservoir that S2 already overwrote in place would pair a slot computed from X_j with fields of a different sample.
- EvanLuo does this correctly: `applyShiftResult` copies `reconnectDensity` (`EnhancedPolicy.slang:101-110`). Falcor does too (`Shift.slang:568-569`).
- The same update applies to a selected temporal sample.

---

## 8. Bit-identity under Chrome's relaxed Metal math: what can and cannot be guaranteed

### 8.1 Facts

- Dawn compiles MSL with `fp math_mode(relaxed)`. **FMA contraction happens** (measured: `a*b+c` returned the fused result). `x != x` is folded (WP §8.1).
- Relaxed math also permits reassociation, reciprocal-for-division, and ignoring NaN/Inf (Chrome's `strictMath` description, VH §5.6).
- `strictMath` is reachable only with the developer-features flag (WP §7, §8.1).
- WGSL has no `precise` qualifier and no way to forbid contraction for a region [SRC WGSL spec: no such attribute; INFERENCE].

### 8.2 Consequence

- The same WGSL function inlined into two pipelines (initial sampling vs shift/replay) *may* compile to different instruction sequences. Examples:
  - different contraction after inlining;
  - constant folding with pipeline `override`s, followed by reassociation.
- So predicate inputs, and hence decisions near a threshold, can differ in the last ulp between the base path and a round-trip re-evaluation of the same path.

**Magnitude [INFERENCE].** A decision flips only when a footprint lies within ~10⁻⁶ relative of `thr`. Log-footprints spread over many e-folds, so the flip probability per predicate is of order 10⁻⁷ to 10⁻⁶.
- The affected set of paths has that measure, so the resulting bias is ≲10⁻⁶ relative.
- That is three orders of magnitude below VH's 0.2% ReSTIR-vs-PT tolerance. It is **not a measurable bias**.
- Logic asymmetries are the real risk: wrong pdf, wrong normal, wrong domain, a missing O1/O2 check. They affect sets of positive measure. The tests in §9 separate the two.

### 8.3 Engineering rules that make predicates as stable as WebGPU allows [INFERENCE]

1. **One module, one formula.** `rc.wgsl` holds `rcPairTest`, `primaryThreshold`, `vertexFromIds`, `bsdfPdfMarginal` and `bsdfPdfJoint`. No second copy of any footprint, cosine or pdf expression exists anywhere.
2. **Same-formula principle for inputs.**
   - The base path recomputes positions from IDs, t² from positions, incoming directions from positions, and p̄ by *evaluation*, exactly as a replay/shift does. It never uses hit `t`, `o + t·d`, the ray-direction variable, or the sampler-returned pdf.
   - Directions leaving a vertex are the sampled bits (base) or the stored bits (shift): the same bits.
3. **Stored `thr` per pixel per frame** (§6.3). Every pipeline reads the same f32 bits. τ and α_min are runtime uniforms, never `override` constants.
4. **Integer comparisons** (`geqPos`, `posFinite`) after computing the two values. Relaxed-math rewriting of float compares and NaN assumptions then cannot change the decision logic.
5. **Lossless rc data in validation builds.** Keep bary and `rcWi` as f32. In compressed builds, the base path must run the predicate and the Jacobian on the *decoded* values (EV-O9).
6. **Canonical argument order** for symmetric predicates (A), and a single evaluation per pair.
7. **Validation build "single path engine" [INFERENCE, optional but strongly recommended for T3].**
   - One compute pipeline runs initial sampling, replay and reconnection through a runtime `mode` uniform, with **exactly one call site** each of `rcPairTest`, `vertexFromIds` and BVH traversal.
   - This also respects the C2-Renderer compile-time lesson (RC §2.4).
   - Identical inputs then give identical bits, and T3's FP-class count must be exactly **0**.
   - Also A/B with `strictMath: true` (developer flag) in the harness.

---

## 9. Pseudo-code of the complete offset check, and the most likely asymmetry bugs

### 9.1 Offset check (used verbatim by spatial, paired and temporal forward and inverse)

```
fn hybridShift(X: Reservoir, D: Domain /* primary ids, camera, thr (cur or prev plane) */) -> ShiftOut {
  thr = D.thr                                              // or tau(X.initSeed)·D.R2 when jitter is on
  y1  = vertexFromIds(D.primaryIds); wi1 = normalize(D.camPos - pos(y1))
  if X.k == 2 || (X.k == NONE && X.d == 2) || (X.forcedNee && X.d == 2):  yPrev = y1, preV = none
  else:
     rr = replay(X.initSeed, y1, upTo = (X.k == NONE ? X.d-1 : X.k-1), thr)   // O1 for pairs j ≤ k-2 (or ≤ d-1 if ∅), O0 misses
     if rr.status != OK: return FAIL(rr.status)
  if X.k == NONE:                                          // O3: terminal pair must FAIL, same technique/length
     if rr.terminal.kind != X.terminalKind: return FAIL
     if rcPairTest(rr.yLast, rr.eLast, rr.terminal, -, thr).pass: return FAIL
     return OK(F = rr.F, J = 1)
  xk  = rcVertexFromIds(X); wP = normalize(pos(xk) - pos(yPrev))
  eY  = X.forcedNee ? RcEvent(0,0,-, bsdfPdfMarginal(yPrev, wiPrev, wP))
                    : RcEvent(X.ellPrev, 0, alpha(yPrev, X.ellPrev), bsdfPdfMarginal(yPrev, wiPrev, wP))
  if preV exists && rcPairTest(preV, preE, yPrev, eY, thr).pass: return FAIL(EARLY_PAIR k-1)   // O1 last pre-rc pair
  if !X.forcedNee:
     eX = (X.kindK == SURFACE) ? (X.suffixIsNee ? RcEvent(0,0,-, bsdfPdfMarginal(xk, -wP, X.rcWi))
                                               : RcEvent(X.ellK, isDelta(X.ellK), -, bsdfPdfMarginal(xk, -wP, X.rcWi)))
                               : none
     if !rcPairTest(yPrev, eY, xk, eX, thr).pass: return FAIL(RC_PAIR)                        // O2
  if pJoint(yPrev, wP, X.ellPrev) <= 0 || (caseA && pJoint(xk, X.rcWi, X.ellK | -wP) <= 0): return FAIL(LOBE)
  if !visible(yPrev, xk /* skip both prims, ignore analytic lights */): return FAIL(VIS)
  jNum = jointNumerator(case); J = jNum / X.jDen
  return OK(F = evalIntegrand(...), J, jNum)
}
```

The replay loop inside `replay(...)`, per replayed vertex, is the same as the base loop's steps (3) and (4) but with **no NEE, no RR and no candidate streaming**. When a shared pair passes, it returns `FAIL(EARLY_PAIR j)`. It records `(preV, preE, yPrev, wiPrev, thp)`.

### 9.2 Most likely asymmetry bugs, ranked by (probability × damage)

1. **No offset re-check at all** (EvanLuo). Also: checking O2 but not O1, or O1 but not the **last** pre-rc pair with the **reconnection** direction ω'. Replaying at y_{k−1} with the replayed random direction instead of ω' is a close relative.
2. **invFP on the offset uses the base's stored p^x_k** (RTXDI approximation) instead of recomputing p̄^y_k with incoming from y_{k−1}.
3. **An NEE candidate's rc depends on the BSDF sample drawn later at the same vertex.** This happens with a lazy update of the streamed candidate, or when a tree-level rc found at pair b is applied to the NEE candidate at x_b.
4. **Wrong domain for R_pri.**
   - the source pixel's R_pri used in the destination;
   - the current camera used for the temporal inverse shift;
   - an un-jittered primary for R_pri while the domain uses a jittered one;
   - R_pri recomputed with a different formula instead of the stored `thr`.
5. **τ jitter drawn from pixel, frame or resampling RNG** (non-bijective), or a different jitter dimension in base and replay.
6. **Asymmetric neighbour acceptance** (EV-C3). Also:
   - S2 re-evaluating A with swapped arguments or in a different pipeline;
   - "slot not computed" treated as FAILED;
   - FAILED partners dropped from S_c.
7. **`jDen` not updated on selection** (EV-C4), or updated with marginal instead of joint pdfs, or with the shading-normal cosine.
8. **Marginal and joint swapped.** The Jacobian needs joint pdfs (S-Eq.4). Footprints need marginal (S-§3).
9. **Shading vs geometric normal mixed** across G, the footprints and R_pri, or different in base vs shift.
10. **Wrong lobe for R at y_{k−1}.** The offset's replay-sampled lobe is used instead of the **copied** ℓ_{k−1}. Or the α unit is wrong (GGX α vs perceptual; EvanLuo compares GGX α).
11. **Delta rules only on one side.** Example: the base rejects a delta ℓ_k and the offset does not.
12. **Visibility vs closest-hit mismatch.** The base segment x_{k−1}→x_k came from a closest-hit ray. The inverse shift re-tests it with a shadow ray whose epsilon or `t_max` can self-intersect or stop short, giving "inverse undefined".
    - Fix: skip both endpoint primitives, use `t_max = distance`, and apply Wächter–Binder offsets from the geometric normals at both ends (WP §8.3).
    - One-sided `max(dot(n,−Δ),0)` with *oriented* normals (EvanLuo) is fine only if the orientation rule is identical on both sides.
13. **Replay RNG desync.** NEE or RR randoms in the replayed stream; variable-count rejection sampling in VNDF; lobe-selection draws depending on branch history. Fix: counter-based dimensions (WP §8.5).
14. **Pass-through lights treated as vertices** in the replayed path (changing vertex indices). Also a Cycles `transparent_max_bounces` cap applied in Blender but not replicated. Set it very high in the reference, per CC §2.8.
15. **K_max or max-length caps applied in the base but not in the offset checks.**
16. **Lossy bary or ω_k compression** used by the shift while the base decided on exact values.
17. **FP (tiny-measure) class.** FMA/reassociation differences between pipelines; `>=` vs `>` mismatches (the paper uses ≥, RTXDI uses >; pick one); floors (10⁻⁶, 10⁻¹²) applied in one place only; `override` constants folded differently.

---

## 10. GPU invertibility and Jacobian-reciprocity tests (VH T2–T5 made concrete)

### 10.1 Signature compared (IDs, not floats)

`σ(x̄)` consists of:
- d, technique, k (or ∅), forced flag, light kind;
- terminal light identity: (lightId, uv-bits) or (inst, prim);
- rc (inst, prim), plus ℓ_{k−1} and ℓ_k;
- **replayed prefix** (inst, prim) and lobe for x₂…x_{k−1};
- `initSeed`.

Positions, barycentrics and F are compared only with tolerances and **reported**, never used for pass/fail. This follows VH T2.

### 10.2 Test list

**T3-0 Self-shift (identity), per pixel.**
- Use every path-tree candidate, not only the RIS-selected one. Initial sampling in test mode writes all candidates.
- Compute `T_{p→p}` with the production shift in the same frame, domain and `thr`.
- Require: defined; σ equal; |log₂J| < 2⁻¹⁶; |F(ȳ)/F(x̄) − 1| < 10⁻⁴ per channel.
- This compares the initial-sampling code path with the replay code path directly, and is the fastest detector of rules 1–4 and 8–13.

**T3-1 Spatial round trip.**
- Random pixel pairs (i,j): offsets from the three pairing textures (σ = 16) plus a uniform disk of radius 1–30 px.
- `ȳ = T_{i→j}(x̄)`. If defined, apply the §7.4 update to ȳ's reservoir fields, then `x̄' = T_{j→i}(ȳ)`.
- Require x̄' defined and `σ(x̄') = σ(x̄)`.
- Run in both directions (candidates born in j too), which covers definedness symmetry.

**T3-2 Temporal round trip.**
- Base in (p', t−1) using the previous camera, previous V-buffer and previous `thr`. Forward to (p, t), then inverse back.
- Camera motions: translation, rotation, FOV-constant zoom-in and zoom-out (changes R_pri). Jitter on and off.
- Static lights first. Moving lights come under Gap 2.

**T3-3 Paired-acceptance symmetry.** For every pixel and slot:
- `A(p,q) == A(q,p)` bitwise;
- slot status "accepted-ness" equal on both sides;
- `partner(partner(p)) == p` after each per-frame flip, mirror, transpose and offset.

**T3-4 Case coverage.** Force scene and material configurations so that each §3.5 case (a)–(h) occurs ≥ 10⁵ times:
- NEE-final rc (g), forced NEE (b) with d = 2 and d ≥ 3, emitter rc (c), pass-through analytic light (d), environment (e, f), full replay (∅), and k = 2 (h);
- mixed-lobe materials (Lambert+GGX constant Mix; V1 BSDF of CR Gap 3) where the copied lobe is missing at y₁;
- delta (mirror) vertices before and after candidate rc vertices.

**T3-5 Chained C4 (transitivity).**
- `ȳ = T_{i→j}(x̄)`, select it (update `jDen`), then `z̄ = T_{j→l}(ȳ)`.
- Compare `J_{i→l}` computed directly from x̄ with `J_{i→j}·J_{j→l}`.
- Require agreement within 10⁻⁴ relative whenever σ(z̄) equals σ(T_{i→l}(x̄)). A missing C4 update fails this immediately.

**T4 Reciprocity.** For every successful round trip, `|log(J_{i→j}(x̄)·J_{j→i}(ȳ))| < 10⁻⁴`.
- Also histogram `log₂J`. Mass should concentrate near 0 (P-Eq.6).

**T5 Change-of-variables census.**
- `E_{ū}[J·1{fwd ok}]` from i must equal `Pr[inverse ok]` from j.
- Do this for h ≡ 1 and for `h = F_j/(1+F_j)`, with 10⁷–10⁸ samples per side and a two-sample z-test (VH T5).
- This catches domain-restriction asymmetries that happen to round-trip. Example: a check applied only in the forward direction.

**T6b Cross-consistency (VH T6).** For each neighbour sample: `p̂_j(X_j)/J_{j→c}` must equal `p̂_j(T_{c→j}(Y_j))·J_{c→j}(Y_j)` within 10⁻³.

**T3-D Dual implementation.**
- A TypeScript float64 reference of `k*` and O0–O3 re-derives every GPU decision on 10⁵ dumped paths, from dumped IDs, stored direction bits, `thr` and material parameters.
- **Any disagreement with margin > 2⁻¹⁶ is a logic bug.**
- This is the only test that finds a bug implemented identically on both GPU sides, e.g. the wrong vertex for the cosine.

### 10.3 Violation classes and targets

Every failed check records which sub-test flipped (O1 at pair j / O2 D, R, F or I / O0 miss, lobe or visibility / σ mismatch at vertex j) and `margin` from `rcPairTest`.

| Class | Definition | Target |
|---|---|---|
| **LOGIC** | Any violation with \|margin\| ≥ 2⁻¹⁶ (for roughness, \|α − α_min\| ≥ 10⁻⁶), any σ mismatch not explained by a margin flip, any T3-3 asymmetry, any T3-D disagreement | **0** in ≥ 10⁷ round trips per case (T3-0 to T3-4), in the production multi-pipeline build |
| **FP-BOUNDARY** | Predicate flips with \|margin\| < 2⁻¹⁶; replay divergence at a triangle edge (both hits within 10⁻⁶ of an edge) | Report the rate. Expected ≲ 10⁻⁶ per round trip [INFERENCE]. **Must be exactly 0** in the single-pipeline validation build (§8.3 item 7) and with `strictMath`. |
| **VIS** | Inverse visibility fails while forward succeeded | 0 after the §9.2 item-12 fix. Any residue is a ray-offset bug. |
| **J** | T4 or T3-5 tolerance exceeded | 0 |

### 10.4 Kernel sketch

Deterministic, counter-based. No float atomics. One `atomicAdd` per violation record.

```wgsl
@compute @workgroup_size(64)
fn t3RoundTrip(@builtin(global_invocation_id) gid: vec3u) {
  let trial = gid.x;                                    // 0 .. N-1
  let cfg   = trialConfig(trial);                        // (i, j or temporal pair, candidate index, seed) from a hash of trial
  let X     = loadCandidate(cfg);                        // written by initial sampling in test mode (all tree candidates)
  let Y     = hybridShift(X, domainOf(cfg.dst));
  if (!Y.ok) { countForwardUndefined(Y.reason); return; }
  let Xs    = applySelectionUpdate(X, Y);                // §7.4 (F, jDen, lightPdf)
  let X2    = hybridShift(Xs, domainOf(cfg.src));
  if (!X2.ok)                 { logViolation(trial, X2.reason, X2.margin); return; }
  if (!sigEqual(X, X2.res))   { logViolation(trial, SIG_MISMATCH, firstDiffVertex(X, X2.res)); }
  let lr = log2(Y.J) + log2(X2.J);
  if (abs(lr) > 1.5e-4)       { logViolation(trial, J_RECIPROCITY, lr); }
  histogramLog2J(Y.J);
}
```

Run it:
- on the VH §8 scene suite plus a Cornell box with Lambert, GGX r ∈ {0.1, 0.2, 0.3, 0.5}, a mirror, a mixed plastic, an emissive triangle, point/spot/rect lights and an env map;
- with **dense stratified PSS sweeps** for short paths (d ≤ 4: 2²⁰ stratified ū per pixel pair on a 32×32 pixel grid). This is the "exhaustive" component: it enumerates every path class reachable in the toy scene rather than relying on random coverage.

---

## 11. Open questions / [UNVERIFIED]

1. Whether the Enhanced authors' own code computes invFP with the offset's recomputed p^y_k or with a stored value. The paper and supplement do not say. Our spec requires recomputation because it is exact.
2. Whether the authors' NEE candidates can reconnect at x_{d−1} (RTXDI/Falcor behaviour) or always force the light (EvanLuo). The paper says only "if no earlier reconnection is found". Both are unbiased; §3.5 item 1 chooses the RTXDI behaviour.
3. The S-Eq.26 PDF-proxy scale: printed `1/p² ≥ α_min` vs RTXDI's `1/√p ≥ α` (EP §3.6). The proxy is irrelevant for parametric materials, where §3 uses per-lobe roughness.
4. The measured FP-BOUNDARY rate on the M5 Pro under relaxed math has not been measured. §8.2 gives only an estimate. Also unconfirmed: whether Tint → Metal inlining actually produces different contraction for the same WGSL function in different pipelines.
5. FOV and resolution dependence of τ (EP §3.9 caveat). Pairing offsets are in pixels, but R_pri is FOV-independent. Only variance is affected.
6. RTXDI's pdf-threshold rule is applied only to mid-path rc and not to emitter rc. The paper's single-vertex rule applies generally. We follow the paper.
7. Whether `strictMath` changes Dawn's `#pragma METAL fp math_mode` from relaxed to safe on macOS 26. WP §8.1 reports `fastMathEnabled = !strictMath` only for older macOS.

---

## 12. Sources

**Enhanced paper.** Page images read directly:
- `scratchpad/pages/enhanced/p-04.png`: §2.3 hybrid shift, Eq. 2, p^x_k := 1 for k = d.
- `p-06.png`: §4, Eq. 4a/4b, fn. 4.
- `p-07.png`: Eq. 5, c = 0.02, single-vertex threshold; §4.1 Eq. 6a/6b, ray footprint.
- `p-08.png`: inverse footprint; §4.2; fn. 5 and 6.
- `p-11.png`: §6.2.3 forced NEE; §6.2.4 RR; §6.3 vector weights.
- `p-12.png`: §7 setup.
- `p-14.png`: §7.2, α = 0.2.

**Supplement** §1–§6. Cached text `~/.claude/projects/-Users-mark-boss-Dev-WebGPURestirPT/4e489933-ea73-491f-b7b7-6d0aab1ffce1/tool-results/bkusnhdet.txt`, lines 12–93 (§1, S-Eq.4), 94–271 (§2, S-Eq.5–25), 272–303 (§3), 304–325 (§4, S-Eq.26), 326–381 (§5–6). URL: https://research.nvidia.com/labs/rtr/publication/lin2026restirptenhanced/lin2026restirptenhanced_supplemental.pdf

**GRIS 2022.** `scratchpad/pages/gris/p-13.png` (§7.2–7.5) and `p-14.png` (§7.5–7.6, §8).

**Falcor** `DQLin/ReSTIR_PT@8d12332` (BSD-3):
- `PathTracer.slang:1085-1118, 1157-1261, 1263-1440`
- `Shift.slang:116-200, 434-470, 550-572`

**EvanLuo42/ReSTIR-PT-Enhanced** HEAD (BSD-3):
- `Common/EnhancedPolicy.slang:1-246`
- `Common/HybridShift.slang:1-528`
- `Path/PathSampling.slang:1-664`
- `Spatial/SpatialShift.cs.slang:1-125`, `Spatial/SpatialReplay.cs.slang:1-65`, `Spatial/SpatialResample.cs.slang:1-152`
- `Common/Pairing.slang:1-68`, `Common/GRIS.slang:1-105`, `Common/Surface.slang:27-67`
- `Temporal/TemporalShift.cs.slang:1-189`
- `Common/SurfaceLoad.slang:104`
- `Params.slang:77-117`

**RTXDI-Library** main (proprietary; semantics only, no code reproduced):
- `Include/Rtxdi/PT/PathReconnectibility.hlsli`
- `InitialSamplingPathTracerContext.hlsli` (`AnalyzePathReconnectibilityBeforeTrace`, `RecordEmissiveLightSample`, `RecordNeeLightSample`)
- `HybridShiftPathTracerContext.hlsli` (`AnalyzePathReconnectibilityBeforeTrace`, `RecordPathIntersection`, `Init`)
- `HybridShift.hlsli` (`ValidateInvertibilityCondition`, `UpdateReconnectionForRTXDIConnectedLight`)

**Prior reports** (`/Users/mark.boss/.claude/plans/do-a-deep-dive-shimmering-ritchie-agent-*.md`):
- enhanced-paper `a91ab1d861819b276` (§1.3, §2.2–2.6, §3, §5.3) and its verification `acb243dbea3ce2a29` (C2–C4, C8, C11, O1, O2, O4–O9)
- gris-math `a7f08dec9968f0324` (§3.4, §6.3–6.8) and its verification `a147516e5d2fb9558` (C1–C3, C9, O5)
- reference-code `a3b89657b22d15f34` (§1.4, §1.5.2, §1.6.3, §2.2.4–2.2.5, §2.3)
- validation-harness `a26c6df2c3f5e4af0` (T1–T6, §5.6)
- webgpu-platform `a64ceeae6903be301` (§8.1–8.5)
- critique `a40267d04641507fa` (X1, X13, X16, Gap 4)
- cycles-conventions `a2d6cc1a462b20a80` (§2.6–2.9, §4.11, §5.1)
