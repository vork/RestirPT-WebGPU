// Independent (sampler-free) f64 quadrature of the directional albedo ∫ f_cos(V, L) dω_L of the reference BSDF.
// Specular part in NDF space: with w ∈ [0,1), tan²θ_h = a2·w/(1−w), φ uniform, the map (w, φ) → H has density
// D(H)·cosθ_h (GGX NDF sampling), and dω_L = 4(V·H) dω_H, so
//   ∫ f_S dω_L = ∫∫ F(V·H)·G2·(V·H)/(N·V·cosθ_h) · dw dφ/(2π)   over V·H > 0, N·L ≥ 0.
// The midpoint rule runs in s with w = sin²s (tanθ_h = α·tan s), dw = sin 2s ds, which resolves both the NDF peak
// (θ_h ~ α) and the grazing region (w → 1) that dominates at small N·V.
// Lambert part: w_D exactly (∫ cos/π = 1). Singular S: F(N·V) (mirror).
import { FS, ggxLambda, prepare, type MatParams, type V3 } from './bsdf-ref.ts';

export function albedoQuadrature(p: MatParams, V: V3, n = 1024): V3 {
  const mu = V[2];
  const c = prepare(p, mu);
  const out: V3 = [0, 0, 0];
  if (c.hasD) for (let k = 0; k < 3; k++) out[k] += c.wD[k];
  if (!c.hasS || !(mu > 0)) return out;
  if (c.singular) {
    const F = FS(c, mu);
    for (let k = 0; k < 3; k++) out[k] += F[k];
    return out;
  }
  const lI = ggxLambda(c.a2, mu);
  const acc: V3 = [0, 0, 0];
  for (let i = 0; i < n; i++) {
    const sv = ((i + 0.5) / n) * (Math.PI / 2);
    const dw = Math.sin(2 * sv) * (Math.PI / 2);   // dw/di·n
    const t2 = c.alpha * c.alpha * Math.tan(sv) ** 2;
    const ch = 1 / Math.sqrt(1 + t2);
    const sh = Math.sqrt(t2) * ch;
    for (let j = 0; j < n; j++) {
      const phi = ((j + 0.5) / n) * 2 * Math.PI;
      const H: V3 = [sh * Math.cos(phi), sh * Math.sin(phi), ch];
      const vh = V[0] * H[0] + V[1] * H[1] + V[2] * H[2];
      if (!(vh > 0)) continue;
      const Lz = 2 * vh * ch - mu;
      if (Lz < 0) continue;
      const G2 = 1 / (1 + lI + ggxLambda(c.a2, Lz));
      const F = FS(c, vh);
      const g = (G2 * vh * dw) / (mu * ch);
      for (let k = 0; k < 3; k++) acc[k] += F[k] * g;
    }
  }
  for (let k = 0; k < 3; k++) out[k] += acc[k] / (n * n);
  return out;
}
