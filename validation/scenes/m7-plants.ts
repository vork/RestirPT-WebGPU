// M7 plant predictions (docs/decisions/m7-api.md §5; asserted by tests/scene/plant-m7.test.ts, read by gate-m7.ts), derived on the CPU BEFORE any planted render was measured (the
// TD29 / E-18 rule: a prediction revised after a measurement is confirmed only on fresh disjoint seeds).
//
// Scene m7_nm_flat_256 (make-m7.ts): three panels on the back wall (z = −0.497, facing +z), Principled ρ 0.8 r 0.7, all
// with constant-tilt normal maps toward tangent-space +y (= +B):
//   P1 tilt 35°, strength 1, ordinary UVs (MikkTSpace w = +1)        → N tilted 35° UP (+y)
//   P2 = P1 with mirrored u (w = −1)                                  → N tilted 35° UP (B = w·(n × T) with T = −x)
//   P3 tilt 60° toward −B (DOWN), strength 0.5 (Cycles: c.xy·0.5, c.z = mix(1, c.z, 0.5) → θ = atan(0.5 sin 60°, (1 + cos 60°)/2)
//      = 30.0° DOWN; chosen (before any measurement) where the direct irradiance changes monotonically with the tilt: the
//      first design, 50° toward +B, sat at the irradiance maximum, Δ = +0.1 %)
// Plants (NM_PLANT):
//   sign      w := +1: P2's B flips to −y ⇒ N tilted 35° DOWN; P1 / P3 / every other w = +1 surface unchanged
//   strength  glTF-style: c.z not mixed ⇒ P3 θ' = atan(0.5 sin 60°, cos 60°) = 40.9° DOWN; also the s 0.6 bumps of the
//             back wall (more tilt; sign of the wall average not predicted)
// The sign of each panel's change is the sign of the change of its direct diffuse irradiance from the scene's two
// lights (ceiling rect 12 W 0.3 × 0.3 m one-sided, spot 6 W 70° blend 0.2), integrated over the panel (a 16 × 16 grid,
// the rect by a 32 × 32 quadrature, the spot as a point with its smoothstep profile). Direct light dominates these
// rough panels (the indirect part changes with the same N but is a fraction of it), so the Stage-A plants (our PT vs
// Cycles) and the Stage-B plants (the same NM_PLANT in every ReSTIR pass but the path tree: shifted paths through the
// panel are evaluated with the planted normal) are predicted with this sign on the panel's mask; "only": no other region.
import { readFileSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { M7_PANELS } from './make-m7.ts';

type V3 = [number, number, number];
const ROOT = fileURLToPath(new URL('../../', import.meta.url));
const sub = (a: V3, b: V3): V3 => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
const dot = (a: V3, b: V3) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
const nrm = (a: V3): V3 => { const l = Math.hypot(a[0], a[1], a[2]); return [a[0] / l, a[1] / l, a[2] / l]; };
const deg = Math.PI / 180;

interface L { type: string; matrix: number[]; power: number; sizeX?: number; sizeY?: number; spotSize?: number; spotBlend?: number }
function lights(): L[] {
  const j = JSON.parse(readFileSync(path.join(ROOT, 'validation/out/m7/scenes/m7_nm_flat_256/scene.json'), 'utf8'));
  return j.lights as L[];
}

/** Direct irradiance E(x, N) from the package lights (Blender units: point/spot I = P/(4π); rect L = P/(π A), one-sided). */
export function directE(x: V3, N: V3, ls: L[]): number {
  let E = 0;
  for (const l of ls) {
    const p: V3 = [l.matrix[12], l.matrix[13], l.matrix[14]];
    const ax: V3 = nrm([-l.matrix[8], -l.matrix[9], -l.matrix[10]]);   // emission axis −Z_obj
    if (l.type === 'spot' || l.type === 'point') {
      const d = sub(p, x), r2 = dot(d, d), w = nrm(d);
      const c = dot(N, w);
      if (c <= 0) continue;
      let S = 1;
      if (l.type === 'spot') {
        const cosH = Math.cos(l.spotSize! / 2);
        const t = (dot(nrm(sub(x, p)), ax) - cosH) / ((1 - cosH) * l.spotBlend!);
        S = t <= 0 ? 0 : t >= 1 ? 1 : 3 * t * t - 2 * t * t * t;
      }
      E += (l.power / (4 * Math.PI)) * S * c / r2;
    } else if (l.type === 'rect') {
      const X: V3 = [l.matrix[0], l.matrix[1], l.matrix[2]], Y: V3 = [l.matrix[4], l.matrix[5], l.matrix[6]];
      const A = l.sizeX! * l.sizeY!, Lr = l.power / (Math.PI * A), n = 32;
      for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) {
        const a = ((i + 0.5) / n - 0.5) * l.sizeX!, b = ((j + 0.5) / n - 0.5) * l.sizeY!;
        const q: V3 = [p[0] + a * X[0] + b * Y[0], p[1] + a * X[1] + b * Y[1], p[2] + a * X[2] + b * Y[2]];
        const d = sub(q, x), r2 = dot(d, d), w = nrm(d);
        const cr = dot(N, w), cl = -dot(ax, w);
        if (cr > 0 && cl > 0) E += Lr * cr * cl / r2 * (A / (n * n));
      }
    }
  }
  return E;
}

/** Mean direct irradiance over a panel (16 × 16 points) for a normal tilted by θ toward world +y (θ < 0: toward −y). */
function panelE(panel: { x: [number, number]; y: [number, number] }, theta: number, ls: L[]): number {
  const N: V3 = [0, Math.sin(theta), Math.cos(theta)];
  let s = 0;
  for (let i = 0; i < 16; i++) for (let j = 0; j < 16; j++) {
    const x: V3 = [panel.x[0] + (panel.x[1] - panel.x[0]) * (i + 0.5) / 16, panel.y[0] + (panel.y[1] - panel.y[0]) * (j + 0.5) / 16, -0.497];
    s += directE(x, N, ls);
  }
  return s / 256;
}

/** The predictions table (consumed by gate-m7.ts; sign '+' / '-' or 'detect' when |Δ| < 2 %). */
export function m7PlantPredictions(): Record<string, { region: string; sign: '+' | '-' | 'detect'; rel: number; derivation: string }> {
  const ls = lights();
  const th35 = 35 * deg, th60 = 60 * deg;
  const thP3 = Math.atan2(0.5 * Math.sin(th60), (1 + Math.cos(th60)) / 2), thP3g = Math.atan2(0.5 * Math.sin(th60), Math.cos(th60));
  const e2 = panelE(M7_PANELS.P2, th35, ls), e2p = panelE(M7_PANELS.P2, -th35, ls);
  const e3 = panelE(M7_PANELS.P3, -thP3, ls), e3p = panelE(M7_PANELS.P3, -thP3g, ls);
  const sg = (r: number): '+' | '-' | 'detect' => (Math.abs(r) < 0.02 ? 'detect' : r > 0 ? '+' : '-');
  const r2 = e2p / e2 - 1, r3 = e3p / e3 - 1;
  return {
    sign: { region: 'M_P2', sign: sg(r2), rel: r2, derivation: `P2 direct E: 35° up ${e2.toFixed(4)} → 35° down ${e2p.toFixed(4)} (${(100 * r2).toFixed(1)} %)` },
    strength: { region: 'M_P3', sign: sg(r3), rel: r3, derivation: `P3 direct E: ${(thP3 / deg).toFixed(1)}° down ${e3.toFixed(4)} → ${(thP3g / deg).toFixed(1)}° down ${e3p.toFixed(4)} (${(100 * r3).toFixed(1)} %)` },
  };
}

