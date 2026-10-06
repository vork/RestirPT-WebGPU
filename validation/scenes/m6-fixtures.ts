// M6 T3 fixtures (restir-m6-api.md §4 T3-M6; browser-safe, built in the Chrome test like make-m4.ts's t3Scene):
//   t3_modeb_256  t3_cases_256 + three extra analytic area lights placed where BSDF rays cross them front-facing with
//                 geometry behind: a vertical rect in the middle of the room facing the camera, a small downward rect
//                 below the ceiling (crossed by upward rays from the floor and the spheres) and a disk facing −X near the
//                 right wall; rendered in Mode B (every crossing a BSDF_ANALYTIC candidate: cases d-ana, c-ana, deep, ∅).
//   t3_alpha_256  t3_cases_256 + six alpha-MASK leaf cards (procedural texture, α ∈ {0, 1}) standing between the camera-side
//                 floor and the back of the room, so reconnection segments cross cutouts and opaque leaves; the card
//                 rectangles are returned for the T3 kernel's alpha-card counter.
//   t3_modeb_rare_256  the same lights on t3_rare_256 (glossy r 0.1 floor / back wall: pair 2 fails R, so ∅ crossing
//                 paths (none-ana) are frequent).
//   t3_glass_pane_256  t3_glass_256 + a thin vertical rough-glass pane (r 0.3) across the room: neighbouring pixels see
//                 its two sides, so reconnections to a G_R vertex on the pane flip sides (glass side-flip counter).
//   t3_glass_256  (make-m4.ts) is gating in M6 (D13 → rung 3.9): no change here.
import type { EnvironmentData, MaterialData, TextureData } from '../../src/core/scene/types.ts';
import type { LightMode } from '../../src/core/render/lights-gpu.ts';
import { T3_MAT, light, lightToward, principled, t3Scene, type T3Scene, type V3 } from './make-m4.ts';

export type T3M6Variant = 't3_modeb_256' | 't3_modeb_rare_256' | 't3_alpha_256' | 't3_glass_pane_256';
export interface T3Card { c: V3; u: V3; v: V3; hu: number; hv: number }

const nrm = (a: V3): V3 => { const l = Math.hypot(a[0], a[1], a[2]); return [a[0] / l, a[1] / l, a[2] / l]; };

function leafTexture(): TextureData {
  let a = 0x1234567;
  const R = () => { a = (Math.imul(a, 1664525) + 1013904223) >>> 0; return a / 4294967296; };
  const blobs = Array.from({ length: 14 }, () => ({ x: 20 + 88 * R(), y: 16 + 96 * R(), rx: 8 + 14 * R(), ry: 5 + 9 * R(), r: Math.PI * R() }));
  const W = 128, H = 128;
  const pixels = new Uint8Array(W * H * 4);
  for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
    let inside = Math.abs(x - 64) < 2 && y > 20;
    for (const b of blobs) {
      const dx = x + 0.5 - b.x, dy = y + 0.5 - b.y, c = Math.cos(b.r), s = Math.sin(b.r);
      const u = (c * dx + s * dy) / b.rx, v = (-s * dx + c * dy) / b.ry;
      if (u * u + v * v <= 1) inside = true;
    }
    const o = 4 * (y * W + x);
    pixels[o] = 40 + (x & 31); pixels[o + 1] = 110 + (y & 63); pixels[o + 2] = 40; pixels[o + 3] = inside ? 255 : 0;
  }
  return { name: 'leaves', width: W, height: H, pixels, wrapS: 'repeat', wrapT: 'repeat', filter: 'linear' };
}

export function t3M6Scene(variant: T3M6Variant, env?: EnvironmentData): T3Scene & { cards: T3Card[]; lightMode: LightMode } {
  const cards: T3Card[] = [];
  if (variant === 't3_glass_pane_256') {
    const t = t3Scene('t3_glass_256', env, (b) => {
      const x = 0.12, m = T3_MAT.roughGlass;
      b.mesh.tri([x, 0.02, 0.6], [x, 0.02, -1.1], [x, 1.4, -1.1], m).tri([x, 0.02, 0.6], [x, 1.4, -1.1], [x, 1.4, 0.6], m);
    });
    return { ...t, notes: `${variant}: t3_glass_256 + a thin rough-glass pane (side flips)`, cards, lightMode: 'A' };
  }
  if (variant === 't3_modeb_256' || variant === 't3_modeb_rare_256') {
    const t = t3Scene(variant === 't3_modeb_256' ? 't3_cases_256' : 't3_rare_256', env, (b) => {
      b.lights.push(
        light(6, 'rect', lightToward([0, 0, 1], [0.05, 0.95, -0.55]), 12, { sizeX: 0.5, sizeY: 0.35 }),
        light(7, 'rect', lightToward([0, -1, 0], [0.35, 1.55, 0.15]), 10, { sizeX: 0.35, sizeY: 0.35 }),
        light(8, 'disk', lightToward([-1, 0, 0], [1.3, 0.8, -0.55]), 8, { sizeX: 0.45 }),
      );
    });
    return { ...t, notes: `${variant}: t3_cases_256 + 3 crossing lights, Mode B (restir-m6-api.md §4)`, cards, lightMode: 'B' };
  }
  const t = t3Scene('t3_cases_256', env, (b) => {
    b.textures.push(leafTexture());
    const mat = b.materials.length;
    b.materials.push(principled('leaf_card', {
      baseColorTexture: { texture: 0, texCoord: 0 }, alphaMode: 'MASK', alphaCutoff: 0.5, roughnessFactor: 0.6,
    } as Partial<MaterialData>));
    const specs: [number, number, number, number, number, number][] = [   // cx, cz, rot, w, h, y0
      [-0.55, 0.35, 0.35, 0.7, 0.9, 0.02], [0.35, 0.25, -0.5, 0.6, 1.0, 0.02], [-0.1, -0.45, 0.1, 0.8, 0.6, 0.55],
      [0.75, -0.75, 0.9, 0.5, 0.8, 0.05], [-1.0, -0.7, -0.3, 0.5, 0.7, 0.4], [0.1, 0.6, 1.2, 0.5, 0.5, 0.9],
    ];
    for (const [cx, cz, rot, w, h, y0] of specs) {
      const c = Math.cos(rot), s = Math.sin(rot);
      const p = (u: number, y: number): V3 => [cx + c * u, y, cz - s * u];
      const q = [p(-w / 2, y0), p(w / 2, y0), p(w / 2, y0 + h), p(-w / 2, y0 + h)];
      const uv: [number, number][] = [[0, 1], [1, 1], [1, 0], [0, 0]];
      b.mesh.tri(q[0], q[1], q[2], mat, [uv[0], uv[1], uv[2]]).tri(q[0], q[2], q[3], mat, [uv[0], uv[2], uv[3]]);
      cards.push({ c: [cx, y0 + h / 2, cz], u: nrm([c, 0, -s]), v: [0, 1, 0], hu: w / 2, hv: h / 2 });
    }
  });
  return { ...t, notes: `${variant}: t3_cases_256 + 6 alpha-MASK leaf cards (restir-m6-api.md §4)`, cards, lightMode: 'A' };
}
