// M7 UsdUVTexture support (docs/decisions/m7-api.md §3.1): rebuild the UsdPreviewSurface texture bindings from the
// root-layer scan (LightUSD rc4 reports texture paths and wrap modes but not the connected output channel, scale /
// bias, the primvar reader or authored-vs-default wraps) and map them the way Blender 5.2's wm.usd_import does
// (measured, m7-api.md §3.1 table): one Image Texture per UsdUVTexture, Linear interpolation, the colour space from
// sourceColorSpace, ONE extension mode = wrapS ('repeat' when unauthored), a connected single-channel output through
// Separate Color, normal → Normal Map node (tangent space, strength 1; scale / bias ignored), the connected input's
// constant value ignored. Our MaterialData has fixed slots, so:
//   diffuseColor.rgb (+ opacity.a of the SAME texture)  → baseColorTexture, factor 1 (opacity factor kept)
//   roughness.<c> + metallic.<c>                         → metallicRoughnessTexture: the source texture itself when roughness
//                                                          reads G and metallic B of one file; otherwise a SYNTHESISED
//                                                          RGBA8 texture (G ← roughness channel, B ← metallic channel, 255
//                                                          for a constant input whose value becomes the factor), texel for
//                                                          texel (same size required, no resampling: plan §1.6)
//   normal.rgb                                            → normalTexture, scale 1
//   emissiveColor.rgb                                     → emissiveTexture, factor 1
// Anything else (opacity from another texture, UsdTransform2d, non-st primvars, 'black' wraps, sizes that differ for a
// synthesised pair) is warned and falls back to the constant (never silently approximated).
import type { MaterialData, TextureData, TextureRef, WrapMode } from '../types.ts';
import type { UsdaScan } from './usda-scan.ts';

export interface DecodedUsdImage { width: number; height: number; pixels: Uint8Array }
export type UsdChannel = 'rgb' | 'r' | 'g' | 'b' | 'a';
export interface UsdTexBinding { tex: string; file: string; channel: UsdChannel; wrap: WrapMode; wrapNote?: string; colorSpace: string; scale?: number[]; bias?: number[]; varname?: string }

const unq = (s: string | undefined): string | undefined => (s === undefined ? undefined : s.trim().replace(/^"(.*)"$/, '$1').replace(/^@+(.*?)@+$/, '$1'));
const nums = (s: string | undefined): number[] | undefined => (s && s.trim().startsWith('(') ? s.replace(/[()]/g, '').split(',').map(Number) : undefined);
/** `</a/b.outputs:rgb>` → ['/a/b', 'rgb']. */
const conn = (s: string | undefined): [string, string] | undefined => {
  const m = s && /<([^>]+)\.outputs:(\w+)>/.exec(s);
  return m ? [m[1], m[2]] : undefined;
};
const WRAP: Record<string, WrapMode> = { repeat: 'repeat', mirror: 'mirror-repeat', clamp: 'clamp-to-edge' };

/** The UsdPreviewSurface shader prim of a material (a child with info:id UsdPreviewSurface) and its texture bindings. */
export function previewSurfaceBindings(matPath: string, scan: UsdaScan | undefined, warn: (m: string) => void): { ps: string | null; bindings: Record<string, UsdTexBinding> } {
  const out: Record<string, UsdTexBinding> = {};
  if (!scan) return { ps: null, bindings: out };
  const ps = Object.keys(scan.attrs).find((p) => p.slice(0, p.lastIndexOf('/')) === matPath && unq(scan.attrs[p]['info:id']) === 'UsdPreviewSurface') ?? null;
  if (!ps) return { ps, bindings: out };
  for (const [k, v] of Object.entries(scan.attrs[ps])) {
    const m = /^inputs:(\w+)\.connect$/.exec(k);
    if (!m) continue;
    const c = conn(v);
    if (!c) continue;
    const [tex, ch] = c;
    const ta = scan.attrs[tex];
    if (!ta || unq(ta['info:id']) !== 'UsdUVTexture') { warn(`material ${matPath}: ${m[1]} is connected to a non-UsdUVTexture shader (${unq(ta?.['info:id']) ?? 'unknown'}); constant used`); continue; }
    if (!['rgb', 'r', 'g', 'b', 'a'].includes(ch)) { warn(`material ${matPath}: ${m[1]} reads output '${ch}'; constant used`); continue; }
    const file = unq(ta['inputs:file']);
    if (!file) { warn(`material ${matPath}: ${tex} has no inputs:file; constant used`); continue; }
    const wS = unq(ta['inputs:wrapS']), wT = unq(ta['inputs:wrapT']);
    let wrap: WrapMode = 'repeat', wrapNote: string | undefined;
    if (wS !== undefined && wS !== 'useMetadata') {
      if (WRAP[wS]) wrap = WRAP[wS];
      else wrapNote = `wrapS '${wS}' (Blender: CLIP) has no sampler equivalent: repeat used`;
    }
    if (wT !== undefined && wT !== wS) wrapNote = `${wrapNote ? `${wrapNote}; ` : ''}wrapT '${wT}' ≠ wrapS (Blender uses one extension mode, wrapS)`;
    let varname: string | undefined;
    const st = conn(ta['inputs:st.connect']);
    if (st) {
      const ra = scan.attrs[st[0]];
      const id = unq(ra?.['info:id']);
      if (id === 'UsdTransform2d') warn(`material ${matPath}: ${tex} uses a UsdTransform2d (not supported: identity used)`);
      varname = unq(ra?.['inputs:varname']);
    }
    out[m[1]] = { tex, file, channel: ch as UsdChannel, wrap, wrapNote, colorSpace: unq(ta['inputs:sourceColorSpace']) ?? 'auto', scale: nums(ta['inputs:scale']), bias: nums(ta['inputs:bias']), varname };
  }
  return { ps, bindings: out };
}

/** Texture table of one USD scene: decoded images by asset path, deduplicated TextureData entries. */
export class UsdTextureTable {
  readonly textures: TextureData[] = [];
  private readonly byKey = new Map<string, number>();
  constructor(private readonly images: Record<string, DecodedUsdImage | undefined>, private readonly warn: (m: string) => void) {}

  image(file: string): DecodedUsdImage | undefined {
    const im = this.images[file] ?? this.images[file.replace(/^\.\//, '')];
    if (!im) this.warn(`texture ${file}: not found / not decodable (PNG in Node; PNG/JPEG in browsers); constant used`);
    return im;
  }

  private add(key: string, name: string, im: DecodedUsdImage, wrap: WrapMode): TextureRef {
    let i = this.byKey.get(key);
    if (i === undefined) {
      i = this.textures.length;
      this.textures.push({ name, width: im.width, height: im.height, pixels: im.pixels, wrapS: wrap, wrapT: wrap, filter: 'linear' });
      this.byKey.set(key, i);
    }
    return { texture: i, texCoord: 0 };
  }

  /** The file itself as a texture. */
  file(b: UsdTexBinding): TextureRef | undefined {
    const im = this.image(b.file);
    return im ? this.add(`${b.file}|${b.wrap}`, b.file, im, b.wrap) : undefined;
  }

  /** A synthesised metallic-roughness texture: G ← rough channel, B ← metal channel (255 where an input is constant). */
  metalRough(rough: UsdTexBinding | undefined, metal: UsdTexBinding | undefined, mat: string): TextureRef | undefined {
    const ri = rough ? this.image(rough.file) : undefined, mi = metal ? this.image(metal.file) : undefined;
    if ((rough && !ri) || (metal && !mi)) return undefined;
    const base = (ri ?? mi)!;
    if (ri && mi && (ri.width !== mi.width || ri.height !== mi.height)) {
      this.warn(`material ${mat}: roughness (${ri.width}x${ri.height}) and metallic (${mi.width}x${mi.height}) textures differ in size: cannot pack without resampling; constants used`);
      return undefined;
    }
    const wrap = (rough ?? metal)!.wrap;
    if (rough && metal && rough.wrap !== metal.wrap) this.warn(`material ${mat}: roughness / metallic wrap modes differ (${rough.wrap} / ${metal.wrap}); ${wrap} used`);
    const ch = (c: UsdChannel | undefined) => ({ r: 0, g: 1, b: 2, a: 3, rgb: 0 })[c ?? 'r'];
    const key = `mr|${rough?.file ?? '-'}.${rough?.channel ?? '-'}|${metal?.file ?? '-'}.${metal?.channel ?? '-'}|${wrap}`;
    let i = this.byKey.get(key);
    if (i === undefined) {
      const n = base.width * base.height;
      const px = new Uint8Array(n * 4);
      for (let k = 0; k < n; k++) {
        px[4 * k] = 0;
        px[4 * k + 1] = ri ? ri.pixels[4 * k + ch(rough!.channel)] : 255;
        px[4 * k + 2] = mi ? mi.pixels[4 * k + ch(metal!.channel)] : 255;
        px[4 * k + 3] = 255;
      }
      i = this.textures.length;
      this.textures.push({ name: `metalRough(${rough?.file ?? 'const'}:${rough?.channel ?? ''}, ${metal?.file ?? 'const'}:${metal?.channel ?? ''})`, width: base.width, height: base.height, pixels: px, wrapS: wrap, wrapT: wrap, filter: 'linear' });
      this.byKey.set(key, i);
    }
    return { texture: i, texCoord: 0 };
  }
}

const isStdNormal = (b: UsdTexBinding) => (!b.scale || b.scale.slice(0, 3).every((x) => x === 2)) && (!b.bias || b.bias.slice(0, 3).every((x) => x === -1));
const isIdentity = (b: UsdTexBinding) => (!b.scale || b.scale.every((x) => x === 1)) && (!b.bias || b.bias.every((x) => x === 0));

/** Apply the texture bindings of one material to its MaterialData (in place). */
export function applyUsdTextures(md: MaterialData, path: string, bindings: Record<string, UsdTexBinding>, table: UsdTextureTable, warn: (m: string) => void): void {
  const b = bindings;
  for (const [k, v] of Object.entries(b)) {
    if (v.wrapNote) warn(`material ${path}: ${k}: ${v.wrapNote}`);
    if (v.varname && v.varname !== 'st') warn(`material ${path}: ${k} reads primvar '${v.varname}' (the loader's UV set is 'st' / the first texcoord)`);
    if (k !== 'normal' && !isIdentity(v)) warn(`material ${path}: ${k} texture scale / bias ignored (as Blender's importer does)`);
  }
  if (b.diffuseColor) {
    if (b.diffuseColor.channel !== 'rgb') warn(`material ${path}: diffuseColor reads '${b.diffuseColor.channel}' (only rgb is mapped); constant used`);
    else {
      if (b.diffuseColor.colorSpace === 'raw') warn(`material ${path}: diffuseColor texture with sourceColorSpace raw is sampled as sRGB here`);
      const r = table.file(b.diffuseColor);
      if (r) { md.baseColorTexture = r; md.baseColorFactor = [1, 1, 1, md.baseColorFactor[3]]; }
    }
  }
  if (b.opacity) {
    if (b.opacity.channel === 'a' && b.diffuseColor && b.opacity.file === b.diffuseColor.file && md.baseColorTexture) md.baseColorFactor[3] = 1;
    else warn(`material ${path}: opacity texture other than diffuseColor's alpha is not supported; constant used`);
  }
  if (b.roughness || b.metallic) {
    const same = b.roughness && b.metallic && b.roughness.file === b.metallic.file && b.roughness.channel === 'g' && b.metallic.channel === 'b' && b.roughness.wrap === b.metallic.wrap;
    const r = same ? table.file(b.roughness!) : table.metalRough(b.roughness, b.metallic, path);
    if (r) {
      md.metallicRoughnessTexture = r;
      if (b.roughness) md.roughnessFactor = 1;
      if (b.metallic) md.metallicFactor = 1;
    }
  }
  if (b.normal) {
    if (b.normal.channel !== 'rgb') warn(`material ${path}: normal reads '${b.normal.channel}'; ignored`);
    else {
      if (!isStdNormal(b.normal)) warn(`material ${path}: normal texture scale / bias ${JSON.stringify(b.normal.scale)} / ${JSON.stringify(b.normal.bias)} is not (2, −1): Blender's Normal Map node assumes it; mapped as 2·rgb − 1`);
      const r = table.file(b.normal);
      if (r) md.normalTexture = { ...r, scale: 1 };
    }
  }
  if (b.emissiveColor) {
    if (b.emissiveColor.channel !== 'rgb') warn(`material ${path}: emissiveColor reads '${b.emissiveColor.channel}'; constant used`);
    else {
      const r = table.file(b.emissiveColor);
      if (r) { md.emissiveTexture = r; md.emissiveFactor = [1, 1, 1]; }
    }
  }
  for (const k of Object.keys(b)) if (!['diffuseColor', 'opacity', 'roughness', 'metallic', 'normal', 'emissiveColor'].includes(k)) warn(`material ${path}: texture on ${k} is not supported; constant used`);
}
