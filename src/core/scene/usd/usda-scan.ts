// Ported from validation/spikes/usd/usda-scan.ts (docs/decisions/usd.md adapter rule 6).
// Tiny USDA text scanner for the fields lightusd rc4's next backend does not expose (treatAsPoint, spot radius,
// PreviewSurface ior/clearcoat/specular*, root-layer `doc`). Input is NextUSDZConverterNative.exportAsUSDA()
// text of the ROOT LAYER (not composed): good enough for single-layer Blender/hand-authored files only.

export interface UsdaScan {
  doc: string | null;
  /** `class` prims (abstract: never drawn directly) */
  abstract: string[];
  /** instanceable prims -> first referenced prim path (the prototype source), or '' */
  instanceable: Record<string, string>;
  /** prim path -> attribute name -> raw value text (e.g. "1", "1.45", "(0.1, 0.2, 0.3)") */
  attrs: Record<string, Record<string, string>>;
  /** The root layer composes other layers (subLayers, external references, payloads): the scan cannot see them. */
  externalLayers: boolean;
}

// M7: also the UsdUVTexture / UsdPrimvarReader shading network (info:id, inputs:file / wrapS / wrapT /
// sourceColorSpace / scale / bias / varname and every `inputs:X.connect = <…>`), so the adapter can rebuild texture
// bindings, output channels and wrap modes that LightUSD rc4 does not report (m7-api.md §3.1).
const WANTED = /^(?:uniform\s+)?(?:bool|float|double|int|token|color3f|color4f|float2|float3|float4|normal3f|half|half3|asset|string)\s+(treatAsPoint|info:id|inputs:[A-Za-z0-9_:]+(?:\.connect)?)\s*=\s*(.+?)\s*$/;
// M7: PointInstancer arrays (pxr computes instance transforms from the AUTHORED quath without normalising; LightUSD rc4
// normalises: up to 2·10⁻⁴ relative differences) and material bindings (instance prototypes: LightUSD rc4 drops them).
const WANTED_ARRAY = /^(?:uniform\s+)?(?:quath|quatf|quatd|point3f|float3|vector3f|int)\[\]\s+(orientations|positions|scales|protoIndices)\s*=\s*(\[.*\])\s*$/;
const WANTED_REL = /^rel\s+(material:binding)\s*=\s*(<[^>]+>)\s*$/;

export function scanUsda(text: string): UsdaScan {
  const out: UsdaScan = { doc: null, abstract: [], instanceable: {}, attrs: {}, externalLayers: /\bsubLayers\s*=|\bpayload\s*=|references\s*=\s*\[?\s*@/.test(text) };
  const head = /^\s*\(([\s\S]*?)^\)/m.exec(text.slice(0, 4096));
  const doc = head && /doc\s*=\s*"((?:[^"\\]|\\.)*)"/.exec(head[1]);
  out.doc = doc ? doc[1] : null;

  const stack: string[] = [];
  let pending: string | null = null; // prim name seen on a def line, waiting for its "{"
  let parenDepth = 0; // prim metadata ( ... ) blocks
  let meta = ''; // collected prim metadata text
  const pathOf = (name: string): string => '/' + [...stack.filter(Boolean), name].join('/');
  const flushMeta = (name: string): void => {
    if (/instanceable\s*=\s*(true|1)\b/.test(meta)) out.instanceable[pathOf(name)] = /references\s*=\s*\[?\s*(?:@[^@]*@)?<([^>]+)>/.exec(meta)?.[1] ?? '';
    meta = '';
  };
  for (const raw of text.split('\n')) {
    const line = raw.trim();
    if (parenDepth > 0) {
      meta += ' ' + line;
      parenDepth += (line.match(/\(/g)?.length ?? 0) - (line.match(/\)/g)?.length ?? 0);
      if (parenDepth === 0 && pending) flushMeta(pending);
      continue;
    }
    const def = /^(def|over|class)\b[^"]*"([^"]+)"\s*(\(?)(.*)$/.exec(line);
    if (def) {
      pending = def[2];
      if (def[1] === 'class') out.abstract.push(pathOf(pending));
      meta = def[4];
      if (def[3] === '(' && !line.includes(')')) parenDepth = 1;
      else flushMeta(pending);
      if (line.endsWith('{')) { stack.push(pending); pending = null; }
      continue;
    }
    if (line === '{') {
      stack.push(pending ?? '');
      pending = null;
      continue;
    }
    if (line === '}') {
      stack.pop();
      continue;
    }
    const m = WANTED.exec(line) ?? WANTED_ARRAY.exec(line) ?? WANTED_REL.exec(line);
    if (m && stack.length) {
      const path = '/' + stack.filter(Boolean).join('/');
      (out.attrs[path] ??= {})[m[1]] = m[2];
    }
  }
  return out;
}

export const parseUsdaValue = (s: string | undefined): number | boolean | number[] | null => {
  if (s === undefined) return null;
  if (s === 'true') return true;
  if (s === 'false') return false;
  if (s.startsWith('(')) return s.replace(/[()]/g, '').split(',').map(Number);
  const n = Number(s);
  return Number.isFinite(n) ? n : null;
};

/** Tuples of a usda array value: "[(1, 0, 0, 0), (…)]" → number[][]; "[0, 1, 2]" → number[][] of length-1 tuples. */
export const parseUsdaTuples = (s: string | undefined): number[][] | null => {
  if (!s || !s.startsWith('[')) return null;
  const body = s.slice(1, -1).trim();
  if (!body) return [];
  if (body.startsWith('(')) return [...body.matchAll(/\(([^)]*)\)/g)].map((m) => m[1].split(',').map(Number));
  return body.split(',').map((x) => [Number(x)]);
};

/** Nearest IEEE half of x (round to nearest even), as a number: the value a quath stores. */
export function toHalf(x: number): number {
  if (!Number.isFinite(x) || x === 0) return x;
  const a = Math.abs(x);
  if (a >= 65520) return Math.sign(x) * Infinity;
  const e = Math.max(Math.floor(Math.log2(a)), -14);
  const ulp = 2 ** (e - 10);
  let q = a / ulp;
  const f = Math.floor(q), r = q - f;
  q = r > 0.5 || (r === 0.5 && f % 2 === 1) ? f + 1 : f;
  return Math.sign(x) * q * ulp;
}
