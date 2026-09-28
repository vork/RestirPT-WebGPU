// Tiny WGSL composer: #include / #if / $DEFINE substitution / requires-enable hoisting / line mapping.
// Identical behaviour in the browser and in Node (plan §1.1). No build-time transforms.
import { ENABLE_TO_FEATURE, SHIPPED_WGSL_DIRECTIVES } from './profile.ts';

export type DefineValue = number | boolean | string;
export type Defines = Record<string, DefineValue>;

export interface ComposeOptions {
  /** Map of shader path (relative to the shaders root, e.g. 'common/rng.wgsl') to source text. */
  sources: Record<string, string>;
  defines?: Defines;
  /** Device features (for `enable` checks and HAS_* defines). */
  features?: Set<string>;
  /** navigator.gpu.wgslLanguageFeatures (for `requires` checks and HAS_* defines). */
  wgslLanguageFeatures?: Set<string>;
}

export interface LineOrigin { file: string; line: number }

export interface ComposedShader {
  code: string;
  /** lineMap[i] = origin of composed line i+1 (1-based lines like GPUCompilationMessage.lineNum). */
  lineMap: LineOrigin[];
  directives: string[];
  defines: Defines;
}

export class WgslComposeError extends Error {}

/** Feature-derived defines available to every shader. */
export function featureDefines(features: Set<string> = new Set(), lf: Set<string> = new Set()): Defines {
  return {
    HAS_SUBGROUPS: features.has('subgroups'),
    HAS_F16: features.has('shader-f16'),
    HAS_TIMESTAMPS: features.has('timestamp-query'),
    HAS_FLOAT32_FILTERABLE: features.has('float32-filterable'),
    HAS_TIER2_STORAGE: features.has('texture-formats-tier2'),
    HAS_IMMEDIATES: lf.has('immediate_address_space'),
    HAS_LINEAR_INDEXING: lf.has('linear_indexing'),
  };
}

export function composeWgsl(entry: string, opts: ComposeOptions): ComposedShader {
  const features = opts.features ?? new Set<string>();
  const lf = opts.wgslLanguageFeatures ?? new Set<string>();
  const defines: Defines = { ...featureDefines(features, lf), ...(opts.defines ?? {}) };
  const out: string[] = [];
  const lineMap: LineOrigin[] = [];
  const directives = new Set<string>();
  const included = new Set<string>();

  const emit = (text: string, origin: LineOrigin) => { out.push(text); lineMap.push(origin); };

  const visit = (path: string, stack: string[]) => {
    const norm = normalizePath(path);
    if (included.has(norm)) return; // include-once semantics
    if (stack.includes(norm)) throw new WgslComposeError(`include cycle: ${[...stack, norm].join(' -> ')}`);
    const src = opts.sources[norm];
    if (src === undefined) throw new WgslComposeError(`unknown shader include '${norm}' (from ${stack.at(-1) ?? 'entry'})`);
    included.add(norm);
    // Conditional stack: each frame = { active: currently emitting, taken: some branch already taken, parent: parent active }
    const cond: { active: boolean; taken: boolean; parent: boolean }[] = [];
    const isActive = () => (cond.length === 0 ? true : cond[cond.length - 1].active);
    const lines = src.split(/\r?\n/);
    lines.forEach((raw, idx) => {
      const lineNo = idx + 1;
      const origin = { file: norm, line: lineNo };
      const t = raw.trim();
      if (t.startsWith('#')) {
        const m = /^#(\w+)\s*(.*)$/.exec(t);
        if (!m) throw new WgslComposeError(`${norm}:${lineNo}: malformed directive`);
        const [, kw, rest] = m;
        switch (kw) {
          case 'include': {
            if (!isActive()) return;
            const pm = /^"([^"]+)"$/.exec(rest.trim());
            if (!pm) throw new WgslComposeError(`${norm}:${lineNo}: #include expects "path"`);
            visit(resolveRelative(norm, pm[1]), [...stack, norm]);
            return;
          }
          case 'ifdef': case 'ifndef': case 'if': {
            const parent = isActive();
            let v: boolean;
            if (kw === 'if') v = truthy(evalExpr(rest, defines, `${norm}:${lineNo}`));
            else { const def = rest.trim() in defines && truthy(defines[rest.trim()]); v = kw === 'ifdef' ? def : !def; }
            cond.push({ active: parent && v, taken: v, parent });
            return;
          }
          case 'elif': {
            const top = cond[cond.length - 1];
            if (!top) throw new WgslComposeError(`${norm}:${lineNo}: #elif without #if`);
            const v = !top.taken && truthy(evalExpr(rest, defines, `${norm}:${lineNo}`));
            top.active = top.parent && v; top.taken = top.taken || v;
            return;
          }
          case 'else': {
            const top = cond[cond.length - 1];
            if (!top) throw new WgslComposeError(`${norm}:${lineNo}: #else without #if`);
            top.active = top.parent && !top.taken; top.taken = true;
            return;
          }
          case 'endif': {
            if (!cond.pop()) throw new WgslComposeError(`${norm}:${lineNo}: #endif without #if`);
            return;
          }
          case 'error': {
            if (isActive()) throw new WgslComposeError(`${norm}:${lineNo}: #error ${rest}`);
            return;
          }
          default:
            throw new WgslComposeError(`${norm}:${lineNo}: unknown directive #${kw}`);
        }
      }
      if (!isActive()) return;
      // Hoist `requires x, y;` and `enable x;` to the top of the module, validated against the shipped whitelist.
      const dm = /^(requires|enable)\s+([\w\s,]+);\s*$/.exec(t);
      if (dm) {
        for (const name of dm[2].split(',').map((s) => s.trim()).filter(Boolean)) {
          if (!SHIPPED_WGSL_DIRECTIVES.has(name)) {
            throw new WgslComposeError(`${norm}:${lineNo}: '${name}' is not a shipped WGSL feature (whitelist)`);
          }
          if (dm[1] === 'enable') {
            const f = ENABLE_TO_FEATURE[name];
            if (f && !features.has(f)) throw new WgslComposeError(`${norm}:${lineNo}: enable ${name} needs feature ${f}`);
          } else if (!lf.has(name)) {
            throw new WgslComposeError(`${norm}:${lineNo}: requires ${name} not supported by this implementation`);
          }
          directives.add(`${dm[1]} ${name};`);
        }
        return;
      }
      emit(substitute(raw, defines, `${norm}:${lineNo}`), origin);
    });
    if (cond.length) throw new WgslComposeError(`${norm}: unterminated #if`);
  };

  visit(entry, []);
  const header = [...directives].sort((a, b) => (a.startsWith('enable') === b.startsWith('enable') ? a.localeCompare(b) : a.startsWith('enable') ? -1 : 1));
  const headerMap = header.map(() => ({ file: '<directives>', line: 0 }));
  return { code: [...header, ...out].join('\n'), lineMap: [...headerMap, ...lineMap], directives: header, defines };
}

/** Translate GPUCompilationMessages to original file:line. */
export function mapCompilationMessages(messages: readonly GPUCompilationMessage[], lineMap: LineOrigin[]): string[] {
  return messages.map((m) => {
    const o = lineMap[m.lineNum - 1];
    const where = o ? `${o.file}:${o.line}:${m.linePos}` : `<composed>:${m.lineNum}:${m.linePos}`;
    return `${m.type}: ${where}: ${m.message}`;
  });
}

/** Create a shader module and throw with mapped messages if compilation reports errors. */
export async function createCheckedShaderModule(device: GPUDevice, shader: ComposedShader, label: string): Promise<GPUShaderModule> {
  const module = device.createShaderModule({ code: shader.code, label });
  const info = await module.getCompilationInfo();
  const errors = info.messages.filter((m) => m.type === 'error');
  if (errors.length) throw new WgslComposeError(`WGSL compile failed (${label}):\n${mapCompilationMessages(errors, shader.lineMap).join('\n')}`);
  return module;
}

// ---------------------------------------------------------------------------------------------------------------

function normalizePath(p: string): string {
  const parts: string[] = [];
  for (const seg of p.split('/')) {
    if (seg === '' || seg === '.') continue;
    if (seg === '..') parts.pop(); else parts.push(seg);
  }
  return parts.join('/');
}

/** Includes are resolved relative to the shaders root unless they start with './' or '../'. */
function resolveRelative(from: string, target: string): string {
  if (!target.startsWith('.')) return normalizePath(target);
  const dir = from.split('/').slice(0, -1).join('/');
  return normalizePath(`${dir}/${target}`);
}

function truthy(v: DefineValue | undefined): boolean {
  return v !== undefined && v !== false && v !== 0 && v !== '' && v !== '0' && v !== 'false';
}

/** `$NAME` -> define value. Unknown names are an error (catches typos in variant switches). */
function substitute(line: string, defines: Defines, where: string): string {
  if (!line.includes('$')) return line;
  return line.replace(/\$([A-Z_][A-Z0-9_]*)/g, (_, name: string) => {
    const v = defines[name];
    if (v === undefined) throw new WgslComposeError(`${where}: undefined define $${name}`);
    if (typeof v === 'boolean') return v ? 'true' : 'false';
    return String(v);
  });
}

// Minimal expression evaluator for #if: identifiers, numbers, true/false, ! && || == != < > <= >= ( ).
function evalExpr(src: string, defines: Defines, where: string): DefineValue {
  const toks = src.match(/\s*(\d+(?:\.\d+)?|[A-Za-z_]\w*|&&|\|\||==|!=|<=|>=|[()!<>])/g)?.map((s) => s.trim()) ?? [];
  if (toks.join('') !== src.replace(/\s+/g, '')) throw new WgslComposeError(`${where}: bad #if expression '${src}'`);
  let i = 0;
  const peek = () => toks[i];
  const next = () => toks[i++];
  const num = (v: DefineValue): number => (typeof v === 'number' ? v : truthy(v) ? 1 : 0);
  const primary = (): DefineValue => {
    const t = next();
    if (t === undefined) throw new WgslComposeError(`${where}: unexpected end of #if expression`);
    if (t === '(') { const v = or(); if (next() !== ')') throw new WgslComposeError(`${where}: missing )`); return v; }
    if (t === '!') return !truthy(primary());
    if (/^\d/.test(t)) return Number(t);
    if (t === 'true') return true;
    if (t === 'false') return false;
    return defines[t] ?? false; // undefined identifiers are false (like C preprocessor)
  };
  const cmp = (): DefineValue => {
    let l = primary();
    while (['==', '!=', '<', '>', '<=', '>='].includes(peek() ?? '')) {
      const op = next(); const r = primary(); const a = num(l); const b = num(r);
      l = op === '==' ? a === b : op === '!=' ? a !== b : op === '<' ? a < b : op === '>' ? a > b : op === '<=' ? a <= b : a >= b;
    }
    return l;
  };
  const and = (): DefineValue => { let l = cmp(); while (peek() === '&&') { next(); const r = cmp(); l = truthy(l) && truthy(r); } return l; };
  const or = (): DefineValue => { let l = and(); while (peek() === '||') { next(); const r = and(); l = truthy(l) || truthy(r); } return l; };
  const v = or();
  if (i !== toks.length) throw new WgslComposeError(`${where}: trailing tokens in #if '${src}'`);
  return v;
}
