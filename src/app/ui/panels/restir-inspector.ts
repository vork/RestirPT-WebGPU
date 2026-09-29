// ReSTIR pixel inspector (PLAN §5 M4 "pixel inspector with a 3D path overlay", §6 M4 "Inspector" row; restir-api.md
// §2.11): decodes the probe records of the ReSTIR hooks and passes (tags 64–72, render/restir/debug.ts) into
//   - the reservoir dump after initial and after spatial (all fields of §2.2),
//   - the streamed candidates (w, lum F, d/k/technique, the selected one),
//   - the slots of the last spatial round: outgoing p→partner and incoming partner→p (code, J, queued), and the MIS
//     records of the resample (m_c, Σm−1, lumRel, per-partner m_j / w_j), i.e. per-candidate m / w / J,
//   - a 3D overlay of the base tree and every shifted path (render-frame positions + the app origin; camera prepended).
// Enable it from the ReSTIR panel; the probe pixel is picked with Alt+click (or 'click = probe').
import { decodeRestirProbe, pathColour, shiftedPolylines, LOBE_NAMES, RCT_NAMES, type RestirProbeDecoded, type RsProbeReservoir } from '../../../core/render/restir/debug.ts';
import { PATH_CLASS_NAMES, RS_TECH_NAMES, lobeHistAt, pathClass } from '../../../core/render/restir/layout.ts';
import type { ProbeFrame } from '../../../core/render/probe.ts';
import type { App } from '../../app.ts';

const g = (x: number, n = 5) => (Number.isFinite(x) ? (Math.abs(x) >= 1e5 || (Math.abs(x) < 1e-3 && x !== 0) ? x.toExponential(n - 1) : x.toPrecision(n)) : String(x));
const g3 = (v: ArrayLike<number>) => `(${Array.from(v).map((x) => g(x, 4)).join(', ')})`;
const hex = (x: number) => `0x${(x >>> 0).toString(16)}`;

export class RestirInspector {
  readonly root: HTMLDivElement;
  private readonly body: HTMLPreElement;
  private readonly overlayIds: number[] = [];
  private lastUpdate = 0;
  /** Last decoded frame (tests / smoke). */
  latest: RestirProbeDecoded | undefined;
  visible = false;

  constructor(private readonly app: App, parent: HTMLElement = document.body) {
    this.root = document.createElement('div');
    this.root.className = 'restir-inspector';
    this.root.style.cssText = 'position:absolute;left:8px;bottom:8px;max-width:760px;max-height:55vh;overflow:auto;background:rgba(10,12,16,0.86);'
      + 'color:#dde;font:11px/1.35 ui-monospace,monospace;padding:6px 8px;border-radius:4px;z-index:5;';
    const head = document.createElement('div');
    head.textContent = 'ReSTIR pixel inspector (Alt+click a pixel)';
    head.style.cssText = 'font-weight:bold;margin-bottom:4px;';
    this.body = document.createElement('pre');
    this.body.style.margin = '0';
    this.root.append(head, this.body);
    this.root.hidden = true;
    parent.append(this.root);
    app.probe.listeners.add((f) => this.onFrame(f));
  }

  setVisible(v: boolean): void {
    this.visible = v;
    this.root.hidden = !v;
    if (!v) this.clearOverlay();
  }

  private clearOverlay(): void {
    for (const id of this.overlayIds) this.app.overlay.remove(id);
    this.overlayIds.length = 0;
  }

  private onFrame(f: ProbeFrame): void {
    if (!this.visible || !this.app.debugSettings.probeEnabled) return;
    const now = performance.now();
    if (now - this.lastUpdate < 200) return;
    this.lastUpdate = now;
    const d = decodeRestirProbe(f.records);
    this.latest = d;
    this.body.textContent = this.format(d, f);
    this.drawPaths(d);
  }

  /** Base path and shifted paths as overlay polylines (render frame + app origin; failed shifts dimmed). */
  private drawPaths(d: RestirProbeDecoded): void {
    this.clearOverlay();
    const o = this.app.origin;
    const cam = this.app.camera.camToWorld();
    const camR: [number, number, number] = [cam[12] - o[0], cam[13] - o[1], cam[14] - o[2]];
    for (const { path, pts, ok } of shiftedPolylines(d, camR)) {
      if (pts.length < 2) continue;
      const seg: number[] = [];
      for (let i = 0; i + 1 < pts.length; i++) for (const p of [pts[i], pts[i + 1]]) seg.push(p[0] + o[0], p[1] + o[1], p[2] + o[2]);
      const c = pathColour(path);
      this.overlayIds.push(this.app.overlay.addLines(seg, [c[0], c[1], c[2], ok ? 1 : 0.35], true));
    }
  }

  private reservoirText(r: RsProbeReservoir): string[] {
    const x = r.rec;
    const tap = r.tap === 1 ? 'after initial' : r.tap === 3 ? `after spatial (round ${r.round})` : `tap ${r.tap}`;
    const cls = pathClass(x);
    const hist: string[] = [];
    for (let b = 1; b <= 8; b++) {
      const n = lobeHistAt(x.lobeHist, b);
      if (n === 0xF) break;
      hist.push(`${LOBE_NAMES[n & 7]}${n & 8 ? 'δ' : ''}`);
    }
    return [
      `reservoir ${tap}  ai ${r.ai}${x.bg ? '  BACKGROUND' : x.d === 0 ? '  EMPTY' : ''}`,
      `  d ${x.d}  k ${x.k === 0 ? '∅' : x.k}  ${RS_TECH_NAMES[x.tech]}  endpoint type ${x.ep}${x.isDelta ? ' (delta)' : ''}  class ${cls >= 0 ? PATH_CLASS_NAMES[cls] : '-'}${x.forced ? '  forced' : ''}`
        + `  ℓk−1 ${LOBE_NAMES[x.lkm1]}${x.dkm1 ? 'δ' : ''}  ℓk ${LOBE_NAMES[x.lk]}${x.dk ? 'δ' : ''}`,
      `  W ${g(x.W)}  F ${g3(x.F)}  p̂ ${g(0.2126 * x.F[0] + 0.7152 * x.F[1] + 0.0722 * x.F[2])}  c ${g(x.c)}  wSum ${g(x.wSum)}  nCand ${x.nCand}  selId ${hex(x.selId)}`,
      `  seed ${hex(x.seed[0])}:${hex(x.seed[1])}  rc ${x.rc.map(hex).join(' ')}  jDen ${g(x.jDen)}  rcWi ${g3(x.rcWi)}  aux ${g(x.aux)}  kMargin ${g(x.kMargin)}`,
      `  rcRad ${g3(x.rcRad)}  end ${x.end.map(hex).join(' ')}  endpointId ${hex(x.endpointId)}  lobes ${hist.join(' ') || '-'}`,
      `  suffix ${x.sfx.map(hex).join(' ')} flags ${x.sfxFlags}  dir ${g3(x.sfxDir)}  t ${g(x.sfxT)}  betaS ${g3(x.betaS)}  p2 ${g(x.sfxP2)}`,
    ];
  }

  private format(d: RestirProbeDecoded, f: ProbeFrame): string {
    const px = this.app.debugSettings.probePixel;
    const W = this.app.targets.width;
    const lines = [`pixel (${px[0]}, ${px[1]})  frame ${f.frame}  ${f.records.length} records${f.counters[1] ? `  (+${f.counters[1]} dropped: probe capacity)` : ''}`];
    if (!d.reservoirs.length && !d.candidates.length && !d.slots.length) lines.push('(no ReSTIR records: ReSTIR off, or nothing wrote this pixel)');
    for (const r of d.reservoirs) lines.push(...this.reservoirText(r));
    if (d.candidates.length) {
      const wSum = d.candidates.reduce((a, c) => a + (Number.isFinite(c.w) && c.w > 0 ? c.w : 0), 0);
      lines.push(`candidates ${d.candidates.length}  Σw ${g(wSum)}`);
      for (const c of d.candidates.slice(0, 48)) {
        lines.push(`  ${c.selected ? '*' : ' '} tree ${c.tree} B ${c.B} ${c.slotInVertex === 0 ? 'NEE ' : 'BSDF'}  d ${c.d} k ${c.k === 0 ? '∅' : c.k} ${RS_TECH_NAMES[c.tech] ?? c.tech}`
          + `  w ${g(c.w)}  lumF ${g(c.lumF)}  P(sel) ${wSum > 0 ? g(c.w / wSum, 3) : '-'}`);
      }
      if (d.candidates.length > 48) lines.push(`  … ${d.candidates.length - 48} more`);
    }
    if (d.slots.length) {
      lines.push('slots (last spatial round): out = this pixel → partner, in = partner → this pixel');
      for (const s of d.slots) {
        const p = s.partnerAi === undefined ? 'no partner' : `partner (${s.partnerAi % W}, ${Math.floor(s.partnerAi / W)})`;
        const out = `${s.code.name}${s.code.term ? ` ${RCT_NAMES[s.code.term]}` : ''}${s.code.pair ? ` pair ${s.code.pair}` : ''} margin ${g(s.code.margin, 3)} ${s.j.status === 'VALID' ? `J ${g(s.j.J)} log2 ${g(Math.log2(s.j.J), 3)}` : s.j.status}${s.queued ? ' replayed' : ''}`;
        const inc = s.incoming ? `${s.incoming.code.name} ${s.incoming.j.status === 'VALID' ? `J ${g(s.incoming.j.J)}` : s.incoming.j.status} lum(G) ${g(s.incoming.lumFJ)}` : '-';
        const m = d.mis.partners.find((x) => x.s === s.s);
        lines.push(`  [${s.s}] ${p}\n      out ${out}\n      in  ${inc}${m ? `  m_j ${g(m.m)}  w_j ${g(m.w)}` : ''}`);
      }
    }
    for (const e of d.slotEvents) lines.push(`  shift event slot ${e.s}: ${e.code.name} J ${g(e.J)}${e.replayed ? ' (replay)' : ''}`);
    const c = d.mis.canonical;
    if (c) lines.push(`MIS  k ${c.k}  selected ${c.sel === 0 ? 'canonical' : `slot ${c.sel - 1}`}  m_c ${g(c.mc)}  w_c ${g(c.wc)}  Σm−1 ${g(c.sumM)}  lumRel ${g(c.lumRel)}`);
    if (d.paths.size) {
      const nm = (p: number) => (p === 0 ? 'base' : p < 8 ? `→slot${p - 1}` : p < 16 ? `slot${p - 8}→` : p < 24 ? `→slot${p - 16}(anchors)` : `slot${p - 24}→(anchors)`);
      const names = [...d.paths.keys()].sort((a, b) => a - b).map((p) => `${nm(p)}:${d.paths.get(p)!.length}`);
      lines.push(`paths (overlay: base white, this→partner warm, partner→this cool, failed dimmed; replayed prefixes not drawn): ${names.join('  ')}`);
    }
    return lines.join('\n');
  }
}
