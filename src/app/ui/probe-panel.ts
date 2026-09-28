// Probe panel (DOM): records appended by shaders at the probe pixel (read through the 3-deep MAP_READ ring).
import { probeTagName, type ProbeFrame } from '../../core/render/probe.ts';

export class ProbePanel {
  readonly root: HTMLDivElement;
  private readonly body: HTMLPreElement;
  private readonly head: HTMLDivElement;

  constructor(parent: HTMLElement) {
    this.root = document.createElement('div');
    this.root.className = 'probe-panel';
    this.head = document.createElement('div');
    this.head.className = 'probe-head';
    this.body = document.createElement('pre');
    this.root.append(this.head, this.body);
    this.root.hidden = true;
    parent.append(this.root);
  }

  setVisible(v: boolean): void { this.root.hidden = !v; }

  update(pixel: [number, number], f: ProbeFrame | undefined, aovIsCode: boolean): void {
    if (this.root.hidden) return;
    this.head.textContent = `probe (${pixel[0]}, ${pixel[1]})  ${f ? `frame ${f.frame}, ${f.records.length} rec` : 'waiting...'}`;
    if (!f) { this.body.textContent = ''; return; }
    const g = (x: number) => (Number.isFinite(x) ? x.toPrecision(6) : String(x));
    const lines = f.records.map((r) => {
      const tag = probeTagName(r.tag).padEnd(10);
      if (r.tag === 1 && aovIsCode) return `${tag} code ${r.bits[0]} (0x${r.bits[0].toString(16)})`;
      return `${tag} ${r.value.map(g).join('  ')}`;
    });
    const overflow = f.counters[1];
    if (overflow) lines.push(`(+${overflow} records dropped: capacity)`);
    this.body.textContent = lines.join('\n') || '(no records: nothing wrote to this pixel)';
  }
}
