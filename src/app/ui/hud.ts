// HUD overlay (DOM): fps, per-pass GPU ms (32-frame averages), NaN/Inf and overflow counters, camera state.
export interface HudData {
  fps: number;
  cpuMs: number;
  passes: { name: string; ms: number }[];
  timestampsSupported: boolean;
  counters?: Uint32Array;
  totals: { nan: number; inf: number; bvhOverflow: number; bvhItercap: number; queueOverflow: number; probeOverflow: number };
  internal: [number, number];
  canvas: [number, number];
  dpr: number;
  frameIndex: number;
  paused: boolean;
  camera: { position: number[]; yawDeg: number; pitchDeg: number; speed: number; recording: boolean; playing: boolean };
  scene?: string;
  extra?: string[];
}

export class Hud {
  readonly root: HTMLPreElement;
  visible = true;

  constructor(parent: HTMLElement) {
    this.root = document.createElement('pre');
    this.root.className = 'hud';
    parent.append(this.root);
  }

  setVisible(v: boolean): void { this.visible = v; this.root.hidden = !v; }

  update(d: HudData): void {
    if (!this.visible) return;
    const f = (x: number, n = 2) => (Number.isFinite(x) ? x.toFixed(n) : '-');
    const lines: string[] = [];
    lines.push(`${f(d.fps, 1)} fps  cpu ${f(d.cpuMs)} ms  frame ${d.frameIndex}${d.paused ? '  [PAUSED]' : ''}`);
    lines.push(`internal ${d.internal[0]}x${d.internal[1]}  canvas ${d.canvas[0]}x${d.canvas[1]} @${f(d.dpr, 2)}x`);
    if (d.scene) lines.push(`scene ${d.scene}`);
    if (!d.timestampsSupported) lines.push('gpu timing: timestamp-query unavailable');
    else {
      // timestamps.ts attributeFrame: each timed pass is its own cost; "untimed" is the rest of the frame's GPU span (the
      // ReSTIR passes and the in-frame denoiser carry no timestamps, Q3); gpu total = the span (first timed begin to the
      // present's end), so the lines add up to it.
      const total = d.passes.find((p) => p.name === 'total');
      const untimed = d.passes.find((p) => p.name === 'untimed');
      const row = (name: string, ms: number) => lines.push(`  ${name.padEnd(18)} ${f(ms, 3).padStart(8)} ms`);
      for (const p of d.passes) if (p.name !== 'total' && p.name !== 'untimed') row(p.name, p.ms);
      if (untimed) row('untimed (Q3)', untimed.ms);
      if (total) row('gpu total', total.ms);
      if (untimed && untimed.ms > 0.05) lines.push('  (untimed = ReSTIR + in-frame denoiser: no timestamps there, Q3)');
    }
    const c = d.counters;
    const t = d.totals;
    const flag = (n: number) => (n > 0 ? ' !' : '');
    lines.push(`NaN ${c?.[2] ?? 0}/${t.nan}${flag(t.nan)}  Inf ${c?.[3] ?? 0}/${t.inf}${flag(t.inf)}  (frame/total)`);
    lines.push(`BVH overflow ${t.bvhOverflow}${flag(t.bvhOverflow)}  itercap ${t.bvhItercap}${flag(t.bvhItercap)}  queue ${t.queueOverflow}${flag(t.queueOverflow)}  probe ${t.probeOverflow}`);
    const cam = d.camera;
    lines.push(`cam [${cam.position.map((x) => f(x, 2)).join(', ')}] yaw ${f(cam.yawDeg, 1)} pitch ${f(cam.pitchDeg, 1)} speed ${f(cam.speed, 2)} m/s${cam.recording ? ' REC' : ''}${cam.playing ? ' PLAY' : ''}`);
    if (d.extra) lines.push(...d.extra);
    this.root.textContent = lines.join('\n');
    this.root.classList.toggle('alarm', t.nan + t.inf + t.bvhOverflow + t.bvhItercap + t.queueOverflow > 0);
  }
}
