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
      const total = d.passes.find((p) => p.name === 'total');
      for (const p of d.passes) if (p.name !== 'total') lines.push(`  ${p.name.padEnd(18)} ${f(p.ms, 3).padStart(8)} ms`);
      if (total) lines.push(`  ${'gpu total'.padEnd(18)} ${f(total.ms, 3).padStart(8)} ms`);
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
