// 'Animation' panel (plan §5 M3a "Animation"): play/pause, interactive (wall-clock) vs validation (frame/fps) time,
// fps/duration/loop, keyframes at the current time for the selected light / camera / env (linear or step), presets
// (orbit, bob, sweep), frames[] export and the editor scene.json save/load.
import { exportFrames, lightTarget, matrixToPoseQ, presetBob, presetOrbit, presetSweep, type TargetId } from '../../../core/scene/animation.ts';
import { DEG } from '../../camera-math.ts';
import type { LightEditor } from '../../editor/light-editor.ts';
import type { TpFolder } from '../tweakpane.ts';

export interface AnimationPanelHooks {
  save(): void;
  load(): void;
}

/** Fills the (pre-created, hidden) Animation folder of the panel and shows it. */
export function addAnimationPanel(f: TpFolder, ed: LightEditor, hooks: AnimationPanelHooks): { folder: TpFolder; refresh(): void } {
  f.hidden = false;
  const a = ed.anim, pl = ed.player;
  let quiet = false;
  const q = <T>(fn: (e: T) => void) => (e: T) => { if (!quiet) fn(e); };
  const ui = {
    time: '', mode: pl.mode as string, fps: a.fps, duration: a.duration, loop: a.loop, interp: ed.keyInterp as string, autoKey: ed.autoKey,
    target: 'selected' as string,
    orbitPeriod: 8, orbitTurns: 1, orbitAim: true, bobAmp: 0.2, bobPeriod: 2, sweepDeg: 30, sweepPeriod: 4, info: '',
  };
  const playBtn = f.addButton({ title: 'Play (Space)' }).on('click', () => { pl.toggle(); refresh(); });
  f.addButton({ title: 'Go to start' }).on('click', () => { if (pl.mode === 'validation') pl.seekFrame(0); else pl.seek(0); });
  f.addBinding(ui, 'time', { readonly: true, label: 'time' });
  f.addBinding(ui, 'mode', { label: 'time mode', options: { 'interactive (wall clock)': 'interactive', 'validation (frame/fps)': 'validation' } })
    .on('change', q((e) => pl.setMode(e.value as 'interactive' | 'validation')));
  f.addBinding(ui, 'fps', { label: 'fps', min: 1, max: 240, step: 1 }).on('change', q((e) => a.setSettings({ fps: e.value })));
  f.addBinding(ui, 'duration', { label: 'duration (s)', min: 0.1, max: 600, step: 0.1 }).on('change', q((e) => a.setSettings({ duration: e.value })));
  f.addBinding(ui, 'loop', { label: 'loop' }).on('change', q((e) => a.setSettings({ loop: e.value })));

  const k = f.addFolder({ title: 'Keys', expanded: true });
  k.addBinding(ui, 'interp', { label: 'new key interp.', options: { linear: 'linear', 'step (constant)': 'step' } }).on('change', q((e) => { ed.keyInterp = e.value as 'linear' | 'step'; }));
  k.addBinding(ui, 'autoKey', { label: 'auto key' }).on('change', q((e) => { ed.autoKey = e.value; }));
  k.addButton({ title: 'Key selected light' }).on('click', () => { if (ed.selected !== undefined) ed.keyTarget(lightTarget(ed.selected)); });
  k.addButton({ title: 'Key camera' }).on('click', () => ed.keyTarget('camera'));
  k.addButton({ title: 'Key environment (γ, strength)' }).on('click', () => { if (ed.app.env) ed.keyTarget('env'); });
  k.addButton({ title: 'Remove light key at t' }).on('click', () => { if (ed.selected !== undefined) ed.removeKeyAt(lightTarget(ed.selected)); });
  k.addButton({ title: 'Remove camera key at t' }).on('click', () => ed.removeKeyAt('camera'));

  const pr = f.addFolder({ title: 'Presets', expanded: false });
  pr.addBinding(ui, 'target', { label: 'target', options: { 'selected light': 'selected', camera: 'camera' } });
  pr.addBinding(ui, 'orbitPeriod', { label: 'orbit period (s)', min: 0.1, max: 120, step: 0.1 });
  pr.addBinding(ui, 'orbitTurns', { label: 'orbit turns', min: 0.25, max: 10, step: 0.25 });
  pr.addBinding(ui, 'orbitAim', { label: 'orbit: aim at centre' });
  pr.addButton({ title: 'Orbit around scene centre' }).on('click', () => {
    const tg = target();
    if (!tg) return;
    const st = startPose(tg);
    const b = ed.scene?.bounds;
    const center: [number, number, number] = b ? [(b.min[0] + b.max[0]) / 2, (b.min[1] + b.max[1]) / 2, (b.min[2] + b.max[2]) / 2] : [0, 0, 0];
    ed.applyPreset(tg, presetOrbit({ start: st.position, center, period: ui.orbitPeriod, turns: ui.orbitTurns, t0: pl.time, aim: ui.orbitAim }), 'orbit preset');
  });
  pr.addBinding(ui, 'bobAmp', { label: 'bob amplitude (m)', min: 0, max: 100, step: 0.01 });
  pr.addBinding(ui, 'bobPeriod', { label: 'bob period (s)', min: 0.1, max: 60, step: 0.1 });
  pr.addButton({ title: 'Bob up and down' }).on('click', () => {
    const tg = target();
    if (!tg) return;
    ed.applyPreset(tg, presetBob({ start: startPose(tg).position, amplitude: ui.bobAmp, period: ui.bobPeriod, cycles: Math.max(1, Math.floor((a.duration - pl.time) / ui.bobPeriod)), t0: pl.time }), 'bob preset');
  });
  pr.addBinding(ui, 'sweepDeg', { label: 'sweep ± angle (°)', min: 1, max: 170, step: 1 });
  pr.addBinding(ui, 'sweepPeriod', { label: 'sweep period (s)', min: 0.1, max: 60, step: 0.1 });
  pr.addButton({ title: 'Sweep (spot rotation about +Y)' }).on('click', () => {
    const tg = target();
    if (!tg) return;
    ed.applyPreset(tg, presetSweep({ start: startPose(tg).quaternion, angle: ui.sweepDeg * DEG, period: ui.sweepPeriod, cycles: Math.max(1, Math.floor((a.duration - pl.time) / ui.sweepPeriod)), t0: pl.time }), 'sweep preset');
  });

  const io = f.addFolder({ title: 'Save / load', expanded: false });
  io.addButton({ title: 'Save scene.json (editor state)' }).on('click', () => hooks.save());
  io.addButton({ title: 'Load scene.json...' }).on('click', () => hooks.load());
  io.addButton({ title: 'Download frames[] JSON' }).on('click', () => {
    try {
      const frames = exportFrames(a, { base: ed.baseState() });
      download(`${(ed.scene?.name ?? 'scene').replace(/[^\w.-]+/g, '_')}.frames.json`, JSON.stringify({ fps: a.fps, frames }, null, 1));
    } catch (e) { ed.message = e instanceof Error ? e.message : String(e); refresh(); }
  });
  f.addBinding(ui, 'info', { readonly: true, multiline: true, rows: 3, label: 'tracks' });

  function target(): TargetId | undefined {
    if (ui.target === 'camera') return 'camera';
    if (ed.selected === undefined) { ed.message = 'select a light first'; return undefined; }
    return lightTarget(ed.selected);
  }
  function startPose(tg: TargetId): { position: [number, number, number]; quaternion: [number, number, number, number] } {
    if (tg === 'camera') return { position: [...ed.app.camera.position], quaternion: [...ed.app.camera.quaternion] };
    return matrixToPoseQ(ed.selectedLight!.matrix);
  }

  const refresh = () => {
    playBtn.title = pl.playing ? 'Pause (Space)' : 'Play (Space)';
    const frame = pl.mode === 'validation' ? pl.frame : Math.round(pl.time * a.fps);
    ui.time = `${pl.time.toFixed(3)} s  frame ${frame}`;
    ui.mode = pl.mode; ui.fps = a.fps; ui.duration = a.duration; ui.loop = a.loop;
    ui.info = a.targets.map((t) => `${t}: ${Object.entries(a.track(t)!.channels).map(([c, ks]) => `${c}×${ks!.length}`).join(' ')}`).join('\n') || '(no keys)';
    quiet = true;
    try { f.refresh(); } finally { quiet = false; }
  };
  let last = 0;
  ed.listeners.add((e) => {
    if (e !== 'time' && e !== 'undo' && e !== 'scene' && e !== 'selection') return;
    const now = performance.now();
    if (e === 'time' && pl.playing && now - last < 200) return;
    last = now;
    refresh();
  });
  refresh();
  return { folder: f, refresh };
}

export function download(name: string, text: string): void {
  const a = document.createElement('a');
  a.href = URL.createObjectURL(new Blob([text], { type: 'application/json' }));
  a.download = name;
  a.click();
  setTimeout(() => URL.revokeObjectURL(a.href), 1000);
}
