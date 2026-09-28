import { describe, expect, it } from 'vitest';
import { DEG, matrixToPose, poseMatrix, quatForward, quatFromYawPitch, type Quat } from '../../src/app/camera-math.ts';
import {
  CameraTrack, FlyCamera, KEY_ACTIONS, MAX_DT, PITCH_LIMIT, clampPitch, flySpeed, keyToAction, moveDirection, type MoveAction,
} from '../../src/app/fly-camera.ts';

const close = (a: number[], b: number[], eps = 1e-9) => a.forEach((x, i) => expect(Math.abs(x - b[i])).toBeLessThan(eps));

describe('fly camera: key mapping', () => {
  it('maps physical key codes (layout independent), E up / Q down', () => {
    expect(keyToAction('KeyW')).toBe('forward');
    expect(keyToAction('KeyS')).toBe('back');
    expect(keyToAction('KeyA')).toBe('left');
    expect(keyToAction('KeyD')).toBe('right');
    expect(keyToAction('KeyE')).toBe('up');
    expect(keyToAction('KeyQ')).toBe('down');
    // character-based names and other keys are not mapped
    expect(keyToAction('w')).toBeUndefined();
    expect(keyToAction('KeyZ')).toBeUndefined();
    expect(Object.keys(KEY_ACTIONS).sort()).toEqual(['KeyA', 'KeyD', 'KeyE', 'KeyQ', 'KeyS', 'KeyW']);
  });

  it('WASD moves in the world XZ plane by yaw only; E/Q along world ±Y', () => {
    const d = (acts: MoveAction[], yaw: number) => moveDirection(new Set(acts), yaw);
    close(d(['forward'], 0), [0, 0, -1]);
    close(d(['right'], 0), [1, 0, 0]);
    close(d(['forward'], 90 * DEG), [-1, 0, 0]); // yaw +90° turns left: forward = -X
    close(d(['up'], 1.234), [0, 1, 0]);
    close(d(['down'], 1.234), [0, -1, 0]);
    close(d(['forward', 'back'], 0.3), [0, 0, 0]);
    // diagonal is normalised (no faster diagonal flight)
    const diag = d(['forward', 'right', 'up'], 0.7);
    expect(Math.hypot(...diag)).toBeCloseTo(1, 12);
    // pitch never tilts WASD motion
    const cam = new FlyCamera({ sceneDiag: 10 });
    cam.lookAt([0, 0, 0], [0, -5, -1]);
    cam.held.add('forward');
    cam.update(0.05);
    expect(cam.position[1]).toBe(0);
  });
});

describe('fly camera: speed', () => {
  it('scales with the scene diagonal (diag/10 per s) and modifiers', () => {
    expect(flySpeed(10, 0, false, false)).toBeCloseTo(1, 12);
    expect(flySpeed(250, 0, false, false)).toBeCloseTo(25, 12);
    expect(flySpeed(10, 0, true, false)).toBeCloseTo(4, 12);
    expect(flySpeed(10, 0, false, true)).toBeCloseTo(0.25, 12);
    expect(flySpeed(10, 4, false, false)).toBeCloseTo(1.25 ** 4, 12);
  });
  it('wheel adjusts in log steps (down = slower) with a clamp', () => {
    const cam = new FlyCamera({ sceneDiag: 10 });
    cam.wheel(120);
    expect(cam.speed).toBeCloseTo(1 / 1.25, 12);
    cam.wheel(-3); cam.wheel(-1);
    expect(cam.speed).toBeCloseTo(1.25, 12);
    for (let i = 0; i < 200; i++) cam.wheel(-1);
    expect(cam.wheelSteps).toBe(40);
  });
  it('is frame-rate independent and clamps dt to 0.1 s', () => {
    const a = new FlyCamera({ sceneDiag: 10 });
    const b = new FlyCamera({ sceneDiag: 10 });
    a.held.add('forward'); b.held.add('forward');
    for (let i = 0; i < 60; i++) a.update(1 / 60);
    for (let i = 0; i < 20; i++) b.update(1 / 20);
    close(a.position, b.position, 1e-9);
    const c = new FlyCamera({ sceneDiag: 10 });
    const z0 = c.position[2];
    c.held.add('forward');
    c.update(5); // stall: clamped
    expect(z0 - c.position[2]).toBeCloseTo(MAX_DT * 1, 12);
  });
});

describe('fly camera: look', () => {
  it('clamps pitch to ±89°', () => {
    expect(clampPitch(2)).toBe(PITCH_LIMIT);
    expect(clampPitch(-2)).toBe(-PITCH_LIMIT);
    const cam = new FlyCamera();
    cam.look(0, -1e6); // mouse up
    expect(cam.pitch).toBeCloseTo(89 * DEG, 12);
    cam.look(0, 1e6);
    expect(cam.pitch).toBeCloseTo(-89 * DEG, 12);
  });
  it('mouse right turns right, mouse down looks down, invert-Y flips', () => {
    const cam = new FlyCamera({ sensitivity: 0.01 });
    cam.lookAt([0, 0, 0], [0, 0, -1]);
    cam.look(10, 0);
    expect(quatForward(cam.quaternion)[0]).toBeGreaterThan(0);
    cam.look(0, 10);
    expect(quatForward(cam.quaternion)[1]).toBeLessThan(0);
    const inv = new FlyCamera({ sensitivity: 0.01, invertY: true });
    inv.look(0, 10);
    expect(inv.pitch).toBeGreaterThan(0);
  });
  it('frozen (paused) camera ignores look', () => {
    const cam = new FlyCamera();
    cam.frozen = true;
    const q = [...cam.quaternion];
    cam.look(100, 100);
    expect(cam.quaternion).toEqual(q);
  });
});

describe('fly camera: poses, file camera, bookmarks', () => {
  it('matrix <-> pose round trip keeps roll (file cameras are exact)', () => {
    const q = quatFromYawPitch(0.7, -0.3);
    const roll: Quat = [0, 0, Math.sin(0.2), Math.cos(0.2)];
    const qr: Quat = [
      q[3] * roll[0] + q[0] * roll[3] + q[1] * roll[2] - q[2] * roll[1],
      q[3] * roll[1] - q[0] * roll[2] + q[1] * roll[3] + q[2] * roll[0],
      q[3] * roll[2] + q[0] * roll[1] - q[1] * roll[0] + q[2] * roll[3],
      q[3] * roll[3] - q[0] * roll[0] - q[1] * roll[1] - q[2] * roll[2],
    ];
    const m = poseMatrix([1, 2, 3], qr);
    const cam = new FlyCamera();
    cam.setFromMatrix(m, 0.6);
    close([...cam.camToWorld()], [...m], 1e-12);
    const back = matrixToPose(m);
    close(back.position, [1, 2, 3]);
  });
  it('Home resets and bookmarks save/recall', () => {
    const cam = new FlyCamera();
    const p0 = cam.pose();
    cam.held.add('right');
    cam.update(0.1);
    cam.saveBookmark(3);
    const p3 = cam.pose();
    cam.update(0.1);
    cam.reset();
    close(cam.position, p0.position);
    expect(cam.recallBookmark(3)).toBe(true);
    close(cam.position, p3.position);
    expect(cam.recallBookmark(5)).toBe(false);
  });
});

describe('camera track', () => {
  const qa = quatFromYawPitch(0, 0);
  const qb = quatFromYawPitch(90 * DEG, 0);
  const track = new CameraTrack([
    { t: 0, position: [0, 0, 0], quaternion: qa, yfov: 0.5 },
    { t: 2, position: [2, 4, -6], quaternion: qb, yfov: 1.0 },
    { t: 3, position: [3, 4, -6], quaternion: qb, yfov: 1.0 },
  ]);

  it('interpolates position/yfov linearly and orientation by slerp', () => {
    const p = track.sample(1)!;
    close(p.position, [1, 2, -3]);
    expect(p.yfov).toBeCloseTo(0.75, 12);
    const f = quatForward(p.quaternion); // halfway between -Z and -X: yaw 45°
    close(f, [-Math.SQRT1_2, 0, -Math.SQRT1_2], 1e-12);
    close(track.sample(2.5)!.position, [2.5, 4, -6]);
  });
  it('clamps outside the range, wraps when looping', () => {
    close(track.sample(-1)!.position, [0, 0, 0]);
    close(track.sample(10)!.position, [3, 4, -6]);
    close(track.sample(4, true)!.position, [1, 2, -3]);
  });
  it('JSON round trip and validation', () => {
    const j = JSON.parse(JSON.stringify(track.toJSON()));
    const t2 = CameraTrack.fromJSON(j);
    close(t2.sample(1)!.position, [1, 2, -3]);
    expect(() => CameraTrack.fromJSON({ keys: [{ t: 0 }] })).toThrow();
    expect(() => new CameraTrack().push({ t: -1, position: [0, 0, 0], quaternion: qa, yfov: 1 })).not.toThrow();
  });
  it('record then playback reproduces the flown path', () => {
    const cam = new FlyCamera({ sceneDiag: 10 });
    cam.startRecording();
    cam.held.add('forward');
    for (let i = 0; i < 10; i++) cam.update(0.05);
    cam.held.clear();
    const end = cam.pose();
    const tr = cam.stopRecording()!;
    expect(tr.keys.length).toBe(11);
    expect(tr.duration).toBeCloseTo(0.5, 12);
    cam.reset();
    cam.play(tr);
    for (let i = 0; i < 10; i++) cam.update(0.05);
    close(cam.position, end.position, 1e-9);
    expect(cam.playing).toBeUndefined();
  });
});
