// Small f64 camera math (glTF frame: right-handed, +Y up, camera looks down local -Z).
export type Vec3 = [number, number, number];
export type Quat = [number, number, number, number]; // x, y, z, w

export const DEG = Math.PI / 180;

export function quatMul(a: Quat, b: Quat): Quat {
  return [
    a[3] * b[0] + a[0] * b[3] + a[1] * b[2] - a[2] * b[1],
    a[3] * b[1] - a[0] * b[2] + a[1] * b[3] + a[2] * b[0],
    a[3] * b[2] + a[0] * b[1] - a[1] * b[0] + a[2] * b[3],
    a[3] * b[3] - a[0] * b[0] - a[1] * b[1] - a[2] * b[2],
  ];
}

export function quatNormalize(q: Quat): Quat {
  const l = Math.hypot(q[0], q[1], q[2], q[3]) || 1;
  return [q[0] / l, q[1] / l, q[2] / l, q[3] / l];
}

/** Yaw about world +Y, then pitch about the camera's local +X (yaw 0 looks down -Z; pitch > 0 looks up). */
export function quatFromYawPitch(yaw: number, pitch: number): Quat {
  const qy: Quat = [0, Math.sin(yaw / 2), 0, Math.cos(yaw / 2)];
  const qx: Quat = [Math.sin(pitch / 2), 0, 0, Math.cos(pitch / 2)];
  return quatMul(qy, qx);
}

export function quatRotate(q: Quat, v: Vec3): Vec3 {
  const [x, y, z, w] = q;
  // v' = v + 2w (q.xyz × v) + 2 q.xyz × (q.xyz × v)
  const tx = 2 * (y * v[2] - z * v[1]);
  const ty = 2 * (z * v[0] - x * v[2]);
  const tz = 2 * (x * v[1] - y * v[0]);
  return [v[0] + w * tx + (y * tz - z * ty), v[1] + w * ty + (z * tx - x * tz), v[2] + w * tz + (x * ty - y * tx)];
}

/** Forward (-Z) of an orientation. */
export function quatForward(q: Quat): Vec3 { return quatRotate(q, [0, 0, -1]); }

export function yawPitchFromQuat(q: Quat): { yaw: number; pitch: number } {
  const f = quatForward(q);
  return { yaw: Math.atan2(-f[0], -f[2]), pitch: Math.asin(Math.max(-1, Math.min(1, f[1]))) };
}

export function slerp(a: Quat, b: Quat, t: number): Quat {
  let d = a[0] * b[0] + a[1] * b[1] + a[2] * b[2] + a[3] * b[3];
  const bb: Quat = d < 0 ? [-b[0], -b[1], -b[2], -b[3]] : [...b];
  d = Math.abs(d);
  if (d > 0.9995) return quatNormalize([0, 1, 2, 3].map((i) => a[i] + (bb[i] - a[i]) * t) as Quat);
  const th = Math.acos(d);
  const s = Math.sin(th);
  const wa = Math.sin((1 - t) * th) / s;
  const wb = Math.sin(t * th) / s;
  return [0, 1, 2, 3].map((i) => wa * a[i] + wb * bb[i]) as Quat;
}

/** Column-major camera-to-world matrix from a position and orientation. */
export function poseMatrix(p: Vec3, q: Quat): Float64Array {
  const [x, y, z, w] = q;
  const m = new Float64Array(16);
  m[0] = 1 - 2 * (y * y + z * z); m[1] = 2 * (x * y + z * w); m[2] = 2 * (x * z - y * w);
  m[4] = 2 * (x * y - z * w); m[5] = 1 - 2 * (x * x + z * z); m[6] = 2 * (y * z + x * w);
  m[8] = 2 * (x * z + y * w); m[9] = 2 * (y * z - x * w); m[10] = 1 - 2 * (x * x + y * y);
  m[12] = p[0]; m[13] = p[1]; m[14] = p[2]; m[15] = 1;
  return m;
}

/** Inverse of poseMatrix for a (possibly scaled) rigid column-major matrix. */
export function matrixToPose(m: ArrayLike<number>): { position: Vec3; quaternion: Quat } {
  const c = [0, 1, 2].map((j) => {
    const l = Math.hypot(m[j * 4], m[j * 4 + 1], m[j * 4 + 2]) || 1;
    return [m[j * 4] / l, m[j * 4 + 1] / l, m[j * 4 + 2] / l];
  });
  const m00 = c[0][0], m10 = c[0][1], m20 = c[0][2];
  const m01 = c[1][0], m11 = c[1][1], m21 = c[1][2];
  const m02 = c[2][0], m12 = c[2][1], m22 = c[2][2];
  const tr = m00 + m11 + m22;
  let q: Quat;
  if (tr > 0) {
    const s = Math.sqrt(tr + 1) * 2;
    q = [(m21 - m12) / s, (m02 - m20) / s, (m10 - m01) / s, 0.25 * s];
  } else if (m00 > m11 && m00 > m22) {
    const s = Math.sqrt(1 + m00 - m11 - m22) * 2;
    q = [0.25 * s, (m01 + m10) / s, (m02 + m20) / s, (m21 - m12) / s];
  } else if (m11 > m22) {
    const s = Math.sqrt(1 + m11 - m00 - m22) * 2;
    q = [(m01 + m10) / s, 0.25 * s, (m12 + m21) / s, (m02 - m20) / s];
  } else {
    const s = Math.sqrt(1 + m22 - m00 - m11) * 2;
    q = [(m02 + m20) / s, (m12 + m21) / s, 0.25 * s, (m10 - m01) / s];
  }
  return { position: [m[12], m[13], m[14]], quaternion: quatNormalize(q) };
}

/** Orientation looking from `eye` to `target` with world +Y up (no roll). */
export function lookAtYawPitch(eye: Vec3, target: Vec3): { yaw: number; pitch: number } {
  const d: Vec3 = [target[0] - eye[0], target[1] - eye[1], target[2] - eye[2]];
  const l = Math.hypot(...d) || 1;
  return { yaw: Math.atan2(-d[0] / l, -d[2] / l), pitch: Math.asin(Math.max(-1, Math.min(1, d[1] / l))) };
}
