// Editor state document ("scene.json" of the editor, plan §5 M3a exit: "scene.json round-trips losslessly"):
// lights (+ id allocator state), animation tracks, camera pose, environment parameters and timeline settings.
// Frame and units: glTF canonical, un-recentred world coordinates, Blender light units (plan §1.2); the geometry is
// NOT included (it is referenced by source.uri and loaded by the scene loaders).
//
// Lossless: every number is written as a JSON double (shortest round-trip form), f32 light matrices as their exact
// f32 values (a double that parses back to the same f32), quaternions/positions as f64. parse(serialize(s)) is
// bit-identical, and serialize(parse(text)) === text (tests/editor/editor-state.test.ts).
import type { AnimationJson } from '../../core/scene/animation.ts';
import { LightValidationError, normalizeLight, validateLight, type LightStoreIdState } from '../../core/scene/light-store.ts';
import type { LightData, LightType } from '../../core/scene/types.ts';

export const EDITOR_STATE_FORMAT = 'restir-editor-scene';
export const EDITOR_STATE_VERSION = 1;

type V3 = [number, number, number];

export interface EditorLightJson {
  id: number; name: string; type: LightType; color: V3; power: number; exposure: number; matrix: number[];
  spotSize?: number; spotBlend?: number; sizeX?: number; sizeY?: number; spread?: number; visibleToCamera: boolean; simplified?: string;
}

export interface EditorStateJson {
  format: typeof EDITOR_STATE_FORMAT;
  version: number;
  /** The scene the lights belong to (informational; the app warns when it differs from the loaded scene). */
  source?: { uri?: string; name?: string };
  lights: EditorLightJson[];
  lightIds?: LightStoreIdState;
  camera: { position: V3; quaternion: [number, number, number, number]; yfov: number };
  env?: { url: string; strength: number; rotationZ: number; tint: V3; visibleToCamera: boolean };
  timeline: { time: number; mode: 'interactive' | 'validation'; frame: number };
  animation: AnimationJson;
}

export interface EditorState {
  source?: { uri?: string; name?: string };
  lights: LightData[];
  lightIds?: LightStoreIdState;
  camera: EditorStateJson['camera'];
  env?: EditorStateJson['env'];
  timeline: EditorStateJson['timeline'];
  animation: AnimationJson;
}

export class EditorStateError extends Error {}

function lightToJson(l: LightData): EditorLightJson {
  const j: EditorLightJson = {
    id: l.id, name: l.name, type: l.type, color: [l.color[0], l.color[1], l.color[2]], power: l.power, exposure: l.exposure,
    matrix: Array.from(l.matrix), visibleToCamera: l.visibleToCamera,
  };
  for (const k of ['spotSize', 'spotBlend', 'sizeX', 'sizeY', 'spread', 'simplified'] as const) if (l[k] !== undefined) (j as unknown as Record<string, unknown>)[k] = l[k];
  return j;
}

function lightFromJson(j: EditorLightJson): LightData {
  if (!Array.isArray(j.matrix) || j.matrix.length !== 16) throw new EditorStateError(`light ${j.id}: matrix must have 16 numbers`);
  const m = Float32Array.from(j.matrix);
  for (let i = 0; i < 16; i++) {
    if (m[i] !== j.matrix[i] && !(Number.isNaN(m[i]) && Number.isNaN(j.matrix[i]))) throw new EditorStateError(`light ${j.id}: matrix[${i}] = ${j.matrix[i]} is not an f32 value (not lossless)`);
  }
  const l = normalizeLight({ ...j, matrix: m }, j.id);
  try { validateLight(l); } catch (e) { throw new EditorStateError((e as LightValidationError).message); }
  return l;
}

export function editorStateToJson(s: EditorState): EditorStateJson {
  const out: EditorStateJson = {
    format: EDITOR_STATE_FORMAT, version: EDITOR_STATE_VERSION,
    ...(s.source ? { source: { ...s.source } } : {}),
    lights: [...s.lights].sort((a, b) => a.id - b.id).map(lightToJson),
    ...(s.lightIds ? { lightIds: { nextGen: [...s.lightIds.nextGen] } } : {}),
    // −0 → +0 (JSON cannot carry −0; same value)
    camera: { position: s.camera.position.map((x) => x + 0) as V3, quaternion: s.camera.quaternion.map((x) => x + 0) as EditorStateJson['camera']['quaternion'], yfov: s.camera.yfov },
    ...(s.env ? { env: { ...s.env, rotationZ: s.env.rotationZ + 0, tint: s.env.tint.map((x) => x + 0) as V3 } } : {}),
    timeline: { ...s.timeline },
    animation: structuredClone(s.animation),
  };
  return out;
}

export function serializeEditorState(s: EditorState): string { return JSON.stringify(editorStateToJson(s), null, 1); }

export function parseEditorState(text: string | EditorStateJson): EditorState {
  const j = (typeof text === 'string' ? JSON.parse(text) : text) as EditorStateJson;
  if (j?.format !== EDITOR_STATE_FORMAT) throw new EditorStateError(`not an editor scene (format '${String(j?.format)}')`);
  if (j.version !== EDITOR_STATE_VERSION) throw new EditorStateError(`unsupported editor scene version ${j.version}`);
  const ids = new Set<number>();
  const lights = (j.lights ?? []).map((lj) => {
    if (ids.has(lj.id)) throw new EditorStateError(`duplicate light id ${lj.id}`);
    ids.add(lj.id);
    return lightFromJson(lj);
  });
  const c = j.camera;
  if (!c || c.position?.length !== 3 || c.quaternion?.length !== 4 || !(c.yfov > 0 && c.yfov < Math.PI)) throw new EditorStateError('bad camera');
  if (!j.animation || j.animation.version !== 1) throw new EditorStateError('missing/unsupported animation');
  return {
    ...(j.source ? { source: { ...j.source } } : {}),
    lights,
    ...(j.lightIds ? { lightIds: { nextGen: [...j.lightIds.nextGen] } } : {}),
    camera: { position: [...c.position], quaternion: [...c.quaternion], yfov: c.yfov },
    ...(j.env ? { env: { ...j.env, tint: [...j.env.tint] as V3 } } : {}),
    timeline: { time: j.timeline?.time ?? 0, mode: j.timeline?.mode === 'validation' ? 'validation' : 'interactive', frame: j.timeline?.frame ?? 0 },
    animation: structuredClone(j.animation),
  };
}
