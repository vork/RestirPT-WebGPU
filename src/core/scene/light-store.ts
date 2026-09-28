// LightStore: the single CPU-side source of truth for the scene's analytic lights (plan §1.4, §3 step 0, §5 M3a).
// The light editor, the animation system and file loaders WRITE it; the renderer (lights-gpu.ts: cur/prev buffers,
// id maps, deterministic alias rebuild) and the Cycles exporter READ it.
//
// Contract for consumers (renderer / lights-gpu.ts):
//   const store = ensureLightStore(scene);            // idempotent: the editor and the renderer share one instance
//   store.list()                                      // LightData[] sorted by ascending stable id (deterministic)
//   store.version                                     // bumps once per notified change batch
//   store.onChange((b) => { ... b.events ... })       // one callback per batch (a user edit, one animation step...)
//   b.lightsChanged / b.pmfChanged                    // convenience flags (see LightChangeBatch)
// The store also mirrors its list into `scene.lights` (same array object, contents replaced) so code that reads
// SceneData (exportScenePackage, loaders' consumers) always sees the edited lights.
//
// Stable ids (plan §1.4 "stable lightId"; temporal curToPrev/prevToCur maps): id = slot | generation << 16.
//   - Slots are recycled through a free list (compact: slot < capacity), generations make every issued id unique for
//     the lifetime of the store, so a new light never inherits a removed light's id (no false temporal reuse).
//   - restore(light) re-inserts a removed light with its ORIGINAL id (undo), which is the same light again.
//   - Changing a light's type is remove + add with a NEW id (plan §1.4), reported as one 'typeChanged' event.
// File lights keep their loader ids when those are valid, unique slot-0-generation ids (glTF/USD assign 0..n−1).
//
// Validation: every stored light passes validateLight() — rigid matrices only (rotation + translation, no scale,
// shear or mirror: lights are unscaled, plan §1.4), finite non-negative radiometry, type-specific shape fields.
import { isRigid } from './scene-package.ts';
import { TRI_EMISSIVE, type LightData, type LightType, type SceneData } from './types.ts';

export const LIGHT_SLOT_BITS = 16;
export const LIGHT_MAX_SLOTS = 1 << LIGHT_SLOT_BITS;
export const LIGHT_MAX_GENERATION = (1 << (31 - LIGHT_SLOT_BITS)) - 1; // ids stay positive in i32 / u32
export const LIGHT_TYPES: readonly LightType[] = ['point', 'spot', 'rect', 'disk', 'sun'];

export const lightSlot = (id: number): number => id & (LIGHT_MAX_SLOTS - 1);
export const lightGeneration = (id: number): number => id >>> LIGHT_SLOT_BITS;
export const makeLightId = (slot: number, gen: number): number => ((gen << LIGHT_SLOT_BITS) | slot) >>> 0;

export type LightChangeKind = 'moved' | 'radiometric' | 'added' | 'removed' | 'typeChanged';

/** Fields whose change is a 'moved' event; every other editable field is 'radiometric'. */
export const GEOMETRIC_FIELDS: readonly (keyof LightData)[] = ['matrix'];
/** Fields that change a light's selection weight (the global power alias table, plan §1.4). */
export const PMF_FIELDS: readonly (keyof LightData)[] = ['power', 'color', 'exposure'];

export interface LightChangeEvent {
  kind: LightChangeKind;
  /** The light's id (for typeChanged: the NEW id). */
  id: number;
  /** typeChanged: the removed (old) id. */
  prevId?: number;
  /** moved / radiometric: which fields changed. */
  fields?: (keyof LightData)[];
  /** State after the change (undefined for 'removed'). Frozen snapshot. */
  light?: Readonly<LightData>;
  /** State before the change (undefined for 'added'). Frozen snapshot. */
  before?: Readonly<LightData>;
}

export interface LightChangeBatch {
  events: LightChangeEvent[];
  /** store.version after this batch. */
  version: number;
  /** Who caused it ('editor', 'animation', 'undo', 'load', ...), informational. */
  source: string;
  /** Any light buffer content changed (always true for a non-empty batch). */
  lightsChanged: boolean;
  /** The set of lights or a selection weight (power/colour/exposure) changed: rebuild the power alias table. */
  pmfChanged: boolean;
  /** Lights were added/removed/retyped: id maps (curToPrev/prevToCur) change. */
  idsChanged: boolean;
}

/** Everything but the id may be given when adding; missing type-specific fields get Blender-like defaults. */
export type LightInit = Partial<Omit<LightData, 'id' | 'type'>> & { type: LightType };
export type LightPatch = Partial<Omit<LightData, 'id' | 'type'>>;

export class LightValidationError extends Error {}

/** Emissive-mesh entry: listed by the editor as static (read-only; emissive triangles live in the BVH). */
export interface EmissiveMeshInfo {
  materialIndex: number;
  name: string;
  triangles: number;
  /** Total one-sided area (m²). */
  area: number;
  /** emissiveFactor · emissiveStrength (textures not included). */
  radiance: [number, number, number];
  /** Approximate emitted power, two-sided Lambertian: 2π·A·mean(L) (W); informational. */
  approxPower: number;
  textured: boolean;
}

const DEFAULTS: Record<LightType, { power: number; name: string }> = {
  point: { power: 100, name: 'Point' },
  spot: { power: 100, name: 'Spot' },
  rect: { power: 50, name: 'Area' },
  disk: { power: 50, name: 'Disk' },
  sun: { power: 3, name: 'Sun' },
};

/** −0 → +0 (same value; keeps JSON round trips bit-exact, since JSON has no −0). */
const canon = (x: number): number => x + 0;
export const canonicalMatrix = (m: ArrayLike<number>): Float32Array => Float32Array.from(m as ArrayLike<number>, canon);

const IDENTITY = (): Float32Array => new Float32Array([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]);

/** Fill defaults and drop fields that do not belong to the type. Does not validate. */
export function normalizeLight(init: LightInit & { id?: number }, id: number): LightData {
  const t = init.type;
  const l: LightData = {
    id,
    name: init.name ?? DEFAULTS[t]?.name ?? t,
    type: t,
    color: init.color ? [canon(init.color[0]), canon(init.color[1]), canon(init.color[2])] : [1, 1, 1],
    power: canon(init.power ?? DEFAULTS[t]?.power ?? 1),
    exposure: canon(init.exposure ?? 0),
    matrix: init.matrix ? canonicalMatrix(init.matrix) : IDENTITY(),
    visibleToCamera: init.visibleToCamera ?? false,
  };
  if (t === 'spot') { l.spotSize = init.spotSize ?? Math.PI / 4; l.spotBlend = init.spotBlend ?? 0.15; }
  if (t === 'rect') { l.sizeX = init.sizeX ?? 1; l.sizeY = init.sizeY ?? init.sizeX ?? 1; }
  if (t === 'disk') { l.sizeX = init.sizeX ?? 1; }
  if (t === 'rect' || t === 'disk') l.spread = init.spread ?? Math.PI;
  if (t !== 'rect' && t !== 'disk') l.visibleToCamera = false; // only area lights can be seen (plan §1.4 table)
  if (init.simplified !== undefined) l.simplified = init.simplified;
  return l;
}

/** Throws LightValidationError describing the first problem. */
export function validateLight(l: LightData): void {
  const bad = (m: string): never => { throw new LightValidationError(`light ${l.id} '${l.name}': ${m}`); };
  if (!Number.isInteger(l.id) || l.id < 0 || l.id > 0x7fffffff) bad(`bad id ${l.id}`);
  if (!LIGHT_TYPES.includes(l.type)) bad(`unknown type '${String(l.type)}'`);
  if (typeof l.name !== 'string') bad('name must be a string');
  if (!Array.isArray(l.color) || l.color.length !== 3 || l.color.some((c) => !(Number.isFinite(c) && c >= 0))) bad('colour must be 3 finite values ≥ 0');
  if (!(Number.isFinite(l.power) && l.power >= 0)) bad(`power must be finite and ≥ 0 (got ${l.power})`);
  if (!Number.isFinite(l.exposure)) bad('exposure must be finite');
  if (!(l.matrix instanceof Float32Array) || l.matrix.length !== 16 || Array.from(l.matrix).some((x) => !Number.isFinite(x))) bad('matrix must be 16 finite floats');
  if (!isRigid(l.matrix)) bad('matrix has scale/shear/mirror (lights are rigid: rotation + translation only)');
  if (typeof l.visibleToCamera !== 'boolean') bad('visibleToCamera must be boolean');
  if (l.type === 'spot') {
    if (!(l.spotSize! > 0 && l.spotSize! <= Math.PI)) bad(`spotSize must be in (0, π] (got ${l.spotSize})`);
    if (!(l.spotBlend! >= 0 && l.spotBlend! <= 1)) bad(`spotBlend must be in [0, 1] (got ${l.spotBlend})`);
  }
  if (l.type === 'rect' || l.type === 'disk') {
    if (!(l.sizeX! > 0 && Number.isFinite(l.sizeX))) bad(`sizeX must be > 0 (got ${l.sizeX})`);
    if (l.type === 'rect' && !(l.sizeY! > 0 && Number.isFinite(l.sizeY))) bad(`sizeY must be > 0 (got ${l.sizeY})`);
    if (l.type === 'disk' && l.sizeY !== undefined && l.sizeY !== l.sizeX) bad('disk lights are circular (sizeY must equal sizeX)');
    if (!(l.spread! > 0 && l.spread! <= Math.PI)) bad(`spread must be in (0, π] (got ${l.spread})`);
  } else if (l.visibleToCamera) bad(`${l.type} lights cannot be visible to the camera`);
}

/** Deep copy (own matrix / colour arrays). */
export function cloneLight(l: Readonly<LightData>): LightData {
  const c: LightData = { ...l, color: [l.color[0], l.color[1], l.color[2]], matrix: Float32Array.from(l.matrix) };
  return c;
}

function freeze(l: LightData): Readonly<LightData> {
  const c = cloneLight(l);
  Object.freeze(c.color);
  return Object.freeze(c);
}

function sameValue(a: unknown, b: unknown): boolean {
  if (a === b) return true;
  if ((a instanceof Float32Array || Array.isArray(a)) && (b instanceof Float32Array || Array.isArray(b))) {
    if (a.length !== b.length) return false;
    for (let i = 0; i < a.length; i++) if (!Object.is(a[i], b[i])) return false;
    return true;
  }
  return false;
}

/** Persisted allocator state (so ids stay unique across save/load of the editor state). */
export interface LightStoreIdState { nextGen: number[] }

interface Slot { gen: number; nextGen: number; light: LightData | undefined }

export class LightStore {
  private readonly slots: Slot[] = [];
  private readonly freeSlots: number[] = [];
  private readonly listeners = new Set<(b: LightChangeBatch) => void>();
  private pending: LightChangeEvent[] | undefined;
  private pendingSource = '';
  private batchDepth = 0;
  private mirror: LightData[] | undefined;
  private sorted: LightData[] | undefined;
  version = 0;

  /** Build from file lights (they become editable entries). Keeps loader ids when valid and unique. */
  static fromLights(lights: readonly LightData[], mirror?: LightData[]): LightStore {
    const s = new LightStore();
    const keep = new Set<number>();
    for (const l of lights) {
      if (Number.isInteger(l.id) && l.id >= 0 && l.id < LIGHT_MAX_SLOTS && !keep.has(l.id)) keep.add(l.id);
    }
    const reassigned: LightData[] = [];
    for (const l of lights) {
      if (keep.has(l.id)) { keep.delete(l.id); s.insert(normalizeLight({ ...l }, l.id)); } else reassigned.push(l);
    }
    for (const l of reassigned) s.insert(normalizeLight({ ...l }, s.allocate()));
    s.mirror = mirror;
    s.syncMirror();
    return s;
  }

  static fromScene(scene: SceneData): LightStore { return LightStore.fromLights(scene.lights, scene.lights); }

  // ---- reads ------------------------------------------------------------------------------------------------------

  /** All lights sorted by ascending id (deterministic order for alias-table rebuilds). Do not mutate. */
  list(): readonly LightData[] {
    if (!this.sorted) {
      this.sorted = [];
      for (const s of this.slots) if (s.light) this.sorted.push(s.light);
      this.sorted.sort((a, b) => a.id - b.id);
    }
    return this.sorted;
  }

  /** Deep-copied list (e.g. the "prev" light buffer of the temporal pipeline). */
  snapshot(): LightData[] { return this.list().map(cloneLight); }

  get size(): number { return this.list().length; }

  get(id: number): Readonly<LightData> | undefined {
    const s = this.slots[lightSlot(id)];
    return s && s.light && s.light.id === id ? s.light : undefined;
  }

  has(id: number): boolean { return this.get(id) !== undefined; }

  /** Highest slot index + 1 ever used (capacity hint for id → index maps). */
  get slotCapacity(): number { return this.slots.length; }

  // ---- writes -----------------------------------------------------------------------------------------------------

  /** Add a light with a freshly allocated id. */
  add(init: LightInit, source = 'editor'): Readonly<LightData> {
    const slot = this.freeSlots.length ? this.freeSlots.pop()! : this.slots.length;
    if (slot >= LIGHT_MAX_SLOTS) throw new LightValidationError(`too many lights (max ${LIGHT_MAX_SLOTS})`);
    const cur = this.slots[slot] ?? { gen: 0, nextGen: 0, light: undefined };
    if (cur.nextGen > LIGHT_MAX_GENERATION) throw new LightValidationError(`light slot ${slot} exhausted its generations`);
    const l = normalizeLight(init, makeLightId(slot, cur.nextGen));
    try { validateLight(l); } catch (e) { if (slot < this.slots.length) this.freeSlots.push(slot); throw e; }
    this.insert(l);
    this.emit({ kind: 'added', id: l.id, light: freeze(l) }, source);
    return this.get(l.id)!;
  }

  /** Re-insert a light with its original id (undo of remove, load). The id must not be in use. */
  restore(light: Readonly<LightData>, source = 'undo'): Readonly<LightData> {
    const l = normalizeLight({ ...light }, light.id);
    validateLight(l);
    const slot = lightSlot(l.id);
    if (this.slots[slot]?.light) throw new LightValidationError(`cannot restore light ${l.id}: slot ${slot} is in use by ${this.slots[slot].light!.id}`);
    this.insert(l);
    this.emit({ kind: 'added', id: l.id, light: freeze(l) }, source);
    return this.get(l.id)!;
  }

  /** Patch editable fields. Emits 'moved' (matrix) and/or 'radiometric' (anything else); no event if nothing changed. */
  update(id: number, patch: LightPatch, source = 'editor'): Readonly<LightData> {
    const cur = this.get(id);
    if (!cur) throw new LightValidationError(`no light with id ${id}`);
    if ('id' in patch || 'type' in patch) throw new LightValidationError('update() cannot change id or type (use changeType)');
    const next = cloneLight(cur);
    const changed: (keyof LightData)[] = [];
    for (const k of Object.keys(patch) as (keyof LightPatch)[]) {
      const v = patch[k];
      if (v === undefined) continue;
      if (sameValue(cur[k], v)) continue;
      changed.push(k);
      if (k === 'matrix') next.matrix = canonicalMatrix(v as ArrayLike<number>);
      else if (k === 'color') next.color = [canon((v as number[])[0]), canon((v as number[])[1]), canon((v as number[])[2])];
      else (next as unknown as Record<string, unknown>)[k] = typeof v === 'number' ? canon(v) : v;
    }
    if (!changed.length) return cur;
    if (next.type === 'disk' && changed.includes('sizeX')) delete next.sizeY;
    validateLight(next);
    const before = freeze(cur as LightData);
    this.slots[lightSlot(id)].light = next;
    this.sorted = undefined;
    const geo = changed.filter((k) => GEOMETRIC_FIELDS.includes(k));
    const rad = changed.filter((k) => !GEOMETRIC_FIELDS.includes(k));
    const snap = freeze(next);
    this.withBatch(source, () => {
      if (geo.length) this.emit({ kind: 'moved', id, fields: geo, light: snap, before }, source);
      if (rad.length) this.emit({ kind: 'radiometric', id, fields: rad, light: snap, before }, source);
    });
    return this.get(id)!;
  }

  remove(id: number, source = 'editor'): Readonly<LightData> {
    const cur = this.get(id);
    if (!cur) throw new LightValidationError(`no light with id ${id}`);
    const slot = lightSlot(id);
    this.slots[slot].light = undefined;
    this.freeSlots.push(slot);
    this.sorted = undefined;
    const before = freeze(cur as LightData);
    this.emit({ kind: 'removed', id, before }, source);
    return before;
  }

  /** Type change = remove + add with a NEW id (plan §1.4). Shared fields (name, colour, power, exposure, matrix,
   *  sizes where meaningful) carry over. Returns the new light. */
  changeType(id: number, type: LightType, source = 'editor', overrides: LightPatch = {}): Readonly<LightData> {
    const cur = this.get(id);
    if (!cur) throw new LightValidationError(`no light with id ${id}`);
    if (cur.type === type) return cur;
    const init: LightInit = {
      type, name: cur.name, color: [...cur.color], power: cur.power, exposure: cur.exposure, matrix: cur.matrix,
      sizeX: cur.sizeX, sizeY: cur.type === 'rect' ? cur.sizeY : cur.sizeX, spread: cur.spread, spotSize: cur.spotSize, spotBlend: cur.spotBlend,
      visibleToCamera: (type === 'rect' || type === 'disk') ? cur.visibleToCamera : false,
      ...overrides,
    };
    let added: Readonly<LightData> | undefined;
    this.withBatch(source, () => {
      const before = freeze(cur as LightData);
      const slot = lightSlot(id);
      this.slots[slot].light = undefined;
      this.freeSlots.push(slot);
      this.sorted = undefined;
      // allocate without emitting 'added': one typeChanged event carries both ids
      const nslot = this.freeSlots.length ? this.freeSlots.pop()! : this.slots.length;
      const cs = this.slots[nslot] ?? { gen: 0, nextGen: 0, light: undefined };
      const l = normalizeLight(init, makeLightId(nslot, cs.nextGen));
      try { validateLight(l); } catch (e) {
        // roll back
        this.freeSlots.push(nslot);
        this.slots[slot].light = cur as LightData;
        this.freeSlots.splice(this.freeSlots.indexOf(slot), 1);
        this.sorted = undefined;
        throw e;
      }
      this.insert(l);
      added = this.get(l.id)!;
      this.emit({ kind: 'typeChanged', id: l.id, prevId: id, light: freeze(l), before }, source);
    });
    return added!;
  }

  /** Undo of changeType: remove `newId` and restore `old` with its original id, as one typeChanged event. */
  revertType(newId: number, old: Readonly<LightData>, source = 'undo'): Readonly<LightData> {
    const cur = this.get(newId);
    if (!cur) throw new LightValidationError(`no light with id ${newId}`);
    let restored: Readonly<LightData> | undefined;
    this.withBatch(source, () => {
      const before = freeze(cur as LightData);
      const slot = lightSlot(newId);
      this.slots[slot].light = undefined;
      this.freeSlots.push(slot);
      this.sorted = undefined;
      const l = normalizeLight({ ...old }, old.id);
      validateLight(l);
      if (this.slots[lightSlot(l.id)]?.light) throw new LightValidationError(`cannot restore light ${l.id}: slot in use`);
      this.insert(l);
      restored = this.get(l.id)!;
      this.emit({ kind: 'typeChanged', id: l.id, prevId: newId, light: freeze(l), before }, source);
    });
    return restored!;
  }

  /** Replace every light (editor-state load). Emits removed/added events in one batch. */
  replaceAll(lights: readonly LightData[], idState?: LightStoreIdState, source = 'load'): void {
    this.withBatch(source, () => {
      for (const l of [...this.list()]) this.remove(l.id, source);
      if (idState) {
        for (let i = 0; i < idState.nextGen.length; i++) {
          const s = this.slots[i] ?? (this.slots[i] = { gen: 0, nextGen: 0, light: undefined });
          s.nextGen = Math.max(s.nextGen, idState.nextGen[i] | 0);
          if (!s.light && !this.freeSlots.includes(i)) this.freeSlots.push(i);
        }
      }
      for (const l of lights) this.restore(l, source);
    });
  }

  idState(): LightStoreIdState { return { nextGen: this.slots.map((s) => s.nextGen) }; }

  /** Group several mutations into ONE change notification (e.g. one animation step updating many lights). */
  batch<T>(source: string, fn: () => T): T { return this.withBatch(source, fn); }

  onChange(cb: (b: LightChangeBatch) => void): () => void {
    this.listeners.add(cb);
    return () => this.listeners.delete(cb);
  }

  /** Emissive-mesh lights of a scene (static, read-only in the editor). */
  static emissiveMeshes(scene: SceneData): EmissiveMeshInfo[] {
    const g = scene.geometry;
    const acc = new Map<number, { tris: number; area: number }>();
    const p = g.positions;
    for (let t = 0; t < g.triMaterial.length; t++) {
      if (!(g.triFlags[t] & TRI_EMISSIVE)) continue;
      const m = g.triMaterial[t];
      const a = g.indices[3 * t] * 3, b = g.indices[3 * t + 1] * 3, c = g.indices[3 * t + 2] * 3;
      const ex = p[b] - p[a], ey = p[b + 1] - p[a + 1], ez = p[b + 2] - p[a + 2];
      const fx = p[c] - p[a], fy = p[c + 1] - p[a + 1], fz = p[c + 2] - p[a + 2];
      const area = 0.5 * Math.hypot(ey * fz - ez * fy, ez * fx - ex * fz, ex * fy - ey * fx);
      const e = acc.get(m) ?? { tris: 0, area: 0 };
      e.tris++; e.area += area;
      acc.set(m, e);
    }
    return [...acc.entries()].sort((x, y) => x[0] - y[0]).map(([mi, e]) => {
      const mat = scene.materials[mi];
      const s = mat.emissiveStrength;
      const L: [number, number, number] = [mat.emissiveFactor[0] * s, mat.emissiveFactor[1] * s, mat.emissiveFactor[2] * s];
      return {
        materialIndex: mi, name: mat.name, triangles: e.tris, area: e.area, radiance: L,
        approxPower: 2 * Math.PI * e.area * (L[0] + L[1] + L[2]) / 3, textured: !!mat.emissiveTexture,
      };
    });
  }

  // ---- internals --------------------------------------------------------------------------------------------------

  private allocate(): number {
    const slot = this.freeSlots.length ? this.freeSlots.pop()! : this.slots.length;
    const cur = this.slots[slot] ?? { gen: 0, nextGen: 0, light: undefined };
    this.slots[slot] = cur;
    return makeLightId(slot, cur.nextGen);
  }

  private insert(l: LightData): void {
    const slot = lightSlot(l.id);
    const gen = lightGeneration(l.id);
    while (this.slots.length <= slot) {
      const i = this.slots.length;
      this.slots.push({ gen: 0, nextGen: 0, light: undefined });
      if (i !== slot) this.freeSlots.push(i);
    }
    const s = this.slots[slot];
    const fi = this.freeSlots.indexOf(slot);
    if (fi >= 0) this.freeSlots.splice(fi, 1);
    s.light = l;
    s.gen = gen;
    s.nextGen = Math.max(s.nextGen, gen + 1);
    this.sorted = undefined;
  }

  private withBatch<T>(source: string, fn: () => T): T {
    if (this.batchDepth === 0) { this.pending = []; this.pendingSource = source; }
    this.batchDepth++;
    let ok = false;
    try {
      const r = fn();
      ok = true;
      return r;
    } finally {
      this.batchDepth--;
      if (this.batchDepth === 0) {
        const ev = this.pending ?? [];
        this.pending = undefined;
        if (ev.length || !ok) this.flush(ev, this.pendingSource);
      }
    }
  }

  private emit(e: LightChangeEvent, source: string): void {
    if (this.batchDepth > 0 && this.pending) { this.pending.push(e); return; }
    this.flush([e], source);
  }

  private flush(events: LightChangeEvent[], source: string): void {
    this.syncMirror();
    if (!events.length) return;
    this.version++;
    const batch: LightChangeBatch = {
      events, version: this.version, source,
      lightsChanged: true,
      idsChanged: events.some((e) => e.kind === 'added' || e.kind === 'removed' || e.kind === 'typeChanged'),
      pmfChanged: events.some((e) => e.kind !== 'moved' && (e.kind !== 'radiometric' || e.fields!.some((f) => PMF_FIELDS.includes(f)))),
    };
    for (const cb of [...this.listeners]) {
      try { cb(batch); } catch (err) { console.error('[light-store] listener failed', err); }
    }
  }

  private syncMirror(): void {
    if (!this.mirror) return;
    this.mirror.length = 0;
    for (const l of this.list()) this.mirror.push(l);
  }
}

const stores = new WeakMap<SceneData, LightStore>();

/** The scene's LightStore, created from its file lights on first use. Editor and renderer share this instance. */
export function ensureLightStore(scene: SceneData): LightStore {
  let s = stores.get(scene);
  if (!s) { s = LightStore.fromScene(scene); stores.set(scene, s); }
  return s;
}

export function lightStoreOf(scene: SceneData): LightStore | undefined { return stores.get(scene); }
