// Undo/redo command stack for the editor (plan §5 M3a "delete, duplicate, undo/redo"). Commands are small
// do/undo closures over the LightStore and the Animation; drags are recorded as ONE command at pointer-up.
import type { Animation, TargetId, Track } from '../../core/scene/animation.ts';
import { lightTarget } from '../../core/scene/animation.ts';
import { cloneLight, type LightStore, type LightInit, type LightPatch } from '../../core/scene/light-store.ts';
import type { LightData, LightType } from '../../core/scene/types.ts';

export interface Command {
  label: string;
  do(): void;
  undo(): void;
}

export class UndoStack {
  private readonly done: Command[] = [];
  private readonly undone: Command[] = [];
  readonly listeners = new Set<() => void>();
  constructor(public limit = 200) {}

  /** Execute (unless `alreadyApplied`) and record. Clears the redo branch. */
  push(cmd: Command, alreadyApplied = false): void {
    if (!alreadyApplied) cmd.do();
    this.done.push(cmd);
    if (this.done.length > this.limit) this.done.shift();
    this.undone.length = 0;
    this.changed();
  }

  undo(): Command | undefined {
    const c = this.done.pop();
    if (!c) return undefined;
    c.undo();
    this.undone.push(c);
    this.changed();
    return c;
  }

  redo(): Command | undefined {
    const c = this.undone.pop();
    if (!c) return undefined;
    c.do();
    this.done.push(c);
    this.changed();
    return c;
  }

  get canUndo(): boolean { return this.done.length > 0; }
  get canRedo(): boolean { return this.undone.length > 0; }
  get undoLabel(): string | undefined { return this.done[this.done.length - 1]?.label; }
  get redoLabel(): string | undefined { return this.undone[this.undone.length - 1]?.label; }
  clear(): void { this.done.length = 0; this.undone.length = 0; this.changed(); }
  private changed(): void { for (const cb of this.listeners) cb(); }
}

// ---- light commands ---------------------------------------------------------------------------------------------

const SRC = 'editor';

/** Add a light. The id is allocated on first do(); redo restores the same id. */
export function addLightCommand(store: LightStore, init: LightInit, onId?: (id: number) => void): Command & { id(): number | undefined } {
  let light: LightData | undefined;
  return {
    label: `add ${init.type}`,
    id: () => light?.id,
    do() {
      light = light ? cloneLight(store.restore(light, SRC)) : cloneLight(store.add(init, SRC));
      onId?.(light.id);
    },
    undo() { if (light) store.remove(light.id, 'undo'); },
  };
}

/** Remove a light together with its animation track. */
export function removeLightCommand(store: LightStore, anim: Animation | undefined, id: number): Command {
  const before = cloneLight(store.get(id)!);
  let track: Track | undefined;
  return {
    label: `delete ${before.name}`,
    do() {
      track = anim?.removeTarget(lightTarget(id));
      store.remove(id, SRC);
    },
    undo() {
      store.restore(before, 'undo');
      if (track && anim) anim.setTrack(lightTarget(id), track);
    },
  };
}

/** Property / transform change from `before` to `after` (only the given fields). */
export function updateLightCommand(store: LightStore, id: number, before: LightPatch, after: LightPatch, label = 'edit light'): Command {
  return {
    label,
    do() { store.update(id, after, SRC); },
    undo() { store.update(id, before, 'undo'); },
  };
}

/** Type change = remove + add with a new id (plan §1.4); the animation track follows the light. */
export function changeTypeCommand(store: LightStore, anim: Animation | undefined, id: number, type: LightType, onId?: (id: number) => void): Command {
  const old = cloneLight(store.get(id)!);
  let created: LightData | undefined;
  let oldTrack: Track | undefined;
  return {
    label: `change ${old.name} to ${type}`,
    do() {
      if (created) {
        // redo: same new id as the first time
        oldTrack = anim?.removeTarget(lightTarget(old.id));
        store.revertType(old.id, created, SRC); // one typeChanged event: old id → the same new id as the first time
      } else {
        created = cloneLight(store.changeType(id, type, SRC));
        oldTrack = anim?.removeTarget(lightTarget(old.id));
      }
      if (oldTrack && anim) anim.setTrack(lightTarget(created.id), oldTrack);
      onId?.(created.id);
    },
    undo() {
      if (!created) return;
      anim?.removeTarget(lightTarget(created.id));
      store.revertType(created.id, old, 'undo');
      if (oldTrack && anim) anim.setTrack(lightTarget(old.id), oldTrack);
      onId?.(old.id);
    },
  };
}

/** Snapshot-based command for animation edits (keys, presets): restores the whole track of one target. */
export function trackCommand(anim: Animation, target: TargetId, label: string, mutate: () => void): Command {
  const before = anim.track(target) ? structuredClone(anim.track(target)) as Track : undefined;
  let after: Track | undefined;
  let first = true;
  return {
    label,
    do() {
      if (first) { mutate(); after = anim.track(target) ? structuredClone(anim.track(target)) as Track : undefined; first = false; }
      else anim.setTrack(target, after);
    },
    undo() { anim.setTrack(target, before); },
  };
}

/** Several commands as one undo step. */
export function compositeCommand(label: string, cmds: Command[]): Command {
  return {
    label,
    do() { for (const c of cmds) c.do(); },
    undo() { for (let i = cmds.length - 1; i >= 0; i--) cmds[i].undo(); },
  };
}
