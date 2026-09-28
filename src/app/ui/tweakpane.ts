// Minimal typings for the Tweakpane 4 API we use. The published package's .d.ts imports '@tweakpane/core',
// which is not shipped, so Pane's inherited methods are untyped; this shim restores type checking.
import { Pane } from 'tweakpane';

export interface TpEvent<T> { value: T; last?: boolean }
export interface TpBlade {
  hidden: boolean;
  disabled: boolean;
  dispose(): void;
  element: HTMLElement;
}
export interface TpBinding<T = unknown> extends TpBlade {
  on(ev: 'change', cb: (e: TpEvent<T>) => void): TpBinding<T>;
  refresh(): void;
  label?: string;
}
export interface TpButton extends TpBlade {
  on(ev: 'click', cb: () => void): TpButton;
  title: string;
}
export interface TpFolder extends TpBlade {
  expanded: boolean;
  title: string | undefined;
  children: TpBlade[];
  addFolder(p: { title: string; expanded?: boolean; index?: number }): TpFolder;
  addBinding<O extends object, K extends keyof O>(obj: O, key: K, params?: Record<string, unknown> & { index?: number }): TpBinding<O[K]>;
  addButton(p: { title: string; label?: string; index?: number }): TpButton;
  addBlade(p: Record<string, unknown>): TpBlade;
  refresh(): void;
}
export interface TpPane extends TpFolder { dispose(): void }

export function createPane(container: HTMLElement, title: string): TpPane {
  return new Pane({ container, title }) as unknown as TpPane;
}
