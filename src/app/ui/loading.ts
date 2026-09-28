// Loading overlay: stage text, progress bar and a warnings list that stays available after the load.
export class LoadingOverlay {
  readonly root: HTMLDivElement;
  private readonly title: HTMLDivElement;
  private readonly stage: HTMLDivElement;
  private readonly bar: HTMLDivElement;
  private readonly warnList: HTMLUListElement;
  private readonly warnHead: HTMLDivElement;
  private readonly closeBtn: HTMLButtonElement;
  private warnings: string[] = [];

  constructor(parent: HTMLElement) {
    this.root = el('div', 'loading');
    this.root.hidden = true;
    this.title = el('div', 'loading-title');
    this.stage = el('div', 'loading-stage');
    const track = el('div', 'loading-track');
    this.bar = el('div', 'loading-bar');
    track.append(this.bar);
    this.warnHead = el('div', 'loading-warn-head');
    this.warnList = document.createElement('ul');
    this.warnList.className = 'loading-warn';
    this.closeBtn = document.createElement('button');
    this.closeBtn.textContent = 'Close';
    this.closeBtn.className = 'loading-close';
    this.closeBtn.addEventListener('click', () => { this.root.hidden = true; });
    this.root.append(this.title, this.stage, track, this.warnHead, this.warnList, this.closeBtn);
    parent.append(this.root);
  }

  start(title: string): void {
    this.warnings = [];
    this.warnList.replaceChildren();
    this.warnHead.textContent = '';
    this.title.textContent = title;
    this.stage.textContent = 'starting...';
    this.root.classList.remove('error', 'done');
    this.closeBtn.hidden = true;
    this.progress(undefined, 'starting...');
    this.root.hidden = false;
  }

  progress(fraction: number | undefined, stage: string): void {
    this.stage.textContent = stage;
    this.bar.classList.toggle('indeterminate', fraction === undefined);
    this.bar.style.width = fraction === undefined ? '30%' : `${Math.round(Math.max(0, Math.min(1, fraction)) * 100)}%`;
  }

  warn(...msgs: string[]): void {
    for (const m of msgs) {
      this.warnings.push(m);
      const li = document.createElement('li');
      li.textContent = m;
      this.warnList.append(li);
    }
    this.warnHead.textContent = `${this.warnings.length} warning${this.warnings.length === 1 ? '' : 's'}`;
  }

  /** Hide unless there are warnings to read (then stay open with a Close button). */
  done(summary: string): void {
    this.progress(1, summary);
    this.root.classList.add('done');
    if (this.warnings.length === 0) this.root.hidden = true;
    else this.closeBtn.hidden = false;
  }

  error(msg: string): void {
    this.root.classList.add('error');
    this.stage.textContent = msg;
    this.closeBtn.hidden = false;
    this.root.hidden = false;
  }
}

function el<K extends keyof HTMLElementTagNameMap>(tag: K, cls: string): HTMLElementTagNameMap[K] {
  const e = document.createElement(tag);
  e.className = cls;
  return e;
}
