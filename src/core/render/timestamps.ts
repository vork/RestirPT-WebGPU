// Per-pass GPU timings via timestamp-query (plan §6: ≥ 32-frame averages). Degrades to a no-op when the feature
// is missing. Results are read 2-3 frames later through a ring of MAP_READ buffers, so timing never stalls a frame.
// Note: Chrome quantises timestamps (100 µs) unless launched with timestamp quantisation disabled; the rolling
// averages still converge on the mean.

export const TIMESTAMP_AVG_FRAMES = 32;

type Slot = { buf: GPUBuffer; state: 'free' | 'pending' | 'mapping'; names: string[] };

export class GpuTimestamps {
  readonly supported: boolean;
  private readonly querySet: GPUQuerySet | undefined;
  private readonly resolveBuf: GPUBuffer | undefined;
  private readonly ring: Slot[] = [];
  private names: string[] = [];
  private active: Slot | undefined;
  private readonly history = new Map<string, { v: number[]; i: number; lastSeen: number }>();
  private frame = 0;
  /** Latest per-frame total (ms), and number of frames skipped because every ring slot was busy. */
  lastTotalMs = 0;
  skipped = 0;

  constructor(private readonly device: GPUDevice, readonly maxPasses = 48, ringSize = 4) {
    this.supported = device.features.has('timestamp-query');
    if (!this.supported) return;
    const n = maxPasses * 2;
    this.querySet = device.createQuerySet({ label: 'timestamps', type: 'timestamp', count: n });
    this.resolveBuf = device.createBuffer({ label: 'timestamps-resolve', size: n * 8, usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC });
    for (let i = 0; i < ringSize; i++) {
      this.ring.push({
        buf: device.createBuffer({ label: `timestamps-read${i}`, size: n * 8, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST }),
        state: 'free',
        names: [],
      });
    }
  }

  /** Start a frame: timing is recorded only if a ring slot is free. */
  beginFrame(): void {
    this.frame++;
    this.names = [];
    this.active = this.ring.find((s) => s.state === 'free');
    if (this.supported && !this.active) this.skipped++;
  }

  /** timestampWrites for a compute or render pass descriptor (same shape), or undefined. */
  pass(name: string): GPUComputePassTimestampWrites | undefined {
    if (!this.active || !this.querySet || this.names.length >= this.maxPasses) return undefined;
    const i = this.names.length;
    this.names.push(name);
    return { querySet: this.querySet, beginningOfPassWriteIndex: 2 * i, endOfPassWriteIndex: 2 * i + 1 };
  }

  /** Resolve this frame's queries into the active ring slot. Call once, after the last timed pass. */
  resolve(encoder: GPUCommandEncoder): void {
    const s = this.active;
    if (!s || !this.querySet || !this.resolveBuf || this.names.length === 0) { this.active = undefined; return; }
    const n = this.names.length * 2;
    encoder.resolveQuerySet(this.querySet, 0, n, this.resolveBuf, 0);
    encoder.copyBufferToBuffer(this.resolveBuf, 0, s.buf, 0, n * 8);
    s.names = this.names;
    s.state = 'pending';
  }

  /** Call after queue.submit(): maps the slot written this frame; results land asynchronously. */
  afterSubmit(): void {
    const s = this.active;
    this.active = undefined;
    if (!s || s.state !== 'pending') return;
    s.state = 'mapping';
    const names = s.names;
    const frame = this.frame;
    s.buf.mapAsync(GPUMapMode.READ, 0, names.length * 16).then(() => {
      const t = new BigInt64Array(s.buf.getMappedRange(0, names.length * 16).slice(0));
      s.buf.unmap();
      s.state = 'free';
      let total = 0;
      const perName = new Map<string, number>();
      names.forEach((name, i) => {
        const d = Number(t[2 * i + 1] - t[2 * i]) / 1e6;
        const ms = Number.isFinite(d) && d > 0 && d < 1e4 ? d : 0;
        perName.set(name, (perName.get(name) ?? 0) + ms);
        total += ms;
      });
      for (const [name, ms] of perName) this.push(name, ms, frame);
      this.push('total', total, frame);
      this.lastTotalMs = total;
    }).catch(() => { s.state = 'free'; });
  }

  private push(name: string, ms: number, frame: number): void {
    let h = this.history.get(name);
    if (!h) { h = { v: [], i: 0, lastSeen: frame }; this.history.set(name, h); }
    if (h.v.length < TIMESTAMP_AVG_FRAMES) h.v.push(ms); else h.v[h.i] = ms;
    h.i = (h.i + 1) % TIMESTAMP_AVG_FRAMES;
    h.lastSeen = frame;
  }

  /** Rolling averages (ms) over the last ≤32 measured frames; passes unseen for 120 frames are dropped. */
  averages(): { name: string; ms: number; samples: number }[] {
    const out: { name: string; ms: number; samples: number }[] = [];
    for (const [name, h] of this.history) {
      if (this.frame - h.lastSeen > 120) { this.history.delete(name); continue; }
      out.push({ name, ms: h.v.reduce((a, b) => a + b, 0) / Math.max(1, h.v.length), samples: h.v.length });
    }
    return out;
  }

  destroy(): void {
    this.querySet?.destroy();
    this.resolveBuf?.destroy();
    for (const s of this.ring) s.buf.destroy();
  }
}

/** CPU frame-time rolling average (for fps). */
export class RollingAverage {
  private readonly v: number[] = [];
  private i = 0;
  constructor(private readonly n = TIMESTAMP_AVG_FRAMES) {}
  push(x: number): void { if (this.v.length < this.n) this.v.push(x); else this.v[this.i] = x; this.i = (this.i + 1) % this.n; }
  get value(): number { return this.v.length ? this.v.reduce((a, b) => a + b, 0) / this.v.length : 0; }
}
