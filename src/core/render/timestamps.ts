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
  /** Latest per-frame GPU span (ms, first timed begin to last timed end; see attributeFrame), and number of frames
   *  skipped because every ring slot was busy. */
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
      const r = attributeFrame(names.map((name, i) => ({ name, begin: t[2 * i], end: t[2 * i + 1] })));
      for (const [name, ms] of r.perName) this.push(name, ms, frame);
      this.push('untimed', r.untimedMs, frame);
      this.push('total', r.spanMs, frame);
      this.lastTotalMs = r.spanMs;
      this.lastRaw = r.raw;
    }).catch(() => { s.state = 'free'; });
  }

  /** The last resolved frame's raw pairs (ms relative to the earliest valid begin), for diagnostics. */
  lastRaw: { name: string; begin: number; end: number }[] = [];

  private push(name: string, ms: number, frame: number): void {
    let h = this.history.get(name);
    if (!h) { h = { v: [], i: 0, lastSeen: frame }; this.history.set(name, h); }
    if (h.v.length < TIMESTAMP_AVG_FRAMES) h.v.push(ms); else h.v[h.i] = ms;
    h.i = (h.i + 1) % TIMESTAMP_AVG_FRAMES;
    h.lastSeen = frame;
  }

  /** Rolling averages (ms) over the last ≤32 measured frames; passes unseen for 30 frames are dropped (a mode switch
   *  leaves no stale line that is not part of the current frame's span). */
  averages(): { name: string; ms: number; samples: number }[] {
    const out: { name: string; ms: number; samples: number }[] = [];
    for (const [name, h] of this.history) {
      if (this.frame - h.lastSeen > 30) { this.history.delete(name); continue; }
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

/**
 * Attribute one frame's timestamp pairs (one command buffer, passes executed in order). Two effects make the raw
 * begin–end pairs misleading:
 * - Q3 (platform-lanes.md): the ReSTIR passes (and the in-frame denoiser) carry no timestampWrites, so their GPU time
 *   is invisible to the per-pass pairs;
 * - on Metal a render pass's begin sample is taken when its vertex stage starts, which may overlap the preceding
 *   compute work (the present blit's begin landed before the ReSTIR passes, so "present" read the whole untimed span).
 * Each pass therefore starts no earlier than the end of the timed pass that finished before it (pairs ordered by end),
 * and `untimedMs` = span − Σ passes, where span = the first timed begin to the last timed end. Ties on the end
 * (timestamps are quantised) keep the reservation order, which callers make the encode order. Pairs that were never
 * written (0) or are inverted are dropped; a reserved pair whose pass is not encoded resolves to stale values, so
 * reserve only passes that are encoded.
 */
export function attributeFrame(pairs: { name: string; begin: bigint; end: bigint }[]): {
  perName: Map<string, number>; spanMs: number; untimedMs: number; raw: { name: string; begin: number; end: number }[];
} {
  const valid = pairs.filter((p) => p.begin > 0n && p.end >= p.begin && Number(p.end - p.begin) < 1e10)
    .sort((a, b) => (a.end < b.end ? -1 : a.end > b.end ? 1 : 0));   // stable: ties keep the reservation order
  const perName = new Map<string, number>();
  for (const p of pairs) perName.set(p.name, 0);
  if (!valid.length) return { perName, spanMs: 0, untimedMs: 0, raw: [] };
  const t0 = valid.reduce((m, p) => (p.begin < m ? p.begin : m), valid[0].begin);
  const start = valid[0].begin;
  let prevEnd = start, sum = 0;
  for (const p of valid) {
    const b = p.begin > prevEnd ? p.begin : prevEnd;
    const ms = Number(p.end - b) / 1e6;
    perName.set(p.name, (perName.get(p.name) ?? 0) + ms);
    sum += ms;
    prevEnd = p.end;
  }
  const spanMs = Number(prevEnd - start) / 1e6;
  return {
    perName, spanMs, untimedMs: Math.max(0, spanMs - sum),
    raw: valid.map((p) => ({ name: p.name, begin: Number(p.begin - t0) / 1e6, end: Number(p.end - t0) / 1e6 })),
  };
}

/** CPU frame-time rolling average (for fps). */
export class RollingAverage {
  private readonly v: number[] = [];
  private i = 0;
  constructor(private readonly n = TIMESTAMP_AVG_FRAMES) {}
  push(x: number): void { if (this.v.length < this.n) this.v.push(x); else this.v[this.i] = x; this.i = (this.i + 1) % this.n; }
  get value(): number { return this.v.length ? this.v.reduce((a, b) => a + b, 0) / this.v.length : 0; }
}
