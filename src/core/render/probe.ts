// Probe pixel readback (plan §6): shaders append {pixel, tag, seq, vec4f} records via probe_record() when
// pixel == probePixel; each frame the DebugBuffer header (counters + records) is copied into a 3-deep ring of
// MAP_READ buffers and parsed asynchronously. The same header carries the per-frame NaN/Inf/overflow counters.
import { DBGC, DEBUG_BUFFER_LAYOUT } from './debug-views.ts';

export interface ProbeRecord {
  pixel: [number, number];
  tag: number;
  seq: number;
  value: [number, number, number, number];
  /** Raw bits of value (for code views stored with bitcast). */
  bits: [number, number, number, number];
}

export interface ProbeFrame {
  frame: number;
  counters: Uint32Array;
  records: ProbeRecord[];
}

/** Tag id -> name. 0..15 framework, 16+ passes (keep in sync with PROBE_TAG_* in debug-common.wgsl). */
export const probeTags = new Map<number, string>([[1, 'aov']]);
/** Last registration wins (the app's test pattern registers 16/17 and a real renderer replaces them). */
export function registerProbeTag(id: number, name: string): void { probeTags.set(id, name); }
export const probeTagName = (id: number): string => probeTags.get(id) ?? `0x${id.toString(16)}`;

/** CSS click position -> internal-resolution pixel (row 0 = top). The canvas shows the whole internal image. */
export function pickPixel(x: number, y: number, cssW: number, cssH: number, w: number, h: number): [number, number] {
  const c = Math.floor((x / Math.max(cssW, 1e-9)) * w);
  const r = Math.floor((y / Math.max(cssH, 1e-9)) * h);
  return [Math.min(w - 1, Math.max(0, c)), Math.min(h - 1, Math.max(0, r))];
}

/** Parse a DebugBuffer header copy. */
export function parseDebugHeader(buf: ArrayBuffer, frame: number): ProbeFrame {
  const L = DEBUG_BUFFER_LAYOUT;
  const u32 = new Uint32Array(buf, 0, L.headerBytes / 4);
  const f32 = new Float32Array(buf, 0, L.headerBytes / 4);
  const counters = u32.slice(0, L.counterCount);
  const n = Math.min(counters[DBGC.PROBE_COUNT], L.probeCapacity);
  const records: ProbeRecord[] = [];
  for (let i = 0; i < n; i++) {
    const b = (L.probeOffset + i * L.probeStride) / 4;
    records.push({
      pixel: [u32[b], u32[b + 1]],
      tag: u32[b + 2],
      seq: u32[b + 3],
      value: [f32[b + 4], f32[b + 5], f32[b + 6], f32[b + 7]],
      bits: [u32[b + 4], u32[b + 5], u32[b + 6], u32[b + 7]],
    });
  }
  records.sort((a, b) => a.seq - b.seq);
  return { frame, counters, records };
}

type Slot = { buf: GPUBuffer; state: 'free' | 'pending' | 'mapping'; frame: number };

export class ProbeRing {
  private readonly ring: Slot[] = [];
  private current: Slot | undefined;
  latest: ProbeFrame | undefined;
  readonly listeners = new Set<(f: ProbeFrame) => void>();
  /** Frames whose counters were never read back (every slot in flight at maxDepth): their NaN/Inf/overflow
   *  counts are lost, so gates must require skipped == 0. */
  skipped = 0;

  constructor(private readonly device: GPUDevice, depth = 3, private readonly maxDepth = 8) {
    for (let i = 0; i < depth; i++) this.addSlot();
  }

  private addSlot(): Slot {
    const s: Slot = {
      buf: this.device.createBuffer({ label: `probe-read${this.ring.length}`, size: DEBUG_BUFFER_LAYOUT.headerBytes, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST }),
      state: 'free',
      frame: 0,
    };
    this.ring.push(s);
    return s;
  }

  /** Copy the header of `debugBuffer` at the end of the frame. Grows the ring up to maxDepth when every slot is in
   *  flight (the counters are cleared each frame, so a skipped copy loses them); skips (no stall) beyond that. */
  encodeCopy(encoder: GPUCommandEncoder, debugBuffer: GPUBuffer, frame: number): void {
    const s = this.ring.find((x) => x.state === 'free') ?? (this.ring.length < this.maxDepth ? this.addSlot() : undefined);
    if (!s) { this.skipped++; this.current = undefined; return; }
    encoder.copyBufferToBuffer(debugBuffer, 0, s.buf, 0, DEBUG_BUFFER_LAYOUT.headerBytes);
    s.state = 'pending';
    s.frame = frame;
    this.current = s;
  }

  afterSubmit(): void {
    const s = this.current;
    this.current = undefined;
    if (!s) return;
    s.state = 'mapping';
    s.buf.mapAsync(GPUMapMode.READ).then(() => {
      const copy = s.buf.getMappedRange().slice(0);
      s.buf.unmap();
      s.state = 'free';
      const f = parseDebugHeader(copy, s.frame);
      if (!this.latest || f.frame >= this.latest.frame) this.latest = f;
      for (const cb of this.listeners) cb(f); // every frame's counters, even if it resolves out of order
    }).catch(() => { s.state = 'free'; });
  }

  destroy(): void { for (const s of this.ring) s.buf.destroy(); }
}
