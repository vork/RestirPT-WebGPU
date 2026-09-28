// Common dump schema written by pxr_dump.py (reference) and spike.ts (LightUSD backends, three.js).
// Matrices: 16 numbers, USD row-major / row-vector layout (translation at 12..14) = column-major for column vectors.

export type Mat16 = number[];

export interface ShapingDump {
  coneAngle: number | null; // degrees (UsdLux)
  coneSoftness: number | null;
  focus: number | null;
  focusTint: number[] | null;
}

export interface LightDump {
  path: string;
  type: string; // sphere | rect | disk | distant | cylinder | dome | three:<class>
  intensity: number | null;
  exposure: number | null;
  color: number[] | null;
  normalize: boolean | null;
  enableColorTemperature: boolean | null;
  colorTemperature: number | null;
  radius: number | null;
  width: number | null;
  height: number | null;
  angle: number | null; // DistantLight angular diameter, degrees
  treatAsPoint: boolean | null;
  shaping: ShapingDump | null;
  world: Mat16 | null;
}

export interface DrawDump {
  path: string;
  points: number;
  triangles: number;
  material: string | null;
  trianglesByMaterial: Record<string, number>;
  world: Mat16 | null;
  bbox?: number[] | null; // prim-local points bounds [minx,miny,minz,maxx,maxy,maxz]
}

export interface MaterialDump {
  path: string;
  inputs: Record<string, unknown>;
}

export interface InstancerDump {
  path: string;
  world?: Mat16 | null; // instancer local-to-world (pxr only)
  prototypes: string[];
  instances: { proto: string | null; world: Mat16 | null }[];
}

export interface CameraDump {
  path: string;
  focalLength: number | null;
  horizontalAperture: number | null;
  verticalAperture: number | null;
  world: Mat16 | null;
}

export interface SceneDump {
  source: string; // pxr | lightusd-next | lightusd-legacy | three
  file: string;
  ok: boolean;
  error?: string;
  stage: { upAxis: string | null; metersPerUnit: number | null; doc?: string | null };
  draws: DrawDump[];
  lights: LightDump[];
  materials: MaterialDump[];
  pointInstancers: InstancerDump[];
  cameras: CameraDump[];
  timings?: Record<string, number>;
  counts?: Record<string, number>;
  notes?: string[];
  raw?: Record<string, unknown>;
}
