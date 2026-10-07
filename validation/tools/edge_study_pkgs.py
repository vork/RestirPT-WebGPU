"""Edge-study packages (docs/decisions/denoiser.md Changelog DN-16, DN-17), derived from (x) x_many_lights_512 into
validation/out/edge-aa/pkgs (not committed): xml_black (a camera on which the blue box's silhouette lies against the
black background, the back wall and the floor), xml_black_pan4 / _pan16 (that camera panned 4 / 16 mm per frame from
frame 16), xml_black_spot (spot light 78 moving +x 15 mm per frame from frame 16), and the material-id map of xml_black
at pixel centres (validation/out/edge-aa/xml_black_ids.npy, for edge_study.py --ids).

    validation/.venv/bin/python validation/tools/edge_study_pkgs.py
"""
import json, shutil, sys, numpy as np
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / 'validation/scenes/x_many_lights_512'
OUT = ROOT / 'validation/out/edge-aa/pkgs'

def look_at(eye, tgt, up=(0, 1, 0)):
    eye, tgt, up = map(np.asarray, (eye, tgt, up))
    f = tgt - eye; f = f / np.linalg.norm(f)
    r = np.cross(f, up); r /= np.linalg.norm(r)
    u = np.cross(r, f)
    m = np.eye(4); m[:3, 0] = r; m[:3, 1] = u; m[:3, 2] = -f; m[:3, 3] = eye
    return m.T.reshape(-1).tolist()   # column-major

def make(name, matrix, yfov=None, src=SRC):
    j = json.loads((src / 'scene.json').read_text())
    j['name'] = name
    j['camera']['matrix'] = matrix
    if yfov: j['camera']['yfov'] = yfov
    d = OUT / name; d.mkdir(parents=True, exist_ok=True)
    (d / 'scene.json').write_text(json.dumps(j))
    shutil.copy(src / 'geometry.bin', d / 'geometry.bin')
    return d

def preview(pkg, W=128):
    j = json.loads((pkg / 'scene.json').read_text())
    g = (pkg / 'geometry.bin').read_bytes()
    b = j['buffers']
    P = np.frombuffer(g, np.float32, b['positions']['length'] // 4, b['positions']['offset']).reshape(-1, 3)
    I = np.frombuffer(g, np.uint32, b['indices']['length'] // 4, b['indices']['offset']).reshape(-1, 3)
    M = np.frombuffer(g, np.uint32, b['triMaterial']['length'] // 4, b['triMaterial']['offset'])
    m = np.array(j['camera']['matrix']).reshape(4, 4).T
    t = np.tan(j['camera']['yfov'] / 2)
    ys, xs = np.mgrid[0:W, 0:W] + 0.5
    d = np.stack([(2 * xs / W - 1) * t, (1 - 2 * ys / W) * t, -np.ones_like(xs)], -1).reshape(-1, 3) @ m[:3, :3].T
    o = m[:3, 3]
    best = np.full(len(d), np.inf); mat = np.full(len(d), -1)
    v0, v1, v2 = P[I[:, 0]], P[I[:, 1]], P[I[:, 2]]
    for k in range(len(I)):
        e1, e2 = v1[k] - v0[k], v2[k] - v0[k]
        pv = np.cross(d, e2); det = pv @ e1
        ok = np.abs(det) > 1e-12
        inv = np.where(ok, 1 / np.where(ok, det, 1), 0)
        tv = o - v0[k]; u = (pv @ tv) * inv
        qv = np.cross(tv, e1); v = (d @ qv) * inv; tt = (qv @ e2) * inv
        hit = ok & (u >= 0) & (v >= 0) & (u + v <= 1) & (tt > 1e-4) & (tt < best)
        best[hit] = tt[hit]; mat[hit] = M[k]
    return mat.reshape(W, W)

def make_pan(name, base, dx, start, count=64, test=(40, 48, 56, 63)):
    """Sequence package: base camera translated along its right axis by dx metres per frame from `start` on."""
    src = OUT / base
    j = json.loads((src / 'scene.json').read_text())
    m0 = np.array(j['camera']['matrix'], dtype=float)
    frames = []
    for t in range(count):
        m = m0.copy(); k = max(0, t - start)
        m[12:15] += k * dx * m[0:3]
        frames.append({'frame': t, 'camera': {'matrix': m.tolist(), 'yfov': j['camera']['yfov']}})
    j['name'] = name; j['frames'] = frames
    j['sequence'] = {'fps': 24, 'frameCount': count, 'testFrames': list(test), 'notes': f'edge study: {base} panned {dx} m/frame from frame {start}'}
    d = OUT / name; d.mkdir(parents=True, exist_ok=True)
    (d / 'scene.json').write_text(json.dumps(j))
    shutil.copy(src / 'geometry.bin', d / 'geometry.bin')


def make_spot(name, base, light=78, dx=0.015, start=16, count=64, test=(40, 48, 56, 63)):
    src = OUT / base
    j = json.loads((src / 'scene.json').read_text())
    m0 = j['lights'][light]['matrix']
    frames = []
    for t in range(count):
        m = list(m0); m[12] += dx * max(0, t - start)
        frames.append({'frame': t, 'lights': {str(j['lights'][light]['id']): {'matrix': m}}})
    j['name'] = name; j['frames'] = frames
    j['sequence'] = {'fps': 24, 'frameCount': count, 'testFrames': list(test), 'notes': f'edge study: spot {light} moving +x {round(dx * 1000)} mm/frame from frame {start} (static camera)'}
    d = OUT / name; d.mkdir(parents=True, exist_ok=True)
    (d / 'scene.json').write_text(json.dumps(j))
    shutil.copy(src / 'geometry.bin', d / 'geometry.bin')


if __name__ == '__main__':
    d = make('xml_black', look_at([3.2, 0.5, -0.6], [-1.7, 0.8, -2.6]))
    np.save(ROOT / 'validation/out/edge-aa/xml_black_ids.npy', preview(d, 512).astype(np.int32))
    make_pan('xml_black_pan4', 'xml_black', 0.004, 16)
    make_pan('xml_black_pan16', 'xml_black', 0.016, 16)
    make_spot('xml_black_spot', 'xml_black')
