# SCRATCH firefly analysis for validation/out/<run>/<cfg>.f32 (diag-fireflies.ts).
import json, sys, os
import numpy as np
from scipy import ndimage
from PIL import Image

RUN = sys.argv[1] if __name__ == '__main__' else 'diag-ff'
OUT = os.path.join(os.path.dirname(__file__), '..', 'out', RUN)
sc = json.load(open(os.path.join(OUT, 'scene.json')))

def load(name):
    m = json.load(open(os.path.join(OUT, name + '.json')))
    a = np.fromfile(os.path.join(OUT, name + '.f32'), dtype=np.float32).reshape(m['h'], m['w'], 3)
    return a, m

def lum(a): return 0.2126 * a[..., 0] + 0.7152 * a[..., 1] + 0.0722 * a[..., 2]

def srgb(x):
    x = np.clip(x, 0, 1)
    return np.where(x <= 0.0031308, 12.92 * x, 1.055 * np.power(x, 1 / 2.4) - 0.055)

def save_png(a, name, scale=1.0):
    Image.fromarray((srgb(a * scale) * 255 + 0.5).astype(np.uint8)).save(os.path.join(OUT, name + '.png'))

_geo = {}
def geometry(w, h):
    if (w, h) in _geo: return _geo[(w, h)]
    P = np.array(sc['positions'], dtype=np.float64).reshape(-1, 3)
    I = np.array(sc['indices']).reshape(-1, 3)
    M = np.array(sc['camera']['camToWorld']).reshape(4, 4).T  # column-major -> row-major
    yf = sc['camera']['yfov']
    ys, xs = np.mgrid[0:h, 0:w] + 0.5
    t = np.tan(yf / 2)
    dx = (2 * xs / w - 1) * t * w / h
    dy = (1 - 2 * ys / h) * t
    d = np.stack([dx, dy, -np.ones_like(dx)], -1) @ M[:3, :3].T
    d /= np.linalg.norm(d, axis=-1, keepdims=True)
    o = M[:3, 3]
    best = np.full((h, w), np.inf); tri = np.full((h, w), -1)
    for k, (i0, i1, i2) in enumerate(I):
        v0, v1, v2 = P[i0], P[i1], P[i2]
        e1, e2 = v1 - v0, v2 - v0
        pv = np.cross(d, e2); det = pv @ e1
        with np.errstate(divide='ignore', invalid='ignore'):
            inv = 1 / det
            tv = o - v0
            u = (pv @ tv) * inv
            qv = np.cross(tv, e1)
            v = (d @ qv) * inv
            tt = (qv @ e2) * inv
        ok = (np.abs(det) > 1e-12) & (u >= 0) & (v >= 0) & (u + v <= 1) & (tt > 1e-6) & (tt < best)
        best[ok] = tt[ok]; tri[ok] = k
    pos = o + d * best[..., None]
    # plane id: triangles of one quad share a normal+offset -> group by (normal, d)
    N = np.cross(P[I[:, 1]] - P[I[:, 0]], P[I[:, 2]] - P[I[:, 0]]); N /= np.linalg.norm(N, axis=1, keepdims=True)
    off = np.einsum('ij,ij->i', N, P[I[:, 0]])
    key = {}
    plane = np.zeros(len(I), int)
    for k in range(len(I)):
        kk = (tuple(np.round(N[k], 3)), round(off[k], 4), sc['triMaterial'][k])
        plane[k] = key.setdefault(kk, len(key))
    pid = np.where(tri >= 0, plane[np.maximum(tri, 0)], -1)
    edge = np.zeros((h, w), bool)
    edge[:, 1:] |= pid[:, 1:] != pid[:, :-1]; edge[:, :-1] |= pid[:, 1:] != pid[:, :-1]
    edge[1:, :] |= pid[1:, :] != pid[:-1, :]; edge[:-1, :] |= pid[1:, :] != pid[:-1, :]
    dist = ndimage.distance_transform_edt(~edge)
    _geo[(w, h)] = (pos, pid, dist, tri)
    return _geo[(w, h)]

def fireflies(a, k=4.0, absmin=0.02, size=7):
    L = lum(a)
    med = ndimage.median_filter(L, size=size, mode='nearest')
    ff = (L > k * med) & (L - med > absmin)
    return ff, L, med

def ref_metrics(a, ref, dist, pid):
    L, Lr = lum(a), lum(ref)
    valid = (pid >= 0) & (Lr > 1e-3)
    rel = np.where(valid, (L - Lr) / np.maximum(Lr, 1e-3), 0)
    out = {}
    for k in (0.25, 0.5, 1.0):
        m = valid & (rel > k)
        out[f'n_rel>{k}'] = int(m.sum())
        if k == 0.5:
            d = dist[m]
            out['ff0.5_edge<=3px'] = round(float((d <= 3).mean()), 3) if m.sum() else None
            out['ff0.5_edge<=10px'] = round(float((d <= 10).mean()), 3) if m.sum() else None
    out['p99.9_rel'] = round(float(np.percentile(np.abs(rel[valid]), 99.9)), 4)
    out['p99_rel'] = round(float(np.percentile(np.abs(rel[valid]), 99)), 4)
    out['rmse_rel'] = round(float(np.sqrt(np.mean(rel[valid] ** 2))), 5)
    band = valid & (dist <= 6); inner = valid & (dist > 6)
    out['rmse_rel_edge6'] = round(float(np.sqrt(np.mean(rel[band] ** 2))), 5)
    out['rmse_rel_inner'] = round(float(np.sqrt(np.mean(rel[inner] ** 2))), 5)
    out['bias_mean'] = round(float(L[valid].mean() / Lr[valid].mean() - 1), 5)
    return out, rel

if __name__ == '__main__':
    names = sys.argv[2:]
    ref, _ = load('pt_ref')
    rows = []
    for n in names:
        a, m = load(n)
        h, w = a.shape[:2]
        pos, pid, dist, tri = geometry(w, h)
        save_png(a, n)
        r = dict(cfg=n, frames=m['frames'], secs=round(m['secs'], 1))
        mm, rel = ref_metrics(a, ref, dist, pid)
        r.update(mm)
        rows.append(r)
        print(json.dumps(r))
        vis = srgb(a).copy(); vis[rel > 0.5] = [1, 0, 1]
        Image.fromarray((vis * 255).astype(np.uint8)).save(os.path.join(OUT, n + '_ffmask.png'))
        np.save(os.path.join(OUT, n + '_rel.npy'), rel.astype(np.float32))
    json.dump(rows, open(os.path.join(OUT, 'summary.json'), 'w'), indent=1)
