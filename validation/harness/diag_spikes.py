# SCRATCH: analysis of diag-spikes.ts runs.  python diag_spikes.py <run> <geoRun> <cfg> [views...]
import json, sys, os
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
import diag_ff

RUN, GEO = sys.argv[1], sys.argv[2]
D = os.path.join(os.path.dirname(__file__), '..', 'out', RUN)
diag_ff.OUT = os.path.join(os.path.dirname(__file__), '..', 'out', GEO)
diag_ff.sc = json.load(open(os.path.join(diag_ff.OUT, 'scene.json')))

def load(tag):
    m = json.load(open(os.path.join(D, tag + '.json')))
    ev = np.fromfile(os.path.join(D, tag + '.ev'), dtype=np.uint32).reshape(-1, 8)
    st = np.fromfile(os.path.join(D, tag + '.st'), dtype=np.float32).reshape(m['h'], m['w'], 12)
    return m, ev, st

def summary(tag):
    m, ev, st = load(tag)
    w, h = m['w'], m['h']
    pos, pid, dist, tri = diag_ff.geometry(w, h)
    ref, _ = diag_ff.load('pt_ref'); Lr = diag_ff.lum(ref)
    hist = np.array(m['cnt'][8:73], dtype=np.float64)
    tot = hist.sum()
    # bin b <-> 2 log2 r in [b-32, b-31)
    def tail(r):  # P(ratio > r) approx via bins with lower edge >= 2 log2 r
        b0 = int(2 * np.log2(r)) + 32
        return hist[b0:64].sum() / tot
    N = m['frames']
    valid = (pid >= 0) & (Lr > 1e-3)
    edge = valid & (dist <= 6); inner = valid & (dist > 6)
    sumL, sumL2, sumLL1 = st[..., 4], st[..., 5], st[..., 6]
    mean = sumL / N; var = np.maximum(sumL2 / N - mean ** 2, 0)
    nB = st[..., 11]; mB = st[..., 9] / np.maximum(nB, 1); varB = np.maximum(st[..., 10] / np.maximum(nB, 1) - mB ** 2, 0)
    tau = 32 * varB / np.maximum(var, 1e-20)
    rho1 = ((sumLL1 / (N - 1)) - mean ** 2) / np.maximum(var, 1e-20)
    relvar = var / np.maximum(Lr, 1e-3) ** 2
    maxrun = st[..., 2]; onsets = st[..., 3]
    out = dict(tag=tag, frames=N, secs=round(m['secs'], 1), events=int(m['cnt'][0]),
               **{f'P(r>{r})': float(f'{tail(r):.3g}') for r in (4, 8, 16, 32, 64)})
    for nm, msk in (('edge', edge), ('inner', inner)):
        out[f'relvar_{nm}'] = round(float(np.mean(relvar[msk])), 4)
        out[f'tau32_{nm}'] = round(float(np.median(tau[msk])), 2)
        out[f'tau32mean_{nm}'] = round(float(np.mean(tau[msk])), 2)
        out[f'rho1_{nm}'] = round(float(np.median(rho1[msk])), 3)
        out[f'maxrun>=10_{nm}'] = float(f'{np.mean(maxrun[msk] >= 10):.3g}')
        out[f'maxrun>=20_{nm}'] = float(f'{np.mean(maxrun[msk] >= 20):.3g}')
        # effective variance of the N-frame mean relative to iid: relvar * tau
        out[f'relvar_x_tau_{nm}'] = round(float(np.mean(relvar[msk] * np.maximum(tau[msk], 0))), 4)
    return out

def events(cfg, views, kev=8):
    """Join events of the same deterministic run across views; return onset events with attributes."""
    base = None
    cols = {}
    for v in views:
        m, ev, st = load(f'{cfg}_v{v}')
        key = ev[:, 0].astype(np.int64) * (1 << 32) + ev[:, 1].astype(np.int64)
        order = np.argsort(key); key = key[order]; ev = ev[order]
        if base is None:
            base = (key, ev)
        else:
            if len(key) != len(base[0]) or not np.array_equal(key, base[0]):
                inter = np.intersect1d(key, base[0])
                print(f'WARN view {v}: {len(key)} events vs base {len(base[0])} (common {len(inter)})')
            idx = np.searchsorted(key, base[0]).clip(0, len(key) - 1)
            ok = key[idx] == base[0]
            col = np.where(ok[:, None], ev[idx, 4:8], 0xFFFFFFFF)
            cols[v] = col
    key, ev = base
    frame = ev[:, 0].astype(int); x = (ev[:, 1] & 0xFFFF).astype(int); y = (ev[:, 1] >> 16).astype(int)
    L = ev[:, 2].view(np.float32); Lr = ev[:, 3].view(np.float32)
    # onset = no event at the same pixel in frame-1
    pk = (ev[:, 1].astype(np.int64))
    s = set(zip(frame.tolist(), pk.tolist()))
    onset = np.array([(f - 1, p) not in s for f, p in zip(frame.tolist(), pk.tolist())])
    return dict(frame=frame, x=x, y=y, L=L, Lr=Lr, onset=onset, cols=cols)

if __name__ == '__main__':
    tags = sys.argv[3:]
    for t in tags:
        print(json.dumps(summary(t)))
