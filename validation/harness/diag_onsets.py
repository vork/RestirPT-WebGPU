import sys, os, json
import numpy as np
sys.argv = [sys.argv[0], 'diag-sp', 'diag-ff'] + sys.argv[1:]
sys.path.insert(0, os.path.dirname(__file__))
import diag_spikes as S, diag_ff
cfg = sys.argv[3]; views = [int(v) for v in sys.argv[4].split(',')]
E = S.events(cfg, [0] + views)
fr = E['frame']; on = E['onset']; ratio = E['L'] / E['Lr']
pos, pid, dist, tri = diag_ff.geometry(704, 540)
d = dist[E['y'], E['x']]
warm = fr >= 30
print('events', len(fr), 'onsets', on.sum(), 'onsets after warmup', (on & warm).sum())
print('event ratio pctl 50/90/99:', np.percentile(ratio[warm], [50, 90, 99]).round(1))
print('edge<=6px share: onsets', (d[on & warm] <= 6).mean().round(3), 'all events', (d[warm] <= 6).mean().round(3), 'image base', (dist[pid >= 0] <= 6).mean().round(3))
# run lengths: for each onset, count consecutive frames at same pixel
key = set(zip(fr.tolist(), (E['x'] + 65536 * E['y']).tolist()))
runs = []
for f, x, y in zip(fr[on & warm], E['x'][on & warm], E['y'][on & warm]):
    k = x + 65536 * y; n = 1
    while (f + n, k) in key: n += 1
    runs.append(n)
runs = np.array(runs)
print('run length of >8x episodes: mean', runs.mean().round(2), 'pctl 50/90/99/max', np.percentile(runs, [50, 90, 99, 100]))
def col(v, kind):
    c = E['cols'][v][:, 0]
    return c.view(np.float32) if kind == 'f' else c
names = {492: ('sel', 'u'), 486: ('log2J', 'f'), 490: ('cprev', 'f'), 491: ('cout', 'f'), 495: ('wpfrac', 'f'), 404: ('d', 'u'), 405: ('k', 'u'),
         406: ('tech', 'u'), 407: ('ep', 'u'), 410: ('kMargin', 'f'), 401: ('W', 'f'), 488: ('pic', 'f'), 489: ('pip', 'f'), 485: ('invCode', 'u'), 493: ('phatRel', 'f')}
m = on & warm
# control: all pixel-frames are not available, so compare onsets vs non-onset (continuing) events
for v in views:
    nm, kind = names[v]
    c = col(v, kind)
    if kind == 'u':
        vals, cnt = np.unique(c[m], return_counts=True)
        print(f'{nm} onset:', dict(zip(vals.tolist(), (cnt / m.sum()).round(3).tolist())))
        vals, cnt = np.unique(c[warm & ~on], return_counts=True)
        print(f'{nm} continuing:', dict(zip(vals.tolist(), (cnt / (warm & ~on).sum()).round(3).tolist())))
    else:
        x = c[m]; x = x[np.isfinite(x)]
        print(f'{nm} onset pctl 1/10/50/90/99:', np.percentile(x, [1, 10, 50, 90, 99]).round(3))
        x = c[warm & ~on]; x = x[np.isfinite(x)]
        print(f'{nm} continuing pctl 1/10/50/90/99:', np.percentile(x, [1, 10, 50, 90, 99]).round(3))
np.savez(os.path.join(S.D, f'{cfg}_joined.npz'), **{k: v for k, v in E.items() if k != 'cols'}, **{f'c{v}': E['cols'][v] for v in views})
