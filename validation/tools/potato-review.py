"""Analyze fixed-resolution Potato captures and create a local image/video review.

Run: validation/.venv/bin/python validation/tools/potato-review.py
Inputs: validation/out/potato/{final,screen,screen2,quality-final-jobs}.json,
12 potato-*-final-s* captures, prior matched perf4-*-current-s* captures and the
existing 12,288 spp reference. Captures/job manifests are described in perf2 §9.
"""
from pathlib import Path
import json
import subprocess
import numpy as np
from PIL import Image, ImageDraw
import flip_evaluator as flip
from imageio_util import read_pfm
from denoise_eval import srgb_oetf, relmse, flip_version

ROOT = Path(__file__).resolve().parents[2]
CAP = ROOT / 'validation/out'
OUT = CAP / 'potato'
MOTIONS = ['static', 'pan', 'light']
FRAMES = [16, 32, 48, 63]
ref = read_pfm(CAP / 'wpq-ref-sponza/ref.pfm')
rd = srgb_oetf(ref).astype(np.float32)
rows, meta = [], []

def png(path, linear):
    image = Image.fromarray(np.uint8(np.clip(srgb_oetf(linear) * 255 + .5, 0, 255)))
    image.save(path)
    return image

for motion in MOTIONS:
    for seed in [1, 2, 3, 4]:
        for mode in ['interactive', 'potato']:
            folder = CAP / (f'perf4-{motion}-current-s{seed}' if mode == 'interactive' else f'potato-{motion}-final-s{seed}')
            m = json.loads((folder / 'meta.json').read_text())
            assert m['ok'] and not m['errors'], (folder, m['errors'])
            assert m['width'] == 960 and m['height'] == 540 and m['frames'] == 64
            if mode == 'interactive':
                control = m
            else:
                assert m['renderer']['restirMode'] == 'potato'
                assert m['renderer']['settings']['maxBounces'] == 1
                assert not m['renderer']['settings']['temporal']
                assert m['denoiser']['settings']['iterations'] == 3
                for key in ['packageSha256', 'seed', 'width', 'height', 'frames', 'panOsc', 'lightOsc', 'jitter', 'accumulate']:
                    assert m.get(key) == control.get(key), (folder, key)
            meta.append(dict(motion=motion, seed=seed, mode=mode, renderer=m['renderer'], denoiser=m['denoiser']['settings'],
                             adapter=m['adapterInfo'], chrome=m['chromeVersion'], packageSha256=m['packageSha256']))
            for frame in FRAMES:
                for kind in ['raw', 'dn']:
                    a = read_pfm(folder / f'{kind}_f{frame}.pfm')
                    assert np.isfinite(a).all(), (folder, frame, kind)
                    _, score, _ = flip.evaluate(rd, srgb_oetf(a).astype(np.float32), 'LDR', inputsRGB=True, applyMagma=False)
                    rows.append(dict(motion=motion, seed=seed, mode=mode, frame=frame, kind=kind,
                                     flip=float(score), relmse=relmse(a, ref), luminance=float(np.mean(a @ [.2126,.7152,.0722]))))
            if seed == 1:
                for frame in list(range(0,64,4)) + [63]:
                    png(OUT / f'{motion}-{mode}-f{frame}.png', read_pfm(folder / f'dn_f{frame}.pfm'))
    print('ANALYSED', motion, flush=True)
summary = []
for motion in MOTIONS:
    for mode in ['interactive', 'potato']:
        for kind in ['raw', 'dn']:
            group = [r for r in rows if r['motion'] == motion and r['mode'] == mode and r['kind'] == kind]
            summary.append(dict(motion=motion, mode=mode, kind=kind, images=len(group),
                                **{k:float(np.mean([r[k] for r in group])) for k in ['flip','relmse','luminance']}))
perf = json.loads((OUT / 'final.json').read_text())
assert all(r['ok'] for r in perf['reports'])
timings = []
for height in [540,720,1080]:
    for mode in ['interactive','potato']:
        reports = [r for r in perf['reports'] if r['label'].startswith(f'{height}p-{mode}-') and r['label'].split('-')[-1].isdigit()]
        ms = float(np.mean([r['frame']['meanMs'] for r in reports]))
        timings.append(dict(height=height, mode=mode, ms=ms, fps=1000/ms, runs=len(reports)))
report = dict(reference='validation/out/wpq-ref-sponza/ref.pfm', reference_spp=12288, reference_relmse_noise_floor=.000322,
              display='exposure 1, clamp and sRGB; identical across modes; rgba32float captures vs rgba16float perf',
              frames=FRAMES, seeds=[1,2,3,4], flip_version=flip_version(), timings=timings, quality=summary, rows=rows, capture_metadata=meta,
              performance=perf, screening=[json.loads((OUT/f'{name}.json').read_text()) for name in ['screen','screen2']],
              manifests={name:json.loads((OUT/f'{name}.json').read_text()) for name in ['final-jobs','quality-final-jobs','screen-jobs','screen2-jobs','quality-screen-jobs']})
(OUT / 'results.json').write_text(json.dumps(report,indent=1))
(ROOT / 'docs/decisions/potato-results.json').write_text(json.dumps(report,indent=1))

for motion in MOTIONS:
    for i, frame in enumerate(list(range(0,64,4)) + [63]):
        sheet = Image.new('RGB',(1920,586),'#111b26'); d = ImageDraw.Draw(sheet)
        for col,mode in enumerate(['interactive','potato']):
            d.text((col*960+12,12), ('Interactive ReSTIR' if mode=='interactive' else 'Potato ReSTIR')+f' | {motion}, frame {frame}',fill='white')
            sheet.paste(Image.open(OUT/f'{motion}-{mode}-f{frame}.png'),(col*960,46))
        sheet.save(OUT / f'{motion}-pair-{i:03}.png')
        if frame==63: sheet.save(OUT/f'{motion}-comparison.png')
    if motion != 'static':
        subprocess.run(['/opt/homebrew/bin/ffmpeg','-y','-loglevel','error','-framerate','6','-i',str(OUT/f'{motion}-pair-%03d.png'),
                        '-vf','fps=24','-c:v','libx264','-crf','18','-pix_fmt','yuv420p',str(OUT/f'{motion}.mp4')],check=True)
table=''.join(f'<tr><td>{t["height"]}p</td><td>{t["mode"]}</td><td>{t["ms"]:.2f} ms</td><td>{t["fps"]:.1f}</td></tr>' for t in timings)
(OUT/'review.html').write_text('''<!doctype html><meta charset="utf-8"><title>Potato ReSTIR comparison</title>
<style>body{background:#111b26;color:#eef3f8;font:16px system-ui;margin:32px auto;max-width:1500px}button,input,select{font:inherit;margin:8px}img,video{width:100%;display:block}.pair{display:grid;grid-template-columns:1fr 1fr;gap:8px}td,th{padding:6px 20px;text-align:left}p{max-width:1100px;color:#bfcbd8}</style>
<h1>Potato ReSTIR · fixed resolution</h1><p>Matched 960×540 renders, same camera, materials, lighting, primary sampling and exposure. Potato uses shorter paths, less reuse and one fewer denoising pass. Darker indirect light and more motion noise are expected.</p>
<table><tr><th>Sponza</th><th>Mode</th><th>Frame time</th><th>Equivalent FPS</th></tr>'''+table+'''
</table><p>GPU harness throughput on this machine; excludes application UI/display pacing. Four-way ABBA at 540p/720p, one paired run at 1080p.</p>
<label>Sequence <select id="motion"><option>static</option><option>pan</option><option>light</option></select></label>
<label>Frame <input id="frame" type="range" min="0" max="16" value="16"><output id="frameText"></output></label>
<div class="pair"><div><h2>Interactive ReSTIR</h2><img id="current"></div><div><h2>Potato ReSTIR</h2><img id="potato"></div></div>
<p>Four-seed reference errors and exact settings are in results.json. The videos show rendered sequences at illustrative playback speed, not a real-time FPS recording.</p>
<h2>Camera motion</h2><video controls loop muted src="pan.mp4"></video><h2>Moving light</h2><video controls loop muted src="light.mp4"></video>
<script>const frames=[0,4,8,12,16,20,24,28,32,36,40,44,48,52,56,60,63];function show(){const m=document.querySelector('#motion').value,f=frames[+document.querySelector('#frame').value];document.querySelector('#frameText').textContent=f;document.querySelector('#current').src=m+'-interactive-f'+f+'.png';document.querySelector('#potato').src=m+'-potato-f'+f+'.png'}document.querySelectorAll('select,input').forEach(e=>e.oninput=show);show()</script>''')
print(json.dumps({'timings':timings,'quality':[s for s in summary if s['kind']=='dn']},indent=2))
