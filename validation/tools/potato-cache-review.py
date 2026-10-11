"""Analyze matched Potato/cached-Potato captures and build image/video review artifacts.

Run: validation/.venv/bin/python validation/tools/potato-cache-review.py
Inputs: validation/out/potato2/{final,final-jobs,final-quality-jobs}.json and
potato{,2}-{static,pan,light}-final-s{1,2,3,4}. No GPU work is performed here.
"""
from pathlib import Path
import json
import subprocess
import numpy as np
from PIL import Image, ImageDraw
import flip_evaluator as flip
from imageio_util import read_pfm
from denoise_eval import srgb_oetf, flip_version

ROOT = Path(__file__).resolve().parents[2]
CAP = ROOT / 'validation/out'
OUT = CAP / 'potato2'
FRAMES = [16, 32, 48, 63]
MOTIONS = ['static', 'pan', 'light']
ref = srgb_oetf(read_pfm(CAP / 'wpq-ref-sponza/ref.pfm')).astype(np.float32)
rows, metadata, smoothness = [], [], []
for motion in MOTIONS:
    for label, prefix in [('old', 'potato'), ('cache', 'potato2')]:
        frames = {f: [] for f in FRAMES}
        for seed in [1, 2, 3, 4]:
            folder = CAP / f'{prefix}-{motion}-final-s{seed}'
            m = json.loads((folder / 'meta.json').read_text())
            assert m['ok'] and not m['errors'], (folder, m['errors'])
            assert m['width'] == 960 and m['height'] == 540 and m['frames'] == 64
            if label == 'old':
                assert m['renderer']['settings'].get('indirectCache', 0) == 0
            else:
                assert m['renderer']['settings']['indirectCache'] == 2
                assert m['denoiser']['settings']['iterations'] == 2
                old = json.loads((CAP / f'potato-{motion}-final-s{seed}' / 'meta.json').read_text())
                for key in ['packageSha256', 'seed', 'width', 'height', 'frames', 'panOsc', 'lightOsc', 'jitter', 'accumulate']:
                    assert m.get(key) == old.get(key), (folder, key)
            metadata.append(dict(mode=label, motion=motion, seed=seed, renderer=m['renderer'],
                                 denoiser=m['denoiser']['settings'], adapter=m['adapterInfo'], chrome=m['chromeVersion'],
                                 packageSha256=m['packageSha256'], rawSignal=m.get('rawSignal', 'full path estimate')))
            for f in FRAMES:
                a = read_pfm(folder / f'dn_f{f}.pfm')
                assert np.isfinite(a).all(), (folder, f)
                display = srgb_oetf(a).astype(np.float32)
                frames[f].append(display)
                _, score, _ = flip.evaluate(ref, display, 'LDR', inputsRGB=True, applyMagma=False)
                rows.append(dict(mode=label, motion=motion, seed=seed, frame=f, flip=float(score),
                                 luminance=float(np.mean(a @ [.2126, .7152, .0722]))))
            if seed == 1:
                for f in list(range(0, 64, 4)) + [63]:
                    a = read_pfm(folder / f'dn_f{f}.pfm')
                    Image.fromarray(np.uint8(np.clip(srgb_oetf(a)*255+.5, 0, 255))).save(OUT / f'{motion}-{label}-f{f}.png')
        for f, images in frames.items():
            std = np.std(np.stack(images), axis=0, ddof=1)
            smoothness.append(dict(mode=label, motion=motion, frame=f, seeds=4,
                                   displayStd=float(std.mean()), darkInteriorStd=float(std[80:480, 0:250].mean())))
    print('ANALYZED', motion, flush=True)
perf = json.loads((OUT / 'final.json').read_text())
assert all(r['ok'] for r in perf['reports'])
timings = []
for height in [540, 720, 1080]:
    for mode in ['old', 'cache']:
        reports = [r for r in perf['reports'] if r['label'].startswith(f'{height}p-{mode}-') and r['label'].split('-')[-1].isdigit()]
        ms = float(np.mean([r['frame']['meanMs'] for r in reports]))
        timings.append(dict(height=height, mode=mode, ms=ms, fps=1000/ms, runs=len(reports)))
summary = []
for motion in MOTIONS:
    for mode in ['old', 'cache']:
        group = [x for x in smoothness if x['mode'] == mode and x['motion'] == motion]
        quality = [x for x in rows if x['mode'] == mode and x['motion'] == motion]
        summary.append(dict(mode=mode, motion=motion,
                            **{k: float(np.mean([r[k] for r in group])) for k in ['displayStd', 'darkInteriorStd']},
                            flip=float(np.mean([r['flip'] for r in quality]))))
report = dict(timings=timings, summary=summary, smoothness=smoothness, quality=rows, capture_metadata=metadata,
              performance=perf, flip_version=flip_version(), reference_spp=12288,
              notes=['GPU harness throughput, not display FPS; Apple M5 Pro, Chrome/Metal.',
                     'Fixed resolution. Sparse indirect sampling, full-resolution primary geometry and direct lighting.',
                     'Display std is variability across four matched seeds, including jitter; lower does not prove accuracy or absence of lag.',
                     'Dark interior ROI: x=0..249, y=80..479, at 960x540.',
                     'FLIP is against the 12288-spp long-path reference at oscillation knots; accuracy is secondary for this explicitly biased mode.',
                     'Cached raw captures contain direct lighting/emission only; compare dn outputs for complete images.',
                     'Quality captures use rgba32float, performance uses rgba16float.'],
              manifests={name: json.loads((OUT/f'{name}.json').read_text()) for name in ['final-jobs', 'final-quality-jobs']})
(OUT / 'results.json').write_text(json.dumps(report, indent=1))
(ROOT / 'docs/decisions/potato-cache-results.json').write_text(json.dumps(report, indent=1))
for motion in MOTIONS:
    for i, f in enumerate(list(range(0, 64, 4)) + [63]):
        sheet = Image.new('RGB', (1920, 580), '#111b26'); d = ImageDraw.Draw(sheet)
        for col, mode in enumerate(['old', 'cache']):
            d.text((col*960+12, 12), ('Previous Potato' if mode == 'old' else 'Experimental Potato: direct ReSTIR + cached indirect') + f' | {motion}, frame {f}', fill='white')
            sheet.paste(Image.open(OUT/f'{motion}-{mode}-f{f}.png'), (col*960, 40))
        sheet.save(OUT/f'{motion}-pair-{i:03}.png')
        if f == 63: sheet.save(OUT/f'{motion}-comparison.png')
    if motion != 'static':
        subprocess.run(['/opt/homebrew/bin/ffmpeg', '-y', '-loglevel', 'error', '-framerate', '6', '-i', str(OUT/f'{motion}-pair-%03d.png'),
                        '-vf', 'fps=24', '-c:v', 'libx264', '-crf', '18', '-pix_fmt', 'yuv420p', str(OUT/f'{motion}.mp4')], check=True)
table = ''.join(f'<tr><td>{t["height"]}p</td><td>{t["mode"]}</td><td>{t["ms"]:.2f} ms</td><td>{t["fps"]:.1f}</td></tr>' for t in timings)
(OUT/'review.html').write_text('''<!doctype html><meta charset="utf-8"><title>Potato: direct ReSTIR and cached indirect</title>
<style>body{background:#111b26;color:#eef3f8;font:16px system-ui;margin:32px auto;max-width:1500px}input,select{font:inherit;margin:8px}img,video{width:100%;display:block}.pair{display:grid;grid-template-columns:1fr 1fr;gap:8px}td,th{padding:6px 20px;text-align:left}p{max-width:1100px;color:#bfcbd8}</style>
<h1>Experimental Potato: direct ReSTIR + cached indirect</h1><p>Opt-in; per-pixel GI remains the default because cross-scene tests found mottling and glossy-detail loss. Full-resolution primary geometry and direct illumination. One indirect update per 2×2 tile, reconstructed with a separate recurrent cache. Biased: softer bounce lighting, attenuated highlights and possible lighting lag. Interactive/unbiased modes remain available.</p>
<table><tr><th>Sponza</th><th>Mode</th><th>Frame time</th><th>Equivalent FPS</th></tr>'''+table+'''
</table><p>GPU harness throughput on this Apple M5 Pro; excludes UI/display pacing. ABBA at 540p/720p, one paired run at 1080p. Images use identical camera, lighting, exposure and resolution.</p>
<label>Sequence <select id="motion"><option>static</option><option>pan</option><option>light</option></select></label>
<label>Frame <input id="frame" type="range" min="0" max="16" value="16"><output id="frameText"></output></label>
<div class="pair"><div><h2>Previous Potato</h2><img id="current"></div><div><h2>Cached indirect</h2><img id="potato"></div></div>
<p>Four-seed variability and reference errors are in results.json. Videos show captured sequences at illustrative playback speed, not a real-time FPS recording.</p>
<h2>Camera motion</h2><video controls loop muted src="pan.mp4"></video><h2>Moving light</h2><video controls loop muted src="light.mp4"></video>
<script>const frames=[0,4,8,12,16,20,24,28,32,36,40,44,48,52,56,60,63];function show(){const m=document.querySelector('#motion').value,f=frames[+document.querySelector('#frame').value];document.querySelector('#frameText').textContent=f;document.querySelector('#current').src=m+'-old-f'+f+'.png';document.querySelector('#potato').src=m+'-cache-f'+f+'.png'}document.querySelectorAll('select,input').forEach(e=>e.oninput=show);show()</script>''')
print(json.dumps(dict(timings=timings, summary=summary), indent=2))
