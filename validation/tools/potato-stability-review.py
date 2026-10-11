"""Matched cache stability A/B, four seeds and full motion sequences.

Inputs: validation/out/potato3/{before,after}-jobs.json captures and performance reports.
Run with validation/.venv/bin/python. Compares complete denoised images, not direct-only raw exports.
"""
from pathlib import Path
import json
import subprocess
import numpy as np
from PIL import Image, ImageDraw
from imageio_util import read_pfm
from denoise_eval import srgb_oetf

ROOT = Path(__file__).resolve().parents[2]
CAP = ROOT / 'validation/out'
OUT = CAP / 'potato3'
SCENES = ['sponza', 'cornell', 'glossy', 'glass', 'foliage']
MODES = ['before', 'after']
MOTIONS = ['static', 'pan', 'light']
FRAMES = [16, 32, 48, 63]

def folder(scene, motion, mode, seed=1):
    return CAP / f'potato3-{scene}-{motion}-{mode}-s{seed}'

def read(folder, f):
    a = read_pfm(folder / f'dn_f{f}.pfm').astype(np.float32)
    assert np.isfinite(a).all(), (folder, f)
    return a

def display(a):
    return srgb_oetf(a).astype(np.float32)

def thumb(a):
    return Image.fromarray(np.uint8(np.clip(display(a)*255+.5, 0, 255)))

def blocks(a):
    h, w, c = a.shape
    return a[:h//8*8, :w//8*8].reshape(h//8, 8, w//8, 8, c).mean(axis=(1,3))

rows, metadata, temporal, steps = [], [], [], []
for scene in SCENES:
    for motion in MOTIONS:
        for mode in MODES:
            images = []
            for seed in range(1,5):
                path = folder(scene, motion, mode, seed)
                m = json.loads((path/'meta.json').read_text())
                assert m['ok'] and not m['errors']
                peer = json.loads((folder(scene,motion,'before',seed)/'meta.json').read_text())
                for k in ['packageSha256','seed','width','height','frames','panOsc','lightOsc','jitter','accumulate','renderer']:
                    assert m.get(k)==peer.get(k), (path,k)
                assert m['denoiser']['settings']==peer['denoiser']['settings']
                metadata.append(dict(scene=scene,motion=motion,mode=mode,seed=seed,packageSha256=m['packageSha256'],adapter=m['adapterInfo'],chrome=m['chromeVersion']))
                images.append(np.stack([read(path,f) for f in FRAMES]))
            a=np.stack(images)
            row=dict(scene=scene,motion=motion,mode=mode,displayStd=float(display(a).std(axis=0,ddof=1).mean()),meanLinearLuminance=float((a @ [.2126,.7152,.0722]).mean()))
            if scene=='sponza':row['darkInteriorStd']=float(display(a).std(axis=0,ddof=1)[:,80:480,:250].mean())
            rows.append(row)
            if motion=='static':
                sequence=np.stack([blocks(display(read(folder(scene,motion,mode),f))) for f in range(16,64)])
                temporal.append(dict(scene=scene,mode=mode,block8TemporalStd=float(sequence.std(axis=0,ddof=1).mean()),block8FrameDelta=float(np.abs(np.diff(sequence,axis=0)).mean())))
        print('Analyzed',scene,motion,flush=True)
    # Review mid-motion at f20 as well as mature static at f63.
    sheet=Image.new('RGB',(1280,3*400),'#111b26');d=ImageDraw.Draw(sheet)
    for row,motion in enumerate(MOTIONS):
        f=63 if motion=='static' else 20
        for col,mode in enumerate(MODES):
            im=thumb(read(folder(scene,motion,mode),f));im.thumbnail((640,360))
            sheet.paste(im,(col*640,row*400+36))
            d.text((col*640+10,row*400+10),f'{scene} | {motion} f{f} | cache {mode}',fill='white')
    sheet.save(OUT/f'{scene}.png')
    # Full-frame A/B sequences. Playback is slowed to 15 FPS for inspection, not a performance recording.
    for motion in (['static','pan','light'] if scene in ['sponza','glossy'] else []):
        for f in range(64):
            sheet=Image.new('RGB',(1280,400),'#111b26');d=ImageDraw.Draw(sheet)
            for col,mode in enumerate(MODES):
                im=thumb(read(folder(scene,motion,mode),f));im.thumbnail((640,360))
                sheet.paste(im,(col*640,36));d.text((col*640+10,10),f'{scene} / {motion} / {mode} / frame {f}',fill='white')
            sheet.save(OUT/f'video-{f:03}.png')
        subprocess.run(['/opt/homebrew/bin/ffmpeg','-y','-loglevel','error','-framerate','15','-i',str(OUT/'video-%03d.png'),'-c:v','libx264','-crf','18','-pix_fmt','yuv420p',str(OUT/f'{scene}-{motion}.mp4')],check=True)
for mode in MODES:
    path=CAP/f'potato3-step-{mode}'
    a=np.stack([read(path,f) for f in range(80)])
    mean=(a @ [.2126,.7152,.0722]).mean(axis=(1,2))
    steps.append(dict(mode=mode,meanLinearLuminance=mean.tolist()))
    sheet=Image.new('RGB',(8*256,290),'#111b26');d=ImageDraw.Draw(sheet)
    for col,f in enumerate([31,32,36,48,55,56,60,79]):
        sheet.paste(thumb(a[f]).resize((256,256)),(col*256,34));d.text((col*256+8,8),f'{mode} frame {f}',fill='white')
    sheet.save(OUT/f'light-step-{mode}.png')
perf={mode:json.loads((OUT/f'{mode}-perf.json').read_text()) for mode in MODES}
assert all(r['ok'] for p in perf.values() for r in p['reports'])
timings=[]
for mode,p in perf.items():
    for height in [540,720]:
        reports=[r for r in p['reports'] if r['label'].startswith(f'{height}p-')]
        ms=float(np.mean([r['frame']['meanMs'] for r in reports]))
        timings.append(dict(mode=mode,height=height,ms=ms,fps=1000/ms,runs=len(reports)))
report=dict(rows=rows,temporal=temporal,lightStep=steps,timings=timings,metadata=metadata,performance=perf,
            manifests={mode:json.loads((OUT/f'{mode}-jobs.json').read_text()) for mode in MODES},
            notes=['Four matched seeds at frames 16/32/48/63; lower variability can include bias and is not accuracy.',
                   'Static temporal metrics: display-space 8x8 block means over frames 16..63, one seed; blocks reduce jitter-edge influence. No motion-compensated temporal metric is claimed.',
                   'Full camera/light sequences are for visual inspection, including mid-motion; all geometry is static.',
                   'Light step: half analytic power at frame32, last analytic light removed at56. Full-image mean includes direct light, so it is not an isolated GI response measurement.',
                   'Shader-only comparison, FSR not applied. Motion videos slowed to15FPS, not a frame-rate measurement.',
                   'GPU timings exclude presentation and are sequential two-run screens, not an interleaved ABBA speedup claim.'])
(OUT/'results.json').write_text(json.dumps(report,indent=1))
(ROOT/'docs/decisions/potato-stability-results.json').write_text(json.dumps(report,indent=1))
body=''.join(f'<h2>{s}</h2><img src="{s}.png">'+''.join(f'<p>{m}, slowed capture sequence</p><video controls loop src="{s}-{m}.mp4"></video>' for m in MOTIONS if s in ['sponza','glossy']) for s in SCENES)
(OUT/'review.html').write_text('<!doctype html><meta charset="utf-8"><title>Potato cache stability</title><style>body{background:#111b26;color:#eee;font:16px system-ui;max-width:1280px;margin:32px auto}img,video{width:100%}p{max-width:900px}</style><h1>Potato: stability over accuracy</h1><p>Left: previous cached GI. Right: robust spatial weighting, cross-phase reconstruction, and a bright-sample cap retained during lighting changes. Fixed resolution and ray count. This deliberately reduces indirect energy in noisy regions. Videos are slowed to15FPS for inspection; they are not performance recordings.</p>'+body+'<h2>Light removal</h2><img src="light-step-before.png"><img src="light-step-after.png">')
print(json.dumps(dict(rows=rows,temporal=temporal,timings=timings),indent=1))
