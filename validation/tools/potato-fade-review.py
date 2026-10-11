"""Review the presentation-only indirect reveal against stabilized, unfaded Potato GI.
Run after potato3/{after,fade}-jobs captures, using validation/.venv/bin/python.
"""
from pathlib import Path
import json
import sys
import subprocess
import numpy as np
from PIL import Image, ImageDraw
from imageio_util import read_pfm
from denoise_eval import srgb_oetf
ROOT=Path(__file__).resolve().parents[2]
CAP=ROOT/'validation/out';OUT=CAP/'potato3'
SCENES=['sponza','cornell','glossy','glass','foliage']
MOTIONS=['static','pan','light']

def load(scene,motion,mode,f):
    a=read_pfm(CAP/f'potato3-{scene}-{motion}-{mode}-s1'/f'dn_f{f}.pfm').astype(np.float32)
    assert np.isfinite(a).all()
    return a

def image(a):
    return Image.fromarray(np.uint8(np.clip(srgb_oetf(a)*255+.5,0,255)))

rows=[]
for scene in SCENES:
    sheet=Image.new('RGB',(1280,1200),'#111b26');d=ImageDraw.Draw(sheet)
    for row,motion in enumerate(MOTIONS):
        before=json.loads((CAP/f'potato3-{scene}-{motion}-after-s1'/'meta.json').read_text())
        after=json.loads((CAP/f'potato3-{scene}-{motion}-fade-s1'/'meta.json').read_text())
        assert after['ok'] and not after['errors']
        for key in ['packageSha256','seed','width','height','frames','panOsc','lightOsc','jitter','renderer','denoiser']:
            if key=='denoiser':assert after[key]['settings']==before[key]['settings']
            else: assert after.get(key)==before.get(key),(scene,motion,key)
        f=8 if motion=='static' else 20
        for col,mode in enumerate(['after','fade']):
            im=image(load(scene,motion,mode,f));im.thumbnail((640,360));sheet.paste(im,(640*col,400*row+36))
            d.text((640*col+10,400*row+10),f'{scene} / {motion} f{f} / '+('no fade' if mode=='after' else '16-update GI fade'),fill='white')
        frames=[]
        for f in [0,4,8,12,16,24,32,48,63]:
            a=load(scene,motion,'after',f);b=load(scene,motion,'fade',f)
            frames.append(dict(frame=f,displayMeanAbsDifference=float(np.abs(srgb_oetf(a)-srgb_oetf(b)).mean()),unfadedLinearMean=float(a.mean()),fadedLinearMean=float(b.mean())))
        rows.append(dict(scene=scene,motion=motion,frames=frames,packageSha256=after['packageSha256']))
    sheet.save(OUT/f'{scene}-fade.png')
    if scene not in ['sponza','glossy'] or '--metrics-only' in sys.argv:continue
    for motion in MOTIONS:
        for f in range(64):
            sheet=Image.new('RGB',(1280,400),'#111b26');d=ImageDraw.Draw(sheet)
            for col,mode in enumerate(['after','fade']):
                im=image(load(scene,motion,mode,f));im.thumbnail((640,360));sheet.paste(im,(640*col,36))
                d.text((640*col+10,10),f'{scene} / {motion} / f{f} / '+('no fade' if mode=='after' else '16-update GI fade'),fill='white')
            sheet.save(OUT/f'fade-video-{f:03}.png')
        subprocess.run(['/opt/homebrew/bin/ffmpeg','-y','-loglevel','error','-framerate','15','-i',str(OUT/'fade-video-%03d.png'),'-c:v','libx264','-crf','18','-pix_fmt','yuv420p',str(OUT/f'{scene}-{motion}-fade.mp4')],check=True)
    print('Rendered',scene,flush=True)
report=dict(rows=rows,manifest=json.loads((OUT/'fade-jobs.json').read_text()),notes=['One-seed visual screen, 15 matched jobs with all 64 frames, not a four-seed stability remeasurement.','Baseline fb1b6f0 includes robust weighting; only the indirect reveal differs. Direct-light samples and cache history are unchanged.','Fade is smoothstep(1,17,valid history age) at combination; the main denoiser can further soften its visible response.','Age is measured in valid updates, not wall-clock time. Missing samples pause aging; history rejection/reset starts over.','Static frame63 should approach the same established lighting. Under motion, fresh areas are deliberately darker.','Videos slowed to15FPS for inspection, not a performance recording.'])
long=[]
for scene in ['cornell','glossy','sponza']:
    for f in [63,127,255]:
        arrays=[read_pfm(CAP/f'potato3-{scene}-static-long-{mode}-s1'/f'dn_f{f}.pfm') for mode in ['no-fade','fade']]
        a,b=arrays
        assert all(np.isfinite(x).all() for x in arrays)
        long.append(dict(scene=scene,frame=f,linearMeanRatio=float(b.mean()/a.mean()),displayMeanAbsDifference=float(np.abs(srgb_oetf(a)-srgb_oetf(b)).mean())))
report['longStatic']=long
report['performance']=json.loads((OUT/'fade-perf.json').read_text())
assert all(r['ok'] for r in report['performance']['reports'])
report['notes'].append('Main resolve history extends settling beyond the GI ramp. At frame255 the faded static mean is still 0.8% lower in Sponza, 4.9% Cornell, 3.8% glossy. Full-image brightness recovery is not instantaneous at update17.')
report['longManifests']={mode:json.loads((OUT/f'long-{mode}-jobs.json').read_text()) for mode in ['no-fade','fade']}
(OUT/'fade-results.json').write_text(json.dumps(report,indent=1));(ROOT/'docs/decisions/potato-fade-results.json').write_text(json.dumps(report,indent=1))
body=''.join(f'<h2>{s}</h2><img src="{s}-fade.png">'+''.join(f'<p>{m}, slowed capture sequence</p><video controls loop src="{s}-{m}-fade.mp4"></video>' for m in MOTIONS if s in ['sponza','glossy']) for s in SCENES)
(OUT/'fade-review.html').write_text('<!doctype html><meta charset="utf-8"><title>Potato indirect reveal</title><style>body{background:#111b26;color:#eee;font:16px system-ui;max-width:1280px;margin:32px auto}img,video{width:100%}p{max-width:900px}</style><h1>Fade new indirect lighting in gently</h1><p>Left: stabilized cache without fading. Right: the same cache with a smooth 16-update reveal. Direct light and geometry are immediately visible. Fresh areas temporarily have less bounce light. Videos are slowed to15FPS; they do not measure rendering performance.</p>'+body)
print('Fade review complete.')
