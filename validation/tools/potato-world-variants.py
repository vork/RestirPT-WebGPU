"""Review opt-in screen/blue-noise experiments; export an inspectable camera-motion comparison."""
from pathlib import Path
import json, subprocess
import numpy as np
from PIL import Image, ImageDraw
from imageio_util import read_pfm
from denoise_eval import srgb_oetf
ROOT=Path(__file__).resolve().parents[2]; CAP=ROOT/'validation/out'; OUT=CAP/'potato4'
def path(variant,scene,motion):return CAP/f'potato4-{variant}-{scene}-{motion}-fade-s1'
def load(variant,scene,motion,f):
 a=read_pfm(path(variant,scene,motion)/f'dn_f{f}.pfm');assert np.isfinite(a).all();return a
rows=[]
for variant in ['blue','screen']:
 for scene in (['sponza'] if variant=='screen' else ['sponza','cornell','glossy','glass','foliage']):
  for motion in ['static','pan','light']:
   old=json.loads((path('world',scene,motion)/'meta.json').read_text());new=json.loads((path(variant,scene,motion)/'meta.json').read_text())
   assert new['ok'] and not new['errors']
   for key in ['packageSha256','seed','width','height','frames','panOsc','lightOsc','jitter']:assert old.get(key)==new.get(key),(variant,scene,motion,key)
   for key in ['settings','restir']:
    a=dict(old['renderer'][key]);b=dict(new['renderer'][key])
    for name in ['indirectScreenTrace','indirectBlueNoise']:a.pop(name,None);b.pop(name,None)
    assert a==b,(variant,scene,motion,key)
   assert new['renderer']['settings']['indirectBlueNoise' if variant=='blue' else 'indirectScreenTrace']
   for key in old['renderer']:
    if key not in ['settings','restir']:assert old['renderer'][key]==new['renderer'][key],key
   a,b=[np.stack([load(v,scene,motion,f) for f in range(32,64)]) for v in ['world',variant]]
   rows.append(dict(variant=variant,scene=scene,motion=motion,linearMeanRatio=float(b.mean()/a.mean()),displayMAD=float(np.abs(srgb_oetf(a)-srgb_oetf(b)).mean()),displayFrameChange=[float(np.abs(np.diff(srgb_oetf(x),axis=0)).mean()) for x in [a,b]]))
(OUT/'variant-review.json').write_text(json.dumps(rows,indent=2));print(json.dumps(rows,indent=2),flush=True)
for scene in ['sponza','glossy']:
 for f in range(64):
  sheet=Image.new('RGB',(1280,400),'#111b26');d=ImageDraw.Draw(sheet)
  for col,prefix in enumerate(['potato3','potato4-world']):
   a=read_pfm(CAP/f'{prefix}-{scene}-pan-fade-s1'/f'dn_f{f}.pfm')
   im=Image.fromarray(np.uint8(np.clip(srgb_oetf(a)*255+.5,0,255)));im.thumbnail((640,360));sheet.paste(im,(col*640,36))
   d.text((col*640+10,10),f'{scene} / camera motion / frame {f} / '+('803231e cache' if col==0 else 'diffuse world cache + per-pixel specular'),fill='white')
  sheet.save(OUT/f'world-video-{f:03}.png')
 subprocess.run(['/opt/homebrew/bin/ffmpeg','-y','-loglevel','error','-framerate','15','-i',str(OUT/'world-video-%03d.png'),'-c:v','libx264','-crf','18','-pix_fmt','yuv420p',str(OUT/f'{scene}-world-pan.mp4')],check=True)
body=''.join(f'<h2>{s}</h2><img src="{s}-world.png">'+(f'<video controls loop src="{s}-world-pan.mp4"></video>' if s in ['sponza','glossy'] else '') for s in ['sponza','cornell','glossy','glass','foliage'])
(OUT/'world-review.html').write_text('<!doctype html><meta charset="utf-8"><title>Potato persistent diffuse lighting</title><style>body{background:#111b26;color:#eee;font:16px system-ui;max-width:1280px;margin:32px auto}img,video{width:100%}</style><h1>Persistent diffuse lighting and separate reflections</h1><p>Left: 803231e cached GI. Right: diffuse world cache and per-pixel specular. Fixed resolution. Static, camera-motion and moving-light images. Videos slowed to 15 FPS for inspection, not a performance recording. This is one-seed screening; it does not establish a general noise improvement. World reuse helps previously seen surfaces, not unseen geometry.</p>'+body)
