"""Four-seed cached Potato sampling comparison and matched motion screens."""
from pathlib import Path
import json,subprocess,sys
import numpy as np
from PIL import Image,ImageDraw
from imageio_util import read_pfm
from denoise_eval import srgb_oetf
VARIANT=sys.argv[1] if len(sys.argv)>1 else 'final'
LABEL='faster cached Potato + four denoising passes'
ROOT=Path(__file__).resolve().parents[2];CAP=ROOT/'validation/out';OUT=CAP/'potato6'
def directory(scene,motion,seed,wide):
 prefix=f'potato6-{VARIANT}' if wide else 'potato5-smooth'
 return CAP/f'{prefix}-{scene}-{motion}-fade-s{seed}'
def load(scene,motion,seed,wide,f):
 a=read_pfm(directory(scene,motion,seed,wide)/f'dn_f{f}.pfm').astype(np.float32);assert np.isfinite(a).all();return a

def blocks(a):
 h,w=a.shape[:2];h=h//8*8;w=w//8*8
 return a[:h,:w].reshape(h//8,8,w//8,8,3).mean((1,3))

def check(scene,motion,seed):
 a,b=[json.loads((directory(scene,motion,seed,v)/'meta.json').read_text()) for v in [False,True]]
 for x in [a,b]:assert x['ok'] and not x['errors']
 for key in ['packageSha256','seed','width','height','frames','panOsc','lightOsc','jitter']:assert a.get(key)==b.get(key),(scene,motion,seed,key)
 # The archived baseline may still expose disabled experiment fields. No sampling override is enabled.
 for x in [a,b]:
  for key in ['indirectBlueNoise','indirectScreenTrace']:assert not x['renderer']['settings'].get(key,False)
 da=dict(a['denoiser']['settings']);db=dict(b['denoiser']['settings'])
 if VARIANT in ['stable','final']:assert db.pop('iterations')==4;assert da.pop('iterations')==3
 assert da==db
 return a['packageSha256']
rows=[];static=[]
for scene in ['sponza','cornell','glossy','glass','foliage']:
 seeds=[s for s in (range(1,5) if scene in ['sponza','cornell','glossy'] else [1]) if (directory(scene,'static',s,True)/'meta.json').exists()]
 noise=[[],[]];flicker=[[],[]];coarse=[[],[]];energy=[[],[]]
 for seed in seeds:
  check(scene,'static',seed)
  for v in [False,True]:
   seq=[];prev=None;change=[];means=[]
   for f in range(32,64):
    a=load(scene,'static',seed,v,f);means.append(float(a.mean()));disp=srgb_oetf(a).astype(np.float32);seq.append(blocks(disp))
    if prev is not None:change.append(float(np.abs(disp-prev).mean()))
    prev=disp
   seq=np.stack(seq);noise[int(v)].append(seq);flicker[int(v)].append(float(np.mean(change)));coarse[int(v)].append(float(np.abs(np.diff(seq,axis=0)).mean()));energy[int(v)].append(float(np.mean(means)))
 stats=dict(scene=scene,seeds=list(seeds),displayFrameChange=flicker,blockFrameChange=coarse,linearMean=energy,frameChangeRatio=float(np.mean(flicker[1])/np.mean(flicker[0])),blockFrameChangeRatio=float(np.mean(coarse[1])/np.mean(coarse[0])),linearMeanRatio=float(np.mean(energy[1])/np.mean(energy[0])))
 if len(list(seeds))>1:
  variance=[float(np.var(np.stack(x),axis=0,ddof=1).mean()) for x in noise];stats.update(blockSeedVariance=variance,blockSeedVarianceRatio=variance[1]/variance[0])
 static.append(stats);print(json.dumps(stats),flush=True)
 sheet=Image.new('RGB',(1280,1200),'#111b26');draw=ImageDraw.Draw(sheet)
 for row,motion in enumerate(['static','pan','light']):
  check(scene,motion,1);f=63 if motion=='static' else 20
  a,b=[load(scene,motion,1,v,f) for v in [False,True]]
  rows.append(dict(scene=scene,motion=motion,frame=f,displayMAD=float(np.abs(srgb_oetf(a)-srgb_oetf(b)).mean()),linearMeanRatio=float(b.mean()/a.mean())))
  for col,x in enumerate([a,b]):
   im=Image.fromarray(np.uint8(np.clip(srgb_oetf(x)*255+.5,0,255)));im.thumbnail((640,360));sheet.paste(im,(col*640,row*400+36))
   draw.text((col*640+10,row*400+10),f'{scene} / {motion} f{f} / '+('previous cached Potato' if col==0 else LABEL),fill='white')
 sheet.save(OUT/f'{scene}-{VARIANT}.png')
report=dict(static=static,screens=rows,notes=['Frames32–63; four seeds for Sponza/Cornell/glossy static, one seed for glass/foliage and motion.','8x8 block metrics target low-frequency temporal changes and variation between independent seeds; they are not accuracy measures.','Rough opaque surfaces use light-only direct sampling and sparse indirect reflection samples. Reflections have separate spatial and temporal reconstruction; final image denoising uses four passes versus three.','Motion screenshots are qualitative screening, not motion-compensated flicker measurements.','Baseline: b9eb1d2; candidate: separate direct-light passes, adaptive diffuse refresh and sparse rough reflection history.'])
(OUT/f'{VARIANT}-review.json').write_text(json.dumps(report,indent=1))
for scene in ['sponza','glossy']:
 for f in range(64):
  sheet=Image.new('RGB',(1280,400),'#111b26');draw=ImageDraw.Draw(sheet)
  for col,v in enumerate([False,True]):
   a=load(scene,'pan',1,v,f);im=Image.fromarray(np.uint8(np.clip(srgb_oetf(a)*255+.5,0,255)));im.thumbnail((640,360));sheet.paste(im,(col*640,36))
   draw.text((col*640+10,10),f'{scene} / camera motion / f{f} / '+('previous cached Potato' if col==0 else LABEL),fill='white')
  sheet.save(OUT/f'wide-video-{f:03}.png')
 subprocess.run(['/opt/homebrew/bin/ffmpeg','-y','-loglevel','error','-framerate','15','-i',str(OUT/'wide-video-%03d.png'),'-c:v','libx264','-crf','18','-pix_fmt','yuv420p',str(OUT/f'{scene}-{VARIANT}-pan.mp4')],check=True)
body=''.join(f'<h2>{s}</h2><img src="{s}-{VARIANT}.png">'+(f'<video controls loop src="{s}-{VARIANT}-pan.mp4"></video>' if s in ['sponza','glossy'] else '') for s in ['sponza','cornell','glossy','glass','foliage'])
(OUT/f'{VARIANT}-review.html').write_text('<!doctype html><meta charset="utf-8"><title>Potato diffuse stability</title><style>body{background:#111b26;color:#eee;font:16px system-ui;max-width:1280px;margin:32px auto}img,video{width:100%}</style><h1>Cached Potato sampling priorities</h1><p>Left: previous cached Potato. Right: candidate filter/settings named in each image. Direct light sampling and rough reflection reconstruction both change. Fixed resolution. Videos slowed to 15 FPS for inspection, not a performance recording. The tradeoff is softer rough reflections and different direct-light noise.</p>'+body)
