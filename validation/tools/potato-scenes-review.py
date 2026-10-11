"""Cross-scene Potato cache screening (two seeds, static/camera/light motion)."""
from pathlib import Path
import json
import numpy as np
from PIL import Image,ImageDraw
from imageio_util import read_pfm
from denoise_eval import srgb_oetf
ROOT=Path(__file__).resolve().parents[2];CAP=ROOT/'validation/out';OUT=CAP/'potato2'
rows=[];metadata=[]
for scene in ['cornell','glossy','glass','foliage']:
 sheet=Image.new('RGB',(768,1260),'#111b26');d=ImageDraw.Draw(sheet)
 for row,motion in enumerate(['static','pan','light']):
  for col,mode in enumerate(['old','cache']):
   frames={f:[] for f in [16,32,48,63]}
   for seed in [1,2]:
    folder=CAP/f'potato2-{scene}-{motion}-{mode}-s{seed}'
    meta=json.loads((folder/'meta.json').read_text());assert meta['ok'] and not meta['errors']
    metadata.append(dict(scene=scene,motion=motion,mode=mode,seed=seed,packageSha256=meta['packageSha256'],lightOsc=meta.get('lightOsc'),panOsc=meta.get('panOsc'),renderer=meta['renderer'],denoiser=meta['denoiser']['settings']))
    for f in frames:
     a=read_pfm(folder/f'dn_f{f}.pfm');assert np.isfinite(a).all();frames[f].append(srgb_oetf(a))
    if seed==1:
     im=Image.fromarray(np.uint8(np.clip(srgb_oetf(a)*255+.5,0,255)));sheet.paste(im,(col*384,row*420+36))
     d.text((col*384+8,row*420+8),f'{scene} / {motion} / {mode}',fill='white')
   rows.append(dict(scene=scene,motion=motion,mode=mode,displayStd=float(np.mean([np.std(np.stack(v),axis=0,ddof=1).mean() for v in frames.values()]))))
 sheet.save(OUT/f'{scene}-comparison.png')
report=dict(rows=rows,metadata=metadata,manifest=json.loads((OUT/'scenes-quality-jobs.json').read_text()),notes=['Screening only: two matched seeds, 384x384, frames 16/32/48/63. No path-traced reference accuracy claim.','Geometry is static; motion covers cameras and analytic lights. Mesh deformation is unsupported by the current scene bridge.'])
(ROOT/'docs/decisions/potato-scenes-results.json').write_text(json.dumps(report,indent=1))
print(json.dumps(rows,indent=2))
