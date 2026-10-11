"""Compare the 803231e cached-GI baseline with split diffuse/specular + world fallback."""
from pathlib import Path
import json
import numpy as np
from PIL import Image, ImageDraw
from imageio_util import read_pfm
from denoise_eval import srgb_oetf
ROOT=Path(__file__).resolve().parents[2]; CAP=ROOT/'validation/out'; OUT=CAP/'potato4'
rows=[]
for scene in ['sponza','cornell','glossy','glass','foliage']:
    sheet=Image.new('RGB',(1280,1200),'#111b26');d=ImageDraw.Draw(sheet)
    for row,motion in enumerate(['static','pan','light']):
        arrays=[]
        for col,prefix in enumerate(['potato3','potato4-world']):
            path=CAP/f'{prefix}-{scene}-{motion}-fade-s1'
            meta=json.loads((path/'meta.json').read_text()); assert meta['ok'] and not meta['errors']
            a=np.stack([read_pfm(path/f'dn_f{f}.pfm') for f in range(32,64)])
            assert np.isfinite(a).all();arrays.append(a)
            f=63 if motion=='static' else 20
            im=Image.fromarray(np.uint8(np.clip(srgb_oetf(read_pfm(path/f'dn_f{f}.pfm'))*255+.5,0,255)))
            im.thumbnail((640,360));sheet.paste(im,(col*640,row*400+36))
            d.text((col*640+10,row*400+10),f'{scene} / {motion} f{f} / '+('before' if col==0 else 'diffuse world cache + per-pixel specular'),fill='white')
        a,b=arrays
        rows.append(dict(scene=scene,motion=motion,linearMeanRatio=float(b.mean()/a.mean()),displayMAD=float(np.abs(srgb_oetf(a)-srgb_oetf(b)).mean()),displayFrameChange=[float(np.abs(np.diff(srgb_oetf(x),axis=0)).mean()) for x in arrays]))
    sheet.save(OUT/f'{scene}-world.png')
(OUT/'world-review.json').write_text(json.dumps(rows,indent=2))
print(json.dumps(rows,indent=2))
