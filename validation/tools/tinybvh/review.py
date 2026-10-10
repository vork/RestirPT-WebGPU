"""Build the perf4 tinybvh review from matched 540p run-denoise captures.

Run with validation/.venv/bin/python validation/tools/tinybvh/review.py.
Requires the 36 jobs in validation/out/perf4/quality-jobs.json and the existing
12,288 spp Sponza reference. Exports metrics, PNGs, and a local comparison page.
"""
from pathlib import Path
import json
import sys
import subprocess
import shutil
import numpy as np
from PIL import Image, ImageDraw
import flip_evaluator as flip

TOOLS = Path(__file__).resolve().parents[1]
ROOT = TOOLS.parents[1]
sys.path.insert(0, str(TOOLS))
from imageio_util import read_pfm
from denoise_eval import srgb_oetf, relmse, flip_version

OUT = ROOT / 'validation/out/perf4'
CAP = ROOT / 'validation/out'
MODES = ['current', 'opt', 'hq']
MOTIONS = ['static', 'pan', 'light']
LABELS = ['Current TS BVH', 'tinybvh optimized object splits', 'tinybvh spatial splits']
EVAL = [16, 32, 48, 63]
ref = read_pfm(CAP / 'wpq-ref-sponza/ref.pfm')
ref_display = srgb_oetf(ref).astype(np.float32)

def save_png(path, linear=None, display=None):
    a = srgb_oetf(linear) if display is None else display
    im = Image.fromarray(np.uint8(np.clip(a * 255 + .5, 0, 255)))
    im.save(path)
    return im

save_png(OUT / 'reference.png', display=ref_display)
rows, comparisons, vbuffers, temporal = [], [], [], []
for motion in MOTIONS:
    for seed in [1, 2, 3, 4]:
        base_dir = CAP / f'perf4-{motion}-current-s{seed}'
        base_meta = json.loads((base_dir / 'meta.json').read_text())
        for mode in MODES:
            folder = CAP / f'perf4-{motion}-{mode}-s{seed}'
            meta = json.loads((folder / 'meta.json').read_text())
            assert meta['ok'], meta['errors']
            for key in ['packageSha256', 'seed', 'width', 'height', 'frames', 'renderer', 'panOsc', 'lightOsc', 'jitter', 'accumulate']:
                assert meta.get(key) == base_meta.get(key), (folder, key)
            assert meta['denoiser']['settings'] == base_meta['denoiser']['settings']
            for frame in EVAL:
                for kind in ['raw', 'dn']:
                    a = read_pfm(folder / f'{kind}_f{frame}.pfm')
                    display = srgb_oetf(a).astype(np.float32)
                    _, score, _ = flip.evaluate(ref_display, display, 'LDR', inputsRGB=True, applyMagma=False)
                    rows.append(dict(motion=motion, mode=mode, seed=seed, frame=frame, kind=kind,
                                     flip=float(score), relmse=relmse(a, ref), luminance=float(np.mean(a @ [.2126, .7152, .0722]))))
                    if mode != 'current':
                        b = read_pfm(base_dir / f'{kind}_f{frame}.pfm')
                        delta = np.abs(display - srgb_oetf(b))
                        comparisons.append(dict(motion=motion, mode=mode, seed=seed, frame=frame, kind=kind,
                                                display_mae=float(delta.mean()), display_max=float(delta.max()),
                                                changed_pixels=int(np.count_nonzero(np.any(a != b, axis=2))),
                                                pixels_over_one_code=int(np.count_nonzero(np.max(delta, axis=2) > 1 / 255))))
            # All captured primary samples, with hit/miss differences distinguished from primitive ties.
            for file in sorted(folder.glob('vbuf_f*.bin')):
                a = np.fromfile(file, dtype='<u4').reshape(-1, 4)
                b = np.fromfile(base_dir / file.name, dtype='<u4').reshape(-1, 4)
                different = a[:, 0] != b[:, 0]
                miss = (a[:, 0] == 0xffffffff) != (b[:, 0] == 0xffffffff)
                vbuffers.append(dict(motion=motion, mode=mode, seed=seed, frame=int(file.stem[6:]),
                                     pixels=len(a), prim_mismatches=int(different.sum()), hit_miss_mismatches=int(miss.sum())))
            if seed == 1:
                sequence = []
                for f in list(range(0, 64, 4)) + [63]:
                    a = read_pfm(folder / f'dn_f{f}.pfm')
                    disp = srgb_oetf(a)
                    sequence.append(disp)
                    save_png(OUT / f'{motion}-{mode}-f{f}.png', display=disp)
                    if mode != 'current':
                        b = srgb_oetf(read_pfm(base_dir / f'dn_f{f}.pfm'))
                        save_png(OUT / f'{motion}-{mode}-diff-f{f}.png', display=np.clip(abs(disp - b) * 16, 0, 1))
                temporal.append(dict(motion=motion, mode=mode, sample_interval_frames=4,
                                     display_step_mae=float(np.mean([abs(sequence[i] - sequence[i-1]).mean() for i in range(1, 16)]))))
            print('ANALYSED', motion, mode, seed, flush=True)

summary = []
for motion in MOTIONS:
    for mode in MODES:
        for kind in ['raw', 'dn']:
            subset = [r for r in rows if r['motion'] == motion and r['mode'] == mode and r['kind'] == kind]
            summary.append(dict(motion=motion, mode=mode, kind=kind, images=len(subset),
                                **{k: float(np.mean([r[k] for r in subset])) for k in ['flip', 'relmse', 'luminance']}))
report = dict(reference=dict(path='validation/out/wpq-ref-sponza/ref.pfm',
              description='Same camera/light state at oscillation knots; saved reference noise floor retained.',
              **json.loads((CAP / 'wpq-ref-sponza/ref.json').read_text())),
              packageSha256=base_meta['packageSha256'], seeds=[1,2,3,4], frames=EVAL,
              display='exposure 1, clamp, sRGB OETF; identical for all builders', flip_version=flip_version(),
              summary=summary, rows=rows, paired_differences=comparisons, primary_hits=vbuffers, temporal=temporal)
(OUT / 'quality.json').write_text(json.dumps(report, indent=1))

# Review image and video frames use labelled render exports, with no aesthetic retouching.
for motion in MOTIONS:
    for fi, f in enumerate(range(0,64,4)):
        sheet = Image.new('RGB', (1440,306), '#101722')
        draw = ImageDraw.Draw(sheet)
        for col, (mode, label) in enumerate(zip(MODES,LABELS)):
            draw.text((col*480+10,10), label, fill='white')
            im = Image.open(OUT / f'{motion}-{mode}-f{f}.png')
            sheet.paste(im.resize((480,270)), (col*480,36))
        sheet.save(OUT / f'film-{motion}-{fi:03}.png')
    sheet = Image.new('RGB', (1440,580), '#101722')
    draw = ImageDraw.Draw(sheet)
    for col,(mode,label) in enumerate(zip(MODES,LABELS)):
        im=Image.open(OUT/f'{motion}-{mode}-f63.png')
        draw.text((col*480+12,10), label, fill='white')
        sheet.paste(im.resize((480,270)), (col*480,32))
        draw.text((col*480+12,307), 'Detail at native pixel size', fill='white')
        sheet.paste(im.crop((200,220,680,470)), (col*480,325))
    sheet.save(OUT/f'{motion}-comparison.png')

# Preserve the largest isolated change; global averages can hide bright individual pixels.
worst = max((r for r in comparisons if r['kind'] == 'dn'), key=lambda r: r['display_max'])
images = [srgb_oetf(read_pfm(CAP / f"perf4-{worst['motion']}-{mode}-s{worst['seed']}" / f"dn_f{worst['frame']}.pfm")) for mode in MODES]
variant = MODES.index(worst['mode'])
y, x = np.unravel_index(np.argmax(np.max(abs(images[0] - images[variant]), axis=2)), images[0].shape[:2])
left, top = max(0, min(x - 40, 880)), max(0, min(y - 40, 460))
sheet = Image.new('RGB', (960, 374), '#101722')
draw = ImageDraw.Draw(sheet)
draw.text((12, 10), f"Largest isolated change: {worst['motion']}, seed {worst['seed']}, frame {worst['frame']}, pixel ({x},{y}); 4x zoom", fill='white')
for col, (a, label) in enumerate(zip(images, LABELS)):
    im = Image.fromarray(np.uint8(np.clip(a * 255 + .5, 0, 255)))
    sheet.paste(im.crop((left, top, left + 80, top + 80)).resize((320, 320), Image.Resampling.NEAREST), (col * 320, 54))
    draw.text((col * 320 + 12, 32), label, fill='white')
sheet.save(OUT / 'worst-pixel.png')
if shutil.which('ffmpeg'):
    for motion in ['pan', 'light']:
        subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-framerate', '6', '-i', str(OUT / f'film-{motion}-%03d.png'),
                        '-vf', 'fps=24', '-c:v', 'libx264', '-crf', '17', '-pix_fmt', 'yuv420p', str(OUT / f'{motion}.mp4')], check=True)
else:
    print('ffmpeg missing; videos not generated, image playback remains available.')

(OUT/'review.html').write_text('''<!doctype html><meta charset="utf-8"><title>Sponza · BVH quality comparison</title>
<style>body{background:#101722;color:#eef2fa;font:16px system-ui;margin:30px auto;max-width:1100px;padding:0 20px}h1{font-size:30px}p{color:#b9c7da;line-height:1.5}button,select{padding:9px;background:#26354b;color:white;border:1px solid #6581a4;border-radius:5px;margin:4px}input{width:260px;vertical-align:middle}.stage{position:relative;width:min(960px,100%);aspect-ratio:16/9;overflow:hidden;background:black}.stage img{position:absolute;width:100%;height:100%;object-fit:contain}#top{clip-path:inset(0 50% 0 0)}#line{position:absolute;left:50%;height:100%;border-left:2px solid white}small{color:#b9c7da}a{color:#7fbbff}</style>
<h1>Sponza · BVH quality comparison</h1>
<p>Fixed 960 × 540. Same seed, sampling, materials and denoiser. Left of the divider: current renderer. Right: selected tinybvh builder. Drag to inspect shadows, foliage and fabric edges.</p>
<label>Sequence <select id="motion"><option value="static">Static</option><option value="pan">Moving camera</option><option value="light">Moving light</option></select></label>
<label>Builder <select id="mode"><option value="opt">Optimized object splits</option><option value="hq">Spatial splits</option></select></label>
<button id="play">Play capture</button><button id="diff">Difference ×16</button>
<p><label>Captured frame <input id="frame" type="range" min="0" max="16" value="16"></label> <span id="number">63</span>
<label>Divider <input id="wipe" type="range" min="0" max="100" value="50"></label></p>
<div class="stage"><img id="bottom" alt="tinybvh render"><img id="top" alt="current renderer"><span id="line"></span></div>
<p id="caption"></p><small>Capture playback is sampled every four simulation frames at 24 Hz. It is an offline image comparison, not a real-time FPS recording. The final capture is frame 63. Difference view shows absolute display RGB differences ×16.</small>
<p><a href="static-comparison.png">Static overview + native detail</a> · <a href="pan.mp4">Camera motion video</a> · <a href="light.mp4">Moving light video</a> · <a href="worst-pixel.png">Largest isolated difference</a> · <a href="reference.png">12,288-sample reference</a> · <a href="quality.json">Measured quality</a> · <a href="profile.json">Timing reports</a></p>
<script>
const frames=[...Array.from({length:16},(_,i)=>i*4),63];
const el=id=>document.getElementById(id);let differences=false,timer;
function update(){let m=el('motion').value,v=el('mode').value,f=frames[+el('frame').value];el('number').textContent=f;el('bottom').src=`${m}-${v}-${differences?'diff-':''}f${f}.png`;el('top').src=`${m}-current-f${f}.png`;el('top').style.display=el('line').style.display=differences?'none':'';el('caption').textContent=differences?'Absolute difference from current renderer, amplified 16×.':'Current renderer ← divider → tinybvh';}
for(let id of ['motion','mode','frame'])el(id).oninput=update;
el('wipe').oninput=()=>{el('top').style.clipPath=`inset(0 ${100-el('wipe').value}% 0 0)`;el('line').style.left=el('wipe').value+'%'};
el('diff').onclick=()=>{differences=!differences;update()};
el('play').onclick=()=>{if(timer){clearInterval(timer);timer=null;el('play').textContent='Play capture'}else{timer=setInterval(()=>{el('frame').value=(+el('frame').value+1)%17;update()},1000/6);el('play').textContent='Pause'}};update();
</script>''')
print('WROTE', OUT/'review.html')
