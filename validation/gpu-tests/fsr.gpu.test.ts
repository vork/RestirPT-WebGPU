// Compare the WGSL port with an independent scalar CPU evaluation of AMD's
// EASU/RCAS equations. Includes FP16 inter-pass quantization and flat endpoints.
import { afterAll, expect, it } from 'vitest';
import { getTestGpu, releaseTestGpu } from './device-factory.ts';
import { Presenter, type PresentDebug, type UpscaleFilter } from '../../src/core/render/present.ts';
afterAll(releaseTestGpu);
const clamp = (x: number, lo = 0, hi = 1) => Math.max(lo, Math.min(hi, x));
const half = (x: number) => new Float16Array([x])[0];
const luma = (v: number[]) => v[1] + (v[0] + v[2]) / 2;
function reference(src: number[][], sw: number, sh: number, dw: number, dh: number, sharp: number): number[][] {
  const get = (x: number, y: number) => src[clamp(y, 0, sh - 1) * sw + clamp(x, 0, sw - 1)];
  const offsets = [[0,-1],[1,-1],[-1,0],[0,0],[1,0],[2,0],[-1,1],[0,1],[1,1],[2,1],[0,2],[1,2]];
  const easu: number[][] = [];
  for (let y = 0; y < dh; y++) for (let x = 0; x < dw; x++) {
    const sx = (x+.5)*sw/dw-.5, sy = (y+.5)*sh/dh-.5;
    const fx = sx-Math.floor(sx), fy = sy-Math.floor(sy);
    const taps = offsets.map(([ox,oy]) => get(Math.floor(sx)+ox, Math.floor(sy)+oy));
    const lum = taps.map(luma);
    let dx = 0, dy = 0, len = 0;
    for (const [w,a,b,c,d,e] of [[(1-fx)*(1-fy),0,2,3,4,7],[fx*(1-fy),1,3,4,5,8],[(1-fx)*fy,3,6,7,8,10],[fx*fy,4,7,8,9,11]]) {
      const gx = lum[d]-lum[b], gy = lum[e]-lum[a];
      dx += w*gx; dy += w*gy;
      len += w*(clamp(Math.abs(gx)/Math.max(1e-8,Math.abs(lum[d]-lum[c]),Math.abs(lum[c]-lum[b])))**2
        +clamp(Math.abs(gy)/Math.max(1e-8,Math.abs(lum[e]-lum[c]),Math.abs(lum[c]-lum[a])))**2);
    }
    const d2 = dx*dx+dy*dy;
    if (d2 < 1/32768) { dx=1; dy=0; } else { dx/=Math.sqrt(d2); dy/=Math.sqrt(d2); }
    len = (len/2)**2;
    const lx = 1+((dx*dx+dy*dy)/Math.max(Math.abs(dx),Math.abs(dy))-1)*len, ly = 1-.5*len, lob=.5-.29*len;
    const acc=[0,0,0]; let weight=0;
    offsets.forEach(([ox,oy],i) => {
      const vx=((ox-fx)*dx+(oy-fy)*dy)*lx, vy=(-(ox-fx)*dy+(oy-fy)*dx)*ly;
      const r=Math.min(vx*vx+vy*vy,1/lob), w=(1.5625*(.4*r-1)**2-.5625)*(lob*r-1)**2;
      weight+=w; for (let c=0;c<3;c++) acc[c]+=w*taps[i][c];
    });
    easu.push(acc.map((a,c)=>half(clamp(a/weight,Math.min(...[3,4,7,8].map(i=>taps[i][c])),Math.max(...[3,4,7,8].map(i=>taps[i][c]))))));
  }
  const eg = (x: number,y: number) => easu[clamp(y,0,dh-1)*dw+clamp(x,0,dw-1)];
  return easu.map((e,i)=>{
    const x=i%dw,y=Math.floor(i/dw), ring=[eg(x,y-1),eg(x-1,y),eg(x+1,y),eg(x,y+1)];
    const lobes=e.map((v,c)=>{const mn=Math.min(...ring.map(r=>r[c])),mx=Math.max(...ring.map(r=>r[c]));return Math.max(-Math.min(mn,v)/Math.max(4*mx,1e-8),(1-Math.max(mx,v))/Math.min(4*mn-4,-1e-8));});
    const ls=[luma(e),...ring.map(luma)];
    const nz=clamp(Math.abs(ls.slice(1).reduce((a,b)=>a+b,0)/4-ls[0])/Math.max(Math.max(...ls)-Math.min(...ls),1e-8));
    const lob=clamp(Math.max(...lobes),-.1875,0)*sharp*(1-.5*nz);
    return e.map((v,c)=>clamp((v+lob*ring.reduce((a,r)=>a+r[c],0))/(4*lob+1)));
  });
}
it('FSR: scalar reference, endpoints, odd sizes, bypass, debug split, resize and strength', async () => {
  const { device } = await getTestGpu();
  const errors: string[]=[]; const onError=(e: GPUUncapturedErrorEvent)=>errors.push(e.error.message);
  device.addEventListener('uncapturederror',onError);
  const canvas=document.createElement('canvas'); const pr=new Presenter(device,canvas); await pr.init();
  const sw=17,sh=13;
  const source=Array.from({length:sw*sh},(_,i)=>{const x=i%sw,y=Math.floor(i/sw);return x<3?[0,0,0]:x>13?[1,1,1]:[half(x/17),half(y/13),half(x>y?.8:.1)];});
  const color=device.createTexture({size:[sw,sh],format:'rgba16float',usage:GPUTextureUsage.TEXTURE_BINDING|GPUTextureUsage.COPY_DST});
  device.queue.writeTexture({texture:color},new Uint16Array(new Float16Array(source.flatMap(c=>[...c,1])).buffer),{bytesPerRow:sw*8},[sw,sh]);
  const dbg: PresentDebug={active:false,split:false,splitPos:.4,probe:false,probePixel:[0,0]};
  async function render(dw:number,dh:number,filter:UpscaleFilter,sharp:number,debug=dbg,pretone=true) {
    const target=device.createTexture({size:[dw,dh],format:pr.format,usage:GPUTextureUsage.RENDER_ATTACHMENT|GPUTextureUsage.COPY_SRC});
    const enc=device.createCommandEncoder();let passes=0;
    pr.encode(enc,{color,debugOut:color},{exposureEV:0,tonemap:'raw',filter,sharpening:sharp,highlightNonFinite:true,pretone},debug,undefined,undefined,{target,upscaleTimestampWrites:()=>{passes++;return undefined;}});
    const bpr=Math.ceil(dw*4/256)*256,buf=device.createBuffer({size:bpr*dh,usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ});
    enc.copyTextureToBuffer({texture:target},{buffer:buf,bytesPerRow:bpr},[dw,dh]);device.queue.submit([enc.finish()]);await buf.mapAsync(GPUMapMode.READ);
    const a=new Uint8Array(buf.getMappedRange()),rgb:number[][]=[];
    const channels=pr.format.startsWith('bgra')?[2,1,0]:[0,1,2];
    for(let y=0;y<dh;y++)for(let x=0;x<dw;x++)rgb.push(channels.map(c=>a[y*bpr+x*4+c]));
    buf.unmap();buf.destroy();target.destroy();return {rgb,passes};
  }
  for(const [dw,dh] of [[34,26],[29,23],[51,39]])for(const sharp of [0,.5,1]){
    const out=await render(dw,dh,'fsr1',sharp), ref=reference(source,sw,sh,dw,dh,sharp);
    expect(out.passes).toBe(1);
    let max=0;out.rgb.forEach((p,i)=>p.forEach((v,c)=>{max=Math.max(max,Math.abs(v-Math.round(ref[i][c]*255)));}));
    expect(max,`${dw}x${dh}, sharp ${sharp}`).toBeLessThanOrEqual(1);
  }
  for(const [dw,dh] of [[sw,sh],[11,9],[22,9]]){
    const a=await render(dw,dh,'fsr1',1), b=await render(dw,dh,'bicubic',1);expect(a.passes).toBe(0);expect(a.rgb).toEqual(b.rgb);
  }
  const debug={...dbg,active:true,split:true};
  const a=await render(34,26,'fsr1',.5,debug),b=await render(34,26,'bicubic',.5,debug);
  for(let y=0;y<26;y++)for(let x=0;x<13;x++)expect(a.rgb[y*34+x]).toEqual(b.rgb[y*34+x]);
  expect((await render(34,26,'fsr1',.5,{...dbg,active:true})).passes).toBe(0);
  expect((await render(34,26,'fsr1',.5,dbg,false)).rgb).toEqual((await render(34,26,'fsr1',.5)).rgb);
  expect(errors).toEqual([]);device.removeEventListener('uncapturederror',onError);pr.destroy();color.destroy();
});
