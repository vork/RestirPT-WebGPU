// Frozen denoised input -> actual Presenter, GPU timing and PNG comparisons.
// Run from repository root under with-gpu-lock.ts. No path tracing or dynamic resolution.
import { chromium } from 'playwright';
import { createServer } from 'vite';
import { mkdirSync, writeFileSync } from 'node:fs';
const out='validation/out/fsr';mkdirSync(out,{recursive:true});
const server=await createServer({root:process.cwd(),server:{port:5188,strictPort:true,host:'127.0.0.1',hmr:false,watch:null},logLevel:'warn'});await server.listen();
const browser=await chromium.launch({channel:'chrome',headless:true,args:['--enable-webgpu-developer-features']});
const page=await browser.newPage();const errors=[];page.on('pageerror',e=>errors.push(String(e)));
try {
 await page.goto('http://127.0.0.1:5188/validation/harness/');
 const result=await page.evaluate(async()=>{
  const {Presenter}=await import('/src/core/render/present.ts');
  const adapter=await navigator.gpu.requestAdapter();const device=await adapter.requestDevice({requiredFeatures:['timestamp-query']});
  const errors=[];device.addEventListener('uncapturederror',e=>errors.push(e.error.message));
  const canvas=document.createElement('canvas'),pr=new Presenter(device,canvas);await pr.init();
  const data=new Uint8Array(await (await fetch('/validation/out/potato2-static-final-s1/dn_f63.pfm')).arrayBuffer());
  let offset=0;const line=()=>{let s='';while(data[offset]!==10)s+=String.fromCharCode(data[offset++]);offset++;return s;};
  if(line()!=='PF')throw Error('not PFM');const [iw,ih]=line().split(' ').map(Number),little=Number(line())<0;const dv=new DataView(data.buffer);
  const pixels=new Float32Array(iw*ih*4);
  for(let y=0;y<ih;y++)for(let x=0;x<iw;x++){for(let c=0;c<3;c++)pixels[4*(y*iw+x)+c]=dv.getFloat32(offset+4*(3*((ih-1-y)*iw+x)+c),little);pixels[4*(y*iw+x)+3]=1;}
  const query=device.createQuerySet({type:'timestamp',count:6}),resolved=device.createBuffer({size:256,usage:GPUBufferUsage.QUERY_RESOLVE|GPUBufferUsage.COPY_SRC}),read=device.createBuffer({size:256,usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ});
  const reports=[],images={};
  for(const [sw,sh] of [[960,540],[1280,720]]){
   const px=new Float32Array(sw*sh*4);
   for(let y=0;y<sh;y++)for(let x=0;x<sw;x++)for(let c=0;c<4;c++)px[4*(y*sw+x)+c]=pixels[4*(Math.floor(y*ih/sh)*iw+Math.floor(x*iw/sw))+c];
   const color=device.createTexture({size:[sw,sh],format:'rgba32float',usage:GPUTextureUsage.TEXTURE_BINDING|GPUTextureUsage.COPY_DST});device.queue.writeTexture({texture:color},px,{bytesPerRow:sw*16},[sw,sh]);
   const dw=1920,dh=1080,target=device.createTexture({size:[dw,dh],format:pr.format,usage:GPUTextureUsage.RENDER_ATTACHMENT|GPUTextureUsage.COPY_SRC});
   // ABBA, warming each switch before measurement. Each timestamp pair is allocated in encode order.
   for(const filter of ['bicubic','fsr1','fsr1','bicubic']){
    const samples=[],st={filter,sharpening:.5,exposureEV:0,tonemap:'standard',highlightNonFinite:true};
    for(let f=0;f<48;f++){
     let n=0;const names=[];const stamp=name=>{names.push(name);const i=n;n+=2;return {querySet:query,beginningOfPassWriteIndex:i,endOfPassWriteIndex:i+1};};
     const enc=device.createCommandEncoder();pr.encode(enc,{color,debugOut:color},st,{active:false,split:false,splitPos:.5,probe:false,probePixel:[0,0]},undefined,()=>stamp('present'),{target,toneTimestampWrites:()=>stamp('tonemap'),upscaleTimestampWrites:()=>stamp('easu')});
     enc.resolveQuerySet(query,0,n,resolved,0);enc.copyBufferToBuffer(resolved,0,read,0,n*8);device.queue.submit([enc.finish()]);await read.mapAsync(GPUMapMode.READ);
     const ts=new BigUint64Array(read.getMappedRange());const row={};names.forEach((name,i)=>row[name]=Number(ts[i*2+1]-ts[i*2])/1e6);read.unmap();if(f>=16)samples.push(row);
    }
    const means=Object.fromEntries(Object.keys(samples[0]).map(k=>[k,samples.reduce((a,r)=>a+r[k],0)/samples.length]));
    reports.push({input:[sw,sh],output:[dw,dh],filter,means,totalMs:Object.values(means).reduce((a,b)=>a+b,0),samples});
    if(sw===960){
     const enc=device.createCommandEncoder(),bpr=dw*4,buf=device.createBuffer({size:bpr*dh,usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ});enc.copyTextureToBuffer({texture:target},{buffer:buf,bytesPerRow:bpr},[dw,dh]);device.queue.submit([enc.finish()]);await buf.mapAsync(GPUMapMode.READ);
     const bytes=new Uint8ClampedArray(buf.getMappedRange().slice(0));if(pr.format.startsWith('bgra'))for(let i=0;i<bytes.length;i+=4){const b=bytes[i];bytes[i]=bytes[i+2];bytes[i+2]=b;}
     const c=document.createElement('canvas');c.width=dw;c.height=dh;c.getContext('2d').putImageData(new ImageData(bytes,dw,dh),0,0);images[filter]=c.toDataURL('image/png').split(',')[1];buf.unmap();buf.destroy();
    }
   }
   target.destroy();color.destroy();
  }
  pr.destroy();query.destroy();resolved.destroy();read.destroy();await device.queue.onSubmittedWorkDone();device.destroy();
  return {reports,images,errors,adapter:adapter.info,browser:navigator.userAgent};
 });
 if(errors.length||result.errors.length)throw Error(JSON.stringify([...errors,...result.errors]));
 for(const [name,data] of Object.entries(result.images))writeFileSync(`${out}/${name}.png`,Buffer.from(data,'base64'));
 delete result.images;writeFileSync(`${out}/results.json`,JSON.stringify(result,null,2));
 console.log(JSON.stringify(result.reports.map(({samples,...r})=>r),null,2));
 writeFileSync(`${out}/review.html`, `<!doctype html><meta charset="utf-8"><title>FSR 1 comparison</title><style>body{background:#111b26;color:white;font:16px system-ui;margin:24px}img{width:100%;display:block}button{font:inherit;margin:8px}#wrap{max-width:1920px}</style><h1>540p → 1080p: frozen denoised Sponza</h1><p>Same input, exposure and tone map. FSR 1 uses EASU + noise-attenuated RCAS at 50% strength. This is a spatial filter; it cannot recover missing path-traced detail.</p><button onclick="im.src='bicubic.png';label.textContent='Bicubic'">Bicubic</button><button onclick="im.src='fsr1.png';label.textContent='FSR 1'">FSR 1</button><span id="label">FSR 1</span><div id="wrap"><img id="im" src="fsr1.png"></div><script>const im=document.querySelector('#im'),label=document.querySelector('#label');</script>`);
} finally {await browser.close();await server.close();}
