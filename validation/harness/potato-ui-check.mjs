import { chromium } from 'playwright';
import { createServer } from 'vite';
import assert from 'node:assert/strict';
import { writeFileSync } from 'node:fs';
const server=await createServer({root:process.cwd(),server:{port:5187,strictPort:true,host:'127.0.0.1',hmr:false,watch:null},logLevel:'warn'});
await server.listen();
const browser=await chromium.launch({channel:'chrome',headless:true});
const page=await browser.newPage({viewport:{width:1280,height:900}});
const errors=[];page.on('pageerror',e=>errors.push(String(e)));
const results=[];
try {
 await page.goto('http://127.0.0.1:5187/');
 await page.waitForFunction(()=>window.__integration?.renderer()?.denoisedLastFrame,{timeout:60000});
 await page.evaluate(()=>{window.__app.panel.folders.restir.expanded=true;window.__app.panel.folders.denoiser.expanded=true;});
 const state=()=>page.evaluate(()=>{const r=window.__integration.renderer();return {mode:r.options.restirMode,integrator:r.options.renderMode,bounces:r.options.maxBounces,settings:r.restirSettings(),denoise:r.denoiseWanted(),dn:r.denoiser.settings.iterations,errors:window.__webgpuErrors};});
 assert.equal(await page.evaluate(()=>window.__app.present.filter),'fsr1');
 assert.equal(await page.evaluate(()=>window.__app.present.sharpening),0.5);
 const first=await state();assert.equal(first.mode,'interactive');assert.equal(first.integrator,'restir');assert.equal(first.dn,4);results.push(first);
 const selector=page.locator('select').filter({has:page.locator('option', {hasText:'Potato ReSTIR'})});
 await selector.selectOption({label:'Potato ReSTIR'});
 await page.waitForFunction(()=>window.__integration.renderer().options.restirMode==='potato' && window.__integration.renderer().denoiser.settings.iterations===3);
 const cache=page.locator('select').filter({has:page.locator('option',{hasText:'cached 1/4 (experimental)'})});
 await cache.selectOption({label:'cached 1/4 (experimental)'});
 await page.waitForFunction(()=>window.__integration.renderer().restir.settings.indirectCache===2 && window.__integration.renderer().denoiser.settings.iterations===2);
 for (const mode of ['potato','interactive','potato']) {
  const frameBefore=await page.evaluate(()=>window.__app.frameCounter);
  await selector.selectOption({label:mode==='potato'?'Potato ReSTIR':'interactive'});
  await page.waitForFunction(m=>{const r=window.__integration.renderer();return r.options.restirMode===m && r.restir?.settings.temporal===(m==='interactive') && r.denoiser?.settings.iterations===(m==='potato'?2:4) && r.denoisedLastFrame;},mode,{timeout:60000});
  await page.waitForFunction(f=>window.__app.frameCounter>=f+12,frameBefore,{timeout:60000});
  await page.evaluate(()=>window.__app.gpu?.device?.queue?.onSubmittedWorkDone());
  const s=await state();assert.equal(s.bounces,first.bounces);assert.equal(s.errors.length,0);results.push(s);
 }
 // Exercise the actual fallback through the running frame loop.
 for (const on of [false, true]) {
  const f = await page.evaluate(on=>{window.__integration.renderer().setDenoise(on);return window.__app.frameCounter;},on);
  await page.waitForFunction(({on,f})=>{const r=window.__integration.renderer();return window.__app.frameCounter>f+12 && r.denoisedLastFrame===on && r.restir.settings.indirectCache===(on?2:0);},{on,f},{timeout:60000});
  assert.deepEqual(await page.evaluate(()=>window.__webgpuErrors),[]);
 }
 await page.evaluate(()=>{for(const [name,f] of Object.entries(window.__app.panel.folders)) f.expanded=['restir','denoiser','render'].includes(name);window.__app.panel.refresh();});
 await page.locator('select').filter({has:page.locator('option', {hasText:'Potato ReSTIR'})}).scrollIntoViewIfNeeded();
 const filter=page.locator('select').filter({has:page.locator('option',{hasText:'FSR 1 (EASU + RCAS)'})});
 await filter.selectOption({label:'FSR 1 (EASU + RCAS)'});
 await page.waitForFunction(()=>window.__app.present.filter==='fsr1');
 await page.setViewportSize({width:1400,height:950});
 const fs=await page.evaluate(()=>window.__app.frameCounter);
 await page.waitForFunction(f=>window.__app.frameCounter>f+12,fs);
 assert.deepEqual(await page.evaluate(()=>window.__webgpuErrors),[]);
 assert.equal(await page.evaluate(()=>window.__app.render.dynamicResolution),false);
 await page.screenshot({path:'validation/out/potato2/app-potato.png'});
 await page.goto('file://'+process.cwd()+'/validation/out/potato2/review.html');
 for(const motion of ['static','pan','light']) {
  await page.selectOption('#motion',motion);
  for(const index of [0,8,16]) {
   await page.locator('#frame').fill(String(index));await page.locator('#frame').dispatchEvent('input');
   await page.waitForFunction(()=>[...document.querySelectorAll('img')].every(i=>i.complete&&i.naturalWidth===960));
  }
 }
 for (const video of await page.locator('video').all()) {
  await video.evaluate(async v=>{await v.play();v.pause();});
 }
 await page.selectOption('#motion','static');
 await page.screenshot({path:'validation/out/potato2/review-checked.png',fullPage:true});
 assert.deepEqual(errors,[]);writeFileSync('validation/out/potato2/ui-check.json',JSON.stringify({ok:true,states:results,errors},null,2));
 console.log('PASS: FSR default at 0.5 sharpening, interactive startup, Potato/interactive switches, image controls, video playback; no WebGPU/page errors.');
} finally {await browser.close();await server.close();}
