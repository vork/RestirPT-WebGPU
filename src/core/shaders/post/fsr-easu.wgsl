// AMD FidelityFX Super Resolution 1 (EASU / RCAS), WGSL port.
// Source: GPUOpen-Effects/FidelityFX-FSR, ffx-fsr/ffx_fsr1.h (v1.20210629).
// Copyright (c) 2021 Advanced Micro Devices, Inc. All rights reserved.
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.
// Port: scalar FP32 texture loads, exact reciprocals with zero guards, clamped
// borders. RCAS enables AMD's noise attenuation; sharpening strength is linear.

struct EasuParams { srcSize: vec2u, dstSize: vec2u }
@group(0) @binding(0) var easuSrc: texture_2d<f32>;
@group(0) @binding(1) var easuDst: texture_storage_2d<rgba16float, write>;
@group(0) @binding(2) var<uniform> E: EasuParams;
fn easu_load(p: vec2i) -> vec3f {
  return clamp(textureLoad(easuSrc, clamp(p, vec2i(0), vec2i(E.srcSize)-1), 0).rgb, vec3f(0), vec3f(1));
}
fn easu_luma(c: vec3f) -> f32 { return c.g + 0.5*(c.r+c.b); }
// Direction.xy and edge strength.z from the cross around each nearest texel.
fn easu_edge(w: f32, a: f32, b: f32, c: f32, d: f32, e: f32) -> vec3f {
  let dir = vec2f(d-b,e-a);
  let len = clamp(abs(dir)/max(vec2f(max(abs(d-c),abs(c-b)),max(abs(e-c),abs(c-a))),vec2f(1e-8)),vec2f(0),vec2f(1));
  return vec3f(dir, dot(len,len))*w;
}
fn easu_tap(off: vec2f, dir: vec2f, len: vec2f, lob: f32, c: vec3f) -> vec4f {
  let v = vec2f(dot(off,dir),dot(off,vec2f(-dir.y,dir.x)))*len;
  let d2 = min(dot(v,v),1.0/lob);
  let base = 0.4*d2-1.0;
  let win = lob*d2-1.0;
  let w = (1.5625*base*base-0.5625)*win*win;
  return vec4f(c*w,w);
}
@compute @workgroup_size(8,8)
fn easu(@builtin(global_invocation_id) id: vec3u) {
  if (any(id.xy >= E.dstSize)) { return; }
  let s = (vec2f(id.xy)+0.5)*vec2f(E.srcSize)/vec2f(E.dstSize)-0.5;
  let p = vec2i(floor(s)); let f = fract(s);
  let b=easu_load(p+vec2i(0,-1)); let c=easu_load(p+vec2i(1,-1));
  let e=easu_load(p+vec2i(-1,0)); let ff=easu_load(p); let g=easu_load(p+vec2i(1,0)); let h=easu_load(p+vec2i(2,0));
  let i=easu_load(p+vec2i(-1,1)); let j=easu_load(p+vec2i(0,1)); let k=easu_load(p+vec2i(1,1)); let l=easu_load(p+vec2i(2,1));
  let n=easu_load(p+vec2i(0,2)); let o=easu_load(p+vec2i(1,2));
  let bl=easu_luma(b); let cl=easu_luma(c); let el=easu_luma(e); let fl=easu_luma(ff);
  let gl=easu_luma(g); let hl=easu_luma(h); let il=easu_luma(i); let jl=easu_luma(j);
  let kl=easu_luma(k); let ll=easu_luma(l); let nl=easu_luma(n); let ol=easu_luma(o);
  let edge=easu_edge((1-f.x)*(1-f.y),bl,el,fl,gl,jl)+easu_edge(f.x*(1-f.y),cl,fl,gl,hl,kl)
          +easu_edge((1-f.x)*f.y,fl,il,jl,kl,nl)+easu_edge(f.x*f.y,gl,jl,kl,ll,ol);
  var dir=edge.xy;
  let d2=dot(dir,dir);
  if (d2 < 1.0/32768.0) { dir=vec2f(1,0); } else { dir*=inverseSqrt(d2); }
  let strength=0.25*edge.z*edge.z;
  let stretch=dot(dir,dir)/max(abs(dir.x),abs(dir.y));
  let len=vec2f(1+(stretch-1)*strength,1-0.5*strength);
  let lob=0.5-0.29*strength;
  let acc=easu_tap(vec2f(0,-1)-f,dir,len,lob,b)+easu_tap(vec2f(1,-1)-f,dir,len,lob,c)
    +easu_tap(vec2f(-1,0)-f,dir,len,lob,e)+easu_tap(-f,dir,len,lob,ff)
    +easu_tap(vec2f(1,0)-f,dir,len,lob,g)+easu_tap(vec2f(2,0)-f,dir,len,lob,h)
    +easu_tap(vec2f(-1,1)-f,dir,len,lob,i)+easu_tap(vec2f(0,1)-f,dir,len,lob,j)
    +easu_tap(vec2f(1,1)-f,dir,len,lob,k)+easu_tap(vec2f(2,1)-f,dir,len,lob,l)
    +easu_tap(vec2f(0,2)-f,dir,len,lob,n)+easu_tap(vec2f(1,2)-f,dir,len,lob,o);
  let lo=min(min(ff,g),min(j,k)); let hi=max(max(ff,g),max(j,k));
  textureStore(easuDst,id.xy,vec4f(clamp(acc.rgb/max(acc.a,1e-8),lo,hi),1));
}
