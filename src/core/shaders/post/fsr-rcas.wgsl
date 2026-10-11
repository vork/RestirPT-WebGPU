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

fn rcas_load(p: vec2i) -> vec3f { return textureLoad(easuTex,clamp(p,vec2i(0),vec2i(P.dstSize)-1),0).rgb; }
fn rcas_luma(c: vec3f) -> f32 { return c.g+0.5*(c.r+c.b); }
fn fsr_rcas(p: vec2i) -> vec3f {
  let e=rcas_load(p);
  if (P.sharpening <= 0.0) { return e; }
  let b=rcas_load(p+vec2i(0,-1)); let d=rcas_load(p+vec2i(-1,0));
  let f=rcas_load(p+vec2i(1,0)); let h=rcas_load(p+vec2i(0,1));
  let mn=min(min(b,d),min(f,h)); let mx=max(max(b,d),max(f,h));
  let hitMin=min(mn,e)/max(4*mx,vec3f(1e-8));
  let hitMax=(1-max(mx,e))/min(4*mn-4,vec3f(-1e-8));
  let lobes=max(-hitMin,hitMax);
  var lobe=clamp(max(max(lobes.r,lobes.g),lobes.b),-0.1875,0.0)*P.sharpening;
  let bl=rcas_luma(b); let dl=rcas_luma(d); let el=rcas_luma(e); let fl=rcas_luma(f); let hl=rcas_luma(h);
  let range=max(max(max(bl,dl),max(el,fl)),hl)-min(min(min(bl,dl),min(el,fl)),hl);
  let noise=clamp(abs(0.25*(bl+dl+fl+hl)-el)/max(range,1e-8),0.0,1.0);
  lobe*=1-0.5*noise;
  return clamp((lobe*(b+d+f+h)+e)/(4*lobe+1),vec3f(0),vec3f(1));
}
