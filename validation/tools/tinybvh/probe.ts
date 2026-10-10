// Research only: no production import. HQ bounds have not passed the conservative-clipping gate.
import { buildBvh, weldVertices } from '../../../src/core/bvh/sah-builder.ts';
import { buildCwbvh } from '../../../src/core/bvh/cwbvh.ts';
import { encodeLeaf, u32View, woopFloatLength, woopPrimIdIndex, type BvhData } from '../../../src/core/bvh/layout.ts';
const moduleUrl = '/validation/out/tinybvh-probe/builder-raw.mjs';
export async function tinyBuild(positions: Float32Array, indices: Uint32Array, opts: {cwbvh?: boolean} = {}, mode = 0): Promise<BvhData> {
  const t0 = performance.now();
  if (positions.length % 3 || indices.length % 3 || indices.some(i => i >= positions.length / 3)) {
    throw new Error('tinybvh probe requires valid triangle indices and xyz positions');
  }
  const fallback = () => { const b=buildBvh(positions,indices,{maxLeafSize:3}); if(opts.cwbvh)b.cwbvh=buildCwbvh(b); return b; };
  if (indices.length < 6 || !positions.every(Number.isFinite)) return fallback();
  const m = await (await import(/* @vite-ignore */ moduleUrl)).default();
  const vertexCount = positions.length / 3;
  const vp = m._malloc(vertexCount * 16), ip = m._malloc(indices.byteLength);
  try {
    if (!vp || !ip) throw new Error('tinybvh input allocation failed');
    for(let v=0;v<vertexCount;v++) for(let k=0;k<3;k++) m.HEAPF32[(vp>>>2)+v*4+k] = positions[v*3+k];
    m.HEAPU32.set(indices, ip>>>2);
    m._build(vp, vertexCount, ip, indices.length/3, mode);
    const np = m._nodes()>>>2, pp=m._prims()>>>2;
    const nf = m.HEAPF32.slice(np, np+m._nodeCount()*8), nu=u32View(nf);
    const prims = m.HEAPU32.slice(pp, pp+m._indexCount());
    const words: number[] = [], order: number[] = [];
    let maxDepth=0, leafCount=0, maxLeafSize=0;
    const refs: [number,number][]=[];
    type Box = number[];
    const nodeBox=(n:number):Box=>[nf[n*8],nf[n*8+1],nf[n*8+2],nf[n*8+4],nf[n*8+5],nf[n*8+6]];
    function splitLeaf(ids:number[], box:Box, depth:number):number {
      maxDepth=Math.max(maxDepth,depth);
      if(ids.length<=3) {
        leafCount++;maxLeafSize=Math.max(maxLeafSize,ids.length);
        const start=order.length;order.push(...ids);return encodeLeaf(ids.length,start);
      }
      const tight=(list:number[]):Box=>{
        const b=[Infinity,Infinity,Infinity,-Infinity,-Infinity,-Infinity];
        for(const id of list)for(let v=0;v<3;v++)for(let k=0;k<3;k++){
          const x=positions[indices[id*3+v]*3+k];b[k]=Math.min(b[k],x);b[3+k]=Math.max(b[3+k],x);
        }
        for(let k=0;k<3;k++){b[k]=Math.max(b[k],box[k]);b[k+3]=Math.min(b[k+3],box[k+3]);}
        return b;
      };
      const ext=[0,1,2].map(k=>box[k+3]-box[k]);const axis=ext.indexOf(Math.max(...ext));
      const centre=(id:number)=>positions[indices[id*3]*3+axis]+positions[indices[id*3+1]*3+axis]+positions[indices[id*3+2]*3+axis];
      ids.sort((a,b)=>centre(a)-centre(b)||a-b);
      const mid=ids.length>>>1, lists=[ids.slice(0,mid),ids.slice(mid)];
      const dst=words.length/16;words.push(...Array(16).fill(0));
      for(let side=0;side<2;side++){
        const bounds=tight(lists[side]);
        for(let k=0;k<3;k++){words[dst*16+side*8+k]=bounds[k];words[dst*16+side*8+4+k]=bounds[3+k];}
        refs.push([dst*16+3+side*4,splitLeaf(lists[side],bounds,depth+1)]);
      }
      return dst;
    }
    function visit(n: number, depth: number): number {
      maxDepth=Math.max(maxDepth,depth);
      if(depth>(opts.cwbvh ? 64 : 30)) throw new Error('tinybvh depth exceeds renderer limit: '+depth);
      const first=nu[n*8+3], count=nu[n*8+7];
      if(count) return splitLeaf(Array.from(prims.slice(first,first+count)),nodeBox(n),depth);
      const dst=words.length/16;words.push(...Array(16).fill(0));
      for(let side=0;side<2;side++) {
        const child=first+side;
        for(let k=0;k<3;k++){ words[dst*16+side*8+k]=nf[child*8+k]; words[dst*16+side*8+4+k]=nf[child*8+4+k]; }
        refs.push([dst*16+3+side*4, visit(child,depth+1)]);
      }
      return dst;
    }
    if(nu[7]) return fallback();
    visit(0,0);
    if (!opts.cwbvh && maxDepth > 30) throw new Error("tinybvh exceeds BVH2 stack limit");
    const nodes=Float32Array.from(words), nodeBits=u32View(nodes);
    for(const [offset,ref] of refs) nodeBits[offset]=ref;
    const primOrder=Uint32Array.from(order), n=order.length;
    const weldedVid=weldVertices(positions), tris=new Float32Array(n*12), trisW=new Float32Array(woopFloatLength(n));
    const tu=u32View(tris), wu=u32View(trisW);
    for(let i=0;i<n;i++) {
      const p=order[i], ia=indices[p*3], ib=indices[p*3+1], ic=indices[p*3+2], o=i*12;
      for(let k=0;k<3;k++) {
        tris[o+k]=positions[ia*3+k]; tris[o+4+k]=positions[ib*3+k]-positions[ia*3+k]; tris[o+8+k]=positions[ic*3+k]-positions[ia*3+k];
        trisW[o+k]=positions[ia*3+k];trisW[o+4+k]=positions[ib*3+k];trisW[o+8+k]=positions[ic*3+k];
      }
      tu[o+3]=p;wu[o+3]=weldedVid[ia];wu[o+7]=weldedVid[ib];wu[o+11]=weldedVid[ic];wu[woopPrimIdIndex(trisW.length/4,i)]=p;
    }
    const area = (lo:number[],hi:number[]) => { const d=hi.map((v,k)=>Math.max(0,v-lo[k]));return d[0]*d[1]+d[1]*d[2]+d[2]*d[0]; };
    let sahSum=0;
    for(let i=0;i<nodes.length/16;i++)for(let side=0;side<2;side++) {
      const o=i*16+side*8,ref=nodeBits[i*16+3+side*4];
      sahSum+=area(Array.from(nodes.slice(o,o+3)),Array.from(nodes.slice(o+4,o+7)))*((ref&0x80000000)?(ref>>>24)&127:1);
    }
    const rootArea=area([nf[0],nf[1],nf[2]],[nf[4],nf[5],nf[6]]);
    const b:BvhData={nodes,tris,trisW,primOrder,weldedVid,bounds:{min:[nf[0],nf[1],nf[2]],max:[nf[4],nf[5],nf[6]]},stats:{triCount:n,skippedNonFinite:0,nodeCount:nodes.length/16,leafCount,maxDepth,maxLeafSize,avgLeafSize:n/leafCount,sahCost:rootArea>0?1+sahSum/rootArea:0,forcedSplits:0,buildMs:performance.now()-t0}};
    if(opts.cwbvh) { b.cwbvh=buildCwbvh(b); if(b.cwbvh.stats.maxDepth>16) throw new Error("tiny CWBVH exceeds 16-level stack"); }
    console.log('[tinybvh]',mode,JSON.stringify(b.stats),JSON.stringify(b.cwbvh?.stats));
    return b;
  } finally {m._dispose();m._free(vp);m._free(ip);}
}
export const build = (p: Float32Array, i: Uint32Array, o?: {cwbvh?: boolean}) => {
  const name = new URL(import.meta.url).searchParams.get('mode') ?? 'sah';
  const mode = ({sah:0,opt:1,hq:2} as Record<string,number>)[name];
  if (mode === undefined) throw new Error('tinybvh probe mode must be sah, opt or hq');
  return tinyBuild(p,i,o,mode);
};
