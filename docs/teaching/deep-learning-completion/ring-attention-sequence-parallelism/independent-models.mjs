import fs from 'node:fs';
import assert from 'node:assert/strict';
import {fileURLToPath} from 'node:url';
import path from 'node:path';
import * as m from '../../../../src/learn/data/ring-attention-models.js';
const here=path.dirname(fileURLToPath(import.meta.url)),root=path.resolve(here,'../../../..'),packet=JSON.parse(fs.readFileSync(path.join(root,'docs/teaching/drafts/ring-attention-sequence-parallelism/movement-attention-model.json'))),native=JSON.parse(fs.readFileSync(path.join(here,'independent-native-fixtures.json')));
let comparisons=0,maxForwardError=0,maxGradientError=0,comparingFiniteDifference=false;
function near(a,b,tolerance=2e-10){if(Array.isArray(a)){assert.equal(a.length,b.length);a.forEach((v,i)=>near(v,b[i],tolerance));return;}assert.ok(Number.isFinite(a)&&Number.isFinite(b));const error=Math.abs(a-b);assert.ok(error<=tolerance*Math.max(1,Math.abs(b)),`${a} != ${b}; error ${error}`);comparisons++;if(!comparingFiniteDifference)maxForwardError=Math.max(maxForwardError,error);}
const checks=[],check=(name,run)=>{run();checks.push({name,passed:true});};
check('Three independent native full readers,arbitrary owner identity and270 actual input gradients',()=>{
 for(const record of native.fullModelCases){
  const e=record.expected,current=m.movementInference(record.points,packet,m.ownership(45,4));
  for(const key of ['features','query','key','value'])near(current[key],e[key]);
  near(current.dense.output,e.output);near(current.ring.output,e.output);near(current.original.logits,e.logits);near(current.original.probabilities,e.probabilities);
  for(const P of [1,3,5,8])for(const direction of [1,-1]){const owners=m.ownership(45,P,'striped').map((row,i)=>i%2?[...row].reverse():row).reverse(),out=m.movementInference(record.points,packet,owners,direction);near(out.partitioned.logits,e.logits);near(out.partitioned.probabilities,e.probabilities);near(out.ring.output,e.output);}
  comparingFiniteDifference=true;
  const loss=points=>{const r=m.movementInference(points,packet,m.ownership(45,1));return r.original.logits.reduce((s,x)=>s+.5*x*x,0);};
  for(let i=0;i<45;i++)for(let c=0;c<2;c++){const p=structuredClone(record.points),h=1e-6;p[i][c]+=h;const plus=loss(p);p[i][c]-=2*h;const minus=loss(p),gradient=(plus-minus)/(2*h),expected=record.gradient[i][c],error=Math.abs(gradient-expected);maxGradientError=Math.max(maxGradientError,error);near(gradient,expected,3e-6);}
  comparingFiniteDifference=false;
 }
});
check('Native unequal-value-width packed empty and saturated rows preserve forward and all owner gradients',()=>{
 for(const record of native.attentionCases){const owners=[[8,0],[5,2,6],[7],[4,1,3]];
  for(const direction of [1,-1]){const r=m.ringAttention(record.query,record.key,record.value,owners,record.mask,direction);near(r.output,record.output);assert.ok(r.lse.every(h=>h[1]===-Infinity&&h[5]===-Infinity));}
  const g=m.attentionBackward(record.query,record.key,record.value,owners,record.mask,record.upstream,{head:2,key:4});near([g.dQ,g.dK,g.dV],record.gradients);
  near(g.partials.reduce((s,p)=>s.map((v,c)=>v+p.dK[c]),[0,0,0,0]),g.dK[2][4]);near(g.partials.reduce((s,p)=>s.map((v,c)=>v+p.dV[c]),[0,0]),g.dV[2][4]);
  const order=[5,8,3,0,6,1,7,4,2],reorder=a=>a.map(h=>order.map(i=>h[i])),mask=order.map(i=>order.map(j=>record.mask[i][j])),permuted=m.ringAttention(reorder(record.query),reorder(record.key),reorder(record.value),m.ownership(9,3),mask,-1);near(permuted.output,reorder(record.output));
 }
});
check('Fresh scalar partition merges/common offsets,all-masked blocks and parallel reductions',()=>{
 for(let L=2;L<=12;L++){const scores=Array.from({length:L},(_,i)=>8*Math.sin(i*1.7)-2),values=Array.from({length:L},(_,i)=>3*Math.cos(i*.91)),allowed=Array.from({length:L},(_,i)=>i%3!==1);
  for(const P of [1,Math.min(3,L),L])for(const shift of [-1000,0,1000]){const owners=m.ownership(L,P,'striped').reverse(),r=m.scalarMerge(scores,values,owners,allowed,shift),exponents=scores.map((x,i)=>allowed[i]?Math.exp(x):0),total=exponents.reduce((a,b)=>a+b,0),expected=exponents.reduce((s,x,i)=>s+x*values[i],0)/total;near(r.output,expected);near(r.trace.at(-1).output,expected);near(r.parallel.numerator/r.parallel.total,expected);}
  const empty=m.scalarMerge(scores,values,m.ownership(L,L),scores.map(()=>false),1000);assert.equal(empty.output,0);assert.equal(empty.valid,false);assert.equal(empty.parallel.total,0);assert.ok(empty.trace.every(r=>r.total===0&&r.numerator===0));comparisons+=4;
 }
});
check('Literal individual cell enumeration for uneven/reversed storage tiles and every visit',()=>{
 for(let L=2;L<=24;L++)for(let P=1;P<=Math.min(6,L);P++)for(const tile of [1,2,3,5]){const owners=m.ownership(L,P,'striped').map((r,i)=>i%2?r.slice().reverse():r),literal=owners.map(q=>owners.map(k=>{let cells=0;for(let a=0;a<q.length;a+=tile)for(let b=0;b<k.length;b+=tile){const list=[];for(let i=a;i<Math.min(a+tile,q.length);i++)for(let j=b;j<Math.min(b+tile,k.length);j++)list.push(k[j]<=q[i]);if(list.includes(true))cells+=list.length;}return cells;}));for(const direction of [1,-1]){const r=m.workCounts(owners,tile,direction);near(r.executed,literal,0);near(r.usefulTotal,L*(L+1)/2,0);const critical=Array.from({length:P},(_,s)=>Math.max(...owners.map((_,i)=>literal[i][(i-direction*s+P)%P]))).reduce((a,b)=>a+b,0);near(r.critical,critical,0);}}
});
check('Exact large resource inventory and timeline dependencies,aliases and singleton rank',()=>{
 for(const options of [{local:16777213,ranks:13,B:63,Hq:255,Hkv:51,d:1023,bytes:7},{local:17,ranks:1,B:3,Hq:9,Hkv:3,d:5,bytes:2,alias:true}]){const r=m.resourceCost(options),{local:c,ranks:P,B,Hq,Hkv,d,bytes}=options,C=BigInt(c),b=BigInt(B),q=BigInt(Hq),kv=BigInt(Hkv),D=BigInt(d),s=BigInt(bytes);assert.equal(r.payload,2n*b*C*kv*D*s);assert.equal(r.flops,4n*b*C*C*q*D);assert.equal(r.forwardBytes,BigInt(P-1)*r.payload);assert.equal(r.inventory.nextKV,P===1?0n:r.payload);assert.equal(r.inventory.distinctOutput,options.alias?0n:b*C*q*D*s);comparisons+=5;}
 for(const P of [1,2,7,16])for(const C of [.001,4,7,100])for(const D of [0,4,7,100]){const r=m.timeline(P,C,D);near(r.compute.at(-1).end,r.overlap);assert.equal(r.transfer.length,P-1);for(let i=1;i<P;i++)near(r.compute[i].start,Math.max(r.compute[i-1].end,r.transfer[i-1].end));comparisons++;}
});
check('Persistent token/head identities and logical target/weighted objective examples',()=>{
 for(const [L,H,P]of [[8,4,2],[12,6,3],[15,9,3],[16,8,4]]){const r=m.reshard(L,H,P);assert.deepEqual(r.restored,r.tensor);assert.equal(r.mismatches.length,0);assert.equal(m.reshard(L,H,P,true).mismatches.length,2);near(r.perRank,L*H/P);}
 const t=m.nextTargets([10,11,12,13,14,15],[0,0,0,0,1,1],[[0,1,2],[3,4,5]]);assert.deepEqual(t.correct,[11,12,13,null,15,null]);assert.equal(t.wrong[2],null);near((2*1+6*3)/8,2.5,0);comparisons+=2;
 assert.equal(m.validateOwnership([[0,1],[1,2]],3),false);assert.equal(m.validateOwnership([[0],[],[1,2]],3),false);assert.equal(m.validateOwnership([[2,0],[1]],3),true);comparisons+=3;
});
fs.writeFileSync(path.join(here,'independent-numerical-checks.json'),JSON.stringify({passed:true,comparisons,maxForwardError,maxGradientError,checks,limits:['Fresh native SDPA/reference parameter values,270 input finite differences (h=1e-6; relative/absolute tolerance3e-6) and independent literal arithmetic.','Existing canonical multiprocess Gloo evidence reused; no GPU or measured overlap claim.','Runtime representation/browser assessment remains separate.']},null,2)+'\n');
console.log(comparisons+' independent Ring comparisons passed; maximum input gradient difference '+maxGradientError);
