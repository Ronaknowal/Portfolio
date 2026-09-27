// Complementary root review: numerical differences, ordering and independent native oracles.
import fs from 'node:fs';
import assert from 'node:assert/strict';
import { route, dispatch, balance, costs, frozenMoE } from '../src/learn/data/moe-models.js';
const directory='docs/teaching/deep-learning-completion/mixture-of-experts-transformers-moe';
let count=0, maximum=0, seed=92741;
const random=()=>((seed=(Math.imul(seed,1664525)+1013904223)>>>0)/4294967296);
const scalar=(a,b,tolerance=2e-8)=>{assert.ok(Number.isFinite(a)&&Number.isFinite(b)); const delta=Math.abs(a-b);maximum=Math.max(maximum,delta);assert.ok(delta<tolerance,`${a} versus ${b}: ${delta}`);count++;};
const compare=(a,b,tolerance)=>{if(Array.isArray(a)){assert.equal(a.length,b.length);a.forEach((v,i)=>compare(v,b[i],tolerance));}else scalar(a,b,tolerance);};
for(let trial=0;trial<60;trial++){
  const n=2+trial%5,k=1+trial%n;
  const scores=Array.from({length:n},(_,i)=>2*random()-1+i*.07),experts=scores.map(()=>[4*random()-2,4*random()-2]);
  const direction=[random(),-random()];
  for(const normalized of [true,false]){
    const actual=route(scores,experts,k,normalized,direction);
    assert.ok(actual.margin===null||actual.margin>1e-5);
    for(let i=0;i<n;i++){
      const plus=[...scores],minus=[...scores];plus[i]+=1e-6;minus[i]-=1e-6;
      const loss=x=>route(x,experts,k,normalized,direction).output.reduce((s,v,j)=>s+v*direction[j],0);
      scalar(actual.gradient[i],(loss(plus)-loss(minus))/2e-6);
    }
    compare(actual.output,route(scores.map(s=>s+713),experts,k,normalized,direction).output);
    if(normalized)compare(route(scores,experts.map(()=>[2,-1]),k,true).output,[2,-1]);
  }
  // A per-expert ordered list gives an oracle with a different loop structure.
  const tokens=3+trial%6, routes=Array.from({length:tokens},()=>{const a=Math.floor(random()*4);return[a,(a+1+Math.floor(random()*3))%4];});
  const order=Array.from({length:tokens},(_,i)=>i).sort(()=>random()-.5),capacity=1+trial%4;
  const gates=routes.map(()=>{const w=random();return[w,1-w];}),values=[-2,.5,3,1];
  const accepted=new Set();
  for(let e=0;e<4;e++)order.flatMap(t=>routes[t].flatMap((v,s)=>v===e?[[t,s]]:[])).slice(0,capacity).forEach(([t,s])=>accepted.add(`${t}:${s}`));
  for(const renormalize of [false,true]){
    const expected=routes.map((row,t)=>{const pairs=row.map((e,s)=>[accepted.has(`${t}:${s}`)?gates[t][s]:0,values[e]]),mass=pairs.reduce((s,[w])=>s+w,0),sum=pairs.reduce((s,[w,v])=>s+w*v,0);return renormalize&&mass>0?sum/mass:sum;});
    compare(dispatch(routes,capacity,{order,gates,values,renormalize}).output,expected);
  }
  compare(dispatch(routes,Infinity,{order,gates,values}).output,dispatch(routes,Infinity,{gates,values}).output);
  const rows=Array.from({length:5},()=>Array.from({length:3},()=>random()*4-2));
  const before=balance(rows,1),after=balance(rows.map(r=>r.map(v=>v+2.4)),1);
  compare(before.probabilities,after.probabilities);compare(before.counts,after.counts);scalar(before.value,after.value);
  scalar(after.zLoss,before.zLoss+4.8*before.partitions.reduce((s,v)=>s+v,0)/5+2.4**2);
}
const fixtures=JSON.parse(fs.readFileSync(`${directory}/independent-forward-fixtures.json`));
const state=JSON.parse(fs.readFileSync('public/learn-assets/mixture-of-experts-transformers-moe/moe-001-17.json'));
const saved=JSON.stringify(state);
assert.equal(fixtures.cases.length,10);
for(const c of fixtures.cases){const r=frozenMoE(state,c.pixels,c);compare(r.logits,c.expected.logits,1e-10);compare(r.combined,c.expected.combined,1e-10);compare(r.classProbabilities,c.expected.probabilities,1e-10);assert.deepEqual(r.selected,c.expected.selected);}
assert.equal(JSON.stringify(state),saved);
// Permuting expert identities with their router rows is an exact architectural symmetry.
const permutation=[2,0,3,1],renamed=structuredClone(state);
renamed['router.weight']=permutation.map(i=>state['router.weight'][i]);
for(let i=0;i<4;i++)for(const part of ['gate','value','down'])renamed[`experts.${i}.${part}.weight`]=state[`experts.${permutation[i]}.${part}.weight`];
const pixels=fixtures.cases[0].pixels;
compare(frozenMoE(state,pixels).logits,frozenMoE(renamed,pixels).logits,1e-10);
const b={d:4096,m:4096,n:256,k:256,tokens:4096,bytes:8,remote:1,shared:4},c=costs(b);
scalar(c.stored,Number(3n*4096n*4096n*260n));scalar(c.weightBytes,Number((3n*4096n*4096n*260n+4096n*256n)*8n));scalar(c.payload,Number(2n*4096n*256n*4096n*8n));
const evidence={passed:true,reviewer:'/root',comparisons:count,maximumError:maximum,fullModelCases:10,checks:['60 fresh local-gradient cases per normalizer with central differences','Independent per-expert capacity oracle, reweighting and dropless-order invariance','Log-normalizer shift identity and balancing invariance','Ten independent functional PyTorch SDPA/dense-expert model cases','Expert/router-row permutation symmetry and input-weight immutability','BigInt-derived largest integer budget controls'],limitations:'No additional fitting or distributed timing claim.'};
fs.writeFileSync(`${directory}/independent-model-checks.json`,JSON.stringify(evidence,null,2)+'\n');console.log(JSON.stringify(evidence));
