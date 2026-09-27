import fs from 'node:fs';
import assert from 'node:assert/strict';
import { latentForecast, latentRead, latentDefaults, difference, latentBudget } from '../../../../src/learn/data/latent-attention-models.js';
const root='docs/teaching/deep-learning-completion/multi-head-latent-attention-mla';
const data=JSON.parse(fs.readFileSync('public/learn-assets/multi-head-latent-attention-mla/runtime.json'));
const inputs=[
  Array.from({length:7},()=>[.17,.83]),
  Array.from({length:11},(_,i)=>i%2?[.91,.04]:[.09,.96]),
  Array.from({length:9},(_,i)=>[.1+.8*i/8,.48+.3*Math.sin(i*.7)])
];
const cases=[];
for(const [shape,points] of inputs.entries()) for(const rank of [null,1,3,5,8]) for(const shift of [0,127]){
  const basis=rank===null?null:data.completeBasis.map(row=>row.slice(0,rank));
  const options={basis,shift},result=latentForecast(data.weights,points,options);
  // Different query-block sizes test rectangular causal masks in cached updates.
  let cache=null,predictions=[];
  for(let offset=0;offset<points.length;offset+=3){
    const part=latentForecast(data.weights,points.slice(offset,offset+3),{...options,cache});cache=part.cache;predictions.push(...part.predictions);
  }
  assert.ok(difference(predictions,result.predictions)<1e-12);
  cases.push({shape,rank,shift,points,predictions:result.predictions,fullLatents:result.fullLatents,cache:result.cache,lastHeads:result.traces.at(-1)});
}
const state=latentDefaults();state.positions=[0,7,2];state.queryPosition=3;
const before=latentRead(state);const future=structuredClone(state);future.latent[1]=[100,-100];future.keyRotary[1]=[-100,100];
assert.equal(difference(before.map(x=>x.output),latentRead(future).map(x=>x.output)),0);
const order=[2,0,1],permuted={...state,latent:order.map(i=>state.latent[i]),keyRotary:order.map(i=>state.keyRotary[i]),positions:order.map(i=>state.positions[i])};
assert.ok(difference(before.map(x=>x.output),latentRead(permuted).map(x=>x.output))<1e-12);
const budget={batch:64,layers:256,length:262144,heads:256,content:2048,value:2048,latent:2048,rotary:256,bytes:4,queries:262144};
const actual=latentBudget(budget);
assert.equal(BigInt(actual.compact),64n*256n*262144n*(2048n+256n)*4n);
assert.equal(BigInt(actual.absorbedOps),2n*64n*256n*262144n*262144n*(2n*2048n+256n));
const oddBudget={batch:63,layers:255,length:262143,heads:255,content:2047,value:2047,latent:2047,rotary:254,bytes:4,queries:262142};
assert.equal(latentBudget(oddBudget).exact.absorbedOps,9600086246924179440n);
fs.writeFileSync(root+'/independent-browser-values.json',JSON.stringify({cases,controls:{futureUnchanged:true,logicalRecordPermutation:true,extremeBudgetBigIntAgreement:true,chunkLength:3}}));
console.log('30 fresh browser-model cases; future exclusion, logical record permutation, chunked cache and BigInt count controls passed');
