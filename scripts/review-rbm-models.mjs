// Independent reviewer: direct scalar joint-state sums, fresh inputs, no author fixtures as oracle.
import fs from 'node:fs';
import assert from 'node:assert/strict';
import crypto from 'node:crypto';
import { completion, freeEnergy, hiddenEnumeration, hiddenProbabilities, probabilityFlow, statistics, tinyDistribution, tinyTransition, updateTiny, persistenceTrace, exactSample } from '../src/learn/data/rbm-models.js';
const id='boltzmann-machines-restricted-boltzmann-machines-rbm', folder=`docs/teaching/deep-learning-completion/${id}`;
let randomState=291127, comparisons=0, maximumError=0;
const random=()=>{randomState=(Math.imul(randomState,1664525)+1013904223)>>>0;return randomState/2**32;};
const enumerate=n=>Array.from({length:2**n},(_,v)=>Array.from({length:n},(_,i)=>Math.floor(v/2**(n-i-1))%2));
const dot=(a,b)=>a.reduce((s,v,i)=>s+v*b[i],0);
const sigmoid=x=>1/(1+Math.exp(-x));
const near=(a,b,t=2e-10)=>{if(Array.isArray(b)){assert.equal(a.length,b.length);b.forEach((v,i)=>near(a[i],v,t));}else{const e=Math.abs(a-b);assert.ok(Number.isFinite(e)&&e<t,`${a} != ${b}`);comparisons++;maximumError=Math.max(maximumError,e);}};
function joint(model, supplied=null, mask=null){
  const missing=mask?mask.flatMap((v,i)=>v?[]:[i]):model.a.map((_,i)=>i), entries=[];
  for(const suffix of enumerate(missing.length))for(const h of enumerate(model.b.length)){
    const v=supplied?[...supplied]:Array(model.a.length).fill(0);missing.forEach((index,i)=>v[index]=suffix[i]);
    let score=dot(v,model.a)+dot(h,model.b);
    for(let i=0;i<v.length;i++)for(let j=0;j<h.length;j++)score+=v[i]*model.w[i][j]*h[j];
    entries.push({v,h,score});
  }
  const max=Math.max(...entries.map(e=>e.score));let sum=0;for(const e of entries){e.mass=Math.exp(e.score-max);sum+=e.mass;}for(const e of entries)e.p=e.mass/sum;
  return {entries,logZ:max+Math.log(sum),v:model.a.map((_,i)=>entries.reduce((s,e)=>s+e.p*e.v[i],0)),h:model.b.map((_,i)=>entries.reduce((s,e)=>s+e.p*e.h[i],0))};
}
const checks=[], check=(name,fn)=>{const start=comparisons;fn();checks.push({name,passed:true,scalarComparisons:comparisons-start});};
check('64 fresh asymmetric 4-visible/3-hidden joint oracles, all/sparse/no evidence and missing-placeholder invariance',()=>{
  for(let caseId=0;caseId<64;caseId++){
    const model={a:Array.from({length:4},()=>8*random()-4),b:Array.from({length:3},()=>8*random()-4),w:Array.from({length:4},()=>Array.from({length:3},()=>8*random()-4))};
    const reference=joint(model), actual=hiddenEnumeration(model);near(actual.logZ,reference.logZ);
    for(const v of enumerate(4)){
      const mass=reference.entries.filter(e=>e.v.every((x,i)=>x===v[i])).reduce((s,e)=>s+e.p,0);
      near(Math.exp(-freeEnergy(v,model)-actual.logZ),mass);
      near(hiddenProbabilities(v,model),joint(model,v,Array(4).fill(true)).h);
    }
    const input=enumerate(4)[caseId%16];
    for(const mask of [Array(4).fill(true),Array(4).fill(false),[true,false,true,false],[false,true,false,true]]){
      const expected=joint(model,input,mask), output=completion(input,mask,model);
      near(output.probabilities,expected.v);near(output.posterior.reduce((a,b)=>a+b,0),1);
      const ignored=input.map((v,i)=>mask[i]?v:1-v);near(completion(ignored,mask,model).probabilities,output.probabilities);
    }
  }
});
check('48 fresh tiny gradients from independent scalar expectations and central differences; simultaneous update and exact transitions',()=>{
  for(let trial=0;trial<48;trial++){
    const theta=Array.from({length:5},()=>6*random()-3), model={w:[[theta[0]],[theta[1]]],a:theta.slice(2,4),b:[theta[4]]}, counts=Array.from({length:4},()=>random()*7), total=counts.reduce((a,b)=>a+b,0), data=counts.map(v=>v/total), visible=enumerate(2), ref=joint(model);
    const expected=[0,1].map(i=>visible.reduce((s,v,j)=>s+data[j]*v[i]*sigmoid(model.b[0]+v[0]*theta[0]+v[1]*theta[1]),0)-ref.entries.reduce((s,e)=>s+e.p*e.v[i]*e.h[0],0));
    expected.push(...[0,1].map(i=>visible.reduce((s,v,j)=>s+data[j]*v[i],0)-ref.v[i]),visible.reduce((s,v,j)=>s+data[j]*sigmoid(model.b[0]+v[0]*theta[0]+v[1]*theta[1]),0)-ref.h[0]);
    const actual=statistics(model,counts);near(actual.gradient,expected);
    const objective=parameters=>{const m={w:[[parameters[0]],[parameters[1]]],a:parameters.slice(2,4),b:[parameters[4]]}, d=joint(m);return visible.reduce((s,v,i)=>s+data[i]*Math.log(d.entries.filter(e=>e.v.every((x,j)=>x===v[j])).reduce((x,e)=>x+e.p,0)),0);};
    for(let k=0;k<5;k++){const plus=[...theta],minus=[...theta];plus[k]+=1e-5;minus[k]-=1e-5;near(actual.gradient[k],(objective(plus)-objective(minus))/2e-5,2e-8);}
    const updated=updateTiny(model,actual.gradient,.013);near([...updated.w.flat(),...updated.a,...updated.b],theta.map((v,i)=>v+.013*expected[i]));
    const t=visible.map(v=>visible.map(next=>[0,1].reduce((sum,h)=>{const ph=sigmoid(model.b[0]+v[0]*theta[0]+v[1]*theta[1]);let p=h?ph:1-ph;for(let i=0;i<2;i++){const on=sigmoid(model.a[i]+theta[i]*h);p*=next[i]?on:1-on;}return sum+p;},0)));
    near(tinyTransition(model),t);near(probabilityFlow(model,tinyDistribution(model).p,11).traces.at(-1).mass,tinyDistribution(model).p);
    for(const offset of [-100,100])near(tinyDistribution(model,offset).p,tinyDistribution(model).p);
  }
});
check('All nine real fits: new three-missing-bit conditionals checked by independent visible-plus-hidden joint enumeration',()=>{
  const images=JSON.parse(fs.readFileSync(`public/learn-code/${id}/assessment-images.json`));
  for(const [index,name] of ['exact-11','exact-29','exact-47','cd1-11','cd1-29','cd1-47','pcd1-11','pcd1-29','pcd1-47'].entries()){
    const {model}=JSON.parse(fs.readFileSync(`public/learn-code/${id}/model-${name}.json`)), input=[...images[7+index*7].pixels], missing=[(index*3+5)%64,(index*3+19)%64,(index*3+43)%64], mask=input.map((_,i)=>!missing.includes(i));
    input[(index*3+4)%64]=1-input[(index*3+4)%64];const expected=joint(model,input,mask), actual=completion(input,mask,model);near(actual.probabilities,expected.v,1e-10);
    const posterior=enumerate(8).map(h=>expected.entries.filter(e=>e.h.every((v,i)=>v===h[i])).reduce((s,e)=>s+e.p,0));near(actual.posterior,posterior,1e-10);
  }
});
check('Persistent-state causality, exact-sample distribution and clamped observed bits on fresh constructed model',()=>{
  const model={w:[[1.7],[-.8]],a:[.3,-.2],b:[.9]}, a=[[0,0],[1,0],[1,1],[0,1]], b=[[1,1],[1,1],[1,1],[1,1]], c=[[0,0],[0,0],[0,0],[0,0]];
  const first=persistenceTrace(model,[a,b],421),second=persistenceTrace(model,[a,c],421);assert.deepEqual(first[1].pcd,second[1].pcd);assert.notDeepEqual(first[1].cd.map(e=>e.previous),second[1].cd.map(e=>e.previous));
  const samples=exactSample(model,1623,12000), frequency=[0,0,0,0];for(const sample of samples)frequency[2*sample.pixels[0]+sample.pixels[1]]++;
  const oracle=joint(model), target=enumerate(2).map(v=>oracle.entries.filter(e=>e.v.every((x,i)=>x===v[i])).reduce((s,e)=>s+e.p,0));near(frequency.map(v=>v/12000),target,.022);
  const draws=exactSample(model,23,100,[1,0],[true,false]);assert.ok(draws.every(row=>row.pixels[0]===1));
});
const receipt={passed:true,reviewer:'/root/graph_attention_implementation',checks,scalarComparisons:comparisons,maximumObservedErrorAcrossDifferentTolerances:maximumError,trustRoot:'Independently written scalar joint-state enumeration and central differences; does not call production free-energy/conditional functions for reference values',reusedAuthorEvidence:['native-checks.json (52 actual checks)','model-checks.json (281 actual checks)'],limitations:['No training campaigns repeated','Browser and painted geometry remain root scope']};
receipt.sourceHashes=Object.fromEntries(['src/learn/data/rbm-models.js','scripts/review-rbm-models.mjs'].map(path=>[path,crypto.createHash('sha256').update(fs.readFileSync(path)).digest('hex')]));
fs.writeFileSync(`${folder}/independent-model-checks.json`,JSON.stringify(receipt,null,2)+'\n');console.log(JSON.stringify(receipt));
