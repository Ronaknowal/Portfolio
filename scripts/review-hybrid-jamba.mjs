import fs from 'node:fs';
import crypto from 'node:crypto';
import assert from 'node:assert/strict';
import {createRequire} from 'node:module';
import {build} from 'esbuild';
import * as m from '../src/learn/data/hybrid-jamba-models.js';
import * as displayed from '../src/learn/data/hybrid-jamba-study.js';
const id='hybrid-ssm-transformer-architectures-jamba',base=`docs/teaching/deep-learning-completion/${id}`,draft=`docs/teaching/drafts/${id}`,read=p=>JSON.parse(fs.readFileSync(p,'utf8')),clone=x=>structuredClone(x),sum=x=>x.reduce((a,b)=>a+b,0);
let checks=0,maxNative=0,maxGradient=0,maxSymmetry=0;
function near(a,b,tol=2e-9,label='numeric') {if(Array.isArray(a)){assert.equal(a.length,b.length,label);a.forEach((v,i)=>near(v,b[i],tol,label));return;}checks++;assert.ok(Number.isFinite(a)&&Number.isFinite(b)&&Math.abs(a-b)<=tol,`${label}: ${a} vs ${b}`);return Math.abs(a-b);}
function diff(a,b){return Math.max(...a.flat(Infinity).map((v,i)=>Math.abs(v-b.flat(Infinity)[i])));}
const native=read(`${base}/independent-native-fixtures.json`),models=Object.fromEntries([...new Set(native.records.map(r=>r.key))].map(k=>[k,read(`public/learn-assets/${id}/model-${k}.json`)]));
for(const r of native.records){
 const model=models[r.key],run=m.hybridStrokeStream(model,r.coordinates);maxNative=Math.max(maxNative,diff(run.logits,r.full));near(run.logits,r.full,3e-9,'fresh native full');
 let cache=null,offset=0,all=[];
 for(const count of [2,1,2,3]){const before=JSON.stringify(cache),part=m.hybridStrokeStream(model,r.coordinates.slice(offset,offset+count),{cache,offset});assert.equal(JSON.stringify(cache),before,'caller cache preserved');cache=part.cache;all.push(...part.logits);offset+=count;}
 near(all,r.chunked,3e-9,'four uneven chunks');
 for(const b of r.branches){const actual=m.hybridStrokeTrace(model,r.coordinates,b.boundary,b.mode);maxNative=Math.max(maxNative,diff(actual.branch,b.logits));near(actual.branch,b.logits,3e-9,'fresh native cache branch');}
 for(let p=0;p<8;p++)for(let axis=0;axis<2;axis++){const plus=clone(r.coordinates),minus=clone(r.coordinates),h=1e-4;plus[p][axis]+=h;minus[p][axis]-=h;const f=x=>sum(m.hybridStrokeStream(model,x).logits.at(-1).map((v,i)=>v*r.probe[i]));const fd=(f(plus)-f(minus))/(2*h);maxGradient=Math.max(maxGradient,Math.abs(fd-r.gradient[p][axis]));near(fd,r.gradient[p][axis],2e-6,'fresh native gradient');}
 const changed=clone(model),w=changed.state,stateOrder=[2,0,3,1],qOrder=Array.from({length:16},(_,i)=>(i*5+3)%16),vOrder=Array.from({length:16},(_,i)=>(i*7+1)%16);
 for(let l=0;l<model.pattern.length;l++){
  const prefix=`layers.${l}.mixer`;
  if(model.pattern[l]==='M'){
   for(const name of ['write.weight','read.weight','write_norm.weight','read_norm.weight']){const key=`${prefix}.${name}`,old=w[key];w[key]=stateOrder.map(i=>clone(old[i]));}
   w[`${prefix}.log_rates`]=w[`${prefix}.log_rates`].map(row=>stateOrder.map(i=>row[i]));
  }else{
   for(const name of ['qkv.weight','qkv.bias']){const key=`${prefix}.${name}`,old=w[key];w[key]=[...qOrder.map(i=>clone(old[i])),...qOrder.map(i=>clone(old[16+i])),...vOrder.map(i=>clone(old[32+i]))];}
   w[`${prefix}.output.weight`]=w[`${prefix}.output.weight`].map(row=>vOrder.map(i=>row[i]));
  }
 }
 const permuted=m.hybridStrokeStream(changed,r.coordinates);maxSymmetry=Math.max(maxSymmetry,diff(run.logits,permuted.logits));near(permuted.logits,run.logits,3e-9,'internal coordinate relabeling');
 const shifted=clone(model);shifted.state['classifier.bias']=shifted.state['classifier.bias'].map(x=>x+3);const out=m.hybridStrokeStream(shifted,r.coordinates);near(out.logits,run.logits.map(row=>row.map(x=>x+3)),3e-9,'common classifier shift');near(out.probabilities,run.probabilities,1e-12,'probability shift invariance');
}
for(const r of native.linear.cases)near(m.hybridStrokeStream(native.linear,r.coordinates).logits[0],r.logits,1e-10,'native flattened baseline');
let seed=34027;const rand=()=>((seed=(1664525*seed+1013904223)>>>0)/2**32);
for(let trial=0;trial<80;trial++){
 const n=1+Math.floor(rand()*8),values=Array.from({length:n},()=>20*rand()-10),keys=values.map(()=>Math.floor(3*rand())),query=Math.floor(4*rand()),decay=trial%4===0?0:trial%4===1?1:rand(),beta=rand()*Math.log(100),r=m.hybridMemoryRead(values,keys,query,decay,beta),powers=values.map((_,i)=>decay**(n-i-1)),recurrence=sum(values.map((v,i)=>v*powers[i]))/sum(powers),exp=keys.map(k=>k===query?Math.exp(beta):1),attention=sum(values.map((v,i)=>v*exp[i]))/sum(exp);
 near(r.summary,recurrence,1e-12,'closed-form recurrence');near(r.attention,attention,1e-12,'direct categorical mixture');near(sum(r.recurrenceWeights),1,1e-12);near(sum(r.weights),1,1e-12);
 const logits=Array.from({length:4},()=>16*rand()-8),vectors=Array.from({length:4},()=>[20*rand()-10,20*rand()-10]),k=trial%4+1,renormalize=trial%2===0,got=m.hybridRoute(logits,vectors,k,renormalize),ordered=[0,1,2,3].sort((a,b)=>logits[b]-logits[a]||a-b),mass=logits.map(Math.exp),denominator=sum((renormalize?ordered.slice(0,k):ordered).map(i=>mass[i])),expected=[0,1].map(j=>sum(ordered.slice(0,k).map(i=>mass[i]*vectors[i][j]))/denominator);
 assert.deepEqual(got.selected,ordered.slice(0,k));near(got.output,expected,1e-12);near(m.hybridRoute(logits.map(x=>x+1000),vectors,k,renormalize).output,expected,2e-11,'stable common router shift');
 const c={batch:trial%5+1,layers:12,attentionLayers:trial%13,width:32*(trial%7+1),kvHeads:trial%4+1,headWidth:8*(trial%3+1),stateSize:trial%9+1,expand:trial%3+1,convWidth:trial%5+1,kvBytes:2,stateBytes:4,convBytes:2,length:trial*71,window:trial%2?null:111};
 const bytes=m.hybridCacheBytes(c),retained=Math.min(c.length,c.window??Infinity);let kv=0n,recurrent=0n,convolution=0n;
 for(let layer=0;layer<c.layers;layer++)if(layer<c.attentionLayers)kv+=BigInt(c.batch*2*c.kvHeads*c.headWidth*retained*c.kvBytes);else{recurrent+=BigInt(c.batch*c.width*c.expand*c.stateSize*c.stateBytes);convolution+=BigInt(c.batch*c.width*c.expand*c.convWidth*c.convBytes);}
 assert.equal(bytes.kv,kv);assert.equal(bytes.recurrent,recurrent);assert.equal(bytes.convolution,convolution);assert.equal(bytes.total,kv+recurrent+convolution);checks+=4;
 const ops=m.hybridAttentionOperations({...c,length:trial});let pairs=0n;for(let t=1;t<=trial;t++)pairs+=BigInt(t);assert.equal(ops.causalPairs,pairs);assert.equal(ops.promptPairs,4n*BigInt(c.batch*c.width)*pairs);checks+=2;
}
const author=read(`${base}/implementation-checks.json`);for(const path of author.sourceFiles){assert.equal(crypto.createHash('sha256').update(fs.readFileSync(path)).digest('hex'),author.sourceHashes[path],`author source changed: ${path}`);checks++;}
for(const [path,hash]of Object.entries(author.evidenceHashes??{})){assert.equal(crypto.createHash('sha256').update(fs.readFileSync(path)).digest('hex'),hash);checks++;}
const study=Object.values(displayed).find(x=>x&&Array.isArray(x.fits)),source=read(`${draft}/stroke-results.json`);assert.ok(study);assert.equal(study.fits.length,6);
for(const f of study.fits){const original=source.fits.find(x=>x.key===f.key);assert.deepEqual(f.history,original.history);assert.deepEqual(f.metrics,original.metrics);assert.equal(f.selected_epoch,original.selected_epoch);assert.equal(f.selected_epoch,f.history.reduce((a,b)=>b.validation<a.validation?b:a).epoch);checks+=4;}
const bundle=await build({stdin:{contents:`import {createElement}from'react';import {renderToStaticMarkup}from'react-dom/server';import{RouterFigure}from'./src/learn/components/lesson-labs/HybridJambaLabs.jsx';export const html=renderToStaticMarkup(createElement(RouterFigure,{logits:[0,0,0,0],values:[[10,10],[10,10],[-10,-10],[-10,-10]],k:4}));`,resolveDir:process.cwd(),loader:'jsx'},bundle:true,format:'cjs',platform:'node',write:false,jsx:'automatic',loader:{'.css':'empty'},external:['react','react-dom/server']});
const module={exports:{}};new Function('require','module','exports',bundle.outputFiles[0].text)(createRequire(import.meta.url),module,module.exports);
const circles=[...module.exports.html.matchAll(/<circle[^>]*cx="([^"]+)"[^>]*cy="([^"]+)"/g)].map(x=>[Number(x[1]),Number(x[2])]);assert.equal(circles.length,4);for(const [x,y]of circles){assert.ok(x>=360&&x<=590&&y>=60&&y<=280,`router prefix outside common plot ${x},${y}`);checks++;}
const receipt={passed:true,checks,freshNativeInputs:10,freshNativeFaultBranches:150,inputGradientCoordinates:160,unevenChunkPartition:[2,1,2,3],linearNativeInputs:2,internalStateAndAttentionRelabelings:10,randomMechanismCases:80,maxNativeError:maxNative,maxGradientError:maxGradient,maxSymmetryError:maxSymmetry,allSixDisplayedHistoriesAndMetricsIdentical:true,sourceFilesVerified:author.sourceFiles.length,cancellingFourExpertPrefixGeometry:circles,scope:'Independent complementary tests against native full tensor forward/autograd plus cache faults, closed-form mechanisms and basis symmetries. Author native checkpoint/float32 evidence reused; no retraining, browser or large checkpoint execution.'};fs.writeFileSync(`${base}/independent-model-checks.json`,JSON.stringify(receipt,null,2)+'\n');console.log(JSON.stringify(receipt));
