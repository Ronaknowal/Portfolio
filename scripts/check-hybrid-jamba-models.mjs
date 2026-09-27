import fs from 'node:fs';
import assert from 'node:assert/strict';
import { hybridMemoryRead, hybridCacheBytes, hybridAttentionOperations, hybridExpertParameters, hybridRoute, hybridStrokeStream, hybridStrokeTrace } from '../src/learn/data/hybrid-jamba-models.js';
const id='hybrid-ssm-transformer-architectures-jamba',base=`docs/teaching/deep-learning-completion/${id}`,fixtures=JSON.parse(fs.readFileSync(`${base}/native-fixtures.json`)),models=new Map();
let comparisons=0,maximumDifference=0;const checks=[];
function close(a,b,tolerance=2e-10,label='comparison'){
 if(Array.isArray(b)){assert.equal(a.length,b.length,label);b.forEach((v,i)=>close(a[i],v,tolerance,`${label}[${i}]`));return;}
 if(b&&typeof b==='object'){for(const key of Object.keys(b))close(a[key],b[key],tolerance,`${label}.${key}`);return;}
 if(typeof b!=='number'){assert.equal(a,b,label);return;}
 assert.ok(Number.isFinite(a),label);const difference=Math.abs(a-b);maximumDifference=Math.max(maximumDifference,difference);assert.ok(difference<=tolerance*(1+Math.abs(b)),`${label}: ${a} != ${b}; ${difference}`);comparisons++;
}
function model(key){if(!models.has(key))models.set(key,JSON.parse(fs.readFileSync(`public/learn-assets/${id}/model-${key}.json`)));return models.get(key);}
for(const row of fixtures.traces){const result=hybridStrokeTrace(model(row.key),row.coordinates,row.boundary,row.mode,row.donor);for(const field of ['full','branch','probabilities','cache'])close(result[field],row[field],2e-10,`${row.key}/${row.label}/${row.mode}/${row.boundary}/${field}`);}
const nativeDoubleMaximum=maximumDifference;
checks.push({name:'180 actual native float64 full/branch/cache cases across all five models and six interventions',passed:true});
for(const row of fixtures.gradients){const snapshot=model(row.key),epsilon=1e-4;for(let t=0;t<8;t++)for(let axis=0;axis<2;axis++){const lower=structuredClone(row.coordinates),upper=structuredClone(row.coordinates);lower[t][axis]-=epsilon;upper[t][axis]+=epsilon;const derivative=(hybridStrokeStream(snapshot,upper).logits.at(-1)[row.classIndex]-hybridStrokeStream(snapshot,lower).logits.at(-1)[row.classIndex])/(2*epsilon);close(derivative,row.rawGradient[t][axis],2e-6,'native autograd versus raw-coordinate central difference');}}
const throughGradientMaximum=maximumDifference;
checks.push({name:'80 raw-coordinate input derivatives versus native autograd',passed:true});
for(const row of fixtures.memory){const result=hybridMemoryRead(row.values,row.keys,row.query,row.decay,row.beta);close(result.states,row.result.states);close(result.weights,row.result.weights);close(result.summary,row.result.summary);close(result.attention,row.result.attention);close(result.recurrenceWeights.reduce((s,v)=>s+v,0),1);}
for(const row of fixtures.routing){const result=hybridRoute(row.logits,row.values,row.k,row.renormalize);for(const [a,b] of [['probabilities','probabilities'],['selected','selected'],['weights','weights'],['output','output'],['selectedMass','selected_mass'],['boundaryTie','boundary_tie']])close(result[a],row.result[b]);}
for(const row of fixtures.budgets){const result=hybridCacheBytes(row);for(const field of ['kv','recurrent','convolution','total'])assert.equal(result[field],BigInt(row.result[field]));comparisons+=4;}
const large=hybridAttentionOperations({batch:32,width:8192,kvHeads:64,headWidth:256,length:262144});assert.equal(large.promptPairs,2n*32n*8192n*262144n*262145n);assert.ok(large.promptPairs>BigInt(Number.MAX_SAFE_INTEGER));
const ffn=hybridExpertParameters();assert.equal(ffn.stored,47915728896n);assert.equal(ffn.active,8455716864n);assert.equal(ffn.dense,5637144576n);
checks.push({name:'Native recurrence/router/storage fixtures, exact BigInt operations and stored/active SwiGLU counts',passed:true});
for(const snapshot of models.values()){
 const original=fixtures.traces.find(r=>r.key===snapshot.key&&r.label==='fresh').coordinates,prefix=hybridStrokeStream(snapshot,original.slice(0,3)),before=JSON.stringify(prefix.cache);hybridStrokeStream(snapshot,original.slice(3),{cache:prefix.cache,offset:3});assert.equal(JSON.stringify(prefix.cache),before);
 const changed=structuredClone(original);changed[6]=[0,100];close(hybridStrokeStream(snapshot,changed).logits.slice(0,6),hybridStrokeStream(snapshot,original).logits.slice(0,6),0,'future exclusion');
 const carry=hybridStrokeTrace(snapshot,original,3,'carry');close(carry.branch,carry.full,0,'exact same-order carry');
 for(const boundary of [0,8])for(const mode of ['recurrent-reset','kv-reset','convolution-reset','position-reset']){const fault=hybridStrokeTrace(snapshot,original,boundary,mode);close(fault.branch,fault.full,0,'empty history or suffix null');}
}
checks.push({name:'Request/branch immutability, future exclusion, exact carry and empty-boundary identities',passed:true});
const author=JSON.parse(fs.readFileSync(`docs/teaching/drafts/${id}/author-results.json`));
for(const name of ['worked','fresh'])for(const [mode,expected] of Object.entries(author.fixtures[name].traces)){const result=hybridStrokeTrace(model('MAM-37'),expected.coordinates,3,mode);close(result.branch,expected.branch_logits,1e-4,`retained native float32 ${name}/${mode}`);}
checks.push({name:'Prepared worked/fresh actual float32 fault fixtures reproduced within stated1e-4 tolerance',passed:true});
const downloads=JSON.parse(fs.readFileSync(`${base}/native-checks.json`)).downloadAllowlist;
for(const name of downloads)assert.ok(fs.readFileSync(`public/learn-assets/${id}/${name}`).equals(fs.readFileSync(`docs/teaching/drafts/${id}/${name}`)),`deployed copy ${name}`);
checks.push({name:'All ten deployed learner downloads exactly match canonical bytes',passed:true});
const receipt={passed:true,comparisons,maximumDifference,nativeDoubleMaximum,throughGradientMaximum,checks,limitations:['Double native comparisons use2e-10 scaled tolerance; retained float32 fixtures use the prepared1e-4 tolerance.','80 derivative comparisons use epsilon1e-4 and2e-6 scaled tolerance; they check the input path without training.']};fs.writeFileSync(`${base}/model-checks.json`,JSON.stringify(receipt,null,2)+'\n');console.log(JSON.stringify(receipt));
