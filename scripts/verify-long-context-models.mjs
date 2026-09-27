import assert from 'node:assert/strict';
import fs from 'node:fs';
import crypto from 'node:crypto';
import { performance } from 'node:perf_hooks';
import { longContextDefaults, retainedRead, recurrenceTrace, latentRead, trajectoryForward, meanTrajectory, composeAffine } from '../src/learn/data/long-context-models.js';

const id = 'long-context-sequence-models-transformer-xl-griffin-perceiver';
const asset = `public/learn-code/${id}/`, draft = `docs/teaching/drafts/${id}/`, evidence = 'docs/teaching/evidence/long-context/';
fs.mkdirSync(evidence, { recursive: true });
fs.writeFileSync(evidence+'model-checks.json', JSON.stringify({ status: 'incomplete', note: 'A run started; only a subsequent passed report is successful evidence.' }, null, 2)+'\n');
const close = (a, b, tolerance = 1e-10) => assert.ok(Math.abs(a - b) <= tolerance, `${a} != ${b} (tol ${tolerance})`);
const source = longContextDefaults();
for (const [memory, expected] of [[0, 0], [2, 10/3], [4, 5]]) close(retainedRead(source, 2, memory, 4).output, expected);
for (let length = 1; length <= 16; length++) for (let segment = 1; segment <= 8; segment++) for (const memory of [0, 1, 4, 16]) {
  const rows = Array.from({ length }, (_, i) => ({ id: i, key: Math.sin(i), value: Math.cos(i) }));
  for (let query = 0; query < length; query++) {
    const result = retainedRead(rows, segment, memory, query, .37), start = Math.floor(query/segment)*segment;
    assert.deepEqual(result.legalIds, Array.from({ length: query-Math.max(0,start-memory)+1 }, (_, j) => Math.max(0,start-memory)+j));
    const changed = rows.map((r,i) => ({ ...r, value: i > query ? 8 : r.value }));
    close(result.output, retainedRead(changed,segment,memory,query,.37).output);
    close(retainedRead(rows.map(r => ({ ...r, value: 7 })),segment,memory,query,.37).output, 7);
  }
}
assert.throws(() => retainedRead([],2,2,0));
assert.throws(() => retainedRead(source,2,4,4,0,source.map(r=>r.id)));
const impulse = Array.from({length:6},(_,i)=>({x:i?0:1,input:1,recurrence:.125}));
close(recurrenceTrace(impulse).at(-1).state,.196608);
close(recurrenceTrace(impulse.map((e,i)=>({...e,recurrence:i?.001:.125}))).at(-1).state,.5946683844778317);
close(recurrenceTrace(impulse.map((e,i)=>({...e,recurrence:i?0:.125}))).at(-1).state,.6);
close(recurrenceTrace(impulse.map((e,i)=>({...e,input:i?0:1}))).at(-1).state,.196608);
for (const initial of [-3,0,3]) for (const base of [.05,.8,.999]) {
  const trace = recurrenceTrace(Array.from({length:16},(_,i)=>({x:Math.cos(i)*3,input:i/15,recurrence:(15-i)/15})),base,initial);
  for(const row of trace) close(row.state,row.initialContribution+row.contributions.reduce((a,b)=>a+b,0));
}
const records = [-1,0,1].map((position,i)=>({position,value:2+4*i})), queries = [-Math.log(2),Math.log(2)];
const latent = latentRead(records,queries);
close(latent[0].output,30/7);close(latent[1].output,54/7);
latentRead([records[2],records[0],records[1]],queries).forEach((r,i)=>close(r.output,latent[i].output));
close(latentRead(records,[0])[0].output,latentRead(records.map((r,i)=>({...r,value:6*i})),[0])[0].output);
composeAffine([.4,1.56],composeAffine([.4,2],[.5,1])).forEach((value,i)=>close(value,composeAffine(composeAffine([.4,1.56],[.4,2]),[.5,1])[i]));

const data = JSON.parse(fs.readFileSync(asset+'trajectory-models.json')), oracle = JSON.parse(fs.readFileSync(evidence+'native-validation-logits.json'));
const fixture = JSON.parse(fs.readFileSync(draft+'investigation-checks.json')).trajectory_fixture;
assert.equal(data.specimens.length,50);assert.equal(new Set(data.specimens.map(s=>s.sourceRow)).size,50);
const nativeReport = JSON.parse(fs.readFileSync(draft+'trajectory-results.json'));
assert.deepEqual(data.specimens.map(s=>s.sourceRow),nativeReport.data_roles.validation_source_rows);
let maxLogitError=0,maxWeightSumError=0,maxNullError=0, cases=0, timing=[];
for(const [name,parameters] of Object.entries(data.models)) for(const [index,specimen] of data.specimens.entries()) {
  const before=performance.now(), current=trajectoryForward(parameters,specimen.records);timing.push(performance.now()-before);
  current.logits.forEach((value,i)=>{const error=Math.abs(value-oracle[name][index][i]);maxLogitError=Math.max(maxLogitError,error);close(value,oracle[name][index][i],5e-5);});
  current.weights.forEach(row=>{maxWeightSumError=Math.max(maxWeightSumError,Math.abs(row.reduce((a,b)=>a+b,0)-1));});
  const reversed=trajectoryForward(parameters,[...specimen.records].reverse());
  const padded=trajectoryForward(parameters,[...specimen.records,...Array.from({length:5},(_,i)=>({id:46+i,x:500.5,y:500.5,position:1000,valid:false}))]);
  current.logits.forEach((value,i)=>{maxNullError=Math.max(maxNullError,Math.abs(value-reversed.logits[i]),Math.abs(value-padded.logits[i]));close(value,reversed.logits[i],1e-9);close(value,padded.logits[i],1e-9);});
  current.weights.forEach(row=>close(row.reduce((a,b)=>a+b,0),1));
  const mean=meanTrajectory(data.mean,specimen.records);close(mean.probabilities.reduce((a,b)=>a+b,0),1);
  cases++;
}
const specimen=data.specimens.find(s=>s.sourceRow===7), model=data.models.latents4_seed29;
const full=trajectoryForward(model,specimen.records), first=trajectoryForward(model,specimen.records.map(r=>({...r,valid:r.id===1}))), moved=trajectoryForward(model,specimen.records.map(r=>r.id===23?{...r,x:1-r.x}:r));
assert.equal(full.predictedClass,1);assert.equal(first.predictedClass,10);assert.equal(moved.predictedClass,1);
close(full.probabilities[0],.7806530594825745,2e-6);close(first.probabilities[0],.0000709838132,2e-7);close(moved.probabilities[0],.7807971239,2e-6);
assert.throws(()=>trajectoryForward(model,specimen.records.map(r=>({...r,valid:false}))));
assert.throws(()=>trajectoryForward(model,[{x:NaN,y:0,position:0,valid:true}]));
for(const name of ['sequence_mechanisms.py','latent_trajectory_classifier.py','memory_library_bridge.py','movement_libras.data','small-fits.npz']) assert.ok(fs.readFileSync(asset+name).equals(fs.readFileSync(draft+name)),`Altered canonical asset: ${name}`);
const files = ['src/learn/data/long-context-models.js','src/learn/components/lesson-labs/LongContextLabs.jsx','src/learn/components/lesson-labs/LongContextTrajectoryLab.jsx','src/learn/components/lesson-labs/LongContextFigures.jsx','src/learn/components/lesson-labs/long-context-labs.css',`src/learn/data/topics/${id}.jsx`,'scripts/render-long-context-lesson.mjs','scripts/verify-long-context-models.mjs',asset+'trajectory-models.json',evidence+'native-validation-logits.json',draft+'small-fits.npz',draft+'investigation-checks.json'];
const hashes=Object.fromEntries(files.map(file=>[file,crypto.createHash('sha256').update(fs.readFileSync(file)).digest('hex')]));
const report={status:'passed',cases,maxLogitError,maxWeightSumError,maxNullError,default:{class:full.predictedClass,pClass1:full.probabilities[0]},firstPoint:{class:first.predictedClass,pClass1:first.probabilities[0]},smallEdit:{class:moved.predictedClass,pClass1:moved.probabilities[0]},nodeInferenceMs:{median:timing.sort((a,b)=>a-b)[Math.floor(timing.length/2)],max:Math.max(...timing),note:'Local Node CPU measurement, not browser or architecture benchmark'},reviewedFiles:hashes};
fs.writeFileSync(evidence+'model-checks.json',JSON.stringify(report,null,2)+'\n');console.log(JSON.stringify(report,null,2));
