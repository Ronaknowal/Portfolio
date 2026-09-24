import fs from 'node:fs';
import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { spawnSync } from 'node:child_process';
import * as models from '../src/learn/data/sampling-measurement-models.js';
import { samplingMeasurementExamples } from '../src/learn/data/sampling-measurement-examples.js';

const directory='scratch/sampling-measurement-independent';
const baseline=JSON.parse(fs.readFileSync(`${directory}/author-baseline.json`,'utf8'));
const original=JSON.parse(fs.readFileSync('scratch/sampling-measurement-verification/cases.json','utf8'));
let comparedNumbers=0, largestDifference=0;
function compare(actual,expected) {
  if(typeof expected==='number') {
    assert(Number.isFinite(actual));
    const difference=Math.abs(actual-expected); largestDifference=Math.max(largestDifference,difference); comparedNumbers++;
    assert(difference<=2e-11*Math.max(1,Math.abs(expected)),`${actual} != ${expected}`);
  } else if(Array.isArray(expected)) {
    assert.equal(actual.length,expected.length); expected.forEach((value,index)=>compare(actual[index],value));
  } else if(expected && typeof expected==='object') {
    assert.deepEqual(Object.keys(actual),Object.keys(expected));
    Object.entries(expected).forEach(([key,value])=>compare(actual[key],value));
  } else assert.equal(actual,expected);
}
const replay={
  samples:s=>models.finiteSampleState(s.population,s.eligible,s.size),
  inclusion:s=>models.inclusionDesignState(s.mode,s.values),
  readings:s=>models.groupedMeasurementState(s.units,s.repeats,s.unitVariance,s.readingVariance,s.bias),
  assignments:s=>models.assignmentState(s.design,s.effect,s.baseline),
  factorial:s=>models.factorialState(s.interaction,s.highBShare),
  missing:s=>({values:s.values,state:models.boundedMissingMean(s.values)})
};
let replayedStates=0;
for(const [name,states] of Object.entries(original.cases)) for(const state of states) {
  compare(replay[name](state),state); replayedStates++;
}
let invalidInputs=0;
const reject=fn=>{assert.throws(fn,RangeError);invalidInputs++;};
for(let length=1;length<=8;length++) for(let hole=0;hole<length;hole++) {
  const values=Array(length).fill(1); delete values[hole];
  const weights=Array(length).fill(1/length); delete weights[hole];
  reject(()=>models.finiteMoments(values));
  reject(()=>models.finiteMoments(Array(length).fill(1),weights));
  reject(()=>models.combinations(values,Math.min(1,length)));
  reject(()=>models.finiteSampleState(values,null,1));
  reject(()=>models.boundedMissingMean(values));
  const inherited=Array(length).fill(1);delete inherited[hole];
  const prototype=Object.create(Array.prototype);prototype[hole]=1;Object.setPrototypeOf(inherited,prototype);
  reject(()=>models.finiteMoments(inherited));
}
for(const probability of [0,false,{},'1',[],[.4,.4]]) reject(()=>models.finiteMoments([1,2],probability));
for(const name of ['toString','constructor','__proto__']) reject(()=>models.inclusionDesignState(name));
for(const call of [()=>models.finiteSampleState([1,2],false),()=>models.finiteSampleState([1,2],[,1]),()=>models.finiteMoments([0,1e-200]),()=>models.boundedMissingMean([undefined]),()=>models.finiteSampleState([0,1e-200],[0],1)])reject(call);
const cases={moments:[],samples:[],inclusion:[],assignments:[],readings:[],missing:[]};
for(const anchor of [0,1,-7,1e9,-1e9]) for(const weights of [[.5,.5],[.5,.5000000000005],[.1,.2,.7],[0,1]]) {
  const values=weights.map(()=>anchor),state=models.finiteMoments(values,weights);
  assert.equal(state.mean,anchor);assert.equal(state.variance,0);
  cases.moments.push({values,weights,state});
}
for(const scale of [2**-23,2**-12,.25,2])for(const offset of [1e9-16,1e6,0,-1e6])for(const weights of [[.5,.5],[.17,.35,.48],[.25,.25,.25,.25]]) {
  const values=weights.map((_,i)=>offset+i*scale);
  cases.moments.push({values,weights,state:models.finiteMoments(values,weights)});
}
for(const size of [3,5,8,12])for(let variant=0;variant<3;variant++) {
  const population=Array.from({length:size},(_,i)=>((i*i+3*variant*i-7)%17-8)/4);
  for(const frame of [population.map((_,i)=>i),population.map((_,i)=>i).filter(i=>i%2===0)]) {
    for(const n of [...new Set([1,Math.ceil(frame.length/2),frame.length])])cases.samples.push(models.finiteSampleState(population,frame,n));
  }
}
for(let i=0;i<25;i++)for(const mode of ['equal','unequal','uncovered'])cases.inclusion.push(models.inclusionDesignState(mode,[i-11,3*i-8,i%7-2,8-i]));
for(let i=0;i<17;i++)for(const design of ['complete','prognostic','mixed'])for(const effect of ['constant','heterogeneous'])cases.assignments.push(models.assignmentState(design,effect,[i-4,2*i+1,3-i,-i,7,i%5-8]));
for(const units of [3,7,13])for(const repeats of [2,5,11])cases.readings.push(models.groupedMeasurementState(units,repeats,1.3,2.7,-.75));
for(const values of [[null,null],[1,null,-2],[4,5],[-3,null,null,2]])cases.missing.push({values,state:models.boundedMissingMean(values,-4,8)});
const sources=baseline.sources.map(({path})=>({path,sha256:createHash('sha256').update(fs.readFileSync(path)).digest('hex')}));
fs.writeFileSync(`${directory}/cases.json`,JSON.stringify({cases,examples:samplingMeasurementExamples,sources,replayedStates,comparedNumbers,largestDifference,invalidInputs}));
const native=spawnSync('scratch/lesson-tools/Scripts/python.exe',['scripts/verify-sampling-measurement-independent.py'],{encoding:'utf8'});
process.stdout.write(native.stdout||'');process.stderr.write(native.stderr||'');
assert.equal(native.status,0);
