import fs from 'node:fs';
import assert from 'node:assert/strict';
import { spawnSync } from 'node:child_process';
import * as model from '../src/learn/data/random-variables-models.js';
import { randomVariableExamples } from '../src/learn/data/random-variables-examples.js';

const fixtures = {outcome:[],loss:[],joint:[],noise:[],conditional:[],squared:[],sample:[],finite:[]};
for (const p of [0,1,13,25,50,87,99,100]) for (const q of [0,1,23,50,99,100]) for (const rule of ['heads','first','equal']) {
  fixtures.outcome.push({p,q,rule,state:model.outcomeState(p,q,rule)});
}
for (const preset of ['asymmetric','symmetric','constant']) for (let c=-12;c<=16;c++) fixtures.loss.push({preset,c:c/4,state:model.meanLossState(c/4,preset)});
for (const preset of ['matching','opposite','independent','nonlinear']) for(let scale=-2;scale<=2;scale++) for(let shift=-3;shift<=3;shift++) {
  fixtures.joint.push({preset,scale,shift,state:model.jointState(preset,scale,shift)});
}
for (const common of [0,.5,1,2,3]) for (const local of [0,.5,1,2]) for (const a of [-1,-.5,0,.25,.5,1]) for (const b of [-1,-.5,0,.25,.5,1]) fixtures.noise.push({common,local,a,b,state:model.sharedNoiseState(common,local,a,b)});
for(let p=0;p<=100;p++) fixtures.conditional.push({p,state:model.conditionalState(p)});
for(let lower=0;lower<=100;lower++) for(let upper=lower;upper<=100;upper++) fixtures.squared.push({lower,upper,state:model.squaredUniformState(lower,upper)});
for(let n=1;n<=16;n++) for(let p=0;p<=100;p++) fixtures.sample.push({n,p,state:model.sampleMeanState(n,p)});
for (const values of [[-3,0,7],[1,1,1],[.1,.1,.1],[-1000,999,1000],[0,1e-130,2e-130]]) for (const masses of [[.1,.2,.7],[0,.4,.6],[1,0,0]]) {
  fixtures.finite.push({values,masses,state:model.finiteMoments(values,masses)});
}
const tiny = model.pairedMoments([{x:0,y:0,mass:.5},{x:1e-130,y:1e-130,mass:.5}]);
assert.equal(tiny.correlation,1);
assert.equal(model.finiteMoments([.1,.1,.1],[.1,.2,.7]).variance,0);
assert.equal(model.pairedMoments([{x:1,y:.1,mass:.3},{x:2,y:.1,mass:.7}]).correlation,null);
const rejected = [
  () => model.finiteMoments([],[]), () => model.finiteMoments([0],[.9]),
  () => model.finiteMoments([0,1],[1-1e-13,1e-13]), () => model.finiteMoments([NaN],[1]),
  () => model.finiteMoments([1001],[1]), () => model.finiteMoments([0,1e-200],[.5,.5]),
  () => model.finiteMoments([0,1e-160],[.5,.5]),
  () => model.pairedMoments([{x:1,y:Infinity,mass:1}]),
  () => model.outcomeState(-1), () => model.outcomeState(50,100.5), () => model.outcomeState(50,50,'bad'),
  () => model.meanLossState(5), () => model.meanLossState(0,'bad'),
  () => model.jointState('bad'), () => model.jointState('matching',.5), () => model.jointState('matching',1,4),
  () => model.sharedNoiseState(4), () => model.sharedNoiseState(2,-1), () => model.sharedNoiseState(2,1,Infinity),
  () => model.conditionalState(1.5), () => model.squaredUniformState(81,25),
  () => model.squaredUniformState(-1,50), () => model.squaredUniformState(1.2,50),
  () => model.sampleMeanState(0), () => model.sampleMeanState(17), () => model.sampleMeanState(2,NaN),
];
for(const reject of rejected) assert.throws(reject,RangeError);
assert(Object.isFrozen(fixtures.outcome[0].state.outcomes[0]));
assert.throws(()=>{fixtures.joint[0].state.rows[0].x=7;},TypeError);
const values=[0,1],masses=[.4,.6];model.finiteMoments(values,masses);assert(!Object.isFrozen(values));assert.deepEqual(masses,[.4,.6]);
const directory='scratch/random-variables-verification';fs.mkdirSync(directory,{recursive:true});
fs.writeFileSync(directory+'/model-fixtures.json',JSON.stringify(fixtures));
fs.writeFileSync(directory+'/actual-examples.json',JSON.stringify(randomVariableExamples));
const run=spawnSync('scratch/lesson-tools/Scripts/python.exe',['scripts/verify-random-variables-native.py'],{encoding:'utf8'});
process.stdout.write(run.stdout);process.stderr.write(run.stderr);assert.equal(run.status,0);
fs.writeFileSync(directory+'/contract-results.json',JSON.stringify({checkedAt:new Date().toISOString(),counts:Object.fromEntries(Object.entries(fixtures).map(([key,value])=>[key,value.length])),rejected:rejected.length,readonly:true,tinyCorrelation:tiny.correlation},null,2));
