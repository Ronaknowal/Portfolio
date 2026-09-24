import fs from 'node:fs';
import path from 'node:path';
import assert from 'node:assert/strict';
import { pathToFileURL } from 'node:url';
import { spawnSync } from 'node:child_process';
import * as model from '../src/learn/data/random-variables-models.js';
import { randomVariableExamples } from '../src/learn/data/random-variables-examples.js';

const directory='scratch/random-variables-independent';
const prior=await import(pathToFileURL(path.resolve(directory,'author-sources/random-variables-models.js')).href);
let conserved=0;
function same(name,...args) { assert.deepEqual(model[name](...args),prior[name](...args)); conserved++; }
for(let p=0;p<=100;p+=5) for(let q=0;q<=100;q+=10) for(const mapping of ['heads','first','equal']) same('outcomeState',p,q,mapping);
for(const preset of ['asymmetric','symmetric','constant']) for(let c=-12;c<=16;c++) same('meanLossState',c/4,preset);
for(const preset of ['matching','opposite','independent','nonlinear']) for(let a=-2;a<=2;a++) for(let b=-3;b<=3;b++) same('jointState',preset,a,b);
for(let common=0;common<=3;common+=.5) for(let local=0;local<=2;local+=.5) for(const a of [-1,-.5,0,.25,1]) for(const b of [-1,0,.5,1]) same('sharedNoiseState',common,local,a,b);
for(let p=0;p<=100;p++) same('conditionalState',p);
for(let a=0;a<=100;a++) for(let b=a;b<=100;b++) same('squaredUniformState',a,b);
for(let n=1;n<=16;n++) for(let p=0;p<=100;p++) same('sampleMeanState',n,p);
let sparseRejected=0;
for(let length=1;length<=8;length++) for(let missing=0;missing<length;missing++) {
  const values=Array(length).fill(1), masses=Array(length).fill(1/length), rows=values.map((x,i)=>({x,y:2*x,mass:masses[i]}));
  delete values[missing]; delete rows[missing];
  assert.throws(()=>model.finiteMoments(values,masses),RangeError);
  assert.throws(()=>model.pairedMoments(rows),RangeError);
  const sparseMasses=masses.slice(); delete sparseMasses[missing];
  assert.throws(()=>model.finiteMoments(Array(length).fill(1),sparseMasses),RangeError);
  sparseRejected+=3;
}
const inherited=[,1]; Object.setPrototypeOf(inherited,Object.assign(Object.create(Array.prototype),{0:0}));
assert.throws(()=>model.finiteMoments(inherited,[.5,.5]),RangeError); sparseRejected++;
const finite=[],paired=[],noise=[],samples=[],conditional=[];
for(const values of [[-9,-2,5],[.1,.3,.7],[999,999.25,1000],[0,1e-130,2e-130]]) for(const masses of [[.125,.375,.5],[0,.25,.75],[.5,.5,0]]) finite.push({values,masses,state:model.finiteMoments(values,masses)});
for(const x of [[-3,1,4],[.1,.2,.4],[0,1e-130,2e-130]]) for(const y of [[-1,5,2],[.1,.1,.1],[0,-1e-130,-2e-130]]) for(const masses of [[.25,.25,.5],[0,.5,.5]]) {
  const rows=x.map((value,i)=>({x:value,y:y[i],mass:masses[i]})); paired.push({rows,state:model.pairedMoments(rows)});
}
for(const common of [1.3,2.7]) for(const local of [.3,1.7]) for(const a of [-.7,.35,.8]) for(const b of [-.2,.6]) noise.push({common,local,a,b,state:model.sharedNoiseState(common,local,a,b)});
for(const n of [1,3,7,10]) for(const percent of [0,17,35,100]) samples.push({n,percent,state:model.sampleMeanState(n,percent)});
for(const p of [0,17,39,71,100]) conditional.push({p,state:model.conditionalState(p)});
const data={conserved,sparseRejected,finite,paired,noise,samples,conditional,examples:randomVariableExamples};
fs.writeFileSync(`${directory}/cases.json`,JSON.stringify(data));
const run=spawnSync(path.resolve('scratch/lesson-tools/Scripts/python.exe'),['scripts/verify-random-variables-independent.py'],{encoding:'utf8'});
process.stdout.write(run.stdout); process.stderr.write(run.stderr); assert.equal(run.status,0);
