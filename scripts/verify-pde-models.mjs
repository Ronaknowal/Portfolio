import fs from 'node:fs';
import { spawnSync } from 'node:child_process';
import * as model from '../src/learn/data/pde-models.js';
import { pdeExamples } from '../src/learn/data/pde-examples.js';

const directory = 'scratch/pde-verification';
fs.mkdirSync(directory, { recursive: true });
const fixtures = { examples: pdeExamples, volume: [], transport: [], modes: [], heat: [], kernel: [], periodic: [], wave: [], standing: [], poisson: [], harmonic: [], weak: [], burgers: [], inverse: [], rod: [] };
for (const t of [0, .25, 1]) for (const [a,b] of [[0,1],[.25,.75],[.125,.875]]) fixtures.volume.push(model.controlVolume(t,a,b));
for (const t of [0,.25,1]) for (const [a,b] of [[.5,.5000000000000001],[.99,.9900000000000001]]) fixtures.volume.push(model.controlVolume(t,a,b));
for (const t of [0,.1,.35,.5,1]) for (const x of [0,.25,.6,1]) for (const c of [0,2]) fixtures.transport.push(model.transportState(t,x,c));
for (const boundary of ['dirichlet','neumann','mixed']) for (let n=boundary==='dirichlet'?1:0;n<=8;n++) fixtures.modes.push(model.boundaryMode(boundary,n));
for (const t of [0,.002,.01,.05,.2,.5]) fixtures.heat.push(model.heatBoundaryState(t));
for (const time of [.01,.1,.4,1,2]) for (const alpha of [.1,.5,2]) for (const x of [-20,-2,0,1,20]) fixtures.kernel.push({x,time,alpha,value:model.heatKernel(x,time,alpha)});
for (const depth of [0,.125,1,2,4]) for (const phase of [0,Math.PI/2,Math.PI,2*Math.PI]) fixtures.periodic.push(model.periodicDepth(depth,phase));
for (const t of [0,.01,.3,.6,1]) for (const x of [-3,-.4,0,.4,3]) for (const v of [0,.5]) fixtures.wave.push(model.waveState(t,x,v));
for (const t of [0,.125,.25,.4,.5,1,2]) fixtures.standing.push(model.standingWave(t));
for (const source of Object.keys(model.POISSON_SOURCES)) for (const boundary of ['dirichlet','neumann']) {
  for (let left=-2;left<=2;left++) for (let right=-2;right<=2;right++) for (const mean of [-1,0,1]) {
    fixtures.poisson.push(model.poissonState(source,boundary,left,right,mean));
  }
}
for (const n of [1,3,5]) fixtures.harmonic.push(model.harmonicState(n));
for (const a of [.1,.25,1/3,.6,.9]) fixtures.weak.push(model.pointSourceState(a));
for (const left of [-2,0,2,3]) for (const right of [-2,0,2,3]) for (const t of [0,.125,.5,1]) for (const expansion of [false,true]) fixtures.burgers.push(model.burgersState(left,right,t,expansion));
for (const n of [1,2,6,12]) for (const t of [.002,.01,.03,.05]) fixtures.inverse.push(model.inverseHeatState(n,t));
for (const t of [0,.1,1,20]) for (const length of [.25,.7,1,2]) for (const amplitude of [0,.02,.2,1]) fixtures.rod.push(model.forcedRod(t,amplitude,.05,length));
for (const tolerance of [.001,.025,.2]) for (const length of [.25,2]) for (const amplitude of [0,.025,.4,1]) fixtures.rod.push(model.forcedRod(0,amplitude,tolerance,length));

const invalid = [
  () => model.controlVolume(0,.5,.5), () => model.controlVolume(NaN),
  () => model.transportState(-1), () => model.transportState(0,0,1),
  () => model.heatValue(.5,.0001), () => model.heatBoundaryState(Infinity),
  () => model.boundaryMode('dirichlet',0), () => model.boundaryMode('mixed',.5),
  () => model.heatCoefficient(2.5), () => model.heatKernel(0,0),
  () => model.waveState(2), () => model.waveValue(0,0,1),
  () => model.poissonState('missing'), () => model.poissonState('uniform','neumann',.5),
  () => model.harmonicValue(2,.5), () => model.harmonicState(2),
  () => model.pointSourceState(0), () => model.burgersState(2,0,.5,'false'),
  () => model.inverseHeatState(0), () => model.periodicDepth(5),
  () => model.forcedRod(0,.2,0), () => model.forcedRod(0,.2,.05,0)
];
for (const call of invalid) {
  let rejected = false;
  try { call(); } catch (error) { if (!(error instanceof RangeError)) throw error; rejected = true; }
  if (!rejected) throw new Error('Invalid model input was accepted');
}
fixtures.invalidInputs = invalid.length;
fs.writeFileSync(`${directory}/fixtures.json`, JSON.stringify(fixtures));
const result = spawnSync('scratch/lesson-tools/Scripts/python.exe', ['scripts/verify-pde.py'], { stdio:'inherit', env:{...process.env,PYTHONIOENCODING:'utf-8'} });
process.exitCode = result.status ?? 1;
