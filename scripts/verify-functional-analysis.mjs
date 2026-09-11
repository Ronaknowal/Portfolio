import fs from 'node:fs';
import path from 'node:path';
import { spawnSync } from 'node:child_process';
import * as model from '../src/learn/data/functional-analysis-models.js';
import { functionalAnalysisExamples } from '../src/learn/data/functional-analysis-examples.js';

const directory = 'scratch/functional-analysis-verification';
fs.mkdirSync(directory, { recursive: true });
const cases = { spikes: [], integrals: [], validity: [], representers: [], ridge: [], means: [], quadrature: [] };
for (const width of [.005, .01, .025, .125, .3, .45]) {
  for (const height of [0, .25, 1, 2, 3]) cases.spikes.push(model.spikeState(width, height));
}
for (const slopes of [[2,-1,1,0], [0,0,0,0], [-4,4,-4,4], [1.25,-.7,2.1,-3.8], [2,2,0,0]]) {
  for (let i = 0; i <= 40; i += 1) cases.integrals.push(model.integralSpaceState(slopes, i / 40));
}
for (const kind of ['polynomial', 'bigrams', 'invalid']) cases.validity.push(model.kernelValidityState(kind));
for (let i = -30; i <= 30; i += 1) {
  for (const extra of [false, true]) {
    const state = model.representerState(i / 20, extra);
    cases.representers.push({ ...state, values: [0,.125,.25,.375,.5,.75,1].map(state.total) });
  }
}
for (const fixture of Object.keys(model.ridgeFixtures)) {
  for (const gamma of [.05,.1,.7,1,3.25,8,16,32]) {
    for (const lambda of [.0001,.001,.05,.2,.5,2]) {
      const state = model.kernelRidgeState(gamma, lambda, .325, fixture);
      cases.ridge.push({ ...state, values: [-.25,0,.125,.5,.875,1,1.25].map(state.predict) });
    }
  }
}
for (const gamma of [.05,.1,.3,1,3,8]) {
  for (const same of [false, true]) {
    const state = model.distributionEmbeddingState(gamma,same);
    cases.means.push({ ...state, values: [-3,-2,-.3,0,.4,1,3].map(state.witness) });
  }
}
for (let i = 0; i <= 40; i += 1) {
  for (const weight of [null,-1,-.2,0,.5,1,2]) cases.quadrature.push(model.kernelQuadratureState(i/40,weight));
}
cases.quadrature.push(model.kernelQuadratureState(2/3));
const invalid = [
  () => model.spikeState(0), () => model.spikeState(Infinity), () => model.spikeState(.5),
  () => model.integralSpaceState([1,2,3]), () => model.integralSpaceState([0,0,0,NaN]),
  () => model.integralSpaceState([0,0,0,0],-1), () => model.integralSpaceState([0,0,0,0],'0'),
  () => model.kernelValidityState('fake'), () => model.representerState(2),
  () => model.representerState(.2,1), () => model.kernelRidgeState(0),
  () => model.kernelRidgeState(1,0), () => model.kernelRidgeState(1,.1,2),
  () => model.kernelRidgeState(1,.1,.5,'constructor'),
  () => model.distributionEmbeddingState(1e-10), () => model.distributionEmbeddingState(1,0),
  () => model.kernelQuadratureState(-1), () => model.kernelQuadratureState(.5,Infinity),
  () => model.kernelQuadratureState(.5,true),
];
for (const run of invalid) {
  let rejected = false;
  try { run(); } catch { rejected = true; }
  if (!rejected) throw Error('Expected supported-domain rejection');
}
fs.writeFileSync(path.join(directory,'inputs.json'), JSON.stringify({ cases, invalidRejections: invalid.length, examples: functionalAnalysisExamples }));
const executable = process.env.LESSON_PYTHON || path.resolve('scratch/lesson-tools/Scripts/python.exe');
const result = spawnSync(executable, ['scripts/verify-functional-analysis.py'], { encoding: 'utf8' });
process.stdout.write(result.stdout || '');
process.stderr.write(result.stderr || '');
if (result.status !== 0) process.exit(result.status || 1);
