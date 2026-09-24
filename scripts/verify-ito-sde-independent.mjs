import fs from 'node:fs';
import { execFileSync } from 'node:child_process';
import { itoSdeExamples } from '../src/learn/data/ito-sde-examples.js';
import { ouNoiseMoments, gbmErrorMoments, integralTrace, groupIncrements } from '../src/learn/data/ito-sde-models.js';

const directory = 'scratch/ito-sde-independent-review';
fs.mkdirSync(directory, { recursive: true });
const ou = [];
for (const theta of [0, .05, .37, 2.6]) {
  for (const step of [1 / 8192, .13, .7, 2.3]) ou.push({ theta, step, result: ouNoiseMoments(theta, step) });
}
const moments = [];
for (const mu of [-.35, .23]) {
  for (const sigma of [0, .17, .91]) {
    for (const method of ['euler', 'milstein']) moments.push({ mu, sigma, method, initial: 1.3, horizon: 1.7, steps: 2, result: gbmErrorMoments({ mu, sigma, method, initial: 1.3, horizon: 1.7, steps: 2 }) });
  }
}
const increments = [.375, -.125, .25, -.75, .5, -.25];
const sums = [1, 2, 3, 6].map(group => ({ group, increments: groupIncrements(increments, group), result: integralTrace(groupIncrements(increments, group), 1.5) }));
fs.writeFileSync(`${directory}/inputs.json`, JSON.stringify({ examples: itoSdeExamples, ou, moments, sums }));
process.stdout.write(execFileSync('scratch/lesson-tools/Scripts/python.exe', ['scripts/verify-ito-sde-independent.py'], { encoding: 'utf8' }));
