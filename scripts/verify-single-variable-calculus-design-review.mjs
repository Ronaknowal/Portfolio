import fs from 'node:fs';
import path from 'node:path';
import { spawnSync } from 'node:child_process';
import { createHash } from 'node:crypto';
import { limitGuarantee, motionRate, motionAccumulation, compositionChange, taylorApproximation } from '../src/learn/data/single-variable-calculus-models.js';
import { singleVariableCalculusExamples } from '../src/learn/data/single-variable-calculus-examples.js';

const directory = 'scratch/single-variable-calculus-design-review';
fs.mkdirSync(directory, { recursive: true });
const limitCases = [];
for (const kind of ['smooth', 'hole', 'jump']) {
  for (const epsilon of [.01, .03, .1, .25, .5, .83, .8399999999999999, .84, .8400000000000001, 1]) {
    for (const delta of [.001, .003, .02, .1, .19999999999999998, .2, .20000000000000004, .333, .5]) {
      limitCases.push({ epsilon: String(epsilon), delta: String(delta), state: limitGuarantee(kind, epsilon, delta) });
    }
  }
}
const rates = [];
for (const time of [1, 1.25, 1.75, 2, 2.75, 3]) {
  for (const h of [-1, -.75, -.125, -.0001, .0001, .125, .75, 1]) rates.push(motionRate(time, h));
}
const accumulations = [];
for (const upper of [.25, .75, 1, 1.5, 2.5, 3, 3.5, 4]) {
  for (const n of [1, 3, 4, 7, 8, 31]) {
    for (const rule of ['left', 'midpoint', 'right']) accumulations.push(motionAccumulation(upper, n, rule));
  }
}
const compositions = [];
for (const x of [-1, -.75, -.5, -.25, 0, .125, .75, 1]) {
  for (const h of [-.5, -.125, 0, .125, .5]) compositions.push(compositionChange(x, h));
}
const taylors = [];
for (const kind of ['exp', 'log']) {
  for (const degree of [0, 1, 2, 3, 6, 12]) {
    for (const x of [-.9, -.5, 0, .5, 1, 1.5]) taylors.push(taylorApproximation(kind, degree, x));
  }
}
fs.writeFileSync(path.join(directory, 'cases.json'), JSON.stringify({ limitCases, rates, accumulations, compositions, taylors, examples: singleVariableCalculusExamples }));
const python = process.env.LESSON_PYTHON || 'scratch/lesson-tools/Scripts/python.exe';
const result = spawnSync(python, ['-X', 'utf8', 'scripts/verify-single-variable-calculus-design-review.py'], { encoding: 'utf8' });
if (result.status !== 0) throw new Error(result.stderr || result.stdout);
const evidence = JSON.parse(result.stdout);
evidence.timestamp = new Date().toISOString();
evidence.scope = 'Initial design, pure contracts and actual displayed native programs only; not final body, visual or production verification.';
evidence.hashes = Object.fromEntries([
  'docs/teaching/SINGLE-VARIABLE-CALCULUS-LESSON-DESIGN.md',
  'src/learn/data/curriculum/blueprints/single-variable-calculus-limits-derivatives-integrals.js',
  'src/learn/data/single-variable-calculus-models.js',
  'src/learn/data/single-variable-calculus-examples.js',
].map(file => [file, createHash('sha256').update(fs.readFileSync(file)).digest('hex')]));
fs.writeFileSync(path.join(directory, 'results.json'), JSON.stringify(evidence, null, 2));
console.log(JSON.stringify(evidence, null, 2));
