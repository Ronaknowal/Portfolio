import assert from 'node:assert/strict';
import { mkdirSync, writeFileSync } from 'node:fs';
import { spawnSync } from 'node:child_process';
import { approachState, circleMotionState, curvaturePresets, curvatureState, descentState, formatCalculusNumber, localChangeState } from '../src/learn/data/multivariate-calculus-models.js';
import { multivariateCalculusExamples } from '../src/learn/data/multivariate-calculus-examples.js';

const directory = 'scratch/multivariate-verification';
mkdirSync(directory, { recursive: true });
const fixtures = { local: [], paths: [], circle: [], curvature: [], descent: [] };
for (const x of [-2, -1, 0, 1, 2]) {
  for (const y of [-2, -1, 0, 1, 2]) {
    for (let angle = 0; angle <= 360; angle += 15) {
      for (const step of [-.5, -.1, 0, .01, .2, .5]) {
        const state = localChangeState(x, y, angle, step);
        fixtures.local.push({ input: [x, y, angle, step], state });
      }
    }
  }
}
for (const kind of ['axis', 'line', 'parabola']) {
  for (const coefficient of [-2, -1, -.5, 0, .2, .5, 1, 2]) {
    fixtures.paths.push({ input: [kind, coefficient], state: approachState(kind, coefficient) });
  }
}
for (let angle = 0; angle <= 360; angle += 3) fixtures.circle.push(circleMotionState(angle));
fixtures.circle.push(circleMotionState(Math.atan2(1, 2) * 180 / Math.PI));
fixtures.circle.push(circleMotionState(180 + Math.atan2(1, 2) * 180 / Math.PI));
for (const preset of Object.keys(curvaturePresets)) {
  for (let angle = 0; angle <= 360; angle += 15) fixtures.curvature.push(curvatureState(preset, angle));
}
for (let rate = 0; rate <= 75; rate += 1) {
  for (let steps = 0; steps <= 12; steps += 1) fixtures.descent.push(descentState(rate / 100, steps));
}
const invalid = [
  () => localChangeState(NaN, 1, 0, 0), () => localChangeState(3, 1, 0, 0),
  () => localChangeState(0, Infinity, 0, 0), () => localChangeState(0, 0, -1, 0),
  () => localChangeState(0, 0, 0, .6), () => localChangeState('1', 0, 0, 0),
  () => approachState('wrong', 1), () => approachState('line', 3),
  () => approachState('parabola', NaN), () => circleMotionState(361),
  () => circleMotionState(Infinity), () => curvatureState('unknown', 0),
  () => curvatureState('bowl', -1), () => descentState(-.1, 0),
  () => descentState(.8, 0), () => descentState(.1, 13),
  () => descentState(.1, 1.2), () => descentState(.1, NaN),
];
invalid.forEach(operation => assert.throws(operation));
assert.equal(formatCalculusNumber(1e-12), '1.000e-12');
assert.equal(formatCalculusNumber(-1e-15), '-1.000e-15');
assert.equal(formatCalculusNumber(0), '0');
assert.equal(formatCalculusNumber(-0), '0');
writeFileSync(`${directory}/fixtures.json`, JSON.stringify(fixtures));
writeFileSync(`${directory}/programs.json`, JSON.stringify(multivariateCalculusExamples));
const result = spawnSync(process.env.LESSON_PYTHON || 'scratch/lesson-tools/Scripts/python.exe', ['scripts/verify-multivariate-calculus-native.py'], { encoding: 'utf8' });
process.stdout.write(result.stdout);
process.stderr.write(result.stderr);
assert.equal(result.status, 0, 'Independent native verifier failed');
console.log(`All ${invalid.length} invalid model input groups rejected.`);
