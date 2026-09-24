// Complementary independent checks. Reuses the author's dense PyTorch oracle
// and complete native run; does not rerun the eighteen training experiments.
import assert from 'node:assert/strict';
import fs from 'node:fs';
import { createHash } from 'node:crypto';
import { activation, activationNames, geometry, xorRows, triangle } from '../src/learn/data/perceptron-models.js';
import { perceptronData } from '../src/learn/data/perceptron-data.js';
import { perceptronExamples } from '../src/learn/data/perceptron-examples.js';

const groups = [];
let assertions = 0;
const verify = (condition, label) => { assert.ok(condition, label); assertions++; };
const close = (actual, expected, tolerance, label) => verify(Math.abs(actual - expected) <= tolerance, `${label}: ${actual} != ${expected}`);
let largestDerivativeError = 0;
for (const name of Object.keys(activationNames)) {
  for (const point of [-5.87, -3.39, -.17, .31, 1.73, 5.19]) {
    const h = 1e-5;
    const numerical = (activation(name, point + h).value - activation(name, point - h).value) / (2 * h);
    const error = Math.abs(numerical - activation(name, point).slope);
    largestDerivativeError = Math.max(largestDerivativeError, error);
    close(numerical, activation(name, point).slope, 2e-8, `${name} off-grid derivative at ${point}`);
  }
}
groups.push('54 off-grid smooth-point derivative checks use finite differences of the forward values, independently of the saved PyTorch oracle.');
for (const state of [
  { x1: -3.75, x2: 2.25, w1: -.5, w2: 3.25, b: -1.75 },
  { x1: .25, x2: -2, w1: 4, w2: 0, b: 3 },
  { x1: -1, x2: 1, w1: 0, w2: .25, b: -4 },
  { x1: 0, x2: 0, w1: 1, w2: -1, b: 0 },
]) {
  const result = geometry(state);
  close(state.w1 * result.foot[0] + state.w2 * result.foot[1] + state.b, 0, 1e-12, 'Projected point is on boundary');
  close((state.x1 - result.foot[0]) * state.w2 - (state.x2 - result.foot[1]) * state.w1, 0, 1e-12, 'Projection displacement is parallel to normal');
  close(Math.hypot(state.x1 - result.foot[0], state.x2 - result.foot[1]), Math.abs(result.distance), 1e-12, 'Distance agrees with actual projected geometry');
  for (const scale of [.25, 1.75, 3]) {
    const scaled = geometry(state, scale);
    close(scaled.distance, result.distance, 1e-12, 'Positive common scale preserves distance');
    close(scaled.hard, result.hard, 0, 'Positive common scale preserves hard decision');
  }
}
groups.push('Orthogonal projection, signed distance and coefficient scaling checked at four changed states, including an off-window foot and exact tie.');
const shifted = xorRows(-.5, -4 / 3);
verify(Math.abs(shifted[3].q) < 1e-12 && shifted[1].residual < -.66 && shifted[2].residual < -.66, 'One repaired corner breaks both single-active cases');
const area = [0, 1].reduce((sum, x) => sum + (triangle(x) + triangle(x + 1)) / 2, 0);
close(area, 1, 0, 'Triangle piecewise trapezoid area');
close(2 * 4096 * 16384, 134217728, 0, 'Plain block count uses feature widths');
close(3 * 4096 * 11008, 135266304, 0, 'Rounded gated count');
close(8 * 4096 * 16384 * 2 / 2 ** 30, 1, 0, 'Raw activation GiB');
groups.push('Changed XOR, piecewise triangle and separate feature/sequence resource calculations.');
const hash = file => createHash('sha256').update(fs.readFileSync(file)).digest('hex');
const native = JSON.parse(fs.readFileSync('docs/teaching/evidence/perceptron-native.json', 'utf8'));
for (const program of native.programs) {
  const current = perceptronExamples.find(example => example.file === program.file);
  verify(current && current.code === program.code && current.expected === program.expected, `Native receipt matches current displayed ${program.file}`);
}
verify(perceptronData.runs.every(run => run.trace.every(point => point.trainLoss >= .001 && point.trainLoss <= 3)), 'All recorded training values lie within the displayed logarithmic domain');
verify(perceptronData.runs.every(run => run.correct >= 116 && run.correct <= 120), 'Optional declared count zoom includes every current run');
groups.push('Exact displayed-program identity and all-sample chart domains checked; existing complete training and dense oracle receipts reused.');
const sources = [
  'src/learn/data/topics/perceptrons-neurons-activation-functions.jsx',
  'src/learn/data/perceptron-models.js', 'src/learn/data/perceptron-data.js', 'src/learn/data/perceptron-examples.js',
  'src/learn/components/lesson-labs/PerceptronShared.jsx', 'src/learn/components/lesson-labs/PerceptronLabs.jsx',
  'src/learn/components/lesson-labs/PerceptronFigures.jsx', 'src/learn/components/lesson-labs/perceptron-labs.css',
  'src/learn/data/curriculum/blueprints/perceptrons-neurons-activation-functions.js', 'scripts/verify-perceptron-review.mjs',
];
const receipt = { status: 'passed', timestamp: new Date().toISOString(), reviewer: 'implement_dl_backprop (independent of Perceptrons author)', groups, assertions, largestDerivativeError, reusedEvidence: ['docs/teaching/evidence/perceptron-native.json', 'docs/teaching/evidence/perceptron-models.json', 'docs/teaching/evidence/perceptron-activation-oracle.json', 'docs/teaching/evidence/perceptron-browser.json'], sourceHashes: Object.fromEntries(sources.map(file => [file, hash(file)])) };
fs.writeFileSync('docs/teaching/evidence/perceptron-independent.json', JSON.stringify(receipt, null, 2) + '\n');
console.log(`Independent Perceptrons review: ${assertions} complementary assertions; max finite-difference slope error ${largestDerivativeError}.`);
