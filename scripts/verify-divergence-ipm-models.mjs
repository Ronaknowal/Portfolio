import assert from 'node:assert/strict';
import fs from 'node:fs';
import { spawnSync } from 'node:child_process';
import * as models from '../src/learn/data/divergence-ipm-models.js';
import { divergenceIpmExamples } from '../src/learn/data/divergence-ipm-examples.js';

const directory = 'scratch/divergence-ipm-native-verification';
fs.mkdirSync(directory, { recursive: true });
const laws = [];
for (let a = 0; a <= 5; a += 1) {
  for (let b = 0; b <= 5 - a; b += 1) laws.push([a, b, 5 - a - b]);
}
const divergence = [];
for (const p of laws) {
  for (const q of laws) divergence.push({ p, q, state: models.divergenceState(p, q) });
}
const closeLaws = [];
for (const power of [8, 16, 24, 32, 40]) {
  const epsilon = 2 ** -power;
  closeLaws.push({ p: [1, 3, 2, 2], q: [1 + epsilon, 3 - epsilon, 2, 2], state: models.divergenceState([1, 3, 2, 2], [1 + epsilon, 3 - epsilon, 2, 2]) });
}
let seed = 2039;
const random = () => { seed = (1664525 * seed + 1013904223) >>> 0; return seed / 2 ** 32; };
const processing = [], observers = [];
for (let index = 0; index < 84; index += 1) {
  const p = Array.from({ length: 4 }, () => 1 + Math.floor(9 * random()));
  const q = Array.from({ length: 4 }, () => 1 + Math.floor(9 * random()));
  const channel = Array.from({ length: 4 }, () => {
    const row = Array.from({ length: 3 }, () => Math.floor(5 * random()));
    if (row.every(value => value === 0)) row[0] = 1;
    const sum = row.reduce((a, b) => a + b, 0);
    return row.map(value => value / sum);
  });
  processing.push(models.processDivergence(p, q, channel));
  const positions = [-3, -2 + random(), random(), 1 + 2 * random()];
  for (const kind of ['event', 'linear', 'lipschitz']) observers.push(models.observableState(p, q, positions, kind));
}
for (const channel of [
  [[1e-8, 1 - 1e-8], [0, 1]],
  [[0, 1], [1e-8, 1 - 1e-8]],
  [[1e-8, 1 - 1e-8], [1 - 1e-8, 1e-8]],
  [[1, 0], [0, 1]],
  [[0, 1], [0, 1]],
]) {
  const state = models.processDivergence([100, 1e-8], [1e-8, 100], channel);
  assert.ok(state.after.js <= Math.LN2);
  assert.ok(state.after.kl <= state.before.kl + 1e-12);
  processing.push(state);
}
const kernel = [];
for (let index = 0; index < 40; index += 1) {
  const x = Array.from({ length: 2 + index % 5 }, () => Math.round((6 * random() - 3) * 10) / 10);
  const y = Array.from({ length: 2 + (index + 2) % 5 }, () => Math.round((6 * random() - 3) * 10) / 10);
  for (const kind of ['rbf', 'linear', 'quadratic']) kernel.push(models.kernelWitnessState(x, y, 0.2 + 2.8 * random(), kind));
}
kernel.push(models.kernelWitnessState([-1, 1], [-1, 1]));
kernel.push(models.kernelWitnessState([-1, -1, 1, 1], [-Math.SQRT2, 0, 0, Math.SQRT2], 1, 'quadratic'));
const permutations = [0.2, 0.3, 1, 2, 3].map(bandwidth => models.permutationMmdState(bandwidth, 35));
const variational = [];
for (const scale of [0, 0.2, 0.5, 1, 1.5]) {
  for (const offset of [-2, -0.8, 0, 0.4, 2]) variational.push(models.variationalDivergenceState(scale, offset));
}
const atoms = [];
for (const displacement of [0, 1e-9, 0.01, 0.1, 0.5, 1, 3]) {
  for (const bandwidth of [0.2, 1, 3]) atoms.push(models.movingAtomState(displacement, bandwidth));
}
const invalid = [
  () => models.divergenceState([0, 0], [1, 1]),
  () => models.divergenceState([1e-200, 1], [1, 1]),
  () => models.divergenceState([1, 1], [1, 1, 1]),
  () => models.divergenceState([1, Infinity], [1, 1]),
  () => models.divergenceState(undefined, undefined, 'unknown'),
  () => models.parseDivergenceWeights('1,2,3'),
  () => models.parseDivergenceWeights('1,2,3,1e-999'),
  () => models.parseDivergenceWeights('1,2,3,101'),
  () => models.parseDivergenceWeights('0,0,0,0'),
  () => models.observableState([1, 1], [1, 1], [0, 0]),
  () => models.observableState(undefined, undefined, undefined, 'unknown'),
  () => models.processDivergence([1, 1], [1, 1], [[0.5, 0], [0, 1]]),
  () => models.processDivergence([100, 1e-8], [1e-8, 100], [[Number.MIN_VALUE, 1], [0, 1]]),
  () => models.processDivergence([100, 1e-8], [1e-8, 100], [[1e-9, 1 - 1e-9], [0, 1]]),
  () => models.movingAtomState(-1),
  () => models.movingAtomState(1, 0),
  () => models.parseKernelSamples('0'),
  () => models.parseKernelSamples('0,Infinity'),
  () => models.parseKernelSamples('0,4'),
  () => models.kernelWitnessState([0], [0, 1]),
  () => models.kernelWitnessState([0, 1], [0, 1], 0),
  () => models.kernelWitnessState(undefined, undefined, 1, 'unknown'),
  () => models.permutationMmdState(1, 70),
  () => models.permutationMmdState(1, 1.5),
  () => models.variationalDivergenceState(2, 0),
  () => models.variationalDivergenceState(1, NaN),
];
invalid.forEach(check => assert.throws(check));
const sourceP = [7, 2, 1, 0], sourceQ = [4, 5, 1, 0];
const snapshot = models.divergenceState(sourceP, sourceQ);
sourceP[0] = 1;
assert.equal(snapshot.p[0], 0.7);
assert.throws(() => { snapshot.rows[0].p = 0; }, TypeError);
assert.throws(() => { kernel[0].xx[0][0] = 0; }, TypeError);
assert.equal(models.parseDivergenceWeights('7 2 1 0').join(), '7,2,1,0');
assert.equal(models.parseKernelSamples('-2, -.5, 1, 2').join(), '-2,-0.5,1,2');
fs.writeFileSync(`${directory}/model-fixtures.json`, JSON.stringify({ divergence, closeLaws, processing, observers, kernel, permutations, variational, atoms, invalidInputs: invalid.length, examples: divergenceIpmExamples }, (_, value) => typeof value === 'number' && !Number.isFinite(value) ? String(value) : value));
const result = spawnSync('scratch/lesson-tools/Scripts/python.exe', ['scripts/verify-divergence-ipm-native.py'], { encoding: 'utf8' });
process.stdout.write(result.stdout);
process.stderr.write(result.stderr);
assert.equal(result.status, 0);
