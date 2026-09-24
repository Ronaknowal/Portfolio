import fs from 'node:fs';
import crypto from 'node:crypto';
import { spawnSync } from 'node:child_process';
import assert from 'node:assert/strict';
import * as model from '../src/learn/data/gradient-boosted-trees-models.js';

const root = 'scratch/gradient-boosted-trees-verification';
fs.mkdirSync(root, { recursive: true });
const hash = path => crypto.createHash('sha256').update(fs.readFileSync(path)).digest('hex');
const trees = [];
for (let code = 0; code < 243; code += 1) {
  let remaining = code;
  const y = Array.from({ length: 5 }, () => {
    const value = remaining % 3 - 1;
    remaining = Math.floor(remaining / 3);
    return value;
  });
  for (const x of [[0, 1, 2, 3, 4], [0, 0, 1, 1, 2]]) {
    for (const depth of [1, 2]) {
      const result = model.fitCorrectionTree(x, y, depth);
      trees.push({ x, y, depth, result, prediction: x.map(value => model.predictCorrection(result, value)) });
    }
  }
}
const boosts = [];
// Price each representable adjacent-value partition independently of a midpoint.
const adjacent = [];
for (const low of [Number.MIN_VALUE, 1 - Number.EPSILON / 2, 1, 1 + Number.EPSILON, -1, -1 - Number.EPSILON, 999]) {
  const buffer = new ArrayBuffer(8);
  const floats = new Float64Array(buffer);
  const bits = new BigUint64Array(buffer);
  floats[0] = low;
  bits[0] += low > 0 ? 1n : -1n;
  const high = floats[0];
  const result = model.fitCorrectionTree([low, high], [-2, 3]);
  assert.equal(result.leaf, false);
  assert(result.threshold >= low && result.threshold < high);
  assert.deepEqual([low, high].map(value => model.predictCorrection(result, value)), [-2, 3]);
  adjacent.push({ low, high, threshold: result.threshold });
}
const changedPractice = model.fitBoosting({ x: [1, 2, 3, 4], y: [0, 0, 4, 4], rate: .25, rounds: 1 });
assert.deepEqual(changedPractice.stages[1].prediction, [1.5, 1.5, 2.5, 2.5]);
assert.equal(changedPractice.stages[1].mse, 2.25);
for (const y of [[2, 2, 3, 7, 8, 8], [0, 0, 0, 0, 0, 0], [3, -1, 2, 5, -2, 1], [-30, 30, -30, 30, -30, 30]]) {
  for (const rate of [0, .1, .5, 1, 1.5]) {
    for (const depth of [0, 1, 3]) boosts.push(model.fitBoosting({ y, rate, depth, rounds: 8 }));
  }
}
const newton = [];
for (const kind of ['square', 'logistic', 'confident']) {
  for (const split of [1, 2, 3, 4]) {
    for (const lambda of [0, .25, 1, 5]) {
      for (const alpha of [0, .25, 2, 5]) {
        for (const gamma of [0, .5, 3]) newton.push(model.newtonInvestigation({ kind, split, lambda, alpha, gamma, rate: .7 }));
      }
    }
  }
}
const histogram = [];
for (const coarse of [false, true]) for (const missingTarget of [0, 2, 5, 8, 10]) histogram.push(model.histogramInvestigation({ coarse, missingTarget }));
const sampling = [];
for (const keep of [1, 2, 3]) {
  for (let draw = 1; draw <= 6 - keep; draw += 1) {
    const first = model.gossInvestigation({ keep, draw });
    for (let sample = 0; sample < first.subsets.length; sample += 1) sampling.push(model.gossInvestigation({ keep, draw, sample }));
  }
}
const categories = [];
for (const order of [[0, 1, 2, 3, 4, 5], [5, 4, 3, 2, 1, 0], [2, 5, 1, 4, 0, 3]]) {
  for (let row = 0; row < 6; row += 1) {
    for (const smoothing of [.25, 1, 4]) {
      for (const flipped of [false, true]) categories.push(model.orderedStatistics({ order, row, smoothing, flipped }));
    }
  }
}
const validation = [];
for (const rate of [.05, .3, .75, 1]) {
  for (const depth of [1, 2, 3]) {
    const state = model.validationInvestigation({ rate, depth });
    state.plottedSegments = [0, 1, 5, 60].map(round => ({ round, segments: model.boostingPredictionSegments(state.model, round) }));
    validation.push(state);
  }
}
const invalid = [
  () => model.fitCorrectionTree([0, , 1], [1, 2, 3]),
  () => model.fitCorrectionTree([0, 1], [1, NaN]),
  () => model.fitBoosting({ y: [1, 2] }),
  () => model.fitBoosting({ rate: Infinity }),
  () => model.fitBoosting({ depth: 1.5 }),
  () => model.leafOptimum([1], [-1]),
  () => model.leafOptimum([1], [0], 0),
  () => model.splitStatistics([1, 2, 3], [1, 1, 1], [0, 0]),
  () => model.splitStatistics([1, 2], [1, 1], [0, 1]),
  () => model.orderedStatistics({ order: [0, 0, 2, 3, 4, 5] }),
  () => model.orderedStatistics({ order: [0, 1, , 3, 4, 5] }),
  () => model.bundleExclusive([[1, 1, 0]]),
  () => model.bundleExclusive([[0, , 0]]),
  () => model.parseBoostingTargets('1 2 3 4 5 1e-9999'),
  () => model.parseBoostingTargets('1 2 3 4 5 31'),
  () => model.gossInvestigation({ draw: 0 }),
];
invalid.forEach(check => assert.throws(check));
const flat = model.fitCorrectionTree([1, 1, 1], [-2, 0, 5], 4);
assert.equal(flat.leaf, true);
assert.deepEqual(model.bundleExclusive([[0, 0, 0], [1, 0, 0], [0, 2, 0], [0, 0, 1]]).decoded, [[0, 0, 0], [1, 0, 0], [0, 2, 0], [0, 0, 1]]);
const stress = model.newtonInvestigation({ kind: 'confident', split: 3, lambda: 0, rate: 1 });
assert(stress.afterLoss > stress.beforeLoss);
const payload = {
  capturedAt: new Date().toISOString(),
  modelSha256: hash('src/learn/data/gradient-boosted-trees-models.js'),
  trees, boosts, newton, histogram, sampling, categories, validation, adjacent, changedPractice,
  invalidInputsRejected: invalid.length,
};
fs.writeFileSync(`${root}/oracle-input.json`, JSON.stringify(payload));
const result = spawnSync('scratch/lesson-tools/Scripts/python.exe', ['scripts/verify-gradient-boosted-trees-native.py'], { encoding: 'utf8', timeout: 180000 });
process.stdout.write(result.stdout);
process.stderr.write(result.stderr);
if (result.status !== 0) process.exit(result.status ?? 1);
