import fs from 'node:fs';
import assert from 'node:assert/strict';
import { convexChordState, allocationCertificateState, ridgeCurvatureState, ridgeContourPoints, softThresholdState } from '../src/learn/data/convex-optimization-models.js';

const fixtures = { chords: [], allocations: [], ridge: [], thresholds: [] };
for (const preset of ['quadratic', 'absolute', 'quartic', 'doubleWell']) {
  for (const [left, right] of [[-2, 2], [-1, 1], [-2, -0.5], [0, 1.5], [-0.5, 2]]) {
    for (const fraction of [0, 0.1, 0.25, 0.5, 0.75, 0.9, 1]) {
      const state = convexChordState(preset, left, right, fraction);
      if (preset !== 'doubleWell') assert.ok(state.gap >= -1e-12);
      fixtures.chords.push(state);
    }
  }
}
for (const budget of [0, 0.5, 1, 1.5, 3, 4, 6.5, 7, 8]) {
  for (const first of [0, 0.5, 1, 2.5, 4, 7, 8]) {
    for (const second of [0, 0.5, 1.5, 3, 6, 8]) {
      fixtures.allocations.push(allocationCertificateState(budget, first, second));
    }
  }
}
for (const preset of ['full', 'duplicate']) {
  for (const penalty of [0, 0.1, 0.5, 1, 2]) {
    for (const stepFactor of [0.1, 0.5, 1, 1.9, 2, 2.1, 2.2]) {
      const state = ridgeCurvatureState(preset, penalty, stepFactor, 24);
      state.contours = [0.5, 2, 8].map(excess => ({ excess, points: ridgeContourPoints(state, excess) }));
      fixtures.ridge.push(state);
    }
  }
}
for (const input of [-4, -3, -1, -0.1, 0, 0.1, 1, 3, 4]) {
  for (const penalty of [0, 0.1, 0.5, 1, 2, 3, 4]) {
    for (const candidate of [-4, -1, 0, 1, 4]) fixtures.thresholds.push(softThresholdState(input, penalty, candidate));
  }
}
const invalid = [
  () => convexChordState('bad', -1, 1, 0.5), () => convexChordState('quadratic', 1, 1, 0.5),
  () => convexChordState('quadratic', -1, 1, NaN), () => allocationCertificateState(-1, 0, 0),
  () => allocationCertificateState(4, Infinity, 0), () => ridgeCurvatureState('bad', 0.5, 1, 4),
  () => ridgeCurvatureState('full', -0.1, 1, 4), () => ridgeCurvatureState('full', 0.5, 1, 25),
  () => ridgeCurvatureState('full', 0.5, 1, 1.5), () => softThresholdState(0, -1, 0),
  () => softThresholdState(0, 1, 5), () => softThresholdState(NaN, 1, 0),
];
invalid.forEach(fn => assert.throws(fn));
// Exact classification must not silently mean "within a display tolerance".
assert.equal(allocationCertificateState(4, 2.5 + 5e-11, 1.5).feasible, false);
assert.equal(allocationCertificateState(4, 2.5 + 5e-11, 1.5).gap, null);
assert.equal(softThresholdState(0, 0, 5e-11).stationary, false);
for (const penalty of [1e-8, 1e-13, 1e-16, 1e-30]) {
  const state = ridgeCurvatureState('duplicate', penalty, 1, 0);
  assert.equal(state.unique, true);
  assert.equal(state.smallestCurvature, 2 * penalty);
  assert.equal(state.largestCurvature, 12 + 2 * penalty);
  assert.deepEqual(state.optimum, [5 / (6 + penalty), 5 / (6 + penalty)]);
  assert.ok(ridgeContourPoints(state, 0.5).flat().every(Number.isFinite));
}
assert.throws(() => ridgeContourPoints(ridgeCurvatureState('duplicate', Number.MIN_VALUE, 1, 0), 100), RangeError);
fs.mkdirSync('scratch/convex-optimization-review', { recursive: true });
fs.writeFileSync('scratch/convex-optimization-review/model-fixtures.json', JSON.stringify(fixtures));
console.log(JSON.stringify({ ...Object.fromEntries(Object.entries(fixtures).map(([key, values]) => [key, values.length])), invalid: invalid.length, precisionRegressions: 8 }));
