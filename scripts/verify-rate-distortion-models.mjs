import fs from 'node:fs';
import assert from 'node:assert/strict';
import { execFileSync } from 'node:child_process';
import { performance } from 'node:perf_hooks';
import { binaryEntropy, binaryRateDistortion, binaryOptimalChannel, binaryBlockCode, finiteRateDistortion, gaussianRateDistortion, gaussianAllocation, RATE_DISTORTION_SCENARIOS, zeroRateFidelity } from '../src/learn/data/rate-distortion-models.js';
import { rateDistortionExamples } from '../src/learn/data/rate-distortion-examples.js';

const directory = 'scratch/rate-distortion-review';
fs.mkdirSync(directory, { recursive: true });
const fixtures = { binary: [], blocks: [], optimizers: [], allocations: [], fidelity: zeroRateFidelity() };
for (let p = 0; p <= 20; p += 1) for (let d = 0; d <= 40; d += 1) {
  const state = binaryOptimalChannel(p / 20, d / 40);
  assert.ok(Object.isFrozen(state.joint[0]));
  assert.ok(Math.abs(state.information - state.rate) < 3e-14);
  fixtures.binary.push(state);
}
// Every nonempty length-three codebook, under five different iid source laws.
for (let mask = 1; mask < 256; mask += 1) for (const probability of [0, 0.2, 0.5, 0.8, 1]) {
  const codewords = Array.from({ length: 8 }, (_, word) => word).filter(word => mask & (1 << word));
  fixtures.blocks.push(binaryBlockCode({ probability, codewords, selected: mask % 8 }));
}
for (const scenario of Object.values(RATE_DISTORTION_SCENARIOS)) for (const lambda of [0, 0.1, 0.5, 1, 2, 5, 10]) {
  const state = finiteRateDistortion({ ...scenario, lambda, iterations: 5000 });
  assert.ok(Object.isFrozen(state.conditional[0]));
  for (let index = 1; index < state.trace.length; index += 1) assert.ok(state.trace[index].upper <= state.trace[index - 1].upper + 2e-12);
  fixtures.optimizers.push({ case: scenario.label, ...state });
}
for (const initial of [[1, 0], [0, 1], [1e-250, 1], [0.999, 0.001]]) {
  fixtures.optimizers.push({ case: 'binary initialization', ...finiteRateDistortion({ lambda: 2, initial, iterations: 5000 }) });
}
for (const input of [
  { source: [1, 0], costs: [[2, 0], [0, 4]], lambda: 3 },
  { source: [0.3, 0.7], costs: [[9, 9, 9], [2, 2, 2]], lambda: 2 },
  { source: [0.2, 0.3, 0.5], costs: [[0], [2], [1]], lambda: 3 },
  { source: [0.2, 0.8], costs: [[0, 3, 4], [2, 0, 1]], lambda: 0.7 },
  { source: [0.5, 0.5], costs: [[1000, 1001], [1001, 1000]], lambda: 2 },
  { source: [0.3, 0.7], costs: [[0, 10000], [10000, 0]], lambda: 1000 },
]) fixtures.optimizers.push({ case: 'boundary or changed alphabet', ...finiteRateDistortion({ ...input, iterations: 5000 }) });
for (const variances of [[9, 1], [1, 9], [0, 4], [0, 0], [4, 4], [0.1, 0.7, 2.3], [1, 2, 3, 4, 5, 6]]) {
  const total = variances.reduce((sum, value) => sum + value, 0);
  for (let portion = 0; portion <= 20; portion += 1) fixtures.allocations.push(gaussianAllocation(variances, total * portion / 16));
}
assert.equal(binaryRateDistortion(0.5, 0.8), 0);
assert.equal(binaryEntropy(0), 0);
assert.equal(gaussianRateDistortion(9, 0), Infinity);
assert.equal(gaussianRateDistortion(0, 0), 0);
const invalid = [
  () => binaryEntropy(NaN), () => binaryRateDistortion(2, 0.1), () => binaryOptimalChannel(0.2, -1),
  () => binaryBlockCode({ codewords: [] }), () => binaryBlockCode({ codewords: [0, 0] }), () => binaryBlockCode({ selected: 8 }),
  () => finiteRateDistortion({ source: [0.2, 0.2] }), () => finiteRateDistortion({ costs: [[0], [1, 2]] }),
  () => finiteRateDistortion({ costs: [[0, Infinity], [1, 0]] }), () => finiteRateDistortion({ iterations: 5001 }),
  () => finiteRateDistortion({ lambda: -1 }), () => finiteRateDistortion({ initial: [0, 0] }), () => finiteRateDistortion({ initial: false }),
  () => gaussianAllocation([], 1), () => gaussianAllocation([1, -1], 1), () => gaussianAllocation([1], Infinity),
  () => gaussianAllocation([1, 1], Number.MIN_VALUE),
  () => gaussianAllocation([1, 1], 3 * Number.MIN_VALUE),
];
invalid.forEach(check => assert.throws(check, RangeError));
for (const input of [[[1, 1], 1e-250], [[0, 1], Number.MIN_VALUE], [[1, 1], 2 * Number.MIN_VALUE]]) {
  const allocation = gaussianAllocation(...input);
  assert.ok(Number.isFinite(allocation.rate));
  assert.equal(allocation.components.reduce((sum, component) => sum + component.distortion, 0), input[1]);
}
const durations = [];
for (let trial = 0; trial < 10; trial += 1) {
  const start = performance.now();
  finiteRateDistortion({ ...RATE_DISTORTION_SCENARIOS.levels, lambda: 0.5, iterations: 5000 });
  durations.push(performance.now() - start);
}
fs.writeFileSync(directory + '/model-fixtures.json', JSON.stringify(fixtures));
fs.writeFileSync(directory + '/native/verify-input.json', JSON.stringify(rateDistortionExamples));
execFileSync('scratch/lesson-tools/Scripts/python.exe', ['scripts/verify-rate-distortion-native.py'], { stdio: 'inherit' });
const report = { at: new Date().toISOString(), node: process.version, binaryCases: fixtures.binary.length, codebookCases: fixtures.blocks.length, optimizerCases: fixtures.optimizers.length, allocationCases: fixtures.allocations.length, invalidCases: invalid.length, localNodeMilliseconds: durations, native: JSON.parse(fs.readFileSync(directory + '/native-results.json')) };
fs.writeFileSync(directory + '/model-results.json', JSON.stringify(report, null, 2) + '\n');
console.log(JSON.stringify({ binary: report.binaryCases, codebooks: report.codebookCases, optimizers: report.optimizerCases, allocations: report.allocationCases, invalid: report.invalidCases, native: report.native }, null, 2));
