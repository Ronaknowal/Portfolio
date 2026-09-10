import assert from 'node:assert/strict';
import fs from 'node:fs';
import { divergenceState, processDivergence, kernelWitnessState, movingAtomState, variationalDivergenceState } from '../src/learn/data/divergence-ipm-models.js';
import { divergenceIpmExamples } from '../src/learn/data/divergence-ipm-examples.js';
const directory = 'scratch/divergence-ipm-independent';
fs.mkdirSync(directory, { recursive: true });
const counts = { bernoulliDivergences: 0, sufficientChannelEqualities: 0, estimatorExpectationLaws: 0, variationalShiftIdentities: 0, atomIdentities: 0, rangeRegressions: 0 };
const near = (actual, expected, tolerance = 2e-11) => assert(Math.abs(actual - expected) <= tolerance * Math.max(1, Math.abs(expected)), actual + ' != ' + expected);
const kl = (p, q) => p.reduce((total, value, index) => total + (value === 0 ? 0 : value * Math.log(value / q[index])), 0);
for (let a = 1; a < 20; a++) for (let b = 1; b < 20; b++) {
  const p = [a / 20, 1 - a / 20], q = [b / 20, 1 - b / 20], mid = p.map((value, index) => (value + q[index]) / 2);
  const state = divergenceState(p, q);
  near(state.values.kl, kl(p, q)); near(state.values.reverse, kl(q, p));
  near(state.values.js, (kl(p, mid) + kl(q, mid)) / 2);
  near(state.values.hellinger, 1 - Math.sqrt(p[0] * q[0]) - Math.sqrt(p[1] * q[1]));
  near(state.values.chi, (p[0] - q[0]) ** 2 / (q[0] * q[1]));
  near(state.values.tv, Math.abs(p[0] - q[0]));
  counts.bernoulliDivergences++;
}
// A lossless sufficient statistic can merge symbols with the same p/q ratio.
for (let a = 1; a < 9; a++) for (let b = 1; b < 9; b++) {
  const state = processDivergence([a, a, 10 - a, 10 - a], [b, b, 10 - b, 10 - b], [[1, 0], [1, 0], [0, 1], [0, 1]]);
  for (const kind of Object.keys(state.before)) near(state.before[kind], state.after[kind]);
  counts.sufficientChannelEqualities++;
}
// Enumerate all samples of two from two-point populations and average with the
// exact Bernoulli sampling probabilities. This tests expectation, not just a Gram sum.
const outcomes = [[-1, -1], [-1, 1], [1, -1], [1, 1]];
for (const p of [.25, .5, .75]) for (const q of [.25, .5, .75]) for (const bandwidth of [.5, 1, 2]) {
  const crossKernel = Math.exp(-2 / bandwidth ** 2);
  const truth = 2 * (p - q) ** 2 * (1 - crossKernel);
  const varianceP = 2 * p * (1 - p) * (1 - crossKernel);
  const varianceQ = 2 * q * (1 - q) * (1 - crossKernel);
  let biased = 0, unbiased = 0;
  const probability = (sample, chance) => sample.reduce((product, x) => product * (x === 1 ? chance : 1 - chance), 1);
  for (const x of outcomes) for (const y of outcomes) {
    const mass = probability(x, p) * probability(y, q);
    const state = kernelWitnessState(x, y, bandwidth);
    near(state.witnessGap, state.biasedSquared);
    biased += mass * state.biasedSquared; unbiased += mass * state.unbiasedSquared;
  }
  near(unbiased, truth); near(biased, truth + varianceP / 2 + varianceQ / 2);
  counts.estimatorExpectationLaws++;
}
for (let tick = -20; tick <= 20; tick++) {
  const offset = tick / 10, state = variationalDivergenceState(1, offset);
  near(state.gap, Math.expm1(offset) - offset);
  counts.variationalShiftIdentities++;
}
for (const displacement of [0, .001, .2, 1, 3]) for (const bandwidth of [.2, .5, 1, 3]) {
  const state = movingAtomState(displacement, bandwidth);
  near(state.mmd ** 2, 2 * (1 - Math.exp(-(displacement ** 2) / (2 * bandwidth ** 2))));
  assert.equal(state.tv, Number(displacement > 0)); near(state.w1, displacement);
  counts.atomIdentities++;
}
assert.throws(() => processDivergence([100, 1e-8], [1e-8, 100], [[Number.MIN_VALUE, 1], [0, 1]]), RangeError);
counts.rangeRegressions++;
const boundary = processDivergence([100, 1e-8], [1e-8, 100], [[1e-8, 1 - 1e-8], [0, 1]]);
assert(boundary.outputP[0] > 0 && boundary.outputQ[0] > 0);
assert(Number.isFinite(boundary.after.kl) && Number.isFinite(boundary.after.js));
for (const kind of Object.keys(boundary.before)) assert(boundary.after[kind] <= boundary.before[kind] + 1e-10);
counts.rangeRegressions++;
fs.writeFileSync(directory + '/displayed-examples.json', JSON.stringify(divergenceIpmExamples));
const record = { checkedAt: new Date().toISOString(), counts, allPassed: true,
  finding: 'Rejected a previously accepted subnormal channel that fabricated support loss and infinite post-processing KL/JS.',
  scope: 'Complementary closed-form Bernoulli checks, sufficient-statistic equality, repeated-sampling estimator expectations and actual accepted/rejected channel range.' };
fs.writeFileSync(directory + '/model-review.json', JSON.stringify(record, null, 2));
console.log(JSON.stringify(record, null, 2));
