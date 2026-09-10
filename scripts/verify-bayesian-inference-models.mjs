import assert from 'node:assert/strict';
import fs from 'node:fs';
import * as model from '../src/learn/data/bayesian-inference-models.js';
import { bayesianInferenceExamples } from '../src/learn/data/bayesian-inference-examples.js';

const destination = 'scratch/bayesian-inference-review';
fs.mkdirSync(destination, { recursive: true });
const evidence = { at: new Date().toISOString(), node: process.version, beta: [], updates: [], batches: [], gamma: [], exposures: [], normals: [], patterns: [] };
for (const [a, b] of [[1, 1], [1, 21], [81, 1], [10, 4], [20, 20], [100, 40], [3, 7], [120, 120]]) {
  for (const x of [0, 1e-6, .01, .2, .5, .7, .95, 1 - 1e-6, 1]) {
    evidence.beta.push({ a, b, x, density: model.betaDensity(x, a, b), cdf: model.betaTail(x, a, b), sf: model.betaTail(x, a, b, true) });
  }
  for (const p of [1e-6, .025, .1, .5, .9, .975, 1 - 1e-6]) {
    evidence.beta.push({ a, b, p, quantile: model.betaQuantile(p, a, b) });
  }
}
for (const prior of model.betaPriors) {
  for (const [successes, failures] of [[0, 0], [0, 20], [80, 0], [80, 20], [8, 2]]) {
    for (const level of [.8, .95]) {
      const state = model.betaUpdateState(prior.id, successes, failures, .7, level);
      evidence.updates.push({ ...state, points: state.points.filter((_, index) => index % 24 === 0), shade: state.shade.filter((_, index) => index % 12 === 0) });
    }
  }
}
for (const [a, b] of [[10, 4], [100, 40], [1, 1], [2, 2]]) {
  for (const m of [1, 2, 10, 20]) {
    for (const threshold of [0, m]) evidence.batches.push(model.batchPredictionState(a, b, m, threshold));
  }
}
for (const shape of [1, 2, 11, 42, 80]) {
  for (const rate of [1e-6, 1, 3.5, 9, 1e6]) {
    for (const multiple of [0, .01, .5, 1, 2, 5]) {
      const x = shape * multiple / rate;
      evidence.gamma.push({ shape, rate, x, density: model.gammaDensity(x, shape, rate), cdf: model.gammaTail(x, shape, rate), sf: model.gammaTail(x, shape, rate, true) });
    }
    for (const p of [1e-6, .025, .5, .975, .995, 1 - 1e-6]) evidence.gamma.push({ shape, rate, p, quantile: model.gammaQuantile(p, shape, rate) });
  }
}
for (const counts of [[0, 0], [3, 6], [20, 20]]) {
  for (const exposures of [[.25, .25], [.5, 2], [4, 4]]) {
    const state = model.exposureUpdateState(counts[0], exposures[0], counts[1], exposures[1]);
    evidence.exposures.push({ ...state, points: state.points.filter((_, index) => index % 20 === 0) });
  }
}
for (const priorSd of [.25, 1, 4]) {
  for (const noiseSd of [.25, 2, 4]) {
    for (const n of [1, 4, 40]) evidence.normals.push(model.normalPrecisionState(0, priorSd, n === 1 ? -6 : 4, n, noiseSd));
  }
}
for (const pattern of model.sequencePatterns) {
  for (const conditioning of ['prior', 'posterior', 'same-count']) evidence.patterns.push(model.predictivePatternState(pattern.id, conditioning));
}
const invalid = [
  () => model.betaDensity(.5, 0, 1), () => model.betaDensity(-1, 1, 1),
  () => model.betaTail(NaN, 2, 2), () => model.betaQuantile(1.1, 2, 2),
  () => model.betaUpdateState('missing'), () => model.betaUpdateState('uniform', 81),
  () => model.betaUpdateState('uniform', 1, -1), () => model.betaUpdateState('uniform', 1, 1, .5, .99),
  () => model.batchPredictionState(10, 4, 0), () => model.batchPredictionState(10, 4, 2, 3),
  () => model.batchPredictionState(120, 4, 2, 1), () => model.gammaQuantile(.5, 1, Number.MIN_VALUE),
  () => model.gammaTail(1, 1, Infinity), () => model.gammaDensity(-1, 1, 1),
  () => model.exposureUpdateState(3, 0, 6, 2), () => model.exposureUpdateState(21, 1, 0, 1),
  () => model.normalPrecisionState(0, 0, 4, 4, 2), () => model.normalPrecisionState(0, 1, 4, 0, 2),
  () => model.predictivePatternState('missing'), () => model.predictivePatternState('clustered', 'missing'),
];
invalid.forEach(call => assert.throws(call, RangeError));
assert.equal(model.gammaTail(Number.MAX_VALUE, 1, 1e6, true), 0);
assert.equal(model.betaQuantile(0, 2, 2), 0);
assert.equal(model.betaQuantile(1, 2, 2), 1);
assert.equal(model.gammaQuantile(0, 2, 1), 0);
assert.equal(model.gammaQuantile(1, 2, 1), Infinity);
evidence.invalidCases = invalid.length;
evidence.examples = bayesianInferenceExamples;
fs.writeFileSync(`${destination}/model-fixtures.json`, JSON.stringify(evidence, null, 2));
console.log(JSON.stringify({ exportedForIndependentOracle: Object.fromEntries(Object.entries(evidence).filter(([, value]) => Array.isArray(value)).map(([key, value]) => [key, value.length])), invalidCases: invalid.length, examples: Object.keys(bayesianInferenceExamples).length }));
