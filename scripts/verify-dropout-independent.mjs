// Complementary independent checks: exact expected objective gradient,
// KL/entropy identity and native BatchNorm state (no fitting/browser work).
import fs from 'node:fs';
import crypto from 'node:crypto';
import assert from 'node:assert/strict';
import { maskedUpdate, maskOutcomes, normalizationStep, monteCarloSummary, executeDepth, geometryOutput, maskShapes } from '../src/learn/data/dropout-models.js';

const reportPath = 'docs/teaching/evidence/dropout-implementation/independent-model.json';
fs.writeFileSync(reportPath, JSON.stringify({ passed: false, state: 'running' }));
function close(actual, expected, tolerance = 1e-9) {
  assert.ok(Number.isFinite(actual) && Math.abs(actual - expected) < tolerance, `${actual} != ${expected}`);
}
let objectiveCases = 0, momentCases = 0, prefixCases = 0;
for (const p of [0, .07, .33, .69, .93]) {
  for (const features of [[1.4, -.7], [0, 2.8], [-2.3, -1.9]]) {
    const weights = [-.6, 1.1], target = .8;
    const weighted = [0, 0];
    let expectedLoss = 0;
    for (const mask of [[0, 0], [0, 1], [1, 0], [1, 1]]) {
      const chance = mask.reduce((product, bit) => product * (bit ? 1 - p : p), 1);
      const result = maskedUpdate({ features, weights, target, mask, probability: p, learningRate: 0 });
      result.weightGradient.forEach((gradient, i) => { weighted[i] += chance * gradient; });
      expectedLoss += chance * result.loss;
    }
    const cleanError = weights.reduce((sum, weight, i) => sum + weight * features[i], 0) - target;
    const penalty = p / (1 - p) * weights.reduce((sum, weight, i) => sum + weight ** 2 * features[i] ** 2, 0) / 2;
    close(expectedLoss, cleanError ** 2 / 2 + penalty);
    weighted.forEach((gradient, i) => close(gradient, cleanError * features[i] + p / (1 - p) * weights[i] * features[i] ** 2));
    objectiveCases++;
    for (const multiplier of [0, .4, 1.8, 1 / (1 - p)]) {
      const outcomes = maskOutcomes(p, multiplier, features);
      features.forEach((feature, i) => {
        close(outcomes.mean[i], (1 - p) * multiplier * feature);
        close(outcomes.variance[i], p * (1 - p) * multiplier ** 2 * feature ** 2);
      });
      momentCases++;
    }
  }
}

const native = JSON.parse(fs.readFileSync('docs/teaching/evidence/dropout-implementation/independent-native.json', 'utf8'));
assert.equal(native.passed, true);
for (const row of native.batchNorm) {
  const actual = normalizationStep(row.inputs);
  for (const [key, expected] of Object.entries(row.expected)) {
    if (Array.isArray(expected)) expected.forEach((value, i) => close(actual[key][i], value));
    else if (typeof expected === 'boolean') assert.equal(actual[key], expected);
    else close(actual[key], expected);
  }
}

const saved = JSON.parse(fs.readFileSync('src/learn/data/dropout-measurements.json', 'utf8'));
for (let specimen = 0; specimen < 3; specimen++) {
  for (let count = 1; count <= saved.samples.length; count++) {
    // Normalize tiny float32 summation drift, then compare an independent
    // average-KL expression with the implemented entropy-difference route.
    const samples = saved.samples.slice(0, count).map(draw => {
      const total = draw[specimen].reduce((sum, value) => sum + value, 0);
      return draw[specimen].map(value => value / total);
    });
    const result = monteCarloSummary(samples);
    const kl = samples.reduce((sum, row) => sum + row.reduce((inner, value, i) =>
      inner + (value === 0 ? 0 : value * Math.log(value / result.mean[i])), 0), 0) / count;
    close(result.disagreement, kl, 2e-13);
    const reversed = monteCarloSummary([...samples].reverse());
    close(result.disagreement, reversed.disagreement, 2e-13);
    result.mean.forEach((mean, i) => {
      close(mean, reversed.mean[i]);
      if (count === 1) assert.equal(result.std[i], null);
      else {
        const secondMoment = samples.reduce((sum, row) => sum + row[i] ** 2, 0) / count;
        close(result.std[i] ** 2, count / (count - 1) * (secondMoment - mean ** 2), 2e-13);
      }
    });
    prefixCases++;
  }
}
// Scoped reviewer corrections: equal branch outputs are compatible with
// different invocation counts, including zero-valued branch inputs.
let executionCases = 0, geometryBoundaryCases = 0;
for (const input of [-1.7, 0, 2.3]) {
  for (const [rates, bits] of [[[0], [0]], [[1], [1]], [[0, .3, 1], [0, 0, 1]],
    [[0, .3, 1], [1, 1, 0]], [[.2, .8, .4], [0, 1, 0]]]) {
    const eager = executeDepth({ rates, bits, input, strategy: 'eager' });
    const lazy = executeDepth({ rates, bits, input, strategy: 'lazy' });
    assert.equal(eager.calls, rates.length);
    assert.equal(lazy.calls, rates.filter((rate, i) => rate === 0 || (rate < 1 && bits[i] === 1)).length);
    close(eager.totalCorrection, lazy.totalCorrection);
    eager.trace.forEach((row, i) => {
      close(row.raw, input * 2);
      close(row.correction, lazy.trace[i].correction);
      assert.equal(lazy.trace[i].raw === null, !Boolean(lazy.trace[i].active));
      assert.equal(lazy.trace[i].called, Boolean(lazy.trace[i].active));
    });
    executionCases++;
  }
}
for (const [mode, shape] of Object.entries(maskShapes)) {
  const bits = Array.from({ length: shape.reduce((a, b) => a * b) }, (_, i) => i % 2);
  for (const probability of [0, .4, 1]) {
    for (const training of [false, true]) {
      const cells = geometryOutput({ mode, bits, probability, training, offset: -1 });
      cells.forEach(cell => {
        const factor = !training || probability === 0 ? 1 : probability === 1 ? 0 : cell.bit / .6;
        close(cell.factor, factor);
        close(cell.output, factor * cell.input);
        // The first activation is zero even if effectively kept: output alone
        // cannot recover the actual boundary action or the fixture bit.
        assert.equal(cell.bit, bits[cell.maskIndex]);
      });
      geometryBoundaryCases++;
    }
  }
}
const paths = ['src/learn/data/dropout-models.js', 'src/learn/data/dropout-measurements.json'];
const result = { passed: true, objectiveCases, momentCases, nativeBatchNormCases: native.batchNorm.length,
  prefixCases, executionCases, geometryBoundaryCases,
  method: 'Exact expected gradient/penalty identity, arbitrary survivor-scale moments, native BN oracle, KL-versus-entropy and second-moment identities for every saved prefix; invocation/output separation and effective geometry boundaries.',
  sourceHashes: Object.fromEntries(paths.map(path => [path, crypto.createHash('sha256').update(fs.readFileSync(path)).digest('hex')])) };
fs.writeFileSync(reportPath, JSON.stringify(result, null, 2) + '\n');
console.log(JSON.stringify(result, null, 2));
