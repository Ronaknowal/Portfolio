import assert from 'node:assert/strict';
import fs from 'node:fs';
import { parse } from '@babel/parser';
import { maskedUpdate, maskOutcomes, geometryOutput, residualMask, depthRates, expectedActive, executeDepth, normalizationStep, monteCarloSummary } from '../src/learn/data/dropout-models.js';

let assertions = 0;
function close(actual, expected, tolerance = 1e-9) {
  if (Array.isArray(expected)) {
    assert.equal(actual.length, expected.length);
    assertions++;
    expected.forEach((value, index) => close(actual[index], value, tolerance));
  } else {
    assert.ok(Math.abs(actual - expected) <= tolerance, actual + ' differs from ' + expected);
    assertions++;
  }
}
const prepared = JSON.parse(fs.readFileSync('docs/teaching/drafts/dropout-droppath-stochastic-depth/calculated-inputs.json', 'utf8'));
const compact = JSON.parse(fs.readFileSync('src/learn/data/dropout-measurements.json', 'utf8'));
const initial = { features: [1, 2], weights: [1, -.5], mask: [1, 0], probability: .5, target: 1, learningRate: .1 };
const result = maskedUpdate(initial);
for (const [key, nativeKey] of [['output', 'output'], ['loss', 'loss'], ['weightGradient', 'weight_gradient'], ['inputGradient', 'input_gradient'], ['nextWeights', 'updated_weight'], ['nextOutput', 'updated_output'], ['nextLoss', 'updated_loss']]) close(result[key], prepared.mechanisms.one_update[nativeKey]);
close(maskedUpdate({ ...initial, mask: [0, 1] }).weightGradient, [0, -12]);
const dropped = maskedUpdate({ ...initial, mask: [0, 0] });
close(dropped.loss, .5);
close(dropped.weightGradient, [0, 0]);
for (const probability of [0, .125, .25, .5, .75, .99, 1]) {
  for (const training of [true, false]) {
    const point = { ...initial, features: [1.7, -.8], weights: [.3, 1.1], probability, training };
    const exact = maskedUpdate(point);
    for (let index = 0; index < 2; index++) {
      const delta = 1e-6;
      const weights = [...point.weights];
      weights[index] += delta;
      const high = maskedUpdate({ ...point, weights }).loss;
      weights[index] -= 2 * delta;
      const low = maskedUpdate({ ...point, weights }).loss;
      close((high - low) / (2 * delta), exact.weightGradient[index], 2e-6);
    }
    assert.ok(Object.values(exact).flat().every(Number.isFinite));
    assertions++;
  }
}
for (const fixture of prepared.mechanisms.enumerations) {
  const output = maskOutcomes(fixture.drop_probability, 1 / (1 - fixture.drop_probability));
  close(output.mean, fixture.mean);
  close(output.variance, fixture.variance);
  close(output.expectedRelu, fixture.expected_relu);
  close(output.outcomes.map(row => row.probability), fixture.outcomes.map(row => row.probability));
}
close(maskOutcomes(.25, .75).mean, [.5625, 1.125]);
close(maskOutcomes(1, 0).mean, [0, 0]);
for (const [mode, fixture] of Object.entries(prepared.mechanisms.granularity)) {
  const bits = fixture.mask.flat(Infinity);
  close(geometryOutput({ mode, bits, probability: .5, training: true }).map(cell => cell.output), fixture.output.flat(Infinity));
  close(geometryOutput({ mode, bits, probability: .5, training: false }).map(cell => cell.output), Array.from({ length: 16 }, (_, i) => i + 1));
}
for (const training of [true, false]) for (const probability of [0, .5, 1]) {
  const cells = geometryOutput({ mode: 'channel', bits: [0, 1, 0, 1], probability, training, offset: -1 });
  close(cells[0].input, 0);
  for (const cell of cells) {
    const expectedFactor = !training || probability === 0 ? 1 : probability === 1 ? 0 : cell.bit * 2;
    close(cell.factor, expectedFactor);
    close(cell.output, cell.input * expectedFactor);
  }
}
for (const rates of [[0, .5, 1, .5], [0, 0, 0, 0], [1, 1, 1, 1]]) {
  const bits = [0, 1, 1, 0];
  const eager = executeDepth({ rates, bits }), lazy = executeDepth({ rates, bits, strategy: 'lazy' });
  close(eager.calls, 4);
  close(lazy.calls, rates.reduce((sum, rate, i) => sum + (rate === 0 || (rate < 1 && bits[i]) ? 1 : 0), 0));
  close(eager.trace.map(row => row.correction), lazy.trace.map(row => row.correction));
  close(eager.totalCorrection, lazy.totalCorrection);
  assert.ok(eager.trace.every(row => row.raw === 2 && row.called));
  assert.ok(lazy.trace.every(row => row.called ? row.raw === 2 : row.raw === null && row.correction === 0));
  assertions += 2;
}
close(executeDepth({ rates: [0, .5, 1, .5], bits: [0, 1, 1, 0], strategy: 'lazy' }).totalCorrection, 6);
for (const bit of [0, 1]) {
  const input = { input: [2, -1], correction: [.5, 1], probability: .5, bit };
  close(residualMask(input).output, prepared.mechanisms.branch[bit].output);
  close(residualMask({ ...input, placement: 'whole' }).output, prepared.mechanisms.branch[bit].wrong_whole_output);
}
close(residualMask({ input: [1, 3], correction: [.5, 1], probability: .5, bit: 0 }).output, [1, 3]);
for (const fixture of prepared.mechanisms.schedules) {
  for (const convention of ['zero-first', 'original']) {
    const rates = depthRates(fixture.blocks, fixture.endpoint, convention);
    const key = convention.replace('-', '_');
    close(rates, fixture[key + '_rates']);
    close(expectedActive(rates), fixture[key + '_active']);
  }
}
const state = normalizationStep({ values: [0, 2, 0, 6], runningMean: 0, runningVariance: 1, batches: 0, batchTraining: true, recordGradients: false });
close(state.runningVariance, 8);
close(state.runningMean, 2);
close(state.batches, 1);
const clean = normalizationStep({ ...state, values: [1, 3], batchTraining: false, recordGradients: true });
close(clean.output, prepared.mechanisms.normalization.eval_output);
close(clean.batches, 1);
close(clean.runningVariance, 8);
for (let specimen = 0; specimen < 3; specimen++) {
  for (const count of [1, 10, 37, 100]) {
    const rows = compact.samples.slice(0, count).map(draw => draw[specimen]);
    const summary = monteCarloSummary(rows);
    close(summary.mean.reduce((a, b) => a + b, 0), 1, 2e-7);
    assert.ok(summary.disagreement >= 0 && summary.predictiveEntropy >= 0);
    assertions++;
    if (count === 1) {
      close(summary.disagreement, 0);
      assert.ok(summary.std.every(value => value === null));
      assertions++;
    }
    if (count === 100) {
      close(summary.mean, prepared.monte_carlo.mean_probabilities[specimen], 1e-7);
      close(summary.std, prepared.monte_carlo.probability_std[specimen], 1e-7);
      close(summary.predictiveEntropy, prepared.monte_carlo.predictive_entropy[specimen], 2e-7);
      close(summary.disagreement, prepared.monte_carlo.disagreement[specimen], 2e-7);
    }
  }
}
assert.deepEqual(compact.fits, prepared.fits);
assert.deepEqual(compact.samples, prepared.monte_carlo.sample_probabilities);
assertions += 2;
const csv = fs.readFileSync('docs/teaching/drafts/dropout-droppath-stochastic-depth/digits-400.csv', 'utf8').trim().split(/\r?\n/).slice(1).map(row => row.split(',').map(Number));
for (const specimen of compact.specimens) {
  const row = csv.find(values => values[0] === specimen.id);
  assert.deepEqual(specimen.pixels, row.slice(1, 65));
  assert.equal(specimen.target, row[65]);
  assertions += 2;
}
const labs = fs.readFileSync('src/learn/components/lesson-labs/DropoutLabs.jsx', 'utf8');
const lesson = fs.readFileSync('src/learn/data/topics/dropout-droppath-stochastic-depth.jsx', 'utf8');
for (const source of [labs, lesson]) parse(source, { sourceType: 'module', plugins: ['jsx'] });
for (const filename of [...lesson.matchAll(/\/learn-assets\/dropout-droppath-stochastic-depth\/([^"}]+)[\"]/g)].map(match => match[1])) {
  assert.ok(fs.existsSync('public/learn-assets/dropout-droppath-stochastic-depth/' + filename));
  assertions++;
}
const report = { passed: true, assertions, groups: ['native fixture parity', 'finite-difference gradients', 'boundaries and effective geometry', 'residuals', 'schedules and actual counted branch calls', 'state', 'saved MC prefixes', 'recorded fit and pixel conservation', 'JSX and downloads'] };
fs.writeFileSync('docs/teaching/evidence/dropout-implementation/model-checks.json', JSON.stringify(report, null, 2) + '\n');
console.log(JSON.stringify(report, null, 2));
