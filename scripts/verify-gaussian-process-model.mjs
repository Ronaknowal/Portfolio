import assert from 'node:assert/strict';
import fs from 'node:fs';
import crypto from 'node:crypto';
import { gaussianProcessData as data } from '../src/learn/data/gaussian-process-data.js';
import { cholesky, compareTarget, createConditioner, directConditional, forecastFromPrefix, initialObservations, measurementGains, scoreForecast, spatialKernel } from '../src/learn/data/gaussian-process-model.js';
const packet = JSON.parse(fs.readFileSync('docs/teaching/drafts/gaussian-processes-gp/checked-results.json', 'utf8'));
let comparisons = 0;
function close(actual, expected, tolerance = 1e-8) {
  assert.ok(Number.isFinite(actual) && Math.abs(actual - expected) <= tolerance, `${actual} != ${expected}`);
  comparisons++;
}

// Independent route: pivoted row elimination in an augmented matrix, never Cholesky.
function eliminate(matrix, rhs) {
  const rows = matrix.map((row, i) => [...row, rhs[i]]);
  for (let column = 0; column < rows.length; column++) {
    let pivot = column;
    for (let i = column + 1; i < rows.length; i++) if (Math.abs(rows[i][column]) > Math.abs(rows[pivot][column])) pivot = i;
    [rows[pivot], rows[column]] = [rows[column], rows[pivot]];
    const value = rows[column][column];
    assert.ok(Math.abs(value) > 1e-14);
    for (let j = column; j <= rows.length; j++) rows[column][j] /= value;
    for (let i = 0; i < rows.length; i++) if (i !== column) {
      const factor = rows[i][column];
      for (let j = column; j <= rows.length; j++) rows[i][j] -= factor * rows[column][j];
    }
  }
  return rows.map(row => row.at(-1));
}

function reference(x, y, targets, length, noise) {
  const kernel = (a, b) => Math.exp(-(a - b) * (a - b) / (2 * length * length));
  const matrix = x.map((a, i) => x.map((b, j) => kernel(a, b) + (i === j ? noise : 0)));
  const weights = eliminate(matrix, y);
  const cross = targets.map(t => x.map(a => kernel(t, a)));
  const solutions = cross.map(row => eliminate(matrix, row));
  return {
    mean: cross.map(row => row.reduce((sum, v, i) => sum + v * weights[i], 0)),
    covariance: targets.map((a, i) => targets.map((b, j) => kernel(a, b) - cross[i].reduce((sum, v, k) => sum + v * solutions[j][k], 0))),
  };
}

for (const length of [0.1, 0.3, 1, 3, 4]) for (const noise of [0.01, 0.25, 2]) {
  const x = [-1, 0, 0, 1.4, 4], y = [0.3, 1, -1, 4, -4], targets = [-1, 0, 1, 1.4, 2, 4];
  const actual = createConditioner().predict({ x, y, targets, noise, kernel: spatialKernel('rbf', length), kernelKey: String(length) });
  const expected = reference(x, y, targets, length, noise);
  actual.mean.forEach((v, i) => close(v, expected.mean[i], 1e-7));
  actual.covariance.forEach((row, i) => row.forEach((v, j) => close(v, expected.covariance[i][j], 1e-8)));
}
for (const length of [0.3, 1, 3]) {
  const actual = createConditioner().predict({ x: [0, 2], y: [1, -1], targets: [0, 1, 2, 4], kernel: spatialKernel('rbf', length), kernelKey: String(length) });
  const expected = data.tiny[length.toFixed(1)];
  actual.mean.forEach((v, i) => close(v, expected.mean[i]));
  actual.variance.forEach((v, i) => close(v, expected.latent_variance[i]));
  close(actual.logMarginal, expected.log_marginal);
}
const conditioner = createConditioner();
const fixed = { x: [0, 2], targets: [0, 1, 2, 4] };
const first = conditioner.predict({ ...fixed, y: [1, -1] });
const changed = conditioner.predict({ ...fixed, y: [3, -2] });
assert.deepEqual(first.covariance, changed.covariance);
assert.equal(conditioner.factorizations, 1);
changed.mean.forEach((value, i) => close(value, data.tiny.changed_y.mean[i]));
assert.equal(compareTarget({ mean: 0, variance: 1 }, { mean: 0, variance: 1 }), 'neither');
assert.equal(compareTarget({ mean: 0, variance: 1 }, { mean: 1, variance: 1 }), 'mean');
assert.equal(compareTarget({ mean: 0, variance: 1 }, { mean: 0, variance: 0.4 }), 'variance');
assert.equal(compareTarget({ mean: 0, variance: 1 }, { mean: 1, variance: 0.4 }), 'both');
const noData = createConditioner().predict({ x: [], y: [], targets: [1, 4] });
assert.deepEqual(noData.mean, [0, 0]);
assert.deepEqual(noData.variance, [1, 1]);
const noiseless = createConditioner().predict({ x: [0, 2], y: [1, -1], targets: [0, 2], noise: 0 });
noiseless.mean.forEach((v, i) => close(v, [1, -1][i]));
noiseless.variance.forEach(v => close(v, 0));
assert.throws(() => createConditioner().predict({ x: [0, 0], y: [1, -1], targets: [0], noise: 0 }), /positive definite/);
assert.throws(() => cholesky([[1, 2], [2, 1]]), /positive definite/);
assert.throws(() => createConditioner().predict({ x: [NaN], y: [1], targets: [0] }), /finite/);
const direct = directConditional({ rho: 0.5, value: 2, noise: 0.25 });
close(direct.mean, 0.8); close(direct.variance, 0.8); close(direct.observationVariance, 1.05);
for (const key of ['mean', 'variance', 'observationVariance']) close(directConditional({ rho: 0, value: 2, noise: 0.25 })[key], directConditional({ rho: 0, value: -2, noise: 0.25 })[key]);
const gains = measurementGains({ observations: initialObservations, target: 1, candidates: [{ x: 1, noise: 0.25 }, { x: 4, noise: 0.25 }] });
gains.forEach((row, i) => {
  close(row.reduction, data.tiny.design_reductions[i]);
  // Adding the candidate to the actual observation system is a second conditioning route.
  const augmented = reference([0, 2, row.x], [1, -1, 0], [1], 1, 0.25);
  close(row.remainingVariance, augmented.covariance[0][0]);
});
const independent = measurementGains({ observations: initialObservations, target: 1, kind: 'independent', candidates: [{ x: 1, noise: 0.25 }, { x: 4, noise: 0.25 }] });
close(independent[0].reduction, 0.8); close(independent[1].reduction, 0);

for (const cutoff of [72, 84, 96]) for (const family of ['rbf', 'trend_periodic']) {
  const settings = { cutoff, family, horizon: 24 };
  const actual = forecastFromPrefix(data.observations, data.real.development, settings);
  const expected = packet.real.explorer[String(cutoff)][family];
  for (const [key, referenceKey] of [['mean', 'mean'], ['latentSD', 'latent_sd'], ['observationSD', 'observation_sd']]) {
    actual[key].forEach((v, i) => close(v, expected[referenceKey][i], 2e-7));
  }
  const short = forecastFromPrefix(data.observations, data.real.development, { ...settings, horizon: 6 });
  short.mean.forEach((v, i) => close(v, actual.mean[i], 1e-12));
  short.latentSD.forEach((v, i) => close(v, actual.latentSD[i], 1e-12));
  const altered = data.observations.map((row, i) => i >= cutoff ? { ...row, co2: row.co2 + 100 } : row);
  const blind = forecastFromPrefix(altered, data.real.development, settings);
  assert.deepEqual(blind.mean, actual.mean);
  assert.deepEqual(blind.covariance, actual.covariance);
  close(actual.center, data.observations.slice(0, cutoff).reduce((sum, row) => sum + row.co2, 0) / cutoff);
}
for (const cutoff of [72, 73, 83, 95, 96]) for (const horizon of [1, 6, 24]) {
  const actual = forecastFromPrefix(data.observations, data.real.development, { cutoff, horizon, family: 'trend_periodic' });
  assert.equal(actual.rows.length, horizon);
  assert.equal(actual.rows[0].month, data.observations[cutoff].month);
  assert.ok(actual.variance.every(v => Number.isFinite(v) && v >= 0));
}
for (const invalid of [{ cutoff: 71, horizon: 24 }, { cutoff: 97, horizon: 24 }, { cutoff: 72.5, horizon: 24 }, { cutoff: 72, horizon: 0 }, { cutoff: 72, horizon: 25 }]) {
  assert.throws(() => forecastFromPrefix(data.observations, data.real.development, { ...invalid, family: 'rbf' }));
}
for (const [forecast, cutoff] of [[data.real.development.rbf, 72], [data.real.development.trend_periodic, 72], [data.real.final_test, 96]]) {
  const score = scoreForecast(data.observations.slice(cutoff, cutoff + 24), forecast.mean, forecast.observation_sd);
  close(score.mae, forecast.mae); close(score.rmse, forecast.rmse); close(score.covered, forecast.covered);
}
assert.equal(data.real.final_test.covered, 13);
const modelPath = 'src/learn/data/gaussian-process-model.js';
const evidence = { passed: true, numericalComparisons: comparisons, independentMethod: 'Pivoted row elimination of augmented systems; no reuse of Cholesky or the native packet posterior helper', nativeReference: 'NumPy/SciPy/sklearn executed packet fixtures, including six frozen-kernel forecasts', properties: ['value-only covariance identity and factor reuse', 'all four change categories', 'prior with zero observations', 'duplicate noisy observations', 'noise-free interpolation and singular failure', 'invalid covariance rejection', 'candidate gain by conditioning the augmented system', 'independent-value exact null', 'six forecast horizon-prefix identities', 'future-value blindness', 'prefix-only centering', 'integer bounds and nonnegative variances', 'reported MAE/RMSE/coverage'], source: { [modelPath]: crypto.createHash('sha256').update(fs.readFileSync(modelPath)).digest('hex') } };
fs.writeFileSync('docs/teaching/evidence/gaussian-process-model.json', JSON.stringify(evidence, null, 2) + '\n');
console.log(JSON.stringify(evidence, null, 2));
