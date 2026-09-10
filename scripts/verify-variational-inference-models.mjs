import assert from 'node:assert/strict';
import { writeFileSync, mkdirSync } from 'node:fs';
import { finiteElbo, finiteApproximation, restrictedFiniteOptimum, gaussianProjection, gaussianKl, covarianceEllipse, coordinateAscent, mixtureProjection, variationalGradients } from '../src/learn/data/variational-inference-models.js';

const close = (actual, expected, tolerance = 1e-10) => assert.ok(Math.abs(actual - expected) <= tolerance, `${actual} != ${expected}`);
let cases = 0;
for (let first = 0; first <= 20; first += 1) {
  for (let share = 0; share <= 20; share += 1) {
    const model = finiteApproximation(first / 20, share / 20);
    const expected = model.q.reduce((sum, q, index) => q === 0 ? sum : sum + q * Math.log([0.2, 0.5, 0.3][index] / q), 0) + Math.log(0.4);
    close(model.elbo, expected);
    close(model.elbo + model.kl, model.logEvidence);
    assert.ok(model.kl >= -1e-14);
    cases += 1;
  }
}
const restricted = restrictedFiniteOptimum();
close(restricted.kl, -Math.log(0.2 + 2 * Math.sqrt(0.15)));
for (let first = 0; first <= 1000; first += 1) assert.ok(finiteApproximation(first / 1000).kl >= restricted.kl - 1e-13);
assert.equal(finiteElbo([0, 1], [1, 0]).kl, Infinity);
assert.equal(finiteElbo([0, 1], [1, 0]).elbo, -Infinity);
close(finiteElbo([1, 0], [1, 0]).kl, 0);
// A large finite log ratio must not become infinite through an intermediate
// division. Exact zero support remains the separate infinite-KL case above.
const tinySupport = finiteElbo([1, 0], [1e-310, 1]);
close(tinySupport.kl, 310 * Math.log(10), 1e-10);
close(tinySupport.elbo + tinySupport.kl, tinySupport.logEvidence, 1e-10);
const roundedPosterior = finiteElbo([1, 0], [1e-310, 1e100]);
assert.equal(roundedPosterior.posterior[0], 0);
close(roundedPosterior.kl, 410 * Math.log(10), 1e-10);
close(roundedPosterior.elbo + roundedPosterior.kl, roundedPosterior.logEvidence, 1e-10);
// Independent sum of two univariate Gaussian KL values, both with a finite
// variance ratio; the determinant ratio alone would overflow.
close(gaussianKl([0, 0], [[1e-100, 0], [0, 1e-100]], [0, 0], [[1e100, 0], [0, 1e100]]), 200 * Math.log(10) - 1, 1e-10);
const scaled = finiteElbo([0.2, 0.4, 0.4], [0.8, 2, 1.2]);
close(scaled.kl, finiteApproximation().kl);
close(scaled.elbo - finiteApproximation().elbo, Math.log(10));

for (let i = -9; i <= 9; i += 1) {
  const rho = i / 10, v = 1 - rho * rho;
  const meanField = gaussianProjection(rho), marginals = gaussianProjection(rho, 'marginals');
  close(meanField.kl, -0.5 * Math.log(v));
  close(marginals.kl, 1 / v - 1 + 0.5 * Math.log(v));
  close(gaussianProjection(rho, 'full').kl, 0);
  for (const point of covarianceEllipse(meanField.target)) {
    close((point[0] ** 2 - 2 * rho * point[0] * point[1] + point[1] ** 2) / v, 1);
  }
  const trace = coordinateAscent(rho, 40);
  for (const row of trace.rows) {
    const [x, y] = row.mean, dx = x - 1, dy = y + 1;
    const independent = -0.5 * Math.log(v) + (dx ** 2 - 2 * rho * dx * dy + dy ** 2) / (2 * v);
    close(row.kl, independent);
    if (row.step) {
      const previous = trace.rows[row.step - 1];
      assert.ok(row.kl <= previous.kl + 1e-10);
      close(row.mean[1 - row.updated], previous.mean[1 - row.updated]);
      // Stationary derivative for the coordinate just optimized.
      close((row.mean[row.updated] - [1, -1][row.updated]) - rho * (row.mean[1 - row.updated] - [1, -1][1 - row.updated]), 0);
    }
  }
  cases += 1;
}

const mixtureFixtures = [];
for (const mean of [-4, -3, 0, 1, 4]) for (const sd of [0.3, 1, 3]) for (const separation of [0, 3, 4]) {
  const model = mixtureProjection(mean, sd, separation);
  close(model.kl, mixtureProjection(mean, sd, separation, 2880).kl, 1e-8);
  close(model.kl, mixtureProjection(-mean, sd, separation).kl, 1e-10);
  close(model.rightProbability + mixtureProjection(-mean, sd, separation).rightProbability, 1, 1e-10);
  if (separation === 0) close(model.kl, -Math.log(sd) + (sd * sd + mean * mean) / 2 - 0.5, 1e-10);
  mixtureFixtures.push({ mean, sd, separation, kl: model.kl, right: model.rightProbability });
  cases += 1;
}

const gradientFixtures = [];
for (const mean of [-2, 0, 1.5, 3]) for (const sd of [0.2, 0.7, 1.2, 2]) {
  const model = variationalGradients({ mean, sd, count: 1000 });
  const h = 1e-5;
  const objective = (m, a) => a - Math.log(0.7) + 0.5 - (Math.exp(2 * a) + (m - 1.5) ** 2) / (2 * 0.49);
  close(model.exactMean, (objective(mean + h, Math.log(sd)) - objective(mean - h, Math.log(sd))) / (2 * h), 1e-8);
  close(model.exactLogSd, (objective(mean, Math.log(sd) + h) - objective(mean, Math.log(sd) - h)) / (2 * h), 1e-8);
  const firstTen = variationalGradients({ mean, sd, count: 10 });
  assert.deepEqual(firstTen.rows, model.rows.slice(0, 10));
  if (mean === 1.5 && sd === 0.7) {
    assert.ok(model.rows.every(row => row.scoreMean === 0 && row.scoreLogSd === 0));
    assert.ok(model.estimates.pathMean.mcse > 0);
  }
  gradientFixtures.push({ mean, sd, exactMean: model.exactMean, exactLogSd: model.exactLogSd, rows: model.rows.slice(0, 10), estimates: model.estimates });
  cases += 1;
}

for (const invalid of [() => finiteApproximation(NaN), () => finiteElbo([0.4, 0.4], [1, 1]), () => finiteElbo([1, 0], [0, 0]), () => gaussianProjection(1), () => gaussianProjection(0, 'unknown'), () => gaussianKl([0, 0], [[1, 2], [2, 1]], [0, 0], [[1, 0], [0, 1]]), () => coordinateAscent(0.8, 2.5), () => mixtureProjection(0, 0), () => variationalGradients({ seed: 0 }), () => variationalGradients({ count: Infinity })]) assert.throws(invalid);
assert.throws(() => { coordinateAscent().rows[0].mean[0] = 0; }, TypeError);
assert.throws(() => { restricted.q.push(0); }, TypeError);
mkdirSync('scratch/variational-inference-native-verification', { recursive: true });
writeFileSync('scratch/variational-inference-native-verification/model-fixtures.json', JSON.stringify({ mixtureFixtures, gradientFixtures }, null, 2));
console.log(`Verified ${cases} compound configurations, three finite log-ratio regressions, restricted-family grid, CAVI stationary conditions, quadrature refinement, gradient finite differences, immutable snapshots and invalid inputs.`);
