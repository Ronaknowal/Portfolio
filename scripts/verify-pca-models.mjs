// Bounded independent checks of the PCA browser models against analytic values,
// exhaustive angle searches and the natively executed Wine record.
import assert from 'node:assert/strict';
import fs from 'node:fs';
import crypto from 'node:crypto';
import { fourPoints, projectAtAngle, principalDirections, rectangleMetric, labelCollisions, wineFullScores, wineValidationCurve, smallestComponentCount, wineReconstruction, storageScalars, meanOf } from '../src/learn/data/pca-models.js';
import { wineRows, wineFullFit, wineSplit, gaussianSpectrum, wineCultivar } from '../src/learn/data/pca-wine-data.js';

const counts = {};
const record = name => { counts[name] = (counts[name] ?? 0) + 1; };
const close = (actual, expected, label, tolerance = 1e-9) => assert(Math.abs(actual - expected) <= tolerance * Math.max(1, Math.abs(expected)), `${label}: ${actual} versus ${expected}`);

// Four-point fixture: manuscript values.
const base = projectAtAngle(fourPoints, 45);
assert.deepEqual(meanOf(fourPoints), [3, 2]);
close(base.total, 20, 'total centered squared length');
close(base.retained, 18, 'retained at 45°');
close(base.sse, 2, 'SSE at 45°');
close(base.scoreVariance, 6, 'sample variance of PC1 scores');
close(base.retainedFraction, 0.9, 'explained fraction');
close(base.scores[0], -3 / Math.SQRT2, 'A score');
base.projections[0].forEach((value, axis) => close(value, [1.5, 0.5][axis], 'A reconstruction'));
base.residuals.forEach((residual, index) => {
  close(residual[0] * base.direction[0] + residual[1] * base.direction[1], 0, `residual ${index} is perpendicular`, 1e-12);
  close(residual[0] ** 2 + residual[1] ** 2, 0.5, `residual ${index} squared length`);
});
for (const [angle, sse] of [[0, 10], [90, 10], [135, 18], [225, 2], [180, 10]]) {
  close(projectAtAngle(fourPoints, angle).sse, sse, `SSE at ${angle}°`);
  record('four-point angle fixtures');
}
// Conservation and the exhaustive optimum for several clouds.
const clouds = {
  original: fourPoints,
  translated: fourPoints.map(point => [point[0] + 2, point[1] - 1]),
  collinear: [[-2, -2], [0, 0], [2, 2]],
  identical: [[1, 1], [1, 1], [1, 1], [1, 1]],
  asymmetric: [[-3, 1], [-1, 0.5], [0, -2], [2, 1.5], [4, -1], [1, 3], [-2, -3]],
  vertical: [[0, -2], [0, -1], [0, 1], [0, 2], [0.5, 0]],
};
for (const [name, points] of Object.entries(clouds)) {
  const fit = principalDirections(points);
  let bestSse = Infinity, bestAngle = null;
  for (let angle = 0; angle < 180; angle += 0.05) {
    const state = projectAtAngle(points, angle);
    close(state.retained + state.sse, state.total, `${name}: retained + SSE = total at ${angle}°`, 1e-9);
    if (state.sse < bestSse) { bestSse = state.sse; bestAngle = angle; }
  }
  if (!fit.degenerate) {
    const fitted = projectAtAngle(points, fit.angles[0]);
    assert(fitted.sse <= bestSse + 1e-6, `${name}: closed-form direction is no worse than the exhaustive search (${fitted.sse} vs ${bestSse})`);
    close(fitted.scoreVariance, fit.eigenvalues[0], `${name}: PC1 variance equals the leading eigenvalue`);
    close(fit.eigenvalues[0] + fit.eigenvalues[1], fit.covariance[0][0] + fit.covariance[1][1], `${name}: eigenvalues sum to the trace`);
    close(fit.eigenvalues[0] * fit.eigenvalues[1], fit.covariance[0][0] * fit.covariance[1][1] - fit.covariance[0][1] ** 2, `${name}: eigenvalues multiply to the determinant`, 1e-8);
    close(fit.directions[0][0] * fit.directions[1][0] + fit.directions[0][1] * fit.directions[1][1], 0, `${name}: directions are orthogonal`, 1e-12);
  }
  record(`cloud ${name}`);
}
assert.deepEqual(principalDirections(clouds.translated).eigenvalues, principalDirections(fourPoints).eigenvalues);
assert.deepEqual(meanOf(clouds.translated), [5, 1]);
close(projectAtAngle(clouds.collinear, 45).sse, 0, 'collinear diagonal SSE');
close(projectAtAngle(clouds.collinear, 135).sse, 16, 'collinear perpendicular SSE');
assert.equal(principalDirections(clouds.identical).degenerate, true);
assert.equal(projectAtAngle(clouds.identical, 30).retainedFraction, null);
assert.equal(principalDirections([[1, 0], [-1, 0], [0, 1], [0, -1]]).tie, true, 'square cloud is a tie');
// Metric lab fixtures from the specification.
close(rectangleMetric(2, 1, 1, false).fractions[0], 0.8, 'raw rectangle PC1 share');
assert.equal(rectangleMetric(2, 1, 1, false).leadingAxis, 'first');
assert.equal(rectangleMetric(2, 1, 2, false).leadingAxis, 'tie', 'm = a/b ties');
assert.equal(rectangleMetric(2, 1, 10, false).leadingAxis, 'second');
assert.equal(rectangleMetric(2, 0.75, 2.6667, false).leadingAxis, 'tie', 'a typed tie multiplier to four decimals reads as a tie');
assert.equal(rectangleMetric(2, 0.75, 2.7, false).leadingAxis, 'second', 'a clearly different multiplier is not a tie');
close(rectangleMetric(2, 1, 10, false).fractions[1], 400 / 416, 'm = 10 second-axis share');
for (const multiplier of [1, 2, 10]) {
  assert.equal(rectangleMetric(2, 1, multiplier, true).leadingAxis, 'tie', `standardized rectangle ties at m = ${multiplier}`);
  record('metric standardized ties');
}
close(rectangleMetric(3, 1, 1, false).variances[0], 3 * 3 * 4 / 3, 'variance uses n − 1');
const scaled = rectangleMetric(4, 2, 1, false), unscaled = rectangleMetric(2, 1, 1, false);
scaled.fractions.forEach((value, axis) => close(value, unscaled.fractions[axis], 'common rescaling keeps ratios'));
close(scaled.variances[0], 4 * unscaled.variances[0], 'common rescaling scales variance by the square');
assert.throws(() => rectangleMetric(0, 1), RangeError);
// Every exposed rectangle parameter combination is supported, including raw
// coordinates beyond the free-point editor's deliberately narrower guard.
for (const a of [0.25, 2, 4]) for (const b of [0.25, 2, 4]) for (const multiplier of [0.25, 5, 10]) {
  const raw = rectangleMetric(a, b, multiplier, false);
  close(raw.variances[0], 4 * a ** 2 / 3, 'rectangle first variance at exposed limits');
  close(raw.variances[1], 4 * (b * multiplier) ** 2 / 3, 'rectangle second variance at exposed limits');
  assert.equal(rectangleMetric(a, b, multiplier, true).leadingAxis, 'tie');
}
assert.throws(() => principalDirections([[0, 0], [21, 1]]), RangeError, 'free-point input bound remains unchanged');
record('rectangle full control range and preserved editor guard');
// Label collisions.
const collide = labelCollisions(10, 1, 'y', 'pc1');
close(collide.retainedFraction, 100 / 101, '99.01% retained');
assert.equal(collide.distinguishable, false);
assert.equal(collide.collisions.length, 2);
assert.equal(collide.distinctLocations, 2);
assert.equal(labelCollisions(10, 1, 'y', 'pc2').distinguishable, true);
assert.equal(labelCollisions(10, 1, 'y', 'both').distinguishable, true);
assert.equal(labelCollisions(10, 1, 'x', 'pc1').distinguishable, true, 'x labels survive PC1');
assert.equal(labelCollisions(10, 1, 'x', 'pc2').distinguishable, false, 'x labels collide on PC2');
assert.deepEqual(labelCollisions(10, 1, 'x', 'pc1').fit.eigenvalues, labelCollisions(10, 1, 'y', 'pc1').fit.eigenvalues, 'label rule does not change the fit');
for (const a of [2, 5, 12]) { assert.equal(labelCollisions(a, 1, 'y', 'pc1').distinguishable, false); record('spread cannot repair the collision'); }
assert.throws(() => labelCollisions(1, 1), RangeError);
// Wine: browser recomputation equals the native record.
const curve = wineValidationCurve();
close(curve.baselineMse, wineSplit.baselineMse, 'validation baseline MSE', 1e-9);
curve.ratios.forEach((ratio, k) => { close(ratio, wineSplit.lossRatios[k], `loss ratio k=${k}`, 1e-8); record('validation loss ratios'); });
assert.equal(curve.ratios[0], 1);
assert(curve.ratios[13] < 1e-12, 'complete basis reconstructs exactly');
for (let k = 1; k <= 13; k += 1) assert(curve.ratios[k] <= curve.ratios[k - 1] + 1e-12, 'nested basis never increases loss');
assert.equal(smallestComponentCount(curve.ratios, 0.10).k, 8);
assert.equal(smallestComponentCount(curve.ratios, 0.06).k, 10);
assert.equal(smallestComponentCount(curve.ratios, 0.11).k, 8);
close(curve.ratios[2], 0.4280959, 'k = 2 ratio', 1e-6);
close(curve.ratios[7], 0.1265121, 'k = 7 ratio', 1e-6);
close(curve.ratios[8], 0.0959546, 'k = 8 ratio', 1e-6);
const scores = wineFullScores(2);
assert.equal(scores.length, 178);
const totalVariance = 13 * 178 / 177;
[0, 1].forEach(component => {
  const mean = scores.reduce((sum, row) => sum + row[component], 0) / 178;
  const variance = scores.reduce((sum, row) => sum + (row[component] - mean) ** 2, 0) / 177;
  close(mean, 0, `PC${component + 1} scores are centered`, 1e-9);
  close(variance / totalVariance, wineFullFit.standardizedRatios[component], `PC${component + 1} score variance matches its ratio`, 1e-8);
  record('full-collection score variance');
});
close(wineFullFit.standardizedRatios.slice(0, 2).reduce((sum, value) => sum + value, 0), 0.5541, 'two-component share', 1e-4);
close(wineFullFit.rawRatios[0], 0.9981, 'raw PC1 share', 1e-4);
const rebuilt = wineReconstruction(0, 13);
rebuilt.rebuiltOriginal.forEach((value, feature) => close(value, rebuilt.original[feature], 'k = 13 recovers the original units', 1e-8));
const partial = wineReconstruction(3, 8);
close(partial.squaredError, partial.residuals.reduce((sum, value) => sum + value * value, 0), 'squared error equals residual sum');
const fullError = wineSplit.validation.reduce((sum, _, position) => sum + wineReconstruction(position, 8).squaredError, 0) / (45 * 13);
close(fullError / curve.baselineMse, curve.ratios[8], 'per-row reconstructions aggregate to the curve');
assert.equal(wineCultivar.filter(label => label === 1).length, 59);
assert.equal(wineRows.length, 178);
close(gaussianSpectrum.ratios.slice(0, 2).reduce((sum, value) => sum + value, 0), 0.2459146, 'Gaussian first two', 1e-6);
close(gaussianSpectrum.ratios.reduce((sum, value) => sum + value, 0), 1, 'Gaussian ratios sum to one', 1e-9);
assert.deepEqual(storageScalars(100, 20, 10), { original: 2000, compressed: 1220 });
assert.deepEqual(storageScalars(1000, 100, 10), { original: 100000, compressed: 11100 });
assert.throws(() => wineReconstruction(45, 2), RangeError);
assert.throws(() => smallestComponentCount(curve.ratios, 0.8), RangeError);

const sources = ['src/learn/data/pca-models.js', 'src/learn/data/pca-wine-data.js', 'src/learn/components/lesson-labs/PcaLabs.jsx', 'src/learn/components/lesson-labs/PcaFigures.jsx', 'src/learn/components/lesson-labs/pca-labs.css'];
const hash = file => crypto.createHash('sha256').update(fs.readFileSync(file)).digest('hex');
const evidence = {
  checkedAt: new Date().toISOString(),
  sourceHashes: Object.fromEntries(sources.filter(file => fs.existsSync(file)).map(file => [file, hash(file)])),
  verifierHash: hash('scripts/verify-pca-models.mjs'),
  counts,
  totalGroupedChecks: Object.values(counts).reduce((sum, value) => sum + value, 0),
  scope: 'Exhaustive 0.05° angle sweeps and conservation on six clouds; analytic four-point, rectangle and label-collision fixtures; browser Wine recomputation against the native 133/45 record and full-collection ratios; storage arithmetic; rejection contracts.',
  limitations: ['Two-dimensional fits only in the browser; Wine directions are natively fitted inputs, not browser eigendecompositions.', 'No claim about arbitrary user input beyond the declared bounds.', 'Rendering, interaction and independent review are separate.'],
  passed: true,
};
fs.mkdirSync('docs/teaching/evidence', { recursive: true });
fs.writeFileSync('docs/teaching/evidence/pca-models.json', JSON.stringify(evidence, null, 2) + '\n');
console.log(`PASS: ${evidence.totalGroupedChecks} grouped PCA model checks.`);
