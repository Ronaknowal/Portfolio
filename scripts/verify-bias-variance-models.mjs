// Bounded independent checks of the bias-variance browser models against the
// content phase's recorded calculations, the manuscript's worked values and
// analytic identities, plus structural checks of the generated data module and
// the displayed programs.
// Run: node scripts/verify-bias-variance-models.mjs
import assert from 'node:assert/strict';
import fs from 'node:fs';
import crypto from 'node:crypto';
import {
  averageVariance, brierDecomposition, defaultGrid, degreeComparison, doubleDescentBranch,
  doubleDescentNoise, doubleDescentRisk, doubleDescentSignalNorm, evaluatePolynomial, finiteExperiment,
  firstCrossing, fixedDesignOptimism, fixtures, hiddenSettingVariance, leastSquares, learningCurveSeries,
  limits, meanOf, newLocationRisk, optimismDesign, polynomialDesign, polynomialRow, populationVariance,
  predictionSpread, predictionWeights, projectionSmoother, restrictionEffect, smootherOptimism,
  solveLinear, spreadComparison, thresholdedError, thresholdedErrorAt, trajectorySeries,
  validationCurveSeries,
} from '../src/learn/data/bias-variance-models.js';
import {
  boostingTrajectory, minSamplesLeafGrid, printedRounds, procedures, provenance, rawFeatureLabels,
  rawFeatureNames, rawFeatureRanges, targetRange, trainSizes, validationCurveRecord,
} from '../src/learn/data/bias-variance-data.js';

const packetDirectory = 'docs/teaching/drafts/bias-variance-tradeoff-learning-curves';
const recorded = JSON.parse(fs.readFileSync(`${packetDirectory}/calculated-inputs.json`, 'utf8'));
const examples = JSON.parse(fs.readFileSync('src/learn/data/bias-variance-examples.js', 'utf8')
  .replace(/^[\s\S]*?export const biasVarianceExamples = /, '').replace(/;\s*$/, ''));

const counts = {};
const record = name => { counts[name] = (counts[name] ?? 0) + 1; };
const close = (actual, expected, label, tolerance = 1e-9) =>
  assert(Math.abs(actual - expected) <= tolerance * Math.max(1, Math.abs(expected)), `${label}: ${actual} versus ${expected}`);
const vector = (actual, expected, label, tolerance = 1e-9) => {
  assert.equal(actual.length, expected.length, `${label}: length ${actual.length} versus ${expected.length}`);
  actual.forEach((value, index) => close(value, expected[index], `${label}[${index}]`, tolerance));
};
/** A value the prose tells a learner to type must land on the control's step. */
const onStep = (value, step) => Math.abs(value / step - Math.round(value / step)) < 1e-9;

/* ------------------------------------------- §1 · the three sensor worlds */

const sensorA = predictionSpread(fixtures.sensorA.predictions, fixtures.sensorA.trueMean, fixtures.sensorA.noiseSpread);
const sensorB = predictionSpread(fixtures.sensorB.predictions, fixtures.sensorB.trueMean, fixtures.sensorB.noiseSpread);
close(sensorA.averagePrediction, 10, 'procedure A averages to the true mean', 1e-12);
close(sensorA.squaredBias, 0, 'so its squared bias is zero', 1e-12);
close(sensorA.variance, 8 / 3, 'its prediction variance is 8/3', 1e-12);
close(sensorA.noiseVariance, 1, 'the fresh outcomes have variance 1', 1e-12);
close(sensorA.total, 11 / 3, 'its total expected squared error is 11/3', 1e-12);
close(sensorB.averagePrediction, 9, 'procedure B always predicts 9', 1e-12);
close(sensorB.squaredBias, 1, 'its squared bias is 1', 1e-12);
close(sensorB.variance, 0, 'with no prediction variance at all', 1e-12);
close(sensorB.total, 2, 'total 2, which is lower than A', 1e-12);
assert(sensorB.total < sensorA.total, 'the stable biased procedure wins in this example');
record('three-world table');

// The decomposition and the direct six-pair average are the same number, every
// time. This is the independent route, not the same helper called twice.
for (const setup of [
  fixtures.sensorA, fixtures.sensorB,
  { predictions: [9, 10, 11], trueMean: 10, noiseSpread: 1 },
  { predictions: [6, 10, 14], trueMean: 10, noiseSpread: 1 },
  { predictions: [10, 10, 10], trueMean: 10, noiseSpread: 1 },
  { predictions: [9, 11, 13], trueMean: 10, noiseSpread: 1 },
  { predictions: [3, 4, 8], trueMean: 5, noiseSpread: 0 },
  { predictions: [0.25, 19.75], trueMean: 7.5, noiseSpread: 4 },
]) {
  const answer = predictionSpread(setup.predictions, setup.trueMean, setup.noiseSpread);
  const direct = meanOf(setup.predictions.flatMap(prediction =>
    [setup.trueMean - setup.noiseSpread, setup.trueMean + setup.noiseSpread]
      .map(outcome => (outcome - prediction) ** 2)));
  close(answer.total, direct, 'the decomposition equals the enumerated pair average', 1e-12);
  close(answer.pairAverage, direct, 'and so does the reported pair average', 1e-12);
  assert(answer.agrees, 'and the model says so');
  assert(answer.pairs.length === setup.predictions.length * 2, 'every prediction meets every outcome');
  close(answer.pairs.reduce((sum, pair) => sum + pair.probability, 0), 1, 'the pair probabilities sum to one', 1e-12);
  assert(answer.squaredBias >= 0 && answer.variance >= 0 && answer.noiseVariance >= 0, 'all three terms are nonnegative');
  record('decomposition against direct enumeration');
}
// Practice 1, by hand: predictions 3, 4, 8 about a true mean of 5.
const practiceOne = predictionSpread([3, 4, 8], 5, 0);
close(practiceOne.squaredBias, 0, 'practice 1 has zero bias', 1e-12);
close(practiceOne.variance, 14 / 3, 'practice 1 variance is 14/3', 1e-12);
close(practiceOne.variance + 2, 20 / 3, 'and its total with noise variance 2 is 20/3', 1e-12);
close(predictionSpread([4, 4, 4], 5, 0).squaredBias + 2, 3, 'the constant procedure totals 3', 1e-12);
record('practice 1');

// The eight suggested setups of investigation 1, each against the baseline.
const spreadSetups = [
  { label: 'inward', predictions: [9, 10, 11], trueMean: 10, noiseSpread: 1, outcome: 'decrease', total: 5 / 3 },
  { label: 'outward', predictions: [6, 10, 14], trueMean: 10, noiseSpread: 1, outcome: 'increase', total: 35 / 3 },
  { label: 'collapsed', predictions: [10, 10, 10], trueMean: 10, noiseSpread: 1, outcome: 'decrease', total: 1 },
  { label: 'shifted', predictions: [9, 11, 13], trueMean: 10, noiseSpread: 1, outcome: 'increase', total: 14 / 3 },
  { label: 'constant B', predictions: [9, 9, 9], trueMean: 10, noiseSpread: 1, outcome: 'decrease', total: 2 },
  { label: 'permuted', predictions: [12, 10, 8], trueMean: 10, noiseSpread: 1, outcome: 'unchanged', total: 11 / 3 },
  { label: 'translated', predictions: [10, 12, 14], trueMean: 12, noiseSpread: 1, outcome: 'unchanged', total: 11 / 3 },
];
for (const setup of spreadSetups) {
  const comparison = spreadComparison(fixtures.sensorA, setup);
  assert.equal(comparison.outcome, setup.outcome, `${setup.label} moves the total ${setup.outcome}`);
  close(comparison.after.total, setup.total, `${setup.label} total`, 1e-12);
  setup.predictions.forEach(value => assert(onStep(value, 0.25), `${setup.label} is typeable at the 0.25 step`));
  assert(onStep(setup.trueMean, 0.25) && onStep(setup.noiseSpread, 0.25), `${setup.label} mean and spread are typeable`);
  record('investigation 1 setup');
}
// The two nulls really are nulls: every term, not only the total.
for (const label of ['permuted', 'translated']) {
  const setup = spreadSetups.find(entry => entry.label === label);
  const comparison = spreadComparison(fixtures.sensorA, setup);
  close(comparison.after.squaredBias, comparison.before.squaredBias, `${label} leaves the squared bias`, 1e-12);
  close(comparison.after.variance, comparison.before.variance, `${label} leaves the variance`, 1e-12);
  close(comparison.after.noiseVariance, comparison.before.noiseVariance, `${label} leaves the noise`, 1e-12);
  assert(!comparison.changed.squaredBias && !comparison.changed.variance && !comparison.changed.noiseVariance,
    `${label} reports no moved term`);
  record('investigation 1 null');
}
// Same bias, different spread; and same spread, different bias.
const collapsed = spreadComparison(fixtures.sensorA, spreadSetups.find(entry => entry.label === 'collapsed'));
assert(collapsed.changed.variance && !collapsed.changed.squaredBias, 'collapsing the dots moves only the spread');
const shifted = spreadComparison(fixtures.sensorA, spreadSetups.find(entry => entry.label === 'shifted'));
assert(shifted.changed.squaredBias && !shifted.changed.variance, 'shifting the dots moves only the offset');
close(shifted.after.variance, 8 / 3, 'a rigid shift leaves the variance at 8/3', 1e-12);
close(shifted.after.squaredBias, 1, 'and makes the squared bias 1', 1e-12);
record('investigation 1 single-term moves');

// The ruler must be wide enough for every mark it draws.
for (const spread of [0, 1, 2, 4]) {
  for (const mean of [0, 10, 20]) {
    const answer = predictionSpread([0, 10, 20], mean, spread);
    assert(answer.domain[0] <= Math.min(...answer.outcomes, ...answer.predictions), 'the ruler covers the lowest mark');
    assert(answer.domain[1] >= Math.max(...answer.outcomes, ...answer.predictions), 'and the highest');
    record('ruler domain');
  }
}
assert.throws(() => predictionSpread([8, 10, 21], 10, 1), RangeError, 'a prediction outside the control range is refused');
assert.throws(() => predictionSpread([8, 10, Number.NaN], 10, 1), RangeError, 'a non-finite prediction is refused');
assert.throws(() => predictionSpread([8], 10, 1), RangeError, 'one prediction is not a distribution of training draws');
assert.throws(() => predictionSpread([8, 10, 12], 10, 5), RangeError, 'a spread beyond the control range is refused');
record('investigation 1 refusals');

/* --------------------------------------------- small dense linear algebra */

vector(solveLinear([[2, 1], [1, 3]], [3, 5]), [0.8, 1.4], 'the small solver is correct', 1e-12);
vector(solveLinear([[4, 0, 0], [0, 2, 0], [0, 0, 1]], [8, 2, 3]), [2, 1, 3], 'and on a diagonal system', 1e-12);
assert.throws(() => solveLinear([[0, 0], [0, 0]], [1, 1]), RangeError, 'a singular system is refused');
assert.throws(() => solveLinear([[1, 2]], [1]), RangeError, 'a non-square system is refused');
vector(polynomialRow(2, 3), [1, 2, 4, 8], 'the design row is increasing powers', 1e-12);
vector(leastSquares(polynomialDesign([-1, 0, 1], 1), [0, 1, 2]), [1, 1], 'a line through three collinear points', 1e-12);
close(evaluatePolynomial([1, 1, 1], 2), 7, 'polynomial evaluation', 1e-12);
close(populationVariance([1, 2, 3]), 2 / 3, 'population variance divides by N', 1e-12);
assert.throws(() => leastSquares(polynomialDesign([-1, 1], 2), [0, 1]), RangeError,
  'fewer rows than fitted coefficients is refused');
record('linear algebra');

/* ---------------------------- §3 · interpolation weights, independently */

const quadraticWeights = predictionWeights([-1, 0, 1], 2, 0.5);
vector(quadraticWeights.weights, [-0.125, 0.75, 0.375], 'the stated interpolation weights', 1e-12);
close(quadraticWeights.sum, 1, 'they sum to one', 1e-12);
assert.equal(quadraticWeights.negativeCount, 1, 'exactly one of them is negative');
close(0.25 * quadraticWeights.sumOfSquares, 23 / 128, 'sigma squared times the squared weights is 23/128', 1e-12);
// A prediction really is that fixed combination of the observed targets, for
// every design, degree and probe in the supported domain.
for (const trainX of [fixtures.threeInputs, fixtures.fiveInputs]) {
  for (let degree = 0; degree <= 2; degree += 1) {
    for (const probe of [-1.5, -0.75, 0, 0.35, 0.5, 1.5]) {
      const { weights, sum } = predictionWeights(trainX, degree, probe);
      close(sum, 1, `the weights sum to one at degree ${degree}, probe ${probe}`, 1e-9);
      const targets = trainX.map((value, index) => 3 - 2 * value + 0.5 * value * value + 0.17 * index);
      const fitted = leastSquares(polynomialDesign(trainX, degree), targets);
      close(weights.reduce((total, weight, index) => total + weight * targets[index], 0),
        evaluatePolynomial(fitted, probe), 'the weighted sum equals the fitted prediction', 1e-9);
      record('prediction weights');
    }
  }
}

/* ------------------------- §3 · every recorded finite experiment, in full */

assert.equal(recorded.finiteExperiments.length, 21, 'the packet recorded twenty-one finite experiments');
assert.equal(recorded.environment.numpy, '2.3.5');
assert.equal(recorded.environment.sklearn, '1.9.1');
let comparedGridValues = 0;
for (const saved of recorded.finiteExperiments) {
  const experiment = finiteExperiment({
    trainX: saved.trainX, degree: saved.degree, curvature: saved.curvature,
    sigma: saved.sigma, probe: 0.5, grid: saved.grid,
  });
  assert.equal(experiment.worldCount, 2 ** saved.trainX.length, 'every sign combination is one world');
  assert.equal(experiment.worlds.length, saved.targets.length, 'the world count matches the packet');
  experiment.worlds.forEach((world, index) => {
    vector(world.targets, saved.targets[index], `world ${index} targets`, 1e-12);
    close(world.probability, 1 / experiment.worldCount, 'each world is equally likely', 1e-12);
  });
  vector(experiment.truthCurve, saved.truth, 'the true mean curve', 1e-9);
  experiment.curves.forEach((curve, index) => vector(curve, saved.predictions[index], `world ${index} curve`, 1e-8));
  vector(experiment.gridDecomposition.map(entry => entry.mean), saved.mean, 'the average fitted curve', 1e-8);
  vector(experiment.gridDecomposition.map(entry => entry.squaredBias), saved.biasSquared, 'the squared bias curve', 1e-8);
  vector(experiment.gridDecomposition.map(entry => entry.variance), saved.variance, 'the variance curve', 1e-8);
  vector(experiment.gridDecomposition.map(entry => entry.expectedError), saved.expectedError, 'the expected-error curve', 1e-8);
  vector(experiment.averageCurve, saved.mean, 'the drawn average curve is the same average', 1e-8);
  // The identity holds pointwise, not only at one probe.
  experiment.gridDecomposition.forEach(entry => {
    const excess = entry.expectedError - saved.sigma ** 2;
    close(excess, entry.squaredBias + entry.variance, 'bias squared plus variance is the excess', 1e-8);
  });
  assert(saved.identityMaxError < 1e-12, 'the packet recorded a negligible identity residual');
  comparedGridValues += saved.grid.length * 4 + saved.predictions.length * saved.grid.length;
  record('recorded finite experiment');
}
assert(comparedGridValues > 12000, `expected a large grid comparison, got ${comparedGridValues}`);

/* ----------------------------- §3 · the manuscript's stated probe values */

const base = { trainX: fixtures.threeInputs, curvature: 1, sigma: 0.5 };
const atProbe = (settings, probe) => finiteExperiment({ ...settings, probe, grid: [probe] });
const statedAtHalf = [
  { degree: 0, mean: 5 / 3, squaredBias: 1 / 144, variance: 1 / 12, total: 49 / 144 },
  { degree: 1, mean: 13 / 6, squaredBias: 25 / 144, variance: 11 / 96, total: 155 / 288 },
  { degree: 2, mean: 7 / 4, squaredBias: 0, variance: 23 / 128, total: 55 / 128 },
];
for (const stated of statedAtHalf) {
  const answer = atProbe({ ...base, degree: stated.degree }, 0.5);
  close(answer.meanPrediction, stated.mean, `degree ${stated.degree} mean prediction`, 1e-12);
  close(answer.squaredBias, stated.squaredBias, `degree ${stated.degree} squared bias`, 1e-12);
  close(answer.variance, stated.variance, `degree ${stated.degree} prediction variance`, 1e-12);
  close(answer.noiseVariance, 0.25, `degree ${stated.degree} target noise`, 1e-12);
  close(answer.expectedError, stated.total, `degree ${stated.degree} expected error`, 1e-12);
  close(answer.varianceFromWeights, stated.variance, 'and the weight route agrees', 1e-12);
  assert(answer.identityResidual < 1e-12, 'with a negligible identity residual');
  record('section 3 stated value');
}
close(atProbe({ ...base, degree: 0 }, 0.5).expectedError, 0.340278, 'the printed constant error', 1e-6);
close(atProbe({ ...base, degree: 1 }, 0.5).expectedError, 0.538194, 'the printed line error', 1e-6);
close(atProbe({ ...base, degree: 2 }, 0.5).expectedError, 0.429688, 'the printed quadratic error', 1e-6);
assert(atProbe({ ...base, degree: 0 }, 0.5).expectedError < atProbe({ ...base, degree: 2 }, 0.5).expectedError,
  'the constant is best at this particular probe');
// At the zero probe the ranking reverses and the constant is badly biased.
close(atProbe({ ...base, degree: 0 }, 0).squaredBias, 4 / 9, 'the constant squared bias at zero', 1e-12);
close(atProbe({ ...base, degree: 2 }, 0).expectedError, 0.5, 'the quadratic expected error at zero', 1e-12);
assert(atProbe({ ...base, degree: 2 }, 0).expectedError < atProbe({ ...base, degree: 0 }, 0).expectedError,
  'the quadratic is better at the zero probe');
record('section 3 probe reversal');

// c = 0: a line already represents this truth.
close(atProbe({ ...base, curvature: 0, degree: 1 }, 0.5).expectedError, 35 / 96, 'the c = 0 line error', 1e-12);
close(atProbe({ ...base, curvature: 0, degree: 1 }, 0.5).squaredBias, 0, 'with zero bias', 1e-12);
close(atProbe({ ...base, curvature: 0, degree: 2 }, 0.5).expectedError, 55 / 128, 'the c = 0 quadratic error', 1e-12);
close(atProbe({ ...base, curvature: 0, degree: 0 }, 0.5).expectedError, 7 / 12, 'the c = 0 constant error', 1e-12);
// Louder noise scales the variance, and practice 2's answer.
const louder = atProbe({ ...base, sigma: 1, degree: 2 }, 0.5);
close(louder.variance, 23 / 32, 'practice 2 variance', 1e-12);
close(louder.expectedError, 55 / 32, 'practice 2 total', 1e-12);
close(louder.squaredBias, 0, 'and the bias stays exactly zero', 1e-12);
close(louder.variance, 4 * atProbe({ ...base, degree: 2 }, 0.5).variance, 'doubling sigma quadruples the variance', 1e-12);
assert(onStep(1, 0.05) && onStep(0.5, 0.05) && onStep(1, 0.05), 'practice 2 is typeable at the investigation step');
record('section 3 changed settings');

// Five inputs: the design-specific reversal the checkpoint claims.
const fiveBase = { trainX: fixtures.fiveInputs, curvature: 1, sigma: 0.5 };
const fiveConstant = atProbe({ ...fiveBase, degree: 0 }, 0.5);
const threeConstant = atProbe({ ...base, degree: 0 }, 0.5);
close(fiveConstant.expectedError, 0.3625, 'the five-input constant error', 1e-12);
close(atProbe({ ...fiveBase, degree: 1 }, 0.5).expectedError, 0.3875, 'the five-input line error', 1e-12);
close(atProbe({ ...fiveBase, degree: 2 }, 0.5).expectedError, 12 / 35, 'the five-input quadratic error', 1e-12);
assert(fiveConstant.expectedError > threeConstant.expectedError, 'adding the two inputs worsens the constant here');
assert(fiveConstant.variance < threeConstant.variance, 'even though its variance falls');
close(fiveConstant.variance, 0.05, 'the five-input constant variance', 1e-12);
close(threeConstant.variance, 1 / 12, 'the three-input constant variance', 1e-12);
close(fiveConstant.squaredBias, 0.0625, 'its squared bias rises to 0.0625', 1e-12);
close(fiveConstant.meanPrediction, 1.5, 'because its average level shifts to 1.5', 1e-12);
assert.equal(finiteExperiment({ ...fiveBase, degree: 2, probe: 0.5, grid: [0.5] }).worldCount, 32,
  'five inputs enumerate thirty-two worlds');
record('five-input design contrast');

// Nulls: no noise, and two degrees that coincide at a probe.
const noiseless = atProbe({ trainX: fixtures.threeInputs, curvature: 0, sigma: 0, degree: 1 }, 0.5);
close(noiseless.variance, 0, 'with no noise there is no prediction variance', 1e-12);
close(noiseless.expectedError, 0, 'and no error at all', 1e-12);
close(atProbe({ trainX: fixtures.threeInputs, curvature: 0, sigma: 0, degree: 2 }, 0.5).expectedError, 0,
  'the quadratic ties it exactly', 1e-12);
const coincide = degreeComparison({ ...base, probe: 0 }, 0, 1);
assert.equal(coincide.outcome, 'same', 'constant and line tie at the zero probe');
assert(coincide.identicalPredictions, 'and they agree in every single world, not just on average');
record('section 3 nulls');

/* ------------------------------- §3 · the comparisons the lab can be asked */

const labPresets = [
  { label: 'default', settings: { ...base, probe: 0.5 }, reference: 1, candidate: 2, outcome: 'lower' },
  { label: 'flat truth', settings: { ...base, curvature: 0, probe: 0.5 }, reference: 1, candidate: 2, outcome: 'higher' },
  { label: 'probe zero', settings: { ...base, probe: 0 }, reference: 0, candidate: 1, outcome: 'same' },
  { label: 'noiseless', settings: { ...base, curvature: 0, sigma: 0, probe: 0.5 }, reference: 1, candidate: 2, outcome: 'same' },
  { label: 'louder', settings: { ...base, sigma: 1, probe: 0.5 }, reference: 1, candidate: 2, outcome: 'higher' },
  { label: 'five inputs', settings: { ...fiveBase, probe: 0.5 }, reference: 0, candidate: 2, outcome: 'lower' },
  { label: 'negative curvature', settings: { ...base, curvature: -1, probe: 0.5 }, reference: 1, candidate: 2, outcome: 'lower' },
  { label: 'edge probe', settings: { ...base, probe: -1.5 }, reference: 1, candidate: 2, outcome: 'lower' },
];
// The offered setups must not all agree, and must include a genuine tie.
assert.equal(new Set(labPresets.map(preset => preset.outcome)).size, 3,
  'the offered setups expose a lower, a higher and an exactly equal verdict');
assert.equal(labPresets.filter(preset => preset.outcome === 'same').length, 2, 'two of them are nulls');
for (const preset of labPresets) {
  const comparison = degreeComparison(preset.settings, preset.reference, preset.candidate);
  assert.equal(comparison.outcome, preset.outcome, `${preset.label} gives a ${preset.outcome} candidate`);
  assert(onStep(preset.settings.curvature, 0.05), `${preset.label} curvature is typeable`);
  assert(onStep(preset.settings.sigma, 0.05), `${preset.label} sigma is typeable`);
  assert(onStep(preset.settings.probe, 0.05), `${preset.label} probe is typeable`);
  assert(preset.settings.curvature >= limits.curvature.minimum && preset.settings.curvature <= limits.curvature.maximum,
    `${preset.label} curvature is inside the control range`);
  assert(preset.settings.sigma >= limits.sigma.minimum && preset.settings.sigma <= limits.sigma.maximum,
    `${preset.label} sigma is inside the control range`);
  assert(preset.settings.probe >= limits.probe.minimum && preset.settings.probe <= limits.probe.maximum,
    `${preset.label} probe is inside the control range`);
  // Exchanging reference and candidate mirrors the verdict.
  const swapped = degreeComparison(preset.settings, preset.candidate, preset.reference);
  const mirror = { lower: 'higher', higher: 'lower', same: 'same' }[preset.outcome];
  assert.equal(swapped.outcome, mirror, `${preset.label} mirrors under exchange`);
  close(swapped.difference, -comparison.difference, 'and its difference flips sign', 1e-9);
  record('degree comparison preset');
}
// Raising the noise flips this verdict: at sigma = 1 the quadratic's extra
// variance outgrows the line's fixed squared bias.
close(degreeComparison({ ...base, sigma: 1, probe: 0.5 }, 1, 2).reference.expectedError, 1.631944, 'the louder line', 1e-6);
close(degreeComparison({ ...base, sigma: 1, probe: 0.5 }, 1, 2).candidate.expectedError, 55 / 32, 'the louder quadratic', 1e-9);
// Flipping the sign of the curvature is a symmetry, not a new number: only the
// magnitude of the line's offset at this probe matters, and the quadratic's
// variance does not depend on the truth at all.
const positiveCurvature = degreeComparison({ ...base, probe: 0.5 }, 1, 2);
const negativeCurvature = degreeComparison({ ...base, curvature: -1, probe: 0.5 }, 1, 2);
close(negativeCurvature.reference.expectedError, positiveCurvature.reference.expectedError,
  'the line costs the same under a mirrored truth', 1e-12);
close(negativeCurvature.candidate.variance, positiveCurvature.candidate.variance,
  'and the quadratic variance is unchanged by the truth', 1e-12);
assert(negativeCurvature.reference.squaredBias > 0 && negativeCurvature.candidate.squaredBias < 1e-12,
  'with the bias still on the line alone');
record('curvature symmetry');
// A degree the design cannot support is refused, not silently reduced.
assert.throws(() => finiteExperiment({ trainX: [-1, 1], degree: 2, curvature: 1, sigma: 0.5, probe: 0 }), RangeError,
  'a degree beyond the design rank is refused');
assert.throws(() => finiteExperiment({ trainX: [-1, 0, -1], degree: 1, curvature: 1, sigma: 0.5, probe: 0 }), RangeError,
  'repeated training inputs are refused');
assert.throws(() => finiteExperiment({ ...base, degree: 3, probe: 0 }), RangeError, 'degree three is outside the model');
assert.throws(() => finiteExperiment({ ...base, degree: 1, probe: 2 }), RangeError, 'a probe outside the range is refused');
assert.throws(() => finiteExperiment({ ...base, degree: 1, sigma: 2, probe: 0 }), RangeError, 'too much noise is refused');
assert.throws(() => finiteExperiment({ trainX: [-1, -0.9, -0.5, 0, 0.5, 1], degree: 1, curvature: 1, sigma: 0.5, probe: 0 }),
  RangeError, 'more than five training inputs would exceed the bounded enumeration');
record('finite experiment refusals');

// A broad sweep: the identity, nonnegativity and the two variance routes agree.
let sweepCount = 0;
for (const curvature of [-1, -0.35, 0, 1, 2]) {
  for (const sigma of [0, 0.05, 0.5, 1, 1.5]) {
    for (const degree of [0, 1, 2]) {
      for (const probe of [-1.5, -0.5, 0, 0.5, 1.25]) {
        const answer = finiteExperiment({ trainX: fixtures.threeInputs, curvature, sigma, degree, probe, grid: [probe] });
        assert(answer.identityResidual <= 1e-9, 'the identity holds across the sweep');
        assert(answer.squaredBias >= 0 && answer.variance >= 0, 'both terms stay nonnegative');
        close(answer.variance, answer.varianceFromWeights, 'both variance routes agree', 1e-9);
        close(answer.expectedError, answer.squaredBias + answer.variance + sigma ** 2, 'the total adds up', 1e-9);
        close(answer.weightSum, 1, 'the weights sum to one', 1e-9);
        if (degree === 2) close(answer.squaredBias, 0, 'a correctly specified quadratic is unbiased', 1e-9);
        sweepCount += 1;
      }
    }
  }
}
assert.equal(sweepCount, 375, 'the sweep covered every combination');
record('finite experiment sweep');
assert.equal(defaultGrid.length, limits.gridPoints);
close(defaultGrid[0], -1.5, 'the default grid starts at -1.5', 1e-12);
close(defaultGrid.at(-1), 1.5, 'and ends at 1.5', 1e-12);
record('default grid');

/* --------------------------------------------- §2 · the hidden setting */

const hidden = hiddenSettingVariance();
close(hidden.givenInputOnly, 1.25, 'observing X alone leaves 1.25', 1e-12);
close(hidden.givenInputAndSetting, 0.25, 'observing X and Z leaves 0.25', 1e-12);
close(hidden.explainedByMeasuringSetting, 1, 'measuring Z explains exactly 1', 1e-12);
close(hidden.givenInputOnly - hidden.givenInputAndSetting, hidden.explainedByMeasuringSetting,
  'the law of total variance closes', 1e-12);
assert.throws(() => hiddenSettingVariance([-1, 1], -1), RangeError, 'a negative variance is refused');
record('hidden setting');

/* ----------------------------------- §5 · the recorded real-data curves */

const series = Object.fromEntries(procedures.map(entry => [entry.model, learningCurveSeries(entry)]));
assert.deepEqual(Object.keys(series).sort(), ['mean', 'ridge', 'tree_leaf1', 'tree_leaf20'].sort());
for (const entry of procedures) {
  const saved = recorded.real.learning[entry.model];
  assert.deepEqual(entry.sizes, saved.sizes, `${entry.model} fitted sizes match the packet`);
  entry.trainMse.forEach((folds, index) => vector(folds, saved.trainMse[index], `${entry.model} training fold row ${index}`, 1e-8));
  entry.validationMse.forEach((folds, index) => vector(folds, saved.validationMse[index], `${entry.model} validation fold row ${index}`, 1e-8));
  // The mean a figure draws is the mean of the published folds, recomputed.
  series[entry.model].validationMeans.forEach((value, index) => {
    close(value, meanOf(saved.validationMse[index]), `${entry.model} validation mean at size ${entry.sizes[index]}`, 1e-8);
  });
  series[entry.model].trainingMeans.forEach((value, index) => {
    close(value, meanOf(saved.trainMse[index]), `${entry.model} training mean at size ${entry.sizes[index]}`, 1e-8);
  });
  assert(entry.trainMse.every(folds => folds.length === 5), 'five folds per size');
  record('published learning curve against the packet');
}
// The manuscript's printed table, to four decimals.
const statedValidation = {
  60: [45.4545, 24.9831, 41.8405, 39.9286],
  120: [45.081, 23.8044, 24.4041, 32.3389],
  240: [45.2469, 23.6478, 20.1754, 28.1945],
  480: [45.2065, 23.4994, 13.7464, 21.6109],
  900: [45.222, 23.372, 8.1412, 16.6298],
};
trainSizes.forEach((size, index) => {
  ['mean', 'ridge', 'tree_leaf1', 'tree_leaf20'].forEach((model, column) => {
    close(Number(series[model].validationMeans[index].toFixed(4)), statedValidation[size][column],
      `${model} at ${size} fitted rows`, 1e-9);
  });
  record('printed validation table row');
});
assert(series.tree_leaf1.trainingExactlyZero, 'the leaf-1 tree interpolates at every inspected size');
assert(!series.ridge.trainingExactlyZero, 'Ridge does not');
assert(series.tree_leaf1.validationDecreasing, 'the tree improves at every step of the inspected range');
assert(!series.mean.validationDecreasing, 'the baseline does not');
close(Number(series.ridge.trainingMeans.at(-1).toFixed(4)), 22.7528, 'Ridge training MSE at 900', 1e-9);
close(Number(series.tree_leaf20.trainingMeans.at(-1).toFixed(4)), 12.5438, 'leaf-20 training MSE at 900', 1e-9);
assert(series.ridge.finalGap < series.tree_leaf20.finalGap, 'Ridge has the smaller gap');
assert(series.ridge.validationMeans.at(-1) > series.tree_leaf20.validationMeans.at(-1),
  'and yet it predicts worse: a small gap is not proximity to the noise floor');
assert.equal(firstCrossing(series.tree_leaf1, series.ridge), 240, 'the tree first passes Ridge at 240 fitted rows');
assert.equal(firstCrossing(series.ridge, series.tree_leaf1), 60, 'and Ridge leads before that');
assert.equal(firstCrossing(series.mean, series.ridge), null, 'the baseline never passes Ridge in this range');
const restriction = restrictionEffect(series.tree_leaf20, series.tree_leaf1);
assert.deepEqual(restriction.helps, [60], 'the leaf-20 restriction helps only at 60 fitted rows');
assert.deepEqual(restriction.hurts, [120, 240, 480, 900], 'and hurts at every larger inspected size');
assert.equal(restriction.firstHurtSize, 120, 'first stopping being useful at 120');
assert.throws(() => firstCrossing(series.ridge, { sizes: [60], validationMeans: [1] }), RangeError,
  'series on different sizes cannot be compared');
record('learning curve claims');

const leaf = validationCurveSeries(validationCurveRecord);
assert.deepEqual(leaf.settings, minSamplesLeafGrid);
assert.deepEqual(leaf.settings, recorded.real.validation.minSamplesLeaf, 'the packet grid');
leaf.validationMeans.forEach((value, index) => {
  close(value, meanOf(recorded.real.validation.validationMse[index]), `leaf ${leaf.settings[index]} validation mean`, 1e-8);
});
leaf.trainingMeans.forEach((value, index) => {
  close(value, meanOf(recorded.real.validation.trainMse[index]), `leaf ${leaf.settings[index]} training mean`, 1e-8);
});
vector(leaf.validationMeans.map(value => Number(value.toFixed(4))),
  [8.2079, 8.5475, 10.1516, 12.3799, 15.7762, 21.8071], 'the printed validation curve', 1e-9);
assert.equal(leaf.bestSetting, 1, 'the least restricted setting scores best');
assert.equal(leaf.restrictionImproves, false, 'no inspected restriction improves the score');
assert(leaf.validationMeans[0] !== series.tree_leaf1.validationMeans.at(-1),
  'the 960-row leaf-1 score is not the 900-row learning-curve value');
close(leaf.validationMeans[0], 8.2079, 'the 960-row value', 1e-4);
close(series.tree_leaf1.validationMeans.at(-1), 8.1412, 'against the 900-row value', 1e-4);
record('validation curve claims');

const trace = trajectorySeries(boostingTrajectory);
assert.equal(trace.lastRound, 120, 'the trajectory has 120 recorded rounds');
vector(trace.train, recorded.real.stage.trainMse, 'the packet training trace', 1e-8);
vector(trace.monitor, recorded.real.stage.validationMse, 'the packet monitoring trace', 1e-8);
assert.equal(trace.bestRound, 120, 'the lowest monitoring value is at the last round');
assert(trace.bestRoundIsLast, 'and the model says so');
close(trace.bestMonitor, Math.min(...recorded.real.stage.validationMse), 'the argmin is over all 120 rounds', 1e-8);
assert(trace.risingRounds.length > 0, 'the trace does rise at some rounds, so it must not be drawn as monotone');
assert(trace.largestRise > 0 && trace.largestRise < 0.1, 'and those rises are small');
printedRounds.forEach(number => assert(number >= 1 && number <= trace.lastRound, 'every printed round exists'));
vector(printedRounds.map(number => Number(trace.train[number - 1].toFixed(4))),
  [41.5026, 26.9413, 17.5512, 13.0405, 9.1042], 'the printed training rounds', 1e-9);
vector(printedRounds.map(number => Number(trace.monitor[number - 1].toFixed(4))),
  [42.9204, 28.0979, 20.0168, 15.3584, 12.3338], 'the printed monitoring rounds', 1e-9);
assert(trace.monitor.every((value, index) => index === 0 || value <= trace.monitor[0]),
  'no round is worse than the first, so no overfitting turn is hidden');
assert.throws(() => trajectorySeries({ trainMse: [1, 2], monitorMse: [1] }), RangeError, 'mismatched traces are refused');
record('boosting trajectory claims');

/* --------------------------------------------- §7 · training optimism */

const optimism = fixedDesignOptimism(6, 2, 4);
close(optimism.trainingMse, 8 / 3, 'expected training MSE', 1e-12);
close(optimism.newOutcomeMse, 16 / 3, 'expected fresh-outcome MSE at the same inputs', 1e-12);
close(optimism.gap, 8 / 3, 'their gap', 1e-12);
close(optimism.predictionVariance, 4 / 3, 'average fitted-prediction variance', 1e-12);
close(optimism.gap, 2 * optimism.predictionVariance, 'the gap is twice the variance, not equal to it', 1e-12);
const practiceSix = fixedDesignOptimism(10, 3, 2);
vector([practiceSix.trainingMse, practiceSix.newOutcomeMse, practiceSix.gap, practiceSix.predictionVariance],
  [1.4, 2.6, 1.2, 0.6], 'practice 6', 1e-12);
assert.throws(() => fixedDesignOptimism(6, 7, 4), RangeError, 'more coefficients than rows is refused');
assert.throws(() => fixedDesignOptimism(6, 2, -1), RangeError, 'a negative noise variance is refused');
assert.throws(() => fixedDesignOptimism(6.5, 2, 4), RangeError, 'a fractional row count is refused');
record('fixed-design optimism');

// The same numbers, reached instead from an explicit hat matrix.
const design = optimismDesign();
const hat = projectionSmoother(design);
close(hat.trace, 2, 'the hat matrix trace equals p', 1e-9);
close(hat.leverages.reduce((sum, value) => sum + value, 0), 2, 'the leverages sum to p', 1e-9);
assert(hat.leverages.every(value => value > 0 && value < 1), 'every leverage lies strictly inside (0, 1)');
// H is idempotent and symmetric: check it rather than assume it.
for (let i = 0; i < 6; i += 1) {
  for (let j = 0; j < 6; j += 1) {
    close(hat.matrix[i][j], hat.matrix[j][i], 'H is symmetric', 1e-9);
    const squared = hat.matrix[i].reduce((sum, value, k) => sum + value * hat.matrix[k][j], 0);
    close(squared, hat.matrix[i][j], 'H is idempotent', 1e-9);
  }
}
const correctMean = fixtures.optimismInputs.map(value => 1 + 2 * value);
const correctAccount = smootherOptimism(hat.matrix, correctMean, 4);
close(correctAccount.approximation, 0, 'a correctly specified mean has no approximation term', 1e-9);
close(correctAccount.training, optimism.trainingMse, 'the smoother route reproduces the training error', 1e-9);
close(correctAccount.fresh, optimism.newOutcomeMse, 'and the fresh-outcome error', 1e-9);
close(correctAccount.difference, correctAccount.expectedDifference, 'the difference is 2 sigma^2 tr(S)/n', 1e-9);
close(correctAccount.difference, optimism.gap, 'which is the same gap', 1e-9);
// A truth the design cannot represent adds the same amount to both sides.
const curvedMean = fixtures.optimismInputs.map(value => 1 + 2 * value + value * value);
const curvedAccount = smootherOptimism(hat.matrix, curvedMean, 4);
close(curvedAccount.approximation, 56 / 9, 'the approximation term of the unrepresentable curvature', 1e-9);
close(curvedAccount.training - correctAccount.training, curvedAccount.approximation, 'training gains it', 1e-9);
close(curvedAccount.fresh - correctAccount.fresh, curvedAccount.approximation, 'and so does the fresh error', 1e-9);
close(curvedAccount.difference, optimism.gap, 'while the gap is unchanged', 1e-9);
assert.throws(() => smootherOptimism(hat.matrix, [1, 2], 4), RangeError, 'a mismatched mean vector is refused');
record('smoother reduction');

// Leverage depends on location; averaged over the design it returns p/n.
const averageLeverage = fixtures.optimismInputs
  .map(value => newLocationRisk(design, [1, value], 4).predictionVariance)
  .reduce((sum, value) => sum + value, 0) / 6;
close(averageLeverage, optimism.predictionVariance, 'averaging leverage over the design gives p sigma^2 / n', 1e-9);
const centre = newLocationRisk(design, [1, 0], 4);
const outside = newLocationRisk(design, [1, 4], 4);
assert(outside.predictionVariance > centre.predictionVariance * 3, 'a far location carries much more variance');
close(centre.leverage, 1 / 6, 'the leverage at the design centre', 1e-9);
close(outside.leverage, 1 / 6 + 16 / 17.5, 'and at x = 4', 1e-9);
close(outside.expectedSquaredError, 4 * (1 + outside.leverage), 'the expected error adds the noise', 1e-9);
assert.throws(() => newLocationRisk(design, [1], 4), RangeError, 'a new input needs the design coordinates');
record('new-location leverage');

/* ------------------------------ §8 · classification and averaging */

const brierA = brierDecomposition(0.7, [0.6]);
const brierB = brierDecomposition(0.7, [0.4, 0.8]);
close(brierA.total, 0.22, 'procedure A Brier loss', 1e-12);
close(brierA.squaredBias, 0.01, 'its squared bias', 1e-12);
close(brierA.variance, 0, 'no prediction variance', 1e-12);
close(brierA.noise, 0.21, 'and the noise term', 1e-12);
close(brierB.total, 0.26, 'procedure B Brier loss', 1e-12);
close(brierB.squaredBias, 0.01, 'the same squared bias', 1e-12);
close(brierB.variance, 0.04, 'plus prediction variance', 1e-12);
assert(brierA.agrees && brierB.agrees, 'both decompositions match the direct expectation');
close(thresholdedError(0.7, [0.6]).error, 0.3, 'A always predicts class 1', 1e-12);
close(thresholdedError(0.7, [0.4, 0.8]).error, 0.5, 'B crosses the threshold half the time', 1e-12);
assert(brierB.total > brierA.total && thresholdedError(0.7, [0.4, 0.8]).error > thresholdedError(0.7, [0.6]).error,
  'both losses prefer A here, for different reasons');
// The zero-one error is a straight line, and variation can help or hurt.
for (const [rate, expected] of [[0, 0.8], [0.25, 0.65], [0.75, 0.35], [1, 0.2]]) {
  close(thresholdedErrorAt(0.8, rate), expected, `zero-one error at q = ${rate}`, 1e-12);
  record('thresholded error point');
}
assert(thresholdedErrorAt(0.8, 0.25) < thresholdedErrorAt(0.8, 0), 'variation helps when it replaces the wrong decision');
assert(thresholdedErrorAt(0.8, 0.75) > thresholdedErrorAt(0.8, 1), 'and hurts when it replaces the right one');
assert.throws(() => brierDecomposition(1.2, [0.5]), RangeError, 'a probability above one is refused');
assert.throws(() => brierDecomposition(0.5, [1.5]), RangeError, 'so is a predicted probability above one');
record('loss-specific decomposition');

const averaging = averageVariance(4, 0.5, 4);
close(averaging.variance, 2.5, 'the correlated average variance', 1e-12);
close(averaging.independentVariance, 1, 'independent predictors would give 1', 1e-12);
close(averaging.perfectlyCorrelated, 4, 'perfectly correlated ones gain nothing', 1e-12);
close(averageVariance(4, 1, 4).variance, 4, 'rho = 1 reproduces the single variance', 1e-12);
close(averageVariance(4, 0, 4).variance, 1, 'rho = 0 gives v/B', 1e-12);
close(averageVariance(4, 0.5, 1).variance, 4, 'one predictor is its own average', 1e-12);
close((averaging.individualTerms + averaging.pairTerms) / 16, averaging.variance,
  'the expansion of the variance of the sum, divided by B squared', 1e-12);
for (const B of [2, 3, 5, 10, 50]) {
  close(averageVariance(4, 0.5, B).variance, 4 * (0.5 + 0.5 / B), `the average variance at B = ${B}`, 1e-12);
  assert(averageVariance(4, 0.5, B).variance > 4 * 0.5 - 1e-12, 'it never falls below the shared part');
  record('averaging at B');
}
assert.throws(() => averageVariance(4, 0.5, 0), RangeError, 'zero predictors are refused');
assert.throws(() => averageVariance(4, -0.9, 3), RangeError, 'an impossible correlation is refused');
assert.throws(() => averageVariance(-1, 0.5, 3), RangeError, 'a negative variance is refused');
record('averaging');

/* --------------------------------------------- §9 · double descent */

assert.equal(doubleDescentNoise, 0.04);
assert.equal(doubleDescentSignalNorm, 1);
for (const saved of recorded.doubleDescentApproximation) {
  const answer = doubleDescentRisk(saved.ratio);
  assert(answer.defined, `gamma ${saved.ratio} is on a branch`);
  close(answer.biasSquared, saved.biasSquared, `squared bias at ${saved.ratio}`, 1e-12);
  close(answer.varianceApprox, saved.varianceApprox, `variance term at ${saved.ratio}`, 1e-12);
  close(answer.riskApprox, saved.riskApprox, `risk at ${saved.ratio}`, 1e-12);
  assert(Number.isFinite(answer.riskApprox), 'and it is a finite number');
  record('double descent against the packet');
}
const statedTable = {
  0.1: 0.944444, 0.5: 0.58, 0.8: 0.4, 0.9: 0.5, 0.99: 4.01,
  1.01: 4.04, 1.1: 0.44, 1.5: 0.12, 2: 0.08,
};
for (const [ratio, expected] of Object.entries(statedTable)) {
  close(doubleDescentRisk(Number(ratio)).riskApprox, expected, `the printed value at gamma ${ratio}`, 1e-6);
  record('printed double-descent row');
}
assert.deepEqual(fixtures.doubleDescentRatios, [0.1, 0.5, 0.8, 0.9, 0.99, 1.01, 1.1, 1.5, 2],
  'the fixture lists exactly the manuscript ratios, in the printed order');
assert.deepEqual([...fixtures.doubleDescentRatios].sort((left, right) => left - right), fixtures.doubleDescentRatios,
  'and they are already increasing, so the printed table reads left to right');
// Practice 8's two ratios, in the manuscript's own three parts.
for (const [ratio, parts] of [[0.75, [0.25, 0.12, 0.04]], [0.9, [0.1, 0.36, 0.04]]]) {
  close(1 - ratio, parts[0], `the first part at gamma ${ratio}`, 1e-12);
  close(doubleDescentNoise * ratio / (1 - ratio), parts[1], `the amplified-noise part at gamma ${ratio}`, 1e-12);
  close(doubleDescentRisk(ratio).riskApprox, parts[0] + parts[1] + parts[2], `their sum at gamma ${ratio}`, 1e-12);
  record('practice 8 ratio');
}
// The boundary is undefined, not a very large number, and not plotted.
const boundary = doubleDescentRisk(1);
assert.equal(boundary.defined, false, 'gamma = 1 is returned as undefined');
assert.equal(boundary.riskApprox, null, 'with no numeric risk to plot');
assert.equal(boundary.branch, 'boundary');
assert(doubleDescentRisk(0.999).riskApprox > 30, 'the approximation diverges from below');
assert(doubleDescentRisk(1.001).riskApprox > 30, 'and from above');
close(doubleDescentRisk(0.9999).riskApprox / doubleDescentRisk(0.99999).riskApprox, 0.1, 'as a reciprocal', 2e-3);
// Beyond the boundary the risk falls back towards the noise floor.
assert(doubleDescentRisk(3).riskApprox < doubleDescentRisk(1.5).riskApprox, 'and keeps falling with more rows');
assert(doubleDescentRisk(100).riskApprox > doubleDescentNoise, 'never below the noise floor');
close(doubleDescentRisk(1e6).riskApprox, doubleDescentNoise, 'approaching it in the limit', 1e-6);
// The plotted branches never touch the boundary and never carry a non-number.
for (const branch of [doubleDescentBranch(0.1, 0.97, 140), doubleDescentBranch(1.03, 3, 140)]) {
  assert(branch.length === 140, 'every sampled point is defined');
  assert(branch.every(point => Number.isFinite(point.riskApprox) && point.riskApprox > 0), 'and finite and positive');
  assert(branch.every(point => Math.abs(point.ratio - 1) > 1e-6), 'and clear of the boundary');
  record('double descent branch');
}
assert.throws(() => doubleDescentBranch(0.5, 1.5), RangeError, 'a branch may not straddle the boundary');
assert.throws(() => doubleDescentBranch(-1, 2), RangeError, 'a nonpositive ratio is refused');
assert.throws(() => doubleDescentRisk(0), RangeError, 'a zero ratio is refused');
// The nonmonotonic shape the figure claims: down, then up, then down again.
const lowPoint = [0.1, 0.3, 0.5, 0.7, 0.8].map(ratio => doubleDescentRisk(ratio).riskApprox);
assert(lowPoint[4] === Math.min(...lowPoint), 'the lower branch bottoms out near gamma 0.8');
assert(doubleDescentRisk(0.9).riskApprox > doubleDescentRisk(0.8).riskApprox, 'then rises before the boundary');
assert(doubleDescentRisk(2).riskApprox < doubleDescentRisk(1.1).riskApprox, 'and falls again beyond it');
record('double descent shape');

/* ------------------------------------------ the generated data module */

assert.equal(provenance.rows, 1503);
assert.equal(provenance.developmentRows + provenance.reservedRows, 1503, 'the split covers every row');
assert.equal(provenance.reservedPredictionsComputed, false, 'no reserved row is scored');
assert.equal(provenance.sha256, '74c75fd71783f1e6b71f8a622b993dc592897a97cd689c5090a07147a1b097b3');
assert.equal(provenance.bytes, 59984);
assert.equal(provenance.license, 'CC BY 4.0');
assert.equal(provenance.file, '/learn-assets/bias-variance/airfoil-self-noise.dat');
assert.ok(!provenance.file.includes('regularization'), 'this lesson serves its own bytes');
assert.equal(provenance.folds * provenance.foldValidationRows, provenance.developmentRows, 'the folds partition development');
assert.equal(provenance.foldTrainRows + provenance.foldValidationRows, provenance.developmentRows / 1,
  'each fold leaves the rest for fitting');
assert.equal(provenance.stageFitRows + provenance.stageMonitorRows, provenance.developmentRows,
  'the trajectory split also covers development exactly');
assert.equal(recorded.real.developmentIds.length, provenance.developmentRows);
assert.equal(recorded.real.untouchedIds.length, provenance.reservedRows);
assert.equal(new Set([...recorded.real.developmentIds, ...recorded.real.untouchedIds]).size, 1503,
  'and the two are disjoint');
assert.deepEqual(trainSizes, [60, 120, 240, 480, 900]);
assert(trainSizes.every(size => size <= provenance.foldTrainRows), 'every requested size fits inside a fold');
assert.equal(procedures.length, 4);
assert.equal(rawFeatureNames.length, 5);
assert.equal(rawFeatureLabels.length, 5);
assert.equal(rawFeatureRanges.length, 5);
assert(rawFeatureRanges.every(range => range[0] < range[1]), 'each observed range is nondegenerate');
assert(targetRange[0] < targetRange[1]);
assert.equal(validationCurveRecord.fitRowsPerFold, provenance.foldTrainRows);
assert.equal(boostingTrajectory.rounds, 120);
assert.equal(boostingTrajectory.trainMse.length, 120);
assert.equal(boostingTrajectory.monitorMse.length, 120);
assert.equal(boostingTrajectory.fitRows, provenance.stageFitRows);
record('published data module');

/* ------------------------------------------- the displayed programs */

assert.equal(Object.keys(examples).length, 3, 'three displayed programs');
assert.equal(examples.finiteWorlds.file, 'bias_variance_worlds.py');
assert.equal(examples.airfoilLearningCurve.file, 'airfoil_learning_curve.py');
assert.equal(examples.settingAndTrajectory.appendedTo, 'airfoil_learning_curve.py');
assert.equal(examples.settingAndTrajectory.file, undefined, 'the appended block is not a separate file');
assert.ok(examples.airfoilLearningCurve.code.includes('airfoil-self-noise.dat'), 'the program reads the served file');
assert.ok(examples.airfoilLearningCurve.code.includes('development, untouched'), 'the reserved rows are split off');
assert.ok(!examples.airfoilLearningCurve.code.includes('untouched]'), 'and never indexed for a score');
assert.ok(examples.finiteWorlds.code.includes('product([-1., 1.], repeat=len(train_x))'), 'every sign combination');
// The printed learning-curve block agrees with the published data module.
const printedBlocks = {};
let currentBlock = null;
for (const line of examples.airfoilLearningCurve.expected.split(/\r?\n/)) {
  if (procedures.some(entry => entry.model === line)) { currentBlock = line; printedBlocks[currentBlock] = []; } else {
    printedBlocks[currentBlock].push(line.trim().split(/\s+/).map(Number));
  }
}
for (const entry of procedures) {
  const printed = printedBlocks[entry.model];
  assert(printed, `the program prints a block for ${entry.model}`);
  printed.forEach((row, index) => {
    assert.equal(row[0], trainSizes[index], `${entry.model} prints size ${trainSizes[index]}`);
    close(row[2], Number(series[entry.model].validationMeans[index].toFixed(4)),
      `${entry.model} printed validation at ${trainSizes[index]}`, 1e-9);
    close(Math.abs(row[1]), Number(series[entry.model].trainingMeans[index].toFixed(4)),
      `${entry.model} printed training at ${trainSizes[index]}`, 1e-9);
  });
  record('printed learning-curve block');
}
const printedLeaf = examples.settingAndTrajectory.expected.split(/\r?\n/).filter(line => line.startsWith('leaf '));
printedLeaf.forEach((line, index) => {
  const parts = line.split(/\s+/);
  assert.equal(Number(parts[1]), leaf.settings[index], 'the printed leaf size');
  close(Number(parts[3]), Number(leaf.validationMeans[index].toFixed(4)), 'the printed leaf validation mean', 1e-9);
  record('printed validation-curve row');
});
assert.ok(examples.settingAndTrajectory.expected.trim().endsWith(`best inspected round ${trace.bestRound}`),
  'the program prints the same best round the model reports');
const printedFinite = examples.finiteWorlds.expected.split(/\r?\n/);
assert.equal(printedFinite.length, 3, 'one printed line per degree');
printedFinite.forEach(line => {
  const degree = Number(line.split(' ')[0]);
  const values = line.replace(/[[\]]/g, ' ').split(/\s+/).filter(Boolean).slice(1).map(Number);
  assert.equal(values.length, 8, 'four pairs of printed values');
  const atHalf = finiteExperiment({ ...base, degree, probe: 0.5, grid: [0.5] });
  close(values[1], Number(atHalf.meanPrediction.toFixed(6)), `printed mean for degree ${degree}`, 1e-6);
  close(values[3], Number(atHalf.squaredBias.toFixed(6)), `printed squared bias for degree ${degree}`, 1e-6);
  close(values[5], Number(atHalf.variance.toFixed(6)), `printed variance for degree ${degree}`, 1e-6);
  close(values[7], Number(atHalf.expectedError.toFixed(6)), `printed expected error for degree ${degree}`, 1e-6);
  record('printed finite-worlds row');
});
record('displayed programs');

/* ------------------------------------------------------ control ranges */

assert(limits.maximumWorlds === 2 ** limits.maximumTrainingInputs, 'the world bound follows the input bound');
assert(limits.prediction.minimum === 0 && limits.prediction.maximum === 20, 'the specified prediction range');
assert(limits.noiseSpread.maximum === 4, 'the specified spread range');
assert(limits.curvature.minimum === -1 && limits.curvature.maximum === 2, 'the specified curvature range');
assert(limits.sigma.maximum === 1.5 && limits.probe.maximum === 1.5, 'the specified noise and probe ranges');
assert(fixtures.threeInputs.every(value => value >= limits.probe.minimum && value <= limits.probe.maximum),
  'the three-input design lies inside the probe range');
assert(fixtures.fiveInputs.every(value => value >= limits.probe.minimum && value <= limits.probe.maximum),
  'and so does the five-input design');
record('control ranges');

/* --------------------------------------------------------------- record */

const sources = [
  'src/learn/data/bias-variance-models.js',
  'src/learn/data/bias-variance-data.js',
  'src/learn/data/bias-variance-examples.js',
  'src/learn/components/lesson-labs/BiasVarianceShared.jsx',
  'src/learn/components/lesson-labs/BiasVarianceLabs.jsx',
  'src/learn/components/lesson-labs/BiasVarianceFigures.jsx',
  'src/learn/components/lesson-labs/bias-variance-labs.css',
  'src/learn/data/topics/bias-variance-tradeoff-learning-curves.jsx',
  'src/learn/data/curriculum/blueprints/bias-variance-tradeoff-learning-curves.js',
  'public/learn-assets/bias-variance/airfoil-self-noise.dat',
];
const hash = file => crypto.createHash('sha256').update(fs.readFileSync(file)).digest('hex');
const evidence = {
  checkedAt: new Date().toISOString(),
  sourceHashes: Object.fromEntries(sources.filter(fs.existsSync).map(file => [file, hash(file)])),
  verifierHash: hash('scripts/verify-bias-variance-models.mjs'),
  counts,
  totalGroupedChecks: Object.values(counts).reduce((sum, value) => sum + value, 0),
  comparedGridValues,
  scope: 'Browser bias-variance models against the content phase\'s calculated-inputs.json (all 21 finite experiments '
    + 'compared world by world and point by point over their 61-point grids, every recorded learning-curve, '
    + 'validation-curve and boosting fold value, and the double-descent table), independent identities (the '
    + 'decomposition against a direct enumeration of every prediction/outcome pair, prediction variance from squared '
    + 'interpolation weights against the enumerated worlds, a 375-setting sweep of the identity, the hat matrix '
    + 'checked symmetric and idempotent and its optimism reproduced through the general smoother trace, leverage '
    + 'averaged back to p sigma squared over n, and the reciprocal divergence at the interpolation boundary), the '
    + 'manuscript\'s stated values in sections 1, 3, 5, 7, 8 and 9 and in every practice task, refused input on every '
    + 'entry point, the typeability of every value the prose asks a learner to enter, and agreement between the '
    + 'published data module and the executed displayed programs.',
  limitations: [
    'The airfoil curves are precomputed native fits; the browser reproduces their recorded outcomes, not the fitting.',
    'Displayed program output is executed separately by scripts/verify-bias-variance-examples.py.',
    'The data module is regenerated and matched to the packet separately by scripts/verify-bias-variance-data.py.',
    'Rendering, interaction, visual layout and independent review are separate steps.',
  ],
  passed: true,
};
fs.mkdirSync('docs/teaching/evidence', { recursive: true });
fs.writeFileSync('docs/teaching/evidence/bias-variance-models.json', JSON.stringify(evidence, null, 2) + '\n');
console.log(`PASS: ${evidence.totalGroupedChecks} grouped bias-variance model checks across ${Object.keys(counts).length} groups, `
  + `including ${comparedGridValues.toLocaleString('en-US')} recorded grid values.`);
