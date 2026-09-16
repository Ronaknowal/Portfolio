// Bounded independent checks of the Gaussian mixture browser models against the
// content phase's native probes, the manuscript's worked values and analytic
// identities, plus structural checks of the generated Iris module.
// Run: node scripts/verify-gmm-models.mjs
import assert from 'node:assert/strict';
import fs from 'node:fs';
import crypto from 'node:crypto';
import {
  boundDecomposition, collapseLogLikelihood, conditionalMixture, correlationGeometry, criteria, emCycle, emTrace,
  expectation, logNormal, maximization, mixtureAt, mixtureMoments, normal, parameterCount, planeComparison,
  planeDensity, presets, quadraticForm, sphericalResponsibility, symmetricEigenpairs, responsibilityCrossings,
} from '../src/learn/data/gmm-models.js';
import {
  bayesianSensitivity, candidates, narrowComponent, observations, scaler, selection, selectedParameters, speciesNames,
  splitCounts, testRows,
} from '../src/learn/data/gmm-iris-data.js';

const packet = 'docs/teaching/drafts/gaussian-mixture-models-gmm-em-algorithm';
const checked = JSON.parse(fs.readFileSync(`${packet}/checked-results.json`, 'utf8'));
const examples = JSON.parse(fs.readFileSync('src/learn/data/gmm-examples.js', 'utf8')
  .replace(/^[\s\S]*?export const gmmExamples = /, '').replace(/;\s*$/, ''));

const counts = {};
const record = name => { counts[name] = (counts[name] ?? 0) + 1; };
const close = (actual, expected, label, tolerance = 1e-9) =>
  assert(Math.abs(actual - expected) <= tolerance * Math.max(1, Math.abs(expected)), `${label}: ${actual} versus ${expected}`);
const componentsOf = probe => probe.weights.map((weight, index) => ({
  weight, mean: probe.means[index], variance: probe.variances[index],
}));

// ------------------------------------------------------------ the density itself
// A Gaussian integrates to one: a coarse independent quadrature is enough to
// catch a missing normalizer or a factor of two in the exponent.
for (const [mean, variance] of [[0, 1], [-2, 1], [1.5, 0.25], [0, 4]]) {
  let area = 0;
  const step = 0.001;
  for (let x = mean - 12 * Math.sqrt(variance); x < mean + 12 * Math.sqrt(variance); x += step) {
    area += normal(x + step / 2, mean, variance) * step;
  }
  close(area, 1, `N(${mean}, ${variance}) integrates to one`, 1e-6);
}
close(normal(0, 0, 1), 1 / Math.sqrt(2 * Math.PI), 'the standard peak');
close(logNormal(2, 2, 1), Math.log(1 / Math.sqrt(2 * Math.PI)), 'log density at the centre');
// A narrow bell exceeds one: density is not probability.
assert(normal(0, 0, 0.01) > 1, 'a narrow density exceeds one');
assert.throws(() => normal(0, 0, 0), RangeError);
assert.throws(() => mixtureAt([{ weight: 0.6, mean: 0, variance: 1 }], 0), RangeError, 'weights must sum to one');
record('density identities');

// ------------------------------------------------------ responsibility fixtures
for (const probe of checked.responsibility) {
  const state = mixtureAt(componentsOf(probe), probe.x);
  close(state.density, probe.density, `density at ${probe.x}`, 1e-12);
  probe.responsibility.forEach((value, index) => close(state.responsibilities[index], value, `responsibility ${index} at ${probe.x}`, 1e-12));
  assert(Math.abs(state.responsibilities.reduce((sum, value) => sum + value, 0) - 1) < 1e-12, 'a row sums to one');
  record('responsibility versus author');
}
const base = mixtureAt(presets.twoBell, 2);
close(base.components[0].weighted, 0.000066915, 'A contributes', 1e-6);
close(base.components[1].weighted, 0.199471140, 'B contributes', 1e-9);
close(base.density, 0.199538055, 'their sum', 1e-9);
close(base.responsibilities[1], 0.999664650, 'B takes', 1e-9);
close(mixtureAt(presets.twoBell, 8).negativeLogDensity, 19.612086, 'the negative log-density at 8', 1e-6);
close(mixtureAt(presets.twoBell, 0).negativeLogDensity, 2.918939, 'the negative log-density at 0', 1e-6);
// The lesson's central contrast: nearly all of an extremely small total.
assert(mixtureAt(presets.twoBell, 8).responsibilities[1] > 0.999999 && mixtureAt(presets.twoBell, 8).density < 1e-8,
  'a decisive share on almost no density');
// Identical components hand the mixing weights straight back, at every location.
for (const x of [-3, 0, 2.5, 7]) {
  const identical = mixtureAt(presets.identical, x);
  close(identical.responsibilities[0], 0.2, `identical components at ${x}`, 1e-12);
  close(identical.density, normal(x, 0, 1), `identical density at ${x}`, 1e-12);
}
record('responsibility identities');

// Log-space evaluation survives what direct evaluation cannot.
const distant = mixtureAt([
  { weight: 0.5, mean: 0, variance: 1 },
  { weight: 0.5, mean: Math.sqrt(2), variance: 1 },
], 0);
assert(Number.isFinite(distant.logDensity), 'a finite log-density');
const underflow = mixtureAt([
  { weight: 1 / (1 + Math.exp(-1)), mean: 0, variance: 1 },
  { weight: Math.exp(-1) / (1 + Math.exp(-1)), mean: 0, variance: 1 },
], 0);
close(underflow.responsibilities[0], checked.underflow.responsibility[0], 'the log-sum-exp fixture', 1e-12);
close(underflow.responsibilities[1], checked.underflow.responsibility[1], 'its complement', 1e-12);
const shifted = mixtureAt([
  { weight: 0.5, mean: 0, variance: 1 },
  { weight: 0.5, mean: 1000, variance: 1 },
], 0);
assert(Number.isFinite(shifted.logDensity) && shifted.responsibilities[0] === 1, 'a very distant component contributes nothing');
record('log-space evaluation');

// ------------------------------------------------------------------ the EM trace
const trace = emTrace(presets.observations, presets.start, 0.05, 50, 1e-8);
checked.em.slice(0, trace.length).forEach((probe, index) => {
  close(trace[index].logLikelihood, probe.log_likelihood, `log-likelihood at iteration ${index}`, 1e-12);
  trace[index].components.forEach((component, column) => {
    close(component.mean, probe.means[column], `mean ${column} at iteration ${index}`, 1e-12);
    close(component.variance, probe.variances[column], `variance ${column} at iteration ${index}`, 1e-12);
    close(component.weight, probe.weights[column], `weight ${column} at iteration ${index}`, 1e-12);
  });
  record('EM trace versus author');
});
// The manuscript's hand calculation, exactly.
const first = emCycle(presets.observations, presets.start, 0.05);
close(first.responsibilities[0][0], 0.982013790, 'A goes to Left', 1e-9);
close(first.responsibilities[1][0], 0.880797078, 'B goes to Left', 1e-9);
close(first.responsibilities[0][0], Math.exp(4) / (1 + Math.exp(4)), 'that share is exactly e^4/(1+e^4)', 1e-15);
close(first.after[0].count, 2, 'the effective count is exactly two', 1e-12);
close(presets.observations.reduce((sum, x, index) => sum + x * first.responsibilities[index][0], 0), -2.689649316, 'the weighted sum', 1e-9);
close(first.after[0].mean, -1.344824658, 'the new Left mean', 1e-9);
close(first.after[0].variance, 0.691446639, 'the new Left variance', 1e-9);
close(2.5 - first.after[0].mean ** 2, first.after[0].variance, 'variance as second moment minus squared mean', 1e-12);
close(first.logLikelihoodBefore, -7.158186977, 'before', 1e-9);
close(first.logLikelihoodAfter, -6.461856301, 'after', 1e-9);
close(expectation(presets.observations, first.after).responsibilities[0][0], 0.999582068, 'the next E-step sharpens', 1e-9);
// Every cycle of every supported start is a nondecrease.
for (const start of [presets.start, presets.asymmetricStart, presets.identicalStart]) {
  let components = start;
  for (let iteration = 0; iteration < 12; iteration += 1) {
    const cycle = emCycle(presets.observations, components, 0.05);
    assert(cycle.gain >= -1e-10, `a cycle never decreases the objective: ${cycle.gain}`);
    assert(cycle.responsibilities.every(row => Math.abs(row[0] + row[1] - 1) < 1e-14), 'rows sum to one');
    components = cycle.after;
  }
  record('nondecrease along a start');
}
// The changed-point checkpoint.
const edited = emCycle([-2, -1, 1, 3], presets.start, 0.05);
close(edited.after[1].mean, 1.844792261, 'the changed Right mean', 1e-9);
close(edited.after[1].weight, 0.503878397, 'its weight', 1e-9);
close(edited.after[1].variance, 1.582910448, 'its variance', 1e-9);
close(edited.responsibilities[3][1], 1 / (1 + Math.exp(-6)), 'the Right share at 3', 1e-12);
// The symmetric stationary null.
const still = emCycle(presets.observations, presets.identicalStart, 0.05);
close(still.logLikelihoodBefore, -7.508335597, 'the identical-start objective', 1e-9);
close(still.gain, 0, 'and it does not move', 1e-12);
assert(still.responsibilities.every(row => row.every(value => Math.abs(value - 0.5) < 1e-15)), 'every allocation is one half');
assert(trace.at(-1).logLikelihood > still.logLikelihoodBefore, 'the separated fit scores higher');
// The floor binds exactly where the scatter falls below it.
const floored = emCycle(presets.repeated, presets.repeatedStart, 0.25);
assert(floored.after.every(component => component.atFloor), 'repeated measurements drive both variances to the floor');
floored.after.forEach(component => close(component.variance, 0.25, 'held at the floor', 1e-15));
const unfloored = emCycle(presets.repeated, presets.repeatedStart, 0.001);
assert(unfloored.after.every(component => !component.atFloor && component.variance < 0.25), 'a lower floor does not bind');
unfloored.after.forEach(component => close(component.variance, component.scatter, 'and the scatter is returned unchanged', 1e-15));
// The recorded floor-active variation, step by step.
checked.em_variations.floor_active.forEach((probe, index) => {
  const state = index === 0
    ? { components: presets.repeatedStart, logLikelihood: expectation(presets.repeated, presets.repeatedStart).logLikelihood }
    : emTrace(presets.repeated, presets.repeatedStart, 0.25, index, 0)[index];
  close(state.logLikelihood, probe.log_likelihood, `floor-active log-likelihood at ${index}`, 1e-12);
  state.components.forEach((component, column) => {
    close(component.mean, probe.means[column], `floor-active mean ${column} at ${index}`, 1e-12);
    close(component.variance, probe.variances[column], `floor-active variance ${column} at ${index}`, 1e-12);
  });
  record('floor-active versus author');
});
// The asymmetric start reaches the same separated solution as the standard one.
const asymmetric = emTrace(presets.observations, presets.asymmetricStart, 0.05, 60, 1e-10);
checked.em_variations.asymmetric_initialization.slice(0, 3).forEach((probe, index) => {
  close(asymmetric[index].logLikelihood, probe.log_likelihood, `asymmetric log-likelihood at ${index}`, 1e-12);
  record('asymmetric start versus author');
});
close(asymmetric.at(-1).logLikelihood, trace.at(-1).logLikelihood, 'both non-identical starts reach the same objective', 1e-6);
// The constrained maximizer really is the maximizer, checked numerically.
for (const scatter of [0.004, 0.02, 0.4]) {
  const objective = v => -(Math.log(v) + scatter / v) / 2;
  const best = [...Array(4000).keys()].map(index => 0.05 + index * 0.001).reduce((leader, v) => (objective(v) > objective(leader) ? v : leader), 0.05);
  close(best, Math.max(scatter, 0.05), `the constrained optimum for scatter ${scatter}`, 2e-2);
}
assert.throws(() => maximization([0, 1], [[0, 1], [0, 1]], 0.05), RangeError, 'an empty component is refused, not reseeded');
record('EM identities');

// ------------------------------------------------------------------- the bound
const start = expectation(presets.observations, presets.start);
const oldBound = boundDecomposition(presets.observations, presets.start, start.responsibilities);
const newBound = boundDecomposition(presets.observations, first.after, start.responsibilities);
close(oldBound.qFunction, checked.bound.old.q_function, 'Q at the old parameters', 1e-12);
close(oldBound.entropy, checked.bound.old.entropy, 'the entropy', 1e-12);
close(oldBound.elbo, checked.bound.old.elbo, 'the old bound', 1e-12);
close(newBound.qFunction, checked.bound.new_with_old_q.q_function, 'Q at the new parameters', 1e-12);
close(newBound.entropy, oldBound.entropy, 'entropy is unchanged by the M-step', 1e-15);
close(newBound.elbo, checked.bound.new_with_old_q.elbo, 'the lifted bound', 1e-12);
close(checked.bound.new_objective, first.logLikelihoodAfter, 'the new objective', 1e-12);
// The touching equality, and the ordering it makes possible.
close(oldBound.elbo, start.logLikelihood, 'the E-step makes the bound touch', 1e-12);
assert(first.logLikelihoodAfter > newBound.elbo, 'the new objective sits above the lifted bound');
assert(newBound.elbo > oldBound.elbo, 'the M-step lifted the same bound');
close(first.logLikelihoodAfter - newBound.elbo, 0.337690713, 'the gap before the next E-step', 1e-9);
// The bound really is a bound, for allocations that are not the posterior.
for (const q of [[[0.5, 0.5], [0.5, 0.5], [0.5, 0.5], [0.5, 0.5]], [[0.9, 0.1], [0.6, 0.4], [0.3, 0.7], [0.2, 0.8]]]) {
  const bound = boundDecomposition(presets.observations, presets.start, q);
  assert(bound.elbo <= start.logLikelihood + 1e-12, 'an arbitrary allocation cannot exceed the objective');
  record('bound below the objective');
}
record('bound decomposition');

// ------------------------------------------------------------- collapse fixture
checked.collapse.forEach(probe => {
  close(collapseLogLikelihood(probe.sigma), probe.log_likelihood, `collapse at sigma ${probe.sigma}`, 1e-9);
  record('collapse versus author');
});
assert(collapseLogLikelihood(0.0001) > collapseLogLikelihood(0.001), 'a narrower spike scores higher still');
assert(collapseLogLikelihood(1e-6) > 0, 'the unconstrained objective passes zero and keeps going');

// ------------------------------------------------------------- plane geometry
checked.geometry.forEach(probe => {
  const value = planeDensity(probe.x, probe.rho);
  close(value.squared, probe.mahalanobis_squared, `D squared at ${probe.x} with rho ${probe.rho}`, 1e-12);
  close(value.density, probe.density, `density at ${probe.x} with rho ${probe.rho}`, 1e-12);
  record('geometry versus author');
});
close(quadraticForm([1, 1], 0.75), 8 / 7, 'the familiar direction', 1e-15);
close(quadraticForm([1, -1], 0.75), 8, 'the unfamiliar one', 1e-15);
const geometry = correlationGeometry(0.75);
close(geometry.determinant, 0.4375, 'the determinant');
close(geometry.axes[0].semiaxis, Math.sqrt(1.75), 'the long semiaxis');
close(geometry.axes[1].semiaxis, 0.5, 'the short semiaxis');
close(geometry.massInsideUnitContour, checked.ellipse_mass, 'the mass inside the unit contour', 1e-12);
assert(Math.abs(geometry.massInsideUnitContour - 0.68) > 0.28, 'it is not the 68% of a one-dimensional interval');
// Eigenpairs: the covariance really does map each direction to its eigenvalue.
geometry.axes.forEach(axis => {
  const [first_, second_] = axis.direction;
  const mapped = [first_ + 0.75 * second_, 0.75 * first_ + second_];
  close(mapped[0], axis.value * first_, 'eigenvector maps to a multiple', 1e-12);
  close(mapped[1], axis.value * second_, 'in both coordinates', 1e-12);
});
// Reversing the correlation reverses the order and keeps the determinant.
const positive = planeComparison([1, 1], [1, -1], 0.75);
const negative = planeComparison([1, 1], [1, -1], -0.75);
assert.equal(positive.order, 'first');
assert.equal(negative.order, 'second');
close(positive.geometry.determinant, negative.geometry.determinant, 'both determinants', 1e-15);
assert.equal(planeComparison([1, 1], [1, -1], 0).order, 'equal', 'no correlation, no preference');
close(planeDensity([1, 1], 0).density, 0.058549832, 'the null density', 1e-9);
// Practice 4.
checked.practice.changed_geometry.forEach(probe => {
  close(quadraticForm([2, 1], probe.rho), probe.mahalanobis_squared[0], `practice 4 at rho ${probe.rho}`, 1e-12);
  close(quadraticForm([2, -1], probe.rho), probe.mahalanobis_squared[1], `practice 4 mirror at rho ${probe.rho}`, 1e-12);
  record('practice 4 versus author');
});
// Eigenpairs of the covariance families the gallery draws: an axis-aligned
// matrix must return two different axes, not one repeated direction.
for (const matrix of [[[2, 0.75], [0.75, 1]], [[1, 0.5], [0.5, 1]], [[2, 0], [0, 1]], [[0.5, 0], [0, 1.5]],
  [[1.5, 0], [0, 1.5]], [[1, 0], [0, 1]], [[0.5, -0.25], [-0.25, 1.5]]]) {
  const { eigenvalues, directions } = symmetricEigenpairs(matrix);
  assert(eigenvalues[0] >= eigenvalues[1], 'ordered largest first');
  assert(eigenvalues[1] > 0, 'both eigenvalues are positive for these covariances');
  directions.forEach(direction => close(Math.hypot(direction[0], direction[1]), 1, 'each direction is a unit vector', 1e-12));
  close(directions[0][0] * directions[1][0] + directions[0][1] * directions[1][1], 0, 'and the two are orthogonal', 1e-12);
  // Reconstruct the matrix from its own eigendecomposition.
  [[0, 0], [0, 1], [1, 0], [1, 1]].forEach(([row, column]) => {
    const rebuilt = eigenvalues.reduce((sum, value, index) => sum + value * directions[index][row] * directions[index][column], 0);
    close(rebuilt, matrix[row][column], `entry ${row}${column} rebuilds`, 1e-12);
  });
  record('gallery eigenpairs');
}
assert.throws(() => symmetricEigenpairs([[1, 0.5], [0.25, 1]]), RangeError, 'an asymmetric matrix is refused');
assert.throws(() => correlationGeometry(1), RangeError);
record('plane geometry');

// ---------------------------------------------------- counts, criteria, limits
assert.deepEqual([parameterCount('full', 3, 2).total, parameterCount('tied', 3, 2).total,
  parameterCount('diag', 3, 2).total, parameterCount('spherical', 3, 2).total], [17, 11, 14, 11]);
assert.equal(parameterCount('full', 2, 2).covariance, 6);
assert.throws(() => parameterCount('banded', 2, 2), RangeError);
const smaller = criteria(-150, 5, 100);
const larger = criteria(-143, 11, 100);
close(smaller.aic, checked.practice.criterion.aic[0], 'AIC of the smaller model', 1e-9);
close(larger.aic, checked.practice.criterion.aic[1], 'AIC of the larger model', 1e-9);
close(smaller.bic, checked.practice.criterion.bic[0], 'BIC of the smaller model', 1e-9);
close(larger.bic, checked.practice.criterion.bic[1], 'BIC of the larger model', 1e-9);
assert(larger.aic < smaller.aic && larger.bic > smaller.bic, 'the two criteria disagree here');
close(larger.bic - smaller.bic, 6 * Math.log(100) - 14, 'the penalty difference', 1e-9);
// The k-means limit.
checked.small_variance.forEach(probe => {
  close(sphericalResponsibility(1, [0, 3], probe.variance)[0], probe.responsibility_at_one, `responsibility at variance ${probe.variance}`, 1e-9);
  close(sphericalResponsibility(1.5, [0, 3], probe.variance)[0], probe.responsibility_at_midpoint, 'the midpoint never moves', 1e-12);
  record('k-means limit versus author');
});
checked.practice.changed_small_variance.forEach(probe => {
  close(sphericalResponsibility(probe.x, [0, 3], probe.variance)[0], probe.responsibility, `practice 7 at variance ${probe.variance}`, 1e-9);
});
close(sphericalResponsibility(0.5, [0, 3], 1)[0], 1 / (1 + Math.exp(-3)), 'practice 7 in closed form', 1e-15);
assert(sphericalResponsibility(1, [0, 3], 0.001)[0] > 0.999999, 'a vanishing variance concentrates');
// Moments and conditioning.
const moments = mixtureMoments(presets.twoBell);
close(moments.mean, 0, 'the two-bell mean', 1e-15);
close(moments.variance, 5, 'within plus between', 1e-15);
const atOne = conditionalMixture(1);
close(atOne.weights[0], checked.practice.conditional_at_one.weights[0], 'the updated weights tie', 1e-12);
close(atOne.mean, checked.practice.conditional_at_one.mean, 'the conditional mean', 1e-12);
close(atOne.variance, checked.practice.conditional_at_one.variance, 'and its spread', 1e-12);
close(atOne.components[0].variance, 0.75, 'each conditional variance', 1e-15);
const atZero = conditionalMixture(0);
close(atZero.weights[1], checked.practice.conditional_at_zero.second_weight, 'practice 8 weight', 1e-12);
close(atZero.mean, checked.practice.conditional_at_zero.mean, 'practice 8 mean', 1e-12);
// Practice 2, the supplied allocation.
const supplied = maximization([0, 2, 5], [[0.8, 0.2], [0.4, 0.6], [0.1, 0.9]], 0.0001)[0];
close(supplied.count, checked.practice.weighted_mstep.count, 'practice 2 count', 1e-12);
close(supplied.weight, checked.practice.weighted_mstep.weight, 'practice 2 weight', 1e-12);
close(supplied.mean, checked.practice.weighted_mstep.mean, 'practice 2 mean', 1e-12);
close(supplied.variance, checked.practice.weighted_mstep.variance, 'practice 2 variance', 1e-12);
close(maximization([0, 2, 5], [[0.8, 0.2], [0.4, 0.6], [0.1, 0.9]], 3)[0].variance, 3, 'the floor clips it upward', 1e-15);
// Practice 1, the unequal-weight boundary.
const boundary = checked.practice.unequal_weight_boundary;
const unequal = [{ weight: 0.25, mean: 0, variance: 1 }, { weight: 0.75, mean: 2, variance: 1 }];
close(mixtureAt(unequal, boundary).responsibilities[0], 0.5, 'the responsibilities tie there', 1e-12);
close(boundary, 1 - Math.log(3) / 2, 'and it is exactly 1 − log(3)/2', 1e-12);
close(mixtureAt(unequal, 1).responsibilities[0], 0.25, 'at 1 the heights cancel', 1e-15);
close(mixtureAt(unequal, 1).density, normal(1, 0, 1), 'so the density is that common height', 1e-15);
record('counts, criteria and limits');

// ------------------------------------------------------- the generated real data
assert.equal(observations.length, 150);
assert.deepEqual(observations.map(row => row[0]), Array.from({ length: 150 }, (_, index) => index + 1));
assert.equal(speciesNames.length, 3);
speciesNames.forEach((_, code) => assert.equal(observations.filter(row => row[4] === code).length, 50, 'fifty of each species'));
[0, 1, 2].forEach(split => assert.equal(observations.filter(row => row[3] === split).length,
  [splitCounts.train, splitCounts.validation, splitCounts.test][split], 'the declared split sizes'));
checked.iris.train_ids.forEach(id => assert.equal(observations[id - 1][3], 0, `row ${id} is a training row`));
checked.iris.test_ids.forEach(id => assert.equal(observations[id - 1][3], 2, `row ${id} is reserved`));
scaler.mean.forEach((value, index) => close(value, checked.iris.scale_mean[index], `the frozen mean ${index}`, 1e-9));
scaler.scale.forEach((value, index) => close(value, checked.iris.scale[index], `the frozen scale ${index}`, 1e-9));
// The scaler is the training rows' own mean and population deviation.
const trainRows = observations.filter(row => row[3] === 0);
[1, 2].forEach(column => {
  const values = trainRows.map(row => row[column]);
  const mean = values.reduce((sum, value) => sum + value, 0) / values.length;
  const deviation = Math.sqrt(values.reduce((sum, value) => sum + (value - mean) ** 2, 0) / values.length);
  close(mean, scaler.mean[column - 1], `training mean of column ${column}`, 1e-9);
  close(deviation, scaler.scale[column - 1], `training scale of column ${column}`, 1e-9);
});
record('real data structure');

assert.equal(candidates.length, 16);
checked.iris.candidates.forEach((probe, index) => {
  const row = candidates[index];
  assert.equal(row[0], probe.covariance_type);
  assert.equal(row[1], probe.components);
  close(row[2], probe.validation_mean_log_density, `validation score ${index}`, 1e-6);
  close(row[3], probe.bic, `training BIC ${index}`, 1e-3);
  record('candidate versus author');
});
const best = candidates.reduce((leader, row) => (row[2] > leader[2] ? row : leader), candidates[0]);
assert.deepEqual([best[0], best[1]], selection.selected, 'the declared rule picks the recorded winner');
const lowestBic = candidates.reduce((leader, row) => (row[3] < leader[3] ? row : leader), candidates[0]);
assert.deepEqual([lowestBic[0], lowestBic[1]], selection.trainingBicWinner, 'training BIC picks a different one');
assert.notDeepEqual(selection.selected, selection.trainingBicWinner, 'which is the whole point of showing both');
// K = 1 is the same model whatever the family name says about covariance.
close(candidates.find(row => row[0] === 'full' && row[1] === 1)[2],
  candidates.find(row => row[0] === 'tied' && row[1] === 1)[2], 'one component, one fit', 1e-9);
close(candidates.find(row => row[0] === 'diag' && row[1] === 1)[2],
  candidates.find(row => row[0] === 'spherical' && row[1] === 1)[2], 'and the two restricted families agree too', 1e-9);
// The reserved test, reported as it came out.
close(selection.testMeanLogDensity, checked.iris.test_mean_log_density, 'the selected test score', 1e-9);
close(selection.baselineTestMeanLogDensity, checked.iris.baseline_test_mean_log_density, 'the baseline test score', 1e-9);
close(selection.testAri, checked.iris.test_ari, 'the ARI diagnostic', 1e-9);
assert(selection.baselineTestMeanLogDensity > selection.testMeanLogDensity, 'the baseline is ahead on the reserved rows');
close(selection.baselineTestMeanLogDensity - selection.testMeanLogDensity, 0.123975, 'by this much per row', 1e-5);
assert.equal(testRows.length, 30);
assert.deepEqual(testRows.map(row => row[0]), checked.iris.test_ids_in_order, 'in the recorded order');
testRows.forEach((row, index) => {
  close(row[1], checked.iris.test_log_density[index], `row ${row[0]} log-density`, 1e-6);
  close(row[2], checked.iris.test_responsibilities[index][0], `row ${row[0]} first responsibility`, 1e-9);
  assert([0, 1].includes(row[3]), 'the argmax names a fitted component');
});
assert.equal(testRows[0][0], 34, 'the first reserved row');
assert(testRows[0][2] > 0.99999 && testRows[0][2] < 1, 'its displayed 1 is a rounded value, not an exact one');
// The selected model is a genuine mixture: two weights that sum to one.
close(selectedParameters.weights.reduce((sum, value) => sum + value, 0), 1, 'the fitted weights sum to one', 1e-9);
assert.equal(selectedParameters.means.length, 2);
selectedParameters.covariances.forEach(matrix => {
  close(matrix[0][1], matrix[1][0], 'each covariance is symmetric', 1e-12);
  assert(matrix[0][0] > 0 && matrix[1][1] > 0 && matrix[0][0] * matrix[1][1] - matrix[0][1] ** 2 > 0, 'and positive definite');
});
// The narrow component that explains the BIC disagreement.
close(narrowComponent.covariance[1][1], narrowComponent.regularization, 'its width sits at the regularization level', 1e-12);
close(narrowComponent.rawWidthStandardDeviation, Math.sqrt(narrowComponent.covariance[1][1]) * scaler.scale[1], 'converted back to centimetres', 1e-9);
assert(narrowComponent.rawWidthStandardDeviation < narrowComponent.recordedResolution / 10, 'far narrower than the recorded resolution');
close(narrowComponent.rawMean[1], narrowComponent.mean[1] * scaler.scale[1] + scaler.mean[1], 'its raw centre', 1e-6);
assert.equal(bayesianSensitivity.length, 3);
bayesianSensitivity.forEach((row, index) => {
  assert.equal(row.concentration, checked.bayesian_sensitivity[index].concentration);
  assert.equal(row.above_one_percent, checked.bayesian_sensitivity[index].active_above_point01, 'the count above one percent');
  assert.equal(row.above_five_percent, 2, 'two components above five percent at every concentration');
  close(row.weights.reduce((sum, value) => sum + value, 0), 1, 'six weights that sum to one', 1e-3);
  record('weight prior versus author');
});
record('real data results');

// ------------------------------------------------ displayed program agreement
assert.equal(Object.keys(examples).length, 3);
assert(examples.emOneDimension.expected.includes('initial log-likelihood -7.158187'));
assert(examples.emOneDimension.expected.includes('means [-1.499994  1.499994]'));
assert(examples.emOneDimension.expected.includes('row sums [1. 1. 1. 1.]'));
assert(examples.irisDensity.expected.includes('selected full 2'));
assert(examples.irisDensity.expected.includes('test log-density -3.011223'));
assert(examples.irisDensity.expected.includes('baseline test log-density -2.887249'));
assert(examples.irisDensity.expected.includes('first test ID 34'));
assert(examples.bayesianWeights.expected.split('\n').length === 3, 'three concentrations');
candidates.forEach(row => assert(examples.irisDensity.expected.includes(`${row[0]} ${row[1]} ${row[2].toFixed(6)}`),
  `the displayed table shows ${row[0]} K=${row[1]}`));
// The displayed program's own trace agrees with the browser model's.
const printed = examples.emOneDimension.expected.split('\n').filter(line => /^\d+ -/.test(line))
  .map(line => Number(line.split(' ')[1]));
printed.forEach((value, index) => close(trace[index + 1].logLikelihood, value, `printed iteration ${index + 1}`, 1e-6));
record('displayed programs');

// Independent boundary equations: distinguish no roots, two roots and a tie
// everywhere. These cases used to be rendered as the same 'identical' outcome.
const pair = (weight, means, variances) => [
  { weight, mean: means[0], variance: variances[0] },
  { weight: 1 - weight, mean: means[1], variance: variances[1] },
];
close(responsibilityCrossings(pair(0.25, [0, 2], [1, 1])).roots[0], 1 - Math.log(3) / 2, 'practice crossing', 1e-12);
assert.deepEqual(responsibilityCrossings(pair(0.5, [0, 0], [1, 1])), { everywhere: true, roots: [] });
assert.deepEqual(responsibilityCrossings(pair(0.2, [0, 0], [1, 1])), { everywhere: false, roots: [] });
assert.deepEqual(responsibilityCrossings(pair(0.05, [0, 0], [0.25, 1])), { everywhere: false, roots: [] });
const wide = pair(0.5, [0, 0], [4, 1]);
const twoRoots = responsibilityCrossings(wide).roots;
assert.equal(twoRoots.length, 2);
twoRoots.forEach((root, index) => {
  close(root, (index === 0 ? -1 : 1) * Math.sqrt(Math.log(4) / 0.75), 'both crossing locations', 1e-12);
  const parts = mixtureAt(wide, root).components;
  close(parts[0].logWeighted, parts[1].logWeighted, 'weighted log densities tie', 1e-12);
});
record('all responsibility crossing cases');

// --------------------------------------------------------------------- record
const sources = [
  'src/learn/data/gmm-models.js',
  'src/learn/data/gmm-iris-data.js',
  'src/learn/data/gmm-examples.js',
  'src/learn/components/lesson-labs/GmmShared.jsx',
  'src/learn/components/lesson-labs/GmmLabs.jsx',
  'src/learn/components/lesson-labs/GmmFigures.jsx',
  'src/learn/components/lesson-labs/gmm-labs.css',
  'src/learn/data/topics/gaussian-mixture-models-gmm-em-algorithm.jsx',
  'src/learn/data/curriculum/blueprints/gaussian-mixture-models-gmm-em-algorithm.js',
];
const hash = file => crypto.createHash('sha256').update(fs.readFileSync(file)).digest('hex');
const evidence = {
  checkedAt: new Date().toISOString(),
  sourceHashes: Object.fromEntries(sources.filter(fs.existsSync).map(file => [file, hash(file)])),
  verifierHash: hash('scripts/verify-gmm-models.mjs'),
  counts,
  totalGroupedChecks: Object.values(counts).reduce((sum, value) => sum + value, 0),
  scope: 'Browser mixture models against the content-phase native probes (responsibility, EM trace, null and floor variations, bound decomposition, collapse, plane geometry, small-variance limit, practice values), independent identities (numeric integration of the density, eigenpair mapping, a scanned constrained optimum, nondecrease along three starts, the bound below the objective for non-posterior allocations), and the generated Iris module: split membership, the frozen scaler recomputed from the training rows, all 16 candidates, the reserved test outcome and the narrow component.',
  limitations: [
    'The Iris candidates are precomputed native fits; the browser reproduces their recorded outcomes, not the fitting.',
    'Displayed program output is executed separately by scripts/verify-gmm-examples.py.',
    'Rendering, interaction and independent review are separate steps.',
  ],
  passed: true,
};
fs.mkdirSync('docs/teaching/evidence', { recursive: true });
fs.writeFileSync('docs/teaching/evidence/gmm-models.json', JSON.stringify(evidence, null, 2) + '\n');
console.log(`PASS: ${evidence.totalGroupedChecks} grouped Gaussian mixture model checks.`);
