// Bounded independent checks of the regularization browser models against the
// content phase's recorded calculations, the manuscript's worked values and
// analytic identities, plus structural checks of the generated data module.
// Run: node scripts/verify-regularization-models.mjs
import assert from 'node:assert/strict';
import fs from 'node:fs';
import crypto from 'node:crypto';
import {
  centre, coordinateFit, criteria, curvatures, duplicateAnalysis, dropoutEnumeration, earlyStoppingFilter,
  encodeSixteenBits, factorOptimum, fixtures, lambdaMax, limits, modelPrediction, nonlinearInset,
  penaltyBoundary, penaltyBoundaryRadius, penaltyMeasure, penalisedObjective, polynomialTerms, ridgeFilter,
  scalarObjective, scalarSlopes, scalarSolution, smoothnessComparison, softThreshold, solveLinear,
  termsTouchedBy, twoCoordinateSolution,
} from '../src/learn/data/regularization-models.js';
import {
  baselineMeanMse, baselines, candidates, coefficientPaths, elasticNetRatio, featureNames, inferenceFixture,
  olsMeanMse, provenance, rawFeatureRanges, ridgeModel, selected, strengths,
} from '../src/learn/data/regularization-data.js';

const packet = 'docs/teaching/drafts/regularization-l1-l2-elastic-net-dropout';
const recorded = JSON.parse(fs.readFileSync(`${packet}/calculated-inputs.json`, 'utf8'));
const constructed = recorded.constructed;
const airfoil = recorded.airfoil;
const examples = JSON.parse(fs.readFileSync('src/learn/data/regularization-examples.js', 'utf8')
  .replace(/^[\s\S]*?export const regularizationExamples = /, '').replace(/;\s*$/, ''));

const counts = {};
const record = name => { counts[name] = (counts[name] ?? 0) + 1; };
const close = (actual, expected, label, tolerance = 1e-9) =>
  assert(Math.abs(actual - expected) <= tolerance * Math.max(1, Math.abs(expected)), `${label}: ${actual} versus ${expected}`);
const vector = (actual, expected, label, tolerance = 1e-9) => {
  assert.equal(actual.length, expected.length, `${label}: length`);
  actual.forEach((value, index) => close(value, expected[index], `${label}[${index}]`, tolerance));
};

/* ------------------------------------------------- soft threshold and scalar */

// An exact zero, never a small number: the interior of the interval collapses.
for (const z of [-1, -0.999999, -0.4, 0, 0.4, 0.999999, 1]) {
  assert.equal(softThreshold(z, 1), 0, `S(${z}, 1) is exactly zero`);
}
close(softThreshold(3, 1), 2, 'S(3, 1)');
close(softThreshold(-3, 1), -2, 'S(-3, 1)');
close(softThreshold(1.4, 1), 0.4000000000000001, 'S(1.4, 1)', 1e-12);
assert.throws(() => softThreshold(1, -1), RangeError, 'a negative threshold is refused');
assert.throws(() => softThreshold(Number.NaN, 1), RangeError, 'a non-finite preference is refused');
assert.throws(() => scalarSolution(1, -1, 1), RangeError, 'a negative strength is refused');
assert.throws(() => scalarSolution(1, 1, 1.5), RangeError, 'a mixing fraction above one is refused');
// Hard thresholding would leave a surviving z unchanged; soft thresholding does not.
assert(scalarSolution(3, 1, 1).coefficient !== 3, 'soft thresholding shrinks a survivor');
record('soft threshold');

// The manuscript's scalar table at lambda = 1.
for (const [z, ridge, lasso, elastic] of [[3, 1.5, 2, 5 / 3], [0.4, 0.2, 0, 0], [-2, -1, -1, -1]]) {
  close(scalarSolution(z, 1, 0).coefficient, ridge, `ridge at z=${z}`, 1e-12);
  close(scalarSolution(z, 1, 1).coefficient, lasso, `lasso at z=${z}`, 1e-12);
  close(scalarSolution(z, 1, 0.5).coefficient, elastic, `elastic net at z=${z}`, 1e-12);
  record('scalar family table');
}
// Practice 1: a negative data preference.
close(scalarSolution(-2.4, 0.6, 0).coefficient, -1.5, 'practice 1 ridge', 1e-12);
close(scalarSolution(-2.4, 0.6, 1).coefficient, -1.8, 'practice 1 lasso', 1e-12);
close(scalarSolution(-2.4, 0.6, 0.5).coefficient, -21 / 13, 'practice 1 elastic net', 1e-12);
// Investigation 1 contrast and nulls.
assert.equal(scalarSolution(0.4, 1, 1).coefficient, 0, 'z = 0.4 stays exactly zero');
close(scalarSolution(1.4, 1, 1).coefficient, 0.4, 'z = 1.4 crosses the threshold', 1e-12);
assert.equal(scalarSolution(-0.6, 1, 1).coefficient, 0, 'z = -0.6 is still exactly zero');
assert.notEqual(scalarObjective(0, -0.6, 1, 1).data, scalarObjective(0, 0.4, 1, 1).data,
  'the data cost changed even though the answer did not');
for (const ratio of [0, 0.25, 0.5, 0.75, 1]) {
  close(scalarSolution(0.4, 0, ratio).coefficient, 0.4, `lambda = 0 returns z for rho = ${ratio}`, 1e-12);
}
record('scalar contrasts and nulls');

// The scalar minimum really is a minimum: scan the objective on a fine grid.
for (const [z, strength, ratio] of [[3, 1, 0], [3, 1, 1], [3, 1, 0.5], [0.4, 1, 1], [-2.4, 0.6, 0.5], [2, 0.3, 0.2]]) {
  const answer = scalarSolution(z, strength, ratio);
  const best = scalarObjective(answer.coefficient, z, strength, ratio).total;
  for (let step = -400; step <= 400; step += 1) {
    const w = answer.coefficient + step * 0.01;
    assert(scalarObjective(w, z, strength, ratio).total >= best - 1e-12,
      `scanned objective below the closed-form minimum at z=${z}, rho=${ratio}, w=${w}`);
  }
  const slopes = scalarSlopes(answer.coefficient, z, strength, ratio);
  assert(slopes.stationary, `the subgradient condition holds at the minimum for z=${z}, rho=${ratio}`);
  record('scanned scalar optimum');
}
// At a threshold endpoint the coefficient is zero and the condition holds with equality.
assert.equal(scalarSolution(1, 1, 1).coefficient, 0, 'an endpoint preference gives zero');
assert(scalarSolution(1, 1, 1).atEndpoint, 'the endpoint is reported as such');
record('threshold endpoint');

// Independent derivative of the smooth branches; the L1 interval only exists at zero.
for (const [w, z, strength, ratio] of [[0.4, 1.4, 1, 1], [-0.4, -1.4, 1, 1], [0.2, 0.4, 1, 0]]) {
  const expected = w - z + strength * (1 - ratio) * w + strength * ratio * Math.sign(w);
  const slopes = scalarSlopes(w, z, strength, ratio);
  close(slopes.fromLeft, expected, 'left derivative away from kink', 1e-12);
  close(slopes.fromRight, expected, 'right derivative away from kink', 1e-12);
}
for (const sign of [-1, 1]) {
  const endpoint = scalarSolution(sign * 0.07, 0.7, 0.1);
  assert.equal(endpoint.coefficient, 0, 'decimal threshold equality gives exact zero');
  assert(endpoint.atEndpoint, 'decimal equality is reported consistently');
  assert.equal(scalarSolution(sign * 0.0701, 0.7, 0.1).sign, sign, 'a real nearby survivor retains its sign');
}
assert.equal(softThreshold(1e-20, 0), 1e-20, 'no absolute floor erases a real unpenalized coefficient');
record('sequential derivative and decimal threshold regressions');

/* --------------------------------------------- the four-row constructed fit */

const { X, y } = fixtures.orthogonal;
vector(centre(X, y).featureMeans, [0, 0], 'the four rows are already centred');
close(centre(X, y).targetMean, 0, 'the target mean is zero', 1e-12);
vector(curvatures(X, y), [1, 1], 'Z transpose Z over n is the identity');
close(lambdaMax(X, y), 3, 'lambda max for the four rows', 1e-12);
for (const [key, ratio] of [['0', 0], ['0.5', 0.5], ['1', 1]]) {
  const fit = coordinateFit(X, y, 1, ratio);
  vector(fit.weights, constructed.orthogonal_base[key].weights, `the packet's slopes at rho=${ratio}`, 1e-9);
  close(fit.intercept, constructed.orthogonal_base[key].intercept, `the packet's intercept at rho=${ratio}`, 1e-12);
  assert.equal(fit.sweeps, constructed.orthogonal_base[key].history.length, `the packet's sweep count at rho=${ratio}`);
  assert.equal(fit.converged, constructed.orthogonal_base[key].converged, `convergence at rho=${ratio}`);
  record('four-row fit against the packet');
}
// The three stated objective values, which come from three different penalties.
close(coordinateFit(X, y, 1, 0).objective, 2.29, 'the ridge objective', 1e-12);
close(coordinateFit(X, y, 1, 1).objective, 2.58, 'the lasso objective', 1e-12);
close(coordinateFit(X, y, 1, 0.5).objective, 2.496666666666667, 'the elastic-net objective', 1e-12);
assert.equal(coordinateFit(X, y, 1, 1).weights[1], 0, 'the lasso second slope is exactly zero');
record('four-row objectives');

// The two investigation-2 contrasts.
const changed = coordinateFit(fixtures.changedTarget.X, fixtures.changedTarget.y, 1, 1);
vector(changed.weights, constructed.changed_first_target.weights, 'the changed-target slopes', 1e-9);
close(changed.intercept, 1, 'the changed-target intercept', 1e-12);
const shifted = coordinateFit(fixtures.shiftedTargets.X, fixtures.shiftedTargets.y, 1, 1);
vector(shifted.weights, constructed.target_shift_null.weights, 'the shifted-target slopes', 1e-9);
close(shifted.intercept, 7, 'the shifted-target intercept', 1e-12);
// The null really is a null: identical residuals and an identical data term.
const base = coordinateFit(X, y, 1, 1);
vector(shifted.residual, base.residual, 'a target shift leaves every residual where it was', 1e-12);
close(shifted.data, base.data, 'and leaves the data term unchanged', 1e-12);
// Reordering the rows changes nothing about the fit.
const reordered = coordinateFit([[-1, -1], [1, 1], [-1, 1], [1, -1]], [-3.4, 3.4, -2.6, 2.6], 1, 1);
vector(reordered.weights, base.weights, 'row order does not change the fit', 1e-12);
close(reordered.intercept, base.intercept, 'nor the intercept', 1e-12);
// Practice 2.
vector(coordinateFit(X, y, 0.5, 1).weights, [2.5, 0], 'practice 2 slopes', 1e-12);
vector(coordinateFit(X, [6.4, 5.6, 0.4, -0.4], 0.5, 1).weights, [2.5, 0], 'practice 2 shifted slopes', 1e-12);
close(coordinateFit(X, [6.4, 5.6, 0.4, -0.4], 0.5, 1).intercept, 3, 'practice 2 shifted intercept', 1e-12);
record('four-row contrasts and nulls');

// A constant centred column: zero curvature, no association, a declared zero.
const constantFit = coordinateFit(fixtures.constantColumn.X, fixtures.constantColumn.y, 1, 1);
close(curvatures(fixtures.constantColumn.X, fixtures.constantColumn.y)[1], 0, 'a constant column has zero curvature', 1e-12);
assert.equal(constantFit.weights[1], 0, 'its coefficient is the declared zero representative');
assert(constantFit.converged, 'and the fit still converges');
assert.equal(constantFit.history[0].steps[1].flat, false, 'positive L1 penalty has a unique zero, not a flat objective');
for (const weight of [-1, -0.1, 0.1, 1]) {
  const trial = [constantFit.weights[0], weight];
  close(penalisedObjective(fixtures.constantColumn.X, fixtures.constantColumn.y, trial, 1, 1).total - constantFit.objective,
    Math.abs(weight), 'moving the constant coordinate costs exactly lambda times absolute weight', 1e-12);
}
assert(coordinateFit(fixtures.constantColumn.X, fixtures.constantColumn.y, 0, 1).history[0].steps[1].flat,
  'without either penalty the zero-curvature coordinate really is flat');
record('constant column');

const tinyDropout = dropoutEnumeration([0.0001, 0], [0.0001, 0], 1, 0.5);
assert(Math.abs(tinyDropout.analyticExtra / 5e-17 - 1) < 1e-12, 'tiny dropout excess is positive 5e-17 by the variance formula');
assert.equal(dropoutEnumeration([0.0001, 0], [0.0001, 0], 1, 1).analyticExtra, 0, 'q=1 has genuinely zero excess');
record('sequential tiny dropout excess');

// The optimality conditions the solver claims, checked independently.
for (const [design, targets, strength, ratio] of [
  [X, y, 1, 1], [X, y, 1, 0], [X, y, 1, 0.5], [X, y, 0.05, 1],
  [fixtures.duplicate.X, fixtures.duplicate.y, 1, 0], [fixtures.duplicate.X, fixtures.duplicate.y, 1, 0.5],
  [[[1, 2], [2, 1], [3, 5], [4, 3]], [2, 3, 7, 6], 0.2, 0.4],
  [[[1, 2], [2, 1], [3, 5], [4, 3]], [2, 3, 7, 6], 2, 1],
]) {
  const fit = coordinateFit(design, targets, strength, ratio);
  const { Z, t, n, p } = centre(design, targets);
  const residual = t.map((value, row) => value - Z[row].reduce((sum, entry, column) => sum + entry * fit.weights[column], 0));
  for (let column = 0; column < p; column += 1) {
    const gradient = -Z.reduce((sum, row, index) => sum + row[column] * residual[index], 0) / n
      + strength * (1 - ratio) * fit.weights[column];
    if (fit.weights[column] !== 0) {
      close(gradient + strength * ratio * Math.sign(fit.weights[column]), 0, 'a nonzero coordinate is stationary', 1e-8);
    } else {
      assert(Math.abs(gradient) <= strength * ratio + 1e-8, 'a zero coordinate sits inside its subgradient interval');
    }
  }
  // A grid of nearby coefficient vectors must not beat the returned one.
  for (const delta of [-0.05, -0.01, 0.01, 0.05]) {
    for (let column = 0; column < p; column += 1) {
      const trial = fit.weights.map((value, index) => (index === column ? value + delta : value));
      assert(penalisedObjective(design, targets, trial, strength, ratio).total >= fit.objective - 1e-9,
        'no nearby coefficient vector has a smaller objective');
    }
  }
  // The objective never rises along the recorded sweep history.
  fit.history.forEach((entry, index) => {
    if (index > 0) assert(entry.objective <= fit.history[index - 1].objective + 1e-12, 'the sweep history is non-increasing');
  });
  record('independent optimality check');
}
// The intercept is recovered from the means, not fitted.
const offset = coordinateFit([[1, 1], [1, -1], [-1, 1], [-1, -1]], [13.4, 12.6, 7.4, 6.6], 1, 1);
close(offset.intercept, 10, 'the recovered intercept absorbs a target shift', 1e-12);
record('intercept recovery');

// Input the fit must refuse rather than silently repair.
assert.throws(() => coordinateFit(X, [1, 2, 3], 1, 1), RangeError, 'one target per row');
assert.throws(() => coordinateFit(X, [1, 2, Number.NaN, 4], 1, 1), RangeError, 'a non-finite target');
assert.throws(() => coordinateFit([[1, 2], [3]], [1, 2], 1, 1), RangeError, 'a ragged design');
assert.throws(() => coordinateFit(X, y, -1, 1), RangeError, 'a negative strength');
assert.throws(() => coordinateFit(X, y, 1, 2), RangeError, 'a mixing fraction outside [0, 1]');
assert.throws(() => coordinateFit(X, y, 1, 1, { order: [0, 0] }), RangeError, 'an order that repeats a column');
assert.throws(() => coordinateFit([], [], 1, 1), RangeError, 'an empty design');
record('refused input');

// The sweep cap is reported honestly instead of being presented as an optimum.
const capped = coordinateFit(fixtures.duplicate.X, fixtures.duplicate.y, 1, 0, { maxSweeps: 2 });
assert.equal(capped.converged, false, 'a capped run says it did not converge');
assert.equal(capped.sweeps, 2, 'and reports the sweeps it used');
assert(capped.kktResidual > 1e-10, 'with its actual optimality violation');
record('honest convergence');

/* --------------------------------------------------- duplicate columns */

for (const [key, ratio] of [['0', 0], ['0.5', 0.5], ['1', 1]]) {
  const fit = coordinateFit(fixtures.duplicate.X, fixtures.duplicate.y, 1, ratio);
  vector(fit.weights, constructed.duplicate[key].weights, `the packet's duplicate fit at rho=${ratio}`, 1e-7);
  assert.equal(fit.sweeps, constructed.duplicate[key].history.length, `the packet's duplicate sweep count at rho=${ratio}`);
  record('duplicate fit against the packet');
}
// Visiting the other column first returns the other endpoint, with the same
// prediction and the same objective.
const forward = coordinateFit(fixtures.duplicate.X, fixtures.duplicate.y, 1, 1, { order: [0, 1] });
const backward = coordinateFit(fixtures.duplicate.X, fixtures.duplicate.y, 1, 1, { order: [1, 0] });
vector(forward.weights, [1, 0], 'the forward order gives (1, 0)', 1e-12);
vector(backward.weights, [0, 1], 'the reversed order gives (0, 1)', 1e-12);
vector(forward.fitted, backward.fitted, 'both give identical fitted values', 1e-12);
close(forward.objective, backward.objective, 'and identical objectives', 1e-12);
const lasso = duplicateAnalysis(1, 1);
close(lasso.sum, 1, 'the optimal lasso sum');
for (const point of lasso.minimizers) close(lasso.objective(point).total, 1.5, 'every listed lasso minimizer ties', 1e-12);
close(duplicateAnalysis(1, 0).objective([2 / 3, 2 / 3]).total, 2 / 3, 'the ridge duplicate objective', 1e-12);
close(duplicateAnalysis(1, 0.5).objective([0.6, 0.6]).total, 1.1, 'the elastic-net duplicate objective', 1e-12);
vector(duplicateAnalysis(1, 0).balanced, [2 / 3, 2 / 3], 'the ridge duplicate answer', 1e-12);
vector(duplicateAnalysis(1, 0.5).balanced, [0.6, 0.6], 'the elastic-net duplicate answer', 1e-12);
// The three families choose different sums, not one sum reallocated.
close(duplicateAnalysis(1, 0).sum, 4 / 3, 'the ridge sum', 1e-12);
close(duplicateAnalysis(1, 0.5).sum, 1.2, 'the elastic-net sum', 1e-12);
// A strictly convex penalty leaves no segment; an off-balance pair costs more.
assert(duplicateAnalysis(1, 0).objective([1, 1 / 3]).total > duplicateAnalysis(1, 0).objective([2 / 3, 2 / 3]).total,
  'ridge strictly prefers the balanced allocation');
// Practice 4.
close(duplicateAnalysis(0.5, 1, 3).sum, 2.5, 'practice 4 sum', 1e-12);
const practiceDuplicate = duplicateAnalysis(0.5, 1, 3);
close(practiceDuplicate.objective([2.5, 0]).total, practiceDuplicate.objective([1.25, 1.25]).total,
  'practice 4 minimizers tie', 1e-12);
record('duplicate analysis');

/* ------------------------------------------------ two-coordinate geometry */

vector(twoCoordinateSolution([3, 0.4], 0.1, 1).weights, [2.9, 0.3], 'lasso at lambda 0.1 keeps both coordinates', 1e-12);
vector(twoCoordinateSolution([3, 0.4], 1, 1).weights, [2, 0], 'lasso at lambda 1 drops one', 1e-12);
vector(twoCoordinateSolution([3, 0.4], 1, 0).weights, [1.5, 0.2], 'ridge is z over 1 + lambda', 1e-12);
vector(twoCoordinateSolution([3, 0.4], 1, 0.5).weights, [5 / 3, 0], 'elastic net at lambda 1', 1e-12);
// A budget is the attained penalty measure, so it differs between families at
// one numerical lambda.
const budgets = ['ridge', 'lasso', 'elastic'].map((name, index) =>
  twoCoordinateSolution([3, 0.4], 1, [0, 1, 0.5][index]).budget);
assert(new Set(budgets.map(value => value.toFixed(9))).size === 3, 'one lambda gives three different budgets');
close(twoCoordinateSolution([3, 0.4], 1, 1).budget, 2, 'the lasso budget is its L1 value', 1e-12);
close(twoCoordinateSolution([3, 0.4], 1, 0).budget, (1.5 ** 2 + 0.2 ** 2) / 2, 'the ridge budget is half its squared norm', 1e-12);
// The boundary really is the level set, sampled in every direction.
for (const ratio of [0, 0.25, 0.5, 0.75, 1]) {
  const solution = twoCoordinateSolution([3, 0.4], 1, ratio);
  for (const point of penaltyBoundary(ratio, solution.budget, 61)) {
    close(penaltyMeasure(point, ratio), solution.budget, `the level set holds at rho=${ratio}`, 1e-9);
  }
  // The solution itself lies on its own boundary.
  close(penaltyMeasure(solution.weights, ratio), solution.budget, 'the solution sits on its boundary', 1e-12);
  record('penalty level set');
}
// A diamond has corners on the axes; a disk does not.
close(penaltyBoundaryRadius([1, 0], 1, 2), 2, 'the lasso boundary reaches 2 on the axis', 1e-12);
close(penaltyBoundaryRadius([1, 1], 1, 2), Math.SQRT2, 'and 1 in each coordinate on the diagonal', 1e-12);
close(penaltyBoundaryRadius([1, 0], 0, 2), 2, 'the ridge boundary is a circle of radius 2', 1e-12);
close(penaltyBoundaryRadius([1, 1], 0, 2), 2, 'the same radius in every direction', 1e-12);
assert(penaltyBoundaryRadius([1, 1], 0.5, 2) < penaltyBoundaryRadius([1, 0], 0.5, 2) * Math.SQRT2,
  'the mixed boundary keeps its axis corners rather than rounding out to a circle');
assert.throws(() => penaltyBoundaryRadius([0, 0], 1, 1), RangeError, 'a zero direction is refused');
assert.throws(() => penaltyBoundaryRadius([1, 0], 1, 0), RangeError, 'a zero budget is refused');
// The contact point is the arithmetic's answer: no point inside the budget has
// a smaller data loss.
for (const [strength, ratio] of [[1, 1], [0.1, 1], [1, 0], [1, 0.5], [0.1, 0.5]]) {
  const solution = twoCoordinateSolution([3, 0.4], strength, ratio);
  for (const point of penaltyBoundary(ratio, solution.budget, 121)) {
    const loss = 0.5 * ((point[0] - 3) ** 2 + (point[1] - 0.4) ** 2);
    assert(loss >= solution.data - 1e-9,
      `a boundary point beat the contact point at lambda=${strength}, rho=${ratio}`);
  }
  record('constrained contact');
}
record('two-coordinate geometry');

/* ------------------------------------------------------------- dropout */

const dropout = dropoutEnumeration([2, 1], [1, -1], 1, 0.5);
constructed.dropout_masks.forEach((saved, index) => {
  assert.deepEqual(dropout.branches[index].mask, saved.mask, 'the mask order matches the packet');
  close(dropout.branches[index].probability, saved.probability, 'the branch probability', 1e-12);
  close(dropout.branches[index].prediction, saved.prediction, 'the branch prediction', 1e-12);
  close(dropout.branches[index].halfSquaredLoss, saved.half_squared_loss, 'the branch half-loss', 1e-12);
  record('dropout branch against the packet');
});
close(dropout.expectedPrediction, constructed.dropout_mean, 'the mean noisy prediction', 1e-12);
close(dropout.expectedLoss, constructed.dropout_expected_half_loss, 'the expected noisy half-loss', 1e-12);
close(dropout.cleanLoss, 0, 'the clean half-loss', 1e-12);
close(dropout.analyticExtra, 2.5, 'the analytic penalty term', 1e-12);
assert(dropout.agrees, 'enumeration and formula agree exactly');
close(dropout.branches.reduce((sum, branch) => sum + branch.probability, 0), 1, 'the branch probabilities sum to one', 1e-12);
// Probabilities are not uniform away from q = 0.5.
const uneven = dropoutEnumeration([2, 1], [1, -1], 1, 0.75);
assert(new Set(uneven.branches.map(branch => branch.probability.toFixed(9))).size > 1,
  'branch probabilities differ once q is not one half');
close(uneven.branches.reduce((sum, branch) => sum + branch.probability, 0), 1, 'and still sum to one', 1e-12);
// Practice 5.
const practice = dropoutEnumeration([1, 2], [2, 0], 1, 0.75);
close(practice.cleanPrediction, 2, 'practice 5 clean prediction', 1e-12);
close(practice.expectedPrediction, 2, 'practice 5 mean prediction', 1e-12);
close(practice.cleanLoss, 0.5, 'practice 5 clean half-loss', 1e-12);
close(practice.expectedLoss, 7 / 6, 'practice 5 expected half-loss', 1e-12);
close(practice.extraLoss, 2 / 3, 'practice 5 difference', 1e-12);
close(practice.analyticExtra, (1 - 0.75) / (2 * 0.75) * 4, 'practice 5 formula', 1e-12);
// A zero coefficient makes its mask irrelevant.
assert.equal(practice.branches[0].prediction, practice.branches[1].prediction, 'the second mask changes nothing');
assert.equal(practice.branches[2].prediction, practice.branches[3].prediction, 'in either state of the first');
// q = 1 removes the extra loss and leaves zero-probability branches labelled.
const deterministic = dropoutEnumeration([2, 1], [1, -1], 1, 1);
close(deterministic.extraLoss, 0, 'q = 1 leaves no extra loss', 1e-12);
assert.equal(deterministic.branches.filter(branch => branch.possible).length, 1, 'only one branch is possible at q = 1');
assert(deterministic.branches.every(branch => Number.isFinite(branch.prediction)), 'nothing is divided by zero at q = 1');
// Changing the target moves both losses and leaves their difference fixed.
const moved = dropoutEnumeration([2, 1], [1, -1], 3, 0.5);
assert.notEqual(moved.cleanLoss, dropout.cleanLoss, 'a different target changes the clean loss');
close(moved.extraLoss, dropout.extraLoss, 'but not the difference', 1e-12);
// The expected loss can never fall below the clean loss here.
for (const keep of [0.1, 0.25, 0.5, 0.75, 0.9, 1]) {
  const trial = dropoutEnumeration([2, 1], [1, -1], 1, keep);
  assert(trial.extraLoss >= -1e-12, `the extra loss is nonnegative at q=${keep}`);
  assert(trial.agrees, `the formula matches at q=${keep}`);
  close(trial.expectedPrediction, trial.cleanPrediction, `the mean is preserved at q=${keep}`, 1e-12);
  record('dropout across keep probabilities');
}
assert.throws(() => dropoutEnumeration([2, 1], [1, -1], 1, 0), RangeError, 'a zero keep probability is refused');
assert.throws(() => dropoutEnumeration([2, 1], [1, -1], 1, 1.2), RangeError, 'a keep probability above one is refused');
assert.throws(() => dropoutEnumeration([2], [1], 1, 0.5), RangeError, 'this enumeration takes two contributions');
const inset = nonlinearInset();
close(inset.meanOutput, 0.5, 'the mean of f');
close(inset.outputOfMean, 0, 'f of the mean input');
record('dropout identities');

/* ----------------------------------------------------- directions and filters */

close(ridgeFilter(4, 1).dataMultiplier, 16 / 17, 'sigma 4 at n lambda 1', 1e-12);
close(ridgeFilter(0.5, 1).dataMultiplier, 0.2, 'sigma 0.5 at n lambda 1', 1e-12);
close(ridgeFilter(4, 4).dataMultiplier, 0.8, 'sigma 4 at n lambda 4', 1e-12);
close(ridgeFilter(0.5, 4).dataMultiplier, 1 / 17, 'sigma 0.5 at n lambda 4', 1e-12);
close(ridgeFilter(4, 0).dataMultiplier, 1, 'no penalty keeps the whole fitted component', 1e-12);
assert.throws(() => ridgeFilter(0, 0), RangeError, 'a zero singular value with no penalty is refused');
assert.throws(() => ridgeFilter(-1, 1), RangeError, 'a negative singular value is refused');
constructed.early_stopping.eigenvalues.forEach((eigenvalue, index) => {
  const filter = earlyStoppingFilter(eigenvalue, constructed.early_stopping.step_size, constructed.early_stopping.steps);
  close(filter.factor, constructed.early_stopping.fit_factors[index], `the packet's fit factor for a=${eigenvalue}`, 1e-9);
  close(filter.matchingStrength, constructed.early_stopping.matching_ridge_strengths[index],
    `the packet's matching ridge strength for a=${eigenvalue}`, 1e-9);
  // The matching strength really reproduces that one factor.
  close(eigenvalue / (eigenvalue + filter.matchingStrength), filter.factor, 'and reproduces it exactly', 1e-9);
  record('early stopping against the packet');
});
assert.notEqual(earlyStoppingFilter(1, 0.1, 1).matchingStrength.toFixed(6),
  earlyStoppingFilter(4, 0.1, 1).matchingStrength.toFixed(6), 'one common lambda cannot match both');
assert.throws(() => earlyStoppingFilter(1, 0.1, 0), RangeError, 'a zero step count is refused');
record('filters');

/* ---------------------------------------------- factor penalty and smoothness */

constructed.factor_penalty.forEach(saved => {
  const answer = factorOptimum(saved.strength);
  close(answer.product, saved.optimal_product, `the packet's optimal product at lambda=${saved.strength}`, 1e-12);
  close(answer.magnitude, saved.balanced_magnitude, `the packet's balanced magnitude at lambda=${saved.strength}`, 1e-12);
  record('factor optimum against the packet');
});
const quarter = factorOptimum(0.25);
close(quarter.data, 0.125, 'the data cost at lambda 0.25', 1e-12);
close(quarter.penalty, 0.25, 'the penalty at lambda 0.25', 1e-12);
close(quarter.total, 0.375, 'the total at lambda 0.25', 1e-12);
close(quarter.balancedZeroLoss.total, 0.5, 'the balanced zero-loss total', 1e-12);
close(quarter.cost(2, 0.5).penalty, 0.25 * 4.25, 'the same-prediction alternative costs more', 1e-12);
close(quarter.cost(quarter.magnitude, quarter.magnitude).total, quarter.total, 'the balanced factors attain the optimum', 1e-12);
close(quarter.cost(-quarter.magnitude, -quarter.magnitude).total, quarter.total, 'so do the equal negative factors', 1e-12);
// Scan the two-factor surface: nothing beats the reduced answer.
for (let a = -2.5; a <= 2.5; a += 0.02) {
  for (let b = -2.5; b <= 2.5; b += 0.02) {
    assert(quarter.cost(a, b).total >= quarter.total - 1e-9, `a factor pair beat the optimum at (${a}, ${b})`);
  }
}
const tenth = factorOptimum(0.1);
close(tenth.product, 0.8, 'practice 7 product', 1e-12);
close(tenth.total, 0.18, 'practice 7 total', 1e-12);
assert(tenth.total < tenth.balancedZeroLoss.total, 'practice 7 beats the balanced zero-loss pair');
assert.equal(factorOptimum(0.5).product, 0, 'both factors are zero at lambda 0.5');
assert(factorOptimum(0).degenerate, 'lambda 0 is flagged as the degenerate case');
assert.throws(() => factorOptimum(-1), RangeError, 'a negative strength is refused');
record('factor penalty');

const smooth = smoothnessComparison(constructed.smoothness.input, 1);
vector(smooth.difference, constructed.smoothness.solution, "the packet's difference-penalty solution", 1e-12);
vector(smooth.identity, constructed.smoothness.the_ridge_solution ?? constructed.smoothness.ridge_solution,
  "the packet's identity-penalty solution", 1e-12);
vector(smooth.differenceEdges, [0.5, -0.5], 'the stated edge differences', 1e-12);
// Substituting the solution back into the system reproduces the observation.
smooth.system.forEach((row, index) => {
  close(row.reduce((sum, value, column) => sum + value * smooth.difference[column], 0), smooth.signal[index],
    'the difference solution satisfies its own system', 1e-12);
});
const shiftedSmooth = smoothnessComparison(fixtures.shiftedDenoising, 1);
vector(shiftedSmooth.difference, [3.5, 4, 3.5], 'practice 10 difference solution', 1e-12);
vector(shiftedSmooth.identity, [1.5, 2.5, 1.5], 'practice 10 identity solution', 1e-12);
shiftedSmooth.difference.forEach((value, index) => {
  close(value - smooth.difference[index], 3, 'a constant shift passes straight through L', 1e-12);
});
assert.throws(() => smoothnessComparison([0, 2], 1), RangeError, 'this example uses three positions');
assert.throws(() => solveLinear([[0, 0], [0, 0]], [1, 1]), RangeError, 'a singular system is refused');
vector(solveLinear([[2, 1], [1, 3]], [3, 5]), [0.8, 1.4], 'the small solver is correct', 1e-12);
record('smoothness');

/* ----------------------------------------------------- complexity accounts */

constructed.criteria.forEach(saved => {
  const answer = criteria(saved.log_likelihood, saved.parameters, 100);
  close(answer.aic, saved.aic, `the packet's AIC for the ${saved.model} model`, 1e-12);
  close(answer.bic, saved.bic, `the packet's BIC for the ${saved.model} model`, 1e-12);
  record('criteria against the packet');
});
close(criteria(-146, 5, 100).bic - criteria(-150, 3, 100).bic + 8, 2 * Math.log(100), 'the BIC charge for two parameters', 1e-12);
close(criteria(-80, 2, 50).aic, 164, 'practice 9 small AIC', 1e-12);
close(criteria(-77, 4, 50).aic, 162, 'practice 9 large AIC', 1e-12);
close(criteria(-80, 2, 50).bic, 160 + 2 * Math.log(50), 'practice 9 small BIC', 1e-12);
close(criteria(-77, 4, 50).bic, 154 + 4 * Math.log(50), 'practice 9 large BIC', 1e-12);
assert(criteria(-77, 4, 50).aic < criteria(-80, 2, 50).aic && criteria(-77, 4, 50).bic > criteria(-80, 2, 50).bic,
  'the two criteria disagree on this pair');
assert.throws(() => criteria(-1, 2.5, 100), RangeError, 'a fractional parameter count is refused');
assert.throws(() => criteria(-1, 2, 0), RangeError, 'a zero observation count is refused');

const [repeating, literal, practiceCode] = fixtures.messages.map(encodeSixteenBits);
assert.equal(repeating.mode, 1);
assert.equal(repeating.payload, '0101');
assert.equal(repeating.totalBits, 5);
assert.equal(repeating.decoded, repeating.message, 'mode 1 decodes back to the message');
assert.equal(literal.mode, 0);
assert.equal(literal.totalBits, 17);
assert.equal(literal.decoded, literal.message, 'mode 0 decodes back to the message');
assert.equal(practiceCode.payload, '1110', 'practice 8 payload');
assert.equal(practiceCode.totalBits, 5, 'practice 8 length');
assert.throws(() => encodeSixteenBits('0101'), RangeError, 'a short message is refused');
assert.throws(() => encodeSixteenBits('01010101010101012'), RangeError, 'a non-binary message is refused');
record('complexity accounts');

/* ------------------------------------------- the saved airfoil ridge model */

assert.equal(featureNames.length, 20, 'twenty polynomial terms');
assert.deepEqual(featureNames, airfoil.feature_names, 'the packet term order');
// The term order is reconstructed, not copied: originals then products i <= j.
const raw = [1000, 10, 0.1, 50, 0.02];
const terms = polynomialTerms(raw);
assert.equal(terms.length, 20);
vector(terms.slice(0, 5), raw, 'the five original terms');
let position = 5;
for (let i = 0; i < 5; i += 1) {
  for (let j = i; j < 5; j += 1) {
    close(terms[position], raw[i] * raw[j], `term ${featureNames[position]}`, 1e-12);
    position += 1;
  }
}
// And the names agree with that construction.
featureNames.slice(0, 5).forEach((name, index) => assert.equal(name, ['frequency_hz', 'attack_degrees', 'chord_m', 'speed_mps', 'displacement_m'][index]));
assert.equal(featureNames[5], 'frequency_hz^2');
assert.equal(featureNames[19], 'displacement_m^2');
assert.deepEqual(termsTouchedBy(0), [0, 5, 6, 7, 8, 9], 'changing frequency moves six terms');
assert.equal(termsTouchedBy(4).length, 6, 'so does changing displacement thickness');
assert.throws(() => termsTouchedBy(5), RangeError, 'there are only five raw measurements');
assert.throws(() => polynomialTerms([1, 2, 3]), RangeError, 'five measurements are required');
assert.throws(() => polynomialTerms([1e9, 10, 0.1, 50, 0.01]), RangeError, 'an out-of-range measurement is refused');
record('polynomial expansion');

const trace = modelPrediction(ridgeModel, inferenceFixture.rawFeatures);
close(trace.prediction, inferenceFixture.basePrediction, 'the recorded development-row prediction', 1e-9);
close(trace.prediction, airfoil.inference_fixture.base_prediction, "the packet's recorded prediction", 1e-9);
const changedTrace = modelPrediction(ridgeModel, inferenceFixture.changedFeatures);
close(changedTrace.prediction, inferenceFixture.changedPrediction, 'the recorded changed prediction', 1e-9);
close(trace.prediction - changedTrace.prediction, 0.836659121, 'the stated decrease', 1e-6);
close(trace.contributions.reduce((sum, value) => sum + value, trace.intercept), trace.prediction,
  'the contributions and the intercept add to the prediction', 1e-12);
// Changing one raw measurement moves exactly the six terms it appears in.
const touched = new Set(termsTouchedBy(0));
trace.terms.forEach((value, index) => {
  if (!touched.has(index)) close(changedTrace.terms[index], value, `term ${featureNames[index]} is untouched`, 1e-12);
  else assert.notEqual(changedTrace.terms[index], value, `term ${featureNames[index]} moved`);
});
assert.throws(() => modelPrediction({ ...ridgeModel, coefficients: [1] }, inferenceFixture.rawFeatures), RangeError,
  'a malformed saved model is refused');
record('saved model trace');

/* -------------------------------------------- the generated data module */

assert.equal(provenance.rows, 1503);
assert.equal(provenance.developmentRows + provenance.reservedRows, 1503, 'the split covers every row');
assert.equal(provenance.reservedPredictionsComputed, false, 'no reserved row is scored');
assert.equal(provenance.sha256, '74c75fd71783f1e6b71f8a622b993dc592897a97cd689c5090a07147a1b097b3');
assert.equal(provenance.license, 'CC BY 4.0');
assert.equal(provenance.foldTrainRows * 3 + provenance.foldValidationRows * 3, provenance.developmentRows * 3);
assert.equal(candidates.length, 18, 'three families times six strengths');
assert.deepEqual(strengths, [0.001, 0.01, 0.1, 1, 10, 100]);
assert.equal(elasticNetRatio, 0.5);
assert.equal(rawFeatureRanges.length, 5);
candidates.forEach(row => {
  const saved = airfoil.candidate_results.find(entry => entry.family === row[0] && entry.strength === row[1]);
  close(row[2], saved.mean_validation_mse, `${row[0]} at ${row[1]} mean MSE`, 1e-6);
  row[3].forEach((value, fold) => close(value, saved.fits[fold].validation_mse, `${row[0]} at ${row[1]} fold ${fold}`, 1e-6));
  assert.deepEqual(row[4], saved.fits.map(fit => fit.nonzero), `${row[0]} at ${row[1]} nonzero counts`);
  // A published mean really is the mean of the published folds.
  close(row[2], row[3].reduce((sum, value) => sum + value, 0) / 3, 'the mean is the mean of its folds', 1e-5);
  record('published candidate row');
});
close(baselineMeanMse, baselines.reduce((sum, row) => sum + row.meanMse, 0) / 3, 'the published mean baseline', 1e-8);
close(olsMeanMse, baselines.reduce((sum, row) => sum + row.olsMse, 0) / 3, 'the published OLS mean', 1e-8);
close(baselineMeanMse, 45.072768, 'the manuscript baseline', 1e-6);
close(olsMeanMse, 17.350268, 'the manuscript OLS score', 1e-6);
for (const family of ['ridge', 'lasso', 'elastic_net']) {
  const rows = candidates.filter(row => row[0] === family);
  const best = rows.reduce((left, right) => (right[2] < left[2] ? right : left));
  assert.equal(best[1], 0.001, `${family} selects the smallest strength`);
  assert.equal(selected.find(entry => entry.family === family).nonzero, 20, `${family}'s refit keeps all twenty terms`);
}
assert.deepEqual(candidates.find(row => row[0] === 'lasso' && row[1] === 0.1)[4], [9, 9, 9], 'lasso keeps nine per fold at 0.1');
assert.deepEqual(candidates.find(row => row[0] === 'lasso' && row[1] === 0.001)[4], [19, 20, 19], 'two folds keep nineteen');
for (const family of ['lasso', 'elastic_net']) {
  for (const strength of [10, 100]) {
    const row = candidates.find(entry => entry[0] === family && entry[1] === strength);
    assert.deepEqual(row[4], [0, 0, 0], `${family} at ${strength} keeps no slope`);
    close(row[2], baselineMeanMse, `${family} at ${strength} reproduces the baseline`, 1e-9);
  }
}
assert.equal(coefficientPaths.length, 18);
coefficientPaths.forEach(entry => {
  assert.equal(entry.folds.length, 3);
  entry.folds.forEach((fold, index) => {
    assert.equal(fold.length, 20, 'twenty coefficients per fold');
    const row = candidates.find(candidate => candidate[0] === entry.family && candidate[1] === entry.strength);
    assert.equal(fold.filter(value => value !== 0).length, row[4][index],
      `${entry.family} at ${entry.strength} fold ${index} nonzero count agrees with the published one`);
  });
  record('published coefficient path');
});
// Ridge never produces an exact zero here; lasso does at the large strengths.
assert(coefficientPaths.filter(entry => entry.family === 'ridge').every(entry => entry.folds.every(fold => fold.every(value => value !== 0))),
  'every ridge coefficient is nonzero in these fits');
record('published data module');

/* ------------------------------------------------ the displayed programs */

assert.equal(Object.keys(examples).length, 3, 'three displayed programs');
assert.equal(examples.coordinateFit.file, 'coordinate_regularization.py');
assert.equal(examples.airfoilComparison.file, 'airfoil_regularization.py');
assert.equal(examples.dropoutMasks.file, 'dropout_masks.py');
assert.ok(examples.airfoilComparison.expected.includes(`baseline mean MSE ${baselineMeanMse.toFixed(6)}`),
  'the printed baseline matches the data module');
assert.ok(examples.airfoilComparison.expected.includes(`OLS mean MSE ${olsMeanMse.toFixed(6)}`),
  'the printed OLS score matches the data module');
candidates.forEach(row => {
  const printed = `${row[0]} ${row[1]} ${Number(row[2].toFixed(6))} [${row[4].join(', ')}]`;
  assert.ok(examples.airfoilComparison.expected.includes(printed), `the program prints ${printed}`);
});
assert.ok(examples.coordinateFit.expected.startsWith('ridge [1.5 0.2] 0.0 1'), 'the ridge line');
assert.ok(examples.coordinateFit.expected.includes('lasso [2. 0.] 0.0 1'), 'the lasso line');
assert.ok(examples.coordinateFit.expected.includes('elastic_net [1.666667 0.'), 'the elastic-net line');
assert.ok(examples.dropoutMasks.expected.endsWith('1.0 2.5'), 'the dropout totals');
// The program's own printed branches, parsed and compared with the browser
// model's rather than string-matched through two languages' float formatting.
const printedBranches = examples.dropoutMasks.expected.split(/\r?\n/).slice(0, 4).map(line => {
  const [, first, second, rest] = line.match(/^\((\d), (\d)\) (.+)$/);
  const [probability, prediction, loss] = rest.split(' ').map(Number);
  return { mask: [Number(first), Number(second)], probability, prediction, loss };
});
printedBranches.forEach((printed, index) => {
  assert.deepEqual(printed.mask, dropout.branches[index].mask, 'the printed mask order');
  close(printed.probability, dropout.branches[index].probability, 'the printed branch probability', 1e-12);
  close(printed.prediction, dropout.branches[index].prediction, 'the printed branch prediction', 1e-12);
  close(printed.loss, dropout.branches[index].halfSquaredLoss, 'the printed branch half-loss', 1e-12);
  record('printed dropout branch');
});
assert.ok(examples.airfoilComparison.code.includes('airfoil-self-noise.dat'), 'the program reads the served file');
assert.ok(examples.airfoilComparison.code.includes('tab separated'), 'and states the separator where it reads it');
assert.ok(!examples.airfoilComparison.code.includes('reserved,') || examples.airfoilComparison.code.includes('development, reserved'),
  'the reserved rows are split off and never used');
record('displayed programs');

/* ------------------------------------------------------- shared limits */

assert.equal(limits.raw.length, 5, 'one control range per raw measurement');
limits.raw.forEach((range, index) => {
  assert(range.minimum <= rawFeatureRanges[index][0] && range.maximum >= rawFeatureRanges[index][1],
    `the control range for measurement ${index + 1} covers the observed column`);
});
assert(limits.keep.minimum > 0, 'a keep probability of zero is outside the controls');
record('control ranges');

/* --------------------------------------------------------------- record */

const sources = [
  'src/learn/data/regularization-models.js',
  'src/learn/data/regularization-data.js',
  'src/learn/data/regularization-examples.js',
  'src/learn/components/lesson-labs/RegularizationShared.jsx',
  'src/learn/components/lesson-labs/RegularizationLabs.jsx',
  'src/learn/components/lesson-labs/RegularizationFigures.jsx',
  'src/learn/components/lesson-labs/regularization-labs.css',
  'src/learn/data/topics/regularization-l1-l2-elastic-net-dropout.jsx',
  'src/learn/data/curriculum/blueprints/regularization-l1-l2-elastic-net-dropout.js',
  'public/learn-assets/regularization/airfoil-self-noise.dat',
];
const hash = file => crypto.createHash('sha256').update(fs.readFileSync(file)).digest('hex');
const evidence = {
  checkedAt: new Date().toISOString(),
  sourceHashes: Object.fromEntries(sources.filter(fs.existsSync).map(file => [file, hash(file)])),
  verifierHash: hash('scripts/verify-regularization-models.mjs'),
  counts,
  totalGroupedChecks: Object.values(counts).reduce((sum, value) => sum + value, 0),
  scope: 'Browser regularization models against the content phase\'s calculated-inputs.json (four-row and duplicate coordinate fits with their sweep counts, dropout masks, AIC/BIC, factor optima, smoothness solutions and early-stopping filters), independent identities (a scanned scalar optimum, independently recomputed KKT conditions and a local objective grid for eight designs, a scanned two-factor surface, a sampled penalty level set and a scanned constrained contact point, back-substitution into the denoising system, reconstructed polynomial term order), refused input on every entry point, and the generated airfoil module: split accounting, every published fold score and nonzero count, the published means against their own folds, the coefficient paths, the selected refits and the saved ridge model\'s traced prediction.',
  limitations: [
    'The airfoil candidates are precomputed native fits; the browser reproduces their recorded outcomes, not the fitting.',
    'Displayed program output is executed separately by scripts/verify-regularization-examples.py.',
    'The data module is regenerated and matched to the packet separately by scripts/verify-regularization-data.py.',
    'Rendering, interaction and independent review are separate steps.',
  ],
  passed: true,
};
fs.mkdirSync('docs/teaching/evidence', { recursive: true });
fs.writeFileSync('docs/teaching/evidence/regularization-models.json', JSON.stringify(evidence, null, 2) + '\n');
console.log(`PASS: ${evidence.totalGroupedChecks} grouped regularization model checks.`);
