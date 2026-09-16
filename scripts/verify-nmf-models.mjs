// Bounded independent checks of the NMF browser models against the content
// phase's native calculations, the manuscript's worked values and analytic
// identities, plus structural checks of the generated data and example modules.
// Run: node scripts/verify-nmf-models.mjs
import assert from 'node:assert/strict';
import fs from 'node:fs';
import crypto from 'node:crypto';
import {
  bilinearMidpoint, cellTerms, checkMatrix, coneCoordinates, contributionRows, crossedZeros, divergingScale,
  fixtures, frobeniusHalf, generalizedKL, gradientH, gradientW, greyLevels, halfSquaredLoss, hCellDirection,
  integerDeterminant, integerRank, itakuraSaito, limits, maskComparison, maskDirection, maskedReconstruction,
  maskedReport, meanSquaredError, mixtureProportions, multiply, nonsingularMinor, normalizePatterns, onLattice,
  reconstruct, residual, scaleLosses, simplexPosition, stationarityReport, supportPairs, sweep, sweepCost,
  sweepTrace, transpose, updateH, updateW,
} from '../src/learn/data/nmf-models.js';
import { NMF_DIGITS, NMF_RECORDED } from '../src/learn/data/nmf-data.js';

const packet = 'docs/teaching/drafts/non-negative-matrix-factorization-nmf';
const saved = JSON.parse(fs.readFileSync(`${packet}/calculated-inputs.json`, 'utf8'));
const examples = JSON.parse(fs.readFileSync('src/learn/data/nmf-examples.js', 'utf8')
  .replace(/^[\s\S]*?export const nmfExamples = /, '').replace(/;\s*$/, ''));

const counts = {};
const record = name => { counts[name] = (counts[name] ?? 0) + 1; };
const close = (actual, expected, label, tolerance = 1e-9) =>
  assert(Math.abs(actual - expected) <= tolerance * Math.max(1, Math.abs(expected)), `${label}: ${actual} versus ${expected}`);
const closeMatrix = (actual, expected, label, tolerance = 1e-9) => {
  assert.equal(actual.length, expected.length, `${label}: row count`);
  actual.forEach((row, index) => {
    assert.equal(row.length, expected[index].length, `${label}: row ${index} width`);
    row.forEach((value, column) => close(value, expected[index][column], `${label} (${index}, ${column})`, tolerance));
  });
};

// ---------------------------------------------------------- matrix arithmetic
closeMatrix(reconstruct(fixtures.W1, fixtures.H1), fixtures.X, 'the first factorization is exact', 0);
closeMatrix(reconstruct(fixtures.W2, fixtures.H2), fixtures.X, 'the second factorization is exact', 0);
closeMatrix(saved.ambiguity_products[0], fixtures.X, 'the recorded first product', 0);
closeMatrix(saved.ambiguity_products[1], fixtures.X, 'the recorded second product', 0);
// An independent route to the same product: column by column rather than by row.
const byColumn = fixtures.W1.map(row => fixtures.H1[0].map((_, column) =>
  row.reduce((sum, amount, component) => sum + amount * fixtures.H1[component][column], 0)));
closeMatrix(byColumn, fixtures.X, 'an independent product routine agrees', 0);
assert.deepEqual(transpose([[1, 2, 3], [4, 5, 6]]), [[1, 4], [2, 5], [3, 6]]);
closeMatrix(multiply([[1, 0], [0, 1]], fixtures.H1), fixtures.H1, 'the identity leaves H alone', 0);
record('matrix arithmetic');

// Validation refuses, rather than repairing.
assert.throws(() => checkMatrix([[1, 2], [3]]), RangeError, 'a ragged matrix is refused');
assert.throws(() => checkMatrix([[1, -2]]), RangeError, 'a negative entry is refused');
assert.throws(() => checkMatrix([[1, Number.NaN]]), RangeError, 'a non-finite entry is refused');
assert.throws(() => checkMatrix([]), RangeError, 'an empty matrix is refused');
assert.throws(() => multiply([[1, 2]], [[1, 2]]), RangeError, 'mismatched shapes are refused');
assert.throws(() => reconstruct([[1, 2]], [[1, 2, 3]]), RangeError, 'a component-count mismatch is refused');
assert.throws(() => halfSquaredLoss([[1, 2]], fixtures.W1, fixtures.H1), RangeError, 'a shape mismatch against X is refused');
assert.throws(() => cellTerms(fixtures.W1, fixtures.H1, 9, 0), RangeError, 'a missing row is refused');
assert.throws(() => onLattice(0.3, limits.amount, 'an amount'), RangeError, 'an off-lattice amount is refused');
assert.throws(() => onLattice(5, limits.amount, 'an amount'), RangeError, 'an out-of-range amount is refused');
assert.doesNotThrow(() => onLattice(2.25, limits.amount, 'an amount'));
assert.throws(() => updateH([[0, 0], [0, 0]], [[1], [1]], [[0, 0]]), RangeError, 'a zero denominator is refused rather than divided');
assert.throws(() => greyLevels([1], 0), RangeError, 'a zero display maximum is refused');
assert.throws(() => itakuraSaito(1, 0), RangeError, 'Itakura-Saito refuses a zero reconstruction');
record('validation refuses bad input');

// ------------------------------------------------------- one reconstructed cell
const cell = cellTerms(fixtures.W1, fixtures.H1, 0, 2);
assert.deepEqual(cell.terms.map(term => term.product), [2, 1]);
close(cell.total, 3, 'observation 1, feature 3');
assert.deepEqual(cellTerms(fixtures.W1, fixtures.H1, 0, 1).terms.map(term => term.product), [0, 1]);
const rows = contributionRows(fixtures.W1, fixtures.H1, 0);
assert.deepEqual(rows, [[2, 0, 2], [0, 1, 1]]);
assert.deepEqual(rows[0].map((value, index) => value + rows[1][index]), fixtures.X[0]);
assert(rows.every(row => row.every(value => value >= 0)), 'no contribution is negative, so nothing cancels');
record('one reconstructed cell');

// --------------------------------------------------------------- the residual
const signed = residual([fixtures.observed], [fixtures.approximated])[0];
assert.deepEqual(signed, [0.5, 0, 0.5]);
close(signed.reduce((sum, value) => sum + value * value, 0) / 2, 0.25, "the row's contribution to F");
close(meanSquaredError(fixtures.observed, fixtures.approximated), 0.5 / 3, 'mean squared residual over three features');
close(halfSquaredLoss([fixtures.observed], [[1]], [fixtures.approximated]), 0.25, 'the same value through the objective');
record('the signed residual');

// ------------------------------------------------------------------ the losses
for (const [x, y] of [[2, 4], [20, 22]]) {
  const recorded = saved.loss_comparison[String(x)];
  close(frobeniusHalf(x, y), recorded.frobenius_half, `half squared error at ${x}`, 1e-12);
  close(generalizedKL(x, y), recorded.kl, `generalized KL at ${x}`, 1e-12);
  close(itakuraSaito(x, y), recorded.is, `Itakura-Saito at ${x}`, 1e-12);
}
close(generalizedKL(2, 4), 0.613706, 'the printed KL value', 1e-6);
close(itakuraSaito(2, 4), 0.193147, 'the printed Itakura-Saito value', 1e-6);
close(generalizedKL(20, 22), 0.093796, 'the printed KL value at twenty', 1e-6);
close(itakuraSaito(20, 22), 0.004401, 'the printed Itakura-Saito value at twenty', 1e-6);
assert(frobeniusHalf(2, 4) === frobeniusHalf(20, 22), 'squared error cannot tell the two contexts apart');
assert(generalizedKL(2, 4) > generalizedKL(20, 22) && itakuraSaito(2, 4) > itakuraSaito(20, 22),
  'the other two losses charge the small-value overestimate more');
close(generalizedKL(3, 3), 0, 'KL is zero at a perfect fit', 1e-15);
close(itakuraSaito(3, 3), 0, 'Itakura-Saito is zero at a perfect fit', 1e-15);
close(generalizedKL(0, 2.5), 2.5, 'a zero observation contributes the reconstruction');
assert.equal(generalizedKL(2, 0), Infinity, 'a positive observation against zero is infinite');
for (const c of [2, 3, 7.5]) {
  const scaled = scaleLosses(2, 4, c);
  close(scaled.scaled.frobeniusHalf, c * c * scaled.base.frobeniusHalf, `squared error scales by c squared at ${c}`, 1e-12);
  close(scaled.scaled.kl, c * scaled.base.kl, `KL scales by c at ${c}`, 1e-12);
  close(scaled.scaled.itakuraSaito, scaled.base.itakuraSaito, `Itakura-Saito is gain-invariant at ${c}`, 1e-12);
}
// Practice 7 asks for exactly these three factors at a gain of three.
const gain = scaleLosses(2, 4, 3);
assert.deepEqual(gain.factors, { frobeniusHalf: 9, kl: 3, itakuraSaito: 1 });
record('the three losses and their scaling');

// -------------------------------------------------------------- one full sweep
const first = sweep(fixtures.X, fixtures.startW, fixtures.startH);
close(first.lossBefore, 17.06, 'the initial loss', 1e-12);
close(first.hPhase.numerator[0][0], 5.5, 'the H11 numerator');
close(first.hPhase.denominator[0][0], 2.65, 'the H11 denominator');
close(first.H[0][0], 2.075471698113208, 'the updated H11', 1e-12);
closeMatrix(first.H, saved.trace[1].H, 'the first H phase against the recording', 1e-9);
closeMatrix(first.W, saved.trace[1].W, 'the first W phase against the recording', 1e-9);
close(first.loss, saved.trace[1].loss, 'the loss after one full sweep', 1e-9);
close(first.loss, 0.0394474, 'the printed sweep loss', 1e-6);
assert(first.lossAfterH < first.lossBefore && first.loss < first.lossAfterH,
  'each phase separately reduces the objective in this fixture');
// The W row that fell while its pattern row rose: the product is what matters.
assert(first.W[0][0] < fixtures.startW[0][0] && first.H[0][0] > fixtures.startH[0][0],
  'an activation can decrease while its pattern entry increases');
// The denominator is the reconstruction term and the numerator the data term.
const gradient = gradientH(fixtures.X, fixtures.startW, fixtures.startH);
closeMatrix(gradient, first.hPhase.denominator.map((row, r) => row.map((value, c) => value - first.hPhase.numerator[r][c])),
  'the gradient is the denominator minus the numerator', 1e-12);
assert(gradient[0][0] < 0 && first.H[0][0] > fixtures.startH[0][0],
  'a negative gradient at a cell means the ratio increases it');
const wGradient = gradientW(fixtures.X, fixtures.startW, first.H);
closeMatrix(wGradient, first.wPhase.denominator.map((row, r) => row.map((value, c) => value - first.wPhase.numerator[r][c])),
  'the same identity for the W phase', 1e-12);
record('one full alternating sweep');

const trace = sweepTrace(fixtures.X, fixtures.startW, fixtures.startH, 40);
assert.equal(trace.length, 41);
for (let step = 0; step <= 40; step += 1) {
  close(trace[step].loss, saved.trace[step].loss, `the recorded loss at sweep ${step}`, 1e-8);
  closeMatrix(trace[step].H, saved.trace[step].H, `the recorded H at sweep ${step}`, 1e-8);
  closeMatrix(trace[step].W, saved.trace[step].W, `the recorded W at sweep ${step}`, 1e-8);
}
assert(trace.every((state, index) => index === 0 || state.loss <= trace[index - 1].loss + 1e-15),
  'the objective never increases along the trace');
assert.deepEqual([0, 1, 2, 10, 40].map(step => Number(trace[step].loss.toFixed(8))),
  [17.06, 0.03944744, 0.02823377, 0.00180608, 5e-8], 'the manuscript prints these five values');
closeMatrix(reconstruct(trace[40].W, trace[40].H), fixtures.X, 'forty sweeps approach the exact product', 1e-3);
assert(trace[40].H.some((row, r) => row.some((value, c) => Math.abs(value - fixtures.H1[r][c]) > 0.3)),
  'the fitted factors need not be the factors that built X');
assert.throws(() => sweepTrace(fixtures.X, fixtures.startW, fixtures.startH, 41), RangeError, 'the bounded history stops at forty');
record('the forty-sweep trace');

// The altered measurement the investigation uses as its contrast.
const altered = fixtures.X.map(row => row.slice());
altered[0][1] = 2;
const alteredStep = sweep(altered, fixtures.startW, fixtures.startH);
close(alteredStep.H[0][1], 0.489795918367347, 'H[0,1] after the altered step', 1e-9);
close(alteredStep.H[1][1], 2.2641509433962264, 'H[1,1] after the altered step', 1e-9);
close(alteredStep.loss, saved.altered_one_step.loss, 'the altered sweep loss', 1e-9);
close(alteredStep.loss, 0.224942552, 'the altered sweep loss to the recorded digits', 1e-8);
closeMatrix(alteredStep.H, saved.altered_one_step.H, 'the altered H', 1e-9);
closeMatrix(alteredStep.W, saved.altered_one_step.W, 'the altered W', 1e-9);
record('the altered-measurement contrast');

// Direction of one selected H cell: the contrast that keeps the prediction real.
assert.equal(hCellDirection(fixtures.X, fixtures.startW, fixtures.startH, 0, 0), 'grows');
const bigger = fixtures.startH.map(row => row.slice());
bigger[0][0] = 3;
assert.equal(hCellDirection(fixtures.X, fixtures.startW, bigger, 0, 0), 'shrinks');
close(updateH(fixtures.X, fixtures.startW, bigger).denominator[0][0], 7.15, 'the denominator when H11 starts at three');
close(updateH(fixtures.X, fixtures.startW, bigger).next[0][0], 30 / 13, 'H11 falls to thirty thirteenths', 1e-12);
// The exact-fit null: every relevant ratio is one and nothing moves.
const exact = sweep(fixtures.X, fixtures.W1, fixtures.H1);
closeMatrix(exact.H, fixtures.H1, 'the exact-fit null leaves H alone', 1e-12);
closeMatrix(exact.W, fixtures.W1, 'the exact-fit null leaves W alone', 1e-12);
close(exact.loss, 0, 'the exact-fit null keeps zero loss', 1e-18);
assert.equal(hCellDirection(fixtures.X, fixtures.W1, fixtures.H1, 0, 0), 'unchanged');
assert.equal(hCellDirection(fixtures.X, fixtures.W1, fixtures.H1, 0, 1), 'unchanged', 'a zero entry stays where it is');
assert.equal(exact.H[0][1], 0, 'and it is still exactly zero');
record('the update direction and the exact-fit null');

// ------------------------------------------------------------ zero locking
const lock = fixtures.zeroLock;
const lockGradient = gradientH(lock.X, lock.W, lock.H);
assert.deepEqual(lockGradient, [[-2, 0]], 'the locked coordinate has gradient minus two');
const lockReport = stationarityReport(lock.X, lock.W, lock.H);
assert.equal(lockReport.stationary, false);
assert.equal(lockReport.violations.length, 1);
assert.equal(lockReport.violations[0].kind, 'zero with a feasible descent direction');
close(halfSquaredLoss(lock.X, lock.W, [lock.nnls]), 0, 'the fixed-W least-squares solution fits exactly', 1e-18);
assert(halfSquaredLoss(lock.X, lock.W, lock.H) > 0, 'the locked state does not');
// Practice 8's boundary case: one violation of each kind.
const eight = fixtures.practice.eight;
assert.deepEqual(gradientH(eight.X, eight.W, eight.H), [[-3, 1]], "practice 8's gradient");
const eightReport = stationarityReport(eight.X, eight.W, eight.H);
assert.deepEqual(eightReport.violations.map(item => item.kind).sort(),
  ['positive with a nonzero gradient', 'zero with a feasible descent direction']);
close(halfSquaredLoss(eight.X, eight.W, [[3, 1]]), 0, "practice 8's fixed-W optimum", 1e-18);
// A genuinely stationary interior point reports no violation.
assert.equal(stationarityReport([[2, 1]], [[1]], [[2, 1]]).stationary, true);
record('zero locking and the boundary conditions');

// --------------------------------------------------------------- normalization
const normalized = normalizePatterns(fixtures.W1, fixtures.H1);
assert.deepEqual(normalized.sums, [2, 2]);
assert.deepEqual(normalized.H, [[0.5, 0, 0.5], [0, 0.5, 0.5]]);
assert.deepEqual(normalized.W[0], [4, 2]);
closeMatrix(reconstruct(normalized.W, normalized.H), fixtures.X, 'normalization preserves every reconstruction', 1e-12);
assert(normalized.H.every(row => Math.abs(row.reduce((sum, value) => sum + value, 0) - 1) < 1e-15), 'each pattern now sums to one');
const proportions = mixtureProportions(fixtures.W1, fixtures.H1, 0);
close(proportions.total, 6, 'the reconstructed total mass');
close(proportions.proportions[0], 2 / 3, 'the first mixture proportion', 1e-15);
close(proportions.proportions[1], 1 / 3, 'the second mixture proportion', 1e-15);
assert.notDeepEqual(fixtures.W1[0], proportions.proportions, 'the original activation vector was not a distribution');
// A zero pattern is reported rather than divided by zero.
const withZero = normalizePatterns([[1, 1]], [[1, 1], [0, 0]]);
assert.deepEqual(withZero.zeroPatterns, [1]);
assert.deepEqual(withZero.H[1], [0, 0]);
// Practice 1, exactly.
const one = fixtures.practice.one;
assert.deepEqual(reconstruct([one.w], one.H)[0], [2, 3, 7]);
const practiceNormalized = normalizePatterns([one.w], one.H);
assert.deepEqual(practiceNormalized.sums, [3, 3]);
assert.deepEqual(practiceNormalized.W[0], [3, 9]);
close(mixtureProportions([one.w], one.H, 0).total, 12, "practice 1's reconstructed total");
close(mixtureProportions([one.w], one.H, 0).proportions[0], 0.25, "practice 1's first proportion", 1e-15);
// Scale invariance: doubling a pattern row and halving its activation column.
const doubled = { H: fixtures.H1.map((row, index) => (index === 0 ? row.map(value => value * 2) : row.slice())),
  W: fixtures.W1.map(row => [row[0] / 2, row[1]]) };
closeMatrix(reconstruct(doubled.W, doubled.H), fixtures.X, 'scale invariance preserves the product', 1e-12);
record('normalization and scale invariance');

// Practice 2: one multiplicative step that happens to reach the NNLS optimum.
const two = fixtures.practice.two;
const twoStep = updateH(two.X, two.W, two.H);
assert.deepEqual(twoStep.numerator[0], [10, 7]);
assert.deepEqual(twoStep.denominator[0], [5, 5]);
assert.deepEqual(twoStep.next[0], [2, 1.4]);
close(halfSquaredLoss(two.X, two.W, twoStep.next), 0.1, "practice 2's half-squared loss", 1e-15);
assert.deepEqual(gradientH(two.X, two.W, twoStep.next)[0].map(value => Number(value.toFixed(12))), [0, 0],
  'this one-dimensional case does land on the fixed-W optimum');
record('practice 2');

// ------------------------------------------------------------- joint convexity
const counterexample = bilinearMidpoint(1, [1, 1], [2, 0.5]);
close(counterexample.first.loss, 0, 'the first endpoint has zero loss', 1e-18);
close(counterexample.second.loss, 0, 'the second endpoint has zero loss', 1e-18);
close(counterexample.midpoint.product, 1.125, 'the midpoint product');
close(counterexample.midpoint.loss, 0.0078125, 'the midpoint loss', 1e-15);
assert.equal(counterexample.convexityRefuted, true);
const six = fixtures.practice.six;
const practiceSix = bilinearMidpoint(six.target, six.endpoints[0], six.endpoints[1]);
close(practiceSix.midpoint.product, 3.125, "practice 6's midpoint product");
close(practiceSix.midpoint.loss, 0.6328125, "practice 6's midpoint loss", 1e-15);
assert.equal(practiceSix.convexityRefuted, true);
// The separate subproblems really are convex: W^T W is positive semidefinite.
const gram = multiply(transpose(fixtures.startW), fixtures.startW);
for (const v of [[1, 0], [0, 1], [1, 1], [1, -1], [2, -3], [-1, 0.5]]) {
  const quadratic = v.reduce((sum, value, index) => sum + value * gram[index].reduce((inner, entry, column) => inner + entry * v[column], 0), 0);
  const image = fixtures.startW.map(row => row.reduce((sum, value, index) => sum + value * v[index], 0));
  close(quadratic, image.reduce((sum, value) => sum + value * value, 0), `v^T W^T W v equals the squared image of ${v}`, 1e-12);
  assert(quadratic >= -1e-12, 'and it is nonnegative');
}
record('joint nonconvexity and separate convexity');

// ----------------------------------------------------------- the cone geometry
for (const point of fixtures.X.map(row => [row[0], row[1]])) {
  const inFirst = coneCoordinates([[1, 0], [0, 1]], point);
  const inSecond = coneCoordinates([[1.25, 0.25], [0.25, 1.25]], point);
  assert(inFirst.inside && inSecond.inside, `${point} lies inside both cones`);
}
closeMatrix([coneCoordinates([[1.25, 0.25], [0.25, 1.25]], [2, 1]).coefficients], [[1.5, 0.5]],
  'the second cone reproduces the recorded activations', 1e-12);
closeMatrix([coneCoordinates([[1, 0], [0, 1]], [2, 1]).coefficients], [[2, 1]], 'and the first reproduces the first', 1e-12);
assert.equal(coneCoordinates([[1.25, 0.25], [0.25, 1.25]], [1, -0.5]).inside, false, 'a point outside is reported outside');
assert.throws(() => coneCoordinates([[1, 1], [2, 2]], [1, 1]), RangeError, 'parallel rays span no cone');
assert(fixtures.X.every(row => Math.abs(row[2] - row[0] - row[1]) < 1e-15),
  'feature three is the sum of the first two, so the plane view keeps the containment');
record('the two cones');

// ------------------------------------------------- nonnegative rank and support
assert.equal(integerRank(fixtures.support), 3, 'the support matrix has ordinary rank three');
assert.equal(integerRank(fixtures.support.slice(0, 3)), 3, 'its first three rows are independent');
assert.deepEqual(fixtures.support[0].map((value, index) => value + fixtures.support[2][index]),
  fixtures.support[1].map((value, index) => value + fixtures.support[3][index]), 'rows one and three sum to rows two and four');
assert.equal(integerRank([[1, 0], [0, 1]]), 2);
assert.equal(integerRank([[1, 2], [2, 4]]), 1);
assert.equal(integerRank([[0, 0], [0, 0]]), 0);
const pairs = supportPairs(fixtures.support, fixtures.supportMarks);
assert.equal(pairs.length, 6);
assert(pairs.every(pair => pair.crossed.length > 0), 'every pair of marked cells crosses a zero');
assert.deepEqual(crossedZeros(fixtures.support, [0, 2], [1, 3]), [[0, 3], [1, 2]].filter(([r, c]) => fixtures.support[r][c] === 0));
assert.throws(() => crossedZeros(fixtures.support, [0, 0], [1, 3]), RangeError, 'a zero cell cannot anchor a positive rectangle');
closeMatrix(multiply([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], fixtures.support), fixtures.support,
  'four rank-one contributions suffice', 0);
assert.deepEqual(NMF_RECORDED.supportPairs.map(pair => pair.crossed.length > 0), [true, true, true, true, true, true]);
// The ordinary rank is shown with a nonsingular minor, so the determinant that
// demonstrates it has to be exact too.
assert.equal(integerDeterminant([[2]]), 2);
assert.equal(integerDeterminant([[1, 2], [3, 4]]), -2);
assert.equal(integerDeterminant([[2, 0, 0], [0, 3, 0], [0, 0, 5]]), 30);
assert.equal(integerDeterminant([[1, 2, 3], [4, 5, 6], [7, 8, 10]]), -3);
// Cross-check against an independent cofactor expansion along the first column.
const byColumnDeterminant = matrix => (matrix.length === 1 ? matrix[0][0]
  : matrix.reduce((total, row, index) => total + (index % 2 === 0 ? 1 : -1) * row[0]
    * byColumnDeterminant(matrix.filter((_, other) => other !== index).map(rest => rest.slice(1))), 0));
for (const probe of [[[1, 2], [3, 4]], [[1, 2, 3], [4, 5, 6], [7, 8, 10]], fixtures.support]) {
  assert.equal(integerDeterminant(probe), byColumnDeterminant(probe), 'two expansions agree');
}
assert.equal(integerDeterminant([[1, 2], [2, 4]]), 0);
assert.equal(integerDeterminant(fixtures.support), 0, 'S is singular, as its row dependency requires');
assert.throws(() => integerDeterminant([[1, 2, 3], [4, 5, 6]]), RangeError, 'a determinant needs a square matrix');
assert.throws(() => integerDeterminant([[1.5, 0], [0, 1]]), RangeError, 'this exact routine refuses non-integers');
const shownMinor = nonsingularMinor(fixtures.support, 3);
assert.equal(shownMinor.minor.length, 3);
assert.notEqual(shownMinor.determinant, 0);
assert.equal(integerRank(shownMinor.minor), 3, 'the displayed minor really has rank three');
assert.deepEqual(shownMinor.minor,
  shownMinor.rows.map(row => shownMinor.columns.map(column => fixtures.support[row][column])),
  'the displayed minor is taken from S itself');
assert.throws(() => nonsingularMinor([[1, 2], [2, 4]], 2), RangeError, 'a singular matrix has no nonsingular minor of full size');
// Row 1 + row 3 equals row 2 + row 4, which is what caps the rank at three.
assert.deepEqual(fixtures.support[0].map((value, index) => value + fixtures.support[2][index]), [1, 1, 1, 1]);
assert.deepEqual(fixtures.support[1].map((value, index) => value + fixtures.support[3][index]), [1, 1, 1, 1]);
record('nonnegative rank');

// ------------------------------------------------------------ separability
const anchors = simplexPosition(fixtures.anchors.rows, fixtures.anchors.anchorIndices);
assert(anchors.every(item => Math.abs(item.row[0] + item.row[1] - 1) < 1e-15), 'normalized rows sum to one');
assert(anchors.every(item => item.inside), 'every observation lies in the hull of the two endpoints');
closeMatrix([anchors[2].weights], [[0.25, 0.75]], 'the third row is a stated mixture', 1e-12);
closeMatrix([anchors[3].weights], [[0.6, 0.4]], 'and so is the fourth', 1e-12);
record('the anchor geometry');

// ------------------------------------------------ words, spectra and cost counts
assert.deepEqual(reconstruct([fixtures.words.w], fixtures.words.H)[0], [6, 4, 1, 4]);
closeMatrix(reconstruct([fixtures.spectrum.amounts], fixtures.spectrum.patterns), [[0.62, 0.39, 0.19]],
  'the three-band mixture', 1e-15);
const cost = sweepCost(180, 64, 8, 9000);
assert.equal(cost.dense, 180 * 64 * 8 + (180 + 64) * 64);
assert.equal(cost.sparse, 9000 * 8 + (180 + 64) * 64);
assert.equal(cost.factorStorage, 8 * (180 + 64));
assert(cost.sparse < cost.dense, 'sparse data products cost less while the factor terms remain');
assert.throws(() => sweepCost(0, 64, 8), RangeError);
record('the transfer fixtures and cost counts');

// -------------------------------------------------------- the real image lab
const { dictionary, activations, test, runs, baselines, splitSizes } = NMF_DIGITS;
assert.equal(dictionary.length, 8);
assert.equal(dictionary[0].length, 64);
assert.equal(activations.length, 60);
assert.equal(test.length, 60);
assert.deepEqual(splitSizes, { train: 180, validation: 60, test: 60 });
assert.equal(NMF_DIGITS.datasetSha256, 'd93f963c4b2610eb07122a312eec3ddceac18a031477370d71e33835eced728e');
assert.equal(test[0].sourceRow, 242, 'the first reserved image is source row 242');
assert(test.every(row => row.pixels.length === 64 && row.pixels.every(value => Number.isInteger(value) && value >= 0 && value <= 16)),
  'every published image is 64 integer block counts from 0 to 16');
assert(dictionary.every(row => row.every(value => value >= 0)) && activations.every(row => row.every(value => value >= 0)),
  'the published factors are nonnegative');
closeMatrix(dictionary, saved.visual.H, 'the published dictionary is the recorded fit', 0);
closeMatrix(activations, saved.visual.W_test, 'the published activations are the recorded fit', 0);
closeMatrix(reconstruct(activations, dictionary), saved.visual.reconstructed_test,
  'the browser rebuilds the recorded reconstruction from the two published factors', 1e-9);
const observed = test[0].pixels.map(value => value / NMF_DIGITS.scale);
const report = maskedReport(observed, activations[0], dictionary, activations[0].map(() => true));
close(report.fullMse, 0.021503642603, 'the row MSE for source row 242', 1e-9);
close(report.fullMse, NMF_RECORDED.removals.baseMse, 'and it matches the recorded probe', 1e-12);
for (let component = 0; component < 8; component += 1) {
  const mask = activations[0].map((_, index) => index !== component);
  const without = maskedReport(observed, activations[0], dictionary, mask);
  close(without.maskedMse, NMF_RECORDED.removals.byComponent[component], `removing component ${component + 1}`, 1e-9);
  assert(without.maskedMse >= report.fullMse - 1e-8, 'no single removal improves the total row error at this row optimum');
  const difference = without.kept.map((value, index) => report.full[index] - value);
  const contribution = dictionary[component].map(value => activations[0][component] * value);
  closeMatrix([difference], [contribution], `the change equals component ${component + 1}'s contribution image`, 1e-12);
  assert(contribution.every(value => value >= 0), 'every removed pixel contribution is nonnegative');
}
close(NMF_RECORDED.removals.byComponent[1], 0.068409087403, 'the recorded removal of component 2', 1e-9);
close(NMF_RECORDED.removals.byComponent[0], 0.024686350768, 'the recorded removal of component 1', 1e-9);
assert.equal(activations[0][7], 0, 'component 8 has exactly zero activation on this image');
close(NMF_RECORDED.removals.byComponent[7], report.fullMse, 'so removing it changes nothing', 1e-15);
const emptyMask = maskedReport(observed, activations[0], dictionary, activations[0].map(() => false));
close(emptyMask.maskedMse, observed.reduce((sum, value) => sum + value * value, 0) / 64,
  'the all-zero mask leaves the mean squared observation', 1e-15);
assert(emptyMask.kept.every(value => value === 0), 'and reconstructs nothing at all');
const direction = maskDirection(observed, activations[0], dictionary,
  activations[0].map(() => true), activations[0].map((_, index) => index !== 1));
assert.equal(direction.direction, 'rises');
close(direction.after, 0.068409087403, 'the graded after value', 1e-9);
assert.equal(maskDirection(observed, activations[0], dictionary,
  activations[0].map(() => true), activations[0].map((_, index) => index !== 7)).direction, 'unchanged',
  'the zero-activation component is the exact null');
const removedSecond = maskedReport(observed, activations[0], dictionary, activations[0].map((_, index) => index !== 1));
assert(removedSecond.improvedPixels > 0 && removedSecond.worsenedPixels > 0,
  'individual pixels move both ways even when the total error rises');
assert.equal(removedSecond.improvedPixels + removedSecond.worsenedPixels + removedSecond.unchangedPixels, 64);
close(report.contributionTotals[1], activations[0][1] * dictionary[1].reduce((sum, value) => sum + value, 0),
  'a contribution total is the coefficient times its pattern mass', 1e-15);
// The transfer task names one reserved image on which the two orderings differ,
// so the numbers it names have to be the ones the data produces.
const patternMass = dictionary.map(row => row.reduce((sum, value) => sum + value, 0));
const topBy = (row, weights) => weights.indexOf(Math.max(...weights)) + 1;
const topCoefficient = index => topBy(index, activations[index]);
const topContribution = index => topBy(index, activations[index].map((value, r) => value * patternMass[r]));
assert.equal(topCoefficient(0), topContribution(0), 'the two orderings agree on the image the lab opens with');
assert.equal(test[1].sourceRow, 256, 'the named transfer image');
assert.equal(topCoefficient(1), 7, 'its largest raw coefficient is component 7');
assert.equal(topContribution(1), 8, 'its largest contribution total is component 8');
assert.equal([...activations.keys()].filter(index => topCoefficient(index) !== topContribution(index)).length, 16,
  'sixteen of the sixty reserved images disagree about the top component');
record('the held-out image contributions');

// The display contract: nothing is clipped and the scale is stated.
const reconstructed = reconstruct(activations, dictionary);
const peak = Math.max(...reconstructed.flat());
assert(peak > 1, 'the reserved reconstruction exceeds one somewhere');
close(Math.max(...reconstructed[0]), 1.017674616234679, 'the first reserved image peaks above one', 1e-9);
const levels = greyLevels(reconstructed[0], 1);
assert(levels.some(item => item.above), 'a display maximum of one would be reported as exceeded, not silently clipped');
assert(greyLevels(reconstructed[0], Math.max(1, Math.max(...reconstructed[0]))).every(item => !item.above),
  'the stated rule, zero to the larger of one and the current maximum, covers every value');
const residualScale = divergingScale(report.fullResidual);
assert(residualScale.extent > 0 && Math.abs(residualScale.level(0)) < 1e-15, 'zero residual sits at the centre of its own scale');
close(residualScale.level(residualScale.extent), 1, 'and the extreme sits at the end', 1e-15);
record('the image display scales');

// ------------------------------------------------------------- candidate runs
assert.equal(runs.length, 8);
runs.forEach(run => {
  const recorded = saved.runs.find(item => item.k === run.k && item.seed === run.seed);
  close(run.trainMse, recorded.train_mse, `training MSE for k=${run.k}, seed=${run.seed}`, 1e-12);
  close(run.validationMse, recorded.validation_mse, `validation MSE for k=${run.k}, seed=${run.seed}`, 1e-12);
  assert(run.iterations < 2000, 'no recorded fit hit its iteration cap');
});
close(baselines.meanTestMse, saved.mean_test_mse, 'the training-mean baseline', 1e-12);
close(baselines.pcaTestMse, saved.pca_test_mse, 'the PCA baseline', 1e-12);
close(baselines.nmfTestMse, saved.visual.test_mse, 'the NMF reserved score', 1e-12);
assert(baselines.pcaTestMse < baselines.nmfTestMse, 'PCA wins the reserved reconstruction comparison, and the page says so');
const best = runs.reduce((low, run) => (run.validationMse < low.validationMse ? run : low));
assert.equal(best.k, 16);
assert.equal(best.seed, 7, 'the best inspected validation candidate is k=16 with seed 7');
assert(Math.abs(runs[0].validationMse - runs[1].validationMse) < 1e-6, 'the two seeds nearly agree at one component');
assert(Math.abs(runs[6].validationMse - runs[7].validationMse) > 1e-3, 'and disagree visibly at sixteen');
record('the recorded candidate fits');

// ----------------------------------------------- the displayed programs agree
assert.equal(Object.keys(examples).length, 3);
assert(examples.multiplicativeStep.expected.split('\n').map(line => line.split(' ')[1])
  .every((value, index) => Math.abs(Number(value) - trace[[0, 1, 2, 10, 40][index]].loss) < 1e-7),
  'the displayed trace agrees with the browser model');
runs.forEach(run => assert(
  examples.digitDictionary.expected.includes(`${run.k} ${run.seed} ${Number(run.trainMse.toFixed(6))} ${Number(run.validationMse.toFixed(6))}`),
  `the displayed table shows k=${run.k}, seed=${run.seed}`));
assert(examples.digitDictionary.expected.includes(`mean ${Number(baselines.meanTestMse.toFixed(6))}`));
assert(examples.digitDictionary.expected.includes(`PCA8 ${Number(baselines.pcaTestMse.toFixed(6))}`));
assert(examples.digitDictionary.expected.includes(`NMF8 ${Number(baselines.nmfTestMse.toFixed(6))}`));
assert(examples.digitDictionary.expected.includes('first held-out source row 242'));
assert(examples.digitDictionary.code.includes("np.loadtxt('digits-300.csv'"), 'the program reads the served CSV by name');
assert(examples.wordPatterns.expected.includes("['goal' 'orbit' 'rocket' 'team']"), 'the recorded vocabulary order');
assert(!examples.multiplicativeStep.code.includes('1e-'), 'no epsilon was added to the displayed update');
record('the displayed programs');

// --------------------------------------------------------------------- record
const sources = [
  'src/learn/data/nmf-models.js',
  'src/learn/data/nmf-data.js',
  'src/learn/data/nmf-examples.js',
  'src/learn/components/lesson-labs/NmfShared.jsx',
  'src/learn/components/lesson-labs/NmfLabs.jsx',
  'src/learn/components/lesson-labs/NmfFigures.jsx',
  'src/learn/components/lesson-labs/nmf-labs.css',
  'src/learn/data/topics/non-negative-matrix-factorization-nmf.jsx',
  'src/learn/data/curriculum/blueprints/non-negative-matrix-factorization-nmf.js',
  'public/learn-assets/nmf/digits-300.csv',
  'public/learn-assets/nmf/data-provenance.md',
];
const hash = file => crypto.createHash('sha256').update(fs.readFileSync(file)).digest('hex');
const evidence = {
  checkedAt: new Date().toISOString(),
  sourceHashes: Object.fromEntries(sources.filter(fs.existsSync).map(file => [file, hash(file)])),
  verifierHash: hash('scripts/verify-nmf-models.mjs'),
  counts,
  totalGroupedChecks: Object.values(counts).reduce((sum, value) => sum + value, 0),
  scope: 'Browser NMF models against the content-phase native calculations (both exact factorizations, all 41 multiplicative states, the altered-input step, the loss comparison, every candidate fit, the reserved factors and all eight component removals from source row 242), independent identities (an alternative product routine, the gradient as denominator minus numerator, positive semidefiniteness of the Gram matrix, exact integer rank by Bareiss elimination, cone coordinates recovering the recorded activations, loss scaling factors, the contribution-equals-difference identity for every mask), refusal behaviour on ragged, negative, non-finite, off-lattice and zero-denominator input, and structural checks of the generated data and example modules.',
  limitations: [
    'The digit candidates, dictionary and activations are precomputed native fits; the browser reproduces their recorded outcomes and rebuilds reconstructions from them, but does not fit.',
    'Displayed program output is executed separately by scripts/verify-nmf-examples.py.',
    'Rendering, interaction, accessibility and independent review are separate steps.',
  ],
  passed: true,
};
fs.mkdirSync('docs/teaching/evidence', { recursive: true });
fs.writeFileSync('docs/teaching/evidence/nmf-models.json', JSON.stringify(evidence, null, 2) + '\n');
console.log(`PASS: ${evidence.totalGroupedChecks} grouped NMF model checks.`);
