// Bounded independent checks of the feature-selection browser models against
// the content phase's recorded calculations, the manuscript's worked values and
// analytic identities, plus structural checks of the generated data module, the
// executed example module and the lesson source.
//
// Where a value can be reached two ways, it is reached two ways here: mutual
// information by an entropy difference and by a divergence sum; Shapley values
// by the weighted formula, by averaging every arrival order, and by a third
// independent enumeration written inside this file; the saved tree's inference
// against the recorded native predictions for every inspection and background
// row.
//
// Run: node scripts/verify-selection-models.mjs
import assertStrict from 'node:assert/strict';

// Count every assertion that actually runs. `record()` below counts GROUPS, many
// of them inside loops, which is a coverage label rather than an assertion count;
// this is the honest number and both are reported.
let assertions = 0;
const counted = fn => (...args) => { assertions += 1; return fn(...args); };
const assert = Object.assign(counted(assertStrict), Object.fromEntries(
  ['ok', 'equal', 'strictEqual', 'notEqual', 'deepEqual', 'deepStrictEqual', 'notDeepEqual', 'match', 'throws']
    .map(name => [name, counted(assertStrict[name].bind(assertStrict))])));
import fs from 'node:fs';
import crypto from 'node:crypto';
import {
  barScale, bits, checkDonor, checkRange, coalitionGame, cumulativeAttributionFraction, dependentGames,
  duplicateColumns, entropy, explainPolynomial, explainTreeInstance, extrapolationFlags,
  familywiseProbability, forwardSelection, groupedUnanimity, hybridRows, impurityImportance,
  drawnThreshold, informationFromCounts, jointInformation, limits, linearPredictions, maskLabel, maskMatrix,
  permutationExperiment, permutationSuite, polynomialModel, probabilityMosaic, searchFitCounts,
  shapleyValues, sigmoid, sizePoints, subsetWorld, thresholdDisagreements, toFloat32, treeClass, treeDecision,
  treeProbability, unanimityGame, waterfall, xorWorld,
} from '../src/learn/data/selection-models.js';
import {
  alternativeReference, backgroundIds, backgroundRows, candidates, changedInference, classLabels,
  developmentIds, explainedCases, featureLabels, featureNames, fittingIds, foldMemberships,
  fourFieldModel, inspectionClasses, inspectionIds, inspectionPredictions, inspectionRows,
  majorityBaseline, permutationRecords, provenance, reservedIds, selectedModel, split,
} from '../src/learn/data/selection-data.js';
import { selectionExamples } from '../src/learn/data/selection-examples.js';

const packetDir = 'docs/teaching/drafts/feature-selection-importance-shap-permutation-mutual-info';
const recorded = JSON.parse(fs.readFileSync(`${packetDir}/calculated-inputs.json`, 'utf8'));
const constructed = recorded.constructed;
const wine = recorded.wine;
const lessonSource = fs.readFileSync('src/learn/data/topics/feature-selection-importance-shap-permutation-mutual-info.jsx', 'utf8');
const nativeEvidence = JSON.parse(fs.readFileSync('docs/teaching/evidence/selection-native.json', 'utf8'));

const counts = {};
const record = name => { counts[name] = (counts[name] ?? 0) + 1; };
const close = (actual, expected, label, tolerance = 1e-12) =>
  assert(Math.abs(actual - expected) <= tolerance * Math.max(1, Math.abs(expected)),
    `${label}: ${actual} versus ${expected}`);
const vector = (actual, expected, label, tolerance = 1e-12) => {
  assert.equal(actual.length, expected.length, `${label}: length ${actual.length} versus ${expected.length}`);
  actual.forEach((value, index) => close(value, expected[index], `${label}[${index}]`, tolerance));
};
const refuses = (run, label) => {
  assert.throws(run, RangeError, label);
  record('refused input');
};

/* ======================================================= §2 information */

// The limiting convention: a zero probability contributes zero, and no literal
// logarithm of zero is evaluated.
assert.equal(bits(0), 0, 'a zero probability contributes exactly zero');
close(bits(0.5), 0.5, 'one half contributes a half bit');
close(entropy([0.5, 0.5]), 1, 'a fair binary variable has one bit');
close(entropy([1, 0]), 0, 'a certain variable has zero bits');
close(entropy([0.25, 0.75]), 0.8112781244591328, 'the 1/4, 3/4 entropy');
refuses(() => bits(-0.1), 'a negative probability is refused');
refuses(() => bits(Number.NaN), 'a non-finite probability is refused');
record('entropy conventions');

// Sequential audit: determinant one means dependence despite cancellation of
// the separately rounded entropy terms. Independent 70-digit mpmath oracle.
const nearIndependent = informationFromCounts([[9999, 9998], [10000, 9999]]);
const nearInformation = 4.510225845067146494022681575455e-18;
assert.equal(nearIndependent.independentExactly, false, 'integer count products are not equal');
assert(Math.abs(nearIndependent.mutualInformation / nearInformation - 1) < 1e-12,
  'stable positive MI agrees relatively with the independent high-precision oracle');
assert.equal(informationFromCounts([[9999, 9999], [9999, 9999]]).mutualInformation, 0, 'true independence remains exact zero');
const tinyDonor = permutationExperiment({ rows: [[-.0001, -.0001], [-.0001, -.0001], [.0001, .0001], [.0001, .0001]],
  target: [0, 0, 0, 0], coefficients: [.0001, -.0001], donor: [2, 3, 0, 1], columns: [0] });
assert(Math.abs(tinyDonor.increase / 4e-16 - 1) < 1e-12, 'zero baseline and residual ±2e-8 give positive MSE4e-16');
const cancelledGame = explainPolynomial({ instance: [2, 2], background: [[0, 0]], gamma: -1 });
assert.deepEqual(cancelledGame.phi, [0, 0], 'different instance can have cancelling, zero allocations');
assert.equal(explainPolynomial({ instance: [0, 3], background: [[0, 0]], gamma: 1 }).orderIndependent, true,
  'a nonzero interaction coefficient need not create order dependence in a particular game');
record('sequential near-null and cancellation regressions');

// The manuscript's worked table, against the packet and against the prose.
const base = informationFromCounts([[3, 1], [1, 3]]);
close(base.targetEntropy, 1, 'H(Y) for the balanced target');
close(base.conditionalEntropy, 0.8112781244591328, 'H(Y | X) for the 3/1/1/3 table');
close(base.mutualInformation, 0.18872187554086717, 'I(X; Y) for the 3/1/1/3 table');
close(base.mutualInformation, constructed.mi.noisy_copy_bits, 'against the packet record');
assert(base.agrees, 'the entropy difference and the divergence sum agree');
close(base.mutualInformationDirect, base.mutualInformation, 'the two routes to MI', 1e-12);
// The two rows have identical entropy despite opposite dominant labels.
close(base.conditionals[0].entropy, base.conditionals[1].entropy, 'both rows leave the same uncertainty');
assert.notDeepEqual(base.conditionals[0].distribution, base.conditionals[1].distribution,
  'and they nevertheless prefer opposite labels');
record('the worked count table');

// The two declared contrasts and the practice table.
close(informationFromCounts([[4, 0], [0, 4]]).mutualInformation, 1, 'a perfect copy reveals one bit');
close(informationFromCounts([[4, 0], [0, 4]]).conditionalEntropy, 0, 'and leaves no conditional uncertainty');
assert.equal(informationFromCounts([[2, 2], [2, 2]]).mutualInformation, 0, 'an independent table reveals exactly zero');
close(informationFromCounts([[2, 2], [2, 2]]).mutualInformation, constructed.mi.independent_bits, 'against the packet');
const practice = informationFromCounts([[2, 0], [0, 6]]);
close(practice.targetEntropy, 0.8112781244591328, 'practice 1 H(Y)');
close(practice.conditionalEntropy, 0, 'practice 1 H(Y | X)');
close(practice.mutualInformation, 0.8112781244591328, 'practice 1 I(X; Y)');
close(practice.mutualInformation, constructed.practice.information_bits, 'against the packet record');
close(practice.mutualInformation, practice.targetEntropy, 'a perfect reveal cannot exceed the original uncertainty');
record('count-table contrasts and the practice fixture');

// Scaling every count is a null for the empirical quantities and only for those.
for (const factor of [2, 3, 10, 1250]) {
  const scaled = informationFromCounts([[3 * factor, factor], [factor, 3 * factor]]);
  close(scaled.mutualInformation, base.mutualInformation, `MI is unchanged at scale ${factor}`, 1e-15);
  close(scaled.targetEntropy, base.targetEntropy, `H(Y) is unchanged at scale ${factor}`, 1e-15);
  assert.equal(scaled.total, base.total * factor, 'while the sample size is not unchanged');
  record('count scaling null');
}

// A degenerate target column is answered, not refused.
const degenerate = informationFromCounts([[4, 0], [4, 0]]);
assert.equal(degenerate.targetEntropy, 0, 'a target that never varies has zero entropy');
assert.equal(degenerate.mutualInformation, 0, 'and nothing can be removed from it');
// An empty input row contributes zero rather than needing a conditional distribution.
const empty = informationFromCounts([[3, 1], [0, 0]]);
assert.equal(empty.conditionals[1].occupied, false, 'the empty row is reported as unoccupied');
assert.equal(empty.conditionals[1].weight, 0, 'and carries no weight');
assert(empty.agrees, 'and both MI routes still agree');
record('degenerate and empty rows');

// An independent identity check over many tables: the two routes must agree and
// the result must be nonnegative, whatever the shape.
let random = 42;
const nextInt = bound => { random = (random * 1103515245 + 12345) % 2147483648; return random % bound; };
for (let trial = 0; trial < 200; trial += 1) {
  const height = 2 + nextInt(3);
  const width = 2 + nextInt(3);
  const table = Array.from({ length: height }, () => Array.from({ length: width }, () => nextInt(9)));
  if (table.flat().reduce((sum, value) => sum + value, 0) === 0) continue;
  const info = informationFromCounts(table);
  assert(info.agrees, `the two MI routes agree on trial ${trial}`);
  assert(info.mutualInformation >= -1e-12, `MI is nonnegative on trial ${trial}`);
  assert(info.mutualInformation <= info.targetEntropy + 1e-12, `MI cannot exceed H(Y) on trial ${trial}`);
  // Symmetry: transposing the table swaps the roles and leaves MI alone.
  const transposed = Array.from({ length: width }, (_, column) => table.map(row => row[column]));
  close(informationFromCounts(transposed).mutualInformation, info.mutualInformation,
    `MI is symmetric on trial ${trial}`, 1e-9);
}
record('randomised MI identities');

refuses(() => informationFromCounts([[-1, 1], [1, 1]]), 'a negative count is refused');
refuses(() => informationFromCounts([[0, 0], [0, 0]]), 'a zero total is refused');
refuses(() => informationFromCounts([[1, 1], [1]]), 'a ragged table is refused');
refuses(() => informationFromCounts([]), 'an empty table is refused');
refuses(() => informationFromCounts([[Number.NaN, 1], [1, 1]]), 'a non-finite count is refused');

// The mosaic is drawn from areas, so its areas must be the cell probabilities.
for (const table of [[[3, 1], [1, 3]], [[4, 0], [0, 4]], [[2, 0], [0, 6]], [[3, 1], [0, 0]]]) {
  const mosaic = probabilityMosaic(table);
  mosaic.tiles.forEach(tile => close(tile.area, tile.joint, `tile area equals its cell probability in ${JSON.stringify(table)}`, 1e-15));
  close(mosaic.tiles.reduce((sum, tile) => sum + tile.area, 0), 1, 'the tiles fill the unit square', 1e-12);
  close(mosaic.spannedWidth, 1, 'and the columns span its whole width', 1e-12);
  record('mosaic areas');
}

/* ------------------------------------------------- XOR and exact copies */

const xor = xorWorld();
assert.equal(xor.projections[0].information, 0, 'I(A; Y) is exactly zero in XOR');
assert.equal(xor.projections[1].information, 0, 'I(B; Y) is exactly zero in XOR');
close(xor.jointInformation, 1, 'the pair carries one bit');
close(xor.jointInformation, constructed.mi.xor_joint_bits, 'against the packet record');
xor.projections.forEach(projection => projection.values.forEach(value => {
  assert.deepEqual(value.counts, [1, 1], 'each projected value carries one of each target');
}));
assert.deepEqual(xor.targets, [0, 1, 1, 0], 'the declared XOR labels');
// Y = A instead: the projection onto A now carries the whole bit.
const alongA = xorWorld([0, 0, 1, 1]);
close(alongA.projections[0].information, 1, 'I(A; Y) is one bit when Y equals A');
assert.equal(alongA.projections[1].information, 0, 'while B still carries nothing');
refuses(() => xorWorld([0, 1, 2, 0]), 'a non-binary label is refused');
record('XOR projections');

const duplicate = duplicateColumns();
close(duplicate.first, 1, 'each exact copy carries one bit');
close(duplicate.second, 1, 'and so does the other');
close(duplicate.sum, 2, 'summing the two rankings claims two bits');
close(duplicate.joint, 1, 'while the pair actually carries one');
assert(duplicate.doubleCounted, 'which is exactly the double count');
close(jointInformation([[1, 0], [0, 0], [0, 0], [0, 1]]), 1, 'the duplicate pair table directly');
record('exact copies');

/* ===================================================== §3 subset search */

const xorLattice = subsetWorld([0, 1, 1, 0]);
xorLattice.subsets.forEach((entry, index) => {
  const saved = constructed.subset_xor[index];
  assert.equal(entry.mask, saved.mask, 'the lattice is in bitmask order');
  assert.deepEqual(entry.predictions, saved.predictions, `the packet's predictions at ${maskLabel(entry.mask)}`);
  close(entry.accuracy, saved.accuracy, `the packet's accuracy at ${maskLabel(entry.mask)}`);
  record('XOR lattice against the packet');
});
assert.deepEqual(xorLattice.subsets.map(entry => entry.accuracy), [0.5, 0.5, 0.5, 1],
  'the manuscript scores 0.5, 0.5, 0.5 and 1');
const alongALattice = subsetWorld([0, 0, 1, 1]);
alongALattice.subsets.forEach((entry, index) => {
  close(entry.accuracy, constructed.subset_first_feature[index].accuracy, `Y = A accuracy at ${maskLabel(entry.mask)}`);
  assert.deepEqual(entry.predictions, constructed.subset_first_feature[index].predictions, 'and its predictions');
  record('Y = A lattice against the packet');
});
assert.deepEqual(alongALattice.subsets.map(entry => entry.accuracy), [0.5, 1, 0.5, 1],
  'the specification scores 0.5, 1, 0.5 and 1');
// The tie rule is declared and deterministic: a tied group predicts 0.
const tied = subsetWorld([0, 1, 1, 0]).subsets[0];
assert(tied.groups[0].tie, 'the empty subset groups every state and ties');
assert.equal(tied.groups[0].majority, 0, 'and a tie predicts 0');
// All-equal labels make every subset perfect.
assert.deepEqual(subsetWorld([0, 0, 0, 0]).subsets.map(entry => entry.accuracy), [1, 1, 1, 1],
  'a constant target is predicted perfectly by every subset');
refuses(() => subsetWorld([0, 1, 1]), 'a world with the wrong number of states is refused');
refuses(() => subsetWorld([0, 1, 1, 5]), 'a non-binary label is refused');
record('lattice tie rule and nulls');

// The search itself, with its evaluated candidates kept separate from its path.
const strict = forwardSelection([0, 1, 1, 0], 'strict');
assert.equal(strict.finalMask, 0, 'strictly improving forward selection never leaves the empty set on XOR');
close(strict.finalScore, 0.5, 'and finishes at 0.5');
assert.equal(strict.steps.length, 1, 'it evaluates exactly one step');
assert.equal(strict.steps[0].candidates.length, 2, 'with two candidates at that step');
assert(strict.steps[0].candidates.every(candidate => candidate.score === 0.5), 'both of which score 0.5');
assert.equal(strict.steps[0].accepted, false, 'and neither is accepted');
assert.match(strict.stopReason, /no strictly higher score/, 'the stop reason is announced');
assert.deepEqual(strict.path, [0], 'the accepted path is the empty subset alone');
const forced = forwardSelection([0, 1, 1, 0], 'forceTwo');
assert.equal(forced.finalMask, 3, 'forcing two additions reaches the pair');
close(forced.finalScore, 1, 'which scores 1');
assert(forced.reachedPair, 'and is reported as such');
assert.equal(forced.steps.length, 2, 'over two steps');
const foundA = forwardSelection([0, 0, 1, 1], 'strict');
assert.equal(foundA.finalMask, 1, 'the strict rule finds A when Y equals A');
close(foundA.finalScore, 1, 'scoring 1');
assert.equal(foundA.steps.length, 2, 'and then evaluates one more step');
assert.equal(foundA.steps[1].accepted, false, 'which it declines');
// A tie between candidates leaves A in front, as declared.
assert.equal(forced.steps[0].best.feature, 'A', 'an equal score is broken toward A');
refuses(() => forwardSelection([0, 1, 1, 0], 'greedy'), 'an undeclared policy is refused');
record('forward search behaviour');

// Fit counts follow the actual search.
assert.deepEqual(
  (({ subsets, candidateFits, total }) => ({ subsets, candidateFits, total }))(searchFitCounts({ features: 5, keep: 2, folds: 3, method: 'forward' })),
  { subsets: 9, candidateFits: 27, total: 28 }, 'forward selection from five to two over three folds');
assert.equal(searchFitCounts({ features: 5, keep: 2, method: 'rfe' }).total, 4, 'simple RFE from five to two takes four fits');
assert.deepEqual(
  (({ subsets, candidateFits, total }) => ({ subsets, candidateFits, total }))(searchFitCounts({ features: 6, keep: 3, folds: 4, method: 'forward' })),
  { subsets: 15, candidateFits: 60, total: 61 }, 'practice 8 forward selection');
assert.equal(searchFitCounts({ features: 6, keep: 3, method: 'rfe' }).total, 4, 'practice 8 RFE');
refuses(() => searchFitCounts({ features: 3, keep: 5, folds: 1, method: 'forward' }), 'keeping more than exists is refused');
refuses(() => searchFitCounts({ features: 5, keep: 2, folds: 0, method: 'forward' }), 'zero folds is refused');
refuses(() => searchFitCounts({ features: 5, keep: 2, method: 'backward' }), 'an unimplemented method is refused');
record('search fit counts');

/* ======================================================= §4 permutation */

const permutationRows = [[-1, -1], [-1, -1], [1, 1], [1, 1]];
const permutationTarget = [-1, -1, 1, 1];
const donor = [2, 3, 0, 1];
assert.deepEqual(permutationRows, constructed.permutation.x, 'the packet rows');
assert.deepEqual(permutationTarget, constructed.permutation.target, 'the packet target');
assert.deepEqual(donor, constructed.permutation.donor, 'the packet donor ordering');
const suite = permutationSuite({
  rows: permutationRows, target: permutationTarget, donor,
  models: [
    { name: 'first_only', coefficients: [1, 0] },
    { name: 'second_only', coefficients: [0, 1] },
    { name: 'average', coefficients: [0.5, 0.5] },
  ],
});
suite.forEach((model, index) => {
  const saved = constructed.permutation.models[index];
  assert.equal(model.name, saved.model, 'the model order matches the packet');
  vector(model.coefficients, saved.coefficient, `${model.name} coefficients`);
  model.results.forEach((result, position) => {
    assert.deepEqual(result.columns, saved.changes[position].group, `${model.name} perturbed group`);
    close(result.increase, saved.changes[position].mse_increase, `${model.name} MSE increase`);
    vector(result.run.alteredPredictions, saved.changes[position].prediction, `${model.name} altered predictions`);
    close(result.baseMse, saved.baseline_mse, `${model.name} baseline MSE`);
  });
  record('permutation suite against the packet');
});
assert.deepEqual(suite.map(model => model.results.map(result => result.increase)),
  [[4, 0, 4], [0, 4, 4], [1, 1, 4]], 'the manuscript table 4/0/4, 0/4/4 and 1/1/4');
record('the manuscript permutation table');

// Practice 3's donor, which has two fixed points.
const practiceSingle = permutationExperiment({
  rows: permutationRows, target: permutationTarget, coefficients: [0.5, 0.5], donor: [0, 2, 1, 3], columns: [0],
});
vector(practiceSingle.alteredPredictions, [-1, 0, 0, 1], 'practice 3 altered predictions');
close(practiceSingle.increase, 0.5, 'practice 3 MSE increase');
assert.deepEqual(practiceSingle.fixedPoints, [0, 3], 'rows 0 and 3 donate to themselves');
const practiceGroup = permutationExperiment({
  rows: permutationRows, target: permutationTarget, coefficients: [0.5, 0.5], donor: [0, 2, 1, 3], columns: [0, 1],
});
vector(practiceGroup.alteredPredictions, [-1, 1, -1, 1], 'practice 3 grouped predictions');
close(practiceGroup.increase, 2, 'practice 3 grouped MSE increase');
record('practice 3 permutation');

// Two exact nulls.
for (const coefficients of [[1, 0], [0, 1], [0.5, 0.5], [2, -3]]) {
  for (const columns of [[0], [1], [0, 1]]) {
    const identity = permutationExperiment({
      rows: permutationRows, target: permutationTarget, coefficients, donor: [0, 1, 2, 3], columns,
    });
    assert.equal(identity.increase, 0, 'the identity donor map changes nothing at all');
    assert.deepEqual(identity.altered, permutationRows, 'and leaves every row untouched');
  }
  record('identity donor null');
}
const unusedColumn = permutationExperiment({
  rows: permutationRows, target: permutationTarget, coefficients: [1, 0], donor, columns: [1],
});
assert.equal(unusedColumn.increase, 0, 'shuffling a column with coefficient zero is an exact null');
assert.notDeepEqual(unusedColumn.altered, permutationRows, 'even though the values really did move');
record('unused-column null');

// A grouped shuffle preserves the within-group pairing; separate shuffles need not.
const grouped = permutationExperiment({
  rows: [[1, 10], [2, 20], [3, 30], [4, 40]], target: [1, 2, 3, 4], coefficients: [1, 0], donor: [1, 2, 3, 0], columns: [0, 1],
});
grouped.altered.forEach((row, index) => {
  assert.equal(row[1], row[0] * 10, `the group's internal relation survives in row ${index}`);
});
record('grouped pairing survives');

// A negative increase is a valid outcome and is reported, not clipped.
const improved = permutationExperiment({
  rows: [[1, 0], [-1, 0], [1, 0], [-1, 0]], target: [-1, 1, -1, 1], coefficients: [1, 0], donor: [1, 0, 3, 2], columns: [0],
});
assert(improved.increase < 0, 'a shuffle that happens to score better gives a negative increase');
close(improved.increase, -4, 'and its exact value is reported');
record('negative increase');

refuses(() => checkDonor([0, 1, 1, 3], 4), 'a repeated donor row is refused');
refuses(() => checkDonor([0, 1, 2], 4), 'a donor list of the wrong length is refused');
refuses(() => checkDonor([0, 1, 2, 4], 4), 'a donor row outside the range is refused');
refuses(() => permutationExperiment({ rows: permutationRows, target: [1, 2, 3], coefficients: [1, 0], donor, columns: [0] }),
  'a mismatched target length is refused');
refuses(() => permutationExperiment({ rows: permutationRows, target: permutationTarget, coefficients: [1], donor, columns: [0] }),
  'a coefficient vector of the wrong width is refused');
refuses(() => permutationExperiment({ rows: permutationRows, target: permutationTarget, coefficients: [1, 0], donor, columns: [] }),
  'an empty perturbation group is refused');
refuses(() => permutationExperiment({ rows: permutationRows, target: permutationTarget, coefficients: [1, 0], donor, columns: [2] }),
  'a column outside the design is refused');
vector(linearPredictions(permutationRows, [0.5, 0.5]), [-1, -1, 1, 1], 'the fixed linear predictor');

/* ======================================================== §5 coalitions */

/** A third, independent route to a Shapley value: average the marginal
 * increment over every permutation of the players, generated here rather than
 * taken from the model layer. */
function shapleyByEnumeration(values) {
  const dimension = Math.round(Math.log2(values.length));
  const players = Array.from({ length: dimension }, (_, index) => index);
  const orders = [];
  const build = (left, taken) => {
    if (left.length === 0) { orders.push(taken); return; }
    left.forEach(player => build(left.filter(other => other !== player), [...taken, player]));
  };
  build(players, []);
  const totals = new Array(dimension).fill(0);
  orders.forEach(order => {
    let mask = 0;
    order.forEach(player => {
      totals[player] += values[mask | (1 << player)] - values[mask];
      mask |= (1 << player);
    });
  });
  return totals.map(value => value / orders.length);
}

const zeroReference = explainPolynomial({ instance: [2, 3], background: [[0, 0]], gamma: 1 });
vector(zeroReference.values, [0, 2, 3, 11], 'the zero-reference coalition values');
vector(zeroReference.phi, [5, 6], 'the zero-reference attributions');
vector(zeroReference.values, constructed.nonlinear_shapley[0].coalitions, 'against the packet coalitions');
vector(zeroReference.phi, constructed.nonlinear_shapley[0].phi, 'against the packet attributions');
vector(shapleyByEnumeration(zeroReference.values), zeroReference.phi, 'independent order enumeration agrees');
close(zeroReference.efficiencyError, 0, 'the allocation reconstructs the prediction');
record('the zero-reference game');

const twoRow = explainPolynomial({ instance: [2, 3], background: [[0, 0], [1, 1]], gamma: 1 });
vector(twoRow.values, [1.5, 3.5, 5, 11], 'the two-row coalition values');
vector(twoRow.phi, [4, 5.5], 'the two-row attributions');
vector(twoRow.values, constructed.nonlinear_shapley[1].coalitions, 'against the packet coalitions');
vector(twoRow.phi, constructed.nonlinear_shapley[1].phi, 'against the packet attributions');
vector(shapleyByEnumeration(twoRow.values), twoRow.phi, 'independent order enumeration agrees');
close(twoRow.phi[0] + twoRow.phi[1], 9.5, 'the contributions sum to 9.5 above the new baseline');
close(twoRow.prediction, zeroReference.prediction, 'the model and its prediction did not change');
close(twoRow.baseline, 1.5, 'only the reference question changed');
close(twoRow.outputOfMeanInput, 1.25, 'f at the mean reference input is 1.25');
assert.notEqual(twoRow.outputOfMeanInput, twoRow.baseline, 'which is not the mean of the outputs');
record('the two-row reference game');

const practiceGame = explainPolynomial({ instance: [1, 2], background: [[0, 0]], gamma: 2 });
vector(practiceGame.values, [0, 1, 2, 7], 'practice 4 coalition values');
vector(practiceGame.phi, [3, 4], 'practice 4 attributions');
vector(shapleyByEnumeration(practiceGame.values), practiceGame.phi, 'independent enumeration agrees');
close(practiceGame.paths.paths[0].steps[0].increment, 1, 'the A-first increment is one');
close(practiceGame.paths.paths[1].steps[1].increment, 5, 'and the A-second increment is five');
record('practice 4 game');

// Gamma = 0 is a null for order-dependent increments, and only for those.
const noInteraction = explainPolynomial({ instance: [2, 3], background: [[0, 0]], gamma: 0 });
assert(noInteraction.orderIndependent, 'no interaction means every order gives the same increments');
vector(noInteraction.phi, [2, 3], 'and each feature simply takes its own value');
assert(!zeroReference.orderIndependent, 'while an interaction makes the orders differ');
// Every ordering enumeration agrees with the weighted formula.
[zeroReference, twoRow, practiceGame, noInteraction].forEach(game => {
  assert(game.paths.agrees, 'averaged increments agree with the weighted formula');
  close(game.baseline + game.phi.reduce((sum, value) => sum + value, 0), game.prediction, 'efficiency holds');
  record('efficiency and order averaging');
});
// An instance that equals every reference row has nothing to allocate.
const atReference = explainPolynomial({ instance: [0, 0], background: [[0, 0]], gamma: 1 });
vector(atReference.phi, [0, 0], 'an instance at its reference receives zero contributions');
close(atReference.baseline, atReference.prediction, 'and its baseline equals its prediction');
record('instance-at-reference null');

// Coalition values really are averages of the actual hybrid rows.
const manual = coalitionGame({
  predict: polynomialModel(1), instance: [2, 3], background: [[0, 0], [1, 1]], keepHybrids: true,
});
const manualInstance = [2, 3];
const manualBackground = [[0, 0], [1, 1]];
manual.masks.forEach(entry => {
  // Rebuild the hybrid rows and the coalition value here, from the declared
  // rule and plain arithmetic, rather than re-running the module's own helper.
  const expectedRows = manualBackground.map(row =>
    row.map((value, column) => ((entry.mask & (1 << column)) !== 0 ? manualInstance[column] : value)));
  assert.deepEqual(entry.rows, expectedRows, 'the hybrid rows are the declared overwrite of the reference rows');
  const expectedValue = expectedRows
    .map(([a, b]) => a + b + a * b)
    .reduce((sum, value) => sum + value, 0) / expectedRows.length;
  close(entry.value, expectedValue, 'a coalition value is the mean of the model at those rows');
  entry.rows.forEach(row => row.forEach((value, column) => {
    if (entry.kept[column]) assert.equal(value, manualInstance[column], 'a retained coordinate takes the instance value');
  }));
  record('hybrid construction');
});
// The same rows, from the public helper, must agree with the game's own copy.
manual.masks.forEach(entry =>
  assert.deepEqual(hybridRows(manualInstance, manualBackground, entry.mask), entry.rows,
    'hybridRows and the game agree on every mask'));
// Missing coordinates come from the same donor row: check the pairing survives.
const paired = hybridRows([9, 9], [[1, 2], [3, 4]], 0);
assert.deepEqual(paired, [[1, 2], [3, 4]], 'with nothing retained, each donor row arrives intact');

refuses(() => shapleyValues([1, 2, 3]), 'a game without a power-of-two size is refused');
refuses(() => coalitionGame({ predict: () => [1], instance: [1, 2], background: [] }), 'an empty reference is refused');
refuses(() => coalitionGame({ predict: () => [1], instance: [1, 2], background: [[1, 2, 3]] }), 'a mismatched reference width is refused');
refuses(() => coalitionGame({ predict: () => [Number.NaN, Number.NaN], instance: [1, 2], background: [[0, 0], [1, 1]] }),
  'a non-finite prediction is refused');
refuses(() => coalitionGame({ predict: () => [1], instance: [1, 2], background: [[0, 0], [1, 1]] }),
  'a predictor returning the wrong number of outputs is refused');
refuses(() => explainPolynomial({ instance: [2, 3], background: [[0, 0]], gamma: 99 }), 'an out-of-range interaction is refused');
refuses(() => hybridRows([1, Number.NaN], [[0, 0]], 1), 'a non-finite instance is refused');

// The two missing-feature rules.
const dependent = dependentGames();
vector(dependent.conditional.values, constructed.dependent_shapley.conditional.coalitions, 'conditional coalition values');
vector(dependent.conditional.phi, constructed.dependent_shapley.conditional.phi, 'conditional attributions');
vector(dependent.replacement.values, constructed.dependent_shapley.replacement.coalitions, 'replacement coalition values');
vector(dependent.replacement.phi, constructed.dependent_shapley.replacement.phi, 'replacement attributions');
vector(dependent.conditional.phi, [0.25, 0.25], 'conditioning splits the bit evenly');
vector(dependent.replacement.phi, [0.5, 0], 'replacement gives the second coordinate exactly zero');
assert.equal(dependent.replacement.phi[1], 0, 'and that zero is exact');
assert(dependent.impossibleHybrids.length > 0, 'the replacement game does evaluate impossible hybrids');
assert(dependent.impossibleHybrids.every(item => item.row[0] !== item.row[1]), 'which are exactly the unequal pairs');
record('dependent-feature comparison');

// The waterfall a figure draws is arithmetic, not placement.
const chart = waterfall(0.33, [-0.45, 0, 0.07, 0.05]);
assert.equal(chart.segments.length, 4, 'one segment per contribution');
close(chart.segments[0].start, 0.33, 'the first segment starts at the baseline');
close(chart.segments[3].end, chart.reconstruction, 'and the last segment ends at the reconstruction');
close(chart.reconstruction, 0, 'which is the reconstructed output');
close(chart.total, -0.33, 'the total movement');
assert.equal(chart.segments[1].sign, 0, 'a zero contribution is signed zero');
chart.segments.forEach((segment, index) => {
  if (index > 0) close(segment.start, chart.segments[index - 1].end, 'each segment starts where the last ended');
});
assert(chart.minimum <= chart.baseline && chart.maximum >= chart.baseline, 'the drawn extent contains the baseline');
refuses(() => waterfall(Number.NaN, [1]), 'a non-finite baseline is refused');
record('waterfall arithmetic');

// A cumulative fraction ends at one, or is explicitly undefined.
const fraction = cumulativeAttributionFraction([-0.45, 0, 0.07, 0.05]);
assert.equal(fraction.defined, true, 'a positive total gives a defined fraction');
// The ordering is by magnitude and the running sums are the cumulative totals.
assert.deepEqual(fraction.fractions.map(entry => entry.index), [0, 2, 3, 1],
  'the four attributions are ordered by magnitude, largest first');
vector(fraction.fractions.map(entry => entry.cumulative), [0.45, 0.52, 0.57, 0.57],
  'the cumulative magnitudes', 1e-12);
vector(fraction.fractions.map(entry => entry.fraction),
  [0.45 / 0.57, 0.52 / 0.57, 1, 1], 'and the fractions of the total', 1e-12);
assert(fraction.endsAtOne, 'and it finishes at exactly one');
close(fraction.total, 0.57, 'the total absolute attribution');
assert.equal(fraction.fractions[0].index, 0, 'the largest magnitude is ordered first');
const undefinedFraction = cumulativeAttributionFraction([0, 0, 0, 0]);
assert.equal(undefinedFraction.defined, false, 'a zero total is undefined, not zero or one');
assert(undefinedFraction.fractions.every(value => value === null), 'and reports no fractions');
record('cumulative attribution fraction');

/* ========================================================= §6 the tree */

const tree = fourFieldModel.tree;
assert.equal(tree.childrenLeft.length, 9, 'the saved tree has nine nodes');
assert.deepEqual(tree.childrenLeft, wine.four_feature_model.tree.children_left, 'left children match the packet');
assert.deepEqual(tree.childrenRight, wine.four_feature_model.tree.children_right, 'right children match the packet');
assert.deepEqual(tree.feature, wine.four_feature_model.tree.feature, 'split fields match the packet');
assert.deepEqual(tree.threshold, wine.four_feature_model.tree.threshold, 'thresholds match the packet at full precision');
assert.deepEqual(tree.samples, wine.four_feature_model.tree.n_node_samples, 'node sample counts match the packet');
tree.value.forEach((row, node) => vector(row, wine.four_feature_model.tree.value[node], `node ${node} distribution`));
assert.equal(tree.threshold[0], 12.78000020980835, 'the root splits alcohol at the recorded double');
assert.equal(tree.threshold[1], 1.0049999952316284, 'the left child splits flavanoids at the recorded double');
assert.equal(tree.threshold[4], 1.5899999737739563, 'the right subtree splits flavanoids at the recorded double');
assert.equal(tree.threshold[6], 655, 'and later splits proline at 655');
assert.deepEqual([2, 3, 5, 7, 8].map(node => tree.value[node]),
  [[0, 0.4, 0.6], [0, 1, 0], [0, 0, 1], [0, 1, 0], [1, 0, 0]], 'the five leaf distributions');
assert(!tree.feature.filter((field, node) => tree.childrenLeft[node] !== -1).includes(1),
  'no split in this tree uses malic acid');
record('the saved tree structure');

// The float32 contract. A value whose double is above a threshold but whose
// float32 cast is not would take the other branch; the cast is what sklearn does.
assert.equal(toFloat32(12.51), Math.fround(12.51), 'the cast is a float32 round trip');
assert(toFloat32(12.78) < tree.threshold[0], 'a raw 12.78 casts below the root threshold and goes left');
assert(toFloat32(12.79) > tree.threshold[0], 'while 12.79 casts above it and goes right');
// The comparison is <=, so a measurement that casts exactly onto a threshold
// takes the left child. Node 1's threshold is itself a float32, so feeding it
// back does exactly that.
const onNodeOne = treeDecision(tree, [12.5, 2, tree.threshold[1], 700]);
assert.equal(onNodeOne.path[1].goesLeft, true, 'a measurement exactly on a representable threshold goes left');
// The root's threshold is a midpoint BETWEEN two float32 values and is not one
// itself, so feeding it back as a measurement casts upward and goes right. The
// cast happens before the comparison; that ordering is the contract.
assert.notEqual(Math.fround(tree.threshold[0]), tree.threshold[0], 'the root threshold is not a float32 value');
assert(Math.fround(tree.threshold[0]) > tree.threshold[0], 'it casts upward');
const onRoot = treeDecision(tree, [tree.threshold[0], 2, 2, 700]);
assert.equal(onRoot.path[0].goesLeft, false, 'so feeding the stored root threshold back goes right, not left');
const justBelow = treeDecision(tree, [12.77, 2, 2, 700]);
assert.equal(justBelow.path[0].goesLeft, true, 'while a hair below it goes left');
const justAbove = treeDecision(tree, [tree.threshold[0] + 0.001, 2, 2, 700]);
assert.equal(justAbove.path[0].goesLeft, false, 'and a hair above it goes right');
// The cast is not decoration. Node 1's threshold IS a float32 value, so a double
// just above it rounds back down onto it and takes the LEFT child, which a
// comparison of unconverted doubles would get wrong.
assert.equal(Math.fround(1.005), tree.threshold[1], 'node 1 stores an exact float32 threshold');
const sensitive = 1.00500001;
assert(sensitive > tree.threshold[1], 'this flavanoid value is above that threshold as a double');
assert(toFloat32(sensitive) <= tree.threshold[1], 'and at or below it once cast to float32');
const castDecision = treeDecision(tree, [12.5, 2, sensitive, 700]);
assert.equal(castDecision.path[1].goesLeft, true, 'so the fitted estimator sends it left');
assert.equal(castDecision.leaf, 2, 'to leaf 2');
assert.notEqual(castDecision.leaf, 3, 'and not to the leaf an unconverted double comparison would reach');
vector(castDecision.probabilities, tree.value[2], 'with leaf 2s own distribution', 0);
refuses(() => treeDecision(tree, [Number.NaN, 1, 1, 1]), 'a non-finite measurement is refused');
refuses(() => toFloat32('12.5'), 'a non-numeric measurement is refused');
record('float32 threshold contract');

/* ---- the drawn rule must equal the applied rule at EVERY enterable value ---- */

// A drawn threshold is a rule the learner reasons from. Rounding produces one
// that can disagree with the tree at a value the control accepts: 1.5899999737…
// rounds to 1.59, and 1.59 casts ABOVE it and takes the other branch. The drawn
// value is instead the largest grid step that still goes left, and the property
// is asserted over the complete enterable domain of every split, not at samples.
let enterableChecked = 0;
tree.childrenLeft.forEach((left, node) => {
  if (left === -1) return;
  const field = tree.feature[node];
  const control = fourFieldModel.limits[field];
  const drawn = drawnThreshold(tree.threshold[node], control);
  assert(drawn !== null, `node ${node} has an enterable value that takes the left branch`);
  const misses = thresholdDisagreements(tree.threshold[node], control, drawn.value);
  assert.deepEqual(misses, [],
    `node ${node}: the drawn rule disagrees with the applied rule at ${misses.length} enterable value(s), e.g. ${JSON.stringify(misses[0])}`);
  const steps = Math.round(control.maximum * 10 ** control.decimals) - Math.round(control.minimum * 10 ** control.decimals) + 1;
  enterableChecked += steps;
  // And the drawn value really is on the control's own grid.
  const scale = 10 ** control.decimals;
  assert.equal(Math.round(drawn.value * scale) / scale, drawn.value, `node ${node}'s drawn threshold sits on the control grid`);
  record('drawn rule equals applied rule over the whole enterable grid');
});
assert(enterableChecked > 20000, `the grid sweep covered ${enterableChecked} enterable values`);
// The specific regression the review found: node 4 must not be drawn as 1.59.
assert.equal(drawnThreshold(tree.threshold[4], fourFieldModel.limits[2]).value, 1.58,
  'node 4 is drawn as 1.58, the last flavanoid value that still goes left');
assert(toFloat32(1.59) > tree.threshold[4], 'because 1.59 itself casts above the stored threshold');
assert(toFloat32(1.58) <= tree.threshold[4], 'while 1.58 casts at or below it');
assert.equal(drawnThreshold(tree.threshold[0], fourFieldModel.limits[0]).value, 12.78, 'node 0 is drawn as 12.78');
assert.equal(drawnThreshold(tree.threshold[1], fourFieldModel.limits[2]).value, 1, 'node 1 is drawn as 1.00');
assert.equal(drawnThreshold(tree.threshold[6], fourFieldModel.limits[3]).value, 655, 'node 6 is drawn as 655.0');
// A control whose whole range sits above the threshold has no equivalent rule.
assert.equal(drawnThreshold(0.5, { minimum: 1, maximum: 2, decimals: 2 }), null,
  'no enterable value goes left, so there is no grid-equivalent rule to draw');
refuses(() => drawnThreshold(1, { minimum: 2, maximum: 1, decimals: 2 }), 'an inverted control range is refused');
refuses(() => drawnThreshold(1, { minimum: 0, maximum: 1, decimals: 1.5 }), 'a fractional decimal count is refused');
record('drawn threshold contract');

// Inference agrees with the recorded native predictions on every row it can.
inspectionRows.forEach((row, index) => {
  assert.equal(treeClass(tree, row, fourFieldModel.classes), inspectionPredictions[index],
    `inspection row ${inspectionIds[index]} reproduces the native hard prediction`);
  const decision = treeDecision(tree, row);
  assert(tree.childrenLeft[decision.leaf] === -1, 'and it lands on a leaf');
  close(treeProbability(tree, row, fourFieldModel.classIndex),
    wine.four_feature_model.tree.value[decision.leaf][fourFieldModel.classIndex],
    'the reported probability is the recorded leaf value from the packet', 0);
});
assert.equal(inspectionRows.length, 38, 'all 38 inspection rows were checked');
assert.equal(inspectionPredictions.filter((value, index) => value === inspectionClasses[index]).length, 36,
  'and 36 of them are correct, as recorded');
record('native inference agreement');
backgroundRows.forEach((row, index) => {
  const decision = treeDecision(tree, row);
  assert(tree.childrenLeft[decision.leaf] === -1, `background row ${backgroundIds[index]} reaches a leaf`);
});
assert.equal(backgroundRows.length, 100, 'all 100 background rows were traced');
record('background inference');

// Impurity importance, recomputed and normalised as the estimator does.
const importance = impurityImportance(tree, 4);
vector(importance.normalized, fourFieldModel.mdi, 'recomputed impurity importances match the fitted values');
vector(importance.normalized, wine.four_feature_model.mdi, 'and match the packet record');
close(importance.normalized.reduce((sum, value) => sum + value, 0), 1, 'they are normalised to one');
assert.equal(importance.normalized[1], 0, 'malic acid receives exactly zero');
assert.equal(importance.nodes.length, 4, 'four internal nodes contribute');
vector(importance.normalized, [0.40289733091541075, 0, 0.4601709357778123, 0.13693173330677688],
  'the manuscript table values', 1e-9);
record('impurity importance');

/* ---------------------------------------- the twelve explained cases */

explainedCases.forEach(entry => {
  const explanation = explainTreeInstance({
    tree, instance: entry.input, background: backgroundRows, classIndex: fourFieldModel.classIndex,
  });
  vector(explanation.values, entry.coalitions, `source row ${entry.sourceId} coalition values`);
  vector(explanation.phi, entry.phi, `source row ${entry.sourceId} attributions`);
  close(explanation.baseline, entry.baseline, `source row ${entry.sourceId} baseline`);
  close(explanation.prediction, entry.prediction, `source row ${entry.sourceId} prediction`);
  assert(explanation.reconstructs, `source row ${entry.sourceId} reconstructs within ${limits.reconstruction}`);
  assert.equal(explanation.phi[1], 0, `malic acid contributes exactly zero for source row ${entry.sourceId}`);
  // A third independent route for the four-player game.
  vector(shapleyByEnumeration(explanation.values), explanation.phi,
    `source row ${entry.sourceId} by independent order enumeration`, 1e-12);
  record('explained wine case');
});
const first = explainedCases[0];
assert.equal(first.sourceId, 104, 'the first explained case is source row 104');
vector(first.input, [12.51, 1.73, 1.92, 672], 'its four measurements');
close(first.baseline, 0.33, 'the background class-1 mean is 0.33');
close(first.prediction, 0, 'its class-1 probability is zero');
vector(first.phi, [-0.45, 0, 0.07, 0.05], 'and its four contributions', 1e-9);
assert(Math.abs(first.efficiencyError) < 1e-15, 'reconstructing to floating-point dust');
vector(first.coalitions, [0.33, 0, 0.33, 0, 0.42, 0, 0.42, 0, 0.38, 0, 0.38, 0, 0.62, 0, 0.62, 0],
  'the sixteen coalition values in bitmask order', 1e-12);
// Every coalition that retains this instance's alcohol has value zero.
first.coalitions.forEach((value, mask) => {
  if ((mask & 1) !== 0) assert.equal(value, 0, `coalition ${mask} retains alcohol and is zero`);
});
close(first.coalitions[4], 0.42, 'flavanoids only');
close(first.coalitions[8], 0.38, 'proline only');
close(first.coalitions[12], 0.62, 'flavanoids and proline');
record('source row 104');

// The leaf tally really produces the coalition value.
const tallied = explainTreeInstance({
  tree, instance: first.input, background: backgroundRows, classIndex: fourFieldModel.classIndex, keepHybrids: true,
});
tallied.leafTally.forEach(entry => {
  assert.equal(entry.leaves.reduce((sum, leaf) => sum + leaf.count, 0), backgroundRows.length,
    `every hybrid row is tallied for coalition ${entry.mask}`);
  close(entry.leaves.reduce((sum, leaf) => sum + leaf.count * leaf.probability, 0) / backgroundRows.length,
    entry.value, `the leaf tally reproduces v(S) for coalition ${entry.mask}`);
  record('leaf tally');
});

// The two recorded input contrasts.
changedInference.forEach(entry => {
  const explanation = explainTreeInstance({
    tree, instance: entry.input, background: backgroundRows, classIndex: fourFieldModel.classIndex,
  });
  vector(explanation.values, entry.coalitions, `${entry.name} coalition values`);
  vector(explanation.phi, entry.phi, `${entry.name} attributions`);
  close(explanation.prediction, entry.prediction, `${entry.name} prediction`);
  record('recorded input contrast');
});
const alcoholContrast = changedInference.find(entry => entry.name === 'alcohol_contrast');
close(alcoholContrast.prediction, 1, 'changing alcohol to 13.5 gives a class-1 probability of one');
vector(alcoholContrast.phi, [0.21666666666666667, 0, 0.20666666666666664, 0.24666666666666665],
  'and the recorded contributions', 1e-12);
close(alcoholContrast.baseline, 0.33, 'above the same 0.33 reference');
const malicNull = changedInference.find(entry => entry.name === 'malic_acid_null');
assert.deepEqual(malicNull.coalitions, first.coalitions, 'changing malic acid alone changes no coalition value');
assert.deepEqual(malicNull.phi, first.phi, 'and no attribution');
close(malicNull.prediction, first.prediction, 'and no prediction');
record('the alcohol contrast and the malic-acid null');

// The deliberately unrepresentative reference cohort.
const cohort = explainTreeInstance({
  tree, instance: first.input, background: alternativeReference.rows, classIndex: fourFieldModel.classIndex,
});
vector(cohort.values, alternativeReference.coalitions, 'cohort coalition values');
vector(cohort.phi, alternativeReference.phi, 'cohort attributions');
close(cohort.baseline, 0, 'the class-3 cohort has a zero class-1 baseline');
close(cohort.prediction, first.prediction, 'while the explained prediction is unchanged');
vector(cohort.phi, [-0.41666666666666663, 0, 0.375, 0.041666666666666664], 'the manuscript cohort contributions', 1e-12);
assert(Math.abs(cohort.baseline + cohort.phi.reduce((sum, value) => sum + value, 0) - cohort.prediction) <= 1e-10,
  'and it still reconstructs the same output');
assert.equal(alternativeReference.ids.length, 12, 'the cohort is twelve rows');
assert.deepEqual(alternativeReference.classes, new Array(12).fill(3), 'and every one of them is cultivar 3');
assert.deepEqual(alternativeReference.ids, [...fittingIds].sort((a, b) => a - b).slice(-12),
  'they are the twelve highest-index fitting rows');
record('the contrast reference cohort');

// Extrapolation is marked, not refused.
const flags = extrapolationFlags([20, 1.73, 1.92, 672], fourFieldModel.fitRanges);
assert.equal(flags[0].outside, true, 'an alcohol of 20 is outside the observed fitting range');
assert.equal(flags[0].above, true, 'and is above it');
assert.equal(flags[1].outside, false, 'while the other fields are inside');
assert.equal(extrapolationFlags(first.input, fourFieldModel.fitRanges).filter(flag => flag.outside).length, 0,
  'the recorded row is entirely inside the observed ranges');
record('extrapolation flags');

/* =============================================== §7 and practice helpers */

close(sigmoid(0.5), 0.6224593312018546, 'practice 7 sigmoid');
close(sigmoid(0.5), constructed.practice.sigmoid_margin_point_five, 'against the packet record');
close(sigmoid(-0.2 + 0.8 - 0.1), sigmoid(0.5), 'the reconstructed margin is 0.5');
assert(Math.abs(sigmoid(-0.2) + sigmoid(0.8) + sigmoid(-0.1) - sigmoid(0.5)) > 0.1,
  'transforming each contribution separately does not reproduce the probability');
close(familywiseProbability(100, 0.05), 0.994079470779666, 'the family-wise probability');
close(familywiseProbability(100, 0.05), constructed.practice.null_family_probability, 'against the packet record');
refuses(() => familywiseProbability(0, 0.05), 'zero tests is refused');
refuses(() => familywiseProbability(100, 1.5), 'a level above one is refused');
record('practice arithmetic');

const three = unanimityGame(3);
vector(three.phi, constructed.practice.group_unanimity_individual, 'three-player unanimity against the packet');
vector(three.phi, [1 / 3, 1 / 3, 1 / 3], 'each individual receives a third');
const grouping = groupedUnanimity([2, 1]);
vector(grouping.grouped, constructed.practice.grouped_unanimity, 'the grouped allocation against the packet');
vector(grouping.grouped, [0.5, 0.5], 'each grouped player receives a half');
vector(grouping.summedIndividual, [2 / 3, 1 / 3], 'while summing individuals gives two thirds and a third');
assert.equal(grouping.agrees, false, 'grouping is not the same as summing afterwards');
const four = unanimityGame(4);
vector(four.phi, constructed.practice.four_player_individual, 'four-player unanimity against the packet');
const fourGrouped = groupedUnanimity([3, 1]);
vector(fourGrouped.summedIndividual, [0.75, 0.25], 'practice 9 summed individuals');
vector(fourGrouped.grouped, [0.5, 0.5], 'practice 9 grouped allocation');
// Equal group sizes are the case where the two do coincide.
assert(groupedUnanimity([2, 2]).agrees, 'equal groups make the two routes agree');
refuses(() => groupedUnanimity([1]), 'a single group is refused');
refuses(() => groupedUnanimity([0, 2]), 'an empty group is refused');
record('grouped attribution games');

/* ---------------------------------------------------- drawing arithmetic */

// Two quantities that are not in the same units get two scales, deliberately.
const permutationScale = barScale(permutationRecords.map(entry => entry.mean));
const impurityScale = barScale(importance.normalized);
close(permutationScale.extent, Math.max(...permutationRecords.map(entry => entry.mean)), 'the permutation scale');
close(impurityScale.extent, Math.max(...importance.normalized), 'the impurity scale');
assert.notEqual(permutationScale.extent, impurityScale.extent, 'they are different scales, as they must be');
const permutationMax = permutationRecords.reduce((best, entry) => Math.max(best, Math.abs(entry.mean)), 0);
permutationScale.shares.forEach((share, index) => {
  assert(Math.abs(share) <= 1 + 1e-12, 'no bar exceeds the panel width');
  close(share, permutationRecords[index].mean / permutationMax, 'a bar share is its value over the panel maximum');
});
assert.equal(permutationScale.shares.filter(share => Math.abs(share - 1) <= 1e-12).length, 1,
  'exactly one bar reaches the full width, and it is the largest');
assert.equal(permutationScale.shares[1], 0, 'the zero field gets a zero-length bar, not a minimum-width one');
assert.equal(barScale([0, 0, 0]).allZero, true, 'an all-zero panel is reported rather than magnified');
record('separate bar scales');

const points = sizePoints(candidates);
assert.deepEqual(points.sizes, [3, 6, 13], 'only three sizes were evaluated');
assert.equal(points.chosen, 6, 'and the best mean is at six');
assert.equal(points.chosen, selectedModel.k, 'which is the size the study selected');
// The drawing may only carry sizes the study actually fitted.
assert.deepEqual(points.sizes, candidates.map(entry => entry.k), 'every drawn size was an evaluated candidate');
assert.equal(points.sizes.length, candidates.length, 'and no extra point was invented between them');
vector(points.scores, candidates.map(entry => entry.meanAccuracy), 'each drawn height is the mean of that size');
assert.equal(points.lowSize, 3, 'the axis starts at the smallest evaluated size');
assert.equal(points.highSize, 13, 'and ends at the largest');
const matrix = maskMatrix(candidates.find(entry => entry.k === 6).folds, 13);
assert.equal(matrix.length, 3, 'three fold masks');
matrix.forEach(entry => {
  assert.equal(entry.membership.filter(Boolean).length, 6, 'each keeps exactly six fields');
  assert.deepEqual(entry.membership.map((kept, column) => (kept ? column : -1)).filter(column => column >= 0),
    entry.selected, 'and the membership row matches its own mask');
});
assert.notDeepEqual(matrix[0].selected, matrix[1].selected, 'the fold masks are not identical');
record('size points and mask matrix');

/* ============================================ the generated data module */

assert.equal(provenance.bytes, 10782, 'the served dataset size');
assert.equal(provenance.sha256, '6be6b1203f3d51df0b553a70e57b8a723cd405683958204f96d23d7cd6aea659', 'its hash');
assert.equal(provenance.rows, 178, '178 source rows');
assert.equal(provenance.license, 'CC BY 4.0', 'the licence travels with the data');
const servedBytes = fs.readFileSync('public/learn-assets/feature-selection/wine.data');
assert.equal(servedBytes.length, provenance.bytes, 'the served file is the recorded size');
assert.equal(crypto.createHash('sha256').update(servedBytes).digest('hex'), provenance.sha256,
  'and byte for byte the checkpointed member');
assert.deepEqual(servedBytes, fs.readFileSync(`${packetDir}/wine.data`), 'identical to the packet file');
assert(fs.existsSync('public/learn-assets/feature-selection/ATTRIBUTION.txt'), 'attribution is served beside it');
assert(!fs.existsSync('public/learn-assets/feature-selection/airfoil-self-noise.dat'),
  'this lesson serves only its own dataset');
record('served dataset');

assert.equal(split.reservedScored, false, 'no reserved row was scored');
assert.equal(split.fits.total, 11, 'eleven fits in total');
assert.equal(split.fits.selectionCv + split.fits.selectedRefit + split.fits.fourFieldRefit, split.fits.total,
  'and the three parts add up');
assert.equal(developmentIds.length, 138, '138 development rows');
assert.equal(reservedIds.length, 40, '40 reserved rows');
assert.equal(fittingIds.length, 100, '100 fitting rows');
assert.equal(inspectionIds.length, 38, '38 inspection rows');
assert.equal(new Set([...developmentIds, ...reservedIds]).size, 178, 'the outer split covers every row exactly once');
assert.equal(new Set([...fittingIds, ...inspectionIds]).size, 138, 'the inner split covers development exactly once');
assert.equal([...fittingIds, ...inspectionIds].filter(id => reservedIds.includes(id)).length, 0,
  'and no reserved row appears in either');
assert.deepEqual([...backgroundIds], [...fittingIds].sort((a, b) => a - b), 'the reference is the fitting rows, sorted');
foldMemberships.forEach(fold => {
  assert.equal(fold.fitIds.length + fold.validationIds.length, 100, 'each fold partitions the fitting rows');
  assert.equal(new Set([...fold.fitIds, ...fold.validationIds]).size, 100, 'without overlap');
  assert(fold.validationIds.every(id => fittingIds.includes(id)), 'and stays inside them');
  record('fold membership');
});
assert.equal(featureNames.length, 13, 'thirteen fields');
assert.equal(featureLabels.length, 13, 'with thirteen labels');
assert.deepEqual(classLabels, [1, 2, 3], 'three cultivars');
record('split accounting');

candidates.forEach(entry => {
  const saved = wine.candidates.find(candidate => candidate.k === entry.k);
  close(entry.meanAccuracy, saved.mean_accuracy, `k = ${entry.k} mean accuracy against the packet`);
  close(entry.meanAccuracy, entry.folds.reduce((sum, fold) => sum + fold.accuracy, 0) / 3,
    `k = ${entry.k} published mean is the mean of its own folds`);
  entry.folds.forEach((fold, index) => {
    assert.deepEqual(fold.selected, saved.folds[index].selected, `k = ${entry.k} fold ${index} mask`);
    assert.equal(fold.correct, saved.folds[index].correct, `k = ${entry.k} fold ${index} correct count`);
    close(fold.correct / fold.total, fold.accuracy, 'and its accuracy is that fraction');
    assert.equal(fold.selected.length, entry.k, `and it retained exactly ${entry.k} fields`);
    assert.equal(fold.miNats.length, 13, 'with one MI estimate per field');
  });
  record('published candidate');
});
assert.deepEqual(candidates.map(entry => entry.k), [3, 6, 13], 'the three declared sizes');
assert.deepEqual(candidates.map(entry => entry.folds.map(fold => `${fold.correct}/${fold.total}`)),
  [['30/34', '25/33', '26/33'], ['30/34', '29/33', '26/33'], ['30/34', '26/33', '28/33']],
  'the manuscript fold counts');
assert.equal(candidates.reduce((best, entry) => (entry.meanAccuracy > best.meanAccuracy ? entry : best)).k, 6,
  'the rule selects six');
assert.deepEqual(candidates.find(entry => entry.k === 6).folds.map(fold => fold.selected),
  [[0, 1, 6, 9, 11, 12], [0, 5, 6, 10, 11, 12], [0, 5, 6, 9, 11, 12]], 'the three specified k = 6 masks');
assert.deepEqual(selectedModel.columns, [0, 5, 6, 9, 11, 12], 'and the refit mask');
assert.deepEqual(selectedModel.names,
  ['alcohol', 'total_phenols', 'flavanoids', 'color_intensity', 'od280_od315', 'proline'],
  'the six selected fields, by name');
assert.equal(selectedModel.correct, 36, 'the selected model is correct on 36');
assert.equal(fourFieldModel.correct, 36, 'and so is the four-field model');
assert.equal(majorityBaseline.class, 2, 'the training-majority class is 2');
assert.equal(majorityBaseline.correct, 15, 'correct on 15 of 38');
assert.deepEqual(fourFieldModel.columns, [0, 1, 6, 12], 'the four predeclared raw columns');
assert.deepEqual(fourFieldModel.classes, [1, 2, 3], 'its three classes');
assert.equal(fourFieldModel.classIndex, 0, 'and class 1 is the first output column');
assert.deepEqual(fourFieldModel.confusion, wine.four_feature_model.confusion_matrix, 'the recorded confusion matrix');
assert.equal(fourFieldModel.confusion.flat().reduce((sum, value) => sum + value, 0), 38,
  'which accounts for all 38 inspection rows');
record('published study results');

permutationRecords.forEach((entry, index) => {
  const saved = wine.permutation[index];
  assert.equal(entry.name, saved.feature, 'the permutation records are in field order');
  assert.equal(entry.drops.length, 20, 'twenty donor repeats');
  vector(entry.drops, saved.drops, `${entry.name} drops against the packet`, 1e-9);
  close(entry.mean, saved.mean, `${entry.name} mean`, 1e-9);
  close(entry.sd, saved.std_population, `${entry.name} population SD`, 1e-9);
  // Independently recompute the mean and the population SD from the twenty values.
  const mean = entry.drops.reduce((sum, value) => sum + value, 0) / 20;
  const sd = Math.sqrt(entry.drops.reduce((sum, value) => sum + (value - mean) ** 2, 0) / 20);
  close(mean, entry.mean, `${entry.name} mean recomputed from its own repeats`, 1e-9);
  close(sd, entry.sd, `${entry.name} SD recomputed from its own repeats`, 1e-9);
  close(entry.mdi, fourFieldModel.mdi[index], `${entry.name} impurity value agrees with the model`);
  record('permutation record');
});
assert.deepEqual(permutationRecords.map(entry => Number(entry.mean.toFixed(6))),
  [0.221053, 0, 0.359211, 0.189474], 'the manuscript permutation means');
assert.deepEqual(permutationRecords.map(entry => Number(entry.sd.toFixed(6))),
  [0.048809, 0, 0.082076, 0.029539], 'the manuscript permutation SDs');
assert.equal(permutationRecords[1].mean, 0, 'malic acid is exactly zero, not a small number');
assert(permutationRecords[1].drops.every(value => value === 0), 'in every one of its twenty repeats');
record('the manuscript permutation table');

// Control bounds contain the observed columns and admit the manuscript edits.
fourFieldModel.limits.forEach((bounds, index) => {
  assert(bounds.minimum < fourFieldModel.fitRanges[index][0], `bound ${index} is below the observed minimum`);
  assert(bounds.maximum > fourFieldModel.fitRanges[index][1], `bound ${index} is above the observed maximum`);
  record('control bound');
});
assert(fourFieldModel.limits[0].maximum >= 13.5, 'the alcohol contrast edit is reachable');
assert(fourFieldModel.limits[1].maximum >= 4.1, 'the malic-acid null edit is reachable');
explainedCases.forEach(entry => entry.input.forEach((value, index) => {
  assert(value >= fourFieldModel.limits[index].minimum && value <= fourFieldModel.limits[index].maximum,
    `every explained value of field ${index} is inside the control bounds`);
}));
// Every investigation bound the lesson points at must actually admit the value.
assert(limits.coordinate.maximum >= 3 && limits.gamma.maximum >= 2, 'the coalition presets fit inside their bounds');
assert(limits.sensor.minimum <= -1 && limits.sensor.maximum >= 1, 'the donor rows fit inside their bounds');
assert(limits.count.maximum >= 9, 'the scaled count preset fits inside its bound');
assert(limits.referenceRows.maximum >= 2, 'the two-row reference preset fits inside its bound');
record('pointer bounds are honoured');

// A pointer into a lab is a claim. Where the prose says an investigation loads a
// practice case from a preset, that preset must exist AND carry those values.
const labsSource = fs.readFileSync('src/learn/components/lesson-labs/SelectionLabs.jsx', 'utf8');

/** The source text of the preset object carrying a given label, so a value can
 * be required to live in THAT preset rather than anywhere in the file. */
function presetBody(label) {
  const at = labsSource.indexOf(`label: '${label}'`);
  assert(at >= 0, `a preset labelled ${label} exists`);
  const open = labsSource.lastIndexOf('{', at);
  let depth = 0;
  for (let index = open; index < labsSource.length; index += 1) {
    if (labsSource[index] === '{') depth += 1;
    if (labsSource[index] === '}') {
      depth -= 1;
      if (depth === 0) return labsSource.slice(open, index + 1);
    }
  }
  throw new Error(`the preset labelled ${label} is not closed`);
}

// Every pointer names a label AND the values that preset must carry, and those
// values are required inside that preset's own braces.
const pointers = [
  { claim: 'The first investigation loads this exact table from its practice preset',
    label: 'Practice 1: 2/0/0/6', values: ['counts: [[2, 0], [0, 6]]'] },
  { claim: 'The donor investigation loads this exact case from its practice preset',
    label: 'Practice 3: average model, donor 0, 2, 1, 3',
    values: ['donor: [0, 2, 1, 3]', 'coefficients: [0.5, 0.5]', "mode: 'first'"] },
  { claim: 'The coalition investigation loads this case from its practice preset',
    label: 'Practice 4: x = (1, 2), γ = 2, reference (0, 0)',
    values: ['instance: [1, 2]', 'gamma: 2', 'reference: [[0, 0]]'] },
  { claim: 'The investigation above loads that cohort from its reference-contrast preset',
    label: 'Reference contrast: the class-3 cohort', values: ["reference: 'cohort'"] },
  { claim: 'Both are reproducible in the investigation above through its two presets',
    label: 'Contrast: alcohol 12.51 → 13.5', values: ['values: changedInference[0].input.slice()'] },
];
pointers.forEach(({ claim, label, values }) => {
  assert.ok(lessonSource.includes(claim), `the lesson makes the pointer: ${claim}`);
  const body = presetBody(label);
  values.forEach(value => assert.ok(body.includes(value),
    `the preset "${label}" carries ${value}, not merely the label`));
  record('lesson pointer resolves to a preset carrying its values');
});
// The two presets whose values are data references, not literals: close the loop
// by checking the referenced data really carries the numbers the label names.
close(changedInference[0].input[0], 13.5, 'the alcohol-contrast preset really moves alcohol to 13.5');
close(explainedCases[0].input[0], 12.51, 'from the 12.51 the label names');
assert.deepEqual(changedInference[0].input.slice(1), explainedCases[0].input.slice(1),
  'and changes nothing else, as the label implies');
const malicBody = presetBody('Null: malic acid 1.73 → 4.1');
assert.ok(malicBody.includes('values: changedInference[1].input.slice()'), 'the malic null preset uses the recorded edit');
close(changedInference[1].input[1], 4.1, 'which really moves malic acid to 4.1');
close(explainedCases[0].input[1], 1.73, 'from the 1.73 the label names');
const cohortSource = labsSource.slice(labsSource.indexOf('const REFERENCES'), labsSource.indexOf('const winePresets'));
assert.ok(cohortSource.includes('rows: alternativeReference.rows'), 'the cohort reference is the recorded class-3 cohort');
assert.deepEqual(alternativeReference.classes, new Array(12).fill(3), 'whose rows really are all cultivar 3');
// And every value those presets name is inside the control range that accepts it.
[[2, 0], [0, 6]].flat().forEach(value =>
  assert(value >= limits.count.minimum && value <= limits.count.maximum, 'practice 1 counts are enterable'));
[1, 2].forEach(value =>
  assert(Math.abs(value) <= limits.coordinate.maximum, 'practice 4 coordinates are enterable'));
assert(Math.abs(2) <= limits.gamma.maximum, 'practice 4 interaction is enterable');
[0.5, 0.5].forEach(value =>
  assert(Math.abs(value) <= limits.coefficient.maximum, 'practice 3 coefficients are enterable'));
record('practice values are enterable');

/* =========================================== the executed example module */

assert.equal(Object.keys(selectionExamples).length, 4, 'four displayed programs');
assert.equal(selectionExamples.informationFromCounts.file, 'information_from_counts.py');
assert.equal(selectionExamples.coalitionAttribution.file, 'coalition_attribution.py');
assert.equal(selectionExamples.wineStudy.file, 'wine_feature_study.py');
assert.equal(selectionExamples.treeExplainerCheck.file, 'check_tree_explanation.py');
assert.equal(selectionExamples.informationFromCounts.expected,
  'MI = 0.188722 bits\nMI = 1.000000 bits\nMI = 0.000000 bits', 'the printed information totals');
assert.ok(selectionExamples.coalitionAttribution.expected.includes('baseline 0.0 contributions [5. 6.]'),
  'the printed zero-reference attributions');
assert.ok(selectionExamples.coalitionAttribution.expected.includes('baseline 1.5 contributions [4.  5.5]'),
  'the printed two-row attributions');
candidates.forEach(entry => {
  assert.ok(selectionExamples.wineStudy.expected.includes(
    `retained size ${entry.k} mean CV accuracy ${entry.meanAccuracy.toFixed(6)}`),
  `the program prints the mean for k = ${entry.k}`);
});
assert.ok(selectionExamples.wineStudy.expected.includes(`selected names ${JSON.stringify(selectedModel.names).replace(/"/g, "'").replace(/,/g, ', ')}`),
  'the program prints the selected field names the data module records');
// Python prints a rounded float, so an exact zero appears as "0.0", not "0".
const asPython = value => {
  const rounded = Number(value.toFixed(6));
  return Number.isInteger(rounded) ? rounded.toFixed(1) : String(rounded);
};
permutationRecords.forEach(entry => {
  assert.ok(selectionExamples.wineStudy.expected.includes(
    `${entry.name} accuracy drop ${asPython(entry.mean)} permutation SD ${asPython(entry.sd)}`),
  `the program prints ${entry.name}'s drop and SD`);
});
assert.ok(selectionExamples.wineStudy.expected.includes('background prediction 0.33 contributions [-0.45  0.    0.07  0.05]'),
  'the program prints the recorded baseline and contributions');
assert.ok(selectionExamples.wineStudy.code.includes('wine.data'), 'the program reads the served file');
assert.ok(!selectionExamples.wineStudy.code.includes('reserved.tolist'), 'and never scores the reserved rows');
assert.ok(selectionExamples.treeExplainerCheck.code.includes('feature_perturbation="interventional"'),
  'the optional program declares its dependence mode explicitly');
assert.ok(selectionExamples.treeExplainerCheck.code.includes('model_output="probability"'),
  'and its output scale explicitly');
// The four programs are verbatim the frozen manuscript blocks.
const manuscript = fs.readFileSync(`${packetDir}/lesson.md`, 'utf8');
Object.values(selectionExamples).forEach(example => {
  assert.ok(manuscript.includes(example.code), `${example.file} is verbatim the manuscript block`);
  record('verbatim displayed program');
});
record('displayed programs');

// The optional native comparison was actually executed, and its version is stated.
assert.equal(nativeEvidence.optionalShapComparison.executed, true, 'the optional shap comparison was executed');
assert.ok(nativeEvidence.optionalShapComparison.version, 'and its version was recorded');
assert.ok(lessonSource.includes(`const shapVersion = '${nativeEvidence.optionalShapComparison.version}'`),
  'and the lesson states the version that actually ran');
assert.ok(nativeEvidence.optionalShapComparison.mode.includes('interventional'), 'in explicit interventional mode');
record('optional native comparison');

/* ================================================== the lesson source */

for (const claim of [
  'reserved rows receive no prediction or score here',
  'not a statement that malic acid has no association with cultivar',
  'Every investigation asks for a prediction before it shows an answer',
  'These are different units and mechanisms',
  'It is not the MI-selected model',
  'not a representative population sample',
  'constructed calculations',
  'no reserved row predicted or scored',
  '{provenance.license}',
]) {
  assert.ok(lessonSource.includes(claim), `the lesson keeps the claim: ${claim}`);
  record('preserved claim');
}
assert.ok(lessonSource.includes('/learn-assets/feature-selection/') || lessonSource.includes('provenance.file'),
  'the lesson serves its own asset directory');
// Any reference to another lesson's asset directory, not just one named one.
const foreignAsset = /learn-assets\/(?!feature-selection[/"'])[a-z0-9-]+/g;
[['the lesson', lessonSource], ['the labs', labsSource],
 ['the figures', fs.readFileSync('src/learn/components/lesson-labs/SelectionFigures.jsx', 'utf8')],
 ['the shared components', fs.readFileSync('src/learn/components/lesson-labs/SelectionShared.jsx', 'utf8')],
 ['the data module', fs.readFileSync('src/learn/data/selection-data.js', 'utf8')],
 ['the models module', fs.readFileSync('src/learn/data/selection-models.js', 'utf8')],
].forEach(([name, source]) => {
  assert.deepEqual(source.match(foreignAsset) ?? [], [],
    `${name} never points at another lesson's asset directory`);
  record('no foreign asset reference');
});
assert.ok(lessonSource.includes('provenance.file') || lessonSource.includes('learn-assets/feature-selection'),
  'and it does serve its own');
assert.equal((lessonSource.match(/<Program /g) ?? []).length, 4, 'four displayed programs are rendered');
for (const component of ['CountInformationLab', 'SubsetSearchLab', 'DonorPermutationLab', 'CoalitionReferenceLab',
  'WineInferenceLab', 'QuestionRoutesFigure', 'XorSquareFigure', 'ArrivalOrderFigure',
  'SelectionProcedureFigure', 'TreeExplanationFigure']) {
  assert.ok(lessonSource.includes(`<${component} />`), `${component} is placed in the lesson`);
  record('placed component');
}
assert.equal((lessonSource.match(/<Practice /g) ?? []).length, 10, 'ten practice tasks');

/* ------------------------------- control guards and a prose claim */

// checkRange is the guard behind every control bound, and nothing tested it.
assert.equal(checkRange(5, limits.sensor, 'a sensor'), 5, 'an in-range value passes through unchanged');
refuses(() => checkRange(limits.sensor.maximum + 0.01, limits.sensor, 'a sensor'), 'a value above a control bound is refused');
refuses(() => checkRange(limits.sensor.minimum - 0.01, limits.sensor, 'a sensor'), 'a value below a control bound is refused');
refuses(() => checkRange(Number.NaN, limits.sensor, 'a sensor'), 'a non-finite value is refused by the range guard');
fourFieldModel.limits.forEach((bounds, index) => {
  refuses(() => checkRange(bounds.maximum + 1, bounds, `field ${index}`), `field ${index} refuses a value above its bound`);
  refuses(() => checkRange(bounds.minimum - 1, bounds, `field ${index}`), `field ${index} refuses a value below its bound`);
});
record('control-bound guard');

// Section 3 claims in prose that backward removal from the perfect pair would
// reject either one-column reduction under the same strict-improvement rule.
// Only forward selection is implemented, so the claim is asserted here from the
// lattice rather than left for the reader to confirm.
const perfectPair = xorLattice.subsets.find(entry => entry.mask === 3);
[1, 2].forEach(remaining => {
  const reduced = xorLattice.subsets.find(entry => entry.mask === remaining);
  assert(reduced.accuracy < perfectPair.accuracy,
    `removing a column from the pair drops accuracy from ${perfectPair.accuracy} to ${reduced.accuracy}`);
  record('backward removal is rejected by strict improvement');
});
assert.ok(lessonSource.includes('backward removal from the perfect pair would reject either one-column reduction'),
  'and the lesson makes exactly that claim');

/* --------------------------------------------------------------- record */

const sources = [
  'src/learn/data/selection-models.js',
  'src/learn/data/selection-data.js',
  'src/learn/data/selection-examples.js',
  'src/learn/components/lesson-labs/SelectionShared.jsx',
  'src/learn/components/lesson-labs/SelectionLabs.jsx',
  'src/learn/components/lesson-labs/SelectionFigures.jsx',
  'src/learn/components/lesson-labs/selection-labs.css',
  'src/learn/data/topics/feature-selection-importance-shap-permutation-mutual-info.jsx',
  'src/learn/data/curriculum/blueprints/feature-selection-importance-shap-permutation-mutual-info.js',
  'public/learn-assets/feature-selection/wine.data',
  'public/learn-assets/feature-selection/ATTRIBUTION.txt',
];
const hash = file => crypto.createHash('sha256').update(fs.readFileSync(file)).digest('hex');
const evidence = {
  checkedAt: new Date().toISOString(),
  sourceHashes: Object.fromEntries(sources.filter(fs.existsSync).map(file => [file, hash(file)])),
  verifierHash: hash('scripts/verify-selection-models.mjs'),
  packetHashes: {
    'lesson.md': hash(`${packetDir}/lesson.md`),
    'calculated-inputs.json': hash(`${packetDir}/calculated-inputs.json`),
    'wine.data': hash(`${packetDir}/wine.data`),
  },
  counts,
  totalGroupedChecks: Object.values(counts).reduce((sum, value) => sum + value, 0),
  // `totalGroupedChecks` counts record() calls, many of them inside loops, so it
  // is a count of checked GROUPS and not of assertions. `assertions` is the
  // honest assertion counter: every assert/close/vector call that ran.
  assertions,
  countsMeaning: 'totalGroupedChecks counts record() calls (groups, some inside loops); assertions counts individual assert/close/vector calls.',
  scope: 'Browser feature-selection models against the content phase\'s calculated-inputs.json and the manuscript\'s worked values: '
    + 'entropy and mutual information by two routes on the worked table, both contrasts, the practice fixture, a scale null, a degenerate '
    + 'target, an empty input row and 200 randomised tables checked for agreement, nonnegativity, the H(Y) bound and symmetry; mosaic tile '
    + 'areas equal to their cell probabilities; the XOR projections and joint bit and the exact-copy double count; the complete four-state '
    + 'lattice against the packet under two label sets, its tie rule and both search policies with their evaluated candidates kept separate '
    + 'from the accepted path; forward and RFE fit counts; the three-model permutation table against the packet with the practice donor, the '
    + 'identity-donor and unused-column nulls, grouped pairing, a negative increase and donor validation; the polynomial coalition games under '
    + 'two references and the practice case, cross-checked by the weighted formula, by the model layer\'s order averaging and by an independent '
    + 'permutation enumeration written in this file; conditional against replacement games on the dependent pair; waterfall segment endpoints '
    + 'and the cumulative-fraction contract; the saved tree\'s structure, its float32 threshold contract and its inference on all 38 inspection '
    + 'rows and all 100 background rows against the recorded native predictions; impurity importances recomputed and normalised; all twelve '
    + 'explained cases with their sixteen coalition values, four attributions, leaf tallies and reconstruction; both recorded input contrasts; '
    + 'the class-3 contrast reference; extrapolation flags; the practice arithmetic and the grouped-unanimity comparison; separate bar scales; '
    + 'and the generated data module\'s split accounting, published candidates, permutation records, served dataset bytes and control bounds.',
  limitations: [
    'The Wine study is a precomputed native record; the browser reproduces its saved tree and coalition games, not the fitting.',
    'Displayed program output is executed separately by scripts/verify-selection-examples.py.',
    'The data module is regenerated and matched to the packet separately by scripts/verify-selection-data.py.',
    'Rendering, interaction, narrow layouts and independent review are separate steps.',
  ],
  passed: true,
};
fs.mkdirSync('docs/teaching/evidence', { recursive: true });
fs.writeFileSync('docs/teaching/evidence/selection-models.json', JSON.stringify(evidence, null, 2) + '\n');
console.log(`PASS: ${evidence.assertions} assertions in ${evidence.totalGroupedChecks} recorded groups `
  + `across ${Object.keys(counts).length} named groups of feature-selection model checks.`);
