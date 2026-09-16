// Bounded independent checks of the feature-preparation browser models against
// the content phase's native calculations, the manuscript's worked values and
// analytic identities, plus structural checks of the generated data module.
//
// Read-only by default, so an independent reviewer can re-run it inside a
// no-write boundary. Pass --write to refresh the evidence file.
// Run: node scripts/verify-scaling-models.mjs [--write]
import assert from 'node:assert/strict';
import fs from 'node:fs';
import crypto from 'node:crypto';
import {
  applyScaler, boxCox, categoryDistances, circularCoordinates, crossFitEncoding, encodeCategory, fitMedianImputer,
  fitOneHot, fitPreparation, fitScaler, knnImpute, l2Normalize, median, nearestCandidate, overlapDistance, percentile,
  poolEstimates, rankCoordinates, scaledSquaredDistance, scalerComparison, signedHash, smoothedEncoding,
  transformRecord, yeoJohnson,
} from '../src/learn/data/scaling-models.js';
import {
  comparison, donorFixture, featureNames, fitted, heldOutPredictions, heldOutRows, heldOutTruth, provenance, rows,
  scaleFixture, speciesNames, split, targetEncodingFixture, transformedHeldOut,
} from '../src/learn/data/scaling-data.js';

const packet = 'docs/teaching/drafts/feature-scaling-encoding-imputation';
const authored = JSON.parse(fs.readFileSync(`${packet}/calculated-inputs.json`, 'utf8'));
const examples = JSON.parse(fs.readFileSync('src/learn/data/scaling-examples.js', 'utf8')
  .replace(/^[\s\S]*?export const scalingExamples = /, '').replace(/;\s*$/, ''));

const counts = {};
const record = name => { counts[name] = (counts[name] ?? 0) + 1; };
const close = (actual, expected, label, tolerance = 1e-9) =>
  assert(Math.abs(actual - expected) <= tolerance * Math.max(1, Math.abs(expected)), `${label}: ${actual} versus ${expected}`);

// ------------------------------------------------------------ fitted rulers
// A fitted scaler is an object, and applying it to a later value never consults
// the training column again.
const fixtureColumn = [1, 2, 3, 4, 100];
for (const kind of ['standard', 'minmax', 'robust']) {
  const ruler = fitScaler(kind, fixtureColumn);
  fixtureColumn.forEach((value, index) => close(applyScaler(ruler, value), scaleFixture[kind].values[index], `${kind} at ${value}`, 1e-12));
  close(applyScaler(ruler, 150), scaleFixture[kind].new150, `${kind} at the later 150`, 1e-12);
  authored.scale_fixture[kind].values.forEach((value, index) => close(scaleFixture[kind].values[index], value, `${kind} against the author`, 1e-12));
  close(scaleFixture[kind].new150, authored.scale_fixture[kind].new_150, `${kind} 150 against the author`, 1e-12);
  record('five-value scaler fixture');
}
close(fitScaler('standard', fixtureColumn).center, 22, 'the fixture mean');
close(fitScaler('standard', fixtureColumn).scale, 39.012818406262, 'the fixture population standard deviation', 1e-10);
assert.deepEqual(
  [fitScaler('minmax', fixtureColumn).low, fitScaler('minmax', fixtureColumn).high], [1, 100], 'the fitted min and max');
const robust = fitScaler('robust', fixtureColumn);
assert.deepEqual([robust.center, robust.first, robust.third, robust.scale], [3, 2, 4, 2], 'the fitted median and quartiles');
// The robust ruler keeps 1..4 apart and does not delete 100.
const separations = [1, 2, 3, 4].map(value => applyScaler(robust, value));
assert.deepEqual(separations, [-1, -0.5, 0, 0.5], 'robust scaling preserves the small values');
close(applyScaler(robust, 100), 48.5, 'the outlier stays 48.5 transformed units away');
assert(applyScaler(fitScaler('minmax', fixtureColumn), 150) > 1, 'a later value leaves the fitted [0,1] interval');
// The population divisor n, not n − 1.
close(fitScaler('standard', [2, 4, 6]).scale, Math.sqrt(8 / 3), 'practice 2 standard deviation', 1e-12);
close(applyScaler(fitScaler('standard', [2, 4, 6]), 8), Math.sqrt(6), 'practice 2 standardized new value', 1e-12);
close(applyScaler(fitScaler('minmax', [2, 4, 6]), 8), 1.5, 'practice 2 min–max value', 1e-12);
record('practice 2 fitted-once values');
// A constant training column divides by one instead of by zero.
const constant = fitScaler('standard', [7, 7, 7]);
assert.equal(constant.constant, true);
assert.deepEqual([applyScaler(constant, 7), applyScaler(constant, 9)], [0, 2], 'a later value need not become zero');
close(percentile([1, 2, 3, 4, 100], 25), 2, 'the linear percentile convention');
close(median([1, 2, 3, 4]), 2.5, 'an even-length median interpolates');
assert.equal(scalerComparison(fixtureColumn, [150]).rows.length, 6, 'the comparison keeps the later value separate');
assert.equal(scalerComparison(fixtureColumn, [150]).rows[5].training, false);
assert.throws(() => fitScaler('quantile', fixtureColumn), RangeError);
assert.throws(() => fitScaler('standard', []), RangeError);
assert.throws(() => applyScaler({ kind: 'nonsense' }, 1), RangeError);
record('scaler identities and validation');

// ------------------------------------------------------------ the ruler chooses
const query = [40, 4000];
const candidates = [{ name: 'A', point: [41, 4100] }, { name: 'B', point: [43, 4001] }];
const raw = nearestCandidate(query, candidates, [1, 1]);
close(raw.scored[0].total, 10001, 'A under raw units', 1e-12);
close(raw.scored[1].total, 10, 'B under raw units', 1e-12);
assert.equal(raw.winner, 'B');
const scaled = nearestCandidate(query, candidates, [1, 100]);
close(scaled.scored[0].total, 2, 'A under 1 mm / 100 g', 1e-12);
close(scaled.scored[1].total, 9.0001, 'B under 1 mm / 100 g', 1e-12);
assert.equal(scaled.winner, 'A');
// The specification's contrast and its two nulls.
const heavier = nearestCandidate(query, [{ name: 'A', point: [41, 4400] }, candidates[1]], [1, 100]);
close(heavier.scored[0].total, 17, 'A with mass 4,400 g', 1e-12);
assert.equal(heavier.winner, 'B');
const doubled = nearestCandidate(query, candidates, [2, 200]);
close(doubled.scored[0].total, 0.5, 'A with both divisors doubled', 1e-12);
close(doubled.scored[1].total, 2.250025, 'B with both divisors doubled', 1e-12);
assert.equal(doubled.winner, 'A', 'a common positive rescaling leaves the ranking alone');
const shifted = nearestCandidate([140, 5000], candidates.map(candidate => ({
  name: candidate.name, point: [candidate.point[0] + 100, candidate.point[1] + 1000],
})), [1, 100]);
assert.equal(shifted.winner, 'A', 'translating everything by one vector changes nothing');
shifted.scored.forEach((row, index) => close(row.total, scaled.scored[index].total, 'translated totals', 1e-12));
assert.equal(nearestCandidate([0, 0], [{ name: 'A', point: [1, 0] }, { name: 'B', point: [0, 1] }], [1, 1]).winner, 'tie');
// Practice 1.
const practiceOne = [{ name: 'A', point: [2, 60] }, { name: 'B', point: [5, 10] }];
close(nearestCandidate([0, 0], practiceOne, [1, 1]).scored[0].total, 3604, 'practice 1 raw A', 1e-12);
close(nearestCandidate([0, 0], practiceOne, [1, 1]).scored[1].total, 125, 'practice 1 raw B', 1e-12);
assert.equal(nearestCandidate([0, 0], practiceOne, [1, 1]).winner, 'B');
close(nearestCandidate([0, 0], practiceOne, [1, 30]).scored[0].total, 8, 'practice 1 scaled A', 1e-12);
close(nearestCandidate([0, 0], practiceOne, [1, 30]).scored[1].total, 25 + 1 / 9, 'practice 1 scaled B', 1e-12);
assert.equal(nearestCandidate([0, 0], practiceOne, [1, 30]).winner, 'A');
record('practice 1 ruler flip');
// Centring cancels in a difference; dividing does not.
const centred = scaledSquaredDistance([40 - 43.93, 4000 - 4190.99], [41 - 43.93, 4100 - 4190.99], [1, 1]);
close(centred.total, scaledSquaredDistance(query, candidates[0].point, [1, 1]).total, 'a shared centre cancels', 1e-9);
assert.throws(() => scaledSquaredDistance(query, candidates[0].point, [1, 0]), RangeError, 'a zero divisor is refused');
assert.throws(() => scaledSquaredDistance(query, candidates[0].point, [1, -100]), RangeError);
assert.throws(() => scaledSquaredDistance(query, [1, 2, 3], [1, 1]), RangeError);
record('distance under a chosen ruler');

// --------------------------------------------------------- row normalization
const shortRow = l2Normalize([3, 4]);
assert.deepEqual(shortRow.values, [0.6, 0.8]);
close(shortRow.norm, 5, 'its length');
close(l2Normalize([6, 8]).values[0], 0.6, 'a three-times-longer row has the same direction', 1e-12);
const spectrum = l2Normalize([2, 1, 2]);
close(spectrum.norm, 3, 'practice 3 first length', 1e-12);
close(l2Normalize([6, 3, 6]).norm, 9, 'practice 3 second length', 1e-12);
[0, 1, 2].forEach(index => close(l2Normalize([6, 3, 6]).values[index], spectrum.values[index], 'both map to the same direction', 1e-12));
close(spectrum.values[0], 2 / 3, 'the normalized coordinate', 1e-12);
assert.deepEqual(l2Normalize([0, 0]).values, [0, 0], 'a zero vector has no direction and stays zero');
record('practice 3 discarded magnitude');
record('row normalization');

// ---------------------------------------------------------- category geometry
const colours = fitOneHot(['red', 'green', 'blue']);
assert.deepEqual(colours.categories, ['blue', 'green', 'red'], 'the fitted vocabulary is sorted');
const full = categoryDistances(colours, { includeUnknown: false });
assert.equal(full.pairs.length, 3);
full.pairs.forEach(pair => close(pair.squared, 2, `${pair.from} to ${pair.to}`, 1e-12));
const dropped = categoryDistances(colours, { drop: 'red', includeUnknown: false });
const distanceOf = (set, from, to) => set.pairs.find(pair => (pair.from === from && pair.to === to) || (pair.from === to && pair.to === from));
close(distanceOf(dropped, 'red', 'green').squared, 1, 'red becomes one unit from green', 1e-12);
close(distanceOf(dropped, 'green', 'blue').squared, 2, 'green and blue stay √2 apart', 1e-12);
assert.deepEqual(encodeCategory(colours, 'red', { drop: 'red' }).vector, [0, 0], 'the reference corner is the origin');
// Practice 4: an unknown value against the fitted vocabulary.
const sizes = fitOneHot(['small', 'medium', 'large']);
const unknown = encodeCategory(sizes, 'enormous');
assert.deepEqual(unknown.vector, [0, 0, 0]);
assert.equal(unknown.state, 'unknown');
const small = encodeCategory(sizes, 'small');
close(Math.sqrt(small.vector.reduce((sum, value, index) => sum + (value - unknown.vector[index]) ** 2, 0)), 1, 'unknown to small is 1', 1e-12);
close(Math.sqrt(2), Math.sqrt(distanceOf(categoryDistances(sizes, { includeUnknown: false }), 'small', 'medium').squared), 'small to medium is √2', 1e-12);
assert.throws(() => encodeCategory(sizes, 'enormous', { handleUnknown: 'error' }), RangeError);
assert.throws(() => encodeCategory(sizes, 'small', { drop: 'gigantic' }), RangeError);
assert.throws(() => fitOneHot(['a', null]), RangeError, 'a missing category needs a declared fill before fitting');
const withFill = fitOneHot(['a', null], { fillMissing: 'not_recorded' });
assert.deepEqual(withFill.categories, ['a', 'not_recorded']);
assert.equal(encodeCategory(withFill, null).state, 'missing');
assert.deepEqual(encodeCategory(withFill, null).vector, [0, 1], 'an absent value takes the fitted fill coordinate');
record('practice 4 unknown against a fitted vocabulary');
record('one-hot geometry and the three category states');

// ------------------------------------------------------------ donor overlap
const donors = donorFixture.donors;
const donorQuery = donorFixture.query;
const base = knnImpute({ donors, donorNames: donorFixture.donorNames, query: donorQuery, target: 2, neighbours: 2 });
close(base.rows[0].squared, 7.5, 'D1 adjusted squared distance', 1e-12);
close(base.rows[1].squared, 3, 'D2 adjusted squared distance', 1e-12);
close(base.rows[2].squared, 12, 'D3 adjusted squared distance', 1e-12);
assert.deepEqual(base.rows.map(row => row.q), [2, 1, 1], 'jointly observed counts');
assert.deepEqual(base.selected.map(row => row.name), ['D2', 'D1'], 'the two nearest eligible donors');
close(base.estimate, 200, 'the imputed value', 1e-12);
assert.equal(base.mode, 'neighbours');
close(base.estimate, donorFixture.estimate, 'against the recorded KNNImputer result', 1e-12);
close(base.estimate, authored.knn_imputed[0][2], 'against the author calculation', 1e-12);
// D2 donates although one of its own features is absent.
assert.equal(base.rows[1].eligible, true);
// Practice 5 and the specification's contrasts.
const withoutTarget = knnImpute({
  donors: [donors[0], [3, null, null], donors[2]], donorNames: donorFixture.donorNames, query: donorQuery, target: 2, neighbours: 2,
});
assert.deepEqual(withoutTarget.selected.map(row => row.name), ['D1', 'D3']);
close(withoutTarget.estimate, 300, 'with D2 ineligible', 1e-12);
const movedTarget = knnImpute({
  donors: [[1, 10, 140], donors[1], donors[2]], donorNames: donorFixture.donorNames, query: donorQuery, target: 2, neighbours: 2,
});
close(movedTarget.estimate, 220, 'with D1 c changed to 140', 1e-12);
assert.deepEqual(movedTarget.selected.map(row => row.name), ['D2', 'D1'], 'the overlap distances did not move');
const unusedEdit = knnImpute({
  donors: [donors[0], donors[1], [null, 14, 900]], donorNames: donorFixture.donorNames, query: donorQuery, target: 2, neighbours: 2,
});
close(unusedEdit.estimate, 200, 'editing an unselected donor is a null', 1e-12);
record('practice 5 changed donors');
const noOverlap = knnImpute({ donors, donorNames: donorFixture.donorNames, query: [null, null, null], target: 2, neighbours: 2 });
assert.equal(noOverlap.mode, 'fallback-mean');
close(noOverlap.estimate, 300, 'the fallback is the observed donor column mean', 1e-12);
const emptyColumn = knnImpute({
  donors: [[1, 10, null], [3, null, null], [null, 14, null]], donorNames: donorFixture.donorNames, query: donorQuery, target: 2, neighbours: 2,
});
assert.equal(emptyColumn.mode, 'no-target-column');
assert.equal(emptyColumn.estimate, null, 'an entirely absent target column is reported, never filled with zero');
assert.equal(overlapDistance([null, null, null], donors[0]).defined, false, 'no overlap gives no distance, not zero');
assert.equal(overlapDistance([null, null, null], donors[0]).squared, null);
assert.throws(() => knnImpute({ donors, query: [2, 12, 5], target: 2 }), RangeError, 'the target cell must be absent');
assert.throws(() => knnImpute({ donors, query: donorQuery, target: 7 }), RangeError);
// A donor tie is broken by source order, and that is stated rather than hidden.
const tied = knnImpute({
  donors: [[1, 12, 10], [3, 12, 20], [2, 11, 30]], donorNames: ['E1', 'E2', 'E3'], query: [2, 12, null], target: 2, neighbours: 2,
});
assert(tied.rows.every(row => Math.abs(row.squared - 1.5) < 1e-12), 'all three donors sit at the same distance');
assert.deepEqual(tied.selected.map(row => row.name), ['E1', 'E2'], 'equal distances keep source order');
record('nan-aware donor eligibility');

// ---------------------------------------------- fit on training rows, only
const trainingRows = rows.filter(row => row[7] === 0);
const heldOut = rows.filter(row => row[7] === 1);
assert.equal(trainingRows.length, split.training);
assert.equal(heldOut.length, split.heldOut);
assert.equal(rows.length, 344);
const numericOf = row => [row[2], row[3], row[4], row[5]];
const sexOf = row => (row[6] === null ? null : ['female', 'male'][row[6]]);
const preparation = fitPreparation({
  numericRows: trainingRows.map(numericOf),
  categoryValues: trainingRows.map(sexOf),
  scaler: 'standard',
});
preparation.imputer.medians.forEach((value, index) => close(value, fitted.medians[index], `refitted median ${index}`, 1e-12));
assert.deepEqual(preparation.imputer.medians, [45, 17.3, 197, 4000], 'the manuscript states these medians');
preparation.scalers.forEach((ruler, index) => {
  close(ruler.center, fitted.standard.center[index], `refitted centre ${index}`, 1e-9);
  close(ruler.scale, fitted.standard.scale[index], `refitted scale ${index}`, 1e-9);
});
assert.deepEqual(preparation.vocabulary.categories, ['female', 'male', 'not_recorded'], 'the fitted vocabulary');
// The leakage check: the same fit over all 344 rows is a different ruler, and
// the frozen one is the one the page uses.
const leaked = fitPreparation({
  numericRows: rows.map(numericOf), categoryValues: rows.map(sexOf), scaler: 'standard',
});
assert(Math.abs(leaked.scalers[3].center - preparation.scalers[3].center) > 1,
  'fitting on every row would move the mass centre, which is why the fit uses training rows alone');
assert.notDeepEqual(leaked.imputer.medians, preparation.imputer.medians, 'and it would move a median too');
// Apply the frozen fit to all 86 held-out rows.
const heldOutOrdered = split.heldOutOrder.map(source => rows[source]);
heldOutOrdered.forEach((row, position) => {
  const result = transformRecord(preparation, { numeric: numericOf(row), category: sexOf(row) });
  result.coordinates.forEach((value, column) => close(value, transformedHeldOut[position][column],
    `held-out row ${row[0]} coordinate ${featureNames[column]}`, 1e-9));
});
record('the fitted preparation reproduces all 86 transformed held-out rows');
const firstRow = heldOutOrdered[0];
assert.equal(firstRow[0], 309, 'the first held-out row is source row 309');
assert.deepEqual(numericOf(firstRow), [51, 18.8, 203, 4100]);
const firstResult = transformRecord(preparation, { numeric: numericOf(firstRow), category: sexOf(firstRow) });
[1.3091, 0.8773, 0.1645, -0.1128, 0, 1, 0].forEach((value, index) =>
  close(Number(firstResult.coordinates[index].toFixed(4)), value, `the manuscript's coordinate ${index}`, 1e-12));
assert(firstResult.coordinates[3] < 0, 'a mass below the fitted mean gives a negative coordinate, not a negative mass');
// Practice 6 and the specification's I3 contrasts.
const missingMass = transformRecord(preparation, { numeric: [51, 18.8, 203, null], category: 'male' });
assert.equal(missingMass.numeric[3].wasMissing, true);
close(missingMass.numeric[3].filled, 4000, 'the fitted median fills the absent cell', 1e-12);
close(missingMass.coordinates[3], (4000 - fitted.standard.center[3]) / fitted.standard.scale[3], 'practice 6', 1e-12);
close(missingMass.coordinates[3], -0.236756305829395, 'practice 6 to full precision', 1e-12);
record('practice 6 a real record with a deleted measurement');
[0, 1, 2, 4, 5, 6].forEach(index => close(missingMass.coordinates[index], firstResult.coordinates[index], `coordinate ${index} is unmoved`, 1e-12));
const asFemale = transformRecord(preparation, { numeric: numericOf(firstRow), category: 'female' });
assert.deepEqual(asFemale.coordinates.slice(4), [1, 0, 0]);
const asUnknown = transformRecord(preparation, { numeric: numericOf(firstRow), category: 'juvenile' });
assert.deepEqual(asUnknown.coordinates.slice(4), [0, 0, 0], 'an unknown value is all zeros across the block');
assert.equal(asUnknown.category.state, 'unknown');
const asAbsent = transformRecord(preparation, { numeric: numericOf(firstRow), category: null });
assert.deepEqual(asAbsent.coordinates.slice(4), [0, 0, 1], 'an absent value takes the fitted not_recorded coordinate');
assert.equal(asAbsent.category.state, 'missing');
// Editing a later record never moves the fitted object.
const before = JSON.stringify(preparation);
transformRecord(preparation, { numeric: [80, 25, 250, 6500], category: 'juvenile' });
assert.equal(JSON.stringify(preparation), before, 'transforming does not refit');
assert.throws(() => transformRecord({}, { numeric: [1], category: 'a' }), RangeError);
assert.throws(() => fitMedianImputer([[null], [null]]), RangeError, 'an entirely absent training column has no median to take');
record('the fit/transform boundary in code');

// The other two rulers the comparison uses, refitted the same frozen way.
for (const kind of ['minmax', 'robust']) {
  const other = fitPreparation({ numericRows: trainingRows.map(numericOf), categoryValues: trainingRows.map(sexOf), scaler: kind });
  other.scalers.forEach((ruler, index) => {
    close(ruler.center, fitted[kind].center[index], `${kind} centre ${index}`, 1e-9);
    close(ruler.scale, fitted[kind].scale[index], `${kind} scale ${index}`, 1e-9);
  });
  record(`${kind} ruler refitted from the training rows`);
}

// ------------------------------------------------------------ target encoding
const encoding = crossFitEncoding({
  categories: targetEncodingFixture.categories,
  target: targetEncodingFixture.target,
  folds: targetEncodingFixture.folds,
  smoothing: 2,
});
[5 / 9, 7 / 9, 2 / 9, 4 / 9, 2 / 9, 7 / 9].forEach((value, index) => close(encoding.encoded[index], value, `base encoding row ${index}`, 1e-12));
encoding.encoded.forEach((value, index) => close(value, targetEncodingFixture.encoded[index], `against the author, row ${index}`, 1e-12));
close(encoding.priors[0], 1 / 3, 'fold 0 prior from fold 1 donors', 1e-12);
close(encoding.priors[1], 2 / 3, 'fold 1 prior from fold 0 donors', 1e-12);
assert.deepEqual(encoding.rows[0].donors, [1, 3, 5], 'row 0 is encoded from the other fold only');
assert.deepEqual(encoding.rows[0].matching, [1], 'and from the same-category donor inside it');
assert(!encoding.rows[0].donors.includes(0), 'a row never donates to its own encoding');
const changed = crossFitEncoding({
  categories: targetEncodingFixture.categories,
  target: targetEncodingFixture.changedTargetRowZero,
  folds: targetEncodingFixture.folds,
  smoothing: 2,
});
[5 / 9, 2 / 9, 2 / 9, 2 / 9, 2 / 9, 5 / 9].forEach((value, index) => close(changed.encoded[index], value, `changed encoding row ${index}`, 1e-12));
close(changed.encoded[0], encoding.encoded[0], 'row 0 own-target edit is a null for row 0', 1e-12);
assert(Math.abs(changed.encoded[2] - encoding.encoded[2]) < 1e-12, 'fold 0 rows are untouched');
assert(Math.abs(changed.encoded[3] - encoding.encoded[3]) > 1e-9, 'B moves although no B target changed: the prior changed');
// Practice 7.
const practiceSeven = crossFitEncoding({
  categories: targetEncodingFixture.categories, target: [1, 1, 0, 1, 1, 0], folds: targetEncodingFixture.folds, smoothing: 2,
});
close(practiceSeven.encoded[0], 7 / 9, 'practice 7 row 0', 1e-12);
close(practiceSeven.encoded[3], 4 / 9, 'practice 7 row 3 is unchanged', 1e-12);
record('practice 7 prior-mediated change');
// The declared zero-smoothing fallback instead of a zero divide.
const noSmoothing = crossFitEncoding({ categories: ['A', 'B', 'A', 'C'], target: [1, 0, 1, 0], folds: [0, 0, 1, 1], smoothing: 0 });
assert.equal(noSmoothing.rows[3].usedPriorFallback, true, 'an unseen category falls back to the fold prior');
close(noSmoothing.rows[3].value, noSmoothing.priors[1], 'and takes exactly that prior', 1e-12);
close(smoothedEncoding(1, 1, 0.5, 2).value, 2 / 3, 'one positive example with α = 2 and μ = .5', 1e-12);
close(smoothedEncoding(1, 1, 0.5, 0).value, 1, 'without smoothing it is the row target itself', 1e-12);
assert.throws(() => crossFitEncoding({ categories: ['A', 'B'], target: [1, 0], folds: [0, 0] }), RangeError, 'both folds must be non-empty');
assert.throws(() => crossFitEncoding({ categories: ['A'], target: [0.5], folds: [0, 1] }), RangeError);
record('cross-fitted target encoding');

// ------------------------------------------------------------------- hashing
const hashMap = { apple: { bucket: 0, sign: 1 }, pear: { bucket: 0, sign: -1 }, banana: { bucket: 1, sign: 1 } };
assert.deepEqual(signedHash({ apple: 3, pear: 1, banana: 2 }, hashMap, 2).vector, [2, 2]);
assert.deepEqual(signedHash({ apple: 2, banana: 2 }, hashMap, 2).vector, [2, 2], 'two different bags collide onto one vector');
assert.throws(() => signedHash({ cherry: 1 }, hashMap, 2), RangeError);
record('signed feature hashing');

// --------------------------------------------------- nonlinear and rank maps
assert.equal(yeoJohnson(3, 1).value, 3, 'λ = 1 is the identity on the nonnegative side');
assert.equal(yeoJohnson(-3, 1).value, -3, 'and on the negative side');
close(yeoJohnson(3, 0).value, Math.log(4), 'λ = 0 uses log1p for nonnegative input', 1e-12);
close(yeoJohnson(-3, 2).value, -Math.log(4), 'λ = 2 uses a logarithm for negative input, not a square root', 1e-12);
assert.match(yeoJohnson(-3, 0).branch, /2 − λ/, 'λ = 0 sends negative input to the quadratic branch');
close(yeoJohnson(-3, 0).value, -((4 ** 2 - 1) / 2), 'the negative branch at λ = 0', 1e-12);
record('practice 8 Yeo–Johnson branches');
close(boxCox(100, 0), Math.log(100), 'Box–Cox at λ = 0', 1e-12);
assert.throws(() => boxCox(0, 0.5), RangeError, 'Box–Cox needs strictly positive input');
assert.throws(() => boxCox(-2, 1), RangeError);
const ranks = rankCoordinates([1, 2, 3, 4, 100]);
assert.deepEqual(ranks.map(row => row.coordinate), [0, 0.25, 0.5, 0.75, 1], 'the explicit rank convention');
close(ranks[4].log, Math.log(100), 'the log coordinate', 1e-12);
const bigger = rankCoordinates([1, 2, 3, 4, 1000]);
assert.equal(bigger[4].coordinate, 1, 'replacing 100 by 1,000 is a null for the rank');
close(bigger[4].log, Math.log(1000), 'and a contrast for the log', 1e-12);
assert(Math.abs(bigger[4].log - ranks[4].log) > 2, 'the two representations preserve different information');
// log turns equal ratios into equal differences.
close(Math.log(100) - Math.log(10), Math.log(10) - Math.log(1), 'equal multiplicative steps', 1e-12);
const near = [circularCoordinates(1), circularCoordinates(359)];
const far = [circularCoordinates(1), circularCoordinates(181)];
const chord = ([a, b]) => Math.hypot(a.cos - b.cos, a.sin - b.sin);
assert(chord(near) < 0.05 && chord(far) > 1.99, '359° is near 1°, and 181° is not');
close(circularCoordinates(180).radians, Math.PI, 'degrees convert to radians', 1e-12);
record('nonlinear, rank and circular representations');

// ------------------------------------------------------------------- pooling
const pooled = poolEstimates([9, 10, 11], [4, 4, 4]);
close(pooled.mean, 10, 'the pooled estimate', 1e-12);
close(pooled.within, 4, 'within-analysis variance', 1e-12);
close(pooled.between, 1, 'between-completion variance', 1e-12);
close(pooled.total, 16 / 3, 'total variance', 1e-12);
close(pooled.standardError, Math.sqrt(16 / 3), 'its standard error', 1e-12);
assert(pooled.standardError > 2, 'larger than treating the completion as certain');
close(pooled.standardError, 2.309401076758503, 'about 2.309', 1e-12);
const practiceNine = poolEstimates([8, 10, 10, 12], [1, 1, 1, 1]);
close(practiceNine.between, 8 / 3, 'practice 9 between variance', 1e-12);
close(practiceNine.total, 13 / 3, 'practice 9 total variance', 1e-12);
close(practiceNine.standardError, Math.sqrt(13 / 3), 'practice 9 standard error', 1e-12);
record('practice 9 pooling four analyses');
assert.throws(() => poolEstimates([9], [4]), RangeError, 'pooling needs at least two completions');
assert.throws(() => poolEstimates([9, 10], [4, -1]), RangeError);
record('multiple-imputation pooling');

// ----------------------------------------------- the generated data module
assert.equal(provenance.sha256, authored.data_sha256, 'the module carries the checkpointed hash');
assert.equal(provenance.sha256, crypto.createHash('sha256').update(fs.readFileSync(`${packet}/penguins.csv`)).digest('hex'));
assert.equal(provenance.sha256, crypto.createHash('sha256').update(fs.readFileSync('public/learn-assets/feature-scaling/penguins.csv')).digest('hex'),
  'the served CSV is the same bytes the packet checked');
assert.deepEqual(provenance.missing, authored.missing_counts);
assert.equal(provenance.licence, 'CC0');
assert.equal(heldOutRows, 86);
assert.equal(featureNames.length, 7);
assert.deepEqual(featureNames.slice(4), ['category__sex_female', 'category__sex_male', 'category__sex_not_recorded']);
assert.equal(transformedHeldOut.length, 86);
assert.equal(heldOutTruth.length, 86);
// Two rows lack every measurement, eleven lack recorded sex, and both are
// carried through rather than dropped.
assert.equal(rows.filter(row => numericOf(row).every(value => value === null)).length, 2);
assert.equal(rows.filter(row => row[6] === null).length, 11);
assert.equal(rows.filter(row => row[2] === null).length, 2);
const counted = Object.fromEntries(comparison.map(row => [row.method, row.correct]));
assert.deepEqual(counted, { majority: 38, raw: 67, standard: 84, minmax: 85, robust: 85 }, "the manuscript's table");
comparison.forEach(row => {
  close(row.accuracy, row.correct / 86, `${row.method} accuracy`, 1e-12);
  if (!row.confusion) return;
  assert.equal(row.confusion.length, 3);
  assert.equal(row.confusion.flat().reduce((sum, value) => sum + value, 0), 86, `${row.method} confusion totals 86`);
  assert.equal(row.confusion.reduce((sum, line, index) => sum + line[index], 0), row.correct, `${row.method} diagonal`);
  row.confusion.forEach((line, index) => assert.equal(
    line.reduce((sum, value) => sum + value, 0), heldOutTruth.filter(value => value === index).length,
    `${row.method} row ${index} totals the true count`));
});
assert.equal(comparison.find(row => row.method === 'majority').confusion, null);
Object.entries(heldOutPredictions).forEach(([method, predicted]) => {
  assert.equal(predicted.length, 86);
  const correct = predicted.filter((value, index) => value === heldOutTruth[index]).length;
  assert.equal(correct, counted[method], `${method} predictions agree with the recorded count`);
});
assert.equal(heldOutPredictions.standard.filter((value, index) => value !== heldOutTruth[index]).length, 2,
  'standard scaling misclassified two rows here');
assert.equal(heldOutPredictions.raw.filter((value, index) => value !== heldOutTruth[index]).length, 19,
  'the raw model misclassified nineteen');
assert.deepEqual(speciesNames, ['Adelie', 'Chinstrap', 'Gentoo']);
assert.equal(new Set(split.heldOutOrder).size, 86, 'the held-out list has no repeats');
assert(split.heldOutOrder.every(source => rows[source][7] === 1), 'every listed source row is marked held out');
record('generated data module structure');

// --------------------------------------------------- the displayed programs
assert.equal(Object.keys(examples).length, 2);
assert(examples.penguinExperiment.expected.includes('majority 38 / 86'));
comparison.filter(row => row.method !== 'majority').forEach(row => assert(
  examples.penguinExperiment.expected.includes(`${row.method} ${row.correct} / 86 ${row.accuracy.toFixed(4)}`),
  `the displayed output shows ${row.method}`));
assert(examples.penguinExperiment.expected.includes('[[ 1.3091  0.8773  0.1645 -0.1128  0.      1.      0.    ]]'));
featureNames.forEach(name => assert(examples.penguinExperiment.expected.includes(name), `the displayed output names ${name}`));
assert(examples.penguinExperiment.code.includes('Path(__file__).with_name("penguins.csv")'), 'the program reads the served CSV');
assert(examples.penguinExperiment.code.includes('keep_empty_features=True'));
assert(examples.crossFitEncoding.expected.split('\n').length === 2, 'two printed arrays');
targetEncodingFixture.encoded.forEach(value => assert(
  examples.crossFitEncoding.expected.includes(value.toFixed(6)), 'the displayed array carries every encoded value'));
record('displayed programs');

// --------------------------------------------------------------------- record
const sources = [
  'src/learn/data/scaling-models.js',
  'src/learn/data/scaling-data.js',
  'src/learn/data/scaling-examples.js',
  'src/learn/components/lesson-labs/ScalingShared.jsx',
  'src/learn/components/lesson-labs/ScalingLabs.jsx',
  'src/learn/components/lesson-labs/ScalingFigures.jsx',
  'src/learn/components/lesson-labs/scaling-labs.css',
  'src/learn/data/topics/feature-scaling-encoding-imputation.jsx',
  'src/learn/data/curriculum/blueprints/feature-scaling-encoding-imputation.js',
  'public/learn-assets/feature-scaling/penguins.csv',
];
const hash = file => crypto.createHash('sha256').update(fs.readFileSync(file)).digest('hex');
const evidence = {
  checkedAt: new Date().toISOString(),
  sourceHashes: Object.fromEntries(sources.filter(fs.existsSync).map(file => [file, hash(file)])),
  verifierHash: hash('scripts/verify-scaling-models.mjs'),
  counts,
  totalGroupedChecks: Object.values(counts).reduce((sum, value) => sum + value, 0),
  scope: 'Browser preparation models against the content phase’s recorded calculations, the manuscript’s worked values and independent identities: the three fitted rulers on the five-value column and the later 150, the Q/A/B distance fixture with its contrast and both nulls, row normalization, one-hot geometry with a dropped reference and an unknown block, nan-aware donor eligibility including the tie, fallback and absent-column cases, a complete refit of the mixed-column preparation from the 258 training rows that reproduces all 86 transformed held-out rows and disagrees with a fit over all 344, cross-fitted target encoding with its own-target null and prior-mediated contrast, signed hashing collisions, the Yeo–Johnson branches, the rank-versus-log contrast, Rubin pooling, and the generated data module’s structure, counts and provenance hash.',
  limitations: [
    'The penguin comparison counts are precomputed native fits; the browser reproduces the fitted preparation and the transformed rows, not the neighbour classification.',
    'Displayed program output is executed separately by scripts/verify-scaling-examples.py.',
    'Rendering, interaction and independent review are separate steps.',
  ],
  passed: true,
};
const evidencePath = 'docs/teaching/evidence/scaling-models.json';
if (process.argv.includes('--write')) {
  fs.mkdirSync('docs/teaching/evidence', { recursive: true });
  fs.writeFileSync(evidencePath, JSON.stringify(evidence, null, 2) + '\n');
} else if (fs.existsSync(evidencePath)) {
  // The recorded evidence must still describe these bytes; only the timestamp
  // is allowed to differ.
  const recordedEvidence = JSON.parse(fs.readFileSync(evidencePath, 'utf8'));
  assert.deepEqual(recordedEvidence.sourceHashes, evidence.sourceHashes, 'recorded evidence describes different sources; re-run with --write');
  assert.deepEqual(recordedEvidence.counts, evidence.counts, 'recorded check counts differ; re-run with --write');
}
console.log(`PASS: ${evidence.totalGroupedChecks} grouped feature-preparation model checks${process.argv.includes('--write') ? ', evidence rewritten' : ' (read-only)'}.`);
