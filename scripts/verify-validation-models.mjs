/** Model checks for the cross-validation and hyperparameter-tuning lesson.
 *
 * Nothing here imports the author's Python. Each group recomputes the answer a
 * second way — brute force, exhaustive enumeration, an independent closed form
 * — and then compares it with the published module, with the retained packet
 * calculation and with the sentences the manuscript prints.
 */
import assert from 'node:assert/strict';
import fs from 'node:fs';
import crypto from 'node:crypto';
import {
  bootstrapDistinctShare, consecutiveFolds, drawsForHitProbability, enumerateSelection, expectedImprovement,
  fitBudget, fixedRuleAccuracyVariance, foldMeanVariance, foldPlan, futureAccuracy, hitProbability,
  meanPredictorRisk, nearestNeighborRun, nestedTrace, outOfFoldCoverage, scoreCandidates, successiveHalving,
  tpeAcquisition,
} from '../src/learn/data/validation-models.js';
import { NESTED_EXPERIMENT, PENGUIN_SOURCE, RUNTIME_VERSIONS } from '../src/learn/data/validation-data.js';

const packet = JSON.parse(fs.readFileSync('docs/teaching/drafts/cross-validation-hyperparameter-tuning/calculated-inputs.json', 'utf8'));

let assertions = 0;
const groups = [];
const close = (actual, expected, tolerance = 1e-12, message = '') => {
  assertions += 1;
  assert.ok(Math.abs(actual - expected) <= tolerance, `${message} ${actual} differs from ${expected} by ${Math.abs(actual - expected)}`);
};
const equal = (actual, expected, message) => { assertions += 1; assert.deepEqual(actual, expected, message); };
const truthy = (value, message) => { assertions += 1; assert.ok(value, message); };
const rejects = (fn, message) => { assertions += 1; assert.throws(fn, RangeError, message); };
const checked = (name, fn) => { fn(); groups.push(name); };

/* ---------------------------------------------------------------- */
checked('Fold plans: remainder preservation, exactly-once partition, grouping leaks, stratification and refusals', () => {
  // np.array_split semantics, recomputed by a different route.
  for (let n = 2; n <= 40; n += 1) {
    for (let k = 2; k <= n; k += 1) {
      const folds = consecutiveFolds(n, k);
      const sizes = folds.map(fold => fold.length);
      equal(folds.flat(), Array.from({ length: n }, (_, index) => index), `every row keeps a turn at n=${n}, k=${k}`);
      equal(sizes.length, k);
      truthy(Math.max(...sizes) - Math.min(...sizes) <= 1, 'fold sizes differ by at most one');
      truthy(sizes.every(size => size >= 1), 'no fold is empty');
      equal(sizes.filter(size => size === Math.ceil(n / k)).length, n % k === 0 ? k : n % k, 'the remainder is distributed, never dropped');
    }
  }
  equal(consecutiveFolds(7, 3).map(fold => fold.length), [3, 2, 2]);
  equal(consecutiveFolds(11, 3).map(fold => fold.length), [4, 4, 3], 'practice 2: n=11, k=3 gives 4, 4, 3');
  rejects(() => consecutiveFolds(5, 1));
  rejects(() => consecutiveFolds(5, 6));
  rejects(() => consecutiveFolds(5.5, 2));

  const rows = [0, 1, 2, 3, 4, 5, 6];
  const plan = foldPlan({ rows, folds: consecutiveFolds(7, 3), labelOf: id => (id >= 3 ? 1 : 0), name: 'tiny' });
  equal(plan.k, 3);
  equal(plan.folds.map(fold => fold.validation), [[0, 1, 2], [3, 4], [5, 6]]);
  equal(plan.folds.map(fold => fold.training), [[3, 4, 5, 6], [0, 1, 2, 5, 6], [0, 1, 2, 3, 4]]);
  equal(plan.folds.map(fold => fold.trainingSize), [4, 5, 5]);
  truthy(plan.coverage.partition, 'the consecutive plan is a partition');
  equal(plan.membership, { 0: 0, 1: 0, 2: 0, 3: 1, 4: 1, 5: 2, 6: 2 });
  // Ordering by class leaves a fold with no positive example at all.
  equal(plan.stratification.missingClass, [{ fold: 0, class: 1 }, { fold: 1, class: 0 }, { fold: 2, class: 0 }]);
  equal(plan.stratification.overall, { 0: 3, 1: 4 });
  plan.folds.forEach(fold => fold.validation.forEach(id => truthy(!fold.training.includes(id), 'a held-out row never trains its own fit')));

  // Groups: whole-person holdout is clean; a later-day split deliberately is not.
  const records = ['A1', 'A2', 'A3', 'B1', 'B2', 'B3'];
  const person = id => id[0];
  const unseenPerson = foldPlan({ rows: records, folds: [['B1', 'B2', 'B3']], groupOf: person, requirePartition: false });
  truthy(unseenPerson.grouping.clean, 'holding out a whole person leaves no group on both sides');
  equal(unseenPerson.grouping.groups, ['A', 'B']);
  const laterDay = foldPlan({ rows: records, folds: [['A3', 'B3']], groupOf: person, requirePartition: false });
  equal(laterDay.grouping.spanning, [{ fold: 0, group: 'A' }, { fold: 0, group: 'B' }]);
  truthy(!laterDay.grouping.clean, 'a later-records split shares both people across the boundary');
  equal(laterDay.coverage.unassessed, ['A1', 'A2', 'B1', 'B2']);

  // The cross_val_predict contract.
  const forward = foldPlan({ rows: [0, 1, 2, 3, 4, 5, 6, 7], folds: [[4, 5], [6, 7]], requirePartition: false });
  equal(forward.coverage.unassessed, [0, 1, 2, 3]);
  truthy(!forward.coverage.partition, 'a forward plan is not a once-per-row partition');
  equal(outOfFoldCoverage(forward).filter(row => !row.usable).map(row => row.row), [0, 1, 2, 3]);
  equal(outOfFoldCoverage(forward).find(row => row.row === 0).status, 'no held-out prediction');
  equal(outOfFoldCoverage(forward).find(row => row.row === 5).status, 'fit 1');
  const kfold = foldPlan({ rows: [0, 1, 2, 3, 4, 5, 6, 7], folds: consecutiveFolds(8, 4) });
  truthy(outOfFoldCoverage(kfold).every(row => row.usable), 'a K-fold partition can produce one prediction per row');
  const repeatedHoldouts = foldPlan({ rows: [0, 1, 2, 3], folds: [[0, 1], [1, 2]], requirePartition: false });
  equal(repeatedHoldouts.coverage.repeated, [1]);
  equal(outOfFoldCoverage(repeatedHoldouts).find(row => row.row === 1).status, 'assessed 2 times');

  rejects(() => foldPlan({ rows: [0, 1, 2], folds: [[0, 1]] }), 'a dropped remainder is refused');
  rejects(() => foldPlan({ rows: [0, 1, 2], folds: [[0], [1], []] }), 'an empty fold is refused');
  rejects(() => foldPlan({ rows: [0, 1, 2], folds: [[0, 0], [1, 2]] }), 'a repeated ID inside one fold is refused');
  rejects(() => foldPlan({ rows: [0, 1, 2], folds: [[0, 9], [1, 2]] }), 'an unknown row ID is refused');
  rejects(() => foldPlan({ rows: [0, 0, 1], folds: [[0], [1]] }), 'duplicate dataset rows are refused');
  rejects(() => foldPlan({ rows: [], folds: [[0]] }));
});

/* ---------------------------------------------------------------- */
const tinyPoints = [0, 1, 2, 3, 4, 5, 6].map(id => ({ id, x: id, y: id >= 3 ? 1 : 0 }));
const tinyPlan = points => foldPlan({ rows: points.map(point => point.id), folds: consecutiveFolds(points.length, 3) });

checked('One-nearest-neighbour run: the packet trace, an independent brute force, weighting identity, contrast, own-label null and tie rule', () => {
  const run = nearestNeighborRun(tinyPoints, tinyPlan(tinyPoints));
  equal(run.folds.map(fold => fold.rows.map(row => row.prediction)), [[1, 1, 1], [0, 1], [1, 1]]);
  equal(run.folds.map(fold => fold.correct), [0, 1, 2]);
  equal(run.folds.map(fold => fold.assessed), [3, 2, 2]);
  equal(run.folds.map(fold => fold.accuracy), [0, 0.5, 1]);
  close(run.foldMean, 0.5, 0, 'the unweighted fold mean is exactly 0.5');
  close(run.pooled, 3 / 7, 0, 'pooled accuracy is exactly 3/7');
  close(run.pooled, 0.42857142857142855, 1e-16, 'and prints as the program does');
  close(run.weightedFoldMean, run.pooled, 1e-15, 'the size-weighted fold mean equals pooled accuracy');
  truthy(run.foldMean !== run.pooled, 'the two summaries genuinely differ on unequal folds');
  // The packet's independently produced trace.
  equal(run.folds.map(fold => fold.rows.map(row => row.row)), packet.tiny.map(fold => fold.validation));
  equal(run.folds.map(fold => fold.training), packet.tiny.map(fold => fold.train));
  equal(run.folds.map(fold => fold.rows.map(row => row.prediction)), packet.tiny.map(fold => fold.prediction));
  equal(run.folds.map(fold => fold.correct), packet.tiny.map(fold => fold.correct));
  // Manuscript sentence: x=3 takes the wrong label from training x=2, x=4 from x=5.
  equal(run.folds[1].rows.map(row => [row.row, row.neighbor, row.prediction]), [[3, 2, 0], [4, 5, 1]]);

  // Independent brute force over arbitrary edits.
  let seed = 20260914;
  const random = () => { seed = (seed * 1103515245 + 12345) % 2147483648; return seed / 2147483648; };
  for (let trial = 0; trial < 400; trial += 1) {
    const points = tinyPoints.map(point => ({ id: point.id, x: Math.round(random() * 24 - 12) / 2, y: random() < 0.5 ? 0 : 1 }));
    const plan = tinyPlan(points);
    const run_ = nearestNeighborRun(points, plan);
    plan.folds.forEach((fold, index) => {
      fold.validation.forEach((id, position) => {
        const best = fold.training
          .map(trainId => ({ trainId, distance: Math.abs(points[id].x - points[trainId].x) }))
          .reduce((a, b) => (b.distance < a.distance - 1e-12 ? b : a));
        const first = fold.training.find(trainId => Math.abs(Math.abs(points[id].x - points[trainId].x) - best.distance) <= 1e-12);
        equal(run_.folds[index].rows[position].neighbor, first, 'the nearest eligible training row, ties to the smallest ID');
        equal(run_.folds[index].rows[position].prediction, points[first].y);
      });
    });
    close(run_.pooled, run_.weightedFoldMean, 1e-12);
  }

  // Contrast: changing training row 3's label changes held-out row 0's prediction.
  const contrast = nearestNeighborRun(tinyPoints.map(point => point.id === 3 ? { ...point, y: 0 } : point), tinyPlan(tinyPoints));
  equal(contrast.folds[0].rows[0].neighbor, 3, 'row 0 still learns from row 3');
  equal(contrast.folds[0].rows[0].prediction, 0, 'its prediction follows that label');
  // Null: a held-out row's own label cannot reach the fit that predicts it.
  const ownLabel = nearestNeighborRun(tinyPoints.map(point => point.id === 0 ? { ...point, y: 1 } : point), tinyPlan(tinyPoints));
  equal(ownLabel.folds[0].rows[0].prediction, 1, 'the prediction is unchanged');
  equal(ownLabel.folds[0].rows[0].correct, true, 'only its correctness changes');
  equal(ownLabel.folds[0].correct, 1);
  // Null: translating every x leaves nearest identities alone.
  const shifted = nearestNeighborRun(tinyPoints.map(point => ({ ...point, x: point.x + 37.5 })), tinyPlan(tinyPoints));
  equal(shifted.folds.map(fold => fold.rows.map(row => row.neighbor)), run.folds.map(fold => fold.rows.map(row => row.neighbor)));
  close(shifted.pooled, run.pooled, 0);
  // Tie rule: two eligible training rows at equal distance take the smaller ID.
  const tied = nearestNeighborRun(
    [{ id: 0, x: 1, y: 0 }, { id: 1, x: 3, y: 1 }, { id: 2, x: 2, y: 0 }, { id: 3, x: 9, y: 1 }],
    foldPlan({ rows: [0, 1, 2, 3], folds: [[2], [0], [1], [3]] }));
  equal(tied.folds[0].rows[0].tied, [0, 1], 'row 2 sits exactly between rows 0 and 1');
  equal(tied.folds[0].rows[0].neighbor, 0, 'the smaller source ID wins');

  rejects(() => nearestNeighborRun(tinyPoints.map(point => ({ ...point, y: 2 })), tinyPlan(tinyPoints)), 'labels must be binary');
  rejects(() => nearestNeighborRun([{ id: 0, x: Number.NaN, y: 0 }, { id: 1, x: 1, y: 1 }], foldPlan({ rows: [0, 1], folds: [[0], [1]] })));
  rejects(() => nearestNeighborRun(tinyPoints.slice(0, 5), tinyPlan(tinyPoints)), 'a plan naming a row without a measurement is refused');
});

/* ---------------------------------------------------------------- */
const constants = [{ id: 'all zeros', pattern: [0, 0, 0, 0] }, { id: 'all ones', pattern: [1, 1, 1, 1] }];

checked('Selection on a criterion with no signal: exact enumeration, duplicate null, sixteen-rule saturation and the fair-label future', () => {
  const outcome = scoreCandidates([1, 1, 1, 0], constants);
  equal(outcome.winner.id, 'all ones');
  close(outcome.selectedValidationAccuracy, 0.75, 0);
  equal(outcome.rows.map(row => row.correct), [1, 3]);
  close(futureAccuracy(outcome.winner.pattern), 0.5, 0, 'a fixed rule is right half the time on fresh fair labels');

  const enumeration = enumerateSelection(constants);
  equal(enumeration.patternCount, 16);
  close(enumeration.probabilityPerPattern, 1 / 16, 0);
  equal(enumeration.rows.length, 16);
  close(enumeration.meanSelectedAccuracy, 0.6875, 0, 'the average selected score is exactly 0.6875');
  close(enumeration.meanSelectedAccuracy, packet.selection_enumeration.two_constant_candidates_mean_selected_accuracy, 0);
  close(enumeration.futureAccuracy, packet.selection_enumeration.true_fresh_accuracy, 0);
  // The manuscript's table: 2 patterns with 4 correct, 8 with 3, 6 with 2.
  const table = enumeration.byOnesCount;
  equal(table.map(row => row.ones), [0, 1, 2, 3, 4]);
  equal(table.map(row => row.patterns), [1, 4, 6, 4, 1]);
  equal(table.map(row => row.bestCorrect), [[4], [3], [2], [3], [4]]);
  const grouped = [[0, 4], [1, 3], [2]].map(group => ({
    patterns: group.reduce((sum, ones) => sum + table.find(row => row.ones === ones).patterns, 0),
    best: table.find(row => row.ones === group[0]).bestCorrect[0],
  }));
  equal(grouped, [{ patterns: 2, best: 4 }, { patterns: 8, best: 3 }, { patterns: 6, best: 2 }]);
  close((2 * 4 + 8 * 3 + 6 * 2) / (16 * 4), 0.6875, 0, 'the arithmetic the manuscript prints');
  equal(table.reduce((sum, row) => sum + row.patterns, 0), 16, 'the pattern counts sum to sixteen');

  // Duplicate null: adding a copy changes neither the best score nor the mean.
  const duplicated = enumerateSelection([...constants, { id: 'all zeros (copy)', pattern: [0, 0, 0, 0] }]);
  close(duplicated.meanSelectedAccuracy, 0.6875, 0);
  equal(scoreCandidates([1, 1, 1, 0], [...constants, { id: 'copy', pattern: [0, 0, 0, 0] }]).selectedValidationAccuracy, 0.75);

  // All sixteen prediction patterns: validation score 1 everywhere, future 0.5.
  const every = Array.from({ length: 16 }, (_, index) => ({
    id: `pattern ${index}`, pattern: [3, 2, 1, 0].map(bit => (index >> bit) & 1),
  }));
  const saturated = enumerateSelection(every);
  close(saturated.meanSelectedAccuracy, 1, 0, 'one rule matches every possible validation set');
  close(saturated.meanSelectedAccuracy, packet.selection_enumeration.all16_patterns_mean_selected_accuracy, 0);
  every.forEach(candidate => close(futureAccuracy(candidate.pattern), 0.5, 0));

  // Practice 5's fixture.
  const fixture = scoreCandidates([0, 1, 0, 1], constants);
  close(fixture.selectedValidationAccuracy, 0.5, 0, 'two constant rules cannot beat one half here');
  equal(fixture.tiedWith, ['all zeros', 'all ones']);
  equal(fixture.tieBrokenBy, 'first candidate in the declared enumeration');
  const matched = scoreCandidates([0, 1, 0, 1], [...constants, { id: 'matching', pattern: [0, 1, 0, 1] }]);
  close(matched.selectedValidationAccuracy, 1, 0, 'a matching rule scores 1');
  close(futureAccuracy([0, 1, 0, 1]), 0.5, 0, 'and still predicts fresh fair labels half the time');
  const duplicateZero = scoreCandidates([0, 1, 0, 1], [...constants, { id: 'another all zeros', pattern: [0, 0, 0, 0] }]);
  close(duplicateZero.selectedValidationAccuracy, 0.5, 0, 'a second all-zero rule changes nothing');

  rejects(() => scoreCandidates([0, 1, 0, 2], constants));
  rejects(() => scoreCandidates([0, 1, 0, 1], [{ id: 'short', pattern: [0, 1, 0] }]));
  rejects(() => scoreCandidates([0, 1, 0, 1], []));
  rejects(() => enumerateSelection(Array.from({ length: 17 }, () => ({ pattern: [0, 0, 0, 0] }))));
});

/* ---------------------------------------------------------------- */
const nestedX = Array.from({ length: 16 }, (_, index) => index);
const nestedY = nestedX.map(value => (value >= 8 ? 1 : 0));
const comparePacket = (trace, saved, label) => {
  trace.forEach((fold, index) => {
    const recorded = saved[index];
    equal(fold.test, recorded.test, `${label} fold ${index}: protected rows`);
    equal(fold.train, recorded.train, `${label} fold ${index}: outer training rows`);
    equal(fold.inner.map(split => split.train), recorded.inner.map(split => split.train), `${label} fold ${index}: inner training rows`);
    equal(fold.inner.map(split => split.validation), recorded.inner.map(split => split.validation), `${label} fold ${index}: inner validation rows`);
    fold.candidates.forEach((candidate, position) => {
      equal(candidate.k, recorded.candidates[position].k);
      equal(candidate.innerScores, recorded.candidates[position].scores, `${label} fold ${index}: k=${candidate.k} inner scores`);
      close(candidate.mean, recorded.candidates[position].mean, 1e-15);
    });
    equal(fold.selectedK, recorded.selected_k, `${label} fold ${index}: selected neighbour count`);
    equal(fold.prediction, recorded.prediction, `${label} fold ${index}: protected predictions`);
    equal(fold.assessedCorrect, recorded.correct, `${label} fold ${index}: protected correct count`);
  });
};

checked('Constructed nested experiment: all three packet traces, the selection contrast, the protected-label null and the role-relative effect', () => {
  const base = nestedTrace({ x: nestedX, y: nestedY });
  comparePacket(base, packet.tiny_nested.trace, 'base');
  equal(base.map(fold => fold.candidates.map(candidate => candidate.mean)), [[0.875, 0.875], [0.875, 0.875]]);
  equal(base.map(fold => fold.selectedK), [1, 1]);
  truthy(base.every(fold => fold.tie), 'both outer folds tie and resolve to the smaller count');
  equal(base[0].prediction, [0, 0, 0, 0, 0, 1, 1, 1]);
  equal(base[0].assessedCorrect, 7);
  equal(base[1].prediction, [0, 0, 0, 0, 1, 1, 1, 1]);
  equal(base[1].assessedCorrect, 8);
  equal(base[0].test, [0, 2, 4, 6, 8, 10, 12, 14]);
  equal(base[1].test, [1, 3, 5, 7, 9, 11, 13, 15]);
  base.forEach(fold => {
    const inner = fold.inner.flatMap(split => [...split.train, ...split.validation]);
    truthy(inner.every(id => !fold.test.includes(id)), 'no protected row ever appears inside the inner comparison');
    equal([...new Set(fold.inner.flatMap(split => split.validation))].sort((a, b) => a - b), [...fold.train].sort((a, b) => a - b));
  });

  // Contrast: row 3 is an outer-training row in fold 0, so its label can change k.
  const changedThree = nestedTrace({ x: nestedX, y: nestedY.map((value, id) => (id === 3 ? 1 : value)) });
  comparePacket(changedThree, packet.tiny_nested.selection_changed_trace, 'row 3 changed');
  equal(changedThree[0].candidates.map(candidate => candidate.mean), [0.5, 0.625]);
  equal(changedThree[0].selectedK, 3, 'the inner comparison now prefers three neighbours');
  equal(changedThree[0].prediction, base[0].prediction, 'the protected predictions happen to be unchanged');
  equal(changedThree[0].assessedCorrect, 7, 'selection changed without improving this assessment');
  equal(changedThree[1].selectedK, 1, 'in the other outer fold row 3 is protected, so the choice is untouched');
  equal(changedThree[1].candidates.map(candidate => candidate.mean), [0.875, 0.875]);

  // Null: row 2 is protected in outer fold 0, so nothing about that fit may move.
  const changedTwo = nestedTrace({ x: nestedX, y: nestedY.map((value, id) => (id === 2 ? 1 : value)) });
  equal(changedTwo[0].candidates.map(candidate => candidate.mean), base[0].candidates.map(candidate => candidate.mean));
  equal(changedTwo[0].selectedK, base[0].selectedK);
  equal(changedTwo[0].prediction, base[0].prediction, 'the protected prediction vector is identical');
  equal(changedTwo[0].assessedCorrect, 6, 'only its correctness moves');

  // Role-relative effect: row 7 trains fold 0 and is protected in fold 1.
  const changedSeven = nestedTrace({ x: nestedX, y: nestedY.map((value, id) => (id === 7 ? 1 : value)) });
  comparePacket(changedSeven, packet.tiny_nested.changed_trace, 'row 7 changed');
  equal(changedSeven[0].candidates.map(candidate => candidate.innerScores), [[1, 0.75], [0.5, 0.75]]);
  equal(changedSeven[0].candidates.map(candidate => candidate.mean), [0.875, 0.625]);
  equal(changedSeven[0].selectedK, 1);
  equal(changedSeven[0].prediction, [0, 0, 0, 0, 1, 1, 1, 1], 'row 8 now receives label 1');
  equal(changedSeven[0].assessedCorrect, 8);
  equal(changedSeven[1].prediction, base[1].prediction, 'the fitting path in the other fold is unchanged');
  equal(changedSeven[1].candidates.map(candidate => candidate.mean), base[1].candidates.map(candidate => candidate.mean));
  equal(changedSeven[1].assessedCorrect, 7, 'only that fold’s correctness changes');

  // Arbitrary edits still recompute real distances and votes.
  let seed = 7717;
  const random = () => { seed = (seed * 1103515245 + 12345) % 2147483648; return seed / 2147483648; };
  for (let trial = 0; trial < 200; trial += 1) {
    const x = nestedX.map(() => Math.round(random() * 60) / 2);
    const y = nestedX.map(() => (random() < 0.5 ? 0 : 1));
    const trace = nestedTrace({ x, y });
    trace.forEach(fold => {
      fold.candidates.forEach(candidate => {
        fold.inner.forEach((split, position) => {
          split.validation.forEach((id, row) => {
            const ranked = split.train
              .map((trainId, order) => ({ trainId, order, distance: Math.abs(x[id] - x[trainId]) }))
              .sort((a, b) => (a.distance - b.distance) || (a.order - b.order))
              .slice(0, candidate.k);
            const share = ranked.reduce((sum, item) => sum + y[item.trainId], 0) / candidate.k;
            equal(candidate.innerDetail[position].rows[row].prediction, share > 0.5 ? 1 : 0);
          });
        });
      });
      truthy(fold.selectedK === fold.candidates.reduce((best, candidate) =>
        (candidate.mean > best.mean ? candidate : best)).k || fold.tie, 'the highest inner mean wins, smaller k on a tie');
      const bestMean = Math.max(...fold.candidates.map(candidate => candidate.mean));
      equal(fold.selectedK, fold.candidates.find(candidate => candidate.mean === bestMean).k);
    });
  }

  rejects(() => nestedTrace({ x: nestedX, y: nestedY, neighborCounts: [2] }), 'an even neighbour count could tie a binary vote');
  rejects(() => nestedTrace({ x: nestedX, y: nestedY, neighborCounts: [1, 9] }), 'a count larger than the inner training set is refused');
  rejects(() => nestedTrace({ x: nestedX.slice(0, 3), y: nestedY.slice(0, 3) }));
  rejects(() => nestedTrace({ x: nestedX, y: nestedY.map(() => 5) }));
});

/* ---------------------------------------------------------------- */
checked('Search coverage: hit probability, required draws, and the distinction from score proximity', () => {
  close(hitProbability(0.05, 60), 0.9539302010130480, 1e-15);
  close(hitProbability(0.01, 60), 0.4528433576092388, 1e-15);
  close(hitProbability(0.05, 60), packet.random_hit['0.05']['60'], 0);
  close(hitProbability(0.01, 60), packet.random_hit['0.01']['60'], 0);
  close(hitProbability(0.05, 20), packet.random_hit['0.05']['20'], 0);
  close(hitProbability(0.01, 299), packet.random_hit['0.01']['299'], 0);
  equal(Number(hitProbability(0.05, 60).toFixed(7)), 0.9539302);
  equal(Number(hitProbability(0.01, 60).toFixed(7)), 0.4528434);
  close(hitProbability(0.5, 1), 0.5, 0);
  close(hitProbability(0.05, 0), 0, 0);
  for (const p of [0.01, 0.02, 0.05, 0.5]) {
    for (let trials = 1; trials <= 60; trials += 1) {
      // Complement of "missed every time", computed by repeated multiplication.
      let missed = 1;
      for (let draw = 0; draw < trials; draw += 1) missed *= 1 - p;
      close(hitProbability(p, trials), 1 - missed, 1e-15);
    }
  }
  // Practice 7.
  equal(drawsForHitProbability(0.02, 0.95), 149);
  close(Math.log(0.05) / Math.log(0.98), 148.283704, 1e-5, "the manuscript rounds this to 148.28");
  truthy(hitProbability(0.02, 149) >= 0.95 && hitProbability(0.02, 148) < 0.95, '149 is the smallest sufficient number of draws');
  rejects(() => hitProbability(1.5, 10));
  rejects(() => hitProbability(0.05, -1));
  rejects(() => drawsForHitProbability(0, 0.95));
  rejects(() => drawsForHitProbability(0.05, 1));
});

/* ---------------------------------------------------------------- */
checked('Conditional versus expected error: exact mean-predictor risks, optimism, and the overlap counterexample', () => {
  close(meanPredictorRisk(4, 8).expectedNewLoss, 4.5, 0, 'a three-fold fit on twelve rows trains on eight');
  close(meanPredictorRisk(4, 12).expectedNewLoss, 4 + 4 / 12, 1e-15);
  close(meanPredictorRisk(4, 12).expectedNewLoss, 4.333333333333333, 1e-15);
  close(meanPredictorRisk(4, 12).expectedTrainingLoss, 4 * (1 - 1 / 12), 1e-15);
  close(meanPredictorRisk(4, 12).optimism, 2 * 4 / 12, 1e-15);
  close(meanPredictorRisk(4, 12).expectedNewLoss - meanPredictorRisk(4, 12).expectedTrainingLoss, meanPredictorRisk(4, 12).optimism, 1e-15);
  for (let n = 2; n <= 40; n += 1) {
    const risk = meanPredictorRisk(4, n);
    close(risk.expectedTrainingLoss, 4 - 4 / n, 1e-15);
    close(risk.expectedNewLoss, 4 + 4 / n, 1e-15);
    truthy(risk.expectedTrainingLoss < risk.irreducible && risk.irreducible < risk.expectedNewLoss, 'training error sits below the irreducible line and new error above it');
  }
  truthy(meanPredictorRisk(4, 8).expectedNewLoss > meanPredictorRisk(4, 12).expectedNewLoss, 'a smaller training size raises expected new loss');

  close(fixedRuleAccuracyVariance(1), 0.25, 0);
  for (let n = 1; n <= 50; n += 1) close(fixedRuleAccuracyVariance(n), 0.25 / n, 1e-15);
  // Direct enumeration for a rule that always predicts 0 on independent fair bits.
  for (const n of [1, 2, 3, 4, 8]) {
    let mean = 0;
    let second = 0;
    for (let pattern = 0; pattern < 2 ** n; pattern += 1) {
      let correct = 0;
      for (let bit = 0; bit < n; bit += 1) correct += ((pattern >> bit) & 1) === 0 ? 1 : 0;
      const accuracy = correct / n;
      mean += accuracy / 2 ** n;
      second += accuracy * accuracy / 2 ** n;
    }
    close(second - mean * mean, fixedRuleAccuracyVariance(n), 1e-14, `enumerated variance at n=${n}`);
  }
  close(foldMeanVariance({ tauSquared: 1, k: 5, rho: 0 }), 0.2, 1e-15, 'independent folds give tau squared over K');
  close(foldMeanVariance({ tauSquared: 1, k: 5, rho: 1 }), 1, 1e-15, 'perfectly correlated folds average nothing away');
  close(foldMeanVariance({ tauSquared: 2, k: 4, rho: 0.5 }), 2 * (1 + 3 * 0.5) / 4, 1e-15);
  rejects(() => foldMeanVariance({ tauSquared: 1, k: 5, rho: -0.5 }), 'an equal correlation below -1/(K-1) is impossible');
  rejects(() => foldMeanVariance({ tauSquared: -1, k: 5, rho: 0 }));
  rejects(() => meanPredictorRisk(4, 0));
  rejects(() => fixedRuleAccuracyVariance(0));

  const bootstrap = bootstrapDistinctShare(344);
  close(bootstrap.absentProbability, (1 - 1 / 344) ** 344, 0);
  close(bootstrap.limit, 1 - Math.exp(-1), 0);
  close(bootstrapDistinctShare(1000000).distinctShare, 0.632, 1e-3, 'the distinct share approaches 63.2 per cent');
  truthy(Math.abs(bootstrap.absentProbability - Math.exp(-1)) < 0.001, 'the absent probability is already near e to the minus one');
  rejects(() => bootstrapDistinctShare(0));
});

/* ---------------------------------------------------------------- */
checked('Acquisition: expected improvement against expected loss, the unchanged-EI null, and the TPE ratio', () => {
  const a = expectedImprovement(0.2, [{ loss: 0.18, probability: 1 }]);
  close(a.expectedImprovement, 0.02, 1e-15);
  close(a.meanLoss, 0.18, 1e-15);
  const b = expectedImprovement(0.2, [{ loss: 0.05, probability: 0.5 }, { loss: 0.45, probability: 0.5 }]);
  close(b.expectedImprovement, 0.075, 1e-15, 'only the improving outcome contributes');
  close(b.meanLoss, 0.25, 1e-15);
  truthy(b.meanLoss > a.meanLoss && b.expectedImprovement > a.expectedImprovement, 'a worse mean can carry a larger expected improvement');
  equal(b.rows.map(row => row.improvement), [0.15000000000000002, 0]);
  // Practice 8: moving only the worse outcome leaves EI alone and moves the mean.
  const c = expectedImprovement(0.3, [{ loss: 0.1, probability: 0.5 }, { loss: 0.5, probability: 0.5 }]);
  const cWorse = expectedImprovement(0.3, [{ loss: 0.1, probability: 0.5 }, { loss: 0.9, probability: 0.5 }]);
  close(c.expectedImprovement, 0.1, 1e-15);
  close(cWorse.expectedImprovement, 0.1, 1e-15);
  close(c.meanLoss, 0.3, 1e-15);
  close(cWorse.meanLoss, 0.5, 1e-15);
  close(expectedImprovement(0.2, [{ loss: 0.25, probability: 1 }]).expectedImprovement, 0, 0, 'a candidate believed worse has no expected improvement');
  rejects(() => expectedImprovement(0.2, [{ loss: 0.1, probability: 0.6 }, { loss: 0.4, probability: 0.6 }]));
  rejects(() => expectedImprovement(0.2, []));

  close(tpeAcquisition({ gamma: 0.25, betterDensity: 1, otherDensity: 1 }), 1, 1e-15);
  truthy(tpeAcquisition({ gamma: 0.25, betterDensity: 4, otherDensity: 1 }) > tpeAcquisition({ gamma: 0.25, betterDensity: 1, otherDensity: 1 }),
    'a higher better-group density raises the acquisition value');
  close(tpeAcquisition({ gamma: 0.25, betterDensity: 1, otherDensity: 0 }), 4, 1e-15, 'the value is bounded by one over gamma');
  rejects(() => tpeAcquisition({ gamma: 0.25, betterDensity: 0, otherDensity: 1 }));
});

/* ---------------------------------------------------------------- */
const trajectories = [
  { id: 'A', losses: [0.3, 0.25, 0.24] },
  { id: 'B', losses: [0.4, 0.2, 0.1] },
  { id: 'C', losses: [0.35, 0.28, 0.27] },
];
const budgets = [10, 30, 90];

checked('Successive halving: early ranking, the late-value null, the first-budget contrast, and exact resource accounting', () => {
  const early = successiveHalving({ candidates: trajectories, budgets, factor: 3, firstStage: 0 });
  equal(early.survivor.id, 'A');
  equal(early.stages[0].observed.map(row => row.loss), [0.3, 0.4, 0.35]);
  equal(early.stages[0].eliminated.map(index => trajectories[index].id), ['C', 'B']);
  equal(early.observedUpTo, [2, 0, 0], 'the schedule never paid to see B or C past the first budget');
  truthy(!early.hindsight.matchesSurvivor, 'the survivor is not the full-budget winner');
  equal(early.hindsight.bestId, 'B');
  close(early.hindsight.bestLoss, 0.1, 1e-15);
  close(early.hindsight.survivorFinalLoss, 0.24, 1e-15);
  close(early.hindsight.regret, 0.14, 1e-15);
  equal(early.cost.nominal, 3 * 10 + 1 * 30 + 1 * 90);
  equal(early.cost.resumable, 3 * 10 + 1 * 20 + 1 * 60);
  equal(early.cost.fullAllocation, 270);

  // Deciding later reaches the eventual winner.
  const later = successiveHalving({ candidates: trajectories, budgets, factor: 3, firstStage: 1 });
  equal(later.survivor.id, 'B');
  equal(later.stages[0].observed.map(row => row.loss), [0.25, 0.2, 0.28]);
  truthy(later.hindsight.matchesSurvivor, 'starting at 30 selects the full-budget winner');
  close(later.hindsight.regret, 0, 0);
  equal(later.cost.nominal, 3 * 30 + 1 * 90);
  equal(later.observedUpTo, [1, 2, 1], "the later start pays for every candidate at budget 30 and only B beyond it");

  // Null: a value the schedule never looked at cannot change the decision.
  const nullCase = successiveHalving({
    candidates: trajectories.map(candidate => candidate.id === 'B' ? { ...candidate, losses: [0.4, 0.2, 0.02] } : candidate),
    budgets, factor: 3, firstStage: 0,
  });
  equal(nullCase.survivor.id, 'A');
  equal(nullCase.stages.map(stage => stage.observed.map(row => row.loss)), early.stages.map(stage => stage.observed.map(row => row.loss)));
  close(nullCase.hindsight.regret, 0.22, 1e-15, 'only the hindsight regret grows');

  // Contrast: an observed value the schedule does pay for changes the survivor.
  const contrast = successiveHalving({
    candidates: trajectories.map(candidate => candidate.id === 'B' ? { ...candidate, losses: [0.2, 0.2, 0.1] } : candidate),
    budgets, factor: 3, firstStage: 0,
  });
  equal(contrast.survivor.id, 'B');
  truthy(contrast.hindsight.matchesSurvivor);

  // Ties keep the earlier candidate in the declared order.
  const tie = successiveHalving({
    candidates: [{ id: 'A', losses: [0.3, 0.2, 0.1] }, { id: 'B', losses: [0.3, 0.1, 0.05] }],
    budgets, factor: 2, firstStage: 0,
  });
  equal(tie.stages[0].ranked.map(row => row.id), ['A', 'B']);
  equal(tie.stages[0].keep, 1);
  equal(tie.survivor.id, 'A');

  // The manuscript's nine-candidate schedule.
  const nine = successiveHalving({
    candidates: Array.from({ length: 9 }, (_, index) => ({ id: `candidate ${index + 1}`, losses: [0.5 - index / 100, 0.4 - index / 100, 0.3 - index / 100] })),
    budgets, factor: 3, firstStage: 0,
  });
  equal(nine.stages.map(stage => stage.assessed), [9, 3, 1]);
  equal(nine.stages.map(stage => stage.nominalStageCost), [90, 90, 90]);
  equal(nine.cost.nominal, 270);
  equal(nine.cost.fullAllocation, 810);
  equal(nine.cost.resumable, 9 * 10 + 3 * 20 + 1 * 60);
  equal(nine.cost.resumable, 210);

  // Practice 9's larger schedule.
  const twentySeven = successiveHalving({
    candidates: Array.from({ length: 9 }, (_, index) => ({ id: `c${index}`, losses: [1 - index / 100, 0.9 - index / 100, 0.8 - index / 100, 0.7 - index / 100] })),
    budgets: [5, 15, 45, 135], factor: 3, firstStage: 0,
  });
  equal(twentySeven.stages.map(stage => stage.assessed), [9, 3, 1, 1]);
  // The manuscript states the 27-candidate arithmetic; recompute it directly.
  equal([27, 9, 3, 1].map((count, index) => count * [5, 15, 45, 135][index]), [135, 135, 135, 135]);
  equal([27, 9, 3, 1].reduce((sum, count, index) => sum + count * [5, 15, 45, 135][index], 0), 540);
  equal(27 * 135, 3645);
  equal(27 * 5 + 9 * 10 + 3 * 30 + 1 * 90, 405);
  equal(9 * 10 + 3 * 20 + 1 * 60, 210, 'the nine-candidate resume total');

  // Factor 2 retains more, and a single retained candidate is the floor.
  const half = successiveHalving({ candidates: trajectories, budgets, factor: 2, firstStage: 0 });
  equal(half.stages[0].keep, 2);
  equal(half.stages[1].assessed, 2);
  equal(half.stages[1].keep, 1);

  rejects(() => successiveHalving({ candidates: trajectories, budgets: [10], factor: 3 }));
  rejects(() => successiveHalving({ candidates: trajectories, budgets: [30, 10, 90], factor: 3 }));
  rejects(() => successiveHalving({ candidates: trajectories, budgets, factor: 1 }));
  rejects(() => successiveHalving({ candidates: trajectories, budgets, factor: 3, firstStage: 2 }), 'a first comparison at the last budget decides nothing');
  rejects(() => successiveHalving({ candidates: [trajectories[0]], budgets, factor: 3 }));
  rejects(() => successiveHalving({ candidates: trajectories.map(candidate => ({ ...candidate, losses: [-1, 0, 1] })), budgets, factor: 3 }));
  rejects(() => successiveHalving({ candidates: Array.from({ length: 10 }, () => ({ losses: [1, 2, 3] })), budgets, factor: 3 }));
});

/* ---------------------------------------------------------------- */
checked('Fit accounting matches the manuscript arithmetic', () => {
  const budget = fitBudget({ candidates: 6, outerFolds: 3, innerFolds: 3 });
  equal(budget.perOuter, 19);
  equal(budget.assessment, 57);
  equal(budget.final, 19);
  equal(budget.total, 76);
  equal(fitBudget({ candidates: 6, outerFolds: 1, innerFolds: 3, refit: false, finalSearch: false }).total, 18,
    'six candidates over three inner folds cost eighteen candidate fits');
  equal(fitBudget({ candidates: 100000, outerFolds: 1, innerFolds: 5, refit: false, finalSearch: false }).total, 500000,
    'ten values on each of five choices at five folds is 500,000 fits');
  equal(10 ** 5, 100000);
  rejects(() => fitBudget({ candidates: 0, outerFolds: 3, innerFolds: 3 }));
});

/* ---------------------------------------------------------------- */
checked('Published real experiment agrees with the manuscript sentences and keeps selection and assessment apart', () => {
  equal(NESTED_EXPERIMENT.folds.length, 3);
  equal(NESTED_EXPERIMENT.folds.map(fold => fold.selected.k), [11, 3, 3]);
  equal(NESTED_EXPERIMENT.folds.map(fold => fold.selected.scaler), ['StandardScaler', 'StandardScaler', 'RobustScaler']);
  equal(NESTED_EXPERIMENT.folds.map(fold => fold.correct), [114, 114, 112]);
  equal(NESTED_EXPERIMENT.folds.map(fold => fold.testRows.length), [115, 115, 114]);
  equal(NESTED_EXPERIMENT.folds.map(fold => fold.trainingRows.length), [229, 229, 230]);
  equal(NESTED_EXPERIMENT.folds.map(fold => fold.baselineCorrect), [51, 51, 50]);
  equal(NESTED_EXPERIMENT.pooledCorrect, 340);
  close(NESTED_EXPERIMENT.pooledAccuracy, 340 / 344, 1e-15);
  close(NESTED_EXPERIMENT.foldMean, packet.fold_mean, 0);
  close(NESTED_EXPERIMENT.foldMean, NESTED_EXPERIMENT.folds.reduce((sum, fold) => sum + fold.accuracy, 0) / 3, 1e-15);
  close(NESTED_EXPERIMENT.finalSelected.selectionScore, 0.991304347826087, 1e-15);
  equal(NESTED_EXPERIMENT.finalSelected.k, 3);
  equal(NESTED_EXPERIMENT.finalSelected.scaler, 'StandardScaler');
  // The final search is a two-way tie the enumeration rule resolves, and the
  // page now says so. Check the shipped record supports that sentence.
  equal(NESTED_EXPERIMENT.finalSelected.candidates.length, 6);
  equal(NESTED_EXPERIMENT.finalSelected.tiedWith, ['3/StandardScaler', '3/RobustScaler']);
  equal(NESTED_EXPERIMENT.finalSelected.bestIndex, 0, 'the first tied candidate in the enumeration is the selected one');
  const finalBest = Math.max(...NESTED_EXPERIMENT.finalSelected.candidates.map(candidate => candidate.meanScore));
  close(finalBest, NESTED_EXPERIMENT.finalSelected.selectionScore, 0);
  equal(NESTED_EXPERIMENT.finalSelected.candidates.filter(candidate => candidate.meanScore === finalBest).length, 2);
  equal(new Set(NESTED_EXPERIMENT.finalSelected.candidates.map(candidate => candidate.meanScore)).size, 3,
    'the six final candidates collapse into three tied pairs, so the scaler is settled by order');
  NESTED_EXPERIMENT.finalSelected.candidates.forEach(candidate =>
    close(candidate.meanScore, candidate.foldScores.reduce((a, b) => a + b, 0) / 3, 1e-12));
  // The coincidence the page names: two protected results print the seven digits
  // the final selection score prints, and they are different quantities.
  close(NESTED_EXPERIMENT.folds[0].accuracy, NESTED_EXPERIMENT.finalSelected.selectionScore, 0,
    '114/115 and 228/230 are the same ratio');
  truthy(NESTED_EXPERIMENT.folds[0].selectionScore !== NESTED_EXPERIMENT.finalSelected.selectionScore,
    'but outer fold 1 chose its setting with a different number again');
  equal(NESTED_EXPERIMENT.enumeration.map(item => `${item.k}/${item.scaler}`),
    ['3/StandardScaler', '3/RobustScaler', '5/StandardScaler', '5/RobustScaler', '11/StandardScaler', '11/RobustScaler']);
  equal(NESTED_EXPERIMENT.fits, { perOuter: 19, assessment: 57, finalSearch: 19, total: 76, baselines: 3 });
  equal(NESTED_EXPERIMENT.folds.flatMap(fold => fold.testRows).sort((a, b) => a - b), Array.from({ length: 344 }, (_, index) => index));
  NESTED_EXPERIMENT.folds.forEach(fold => {
    const held = new Set(fold.testRows);
    truthy(fold.trainingRows.every(row => !held.has(row)), 'no protected row is in its own outer training set');
    const innerRows = fold.innerSplits.flatMap(split => split.validationRows);
    equal(innerRows.slice().sort((a, b) => a - b), fold.trainingRows.slice().sort((a, b) => a - b));
    truthy(innerRows.every(row => !held.has(row)), 'no protected row reaches the inner comparison that chose this fold’s setting');
    const best = Math.max(...fold.candidates.map(candidate => candidate.meanScore));
    equal(fold.candidates.findIndex(candidate => candidate.meanScore === best), fold.bestIndex);
    close(fold.selectionScore, best, 1e-15);
    fold.candidates.forEach(candidate => close(candidate.meanScore, candidate.foldScores.reduce((a, b) => a + b, 0) / 3, 1e-12));
    equal(fold.correct, fold.prediction.filter((value, index) => value === fold.truth[index]).length);
    equal(fold.missed.length, fold.testRows.length - fold.correct);
    truthy(fold.selectionScore !== fold.accuracy || fold.fold === undefined, 'the selection score is stored apart from the assessment');
  });
  // The fold whose selection score exceeds its own assessed accuracy is exactly
  // the winner's-curse shape the lesson describes; check it is visible, not hidden.
  truthy(NESTED_EXPERIMENT.folds.some(fold => fold.selectionScore > fold.accuracy), 'at least one inner selection score sits above its protected result');
  equal(PENGUIN_SOURCE.rows, 344);
  equal(PENGUIN_SOURCE.sha256, packet.data_sha256);
  equal(PENGUIN_SOURCE.missing.sex, 11);
  equal(PENGUIN_SOURCE.licence, 'CC0');
  equal(Object.values(PENGUIN_SOURCE.speciesCounts).reduce((a, b) => a + b, 0), 344);
  equal(RUNTIME_VERSIONS, { python: '3.12.14', numpy: '2.3.5', pandas: '3.0.1', sklearn: '1.9.1' });
});

const files = [
  'src/learn/data/validation-models.js',
  'src/learn/data/validation-data.js',
  'src/learn/data/validation-examples.js',
  'scripts/verify-validation-models.mjs',
];
const evidence = {
  status: 'passed',
  generatedAt: new Date().toISOString(),
  command: 'node scripts/verify-validation-models.mjs',
  stage: 'author model verification; browser, independent and integration review are separate',
  groups,
  groupCount: groups.length,
  assertions,
  independentOracles: [
    'array_split fold sizes recomputed for every n up to 40 and every k',
    '400 randomised 1-NN runs graded against a brute-force nearest-eligible-neighbour search',
    '200 randomised nested runs graded against a directly written stable-sort majority vote',
    'all 16 fair-label patterns enumerated rather than sampled',
    'leave-one-out accuracy variance of the fixed rule enumerated over every label pattern up to n=8',
    'hit probabilities recomputed by repeated multiplication for 240 (p, T) pairs',
    'all three retained packet traces compared value by value',
  ],
  limits: [
    'These are model checks. They say nothing about how the page renders, and nothing about penguins beyond the one recorded experiment.',
    'Fit counts are exact; no runtime or speedup is asserted anywhere in this file.',
  ],
  sourceHashes: Object.fromEntries(files.map(file => [file, crypto.createHash('sha256').update(fs.readFileSync(file)).digest('hex')])),
};
fs.mkdirSync('docs/teaching/evidence', { recursive: true });
fs.writeFileSync('docs/teaching/evidence/validation-models.json', `${JSON.stringify(evidence, null, 2)}\n`);
console.log(`PASS: ${groups.length} model groups, ${assertions} assertions.`);
