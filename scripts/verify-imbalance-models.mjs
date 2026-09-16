// Bounded independent checks of the imbalanced-learning browser models against
// the content packet's recorded calculations, the manuscript's worked values and
// analytic identities, plus structural checks of the generated data module and
// the executed displayed programs.
//
// Every assertion here is written so that it can fail: nothing is compared with
// itself, no loop is allowed to inspect an empty subject set, and every group
// asserts how many subjects it actually saw.
//
// Run: node scripts/verify-imbalance-models.mjs
import assert from 'node:assert/strict';
import fs from 'node:fs';
import crypto from 'node:crypto';
import {
  averagePrecisionOf, averagePrecisionOfScores, actionRisks, balancedWeights, brierOfScores,
  checkFinite, checkRange, classBands, confusion, costOf, countsAt, fBeta, fixtures, focalDerivative,
  focalMass, inverseWeightedOptimum, limits, neighbourWork, populationFlow, precisionRecallPoints,
  priorShift, queueAt, queueComparison, rankedQueue, rocAucOfScores, rowExpansion, smoteConstruction,
  thresholdLadder, thresholdSweep, weightedLossCurve, weightedOptimum, weightedStep,
} from '../src/learn/data/imbalance-models.js';
import {
  inspectionRecords, methods, provenance, roles, study, syntheticSample, tuningRecords,
} from '../src/learn/data/imbalance-data.js';

const packetDirectory = 'docs/teaching/drafts/imbalanced-learning-smote-cost-sensitive-learning';
const recorded = JSON.parse(fs.readFileSync(`${packetDirectory}/calculated-inputs.json`, 'utf8'));
const examples = JSON.parse(fs.readFileSync('src/learn/data/imbalance-examples.js', 'utf8')
  .replace(/^[\s\S]*?export const imbalanceExamples = /, '').replace(/;\s*$/, ''));

const counts = {};
const record = name => { counts[name] = (counts[name] ?? 0) + 1; };
const close = (actual, expected, label, tolerance = 1e-9) => {
  assert(typeof actual === 'number' && Number.isFinite(actual), `${label}: ${actual} is not a finite number`);
  assert(Math.abs(actual - expected) <= tolerance * Math.max(1, Math.abs(expected)),
    `${label}: ${actual} versus ${expected}`);
};
const vector = (actual, expected, label, tolerance = 1e-9) => {
  assert.equal(actual.length, expected.length, `${label}: length ${actual.length} versus ${expected.length}`);
  actual.forEach((value, index) => close(value, expected[index], `${label}[${index}]`, tolerance));
};
/** A loop must have something to look at, or it asserts nothing. */
const nonEmpty = (collection, expected, label) => {
  assert(collection.length > 0, `${label}: the subject set is empty, so nothing was checked`);
  if (expected !== undefined) assert.equal(collection.length, expected, `${label}: saw ${collection.length}, expected ${expected}`);
};
/** A call that must be refused. A silent default here would be a wrong answer. */
const refuses = (call, label) => {
  assert.throws(call, RangeError, `${label}: should have been refused`);
  record('refused input');
};
/** A value the prose tells a learner to type must land on its control's step. */
const onStep = (value, step) => Math.abs(value / step - Math.round(value / step)) < 1e-9;

/* ================================================= §1 · a rare class */

const model = confusion(fixtures.modelCounts);
const baseline = confusion(fixtures.baselineCounts);
close(model.total, 1000, 'the example holds 1,000 cases', 1e-12);
close(baseline.total, 1000, 'and so does the baseline', 1e-12);
close(model.positives, 20, 'twenty of them are positive', 1e-12);
close(model.negatives, 980, 'and 980 negative', 1e-12);
close(model.alerts, 32, 'the model raises 32 alerts', 1e-12);
close(model.cleared, 968, 'and clears 968', 1e-12);
close(model.accuracy, 0.976, 'the model is 97.6% accurate', 1e-12);
close(baseline.accuracy, 0.98, 'the always-negative rule is 98% accurate', 1e-12);
assert(baseline.accuracy > model.accuracy, 'the baseline really is the more accurate of the two');
close(model.precision, 14 / 32, 'its precision is 14/32', 1e-12);
close(model.precision, 0.4375, 'that is .4375', 1e-12);
close(model.recall, 0.7, 'its recall is .7', 1e-12);
close(model.f1, 28 / 52, 'its F1 is 28/52', 1e-12);
close(model.f1, fBeta(fixtures.modelCounts, 1), 'F1 is F-beta at beta one', 1e-12);
close(model.specificity, 962 / 980, 'specificity is TN/(TN+FP)', 1e-12);
close(model.falsePositiveRate, 1 - model.specificity, 'FPR is one minus specificity', 1e-12);
close(model.balancedAccuracy, (0.7 + 962 / 980) / 2, 'balanced accuracy averages recall and specificity', 1e-12);
assert(baseline.precision === null && baseline.undefinedBecause.precision !== null,
  'the baseline selects nothing, so its precision is undefined and says why');
close(baseline.recall, 0, 'its recall is exactly zero, which is defined', 1e-12);
assert(baseline.fn + baseline.fp === 20 && model.fn + model.fp === 24,
  'the baseline makes 20 errors and the model 24');
// Increasing beta must weight missed positives more, on these very counts.
const betas = [0, 0.5, 1, 2, 4].map(beta => ({ beta, value: fBeta(fixtures.modelCounts, beta) }));
nonEmpty(betas, 5, 'F-beta sweep');
close(betas[0].value, model.precision, 'F-beta at beta zero is precision', 1e-12);
betas.slice(1).forEach((entry, index) => {
  assert(entry.value > betas[index].value, `F${entry.beta} exceeds F${betas[index].beta} when recall beats precision`);
  record('F-beta step');
});
record('section 1 counts');

const bands = classBands(fixtures.modelCounts);
assert(bands.conserved, 'the band parts sum back to their classes and to the whole');
nonEmpty(bands.bands, 2, 'class bands');
close(bands.bands[0].parts[0].withinClass, 14 / 20, 'detected is 70% of the positive band', 1e-12);
close(bands.bands[0].parts[1].withinClass, 6 / 20, 'missed is 30% of it', 1e-12);
close(bands.bands[1].parts[0].withinClass, 18 / 980, 'false alarms are 18/980 of the negative band', 1e-12);
close(bands.bands[0].parts[1].ofPopulation, 6 / 1000, 'but missed positives are 0.6% of the page', 1e-12);
close(bands.bands[0].parts[1].withinClass / bands.bands[0].parts[1].ofPopulation, 1000 / 20,
  'the positive band magnifies its parts fiftyfold, which is why the two bands are normalised separately', 1e-12);
close(bands.bands[1].parts[0].withinClass / bands.bands[1].parts[0].ofPopulation, 1000 / 980,
  'while the negative band barely magnifies its own', 1e-12);
bands.bands.forEach(band => {
  close(band.parts.reduce((total, part) => total + part.withinClass, 0), 1, `${band.label} fills its band`, 1e-12);
  record('band conservation');
});
record('figure 1 bands');

// Practice 1: two tables, the same accuracy, opposite recall.
const practiceOne = fixtures.practiceEqualAccuracy.map(confusion);
nonEmpty(practiceOne, 2, 'practice 1 tables');
practiceOne.forEach(table => {
  close(table.total, 1200, 'practice 1 holds 1,200 observations', 1e-12);
  close(table.positives, 12, 'twelve of them positive', 1e-12);
  close(table.accuracy, 0.99, 'and both tables are 99% accurate', 1e-12);
  record('practice 1 table');
});
close(practiceOne[0].recall, 0, 'the first detects nothing', 1e-12);
close(practiceOne[1].recall, 1, 'the second detects everything', 1e-12);
assert(practiceOne[0].accuracy === practiceOne[1].accuracy && practiceOne[0].recall !== practiceOne[1].recall,
  'the same accuracy describes both');
record('practice 1');

/* ================================================= §2 · the score queue */

const ladder = thresholdLadder(fixtures.queue);
nonEmpty(ladder, 4, 'the counterexample ladder');
const stated = [
  { threshold: Infinity, precision: null, recall: 0, tp: 0, fp: 0 },
  { threshold: 0.9, precision: 0, recall: 0, tp: 0, fp: 1 },
  { threshold: 0.8, precision: 1 / 2, recall: 1 / 2, tp: 1, fp: 1 },
  { threshold: 0.7, precision: 2 / 3, recall: 1, tp: 2, fp: 1 },
];
stated.forEach(expected => {
  const point = ladder.find(entry => entry.threshold === expected.threshold);
  assert(point, `the ladder reaches the threshold ${expected.threshold}`);
  if (expected.precision === null) {
    assert(point.precision === null && point.undefinedBecause.precision !== null,
      'above every score nothing is selected and precision is undefined, with its denominator named');
  } else {
    close(point.precision, expected.precision, `precision at ${expected.threshold}`, 1e-12);
  }
  close(point.recall, expected.recall, `recall at ${expected.threshold}`, 1e-12);
  assert(point.tp === expected.tp && point.fp === expected.fp, `counts at ${expected.threshold}`);
  assert(point.tp + point.fp + point.fn + point.tn === 3, 'every record lands in exactly one bin');
  record('counterexample operating point');
});
// The manuscript's claim: raising the gate makes precision worse throughout.
const descending = ladder.filter(point => !point.isNoAlert);
nonEmpty(descending, 3, 'finite operating points');
descending.forEach((point, index) => {
  if (index === 0) return;
  assert(point.precision > descending[index - 1].precision,
    'precision falls monotonically as the gate rises through this particular queue');
  assert(point.recall >= descending[index - 1].recall, 'and recall never rises with it');
  record('counterexample monotonicity');
});
// Correcting the highest record's label makes every nonempty precision one.
const corrected = fixtures.queue.map(row => (row.id === 'A' ? { ...row, truth: 1 } : row));
const correctedLadder = thresholdLadder(corrected).filter(point => !point.isNoAlert);
nonEmpty(correctedLadder, 3, 'corrected ladder');
correctedLadder.forEach(point => {
  close(point.precision, 1, 'with every record positive, precision is one at every nonempty threshold', 1e-12);
  record('corrected queue point');
});
record('investigation 1 default');

// Display order cannot change a threshold decision. Check every permutation.
const permutations = [];
const permute = (rest, built) => {
  if (rest.length === 0) { permutations.push(built); return; }
  rest.forEach((row, index) => permute([...rest.slice(0, index), ...rest.slice(index + 1)], [...built, row]));
};
permute(fixtures.tiedQueue, []);
nonEmpty(permutations, 24, 'display permutations of the tied queue');
const canonicalTie = queueAt(fixtures.tiedQueue, 0.8);
const canonicalAp = averagePrecisionOf(fixtures.tiedQueue).value;
permutations.forEach(order => {
  const point = queueAt(order, 0.8);
  assert(point.tp === canonicalTie.tp && point.fp === canonicalTie.fp
    && point.fn === canonicalTie.fn && point.tn === canonicalTie.tn,
    'a reordered display gives the same counts');
  assert.deepEqual([...point.selectedIds].sort(), [...canonicalTie.selectedIds].sort(),
    'and selects exactly the same identities');
  close(averagePrecisionOf(order).value, canonicalAp, 'and the same average precision', 1e-12);
  record('tie permutation null');
});
// Practice 2: the tied queue at .8.
assert(canonicalTie.tp === 1 && canonicalTie.fp === 2 && canonicalTie.fn === 1 && canonicalTie.tn === 0,
  'practice 2 counts at .8 are TP 1, FP 2, FN 1, TN 0');
close(canonicalTie.precision, 1 / 3, 'practice 2 precision is 1/3', 1e-12);
close(canonicalTie.recall, 1 / 2, 'practice 2 recall is 1/2', 1e-12);
assert(canonicalTie.selectedIds.length === 3, 'the threshold selects three records, not two');
const budget = rankedQueue(fixtures.tiedQueue.map(row => row.id), fixtures.tiedQueue.map(row => row.truth),
  fixtures.tiedQueue.map(row => row.score), 2);
assert(budget.top.length === 2 && budget.boundaryTie,
  'an exact budget of two straddles the tie, so the policy has to declare a rule');
assert(budget.top[0].id === 'P', 'the highest score is taken first');
assert(budget.top[1].score === 0.8, 'and one of the tied records, chosen by stored order rather than by truth');
record('practice 2');

// Average precision: the manuscript's 5/6, plus the grouped increments.
const apModel = averagePrecisionOf(fixtures.averagePrecisionQueue);
close(apModel.value, 5 / 6, 'average precision of descending labels 1,0,1,0 is 5/6', 1e-12);
close(apModel.value, recorded.constructed.ap_distinct_example, 'and it matches the packet', 1e-12);
nonEmpty(apModel.steps, 4, 'average-precision steps');
close(apModel.steps.filter(step => step.increment > 0).length, 2, 'recall rises at exactly two ranks', 1e-12);
close(apModel.steps[0].contribution, 0.5 * 1, 'the first contribution is (1/2)·1', 1e-12);
close(apModel.steps[2].contribution, 0.5 * (2 / 3), 'the third is (1/2)·(2/3)', 1e-12);
close(apModel.steps.reduce((total, step) => total + step.increment, 0), 1, 'the increments sum to full recall', 1e-12);
// A constant score gives the prevalence, and a queue without positives is undefined.
const constantQueue = [
  { id: 'A', score: 0.5, truth: 1 }, { id: 'B', score: 0.5, truth: 0 },
  { id: 'C', score: 0.5, truth: 0 }, { id: 'D', score: 0.5, truth: 0 },
];
close(averagePrecisionOf(constantQueue).value, 0.25, 'a constant score gives the sample prevalence', 1e-12);
const noPositives = averagePrecisionOf(constantQueue.map(row => ({ ...row, truth: 0 })));
assert(noPositives.value === null && noPositives.undefinedBecause !== null,
  'with no positives, average precision is reported undefined rather than zero');
nonEmpty(precisionRecallPoints(fixtures.averagePrecisionQueue), 4, 'exact PR coordinates');
assert(precisionRecallPoints(constantQueue).length === 1,
  'a constant score has exactly one operating point, so no curve may be drawn through it');
record('average precision');

// Recall can never rise as the gate rises: swept over every ordered pair of
// candidates on several queues, not sampled at two of them.
let monotoneChecks = 0;
[fixtures.queue, fixtures.tiedQueue, fixtures.averagePrecisionQueue, constantQueue, corrected].forEach(queue => {
  const points = thresholdLadder(queue);
  nonEmpty(points, undefined, 'ladder for the monotonicity sweep');
  for (const from of points) {
    for (const to of points) {
      if (to.threshold <= from.threshold) continue;
      const comparison = queueComparison(queue, from.threshold, to.threshold);
      assert(comparison.after.tp <= comparison.before.tp, 'a higher gate never admits more true positives');
      assert(comparison.after.recall === null || comparison.before.recall === null
        || comparison.after.recall <= comparison.before.recall + 1e-12, 'so recall cannot rise');
      assert(comparison.recallMonotone, 'and the model says so');
      monotoneChecks += 1;
    }
  }
});
assert(monotoneChecks >= 20, `the monotonicity sweep compared ${monotoneChecks} ordered pairs`);
record('recall monotonicity sweep');

/* ============================================= §2 · two populations */

const flowA = populationFlow(fixtures.flowA);
const flowB = populationFlow(fixtures.flowB);
nonEmpty([flowA, flowB], 2, 'population flows');
close(flowA.positives, 100, '1% of 10,000 is 100 positives', 1e-12);
close(flowA.tp, 80, 'eighty are detected', 1e-12);
close(flowA.fn, 20, 'twenty missed', 1e-12);
close(flowA.fp, 99, 'and 99 of 9,900 negatives raise an alarm', 1e-12);
close(flowA.tn, 9801, 'leaving 9,801 cleared', 1e-12);
close(flowA.precision, 80 / 179, 'precision is 80/179', 1e-12);
close(flowA.precision, recorded.constructed.population_shift.initial_precision, 'matching the packet', 1e-12);
assert(flowA.integral, 'at 1% prevalence every flow count is a whole number');
close(flowB.positives, 10, 'at 0.1% prevalence there are ten positives', 1e-12);
close(flowB.tp, 8, 'eight detected', 1e-12);
close(flowB.fp, 99.9, 'and an expected 99.9 false alarms', 1e-12);
close(flowB.tn, 9890.1, 'with an expected 9,890.1 cleared', 1e-12);
close(flowB.precision, 8 / 107.9, 'expected precision is 8/107.9', 1e-12);
close(flowB.precision, recorded.constructed.population_shift.shifted_precision, 'matching the packet', 1e-12);
assert(!flowB.integral, 'and its counts are deliberately fractional expectations, not a rounded census');
[flowA, flowB].forEach(flow => {
  assert(flow.conserved, 'the four flow counts sum back to the population');
  close(flow.precision, flow.precisionByFormula, 'the flow and the prevalence identity are one claim', 1e-12);
  close(flow.tp / flow.positives, flow.tpr, 'the detector keeps its declared true-positive rate', 1e-12);
  close(flow.fp / flow.negatives, flow.fpr, 'and its declared false-positive rate', 1e-12);
  record('population flow');
});
close(flowA.tpr, flowB.tpr, 'both panels share the same TPR, so the ROC marker cannot move', 1e-12);
close(flowA.fpr, flowB.fpr, 'and the same FPR', 1e-12);
assert(flowA.precision > 5 * flowB.precision, 'only the workload composition changed');
// Rounding 99.9 up first would give a visibly different precision, so say so.
close(8 / (8 + 100), 0.07407407407407407, 'rounding the expected false alarms first changes the answer', 1e-12);
assert(Math.abs(8 / (8 + 100) - flowB.precision) > 1e-5, 'which is why the fractional expectation is kept');
record('figure 2 flows');

/* ============================================ §3 · expected costs */

const risk = actionRisks(fixtures.casePosterior, fixtures.costs.costFP, fixtures.costs.costFN);
close(risk.selectRisk, 0.9, 'selecting costs .9 in expectation', 1e-12);
close(risk.skipRisk, 1.2, 'skipping costs 1.2', 1e-12);
close(risk.cutoff, 1 / 13, 'the cutoff is 1/13', 1e-12);
close(risk.cutoff, 0.07692307692307693, 'that is about .076923', 1e-12);
assert(risk.action === 'select', 'so selecting is preferable');
assert(risk.posterior > risk.cutoff, 'and the posterior does sit above the cutoff');
const contrast = actionRisks(fixtures.contrastPosterior, fixtures.costs.costFP, fixtures.costs.costFN);
close(contrast.selectRisk, 0.95, 'at p = .05 selecting costs .95', 1e-12);
close(contrast.skipRisk, 0.6, 'and skipping costs .6', 1e-12);
assert(contrast.action === 'skip', 'so the action flips');
close(contrast.cutoff, risk.cutoff, 'while the cutoff does not move with the posterior', 1e-12);
record('investigation 2 default and contrast');

// The common-factor null, over several factors and several posteriors.
const factors = [2, 3, 4, 0.5];
const posteriors = [0.02, 0.1, 0.5, 0.9];
let nullChecks = 0;
factors.forEach(factor => posteriors.forEach(posterior => {
  const base = actionRisks(posterior, fixtures.costs.costFP, fixtures.costs.costFN);
  const scaled = actionRisks(posterior, fixtures.costs.costFP * factor, fixtures.costs.costFN * factor);
  close(scaled.cutoff, base.cutoff, `scaling both costs by ${factor} leaves the cutoff`, 1e-12);
  assert(scaled.action === base.action, `and leaves the action at p = ${posterior}`);
  close(scaled.selectRisk, base.selectRisk * factor, 'while both risks scale by the factor', 1e-12);
  close(scaled.skipRisk, base.skipRisk * factor, 'both of them', 1e-12);
  nullChecks += 1;
}));
assert(nullChecks === 16, `the common-factor null was checked at ${nullChecks} settings`);
const tripled = actionRisks(0.1, 3, 36);
close(tripled.selectRisk, 2.7, 'tripling the declared costs gives risks 2.7', 1e-12);
close(tripled.skipRisk, 3.6, 'and 3.6', 1e-12);
record('investigation 2 common-factor null');

const practiceThree = actionRisks(fixtures.practiceCosts.posterior,
  fixtures.practiceCosts.costFP, fixtures.practiceCosts.costFN);
close(practiceThree.selectRisk, 1.6, 'practice 3: selecting costs 1.6', 1e-12);
close(practiceThree.skipRisk, 1.4, 'skipping costs 1.4', 1e-12);
close(practiceThree.cutoff, 2 / 9, 'and the cutoff is 2/9', 1e-12);
close(practiceThree.cutoff, recorded.constructed.changed_practice.cost_cutoff, 'matching the packet', 1e-12);
assert(practiceThree.action === 'skip', 'so the answer is to skip');
close(recorded.constructed.changed_practice.select_risk, 1.6, 'and the packet records the same select risk', 1e-12);
close(recorded.constructed.changed_practice.skip_risk, 1.4, 'and the same skip risk', 1e-9);
const atCutoff = actionRisks(2 / 9, 2, 7);
assert(atCutoff.tie && atCutoff.action === 'select', 'exactly at the cutoff the risks tie and the declared rule selects');
close(atCutoff.selectRisk, atCutoff.skipRisk, 'and the two risks really are equal there', 1e-12);
// Equal costs put the cutoff at .5 however rare the positive class is.
[1, 4, 17.5].forEach(cost => {
  close(actionRisks(0.001, cost, cost).cutoff, 0.5, 'equal costs give a cutoff of .5', 1e-12);
  record('equal-cost cutoff');
});
// The degenerate corners, handled rather than smoothed over.
const noAlarmCost = actionRisks(0.3, 0, 5);
assert(noAlarmCost.cutoff === 0 && noAlarmCost.action === 'select',
  'with no false-alarm cost every positive probability favours selection');
const noMissCost = actionRisks(0.3, 5, 0);
assert(noMissCost.cutoff === 1 && noMissCost.action === 'skip',
  'with no missed-positive cost every probability below one favours skipping');
const bothZero = actionRisks(0.3, 0, 0);
assert(bothZero.cutoff === null && bothZero.undefinedBecause !== null && bothZero.tie,
  'with both costs zero there is no cutoff at all, and the model says why');
record('investigation 2 corners');

/* ======================================== §4 · one weighted update */

const step = weightedStep({ rows: fixtures.stepRows, rate: fixtures.stepRate, penalty: fixtures.stepPenalty });
nonEmpty(step.rows, 2, 'weighted step rows');
close(step.totalWeight, 4, 'the two weights total four', 1e-12);
step.rows.forEach(row => {
  close(row.score, 0.5, 'at zero parameters both rows score one half', 1e-12);
  record('initial score');
});
close(step.rows[0].residual, 0.5, 'the first residual is .5', 1e-12);
close(step.rows[1].residual, -0.5, 'the second is −.5', 1e-12);
close(step.rows[0].interceptContribution, 0.5, 'its intercept contribution is .5', 1e-12);
close(step.rows[1].interceptContribution, -1.5, 'and the weighted one is −1.5', 1e-12);
close(step.rows[0].coefficientContribution, 0, 'a feature of zero contributes nothing to the coefficient', 1e-12);
close(step.rows[1].coefficientContribution, -3, 'and the other contributes −3', 1e-12);
close(step.interceptGradient, -0.25, 'the intercept gradient is −0.25', 1e-12);
close(step.coefficientGradient, -0.75, 'the coefficient gradient is −0.75', 1e-12);
close(step.penaltyContribution, 0, 'the penalty contributes nothing at zero parameters', 1e-12);
close(step.interceptAfter, 0.1, 'a step of .4 gives intercept .1', 1e-12);
close(step.coefficientAfter, 0.3, 'and coefficient .3', 1e-12);
vector(step.scoresAfter, recorded.constructed.weighted_single_step.scores_after,
  'the two updated scores match the packet', 1e-12);
close(step.scoresAfter[0], 0.524979, 'the first updated score is about .524979', 1e-5);
close(step.scoresAfter[1], 0.668188, 'the second is about .668188', 1e-5);
close(step.interceptGradient, recorded.constructed.weighted_single_step.intercept_gradient,
  'the packet records the same intercept gradient', 1e-12);
close(step.coefficientGradient, recorded.constructed.weighted_single_step.coefficient_gradient,
  'and the same coefficient gradient', 1e-12);
// The before and after curves are functions of the parameters, not drawn points.
[0, 0.5, 1, 1.5, 2].forEach(x => {
  close(step.before(x), 0.5, 'the initial curve is flat at one half', 1e-12);
  close(step.after(x), 1 / (1 + Math.exp(-(0.1 + 0.3 * x))), 'and the updated curve is the fitted sigmoid', 1e-12);
  record('drawn curve point');
});
assert(step.after(2) > step.after(0), 'the updated curve rises across the drawn interval');
// Scaling every weight by a common factor leaves this normalised gradient alone.
[2, 5, 0.25].forEach(factor => {
  const scaled = weightedStep({
    rows: fixtures.stepRows.map(row => ({ ...row, weight: row.weight * factor })),
    rate: fixtures.stepRate, penalty: fixtures.stepPenalty,
  });
  close(scaled.interceptGradient, step.interceptGradient, 'a common weight factor cancels in the intercept gradient', 1e-12);
  close(scaled.coefficientGradient, step.coefficientGradient, 'and in the coefficient gradient', 1e-12);
  record('weight-scaling null');
});
// A large weight is not another observation: three rows of weight one differ.
const threeRows = weightedStep({
  rows: [{ x: 0, y: 0, weight: 1 }, { x: 2, y: 1, weight: 1 }, { x: 2, y: 1, weight: 1 }],
  rate: fixtures.stepRate, penalty: fixtures.stepPenalty,
});
assert(Math.abs(threeRows.interceptGradient - step.interceptGradient) > 1e-6,
  'weight 3 on one row and three rows of weight one give different normalised gradients here');
record('figure 3 weighted step');

const balanced = balancedWeights([study.fittingNegatives, study.fittingPositives]);
assert(balanced.equalised, 'the balanced rule gives both classes equal total weight');
close(balanced.weights[1], study.positiveWeight, 'the positive weight is 600/42', 1e-12);
close(balanced.weights[0], study.negativeWeight, 'and the negative weight 600/1158', 1e-12);
close(balanced.weights[1], 600 / 42, 'recomputed from n/(K n_c)', 1e-12);
assert(!balanced.allOne, 'on the imbalanced fitting rows the balanced weights are not all one');
const afterResampling = balancedWeights([579, 579]);
assert(afterResampling.allOne, 'but after a resampler has equalised the counts they are');
record('balanced weights');

/* ================================== §4 · what a weighted score means */

const optimum = weightedOptimum(fixtures.weighted.probability,
  fixtures.weighted.positiveWeight, fixtures.weighted.negativeWeight);
close(optimum.optimum, 0.5, 'at p = .1 with weights 9 and 1 the optimum is exactly .5', 1e-12);
close(optimum.optimum, recorded.constructed.weighted_p_point_one, 'matching the packet', 1e-12);
close(inverseWeightedOptimum(0.5, 9, 1), 0.1, 'inverting it recovers p = .1', 1e-12);
close(optimum.equivalentProbabilityCutoff, 0.1, 'thresholding q* at .5 is thresholding p at 1/10', 1e-12);
const equalWeights = weightedOptimum(fixtures.weightedEqual.probability, 1, 1);
close(equalWeights.optimum, 0.1, 'with equal weights the optimum is the probability itself', 1e-12);
const doubled = weightedOptimum(fixtures.weightedDoubled.probability, 18, 2);
close(doubled.optimum, optimum.optimum, 'doubling both weights leaves the minimiser', 1e-12);
close(doubled.lossAtOptimum, 2 * optimum.lossAtOptimum, 'while the unnormalised loss height doubles', 1e-12);
assert(doubled.lossAtOptimum > optimum.lossAtOptimum + 1e-9,
  'so a shared vertical scale is needed to see the null is about the minimiser, not the curve');
const practiceFour = weightedOptimum(fixtures.weightedPractice.probability, 4, 1);
close(practiceFour.optimum, 0.5, 'practice 4: weights 4 and 1 at p = .2 also give .5', 1e-12);
close(practiceFour.optimum, recorded.constructed.changed_practice.weighted_optimum, 'matching the packet', 1e-12);
close(inverseWeightedOptimum(0.5, 4, 1), 0.2, 'and inverting recovers p = .2', 1e-12);
close(inverseWeightedOptimum(0.5, 4, 1), recorded.constructed.changed_practice.inverse_probability,
  'as the packet records', 1e-12);
record('investigation 3 stated values');

// The optimum really is a stationary interior minimum, checked analytically and
// by a central difference, across a grid of settings rather than at one point.
const optimumGrid = [];
[0.01, 0.05, 0.1, 0.2, 0.5, 0.8, 0.99].forEach(p =>
  [0.1, 1, 4, 9, 18, 20].forEach(wp =>
    [0.1, 1, 2, 20].forEach(wn => optimumGrid.push(weightedOptimum(p, wp, wn)))));
nonEmpty(optimumGrid, 168, 'weighted-optimum grid');
let roundTrips = 0;
optimumGrid.forEach(entry => {
  assert(entry.optimum > 0 && entry.optimum < 1, 'the optimum is strictly interior');
  close(entry.derivativeAtOptimum, 0, 'the analytic derivative vanishes there', 1e-9);
  assert(entry.secondDerivativeAtOptimum > 0, 'and the second derivative is positive');
  // The derivative formula itself is checked against a central difference at
  // well-conditioned interior probes, where a fixed step is safe. Checking it
  // at an extreme optimum instead would measure floating-point cancellation.
  [0.25, 0.5, 0.75].forEach(probe => {
    const h = 1e-6;
    const numerical = (entry.loss(probe + h) - entry.loss(probe - h)) / (2 * h);
    close(numerical, entry.derivative(probe),
      `the analytic derivative matches a central difference at q = ${probe}`, 1e-5);
    const secondNumerical = (entry.loss(probe + h) - 2 * entry.loss(probe) + entry.loss(probe - h)) / h ** 2;
    close(secondNumerical, entry.secondDerivative(probe),
      `and so does the second derivative at q = ${probe}`, 1e-3);
  });
  // A step that stays inside (0, 1) however extreme the optimum is, so this is
  // a claim about the minimum rather than an artefact of a fixed offset.
  const offset = Math.min(0.01, entry.optimum / 2, (1 - entry.optimum) / 2);
  assert(offset > 0, 'the probe offset is positive');
  assert(entry.loss(entry.optimum) < entry.loss(entry.optimum + offset), 'the loss rises to its right');
  assert(entry.loss(entry.optimum) < entry.loss(entry.optimum - offset), 'and to its left');
  // The inverse is a learner control with its own bounds, so it is exercised
  // wherever the optimum is enterable — and the count is asserted so this
  // branch cannot quietly become unreachable.
  if (entry.optimum >= limits.weightedScore.minimum && entry.optimum <= limits.weightedScore.maximum) {
    close(inverseWeightedOptimum(entry.optimum, entry.positiveWeight, entry.negativeWeight),
      entry.probability, 'the forward and inverse maps round-trip', 1e-9);
    roundTrips += 1;
  }
  record('weighted optimum grid point');
});
assert(roundTrips > 80, `the forward/inverse round trip was checked ${roundTrips} times`);
// Only the ratio matters, over the same grid.
let ratioChecks = 0;
optimumGrid.forEach(entry => {
  [2, 10].forEach(factor => {
    if (entry.positiveWeight * factor > limits.weight.maximum || entry.negativeWeight * factor > limits.weight.maximum) return;
    const scaled = weightedOptimum(entry.probability, entry.positiveWeight * factor, entry.negativeWeight * factor);
    close(scaled.optimum, entry.optimum, 'a common weight factor leaves the optimum', 1e-12);
    ratioChecks += 1;
  });
});
assert(ratioChecks > 50, `the weight-ratio null was checked ${ratioChecks} times`);
record('weighted optimum ratio null');

const curve = weightedLossCurve(0.1, 9, 1);
nonEmpty(curve.points, limits.curveSamples, 'weighted loss curve samples');
assert(curve.points.every(([q]) => q > 0 && q < 1), 'every sampled q lies strictly inside the open interval');
assert(curve.points.every(([, value]) => Number.isFinite(value)), 'so no sample evaluates log 0');
close(curve.exactMinimum, 0.5, 'the formula supplies the exact minimum', 1e-12);
assert(Math.abs(curve.sampledMinimum - curve.exactMinimum) < 0.02,
  'and the sampled grid lands near it without being treated as authoritative');
record('weighted loss curve');

/* ================================================== §5 · SMOTE geometry */

const smote = smoteConstruction({ points: fixtures.cloud, ...fixtures.cloudSetup });
close(smote.generated.x, 1, 'the default construction generates x = 1', 1e-12);
close(smote.generated.y, 0, 'and y = 0', 1e-12);
assert(smote.neighbour.id === 'B', 'B is the nearest minority point under the default metric');
assert(smote.collidesWithMajority && smote.collisionIds.includes('M'),
  'the generated point lands exactly on the majority point M');
assert.deepEqual(smote.consulted, ['A', 'B'], 'the construction consulted only the two minority endpoints');
assert.deepEqual(smote.ignored, ['M', 'N'], 'and never looked at either majority point');
vector([smote.generated.x, smote.generated.y], recorded.constructed.synthetic,
  'the packet records the same declared synthetic point', 1e-12);
nonEmpty(smote.distances, 2, 'the minority distance row');
close(smote.distances[0].distance, 2, 'A to B is distance 2', 1e-12);
close(smote.distances[1].distance, 2.5, 'A to C is distance 2.5', 1e-12);
record('investigation 4 default');

// The exact null: move only the majority point. The output must not move.
const movedCloud = fixtures.cloud.map(point =>
  (point.id === 'M' ? { ...point, ...fixtures.cloudNullMajority } : point));
const moved = smoteConstruction({ points: movedCloud, ...fixtures.cloudSetup });
close(moved.generated.x, smote.generated.x, 'moving the majority point leaves the generated x', 1e-12);
close(moved.generated.y, smote.generated.y, 'and the generated y', 1e-12);
assert(moved.neighbour.id === smote.neighbour.id, 'and leaves the chosen neighbour');
vector(moved.distances.map(entry => entry.distance), smote.distances.map(entry => entry.distance),
  'and every minority distance', 1e-12);
assert(!moved.collidesWithMajority, 'only the visible overlap changed');
record('investigation 4 majority-move null');
// A sweep of majority positions, so the null is a property and not one lucky move.
const majoritySweep = [];
[-4, -1, 0, 1, 3, 5].forEach(x => [-4, 0, 1.5, 4].forEach(y => {
  const cloud = fixtures.cloud.map(point => (point.id === 'M' ? { ...point, x, y } : point));
  const result = smoteConstruction({ points: cloud, ...fixtures.cloudSetup });
  close(result.generated.x, 1, 'the generated x is unmoved wherever M goes', 1e-12);
  close(result.generated.y, 0, 'and so is the generated y', 1e-12);
  majoritySweep.push(result);
}));
nonEmpty(majoritySweep, 24, 'majority-position sweep');
record('investigation 4 majority sweep');

// The geometric contrast: a changed divisor changes which neighbour is nearest.
const rescaled = smoteConstruction({
  points: fixtures.cloud, ...fixtures.cloudSetup, scaleY: fixtures.cloudContrastScaleY,
});
assert(rescaled.neighbour.id === 'C', 'with the y divisor at 10 the nearest minority point becomes C');
close(rescaled.distances[0].distance, 0.25, 'the scaled A–C distance is .25', 1e-12);
close(rescaled.distances[1].distance, 2, 'against a scaled A–B distance of 2', 1e-12);
close(rescaled.generated.x, 0, 'so the generated point is (0, 1.25): x is 0', 1e-12);
close(rescaled.generated.y, 1.25, 'and y is 1.25', 1e-12);
assert(!rescaled.collidesWithMajority, 'and it no longer collides with anything');
record('investigation 4 metric contrast');

// Endpoints, the whole segment, and one scalar fraction for the entire vector.
const fractionSweep = [0, 0.1, 0.25, 0.5, 0.75, 0.9, 1].map(fraction =>
  ({ fraction, result: smoteConstruction({ points: fixtures.cloud, ...fixtures.cloudSetup, fraction }) }));
nonEmpty(fractionSweep, 7, 'fraction sweep');
fractionSweep.forEach(({ fraction, result }) => {
  close(result.generated.x, 0 + fraction * 2, 'x interpolates with the shared fraction', 1e-12);
  close(result.generated.y, 0 + fraction * 0, 'and y uses the same one', 1e-12);
  record('fraction sweep point');
});
close(fractionSweep[0].result.generated.x, 0, 'a fraction of zero returns the anchor exactly', 1e-12);
close(fractionSweep.at(-1).result.generated.x, 2, 'and a fraction of one returns the neighbour exactly', 1e-12);
close(fractionSweep[2].result.generated.x, 0.5, 'a quarter of the way gives (0.5, 0)', 1e-12);
// Practice 5: a different anchor, neighbour and fraction.
const practiceFive = smoteConstruction({
  points: [
    { id: 'A', cls: 'minority', x: fixtures.practiceInterpolation.anchor[0], y: fixtures.practiceInterpolation.anchor[1] },
    { id: 'B', cls: 'minority', x: fixtures.practiceInterpolation.neighbour[0], y: fixtures.practiceInterpolation.neighbour[1] },
  ],
  anchorId: 'A', k: 1, neighbourRank: 0, fraction: fixtures.practiceInterpolation.fraction,
});
vector([practiceFive.generated.x, practiceFive.generated.y], fixtures.practiceInterpolation.expected,
  'practice 5 generates (2, 2.5)', 1e-12);
vector([practiceFive.generated.x, practiceFive.generated.y], recorded.constructed.changed_practice.smote_point,
  'matching the packet', 1e-12);
assert(practiceFive.availableNeighbours === 1, 'with two minority rows there is exactly one other neighbour');
refuses(() => smoteConstruction({
  points: [
    { id: 'A', cls: 'minority', x: 0, y: 0 }, { id: 'B', cls: 'minority', x: 1, y: 1 },
    { id: 'C', cls: 'minority', x: 2, y: 2 },
  ], anchorId: 'A', k: 5, neighbourRank: 0, fraction: 0.5,
}), 'five-neighbour SMOTE in a fold with three minority rows');
// A zero-length segment is a legitimate input, not an error.
const coincident = smoteConstruction({
  points: [
    { id: 'A', cls: 'minority', x: 1, y: 1 }, { id: 'B', cls: 'minority', x: 1, y: 1 },
    { id: 'M', cls: 'majority', x: 0, y: 0 },
  ], anchorId: 'A', k: 1, neighbourRank: 0, fraction: 0.37,
});
assert(coincident.zeroLength, 'two identical minority observations give a zero-length segment');
close(coincident.generated.x, 1, 'which generates that same location', 1e-12);
close(coincident.distances[0].distance, 0, 'at distance zero', 1e-12);
// A distance tie is disclosed rather than silently resolved by sort order.
const tiedCloud = [
  { id: 'A', cls: 'minority', x: 0, y: 0 },
  { id: 'B', cls: 'minority', x: 2, y: 0 },
  { id: 'C', cls: 'minority', x: -2, y: 0 },
  { id: 'D', cls: 'minority', x: 0, y: 4 },
];
const tiedSmote = smoteConstruction({ points: tiedCloud, anchorId: 'A', k: 1, neighbourRank: 0, fraction: 0.5 });
assert(tiedSmote.boundaryTie, 'a tie straddling the k boundary is reported');
assert(tiedSmote.tiedWithChosen.includes('C'), 'and the tied identity is named');
assert(tiedSmote.neighbour.id === 'B', 'while stable identifier order decides which one is used');
const tiedAtTwo = smoteConstruction({ points: tiedCloud, anchorId: 'A', k: 2, neighbourRank: 1, fraction: 0.5 });
assert(!tiedAtTwo.boundaryTie, 'with k = 2 the tie no longer straddles the boundary');
assert(tiedAtTwo.neighbour.id === 'C', 'and the second ranked neighbour is the other tied point');
record('investigation 4 edge cases');

/* ================================== §7 · the recorded Yeast outcomes */

nonEmpty(methods, 5, 'recorded procedures');
assert.deepEqual(methods.map(entry => entry.name),
  ['original', 'balanced_weight', 'random_over', 'random_under', 'smote'],
  'the five procedures are in their declared order');
assert(tuningRecords.labels.length === 200 && inspectionRecords.labels.length === 200,
  'the tuning and inspection roles each hold 200 proteins');
assert(tuningRecords.sourceIds.length === 200 && tuningRecords.proteinIds.length === 200,
  'with a source index and a protein identifier for each');
assert(new Set(tuningRecords.sourceIds).size === 200, 'the tuning identities are distinct');
assert(new Set(inspectionRecords.sourceIds).size === 200, 'and so are the inspection identities');
assert(!tuningRecords.sourceIds.some(id => inspectionRecords.sourceIds.includes(id)),
  'and the two roles share no protein');
close(tuningRecords.labels.filter(label => label === 1).length, 7, 'seven tuning positives', 1e-12);
close(inspectionRecords.labels.filter(label => label === 1).length, 7, 'seven inspection positives', 1e-12);
assert(provenance.reserveScored === false && roles.reserve.records === 462,
  'the 462 reserved proteins carry no score anywhere in this module');
assert(!Object.keys(roles).includes('reserveScores'), 'and no reserved score is published');
close(roles.fitting.records + roles.tuning.records + roles.inspection.records + roles.reserve.records,
  provenance.studyRows, 'the four roles cover the 1,462 retained proteins', 1e-12);
close(roles.fitting.positives + roles.tuning.positives + roles.inspection.positives + roles.reserve.positives,
  provenance.positives, 'and account for all 51 positives', 1e-12);
record('recorded roles');

const statedTable = {
  original: { threshold: 0.145006, tuned: [2, 12, 5, 181], cost: 72, ap: 0.173459, topTen: 2 },
  balanced_weight: { threshold: 0.740873, tuned: [4, 15, 3, 178], cost: 51, ap: 0.273317, topTen: 1 },
  random_over: { threshold: 0.706116, tuned: [4, 17, 3, 176], cost: 53, ap: 0.278679, topTen: 2 },
  random_under: { threshold: 0.772569, tuned: [4, 19, 3, 174], cost: 55, ap: 0.193550, topTen: 2 },
  smote: { threshold: 0.759272, tuned: [4, 13, 3, 180], cost: 49, ap: 0.272615, topTen: 1 },
};
const statedDefault = {
  original: [0, 1, 7, 192], balanced_weight: [4, 35, 3, 158], random_over: [4, 34, 3, 159],
  random_under: [5, 41, 2, 152], smote: [4, 29, 3, 164],
};
let sweptCandidates = 0;
methods.forEach(method => {
  const stored = recorded.methods.find(entry => entry.name === method.name);
  assert(stored, `${method.name} appears in the packet`);
  const expected = statedTable[method.name];
  // Recompute the published confusion counts from the published scores.
  const tuned = countsAt(inspectionRecords.labels, method.inspectionScores, method.chosenThreshold);
  const atHalf = countsAt(inspectionRecords.labels, method.inspectionScores, 0.5);
  assert.deepEqual([tuned.tp, tuned.fp, tuned.fn, tuned.tn], expected.tuned,
    `${method.name}: the manuscript's tuned counts come out of the saved scores`);
  assert.deepEqual([atHalf.tp, atHalf.fp, atHalf.fn, atHalf.tn], statedDefault[method.name],
    `${method.name}: and so do its counts at 0.5`);
  close(costOf(tuned, study.costFalsePositive, study.costFalseNegative), expected.cost,
    `${method.name}: realised cost FP + 12 FN`, 1e-12);
  close(tuned.total, 200, `${method.name}: the counts total 200`, 1e-12);
  close(tuned.positives, 7, `${method.name}: with seven positives`, 1e-12);
  close(method.chosenThreshold, expected.threshold, `${method.name}: printed threshold`, 5e-6);
  close(method.chosenThreshold, stored.chosen_threshold, `${method.name}: exact threshold from the packet`, 1e-15);
  // Reselect the threshold from the tuning scores, sweeping every candidate.
  const sweep = thresholdSweep(tuningRecords.labels, method.tuningScores,
    study.costFalsePositive, study.costFalseNegative);
  nonEmpty(sweep.candidates, undefined, `${method.name}: threshold candidates`);
  close(sweep.chosen.threshold, method.chosenThreshold, `${method.name}: the sweep reselects it`, 1e-15);
  assert(sweep.tieKeptHigher, `${method.name}: a cost tie kept the higher threshold`);
  sweep.candidates.forEach(candidate => {
    assert(candidate.cost >= sweep.chosen.cost, `${method.name}: no candidate beats the selected cost`);
    if (candidate.cost === sweep.chosen.cost) {
      assert(candidate.threshold <= sweep.chosen.threshold, `${method.name}: and no tied candidate is higher`);
    }
    sweptCandidates += 1;
  });
  // Metrics, recomputed from definitions rather than read back.
  close(averagePrecisionOfScores(inspectionRecords.labels, method.inspectionScores), method.averagePrecision,
    `${method.name}: average precision`, 1e-12);
  close(method.averagePrecision, expected.ap, `${method.name}: the printed AP`, 5e-6);
  close(rocAucOfScores(inspectionRecords.labels, method.inspectionScores), method.rocAuc,
    `${method.name}: ROC-AUC`, 1e-12);
  close(brierOfScores(inspectionRecords.labels, method.inspectionScores), method.brierScore,
    `${method.name}: Brier score`, 1e-12);
  // The top ten, ranked without consulting a label.
  const queue = rankedQueue(inspectionRecords.sourceIds, inspectionRecords.labels, method.inspectionScores, 10);
  assert.deepEqual(queue.top.map(entry => entry.id), method.topTenSourceIds,
    `${method.name}: the top-ten identities`);
  close(queue.positives, method.topTenPositives, `${method.name}: the top-ten positive count`, 1e-12);
  close(queue.positives, expected.topTen, `${method.name}: as the manuscript states`, 1e-12);
  assert(method.inspectionScores.length === 200 && method.tuningScores.length === 200,
    `${method.name}: 200 saved scores in each role`);
  assert(method.inspectionScores.every(score => score > 0 && score < 1),
    `${method.name}: every saved score is a probability strictly inside (0, 1)`);
  record('recorded procedure');
});
assert(sweptCandidates > 500, `the threshold sweeps inspected ${sweptCandidates} candidates`);

// The three disagreeing winners the narrative depends on.
const lowestCost = methods.reduce((best, entry) =>
  (entry.inspectionTuned.cost < best.inspectionTuned.cost ? entry : best));
const highestAp = methods.reduce((best, entry) => (entry.averagePrecision > best.averagePrecision ? entry : best));
const bestTopTen = Math.max(...methods.map(entry => entry.topTenPositives));
const topWinners = methods.filter(entry => entry.topTenPositives === bestTopTen).map(entry => entry.name).sort();
assert(lowestCost.name === 'smote' && lowestCost.inspectionTuned.cost === 49,
  'SMOTE has the lowest realised tuned cost, 49');
assert(highestAp.name === 'random_over', 'random oversampling has the highest average precision');
assert.deepEqual(topWinners, ['original', 'random_over', 'random_under'],
  'and three procedures share the best top-ten count');
assert(lowestCost.name !== highestAp.name, 'the two questions have different winners');
assert(!topWinners.includes(lowestCost.name), 'and the cost winner is not among the top-ten winners either');
close(study.baselineCost, 84, 'the always-negative baseline costs 84', 1e-12);
close(study.baselineAccuracy, 193 / 200, 'and is 96.5% accurate', 1e-12);
const emptySelection = countsAt(inspectionRecords.labels, methods[0].inspectionScores, Infinity);
assert(emptySelection.precision === null, 'the baseline selects nothing, so its precision is undefined');
close(costOf(emptySelection, study.costFalsePositive, study.costFalseNegative), 84,
  'and its cost is 12 × 7 = 84', 1e-12);
methods.forEach(method => {
  assert(method.inspectionTuned.cost < study.baselineCost,
    `${method.name}: the tuned threshold beats the no-alert policy`);
  record('beats the baseline');
});
// Brier: the three values the manuscript quotes, and what they do not prove.
const original = methods.find(entry => entry.name === 'original');
close(original.brierScore, 0.033475, 'the original model scores .033475', 5e-7);
close(methods.find(entry => entry.name === 'balanced_weight').brierScore, 0.131518,
  'balanced weighting .131518', 5e-7);
close(methods.find(entry => entry.name === 'smote').brierScore, 0.121856, 'and SMOTE .121856', 5e-7);
assert(original.brierScore < lowestCost.brierScore && original.inspectionTuned.cost > lowestCost.inspectionTuned.cost,
  'the smaller Brier value belongs to the procedure with the worse realised cost');
record('section 7 outcomes');

// One extra detection moves recall by 1/7.
close(1 / 7, 0.14285714285714285, 'one more detection changes recall by about .143', 1e-12);
const sevenPositives = confusion({ tp: 4, fp: 13, fn: 3, tn: 180 });
const oneMore = confusion({ tp: 5, fp: 13, fn: 2, tn: 180 });
close(oneMore.recall - sevenPositives.recall, 1 / 7, 'measured on the actual inspection counts', 1e-12);
record('small-sample sensitivity');

/* ============================ §7 · investigation 5 on real records */

const i5 = methods.find(entry => entry.name === 'original');
const atFive = countsAt(tuningRecords.labels, i5.tuningScores, 0.5);
const atFifteen = countsAt(tuningRecords.labels, i5.tuningScores, 0.15);
assert.deepEqual([atFive.tp, atFive.fp, atFive.fn, atFive.tn], [1, 0, 6, 193],
  'the tuning queue at .5 gives TP 1, FP 0, FN 6, TN 193');
assert.deepEqual([atFifteen.tp, atFifteen.fp, atFifteen.fn, atFifteen.tn], [5, 4, 2, 189],
  'and at .15 gives TP 5, FP 4, FN 2, TN 189');
close(costOf(atFive, 1, 12), 72, 'cost 72 at .5', 1e-12);
close(costOf(atFifteen, 1, 12), 28, 'and cost 28 at .15', 1e-12);
assert(atFifteen.tp > atFive.tp, 'lowering the gate detects more positives');
assert(costOf(atFifteen, 1, 12) < costOf(atFive, 1, 12), 'and lowers the realised cost');
close(atFive.precision, 1, 'while precision falls from 1', 1e-12);
close(atFifteen.precision, 5 / 9, 'to 5/9', 1e-12);
assert(atFifteen.precision < atFive.precision, 'so better recall and cost came with worse precision');
const packetFixture = recorded.methods.find(entry => entry.name === 'original').tuning_investigation_fixture;
assert(packetFixture['0.5'].tp === 1 && packetFixture['0.15'].tp === 5,
  'and the packet records the same fixture');
close(packetFixture['0.5'].cost, 72, 'with the same cost at .5', 1e-12);
close(packetFixture['0.15'].cost, 28, 'and at .15', 1e-12);
record('investigation 5 fixture');

// The exact cost-scaling null on the real tuning records, for every method.
methods.forEach(method => {
  const base = thresholdSweep(tuningRecords.labels, method.tuningScores, 1, 12);
  [2, 4, 0.5].forEach(factor => {
    const scaled = thresholdSweep(tuningRecords.labels, method.tuningScores, factor, 12 * factor);
    close(scaled.chosen.threshold, base.chosen.threshold,
      `${method.name}: scaling both costs by ${factor} leaves the selected threshold`, 1e-15);
    close(scaled.chosen.cost, base.chosen.cost * factor, 'while the total cost scales with the factor', 1e-9);
    assert.deepEqual(
      [scaled.chosen.counts.tp, scaled.chosen.counts.fp],
      [base.chosen.counts.tp, base.chosen.counts.fp],
      'and the selected records are identical');
    record('cost-scaling null on real records');
  });
  // A moved threshold cannot move the ranking or the average precision.
  const apAtRest = averagePrecisionOfScores(tuningRecords.labels, method.tuningScores);
  [0.1, 0.3, 0.5, 0.9].forEach(trial => {
    const point = countsAt(tuningRecords.labels, method.tuningScores, trial);
    assert(point.total === 200, 'a trial threshold partitions all 200 records');
    close(averagePrecisionOfScores(tuningRecords.labels, method.tuningScores), apAtRest,
      'and leaves average precision exactly where it was', 1e-15);
    record('ranking null');
  });
});
// Changing the missed-positive cost can move the optimum, or leave it: both are
// informative, and the discreteness of the candidate set is why.
const costResponse = [1, 3, 6, 12, 24, 48].map(costFN => ({
  costFN, threshold: thresholdSweep(tuningRecords.labels, i5.tuningScores, 1, costFN).chosen.threshold,
}));
nonEmpty(costResponse, 6, 'missed-positive cost response');
assert(new Set(costResponse.map(entry => entry.threshold)).size > 1,
  'raising the missed-positive cost does move the selected threshold somewhere in this range');
assert(costResponse.every((entry, index) => index === 0 || entry.threshold <= costResponse[index - 1].threshold + 1e-12),
  'and a costlier miss never raises the gate');
record('investigation 5 cost response');

/* ================================================= §8 · deeper branches */

const focal = focalMass(fixtures.focal);
nonEmpty(focal.rows, 2, 'focal loss groups');
close(focal.rows[0].crossEntropyTotal, 1053.6051565782627, 'ten thousand easy examples total 1,053.605157', 1e-9);
close(focal.rows[1].crossEntropyTotal, 16.094379124341003, 'ten difficult ones total 16.094379', 1e-12);
close(focal.rows[0].focalTotal, 10.536051565782623, 'under focal loss the easy total is 10.536052', 1e-9);
close(focal.rows[1].focalTotal, 10.300402639578245, 'and the difficult total is 10.300403', 1e-12);
close(focal.rows[0].modulator, 0.01, 'an easy example at p_t = .9 is multiplied by .01', 1e-12);
close(focal.rows[1].modulator, 0.64, 'a difficult one at p_t = .2 by .64', 1e-12);
const packetFocal = recorded.constructed.focal_loss_mass;
close(focal.rows[0].crossEntropyTotal, packetFocal.easy_ce_total, 'matching the packet', 1e-12);
close(focal.rows[1].focalTotal, packetFocal.hard_focal_total, 'and so does the difficult focal total', 1e-12);
assert(focal.crossEntropyTotal > 60 * focal.rows[1].crossEntropyTotal,
  'under cross-entropy the easy population dominates the loss mass');
assert(Math.abs(focal.rows[0].focalTotal - focal.rows[1].focalTotal) < 0.3,
  'under focal loss the two masses become comparable');
close(focal.shares.crossEntropy[0] + focal.shares.crossEntropy[1], 1, 'the shares are shares', 1e-12);
const gammaZero = focalMass({ ...fixtures.focal, gamma: 0 });
assert(gammaZero.reducesToCrossEntropy, 'at gamma 0 this is weighted cross-entropy');
close(gammaZero.focalTotal, gammaZero.crossEntropyTotal, 'and its totals agree exactly', 1e-12);
record('figure 6 loss mass');

// The derivative keeps its product-rule term; a central difference agrees.
const derivatives = [0.05, 0.2, 0.5, 0.9, 0.99].map(pt => focalDerivative(pt, 2, 1));
nonEmpty(derivatives, 5, 'focal derivatives');
derivatives.forEach(entry => {
  const h = 1e-7;
  const loss = pt => -entry.alpha * (1 - pt) ** entry.gamma * Math.log(pt);
  const numerical = (loss(entry.pt + h) - loss(entry.pt - h)) / (2 * h);
  close(entry.value, numerical, `the focal derivative at p_t = ${entry.pt}`, 2e-6);
  assert(Math.abs(entry.modulatorTerm) > 1e-12, 'and its first term is not zero, so dropping it would be wrong');
  assert(Math.abs(entry.value - entry.crossEntropyOnly) > 1e-9,
    'the "cross-entropy gradient times the modulator" claim gives a different number');
  record('focal derivative');
});
close(focalDerivative(0.5, 0, 1).modulatorTerm, 0, 'at gamma 0 the extra term vanishes', 1e-12);
record('focal derivative term');

const prior = priorShift(fixtures.prior);
close(prior.sampledOdds, 4, 'a sampled posterior of .8 is odds 4', 1e-12);
close(prior.multiplier, 1 / 99, 'the odds multiplier is 1/99', 1e-12);
close(prior.posterior, 4 / 103, 'so the deployment posterior is 4/103', 1e-12);
close(prior.posterior, 0.038834951456310676, 'about .038835', 1e-12);
close(prior.posterior, recorded.constructed.prior_q_point_eight, 'matching the packet', 1e-12);
const priorPractice = priorShift(fixtures.priorPractice);
close(priorPractice.multiplier, 4 / 49, 'practice 8: the odds multiplier is 4/49', 1e-12);
close(priorPractice.posterior, 4 / 53, 'and the deployment posterior 4/53', 1e-12);
close(priorPractice.posterior, recorded.constructed.changed_practice.prior_probability, 'matching the packet', 1e-12);
const neutral = priorShift({ sampledPosterior: 0.8, sampledPrevalence: 0.3, deploymentPrevalence: 0.3 });
close(neutral.multiplier, 1, 'an unchanged prior has multiplier one', 1e-12);
close(neutral.posterior, 0.8, 'and leaves the posterior alone', 1e-12);
const balancedCorrection = priorShift({ sampledPosterior: 0.5, sampledPrevalence: 0.5, deploymentPrevalence: 0.01 });
close(balancedCorrection.posterior, 0.01, 'a balanced .5 corrects to the deployment prevalence', 1e-12);
close(balancedCorrection.posterior, recorded.constructed.prior_corrected_q_point_five, 'matching the packet', 1e-12);
record('prior shift');

const expansion = rowExpansion(fixtures.expansion.majority, fixtures.expansion.minority);
close(expansion.oversampledRows, 198, 'a 99-to-1 set becomes 198 rows', 1e-12);
close(expansion.originalRows, 100, 'from 100', 1e-12);
close(expansion.factor, 1.98, 'a factor of 1.98, not a hundredfold', 1e-12);
close(expansion.factor, recorded.constructed.expansion_99_to_1, 'matching the packet', 1e-12);
assert(expansion.factor < 2, 'the balancing expansion is always below two');
close(expansion.undersampledRows, 2, 'undersampling leaves 2m rows', 1e-12);
const practiceNine = rowExpansion(fixtures.expansionPractice.majority, fixtures.expansionPractice.minority);
close(practiceNine.oversampledRows, 1920, 'practice 9: oversampling gives 1,920 rows', 1e-12);
close(practiceNine.undersampledRows, 80, 'undersampling gives 80', 1e-12);
close(practiceNine.factor, 1.92, 'a 1.92-fold expansion', 1e-12);
close(practiceNine.independentMinorityObservations, 40,
  'and the 40 underlying minority observations are unchanged', 1e-12);
close(practiceNine.oversampledRows, recorded.constructed.changed_practice.oversampled_rows, 'matching the packet', 1e-12);
close(practiceNine.undersampledRows, recorded.constructed.changed_practice.undersampled_rows, 'and so does the other', 1e-12);
// The expansion factor is below two for every admissible pair, not only these.
const expansionSweep = [];
[10, 100, 1000, 99999].forEach(majority => [1, 2, 9, 10].forEach(minority => {
  if (minority > majority) return;
  const entry = rowExpansion(majority, minority);
  assert(entry.factor < 2 && entry.factor >= 1, 'the expansion factor stays in [1, 2)');
  expansionSweep.push(entry);
}));
nonEmpty(expansionSweep, 16, 'expansion sweep');
const work = neighbourWork(21, 6, 558);
close(work.distanceArithmetic, 21 ** 2 * 6, 'the all-pairs distance work is m² d', 1e-12);
close(work.storedDistances, 441, 'with m² stored distances', 1e-12);
close(work.generationArithmetic, 558 * 6, 'and generation costs G d', 1e-12);
record('section 8 budget');

/* ============================================ refused input */

refuses(() => confusion({ tp: -1, fp: 0, fn: 0, tn: 1 }), 'a negative cell count');
refuses(() => confusion({ tp: 0, fp: 0, fn: 0, tn: 0 }), 'an empty confusion table');
refuses(() => confusion({ tp: 1.5, fp: 0, fn: 0, tn: 1 }), 'a fractional cell count');
refuses(() => fBeta(fixtures.modelCounts, -1), 'a negative beta');
refuses(() => queueAt([], 0.5), 'an empty queue');
refuses(() => queueAt(Array.from({ length: 13 }, (_, index) => ({ id: `R${index}`, score: 0.5, truth: 0 })), 0.5),
  'a queue past its bound');
refuses(() => queueAt([{ id: 'A', score: 1.5, truth: 0 }], 0.5), 'a score outside [0, 1]');
refuses(() => queueAt([{ id: 'A', score: 0.5, truth: 2 }], 0.5), 'a truth value that is not 0 or 1');
refuses(() => queueAt([{ id: 'A', score: 0.5, truth: 0 }, { id: 'A', score: 0.4, truth: 1 }], 0.5),
  'a repeated record identifier');
refuses(() => populationFlow({ population: 0, prevalence: 0.1, tpr: 0.8, fpr: 0.01 }), 'an empty population');
refuses(() => populationFlow({ population: 100, prevalence: 1.5, tpr: 0.8, fpr: 0.01 }), 'a prevalence above one');
refuses(() => actionRisks(1.2, 1, 12), 'a posterior above one');
refuses(() => actionRisks(0.1, -1, 12), 'a negative cost');
refuses(() => actionRisks(0.1, 1, 500), 'a cost past its bound');
refuses(() => weightedStep({ rows: [] }), 'a weighted step with no rows');
refuses(() => weightedStep({ rows: [{ x: 0, y: 0, weight: 0 }] }), 'a zero weight');
refuses(() => weightedStep({ rows: [{ x: 0, y: 2, weight: 1 }] }), 'a label that is not 0 or 1');
refuses(() => weightedStep({ rows: [{ x: Number.NaN, y: 0, weight: 1 }] }), 'a non-finite feature');
refuses(() => balancedWeights([10]), 'balanced weights for a single class');
refuses(() => balancedWeights([10, 0]), 'a class with no observations');
refuses(() => weightedOptimum(0, 9, 1), 'a probability at the endpoint');
refuses(() => weightedOptimum(0.1, 0, 1), 'a zero class weight');
refuses(() => weightedOptimum(0.1, 9, -1), 'a negative class weight');
refuses(() => inverseWeightedOptimum(1, 9, 1), 'an inverse at the endpoint');
refuses(() => smoteConstruction({ points: [{ id: 'A', cls: 'minority', x: 0, y: 0 }], anchorId: 'A' }),
  'a cloud with one minority point');
refuses(() => smoteConstruction({ points: fixtures.cloud, ...fixtures.cloudSetup, anchorId: 'M' }),
  'a majority point as the anchor');
refuses(() => smoteConstruction({ points: fixtures.cloud, ...fixtures.cloudSetup, k: 0 }), 'k below one');
refuses(() => smoteConstruction({ points: fixtures.cloud, ...fixtures.cloudSetup, neighbourRank: 3 }),
  'a neighbour rank outside the chosen k');
refuses(() => smoteConstruction({ points: fixtures.cloud, ...fixtures.cloudSetup, fraction: 1.5 }),
  'an interpolation fraction above one');
refuses(() => smoteConstruction({ points: fixtures.cloud, ...fixtures.cloudSetup, scaleX: 0 }), 'a zero scale divisor');
refuses(() => smoteConstruction({
  points: [{ id: 'A', cls: 'minority', x: 9, y: 0 }, { id: 'B', cls: 'minority', x: 0, y: 0 }], anchorId: 'A',
}), 'a coordinate outside the enterable range');
refuses(() => countsAt([1, 0], [0.5], 0.5), 'labels and scores of different lengths');
refuses(() => countsAt([], [], 0.5), 'an empty recorded queue');
refuses(() => focalMass({ groups: [], gamma: 2 }), 'focal mass with no groups');
refuses(() => focalMass({ groups: [{ name: 'a', count: 0, pt: 0.5 }] }), 'a group with no examples');
refuses(() => focalMass({ groups: fixtures.focal.groups, gamma: -1 }), 'a negative gamma');
refuses(() => focalDerivative(1.5), 'a true-class probability above one');
refuses(() => priorShift({ sampledPosterior: 0.8, sampledPrevalence: 0, deploymentPrevalence: 0.01 }),
  'a degenerate sampled prevalence');
refuses(() => rowExpansion(10, 20), 'a minority class larger than the majority');
refuses(() => rowExpansion(0, 0), 'two empty classes');
refuses(() => rankedQueue(['A'], [1], [0.5], 0), 'a queue size of zero');
refuses(() => checkRange(5, limits.score, 'a score'), 'a range guard that should reject');
refuses(() => checkFinite(Number.POSITIVE_INFINITY, 'a value'), 'a non-finite value');

/* ================================= typeability and control ranges */

const typeable = [
  [fixtures.queue.map(row => row.score), 0.01, 'the default queue scores'],
  [fixtures.tiedQueue.map(row => row.score), 0.01, 'the practice queue scores'],
  [[fixtures.casePosterior, fixtures.contrastPosterior, fixtures.practiceCosts.posterior], 0.01, 'the posteriors'],
  [[fixtures.costs.costFP, fixtures.costs.costFN, fixtures.practiceCosts.costFP, fixtures.practiceCosts.costFN], 0.1, 'the costs'],
  [[fixtures.weighted.positiveWeight, fixtures.weighted.negativeWeight, fixtures.weightedDoubled.positiveWeight,
    fixtures.weightedDoubled.negativeWeight, fixtures.weightedPractice.positiveWeight], 0.1, 'the class weights'],
  [fixtures.cloud.flatMap(point => [point.x, point.y]), 0.1, 'the cloud coordinates'],
  [[fixtures.cloudSetup.fraction, 0.25, 0, 1], 0.05, 'the interpolation fractions'],
];
nonEmpty(typeable, 7, 'typeability groups');
typeable.forEach(([values, stepSize, label]) => {
  nonEmpty(values, undefined, `${label}: values`);
  values.forEach(value => assert(onStep(value, stepSize), `${label}: ${value} lands on the ${stepSize} step`));
  record('typeable values');
});
assert(limits.score.minimum === 0 && limits.score.maximum === 1, 'scores are bounded to [0, 1]');
assert(limits.cost.maximum === 50, 'the declared cost range reaches 50');
assert(limits.coordinate.minimum === -5 && limits.coordinate.maximum === 5, 'the declared coordinate range');
assert(limits.scaleDivisor.minimum === 0.1 && limits.scaleDivisor.maximum === 10, 'the declared scale range');
assert(fixtures.cloudContrastScaleY <= limits.scaleDivisor.maximum, 'the contrast divisor is reachable');
assert(fixtures.cloud.every(point =>
  point.x >= limits.coordinate.minimum && point.x <= limits.coordinate.maximum
  && point.y >= limits.coordinate.minimum && point.y <= limits.coordinate.maximum),
  'every default point is inside the enterable grid');
assert(fixtures.cloudNullMajority.y <= limits.coordinate.maximum, 'and so is the moved majority point');
assert(fixtures.cloud.length <= limits.maximumPoints, 'the default cloud is inside its own bound');
assert(fixtures.tiedQueue.length <= limits.maximumRecords, 'and the practice queue inside its bound');
record('control ranges');

/* ===================================== the displayed programs agree */

const programFiles = Object.values(examples).map(entry => entry.file);
nonEmpty(programFiles, 3, 'displayed programs');
assert.deepEqual(programFiles, ['imbalance_models.py', 'yeast_imbalance.py', 'yeast_smote_cv.py'],
  'the three programs are the ones the manuscript names');
// The manuscript is stored with CRLF endings; the published programs carry LF,
// so the comparison normalises rather than pretending the bytes match.
const manuscript = fs.readFileSync(`${packetDirectory}/lesson.md`, 'utf8').split('\r\n').join('\n');
Object.values(examples).forEach(entry => {
  assert(manuscript.includes(entry.code), `${entry.file}: the displayed code appears verbatim in the manuscript`);
  assert(entry.expected.trim().length > 0, `${entry.file}: it has recorded output`);
  record('program verbatim');
});
const studyLines = examples.study.expected.split('\n');
nonEmpty(studyLines, 20, 'study program output lines');
methods.forEach(method => {
  const header = studyLines.find(line => line.startsWith(`${method.name} training rows`));
  assert(header, `${method.name}: prints a header line`);
  assert(header.includes(String(method.fitRows)), `${method.name}: prints its ${method.fitRows} training rows`);
  assert(header.includes(String(method.chosenThreshold)), `${method.name}: prints its exact threshold`);
  const tunedLine = studyLines[studyLines.indexOf(header) + 2];
  assert(tunedLine.includes(`(${method.inspectionTuned.tp}, ${method.inspectionTuned.fp}, `
    + `${method.inspectionTuned.fn}, ${method.inspectionTuned.tn}, ${method.inspectionTuned.cost})`),
    `${method.name}: the program's tuned counts match the published module`);
  record('program row agrees with the module');
});
const foldLine = examples.pipeline.expected.split('\n')[0];
const foldValues = (foldLine.match(/[\d.]+/g) ?? []).map(Number);
nonEmpty(foldValues, 3, 'optional-program fold scores');
foldValues.forEach(value => {
  assert(value > 0 && value < 1, 'every fold average precision is a proportion');
  assert(!methods.some(method => Math.abs(method.averagePrecision - value) < 1e-9),
    'and no fold score coincides with a five-model inspection AP');
  record('optional-program fold score');
});
record('displayed programs');

/* ================================================= data module shape */

assert(provenance.sha256 === '7cf61776fc04f527f93bf57a327b863893a1225d82df02d457e8950173218258',
  'the module records the dataset hash');
assert(provenance.bytes === 94976, 'and its byte count');
assert(provenance.sourceRows === 1484 && provenance.studyRows === 1462 && provenance.duplicateIdentifiers === 22,
  '1,484 source rows, 1,462 distinct proteins, 22 repeats');
close(Object.values(provenance.classCounts).reduce((total, value) => total + value, 0), 1484,
  'the ten original location counts total the source rows', 1e-12);
close(provenance.classCounts.ME2, 51, 'and ME2 holds 51 of them', 1e-12);
assert(provenance.columns.length === 6 && provenance.names.length === 6 && provenance.meanings.length === 6,
  'six score columns, each named and explained');
assert.deepEqual(provenance.columns, [0, 1, 2, 3, 6, 7], 'at the declared source indices');
assert.deepEqual(provenance.excluded, ['erl', 'pox'], 'with the two excluded fields named');
nonEmpty(provenance.limits, 4, 'recorded dataset limitations');
close(study.fittingPositives + study.fittingNegatives, 600, 'the fitting role holds 600 rows', 1e-12);
close(study.addedMinorityRows, 558, '558 minority rows are added to balance it', 1e-12);
close(study.balancedRows, 1158, 'giving 1,158 training rows', 1e-12);
close(study.undersampledRows, 42, 'while undersampling leaves 42', 1e-12);
close(study.costFalsePositive, 1, 'the declared false-alarm cost is 1', 1e-12);
close(study.costFalseNegative, 12, 'and the missed-positive cost 12', 1e-12);
nonEmpty(syntheticSample.standardised, 10, 'published synthetic lineage');
syntheticSample.standardised.forEach((row, index) => {
  assert(row.length === 6, 'each synthetic vector has six coordinates');
  assert(syntheticSample.sourceScale[index].length === 6, 'in both coordinate systems');
  const fractionValue = syntheticSample.fractions[index];
  assert(fractionValue >= 0 && fractionValue <= 1, 'with a fraction in [0, 1]');
  record('synthetic lineage row');
});
close(syntheticSample.count, 558, 'the lineage names all 558 generated vectors', 1e-12);
assert(syntheticSample.anchorSourceIds.length === 10 && syntheticSample.neighbourSourceIds.length === 10,
  'and identifies each published vector with the two fitting proteins behind it');
record('data module shape');

// Sequential review: bounded legal endpoints omitted from the original grid.
for (const [p, positive, negative] of [[.01, .1, 20], [.99, 20, .1]]) {
  const expected = positive * p / (positive * p + negative * (1 - p));
  const actual = weightedOptimum(p, positive, negative);
  close(actual.optimum, expected, 'extreme weighted optimum', 1e-13);
  close(inverseWeightedOptimum(actual.optimum, positive, negative), p,
    'extreme optimum round trip without UI-range clipping', 1e-12);
  record('sequential weighted extreme round trip');
}
const inverseExtreme = inverseWeightedOptimum(.01, 20, .1);
close(inverseExtreme, .001 / 19.801, 'inverse probability below the input slider minimum', 1e-13);
close(weightedOptimum(inverseExtreme, 20, .1).optimum, .01,
  'inverse-mode loss minimum agrees with the entered score', 1e-13);
record('sequential inverse curve context');
const dustTie = smoteConstruction({
  points: [{ id: 'A', cls: 'minority', x: .1, y: 0 },
    { id: 'B', cls: 'minority', x: -.1, y: 0 },
    { id: 'C', cls: 'minority', x: .3, y: 0 }],
  anchorId: 'A', k: 1, neighbourRank: 0, fraction: .5, scaleX: 1, scaleY: 1,
});
assert.equal(dustTie.neighbour.id, 'B', 'equal decimal distances use the declared stable ID tie rule');
assert.equal(dustTie.generated.x, 0, 'the selected tied endpoint determines the actual midpoint');
assert.equal(dustTie.boundaryTie, true, 'the same tie is disclosed at the k boundary');
record('sequential SMOTE decimal tie');

/* --------------------------------------------------------------- record */

const sources = [
  'src/learn/data/imbalance-models.js',
  'src/learn/data/imbalance-data.js',
  'src/learn/data/imbalance-examples.js',
  'src/learn/components/lesson-labs/ImbalanceShared.jsx',
  'src/learn/components/lesson-labs/ImbalanceLabs.jsx',
  'src/learn/components/lesson-labs/ImbalanceFigures.jsx',
  'src/learn/components/lesson-labs/imbalance-labs.css',
  'src/learn/data/topics/imbalanced-learning-smote-cost-sensitive-learning.jsx',
  'src/learn/data/curriculum/blueprints/imbalanced-learning-smote-cost-sensitive-learning.js',
  'public/learn-assets/imbalanced-learning/yeast.data',
];
const hash = file => crypto.createHash('sha256').update(fs.readFileSync(file)).digest('hex');
const total = Object.values(counts).reduce((sumValue, value) => sumValue + value, 0);
const evidence = {
  checkedAt: new Date().toISOString(),
  sourceHashes: Object.fromEntries(sources.filter(fs.existsSync).map(file => [file, hash(file)])),
  packetCalculatedInputsSha256: crypto.createHash('sha256')
    .update(fs.readFileSync(`${packetDirectory}/calculated-inputs.json`)).digest('hex'),
  verifierHash: hash('scripts/verify-imbalance-models.mjs'),
  counts,
  totalGroupedChecks: total,
  sweptThresholdCandidates: sweptCandidates,
  comparedOrderedThresholdPairs: monotoneChecks,
  weightedOptimumGridPoints: optimumGrid.length,
  scope: 'Browser imbalanced-learning models against the content packet\'s calculated-inputs.json and every number '
    + 'the manuscript states: the 1,000-case confusion table and its separately normalised class bands; the '
    + 'three-record precision counterexample over every operating point including the undefined no-alert state; all '
    + '24 display permutations of the tied practice queue; grouped noninterpolated average precision including the '
    + 'constant-score and no-positive cases; a recall-monotonicity sweep over every ordered pair of candidates on '
    + 'five queues; both population flows with their fractional expectations and the prevalence identity; the two '
    + 'action risks with a 16-setting common-factor null, the equal-cost cutoff and all three degenerate cost '
    + 'corners; the single weighted gradient step with its drawn before and after curves and a weight-scaling null; '
    + 'the weighted optimum over a 168-point grid, each checked stationary analytically and by central difference, '
    + 'with a positive second derivative and a forward/inverse round trip; the SMOTE construction with a 24-position '
    + 'majority-move null, the metric contrast that switches the nearest neighbour, both segment endpoints, a '
    + 'zero-length segment and a disclosed distance tie; every recorded Yeast confusion count, threshold, average '
    + 'precision, ROC-AUC, Brier score and top-ten identity recomputed from the published scores, with the tie rule '
    + 'checked against every swept candidate and a cost-scaling null on the real tuning records; the focal loss '
    + 'masses with a central-difference check of the product-rule derivative; the prior-shift corrections; the '
    + 'balancing expansion over a 16-pair sweep; refused input on every entry point; the typeability of every value '
    + 'the prose asks a learner to enter; and agreement between the published data module and the three executed '
    + 'displayed programs.',
  limitations: [
    'The Yeast results are precomputed native fits; the browser reproduces their recorded outcomes, not the fitting.',
    'Displayed program output is executed separately by scripts/verify-imbalance-examples.py.',
    'The data module is regenerated and matched to the packet separately by scripts/verify-imbalance-data.py.',
    'Rendering, interaction, visual layout and independent review are separate steps.',
  ],
  passed: true,
};
fs.mkdirSync('docs/teaching/evidence', { recursive: true });
fs.writeFileSync('docs/teaching/evidence/imbalance-models.json', JSON.stringify(evidence, null, 2) + '\n');
console.log(`PASS: ${total} grouped imbalanced-learning model checks across ${Object.keys(counts).length} groups, `
  + `including ${sweptCandidates.toLocaleString('en-US')} swept threshold candidates and `
  + `${optimumGrid.length} weighted-optimum grid points.`);
