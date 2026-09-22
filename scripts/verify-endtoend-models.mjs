// Bounded independent checks of the end-to-end study's browser models against
// the content packet's recorded calculations, the manuscript's stated values,
// and a second derivation for every claim a figure draws.
//
// Every assertion here is written so that it can fail: nothing is compared with
// itself, no loop is allowed to inspect an empty subject set, and every group
// asserts how many subjects it actually saw.
//
// Four rules shaped this file.
//
//   1. A SECOND ROUTE MEANS A DIFFERENT DERIVATION. Accuracy and balanced
//      accuracy are recomputed from the confusion matrix -- a diagonal over a
//      total, and a diagonal over its own row sum -- while the module computes
//      them by walking the label vectors. Log loss is recomputed by summing in
//      the opposite order with a compensated sum. The Wilson interval is
//      recovered by solving the quadratic the interval is defined by, rather
//      than by evaluating the same closed form. The paired repair and regression
//      sets are recomputed from set differences over specimen identifiers rather
//      than by zipping two lists.
//
//   2. A RULE DRAWN MUST EQUAL THE RULE APPLIED ACROSS THE WHOLE ENTERABLE GRID.
//      The slice rule is swept over every one of the 1,401 cutoffs the control
//      admits, on both sides, for every ordered pair of candidates -- 44,832
//      slice comparisons -- and at each one the scatter's own marks, the cutoff
//      line's own position and the graded difference are required to agree. The
//      acceptance ledger is swept over every threshold and every integer cost
//      pair its controls admit.
//
//   3. EVERY GRADED COMPARISON IS EXERCISED AT ITS DEGENERATE INPUTS. An empty
//      slice, a one-row slice, a slice containing every row, an exact tie, the
//      same candidate on both sides, an empty eligible set, a rule that accepts
//      nothing and a cost pair that makes every rule equal all appear below, and
//      the ungradable cases are asserted to be ungradable rather than to fall
//      into a category.
//
//   4. EVERY SCORE CARRIES ITS ROLE. This is the defect class this lesson is
//      most exposed to, so the role is checked as a property of every exported
//      score and the held-out block is checked to be absent from everything the
//      page shows before the gate.
//
// Run: node scripts/verify-endtoend-models.mjs
import assert from 'node:assert/strict';
import fs from 'node:fs';
import crypto from 'node:crypto';
import {
  CUTOFF_MAX_HUNDREDTHS, CUTOFF_MIN_HUNDREDTHS, DEFERRAL_CASES, MINIMUM_INSET, ROLE_LABELS,
  SCORE_MAX_HUNDREDTHS, SCORE_MIN_HUNDREDTHS, SCORE_ROLES, SELECTION_METRICS, THRESHOLD_MAX_HUNDREDTHS,
  THRESHOLD_MIN_HUNDREDTHS, accuracyOf, acceptanceComparison, acceptanceLedger, acceptanceRailGeometry,
  asInput, asSelectionCriterion, balancedAccuracyOf, bestByRule, candidateBarGeometry, candidateByKey,
  candidateOrder,
  classRecallTable, confusionOf, cutoffValue, eligibleByDeclaration, everySelectionSetting, fixed,
  flavanoidStripGeometry, inSlice, informationLaneGeometry, linearScale, logLossOf, natsFor, pairedChange,
  pairedStripGeometry, recallComparisonGeometry, scatterGeometry, scoreRecord, selectionMetricByKey,
  selectionOutcome, sliceComparison, sliceOf, validationActual, validationRows, wilsonInterval,
} from '../src/learn/data/endtoend-models.js';
import { endToEndData } from '../src/learn/data/endtoend-data.js';
import { endToEndExamples } from '../src/learn/data/endtoend-examples.js';

const packetDirectory = 'docs/teaching/drafts/end-to-end-supervised-learning-error-analysis';
const packet = JSON.parse(fs.readFileSync(`${packetDirectory}/calculated-inputs.json`, 'utf8'));
const manuscript = fs.readFileSync(`${packetDirectory}/lesson.md`, 'utf8');
const evidencePath = 'docs/teaching/evidence/endtoend-models.json';

const counts = {};
const record = name => { counts[name] = (counts[name] ?? 0) + 1; };
const close = (actual, expected, label, tolerance = 1e-12) => {
  assert(typeof actual === 'number' && Number.isFinite(actual), `${label}: ${actual} is not a finite number`);
  assert(Math.abs(actual - expected) <= tolerance * Math.max(1, Math.abs(expected)),
    `${label}: ${actual} versus ${expected}`);
};
/** A loop must have something to look at, or it asserts nothing. */
const nonEmpty = (collection, expected, label) => {
  assert(collection.length > 0, `${label}: the subject set is empty, so nothing was checked`);
  if (expected !== undefined) {
    assert.equal(collection.length, expected, `${label}: saw ${collection.length}, expected ${expected}`);
  }
};
/** A call that must be refused. A silent default here would be a wrong answer. */
const refuses = (call, label) => {
  assert.throws(call, RangeError, `${label}: should have been refused`);
  record('refused input');
};

/* ================================================== a provisional record first
 *
 * Written before the first assertion, with `passed: false`, and stamped only
 * after the last one. A verifier that writes its record at the end leaves the
 * PREVIOUS run's passing record on disk when this run dies part way through,
 * and that record then documents a state nobody verified.
 */
const keepEvidence = !process.argv.includes('--no-evidence');
const startedAt = new Date().toISOString();
if (keepEvidence) {
  fs.mkdirSync('docs/teaching/evidence', { recursive: true });
  fs.writeFileSync(evidencePath, `${JSON.stringify({
    startedAt,
    verifier: 'scripts/verify-endtoend-models.mjs',
    status: 'a run is in progress, or one died before its last assertion',
    passed: false,
  }, null, 2)}\n`);
}

/* ======================================= second routes, written from scratch */

/** Accuracy from the confusion matrix: the diagonal over the total. The module
 *  walks the label vectors instead, so this is a different derivation of the
 *  same number rather than a second call to the same one. */
function accuracyFromConfusion(confusion) {
  const total = confusion.reduce((sum, row) => sum + row.reduce((inner, value) => inner + value, 0), 0);
  const diagonal = confusion.reduce((sum, row, index) => sum + row[index], 0);
  return diagonal / total;
}

/** Balanced accuracy from the confusion matrix: each diagonal entry over its
 *  own row sum, averaged over the rows that have one. */
function balancedFromConfusion(confusion) {
  const recalls = confusion
    .map((row, index) => [row[index], row.reduce((sum, value) => sum + value, 0)])
    .filter(([, support]) => support > 0)
    .map(([right, support]) => right / support);
  return recalls.reduce((sum, value) => sum + value, 0) / recalls.length;
}

/** Log loss summed in reverse order with a Kahan compensation, so the result is
 *  not the module's accumulation in a different costume. */
function logLossByCompensatedSum(actual, probabilities) {
  let total = 0;
  let compensation = 0;
  for (let index = actual.length - 1; index >= 0; index -= 1) {
    const term = -Math.log(probabilities[index][actual[index]]) - compensation;
    const next = total + term;
    compensation = (next - total) - term;
    total = next;
  }
  return total / actual.length;
}

/** The Wilson centre and radius recovered from the quadratic they solve.
 *
 *  The interval's endpoints are the roots of (p̂ − q)² = z²q(1 − q)/n in q. This
 *  finds them with the quadratic formula rather than by evaluating the closed
 *  form the module evaluates.
 */
function wilsonByQuadratic(successes, trials, z) {
  const proportion = successes / trials;
  const a = 1 + (z * z) / trials;
  const b = -(2 * proportion + (z * z) / trials);
  const c = proportion * proportion;
  const discriminant = b * b - 4 * a * c;
  const lower = (-b - Math.sqrt(discriminant)) / (2 * a);
  const upper = (-b + Math.sqrt(discriminant)) / (2 * a);
  return { lower, upper };
}

/** The repaired and newly wrong sets, by set difference over identifiers. */
function changedBySets(reference, candidate) {
  const rightUnder = key => new Set(validationRows
    .filter(row => row.prediction[key] === row.actual).map(row => row.id));
  const before = rightUnder(reference);
  const after = rightUnder(candidate);
  return {
    repaired: validationRows.map(row => row.id).filter(id => !before.has(id) && after.has(id)),
    broken: validationRows.map(row => row.id).filter(id => before.has(id) && !after.has(id)),
  };
}

/** The winner under a stated rule, found by an explicit scan with the declared
 *  tie rule written out rather than by the module's reducer. */
function winnerByScan(eligible, metricKey) {
  const metric = selectionMetricByKey[metricKey];
  const ordered = candidateOrder.filter(key => eligible.includes(key));
  if (!ordered.length) return null;
  const scores = ordered.map(key => candidateByKey[key][metricKey].value);
  const best = metric.direction === 'higher' ? Math.max(...scores) : Math.min(...scores);
  return ordered[scores.indexOf(best)];
}

/* ====================================================== 1 · roles and scores */

nonEmpty(SCORE_ROLES, 4, 'the four score roles');
assert.deepEqual(SCORE_ROLES, ['training', 'validation', 'selection', 'held-out'],
  'the four roles are training, validation, selection and held-out');
for (const role of SCORE_ROLES) {
  assert.equal(typeof ROLE_LABELS[role], 'string', `role ${role} has a label`);
  assert.ok(ROLE_LABELS[role].length > 0, `role ${role}'s label is not empty`);
  record('role labels');
}
refuses(() => scoreRecord('accuracy', 0.5, 'test'), 'a role outside the four');
refuses(() => scoreRecord('accuracy', 0.5, 'development'), 'a plausible-sounding role outside the four');
refuses(() => scoreRecord('accuracy', Number.NaN, 'validation'), 'a score that is not a number');
refuses(() => scoreRecord('accuracy', Infinity, 'validation'), 'an infinite score');
refuses(() => asSelectionCriterion(candidateByKey.linear_three.trainingBalancedAccuracy),
  'reading a training score as a selection criterion');
refuses(() => asSelectionCriterion(endToEndData.heldOut.balancedAccuracy),
  'reading a held-out score as a selection criterion');
refuses(() => asSelectionCriterion(null), 'reading nothing as a selection criterion');

const promoted = asSelectionCriterion(candidateByKey.linear_three.validationBalancedAccuracy);
assert.equal(promoted.role, 'selection', 'a validation score read to choose becomes a selection criterion');
assert.equal(promoted.value, candidateByKey.linear_three.validationBalancedAccuracy.value,
  'and its value does not change when its role does');
assert.equal(promoted.from, 'validation', 'and it remembers where it came from');
record('the selection re-badging');

/* Every score in the data module carries a role, and it is the right one. */
let scoreFields = 0;
for (const candidate of endToEndData.candidates) {
  for (const [field, expectedRole] of [
    ['trainingBalancedAccuracy', 'training'],
    ['validationBalancedAccuracy', 'validation'],
    ['validationAccuracy', 'validation'],
    ['validationLogLoss', 'validation'],
  ]) {
    const score = candidate[field];
    assert.ok(score && typeof score === 'object', `${candidate.key}.${field} is a score record`);
    assert.equal(score.role, expectedRole, `${candidate.key}.${field} carries the role ${expectedRole}`);
    assert.ok(SCORE_ROLES.includes(score.role), `${candidate.key}.${field} has a known role`);
    assert.equal(typeof score.metric, 'string', `${candidate.key}.${field} names its metric`);
    assert.ok(Number.isFinite(score.value), `${candidate.key}.${field} is a finite number`);
    scoreFields += 1;
    record('score records carry their role');
  }
}
for (const field of ['accuracy', 'balancedAccuracy', 'logLoss']) {
  assert.equal(endToEndData.heldOut[field].role, 'held-out',
    `the held-out ${field} is labelled held-out and nothing else`);
  scoreFields += 1;
  record('the held-out block is labelled held-out');
}
assert.equal(scoreFields, 19, `${scoreFields} score records were inspected, expected 19`);
for (const role of SCORE_ROLES) {
  assert.equal(typeof endToEndData.roles[role], 'string', `the data module explains the role ${role}`);
  record('the roles are explained in the data');
}

/* ============================================ 2 · metrics by a second route */

for (const candidate of endToEndData.candidates) {
  const key = candidate.key;
  const block = packet.candidates[key];
  const confusion = confusionOf(validationActual, block.validationPrediction);
  assert.deepEqual(confusion, block.validationConfusion,
    `${key}: the recomputed validation confusion matrix differs from the packet`);
  assert.deepEqual(confusion, candidate.validationConfusion,
    `${key}: the data module's confusion matrix differs from the recomputed one`);
  close(accuracyOf(validationActual, block.validationPrediction), accuracyFromConfusion(confusion),
    `${key}: accuracy by label walk against accuracy by confusion diagonal`);
  close(accuracyOf(validationActual, block.validationPrediction), block.validationAccuracy,
    `${key}: accuracy against the packet`);
  close(balancedAccuracyOf(validationActual, block.validationPrediction), balancedFromConfusion(confusion),
    `${key}: balanced accuracy by label walk against balanced accuracy by row sums`);
  close(balancedAccuracyOf(validationActual, block.validationPrediction), block.validationBalancedAccuracy,
    `${key}: balanced accuracy against the packet`);
  close(logLossOf(validationActual, block.validationProbability),
    logLossByCompensatedSum(validationActual, block.validationProbability),
    `${key}: log loss forwards against log loss compensated backwards`, 1e-14);
  close(logLossOf(validationActual, block.validationProbability), block.validationLogLoss,
    `${key}: log loss against the packet`, 1e-12);
  close(candidate.validationBalancedAccuracy.value, block.validationBalancedAccuracy,
    `${key}: the data module's validation balanced accuracy against the packet`);
  close(candidate.trainingBalancedAccuracy.value, block.trainBalancedAccuracy,
    `${key}: the data module's training balanced accuracy against the packet`);
  /* The training score is not the validation score. Asserting the ordering
     would be a claim about learning; asserting they DIFFER is a claim that the
     two roles are not accidentally the same number. */
  if (key !== 'majority') {
    assert.notEqual(candidate.trainingBalancedAccuracy.value, candidate.validationBalancedAccuracy.value,
      `${key}: the training and validation scores are distinct numbers`);
  }
  const correct = validationActual.filter((value, index) => value === block.validationPrediction[index]).length;
  assert.equal(candidate.validationCorrect, correct, `${key}: the recorded correct count`);
  assert.equal(correct, confusion.reduce((sum, row, index) => sum + row[index], 0),
    `${key}: the correct count is the confusion diagonal`);
  const table = classRecallTable(confusion);
  assert.equal(table.reduce((sum, row) => sum + row.support, 0), 36,
    `${key}: the three class supports add to 36`);
  for (const row of table) {
    close(row.recall, row.correct / row.support, `${key}: recall for cultivar ${row.actual}`);
    assert.equal(row.errors, row.support - row.correct, `${key}: errors for cultivar ${row.actual}`);
    record('per-class recall with its denominator');
  }
  close(balancedFromConfusion(confusion),
    table.reduce((sum, row) => sum + row.recall, 0) / table.length,
    `${key}: balanced accuracy is the mean of the recall table`);
  record('metrics by two routes against the packet');
}
nonEmpty(endToEndData.candidates, 4, 'the four candidates');

/* Degenerate metric inputs are refused rather than answered. */
refuses(() => accuracyOf([], []), 'accuracy over an empty list');
refuses(() => balancedAccuracyOf([], []), 'balanced accuracy over an empty list');
refuses(() => accuracyOf([0, 1], [0]), 'accuracy over mismatched lengths');
refuses(() => logLossOf([0], [[0, 1, 0]]), 'log loss where the actual class has probability zero');
refuses(() => natsFor(0), 'the cost of a probability of zero');
refuses(() => natsFor(1.5), 'the cost of a probability above one');
close(natsFor(1), 0, 'a probability of one costs nothing');
close(natsFor(0.8), 0.22314355131420976, 'the nats the lesson quotes for 0.8', 1e-15);
close(natsFor(0.2), 1.6094379124341003, 'the nats the lesson quotes for 0.2', 1e-15);
record('the log-loss illustration');

/* A one-class edge: balanced accuracy averages only the classes with support. */
close(balancedAccuracyOf([1, 1, 1], [1, 1, 0]), 2 / 3,
  'balanced accuracy over a single supported class is that class\'s recall');
close(balancedAccuracyOf([0, 1], [0, 1]), 1, 'balanced accuracy at a perfect two-class fit');
close(balancedAccuracyOf([0, 1], [1, 0]), 0, 'balanced accuracy at a completely wrong two-class fit');
record('balanced accuracy at its edges');

/* ============================================ 3 · the selection rule, swept */

const settings = everySelectionSetting();
nonEmpty(settings, 2 ** candidateOrder.length * SELECTION_METRICS.length, 'every selection setting');
let selectionCases = 0;
let heldOutOffered = 0;
let heldOutRefused = 0;
let emptySelections = 0;
for (const setting of settings) {
  const outcome = selectionOutcome(setting);
  const expected = winnerByScan(setting.eligible, setting.metricKey);
  assert.equal(outcome.winner, expected,
    `selection over [${setting.eligible}] by ${setting.metricKey}: the module and an explicit scan disagree`);
  if (expected === null) {
    emptySelections += 1;
    assert.equal(outcome.heldOutAvailable, false, 'an empty eligible set offers no held-out estimate');
    assert.equal(outcome.ranking.length, 0, 'an empty eligible set has an empty ranking');
    assert.ok(outcome.reason.length > 0, 'an empty eligible set explains itself');
  } else {
    assert.ok(setting.eligible.includes(outcome.winner), 'the winner is one of the eligible candidates');
    const metric = selectionMetricByKey[setting.metricKey];
    const best = candidateByKey[outcome.winner][setting.metricKey].value;
    for (const key of setting.eligible) {
      const value = candidateByKey[key][setting.metricKey].value;
      assert.ok(metric.direction === 'higher' ? value <= best : value >= best,
        `selection over [${setting.eligible}] by ${setting.metricKey}: ${key} beats the declared winner`);
    }
    assert.equal(outcome.ranking[0], outcome.winner, 'the ranking leads with the winner');
    assert.equal(outcome.ranking.length, setting.eligible.length, 'the ranking holds every eligible candidate');
    assert.deepEqual([...outcome.ranking].sort(), [...setting.eligible].sort(),
      'the ranking is a permutation of the eligible set');
    for (const tied of outcome.tiedWith) {
      assert.equal(candidateByKey[tied][setting.metricKey].value, best,
        'a candidate reported as tied has exactly the winner\'s score');
      assert.ok(candidateOrder.indexOf(tied) > candidateOrder.indexOf(outcome.winner),
        'the tie was broken by the declared candidate order, so any tied candidate comes later in it');
    }
  }
  const isDeclaredSet = [...setting.eligible].sort().join(',') === [...eligibleByDeclaration].sort().join(',');
  const isDeclaredMetric = selectionMetricByKey[setting.metricKey].declared;
  assert.equal(outcome.departsFromProtocol, !(isDeclaredSet && isDeclaredMetric),
    `selection over [${setting.eligible}] by ${setting.metricKey}: the protocol flag is wrong`);
  /* The refusal is the teaching claim: a held-out number exists for exactly one
     selection, and no other choice may borrow it. */
  const shouldOffer = isDeclaredSet && isDeclaredMetric && outcome.winner === endToEndData.heldOut.selected;
  assert.equal(outcome.heldOutAvailable, shouldOffer,
    `selection over [${setting.eligible}] by ${setting.metricKey}: held-out availability is wrong`);
  if (outcome.heldOutAvailable) {
    heldOutOffered += 1;
    assert.equal(outcome.heldOutRefusal, null, 'an available report carries no refusal message');
  } else {
    heldOutRefused += 1;
    assert.ok(typeof outcome.heldOutRefusal === 'string' && outcome.heldOutRefusal.length > 20,
      'a refused report says why');
  }
  selectionCases += 1;
}
assert.equal(selectionCases, 48, `${selectionCases} selection settings were swept, expected 48`);
assert.equal(emptySelections, SELECTION_METRICS.length,
  'the empty eligible set occurs once per metric and is reported as having no winner');
assert.equal(heldOutOffered, 1,
  `${heldOutOffered} settings offer the held-out report; exactly one may, the declared comparison`);
assert.equal(heldOutRefused, 47, 'every other setting refuses it');
record('the selection rule over every setting its controls admit');

/* The declared comparison, named explicitly, against the packet. */
const declared = selectionOutcome({
  eligible: eligibleByDeclaration, metricKey: 'validationBalancedAccuracy',
});
assert.equal(declared.winner, packet.selected, 'the declared comparison selects what the packet recorded');
assert.equal(declared.winner, 'linear_three', 'and that is the three-feature linear model');
assert.equal(declared.departsFromProtocol, false, 'the declared comparison does not depart from itself');
assert.equal(declared.heldOutAvailable, true, 'and it is the one comparison the held-out report belongs to');
assert.deepEqual(declared.ranking, ['linear_three', 'forest_two', 'linear_two'],
  'the declared ranking, best first');
assert.deepEqual(declared.tiedWith, [], 'no two candidates tie under the declared metric');
record('the declared comparison');

/* No two candidates tie on this study's data, so the tie rule inside the
   reducer would be a rule nothing could exercise. The reducer is therefore
   exported as `bestByRule` and swept here over score vectors that DO contain
   ties: every assignment of three distinct values to four positions, in both
   directions. The rule is "the first entry achieving the extremum", and a
   reducer that took the LAST one -- which is what relaxing its comparison to
   `>=` does -- disagrees on every vector whose extremum repeats. */
let tieCases = 0;
let tiedExtremumCases = 0;
const tieValues = [0, 1, 2];
for (let code = 0; code < tieValues.length ** 4; code += 1) {
  const values = [0, 1, 2, 3].map(position =>
    tieValues[Math.floor(code / tieValues.length ** position) % tieValues.length]);
  const entries = candidateOrder.map((key, index) => [key, values[index]]);
  for (const direction of ['higher', 'lower']) {
    const extremum = direction === 'higher' ? Math.max(...values) : Math.min(...values);
    const firstAchieving = candidateOrder[values.indexOf(extremum)];
    assert.equal(bestByRule(entries, direction), firstAchieving,
      `scores [${values}] going ${direction}: the rule must pick the FIRST entry achieving the extremum, `
      + 'which is what "ties use the declared candidate order" means');
    if (values.filter(value => value === extremum).length > 1) tiedExtremumCases += 1;
    tieCases += 1;
  }
}
assert.equal(tieCases, 2 * tieValues.length ** 4, `${tieCases} tie-rule cases ran`);
/* 90 of the 162 swept cases have a repeated extremum. The floor sits just below
   that: a sweep whose ties disappeared would be a sweep that no longer
   exercises the rule it exists for, and would read as a pass. */
assert.ok(tiedExtremumCases >= 80,
  `only ${tiedExtremumCases} of the swept vectors had a repeated extremum; the tie rule was barely exercised`);
assert.equal(bestByRule([], 'higher'), null, 'no entries means no winner');
assert.equal(bestByRule([['only', 5]], 'lower'), 'only', 'a single entry wins whichever way the rule points');
refuses(() => bestByRule([['a', 1]], 'sideways'), 'a direction that is not a direction');
record('the tie rule over every score vector with a repeated extremum');
refuses(() => selectionOutcome({ eligible: ['linear_two'], metricKey: 'testAccuracy' }),
  'selecting by a metric this study does not have');
refuses(() => selectionOutcome({ eligible: ['nothing'], metricKey: 'validationBalancedAccuracy' }),
  'selecting among a candidate that does not exist');
refuses(() => selectionOutcome({ eligible: 'linear_two', metricKey: 'validationBalancedAccuracy' }),
  'an eligible set that is not a list');

/* ============================================= 4 · the slice rule, swept whole */

/* EVERY CONTAINMENT CHECK BELOW COMPARES AGAINST THIS LITERAL, never against a
 * geometry's own inset or padding.
 *
 * The previous version asserted `MINIMUM_INSET >= 12` and said it was "the 12
 * units every containment check assumes" -- and no containment check read it.
 * The three geometries carrying `inset: MINIMUM_INSET` had their containment
 * checked against their `padding`; the four with literal insets were checked
 * against those same insets, so `lane.x >= lanes.inset` compared `inset` with
 * itself and the rail's tiles were inside a scale built from the inset for any
 * inset whatsoever. Setting the paired strip's, the lane figure's or the rail's
 * inset to zero would have broken nothing.
 *
 * The guard written to prevent self-referential containment checks was itself
 * self-referential. A bound has to be something the position is not derived
 * from: a literal, or the SVG's declared width and height. */
const HARD_MARGIN = 12;
assert.ok(MINIMUM_INSET >= HARD_MARGIN,
  `the shared minimum inset is ${MINIMUM_INSET}, below the ${HARD_MARGIN}-unit literal every containment `
  + 'check below compares against');
/** Every drawn x must sit inside the frame by at least the literal margin. */
const insideFrame = (positions, width, label) => {
  nonEmpty(positions, undefined, `${label}: nothing was checked for containment`);
  for (const position of positions) {
    assert.ok(position >= HARD_MARGIN && position <= width - HARD_MARGIN,
      `${label}: a drawn position at ${position} is outside the frame's ${HARD_MARGIN}-unit margin in a `
      + `${width}-unit viewBox`);
  }
  record(`${label} inside a literal frame margin`);
};

const allPairs = candidateOrder.flatMap(reference => candidateOrder.map(candidate => [reference, candidate]));
nonEmpty(allPairs, 16, 'every ordered pair of candidates');
let sliceCases = 0;
let emptySlices = 0;
let singletonSlices = 0;
let fullSlices = 0;
let fewerSeen = 0;
let sameSeen = 0;
let moreSeen = 0;
let boundaryCases = 0;
const geometryPairs = [['linear_two', 'linear_three'], ['linear_two', 'forest_two'],
  ['linear_three', 'linear_two'], ['linear_two', 'linear_two']];

for (let hundredths = CUTOFF_MIN_HUNDREDTHS; hundredths <= CUTOFF_MAX_HUNDREDTHS; hundredths += 1) {
  const cutoff = cutoffValue(hundredths);
  const lower = sliceOf(hundredths, 'lower');
  const upper = sliceOf(hundredths, 'upper');
  /* The two sides partition the development rows. Not "usually": always, at
     every cutoff the control admits. */
  assert.equal(lower.n + upper.n, validationRows.length,
    `cutoff ${cutoff}: the two sides hold ${lower.n} + ${upper.n}, not all ${validationRows.length} rows`);
  assert.equal(new Set([...lower.ids, ...upper.ids]).size, validationRows.length,
    `cutoff ${cutoff}: the two sides share a specimen`);
  for (const row of validationRows) {
    assert.equal(inSlice(row, hundredths, 'lower'), row.colorIntensity < cutoff,
      `cutoff ${cutoff}, specimen ${row.id}: the lower rule disagrees with a bare comparison`);
    assert.equal(inSlice(row, hundredths, 'upper'), row.colorIntensity >= cutoff,
      `cutoff ${cutoff}, specimen ${row.id}: the upper rule disagrees with a bare comparison`);
    if (row.colorIntensity === cutoff) {
      assert.ok(inSlice(row, hundredths, 'upper') && !inSlice(row, hundredths, 'lower'),
        `cutoff ${cutoff}, specimen ${row.id}: a specimen exactly on the cutoff belongs to the upper side`);
      boundaryCases += 1;
    }
  }
  if (lower.n === 0) emptySlices += 1;
  if (lower.n === 1) singletonSlices += 1;
  if (lower.n === validationRows.length) fullSlices += 1;

  for (const side of ['lower', 'upper']) {
    for (const [reference, candidate] of allPairs) {
      const comparison = sliceComparison({ reference, candidate, cutoffHundredths: hundredths, side });
      sliceCases += 1;
      const slice = side === 'lower' ? lower : upper;
      assert.equal(comparison.slice.n, slice.n, `cutoff ${cutoff} ${side}: the comparison's slice size`);
      if (slice.n === 0) {
        assert.equal(comparison.outcome, null,
          `cutoff ${cutoff} ${side}: an empty slice must have no outcome, not a tie`);
        assert.equal(comparison.difference, null, 'and no difference');
        assert.equal(comparison.referenceErrors, null, 'and no error count');
        assert.deepEqual(comparison.repairedIds, [], 'and no changed specimen');
        continue;
      }
      /* The counts, recomputed here from the rows rather than read back. */
      const rows = slice.rows;
      const referenceErrors = rows.filter(row => row.prediction[reference] !== row.actual).length;
      const candidateErrors = rows.filter(row => row.prediction[candidate] !== row.actual).length;
      assert.equal(comparison.referenceErrors, referenceErrors,
        `cutoff ${cutoff} ${side} ${reference}: reference error count`);
      assert.equal(comparison.candidateErrors, candidateErrors,
        `cutoff ${cutoff} ${side} ${candidate}: candidate error count`);
      assert.equal(comparison.difference, candidateErrors - referenceErrors,
        `cutoff ${cutoff} ${side}: the difference is not the two counts subtracted`);
      const expectedOutcome = candidateErrors < referenceErrors ? 'fewer'
        : candidateErrors > referenceErrors ? 'more' : 'same';
      assert.equal(comparison.outcome, expectedOutcome,
        `cutoff ${cutoff} ${side} ${reference}->${candidate}: the graded outcome`);
      assert.ok(comparison.netIdentityHolds,
        `cutoff ${cutoff} ${side} ${reference}->${candidate}: the difference is not repairs minus regressions`);
      /* Written as a sum rather than a negation: `-0` and `0` are different
         values to a strict equality check, and a tie would fail on the sign of
         a zero rather than on anything about the counts. */
      assert.equal(comparison.repairedIds.length - comparison.brokenIds.length + comparison.difference, 0,
        `cutoff ${cutoff} ${side}: repairs minus regressions does not match the error difference`);
      assert.equal(new Set([...comparison.repairedIds, ...comparison.brokenIds]).size,
        comparison.repairedIds.length + comparison.brokenIds.length,
        'no specimen is both repaired and newly wrong');
      for (const id of [...comparison.repairedIds, ...comparison.brokenIds]) {
        assert.ok(slice.ids.includes(id), `a changed specimen ${id} is outside the slice it is reported in`);
      }
      /* The null case: the same candidate on both sides never changes anything. */
      if (reference === candidate) {
        assert.equal(comparison.outcome, 'same', 'a candidate compared with itself moves nothing');
        assert.equal(comparison.difference, 0, 'and its difference is exactly zero');
        assert.deepEqual(comparison.repairedIds, [], 'and repairs nothing');
        assert.deepEqual(comparison.brokenIds, [], 'and breaks nothing');
      }
      if (expectedOutcome === 'fewer') fewerSeen += 1;
      else if (expectedOutcome === 'more') moreSeen += 1;
      else sameSeen += 1;
    }
  }

  /* The picture must say what the grading says. Checked at every cutoff on a
     representative set of pairs -- the geometry does not depend on the pair
     beyond which marks are outlined, and running all sixteen at every cutoff
     would be sixteen copies of one claim. */
  for (const [reference, candidate] of geometryPairs) {
    for (const side of ['lower', 'upper']) {
      const geometry = scatterGeometry({ cutoffHundredths: hundredths, side, reference, candidate });
      assert.equal(geometry.points.length, validationRows.length, 'the scatter draws every development row');
      close(geometry.cutoffY, geometry.y(cutoff), `cutoff ${cutoff}: the drawn line is not at the cutoff`);
      for (const point of geometry.points) {
        const row = validationRows.find(item => item.id === point.id);
        assert.equal(point.inSlice, inSlice(row, hundredths, side),
          `cutoff ${cutoff} ${side}, specimen ${point.id}: the mark drawn disagrees with the rule applied`);
        /* And the mark is on the side of the line the rule puts it on. A mark
           drawn on the wrong side of its own cutoff is a picture that teaches
           the opposite of the grading, and no attribute check sees it. */
        if (side === 'lower') {
          assert.ok(point.inSlice ? point.cy >= geometry.cutoffY : point.cy <= geometry.cutoffY,
            `cutoff ${cutoff}: specimen ${point.id} is drawn on the wrong side of the line`);
        } else {
          assert.ok(point.inSlice ? point.cy <= geometry.cutoffY : point.cy >= geometry.cutoffY,
            `cutoff ${cutoff}: specimen ${point.id} is drawn on the wrong side of the line`);
        }
        if (row.colorIntensity !== cutoff) {
          assert.notEqual(point.cy, geometry.cutoffY,
            `cutoff ${cutoff}: specimen ${point.id} is drawn on the line without being on it`);
        }
        assert.equal(point.referenceWrong, row.prediction[reference] !== row.actual,
          `specimen ${point.id}: the reference outline disagrees with the prediction`);
        assert.equal(point.candidateWrong, row.prediction[candidate] !== row.actual,
          `specimen ${point.id}: the candidate outline disagrees with the prediction`);
      }
    }
  }
}
assert.equal(sliceCases, (CUTOFF_MAX_HUNDREDTHS - CUTOFF_MIN_HUNDREDTHS + 1) * 2 * allPairs.length,
  `${sliceCases} slice comparisons ran; the whole enterable grid was not covered`);
assert.ok(emptySlices > 0, 'the sweep met an empty slice');
assert.ok(singletonSlices > 0, 'the sweep met a one-specimen slice');
assert.ok(fullSlices > 0, 'the sweep met a slice holding every specimen');
assert.ok(fewerSeen > 0 && sameSeen > 0 && moreSeen > 0,
  `all three graded outcomes must occur in the sweep; saw ${fewerSeen}/${sameSeen}/${moreSeen}`);
/* Every development specimen's own boundary case is REACHABLE by the control,
   and was reached: each of the 36 sits exactly on one of the 1,401 cutoffs, so
   the sweep exercised the tie at every specimen rather than at a chosen one. */
assert.equal(boundaryCases, validationRows.length,
  `${boundaryCases} boundary hits occurred across the sweep; each of the ${validationRows.length} development `
  + 'specimens should sit exactly on one enterable cutoff, so every tie the control admits was exercised');
const onDefaultCutoff = validationRows.filter(row => row.colorIntensity === 4);
assert.equal(onDefaultCutoff.length, 1,
  `${onDefaultCutoff.length} specimens sit exactly on the default cutoff of 4; the lesson's slice tables `
  + 'depend on exactly one doing so, and on its being counted in the upper slice');
assert.ok(sliceOf(400, 'upper').ids.includes(onDefaultCutoff[0].id),
  'and that specimen is in the upper slice, not the lower one');
record('the slice rule and the picture over the whole enterable grid');

/* The four fixtures the visual specification recorded, by name. */
for (const [reference, candidate, hundredths, side, expectedN, expectedReference, expectedCandidate, expected]
  of [
    ['linear_two', 'linear_three', 400, 'lower', 17, 2, 3, 'more'],
    ['linear_two', 'linear_three', 400, 'upper', 19, 5, 1, 'fewer'],
    ['linear_two', 'forest_two', 400, 'upper', 19, 5, 5, 'same'],
    ['linear_two', 'linear_two', 0, 'upper', 36, 7, 7, 'same'],
  ]) {
  const comparison = sliceComparison({ reference, candidate, cutoffHundredths: hundredths, side });
  assert.equal(comparison.slice.n, expectedN, `fixture ${reference}->${candidate} ${side}: support`);
  assert.equal(comparison.referenceErrors, expectedReference, 'reference errors');
  assert.equal(comparison.candidateErrors, expectedCandidate, 'candidate errors');
  assert.equal(comparison.outcome, expected, 'graded outcome');
  record('the recorded specification fixtures');
}
/* The lower and upper sides point in OPPOSITE directions at the same cutoff.
   That contrast is the reason the lesson keeps the unfavourable slice. */
assert.equal(sliceComparison({
  reference: 'linear_two', candidate: 'linear_three', cutoffHundredths: 400, side: 'lower',
}).outcome, 'more', 'the low-colour slice goes backwards');
assert.equal(sliceComparison({
  reference: 'linear_two', candidate: 'linear_three', cutoffHundredths: 400, side: 'upper',
}).outcome, 'fewer', 'while the high-colour slice improves');
record('the deliberately opposed slices');

/* The packet's own slice records. */
for (const key of ['linear_two', 'forest_two', 'linear_three']) {
  for (const [label, hundredths, side] of [['color<4', 400, 'lower'], ['color>=4', 400, 'upper']]) {
    const slice = sliceOf(hundredths, side);
    const errors = slice.rows.filter(row => row.prediction[key] !== row.actual).length;
    assert.equal(slice.n, packet.slices[key][label].n, `${key} ${label}: support against the packet`);
    assert.equal(errors, packet.slices[key][label].errors, `${key} ${label}: errors against the packet`);
    assert.deepEqual(slice.ids, packet.slices[key][label].ids, `${key} ${label}: identifiers against the packet`);
    record('slices against the packet');
  }
  const classOne = validationRows.filter(row => row.actual === 1);
  const classOneErrors = classOne.filter(row => row.prediction[key] !== row.actual).length;
  assert.equal(classOne.length, packet.slices[key].class1.n, `${key} class1: support against the packet`);
  assert.equal(classOneErrors, packet.slices[key].class1.errors, `${key} class1: errors against the packet`);
  record('the cultivar slice against the packet');
}
/* A partition by label and a partition by a feature range are different
   partitions, and the lesson says so. Asserting it makes the claim checkable. */
const classOneIds = new Set(validationRows.filter(row => row.actual === 1).map(row => row.id));
const lowColourIds = new Set(sliceOf(400, 'lower').ids);
const overlap = [...classOneIds].filter(id => lowColourIds.has(id));
assert.ok(overlap.length > 0 && overlap.length < classOneIds.size,
  'the cultivar-1 set and the low-colour slice overlap without either containing the other, which is why '
  + 'they need not move in the same direction');
record('the two partitions genuinely overlap');

refuses(() => cutoffValue(-1), 'a cutoff below the control\'s range');
refuses(() => cutoffValue(CUTOFF_MAX_HUNDREDTHS + 1), 'a cutoff above the control\'s range');
refuses(() => cutoffValue(4.5), 'a cutoff that is not a whole number of hundredths');
refuses(() => inSlice(validationRows[0], 400, 'middle'), 'a slice side that is not a side');
refuses(() => sliceComparison({
  reference: 'linear_two', candidate: 'nothing', cutoffHundredths: 400, side: 'lower',
}), 'a comparison against a candidate that does not exist');

/* ================================================= 5 · the paired change */

let pairCases = 0;
for (const [reference, candidate] of allPairs) {
  const change = pairedChange(reference, candidate);
  const bySets = changedBySets(reference, candidate);
  assert.deepEqual(change.repaired.map(row => row.id), bySets.repaired,
    `${reference}->${candidate}: the repaired set by zip and by set difference disagree`);
  assert.deepEqual(change.broken.map(row => row.id), bySets.broken,
    `${reference}->${candidate}: the newly wrong set by zip and by set difference disagree`);
  assert.ok(change.netIdentityHolds,
    `${reference}->${candidate}: the net change is not repairs minus regressions`);
  assert.equal(change.total, 36, 'the paired view holds every development specimen');
  assert.equal(new Set(change.rows.map(row => row.id)).size, 36, 'each specimen appears once');
  assert.equal(change.referenceCorrect, candidateByKey[reference].validationCorrect,
    `${reference}: the before count matches the recorded correct count`);
  assert.equal(change.candidateCorrect, candidateByKey[candidate].validationCorrect,
    `${candidate}: the after count matches the recorded correct count`);
  if (reference === candidate) {
    assert.equal(change.repaired.length, 0, 'a candidate compared with itself repairs nothing');
    assert.equal(change.broken.length, 0, 'and breaks nothing');
    assert.equal(change.net, 0, 'and moves nothing');
  }
  for (const row of change.rows) {
    const expected = row.before === row.after
      ? (row.before ? 'right both times' : 'wrong both times')
      : (row.after ? 'repaired' : 'new error');
    assert.equal(row.state, expected, `specimen ${row.id}: the category does not follow from the two outcomes`);
  }
  pairCases += 1;
  record('the paired change by two routes');
}
assert.equal(pairCases, 16, 'every ordered pair of candidates was paired');

for (const key of ['forest_two', 'linear_three']) {
  const change = pairedChange('linear_two', key);
  assert.equal(change.repaired.length, packet.paired[key].fixed, `${key}: repairs against the packet`);
  assert.equal(change.broken.length, packet.paired[key].broken, `${key}: regressions against the packet`);
  record('the paired counts against the packet');
}
const headline = pairedChange('linear_two', 'linear_three');
assert.equal(headline.referenceCorrect, 29, 'the manuscript\'s 29 of 36');
assert.equal(headline.candidateCorrect, 32, 'the manuscript\'s 32 of 36');
assert.equal(headline.net, 3, 'the manuscript\'s net gain of three');
assert.equal(headline.repaired.length, 4, 'made of four repairs');
assert.equal(headline.broken.length, 1, 'and one new error');
const forest = pairedChange('linear_two', 'forest_two');
assert.equal(forest.candidateCorrect, 30, 'the forest\'s 30 of 36');
assert.equal(forest.repaired.length, 1, 'one repair');
assert.equal(forest.broken.length, 0, 'and no new error');
record('the headline paired counts');

/* The improvement goes backwards somewhere, and the figure must be able to say
   where. Derived, not typed: a literal would be unfalsifiable. */
const twoRecalls = classRecallTable(candidateByKey.linear_two.validationConfusion);
const threeRecalls = classRecallTable(candidateByKey.linear_three.validationConfusion);
const worsened = twoRecalls.filter((row, index) => threeRecalls[index].recall < row.recall);
const improved = twoRecalls.filter((row, index) => threeRecalls[index].recall > row.recall);
assert.equal(worsened.length, 1,
  `${worsened.length} cultivars went backwards; the lesson's claim is that exactly one does`);
assert.equal(worsened[0].actual, 0, 'and it is cultivar 0');
assert.equal(improved.length, 2, 'while two cultivars improved');
assert.ok(candidateByKey.linear_three.validationBalancedAccuracy.value
  > candidateByKey.linear_two.validationBalancedAccuracy.value,
  'the mean of the three recalls rises even so, which is the whole point');
for (const [index, row] of threeRecalls.entries()) {
  if (index === 0) continue;
  close(row.recall, 1, `cultivar ${index} reaches every validation specimen correct under linear_three`);
}
record('the subgroup that goes backwards');

/* ========================================= 6 · acceptance and cost, swept */

nonEmpty(DEFERRAL_CASES, 10, 'the ten constructed acceptance cases');
let ledgerCases = 0;
let undefinedErrorCases = 0;
let lowerSeen = 0;
let sameCostSeen = 0;
let higherSeen = 0;
for (let threshold = THRESHOLD_MIN_HUNDREDTHS; threshold <= THRESHOLD_MAX_HUNDREDTHS; threshold += 1) {
  for (let wrongCost = 0; wrongCost <= 50; wrongCost += 1) {
    for (let deferCost = 0; deferCost <= 20; deferCost += 1) {
      const ledger = acceptanceLedger({ thresholdHundredths: threshold, wrongCost, deferCost });
      ledgerCases += 1;
      /* Recounted here from the fixture rather than read back. */
      const accepted = DEFERRAL_CASES.filter(item => item.confidence >= threshold / 100);
      const wrong = accepted.filter(item => !item.correct);
      assert.equal(ledger.accepted, accepted.length, `threshold ${threshold}: accepted count`);
      assert.equal(ledger.wrong, wrong.length, `threshold ${threshold}: wrong-among-accepted count`);
      assert.equal(ledger.deferred, DEFERRAL_CASES.length - accepted.length,
        `threshold ${threshold}: deferred count`);
      assert.equal(ledger.accepted + ledger.deferred, DEFERRAL_CASES.length,
        `threshold ${threshold}: the two queues must hold every case`);
      assert.equal(new Set([...ledger.acceptedIds, ...ledger.deferredIds]).size, DEFERRAL_CASES.length,
        `threshold ${threshold}: a case is in both queues`);
      close(ledger.coverage, accepted.length / DEFERRAL_CASES.length, `threshold ${threshold}: coverage`);
      assert.equal(ledger.cost, wrongCost * wrong.length + deferCost * (DEFERRAL_CASES.length - accepted.length),
        `threshold ${threshold}, costs ${wrongCost}/${deferCost}: total cost`);
      /* The null this fixture exists for: no answers means no error rate among
         answers. Not zero, and not perfect accuracy. */
      if (accepted.length === 0) {
        assert.equal(ledger.conditionalError, null,
          `threshold ${threshold}: a rule that answers nothing must report no conditional error`);
        assert.ok(typeof ledger.conditionalErrorNote === 'string' && ledger.conditionalErrorNote.includes('undefined'),
          'and must say it is undefined');
        undefinedErrorCases += 1;
      } else {
        close(ledger.conditionalError, wrong.length / accepted.length,
          `threshold ${threshold}: conditional error`);
        assert.equal(ledger.conditionalErrorNote, null, 'a defined conditional error carries no null note');
      }
      if (wrongCost === 10 && deferCost === 2) {
        const baseline = acceptanceLedger({ thresholdHundredths: 60, wrongCost, deferCost });
        const comparison = acceptanceComparison(baseline, ledger);
        const expected = ledger.cost < baseline.cost ? 'lower' : ledger.cost > baseline.cost ? 'higher' : 'same';
        assert.equal(comparison.outcome, expected, `threshold ${threshold}: the graded cost direction`);
        assert.equal(comparison.difference, ledger.cost - baseline.cost, 'and the difference it reports');
        if (expected === 'lower') lowerSeen += 1;
        else if (expected === 'higher') higherSeen += 1;
        else sameCostSeen += 1;
      }
    }
  }
}
assert.equal(ledgerCases, (THRESHOLD_MAX_HUNDREDTHS - THRESHOLD_MIN_HUNDREDTHS + 1) * 51 * 21,
  `${ledgerCases} acceptance ledgers ran; the enterable grid was not covered`);
assert.ok(undefinedErrorCases > 0, 'the sweep met a rule that answers nothing');
assert.ok(lowerSeen > 0 && sameCostSeen > 0 && higherSeen > 0,
  `all three cost directions must occur; saw ${lowerSeen}/${sameCostSeen}/${higherSeen}`);
record('the acceptance ledger over the whole enterable grid');

/* Zero costs make every rule equal, whatever the threshold. An always-equal
   region is worth asserting: it is where a grading rule that secretly compared
   coverage rather than cost would disagree. */
for (let threshold = THRESHOLD_MIN_HUNDREDTHS; threshold <= THRESHOLD_MAX_HUNDREDTHS; threshold += 7) {
  const ledger = acceptanceLedger({ thresholdHundredths: threshold, wrongCost: 0, deferCost: 0 });
  assert.equal(ledger.cost, 0, 'with both costs zero every rule costs nothing');
  const baseline = acceptanceLedger({ thresholdHundredths: 60, wrongCost: 0, deferCost: 0 });
  assert.equal(acceptanceComparison(baseline, ledger).outcome, 'same',
    'and every comparison between them is a tie, however different their coverage');
  record('the zero-cost tie region');
}

/* The contrasts and nulls the specification recorded, by name. */
for (const [threshold, wrongCost, deferCost, expected] of [
  [60, 10, 2, { accepted: 8, wrong: 1, deferred: 2, coverage: 0.8, conditionalError: 0.125, cost: 14 }],
  [80, 10, 2, { accepted: 4, wrong: 0, deferred: 6, coverage: 0.4, conditionalError: 0, cost: 12 }],
  [60, 2, 2, { cost: 6 }],
  [80, 2, 2, { cost: 12 }],
  [61, 10, 2, { accepted: 7, wrong: 1, deferred: 3, cost: 16 }],
  [64, 10, 2, { accepted: 7, wrong: 1, deferred: 3, cost: 16 }],
  [101, 10, 2, { accepted: 0, wrong: 0, deferred: 10, coverage: 0, conditionalError: null, cost: 20 }],
]) {
  const ledger = acceptanceLedger({ thresholdHundredths: threshold, wrongCost, deferCost });
  for (const [field, value] of Object.entries(expected)) {
    if (value === null) assert.equal(ledger[field], null, `ledger ${threshold}/${wrongCost}/${deferCost}: ${field}`);
    else if (typeof value === 'number' && !Number.isInteger(value)) {
      close(ledger[field], value, `ledger ${threshold}/${wrongCost}/${deferCost}: ${field}`);
    } else assert.equal(ledger[field], value, `ledger ${threshold}/${wrongCost}/${deferCost}: ${field}`);
  }
  record('the recorded acceptance fixtures');
}
/* The reversal: the same threshold move changes direction when a wrong answer
   is repriced. If this ever stopped reversing, the lesson's claim would be
   wrong and the investigation would teach the opposite of its own prose. */
assert.equal(acceptanceComparison(
  acceptanceLedger({ thresholdHundredths: 60, wrongCost: 10, deferCost: 2 }),
  acceptanceLedger({ thresholdHundredths: 80, wrongCost: 10, deferCost: 2 })).outcome, 'lower',
'at 10 and 2, tightening the threshold lowers total cost');
assert.equal(acceptanceComparison(
  acceptanceLedger({ thresholdHundredths: 60, wrongCost: 2, deferCost: 2 }),
  acceptanceLedger({ thresholdHundredths: 80, wrongCost: 2, deferCost: 2 })).outcome, 'higher',
'at 2 and 2, the same move raises it');
record('the cost reversal the investigation promises');
/* Two different thresholds giving the identical ledger: a genuine null where a
   control moved and nothing did. */
const at61 = acceptanceLedger({ thresholdHundredths: 61, wrongCost: 10, deferCost: 2 });
const at64 = acceptanceLedger({ thresholdHundredths: 64, wrongCost: 10, deferCost: 2 });
assert.deepEqual(at61.acceptedIds, at64.acceptedIds, 'thresholds .61 and .64 accept exactly the same cases');
assert.equal(acceptanceComparison(at61, at64).outcome, 'same', 'and cost the same');
record('the moved-control null');

/* Editing a score changes the decision, never the outcome. */
const edited = acceptanceLedger({ thresholdHundredths: 60, wrongCost: 10, deferCost: 2,
  scoreHundredths: { 6: 95 } });
assert.deepEqual(edited.cases.map(item => item.correct), DEFERRAL_CASES.map(item => item.correct),
  'editing a score must not change whether a case was answered correctly');
assert.ok(edited.acceptedIds.includes(6), 'the edited case is now accepted');
assert.ok(edited.cases.find(item => item.id === 6).edited, 'and is marked as edited');
const unedited = acceptanceLedger({ thresholdHundredths: 60, wrongCost: 10, deferCost: 2,
  scoreHundredths: { 6: 70 } });
assert.ok(!unedited.cases.find(item => item.id === 6).edited,
  'setting a score to the value it already had is not an edit');
record('score edits change the decision, not the outcome');

refuses(() => acceptanceLedger({ thresholdHundredths: 49, wrongCost: 10, deferCost: 2 }),
  'a threshold below the control\'s range');
refuses(() => acceptanceLedger({ thresholdHundredths: 102, wrongCost: 10, deferCost: 2 }),
  'a threshold above the control\'s range');
refuses(() => acceptanceLedger({ thresholdHundredths: 60, wrongCost: -1, deferCost: 2 }), 'a negative cost');
refuses(() => acceptanceLedger({ thresholdHundredths: 60, wrongCost: 51, deferCost: 2 }),
  'a cost above the control\'s range');
refuses(() => acceptanceLedger({ thresholdHundredths: 60, wrongCost: Number.NaN, deferCost: 2 }),
  'a cost that is not a number');
refuses(() => acceptanceLedger({ thresholdHundredths: 60, wrongCost: 10, deferCost: 2,
  scoreHundredths: { 1: SCORE_MAX_HUNDREDTHS + 1 } }), 'an edited score above the control\'s range');
refuses(() => acceptanceLedger({ thresholdHundredths: 60, wrongCost: 10, deferCost: 2,
  scoreHundredths: { 1: SCORE_MIN_HUNDREDTHS - 1 } }), 'an edited score below the control\'s range');

/* ==================================== 7 · the held-out block and its gating */

close(endToEndData.heldOut.accuracy.value, packet.test.accuracy, 'held-out accuracy against the packet');
close(endToEndData.heldOut.balancedAccuracy.value, packet.test.balancedAccuracy,
  'held-out balanced accuracy against the packet');
close(endToEndData.heldOut.logLoss.value, packet.test.logLoss, 'held-out log loss against the packet');
assert.deepEqual(endToEndData.heldOut.confusion, packet.test.confusion,
  'the held-out confusion matrix against the packet');
close(accuracyFromConfusion(packet.test.confusion), packet.test.accuracy,
  'the held-out accuracy is its own confusion matrix\'s diagonal over its total');
close(balancedFromConfusion(packet.test.confusion), packet.test.balancedAccuracy,
  'the held-out balanced accuracy is the mean of its own matrix\'s recalls');
assert.equal(endToEndData.heldOut.correct, 35, '35 of the 36 held-out specimens are correct');
assert.equal(endToEndData.heldOut.selected, 'linear_three', 'the frozen model is the declared selection');
assert.ok(endToEndData.heldOut.balancedAccuracy.value
  > candidateByKey.linear_three.validationBalancedAccuracy.value,
  'the held-out estimate exceeds the validation one here, which is allowed and is what the lesson says');
record('the held-out block against the packet');

const wilson = wilsonInterval(35, 36, 1.96);
const byQuadratic = wilsonByQuadratic(35, 36, 1.96);
close(wilson.lower, byQuadratic.lower, 'the Wilson lower bound by closed form and by the quadratic it solves');
close(wilson.upper, byQuadratic.upper, 'the Wilson upper bound by closed form and by the quadratic it solves');
close(wilson.lower, packet.wilsonIllustration[0], 'the Wilson lower bound against the packet');
close(wilson.upper, packet.wilsonIllustration[1], 'the Wilson upper bound against the packet');
assert.ok(wilson.upper < 1, 'the interval stops short of 1 even at 35 of 36');
assert.ok(wilson.lower < 35 / 36 && 35 / 36 < wilson.upper, 'and contains the observed proportion');
/* Edges. At an observed proportion of exactly 1 the Wilson upper limit is
   exactly 1, and at exactly 0 the lower limit is exactly 0 -- the centre and
   the radius coincide there. That is a property of the interval, not a
   rounding artefact, and asserting the equality rather than a strict
   inequality is what makes this a check rather than a guess. */
const perfect = wilsonInterval(36, 36, 1.96);
assert.equal(perfect.upper, 1, 'at 36 of 36 the Wilson upper limit is exactly 1');
assert.ok(perfect.lower > 0 && perfect.lower < 1, 'while its lower limit stays strictly inside the unit range');
const empty = wilsonInterval(0, 36, 1.96);
assert.equal(empty.lower, 0, 'at 0 of 36 the Wilson lower limit is exactly 0');
assert.ok(empty.upper > 0 && empty.upper < 1, 'while its upper limit stays strictly inside the unit range');
assert.ok(wilson.upper < perfect.upper,
  'and 35 of 36 gives a strictly narrower upper reach than 36 of 36, so the interval responds to the count');
refuses(() => wilsonInterval(37, 36, 1.96), 'more successes than trials');
refuses(() => wilsonInterval(-1, 36, 1.96), 'a negative count of successes');
refuses(() => wilsonInterval(1, 0, 1.96), 'a Wilson interval over no trials');
record('the Wilson interval by two routes');

/* THE property this topic exists for, asserted on the actual text the page
   shows before the gate: the development half of the study's output contains no
   held-out quantity. */
const developmentText = endToEndExamples.study.developmentOutput;
const heldOutText = endToEndExamples.study.heldOutOutput;
/* THE PAIRED RULE, APPLIED WITHOUT EXCEPTION IN THIS BLOCK.
 *
 * Every quantity below is asserted ABSENT from the development half and PRESENT
 * in the held-out half, from one list, so an entry cannot be absent-only. The
 * two Wilson bounds used to sit in the absence list and in no presence list,
 * and they could never have matched for two independent reasons: the page
 * prints them at three decimals, and the subject here is the PROGRAM'S STDOUT,
 * which computes no interval at any precision. Two of five absence assertions
 * were therefore permanently inert, and each still bumped the counter.
 *
 * That is the identical defect found and fixed in verify-endtoend-browser.cjs
 * during phase C and not back-ported here — and it survived precisely because
 * this file broke the paired rule for those two entries. An absence check alone
 * cannot distinguish "correctly withheld" from "looking in the wrong place".
 * The Wilson bounds belong to the page, not to this program's output, and are
 * checked there with both halves of the pair.
 */
const heldOutQuantities = [
  packet.test.balancedAccuracy.toFixed(6), packet.test.accuracy.toFixed(6), packet.test.logLoss.toFixed(6),
  '[[',
];
nonEmpty(heldOutQuantities, 4, 'the held-out quantities that must not leak into the development output');
for (const quantity of heldOutQuantities) {
  assert.ok(!developmentText.includes(quantity),
    `the development half of the study's output leaks the held-out quantity ${quantity}`);
  assert.ok(heldOutText.includes(quantity),
    `the held-out half is missing ${quantity}, so the absence assertion just made could never have failed: `
    + 'it would pass for a quantity this subject cannot contain at all');
  record('a held-out quantity absent from the development output and present in the held-out half');
}
/* The Wilson bounds are NOT in that list, and the reason is recorded rather
   than left to inference: the program never computes them. What they must not
   do is appear in either half of its output at the precision the page prints. */
for (const bound of [fixed(wilson.lower, 3), fixed(wilson.upper, 3)]) {
  assert.ok(!developmentText.includes(bound) && !heldOutText.includes(bound),
    `${bound} appears in the study program's output, which computes no Wilson interval; if that changed, `
    + 'this check belongs in the leak list above with a presence partner');
  record('the program produces no Wilson bound in either half');
}
assert.ok(developmentText.includes('selected linear_three'),
  'the selection stays in the development half, because a selection is development evidence');
/* A validation quantity and a held-out quantity must not happen to be the same
   text, or the leak check above could not tell them apart. */
for (const candidate of endToEndData.candidates) {
  for (const field of ['validationBalancedAccuracy', 'validationAccuracy', 'validationLogLoss']) {
    assert.ok(!heldOutQuantities.includes(candidate[field].value.toFixed(6)),
      `${candidate.key}.${field} prints the same six decimals as a held-out quantity, which would make the `
      + 'leak check unable to distinguish them');
    record('validation and held-out text do not collide');
  }
}

/* ================================================= 8 · the drawn geometry */

const geometries = [
  ['scatter', scatterGeometry({
    cutoffHundredths: 400, side: 'lower', reference: 'linear_two', candidate: 'linear_three',
  })],
  ['candidate bars', candidateBarGeometry({ metricKey: 'validationBalancedAccuracy' })],
  ['candidate bars, log loss', candidateBarGeometry({ metricKey: 'validationLogLoss' })],
  ['recall comparison', recallComparisonGeometry({ reference: 'linear_two', candidate: 'linear_three' })],
];
for (const [name, geometry] of geometries) {
  assert.ok(Array.isArray(geometry.xTicks), `${name}: xTicks is a list, empty for a categorical axis`);
  assert.ok(Array.isArray(geometry.yTicks), `${name}: yTicks is a list`);
  for (const tick of geometry.yTicks) {
    assert.ok(tick.y >= geometry.padding.top - 1 && tick.y <= geometry.height - geometry.padding.bottom + 1,
      `${name}: a y tick at ${tick.y} falls outside the plotting area`);
  }
  for (const tick of geometry.xTicks) {
    assert.ok(tick.x >= geometry.padding.left - 1 && tick.x <= geometry.width - geometry.padding.right + 1,
      `${name}: an x tick at ${tick.x} falls outside the plotting area`);
  }
  record('plot frames put their ticks inside themselves');
}

const scatter = scatterGeometry({
  cutoffHundredths: 400, side: 'lower', reference: 'linear_two', candidate: 'linear_three',
});
assert.deepEqual(scatter.xDomain, [11, 15], 'the alcohol axis spans the fixed bounds the specification names');
assert.deepEqual(scatter.yDomain, [0, 14], 'and the colour axis spans its own');
for (const row of validationRows) {
  assert.ok(row.alcohol > scatter.xDomain[0] && row.alcohol < scatter.xDomain[1],
    `specimen ${row.id}: alcohol ${row.alcohol} falls outside the fixed x bounds`);
  assert.ok(row.colorIntensity > scatter.yDomain[0] && row.colorIntensity < scatter.yDomain[1],
    `specimen ${row.id}: colour ${row.colorIntensity} falls outside the fixed y bounds`);
  record('every specimen is inside the fixed axis bounds');
}
for (const point of scatter.points) {
  assert.ok(point.cx >= scatter.padding.left && point.cx <= scatter.width - scatter.padding.right,
    `specimen ${point.id} is drawn outside the plot horizontally`);
  assert.ok(point.cy >= scatter.padding.top && point.cy <= scatter.height - scatter.padding.bottom,
    `specimen ${point.id} is drawn outside the plot vertically`);
}
/* The coordinates do not move when the model changes. The specification is
   explicit about this and it is the reason flavanoids gets its own strip. */
for (const candidate of candidateOrder) {
  const other = scatterGeometry({
    cutoffHundredths: 400, side: 'lower', reference: 'linear_two', candidate,
  });
  for (let index = 0; index < scatter.points.length; index += 1) {
    assert.equal(other.points[index].cx, scatter.points[index].cx,
      `changing the candidate to ${candidate} moved a specimen horizontally`);
    assert.equal(other.points[index].cy, scatter.points[index].cy,
      `changing the candidate to ${candidate} moved a specimen vertically`);
  }
  record('specimen positions do not depend on the model');
}

const strip = flavanoidStripGeometry({});
nonEmpty(strip.marks, 36, 'the flavanoid strip carries every development specimen');
insideFrame(strip.marks.map(mark => mark.x), strip.width, 'flavanoid marks');
assert.ok(strip.inset >= HARD_MARGIN, 'the strip inset is at or above the literal margin');
assert.ok(strip.domain[1] >= Math.max(...validationRows.map(row => row.flavanoids)),
  'the flavanoid axis reaches the largest value it must show');
record('the flavanoid strip');

const paired = pairedStripGeometry({ reference: 'linear_two', candidate: 'linear_three' });
nonEmpty(paired.columns, 36, 'the paired strip draws every specimen');
for (const column of paired.columns) {
  assert.ok(column.left >= HARD_MARGIN - 1
    && column.left + column.barWidth <= paired.width - HARD_MARGIN + 1,
  `specimen ${column.id}'s column falls outside the ${HARD_MARGIN}-unit frame margin`);
  assert.ok(column.barWidth > 0, `specimen ${column.id}'s column has no width`);
  assert.ok(column.afterY > column.beforeY, 'the after row is drawn below the before row');
  assert.equal(column.changed, column.state === 'repaired' || column.state === 'new error',
    `specimen ${column.id}: a link is drawn if and only if the outcome changed`);
}
assert.equal(paired.columns.filter(column => column.changed).length, 5,
  'exactly five specimens are linked, which is four repairs plus one new error');
/* Consecutive columns do not overlap: an aggregate strip whose marks collide
   cannot show which specimen moved. */
for (let index = 1; index < paired.columns.length; index += 1) {
  assert.ok(paired.columns[index].left >= paired.columns[index - 1].left + paired.columns[index - 1].barWidth,
    `columns ${index - 1} and ${index} of the paired strip overlap`);
  record('paired columns do not overlap');
}

const bars = candidateBarGeometry({ metricKey: 'validationBalancedAccuracy' });
nonEmpty(bars.bars, 4, 'the candidate chart draws every candidate');
assert.ok(bars.baselineIncluded, 'the majority baseline is in the frame, which is what anchors the comparison');
/* Every metric the chart can be asked for, so a minimum length that binds on
   one chart and not another cannot hide. The shortest bar on the page is the
   majority baseline's, which is exactly the one a floor would lengthen. */
let barCases = 0;
let shortestBar = Infinity;
for (const metric of SELECTION_METRICS) {
  const chart = candidateBarGeometry({ metricKey: metric.key });
  nonEmpty(chart.bars, 4, `the ${metric.key} chart draws every candidate`);
  for (const bar of chart.bars) {
    close(bar.barHeight, (chart.height - chart.padding.bottom) - chart.y(bar.value),
      `${metric.key}/${bar.key}: the bar's height is not its own value's position`);
    assert.ok(bar.barHeight >= 0, `${metric.key}/${bar.key}: a negative bar height`);
    assert.ok(bar.y >= chart.padding.top - 1e-9, `${metric.key}/${bar.key}: the bar starts above the plot`);
    assert.ok(bar.x >= chart.padding.left && bar.x + bar.barWidth <= chart.width - chart.padding.right,
      `${metric.key}/${bar.key}: the bar falls outside the plot`);
    assert.equal(bar.role, 'validation', `${metric.key}/${bar.key}: drawn from a validation quantity`);
    shortestBar = Math.min(shortestBar, bar.barHeight);
    barCases += 1;
    record('bars encode their own values');
  }
  /* Proportionality, asserted rather than assumed: a bar twice as long means
     twice the value, and no bar has an unexplained minimum length. */
  const sortedBars = [...chart.bars].sort((a, b) => a.value - b.value);
  for (let index = 1; index < sortedBars.length; index += 1) {
    assert.ok(sortedBars[index].barHeight >= sortedBars[index - 1].barHeight,
      `${metric.key}: a larger value is drawn shorter than a smaller one`);
    const ratio = sortedBars[index].barHeight / sortedBars[index - 1].barHeight;
    close(ratio, sortedBars[index].value / sortedBars[index - 1].value,
      `${metric.key}: bar lengths are proportional to the values they encode`, 1e-9);
    record('bar lengths are proportional');
  }
}
assert.equal(barCases, 4 * SELECTION_METRICS.length, `${barCases} bars were checked`);
assert.ok(shortestBar < 40,
  `the shortest bar on any of these charts is ${shortestBar} units. The proportionality check can only `
  + 'catch a minimum-length floor that actually binds, so if every bar were long the check would be inert; '
  + 'this records that at least one bar is short enough for a floor to change it');

const recall = recallComparisonGeometry({ reference: 'linear_two', candidate: 'linear_three' });
nonEmpty(recall.groups, 3, 'the recall figure draws all three cultivars');
assert.deepEqual(recall.worsenedClasses, [0], 'and marks cultivar 0 as the one that went backwards');
for (const group of recall.groups) {
  nonEmpty(group.bars, 2, `cultivar ${group.actual} has a bar for each candidate`);
  for (const bar of group.bars) {
    close(bar.barHeight, (recall.height - recall.padding.bottom) - recall.y(bar.recall),
      `cultivar ${group.actual}, ${bar.key}: the bar's height is not its recall's position`);
    assert.ok(bar.x >= recall.padding.left && bar.x + bar.barWidth <= recall.width - recall.padding.right,
      `cultivar ${group.actual}, ${bar.key}: the bar falls outside the plot`);
    close(bar.recall, bar.correct / bar.support, 'the recall drawn is the count over its denominator');
  }
  assert.ok(group.bars[0].x + group.bars[0].barWidth <= group.bars[1].x,
    `cultivar ${group.actual}: the two bars overlap`);
  assert.equal(group.worsened, group.bars[1].recall < group.bars[0].recall,
    `cultivar ${group.actual}: the worsened flag disagrees with its own two bars`);
  record('recall bars');
}

const lanes = informationLaneGeometry();
nonEmpty(lanes.lanes, 3, 'the lane figure draws three lanes');
nonEmpty(lanes.arrows, 6, 'and its six arrows');
assert.deepEqual(lanes.lanes.map(lane => lane.key), ['train', 'validation', 'test'],
  'in reading order: train, then validation, then test');
assert.deepEqual(lanes.lanes.map(lane => lane.rows), [106, 36, 36], 'with the contract\'s own split sizes');
for (const lane of lanes.lanes) {
  assert.ok(lane.x >= HARD_MARGIN && lane.x + lane.width <= lanes.width - HARD_MARGIN,
    `lane ${lane.key} falls outside the ${HARD_MARGIN}-unit frame margin`);
  assert.ok(lane.y >= HARD_MARGIN && lane.y + lane.height <= lanes.height - HARD_MARGIN,
    `lane ${lane.key} falls outside the frame margin vertically`);
}
for (let index = 1; index < lanes.lanes.length; index += 1) {
  assert.ok(lanes.lanes[index].y >= lanes.lanes[index - 1].y + lanes.lanes[index - 1].height,
    'two lanes overlap');
  record('lanes do not overlap');
}
for (const arrow of lanes.arrows) {
  for (const [name, value] of [['x1', arrow.x1], ['x2', arrow.x2]]) {
    assert.ok(value >= 0 && value <= lanes.width, `arrow ${arrow.key}'s ${name} leaves the figure`);
  }
  for (const [name, value] of [['y1', arrow.y1], ['y2', arrow.y2]]) {
    assert.ok(value >= 0 && value <= lanes.height, `arrow ${arrow.key}'s ${name} leaves the figure`);
  }
  assert.ok(arrow.label.length > 0, `arrow ${arrow.key} carries a word as well as a colour`);
  record('arrows stay inside the figure and carry words');
}
/* The absent edges are the teaching claim. An arrow from a label into a
   transform, or from the test lane into the selection, would say the opposite
   of everything the section argues. */
assert.equal(lanes.arrows.filter(arrow => arrow.from === 'test' && arrow.to === 'selection').length, 0,
  'no arrow runs from the test lane into candidate selection');
assert.equal(lanes.arrows.filter(arrow => arrow.from === 'label').length, 0,
  'no arrow runs from a target label into anything');
assert.equal(lanes.arrows.filter(arrow => arrow.to === 'pipeline' && arrow.kind === 'fits').length, 1,
  'exactly one lane fits the pipeline');
assert.equal(lanes.arrows.find(arrow => arrow.kind === 'fits').from, 'train',
  'and it is the training lane');
assert.equal(lanes.arrows.filter(arrow => arrow.kind === 'applies').length, 2,
  'the other two lanes only apply it');
assert.equal(lanes.arrows.filter(arrow => arrow.kind === 'counterexample').length, 1,
  'the human-mediated leak is drawn once, as a labelled counterexample');
nonEmpty(lanes.forbiddenEdges, 2, 'the two edges the figure exists to leave out are named');
record('the lane figure\'s absent edges');

const rail = acceptanceRailGeometry({
  ledger: acceptanceLedger({ thresholdHundredths: 60, wrongCost: 10, deferCost: 2 }),
});
nonEmpty(rail.tiles, 10, 'the rail draws every constructed case');
for (const tile of rail.tiles) {
  assert.ok(tile.x >= HARD_MARGIN && tile.x <= rail.width - HARD_MARGIN,
    `case ${tile.id}'s tile falls outside the ${HARD_MARGIN}-unit frame margin`);
  assert.equal(tile.accepted, tile.confidence >= 0.6,
    `case ${tile.id}: the tile's routing disagrees with the rule`);
}
assert.equal(rail.queues.reduce((sum, queue) => sum + queue.ids.length, 0), 10,
  'every case lands in exactly one queue');
close(rail.thresholdX, rail.x(0.6), 'the threshold line is drawn at the threshold');
/* The rule is drawn at the true threshold AND around the tile band, so the tile
   sitting exactly on the threshold does not have its own number crossed out.
   That case is the most interesting one on the rail, and relying on an opaque
   rectangle painted afterwards to hide the crossing is paint order doing the
   work of layout. */
nonEmpty(rail.thresholdSegments, 2, 'the threshold rule is drawn as two segments');
/* Asserted against the TWO LABEL POSITIONS the component actually draws text
   at, not against the band the segments are defined from.
   The previous form compared each segment's end with `tileBand.top`/`.bottom`
   — the very values `acceptanceRailGeometry` derives those ends from — so both
   disjuncts were true by construction for any band and any railY, and the
   tick-label conjunct reduced to `railY + 3 < railY + 7`, in which railY
   cancels. The geometry was right and the check could not have noticed if it
   had stopped being right, which is the whole point of the Phase C defect it
   was written for. */
const CLEARANCE = 3;
for (const segment of rail.thresholdSegments) {
  assert.ok(segment.y2 > segment.y1, 'a threshold segment has no length');
  for (const [name, labelY] of [['the tile number', rail.tileLabelY], ['the axis tick label', rail.tickLabelY]]) {
    assert.ok(labelY <= segment.y1 - CLEARANCE || labelY >= segment.y2 + CLEARANCE,
      `a threshold segment spanning ${segment.y1}..${segment.y2} reaches ${name} at y=${labelY}, within `
      + `${CLEARANCE} units: the rule would be drawn through a number the learner has to read`);
  }
  record('the threshold rule clears both label rows it could cross');
}
/* And the gap between the segments is actually where the tile number is, so the
   two segments are not merely short but placed around it. */
assert.ok(rail.thresholdSegments[0].y2 < rail.tileLabelY && rail.tileLabelY < rail.thresholdSegments[1].y1,
  `the tile number at y=${rail.tileLabelY} does not sit in the gap between the two threshold segments, so `
  + 'the rule is interrupted somewhere other than where the number is');
record('the interruption in the rule is where the number is');
const onThreshold = rail.tiles.filter(tile => tile.x === rail.thresholdX);
assert.equal(onThreshold.length, 1,
  `${onThreshold.length} tiles sit at exactly the default threshold; one does, and it is the reason the rule `
  + 'is drawn in two segments rather than one');
const beyond = acceptanceRailGeometry({
  ledger: acceptanceLedger({ thresholdHundredths: 101, wrongCost: 10, deferCost: 2 }),
});
assert.ok(beyond.thresholdBeyondRail,
  'a threshold above the rail says so rather than being drawn off the end of it');
assert.equal(beyond.tiles.filter(tile => tile.accepted).length, 0, 'and accepts nothing');
record('the acceptance rail');

/* The scale helper itself, since every figure above depends on it. */
const scale = linearScale([0, 10], [100, 0]);
close(scale(0), 100, 'a decreasing scale maps its domain start to its range start');
close(scale(10), 0, 'and its domain end to its range end');
close(scale(5), 50, 'and is linear between them');
close(scale.invert(50), 5, 'and inverts');
refuses(() => linearScale([3, 3], [0, 1]), 'a scale over an empty domain');
record('the shared scale');

/* ======================================== 9 · the manuscript's stated values */

const stated = [
  ['0.792857', 'the two-feature linear validation balanced accuracy'],
  ['0.820635', 'the forest validation balanced accuracy'],
  ['0.888889', 'the three-feature linear validation balanced accuracy'],
  ['0.333333', 'the majority validation balanced accuracy'],
  ['0.522805', 'the two-feature linear validation log loss'],
  ['0.218008', 'the three-feature linear validation log loss'],
];
for (const [text, label] of stated) {
  assert.ok(manuscript.includes(text), `${label}: the manuscript no longer states ${text}`);
  record('the manuscript\'s stated values are still stated');
}
close(Number('0.792857'), Number(candidateByKey.linear_two.validationBalancedAccuracy.value.toFixed(6)),
  'the two-feature linear score the manuscript prints is the one the page computes');
close(Number('0.888889'), Number(candidateByKey.linear_three.validationBalancedAccuracy.value.toFixed(6)),
  'and so is the three-feature one');
assert.equal(asInput(4), '4', 'an input is never printed with six decimals');
assert.equal(fixed(4, 6), '4.000000', 'while a computed value always is');
assert.notEqual(asInput(4), fixed(4, 6),
  'the two forms differ, which is what lets the browser verifier pin a computed value');
record('the two print forms are distinguishable');

/* The practice values the lesson states, recomputed. */
const practiceConfusion = endToEndData.practice.confusion;
close(accuracyFromConfusion(practiceConfusion), endToEndData.practice.accuracy,
  'practice 1 accuracy from its own matrix');
close(balancedFromConfusion(practiceConfusion),
  endToEndData.practice.balancedAccuracyNumerator / endToEndData.practice.balancedAccuracyDenominator,
  'practice 1 balanced accuracy from its own matrix');
close(endToEndData.practice.balancedAccuracyNumerator / endToEndData.practice.balancedAccuracyDenominator,
  32 / 45, 'and it is 32/45');
const practiceRecalls = classRecallTable(practiceConfusion);
assert.equal(practiceRecalls.reduce((best, row) => (row.recall < best.recall ? row : best)).actual, 2,
  'practice 1: class 2 has the lowest recall, which is the class to inspect first');
record('the practice arithmetic');

/* ================================================================ evidence */

const sources = [
  'src/learn/data/endtoend-models.js',
  'src/learn/data/endtoend-data.js',
  'src/learn/data/endtoend-examples.js',
  'src/learn/data/topics/end-to-end-supervised-learning-error-analysis.jsx',
  'src/learn/data/curriculum/blueprints/end-to-end-supervised-learning-error-analysis.js',
  'src/learn/components/lesson-labs/EndToEndShared.jsx',
  'src/learn/components/lesson-labs/EndToEndLabs.jsx',
  'src/learn/components/lesson-labs/EndToEndFigures.jsx',
  'src/learn/components/lesson-labs/endtoend-labs.css',
  'public/learn-assets/end-to-end/wine.csv',
  'public/learn-assets/end-to-end/wine_study.py',
  'public/learn-assets/end-to-end/ATTRIBUTION.txt',
];
// Unfiltered on purpose: silently dropping a declared source that no longer
// exists and still writing `passed: true` is how a deleted file passes.
const missingSources = sources.filter(file => !fs.existsSync(file));
assert.deepEqual(missingSources, [], `declared source files are missing: ${missingSources.join(', ')}`);

const hash = file => crypto.createHash('sha256').update(fs.readFileSync(file)).digest('hex');
const total = Object.values(counts).reduce((sum, value) => sum + value, 0);
/* A counter that is reported but never floored is decoration: deleting every
 * `record()` call still printed PASS with a smaller number. These floors sit
 * below the current values but far above zero, so they catch a wholesale loss
 * of checking rather than the removal of one call. */
assert(total >= 240, `only ${total} grouped checks ran; the suite has lost coverage`);
assert(Object.keys(counts).length >= 45, `only ${Object.keys(counts).length} groups ran`);
/* The floor was written against a stale 25,218 from a nine-pair version and
   left behind when the sweep grew to sixteen pairs, so it sat 44% below what
   the suite actually runs. The exact equality above binds the total; this
   floor catches a wholesale loss, and now sits just under the real figure. */
assert(sliceCases >= 44000, `only ${sliceCases} slice comparisons ran`);
assert(ledgerCases >= 55000, `only ${ledgerCases} acceptance ledgers ran`);
assert(selectionCases === 48, `only ${selectionCases} selection settings ran`);
assert(pairCases === 16, `only ${pairCases} candidate pairings ran`);
assert(scoreFields === 19, `only ${scoreFields} score records were inspected`);

const evidence = {
  startedAt,
  checkedAt: new Date().toISOString(),
  verifier: 'scripts/verify-endtoend-models.mjs',
  verifierSha256: hash('scripts/verify-endtoend-models.mjs'),
  sourceHashes: Object.fromEntries(sources.map(file => [file, hash(file)])),
  packetResultsSha256: hash(`${packetDirectory}/calculated-inputs.json`),
  manuscriptSha256: hash(`${packetDirectory}/lesson.md`),
  counts,
  totalGroupedChecks: total,
  sliceComparisonsSwept: sliceCases,
  acceptanceLedgersSwept: ledgerCases,
  selectionSettingsSwept: selectionCases,
  candidatePairingsSwept: pairCases,
  scoreRecordsInspected: scoreFields,
  heldOutOfferedSettings: heldOutOffered,
  heldOutRefusedSettings: heldOutRefused,
  sliceOutcomesSeen: { fewer: fewerSeen, same: sameSeen, more: moreSeen },
  costOutcomesSeen: { lower: lowerSeen, same: sameCostSeen, higher: higherSeen },
  scope: 'The browser models for the end-to-end study against the content packet\'s calculated-inputs.json, '
    + 'the manuscript\'s stated values, and a second derivation for every claim a figure draws. Accuracy and '
    + 'balanced accuracy are recomputed from the confusion matrix while the module walks the label vectors; '
    + 'log loss is recomputed by a compensated sum in the opposite order; the Wilson interval is recovered by '
    + 'solving the quadratic it is defined by; the repaired and newly wrong sets are recomputed by set '
    + 'difference over identifiers; the selection winner is recomputed by an explicit scan with the declared '
    + 'tie rule written out. The slice rule is swept over all 1,401 cutoffs the control admits, on both '
    + 'sides, for all 16 ordered candidate pairs, and at each of the 44,832 comparisons the scatter\'s own '
    + 'marks, the cutoff line\'s own position and the graded difference are required to agree; the one '
    + 'specimen sitting exactly on a cutoff is checked to belong to the upper side. The acceptance ledger is '
    + 'swept over every threshold and every integer cost pair its controls admit. Every score record is '
    + 'checked to carry one of the four roles, the held-out block is checked to carry only the held-out role, '
    + 'and the development half of the study\'s recorded output is checked to contain none of the five '
    + 'held-out quantities while the held-out half contains them all.',
  limitations: [
    'The recorded predictions and probabilities are the fitted estimators\' own output. They are regenerated '
      + 'from the served dataset by scripts/verify-endtoend-data.py, which rebuilds the standardisation, the '
      + 'softmax and the forest\'s tree walk independently; this file checks that the browser modules carry '
      + 'those recorded values unchanged and that everything computed from them is right.',
    'Displayed program output is executed separately by scripts/verify-endtoend-examples.py; this file checks '
      + 'that what the program printed agrees with what the browser models hold.',
    'The tie rule is exercised on constructed ties, because no two candidates tie under any of the three '
      + 'metrics on these data. What is checked is that the rule is "first in the declared order" rather than '
      + 'the order a control happened to offer.',
    'Geometry is checked as numbers. Whether a mark is legible, whether a label collides with a line, and '
      + 'whether the held-out gate actually withholds anything in a browser are separate steps, checked by '
      + 'scripts/verify-endtoend-browser.cjs.',
  ],
  passed: true,
};
if (keepEvidence) {
  fs.writeFileSync(evidencePath, `${JSON.stringify(evidence, null, 2)}\n`);
}
console.log(`PASS: ${total} grouped end-to-end model checks across ${Object.keys(counts).length} groups, `
  + `including ${sliceCases.toLocaleString('en-US')} slice comparisons over every cutoff the control admits `
  + `on both sides for all 16 candidate pairs (${fewerSeen.toLocaleString('en-US')} fewer, `
  + `${sameSeen.toLocaleString('en-US')} same, ${moreSeen.toLocaleString('en-US')} more), `
  + `${ledgerCases.toLocaleString('en-US')} acceptance ledgers over every threshold and cost pair, `
  + `${selectionCases} selection settings of which exactly ${heldOutOffered} may open the held-out report, `
  + `${pairCases} candidate pairings and ${scoreFields} score records each carrying its role.`);
