// Bounded independent checks of the hidden Markov model browser models against
// the content packet's recorded calculations, the manuscript's worked values and
// analytic identities, plus the quantities the figures draw, the rules the
// investigations grade with, and the shape of the generated data and example
// modules.
//
// Every assertion here is written so that it can fail: nothing is compared with
// itself, no loop is allowed to inspect an empty subject set, and every group
// asserts how many subjects it actually saw.
//
// Two things this topic needs beyond the usual:
//
//   * The browser's inference uses RETAINED SCALING while the content packet's
//     author program uses log space throughout, so matching the two is evidence
//     rather than an echo. Where a third route is cheap - the sixteen-path
//     enumeration, brute-force search over legal paths - it is used as well.
//   * Every rule an investigation grades with is exercised over the WHOLE grid
//     of inputs its control can reach, not at samples: all sixteen paths of four
//     models, all 256 four-step recordings at every query time, all 231 priors
//     on a twentieths grid of the probability simplex on three states, every composition of up to
//     eight reports into up to four recordings, and every self-transition
//     probability on a hundredths grid.
//
// Run: node scripts/verify-hmm-models.mjs
import assert from 'node:assert/strict';
import fs from 'node:fs';
import crypto from 'node:crypto';
import {
  MISSING, backtrack, barHeight, beliefMoveOutcomes, beliefTrack, bestLegalPath, checkModel, checkRow,
  countFlow, direction, durationBars, durationModel, edgeWidth, edgeWidthRange, emStep, enumeratePaths,
  envelopes, exitAfter, expectedCounts, fixtures, forwardScaled, gradeBoundaryConstruction,
  gradeLegalPathConstruction, gradeSmoothedConstruction, hazardOutcome, infer, insideWindow,
  legalPathGuarantee, limits,
  localLikelihood, logSumExp, normalizeCounts, objectiveWindows, parameterCount, pathJointOf, pathLegal,
  pathRankOutcome, pathShare, predictNext, recordingTotals, repeatedProduct, separationInPixels, trellis,
  trellisEdges, withRainyEmission, withStart,
} from '../src/learn/data/hmm-models.js';
import {
  configurations, decisionChanges, developmentSentences, emTrack, fittedModels, majority, provenance,
  selectedConfigurationIndex, tieAudit, unknownDevelopmentTokens, vocabulary,
} from '../src/learn/data/hmm-data.js';
import { hmmExamples } from '../src/learn/data/hmm-examples.js';

const packetDirectory = 'docs/teaching/drafts/hidden-markov-models-hmm';
const recorded = JSON.parse(fs.readFileSync(`${packetDirectory}/calculated-inputs.json`, 'utf8'));
const mechanisms = recorded.mechanisms;

const counts = {};
const record = name => { counts[name] = (counts[name] ?? 0) + 1; };
/* Executed assertions, not group markers. `record()` counts hand-placed labels,
   so a grid of two thousand comparisons contributed one to it; a number in a
   summary line is a claim about coverage and has to be the number of things
   actually checked. Every assertion in this file goes through one of the
   helpers below or through `ok`/`equals`/`deepEquals`. */
let assertions = 0;
const ok = (condition, message) => { assertions += 1; assert.ok(condition, message); };
const equals = (actual, expected, message) => { assertions += 1; assert.equal(actual, expected, message); };
const notEquals = (actual, expected, message) => { assertions += 1; assert.notEqual(actual, expected, message); };
const deepEquals = (actual, expected, message) => { assertions += 1; assert.deepEqual(actual, expected, message); };
const throwsRange = (call, message) => { assertions += 1; assert.throws(call, RangeError, message); };
const close = (actual, expected, label, tolerance = 1e-9) => {
  assertions += 2;
  assert(typeof actual === 'number' && Number.isFinite(actual), `${label}: ${actual} is not a finite number`);
  assert(globalThis.Math.abs(actual - expected) <= tolerance * globalThis.Math.max(1, globalThis.Math.abs(expected)),
    `${label}: ${actual} versus ${expected}`);
};
const vector = (actual, expected, label, tolerance = 1e-9) => {
  assert.equal(actual.length, expected.length, `${label}: length ${actual.length} versus ${expected.length}`);
  actual.forEach((value, index) => close(value, expected[index], `${label}[${index}]`, tolerance));
};
const grid = (actual, expected, label, tolerance = 1e-9) => {
  assert.equal(actual.length, expected.length, `${label}: rows ${actual.length} versus ${expected.length}`);
  actual.forEach((row, index) => vector(row, expected[index], `${label}[${index}]`, tolerance));
};
/** A loop must have something to look at, or it asserts nothing. */
const nonEmpty = (collection, expected, label) => {
  assertions += 1;
  assert(collection.length > 0, `${label}: the subject set is empty, so nothing was checked`);
  if (expected !== undefined) {
    assertions += 1;
    assert.equal(collection.length, expected, `${label}: saw ${collection.length}, expected ${expected}`);
  }
};
/** A call that must be refused. A silent default here would be a wrong answer. */
const refuses = (call, label) => {
  assertions += 1;
  assert.throws(call, RangeError, `${label}: should have been refused`);
  record('refused input');
};
const exp = value => globalThis.Math.exp(value);
const abs = value => globalThis.Math.abs(value);
const names = model => model.stateNames;

const weather = fixtures.weather;
const reports = fixtures.reports;

/* ============================================ §1 · the model and one path */

checkModel(weather);
checkModel(fixtures.constrained);
vector(weather.start, mechanisms.model.start, 'the declared initial row', 0);
grid(weather.transition, mechanisms.model.transition, 'the declared transition matrix', 0);
grid(weather.emission, mechanisms.model.emission, 'the declared emission matrix', 0);
assert.deepEqual(reports, mechanisms.observations, 'the declared four reports');
assert.deepEqual(names(weather), ['Rainy', 'Sunny'], 'the two state names');
assert.deepEqual(weather.symbolNames, ['Walk', 'Shop', 'Clean'], 'the three symbol names');
[weather.start, ...weather.transition, ...weather.emission].forEach((row, index) => {
  close(row.reduce((total, value) => total + value, 0), 1, `row ${index} is a distribution`, 1e-15);
  record('declared probability row');
});
record('section 1 model');

const enumerated = enumeratePaths(weather, reports);
nonEmpty(enumerated.paths, 16, 'the sixteen complete paths');
enumerated.paths.forEach((entry, index) => {
  assert.deepEqual(entry.path, mechanisms.enumerated_paths[index].states, `path ${index} order`);
  close(entry.joint, mechanisms.enumerated_paths[index].joint, `path ${index} joint mass`, 1e-12);
  record('enumerated path');
});
close(enumerated.evidence, exp(mechanisms.main.log_evidence), 'the paths add to the forward evidence', 1e-12);
close(enumerated.largest.joint, mechanisms.main.path_joint, 'the largest is the Viterbi path', 1e-12);
close(enumerated.remainingPosterior, 1 - mechanisms.main.path_posterior, 'and the rest is everything else', 1e-12);
// One path, factor by factor, against a second construction of the same product.
const manualJoint = (() => {
  const path = mechanisms.main.path;
  let value = weather.start[path[0]] * weather.emission[path[0]][reports[0]];
  for (let time = 1; time < path.length; time += 1) {
    value *= weather.transition[path[time - 1]][path[time]] * weather.emission[path[time]][reports[time]];
  }
  return value;
})();
close(manualJoint, pathJointOf(weather, mechanisms.main.path, reports),
  'the factor-by-factor product agrees with pathJointOf', 1e-15);
close(reports.length - 1, 3, 'four reports contain three transitions', 0);
record('section 1 paths');

/* =================================================== §3 · forward, filtering */

const forward = trellis(weather, reports, 'sum');
nonEmpty(forward.columns, 4, 'forward trellis columns');
forward.columns.forEach(column => {
  vector(column.cells.map(cell => cell.value), mechanisms.main.forward_log[column.time].map(exp),
    `forward column ${column.time}`, 1e-12);
  record('forward column');
});
close(forward.columnTotals[0], 0.3, 'the first report has probability .3', 1e-15);
close(forward.final, 0.00933936, 'the whole sequence has probability .00933936', 1e-12);
close(forward.columns[1].cells[0].value, 0.0552, 'the manuscript alpha_1(R)', 1e-15);
close(forward.columns[1].cells[1].value, 0.0486, 'the manuscript alpha_1(S)', 1e-15);
// The bracket the manuscript prints, rebuilt from the edge record.
const shopEdges = trellisEdges(weather, reports, 'sum')[0].edges.filter(edge => edge.to === 0);
nonEmpty(shopEdges, 2, 'incoming edges to Rainy at the Shop step');
close(shopEdges.reduce((total, edge) => total + edge.carried, 0), 0.138,
  'the bracket adds to .138 before the destination emission', 1e-15);
close(0.138 * weather.emission[0][1], 0.0552, 'and .138 times the Shop emission is .0552', 1e-15);
// The Rainy mass rises at the last step while the column total falls.
assert(forward.columns[3].cells[0].value > forward.columns[2].cells[0].value,
  'the Rainy forward mass rises at the last step');
assert(forward.columnTotals[3] < forward.columnTotals[2],
  'while the column total falls, so no cell has to shrink monotonically');
record('section 3 forward');

const main = infer(weather, reports);
grid(main.filtered, mechanisms.main.filtered, 'filtered rows', 1e-11);
main.filtered.forEach((row, time) => {
  close(row.reduce((total, value) => total + value, 0), 1, `filtered row ${time} is a distribution`, 1e-12);
  vector(row, forward.columns[time].cells.map(cell => cell.value / forward.columnTotals[time]),
    `filtered row ${time} is the normalised forward column`, 1e-11);
  record('filtered row');
});
vector(main.factors, mechanisms.scaled.factors, 'the retained scale factors', 1e-12);
close(main.factors.reduce((product, value) => product * value, 1), forward.final,
  'the scale factors multiply back to the evidence', 1e-12);
close(main.logEvidence, mechanisms.scaled.log_evidence, 'and their logs add to its logarithm', 1e-12);
vector(forwardScaled(weather, reports).filteredLast, mechanisms.scaled.filtered_last,
  'the final scaled belief', 1e-11);
record('section 3 scaling');

const forecast = predictNext(weather, infer(weather, [0]).filtered[0]);
vector(infer(weather, [0]).filtered[0], [0.2, 0.8], 'filtering after one Walk', 1e-14);
vector(forecast.nextState, [0.46, 0.54], 'the next-state prediction', 1e-14);
close(forecast.nextObservation[2], 0.284, 'the predicted probability of Clean next', 1e-14);
close(forecast.nextObservation[1], 0.346, 'and of Shop next', 1e-14);
close(forecast.collapsedObservation[1], 0.34, 'collapsing to the mode first gives .34 instead', 1e-14);
assert(forecast.mode === 1, 'and the mode after one Walk is Sunny');
assert(abs(forecast.nextObservation[1] - forecast.collapsedObservation[1]) > 1e-3,
  'the two forecasts really differ, so the contrast is not a rounding artefact');
close(forecast.nextObservation.reduce((total, value) => total + value, 0), 1,
  'the predicted observation row is a distribution', 1e-14);
record('section 3 forecast');

/* ============================================ §4 · backward and smoothing */

grid(main.smoothed, mechanisms.main.smoothed, 'smoothed rows', 1e-10);
main.smoothed.forEach((row, time) => {
  close(row.reduce((total, value) => total + value, 0), 1, `smoothed row ${time} is a distribution`, 1e-12);
  record('smoothed row');
});
vector(main.smoothed[3], main.filtered[3], 'the last smoothed row equals the last filtered row', 0);
assert(abs(main.smoothed[1][0] - main.filtered[1][0]) > 1e-3,
  'while the time-1 rows genuinely differ');
// The backward values the manuscript prints, recovered from the scaled rows.
const tail = main.factors.map((unused, time) =>
  main.factors.slice(time + 1).reduce((product, value) => product * value, 1));
close(main.backward[2][0] * tail[2], 0.38, 'beta_2(Rainy) is .38', 1e-12);
close(main.backward[2][1] * tail[2], 0.26, 'beta_2(Sunny) is .26', 1e-12);
grid(main.backward.map((row, time) => row.map(value => globalThis.Math.log(value * tail[time]))),
  mechanisms.main.backward_log, 'backward likelihoods', 1e-11);
nonEmpty(main.pair, 3, 'pair posteriors');
main.pair.forEach((block, time) => {
  grid(block, mechanisms.main.pair[time], `pair block ${time}`, 1e-10);
  close(block.flat().reduce((total, value) => total + value, 0), 1, `pair block ${time} sums to one`, 1e-11);
  vector(block.map(row => row.reduce((total, value) => total + value, 0)), main.smoothed[time],
    `pair block ${time} row margins are gamma(t)`, 1e-10);
  vector(block[0].map((unused, column) => block.reduce((total, row) => total + row[column], 0)),
    main.smoothed[time + 1], `pair block ${time} column margins are gamma(t+1)`, 1e-10);
  record('pair block');
});
record('section 4 smoothing');

const corrected = infer(weather, fixtures.correctedFinal);
grid(corrected.filtered.slice(0, 3), main.filtered.slice(0, 3),
  'a changed final report leaves the filtered prefix bit-identical', 0);
close(corrected.smoothed[1][0], 0.3976241066, 'while time-1 smoothing falls to .397624', 1e-9);
assert.deepEqual(corrected.path, [1, 1, 1, 1], 'and the whole best path becomes Sunny');
grid(corrected.smoothed, mechanisms.changed_future.smoothed, 'changed-future smoothed rows', 1e-10);
const missing = infer(weather, fixtures.missingMiddle);
const deleted = infer(weather, fixtures.deletedMiddle);
close(missing.smoothed[3][0], 0.802780059665355, 'a retained missing report gives final Rainy .802780', 1e-11);
close(deleted.smoothed[2][0], 0.7953204876130554, 'deleting the step gives .795320', 1e-11);
assert(abs(missing.smoothed[3][0] - deleted.smoothed[2][0]) > 1e-3,
  'the two really differ, so a missing step is not a deleted one');
assert(missing.smoothed.length === 4 && deleted.smoothed.length === 3,
  'and the retained version still has four time steps');
vector(localLikelihood(weather, MISSING), [1, 1], 'a missing report has emission likelihood one in every state', 0);
record('section 4 missing');

/* =========================================================== §5 · Viterbi */

const best = trellis(weather, reports, 'max');
best.columns.forEach(column => {
  vector(column.cells.map(cell => cell.value), mechanisms.main.viterbi_log[column.time].map(exp),
    `Viterbi column ${column.time}`, 1e-12);
  if (column.time > 0) {
    assert.deepEqual(column.cells.map(cell => cell.chosen), mechanisms.main.predecessor[column.time],
      `stored predecessors at time ${column.time}`);
  }
  record('Viterbi column');
});
const backtracked = backtrack(best);
assert.deepEqual(backtracked.path, mechanisms.main.path, 'the backtracked path');
assert.deepEqual(backtracked.path, [1, 1, 1, 0], 'which is Sunny Sunny Sunny Rainy');
close(backtracked.joint, 0.0031104, 'with joint mass .0031104', 1e-12);
close(main.pathPosterior, 0.33304209282006475, 'and posterior share .333042', 1e-12);
// The predecessor claim the manuscript makes, asserted on the edge record.
const walkEdges = trellisEdges(weather, reports, 'max')[1].edges.filter(edge => edge.to === 0);
nonEmpty(walkEdges, 2, 'incoming max-mode edges to Rainy at the third report');
close(walkEdges.find(edge => edge.from === 0).carried, 0.02688, 'Rainy carries .02688', 1e-15);
close(walkEdges.find(edge => edge.from === 1).carried, 0.01728, 'Sunny carries .01728', 1e-15);
assert(walkEdges.find(edge => edge.from === 0).chosen, 'so the chosen predecessor is Rainy');
assert(best.columns[1].cells[1].value > best.columns[1].cells[0].value,
  'even though the Sunny cell before it is the larger of the two');
assert(!backtracked.path.includes(0) || backtracked.path[2] === 1,
  'and the final path does not pass through that Rainy cell');
record('section 5 Viterbi');

const share = pathShare(weather, reports);
close(share.posterior + share.remaining, 1, 'the two posterior shares add to one', 1e-15);
assert(share.conserved, 'and the model says so itself');
assert(share.remaining > 0.6, 'about two thirds of the posterior belongs to other paths');
const alteredShare = pathShare(withRainyEmission(fixtures.changedRainyEmission), reports);
close(alteredShare.joint, share.joint, 'a changed Rainy row leaves the best path joint mass exactly alone', 0);
close(alteredShare.posterior, 9 / 35, 'while its posterior becomes 9/35', 1e-12);
assert(abs(alteredShare.evidence - share.evidence) > 1e-4, 'because the evidence moved');
assert.deepEqual(alteredShare.path, share.path, 'and the best path itself is unchanged');
record('section 5 changed emission');

const constrained = fixtures.constrained;
const constrainedReports = [0, 0];
const pointwise = infer(constrained, constrainedReports);
vector(pointwise.smoothed[0], [0.4, 0.35, 0.25], 'first-time marginals', 1e-12);
vector(pointwise.smoothed[1], [0.6, 0.2, 0.2], 'second-time marginals', 1e-12);
assert.deepEqual(pointwise.marginalModes, [0, 0], 'the pointwise modes are A then A');
assert.deepEqual(pointwise.path, [1, 0], 'and Viterbi returns B then A');
close(pointwise.pathJoint, 0.35, 'at joint probability .35', 1e-12);
close(pointwise.modesJoint, 0, 'while the pointwise route has joint probability exactly zero', 0);
assert(pointwise.modesLegal === false, 'because one of its edges does not exist');
close(pointwise.modesExpectedCorrect, 1.0, 'the pointwise modes expect 1.0 correct positions', 1e-12);
close(pointwise.pathExpectedCorrect, 0.95, 'and B to A expects .95', 1e-12);
assert(pointwise.modesExpectedCorrect > pointwise.pathExpectedCorrect,
  'so the impossible route wins the expected-position count');
close(pointwise.evidence, 1, 'a certain single symbol carries no evidence at all', 1e-12);
// Brute force over legal paths, which uses no recurrence, agrees with the DP.
const brute = bestLegalPath(constrained, constrainedReports);
assert.deepEqual(brute.path, pointwise.path, 'brute force over legal paths agrees with the dynamic program');
close(brute.joint, pointwise.pathJoint, 'on the value too', 1e-15);
assert.equal(brute.legalPaths, 4, 'and finds exactly four legal two-step paths');
// The number the figure heading and the prose both state, pinned to the model
// so neither can drift from it. The heading said three for one revision.
const permittedEdges = constrained.transition.flat().filter(value => value > 0).length;
assert.equal(permittedEdges, 4, 'the constrained graph permits exactly four transitions');
assert.equal(constrained.transition.flat().length - permittedEdges, 5,
  'and forbids the other five');
// The joint mass table of the manuscript, from the model.
const jointTable = [0, 1, 2].map(first => [0, 1, 2].map(second =>
  pathJointOf(constrained, [first, second], constrainedReports)));
grid(jointTable, [[0, 0.2, 0.2], [0.35, 0, 0], [0.25, 0, 0]], 'the joint mass table', 1e-12);
vector(jointTable.map(row => row.reduce((total, value) => total + value, 0)), constrained.start,
  'whose row sums are the initial probabilities', 1e-12);
record('section 5 pointwise');

const changedPrior = infer(withStart(constrained, fixtures.changedConstrainedStart), constrainedReports);
assert.deepEqual(changedPrior.marginalModes, [1, 0], 'under the changed prior the modes are B then A');
assert.deepEqual(changedPrior.path, [1, 0], 'and so is Viterbi');
close(changedPrior.pathJoint, 0.55, 'at joint probability .55', 1e-12);
close(changedPrior.evidence, 1, 'with the evidence still exactly one', 1e-12);
assert(changedPrior.modesLegal, 'and the modes are now a path the model permits');
record('section 5 changed prior');

/* ==================================================== §6 · counts and EM */

const split = expectedCounts(weather, fixtures.splitRecordings);
vector(split.initial, mechanisms.split_counts[0], 'split initial counts', 1e-10);
grid(split.edge, mechanisms.split_counts[1], 'split edge counts', 1e-10);
grid(split.symbol, mechanisms.split_counts[2], 'split symbol counts', 1e-10);
close(split.logLikelihood, mechanisms.split_counts[3], 'split training log likelihood', 1e-11);
close(split.totals.startMass, 2, 'two recordings contribute two starts', 1e-10);
close(split.totals.edgeMass, 2, 'and two within-recording transitions', 1e-10);
close(split.totals.symbolMass, 4, 'and four emissions', 1e-10);
const joinedCounts = expectedCounts(weather, fixtures.joinedRecording);
close(joinedCounts.totals.startMass, 1, 'concatenating leaves one start', 1e-10);
close(joinedCounts.totals.edgeMass, 3, 'and three transitions', 1e-10);
assert(abs(joinedCounts.edge[0][0] - split.edge[0][0]) > 1e-3,
  'and genuinely changes the expected Rainy to Rainy count');
grid(joinedCounts.edge, mechanisms.joined_counts[1], 'joined edge counts', 1e-10);
const flow = countFlow(weather, fixtures.splitRecordings);
assert(flow.conserved, 'the drawn count flow conserves its totals');
assert(countFlow(weather, fixtures.joinedRecording).conserved, 'and so does the concatenated one');
record('section 6 counts');

const updated = emStep(weather, fixtures.splitRecordings);
vector(updated.model.start, mechanisms.split_update.start, 'the updated initial row', 1e-10);
grid(updated.model.transition, mechanisms.split_update.transition, 'the updated transition rows', 1e-10);
grid(updated.model.emission, mechanisms.split_update.emission, 'the updated emission rows', 1e-10);
close(updated.logLikelihoodBefore, -4.728043153397216, 'the objective before the update', 1e-11);
close(updated.logLikelihoodAfter, -3.529108143179231, 'and after it', 1e-11);
assert(updated.logLikelihoodAfter > updated.logLikelihoodBefore, 'one exact update did not decrease it');
checkModel(updated.model);
const doubled = emStep(weather, [...fixtures.splitRecordings, ...fixtures.splitRecordings]);
vector(doubled.model.start, updated.model.start, 'duplicating the dataset leaves the update alone', 1e-11);
grid(doubled.model.transition, updated.model.transition, 'on transitions too', 1e-11);
grid(doubled.model.emission, updated.model.emission, 'and on emissions', 1e-11);
vector(doubled.counts.initial, split.initial.map(value => 2 * value), 'while doubling the counts', 1e-10);
// normalizeCounts keeps an unvisited row rather than dividing by zero. This is
// checked BEFORE the length-one EM step, because that step's own model
// validation would otherwise stop the run first and report a different guard.
const normalised = normalizeCounts([[0, 0], [1, 3]], [[0.5, 0.5], [0.9, 0.1]]);
assert(normalised.flat().every(Number.isFinite),
  'an unvisited count row is retained rather than divided by zero into NaN');
grid(normalised, [[0.5, 0.5], [0.25, 0.75]],
  'an unvisited count row is retained exactly, and a visited one is normalised', 1e-15);
const singleton = emStep(weather, [[0]]);
grid(singleton.model.transition, weather.transition,
  'a length-one recording has no transition, so its rows are retained exactly', 0);
close(recordingTotals([[0]]).transitions, 0, 'and it contributes zero transitions', 0);
record('section 6 EM step');

/* ============================================ §7 · numerical representation */

const rare = repeatedProduct(fixtures.rareFactor, fixtures.rareCount);
assert(rare.underflowed && rare.ordinary === 0, 'point zero one to the four hundredth underflows to exactly zero');
close(rare.logValue, -1842.0680743952303, 'while its logarithm is manageable', 1e-12);
close(rare.logValue, rare.scaledLogValue, 'and the scaled route agrees with it', 1e-10);
const representable = repeatedProduct(fixtures.representableFactor, fixtures.representableCount);
assert(!representable.underflowed && representable.ordinary > 0, 'point three to the hundredth is representable');
close(representable.ordinary, 5.153775207320094e-53, 'at about 5.15 times ten to the minus 53', 1e-13);
close(logSumExp([0, 0]), globalThis.Math.log(2), 'log-sum-exp of two zeros is log 2', 1e-15);
close(logSumExp([-1000, -1000]), -1000 + globalThis.Math.log(2), 'and it is stable far from zero', 1e-12);
assert(logSumExp([-Infinity, -Infinity]) === -Infinity,
  'an all-negative-infinity row is a zero sum, said rather than computed into a NaN');
assert(!Number.isNaN(logSumExp([-Infinity, -Infinity])), 'and never a NaN');
const impossible = infer(fixtures.impossibleModel, fixtures.impossibleObservations);
assert(impossible.impossible && impossible.evidence === 0, 'an impossible sequence has exactly zero evidence');
assert(impossible.smoothed === null && impossible.pathPosterior === null,
  'and no posterior at all, rather than a normalised zero');
assert(typeof impossible.reason === 'string' && impossible.reason.length > 20, 'and it says why');
record('section 7 numerics');

/* ================================================== §9 · duration and size */

mechanisms.duration.forEach(entry => {
  const model = durationModel(entry.stay);
  vector(model.probabilities, entry.probabilities_d1_to_8, `duration bars at a = ${entry.stay}`, 1e-12);
  close(model.mean, entry.mean, `mean dwell time at a = ${entry.stay}`, 1e-12);
  assert(model.conserved, `bars plus tail conserve mass at a = ${entry.stay}`);
  record('declared duration setting');
});
const absorbing = durationModel(1);
assert(absorbing.absorbing && absorbing.mean === null,
  'an absorbing state has no finite mean rather than a very large one');
close(absorbing.exitProbability, 0, 'and it never leaves', 0);
close(absorbing.tailMass, 1, 'so all of its mass is beyond eight steps', 0);
const practice = durationModel(fixtures.durationPractice);
close(practice.mean, 5, 'a = .8 gives mean duration five', 1e-15);
close(practice.probabilities[2], 0.128, 'and P(D = 3) = .128', 1e-15);
close(practice.exitProbability, 0.2, 'with a constant exit probability of .2', 1e-15);
close(parameterCount(2, 3), 7, 'the two-state three-symbol model has seven free parameters', 0);
close(parameterCount(3, 3), 14, 'a three-state three-symbol model has fourteen: 2 + 6 + 6', 0);
close(parameterCount(1, 1), 0, 'and a one-state one-symbol model has none', 0);
close(forward.transitionContributions, 12, 'the four-step two-state trellis evaluates twelve contributions', 0);
close(trellis(constrained, constrainedReports, 'sum').transitionContributions, 9,
  'and the three-state two-step one evaluates nine', 0);
record('section 9 duration and size');

/* ===================================== what the figures actually draw */

// Edge widths: a declared zero width below the positive range, then strictly
// increasing across it. Returning null at zero left the drawing with no
// attribute, so a zero-mass edge painted at the SVG initial 1px while the
// smallest positive share painted at 0.9 - this lesson's own distinction,
// inverted on screen.
const shares = Array.from({ length: 201 }, (unused, index) => index / 200);
nonEmpty(shares, 201, 'edge-width grid');
let previousWidth = -Infinity;
shares.forEach(value => {
  const width = edgeWidth(value);
  ok(Number.isFinite(width), `edge width at share ${value} is a number, so the drawing always sets an attribute`);
  if (value === 0) {
    close(width, edgeWidthRange.zero, 'a zero share takes the declared zero width', 0);
    ok(width < edgeWidthRange.minimum, 'which is thinner than any positive share');
  } else {
    ok(width >= edgeWidthRange.minimum && width <= edgeWidthRange.maximum,
      `edge width at share ${value} stays inside its declared range`);
    ok(width > previousWidth, `edge width is strictly increasing at share ${value}`);
  }
  previousWidth = width;
  record('edge-width grid point');
});
close(edgeWidth(1), edgeWidthRange.maximum, 'a full share gets the widest stroke', 1e-15);
refuses(() => edgeWidth(1.2), 'a share above one');
refuses(() => edgeWidth(-0.1), 'a negative share');
// Every drawn edge's share, on every fixture, in both modes.
/** An independent forward or max recursion, written here rather than imported.
 *
 * The per-edge sweep below used to compare `edge.carried` against
 * `edge.previousValue * edge.transition` — the same IEEE multiply of the same
 * two stored operands, compared with itself, ninety times. It now compares the
 * drawing against arithmetic this file performs on the model's own matrices.
 */
function recomputeColumns(model, observations, mode) {
  const local = value => (value === MISSING
    ? model.start.map(() => 1)
    : model.emission.map(row => row[value]));
  const states = model.start.length;
  let column = model.start.map((prior, state) => prior * local(observations[0])[state]);
  const columns = [column];
  for (let time = 1; time < observations.length; time += 1) {
    const emission = local(observations[time]);
    const previous = column;
    column = Array.from({ length: states }, (unused, destination) => {
      const carried = previous.map((mass, origin) => mass * model.transition[origin][destination]);
      const combined = mode === 'sum'
        ? carried.reduce((total, value) => total + value, 0)
        : carried.reduce((best, value) => (value > best ? value : best), -Infinity);
      return combined * emission[destination];
    });
    columns.push(column);
  }
  return columns;
}

const drawnFixtures = [
  { model: weather, observations: reports, label: 'the four reports' },
  { model: weather, observations: fixtures.correctedFinal, label: 'the corrected reports' },
  { model: weather, observations: fixtures.missingMiddle, label: 'the missing middle' },
  { model: constrained, observations: constrainedReports, label: 'the constrained graph' },
];
nonEmpty(drawnFixtures, 4, 'drawn fixtures');
let drawnEdges = 0;
drawnFixtures.forEach(entry => {
  ['sum', 'max'].forEach(mode => {
    const rows = trellisEdges(entry.model, entry.observations, mode);
    const built = trellis(entry.model, entry.observations, mode);
    const recomputed = recomputeColumns(entry.model, entry.observations, mode);
    built.columns.forEach((column, time) => {
      column.cells.forEach(cell => {
        close(cell.value, recomputed[time][cell.state],
          `${entry.label}: cell ${time}/${cell.state} matches the independent recursion in ${mode} mode`, 1e-12);
      });
    });
    nonEmpty(rows, entry.observations.length - 1, `${entry.label} edge rows in ${mode} mode`);
    rows.forEach(row => {
      built.columns[row.time].cells.forEach(cell => {
        const incoming = row.edges.filter(edge => edge.to === cell.state);
        nonEmpty(incoming, entry.model.start.length, `incoming edges to state ${cell.state} at ${row.time}`);
        const total = incoming.reduce((sum, edge) => sum + edge.carried, 0);
        if (mode === 'sum') {
          const independentTotal = incoming.reduce(
            (sum, edge) => sum + recomputed[row.time - 1][edge.from] * entry.model.transition[edge.from][edge.to], 0);
          close(cell.aggregate, independentTotal,
            `${entry.label}: the destination aggregate is the independently summed contributions`, 1e-12);
          close(total, cell.aggregate, `${entry.label}: summed edges equal the destination aggregate`, 1e-12);
          close(incoming.reduce((sum, edge) => sum + edge.share, 0), 1,
            `${entry.label}: the incoming shares add to one at ${row.time}/${cell.state}`, 1e-12);
        } else {
          const chosen = incoming.filter(edge => edge.chosen);
          assert.equal(chosen.length, 1, `${entry.label}: exactly one predecessor is drawn solid`);
          close(chosen[0].carried, cell.aggregate,
            `${entry.label}: the solid edge carries the destination's own maximum`, 1e-15);
          incoming.forEach(edge => {
            assert(edge.carried <= chosen[0].carried + 1e-15,
              `${entry.label}: no rejected edge carries more than the chosen one`);
          });
        }
        incoming.forEach(edge => {
          // Expectations from the model's own matrices and the independent
          // recursion above, never from the edge record being checked.
          const sourceMass = recomputed[row.time - 1][edge.from];
          const transition = entry.model.transition[edge.from][edge.to];
          close(edge.transition, transition, `${entry.label}: the edge's transition is the model's`, 0);
          close(edge.previousValue, sourceMass, `${entry.label}: and its source mass is the previous cell`, 1e-15);
          close(edge.carried, sourceMass * transition,
            `${entry.label}: an edge carries previous cell times transition`, 1e-15);
          equals(edge.forbidden, transition === 0,
            `${entry.label}: an edge is drawn forbidden exactly when the model has no such transition`);
          equals(edge.carriesNothing, edge.carried === 0 && transition !== 0,
            `${entry.label}: and drawn empty exactly when a permitted edge carries nothing`);
          drawnEdges += 1;
        });
      });
    });
    record('drawn trellis mode');
  });
});
equals(drawnEdges, 90, `exactly ninety drawn edges were inspected, not ${drawnEdges}`);
// The property S4 violated: an edge that carries nothing must never be painted
// wider than one that carries a little. Checked on a model where the case is
// reachable, which is the one the reviewer reached through the interface.
const emptyCaseModel = withRainyEmission([0, 0.5, 0.5]);
const emptyCaseEdges = trellisEdges(emptyCaseModel, reports, 'sum').flatMap(row => row.edges);
const emptyEdges = emptyCaseEdges.filter(edge => edge.carriesNothing);
const carryingEdges = emptyCaseEdges.filter(edge => edge.carried > 0);
nonEmpty(emptyEdges, 4, 'edges that exist and carry nothing');
nonEmpty(carryingEdges, undefined, 'edges that carry something');
emptyEdges.forEach(edge => {
  close(edge.width, edgeWidthRange.zero, 'an edge carrying nothing takes the declared zero width', 0);
  carryingEdges.forEach(other => {
    ok(edge.width < other.width,
      `an edge carrying nothing (${edge.width}) is thinner than one carrying ${other.carried} (${other.width})`);
  });
});
// And the function is monotone across its whole domain, zero included.
const widthGrid = Array.from({ length: 1001 }, (unused, index) => edgeWidth(index / 1000));
nonEmpty(widthGrid, 1001, 'edge-width monotonicity grid');
widthGrid.forEach((width, index) => {
  if (index === 0) return;
  ok(width >= widthGrid[index - 1], `edge width does not decrease at share ${index / 1000}`);
});
ok(edgeWidth(0) < edgeWidth(1e-12), 'and a zero share is strictly thinner than the smallest positive one');
record('drawn edges');

// Bar heights come from the model, share one axis, and refuse an overflow.
close(barHeight(0.3, 0.6), 0.5, 'a bar at half its axis is drawn at half height', 1e-15);
close(barHeight(0, 0.6), 0, 'a zero bar has zero height', 0);
close(barHeight(0.6, 0.6), 1, 'a full bar fills the axis', 1e-15);
refuses(() => barHeight(0.7, 0.6), 'a bar taller than its own axis');
refuses(() => barHeight(-0.1, 0.6), 'a negative bar');
refuses(() => barHeight(0.3, 0), 'an axis with no height');
const bars = durationBars(fixtures.durationDefault);
nonEmpty(bars.bars, limits.durationBars, 'duration bars');
bars.bars.forEach(bar => {
  close(bar.hazard, bars.exitProbability, `bar ${bar.duration} carries the same constant hazard`, 1e-15);
  record('duration bar');
});
assert(bars.tailMass > 0, 'the tail bar carries positive mass, so the eight bars do not sum to one');
close(bars.probabilities.reduce((total, value) => total + value, 0) + bars.tailMass, 1,
  'bars plus tail conserve the whole distribution', 1e-12);
record('drawn bars');

// The belief track the paired bars draw.
const track = beliefTrack(weather, reports, 0);
nonEmpty(track.rows, 4, 'belief track rows');
track.rows.forEach((row, time) => {
  close(row.filtered, main.filtered[time][0], `belief track filtered ${time}`, 0);
  close(row.smoothed, main.smoothed[time][0], `belief track smoothed ${time}`, 0);
  close(row.difference, row.smoothed - row.filtered, `belief track difference ${time}`, 0);
  assert(row.filtered >= 0 && row.filtered <= 1 && row.smoothed >= 0 && row.smoothed <= 1,
    `belief track row ${time} stays inside the drawn 0 to 1 scale`);
  record('belief track row');
});
close(track.rows[3].difference, 0, 'and the last difference is exactly zero', 0);

// The EM plot's two windows, and why there are two.
const finals = [...emTrack.fits.map(fit => fit.logLikelihood[40]), emTrack.generatingLogLikelihood,
  emTrack.uniformStart.logLikelihood[1]];
nonEmpty(finals, 5, 'plotted final objectives');
finals.forEach(value => {
  assert(insideWindow(value, objectiveWindows.detail),
    `the detail window contains the final value ${value}, so nothing it claims to draw is clipped`);
  record('final objective inside the detail window');
});
assert(separationInPixels(finals, objectiveWindows.detail) > 3,
  'the detail window separates the closest pair of final values by more than three pixels');
assert(separationInPixels(finals, objectiveWindows.overview) < 1,
  'while the overview separates them by under one pixel, which is why there are two panels');
emTrack.fits.forEach(fit => {
  assert(insideWindow(fit.logLikelihood[0], objectiveWindows.overview),
    `the overview contains seed ${fit.seed}'s initial score`);
  assert(!insideWindow(fit.logLikelihood[0], objectiveWindows.detail),
    `and the detail window deliberately excludes it`);
  record('plotted initial objective');
});
record('plot windows');

/* ================================ the rules the investigations grade with */

// `direction`, over a grid that includes both sides of its own tolerance.
const directionGrid = [0, 1e-16, 1e-13, 1e-12, 1e-11, 1e-9, 1e-3, 0.5, 1];
nonEmpty(directionGrid, 9, 'direction grid');
directionGrid.forEach(base => {
  assert.equal(direction(base, base), 'unchanged', `an identical pair at ${base} is unchanged`);
  assert.equal(direction(base, base + 1), 'rises', `a clear rise at ${base}`);
  assert.equal(direction(base + 1, base), 'falls', `a clear fall at ${base}`);
  record('direction grid point');
});
assert.equal(direction(0.5, 0.5 + 1e-15), 'unchanged', 'a sub-tolerance move is reported as no move');
assert.equal(direction(0.5, 0.5 + 1e-9), 'rises', 'and a move above tolerance is reported');
assert.equal(direction(null, 0.5), 'undefined', 'a missing earlier value has no direction');
assert.equal(direction(0.5, null), 'undefined', 'nor a missing later one');
assert(limits.moveTolerance <= 10 ** -limits.moveDigits,
  'the tolerance is no coarser than the precision the verdict prints, so "does not move" is readable');
refuses(() => direction(NaN, 0.5), 'a non-finite earlier value');
record('direction rule');

// `pathRankOutcome`, over every path of four models including degenerate ones.
const uniformModel = {
  stateNames: ['U0', 'U1'], symbolNames: ['a', 'b'],
  start: [0.5, 0.5], transition: [[0.5, 0.5], [0.5, 0.5]], emission: [[0.5, 0.5], [0.5, 0.5]],
};
const rankFixtures = [
  { model: weather, observations: reports, label: 'the weather model' },
  { model: withRainyEmission(fixtures.changedRainyEmission), observations: reports, label: 'the changed row' },
  { model: uniformModel, observations: [0, 1, 0, 1], label: 'a model where every path ties' },
  { model: constrained, observations: constrainedReports, label: 'the constrained graph' },
  /* The fixture that distinguishes the two possible orders inside the rule. If
     zero is tested after largest, every path of a model where EVERY path is
     impossible gets reported as "the most likely one", and no fixture whose
     maximum is positive can tell. */
  { model: fixtures.impossibleModel, observations: fixtures.impossibleObservations,
    label: 'a model where every path is impossible' },
];
nonEmpty(rankFixtures, 5, 'rank fixtures');
let rankChecks = 0;
rankFixtures.forEach(entry => {
  const all = enumeratePaths(entry.model, entry.observations);
  const top = all.largest.joint;
  nonEmpty(all.paths, entry.model.start.length ** entry.observations.length, `${entry.label} paths`);
  all.paths.forEach(path => {
    const outcome = pathRankOutcome(entry.model, path.path, entry.observations);
    // The property, not the example: zero wins over largest, and largest means
    // exactly "nothing beats it".
    const expected = path.joint === 0 ? 'zero' : (path.joint === top ? 'largest' : 'smaller');
    equals(outcome, expected, `${entry.label}: rank of ${path.path.join('')}`);
    rankChecks += 1;
  });
  record('rank fixture');
});
const uniformRanks = enumeratePaths(uniformModel, [0, 1, 0, 1]).paths
  .map(path => pathRankOutcome(uniformModel, path.path, [0, 1, 0, 1]));
assert(uniformRanks.every(outcome => outcome === 'largest'),
  'when every path ties, every one of them is reported as largest');
const allImpossibleRanks = enumeratePaths(fixtures.impossibleModel, fixtures.impossibleObservations).paths
  .map(path => pathRankOutcome(fixtures.impossibleModel, path.path, fixtures.impossibleObservations));
nonEmpty(allImpossibleRanks, 2, 'ranks under a model where every path is impossible');
assert(allImpossibleRanks.every(outcome => outcome === 'zero'),
  'when every path is impossible, every one of them is reported as impossible and none as the most likely');
const constrainedRanks = enumeratePaths(constrained, constrainedReports).paths
  .map(path => pathRankOutcome(constrained, path.path, constrainedReports));
assert.equal(constrainedRanks.filter(outcome => outcome === 'zero').length, 5,
  'five of the nine two-step selections are impossible and are reported as zero');
assert.equal(constrainedRanks.filter(outcome => outcome === 'largest').length, 1,
  'and exactly one is the largest');
record('path rank rule');

// `beliefMoveOutcomes`, over all 256 four-step recordings at every query time,
// plus the teaching property that filtering at q depends only on the prefix.
const symbols = [0, 1, 2, MISSING];
const allSequences = [];
symbols.forEach(a => symbols.forEach(b => symbols.forEach(c => symbols.forEach(d => {
  allSequences.push([a, b, c, d]);
}))));
nonEmpty(allSequences, 256, 'every four-step recording');
let moveChecks = 0;
let prefixChecks = 0;
allSequences.forEach(sequence => {
  const result = infer(weather, sequence);
  for (let query = 0; query < 4; query += 1) {
    // The property the section teaches: filtering at q is a function of the
    // prefix alone, so truncating everything after q cannot move it at all.
    const prefix = infer(weather, sequence.slice(0, query + 1));
    vector(result.filtered[query], prefix.filtered[query],
      `filtering at ${query} depends only on the prefix of ${sequence.join(',')}`, 0);
    prefixChecks += 1;
    const moved = beliefMoveOutcomes(weather, reports, weather, sequence, query);
    const before = infer(weather, reports);
    const expectedFiltering = direction(before.filtered[globalThis.Math.min(query, 3)][0], result.filtered[query][0]);
    const expectedSmoothing = direction(before.smoothed[globalThis.Math.min(query, 3)][0], result.smoothed[query][0]);
    assert.equal(moved.outcomes.filtering, expectedFiltering,
      `filtering verdict for ${sequence.join(',')} at ${query}`);
    assert.equal(moved.outcomes.smoothing, expectedSmoothing,
      `smoothing verdict for ${sequence.join(',')} at ${query}`);
    // The verdict says "unchanged" exactly when the printed value did not move.
    const printed = value => value.toFixed(limits.moveDigits);
    if (moved.outcomes.filtering === 'unchanged') {
      assert.equal(printed(moved.before.filtered), printed(moved.after.filtered),
        `an unchanged filtering verdict shows two identical printed values for ${sequence.join(',')}`);
    } else {
      assert.notEqual(printed(moved.before.filtered), printed(moved.after.filtered),
        `a moved filtering verdict shows two different printed values for ${sequence.join(',')}`);
    }
    moveChecks += 1;
  }
});
// The declared null, exactly: changing only the last report holds filtering at
// every earlier time, for all four replacements.
symbols.forEach(replacement => {
  const changed = [...reports.slice(0, 3), replacement];
  [0, 1, 2].forEach(query => {
    const moved = beliefMoveOutcomes(weather, reports, weather, changed, query);
    assert.equal(moved.outcomes.filtering, 'unchanged',
      `replacing only the last report holds filtering at ${query}`);
    record('changed-future filtering null');
  });
});
// And the smoothed value genuinely moves for at least one replacement, so the
// null is not holding because nothing ever changes.
assert(symbols.some(replacement =>
  beliefMoveOutcomes(weather, reports, weather, [...reports.slice(0, 3), replacement], 1)
    .outcomes.smoothing !== 'unchanged'),
'while smoothing at time 1 does move for at least one replacement');
record('belief-move rule');

// `legalPathGuarantee`, over every prior on a twentieths grid of the simplex.
const priors = [];
for (let first = 0; first <= 20; first += 1) {
  for (let second = 0; second + first <= 20; second += 1) {
    priors.push([first / 20, second / 20, (20 - first - second) / 20]);
  }
}
nonEmpty(priors, 231, 'priors on a twentieths grid of the probability simplex on three states');
let guaranteeYes = 0;
let guaranteeNo = 0;
priors.forEach(prior => {
  const model = withStart(constrained, prior);
  const verdict = legalPathGuarantee(model, constrainedReports).outcome;
  const result = infer(model, constrainedReports);
  const brute = bestLegalPath(model, constrainedReports);
  const modesJoint = pathJointOf(model, result.marginalModes, constrainedReports);
  const expected = pathLegal(model, result.marginalModes) && modesJoint === brute.joint ? 'yes' : 'no';
  equals(verdict, expected, `guarantee verdict at prior ${prior.join(',')}`);
  if (verdict === 'yes') guaranteeYes += 1;
  else guaranteeNo += 1;
});
assert(guaranteeYes > 0 && guaranteeNo > 0,
  `both verdicts occur on the grid (${guaranteeYes} yes, ${guaranteeNo} no), so neither is vacuous`);
assert.equal(legalPathGuarantee(constrained, constrainedReports).outcome, 'no',
  'the declared prior gives no');
assert.equal(legalPathGuarantee(withStart(constrained, fixtures.changedConstrainedStart), constrainedReports).outcome,
  'yes', 'and the changed prior gives yes');
assert.equal(legalPathGuarantee(constrained, constrainedReports).reason, 'the modes are not a path',
  'and it says which of the two ways it is a no');
record('legal-path guarantee rule');

// `gradeLegalPathConstruction`, over every selection at both declared priors.
let solvedSelections = 0;
[constrained.start, fixtures.changedConstrainedStart].forEach(prior => {
  const model = withStart(constrained, prior);
  [0, 1, 2].forEach(first => [0, 1, 2].forEach(second => {
    const verdict = gradeLegalPathConstruction(model, [first, second], constrainedReports,
      fixtures.legalPathTarget);
    assert.equal(verdict.legal, pathLegal(model, [first, second]),
      `legality of ${first}${second} at prior ${prior.join(',')}`);
    close(verdict.joint, pathJointOf(model, [first, second], constrainedReports),
      `joint of ${first}${second}`, 0);
    // The expectation is built from the model, not from the verdict's own fields.
    equals(verdict.solved,
      pathLegal(model, [first, second])
        && pathJointOf(model, [first, second], constrainedReports) >= fixtures.legalPathTarget,
      `solved verdict for ${first}${second}`);
    if (verdict.solved) solvedSelections += 1;
    record('legal-path construction selection');
  }));
});
assert.equal(solvedSelections, 2, 'exactly one selection solves it at each of the two declared priors');
assert(gradeLegalPathConstruction(constrained, [1, 0], constrainedReports, fixtures.legalPathTarget).solved,
  'and that selection is B to A');
assert(!gradeLegalPathConstruction(constrained, [0, 0], constrainedReports, fixtures.legalPathTarget).solved,
  'while the pointwise modes do not solve it');
record('legal-path construction rule');

// `recordingTotals` and `gradeBoundaryConstruction`, over every composition of
// up to eight reports into up to four recordings, under three symbol patterns.
const compositions = [];
const buildCompositions = (remaining, parts) => {
  if (remaining === 0 && parts.length > 0) { compositions.push([...parts]); return; }
  if (parts.length >= limits.maximumSequences) return;
  for (let size = 1; size <= remaining; size += 1) {
    parts.push(size);
    buildCompositions(remaining - size, parts);
    parts.pop();
  }
};
for (let total = 1; total <= 8; total += 1) buildCompositions(total, []);
nonEmpty(compositions, undefined, 'recording compositions');
const patterns = [
  { name: 'all Walk', symbol: () => 0 },
  { name: 'all missing', symbol: () => MISSING },
  { name: 'alternating Walk and missing', symbol: index => (index % 2 === 0 ? 0 : MISSING) },
];
let boundaryChecks = 0;
let solvedBoundaries = 0;
compositions.forEach(lengths => {
  patterns.forEach(pattern => {
    let cursor = 0;
    const recordings = lengths.map(size => Array.from({ length: size }, () => {
      const value = pattern.symbol(cursor);
      cursor += 1;
      return value;
    }));
    const totals = recordingTotals(recordings);
    assert.equal(totals.starts, lengths.length, `starts for ${lengths.join('+')} under ${pattern.name}`);
    assert.equal(totals.transitions, lengths.reduce((sum, size) => sum + size - 1, 0),
      `transitions for ${lengths.join('+')}`);
    assert.equal(totals.emissions, recordings.flat().filter(value => value !== MISSING).length,
      `emissions for ${lengths.join('+')} under ${pattern.name}`);
    const verdict = gradeBoundaryConstruction(recordings, [2, 2]);
    /* The property, written out rather than read off the rule: the declared
       task needs two sessions of two AND four observed reports. Marking a
       report missing keeps the structure and removes an emission, which is a
       change to the data rather than to the boundary, so it must fail. */
    const expectedSolved = lengths.length === 2 && lengths[0] === 2 && lengths[1] === 2
      && recordings.flat().filter(value => value !== MISSING).length === 4;
    assert.equal(verdict.solved, expectedSolved,
      `boundary verdict for ${lengths.join('+')} under ${pattern.name}`);
    if (verdict.solved) solvedBoundaries += 1;
    boundaryChecks += 1;
  });
});
equals(boundaryChecks, compositions.length * patterns.length,
  `every composition was exercised under every symbol pattern (${boundaryChecks})`);
assert.equal(solvedBoundaries, 1,
  'exactly one of the exercised setups solves it: two sessions of two with all four reports observed');
// The degenerate neighbour: the right structure with a report marked missing.
const missingReport = gradeBoundaryConstruction([[0, MISSING], [0, 2]], [2, 2]);
assert(missingReport.structureMatches && !missingReport.totalsMatch && !missingReport.solved,
  'the right structure with one report missing keeps its starts and transitions, loses an emission, and fails');
assert.equal(missingReport.totals.transitions, 2, 'because a missing report keeps its time step and its transition');
assert.equal(missingReport.totals.emissions, 3, 'while contributing no emission of its own');
// The discriminating case the task exists for.
const oneAndThree = gradeBoundaryConstruction([[0], [1, 0, 2]], [2, 2]);
assert(oneAndThree.totalsMatch && !oneAndThree.structureMatches && !oneAndThree.solved,
  'a one-and-three split reaches the same three totals and is still rejected');
const twoAndTwo = gradeBoundaryConstruction([[0, 1], [0, 2]], [2, 2]);
assert(twoAndTwo.totalsMatch && twoAndTwo.structureMatches && twoAndTwo.solved,
  'while the declared split is accepted');
refuses(() => recordingTotals([]), 'an empty recording set');
refuses(() => recordingTotals([[]]), 'a recording with no time steps');
record('boundary rules');

// `gradeSmoothedConstruction`, over every single-report replacement.
let smoothedSolutions = 0;
symbols.forEach(replacement => {
  const observations = [...reports.slice(0, 3), replacement];
  const verdict = gradeSmoothedConstruction({ observations, modelEdited: false }, reports,
    fixtures.smoothedTarget, 1);
  assert(verdict.prefixKept, `replacing only the last report keeps the prefix (${replacement})`);
  assert(verdict.held, `and holds time-1 filtering (${replacement})`);
  assert.equal(verdict.solved, verdict.smoothed < fixtures.smoothedTarget,
    `the verdict for replacement ${replacement} turns only on the smoothed threshold`);
  if (verdict.solved) smoothedSolutions += 1;
  record('smoothed construction replacement');
});
assert.equal(smoothedSolutions, 1, 'exactly one replacement solves it');
assert(gradeSmoothedConstruction({ observations: fixtures.correctedFinal, modelEdited: false },
  reports, fixtures.smoothedTarget, 1).solved, 'and it is Walk');
assert(!gradeSmoothedConstruction({ observations: reports, modelEdited: false },
  reports, fixtures.smoothedTarget, 1).solved, 'while the original Clean does not');
assert(!gradeSmoothedConstruction({ observations: fixtures.correctedFinal, modelEdited: true },
  reports, fixtures.smoothedTarget, 1).solved, 'and editing the model fails it even with the right report');
assert(!gradeSmoothedConstruction({ observations: [0, 0, 0, 0], modelEdited: false },
  reports, fixtures.smoothedTarget, 1).prefixKept, 'changing an earlier report breaks the prefix condition');
record('smoothed construction rule');

// `hazardOutcome`, over every self-transition on a hundredths grid and every
// pair of elapsed times up to twenty.
/* Distinct elapsed times only. The inner loop used to start at `late = early`,
   so on its first iteration both assertions compared a function with itself:
   606 of the advertised 2,121 could not fail. Each assertion now compares
   against 1 - stay, computed here rather than by the function under test. */
let hazardChecks = 0;
const elapsedValues = [0, 4, 8, 12, 16, 20];
for (let step = 0; step <= 100; step += 1) {
  const stay = step / 100;
  const expectedHazard = 1 - stay;
  for (let first = 0; first < elapsedValues.length; first += 1) {
    for (let second = first + 1; second < elapsedValues.length; second += 1) {
      const early = elapsedValues[first];
      const late = elapsedValues[second];
      const earlySupported = stay > 0 || early === 0;
      const lateSupported = stay > 0 || late === 0;
      equals(hazardOutcome(stay, early, late), earlySupported && lateSupported ? 'same' : 'undefined',
        `the hazard comparison at a = ${stay} requires positive survival at ${early} and ${late}`);
      if (earlySupported) close(exitAfter(stay, early), expectedHazard, `supported hazard after ${early} steps is 1 - a`, 0);
      else equals(exitAfter(stay, early), null, `surviving ${early} steps is impossible when a=0`);
      if (lateSupported) close(exitAfter(stay, late), expectedHazard, `supported hazard after ${late} steps is 1 - a`, 0);
      else equals(exitAfter(stay, late), null, `surviving ${late} steps is impossible when a=0`);
      hazardChecks += 1;
    }
  }
}
equals(hazardChecks, 101 * 15, `exactly ${101 * 15} distinct hazard settings were exercised, not ${hazardChecks}`);
equals(exitAfter(0, 5), null, 'a state with no self-transition cannot have survived five steps');
close(exitAfter(0, 0), 1, 'at entry that state leaves next with certainty', 0);
close(exitAfter(1, 5), 0, 'and an absorbing one never does', 0);
refuses(() => exitAfter(1.1, 5), 'a self-transition probability above one');
refuses(() => exitAfter(0.5, -1), 'a negative elapsed time');
refuses(() => exitAfter(0.5, 1.5), 'a fractional elapsed time');
record('hazard rule');

/* ============================================= refused input everywhere */

refuses(() => checkRow([0.5, 0.4], 'a row'), 'a row that does not sum to one');
// Just outside the declared tolerance, so loosening the tolerance is caught
// rather than merely making a far-out case slightly less far out.
refuses(() => checkRow([0.5, 0.500000001], 'a row'), 'a row one part in a billion away from a distribution');
checkRow([0.5, 0.4999999999], 'a row');
record('row-sum tolerance boundary');
refuses(() => checkRow([1.5, -0.5], 'a row'), 'a row with a negative entry');
refuses(() => checkRow([], 'a row'), 'an empty row');
refuses(() => checkModel({ ...weather, start: [0.6, 0.5] }), 'a model whose initial row is not a distribution');
refuses(() => checkModel({ ...weather, transition: [[0.7, 0.3]] }), 'a transition matrix with too few rows');
refuses(() => checkModel({ ...weather, emission: [[0.1, 0.9], [0.6, 0.3, 0.1]] }), 'ragged emission rows');
refuses(() => infer(weather, []), 'an empty sequence');
refuses(() => infer(weather, [0, 5]), 'a symbol the model has no column for');
refuses(() => infer(weather, [0, 1.5]), 'a fractional symbol index');
refuses(() => infer(weather, new Array(envelopes.toy.maximumTimeSteps + 1).fill(0)),
  'a sequence longer than the toy envelope');
refuses(() => trellis(weather, reports, 'average'), 'an operator that is neither sum nor max');
refuses(() => enumeratePaths(weather, new Array(13).fill(0)), 'an enumeration larger than the declared bound');
refuses(() => expectedCounts(weather, []), 'expected counts with no recordings');
refuses(() => expectedCounts(weather, [[0], [0], [0], [0], [0]]), 'more recordings than the declared bound');
refuses(() => expectedCounts(fixtures.impossibleModel, [fixtures.impossibleObservations]),
  'expected counts for an impossible recording');
refuses(() => durationModel(1.5), 'a self-transition probability above one');
refuses(() => durationModel(0.5, 0), 'a duration histogram with no bars');
refuses(() => parameterCount(0, 3), 'a model with no states');
refuses(() => pathJointOf(weather, [0, 0], reports), 'a path with the wrong number of positions');
refuses(() => pathJointOf(weather, [0, 0, 0, 7], reports), 'a path naming a state the model has not got');
refuses(() => logSumExp([]), 'log-sum-exp of nothing');
refuses(() => repeatedProduct(0.5, 0), 'a repeated product of no factors');
refuses(() => repeatedProduct(1.5, 3), 'a repeated factor above one');
// The real envelope accepts what the toy one refuses, and still has a bound.
const realModel = {
  stateNames: ['Noun', 'Verb', 'Other'], symbolNames: vocabulary,
  start: fittedModels[1].start, transition: fittedModels[1].transition, emission: fittedModels[1].emission,
};
checkModel(realModel, envelopes.real);
refuses(() => checkModel(realModel), 'the 146-symbol tagger under the toy envelope');
refuses(() => infer(realModel, new Array(envelopes.real.maximumTimeSteps + 1).fill(0), envelopes.real),
  'a sentence longer than the real envelope');
record('refused input coverage');

/* ================================================ the generated data module */

assert.equal(provenance.sha256, 'e801e665c4e2a6fa002e04ebb52b00b4fbc1421c7f029494565c0ecbaa016a4d',
  'the module records the served extract hash');
assert.equal(provenance.bytes, 89597, 'and its byte count');
assert.equal(provenance.release, 'r2.16', 'and the release');
assert.equal(provenance.reservedScored, false, 'and that the reserved split is not scored');
assert.equal(provenance.sentences.train + provenance.sentences.development + provenance.sentences.reserved, 200,
  'the three splits hold 200 sentences');
assert.equal(provenance.tokens.train, 1188, '1,188 training tokens');
assert.equal(provenance.tokens.development, 341, '341 development tokens');
assert.equal(provenance.tokens.reserved, 370, 'and 370 reserved ones');
assert.equal(vocabulary.length, 146, 'the vocabulary holds 146 symbols');
assert.equal(vocabulary[0], '<UNKNOWN>', 'whose first entry is the unknown category');
nonEmpty(developmentSentences, 40, 'development sentences');
developmentSentences.forEach(sentence => {
  assert.equal(sentence.tokens.length, sentence.upos.length, `${sentence.id} has one tag per token`);
  assert.equal(sentence.tokens.length, sentence.truth.length, 'and one coarse label per token');
  assert.equal(sentence.tokens.length, sentence.symbols.length, 'and one symbol per token');
  sentence.symbols.forEach(symbol => {
    assert(Number.isInteger(symbol) && symbol >= 0 && symbol < vocabulary.length, 'every symbol indexes the vocabulary');
  });
  assert.deepEqual(sentence.unknownPositions, sentence.symbols
    .map((symbol, index) => (symbol === 0 ? index : -1)).filter(index => index >= 0),
  `${sentence.id} lists exactly its unknown positions`);
  record('development sentence');
});
assert.equal(developmentSentences.reduce((total, sentence) => total + sentence.tokens.length, 0), 341,
  'the forty sentences hold 341 tokens');
assert.equal(developmentSentences.reduce((total, sentence) => total + sentence.unknownPositions.length, 0),
  unknownDevelopmentTokens, 'and the module’s unknown count is the sum of the per-sentence lists');
assert.equal(unknownDevelopmentTokens, 140, 'which is 140');
nonEmpty(fittedModels, 2, 'fitted models');
fittedModels.forEach((fit, index) => {
  checkModel({ ...realModel, start: fit.start, transition: fit.transition, emission: fit.emission },
    envelopes.real);
  close(fit.occupancy.reduce((total, value) => total + value, 0), 1,
    `fit ${index} lexical prior is a distribution`, 1e-12);
  record('fitted model');
});
nonEmpty(configurations, 4, 'decoder configurations');
configurations.forEach((configuration, index) => {
  assert(configuration.fitIndex === 0 || configuration.fitIndex === 1, 'each configuration names a fitted model');
  assert.equal(fittedModels[configuration.fitIndex].smoothing, configuration.smoothing,
    `configuration ${index} and its fit agree on the smoothing strength`);
  assert.equal(configuration.rows.length, 40, 'with one row per development sentence');
  let correct = 0;
  configuration.rows.forEach((row, position) => {
    const sentence = developmentSentences[position];
    assert.equal(row.predicted.length, sentence.tokens.length, 'one prediction per token');
    assert.equal(row.beliefs.length, sentence.tokens.length * 3, 'and three belief entries per token');
    for (let token = 0; token < sentence.tokens.length; token += 1) {
      const belief = row.beliefs.slice(token * 3, token * 3 + 3);
      close(belief.reduce((total, value) => total + value, 0), 1,
        `configuration ${index} sentence ${position} token ${token} beliefs are a distribution`, 5e-6);
    }
    assert.equal(row.correct, row.predicted.filter((value, token) => value === sentence.truth[token]).length,
      `configuration ${index} sentence ${position} correct count`);
    correct += row.correct;
  });
  assert.equal(correct, configuration.correct, `configuration ${index} total correct tokens`);
  assert.equal(configuration.tokens, 341, 'out of 341');
  assert.equal(configuration.sentencesCorrect,
    configuration.rows.filter((row, position) => row.correct === developmentSentences[position].tokens.length).length,
    `configuration ${index} whole-sentence count`);
  record('decoder configuration');
});
assert.deepEqual(configurations.map(entry => entry.correct), [266, 268, 266, 270], 'the four recorded token counts');
assert.deepEqual(configurations.map(entry => entry.sentencesCorrect), [8, 9, 8, 9], 'and whole-sentence counts');
assert.equal(selectedConfigurationIndex, 3, 'the selected configuration is the HMM at smoothing 1.0');
assert.equal(configurations[selectedConfigurationIndex].correct,
  largestOf(configurations.map(entry => entry.correct)), 'which has the highest token count');
assert.equal(configurations[0].correct, configurations[2].correct,
  'and the two lexical configurations genuinely tie, so the declared tie rule is exercised');
assert.equal(majority.correct, 216, 'the majority baseline gets 216 tokens');
assert(majority.correct < smallestOf(configurations.map(entry => entry.correct)),
  'which every fitted configuration beats');
record('data module shape');

// Repairs and breaks, recomputed from the stored predictions.
const lexicalRows = configurations[2].rows;
const hmmRows = configurations[3].rows;
let repairCount = 0;
let breakCount = 0;
developmentSentences.forEach((sentence, position) => {
  sentence.truth.forEach((truth, token) => {
    const lexical = lexicalRows[position].predicted[token];
    const hmm = hmmRows[position].predicted[token];
    if (lexical !== truth && hmm === truth) repairCount += 1;
    if (lexical === truth && hmm !== truth) breakCount += 1;
  });
});
assert.equal(repairCount, decisionChanges.repairs.length, 'the recorded repairs are recomputable');
assert.equal(breakCount, decisionChanges.breaks.length, 'and so are the breaks');
assert.equal(repairCount, 10, 'there are ten repairs');
assert.equal(breakCount, 6, 'and six breaks');
assert.equal(repairCount - breakCount, configurations[3].correct - configurations[2].correct,
  'and their difference is the token-count difference');
decisionChanges.repairs.forEach(entry => {
  const sentence = developmentSentences[entry.sentence];
  assert.equal(sentence.tokens[entry.position], entry.token, 'a repair names its own token');
  assert(lexicalRows[entry.sentence].predicted[entry.position] !== sentence.truth[entry.position],
    'the lexical rule really was wrong there');
  assert.equal(hmmRows[entry.sentence].predicted[entry.position], sentence.truth[entry.position],
    'and the HMM really was right');
  record('recorded repair');
});
decisionChanges.breaks.forEach(entry => {
  const sentence = developmentSentences[entry.sentence];
  assert.equal(lexicalRows[entry.sentence].predicted[entry.position], sentence.truth[entry.position],
    'a break really was right lexically');
  assert(hmmRows[entry.sentence].predicted[entry.position] !== sentence.truth[entry.position],
    'and really is wrong under the HMM');
  record('recorded break');
});
// The two named specimens.
assert.deepEqual(developmentSentences[27].tokens, ['Dear', 'Nina', ','], 'development index 27');
assert.deepEqual(lexicalRows[27].predicted, [0, 0, 2], 'whose lexical prediction is Noun Noun Other');
assert.deepEqual(hmmRows[27].predicted, [2, 0, 2], 'and whose HMM prediction is Other Noun Other');
assert.deepEqual(hmmRows[27].predicted, developmentSentences[27].truth, 'matching the coarse reference');
assert.equal(developmentSentences[3].tokens[3], 'article', 'development index 3 holds the word article');
assert.equal(lexicalRows[3].predicted[3], developmentSentences[3].truth[3], 'which the lexical rule tags correctly');
assert.notEqual(hmmRows[3].predicted[3], developmentSentences[3].truth[3], 'and the HMM breaks');
record('named specimens');

// WHICH SPLIT the ties belong to, established from the packet rather than from
// a word in the prose. The intro asserted them on the reserved test split, in a
// lesson that says three times that those forty are never scored.
const packetTagging = recorded.real_tagging;
const developmentIds = new Set(packetTagging.development_ids);
nonEmpty(packetTagging.development_ids, 40, 'recorded development sentence ids');
tieAudit.configurations.forEach(entry => {
  entry.tiedSentences.forEach(tie => {
    const sentence = developmentSentences[tie.sentence];
    equals(sentence.id, tie.id, `tied sentence ${tie.sentence} names its own id`);
    ok(developmentIds.has(tie.id),
      `tied sentence ${tie.sentence} is a DEVELOPMENT sentence, which is the split the tie audit was computed from`);
    record('tied sentence belongs to the development split');
  });
});
equals(packetTagging.reserved_test_scored, false, 'and the reserved split is scored nowhere');
ok(developmentSentences.every(sentence => developmentIds.has(sentence.id)),
  'every sentence the lesson serves is one of the forty development sentences');
record('tie split provenance');

// The tie audit: three sentences, two optimal paths each, and a reported band.
nonEmpty(tieAudit.configurations, 2, 'tie audit configurations');
tieAudit.configurations.forEach(entry => {
  nonEmpty(entry.tiedSentences, 3, `tied sentences at smoothing ${entry.smoothing}`);
  assert.deepEqual(entry.tiedSentences.map(tie => tie.sentence), [1, 2, 39],
    'and they are development sentences 1, 2 and 39');
  entry.tiedSentences.forEach(tie => {
    assert.equal(tie.optimalPaths, 2, `sentence ${tie.sentence} has exactly two optimal paths`);
    nonEmpty(tie.adjacentUnknownPairs, undefined, `sentence ${tie.sentence} adjacent unknown pairs`);
    tie.adjacentUnknownPairs.forEach(([left, right]) => {
      const sentence = developmentSentences[tie.sentence];
      assert.equal(sentence.symbols[left], 0, 'the first of a named pair is an unknown word');
      assert.equal(sentence.symbols[right], 0, 'and so is the second');
      assert.equal(right, left + 1, 'and they are adjacent');
      record('tied adjacent unknown pair');
    });
    record('tied sentence');
  });
  const band = entry.tokenTotals;
  /* The band is established in exact rational arithmetic by
     scripts/verify-hmm-data.py, which recomputes all three members from the
     served extract; this file pins the values so that a tampered or narrowed
     band is caught here too rather than only in the slower verifier. */
  const declaredBands = { 0.1: { first: 267, stored: 268, last: 270 }, 1: { first: 269, stored: 270, last: 272 } };
  const declared = declaredBands[entry.smoothing];
  assert(declared, `the tie band at smoothing ${entry.smoothing} is one of the declared ones`);
  assert.equal(band.first, declared.first, `tie band lower member at smoothing ${entry.smoothing}`);
  assert.equal(band.stored, declared.stored, `tie band stored member at smoothing ${entry.smoothing}`);
  assert.equal(band.last, declared.last, `tie band upper member at smoothing ${entry.smoothing}`);
  assert(band.first < band.stored && band.stored < band.last,
    `the recorded count sits strictly inside the tie band at smoothing ${entry.smoothing}`);
  assert(band.last > band.first,
    `and the band is genuinely wide at smoothing ${entry.smoothing}, not a restatement of one number`);
  const configuration = configurations.find(item => item.method === 'hmm' && item.smoothing === entry.smoothing);
  assert.equal(band.stored, configuration.correct, 'and the stored member is the published count');
  record('tie band');
});
// The comparison survives the whole band, which is the claim the prose makes.
['first', 'last', 'stored'].forEach(rule => {
  assert(tieAudit.configurations[1].tokenTotals[rule] > configurations[2].correct,
    `the HMM beats its matching lexical baseline under the ${rule} tie rule`);
  assert(tieAudit.configurations[1].tokenTotals[rule] > tieAudit.configurations[0].tokenTotals[rule],
    `and smoothing 1.0 beats 0.1 under the ${rule} tie rule`);
  record('tie-band comparison');
});
record('tie audit');

// The EM track.
assert.equal(emTrack.dataSeed, 71, 'the recordings were sampled with seed 71');
assert.equal(emTrack.recordings, 12, 'twelve of them');
assert.equal(emTrack.length, 30, 'each thirty reports long');
assert.equal(emTrack.sequences.length, 12, 'and the module carries all twelve');
emTrack.sequences.forEach(sequence => {
  assert.equal(sequence.length, 30, 'each recording holds thirty reports');
  sequence.forEach(value => assert(value >= 0 && value < 3, 'every report is one of the three symbols'));
  record('recorded EM sequence');
});
nonEmpty(emTrack.fits, 3, 'EM fits');
emTrack.fits.forEach(fit => {
  assert.equal(fit.logLikelihood.length, 41, `seed ${fit.seed} records the initial model and forty updates`);
  for (let index = 1; index < fit.logLikelihood.length; index += 1) {
    assert(fit.logLikelihood[index] >= fit.logLikelihood[index - 1] - 1e-6,
      `seed ${fit.seed} never decreased at update ${index}`);
  }
  nonEmpty(fit.checkpoints, 3, `seed ${fit.seed} parameter checkpoints`);
  fit.checkpoints.forEach(checkpoint => {
    checkModel({ ...weather, start: checkpoint.start, transition: checkpoint.transition,
      emission: checkpoint.emission });
    close(checkpoint.logLikelihood, fit.logLikelihood[checkpoint.update],
      `seed ${fit.seed} checkpoint ${checkpoint.name} objective`, 1e-6);
    record('EM checkpoint');
  });
  record('EM fit');
});
assert.deepEqual(emTrack.fits.map(fit => fit.seed), [3, 7, 19], 'the three declared seeds');
close(emTrack.generatingLogLikelihood, -393.562196, 'the generating score on this sample', 1e-5);
assert(emTrack.fits.every(fit => fit.logLikelihood[40] > emTrack.generatingLogLikelihood),
  'all three fits score above the generating parameters on this finite sample');
assert(emTrack.uniformStart.logLikelihood[1] < emTrack.generatingLogLikelihood,
  'while the symmetric start stays below it, so "above" is not vacuous here');
vector(emTrack.uniformStart.emission[0], emTrack.uniformStart.emission[1],
  'the uniform start leaves both emission rows identical', 0);
vector(emTrack.uniformStart.emission[0], emTrack.uniformStart.empiricalSymbolFrequencies,
  'and equal to the empirical symbol frequencies', 1e-9);
close(emTrack.uniformStart.empiricalSymbolFrequencies.reduce((total, value) => total + value, 0), 1,
  'which are themselves a distribution', 1e-12);
close(emTrack.uniformStart.logLikelihood[1], emTrack.uniformStart.logLikelihood[2],
  'and its objective plateaus after one update', 1e-6);
record('EM track');

/* =========================================== the generated examples module */

assert.equal(hmmExamples.experiments.sha256, '25e7e4016c5420b7aae33a21e869275564d6951bc22865fd173f566c1d7b2f77',
  'the complete program is the pinned revision');
assert(hmmExamples.experiments.executed, 'and it was executed');
assert.equal(hmmExamples.experiments.setup, 'python hmm-experiments.py', 'with the manuscript’s own run command');
assert(hmmExamples.experiments.expected.includes('Recorded exact mechanisms'),
  'and its recorded output is the line it prints');
nonEmpty(hmmExamples.experiments.excerpts, 4, 'displayed excerpts');
const programSource = fs.readFileSync(`${packetDirectory}/hmm-experiments.py`, 'utf8');
hmmExamples.experiments.excerpts.forEach(excerpt => {
  assert(programSource.includes(excerpt.code),
    `the ${excerpt.key} excerpt is a literal slice of the frozen program`);
  assert.equal(crypto.createHash('sha256').update(excerpt.code).digest('hex'), excerpt.sha256,
    `and its recorded hash matches (${excerpt.key})`);
  assert(excerpt.lines[0] >= 1 && excerpt.lines[1] >= excerpt.lines[0],
    `and its line range is sensible (${excerpt.key})`);
  assert(excerpt.functions.every(name => excerpt.code.includes(`def ${name}(`)),
    `and it contains every function it names (${excerpt.key})`);
  record('displayed excerpt');
});
const excerptFunctions = hmmExamples.experiments.excerpts.flatMap(excerpt => excerpt.functions);
assert.equal(new Set(excerptFunctions).size, excerptFunctions.length, 'no function is displayed twice');
assert(hmmExamples.hmmlearn.executed, 'the optional program was executed too');
assert.equal(hmmExamples.hmmlearn.environment.hmmlearn, '0.3.3', 'against hmmlearn 0.3.3');
assert(hmmExamples.hmmlearn.expected.includes('-4.673517551573099'),
  'and its output carries the observation log probability this lesson computes');
close(globalThis.Math.log(forward.final), -4.673517551573099,
  'which is exactly the log of the evidence computed here', 1e-12);
assert(hmmExamples.hmmlearn.expected.includes('2.940021586061572'),
  'and the MAP decoder score it warns about');
close(main.smoothed.reduce((total, row) => total + largestOf(row), 0), 2.940021586061572,
  'which is the sum of the four smoothed maxima, not a log probability', 1e-9);
assert(main.smoothed.reduce((total, row) => total + largestOf(row), 0) > 0,
  'and is positive, so it cannot be a log probability of a probability');
assert(hmmExamples.hmmlearn.warning.includes('7 free scalar parameters'),
  'the recorded warning counts seven free parameters');
close(parameterCount(2, 3), 7, 'which is what this lesson’s own formula gives', 0);
record('examples module');

/* ------------------------------------ sequential audit: shortened construction */
const shortConstruction = gradeSmoothedConstruction(
  { observations: [0], modelEdited: false }, reports, fixtures.smoothedTarget, 1);
assert.equal(shortConstruction.solved, false, 'one report cannot solve the fixed time-1 task');
assert.equal(shortConstruction.filtered, null, 'unavailable query returns null instead of indexing past the array');
assert.equal(shortConstruction.smoothed, null);
assert.equal(shortConstruction.prefixKept, false);
record('sequential one-report construction');

/* --------------------------------------------------------------- record */

function largestOf(values) { return values.reduce((top, value) => (value > top ? value : top)); }
function smallestOf(values) { return values.reduce((low, value) => (value < low ? value : low)); }
const sources = [
  'src/learn/data/hmm-models.js',
  'src/learn/data/hmm-data.js',
  'src/learn/data/hmm-examples.js',
  'src/learn/components/lesson-labs/HmmShared.jsx',
  'src/learn/components/lesson-labs/HmmLabs.jsx',
  'src/learn/components/lesson-labs/HmmFigures.jsx',
  'src/learn/components/lesson-labs/hmm-labs.css',
  'src/learn/data/topics/hidden-markov-models-hmm.jsx',
  'src/learn/data/curriculum/blueprints/hidden-markov-models-hmm.js',
  'public/learn-assets/hmm/ewt-sequences.json',
  'public/learn-assets/hmm/hmm-experiments.py',
  'public/learn-assets/hmm/hmmlearn-examples.py',
  'public/learn-assets/hmm/ATTRIBUTION.txt',
];
const hash = file => crypto.createHash('sha256').update(fs.readFileSync(file)).digest('hex');
const total = Object.values(counts).reduce((sumValue, value) => sumValue + value, 0);
const evidence = {
  checkedAt: new Date().toISOString(),
  sourceHashes: Object.fromEntries(sources.filter(fs.existsSync).map(file => [file, hash(file)])),
  packetCalculatedInputsSha256: crypto.createHash('sha256')
    .update(fs.readFileSync(`${packetDirectory}/calculated-inputs.json`)).digest('hex'),
  verifierHash: hash('scripts/verify-hmm-models.mjs'),
  counts,
  totalGroupedChecks: total,
  executedAssertions: assertions,
  countingNote: 'totalGroupedChecks counts hand-placed group markers; executedAssertions counts assertions that '
    + 'actually ran. The second is the coverage number.',
  gridCoverage: {
    enumeratedPathRankChecks: rankChecks,
    fourStepRecordingsAtEveryQueryTime: moveChecks,
    filteringPrefixIndependenceChecks: prefixChecks,
    priorsOnTheThreeSimplex: priors.length,
    recordingCompositionsUnderThreeSymbolPatterns: boundaryChecks,
    hazardSettings: hazardChecks,
    edgeWidthGridPoints: shares.length,
    drawnTrellisEdges: drawnEdges,
  },
  scope: 'The browser hidden Markov model layer against the content packet’s calculated-inputs.json and every '
    + 'number the lesson states, by a second numerical route: the browser infers by retained scaling while the '
    + 'packet’s author program works in log space throughout, and the sixteen-path enumeration and a '
    + 'brute-force search over legal paths supply third routes where they are cheap. Covered: the declared model '
    + 'and all sixteen path products; the forward trellis, its column totals, the manuscript’s bracket '
    + 'arithmetic and the rising Rainy cell under a falling total; filtering, the retained scale factors and both '
    + 'forecasts; the backward likelihoods recovered from the scaled rows, the smoothed rows, all three pair '
    + 'blocks with both margins and the last-row identity; the changed-future filtered prefix held bit for bit; a '
    + 'retained missing report against a deleted step; the Viterbi cells, stored predecessors, the third-column '
    + 'predecessor claim on the edge record, backtracking and the posterior share; a changed emission row leaving '
    + 'the best path’s joint mass exactly alone; the constrained graph’s marginals, its impossible '
    + 'pointwise route, both expected-position counts, the joint mass table and the changed prior; expected '
    + 'counts, the boundary contrast, one complete update, the duplication null and length-one retention; '
    + 'underflow, log-sum-exp including an all-negative-infinity row, and an impossible sequence with no '
    + 'posterior; the duration family including the absorbing case; parameter and transition-contribution counts. '
    + 'Also covered: every quantity the figures draw - edge widths over a 201-point share grid, the shares of '
    + 'every drawn edge of four fixtures in both modes, bar heights, belief-track rows and the two plot windows '
    + 'with the pixel separation that justifies having two. And every rule the investigations grade with, over '
    + 'the whole grid its control can reach: all sixteen paths of four models including one where every path ties '
    + 'and one where six are impossible; all 256 four-step recordings at every query time, with filtering '
    + 'independence of everything after the queried time asserted by exact equality; all 231 priors on a '
    + 'twentieths grid of the probability simplex on three states; every composition of up to eight reports into up to four recordings '
    + 'under three symbol patterns, including the one-and-three split that matches the totals and is still '
    + 'rejected; every single-report replacement for the smoothed construction; and every self-transition '
    + 'probability on a hundredths grid against elapsed times up to twenty. Refused input is exercised on every '
    + 'entry point, including the 146-symbol tagger under the toy envelope.',
  limitations: [
    'The English Web Treebank results are recorded fits; the browser reproduces their stored outcomes, not the fitting.',
    'Displayed program execution is verified separately by scripts/verify-hmm-examples.py.',
    'The data module is regenerated and matched to the packet separately by scripts/verify-hmm-data.py, '
      + 'which also re-derives the packet’s own calculated inputs leaf by leaf.',
    'Rendering, interaction, visual layout and independent review are separate steps.',
  ],
  passed: true,
};
fs.mkdirSync('docs/teaching/evidence', { recursive: true });
fs.writeFileSync('docs/teaching/evidence/hmm-models.json', JSON.stringify(evidence, null, 2) + '\n');
console.log(`PASS: ${assertions.toLocaleString('en-US')} executed assertions in `
  + `${Object.keys(counts).length} recorded groups, including `
  + `${moveChecks.toLocaleString('en-US')} belief-move verdicts over all 256 four-step recordings, `
  + `${priors.length} priors, ${boundaryChecks} recording compositions and `
  + `${hazardChecks.toLocaleString('en-US')} distinct hazard settings.`);
