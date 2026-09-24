// Bounded independent checks of the PAC/VC browser models against the content
// packet's recorded calculations, the manuscript's stated values, analytic
// identities, and a second derivation for every claim a figure draws.
//
// Every assertion here is written so that it can fail: nothing is compared with
// itself, no loop is allowed to inspect an empty subject set, and every group
// asserts how many subjects it actually saw.
//
// Three rules shaped this file.
//
//   1. A SECOND ROUTE MEANS A DIFFERENT DERIVATION. Pattern counts are checked
//      against brute force over all 2^n binary vectors AND against the closed
//      form -- three routes, none of them a second call to the enumeration.
//      Feasibility is checked against membership in the enumerated pattern set.
//      The four-input occupancy probabilities are checked against exhaustive
//      enumeration of all 4^n draw sequences at small n and against
//      inclusion-exclusion at large n. The sine witnesses are checked against
//      the actual floating-point sign of sin(theta x), which is what the class
//      is defined by. Symmetric-difference risk is checked against a
//      union-minus-intersection route.
//
//   2. A RULE DRAWN MUST EQUAL THE RULE APPLIED ACROSS THE WHOLE ENTERABLE
//      GRID. Investigation 2's feasibility verdict is swept over every ordering
//      of 2 to 6 points and every labeling of them -- 100,000+ cases -- not over
//      samples of them, because a lab that grades a correct answer wrong is the
//      defect class this topic is most exposed to.
//
//   3. EVERY GRADED COMPARISON IS EXERCISED AT ITS DEGENERATE INPUTS. Exact
//      ties, identical states, zero-width targets, empty positive sets, an
//      empty observation sequence and a zero difference all appear below, and
//      the unchanged rule is asserted as an IF AND ONLY IF rather than by
//      example.
//
// Run: node scripts/verify-pac-models.mjs
import assert from 'node:assert/strict';
import fs from 'node:fs';
import crypto from 'node:crypto';
import {
  binomial, boundGrid, boundPlotGeometry, candidateBandGeometry, candidateErrorCounts, checkFinite,
  diagonalGeometry, eliminationTrace, finiteClassFailureBound, finiteFamilyRadii, finiteRadius,
  finiteRealizableSampleBound, finiteWorldHypotheses, finiteWorldInputs, finiteWorldProbability,
  finiteWorldRisk, finiteWorldRun, fixtures, fraction, fractionAdd, fractionCompare, fractionMultiply,
  fractionText, fractionToNumber, ghostCollapse, growthPlotGeometry, growthRow, growthTable,
  halfPlanePredictions, interiorCombination, intervalExperiment, intervalPatterns, intervalRisk,
  labelsFromTarget, learningCurveGeometry, lineSegmentInBox, linearScale, logScale, manualIntervalRequest,
  meterGeometry, movementOf, patternCountFormula, planePanelGeometry, realizableSufficientTable,
  realizableVcSampleBound, requestVerdict, riskSegments, sauerSum, seededUniformSample, segmentsCross,
  labelCollisions,
  selectedHypothesis, signedMargin, simulationPlotGeometry, sineWitness, sinePlotGeometry, sortWithIdentity,
  stripGeometry, thresholdPatterns, twoStripBound, unchangedTolerance, vcRadius, witnessPredictions,
} from '../src/learn/data/pac-models.js';
import { pacData } from '../src/learn/data/pac-data.js';
import { pacExamples } from '../src/learn/data/pac-examples.js';

const packetDirectory = 'docs/teaching/drafts/pac-learning-vc-dimension';
const recorded = JSON.parse(fs.readFileSync(`${packetDirectory}/checked-results.json`, 'utf8'));
const recordedCurves = JSON.parse(fs.readFileSync(`${packetDirectory}/banknote-learning-curve-results.json`, 'utf8'));

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

/* ======================================= second routes, written from scratch */

/** Every binary vector of length n. */
function everyPattern(n) {
  return Array.from({ length: 2 ** n }, (_unused, code) =>
    Array.from({ length: n }, (_ignored, index) => (code >> (n - 1 - index)) & 1));
}

/** The ones form one unbroken run. Nothing to do with interval endpoints. */
function contiguousRun(pattern) {
  const ones = pattern.map((bit, index) => (bit === 1 ? index : -1)).filter(index => index >= 0);
  return ones.length === 0 || ones.every((index, position) => index === ones[0] + position);
}

/** The ones form a suffix. Nothing to do with a threshold value. */
function suffixOfOnes(pattern) {
  const ones = pattern.map((bit, index) => (bit === 1 ? index : -1)).filter(index => index >= 0);
  return ones.length === 0 || (ones[ones.length - 1] === pattern.length - 1
    && ones.every((index, position) => index === ones[0] + position));
}

/** Every permutation of a list. */
function permutations(items) {
  if (items.length <= 1) return [items];
  return items.flatMap((item, index) =>
    permutations([...items.slice(0, index), ...items.slice(index + 1)]).map(rest => [item, ...rest]));
}

/** Symmetric-difference length by union minus intersection, not by the module's
 *  |A| + |B| − 2|A ∩ B|. */
function symmetricDifferenceByUnion(interval, target) {
  const [a, b] = target;
  if (interval === null) return b - a;
  const [left, right] = interval;
  const intersection = Math.max(0, Math.min(b, right) - Math.max(a, left));
  const union = intersection > 0
    ? Math.max(b, right) - Math.min(a, left)
    : (b - a) + (right - left);
  return union - intersection;
}

/** Exhaustive enumeration of every draw sequence in the four-input world.
 *  4^n sequences, so only usable at small n -- which is exactly why it is a
 *  useful independent check on the occupancy argument. */
function finiteWorldByEnumeration(target, n, epsilon) {
  let failing = 0;
  const total = 4 ** n;
  for (let code = 0; code < total; code += 1) {
    const sample = [];
    let rest = code;
    for (let position = 0; position < n; position += 1) { sample.push(rest % 4); rest = Math.floor(rest / 4); }
    const rule = selectedHypothesis(target, sample);
    if (finiteWorldRisk(rule, target).risk > epsilon) failing += 1;
  }
  return failing / total;
}

/* ============================================== §5-§6 · counting the patterns */

for (let n = 1; n <= 10; n += 1) {
  const enumerated = intervalPatterns(n);
  const byBruteForce = everyPattern(n).filter(contiguousRun);
  assert.equal(enumerated.length, byBruteForce.length,
    `interval patterns at n=${n}: endpoint enumeration gives ${enumerated.length}, contiguity test gives ${byBruteForce.length}`);
  assert.deepEqual(enumerated.map(pattern => pattern.join('')), byBruteForce.map(pattern => pattern.join('')),
    `interval patterns at n=${n}: the two routes disagree on which patterns`);
  assert.equal(enumerated.length, patternCountFormula('interval', n),
    `interval patterns at n=${n}: the closed form disagrees`);
  const thresholds = thresholdPatterns(n);
  const thresholdsByBruteForce = everyPattern(n).filter(suffixOfOnes);
  assert.equal(thresholds.length, thresholdsByBruteForce.length, `threshold patterns at n=${n}`);
  assert.deepEqual([...thresholds].map(p => p.join('')).sort(),
    thresholdsByBruteForce.map(p => p.join('')).sort(), `threshold pattern sets at n=${n}`);
  assert.equal(thresholds.length, patternCountFormula('threshold', n), `threshold closed form at n=${n}`);
  record('pattern counts by three routes');
}
nonEmpty(intervalPatterns(3), 7, 'the seven patterns on three points');
assert.ok(!intervalPatterns(3).some(pattern => pattern.join('') === '101'),
  'the one labelling intervals cannot make on three points is 101');
assert.equal(everyPattern(3).filter(contiguousRun).length + 1, 8,
  'and exactly one of the eight is missing');
nonEmpty(intervalPatterns(4), 11, 'eleven patterns on four points');
assert.deepEqual(everyPattern(4).filter(pattern => !contiguousRun(pattern)).map(pattern => pattern.join('')),
  ['0101', '1001', '1010', '1011', '1101'], 'the five impossible four-point patterns the practice lists');
record('the missing patterns');

growthTable.forEach((row, index) => {
  /* O1: comparing growthTable[i] with growthRow(i+1) was a tautology --
     growthTable IS Array.from(..., growthRow). What is worth asserting is that
     the exported table has the shape and length the figures index into. */
  assert.equal(row.n, index + 1, 'the exported growth table is indexed by n');
  const packetRow = recorded.growth[index];
  assert.equal(row.n, packetRow.n);
  assert.equal(row.thresholds, packetRow.thresholds, `thresholds at n=${row.n}`);
  assert.equal(row.intervals, packetRow.intervals, `intervals at n=${row.n}`);
  assert.equal(row.allBinary, packetRow.all_binary, `2^n at n=${row.n}`);
  assert.equal(row.sauerD2, packetRow.sauer_d2, `Sauer d=2 at n=${row.n}`);
  assert.ok(row.intervals <= row.sauerD2, `Sauer d=2 bounds the interval class at n=${row.n}`);
  assert.ok(row.thresholds <= row.intervals, `thresholds are a subclass in count at n=${row.n}`);
  record('growth table against the packet');
});
assert.equal(growthTable[4].intervals, 16);
assert.equal(growthTable[4].allBinary, 32);
assert.equal(growthTable[9].intervals, 56);
assert.equal(growthTable[9].allBinary, 1024);
record('the two fractions the prose quotes');

// Sauer's sum, by binomials and by Pascal's recurrence.
for (let n = 1; n <= 30; n += 1) {
  for (let d = 0; d <= 6; d += 1) {
    let byRecurrence = 0;
    for (let i = 0; i <= Math.min(d, n); i += 1) {
      let term = 1;
      for (let step = 0; step < i; step += 1) term = (term * (n - step)) / (step + 1);
      byRecurrence += Math.round(term);
    }
    assert.equal(sauerSum(n, d), byRecurrence, `Sauer sum at n=${n}, d=${d}`);
    if (n <= 20 && d <= n) {
      assert.ok(sauerSum(n, d) <= 2 ** n, `Sauer sum never exceeds 2^n at n=${n}, d=${d}`);
    }
    record('Sauer sum by two routes');
  }
}
assert.equal(sauerSum(100, 5), recorded.sauer_n100_d5, 'the n=100, d=5 sum the prose quotes');
assert.equal(sauerSum(100, 5), 79375496);
assert.equal(sauerSum(6, 2), recorded.practice.sauer_n6_d2);
assert.equal(binomial(100, 5), 75287520, 'one binomial by itself');
assert.equal(binomial(5, 0), 1);
assert.equal(binomial(5, 6), 0, 'k above n is zero, not an error');
refuses(() => binomial(-1, 0), 'a negative n');
record('the quoted Sauer values');

/* ======================== §5 · the feasibility rule over its whole grid */

let witnessCases = 0;
let feasibleSeen = 0;
let infeasibleSeen = 0;
for (let n = 2; n <= 6; n += 1) {
  const coordinates = Array.from({ length: n }, (_unused, index) => (index + 1) / (n + 1));
  const names = ['A', 'B', 'C', 'D', 'E', 'F'].slice(0, n);
  const orderings = permutations(coordinates);
  nonEmpty(orderings, undefined, `orderings at n=${n}`);
  const intervalSet = new Set(intervalPatterns(n).map(pattern => pattern.join('')));
  const thresholdSet = new Set(thresholdPatterns(n).map(pattern => pattern.join('')));
  for (const ordering of orderings) {
    for (const labeling of everyPattern(n)) {
      for (const family of ['interval', 'threshold']) {
        const points = names.map((id, index) => ({ id, x: ordering[index], label: labeling[index] }));
        const verdict = requestVerdict({ points, family });
        // Second route: is the SORTED pattern one the class realizes at all?
        const sorted = [...points].sort((left, right) => left.x - right.x).map(point => point.label);
        const expected = (family === 'interval' ? intervalSet : thresholdSet).has(sorted.join(''));
        assert.equal(verdict.feasible, expected,
          `${family} on ${ordering.join(',')} with labels ${labeling.join('')}: `
          + `the verdict says ${verdict.feasible} and the enumerated pattern set says ${expected}`);
        assert.deepEqual(verdict.pattern, sorted, 'the reported pattern is the sorted one');
        if (verdict.feasible) {
          feasibleSeen += 1;
          // Third route: the witness itself must reproduce the request exactly.
          const predicted = witnessPredictions(verdict.witness, points);
          assert.deepEqual(predicted.map(entry => entry.predicted), points.map(point => point.label),
            `${family}: the witness does not reproduce the request it witnesses`);
          assert.equal(verdict.obstruction, null, 'a feasible request reports no obstruction');
        } else {
          infeasibleSeen += 1;
          assert.equal(verdict.witness, null, 'an infeasible request offers no witness');
          assert.ok(verdict.obstruction, 'an infeasible request names its obstruction');
          const named = Object.values(verdict.obstruction);
          assert.ok(named.every(id => names.includes(id)), 'the obstruction names real points');
        }
        witnessCases += 1;
      }
    }
  }
  record(`feasibility swept at n=${n}`);
}
assert.ok(witnessCases >= 100000, `only ${witnessCases} feasibility cases were swept`);
assert.ok(feasibleSeen > 0 && infeasibleSeen > 0,
  `the sweep saw ${feasibleSeen} feasible and ${infeasibleSeen} infeasible cases; both verdicts must occur`);
record('both verdicts occur in the sweep');

// The named fixtures, including the nulls the lab offers as buttons.
const baseRequest = requestVerdict(fixtures.request);
assert.equal(baseRequest.feasible, false, '101 on .2/.5/.8 is not realizable by one interval');
assert.equal(baseRequest.obstruction.negative, 'B', 'and B is the obstruction');
const repaired = requestVerdict(fixtures.requestRepaired);
assert.equal(repaired.feasible, true, 'repairing B to positive makes 111 realizable');
assert.deepEqual(repaired.witness.interval, [0.2, 0.8], 'with the tight witness [.2, .8]');
const orderNull = requestVerdict(fixtures.requestOrderNull);
assert.equal(orderNull.feasible, baseRequest.feasible, 'an order-preserving move is an exact null');
assert.equal(orderNull.obstruction.negative, baseRequest.obstruction.negative, 'with the same obstruction');
const translated = requestVerdict(fixtures.requestTranslated);
assert.equal(translated.feasible, baseRequest.feasible, 'a common translation is an exact null');
const crossed = requestVerdict(fixtures.requestCrossed);
assert.equal(crossed.feasible, true, 'moving B past C changes the sorted pattern and makes it realizable');
assert.deepEqual(crossed.order, ['A', 'C', 'B'], 'and the reported order shows why');
const thresholdTwo = requestVerdict(fixtures.requestThreshold);
assert.equal(thresholdTwo.feasible, false, 'the labelling 10 is impossible for an increasing threshold');
assert.equal(requestVerdict({
  ...fixtures.requestThreshold,
  points: fixtures.requestThreshold.points.map(point => ({ ...point, label: 1 - point.label })),
}).feasible, true, 'while 01 is possible');
record('the lab fixtures and their nulls');

refuses(() => sortWithIdentity([{ id: 'A', x: 0.5 }, { id: 'B', x: 0.5 }]), 'coincident coordinates');
refuses(() => requestVerdict({ family: 'circle', points: fixtures.request.points }), 'an unknown family');
refuses(() => requestVerdict({
  family: 'interval', points: [{ id: 'A', x: 0.1, label: 2 }],
}), 'a label that is not 0 or 1');
refuses(() => intervalPatterns(-1), 'a negative point count');
refuses(() => checkFinite(Number.NaN, 'a test value'), 'a non-finite value');

// The empty positive set is a member, and the all-negative request uses it.
const allNegative = requestVerdict({
  family: 'interval', points: [{ id: 'A', x: 0.2, label: 0 }, { id: 'B', x: 0.7, label: 0 }],
});
assert.equal(allNegative.feasible, true);
assert.equal(allNegative.witness.empty, true, 'all-negative uses the empty positive region explicitly');
assert.equal(allNegative.witness.interval, null);
const allPositive = requestVerdict({
  family: 'interval', points: [{ id: 'A', x: 0.2, label: 1 }, { id: 'B', x: 0.7, label: 1 }],
});
assert.deepEqual(allPositive.witness.interval, [0.2, 0.7], 'all-positive uses the tight interval');
const thresholdAllNegative = requestVerdict({
  family: 'threshold', points: [{ id: 'A', x: 0.2, label: 0 }, { id: 'B', x: 0.7, label: 0 }],
});
assert.ok(thresholdAllNegative.witness.threshold > 0.7, 'a threshold above every point labels them all negative');
record('the degenerate label requests');

/* ================================== §3 · the exact four-input world */

// The lexicographic selection rule against its closed form, over every target
// and every observable subset -- not over a sample of them.
let selectionCases = 0;
for (const target of everyPattern(4)) {
  for (let mask = 0; mask < 16; mask += 1) {
    const sample = finiteWorldInputs.filter(input => mask & (1 << input));
    const chosen = selectedHypothesis(target, sample);
    const closedForm = finiteWorldInputs.map(input => ((mask & (1 << input)) ? target[input] : 0));
    assert.deepEqual(chosen, closedForm,
      `target ${target.join('')} with mask ${mask}: enumeration gives ${chosen.join('')}, `
      + `the closed form gives ${closedForm.join('')}`);
    // And it really is the FIRST consistent rule in the listed order.
    const firstConsistent = finiteWorldHypotheses.findIndex(rule =>
      finiteWorldInputs.every(input => !(mask & (1 << input)) || rule[input] === target[input]));
    assert.deepEqual(finiteWorldHypotheses[firstConsistent], chosen, 'and it is the first consistent rule');
    selectionCases += 1;
  }
}
assert.equal(selectionCases, 256, 'every target and every observable subset was covered');
record('the selection rule by two routes');

// A repeated observation reveals nothing new: an exact null, over every target.
for (const target of everyPattern(4)) {
  for (const input of finiteWorldInputs) {
    const once = finiteWorldRun({ target, sample: [input], epsilon: 0.25 });
    const twice = finiteWorldRun({ target, sample: [input, input], epsilon: 0.25 });
    assert.deepEqual(once.rule, twice.rule, 'a repeated observation does not change the rule');
    assert.equal(once.risk, twice.risk, 'nor the risk');
    assert.equal(twice.repeatedObservations, 1, 'and the repetition is counted and named');
    assert.equal(once.repeatedObservations, 0);
  }
  record('the repeat null');
}

// meetsTarget is exactly risk <= epsilon, including at the tie.
let tieCases = 0;
for (const target of everyPattern(4)) {
  for (let mask = 0; mask < 16; mask += 1) {
    const sample = finiteWorldInputs.filter(input => mask & (1 << input));
    for (const epsilon of [0.05, 0.1, 0.25, 0.5, 0.75, 1]) {
      const run = finiteWorldRun({ target, sample, epsilon });
      assert.equal(run.meetsTarget, run.risk <= epsilon,
        `meets iff risk <= epsilon at target ${target.join('')}, epsilon ${epsilon}`);
      if (run.risk === epsilon) {
        assert.equal(run.meetsTarget, true, 'a risk exactly equal to epsilon meets the target');
        tieCases += 1;
      }
      assert.equal(run.empiricalRisk, 0, 'the realizable setting always fits the sample exactly');
    }
  }
}
assert.ok(tieCases >= 100, `only ${tieCases} exact ties were exercised`);
record('the epsilon comparison at its tie');

// The empty observation sequence is a real state, not an error.
const beforeAnyData = finiteWorldRun({ target: [0, 0, 1, 1], sample: [], epsilon: 0.25 });
assert.deepEqual(beforeAnyData.rule, [0, 0, 0, 0], 'before any data the learner returns the all-zero rule');
assert.equal(beforeAnyData.risk, 0.5);
assert.equal(beforeAnyData.meetsTarget, false);
assert.deepEqual(beforeAnyData.unseen, [0, 1, 2, 3]);
record('the before-data state');

refuses(() => finiteWorldRun({ target: [0, 0, 1], sample: [0], epsilon: 0.25 }), 'a three-bit target');
refuses(() => finiteWorldRun({ target: [0, 0, 1, 1], sample: [4], epsilon: 0.25 }), 'an input outside 0..3');
refuses(() => finiteWorldRun({ target: [0, 0, 1, 1], sample: [0], epsilon: 0 }), 'epsilon zero');

// Occupancy probabilities: exhaustive enumeration at small n, then the packet.
for (const n of [1, 2, 4, 6]) {
  const byOccupancy = finiteWorldProbability({ target: [0, 0, 1, 1], n, epsilon: 0.25 });
  const byEnumeration = finiteWorldByEnumeration([0, 0, 1, 1], n, 0.25);
  close(byOccupancy.failure, byEnumeration, `four-input failure at n=${n}`, 1e-12);
  const total = byOccupancy.states.reduce((sum, entry) => sum + entry.probabilityValue, 0);
  close(total, 1, `the reachable states at n=${n} carry all the mass`);
  record('occupancy against exhaustive enumeration');
}
recorded.finite_world.forEach(row => {
  const mine = finiteWorldProbability({ target: row.target, n: row.n, epsilon: row.epsilon });
  assert.equal(mine.failureExact, row.failure_fraction_exact, `exact failure fraction at n=${row.n}`);
  close(mine.failure, row.failure_probability, `failure probability at n=${row.n}`);
  close(mine.bound.raw, row.bound_raw, `raw bound at n=${row.n}`);
  close(mine.bound.clipped, row.bound_clipped, `clipped bound at n=${row.n}`);
  assert.equal(mine.states.length, row.states.length, `reachable states at n=${row.n}`);
  assert.equal(mine.bound.vacuous, row.bound_raw >= 1, `vacuity flag at n=${row.n}`);
  assert.ok(mine.failure <= row.bound_raw, `the exact probability respects the bound at n=${row.n}`);
  record('four-input world against the packet');
});
assert.equal(finiteWorldProbability({ target: [0, 0, 1, 1], n: 4, epsilon: 0.25 }).failureExact, '1/16');
assert.equal(finiteWorldProbability({ target: [0, 0, 1, 1], n: 24, epsilon: 0.25 }).failureExact, '1/16777216');
const zeroTargetNull = finiteWorldProbability({ target: [0, 0, 0, 0], n: 4, epsilon: 0.25 });
assert.equal(zeroTargetNull.failureExact, recorded.finite_world_zero_target_null.failure_fraction_exact);
assert.equal(zeroTargetNull.failureExact, '0', 'the all-zero target never fails');
assert.equal(zeroTargetNull.states.length, recorded.finite_world_zero_target_null.states.length);
assert.ok(zeroTargetNull.bound.raw > 1, 'while the class-level bound at that n is still vacuous');
record('the changed-target null');

// n = 0 is mathematically meaningful and must not be a special case.
const atZero = finiteWorldProbability({ target: [0, 0, 1, 1], n: 0, epsilon: 0.25 });
assert.equal(atZero.states.length, 1, 'at n=0 exactly one state is reachable');
assert.equal(atZero.states[0].probability, '1');
assert.equal(atZero.failureExact, '1', 'and the all-zero rule fails for this target with certainty');
record('the n=0 edge');

// Exact rational arithmetic, on its own.
assert.equal(fractionText(fractionAdd(fraction(1n, 3n), fraction(1n, 6n))), '1/2');
assert.equal(fractionText(fractionMultiply(fraction(2n, 3n), fraction(3n, 4n))), '1/2');
assert.equal(fractionCompare(fraction(1n, 3n), fraction(2n, 6n)), 0, 'equal fractions compare equal');
assert.equal(fractionCompare(fraction(1n, 3n), fraction(1n, 2n)), -1);
close(fractionToNumber(fraction(1n, 16777216n)), 1 / 16777216, 'a very small fraction survives conversion');
assert.equal(fractionText(fraction(-2n, -4n)), '1/2', 'a doubly negative fraction normalises');
assert.equal(fractionText(fraction(0n, 5n)), '0');
refuses(() => fraction(1n, 0n), 'a zero denominator');
record('exact rational arithmetic');

/* ============================== elimination trace behind figure 2 */

const trace = eliminationTrace({ target: [0, 1, 1, 0], sample: [1, 3, 0] });
assert.equal(trace.steps.length, 4, 'one step before any observation and one after each');
assert.equal(trace.steps[0].survivors, 16, 'nothing is eliminated before anything is seen');
trace.steps.forEach((step, index) => {
  assert.equal(step.rules.length, 16, 'every step lists the whole predeclared class');
  assert.equal(step.rules.filter(entry => entry.consistent).length, step.survivors,
    'the survivor count matches the flags the table renders');
  if (index > 0) {
    assert.ok(step.survivors <= trace.steps[index - 1].survivors,
      'survivors never increase when another observation arrives');
  }
  const firstSurvivor = step.rules.find(entry => entry.consistent);
  assert.deepEqual(step.selected.rule, firstSurvivor.rule, 'the selected rule is the first survivor');
  assert.equal(step.selected.risk, finiteWorldRisk(step.selected.rule, trace.target).risk);
  record('elimination step');
});
assert.equal(trace.steps[3].survivors, 2,
  'three distinct inputs leave two rules, differing only at the unseen input');
const fullyObserved = eliminationTrace({ target: [0, 1, 1, 0], sample: [0, 1, 2, 3] });
assert.equal(fullyObserved.steps[4].survivors, 1, 'observing everything leaves exactly one rule');
assert.deepEqual(fullyObserved.steps[4].selected.rule, [0, 1, 1, 0], 'and it is the target');
assert.equal(fullyObserved.steps[4].selected.risk, 0);
record('the elimination endpoint');

/* ================================================= §3-§7 · the bounds */

close(finiteRadius(25, 500, 0.05), recorded.finite_family.k25_n500_delta005, 'K=25 radius against the packet');
close(finiteRadius(25, 500, 0.05), Math.sqrt(Math.log(1000) / 1000),
  'and against the destination note\'s own expression sqrt(log(1000)/1000)');
close(finiteRadius(25, 500, 0.05), 0.08311291, 'and against the note\'s rounded .08311291', 1e-7);
close(finiteFamilyRadii.selection, finiteRadius(25, 500, 0.05), 'the exported value is the same expression');
close(finiteRadius(1, 500, 0.05), recorded.finite_family.single_n500_delta005, 'the single-rule radius');
close(finiteRadius(25, 2000, 0.05), recorded.finite_family.k25_n2000_delta005, 'the quadrupled-sample radius');
close(finiteRadius(12, 800, 0.02), recorded.practice.k12_n800_delta002, 'the practice radius');
close(finiteRadius(12, 800, 0.02), Math.sqrt(Math.log(1200) / 1600), 'by its own closed form');
// A second route: r is where 2K exp(-2 n r^2) equals delta.
for (const [k, n, delta] of [[1, 500, 0.05], [3, 500, 0.05], [25, 500, 0.05], [12, 800, 0.02], [25, 2000, 0.05]]) {
  const r = finiteRadius(k, n, delta);
  close(2 * k * Math.exp(-2 * n * r * r), delta, `inverting Hoeffding at K=${k}, n=${n}`, 1e-9);
  assert.ok(r > 0 && r < 1, 'a two-sided radius on a 0-1 loss lies in (0, 1) at these sizes');
  record('finite radius inverted');
}
assert.ok(finiteRadius(25, 500, 0.05) > finiteRadius(1, 500, 0.05), 'selection costs more than a fixed rule');
assert.ok(finiteRadius(3, 500, 0.05) < finiteRadius(25, 500, 0.05),
  'and claiming K=3 would claim a narrower band than the search earns');
close(finiteRadius(25, 2000, 0.05) * 2, finiteRadius(25, 500, 0.05), 'quadrupling n halves the radius exactly');
refuses(() => finiteRadius(0, 500, 0.05), 'K = 0');
refuses(() => finiteRadius(25, 0, 0.05), 'n = 0');
refuses(() => finiteRadius(25, 500, 1), 'delta = 1');
refuses(() => finiteRadius(25, 500, 0), 'delta = 0');
record('the finite-family radius');

recorded.bounds.forEach(row => {
  close(vcRadius(row.d, row.n, row.delta), row.uniform_radius_raw, `VC radius at d=${row.d}, n=${row.n}`);
  record('VC radius against the packet');
});
// Second route: rebuild the expression from exp/log identities.
for (const d of [1, 2, 10]) {
  for (const n of [100, 1000, 10000, 100000]) {
    const r = vcRadius(d, n, 0.05);
    const rebuilt = Math.sqrt((32 / n) * (Math.log((Math.E * n / d) ** d) + Math.log(1 / (0.05 / 8))));
    close(r, rebuilt, `VC radius rebuilt at d=${d}, n=${n}`, 1e-12);
    assert.ok(r > 0);
    record('VC radius rebuilt');
  }
  for (let index = 1; index < 4; index += 1) {
    const sizes = [100, 1000, 10000, 100000];
    assert.ok(vcRadius(d, sizes[index], 0.05) < vcRadius(d, sizes[index - 1], 0.05),
      `the radius falls with n at d=${d}`);
  }
  record('monotone in n');
}
for (const n of [100, 1000, 10000, 100000]) {
  assert.ok(vcRadius(10, n, 0.05) > vcRadius(2, n, 0.05) && vcRadius(2, n, 0.05) > vcRadius(1, n, 0.05),
    `the radius rises with d at n=${n}`);
  assert.ok(vcRadius(2, n, 0.01) > vcRadius(2, n, 0.05), `and rises as delta falls at n=${n}`);
}
assert.ok([1, 2, 10].every(d => vcRadius(d, 100, 0.05) > 1),
  'every displayed dimension is vacuous at n=100, which is the point of the figure');
assert.ok(vcRadius(1, 100000, 0.05) < 1 && vcRadius(2, 100000, 0.05) < 1,
  'and both small dimensions say something by n=100,000');
close(vcRadius(2, 2000, 0.05), recorded.practice.uniform_bound_n2000_d2, 'the practice radius');
refuses(() => vcRadius(2, 1, 0.05), 'n below d');
refuses(() => vcRadius(0, 100, 0.05), 'd = 0, where the displayed form does not apply');
refuses(() => vcRadius(2, 100, 1.5), 'delta above 1');
record('the explicit VC radius');

recorded.realizable_sample_bounds.forEach(row => {
  assert.equal(realizableVcSampleBound(row.d, row.epsilon, row.delta), row.sufficient_n,
    `realizable sufficient size at d=${row.d}, epsilon=${row.epsilon}`);
  record('realizable sufficient size against the packet');
});
assert.deepEqual(realizableSufficientTable().filter(row => row.d === 2).map(row => row.sufficientN),
  [482, 1124, 2568], 'the three sizes the prose quotes for d=2');
// Second route: check that the returned integer is the smallest one satisfying
// both terms of the max, which is what "sufficient" means here.
for (const d of [1, 2, 10]) {
  for (const epsilon of [0.2, 0.1, 0.05]) {
    const n = realizableVcSampleBound(d, epsilon, 0.05);
    const requirement = Math.max((4 / epsilon) * Math.log2(2 / 0.05),
      ((8 * d) / epsilon) * Math.log2(13 / epsilon));
    assert.ok(n >= requirement, `n=${n} meets the requirement ${requirement}`);
    assert.ok(n - 1 < requirement, `and n-1 does not: it is the smallest integer that does`);
    record('sufficient size is the ceiling of its own requirement');
  }
}
assert.equal(finiteRealizableSampleBound(32, 0.05, 0.01), recorded.finite_family.realizable_k32_eps005_delta001);
assert.equal(finiteRealizableSampleBound(32, 0.05, 0.01), 162, 'the 162 the prose quotes');
assert.ok((Math.log(32) + Math.log(100)) / 0.05 > 161 && (Math.log(32) + Math.log(100)) / 0.05 < 162,
  'and 161 examples would not satisfy the same inequality, so the ceiling matters');
// O4: the prose quotes "about 161.42"; pin that literal against the computation.
assert.equal(((Math.log(32) + Math.log(100)) / 0.05).toFixed(2), '161.42',
  'the two-decimal figure the prose quotes');
record('the finite realizable condition');

const vacuousBound = finiteClassFailureBound(16, 4, 0.25);
close(vacuousBound.raw, recorded.finite_world[2].bound_raw, 'the raw union bound at n=4');
assert.equal(vacuousBound.clipped, 1, 'clipped to a probability it says only "at most 1"');
assert.equal(vacuousBound.vacuous, true);
const usefulBound = finiteClassFailureBound(16, 24, 0.25);
assert.equal(usefulBound.vacuous, false);
assert.ok(usefulBound.raw < 0.05, 'and at n=24 it is below delta=.05');
assert.equal(finiteClassFailureBound(16, 0, 0.25).raw, 16, 'at n=0 the bound is the class size');
record('the finite-class failure bound');

recorded.simulation.rows.forEach(row => {
  const strip = twoStripBound(row.epsilon, row.n);
  close(strip.clipped, row.distribution_specific_failure_bound, `two-strip bound at n=${row.n}`);
  close(vcRadius(2, row.n, 0.05), row.uniform_vc_radius_raw_delta005, `VC radius at n=${row.n}`);
  assert.ok(row.failure_fraction <= strip.clipped + 1e-12,
    `the observed failure fraction at n=${row.n} respects the bound, which is an observation`);
  record('two-strip bound against the packet');
});
assert.equal(twoStripBound(0.1, 0).raw, 2, 'at n=0 the union of two events bounds nothing below 2');
assert.equal(twoStripBound(0.1, 0).vacuous, true);
assert.ok(twoStripBound(0.1, 200).raw > 0, 'and it is strictly positive where zero failures were observed');
/* S7: `close()` scales by max(1, |expected|), so an explicit 1e-6 against a
   value of 7.0e-5 was a band of about +/-1.4% -- it would have accepted
   0.0000691. The full double is compared at the 1e-12 default instead, which is
   tight enough to detect the class of change this guard exists for. */
close(twoStripBound(0.1, 200).raw, 0.000070105332497658006,
  'the value the prose quotes beside those zero failures');
assert.equal(twoStripBound(0.1, 200).raw.toFixed(10), '0.0000701053',
  'and it prints as the ten decimals the page shows');
record('the two-strip bound edges');

/* ============================= §8 · the interval world and its exact risk */

// Symmetric-difference risk by a second route, over a grid that covers
// disjoint, touching, nested, overlapping, empty and zero-width cases.
let riskCases = 0;
const grid = [0, 0.1, 0.25, 0.3, 0.5, 0.65, 0.7, 0.9, 1];
for (const a of grid) {
  for (const b of grid) {
    if (b < a) continue;
    for (const left of grid) {
      for (const right of grid) {
        if (right < left) continue;
        const direct = intervalRisk([left, right], [a, b]);
        const byUnion = symmetricDifferenceByUnion([left, right], [a, b]);
        close(direct, byUnion, `risk at target [${a},${b}] and fit [${left},${right}]`, 1e-12);
        const segments = riskSegments([left, right], [a, b]);
        const segmentTotal = segments.reduce((sum, segment) => sum + (segment.to - segment.from), 0);
        close(segmentTotal, direct, 'the drawn segments add up to the risk they claim to show', 1e-12);
        assert.ok(direct >= -1e-15, 'a length is never negative');
        segments.forEach(segment => assert.ok(segment.to >= segment.from, 'and no segment runs backwards'));
        riskCases += 1;
      }
    }
  }
}
assert.ok(riskCases >= 1000, `only ${riskCases} risk geometries were checked`);
for (const a of grid) {
  for (const b of grid) {
    if (b < a) continue;
    close(intervalRisk(null, [a, b]), b - a, 'the empty rule costs exactly the target mass');
    const segments = riskSegments(null, [a, b]);
    close(segments.reduce((sum, segment) => sum + (segment.to - segment.from), 0), b - a,
      'and its drawn segment says the same');
  }
}
assert.equal(intervalRisk([0.3, 0.7], [0.3, 0.7]), 0, 'an exact fit costs nothing');
assert.equal(riskSegments([0.3, 0.7], [0.3, 0.7]).length, 0, 'and draws nothing');
assert.equal(intervalRisk([0.5, 0.5], [0.5, 0.5]), 0, 'two coincident zero-width intervals agree');
refuses(() => intervalRisk([0.7, 0.3], [0.3, 0.7]), 'a reversed fit');
refuses(() => intervalRisk([0.3, 0.7], [0.7, 0.3]), 'a reversed target');
record('symmetric-difference risk by two routes');

const worked = intervalExperiment(fixtures.interval);
close(worked.risk, recorded.intervals.base.true_error, 'the worked fit\'s risk');
assert.deepEqual(worked.interval, recorded.intervals.base.interval);
assert.deepEqual(worked.labels, recorded.intervals.base.labels);
assert.equal(worked.empiricalRisk, 0);
const nearEdges = intervalExperiment(fixtures.intervalCloserEdges);
close(nearEdges.risk, recorded.intervals.closer_edges.true_error, 'the near-edge fit\'s risk');
assert.ok(nearEdges.risk < worked.risk, 'a positive nearer the edge lowers the risk');
const negativeNull = intervalExperiment(fixtures.intervalNegativeNull);
assert.deepEqual(negativeNull.interval, worked.interval, 'exterior negatives move neither the fit');
assert.equal(negativeNull.risk, worked.risk, 'nor the risk: an exact null, not a small change');
assert.equal(movementOf(worked.risk, negativeNull.risk).outcome, 'unchanged',
  'and the grading rule says unchanged for it');
assert.equal(movementOf(worked.risk, negativeNull.risk).difference, 0);
const noPositives = intervalExperiment(fixtures.intervalNoPositives);
assert.equal(noPositives.interval, null, 'a sample with no positive returns the empty rule');
assert.equal(noPositives.empty, true);
close(noPositives.risk, recorded.intervals.no_positives.true_error, 'whose risk is the target mass');
close(noPositives.risk, 0.4, 'which is .4 here');
assert.equal(noPositives.empiricalRisk, 0, 'and it still fits every observed label');
record('the worked interval fixtures');

[['changed_target_interval', fixtures.intervalPractice],
  ['changed_target_positive', fixtures.intervalPracticePositive],
  ['changed_target_negative_null', fixtures.intervalPracticeNegativeNull]].forEach(([key, setup]) => {
  const experiment = intervalExperiment(setup);
  close(experiment.risk, recorded.practice[key].true_error, `practice fixture ${key}`);
  assert.deepEqual(experiment.interval, recorded.practice[key].interval, `practice fit ${key}`);
  record('practice interval fixture');
});
const practiceBase = intervalExperiment(fixtures.intervalPractice);
const practicePositive = intervalExperiment(fixtures.intervalPracticePositive);
const practiceNegative = intervalExperiment(fixtures.intervalPracticeNegativeNull);
assert.equal(movementOf(practiceBase.risk, practicePositive.risk).outcome, 'falls',
  'adding .26 lowers the practice risk');
assert.equal(movementOf(practiceBase.risk, practiceNegative.risk).outcome, 'unchanged',
  'while adding .99 changes nothing at all');
record('the practice movement outcomes');

// The manual-label mode: an inconsistent request is a result, not an error.
const contradiction = manualIntervalRequest(fixtures.manualContradiction);
assert.equal(contradiction.feasible, false, 'positives at .2 and .8 with a negative at .5 have no interval');
assert.equal(contradiction.witness, null);
assert.ok(contradiction.obstruction, 'and the obstruction is named');
assert.equal(recorded.intervals.inconsistent.feasible, false, 'which is what the packet recorded');
const consistentRequest = manualIntervalRequest({ points: [0.2, 0.5, 0.8], labels: [1, 1, 0] });
assert.equal(consistentRequest.feasible, true);
assert.deepEqual(consistentRequest.witness.interval, [0.2, 0.5]);
record('the manual-label mode');

// Labels really do come from the target, including at both closed endpoints.
assert.deepEqual(labelsFromTarget([0.3, 0.7, 0.29999, 0.70001], [0.3, 0.7]), [1, 1, 0, 0],
  'the target interval is closed at both ends');
assert.deepEqual(labelsFromTarget([0.5], [0.5, 0.5]), [1], 'a zero-width target still labels its own point');
const zeroWidth = intervalExperiment({ points: [0.2, 0.5, 0.8], target: [0.5, 0.5], epsilon: 0.1 });
assert.equal(zeroWidth.risk, 0, 'a zero-width target is fitted exactly by the point on it');
refuses(() => labelsFromTarget([0.5], [0.7, 0.3]), 'an inverted target');
refuses(() => intervalExperiment({ points: [], target: [0.3, 0.7], epsilon: 0.1 }), 'an empty sample');
record('the label generator at its boundaries');

/* ===================================== the unchanged rule, as an iff */

let movementCases = 0;
for (const before of [0, 1e-13, 1e-12, 1e-6, 0.1, 0.5, 1]) {
  for (const after of [0, 1e-13, 1e-12, 1e-6, 0.1, 0.5, 1]) {
    const movement = movementOf(before, after);
    const same = Math.abs(after - before) <= unchangedTolerance;
    assert.equal(movement.outcome === 'unchanged', same,
      `unchanged iff the difference is within tolerance: ${before} to ${after}`);
    if (!same) {
      assert.equal(movement.outcome, after > before ? 'rises' : 'falls', 'and otherwise names the direction');
      assert.equal(movement.difference, after - before, 'reporting the real difference');
    } else {
      assert.equal(movement.difference, 0, 'an unchanged verdict reports exactly zero, not a residue');
    }
    movementCases += 1;
  }
}
assert.equal(movementOf(0.1, 0.1).outcome, 'unchanged', 'identical values are unchanged');
assert.equal(movementOf(0, -0).outcome, 'unchanged', 'and so are positive and negative zero');
/* The boundary is tested at values whose difference is EXACTLY the tolerance.
 * Writing 0.1 and 0.1 + 1e-12 would test floating-point subtraction instead:
 * that difference is 1.0000000827e-12, which is outside the tolerance and would
 * make a correct rule look wrong. */
assert.equal(movementOf(0, unchangedTolerance).outcome, 'unchanged', 'the tolerance boundary is inclusive');
assert.equal(movementOf(0, 2 * unchangedTolerance).outcome, 'rises', 'just beyond it is not');
assert.equal(movementOf(0, -unchangedTolerance).outcome, 'unchanged', 'in both directions');
assert.equal(movementOf(0, -2 * unchangedTolerance).outcome, 'falls');
assert.ok(movementCases >= 49);
refuses(() => movementOf(Number.NaN, 0.1), 'a non-finite earlier value');
refuses(() => movementOf(0.1, Number.POSITIVE_INFINITY), 'a non-finite later value');
record('the unchanged rule as an iff');

/* ============================================= §5 · half-plane geometry */

let planeCases = 0;
Object.entries(pacData.halfPlaneWitnesses).forEach(([name, block]) => {
  nonEmpty(block.realized, undefined, `${name} witnesses`);
  assert.equal(block.count, block.realized.length, `${name}: the count matches the list`);
  assert.equal(block.realized.length + block.infeasible.length, 2 ** block.points.length,
    `${name}: realizable and infeasible labelings partition all 2^n`);
  block.realized.forEach(entry => {
    const predicted = halfPlanePredictions(block.points, entry.w, entry.b);
    assert.deepEqual(predicted, entry.labels, `${name}: the witness does not realize ${entry.labels.join('')}`);
    const margins = block.points.map((point, index) =>
      (entry.labels[index] === 1 ? 1 : -1) * signedMargin(point, entry.w, entry.b));
    close(Math.min(...margins), entry.minimumSignedMargin, `${name}: recorded minimum margin`, 1e-9);
    assert.ok(Math.min(...margins) >= 1 - 1e-8, `${name}: every witness keeps the linear program's own floor`);
    const panel = planePanelGeometry({ points: block.points, labels: entry.labels, witness: entry });
    assert.equal(panel.agrees, true, 'and the drawn panel agrees with it');
    close(panel.unitsPerPixel, (panel.box.maxX - panel.box.minX) / (panel.size - 2 * panel.margin),
      'the panel axes share one physical scale');
    if (panel.separator.kind === 'line') {
      // The drawn segment really lies on the line it claims to draw.
      [panel.separator.from, panel.separator.to].forEach(pixel => {
        const modelX = panel.x.invert(pixel[0]);
        const modelY = panel.y.invert(pixel[1]);
        close(signedMargin([modelX, modelY], entry.w, entry.b), 0, `${name}: a drawn endpoint is on the line`, 1e-8);
        assert.ok(modelX >= panel.box.minX - 1e-9 && modelX <= panel.box.maxX + 1e-9, 'and inside the box');
        assert.ok(modelY >= panel.box.minY - 1e-9 && modelY <= panel.box.maxY + 1e-9, 'in both coordinates');
      });
    } else {
      // "No attribute" is a case to examine, not to skip: these are the
      // constant-sign rules, and they must be the all-positive or all-negative
      // labelings and nothing else.
      assert.equal(panel.separator.kind, 'constant',
        `${name}: a witness with no drawn line must be a constant-sign rule`);
      const distinct = new Set(entry.labels);
      assert.equal(distinct.size, 1, `${name}: only a uniform labeling may draw no line`);
      assert.equal(panel.separator.sign, entry.labels[0], 'and the constant sign matches the labeling');
    }
    planeCases += 1;
  });
  block.infeasible.forEach(labels => {
    // The recorded infeasible labelings are exactly the ones the packet's own
    // program rejected; here they are re-checked against the geometry the prose
    // argues, rather than re-run through a solver.
    assert.equal(new Set(labels).size, 2, `${name}: an infeasible labeling uses both labels`);
    planeCases += 1;
  });
  record(`half-plane witnesses for ${name}`);
});
assert.ok(planeCases >= 40, `only ${planeCases} half-plane cases were checked`);
assert.equal(pacData.halfPlaneWitnesses.triangle.count, 8, 'a noncollinear triple is shattered');
assert.equal(pacData.halfPlaneWitnesses.collinear.count, 6, 'a collinear triple is not');
assert.equal(pacData.halfPlaneWitnesses.square.count, 14);
assert.equal(pacData.halfPlaneWitnesses.interior.count, 14);
assert.deepEqual(pacData.halfPlaneWitnesses.square.infeasible, [[0, 1, 0, 1], [1, 0, 1, 0]],
  'the square loses exactly the two alternating labelings');
assert.deepEqual(pacData.halfPlaneWitnesses.interior.infeasible, [[0, 0, 0, 1], [1, 1, 1, 0]],
  'and the interior configuration loses exactly the two that isolate the interior point');
record('the shattering counts');

// The crossing diagonals the figure draws really do cross.
const squarePanel = planePanelGeometry({
  points: pacData.halfPlaneWitnesses.square.points, labels: [0, 1, 0, 1], witness: null,
});
const diagonals = diagonalGeometry({
  points: pacData.halfPlaneWitnesses.square.points, labels: [0, 1, 0, 1], panel: squarePanel,
});
assert.equal(diagonals.crosses, true, 'the positive and negative diagonals of the square intersect');
assert.equal(diagonals.positive.length, 2);
assert.equal(diagonals.negative.length, 2);
assert.equal(segmentsCross([0, 0], [1, 1], [1, 0], [0, 1]), true, 'the crossing test on a known crossing');
assert.equal(segmentsCross([0, 0], [1, 0], [0, 1], [1, 1]), false, 'and on two parallel segments');
// The interior point really is a convex combination of the three around it.
const combination = interiorCombination({
  triangle: pacData.halfPlaneWitnesses.interior.points.slice(0, 3),
  interior: pacData.halfPlaneWitnesses.interior.points[3],
});
assert.equal(combination.inside, true, 'the fourth point lies inside the triangle');
close(combination.weights.reduce((sum, weight) => sum + weight, 0), 1, 'its weights sum to one');
assert.ok(combination.weights.every(weight => weight > 0), 'and all three are strictly positive');
close(combination.reconstructed[0], pacData.halfPlaneWitnesses.interior.points[3][0], 'reconstructed x');
close(combination.reconstructed[1], pacData.halfPlaneWitnesses.interior.points[3][1], 'reconstructed y');
const outsideCombination = interiorCombination({ triangle: [[0, 0], [2, 0], [0, 2]], interior: [3, 3] });
assert.equal(outsideCombination.inside, false, 'a point outside the triangle is reported as outside');
refuses(() => interiorCombination({ triangle: [[0, 0], [1, 0], [2, 0]], interior: [1, 1] }),
  'three collinear surrounding points');
record('the impossibility geometry');

// The line clipper itself, including the cases that draw nothing.
const box = { minX: -1, maxX: 1, minY: -1, maxY: 1 };
assert.equal(lineSegmentInBox([0, 0], 1, box).kind, 'constant', 'a zero normal draws no line');
assert.equal(lineSegmentInBox([0, 0], 1, box).sign, 1, 'and reports the constant it gives');
assert.equal(lineSegmentInBox([0, 0], -1, box).sign, 0);
assert.equal(lineSegmentInBox([1, 0], 0, box).kind, 'line', 'a vertical line through the origin is drawn');
assert.equal(lineSegmentInBox([0, 1], 0, box).kind, 'line', 'and so is a horizontal one');
assert.equal(lineSegmentInBox([1, 0], 5, box).kind, 'outside', 'a line outside the box is reported as outside');
refuses(() => lineSegmentInBox([1, 1], 0, { minX: 0, maxX: 0, minY: -1, maxY: 1 }), 'a degenerate box');
record('the line clipper');

/* ====================================== §10 · the one-parameter sine class */

recorded.sine_four_labels.forEach(row => {
  const witness = sineWitness(row.labels);
  assert.equal(witness.rText, row.r_fraction, `sine r for ${row.labels.join('')}`);
  assert.deepEqual(witness.cycles, row.fractional_cycles, `sine cycles for ${row.labels.join('')}`);
  close(witness.theta, row.theta_approx, `sine theta for ${row.labels.join('')}`);
  assert.deepEqual(witness.predicted, row.labels, 'and it realizes the labels it was built for');
  // The real second route: the class is defined by the sign of sin(theta x).
  witness.points.forEach((x, index) => {
    const value = Math.sin(witness.theta * x);
    assert.equal(value >= 0 ? 1 : 0, row.labels[index],
      `sin(theta x) at x=${x} must have the requested sign for ${row.labels.join('')}`);
    assert.ok(Math.abs(value) > 1e-6,
      'and the guard bits must keep it away from the boundary where the sign is undecided');
  });
  record('sine witness against the actual sine');
});
assert.equal(sineWitness([1, 0, 1, 1]).rText, '17/64', 'the r the prose quotes');
close(sineWitness([1, 0, 1, 1]).theta, 1.668971, 'and its theta', 1e-6);
assert.equal(sineWitness([1, 0, 1, 1]).rBits, '.010001', 'with the bit string the figure shows');
// Beyond four labels: the construction has to keep working, and the exact
// arithmetic is what makes that checkable rather than plausible.
for (let n = 1; n <= 8; n += 1) {
  const labels = Array.from({ length: n }, (_unused, index) => (index * 3 + 1) % 2);
  const witness = sineWitness(labels);
  assert.deepEqual(witness.predicted, labels, `the construction realizes ${labels.join('')}`);
  assert.equal(witness.points[n - 1], 2 ** (n - 1), 'and the input range doubles with each extra label');
  record('sine construction beyond four labels');
}
refuses(() => sineWitness([]), 'an empty label list');
refuses(() => sineWitness([0, 2]), 'a label that is not 0 or 1');
record('the sine class');

/* ============================================= scales and drawn geometry */

const linear = linearScale({ domain: [0, 1], range: [0, 100] });
close(linear(0), 0); close(linear(1), 100); close(linear(0.25), 25);
close(linear.invert(25), 0.25, 'the inverse really inverts');
const logarithmic = logScale({ domain: [1, 1000], range: [0, 300] });
close(logarithmic(1), 0); close(logarithmic(10), 100); close(logarithmic(1000), 300);
close(logarithmic.invert(200), 100, 'and so does the logarithmic inverse');
refuses(() => linearScale({ domain: [1, 1], range: [0, 100] }), 'a degenerate domain');
refuses(() => logScale({ domain: [0, 100], range: [0, 300] }), 'a log scale starting at zero');
refuses(() => logScale({ domain: [1, 100], range: [0, 300] })(0), 'placing zero on a log axis');
record('the scales');

const meters = meterGeometry();
assert.equal(meters.totalDraws, 4);
assert.equal(meters.exceedingDraws, 1, 'exactly one schematic draw exceeds the default cutoff');
meters.rows.forEach(row => {
  close(row.barWidth, (row.populationError / meters.axisMaximum) * meters.width, 'a meter bar is proportional');
  assert.equal(row.exceeds, row.populationError > meters.epsilon, 'and its flag is the comparison itself');
  assert.ok(row.barWidth >= 0 && row.barWidth <= meters.width, 'and it stays inside its track');
  record('meter row');
});
close(meterGeometry({ epsilon: 0.05 }).cutoffX, (0.05 / 0.25) * meters.width, 'the cutoff moves with epsilon');
assert.equal(meterGeometry({ epsilon: 0.05 }).exceedingDraws, 3, 'and more rows fail at a tighter epsilon');
assert.equal(meterGeometry({ epsilon: 0.15 }).exceedingDraws, 1);
refuses(() => meterGeometry({ epsilon: 0 }), 'a zero epsilon');
record('the schematic meters');

const band = candidateBandGeometry();
assert.equal(band.k, candidateErrorCounts.length);
assert.equal(band.k, 25, 'the figure draws the manuscript\'s twenty-five candidates');
assert.equal(band.n, 500);
close(band.radius, finiteRadius(25, 500, 0.05), 'and its band half-width is the theorem\'s own radius');
close(band.singleRadius, finiteRadius(1, 500, 0.05));
close(band.shrunkRadius, finiteRadius(3, 500, 0.05));
band.rows.forEach((row, index) => {
  close(row.empirical, candidateErrorCounts[index] / 500, 'each observed error is its count over 500');
  close(row.low, row.empirical - band.radius, 'the band runs a radius below');
  close(row.high, row.empirical + band.radius, 'and a radius above');
  close(row.centreX, band.scale(row.empirical), 'the drawn dot sits at the scaled observed error');
  close(row.lowX, band.scale(row.low), 'and the drawn ends at the scaled band ends');
  close(row.highX, band.scale(row.high));
  assert.ok(row.lowX >= -1e-9 && row.highX <= band.width + 1e-9, 'every band stays inside the drawing');
  assert.ok(row.y > 0 && row.y < band.height, 'and inside it vertically');
  assert.equal(row.selected, row.index === band.best.index, 'exactly the lowest row is marked selected');
  record('candidate band row');
});
assert.equal(band.rows.filter(row => row.selected).length, 1, 'one row is selected, not none and not two');
close(band.best.empirical, 0.262);
close(band.runnerUp.empirical, 0.272);
close(band.separation, band.runnerUp.empirical - band.best.empirical);
assert.equal(band.rankingCertified, false,
  'the two best candidates are closer than twice the radius, so the ranking is not settled');
assert.ok(band.separation < 2 * band.radius, 'which is the inequality that claim rests on');
assert.ok(band.shrunkRadius < band.radius, 'and the K=3 band the figure crosses out is narrower');
refuses(() => candidateBandGeometry({ counts: [600], n: 500 }), 'an error count above n');
refuses(() => candidateBandGeometry({ counts: [10] }), 'a family of one, where selection is not a question');
record('the candidate bands');

const growthPlot = growthPlotGeometry();
assert.equal(growthPlot.series.length, 4);
growthPlot.series.forEach(series => {
  nonEmpty(series.points, 10, `${series.key} points`);
  series.points.forEach(point => {
    close(point.x, growthPlot.x(point.n), 'x comes from the scale');
    close(point.y, growthPlot.y(point.value), 'and so does y');
    assert.ok(point.x >= growthPlot.padding.left - 1e-9
      && point.x <= growthPlot.width - growthPlot.padding.right + 1e-9, 'inside the frame horizontally');
    assert.ok(point.y >= growthPlot.padding.top - 1e-9
      && point.y <= growthPlot.height - growthPlot.padding.bottom + 1e-9, 'and vertically');
  });
  for (let index = 1; index < series.points.length; index += 1) {
    assert.ok(series.points[index].value >= series.points[index - 1].value,
      `${series.key} never decreases with n`);
    assert.ok(series.points[index].y <= series.points[index - 1].y,
      `${series.key} is drawn rising, because the y axis is inverted`);
  }
  record(`growth series ${series.key}`);
});
growthPlot.yTicks.forEach(tick => close(tick.y, growthPlot.y(tick.value), 'a y tick sits at its own value'));
growthPlot.xTicks.forEach(tick => close(tick.x, growthPlot.x(tick.value), 'and so does an x tick'));
nonEmpty(growthPlot.yTicks, 4, 'log ticks');
record('the growth plot');

const boundPlot = boundPlotGeometry();
assert.equal(boundPlot.series.length, 3);
close(boundPlot.vacuousFrom, boundPlot.y(1), 'the shaded region starts exactly at radius 1');
close(boundPlot.vacuousTo, boundPlot.y(boundPlot.y.domain[1]), 'and runs to the top of the axis');
assert.ok(boundPlot.vacuousFrom > boundPlot.vacuousTo, 'the shaded rectangle has positive height');
boundPlot.series.forEach(series => {
  nonEmpty(series.points, 13, `d=${series.d} points`);
  series.points.forEach(point => {
    close(point.value, vcRadius(series.d, point.n, boundPlot.delta), 'every plotted value is the theorem');
    close(point.x, boundPlot.x(point.n));
    close(point.y, boundPlot.y(point.value));
    assert.ok(point.y >= boundPlot.padding.top - 1e-9, 'no point is drawn above the frame');
    assert.ok(point.y <= boundPlot.height - boundPlot.padding.bottom + 1e-9, 'or below it');
  });
  for (let index = 1; index < series.points.length; index += 1) {
    assert.ok(series.points[index].value < series.points[index - 1].value, `d=${series.d} falls with n`);
  }
  record(`bound series d=${series.d}`);
});
/* All three curves eventually leave the shaded region on this axis; what
 * separates them is WHEN. At n = 1,000 the d = 1 curve is already saying
 * something and the d = 10 curve is still saying nothing, and that contrast is
 * what the figure has to make visible. */
boundPlot.series.forEach(series => assert.equal(series.crossesOne, true,
  `the d=${series.d} curve does leave the vacuous region somewhere on this axis`));
assert.ok(vcRadius(1, 1000, 0.05) < 1, 'd=1 is informative by n=1,000');
assert.ok(vcRadius(10, 1000, 0.05) > 1, 'while d=10 is still vacuous there');
assert.ok(vcRadius(10, 100000, 0.05) < 1, 'and only becomes informative much later');
boundPlot.table.forEach(row => close(row.radius, vcRadius(row.d, row.n, row.delta), 'the printed table is the same expression'));
nonEmpty(boundPlot.table, boundGrid.dimensions.length * boundGrid.sizes.length, 'bound table rows');
record('the bound plot');

const stripDrawing = stripGeometry({ experiment: worked });
const usable = stripDrawing.width - 2 * stripDrawing.inset;
close(stripDrawing.target.x, stripDrawing.scale(0.3), 'the target band starts at its own coordinate');
close(stripDrawing.target.width, 0.4 * usable, 'and has its own width on the inset line');
close(stripDrawing.fitted.x, stripDrawing.scale(0.35));
assert.equal(stripDrawing.segments.length, 2, 'two missed strips are drawn for the worked fit');
close(stripDrawing.segments.reduce((sum, segment) => sum + (segment.to - segment.from), 0), worked.risk,
  'and the drawn strips are exactly the risk');
/* Everything drawn must sit INSIDE the inset line. Placing 0 at x = 0 put the
 * centred tick label half outside the viewBox; the inspector found eight such
 * labels on one number line, so containment is asserted here rather than left
 * to the browser to notice. */
/* The inset itself is floored FIRST, against a fixed minimum rather than against
 * whatever the module happens to be configured with. Comparing positions to
 * `drawing.inset` alone is vacuous at inset 0 -- every position clears it -- and
 * that is not hypothetical: a concurrent run left `inset = 0` in the module and
 * this whole block still passed while the browser inspector reported eighteen
 * labels outside their viewBox. A guard whose domain is set by the value it is
 * checking is not a guard.
 *
 * Twelve units is half the width of the widest edge label the line can carry
 * ("0.999" at roughly 4.8 units a character). */
const MINIMUM_INSET = 12;
const withinLineOf = drawing => {
  assert.ok(drawing.inset >= MINIMUM_INSET,
    `the number line's inset is ${drawing.inset}, below the ${MINIMUM_INSET} units the widest edge label needs; `
    + 'containment below would then be trivially satisfied');
  return (position, label) => assert.ok(
    position >= drawing.inset - 1e-9 && position <= drawing.width - drawing.inset + 1e-9,
    `${label} at ${position} falls outside the inset line [${drawing.inset}, ${drawing.width - drawing.inset}]`);
};
const withinLine = withinLineOf(stripDrawing);
stripDrawing.segments.forEach(segment => {
  close(segment.width, (segment.to - segment.from) * usable, 'each drawn width is its own length');
  withinLine(segment.x, 'a strip start');
  withinLine(segment.x + segment.width, 'a strip end');
});
assert.equal(stripDrawing.strips.length, 2, 'both epsilon/2 coverage strips are drawn');
stripDrawing.strips.forEach(strip => {
  close(strip.to - strip.from, worked.epsilon / 2, 'each of width epsilon/2');
  withinLine(strip.x, 'a coverage strip start');
});
assert.equal(stripDrawing.points.length, worked.points.length, 'every observation is drawn');
stripDrawing.points.forEach((point, index) => {
  close(point.position, stripDrawing.scale(point.x), 'at its own coordinate');
  withinLine(point.position, `the observation at ${point.x}`);
  assert.equal(point.label, worked.labels[index], 'carrying the label the target generated');
});
stripDrawing.ticks.forEach(tick => withinLine(tick.x, `the tick at ${tick.value}`));
/* The rows must be separated and in meaning order, and nothing may be drawn
 * above the viewBox. The first version put the observation labels at y = -3 and
 * piled four ribbons around one axis; that is the defect these assert. */
const rows = stripDrawing.rows;
const ordered = [
  ['pointLabelY', rows.pointLabelY],
  ['markY', rows.markY],
  ['targetY', rows.targetY],
  ['stripY', rows.stripY],
  ['segmentY', rows.segmentY],
  ['fittedY', rows.fittedY],
  ['axisY', rows.axisY],
  ['tickLabelY', rows.tickLabelY],
];
ordered.forEach(([name, value]) => {
  assert.ok(value >= 8, `${name} at ${value} would be clipped by the top of the viewBox`);
  assert.ok(value <= stripDrawing.height, `${name} at ${value} falls below the viewBox`);
});
for (let index = 1; index < ordered.length; index += 1) {
  assert.ok(ordered[index][1] > ordered[index - 1][1],
    `${ordered[index][0]} must sit below ${ordered[index - 1][0]}`);
}
const bandSpans = [
  ['target', rows.targetY, rows.targetHeight],
  ['coverage strips', rows.stripY, rows.stripHeight],
  ['disagreement', rows.segmentY, rows.segmentHeight],
  ['fitted', rows.fittedY, rows.fittedHeight],
];
for (let index = 1; index < bandSpans.length; index += 1) {
  const [name, top] = bandSpans[index];
  const [previousName, previousTop, previousHeight] = bandSpans[index - 1];
  assert.ok(top >= previousTop + previousHeight,
    `the ${name} row overlaps the ${previousName} row, which is what made this figure unreadable`);
}
assert.ok(rows.axisY >= rows.fittedY + rows.fittedHeight + 6,
  'the axis sits clear below every band it indexes');
assert.ok(rows.tickLabelY > rows.tickY, 'and its tick labels below the tick marks');
record('the number line rows are separated and ordered');

/* No two drawn coordinate labels may overlap, on any sample the lab can reach.
 * The worked sample alone would not have caught it: it is the near-edge sample,
 * with .31 beside .35 and .65 beside .69, that merged into "0.31|35". */
let labelCases = 0;
const labelFixtures = [
  ['worked', fixtures.interval],
  ['near the edges', fixtures.intervalCloserEdges],
  ['exterior negatives', fixtures.intervalNegativeNull],
  ['no positives', fixtures.intervalNoPositives],
  ['practice', fixtures.intervalPractice],
  ['practice plus a positive', fixtures.intervalPracticePositive],
  ['both extremes', { points: [0, 0.01, 0.5, 0.99, 1], target: [0.3, 0.7], epsilon: 0.1 }],
  ['four in a row', { points: [0.3, 0.31, 0.32, 0.33], target: [0.3, 0.7], epsilon: 0.1 }],
];
labelFixtures.forEach(([name, setup]) => {
  const drawing = stripGeometry({ experiment: intervalExperiment(setup) });
  assert.deepEqual(labelCollisions(drawing), [],
    `coordinate labels overlap on the ${name} sample`);
  drawing.points.filter(point => point.labelRow !== null).forEach(point => {
    assert.ok(point.labelRow === 0 || point.labelRow === 1, 'a label sits on one of the two rows');
    const y = drawing.rows.pointLabelY - point.labelRow * drawing.rows.pointLabelRowGap;
    assert.ok(y >= 8, `a label at y=${y} would be clipped by the top of the viewBox`);
  });
  labelCases += 1;
});
// A greedy two-row placement must actually use the second row somewhere, or it
// is not doing anything and the guard above is inert.
const crowded = stripGeometry({ experiment: intervalExperiment(fixtures.intervalCloserEdges) });
assert.ok(crowded.points.some(point => point.labelRow === 1),
  'the near-edge sample uses the second label row, so the staggering is doing work');
// And a genuinely impossible case suppresses rather than overlaps.
const dense = stripGeometry({
  experiment: intervalExperiment({
    points: [0.3, 0.305, 0.31, 0.315, 0.32, 0.325], target: [0.3, 0.7], epsilon: 0.1,
  }),
});
assert.deepEqual(labelCollisions(dense), [], 'a dense sample still draws no overlapping label');
assert.ok(dense.points.some(point => point.labelRow === null),
  'and suppresses the ones that cannot fit rather than stacking them');
assert.equal(labelCases, labelFixtures.length);
record('no two coordinate labels overlap on any reachable sample');
// The extreme cases the lab can actually reach: an observation at 0 and at 1.
const extremeDrawing = stripGeometry({
  experiment: intervalExperiment({ points: [0, 0.01, 0.5, 0.99, 1], target: [0.3, 0.7], epsilon: 0.1 }),
});
const withinExtremeLine = withinLineOf(extremeDrawing);
extremeDrawing.points.forEach(point => withinExtremeLine(point.position, `an extreme observation at ${point.x}`));
assert.equal(extremeDrawing.points.length, 5);
record('the number line stays inside its own viewBox');
const emptyDrawing = stripGeometry({ experiment: noPositives });
assert.equal(emptyDrawing.fitted, null, 'the empty rule draws no fitted band, and says so rather than drawing zero');
assert.equal(emptyDrawing.segments.length, 1, 'and its single missed strip is the whole target');
record('the strip drawing');

const simulationPlot = simulationPlotGeometry({ rows: pacData.simulation.rows });
nonEmpty(simulationPlot.bars, 5, 'simulation bars');
simulationPlot.bars.forEach((bar, index) => {
  const row = pacData.simulation.rows[index];
  close(bar.lowY, simulationPlot.y(row.riskQuantiles[0]));
  close(bar.medianY, simulationPlot.y(row.riskQuantiles[1]));
  close(bar.highY, simulationPlot.y(row.riskQuantiles[2]));
  close(bar.meanY, simulationPlot.y(row.meanTrueError));
  assert.ok(bar.lowY >= bar.medianY && bar.medianY >= bar.highY,
    'the drawn quantiles are ordered, on an inverted axis');
  assert.ok(row.riskQuantiles[0] <= row.riskQuantiles[1] && row.riskQuantiles[1] <= row.riskQuantiles[2],
    'because the recorded quantiles are ordered');
  assert.equal(bar.failureCount, row.failureCount);
  record('simulation bar');
});
close(simulationPlot.epsilonY, simulationPlot.y(0.1), 'the epsilon line sits at epsilon');
/* The axis is logarithmic on purpose. On a linear axis the last three bars
 * spanned one or two pixels, so the figure could not show the thing its own
 * caption claims. That is asserted here as a MEASURED separation, not as a
 * property of the scale: every bar must be tall enough to read. */
assert.equal(simulationPlot.logarithmic, true, 'the risk axis is logarithmic');
simulationPlot.bars.forEach(bar => {
  assert.ok(bar.lowY - bar.highY >= 8,
    `the quantile bar at n=${bar.n} is only ${(bar.lowY - bar.highY).toFixed(1)} units tall, which is not readable`);
});
assert.ok(Math.min(...pacData.simulation.rows.map(row => row.riskQuantiles[0])) > 0,
  'and every plotted quantile is strictly positive, which is what makes that axis available');
nonEmpty(simulationPlot.yTicks, 4, 'logarithmic risk ticks, one per decade from .001 to 1');
record('the simulation plot is readable at every size');
/* The y axis is inverted, so a falling risk is drawn at a LARGER y. Asserting
 * the mark moves up the screen would assert the opposite of the trend. */
for (let index = 1; index < simulationPlot.bars.length; index += 1) {
  assert.ok(pacData.simulation.rows[index].meanTrueError < pacData.simulation.rows[index - 1].meanTrueError,
    'the mean risk falls as n grows');
  assert.ok(simulationPlot.bars[index].meanY > simulationPlot.bars[index - 1].meanY,
    'and the drawn mark moves down the inverted axis, so the figure\'s claimed trend is visible');
}
refuses(() => simulationPlotGeometry({ rows: [pacData.simulation.rows[0]] }), 'a single-row summary');
record('the simulation plot');

const curvePlot = learningCurveGeometry({
  rows: pacData.learningCurves.rows, sizes: pacData.learningCurves.sizes,
});
assert.equal(curvePlot.models.length, 3);
assert.equal(curvePlot.series.length, 6, 'one training and one development curve per procedure');
curvePlot.series.forEach(series => {
  nonEmpty(series.points, 5, `${series.model} ${series.kind} points`);
  series.points.forEach(point => {
    const expected = series.kind === 'train'
      ? 1 - point.row.trainCorrect / point.row.n
      : 1 - point.row.developmentCorrect / point.row.developmentN;
    close(point.value, expected, 'every plotted error is one minus its own recorded count over its own denominator');
    close(point.x, curvePlot.x(point.n));
    close(point.y, curvePlot.y(point.value));
    assert.ok(point.y >= curvePlot.padding.top - 1e-9, 'no point above the frame');
    assert.ok(point.y <= curvePlot.height - curvePlot.padding.bottom + 1e-9, 'none below it');
  });
  record(`learning-curve series ${series.model} ${series.kind}`);
});
record('the learning-curve plot');

const sineDrawing = sinePlotGeometry({ labels: fixtures.sineLabels });
nonEmpty(sineDrawing.dials, 4, 'sine dials');
sineDrawing.dials.forEach(dial => {
  close(Math.hypot(dial.markX, dial.markY), sineDrawing.radius, 'every dial mark sits on its own circle');
  close(dial.cycle, fractionToNumber(fraction(BigInt(dial.cycleText.split('/')[0]),
    BigInt(dial.cycleText.split('/')[1] ?? '1'))), 'the printed fraction is the value the mark was placed by');
  // The drawn half and the predicted label must agree: a mark above the centre
  // line is the positive half, and the y axis points down in SVG.
  assert.equal(dial.positive, dial.markY < -1e-9, 'a positive label is drawn in the upper half and nowhere else');
  record('sine dial');
});
assert.deepEqual(sineDrawing.dials.map(dial => (dial.positive ? 1 : 0)), fixtures.sineLabels,
  'the four drawn dials spell out the requested labels');
record('the sine drawing');

const ghost = ghostCollapse();
assert.equal(ghost.combinedSize, 8);
assert.equal(ghost.patternCount, ghost.patternCountFormula, 'the collapse count matches the closed form');
/* O1: comparing ghost.patternCount with intervalPatterns(8).length repeated the
   computation. The independent statement is the closed form on the combined
   size, which the module does not use to produce the count. */
assert.equal(ghost.patternCount, 1 + (ghost.combinedSize * (ghost.combinedSize + 1)) / 2,
  'the collapsed row count equals 1 + n(n+1)/2 on the combined inputs');
assert.equal(ghost.allBinary, 256);
assert.ok(ghost.trainingOnlyPatternCount < ghost.patternCount,
  'the training inputs alone admit fewer rows, which is the number that may not be substituted for K');
assert.equal(ghost.trainingOnlyPatternCount, intervalPatterns(4).length);
assert.equal(ghost.combined.filter(entry => entry.from === 'training').length, 4);
assert.equal(ghost.combined.filter(entry => entry.from === 'ghost').length, 4);
for (let index = 1; index < ghost.combined.length; index += 1) {
  assert.ok(ghost.combined[index].x > ghost.combined[index - 1].x, 'the combined inputs are sorted and distinct');
}
record('the ghost collapse');

/* ================================ the browser RNG, declared and reproducible */

const drawA = seededUniformSample(7, 8);
const drawB = seededUniformSample(7, 8);
assert.deepEqual(drawA, drawB, 'the same seed gives the same sample');
assert.notDeepEqual(drawA, seededUniformSample(8, 8), 'a different seed gives a different one');
assert.equal(drawA.length, 8);
drawA.forEach(value => {
  assert.ok(value >= 0 && value <= 1, 'every draw lands in the unit interval');
  assert.equal(Number(value.toFixed(3)), value, 'and is representable at the three decimals the fields accept');
});
assert.equal(new Set(seededUniformSample(11, 24)).size >= 20, true, 'a draw of 24 is not a constant sequence');
refuses(() => seededUniformSample(7, 0), 'a draw of nothing');
refuses(() => seededUniformSample(7, 25), 'a draw beyond the control\'s bound');
refuses(() => seededUniformSample(-1, 8), 'a negative seed');
record('the declared browser generator');

/* ==================================== the data module against the packet */

assert.equal(pacData.provenance.sha256, 'd28fa993ed459d2f706816395475af08eebd2f394be67f2ad42dd9b511fc6b5a');
assert.equal(pacData.provenance.bytes, 21237);
assert.equal(pacData.provenance.file, '/learn-assets/pac-learning/banknote-subset.csv',
  'the page serves its own copy, not a sibling lesson\'s');
assert.ok(!pacData.provenance.file.includes('evaluation-metrics'));
assert.ok(!pacData.provenance.file.includes('semi-supervised'));
assert.equal(pacData.learningCurves.testEvaluated, false);
assert.equal(pacData.learningCurves.rows.length, 15);
pacData.learningCurves.rows.forEach((row, index) => {
  const packetRow = recordedCurves.rows[index];
  assert.equal(row.model, packetRow.model);
  assert.equal(row.n, packetRow.n);
  assert.equal(row.trainCorrect, packetRow.train_correct, `training count for ${row.model} at n=${row.n}`);
  assert.equal(row.developmentCorrect, packetRow.development_correct, `development count for ${row.model} at n=${row.n}`);
  assert.equal(row.developmentN, 80);
  assert.ok(row.trainCorrect <= row.n, 'a count never exceeds its denominator');
  assert.ok(row.developmentCorrect <= row.developmentN);
  record('learning-curve row against the packet');
});
const treeAtEighty = pacData.learningCurves.rows.find(row => row.model === 'depth5_tree' && row.n === 80);
const svcAtEighty = pacData.learningCurves.rows.find(row => row.model === 'rbf_svc' && row.n === 80);
assert.equal(treeAtEighty.trainCorrect, 80, 'the tree fits every training label at n=80');
assert.equal(treeAtEighty.developmentCorrect, 74);
assert.equal(svcAtEighty.trainCorrect, 77, 'the SVC does not');
assert.equal(svcAtEighty.developmentCorrect, 79);
assert.ok(svcAtEighty.developmentCorrect > treeAtEighty.developmentCorrect,
  'and the imperfect training fit wins on development, which is the point of the section');
record('the headline measured comparison');

pacData.simulation.rows.forEach((row, index) => {
  const packetRow = recorded.simulation.rows[index];
  assert.equal(row.n, packetRow.n);
  assert.equal(row.failureCount, packetRow.failure_count, `failure count at n=${row.n}`);
  close(row.meanTrueError, packetRow.mean_true_error, `mean risk at n=${row.n}`);
  close(row.twoStripBound, packetRow.distribution_specific_failure_bound, `bound at n=${row.n}`);
  assert.deepEqual(row.riskQuantiles.map(value => Number(value.toFixed(12))),
    packetRow.risk_quantiles.map(value => Number(value.toFixed(12))), `quantiles at n=${row.n}`);
  assert.equal(row.previewTrueErrors.length, 20, 'exactly the twenty retained risks, no more');
  assert.ok(row.meanTrueError > row.riskQuantiles[1], `the risk distribution is right-skewed at n=${row.n}`);
  record('simulation row against the packet');
});
assert.deepEqual(pacData.simulation.rows.map(row => row.failureCount), [743, 368, 46, 1, 0]);
assert.equal(pacData.simulation.rows[4].failureCount, 0);
assert.ok(pacData.simulation.rows[4].twoStripBound > 0,
  'zero observed failures sits beside a strictly positive bound');
record('the retained experiment');

/* ============================ the displayed programs against the models */

const patternsOutput = pacExamples.patterns.expected.split('\n');
assert.equal(patternsOutput.length, 6, 'the pattern program printed five rows and one list');
for (let n = 1; n <= 5; n += 1) {
  assert.equal(patternsOutput[n - 1], `${n} ${thresholdPatterns(n).length} ${intervalPatterns(n).length} ${2 ** n}`,
    `the pattern program's row at n=${n} is what the browser model computes`);
  record('displayed pattern row');
}
assert.equal(patternsOutput[5], `[${intervalPatterns(3).map(pattern => `'${pattern.join('')}'`).join(', ')}]`,
  'and its list of three-point patterns is the same enumeration');
assert.ok(!patternsOutput[5].includes("'101'"), 'which does not contain 101');
const worldOutput = pacExamples['finite-world'].expected.split('\n');
assert.equal(worldOutput.length, 7);
[1, 2, 4, 8, 16, 24].forEach((n, index) => {
  const model = finiteWorldProbability({ target: [0, 0, 1, 1], n, epsilon: 0.25 });
  const parts = worldOutput[index].split(' ');
  assert.equal(parts[0], String(n));
  assert.equal(parts[1], model.failureExact, `the program's exact fraction at n=${n}`);
  close(Number(parts[2]), model.bound.raw, `the program's raw bound at n=${n}`, 1e-6);
  close(Number(parts[3]), model.bound.clipped, `and its clipped bound at n=${n}`, 1e-6);
  record('displayed finite-world row');
});
assert.equal(worldOutput[6], 'all-zero target at n=4: 0');
const riskOutput = pacExamples['interval-risk'].expected.split('\n');
assert.equal(riskOutput.length, 5);
[worked, nearEdges, negativeNull, noPositives].forEach((experiment, index) => {
  const expected = `${experiment.interval ? `[${experiment.interval[0]}, ${experiment.interval[1]}]` : 'None'}`
    + ` ${experiment.empiricalRisk.toFixed(1)} ${Number(experiment.risk.toFixed(6))}`;
  assert.equal(riskOutput[index], expected, `the program's fit line ${index} matches the browser model`);
  record('displayed interval row');
});
assert.ok(riskOutput[4].includes("'feasible': False"), 'and the contradictory request is reported as infeasible');
assert.equal(riskOutput[0], riskOutput[2], 'the exterior-negative null is visible in the printed output itself');
pacExamples.patterns.extraction.functions.forEach(name =>
  assert.ok(pacExamples.patterns.code.includes(`def ${name}(`), `the displayed code really defines ${name}`));
assert.ok(pacExamples.patterns.extraction.extractedLines > pacExamples.patterns.extraction.composedLines,
  'the mechanism occupies more lines than the scaffolding around it');
assert.ok(pacExamples['finite-world'].extraction.extractedLines
  > pacExamples['finite-world'].extraction.composedLines, 'in the finite-world program too');
assert.ok(pacExamples['interval-risk'].extraction.extractedLines
  > pacExamples['interval-risk'].extraction.composedLines, 'and in the interval program');
/* O5: the install snippet pins library versions as prose literals while the
   examples module records what actually resolved. Nothing kept the two in step,
   so a runtime upgrade would leave the page telling readers to install the old
   versions. */
const lessonBody = fs.readFileSync('src/learn/data/topics/pac-learning-vc-dimension.jsx', 'utf8');
['numpy', 'scipy', 'scikit-learn'].forEach(name => {
  const resolved = pacExamples.patterns.environment[name];
  assert.ok(resolved, `the examples module records a ${name} version`);
  assert.ok(lessonBody.includes(`${name}==${resolved}`),
    `the install snippet pins ${name}==${resolved}, the version the programs actually ran on`);
  record('install pin matches the recorded runtime');
});
/* O6: the prose used to name three of the four configurations the program
   checks. All four counts must now appear in the body. */
[['triangle', 8], ['square', 14], ['collinear', 6], ['interior', 14]].forEach(([name, count]) => {
  assert.equal(pacData.halfPlaneWitnesses[name].count, count, `${name} realizable count`);
});
assert.ok(lessonBody.includes('14 for the interior-point configuration'),
  'the prose names the interior configuration alongside the other three');
record('the prose enumerates every checked configuration');
/* S3: the closing callout must not claim more than the pin covers. */
assert.ok(!lessonBody.includes('byte-exact slices of that program'),
  'the closing callout no longer calls the assembled snippets byte-exact slices of the whole program');
assert.ok(lessonBody.includes('adapted import line'),
  'and says what was added to them');
record('the callout matches what is pinned');

assert.equal(pacExamples.calculations.downloadOnly, true);
assert.equal(pacExamples.curves.downloadOnly, true);
assert.equal(pacExamples.curves.expected.split('\n').length, 15, 'the curve program printed one line per fit');
record('the displayed programs');

/* ================================================================ evidence */

const sources = [
  'src/learn/data/pac-models.js',
  'src/learn/data/pac-data.js',
  'src/learn/data/pac-examples.js',
  'src/learn/data/topics/pac-learning-vc-dimension.jsx',
  'src/learn/data/curriculum/blueprints/pac-learning-vc-dimension.js',
  'src/learn/components/lesson-labs/PacShared.jsx',
  'src/learn/components/lesson-labs/PacLabs.jsx',
  'src/learn/components/lesson-labs/PacFigures.jsx',
  'src/learn/components/lesson-labs/pac-labs.css',
  'public/learn-assets/pac-learning/banknote-subset.csv',
  'public/learn-assets/pac-learning/ATTRIBUTION.txt',
];
// Unfiltered on purpose: silently dropping a declared source that no longer
// exists and still writing `passed: true` is how a deleted file passes.
const missingSources = sources.filter(file => !fs.existsSync(file));
assert.deepEqual(missingSources, [], `declared source files are missing: ${missingSources.join(', ')}`);

const hash = file => crypto.createHash('sha256').update(fs.readFileSync(file)).digest('hex');
const total = Object.values(counts).reduce((sum, value) => sum + value, 0);
/* A counter that is reported but never floored is decoration: deleting every
 * `record()` call still printed PASS with a smaller number. These floors sit
 * below the current values but far above zero. */
/* S5: floors close to the real values, not at half of them. At 300 the suite
   could silently lose 220 checks and still print PASS. Set just below the
   current 520/95 so a lost section fails the run while a legitimate addition
   does not. */
assert(total >= 505, `only ${total} grouped checks ran; the suite has lost coverage`);
assert(Object.keys(counts).length >= 92, `only ${Object.keys(counts).length} groups ran`);
assert(witnessCases >= 100000, `only ${witnessCases} feasibility cases ran`);
assert(riskCases >= 1000, `only ${riskCases} risk geometries ran`);
assert(selectionCases === 256, `only ${selectionCases} selection cases ran`);
assert(planeCases >= 40, `only ${planeCases} half-plane cases ran`);

const evidence = {
  checkedAt: new Date().toISOString(),
  sourceHashes: Object.fromEntries(sources.map(file => [file, hash(file)])),
  packetCheckedResultsSha256: hash(`${packetDirectory}/checked-results.json`),
  packetCurveResultsSha256: hash(`${packetDirectory}/banknote-learning-curve-results.json`),
  verifierHash: hash('scripts/verify-pac-models.mjs'),
  counts,
  totalGroupedChecks: total,
  feasibilityCasesSwept: witnessCases,
  feasibleVerdictsSeen: feasibleSeen,
  infeasibleVerdictsSeen: infeasibleSeen,
  selectionCasesSwept: selectionCases,
  epsilonTieCasesSwept: tieCases,
  riskGeometriesSwept: riskCases,
  movementCasesSwept: movementCases,
  halfPlaneCasesSwept: planeCases,
  scope: 'The browser PAC/VC models against the content packet\'s checked-results.json and '
    + 'banknote-learning-curve-results.json, the manuscript\'s stated values, and a second derivation for every '
    + 'claim a figure draws. Pattern counts are checked by three independent routes: endpoint enumeration, '
    + 'brute force over all 2^n binary vectors testing contiguity or suffix, and the closed form. '
    + 'Investigation 2\'s feasibility verdict is swept over EVERY ordering of 2 to 6 points and EVERY labeling '
    + 'of them, against membership in the enumerated pattern set, with the witness itself required to reproduce '
    + 'the request it witnesses; both verdicts are shown to occur. The four-input selection rule is checked '
    + 'against its closed form and against first-consistent search over all 256 target-and-subset combinations, '
    + 'and its occupancy probabilities against exhaustive enumeration of all 4^n draw sequences at small n. '
    + 'Symmetric-difference risk is recomputed by a union-minus-intersection route over a 1,000-case grid '
    + 'covering disjoint, touching, nested, overlapping, empty and zero-width intervals, with the drawn '
    + 'segments required to add up to the risk they display. The sine witnesses are checked against the actual '
    + 'floating-point sign of sin(theta x) and against a margin that keeps them off the boundary. Every bound '
    + 'is checked against the packet, against an independent rebuilding of its own expression, and for '
    + 'monotonicity in each argument; the finite radius is additionally checked by inverting Hoeffding. Every '
    + 'drawn coordinate -- meters, candidate bands, growth curves, bound curves, interval strips, simulation '
    + 'quantiles, learning curves and sine dials -- is checked against its own scale and required to stay '
    + 'inside its frame. The unchanged rule is asserted as an if and only if over a degenerate grid including '
    + 'identical values, signed zeros and the exact tolerance boundary.',
  limitations: [
    'The recorded half-plane weights are validated as witnesses here -- this (w, b) realizes exactly this '
      + 'labeling with this minimum margin -- not re-derived. Which labelings are realizable at all is '
      + 're-derived by exact convex-hull intersection in scripts/verify-pac-data.py.',
    'The measured banknote counts are regenerated from the served dataset by scripts/verify-pac-data.py; this '
      + 'file checks that the browser module carries those recorded counts unchanged.',
    'Displayed program output is executed separately by scripts/verify-pac-examples.py; this file checks that '
      + 'what those programs printed agrees with what the browser models compute.',
    'The class-wide VC upper bounds are the manuscript\'s geometric arguments. The finite checks here support '
      + 'the lower bounds and the drawn fixtures; no finite computation establishes a universal statement.',
    'Rendering, interaction, visual layout and independent review are separate steps and are not claimed here.',
  ],
  passed: true,
};
/* `--no-evidence` lets an independent reviewer re-run this without writing to
 * the record they are reviewing. */
if (!process.argv.includes('--no-evidence')) {
  fs.mkdirSync('docs/teaching/evidence', { recursive: true });
  fs.writeFileSync('docs/teaching/evidence/pac-models.json', JSON.stringify(evidence, null, 2) + '\n');
}
console.log(`PASS: ${total} grouped PAC/VC model checks across ${Object.keys(counts).length} groups, including `
  + `${witnessCases.toLocaleString('en-US')} feasibility cases over every ordering and labeling of 2 to 6 points `
  + `(${feasibleSeen.toLocaleString('en-US')} feasible, ${infeasibleSeen.toLocaleString('en-US')} not), `
  + `${selectionCases} selection cases, ${tieCases} exact epsilon ties, `
  + `${riskCases.toLocaleString('en-US')} risk geometries by two routes, ${movementCases} movement cases and `
  + `${planeCases} half-plane witnesses.`);
