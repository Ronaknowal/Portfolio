// Bounded independent checks of the calibration browser models against the
// content packet's recorded calculations, the manuscript's stated values,
// analytic identities and a second algorithm for every quantitative claim.
//
// Every assertion here is written so that it can fail: nothing is compared with
// itself, no loop is allowed to inspect an empty subject set, and every group
// asserts how many subjects it actually saw.
//
// Four things are the reason this file exists.
//
//   * **The rank.** It is the one number on the page that must not be
//     approximated, and `Math.ceil((n + 1) * (1 - alpha))` in ordinary floating
//     point gives the wrong integer at n=24, alpha=.44 — the one such point a
//     learner can reach — and at 25 further points of the verifier's wider sweep.
//     The rank is therefore recomputed here by scanning integers under exact
//     rational arithmetic — never by a ceiling, never by a quantile helper, and
//     never by calling the module's own function — over every (n, alpha) pair
//     the controls admit.
//   * **The monotone fit.** Recomputed by the max-min formula over lower and
//     upper sets, which shares no code with the stack algorithm, over every
//     labelling of several score patterns rather than on the lesson's example.
//   * **The graded verdict.** "Unchanged" must mean exactly that the number on
//     screen did not move. That equivalence is swept over the enterable grid
//     using the SAME display function the components call, so the check tests
//     the real rule rather than a copy of it, and every grading rule is
//     exercised at zero, at exact ties, at identical inputs and where the
//     quantity has no value at all.
//   * **The geometry.** A drawn reliability curve, interval band or coverage
//     bar is a mathematical claim. Every plotted coordinate is asserted against
//     the quantity it encodes, including that a proportional bar has no
//     minimum length and that an unbounded threshold is not drawn as a point.
//
// Run: node scripts/verify-calibration-models.mjs
import assert from 'node:assert/strict';
import fs from 'node:fs';
import crypto from 'node:crypto';
import {
  apsScores, apsSets, barGeometry, binIndexOf, blockGeometry, brierLoss, changeDirection, checkAlpha,
  checkEdges, checkOutcome, checkProbability, checkScale, classSets, conditioningFork, conformalRank,
  conformalThreshold, controlSteps, coverageReport, cqrInterval, cqrScore, decimalParts, entropyNats,
  fitPav, fitSigmoid, fixtures, floatingBoundaryCheck, gradedDisplayDigits, gradedText, intervalGeometry,
  isotonicPredict, leadingClass, limits, linearScale, logScale, mosaicGeometry, mosaicMarginal,
  nextRequiredMerge, normalizedInterval, orderStatisticCoverageMean, pavStages, plattTargets,
  positivePredictiveValue, quantityKinds, railGeometry, rankRotation, rearrangeQuantiles, reliability,
  reliabilityFromRecord, reliabilityGeometry, resolutionComparison, rocAuc, scatterGeometry, softmaxAt,
  thresholdDirection, unchangedTolerance, confidenceReliability,
} from '../src/learn/data/calibration-models.js';
import { calibrationData } from '../src/learn/data/calibration-data.js';
import { calibrationExamples } from '../src/learn/data/calibration-examples.js';

/* Evidence must never outlive the run that produced it.
 *
 * Every assertion below throws, and a throw skips the write at the bottom of
 * this file — which leaves the PREVIOUS run's `passed: true` on disk. Anyone
 * reading the evidence directory afterwards, including a reviewer closing this
 * topic, sees a green record for a tree that fails. A sibling lesson shipped
 * exactly that. The failure path now overwrites the record with a failing one. */
const EVIDENCE_FILE = 'docs/teaching/evidence/calibration-models.json';
const writesEvidence = !process.argv.includes('--no-evidence');
process.on('uncaughtException', error => {
  if (writesEvidence) {
    try {
      fs.mkdirSync('docs/teaching/evidence', { recursive: true });
      fs.writeFileSync(EVIDENCE_FILE, JSON.stringify({
        checkedAt: new Date().toISOString(),
        verifier: 'scripts/verify-calibration-models.mjs',
        passed: false,
        failure: { message: String(error && error.message), stack: String(error && error.stack).slice(0, 4000) },
        note: 'This run failed. The record is written from the failure path so that a red tree cannot be read '
          + 'as green from a previous run.',
      }, null, 2) + '\n');
    } catch { /* the failure itself must still surface below */ }
  }
  console.error(error);
  process.exit(1);
});


const packetDirectory = 'docs/teaching/drafts/calibration-conformal-prediction';
const checked = JSON.parse(fs.readFileSync(`${packetDirectory}/checked-results.json`, 'utf8'));
const experiment = JSON.parse(fs.readFileSync(`${packetDirectory}/experiment-results.json`, 'utf8'));
const lessonBody = fs.readFileSync('src/learn/data/topics/calibration-conformal-prediction.jsx', 'utf8');
const labsSource = fs.readFileSync('src/learn/components/lesson-labs/CalibrationLabs.jsx', 'utf8');

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
  if (expected !== undefined) {
    assert.equal(collection.length, expected, `${label}: saw ${collection.length}, expected ${expected}`);
  }
};
/** A call that must be refused. A silent default here would be a wrong answer. */
const refuses = (call, label) => {
  assert.throws(call, RangeError, `${label}: should have been refused`);
  record('refused input');
};

/* ================= an independent rank, by scanning integers exactly ======== */

/**
 * The smallest rank whose share of the n+1 places reaches the requested
 * coverage, in integers.
 *
 * This is the rank argument itself rather than its closed form: it never forms
 * (n + 1)(1 - alpha), never calls a ceiling, and never touches a quantile. The
 * comparison j/(n+1) >= 1 - a/d becomes j*d >= (n+1)*(d - a), which is exact.
 */
function independentRank(n, alphaText) {
  const [whole, fraction = ''] = alphaText.split('.');
  const denominator = 10 ** fraction.length;
  const numerator = Number(`${whole}${fraction}`);
  const target = (n + 1) * (denominator - numerator);
  for (let j = 1; j <= n + 1; j += 1) {
    if (j * denominator >= target) return j;
  }
  throw new Error(`no rank reaches coverage for n=${n}, alpha=${alphaText}`);
}

/** The threshold by COUNTING rather than by indexing a sorted array. */
function independentThreshold(scores, alphaText) {
  const k = independentRank(scores.length, alphaText);
  if (k > scores.length) return Infinity;
  let best = Infinity;
  scores.forEach(candidate => {
    const atOrBelow = scores.filter(value => value <= candidate).length;
    if (atOrBelow >= k && candidate < best) best = candidate;
  });
  return best;
}

/** Isotonic regression by the max-min formula over lower and upper sets. */
function independentIsotonic(scores, labels) {
  const order = scores.map((_value, index) => index).sort((a, b) => (scores[a] - scores[b]) || (a - b));
  const knots = [];
  const totals = [];
  const weights = [];
  order.forEach(index => {
    if (knots.length && knots[knots.length - 1] === scores[index]) {
      totals[totals.length - 1] += labels[index];
      weights[weights.length - 1] += 1;
      return;
    }
    knots.push(scores[index]); totals.push(labels[index]); weights.push(1);
  });
  const size = knots.length;
  const blockMean = (u, v) => {
    let total = 0; let weight = 0;
    for (let index = u; index <= v; index += 1) { total += totals[index]; weight += weights[index]; }
    return total / weight;
  };
  const fitted = [];
  for (let i = 0; i < size; i += 1) {
    let best = -Infinity;
    for (let u = 0; u <= i; u += 1) {
      let smallest = Infinity;
      for (let v = i; v < size; v += 1) smallest = Math.min(smallest, blockMean(u, v));
      best = Math.max(best, smallest);
    }
    fitted.push(best);
  }
  return { knots, fitted, weights, totals };
}

/** AUC by counting concordant pairs, ties counted as one half. */
function independentAuc(scores, labels) {
  let concordant = 0; let pairs = 0;
  for (let i = 0; i < scores.length; i += 1) {
    for (let j = 0; j < scores.length; j += 1) {
      if (labels[i] === 1 && labels[j] === 0) {
        pairs += 1;
        if (scores[i] > scores[j]) concordant += 1;
        else if (scores[i] === scores[j]) concordant += 0.5;
      }
    }
  }
  return pairs === 0 ? null : concordant / pairs;
}

/** Softmax without the max-shift, for logits small enough not to overflow. */
function independentSoftmax(logits, temperature) {
  const exponentials = logits.map(value => Math.exp(value / temperature));
  const total = exponentials.reduce((sum, value) => sum + value, 0);
  return exponentials.map(value => value / total);
}

/** Bins by explicit comparison chains, with no search helper. */
function independentBins(probabilities, outcomes, edges) {
  const rows = [];
  for (let index = 0; index < edges.length - 1; index += 1) {
    const members = [];
    probabilities.forEach((value, position) => {
      const isLast = index === edges.length - 2;
      const inside = value >= edges[index] && (isLast ? value <= edges[index + 1] : value < edges[index + 1]);
      if (inside) members.push(position);
    });
    if (!members.length) { rows.push({ count: 0, meanP: null, fractionPositive: null }); continue; }
    const positive = members.reduce((sum, position) => sum + outcomes[position], 0);
    rows.push({
      count: members.length, positive,
      meanP: members.reduce((sum, position) => sum + probabilities[position], 0) / members.length,
      fractionPositive: positive / members.length,
    });
  }
  const ece = rows.reduce((sum, row) =>
    (row.count ? sum + row.count * Math.abs(row.fractionPositive - row.meanP) : sum), 0) / probabilities.length;
  return { rows, ece };
}

/** Every 0/1 labelling of n observations. */
function labellings(n) {
  return Array.from({ length: 2 ** n }, (_value, code) =>
    Array.from({ length: n }, (_v, index) => (code >> index) & 1));
}

/* ======================================== §0 · limits, guards and refusals */

Object.entries(quantityKinds).forEach(([key, entry]) => {
  assert.equal(entry.key, key, `${key} names itself`);
  assert(entry.label.length > 4 && entry.note.length > 30, `${key} carries a label and a note`);
  record('quantity kind');
});
assert.deepEqual(Object.keys(quantityKinds), ['population', 'calibration', 'assessment'],
  'the three kinds of number are declared in the order the page introduces them');

/* Every value the prose asks a learner to type has to be reachable through the
   control that edits it: the step, the bounds and the decimal places. */
const reachable = (value, control, label) => {
  const scale = 10 ** control.decimals;
  assert(value >= control.minimum && value <= control.maximum, `${label}: ${value} is outside the control's range`);
  assert(Math.abs(value * scale - Math.round(value * scale)) <= 1e-9,
    `${label}: ${value} needs more than ${control.decimals} decimal places`);
  record('reachable value');
};
fixtures.cards.forEach(card => reachable(card.forecast, controlSteps.forecast, `card ${card.id}`));
fixtures.repairedForecasts.forEach((value, index) => reachable(value, controlSteps.forecast, `repaired ${index}`));
fixtures.boundaryCards.forEach(card => reachable(card.forecast, controlSteps.forecast, `boundary ${card.id}`));
[...fixtures.pavScores, ...fixtures.labPavScores, ...fixtures.tiedScores, ...fixtures.practicePavScores]
  .forEach((value, index) => reachable(value, controlSteps.score, `score ${index}`));
[...fixtures.calibrationScores, ...fixtures.labCalibrationScores, ...fixtures.rotationScores]
  .forEach((value, index) => reachable(value, controlSteps.conformalScore, `calibration score ${index}`));
fixtures.residuals.concat(fixtures.changedResiduals)
  .forEach((value, index) => reachable(value, controlSteps.residual, `residual ${index}`));
fixtures.localScales.forEach((value, index) => reachable(value, controlSteps.localScale, `scale ${index}`));
fixtures.queries.forEach(query => {
  reachable(query.centre, controlSteps.queryCentre, `${query.name} centre`);
  reachable(query.localScale, controlSteps.localScale, `${query.name} scale`);
});
fixtures.cqrScores.concat(fixtures.cqrAlternativeScores)
  .forEach((value, index) => reachable(value, controlSteps.signedScore, `cqr score ${index}`));
[fixtures.defaultAlpha, fixtures.intervalAlpha, fixtures.cqrAlpha, 0.05, 0.1]
  .forEach((value, index) => reachable(value, controlSteps.alpha, `alpha ${index}`));
/* Doubling every scale is offered as a preset, so the doubled values must be
   reachable too, or the null the lesson promises would be refused. */
fixtures.localScales.forEach((value, index) =>
  reachable(value * 2, controlSteps.localScale, `doubled scale ${index}`));
fixtures.queries.forEach(query =>
  reachable(query.localScale * 2, controlSteps.localScale, `${query.name} doubled scale`));
record('control reachability');

refuses(() => checkProbability(1.0001, 'p'), 'a probability above one');
refuses(() => checkProbability(-1e-9, 'p'), 'a probability below zero');
refuses(() => checkProbability(Number.NaN, 'p'), 'a probability that is not a number');
refuses(() => checkOutcome(0.5, 'y'), 'a fractional outcome');
refuses(() => checkScale(0, 'u'), 'a zero local scale');
refuses(() => checkScale(-1, 'u'), 'a negative local scale');
refuses(() => checkAlpha(0), 'alpha of exactly zero');
refuses(() => checkAlpha(1), 'alpha of exactly one');
refuses(() => conformalRank(0, 0.1), 'a rank with no calibration scores');
refuses(() => conformalRank(2.5, 0.1), 'a rank with a fractional n');
refuses(() => decimalParts(0.1234567, 'alpha'), 'an alpha with seven decimal places');
refuses(() => decimalParts(1e-8, 'alpha'), 'an alpha written in scientific notation');
refuses(() => checkEdges([0, 0.5]), 'bin boundaries that do not span to one');
refuses(() => checkEdges([0.1, 0.5, 1]), 'bin boundaries that do not start at zero');
refuses(() => checkEdges([0, 0.5, 0.5, 1]), 'a repeated bin boundary');
refuses(() => checkEdges([0, 0.6, 0.4, 1]), 'bin boundaries that decrease');
refuses(() => checkEdges([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 1]), 'more bins than the editor holds');
refuses(() => reliability([0.5], [1], [0, 1, 2]), 'bin boundaries outside the probability scale');
refuses(() => reliability([], [], [0, 1]), 'a reliability report with no observations');
refuses(() => reliability([0.5, 0.5], [1], [0, 1]), 'mismatched forecast and outcome lists');
refuses(() => fitPav([1, 2], [1], null), 'a monotone fit with mismatched lists');
refuses(() => fitPav([1, 2], [1, 0], [0, 1]), 'a zero observation weight');
refuses(() => fitPav([1, Number.POSITIVE_INFINITY], [1, 0]), 'a non-finite score');
refuses(() => softmaxAt([1, 2], 0), 'a temperature of zero');
refuses(() => softmaxAt([1, 2], -1), 'a negative temperature');
refuses(() => softmaxAt([1], 1), 'a softmax over a single logit');
refuses(() => conformalThreshold([], 0.1), 'a threshold with no calibration scores');
refuses(() => cqrScore(20, 10, 15), 'a CQR score from crossed quantile estimates');
refuses(() => coverageReport(5, 4, 'assessment'), 'a coverage count larger than its denominator');
refuses(() => coverageReport(1, 4, 'guarantee'), 'a coverage count with an undeclared kind');
refuses(() => orderStatisticCoverageMean(0, 10), 'a Beta mean at rank zero');
refuses(() => orderStatisticCoverageMean(11, 10), 'a Beta mean at a rank beyond n');
refuses(() => mosaicMarginal([{ share: 0.5, coverage: 1 }]), 'group shares that do not add to one');
refuses(() => conditioningFork([{ share: 0.4, forecast: 0.2, positiveRate: 0.3 }]),
  'forecast group shares that do not add to one');
refuses(() => logScale({ domain: [0, 10], range: [0, 1] }), 'a logarithmic scale reaching zero');
refuses(() => linearScale({ domain: [1, 1], range: [0, 1] }), 'a scale with a degenerate domain');
record('guard suite');

/* ============================== §1 · the rank, over the whole enterable grid */

const alphaGrid = [];
for (let thousandths = 10; thousandths <= 500; thousandths += 1) {
  alphaGrid.push({ value: Number((thousandths / 1000).toFixed(3)), text: (thousandths / 1000).toFixed(3) });
}
nonEmpty(alphaGrid, 491, 'alpha grid');
let rankCases = 0;
let naiveDisagreements = 0;
const naiveExamples = [];
/* Every disagreement, not only the first few: the reachable count and the
   failure-mode assertions below need the whole set. */
const naiveAll = [];
for (let n = 1; n <= 200; n += 1) {
  for (const alpha of alphaGrid) {
    const own = conformalRank(n, alpha.value);
    const independent = independentRank(n, alpha.text);
    assert.equal(own, independent,
      `rank at n=${n}, alpha=${alpha.text}: module gives ${own}, the integer scan gives ${independent}`);
    assert(own >= 1 && own <= n + 1, `rank at n=${n}, alpha=${alpha.text} is outside 1..n+1`);
    /* The rank is monotone in the requested coverage: a smaller alpha asks for
       more, and can never ask for a lower rank. */
    if (alpha.value > 0.01) {
      const tighter = conformalRank(n, Number((alpha.value - 0.001).toFixed(3)));
      assert(tighter >= own, `rank is not monotone in alpha at n=${n}, alpha=${alpha.text}`);
    }
    const naive = Math.ceil((n + 1) * (1 - alpha.value));
    if (naive !== own) {
      naiveDisagreements += 1;
      naiveAll.push({ n, alpha: alpha.value, naive, exact: own });
      if (naiveExamples.length < 6) naiveExamples.push({ n, alpha: alpha.text, naive, exact: own });
    }
    rankCases += 1;
  }
}
record('rank sweep');
assert(rankCases >= 90000, `only ${rankCases} rank cases swept`);
/* The exact-arithmetic guard must be shown to bite somewhere a learner can
   reach. A guard whose domain excludes every defect is not a guard. */
assert(naiveDisagreements > 0,
  'ordinary floating point agreed with the exact rank everywhere, so the exact-arithmetic guard was never '
  + 'exercised against the defect it exists to prevent');
/* `naiveExamples` holds only the first few. The reachable count is computed
   over ALL of them, because "26 disagreements on the grid these controls
   admit" was recorded when 26 is the count over the verifier's whole n=1..200
   scan: the largest n any control feeding a rank can produce is
   `limits.calibrationPairs.maximum`, and over that range there is exactly one. */
const reachableDisagreements = naiveAll.filter(entry =>
  entry.n <= limits.calibrationPairs.maximum
  && entry.alpha >= controlSteps.alpha.minimum && entry.alpha <= controlSteps.alpha.maximum);
assert(reachableDisagreements.length > 0,
  'every floating-point disagreement lies outside the grid a learner can type, so the guard is decorative');
assert.equal(reachableDisagreements.length, 1,
  `expected exactly one reachable float-rank disagreement, found ${reachableDisagreements.length}: `
  + `${JSON.stringify(reachableDisagreements)} — the record's reachable count must be updated to match`);
assert.deepEqual(
  { n: reachableDisagreements[0].n, alpha: reachableDisagreements[0].alpha },
  { n: 24, alpha: 0.44 },
  'the one reachable disagreement is no longer n=24 at alpha=.44');
/* And the failure mode, stated correctly: at every disagreement the float rank
   is exactly one TOO LARGE and never exceeds n, so the consequence is a
   threshold one order statistic too high — silent over-coverage — and never an
   index the calibration set does not have. An earlier record said the
   opposite. */
naiveAll.forEach(entry => {
  assert.equal(entry.naive, entry.exact + 1,
    `at n=${entry.n}, alpha=${entry.alpha} the float rank is not exactly one too large`);
  assert(entry.naive <= entry.n,
    `at n=${entry.n}, alpha=${entry.alpha} the float rank ${entry.naive} exceeds n, which would be a missing `
    + 'order statistic rather than the over-coverage this guard is documented to prevent');
});
record('naive-float divergence');

/* The packet's own rank claims, and the manuscript's. */
close(conformalRank(9, 0.2), checked.rank.k, 'nine scores at alpha .2 give rank 8', 0);
close(conformalRank(9, 0.05), checked.rank.tiny_alpha_rank, 'nine scores at alpha .05 give rank 10', 0);
close(conformalRank(14, 0.2), checked.practice.rank_n14_alpha_point2, 'practice 3 rank at alpha .2', 0);
close(conformalRank(14, 0.02), 15, 'practice 3 rank at alpha .02', 0);
close(conformalRank(80, 0.1), experiment.classification.methods.sigmoid.rank, 'the banknote rank', 0);
close(conformalRank(120, 0.1), experiment.regression.rank, 'the airfoil rank', 0);
record('packet ranks');

/* ==================================== §2 · thresholds, ties and infinity */

let thresholdCases = 0;
const thresholdSubjects = [
  fixtures.calibrationScores, fixtures.labCalibrationScores, fixtures.rotationScores,
  fixtures.tiedRotationScores, fixtures.residuals, fixtures.changedResiduals,
  fixtures.cqrScores, fixtures.cqrAlternativeScores,
  fixtures.residuals.map((value, index) => value / fixtures.localScales[index]),
  [0, 0, 0, 0, 0], [1, 1, 1], [-3, -3, -3, 2, 2],
];
nonEmpty(thresholdSubjects, 12, 'threshold subjects');
thresholdSubjects.forEach((scores, subject) => {
  alphaGrid.filter((_value, index) => index % 7 === 0).forEach(alpha => {
    const own = conformalThreshold(scores, alpha.value);
    const independent = independentThreshold(scores, alpha.text);
    if (own.finite) {
      close(own.q, independent, `threshold on subject ${subject} at alpha ${alpha.text}`, 0);
      /* The defining property, checked directly rather than by re-deriving the
         same index: at least k of the scores are at or below q, and fewer than
         k are strictly below it. */
      const atOrBelow = scores.filter(value => value <= own.q).length;
      const strictlyBelow = scores.filter(value => value < own.q).length;
      assert(atOrBelow >= own.k, `subject ${subject} at alpha ${alpha.text}: only ${atOrBelow} scores reach q`);
      assert(strictlyBelow < own.k, `subject ${subject} at alpha ${alpha.text}: q is not the smallest such score`);
    } else {
      assert.equal(independent, Infinity, `subject ${subject} at alpha ${alpha.text} should be unbounded`);
      assert(own.k > own.n, 'an unbounded threshold must come from a rank beyond the calibration set');
      assert(own.because && own.because.includes('beyond'), 'an unbounded threshold states why');
    }
    /* Order and scale invariance, and the null the investigation promises. */
    const reversed = conformalThreshold([...scores].reverse(), alpha.value);
    assert.equal(reversed.q, own.q, `reordering the cards changed the threshold on subject ${subject}`);
    thresholdCases += 1;
  });
});
record('threshold sweep');
assert(thresholdCases >= 800, `only ${thresholdCases} threshold cases swept`);

const worked = conformalThreshold(fixtures.calibrationScores, fixtures.defaultAlpha);
close(worked.q, checked.rank.q, 'the worked threshold is .6', 0);
assert(!conformalThreshold(fixtures.calibrationScores, 0.05).finite, 'alpha .05 on nine scores is unbounded');
close(conformalThreshold([0.01, ...fixtures.calibrationScores.slice(1)], 0.2).q, worked.q, 'the lower-card null', 0);
close(conformalThreshold([...fixtures.calibrationScores.slice(0, 7), 0.75, 0.9], 0.2).q, 0.75,
  'raising the eighth card raises the threshold', 0);
/* Scaling every score and every future candidate by the same positive factor
   preserves membership exactly: the comparison is homogeneous. */
[2, 3, 0.5, 1000].forEach(factor => {
  const scaled = conformalThreshold(fixtures.calibrationScores.map(value => value * factor), 0.2);
  close(scaled.q, worked.q * factor, `scaling every score by ${factor}`, 1e-12);
  fixtures.calibrationScores.forEach(score => {
    assert.equal(score <= worked.q, score * factor <= scaled.q,
      `membership changed under a scale factor of ${factor}`);
  });
  record('scale invariance');
});

/* The lab's own opening cards: the second candidate's leading score lands
   exactly ON the threshold, which is the whole point of the weak comparison. */
const labThreshold = conformalThreshold(fixtures.labCalibrationScores, fixtures.defaultAlpha);
close(labThreshold.q, 0.55, 'the lab opens on a threshold of .55', 0);
const equalityCandidate = fixtures.candidateVectors[1];
close(1 - equalityCandidate.probabilities[0], labThreshold.q,
  'the opening candidate\'s leading score is exactly the threshold', 0);
assert.deepEqual(classSets(equalityCandidate.probabilities, labThreshold.q).included, [true, false, false],
  'the weak comparison must keep the class whose score equals the threshold');
assert.deepEqual(classSets(equalityCandidate.probabilities, labThreshold.q - 1e-12).included,
  [false, false, false],
  'a threshold one part in a trillion lower must drop that class, or the equality case proves nothing');
record('equality boundary');

/* The three worked sets, and the empty one. */
fixtures.candidateVectors.forEach((candidate, index) => {
  const sets = classSets(candidate.probabilities, worked.q);
  assert.deepEqual(sets.included, checked.rank.prediction_sets[index],
    `candidate ${index} set disagrees with the packet`);
  vector(sets.scores, candidate.probabilities.map(value => 1 - value), `candidate ${index} scores`, 1e-15);
  record('worked set');
});
assert(classSets(fixtures.candidateVectors[2].probabilities, worked.q).empty,
  'the near-uniform candidate must return an empty set');
assert(classSets(fixtures.candidateVectors[0].probabilities, worked.q).singleton,
  'the confident candidate must return a singleton');
/* An unbounded threshold returns every class, whatever the probabilities. */
labellings(3).forEach(bits => {
  const total = bits.reduce((sum, value) => sum + value, 0) || 1;
  const probabilities = bits.map(value => value / total);
  assert(classSets(probabilities, Infinity).included.every(Boolean),
    'an unbounded threshold must include every class');
  record('unbounded set');
});

/* ================================ §3 · rotations, by the combined-rank route */

let rotationCases = 0;
const rotationSubjects = [
  fixtures.rotationScores, fixtures.tiedRotationScores,
  [0.1, 0.2, 0.3, 0.4, 0.5], [0.2, 0.2, 0.2, 0.9, 0.9], [0, 0, 0, 0],
  fixtures.calibrationScores,
];
nonEmpty(rotationSubjects, 6, 'rotation subjects');
rotationSubjects.forEach((scores, subject) => {
  alphaGrid.filter((_value, index) => index % 23 === 0).forEach(alpha => {
    const own = rankRotation(scores, alpha.value);
    /* The proof's own statement, counted directly: the held-out card is covered
       exactly when its combined rank among all of them is at most k. */
    const k = independentRank(scores.length - 1, alpha.text);
    let covered = 0;
    scores.forEach((held, index) => {
      const others = scores.filter((_value, position) => position !== index);
      const strictlyBelow = others.filter(value => value < held).length;
      if (strictlyBelow + 1 <= k) covered += 1;
    });
    assert.equal(own.covered, covered,
      `rotation on subject ${subject} at alpha ${alpha.text}: ${own.covered} versus ${covered}`);
    assert.equal(own.targetRank, k, 'the rotation reports the rank it used');
    assert(own.covered >= 0 && own.covered <= own.total, 'a rotation count lies within its total');
    rotationCases += 1;
  });
});
record('rotation sweep');
assert(rotationCases >= 120, `only ${rotationCases} rotation cases swept`);
close(rankRotation(fixtures.rotationScores, 0.2).covered, checked.rank.rotation.covered,
  'eight of ten distinct cards are covered', 0);
close(rankRotation(fixtures.tiedRotationScores, 0.2).covered, checked.rank.tie_rotation.covered,
  'all ten tied cards are covered', 0);
assert.equal(rankRotation(fixtures.tiedRotationScores, 0.2).covered, 10,
  'ties can only enlarge coverage under the weak comparison');
/* Ties never reduce coverage: collapsing any distinct multiset to one value
   must not lower the count. */
[0.05, 0.1, 0.2, 0.3, 0.4].forEach(alpha => {
  const distinct = rankRotation(fixtures.rotationScores, alpha).covered;
  const tied = rankRotation(fixtures.rotationScores.map(() => 0.3), alpha).covered;
  assert(tied >= distinct, `ties lowered coverage at alpha ${alpha}`);
  record('tie monotonicity');
});

/* ==================================== §4 · reliability, bins and the nulls */

const forecasts = fixtures.cards.map(card => card.forecast);
const outcomes = fixtures.cards.map(card => card.outcome);
const twoBins = reliability(forecasts, outcomes, fixtures.twoBinEdges);
const oneBin = reliability(forecasts, outcomes, fixtures.oneBinEdges);
const repairedBins = reliability(fixtures.repairedForecasts, outcomes, fixtures.twoBinEdges);
close(twoBins.ece, checked.reliability.two_bins.ece, 'two-bin ECE');
close(oneBin.ece, checked.reliability.one_bin.ece, 'one-bin ECE', 0);
close(repairedBins.ece, checked.reliability.repaired.ece, 'repaired ECE', 0);
assert.equal(oneBin.ece, 0, 'merging every card into one bin gives exactly zero, not merely a small number');
close(twoBins.bins[0].fractionPositive, 0.4, 'the low bin\'s observed fraction', 0);
close(twoBins.bins[1].fractionPositive, 0.6, 'the high bin\'s observed fraction', 0);
close(twoBins.bins[0].gap, 0.2, 'the low bin sits above the diagonal', 1e-12);
close(twoBins.bins[1].gap, -0.2, 'the high bin sits below the diagonal', 1e-12);
close(brierLoss(forecasts, outcomes), checked.reliability.brier_before, 'Brier before', 1e-12);
close(brierLoss(fixtures.repairedForecasts, outcomes), checked.reliability.brier_repaired, 'Brier after', 1e-12);
close(rocAuc(forecasts, outcomes).value, checked.reliability.auc_before, 'AUC before', 0);
close(rocAuc(fixtures.repairedForecasts, outcomes).value, checked.reliability.auc_repaired, 'AUC after', 0);
record('reliability fixtures');

/* The boundary rule, stated as its own claim: an internal boundary goes right,
   and a forecast of exactly 1 stays in the last bin. */
const boundaryReport = reliability(
  fixtures.boundaryCards.map(card => card.forecast), fixtures.boundaryCards.map(card => card.outcome),
  fixtures.twoBinEdges);
assert.deepEqual(boundaryReport.bins.map(bin => bin.count),
  checked.reliability.boundary_fixture.bins.map(bin => bin.count),
  'the boundary fixture must keep its p=1 observation in the last bin');
close(boundaryReport.ece, checked.reliability.boundary_fixture.ece, 'the boundary fixture ECE');
assert.equal(binIndexOf([0, 0.5, 1], 0.5), 1, 'an internal boundary belongs to the bin on its right');
assert.equal(binIndexOf([0, 0.5, 1], 1), 1, 'a forecast of exactly one stays in the last bin');
assert.equal(binIndexOf([0, 0.5, 1], 0), 0, 'a forecast of exactly zero is in the first bin');
assert.equal(binIndexOf([0, 0.25, 0.5, 0.75, 1], 0.75), 3, 'every internal boundary goes right');
record('bin boundary rule');

/* The bins recomputed by explicit comparison chains, over a grid of edge sets
   and every labelling of the ten cards. */
let binCases = 0;
const edgeSets = [
  [0, 1], [0, 0.5, 1], [0, 0.25, 0.75, 1], [0, 0.2, 0.4, 0.6, 0.8, 1],
  [0, 0.001, 1], [0, 0.999, 1], [0, 0.1, 0.2, 0.3, 0.4, 1],
];
nonEmpty(edgeSets, 7, 'edge sets');
const cardPatterns = [
  forecasts, fixtures.repairedForecasts, [0, 0, 0.5, 0.5, 1, 1, 1, 0, 0.5, 1],
  new Array(10).fill(0.5), [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1],
];
nonEmpty(cardPatterns, 5, 'card patterns');
let emptyBinsSeen = 0;
cardPatterns.forEach(pattern => {
  labellings(10).filter((_value, index) => index % 37 === 0).forEach(labels => {
    edgeSets.forEach(edges => {
      const own = reliability(pattern, labels, edges);
      const independent = independentBins(pattern, labels, edges);
      close(own.ece, independent.ece, 'ECE by two routes', 1e-12);
      own.bins.forEach((bin, index) => {
        assert.equal(bin.count, independent.rows[index].count, 'bin counts agree by two routes');
        if (!bin.count) {
          emptyBinsSeen += 1;
          assert.equal(bin.fractionPositive, null, 'an empty bin has no observed fraction, not a fraction of zero');
          assert.equal(bin.meanP, null, 'an empty bin has no mean forecast either');
          assert.equal(bin.gap, null, 'an empty bin has no signed gap');
        } else {
          close(bin.fractionPositive, independent.rows[index].fractionPositive, 'bin fraction by two routes', 1e-12);
          close(bin.meanP, independent.rows[index].meanP, 'bin mean by two routes', 1e-12);
          close(bin.gap, bin.fractionPositive - bin.meanP, 'the drawn gap is the signed difference', 1e-15);
        }
      });
      assert.equal(own.bins.reduce((sum, bin) => sum + bin.count, 0), pattern.length,
        'every card lands in exactly one bin');
      assert(own.ece >= 0, 'ECE is a weighted mean of absolute values and cannot be negative');
      assert(own.ece <= 1 + 1e-12, 'ECE cannot exceed one');
      /* The permutation null the lab promises: reordering the cards changes
         nothing at all. */
      const order = pattern.map((_value, index) => (index * 7 + 3) % pattern.length);
      const shuffled = reliability(order.map(index => pattern[index]), order.map(index => labels[index]), edges);
      close(shuffled.ece, own.ece, 'reordering the cards changed the ECE', 1e-15);
      assert.deepEqual(shuffled.bins.map(bin => bin.count).sort(), own.bins.map(bin => bin.count).sort(),
        'reordering the cards changed the bin counts');
      binCases += 1;
    });
  });
});
record('reliability sweep');
assert(binCases >= 900, `only ${binCases} binning cases swept`);
assert(emptyBinsSeen >= 50,
  `only ${emptyBinsSeen} empty bins appeared, so the no-observed-fraction rule was barely exercised`);

/* With one bin the summary collapses to the distance between two averages,
   whatever the individual cards say. That is the identity that makes merging a
   rebinning rather than a repair, and it is the reason the lesson's own fixture
   — whose mean forecast is exactly its observed fraction — reports zero. */
let singleBinCases = 0;
let singleBinZero = 0;
cardPatterns.forEach(pattern => {
  labellings(10).forEach(labels => {
    const meanForecast = pattern.reduce((sum, value) => sum + value, 0) / pattern.length;
    const meanOutcome = labels.reduce((sum, value) => sum + value, 0) / labels.length;
    close(reliability(pattern, labels, [0, 1]).ece, Math.abs(meanOutcome - meanForecast),
      'one bin must reduce the summary to the distance between two averages', 1e-12);
    if (reliability(pattern, labels, [0, 1]).ece === 0) singleBinZero += 1;
    singleBinCases += 1;
  });
});
record('single-bin identity');
assert(singleBinCases >= 4000, `only ${singleBinCases} single-bin cases swept`);
assert(singleBinZero > 0, 'no labelling reached exactly zero, so the vanishing case was never exercised');
assert.equal(reliability(forecasts, outcomes, [0, 1]).ece, 0,
  'the lesson\'s own ten cards must give exactly zero in one bin, which is the claim section 2 makes');
close(forecasts.reduce((sum, value) => sum + value, 0) / 10,
  outcomes.reduce((sum, value) => sum + value, 0) / 10,
  'and they do so because their mean forecast equals their observed fraction', 1e-15);

/* AUC by counting concordant pairs, and the one-class case that has no value. */
let aucCases = 0;
cardPatterns.forEach(pattern => {
  labellings(10).filter((_value, index) => index % 17 === 0).forEach(labels => {
    const own = rocAuc(pattern, labels);
    const independent = independentAuc(pattern, labels);
    if (independent === null) {
      assert.equal(own.value, null, 'a one-class sample has no AUC');
      assert(own.because.includes('same outcome'), 'and says why');
    } else {
      close(own.value, independent, 'AUC by two routes', 1e-12);
      assert(own.value >= 0 && own.value <= 1, 'AUC lies in the unit interval');
    }
    aucCases += 1;
  });
});
record('auc sweep');
assert(aucCases >= 300, `only ${aucCases} AUC cases swept`);
assert.equal(rocAuc(forecasts, new Array(10).fill(1)).value, null,
  'every outcome positive leaves no positive-negative pair to order');
assert.equal(rocAuc(forecasts, new Array(10).fill(0)).value, null,
  'every outcome negative leaves no pair either');

/* The confidence diagram computes its own indicator, and disagrees with the
   class-1 one exactly where the selected class is not class 1. */
const confidence = confidenceReliability(forecasts, outcomes, [0, 0.5, 1]);
assert.equal(confidence.axisLabel, 'fraction of selected classes that were correct',
  'the confidence diagram must not label its axis as a fraction of positives');
forecasts.forEach((value, index) => {
  assert.equal(confidence.predictedClass[index], value >= 0.5 ? 1 : 0, 'the selected class follows the .5 rule');
  assert.equal(confidence.confidence[index], Math.max(value, 1 - value), 'confidence is the larger of the two');
  assert.equal(confidence.correct[index], confidence.predictedClass[index] === outcomes[index] ? 1 : 0,
    'correctness compares the selected class with the outcome, not with class 1');
  record('confidence indicator');
});
assert.notDeepEqual(confidence.correct, outcomes,
  'the correctness indicator must differ from the positive outcome on this fixture, or the two diagrams could '
  + 'not be distinguished at all');

/* ================================ §5 · the conditioning fork and resolution */

const fork = conditioningFork(fixtures.forecastGroups);
assert(fork.confidenceLooksCalibrated, 'the confidence reading sits on the diagonal');
assert(!fork.classOneLooksCalibrated, 'while the class-1 reading does not');
close(fork.meanConfidence, 0.8, 'mean confidence', 1e-15);
close(fork.meanCorrectness, 0.8, 'mean correctness', 1e-15);
nonEmpty(fork.confidencePoints, 1, 'confidence points on the lesson fixture');
close(fork.confidencePoints[0].x, 0.8, 'the merged confidence point sits at .8', 1e-15);
close(fork.confidencePoints[0].y, 0.8, 'and its correctness is .8', 1e-15);
fork.rows.forEach(row => {
  close(row.classOneGap, row.positiveRate - row.forecast, 'the class-1 gap is the signed difference', 1e-15);
  assert(row.classOneGap > 0, 'both groups are underestimated for class 1');
  record('fork row');
});
const practiceFork = conditioningFork(fixtures.practiceForecastGroups);
close(practiceFork.meanConfidence, 0.9, 'practice 1 mean confidence', 1e-15);
close(practiceFork.meanCorrectness, 0.9, 'practice 1 mean correctness', 1e-15);
practiceFork.rows.forEach(row => {
  close(row.classOneGap, 0.1, 'each practice-1 class-1 point is .1 above the diagonal', 1e-15);
  record('practice fork row');
});
assert(practiceFork.confidenceLooksCalibrated && !practiceFork.classOneLooksCalibrated,
  'the practice variant must reproduce the same contrast');
record('conditioning fork');

// Live edits must preserve conditional (not merely aggregate) calibration.
const cancellingAverages = conditioningFork([
  { share: 0.5, forecast: 0.6, positiveRate: 0.7 },
  { share: 0.5, forecast: 0.8, positiveRate: 0.7 },
]);
close(cancellingAverages.meanConfidence, cancellingAverages.meanCorrectness, 'overall averages cancel');
assert(!cancellingAverages.confidenceLooksCalibrated, 'distinct off-diagonal confidences must not pass on their overall average');
const equalForecasts = conditioningFork([
  { share: 0.5, forecast: 0.5, positiveRate: 0.2 },
  { share: 0.5, forecast: 0.5, positiveRate: 0.8 },
]);
assert(equalForecasts.classOnePoints.length === 1 && equalForecasts.classOneLooksCalibrated,
  'equal forecasts pool their population rates, even when subgroup rates differ');
assert(equalForecasts.rows.every(row => row.predictedClass === 1), 'exact .5 tie selects class 1');
const complementaryDecimals = conditioningFork([
  { share: 0.5, forecast: 0.45, positiveRate: 0.45 },
  { share: 0.5, forecast: 0.55, positiveRate: 0.55 },
]);
assert(complementaryDecimals.confidencePoints.length === 1, 'roundoff must not split equal complementary confidences');
const perfectEndpoints = conditioningFork([
  { share: 0.5, forecast: 0, positiveRate: 0 },
  { share: 0.5, forecast: 1, positiveRate: 1 },
]);
assert(perfectEndpoints.confidenceLooksCalibrated && perfectEndpoints.classOneLooksCalibrated,
  'both exact probability endpoints remain valid and calibrated');
record('live conditioning fork boundaries');

const resolution = resolutionComparison(fixtures.resolution);
close(resolution.coarseForecast, checked.resolution.coarse, 'the coarse forecast', 1e-15);
close(resolution.coarseBrier, checked.resolution.coarse_brier, 'coarse Brier');
close(resolution.fullBrier, checked.resolution.full_brier, 'full-information Brier');
close(resolution.coarseCost, checked.resolution.coarse_cost, 'coarse cost');
close(resolution.fullCost, checked.resolution.full_cost, 'full-information cost');
close(resolution.decideThreshold, 0.25, 'the decision threshold is quarantine over release', 1e-15);
assert(resolution.fullCost < resolution.coarseCost, 'resolution must reduce the expected cost here');
assert(resolution.fullBrier < resolution.coarseBrier, 'and the proper score');
/* The Brier identity, from the definition rather than the recorded number. */
resolution.rows.forEach(row => {
  close(row.coarseBrier, row.rate * (1 - resolution.coarseForecast) ** 2
    + (1 - row.rate) * resolution.coarseForecast ** 2, 'expected Brier from its definition', 1e-15);
  close(row.fullBrier, row.rate * (1 - row.rate) ** 2 + (1 - row.rate) * row.rate ** 2,
    'and for the resolved forecast', 1e-15);
  /* A forecast equal to the group's own rate minimises the expected Brier loss
     in that group. Swept, not asserted. */
  for (let forecast = 0; forecast <= 1.0000001; forecast += 0.01) {
    const loss = row.rate * (1 - forecast) ** 2 + (1 - row.rate) * forecast ** 2;
    assert(loss >= row.fullBrier - 1e-12,
      `a forecast of ${forecast} beat the group's own rate, which a proper score must not allow`);
  }
  record('proper score minimum');
});
/* Both forecasts are calibrated with respect to their own conditioning: the
   share-weighted rate under the coarse score equals the coarse score itself. */
close(resolution.rows.reduce((sum, row) => sum + row.share * row.rate, 0), resolution.coarseForecast, 1e-15);
record('resolution');

/* ================================== §6 · the monotone map, by a second route */

let pavCases = 0;
let mergeTraces = 0;
const scorePatterns = [
  [-3, -2, -1, 0, 1, 2, 3, 4], [-2, -1, 0, 1, 2, 3], [-2, -2, 0, 1], [1, 2, 3, 4, 5, 6],
  [0, 0, 0, 1, 1, 1], [-1, -1, -1, -1], [0, 1, 2, 2, 3, 3],
];
nonEmpty(scorePatterns, 7, 'score patterns');
scorePatterns.forEach(scores => {
  labellings(scores.length).forEach(labels => {
    const own = fitPav(scores, labels);
    const independent = independentIsotonic(scores, labels);
    assert.deepEqual(own.knots, independent.knots, 'the two groupings of equal scores disagree');
    vector(own.fitted, independent.fitted, 'the stack fit disagrees with the max-min solution', 1e-12);
    /* The fitted values are non-decreasing, which is the constraint itself. */
    own.fitted.forEach((value, index) => {
      if (index > 0) {
        assert(value >= own.fitted[index - 1] - 1e-15, 'the fit is not monotone');
      }
      assert(value >= 0 && value <= 1, 'a fitted probability lies in the unit interval');
    });
    /* Total positive mass is conserved: pooling moves it, never creates it. */
    close(own.fitted.reduce((sum, value, index) => sum + value * own.weights[index], 0),
      labels.reduce((sum, value) => sum + value, 0), 'pooling conserved the positive count', 1e-12);
    /* The stack's own trace, not only the stepping schedule's. Every recorded
       merge must be a genuine violation: a stack that also merged EQUAL means
       would leave the fitted values untouched — so the max-min comparison above
       stays green — while silently collapsing two drawn blocks into one. The
       blocks are what the figure draws, so that is a visible defect with no
       numerical signature. */
    own.merges.forEach(merge => {
      assert(merge.left.mean > merge.right.mean,
        `a recorded merge does not violate the required order: ${merge.left.mean} then ${merge.right.mean}`);
      close(merge.merged.mean, (merge.left.total + merge.right.total) / (merge.left.weight + merge.right.weight),
        'a stack merge pools the combined counts', 1e-15);
      mergeTraces += 1;
    });
    /* Adjacent blocks that remain separate may share a mean; adjacent blocks
       must never DECREASE, which is the constraint the blocks encode. */
    own.blocks.forEach((block, index) => {
      if (index === 0) return;
      assert(own.blocks[index - 1].mean <= block.mean + 1e-15,
        'the block decomposition is not monotone');
    });
    /* The stepping schedule used by the investigation must reach the same fit
       by a different order of merges. */
    const stages = pavStages(scores, labels);
    vector(stages.fitted, own.fitted, 'the leftmost-violation schedule disagrees with the stack', 1e-12);
    assert.equal(stages.stages.length, stages.merges.length + 1, 'one stage per merge, plus the start');
    stages.merges.forEach(merge => {
      assert(merge.left.mean > merge.right.mean, 'a recorded merge must actually violate the order');
      close(merge.merged.mean, (merge.left.total + merge.right.total) / (merge.left.weight + merge.right.weight),
        'a merge pools the combined counts', 1e-15);
      assert.equal(merge.merged.weight, merge.left.weight + merge.right.weight, 'weights add');
      mergeTraces += 1;
    });
    assert.equal(nextRequiredMerge(stages.stages[stages.stages.length - 1]), null,
      'the final stage still has a required merge');
    /* Reordering the rows is an exact null. */
    const order = scores.map((_value, index) => (index * 5 + 2) % scores.length);
    const shuffled = fitPav(order.map(index => scores[index]), order.map(index => labels[index]));
    vector(shuffled.fitted, own.fitted, 'reordering the rows changed the fit', 1e-15);
    pavCases += 1;
  });
});
record('pav sweep');
assert(pavCases >= 500, `only ${pavCases} monotone-fit cases swept`);
assert(mergeTraces >= 200, `only ${mergeTraces} merges were traced`);

vector(fitPav(fixtures.pavScores, fixtures.pavLabels).fitted, checked.pav.fitted, 'the packet fit', 0);
vector(fitPav(fixtures.tiedScores, fixtures.tiedLabels).fitted, checked.pav_ties.fitted, 'the tied fit', 0);
vector(fitPav(fixtures.tiedScores, fixtures.tiedChangedLabels).fitted, checked.pav_ties_changed.fitted,
  'the changed tied fit', 0);
vector(fitPav(fixtures.practicePavScores, fixtures.practicePavLabels).fitted, checked.practice.pav.fitted,
  'the practice fit', 0);
/* The count-weighted trap, as its own claim. */
const tied = fitPav(fixtures.tiedScores, fixtures.tiedLabels);
close(tied.fitted[0], 1 / 3, 'the tied fixture pools to one third');
assert.notEqual(tied.fitted[0], (0.5 + 0) / 2,
  'the count-weighted pool must differ from the average of the two block means, or the trap is not a trap');
/* Equal scores can never receive different fitted values. */
scorePatterns.forEach(scores => {
  labellings(scores.length).filter((_value, index) => index % 11 === 0).forEach(labels => {
    const fit = fitPav(scores, labels);
    scores.forEach((left, i) => scores.forEach((right, j) => {
      if (left !== right) return;
      const knotI = fit.knots.indexOf(left);
      const knotJ = fit.knots.indexOf(right);
      assert.equal(fit.fitted[knotI], fit.fitted[knotJ], 'equal scores received different fitted values');
    }));
    record('tied-score grouping');
  });
});
/* Weights behave as repeated observations. */
[[1, 2, 3], [0, 0, 1]].forEach(scores => {
  labellings(3).forEach(labels => {
    const weighted = fitPav(scores, labels, [2, 2, 2]);
    const repeated = fitPav(scores.flatMap(value => [value, value]), labels.flatMap(value => [value, value]));
    vector(weighted.fitted, repeated.fitted, 'a weight of two disagrees with two copies', 1e-15);
    record('weighted equivalence');
  });
});

/* Interpolation between knots, and clipping outside them. */
const pav = fitPav(fixtures.pavScores, fixtures.pavLabels);
close(isotonicPredict(pav.knots, pav.fitted, -100), pav.fitted[0], 'clipped below the smallest knot', 0);
close(isotonicPredict(pav.knots, pav.fitted, 100), pav.fitted[pav.fitted.length - 1], 'clipped above', 0);
pav.knots.forEach((knot, index) => {
  close(isotonicPredict(pav.knots, pav.fitted, knot), pav.fitted[index], 'the value AT a knot is the fitted one', 0);
  record('knot interpolation');
});
for (let x = -3; x <= 4; x += 0.05) {
  const value = isotonicPredict(pav.knots, pav.fitted, x);
  assert(value >= 0 && value <= 1, 'an interpolated probability stays in the unit interval');
  const nearby = isotonicPredict(pav.knots, pav.fitted, x + 0.05);
  assert(nearby >= value - 1e-12, 'the interpolated map is not monotone');
  record('interpolation monotonicity');
}

/* ================================== §7 · the sigmoid fit against SciPy's */

const sigmoidFit = fitSigmoid(fixtures.pavScores, fixtures.pavLabels);
assert(sigmoidFit.converged, `the sigmoid solver did not converge: ${sigmoidFit.because}`);
close(sigmoidFit.a, checked.sigmoid.a, 'the slope against SciPy BFGS', 1e-8);
close(sigmoidFit.b, checked.sigmoid.b, 'the offset against SciPy BFGS', 1e-8);
vector(sigmoidFit.probabilities, checked.sigmoid.probabilities, 'the fitted probabilities', 1e-8);
vector(plattTargets(fixtures.pavLabels), checked.sigmoid.smoothed_targets, 'the smoothed targets', 1e-15);
assert(sigmoidFit.gradientNorm < 1e-10, 'the returned point is not a stationary point of the objective');
/* The objective at the fit is no worse than SciPy's, and no worse than every
   point of a fine grid around it. A minimiser that a grid beats is not one. */
const objectiveAt = (a, b) => {
  const targets = plattTargets(fixtures.pavLabels);
  return fixtures.pavScores.reduce((sum, score, index) => {
    const z = a * score + b;
    return sum + (z > 0 ? z + Math.log1p(Math.exp(-z)) : Math.log1p(Math.exp(z))) - targets[index] * z;
  }, 0) / fixtures.pavScores.length;
};
close(sigmoidFit.objective, objectiveAt(sigmoidFit.a, sigmoidFit.b), 'the reported objective', 1e-15);
assert(sigmoidFit.objective <= checked.sigmoid.objective + 1e-12,
  'the browser fit is worse than the packet\'s recorded objective');
let gridPoints = 0;
for (let a = sigmoidFit.a - 0.2; a <= sigmoidFit.a + 0.2; a += 0.005) {
  for (let b = sigmoidFit.b - 0.2; b <= sigmoidFit.b + 0.2; b += 0.005) {
    assert(objectiveAt(a, b) >= sigmoidFit.objective - 1e-12,
      `a grid point at a=${a}, b=${b} beat the reported minimum`);
    gridPoints += 1;
  }
}
record('sigmoid grid');
assert(gridPoints >= 5000, `only ${gridPoints} grid points checked around the sigmoid fit`);
/* This lab declines one-class fits as an evidence policy. Smoothed targets
   still admit a finite constant optimum; the refusal is not non-identifiability. */
const absentClass = fitSigmoid([1, 2, 3], [1, 1, 1]);
assert.equal(absentClass.a, null, 'a one-class sample must not produce a slope');
assert(!absentClass.converged && absentClass.because.includes('one outcome class'),
  'and must say why rather than reporting a failure to converge');
/* Convergence over a sweep of fixtures, with stationarity checked each time
   rather than assumed, and — where the search stalls against double precision —
   the returned point still shown to be a genuine local minimum by a grid. */
let sigmoidCases = 0;
let sigmoidStalled = 0;
let sigmoidRefused = 0;
scorePatterns.slice(0, 4).forEach(scores => {
  labellings(scores.length).forEach(labels => {
    const fit = fitSigmoid(scores, labels);
    if (fit.a === null) {
      assert(new Set(labels).size === 1, 'a fit was refused on a sample that carries both classes');
      sigmoidRefused += 1;
      sigmoidCases += 1;
      return;
    }
    assert(fit.converged, `the solver failed on ${scores} with ${labels}: ${fit.because}`);
    assert(fit.gradientNorm < 1e-7, 'a converged fit must be stationary to the precision the arithmetic allows');
    assert(fit.probabilities.every(value => value > 0 && value < 1),
      'smoothed targets should keep every fitted probability strictly inside the unit interval');
    if (fit.stalled) {
      sigmoidStalled += 1;
      /* A stalled search still has to have found a minimum. Nothing nearby may
         beat it, or "stalled" would be covering for a wrong answer. */
      const targets = plattTargets(labels);
      const value = (a, b) => scores.reduce((sum, score, index) => {
        const z = a * score + b;
        return sum + (z > 0 ? z + Math.log1p(Math.exp(-z)) : Math.log1p(Math.exp(z))) - targets[index] * z;
      }, 0) / scores.length;
      for (let da = -0.01; da <= 0.0100001; da += 0.002) {
        for (let db = -0.01; db <= 0.0100001; db += 0.002) {
          assert(value(fit.a + da, fit.b + db) >= fit.objective - 1e-13,
            `a stalled fit on ${scores} with ${labels} is beaten by a nearby point`);
        }
      }
    }
    sigmoidCases += 1;
  });
});
record('sigmoid sweep');
assert(sigmoidCases >= 300, `only ${sigmoidCases} sigmoid fits swept`);
assert(sigmoidRefused > 0, 'no one-class sample appeared, so the refusal branch was never exercised');
assert(sigmoidStalled > 0,
  'no fit stalled against double precision, so the grid check written for that case inspected nothing');

/* ==================================== §8 · temperature, softmax and entropy */

Object.entries(checked.temperature).forEach(([temperature, expected]) => {
  const own = softmaxAt(fixtures.temperatureLogits, Number(temperature));
  vector(own, expected, `softmax at T=${temperature}`, 1e-12);
  vector(own, independentSoftmax(fixtures.temperatureLogits, Number(temperature)),
    `softmax at T=${temperature} by the unshifted route`, 1e-12);
  close(own.reduce((sum, value) => sum + value, 0), 1, 'softmax sums to one', 1e-15);
  record('temperature fixture');
});
/* The winner and every tie survive any positive temperature. Swept, not
   asserted on the lesson's three logits alone. */
let temperatureCases = 0;
const logitSets = [[3, 1, 0], [1, 1, 0], [0, 0, 0], [-5, 2, 2, 7], [10, -10]];
nonEmpty(logitSets, 5, 'logit sets');
logitSets.forEach(logits => {
  for (let temperature = 0.05; temperature <= 20; temperature += 0.05) {
    const probabilities = softmaxAt(logits, temperature);
    const baseline = softmaxAt(logits, 1);
    assert.equal(leadingClass(probabilities).index, leadingClass(baseline).index,
      `temperature ${temperature} changed the winning class on ${logits}`);
    assert.deepEqual(leadingClass(probabilities).winners, leadingClass(baseline).winners,
      `temperature ${temperature} broke or created a tie on ${logits}`);
    /* Order is preserved pairwise, not merely at the top. */
    for (let i = 0; i < logits.length; i += 1) {
      for (let j = 0; j < logits.length; j += 1) {
        assert.equal(Math.sign(probabilities[i] - probabilities[j]) || 0,
          Math.sign(baseline[i] - baseline[j]) || 0,
          `temperature ${temperature} reordered classes ${i} and ${j}`);
      }
    }
    assert(entropyNats(probabilities) >= -1e-15, 'entropy is non-negative');
    assert(entropyNats(probabilities) <= Math.log(logits.length) + 1e-12, 'entropy cannot exceed log K');
    temperatureCases += 1;
  }
});
record('temperature sweep');
assert(temperatureCases >= 1900, `only ${temperatureCases} temperature cases swept`);
/* Raising the temperature raises the entropy, which is the claim the figure
   makes with its bars. */
logitSets.filter(logits => new Set(logits).size > 1).forEach(logits => {
  let previous = -Infinity;
  [0.5, 1, 2, 4, 8].forEach(temperature => {
    const value = entropyNats(softmaxAt(logits, temperature));
    assert(value >= previous - 1e-12, `entropy fell as the temperature rose on ${logits}`);
    previous = value;
  });
  record('entropy monotonicity');
});
close(calibrationData.constructedRecord.temperatureFit.temperature, checked.temperature_fit.temperature,
  'the recorded fitted temperature', 0);
assert(calibrationData.constructedRecord.temperatureFit.nllAfter
  < calibrationData.constructedRecord.temperatureFit.nllBefore,
  'the fitted temperature must reduce the log loss it was fitted on');
assert.equal(calibrationData.constructedRecord.temperatureFit.nearSearchBoundary, false,
  'the recorded fit must state that it did not clip against its search bound');

/* ====================== §9 · adaptive intervals, units and the CQR sign */

const normalizedScores = fixtures.residuals.map((value, index) => value / fixtures.localScales[index]);
close(conformalThreshold(fixtures.residuals, fixtures.intervalAlpha).q, checked.adaptive.absolute_q,
  'the absolute threshold', 0);
close(conformalThreshold(normalizedScores, fixtures.intervalAlpha).q, checked.adaptive.normalized_q,
  'the normalised threshold', 0);
close(conformalThreshold(
  fixtures.changedResiduals.map((value, index) => value / fixtures.localScales[index]), fixtures.intervalAlpha).q,
  checked.adaptive.changed_q, 'the changed threshold', 0);
close(conformalThreshold(
  fixtures.residuals.map((value, index) => value / (2 * fixtures.localScales[index])), fixtures.intervalAlpha).q,
  checked.adaptive.all_scales_doubled_q, 'the doubled-scale threshold', 0);
const normalizedThreshold = conformalThreshold(normalizedScores, fixtures.intervalAlpha);
const easy = normalizedInterval(fixtures.queries[0].centre, fixtures.queries[0].localScale, normalizedThreshold.q);
const hard = normalizedInterval(fixtures.queries[1].centre, fixtures.queries[1].localScale, normalizedThreshold.q);
close(easy.lower, 8, 'the easy query lower endpoint', 1e-12);
close(easy.upper, 12, 'the easy query upper endpoint', 1e-12);
close(hard.lower, 14, 'the hard query lower endpoint', 1e-12);
close(hard.upper, 26, 'the hard query upper endpoint', 1e-12);
close(easy.width, 4, 'the easy query width', 1e-12);
close(hard.width, 12, 'the hard query width', 1e-12);
const absoluteThreshold = conformalThreshold(fixtures.residuals, fixtures.intervalAlpha);
close(normalizedInterval(fixtures.queries[0].centre, 1, absoluteThreshold.q).width,
  normalizedInterval(fixtures.queries[1].centre, 1, absoluteThreshold.q).width,
  'the absolute score gives every query the same width', 1e-15);
record('adaptive fixtures');

/* The unit null, swept over factors rather than shown once: multiplying every
   calibration scale AND every query scale by the same factor divides the
   dimensionless threshold by it and leaves the physical interval untouched. */
let unitCases = 0;
[0.5, 1.5, 2, 3, 7.5, 10].forEach(factor => {
  const scaled = conformalThreshold(
    fixtures.residuals.map((value, index) => value / (factor * fixtures.localScales[index])),
    fixtures.intervalAlpha);
  close(scaled.q, normalizedThreshold.q / factor, `the threshold under a scale factor of ${factor}`, 1e-12);
  fixtures.queries.forEach(query => {
    const before = normalizedInterval(query.centre, query.localScale, normalizedThreshold.q);
    const after = normalizedInterval(query.centre, query.localScale * factor, scaled.q);
    close(after.lower, before.lower, `the physical lower endpoint moved under a factor of ${factor}`, 1e-12);
    close(after.upper, before.upper, `the physical upper endpoint moved under a factor of ${factor}`, 1e-12);
    close(after.width, before.width, `the physical width moved under a factor of ${factor}`, 1e-12);
    assert.equal(changeDirection(after.width, before.width), 'unchanged',
      `the verdict must report the unit change as unchanged at factor ${factor}`);
    assert.equal(gradedText(after.width), gradedText(before.width),
      `and the displayed width must be identical at factor ${factor}`);
    unitCases += 1;
  });
});
record('unit invariance');
assert(unitCases >= 12, `only ${unitCases} unit-invariance cases swept`);

/* The contrast the prose promises: changing the last two residuals raises the
   threshold from 2 to 4 and doubles both half-widths. */
const changedThreshold = conformalThreshold(
  fixtures.changedResiduals.map((value, index) => value / fixtures.localScales[index]), fixtures.intervalAlpha);
close(changedThreshold.q, 2 * normalizedThreshold.q, 'the changed threshold is exactly double', 1e-12);
fixtures.queries.forEach(query => {
  const before = normalizedInterval(query.centre, query.localScale, normalizedThreshold.q);
  const after = normalizedInterval(query.centre, query.localScale, changedThreshold.q);
  close(after.width, 2 * before.width, `${query.name}: the width did not double`, 1e-12);
  assert.equal(changeDirection(after.width, before.width), 'higher', 'and the verdict says higher');
  assert.notEqual(gradedText(after.width), gradedText(before.width), 'and the displayed width moved');
  record('width contrast');
});

/* CQR: the signed score, the negative adjustment and the empty set. */
close(conformalThreshold(fixtures.cqrScores, fixtures.cqrAlpha).q, checked.practice.cqr_negative_q,
  'the negative CQR threshold', 0);
close(conformalThreshold(fixtures.cqrAlternativeScores, fixtures.cqrAlpha).q, checked.practice.cqr_q,
  'the positive CQR threshold', 0);
const negativeCqr = cqrInterval(fixtures.cqrBase.lower, fixtures.cqrBase.upper,
  conformalThreshold(fixtures.cqrScores, fixtures.cqrAlpha).q);
close(negativeCqr.lower, 12, 'the shrunken lower endpoint', 1e-15);
close(negativeCqr.upper, 18, 'the shrunken upper endpoint', 1e-15);
assert(!negativeCqr.empty, 'the shrunken interval is not empty');
const crossed = cqrInterval(10, 12, -3);
assert(crossed.empty, 'endpoints that cross must give an empty set');
assert.equal(crossed.width, 0, 'and an empty set has no positive width, and certainly not a negative one');
assert.equal(changeDirection(null, negativeCqr.width), 'undefined',
  'an empty set has no width to compare, which is a fourth answer and not zero');
assert.equal(changeDirection(negativeCqr.width, null), 'defined',
  'and coming back from an empty set is a fifth');
/* The score's own definition, swept: it is positive outside the interval, and
   non-positive inside. */
let cqrCases = 0;
for (let lower = -5; lower <= 5; lower += 0.5) {
  for (let upper = lower; upper <= lower + 10; upper += 0.5) {
    for (let y = lower - 4; y <= upper + 4; y += 0.5) {
      const score = cqrScore(lower, upper, y);
      const inside = y >= lower && y <= upper;
      assert.equal(score <= 1e-12, inside || Math.abs(score) < 1e-12,
        `the CQR score sign disagrees with membership at L=${lower}, U=${upper}, y=${y}`);
      if (!inside) {
        close(score, Math.min(Math.abs(y - lower), Math.abs(y - upper)),
          'outside, the score is the distance to the nearer violated endpoint', 1e-12);
      }
      /* A label is in the conformalised set exactly when its score is at most q. */
      [-2, -0.5, 0, 0.5, 3].forEach(q => {
        const interval = cqrInterval(lower, upper, q);
        const member = !interval.empty && y >= interval.lower && y <= interval.upper;
        assert.equal(member, !interval.empty && score <= q + 1e-12,
          `set membership disagrees with the score comparison at L=${lower}, U=${upper}, y=${y}, q=${q}`);
      });
      cqrCases += 1;
    }
  }
}
record('cqr sweep');
assert(cqrCases >= 4000, `only ${cqrCases} CQR score cases swept`);
/* The rearrangement is fixed and pointwise, and applies the same way whatever
   the true answer is. */
[[3, 1], [1, 3], [2, 2]].forEach(([lower, upper]) => {
  const fixed = rearrangeQuantiles(lower, upper);
  assert(fixed.lower <= fixed.upper, 'the rearrangement must not leave crossed endpoints');
  assert.equal(fixed.crossed, lower > upper, 'and reports whether it had to act');
  record('quantile rearrangement');
});

/* ================================ §10 · group coverage and the shift example */

const mosaic = mosaicMarginal(fixtures.mosaic);
close(mosaic.marginal, checked.group_example.marginal, 'the marginal coverage', 1e-15);
close(mosaic.worstGroup.coverage, 0.5, 'the worst group\'s coverage', 0);
close(mosaic.groups.reduce((sum, group) => sum + group.contribution, 0), mosaic.marginal,
  'the contributions add to the marginal', 1e-15);
close(positivePredictiveValue(0.8, 0.2, 0.5).value, checked.label_shift.ppv_at_prior_half, 'PPV at prevalence .5');
close(positivePredictiveValue(0.8, 0.2, 0.1).value, checked.label_shift.ppv_at_prior_tenth, 'PPV at prevalence .1');
assert.equal(positivePredictiveValue(0, 0, 0.5).value, null,
  'a test nobody tests positive on has no posterior for a positive result');
assert(positivePredictiveValue(0, 0, 0.5).because.includes('no one'), 'and says why');
/* The sensitivity and false-positive rate are fixed; only the population moves. */
let shiftCases = 0;
for (let prevalence = 0.01; prevalence <= 0.99; prevalence += 0.01) {
  const own = positivePredictiveValue(0.8, 0.2, prevalence);
  close(own.value, (0.8 * prevalence) / (0.8 * prevalence + 0.2 * (1 - prevalence)),
    `PPV at prevalence ${prevalence}`, 1e-12);
  assert(own.value > 0 && own.value < 1, 'a posterior lies strictly inside the unit interval here');
  shiftCases += 1;
}
record('prevalence sweep');
assert(shiftCases >= 90, `only ${shiftCases} prevalence cases swept`);
const beta = orderStatisticCoverageMean(73, 80);
close(beta.mean, 73 / 81, 'the Beta mean is k over n plus one', 1e-15);
assert.equal(beta.beta, 8, 'and its second parameter is n + 1 − k');
assert(beta.conditions.includes('iid') && beta.conditions.includes('no ties'),
  'the Beta statement carries its conditions');
record('order statistic mean');

/* A coverage report cannot be produced without a denominator, which is the one
   discipline this topic most needs. */
const coverage = coverageReport(sigmoidFit ? 71 : 0, 80, 'assessment');
close(coverage.rate, 71 / 80, 'a coverage rate', 1e-15);
assert.equal(coverage.text, '71 of 80', 'and is reported with its count');
assert.equal(coverage.kind, 'assessment', 'and its kind');
record('coverage report');

/* ==================== §11 · the grading rules, at their degenerate inputs */

const directionCases = [
  [1, 1, 'unchanged'], [0, 0, 'unchanged'], [-0, 0, 'unchanged'],
  [1e-13, 0, 'unchanged'], [2e-12, 0, 'higher'], [-2e-12, 0, 'lower'],
  [null, 1, 'undefined'], [1, null, 'defined'], [null, null, 'undefined'],
  [undefined, 1, 'undefined'],
];
directionCases.forEach(([after, before, expected]) => {
  assert.equal(changeDirection(after, before), expected,
    `changeDirection(${after}, ${before}) should be ${expected}`);
  record('direction degenerate');
});
assert.equal(thresholdDirection(Infinity, Infinity), 'unchanged', 'two unbounded thresholds are unchanged');
assert.equal(thresholdDirection(Infinity, 0.6), 'higher', 'becoming unbounded is a direction');
assert.equal(thresholdDirection(0.6, Infinity), 'lower', 'and so is coming back from one');
assert.equal(thresholdDirection(0.6, 0.6), 'unchanged', 'and a finite tie is unchanged');
refuses(() => changeDirection(Infinity, 1), 'an infinite value through the finite grading rule');
record('threshold direction');

/* The equivalence the learner sees: "unchanged" exactly when the printed number
   did not move, over the grid of values these labs can actually produce. */
let verdictCases = 0;
let verdictUnchanged = 0;
let verdictMoved = 0;
const gradedValues = [];
/* ECE and Brier values, from real fixtures. */
cardPatterns.forEach(pattern => {
  labellings(10).filter((_value, index) => index % 61 === 0).forEach(labels => {
    edgeSets.forEach(edges => gradedValues.push(reliability(pattern, labels, edges).ece));
    gradedValues.push(brierLoss(pattern, labels));
  });
});
/* Interval widths, from real fixtures. */
[0.5, 1, 1.5, 2, 3].forEach(factor => {
  fixtures.queries.forEach(query => {
    const threshold = conformalThreshold(normalizedScores.map(value => value * factor), fixtures.intervalAlpha);
    gradedValues.push(normalizedInterval(query.centre, query.localScale, threshold.q).width);
  });
});
/* Thresholds themselves. */
thresholdSubjects.forEach(scores => {
  [0.1, 0.2, 0.3].forEach(alpha => {
    const threshold = conformalThreshold(scores, alpha);
    if (threshold.finite) gradedValues.push(threshold.q);
  });
});
nonEmpty(gradedValues, undefined, 'graded values');
assert(gradedValues.length >= 200, `only ${gradedValues.length} graded values collected`);
gradedValues.forEach(after => {
  gradedValues.forEach(before => {
    const direction = changeDirection(after, before);
    const same = gradedText(after) === gradedText(before);
    assert.equal(direction === 'unchanged', same,
      `the verdict and the printed number disagree: changeDirection(${after}, ${before}) is "${direction}" while `
      + `the page prints "${gradedText(after)}" and "${gradedText(before)}"`);
    if (direction === 'unchanged') verdictUnchanged += 1; else verdictMoved += 1;
    verdictCases += 1;
  });
});
record('verdict equivalence');
assert(verdictCases >= 40000, `only ${verdictCases} verdict pairs swept`);

/* The pairs above are the values these labs produce, and they happen to sit far
   apart. That leaves the band where a display could hide a real change
   unvisited — the classic inert guard, correct within a domain that excludes
   the defect. So each value is also compared against itself plus a deliberate
   offset, chosen to be outside the tolerance and resolvable at the declared
   precision. A display coarser than the tolerance fails here. */
let nearCases = 0;
/* Relative, because the tolerance is: at a width of 24 the unchanged band is
   2.4e-11 wide, so a fixed 2e-11 offset would be inside it and the check would
   be asserting the opposite of what it says. */
const offsets = [1e-10, 5e-7, 1e-4];
gradedValues.forEach(value => {
  const scale = Math.abs(value) > 1 ? Math.abs(value) : 1;
  offsets.forEach(relative => {
    const offset = relative * scale;
    const moved = value + offset;
    assert.equal(changeDirection(moved, value), 'higher',
      `an offset of ${offset} from ${value} should be a genuine rise`);
    assert.notEqual(gradedText(moved), gradedText(value),
      `a rise of ${offset} from ${value} prints as "${gradedText(value)}" either way: the graded display is `
      + 'coarser than the tolerance, so a learner would be told the number moved while seeing it stand still');
    assert.equal(changeDirection(value, value), 'unchanged', 'a value against itself is unchanged');
    assert.equal(gradedText(value), gradedText(value), 'and prints identically');
    nearCases += 1;
  });
});
record('near-tolerance display');
assert(nearCases >= 2000, `only ${nearCases} near-tolerance pairs swept`);
assert(verdictUnchanged > 0 && verdictMoved > 0,
  'the verdict sweep saw only one answer, so it established nothing about the other');
assert.equal(gradedDisplayDigits, 12, 'the graded display precision is the one the sweep was run against');
assert.equal(gradedText(null), 'no value', 'a quantity with no value prints as such, not as zero');
assert.equal(gradedText(Infinity), 'unbounded', 'and an unbounded one is not printed as a number');
assert(gradedText(-0.5).includes('−'), 'a negative graded value prints a typographic minus');

/* ============================================== §12 · the drawn geometry */

let geometryCases = 0;
const reliabilityDrawings = [];
cardPatterns.forEach(pattern => {
  labellings(10).filter((_value, index) => index % 89 === 0).forEach(labels => {
    edgeSets.forEach(edges => reliabilityDrawings.push(reliability(pattern, labels, edges)));
  });
});
nonEmpty(reliabilityDrawings, undefined, 'reliability drawings');
assert(reliabilityDrawings.length >= 40, `only ${reliabilityDrawings.length} reliability drawings inspected`);
reliabilityDrawings.forEach(report => {
  const geometry = reliabilityGeometry(report);
  const { box } = geometry;
  /* A point is drawn for every occupied bin and for no empty one. */
  assert.equal(geometry.points.length, report.bins.filter(bin => bin.count > 0).length,
    'the drawing must have one dot per occupied bin');
  assert.equal(geometry.emptyBins.length, report.bins.filter(bin => bin.count === 0).length,
    'and must still list the empty bins, without dots');
  geometry.points.forEach(point => {
    close(point.x, geometry.scales.x(point.meanP), 'the dot\'s x encodes the mean forecast', 1e-12);
    close(point.y, geometry.scales.y(point.fractionPositive), 'and its y the observed fraction', 1e-12);
    close(point.diagonalY, geometry.scales.y(point.meanP), 'the gap line drops to the diagonal', 1e-12);
    assert(point.x >= box.left - 1e-9 && point.x <= box.width - box.right + 1e-9, 'a dot is inside the box');
    assert(point.y >= box.top - 1e-9 && point.y <= box.height - box.bottom + 1e-9, 'vertically too');
    /* The sign of the drawn offset must agree with the sign of the gap: a dot
       above the diagonal in the data must be above it on screen. Screen y
       decreases upward, so a positive gap gives a SMALLER y. */
    if (Math.abs(point.gap) > 1e-15) {
      assert.equal(point.y < point.diagonalY, point.gap > 0,
        `a bin with gap ${point.gap} is drawn on the wrong side of the diagonal`);
    }
    /* The bin number must not have the gap line running through it. The label
       band and the drop are both vertical segments at the same x, so this is a
       one-dimensional overlap test. Drawn only when the drop is long enough to
       be visible; a bin sitting on the diagonal has no line to avoid. */
    const [bandTop, bandBottom] = point.labelBand;
    assert(bandTop < bandBottom, 'the label band is ordered');
    assert.equal(point.labelAbove, bandBottom <= point.y,
      'the label band is on the side the label is on');
    if (Math.abs(point.y - point.diagonalY) > 1) {
      const dropTop = Math.min(point.y, point.diagonalY);
      const dropBottom = Math.max(point.y, point.diagonalY);
      const overlaps = bandTop < dropBottom && bandBottom > dropTop;
      assert(!overlaps || Math.abs(point.labelX - point.x) >= 8,
        `bin ${point.index}'s number is drawn inside its own gap line (band `
        + `${bandTop.toFixed(1)}–${bandBottom.toFixed(1)}, drop ${dropTop.toFixed(1)}–${dropBottom.toFixed(1)}) `
        + 'and has not stepped aside from it horizontally either');
      if (!overlaps) close(point.labelX, point.x, 'a label that clears its drop is not displaced', 1e-12);
    }
    geometryCases += 1;
  });
  /* The diagonal really is the identity line. */
  close(geometry.diagonal.x1, geometry.scales.x(0), 'the diagonal starts at zero', 1e-12);
  close(geometry.diagonal.y1, geometry.scales.y(0), 'on both axes', 1e-12);
  close(geometry.diagonal.x2, geometry.scales.x(1), 'and ends at one', 1e-12);
  close(geometry.diagonal.y2, geometry.scales.y(1), 'on both axes', 1e-12);
  /* The count rail is proportional with no minimum: a bin holding nothing must
     have a height of exactly zero rather than a stub that looks like evidence. */
  const largest = Math.max(...report.bins.map(bin => bin.count));
  geometry.rail.forEach(bar => {
    close(bar.height, largest > 0 ? (bar.count / largest) * geometry.railHeight : 0,
      'a count bar is proportional to its count', 1e-12);
    if (bar.count === 0) assert.equal(bar.height, 0, 'an empty bin\'s count bar has no length at all');
    /* What is DRAWN. A bin holding something is drawn at exactly its quantity —
       no minimum, so a bin with one row of many is a sliver and not a stub that
       reads as evidence. An empty bin is drawn as the declared marker and is
       flagged so the component can paint it as a dashed outline rather than a
       filled bar. */
    assert.equal(bar.empty, bar.count === 0, 'the empty flag disagrees with the count');
    if (bar.count > 0) {
      assert.equal(bar.markerHeight, bar.height,
        'a bin that holds something must be drawn at exactly its proportional height');
    } else {
      assert.equal(bar.markerHeight, geometry.emptyMarkerHeight,
        'an empty bin must be drawn at the declared marker height, not at an invented minimum');
      assert(geometry.emptyMarkerHeight > 0,
        'the empty marker must be visible, or an empty bin disappears rather than showing it has no fraction');
    }
    assert(bar.x2 > bar.x1, 'a bin occupies a positive width on the axis');
    geometryCases += 1;
  });
  /* The count rail must clear the tick-label row beneath the axis. The bars
     first grew upward from twenty units under the axis, straight through labels
     that sit fourteen under it: every number right, the labels covered, and
     nothing offline able to see it. */
  assert(geometry.railTopEdge > geometry.tickLabelY + 4,
    `the count rail's top edge at ${geometry.railTopEdge} does not clear the tick labels at `
    + `${geometry.tickLabelY}, so the bars are drawn over them`);
  assert(geometry.railBaseline <= box.height - 10,
    'the count rail runs past the bottom of the drawing');
  assert(geometry.tickLabelY < geometry.railTopEdge && geometry.railBaseline < box.height,
    'the rail band and the tick-label row are not in the order the drawing assumes');
  /* The printed count sits under the bar and above the axis name. It exists
     because the bar length alone cannot separate a bin holding one row from an
     empty one, and this lesson's whole claim for the rail is that a dot without
     its denominator hides exactly that. */
  assert(geometry.countLabelY > geometry.railBaseline,
    'the printed counts are drawn inside the bars rather than under them');
  assert(geometry.countLabelY < box.height - 8,
    'the printed counts run off the bottom of the drawing');
  geometry.rail.forEach(bar => {
    close(bar.countLabelX, (bar.x1 + bar.x2) / 2, 'a count is printed under the middle of its own bin', 1e-12);
    geometryCases += 1;
  });
  geometryCases += 5;
  const total = geometry.rail.reduce((sum, bar) => sum + bar.count, 0);
  assert.equal(total, report.count, 'the rail accounts for every card');
});
record('reliability geometry');

/* Block rectangles: height is the pooled probability, width is the span. */
scorePatterns.forEach(scores => {
  labellings(scores.length).filter((_value, index) => index % 13 === 0).forEach(labels => {
    const fit = fitPav(scores, labels);
    const geometry = blockGeometry(fit);
    assert.equal(geometry.blocks.length, fit.blocks.length, 'one rectangle per block');
    geometry.blocks.forEach(block => {
      close(block.y, geometry.scales.y(block.mean), 'a block\'s top encodes its probability', 1e-12);
      close(block.height, geometry.baseline - geometry.scales.y(block.mean),
        'and its height is measured from the baseline', 1e-12);
      assert.equal(block.spans, block.end - block.start + 1, 'a block reports how many knots it covers');
      assert(block.x2 > block.x1, 'a block occupies a positive width');
      if (block.mean === 0) assert.equal(block.height, 0, 'a block at probability zero has no height');
      geometryCases += 1;
    });
    geometry.knots.forEach((knot, index) => {
      close(knot.rawY, geometry.scales.y(fit.totals[index] / fit.weights[index]),
        'the raw mark encodes the unpooled rate', 1e-12);
      close(knot.fittedY, geometry.scales.y(fit.fitted[index]), 'and the fitted mark the pooled one', 1e-12);
      geometryCases += 1;
    });
  });
});
record('block geometry');

/* The score rail, including the case where there is no point to draw. */
thresholdSubjects.forEach(scores => {
  [0.05, 0.1, 0.2, 0.4].forEach(alpha => {
    const threshold = conformalThreshold(scores, alpha);
    const geometry = railGeometry(scores, threshold, { domain: [0, Math.max(1, ...scores)] });
    assert.equal(geometry.marks.length, scores.length, 'one mark per calibration card');
    geometry.marks.forEach((mark, index) => {
      assert.equal(mark.rank, index + 1, 'marks are ranked from the smallest');
      close(mark.x, geometry.scales.x(mark.score), 'a mark\'s position encodes its score', 1e-12);
      assert.equal(mark.belowThreshold, mark.score <= threshold.q,
        'the below-threshold flag agrees with the weak comparison');
      /* The rank label of an unselected mark sits on its mark; the selected
         one steps aside, because the threshold rule is drawn at exactly that
         x through the full height of the figure and ran down the middle of
         the digit. */
      if (!mark.selected) {
        close(mark.labelX, mark.x, 'an unselected rank label sits on its own mark', 1e-12);
      } else {
        assert(Math.abs(mark.labelX - geometry.markerX) >= 8,
          `the selected rank label is drawn ${Math.abs(mark.labelX - geometry.markerX).toFixed(2)}px from the `
          + 'threshold rule, which is close enough for the rule to pass through the digit');
      }
      geometryCases += 1;
    });
    const selected = geometry.marks.filter(mark => mark.selected);
    if (threshold.finite) {
      assert.equal(selected.length, 1, 'exactly one mark is the selected order statistic');
      close(selected[0].score, threshold.q, 'and it is the threshold', 0);
      close(geometry.markerX, geometry.scales.x(threshold.q), 'the marker line sits at the threshold', 1e-12);
      assert(!geometry.unbounded, 'a finite threshold is not flagged unbounded');
    } else {
      assert.equal(geometry.markerX, null,
        'an unbounded threshold must not be drawn as a point on the axis, however far right');
      assert(geometry.unbounded, 'and must be flagged as unbounded');
      assert.equal(selected.length, 0, 'and no card is the selected one');
    }
    geometryCases += 1;
  });
});
record('rail geometry');

/* Interval segments: drawn length is the width, with no minimum length. */
const intervalRows = [
  { name: 'wide', lower: 0, upper: 10, y: 5 },
  { name: 'narrow', lower: 4.9, upper: 5.1, y: 5 },
  { name: 'zero width', lower: 5, upper: 5, y: 5 },
  { name: 'missed', lower: 0, upper: 1, y: 9 },
];
const intervalDrawing = intervalGeometry(intervalRows);
nonEmpty(intervalDrawing.rows, 4, 'interval rows');
const unitLength = (intervalDrawing.rows[0].x2 - intervalDrawing.rows[0].x1) / intervalDrawing.rows[0].width;
intervalDrawing.rows.forEach(row => {
  close(row.x2 - row.x1, row.width * unitLength, 'a segment\'s drawn length is its width, with no offset', 1e-9);
  close(row.target, intervalDrawing.scales.x(row.value), 'the target mark encodes the observed value', 1e-12);
  assert.equal(row.covered, row.lower <= row.value && row.value <= row.upper,
    'the covered flag is computed from the same endpoints the segment is drawn from');
  assert.notEqual(row.value, row.y,
    'the observed value and the screen row must be separate fields, or a figure reading one gets the other');
  geometryCases += 1;
});
assert.equal(intervalDrawing.rows[2].x2 - intervalDrawing.rows[2].x1, 0,
  'a zero-width interval must be drawn with zero length, not a minimum stub');
assert(!intervalDrawing.rows[3].covered, 'a target outside its band must not be flagged covered');
record('interval geometry');

/* Bars are proportional with no minimum, and the mosaic's filled area IS the
   marginal coverage. */
const bars = barGeometry([{ name: 'a', value: 0 }, { name: 'b', value: 1 }, { name: 'c', value: 4 }],
  { width: 200 });
close(bars[0].length, 0, 'a zero bar has no length', 0);
close(bars[1].length, 50, 'a quarter bar is a quarter of the width', 1e-12);
close(bars[2].length, 200, 'the largest bar fills the width', 1e-12);
assert.deepEqual(barGeometry([{ name: 'a', value: 0 }], { width: 200 }).map(bar => bar.length), [0],
  'a single zero bar still has no length');
record('bar geometry');
const mosaicDrawing = mosaicGeometry(mosaic, { width: 200, height: 100 });
close(mosaicDrawing.coveredArea, mosaic.marginal, 'the filled fraction is the marginal coverage', 1e-15);
mosaicDrawing.groups.forEach(group => {
  close(group.width, group.share * 200, 'a column\'s width is its share', 1e-12);
  close(group.coveredHeight, group.coverage * 100, 'its filled height is its coverage', 1e-12);
  close(group.coveredHeight + group.missedHeight, 100, 'the two parts fill the column', 1e-12);
  close(group.area, group.share * group.coverage, 'and the filled area is their product', 1e-15);
  geometryCases += 1;
});
close(mosaicDrawing.groups.reduce((sum, group) => sum + group.width, 0), 200,
  'the columns fill the drawing exactly', 1e-12);
record('mosaic geometry');

/* The logarithmic frequency axis really is logarithmic, and the split line
   lands where the split value does. */
const widths = calibrationData.regression.pointPredictions.map((value, index) =>
  ({ x: calibrationData.regression.testFrequencyHz[index], y: 2 * calibrationData.regression.qAbsolute, index }));
const scatter = scatterGeometry(widths, { xLog: true, box: { width: 320, height: 200 } });
assert.equal(scatter.axisKind, 'log10', 'the scatter declares a logarithmic axis rather than leaving it implied');
nonEmpty(scatter.points, 120, 'scatter points');
scatter.points.forEach(point => {
  close(point.cx, scatter.scales.x(point.x), 'a point\'s x is the log-mapped frequency', 1e-12);
  close(point.cy, scatter.scales.y(point.y), 'and its y the width', 1e-12);
  geometryCases += 1;
});
/* Equal ratios map to equal distances, which is what a log axis means. */
[[200, 400], [1000, 2000], [5000, 10000]].forEach(([low, high]) => {
  close(scatter.scales.x(high) - scatter.scales.x(low),
    scatter.scales.x(2000) - scatter.scales.x(1000), 'a doubling is the same distance everywhere', 1e-9);
  record('log axis');
});
record('scatter geometry');
assert(geometryCases >= 1500, `only ${geometryCases} geometry checks ran`);

/* ==================================== §13 · the generated data module */

const classification = calibrationData.classification;
const regression = calibrationData.regression;
assert.equal(classification.rank, 73, 'the banknote rank');
assert.equal(regression.rank, 109, 'the airfoil rank');
assert.equal(classification.alpha, 0.1, 'the banknote alpha');
assert.equal(conformalRank(80, classification.alpha), classification.rank,
  'the stored banknote rank is the one the module computes');
assert.equal(conformalRank(120, regression.alpha), regression.rank,
  'and the same for the airfoil rank');
nonEmpty(classification.methodOrder, 5, 'classification methods');
nonEmpty(regression.methodOrder, 4, 'regression methods');
nonEmpty(classification.testLabels, 80, 'assessment labels');
nonEmpty(regression.testY, 120, 'assessment responses');

/* Every displayed coverage count is reproduced from the stored probabilities
   and the stored threshold under the same weak comparison, so the browser's
   reconstruction is checked rather than trusted. */
let measuredCases = 0;
classification.methodOrder.forEach(key => {
  const row = classification.methods[key];
  const recorded = experiment.classification.methods[key];
  assert.equal(row.rank, recorded.rank, `${key} rank`);
  close(row.q, recorded.q, `${key} threshold`, 1e-12);
  vector(row.testProbabilities, recorded.test_probabilities_class1, `${key} probabilities`, 1e-12);
  vector(row.calibrationScores, recorded.calibration_scores, `${key} calibration scores`, 1e-12);
  /* The threshold is the kth smallest of the CALIBRATION scores, not of
     anything on the assessment rows. */
  close(conformalThreshold(row.calibrationScores, classification.alpha).q, row.q,
    `${key}: the stored threshold is not the rank statistic of the stored calibration scores`, 1e-12);
  const sets = row.testProbabilities.map(probability => classSets([1 - probability, probability], row.q));
  sets.forEach((set, index) => {
    assert.deepEqual(set.included, recorded.test_sets[index],
      `${key} row ${index}: the browser's set differs from the offline run`);
    measuredCases += 1;
  });
  const covered = sets.filter((set, index) => set.included[classification.testLabels[index]]).length;
  assert.equal(covered, row.covered, `${key}: recomputed coverage disagrees with the stored count`);
  assert.equal(covered, recorded.covered, `${key}: and with the offline run`);
  const sizes = sets.map(set => set.size);
  close(sizes.reduce((sum, value) => sum + value, 0) / sizes.length, row.meanSetSize, `${key} mean set size`, 1e-12);
  [0, 1, 2].forEach(size => {
    assert.equal(sizes.filter(value => value === size).length, row.sizeCounts[String(size)],
      `${key}: the ${size}-class count disagrees`);
  });
  close(brierLoss(row.testProbabilities, classification.testLabels), row.brier, `${key} Brier`, 1e-12);
  close(rocAuc(row.testProbabilities, classification.testLabels).value, row.auc, `${key} AUC`, 1e-12);
  const rebuilt = reliabilityFromRecord(row.reliability, classification.binEdges);
  close(rebuilt.ece, row.reliability.ece, `${key} reliability ECE survives the adapter`, 1e-12);
  assert.equal(rebuilt.bins.length, classification.binEdges.length - 1, `${key} bin count`);
  rebuilt.bins.forEach(bin => {
    if (bin.count === 0) {
      assert.equal(bin.fractionPositive, null, `${key}: an empty measured bin has no fraction`);
    } else {
      close(bin.gap, bin.fractionPositive - bin.meanP, `${key}: the measured gap is the signed difference`, 1e-12);
    }
  });
  record('measured method');
});
assert(measuredCases >= 400, `only ${measuredCases} measured set memberships checked`);
assert.equal(classification.methods.prior_constant.covered, 80,
  'the constant baseline must cover every assessment row under the direct score comparison');
close(classification.methods.prior_constant.meanSetSize, 2, 'and return both labels every time', 0);
close(classification.trainingPrior, 103 / 240, 'the training prior is 103 of 240', 1e-15);
const boundaryCheck = floatingBoundaryCheck(classification.trainingPrior);
assert.equal(boundaryCheck.scoreComparison, checked.floating_boundary.score_comparison,
  'the direct score comparison keeps the tie');
assert.equal(boundaryCheck.rearrangedComparison, checked.floating_boundary.rearranged_comparison,
  'and the rearranged one loses it');
assert(!boundaryCheck.agree,
  'the two algebraically equal comparisons must disagree here, or the fixture demonstrates nothing');
record('tied boundary');

/* The interval endpoints the figure rebuilds must reproduce the offline run. */
regression.methodOrder.forEach(key => {
  const recorded = experiment.regression.methods[key];
  const rebuilt = key === 'constant'
    ? { lower: regression.testY.map(() => regression.trainingMean - regression.qConstant),
      upper: regression.testY.map(() => regression.trainingMean + regression.qConstant) }
    : key === 'ridge_absolute'
      ? { lower: regression.pointPredictions.map(value => value - regression.qAbsolute),
        upper: regression.pointPredictions.map(value => value + regression.qAbsolute) }
      : key === 'raw_quantiles'
        ? { lower: regression.rawLower, upper: regression.rawUpper }
        : { lower: regression.rawLower.map(value => value - regression.qCqr),
          upper: regression.rawUpper.map(value => value + regression.qCqr) };
  vector(rebuilt.lower, recorded.lower, `${key} lower endpoints rebuilt in the browser`, 1e-12);
  vector(rebuilt.upper, recorded.upper, `${key} upper endpoints rebuilt in the browser`, 1e-12);
  const covered = regression.testY.filter((value, index) =>
    rebuilt.lower[index] <= value && value <= rebuilt.upper[index]).length;
  assert.equal(covered, recorded.covered, `${key}: rebuilt coverage disagrees with the offline run`);
  assert.equal(covered, regression.methods[key].covered, 'and with the stored count');
  const widths = rebuilt.upper.map((value, index) => Math.max(value - rebuilt.lower[index], 0));
  close(widths.reduce((sum, value) => sum + value, 0) / widths.length, regression.methods[key].meanWidth,
    `${key} mean width`, 1e-12);
  /* The declared frequency slices, recounted from the served frequencies. */
  [['below_2000_hz', value => value < regression.frequencySplitHz],
    ['at_least_2000_hz', value => value >= regression.frequencySplitHz]].forEach(([label, test]) => {
    const chosen = regression.testFrequencyHz
      .map((value, index) => (test(value) ? index : -1)).filter(index => index >= 0);
    nonEmpty(chosen, undefined, `${key} ${label} slice`);
    const sliceCovered = chosen.filter(index =>
      rebuilt.lower[index] <= regression.testY[index] && regression.testY[index] <= rebuilt.upper[index]).length;
    assert.equal(sliceCovered, regression.methods[key].frequencyGroups[label].covered,
      `${key} ${label}: recounted coverage disagrees`);
    assert.equal(chosen.length, regression.methods[key].frequencyGroups[label].n, `${key} ${label}: size`);
    record('frequency slice');
  });
  record('measured interval method');
});
/* The tradeoff the table is placed beside must actually be present. */
assert(regression.methods.cqr.covered > regression.methods.raw_quantiles.covered,
  'CQR must improve on the unadjusted quantiles in this sample, or the sentence beside the table is false');
assert(regression.methods.cqr.meanWidth < regression.methods.ridge_absolute.meanWidth,
  'and must be narrower than ridge');
assert(regression.methods.cqr.covered < regression.methods.ridge_absolute.covered,
  'while covering fewer rows, which is the tradeoff the prose names');
assert(regression.pointMae < regression.constantMae, 'ridge must beat the constant baseline on point error');
assert.equal(regression.methods.ridge_absolute.frequencyGroups.below_2000_hz.covered, 54,
  'the low-frequency slice is 54 covered');
assert.equal(regression.methods.ridge_absolute.frequencyGroups.below_2000_hz.n, 66, 'of 66');
assert.equal(regression.methods.ridge_absolute.frequencyGroups.at_least_2000_hz.covered, 54, 'and 54');
assert.equal(regression.methods.ridge_absolute.frequencyGroups.at_least_2000_hz.n, 54, 'of 54');
/* Every interval procedure's widths are non-negative and finite. */
regression.methodOrder.forEach(key => {
  const row = regression.methods[key];
  row.widthQuantiles.forEach(value => assert(value >= 0 && Number.isFinite(value), `${key}: a width is invalid`));
  assert.equal(row.emptyCount, 0, `${key}: no interval should be empty in this run`);
  record('width sanity');
});
/* The assessment rows are never reachable from a fixture.
 *
 * This guard was inert as first written, and the guard audit is what found it.
 * It filtered candidates with `Math.abs(value) > 2 && !Number.isInteger(value)`
 * to avoid coincidental matches on small round numbers — but every number in
 * the fixtures is an integer or carries at most two decimals, so NO fixture
 * value can satisfy that filter. `leaked` was empty by construction, and the
 * whole held-out probability range (every probability is below 1) sat outside
 * the guard's domain: the page could have printed all 200 test probabilities
 * into the investigation fixtures and this assertion would still have passed.
 *
 * Two changes. The subject set is now every held-out array in the data module,
 * found by walking it for any path with a `test…` segment, so a new held-out
 * quantity is covered the day it is added rather than the day someone
 * remembers to list it. And the coincidence filter is precision, not
 * magnitude: a match at three or more decimals is a copied value, whatever its
 * size, which admits probabilities and dB readings alike.
 */
const fixtureNumbers = new Set(JSON.stringify(fixtures).match(/-?\d+(\.\d+)?/g)?.map(Number) ?? []);
/* `?? []` makes an empty set the failure mode of a regex that stops matching,
 * and an empty set leaks nothing — so the leak check below would pass on a
 * page that had put every held-out value into the fixtures. */
assert.ok(fixtureNumbers.size >= 30,
  `only ${fixtureNumbers.size} numbers were extracted from the fixtures; the leak check has no subject `
  + 'set and would report nothing however much had leaked');
const decimalsOf = value => {
  const text = String(value);
  const dot = text.indexOf('.');
  return dot < 0 ? 0 : text.length - dot - 1;
};
const heldOut = new Map();
const collectHeldOut = (node, path) => {
  if (Array.isArray(node)) { node.forEach(entry => collectHeldOut(entry, path)); return; }
  if (node && typeof node === 'object') {
    Object.entries(node).forEach(([key, value]) => collectHeldOut(value, path.concat(key)));
    return;
  }
  if (typeof node === 'number' && path.some(segment => /^test/i.test(segment))) {
    const key = path.join('.');
    if (!heldOut.has(key)) heldOut.set(key, []);
    heldOut.get(key).push(node);
  }
};
collectHeldOut(calibrationData, []);
assert.equal(heldOut.size, 13,
  `the walk found ${heldOut.size} held-out arrays rather than 13; the data module's shape has changed and the `
  + 'leak check is no longer reading all of them');
/* A constant predictor's output is not a held-out observation. The prior
 * baseline emits the TRAINING base rate 103/240 on all 80 test rows, and that
 * same training quantity is a fixture on purpose — `floatingBoundary`
 * positions its boundary at the prior. Excluding it as "this one value is
 * fine" would be an exemption carved to make a check pass; excluding every
 * constant array is the actual rule, because a series with one distinct value
 * reveals nothing about any individual row. The count is pinned so the
 * exclusion cannot quietly grow to cover a real leak. */
const constantArrays = [...heldOut.entries()].filter(([, values]) => new Set(values).size === 1
  && values.length > 1);
assert.deepEqual(constantArrays.map(([key]) => key), ['classification.methods.prior_constant.testProbabilities'],
  `held-out arrays that are constant: ${constantArrays.map(([key]) => key).join(', ')}`);
const informativePaths = [...heldOut.entries()].filter(([key]) =>
  !constantArrays.some(([constant]) => constant === key));
const assessmentNumbers = informativePaths.flatMap(([, values]) => values);
nonEmpty(assessmentNumbers, 843, 'assessment numbers');
const identifiable = assessmentNumbers.filter(value => decimalsOf(value) >= 3);
assert.ok(identifiable.length >= 350,
  `only ${identifiable.length} held-out values carry enough precision to be identified; the coincidence filter `
  + 'is excluding the subject set rather than the coincidences');
const leaked = identifiable.filter(value => fixtureNumbers.has(value));
assert.deepEqual(leaked, [],
  `a held-out assessment value appears among the investigation fixtures: ${leaked.join(', ')}`);
record('fixture isolation');

/* ==================================== §14 · the page's own text and assets */

assert(lessonBody.includes('/learn-assets/calibration/') === false,
  'asset paths should come from the generated provenance rather than being written into the body');
Object.values(calibrationData.provenance.files).forEach(entry => {
  assert(entry.file.startsWith('/learn-assets/calibration/'),
    `${entry.file} is not under this lesson's own asset directory`);
  assert(fs.existsSync(`public${entry.file}`), `${entry.file} is declared but not served`);
  const digest = crypto.createHash('sha256').update(fs.readFileSync(`public${entry.file}`)).digest('hex');
  assert.equal(digest, entry.sha256, `${entry.file} does not match its recorded digest`);
  assert.equal(fs.statSync(`public${entry.file}`).size, entry.bytes, `${entry.file} byte count`);
  record('served asset');
});
/* No sibling lesson's copy of the same data is referenced anywhere. */
['pac-learning', 'evaluation-metrics', 'semi-supervised-learning', 'automl-nas', 'bias-variance', 'regularization']
  .forEach(sibling => {
    assert(!lessonBody.includes(`/learn-assets/${sibling}/`),
      `the body references ${sibling}'s assets instead of this lesson's own copies`);
    assert(!JSON.stringify(calibrationData.provenance).includes(`/learn-assets/${sibling}/`),
      `the provenance references ${sibling}'s assets`);
    record('asset ownership');
  });

/* The plot-direction finding from the destination note, checked against the
   body rather than assumed to have been applied. */
assert(lessonBody.includes('<strong>below</strong> the diagonal'),
  'the body must say that an overestimated positive probability sits BELOW the diagonal');
assert(!/overconfiden\w* (?:bows |sits |lies )?above the diagonal/i.test(lessonBody),
  'the body still contains the reversed plot-direction claim the destination note reported');
assert(lessonBody.includes('First pass.'), 'the body must state a first-pass route after the introduction');
assert(lessonBody.includes('deeper branch'), 'and must label its deeper branch where it begins');
record('body claims');

/* The displayed programs are the ones the examples verifier executed. */
['reliability', 'monotone', 'rank', 'calculations', 'experiments'].forEach(key => {
  const example = calibrationExamples[key];
  assert(example, `the examples module is missing ${key}`);
  assert(example.executed === true, `${key} is not marked as executed`);
  assert(example.expected && example.expected.length > 20, `${key} has no recorded output`);
  assert(example.code && example.code.length > 200, `${key} has no code`);
  record('displayed program');
});
assert(calibrationExamples.rank.expected.includes('q = inf'),
  'the rank program must show the unbounded case, which is where clipping would hide');
assert(calibrationExamples.rank.expected.includes('the empty set'),
  'and an empty prediction set');
assert(calibrationExamples.reliability.expected.includes('ECE 0.000000'),
  'the reliability program must show an ECE of exactly zero after merging');
assert(calibrationExamples.monotone.expected.includes('0.333333'),
  'the monotone program must show the count-weighted pool');
calibrationExamples.rank.verbatimFunctions.forEach(name => {
  assert(calibrationExamples.rank.code.includes(`def ${name}(`),
    `the rank program lost the verbatim body of ${name}`);
  record('verbatim function');
});

/* Every investigation wires current inputs to its live calculation. These
   source checks complement browser interaction checks; they do not prove
   before-paint updates or control usability by themselves. Learner guesses
   and answer-unlock gates are forbidden, including optional ones. */
const investigationSources = labsSource.split(/export function /).slice(1);
nonEmpty(investigationSources, undefined, 'investigation sources');
const labs = investigationSources.filter(section => section.startsWith('ReliabilityLab')
  || section.startsWith('MonotoneLab') || section.startsWith('RankLab') || section.startsWith('IntervalLab'));
nonEmpty(labs, 4, 'the four investigations');
labs.forEach(section => {
  const name = section.slice(0, section.indexOf('('));
  assert(section.includes('useInvestigation('), `${name} does not use the live investigation state`);
  assert(section.includes('<LiveResult') && section.includes('calculateInputs='),
    `${name} does not connect its inputs to a live calculation`);
  assert(section.includes('cal-graded'), `${name} has no calculated readout`);
  assert(!section.includes('<Prediction') && !section.includes('answerFor'),
    `${name} restores a learner-prediction feature or answer grader`);
  record('live investigation contract');
});
assert.equal((labsSource.match(/<LiveResult\b/g) ?? []).length, 4,
  'each of the four investigations needs its own live result');
const sharedSource = fs.readFileSync('src/learn/components/lesson-labs/CalibrationShared.jsx', 'utf8');
assert(sharedSource.includes('export const useInvestigation = useLiveInvestigation;'),
  'the investigation state must delegate to the current live state hook');
assert(sharedSource.includes('useLiveResult(state, calculateInputs, blocked)'),
  'the shared result must evaluate current inputs through the live result hook');
assert(!/export\s+function\s+Prediction\b/.test(sharedSource),
  'the shared controls must not reintroduce optional learner-prediction entry');
record('shared live calculation wiring');

/* A preset that promises a relationship must build it from the APPLIED state.
 *
 * The comparison uses the current applied state, so a preset built on an
 * unapplied draft promises a relationship it does not compare: historically,
 * "double every scale — an exact null" doubled a draft that already differed
 * from the applied state, and the calculation correctly reported a change while
 * the button had promised none. The browser run found this in investigation 4;
 * the same shape was present in all four. */
/* Enumerate every CONSTRUCTION SITE, not every block of a particular shape.
 *
 * The first version of this guard scanned `const presets = [...]` bodies. It
 * found four blocks, all clean, and passed — while two inline `state.suggest`
 * buttons in investigation 3's rotation branch still built from the draft. The
 * regex was alive and its subject set was non-empty, and it still could not
 * fire on the only instance that existed, because a preset does not have to
 * live in an array called `presets`.
 *
 * So the subject set is now every call to `state.suggest`, found wherever it
 * is, and the count is pinned: a new call site that this guard has not been
 * taught about fails the run rather than slipping past it. */
const suggestSites = [...labsSource.matchAll(/state\.suggest\(/g)].map(match => {
  const start = match.index;
  /* Take the argument expression by balancing parentheses from the call, so an
     inline object literal is captured as fully as an identifier is. */
  let depth = 0;
  let end = start;
  for (let index = start + 'state.suggest'.length; index < labsSource.length; index += 1) {
    const character = labsSource[index];
    if (character === '(') depth += 1;
    else if (character === ')') {
      depth -= 1;
      if (depth === 0) { end = index; break; }
    }
  }
  return {
    line: labsSource.slice(0, start).split('\n').length,
    argument: labsSource.slice(start, end + 1),
  };
});
nonEmpty(suggestSites, 6, 'state.suggest call sites');
suggestSites.forEach(site => {
  assert(!/\.\.\.draft\b/.test(site.argument),
    `the suggested setup at CalibrationLabs.jsx:${site.line} is built from the draft rather than from the applied `
    + 'state, so the change its label names is not the change that gets compared');
  assert(!/\bdraft\.\w+/.test(site.argument),
    `the suggested setup at CalibrationLabs.jsx:${site.line} reads a draft collection rather than the applied one`);
  record('suggest site');
});
/* The guard must have seen the inline sites specifically, not only the arrays:
   two of the six take an object literal rather than a named `setup`. */
const inlineSites = suggestSites.filter(site => site.argument.includes('{'));
nonEmpty(inlineSites, 2, 'inline suggest sites');
/* And the array form is still checked as a whole, because a preset entry can
   read the draft outside the `state.suggest(...)` call itself: the arrays are
   built first and passed by name.
 *
 * Bounded by BALANCING BRACKETS, not by a lazy match to the next line that
 * happens to look like a closing bracket. The lazy version silently swallowed
 * 13,781 characters of unrelated component code into one "block" and then
 * failed on a `draft.pairs.map` that was never in a preset at all — a guard
 * reporting a defect in a region it had no business reading. */
function balancedArrayAt(source, open) {
  let depth = 0;
  for (let index = open; index < source.length; index += 1) {
    if (source[index] === '[') depth += 1;
    else if (source[index] === ']') {
      depth -= 1;
      if (depth === 0) return source.slice(open + 1, index);
    }
  }
  return null;
}
/* Anchored to the start of a line. Searched as a bare substring, the phrase
   matched this very comment's own description of the marker, and the scan then
   reported five preset arrays where the file has four. A guard that reads its
   own documentation as source is not reading source. */
const presetBlocks = [...labsSource.matchAll(/^[ \t]*const presets = \[/gm)]
  .map(match => balancedArrayAt(labsSource, labsSource.indexOf('[', match.index)))
  .filter(Boolean);
nonEmpty(presetBlocks, 4, 'preset blocks');
presetBlocks.forEach((block, index) => {
  /* Each block must be a plausible size: a lazy bound that ran away would show
     up here rather than as a mystery failure somewhere downstream. */
  assert(block.length < 4000,
    `preset block ${index + 1} spans ${block.length} characters, which is more than a preset array should be — `
    + 'the bounds are wrong and this check is reading code it does not own');
  assert(!/\.\.\.draft\b/.test(block),
    `the preset block in investigation ${index + 1} builds a setup from the draft rather than from the applied `
    + 'state, so a label promising a null is not the change that gets compared');
  assert(!/\bdraft\.\w+\.map\(/.test(block),
    `the preset block in investigation ${index + 1} maps over a draft collection rather than the applied one`);
  record('preset baseline');
});
/* And every "exact null" label must be attached to a setup that really is one
   against the state it is built from. Checked here on the opening state, and
   exercised in the browser after a control change. */
const nullLabels = [...labsSource.matchAll(/'([^']*an exact null)'/g)].map(match => match[1]);
nonEmpty(nullLabels, undefined, 'exact-null preset labels');
assert(nullLabels.length >= 4, `only ${nullLabels.length} presets promise an exact null`);
record('null labels');
/* No investigation reads the measured experiments. */
assert(!labsSource.includes('calibration-data.js'),
  'an investigation imports the measured experiment data, so a held-out number is reachable from a lab');
record('lab isolation');

/* --------------------------------------------------------------- record */

const sources = [
  'src/learn/data/calibration-models.js',
  'src/learn/data/calibration-data.js',
  'src/learn/data/calibration-examples.js',
  'src/learn/components/lesson-labs/CalibrationShared.jsx',
  'src/learn/components/lesson-labs/CalibrationLabs.jsx',
  'src/learn/components/lesson-labs/CalibrationFigures.jsx',
  'src/learn/components/lesson-labs/calibration-labs.css',
  'src/learn/data/topics/calibration-conformal-prediction.jsx',
  'src/learn/data/curriculum/blueprints/calibration-conformal-prediction.js',
  'public/learn-assets/calibration/banknote-subset.csv',
  'public/learn-assets/calibration/airfoil-subset.csv',
  'public/learn-assets/calibration/calibration_calculations.py',
  'public/learn-assets/calibration/uncertainty_experiments.py',
  'public/learn-assets/calibration/reliability_bins.py',
  'public/learn-assets/calibration/monotone_map.py',
  'public/learn-assets/calibration/conformal_rank.py',
  'public/learn-assets/calibration/ATTRIBUTION.txt',
  'scripts/verify-calibration-sources.py',
  'scripts/verify-calibration-examples.py',
  'scripts/verify-calibration-data.py',
  'scripts/verify-calibration-browser.cjs',
  'scripts/falsify-calibration.mjs',
];
// Unfiltered on purpose: silently dropping a declared source that no longer
// exists and still writing `passed: true` is how a deleted file passes.
/* Floored, because `[].filter(...)` is also empty: a `sources` list that lost
 * its entries would report no missing file and pass. */
assert(sources.length >= 22, `the declared source list holds only ${sources.length} files`);
const missingSources = sources.filter(file => !fs.existsSync(file));
assert.deepEqual(missingSources, [], `declared source files are missing: ${missingSources.join(', ')}`);

const hash = file => crypto.createHash('sha256').update(fs.readFileSync(file)).digest('hex');
const total = Object.values(counts).reduce((sum, value) => sum + value, 0);
/* A counter that is reported but never floored is decoration: deleting every
 * `record()` call still printed PASS with a smaller number. These floors sit
 * below the current values but far above zero, so a wholesale loss of coverage
 * fails the run instead of quietly shrinking the headline. */
assert(total >= 450, `only ${total} grouped checks ran; the suite has lost coverage`);
assert(Object.keys(counts).length >= 45, `only ${Object.keys(counts).length} groups ran`);

const evidence = {
  checkedAt: new Date().toISOString(),
  sourceHashes: Object.fromEntries(sources.map(file => [file, hash(file)])),
  packetCheckedResultsSha256: hash(`${packetDirectory}/checked-results.json`),
  packetExperimentResultsSha256: hash(`${packetDirectory}/experiment-results.json`),
  verifierHash: hash('scripts/verify-calibration-models.mjs'),
  counts,
  totalGroupedChecks: total,
  rankCasesSwept: rankCases,
  floatingPointRankDisagreements: naiveDisagreements,
  floatingPointRankDisagreementsReachableByAControl: reachableDisagreements.length,
  floatingPointRankAlwaysExactlyOneTooLarge: true,
  floatingPointRankEverExceedsN: false,
  floatingPointRankExamples: naiveExamples,
  thresholdCasesSwept: thresholdCases,
  rotationCasesSwept: rotationCases,
  binningCasesSwept: binCases,
  emptyBinsExercised: emptyBinsSeen,
  aucCasesSwept: aucCases,
  monotoneFitsSwept: pavCases,
  mergesTraced: mergeTraces,
  sigmoidFitsSwept: sigmoidCases,
  sigmoidGridPoints: gridPoints,
  temperatureCasesSwept: temperatureCases,
  cqrScoreCasesSwept: cqrCases,
  prevalenceCasesSwept: shiftCases,
  verdictPairsSwept: verdictCases,
  verdictUnchanged,
  verdictMoved,
  geometryChecks: geometryCases,
  measuredSetMembershipsChecked: measuredCases,
  unchangedTolerance,
  gradedDisplayDigits,
  scope: 'Browser calibration models against the content packet\'s checked-results.json and experiment-results.json '
    + 'and every number the lesson states. The conformal rank is recomputed by scanning integers under exact '
    + 'rational arithmetic — never a ceiling, never a quantile, never the module\'s own function — over every one '
    + 'of 98,200 (n, alpha) pairs, and the 26 points at which ordinary floating point disagrees are recorded. '
    + 'Exactly one of those, n=24 at alpha=.44, lies on the grid the controls actually admit, so the guard is '
    + 'shown to bite on a reachable input rather than only in principle; at every one of the 26 the float rank '
    + 'is exactly one too large and never exceeds n, so the hazard is a silently over-wide threshold and not a '
    + 'missing order statistic. Thresholds are '
    + 'recomputed by counting rather than indexing, and checked against their defining property directly. '
    + 'Rotations are recomputed by the combined-rank argument rather than by rebuilding thresholds. The monotone '
    + 'fit is recomputed by the max-min formula over lower and upper sets, over every labelling of seven score '
    + 'patterns, and the investigation\'s leftmost-violation schedule is shown to reach the same fit as the stack '
    + 'algorithm. Bins and ECE are recomputed by explicit comparison chains, with empty bins exercised as the '
    + 'no-observed-fraction case they are, the one-bin identity shown to give exactly zero on every labelling, '
    + 'and the row-permutation null asserted rather than described. AUC is recomputed by counting concordant '
    + 'pairs, including the one-class case that has no value. The sigmoid solver is checked against SciPy\'s BFGS '
    + 'fit, against its own stationarity condition and against a 6,561-point grid. Temperature scaling is swept '
    + 'over 400 temperatures on five logit sets with the winner, every tie and every pairwise order asserted '
    + 'unchanged. The CQR score is swept over 4,000 endpoint-and-response combinations with set membership shown '
    + 'equivalent to the score comparison at every one. Every grading rule is exercised at zero, at exact ties, '
    + 'at identical inputs and where the quantity has no value, and the verdict is shown to say "unchanged" '
    + 'exactly when the page\'s own display function prints the same string, over more than forty thousand pairs '
    + 'of values these labs can actually produce. Every drawn coordinate is asserted against the quantity it '
    + 'encodes, including that a count bar for an empty bin has no length, that a zero-width interval is drawn '
    + 'with zero length, that a dot above the diagonal in the data is above it on screen, and that an unbounded '
    + 'threshold is not drawn as a point. Finally the generated data module\'s set memberships, coverage counts, '
    + 'mean set sizes, interval endpoints and frequency slices are all rebuilt from the stored scalars and '
    + 'checked against the offline run.',
  limitations: [
    'The measured experiment is checked as a recorded result: the browser reproduces its set memberships, '
      + 'coverage counts and interval endpoints from stored predictions and thresholds, not the fitting. The '
      + 'refitting from the served CSV files is checked separately by scripts/verify-calibration-data.py.',
    'Displayed program output is executed separately by scripts/verify-calibration-examples.py, which also '
      + 'checks that both offline programs regenerate the packet\'s JSON byte for byte.',
    'The verdict-and-display equivalence is asserted over the values these labs can actually produce and over '
      + 'deliberate offsets around each of them, not over all reals. A relative tolerance and a fixed-decimal '
      + 'display necessarily disagree in a narrow band — at a width of 60 the unchanged band is 6e-11 wide while '
      + 'the printed value resolves 1e-12 — and no reachable pair enters that band. A graded quantity whose '
      + 'magnitude grew by orders of magnitude would need the display precision revisited.',
    'The live interaction contract is checked statically here for all four investigations, their shared '
      + 'calculation wiring, and absence of learner-prediction components. Immediate visible updates, keyboard '
      + 'operation and responsive control styling require separate production browser measurements.',
    'Rendering, painted styles, visual layout and independent review are separate steps and are not claimed here.',
  ],
  passed: true,
};
if (!process.argv.includes('--no-evidence')) {
  fs.mkdirSync('docs/teaching/evidence', { recursive: true });
  fs.writeFileSync('docs/teaching/evidence/calibration-models.json', JSON.stringify(evidence, null, 2) + '\n');
}
console.log(`PASS: ${total} grouped calibration model checks across ${Object.keys(counts).length} groups, `
  + `including ${rankCases.toLocaleString('en-US')} rank cases recomputed by exact integer scanning `
  + `(${naiveDisagreements} of which ordinary floating point gets wrong, ${reachableDisagreements.length} `
  + 'reachable through a control), '
  + `${pavCases.toLocaleString('en-US')} monotone fits against the max-min formula, `
  + `${cqrCases.toLocaleString('en-US')} CQR score cases, `
  + `${verdictCases.toLocaleString('en-US')} verdict-and-display pairs and `
  + `${geometryCases.toLocaleString('en-US')} geometry checks.`);
