// Bounded independent checks of the anomaly-detection browser models against
// the content phase's native probes, the manuscript's exact fixtures and
// hand-derived identities, plus structural checks of the generated real-data
// module. Run: node scripts/verify-anomaly-detection-models.mjs
import assert from 'node:assert/strict';
import fs from 'node:fs';
import crypto from 'node:crypto';
import {
  exact, correction, firstCutIntervals, isolationExpectations, isolationPath,
  lofState, lofModeComparison, kernelBoundary, alarmCounts, quantileIndex, windowHits,
  rowTimestamp, formatTimestamp,
} from '../src/learn/data/anomaly-detection-models.js';
import {
  seriesStart, stepMinutes, rowCounts, sourceFacts, eventWindows, referenceFit,
  methodOrder, methodLabels, overviewBins, quantileOutcomes, sweepOutcomes,
  publishedOutcomes, windowDetail,
} from '../src/learn/data/anomaly-temperature-data.js';

const draft = 'docs/teaching/drafts/anomaly-outlier-detection-isolation-forest-one-class-svm-lof';
const author = JSON.parse(fs.readFileSync(`${draft}/author-calculations.json`, 'utf8'));
const examples = JSON.parse(fs.readFileSync('src/learn/data/anomaly-detection-examples.js', 'utf8')
  .replace(/^[\s\S]*?export const anomalyExamples = /, '').replace(/;\s*$/, ''));

const counts = {};
const record = name => { counts[name] = (counts[name] ?? 0) + 1; };
const close = (actual, expected, label, tolerance = 1e-9) =>
  assert(Math.abs(actual - expected) <= tolerance * Math.max(1, Math.abs(expected)), `${label}: ${actual} versus ${expected}`);

// ---------------------------------------------------------------- corrections
assert.equal(correction(0).toString(), '0');
assert.equal(correction(1).toString(), '0');
assert.equal(correction(2).toString(), '1');
assert.equal(correction(4).toString(), '13/6');
assert.equal(correction(5).toString(), '77/30');
// c(m) = 2H(m-1) - 2(m-1)/m, rebuilt independently with exact rationals.
for (let size = 2; size <= 40; size += 1) {
  let harmonic = exact(0);
  for (let term = 1; term < size; term += 1) harmonic = harmonic.add(exact(1, term));
  const expected = harmonic.multiply(exact(2)).subtract(exact(2 * (size - 1), size));
  assert(correction(size).equals(expected), `c(${size})`);
}
assert.throws(() => correction(-1), RangeError);
record('leaf corrections');

// ------------------------------------------------------------- isolation cuts
const gaps = firstCutIntervals([0, 1, 2, 3, 12]);
assert.equal(gaps.intervals.length, 4);
assert.equal(gaps.intervals.at(-1).probability.toString(), '3/4');
assert.deepEqual(gaps.intervals.at(-1).right, [4]);
assert.equal(gaps.intervals[0].probability.toString(), '1/12');
assert.deepEqual(gaps.intervals[0].left, [0]);
// Every gap probability is a length ratio, and they sum to exactly one.
assert(gaps.intervals.reduce((sum, gap) => sum.add(gap.probability), exact(0)).equals(exact(1)), 'gap probabilities sum to 1');
// Practice A: 0, 2, 3, 4, 10.
const practiceA = firstCutIntervals([0, 2, 3, 4, 10]);
assert.equal(practiceA.intervals.at(-1).probability.toString(), '3/5');
assert.equal(practiceA.intervals[0].probability.toString(), '1/5');
assert.equal(firstCutIntervals([2, 2, 2, 2, 2]).constant, true);
assert.equal(firstCutIntervals([2, 2, 2, 2, 2]).intervals.length, 0);
record('first-cut intervals');

// Expected corrected paths against the author's exact integration.
for (const probe of author.isolation) {
  const fit = isolationExpectations(probe.values, probe.maxDepth);
  assert.equal(fit.normalizer.toString(), probe.normalizer, 'normalizer');
  assert.deepEqual(fit.rows.map(row => row.meanPath.toString()), probe.meanLengths, `mean paths for ${probe.values}`);
  fit.rows.forEach((row, index) => close(row.score, probe.scores[index], `score ${index} for ${probe.values}`, 1e-12));
  record('isolation expectations versus author');
}
const spread = isolationExpectations([0, 1, 2, 3, 12], 3);
assert.equal(spread.shortest, 4, '12 has the shortest expected path');
assert.equal(spread.tied, false);
const flat = isolationExpectations([0, 0, 0, 0, 0], 3);
assert.equal(flat.constant, true);
assert(flat.rows.every(row => row.score === 0.5), 'constant data score exactly one half');
assert.equal(isolationExpectations([0, 1, 2, 3, 4], 3).tied, true, 'symmetric endpoints tie');
// A path equal to the normalizer gives the exponent minus one.
close(2 ** -(correction(5).toNumber() / correction(5).toNumber()), 0.5, 'exponent minus one');
record('isolation identities');

// Practice B: depth 2 into a two-row leaf of a four-row tree.
const truncated = isolationPath([0, 1, 2, 10], [5, 1.5], 0, 3);
assert.equal(truncated.leaf.size, 2);
assert.equal(truncated.leaf.depth, 2);
assert.equal(truncated.pathLength.toString(), '3');
assert.equal(truncated.normalizer.toString(), '13/6');
close(truncated.score, 2 ** (-18 / 13), 'practice B score', 1e-15);
close(truncated.score, 0.3829915893, 'practice B rounded', 1e-9);
// Normalising a four-row tree by c(256) inflates the score, as the section says.
assert(2 ** -(3 / correction(256).toNumber()) > truncated.score, 'c(256) inflates the score');
// A cut outside the node's span is reported rather than silently applied.
assert.equal(isolationPath([0, 1, 2, 10], [20], 0, 3).steps[0].usable, false);
record('truncated path');

// ------------------------------------------------------------------ LOF state
for (const probe of author.lof) {
  const state = lofState(probe.values, probe.k);
  assert.deepEqual(state.rows.map(row => row.neighbours.map(entry => entry.id)), probe.state.neighbors, `neighbours at k=${probe.k}`);
  state.rows.forEach((row, index) => {
    close(row.radius.toNumber(), probe.state.radii[index], `radius ${index} at k=${probe.k}`);
    assert.deepEqual(row.neighbours.map(entry => entry.reach.toNumber()), probe.state.reach[index], `reach ${index} at k=${probe.k}`);
    close(row.density.toNumber(), probe.state.lrd[index], `lrd ${index} at k=${probe.k}`);
    close(row.factor.toNumber(), probe.state.lof[index], `factor ${index} at k=${probe.k}`);
  });
  probe.queries.forEach((query, index) => {
    const scored = lofState(probe.values, probe.k, query);
    close(scored.query.factor.toNumber(), probe.queryLOF[index], `query ${query} at k=${probe.k}`);
    close(scored.query.factor.toNumber(), probe.libraryQueryLOF[index], `query ${query} versus library at k=${probe.k}`, 1e-9);
  });
  record('LOF state versus author');
}
const six = lofState([0, 1, 2, 20, 24, 28], 2);
assert.deepEqual(six.rows.map(row => row.radius.toString()), ['2', '1', '2', '8', '4', '8']);
assert.deepEqual(six.rows.map(row => row.density.toString()), ['2/3', '1/2', '2/3', '1/6', '1/8', '1/6']);
assert.deepEqual(six.rows.map(row => row.factor.toString()), ['7/8', '4/3', '7/8', '7/8', '4/3', '7/8']);
assert.equal(lofState([0, 1, 2, 20, 24, 28], 2, 4).query.factor.toString(), '35/24');
assert.equal(lofState([0, 1, 2, 20, 24, 28], 2, 17).query.factor.toString(), '35/32');
assert.equal(lofState([0, 1, 2, 20, 24, 28], 2, 6).query.factor.toString(), '21/8');
// The farther query takes the lower factor.
const near = lofState([0, 1, 2, 20, 24, 28], 2, 4).query;
const far = lofState([0, 1, 2, 20, 24, 28], 2, 17).query;
assert(far.neighbours[0].distance.toNumber() > near.neighbours[0].distance.toNumber(), '17 is farther from its nearest reference');
assert(far.factor.toNumber() < near.factor.toNumber(), 'and yet takes the lower factor');
// The floor belongs to the neighbour, so reachability is not symmetric here.
const fromZero = six.rows[0].neighbours.find(entry => entry.id === 1);
const fromOne = six.rows[1].neighbours.find(entry => entry.id === 0);
assert.equal(fromZero.reach.toString(), '1');
assert.equal(fromOne.reach.toString(), '2');
assert(!fromZero.reach.equals(fromOne.reach), 'reach(0,1) differs from reach(1,0)');
// Rescaling every coordinate by a positive factor leaves every factor unchanged.
const plain = lofState([0, 2, 4, 5, 8], 2);
const doubled = lofState([0, 2, 4, 5, 8].map(value => value * 2), 2);
assert.deepEqual(plain.rows.map(row => row.factor.toString()), doubled.rows.map(row => row.factor.toString()));
assert.throws(() => lofState([0, 1, 1, 4], 2), RangeError);
assert.throws(() => lofState([0, 1, 2, 3], 4), RangeError);
record('LOF identities');

// Fitting mode: the training row excludes its own identity, the query does not.
const mode2 = lofModeComparison([0, 1, 2, 20, 24, 28], 2, 1);
assert.equal(mode2.trainingRow.factor.toString(), '4/3');
assert.equal(mode2.queryRow.factor.toString(), '7/8');
assert.equal(mode2.agree, false);
const mode24 = lofModeComparison([0, 1, 2, 20, 24, 28], 2, 24);
assert.equal(mode24.trainingRow.factor.toString(), '4/3');
assert.equal(mode24.queryRow.factor.toString(), '7/8');
const mode3 = lofModeComparison([0, 1, 2, 20, 24, 28], 3, 1);
close(mode3.trainingRow.factor.toNumber(), 0.9545454545454546, 'k=3 training factor');
assert.equal(mode3.queryRow.factor.toString(), '1');
assert.equal(mode3.agree, false);
record('fitting mode');

// ----------------------------------------------------------- kernel boundary
for (const probe of author.kernel) {
  const fit = kernelBoundary(1, probe.gamma, 0);
  close(fit.rho, probe.rho, `rho at gamma ${probe.gamma}`);
  close(fit.atMidpoint, probe.atCenter, `midpoint at gamma ${probe.gamma}`);
  close(fit.atAnchor, probe.atReference, `anchor at gamma ${probe.gamma}`, 1e-12);
  close(kernelBoundary(1, probe.gamma, 3).decision, probe.atFarQuery, `far query at gamma ${probe.gamma}`);
  record('kernel versus author');
}
close(kernelBoundary(1, 0.1, 0).decision, 0.069677395, 'gamma 0.1 midpoint', 1e-8);
close(kernelBoundary(1, 1, 0).decision, -0.141278378, 'gamma 1 midpoint', 1e-8);
assert.equal(kernelBoundary(1, 1, 1).classification, 'boundary', 'an anchor sits on the boundary at every gamma');
assert.equal(kernelBoundary(1, 0.1, 1).classification, 'boundary');
assert.equal(kernelBoundary(1, 0.1, 0).classification, 'inside');
assert.equal(kernelBoundary(1, 1, 0).classification, 'outside');
// At gamma 1 the accepted region is two separated pieces; at gamma 0.1 it is one.
assert.equal(kernelBoundary(1, 1, 0).positiveIntervals.length, 2, 'a disconnected accepted region');
assert.equal(kernelBoundary(1, 0.1, 0).positiveIntervals.length, 1, 'one connected piece at a small gamma');
const pieces = kernelBoundary(1, 1, 0).positiveIntervals;
assert(pieces[0][1] < 0 && pieces[1][0] > 0, 'the midpoint falls between the two pieces');
assert(pieces.every(([low, high]) => low < high), 'each piece is a real interval');
// Practice E: doubling the anchors and quartering gamma preserves every exponent.
close(kernelBoundary(2, 0.25, 0).decision, kernelBoundary(1, 1, 0).decision, 'practice E agreement', 1e-12);
close(kernelBoundary(2, 0.25, 0).rho, (1 + Math.exp(-4)) / 2, 'practice E rho', 1e-12);
assert.throws(() => kernelBoundary(1, 0, 0), RangeError);
record('kernel boundary');

// ------------------------------------------------------------ alert workload
for (const probe of author.alarm) {
  const fit = alarmCounts(100000, probe.prevalence, probe.sensitivity, probe.falsePositiveRate, 200);
  close(fit.trueAlerts, probe.trueAlerts, 'true alerts');
  close(fit.falseAlerts, probe.falseAlerts, 'false alerts');
  close(fit.precision, probe.precision, 'precision');
  record('alert population versus author');
}
const section8 = alarmCounts(100000, 0.001, 0.8, 0.01, 200);
close(section8.trueAlerts, 80, 'section 8 true alerts');
close(section8.falseAlerts, 999, 'section 8 false alerts');
close(section8.total, 1079, 'section 8 total');
close(section8.precision, 80 / 1079, 'section 8 precision');
// Practice F.
const practiceF = alarmCounts(50000, 0.002, 0.9, 0.005, 200);
close(practiceF.trueAlerts, 90, 'practice F true alerts');
close(practiceF.falseAlerts, 249.5, 'practice F false alerts');
close(practiceF.total, 339.5, 'practice F total');
close(practiceF.precision, 90 / 339.5, 'practice F precision');
close(100 * practiceF.precision, 26.5096, 'practice F percentage', 1e-5);
assert.equal(practiceF.withinBudget, false);
close(practiceF.budgetShortfall, 139.5, 'practice F shortfall');
// Nobody flagged leaves precision undefined rather than zero or one.
assert.equal(alarmCounts(100000, 0.001, 0, 0, 200).precision, null);
// Holding both conditional rates fixed, precision still moves with prevalence.
assert(alarmCounts(100000, 0.01, 0.8, 0.01, 200).precision > section8.precision, 'precision rises with prevalence');
assert.throws(() => alarmCounts(100000, 1.5, 0.8, 0.01, 200), RangeError);
record('alert workload');

// --------------------------------------------------- threshold and tie ruler
// Sorted calibration scores 1, 1, 2, 4, 4 with a strict threshold.
const ruler = [1, 1, 2, 4, 4];
assert.equal(ruler.filter(score => score > 4).length, 0, 'threshold 4 flags nothing');
assert.equal(ruler.filter(score => score > 2).length, 2, 'threshold 2 flags both fours');
assert.equal(ruler.filter(score => score > 3).length, 2, 'no threshold between them flags one');
const reachableCounts = new Set([0, 1, 2, 3, 4, 5].map(bound => ruler.filter(score => score > bound - 0.5).length));
assert.equal(reachableCounts.has(1), false, 'the top fifth of these five rows is unreachable');
// NumPy's higher-interpolation quantile index.
assert.equal(quantileIndex(0.95, 1152), Math.ceil(0.95 * 1151));
assert.equal(quantileIndex(0.99, 1152), Math.ceil(0.99 * 1151));
assert.equal(quantileIndex(1, 1152), 1151);
assert.equal(quantileIndex(0, 1152), 0);
assert.deepEqual(windowHits(15), [true, true, true, true]);
assert.deepEqual(windowHits(5), [true, false, true, false]);
assert.deepEqual(windowHits(0), [false, false, false, false]);
record('threshold arithmetic');

// -------------------------------------------------- the generated real series
assert.equal(rowCounts.reference + rowCounts.calibration + rowCounts.test, rowCounts.total);
assert.equal(rowCounts.insideWindows + rowCounts.outsideWindows, rowCounts.test);
assert.equal(rowCounts.reference, author.realData.fitRows);
assert.equal(rowCounts.calibration, author.realData.calibrationRows);
assert.equal(rowCounts.test, author.realData.testRows);
assert.equal(rowCounts.insideWindows, author.realData.testRowsInsideWindows);
assert.equal(rowCounts.outsideWindows, author.realData.testRowsOutsideWindows);
assert.equal(sourceFacts.rawRows, author.realData.rawRows);
assert.equal(sourceFacts.uniqueTimestamps, author.realData.uniqueTimes);
assert.equal(sourceFacts.duplicateExcess, author.realData.averagedDuplicateExcess);
assert.equal(sourceFacts.droppedMissingLag, author.realData.removedMissingLagRows);
assert.equal(sourceFacts.rawRows - sourceFacts.uniqueTimestamps, sourceFacts.duplicateExcess);
assert.equal(sourceFacts.uniqueTimestamps - sourceFacts.droppedMissingLag, rowCounts.total);
close(referenceFit.median, author.realData.baselineMedian, 'baseline median');
close(referenceFit.mad, author.realData.baselineMAD, 'baseline MAD');
assert.deepEqual(eventWindows, author.realData.windows);
assert.equal(methodOrder.length, 4);
assert(methodOrder.every(key => typeof methodLabels[key] === 'string'));
record('series counts');

// The published table, method by method and quantile by quantile.
const authorKey = {
  'median absolute level deviation': 'baseline',
  'Isolation Forest': 'isolation',
  'One-Class SVM': 'oneClassSvm',
  'LOF novelty': 'lof',
};
for (const row of author.realData.results) {
  const method = authorKey[row.method];
  assert(method, `unknown author method ${row.method}`);
  const published = publishedOutcomes[method][row.calibrationQuantile.toFixed(2)];
  assert(published, `published row for ${row.method} at ${row.calibrationQuantile}`);
  close(published[0], row.threshold, `threshold ${row.method} ${row.calibrationQuantile}`, 1e-12);
  assert.equal(published[1], row.calibrationAlerts, 'calibration alerts');
  assert.equal(published[2], row.testAlerts, 'test alerts');
  assert.equal(published[3], row.alertsInsideWindows, 'alerts inside windows');
  assert.equal(published[4], row.alertsOutsideWindows, 'alerts outside windows');
  assert.equal(windowHits(published[5]).filter(Boolean).length, 4, 'all four windows carry an alert');
  record('published outcome versus author');
}
for (const method of methodOrder) {
  for (const [quantile, outcome] of Object.entries(publishedOutcomes[method])) {
    assert.equal(outcome[3] + outcome[4], outcome[2], `${method} ${quantile}: inside plus outside is the alert total`);
    assert.equal(outcome[1], rowCounts.calibration - 1 - quantileIndex(Number(quantile), rowCounts.calibration),
      `${method} ${quantile}: calibration alerts follow the higher-quantile index`);
    const indexed = quantileOutcomes[method].find(entry => entry[0] === quantileIndex(Number(quantile), rowCounts.calibration));
    assert.deepEqual(indexed.slice(1), outcome, `${method} ${quantile}: the quantile table agrees with the published row`);
    record('published outcome consistency');
  }
}

// The sweep is a monotone ladder: a higher threshold can never alert more rows.
for (const method of methodOrder) {
  const sweep = sweepOutcomes[method];
  for (let index = 1; index < sweep.length; index += 1) {
    assert(sweep[index][0] > sweep[index - 1][0], `${method}: sweep thresholds ascend`);
    assert(sweep[index][2] <= sweep[index - 1][2], `${method}: alert counts fall as the threshold rises`);
    assert(sweep[index][3] <= sweep[index - 1][3], `${method}: inside-window alerts fall too`);
  }
  assert.equal(sweep[0][2], rowCounts.test, `${method}: the lowest level flags every later row`);
  assert.equal(sweep.at(-1)[2], 0, `${method}: the highest level flags none`);
  assert(sweep.every(row => row[3] + row[4] === row[2]), `${method}: every sweep row splits into inside and outside`);
  const quantiles = quantileOutcomes[method];
  assert(quantiles.every(row => row[4] + row[5] === row[3]), `${method}: every quantile row splits`);
  assert(quantiles.every((row, index) => index === 0 || row[0] > quantiles[index - 1][0]), `${method}: quantile indices ascend`);
  record('threshold ladders');
}

// Overview bins tile the whole series once, keeping the level extremes.
assert.equal(overviewBins[0][0], 0);
assert.equal(overviewBins.reduce((sum, bin) => sum + bin[1], 0), rowCounts.total, 'bins tile every row');
overviewBins.forEach((bin, index) => {
  if (index > 0) assert.equal(bin[0], overviewBins[index - 1][0] + overviewBins[index - 1][1], `bin ${index} follows the previous one`);
  assert(bin[2] <= bin[3], `bin ${index} keeps a low and a high level`);
  assert([0, 1, 2].includes(bin[4]), `bin ${index} names a period`);
});
// Scores begin where the reference period ends: the rows used to fit are never
// scored, and every later bin carries a maximum for all four methods.
const firstScoredRow = rowCounts.reference;
overviewBins.forEach((bin, index) => {
  const scores = bin.slice(5);
  if (bin[0] + bin[1] <= firstScoredRow) {
    assert(scores.every(value => value === null), `bin ${index} inside the reference period carries no score`);
  }
  if (bin[0] >= firstScoredRow) {
    assert(scores.every(value => value !== null), `bin ${index} after the reference period carries every method's score`);
    assert.equal(scores.length, methodOrder.length, `bin ${index} carries one score column per method`);
  }
});
record('overview bins');

// Window detail rows: exact alert marking through the rank of each threshold,
// checked against the published inside-window counts and window hits.
assert.equal(windowDetail.length, 4);
const available = Object.fromEntries(methodOrder.map(method => [method,
  [...quantileOutcomes[method].map(row => row[1]), ...sweepOutcomes[method].map(row => row[0])]
    .sort((left, right) => left - right)]));
for (const method of methodOrder) {
  assert.equal(new Set(available[method]).size, available[method].length, `${method}: offerable thresholds are distinct`);
  for (const [quantile, outcome] of Object.entries(publishedOutcomes[method])) {
    const rank = available[method].indexOf(outcome[0]);
    assert(rank >= 0, `${method} ${quantile}: the published threshold is offerable`);
    let inside = 0;
    const hits = windowDetail.map(detail => {
      let hit = false;
      for (let index = detail.insideFrom; index <= detail.insideTo; index += 1) {
        if (detail.exceeds[method][index] > rank) { inside += 1; hit = true; }
      }
      return hit;
    });
    assert.equal(inside, outcome[3], `${method} ${quantile}: detail rows reproduce the inside-window alert count`);
    assert.deepEqual(hits, windowHits(outcome[5]), `${method} ${quantile}: detail rows reproduce the window hits`);
    record('detail rows versus published counts');
  }
}
windowDetail.forEach((detail, index) => {
  const length = detail.level.length;
  assert.equal(detail.change.length, length, `window ${index + 1} change column`);
  assert(detail.insideFrom >= 0 && detail.insideTo < length && detail.insideFrom <= detail.insideTo);
  for (const method of methodOrder) {
    assert.equal(detail.score[method].length, length, `window ${index + 1} ${method} scores`);
    assert.equal(detail.exceeds[method].length, length, `window ${index + 1} ${method} ranks`);
    assert(detail.exceeds[method].every(rank => rank >= 0 && rank <= available[method].length));
    // The rank must bracket the shown score between two offerable thresholds.
    // Shown scores are rounded to four decimals, so allow that much slack; the
    // rank itself is exact, which is why it and not the score marks alerts.
    const slack = 1e-4;
    const levels = available[method];
    detail.exceeds[method].forEach((rank, position) => {
      const score = detail.score[method][position];
      if (rank > 0) assert(levels[rank - 1] < score + slack, `${method}: level below the score at row ${position}`);
      if (rank < levels.length) assert(levels[rank] > score - slack, `${method}: level above the score at row ${position}`);
    });
  }
});
// Timestamps: the recorded start, the verified five-minute step and the windows.
assert.equal(formatTimestamp(rowTimestamp(seriesStart, 0, stepMinutes)), seriesStart.slice(0, 16));
windowDetail.forEach((detail, index) => {
  const start = formatTimestamp(rowTimestamp(seriesStart, detail.firstRow + detail.insideFrom, stepMinutes));
  const end = formatTimestamp(rowTimestamp(seriesStart, detail.firstRow + detail.insideTo, stepMinutes));
  assert.equal(start, eventWindows[index][0].slice(0, 16), `window ${index + 1} starts where the annotation does`);
  assert.equal(end, eventWindows[index][1].slice(0, 16), `window ${index + 1} ends where the annotation does`);
});
record('window detail');

// ------------------------------------------------ displayed program agreement
assert.equal(Object.keys(examples).length, 5);
const isolationScores = isolationExpectations([0, 1, 2, 3, 12], 3).rows.map(row => row.score);
assert.equal(isolationScores[4], Math.max(...isolationScores), 'the exact model and the displayed program agree on the winner');
assert(examples.isolation.expected.includes('Most easily isolated position: 12.0'));
assert(examples.lofModes.expected.includes('1.333333'), 'the training factor four thirds appears');
assert(examples.lofModes.expected.includes('Wrong training comparison: [0.875 0.875 0.875 0.875 0.875 0.875]'));
assert(examples.lofArithmetic.expected.includes('1.458333'), 'the factor of query 4 appears');
assert(examples.kernel.expected.includes('midpoint=-0.141278'));
assert(examples.temperature.expected.includes(`fit/cal/test: ${rowCounts.reference} ${rowCounts.calibration} ${rowCounts.test}`));
for (const row of author.realData.results) {
  assert(examples.temperature.expected.includes(String(row.alertsOutsideWindows)),
    `the displayed table shows ${row.alertsOutsideWindows}`);
}
record('displayed programs');

// --------------------------------------------------------------------- record
const sources = [
  'src/learn/data/anomaly-detection-models.js',
  'src/learn/data/anomaly-temperature-data.js',
  'src/learn/data/anomaly-detection-examples.js',
  'src/learn/components/lesson-labs/AnomalyDetectionLabs.jsx',
  'src/learn/components/lesson-labs/AnomalyTemperatureLab.jsx',
  'src/learn/components/lesson-labs/AnomalyDetectionShared.jsx',
  'src/learn/components/lesson-labs/AnomalyDetectionFigures.jsx',
  'src/learn/components/lesson-labs/anomaly-detection-labs.css',
  'src/learn/data/topics/anomaly-outlier-detection-isolation-forest-one-class-svm-lof.jsx',
];
const hash = file => crypto.createHash('sha256').update(fs.readFileSync(file)).digest('hex');
const evidence = {
  checkedAt: new Date().toISOString(),
  sourceHashes: Object.fromEntries(sources.filter(fs.existsSync).map(file => [file, hash(file)])),
  verifierHash: hash('scripts/verify-anomaly-detection-models.mjs'),
  counts,
  totalGroupedChecks: Object.values(counts).reduce((sum, value) => sum + value, 0),
  scope: 'Browser anomaly models against the content-phase native probes (three isolation fixtures, three LOF neighbour counts with four queries each, four kernel gammas, four alert populations), the manuscript fixtures and practice answers A, B, E and F, and the generated real-series module: counts, the published table, the threshold ladders, the overview tiling and exact window-detail alert marking reproducing every published inside-window count and window hit.',
  limitations: [
    'The real-series scores are precomputed native outputs; the browser reproduces outcomes, not the fits.',
    'Displayed program output is executed separately by scripts/verify-anomaly-detection-examples.py.',
    'Rendering, interaction and independent review are separate steps.',
  ],
  passed: true,
};
fs.mkdirSync('docs/teaching/evidence', { recursive: true });
fs.writeFileSync('docs/teaching/evidence/anomaly-detection-models.json', JSON.stringify(evidence, null, 2) + '\n');
console.log(`PASS: ${evidence.totalGroupedChecks} grouped anomaly-detection model checks.`);
