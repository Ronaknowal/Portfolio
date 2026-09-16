/** Pure models for the imbalanced-learning lesson.
 *
 * Everything a figure or an investigation draws is computed here, so that a
 * drawn band, flow, step, crossing, segment or bar is an asserted mathematical
 * claim rather than a shape chosen inside a component. Nothing is interpolated
 * from a stored picture, and no proportion is eyeballed in a stylesheet.
 *
 * Three categories of quantity live here and are never mixed:
 *
 *   1. Exact declared constructions. The 1,000-case confusion table, the
 *      three-record score queue, the two population flows, the single weighted
 *      gradient step, the weighted-loss optimum, the small SMOTE geometry and
 *      the focal loss-mass example are exact arithmetic on stated inputs.
 *   2. Recorded observations. The Yeast quantities are read from
 *      imbalance-data.js; the functions here only threshold, count, rank and
 *      compare them. Nothing here refits a classifier.
 *   3. Hypothetical decision costs. One unit per false alarm and twelve per
 *      missed positive are a teaching assumption, carried as data rather than
 *      baked into a formula, so multiplying both by a constant is testable.
 *
 * Two conventions are load-bearing and are kept everywhere:
 *
 *   * An undefined quantity is `null`, never zero and never a tiny denominator.
 *     Precision is undefined when nothing is selected; recall is undefined when
 *     there are no actual positives; the cost cutoff is undefined when both
 *     costs are zero.
 *   * Selection is `score >= threshold`, so a threshold always takes a whole
 *     tied group. The no-alert policy is the candidate above every score, and
 *     it is represented by `Infinity` rather than by a magic number.
 *
 * Every entry point refuses input it cannot honour rather than substituting a
 * silent default: a non-finite number, an out-of-range setting, an empty record
 * set or a neighbour count with nothing to choose from raises a RangeError.
 */

/* ------------------------------------------------------------------ guards */

export const limits = {
  /** Investigation 1: an editable scored queue. */
  score: { minimum: 0, maximum: 1 },
  maximumRecords: 12,
  minimumRecords: 1,
  /** Investigation 2: two declared action costs and one posterior. */
  cost: { minimum: 0, maximum: 50 },
  posterior: { minimum: 0, maximum: 1 },
  /** Investigation 3: a population probability and two positive class weights. */
  probability: { minimum: 0.01, maximum: 0.99 },
  weight: { minimum: 0.1, maximum: 20 },
  weightedScore: { minimum: 0.01, maximum: 0.99 },
  /** Investigation 4: a small editable point cloud and its declared metric. */
  coordinate: { minimum: -5, maximum: 5 },
  scaleDivisor: { minimum: 0.1, maximum: 10 },
  fraction: { minimum: 0, maximum: 1 },
  maximumPoints: 12,
  minimumMinority: 2,
  /** Investigation 5: the recorded 200-row tuning queue. */
  trialThreshold: { minimum: 0, maximum: 1 },
  tolerance: 1e-12,
  /** Curves are sampled, never drawn freehand. */
  curveSamples: 121,
};

export function checkFinite(value, name) {
  if (typeof value !== 'number' || !Number.isFinite(value)) {
    throw new RangeError(`${name} must be a finite number.`);
  }
  return value;
}
export function checkRange(value, range, name) {
  checkFinite(value, name);
  if (value < range.minimum || value > range.maximum) {
    throw new RangeError(`Keep ${name} between ${range.minimum} and ${range.maximum}.`);
  }
  return value;
}
function checkWhole(value, range, name) {
  if (!Number.isInteger(value)) throw new RangeError(`${name} must be a whole number.`);
  return checkRange(value, range, name);
}
const sum = values => values.reduce((total, value) => total + value, 0);

/* ================================================================= §1 counts */

/** The four cells of one confusion table, and every quantity derived from them.
 *
 * Precision and recall are `null` rather than 0 when their denominator is empty,
 * and the reason is reported so an interface can say which denominator vanished
 * instead of printing a number with no meaning.
 */
export function confusion({ tp, fp, fn, tn }) {
  for (const [name, value] of Object.entries({ tp, fp, fn, tn })) {
    if (!Number.isInteger(value) || value < 0) throw new RangeError(`${name} must be a nonnegative whole number.`);
  }
  const total = tp + fp + fn + tn;
  if (total === 0) throw new RangeError('A confusion table needs at least one case.');
  const positives = tp + fn;
  const negatives = fp + tn;
  const alerts = tp + fp;
  const cleared = fn + tn;
  return {
    tp, fp, fn, tn, total, positives, negatives, alerts, cleared,
    accuracy: (tp + tn) / total,
    precision: alerts === 0 ? null : tp / alerts,
    recall: positives === 0 ? null : tp / positives,
    specificity: negatives === 0 ? null : tn / negatives,
    falsePositiveRate: negatives === 0 ? null : fp / negatives,
    balancedAccuracy: positives === 0 || negatives === 0 ? null
      : (tp / positives + tn / negatives) / 2,
    f1: 2 * tp + fp + fn === 0 ? null : (2 * tp) / (2 * tp + fp + fn),
    prevalence: positives / total,
    /** Why a quantity is undefined, so an interface can name the denominator. */
    undefinedBecause: {
      precision: alerts === 0 ? 'nothing was selected, so TP + FP is zero' : null,
      recall: positives === 0 ? 'the assessment holds no actual positives, so TP + FN is zero' : null,
      specificity: negatives === 0 ? 'the assessment holds no actual negatives, so TN + FP is zero' : null,
    },
  };
}

/** F-beta on the counts, with beta emphasising missed positives. */
export function fBeta({ tp, fp, fn }, beta) {
  checkFinite(beta, 'beta');
  if (beta < 0) throw new RangeError('beta must be nonnegative.');
  const denominator = (1 + beta ** 2) * tp + beta ** 2 * fn + fp;
  return denominator === 0 ? null : ((1 + beta ** 2) * tp) / denominator;
}

/** The two class bands of figure 1, each normalised within its own class.
 *
 * The widths are fractions of a class, not of the population, which is exactly
 * why the figure has to say so: six missed positives are 30% of one band and
 * 0.6% of the page. Both numbers are returned so the caption can carry them.
 */
export function classBands(counts) {
  const table = confusion(counts);
  const band = (label, parts, size) => ({
    label, size,
    shareOfPopulation: size / table.total,
    parts: parts.map(part => ({
      ...part,
      withinClass: size === 0 ? null : part.count / size,
      ofPopulation: part.count / table.total,
    })),
  });
  return {
    table,
    bands: [
      band('actual positives', [
        { name: 'detected', key: 'tp', count: table.tp },
        { name: 'missed', key: 'fn', count: table.fn },
      ], table.positives),
      band('actual negatives', [
        { name: 'false alarm', key: 'fp', count: table.fp },
        { name: 'cleared', key: 'tn', count: table.tn },
      ], table.negatives),
    ],
    /** Every band part sums back to its class, and the classes to the whole. */
    conserved: table.tp + table.fn === table.positives
      && table.fp + table.tn === table.negatives
      && table.positives + table.negatives === table.total,
  };
}

/* ============================================================= §2 the queue */

/** One threshold applied to a fixed ranking. Selection is `score >= threshold`,
 * so a tied group is always taken whole. */
export function queueAt(records, threshold) {
  const rows = checkRecords(records);
  if (threshold !== Infinity) checkFinite(threshold, 'the threshold');
  const selected = rows.filter(row => row.score >= threshold);
  const unselected = rows.filter(row => !(row.score >= threshold));
  const table = confusion({
    tp: selected.filter(row => row.truth === 1).length,
    fp: selected.filter(row => row.truth === 0).length,
    fn: unselected.filter(row => row.truth === 1).length,
    tn: unselected.filter(row => row.truth === 0).length,
  });
  return {
    threshold, ...table,
    selectedIds: selected.map(row => row.id),
    unselectedIds: unselected.map(row => row.id),
    bins: {
      tp: selected.filter(row => row.truth === 1).map(row => row.id),
      fp: selected.filter(row => row.truth === 0).map(row => row.id),
      fn: unselected.filter(row => row.truth === 1).map(row => row.id),
      tn: unselected.filter(row => row.truth === 0).map(row => row.id),
    },
  };
}

function checkRecords(records) {
  if (!Array.isArray(records) || records.length < limits.minimumRecords) {
    throw new RangeError('Keep at least one record in the queue.');
  }
  if (records.length > limits.maximumRecords) {
    throw new RangeError(`Keep at most ${limits.maximumRecords} records in the queue.`);
  }
  const seen = new Set();
  return records.map(record => {
    if (typeof record.id !== 'string' || record.id === '') throw new RangeError('Every record needs an identifier.');
    if (seen.has(record.id)) throw new RangeError(`Record identifiers must be distinct; ${record.id} repeats.`);
    seen.add(record.id);
    checkRange(record.score, limits.score, `the score of ${record.id}`);
    if (record.truth !== 0 && record.truth !== 1) throw new RangeError(`The truth of ${record.id} must be 0 or 1.`);
    return { id: record.id, score: record.score, truth: record.truth };
  });
}

/** Every operating point the data actually has: each distinct score, descending,
 * plus the no-alert policy above all of them. There is nothing between two
 * observed scores, which is why the figure draws steps and not a line. */
export function thresholdLadder(records) {
  const rows = checkRecords(records);
  const distinct = [...new Set(rows.map(row => row.score))].sort((a, b) => b - a);
  return [Infinity, ...distinct].map(threshold => ({
    ...queueAt(rows, threshold),
    label: threshold === Infinity ? 'above every score (no alert)' : `score ≥ ${threshold}`,
    isNoAlert: threshold === Infinity,
    /** The group of records this threshold newly admits, always taken whole. */
    group: threshold === Infinity ? [] : rows.filter(row => row.score === threshold).map(row => row.id),
  }));
}

/** Noninterpolated average precision: recall increments at distinct score
 * groups, weighted by the precision at each. Undefined without positives. */
export function averagePrecisionOf(records) {
  const rows = checkRecords(records);
  const positives = rows.filter(row => row.truth === 1).length;
  if (positives === 0) {
    return { value: null, steps: [], undefinedBecause: 'the queue holds no actual positives' };
  }
  const distinct = [...new Set(rows.map(row => row.score))].sort((a, b) => b - a);
  let previousRecall = 0;
  const steps = distinct.map(threshold => {
    const point = queueAt(rows, threshold);
    const recall = point.recall;
    const increment = recall - previousRecall;
    previousRecall = recall;
    return {
      threshold, recall, increment,
      precision: point.precision ?? 0,
      contribution: increment * (point.precision ?? 0),
      group: rows.filter(row => row.score === threshold).map(row => row.id),
    };
  });
  return { value: sum(steps.map(step => step.contribution)), steps, undefinedBecause: null };
}

/** Exact precision-recall coordinates, one per distinct score group. No point
 * is invented between two observed groups. */
export function precisionRecallPoints(records) {
  return thresholdLadder(records)
    .filter(point => !point.isNoAlert)
    .map(point => ({ threshold: point.threshold, recall: point.recall, precision: point.precision }));
}

/** Compare two thresholds on the same queue. Recall cannot rise as the gate
 * rises; precision can move either way, which is the whole point. */
export function queueComparison(records, from, to) {
  const before = queueAt(records, from);
  const after = queueAt(records, to);
  const direction = (a, b) => {
    if (a === null || b === null) return 'undefined';
    const difference = b - a;
    if (Math.abs(difference) <= limits.tolerance) return 'unchanged';
    return difference > 0 ? 'increase' : 'decrease';
  };
  return {
    before, after,
    precision: direction(before.precision, after.precision),
    recall: direction(before.recall, after.recall),
    /** Asserted, not assumed: a rising gate never admits more positives. */
    recallMonotone: to < from || after.tp <= before.tp,
  };
}

/* ================================================== §2 · two populations */

/** A fixed conditional detector applied to two populations.
 *
 * The counts are expectations under the declared rates, so they are deliberately
 * left fractional rather than rounded into a fake integer census.
 */
export function populationFlow({ population, prevalence, tpr, fpr }) {
  checkFinite(population, 'the population size');
  if (population <= 0) throw new RangeError('The population must be positive.');
  checkRange(prevalence, { minimum: 0, maximum: 1 }, 'the prevalence');
  checkRange(tpr, { minimum: 0, maximum: 1 }, 'the true-positive rate');
  checkRange(fpr, { minimum: 0, maximum: 1 }, 'the false-positive rate');
  const positives = population * prevalence;
  const negatives = population - positives;
  const tp = positives * tpr;
  const fn = positives - tp;
  const fp = negatives * fpr;
  const tn = negatives - fp;
  const alerts = tp + fp;
  return {
    population, prevalence, tpr, fpr, positives, negatives, tp, fn, fp, tn, alerts,
    precision: alerts === 0 ? null : tp / alerts,
    /** The same number by the manuscript's prevalence identity, as a check that
     * the flow and the formula are one claim and not two. */
    precisionByFormula: prevalence * tpr + (1 - prevalence) * fpr === 0 ? null
      : (prevalence * tpr) / (prevalence * tpr + (1 - prevalence) * fpr),
    integral: [tp, fn, fp, tn].every(value => Math.abs(value - Math.round(value)) <= limits.tolerance),
    conserved: Math.abs(tp + fn + fp + tn - population) <= limits.tolerance * population,
  };
}

/* ============================================== §3 · two expected costs */

/** The two action risks under a declared cost model, and where they cross.
 *
 * Correct decisions cost zero. Selecting a negative costs `costFP`; missing a
 * positive costs `costFN`. With both costs zero every action ties and there is
 * no cutoff at all, which is returned as `null` rather than as 0 or 0.5.
 */
export function actionRisks(posterior, costFP, costFN) {
  checkRange(posterior, limits.posterior, 'the posterior');
  checkRange(costFP, limits.cost, 'the false-alarm cost');
  checkRange(costFN, limits.cost, 'the missed-positive cost');
  const selectRisk = (1 - posterior) * costFP;
  const skipRisk = posterior * costFN;
  const difference = selectRisk - skipRisk;
  const tie = Math.abs(difference) <= limits.tolerance * Math.max(1, selectRisk, skipRisk);
  const total = costFP + costFN;
  return {
    posterior, costFP, costFN, selectRisk, skipRisk, difference, tie,
    cutoff: total === 0 ? null : costFP / total,
    /** The manuscript's declared convention: select when the risks are equal. */
    action: tie || difference < 0 ? 'select' : 'skip',
    undefinedBecause: total === 0
      ? 'both costs are zero, so every action has expected cost zero and no cutoff separates them'
      : null,
    /** Both lines over the whole posterior axis, on one shared vertical scale. */
    lines: {
      select: [[0, costFP], [1, 0]],
      skip: [[0, 0], [1, costFN]],
      ceiling: Math.max(costFP, costFN, selectRisk, skipRisk),
    },
  };
}

/* ============================================ §4 · one weighted update */

/** One gradient step on the declared normalised weighted objective.
 *
 * The objective divides by the total weight, so scaling every weight by a common
 * factor leaves the gradient unchanged. The intercept is unpenalised, and at the
 * initial parameters the penalty contributes nothing, so the drawn arithmetic is
 * exactly the two weighted sums.
 */
export function weightedStep({ rows, intercept = 0, coefficient = 0, penalty = 0.01, rate = 0.4 }) {
  if (!Array.isArray(rows) || rows.length === 0) throw new RangeError('A weighted step needs at least one row.');
  checkFinite(intercept, 'the intercept');
  checkFinite(coefficient, 'the coefficient');
  checkFinite(penalty, 'the penalty');
  checkFinite(rate, 'the step size');
  if (penalty < 0) throw new RangeError('The penalty must be nonnegative.');
  const prepared = rows.map((row, index) => {
    checkFinite(row.x, `the feature of row ${index + 1}`);
    if (row.y !== 0 && row.y !== 1) throw new RangeError(`The label of row ${index + 1} must be 0 or 1.`);
    checkFinite(row.weight, `the weight of row ${index + 1}`);
    if (row.weight <= 0) throw new RangeError(`The weight of row ${index + 1} must be positive.`);
    return { ...row };
  });
  const totalWeight = sum(prepared.map(row => row.weight));
  const logistic = value => 1 / (1 + Math.exp(-value));
  const contributions = prepared.map(row => {
    const margin = intercept + coefficient * row.x;
    const score = logistic(margin);
    const residual = score - row.y;
    return {
      ...row, margin, score, residual,
      interceptContribution: row.weight * residual,
      coefficientContribution: row.weight * residual * row.x,
    };
  });
  const interceptGradient = sum(contributions.map(row => row.interceptContribution)) / totalWeight;
  // The intercept is unpenalised; the coefficient carries the penalty term.
  const coefficientGradient = sum(contributions.map(row => row.coefficientContribution)) / totalWeight
    + penalty * coefficient;
  const interceptAfter = intercept - rate * interceptGradient;
  const coefficientAfter = coefficient - rate * coefficientGradient;
  const scoreAt = (b, w) => x => logistic(b + w * x);
  return {
    rows: contributions, totalWeight, interceptGradient, coefficientGradient,
    interceptAfter, coefficientAfter,
    scoresAfter: prepared.map(row => logistic(interceptAfter + coefficientAfter * row.x)),
    before: scoreAt(intercept, coefficient),
    after: scoreAt(interceptAfter, coefficientAfter),
    penaltyContribution: penalty * coefficient,
  };
}

/** The common balanced rule: each class carries equal total weight. */
export function balancedWeights(classCounts) {
  if (!Array.isArray(classCounts) || classCounts.length < 2) {
    throw new RangeError('Balanced weights need at least two observed classes.');
  }
  if (classCounts.some(count => !Number.isInteger(count) || count <= 0)) {
    throw new RangeError('Every observed class needs a positive whole count.');
  }
  const n = sum(classCounts);
  const k = classCounts.length;
  const weights = classCounts.map(count => n / (k * count));
  return {
    weights, n, k,
    totalPerClass: weights.map((weight, index) => weight * classCounts[index]),
    /** Equal total weight per class is the defining property, so check it. */
    equalised: weights.every((weight, index) =>
      Math.abs(weight * classCounts[index] - n / k) <= limits.tolerance * n),
    /** After a resampler has already equalised the counts, these are all one. */
    allOne: weights.every(weight => Math.abs(weight - 1) <= limits.tolerance),
  };
}

/* =========================================== §4 · the weighted optimum */

/** The population-optimal score under weighted log loss, and its inverse.
 *
 * Only the ratio of the weights matters, which is why multiplying both by a
 * common positive factor moves the loss height and not the minimiser. The
 * derivative at the optimum is zero and the second derivative is positive; both
 * are returned so a verifier can check the claim rather than trust the algebra.
 */
export function weightedOptimum(probability, positiveWeight, negativeWeight) {
  checkFinite(probability, 'the population probability');
  if (probability <= 0 || probability >= 1) throw new RangeError('The population probability must be strictly between zero and one.');
  checkRange(positiveWeight, limits.weight, 'the positive weight');
  checkRange(negativeWeight, limits.weight, 'the negative weight');
  const massPositive = positiveWeight * probability;
  const massNegative = negativeWeight * (1 - probability);
  const optimum = massPositive / (massPositive + massNegative);
  const loss = q => -massPositive * Math.log(q) - massNegative * Math.log(1 - q);
  const derivative = q => -massPositive / q + massNegative / (1 - q);
  const secondDerivative = q => massPositive / q ** 2 + massNegative / (1 - q) ** 2;
  return {
    probability, positiveWeight, negativeWeight, massPositive, massNegative, optimum,
    loss, derivative, secondDerivative,
    derivativeAtOptimum: derivative(optimum),
    secondDerivativeAtOptimum: secondDerivative(optimum),
    lossAtOptimum: loss(optimum),
    /** The equivalent cutoff on the original probability. */
    equivalentProbabilityCutoff: negativeWeight / (positiveWeight + negativeWeight),
    ratio: positiveWeight / negativeWeight,
  };
}

/** Recover the original probability from a weighted-loss optimum. */
export function inverseWeightedOptimum(optimum, positiveWeight, negativeWeight) {
  checkFinite(optimum, 'the weighted score');
  if (optimum <= 0 || optimum >= 1) throw new RangeError('The weighted score must be strictly between zero and one.');
  checkRange(positiveWeight, limits.weight, 'the positive weight');
  checkRange(negativeWeight, limits.weight, 'the negative weight');
  return (negativeWeight * optimum) / (positiveWeight * (1 - optimum) + negativeWeight * optimum);
}

/** The loss curve, sampled strictly inside (0, 1) so no point evaluates log 0. */
export function weightedLossCurve(probability, positiveWeight, negativeWeight, samples = limits.curveSamples) {
  const model = weightedOptimum(probability, positiveWeight, negativeWeight);
  const margin = 0.5 / (samples + 1);
  const points = Array.from({ length: samples }, (_, index) => {
    const q = margin + (1 - 2 * margin) * index / (samples - 1);
    return [q, model.loss(q)];
  });
  return {
    model, points,
    domain: [points[0][0], points.at(-1)[0]],
    /** The sampled minimum is a drawing artefact; the formula is authoritative. */
    sampledMinimum: points.reduce((best, point) => (point[1] < best[1] ? point : best))[0],
    exactMinimum: model.optimum,
  };
}

/* ============================================== §5 · SMOTE geometry */

/** One interpolated vector, the neighbour ranking behind it, and what the rule
 * did not look at.
 *
 * Distance uses the declared scale divisors; interpolation happens in the
 * original feature units. Ties break on distance and then on stable identifier
 * order, and the tied identifiers are reported rather than quietly resolved.
 */
export function smoteConstruction({ points, scaleX = 1, scaleY = 1, anchorId, k = 1, neighbourRank = 0, fraction = 0.5 }) {
  const cloud = checkCloud(points);
  checkRange(scaleX, limits.scaleDivisor, 'the x scale divisor');
  checkRange(scaleY, limits.scaleDivisor, 'the y scale divisor');
  checkRange(fraction, limits.fraction, 'the interpolation fraction');
  const minority = cloud.filter(point => point.cls === 'minority');
  const majority = cloud.filter(point => point.cls === 'majority');
  if (minority.length < limits.minimumMinority) {
    throw new RangeError(`Keep at least ${limits.minimumMinority} minority points.`);
  }
  const anchor = minority.find(point => point.id === anchorId);
  if (!anchor) throw new RangeError(`The anchor ${anchorId} is not a minority point.`);
  const available = minority.length - 1;
  checkWhole(k, { minimum: 1, maximum: available }, 'k');
  checkWhole(neighbourRank, { minimum: 0, maximum: k - 1 }, 'the neighbour rank');

  const distances = minority
    .filter(point => point.id !== anchor.id)
    .map(point => ({
      id: point.id, x: point.x, y: point.y,
      distance: Math.hypot((point.x - anchor.x) / scaleX, (point.y - anchor.y) / scaleY),
    }))
    .sort((a, b) => Math.abs(a.distance - b.distance) > limits.tolerance
      ? a.distance - b.distance : (a.id < b.id ? -1 : a.id > b.id ? 1 : 0))
    .map((entry, index) => ({ ...entry, rank: index + 1 }));
  const neighbours = distances.slice(0, k);
  const neighbour = neighbours[neighbourRank];
  // A tie that straddles the k boundary changes which points are even eligible,
  // so it is named rather than left to a sort's stability.
  const boundaryTie = k < distances.length
    && Math.abs(distances[k - 1].distance - distances[k].distance) <= limits.tolerance;
  const tiedWithChosen = distances
    .filter(entry => entry.id !== neighbour.id
      && Math.abs(entry.distance - neighbour.distance) <= limits.tolerance)
    .map(entry => entry.id);

  // One scalar fraction for the entire vector: the line-segment construction.
  const generated = {
    x: anchor.x + fraction * (neighbour.x - anchor.x),
    y: anchor.y + fraction * (neighbour.y - anchor.y),
  };
  const collisions = cloud.filter(point =>
    Math.abs(point.x - generated.x) <= limits.tolerance
    && Math.abs(point.y - generated.y) <= limits.tolerance);
  return {
    anchor, neighbour, neighbours, distances, k, neighbourRank, fraction, scaleX, scaleY,
    minority, majority, generated,
    segment: [[anchor.x, anchor.y], [neighbour.x, neighbour.y]],
    zeroLength: Math.abs(neighbour.x - anchor.x) <= limits.tolerance
      && Math.abs(neighbour.y - anchor.y) <= limits.tolerance,
    boundaryTie, tiedWithChosen,
    collidesWithMajority: collisions.some(point => point.cls === 'majority'),
    collisionIds: collisions.map(point => point.id),
    /** What the interpolation consulted, and what it never looked at. */
    consulted: [anchor.id, neighbour.id],
    ignored: majority.map(point => point.id),
    availableNeighbours: available,
  };
}

function checkCloud(points) {
  if (!Array.isArray(points) || points.length === 0) throw new RangeError('The cloud needs at least one point.');
  if (points.length > limits.maximumPoints) {
    throw new RangeError(`Keep at most ${limits.maximumPoints} points in the cloud.`);
  }
  const seen = new Set();
  return points.map(point => {
    if (typeof point.id !== 'string' || point.id === '') throw new RangeError('Every point needs an identifier.');
    if (seen.has(point.id)) throw new RangeError(`Point identifiers must be distinct; ${point.id} repeats.`);
    seen.add(point.id);
    if (point.cls !== 'minority' && point.cls !== 'majority') {
      throw new RangeError(`The class of ${point.id} must be minority or majority.`);
    }
    checkRange(point.x, limits.coordinate, `the x coordinate of ${point.id}`);
    checkRange(point.y, limits.coordinate, `the y coordinate of ${point.id}`);
    return { id: point.id, cls: point.cls, x: point.x, y: point.y };
  });
}

/* =================================== §7 · thresholds on recorded scores */

/** Counts at one threshold over recorded labels and scores. */
export function countsAt(labels, scores, threshold) {
  if (!Array.isArray(labels) || !Array.isArray(scores) || labels.length !== scores.length) {
    throw new RangeError('Labels and scores must be arrays of the same length.');
  }
  if (labels.length === 0) throw new RangeError('There must be at least one record.');
  let tp = 0; let fp = 0; let fn = 0; let tn = 0;
  for (let index = 0; index < labels.length; index += 1) {
    const selected = scores[index] >= threshold;
    if (labels[index] === 1) { if (selected) tp += 1; else fn += 1; } else if (selected) fp += 1; else tn += 1;
  }
  return confusion({ tp, fp, fn, tn });
}

/** The cost of one operating point under a declared cost pair. */
export function costOf(counts, costFP, costFN) {
  checkRange(costFP, limits.cost, 'the false-alarm cost');
  checkRange(costFN, limits.cost, 'the missed-positive cost');
  return costFP * counts.fp + costFN * counts.fn;
}

/** Sweep every candidate: each distinct score, descending, plus the no-alert
 * policy above all of them.
 *
 * Candidates are traversed descending and the minimum is kept with a strict
 * comparison, which is exactly the declared tie rule: a cost tie keeps the
 * higher threshold. Every tied optimum is reported so the rule can be shown.
 */
export function thresholdSweep(labels, scores, costFP, costFN) {
  const distinct = [...new Set(scores)].sort((a, b) => b - a);
  const candidates = [Infinity, ...distinct].map(threshold => {
    const counts = countsAt(labels, scores, threshold);
    return { threshold, counts, cost: costOf(counts, costFP, costFN), isNoAlert: threshold === Infinity };
  });
  let chosen = candidates[0];
  for (const candidate of candidates) {
    if (candidate.cost < chosen.cost) chosen = candidate;
  }
  const tied = candidates.filter(candidate => candidate.cost === chosen.cost);
  return {
    candidates, chosen, tiedCandidates: tied,
    /** The declared rule, stated as a checkable property of the result. */
    tieKeptHigher: tied.every(candidate => candidate.threshold <= chosen.threshold),
    costFP, costFN,
  };
}

/** The ranked queue behind a top-k claim. Ties break on stored order, never on
 * a label, so no unseen truth can reach the policy. */
export function rankedQueue(ids, labels, scores, size) {
  if (!Number.isInteger(size) || size <= 0) throw new RangeError('The queue size must be a positive whole number.');
  const order = ids.map((id, index) => ({ id, index, label: labels[index], score: scores[index] }))
    .sort((a, b) => (b.score - a.score) || (a.index - b.index));
  const top = order.slice(0, size);
  const boundary = top.length === order.length ? null : order[top.length];
  return {
    top,
    positives: top.filter(entry => entry.label === 1).length,
    precisionAtK: top.length === 0 ? null : top.filter(entry => entry.label === 1).length / top.length,
    /** A tie straddling the capacity boundary is a policy question, so say so. */
    boundaryTie: boundary !== null && top.length > 0
      && Math.abs(top.at(-1).score - boundary.score) <= limits.tolerance,
  };
}

/** Average precision over recorded labels and scores, grouped at distinct
 * scores. Used to reproduce the study's recorded values. */
export function averagePrecisionOfScores(labels, scores) {
  const positives = labels.filter(label => label === 1).length;
  if (positives === 0) return null;
  const distinct = [...new Set(scores)].sort((a, b) => b - a);
  let previousRecall = 0;
  let total = 0;
  for (const threshold of distinct) {
    const counts = countsAt(labels, scores, threshold);
    const recall = counts.tp / positives;
    total += (recall - previousRecall) * (counts.precision ?? 0);
    previousRecall = recall;
  }
  return total;
}

/** ROC-AUC by the pairwise definition, ties counted as a half. */
export function rocAucOfScores(labels, scores) {
  const positive = scores.filter((_, index) => labels[index] === 1);
  const negative = scores.filter((_, index) => labels[index] === 0);
  if (positive.length === 0 || negative.length === 0) return null;
  let wins = 0;
  for (const p of positive) for (const n of negative) wins += p > n ? 1 : p === n ? 0.5 : 0;
  return wins / (positive.length * negative.length);
}

/** Mean squared probability error. Named for what it is: a squared error on
 * probabilities, not a calibration measurement. */
export function brierOfScores(labels, scores) {
  return sum(scores.map((score, index) => (score - labels[index]) ** 2)) / scores.length;
}

/* ================================================= §8 · focal loss mass */

/** Two populations of examples, their cross-entropy mass and their focal mass.
 *
 * The modulating factor is `(1 - p_t)^gamma`, so at gamma 0 this returns
 * weighted cross-entropy exactly. The ratio of the two totals is returned
 * because that ratio, not either total, is the mechanism the figure shows.
 */
export function focalMass({ groups, gamma = 2, alpha = 1 }) {
  checkFinite(gamma, 'gamma');
  if (gamma < 0) throw new RangeError('gamma must be nonnegative.');
  checkFinite(alpha, 'alpha');
  if (alpha <= 0) throw new RangeError('alpha must be positive.');
  if (!Array.isArray(groups) || groups.length === 0) throw new RangeError('Focal mass needs at least one group.');
  const rows = groups.map(group => {
    if (!Number.isInteger(group.count) || group.count <= 0) {
      throw new RangeError(`${group.name}: the count must be a positive whole number.`);
    }
    checkRange(group.pt, { minimum: 1e-6, maximum: 1 - 1e-9 }, `${group.name}: the true-class probability`);
    const crossEntropy = -Math.log(group.pt);
    const modulator = (1 - group.pt) ** gamma;
    const focal = alpha * modulator * crossEntropy;
    return {
      ...group, crossEntropy, modulator, focal,
      crossEntropyTotal: group.count * crossEntropy,
      focalTotal: group.count * focal,
    };
  });
  const crossEntropyTotal = sum(rows.map(row => row.crossEntropyTotal));
  const focalTotal = sum(rows.map(row => row.focalTotal));
  return {
    rows, gamma, alpha, crossEntropyTotal, focalTotal,
    /** At gamma 0 every modulator is exactly 1, so this is weighted CE. */
    reducesToCrossEntropy: gamma === 0,
    shares: {
      crossEntropy: rows.map(row => row.crossEntropyTotal / crossEntropyTotal),
      focal: rows.map(row => row.focalTotal / focalTotal),
    },
  };
}

/** The derivative of focal loss in p_t. The product rule keeps a term that a
 * "cross-entropy gradient times 0.01" claim would drop, so it is returned
 * split into its two parts. */
export function focalDerivative(pt, gamma = 2, alpha = 1) {
  checkRange(pt, { minimum: 1e-6, maximum: 1 - 1e-9 }, 'the true-class probability');
  checkFinite(gamma, 'gamma');
  if (gamma < 0) throw new RangeError('gamma must be nonnegative.');
  const modulatorTerm = alpha * gamma * (1 - pt) ** (gamma - 1) * Math.log(pt);
  const logTerm = -alpha * ((1 - pt) ** gamma) / pt;
  return {
    pt, gamma, alpha, modulatorTerm, logTerm,
    value: modulatorTerm + logTerm,
    /** What the same derivative becomes with respect to a logit. */
    withRespectToLogit: (modulatorTerm + logTerm) * pt * (1 - pt),
    /** The dropped term, for the claim the manuscript refutes. */
    crossEntropyOnly: logTerm,
  };
}

/* ============================================ §8 · a changed prior */

/** Prior shift under unchanged class-conditional feature distributions. */
export function priorShift({ sampledPosterior, sampledPrevalence, deploymentPrevalence }) {
  checkRange(sampledPosterior, { minimum: 0, maximum: 1 }, 'the sampled posterior');
  checkRange(sampledPrevalence, { minimum: 0, maximum: 1 }, 'the sampled prevalence');
  checkRange(deploymentPrevalence, { minimum: 0, maximum: 1 }, 'the deployment prevalence');
  if (sampledPrevalence === 0 || sampledPrevalence === 1) {
    throw new RangeError('The sampled prevalence must be strictly between 0 and 1.');
  }
  if (sampledPosterior === 0) return { posterior: 0, odds: 0, multiplier: null, degenerate: true };
  if (sampledPosterior === 1) return { posterior: 1, odds: Infinity, multiplier: null, degenerate: true };
  const sampledOdds = sampledPosterior / (1 - sampledPosterior);
  const multiplier = (deploymentPrevalence / (1 - deploymentPrevalence))
    / (sampledPrevalence / (1 - sampledPrevalence));
  const odds = sampledOdds * multiplier;
  return {
    sampledOdds, multiplier, odds, posterior: odds / (1 + odds), degenerate: false,
  };
}

/* ============================================ §8 · budget the operations */

/** What balancing does to stored rows, and what it does not do to evidence. */
export function rowExpansion(majority, minority) {
  if (!Number.isInteger(majority) || !Number.isInteger(minority) || majority <= 0 || minority <= 0) {
    throw new RangeError('Both class counts must be positive whole numbers.');
  }
  if (minority > majority) throw new RangeError('The minority count cannot exceed the majority count.');
  return {
    majority, minority,
    originalRows: majority + minority,
    oversampledRows: 2 * majority,
    undersampledRows: 2 * minority,
    factor: (2 * majority) / (majority + minority),
    /** Materialised rows rise; independently observed units do not. */
    independentMinorityObservations: minority,
  };
}

/** The explicit all-pairs neighbour work of the teaching implementation. */
export function neighbourWork(minorityCount, dimensions, generated) {
  if ([minorityCount, dimensions, generated].some(value => !Number.isInteger(value) || value < 0)) {
    throw new RangeError('Counts and dimensions must be nonnegative whole numbers.');
  }
  return {
    distanceArithmetic: minorityCount ** 2 * dimensions,
    storedDistances: minorityCount ** 2,
    sortWork: minorityCount === 0 ? 0 : minorityCount ** 2 * Math.log2(minorityCount),
    generationArithmetic: generated * dimensions,
  };
}

/* =================================================== declared fixtures */

/** Every declared setup the manuscript states, in one place, so the lesson, the
 * investigations and the verifier all read the same numbers. */
export const fixtures = {
  /** §1: one thousand cases, twenty of them positive. */
  modelCounts: { tp: 14, fp: 18, fn: 6, tn: 962 },
  baselineCounts: { tp: 0, fp: 0, fn: 20, tn: 980 },
  /** §2: the three-record counterexample, in stable identifier order. */
  queue: [
    { id: 'A', score: 0.9, truth: 0 },
    { id: 'B', score: 0.8, truth: 1 },
    { id: 'C', score: 0.7, truth: 1 },
  ],
  queueStartThreshold: 0.7,
  queueTargetThreshold: 0.9,
  /** §2 and practice 2: four records, two of them tied at .8. */
  tiedQueue: [
    { id: 'P', score: 0.95, truth: 0 },
    { id: 'Q', score: 0.8, truth: 1 },
    { id: 'R', score: 0.8, truth: 0 },
    { id: 'S', score: 0.4, truth: 1 },
  ],
  /** §2: four distinct scores whose average precision is 5/6. */
  averagePrecisionQueue: [
    { id: 'A', score: 0.9, truth: 1 },
    { id: 'B', score: 0.8, truth: 0 },
    { id: 'C', score: 0.7, truth: 1 },
    { id: 'D', score: 0.6, truth: 0 },
  ],
  /** §2: one detector, two populations. */
  flowA: { population: 10000, prevalence: 0.01, tpr: 0.8, fpr: 0.01 },
  flowB: { population: 10000, prevalence: 0.001, tpr: 0.8, fpr: 0.01 },
  /** §3: the declared hypothetical costs and the case posterior. */
  costs: { costFP: 1, costFN: 12 },
  casePosterior: 0.1,
  contrastPosterior: 0.05,
  practiceCosts: { costFP: 2, costFN: 7, posterior: 0.2 },
  /** §4: two rows, one weighted step. */
  stepRows: [{ id: 'A', x: 0, y: 0, weight: 1 }, { id: 'B', x: 2, y: 1, weight: 3 }],
  stepRate: 0.4,
  stepPenalty: 0.01,
  /** §4: the weighted optimum and its contrast, null and practice. */
  weighted: { probability: 0.1, positiveWeight: 9, negativeWeight: 1 },
  weightedEqual: { probability: 0.1, positiveWeight: 1, negativeWeight: 1 },
  weightedDoubled: { probability: 0.1, positiveWeight: 18, negativeWeight: 2 },
  weightedPractice: { probability: 0.2, positiveWeight: 4, negativeWeight: 1 },
  /** §5: the small cloud, whose generated point lands on a majority point. */
  cloud: [
    { id: 'A', cls: 'minority', x: 0, y: 0 },
    { id: 'B', cls: 'minority', x: 2, y: 0 },
    { id: 'C', cls: 'minority', x: 0, y: 2.5 },
    { id: 'M', cls: 'majority', x: 1, y: 0 },
    { id: 'N', cls: 'majority', x: 2, y: 2 },
  ],
  cloudSetup: { anchorId: 'A', k: 1, neighbourRank: 0, fraction: 0.5, scaleX: 1, scaleY: 1 },
  /** §5: the majority point moved, which changes nothing about the output. */
  cloudNullMajority: { id: 'M', x: 1, y: 1.5 },
  /** §5: the y divisor that changes which neighbour is nearest. */
  cloudContrastScaleY: 10,
  /** Practice 5: a changed interpolation. */
  practiceInterpolation: { anchor: [1, 2], neighbour: [5, 4], fraction: 0.25, expected: [2, 2.5] },
  /** §8: ten thousand easy examples against ten difficult ones. */
  focal: {
    groups: [
      { name: 'easy', count: 10000, pt: 0.9 },
      { name: 'difficult', count: 10, pt: 0.2 },
    ],
    gamma: 2, alpha: 1,
  },
  /** §8 and practice 8: two prior corrections. */
  prior: { sampledPosterior: 0.8, sampledPrevalence: 0.5, deploymentPrevalence: 0.01 },
  priorPractice: { sampledPosterior: 0.5, sampledPrevalence: 0.2, deploymentPrevalence: 0.02 },
  /** §8 and practice 9: what balancing costs in rows. */
  expansion: { majority: 99, minority: 1 },
  expansionPractice: { majority: 960, minority: 40 },
  /** Practice 1: two tables with the same accuracy and opposite recall. */
  practiceEqualAccuracy: [
    { tp: 0, fp: 0, fn: 12, tn: 1188 },
    { tp: 12, fp: 12, fn: 0, tn: 1176 },
  ],
};
