/** Pure models for the Calibration & Conformal Prediction lesson.
 *
 * Three kinds of number live on this page and are never allowed to blur into
 * one another. Every function that returns a quantity a reader will see also
 * returns the `kind` tag naming which of the three it is, and the components
 * print that tag beside the number:
 *
 *   1. `population` — an exact constructed population quantity. The two
 *      forecast groups in section 1, the group mosaic in section 8 and the
 *      prevalence example are declared populations, so their rates are exact
 *      by construction and carry no sampling error at all.
 *   2. `calibration` — a quantity computed FROM a calibration sample: a fitted
 *      probability map, a rank, a threshold. These are learned objects. A
 *      threshold is a random variable, not a property of the world.
 *   3. `assessment` — a finite count of recorded outcomes on held-out rows.
 *      69 of 80. This is a measurement with a denominator, never a guarantee.
 *
 * This separation is the whole safety argument of the topic. Conformal
 * validity is *marginal* and rests on exchangeability; an empirical coverage
 * figure read as a conditional or future guarantee is the exact mistake the
 * lesson exists to prevent. So no function here returns a bare coverage number
 * without its denominator and its kind, and `coverageReport` refuses to
 * produce a rate without the count it came from.
 *
 * Two further conventions are load-bearing.
 *
 *   * A quantity with no value is `null` together with a stated reason. An
 *     empty bin has no observed fraction — that is not a fraction of zero. A
 *     one-class sample has no AUC. A conformal threshold beyond the resolution
 *     of the calibration set is `Infinity`, which means the whole label space
 *     and is not a large finite number.
 *   * `conformalRank` does exact decimal arithmetic on alpha. At n=24 and
 *     alpha=.44 — reachable through investigation 4's calibration pairs —
 *     `(n+1)*(1-alpha)` evaluates to 14.000000000000002 in ordinary floating
 *     point, so its ceiling is 15 where the exact rank is 14. The consequence
 *     is a threshold one order statistic too high: silent over-coverage, a
 *     number no plot would show to be wrong. The rank is the one number on
 *     this page that must not be approximated.
 *
 * Anything a figure draws geometrically is computed here, not in the
 * component, so scripts/verify-calibration-models.mjs can assert it: the
 * plotted reliability points and their count rail, the pooled-block
 * rectangles, the score rail and its threshold marker, every interval segment
 * and its width bar, the mosaic areas and the cumulative APS segments. A drawn
 * reliability curve is a mathematical claim.
 */

/* ========================================================== input limits */

/**
 * Every editable control, declared once.
 *
 * The components read these to configure their number fields and the verifier
 * reads the same object to check that every value the prose asks a learner to
 * type is reachable. Hard-coding a step in both places makes the check agree
 * with a copy of the control rather than with the control.
 */
export const controlSteps = {
  forecast: { decimals: 3, minimum: 0, maximum: 1, step: '0.001' },
  binEdge: { decimals: 3, minimum: 0, maximum: 1, step: '0.001' },
  /* Wide enough to reach every preset this lesson offers. At [-5, 5] the
     practice-2 fixture's score of 6 was loadable as a preset but not typeable,
     so a learner could reach a state the control refused to reproduce. */
  score: { decimals: 3, minimum: -10, maximum: 10, step: '0.001' },
  conformalScore: { decimals: 3, minimum: 0, maximum: 2, step: '0.001' },
  signedScore: { decimals: 3, minimum: -20, maximum: 20, step: '0.001' },
  alpha: { decimals: 3, minimum: 0.01, maximum: 0.5, step: '0.001' },
  residual: { decimals: 3, minimum: 0, maximum: 30, step: '0.001' },
  localScale: { decimals: 3, minimum: 0.1, maximum: 10, step: '0.001' },
  queryCentre: { decimals: 3, minimum: -30, maximum: 50, step: '0.001' },
  logit: { decimals: 3, minimum: -10, maximum: 10, step: '0.001' },
  temperature: { decimals: 3, minimum: 0.05, maximum: 20, step: '0.001' },
  weight: { decimals: 0, minimum: 1, maximum: 5, step: '1' },
};

export const limits = {
  cards: { minimum: 4, maximum: 30 },
  bins: { minimum: 1, maximum: 6 },
  pavRows: { minimum: 4, maximum: 24 },
  calibrationScores: { minimum: 3, maximum: 20 },
  calibrationPairs: { minimum: 3, maximum: 24 },
  queries: { minimum: 1, maximum: 6 },
  classes: { minimum: 2, maximum: 6 },
  alphaDecimals: 6,
  tolerance: 1e-12,
};

/** The three kinds of number, with the sentence each one is printed beside. */
export const quantityKinds = {
  population: {
    key: 'population',
    label: 'exact constructed population',
    note: 'A declared population. These rates are exact by construction and carry no sampling error.',
  },
  calibration: {
    key: 'calibration',
    label: 'computed from a calibration sample',
    note: 'Learned from calibration observations. A threshold or a fitted map is a random object that would '
      + 'differ on another calibration sample.',
  },
  assessment: {
    key: 'assessment',
    label: 'counted on a held-out assessment sample',
    note: 'A finite count of recorded outcomes with a stated denominator. It estimates a population quantity; '
      + 'it does not establish one.',
  },
};

export function checkFinite(value, name) {
  if (typeof value !== 'number' || !Number.isFinite(value)) {
    throw new RangeError(`${name} must be a finite number, not ${value}.`);
  }
  return value;
}

/** A probability outside [0, 1] is not a probability. Refuse it; never clamp. */
export function checkProbability(value, name) {
  checkFinite(value, name);
  if (value < 0 || value > 1) throw new RangeError(`${name} must lie between 0 and 1, not ${value}.`);
  return value;
}

export function checkOutcome(value, name) {
  if (value !== 0 && value !== 1) throw new RangeError(`${name} must be the outcome 0 or 1, not ${value}.`);
  return value;
}

/** A positive local scale. Zero or negative makes the normalised score undefined. */
export function checkScale(value, name) {
  checkFinite(value, name);
  if (value <= 0) throw new RangeError(`${name} must be strictly positive, not ${value}.`);
  return value;
}

/**
 * Exactly what decimal a JavaScript number denotes, as an integer pair.
 *
 * `conformalRank` cannot round. `(24+1)*(1-0.44)` is 14.000000000000002 in
 * double arithmetic, whose ceiling is 15 rather than the correct 14, so the
 * threshold becomes the fifteenth smallest calibration score instead of the
 * fourteenth. The set is then wider than the target asked for and every
 * downstream coverage figure is quietly optimistic — wrong in a way no plot
 * would show. So alpha is read as the decimal it prints as and the rank is
 * computed in integers.
 *
 * Over every (n, alpha) pair these controls admit this is the ONLY input at
 * which the float recipe disagrees, and across the verifier's wider n=1..200
 * sweep there are 26, at each of which the float rank is exactly one too
 * large and never exceeds n. Two earlier versions of this comment cited
 * n=9, alpha=.25 and alpha=.35 as failing cases; neither does — 0.25 and the
 * products involved are exactly representable — and the claim that the float
 * rank "asks for an order statistic the calibration set does not have" is
 * false at all 26. The real hazard is the quieter one described above.
 */
export function decimalParts(value, name) {
  checkFinite(value, name);
  const text = String(value);
  if (/e/i.test(text)) throw new RangeError(`${name} must be written as a plain decimal, not ${text}.`);
  const [whole, fraction = ''] = text.replace('-', '').split('.');
  if (fraction.length > limits.alphaDecimals) {
    throw new RangeError(`${name} carries ${fraction.length} decimal places; at most ${limits.alphaDecimals} are supported.`);
  }
  const denominator = 10 ** fraction.length;
  const magnitude = Number(`${whole}${fraction}`);
  return { numerator: value < 0 ? -magnitude : magnitude, denominator };
}

export function checkAlpha(alpha) {
  checkFinite(alpha, 'alpha');
  if (!(alpha > 0 && alpha < 1)) throw new RangeError(`alpha must lie strictly between 0 and 1, not ${alpha}.`);
  return alpha;
}

/* ================================== §1 · a probability conditions on a population */

/**
 * Two ways to read the same forecasts, from one declared population.
 *
 * Each group is `{ share, forecast, positiveRate }` — exact population
 * quantities, not counted outcomes. The class-1 reading pairs the forecast for
 * class 1 with that class's rate. The top-confidence reading pairs
 * max(p, 1−p) with whether the SELECTED class is right, which for a group
 * predicting class 0 is one minus its positive rate. They are different
 * conditioning questions and this function computes both rather than relabelling
 * one as the other.
 */
export function conditioningFork(groups) {
  if (!Array.isArray(groups) || groups.length < 1) {
    throw new RangeError('A conditioning fork needs at least one declared group.');
  }
  const rows = groups.map((group, index) => {
    const share = checkProbability(group.share, `group ${index + 1} share`);
    const forecast = checkProbability(group.forecast, `group ${index + 1} forecast`);
    const positiveRate = checkProbability(group.positiveRate, `group ${index + 1} positive rate`);
    const predictedClass = forecast >= 0.5 ? 1 : 0;
    const confidence = Math.max(forecast, 1 - forecast);
    const correctness = predictedClass === 1 ? positiveRate : 1 - positiveRate;
    return {
      name: group.name ?? `group ${index + 1}`,
      share, forecast, positiveRate, predictedClass, confidence, correctness,
      classOneGap: positiveRate - forecast,
      confidenceGap: correctness - confidence,
    };
  });
  const totalShare = rows.reduce((sum, row) => sum + row.share, 0);
  if (Math.abs(totalShare - 1) > 1e-12) {
    throw new RangeError(`The group shares must add to one; they add to ${totalShare}.`);
  }
  // A reliability point conditions on the reported value, pooling every group
  // with that value. Round only the grouping key to remove binary roundoff in
  // complementary decimal forecasts (for example .45 and .55).
  const pool = (key, outcome) => {
    const buckets = new Map();
    for (const row of rows) {
      if (row.share === 0) continue;
      const value = Number(row[key].toFixed(12));
      const bucket = buckets.get(value) ?? { x: value, total: 0, weight: 0, names: [] };
      bucket.total += row.share * row[outcome];
      bucket.weight += row.share;
      bucket.names.push(row.name);
      buckets.set(value, bucket);
    }
    return [...buckets.values()].map(({ total, weight, ...point }) => ({
      ...point, y: total / weight, weight,
    }));
  };
  const classOnePoints = pool('forecast', 'positiveRate');
  const confidencePoints = pool('confidence', 'correctness');
  const meanConfidence = rows.reduce((sum, row) => sum + row.share * row.confidence, 0);
  const meanCorrectness = rows.reduce((sum, row) => sum + row.share * row.correctness, 0);
  return {
    kind: quantityKinds.population.key,
    rows,
    classOnePoints,
    confidencePoints,
    meanConfidence,
    meanCorrectness,
    confidenceLooksCalibrated: confidencePoints.every(point => Math.abs(point.y - point.x) <= 1e-12),
    classOneLooksCalibrated: classOnePoints.every(point => Math.abs(point.y - point.x) <= 1e-12),
  };
}

/**
 * What a coarser score costs, in a proper score and in a decision.
 *
 * Both forecasts here are calibrated with respect to their own conditioning
 * information. The difference is resolution, and the cost comparison is the
 * one that makes that concrete.
 */
export function resolutionComparison({ shares, rates, releaseCost, quarantineCost }) {
  if (shares.length !== rates.length || !shares.length) {
    throw new RangeError('Resolution needs one share per group rate.');
  }
  shares.forEach((share, index) => checkProbability(share, `group ${index + 1} share`));
  rates.forEach((rate, index) => checkProbability(rate, `group ${index + 1} rate`));
  checkFinite(releaseCost, 'the cost of releasing a faulty item');
  checkFinite(quarantineCost, 'the cost of quarantining an item');
  const total = shares.reduce((sum, share) => sum + share, 0);
  if (Math.abs(total - 1) > 1e-12) throw new RangeError(`The group shares must add to one; they add to ${total}.`);
  const coarse = shares.reduce((sum, share, index) => sum + share * rates[index], 0);
  /* Expected squared error of forecasting p in a group whose rate is r:
     weight each possible outcome's squared error by how often it occurs. */
  const expectedBrier = (rate, forecast) => rate * (1 - forecast) ** 2 + (1 - rate) * forecast ** 2;
  const decideThreshold = releaseCost === 0 ? null : quarantineCost / releaseCost;
  const decide = forecast => (decideThreshold !== null && forecast > decideThreshold ? 'quarantine' : 'release');
  const groupCost = (rate, forecast) =>
    (decide(forecast) === 'quarantine' ? quarantineCost : releaseCost * rate);
  return {
    kind: quantityKinds.population.key,
    coarseForecast: coarse,
    decideThreshold,
    rows: shares.map((share, index) => ({
      share,
      rate: rates[index],
      coarseForecast: coarse,
      coarseAction: decide(coarse),
      coarseCost: groupCost(rates[index], coarse),
      coarseBrier: expectedBrier(rates[index], coarse),
      fullForecast: rates[index],
      fullAction: decide(rates[index]),
      fullCost: groupCost(rates[index], rates[index]),
      fullBrier: expectedBrier(rates[index], rates[index]),
    })),
    coarseBrier: shares.reduce((sum, share, index) => sum + share * expectedBrier(rates[index], coarse), 0),
    fullBrier: shares.reduce((sum, share, index) => sum + share * expectedBrier(rates[index], rates[index]), 0),
    coarseCost: shares.reduce((sum, share, index) => sum + share * groupCost(rates[index], coarse), 0),
    fullCost: shares.reduce((sum, share, index) => sum + share * groupCost(rates[index], rates[index]), 0),
  };
}

/* ============================================ §2 · reliability from observations */

/**
 * Which bin a forecast falls in.
 *
 * An internal boundary belongs to the bin on its RIGHT, and a forecast of
 * exactly 1 belongs to the last bin. Dropping p=1 loses observations, and they
 * are usually the consequential ones. This is the same rule the packet's
 * `reliability` applies through `searchsorted(..., side="right")`.
 */
export function binIndexOf(edges, probability) {
  let above = edges.findIndex(edge => edge > probability);
  if (above === -1) above = edges.length;
  return Math.min(above - 1, edges.length - 2);
}

export function checkEdges(edges) {
  if (!Array.isArray(edges) || edges.length < 2) throw new RangeError('Bin boundaries need at least two values.');
  edges.forEach((edge, index) => checkProbability(edge, `bin boundary ${index + 1}`));
  if (edges[0] !== 0 || edges[edges.length - 1] !== 1) {
    throw new RangeError('Bin boundaries must span the whole probability scale from 0 to 1.');
  }
  for (let index = 1; index < edges.length; index += 1) {
    if (edges[index] <= edges[index - 1]) {
      throw new RangeError(`Bin boundaries must strictly increase; ${edges[index - 1]} is followed by ${edges[index]}.`);
    }
  }
  if (edges.length - 1 > limits.bins.maximum) {
    throw new RangeError(`This editor holds at most ${limits.bins.maximum} bins; ${edges.length - 1} were given.`);
  }
  return edges;
}

/**
 * A reliability report: one row per bin, plus the binned ECE summary.
 *
 * An empty bin has `count` 0 and `meanP` and `fractionPositive` of `null`. It
 * contributes nothing to ECE and it is NOT a bin whose observed fraction is
 * zero. Keeping it as a row with no point is the difference between "no
 * evidence here" and "no positives here".
 */
export function reliability(predictions, outcomes, edges) {
  if (!Array.isArray(predictions) || !Array.isArray(outcomes) || !predictions.length
    || predictions.length !== outcomes.length) {
    throw new RangeError('Reliability needs equally sized, non-empty forecast and outcome lists.');
  }
  predictions.forEach((value, index) => checkProbability(value, `forecast ${index + 1}`));
  outcomes.forEach((value, index) => checkOutcome(value, `outcome ${index + 1}`));
  checkEdges(edges);
  const assigned = predictions.map(value => binIndexOf(edges, value));
  const bins = [];
  for (let index = 0; index < edges.length - 1; index += 1) {
    const members = assigned
      .map((bin, position) => (bin === index ? position : -1))
      .filter(position => position >= 0);
    if (!members.length) {
      bins.push({
        index, lower: edges[index], upper: edges[index + 1], count: 0, positive: 0,
        meanP: null, fractionPositive: null, gap: null, members: [],
      });
      continue;
    }
    const positive = members.reduce((sum, position) => sum + outcomes[position], 0);
    const meanP = members.reduce((sum, position) => sum + predictions[position], 0) / members.length;
    const fractionPositive = positive / members.length;
    bins.push({
      index, lower: edges[index], upper: edges[index + 1],
      count: members.length, positive, meanP, fractionPositive,
      gap: fractionPositive - meanP, members,
    });
  }
  const ece = bins.reduce((sum, bin) =>
    (bin.count ? sum + bin.count * Math.abs(bin.fractionPositive - bin.meanP) : sum), 0) / predictions.length;
  return {
    kind: quantityKinds.assessment.key,
    bins, ece, count: predictions.length,
    occupiedBins: bins.filter(bin => bin.count > 0).length,
    emptyBins: bins.filter(bin => bin.count === 0).length,
  };
}

/**
 * The other diagram: top-class confidence against whether the selected class
 * was right.
 *
 * Its indicator is computed explicitly here rather than reusing the positive
 * outcome, because the two agree only for observations that predict class 1.
 * Labelling this axis "fraction positive" would be the defect the destination
 * note reports.
 */
export function confidenceReliability(predictions, outcomes, edges) {
  if (!Array.isArray(predictions) || predictions.length !== outcomes.length || !predictions.length) {
    throw new RangeError('A confidence diagram needs equally sized, non-empty forecast and outcome lists.');
  }
  predictions.forEach((value, index) => checkProbability(value, `forecast ${index + 1}`));
  outcomes.forEach((value, index) => checkOutcome(value, `outcome ${index + 1}`));
  const predictedClass = predictions.map(value => (value >= 0.5 ? 1 : 0));
  const confidence = predictions.map(value => Math.max(value, 1 - value));
  const correct = predictedClass.map((chosen, index) => (chosen === outcomes[index] ? 1 : 0));
  const report = reliability(confidence, correct, edges);
  return {
    ...report,
    axisLabel: 'fraction of selected classes that were correct',
    confidence, correct, predictedClass,
  };
}

export function brierLoss(predictions, outcomes) {
  if (!predictions.length || predictions.length !== outcomes.length) {
    throw new RangeError('Brier loss needs equally sized, non-empty forecast and outcome lists.');
  }
  predictions.forEach((value, index) => checkProbability(value, `forecast ${index + 1}`));
  outcomes.forEach((value, index) => checkOutcome(value, `outcome ${index + 1}`));
  return predictions.reduce((sum, value, index) => sum + (value - outcomes[index]) ** 2, 0) / predictions.length;
}

/**
 * Rank-based AUC with average ranks for ties, so a map that creates ties is
 * scored the way a library scores it.
 *
 * `null` when the sample carries only one outcome class: AUC compares
 * positives with negatives and there is no comparison to make. Reporting .5,
 * or 0, would be inventing a number.
 */
export function rocAuc(predictions, outcomes) {
  if (!predictions.length || predictions.length !== outcomes.length) {
    throw new RangeError('AUC needs equally sized, non-empty forecast and outcome lists.');
  }
  predictions.forEach((value, index) => checkFinite(value, `score ${index + 1}`));
  outcomes.forEach((value, index) => checkOutcome(value, `outcome ${index + 1}`));
  const positives = outcomes.reduce((sum, value) => sum + value, 0);
  const negatives = outcomes.length - positives;
  if (positives === 0 || negatives === 0) {
    return { value: null, because: 'every observation has the same outcome, so there is no positive-negative pair to order' };
  }
  const order = predictions.map((value, index) => index).sort((a, b) => predictions[a] - predictions[b]);
  const ranks = new Array(predictions.length);
  let position = 0;
  while (position < order.length) {
    let end = position;
    while (end + 1 < order.length && predictions[order[end + 1]] === predictions[order[position]]) end += 1;
    const average = (position + end) / 2 + 1;
    for (let index = position; index <= end; index += 1) ranks[order[index]] = average;
    position = end + 1;
  }
  const positiveRankSum = outcomes.reduce((sum, value, index) => (value === 1 ? sum + ranks[index] : sum), 0);
  return {
    value: (positiveRankSum - (positives * (positives + 1)) / 2) / (positives * negatives),
    because: null, positives, negatives,
  };
}

/* =============================================== §3 · fitting a probability map */

/**
 * Pool-adjacent-violators, with equal scores grouped FIRST and every block
 * carrying its own weight.
 *
 * Equal scores must receive equal fitted values, so they become one block
 * before any merging starts. A merge replaces two blocks by their
 * count-weighted mean, not the average of their means: for the tied fixture
 * that is 1/3 rather than 1/4, and averaging the means is the classic wrong
 * answer this returns the trace for.
 *
 * The trace records every merge in order, so the investigation can step
 * through the same algorithm the figure draws.
 */
export function fitPav(scores, labels, weights = null) {
  if (!Array.isArray(scores) || !Array.isArray(labels) || !scores.length || scores.length !== labels.length) {
    throw new RangeError('The monotone fit needs equally sized, non-empty score and label lists.');
  }
  scores.forEach((value, index) => checkFinite(value, `score ${index + 1}`));
  labels.forEach((value, index) => checkOutcome(value, `label ${index + 1}`));
  const counts = weights ?? scores.map(() => 1);
  if (counts.length !== scores.length) throw new RangeError('One weight per observation, or none at all.');
  counts.forEach((value, index) => {
    if (!Number.isInteger(value) || value < 1) throw new RangeError(`weight ${index + 1} must be a positive whole number, not ${value}.`);
  });
  const order = scores.map((_value, index) => index).sort((a, b) => (scores[a] - scores[b]) || (a - b));
  const knots = [];
  const groupTotal = [];
  const groupWeight = [];
  const groupMembers = [];
  order.forEach(index => {
    const last = knots.length - 1;
    if (last >= 0 && knots[last] === scores[index]) {
      groupTotal[last] += labels[index] * counts[index];
      groupWeight[last] += counts[index];
      groupMembers[last].push(index);
      return;
    }
    knots.push(scores[index]);
    groupTotal.push(labels[index] * counts[index]);
    groupWeight.push(counts[index]);
    groupMembers.push([index]);
  });
  const blocks = [];
  const merges = [];
  groupTotal.forEach((total, position) => {
    blocks.push({ start: position, end: position, total, weight: groupWeight[position] });
    while (blocks.length >= 2) {
      const right = blocks[blocks.length - 1];
      const left = blocks[blocks.length - 2];
      if (left.total / left.weight <= right.total / right.weight) break;
      blocks.pop(); blocks.pop();
      const merged = {
        start: left.start, end: right.end, total: left.total + right.total, weight: left.weight + right.weight,
      };
      merges.push({
        left: { ...left, mean: left.total / left.weight },
        right: { ...right, mean: right.total / right.weight },
        merged: { ...merged, mean: merged.total / merged.weight },
      });
      blocks.push(merged);
    }
  });
  const fitted = new Array(knots.length);
  blocks.forEach(block => {
    for (let index = block.start; index <= block.end; index += 1) fitted[index] = block.total / block.weight;
  });
  return {
    kind: quantityKinds.calibration.key,
    knots,
    fitted,
    weights: groupWeight,
    totals: groupTotal,
    members: groupMembers,
    blocks: blocks.map(block => ({ ...block, mean: block.total / block.weight })),
    merges,
  };
}

/**
 * One step of the same algorithm, for the investigation that asks a learner to
 * choose the next merge.
 *
 * `required` is true only for the pair the deterministic left-to-right stack
 * would merge next. A learner who selects a different adjacent pair gets an
 * explanation, not a silently altered algorithm: choosing a non-violating pair
 * would change the answer, so it is refused rather than applied.
 */
export function nextRequiredMerge(blocks) {
  for (let index = 0; index + 1 < blocks.length; index += 1) {
    const left = blocks[index];
    const right = blocks[index + 1];
    if (left.total / left.weight > right.total / right.weight) {
      return {
        index,
        left: { ...left, mean: left.total / left.weight },
        right: { ...right, mean: right.total / right.weight },
        mergedMean: (left.total + right.total) / (left.weight + right.weight),
        required: true,
      };
    }
  }
  return null;
}

/**
 * The same fit as a sequence of stages a learner can step through.
 *
 * Start with every distinct score as its own block, then repeatedly merge the
 * LEFTMOST adjacent pair whose means decrease. That is a different schedule
 * from the left-to-right stack `fitPav` uses, and pool-adjacent-violators is
 * confluent, so the two must reach the same fitted values. The verifier asserts
 * that agreement over every fixture rather than on the lesson's example.
 */
export function pavStages(scores, labels, weights = null) {
  const grouped = fitPav(scores, labels, weights);
  let blocks = grouped.knots.map((_knot, index) => ({
    start: index, end: index, total: grouped.totals[index], weight: grouped.weights[index],
  }));
  const stages = [blocks.map(block => ({ ...block, mean: block.total / block.weight }))];
  const merges = [];
  for (let guard = 0; guard < grouped.knots.length; guard += 1) {
    const next = nextRequiredMerge(blocks);
    if (!next) break;
    const left = blocks[next.index];
    const right = blocks[next.index + 1];
    const merged = {
      start: left.start, end: right.end, total: left.total + right.total, weight: left.weight + right.weight,
    };
    merges.push({
      index: next.index,
      left: { ...left, mean: left.total / left.weight },
      right: { ...right, mean: right.total / right.weight },
      merged: { ...merged, mean: merged.total / merged.weight },
    });
    blocks = [...blocks.slice(0, next.index), merged, ...blocks.slice(next.index + 2)];
    stages.push(blocks.map(block => ({ ...block, mean: block.total / block.weight })));
  }
  const fitted = new Array(grouped.knots.length);
  blocks.forEach(block => {
    for (let index = block.start; index <= block.end; index += 1) fitted[index] = block.total / block.weight;
  });
  return {
    knots: grouped.knots, weights: grouped.weights, totals: grouped.totals, members: grouped.members,
    stages, merges, fitted, finalBlocks: stages[stages.length - 1],
  };
}

/**
 * Predict between and beyond the fitted knots, by linear interpolation with
 * endpoint clipping.
 *
 * This is the contract of `IsotonicRegression(out_of_bounds="clip")`, which
 * our offline program uses. It is stated as its own function because the
 * fitted values are defined only AT the knots: how a library fills the gaps is
 * a separate choice, and the result need not be a staircase.
 */
export function isotonicPredict(knots, fitted, x) {
  if (!knots.length || knots.length !== fitted.length) {
    throw new RangeError('Interpolation needs one fitted value per knot.');
  }
  checkFinite(x, 'the score to predict at');
  if (x <= knots[0]) return fitted[0];
  if (x >= knots[knots.length - 1]) return fitted[fitted.length - 1];
  const upper = knots.findIndex(knot => knot >= x);
  const lower = upper - 1;
  const span = knots[upper] - knots[lower];
  if (span === 0) return fitted[upper];
  return fitted[lower] + ((x - knots[lower]) / span) * (fitted[upper] - fitted[lower]);
}

export const sigmoid = z => (z >= 0 ? 1 / (1 + Math.exp(-z)) : Math.exp(z) / (1 + Math.exp(z)));

/** log(1 + e^z), evaluated without overflowing for large positive z. */
export const softplus = z => (z > 0 ? z + Math.log1p(Math.exp(-z)) : Math.log1p(Math.exp(z)));

export function sigmoidProbability(a, b, score) {
  checkFinite(a, 'the slope a');
  checkFinite(b, 'the offset b');
  checkFinite(score, 'the score');
  return sigmoid(a * score + b);
}

/**
 * Platt's smoothed targets. Positives receive (N+ + 1)/(N+ + 2) and negatives
 * 1/(N− + 2), which keeps a separable calibration sample from driving the
 * logits to infinity. It is part of the stated fit, not a promise of exact
 * calibration.
 */
export function plattTargets(labels) {
  const positives = labels.reduce((sum, value) => sum + value, 0);
  const negatives = labels.length - positives;
  return labels.map(value => (value === 1 ? (positives + 1) / (positives + 2) : 1 / (negatives + 2)));
}

/**
 * Fit the two-parameter sigmoid by minimising average log loss.
 *
 * Damped Newton with a backtracking line search. Plain Newton oscillates and
 * diverges on this lesson's own eight-score fixture — it reaches a = 88 at
 * iteration four and then overflows — so the step is halved until the
 * objective actually decreases. A run that does not reach a small gradient
 * returns `converged: false` with the residual, rather than a number presented
 * as a fit.
 *
 * The frozen packet fitted the same objective with SciPy's BFGS. That is the
 * independent second route, and the verifier checks the two agree.
 */
export function fitSigmoid(scores, labels, { smoothing = true, maximumIterations = 200 } = {}) {
  if (!scores.length || scores.length !== labels.length) {
    throw new RangeError('A sigmoid fit needs equally sized, non-empty score and label lists.');
  }
  scores.forEach((value, index) => checkFinite(value, `score ${index + 1}`));
  labels.forEach((value, index) => checkOutcome(value, `label ${index + 1}`));
  const present = new Set(labels);
  if (!(present.has(0) && present.has(1))) {
    return {
      kind: quantityKinds.calibration.key,
      a: null, b: null, converged: false, iterations: 0, objective: null, probabilities: null,
      because: 'the sample carries only one outcome class. This lab requires both classes as an evidence policy; the sample supplies no evidence '
        + 'about the missing class. A smoothed constant fit can exist, but this lab declines to fit it',
    };
  }
  const targets = smoothing ? plattTargets(labels) : labels.slice();
  const n = scores.length;
  const objective = (a, b) => scores.reduce((sum, score, index) => {
    const z = a * score + b;
    return sum + softplus(z) - targets[index] * z;
  }, 0) / n;
  let a = 0;
  let b = 0;
  let iterations = 0;
  let gradientNorm = Infinity;
  let stalled = false;
  for (let step = 0; step < maximumIterations; step += 1) {
    let g0 = 0; let g1 = 0; let h00 = 0; let h01 = 0; let h11 = 0;
    for (let index = 0; index < n; index += 1) {
      const z = a * scores[index] + b;
      const q = sigmoid(z);
      const residual = q - targets[index];
      const curvature = q * (1 - q);
      g0 += residual * scores[index]; g1 += residual;
      h00 += curvature * scores[index] * scores[index];
      h01 += curvature * scores[index];
      h11 += curvature;
    }
    g0 /= n; g1 /= n; h00 /= n; h01 /= n; h11 /= n;
    gradientNorm = Math.max(Math.abs(g0), Math.abs(g1));
    if (gradientNorm < 1e-14) { iterations = step; break; }
    const determinant = h00 * h11 - h01 * h01;
    // A flat curvature makes the Newton direction meaningless; fall back to the
    // gradient direction, which the line search will scale.
    const [da, db] = determinant > 1e-18
      ? [(h11 * g0 - h01 * g1) / determinant, (h00 * g1 - h01 * g0) / determinant]
      : [g0, g1];
    const base = objective(a, b);
    let scale = 1;
    let moved = false;
    for (let attempt = 0; attempt < 60; attempt += 1) {
      const nextA = a - scale * da;
      const nextB = b - scale * db;
      if (Number.isFinite(nextA) && Number.isFinite(nextB) && objective(nextA, nextB) < base) {
        a = nextA; b = nextB; moved = true; break;
      }
      scale /= 2;
    }
    iterations = step + 1;
    if (!moved) { stalled = true; break; }
  }
  /* Why 1e-7 and not something tighter.
   *
   * Near the optimum this objective is flat to second order: a step of size d
   * changes it by about d^2 H / 2, so once the gradient reaches ~1e-8 the
   * improvement a step would buy is ~1e-16 — the relative precision of a
   * double. The line search then cannot find a strictly decreasing step and
   * stops, with a gradient that is as small as the arithmetic allows. Demanding
   * 1e-10 would report a perfectly good fit as a failure on some labellings,
   * which is worse than useless: it would hide the honest curve from a learner
   * whose only mistake was editing a label. `stalled` records that the search
   * ran out of representable progress rather than reaching the tolerance.
   */
  const converged = gradientNorm < 1e-7;
  return {
    kind: quantityKinds.calibration.key,
    a, b, converged, iterations, gradientNorm, stalled,
    objective: objective(a, b),
    targets,
    probabilities: scores.map(score => sigmoid(a * score + b)),
    because: converged ? null
      : `the search stopped with a gradient of ${gradientNorm}, so these parameters are not a fitted optimum`,
  };
}

/**
 * Temperature scaling. One positive T divides every logit before the softmax.
 *
 * Dividing all logits by the same positive number preserves their within-example
 * order and their ties, so the top-1 class and its accuracy do not move under
 * the same tie rule. Probability thresholds and cost-sensitive decisions still can.
 */
export function softmaxAt(logits, temperature) {
  if (!Array.isArray(logits) || logits.length < 2) throw new RangeError('Softmax needs at least two logits.');
  logits.forEach((value, index) => checkFinite(value, `logit ${index + 1}`));
  checkFinite(temperature, 'the temperature');
  if (temperature <= 0) throw new RangeError(`The temperature must be strictly positive, not ${temperature}.`);
  const scaled = logits.map(value => value / temperature);
  const largest = Math.max(...scaled);
  const exponentials = scaled.map(value => Math.exp(value - largest));
  const total = exponentials.reduce((sum, value) => sum + value, 0);
  return exponentials.map(value => value / total);
}

export function entropyNats(probabilities) {
  return -probabilities.reduce((sum, value) => (value > 0 ? sum + value * Math.log(value) : sum), 0);
}

/** Which class wins, and whether the win is shared. */
export function leadingClass(probabilities) {
  const best = Math.max(...probabilities);
  const winners = probabilities
    .map((value, index) => (value === best ? index : -1)).filter(index => index >= 0);
  return { index: winners[0], shared: winners.length > 1, winners, value: best };
}

/* ================================================== §5 · the exact finite rank */

/**
 * k = ceil((n + 1)(1 − alpha)), in integers.
 *
 * The (n + 1) is the whole argument: it reserves a rank for the unseen next
 * observation. `Math.ceil((n + 1) * (1 - alpha))` in ordinary floating point
 * returns the wrong integer for several alphas a learner can type, and a rank
 * one too large asks for an order statistic the calibration set does not have.
 */
export function conformalRank(n, alpha) {
  if (!Number.isInteger(n) || n < 1) throw new RangeError(`At least one calibration score is required, not ${n}.`);
  checkAlpha(alpha);
  const { numerator, denominator } = decimalParts(alpha, 'alpha');
  const scaled = (n + 1) * (denominator - numerator);
  return Math.floor((scaled + denominator - 1) / denominator);
}

/**
 * The kth smallest calibration score, or infinity.
 *
 * `Infinity` means the whole label space, and it is the correct answer when the
 * requested coverage is beyond the resolution of this calibration set. Clipping
 * k to n would quietly return a finite threshold and discard the protection the
 * learner asked for.
 */
export function conformalThreshold(scores, alpha) {
  if (!Array.isArray(scores) || !scores.length) throw new RangeError('A threshold needs at least one calibration score.');
  scores.forEach((value, index) => checkFinite(value, `calibration score ${index + 1}`));
  const k = conformalRank(scores.length, alpha);
  if (k > scores.length) {
    return {
      kind: quantityKinds.calibration.key,
      k, n: scores.length, q: Infinity, finite: false,
      because: `rank ${k} is beyond the ${scores.length} calibration scores available, so every candidate answer is included`,
    };
  }
  const sorted = [...scores].sort((a, b) => a - b);
  return {
    kind: quantityKinds.calibration.key,
    k, n: scores.length, q: sorted[k - 1], finite: true, sorted, because: null,
  };
}

/**
 * Which classes survive the weak comparison s <= q.
 *
 * Equality is included, and the score is computed the same way here as it was
 * for calibration. Rearranging `1 - p <= q` into `p >= 1 - q` is algebraically
 * the same and numerically is not: the second subtraction is inexact, and the
 * lesson's own constant baseline loses its tied class by one rounding unit.
 */
export function classSets(probabilityVector, q) {
  probabilityVector.forEach((value, index) => checkProbability(value, `class ${index + 1} probability`));
  const scores = probabilityVector.map(value => 1 - value);
  const included = scores.map(score => score <= q);
  return {
    scores, included,
    size: included.filter(Boolean).length,
    empty: included.every(value => !value),
    singleton: included.filter(Boolean).length === 1,
  };
}

/**
 * Hold each position out once, threshold on the rest, and record whether it was
 * covered.
 *
 * This is an exact combinatorial count over a fixed multiset — every placement
 * of the held-out card is equally likely, which is what exchangeability means
 * here. It is not a Monte Carlo simulation and it is not a statement about a
 * test sample.
 */
export function rankRotation(scores, alpha) {
  if (scores.length < 2) throw new RangeError('A rotation needs at least two scores.');
  const rows = scores.map((held, index) => {
    const calibration = scores.filter((_value, position) => position !== index);
    const threshold = conformalThreshold(calibration, alpha);
    return {
      index, held, q: threshold.q, k: threshold.k, n: threshold.n,
      covered: held <= threshold.q,
    };
  });
  const covered = rows.filter(row => row.covered).length;
  return {
    kind: quantityKinds.population.key,
    rows, covered, total: rows.length,
    /* Exact, because every rotation was enumerated. Still a statement about
       this multiset, not about a population. */
    exact: true,
    targetRank: conformalRank(scores.length - 1, alpha),
  };
}

/**
 * Adaptive prediction sets: the cumulative probability mass up to a candidate
 * class, after sorting classes by descending probability.
 *
 * The score of a class therefore depends on the competition among labels, not
 * only on that class's own probability.
 */
export function apsScores(probabilities) {
  probabilities.forEach((value, index) => checkProbability(value, `class ${index + 1} probability`));
  const order = probabilities.map((_value, index) => index)
    .sort((a, b) => (probabilities[b] - probabilities[a]) || (a - b));
  const result = new Array(probabilities.length);
  const steps = [];
  let running = 0;
  order.forEach((index, rank) => {
    running += probabilities[index];
    result[index] = running;
    steps.push({ rank, classIndex: index, probability: probabilities[index], cumulative: running });
  });
  return { order, scores: result, steps };
}

/** The direct sublevel set, and the boundary-expanded variant, kept apart. */
export function apsSets(probabilities, q) {
  const { scores, order, steps } = apsScores(probabilities);
  const direct = scores.map(score => score <= q);
  const expanded = direct.slice();
  if (!direct.some(Boolean) || direct.filter(Boolean).length < probabilities.length) {
    const firstExcluded = order.find(index => !direct[index]);
    if (firstExcluded !== undefined) expanded[firstExcluded] = true;
  }
  return {
    order, scores, steps, direct, expanded,
    directSize: direct.filter(Boolean).length,
    expandedSize: expanded.filter(Boolean).length,
    strictSuperset: expanded.filter(Boolean).length > direct.filter(Boolean).length,
  };
}

/* ======================================== §6 · residuals, scales and intervals */

export function absoluteInterval(prediction, q) {
  checkFinite(prediction, 'the point prediction');
  if (!Number.isFinite(q)) return { lower: -Infinity, upper: Infinity, width: Infinity, unbounded: true };
  return { lower: prediction - q, upper: prediction + q, width: 2 * q, unbounded: false };
}

export function normalizedInterval(prediction, localScale, q) {
  checkFinite(prediction, 'the point prediction');
  checkScale(localScale, 'the local scale');
  if (!Number.isFinite(q)) return { lower: -Infinity, upper: Infinity, width: Infinity, unbounded: true };
  const halfWidth = q * localScale;
  return {
    lower: prediction - halfWidth, upper: prediction + halfWidth,
    width: 2 * halfWidth, halfWidth, unbounded: false,
  };
}

/**
 * The conformalised quantile-regression score.
 *
 * Positive when the response falls outside the initial interval, by the
 * distance beyond the nearer violated endpoint; non-positive when it falls
 * inside. A negative threshold therefore shrinks an over-wide initial interval.
 */
export function cqrScore(lower, upper, y) {
  checkFinite(lower, 'the lower quantile estimate');
  checkFinite(upper, 'the upper quantile estimate');
  checkFinite(y, 'the response');
  if (lower > upper) {
    throw new RangeError(`The quantile estimates cross: ${lower} is above ${upper}. Rearrange them before scoring.`);
  }
  return Math.max(lower - y, y - upper);
}

/**
 * The conformalised interval. Crossed endpoints are an empty set, which is a
 * legitimate outcome of this construction and not a negative width.
 */
export function cqrInterval(lower, upper, q) {
  checkFinite(lower, 'the lower quantile estimate');
  checkFinite(upper, 'the upper quantile estimate');
  if (!Number.isFinite(q)) return { lower: -Infinity, upper: Infinity, width: Infinity, empty: false, unbounded: true };
  const adjustedLower = lower - q;
  const adjustedUpper = upper + q;
  const empty = adjustedLower > adjustedUpper;
  return {
    lower: adjustedLower, upper: adjustedUpper,
    width: empty ? 0 : adjustedUpper - adjustedLower,
    empty, unbounded: false,
  };
}

/** A fixed pointwise rearrangement, applied identically to calibration and test. */
export function rearrangeQuantiles(lower, upper) {
  return { lower: Math.min(lower, upper), upper: Math.max(lower, upper), crossed: lower > upper };
}

/* ================================== §8 · what a coverage number does and does not say */

/**
 * A coverage report that cannot be quoted without its denominator.
 *
 * `rate` is only ever accompanied by `covered`, `total` and the kind tag. The
 * whole failure mode this lesson guards against is a reader lifting "90%" out
 * of a table and carrying it away as a guarantee.
 */
export function coverageReport(covered, total, kind) {
  if (!Number.isInteger(covered) || !Number.isInteger(total) || total < 1 || covered < 0 || covered > total) {
    throw new RangeError(`A coverage count needs 0 <= covered <= total with total >= 1; got ${covered} of ${total}.`);
  }
  if (!quantityKinds[kind]) throw new RangeError(`${kind} is not one of the three declared quantity kinds.`);
  return { kind, covered, total, rate: covered / total, text: `${covered} of ${total}` };
}

/** Marginal coverage as the share-weighted average of group coverages. */
export function mosaicMarginal(groups) {
  if (!groups.length) throw new RangeError('A mosaic needs at least one group.');
  groups.forEach((group, index) => {
    checkProbability(group.share, `group ${index + 1} share`);
    checkProbability(group.coverage, `group ${index + 1} coverage`);
  });
  const total = groups.reduce((sum, group) => sum + group.share, 0);
  if (Math.abs(total - 1) > 1e-12) throw new RangeError(`The group shares must add to one; they add to ${total}.`);
  return {
    kind: quantityKinds.population.key,
    groups: groups.map(group => ({ ...group, contribution: group.share * group.coverage })),
    marginal: groups.reduce((sum, group) => sum + group.share * group.coverage, 0),
    worstGroup: groups.reduce((worst, group) => (group.coverage < worst.coverage ? group : worst)),
  };
}

/** The posterior meaning of a positive result changes with prevalence alone. */
export function positivePredictiveValue(sensitivity, falsePositiveRate, prevalence) {
  checkProbability(sensitivity, 'the sensitivity');
  checkProbability(falsePositiveRate, 'the false-positive rate');
  checkProbability(prevalence, 'the prevalence');
  const positives = sensitivity * prevalence;
  const denominator = positives + falsePositiveRate * (1 - prevalence);
  if (denominator === 0) {
    return { value: null, because: 'no one in this population tests positive, so a positive result has no posterior' };
  }
  return { value: positives / denominator, because: null };
}

/**
 * The mean of the Beta(k, n+1−k) distribution of the coverage a fixed
 * threshold achieves, under iid continuous scores.
 *
 * Stated as its own function with its conditions attached, because it does not
 * describe tied isotonic scores or our finite-corpus sampling scheme.
 */
export function orderStatisticCoverageMean(k, n) {
  if (!Number.isInteger(k) || !Number.isInteger(n) || k < 1 || k > n) {
    throw new RangeError(`The Beta mean needs 1 <= k <= n; got k=${k}, n=${n}.`);
  }
  return {
    mean: k / (n + 1), alpha: k, beta: n + 1 - k,
    conditions: 'iid continuous scores with no ties and a model fitted independently of this calibration sample',
  };
}

/* ============================================================== grading rules */

export const unchangedTolerance = 1e-12;

/**
 * Did the graded quantity rise, fall, or stay within tolerance?
 *
 * Four answers, not two. `null` means the quantity has no value — an empty
 * bin's fraction, an AUC on one outcome class — which is neither zero nor a
 * small number. The verdict says "unchanged" exactly when the displayed output
 * did not move; that equivalence is exercised at zero, at exact ties and at
 * identical inputs rather than asserted.
 */
export function changeDirection(after, before) {
  if (after === null || after === undefined) return 'undefined';
  if (before === null || before === undefined) {
    checkFinite(after, 'the new value');
    return 'defined';
  }
  checkFinite(after, 'the new value');
  checkFinite(before, 'the previous value');
  const gap = after - before;
  const scale = Math.abs(before) > 1 ? Math.abs(before) : 1;
  if (Math.abs(gap) <= unchangedTolerance * scale) return 'unchanged';
  return gap > 0 ? 'higher' : 'lower';
}

/**
 * How a graded quantity is printed beside its verdict.
 *
 * The verdict and the number a learner can see have to agree: being told
 * "unchanged" while two different numbers are on screen, or "higher" while the
 * same number is, is the same defect wearing two hats. Both the components and
 * the verifier call THIS function, so the check tests the real rule rather than
 * a copy of it, and the verifier sweeps the whole enterable grid asserting
 * `changeDirection(a, b) === 'unchanged'` exactly when these two strings match.
 */
export const gradedDisplayDigits = 12;

export function gradedText(value) {
  if (value === null || value === undefined) return 'no value';
  if (!Number.isFinite(value)) return value > 0 ? 'unbounded' : 'unbounded below';
  return value.toFixed(gradedDisplayDigits).replace('-', '−');
}

/**
 * The same rule for a threshold, which may legitimately be infinite.
 *
 * Two infinite thresholds are unchanged; a move between finite and infinite is
 * a direction, not an undefined value. Passing `Infinity` to `changeDirection`
 * would throw, and clamping it to a large number would make "the threshold rose
 * to infinity" indistinguishable from "the threshold rose".
 */
export function thresholdDirection(after, before) {
  if (after === Infinity && before === Infinity) return 'unchanged';
  if (after === Infinity) return 'higher';
  if (before === Infinity) return 'lower';
  return changeDirection(after, before);
}

/* ================================================================== geometry */

/** A linear map from a declared data domain to a drawing range. */
export function linearScale({ domain, range }) {
  const [d0, d1] = domain;
  const [r0, r1] = range;
  if (d1 === d0) throw new RangeError('A scale needs a domain with two different endpoints.');
  const map = value => r0 + ((value - d0) / (d1 - d0)) * (r1 - r0);
  map.domain = domain;
  map.range = range;
  map.invert = value => d0 + ((value - r0) / (r1 - r0)) * (d1 - d0);
  return map;
}

/**
 * A base-ten logarithmic map, for a quantity whose interesting region a linear
 * axis would flatten.
 *
 * The airfoil frequencies run from 200 Hz to 20,000 Hz. On a linear axis every
 * observation below 2,000 Hz — the slice whose coverage actually differs —
 * collapses into the first twentieth of the drawing.
 */
export function logScale({ domain, range }) {
  const [d0, d1] = domain;
  if (!(d0 > 0) || !(d1 > 0)) throw new RangeError('A logarithmic scale needs a strictly positive domain.');
  const l0 = Math.log10(d0);
  const l1 = Math.log10(d1);
  const [r0, r1] = range;
  const map = value => r0 + ((Math.log10(value) - l0) / (l1 - l0)) * (r1 - r0);
  map.domain = domain;
  map.range = range;
  map.kind = 'log10';
  return map;
}

/** The plotting box every square diagram on this page shares. */
export const plotBox = { width: 300, height: 240, left: 44, right: 12, top: 14, bottom: 42 };

/**
 * A scatter of points, with the axis type declared rather than assumed.
 *
 * `xLog` chooses the logarithmic map. The figure prints which it used, because
 * a reader who takes a log axis for a linear one misreads every distance on it.
 */
export function scatterGeometry(points, options = {}) {
  const box = { ...plotBox, ...options.box };
  const xValues = points.map(point => point.x);
  const yValues = points.map(point => point.y);
  const xDomain = options.xDomain ?? [Math.min(...xValues), Math.max(...xValues)];
  const yDomain = options.yDomain ?? [Math.min(...yValues, 0), Math.max(...yValues)];
  const x = (options.xLog ? logScale : linearScale)({
    domain: xDomain, range: [box.left, box.width - box.right],
  });
  const y = linearScale({ domain: yDomain, range: [box.height - box.bottom, box.top] });
  return {
    box, axisKind: options.xLog ? 'log10' : 'linear',
    points: points.map(point => ({ ...point, cx: x(point.x), cy: y(point.y) })),
    scales: { x, y },
    xDomain, yDomain,
  };
}

/**
 * Adapt a recorded reliability report — the snake_case shape the offline program
 * wrote — into the shape `reliabilityGeometry` reads.
 *
 * Written as an adapter rather than by teaching the geometry two shapes, so
 * there is exactly one place where a renamed field would fail.
 */
export function reliabilityFromRecord(record, edges) {
  const bins = record.bins.map((row, index) => ({
    index: row.bin ?? index,
    lower: edges[index], upper: edges[index + 1],
    count: row.count,
    positive: row.positive ?? 0,
    meanP: row.mean_p ?? null,
    fractionPositive: row.fraction_positive ?? null,
    gap: row.count ? row.fraction_positive - row.mean_p : null,
    members: row.source_indices ?? [],
  }));
  return {
    kind: quantityKinds.assessment.key,
    bins, ece: record.ece, count: record.count,
    occupiedBins: bins.filter(bin => bin.count > 0).length,
    emptyBins: bins.filter(bin => bin.count === 0).length,
  };
}

/**
 * Where the reliability dots, the diagonal and the count rail actually go.
 *
 * A drawn reliability curve is a mathematical claim, so the coordinates are
 * computed here and asserted. An empty bin contributes NO point — it is
 * returned in `emptyBins` so the figure can say the bin exists without drawing
 * a dot at a fraction it does not have.
 */
/**
 * The reliability plot is taller than the shared box because it carries a count
 * rail BELOW its axis.
 *
 * The rail first shared the shared box, and its bars were drawn upward from 20
 * units under the axis — straight through the tick-label row, which sits 14
 * under it. Every number was right and the bars covered the labels. Nothing
 * offline could see it: the shared layout inspector reads `<line>` elements,
 * and the curve sampler reads paths; neither looks at a `<rect>` against a
 * `<text>`. So the band is now part of the returned geometry and is asserted
 * to clear the labels.
 */
export const reliabilityBox = { width: 300, height: 280, left: 44, right: 12, top: 14, bottom: 82 };

export function reliabilityGeometry(report, options = {}) {
  const box = { ...reliabilityBox, ...options.box };
  const railHeight = options.railHeight ?? 26;
  /* Where the bars stand, and where their tops may reach. Both are below the
     tick-label row at `box.height - box.bottom + 14`. */
  const railBaseline = box.height - box.bottom + 48;
  const tickLabelY = box.height - box.bottom + 14;
  const emptyMarkerHeight = options.emptyMarkerHeight ?? 6;
  const countLabelY = railBaseline + 12;
  const x = linearScale({ domain: [0, 1], range: [box.left, box.width - box.right] });
  const y = linearScale({ domain: [0, 1], range: [box.height - box.bottom, box.top] });
  const occupied = report.bins.filter(bin => bin.count > 0);
  const largest = occupied.reduce((best, bin) => Math.max(best, bin.count), 0);
  return {
    box, railHeight, railBaseline, tickLabelY,
    railTopEdge: railBaseline - railHeight,
    diagonal: { x1: x(0), y1: y(0), x2: x(1), y2: y(1) },
    points: occupied.map(bin => {
      const pointY = y(bin.fractionPositive);
      const diagonalY = y(bin.meanP);
      /* The bin number sits above its dot, except in two cases.
         1. Near the top of the box an ascender would climb out of the viewBox
            and be clipped, and a bin whose observed fraction is 1 is exactly
            the interesting case, so the label moves below the dot rather than
            being trimmed.
         2. The gap is drawn as a vertical drop from the dot to the diagonal.
            When that drop goes UP, a label above the dot is a label with a
            line through it. The browser verifier's curve sampler was blind to
            <line> until the guard audit, so nothing reported this: bin 2 of
            the opening reliability figure had 30% of its gap line inside its
            own number. The label goes to whichever side the drop is not on,
            and clipping wins over that when only one side fits. */
      const dropGoesUp = diagonalY < pointY - 1;
      const dropGoesDown = diagonalY > pointY + 1;
      const roomAbove = pointY - 9 > box.top;
      const roomBelow = pointY + 14 < box.height - box.bottom;
      let labelAbove;
      if (dropGoesUp && roomBelow) labelAbove = false;
      else if (dropGoesDown && roomAbove) labelAbove = true;
      else labelAbove = roomAbove;
      const labelBand = labelAbove ? [pointY - 18, pointY - 7] : [pointY + 5, pointY + 16];
      /* A bin sitting on the floor of the box with its drop going up has no
         free side: below is outside the plot area, where the tick labels live.
         It steps sideways instead, the same move the rank label makes against
         the threshold rule. */
      const dropTop = Math.min(pointY, diagonalY);
      const dropBottom = Math.max(pointY, diagonalY);
      const stillOverlaps = Math.abs(pointY - diagonalY) > 1
        && labelBand[0] < dropBottom && labelBand[1] > dropTop;
      return {
        index: bin.index,
        x: x(bin.meanP), y: pointY,
        labelX: stillOverlaps ? x(bin.meanP) + 10 : x(bin.meanP),
        labelY: labelAbove ? pointY - 9 : pointY + 14,
        labelAbove,
        /* Exported so the drawn rule can be asserted rather than eyeballed:
           the label band and the gap segment must not overlap, or the label
           must have stepped clear of the line horizontally. */
        labelBand,
        meanP: bin.meanP, fractionPositive: bin.fractionPositive, count: bin.count, gap: bin.gap,
        /* The gap is drawn as a vertical drop to the diagonal, so it reads as a
           signed distance rather than as a colour. */
        diagonalY: y(bin.meanP),
      };
    }),
    emptyBins: report.bins.filter(bin => bin.count === 0).map(bin => ({
      index: bin.index, lower: bin.lower, upper: bin.upper,
      centre: x((bin.lower + bin.upper) / 2),
    })),
    /* `height` is the quantity: strictly proportional to the count, with no
       minimum, so a bin holding one row of five hundred draws a fifth of a
       unit and not a stub that reads as evidence.

       `markerHeight` is what is DRAWN. It equals `height` for every bin that
       holds anything. An empty bin gets the declared marker instead — a dashed
       outline, not a filled bar — because the row still exists and has no
       observed fraction, which is a different statement from a count of zero.
       The two are separate fields so the verifier can assert the quantity is
       untouched while the marker is exactly the declared constant. */
    emptyMarkerHeight,
    countLabelY,
    rail: report.bins.map(bin => ({
      index: bin.index,
      x1: x(bin.lower), x2: x(bin.upper), count: bin.count,
      height: largest > 0 ? (bin.count / largest) * railHeight : 0,
      markerHeight: bin.count > 0
        ? (largest > 0 ? (bin.count / largest) * railHeight : 0)
        : emptyMarkerHeight,
      empty: bin.count === 0,
      /* The count printed under the bar.
       *
       * The bar length is the quantity and carries no minimum, which is right —
       * but it means a bin holding one row of fifty draws half a unit and is
       * indistinguishable from an empty one. The whole job of this rail is that
       * "a dot without its denominator hides that difference", so the
       * denominator is printed rather than left to a length nobody can read. */
      countLabelX: (x(bin.lower) + x(bin.upper)) / 2,
    })),
    ticks: [0, 0.25, 0.5, 0.75, 1].map(value => ({ value, x: x(value), y: y(value) })),
    scales: { x, y },
  };
}

/**
 * The pooled blocks as rectangles: one per block, spanning its knots, with its
 * height set by its fitted probability.
 */
export function blockGeometry(fit, options = {}) {
  const box = { ...plotBox, ...options.box };
  const knotCount = fit.knots.length;
  /* A block is drawn half a slot either side of its end knots, so the knots
     themselves must sit half a slot inside the plot area. Mapping them to the
     full width instead put the first block's left edge at `box.left - slot/2`,
     out over the y-axis value labels, which it then covered. The spacing
     between knots is unchanged — `available / knotCount` either way — so this
     moves nothing quantitative; it only stops the drawing leaving its box. */
  const available = (box.width - box.right) - box.left;
  const slot = available / Math.max(knotCount, 1);
  const x = knotCount > 1
    ? linearScale({
      domain: [0, knotCount - 1],
      range: [box.left + slot / 2, box.width - box.right - slot / 2],
    })
    : linearScale({ domain: [0, 1], range: [box.left + slot / 2, box.width - box.right - slot / 2] });
  const y = linearScale({ domain: [0, 1], range: [box.height - box.bottom, box.top] });
  return {
    box,
    knots: fit.knots.map((knot, index) => ({
      index, knot, x: x(index), weight: fit.weights[index], total: fit.totals[index],
      rawMean: fit.totals[index] / fit.weights[index],
      rawY: y(fit.totals[index] / fit.weights[index]),
      fitted: fit.fitted[index], fittedY: y(fit.fitted[index]),
    })),
    blocks: fit.blocks.map(block => ({
      ...block,
      x1: x(block.start) - slot / 2,
      x2: x(block.end) + slot / 2,
      y: y(block.mean),
      height: y(0) - y(block.mean),
      spans: block.end - block.start + 1,
    })),
    baseline: y(0),
    slot,
    scales: { x, y },
  };
}

/**
 * The rank rail: every calibration score as a mark on one axis, with the
 * selected order statistic named.
 *
 * An infinite threshold is returned as `markerX: null` and `unbounded: true`.
 * Drawing it at the right-hand edge would make the whole label space look like
 * a finite score just past the last card.
 */
export function railGeometry(scores, threshold, options = {}) {
  const box = { ...plotBox, ...options.box, height: options.height ?? 120 };
  const upper = options.domain ? options.domain[1]
    : Math.max(...scores, Number.isFinite(threshold.q) ? threshold.q : 0, 0);
  const lower = options.domain ? options.domain[0] : Math.min(...scores, 0);
  const x = linearScale({
    domain: [lower, upper === lower ? lower + 1 : upper],
    range: [box.left, box.width - box.right],
  });
  const sorted = [...scores].sort((a, b) => a - b);
  return {
    box,
    axis: { x1: x(x.domain[0]), x2: x(x.domain[1]), y: box.height - box.bottom },
    marks: sorted.map((score, rank) => {
      const selected = threshold.finite && rank + 1 === threshold.k;
      return {
        rank: rank + 1, score, x: x(score),
        selected,
        /* The rank label steps aside for the selected mark, because the
           threshold rule is drawn at exactly that x through the whole figure
           and would otherwise run down the middle of the digit. The browser
           verifier could not see this: its curve sampler queried
           `path, polyline, polygon` and the rule is a <line>. The offset is
           returned here rather than applied in the component so that the
           clearance can be asserted against `markerX`. */
        labelX: selected ? x(score) + 10 : x(score),
        belowThreshold: score <= threshold.q,
      };
    }),
    markerX: threshold.finite ? x(threshold.q) : null,
    unbounded: !threshold.finite,
    scales: { x },
  };
}

/**
 * Horizontal interval segments against their observed responses.
 *
 * `miss` is computed from the same endpoints the segment is drawn from, so a
 * marker outside its band cannot be styled as covered.
 */
export function intervalGeometry(rows, options = {}) {
  const box = { ...plotBox, ...options.box, height: options.height ?? 260 };
  const values = rows.flatMap(row => [row.lower, row.upper, row.y].filter(Number.isFinite));
  const lowest = options.domain ? options.domain[0] : Math.min(...values);
  const highest = options.domain ? options.domain[1] : Math.max(...values);
  const pad = (highest - lowest) * 0.04 || 1;
  const x = linearScale({ domain: [lowest - pad, highest + pad], range: [box.left, box.width - box.right] });
  const top = box.top;
  const bottom = box.height - box.bottom;
  const step = rows.length > 1 ? (bottom - top) / (rows.length - 1) : 0;
  return {
    box,
    /* `value` is the observed response and `y` is the screen row. They were one
       field once, and a figure that read `row.y` for the data got a pixel
       coordinate — which is how a mark ends up drawn at the wrong place while
       every number in the table is right. */
    rows: rows.map((row, index) => ({
      ...row,
      value: row.y,
      y: top + index * step,
      x1: x(row.lower), x2: x(row.upper), target: x(row.y),
      width: row.upper - row.lower,
      covered: row.lower <= row.y && row.y <= row.upper,
    })),
    scales: { x },
  };
}

/** Proportional bars whose declared quantity is the length, with no minimum. */
export function barGeometry(values, options = {}) {
  const width = options.width ?? 220;
  const largest = options.maximum ?? Math.max(...values.map(entry => entry.value), 0);
  return values.map(entry => ({
    ...entry,
    length: largest > 0 ? (entry.value / largest) * width : 0,
    share: largest > 0 ? entry.value / largest : 0,
  }));
}

/**
 * The group mosaic: area is share times coverage, so the covered area really is
 * the marginal coverage.
 */
export function mosaicGeometry(model, options = {}) {
  const width = options.width ?? 280;
  const height = options.height ?? 110;
  let cursor = 0;
  return {
    width, height,
    groups: model.groups.map(group => {
      const columnWidth = group.share * width;
      const coveredHeight = group.coverage * height;
      const rectangle = {
        ...group,
        x: cursor, width: columnWidth,
        coveredY: height - coveredHeight, coveredHeight,
        missedY: 0, missedHeight: height - coveredHeight,
        area: group.share * group.coverage,
      };
      cursor += columnWidth;
      return rectangle;
    }),
    marginal: model.marginal,
    coveredArea: model.groups.reduce((sum, group) => sum + group.share * group.coverage, 0),
  };
}

/** A residual, a prediction and a response on one number line, in target units. */
export function numberLineGeometry({ centre, response, halfWidth }, options = {}) {
  const width = options.width ?? 300;
  const left = options.left ?? 28;
  const right = options.right ?? 12;
  const y = options.y ?? 36;
  const span = Math.max(Math.abs(response - centre), halfWidth) * 1.35 || 1;
  const x = linearScale({ domain: [centre - span, centre + span], range: [left, width - right] });
  return {
    width, y,
    axis: { x1: left, x2: width - right, y },
    centre: x(centre), response: x(response),
    lower: x(centre - halfWidth), upper: x(centre + halfWidth),
    residual: Math.abs(response - centre),
    inside: Math.abs(response - centre) <= halfWidth,
    scales: { x },
  };
}

/* ================================================================= fixtures */

/**
 * Every declared starting state on this page, in one place, so the verifier
 * checks the fixtures the components actually load.
 *
 * The constructed fixtures are exact designed examples. The two measured
 * experiments live in calibration-data.js, generated from the frozen packet.
 */
export const fixtures = {
  /* §1 */
  forecastGroups: [
    { name: 'line A', share: 0.5, forecast: 0.2, positiveRate: 0.3 },
    { name: 'line B', share: 0.5, forecast: 0.8, positiveRate: 0.9 },
  ],
  practiceForecastGroups: [
    { name: 'group 1', share: 0.5, forecast: 0.1, positiveRate: 0.2 },
    { name: 'group 2', share: 0.5, forecast: 0.9, positiveRate: 1 },
  ],
  resolution: { shares: [0.5, 0.5], rates: [0.1, 0.3], releaseCost: 80, quarantineCost: 20 },

  /* §2 · investigation 1 */
  cards: [
    { id: 'A', forecast: 0.2, outcome: 1 }, { id: 'B', forecast: 0.2, outcome: 1 },
    { id: 'C', forecast: 0.2, outcome: 0 }, { id: 'D', forecast: 0.2, outcome: 0 },
    { id: 'E', forecast: 0.2, outcome: 0 }, { id: 'F', forecast: 0.8, outcome: 1 },
    { id: 'G', forecast: 0.8, outcome: 1 }, { id: 'H', forecast: 0.8, outcome: 1 },
    { id: 'I', forecast: 0.8, outcome: 0 }, { id: 'J', forecast: 0.8, outcome: 0 },
  ],
  twoBinEdges: [0, 0.5, 1],
  oneBinEdges: [0, 1],
  repairedForecasts: [0.4, 0.4, 0.4, 0.4, 0.4, 0.6, 0.6, 0.6, 0.6, 0.6],
  boundaryCards: [
    { id: 'P', forecast: 0, outcome: 0 }, { id: 'Q', forecast: 0.5, outcome: 1 },
    { id: 'R', forecast: 1, outcome: 1 },
  ],

  /* §3 · investigation 2
   *
   * The lab opens on its OWN six observations, not on the eight the prose works
   * through. Two of these investigations ask for an absolute answer rather than
   * a change, so there is no "an edit is required first" to stop a learner
   * reading the opening answer straight out of the surrounding explanation. The
   * worked fixture is one button away as a preset, which is the right way to
   * reproduce a stated result — after having tried one that is not stated. */
  labPavScores: [-2, -1, 0, 1, 2, 3],
  labPavLabels: [1, 0, 1, 0, 1, 1],
  pavScores: [-3, -2, -1, 0, 1, 2, 3, 4],
  pavLabels: [0, 1, 0, 0, 1, 0, 1, 1],
  tiedScores: [-2, -2, 0, 1],
  tiedLabels: [1, 0, 0, 1],
  tiedChangedLabels: [1, 0, 1, 1],
  practicePavScores: [1, 2, 3, 4, 5, 6],
  practicePavLabels: [0, 1, 0, 1, 0, 1],
  temperatureLogits: [3, 1, 0],
  temperatures: [0.5, 1, 2, 4],

  /* §5 · investigation 3
   *
   * The lab's own nine cards differ from the manuscript's in one place: the
   * eighth is .55 rather than .6. That makes the threshold .55, and the second
   * candidate's leading score is exactly .55 — so the very first question a
   * learner meets turns on whether the comparison includes equality, and its
   * answer is nowhere in the prose. The manuscript's nine cards are a preset. */
  labCalibrationScores: [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.55, 0.9],
  calibrationScores: [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.6, 0.9],
  rotationScores: [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.6, 0.9, 0.95],
  tiedRotationScores: [0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2],
  defaultAlpha: 0.2,
  candidateVectors: [
    { name: 'confident', probabilities: [0.8, 0.15, 0.05] },
    { name: 'two plausible classes', probabilities: [0.45, 0.4, 0.15] },
    { name: 'nearly uniform', probabilities: [0.34, 0.33, 0.33] },
  ],
  classNames: ['A', 'B', 'C'],
  practiceVector: [0.5, 0.35, 0.15],
  practiceQ: 0.65,
  apsProbabilities: [0.5, 0.3, 0.2],

  /* §6 · investigation 4 */
  residuals: [0.5, 1, 1.5, 2, 3, 4, 5, 6, 8],
  localScales: [1, 1, 1, 1, 2, 2, 2, 3, 4],
  changedResiduals: [0.5, 1, 1.5, 2, 3, 4, 5, 12, 16],
  queries: [
    { name: 'easy case', centre: 10, localScale: 1 },
    { name: 'hard case', centre: 20, localScale: 3 },
  ],
  intervalAlpha: 0.2,
  cqrBase: { lower: 10, upper: 20 },
  cqrScores: [-5, -4, -3, -2, -1],
  cqrAlternativeScores: [-2, -1, 0, 1, 3],
  cqrAlpha: 0.4,

  /* §8 */
  mosaic: [
    { name: 'group A', share: 0.8, coverage: 1 },
    { name: 'group B', share: 0.2, coverage: 0.5 },
  ],
  labelShift: { sensitivity: 0.8, falsePositiveRate: 0.2, prevalences: [0.5, 0.1] },

  /* §7 · the floating-point boundary the offline program retains */
  floatingBoundary: { positiveProbability: 103 / 240 },
};

/**
 * The tied boundary, recomputed rather than quoted.
 *
 * `1 - p <= q` includes the equality; the algebraically identical
 * `p >= 1 - q` does not, because the second subtraction is inexact. The
 * offline experiment's constant baseline lost a whole class to this and
 * reported 51 of 80 correct instead of covering all 80.
 */
export function floatingBoundaryCheck(positiveProbability) {
  checkProbability(positiveProbability, 'the constant class-1 probability');
  const q = 1 - positiveProbability;
  return {
    positiveProbability, q,
    scoreComparison: 1 - positiveProbability <= q,
    rearrangedComparison: positiveProbability >= 1 - q,
    agree: (1 - positiveProbability <= q) === (positiveProbability >= 1 - q),
  };
}
