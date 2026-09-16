/** Pure models for the bias–variance and learning-curve lesson.
 *
 * Everything a figure or investigation draws is computed here, so that a drawn
 * curve, band, bar or crossover is an asserted mathematical claim rather than a
 * shape chosen in a component. Nothing is interpolated from a stored picture.
 *
 * Three categories of quantity live here and are never mixed:
 *
 *   1. Exact finite enumerations. The three-world sensor and the 8- or 32-world
 *      polynomial experiment enumerate an entire equally weighted distribution,
 *      so their variances divide by the number of worlds and are exact.
 *   2. Recorded real measurements. The airfoil learning, validation and
 *      boosting records are read from bias-variance-data.js; the functions here
 *      only average, compare and bound them.
 *   3. Asymptotic theory. The double-descent expressions are the manuscript's
 *      stated large-n, large-p approximation, not a finite simulation.
 *
 * Every entry point refuses input it cannot honour rather than substituting a
 * silent default: a non-finite number, an out-of-range setting, a design that is
 * not full column rank or an undefined ratio all raise a RangeError.
 */

/* ------------------------------------------------------------------ guards */

export const limits = {
  /** Investigation 1: one prediction ruler, shared by the outcome ruler. */
  prediction: { minimum: 0, maximum: 20 },
  trueMean: { minimum: 0, maximum: 20 },
  noiseSpread: { minimum: 0, maximum: 4 },
  /** Investigation 2: the data-generating mechanism and the queried input. */
  curvature: { minimum: -1, maximum: 2 },
  sigma: { minimum: 0, maximum: 1.5 },
  probe: { minimum: -1.5, maximum: 1.5 },
  degree: { minimum: 0, maximum: 2 },
  /** Bounded enumeration: five training inputs give 32 equally likely worlds. */
  maximumTrainingInputs: 5,
  maximumWorlds: 32,
  gridPoints: 61,
  tolerance: 1e-10,
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
function checkWholeRange(value, range, name) {
  if (!Number.isInteger(value)) throw new RangeError(`${name} must be a whole number.`);
  return checkRange(value, range, name);
}

const mean = values => values.reduce((sum, value) => sum + value, 0) / values.length;
/** Population variance over an enumerated, equally weighted set: divide by N. */
const populationVariance = values => {
  const centre = mean(values);
  return mean(values.map(value => (value - centre) ** 2));
};
export { mean as meanOf, populationVariance };

/* --------------------------------------------------- §1 · three sensor worlds */

/** One fixed input. Equally likely fitted predictions, a known target mean, and
 * fresh outcomes constructed as mean ± spread with equal probability.
 *
 * The two kinds of spread are deliberately kept apart: the predictions belong to
 * different training draws, and the outcomes belong to the same input. The total
 * is computed twice — once as the three-term decomposition, once by averaging
 * every prediction/outcome pair — and the two must agree exactly.
 */
export function predictionSpread(predictions, trueMean, noiseSpread) {
  if (!Array.isArray(predictions) || predictions.length < 2 || predictions.length > 6) {
    throw new RangeError('Use between two and six equally likely predictions.');
  }
  const values = predictions.map((value, index) => checkRange(value, limits.prediction, `prediction ${index + 1}`));
  checkRange(trueMean, limits.trueMean, 'the true target mean');
  checkRange(noiseSpread, limits.noiseSpread, 'the fresh-outcome spread');
  const averagePrediction = mean(values);
  const signedOffset = averagePrediction - trueMean;
  const squaredBias = signedOffset ** 2;
  const variance = populationVariance(values);
  const noiseVariance = noiseSpread ** 2;
  const total = squaredBias + variance + noiseVariance;
  const outcomes = [trueMean - noiseSpread, trueMean + noiseSpread];
  const pairs = values.flatMap((prediction, index) => outcomes.map((outcome, slot) => ({
    predictionIndex: index, prediction, outcome, outcomeIndex: slot,
    probability: 1 / (values.length * outcomes.length),
    squaredError: (outcome - prediction) ** 2,
  })));
  const pairAverage = mean(pairs.map(pair => pair.squaredError));
  return {
    predictions: values, trueMean, noiseSpread,
    averagePrediction, signedOffset, squaredBias, variance, noiseVariance, total,
    outcomes, pairs, pairAverage,
    /** The decomposition and the direct enumeration are the same number. */
    agrees: Math.abs(total - pairAverage) <= limits.tolerance * Math.max(1, Math.abs(total)),
    /** A ruler wide enough for every mark, never narrower than the control range. */
    domain: [
      Math.min(limits.prediction.minimum, averagePrediction, ...values, ...outcomes),
      Math.max(limits.prediction.maximum, averagePrediction, ...values, ...outcomes),
    ],
  };
}

/** Compare two applied states of the same experiment. The direction is the only
 * graded claim; the tolerance makes an exact tie an exact tie. */
export function spreadComparison(previous, proposed) {
  const before = predictionSpread(previous.predictions, previous.trueMean, previous.noiseSpread);
  const after = predictionSpread(proposed.predictions, proposed.trueMean, proposed.noiseSpread);
  const difference = after.total - before.total;
  const unchanged = Math.abs(difference) <= limits.tolerance * Math.max(1, Math.abs(before.total));
  return {
    before, after, difference,
    outcome: unchanged ? 'unchanged' : difference < 0 ? 'decrease' : 'increase',
    changed: {
      predictions: proposed.predictions.some((value, index) => value !== previous.predictions[index]),
      trueMean: proposed.trueMean !== previous.trueMean,
      noiseSpread: proposed.noiseSpread !== previous.noiseSpread,
      squaredBias: Math.abs(after.squaredBias - before.squaredBias) > limits.tolerance,
      variance: Math.abs(after.variance - before.variance) > limits.tolerance,
      noiseVariance: Math.abs(after.noiseVariance - before.noiseVariance) > limits.tolerance,
    },
  };
}

/* ------------------------------------------- small dense linear algebra */

/** Gaussian elimination with partial pivoting. A singular system is refused
 * rather than returned as a huge number. */
export function solveLinear(matrix, vector) {
  const n = vector.length;
  if (matrix.length !== n || matrix.some(row => row.length !== n)) {
    throw new RangeError('The system must be square.');
  }
  const a = matrix.map((row, index) => [
    ...row.map(value => checkFinite(value, 'a matrix entry')),
    checkFinite(vector[index], 'a right-hand value'),
  ]);
  for (let column = 0; column < n; column += 1) {
    let pivot = column;
    for (let row = column + 1; row < n; row += 1) {
      if (Math.abs(a[row][column]) > Math.abs(a[pivot][column])) pivot = row;
    }
    if (Math.abs(a[pivot][column]) < 1e-12) {
      throw new RangeError('This design is not full column rank; no unique least-squares fit exists.');
    }
    [a[column], a[pivot]] = [a[pivot], a[column]];
    for (let row = column + 1; row < n; row += 1) {
      const factor = a[row][column] / a[column][column];
      for (let entry = column; entry <= n; entry += 1) a[row][entry] -= factor * a[column][entry];
    }
  }
  const solution = new Array(n).fill(0);
  for (let row = n - 1; row >= 0; row -= 1) {
    let sum = a[row][n];
    for (let column = row + 1; column < n; column += 1) sum -= a[row][column] * solution[column];
    solution[row] = sum / a[row][row];
  }
  return solution;
}

/** One row of the design: [1, x, x², …], matching numpy.vander(increasing=True). */
export function polynomialRow(x, degree) {
  checkFinite(x, 'an input');
  const row = [];
  for (let power = 0; power <= degree; power += 1) row.push(x ** power);
  return row;
}
export function polynomialDesign(trainX, degree) {
  return trainX.map(value => polynomialRow(value, degree));
}
function gram(design) {
  const width = design[0].length;
  return Array.from({ length: width }, (_, i) => Array.from({ length: width }, (_, j) =>
    design.reduce((sum, row) => sum + row[i] * row[j], 0)));
}
/** Least-squares coefficients for one target vector. */
export function leastSquares(design, targets) {
  if (design.length !== targets.length) throw new RangeError('Give one target per training input.');
  const width = design[0].length;
  if (design.length < width) throw new RangeError('This design has fewer rows than fitted coefficients.');
  const right = Array.from({ length: width }, (_, column) =>
    design.reduce((sum, row, index) => sum + row[column] * targets[index], 0));
  return solveLinear(gram(design), right);
}
export function evaluatePolynomial(coefficients, x) {
  return coefficients.reduce((sum, value, power) => sum + value * x ** power, 0);
}

/** The prediction at one probe is a fixed linear functional of the training
 * targets: w = v(probe)ᵀ(XᵀX)⁻¹Xᵀ. These are the interpolation weights the
 * manuscript prints, and they always sum to one because a constant lies in the
 * fitted column space. */
export function predictionWeights(trainX, degree, probe) {
  const design = polynomialDesign(trainX, degree);
  const direction = solveLinear(gram(design), polynomialRow(probe, degree));
  const weights = design.map(row => row.reduce((sum, value, index) => sum + value * direction[index], 0));
  return {
    weights,
    sum: weights.reduce((sum, value) => sum + value, 0),
    sumOfSquares: weights.reduce((sum, value) => sum + value * value, 0),
    negativeCount: weights.filter(value => value < 0).length,
  };
}

/* --------------------------------- §3 · enumerate every possible tiny dataset */

export const defaultGrid = Array.from({ length: limits.gridPoints }, (_, index) =>
  -1.5 + 3 * index / (limits.gridPoints - 1));

function signVectors(count) {
  const rows = [];
  for (let mask = 0; mask < 2 ** count; mask += 1) {
    rows.push(Array.from({ length: count }, (_, position) =>
      ((mask >> (count - 1 - position)) & 1) ? 1 : -1));
  }
  return rows;
}

/** The complete finite experiment of section 3.
 *
 * Training inputs are fixed. At each one the outcome is the true mean plus or
 * minus sigma, equally likely and independent, so k inputs give exactly 2^k
 * equally likely training datasets. Every one of them is fitted by unregularized
 * least squares; nothing is sampled, bootstrapped or seeded.
 */
export function finiteExperiment({ trainX, degree, curvature, sigma, probe, grid = defaultGrid }) {
  if (!Array.isArray(trainX) || trainX.length < 2 || trainX.length > limits.maximumTrainingInputs) {
    throw new RangeError(`Use between two and ${limits.maximumTrainingInputs} training inputs.`);
  }
  const inputs = trainX.map((value, index) => checkRange(value, limits.probe, `training input ${index + 1}`));
  if (new Set(inputs).size !== inputs.length) throw new RangeError('Training inputs must be distinct.');
  checkWholeRange(degree, limits.degree, 'the polynomial degree');
  checkRange(curvature, limits.curvature, 'the true curvature');
  checkRange(sigma, limits.sigma, 'the training-noise level');
  checkRange(probe, limits.probe, 'the probe input');
  if (degree + 1 > inputs.length) throw new RangeError('This degree needs more training inputs than are supplied.');

  const truthAt = x => 1 + x + curvature * x * x;
  const cleanTargets = inputs.map(truthAt);
  const design = polynomialDesign(inputs, degree);
  const worlds = signVectors(inputs.length).map((signs, index) => {
    const targets = cleanTargets.map((value, position) => value + sigma * signs[position]);
    const coefficients = leastSquares(design, targets);
    return {
      index, signs, targets, coefficients,
      probability: 1 / 2 ** inputs.length,
      probePrediction: evaluatePolynomial(coefficients, probe),
      fitted: inputs.map(value => evaluatePolynomial(coefficients, value)),
    };
  });

  const probeTruth = truthAt(probe);
  const probePredictions = worlds.map(world => world.probePrediction);
  const meanPrediction = mean(probePredictions);
  const squaredBias = (meanPrediction - probeTruth) ** 2;
  const variance = populationVariance(probePredictions);
  const noiseVariance = sigma * sigma;
  const excess = mean(probePredictions.map(value => (value - probeTruth) ** 2));
  const weights = predictionWeights(inputs, degree, probe);

  const curves = worlds.map(world => grid.map(x => evaluatePolynomial(world.coefficients, x)));
  const truthCurve = grid.map(truthAt);
  const averageCurve = grid.map((_, column) => mean(curves.map(curve => curve[column])));
  const gridDecomposition = grid.map((x, column) => {
    const values = curves.map(curve => curve[column]);
    const centre = mean(values);
    const truth = truthCurve[column];
    return {
      x, mean: centre,
      squaredBias: (centre - truth) ** 2,
      variance: populationVariance(values),
      expectedError: mean(values.map(value => (value - truth) ** 2)) + noiseVariance,
    };
  });

  const low = Math.min(...curves.flat(), ...truthCurve);
  const high = Math.max(...curves.flat(), ...truthCurve);
  const pad = Math.max(0.2, (high - low) * 0.08);
  return {
    trainX: inputs, degree, curvature, sigma, probe, grid,
    worldCount: worlds.length, worlds, cleanTargets,
    probeTruth, probePredictions, meanPrediction, squaredBias, variance, noiseVariance,
    excess, expectedError: excess + noiseVariance,
    /** Squared bias plus prediction variance is the excess over noise, exactly. */
    identityResidual: Math.abs(excess - squaredBias - variance),
    weights: weights.weights, weightSum: weights.sum, weightSumOfSquares: weights.sumOfSquares,
    negativeWeights: weights.negativeCount,
    varianceFromWeights: noiseVariance * weights.sumOfSquares,
    curves, truthCurve, averageCurve, gridDecomposition,
    /** Never clip a curve to manufacture stability: the range follows the data. */
    valueRange: [low - pad, high + pad],
  };
}

/** Reference against candidate at the same probe, same truth and same design. */
export function degreeComparison(settings, referenceDegree, candidateDegree) {
  const reference = finiteExperiment({ ...settings, degree: referenceDegree });
  const candidate = finiteExperiment({ ...settings, degree: candidateDegree });
  const difference = candidate.expectedError - reference.expectedError;
  const scale = Math.max(1, Math.abs(reference.expectedError));
  return {
    reference, candidate, referenceDegree, candidateDegree, difference,
    outcome: Math.abs(difference) <= limits.tolerance * scale
      ? 'same' : difference < 0 ? 'lower' : 'higher',
    /** Which of the three terms actually moved, so feedback names the operand. */
    movedBias: Math.abs(candidate.squaredBias - reference.squaredBias) > limits.tolerance,
    movedVariance: Math.abs(candidate.variance - reference.variance) > limits.tolerance,
    identicalPredictions: reference.probePredictions.every((value, index) =>
      Math.abs(value - candidate.probePredictions[index]) <= 1e-12),
  };
}

/* ------------------------------- §2 · information changes the noise floor */

/** Y = X + Z + noise, with Z equally likely −1 or +1 and independent of X.
 * Var(Y|X) = E[Var(Y|X,Z)|X] + Var(E[Y|X,Z]|X): the second term is exactly what
 * measuring Z removes. */
export function hiddenSettingVariance(settingValues = [-1, 1], noiseVariance = 0.25) {
  const values = settingValues.map((value, index) => checkFinite(value, `setting value ${index + 1}`));
  if (!(checkFinite(noiseVariance, 'the independent noise variance') >= 0)) {
    throw new RangeError('A variance cannot be negative.');
  }
  const settingVariance = populationVariance(values);
  return {
    settingValues: values, noiseVariance, settingVariance,
    givenInputOnly: settingVariance + noiseVariance,
    givenInputAndSetting: noiseVariance,
    explainedByMeasuringSetting: settingVariance,
  };
}

/* --------------------------------------- §§4–5 · recorded real-data curves */

function checkFolds(folds, label) {
  if (!Array.isArray(folds) || folds.length === 0) throw new RangeError(`${label} needs at least one value.`);
  return folds.map((value, index) => checkFinite(value, `${label} fold ${index + 1}`));
}

/** Per-size means and spreads for one recorded procedure. The mean line a figure
 * draws is computed here from the stored fold values, never stored separately. */
export function learningCurveSeries(record) {
  const sizes = record.sizes.map((value, index) => {
    if (!Number.isInteger(value) || value <= 0) throw new RangeError(`training size ${index + 1} must be a positive whole number.`);
    return value;
  });
  const summarize = (folds, label) => folds.map((values, index) => {
    const checked = checkFolds(values, `${label} at size ${sizes[index]}`);
    return {
      size: sizes[index], folds: checked, mean: mean(checked),
      minimum: Math.min(...checked), maximum: Math.max(...checked),
    };
  });
  const training = summarize(record.trainMse, 'training MSE');
  const validation = summarize(record.validationMse, 'validation MSE');
  return {
    model: record.model, label: record.label, sizes, training, validation,
    trainingMeans: training.map(entry => entry.mean),
    validationMeans: validation.map(entry => entry.mean),
    /** A training curve that is exactly zero at every inspected size is a claim
     * about interpolation, so it is detected rather than eyeballed. */
    trainingExactlyZero: training.every(entry => entry.folds.every(value => value === 0)),
    validationDecreasing: validation.every((entry, index) =>
      index === 0 || entry.mean < validation[index - 1].mean),
    finalGap: validation.at(-1).mean - training.at(-1).mean,
  };
}

/** The smallest inspected size at which one procedure's mean validation error
 * first drops below another's, or null when no crossing occurs in range. */
export function firstCrossing(series, reference) {
  if (series.sizes.length !== reference.sizes.length) throw new RangeError('Compare series on the same sizes.');
  for (let index = 0; index < series.sizes.length; index += 1) {
    if (series.sizes[index] !== reference.sizes[index]) throw new RangeError('Compare series on the same sizes.');
    if (series.validationMeans[index] < reference.validationMeans[index]) return series.sizes[index];
  }
  return null;
}

/** The sizes at which a restriction helps and where it stops helping. */
export function restrictionEffect(restricted, unrestricted) {
  const helps = restricted.sizes.filter((size, index) =>
    restricted.validationMeans[index] < unrestricted.validationMeans[index]);
  const hurts = restricted.sizes.filter((size, index) =>
    restricted.validationMeans[index] > unrestricted.validationMeans[index]);
  return { helps, hurts, firstHurtSize: hurts.length ? hurts[0] : null };
}

export function validationCurveSeries(record) {
  const settings = record.minSamplesLeaf;
  const summarize = folds => folds.map((values, index) => {
    const checked = checkFolds(values, `leaf size ${settings[index]}`);
    return { setting: settings[index], folds: checked, mean: mean(checked) };
  });
  const training = summarize(record.trainMse);
  const validation = summarize(record.validationMse);
  return {
    settings, training, validation,
    trainingMeans: training.map(entry => entry.mean),
    validationMeans: validation.map(entry => entry.mean),
    bestSetting: settings[validation.reduce((best, entry, index) =>
      (entry.mean < validation[best].mean ? index : best), 0)],
    /** This inspected restriction never improves the score; say so from the data. */
    restrictionImproves: validation.some(entry => entry.mean < validation[0].mean),
  };
}

/** The staged boosting trace. The best round is an argmin over every recorded
 * round, not over the five rounds the manuscript prints. */
export function trajectorySeries(record) {
  const train = checkFolds(record.trainMse, 'training MSE');
  const monitor = checkFolds(record.monitorMse, 'monitoring MSE');
  if (train.length !== monitor.length) throw new RangeError('Both traces need the same number of rounds.');
  let bestIndex = 0;
  monitor.forEach((value, index) => { if (value < monitor[bestIndex]) bestIndex = index; });
  return {
    rounds: monitor.map((_, index) => index + 1),
    train, monitor,
    bestRound: bestIndex + 1,
    bestMonitor: monitor[bestIndex],
    lastRound: monitor.length,
    /** The lowest monitoring error over every recorded round is the last one, so
     * no stopping point followed by deterioration was observed. */
    bestRoundIsLast: bestIndex + 1 === monitor.length,
    /** The trace still wiggles: these are the rounds where it rose from the one
     * before. They are recorded and checkable, not visible: the largest is a few
     * hundredths of a squared decibel against a 0-50 axis, so a figure must say
     * where to read them rather than claim they can be seen. */
    risingRounds: monitor.reduce((rounds, value, index) =>
      (index > 0 && value > monitor[index - 1] ? [...rounds, index + 1] : rounds), []),
    largestRise: monitor.reduce((largest, value, index) =>
      (index > 0 ? Math.max(largest, value - monitor[index - 1]) : largest), 0),
    finalGap: monitor.at(-1) - train.at(-1),
  };
}

/* ---------------------------------------- §7 · why training error is optimistic */

/** Correctly specified fixed-design least squares: n rows, p fitted
 * coefficients including any intercept, homoscedastic noise. */
export function fixedDesignOptimism(n, p, noiseVariance) {
  if (!Number.isInteger(n) || n < 1) throw new RangeError('Use a positive whole number of rows.');
  if (!Number.isInteger(p) || p < 1 || p > n) throw new RangeError('Use between one and n fitted coefficients.');
  if (!(checkFinite(noiseVariance, 'the noise variance') >= 0)) throw new RangeError('A variance cannot be negative.');
  return {
    n, p, noiseVariance,
    trainingMse: noiseVariance * (1 - p / n),
    newOutcomeMse: noiseVariance * (1 + p / n),
    gap: 2 * p * noiseVariance / n,
    predictionVariance: p * noiseVariance / n,
    /** The gap is twice the average fitted-prediction variance, not equal to it. */
    gapOverVariance: 2,
  };
}

/** The hat matrix of an explicit design, plus its leverages. */
export function projectionSmoother(design) {
  if (!Array.isArray(design) || design.length === 0) throw new RangeError('Give at least one design row.');
  const width = design[0].length;
  if (design.some(row => row.length !== width)) throw new RangeError('Every design row needs the same width.');
  const G = gram(design);
  const columns = Array.from({ length: width }, (_, index) =>
    solveLinear(G, Array.from({ length: width }, (_, position) => (position === index ? 1 : 0))));
  const inverse = Array.from({ length: width }, (_, i) => Array.from({ length: width }, (_, j) => columns[j][i]));
  const S = design.map(rowA => design.map(rowB =>
    rowA.reduce((sum, value, i) => sum + value * rowB.reduce((inner, other, j) => inner + inverse[i][j] * other, 0), 0)));
  return {
    matrix: S, inverse, rank: width,
    leverages: S.map((row, index) => row[index]),
    trace: S.reduce((sum, row, index) => sum + row[index], 0),
  };
}

/** Training and fresh-outcome error for any fixed linear smoother ŷ = Sy, under
 * a mean vector f. Their difference is exactly 2σ²tr(S)/n. */
export function smootherOptimism(S, meanVector, noiseVariance) {
  const n = meanVector.length;
  if (S.length !== n || S.some(row => row.length !== n)) throw new RangeError('The smoother must be n by n.');
  meanVector.forEach((value, index) => checkFinite(value, `mean entry ${index + 1}`));
  if (!(checkFinite(noiseVariance, 'the noise variance') >= 0)) throw new RangeError('A variance cannot be negative.');
  const fitted = S.map(row => row.reduce((sum, value, index) => sum + value * meanVector[index], 0));
  const approximation = meanVector.reduce((sum, value, index) => sum + (value - fitted[index]) ** 2, 0) / n;
  const trace = S.reduce((sum, row, index) => sum + row[index], 0);
  const traceSquared = S.reduce((sum, row) => sum + row.reduce((inner, value) => inner + value * value, 0), 0);
  const training = approximation + noiseVariance * (n - 2 * trace + traceSquared) / n;
  const fresh = approximation + noiseVariance + noiseVariance * traceSquared / n;
  return {
    n, trace, traceSquared, approximation, training, fresh,
    difference: fresh - training,
    expectedDifference: 2 * noiseVariance * trace / n,
  };
}

/** Expected squared error at a genuinely new input, conditional on the fixed
 * training design. Its leverage depends on where the new input sits. */
export function newLocationRisk(design, newRow, noiseVariance) {
  const { inverse } = projectionSmoother(design);
  if (newRow.length !== inverse.length) throw new RangeError('The new input needs the design’s own coordinates.');
  newRow.forEach((value, index) => checkFinite(value, `new-input coordinate ${index + 1}`));
  const leverage = newRow.reduce((sum, value, i) =>
    sum + value * newRow.reduce((inner, other, j) => inner + inverse[i][j] * other, 0), 0);
  return {
    newRow, leverage,
    predictionVariance: noiseVariance * leverage,
    expectedSquaredError: noiseVariance * (1 + leverage),
  };
}

/* --------------------------------- §8 · classification and averaging */

/** The exact squared-probability (Brier) decomposition at one input. */
export function brierDecomposition(eta, probabilities) {
  checkRange(eta, { minimum: 0, maximum: 1 }, 'the conditional probability');
  if (!Array.isArray(probabilities) || probabilities.length === 0) throw new RangeError('Give at least one predicted probability.');
  const values = probabilities.map((value, index) => checkRange(value, { minimum: 0, maximum: 1 }, `predicted probability ${index + 1}`));
  const averageProbability = mean(values);
  const squaredBias = (eta - averageProbability) ** 2;
  const variance = populationVariance(values);
  const noise = eta * (1 - eta);
  const direct = mean(values.map(value => eta * (1 - value) ** 2 + (1 - eta) * value ** 2));
  return {
    eta, probabilities: values, averageProbability, squaredBias, variance, noise,
    total: squaredBias + variance + noise, direct,
    agrees: Math.abs(squaredBias + variance + noise - direct) <= limits.tolerance,
  };
}

/** Expected zero-one error when a procedure predicts class 1 with probability q.
 * A thresholded decision behaves differently from the squared probability loss. */
export function thresholdedError(eta, probabilities, threshold = 0.5) {
  checkRange(eta, { minimum: 0, maximum: 1 }, 'the conditional probability');
  const values = probabilities.map((value, index) => checkRange(value, { minimum: 0, maximum: 1 }, `predicted probability ${index + 1}`));
  checkRange(threshold, { minimum: 0, maximum: 1 }, 'the decision threshold');
  const classOneRate = mean(values.map(value => (value >= threshold ? 1 : 0)));
  return {
    eta, threshold, classOneRate,
    error: eta * (1 - classOneRate) + (1 - eta) * classOneRate,
    /** The straight-line error in q: eta − (2·eta − 1)·q. */
    slope: 1 - 2 * eta,
  };
}
export function thresholdedErrorAt(eta, classOneRate) {
  checkRange(eta, { minimum: 0, maximum: 1 }, 'the conditional probability');
  checkRange(classOneRate, { minimum: 0, maximum: 1 }, 'the class-1 rate');
  return eta * (1 - classOneRate) + (1 - eta) * classOneRate;
}

/** The variance of an equal-weight average of B predictors with common variance
 * v and common pairwise correlation rho. */
export function averageVariance(v, rho, B) {
  if (!(checkFinite(v, 'the common variance') >= 0)) throw new RangeError('A variance cannot be negative.');
  checkRange(rho, { minimum: -1, maximum: 1 }, 'the pairwise correlation');
  if (!Number.isInteger(B) || B < 1) throw new RangeError('Use a whole number of predictors, at least one.');
  if (B > 1 && rho < -1 / (B - 1)) throw new RangeError('That correlation is impossible for this many predictors.');
  return {
    v, rho, B,
    individualTerms: B * v,
    pairTerms: B * (B - 1) * rho * v,
    variance: v * (rho + (1 - rho) / B),
    independentVariance: v / B,
    perfectlyCorrelated: v,
  };
}

/* ------------------------------------- §9 · a tradeoff need not draw a U */

export const doubleDescentNoise = 0.04;
export const doubleDescentSignalNorm = 1;

/** The manuscript's asymptotic excess risk for the minimum-norm ridgeless fit,
 * plus the fresh-target noise. gamma = n/p; gamma = 1 is the singular boundary
 * and is returned as undefined rather than joined by a line. */
export function doubleDescentRisk(ratio) {
  if (!(checkFinite(ratio, 'the ratio n/p') > 0)) throw new RangeError('Use a positive ratio n/p.');
  if (Math.abs(ratio - 1) < 1e-12) {
    return { ratio, defined: false, biasSquared: null, varianceApprox: null, riskApprox: null, branch: 'boundary' };
  }
  const below = ratio < 1;
  const biasSquared = below ? (1 - ratio) ** 2 : 0;
  const varianceApprox = below
    ? ratio * (1 - ratio) + doubleDescentNoise * ratio / (1 - ratio)
    : doubleDescentNoise / (ratio - 1);
  return {
    ratio, defined: true, branch: below ? 'below' : 'above',
    biasSquared, varianceApprox,
    excess: biasSquared + varianceApprox,
    riskApprox: biasSquared + varianceApprox + doubleDescentNoise,
  };
}

/** One branch of the curve, sampled without ever touching the boundary. */
export function doubleDescentBranch(from, to, samples = 120) {
  if (!(from > 0) || !(to > from)) throw new RangeError('Sample an increasing positive interval.');
  if ((from < 1) !== (to < 1)) throw new RangeError('A branch must stay on one side of the interpolation boundary.');
  return Array.from({ length: samples }, (_, index) => {
    const ratio = from + (to - from) * index / (samples - 1);
    return doubleDescentRisk(Math.abs(ratio - 1) < 1e-9 ? from + (to - from) * (index + 0.5) / (samples - 1) : ratio);
  }).filter(point => point.defined);
}

/* ------------------------------------------------------------ fixtures */

export const fixtures = {
  /** Section 1's two teaching distributions. */
  sensorA: { predictions: [8, 10, 12], trueMean: 10, noiseSpread: 1 },
  sensorB: { predictions: [9, 9, 9], trueMean: 10, noiseSpread: 1 },
  /** Section 3's designs. */
  threeInputs: [-1, 0, 1],
  fiveInputs: [-1, -0.5, 0, 0.5, 1],
  worldsDefault: { trainX: [-1, 0, 1], curvature: 1, sigma: 0.5, probe: 0.5 },
  /** Section 7's constructed fixed design: six symmetric inputs and an
   * intercept, so n = 6, p = 2 and the column means are exactly zero. */
  optimismInputs: [-2.5, -1.5, -0.5, 0.5, 1.5, 2.5],
  optimismNoiseVariance: 4,
  /** Section 8's stated records. */
  brierA: { eta: 0.7, probabilities: [0.6] },
  brierB: { eta: 0.7, probabilities: [0.4, 0.8] },
  averaging: { v: 4, rho: 0.5, B: 4 },
  /** Section 9's printed ratios. */
  doubleDescentRatios: [0.1, 0.5, 0.8, 0.9, 0.99, 1.01, 1.1, 1.5, 2],
};

/** The fixed design of section 7, built once from its inputs. */
export function optimismDesign(inputs = fixtures.optimismInputs) {
  return inputs.map(value => [1, value]);
}
