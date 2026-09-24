/** Pure models for the regularization lesson.
 *
 * Everything here is computed from the stated inputs. The penalised fits are the
 * entities: a coefficient path, a soft-threshold step and a penalised objective
 * are each solved exactly on the fixtures the manuscript works by hand, never
 * interpolated from a stored curve.
 *
 * One convention governs the regression calculations, matching the manuscript:
 *
 *   J(b, w) = (1/2n) SUM (y - b - x.w)^2 + lambda [ rho SUM |w| + (1-rho)/2 SUM w^2 ]
 *
 * The intercept is never penalised. Two local exceptions are declared where they
 * occur: the denoising example in section 8 uses an unaveraged data term, and
 * the two-dimensional geometry uses the normalized scalar form (1/2)||w - z||^2.
 *
 * Every entry point refuses input it cannot honour rather than substituting a
 * silent default: a non-finite number, a negative strength, a mixing fraction
 * outside [0, 1] or a ragged design all raise a RangeError.
 */

/* ------------------------------------------------------------------ guards */

export const limits = {
  preference: { minimum: -4, maximum: 4 },
  strength: { minimum: 0, maximum: 4 },
  ratio: { minimum: 0, maximum: 1 },
  entry: { minimum: -100, maximum: 100 },
  keep: { minimum: 0.1, maximum: 1 },
  contribution: { minimum: -10, maximum: 10 },
  target: { minimum: -20, maximum: 20 },
  rows: { minimum: 2, maximum: 4 },
  sweeps: { minimum: 1, maximum: 10000 },
  tolerance: 1e-10,
  /** Raw airfoil inputs: wider than the observed columns, still physical. */
  raw: [
    { minimum: 100, maximum: 25000 },
    { minimum: 0, maximum: 25 },
    { minimum: 0.01, maximum: 0.4 },
    { minimum: 10, maximum: 90 },
    { minimum: 0.0001, maximum: 0.08 },
  ],
};

export function checkFinite(value, name) {
  if (typeof value !== 'number' || !Number.isFinite(value)) {
    throw new RangeError(`${name} must be a finite number; handle missing observations before fitting.`);
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
function checkStrength(strength, ratio) {
  checkFinite(strength, 'the penalty strength');
  checkFinite(ratio, 'the mixing fraction');
  if (strength < 0) throw new RangeError('Use a nonnegative penalty strength.');
  if (ratio < 0 || ratio > 1) throw new RangeError('Use a mixing fraction between 0 and 1.');
  return { strength, ratio };
}

/* ------------------------------------------------ one coefficient at a time */

/** S(z, t) = sign(z) max(|z| - t, 0). An exact zero, not a small number. */
export function softThreshold(value, threshold) {
  checkFinite(value, 'the data preference');
  if (checkFinite(threshold, 'the threshold') < 0) throw new RangeError('A soft threshold cannot be negative.');
  const magnitude = Math.abs(value) - threshold;
  const endpointSlack = threshold > 0 ? 4 * Number.EPSILON * Math.max(Math.abs(value), threshold) : 0;
  if (magnitude <= endpointSlack) return 0;
  return Math.sign(value) * magnitude;
}

/** The exact minimizer of (1/2)(w - z)^2 + lambda [rho|w| + (1-rho)w^2/2],
 * with the threshold and denominator it came from. */
export function scalarSolution(preference, strength, ratio) {
  checkFinite(preference, 'the data preference');
  checkStrength(strength, ratio);
  const threshold = strength * ratio;
  const denominator = 1 + strength * (1 - ratio);
  const thresholded = softThreshold(preference, threshold);
  const coefficient = thresholded / denominator;
  return {
    preference, strength, ratio, threshold, denominator, thresholded, coefficient,
    // Exactly zero whenever the association sits inside the threshold interval.
    isZero: coefficient === 0,
    interval: [-threshold, threshold],
    // At an endpoint the coefficient is zero and the subgradient condition holds
    // with equality; that is not the same as a strict interior zero.
    atEndpoint: threshold > 0 && Math.abs(Math.abs(preference) - threshold)
      <= 4 * Number.EPSILON * Math.max(Math.abs(preference), threshold),
    sign: coefficient === 0 ? 0 : Math.sign(coefficient),
  };
}

/** The scalar objective and its two named parts at any w. */
export function scalarObjective(w, preference, strength, ratio) {
  checkFinite(w, 'the coefficient');
  checkFinite(preference, 'the data preference');
  checkStrength(strength, ratio);
  const data = 0.5 * (w - preference) ** 2;
  const penalty = strength * (ratio * Math.abs(w) + (1 - ratio) * w * w / 2);
  return { w, data, penalty, total: data + penalty };
}

/** The one-sided derivatives of that objective. At w = 0 with lambda*rho > 0 the
 * absolute value has a corner, so the two sides differ by 2*lambda*rho. */
export function scalarSlopes(w, preference, strength, ratio) {
  checkFinite(w, 'the coefficient');
  checkStrength(strength, ratio);
  const smooth = (w - preference) + strength * (1 - ratio) * w;
  const kink = strength * ratio;
  const fromLeft = smooth + (w === 0 ? -kink : kink * Math.sign(w));
  const fromRight = smooth + (w === 0 ? kink : kink * Math.sign(w));
  return {
    smooth,
    fromLeft,
    fromRight,
    // A minimum sits here exactly when zero lies between the two one-sided slopes.
    stationary: fromLeft <= 1e-12 && fromRight >= -1e-12,
  };
}

/* ---------------------------------------------- two coordinates, geometrically */

/** The mixed penalty measure, without lambda: rho*|w|_1 + (1-rho)*|w|^2/2. */
export function penaltyMeasure(weights, ratio) {
  checkFinite(ratio, 'the mixing fraction');
  if (ratio < 0 || ratio > 1) throw new RangeError('Use a mixing fraction between 0 and 1.');
  const values = weights.map((value, index) => checkFinite(value, `coordinate ${index + 1}`));
  const absolute = values.reduce((sum, value) => sum + Math.abs(value), 0);
  const squared = values.reduce((sum, value) => sum + value * value, 0);
  return ratio * absolute + (1 - ratio) * squared / 2;
}

/** The penalised solution for independent normalized coordinates, plus the
 * budget that makes the constrained picture the same problem. */
export function twoCoordinateSolution(preferences, strength, ratio) {
  if (!Array.isArray(preferences) || preferences.length !== 2) {
    throw new RangeError('This geometry needs exactly two coordinates.');
  }
  checkStrength(strength, ratio);
  const parts = preferences.map(value => scalarSolution(value, strength, ratio));
  const weights = parts.map(part => part.coefficient);
  const data = 0.5 * weights.reduce((sum, value, index) => sum + (value - preferences[index]) ** 2, 0);
  const budget = penaltyMeasure(weights, ratio);
  return {
    preferences, strength, ratio, parts, weights,
    data,
    penalty: strength * budget,
    objective: data + strength * budget,
    // The constrained twin uses this attained penalty value as its budget. A
    // numerical lambda is not a radius and is not comparable across families.
    budget,
    // The contact contour is the data-loss level set through the solution, so
    // the touching point is the arithmetic's answer, not a placed dot.
    contactRadius: Math.hypot(weights[0] - preferences[0], weights[1] - preferences[1]),
    zeroCoordinates: weights.map(value => value === 0),
  };
}

/** The exact boundary {w : penaltyMeasure(w, rho) = budget}, as a closed ray
 * length in each direction. For rho = 1 it is a diamond, for rho = 0 a circle,
 * and in between the nonsmooth corners on the axes survive. */
export function penaltyBoundaryRadius(direction, ratio, budget) {
  const [ux, uy] = direction;
  const length = Math.hypot(checkFinite(ux, 'a direction'), checkFinite(uy, 'a direction'));
  if (!(length > 0)) throw new RangeError('A direction cannot be the zero vector.');
  if (!(checkFinite(budget, 'the budget') > 0)) throw new RangeError('A penalty budget must be positive.');
  checkFinite(ratio, 'the mixing fraction');
  const x = ux / length;
  const y = uy / length;
  const absolute = Math.abs(x) + Math.abs(y);
  const quadratic = (1 - ratio) / 2;
  const linear = ratio * absolute;
  if (quadratic === 0) return budget / linear;
  if (linear === 0) return Math.sqrt(budget / quadratic);
  return (-linear + Math.sqrt(linear * linear + 4 * quadratic * budget)) / (2 * quadratic);
}

/** Sample that boundary as a closed polygon on one equal scale. */
export function penaltyBoundary(ratio, budget, samples = 241) {
  return Array.from({ length: samples }, (_, index) => {
    const angle = 2 * Math.PI * index / (samples - 1);
    const direction = [Math.cos(angle), Math.sin(angle)];
    const radius = penaltyBoundaryRadius(direction, ratio, budget);
    return [direction[0] * radius, direction[1] * radius];
  });
}

/* ------------------------------------------------------------- complete fits */

function checkDesign(X, y) {
  if (!Array.isArray(X) || X.length === 0) throw new RangeError('X must be a nonempty list of rows.');
  if (!Array.isArray(y) || y.length !== X.length) throw new RangeError('Give one target per row.');
  const width = X[0].length;
  if (!Number.isInteger(width) || width < 1) throw new RangeError('Every row needs at least one feature.');
  const rows = X.map((row, index) => {
    if (!Array.isArray(row) || row.length !== width) throw new RangeError('Every row needs the same number of features.');
    return row.map((value, column) => checkFinite(value, `feature ${column + 1} of row ${index}`));
  });
  const targets = y.map((value, index) => checkFinite(value, `the target of row ${index}`));
  return { rows, targets, n: rows.length, p: width };
}

/** Centre the design and the target on their own means, exactly as the fit does. */
export function centre(X, y) {
  const { rows, targets, n, p } = checkDesign(X, y);
  const featureMeans = Array.from({ length: p }, (_, column) => rows.reduce((sum, row) => sum + row[column], 0) / n);
  const targetMean = targets.reduce((sum, value) => sum + value, 0) / n;
  return {
    featureMeans, targetMean, n, p,
    Z: rows.map(row => row.map((value, column) => value - featureMeans[column])),
    t: targets.map(value => value - targetMean),
    rows, targets,
  };
}

/** Data curvature a_j = Z_j.Z_j / n for each column. */
export function curvatures(X, y) {
  const { Z, n, p } = centre(X, y);
  return Array.from({ length: p }, (_, column) => Z.reduce((sum, row) => sum + row[column] * row[column], 0) / n);
}

/** lambda_max: the smallest strength at which pure lasso leaves every slope zero. */
export function lambdaMax(X, y) {
  const { Z, t, n, p } = centre(X, y);
  return Math.max(...Array.from({ length: p }, (_, column) =>
    Math.abs(Z.reduce((sum, row, index) => sum + row[column] * t[index], 0) / n)));
}

/** The penalised objective at a given slope vector, with its named parts and the
 * intercept the centring implies. */
export function penalisedObjective(X, y, weights, strength, ratio) {
  const { Z, t, n, featureMeans, targetMean } = centre(X, y);
  checkStrength(strength, ratio);
  if (weights.length !== Z[0].length) throw new RangeError('Give one coefficient per feature.');
  const residual = t.map((value, index) => value - Z[index].reduce((sum, entry, column) => sum + entry * weights[column], 0));
  const data = residual.reduce((sum, value) => sum + value * value, 0) / (2 * n);
  const absolute = weights.reduce((sum, value) => sum + Math.abs(value), 0);
  const squared = weights.reduce((sum, value) => sum + value * value, 0);
  const penalty = strength * (ratio * absolute + (1 - ratio) * squared / 2);
  return {
    residual, data, penalty, total: data + penalty,
    meanSquaredError: residual.reduce((sum, value) => sum + value * value, 0) / n,
    intercept: targetMean - featureMeans.reduce((sum, mean, column) => sum + mean * weights[column], 0),
  };
}

/** Coordinate descent for the common objective, with an unpenalized intercept.
 *
 * The stopping rule is the optimality condition itself, not a small step: a
 * nonzero coordinate needs its smooth gradient plus lambda*rho*sign(w) to
 * vanish, and a zero coordinate needs its smooth gradient inside
 * [-lambda*rho, lambda*rho]. A run that reaches the sweep cap is returned
 * unconverged and labelled, never presented as an exact optimum.
 *
 * `order` is the complete deterministic coordinate order for one sweep; it is
 * part of the recorded contract, because a tied lasso problem returns a
 * different endpoint under a different order.
 */
export function coordinateFit(X, y, strength, ratio, options = {}) {
  const { featureMeans, targetMean, Z, t, n, p } = centre(X, y);
  checkStrength(strength, ratio);
  const tolerance = options.tolerance ?? limits.tolerance;
  const maxSweeps = options.maxSweeps ?? 10000;
  if (!(checkFinite(tolerance, 'the tolerance') > 0)) throw new RangeError('Use a positive tolerance.');
  if (!Number.isInteger(maxSweeps) || maxSweeps < 1) throw new RangeError('Use at least one sweep.');
  const order = options.order ?? Array.from({ length: p }, (_, column) => column);
  if (order.length !== p || new Set(order).size !== p || order.some(column => !Number.isInteger(column) || column < 0 || column >= p)) {
    throw new RangeError('The coordinate order must visit every column exactly once.');
  }
  const curvature = Array.from({ length: p }, (_, column) => Z.reduce((sum, row) => sum + row[column] * row[column], 0) / n);
  const weights = new Array(p).fill(0);
  let residual = t.slice();
  const history = [];
  let converged = false;
  let sweep = 0;
  let violation = Infinity;
  while (sweep < maxSweeps && !converged) {
    sweep += 1;
    const steps = [];
    for (const column of order) {
      const partial = residual.map((value, index) => value + Z[index][column] * weights[column]);
      const association = Z.reduce((sum, row, index) => sum + row[column] * partial[index], 0) / n;
      const denominator = curvature[column] + strength * (1 - ratio);
      const threshold = strength * ratio;
      // A constant centred column has zero curvature and no association.
      // A positive L1 penalty uniquely selects zero even without a quadratic
      // term. Only a completely unpenalized coordinate is flat.
      const next = denominator > 0 ? softThreshold(association, threshold) / denominator : 0;
      weights[column] = next;
      residual = partial.map((value, index) => value - Z[index][column] * next);
      steps.push({
        column, partial, association, curvature: curvature[column], threshold, denominator,
        weight: next, flat: denominator === 0 && threshold === 0,
        residual: residual.slice(),
      });
    }
    const gradient = Array.from({ length: p }, (_, column) =>
      -Z.reduce((sum, row, index) => sum + row[column] * residual[index], 0) / n + strength * (1 - ratio) * weights[column]);
    const violations = gradient.map((value, column) => (weights[column] !== 0
      ? Math.abs(value + strength * ratio * Math.sign(weights[column]))
      : Math.max(Math.abs(value) - strength * ratio, 0)));
    violation = violations.length ? Math.max(...violations) : 0;
    const objective = penalisedObjective(X, y, weights, strength, ratio);
    history.push({
      sweep, steps, weights: weights.slice(), gradient, violations,
      kktResidual: violation, objective: objective.total, data: objective.data, penalty: objective.penalty,
      meanSquaredError: objective.meanSquaredError,
    });
    if (violation <= tolerance) converged = true;
  }
  const final = penalisedObjective(X, y, weights, strength, ratio);
  return {
    weights: weights.slice(),
    intercept: targetMean - featureMeans.reduce((sum, mean, column) => sum + mean * weights[column], 0),
    featureMeans, targetMean, curvature, order,
    sweeps: sweep, converged, kktResidual: violation, tolerance, maxSweeps,
    history,
    objective: final.total, data: final.data, penalty: final.penalty,
    meanSquaredError: final.meanSquaredError, residual: final.residual,
    fitted: X.map(row => (targetMean - featureMeans.reduce((sum, mean, column) => sum + mean * weights[column], 0))
      + row.reduce((sum, value, column) => sum + value * weights[column], 0)),
    lambdaMax: lambdaMax(X, y),
  };
}

/* ------------------------------------------------------- duplicate columns */

/** Two identical centred columns: the prediction depends only on the sum, so the
 * lasso minimizer is a segment while ridge and elastic net are single points. */
export function duplicateAnalysis(strength, ratio, association = 2) {
  checkStrength(strength, ratio);
  checkFinite(association, 'the association');
  // Data loss is (1/2)(w1 + w2 - association)^2 on the two-row fixture.
  // Along a fixed sum s the L1 cost is constant for nonnegative pairs, while the
  // squared cost is smallest at the balanced point.
  const lassoSum = softThreshold(association, strength * ratio) / (1 + strength * (1 - ratio) / 2);
  const balanced = lassoSum / 2;
  const pure = ratio === 1;
  return {
    strength, ratio, association,
    sum: lassoSum,
    balanced: [balanced, balanced],
    endpoints: [[lassoSum, 0], [0, lassoSum]],
    // Every nonnegative allocation of the sum ties only when the penalty has no
    // strictly convex part. Otherwise the balanced point is the unique answer.
    segment: pure,
    unique: !pure,
    minimizers: pure ? [[lassoSum, 0], [lassoSum / 2, lassoSum / 2], [0, lassoSum]] : [[balanced, balanced]],
    objective: allocation => {
      const [first, second] = allocation;
      const data = 0.5 * (first + second - association) ** 2;
      const penalty = strength * (ratio * (Math.abs(first) + Math.abs(second))
        + (1 - ratio) * (first * first + second * second) / 2);
      return { data, penalty, total: data + penalty };
    },
  };
}

/* --------------------------------------------------------------- dropout */

/** Every mask of two contributions, with its actual probability.
 *
 * The masks are enumerated, not sampled: there is no seed and no realisation
 * here, and the expectation is exact. */
export function dropoutEnumeration(inputs, weights, target, keep) {
  if (inputs.length !== 2 || weights.length !== 2) throw new RangeError('This enumeration takes exactly two contributions.');
  inputs.forEach((value, index) => checkFinite(value, `input ${index + 1}`));
  weights.forEach((value, index) => checkFinite(value, `coefficient ${index + 1}`));
  checkFinite(target, 'the target');
  if (!(checkFinite(keep, 'the keep probability') > 0) || keep > 1) {
    throw new RangeError('The keep probability must be greater than 0 and at most 1.');
  }
  const branches = [[0, 0], [0, 1], [1, 0], [1, 1]].map(mask => {
    const probability = mask.reduce((product, kept) => product * (kept ? keep : 1 - keep), 1);
    const scaled = mask.map((kept, index) => inputs[index] * kept / keep);
    const prediction = scaled.reduce((sum, value, index) => sum + value * weights[index], 0);
    const halfSquaredLoss = (target - prediction) ** 2 / 2;
    return { mask, probability, scaled, prediction, halfSquaredLoss, possible: probability > 0 };
  });
  const cleanPrediction = inputs.reduce((sum, value, index) => sum + value * weights[index], 0);
  const cleanLoss = (target - cleanPrediction) ** 2 / 2;
  const expectedPrediction = branches.reduce((sum, branch) => sum + branch.probability * branch.prediction, 0);
  const expectedLoss = branches.reduce((sum, branch) => sum + branch.probability * branch.halfSquaredLoss, 0);
  const analyticExtra = (1 - keep) / (2 * keep)
    * inputs.reduce((sum, value, index) => sum + (weights[index] * value) ** 2, 0);
  return {
    inputs, weights, target, keep, branches,
    cleanPrediction, cleanLoss, expectedPrediction, expectedLoss,
    extraLoss: expectedLoss - cleanLoss,
    analyticExtra,
    // The two must agree exactly for this linear, squared-loss, independent-mask
    // setting; that agreement is the point of the section.
    agrees: Math.abs(expectedLoss - cleanLoss - analyticExtra) <= 1e-12 * Math.max(1, Math.abs(analyticExtra)),
  };
}

/** An average input through a nonlinearity is not the average of the outputs. */
export function nonlinearInset(values = [0, 2], shift = 1) {
  const rectify = u => Math.max(0, u - shift);
  const meanInput = values.reduce((sum, value) => sum + value, 0) / values.length;
  return {
    values, shift,
    outputs: values.map(rectify),
    meanOutput: values.reduce((sum, value) => sum + rectify(value), 0) / values.length,
    meanInput,
    outputOfMean: rectify(meanInput),
  };
}

/* ------------------------------------------------- directions and filters */

/** Ridge's fitted-data multiplier for one singular value at a = n*lambda. */
export function ridgeFilter(singularValue, a) {
  if (!(checkFinite(singularValue, 'a singular value') >= 0)) throw new RangeError('A singular value cannot be negative.');
  if (!(checkFinite(a, 'n times lambda') >= 0)) throw new RangeError('Use a nonnegative n*lambda.');
  const squared = singularValue * singularValue;
  if (squared + a === 0) throw new RangeError('A zero singular value with no penalty leaves this direction undetermined.');
  return {
    singularValue, a,
    dataMultiplier: squared / (squared + a),
    coefficientMultiplier: singularValue / (squared + a),
  };
}

/** Gradient descent's filter after t steps, and the single ridge strength that
 * would match it for this one eigenvalue. */
export function earlyStoppingFilter(eigenvalue, stepSize, steps) {
  if (!(checkFinite(eigenvalue, 'a Gram eigenvalue') > 0)) throw new RangeError('Use a positive Gram eigenvalue.');
  if (!(checkFinite(stepSize, 'the step size') > 0)) throw new RangeError('Use a positive step size.');
  if (!Number.isInteger(steps) || steps < 1) throw new RangeError('Use a whole number of steps.');
  const factor = 1 - (1 - stepSize * eigenvalue) ** steps;
  return {
    eigenvalue, stepSize, steps, factor,
    matchingStrength: factor === 0 ? Infinity : eigenvalue * (1 - factor) / factor,
    stable: Math.abs(1 - stepSize * eigenvalue) <= 1,
  };
}

/* -------------------------------------------- a penalty encodes a representation */

/** min over p of (1/2)(p - 1)^2 + 2*lambda*|p|, and the balanced factors. */
export function factorOptimum(strength) {
  if (!(checkFinite(strength, 'the penalty strength') >= 0)) throw new RangeError('Use a nonnegative penalty strength.');
  const product = Math.max(1 - 2 * strength, 0);
  const magnitude = Math.sqrt(product);
  const data = 0.5 * (product - 1) ** 2;
  const penalty = 2 * strength * product;
  return {
    strength, product, magnitude,
    factors: [magnitude, magnitude],
    data, penalty, total: data + penalty,
    balancedZeroLoss: { factors: [1, 1], data: 0, penalty: 2 * strength, total: 2 * strength },
    // Every ab = 1 pair minimises only when there is no penalty at all.
    degenerate: strength === 0,
    cost: (a, b) => {
      checkFinite(a, 'the first factor');
      checkFinite(b, 'the second factor');
      return { data: 0.5 * (a * b - 1) ** 2, penalty: strength * (a * a + b * b), total: 0.5 * (a * b - 1) ** 2 + strength * (a * a + b * b) };
    },
  };
}

/** Solve a small symmetric positive-definite system by Gaussian elimination with
 * partial pivoting. Used for the three-position denoising example. */
export function solveLinear(matrix, vector) {
  const n = vector.length;
  if (matrix.length !== n || matrix.some(row => row.length !== n)) throw new RangeError('The system must be square.');
  const a = matrix.map((row, index) => [...row.map(value => checkFinite(value, 'a matrix entry')), checkFinite(vector[index], 'a right-hand value')]);
  for (let column = 0; column < n; column += 1) {
    let pivot = column;
    for (let row = column + 1; row < n; row += 1) if (Math.abs(a[row][column]) > Math.abs(a[pivot][column])) pivot = row;
    if (Math.abs(a[pivot][column]) < 1e-12) throw new RangeError('This system is singular; no unique solution exists.');
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

/** The three-position denoising comparison, under the declared UNAVERAGED data
 * objective (1/2)||y - w||^2 + (lambda/2)||Lw||^2. */
export function smoothnessComparison(signal, strength) {
  if (!Array.isArray(signal) || signal.length !== 3) throw new RangeError('This example uses exactly three positions.');
  signal.forEach((value, index) => checkFinite(value, `position ${index + 1}`));
  if (!(checkFinite(strength, 'the penalty strength') >= 0)) throw new RangeError('Use a nonnegative penalty strength.');
  const L = [[-1, 1, 0], [0, -1, 1]];
  const gram = [0, 1, 2].map(row => [0, 1, 2].map(column =>
    L.reduce((sum, line) => sum + line[row] * line[column], 0)));
  const difference = solveLinear(
    [0, 1, 2].map(row => [0, 1, 2].map(column => (row === column ? 1 : 0) + strength * gram[row][column])),
    signal,
  );
  const identity = signal.map(value => value / (1 + strength));
  const edges = weights => [weights[1] - weights[0], weights[2] - weights[1]];
  return {
    signal, strength, L, gram,
    difference, identity,
    differenceEdges: edges(difference), identityEdges: edges(identity), signalEdges: edges(signal),
    system: [0, 1, 2].map(row => [0, 1, 2].map(column => (row === column ? 1 : 0) + strength * gram[row][column])),
    convention: 'unaveraged data term',
  };
}

/* ------------------------------------------------ complexity accounts */

/** AIC and BIC from a maximized natural-log likelihood. Smaller is preferred. */
export function criteria(logLikelihood, parameters, n) {
  checkFinite(logLikelihood, 'the log-likelihood');
  if (!Number.isInteger(parameters) || parameters < 0) throw new RangeError('Use a whole parameter count.');
  if (!Number.isInteger(n) || n < 1) throw new RangeError('Use a positive observation count.');
  return {
    logLikelihood, parameters, n,
    fit: -2 * logLikelihood,
    aicPenalty: 2 * parameters,
    bicPenalty: parameters * Math.log(n),
    aic: -2 * logLikelihood + 2 * parameters,
    bic: -2 * logLikelihood + parameters * Math.log(n),
  };
}

/** The declared sixteen-bit two-part code from section 9.
 *
 * Mode 0: flag 0 then sixteen literal bits, seventeen in all.
 * Mode 1: flag 1 then a four-bit pattern the receiver repeats four times, five
 * in all. The pattern costs four bits because it was chosen after seeing the
 * data. Both parties already know the length and the repeat rule. */
export function encodeSixteenBits(message) {
  if (typeof message !== 'string' || message.length !== 16 || /[^01]/.test(message)) {
    throw new RangeError('This declared code takes exactly sixteen bits, each 0 or 1.');
  }
  const pattern = message.slice(0, 4);
  const repeats = pattern.repeat(4) === message;
  return {
    message, pattern: repeats ? pattern : null, repeats,
    mode: repeats ? 1 : 0,
    payload: repeats ? pattern : message,
    flagBits: 1,
    payloadBits: repeats ? 4 : 16,
    totalBits: repeats ? 5 : 17,
    decoded: repeats ? pattern.repeat(4) : message,
  };
}

/* ------------------------------------------- the saved airfoil ridge model */

/** The twenty degree-two terms of five raw measurements, in the exact
 * PolynomialFeatures(degree=2, include_bias=False) order: the five originals,
 * then every product x_i * x_j with i <= j. */
export function polynomialTerms(raw) {
  if (!Array.isArray(raw) || raw.length !== 5) throw new RangeError('The airfoil model takes exactly five raw measurements.');
  const values = raw.map((value, index) => checkRange(value, limits.raw[index], `measurement ${index + 1}`));
  const terms = [...values];
  for (let i = 0; i < 5; i += 1) for (let j = i; j < 5; j += 1) terms.push(values[i] * values[j]);
  return terms;
}

/** Which of the twenty terms a given raw measurement enters. */
export function termsTouchedBy(index) {
  if (!Number.isInteger(index) || index < 0 || index > 4) throw new RangeError('Use a raw measurement index from 0 to 4.');
  const touched = [index];
  let position = 5;
  for (let i = 0; i < 5; i += 1) for (let j = i; j < 5; j += 1) {
    if (i === index || j === index) touched.push(position);
    position += 1;
  }
  return touched;
}

/** Trace one observation through the saved fitted model: raw measurements to
 * polynomial terms to saved standardization to signed contributions in decibels.
 * Nothing is fitted here; the parameters are read, never updated. */
export function modelPrediction(model, raw) {
  const { intercept, coefficients, scaleMean, scaleScale } = model;
  if (coefficients.length !== 20 || scaleMean.length !== 20 || scaleScale.length !== 20) {
    throw new RangeError('The saved model needs twenty coefficients, means and scales.');
  }
  const terms = polynomialTerms(raw);
  const standardized = terms.map((value, index) => {
    if (!(scaleScale[index] > 0)) throw new RangeError('A saved scale must be positive.');
    return (value - scaleMean[index]) / scaleScale[index];
  });
  const contributions = standardized.map((value, index) => value * coefficients[index]);
  return {
    raw, terms, standardized, contributions, intercept,
    prediction: contributions.reduce((sum, value) => sum + value, intercept),
    total: contributions.reduce((sum, value) => sum + value, 0),
  };
}

/* ------------------------------------------------------------ shared fixtures */

export const fixtures = {
  /** The four constructed rows of section 3. Their means are zero, Z'Z/n = I
   * and Z't/n = (3, 0.4). */
  orthogonal: { X: [[1, 1], [1, -1], [-1, 1], [-1, -1]], y: [3.4, 2.6, -2.6, -3.4] },
  /** Row 0's target moved from 3.4 to 7.4: a feature-specific association. */
  changedTarget: { X: [[1, 1], [1, -1], [-1, 1], [-1, -1]], y: [7.4, 2.6, -2.6, -3.4] },
  /** Every target raised by seven: a changed measurement origin. */
  shiftedTargets: { X: [[1, 1], [1, -1], [-1, 1], [-1, -1]], y: [10.4, 9.6, 4.4, 3.6] },
  /** Two rows of duplicated sensors. */
  duplicate: { X: [[-1, -1], [1, 1]], y: [-2, 2] },
  /** A constant second column: no data curvature; penalty determines uniqueness. */
  constantColumn: { X: [[1, 2], [1, 2], [-1, 2], [-1, 2]], y: [3, 3, -3, -3] },
  preferences: [3, 0.4],
  dropout: { inputs: [2, 1], weights: [1, -1], target: 1, keep: 0.5 },
  practiceDropout: { inputs: [1, 2], weights: [2, 0], target: 1, keep: 0.75 },
  singularValues: [4, 0.5],
  denoising: [0, 2, 0],
  shiftedDenoising: [3, 5, 3],
  criteriaRecords: [
    { model: 'Smaller', logLikelihood: -150, parameters: 3 },
    { model: 'Larger', logLikelihood: -146, parameters: 5 },
  ],
  messages: ['0101010101010101', '0101010001010101', '1110111011101110'],
};
