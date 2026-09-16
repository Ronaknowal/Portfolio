/** Pure models for the Gaussian mixture lesson.
 *
 * Everything here is computed from the stated parameters: no fixture is
 * interpolated and no result is looked up. Densities are evaluated in log space
 * so that a distant observation keeps its component ratios instead of
 * underflowing both terms to zero. Natural logarithms throughout.
 *
 * A density is height per measurement unit, never a probability. A
 * responsibility is a probability under the fitted model with its parameters
 * held fixed, never a verified category probability.
 */

const LOG_TWO_PI = Math.log(2 * Math.PI);

export const limits = {
  observation: { minimum: -4, maximum: 4 },
  location: { minimum: -10, maximum: 10 },
  mean: { minimum: -3, maximum: 3 },
  weight: { minimum: 0.05, maximum: 0.95 },
  variance: { minimum: 0.25, maximum: 4 },
  correlation: { minimum: -0.9, maximum: 0.9 },
  point: { minimum: -3, maximum: 3 },
  floor: { minimum: 0.01, maximum: 1 },
  components: { minimum: 2, maximum: 2 },
  observations: { minimum: 4, maximum: 4 },
  emptyComponent: 1e-12,
};

const finite = (value, name) => {
  if (typeof value !== 'number' || !Number.isFinite(value)) throw new RangeError(`${name} must be a finite number.`);
  return value;
};
const within = (value, range, name) => {
  finite(value, name);
  if (value < range.minimum || value > range.maximum) {
    throw new RangeError(`Keep ${name} between ${range.minimum} and ${range.maximum}.`);
  }
  return value;
};

/** log N(x; mean, variance) for a positive variance. */
export function logNormal(x, mean, variance) {
  finite(x, 'the location');
  finite(mean, 'a mean');
  if (!(finite(variance, 'a variance') > 0)) throw new RangeError('A Gaussian needs a positive variance.');
  return -0.5 * (LOG_TWO_PI + Math.log(variance) + (x - mean) ** 2 / variance);
}
export const normal = (x, mean, variance) => Math.exp(logNormal(x, mean, variance));

/** All locations where two positive weighted Gaussian densities agree.
 * Empty roots can mean no crossing; only equal parameters AND weights give
 * equality everywhere. Keep both quadratic roots instead of hiding one. */
export function responsibilityCrossings(components) {
  const [first, second] = checkComponents(components);
  if (components.length !== 2 || first.weight <= 0 || second.weight <= 0) {
    throw new RangeError('Crossings require two positive-weight components.');
  }
  const constant = Math.log(first.weight / second.weight) - 0.5 * Math.log(first.variance / second.variance);
  const a = 1 / second.variance - 1 / first.variance;
  const b = 2 * (first.mean / first.variance - second.mean / second.variance);
  const c = second.mean ** 2 / second.variance - first.mean ** 2 / first.variance + 2 * constant;
  if (a === 0) return b === 0
    ? { everywhere: c === 0, roots: [] }
    : { everywhere: false, roots: [-c / b] };
  const discriminant = b * b - 4 * a * c;
  if (discriminant < 0) return { everywhere: false, roots: [] };
  if (discriminant === 0) return { everywhere: false, roots: [-b / (2 * a)] };
  // This form avoids subtracting nearly equal terms for one of the roots.
  const numerator = -0.5 * (b + (b < 0 ? -1 : 1) * Math.sqrt(discriminant));
  return { everywhere: false, roots: [numerator / a, c / numerator].sort((x, y) => x - y) };
}

/** Check a component list and return a defensive copy. */
export function checkComponents(components) {
  if (!Array.isArray(components) || components.length < 1) throw new RangeError('A mixture needs at least one component.');
  const total = components.reduce((sum, component) => sum + finite(component.weight, 'a weight'), 0);
  if (Math.abs(total - 1) > 1e-9) throw new RangeError('Mixing weights must sum to 1.');
  return components.map(component => {
    if (component.weight < 0) throw new RangeError('A mixing weight cannot be negative.');
    if (!(component.variance > 0)) throw new RangeError('Every component needs a positive variance.');
    return { weight: component.weight, mean: finite(component.mean, 'a mean'), variance: component.variance };
  });
}

/** The mixture at one location: what each component contributes, the total, and
 * the allocation that Bayes' rule reverses out of it. */
export function mixtureAt(components, x) {
  const parts = checkComponents(components);
  finite(x, 'the measurement');
  const logWeighted = parts.map(component =>
    (component.weight === 0 ? -Infinity : Math.log(component.weight) + logNormal(x, component.mean, component.variance)));
  const highest = Math.max(...logWeighted);
  if (!Number.isFinite(highest)) throw new RangeError('Every component has zero weight at this location.');
  const shifted = logWeighted.map(value => Math.exp(value - highest));
  const total = shifted.reduce((sum, value) => sum + value, 0);
  const logDensity = highest + Math.log(total);
  return {
    x,
    components: parts.map((component, index) => ({
      ...component,
      density: normal(x, component.mean, component.variance),
      weighted: Math.exp(logWeighted[index]),
      logWeighted: logWeighted[index],
      responsibility: Math.exp(logWeighted[index] - logDensity),
    })),
    density: Math.exp(logDensity),
    logDensity,
    negativeLogDensity: -logDensity,
    responsibilities: logWeighted.map(value => Math.exp(value - logDensity)),
  };
}

/** The E-step: every responsibility row, plus the observed log-likelihood the
 * current parameters achieve. Parameters do not move here. */
export function expectation(observations, components) {
  if (!Array.isArray(observations) || observations.length === 0) throw new RangeError('Give at least one observation.');
  const rows = observations.map(x => mixtureAt(components, finite(x, 'an observation')));
  return {
    responsibilities: rows.map(row => row.responsibilities),
    logDensities: rows.map(row => row.logDensity),
    logLikelihood: rows.reduce((sum, row) => sum + row.logDensity, 0),
    rows,
  };
}

/** The constrained M-step. The variance floor is part of the model: the exact
 * maximizer of the expected log-likelihood over variances at or above the floor
 * is the unconstrained scatter when feasible and the floor otherwise. */
export function maximization(observations, responsibilities, floor = 0.05) {
  if (!(finite(floor, 'the variance floor') > 0)) throw new RangeError('The variance floor must be positive.');
  const n = observations.length;
  if (responsibilities.length !== n) throw new RangeError('One responsibility row per observation.');
  const k = responsibilities[0].length;
  const counts = Array.from({ length: k }, (_, column) => responsibilities.reduce((sum, row) => sum + row[column], 0));
  return counts.map((count, column) => {
    if (count <= limits.emptyComponent) {
      throw new RangeError('This component received too little weight for this teaching update; choose another start or reset.');
    }
    const mean = observations.reduce((sum, x, index) => sum + responsibilities[index][column] * x, 0) / count;
    const scatter = observations.reduce((sum, x, index) => sum + responsibilities[index][column] * (x - mean) ** 2, 0) / count;
    return { weight: count / n, mean, variance: Math.max(scatter, floor), count, scatter, atFloor: scatter < floor };
  });
}

/** One complete cycle, keeping the responsibility matrix that the M-step
 * actually used rather than the fresh one computed afterwards. */
export function emCycle(observations, components, floor = 0.05) {
  const step = expectation(observations, components);
  const updated = maximization(observations, step.responsibilities, floor);
  const after = expectation(observations, updated);
  return {
    before: components,
    responsibilities: step.responsibilities,
    logLikelihoodBefore: step.logLikelihood,
    after: updated,
    logLikelihoodAfter: after.logLikelihood,
    gain: after.logLikelihood - step.logLikelihood,
    freshResponsibilities: after.responsibilities,
  };
}

/** Repeat cycles until the average log-likelihood change is small. Iteration 0
 * is the starting state, so the trace can be read as a history. */
export function emTrace(observations, components, floor = 0.05, maxIterations = 50, tolerance = 1e-8) {
  const start = expectation(observations, components);
  const history = [{
    iteration: 0,
    components: checkComponents(components),
    responsibilities: start.responsibilities,
    logLikelihood: start.logLikelihood,
  }];
  let current = components;
  let previous = start.logLikelihood;
  for (let iteration = 1; iteration <= maxIterations; iteration += 1) {
    const cycle = emCycle(observations, current, floor);
    current = cycle.after;
    history.push({
      iteration,
      components: cycle.after,
      responsibilities: cycle.freshResponsibilities,
      usedResponsibilities: cycle.responsibilities,
      logLikelihood: cycle.logLikelihoodAfter,
      gain: cycle.logLikelihoodAfter - previous,
    });
    if (Math.abs((cycle.logLikelihoodAfter - previous) / observations.length) < tolerance) break;
    previous = cycle.logLikelihoodAfter;
  }
  return history;
}

/** The evidence lower bound split into its two named pieces.
 *
 * Q is the expected complete-data log-likelihood under the supplied
 * responsibilities; H is their entropy. The bound is Q + H, and it equals the
 * observed log-likelihood exactly when the responsibilities came from these
 * same parameters. */
export function boundDecomposition(observations, components, responsibilities) {
  const parts = checkComponents(components);
  let qFunction = 0;
  let entropy = 0;
  observations.forEach((x, index) => {
    parts.forEach((component, column) => {
      const share = responsibilities[index][column];
      if (share <= 0) return;
      qFunction += share * (Math.log(component.weight) + logNormal(x, component.mean, component.variance));
      entropy -= share * Math.log(share);
    });
  });
  return { qFunction, entropy, elbo: qFunction + entropy };
}

/** The unconstrained collapse: one broad component holds the other rows while
 * the second narrows onto an observed location. */
export function collapseLogLikelihood(standardDeviation, observations = [-2, -1, 1, 2]) {
  if (!(finite(standardDeviation, 'the standard deviation') > 0)) throw new RangeError('Use a positive standard deviation.');
  return expectation(observations, [
    { weight: 0.5, mean: -2, variance: standardDeviation ** 2 },
    { weight: 0.5, mean: 0, variance: 4 },
  ]).logLikelihood;
}

/** Two-dimensional geometry for a unit-variance pair with one correlation. */
export function correlationGeometry(rho) {
  within(rho, limits.correlation, 'the correlation');
  const determinant = 1 - rho * rho;
  return {
    rho,
    covariance: [[1, rho], [rho, 1]],
    inverse: [[1 / determinant, -rho / determinant], [-rho / determinant, 1 / determinant]],
    determinant,
    // The two fixed diagonal directions are the eigenvectors for every non-zero
    // rho; at rho = 0 any orthogonal pair works, so we keep these by convention.
    axes: [
      { value: 1 + rho, direction: [Math.SQRT1_2, Math.SQRT1_2], semiaxis: Math.sqrt(1 + rho), label: 'along (1, 1)' },
      { value: 1 - rho, direction: [Math.SQRT1_2, -Math.SQRT1_2], semiaxis: Math.sqrt(1 - rho), label: 'along (1, −1)' },
    ],
    // A squared-Mahalanobis contour of 1 encloses this share of the mass in two
    // dimensions. The familiar 68% belongs to a one-dimensional interval.
    massInsideUnitContour: 1 - Math.exp(-0.5),
  };
}

/** Eigenvalues and orthonormal eigenvectors of a symmetric two-by-two matrix,
 * in closed form and ordered largest first.
 *
 * A drawn contour uses these directions, so an axis-aligned covariance must
 * return two different axes rather than one repeated direction. */
export function symmetricEigenpairs(covariance) {
  const [[a, b], [c, d]] = covariance;
  if (Math.abs(b - c) > 1e-12) throw new RangeError('This closed form needs a symmetric matrix.');
  const middle = (a + d) / 2;
  const spread = Math.sqrt(((a - d) / 2) ** 2 + b * b);
  const eigenvalues = [middle + spread, middle - spread];
  const directions = Math.abs(b) > 1e-12
    ? eigenvalues.map(value => {
      const vector = [b, value - a];
      const length = Math.hypot(vector[0], vector[1]);
      return [vector[0] / length, vector[1] / length];
    })
    // With no cross term the axes are the coordinate axes, paired with whichever
    // diagonal entry is larger.
    : (a >= d ? [[1, 0], [0, 1]] : [[0, 1], [1, 0]]);
  return { eigenvalues, directions };
}

/** Squared Mahalanobis distance from the origin under [[1, rho], [rho, 1]]. */
export function quadraticForm(point, rho) {
  const [first, second] = point.map(value => within(value, limits.point, 'a coordinate'));
  const geometry = correlationGeometry(rho);
  return (first * first + second * second - 2 * rho * first * second) / geometry.determinant;
}

/** The two-dimensional Gaussian density at a point, with its two named parts. */
export function planeDensity(point, rho) {
  const geometry = correlationGeometry(rho);
  const squared = quadraticForm(point, rho);
  const logDensity = -0.5 * squared - LOG_TWO_PI - 0.5 * Math.log(geometry.determinant);
  return { point, rho, squared, determinant: geometry.determinant, logDensity, density: Math.exp(logDensity) };
}

/** Compare two locations under one declared Gaussian: the determinant is shared,
 * so inside a panel the quadratic forms decide the order. */
export function planeComparison(first, second, rho) {
  const left = planeDensity(first, rho);
  const right = planeDensity(second, rho);
  const difference = left.logDensity - right.logDensity;
  return {
    left, right, difference,
    order: Math.abs(difference) <= 1e-10 ? 'equal' : difference > 0 ? 'first' : 'second',
    geometry: correlationGeometry(rho),
  };
}

/** Parameter counts for the four covariance families, means and free weights. */
export function parameterCount(kind, k, d) {
  if (!Number.isInteger(k) || k < 1) throw new RangeError('Use a positive component count.');
  if (!Number.isInteger(d) || d < 1) throw new RangeError('Use a positive dimension.');
  const covariance = {
    full: k * d * (d + 1) / 2,
    tied: d * (d + 1) / 2,
    diag: k * d,
    spherical: k,
  }[kind];
  if (covariance === undefined) throw new RangeError('Use full, tied, diag or spherical.');
  return { covariance, means: k * d, weights: k - 1, total: covariance + k * d + k - 1 };
}

/** AIC and BIC from a fitted total log-likelihood. Smaller is preferred, and
 * the two penalties differ once n exceeds e squared. */
export function criteria(logLikelihood, parameters, n) {
  finite(logLikelihood, 'the log-likelihood');
  if (!Number.isInteger(parameters) || parameters < 0) throw new RangeError('Use a whole parameter count.');
  if (!Number.isInteger(n) || n < 1) throw new RangeError('Use a positive observation count.');
  return { aic: -2 * logLikelihood + 2 * parameters, bic: -2 * logLikelihood + parameters * Math.log(n) };
}

/** The isotropic equal-weight responsibility that produces the k-means limit. */
export function sphericalResponsibility(x, means, variance) {
  if (!(finite(variance, 'the shared variance') > 0)) throw new RangeError('Use a positive shared variance.');
  const components = means.map(mean => ({ weight: 1 / means.length, mean, variance }));
  return mixtureAt(components, x).responsibilities;
}

/** Conditioning a two-component joint model on one observed coordinate. */
export function conditionalMixture(u, componentMeans = [[0, 0], [2, 4]], covariance = [[1, 0.5], [0.5, 1]]) {
  finite(u, 'the observed coordinate');
  const [[varianceU, covarianceUV], [, varianceV]] = covariance;
  if (!(varianceU > 0) || !(varianceV > 0)) throw new RangeError('Both marginal variances must be positive.');
  const marginal = componentMeans.map(mean => ({ weight: 1 / componentMeans.length, mean: mean[0], variance: varianceU }));
  const weights = mixtureAt(marginal, u).responsibilities;
  const conditionalVariance = varianceV - covarianceUV * covarianceUV / varianceU;
  const parts = componentMeans.map((mean, index) => ({
    weight: weights[index],
    mean: mean[1] + covarianceUV / varianceU * (u - mean[0]),
    variance: conditionalVariance,
  }));
  const mean = parts.reduce((sum, part) => sum + part.weight * part.mean, 0);
  const variance = parts.reduce((sum, part) => sum + part.weight * (part.variance + (part.mean - mean) ** 2), 0);
  return { u, weights, components: parts, mean, variance };
}

/** The mean and variance of a one-dimensional mixture: within-component spread
 * plus the spread between the component means. */
export function mixtureMoments(components) {
  const parts = checkComponents(components);
  const mean = parts.reduce((sum, component) => sum + component.weight * component.mean, 0);
  const variance = parts.reduce((sum, component) => sum + component.weight * (component.variance + (component.mean - mean) ** 2), 0);
  return { mean, variance };
}

export const presets = {
  twoBell: [{ weight: 0.5, mean: -2, variance: 1 }, { weight: 0.5, mean: 2, variance: 1 }],
  identical: [{ weight: 0.2, mean: 0, variance: 1 }, { weight: 0.8, mean: 0, variance: 1 }],
  observations: [-2, -1, 1, 2],
  start: [{ weight: 0.5, mean: -1, variance: 1 }, { weight: 0.5, mean: 1, variance: 1 }],
  identicalStart: [{ weight: 0.5, mean: 0, variance: 2.5 }, { weight: 0.5, mean: 0, variance: 2.5 }],
  asymmetricStart: [{ weight: 0.5, mean: -2, variance: 1 }, { weight: 0.5, mean: -1, variance: 1 }],
  repeated: [-2, -2, 2, 2],
  repeatedStart: [{ weight: 0.5, mean: -2, variance: 1 }, { weight: 0.5, mean: 2, variance: 1 }],
};

export { within as checkRange, finite as checkFinite };
