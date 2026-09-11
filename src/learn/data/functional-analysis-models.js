/** Finite, deterministic teaching models; ranges are intentional arithmetic contracts. */
function bounded(value, minimum, maximum, label) {
  if (typeof value !== 'number' || !Number.isFinite(value) || value < minimum || value > maximum) {
    throw new RangeError(`${label} must be a finite number from ${minimum} to ${maximum}.`);
  }
  return value;
}
export function functionalNumber(value) {
  if (!Number.isFinite(value)) return String(value);
  if (value === 0) return '0';
  if (Math.abs(value) < 0.0001 || Math.abs(value) >= 100000) return value.toExponential(3);
  return Number(value.toFixed(5)).toString();
}
export function spikeState(epsilon = 0.125, height = 1) {
  bounded(epsilon, 0.005, 0.45, 'Half-width');
  bounded(height, 0, 3, 'Height');
  return {
    epsilon,
    height,
    center: 0.5,
    squaredSize: 2 * epsilon * height ** 2 / 3,
    slopeEnergy: 2 * height ** 2 / epsilon,
    maximum: height,
    value: t => height * Math.max(0, 1 - Math.abs(t - 0.5) / epsilon)
  };
}
export const slopeBreaks = [0, 0.25, 0.5, 0.75, 1];
export function integralSpaceState(slopes = [2, -1, 1, 0], query = 0.6) {
  if (!Array.isArray(slopes) || slopes.length !== 4) throw new RangeError('Supply exactly four slopes.');
  slopes.forEach(value => bounded(value, -4, 4, 'Slope'));
  bounded(query, 0, 1, 'Query');
  const intervals = slopes.map((slope, i) => {
    const left = slopeBreaks[i];
    const right = slopeBreaks[i + 1];
    const overlap = Math.max(0, Math.min(query, right) - left);
    return {
      left,
      right,
      slope,
      overlap,
      contribution: slope * overlap,
      energy: slope ** 2 * (right - left)
    };
  });
  const squaredNorm = intervals.reduce((sum, item) => sum + item.energy, 0);
  const valueAt = t => intervals.reduce((sum, item) => sum + item.slope * Math.max(0, Math.min(t, item.right) - item.left), 0);
  return {
    query,
    slopes: [...slopes],
    intervals,
    squaredNorm,
    evaluation: valueAt(query),
    bound: Math.sqrt(query * squaredNorm),
    valueAt
  };
}
export function kernelValidityState(kind = 'polynomial') {
  let features;
  let labels;
  if (kind === 'polynomial') {
    labels = ['−1', '0', '1'];
    features = [-1, 0, 1].map(x => [1, Math.sqrt(2) * x, x * x]);
  } else if (kind === 'bigrams') {
    labels = ['ABA', 'BAB', 'ABAB'];
    features = [[1, 1], [1, 1], [2, 1]];
  } else if (kind !== 'invalid') {
    throw new RangeError('Choose polynomial, bigrams or invalid.');
  }
  const gram = kind === 'invalid' ? [[1, 0.9, 0], [0.9, 1, 0.9], [0, 0.9, 1]] : features.map(a => features.map(b => a.reduce((sum, value, i) => sum + value * b[i], 0)));
  const coefficients = [1, -1, 1];
  const contributions = gram.map((row, i) => row.map((value, j) => value * coefficients[i] * coefficients[j]));
  const quadratic = contributions.flat().reduce((sum, value) => sum + value, 0);
  return {
    kind,
    labels: labels ?? ['A', 'B', 'C'],
    features: features ?? null,
    gram,
    coefficients,
    contributions,
    quadratic,
    valid: kind !== 'invalid'
  };
}
export function representerState(amplitude = 0.5, extraObservation = false) {
  bounded(amplitude, -1.5, 1.5, 'Wiggle amplitude');
  if (typeof extraObservation !== 'boolean') throw new TypeError('Extra observation must be boolean.');
  const baseline = t => t <= 0.5 ? 2 * t : 2 - 2 * t;
  const wiggle = t => t < 0.25 ? 4 * amplitude * t : t <= 0.5 ? 4 * amplitude * (0.5 - t) : 0;
  return {
    amplitude,
    baseline,
    wiggle,
    total: t => baseline(t) + wiggle(t),
    baselineEnergy: 4,
    addedEnergy: 8 * amplitude ** 2,
    totalEnergy: 4 + 8 * amplitude ** 2,
    observations: extraObservation ? [0.25, 0.5, 1] : [0.5, 1],
    extraObservation,
    extraResidual: extraObservation ? amplitude : 0,
    derivativeSlopes: [2 + 4 * amplitude, 2 - 4 * amplitude, -2]
  };
}
function solvePositiveDefinite(matrix, values) {
  const n = values.length;
  const lower = Array.from({
    length: n
  }, () => Array(n).fill(0));
  for (let row = 0; row < n; row += 1) {
    for (let col = 0; col <= row; col += 1) {
      let value = matrix[row][col];
      for (let k = 0; k < col; k += 1) value -= lower[row][k] * lower[col][k];
      if (row === col) {
        if (!(value > 0) || !Number.isFinite(value)) throw new RangeError('The shifted system exceeds this solver’s arithmetic range.');
        lower[row][col] = Math.sqrt(value);
      } else lower[row][col] = value / lower[col][col];
    }
  }
  const forward = values.slice();
  for (let i = 0; i < n; i += 1) {
    for (let j = 0; j < i; j += 1) forward[i] -= lower[i][j] * forward[j];
    forward[i] /= lower[i][i];
  }
  const result = forward.slice();
  for (let i = n - 1; i >= 0; i -= 1) {
    for (let j = i + 1; j < n; j += 1) result[i] -= lower[j][i] * result[j];
    result[i] /= lower[i][i];
  }
  return result;
}
export const ridgeFixtures = {
  original: {
    xs: [0, 1],
    ys: [0, 1],
    validation: [[0.5, 0.5]]
  },
  curve: {
    xs: [0, 0.25, 0.5, 0.75, 1],
    ys: [0.1, 0.8, 1.1, 0.6, -0.1],
    validation: [[0.125, 0.4375], [0.375, 0.9375], [0.625, 0.9375], [0.875, 0.4375]]
  },
  duplicates: {
    xs: [0, 0.5, 0.5, 1],
    ys: [0, 0.5, 1.5, 0],
    validation: [[0.25, 0.75], [0.75, 0.75]]
  }
};
export function kernelRidgeState(gamma = 1, lambda = 0.05, query = 0.5, fixture = 'original') {
  bounded(gamma, 0.05, 32, 'Gamma');
  bounded(lambda, 0.0001, 2, 'Average-loss lambda');
  bounded(query, -0.25, 1.25, 'Query');
  if (!Object.hasOwn(ridgeFixtures, fixture)) throw new RangeError('Unknown data fixture.');
  const {
    xs,
    ys,
    validation
  } = ridgeFixtures[fixture];
  const n = xs.length;
  const kernel = (a, b) => Math.exp(-gamma * (a - b) ** 2);
  const gram = xs.map(a => xs.map(b => kernel(a, b)));
  const shift = n * lambda;
  const shifted = gram.map((row, i) => row.map((value, j) => value + (i === j ? shift : 0)));
  const alpha = solvePositiveDefinite(shifted, ys);
  const predict = t => xs.reduce((sum, value, i) => sum + alpha[i] * kernel(value, t), 0);
  const contributions = xs.map((value, i) => alpha[i] * kernel(value, query));
  const residuals = xs.map((value, i) => predict(value) - ys[i]);
  // This finite Gaussian kernel is PSD; use its explicit Gaussian feature series
  // for a nonnegative norm sum, avoiding cancellation in alpha^T K alpha.
  // exp(-gamma*x²) (sqrt(2gamma)*x)^m/sqrt(m!), with m up to192.
  let features = xs.map(value => Math.exp(-gamma * value ** 2));
  let squaredNorm = 0;
  for (let degree = 0; degree <= 192; degree += 1) {
    const coordinate = features.reduce((sum, value, i) => sum + alpha[i] * value, 0);
    squaredNorm += coordinate ** 2;
    features = features.map((value, i) => value * Math.sqrt(2 * gamma / (degree + 1)) * xs[i]);
  }
  const trainMse = residuals.reduce((sum, value) => sum + value ** 2, 0) / n;
  const validationMse = validation.reduce((sum, [x, y]) => sum + (predict(x) - y) ** 2, 0) / validation.length;
  return {
    gamma,
    lambda,
    query,
    fixture,
    xs,
    ys,
    validation,
    gram,
    shift,
    alpha,
    predict,
    contributions,
    prediction: predict(query),
    residuals,
    trainMse,
    validationMse,
    squaredNorm,
    objective: trainMse + lambda * squaredNorm,
    normMethod: 'Nonnegative Gaussian feature sum through degree192; oracle-checked on these bounded fixtures.'
  };
}
export function distributionEmbeddingState(gamma = 1, sameLaw = false) {
  bounded(gamma, 0.05, 8, 'Witness gamma');
  if (typeof sameLaw !== 'boolean') throw new TypeError('Same-law choice must be boolean.');
  const support = [-2, -1, 0, 1, 2];
  const p = [0, 0.5, 0, 0.5, 0];
  const q = sameLaw ? p.slice() : [0.125, 0, 0.75, 0, 0.125];
  const difference = p.map((value, i) => value - q[i]);
  const witness = t => support.reduce((sum, x, i) => sum + difference[i] * Math.exp(-gamma * (t - x) ** 2), 0);
  const squaredMmd = sameLaw ? 0 : difference.reduce((sum, value, i) => sum + value * witness(support[i]), 0);
  if (squaredMmd < 0 || !Number.isFinite(squaredMmd)) throw new RangeError('Discrepancy exceeds the supported arithmetic range.');
  const moments = weights => [0, 1, 2, 4].map(power => support.reduce((sum, x, i) => sum + weights[i] * x ** power, 0));
  return {
    gamma,
    sameLaw,
    support,
    p,
    q,
    difference,
    witness,
    squaredMmd,
    pMoments: moments(p),
    qMoments: moments(q),
    mmd: Math.sqrt(squaredMmd),
    normalizedWitness: t => squaredMmd === 0 ? 0 : witness(t) / Math.sqrt(squaredMmd)
  };
}
export function kernelQuadratureState(node = 0.5, weight = null) {
  bounded(node, 0, 1, 'Quadrature node');
  if (weight !== null) bounded(weight, -1, 2, 'Quadrature weight');
  const optimal = node === 0 ? 0 : 1 - node / 2;
  const chosen = weight === null ? optimal : weight;
  const mean = t => t - t * t / 2;
  const residual = t => mean(t) - chosen * Math.min(node, t);
  // Exact integrals of (1-t-weight)^2 on[0,node] and (1-t)^2 on[node,1].
  // Midpoint mean-square + interval variance is nonnegative term by term.
  const first = node * ((1 - node / 2 - chosen) ** 2 + node ** 2 / 12);
  const second = (1 - node) ** 3 / 3;
  const squaredError = first + second;
  return {
    node,
    weight: chosen,
    optimalWeight: optimal,
    mean,
    residual,
    squaredError,
    error: Math.sqrt(squaredError),
    estimateLinear: chosen * node,
    actualLinear: 0.5,
    linearError: Math.abs(0.5 - chosen * node),
    worstCase: t => residual(t) / Math.sqrt(squaredError),
    endpointNote: node === 0 ? 'Every function is zero at the anchor; this observation supplies no integral information. Any weight has the same error.' : null
  };
}
