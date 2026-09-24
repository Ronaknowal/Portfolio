// Small exact Gaussian conditioning; no browser fitting or external ML runtime.
export const normal95 = 1.959963984540054;
export const spatialGrid = Array.from({ length: 81 }, (_, i) => -1 + i / 16);
export const initialObservations = [{ id: 1, x: 0, y: 1 }, { id: 2, x: 2, y: -1 }];

function finite(value, name) {
  if (!Number.isFinite(value)) throw new Error(`${name} must be finite.`);
  return value;
}

export function spatialKernel(kind = 'rbf', length = 1) {
  if (!(length > 0) || !Number.isFinite(length)) throw new Error('Length must be positive.');
  if (!['rbf', 'independent'].includes(kind)) throw new Error('Unknown kernel.');
  return kind === 'independent'
    ? (x, z) => Number(x === z)
    : (x, z) => Math.exp(-0.5 * ((x - z) / length) ** 2);
}

export function cholesky(matrix) {
  const lower = matrix.map(row => row.map(() => 0));
  for (let i = 0; i < matrix.length; i++) {
    for (let j = 0; j <= i; j++) {
      let value = matrix[i][j];
      for (let k = 0; k < j; k++) value -= lower[i][k] * lower[j][k];
      if (i === j) {
        if (!(value > 0) || !Number.isFinite(value)) {
          throw new Error('Covariance is not positive definite. Check noise, duplicate inputs and kernel assumptions.');
        }
        lower[i][j] = Math.sqrt(value);
      } else lower[i][j] = value / lower[j][j];
    }
  }
  return lower;
}

function forward(lower, vector) {
  return vector.reduce((out, value, i) => {
    out.push((value - out.reduce((sum, v, j) => sum + lower[i][j] * v, 0)) / lower[i][i]);
    return out;
  }, []);
}

function backward(lower, vector) {
  const out = Array(vector.length).fill(0);
  for (let i = vector.length - 1; i >= 0; i--) {
    let value = vector[i];
    for (let j = i + 1; j < out.length; j++) value -= lower[j][i] * out[j];
    out[i] = value / lower[i][i];
  }
  return out;
}

const dot = (a, b) => a.reduce((sum, value, i) => sum + value * b[i], 0);

// One-entry factor cache per mounted investigation; values do not affect it.
export function createConditioner() {
  let previousKey = null;
  let previousFactor = null;
  let factorizations = 0;
  return {
    get factorizations() { return factorizations; },
    predict({ x, y, targets, noise = 0.25, kernel = spatialKernel(), kernelKey = 'rbf:1', center = 0 }) {
      if (x.length !== y.length || x.length > 96 || targets.length > 96) throw new Error('Invalid bounded regression shape.');
      [...x, ...y, ...targets, center, noise].forEach(value => finite(value, 'Regression input'));
      if (noise < 0) throw new Error('Noise variance must be nonnegative.');
      const covariance = x.map((a, i) => x.map((b, j) => kernel(a, b) + (i === j ? noise : 0)));
      const key = JSON.stringify([x, noise, kernelKey]);
      if (key !== previousKey) {
        previousFactor = cholesky(covariance);
        previousKey = key;
        factorizations++;
      }
      const lower = previousFactor;
      const residual = y.map(value => value - center);
      const weights = backward(lower, forward(lower, residual));
      const cross = targets.map(target => x.map(value => kernel(value, target)));
      const solved = cross.map(column => forward(lower, column));
      const mean = cross.map(column => center + dot(column, weights));
      let clampedDiagonals = 0;
      const posteriorCovariance = targets.map((a, i) => targets.map((b, j) => {
        const value = kernel(a, b) - dot(solved[i], solved[j]);
        if (i === j && value < -1e-9) throw new Error(`Materially negative variance: ${value}`);
        if (i === j && value < 0) { clampedDiagonals++; return 0; }
        return value;
      }));
      const variance = posteriorCovariance.map((row, i) => row[i]);
      return {
        mean, covariance: posteriorCovariance, variance, cross, weights, observationMatrix: covariance,
        latentSD: variance.map(Math.sqrt), observationSD: variance.map(v => Math.sqrt(v + noise)),
        logMarginal: -0.5 * dot(residual, weights) - lower.reduce((sum, row, i) => sum + Math.log(row[i]), 0) - x.length / 2 * Math.log(2 * Math.PI),
        clampedDiagonals,
      };
    },
  };
}

export function directConditional({ rho, value, noise }) {
  [rho, value, noise].forEach(v => finite(v, 'Direct input'));
  if (Math.abs(rho) > 1 || noise < 0) throw new Error('Invalid direct covariance.');
  const mean = rho * value / (1 + noise);
  const variance = 1 - rho * rho / (1 + noise);
  return { mean, variance, observationVariance: variance + noise };
}

export function compareTarget(before, after) {
  const meanChanged = Math.abs(before.mean - after.mean) > 1e-9;
  const varianceChanged = Math.abs(before.variance - after.variance) > 1e-9;
  return meanChanged ? (varianceChanged ? 'both' : 'mean') : (varianceChanged ? 'variance' : 'neither');
}

export function measurementGains({ observations, target, candidates, kind = 'rbf', length = 1, noise = 0.25 }) {
  const posterior = createConditioner().predict({
    x: observations.map(row => row.x), y: observations.map(row => row.y),
    targets: [target, ...candidates.map(row => row.x)], noise,
    kernel: spatialKernel(kind, length), kernelKey: `${kind}:${length}`,
  });
  const currentVariance = posterior.variance[0];
  return candidates.map((candidate, index) => {
    if (!(candidate.noise > 0)) throw new Error('Candidate noise must be positive.');
    const covariance = posterior.covariance[0][index + 1];
    const ownVariance = posterior.variance[index + 1];
    const reduction = covariance ** 2 / (ownVariance + candidate.noise);
    return { ...candidate, target, covariance, ownVariance, reduction, currentVariance, remainingVariance: currentVariance - reduction };
  });
}

export function forecastKernel(family, logTheta) {
  const theta = logTheta.map(Math.exp);
  if (family === 'rbf' && theta.length === 2) {
    return (x, z) => theta[0] * Math.exp(-0.5 * ((x - z) / theta[1]) ** 2);
  }
  if (family === 'trend_periodic' && theta.length === 3) {
    return (x, z) => 1 + x * z
      + theta[0] * Math.exp(-2 * (Math.sin(Math.PI * (x - z)) / theta[1]) ** 2)
      + Math.exp(-0.5 * ((x - z) / theta[2]) ** 2);
  }
  throw new Error('Unknown frozen kernel shape.');
}

export function forecastFromPrefix(data, fitted, { family, cutoff, horizon }, conditioner = createConditioner()) {
  if (!Number.isInteger(cutoff) || cutoff < 72 || cutoff > 96 || !Number.isInteger(horizon)
    || horizon < 1 || horizon > 24 || cutoff + horizon > data.length) throw new Error('Choose 72–96 conditioning months and 1–24 future months, within the data.');
  const prefix = data.slice(0, cutoff);
  const future = data.slice(cutoff, cutoff + horizon);
  const theta = fitted[family].fitted_log_theta;
  const center = prefix.reduce((sum, row) => sum + row.co2, 0) / cutoff;
  const result = conditioner.predict({
    x: prefix.map(row => row.x), y: prefix.map(row => row.co2), targets: future.map(row => row.x),
    noise: 0.09, center, kernel: forecastKernel(family, theta), kernelKey: JSON.stringify([family, theta]),
  });
  return { ...result, rows: future, center, cutoff, horizon, family };
}

export function scoreForecast(rows, mean, observationSD) {
  const errors = rows.map((row, i) => row.co2 - mean[i]);
  const inside = errors.map((error, i) => Math.abs(error) <= normal95 * observationSD[i]);
  return {
    errors, inside,
    mae: errors.reduce((sum, error) => sum + Math.abs(error), 0) / rows.length,
    rmse: Math.sqrt(errors.reduce((sum, error) => sum + error * error, 0) / rows.length),
    covered: inside.filter(Boolean).length,
  };
}
