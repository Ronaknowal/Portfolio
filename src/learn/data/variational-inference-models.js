// Small independently checkable probability models for this lesson; no inference-library dependency.
const LOG_TWO_PI = Math.log(2 * Math.PI);
export const FINITE_JOINT = Object.freeze([0.08, 0.20, 0.12]);
function bounded(value, low, high, label) {
  if (!Number.isFinite(value) || value < low || value > high) {
    throw new RangeError(`${label} must be finite and between ${low} and ${high}.`);
  }
}
function readonly(value) {
  if (value && typeof value === 'object') {
    Object.values(value).forEach(readonly);
    Object.freeze(value);
  }
  return value;
}
export function finiteElbo(q, joint = FINITE_JOINT) {
  if (!Array.isArray(q) || !Array.isArray(joint) || q.length !== joint.length || !q.length) {
    throw new TypeError('Provide equally sized nonempty probability and joint arrays.');
  }
  q.forEach(value => bounded(value, 0, 1, 'Approximation probability'));
  joint.forEach(value => bounded(value, 0, 1e100, 'Joint weight'));
  if (Math.abs(q.reduce((a, b) => a + b, 0) - 1) > 1e-10) throw new RangeError('Approximation probabilities must sum to one.');
  const evidence = joint.reduce((a, b) => a + b, 0);
  if (!(evidence > 0) || !Number.isFinite(evidence)) throw new RangeError('The joint must have a positive finite total.');
  const posterior = joint.map(value => value / evidence);
  const rows = q.map((probability, index) => {
    const expectedLogJoint = probability === 0 ? 0 : probability * Math.log(joint[index]);
    const entropy = probability === 0 ? 0 : -probability * Math.log(probability);
    // Keep the ratio in log space: normalization can underflow even though the
    // original joint weight is positive and its log probability is finite.
    const kl = probability === 0 ? 0 : joint[index] === 0 ? Infinity : probability * (Math.log(probability) - Math.log(joint[index]) + Math.log(evidence));
    return {
      probability,
      posterior: posterior[index],
      joint: joint[index],
      expectedLogJoint,
      entropy,
      kl
    };
  });
  return readonly({
    q: [...q],
    posterior,
    rows,
    evidence,
    logEvidence: Math.log(evidence),
    expectedLogJoint: rows.reduce((sum, row) => sum + row.expectedLogJoint, 0),
    entropy: rows.reduce((sum, row) => sum + row.entropy, 0),
    elbo: rows.reduce((sum, row) => sum + row.expectedLogJoint + row.entropy, 0),
    kl: rows.reduce((sum, row) => sum + row.kl, 0)
  });
}
export function finiteApproximation(first = 0.2, share = 0.5) {
  bounded(first, 0, 1, 'A probability');
  bounded(share, 0, 1, 'B share');
  return finiteElbo([first, (1 - first) * share, (1 - first) * (1 - share)]);
}
export function restrictedFiniteOptimum() {
  const scale = 0.2 + 2 * Math.sqrt(0.5 * 0.3);
  return finiteApproximation(0.2 / scale, 0.5);
}
function correlation(value) {
  bounded(value, -0.95, 0.95, 'Correlation');
}
export function gaussianKl(mean, covariance, targetMean, targetCovariance) {
  for (const values of [mean, targetMean, covariance.flat(), targetCovariance.flat()]) {
    if (!values.every(Number.isFinite)) throw new RangeError('Gaussian inputs must be finite.');
  }
  const determinant = matrix => matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0];
  const pdet = determinant(targetCovariance),
    qdet = determinant(covariance);
  if (pdet <= 0 || qdet <= 0 || targetCovariance[0][0] <= 0 || covariance[0][0] <= 0 || covariance[0][1] !== covariance[1][0] || targetCovariance[0][1] !== targetCovariance[1][0]) {
    throw new RangeError('Covariances must be symmetric positive definite.');
  }
  const inverse = [[targetCovariance[1][1] / pdet, -targetCovariance[0][1] / pdet], [-targetCovariance[1][0] / pdet, targetCovariance[0][0] / pdet]];
  const delta = mean.map((value, index) => value - targetMean[index]);
  const trace = inverse[0][0] * covariance[0][0] + inverse[0][1] * covariance[1][0] + inverse[1][0] * covariance[0][1] + inverse[1][1] * covariance[1][1];
  const quadratic = delta[0] * (inverse[0][0] * delta[0] + inverse[0][1] * delta[1]) + delta[1] * (inverse[1][0] * delta[0] + inverse[1][1] * delta[1]);
  return 0.5 * (trace + quadratic - 2 + Math.log(pdet) - Math.log(qdet));
}
export function gaussianProjection(rho = 0.8, family = 'mean-field') {
  correlation(rho);
  if (!['mean-field', 'marginals', 'full'].includes(family)) throw new RangeError('Unknown Gaussian family.');
  const target = [[1, rho], [rho, 1]];
  const variance = family === 'mean-field' ? 1 - rho * rho : 1;
  const covariance = family === 'full' ? target.map(row => [...row]) : [[variance, 0], [0, variance]];
  return readonly({
    rho,
    family,
    target,
    covariance,
    variance,
    kl: gaussianKl([0, 0], covariance, [0, 0], target),
    targetSumVariance: 2 + 2 * rho,
    targetDifferenceVariance: 2 - 2 * rho,
    sumVariance: 2 * variance + 2 * covariance[0][1],
    differenceVariance: 2 * variance - 2 * covariance[0][1]
  });
}

// Cholesky transforms a unit circle: an equal Mahalanobis-distance contour, not marginal error bars.
export function covarianceEllipse(covariance, mean = [0, 0], radius = 1) {
  const a = Math.sqrt(covariance[0][0]);
  const b = covariance[1][0] / a;
  const c = Math.sqrt(covariance[1][1] - b * b);
  return Array.from({
    length: 121
  }, (_, index) => {
    const angle = 2 * Math.PI * index / 120;
    return [mean[0] + radius * a * Math.cos(angle), mean[1] + radius * (b * Math.cos(angle) + c * Math.sin(angle))];
  });
}
export function coordinateAscent(rho = 0.8, updates = 12) {
  correlation(rho);
  if (!Number.isInteger(updates) || updates < 0 || updates > 40) throw new RangeError('Use zero through40 updates.');
  const variance = 1 - rho * rho;
  const covariance = [[variance, 0], [0, variance]];
  const target = [[1, rho], [rho, 1]],
    targetMean = [1, -1];
  const familyGap = -0.5 * Math.log(variance);
  let mean = [-2, 2];
  const rows = [];
  for (let step = 0; step <= updates; step += 1) {
    const kl = gaussianKl(mean, covariance, targetMean, target);
    rows.push({
      step,
      updated: step === 0 ? null : (step - 1) % 2,
      mean: [...mean],
      kl,
      elbo: -kl,
      familyGap,
      optimizationGap: Math.max(0, kl - familyGap)
    });
    const index = step % 2;
    mean[index] = targetMean[index] + rho * (mean[1 - index] - targetMean[1 - index]);
  }
  return readonly({
    rows,
    variance,
    targetMean,
    target,
    familyGap
  });
}
export function normalLogDensity(value, mean = 0, sd = 1) {
  return -0.5 * LOG_TWO_PI - Math.log(sd) - 0.5 * ((value - mean) / sd) ** 2;
}
export function mixtureLogDensity(value, separation = 3) {
  const first = normalLogDensity(value, -separation),
    second = normalLogDensity(value, separation);
  const top = Math.max(first, second);
  return top + Math.log(Math.exp(first - top) + Math.exp(second - top)) - Math.log(2);
}
function simpson(functionValue, start, end, intervals) {
  const step = (end - start) / intervals;
  let total = functionValue(start) + functionValue(end);
  for (let index = 1; index < intervals; index += 1) total += (index % 2 ? 4 : 2) * functionValue(start + index * step);
  return total * step / 3;
}

// Bounded lesson calculations; Simpson integration over standard-normal noise ±9.
export function mixtureProjection(mean = 0, sd = 3, separation = 3, intervals = 1440) {
  bounded(mean, -4, 4, 'Mean');
  bounded(sd, 0.3, 3, 'Standard deviation');
  bounded(separation, 0, 4, 'Mode separation');
  if (!Number.isInteger(intervals) || intervals < 360 || intervals > 5760 || intervals % 2) throw new RangeError('Use an even integration size360–5760.');
  const kl = simpson(noise => {
    const value = mean + sd * noise;
    return Math.exp(normalLogDensity(noise)) * (normalLogDensity(value, mean, sd) - mixtureLogDensity(value, separation));
  }, -9, 9, intervals);
  const z = mean / sd;
  const rightProbability = z >= 9 ? 1 : z <= -9 ? 0 : 0.5 + Math.sign(z) * simpson(value => Math.exp(normalLogDensity(value)), 0, Math.abs(z), 720);
  const curves = Array.from({
    length: 241
  }, (_, index) => {
    const value = -8 + index / 15;
    return {
      value,
      target: Math.exp(mixtureLogDensity(value, separation)),
      approximation: Math.exp(normalLogDensity(value, mean, sd))
    };
  });
  return readonly({
    mean,
    sd,
    separation,
    kl,
    rightProbability,
    curves,
    integration: 'Simpson1440 subintervals over standard-normal noise[-9,9] by default; chart window[-8,8].'
  });
}
function normalDraws(seed, count) {
  if (!Number.isInteger(seed) || seed < 1 || seed > 4294967295) throw new RangeError('Seed must be an integer1–4294967295.');
  let state = seed >>> 0;
  const uniform = () => {
    state ^= state << 13;
    state ^= state >>> 17;
    state ^= state << 5;
    return ((state >>> 0) + 0.5) / 4294967296;
  };
  const values = [];
  while (values.length < count) {
    const radius = Math.sqrt(-2 * Math.log(uniform())),
      angle = 2 * Math.PI * uniform();
    values.push(radius * Math.cos(angle));
    if (values.length < count) values.push(radius * Math.sin(angle));
  }
  return values;
}
export function variationalGradients({
  mean = 0,
  sd = 1.2,
  seed = 7,
  count = 100,
  targetMean = 1.5,
  targetSd = 0.7
} = {}) {
  bounded(mean, -2, 3, 'Mean');
  bounded(sd, 0.2, 2, 'Standard deviation');
  bounded(targetMean, -2, 3, 'Target mean');
  bounded(targetSd, 0.2, 2, 'Target standard deviation');
  if (!Number.isInteger(count) || count < 2 || count > 1000) throw new RangeError('Use2–1000 draws.');
  const rows = normalDraws(seed, count).map((noise, index) => {
    const value = mean + sd * noise;
    const targetSlope = (targetMean - value) / (targetSd * targetSd);
    const logRatio = normalLogDensity(value, targetMean, targetSd) - normalLogDensity(value, mean, sd);
    return {
      index: index + 1,
      noise,
      value,
      targetSlope,
      logRatio,
      pathMean: targetSlope,
      pathLogSd: 1 + sd * noise * targetSlope,
      scoreMean: logRatio * noise / sd,
      scoreLogSd: logRatio * (noise * noise - 1)
    };
  });
  const estimates = {};
  for (const name of ['pathMean', 'pathLogSd', 'scoreMean', 'scoreLogSd']) {
    const average = rows.reduce((total, row) => total + row[name], 0) / count;
    const variance = rows.reduce((total, row) => total + (row[name] - average) ** 2, 0) / (count - 1);
    estimates[name] = {
      average,
      mcse: Math.sqrt(variance / count)
    };
  }
  const exactMean = (targetMean - mean) / (targetSd * targetSd),
    exactLogSd = 1 - sd * sd / (targetSd * targetSd);
  return readonly({
    mean,
    sd,
    seed,
    count,
    targetMean,
    targetSd,
    rows,
    estimates,
    exactMean,
    exactLogSd,
    elbo: Math.log(sd / targetSd) + 0.5 - (sd * sd + (mean - targetMean) ** 2) / (2 * targetSd * targetSd)
  });
}
