// Bounded teaching models. Curves evaluate formulas; reported optima are analytic.
function integer(value, low, high, name) {
  if (!Number.isInteger(value) || value < low || value > high) throw new RangeError(`${name} must be an integer from ${low} to ${high}.`);
}
export function probability(value) {
  if (!Number.isFinite(value) || value < 0 || value > 1) throw new RangeError('Probability must be between 0 and 1.');
  return value;
}
export function parseBinary(text) {
  const tokens = text.trim() ? text.trim().split(/[\s,]+/) : [];
  if (tokens.length > 40 || tokens.some(token => !/^[01]$/.test(token))) throw new Error('Use at most 40 zeros and ones, separated by spaces or commas. Blank means no observations.');
  return Object.freeze(tokens.map(Number));
}
export function parseMeasurements(text) {
  const tokens = text.trim() ? text.trim().split(/[\s,]+/) : [];
  const values = tokens.map(Number);
  if (!tokens.length || tokens.length > 12 || values.some(value => !Number.isFinite(value) || Math.abs(value) > 50)) throw new Error('Use 1–12 finite measurements between −50 and 50, separated by spaces or commas.');
  return Object.freeze(values);
}
export function xlog(count, value) {
  return count === 0 ? 0 : value === 0 ? -Infinity : count * Math.log(value);
}
export function logChoose(n, k) {
  let result = 0;
  for (let i = 1; i <= k; i += 1) result += Math.log(n - k + i) - Math.log(i);
  return result;
}
export function bernoulli(s, f, p, countEvent = false) {
  integer(s, 0, 40, 'Successes');
  integer(f, 0, 40 - s, 'Failures');
  probability(p);
  const log = xlog(s, p) + xlog(f, 1 - p) + (countEvent ? logChoose(s + f, s) : 0);
  return Object.freeze({
    log,
    value: Math.exp(log),
    mle: s + f === 0 ? null : s / (s + f)
  });
}
export function locationFit(values, center) {
  if (!values.length || values.some(value => !Number.isFinite(value)) || !Number.isFinite(center)) throw new Error('Finite nonempty measurements and center required.');
  const sorted = [...values].sort((a, b) => a - b);
  const n = values.length;
  const mean = values.reduce((sum, value) => sum + value, 0) / n;
  const medianLow = sorted[Math.floor((n - 1) / 2)];
  const medianHigh = sorted[Math.floor(n / 2)];
  const residuals = Object.freeze(values.map(value => value - center));
  return Object.freeze({
    mean,
    medianLow,
    medianHigh,
    residuals,
    squared: residuals.reduce((sum, value) => sum + value * value, 0),
    absolute: residuals.reduce((sum, value) => sum + Math.abs(value), 0),
    varianceMLE: values.reduce((sum, value) => sum + (value - mean) ** 2, 0) / n
  });
}
export function betaDensity(p, a, b) {
  probability(p);
  integer(a, 1, 52, 'Alpha');
  integer(b, 1, 52, 'Beta');
  // 1/B(a,b) = (a+b−1) choose(a+b−2,a−1) for integer shapes.
  return Math.exp(Math.log(a + b - 1) + logChoose(a + b - 2, a - 1) + xlog(a - 1, p) + xlog(b - 1, 1 - p));
}
export function betaPosterior(s, f, a, b) {
  bernoulli(s, f, 0.5);
  integer(a, 1, 12, 'Prior alpha');
  integer(b, 1, 12, 'Prior beta');
  const A = a + s;
  const B = b + f;
  const mode = A === 1 && B === 1 ? null : A === 1 ? 0 : B === 1 ? 1 : (A - 1) / (A + B - 2);
  return Object.freeze({
    a: A,
    b: B,
    mode,
    mean: A / (A + B),
    variance: A * B / ((A + B) ** 2 * (A + B + 1))
  });
}
export function samplingMass(n, p) {
  integer(n, 1, 30, 'Sample size');
  probability(p);
  return Object.freeze(Array.from({
    length: n + 1
  }, (_, k) => Object.freeze({
    k,
    estimate: k / n,
    mass: Math.exp(logChoose(n, k) + xlog(k, p) + xlog(n - k, 1 - p))
  })));
}
export function normalPosterior(values, variance, priorMean, priorVariance) {
  if (![variance, priorVariance].every(value => Number.isFinite(value) && value > 0) || !Number.isFinite(priorMean)) throw new Error('Positive finite variances and finite prior mean required.');
  const fit = locationFit(values, 0);
  const precision = values.length / variance + 1 / priorVariance;
  return Object.freeze({
    mean: (values.length * fit.mean / variance + priorMean / priorVariance) / precision,
    variance: 1 / precision
  });
}
export function logit(p) {
  if (!(p > 0 && p < 1)) throw new RangeError('Log-odds requires 0 < p < 1.');
  return Math.log(p / (1 - p));
}
export function logistic(eta) {
  return eta >= 0 ? 1 / (1 + Math.exp(-eta)) : Math.exp(eta) / (1 + Math.exp(eta));
}
export function transformedBetaDensity(eta, a, b) {
  const p = logistic(eta);
  return betaDensity(p, a, b) * p * (1 - p);
}
