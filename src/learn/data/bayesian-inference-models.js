// Small analytic teaching models, not a general numerical inference library.
const logFactorials = [0];
for (let n = 1; n <= 256; n += 1) logFactorials.push(logFactorials[n - 1] + Math.log(n));

function integer(value, low, high, name) {
  if (!Number.isInteger(value) || value < low || value > high) {
    throw new RangeError(`${name} must be an integer from ${low} to ${high}.`);
  }
  return value;
}
function gammaRate(value) {
  if (!Number.isFinite(value) || value < 1e-6 || value > 1e6) {
    throw new RangeError('Gamma rate must be within the numerical teaching domain 1e-6–1e6.');
  }
  return value;
}
function probability(value) {
  if (!Number.isFinite(value) || value < 0 || value > 1) throw new RangeError('Probability must be between zero and one.');
  return value;
}
const countLog = (count, value) => count === 0 ? 0 : value === 0 ? -Infinity : count * Math.log(value);
function shapes(a, b) {
  integer(a, 1, 120, 'Alpha');
  integer(b, 1, 120, 'Beta');
}
export function logBeta(a, b) {
  shapes(a, b);
  return logFactorials[a - 1] + logFactorials[b - 1] - logFactorials[a + b - 1];
}
export function binomialMass(n, k, p) {
  integer(n, 0, 239, 'Trials');
  integer(k, 0, n, 'Successes');
  probability(p);
  return Math.exp(logFactorials[n] - logFactorials[k] - logFactorials[n - k] + countLog(k, p) + countLog(n - k, 1 - p));
}
export function betaDensity(x, a, b) {
  shapes(a, b);
  probability(x);
  return Math.exp(countLog(a - 1, x) + countLog(b - 1, 1 - x) - logBeta(a, b));
}
export function betaTail(x, a, b, upper = false) {
  shapes(a, b);
  probability(x);
  // Integer-shape identity; sum the requested tail directly, never subtract a
  // tiny survival probability from an already rounded CDF.
  const n = a + b - 1;
  let sum = 0;
  for (let k = upper ? 0 : a; k <= (upper ? a - 1 : n); k += 1) sum += binomialMass(n, k, x);
  return Math.min(1, sum);
}
export function betaQuantile(p, a, b) {
  probability(p);
  shapes(a, b);
  if (p === 0 || p === 1) return p;
  let low = 0, high = 1;
  for (let iteration = 0; iteration < 54; iteration += 1) {
    const mid = (low + high) / 2;
    const below = p <= .5 ? betaTail(mid, a, b) < p : betaTail(mid, a, b, true) > 1 - p;
    if (below) low = mid;
    else high = mid;
  }
  return (low + high) / 2;
}
export const betaPriors = Object.freeze([
  Object.freeze({ id: 'moderate', a: 2, b: 2, label: 'Beta(2, 2): moderate middle preference' }),
  Object.freeze({ id: 'uniform', a: 1, b: 1, label: 'Beta(1, 1): uniform' }),
  Object.freeze({ id: 'strong', a: 20, b: 20, label: 'Beta(20, 20): strong middle preference' }),
  Object.freeze({ id: 'low', a: 3, b: 7, label: 'Beta(3, 7): lower-rate preference' }),
]);
export function betaUpdateState(priorId = 'moderate', successes = 8, failures = 2, threshold = .7, level = .95) {
  const prior = betaPriors.find(item => item.id === priorId);
  if (!prior) throw new RangeError('Unknown prior.');
  integer(successes, 0, 80, 'Successes');
  integer(failures, 0, 20, 'Failures');
  probability(threshold);
  if (![.8, .9, .95].includes(level)) throw new RangeError('Use a displayed interval level.');
  const a = prior.a + successes, b = prior.b + failures, total = a + b;
  const tail = (1 - level) / 2;
  const low = betaQuantile(tail, a, b), high = betaQuantile(1 - tail, a, b);
  const points = Array.from({ length: 241 }, (_, index) => {
    const x = index / 240;
    return { x, prior: betaDensity(x, prior.a, prior.b), posterior: betaDensity(x, a, b) };
  });
  const shade = Array.from({ length: 121 }, (_, index) => {
    const x = low + (high - low) * index / 120;
    return { x, density: betaDensity(x, a, b) };
  });
  return {
    prior, successes, failures, a, b, level, threshold, low, high, points, shade,
    mean: a / total,
    variance: a * b / (total * total * (total + 1)),
    above: betaTail(threshold, a, b, true),
    mode: a === 1 && b === 1 ? null : a === 1 ? 0 : b === 1 ? 1 : (a - 1) / (total - 2),
    sampleRate: successes + failures === 0 ? null : successes / (successes + failures),
  };
}
export function betaBinomialMass(m, k, a, b) {
  integer(m, 0, 20, 'Batch size');
  integer(k, 0, m, 'Future successes');
  shapes(a, b);
  if (a + m > 120 || b + m > 120) throw new RangeError('Posterior plus batch exceeds the teaching bound.');
  return Math.exp(logFactorials[m] - logFactorials[k] - logFactorials[m - k] + logBeta(a + k, b + m - k) - logBeta(a, b));
}
export function batchPredictionState(a = 10, b = 4, m = 10, threshold = 9) {
  integer(m, 1, 20, 'Batch size');
  integer(threshold, 0, m, 'Count threshold');
  const meanRate = a / (a + b);
  const points = Array.from({ length: m + 1 }, (_, k) => ({
    k, integrated: betaBinomialMass(m, k, a, b), plugin: binomialMass(m, k, meanRate),
  }));
  const pluginVariance = m * meanRate * (1 - meanRate);
  return {
    a, b, m, threshold, points, meanRate, mean: m * meanRate, pluginVariance,
    variance: pluginVariance * (a + b + m) / (a + b + 1),
    correlation: 1 / (a + b + 1),
    integratedTail: points.filter(point => point.k >= threshold).reduce((sum, point) => sum + point.integrated, 0),
    pluginTail: points.filter(point => point.k >= threshold).reduce((sum, point) => sum + point.plugin, 0),
  };
}
export function gammaDensity(x, shape, rate) {
  integer(shape, 1, 80, 'Gamma shape');
  gammaRate(rate);
  if (!Number.isFinite(x) || x < 0) throw new RangeError('Rate variable must be nonnegative and finite.');
  return Math.exp(shape * Math.log(rate) - logFactorials[shape - 1] + countLog(shape - 1, x) - rate * x);
}
export function gammaTail(x, shape, rate, upper = false) {
  integer(shape, 1, 80, 'Gamma shape');
  gammaRate(rate);
  if (!Number.isFinite(x) || x < 0) throw new RangeError('Gamma coordinate must be nonnegative and finite.');
  const mean = x * rate;
  if (mean === 0) return upper ? 1 : 0;
  if (!Number.isFinite(mean)) return upper ? 0 : 1;
  if (upper || mean >= shape) {
    let sum = 0;
    for (let k = 0; k < shape; k += 1) sum += Math.exp(-mean + k * Math.log(mean) - logFactorials[k]);
    return upper ? Math.min(1, sum) : Math.max(0, 1 - sum);
  }
  let term = Math.exp(-mean + shape * Math.log(mean) - logFactorials[shape]);
  let sum = term;
  for (let k = shape + 1; k <= 1000; k += 1) {
    term *= mean / k;
    const next = sum + term;
    if (next === sum) break;
    sum = next;
  }
  return Math.min(1, sum);
}
export function gammaQuantile(p, shape, rate) {
  probability(p);
  integer(shape, 1, 80, 'Gamma shape');
  gammaRate(rate);
  if (p === 0) return 0;
  if (p === 1) return Infinity;
  let low = 0, high = Math.max(1, shape) / rate;
  const below = x => p <= .5 ? gammaTail(x, shape, rate) < p : gammaTail(x, shape, rate, true) > 1 - p;
  while (below(high)) high *= 2;
  for (let iteration = 0; iteration < 56; iteration += 1) {
    const mid = (low + high) / 2;
    if (below(mid)) low = mid;
    else high = mid;
  }
  return (low + high) / 2;
}
export function exposureUpdateState(count1 = 3, hours1 = .5, count2 = 6, hours2 = 2) {
  integer(count1, 0, 20, 'First event count');
  integer(count2, 0, 20, 'Second event count');
  for (const hours of [hours1, hours2]) if (!Number.isFinite(hours) || hours < .25 || hours > 4) throw new RangeError('Exposure must be .25–4 hours.');
  const priorShape = 2, priorRate = 1;
  const shape = priorShape + count1 + count2, rate = priorRate + hours1 + hours2;
  const max = Math.max(gammaQuantile(.995, priorShape, priorRate), gammaQuantile(.995, shape, rate));
  return {
    count1, hours1, count2, hours2, priorShape, priorRate, shape, rate, max,
    mean: shape / rate, variance: shape / (rate * rate),
    mle: (count1 + count2) / (hours1 + hours2),
    low: gammaQuantile(.025, shape, rate), high: gammaQuantile(.975, shape, rate),
    points: Array.from({ length: 201 }, (_, index) => {
      const x = max * index / 200;
      return { x, prior: gammaDensity(x, priorShape, priorRate), posterior: gammaDensity(x, shape, rate) };
    }),
  };
}
export function normalPrecisionState(priorMean = 0, priorSd = 1, observedMean = 4, n = 4, noiseSd = 2) {
  for (const value of [priorMean, observedMean]) if (!Number.isFinite(value) || Math.abs(value) > 10) throw new RangeError('Means must be finite within −10..10.');
  for (const value of [priorSd, noiseSd]) if (!Number.isFinite(value) || value < .25 || value > 4) throw new RangeError('Standard deviations must be .25–4.');
  integer(n, 1, 40, 'Measurement count');
  const priorPrecision = 1 / (priorSd * priorSd), dataPrecision = n / (noiseSd * noiseSd);
  const variance = 1 / (priorPrecision + dataPrecision);
  const mean = variance * (priorMean * priorPrecision + observedMean * dataPrecision);
  const z = 1.959963984540054;
  const intervals = [
    { name: 'Prior for the mean', mean: priorMean, sd: priorSd, kind: 'prior' },
    { name: 'Posterior for the mean', mean, sd: Math.sqrt(variance), kind: 'posterior' },
    { name: 'One future measurement', mean, sd: Math.sqrt(variance + noiseSd * noiseSd), kind: 'prediction' },
  ].map(item => ({ ...item, low: item.mean - z * item.sd, high: item.mean + z * item.sd }));
  return { priorMean, priorSd, observedMean, n, noiseSd, priorPrecision, dataPrecision, variance, mean, intervals };
}

// Precompute combinatorial counts once (1,024 sequences), not on every render.
const runCounts = Array.from({ length: 11 }, () => Array(11).fill(0));
for (let mask = 0; mask < 1024; mask += 1) {
  const sequence = Array.from({ length: 10 }, (_, index) => (mask >> index) & 1);
  const successes = sequence.reduce((sum, value) => sum + value, 0);
  const runs = 1 + sequence.slice(1).filter((value, index) => value !== sequence[index]).length;
  runCounts[successes][runs] += 1;
}
export const sequencePatterns = Object.freeze([
  Object.freeze({ id: 'clustered', label: 'Eight together, then two failures', values: Object.freeze([1, 1, 1, 1, 1, 1, 1, 1, 0, 0]) }),
  Object.freeze({ id: 'spread', label: 'Failures separated', values: Object.freeze([1, 1, 0, 1, 1, 1, 0, 1, 1, 1]) }),
]);
export function predictivePatternState(patternId = 'clustered', conditioning = 'posterior') {
  const pattern = sequencePatterns.find(item => item.id === patternId);
  if (!pattern || !['prior', 'posterior', 'same-count'].includes(conditioning)) throw new RangeError('Unknown pattern or predictive conditioning.');
  const observedRuns = 1 + pattern.values.slice(1).filter((value, index) => value !== pattern.values[index]).length;
  const a = conditioning === 'prior' ? 2 : 10, b = conditioning === 'prior' ? 2 : 4;
  const masses = Array.from({ length: 10 }, (_, index) => {
    const runs = index + 1;
    let mass = 0;
    if (conditioning === 'same-count') mass = runCounts[8][runs] / 45;
    else for (let successes = 0; successes <= 10; successes += 1) {
      mass += runCounts[successes][runs] * Math.exp(logBeta(a + successes, b + 10 - successes) - logBeta(a, b));
    }
    return { runs, mass };
  });
  return {
    pattern, conditioning, observedRuns, masses, posterior: { a: 10, b: 4 },
    lowerTail: masses.filter(item => item.runs <= observedRuns).reduce((sum, item) => sum + item.mass, 0),
  };
}
export function bayesianNumber(value, digits = 4) {
  if (value === null) return 'not unique';
  if (!Number.isFinite(value)) return String(value);
  if (value !== 0 && Math.abs(value) < 10 ** -digits) return value.toExponential(2);
  if (value < 1 && value > 1 - 10 ** -digits) return `1 − ${(1 - value).toExponential(2)}`;
  return Number(value.toFixed(digits)).toString();
}
