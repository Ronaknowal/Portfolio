// Bounded educational models. Exact laws are separated from finite seeded samples.
// Rows index current states; columns index next states. Brownian normals use
// standard deviation sqrt(elapsed time), never elapsed time itself.
function freeze(value) {
  if (value && typeof value === 'object') {
    Object.values(value).forEach(freeze);
    Object.freeze(value);
  }
  return value;
}

function bounded(value, minimum, maximum, name) {
  if (!Number.isFinite(value) || value < minimum || value > maximum) {
    throw new RangeError(name + ' must be between ' + minimum + ' and ' + maximum + '.');
  }
}

function integer(value, minimum, maximum, name) {
  bounded(value, minimum, maximum, name);
  if (!Number.isInteger(value)) throw new RangeError(name + ' must be a whole number.');
}

function probability(value, name) {
  bounded(value, 0, 1, name);
  if ((value > 0 && value < 1e-6) || (value < 1 && 1 - value < 1e-6)) {
    throw new RangeError(name + ': use exact 0/1 or stay at least 0.000001 from each endpoint.');
  }
}

export function parseProcessNumber(text, minimum, maximum, name) {
  const value = Number(text);
  if (String(text).trim() === '') throw new RangeError(name + ' needs a number.');
  bounded(value, minimum, maximum, name);
  return value;
}

export const MARKOV_PRESETS = freeze({
  weather: { name: 'Weather', a: 0.2, b: 0.3 },
  sticky: { name: 'Same equilibrium, slower', a: 0.02, b: 0.03 },
  independent: { name: 'Fresh draw each step', a: 0.4, b: 0.6 },
  alternating: { name: 'Always alternate', a: 1, b: 1 },
  identity: { name: 'Never change state', a: 0, b: 0 },
  absorbing: { name: 'Sunny is absorbing', a: 0, b: 0.3 },
});

export function finiteProcessLaw(kind, length = 4) {
  integer(length, 2, 8, 'Length');
  if (!['fresh', 'frozen', 'alternating'].includes(kind)) throw new RangeError('Unknown process.');
  const paths = [];
  for (let mask = 0; mask < 2 ** length; mask += 1) {
    const values = Array.from({ length }, (_, index) => (mask >> (length - index - 1)) & 1);
    const allowed = kind === 'fresh' || values.every((value, index) =>
      kind === 'frozen' ? value === values[0] : value === (values[0] + index) % 2);
    if (allowed) paths.push({ values, probability: kind === 'fresh' ? 2 ** -length : 0.5 });
  }
  const marginalOne = Array.from({ length }, (_, time) =>
    paths.reduce((sum, path) => sum + path.probability * path.values[time], 0));
  const adjacentEqual = paths.reduce((sum, path) =>
    sum + path.probability * Number(path.values[0] === path.values[1]), 0);
  return freeze({ kind, length, paths, marginalOne, adjacentEqual });
}

export function markovState({ a = 0.2, b = 0.3, initialSunny = 1, steps = 12 } = {}) {
  probability(a, 'Sunny to Rainy');
  probability(b, 'Rainy to Sunny');
  probability(initialSunny, 'Initial sunny probability');
  integer(steps, 0, 60, 'Steps');
  const matrix = [[1 - a, a], [b, 1 - b]];
  const history = [{ step: 0, distribution: [initialSunny, 1 - initialSunny], flows: null }];
  for (let step = 1; step <= steps; step += 1) {
    const previous = history.at(-1).distribution;
    const flows = matrix.map((row, source) => row.map(value => previous[source] * value));
    const distribution = [flows[0][0] + flows[1][0], flows[0][1] + flows[1][1]];
    history.push({ step, distribution, flows });
  }
  const stationary = a + b === 0 ? null : [b / (a + b), a / (a + b)];
  return freeze({
    a, b, initialSunny, steps, matrix, history, stationary,
    multiplier: 1 - a - b,
    irreducible: a > 0 && b > 0,
    periodTwo: a === 1 && b === 1,
    convergesFromEveryStart: !(a === 1 && b === 1),
    uniqueStationary: stationary !== null,
    limitingDistribution: a + b === 0 ? history[0].distribution :
      a === 1 && b === 1 ? initialSunny === 0.5 ? [0.5, 0.5] : null : stationary,
  });
}

export function transitionEstimate(trajectories, stateCount = 3) {
  integer(stateCount, 2, 6, 'State count');
  if (!Array.isArray(trajectories) || trajectories.length < 1 || trajectories.length > 30) {
    throw new RangeError('Provide 1–30 separate trajectories.');
  }
  const counts = Array.from({ length: stateCount }, () => Array(stateCount).fill(0));
  for (const path of trajectories) {
    if (!Array.isArray(path) || path.length < 1 || path.length > 200) {
      throw new RangeError('Each trajectory needs 1–200 states.');
    }
    path.forEach(state => integer(state, 0, stateCount - 1, 'State'));
    for (let time = 1; time < path.length; time += 1) counts[path[time - 1]][path[time]] += 1;
  }
  const departures = counts.map(row => row.reduce((sum, value) => sum + value, 0));
  const estimate = counts.map((row, index) =>
    departures[index] === 0 ? null : row.map(value => value / departures[index]));
  return freeze({ counts, departures, estimate });
}

function solveLinear(matrix, rhs) {
  const augmented = matrix.map((row, index) => [...row, rhs[index]]);
  const size = rhs.length;
  for (let column = 0; column < size; column += 1) {
    let pivot = column;
    for (let row = column + 1; row < size; row += 1) {
      if (Math.abs(augmented[row][column]) > Math.abs(augmented[pivot][column])) pivot = row;
    }
    if (Math.abs(augmented[pivot][column]) < 1e-12) {
      throw new RangeError('This transient system is singular in the finite calculator.');
    }
    [augmented[column], augmented[pivot]] = [augmented[pivot], augmented[column]];
    const divisor = augmented[column][column];
    for (let j = column; j <= size; j += 1) augmented[column][j] /= divisor;
    for (let row = 0; row < size; row += 1) {
      if (row === column) continue;
      const coefficient = augmented[row][column];
      for (let j = column; j <= size; j += 1) augmented[row][j] -= coefficient * augmented[column][j];
    }
  }
  return augmented.map(row => row[size]);
}

export function absorptionState({ boundary = 4, start = 2, upward = 0.5, steps = 16 } = {}) {
  integer(boundary, 3, 8, 'Upper boundary');
  integer(start, 0, boundary, 'Starting reserve');
  bounded(upward, 0.1, 0.9, 'Upward probability');
  integer(steps, 0, 60, 'Steps');
  const matrix = Array.from({ length: boundary + 1 }, () => Array(boundary + 1).fill(0));
  matrix[0][0] = 1;
  matrix[boundary][boundary] = 1;
  for (let state = 1; state < boundary; state += 1) {
    matrix[state][state - 1] = 1 - upward;
    matrix[state][state + 1] = upward;
  }
  const q = matrix.slice(1, boundary).map(row => row.slice(1, boundary));
  const system = q.map((row, i) => row.map((value, j) => Number(i === j) - value));
  const upperRhs = matrix.slice(1, boundary).map(row => row[boundary]);
  const success = [0, ...solveLinear(system, upperRhs), 1];
  const meanSteps = [0, ...solveLinear(system, Array(boundary - 1).fill(1)), 0];
  const columns = system.map((_, index) =>
    solveLinear(system, Array.from({ length: boundary - 1 }, (__, row) => Number(row === index))));
  const visits = system.map((_, row) => columns.map(column => column[row]));
  const initial = Array.from({ length: boundary + 1 }, (_, state) => Number(state === start));
  const history = [{ step: 0, distribution: initial, firstLower: initial[0],
    firstUpper: initial[boundary], surviving: 1 - initial[0] - initial[boundary] }];
  for (let step = 1; step <= steps; step += 1) {
    const previous = history.at(-1).distribution;
    const distribution = matrix.map((_, destination) =>
      previous.reduce((sum, mass, source) => sum + mass * matrix[source][destination], 0));
    // First hits arise only from a transient neighbour, not old absorbed mass.
    const firstLower = previous[1] * (1 - upward);
    const firstUpper = previous[boundary - 1] * upward;
    const surviving = distribution.slice(1, boundary).reduce((sum, mass) => sum + mass, 0);
    history.push({ step, distribution, firstLower, firstUpper, surviving });
  }
  return freeze({ boundary, start, upward, steps, matrix, q, success, meanSteps, visits, history });
}

// Repeatable illustrations, not a security RNG or evidence for stochastic laws.
export function processUniforms(seed) {
  integer(seed, 0, 4294967295, 'Seed');
  let state = seed >>> 0;
  return () => {
    state = (state + 0x6D2B79F5) >>> 0;
    let mixed = Math.imul(state ^ (state >>> 15), 1 | state);
    mixed ^= mixed + Math.imul(mixed ^ (mixed >>> 7), 61 | mixed);
    return (((mixed ^ (mixed >>> 14)) >>> 0) + 0.5) / 4294967296;
  };
}

export function poissonMass(mean, count) {
  bounded(mean, 0, 72, 'Poisson mean');
  integer(count, 0, 160, 'Count');
  let mass = Math.exp(-mean);
  for (let k = 1; k <= count; k += 1) mass *= mean / k;
  return mass;
}

function validateRates(rates, switchTime, horizon) {
  if (!Array.isArray(rates) || rates.length !== 2) throw new RangeError('Use two rates.');
  rates.forEach(rate => bounded(rate, 0, 6, 'Rate'));
  rates.forEach(rate => {
    if (rate > 0 && rate < 0.01) throw new RangeError('Use rate zero or at least 0.01 per minute.');
  });
  bounded(horizon, 0.1, 6, 'Horizon');
  bounded(switchTime, 0, horizon, 'Rate-change time');
}

export function integratedRate(time, rates, switchTime) {
  bounded(time, 0, 6, 'Time');
  bounded(switchTime, 0, 6, 'Rate-change time');
  if (!Array.isArray(rates) || rates.length !== 2) throw new RangeError('Use two rates.');
  rates.forEach(rate => bounded(rate, 0, 6, 'Rate'));
  return rates[0] * Math.min(time, switchTime) + rates[1] * Math.max(0, time - switchTime);
}

function realTimeFromIntensity(unitTime, rates, switchTime) {
  const firstArea = rates[0] * switchTime;
  if (rates[0] > 0 && unitTime <= firstArea) return unitTime / rates[0];
  if (rates[1] === 0) return Infinity;
  return switchTime + (unitTime - firstArea) / rates[1];
}

export function arrivalState({
  rates = [2.5, 2.5], switchTime = 2, horizon = 3, interval = [0, 1],
  seed = 11, splitProbability = 0.5, routing = 'independent', maxEvents = 120,
} = {}) {
  validateRates(rates, switchTime, horizon);
  probability(splitProbability, 'Routing probability');
  integer(maxEvents, 1, 300, 'Event cap');
  if (!['independent', 'alternating'].includes(routing)) throw new RangeError('Unknown routing.');
  if (!Array.isArray(interval) || interval.length !== 2) throw new RangeError('Use a time interval.');
  interval.forEach(time => bounded(time, 0, horizon, 'Interval endpoint'));
  if (interval[0] > interval[1]) throw new RangeError('Interval start must not exceed its end.');
  const clockUniform = processUniforms(seed);
  const markUniform = processUniforms((seed ^ 0x9E3779B9) >>> 0);
  const totalIntensity = integratedRate(horizon, rates, switchTime);
  const events = [];
  let unitTime = 0;
  let complete = totalIntensity === 0;
  for (let index = 0; index < maxEvents && !complete; index += 1) {
    unitTime += -Math.log(clockUniform());
    if (unitTime > totalIntensity) {
      complete = true;
      break;
    }
    const time = realTimeFromIntensity(unitTime, rates, switchTime);
    const mark = routing === 'independent' ? Number(markUniform() >= splitProbability) : index % 2;
    events.push({ index: index + 1, time, unitTime, mark,
      gap: time - (events.at(-1)?.time ?? 0) });
  }
  const observedUntil = complete ? horizon : events.at(-1)?.time ?? 0;
  const intervalMean = integratedRate(interval[1], rates, switchTime) -
    integratedRate(interval[0], rates, switchTime);
  const selected = events.filter(event => event.time > interval[0] && event.time <= interval[1]);
  const pmf = Array.from({ length: 41 }, (_, count) => poissonMass(intervalMean, count));
  // Sum the small tail directly: 1 - CDF can erase a representable rare event.
  const pmfTail = Array.from({ length: 120 }, (_, index) =>
    poissonMass(intervalMean, 41 + index)).reduce((sum, value) => sum + value, 0);
  return freeze({ rates: [...rates], switchTime, horizon, interval: [...interval], seed,
    splitProbability, routing, totalIntensity, events, complete, observedUntil, intervalMean,
    intervalCount: interval[1] <= observedUntil ? selected.length : null,
    markedCounts: interval[1] <= observedUntil ?
      [0, 1].map(mark => selected.filter(event => event.mark === mark).length) : null,
    zeroProbability: Math.exp(-intervalMean), pmf, pmfTail,
    tailBelowFloatingPointRange: intervalMean > 0 && pmfTail === 0 });
}

export function splitCountLaw(mean, first, second, probabilityFirst) {
  bounded(mean, 0, 36, 'Mean');
  integer(first, 0, 30, 'First count');
  integer(second, 0, 30, 'Second count');
  probability(probabilityFirst, 'Mark probability');
  let combinations = 1;
  for (let index = 1; index <= first; index += 1) {
    combinations *= (first + second - index + 1) / index;
  }
  const conditional = combinations * probabilityFirst ** first *
    (1 - probabilityFirst) ** second;
  return freeze({ conditional, joint: poissonMass(mean, first + second) * conditional,
    product: poissonMass(mean * probabilityFirst, first) *
      poissonMass(mean * (1 - probabilityFirst), second) });
}

export function jumpClockState({
  alpha = 0.5, beta = 2, horizon = 4, initial = 0, seed = 11, maxJumps = 120,
} = {}) {
  bounded(alpha, 0.05, 6, 'On to Off rate');
  bounded(beta, 0.05, 6, 'Off to On rate');
  bounded(horizon, 0.1, 12, 'Horizon');
  integer(initial, 0, 1, 'Initial state');
  integer(maxJumps, 1, 300, 'Jump cap');
  const uniform = processUniforms(seed);
  const stationaryOn = beta / (alpha + beta);
  const probabilityOn = time =>
    stationaryOn + (Number(initial === 0) - stationaryOn) * Math.exp(-(alpha + beta) * time);
  const history = Array.from({ length: 41 }, (_, index) => {
    const time = horizon * index / 40;
    return { time, probabilityOn: probabilityOn(time) };
  });
  const expectedOnExposure = stationaryOn * horizon +
    (Number(initial === 0) - stationaryOn) * -Math.expm1(-(alpha + beta) * horizon) / (alpha + beta);
  const segments = [];
  let time = 0;
  let state = initial;
  for (let index = 0; index < maxJumps && time < horizon; index += 1) {
    const duration = -Math.log(uniform()) / (state === 0 ? alpha : beta);
    const end = Math.min(horizon, time + duration);
    segments.push({ state, start: time, end, holdingTime: duration, jumped: time + duration <= horizon });
    time = end;
    state = 1 - state;
  }
  const exposure = [0, 1].map(value => segments.filter(segment => segment.state === value)
    .reduce((sum, segment) => sum + segment.end - segment.start, 0));
  const departures = [0, 1].map(value =>
    segments.filter(segment => segment.state === value && segment.jumped).length);
  return freeze({ alpha, beta, horizon, initial, seed, generator: [[-alpha, alpha], [beta, -beta]],
    stationaryOn, history, expectedOnExposure, segments, exposure, departures,
    complete: time === horizon, observedUntil: time });
}

export function brownianFromNormals(normals, { horizon = 1, drift = 0, scale = 1 } = {}) {
  bounded(horizon, 0.1, 4, 'Horizon');
  bounded(drift, -2, 2, 'Drift');
  bounded(scale, 0.05, 4, 'Diffusion scale');
  if (!Array.isArray(normals) || normals.length < 1 || normals.length > 256) {
    throw new RangeError('Use 1–256 normal increments.');
  }
  normals.forEach(normal => bounded(normal, -40, 40, 'Normal draw'));
  const dt = horizon / normals.length;
  const values = [0];
  normals.forEach(normal => values.push(values.at(-1) + drift * dt + scale * Math.sqrt(dt) * normal));
  return freeze(values);
}

export function brownianState({
  seed = 11, horizon = 1, drift = 0, scale = 1, level = 4, pathIndex = 0,
} = {}) {
  bounded(horizon, 0.1, 4, 'Horizon');
  bounded(drift, -2, 2, 'Drift');
  bounded(scale, 0.05, 4, 'Diffusion scale');
  integer(level, 1, 8, 'Resolution level');
  integer(pathIndex, 0, 7, 'Path');
  const uniform = processUniforms(seed);
  const fine = Array.from({ length: 8 }, () => {
    const normals = Array.from({ length: 256 }, () =>
      Math.sqrt(-2 * Math.log(uniform())) * Math.cos(2 * Math.PI * uniform()));
    return brownianFromNormals(normals, { horizon, drift, scale });
  });
  const count = 2 ** level;
  const stride = 256 / count;
  const paths = fine.map(values => Array.from({ length: count + 1 }, (_, index) => values[index * stride]));
  const selected = paths[pathIndex];
  const dt = horizon / count;
  const increments = selected.slice(1).map((value, index) => value - selected[index]);
  const rawVariation = increments.reduce((sum, value) => sum + value ** 2, 0);
  const centeredVariation = increments.reduce((sum, value) => sum + (value - drift * dt) ** 2, 0);
  const times = Array.from({ length: count + 1 }, (_, index) => index * dt);
  const covarianceTimes = [0, horizon / 4, horizon / 2, horizon];
  const covariance = covarianceTimes.map(s => covarianceTimes.map(t => scale ** 2 * Math.min(s, t)));
  return freeze({ seed, horizon, drift, scale, level, pathIndex, count, dt, paths, times, selected,
    increments, rawVariation, centeredVariation, covarianceTimes, covariance,
    rawVariationExpectation: scale ** 2 * horizon + drift ** 2 * horizon ** 2 / count,
    rawVariationVariance: 2 * scale ** 4 * horizon ** 2 / count +
      4 * drift ** 2 * scale ** 2 * horizon ** 3 / count ** 2,
    centeredVariationExpectation: scale ** 2 * horizon,
    centeredVariationVariance: 2 * scale ** 4 * horizon ** 2 / count,
    terminalMean: drift * horizon, terminalVariance: scale ** 2 * horizon,
    sampledMaximum: Math.max(...selected), sampledMinimum: Math.min(...selected) });
}

export function bridgeState({ left = 0, right = 0, duration = 1, scale = 1,
  fraction = 0.5, barrier = 1 } = {}) {
  bounded(left, -10, 10, 'Left endpoint');
  bounded(right, -10, 10, 'Right endpoint');
  bounded(duration, 0.01, 12, 'Duration');
  bounded(scale, 0.05, 4, 'Scale');
  bounded(fraction, 0, 1, 'Time fraction');
  bounded(barrier, -10, 10, 'Barrier');
  const mean = left + (right - left) * fraction;
  const variance = scale ** 2 * duration * fraction * (1 - fraction);
  const logCrossing = barrier <= Math.max(left, right) ? 0 :
    -2 * (barrier - left) * (barrier - right) / (scale ** 2 * duration);
  const crossingProbability = Math.exp(logCrossing);
  return freeze({ left, right, duration, scale, fraction, barrier, mean, variance,
    logCrossing, crossingProbability,
    probabilityBelowFloatingPointRange: crossingProbability === 0 });
}
