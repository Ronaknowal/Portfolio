// Finite teaching models, with explicit arithmetic and simulation contracts.
// Analytic laws describe the stated SDEs; seeded paths are synthetic examples.
function finite(value, low, high, name) {
  if (typeof value !== 'number' || !Number.isFinite(value) || value < low || value > high) {
    throw new RangeError(`${name} must be finite and between ${low} and ${high}.`);
  }
  return value;
}
function integer(value, low, high, name) {
  finite(value, low, high, name);
  if (!Number.isInteger(value)) throw new RangeError(`${name} must be an integer.`);
  return value;
}
function nonzeroRange(value, low, high, name) {
  if (value === 0) return value;
  return finite(value, low, high, name);
}
function freeze(value) {
  if (value && typeof value === 'object' && !Object.isFrozen(value)) {
    for (const child of Object.values(value)) freeze(child);
    Object.freeze(value);
  }
  return value;
}
export function normalSource(seed = 5) {
  integer(seed, 1, 2147483647, 'Seed');
  let state = seed >>> 0;
  const uniform = () => {
    state = Math.imul(1664525, state) + 1013904223 >>> 0;
    return (state + 0.5) / 4294967296;
  };
  return () => Math.sqrt(-2 * Math.log(uniform())) * Math.cos(2 * Math.PI * uniform());
}
export function brownianIncrements({
  seed = 5,
  steps = 256,
  horizon = 1
} = {}) {
  integer(steps, 1, 512, 'Fine step count');
  finite(horizon, 0.0625, 4, 'Time horizon');
  const normal = normalSource(seed);
  return freeze(Array.from({
    length: steps
  }, () => Math.sqrt(horizon / steps) * normal()));
}
export function groupIncrements(increments, group) {
  if (!Array.isArray(increments) || !increments.length || increments.length > 512) {
    throw new RangeError('Use one to 512 finite increments.');
  }
  increments.forEach(value => finite(value, -20, 20, 'Increment'));
  integer(group, 1, increments.length, 'Group size');
  if (increments.length % group) throw new RangeError('The group must divide the fine grid.');
  const result = [];
  for (let start = 0; start < increments.length; start += group) {
    let sum = 0;
    for (let index = start; index < start + group; index += 1) sum += increments[index];
    result.push(sum);
  }
  return freeze(result);
}
export function integralTrace(increments = [0.5, -0.25, 0.75, -0.5], horizon = 1) {
  const active = groupIncrements(increments, 1);
  finite(horizon, 0.0625, 4, 'Time horizon');
  let value = 0;
  let left = 0;
  let right = 0;
  let symmetric = 0;
  let quadratic = 0;
  const rows = active.map((increment, index) => {
    const before = value;
    value += increment;
    const leftContribution = before * increment;
    const rightContribution = value * increment;
    left += leftContribution;
    right += rightContribution;
    symmetric += (before + value) * increment / 2;
    quadratic += increment * increment;
    return {
      index,
      time: (index + 1) * horizon / active.length,
      before,
      increment,
      after: value,
      leftContribution,
      rightContribution,
      left,
      right,
      symmetric,
      quadratic
    };
  });
  return freeze({
    rows,
    horizon,
    terminal: value,
    left,
    right,
    symmetric,
    quadratic,
    itoLimitAtTerminal: (value * value - horizon) / 2,
    quadraticMean: horizon,
    quadraticVariance: 2 * horizon * horizon / active.length
  });
}
function growthContract(mu, sigma, initial, horizon) {
  finite(mu, -0.5, 1, 'Growth drift');
  nonzeroRange(sigma, 0.05, 1.2, 'Noise scale');
  finite(initial, 0.1, 3, 'Positive initial state');
  nonzeroRange(horizon, 0.0625, 4, 'Elapsed time');
}
export function growthLaw({
  mu = 0.4,
  sigma = 0.3,
  initial = 1,
  horizon = 1
} = {}) {
  growthContract(mu, sigma, initial, horizon);
  const logMean = Math.log(initial) + (mu - sigma * sigma / 2) * horizon;
  const logVariance = sigma * sigma * horizon;
  const mean = initial * Math.exp(mu * horizon);
  const variance = mean * mean * Math.expm1(logVariance);
  const quantile = z => Math.exp(logMean + Math.sqrt(logVariance) * z);
  return freeze({
    mu,
    sigma,
    initial,
    horizon,
    logMean,
    logVariance,
    mean,
    variance,
    median: Math.exp(logMean),
    lower: quantile(-1.6448536269514722),
    upper: quantile(1.6448536269514722),
    atom: logVariance === 0,
    almostSureLogRate: mu - sigma * sigma / 2,
    secondMomentRate: 2 * mu + sigma * sigma
  });
}
export function growthPath({
  mu = 0.4,
  sigma = 0.3,
  initial = 1,
  horizon = 1,
  increments = brownianIncrements(),
  method = 'all'
} = {}) {
  growthContract(mu, sigma, initial, horizon);
  if (horizon === 0) throw new RangeError('A path needs a positive horizon.');
  if (!['all', 'euler', 'milstein'].includes(method)) throw new RangeError('Unknown method.');
  const active = groupIncrements(increments, 1);
  const step = horizon / active.length;
  let brownian = 0;
  let euler = initial;
  let milstein = initial;
  const rows = [{
    index: 0,
    time: 0,
    brownian: 0,
    exact: initial,
    logExact: Math.log(initial),
    euler,
    milstein,
    increment: 0,
    drift: 0,
    noise: 0,
    correction: 0
  }];
  for (const [index, increment] of active.entries()) {
    const drift = mu * milstein * step;
    const noise = sigma * milstein * increment;
    const correction = sigma * sigma * milstein * (increment * increment - step) / 2;
    euler += mu * euler * step + sigma * euler * increment;
    milstein += drift + noise + correction;
    brownian += increment;
    const time = (index + 1) * step;
    const logExact = Math.log(initial) + (mu - sigma * sigma / 2) * time + sigma * brownian;
    const exact = Math.exp(logExact);
    if (![euler, milstein, exact].every(Number.isFinite) || exact === 0) {
      throw new RangeError('This path exceeded the finite arithmetic range.');
    }
    rows.push({
      index: index + 1,
      time,
      brownian,
      exact,
      logExact,
      euler,
      milstein,
      increment,
      drift,
      noise,
      correction
    });
  }
  return freeze({
    rows,
    mu,
    sigma,
    initial,
    horizon,
    step,
    steps: active.length,
    eulerNegative: rows.some(row => row.euler <= 0),
    milsteinNegative: rows.some(row => row.milstein <= 0)
  });
}
function exponentialRelative(value) {
  if (Math.abs(value) < 1e-7) return 1 + value / 2 + value * value / 6;
  return Math.expm1(value) / value;
}
export function ouNoiseMoments(theta, step) {
  nonzeroRange(theta, 0.05, 3, 'Reversion rate');
  finite(step, 1 / 8192, 4, 'OU time step');
  const z = theta * step;
  const coefficient = exponentialRelative(-z);
  const variance = step * exponentialRelative(-2 * z);
  // Var(J | deltaW) / h = exp(-z) sum_{k>=1} 2k z^(2k)/(2k+2)!.
  // A positive series avoids subtracting almost equal covariances at small z.
  let term = z * z / 12;
  let series = term;
  for (let k = 1; k < 80; k += 1) {
    term *= (k + 1) / k * z * z / ((2 * k + 3) * (2 * k + 4));
    series += term;
    if (term <= Number.EPSILON * Math.max(series, Number.MIN_VALUE)) break;
  }
  const residualVariance = step * Math.exp(-z) * series;
  return freeze({
    coefficient,
    variance,
    covariance: step * coefficient,
    residualVariance,
    attenuation: Math.exp(-z)
  });
}
export function ouLaw({
  theta = 1,
  target = 0,
  eta = 0.8,
  initial = 1.5,
  initialVariance = 0,
  time = 1
} = {}) {
  nonzeroRange(theta, 0.05, 3, 'Reversion rate');
  nonzeroRange(eta, 0.05, 1.2, 'OU diffusion scale');
  finite(target, -2, 2, 'Restoring target');
  finite(initial, -2, 2, 'Initial mean');
  finite(initialVariance, 0, 4, 'Initial variance');
  nonzeroRange(time, 1 / 8192, 4, 'Time');
  const attenuation = Math.exp(-theta * time);
  const variance = initialVariance * attenuation * attenuation + eta * eta * time * exponentialRelative(-2 * theta * time);
  return freeze({
    theta,
    target,
    eta,
    initial,
    initialVariance,
    time,
    mean: target + (initial - target) * attenuation,
    variance,
    standardDeviation: Math.sqrt(variance),
    stationaryVariance: theta > 0 ? eta * eta / (2 * theta) : null,
    varianceInjection: eta * eta,
    varianceRemoval: 2 * theta * variance,
    varianceRate: eta * eta - 2 * theta * variance,
    atom: variance === 0
  });
}
export function ouPath({
  theta = 1,
  target = 0,
  eta = 0.8,
  initial = 1.5,
  horizon = 2,
  steps = 128,
  seed = 11
} = {}) {
  integer(steps, 1, 512, 'OU step count');
  ouLaw({
    theta,
    target,
    eta,
    initial,
    time: horizon
  });
  if (horizon <= 0) throw new RangeError('A path needs positive time.');
  const step = horizon / steps;
  const moments = ouNoiseMoments(theta, step);
  const normal = normalSource(seed);
  let exact = initial;
  let euler = initial;
  const rows = [{
    time: 0,
    exact,
    euler,
    deltaW: 0,
    weightedNoise: 0
  }];
  for (let index = 1; index <= steps; index += 1) {
    const deltaW = Math.sqrt(step) * normal();
    const weightedNoise = moments.coefficient * deltaW + Math.sqrt(moments.residualVariance) * normal();
    exact = target + (exact - target) * moments.attenuation + eta * weightedNoise;
    euler += theta * (target - euler) * step + eta * deltaW;
    rows.push({
      time: index * step,
      exact,
      euler,
      deltaW,
      weightedNoise
    });
  }
  return freeze({
    rows,
    step,
    steps,
    horizon,
    moments
  });
}
function exponentialTail(value, firstPower) {
  let term = 1;
  for (let k = 1; k <= firstPower; k += 1) term *= value / k;
  let sum = term;
  for (let k = firstPower + 1; k <= 120; k += 1) {
    term *= value / k;
    sum += term;
    if (Math.abs(term) <= Number.EPSILON * Math.max(Math.abs(sum), Number.MIN_VALUE)) break;
  }
  return sum;
}
function log1pMinusValue(value) {
  if (Math.abs(value) >= 1e-4) return Math.log1p(value) - value;
  let power = value * value;
  let sum = -power / 2;
  for (let k = 3; k < 30; k += 1) {
    power *= value;
    const term = (k % 2 ? 1 : -1) * power / k;
    sum += term;
    if (Math.abs(term) <= Number.EPSILON * Math.max(Math.abs(sum), Number.MIN_VALUE)) break;
  }
  return sum;
}
export function gbmErrorMoments({
  mu = 0.4,
  sigma = 0.6,
  initial = 1,
  horizon = 1,
  steps = 16,
  method = 'euler'
} = {}) {
  growthContract(mu, sigma, initial, horizon);
  finite(horizon, 0.25, 2, 'Error experiment horizon');
  integer(steps, 1, 512, 'Error step count');
  if (!['euler', 'milstein'].includes(method)) throw new RangeError('Unknown method.');
  const h = horizon / steps;
  const u = 1 + mu * h;
  const s = sigma * sigma * h;
  const r = s + (method === 'milstein' ? s * s / 2 : 0);
  const factorSecondMoment = u * u + r;
  const exactMean = initial * Math.exp(mu * horizon);
  const meanBias = u === 0 ? -exactMean : exactMean * Math.expm1(steps * log1pMinusValue(mu * h));
  const numericalMean = exactMean + meanBias;
  const exactSecond = initial * initial * Math.exp((2 * mu + sigma * sigma) * horizon);
  const driftStep = mu * h;
  const secondExponent = 2 * driftStep + s;
  const secondRemainder = driftStep * driftStep + 2 * driftStep * s + (method === 'euler' ? s * s / 2 : 0) + exponentialTail(secondExponent, 3);
  const secondRatioChange = -secondRemainder * Math.exp(-secondExponent);
  const secondLogRatio = factorSecondMoment === 0 ? -Infinity : Math.log1p(secondRatioChange);
  const secondBias = exactSecond * Math.expm1(steps * secondLogRatio);
  const numericalSecond = exactSecond + secondBias;
  let mse;
  let correlationLoss;
  if (sigma === 0) {
    mse = meanBias * meanBias;
    correlationLoss = 0;
  } else {
    // Hermite orthogonality gives this nonnegative local correlation defect.
    // It avoids E(Y^2)+E(X^2)-2E(XY), which cancels at fine resolution.
    const tail = exponentialTail(s, method === 'milstein' ? 3 : 2);
    const defect = Math.exp(-s) * (tail + r * (mu * h) ** 2 / factorSecondMoment);
    if (!(defect >= 0 && defect < 1)) throw new RangeError('The correlation defect is not representable.');
    correlationLoss = -Math.expm1(steps / 2 * Math.log1p(-defect));
    const exactLogNorm = Math.log(initial) + (mu + sigma * sigma / 2) * horizon;
    const normLogRatio = steps / 2 * secondLogRatio;
    const numericalLogNorm = exactLogNorm + normLogRatio;
    const normDifference = Math.exp(exactLogNorm) * Math.expm1(normLogRatio);
    mse = normDifference * normDifference + 2 * Math.exp(exactLogNorm + numericalLogNorm) * correlationLoss;
  }
  return freeze({
    mu,
    sigma,
    initial,
    horizon,
    steps,
    h,
    method,
    exactMean,
    numericalMean,
    exactSecond,
    numericalSecond,
    meanBias,
    secondBias,
    mse,
    rms: Math.sqrt(mse),
    correlationLoss,
    eulerMeanSquareMultiplier: (1 + mu * h) ** 2 + s
  });
}
export function sampledErrors({
  mu = 0.4,
  sigma = 0.6,
  initial = 1,
  horizon = 1,
  steps = 16,
  samples = 512,
  seed = 13,
  method = 'euler'
} = {}) {
  const analytic = gbmErrorMoments({
    mu,
    sigma,
    initial,
    horizon,
    steps,
    method
  });
  integer(samples, 16, 4096, 'Sample count');
  if (samples * steps > 1048576) throw new RangeError('Keep this experiment below 1,048,576 updates.');
  const normal = normalSource(seed);
  let biasMean = 0;
  let biasM2 = 0;
  let squareMean = 0;
  let squareM2 = 0;
  for (let sample = 1; sample <= samples; sample += 1) {
    let value = initial;
    let brownian = 0;
    for (let index = 0; index < steps; index += 1) {
      const dw = Math.sqrt(analytic.h) * normal();
      value += mu * value * analytic.h + sigma * value * dw + (method === 'milstein' ? sigma * sigma * value * (dw * dw - analytic.h) / 2 : 0);
      brownian += dw;
    }
    const exact = initial * Math.exp((mu - sigma * sigma / 2) * horizon + sigma * brownian);
    const error = value - exact;
    const biasDelta = error - biasMean;
    biasMean += biasDelta / sample;
    biasM2 += biasDelta * (error - biasMean);
    const square = error * error;
    const squareDelta = square - squareMean;
    squareMean += squareDelta / sample;
    squareM2 += squareDelta * (square - squareMean);
  }
  return freeze({
    analytic,
    samples,
    seed,
    bias: biasMean,
    biasStandardError: Math.sqrt(biasM2 / (samples - 1) / samples),
    mse: squareMean,
    rms: Math.sqrt(squareMean),
    mseStandardError: Math.sqrt(squareM2 / (samples - 1) / samples)
  });
}
