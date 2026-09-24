// Topic-owned, bounded teaching models. Distribution values are checked against SciPy.
export const PAIRED_OLD = Object.freeze([102, 110, 98, 105, 100]);
export const PAIRED_NEW = Object.freeze([100, 106, 99, 102, 98]);
export const PAIRED_DIFFERENCES = Object.freeze([2, 4, -1, 3, 2]);
export const NORMAL_CRITICAL = Object.freeze({
  90: 1.6448536269514722,
  95: 1.959963984540054,
  99: 2.5758293035489004
});
function bounded(value, low, high, name, integer = false) {
  if (typeof value !== 'number' || !Number.isFinite(value) || value < low || value > high || integer && !Number.isInteger(value)) {
    throw new RangeError(`${name} must be ${integer ? 'an integer' : 'a finite number'} from ${low} to ${high}.`);
  }
  return value;
}
function option(value, choices, name) {
  if (!choices.includes(value)) throw new RangeError(`${name} must be one of ${choices.join(', ')}.`);
  return value;
}
export function inferenceNumber(value, digits = 3) {
  if (value === Infinity) return '+∞';
  if (value === -Infinity) return '−∞';
  if (!Number.isFinite(value)) return 'undefined';
  if (value === 0) return '0';
  if (Math.abs(value) < 10 ** -digits) return value.toExponential(2);
  return Number(value.toFixed(digits)).toString();
}
export function inferencePercent(value) {
  if (value > 0 && value < 0.000001) return `${inferenceNumber(value * 100, 5)}%`;
  return `${(value * 100).toFixed(2)}%`;
}
export function sampleSummary(values) {
  if (!Array.isArray(values) || values.length < 2 || values.length > 10000 || values.some(value => typeof value !== 'number' || !Number.isFinite(value) || Math.abs(value) > 1e6)) {
    throw new RangeError('Use 2–10,000 finite observations with absolute values at most 1,000,000.');
  }
  const n = values.length;
  const mean = values.reduce((sum, value) => sum + value, 0) / n;
  const residuals = values.map(value => value - mean);
  const sumSquares = residuals.reduce((sum, value) => sum + value * value, 0);
  const variance = sumSquares / (n - 1);
  const sd = Math.sqrt(variance);
  return {
    n,
    mean,
    residuals,
    sumSquares,
    variance,
    sd,
    se: sd / Math.sqrt(n),
    df: n - 1
  };
}

// Exact Student t4 density and a cancellation-resistant closed-form upper tail.
export function t4Density(t) {
  bounded(t, -1e6, 1e6, 't');
  return 3 / 8 * (1 + t * t / 4) ** -2.5;
}
export function t4Survival(t) {
  bounded(t, -1e6, 1e6, 't');
  const magnitude = Math.abs(t);
  const root = Math.hypot(magnitude, 2);
  const s = magnitude / root;
  const oneMinusS = 4 / (root * (root + magnitude));
  const upper = oneMinusS * oneMinusS * (2 + s) / 4;
  return t >= 0 ? upper : 1 - upper;
}
export function t4Quantile(probability) {
  bounded(probability, 0.001, 0.999, 'Quantile probability');
  if (probability === 0.5) return 0;
  if (probability < 0.5) return -t4Quantile(1 - probability);
  let low = 0,
    high = 100;
  for (let iteration = 0; iteration < 64; iteration++) {
    const middle = (low + high) / 2;
    if (t4Survival(middle) > 1 - probability) low = middle;else high = middle;
  }
  return (low + high) / 2;
}
export function pairedTailState({
  shift = 0,
  spread = 1,
  reference = 0,
  alpha = 0.05,
  alternative = 'two-sided'
} = {}) {
  bounded(shift, -2, 3, 'Mean shift');
  bounded(spread, 0.25, 2, 'Spread multiplier');
  bounded(reference, -1, 2, 'Null reference');
  option(alpha, [0.01, 0.05, 0.1], 'Alpha');
  option(alternative, ['two-sided', 'greater', 'less'], 'Alternative');
  const values = PAIRED_DIFFERENCES.map(value => 2 + shift + spread * (value - 2));
  const summary = sampleSummary(values);
  const t = (summary.mean - reference) / summary.se;
  const p = alternative === 'two-sided' ? 2 * t4Survival(Math.abs(t)) : alternative === 'greater' ? t4Survival(t) : t4Survival(-t);
  const critical = t4Quantile(1 - alpha / (alternative === 'two-sided' ? 2 : 1));
  const low = alternative === 'less' ? -Infinity : summary.mean - critical * summary.se;
  const high = alternative === 'greater' ? Infinity : summary.mean + critical * summary.se;
  return {
    ...summary,
    values,
    reference,
    alternative,
    alpha,
    t,
    p,
    critical,
    low,
    high,
    reject: p < alpha,
    nullInside: low <= reference && reference <= high
  };
}
function uniformGenerator(seed) {
  let state = seed >>> 0;
  return () => {
    state = Math.imul(1664525, state) + 1013904223 >>> 0;
    return (state + 0.5) / 4294967296;
  };
}
function normalDraw(random) {
  return Math.sqrt(-2 * Math.log(random())) * Math.cos(2 * Math.PI * random());
}
export function intervalCoverageState(n = 25, level = 95, batch = 0, varianceMode = 'known') {
  bounded(n, 5, 100, 'Observations per experiment', true);
  if (n % 5 !== 0) throw new RangeError('Use a sample size in steps of five.');
  option(level, [90, 95, 99], 'Confidence level');
  bounded(batch, 0, 1000000, 'Batch', true);
  option(varianceMode, ['known', 'estimated'], 'Variance mode');
  // The known-σ stream preserves the approved original 40-row pilot exactly.
  const meanRandom = uniformGenerator(407 + batch * 97);
  const varianceRandom = uniformGenerator(93017 + batch * 193);
  const se = 10 / Math.sqrt(n);
  const critical = varianceMode === 'known' ? NORMAL_CRITICAL[level] : T_CRITICAL[n][level];
  const intervals = Array.from({
    length: 40
  }, (_, index) => {
    const z = normalDraw(meanRandom);
    const mean = 100 + se * z;
    let sd = 10;
    if (varianceMode === 'estimated') {
      let chiSquare = 0;
      for (let degree = 0; degree < n - 1; degree++) chiSquare += normalDraw(varianceRandom) ** 2;
      sd = 10 * Math.sqrt(chiSquare / (n - 1));
    }
    const estimatedSe = sd / Math.sqrt(n);
    const halfWidth = critical * estimatedSe;
    const low = mean - halfWidth,
      high = mean + halfWidth;
    return {
      index: index + 1,
      mean,
      z,
      sd,
      se: estimatedSe,
      low,
      high,
      covers: low <= 100 && high >= 100
    };
  });
  return {
    n,
    level,
    batch,
    varianceMode,
    critical,
    se,
    intervals,
    covered: intervals.filter(row => row.covers).length
  };
}
export const EFFECT_PRESETS = Object.freeze({
  original: {
    label: 'Original paired data',
    mean: 2,
    spread: 1
  },
  shifted: {
    label: 'Each saving +1 ms',
    mean: 3,
    spread: 1
  },
  centered: {
    label: 'Mean zero, original spread',
    mean: 0,
    spread: 1
  },
  precise: {
    label: 'Mean zero, quarter spread',
    mean: 0,
    spread: 0.25
  }
});
export function practicalEffectState(preset = 'original', tolerance = 0.5) {
  option(preset, Object.keys(EFFECT_PRESETS), 'Dataset');
  bounded(tolerance, 0.25, 2, 'Equivalence tolerance');
  const configuration = EFFECT_PRESETS[preset];
  const values = PAIRED_DIFFERENCES.map(value => configuration.mean + configuration.spread * (value - 2));
  const summary = sampleSummary(values);
  const width95 = t4Quantile(0.975) * summary.se;
  const width90 = t4Quantile(0.95) * summary.se;
  const lowerP = t4Survival((summary.mean + tolerance) / summary.se);
  const upperP = t4Survival((tolerance - summary.mean) / summary.se);
  return {
    ...summary,
    values,
    preset,
    tolerance,
    low95: summary.mean - width95,
    high95: summary.mean + width95,
    low90: summary.mean - width90,
    high90: summary.mean + width90,
    pZero: 2 * t4Survival(Math.abs(summary.mean / summary.se)),
    pUseful: t4Survival((summary.mean - 1) / summary.se),
    lowerP,
    upperP,
    equivalenceP: Math.max(lowerP, upperP),
    equivalent: lowerP < 0.05 && upperP < 0.05
  };
}

// 96-point Gauss–Legendre quadrature on [0, 12] for the positive normal tail.
// sf(z) = phi(z) * integral exp(-z*s - s²/2) ds, for z >= 0.
// The omitted integration tail is below 2e-33 at z=0 and decreases for z>0.
function legendreQuadrature(order) {
  const rows = [];
  for (let index = 1; index <= order; index++) {
    let x = Math.cos(Math.PI * (index - 0.25) / (order + 0.5));
    let derivative = 0;
    for (let iteration = 0; iteration < 30; iteration++) {
      let previous = 1,
        current = x;
      for (let degree = 2; degree <= order; degree++) {
        const next = ((2 * degree - 1) * x * current - (degree - 1) * previous) / degree;
        previous = current;
        current = next;
      }
      derivative = order * (x * current - previous) / (x * x - 1);
      const next = x - current / derivative;
      if (Math.abs(next - x) < 2e-16) {
        x = next;
        break;
      }
      x = next;
    }
    rows.push({
      x: 6 * (x + 1),
      weight: 12 / ((1 - x * x) * derivative * derivative)
    });
  }
  return rows;
}
const NORMAL_QUADRATURE = legendreQuadrature(96);
export function normalDensity(z) {
  return Math.exp(-z * z / 2) / Math.sqrt(2 * Math.PI);
}
export function normalSurvival(z) {
  bounded(z, -40, 40, 'Normal coordinate');
  if (z === 0) return 0.5;
  if (z < 0) return 1 - normalSurvival(-z);
  const integral = NORMAL_QUADRATURE.reduce((sum, row) => sum + row.weight * Math.exp(-z * row.x - row.x * row.x / 2), 0);
  return normalDensity(z) * integral;
}
export function plannedPowerState(n = 25, sigma = 4, effect = 2, alpha = 0.05, alternative = 'greater') {
  bounded(n, 5, 200, 'Planned independent observations', true);
  bounded(sigma, 2, 10, 'Known population SD');
  bounded(effect, 0, 4, 'Hypothetical true saving');
  option(alpha, [0.01, 0.05, 0.1], 'Alpha');
  option(alternative, ['greater', 'two-sided'], 'Planning alternative');
  const critical = alternative === 'two-sided' ? NORMAL_CRITICAL[100 * (1 - alpha)] : {
    0.01: 2.3263478740408408,
    0.05: 1.6448536269514722,
    0.1: 1.2815515655446004
  }[alpha];
  const se = sigma / Math.sqrt(n);
  const boundary = critical * se;
  const rightPower = normalSurvival((boundary - effect) / se);
  const leftPower = alternative === 'two-sided' ? normalSurvival((boundary + effect) / se) : 0;
  const power = rightPower + leftPower;
  return {
    n,
    sigma,
    effect,
    alpha,
    alternative,
    critical,
    se,
    boundary,
    power,
    beta: 1 - power,
    rightPower,
    leftPower
  };
}
export function predictionWidths(n = 25, level = 95, mean = 100, sigma = 10) {
  bounded(n, 5, 100, 'Sample size', true);
  option(level, [90, 95, 99], 'Confidence level');
  bounded(mean, -1000, 1000, 'Sample mean');
  bounded(sigma, 0.1, 100, 'Known population SD');
  const critical = NORMAL_CRITICAL[level];
  return {
    mean,
    n,
    sigma,
    level,
    meanHalfWidth: critical * sigma / Math.sqrt(n),
    predictionHalfWidth: critical * sigma * Math.sqrt(1 + 1 / n)
  };
}
export function familyErrorState(count = 20, alpha = 0.05) {
  bounded(count, 1, 100, 'Number of tests', true);
  option(alpha, [0.01, 0.05, 0.1], 'Alpha');
  return {
    count,
    alpha,
    independentFamilyError: -Math.expm1(count * Math.log1p(-alpha)),
    bonferroniPerTest: alpha / count,
    independentBonferroniError: -Math.expm1(count * Math.log1p(-alpha / count))
  };
}
function binomialChoose(n, k) {
  let value = 1;
  for (let index = 1; index <= k; index++) value = value * (n - index + 1) / index;
  return Math.round(value);
}
export function fairCoinPValue(n, heads) {
  bounded(n, 1, 12, 'Number of flips', true);
  bounded(heads, 0, n, 'Heads', true);
  const distance = Math.abs(2 * heads - n);
  let count = 0;
  for (let k = 0; k <= n; k++) if (Math.abs(2 * k - n) >= distance) count += binomialChoose(n, k);
  return count / 2 ** n;
}
export function optionalLooksState(maximum = 12, alpha = 0.05) {
  bounded(maximum, 1, 12, 'Maximum looks', true);
  option(alpha, [0.01, 0.05, 0.1], 'Alpha');
  let survivingCounts = [1],
    cumulativeError = 0;
  const rows = [];
  for (let n = 1; n <= maximum; n++) {
    const next = Array(n + 1).fill(0);
    for (let heads = 0; heads < survivingCounts.length; heads++) {
      next[heads] += survivingCounts[heads];
      next[heads + 1] += survivingCounts[heads];
    }
    let firstRejectCount = 0,
      fixedRejectCount = 0;
    for (let heads = 0; heads <= n; heads++) if (fairCoinPValue(n, heads) < alpha) {
      firstRejectCount += next[heads];
      next[heads] = 0;
      fixedRejectCount += binomialChoose(n, heads);
    }
    cumulativeError += firstRejectCount / 2 ** n;
    rows.push({
      n,
      firstRejectCount,
      firstRejectProbability: firstRejectCount / 2 ** n,
      fixedRejectCount,
      fixedError: fixedRejectCount / 2 ** n,
      cumulativeError
    });
    survivingCounts = next;
  }
  return {
    maximum,
    alpha,
    rows,
    fixedError: rows.at(-1).fixedError,
    cumulativeError
  };
}
export function clusterPrecisionState(clusters = 5, repeats = 2000, correlation = 0.2) {
  bounded(clusters, 2, 1000, 'Independent clusters', true);
  bounded(repeats, 1, 10000, 'Equal observations per cluster', true);
  bounded(correlation, 0, 1, 'Within-cluster correlation');
  const n = clusters * repeats;
  const designEffect = 1 + (repeats - 1) * correlation;
  return {
    clusters,
    repeats,
    correlation,
    n,
    designEffect,
    effectiveN: n / designEffect,
    seRatio: Math.sqrt(designEffect)
  };
}
export const SIGN_FLIP_PRESETS = Object.freeze({
  original: PAIRED_DIFFERENCES,
  allPositive: Object.freeze([1, 1, 1, 1, 1]),
  centered: Object.freeze([0, 2, -3, 1, 0])
});
export function signFlipState(preset = 'original', mask = 0) {
  option(preset, Object.keys(SIGN_FLIP_PRESETS), 'Sign-flip dataset');
  bounded(mask, 0, 31, 'Assignment', true);
  const values = SIGN_FLIP_PRESETS[preset];
  const observedSum = values.reduce((sum, value) => sum + value, 0);
  const assignments = Array.from({
    length: 32
  }, (_, index) => {
    const signs = values.map((_, bit) => index & 1 << bit ? -1 : 1);
    const transformed = values.map((value, bit) => value * signs[bit]);
    const sum = transformed.reduce((total, value) => total + value, 0);
    return {
      mask: index,
      signs,
      transformed,
      sum,
      mean: sum / 5,
      extreme: Math.abs(sum) >= Math.abs(observedSum)
    };
  });
  const counts = new Map();
  for (const row of assignments) counts.set(row.sum, (counts.get(row.sum) || 0) + 1);
  const histogram = [...counts].sort((a, b) => a[0] - b[0]).map(([sum, count]) => ({
    mean: sum / 5,
    sum,
    count,
    extreme: Math.abs(sum) >= Math.abs(observedSum)
  }));
  const extremeCount = assignments.filter(row => row.extreme).length;
  return {
    preset,
    values,
    observedSum,
    observedMean: observedSum / 5,
    assignments,
    current: assignments[mask],
    histogram,
    extremeCount,
    p: extremeCount / 32
  };
}
export function binomialMasses(n, p) {
  bounded(n, 1, 20, 'Binomial sample size', true);
  bounded(p, 0, 1, 'True success probability');
  return Array.from({
    length: n + 1
  }, (_, k) => binomialChoose(n, k) * p ** k * (1 - p) ** (n - k));
}
function exactBinomialLower(k, n, alpha) {
  if (k === 0) return 0;
  let low = 0,
    high = 1;
  for (let iteration = 0; iteration < 60; iteration++) {
    const middle = (low + high) / 2;
    const upperTail = binomialMasses(n, middle).slice(k).reduce((sum, value) => sum + value, 0);
    if (upperTail < alpha / 2) low = middle;else high = middle;
  }
  return (low + high) / 2;
}
export function proportionIntervals(k, n, level = 95) {
  bounded(n, 1, 20, 'Binomial sample size', true);
  bounded(k, 0, n, 'Observed successes', true);
  option(level, [90, 95, 99], 'Confidence level');
  const pHat = k / n,
    z = NORMAL_CRITICAL[level];
  const waldMargin = z * Math.sqrt(pHat * (1 - pHat) / n);
  const denominator = 1 + z * z / n;
  const center = (pHat + z * z / (2 * n)) / denominator;
  const margin = z * Math.sqrt(pHat * (1 - pHat) / n + z * z / (4 * n * n)) / denominator;
  const alpha = 1 - level / 100;
  return {
    wald: [pHat - waldMargin, pHat + waldMargin],
    wilson: [k === 0 ? 0 : center - margin, k === n ? 1 : center + margin],
    exact: [exactBinomialLower(k, n, alpha), k === n ? 1 : 1 - exactBinomialLower(n - k, n, alpha)]
  };
}
export function proportionCoverageState(n = 10, observed = 0, truth = 0.2) {
  bounded(n, 5, 20, 'Binomial sample size', true);
  bounded(observed, 0, n, 'Observed successes', true);
  bounded(truth, 0, 1, 'Hypothetical true probability');
  const masses = binomialMasses(n, truth);
  const rows = masses.map((mass, k) => {
    const intervals = proportionIntervals(k, n);
    const covers = Object.fromEntries(Object.entries(intervals).map(([method, [low, high]]) => [method, low <= truth && truth <= high]));
    return {
      k,
      mass,
      intervals,
      covers
    };
  });
  const coverage = Object.fromEntries(['wald', 'wilson', 'exact'].map(method => [method, rows.reduce((sum, row) => sum + (row.covers[method] ? row.mass : 0), 0)]));
  return {
    n,
    observed,
    truth,
    rows,
    current: rows[observed],
    coverage
  };
}

// Generated below from scipy.stats.t.ppf((1 + confidence)/2, n - 1),
// SciPy 1.18.1. These are lookup constants for the explicitly bounded coverage UI.
const T_CRITICAL = {
  5: {
    90: 2.1318467863266495,
    95: 2.7764451051977934,
    99: 4.604094871349992
  },
  10: {
    90: 1.8331129326562368,
    95: 2.262157162798205,
    99: 3.249835541592126
  },
  15: {
    90: 1.761310135774891,
    95: 2.144786687917804,
    99: 2.9768427343708344
  },
  20: {
    90: 1.7291328115213682,
    95: 2.0930240544083087,
    99: 2.8609346064649794
  },
  25: {
    90: 1.710882079909428,
    95: 2.0638985616280245,
    99: 2.796939504774456
  },
  30: {
    90: 1.6991270265334972,
    95: 2.045229642132704,
    99: 2.756385903670605
  },
  35: {
    90: 1.6909242551868546,
    95: 2.0322445093177186,
    99: 2.7283943670707203
  },
  40: {
    90: 1.6848751217112248,
    95: 2.022690920036761,
    99: 2.707913183517662
  },
  45: {
    90: 1.6802299765721167,
    95: 2.0153675744437636,
    99: 2.6922782656930218
  },
  50: {
    90: 1.6765508926168535,
    95: 2.0095752371292392,
    99: 2.679951973631552
  },
  55: {
    90: 1.6735649063521605,
    95: 2.0048792881880564,
    99: 2.6699847957348912
  },
  60: {
    90: 1.6710930321038948,
    95: 2.000995378088267,
    99: 2.661758752162967
  },
  65: {
    90: 1.6690130250240895,
    95: 1.997729654317693,
    99: 2.6548543374110847
  },
  70: {
    90: 1.6672385486685526,
    95: 1.9949454151072379,
    99: 2.6489767743886263
  },
  75: {
    90: 1.665706892734023,
    95: 1.9925434951809318,
    99: 2.6439128716530895
  },
  80: {
    90: 1.6643714091365505,
    95: 1.9904502102301285,
    99: 2.6395046274532206
  },
  85: {
    90: 1.663196679048909,
    95: 1.9886096669757085,
    99: 2.6356324580479606
  },
  90: {
    90: 1.6621553258697,
    95: 1.986978699506281,
    99: 2.6322041912000085
  },
  95: {
    90: 1.6612258552965113,
    95: 1.9855234418666037,
    99: 2.629147638261705
  },
  100: {
    90: 1.6603911560169906,
    95: 1.984216951586417,
    99: 2.626405457280827
  }
};
