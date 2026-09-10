// Finite educational calculations, not arbitrary-range probability software.
// Weight inputs have 2–8 entries: exact zero or [1e-8,100]. All snapshots are frozen.
export const DIVERGENCE_NAMES = Object.freeze({
  kl: 'KL(P ∥ Q), nats',
  reverse: 'KL(Q ∥ P), nats',
  js: 'Jensen–Shannon, nats',
  hellinger: 'Squared Hellinger H²',
  chi: 'Pearson χ²(P ∥ Q)',
  tv: 'Total variation'
});
export const ORIGINAL_P = Object.freeze([7, 2, 1, 0]);
export const ORIGINAL_Q = Object.freeze([4, 5, 1, 0]);
export const OBSERVER_POSITIONS = Object.freeze([-3, -1, 1, 3]);
export const PERMUTATION_POOL = Object.freeze([-2, -1.5, -1, -0.5, 0.5, 1, 1.5, 2]);
function freeze(value) {
  if (value && typeof value === 'object') {
    Object.values(value).forEach(freeze);
    Object.freeze(value);
  }
  return value;
}
function bounded(value, minimum, maximum, label) {
  if (!Number.isFinite(value) || value < minimum || value > maximum) {
    throw new RangeError(`${label} must be between ${minimum} and ${maximum}.`);
  }
}
export function normalizeDivergenceWeights(weights) {
  if (!Array.isArray(weights) || weights.length < 2 || weights.length > 8) {
    throw new RangeError('Use 2–8 weights.');
  }
  for (const weight of weights) {
    bounded(weight, 0, 100, 'Weight');
    if (weight > 0 && weight < 1e-8) throw new RangeError('Positive weights must be at least 0.00000001 in this finite calculator.');
  }
  const total = weights.reduce((sum, value) => sum + value, 0);
  if (total === 0) throw new RangeError('At least one weight must be positive.');
  return weights.map(weight => weight / total);
}
export function parseDivergenceWeights(text) {
  const parts = text.trim().split(/[\s,]+/);
  if (parts.length !== 4 || parts.some(part => !/^\d+$/.test(part))) {
    throw new RangeError('Enter four whole-number weights from 0 to 100.');
  }
  const values = parts.map(Number);
  normalizeDivergenceWeights(values);
  return values;
}

// q f(p/q) for f(t)=t ln(t)-t+1. The linear terms cancel in the total.
function centeredKlContribution(p, q) {
  if (p === 0) return q;
  if (q === 0) return Infinity;
  const delta = (p - q) / q;
  if (Math.abs(delta) < 0.01) {
    let power = delta * delta;
    let sum = 0;
    for (let order = 2; order <= 14; order += 1) {
      sum += (order % 2 === 0 ? 1 : -1) * power / (order * (order - 1));
      power *= delta;
    }
    return q * sum;
  }
  return p * (Math.log(p) - Math.log(q)) - p + q;
}
function contribution(p, q, kind) {
  if (kind === 'kl') return centeredKlContribution(p, q);
  if (kind === 'reverse') return centeredKlContribution(q, p);
  if (kind === 'js') {
    const middle = (p + q) / 2;
    return (centeredKlContribution(p, middle) + centeredKlContribution(q, middle)) / 2;
  }
  if (kind === 'hellinger') return (Math.sqrt(p) - Math.sqrt(q)) ** 2 / 2;
  if (kind === 'chi') return q === 0 ? p === 0 ? 0 : Infinity : (p - q) ** 2 / q;
  if (kind === 'tv') return Math.abs(p - q) / 2;
  throw new RangeError('Choose a supported divergence.');
}
function knownLawValues(p, q) {
  return Object.fromEntries(Object.keys(DIVERGENCE_NAMES).map(kind => [kind, p.reduce((sum, mass, index) => sum + contribution(mass, q[index], kind), 0)]));
}
export function divergenceState(pWeights = ORIGINAL_P, qWeights = ORIGINAL_Q, kind = 'kl') {
  if (!Object.hasOwn(DIVERGENCE_NAMES, kind)) throw new RangeError('Choose a supported divergence.');
  const p = normalizeDivergenceWeights(pWeights);
  const q = normalizeDivergenceWeights(qWeights);
  if (p.length !== q.length) throw new RangeError('Both laws need the same labeled outcomes.');
  const rows = p.map((mass, index) => ({
    label: String.fromCharCode(65 + index),
    p: mass,
    q: q[index],
    ratio: q[index] === 0 ? mass === 0 ? null : Infinity : mass / q[index],
    penalty: contribution(mass, q[index], kind)
  }));
  return freeze({
    p,
    q,
    kind,
    rows,
    values: knownLawValues(p, q),
    total: rows.reduce((sum, row) => sum + row.penalty, 0)
  });
}
// Channel probabilities are exact zero or [1e-8, 1]. Combined with the
// 2–8 input weights in {0} union [1e-8, 100], each positive processed mass
// is at least 1.25e-19. Subnormal products must not invent lost support.
export function processDivergence(pWeights, qWeights, channel) {
  const p = normalizeDivergenceWeights(pWeights);
  const q = normalizeDivergenceWeights(qWeights);
  if (p.length !== q.length || !Array.isArray(channel) || channel.length !== p.length) throw new RangeError('Channel rows must match both laws.');
  const count = channel[0]?.length;
  if (!Number.isInteger(count) || count < 1 || count > 8) throw new RangeError('Use 1–8 channel outputs.');
  for (const row of channel) {
    if (!Array.isArray(row) || row.length !== count || row.some(value => !Number.isFinite(value) || value < 0 || value > 1) || Math.abs(row.reduce((sum, value) => sum + value, 0) - 1) > 1e-12) {
      throw new RangeError('Every channel row must be a probability vector.');
    }
    if (row.some(value => value > 0 && value < 1e-8)) {
      throw new RangeError('Positive channel probabilities must be at least 0.00000001 in this finite calculator; exact zero remains allowed.');
    }
  }
  const push = input => Array.from({
    length: count
  }, (_, output) => input.reduce((sum, mass, index) => sum + mass * channel[index][output], 0));
  const outputP = push(p),
    outputQ = push(q);
  return freeze({
    p,
    q,
    channel: channel.map(row => [...row]),
    outputP,
    outputQ,
    before: knownLawValues(p, q),
    after: knownLawValues(outputP, outputQ)
  });
}
export function observableState(pWeights = [4, 1, 1, 4], qWeights = [1, 4, 4, 1], positions = OBSERVER_POSITIONS, kind = 'event') {
  const p = normalizeDivergenceWeights(pWeights);
  const q = normalizeDivergenceWeights(qWeights);
  if (p.length !== q.length || positions.length !== p.length) throw new RangeError('Laws and positions must have the same size.');
  positions.forEach((value, index) => {
    bounded(value, -10, 10, 'Position');
    if (index > 0 && value <= positions[index - 1]) throw new RangeError('Positions must be strictly increasing.');
  });
  if (!['event', 'linear', 'lipschitz'].includes(kind)) throw new RangeError('Choose a supported observer class.');
  const difference = p.map((mass, index) => mass - q[index]);
  const meanDifference = difference.reduce((sum, value, index) => sum + value * positions[index], 0);
  const cumulative = [];
  let running = 0;
  const lipschitz = [0];
  for (let index = 0; index < p.length - 1; index += 1) {
    running += difference[index];
    cumulative.push(running);
    lipschitz.push(lipschitz[index] - Math.sign(running) * (positions[index + 1] - positions[index]));
  }
  const event = difference.map(value => Number(value > 0));
  const linear = positions.map(value => Math.sign(meanDifference) * value);
  const scores = kind === 'event' ? event : kind === 'linear' ? linear : lipschitz;
  const contributions = scores.map((score, index) => score * difference[index]);
  const value = contributions.reduce((sum, term) => sum + term, 0);
  const tv = difference.reduce((sum, term) => sum + Math.abs(term), 0) / 2;
  const w1 = cumulative.reduce((sum, amount, index) => sum + Math.abs(amount) * (positions[index + 1] - positions[index]), 0);
  return freeze({
    p,
    q,
    positions: [...positions],
    kind,
    scores,
    difference,
    cumulative,
    contributions,
    value,
    tv,
    w1,
    linearValue: Math.abs(meanDifference),
    optimalAccuracy: (1 + tv) / 2
  });
}
export function movingAtomState(displacement = 0.5, bandwidth = 1) {
  bounded(displacement, 0, 3, 'Displacement');
  bounded(bandwidth, 0.2, 3, 'Kernel bandwidth');
  const unequal = displacement !== 0;
  const mmdSquared = -2 * Math.expm1(-((displacement / bandwidth) ** 2) / 2);
  return freeze({
    displacement,
    bandwidth,
    kl: unequal ? Infinity : 0,
    tv: Number(unequal),
    js: unequal ? Math.LN2 : 0,
    w1: displacement,
    mmdSquared,
    mmd: Math.sqrt(mmdSquared)
  });
}
export function parseKernelSamples(text) {
  const parts = text.trim().split(/[\s,]+/);
  if (parts.length < 2 || parts.length > 6 || parts.some(part => !/^[+-]?(?:\d+(?:\.\d*)?|\.\d+)$/.test(part))) throw new RangeError('Enter 2–6 numeric observations from −3 to 3.');
  const values = parts.map(Number);
  values.forEach(value => bounded(value, -3, 3, 'Observation'));
  return values;
}
function kernel(x, y, kind, bandwidth) {
  if (kind === 'linear') return x * y;
  if (kind === 'quadratic') return x * y + x * x * y * y;
  return Math.exp(-(((x - y) / bandwidth) ** 2) / 2);
}
function sampleMmd(x, y, kind, bandwidth) {
  const xx = x.map(a => x.map(b => kernel(a, b, kind, bandwidth)));
  const yy = y.map(a => y.map(b => kernel(a, b, kind, bandwidth)));
  const xy = x.map(a => y.map(b => kernel(a, b, kind, bandwidth)));
  const sum = matrix => matrix.flat().reduce((total, value) => total + value, 0);
  const xxSum = sum(xx),
    yySum = sum(yy),
    xySum = sum(xy);
  const withinX = xxSum / x.length ** 2;
  const withinY = yySum / y.length ** 2;
  const cross = xySum / (x.length * y.length);
  const rawBiasedSquared = withinX + withinY - 2 * cross;
  if (rawBiasedSquared < -1e-12) throw new Error('Unexpected negative empirical kernel norm.');
  const biasedSquared = Math.max(0, rawBiasedSquared);
  const diagonalX = xx.reduce((total, row, index) => total + row[index], 0);
  const diagonalY = yy.reduce((total, row, index) => total + row[index], 0);
  const unbiasedSquared = (xxSum - diagonalX) / (x.length * (x.length - 1)) + (yySum - diagonalY) / (y.length * (y.length - 1)) - 2 * cross;
  return {
    xx,
    yy,
    xy,
    withinX,
    withinY,
    cross,
    rawBiasedSquared,
    biasedSquared,
    unbiasedSquared,
    mmd: Math.sqrt(biasedSquared)
  };
}
export function kernelWitnessState(x = [-2, -2, 2, 2], y = [-1, -1, 1, 1], bandwidth = 1, kind = 'rbf') {
  if (!Array.isArray(x) || !Array.isArray(y) || x.length < 2 || x.length > 6 || y.length < 2 || y.length > 6) throw new RangeError('Each group needs 2–6 observations.');
  [...x, ...y].forEach(value => bounded(value, -3, 3, 'Observation'));
  bounded(bandwidth, 0.2, 3, 'Kernel bandwidth');
  if (!['rbf', 'linear', 'quadratic'].includes(kind)) throw new RangeError('Choose a supported kernel.');
  const values = sampleMmd(x, y, kind, bandwidth);
  const at = point => x.reduce((sum, value) => sum + kernel(value, point, kind, bandwidth), 0) / x.length - y.reduce((sum, value) => sum + kernel(value, point, kind, bandwidth), 0) / y.length;
  const witness = Array.from({
    length: 121
  }, (_, index) => {
    const position = -3 + index / 20;
    return {
      position,
      value: at(position)
    };
  });
  const witnessGap = x.reduce((sum, value) => sum + at(value), 0) / x.length - y.reduce((sum, value) => sum + at(value), 0) / y.length;
  return freeze({
    x: [...x],
    y: [...y],
    bandwidth,
    kind,
    ...values,
    witness,
    witnessGap
  });
}
function combinations(count, chosen) {
  const result = [];
  function visit(start, selection) {
    if (selection.length === chosen) {
      result.push([...selection]);
      return;
    }
    for (let index = start; index <= count - (chosen - selection.length); index += 1) {
      selection.push(index);
      visit(index + 1, selection);
      selection.pop();
    }
  }
  visit(0, []);
  return result;
}
export function permutationMmdState(bandwidth = 1, selected = 0) {
  bounded(bandwidth, 0.2, 3, 'Kernel bandwidth');
  if (!Number.isInteger(selected) || selected < 0 || selected >= 70) throw new RangeError('Choose allocation 0–69.');
  const allocations = combinations(8, 4).map(indices => {
    const x = indices.map(index => PERMUTATION_POOL[index]);
    const y = PERMUTATION_POOL.filter((_, index) => !indices.includes(index));
    return {
      indices,
      x,
      y,
      statistic: sampleMmd(x, y, 'rbf', bandwidth).biasedSquared
    };
  });
  const observed = allocations[0].statistic;
  // Numerical ties within 1e-12 are included conservatively in the upper tail.
  const tailCount = allocations.filter(allocation => allocation.statistic >= observed - 1e-12).length;
  return freeze({
    bandwidth,
    selected,
    pool: [...PERMUTATION_POOL],
    allocations,
    current: allocations[selected],
    observed,
    tailCount,
    pValue: tailCount / 70
  });
}
export function variationalDivergenceState(scale = 1, offset = 0) {
  bounded(scale, 0, 1.5, 'Critic scale');
  bounded(offset, -2, 2, 'Critic offset');
  const p = [0.7, 0.2, 0.1],
    q = [0.4, 0.5, 0.1];
  const rows = p.map((mass, index) => {
    const logRatio = Math.log(mass) - Math.log(q[index]);
    const optimum = 1 + logRatio;
    const score = 1 + scale * logRatio + offset;
    const reward = mass * score;
    const penalty = q[index] * Math.exp(score - 1);
    return {
      label: String.fromCharCode(65 + index),
      p: mass,
      q: q[index],
      optimum,
      score,
      reward,
      penalty,
      bound: reward - penalty
    };
  });
  const bound = rows.reduce((sum, row) => sum + row.bound, 0);
  const truth = p.reduce((sum, mass, index) => sum + mass * (Math.log(mass) - Math.log(q[index])), 0);
  return freeze({
    p,
    q,
    scale,
    offset,
    rows,
    bound,
    truth,
    gap: truth - bound,
    shiftedOptimumGap: scale === 1 ? Math.expm1(offset) - offset : null
  });
}
