// Small deterministic operators, independent of React and downloadable fit files.
export const dot = (a, b) => a.reduce((sum, value, i) => sum + value * b[i], 0);
const sum = values => values.reduce((a, b) => a + b, 0);
const zeros = (rows, columns) => Array.from({ length: rows }, () => Array(columns).fill(0));
export function normalizedScores(scores, legal = scores.map(() => true)) {
  if (!legal.some(Boolean)) return null;
  const maximum = Math.max(...scores.filter((_, i) => legal[i]));
  const mass = scores.map((value, i) => legal[i] ? Math.exp(value - maximum) : 0);
  return mass.map(value => value / sum(mass));
}
export function sparseRead(scores, values, legal) {
  const denseWeights = normalizedScores(scores);
  const weights = normalizedScores(scores, legal);
  const mix = row => values[0].map((_, axis) => sum(values.map((value, i) => value[axis] * row[i])));
  return { denseWeights, weights, dense: mix(denseWeights), output: weights ? mix(weights) : null,
    removedMass: sum(denseWeights.filter((_, i) => !legal[i])),
    contributions: weights ? values.map((value, i) => value.map(x => x * weights[i])) : null };
}
export function causalEdges(length, window, hub = -1) {
  return Array.from({ length }, (_, receiver) => Array.from({ length }, (_, donor) => donor <= receiver && (receiver - donor < window || receiver === hub || donor === hub)));
}
export function graphReach(mask, source, layers) {
  const stages = [mask.map((_, i) => i === source)];
  const paths = [mask.map((_, i) => i === source ? [source] : null)];
  for (let layer = 0; layer < layers; layer += 1) {
    const previous = paths.at(-1);
    const next = mask.map((row, receiver) => {
      const donor = row.findIndex((allowed, j) => (allowed || j === receiver) && previous[j]);
      return donor < 0 ? null : [...previous[donor], receiver];
    });
    paths.push(next); stages.push(next.map(Boolean));
  }
  return { stages, paths: paths.at(-1), edges: sum(mask.map(row => row.filter(Boolean).length)) };
}
export const memoryDefaults = {
  query: [1, 3], keys: [[1, 2], [2, 1], [1, 1]], values: [[1, -1], [0, 2], [3, 1]],
};
export function featureMemory(query, keys, values) {
  const matrix = zeros(query.length, values[0].length);
  const normalizer = Array(query.length).fill(0);
  const history = [];
  keys.forEach((key, record) => {
    const contribution = key.map(k => values[record].map(v => k * v));
    key.forEach((k, axis) => {
      normalizer[axis] += k;
      values[record].forEach((_, j) => { matrix[axis][j] += contribution[axis][j]; });
    });
    history.push({ contribution, matrix: matrix.map(row => [...row]), normalizer: [...normalizer] });
  });
  const denominator = dot(query, normalizer);
  const weights = denominator > 0 ? keys.map(key => dot(query, key) / denominator) : null;
  const numerator = values[0].map((_, axis) => sum(matrix.map((row, i) => row[axis] * query[i])));
  return { matrix, normalizer, denominator, numerator, weights,
    output: denominator > 0 ? numerator.map(value => value / denominator) : null, history };
}
export function randomFeatureRead(query, keys, values, projection) {
  const logs = vectors => vectors.map(vector => projection.map(row => dot(vector, row) - dot(vector, vector) / 2));
  const qLogs = logs(query), kLogs = logs(keys);
  const commonKeyScale = Math.max(...kLogs.flat());
  const qPhi = qLogs.map(row => row.map(value => Math.exp(value - Math.max(...row))));
  const kPhi = kLogs.map(row => row.map(value => Math.exp(value - commonKeyScale)));
  const exactWeights = query.map((q, receiver) => normalizedScores(keys.map(k => dot(q, k)), keys.map((_, donor) => donor <= receiver)));
  const weights = qPhi.map((q, receiver) => {
    const row = kPhi.map((key, donor) => donor <= receiver ? dot(q, key) : 0);
    return sum(row) > 0 ? row.map(value => value / sum(row)) : null;
  });
  const mix = row => row && values[0].map((_, axis) => sum(values.map((v, j) => v[axis] * row[j])));
  const exact = exactWeights.map(mix), output = weights.map(mix);
  const referenceNorm = sum(exact.flat().map(x => x * x));
  const squaredError = output.some(row => !row) ? null : sum(output.flat().map((x, i) => (x - exact.flat()[i]) ** 2));
  return { exact, output, weights, exactWeights, qPhi, kPhi, commonKeyScale,
    relativeError: squaredError !== null && referenceNorm > 0 ? Math.sqrt(squaredError / referenceNorm) : null };
}
export function projectedRead(values, coefficients, position) {
  const contributions = values.map((value, i) => value * coefficients[i]);
  return { contributions, full: sum(contributions), prefix: sum(contributions.slice(0, position + 1)) };
}
export function blockOccupancy(mask, size) {
  const count = mask.length / size;
  const tiles = Array.from({ length: count }, (_, r) => Array.from({ length: count }, (_, c) => {
    for (let i = 0; i < size; i += 1) for (let j = 0; j < size; j += 1) if (mask[r * size + i][c * size + j]) return true;
    return false;
  }));
  const occupied = tiles.flat().filter(Boolean).length;
  return { tiles, occupied, edges: mask.flat().filter(Boolean).length, candidates: occupied * size * size };
}

// Positive-term expansion of erf, evaluated to double precision. This implements
// exact-erf GELU numerically; it does not substitute the tanh GELU approximation.
export function erf(value) {
  const x = Math.abs(value);
  if (x > 8) return Math.sign(value);
  let term = x, total = x;
  for (let n = 1; n < 400; n += 1) {
    term *= 2 * x * x / (2 * n + 1);
    total += term;
    if (term <= total * 2e-16) break;
  }
  return Math.sign(value) * Math.min(1, 2 / Math.sqrt(Math.PI) * Math.exp(-x * x) * total);
}
const linear = (input, weight, bias) => weight.map((row, i) => dot(row, input) + (bias?.[i] ?? 0));
function norm(input, weights, name) {
  const mean = sum(input) / input.length;
  const variance = sum(input.map(value => (value - mean) ** 2)) / input.length;
  return input.map((value, i) => (value - mean) / Math.sqrt(variance + 1e-5) * weights[`${name}.weight`][i] + weights[`${name}.bias`][i]);
}
const add = (a, b) => a.map((value, i) => value + b[i]);
const phi = value => value >= 0 ? value + 1 : Math.exp(value);

export function forecastTrajectory(weights, points, mode, { cache = null, start = 0 } = {}) {
  if (!points.length) throw new Error('A forecast needs at least one observed point.');
  const state = points.map((point, index) => linear(point.map(x => 2 * x - 1), weights['stem.weight'], weights['stem.bias']).map((value, axis) => {
    const angle = (start + index) * 10000 ** (-2 * Math.floor(axis / 2) / 24);
    return value + (axis % 2 ? Math.cos(angle) : Math.sin(angle));
  }));
  const qkv = state.map(row => linear(norm(row, weights, 'norm_attention'), weights['qkv.weight']));
  const queries = [], keys = [], values = [], headOutputs = [], weightRows = [], matrices = [], normalizers = [];
  const nextCache = [];
  for (let head = 0; head < 3; head += 1) {
    const q = qkv.map(row => row.slice(head * 8, head * 8 + 8));
    const k = qkv.map(row => row.slice(24 + head * 8, 32 + head * 8));
    const v = qkv.map(row => row.slice(48 + head * 8, 56 + head * 8));
    queries.push(q); keys.push(k); values.push(v);
    const outputs = [], rows = [];
    if (mode === 'kernel') {
      const matrix = cache ? cache[head].matrix.map(row => [...row]) : zeros(8, 8);
      const normalizer = cache ? [...cache[head].normalizer] : Array(8).fill(0);
      const kFeatures = k.map(row => row.map(x => phi(x / 8 ** .25)));
      q.forEach((query, receiver) => {
        const qFeature = query.map(x => phi(x / 8 ** .25));
        kFeatures[receiver].forEach((value, axis) => {
          normalizer[axis] += value;
          v[receiver].forEach((entry, column) => { matrix[axis][column] += value * entry; });
        });
        const denominator = Math.max(1e-9, dot(qFeature, normalizer));
        outputs.push(Array.from({ length: 8 }, (_, column) => sum(matrix.map((row, axis) => row[column] * qFeature[axis])) / denominator));
        rows.push(cache ? null : kFeatures.map((key, donor) => donor <= receiver ? dot(qFeature, key) / denominator : 0));
      });
      matrices.push(matrix); normalizers.push(normalizer); nextCache.push({ matrix, normalizer });
    } else {
      const previous = cache?.[head] ?? { keys: [], values: [], positions: [] };
      const allKeys = [...previous.keys, ...k], allValues = [...previous.values, ...v];
      const positions = [...previous.positions, ...points.map((_, i) => start + i)];
      q.forEach((query, receiver) => {
        const absolute = start + receiver;
        const row = normalizedScores(allKeys.map(key => dot(query, key) / Math.sqrt(8)), positions.map(position => position <= absolute && (mode !== 'window' || absolute - position < 5)));
        rows.push(row); outputs.push(Array.from({ length: 8 }, (_, axis) => sum(allValues.map((value, j) => value[axis] * row[j]))));
      });
      const retained = mode === 'window' ? 5 : allKeys.length;
      nextCache.push({ keys: allKeys.slice(-retained), values: allValues.slice(-retained), positions: positions.slice(-retained) });
    }
    headOutputs.push(outputs); weightRows.push(rows);
  }
  const predictions = state.map((row, receiver) => {
    const merged = headOutputs.flatMap(head => head[receiver]);
    const residual = add(row, linear(merged, weights['output.weight']));
    const hidden = linear(norm(residual, weights, 'norm_feedforward'), weights['feedforward.0.weight'], weights['feedforward.0.bias']).map(value => value * .5 * (1 + erf(value / Math.sqrt(2))));
    const fed = add(residual, linear(hidden, weights['feedforward.2.weight'], weights['feedforward.2.bias']));
    return linear(norm(fed, weights, 'norm_final'), weights['forecast.weight'], weights['forecast.bias']).map(value => (value + 1) / 2);
  });
  return { predictions, forecast: predictions.at(-1), queries, keys, values, weightRows, matrices, normalizers, cache: nextCache,
    payloadBytes: mode === 'kernel' ? 3 * 8 * 9 * 4 : 3 * 8 * 2 * Math.min(start + points.length, mode === 'window' ? 5 : Infinity) * 4 };
}
