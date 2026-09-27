// Topic-owned grouped attention and frozen causal inference. Arrays use row-major
// Python weight orientation: output coordinate, then input coordinate.
export const dot = (a, b) => a.reduce((sum, value, i) => sum + value * b[i], 0);
export const add = (a, b) => a.map((value, i) => value + b[i]);
export function softmax(scores, legal = scores.map(() => true)) {
  if (!legal.some(Boolean)) throw new RangeError('No legal key: move the query or a key position.');
  const maximum = Math.max(...scores.filter((_, i) => legal[i]));
  const values = scores.map((value, i) => legal[i] ? Math.exp(value - maximum) : 0);
  const total = values.reduce((a, b) => a + b, 0);
  return values.map(value => value / total);
}
export const linear = (x, weights, name) => weights[name + '.weight'].map((row, i) => dot(row, x) + (weights[name + '.bias']?.[i] ?? 0));
export function layerNorm(x, weights, name) {
  const mean = x.reduce((a, b) => a + b, 0) / x.length;
  const variance = x.reduce((sum, value) => sum + (value - mean) ** 2, 0) / x.length;
  return x.map((value, i) => (value - mean) / Math.sqrt(variance + 1e-5) * weights[name + '.weight'][i] + weights[name + '.bias'][i]);
}
export function gelu(value) {
  // Abramowitz-Stegun erf approximation; numerical parity is checked against
  // the saved float32 PyTorch network, including changed inputs.
  const x = Math.abs(value / Math.sqrt(2)),
    t = 1 / (1 + .3275911 * x);
  const erf = Math.sign(value) * (1 - ((((1.061405429 * t - 1.453152027) * t + 1.421413741) * t - .284496736) * t + .254829592) * t * Math.exp(-x * x));
  return .5 * value * (1 + erf);
}
export function rotate(x, position) {
  return x.flatMap((value, i) => {
    if (i % 2) return [];
    const angle = position * 10000 ** (-i / x.length),
      c = Math.cos(angle),
      s = Math.sin(angle);
    return [value * c - x[i + 1] * s, value * s + x[i + 1] * c];
  });
}
export function groupedDefaults() {
  return {
    query: [[1, 0], [0, 1], [-1, 0], [0, -1]].map(q => q.map(x => x * Math.sqrt(2))),
    keys: [[[1, 0], [0, 1], [1, 1]], [[1, 1], [-1, 0], [0, -1]]],
    values: [[[2, 0], [0, 4], [2, 2]], [[1, 3], [-1, 2], [3, 0]]],
    mapping: [0, 0, 1, 1],
    positions: [0, 1, 2],
    queryPosition: 2
  };
}
export function groupedRead({
  query,
  keys,
  values,
  mapping,
  positions,
  queryPosition
}, upperLeft = false) {
  const legal = positions.map((position, i) => upperLeft ? i === 0 : position <= queryPosition);
  return query.map((q, h) => {
    const group = mapping[h],
      scores = keys[group].map(key => dot(q, key) / Math.sqrt(q.length));
    const weights = softmax(scores, legal);
    const contributions = values[group].map((value, i) => value.map(v => v * weights[i]));
    const output = values[group][0].map((_, d) => contributions.reduce((sum, row) => sum + row[d], 0));
    return {
      group,
      scores,
      weights,
      output,
      legal,
      contributions
    };
  });
}
export function cacheBudget({
  batch,
  layers,
  length,
  queryHeads,
  kvHeads,
  keyWidth,
  valueWidth,
  bytes,
  modelWidth = 512
}) {
  if (queryHeads % kvHeads !== 0) throw new RangeError('Query heads must be divisible by KV heads.');
  const perToken = batch * layers * kvHeads * (keyWidth + valueWidth) * bytes;
  return {
    perToken,
    total: perToken * length,
    mha: perToken * length * queryHeads / kvHeads,
    parameters: modelWidth * (queryHeads + kvHeads) * (keyWidth + valueWidth),
    denseOps: 2 * batch * queryHeads * length * (keyWidth + valueWidth)
  };
}
export function conversionRead(query, keys, values, candidate = null) {
  const averagedKeys = keys[0].map((value, i) => (value + keys[1][i]) / 2);
  const averagedValues = values[0].map((value, i) => (value + values[1][i]) / 2);
  const sharedKeys = candidate?.keys ?? averagedKeys;
  const sharedValues = candidate?.values ?? averagedValues;
  const squaredDistance = (original, shared) => original.reduce((total, row) => total + row.reduce((sum, value, i) => sum + (value - shared[i]) ** 2, 0), 0);
  const keyDistance = squaredDistance(keys, sharedKeys), valueDistance = squaredDistance(values, sharedValues);
  const minimumKeyDistance = squaredDistance(keys, averagedKeys), minimumValueDistance = squaredDistance(values, averagedValues);
  return query.map((q, h) => {
    const originalWeights = softmax(keys[h].map(key => q * key));
    const convertedWeights = softmax(sharedKeys.map(key => q * key));
    const original = dot(originalWeights, values[h]), converted = dot(convertedWeights, sharedValues);
    return {
      originalWeights,
      convertedWeights,
      original, converted, outputDelta: converted - original,
      keyDistance, valueDistance, minimumKeyDistance, minimumValueDistance,
      sharedKeys, sharedValues,
      averagedKeys,
      averagedValues
    };
  });
}
export function groupedForecast(weights, points, kvHeads, shift = 0, oldCache = null) {
  const start = oldCache?.positions.length ?? 0;
  const states = points.map(point => linear(point.map(x => 2 * x - 1), weights, 'stem'));
  const normalized = states.map(state => layerNorm(state, weights, 'norm_attention'));
  const split = (rows, headCount) => Array.from({
    length: headCount
  }, (_, h) => rows.map(row => row.slice(h * 6, (h + 1) * 6)));
  const queries = split(normalized.map(x => linear(x, weights, 'query')), 4).map(head => head.map((q, t) => rotate(q, t + start + shift)));
  const newKeys = split(normalized.map(x => linear(x, weights, 'key')), kvHeads).map(head => head.map((k, t) => rotate(k, t + start + shift)));
  const newValues = split(normalized.map(x => linear(x, weights, 'value')), kvHeads);
  const positions = [...(oldCache?.positions ?? []), ...points.map((_, t) => t + start + shift)];
  const keys = newKeys.map((head, h) => [...(oldCache?.keys[h] ?? []), ...head]);
  const values = newValues.map((head, h) => [...(oldCache?.values[h] ?? []), ...head]);
  const attention = [],
    heads = [];
  const predictions = states.map((state, t) => {
    const headRows = [],
      weightRows = [];
    for (let h = 0; h < 4; h++) {
      const group = Math.floor(h / (4 / kvHeads));
      const scores = keys[group].map(key => dot(queries[h][t], key) / Math.sqrt(6));
      const probabilities = softmax(scores, positions.map(position => position <= start + t + shift));
      const output = Array.from({
        length: 6
      }, (_, d) => values[group].reduce((sum, value, j) => sum + probabilities[j] * value[d], 0));
      weightRows.push(probabilities);
      headRows.push(output);
    }
    attention.push(weightRows);
    heads.push(headRows);
    const residual = add(state, linear(headRows.flat(), weights, 'output'));
    const hidden = linear(layerNorm(residual, weights, 'norm_feedforward'), weights, 'feedforward.0').map(gelu);
    const final = add(residual, linear(hidden, weights, 'feedforward.2'));
    return linear(layerNorm(final, weights, 'norm_final'), weights, 'forecast').map(x => (x + 1) / 2);
  });
  return {
    predictions,
    attention,
    heads,
    cache: {
      keys,
      values,
      positions
    }
  };
}
export function groupedRollout(weights, points, kvHeads, count, shift = 0) {
  const full = groupedForecast(weights, points, kvHeads, shift);
  let cache = full.cache,
    point = full.predictions.at(-1);
  const result = [point];
  for (let i = 1; i < count; i++) {
    const step = groupedForecast(weights, [point], kvHeads, shift, cache);
    cache = step.cache;
    point = step.predictions[0];
    result.push(point);
  }
  return {
    ...full,
    rollout: result
  };
}
