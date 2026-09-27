// Small deterministic tensor primitives. Dense matrices use output-by-input storage.
export const dot = (left, right) => left.reduce((sum, value, i) => sum + value * right[i], 0);
export const add = (left, right) => left.map((value, i) => value + right[i]);
export const scale = (values, factor) => values.map(value => value * factor);
export const mean = values => values.reduce((sum, value) => sum + value, 0) / values.length;
export const dense = (values, weights, bias = []) => weights.map((row, i) => dot(values, row) + (bias[i] ?? 0));
export function softmax(values) {
  const maximum = Math.max(...values);
  if (!Number.isFinite(maximum)) throw new Error('An attention row needs at least one allowed key.');
  const exponentials = values.map(value => Math.exp(value - maximum));
  const total = exponentials.reduce((sum, value) => sum + value, 0);
  return exponentials.map(value => value / total);
}
export function layerNorm(values, gain = [], bias = [], epsilon = 1e-5) {
  const center = mean(values);
  const denominator = Math.sqrt(mean(values.map(value => (value - center) ** 2)) + epsilon);
  return values.map((value, i) => (value - center) / denominator * (gain[i] ?? 1) + (bias[i] ?? 0));
}
// erf approximation has < 1.5e-7 absolute error; native output parity is checked.
export function gelu(value) {
  const sign = value < 0 ? -1 : 1;
  const x = Math.abs(value) / Math.SQRT2;
  const t = 1 / (1 + .3275911 * x);
  const erf = sign * (1 - ((((1.061405429 * t - 1.453152027) * t + 1.421413741) * t - .284496736) * t + .254829592) * t * Math.exp(-x * x));
  return .5 * value * (1 + erf);
}
export function maxDifference(left, right) {
  const leftValues = left.flat(Infinity);
  const rightValues = right.flat(Infinity);
  if (leftValues.length !== rightValues.length) throw new Error('Compared tensors must have the same number of entries.');
  return leftValues.reduce((maximum, value, i) => Math.max(maximum, Math.abs(value - rightValues[i])), 0);
}
export function attention(queries, keys, values, allowed = () => true) {
  const width = queries[0].length;
  const weights = queries.map((query, i) => softmax(keys.map((key, j) => allowed(i, j) ? dot(query, key) / Math.sqrt(width) : -Infinity)));
  return {
    weights,
    output: weights.map(row => values[0].map((_, feature) => dot(row, values.map(value => value[feature]))))
  };
}
