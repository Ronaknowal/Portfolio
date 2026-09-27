// One head keeps matching, legal access and value content separate.
const dot = (a, b) => a.reduce((sum, value, i) => sum + value * b[i], 0);
export const attentionSoftmax = scores => {
  const maximum = Math.max(...scores), terms = scores.map(s => Math.exp(s - maximum));
  const total = terms.reduce((a, b) => a + b, 0);
  return terms.map(v => v / total);
};
export const attentionEntropy = weights => -weights.reduce((sum, p) => sum + (p > 0 ? p * Math.log(p) : 0), 0);
export function attentionRead(query, keys, values, { allowed = keys.map(() => true), temperature = 1, offset = 0, dimension = query.length } = {}) {
  if (!allowed.some(Boolean)) return { error: 'Choose at least one legal donor.', weights: null, output: null };
  const raw = keys.map(key => dot(query, key));
  const scores = raw.map((score, i) => allowed[i] ? (score / Math.sqrt(dimension) + offset) / temperature : -Infinity);
  const weights = attentionSoftmax(scores);
  const contributions = values.map((v, i) => v.map(value => value * weights[i]));
  const output = values[0].map((_, j) => contributions.reduce((sum, row) => sum + row[j], 0));
  return { raw, scores, weights, contributions, output, entropy: attentionEntropy(weights) };
}
export const messageDefault = () => ({ query: [.5, 1.5], keys: [[1, 0], [0, 1], [1, 1]], values: [[2, 0], [0, 2], [1, 1]], temperature: 1, offset: 0 });
const linear = (input, weight, bias) => weight.map((row, i) => dot(input, row) + (bias?.[i] || 0));
export function multiHeadRead(inputs, maps, heads) {
  const width = inputs[0].length, headWidth = width / heads;
  if (!Number.isInteger(headWidth)) throw new Error('The feature width must divide evenly into heads.');
  const projected = Object.fromEntries(['query', 'key', 'value'].map(name => [name, inputs.map(row => linear(row, maps[name]))]));
  const headReads = Array.from({ length: heads }, (_, head) => {
    const begin = head * headWidth, end = begin + headWidth;
    const query = projected.query.map(row => row.slice(begin, end));
    const keys = projected.key.map(row => row.slice(begin, end));
    const values = projected.value.map(row => row.slice(begin, end));
    return query.map(row => attentionRead(row, keys, values));
  });
  const merged = inputs.map((_, i) => headReads.flatMap(rows => rows[i].output));
  return { projected, headReads, merged, output: merged.map(row => linear(row, maps.output)) };
}
export function attentionTrajectory(parameters, points, valid = points.map(() => true), maskKeys = true, maskPool = true) {
  if (!valid.some(Boolean)) return { error: 'Keep at least one valid point for attention and pooling.' };
  const features = points.map(point => linear(point.map(v => 2 * v - 1), parameters['stem.weight'], parameters['stem.bias']).map(Math.tanh));
  const projected = Object.fromEntries(['query', 'key', 'value'].map(name => [name, features.map(row => linear(row, parameters[`mixer.${name}.weight`]))]));
  const heads = [0, 1].map(head => {
    const first = 12 * head, last = first + 12;
    const keys = projected.key.map(row => row.slice(first, last)), values = projected.value.map(row => row.slice(first, last));
    return projected.query.map(row => attentionRead(row.slice(first, last), keys, values, { allowed: maskKeys ? valid : valid.map(() => true) }));
  });
  const contextual = features.map((row, i) => {
    const merged = heads.flatMap(head => head[i].output);
    const mixed = linear(merged, parameters['mixer.output.weight']);
    return row.map((value, j) => value + mixed[j]);
  });
  const count = maskPool ? valid.filter(Boolean).length : valid.length;
  const pooled = contextual[0].map((_, j) => contextual.reduce((sum, row, i) => sum + (!maskPool || valid[i] ? row[j] : 0), 0) / count);
  const logits = linear(pooled, parameters['classifier.weight'], parameters['classifier.bias']);
  const probabilities = attentionSoftmax(logits);
  return { logits, probabilities, predicted: probabilities.indexOf(Math.max(...probabilities)) + 1, heads, contextual };
}
export function mixtures(first, second, values) {
  const normalize = weights => {
    const sum = weights.reduce((a, b) => a + b, 0);
    return sum > 0 ? weights.map(value => value / sum) : null;
  };
  const a = normalize(first), b = normalize(second);
  if (!a || !b) return { error: 'Each weight row needs a positive total.' };
  return { first: a, second: b, outputA: dot(a, values), outputB: dot(b, values), entropyA: attentionEntropy(a), entropyB: attentionEntropy(b),
    contributionsA: values.map((v, i) => v * a[i]), contributionsB: values.map((v, i) => v * b[i]) };
}
export function attentionStorage({ length = 2048, width = 1024, heads = 16, batch = 1, layers = 12, bytes = 2 } = {}) {
  return { matrixBytes: batch * heads * length ** 2 * bytes, cacheBytes: 2 * batch * layers * length * width * bytes,
    projectionMACs: 4 * batch * length * width ** 2, pairMACs: 2 * batch * length ** 2 * width };
}
