// Bounded, deterministic attention calculations. No training or hidden randomness.
export const attentionTokens = ['<pad>', '<bos>', '<eos>', '<past>', '<participle>', '<third_person>', ...'abcdefghijklmnopqrstuvwxyz'];
export const initialKeys = [[1, 0], [0, 1], [-1, 0]];
export const initialValues = [[2, 0], [0, 2], [-1, 1]];

const dot = (left, right) => left.reduce((sum, value, index) => sum + value * right[index], 0);
const project = (matrix, vector, bias) => matrix.map((row, index) => dot(row, vector) + (bias?.[index] ?? 0));
const sumVectors = (vectors, weights) => vectors[0].map((_, coordinate) => vectors.reduce((sum, vector, index) => sum + weights[index] * vector[coordinate], 0));

export function attentionSoftmax(scores, valid = scores.map(() => true)) {
  if (scores.length !== valid.length || !valid.some(Boolean)) throw new Error('At least one memory must remain valid.');
  const maximum = Math.max(...scores.filter((_, index) => valid[index]));
  if (!Number.isFinite(maximum)) throw new Error('Valid scores must be finite.');
  const numerators = scores.map((value, index) => valid[index] ? Math.exp(value - maximum) : 0);
  const denominator = numerators.reduce((sum, value) => sum + value, 0);
  return numerators.map(value => value / denominator);
}

export function memoryRead(query, keys = initialKeys, values = initialValues, valid = keys.map(() => true), rate = .1) {
  if ([...query, ...keys.flat(), ...values.flat(), rate].some(value => !Number.isFinite(value))) throw new Error('Use finite coordinates.');
  const scores = keys.map(key => dot(key, query));
  const attention = attentionSoftmax(scores, valid);
  const context = sumVectors(values, attention);
  const probabilities = attentionSoftmax(context);
  const loss = -Math.log(probabilities[1]);
  const contextGradient = [probabilities[0], probabilities[1] - 1];
  const scoreGradient = values.map((value, index) => attention[index] * dot(contextGradient, value.map((coordinate, axis) => coordinate - context[axis])));
  const queryGradient = sumVectors(keys, scoreGradient);
  const nextQuery = query.map((value, index) => value - rate * queryGradient[index]);
  const nextAttention = attentionSoftmax(keys.map(key => dot(key, nextQuery)), valid);
  const nextContext = sumVectors(values, nextAttention);
  return { scores, attention, context, probabilities, loss, contextGradient, scoreGradient, queryGradient, nextQuery, nextContext, nextLoss: -Math.log(attentionSoftmax(nextContext)[1]) };
}

export function cancellationRead(query, nonlinear = false) {
  const scores = [-.7, .1, 1.3].map(key => nonlinear ? Math.tanh(2 * query + key) : 2 * query + key);
  return { scores, attention: attentionSoftmax(scores) };
}

export function localAttention(center, radius, renormalize = false, scores = [0, .5, 1, -.5, 2]) {
  if (!Number.isFinite(center) || !Number.isFinite(radius) || radius <= 0 || scores.some(value => !Number.isFinite(value))) throw new Error('Use a finite center, positive radius and finite scores.');
  const positions = [1, 2, 3, 4, 5];
  const valid = positions.map(position => Math.abs(position - center) <= radius);
  const base = attentionSoftmax(scores, valid);
  const gaussian = positions.map(position => Math.exp(-((position - center) ** 2) / (2 * (radius / 2) ** 2)));
  const product = base.map((weight, index) => weight * gaussian[index]);
  const mass = product.reduce((sum, value) => sum + value, 0);
  const weights = renormalize ? product.map(value => value / mass) : product;
  return { positions, valid, base, gaussian, weights, sum: weights.reduce((sum, value) => sum + value, 0), context: dot(weights, positions) };
}

export function copyDistribution(source, attention, vocabulary, pGenerate) {
  const names = [...new Set([...Object.keys(vocabulary), ...source])];
  return names.map(word => {
    const copy = attention.reduce((sum, weight, index) => sum + (source[index] === word ? weight : 0), 0);
    const generated = pGenerate * (vocabulary[word] ?? 0);
    const copied = (1 - pGenerate) * copy;
    return { word, copy, generated, copied, probability: generated + copied };
  });
}

// PyTorch's r,z,n order and reset-after-hidden-projection convention.
function recurrentCell(input, state, weights, prefix, encoder = false) {
  const suffix = encoder ? '_l0' : '';
  const incoming = project(weights[`${prefix}.weight_ih${suffix}`], input, weights[`${prefix}.bias_ih${suffix}`]);
  const previous = project(weights[`${prefix}.weight_hh${suffix}`], state, weights[`${prefix}.bias_hh${suffix}`]);
  const width = state.length;
  return state.map((value, index) => {
    const reset = 1 / (1 + Math.exp(-incoming[index] - previous[index]));
    const update = 1 / (1 + Math.exp(-incoming[width + index] - previous[width + index]));
    const candidate = Math.tanh(incoming[2 * width + index] + reset * previous[2 * width + index]);
    return (1 - update) * candidate + update * value;
  });
}

export function encodeAttention(weights, lemma, feature) {
  if (!/^[a-z]{3,8}$/.test(lemma) || !['past', 'participle', 'third_person'].includes(feature)) throw new Error('Use 3–8 lowercase letters and a supported request.');
  const sourceIds = [attentionTokens.indexOf(`<${feature}>`), ...[...lemma].map(character => attentionTokens.indexOf(character)), 2];
  let state = Array(64).fill(0);
  const memory = sourceIds.map(token => {
    state = recurrentCell(weights['embedding.weight'][token], state, weights, 'encoder', true);
    return state;
  });
  return { sourceIds, memory, keys: memory.map(row => project(weights['key_projection.weight'], row)), initialState: state };
}

export function traceAttention(weights, kind, encoded, { prefix = '', padding = 0, admitted = [], cap = 16 } = {}) {
  if (!['additive', 'general'].includes(kind) || !/^[a-z]{0,8}$/.test(prefix) || !Number.isInteger(cap) || cap < 1 || cap > 16 || !Number.isInteger(padding) || padding < 0 || padding > 2) throw new Error('Unsupported attention trace inputs.');
  const zeroMemory = Array(64).fill(0);
  const memory = [...encoded.memory, ...Array.from({ length: padding }, () => zeroMemory)];
  const keys = [...encoded.keys, ...Array.from({ length: padding }, () => project(weights['key_projection.weight'], zeroMemory))];
  const valid = memory.map((_, index) => index < encoded.memory.length || admitted[index - encoded.memory.length] === true);
  let state = encoded.initialState, previous = 1;
  const rows = [], output = [];
  for (let step = 0; step < cap; step++) {
    const embedding = weights['embedding.weight'][previous];
    if (kind === 'general') state = recurrentCell(embedding, state, weights, 'decoder');
    const query = state;
    const projectedQuery = kind === 'additive' ? project(weights['query_projection.weight'], query) : null;
    const scores = keys.map(key => kind === 'general' ? dot(key, query) : dot(weights['score_projection.weight'][0], key.map((value, index) => Math.tanh(value + projectedQuery[index]))));
    const attention = attentionSoftmax(scores, valid);
    const context = sumVectors(memory, attention);
    if (kind === 'additive') state = recurrentCell([...embedding, ...context], state, weights, 'decoder');
    const attentional = project(weights['combine.weight'], [...state, ...context], weights['combine.bias']).map(Math.tanh);
    const logits = project(weights['readout.weight'], attentional, weights['readout.bias']);
    const probabilities = attentionSoftmax(logits, weights.invalid_output.map(value => !value));
    const argmaxToken = probabilities.indexOf(Math.max(...probabilities));
    const emittedToken = step < prefix.length ? attentionTokens.indexOf(prefix[step]) : argmaxToken;
    rows.push({ step, previous, query, state, scores, attention, context, probabilities, argmaxToken, emittedToken });
    output.push(emittedToken);
    if (emittedToken === 2) break;
    previous = emittedToken;
  }
  return { rows, memory, keys, valid, sourceIds: [...encoded.sourceIds, ...Array(padding).fill(0)], prediction: output.filter(token => token !== 2).map(token => attentionTokens[token]).join(''), ended: output.at(-1) === 2 };
}
