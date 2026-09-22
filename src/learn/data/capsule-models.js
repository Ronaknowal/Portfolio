// Topic-owned bounded arithmetic. All browser inference uses the recorded frozen model.
export const capsuleVotes = [[[2, 0], [0, 1]], [[2, 0], [0, -1]], [[0, 1], [0, 2]]];
export const emVotes = [[[0, 0], [0, 0]], [[.2, 0], [3, 0]], [[2, 0], [3.2, 0]]];
export const norm = vector => Math.hypot(...vector);
export const dot = (left, right) => left.reduce((sum, value, i) => sum + value * right[i], 0);
export const squash = vector => { const radius = norm(vector); return vector.map(value => value * radius / (1 + radius * radius)); };
export function softmax(values, temperature = 1) {
  if (!(temperature > 0) || !Number.isFinite(temperature)) throw new Error('Temperature must be positive and finite.');
  const maximum = Math.max(...values), weights = values.map(value => Math.exp((value - maximum) / temperature));
  const total = weights.reduce((sum, value) => sum + value, 0);
  return weights.map(value => value / total);
}
export function routeCapsules(votes, iterations = 3, temperature = 1) {
  if (!Number.isInteger(iterations) || iterations < 1 || iterations > 8) throw new Error('Use 1–8 routing steps.');
  const parents = votes[0].length, dimensions = votes[0][0].length;
  let logits = votes.map(() => Array(parents).fill(0));
  const history = [];
  for (let step = 0; step < iterations; step++) {
    const coupling = logits.map(row => softmax(row, temperature));
    const sums = Array.from({ length: parents }, (_, parent) => Array.from({ length: dimensions }, (_, coordinate) => votes.reduce((sum, row, child) => sum + coupling[child][parent] * row[parent][coordinate], 0)));
    const outputs = sums.map(squash), lengths = outputs.map(norm);
    const agreement = votes.map(row => row.map((vote, parent) => dot(vote, outputs[parent])));
    history.push({ step: step + 1, logits, coupling, sums, outputs, lengths, agreement, entropy: -coupling.reduce((sum, row) => sum + row.reduce((inner, value) => inner + (value ? value * Math.log(value) : 0), 0), 0) / votes.length });
    logits = logits.map((row, child) => row.map((value, parent) => value + agreement[child][parent]));
  }
  return history;
}
export function capsuleIndex(row, column, type, coordinate) {
  return { channel: type * 4 + coordinate, child: (row * 4 + column) * 4 + type, coordinate, row, column };
}
export function squashProbe(vector, magnitude, direction) {
  const radius = norm(vector), unit = radius ? vector.map(value => value / radius) : [1, 0];
  const axis = direction === 'radial' ? unit : [-unit[1], unit[0]];
  const changed = vector.map((value, i) => value + magnitude * axis[i]), output = squash(vector), changedOutput = squash(changed);
  return { radius, output, changed, changedOutput, change: norm(output.map((value, i) => changedOutput[i] - value)), radial: 2 * radius / (1 + radius * radius) ** 2, tangent: radius / (1 + radius * radius) };
}
export function diagonalCapsuleEM(votes, activation, iterations = 3) {
  if (!activation.some(value => value > 0)) return [];
  let responsibility = votes.map(row => row.map(() => 1 / row.length));
  const history = [], parents = votes[0].length, dimensions = votes[0][0].length;
  for (let step = 0; step < iterations; step++) {
    const effective = responsibility.map((row, i) => row.map(value => value * activation[i]));
    const mass = Array.from({ length: parents }, (_, j) => effective.reduce((sum, row) => sum + row[j], 0));
    const means = mass.map((value, j) => Array.from({ length: dimensions }, (_, h) => effective.reduce((sum, row, i) => sum + row[j] * votes[i][j][h], 0) / Math.max(value, 1e-12)));
    const variance = mass.map((value, j) => Array.from({ length: dimensions }, (_, h) => Math.max(.01, effective.reduce((sum, row, i) => sum + row[j] * (votes[i][j][h] - means[j][h]) ** 2, 0) / Math.max(value, 1e-12))));
    const inverseTemperature = .5 + .25 * step;
    const parentActivation = mass.map((value, j) => 1 / (1 + Math.exp(inverseTemperature * value * variance[j].reduce((sum, item) => sum + .5 * Math.log(item), 0))));
    const next = votes.map(row => softmax(row.map((vote, j) => Math.log(Math.max(parentActivation[j], 1e-300)) - .5 * vote.reduce((sum, item, h) => sum + Math.log(2 * Math.PI * variance[j][h]) + (item - means[j][h]) ** 2 / variance[j][h], 0))));
    history.push({ mass, means, variance, effective, parentActivation, responsibility, next });
    responsibility = next;
  }
  return history;
}
export function shiftCapsuleImage(image, dy, dx) {
  return image.map((_, index) => { const row = Math.floor(index / 8) - dy, column = index % 8 - dx; return row >= 0 && row < 8 && column >= 0 && column < 8 ? image[row * 8 + column] : 0; });
}
export function loadCapsuleWeights(buffer, metadata) {
  if (metadata.model !== 'seed-1-routing-3' || buffer.byteLength !== 47184 * 4) throw new Error('The frozen model has an unexpected identity or size.');
  const values = new Float32Array(buffer);
  if (!values.every(Number.isFinite)) throw new Error('The frozen model contains invalid weights.');
  const expected = { 'transforms': 20480, 'features.weight': 288, 'features.bias': 32, 'primary.weight': 4608, 'primary.bias': 16, 'decoder.0.weight': 5120, 'decoder.0.bias': 64, 'decoder.2.weight': 8192, 'decoder.2.bias': 128, 'decoder.4.weight': 8192, 'decoder.4.bias': 64 };
  const state = {}; let consumed = 0;
  for (const [name, count] of Object.entries(expected)) {
    const entry = metadata.layout[name];
    if (!entry || entry.length !== count || entry.offset !== consumed) throw new Error('Frozen model tensor layout mismatch.');
    state[name] = values.subarray(entry.offset, entry.offset + count); consumed += count;
  }
  return state;
}
function convolution(image, inChannels, size, weights, bias, stride = 1) {
  const outSize = Math.floor((size - 1) / stride) + 1, result = new Float64Array(bias.length * outSize * outSize);
  for (let channel = 0; channel < bias.length; channel++) for (let row = 0; row < outSize; row++) for (let column = 0; column < outSize; column++) {
    let value = bias[channel];
    for (let input = 0; input < inChannels; input++) for (let ky = 0; ky < 3; ky++) for (let kx = 0; kx < 3; kx++) {
      const y = row * stride + ky - 1, x = column * stride + kx - 1;
      if (y >= 0 && y < size && x >= 0 && x < size) value += image[(input * size + y) * size + x] * weights[((channel * inChannels + input) * 3 + ky) * 3 + kx];
    }
    result[(channel * outSize + row) * outSize + column] = value;
  }
  return result;
}
export function encodeCapsuleImage(image, state) {
  const features = convolution(image, 1, 8, state['features.weight'], state['features.bias']).map(value => Math.max(0, value));
  const channels = convolution(features, 32, 8, state['primary.weight'], state['primary.bias'], 2);
  const primary = Array.from({ length: 64 }, (_, child) => { const type = child % 4, cell = Math.floor(child / 4); return squash(Array.from({ length: 4 }, (_, coordinate) => channels[(type * 4 + coordinate) * 16 + cell])); });
  const votes = primary.map((vector, child) => Array.from({ length: 10 }, (_, parent) => Array.from({ length: 8 }, (_, output) => vector.reduce((sum, value, input) => sum + value * state.transforms[((child * 10 + parent) * 8 + output) * 4 + input], 0))));
  const last = routeCapsules(votes, 3).at(-1), predicted = last.lengths.indexOf(Math.max(...last.lengths));
  return { capsules: last.outputs, lengths: last.lengths, predicted, reconstruction: decodeCapsules(last.outputs, predicted, state) };
}
export function decodeCapsules(capsules, selected, state) {
  let values = capsules.flatMap((vector, label) => vector.map(value => label === selected ? value : 0));
  for (const layer of [0, 2, 4]) {
    const weights = state[`decoder.${layer}.weight`], bias = state[`decoder.${layer}.bias`], inputLength = values.length;
    values = Array.from(bias, (offset, row) => { const value = values.reduce((sum, item, column) => sum + item * weights[row * inputLength + column], offset); return layer === 4 ? 1 / (1 + Math.exp(-value)) : Math.max(0, value); });
  }
  return values;
}
