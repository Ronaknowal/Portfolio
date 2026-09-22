// Small, explicit recurrent mechanisms. Learned weights are fetched by the lab.
export const sigmoid = value => value >= 0 ? 1 / (1 + Math.exp(-value)) : Math.exp(value) / (1 + Math.exp(value));
const dot = (a, b) => a.reduce((sum, value, i) => sum + value * b[i], 0);
const affine = (matrix, vector, bias) => matrix.map((row, i) => dot(row, vector) + bias[i]);
export function maxDifference(a, b) {
  const left = a.flat(Infinity), right = b.flat(Infinity);
  if (left.length !== right.length) throw new Error('Compared values must have the same number of elements.');
  return left.reduce((maximum, value, index) => Math.max(maximum, Math.abs(value - right[index])), 0);
}

export function scalarRecurrence({ inputs, inputWeight, recurrentWeight, bias, initial, target, rate }) {
  const states = [initial], steps = [];
  inputs.forEach((input, i) => {
    const incoming = inputWeight * input, retained = recurrentWeight * states[i];
    const preactivation = incoming + retained + bias;
    states.push(Math.tanh(preactivation));
    steps.push({ incoming, retained, preactivation, hidden: states.at(-1) });
  });
  let credit = states.at(-1) - target;
  const contributions = Array(inputs.length);
  for (let i = inputs.length - 1; i >= 0; i--) {
    const delta = credit * (1 - states[i + 1] ** 2);
    contributions[i] = { credit, delta, inputWeight: delta * inputs[i], recurrentWeight: delta * states[i], bias: delta };
    credit = recurrentWeight * delta;
  }
  const gradient = Object.fromEntries(['inputWeight', 'recurrentWeight', 'bias'].map(name => [name, contributions.reduce((sum, row) => sum + row[name], 0)]));
  const updated = { inputWeight: inputWeight - rate * gradient.inputWeight, recurrentWeight: recurrentWeight - rate * gradient.recurrentWeight, bias: bias - rate * gradient.bias };
  let next = initial;
  for (const input of inputs) next = Math.tanh(updated.inputWeight * input + updated.recurrentWeight * next + updated.bias);
  return { states, steps, contributions, gradient, updated, loss: .5 * (states.at(-1) - target) ** 2, updatedFinal: next, updatedLoss: .5 * (next - target) ** 2 };
}

export function lstmAccounting({ cell, forget, input, candidate, output }) {
  const retained = forget * cell, written = input * candidate, nextCell = retained + written;
  return { retained, written, cell: nextCell, exposed: Math.tanh(nextCell), hidden: output * Math.tanh(nextCell) };
}

export function retentionPath(retainedFraction, horizon, writeStep, write) {
  const forget = retainedFraction ** (1 / horizon);
  const plain = [[0, 1]], changed = [[0, 1]];
  for (let step = 1; step <= horizon; step++) {
    plain.push([step, forget ** step]);
    changed.push([step, forget * changed.at(-1)[1] + (step === writeStep ? write : 0)]);
  }
  return { forget, halfLife: Math.log(.5) / Math.log(forget), bias: Math.log(forget / (1 - forget)), plain, changed };
}

export function resetPlacement(hidden, reset, matrix, bias) {
  return { before: affine(matrix, hidden.map((value, i) => value * reset[i]), bias), after: affine(matrix, hidden, bias).map((value, i) => value * reset[i]) };
}

// The affine rows use PyTorch's i/f/g/o or r/z/n order and two biases.
export function recurrentSequence(kind, inputs, weights, initial = {}) {
  const hiddenSize = weights['recurrent.weight_hh_l0'][0].length;
  let hidden = initial.hidden ? [...initial.hidden] : Array(hiddenSize).fill(0);
  let cell = initial.cell ? [...initial.cell] : Array(hiddenSize).fill(0);
  return inputs.map(point => {
    const incoming = affine(weights['recurrent.weight_ih_l0'], point, weights['recurrent.bias_ih_l0']);
    const recurrent = affine(weights['recurrent.weight_hh_l0'], hidden, weights['recurrent.bias_hh_l0']);
    const previousHidden = hidden, previousCell = cell;
    let gates = {};
    if (kind === 'rnn') hidden = incoming.map((value, i) => Math.tanh(value + recurrent[i]));
    else if (kind === 'lstm') {
      const slice = (group, activation) => Array.from({ length: hiddenSize }, (_, j) => activation(incoming[group * hiddenSize + j] + recurrent[group * hiddenSize + j]));
      const input = slice(0, sigmoid), forget = slice(1, sigmoid), candidate = slice(2, Math.tanh), output = slice(3, sigmoid);
      const retained = cell.map((value, i) => value * forget[i]), written = input.map((value, i) => value * candidate[i]);
      cell = retained.map((value, i) => value + written[i]);
      hidden = cell.map((value, i) => output[i] * Math.tanh(value));
      gates = { input, forget, candidate, output, retained, written, cell };
    } else if (kind === 'gru') {
      const reset = Array.from({ length: hiddenSize }, (_, i) => sigmoid(incoming[i] + recurrent[i]));
      const retain = reset.map((_, i) => sigmoid(incoming[hiddenSize + i] + recurrent[hiddenSize + i]));
      const candidate = reset.map((value, i) => Math.tanh(incoming[2 * hiddenSize + i] + value * recurrent[2 * hiddenSize + i]));
      hidden = hidden.map((value, i) => retain[i] * value + (1 - retain[i]) * candidate[i]);
      gates = { reset, retain_update: retain, candidate };
    } else throw new Error('Unknown recurrent cell');
    let probabilities = [];
    if (weights['readout.weight']) {
      const scores = affine(weights['readout.weight'], hidden, weights['readout.bias']);
      const maximum = Math.max(...scores), exponentials = scores.map(value => Math.exp(value - maximum));
      const total = exponentials.reduce((sum, value) => sum + value, 0);
      probabilities = exponentials.map(value => value / total);
    }
    return { point, previousHidden, previousCell, hidden, cell, gates, probabilities };
  });
}

export function nativeWeights(weights, reverse = false) {
  const suffix = reverse ? '_reverse' : '';
  return Object.fromEntries(['weight_ih_l0', 'weight_hh_l0', 'bias_ih_l0', 'bias_hh_l0'].map(name => ['recurrent.' + name, weights[name + suffix]]));
}

export function boundaryExperiment(sequence, weights, boundary, mode) {
  const model = nativeWeights(weights);
  const whole = recurrentSequence('gru', sequence, model);
  const carried = whole[boundary - 1].hidden;
  const run = points => {
    const prefix = recurrentSequence('gru', points.slice(0, boundary), model);
    const initial = mode === 'reset' ? Array(carried.length).fill(0) : mode === 'detach' ? carried : prefix.at(-1).hidden;
    return [...prefix, ...recurrentSequence('gru', points.slice(boundary), model, { hidden: initial })];
  };
  const current = run(sequence);
  const loss = states => states.slice(boundary).reduce((sum, state) => sum + dot(state.hidden, state.hidden), 0);
  const epsilon = 1e-5;
  const gradients = sequence.map((point, i) => point.map((_, j) => {
    if (mode !== 'carry' && i < boundary) return 0;
    const plus = sequence.map(row => [...row]), minus = sequence.map(row => [...row]);
    plus[i][j] += epsilon; minus[i][j] -= epsilon;
    return (loss(run(plus)) - loss(run(minus))) / (2 * epsilon);
  }));
  return { whole, current, gradients, finalError: maxDifference(whole.at(-1).hidden, current.at(-1).hidden), loss: loss(current) };
}

export function streamOwnership(sequence, weights, boundary, swapOwners) {
  const model = nativeWeights(weights), streams = [sequence, [...sequence].reverse()];
  const prefixes = streams.map(points => recurrentSequence('gru', points.slice(0, boundary), model).at(-1).hidden);
  const correct = streams.map((points, i) => recurrentSequence('gru', points.slice(boundary), model, { hidden: prefixes[i] }).at(-1).hidden);
  const current = streams.map((points, i) => recurrentSequence('gru', points.slice(boundary), model, { hidden: prefixes[swapOwners ? 1 - i : i] }).at(-1).hidden);
  return { prefixes, correct, current, difference: maxDifference(correct, current) };
}

export function paddingExperiment(sequence, weights, length, paddingValue) {
  const valid = sequence.slice(0, length);
  const padded = [...valid, ...Array.from({ length: sequence.length - length }, () => [paddingValue, paddingValue])];
  const base = [...valid, ...Array.from({ length: sequence.length - length }, () => [0, 0])];
  const forwardWeights = nativeWeights(weights), backwardWeights = nativeWeights(weights, true);
  const evaluate = (points, reverse) => {
    const states = recurrentSequence('gru', reverse ? [...points].reverse() : points, reverse ? backwardWeights : forwardWeights);
    return reverse ? states.reverse() : states;
  };
  const forward = evaluate(padded, false), backward = evaluate(padded, true);
  const baseForward = evaluate(base, false), baseBackward = evaluate(base, true);
  const packedForward = evaluate(valid, false), packedBackward = evaluate(valid, true);
  return { padded, forward, backward, packedForward, packedBackward,
    forwardEditError: maxDifference(forward.slice(0, length).map(row => row.hidden), baseForward.slice(0, length).map(row => row.hidden)),
    backwardEditError: maxDifference(backward.slice(0, length).map(row => row.hidden), baseBackward.slice(0, length).map(row => row.hidden)),
    paddedFinalError: maxDifference(forward.at(-1).hidden, packedForward.at(-1).hidden),
    backwardPaddingError: maxDifference(backward.slice(0, length).map(row => row.hidden), packedBackward.map(row => row.hidden)) };
}

export function pointStatistics(points) {
  return [0, 1].map(axis => {
    const values = points.map(point => point[axis]), mean = values.reduce((a, b) => a + b, 0) / values.length;
    return { mean, standardDeviation: Math.sqrt(values.reduce((sum, value) => sum + (value - mean) ** 2, 0) / values.length), min: Math.min(...values), max: Math.max(...values) };
  });
}
