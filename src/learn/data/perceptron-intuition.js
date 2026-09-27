// Small, constructed calculations used only by the neuron lesson.
export function perceptronUpdateExample() {
  const input = [1, 2], weights = [-1, 0], bias = 0, label = 1, step = .5;
  const nextWeights = weights.map((weight, i) => weight + step * label * input[i]);
  const nextBias = bias + step * label;
  const score = (w, b) => input.reduce((sum, value, i) => sum + value * w[i], b);
  return { input, weights, bias, label, step, nextWeights, nextBias,
    before: score(weights, bias), after: score(nextWeights, nextBias) };
}

export const batchNeuronExample = {
  inputs: [[2, -1], [0, 3]], weights: [[1.5, -2], [-1, 1]], biases: [-1, .5],
};

export function evaluateBatchNeuronExample() {
  const { inputs, weights, biases } = batchNeuronExample;
  return inputs.map(input => weights.map((weight, j) =>
    input.reduce((sum, value, i) => sum + value * weight[i], biases[j])));
}

export function neuronOutputShares(scores) {
  const maximum = Math.max(...scores);
  const masses = scores.map(score => Math.exp(score - maximum));
  const total = masses.reduce((sum, mass) => sum + mass, 0);
  return { masses, total, shares: masses.map(mass => mass / total) };
}
