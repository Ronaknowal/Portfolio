export const formatDropout = (value, digits = 5) => Number.isFinite(value)
  ? Number(value.toFixed(digits)).toString() : 'not defined';

export function maskedUpdate({ features, weights, mask, probability, target, learningRate, training = true }) {
  const factors = mask.map(bit => !training || probability === 0 ? 1 : probability === 1 ? 0 : bit / (1 - probability));
  const masked = features.map((value, index) => value * factors[index]);
  const contributions = masked.map((value, index) => value * weights[index]);
  const output = contributions.reduce((sum, value) => sum + value, 0);
  const error = output - target;
  const weightGradient = masked.map(value => error * value);
  const inputGradient = weights.map((value, index) => error * value * factors[index]);
  const nextWeights = weights.map((value, index) => value - learningRate * weightGradient[index]);
  const nextOutput = nextWeights.reduce((sum, value, index) => sum + value * masked[index], 0);
  return { factors, masked, contributions, output, error, loss: error ** 2 / 2,
    weightGradient, inputGradient, nextWeights, nextOutput, nextLoss: (nextOutput - target) ** 2 / 2 };
}

export function maskOutcomes(probability, multiplier, features = [1, 2]) {
  const keep = 1 - probability;
  const outcomes = [[0, 0], [0, 1], [1, 0], [1, 1]].map(mask => ({
    mask,
    probability: mask.reduce((product, bit) => product * (bit ? keep : probability), 1),
    values: features.map((value, index) => value * mask[index] * multiplier),
    relu: Math.max(0, multiplier * (mask[0] - mask[1])),
  }));
  const mean = features.map((_, index) => outcomes.reduce((sum, row) => sum + row.probability * row.values[index], 0));
  const variance = mean.map((center, index) => outcomes.reduce((sum, row) => sum + row.probability * (row.values[index] - center) ** 2, 0));
  return { outcomes, mean, variance, expectedRelu: outcomes.reduce((sum, row) => sum + row.probability * row.relu, 0) };
}

export const maskShapes = { element: [2, 2, 2, 2], channel: [2, 2, 1, 1], row: [2, 1, 1, 1], batch: [1, 1, 1, 1] };
export function geometryMaskIndex(mode, index) {
  return mode === 'element' ? index : mode === 'channel' ? Math.floor(index / 4) : mode === 'row' ? Math.floor(index / 8) : 0;
}
export function geometryOutput({ mode, bits, probability, training, offset = 0 }) {
  return Array.from({ length: 16 }, (_, index) => {
    const input = index + 1 + offset;
    const maskIndex = geometryMaskIndex(mode, index);
    const factor = !training || probability === 0 ? 1 : probability === 1 ? 0 : bits[maskIndex] / (1 - probability);
    return { input, maskIndex, bit: bits[maskIndex], factor, output: input * factor };
  });
}

export function residualMask({ input, correction, probability, bit, placement = 'branch', training = true }) {
  const factor = !training || probability === 0 ? 1 : probability === 1 ? 0 : bit / (1 - probability);
  const output = input.map((value, index) => placement === 'branch'
    ? value + factor * correction[index] : factor * (value + correction[index]));
  return { factor, output, preservesInput: output.every((value, index) => value === input[index]) };
}

export function depthRates(length, endpoint, convention = 'zero-first') {
  return Array.from({ length }, (_, index) => convention === 'original'
    ? endpoint * (index + 1) / length : length === 1 ? 0 : endpoint * index / (length - 1));
}
export const expectedActive = rates => rates.reduce((sum, rate) => sum + 1 - rate, 0);

export function executeDepth({ rates, bits, strategy = 'eager', input = 1 }) {
  let calls = 0;
  const countedBranch = value => { calls += 1; return 2 * value; };
  const trace = rates.map((rate, index) => {
    const active = rate === 0 ? 1 : rate === 1 ? 0 : bits[index];
    const factor = rate === 0 ? 1 : rate === 1 ? 0 : active / (1 - rate);
    const before = calls;
    const raw = strategy === 'eager' || active ? countedBranch(input) : null;
    return { active, factor, called: calls > before, raw, correction: raw === null ? 0 : raw * factor };
  });
  return { calls, trace, totalCorrection: trace.reduce((sum, row) => sum + row.correction, 0) };
}

export function normalizationStep({ values, runningMean, runningVariance, batches, batchTraining, recordGradients, momentum = 1 }) {
  const mean = values.reduce((sum, value) => sum + value, 0) / values.length;
  const variance = values.reduce((sum, value) => sum + (value - mean) ** 2, 0) / values.length;
  return {
    output: values.map(value => (value - (batchTraining ? mean : runningMean)) / Math.sqrt((batchTraining ? variance : runningVariance) + 1e-5)),
    runningMean: batchTraining ? (1 - momentum) * runningMean + momentum * mean : runningMean,
    runningVariance: batchTraining ? (1 - momentum) * runningVariance + momentum * variance * values.length / (values.length - 1) : runningVariance,
    batches: batches + Number(batchTraining),
    recordGradients,
  };
}

export function entropy(probabilities) {
  return -probabilities.reduce((sum, value) => sum + (value > 0 ? value * Math.log(value) : 0), 0);
}
export function monteCarloSummary(samples) {
  if (!samples.length) throw new Error('At least one saved draw is required');
  const count = samples.length;
  const mean = samples[0].map((_, index) => samples.reduce((sum, row) => sum + row[index], 0) / count);
  const std = mean.map((value, index) => count < 2 ? null
    : Math.sqrt(samples.reduce((sum, row) => sum + (row[index] - value) ** 2, 0) / (count - 1)));
  const predictiveEntropy = entropy(mean);
  const meanEntropy = samples.reduce((sum, row) => sum + entropy(row), 0) / count;
  return { mean, std, predictiveEntropy, meanEntropy, disagreement: Math.max(0, predictiveEntropy - meanEntropy),
    choice: mean.indexOf(Math.max(...mean)) };
}
