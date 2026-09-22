export const normalizationMean = x => x.reduce((sum, value) => sum + value, 0) / x.length;
export function normalizeVector(x, epsilon = 1e-5, rms = false) {
  const center = rms ? 0 : normalizationMean(x);
  const variance = normalizationMean(x.map(value => (value - center) ** 2));
  return { center, variance, denominator: Math.sqrt(variance + epsilon), output: x.map(value => (value - center) / Math.sqrt(variance + epsilon)) };
}
export function tensorGroup(index, method, groups = 2) {
  const example = Math.floor(index / 8), channel = Math.floor(index % 8 / 2);
  return Array.from({ length: 16 }, (_, i) => i).filter(i => {
    const n = Math.floor(i / 8), c = Math.floor(i % 8 / 2);
    if (method === 'batch') return c === channel;
    if (method === 'layer') return n === example;
    if (method === 'instance') return n === example && c === channel;
    return n === example && Math.floor(c / (4 / groups)) === Math.floor(channel / (4 / groups));
  });
}
export function normalizeTensor(values, method, groups = 2, epsilon = 1e-5) {
  if (values.length !== 16 || ![1, 2, 4].includes(groups)) throw new RangeError('Use the declared 2 × 4 × 1 × 2 fixture and valid groups');
  return values.map((value, index) => { const group = tensorGroup(index, method, groups), calculation = normalizeVector(group.map(i => values[i]), epsilon); return (value - calculation.center) / calculation.denominator; });
}
export function batchNormalizationStep(values, buffers, momentum, training, epsilon = 1e-5, gamma = 1, beta = 0) {
  const calculation = normalizeVector(values, epsilon);
  const correctedVariance = calculation.variance * values.length / (values.length - 1);
  const next = training ? { mean: (1 - momentum) * buffers.mean + momentum * calculation.center, variance: (1 - momentum) * buffers.variance + momentum * correctedVariance } : { ...buffers };
  const center = training ? calculation.center : buffers.mean, variance = training ? calculation.variance : buffers.variance;
  return { batchMean: calculation.center, populationVariance: calculation.variance, correctedVariance, next,
    output: values.map(value => gamma * (value - center) / Math.sqrt(variance + epsilon) + beta),
    evaluationAfter: values.map(value => gamma * (value - next.mean) / Math.sqrt(next.variance + epsilon) + beta) };
}
export function normalizationGradient(x, gamma, beta, target, epsilon = 1e-5) {
  const calculation = normalizeVector(x, epsilon), output = calculation.output.map((value, i) => gamma[i] * value + beta[i]);
  const loss = normalizationMean(output.map((value, i) => (value - target[i]) ** 2));
  const upstream = output.map((value, i) => 2 * (value - target[i]) / x.length), u = upstream.map((d, i) => d * gamma[i]);
  const average = normalizationMean(u), coupling = normalizationMean(u.map((value, i) => value * calculation.output[i]));
  return { ...calculation, output, loss, scaleGradient: upstream.map((d, i) => d * calculation.output[i]), shiftGradient: upstream,
    inputGradient: u.map((value, i) => (value - average - calculation.output[i] * coupling) / calculation.denominator) };
}
