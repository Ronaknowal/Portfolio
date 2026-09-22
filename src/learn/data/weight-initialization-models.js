// Tiny deterministic mechanisms. Recorded training runs live in a separate dataset.
export const initializationSchemes = [['zero', 'All-zero hidden weights'], ['small', 'Normal · std 0.01'], ['xavier', 'Xavier · gain 1'], ['kaiming', 'Kaiming · ReLU'], ['orthogonal', 'Orthogonal · gain √2'], ['large', 'Normal · std 0.5']];
export function activationMoments(values, activation = 'relu') {
  const statistics = row => {
    const mean = row.reduce((sum, value) => sum + value, 0) / row.length;
    const secondMoment = row.reduce((sum, value) => sum + value * value, 0) / row.length;
    return { values: row, mean, secondMoment, variance: row.reduce((sum, value) => sum + (value - mean) ** 2, 0) / row.length };
  };
  return { before: statistics(values), after: statistics(values.map(value => activation === 'relu' ? Math.max(0, value) : value)) };
}
export function directionalGeometry(smaller, vector, depth = 1) {
  const larger = Math.sqrt(2 - smaller * smaller);
  const gains = [larger ** depth, smaller ** depth];
  const output = vector.map((value, i) => gains[i] * value);
  const inputNorm = Math.hypot(...vector);
  return { smaller, larger, gains, averageSquaredGain: (smaller ** 2 + larger ** 2) / 2, output, normGain: inputNorm === 0 ? null : Math.hypot(...output) / inputNorm };
}
export function gatedIdentity(base, vector) {
  const diagonal = base.map(value => value > 0 ? Math.SQRT2 : 0);
  return { diagonal, output: vector.map((value, i) => diagonal[i] * value), atKink: base.some(value => value === 0) };
}
export function symmetryState(weights, outgoing, input = 1, target = 1) {
  const features = weights.map(weight => Math.tanh(weight * input));
  const output = features.reduce((sum, feature, i) => sum + feature * outgoing[i], 0);
  const residual = output - target;
  return { features, output, loss: .5 * residual ** 2, hiddenGradient: features.map((feature, i) => residual * outgoing[i] * (1 - feature * feature) * input), headGradient: features.map(feature => residual * feature) };
}
export function widthConfiguration(width, rate, mode = 'mu') {
  const multiplier = width / 32;
  return { multiplier, inputStd: Math.sqrt(2 / 64), hiddenStd: Math.sqrt(2 / width), readoutStd: 1 / Math.sqrt(mode === 'mu' ? 32 : width), inputRate: rate, hiddenRate: mode === 'mu' ? rate / multiplier : rate, readoutRate: rate, readoutMultiplier: mode === 'mu' ? 1 / multiplier : 1 };
}
export function idealSignal(variance, depth, fanIn = 100) {
  const factor = fanIn * variance / 2;
  return { factor, ratio: factor ** depth };
}
