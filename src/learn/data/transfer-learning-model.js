/** Exact small teaching mechanisms; no browser training of the digit experiment. */
export const TRANSFER_METHODS = ['scratch', 'probe', 'full', 'discriminative', 'lora2', 'adapter4'];
export const TRANSFER_LABELS = { scratch: 'Scratch', probe: 'Linear probe', full: 'Full fine-tuning', discriminative: 'Discriminative rates', lora2: 'LoRA rank 2', adapter4: 'Adapter width 4' };
export const finiteIn = (value, min, max, integer = false) => Number.isFinite(value) && value >= min && value <= max && (!integer || Number.isInteger(value));
const dot = (a, b) => a.reduce((sum, value, i) => sum + value * b[i], 0);
const multiply = (matrix, vector) => matrix.map(row => dot(row, vector));

export function loraFixture(rank = 1) {
  return { rank, A: rank === 1 ? [[1, -1]] : [[1, -1], [1, 1]], B: Array.from({ length: 2 }, () => Array(rank).fill(0)), x: [2, 1], target: [0, 0], alpha: rank, rate: 0.1 };
}

export function loraForward({ A, B, x, target, alpha, rank }) {
  const scale = alpha / rank;
  const bottleneck = multiply(A, x);
  const correction = multiply(B, bottleneck).map(value => scale * value);
  const output = x.map((value, i) => value + correction[i]); // W = I₂
  const delta = output.map((value, i) => value - target[i]); // mean of two squares
  const loss = dot(delta, delta) / 2;
  const gradB = delta.map(value => bottleneck.map(z => scale * value * z));
  const back = Array.from({ length: rank }, (_, j) => B.reduce((sum, row, i) => sum + row[j] * delta[i], 0));
  const gradA = back.map(value => x.map(input => scale * value * input));
  const inputGradient = delta.map((value, i) => value + scale * A.reduce((sum, row, j) => sum + row[i] * back[j], 0));
  const merged = B.map((row, i) => x.map((_, j) => (i === j ? 1 : 0) + scale * row.reduce((sum, value, k) => sum + value * A[k][j], 0)));
  const mergedOutput = multiply(merged, x);
  return { scale, bottleneck, correction, output, loss, gradA, gradB, inputGradient, merged, mergedOutput, mergeError: Math.max(...output.map((value, i) => Math.abs(value - mergedOutput[i]))) };
}

export function loraStep(state) {
  const before = loraForward(state);
  const next = { ...state, A: state.A.map((row, i) => row.map((value, j) => value - state.rate * before.gradA[i][j])), B: state.B.map((row, i) => row.map((value, j) => value - state.rate * before.gradB[i][j])) };
  const withinEditorBounds = [...next.A.flat(), ...next.B.flat()].every(value => finiteIn(value, -5, 5));
  return { before, next, after: loraForward(next), withinEditorBounds };
}

export function batchNormFixture() {
  return { input: [1, 3], momentum: 0.1, training: true, recording: false, trainable: false, optimizer: false, mean: 0, variance: 1, gamma: 1, beta: 0 };
}

export function batchNormForward(state) {
  const { input, momentum, training, recording, trainable, optimizer, mean, variance, gamma, beta } = state;
  const batchMean = (input[0] + input[1]) / 2;
  const batchVariance = input.reduce((sum, value) => sum + (value - batchMean) ** 2, 0) / 2;
  const usedMean = training ? batchMean : mean;
  const usedVariance = training ? batchVariance : variance;
  const inverseStd = 1 / Math.sqrt(usedVariance + 1e-5);
  const normalized = input.map(value => (value - usedMean) * inverseStd);
  const output = normalized.map(value => gamma * value + beta);
  const delta = output; // MSE against zero, averaged across the two outputs
  const gradGamma = dot(delta, normalized);
  const gradBeta = delta[0] + delta[1];
  const meanDelta = gradBeta / 2;
  const meanDeltaNormalized = gradGamma / 2;
  const inputGradient = recording ? delta.map((value, i) => training ? gamma * inverseStd * (value - meanDelta - normalized[i] * meanDeltaNormalized) : gamma * inverseStd * value) : null;
  return {
    batchMean, batchVariance, unbiasedVariance: 2 * batchVariance, usedMean, usedVariance, normalized, output,
    nextMean: training ? (1 - momentum) * mean + momentum * batchMean : mean,
    nextVariance: training ? (1 - momentum) * variance + momentum * 2 * batchVariance : variance,
    gradGamma: recording && trainable ? gradGamma : null, gradBeta: recording && trainable ? gradBeta : null, inputGradient,
    canUpdate: recording && trainable && optimizer,
    nextGamma: recording && trainable && optimizer ? gamma - 0.1 * gradGamma : gamma,
    nextBeta: recording && trainable && optimizer ? beta - 0.1 * gradBeta : beta,
  };
}

export function adapterBudget(dimension, bottleneck, includeHead = true) {
  const down = dimension * bottleneck + bottleneck;
  const up = bottleneck * dimension + dimension;
  const head = dimension * 5 + 5;
  const count = down + up + (includeHead ? head : 0);
  return { down, up, adapter: down + up, head, count, bytes: 12 * count };
}

export function selectTransferCandidate(runs, budget = Infinity, seed = 1) {
  const candidates = TRANSFER_METHODS.map(method => runs.find(row => row.seed === seed && row.method === method)).filter(row => row && row.trainable_parameters <= budget);
  const winner = candidates.reduce((best, row) => !best || row.validation.ce < best.validation.ce ? row : best, null);
  return { candidates, winner };
}
