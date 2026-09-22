// Small, deterministic mechanisms. These are not browser-trained digit models.
export const correctionFixture = () => [[0.1, 0.2], [0, -0.5]];
export const dot = (left, right) => left.reduce((sum, value, i) => sum + value * right[i], 0);
export const matvec = (matrix, vector) => matrix.map(row => dot(row, vector));
export const transposeMultiply = (matrix, vector) => matrix[0].map((_, column) => matrix.reduce((sum, row, i) => sum + row[column] * vector[i], 0));
export function correctionModel(weight, rate = 0.1, input = [2, -1], target = [1, 0]) {
  const correction = matvec(weight, input);
  const output = input.map((value, i) => value + correction[i]);
  const upstream = output.map((value, i) => value - target[i]);
  const branchGradient = transposeMultiply(weight, upstream);
  const totalGradient = upstream.map((value, i) => value + branchGradient[i]);
  const weightGradient = upstream.map(value => input.map(entry => value * entry));
  const newWeight = weight.map((row, i) => row.map((value, j) => value - rate * weightGradient[i][j]));
  const newOutput = matvec(newWeight, input).map((value, i) => value + input[i]);
  return { input, target, correction, output, upstream, branchGradient, totalGradient, weightGradient, newWeight, newOutput, loss: 0.5 * dot(upstream, upstream), newLoss: 0.5 * newOutput.reduce((sum, value, i) => sum + (value - target[i]) ** 2, 0) };
}
export function scalarStack(slope, depth) {
  if (!Number.isFinite(slope) || !Number.isInteger(depth) || depth < 1 || depth > 20) throw new RangeError('Finite slope and 1–20 blocks required');
  const multiplier = 1 + slope;
  const trace = Array.from({ length: depth + 1 }, (_, step) => ({ step, value: multiplier ** step }));
  return { multiplier, trace, gain: trace.at(-1).value };
}
export function operatorPlacement(mode, commonShift = 0, positive = false) {
  const input = (positive ? [1, 2] : [-2, 1]).map(value => value + commonShift);
  let output = [...input]; let jacobian = [[1, 0], [0, 1]];
  if (mode === 'post-relu') {
    output = input.map(value => Math.max(0, value));
    jacobian = [[input[0] > 0 ? 1 : 0, 0], [0, input[1] > 0 ? 1 : 0]];
  } else if (mode === 'post-ln') {
    const mean = (input[0] + input[1]) / 2;
    const centered = input.map(value => value - mean);
    const variance = dot(centered, centered) / 2;
    const standardDeviation = Math.sqrt(variance + 1e-5);
    output = centered.map(value => value / standardDeviation);
    jacobian = input.map((_, i) => input.map((__, j) => ((i === j ? 1 : 0) - 0.5 - centered[i] * centered[j] / (2 * (variance + 1e-5))) / standardDeviation));
  }
  return { input, output, jacobian, shiftDirection: matvec(jacobian, [1, 1]), difference: output.map((value, i) => value - input[i]) };
}
export function projectionModel(thirdRow, changed = false, enabled = true, broadcast = false) {
  const input = changed ? [1, 3] : [2, -1];
  const upstream = changed ? [2, -1, 4] : [1, 2, 3];
  const projection = [[1, 0], [0, 1], thirdRow];
  const correction = broadcast ? [0.5, 0.5, 0.5] : [0.1, -0.2, 0.3];
  const skip = enabled ? matvec(projection, input) : input;
  return { input, projection, upstream, skip, correction, compatible: enabled, output: enabled ? skip.map((value, i) => value + correction[i]) : null, gradient: enabled ? transposeMultiply(projection, upstream) : upstream.slice(0, 2) };
}
export function gateModel(scale, kind = 'scalar', firstCorrection = 0) {
  const input = [1, 2], branch = [firstCorrection, 0.7];
  const output = input.map((value, i) => value + scale * branch[i]);
  const featureGradient = output.map((value, i) => value * branch[i]);
  const scaleGradient = kind === 'channel' ? featureGradient : dot(output, branch);
  const nextScale = kind === 'channel' ? featureGradient.map(value => scale - 0.1 * value) : scale - 0.1 * scaleGradient;
  const branchGradient = output.map(value => input.map(entry => scale * value * entry));
  const nextOutput = input.map((value, i) => value + (kind === 'channel' ? nextScale[i] : nextScale) * branch[i]);
  return { input, branch, output, scaleGradient, branchGradient, nextScale, nextOutput, loss: dot(output, output) / 2, nextLoss: dot(nextOutput, nextOutput) / 2 };
}
export function nonlinearPaths(input) {
  const first = input + input ** 2;
  return { first, actual: first + first ** 2, invalid: input + 2 * input ** 2 + input ** 4 };
}
