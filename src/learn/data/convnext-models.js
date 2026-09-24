// Small deterministic mechanisms. HWC storage is explicit; no training runs here.
export const mean = values => values.reduce((sum, value) => sum + value, 0) / values.length;
export function maxDifference(left, right) {
  const a = left.flat(Infinity), b = right.flat(Infinity);
  if (a.length !== b.length) throw new Error('Comparison shapes have different element counts.');
  return a.reduce((maximum, value, i) => Math.max(maximum, Math.abs(value - b[i])), -Infinity);
}
export function normalizeLocations(locations, mode = 'channels') {
  const groups = mode === 'channels' ? locations : [locations.flat()];
  const statistics = groups.map(values => { const average = mean(values); return { mean: average, variance: mean(values.map(value => (value - average) ** 2)) }; });
  return { statistics, output: locations.map((values, row) => values.map(value => { const stats = statistics[mode === 'channels' ? row : 0]; return (value - stats.mean) / Math.sqrt(stats.variance + 1e-6); })) };
}
export function responseNormalize(maps, scale, shift) {
  const norms = maps.map(values => Math.hypot(...values));
  const denominator = mean(norms) + 1e-6;
  const relative = norms.map(norm => norm / denominator);
  return { norms, denominator, relative, output: maps.map((values, channel) => values.map(value => value + scale[channel] * value * relative[channel] + shift[channel])) };
}
export function blockBudget(channels, expansion, kernel, height, width, version = 1) {
  const depthwise = kernel ** 2 * channels + channels;
  const norm = 2 * channels;
  const expand = expansion * channels ** 2 + expansion * channels;
  const project = expansion * channels ** 2 + channels;
  const response = version === 1 ? channels : 2 * expansion * channels;
  return { depthwise, norm, expand, project, response, parameters: depthwise + norm + expand + project + response,
    spatialMacs: height * width * kernel ** 2 * channels, channelMacs: height * width * 2 * expansion * channels ** 2,
    macs: height * width * (kernel ** 2 * channels + 2 * expansion * channels ** 2) };
}
export const defaultFusion = { kernel: Array.from({ length: 9 }, (_, i) => i / 10), small: 2, bias: .4, smallBias: -.3, mean: 1, variance: 4, gamma: 3, beta: -.2, smallMean: -2, smallVariance: 1, smallGamma: .5, smallBeta: .7, image: Array.from({ length: 25 }, (_, i) => i + 1) };
function stencil(image, kernel, bias) {
  return image.map((_, position) => { const row = Math.floor(position / 5), column = position % 5;
    return kernel.reduce((sum, coefficient, index) => { const y = row + Math.floor(index / 3) - 1, x = column + index % 3 - 1; return sum + (y >= 0 && y < 5 && x >= 0 && x < 5 ? image[y * 5 + x] * coefficient : 0); }, bias); });
}
export function foldBranches(config, nonlinear = false) {
  const a = config.gamma / Math.sqrt(config.variance + 1e-5), b = config.smallGamma / Math.sqrt(config.smallVariance + 1e-5);
  const firstKernel = config.kernel.map(value => value * a), firstBias = (config.bias - config.mean) * a + config.beta;
  const secondKernel = config.small * b, secondBias = (config.smallBias - config.smallMean) * b + config.smallBeta;
  const foldedKernel = firstKernel.map((value, i) => value + (i === 4 ? secondKernel + 1 : 0));
  const first = stencil(config.image, firstKernel, firstBias), second = config.image.map(value => (nonlinear ? -value : value) * secondKernel + secondBias);
  const separate = first.map((value, i) => nonlinear ? Math.max(0, value) + Math.max(0, second[i]) : value + second[i] + config.image[i]);
  const folded = nonlinear ? first.map((value, i) => Math.max(0, value + second[i])) : stencil(config.image, foldedKernel, firstBias + secondBias);
  return { firstKernel, firstBias, secondKernel, secondBias, foldedKernel, foldedBias: firstBias + secondBias, first, second, separate, folded, difference: maxDifference(separate, folded) };
}

// erf approximation (maximum absolute error about 1.5e-7) for the bounded browser
// inference. End-to-end tolerance is checked against exact-erf NumPy and PyTorch.
export function gelu(value) {
  const z = Math.abs(value) / Math.SQRT2, t = 1 / (1 + .3275911 * z);
  const erf = Math.sign(value) * (1 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - .284496736) * t + .254829592) * t * Math.exp(-z * z));
  return .5 * value * (1 + erf);
}
function convolution(values, weights, bias, stride = 1, padding = 0, depthwise = false) {
  const height = Math.floor((values.length + 2 * padding - weights[0][0].length) / stride) + 1;
  const width = Math.floor((values[0].length + 2 * padding - weights[0][0][0].length) / stride) + 1;
  return Array.from({ length: height }, (_, row) => Array.from({ length: width }, (_, column) => weights.map((kernel, output) => {
    let sum = bias[output];
    for (let channel = 0; channel < kernel.length; channel++) for (let y = 0; y < kernel[channel].length; y++) for (let x = 0; x < kernel[channel][y].length; x++) {
      const source = values[row * stride + y - padding]?.[column * stride + x - padding];
      if (source) sum += source[depthwise ? output : channel] * kernel[channel][y][x];
    }
    return sum;
  })));
}
function channelNorm(values, weights, bias) {
  return values.map(row => row.map(cell => { const average = mean(cell), variance = mean(cell.map(value => (value - average) ** 2)); return cell.map((value, i) => (value - average) / Math.sqrt(variance + 1e-6) * weights[i] + bias[i]); }));
}
function linear(values, weights, bias) { return values.map(row => row.map(cell => weights.map((weight, i) => weight.reduce((sum, value, j) => sum + value * cell[j], bias[i])))); }
function maskFeatures(values, mask) { return values.map((row, y) => row.map((cell, x) => cell.map(value => value * mask[y][x]))); }
export function reconstruct(state, pixels, mask) {
  if (pixels.length !== 8 || pixels.some(row => row.length !== 8 || row.some(value => !Number.isFinite(value) || value < 0 || value > 1))) throw new Error('Expected 8 × 8 pixels in [0, 1].');
  if (mask.length !== 4 || mask.some(row => row.length !== 4 || row.some(value => value !== 0 && value !== 1)) || mask.flat().reduce((sum, value) => sum + value, 0) !== 6) throw new Error('Exactly six of sixteen patches must be visible.');
  function block(inputs, prefix, visibility) {
    const original = maskFeatures(inputs, visibility);
    let values = maskFeatures(convolution(original, state[prefix + '.spatial.weight'], state[prefix + '.spatial.bias'], 1, 1, true), visibility);
    values = channelNorm(values, state[prefix + '.norm.weight'], state[prefix + '.norm.bias']);
    values = maskFeatures(linear(values, state[prefix + '.expand.weight'], state[prefix + '.expand.bias']).map(row => row.map(cell => cell.map(gelu))), visibility);
    if (state[prefix + '.response.scale']) {
      const channels = values[0][0].length, maps = Array.from({ length: channels }, (_, c) => values.flat().map(cell => cell[c]));
      const result = responseNormalize(maps, state[prefix + '.response.scale'], state[prefix + '.response.shift']);
      values = values.map((row, y) => row.map((cell, x) => cell.map((_, c) => result.output[c][y * 4 + x])));
    }
    values = linear(values, state[prefix + '.project.weight'], state[prefix + '.project.bias']);
    return maskFeatures(values.map((row, y) => row.map((cell, x) => cell.map((value, c) => value + original[y][x][c]))), visibility);
  }
  const visibleInput = pixels.map((row, y) => row.map((value, x) => value * mask[Math.floor(y / 2)][Math.floor(x / 2)]));
  let values = convolution(visibleInput.map(row => row.map(value => [value])), state['stem.weight'], state['stem.bias'], 2);
  values = maskFeatures(channelNorm(values, state['stem_norm.weight'], state['stem_norm.bias']), mask);
  values = block(block(values, 'encoder.0', mask), 'encoder.1', mask);
  values = values.map((row, y) => row.map((cell, x) => cell.map((value, c) => value + state.mask_token[0][c][0][0] * (1 - mask[y][x]))));
  values = block(values, 'decoder', Array.from({ length: 4 }, () => [1, 1, 1, 1]));
  const packed = convolution(values, state['pixel_head.weight'], state['pixel_head.bias']);
  const output = Array.from({ length: 8 }, (_, y) => Array.from({ length: 8 }, (_, x) => packed[Math.floor(y / 2)][Math.floor(x / 2)][(y % 2) * 2 + x % 2]));
  const errors = output.map((row, y) => row.map((value, x) => mask[Math.floor(y / 2)][Math.floor(x / 2)] ? 0 : (value - pixels[y][x]) ** 2));
  return { output, visibleInput, errors, mse: errors.flat().reduce((sum, value) => sum + value, 0) / 40 };
}
