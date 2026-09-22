// Bounded numerical mechanisms for the convolution lesson. Training runs offline.
const finite = values => values.every(Number.isFinite);
export const initialImage = [[1, 2, 0], [0, 1, 3], [2, 1, 0]];
export const initialKernel = [[1, -1], [0, 1]];

export function correlate2d(image, kernel) {
  const height = image.length - kernel.length + 1;
  const width = image[0].length - kernel[0].length + 1;
  if (height < 1 || width < 1 || !finite([...image.flat(), ...kernel.flat()])) throw new Error('A finite filter must fit the image.');
  return Array.from({ length: height }, (_, row) => Array.from({ length: width }, (_, column) =>
    kernel.reduce((sum, weights, kr) => sum + weights.reduce((part, weight, kc) => part + weight * image[row + kr][column + kc], 0), 0)));
}

export function sharedFilterUpdate(targets = [0, 0], rate = .1) {
  const inputs = [1, 3, 2], weights = [1, -1];
  const output = [-2, 1];
  const errors = output.map((value, i) => value - targets[i]);
  const contributions = errors.map((error, position) => weights.map((_, tap) => error * inputs[position + tap]));
  const gradient = weights.map((_, tap) => contributions.reduce((sum, row) => sum + row[tap], 0));
  const nextWeights = weights.map((weight, tap) => weight - rate * gradient[tap]);
  const nextOutput = output.map((_, position) => nextWeights.reduce((sum, weight, tap) => sum + weight * inputs[position + tap], 0));
  const inputGradient = [errors[0], -errors[0] + errors[1], -errors[1]];
  return { output, errors, contributions, gradient, inputGradient, nextWeights, nextOutput,
    loss: errors.reduce((sum, value) => sum + value ** 2 / 2, 0),
    nextLoss: nextOutput.reduce((sum, value, i) => sum + (value - targets[i]) ** 2 / 2, 0) };
}

export function windowGeometry({ n, k, s = 1, d = 1, left = 0, right = 0 }) {
  if (![n, k, s, d, left, right].every(Number.isInteger) || Math.min(n, k, s, d) < 1 || Math.min(left, right) < 0) throw new Error('Use whole-number sizes and nonnegative padding.');
  const span = d * (k - 1) + 1;
  const count = Math.floor((n + left + right - span) / s) + 1;
  return { span, count, center: .5 + (span - 1) / 2 - left,
    windows: Array.from({ length: Math.max(0, count) }, (_, index) => Array.from({ length: k }, (_, tap) => index * s - left + tap * d)),
    remainder: count > 0 ? n + left + right - ((count - 1) * s + span) : null };
}

export function poolVector(values, size = 2, stride = 1, mode = 'max') {
  const windows = windowGeometry({ n: values.length, k: size, s: stride }).windows;
  const gradient = values.map(() => 0);
  const outputs = windows.map(indices => {
    const items = indices.map(index => values[index]);
    if (mode === 'max') {
      const maximum = Math.max(...items);
      gradient[indices[items.indexOf(maximum)]] += 1;
      return maximum;
    }
    for (const index of indices) gradient[index] += 1 / size;
    return items.reduce((a, b) => a + b, 0) / size;
  });
  return { outputs, gradient, windows };
}

export const cnnAxisLayers = [
  { name: 'Conv 3', k: 3, s: 1, left: 1, right: 1 },
  { name: 'Pool 2', k: 2, s: 2 },
  { name: 'Conv 3', k: 3, s: 1, left: 1, right: 1 },
  { name: 'Pool 2', k: 2, s: 2 },
  { name: 'Conv 3', k: 3, s: 1, left: 1, right: 1 },
  { name: 'Global average', k: 8, s: 1 },
];

export function receptiveTrace(layers, inputSize = 32) {
  let n = inputSize, r = 1, j = 1, a = .5;
  return layers.map(layer => {
    const { k, s = 1, d = 1, left = 0 } = layer;
    const geometry = windowGeometry({ n, ...layer });
    r += d * (k - 1) * j;
    a += (d * (k - 1) / 2 - left) * j;
    j *= s;
    n = geometry.count;
    return { ...layer, n, r, j, a };
  });
}

export function observedAncestors(layers, outputIndex, inputSize = 32) {
  const trace = receptiveTrace(layers, inputSize);
  let indices = new Set([outputIndex]);
  // Padding at any intermediate feature map is a constant, not a path to pixels.
  for (let level = layers.length - 1; level >= 0; level--) {
    const { k, s = 1, d = 1, left = 0 } = layers[level];
    const previousSize = level === 0 ? inputSize : trace[level - 1].n;
    const previous = new Set();
    for (const index of indices) for (let tap = 0; tap < k; tap++) {
      const source = index * s - left + tap * d;
      if (source >= 0 && source < previousSize) previous.add(source);
    }
    indices = previous;
  }
  return [...indices].sort((a, b) => a - b);
}

export function dilationOffsets(first, second) {
  return [...new Set([-1, 0, 1].flatMap(a => [-1, 0, 1].map(b => a * first + b * second)))].sort((a, b) => a - b);
}

export function averagingProfile(depth, threshold) {
  let coefficients = [1];
  for (let layer = 0; layer < depth; layer++) {
    const next = Array(coefficients.length + 2).fill(0);
    coefficients.forEach((value, index) => { for (let tap = 0; tap < 3; tap++) next[index + tap] += value / 3; });
    coefficients = next;
  }
  const peak = Math.max(...coefficients);
  const retained = coefficients.flatMap((value, index) => value >= threshold * peak ? [index - depth] : []);
  return { coefficients, peak, retained, width: retained.at(-1) - retained[0] + 1 };
}

export function transpose1d(values, kernel, stride = 1) {
  const length = (values.length - 1) * stride + kernel.length;
  const contributions = values.map((value, index) => {
    const row = Array(length).fill(0);
    kernel.forEach((weight, tap) => { row[index * stride + tap] += weight * value; });
    return row;
  });
  return { contributions, output: Array.from({ length }, (_, i) => contributions.reduce((sum, row) => sum + row[i], 0)) };
}

export function shiftComparison(boundary = 'circular') {
  const source = [1, 0, 0, 0, 0, 0];
  const shift = values => [boundary === 'circular' ? values.at(-1) : 0, ...values.slice(0, -1)];
  const sample = (values, index) => boundary === 'circular' ? values[(index + values.length) % values.length] : (values[index] ?? 0);
  const correlation = values => values.map((_, i) => sample(values, i - 1) - sample(values, i + 1));
  const shiftedInput = shift(source), first = correlation(shiftedInput), second = shift(correlation(source));
  return { source, shiftedInput, first, second, difference: Math.max(...first.map((v, i) => Math.abs(v - second[i]))) };
}
