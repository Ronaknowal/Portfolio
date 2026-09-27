// Bounded, pure teaching operators. Rows represent time; state is read after updating.
export const dot = (a, b) => a.reduce((sum, value, i) => sum + value * b[i], 0);
const zeros = (rows, columns) => Array.from({ length: rows }, () => Array(columns).fill(0));
const softplus = value => Math.max(0, value) + Math.log1p(Math.exp(-Math.abs(value)));
export function maximumDifference(a, b) {
  const left = a.flat(Infinity), right = b.flat(Infinity);
  if (left.length !== right.length) throw new Error('Compared arrays must have the same size.');
  return Math.max(0, ...left.map((value, i) => Math.abs(value - right[i])));
}

// Iterative radix-2 FFT: genuinely independent of the direct convolution below.
function fft(real, imaginary, inverse = false) {
  const size = real.length;
  for (let i = 1, j = 0; i < size; i++) {
    let bit = size >> 1;
    for (; j & bit; bit >>= 1) j ^= bit;
    j ^= bit;
    if (i < j) {
      [real[i], real[j]] = [real[j], real[i]];
      [imaginary[i], imaginary[j]] = [imaginary[j], imaginary[i]];
    }
  }
  for (let length = 2; length <= size; length *= 2) {
    const angle = (inverse ? 2 : -2) * Math.PI / length;
    for (let start = 0; start < size; start += length) {
      for (let j = 0; j < length / 2; j++) {
        const a = start + j, b = a + length / 2;
        const cosine = Math.cos(angle * j), sine = Math.sin(angle * j);
        const r = real[b] * cosine - imaginary[b] * sine;
        const im = real[b] * sine + imaginary[b] * cosine;
        real[b] = real[a] - r; imaginary[b] = imaginary[a] - im;
        real[a] += r; imaginary[a] += im;
      }
    }
  }
  if (inverse) for (let i = 0; i < size; i++) { real[i] /= size; imaginary[i] /= size; }
}

export function fftConvolution(input, kernel) {
  const size = 2 ** Math.ceil(Math.log2(input.length + kernel.length - 1));
  const real = Array(size).fill(0), imaginary = Array(size).fill(0);
  const kr = Array(size).fill(0), ki = Array(size).fill(0);
  input.forEach((value, i) => { real[i] = value; });
  kernel.forEach((value, i) => { kr[i] = value; });
  fft(real, imaginary); fft(kr, ki);
  for (let i = 0; i < size; i++) {
    const product = real[i] * kr[i] - imaginary[i] * ki[i];
    imaginary[i] = real[i] * ki[i] + imaginary[i] * kr[i]; real[i] = product;
  }
  fft(real, imaginary, true);
  return real.slice(0, input.length);
}

export const systemDefault = () => ({ rates: [-1, -2], write: [1, 1], read: [1, -.5], direct: .25, interval: Math.log(2), initial: [0, 0], inputs: [1, -2, 3, 0] });
export function linearSystem({ rates, write, read, direct, interval, initial, inputs }) {
  const transitions = rates.map(rate => Math.exp(interval * rate));
  const injections = rates.map((rate, i) => write[i] * (rate === 0 ? interval : Math.expm1(interval * rate) / rate));
  const kernel = inputs.map((_, t) => dot(read, transitions.map((a, i) => a ** t * injections[i])));
  const initialResponse = inputs.map((_, t) => dot(read, transitions.map((a, i) => a ** (t + 1) * initial[i])));
  const contributions = inputs.map((value, j) => inputs.map((_, t) => t < j ? 0 : value * kernel[t - j]));
  let state = [...initial];
  const states = [], memory = [], recurrent = [];
  for (const input of inputs) {
    state = state.map((previous, i) => transitions[i] * previous + injections[i] * input);
    states.push([...state]); memory.push(dot(read, state)); recurrent.push(dot(read, state) + direct * input);
  }
  const directConvolution = inputs.map((input, t) => contributions.reduce((sum, row) => sum + row[t], 0) + initialResponse[t] + direct * input);
  const frequencyConvolution = fftConvolution(inputs, kernel).map((value, t) => value + initialResponse[t] + direct * inputs[t]);
  return { transitions, injections, kernel, initialResponse, contributions, states, memory, recurrent, directConvolution, frequencyConvolution, difference: Math.max(maximumDifference(recurrent, directConvolution), maximumDifference(recurrent, frequencyConvolution)) };
}

export const selectionDefault = () => ({ inputs: [3, -8, 5, -2], marked: [true, false, false, true], gates: [.99, .01, .01, .99], initial: 0, constant: .5, independent: false, retention: .5 });
export function selectiveMemory({ inputs, marked, gates, initial, constant, independent, retention }) {
  let state = initial, fixed = initial, target = null;
  return inputs.map((input, index) => {
    if (marked[index]) target = input;
    const retained = (independent ? retention : 1 - gates[index]) * state;
    const incoming = gates[index] * input;
    state = retained + incoming;
    fixed = (1 - constant) * fixed + constant * input;
    return { input, retained, incoming, state, fixed, target, error: target === null ? null : Math.abs(state - target), fixedError: target === null ? null : Math.abs(fixed - target) };
  });
}

export const ssdDefault = () => ({ decay: [.5, .5, .25, .8], write: [[1, 0], [0, 1], [1, 1], [1, -1]], read: [[1, 0], [1, 1], [0, 1], [1, 2]], values: [[2, 1], [3, -1], [-1, 3], [-2, 1]], initial: [[0, 0], [0, 0]], chunkSize: 3 });
export function ssdOperator({ decay, write, read, values, initial, chunkSize }) {
  const length = decay.length, n = write[0].length, p = values[0].length;
  const content = read.map(c => write.map(b => dot(c, b)));
  const mask = zeros(length, length);
  for (let i = 0; i < length; i++) {
    let product = 1; mask[i][i] = 1;
    for (let j = i - 1; j >= 0; j--) { product *= decay[j + 1]; mask[i][j] = product; }
  }
  const influence = content.map((row, i) => row.map((value, j) => value * mask[i][j]));
  let state = initial.map(row => [...row]);
  const states = [], recurrent = [];
  for (let t = 0; t < length; t++) {
    state = state.map((row, i) => row.map((value, j) => decay[t] * value + write[t][i] * values[t][j]));
    states.push(state); recurrent.push(Array.from({ length: p }, (_, j) => dot(read[t], state.map(row => row[j]))));
  }
  let product = 1;
  const matrix = influence.map((row, t) => {
    product *= decay[t];
    return Array.from({ length: p }, (_, j) => row.reduce((sum, coefficient, i) => sum + coefficient * values[i][j], 0) + product * dot(read[t], initial.map(s => s[j])));
  });
  const chunks = [], chunked = [];
  let carry = initial.map(row => [...row]);
  for (let start = 0; start < length; start += chunkSize) {
    const stop = Math.min(start + chunkSize, length), local = [], incoming = [];
    const ownFinal = zeros(n, p);
    let totalDecay = 1;
    for (let t = start; t < stop; t++) {
      totalDecay *= decay[t];
      local.push(Array.from({ length: p }, (_, j) => {
        let sum = 0;
        for (let k = start; k <= t; k++) sum += influence[t][k] * values[k][j];
        return sum;
      }));
      incoming.push(Array.from({ length: p }, (_, j) => totalDecay * dot(read[t], carry.map(s => s[j]))));
      let remaining = 1;
      for (let next = t + 1; next < stop; next++) remaining *= decay[next];
      for (let i = 0; i < n; i++) for (let j = 0; j < p; j++) ownFinal[i][j] += remaining * write[t][i] * values[t][j];
    }
    const total = local.map((row, i) => row.map((value, j) => value + incoming[i][j]));
    chunks.push({ start, stop, local, incoming, total, ownFinal, carry, totalDecay });
    chunked.push(...total);
    carry = carry.map((row, i) => row.map((value, j) => totalDecay * value + ownFinal[i][j]));
  }
  return { content, mask, influence, states, recurrent, matrix, chunks, chunked, difference: Math.max(maximumDifference(recurrent, matrix), maximumDifference(recurrent, chunked)) };
}

const affine = (input, weight, bias) => weight.map((row, i) => dot(row, input) + bias[i]);
function erf(value) {
  const sign = value < 0 ? -1 : 1, x = Math.abs(value);
  const t = 1 / (1 + .3275911 * x);
  return sign * (1 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - .284496736) * t + .254829592) * t * Math.exp(-x * x));
}
function layerNorm(input, weight, bias) {
  const mean = input.reduce((sum, value) => sum + value, 0) / input.length;
  const variance = input.reduce((sum, value) => sum + (value - mean) ** 2, 0) / input.length;
  return input.map((value, i) => (value - mean) / Math.sqrt(variance + 1e-5) * weight[i] + bias[i]);
}
function diagonalMixer(inputs, weights, prefix) {
  const decay = weights[prefix + 'log_decay'], frequency = weights[prefix + 'frequency'];
  const step = weights[prefix + 'log_step'].map(Math.exp);
  const stateReal = zeros(16, 4), stateImaginary = zeros(16, 4);
  const coefficients = decay.map((row, d) => row.map((logDecay, n) => {
    const a = -Math.exp(logDecay), omega = frequency[d][n], radius = Math.exp(step[d] * a), angle = step[d] * omega;
    const ar = radius * Math.cos(angle), ai = radius * Math.sin(angle);
    const denominator = a * a + omega * omega;
    return [ar, ai, ((ar - 1) * a + ai * omega) / denominator, (ai * a - (ar - 1) * omega) / denominator];
  }));
  return inputs.map(input => input.map((value, d) => {
    let result = weights[prefix + 'skip'][d] * value;
    for (let n = 0; n < 4; n++) {
      const [ar, ai, br, bi] = coefficients[d][n];
      const nextReal = ar * stateReal[d][n] - ai * stateImaginary[d][n] + br * value;
      stateImaginary[d][n] = ar * stateImaginary[d][n] + ai * stateReal[d][n] + bi * value;
      stateReal[d][n] = nextReal;
      result += 2 * (weights[prefix + 'read_real'][d][n] * nextReal - weights[prefix + 'read_imag'][d][n] * stateImaginary[d][n]);
    }
    return result;
  }));
}
function selectiveMixer(inputs, weights, prefix) {
  const state = zeros(16, 8);
  return inputs.map(input => {
    const coefficients = affine(input, weights[prefix + 'coefficients.weight'], weights[prefix + 'coefficients.bias']);
    return input.map((value, d) => {
      const delta = softplus(coefficients[16 + d]);
      let result = weights[prefix + 'skip'][d] * value;
      for (let n = 0; n < 8; n++) {
        state[d][n] = Math.exp(-delta * Math.exp(weights[prefix + 'log_decay'][d][n])) * state[d][n] + delta * coefficients[n] * value;
        result += coefficients[8 + n] * state[d][n];
      }
      return result;
    });
  });
}
export function trajectoryForward(points, weights, kind) {
  let hidden = points.map(point => affine(point.map(value => 2 * value - 1), weights['input.weight'], weights['input.bias']));
  const traces = [];
  for (let block = 0; block < 2; block++) {
    const normalized = hidden.map(row => layerNorm(row, weights[`norms.${block}.weight`], weights[`norms.${block}.bias`]));
    const mixed = (kind === 'diagonal' ? diagonalMixer : selectiveMixer)(normalized, weights, `mixers.${block}.`);
    traces.push(mixed);
    hidden = hidden.map((row, t) => {
      const activated = mixed[t].map(value => .5 * value * (1 + erf(value / Math.SQRT2)));
      const projected = affine(activated, weights[`outputs.${block}.weight`], weights[`outputs.${block}.bias`]);
      return row.map((value, j) => value + projected[j]);
    });
  }
  const pooled = hidden[0].map((_, j) => hidden.reduce((sum, row) => sum + row[j], 0) / hidden.length);
  const logits = affine(pooled, weights['classifier.weight'], weights['classifier.bias']);
  const exponentials = logits.map(value => Math.exp(value - Math.max(...logits)));
  const denominator = exponentials.reduce((a, b) => a + b, 0);
  const probabilities = exponentials.map(value => value / denominator);
  return { logits, probabilities, top: probabilities.indexOf(Math.max(...probabilities)) + 1, traces };
}
