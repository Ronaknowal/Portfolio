// Finite educational models. Complex numbers are [real, imaginary] pairs.
// Forward Fourier sums use exp(-2πikn/N); inverse sums divide by N.
const TAU = 2 * Math.PI;
function bounded(value, name, low, high) {
  if (typeof value !== 'number' || !Number.isFinite(value) || value < low || value > high) {
    throw new RangeError(`${name} must be finite and between ${low} and ${high}.`);
  }
  return value;
}
function integer(value, name, low, high) {
  bounded(value, name, low, high);
  if (!Number.isInteger(value)) throw new RangeError(`${name} must be an integer.`);
  return value;
}
function denseArray(values, name, low, high) {
  if (!Array.isArray(values) || values.length < low || values.length > high) {
    throw new RangeError(`${name} needs ${low}–${high} entries.`);
  }
  for (let index = 0; index < values.length; index += 1) {
    if (!Object.hasOwn(values, index)) throw new RangeError(`${name} cannot contain missing entries.`);
  }
  return values;
}
function finiteResult(value) {
  if (!Number.isFinite(value)) throw new RangeError('This result cannot be represented in the finite arithmetic model.');
  return value;
}
function pair(value, limit = 1e6) {
  denseArray(value, 'Complex coordinate', 2, 2);
  return value.map(coordinate => bounded(coordinate, 'Complex coordinate', -limit, limit));
}
export const addComplex = (left, right) => [left[0] + right[0], left[1] + right[1]];
export const multiplyComplex = (left, right) => [left[0] * right[0] - left[1] * right[1], left[0] * right[1] + left[1] * right[0]];
const scaleComplex = (value, scale) => [value[0] * scale, value[1] * scale];
const unitComplex = angle => [Math.cos(angle), Math.sin(angle)];
export function complexPolar(value, displayTolerance = 0) {
  const magnitude = Math.hypot(...value);
  let phase = magnitude <= displayTolerance ? null : Math.atan2(value[1], value[0]);
  if (phase === -Math.PI) phase = Math.PI;
  return {
    magnitude,
    phase
  };
}
export function complexOperation(left, right, operation = 'multiply') {
  const z = pair(left),
    w = pair(right);
  let result;
  if (operation === 'add') result = addComplex(z, w);else if (operation === 'multiply') result = multiplyComplex(z, w);else if (operation === 'divide') {
    const norm = Math.hypot(...w);
    if (norm === 0) throw new RangeError('Division by zero is undefined. Keep a nonzero divisor.');
    const unit = [w[0] / norm, -w[1] / norm];
    result = scaleComplex(multiplyComplex(z, unit), 1 / norm).map(finiteResult);
  } else throw new RangeError('Choose add, multiply or divide.');
  return {
    z,
    w,
    result,
    zPolar: complexPolar(z),
    wPolar: complexPolar(w),
    resultPolar: complexPolar(result)
  };
}
export function rootsOfComplex(value, count) {
  const z = pair(value);
  integer(count, 'Root count', 1, 12);
  const {
    magnitude,
    phase
  } = complexPolar(z);
  if (magnitude === 0) return [[0, 0]];
  return Array.from({
    length: count
  }, (_, k) => scaleComplex(unitComplex((phase + TAU * k) / count), magnitude ** (1 / count)));
}
export function harmonicValue(time, amplitudes = [2, 1], phase = Math.PI / 2, offset = 0) {
  return offset + amplitudes[0] * Math.cos(TAU * time) + amplitudes[1] * Math.cos(3 * TAU * time + phase);
}
export function phasorSynthesis(time = 0, phase = Math.PI / 2) {
  bounded(time, 'Phasor time', 0, 1);
  bounded(phase, 'Third-tone phase', -Math.PI, Math.PI);
  const first = scaleComplex(unitComplex(TAU * time), 2);
  const third = unitComplex(3 * TAU * time + phase);
  return {
    time,
    phase,
    first,
    third,
    signal: first[0] + third[0]
  };
}
function cosineIntegral(frequency, phase, end) {
  if (frequency === 0) return end * Math.cos(phase);
  // Midpoint/sinc form avoids cancellation at small end.
  return end * sinc(frequency * end) * Math.cos(Math.PI * frequency * end + phase);
}
function sineIntegral(frequency, phase, end) {
  if (frequency === 0) return end * Math.sin(phase);
  return end * sinc(frequency * end) * Math.sin(Math.PI * frequency * end + phase);
}
export function harmonicProjection(amplitude1 = 2, amplitude3 = 1, phase = Math.PI / 2, offset = 0, harmonic = 1) {
  bounded(amplitude1, 'First amplitude', 0, 3);
  bounded(amplitude3, 'Third amplitude', 0, 3);
  bounded(phase, 'Phase', -Math.PI, Math.PI);
  bounded(offset, 'Offset', -2, 2);
  integer(harmonic, 'Harmonic', -4, 4);
  const integralAt = time => {
    let real = offset * cosineIntegral(harmonic, 0, time);
    let imaginary = -offset * sineIntegral(harmonic, 0, time);
    for (const [frequency, amplitude, angle] of [[1, amplitude1, 0], [3, amplitude3, phase]]) {
      real += amplitude / 2 * (cosineIntegral(frequency - harmonic, angle, time) + cosineIntegral(frequency + harmonic, angle, time));
      imaginary += amplitude / 2 * (sineIntegral(frequency - harmonic, angle, time) - sineIntegral(frequency + harmonic, angle, time));
    }
    return [real, imaginary];
  };
  const points = Array.from({
    length: 257
  }, (_, index) => {
    const time = index / 256;
    const signal = harmonicValue(time, [amplitude1, amplitude3], phase, offset);
    const integrand = scaleComplex(unitComplex(-TAU * harmonic * time), signal);
    return {
      time,
      signal,
      integrand,
      accumulated: integralAt(time)
    };
  });
  return {
    harmonic,
    points,
    coefficient: integralAt(1),
    meanSquare: offset ** 2 + (amplitude1 ** 2 + amplitude3 ** 2) / 2
  };
}
export function squarePartial(time, terms) {
  integer(terms, 'Odd harmonic terms', 1, 64);
  bounded(time, 'Time', -2, 2);
  let result = 0;
  for (let index = 0; index < terms; index += 1) {
    const frequency = 2 * index + 1;
    result += 4 * Math.sin(TAU * frequency * time) / (Math.PI * frequency);
  }
  return result;
}
export function squareConvergence(terms = 4) {
  integer(terms, 'Odd harmonic terms', 1, 64);
  // Derivative: 8 Σ cos(2π(2j+1)t), whose first positive zero is 1/(4m).
  const firstPeakTime = 1 / (4 * terms);
  let retainedEnergy = 0;
  for (let index = 0; index < terms; index += 1) retainedEnergy += 8 / (Math.PI ** 2 * (2 * index + 1) ** 2);
  return {
    terms,
    firstPeakTime,
    firstPeak: squarePartial(firstPeakTime, terms),
    midpoint: squarePartial(0, terms),
    away: squarePartial(.2, terms),
    meanSquaredError: 1 - retainedEnergy,
    nearPoints: Array.from({
      length: 193
    }, (_, index) => {
      const scaledTime = 1.8 * index / 192;
      const time = scaledTime / (4 * terms);
      return {
        scaledTime,
        time,
        value: squarePartial(time, terms)
      };
    }),
    points: Array.from({
      length: 1025
    }, (_, index) => {
      const time = -.25 + index / 2048;
      return {
        time,
        value: squarePartial(time, terms)
      };
    })
  };
}
export function sinc(value) {
  if (value === 0) return 1;
  const angle = Math.PI * value;
  if (Math.abs(angle) < 1e-4) return 1 - angle ** 2 / 6 + angle ** 4 / 120;
  return Math.sin(angle) / angle;
}
export function parseRealSamples(text, allowedLengths = [4, 8]) {
  if (typeof text !== 'string') throw new RangeError('Enter a comma-separated sample list.');
  const pieces = text.split(',');
  if (!allowedLengths.includes(pieces.length)) throw new RangeError(`Use exactly ${allowedLengths.join(' or ')} samples.`);
  return pieces.map(piece => {
    const token = piece.trim();
    if (!/^[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:e[+-]?\d+)?$/i.test(token)) throw new RangeError('Every sample must be a finite number; empty entries are not zero.');
    const value = bounded(Number(token), 'Sample', -4, 4);
    if (value === 0 && /[1-9]/.test(token.split(/[eE]/)[0])) throw new RangeError('A nonzero sample is too small to represent.');
    return value;
  });
}
function complexSamples(values, limit = 1e6) {
  denseArray(values, 'Samples', 1, 512);
  return values.map(value => typeof value === 'number' ? [bounded(value, 'Sample', -limit, limit), 0] : pair(value, limit));
}
export function dft(values, inverse = false) {
  if (typeof inverse !== 'boolean') throw new RangeError('Inverse selection must be boolean.');
  // The inverse admits every coefficient produced by the bounded forward call.
  const samples = complexSamples(values, inverse ? 1024e6 : 1e6);
  const count = samples.length,
    direction = inverse ? 1 : -1;
  return Array.from({
    length: count
  }, (_, frequency) => {
    let sum = [0, 0];
    for (let index = 0; index < count; index += 1) {
      // Reducing the integer product modulo N improves known-root accuracy.
      sum = addComplex(sum, multiplyComplex(samples[index], unitComplex(direction * TAU * (index * frequency % count) / count)));
    }
    return scaleComplex(sum, inverse ? 1 / count : 1);
  });
}
export function finiteFourier(values = [1, 2, 0, -1], bin = 1, removePair = false) {
  const samples = complexSamples(values);
  integer(bin, 'Selected bin', 0, samples.length - 1);
  const spectrum = dft(samples);
  const filtered = spectrum.map((value, index) => removePair && (index === bin || index === (samples.length - bin) % samples.length) ? [0, 0] : [...value]);
  const contributions = samples.map((value, index) => multiplyComplex(value, unitComplex(-TAU * (index * bin % samples.length) / samples.length)));
  return {
    samples,
    spectrum,
    filtered,
    contributions,
    reconstructed: dft(filtered, true),
    timeEnergy: samples.reduce((sum, value) => sum + value[0] ** 2 + value[1] ** 2, 0),
    frequencyEnergy: spectrum.reduce((sum, value) => sum + value[0] ** 2 + value[1] ** 2, 0) / samples.length
  };
}
export function aliasState(frequency = 13, sampleRate = 16, phase = Math.PI / 3) {
  bounded(frequency, 'Tone frequency', 0, 24);
  bounded(sampleRate, 'Sample rate', 4, 32);
  bounded(phase, 'Tone phase', -Math.PI, Math.PI);
  const signedAlias = frequency - sampleRate * Math.floor(frequency / sampleRate + .5);
  const foldedFrequency = Math.abs(signedAlias),
    foldedPhase = signedAlias < 0 ? -phase : phase;
  const originalAt = time => Math.cos(TAU * frequency * time + phase);
  const aliasAt = time => Math.cos(TAU * foldedFrequency * time + foldedPhase);
  return {
    frequency,
    sampleRate,
    phase,
    signedAlias,
    foldedFrequency,
    foldedPhase,
    samples: Array.from({
      length: Math.floor(sampleRate) + 1
    }, (_, index) => ({
      index,
      time: index / sampleRate,
      original: originalAt(index / sampleRate),
      alias: aliasAt(index / sampleRate)
    })),
    curves: Array.from({
      length: 1025
    }, (_, index) => ({
      time: index / 1024,
      original: originalAt(index / 1024),
      alias: aliasAt(index / 1024)
    }))
  };
}
function transformAt(samples, frequency, sampleRate) {
  let real = 0,
    imaginary = 0;
  for (let index = 0; index < samples.length; index += 1) {
    const angle = TAU * frequency * index / sampleRate;
    real += samples[index] * Math.cos(angle);
    imaginary -= samples[index] * Math.sin(angle);
  }
  return [real, imaginary];
}
export function windowSpectrum(count = 64, paddedCount = 64, windowName = 'rectangular', tone = 5.5) {
  if (![32, 64, 128].includes(count)) throw new RangeError('Observation count must be 32, 64 or 128.');
  if (![32, 64, 128, 256, 512].includes(paddedCount) || paddedCount < count) throw new RangeError('Transform length must be a supported size at least as long as the observation.');
  if (!['rectangular', 'hann'].includes(windowName)) throw new RangeError('Choose rectangular or periodic Hann.');
  bounded(tone, 'Window tone', 3, 9);
  const sampleRate = 64;
  const samples = Array.from({
    length: count
  }, (_, index) => Math.cos(TAU * tone * index / sampleRate));
  const weights = samples.map((_, index) => windowName === 'hann' ? .5 - .5 * Math.cos(TAU * index / count) : 1);
  const weighted = samples.map((value, index) => value * weights[index]);
  const weightSum = weights.reduce((sum, value) => sum + value, 0);
  const weightSquares = weights.reduce((sum, value) => sum + value * value, 0);
  const bins = Array.from({
    length: paddedCount / 2 + 1
  }, (_, index) => {
    const frequency = index * sampleRate / paddedCount;
    const value = transformAt(weighted, frequency, sampleRate);
    const factor = index === 0 || index === paddedCount / 2 ? 1 : 2;
    const magnitude = Math.hypot(...value);
    return {
      index,
      frequency,
      value,
      amplitude: factor * magnitude / weightSum,
      density: factor * magnitude ** 2 / (sampleRate * weightSquares)
    };
  });
  const curve = Array.from({
    length: 769
  }, (_, index) => {
    const frequency = index / 64;
    return {
      frequency,
      amplitude: (index === 0 ? 1 : 2) * Math.hypot(...transformAt(weighted, frequency, sampleRate)) / weightSum
    };
  });
  return {
    sampleRate,
    count,
    paddedCount,
    windowName,
    tone,
    samples,
    weights,
    weighted,
    weightSum,
    weightSquares,
    bins,
    curve,
    gridSpacing: sampleRate / paddedCount,
    observationPeriod: count / sampleRate,
    weightedMeanSquare: weighted.reduce((sum, value) => sum + value * value, 0) / weightSquares,
    densityIntegral: bins.reduce((sum, row) => sum + row.density, 0) * sampleRate / paddedCount
  };
}
export function convolutionState(values = [1, 2, 0, -1], kernel = [1, 1], circular = false, outputIndex = 0) {
  const samples = complexSamples(values).map(value => {
    if (value[1] !== 0) throw new RangeError('The convolution display uses real sequences.');
    return value[0];
  });
  const weights = complexSamples(kernel).map(value => {
    if (value[1] !== 0) throw new RangeError('The convolution display uses real kernels.');
    return value[0];
  });
  if (samples.length > 16 || weights.length > 8 || weights.length > samples.length) throw new RangeError('Use at most 16 samples and 8 kernel entries, with kernel no longer than samples.');
  if (typeof circular !== 'boolean') throw new RangeError('Circular selection must be boolean.');
  const outputLength = circular ? samples.length : samples.length + weights.length - 1;
  integer(outputIndex, 'Output index', 0, outputLength - 1);
  const termsAt = output => samples.map((value, index) => {
    const raw = output - index;
    const kernelIndex = circular ? (raw % samples.length + samples.length) % samples.length : raw;
    const weight = kernelIndex >= 0 && kernelIndex < weights.length ? weights[kernelIndex] : 0;
    return {
      index,
      value,
      kernelIndex,
      weight,
      product: value * weight,
      wraps: circular && raw !== kernelIndex && weight !== 0
    };
  });
  return {
    samples,
    kernel: weights,
    circular,
    outputIndex,
    terms: termsAt(outputIndex),
    output: Array.from({
      length: outputLength
    }, (_, index) => termsAt(index).reduce((sum, term) => sum + term.product, 0))
  };
}
export function filterResponse(rate = TAU, phase = Math.PI / 2, initial = 0) {
  bounded(rate, 'Filter decay rate', .5, 20);
  bounded(phase, 'Third-tone phase', -Math.PI, Math.PI);
  bounded(initial, 'Initial output', -3, 3);
  const modes = [[1, 2, 0], [3, 1, phase]].map(([frequency, amplitude, angle]) => ({
    frequency,
    amplitude,
    phase: angle,
    gain: rate / Math.hypot(rate, TAU * frequency),
    lag: -Math.atan2(TAU * frequency, rate)
  }));
  const steadyAt = time => modes.reduce((sum, mode) => sum + mode.amplitude * mode.gain * Math.cos(TAU * mode.frequency * time + mode.phase + mode.lag), 0);
  const correction = initial - steadyAt(0);
  const points = Array.from({
    length: 513
  }, (_, index) => {
    const time = index / 256;
    const steady = steadyAt(time),
      transient = correction * Math.exp(-rate * time);
    return {
      time,
      input: harmonicValue(time, [2, 1], phase),
      steady,
      transient,
      total: steady + transient
    };
  });
  return {
    rate,
    phase,
    initial,
    modes,
    points,
    correction
  };
}

// Integral_0^T exp(-(real+i imaginary)t)dt. The series handles removable limits
// without subtracting nearly equal exponentials. Inputs are bounded by the caller.
export function finiteExponentialIntegral(real, imaginary, horizon) {
  bounded(real, 'Weighted real exponent', -8, 8);
  bounded(imaginary, 'Weighted angular frequency', -8, 8);
  bounded(horizon, 'Horizon', 0, 8);
  const z = [-real * horizon, -imaginary * horizon];
  if (Math.hypot(...z) < .5) {
    let term = [1, 0],
      sum = [1, 0];
    for (let order = 1; order <= 24; order += 1) {
      term = scaleComplex(multiplyComplex(term, z), 1 / (order + 1));
      sum = addComplex(sum, term);
    }
    return scaleComplex(sum, horizon);
  }
  const numerator = [1 - Math.exp(z[0]) * Math.cos(z[1]), -Math.exp(z[0]) * Math.sin(z[1])];
  const denominator = real ** 2 + imaginary ** 2;
  return [(numerator[0] * real + numerator[1] * imaginary) / denominator, (numerator[1] * real - numerator[0] * imaginary) / denominator];
}
export function laplaceRegion(decay = 1, sigma = 0, omega = 1, horizon = 2, side = 'right') {
  bounded(decay, 'Signal decay', -1, 2);
  bounded(sigma, 'Real part sigma', -3, 3);
  bounded(omega, 'Angular frequency omega', -4, 4);
  bounded(horizon, 'Finite horizon', .25, 8);
  if (!['right', 'left'].includes(side)) throw new RangeError('Choose right-sided or left-sided.');
  const real = decay + sigma;
  const converges = side === 'right' ? real > 0 : real < 0;
  const pole = real === 0 && omega === 0;
  const radius = Math.hypot(real, omega);
  const rational = pole ? null : [finiteResult(real / radius / radius), finiteResult(-omega / radius / radius)];
  const at = time => side === 'right' ? finiteExponentialIntegral(real, omega, time) : scaleComplex(finiteExponentialIntegral(-real, -omega, time), -1);
  return {
    decay,
    sigma,
    omega,
    horizon,
    side,
    converges,
    boundary: real === 0,
    pole,
    rational,
    value: at(horizon),
    points: Array.from({
      length: 129
    }, (_, index) => ({
      horizon: horizon * index / 128,
      value: at(horizon * index / 128)
    }))
  };
}
