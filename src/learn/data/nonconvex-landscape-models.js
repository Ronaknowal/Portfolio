function bounded(value, low, high, name) {
  if (!Number.isFinite(value) || value < low || value > high) {
    throw new RangeError(`${name} must be a finite number from ${low} to ${high}.`);
  }
}
export function landscapeNumber(value) {
  if (!Number.isFinite(value)) throw new RangeError('Displayed values must be finite.');
  if (value === 0) return '0';
  if (Math.abs(value) < 0.0001 || Math.abs(value) >= 10000) return value.toExponential(3);
  return Number(value.toFixed(5)).toString();
}
export function wellValue(x, tilt) {
  return (x * x - 1) ** 2 / 4 + tilt * x;
}
export function wellGradient(x, tilt) {
  return x ** 3 - x + tilt;
}
export function wellState(tilt = 0.15, initial = 0.8, rate = 0.12, steps = 60) {
  bounded(tilt, -0.25, 0.25, 'Tilt');
  bounded(initial, -1.5, 1.5, 'Initial x');
  bounded(rate, 0, 0.3, 'Learning rate');
  if (!Number.isInteger(steps) || steps < 0 || steps > 80) throw new RangeError('Steps must be an integer from 0 to 80.');
  const turning = 1 / Math.sqrt(3);
  const intervals = [[-1.5, -turning], [-turning, turning], [turning, 1.5]];
  const critical = intervals.map(([left, right], index) => {
    let lo = left;
    let hi = right;
    for (let iteration = 0; iteration < 60; iteration += 1) {
      const middle = (lo + hi) / 2;
      if (wellGradient(lo, tilt) * wellGradient(middle, tilt) <= 0) hi = middle;else lo = middle;
    }
    const x = (lo + hi) / 2;
    return {
      x,
      value: wellValue(x, tilt),
      curvature: 3 * x * x - 1,
      kind: index === 1 ? 'local maximum' : 'local minimum'
    };
  });
  const best = Math.min(critical[0].value, critical[2].value);
  const frames = [];
  let x = initial;
  for (let step = 0; step <= steps; step += 1) {
    frames.push({
      step,
      x,
      value: wellValue(x, tilt),
      gradient: wellGradient(x, tilt)
    });
    if (step < steps) x -= rate * wellGradient(x, tilt);
    if (!Number.isFinite(x) || Math.abs(x) > 2) throw new RangeError('Trajectory left the declared plotting domain.');
  }
  return {
    tilt,
    initial,
    rate,
    critical,
    best,
    frames
  };
}
export const stationaryPresets = {
  bowl: {
    label: 'Quadratic bowl: x² + y²',
    eigenvalues: [2, 2],
    classification: 'Strict global minimum',
    powers: [2, 2],
    signs: [1, 1],
    reason: 'Every nonzero point has positive value. The positive-definite Hessian already proves a strict local minimum.'
  },
  cap: {
    label: 'Quadratic cap: −x² − y²',
    eigenvalues: [-2, -2],
    classification: 'Strict global maximum',
    powers: [2, 2],
    signs: [-1, -1],
    reason: 'Every nonzero point has negative value. Both curvature directions point downward.'
  },
  saddle: {
    label: 'Quadratic saddle: x² − y²',
    eigenvalues: [2, -2],
    classification: 'Saddle with negative curvature',
    powers: [2, 2],
    signs: [1, -1],
    reason: 'The x-axis increases the value; the y-axis decreases it. The Hessian is indefinite.'
  },
  flatMinimum: {
    label: 'Quartic bowl: x⁴ + y⁴',
    eigenvalues: [0, 0],
    classification: 'Strict global minimum',
    powers: [4, 4],
    signs: [1, 1],
    reason: 'The Hessian is zero, but the fourth powers are positive away from the origin. The second-order test is inconclusive; the exact function settles it.'
  },
  flatSaddle: {
    label: 'Quartic saddle: x⁴ − y⁴',
    eigenvalues: [0, 0],
    classification: 'Degenerate saddle',
    powers: [4, 4],
    signs: [1, -1],
    reason: 'The same zero Hessian now hides a decreasing y-axis. No negative eigenvalue at the origin is available to detect it.'
  },
  valley: {
    label: 'Flat valley: x²',
    eigenvalues: [2, 0],
    classification: 'Non-strict global minimum',
    powers: [2, 2],
    signs: [1, 0],
    reason: 'Every point on x=0 has value zero. The origin is a minimum but is not isolated or strict.'
  }
};
export function stationaryValue(kind, x, y) {
  if (!Object.hasOwn(stationaryPresets, kind)) throw new RangeError('Choose a listed stationary geometry.');
  const preset = stationaryPresets[kind];
  return preset.signs[0] * x ** preset.powers[0] + preset.signs[1] * y ** preset.powers[1];
}
export function stationaryState(kind = 'flatSaddle', degrees = 90, radius = 0.5) {
  if (!Object.hasOwn(stationaryPresets, kind)) throw new RangeError('Choose a listed stationary geometry.');
  bounded(degrees, 0, 360, 'Direction angle');
  bounded(radius, 0, 1, 'Radius');
  const radians = degrees * Math.PI / 180;
  // Preserve exact shared components on the axes and diagonals. Independent
  // sine/cosine rounding must not turn the neutral x=y slice into a slope.
  const diagonal = Math.SQRT1_2;
  const octants = [[1, 0], [diagonal, diagonal], [0, 1], [-diagonal, diagonal], [-1, 0], [-diagonal, -diagonal], [0, -1], [diagonal, -diagonal], [1, 0]];
  const direction = degrees % 45 === 0 ? octants[degrees / 45] : [Math.cos(radians), Math.sin(radians)];
  const point = direction.map(value => radius * value);
  const eigenvalues = stationaryPresets[kind].eigenvalues;
  const quadratic = 0.5 * point.reduce((sum, value, index) => sum + eigenvalues[index] * value * value, 0);
  return {
    kind,
    degrees,
    radius,
    direction,
    point,
    quadratic,
    actual: stationaryValue(kind, ...point),
    ...stationaryPresets[kind]
  };
}

// A disclosed possible sequence of independent ±1 samples, held fixed for comparisons.
// It is a controlled realization, not an estimate of an escape probability.
export const saddleSigns = [1, -1, -1, 1, 1, 1, -1, 1, -1, -1, 1, -1, 1, 1, -1, -1, -1, 1, 1, -1, 1, -1, -1, 1];
export function saddleNoiseState(direction = 'none', rate = 0.12, amplitude = 0.15, initialY = 0) {
  const vectors = {
    none: [0, 0],
    stable: [1, 0],
    unstable: [0, 1],
    both: [Math.SQRT1_2, Math.SQRT1_2]
  };
  if (!Object.hasOwn(vectors, direction)) throw new RangeError('Choose a listed noise direction.');
  bounded(rate, 0.02, 0.25, 'Learning rate');
  bounded(amplitude, 0, 0.5, 'Noise amplitude');
  bounded(initialY, -0.05, 0.05, 'Initial y');
  const vector = vectors[direction];
  const frames = [];
  let point = [0.6, initialY];
  let stopped = null;
  for (let step = 0; step <= saddleSigns.length; step += 1) {
    const sign = step === saddleSigns.length ? null : saddleSigns[step];
    const trueGradient = [2 * point[0], -2 * point[1]];
    const noise = vector.map(value => sign === null ? 0 : sign * amplitude * value);
    const sampleGradient = trueGradient.map((value, index) => value + noise[index]);
    frames.push({
      step,
      point: [...point],
      value: point[0] ** 2 - point[1] ** 2,
      sign,
      trueGradient,
      noise,
      sampleGradient
    });
    if (sign === null) break;
    const next = point.map((value, index) => value - rate * sampleGradient[index]);
    if (next.some(value => Math.abs(value) > 4)) {
      stopped = {
        attemptedStep: step + 1,
        point: next
      };
      break;
    }
    point = next;
  }
  return {
    direction,
    rate,
    amplitude,
    initialY,
    vector,
    frames,
    stopped,
    stableFactor: 1 - 2 * rate,
    unstableFactor: 1 + 2 * rate
  };
}
export function factorValue(a, b) {
  return 0.5 * (a * b - 1) ** 2;
}
export function symmetryState(logScale = 0, displacement = 0.15, direction = 'normal') {
  bounded(logScale, -3, 3, 'Base-2 scale exponent');
  bounded(displacement, -0.3, 0.3, 'Displacement');
  if (!['normal', 'tangent'].includes(direction)) throw new RangeError('Choose normal or tangent.');
  const a = 2 ** logScale;
  const b = 1 / a;
  const norm = Math.hypot(a, b);
  const unit = direction === 'normal' ? [b / norm, a / norm] : [a / norm, -b / norm];
  const perturbed = [a + displacement * unit[0], b + displacement * unit[1]];
  return {
    logScale,
    displacement,
    direction,
    a,
    b,
    unit,
    perturbed,
    coefficient: a * b,
    perturbedCoefficient: perturbed[0] * perturbed[1],
    eigenvalues: [0, a * a + b * b],
    loss: factorValue(...perturbed),
    quadratic: direction === 'normal' ? 0.5 * (a * a + b * b) * displacement ** 2 : 0,
    functionPerturbationLoss: 0.5 * displacement ** 2
  };
}
export function modePaths(t) {
  bounded(t, 0, 1, 'Path fraction');
  const straight = [1 + t, 1 - t / 2];
  const curved = [1 + t, 1 / (1 + t)];
  const opposite = [1 - 2 * t, 1 - 2 * t];
  return {
    t,
    straight,
    curved,
    opposite,
    straightLoss: factorValue(...straight),
    curvedLoss: factorValue(...curved),
    oppositeLoss: factorValue(...opposite)
  };
}
export function interpolationValue(parameter, input) {
  return input + parameter * (input * input - 1);
}
