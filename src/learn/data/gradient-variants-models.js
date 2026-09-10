// Bounded, deterministic teaching investigations. Synthetic gradient replay is
// an arithmetic comparison, not an optimizer benchmark on a shared objective.
export const optimizerMeasurements = Object.freeze([-3, -1, 1, 3]);
function finite(value, minimum, maximum, name) {
  if (!Number.isFinite(value) || value < minimum || value > maximum) {
    throw new Error(`${name} must be a finite number from ${minimum} through ${maximum}.`);
  }
}
function integer(value, minimum, maximum, name) {
  finite(value, minimum, maximum, name);
  if (!Number.isInteger(value)) throw new Error(`${name} must be a whole number.`);
}
function choice(value, allowed, name) {
  if (!allowed.includes(value)) throw new Error(`Choose a valid ${name}.`);
}
function sum(values) {
  return values.reduce((total, value) => total + value, 0);
}
function norm(values) {
  return Math.hypot(...values);
}
function freeze(value) {
  if (Array.isArray(value)) return Object.freeze(value.map(freeze));
  if (value && typeof value === 'object') {
    return Object.freeze(Object.fromEntries(Object.entries(value).map(([key, item]) => [key, freeze(item)])));
  }
  return value;
}
export function formatOptimizerNumber(value) {
  if (value === null) return 'undefined';
  if (value === 0) return '0';
  if (!Number.isFinite(value)) return String(value);
  if (Math.abs(value) < 0.0001 || Math.abs(value) >= 10000) return value.toExponential(3);
  return String(Number(value.toFixed(5)));
}
function subsetsOfSize(size) {
  return Array.from({
    length: 16
  }, (_, mask) => Array.from({
    length: 4
  }, (_, index) => index).filter(index => mask & 1 << index)).filter(indices => indices.length === size);
}
export function batchGradientState(theta = 1, selected = [3], rate = 0.2, reduction = 'mean') {
  finite(theta, -4, 4, 'Parameter');
  finite(rate, 0, 0.8, 'Learning rate');
  choice(reduction, ['mean', 'sum'], 'reduction');
  if (!Array.isArray(selected) || selected.length < 1 || selected.length > 4 || new Set(selected).size !== selected.length || selected.some(index => !Number.isInteger(index) || index < 0 || index > 3)) {
    throw new Error('Select one through four distinct observations.');
  }
  const gradients = optimizerMeasurements.map(value => theta - value);
  const meanGradient = sum(selected.map(index => gradients[index])) / selected.length;
  const usedGradient = reduction === 'mean' ? meanGradient : meanGradient * selected.length;
  const next = theta - rate * usedGradient;
  const possibleMeans = subsetsOfSize(selected.length).map(indices => sum(indices.map(index => gradients[index])) / indices.length);
  const expectedMean = sum(possibleMeans) / possibleMeans.length;
  const variance = sum(possibleMeans.map(value => (value - expectedMean) ** 2)) / possibleMeans.length;
  return freeze({
    theta,
    selected,
    rate,
    reduction,
    measurements: optimizerMeasurements,
    gradients,
    meanGradient,
    usedGradient,
    fullGradient: theta,
    next,
    fullLoss: (theta ** 2 + 5) / 2,
    nextFullLoss: (next ** 2 + 5) / 2,
    possibleMeans,
    expectedMean,
    variance,
    batchSize: selected.length
  });
}
export function momentumTrajectoryState(method = 'momentum', rate = 0.08, beta = 0.8, steps = 12, curvature = 20) {
  choice(method, ['sgd', 'momentum', 'nesterov'], 'trajectory method');
  finite(rate, 0.005, 0.25, 'Learning rate');
  finite(beta, 0, 0.95, 'Momentum');
  integer(steps, 0, 24, 'Steps');
  finite(curvature, 1, 30, 'Vertical curvature');
  const loss = point => (point[0] ** 2 + curvature * point[1] ** 2) / 2;
  const derivative = point => [point[0], curvature * point[1]];
  let theta = [2, 1];
  let velocity = [0, 0];
  const frames = [{
    step: 0,
    theta,
    loss: loss(theta),
    velocity,
    gradient: derivative(theta),
    evaluationPoint: theta,
    displacement: [0, 0],
    before: theta
  }];
  let truncated = false;
  for (let step = 1; step <= steps; step++) {
    const before = [...theta];
    const oldVelocity = [...velocity];
    const evaluationPoint = method === 'nesterov' ? theta.map((value, index) => value - rate * beta * velocity[index]) : [...theta];
    const gradient = derivative(evaluationPoint);
    velocity = gradient.map((value, index) => method === 'sgd' ? value : beta * velocity[index] + value);
    const displacement = velocity.map(value => -rate * value);
    const proposed = theta.map((value, index) => value + displacement[index]);
    if (proposed.some(value => !Number.isFinite(value) || Math.abs(value) > 100000)) {
      truncated = true;
      break;
    }
    theta = proposed;
    frames.push({
      step,
      theta,
      loss: loss(theta),
      before,
      oldVelocity,
      velocity,
      gradient,
      currentGradient: derivative(before),
      evaluationPoint,
      displacement
    });
  }
  return freeze({
    method,
    rate,
    beta,
    steps,
    curvature,
    frames,
    truncated,
    bound: 100000,
    initial: [2, 1],
    final: frames.at(-1)
  });
}
export const optimizerGradientProfiles = freeze({
  constant: {
    title: 'Constant, unequal coordinates',
    gradients: Array.from({
      length: 8
    }, () => [2, 0.2])
  },
  alternating: {
    title: 'Alternating direction',
    gradients: Array.from({
      length: 8
    }, (_, index) => [index % 2 ? -2 : 2, 0.2])
  },
  sparse: {
    title: 'An occasionally active coordinate',
    gradients: [[2, 0], [2, 0], [2, 4], [2, 0], [2, 0], [-2, 0], [2, 1], [0, 0]]
  },
  spike: {
    title: 'One large gradient, then small ones',
    gradients: [[0.2, 0.2], [0.2, 0.2], [6, 0.2], [0.2, 0.2], [0.2, 0.2], [0.2, 0.2], [0.2, 0.2], [0.2, 0.2]]
  }
});
function adaptiveFrame(method, gradient, state, rate, beta1, beta2, epsilon, correction) {
  const step = state.step + 1;
  const first = method === 'adam' ? gradient.map((value, index) => beta1 * state.first[index] + (1 - beta1) * value) : [...state.first];
  const second = gradient.map((value, index) => method === 'adagrad' ? state.second[index] + value ** 2 : beta2 * state.second[index] + (1 - beta2) * value ** 2);
  const numerator = method === 'adam' ? first.map(value => value / (correction ? 1 - beta1 ** step : 1)) : [...gradient];
  const scaledSecond = method === 'adam' && correction ? second.map(value => value / (1 - beta2 ** step)) : [...second];
  const denominator = scaledSecond.map(value => Math.sqrt(value) + epsilon);
  const direction = numerator.map((value, index) => value / denominator[index]);
  const displacement = direction.map(value => -rate * value);
  return {
    step,
    gradient,
    first,
    second,
    numerator,
    scaledSecond,
    denominator,
    direction,
    displacement
  };
}
export function adaptiveHistoryState(method = 'adam', profile = 'sparse', rate = 0.1, beta1 = 0.9, beta2 = 0.9, epsilon = 0.000001, correction = true) {
  choice(method, ['adagrad', 'rmsprop', 'adam'], 'adaptive method');
  choice(profile, Object.keys(optimizerGradientProfiles), 'gradient history');
  finite(rate, 0.001, 0.5, 'Learning rate');
  finite(beta1, 0, 0.99, 'First-moment decay');
  finite(beta2, 0, 0.999, 'Second-moment decay');
  finite(epsilon, 1e-8, 1, 'Epsilon');
  if (typeof correction !== 'boolean') throw new Error('Bias correction must be on or off.');
  let state = {
    step: 0,
    first: [0, 0],
    second: [0, 0]
  };
  const frames = [];
  for (const gradient of optimizerGradientProfiles[profile].gradients) {
    state = adaptiveFrame(method, gradient, state, rate, beta1, beta2, epsilon, correction);
    frames.push(state);
  }
  return freeze({
    method,
    profile,
    rate,
    beta1,
    beta2,
    epsilon,
    correction,
    gradients: optimizerGradientProfiles[profile].gradients,
    frames
  });
}
export const optimizerDecayPresets = freeze({
  zero: {
    title: 'Zero current data gradient',
    theta: [2, 10],
    gradient: [0, 0]
  },
  unequal: {
    title: 'Unequal data gradients',
    theta: [2, 2],
    gradient: [1, 10]
  },
  opposed: {
    title: 'Opposed gradient and parameter',
    theta: [2, -2],
    gradient: [-1, 1]
  }
});
export function decayComparisonState(preset = 'zero', rate = 0.1, decay = 0.1, steps = 1) {
  choice(preset, Object.keys(optimizerDecayPresets), 'decay preset');
  finite(rate, 0.01, 0.3, 'Learning rate');
  finite(decay, 0, 1, 'Decay coefficient');
  integer(steps, 1, 8, 'Steps');
  const initial = optimizerDecayPresets[preset];
  const methods = ['coupled', 'adamw'].map(method => {
    let theta = [...initial.theta];
    let state = {
      step: 0,
      first: [0, 0],
      second: [0, 0]
    };
    const frames = [];
    for (let step = 1; step <= steps; step++) {
      const before = [...theta];
      const gradient = initial.gradient.map((value, index) => method === 'coupled' ? value + decay * theta[index] : value);
      state = adaptiveFrame('adam', gradient, state, rate, 0.9, 0.99, 1e-8, true);
      const shrinkage = theta.map(value => method === 'adamw' ? -rate * decay * value : 0);
      const displacement = state.displacement.map((value, index) => value + shrinkage[index]);
      theta = theta.map((value, index) => value + displacement[index]);
      frames.push({
        ...state,
        before,
        theta,
        shrinkage,
        displacement
      });
    }
    return {
      method,
      theta,
      frames,
      final: frames.at(-1)
    };
  });
  return freeze({
    preset,
    rate,
    decay,
    steps,
    initial,
    methods
  });
}
export function layerScaleState(method = 'lars', smallScale = 0.1, rate = 0.1, decay = 0, trustCoefficient = 0.1, preset = 'ordinary') {
  choice(method, ['sgd', 'lars', 'lamb'], 'layer method');
  choice(preset, ['ordinary', 'zeroWeight', 'zeroGradient'], 'layer case');
  finite(smallScale, 0.02, 2, 'Second-block scale');
  finite(rate, 0.001, 0.5, 'Learning rate');
  finite(decay, 0, 1, 'Decay coefficient');
  finite(trustCoefficient, 0.001, 1, 'LARS trust coefficient');
  const weights = [[3, 4], preset === 'zeroWeight' ? [0, 0] : [3 * smallScale, 4 * smallScale]];
  const blocks = weights.map((theta, blockIndex) => {
    const gradient = preset === 'zeroGradient' && blockIndex === 1 ? [0, 0] : [0.6, 0.8];
    // First Adam step with zero initial moments: corrected moments equal g and g².
    const adaptive = gradient.map(value => value / (Math.abs(value) + 1e-8));
    const base = method === 'lamb' ? adaptive : gradient;
    const direction = base.map((value, index) => value + decay * theta[index]);
    const weightNorm = norm(theta);
    const directionNorm = norm(direction);
    const denominator = method === 'lars' ? norm(gradient) + decay * weightNorm : directionNorm;
    const fallback = method !== 'sgd' && (weightNorm === 0 || denominator === 0);
    const ratio = method === 'sgd' || fallback ? 1 : (method === 'lars' ? trustCoefficient : 1) * weightNorm / denominator;
    const displacement = direction.map(value => -rate * ratio * value);
    const updateNorm = norm(displacement);
    return {
      theta,
      gradient,
      adaptive,
      direction,
      weightNorm,
      directionNorm,
      denominator,
      ratio,
      fallback,
      displacement,
      updateNorm,
      relativeUpdate: weightNorm === 0 ? null : updateNorm / weightNorm,
      next: theta.map((value, index) => value + displacement[index])
    };
  });
  return freeze({
    method,
    smallScale,
    rate,
    decay,
    trustCoefficient,
    preset,
    blocks
  });
}
