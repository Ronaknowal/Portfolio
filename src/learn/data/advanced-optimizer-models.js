// Complete bounded teaching rules. Float64 JavaScript arrays mirror the supplied
// NumPy paper variants; specialist package differences remain explicit in prose.
const map = (fn, ...arrays) => Array.isArray(arrays[0]) ? arrays[0].map((_, i) => map(fn, ...arrays.map(a => a[i]))) : fn(...arrays);
const sum = array => array.flat(Infinity).reduce((a, b) => a + b, 0);
const zeros = array => map(() => 0, array);
export function createOptimizer(parameters, method, rate) {
  const state = { step: 0 };
  if (['adamw', 'adamw_cosine', 'lion', 'sophia_g', 'prodigy'].includes(method)) state.momentum = zeros(parameters);
  if (['adamw', 'adamw_cosine', 'schedule_free', 'prodigy'].includes(method)) state.second = zeros(parameters);
  if (method === 'sophia_g') state.curvature = zeros(parameters);
  if (method === 'prodigy') Object.assign(state, { initial: structuredClone(parameters), displacement_sum: zeros(parameters), numerator: 0, distance: 1e-6 });
  if (method === 'schedule_free') Object.assign(state, { average: structuredClone(parameters), fast: structuredClone(parameters), weight_sum: 0 });
  return { method, rate, parameters: structuredClone(parameters), state };
}
export function optimizerStep(snapshot, gradient, curvature = null, decay = 0) {
  const result = structuredClone(snapshot), s = result.state;
  const step = ++s.step;
  let rate = result.rate;
  const diagnostics = {};
  if (['adamw', 'adamw_cosine'].includes(result.method)) {
    if (result.method === 'adamw_cosine') rate *= Math.min(1, step / 20) * (1 + Math.cos(Math.PI * Math.min(Math.max(0, (step - 20) / 380), 1))) / 2;
    s.momentum = map((m, g) => .9 * m + .1 * g, s.momentum, gradient);
    s.second = map((v, g) => .999 * v + .001 * g * g, s.second, gradient);
    result.parameters = map((p, m, v) => p * (1 - rate * decay) - rate * (m / (1 - .9 ** step)) / (Math.sqrt(v / (1 - .999 ** step)) + 1e-8), result.parameters, s.momentum, s.second);
  } else if (result.method === 'lion') {
    result.parameters = map((p, m, g) => p * (1 - rate * decay) - rate * Math.sign(.9 * m + .1 * g), result.parameters, s.momentum, gradient);
    s.momentum = map((m, g) => .99 * m + .01 * g, s.momentum, gradient);
  } else if (result.method === 'sophia_g') {
    if (curvature) s.curvature = map((h, estimate) => .99 * h + .01 * estimate, s.curvature, curvature);
    s.momentum = map((m, g) => .965 * m + .035 * g, s.momentum, gradient);
    const ratio = map((m, h) => m / Math.max(.04 * h, 1e-12), s.momentum, s.curvature);
    diagnostics.clipped_fraction = ratio.flat(Infinity).filter(x => Math.abs(x) > 1).length / ratio.flat(Infinity).length;
    result.parameters = map((p, r) => p * (1 - rate * decay) - rate * Math.max(-1, Math.min(1, r)), result.parameters, ratio);
  } else if (result.method === 'prodigy') {
    if (decay !== 0) throw new RangeError('This paper Prodigy rule has no weight decay.');
    const d = s.distance, beta = Math.sqrt(.999), weight = (1 - beta) * rate * d ** 2;
    s.numerator = beta * s.numerator + weight * sum(map((g, initial, current) => g * (initial - current), gradient, s.initial, result.parameters));
    s.displacement_sum = map((old, g) => beta * old + weight * g, s.displacement_sum, gradient);
    s.momentum = map((m, g) => .9 * m + .1 * d * g, s.momentum, gradient);
    s.second = map((v, g) => .999 * v + .001 * d ** 2 * g ** 2, s.second, gradient);
    const denominator = sum(map(Math.abs, s.displacement_sum));
    s.distance = Math.max(d, denominator > 0 ? s.numerator / denominator : d);
    result.parameters = map((p, m, v) => p - rate * d * m / (Math.sqrt(v) + d * 1e-8), result.parameters, s.momentum, s.second);
    Object.assign(diagnostics, { distance_used: d, distance_next: s.distance });
  } else if (result.method === 'schedule_free') {
    rate *= Math.min(1, step / 20);
    s.second = map((v, g) => .999 * v + .001 * g ** 2, s.second, gradient);
    s.fast = map((z, v, g, p) => z - rate * (g / (Math.sqrt(v / (1 - .999 ** step)) + 1e-8) + decay * p), s.fast, s.second, gradient, result.parameters);
    s.weight_sum += rate ** 2;
    const coefficient = rate ** 2 / s.weight_sum;
    s.average = map((x, z) => x + coefficient * (z - x), s.average, s.fast);
    result.parameters = map((x, z) => .9 * x + .1 * z, s.average, s.fast);
    diagnostics.averaging_coefficient = coefficient;
  } else throw new RangeError('Unknown optimizer.');
  diagnostics.rate = rate;
  return { ...result, diagnostics };
}
export const evaluationParameters = snapshot => snapshot.method === 'schedule_free' ? snapshot.state.average : snapshot.parameters;
export function digitProbabilities(pixels, parameters) {
  const features = [...pixels.map(p => p / 16), 1];
  const scores = parameters[0].map((_, j) => features.reduce((total, x, i) => total + x * parameters[i][j], 0));
  const maximum = Math.max(...scores), exponential = scores.map(v => Math.exp(v - maximum)), total = sum(exponential);
  return { features, scores, probabilities: exponential.map(v => v / total) };
}
export function digitUpdate(snapshot, pixels, target) {
  const before = digitProbabilities(pixels, evaluationParameters(snapshot));
  const training = digitProbabilities(pixels, snapshot.parameters);
  const gradient = training.features.map(x => training.probabilities.map((p, j) => x * (p - Number(j === target))));
  const curvature = snapshot.method === 'sophia_g' ? training.features.map(x => training.probabilities.map(p => x ** 2 * p * (1 - p))) : null;
  const next = optimizerStep(snapshot, gradient, curvature);
  const after = digitProbabilities(pixels, evaluationParameters(next));
  return { before, training, gradient, curvature, next, after, delta: after.probabilities.map((v, i) => v - before.probabilities[i]) };
}
export function lionTrace({ initial = -.4, momentum = -.3, gradients = [2, 2, 2], rate = .02, decay = .1, beta1 = .9, beta2 = .99 }) {
  let parameter = initial, memory = momentum;
  return gradients.map((gradient, i) => {
    const blend = beta1 * memory + (1 - beta1) * gradient;
    const row = { step: i + 1, parameter, memory, gradient, historyPart: beta1 * memory, freshPart: (1 - beta1) * gradient, blend, direction: Math.sign(blend), shrunk: parameter * (1 - rate * decay) };
    parameter = row.shrunk - rate * row.direction;
    memory = beta2 * memory + (1 - beta2) * gradient;
    return { ...row, next: parameter, nextMemory: memory };
  });
}
export function sampledCurvature(inputs, probabilities, labels) {
  const outcomes = [0, 1, 2, 3].map(code => {
    const sampled = [Math.floor(code / 2), code % 2];
    const probability = sampled.reduce((mass, y, i) => mass * (y ? probabilities[i] : 1 - probabilities[i]), 1);
    const gradient = inputs.reduce((total, x, i) => total + x * (probabilities[i] - sampled[i]), 0) / 2;
    const estimate = 2 * gradient ** 2;
    return { sampled, probability, gradient, estimate, contribution: probability * estimate };
  });
  return { outcomes, expectation: outcomes.reduce((s, row) => s + row.contribution, 0), exact: inputs.reduce((s, x, i) => s + x ** 2 * probabilities[i] * (1 - probabilities[i]), 0) / 2, trueSquare: (inputs.reduce((s, x, i) => s + x * (probabilities[i] - labels[i]), 0) / 2) ** 2 };
}
export function hessianProbes(a, b, c) {
  const values = [[1, 1], [1, -1], [-1, 1], [-1, -1]].map(u => ({ u, estimate: [u[0] * (a * u[0] + b * u[1]), u[1] * (b * u[0] + c * u[1])] }));
  const spread = Math.hypot(a - c, 2 * b);
  return { values, mean: [a, c], eigenvalues: [(a + c - spread) / 2, (a + c + spread) / 2] };
}
export function prodigyTrace(initial = 1, target = -2, distance = .01, rate = .3, steps = 12) {
  let state = createOptimizer([initial], 'prodigy', rate); state.state.distance = distance;
  const rows = []; let stopped = null;
  for (let i = 0; i < steps; i++) {
    const before = state.parameters[0], gradient = before - target;
    state = optimizerStep(state, [gradient]);
    const after = state.parameters[0];
    if (!Number.isFinite(after) || !Number.isFinite(state.state.distance)) { stopped = i + 1; break; }
    rows.push({ step: i + 1, before, gradient, after, loss: .5 * (after - target) ** 2, numerator: state.state.numerator, denominator: Math.abs(state.state.displacement_sum[0]), momentum: state.state.momentum[0], second: state.state.second[0], ...state.diagnostics });
  }
  return { rows, stopped };
}
export function scheduleFreeTrace(initial = -1, target = 1, noise = [-.5, .5, 1, -1], beta = .9, rate = .2) {
  let average = initial, fast = initial;
  return noise.map((perturbation, i) => {
    const oldAverage = average, oldFast = fast, training = beta * average + (1 - beta) * fast, gradient = training - target + perturbation;
    fast -= rate * gradient; average += (fast - average) / (i + 1);
    return { step: i + 1, oldAverage, oldFast, training, gradient, fast, average, nextTraining: beta * average + (1 - beta) * fast, coefficient: 1 / (i + 1), loss: .5 * (average - target) ** 2 };
  });
}
export function bowlGeometry(rotated = true) {
  const angle = rotated ? Math.PI / 6 : 0, c = Math.cos(angle), s = Math.sin(angle);
  const H = [[c * c + 20 * s * s, -19 * c * s], [-19 * c * s, s * s + 20 * c * c]];
  const point = [2, -1], gradient = H.map(row => row[0] * point[0] + row[1] * point[1]);
  return { H, point, gradient, descent: point.map((p, i) => p - .08 * gradient[i]), diagonal: point.map((p, i) => p - gradient[i] / H[i][i]), newton: [0, 0], angle };
}
