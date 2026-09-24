// Finite teaching models. Rates are bits; distortion units are declared by each fixture.
const LOG_TWO = Math.log(2);
function bounded(value, minimum, maximum, label) {
  if (!Number.isFinite(value) || value < minimum || value > maximum) {
    throw new RangeError(`${label} must be finite and in [${minimum}, ${maximum}].`);
  }
  return value;
}
function integer(value, minimum, maximum, label) {
  bounded(value, minimum, maximum, label);
  if (!Number.isInteger(value)) throw new RangeError(`${label} must be an integer.`);
  return value;
}
function probabilityVector(values, label) {
  if (!Array.isArray(values) || values.length < 1 || values.length > 6) {
    throw new RangeError(`${label} needs one to six entries.`);
  }
  values.forEach(value => bounded(value, 0, 1, label));
  if (Math.abs(values.reduce((sum, value) => sum + value, 0) - 1) > 1e-12) {
    throw new RangeError(`${label} must sum to one.`);
  }
}
function freeze(value) {
  if (value && typeof value === 'object' && !Object.isFrozen(value)) {
    Object.values(value).forEach(freeze);
    Object.freeze(value);
  }
  return value;
}
function logSumExp(values) {
  const maximum = Math.max(...values);
  if (maximum === -Infinity) return -Infinity;
  return maximum + Math.log(values.reduce((sum, value) => sum + Math.exp(value - maximum), 0));
}
export function binaryEntropy(probability) {
  bounded(probability, 0, 1, 'Probability');
  if (probability === 0 || probability === 1) return 0;
  return -(probability * Math.log2(probability) + (1 - probability) * Math.log2(1 - probability));
}
export function binaryRateDistortion(probability, distortion) {
  bounded(probability, 0, 1, 'Source probability');
  bounded(distortion, 0, 1, 'Distortion budget');
  const zeroRateDistortion = Math.min(probability, 1 - probability);
  return distortion >= zeroRateDistortion ? 0 : binaryEntropy(probability) - binaryEntropy(distortion);
}
export function binaryOptimalChannel(probability = 0.2, budget = 0.1) {
  bounded(probability, 0, 1, 'Source probability');
  bounded(budget, 0, 1, 'Distortion budget');
  const threshold = Math.min(probability, 1 - probability);
  const distortion = Math.min(budget, threshold);
  const reproductionOne = budget >= threshold ? probability > 0.5 ? 1 : 0 : (probability - distortion) / (1 - 2 * distortion);
  const joint = budget >= threshold ? [[(1 - probability) * (1 - reproductionOne), (1 - probability) * reproductionOne], [probability * (1 - reproductionOne), probability * reproductionOne]] : [[(1 - reproductionOne) * (1 - distortion), reproductionOne * distortion], [(1 - reproductionOne) * distortion, reproductionOne * (1 - distortion)]];
  const source = [1 - probability, probability];
  const conditional = joint.map((row, index) => source[index] === 0 ? null : row.map(mass => mass / source[index]));
  const output = [joint[0][0] + joint[1][0], joint[0][1] + joint[1][1]];
  let information = 0;
  joint.forEach((row, i) => row.forEach((mass, j) => {
    if (mass > 0) information += mass * (Math.log2(mass) - Math.log2(source[i]) - Math.log2(output[j]));
  }));
  return freeze({
    probability,
    budget,
    distortion,
    threshold,
    reproductionOne,
    source,
    joint,
    conditional,
    output,
    information,
    rate: binaryRateDistortion(probability, budget)
  });
}
export const BINARY_CODEBOOKS = freeze({
  constant: {
    label: 'One reconstruction: 000',
    words: [0]
  },
  majority: {
    label: 'Two reconstructions: 000 / 111',
    words: [0, 7]
  },
  parity: {
    label: 'Four even-parity reconstructions',
    words: [0, 3, 5, 6]
  },
  lossless: {
    label: 'All eight reconstructions',
    words: [0, 1, 2, 3, 4, 5, 6, 7]
  }
});
function hamming(first, second) {
  let count = 0;
  for (let position = 0; position < 3; position += 1) count += (first >> position & 1) !== (second >> position & 1) ? 1 : 0;
  return count;
}
export function binaryBlockCode({
  probability = 0.5,
  codewords = [0, 7],
  selected = 3
} = {}) {
  bounded(probability, 0, 1, 'Source probability');
  integer(selected, 0, 7, 'Selected input');
  if (!Array.isArray(codewords) || codewords.length === 0 || codewords.length > 8 || new Set(codewords).size !== codewords.length) {
    throw new RangeError('A codebook needs one to eight distinct three-bit words.');
  }
  codewords.forEach(word => integer(word, 0, 7, 'Codeword'));
  const rows = Array.from({
    length: 8
  }, (_, input) => {
    const ones = hamming(input, 0);
    const mass = probability ** ones * (1 - probability) ** (3 - ones);
    let index = 0;
    for (let candidate = 1; candidate < codewords.length; candidate += 1) {
      if (hamming(input, codewords[candidate]) < hamming(input, codewords[index])) index = candidate;
    }
    const reconstruction = codewords[index];
    return {
      input,
      inputBits: input.toString(2).padStart(3, '0'),
      index,
      reconstruction,
      reconstructionBits: reconstruction.toString(2).padStart(3, '0'),
      errors: hamming(input, reconstruction),
      mass
    };
  });
  const bitsPerBlock = Math.ceil(Math.log2(codewords.length));
  const distortion = rows.reduce((sum, row) => sum + row.mass * row.errors / 3, 0);
  const outputMass = codewords.map((_, index) => rows.reduce((sum, row) => sum + (row.index === index ? row.mass : 0), 0));
  const indexEntropy = -outputMass.reduce((sum, mass) => sum + (mass ? mass * Math.log2(mass) : 0), 0);
  return freeze({
    probability,
    codewords: [...codewords],
    selected: rows[selected],
    rows,
    bitsPerBlock,
    rate: bitsPerBlock / 3,
    indexEntropy,
    distortion,
    worstDistortion: Math.max(...rows.filter(row => row.mass > 0).map(row => row.errors / 3)),
    outputMass,
    lowerBound: binaryRateDistortion(probability, distortion)
  });
}
export const RATE_DISTORTION_SCENARIOS = freeze({
  binary: {
    label: 'Fair bits / Hamming error',
    labels: ['0', '1'],
    source: [0.5, 0.5],
    costs: [[0, 1], [1, 0]],
    unit: 'wrong-bit probability'
  },
  levels: {
    label: 'Three levels / squared error',
    labels: ['0', '1', '3'],
    source: [0.7, 0.2, 0.1],
    costs: [[0, 1, 9], [1, 0, 4], [9, 4, 0]],
    unit: 'squared level units'
  },
  alarm: {
    label: 'Same source / costly missed high level',
    labels: ['0', '1', '3'],
    source: [0.7, 0.2, 0.1],
    costs: [[0, 1, 9], [1, 0, 4], [90, 40, 0]],
    unit: 'declared weighted cost'
  }
});

// The lower bound is f(r) - log(max_j g_j), in nats before conversion.
// Jensen gives f(r*) - f(r) >= -log(sum_j r*_j g_j) >= -log(max_j g_j).
// It includes missing output columns, so a stuck zero-support run cannot hide its gap.
export function finiteRateDistortion({
  source = [0.5, 0.5],
  costs = [[0, 1], [1, 0]],
  lambda = 2,
  initial = null,
  iterations = 120,
  tolerance = 1e-9,
  traceLimit = 120
} = {}) {
  probabilityVector(source, 'Source probabilities');
  bounded(lambda, 0, 1000, 'Distortion weight');
  integer(iterations, 0, 5000, 'Iteration count');
  integer(traceLimit, 0, 120, 'Trace length');
  bounded(tolerance, 0, 1, 'Tolerance');
  if (!Array.isArray(costs) || costs.length !== source.length || !Array.isArray(costs[0]) || costs[0].length < 1 || costs[0].length > 6) throw new RangeError('Cost shape must match the source and one to six reconstruction symbols.');
  const columns = costs[0].length;
  costs.forEach(row => {
    if (!Array.isArray(row) || row.length !== columns) throw new RangeError('Cost rows must have equal length.');
    row.forEach(value => bounded(value, 0, 10000, 'Distortion cost'));
  });
  if (initial !== null && !Array.isArray(initial)) throw new RangeError('Initial output probabilities must be an array.');
  const starting = initial === null ? Array(columns).fill(1 / columns) : [...initial];
  probabilityVector(starting, 'Initial output probabilities');
  if (starting.length !== columns) throw new RangeError('Initial output shape does not match costs.');
  let logOutput = starting.map(value => value === 0 ? -Infinity : Math.log(value));
  const logSource = source.map(value => value === 0 ? -Infinity : Math.log(value));
  const scaledCosts = costs.map(row => row.map(value => value * lambda * LOG_TWO));
  const trace = [];
  let state;
  for (let iteration = 0; iteration <= iterations; iteration += 1) {
    const normalizers = scaledCosts.map(row => logSumExp(row.map((cost, j) => logOutput[j] - cost)));
    const logConditional = scaledCosts.map((row, i) => row.map((cost, j) => logOutput[j] - cost - normalizers[i]));
    const conditional = logConditional.map(row => row.map(Math.exp));
    const logMarginal = Array.from({
      length: columns
    }, (_, j) => logSumExp(source.map((_, i) => logSource[i] + logConditional[i][j])));
    const marginal = logMarginal.map(Math.exp);
    let distortion = 0;
    let informationNats = 0;
    source.forEach((mass, i) => conditional[i].forEach((probability, j) => {
      const joint = mass * probability;
      if (joint > 0) {
        distortion += joint * costs[i][j];
        informationNats += joint * (logConditional[i][j] - logMarginal[j]);
      }
    }));
    const variationalNats = -source.reduce((sum, mass, i) => sum + mass * normalizers[i], 0);
    const logGradients = Array.from({
      length: columns
    }, (_, j) => logSumExp(source.map((_, i) => logSource[i] - scaledCosts[i][j] - normalizers[i])));
    const information = informationNats / LOG_TWO;
    const upper = information + lambda * distortion;
    const lower = (variationalNats - Math.max(...logGradients)) / LOG_TWO;
    const gap = Math.max(0, upper - lower);
    state = {
      iteration,
      source: [...source],
      costs: costs.map(row => [...row]),
      lambda,
      initial: [...starting],
      outputGuess: logOutput.map(Math.exp),
      conditional,
      output: marginal,
      distortion,
      information,
      variational: variationalNats / LOG_TWO,
      lower,
      upper,
      gap,
      converged: gap <= tolerance,
      supportLimited: starting.some(value => value === 0)
    };
    if (iteration <= traceLimit) trace.push(state);
    if (state.converged || iteration === iterations) break;
    logOutput = logMarginal;
  }
  return freeze({
    ...state,
    trace
  });
}
export function gaussianRateDistortion(variance, distortion) {
  bounded(variance, 0, 1000000, 'Variance');
  bounded(distortion, 0, 1000000, 'Distortion');
  if (variance === 0 || distortion >= variance) return 0;
  if (distortion === 0) return Infinity;
  return 0.5 * (Math.log2(variance) - Math.log2(distortion));
}
export function gaussianAllocation(variances = [9, 1], budget = 2) {
  if (!Array.isArray(variances) || variances.length < 1 || variances.length > 6) throw new RangeError('Use one to six component variances.');
  variances.forEach(value => bounded(value, 0, 10000, 'Component variance'));
  bounded(budget, 0, 60000, 'Total distortion budget');
  const totalVariance = variances.reduce((sum, value) => sum + value, 0);
  const distortion = Math.min(budget, totalVariance);
  let level = 0;
  if (distortion >= totalVariance) level = Math.max(...variances);else if (distortion > 0) {
    // Solve the piecewise-linear budget equation exactly on its active interval.
    const sorted = [...variances].sort((a, b) => a - b);
    let exhausted = 0;
    for (let index = 0; index < sorted.length; index += 1) {
      if (sorted[index] === 0) continue;
      const candidate = (distortion - exhausted) / (sorted.length - index);
      if (candidate <= sorted[index]) {
        level = candidate;
        break;
      }
      exhausted += sorted[index];
    }
  }
  const components = variances.map((variance, index) => {
    const error = Math.min(variance, level);
    return {
      index,
      variance,
      distortion: error,
      retainedVariance: variance - error,
      rate: gaussianRateDistortion(variance, error)
    };
  });
  const allocatedDistortion = components.reduce((sum, component) => sum + component.distortion, 0);
  if (distortion > 0 && (level === 0 || Math.abs(allocatedDistortion / distortion - 1) > 1e-12)) {
    throw new RangeError('This positive distortion budget cannot be represented accurately at the chosen numerical scale. Rescale the variances and budget together.');
  }
  return freeze({
    variances: [...variances],
    budget,
    totalVariance,
    distortion,
    level,
    components,
    rate: components.reduce((sum, component) => sum + component.rate, 0)
  });
}
export function zeroRateFidelity() {
  return freeze([{
    label: 'Always reconstruct 0',
    output: [0, 1, 0],
    pairs: [[-1, 0, 0.5], [1, 0, 0.5]],
    distortion: 1,
    information: 0
  }, {
    label: 'Independent fair −1 / +1',
    output: [0.5, 0, 0.5],
    pairs: [[-1, -1, 0.25], [-1, 1, 0.25], [1, -1, 0.25], [1, 1, 0.25]],
    distortion: 2,
    information: 0
  }]);
}
