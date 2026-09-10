// Finite declared probability laws and bounded simulations, not learned MI estimators.
const LOG_TWO = Math.log(2);
const sum = values => values.reduce((total, value) => total + value, 0);
function bounded(value, minimum, maximum, label) {
  if (!Number.isFinite(value) || value < minimum || value > maximum) {
    throw new RangeError(`${label} must be finite and between ${minimum} and ${maximum}.`);
  }
}
function readonly(value) {
  if (value && typeof value === 'object' && !Object.isFrozen(value)) {
    Object.values(value).forEach(readonly);
    Object.freeze(value);
  }
  return value;
}
export function entropyBits(probabilities) {
  if (!Array.isArray(probabilities) || !probabilities.length) throw new TypeError('Use a nonempty probability array.');
  probabilities.forEach(p => bounded(p, 0, 1, 'Probability'));
  if (Math.abs(sum(probabilities) - 1) > 1e-10) throw new RangeError('Probabilities must sum to one.');
  return -sum(probabilities.map(p => p === 0 ? 0 : p * Math.log2(p)));
}
export function binaryEntropy(error) {
  bounded(error, 0, 1, 'Binary probability');
  return entropyBits([error, 1 - error]);
}
export function finiteInformation(joint) {
  if (!Array.isArray(joint) || !joint.length || !Array.isArray(joint[0]) || !joint[0].length) {
    throw new TypeError('Use a nonempty rectangular joint probability table.');
  }
  const width = joint[0].length;
  if (joint.some(row => !Array.isArray(row) || row.length !== width)) throw new TypeError('Joint rows must have equal lengths.');
  joint.flat().forEach(value => bounded(value, 0, 1, 'Joint probability'));
  if (Math.abs(sum(joint.flat()) - 1) > 1e-10) throw new RangeError('Joint probabilities must sum to one.');
  const pX = joint.map(sum);
  const pY = Array.from({
    length: width
  }, (_, j) => sum(joint.map(row => row[j])));
  const cells = joint.map((row, i) => row.map((mass, j) => {
    // Differences of logs avoid an underflowed product or overflowed ratio.
    const information = mass === 0 ? null : (Math.log(mass) - Math.log(pX[i]) - Math.log(pY[j])) / LOG_TWO;
    return {
      i,
      j,
      mass,
      independent: pX[i] * pY[j],
      information,
      contribution: mass === 0 ? 0 : mass * information
    };
  }));
  const conditional = joint.map((row, i) => pX[i] === 0 ? null : row.map(mass => mass / pX[i]));
  const conditionalEntropy = sum(conditional.map((row, i) => row ? pX[i] * entropyBits(row) : 0));
  return readonly({
    joint: joint.map(row => [...row]),
    pX,
    pY,
    cells,
    conditional,
    hx: entropyBits(pX),
    hy: entropyBits(pY),
    hxy: entropyBits(joint.flat()),
    conditionalEntropy,
    mi: sum(cells.flat().map(cell => cell.contribution))
  });
}
export function binaryChannel(prevalence = 0.5, error = 0.2) {
  bounded(prevalence, 0, 1, 'Probability of X=1');
  bounded(error, 0, 1, 'Channel flip probability');
  return readonly({
    prevalence,
    error,
    ...finiteInformation([[(1 - prevalence) * (1 - error), (1 - prevalence) * error], [prevalence * error, prevalence * (1 - error)]])
  });
}
export function xorInformation(bias = 0.5, reveal = 'a') {
  bounded(bias, 0.05, 0.95, 'Probability of B=1');
  if (!['a', 'b', 'both'].includes(reveal)) throw new RangeError('Choose A, B or both.');
  const rows = [];
  for (let a = 0; a < 2; a += 1) {
    for (let b = 0; b < 2; b += 1) rows.push({
      a,
      b,
      y: a ^ b,
      mass: 0.5 * (b ? bias : 1 - bias)
    });
  }
  const joint = Array.from({
    length: reveal === 'both' ? 4 : 2
  }, () => [0, 0]);
  rows.forEach(row => {
    const index = reveal === 'both' ? 2 * row.a + row.b : row[reveal];
    joint[index][row.y] += row.mass;
  });
  const slices = [0, 1].map(b => {
    const table = [[0, 0], [0, 0]];
    rows.filter(row => row.b === b).forEach(row => {
      table[row.a][row.y] = 0.5;
    });
    return {
      b,
      weight: b ? bias : 1 - bias,
      ...finiteInformation(table)
    };
  });
  return readonly({
    bias,
    reveal,
    rows,
    slices,
    conditionalMi: sum(slices.map(slice => slice.weight * slice.mi)),
    ...finiteInformation(joint)
  });
}
export const SIGNAL_INPUTS = Object.freeze(['S0 N0', 'S0 N1', 'S1 N0', 'S1 N1']);
function encoderState(encoder, labelError, beta) {
  const pX = [0.25, 0.25, 0.25, 0.25];
  const pYGivenX = pX.map((_, x) => x < 2 ? [1 - labelError, labelError] : [labelError, 1 - labelError]);
  const pXZ = encoder.map((row, x) => row.map(value => value * pX[x]));
  const pZY = encoder[0].map(() => [0, 0]);
  const triple = [];
  for (let x = 0; x < 4; x += 1) {
    for (let z = 0; z < encoder[0].length; z += 1) {
      for (let y = 0; y < 2; y += 1) {
        const mass = pX[x] * encoder[x][z] * pYGivenX[x][y];
        pZY[z][y] += mass;
        triple.push({
          x,
          z,
          y,
          mass
        });
      }
    }
  }
  const rate = finiteInformation(pXZ),
    relevance = finiteInformation(pZY);
  return {
    encoder: encoder.map(row => [...row]),
    pX,
    pYGivenX,
    pXZ,
    pZY,
    triple,
    pZ: relevance.pX,
    decoder: relevance.conditional,
    rate: rate.mi,
    relevance: relevance.mi,
    inputRelevance: 1 - binaryEntropy(labelError),
    discardedRelevance: 1 - binaryEntropy(labelError) - relevance.mi,
    objective: rate.mi - beta * relevance.mi,
    beta,
    labelError
  };
}
export function bottleneckRepresentation(mode = 'signal', noise = 0.2, labelError = 0.1, beta = 3) {
  bounded(noise, 0, 0.5, 'Encoder flip probability');
  bounded(labelError, 0, 0.5, 'Label flip probability');
  bounded(beta, 0, 12, 'Relevance weight');
  if (!['constant', 'signal', 'nuisance', 'both', 'noisy'].includes(mode)) throw new RangeError('Unknown representation.');
  const encoder = SIGNAL_INPUTS.map((_, x) => {
    const signal = Math.floor(x / 2),
      nuisance = x % 2;
    if (mode === 'constant') return [1];
    if (mode === 'both') return [0, 1, 2, 3].map(z => z === x ? 1 : 0);
    const value = mode === 'nuisance' ? nuisance : signal;
    const flip = mode === 'noisy' ? noise : 0;
    return value ? [flip, 1 - flip] : [1 - flip, flip];
  });
  return readonly({
    mode,
    noise,
    ...encoderState(encoder, labelError, beta)
  });
}
export function bottleneckCurve(labelError = 0.1, beta = 3) {
  return readonly(Array.from({
    length: 51
  }, (_, step) => {
    const state = bottleneckRepresentation('noisy', step / 100, labelError, beta);
    return {
      noise: state.noise,
      rate: state.rate,
      relevance: state.relevance,
      objective: state.objective
    };
  }));
}
function klNats(p, q) {
  return sum(p.map((value, index) => value === 0 ? 0 : q[index] === 0 ? Infinity : value * (Math.log(value) - Math.log(q[index]))));
}
export function bottleneckIterations(beta = 3, steps = 12, initialization = 'signal', labelError = 0.1) {
  bounded(beta, 0, 12, 'Relevance weight');
  bounded(labelError, 0.05, 0.45, 'Label flip probability');
  if (!Number.isInteger(steps) || steps < 0 || steps > 40) throw new RangeError('Use zero through forty complete updates.');
  if (!['signal', 'symmetric', 'nuisance'].includes(initialization)) throw new RangeError('Unknown initialization.');
  let encoder = initialization === 'symmetric' ? Array.from({
    length: 4
  }, () => [0.5, 0.5]) : initialization === 'nuisance' ? [[0.8, 0.2], [0.2, 0.8], [0.8, 0.2], [0.2, 0.8]] : [[0.8, 0.2], [0.6, 0.4], [0.3, 0.7], [0.2, 0.8]];
  const rows = [{
    step: 0,
    ...encoderState(encoder, labelError, beta),
    previousDistortion: null
  }];
  for (let step = 1; step <= steps; step += 1) {
    const previous = rows.at(-1);
    const distortion = previous.pYGivenX.map(probabilities => previous.decoder.map(decoder => klNats(probabilities, decoder)));
    encoder = distortion.map(costs => {
      const logWeights = costs.map((cost, z) => Math.log(previous.pZ[z]) - beta * cost);
      const maximum = Math.max(...logWeights);
      const weights = logWeights.map(value => Math.exp(value - maximum));
      const normalizer = sum(weights);
      return weights.map(value => value / normalizer);
    });
    rows.push({
      step,
      ...encoderState(encoder, labelError, beta),
      previousDistortion: distortion.map(row => row.map(value => value / LOG_TWO))
    });
  }
  return readonly({
    beta,
    steps,
    initialization,
    labelError,
    rows,
    current: rows.at(-1)
  });
}
export function variationalInformationBounds(referenceOne = 0.5, decoderError = 0.26, beta = 3) {
  bounded(referenceOne, 0.02, 0.98, 'Reference probability of Z=1');
  bounded(decoderError, 0.02, 0.98, 'Decoder probability of a flipped label');
  bounded(beta, 0, 12, 'Relevance weight');
  const state = bottleneckRepresentation('noisy', 0.2, 0.1, beta);
  const reference = [1 - referenceOne, referenceOne];
  const decoder = [[1 - decoderError, decoderError], [decoderError, 1 - decoderError]];
  const rateUpper = sum(state.encoder.map((row, x) => state.pX[x] * klNats(row, reference))) / LOG_TWO;
  const rateGap = klNats(state.pZ, reference) / LOG_TWO;
  const crossEntropy = -sum(state.pZY.flatMap((row, z) => row.map((mass, y) => mass * Math.log2(decoder[z][y]))));
  const predictiveLower = 1 - crossEntropy;
  const predictiveGap = sum(state.pZ.map((mass, z) => mass * klNats(state.decoder[z], decoder[z]))) / LOG_TWO;
  return readonly({
    ...state,
    referenceOne,
    decoderError,
    reference,
    approximateDecoder: decoder,
    rateUpper,
    rateGap,
    crossEntropy,
    predictiveLower,
    predictiveGap,
    objectiveUpper: rateUpper + beta * crossEntropy - beta
  });
}
function seededRandom(seed) {
  let state = seed >>> 0 || 1;
  return () => {
    state ^= state << 13;
    state ^= state >>> 17;
    state ^= state << 5;
    return (state >>> 0) / 4294967296;
  };
}
function informationFromPairs(pairs, categories) {
  const counts = Array.from({
    length: categories
  }, () => Array(categories).fill(0));
  pairs.forEach(([x, y]) => {
    counts[x][y] += 1;
  });
  return {
    counts,
    ...finiteInformation(counts.map(row => row.map(count => count / pairs.length)))
  };
}
export function sampledInformation({
  categories = 4,
  count = 100,
  seed = 831,
  mode = 'independent'
} = {}) {
  if (![2, 4, 8].includes(categories)) throw new RangeError('Use two, four or eight categories.');
  if (!Number.isInteger(count) || count < 20 || count > 2000) throw new RangeError('Use twenty through two thousand observations.');
  if (!Number.isInteger(seed) || seed < 1 || seed > 2147483647) throw new RangeError('Seed must be an integer from1 through2147483647.');
  if (!['independent', 'channel'].includes(mode)) throw new RangeError('Unknown sampling law.');
  const random = seededRandom(seed);
  const pairs = Array.from({
    length: count
  }, () => {
    const x = Math.floor(random() * categories),
      flip = random(),
      alternative = random();
    const y = mode === 'independent' ? Math.floor(flip * categories) : flip < 0.2 ? (x + 1 + Math.floor(alternative * (categories - 1))) % categories : x;
    return [x, y];
  });
  const shuffleRandom = seededRandom((seed ^ 0x5a39b174) >>> 0);
  const shuffled = Array.from({
    length: 20
  }, () => {
    const labels = pairs.map(pair => pair[1]);
    for (let i = labels.length - 1; i > 0; i -= 1) {
      const j = Math.floor(shuffleRandom() * (i + 1));
      [labels[i], labels[j]] = [labels[j], labels[i]];
    }
    return informationFromPairs(pairs.map((pair, i) => [pair[0], labels[i]]), categories).mi;
  });
  const sorted = [...shuffled].sort((left, right) => left - right);
  return readonly({
    categories,
    count,
    seed,
    mode,
    pairs,
    ...informationFromPairs(pairs, categories),
    populationMi: mode === 'independent' ? 0 : Math.log2(categories) - binaryEntropy(0.2) - 0.2 * Math.log2(categories - 1),
    shuffled,
    shuffleMedian: (sorted[9] + sorted[10]) / 2,
    shuffleMinimum: sorted[0],
    shuffleMaximum: sorted.at(-1)
  });
}
export function gaussianNoiseInformation(sigma) {
  bounded(sigma, 0, 4, 'Observation noise standard deviation');
  if (sigma === 0) return Infinity;
  const logarithm = sigma <= 1 ? Math.log1p(sigma * sigma) - 2 * Math.log(sigma) : Math.log1p(1 / (sigma * sigma));
  return 0.5 * logarithm / LOG_TWO;
}
