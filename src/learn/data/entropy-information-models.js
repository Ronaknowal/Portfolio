export const SYMBOLS = ['A', 'B', 'C', 'D'];
export const SOURCE_WEIGHTS = [4, 2, 1, 1];
export const CODEBOOKS = {
  matched: {
    label: 'Short words for frequent A',
    words: ['0', '10', '110', '111']
  },
  reversed: {
    label: 'Short words for rare D',
    words: ['111', '110', '10', '0']
  },
  fixed: {
    label: 'Two bits for every symbol',
    words: ['00', '01', '10', '11']
  }
};
export function entropyNumber(value, digits = 5) {
  if (value === Infinity) return '∞';
  if (!Number.isFinite(value)) return 'undefined';
  if (value === 0) return '0';
  if (Math.abs(value) < 10 ** -digits) return value.toExponential(2);
  return Number(value.toFixed(digits)).toString();
}
function bounded(value, min, max, name) {
  if (!Number.isFinite(value) || value < min || value > max) throw new Error(`${name} must be from ${min} to ${max}.`);
}
export function normalizeWeights(weights) {
  if (!Array.isArray(weights) || weights.length < 2 || weights.length > 8) throw new Error('Use between two and eight weights.');
  weights.forEach(value => bounded(value, 0, 100, 'Each weight'));
  const total = weights.reduce((sum, value) => sum + value, 0);
  if (!(total > 0)) throw new Error('At least one weight must be positive.');
  const normalized = weights.map(value => value / total);
  if (weights.some((value, index) => value > 0 && normalized[index] === 0)) {
    throw new RangeError('A positive weight is too small to normalize without becoming zero. Use weights with a smaller ratio.');
  }
  return normalized;
}
export function parseWeights(text) {
  const parts = text.trim().split(/[\s,]+/);
  if (parts.length !== 4 || parts.some(part => part === '')) throw new Error('Enter exactly four comma-separated weights for A, B, C, D.');
  const weights = parts.map(Number);
  if (weights.some((value, index) => value === 0 && /[1-9]/.test(parts[index].split(/[eE]/)[0]))) {
    throw new RangeError('A nonzero weight is too small for this calculator. Use a larger value or an intentional exact zero.');
  }
  normalizeWeights(weights);
  return weights;
}
export function informationState(pWeights, qWeights = pWeights, base = 2) {
  if (base !== 2 && base !== Math.E) throw new Error('Choose bits or nats.');
  const p = normalizeWeights(pWeights),
    q = normalizeWeights(qWeights);
  if (p.length !== q.length) throw new Error('Both distributions need the same labeled outcomes.');
  const factor = Math.log(base);
  const rows = p.map((mass, index) => {
    const model = q[index];
    const surprise = mass === 0 ? Infinity : -Math.log(mass) / factor;
    const price = model === 0 ? Infinity : -Math.log(model) / factor;
    return {
      label: SYMBOLS[index] ?? String(index),
      p: mass,
      q: model,
      surprise,
      price,
      entropy: mass === 0 ? 0 : mass * surprise,
      crossEntropy: mass === 0 ? 0 : mass * price,
      kl: mass === 0 ? 0 : model === 0 ? Infinity : mass * (Math.log(mass) - Math.log(model)) / factor
    };
  });
  const sum = key => rows.reduce((total, row) => total + row[key], 0);
  // The signed row terms remain visible. For the total, sum the equivalent
  // nonnegative integrand p log(p/q) - p + q. The linear terms cancel for
  // probability vectors; a local series avoids subtractive cancellation near p=q.
  const totalKl = rows.reduce((total, row) => {
    if (row.p === 0) return total + row.q / factor;
    if (row.q === 0) return Infinity;
    const delta = (row.p - row.q) / row.q;
    if (Math.abs(delta) < 0.01) {
      let term = delta * delta;
      let kernel = 0;
      for (let power = 2; power <= 12; power++) {
        kernel += (power % 2 === 0 ? 1 : -1) * term / (power * (power - 1));
        term *= delta;
      }
      return total + row.q * kernel / factor;
    }
    return total + (row.p * (Math.log(row.p) - Math.log(row.q)) - row.p + row.q) / factor;
  }, 0);
  return {
    p,
    q,
    rows,
    entropy: sum('entropy'),
    crossEntropy: sum('crossEntropy'),
    kl: totalKl,
    base
  };
}
export function binaryEntropyState(probability) {
  bounded(probability, 0, 1, 'Probability');
  const state = informationState([probability, 1 - probability]);
  return {
    ...state,
    probability,
    curve: Array.from({
      length: 101
    }, (_, index) => {
      const p = index / 100;
      return [p, informationState([p, 1 - p]).entropy];
    })
  };
}
export function prefixCodeState(message, codebook = 'matched', consumed = 0) {
  if (typeof message !== 'string' || !/^[ABCD]{1,16}$/.test(message)) throw new Error('Use 1–16 uppercase symbols A, B, C or D, without spaces.');
  if (!Object.hasOwn(CODEBOOKS, codebook)) throw new Error('Choose a listed codebook.');
  const words = CODEBOOKS[codebook].words;
  const chunks = [...message].map(symbol => words[SYMBOLS.indexOf(symbol)]);
  const bits = chunks.join('');
  if (!Number.isInteger(consumed)) throw new Error('The bit position must be an integer.');
  bounded(consumed, 0, bits.length, 'Bit position');
  let buffer = '',
    decoded = '';
  for (const bit of bits.slice(0, consumed)) {
    buffer += bit;
    const found = words.indexOf(buffer);
    if (found !== -1) {
      decoded += SYMBOLS[found];
      buffer = '';
    }
  }
  const p = normalizeWeights(SOURCE_WEIGHTS);
  return {
    message,
    words,
    chunks,
    bits,
    consumed,
    decoded,
    buffer,
    expectedLength: p.reduce((sum, mass, index) => sum + mass * words[index].length, 0),
    messageAverage: bits.length / message.length,
    kraft: words.reduce((sum, word) => sum + 2 ** -word.length, 0)
  };
}
export function conditionalLossState(noise = 0.1, trust = 0.9) {
  bounded(noise, 0, 0.5, 'Noise probability');
  bounded(trust, 0, 1, 'Model probability');
  const rows = [0, 1].flatMap(x => [0, 1].map(y => {
    const conditional = x === y ? 1 - noise : noise;
    const predicted = x === y ? trust : 1 - trust;
    const mass = conditional / 2;
    return {
      x,
      y,
      conditional,
      predicted,
      mass,
      loss: mass === 0 ? 0 : predicted === 0 ? Infinity : -mass * Math.log2(predicted)
    };
  }));
  const state = informationState([1 - noise, noise], [trust, 1 - trust]);
  return {
    rows,
    noise,
    trust,
    marginalEntropy: 1,
    conditionalEntropy: state.entropy,
    modelLoss: state.crossEntropy,
    excess: state.kl,
    jointEntropy: 1 + state.entropy
  };
}
export function logitsState(gap = 2, offset = 0, target = 0) {
  bounded(gap, 0, 1000, 'Logit gap');
  bounded(offset, -1000, 1000, 'Common offset');
  if (![0, 1, 2].includes(target)) throw new Error('Choose target class 0, 1 or 2.');
  const logits = [gap, 0, -gap].map(value => value + offset);
  const maximum = Math.max(...logits);
  const shifted = logits.map(value => value - maximum);
  const exponentials = shifted.map(Math.exp);
  const normalizer = exponentials.reduce((sum, value) => sum + value, 0);
  const logNormalizer = Math.log(normalizer);
  const rows = logits.map((logit, index) => ({
    index,
    logit,
    shifted: shifted[index],
    exponential: exponentials[index],
    probability: exponentials[index] / normalizer,
    logProbability: shifted[index] - logNormalizer
  }));
  return {
    gap,
    offset,
    target,
    rows,
    maximum,
    normalizer,
    loss: -rows[target].logProbability
  };
}
export function continuousEntropyState(width = 0.25, scale = 100, bins = 4) {
  bounded(width, 0.125, 4, 'Width in metres');
  bounded(scale, 0.1, 100, 'Units per metre');
  if (!Number.isInteger(bins)) throw new Error('Bin count must be an integer.');
  bounded(bins, 2, 16, 'Bin count');
  const coordinate = factor => ({
    width: factor * width,
    density: 1 / (factor * width),
    entropy: Math.log2(factor * width),
    binWidth: factor * width / bins,
    event: [factor * width / 4, 3 * factor * width / 4],
    probability: 0.5
  });
  return {
    original: coordinate(1),
    transformed: coordinate(scale),
    bins,
    quantizedEntropy: Math.log2(bins),
    cellProbability: 1 / bins,
    scale,
    gaussianKl: Math.log(2) - 0.25
  };
}
export function maxEntropyState(mean = 10 / 7, fraction = 0.5) {
  bounded(mean, 0, 2, 'Required mean');
  // Exact endpoint laws are handled separately. Closer interior means need
  // log-domain/high-precision probabilities beyond this finite visual model.
  if (mean !== 0 && mean !== 2 && (mean < 1e-6 || mean > 2 - 1e-6)) {
    throw new RangeError('Use mean 0 or 2, or an interior mean between 0.000001 and 1.999999.');
  }
  bounded(fraction, 0, 1, 'Position on the feasible slice');
  const lower = Math.max(0, mean - 1),
    upper = mean / 2;
  // The chosen interval enforces q0,q1,q2 >= 0; do not renormalize away its exact mean.
  const at = t => [1 - mean + t, mean - 2 * t, t].map(value => Math.max(0, value));
  const t = lower + fraction * (upper - lower);
  let optimum, eta;
  if (mean === 0 || mean === 2) {
    optimum = mean === 0 ? [1, 0, 0] : [0, 0, 1];
    eta = mean === 0 ? -Infinity : Infinity;
  } else {
    const root = Math.sqrt(1 + 6 * mean - 3 * mean * mean);
    const ratio = mean <= 1 ? 2 * mean / (1 - mean + root) : (mean - 1 + root) / (2 * (2 - mean));
    const total = 1 + ratio + ratio * ratio;
    optimum = [1 / total, ratio / total, ratio * ratio / total];
    eta = Math.log(ratio);
  }
  const q = at(t),
    entropy = informationState(q).entropy,
    maximumEntropy = informationState(optimum).entropy;
  const gap = informationState(q, optimum).kl;
  const curve = Array.from({
    length: 101
  }, (_, index) => {
    const location = lower + index / 100 * (upper - lower);
    return [location, informationState(at(location)).entropy];
  });
  return {
    mean,
    fraction,
    lower,
    upper,
    t,
    q,
    optimum,
    eta,
    entropy,
    maximumEntropy,
    gap,
    curve
  };
}
