// Bounded teaching models. All plotted values are computed from these declared fixtures.
export const naiveBayesVocabulary = Object.freeze(['free', 'money', 'win', 'meeting', 'agenda']);
export const naiveBayesDocuments = Object.freeze([Object.freeze({
  label: 1,
  text: 'free free money',
  counts: Object.freeze([2, 1, 0, 0, 0])
}), Object.freeze({
  label: 1,
  text: 'free money win',
  counts: Object.freeze([1, 1, 1, 0, 0])
}), Object.freeze({
  label: 0,
  text: 'meeting agenda',
  counts: Object.freeze([0, 0, 0, 1, 1])
})]);
function finiteNumber(value, name, minimum, maximum) {
  if (typeof value !== 'number' || !Number.isFinite(value) || value < minimum || value > maximum) {
    throw new RangeError(name + ' must be a finite number from ' + minimum + ' to ' + maximum + '.');
  }
  return value;
}
function denseNumbers(values, name, minimum, maximum, length) {
  if (!Array.isArray(values) || values.length !== length) {
    throw new RangeError(name + ' must contain exactly ' + length + ' values.');
  }
  for (let i = 0; i < values.length; i += 1) {
    if (!Object.hasOwn(values, i)) throw new RangeError(name + ' must not contain empty slots.');
    finiteNumber(values[i], name + '[' + i + ']', minimum, maximum);
  }
  return values;
}
export function formatNaiveBayesNumber(value, digits = 4) {
  if (value === null) return 'undefined';
  if (value === Infinity) return '+∞';
  if (value === -Infinity) return '−∞';
  if (!Number.isFinite(value)) throw new RangeError('Cannot display a nonfinite numerical result.');
  if (value === 0) return '0';
  if (Math.abs(value) < 10 ** -digits || Math.abs(value) >= 1e6) return value.toExponential(3);
  return value.toFixed(digits).replace(/\.?0+$/, '');
}
export function normalizeNaiveBayesScores(scores) {
  if (!Array.isArray(scores) || scores.length < 1 || scores.length > 20) {
    throw new RangeError('Supply 1–20 class scores.');
  }
  for (let i = 0; i < scores.length; i += 1) {
    if (!Object.hasOwn(scores, i) || typeof scores[i] !== 'number' || !Number.isFinite(scores[i]) && scores[i] !== -Infinity) {
      throw new RangeError('Each score must be finite or negative infinity.');
    }
  }
  const maximum = Math.max(...scores);
  if (maximum === -Infinity) return {
    probabilities: null,
    logProbabilities: null,
    logNormalizer: -Infinity
  };
  const shifted = scores.map(score => score - maximum);
  // Scores in the UI are small. Reject finite subtraction overflow in exported helper calls.
  for (let i = 0; i < scores.length; i += 1) {
    if (scores[i] !== -Infinity && !Number.isFinite(shifted[i])) {
      throw new RangeError('Score differences exceed this arithmetic range.');
    }
  }
  const logShiftedSum = Math.log(shifted.reduce((sum, score) => sum + Math.exp(score), 0));
  const logProbabilities = shifted.map(score => score - logShiftedSum);
  const logNormalizer = maximum + logShiftedSum;
  if (!Number.isFinite(logNormalizer)) throw new RangeError('The log normalizer exceeds this arithmetic range.');
  return {
    probabilities: logProbabilities.map(Math.exp),
    logProbabilities,
    logNormalizer
  };
}
export function tokenizeNaiveBayesMessage(text) {
  if (typeof text !== 'string' || text.length > 300) throw new RangeError('Use at most 300 characters.');
  const tokens = text.toLowerCase().match(/[a-z]+/g) || [];
  if (tokens.length > 30) throw new RangeError('Use at most 30 word tokens in this teaching view.');
  const counts = naiveBayesVocabulary.map(word => tokens.filter(token => token === word).length);
  const knownTokens = tokens.filter(token => naiveBayesVocabulary.includes(token));
  const unknownTokens = tokens.filter(token => !naiveBayesVocabulary.includes(token));
  return {
    tokens,
    counts,
    knownTokens,
    unknownTokens
  };
}
export function naiveBayesCountParameters(alpha = 1) {
  finiteNumber(alpha, 'Alpha', 0, 10);
  if (alpha > 0 && alpha < 1e-8) throw new RangeError('Use alpha=0 or alpha at least 1e-8 in this teaching model.');
  const classCounts = [1, 2];
  const wordCounts = [[0, 0, 0, 1, 1], [3, 2, 1, 0, 0]];
  const probabilities = wordCounts.map(row => {
    const denominator = row.reduce((sum, value) => sum + value, 0) + alpha * row.length;
    return row.map(value => (value + alpha) / denominator);
  });
  return {
    alpha,
    classCounts,
    wordCounts,
    probabilities,
    logProbabilities: probabilities.map(row => row.map(Math.log))
  };
}
function scoreSnapshot(scores, label) {
  const normalized = normalizeNaiveBayesScores(scores);
  const logOdds = scores[0] === -Infinity && scores[1] === -Infinity ? null : scores[1] - scores[0];
  return {
    label,
    scores: [...scores],
    ...normalized,
    logOdds,
    decision: normalized.probabilities === null ? null : normalized.probabilities[1] > normalized.probabilities[0] ? 1 : 0
  };
}
export function tokenEvidenceState(text = 'free', alpha = 1, spamPrior = 2 / 3) {
  finiteNumber(spamPrior, 'Spam prior', 0.01, 0.99);
  const message = tokenizeNaiveBayesMessage(text);
  const parameters = naiveBayesCountParameters(alpha);
  const scores = [Math.log1p(-spamPrior), Math.log(spamPrior)];
  const frames = [scoreSnapshot(scores, 'Start with the class prior')];
  for (const token of message.knownTokens) {
    const index = naiveBayesVocabulary.indexOf(token);
    const terms = parameters.logProbabilities.map(row => row[index]);
    scores[0] += terms[0];
    scores[1] += terms[1];
    const contribution = terms[0] === -Infinity && terms[1] === -Infinity ? null : terms[1] - terms[0];
    frames.push({
      ...scoreSnapshot(scores, 'Observe ' + token),
      token,
      index,
      terms,
      contribution
    });
  }
  return {
    text,
    spamPrior,
    ...message,
    parameters,
    frames,
    final: frames.at(-1)
  };
}
export function presenceEvidenceState(counts = [1, 0, 0, 0, 0]) {
  denseNumbers(counts, 'Word counts', 0, 20, 5);
  if (counts.some(count => !Number.isInteger(count))) throw new RangeError('Word counts must be integers.');
  const binary = counts.map(count => Number(count > 0));
  const documentCounts = [[0, 0, 0, 1, 1], [2, 2, 1, 0, 0]];
  const classCounts = [1, 2];
  const parameters = documentCounts.map((row, c) => row.map(count => (count + 1) / (classCounts[c] + 2)));
  const terms = parameters.map(row => row.map((probability, j) => binary[j] ? Math.log(probability) : Math.log1p(-probability)));
  const scores = terms.map((row, c) => Math.log(classCounts[c] / 3) + row.reduce((sum, term) => sum + term, 0));
  const multinomial = naiveBayesCountParameters(1);
  const countScores = multinomial.logProbabilities.map((row, c) => Math.log(classCounts[c] / 3) + row.reduce((sum, logProbability, j) => sum + counts[j] * logProbability, 0));
  return {
    counts: [...counts],
    binary,
    documentCounts,
    classCounts,
    parameters,
    terms,
    bernoulli: scoreSnapshot(scores, 'Bernoulli model'),
    multinomial: scoreSnapshot(countScores, 'Multinomial model')
  };
}
export function gaussianLogDensity(value, mean, variance) {
  finiteNumber(value, 'Reading', -1e6, 1e6);
  finiteNumber(mean, 'Mean', -1e6, 1e6);
  finiteNumber(variance, 'Variance', 1e-12, 1e12);
  return -0.5 * (Math.log(2 * Math.PI) + Math.log(variance) + (value - mean) ** 2 / variance);
}
function intervalMass(center, width, mean, variance) {
  // Composite Simpson integration; a positive bounded integral avoids tail-CDF subtraction.
  const pieces = 64;
  const left = center - width / 2;
  const step = width / pieces;
  let total = 0;
  for (let i = 0; i <= pieces; i += 1) {
    const weight = i === 0 || i === pieces ? 1 : i % 2 ? 4 : 2;
    total += weight * Math.exp(gaussianLogDensity(left + i * step, mean, variance));
  }
  return total * step / 3;
}
export function gaussianObservationState(reading = 0, width = 0.4) {
  finiteNumber(reading, 'Reading', -5, 5);
  finiteNumber(width, 'Interval width', 0.1, 1);
  const variances = [1, 4];
  const logDensities = variances.map(variance => gaussianLogDensity(reading, 0, variance));
  const densities = logDensities.map(Math.exp);
  const masses = variances.map(variance => intervalMass(reading, width, 0, variance));
  const scores = logDensities.map(logDensity => Math.log(0.5) + logDensity);
  const curves = variances.map(variance => Array.from({
    length: 241
  }, (_, i) => {
    const x = -6 + i / 20;
    return [x, Math.exp(gaussianLogDensity(x, 0, variance))];
  }));
  return {
    reading,
    width,
    variances,
    logDensities,
    densities,
    masses,
    curves,
    midpointAreas: densities.map(density => density * width),
    crossing: Math.sqrt(8 * Math.log(2) / 3),
    ...scoreSnapshot(scores, 'Continuous observation')
  };
}
export function gaussianGeometryState(mode = 'unequal', point = [0, 0]) {
  if (!['equal', 'unequal'].includes(mode)) throw new RangeError('Choose equal or unequal variance.');
  denseNumbers(point, 'Probe coordinates', -8, 6, 2);
  const means = [[-2, -2], [2, 2]];
  const variances = [0.5, mode === 'equal' ? 0.5 : 2];
  // Sum the squared distances before the shared normalization. For the equal-
  // variance fixture, points on x+y=0 then preserve the exact symmetry/tie.
  const scoreAt = coordinates => means.map((mean, c) => Math.log(0.5) - Math.log(2 * Math.PI) - Math.log(variances[c]) - ((coordinates[0] - mean[0]) ** 2 + (coordinates[1] - mean[1]) ** 2) / (2 * variances[c]));
  const center = [-10 / 3, -10 / 3];
  const radius = Math.sqrt(200 / 9 - (6 - Math.log(4)) / 0.75);
  const boundary = mode === 'equal' ? [[-6, 6], [6, -6]] : Array.from({
    length: 241
  }, (_, i) => {
    const angle = 2 * Math.PI * i / 240;
    return [center[0] + radius * Math.cos(angle), center[1] + radius * Math.sin(angle)];
  });
  const grid = [];
  for (let i = 0; i < 28; i += 1) {
    for (let j = 0; j < 28; j += 1) {
      const x = -8 + (i + 0.5) * 0.5;
      const y = -8 + (j + 0.5) * 0.5;
      const scores = scoreAt([x, y]);
      grid.push({
        x,
        y,
        label: scores[1] > scores[0] ? 1 : 0
      });
    }
  }
  return {
    mode,
    point: [...point],
    means,
    variances,
    center,
    radius,
    boundary,
    grid,
    ...scoreSnapshot(scoreAt(point), 'Probe')
  };
}
export function copiedAlarmState(copies = 3, positive = true, prior = 0.2) {
  if (!Number.isInteger(copies) || copies < 1 || copies > 5) throw new RangeError('Use 1–5 copies.');
  if (typeof positive !== 'boolean') throw new TypeError('Alarm state must be true or false.');
  finiteNumber(prior, 'Fault prior', 0.05, 0.5);
  const likelihoods = positive ? [0.4, 0.8] : [0.6, 0.2];
  const priors = [1 - prior, prior];
  const trueScores = likelihoods.map((probability, c) => Math.log(priors[c]) + Math.log(probability));
  const naiveScores = likelihoods.map((probability, c) => Math.log(priors[c]) + copies * Math.log(probability));
  const truth = scoreSnapshot(trueScores, 'One actual alarm');
  const naive = scoreSnapshot(naiveScores, 'Copies treated as independent');
  const rows = [true, false].map(observed => {
    const conditional = observed ? [0.4, 0.8] : [0.6, 0.2];
    const masses = conditional.map((probability, c) => priors[c] * probability);
    const scores = conditional.map((probability, c) => Math.log(priors[c]) + copies * Math.log(probability));
    const predicted = scoreSnapshot(scores, 'Observed state');
    return {
      positive: observed,
      masses,
      total: masses[0] + masses[1],
      prediction: predicted.decision,
      reported: predicted.probabilities[1],
      actual: masses[1] / (masses[0] + masses[1])
    };
  });
  const accuracy = rows.reduce((sum, row) => sum + row.masses[row.prediction], 0);
  const brier = rows.reduce((sum, row) => sum + row.masses[0] * row.reported ** 2 + row.masses[1] * (1 - row.reported) ** 2, 0);
  return {
    copies,
    positive,
    prior,
    likelihoods,
    truth,
    naive,
    rows,
    accuracy,
    brier
  };
}
export const reliabilityCases = Object.freeze([[0.05, 0], [0.10, 0], [0.10, 1], [0.20, 0], [0.25, 0], [0.40, 1], [0.55, 0], [0.60, 1], [0.75, 1], [0.80, 0], [0.90, 1], [0.95, 1]].map((row, i) => Object.freeze({
  id: i + 1,
  probability: row[0],
  outcome: row[1]
})));
export function reliabilityState(binCount = 4, compression = false) {
  if (!Number.isInteger(binCount) || binCount < 2 || binCount > 6) throw new RangeError('Use 2–6 bins.');
  if (typeof compression !== 'boolean') throw new TypeError('Compression must be true or false.');
  const cases = reliabilityCases.map(item => ({
    ...item,
    probability: compression ? 0.15 + 0.7 * item.probability : item.probability
  }));
  const bins = Array.from({
    length: binCount
  }, (_, index) => ({
    index,
    lower: index / binCount,
    upper: (index + 1) / binCount,
    cases: []
  }));
  cases.forEach(item => bins[Math.min(binCount - 1, Math.floor(item.probability * binCount))].cases.push(item));
  bins.forEach(bin => {
    bin.count = bin.cases.length;
    bin.meanPrediction = bin.count ? bin.cases.reduce((sum, item) => sum + item.probability, 0) / bin.count : null;
    bin.observedFraction = bin.count ? bin.cases.reduce((sum, item) => sum + item.outcome, 0) / bin.count : null;
  });
  const brier = cases.reduce((sum, item) => sum + (item.probability - item.outcome) ** 2, 0) / cases.length;
  const logLoss = cases.reduce((sum, item) => sum - (item.outcome ? Math.log(item.probability) : Math.log1p(-item.probability)), 0) / cases.length;
  return {
    binCount,
    compression,
    cases,
    bins,
    brier,
    logLoss
  };
}
