// Small declared finite models used only by the exponential-family lesson.
const OUTCOMES = [-1, 0, 1];
function finiteNumber(value, name, maximum = 100) {
  if (!Number.isFinite(value) || Math.abs(value) > maximum) {
    throw new RangeError(`${name} must be finite with absolute value at most ${maximum}.`);
  }
}
export function formatFamilyNumber(value) {
  if (!Number.isFinite(value)) return String(value);
  if (value === 0) return '0';
  if (Math.abs(value) < 0.0001) return value.toExponential(3);
  return Number(value.toPrecision(6)).toString();
}
export function finiteExponentialFamily(etaLinear, etaSquare = 0, base = [1, 1, 1]) {
  finiteNumber(etaLinear, 'Linear natural parameter');
  finiteNumber(etaSquare, 'Square natural parameter');
  if (base.length !== 3 || base.some(weight => !Number.isFinite(weight) || weight <= 0 || weight > 100)) {
    throw new RangeError('Supply three positive base weights, each at most 100.');
  }
  const logWeights = OUTCOMES.map((value, index) => Math.log(base[index]) + etaLinear * value + etaSquare * value ** 2);
  const shift = Math.max(...logWeights);
  const relativeWeights = logWeights.map(weight => Math.exp(weight - shift));
  const relativeTotal = relativeWeights.reduce((total, weight) => total + weight, 0);
  const probabilities = relativeWeights.map(weight => weight / relativeTotal);
  const mean = [0, 1].map(powerIndex => OUTCOMES.reduce((total, value, index) => total + probabilities[index] * value ** (powerIndex + 1), 0));
  // Centered products preserve small positive variance better than E[T²]−E[T]².
  const covariance = mean.map((_, row) => mean.map((__, column) => OUTCOMES.reduce((total, value, index) => total + probabilities[index] * (value ** (row + 1) - mean[row]) * (value ** (column + 1) - mean[column]), 0)));
  return {
    outcomes: [...OUTCOMES],
    base: [...base],
    logWeights,
    shift,
    relativeWeights,
    relativeTotal,
    logPartition: shift + Math.log(relativeTotal),
    probabilities,
    mean,
    covariance
  };
}
export function finiteMomentFit(counts) {
  if (counts.length !== 3 || counts.some(count => !Number.isInteger(count) || count < 0 || count > 999)) {
    throw new RangeError('Three counts must be integers from 0 to 999.');
  }
  const count = counts.reduce((total, value) => total + value, 0);
  if (count === 0) return {
    status: 'empty',
    count,
    mean: null,
    probabilities: null,
    parameters: null
  };
  const probabilities = counts.map(value => value / count);
  const mean = [probabilities[2] - probabilities[0], probabilities[2] + probabilities[0]];
  if (counts.some(value => value === 0)) return {
    status: 'boundary',
    count,
    probabilities,
    mean,
    parameters: null
  };
  const parameters = [0.5 * Math.log(counts[2] / counts[0]), 0.5 * (Math.log(counts[2]) + Math.log(counts[0]) - 2 * Math.log(counts[1]))];
  return {
    status: 'finite',
    count,
    probabilities,
    mean,
    parameters
  };
}
export const SUFFICIENCY_REFERENCE = [1, 1, 1, 1, 0, 0, 1, 1];
export function summaryComparison(observations, groupProbability = 0.8, secondProbability = 0.3, grouped = false) {
  if (observations.length !== 8 || observations.some(value => value !== 0 && value !== 1)) throw new RangeError('Supply eight binary observations.');
  for (const probability of [groupProbability, secondProbability]) {
    if (!Number.isFinite(probability) || probability <= 0 || probability >= 1) throw new RangeError('Probabilities must be strictly between zero and one.');
  }
  const summarize = values => [values.slice(0, 4).reduce((a, b) => a + b, 0), values.slice(4).reduce((a, b) => a + b, 0)];
  const groupCounts = summarize(observations);
  const referenceCounts = summarize(SUFFICIENCY_REFERENCE);
  const probabilities = [groupProbability, grouped ? secondProbability : groupProbability];
  const logLikelihood = counts => counts.reduce((total, successes, index) => total + successes * Math.log(probabilities[index]) + (4 - successes) * Math.log1p(-probabilities[index]), 0);
  const logRatio = logLikelihood(groupCounts) - logLikelihood(referenceCounts);
  return {
    groupCounts,
    referenceCounts,
    total: groupCounts[0] + groupCounts[1],
    referenceTotal: 6,
    logLikelihood: logLikelihood(groupCounts),
    referenceLogLikelihood: logLikelihood(referenceCounts),
    logRatio,
    ratio: Math.exp(logRatio)
  };
}
export function gaussianSummary(values) {
  if (!values.length || values.some(value => !Number.isFinite(value))) throw new RangeError('Supply nonempty finite observations.');
  let count = 0;
  let mean = 0;
  let squaredDeviations = 0;
  for (const value of values) {
    count += 1;
    const delta = value - mean;
    mean += delta / count;
    squaredDeviations += delta * (value - mean);
  }
  return {
    count,
    mean,
    squaredDeviations
  };
}
export function mergeGaussianSummaries(left, right) {
  for (const summary of [left, right]) {
    if (!Number.isInteger(summary.count) || summary.count < 1 || !Number.isFinite(summary.mean) || !Number.isFinite(summary.squaredDeviations) || summary.squaredDeviations < 0) throw new RangeError('Summaries need a positive integer count, finite mean and nonnegative squared deviations.');
  }
  const count = left.count + right.count;
  const difference = right.mean - left.mean;
  return {
    count,
    mean: left.mean + difference * right.count / count,
    squaredDeviations: left.squaredDeviations + right.squaredDeviations + difference ** 2 * left.count * right.count / count
  };
}
function chooseInteger(total, selected) {
  let result = 1;
  for (let index = 1; index <= selected; index += 1) result *= (total - index + 1) / index;
  return result;
}
export function betaCoordinateModel(alpha, beta) {
  if (![alpha, beta].every(value => Number.isInteger(value) && value >= 1 && value <= 8)) throw new RangeError('The finite teaching model uses integer alpha and beta from 1 to 8.');
  const normalizer = (alpha + beta - 1) * chooseInteger(alpha + beta - 2, alpha - 1);
  const probabilityDensity = probability => {
    if (!Number.isFinite(probability) || probability < 0 || probability > 1) throw new RangeError('p must lie in [0, 1].');
    return normalizer * probability ** (alpha - 1) * (1 - probability) ** (beta - 1);
  };
  const etaDensity = eta => {
    finiteNumber(eta, 'Log odds', 30);
    const probability = eta >= 0 ? 1 / (1 + Math.exp(-eta)) : Math.exp(eta) / (1 + Math.exp(eta));
    return probabilityDensity(probability) * probability * (1 - probability);
  };
  const cdf = probability => {
    const total = alpha + beta - 1;
    let result = 0;
    for (let selected = alpha; selected <= total; selected += 1) result += chooseInteger(total, selected) * probability ** selected * (1 - probability) ** (total - selected);
    return result;
  };
  const intervalMass = cdf(0.75) - cdf(0.25);
  const probabilityCurve = Array.from({
    length: 201
  }, (_, index) => [index / 200, probabilityDensity(index / 200)]);
  const etaCurve = Array.from({
    length: 241
  }, (_, index) => [-6 + index / 20, etaDensity(-6 + index / 20)]);
  const interval = [Math.log(1 / 3), Math.log(3)];
  return {
    alpha,
    beta,
    normalizer,
    probabilityCurve,
    etaCurve,
    probabilityDensity,
    etaDensity,
    interval,
    intervalMass,
    displayedEtaMass: cdf(1 / (1 + Math.exp(-6))) - cdf(1 / (1 + Math.exp(6)))
  };
}
