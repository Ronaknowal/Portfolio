// Exact finite designs or declared analytic variance models, not empirical data.
export const samplingPopulation = [2, 4, 6, 8, 10, 12, 14, 16];
export const assignmentBaselines = [0, 1, 5, 6, 10, 11];
function bounded(value, name, minimum, maximum) {
  if (!Number.isFinite(value) || value < minimum || value > maximum) {
    throw new RangeError(`${name} must be finite and between ${minimum} and ${maximum}.`);
  }
}
function integer(value, name, minimum, maximum) {
  bounded(value, name, minimum, maximum);
  if (!Number.isInteger(value)) throw new RangeError(`${name} must be an integer.`);
}
function denseArray(values, name, minimumLength, maximumLength) {
  if (!Array.isArray(values) || values.length < minimumLength || values.length > maximumLength) {
    throw new RangeError(`${name} needs ${minimumLength} to ${maximumLength} values.`);
  }
  for (let index = 0; index < values.length; index += 1) {
    if (!Object.hasOwn(values, index)) throw new RangeError(`${name} must contain an explicit value at every index.`);
  }
}
function numericArray(values, name, minimumLength = 1, maximumLength = 12) {
  denseArray(values, name, minimumLength, maximumLength);
  values.forEach(value => bounded(value, name, -1e6, 1e6));
}
export function combinations(values, count) {
  denseArray(values, 'Subset source', 0, 12);
  integer(count, 'Subset size', 0, values.length);
  const result = [];
  const collect = (start, partial) => {
    if (partial.length === count) {
      result.push(partial);
      return;
    }
    for (let index = start; index <= values.length - (count - partial.length); index += 1) {
      collect(index + 1, [...partial, values[index]]);
    }
  };
  collect(0, []);
  return result;
}
export function finiteMoments(values, probabilities = null) {
  denseArray(values, 'Finite distribution', 1, 10000);
  values.forEach(value => bounded(value, 'Distribution value', -1e9, 1e9));
  const weights = probabilities === null ? values.map(() => 1 / values.length) : probabilities;
  denseArray(weights, 'Probabilities', 1, 10000);
  if (weights.length !== values.length) throw new RangeError('Values and probabilities must align.');
  weights.forEach(value => bounded(value, 'Probability', 0, 1));
  const total = weights.reduce((sum, value) => sum + value, 0);
  if (Math.abs(total - 1) > 1e-12) throw new RangeError('Probabilities must sum to one.');
  // Normalize the accepted rounding tolerance and center before adding a large baseline.
  // A constant supported law must retain exactly zero spread.
  const anchor = values[weights.findIndex(weight => weight > 0)];
  const meanOffset = values.reduce((sum, value, index) => sum + (value - anchor) * (weights[index] / total), 0);
  const mean = anchor + meanOffset;
  const variance = values.reduce((sum, value, index) => {
    const residual = (value - anchor) - meanOffset;
    const contribution = (weights[index] / total) * residual ** 2;
    if (weights[index] > 0 && residual !== 0 && contribution === 0) {
      throw new RangeError('A nonzero variance contribution is below this model’s arithmetic range. Rescale the values.');
    }
    return sum + contribution;
  }, 0);
  const supportedValues = values.filter((_, index) => weights[index] > 0);
  if (variance === 0 && supportedValues.some(value => value !== supportedValues[0])) {
    throw new RangeError('The nonzero variance is below this model’s arithmetic range. Rescale the values.');
  }
  return {
    mean,
    variance
  };
}
function average(values) {
  return values.reduce((sum, value) => sum + value, 0) / values.length;
}
export function finiteSampleState(population = samplingPopulation, frame = null, size = 2) {
  numericArray(population, 'Population');
  const eligible = frame === null ? population.map((_, index) => index) : frame;
  denseArray(eligible, 'Frame', 1, population.length);
  if (new Set(eligible).size !== eligible.length) throw new RangeError('The frame must contain distinct population indices.');
  eligible.forEach(index => integer(index, 'Frame index', 0, population.length - 1));
  integer(size, 'Sample size', 1, eligible.length);
  const samples = combinations(eligible, size).map(indices => ({
    indices,
    mean: average(indices.map(index => population[index]))
  }));
  const target = average(population);
  const frameMean = average(eligible.map(index => population[index]));
  const moments = finiteMoments(samples.map(sample => sample.mean));
  const frameVariance = eligible.length === 1 ? null : eligible.reduce((sum, index) => sum + (population[index] - frameMean) ** 2, 0) / (eligible.length - 1);
  const formulaVariance = eligible.length === 1 ? 0 : (1 - size / eligible.length) * frameVariance / size;
  const bias = frameMean - target;
  const squaredBias = bias ** 2;
  if (bias !== 0 && squaredBias === 0) throw new RangeError('The nonzero squared bias is below this model’s arithmetic range. Rescale the values.');
  return {
    population: [...population],
    eligible: [...eligible],
    size,
    samples,
    target,
    frameMean,
    frameVariance,
    expectation: moments.mean,
    variance: moments.variance,
    formulaVariance,
    bias,
    mse: formulaVariance + squaredBias
  };
}
export function inclusionDesignState(mode = 'unequal', values = [2, 4, 8, 10]) {
  numericArray(values, 'Unit values', 4, 4);
  const presets = {
    unequal: [4, 2, 1, 1, 1, 1],
    equal: [1, 1, 1, 1, 1, 1],
    uncovered: [1, 0, 0, 0, 0, 0]
  };
  if (!Object.hasOwn(presets, mode)) throw new RangeError('Unknown inclusion design.');
  const subsets = combinations([0, 1, 2, 3], 2);
  const total = presets[mode].reduce((sum, value) => sum + value, 0);
  const probabilities = presets[mode].map(weight => weight / total);
  const inclusion = values.map((_, unit) => subsets.reduce((sum, subset, index) => sum + (subset.includes(unit) ? probabilities[index] : 0), 0));
  const covered = inclusion.every(probability => probability > 0);
  const samples = subsets.map((indices, index) => {
    const possible = probabilities[index] > 0;
    const contributions = possible ? indices.map(unit => values[unit] / inclusion[unit]) : null;
    return {
      indices,
      probability: probabilities[index],
      raw: average(indices.map(unit => values[unit])),
      contributions,
      ht: covered ? indices.reduce((sum, unit) => sum + values[unit] / inclusion[unit], 0) / values.length : null,
      ratio: covered ? indices.reduce((sum, unit) => sum + values[unit] / inclusion[unit], 0) / indices.reduce((sum, unit) => sum + 1 / inclusion[unit], 0) : null
    };
  });
  return {
    mode,
    values: [...values],
    inclusion,
    covered,
    samples,
    target: average(values),
    raw: finiteMoments(samples.map(sample => sample.raw), probabilities),
    ht: covered ? finiteMoments(samples.map(sample => sample.ht), probabilities) : null,
    ratio: covered ? finiteMoments(samples.map(sample => sample.ratio), probabilities) : null
  };
}
export function groupedMeasurementState(units = 4, repeats = 4, unitVariance = 4, readingVariance = 1, bias = 0) {
  integer(units, 'Independent units', 1, 16);
  integer(repeats, 'Readings per unit', 1, 16);
  bounded(unitVariance, 'Unit variance', 0, 9);
  bounded(readingVariance, 'Reading variance', 0, 9);
  bounded(bias, 'Fixed bias', -3, 3);
  const unitComponent = unitVariance / units;
  const readingComponent = readingVariance / (units * repeats);
  if (unitVariance > 0 && unitComponent === 0 || readingVariance > 0 && readingComponent === 0 || bias !== 0 && bias ** 2 === 0) {
    throw new RangeError("A nonzero variance or squared bias is below this model’s arithmetic range. Rescale the inputs.");
  }
  const variance = unitComponent + readingComponent;
  const individualVariance = unitVariance + readingVariance;
  return {
    units,
    repeats,
    unitVariance,
    readingVariance,
    bias,
    readings: units * repeats,
    unitComponent,
    readingComponent,
    variance,
    standardError: Math.sqrt(variance),
    mse: variance + bias ** 2,
    correlation: individualVariance === 0 ? null : unitVariance / individualVariance,
    independentVariance: individualVariance / (units * repeats),
    designEffect: individualVariance === 0 ? null : (unitVariance * repeats + readingVariance) / individualVariance
  };
}
export function assignmentState(design = 'complete', effect = 'constant', baseline = assignmentBaselines) {
  numericArray(baseline, 'Baseline outcomes', 6, 6);
  if (!['complete', 'prognostic', 'mixed'].includes(design)) throw new RangeError('Unknown assignment rule.');
  if (!['constant', 'heterogeneous'].includes(effect)) throw new RangeError('Unknown effect rule.');
  const effects = effect === 'constant' ? [2, 2, 2, 2, 2, 2] : [0, 2, 4, 0, 2, 4];
  const treated = baseline.map((value, index) => value + effects[index]);
  const blocks = design === 'prognostic' ? [[0, 1], [2, 3], [4, 5]] : design === 'mixed' ? [[0, 5], [1, 4], [2, 3]] : null;
  const allocations = blocks ? Array.from({
    length: 8
  }, (_, mask) => blocks.map((pair, index) => pair[mask >> index & 1]).sort((a, b) => a - b)) : combinations([0, 1, 2, 3, 4, 5], 3);
  const states = allocations.map(indices => {
    const observed = baseline.map((value, index) => indices.includes(index) ? treated[index] : value);
    const treatedMean = average(indices.map(index => observed[index]));
    const controlMean = average(observed.filter((_, index) => !indices.includes(index)));
    return {
      indices,
      observed,
      treatedMean,
      controlMean,
      difference: treatedMean - controlMean
    };
  });
  const moments = finiteMoments(states.map(state => state.difference));
  const variance = values => values.reduce((sum, value) => sum + (value - average(values)) ** 2, 0) / 5;
  const neymanVariance = variance(treated) / 3 + variance(baseline) / 3 - variance(effects) / 6;
  return {
    design,
    effect,
    baseline: [...baseline],
    treated,
    effects,
    blocks,
    states,
    target: average(effects),
    expectation: moments.mean,
    variance: moments.variance,
    neymanVariance: design === 'complete' ? neymanVariance : null
  };
}
export function factorialState(interaction = 4, highBShare = .5) {
  bounded(interaction, 'Interaction', -6, 6);
  bounded(highBShare, 'High-B population share', 0, 1);
  const cells = [[10, 12], [9, 11 + interaction]];
  return {
    interaction,
    highBShare,
    cells,
    lowBEffect: 2,
    highBEffect: 2 + interaction,
    averageAEffect: 2 + highBShare * interaction
  };
}
export function boundedMissingMean(values, lower = 0, upper = 10) {
  denseArray(values, 'Assigned outcomes (use null for a missing outcome)', 1, 1000);
  bounded(lower, 'Lower score bound', -1e6, 1e6);
  bounded(upper, 'Upper score bound', lower, 1e6);
  const recorded = values.filter(value => value !== null);
  recorded.forEach(value => bounded(value, 'Recorded score', lower, upper));
  const sum = recorded.reduce((total, value) => total + value, 0);
  const missing = values.length - recorded.length;
  return {
    count: values.length,
    observed: recorded.length,
    missing,
    observedMean: recorded.length ? sum / recorded.length : null,
    lower: (sum + missing * lower) / values.length,
    upper: (sum + missing * upper) / values.length
  };
}
export function formatSampling(value, digits = 5) {
  if (value === null || value === undefined) return 'undefined';
  if (!Number.isFinite(value)) return String(value);
  if (value === 0) return '0';
  if (Math.abs(value) < 10 ** -digits) return value.toExponential(2);
  return Number(value.toFixed(digits)).toString();
}
