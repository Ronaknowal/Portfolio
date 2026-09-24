// Bounded probability teaching models. Native examples and independent oracles
// verify these states; none is an empirical dataset or a browser RNG.
function probability(value, name) {
  if (!Number.isFinite(value) || value < 0 || value > 1) {
    throw new RangeError(`${name} must be a finite probability from 0 to 1.`);
  }
}
function integer(value, minimum, maximum, name) {
  if (!Number.isInteger(value) || value < minimum || value > maximum) {
    throw new RangeError(`${name} must be an integer from ${minimum} to ${maximum}.`);
  }
}
function bounded(value, minimum, maximum, name) {
  if (!Number.isFinite(value) || value < minimum || value > maximum) {
    throw new RangeError(`${name} must be finite and between ${minimum} and ${maximum}.`);
  }
}
export function probabilityNumber(value) {
  if (value === null) return 'undefined: zero evidence';
  if (!Number.isFinite(value)) throw new RangeError('A finite displayed value is required.');
  if (value === 0) return '0';
  if (value > 0.99999 && value < 1) return `1 − ${(1 - value).toExponential(3)}`;
  if (Math.abs(value) < 0.0001 || Math.abs(value) >= 100000) return value.toExponential(3);
  return Number(value.toFixed(6)).toString();
}
export function eventConditionState(selectedEvent = [2, 4, 6], condition = [4, 5, 6]) {
  for (const values of [selectedEvent, condition]) {
    if (!Array.isArray(values) || new Set(values).size !== values.length) {
      throw new RangeError('An event is a set of distinct face numbers.');
    }
    values.forEach(value => integer(value, 1, 6, 'Face'));
  }
  const eventSet = new Set(selectedEvent);
  const conditionSet = new Set(condition);
  const outcomes = Array.from({
    length: 6
  }, (_, index) => ({
    value: index + 1,
    inEvent: eventSet.has(index + 1),
    inCondition: conditionSet.has(index + 1),
    mass: 1 / 6
  }));
  const intersection = outcomes.filter(outcome => outcome.inEvent && outcome.inCondition).length;
  const union = outcomes.filter(outcome => outcome.inEvent || outcome.inCondition).length;
  return {
    outcomes,
    eventProbability: eventSet.size / 6,
    conditionProbability: conditionSet.size / 6,
    intersectionProbability: intersection / 6,
    unionProbability: union / 6,
    conditionalProbability: conditionSet.size ? intersection / conditionSet.size : null,
    independent: intersection * 6 === eventSet.size * conditionSet.size,
    disjoint: intersection === 0
  };
}
export function bayesPopulationState(prior = 0.01, sensitivity = 0.95, falsePositiveRate = 0.1) {
  probability(prior, 'Prior');
  probability(sensitivity, 'Sensitivity');
  probability(falsePositiveRate, 'False-positive rate');
  const cells = [{
    hypothesis: true,
    positive: true,
    mass: prior * sensitivity
  }, {
    hypothesis: true,
    positive: false,
    mass: prior * (1 - sensitivity)
  }, {
    hypothesis: false,
    positive: true,
    mass: (1 - prior) * falsePositiveRate
  }, {
    hypothesis: false,
    positive: false,
    mass: (1 - prior) * (1 - falsePositiveRate)
  }];
  const positiveMass = cells[0].mass + cells[2].mass;
  const negativeMass = cells[1].mass + cells[3].mass;
  return {
    prior,
    sensitivity,
    falsePositiveRate,
    cells,
    positiveMass,
    negativeMass,
    positivePosterior: positiveMass === 0 ? null : cells[0].mass / positiveMass,
    negativePosterior: negativeMass === 0 ? null : cells[1].mass / negativeMass,
    expectedCounts: cells.map(cell => cell.mass * 100000)
  };
}

// With probability copyShare, sample one indicator and copy it. Otherwise sample
// two conditionally independent indicators. Both single-result marginals stay p.
function pairedOutcomes(p, copyShare) {
  return [p * (copyShare + (1 - copyShare) * p), (1 - copyShare) * p * (1 - p), (1 - copyShare) * p * (1 - p), (1 - p) * (copyShare + (1 - copyShare) * (1 - p))];
}
export function pairedEvidenceState(prior = 0.01, sensitivity = 0.95, falsePositiveRate = 0.1, copyShare = 0) {
  const population = bayesPopulationState(prior, sensitivity, falsePositiveRate);
  probability(copyShare, 'Copy share');
  const givenHypothesis = pairedOutcomes(sensitivity, copyShare);
  const givenAlternative = pairedOutcomes(falsePositiveRate, copyShare);
  const independentHypothesis = pairedOutcomes(sensitivity, 0);
  const independentAlternative = pairedOutcomes(falsePositiveRate, 0);
  const patterns = ['++', '+−', '−+', '−−'].map((label, index) => {
    const hypothesisMass = prior * givenHypothesis[index];
    const alternativeMass = (1 - prior) * givenAlternative[index];
    const evidenceMass = hypothesisMass + alternativeMass;
    const incorrectlyIndependentMass = prior * independentHypothesis[index] + (1 - prior) * independentAlternative[index];
    return {
      label,
      givenHypothesis: givenHypothesis[index],
      givenAlternative: givenAlternative[index],
      evidenceMass,
      posterior: evidenceMass === 0 ? null : hypothesisMass / evidenceMass,
      independentPosterior: incorrectlyIndependentMass === 0 ? null : prior * independentHypothesis[index] / incorrectlyIndependentMass
    };
  });
  return {
    ...population,
    copyShare,
    patterns
  };
}
export function chooseCount(n, k) {
  integer(n, 0, 30, 'Population size');
  if (!Number.isInteger(k)) throw new RangeError('Subset size must be an integer.');
  if (k < 0 || k > n) return 0;
  const smaller = Math.min(k, n - k);
  let value = 1;
  for (let index = 1; index <= smaller; index += 1) value = value * (n - smaller + index) / index;
  return Math.round(value);
}
export function urnCountState(marked = 2, draws = 3, replacement = false, threshold = 1) {
  integer(marked, 0, 6, 'Marked objects');
  integer(draws, 0, 6, 'Draw count');
  integer(threshold, 0, 6, 'Selected count');
  if (typeof replacement !== 'boolean') throw new RangeError('Replacement must be true or false.');
  const population = 6;
  const p = marked / population;
  let cumulative = 0;
  const masses = Array.from({
    length: draws + 1
  }, (_, successes) => {
    const mass = replacement ? chooseCount(draws, successes) * p ** successes * (1 - p) ** (draws - successes) : chooseCount(marked, successes) * chooseCount(population - marked, draws - successes) / chooseCount(population, draws);
    cumulative += mass;
    return {
      successes,
      mass,
      cumulative
    };
  });
  const selected = masses.find(item => item.successes === threshold);
  return {
    population,
    marked,
    draws,
    replacement,
    threshold,
    masses,
    selectedMass: selected?.mass ?? 0,
    cumulativeMass: masses.filter(item => item.successes <= threshold).reduce((sum, item) => sum + item.mass, 0),
    mean: draws * p,
    variance: draws * p * (1 - p) * (replacement ? 1 : (population - draws) / (population - 1))
  };
}

// X=0 with probability atom; otherwise X is uniform on (0,width).
// The continuous density excludes the atom; its integral is 1-atom.
export function mixedDelayState(width = 0.2, atom = 0, leftFraction = 0.25, rightFraction = 0.75, unit = 'seconds') {
  bounded(width, 0.1, 2, 'Continuous interval width');
  probability(atom, 'Atom probability');
  probability(leftFraction, 'Left fraction');
  probability(rightFraction, 'Right fraction');
  if (leftFraction > rightFraction) throw new RangeError('Left endpoint must not exceed the right endpoint.');
  if (!['seconds', 'milliseconds'].includes(unit)) throw new RangeError('Choose seconds or milliseconds.');
  const scale = unit === 'seconds' ? 1 : 1000;
  const left = leftFraction * width;
  const right = rightFraction * width;
  const cdf = x => x < 0 ? 0 : atom + (1 - atom) * Math.min(x / width, 1);
  const leftLimit = left === 0 ? 0 : cdf(left);
  const continuousMass = (1 - atom) * (rightFraction - leftFraction);
  const includedAtom = left === 0 ? atom : 0;
  return {
    width: width * scale,
    atom,
    unit,
    left: left * scale,
    right: right * scale,
    density: (1 - atom) / (width * scale),
    continuousMass,
    includedAtom,
    intervalMass: continuousMass + includedAtom,
    cdfLeftLimit: leftLimit,
    cdfRight: cdf(right),
    pointMassAtZero: atom,
    mean: (1 - atom) * width * scale / 2,
    variance: ((1 - atom) / 3 - (1 - atom) ** 2 / 4) * (width * scale) ** 2,
    cdfSamples: Array.from({
      length: 41
    }, (_, index) => ({
      x: index / 40 * width * scale,
      value: atom + (1 - atom) * index / 40
    }))
  };
}

// Fixed interior uniforms couple the two rate settings reproducibly. The plot
// represents one disclosed realization, never a measured arrival stream.
export const arrivalUniforms = [0.18, 0.72, 0.41, 0.86, 0.09, 0.55, 0.28, 0.64, 0.33, 0.91, 0.12, 0.47];
export function arrivalWindowState(rate = 2, window = 1.5, quantile = 0.5) {
  bounded(rate, 0.25, 6, 'Rate per minute');
  bounded(window, 0, 2, 'Window in minutes');
  bounded(quantile, 0.01, 0.99, 'Quantile probability');
  const meanCount = rate * window;
  let mass = Math.exp(-meanCount);
  const masses = Array.from({
    length: 25
  }, (_, count) => {
    if (count > 0) mass *= meanCount / count;
    return {
      count,
      mass
    };
  });
  // Sum positive tail terms directly: subtracting a CDF from one would erase
  // very small nonzero tails. Here meanCount <= 12, so terms after 24 decrease.
  let tailTerm = mass * meanCount / 25;
  let tailBeyond24 = tailTerm;
  for (let count = 26; count <= 160; count += 1) {
    tailTerm *= meanCount / count;
    const next = tailBeyond24 + tailTerm;
    if (next === tailBeyond24) break;
    tailBeyond24 = next;
  }
  let time = 0;
  const arrivals = arrivalUniforms.map((uniform, index) => {
    const wait = -Math.log1p(-uniform) / rate;
    time += wait;
    return {
      event: index + 1,
      uniform,
      wait,
      time,
      inWindow: time <= window
    };
  });
  return {
    rate,
    window,
    quantile,
    meanCount,
    masses,
    tailBeyond24,
    noArrivals: Math.exp(-meanCount),
    atLeastOne: -Math.expm1(-meanCount),
    quantileWait: -Math.log1p(-quantile) / rate,
    meanWait: 1 / rate,
    arrivals,
    displayedArrivalsInWindow: arrivals.filter(event => event.inWindow).length,
    realizationTruncated: arrivals.at(-1).time <= window
  };
}
