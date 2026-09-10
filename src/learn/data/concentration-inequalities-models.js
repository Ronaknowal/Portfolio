function integerIn(value, low, high, name) {
  if (!Number.isInteger(value) || value < low || value > high) {
    throw new RangeError(`${name}: choose an integer from ${low} to ${high}.`);
  }
}
function logSumExp(values) {
  if (!values.length) return -Infinity;
  const maximum = Math.max(...values);
  if (maximum === -Infinity) return -Infinity;
  return maximum + Math.log(values.reduce((sum, value) => sum + Math.exp(value - maximum), 0));
}
function logChoose(total, selected) {
  if (selected < 0 || selected > total) return -Infinity;
  let result = 0;
  for (let index = 1; index <= Math.min(selected, total - selected); index++) {
    result += Math.log(total - index + 1) - Math.log(index);
  }
  return result;
}
export function binomialLaw(count, probabilityPercent) {
  integerIn(count, 1, 2000, 'Observation count');
  integerIn(probabilityPercent, 0, 100, 'Success percentage');
  const probability = probabilityPercent / 100;
  if (probability === 0 || probability === 1) {
    return Array.from({
      length: count + 1
    }, (_, successes) => successes === count * probability ? 0 : -Infinity);
  }
  const raw = [];
  let combinationLog = 0;
  for (let successes = 0; successes <= count; successes++) {
    if (successes) combinationLog += Math.log(count - successes + 1) - Math.log(successes);
    raw.push(combinationLog + successes * Math.log(probability) + (count - successes) * Math.log1p(-probability));
  }
  // Normalize accumulated floating-point log weights, not observed frequencies.
  const normalization = logSumExp(raw);
  return raw.map(value => value - normalization);
}
export function formatLogProbability(logProbability, logComplement) {
  if (logComplement !== undefined && Number.isFinite(logComplement) && logComplement < -14) {
    const remainder = logComplement < -700 ? `10^(${(logComplement / Math.LN10).toFixed(3)})` : Math.exp(logComplement).toExponential(3);
    return `1 − ${remainder}`;
  }
  if (logProbability === -Infinity) return '0';
  if (!Number.isFinite(logProbability) || logProbability > 1e-10) throw new RangeError('Expected a log probability at most zero.');
  if (logProbability >= 0) return '1';
  if (logProbability > -1e-6) return `1 − ${(-Math.expm1(logProbability)).toExponential(3)}`;
  if (logProbability < -700) return `≈ 10^(${(logProbability / Math.LN10).toFixed(3)})`;
  const probability = Math.exp(logProbability);
  return probability < .0001 ? probability.toExponential(3) : probability.toPrecision(5);
}
export function countTail(count = 100, probabilityPercent = 10, threshold = 20) {
  integerIn(threshold, 0, count + 1, 'Inclusive count threshold');
  const law = binomialLaw(count, probabilityPercent);
  const probability = probabilityPercent / 100;
  const mean = count * probability;
  const variance = count * probability * (1 - probability);
  const deviation = (100 * threshold - count * probabilityPercent) / 100;
  const exactComplementLog = logSumExp(law.slice(0, threshold));
  const summedTailLog = logSumExp(law.slice(threshold));
  const exactLog = threshold === 0 ? 0 : summedTailLog > -Math.LN2 ? Math.log1p(-Math.exp(Math.min(0, exactComplementLog))) : Math.min(0, summedTailLog);
  const supportDetermines = threshold === 0 || threshold > count || probability === 0 || probability === 1;
  let bounds;
  if (supportDetermines) {
    bounds = {
      markov: exactLog,
      chebyshev: exactLog,
      hoeffding: exactLog,
      bernstein: exactLog,
      multiplicative: exactLog,
      kl: exactLog
    };
  } else if (deviation <= 0) {
    bounds = {
      markov: Math.min(0, Math.log(mean / threshold)),
      chebyshev: 0,
      hoeffding: 0,
      bernstein: 0,
      multiplicative: 0,
      kl: 0
    };
  } else {
    const relative = deviation / mean;
    const proportion = threshold / count;
    const divergence = proportion * Math.log(proportion / probability) + (proportion === 1 ? 0 : (1 - proportion) * Math.log((1 - proportion) / (1 - probability)));
    bounds = {
      markov: Math.min(0, Math.log(mean / threshold)),
      // Two-sided Chebyshev is also a valid (possibly loose) upper-tail bound.
      chebyshev: Math.min(0, Math.log(variance / deviation ** 2)),
      hoeffding: -2 * deviation ** 2 / count,
      bernstein: -(deviation ** 2) / (2 * (variance + deviation / 3)),
      multiplicative: -mean * ((1 + relative) * Math.log1p(relative) - relative),
      kl: -count * divergence
    };
  }
  return {
    count,
    probabilityPercent,
    probability,
    mean,
    variance,
    threshold,
    deviation,
    law,
    exactLog,
    exactComplementLog,
    bounds,
    supportDetermines
  };
}
export function exponentialWitness(lambda = 1) {
  if (!Number.isFinite(lambda) || lambda < 0 || lambda > 2) throw new RangeError('Use lambda between zero and two.');
  const count = 20,
    probability = .25,
    threshold = 10;
  const law = binomialLaw(count, 25);
  const logObjective = value => count * Math.log1p(probability * Math.expm1(value)) - value * threshold;
  const hoeffdingObjective = value => count * value ** 2 / 8 - value * (threshold - count * probability);
  const optimalLambda = Math.log(3);
  return {
    count,
    probability,
    threshold,
    lambda,
    optimalLambda,
    logBound: logObjective(lambda),
    logHoeffdingEnvelope: hoeffdingObjective(lambda),
    exactLog: logSumExp(law.slice(threshold)),
    contributions: law.map((logWeight, successes) => ({
      successes,
      tail: successes >= threshold ? Math.exp(logWeight) : 0,
      witness: Math.exp(logWeight + lambda * (successes - threshold))
    })),
    curve: Array.from({
      length: 101
    }, (_, index) => {
      const value = index / 50;
      return {
        lambda: value,
        exact: logObjective(value),
        envelope: hoeffdingObjective(value)
      };
    })
  };
}
export function precisionBudget(count = 100, delta = .05, width = 1, normalizedVariance = .09, epsilon = .1) {
  integerIn(count, 1, 1000000, 'Observation count');
  if (![delta, width, normalizedVariance, epsilon].every(Number.isFinite) || delta <= 0 || delta >= 1 || width <= 0 || epsilon <= 0 || normalizedVariance < 0 || normalizedVariance > .25) {
    throw new RangeError('Use 0<delta<1, positive range/tolerance and a normalized variance bound from 0 to .25.');
  }
  const logarithm = Math.log(2 / delta);
  const variance = normalizedVariance * width ** 2;
  const radii = n => {
    const linear = width * logarithm / (3 * n);
    const varianceTerm = 2 * variance * logarithm / n;
    return {
      n,
      hoeffding: width * Math.sqrt(logarithm / (2 * n)),
      bernsteinExact: linear + Math.sqrt(varianceTerm + linear ** 2),
      bernsteinRelaxed: Math.sqrt(varianceTerm) + 2 * linear,
      chebyshev: Math.sqrt(variance / (n * delta)),
      linear,
      varianceTerm
    };
  };
  const needed = Math.ceil(width ** 2 * logarithm / (2 * epsilon ** 2));
  return {
    count,
    delta,
    width,
    variance,
    normalizedVariance,
    epsilon,
    logarithm,
    needed,
    ...radii(count),
    curve: Array.from({
      length: 101
    }, (_, index) => radii(10 ** (1 + index / 50)))
  };
}
export function samplingComparison(count = 20, epsilonPercent = 20) {
  integerIn(count, 1, 20, 'Sample count');
  integerIn(epsilonPercent, 1, 80, 'Absolute tolerance percentage points');
  const population = 20,
    marked = 5,
    probability = marked / population;
  const independent = binomialLaw(count, 25);
  const copied = Array.from({
    length: count + 1
  }, (_, successes) => successes === 0 ? Math.log1p(-probability) : successes === count ? Math.log(probability) : -Infinity);
  const withoutReplacement = Array.from({
    length: count + 1
  }, (_, successes) => {
    const numerator = logChoose(marked, successes) + logChoose(population - marked, count - successes);
    return numerator - logChoose(population, count);
  });
  const inEvent = successes => Math.abs(100 * successes - 25 * count) >= epsilonPercent * count;
  const failure = law => {
    const included = logSumExp(law.filter((_, successes) => inEvent(successes)));
    const excluded = logSumExp(law.filter((_, successes) => !inEvent(successes)));
    if (excluded === -Infinity) return 0;
    return included > -Math.LN2 ? Math.log1p(-Math.exp(Math.min(0, excluded))) : Math.min(0, included);
  };
  return {
    count,
    epsilonPercent,
    population,
    marked,
    probability,
    nominalHoeffdingLog: Math.min(0, Math.log(2) - 2 * count * (epsilonPercent / 100) ** 2),
    laws: [{
      id: 'independent',
      label: 'Independent, with replacement',
      logMass: independent,
      logFailure: failure(independent),
      meanVariance: probability * (1 - probability) / count,
      guarantee: 'Hoeffding: independent bounded draws'
    }, {
      id: 'copied',
      label: 'One draw copied n times',
      logMass: copied,
      logFailure: failure(copied),
      meanVariance: probability * (1 - probability),
      guarantee: 'Independent-draw bound does not apply'
    }, {
      id: 'without',
      label: 'Uniform subset, without replacement',
      logMass: withoutReplacement,
      logFailure: failure(withoutReplacement),
      meanVariance: probability * (1 - probability) / count * (population - count) / (population - 1),
      guarantee: 'Same bound justified by a separate convex-order theorem'
    }],
    eventCounts: Array.from({
      length: count + 1
    }, (_, successes) => successes).filter(inEvent)
  };
}
export function familyBudget(count = 100, checks = 100, delta = .05, dependence = 'independent') {
  integerIn(count, 10, 400, 'Observations per check');
  integerIn(checks, 1, 100, 'Number of fixed checks');
  if (!Number.isFinite(delta) || delta <= 0 || delta >= 1 || !['independent', 'identical'].includes(dependence)) throw new RangeError('Choose a valid error budget and failure-event relation.');
  const law = binomialLaw(count, 50);
  const oneCheckLog = budget => {
    const boundary = 2 * count * Math.log(2 / budget);
    return Math.min(0, logSumExp(law.filter((_, successes) => (2 * successes - count) ** 2 >= boundary)));
  };
  const logFamily = (logSingle, number) => {
    if (dependence === 'identical') return logSingle;
    if (logSingle === -Infinity) return -Infinity;
    return Math.log(-Math.expm1(number * Math.log1p(-Math.exp(logSingle))));
  };
  const naiveLog = oneCheckLog(delta),
    allocatedLog = oneCheckLog(delta / checks);
  return {
    count,
    checks,
    delta,
    dependence,
    naiveLog,
    allocatedLog,
    naiveRadius: Math.sqrt(Math.log(2 / delta) / (2 * count)),
    allocatedRadius: Math.sqrt(Math.log(2 * checks / delta) / (2 * count)),
    naiveFamilyLog: logFamily(naiveLog, checks),
    allocatedFamilyLog: logFamily(allocatedLog, checks),
    naiveCertificate: Math.min(1, checks * delta),
    allocatedCertificate: delta,
    curve: Array.from({
      length: 100
    }, (_, index) => {
      const number = index + 1;
      return {
        checks: number,
        naive: Math.exp(logFamily(naiveLog, number)),
        allocated: Math.exp(logFamily(oneCheckLog(delta / number), number)),
        nominal: delta
      };
    })
  };
}
