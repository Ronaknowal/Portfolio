// Finite, synthetic decision models. Every probability is a model input,
// not an estimate or a claim about an actual production process.
function finite(value, name) {
  if (!Number.isFinite(value)) throw new RangeError(`${name} must be finite.`);
  return value;
}
function probability(value, name = 'Probability') {
  finite(value, name);
  if (value < 0 || value > 1) throw new RangeError(`${name} must be in [0, 1].`);
  return value;
}
function dense(values, name, minimum = 1) {
  if (!Array.isArray(values) || values.length < minimum || Array.from({
    length: values.length
  }, (_, index) => index).some(index => !Object.hasOwn(values, index))) {
    throw new TypeError(`${name} must be a dense array with at least ${minimum} entries.`);
  }
  return values;
}
function masses(values) {
  dense(values, 'Masses').forEach(value => probability(value, 'Mass'));
  const total = values.reduce((sum, value) => sum + value, 0);
  if (Math.abs(total - 1) > 2e-12) throw new RangeError('Masses must sum to one.');
  // Normalize only admitted roundoff, never an arbitrary unnormalized model.
  return values.map(value => value / total);
}

// Finite decision inputs are interpreted as their written decimal Number values.
// Exact fractions preserve co-optimal finite policies, without turning a
// floating tolerance into a mathematical tie.
function rational(numerator, denominator = 1n) {
  if (denominator === 0n) throw new RangeError('A rational denominator cannot be zero.');
  if (denominator < 0n) {
    numerator = -numerator;
    denominator = -denominator;
  }
  let left = numerator < 0n ? -numerator : numerator;
  let right = denominator;
  while (right !== 0n) [left, right] = [right, left % right];
  const divisor = left || 1n;
  return {
    numerator: numerator / divisor,
    denominator: denominator / divisor
  };
}
function decimal(value) {
  finite(value, 'Decimal input');
  const [coefficient, exponentText = '0'] = value.toString().toLowerCase().split('e');
  const places = coefficient.includes('.') ? coefficient.length - coefficient.indexOf('.') - 1 : 0;
  const exponent = Number(exponentText) - places;
  const integer = BigInt(coefficient.replace('.', ''));
  return exponent >= 0 ? rational(integer * 10n ** BigInt(exponent)) : rational(integer, 10n ** BigInt(-exponent));
}
function plus(left, right) {
  return rational(left.numerator * right.denominator + right.numerator * left.denominator, left.denominator * right.denominator);
}
function times(left, right) {
  return rational(left.numerator * right.numerator, left.denominator * right.denominator);
}
function compare(left, right) {
  const difference = left.numerator * right.denominator - right.numerator * left.denominator;
  return difference < 0n ? -1 : difference > 0n ? 1 : 0;
}
function divide(left, right) {
  if (right.numerator <= 0n) throw new RangeError('This probability divisor must be positive.');
  return rational(left.numerator * right.denominator, left.denominator * right.numerator);
}
function complement(value) {
  return plus(rational(1n), rational(-value.numerator, value.denominator));
}
function exactMinimum(values) {
  const minimum = values.reduce((best, value) => compare(value, best) < 0 ? value : best);
  return {
    minimum: rationalNumber(minimum),
    actions: values.flatMap((value, index) => compare(value, minimum) === 0 ? [index] : [])
  };
}
function exactMasses(values) {
  masses(values);
  const exact = values.map(decimal);
  const total = exact.reduce(plus, rational(0n));
  return exact.map(value => divide(value, total));
}
function rationalNumber(value) {
  if (value.numerator === 0n) return 0;
  const negative = value.numerator < 0n;
  const numerator = (negative ? -value.numerator : value.numerator).toString();
  const denominator = value.denominator.toString();
  const mantissa = Number(`0.${numerator.slice(0, 17)}`) / Number(`0.${denominator.slice(0, 17)}`);
  const result = Number(`${negative ? '-' : ''}${mantissa}e${numerator.length - denominator.length}`);
  if (!Number.isFinite(result)) throw new RangeError('Exact expected loss exceeds the display arithmetic range.');
  if (result === 0) throw new RangeError('A nonzero exact result is below the display arithmetic range.');
  return result;
}
function selectedCost(probabilities, selected, handling, damage) {
  return probabilities.reduce((sum, p, index) => plus(sum, selected.includes(index) ? decimal(handling) : times(decimal(damage), decimal(p))), rational(0n));
}
export function formatDecisionNumber(value, digits = 3) {
  if (value === null) return 'undefined';
  if (!Number.isFinite(value)) return String(value);
  if (value === 0) return '0';
  if (Math.abs(value) < 10 ** -digits || Math.abs(value) >= 1e7) return value.toExponential(2);
  return Number(value.toFixed(digits)).toString();
}
export function evaluateFiniteActions(stateMasses, lossRows) {
  const exactWeights = exactMasses(stateMasses);
  const weights = exactWeights.map(rationalNumber);
  dense(lossRows, 'Loss rows').forEach(row => {
    dense(row, 'Loss row');
    if (row.length !== weights.length) throw new RangeError('Each row needs one loss per state.');
    row.forEach(value => finite(value, 'Loss'));
  });
  const exactContributions = lossRows.map(row => row.map((loss, index) => times(decimal(loss), exactWeights[index])));
  const contributions = exactContributions.map(row => row.map(rationalNumber));
  const exactRisks = exactContributions.map(row => row.reduce(plus, rational(0n)));
  const risks = exactRisks.map(rationalNumber);
  return {
    weights,
    contributions,
    risks,
    ...exactMinimum(exactRisks)
  };
}
export function binaryDecision(p, losses = [[0, 80], [10, 10]]) {
  probability(p);
  if (losses.length !== 2 || losses.some(row => row.length !== 2)) throw new RangeError('Use two actions and two states.');
  const result = evaluateFiniteActions([1 - p, p], losses);
  const exactP = decimal(p);
  const exactContributions = losses.map(row => [times(complement(exactP), decimal(row[0])), times(exactP, decimal(row[1]))]);
  const exactRisks = exactContributions.map(row => row.reduce(plus, rational(0n)));
  const intercept = plus(decimal(losses[1][0]), decimal(-losses[0][0]));
  const endDifference = plus(decimal(losses[1][1]), decimal(-losses[0][1]));
  const slope = plus(endDifference, rational(-intercept.numerator, intercept.denominator));
  const crossing = slope.numerator === 0n ? null : rationalNumber(rational(-intercept.numerator * slope.denominator, intercept.denominator * slope.numerator));
  return {
    ...result,
    weights: [rationalNumber(complement(exactP)), p],
    contributions: exactContributions.map(row => row.map(rationalNumber)),
    risks: exactRisks.map(rationalNumber),
    ...exactMinimum(exactRisks),
    crossing,
    identical: intercept.numerator === 0n && slope.numerator === 0n
  };
}
export function signalRuleRisks(priorHigh = 0.3) {
  probability(priorHigh);
  const prior = [1 - priorHigh, priorHigh];
  const exactPrior = [complement(decimal(priorHigh)), decimal(priorHigh)];
  const signalGivenState = [[0.8, 0.2], [0.2, 0.8]];
  const rules = [[0, 0], [1, 1], [0, 1], [1, 0]];
  const exactJoint = signalGivenState.map((row, state) => row.map(weight => times(decimal(weight), exactPrior[state])));
  const joint = exactJoint.map(row => row.map(rationalNumber));
  const ruleRisks = rules.map(rule => signalGivenState.map((row, state) => row.reduce((sum, weight, signal) => sum + weight * Number(rule[signal] !== state), 0)));
  const exactBayesRisks = ruleRisks.map(row => row.reduce((sum, risk, state) => plus(sum, times(decimal(risk), exactPrior[state])), rational(0n)));
  const bayesRisks = exactBayesRisks.map(rationalNumber);
  const signals = [0, 1].map(signal => {
    const exactTotal = plus(exactJoint[0][signal], exactJoint[1][signal]);
    const total = rationalNumber(exactTotal);
    const posterior = exactTotal.numerator === 0n ? null : rationalNumber(divide(exactJoint[1][signal], exactTotal));
    return {
      mass: total,
      posterior,
      risks: posterior === null ? null : [posterior, 1 - posterior]
    };
  });
  return {
    prior,
    rules,
    joint,
    ruleRisks,
    bayesRisks,
    signals,
    ...exactMinimum(exactBayesRisks)
  };
}
export function provisioningRisk(quantity, values = [0, 2, 10], weights = [0.2, 0.5, 0.3], under = 4, over = 1) {
  finite(quantity, 'Quantity');
  dense(values, 'Demand values').forEach(value => finite(value, 'Demand'));
  const normalized = masses(weights);
  if (values.length !== normalized.length) throw new RangeError('Demands and masses must align.');
  if (!(finite(under, 'Underage cost') > 0) || !(finite(over, 'Overage cost') > 0)) throw new RangeError('Costs must be positive.');
  const terms = values.map((value, index) => ({
    value,
    mass: normalized[index],
    shortage: Math.max(value - quantity, 0),
    surplus: Math.max(quantity - value, 0),
    contribution: normalized[index] * (under * Math.max(value - quantity, 0) + over * Math.max(quantity - value, 0))
  }));
  const risk = terms.reduce((sum, term) => sum + term.contribution, 0);
  if (!Number.isFinite(risk)) throw new RangeError('Provisioning risk exceeds the arithmetic range.');
  return {
    terms,
    risk,
    quantileLevel: under / (under + over)
  };
}
export function abstentionDecision(p, fallback = 0.6) {
  probability(p);
  if (finite(fallback, 'Fallback cost') < 0) throw new RangeError('Fallback cost must be nonnegative.');
  const exactRisks = [times(decimal(8), decimal(p)), times(decimal(2), complement(decimal(p))), decimal(fallback)];
  const risks = exactRisks.map(rationalNumber);
  return {
    risks,
    ...exactMinimum(exactRisks)
  };
}
export const inspectionProbabilities = [0.02, 0.06, 0.1, 0.2, 0.35, 0.6];
export function allocateInspections(probabilities = inspectionProbabilities, capacity = 2, handling = 10, damage = 80) {
  dense(probabilities, 'Item probabilities');
  if (probabilities.length > 12) throw new RangeError('This teaching allocation model supports at most 12 items.');
  probabilities.forEach(value => probability(value));
  if (!Number.isInteger(capacity) || capacity < 0 || capacity > probabilities.length) throw new RangeError('Capacity must be an integer between zero and the item count.');
  if (finite(handling, 'Handling cost') < 0 || finite(damage, 'Damage cost') < 0) throw new RangeError('Costs must be nonnegative.');
  const exactBenefits = probabilities.map(p => plus(times(decimal(damage), decimal(p)), decimal(-handling)));
  const benefits = exactBenefits.map(rationalNumber);
  // Deterministic documented tie-break: lower item index first. Do not spend a
  // slot on a zero benefit; other zero-benefit policies have the same loss.
  const selected = exactBenefits.map((benefit, index) => ({
    benefit,
    index
  })).filter(item => item.benefit.numerator > 0n).sort((left, right) => {
    const difference = right.benefit.numerator * left.benefit.denominator - left.benefit.numerator * right.benefit.denominator;
    return difference > 0n ? 1 : difference < 0n ? -1 : left.index - right.index;
  }).slice(0, capacity).map(item => item.index).sort((left, right) => left - right);
  const contributions = probabilities.map((p, index) => selected.includes(index) ? handling : damage * p);
  return {
    benefits,
    selected,
    contributions,
    risk: rationalNumber(selectedCost(probabilities, selected, handling, damage))
  };
}
export function informationValue(p = 0.08, sensitivity = 0.8, falsePositive = 0.1, testCost = 2) {
  [p, sensitivity, falsePositive].forEach(value => probability(value));
  if (finite(testCost, 'Test cost') < 0) throw new RangeError('Test cost must be nonnegative.');
  const exactP = decimal(p),
    exactSensitivity = decimal(sensitivity),
    exactFalsePositive = decimal(falsePositive);
  const exactJoints = [[times(complement(exactP), complement(exactFalsePositive)), times(exactP, complement(exactSensitivity))], [times(complement(exactP), exactFalsePositive), times(exactP, exactSensitivity)]];
  const branchOptima = [];
  const branches = exactJoints.map(exactJoint => {
    const exactMass = exactJoint.reduce(plus, rational(0n));
    const exactRisks = [times(decimal(80), exactJoint[1]), times(decimal(10), exactMass)];
    const best = exactMinimum(exactRisks);
    branchOptima.push(exactRisks[best.actions[0]]);
    const mass = rationalNumber(exactMass);
    return {
      joint: exactJoint.map(rationalNumber),
      mass,
      posterior: exactMass.numerator === 0n ? null : rationalNumber(divide(exactJoint[1], exactMass)),
      weightedRisks: exactRisks.map(rationalNumber),
      conditionalRisks: exactMass.numerator === 0n ? null : exactRisks.map(value => rationalNumber(divide(value, exactMass))),
      ...best,
      actions: exactMass.numerator === 0n ? [] : best.actions
    };
  });
  const release = times(decimal(80), exactP);
  const exactBaseline = compare(release, decimal(10)) < 0 ? release : decimal(10);
  const exactAfter = branchOptima.reduce(plus, rational(0n));
  const exactPerfect = times(decimal(10), exactP);
  const subtract = (left, right) => plus(left, rational(-right.numerator, right.denominator));
  return {
    branches,
    baseline: rationalNumber(exactBaseline),
    afterSignal: rationalNumber(exactAfter),
    total: rationalNumber(plus(exactAfter, decimal(testCost))),
    value: rationalNumber(subtract(exactBaseline, exactAfter)),
    perfect: rationalNumber(exactPerfect),
    perfectValue: rationalNumber(subtract(exactBaseline, exactPerfect))
  };
}
export function mixedStateRisk(weightA = 0.4) {
  probability(weightA, 'Weight on rule A');
  const risk = [4 * (1 - weightA), 6 * weightA];
  return {
    risk,
    worst: Math.max(...risk),
    deterministicWorst: [6, 4, 2.5],
    certificatePrior: [0.6, 0.4],
    certificateRisks: [2.4, 2.4, 2.5]
  };
}
export function utilityLottery(low = 50, high = 150, probabilityHigh = 0.5, sure = 95) {
  [low, high, sure].forEach(value => {
    if (finite(value, 'Final outcome') < 0) throw new RangeError('Square-root utility requires nonnegative outcomes.');
  });
  probability(probabilityHigh);
  const expectedUtility = (1 - probabilityHigh) * Math.sqrt(low) + probabilityHigh * Math.sqrt(high);
  const certaintyEquivalent = expectedUtility ** 2;
  const mean = (1 - probabilityHigh) * low + probabilityHigh * high;
  const hasPositiveSupport = low > 0 && probabilityHigh < 1 || high > 0 && probabilityHigh > 0;
  if (![expectedUtility, certaintyEquivalent, mean].every(Number.isFinite) || hasPositiveSupport && certaintyEquivalent === 0) {
    throw new RangeError('Utility or certainty equivalent is outside the supported arithmetic range.');
  }
  return {
    mean,
    expectedUtility,
    certaintyEquivalent,
    sureUtility: Math.sqrt(sure)
  };
}
export function finiteTailRisk(values = [0, 10, 100], weights = [0.8, 0.15, 0.05], alpha = 0.9) {
  dense(values, 'Loss values').forEach(value => finite(value, 'Loss'));
  const exactWeights = exactMasses(weights);
  if (values.length !== exactWeights.length) throw new RangeError('Losses and masses must align.');
  if (finite(alpha, 'Tail level') < 0 || alpha >= 1) throw new RangeError('Use 0 ≤ alpha < 1.');
  const exactAlpha = decimal(alpha),
    exactTailMass = complement(exactAlpha);
  const atoms = values.map((value, index) => ({
    value,
    mass: rationalNumber(exactWeights[index]),
    index
  })).filter(atom => exactWeights[atom.index].numerator > 0n).sort((left, right) => left.value - right.value || left.index - right.index);
  let cumulative = rational(0n);
  let valueAtRisk = null;
  for (const atom of atoms) {
    cumulative = plus(cumulative, exactWeights[atom.index]);
    if (alpha > 0 && valueAtRisk === null && compare(cumulative, exactAlpha) >= 0) valueAtRisk = atom.value;
  }
  let remaining = exactTailMass,
    numerator = rational(0n);
  const tail = [...atoms].reverse().map(atom => {
    const exactUsed = compare(exactWeights[atom.index], remaining) < 0 ? exactWeights[atom.index] : remaining;
    remaining = plus(remaining, rational(-exactUsed.numerator, exactUsed.denominator));
    numerator = plus(numerator, times(decimal(atom.value), exactUsed));
    return {
      ...atom,
      usedMass: rationalNumber(exactUsed)
    };
  });
  const cvar = rationalNumber(divide(numerator, exactTailMass));
  const mean = rationalNumber(atoms.reduce((sum, atom) => plus(sum, times(decimal(atom.value), exactWeights[atom.index])), rational(0n)));
  return {
    atoms,
    tail,
    alpha,
    tailMass: rationalNumber(exactTailMass),
    valueAtRisk,
    cvar,
    mean
  };
}
export function contingentInspection(testIndex = 3, testCost = 1, capacity = 2, probabilities = inspectionProbabilities) {
  const baseline = allocateInspections(probabilities, capacity);
  if (!Number.isInteger(testIndex) || testIndex < 0 || testIndex >= probabilities.length) throw new RangeError('Select an existing item.');
  if (finite(testCost, 'Test cost') < 0) throw new RangeError('Test cost must be nonnegative.');
  const branches = [0, 1].map(state => {
    const posterior = probabilities.map((value, index) => index === testIndex ? state : value);
    const mass = state === 1 ? probabilities[testIndex] : 1 - probabilities[testIndex];
    return {
      state,
      mass,
      ...allocateInspections(posterior, capacity)
    };
  });
  const exactAfter = branches.reduce((sum, branch) => {
    const posterior = probabilities.map((value, index) => index === testIndex ? branch.state : value);
    const exactMass = branch.state ? decimal(probabilities[testIndex]) : plus(rational(1n), decimal(-probabilities[testIndex]));
    return plus(sum, times(exactMass, selectedCost(posterior, branch.selected, 10, 80)));
  }, rational(0n));
  const afterSignal = rationalNumber(exactAfter);
  return {
    baseline,
    branches,
    afterSignal,
    total: rationalNumber(plus(exactAfter, decimal(testCost))),
    grossValue: rationalNumber(plus(selectedCost(probabilities, baseline.selected, 10, 80), rational(-exactAfter.numerator, exactAfter.denominator)))
  };
}
export function thresholdLoss(scores, labels, threshold) {
  dense(scores, 'Scores');
  dense(labels, 'Labels');
  if (scores.length !== labels.length) throw new RangeError('Scores and labels must align.');
  scores.forEach(value => probability(value, 'Score'));
  labels.forEach(label => {
    if (label !== 0 && label !== 1) throw new RangeError('Labels must be zero or one.');
  });
  finite(threshold, 'Threshold');
  const predictions = scores.map(score => Number(score >= threshold));
  const losses = labels.map((label, index) => predictions[index] === 1 ? 10 : 80 * label);
  return {
    predictions,
    losses,
    total: losses.reduce((sum, value) => sum + value, 0),
    correct: labels.filter((label, index) => predictions[index] === label).length
  };
}
