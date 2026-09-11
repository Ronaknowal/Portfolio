// Small deterministic teaching models. These are not library replacements.
export const parcelObservations = Object.freeze([
  { id: 'A', distance: 0, hours: 1 },
  { id: 'B', distance: 1, hours: 2 },
  { id: 'C', distance: 2, hours: 2 },
  { id: 'D', distance: 3, hours: 4 },
]);

function finite(value, name) {
  if (!Number.isFinite(value)) throw new TypeError(`${name} must be finite.`);
  return value;
}

export function residualReport(intercept, slope, lastHours = 4) {
  finite(intercept, 'Intercept');
  finite(slope, 'Slope');
  finite(lastHours, 'Last observation');
  const rows = parcelObservations.map((row, index) => {
    const hours = index === 3 ? lastHours : row.hours;
    const predicted = intercept + slope * row.distance;
    const residual = hours - predicted;
    return { ...row, hours, predicted, residual, squared: residual * residual };
  });
  const meanDistance = 1.5;
  const meanHours = rows.reduce((sum, row) => sum + row.hours, 0) / rows.length;
  const centeredCross = rows.reduce((sum, row) => sum + (row.distance - meanDistance) * (row.hours - meanHours), 0);
  const bestSlope = centeredCross / 5;
  const bestIntercept = meanHours - bestSlope * meanDistance;
  if (rows.some(row => !Number.isFinite(row.squared)) || !Number.isFinite(centeredCross)) throw new RangeError('Values exceed the numerical teaching model.');
  return {
    rows,
    meanHours,
    bestSlope,
    bestIntercept,
    // Average before summing so a representable mean does not overflow its total.
    mse: rows.reduce((sum, row) => sum + row.squared / rows.length, 0),
    gradient: [
      -2 * rows.reduce((sum, row) => sum + row.residual, 0) / rows.length,
      -2 * rows.reduce((sum, row) => sum + row.distance * row.residual, 0) / rows.length,
    ],
  };
}

export function gradientTrace(rate = 0.05, count = 16) {
  if (![0.05, 0.2, 0.3].includes(rate)) throw new RangeError('Choose a supported learning rate.');
  if (!Number.isInteger(count) || count < 0 || count > 40) throw new RangeError('Use 0–40 steps.');
  let intercept = 0;
  let slope = 0;
  const states = [];
  for (let step = 0; step <= count; step += 1) {
    const report = residualReport(intercept, slope);
    states.push({ step, intercept, slope, ...report });
    [intercept, slope] = [intercept - rate * report.gradient[0], slope - rate * report.gradient[1]];
  }
  return states;
}

export function sigmoid(score) {
  finite(score, 'Score');
  if (score >= 0) return 1 / (1 + Math.exp(-score));
  const exponential = Math.exp(score);
  return exponential / (1 + exponential);
}

export function binaryScoreLoss(score, label) {
  finite(score, 'Score');
  if (label !== 0 && label !== 1) throw new RangeError('Label must be 0 or 1.');
  // log(1 + exp(z)) - y*z, evaluated without overflow or positive-tail cancellation.
  return Math.max((1 - 2 * label) * score, 0) + Math.log1p(Math.exp(-Math.abs(score)));
}

export function logisticReport({ intercept = -1, weight = 1, feature = 2, label = 1 } = {}) {
  [intercept, weight, feature].forEach(value => finite(value, 'Input'));
  const score = intercept + weight * feature;
  const probability = sigmoid(score);
  if (!Number.isFinite(Math.exp(score)) || !Number.isFinite((probability - label) * feature)) throw new RangeError('Score or derivative exceeds representable arithmetic.');
  return { score, probability, odds: Math.exp(score), loss: binaryScoreLoss(score, label), biasGradient: probability - label, weightGradient: (probability - label) * feature };
}

export const validationScores = Object.freeze([
  { id: 'V1', probability: 0.1, label: 0 },
  { id: 'V2', probability: 0.2, label: 1 },
  { id: 'V3', probability: 0.35, label: 0 },
  { id: 'V4', probability: 0.45, label: 1 },
  { id: 'V5', probability: 0.65, label: 0 },
  { id: 'V6', probability: 0.8, label: 1 },
]);

export function thresholdReport(threshold = 0.5, missedCost = 4) {
  if (!Number.isFinite(threshold) || threshold < 0 || threshold > 1) throw new RangeError('Threshold must lie in [0,1].');
  if (!Number.isFinite(missedCost) || missedCost < 0 || missedCost > 1e12) throw new RangeError('Cost must be between 0 and 10¹².');
  const counts = { tp: 0, fp: 0, tn: 0, fn: 0 };
  const rows = validationScores.map(row => {
    const prediction = Number(row.probability >= threshold);
    const outcome = prediction ? (row.label ? 'tp' : 'fp') : (row.label ? 'fn' : 'tn');
    counts[outcome] += 1;
    return { ...row, prediction, outcome };
  });
  return {
    rows,
    ...counts,
    cost: counts.fp + missedCost * counts.fn,
    accuracy: (counts.tp + counts.tn) / rows.length,
    precision: counts.tp + counts.fp ? counts.tp / (counts.tp + counts.fp) : null,
    recall: counts.tp / 3,
  };
}

export function separationReport(weight, penalty) {
  finite(weight, 'Weight');
  if (Math.abs(weight) > 1e6) throw new RangeError('Weight exceeds this teaching model.');
  if (![0, 0.02, 0.2].includes(penalty)) throw new RangeError('Choose a supported penalty.');
  // Two equally weighted observations (-1,0), (+1,1), intercept fixed at zero.
  const dataLoss = binaryScoreLoss(weight, 1);
  return { dataLoss, penaltyLoss: penalty * weight * weight / 2, objective: dataLoss + penalty * weight * weight / 2, derivative: sigmoid(weight) - 1 + penalty * weight };
}

export function uncertaintyReport(input) {
  if (!Number.isFinite(input) || input < -1 || input > 9) throw new RangeError('Use the illustrated interval −1 to 9.');
  const observations = [1, 1.7, 3.2, 3.7, 5.3, 5.8];
  const meanInput = 2.5;
  const meanOutput = 3.45;
  const centeredSum = 17.5;
  const slope = observations.reduce((sum, output, index) => sum + (index - meanInput) * (output - meanOutput), 0) / centeredSum;
  const intercept = meanOutput - slope * meanInput;
  const residualVariance = observations.reduce((sum, output, index) => sum + (output - intercept - slope * index) ** 2, 0) / 4;
  const leverage = 1 / 6 + (input - meanInput) ** 2 / centeredSum;
  // scipy.stats.t.ppf(.975, df=4), independently checked by the native verifier.
  const critical = 2.7764451051977987;
  return { input, predicted: intercept + slope * input, meanHalfWidth: critical * Math.sqrt(residualVariance * leverage), individualHalfWidth: critical * Math.sqrt(residualVariance * (1 + leverage)), residualVariance, observations };
}

export function separationOptimum(penalty) {
  if (penalty === 0) return null;
  separationReport(0, penalty);
  let low = 0;
  let high = 32;
  for (let step = 0; step < 70; step += 1) {
    const middle = (low + high) / 2;
    if (separationReport(middle, penalty).derivative < 0) low = middle;
    else high = middle;
  }
  return (low + high) / 2;
}

export function featureMapReport(linearWeight = 0, squareWeight = 1, intercept = -1) {
  [linearWeight, squareWeight, intercept].forEach(value => finite(value, 'Coefficient'));
  return [-2, -1, 0, 1, 2].map(input => {
    const score = intercept + linearWeight * input + squareWeight * input * input;
    return { input, square: input * input, score, probability: sigmoid(score) };
  });
}
