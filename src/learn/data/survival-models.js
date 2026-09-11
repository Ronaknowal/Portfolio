// Finite teaching models: observations are independent first-event records.
// These deterministic helpers are not a clinical or production fitting API.
function dense(values, name) {
  if (!Array.isArray(values) || values.length === 0 || values.length > 200 ||
      Array.from({ length: values.length }, (_, index) => !Object.hasOwn(values, index)).some(Boolean)) {
    throw new RangeError(`${name} must contain 1–200 explicit entries.`);
  }
}
function number(value, name, low, high) {
  if (!Number.isFinite(value) || value < low || value > high) {
    throw new RangeError(`${name} must be finite and between ${low} and ${high}.`);
  }
}
function records(times, events) {
  dense(times, 'Times');
  dense(events, 'Event indicators');
  if (times.length !== events.length) throw new RangeError('One indicator is required per time.');
  times.forEach(time => number(time, 'Time', 0.001, 10000));
  events.forEach(event => { if (typeof event !== 'boolean') throw new TypeError('Event indicators must be Boolean.'); });
}

export const pumpTimes = [2, 3, 4, 4, 6, 7, 8, 9];
export const pumpEvents = [true, false, true, false, true, false, true, false];
export const coxTimes = [1, 2, 2, 3, 4, 5];
export const coxEvents = [true, true, true, false, true, false];
export const coxFeatures = [-1, 0, 1, -1, 0, 1];

export function kaplanMeier(times, events) {
  records(times, events);
  let survival = 1;
  let greenwood = 0;
  let nelsonAalen = 0;
  const rows = [...new Set(times)].sort((a, b) => a - b).map(time => {
    const risk = times.filter(value => value >= time).length;
    const failures = times.filter((value, index) => value === time && events[index]).length;
    const censored = times.filter((value, index) => value === time && !events[index]).length;
    const previous = survival;
    survival *= 1 - failures / risk;
    nelsonAalen += failures / risk;
    if (failures > 0 && risk > failures) greenwood += failures / (risk * (risk - failures));
    if (failures === risk) greenwood = Infinity;
    let interval = null;
    if (survival > 0 && survival < 1 && Number.isFinite(greenwood)) {
      const transformed = Math.log(-Math.log(survival));
      const standardError = Math.sqrt(greenwood) / Math.abs(Math.log(survival));
      interval = [Math.exp(-Math.exp(transformed + 1.959963984540054 * standardError)),
        Math.exp(-Math.exp(transformed - 1.959963984540054 * standardError))];
    }
    return { time, risk, failures, censored, previous, survival, greenwood, nelsonAalen, interval };
  });
  return { rows, lastTime: Math.max(...times), median: rows.find(row => row.survival <= 0.5)?.time ?? null };
}

export function survivalAt(table, time) {
  number(time, 'Evaluation time', 0, table.lastTime);
  return table.rows.filter(row => row.time <= time).at(-1)?.survival ?? 1;
}

export function restrictedMean(table, horizon) {
  number(horizon, 'Restricted horizon', 0, table.lastTime);
  let previousTime = 0;
  let survival = 1;
  const rectangles = [];
  for (const row of table.rows) {
    const right = Math.min(row.time, horizon);
    if (right > previousTime) rectangles.push({ left: previousTime, right, height: survival, area: (right - previousTime) * survival });
    if (row.time >= horizon) break;
    survival = row.survival;
    previousTime = row.time;
  }
  return { value: rectangles.reduce((sum, rectangle) => sum + rectangle.area, 0), rectangles };
}

export function observedPumps({ cutoff = 9, toggled = null, allCensored = false } = {}) {
  number(cutoff, 'Follow-up cutoff', 1, 9);
  if (toggled !== null && (!Number.isInteger(toggled) || toggled < 0 || toggled >= 8)) throw new RangeError('Choose a pump index.');
  const events = pumpEvents.map((event, index) => !allCensored && (index === toggled ? !event : event) && pumpTimes[index] <= cutoff);
  const times = pumpTimes.map(time => Math.min(time, cutoff));
  return { times, events, table: kaplanMeier(times, events) };
}

export function weibullClock({ scale = 12, shape = 2, age = 4, interval = 3, multiplier = 1, mode = 'time' } = {}) {
  number(scale, 'Time scale', 1, 100);
  number(shape, 'Shape', 0.5, 4);
  number(age, 'Age', 0, 30);
  number(interval, 'Interval', 0.1, 15);
  number(multiplier, 'Multiplier', 0.25, 4);
  if (!['time', 'hazard'].includes(mode)) throw new RangeError('Choose time or hazard multiplication.');
  const adjustedScale = mode === 'time' ? scale * multiplier : scale;
  const hazardMultiplier = mode === 'hazard' ? multiplier : 1;
  const cumulative = time => hazardMultiplier * (time / adjustedScale) ** shape;
  const survival = time => Math.exp(-cumulative(time));
  const hazard = time => time === 0 && shape < 1 ? null : hazardMultiplier * shape / adjustedScale * (time / adjustedScale) ** (shape - 1);
  const integratedHazard = cumulative(age + interval) - cumulative(age);
  return { adjustedScale, hazardRatio: mode === 'time' ? multiplier ** -shape : multiplier,
    timeRatio: mode === 'time' ? multiplier : multiplier ** (-1 / shape),
    median: adjustedScale * (Math.log(2) / hazardMultiplier) ** (1 / shape),
    currentHazard: hazard(age), currentSurvival: survival(age),
    conditionalFailure: -Math.expm1(-integratedHazard), integratedHazard,
    curve: Array.from({ length: 161 }, (_, index) => {
      const time = index * 30 / 160;
      return { time, survival: survival(time), hazard: hazard(time) };
    }) };
}

export function coxRiskSets({ beta = Math.log(2), ties = 'efron', times = coxTimes, events = coxEvents, features = coxFeatures } = {}) {
  records(times, events);
  dense(features, 'Features');
  if (features.length !== times.length) throw new RangeError('One feature is required per record.');
  features.forEach(value => number(value, 'Feature', -10, 10));
  number(beta, 'Coefficient', -5, 5);
  if (!['efron', 'breslow'].includes(ties)) throw new RangeError('Choose Efron or Breslow ties.');
  const rows = [...new Set(times.filter((_, index) => events[index]))].sort((a, b) => a - b).map(time => {
    const riskIndices = times.flatMap((value, index) => value >= time ? [index] : []);
    const eventIndices = times.flatMap((value, index) => value === time && events[index] ? [index] : []);
    const shift = Math.max(...riskIndices.map(index => beta * features[index]));
    const weights = features.map(value => Math.exp(beta * value - shift));
    const riskWeight = riskIndices.reduce((sum, index) => sum + weights[index], 0);
    const eventWeight = eventIndices.reduce((sum, index) => sum + weights[index], 0);
    const riskMoment = riskIndices.reduce((sum, index) => sum + weights[index] * features[index], 0);
    const eventMoment = eventIndices.reduce((sum, index) => sum + weights[index] * features[index], 0);
    const riskSecond = riskIndices.reduce((sum, index) => sum + weights[index] * features[index] ** 2, 0);
    const eventSecond = eventIndices.reduce((sum, index) => sum + weights[index] * features[index] ** 2, 0);
    let logContribution = eventIndices.reduce((sum, index) => sum + beta * features[index], 0);
    let score = eventIndices.reduce((sum, index) => sum + features[index], 0);
    let information = 0;
    const denominators = eventIndices.map((_, step) => {
      const fraction = ties === 'efron' ? step / eventIndices.length : 0;
      const denominator = riskWeight - fraction * eventWeight;
      const mean = (riskMoment - fraction * eventMoment) / denominator;
      logContribution -= shift + Math.log(denominator);
      score -= mean;
      information += (riskSecond - fraction * eventSecond) / denominator - mean ** 2;
      return { step, fraction, normalizedDenominator: denominator, denominator: denominator * Math.exp(shift), mean };
    });
    return { time, riskIndices, eventIndices, denominators, logContribution, contribution: Math.exp(logContribution), score, information,
      weights: riskIndices.map(index => ({ index, feature: features[index], probability: weights[index] / riskWeight, isEvent: eventIndices.includes(index) })) };
  });
  return { rows, logLikelihood: rows.reduce((sum, row) => sum + row.logContribution, 0),
    score: rows.reduce((sum, row) => sum + row.score, 0), information: rows.reduce((sum, row) => sum + row.information, 0) };
}

export function proportionalComparison({ time = 6, mode = 'switch' } = {}) {
  number(time, 'Time', 0, 15);
  if (!['switch', 'mixture'].includes(mode)) throw new RangeError('Choose switching or mixture model.');
  const point = at => {
    if (mode === 'switch') {
      const referenceHazard = 0.1;
      const comparisonHazard = at < 4 ? 0.05 : 0.2;
      return { time: at, referenceHazard, comparisonHazard, referenceSurvival: Math.exp(-0.1 * at),
        comparisonSurvival: Math.exp(-0.05 * Math.min(at, 4) - 0.2 * Math.max(0, at - 4)), ratio: comparisonHazard / referenceHazard };
    }
    const survival = atTime => (Math.exp(-0.1 * atTime) + Math.exp(-0.4 * atTime)) / 2;
    const hazard = atTime => (0.1 * Math.exp(-0.1 * atTime) + 0.4 * Math.exp(-0.4 * atTime)) / (2 * survival(atTime));
    const referenceHazard = hazard(at);
    const comparisonHazard = 0.5 * hazard(0.5 * at);
    return { time: at, referenceHazard, comparisonHazard, referenceSurvival: survival(at), comparisonSurvival: survival(0.5 * at), ratio: comparisonHazard / referenceHazard };
  };
  // Explicit duplicate switch coordinate shows the one-sided jump, not a diagonal ramp.
  const curve = Array.from({ length: 151 }, (_, index) => point(index / 10));
  if (mode === 'switch') curve.splice(40, 0, { ...point(4), comparisonHazard: 0.05, ratio: 0.5 });
  return { selected: point(time), curve };
}

export function concordancePairs(times, events, scores, tolerance = 1e-8) {
  records(times, events);
  dense(scores, 'Risk scores');
  if (scores.length !== times.length) throw new RangeError('One score is required per record.');
  scores.forEach(score => number(score, 'Risk score', -100, 100));
  number(tolerance, 'Risk tie tolerance', 0, 0.1);
  const pairs = [];
  let concordant = 0;
  let discordant = 0;
  let tied = 0;
  for (let first = 0; first < times.length; first += 1) {
    for (let second = first + 1; second < times.length; second += 1) {
      let earlier = times[first] < times[second] ? first : second;
      let later = earlier === first ? second : first;
      const sameTime = times[first] === times[second];
      if (sameTime && events[first] !== events[second]) { earlier = events[first] ? first : second; later = earlier === first ? second : first; }
      let status = 'not comparable';
      if (events[earlier] && (!sameTime || events[first] !== events[second])) {
        const difference = scores[earlier] - scores[later];
        if (Math.abs(difference) <= tolerance) { status = 'risk tie'; tied += 1; }
        else if (difference > 0) { status = 'concordant'; concordant += 1; }
        else { status = 'discordant'; discordant += 1; }
      }
      pairs.push({ first, second, earlier, later, sameTime, status });
    }
  }
  const comparable = concordant + discordant + tied;
  return { pairs, comparable, concordant, discordant, tied, value: comparable ? (concordant + 0.5 * tied) / comparable : null };
}

export function censorWeightedBrier({ prediction = 0.6, lateCensorProbability = 0.5 } = {}) {
  number(prediction, 'Predicted survival', 0, 1);
  number(lateCensorProbability, 'Chance of observing through day 5', 0.05, 1);
  const rows = [];
  for (const [lifetime, probability] of [[2, 0.4], [8, 0.6]]) {
    for (const [followup, censorProbability] of [[3, 1 - lateCensorProbability], [10, lateCensorProbability]]) {
      const time = Math.min(lifetime, followup);
      const event = lifetime <= followup;
      const known = event || time > 5;
      const loss = (Number(lifetime > 5) - prediction) ** 2;
      const observationProbability = lifetime === 2 ? 1 : lateCensorProbability;
      rows.push({ lifetime, followup, time, event, probability: probability * censorProbability, known,
        loss, weight: known ? 1 / observationProbability : 0, contribution: known ? loss / observationProbability : 0 });
    }
  }
  const full = 0.4 * prediction ** 2 + 0.6 * (1 - prediction) ** 2;
  const weighted = rows.reduce((sum, row) => sum + row.probability * row.contribution, 0);
  const knownMass = rows.filter(row => row.known).reduce((sum, row) => sum + row.probability, 0);
  const completeCases = rows.filter(row => row.known).reduce((sum, row) => sum + row.probability * row.loss, 0) / knownMass;
  const censoredAsFailure = rows.reduce((sum, row) => sum + row.probability * (Number(row.time > 5) - prediction) ** 2, 0);
  return { rows, full, weighted, completeCases, censoredAsFailure, knownMass };
}

export function competingIncidence(times = [1, 2, 3, 4, 5, 6], statuses = [2, 1, 0, 2, 1, 0]) {
  dense(statuses, 'Statuses');
  statuses.forEach(status => { if (![0, 1, 2].includes(status)) throw new RangeError('Statuses are 0, 1 or 2.'); });
  records(times, statuses.map(status => status > 0));
  let survival = 1;
  let first = 0;
  let second = 0;
  const rows = [...new Set(times)].sort((a, b) => a - b).map(time => {
    const risk = times.filter(value => value >= time).length;
    const firstEvents = times.filter((value, index) => value === time && statuses[index] === 1).length;
    const secondEvents = times.filter((value, index) => value === time && statuses[index] === 2).length;
    const previous = survival;
    const firstIncrement = previous * firstEvents / risk;
    const secondIncrement = previous * secondEvents / risk;
    first += firstIncrement;
    second += secondIncrement;
    survival *= 1 - (firstEvents + secondEvents) / risk;
    return { time, risk, firstEvents, secondEvents, previous, firstIncrement, secondIncrement, first, second, survival };
  });
  return { rows, naiveFirstRisk: 1 - kaplanMeier(times, statuses.map(status => status === 1)).rows.at(-1).survival };
}

export function constantCompeting({ firstRate = 0.1, secondRate = 0.3, horizon = 5 } = {}) {
  number(firstRate, 'First hazard', 0, 1);
  number(secondRate, 'Competing hazard', 0, 1);
  number(horizon, 'Horizon', 0, 20);
  const total = firstRate + secondRate;
  const failure = -Math.expm1(-total * horizon);
  return { first: total ? firstRate / total * failure : 0, second: total ? secondRate / total * failure : 0,
    survival: Math.exp(-total * horizon), netFirst: -Math.expm1(-firstRate * horizon) };
}

export function logrankTable(times, events, groups) {
  records(times, events);
  dense(groups, 'Groups');
  if (groups.length !== times.length || groups.some(group => ![0, 1].includes(group))) throw new RangeError('One 0/1 group is required per record.');
  const rows = [...new Set(times.filter((_, index) => events[index]))].sort((a, b) => a - b).map(time => {
    const risk = times.filter(value => value >= time).length;
    const groupRisk = times.filter((value, index) => value >= time && groups[index] === 1).length;
    const failures = times.filter((value, index) => value === time && events[index]).length;
    const observed = times.filter((value, index) => value === time && events[index] && groups[index] === 1).length;
    return { time, risk, groupRisk, failures, observed, expected: failures * groupRisk / risk,
      variance: risk > 1 ? failures * groupRisk * (risk - groupRisk) * (risk - failures) / (risk ** 2 * (risk - 1)) : 0 };
  });
  const difference = rows.reduce((sum, row) => sum + row.observed - row.expected, 0);
  const variance = rows.reduce((sum, row) => sum + row.variance, 0);
  return { rows, difference, variance, statistic: variance > 0 ? difference ** 2 / variance : null };
}
