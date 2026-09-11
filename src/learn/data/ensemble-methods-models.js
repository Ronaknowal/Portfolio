// Bounded deterministic teaching models. Native library fits are separate artifacts.
const owns = (object, key) => Object.prototype.hasOwnProperty.call(object, key);
function finiteRange(value, low, high, name) {
  if (!Number.isFinite(value) || value < low || value > high) {
    throw new RangeError(`${name} must be finite and between ${low} and ${high}.`);
  }
  return value;
}
function integerRange(value, low, high, name) {
  finiteRange(value, low, high, name);
  if (!Number.isInteger(value)) throw new RangeError(`${name} must be an integer.`);
  return value;
}
function denseVector(values, length, check, name) {
  if (!Array.isArray(values) || values.length !== length) throw new TypeError(`${name} has the wrong length.`);
  for (let index = 0; index < length; index += 1) {
    if (!owns(values, index)) throw new TypeError(`${name} must contain every entry.`);
    check(values[index], index);
  }
  return values;
}
export function ensembleNumber(value, places = 4) {
  if (value === null) return 'not available';
  if (!Number.isFinite(value)) return String(value);
  if (value === 0) return '0';
  if (Math.abs(value) < 10 ** -places) return value.toExponential(2);
  return value.toFixed(places).replace(/\.?0+$/, '');
}
export const errorBlendPresets = Object.freeze({
  complementary: {
    label: 'Different signed errors',
    a: [-3, -1, 2, 2],
    b: [1, 1, -1, -3]
  },
  copies: {
    label: 'Two copies of one forecast',
    a: [-3, -1, 2, 2],
    b: [-3, -1, 2, 2]
  },
  sameSide: {
    label: 'Both forecasts too high',
    a: [1, 2, 1, 2],
    b: [3, 2, 4, 3]
  }
});
export function errorBlendState(preset = 'complementary', weight = 0.5) {
  if (!owns(errorBlendPresets, preset)) throw new RangeError('Choose a known error preset.');
  finiteRange(weight, 0, 1, 'Weight on A');
  const {
    a,
    b,
    label
  } = errorBlendPresets[preset];
  const rows = a.map((first, index) => {
    const second = b[index];
    const blended = weight * first + (1 - weight) * second;
    return {
      id: String.fromCharCode(65 + index),
      a: first,
      b: second,
      blended
    };
  });
  const mean = values => values.reduce((total, value) => total + value / values.length, 0);
  const mseA = mean(a.map(value => value * value));
  const mseB = mean(b.map(value => value * value));
  const mse = mean(rows.map(row => row.blended ** 2));
  const denominator = a.reduce((total, first, index) => total + (first - b[index]) ** 2, 0);
  const rawOptimum = denominator === 0 ? null : b.reduce((total, second, index) => total + second * (second - a[index]), 0) / denominator;
  return {
    label,
    rows,
    weight,
    mseA,
    mseB,
    mse,
    averageMemberLoss: weight * mseA + (1 - weight) * mseB,
    disagreement: mean(rows.map(row => weight * (row.a - row.blended) ** 2 + (1 - weight) * (row.b - row.blended) ** 2)),
    optimum: rawOptimum === null ? null : Math.max(0, Math.min(1, rawOptimum)),
    rawOptimum
  };
}
export function votingState(probabilities = [0.51, 0.51, 0.01], weights = [1, 1, 1]) {
  denseVector(probabilities, 3, value => finiteRange(value, 0, 1, 'Probability'), 'Three probabilities');
  denseVector(weights, 3, value => finiteRange(value, 0, 10, 'Model weight'), 'Three weights');
  const total = weights.reduce((sum, value) => sum + value, 0);
  if (total === 0) throw new RangeError('At least one model needs positive weight.');
  const shares = weights.map(value => value / total);
  // Binary class tie policy matches argmax over ordered classes [0, 1].
  const labels = probabilities.map(value => value > 0.5 ? 1 : 0);
  const ballotMass = labels.reduce((sum, label, index) => sum + shares[index] * label, 0);
  const probability = probabilities.reduce((sum, value, index) => sum + shares[index] * value, 0);
  return {
    shares,
    labels,
    ballotMass,
    probability,
    hardClass: ballotMass > 0.5 ? 1 : 0,
    softClass: probability > 0.5 ? 1 : 0
  };
}
export const bootstrapRows = Object.freeze([1, 2, 2, 6, 7, 8].map((y, x) => Object.freeze({
  id: String.fromCharCode(65 + x),
  x,
  y
})));
export const bootstrapPresets = Object.freeze({
  repeated: [[0, 0, 2, 3, 3, 5], [1, 2, 2, 4, 4, 5], [0, 1, 1, 3, 4, 4]],
  allRows: [[0, 1, 2, 3, 4, 5], [0, 1, 2, 3, 4, 5], [0, 1, 2, 3, 4, 5]],
  smallBags: [[0, 0, 5], [1, 2, 4], [0, 3, 3]]
});
export function fitBootstrapStump(draws) {
  if (!Array.isArray(draws) || draws.length < 1 || draws.length > 12) throw new RangeError('Use 1–12 row draws.');
  denseVector(draws, draws.length, value => integerRange(value, 0, 5, 'Drawn row'), 'Draws');
  const observations = draws.map(index => bootstrapRows[index]);
  const mean = values => values.reduce((sum, row) => sum + row.y, 0) / values.length;
  const allMean = mean(observations);
  const score = (left, right, leftMean, rightMean) => [...left.map(row => (row.y - leftMean) ** 2), ...right.map(row => (row.y - rightMean) ** 2)].reduce((sum, value) => sum + value, 0);
  let best = {
    threshold: null,
    leftMean: allMean,
    rightMean: allMean,
    sse: score(observations, [], allMean, allMean)
  };
  const distinctInputs = [...new Set(observations.map(row => row.x))].sort((a, b) => a - b);
  const thresholds = distinctInputs.slice(1).map((value, index) => (value + distinctInputs[index]) / 2);
  for (const threshold of thresholds) {
    const left = observations.filter(row => row.x <= threshold);
    const right = observations.filter(row => row.x > threshold);
    if (!left.length || !right.length) continue;
    const leftMean = mean(left);
    const rightMean = mean(right);
    const sse = score(left, right, leftMean, rightMean);
    // Integer fixtures have well-separated distinct costs; tolerance resolves rounding of exact ties.
    if (sse < best.sse - 1e-12) best = {
      threshold,
      leftMean,
      rightMean,
      sse
    };
  }
  const counts = bootstrapRows.map((_, index) => draws.filter(drawn => drawn === index).length);
  return {
    ...best,
    counts,
    predictions: bootstrapRows.map(row => best.threshold === null || row.x <= best.threshold ? best.leftMean : best.rightMean)
  };
}
export function bootstrapState(preset = 'repeated', activeBag = 0, inspectedRow = 0) {
  if (!owns(bootstrapPresets, preset)) throw new RangeError('Choose a known bootstrap preset.');
  integerRange(activeBag, 0, 2, 'Bag index');
  integerRange(inspectedRow, 0, 5, 'Inspected row');
  const draws = bootstrapPresets[preset];
  const fits = draws.map(fitBootstrapStump);
  const eligible = fits.flatMap((fit, index) => fit.counts[inspectedRow] === 0 ? [index] : []);
  const oobPrediction = eligible.length ? eligible.reduce((sum, index) => sum + fits[index].predictions[inspectedRow] / eligible.length, 0) : null;
  return {
    preset,
    activeBag,
    inspectedRow,
    draws,
    fits,
    eligible,
    oobPrediction,
    ensemblePrediction: fits.reduce((sum, fit) => sum + fit.predictions[inspectedRow] / fits.length, 0),
    missingProbability: (5 / 6) ** draws[0].length,
    expectedRepresented: 6 * (1 - (5 / 6) ** draws[0].length)
  };
}
export const boostingPresets = Object.freeze({
  mixed: {
    label: 'Six distinct cases',
    x: [0, 1, 2, 3, 4, 5],
    y: [-1, -1, 1, 1, -1, 1]
  },
  perfect: {
    label: 'One threshold is enough',
    x: [0, 1, 2, 3, 4, 5],
    y: [-1, -1, -1, 1, 1, 1]
  },
  contradiction: {
    label: 'Identical inputs, opposite labels',
    x: [0, 0, 1, 1],
    y: [-1, 1, -1, 1]
  }
});
function weightedClassifierStump(x, y, weights) {
  const unique = [...new Set(x)].sort((a, b) => a - b);
  const thresholds = [unique[0] - 0.5, ...unique.slice(1).map((value, index) => (value + unique[index]) / 2), unique.at(-1) + 0.5];
  let best = null;
  for (const threshold of thresholds) {
    for (const polarity of [-1, 1]) {
      const predictions = x.map(value => value <= threshold ? polarity : -polarity);
      const error = predictions.reduce((sum, prediction, index) => sum + (prediction === y[index] ? 0 : weights[index]), 0);
      if (best === null || error < best.error - 1e-14) best = {
        threshold,
        polarity,
        predictions,
        error
      };
    }
  }
  return best;
}
export function signedBoostingTrace(preset = 'mixed', rounds = 8) {
  if (!owns(boostingPresets, preset)) throw new RangeError('Choose a known boosting fixture.');
  integerRange(rounds, 1, 12, 'Round limit');
  const {
    x,
    y,
    label
  } = boostingPresets[preset];
  let weights = x.map(() => 1 / x.length);
  let scores = x.map(() => 0);
  let bound = 1;
  const frames = [];
  for (let round = 1; round <= rounds; round += 1) {
    const stump = weightedClassifierStump(x, y, weights);
    const before = [...weights];
    if (stump.error === 0) {
      frames.push({
        round,
        status: 'perfect',
        stump,
        before,
        after: null,
        alpha: null,
        scores: null,
        labels: stump.predictions,
        trainingError: 0,
        loss: 0,
        bound: 0
      });
      break;
    }
    if (stump.error >= 0.5 - 1e-14) {
      const labels = scores.map(value => value >= 0 ? 1 : -1);
      frames.push({
        round,
        status: 'no-edge',
        stump,
        before,
        after: null,
        alpha: 0,
        scores: [...scores],
        labels,
        trainingError: labels.filter((value, index) => value !== y[index]).length / y.length,
        loss: weights.length ? y.reduce((sum, value, index) => sum + Math.exp(-value * scores[index]) / y.length, 0) : 1,
        bound
      });
      break;
    }
    const alpha = 0.5 * (Math.log1p(-stump.error) - Math.log(stump.error));
    const unnormalized = weights.map((weight, index) => weight * Math.exp(-alpha * y[index] * stump.predictions[index]));
    const normalizer = unnormalized.reduce((sum, value) => sum + value, 0);
    weights = unnormalized.map(value => value / normalizer);
    scores = scores.map((value, index) => value + alpha * stump.predictions[index]);
    bound *= normalizer;
    const labels = scores.map(value => value >= 0 ? 1 : -1);
    frames.push({
      round,
      status: 'accepted',
      stump,
      before,
      after: [...weights],
      alpha,
      normalizer,
      scores: [...scores],
      labels,
      trainingError: labels.filter((value, index) => value !== y[index]).length / y.length,
      loss: y.reduce((sum, value, index) => sum + Math.exp(-value * scores[index]) / y.length, 0),
      bound
    });
  }
  return {
    x,
    y,
    label,
    frames
  };
}
const stackFolds = Object.freeze([[0, 3], [1, 4], [2, 5]]);
function fitLineOnRows(indices) {
  const averageX = indices.reduce((sum, index) => sum + bootstrapRows[index].x / indices.length, 0);
  const averageY = indices.reduce((sum, index) => sum + bootstrapRows[index].y / indices.length, 0);
  const variance = indices.reduce((sum, index) => sum + (bootstrapRows[index].x - averageX) ** 2, 0);
  const slope = indices.reduce((sum, index) => sum + (bootstrapRows[index].x - averageX) * (bootstrapRows[index].y - averageY), 0) / variance;
  return {
    intercept: averageY - slope * averageX,
    slope
  };
}
function nearestOnRows(indices, query) {
  let best = indices[0];
  for (const index of indices.slice(1)) {
    if (Math.abs(bootstrapRows[index].x - query) < Math.abs(bootstrapRows[best].x - query)) best = index;
  }
  return bootstrapRows[best].y;
}
export function oofOwnershipState(mode = 'honest', completed = 1, query = 2.5) {
  if (!['honest', 'leaky'].includes(mode)) throw new RangeError('Choose honest or leaky ownership.');
  integerRange(completed, 0, 3, 'Completed folds');
  finiteRange(query, 0, 5, 'New query');
  const all = bootstrapRows.map((_, index) => index);
  const matrix = bootstrapRows.map(() => [null, null]);
  const stages = stackFolds.map((held, fold) => {
    const train = mode === 'honest' ? all.filter(index => !held.includes(index)) : all;
    const line = fitLineOnRows(train);
    const predictions = held.map(index => [nearestOnRows(train, bootstrapRows[index].x), line.intercept + line.slope * bootstrapRows[index].x]);
    if (fold < completed) held.forEach((index, position) => {
      matrix[index] = predictions[position];
    });
    return {
      held,
      train,
      line,
      predictions
    };
  });
  let weightNearest = null;
  let trainMse = null;
  if (completed === 3) {
    const errors = matrix.map((row, index) => row.map(value => value - bootstrapRows[index].y));
    const denominator = errors.reduce((sum, [a, b]) => sum + (a - b) ** 2, 0);
    weightNearest = denominator === 0 ? 0.5 : Math.max(0, Math.min(1, errors.reduce((sum, [a, b]) => sum + b * (b - a), 0) / denominator));
    trainMse = errors.reduce((sum, [a, b]) => sum + (weightNearest * a + (1 - weightNearest) * b) ** 2 / errors.length, 0);
  }
  const fullLine = fitLineOnRows(all);
  const nearest = nearestOnRows(all, query);
  const linear = fullLine.intercept + fullLine.slope * query;
  return {
    mode,
    completed,
    query,
    matrix,
    stages,
    weightNearest,
    trainMse,
    fullLine,
    nearest,
    linear,
    ensemble: weightNearest === null ? null : weightNearest * nearest + (1 - weightNearest) * linear
  };
}
export function calibratedAverageLaw() {
  return [0, 1].flatMap(a => [0, 1].map(b => ({
    a,
    b,
    mass: 0.25,
    trueChance: (a + b) / 2,
    forecastA: 0.25 + a / 2,
    forecastB: 0.25 + b / 2,
    average: 0.25 + (a + b) / 4
  })));
}
