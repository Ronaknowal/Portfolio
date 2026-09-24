// Small, explicitly bounded teaching models. Null targets mean unobserved labels.
export const multioutputTruth = [[1, 0, 1], [0, 1, 1], [1, 1, 0], [0, 0, 1]];
export const multioutputPredictions = [[1, 0, 1], [0, 1, 0], [1, 0, 0], [0, 1, 1]];
export const multioutputScores = [.9, .8, .8, .55, .4, .2];
export const thresholdTargets = [0, 1, 0, 1, 1, 0];
function dense(values, minimum, maximum, name) {
  if (!Array.isArray(values) || values.length < minimum || values.length > maximum) {
    throw new TypeError(`${name} has an unsupported length.`);
  }
  for (let index = 0; index < values.length; index += 1) {
    if (!Object.hasOwn(values, index)) throw new TypeError(`${name} must contain every own element.`);
  }
  return values;
}
function finite(value, minimum, maximum, name) {
  if (!Number.isFinite(value) || value < minimum || value > maximum) {
    throw new RangeError(`${name} must be finite in [${minimum}, ${maximum}].`);
  }
  return value;
}
function bit(value, name) {
  if (value !== 0 && value !== 1) throw new TypeError(`${name} must be 0 or 1.`);
  return value;
}
function blankCounts() {
  return {
    tp: 0,
    fp: 0,
    fn: 0,
    tn: 0,
    observed: 0
  };
}
function finishCounts(counts) {
  const {
    tp,
    fp,
    fn,
    observed
  } = counts;
  return {
    ...counts,
    precision: observed ? tp + fp ? tp / (tp + fp) : 0 : null,
    recall: observed ? tp + fn ? tp / (tp + fn) : 0 : null,
    f1: observed ? 2 * tp + fp + fn ? 2 * tp / (2 * tp + fp + fn) : 0 : null,
    jaccard: observed ? tp + fp + fn ? tp / (tp + fp + fn) : 0 : null,
    hamming: observed ? (fp + fn) / observed : null
  };
}
const mean = values => values.length ? values.reduce((sum, value) => sum + value, 0) / values.length : null;
export function multioutputMetrics(truth = multioutputTruth, predictions = multioutputPredictions) {
  dense(truth, 1, 50, 'Target rows');
  dense(predictions, truth.length, truth.length, 'Prediction rows');
  const columns = dense(truth[0], 1, 20, 'Target columns').length;
  const labelCounts = Array.from({
    length: columns
  }, blankCounts);
  const total = blankCounts();
  const rows = truth.map((row, rowIndex) => {
    dense(row, columns, columns, 'Target row');
    dense(predictions[rowIndex], columns, columns, 'Prediction row');
    const counts = blankCounts();
    const cells = row.map((target, column) => {
      const prediction = bit(predictions[rowIndex][column], 'Prediction');
      if (target === null) return 'unknown';
      bit(target, 'Observed target');
      const category = target ? prediction ? 'tp' : 'fn' : prediction ? 'fp' : 'tn';
      for (const record of [counts, labelCounts[column], total]) {
        record[category] += 1;
        record.observed += 1;
      }
      return category;
    });
    return {
      ...finishCounts(counts),
      cells,
      complete: counts.observed === columns,
      exact: counts.observed === columns ? counts.fp + counts.fn === 0 : null
    };
  });
  const labels = labelCounts.map(finishCounts);
  const completeRows = rows.filter(row => row.complete);
  return {
    ...finishCounts(total),
    rows,
    labels,
    columns,
    missing: truth.length * columns - total.observed,
    macroF1: mean(labels.filter(label => label.observed).map(label => label.f1)),
    observedSampleF1: mean(rows.filter(row => row.observed).map(row => row.f1)),
    observedSampleJaccard: mean(rows.filter(row => row.observed).map(row => row.jaccard)),
    completeRowCount: completeRows.length,
    subsetAccuracy: mean(completeRows.map(row => Number(row.exact)))
  };
}
export function labelJointDecisions(counts = [6, 5, 1, 8], order = 'AB') {
  dense(counts, 4, 4, 'Four outcome counts').forEach(value => {
    finite(value, 0, 50, 'Outcome count');
    if (!Number.isInteger(value)) throw new TypeError('Outcome counts must be integers.');
  });
  if (!['AB', 'BA'].includes(order)) throw new RangeError('Order must be AB or BA.');
  const total = counts.reduce((sum, value) => sum + value, 0);
  if (!total) throw new RangeError('At least one outcome needs positive mass.');
  const states = [[0, 0], [0, 1], [1, 0], [1, 1]];
  const mass = counts.map(value => value / total);
  const marginals = [mass[2] + mass[3], mass[1] + mass[3]];
  const first = order === 'AB' ? 0 : 1;
  const second = 1 - first;
  const branches = [0, 1].map(firstValue => {
    const leaves = states.map((state, index) => ({
      state,
      index,
      mass: mass[index]
    })).filter(leaf => leaf.state[first] === firstValue);
    const support = leaves.reduce((sum, leaf) => sum + leaf.mass, 0);
    const conditionalOne = support ? leaves.find(leaf => leaf.state[second] === 1).mass / support : null;
    return {
      firstValue,
      support,
      conditionalOne,
      leaves: leaves.map(leaf => ({
        ...leaf,
        conditional: support ? leaf.mass / support : null
      }))
    };
  });
  const greedy = [0, 0];
  greedy[first] = Number(marginals[first] >= .5);
  greedy[second] = Number(branches[greedy[first]].conditionalOne >= .5);
  const marginalDecision = marginals.map(probability => Number(probability >= .5));
  const candidates = states.map((prediction, index) => ({
    prediction,
    mass: mass[index],
    subsetLoss: 1 - mass[index],
    hammingRisk: mass.reduce((sum, probability, outcome) => sum + probability * states[outcome].reduce((errors, target, label) => errors + Number(target !== prediction[label]), 0) / 2, 0)
  }));
  const largestCount = Math.max(...counts);
  const modes = states.filter((_, index) => counts[index] === largestCount);
  return {
    counts: [...counts],
    total,
    states,
    mass,
    marginals,
    order,
    first,
    second,
    branches,
    greedy,
    marginalDecision,
    modes,
    candidates
  };
}
export function pooledLabelAssociation({
  low = .1,
  high = .9,
  highShare = .5
} = {}) {
  [low, high, highShare].forEach(value => finite(value, 0, 1, 'Probability'));
  const independentMass = probability => [(1 - probability) ** 2, (1 - probability) * probability, probability * (1 - probability), probability ** 2];
  const strata = [low, high].map(probability => ({
    probability,
    mass: independentMass(probability)
  }));
  const pooled = strata[0].mass.map((value, index) => (1 - highShare) * value + highShare * strata[1].mass[index]);
  const marginal = (1 - highShare) * low + highShare * high;
  return {
    low,
    high,
    highShare,
    strata,
    pooled,
    marginal,
    conditional: marginal ? pooled[3] / marginal : null,
    covariance: pooled[3] - marginal ** 2
  };
}
export function thresholdInspection({
  scores = multioutputScores,
  targets = thresholdTargets,
  threshold = .5
} = {}) {
  dense(scores, 1, 50, 'Validation scores').forEach(value => finite(value, 0, 1, 'Score'));
  dense(targets, scores.length, scores.length, 'Validation targets').forEach(value => bit(value, 'Validation target'));
  finite(threshold, 0, 1.01, 'Threshold');
  const evaluate = cutoff => {
    const counts = blankCounts();
    const predictions = scores.map((score, index) => {
      const prediction = Number(score >= cutoff);
      const category = targets[index] ? prediction ? 'tp' : 'fn' : prediction ? 'fp' : 'tn';
      counts[category] += 1;
      counts.observed += 1;
      return prediction;
    });
    return {
      threshold: cutoff,
      ...finishCounts(counts),
      predictions
    };
  };
  const knots = [...new Set([0, ...scores, 1.01])].sort((left, right) => left - right);
  const candidates = knots.map(evaluate);
  const best = Math.max(...candidates.map(candidate => candidate.f1));
  return {
    ...evaluate(threshold),
    scores: [...scores],
    targets: [...targets],
    candidates,
    best: candidates.filter(candidate => candidate.f1 === best),
    ranked: scores.map((score, index) => ({
      score,
      target: targets[index],
      index,
      selected: score >= threshold
    })).sort((left, right) => right.score - left.score || left.index - right.index)
  };
}
export function sharedOutputStump({
  energyScale = 1,
  temperature = [0, 0, 2, 2],
  energy = [0, 100, 100, 100]
} = {}) {
  finite(energyScale, .1, 1000, 'Energy scale');
  dense(temperature, 4, 4, 'Temperature values').forEach(value => finite(value, -100, 100, 'Temperature'));
  dense(energy, 4, 4, 'Energy values').forEach(value => finite(value, -1e4, 1e4, 'Energy'));
  const targets = [temperature, energy];
  const candidates = [1, 2, 3].map(cut => {
    const outputs = targets.map((values, output) => {
      const leftMean = mean(values.slice(0, cut));
      const rightMean = mean(values.slice(cut));
      const predictions = values.map((_, index) => index < cut ? leftMean : rightMean);
      const sse = values.reduce((sum, value, index) => sum + (value - predictions[index]) ** 2, 0);
      const scale = output === 0 ? 1 : energyScale;
      return {
        values: [...values],
        leftMean,
        rightMean,
        predictions,
        sse,
        scaledSse: sse / scale ** 2
      };
    });
    return {
      cut,
      threshold: cut - .5,
      outputs,
      scaledSse: outputs.reduce((sum, output) => sum + output.scaledSse, 0)
    };
  });
  const shared = candidates.reduce((best, candidate) => candidate.scaledSse < best.scaledSse ? candidate : best);
  const separate = targets.map((_, output) => candidates.reduce((best, candidate) => candidate.outputs[output].sse < best.outputs[output].sse ? candidate : best));
  return {
    energyScale,
    candidates,
    shared,
    separate,
    targets
  };
}
export function sharedFeatureShrinkage({
  first = 3,
  second = 4,
  penalty = 2
} = {}) {
  [first, second].forEach(value => finite(value, -6, 6, 'Unpenalized coefficient'));
  finite(penalty, 0, 8, 'Penalty');
  const initial = [first, second];
  const norm = Math.hypot(first, second);
  const multiplier = norm ? Math.max(0, 1 - penalty / norm) : 0;
  const grouped = initial.map(value => multiplier * value);
  const separate = initial.map(value => Math.sign(value) * Math.max(0, Math.abs(value) - penalty));
  const groupedObjective = coefficients => .5 * coefficients.reduce((sum, value, index) => sum + (value - initial[index]) ** 2, 0) + penalty * Math.hypot(...coefficients);
  return {
    initial,
    norm,
    penalty,
    multiplier,
    grouped,
    separate,
    groupedObjective: groupedObjective(grouped),
    independentAtGroupedObjective: groupedObjective(separate)
  };
}
