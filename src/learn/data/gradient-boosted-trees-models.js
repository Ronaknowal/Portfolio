// Bounded teaching calculations, not replacements for the production libraries.
// All state is derived from supplied observations; no measured performance data.

export const BOOSTING_FIXTURE = Object.freeze({
  x: Object.freeze([1, 2, 3, 4, 5, 6]),
  y: Object.freeze([2, 2, 3, 7, 8, 8])
});
function finite(value, label, minimum, maximum) {
  if (typeof value !== 'number' || !Number.isFinite(value) || value < minimum || value > maximum) {
    throw new Error(`${label} must be a finite number from ${minimum} through ${maximum}.`);
  }
  return value;
}
function integer(value, label, minimum, maximum) {
  finite(value, label, minimum, maximum);
  if (!Number.isInteger(value)) throw new Error(`${label} must be an integer.`);
  return value;
}
function vector(values, label, minimum = 1, maximum = 96) {
  if (!Array.isArray(values) || values.length < minimum || values.length > maximum) {
    throw new Error(`${label} needs ${minimum}–${maximum} entries.`);
  }
  return Array.from({
    length: values.length
  }, (_, index) => {
    if (!Object.hasOwn(values, index)) throw new Error(`${label} cannot contain a missing entry.`);
    return finite(values[index], `${label}[${index}]`, -1000, 1000);
  });
}
export function parseBoostingTargets(text) {
  const fields = String(text).trim().split(/[\s,]+/);
  if (fields.length !== 6 || fields.some(field => !/^-?\d{1,2}(?:\.\d{1,3})?$/.test(field))) {
    throw new Error('Enter six numbers from −30 to 30, with at most three decimal places.');
  }
  return fields.map(field => finite(Number(field), 'Target', -30, 30));
}
const sum = values => values.reduce((total, value) => total + value, 0);
const mean = values => sum(values) / values.length;
const squaredError = (values, center) => sum(values.map(value => (value - center) ** 2));
export function meanSquaredError(target, predicted) {
  const checked = vector(target, 'Targets');
  const prediction = vector(predicted, 'Predictions', checked.length, checked.length);
  return sum(checked.map((value, index) => (value - prediction[index]) ** 2)) / checked.length;
}

/** Exact candidate enumeration on a one-feature teaching tree, with <= routing.
 * Ties retain the first ascending boundary; an unsplittable node remains a leaf.
 * Complexity: this deliberately transparent scan is quadratic per visited node.
 */
export function fitCorrectionTree(xInput, targetInput, maxDepth = 1) {
  const x = vector(xInput, 'Feature');
  const target = vector(targetInput, 'Targets', x.length, x.length);
  integer(maxDepth, 'Depth', 0, 4);
  let nextId = 0;
  function build(indices, depth) {
    const id = `n${nextId++}`;
    const values = indices.map(index => target[index]);
    const value = mean(values);
    const error = squaredError(values, value);
    const leaf = {
      id,
      indices: [...indices],
      value,
      error,
      depth,
      leaf: true
    };
    if (depth === maxDepth || indices.length < 2 || error === 0) return leaf;
    const unique = [...new Set(indices.map(index => x[index]))].sort((a, b) => a - b);
    let best = null;
    for (let boundary = 0; boundary + 1 < unique.length; boundary += 1) {
      const low = unique[boundary];
      const high = unique[boundary + 1];
      const midpoint = low + (high - low) / 2;
      // Adjacent floats may round a midpoint to the right value. The lower
      // observed value still represents exactly this <= / > partition.
      const threshold = midpoint < high ? midpoint : low;
      const left = indices.filter(index => x[index] <= threshold);
      const right = indices.filter(index => x[index] > threshold);
      const leftValues = left.map(index => target[index]);
      const rightValues = right.map(index => target[index]);
      const cost = squaredError(leftValues, mean(leftValues)) + squaredError(rightValues, mean(rightValues));
      if (cost < error && (!best || cost < best.cost)) best = {
        threshold,
        left,
        right,
        cost
      };
    }
    if (!best) return leaf;
    return {
      ...leaf,
      leaf: false,
      threshold: best.threshold,
      gain: error - best.cost,
      left: build(best.left, depth + 1),
      right: build(best.right, depth + 1)
    };
  }
  return build(x.map((_, index) => index), 0);
}
export function predictCorrection(tree, feature) {
  finite(feature, 'Prediction feature', -1000, 1000);
  let node = tree;
  while (!node.leaf) node = feature <= node.threshold ? node.left : node.right;
  return node.value;
}
export function correctionLeaves(tree) {
  if (tree.leaf) return [tree];
  return [...correctionLeaves(tree.left), ...correctionLeaves(tree.right)];
}
export function fitBoosting({
  x = BOOSTING_FIXTURE.x,
  y = BOOSTING_FIXTURE.y,
  rate = 0.5,
  rounds = 6,
  depth = 1
} = {}) {
  const feature = vector(x, 'Feature');
  const target = vector(y, 'Targets', feature.length, feature.length);
  finite(rate, 'Learning rate', 0, 1.5);
  integer(rounds, 'Rounds', 0, 80);
  integer(depth, 'Depth', 0, 4);
  const base = mean(target);
  let prediction = feature.map(() => base);
  const stages = [{
    round: 0,
    prediction: [...prediction],
    mse: meanSquaredError(target, prediction),
    tree: null
  }];
  const trees = [];
  for (let round = 1; round <= rounds; round += 1) {
    const before = [...prediction];
    const residual = target.map((value, index) => value - before[index]);
    const tree = fitCorrectionTree(feature, residual, depth);
    const correction = feature.map(value => predictCorrection(tree, value));
    prediction = before.map((value, index) => value + rate * correction[index]);
    trees.push(tree);
    stages.push({
      round,
      before,
      residual,
      tree,
      correction,
      prediction: [...prediction],
      mse: meanSquaredError(target, prediction)
    });
  }
  return {
    x: feature,
    y: target,
    base,
    rate,
    depth,
    trees,
    stages
  };
}
export function predictBoosting(model, feature, count = model.trees.length) {
  integer(count, 'Tree count', 0, model.trees.length);
  return model.base + model.rate * sum(model.trees.slice(0, count).map(tree => predictCorrection(tree, feature)));
}

/** Constant pieces on (lower, upper], with breakpoints from actual saved trees.
 * A <= split takes its left value at the breakpoint. Drawing vertical joins is
 * a visual discontinuity marker, not an interpolated model prediction.
 */
export function boostingPredictionSegments(model, count, lower = -1, upper = 7) {
  integer(count, 'Tree count', 0, model.trees.length);
  finite(lower, 'Plot lower bound', -1000, 1000);
  finite(upper, 'Plot upper bound', -1000, 1000);
  if (lower >= upper) throw new Error('Plot bounds must be increasing.');
  const boundaries = new Set([lower, upper]);
  function visit(node) {
    if (node.leaf) return;
    if (node.threshold > lower && node.threshold < upper) boundaries.add(node.threshold);
    visit(node.left);
    visit(node.right);
  }
  model.trees.slice(0, count).forEach(visit);
  const ordered = [...boundaries].sort((a, b) => a - b);
  return ordered.slice(1).map((right, index) => ({
    left: ordered[index],
    right,
    value: predictBoosting(model, right, count)
  }));
}
export function sigmoid(score) {
  finite(score, 'Score', -1000, 1000);
  return score >= 0 ? 1 / (1 + Math.exp(-score)) : Math.exp(score) / (1 + Math.exp(score));
}
export function logisticLoss(target, score) {
  finite(target, 'Binary target', 0, 1);
  finite(score, 'Score', -1000, 1000);
  return Math.max(score, 0) - target * score + Math.log1p(Math.exp(-Math.abs(score)));
}
export function leafOptimum(gradients, hessians, lambda = 1, alpha = 0) {
  const gradient = vector(gradients, 'Gradients');
  const hessian = vector(hessians, 'Hessians', gradient.length, gradient.length);
  if (hessian.some(value => value < 0)) throw new Error('This convex quadratic model needs nonnegative Hessians.');
  finite(lambda, 'Lambda', 0, 20);
  finite(alpha, 'Alpha', 0, 20);
  const G = sum(gradient);
  const H = sum(hessian);
  const denominator = H + lambda;
  if (!(denominator >= 1e-10)) throw new Error('H + lambda must be at least 1e−10 in this numerical teaching model.');
  const softened = Math.sign(G) * Math.max(Math.abs(G) - alpha, 0);
  const weight = -softened / denominator;
  const objective = G * weight + denominator * weight ** 2 / 2 + alpha * Math.abs(weight);
  return {
    G,
    H,
    weight,
    objective,
    improvement: softened ** 2 / (2 * denominator),
    count: gradient.length
  };
}
export function splitStatistics(gradients, hessians, leftIndices, {
  lambda = 1,
  alpha = 0,
  gamma = 0
} = {}) {
  const gradient = vector(gradients, 'Gradients', 2);
  const hessian = vector(hessians, 'Hessians', gradient.length, gradient.length);
  finite(gamma, 'Gamma', 0, 20);
  if (!Array.isArray(leftIndices) || !leftIndices.length || leftIndices.length >= gradient.length) {
    throw new Error('A split needs two nonempty children.');
  }
  const leftSet = new Set();
  for (let index = 0; index < leftIndices.length; index += 1) {
    const value = leftIndices[index];
    integer(value, 'Row index', 0, gradient.length - 1);
    if (leftSet.has(value)) throw new Error('A row cannot be repeated in a child.');
    leftSet.add(value);
  }
  const rightIndices = gradient.map((_, index) => index).filter(index => !leftSet.has(index));
  const aggregate = indices => leafOptimum(indices.map(index => gradient[index]), indices.map(index => hessian[index]), lambda, alpha);
  const parent = leafOptimum(gradient, hessian, lambda, alpha);
  const left = aggregate([...leftSet]);
  const right = aggregate(rightIndices);
  const grossGain = left.improvement + right.improvement - parent.improvement;
  return {
    parent,
    left,
    right,
    leftIndices: [...leftSet],
    rightIndices,
    grossGain,
    netGain: grossGain - gamma,
    gamma
  };
}
export function newtonInvestigation({
  kind = 'square',
  split = 2,
  lambda = 1,
  alpha = 0,
  gamma = 0,
  rate = 0.3
} = {}) {
  if (!['square', 'logistic', 'confident'].includes(kind)) throw new Error('Unknown loss fixture.');
  integer(split, 'Left row count', 1, 4);
  finite(rate, 'Rate', 0, 1);
  const x = [1, 2, 3, 4, 5];
  const y = kind === 'square' ? [1, 2, 3, 4, 5] : [0, 0, 1, 1, 1];
  const before = x.map(() => kind === 'square' ? 2.5 : kind === 'confident' ? -4 : 0);
  const probabilities = before.map(sigmoid);
  const gradients = y.map((value, index) => (kind === 'square' ? before[index] : probabilities[index]) - value);
  const hessians = y.map((_, index) => kind === 'square' ? 1 : probabilities[index] * (1 - probabilities[index]));
  const statistics = splitStatistics(gradients, hessians, x.slice(0, split).map((_, index) => index), {
    lambda,
    alpha,
    gamma
  });
  const after = before.map((value, index) => value + rate * (index < split ? statistics.left.weight : statistics.right.weight));
  const loss = prediction => sum(y.map((value, index) => kind === 'square' ? (prediction[index] - value) ** 2 / 2 : logisticLoss(value, prediction[index])));
  return {
    x,
    y,
    before,
    after,
    probabilities,
    afterProbabilities: after.map(sigmoid),
    gradients,
    hessians,
    statistics,
    beforeLoss: loss(before),
    afterLoss: loss(after),
    kind,
    split,
    rate
  };
}
export function histogramInvestigation({
  coarse = true,
  missingTarget = 8
} = {}) {
  if (typeof coarse !== 'boolean') throw new Error('Coarse must be boolean.');
  finite(missingTarget, 'Missing-row target', 0, 10);
  const x = [1, 2, 3, 4, 5, 6, null];
  const y = [1, 1, 1, 9, 9, 9, missingTarget];
  const base = mean(y);
  const gradients = y.map(value => base - value);
  const hessians = y.map(() => 1);
  const cuts = coarse ? [2.5, 4.5] : [1.5, 2.5, 3.5, 4.5, 5.5];
  const candidates = cuts.flatMap(threshold => ['left', 'right'].map(missing => {
    const leftIndices = x.map((value, index) => (value === null ? missing === 'left' : value <= threshold) ? index : -1).filter(index => index >= 0);
    return {
      threshold,
      missing,
      ...splitStatistics(gradients, hessians, leftIndices, {
        lambda: 0
      })
    };
  }));
  const best = candidates.reduce((winner, candidate) => candidate.netGain > winner.netGain ? candidate : winner);
  const bins = cuts.map((_, index) => ({
    index,
    rows: []
  }));
  bins.push({
    index: cuts.length,
    rows: []
  });
  x.forEach((value, index) => {
    if (value !== null) bins[cuts.filter(cut => value > cut).length].rows.push(index);
  });
  return {
    x,
    y,
    base,
    gradients,
    hessians,
    cuts,
    bins,
    candidates,
    best,
    coarse,
    missingTarget
  };
}
function combinations(items, count) {
  if (count === 0) return [[]];
  if (items.length < count) return [];
  return [...combinations(items.slice(1), count - 1).map(rest => [items[0], ...rest]), ...combinations(items.slice(1), count)];
}
export function gossInvestigation({
  sample = 0,
  keep = 2,
  draw = 2
} = {}) {
  const gradients = [8, -6, 3, -2, 1, -1];
  integer(keep, 'Kept large gradients', 1, 3);
  integer(draw, 'Drawn small gradients', 1, gradients.length - keep);
  const order = gradients.map((_, index) => index).sort((a, b) => Math.abs(gradients[b]) - Math.abs(gradients[a]) || a - b);
  const retained = order.slice(0, keep);
  const remaining = order.slice(keep);
  const subsets = combinations(remaining, draw);
  integer(sample, 'Sample index', 0, subsets.length - 1);
  const probability = draw / remaining.length;
  const weight = remaining.length / draw;
  const totals = subsets.map(subset => sum(retained.map(index => gradients[index])) + weight * sum(subset.map(index => gradients[index])));
  const selected = subsets[sample];
  const contribution = gradients.map((value, index) => retained.includes(index) ? value : selected.includes(index) ? weight * value : 0);
  return {
    gradients,
    retained,
    remaining,
    subsets,
    selected,
    sample,
    probability,
    weight,
    contribution,
    totals,
    full: sum(gradients),
    estimate: totals[sample],
    average: mean(totals),
    averageSquare: mean(totals.map(value => value ** 2))
  };
}
export function bundleExclusive(rowsInput) {
  if (!Array.isArray(rowsInput) || rowsInput.length < 1 || rowsInput.length > 20) throw new Error('Use 1–20 three-feature rows.');
  const rows = Array.from({
    length: rowsInput.length
  }, (_, index) => {
    const row = vector(rowsInput[index], 'Bundle row', 3, 3);
    row.forEach(value => integer(value, 'Feature bin', 0, 2));
    if (row.filter(value => value !== 0).length > 1) throw new Error(`Row ${index + 1} has a conflict: exact bundling cannot recover both nonzero values.`);
    return row;
  });
  const encoded = rows.map(row => row.reduce((code, value, index) => value === 0 ? code : 2 * index + value, 0));
  const decoded = encoded.map(code => [0, 1, 2].map(index => code > 2 * index && code <= 2 * index + 2 ? code - 2 * index : 0));
  return {
    rows,
    encoded,
    decoded
  };
}
export const CATEGORY_ROWS = Object.freeze([Object.freeze({
  category: 'A',
  target: 0
}), Object.freeze({
  category: 'B',
  target: 1
}), Object.freeze({
  category: 'A',
  target: 1
}), Object.freeze({
  category: 'C',
  target: 1
}), Object.freeze({
  category: 'B',
  target: 0
}), Object.freeze({
  category: 'A',
  target: 1
})]);
export function orderedStatistics({
  order = [0, 1, 2, 3, 4, 5],
  row = 2,
  flipped = false,
  smoothing = 1,
  prior = 0.5
} = {}) {
  if (!Array.isArray(order) || order.length !== 6 || new Set(order).size !== 6) throw new Error('Use a permutation of all six rows.');
  Array.from({
    length: 6
  }, (_, index) => integer(order[index], 'Permutation row', 0, 5));
  integer(row, 'Selected row', 0, 5);
  if (typeof flipped !== 'boolean') throw new Error('Flipped must be boolean.');
  finite(smoothing, 'Smoothing', 0.25, 4);
  finite(prior, 'External prior', 0, 1);
  const rows = CATEGORY_ROWS.map((item, index) => ({
    ...item,
    id: index,
    target: index === row && flipped ? 1 - item.target : item.target
  }));
  const position = order.indexOf(row);
  const preceding = order.slice(0, position).filter(index => rows[index].category === rows[row].category);
  const sameCategory = rows.map((_, index) => index).filter(index => rows[index].category === rows[row].category);
  const encode = indices => (sum(indices.map(index => rows[index].target)) + smoothing * prior) / (indices.length + smoothing);
  const prefixes = order.map((index, current) => {
    const eligible = order.slice(0, current).filter(other => rows[other].category === rows[index].category);
    return {
      index,
      eligible,
      value: encode(eligible)
    };
  });
  return {
    rows,
    order: [...order],
    row,
    position,
    preceding,
    sameCategory,
    prefix: encode(preceding),
    naive: encode(sameCategory),
    numerator: sum(preceding.map(index => rows[index].target)) + smoothing * prior,
    denominator: preceding.length + smoothing,
    prefixes,
    unseen: prior,
    inference: encode(sameCategory),
    smoothing,
    prior
  };
}
export function validationInvestigation({
  rate = 0.3,
  depth = 2,
  rounds = 60,
  patience = 6
} = {}) {
  finite(rate, 'Rate', 0.05, 1);
  integer(depth, 'Depth', 1, 3);
  integer(rounds, 'Rounds', 1, 80);
  integer(patience, 'Patience', 1, 20);
  const x = Array.from({
    length: 18
  }, (_, index) => index / 3);
  const signal = value => 2 + 0.8 * value + 1.5 * Math.sin(value);
  const noise = [0.2, -0.3, 0.4, -0.5, 0.2, -0.1, 1.8, -0.4, 0.3, -0.2, 0.5, -1.5, 0.2, -0.4, 0.1, -0.2, 0.3, -0.1];
  const y = x.map((value, index) => signal(value) + noise[index]);
  const validationX = x.slice(0, -1).map(value => value + 1 / 6);
  const validationY = validationX.map(signal);
  const model = fitBoosting({
    x,
    y,
    rate,
    depth,
    rounds
  });
  const curves = model.stages.map(stage => ({
    round: stage.round,
    train: stage.mse,
    validation: meanSquaredError(validationY, validationX.map(value => predictBoosting(model, value, stage.round)))
  }));
  // Start selection at round 1, matching the three production training recipes.
  let best = 1;
  let stale = 0;
  let stopped = rounds;
  for (let round = 2; round <= rounds; round += 1) {
    if (curves[round].validation < curves[best].validation) {
      best = round;
      stale = 0;
    } else stale += 1;
    if (stale >= patience) {
      stopped = round;
      break;
    }
  }
  return {
    model,
    validationX,
    validationY,
    curves,
    best,
    stopped,
    patience,
    baselineValidation: curves[0].validation,
    provenance: 'Constructed 18 training rows with fixed noise; 17 interleaved validation rows use the stated noiseless signal. This is an algorithm illustration, not a population estimate or named-library benchmark.'
  };
}
export function growthPolicies() {
  // A deliberately specified candidate-gain tree illustrates frontier order only.
  // Gains are hypothetical policy inputs, not measured library fits.
  const candidates = {
    root: 10,
    L: 6,
    R: 2,
    LL: 5,
    LR: 0,
    RL: 1,
    RR: 0,
    LLL: 0,
    LLR: 0
  };
  function grow(policy) {
    const frontier = ['root'];
    const splits = [];
    while (splits.length < 3) {
      const eligible = frontier.filter(node => candidates[node] > 0);
      if (!eligible.length) break;
      eligible.sort((a, b) => policy === 'best' ? candidates[b] - candidates[a] || a.localeCompare(b) : (a === 'root' ? 0 : a.length) - (b === 'root' ? 0 : b.length) || a.localeCompare(b));
      const selected = eligible[0];
      frontier.splice(frontier.indexOf(selected), 1);
      frontier.push(selected === 'root' ? 'L' : `${selected}L`, selected === 'root' ? 'R' : `${selected}R`);
      splits.push(selected);
    }
    return {
      policy,
      splits,
      leaves: frontier,
      depth: Math.max(...frontier.map(node => node.length))
    };
  }
  return {
    candidates,
    level: grow('level'),
    best: grow('best')
  };
}
