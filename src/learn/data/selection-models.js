/** Pure models for the feature-selection lesson.
 *
 * Five different questions are answered here, and they are kept apart on
 * purpose, because conflating them is the mistake the lesson exists to correct:
 *
 *   1. Information in the observed data, before any model: entropy and mutual
 *      information from an exact count table, in bits.
 *   2. Subset search over a complete finite world: exact lookup accuracies over
 *      four declared states, not held-out estimates.
 *   3. A perturbation of a FIXED predictor: donor rows are exchanged inside one
 *      column while the prediction function is never refitted.
 *   4. Allocation of one output relative to a declared reference: coalition
 *      values from actual hybrid rows, then exact Shapley averages.
 *   5. Inference in one saved tree: float32 threshold comparisons matching
 *      scikit-learn's input convention, and coalition games built on it.
 *
 * Everything a figure draws is computed here, including the mosaic tile areas,
 * the waterfall segment endpoints, the bar scales and the cumulative fraction,
 * so a drawn claim is an asserted claim.
 *
 * Every entry point refuses input it cannot honour rather than substituting a
 * silent default: a non-finite number, a negative count, a donor list that is
 * not a permutation or a background with the wrong width all raise a RangeError.
 */

/* ------------------------------------------------------------------ guards */

export const limits = {
  /** I1 cell counts. The cap keeps tile rendering independent of sample size;
   * a cell is one rectangle whatever the count. */
  count: { minimum: 0, maximum: 10000 },
  /** I2 binary target labels attached to the four fixed input states. */
  label: { minimum: 0, maximum: 1 },
  /** I3 sensor readings, targets and the fixed linear coefficients. */
  sensor: { minimum: -10, maximum: 10 },
  target: { minimum: -10, maximum: 10 },
  coefficient: { minimum: -5, maximum: 5 },
  /** I4 explained coordinates, reference coordinates and interaction strength. */
  coordinate: { minimum: -10, maximum: 10 },
  gamma: { minimum: -5, maximum: 5 },
  referenceRows: { minimum: 1, maximum: 6 },
  /** The exhaustive coalition enumeration is deliberately bounded. */
  players: { minimum: 1, maximum: 10 },
  tolerance: 1e-12,
  reconstruction: 1e-10,
};

export function checkFinite(value, name) {
  if (typeof value !== 'number' || !Number.isFinite(value)) {
    throw new RangeError(`${name} must be a finite number; handle a missing value before computing.`);
  }
  return value;
}
export function checkRange(value, range, name) {
  checkFinite(value, name);
  if (value < range.minimum || value > range.maximum) {
    throw new RangeError(`Keep ${name} between ${range.minimum} and ${range.maximum}.`);
  }
  return value;
}
export function checkInteger(value, name) {
  if (!Number.isInteger(value)) throw new RangeError(`${name} must be a whole number.`);
  return value;
}

/* ============================================================ §2 information */

/** log2 with the limiting convention 0 log 0 = 0. A literal logarithm of zero
 * is never evaluated. */
export function bits(probability) {
  checkFinite(probability, 'a probability');
  if (probability < 0) throw new RangeError('A probability cannot be negative.');
  return probability === 0 ? 0 : -probability * Math.log2(probability);
}

/** Entropy of a discrete distribution given as probabilities, in bits. */
export function entropy(distribution) {
  return distribution.reduce((sum, probability) => sum + bits(probability), 0);
}

function checkTable(counts, name = 'the count table') {
  if (!Array.isArray(counts) || counts.length === 0) throw new RangeError(`${name} needs at least one row.`);
  const width = Array.isArray(counts[0]) ? counts[0].length : 0;
  if (width === 0) throw new RangeError(`${name} needs at least one column.`);
  const rows = counts.map((row, index) => {
    if (!Array.isArray(row) || row.length !== width) throw new RangeError(`${name} must be rectangular.`);
    return row.map((value, column) => {
      checkFinite(value, `count (${index}, ${column})`);
      if (value < 0) throw new RangeError('Counts must be nonnegative.');
      return value;
    });
  });
  const total = rows.reduce((sum, row) => sum + row.reduce((inner, value) => inner + value, 0), 0);
  if (!(total > 0)) throw new RangeError('The counts must have a positive total.');
  return { rows, total, width, height: rows.length };
}

/** Entropy, conditional entropy and mutual information from an exact table of
 * counts, by two independent routes that must agree.
 *
 * Rows are values of the input X, columns values of the target Y. An empty row
 * contributes zero rather than requiring a conditional distribution, and a
 * degenerate target column simply gives H(Y) = I = 0 instead of an error. */
export function informationFromCounts(counts) {
  const { rows, total, width, height } = checkTable(counts);
  const joint = rows.map(row => row.map(value => value / total));
  const rowTotals = rows.map(row => row.reduce((sum, value) => sum + value, 0));
  const columnTotals = Array.from({ length: width }, (_, column) =>
    rows.reduce((sum, row) => sum + row[column], 0));
  const rowProbabilities = rowTotals.map(value => value / total);
  const columnProbabilities = columnTotals.map(value => value / total);

  const targetEntropy = entropy(columnProbabilities);
  const conditionals = rows.map((row, index) => {
    const weight = rowProbabilities[index];
    const distribution = weight === 0 ? row.map(() => 0) : row.map(value => value / rowTotals[index]);
    return {
      index,
      count: rowTotals[index],
      weight,
      occupied: rowTotals[index] > 0,
      distribution,
      entropy: rowTotals[index] > 0 ? entropy(distribution) : 0,
    };
  });
  const conditionalEntropy = conditionals.reduce((sum, row) => sum + row.weight * row.entropy, 0);

  const cells = [];
  for (let x = 0; x < height; x += 1) {
    for (let y = 0; y < width; y += 1) {
      const independent = rowProbabilities[x] * columnProbabilities[y];
      const occupied = joint[x][y] > 0;
      cells.push({
        x, y, count: rows[x][y], joint: joint[x][y], independent, occupied,
        contribution: occupied ? joint[x][y] * Math.log2(joint[x][y] / independent) : 0,
      });
    }
  }
  // KL = sum q[(1+r)log(1+r)-r], r=(p-q)/q. The linear terms
  // sum to zero. Use count products for r and a Taylor remainder near zero
  // so nearly independent integer tables do not lose their positive MI.
  const independentExactly = cells.every(cell => rows[cell.x][cell.y] * total === rowTotals[cell.x] * columnTotals[cell.y]);
  const stableInformation = cells.reduce((sum, cell) => {
    const product = rowTotals[cell.x] * columnTotals[cell.y];
    if (product === 0) return sum;
    const r = (rows[cell.x][cell.y] * total - product) / product;
    const remainder = r === -1 ? 1 : Math.abs(r) < 1e-4
      ? r * r * (0.5 + r * (-1 / 6 + r * (1 / 12 + r * (-1 / 20 + r / 30))))
      : (1 + r) * Math.log1p(r) - r;
    return sum + cell.independent * remainder / Math.LN2;
  }, 0);
  const direct = cells.reduce((sum, cell) => sum + cell.contribution, 0);
  const difference = targetEntropy - conditionalEntropy;
  return {
    counts: rows, total, joint, rowTotals, columnTotals, rowProbabilities, columnProbabilities,
    conditionals, cells,
    targetEntropy,
    conditionalEntropy,
    mutualInformation: independentExactly ? 0 : stableInformation,
    entropyDifference: difference,
    independentExactly,
    mutualInformationDirect: direct,
    // The two routes are the section's identity; they must agree exactly.
    agrees: Math.abs(direct - difference) <= 1e-12,
    // Zero MI on a sample is an estimate, never a certificate of independence.
    empirical: true,
  };
}

/** Unit-square mosaic tiles whose AREA is the cell probability: a column of
 * width p(x) split by the conditional p(y | x). An empty row occupies no width
 * and is reported rather than dropped. */
export function probabilityMosaic(counts) {
  const info = informationFromCounts(counts);
  let left = 0;
  const tiles = [];
  info.conditionals.forEach(row => {
    const width = row.weight;
    let top = 0;
    info.columnProbabilities.forEach((_, column) => {
      const height = row.occupied ? row.distribution[column] : 0;
      tiles.push({
        x: row.index, y: column, left, top, width, height,
        area: width * height,
        joint: info.joint[row.index][column],
        occupied: info.joint[row.index][column] > 0,
      });
      top += height;
    });
    left += width;
  });
  return { tiles, info, spannedWidth: left };
}

/** Mutual information of a joint table with more than two input rows, used for
 * the XOR pair and the exact-copy comparison. */
export function jointInformation(counts) {
  return informationFromCounts(counts).mutualInformation;
}

/** The four equally likely XOR states, their one-dimensional projections and
 * the three exact information quantities. */
export function xorWorld(labels = [0, 1, 1, 0]) {
  const states = [[0, 0], [0, 1], [1, 0], [1, 1]];
  const targets = labels.map((value, index) => {
    checkInteger(value, `label ${index}`);
    if (value !== 0 && value !== 1) throw new RangeError('This world uses binary target labels.');
    return value;
  });
  const marginal = axis => {
    const table = [[0, 0], [0, 0]];
    states.forEach((state, index) => { table[state[axis]][targets[index]] += 1; });
    return { table, information: jointInformation(table) };
  };
  const jointTable = states.map((state, index) => {
    const row = [0, 0];
    row[targets[index]] = 1;
    return row;
  });
  const projections = [0, 1].map(axis => {
    const { table, information } = marginal(axis);
    return {
      axis,
      information,
      values: [0, 1].map(value => ({
        value,
        stateIds: states.map((state, index) => [state, index]).filter(([state]) => state[axis] === value)
          .map(([, index]) => index),
        counts: table[value],
      })),
    };
  });
  return {
    states, targets, projections,
    jointInformation: jointInformation(jointTable),
    // A point at unit coordinates; the square is drawn from these, not placed.
    points: states.map((state, index) => ({ index, a: state[0], b: state[1], target: targets[index] })),
  };
}

/** An exact copy reveals the same bit twice: each column carries one bit and
 * the pair still carries only one. */
export function duplicateColumns() {
  const single = [[1, 0], [0, 1]];
  const pairTable = [[1, 0], [0, 0], [0, 0], [0, 1]];
  return {
    first: jointInformation(single),
    second: jointInformation(single),
    sum: jointInformation(single) * 2,
    joint: jointInformation(pairTable),
    // Adding the two individual values double counts the same bit.
    doubleCounted: jointInformation(single) * 2 !== jointInformation(pairTable),
  };
}

/* ========================================================= §3 subset search */

const SUBSET_STATES = [[0, 0], [0, 1], [1, 0], [1, 1]];
const SUBSET_NAMES = ['A', 'B'];

function checkLabels(labels) {
  if (!Array.isArray(labels) || labels.length !== 4) {
    throw new RangeError('This declared world has exactly four states.');
  }
  return labels.map((value, index) => {
    checkInteger(value, `label ${index}`);
    if (value !== 0 && value !== 1) throw new RangeError('Use binary target labels, 0 or 1.');
    return value;
  });
}

export function maskFeatures(mask) {
  return SUBSET_NAMES.filter((_, index) => (mask & (1 << index)) !== 0);
}
export function maskLabel(mask) {
  const names = maskFeatures(mask);
  return names.length === 0 ? '{ }' : `{${names.join(', ')}}`;
}

/** Every subset of the two inputs, scored exactly over the complete four-state
 * world. A subset's predictor is a lookup table: group the states that share the
 * retained coordinates and predict the most common target in that group, with a
 * tie predicting 0. These are exact scores over a declared finite world, not
 * held-out estimates. */
export function subsetWorld(labels) {
  const targets = checkLabels(labels);
  const subsets = [0, 1, 2, 3].map(mask => {
    const columns = [0, 1].filter(index => (mask & (1 << index)) !== 0);
    const key = state => columns.map(column => state[column]).join('|');
    const groups = new Map();
    SUBSET_STATES.forEach((state, index) => {
      const id = key(state);
      if (!groups.has(id)) groups.set(id, { key: id, values: columns.map(column => state[column]), stateIds: [] });
      groups.get(id).stateIds.push(index);
    });
    const described = [...groups.values()].map(group => {
      const counts = [0, 0];
      group.stateIds.forEach(index => { counts[targets[index]] += 1; });
      const tie = counts[0] === counts[1];
      return { ...group, counts, tie, majority: counts[1] > counts[0] ? 1 : 0 };
    });
    const predictions = SUBSET_STATES.map((state, index) => {
      const group = described.find(entry => entry.stateIds.includes(index));
      return group.majority;
    });
    const correct = predictions.filter((value, index) => value === targets[index]).length;
    return {
      mask, columns, features: maskFeatures(mask), label: maskLabel(mask),
      groups: described, predictions, correct, accuracy: correct / 4,
    };
  });
  return { states: SUBSET_STATES, targets, subsets, exact: true };
}

/** Greedy forward selection over that world.
 *
 * `policy` is part of the algorithm, not an administrative detail:
 *   'strict'   accepts an addition only when its accuracy is strictly larger.
 *   'forceTwo' accepts the best available addition twice, whatever it scores.
 * Candidates are inspected in the order A then B, and A breaks an equal score.
 * Every evaluated candidate is returned, separately from the accepted path. */
export function forwardSelection(labels, policy = 'strict') {
  if (policy !== 'strict' && policy !== 'forceTwo') {
    throw new RangeError('The declared policies are strict improvement or two forced additions.');
  }
  const world = subsetWorld(labels);
  const scoreOf = mask => world.subsets.find(entry => entry.mask === mask).accuracy;
  const steps = [];
  let current = 0;
  let stopReason = null;
  while (steps.length < 2) {
    const available = [0, 1].filter(index => (current & (1 << index)) === 0);
    if (available.length === 0) { stopReason = 'every input is already selected'; break; }
    const candidates = available.map(index => ({
      index, feature: SUBSET_NAMES[index], mask: current | (1 << index), score: scoreOf(current | (1 << index)),
    }));
    // The first strictly larger score wins, so an equal score leaves A in front.
    const best = candidates.reduce((left, right) => (right.score > left.score ? right : left));
    const improves = best.score > scoreOf(current) + 1e-12;
    const accepted = policy === 'forceTwo' ? true : improves;
    steps.push({
      step: steps.length + 1, from: current, fromScore: scoreOf(current),
      candidates, best, improves, accepted,
      to: accepted ? best.mask : current,
    });
    if (!accepted) { stopReason = 'stopped because no strictly higher score was available'; break; }
    current = best.mask;
  }
  if (stopReason === null) stopReason = 'stopped because the forced number of additions was reached';
  return {
    policy, world, steps, finalMask: current, finalScore: scoreOf(current), stopReason,
    path: [0, ...steps.filter(step => step.accepted).map(step => step.to)],
    reachedPair: current === 3,
  };
}

/** How many fits a declared search actually performs. The counts follow the
 * search, not a universal "one fit per feature" rule. */
export function searchFitCounts({ features, keep, folds = 1, method = 'forward' }) {
  checkInteger(features, 'the number of inputs');
  checkInteger(keep, 'the retained size');
  checkInteger(folds, 'the number of folds');
  if (features < 1 || keep < 1 || keep > features || folds < 1) {
    throw new RangeError('Use at least one fold and a retained size between one and the number of inputs.');
  }
  if (method === 'forward') {
    let subsets = 0;
    for (let size = 0; size < keep; size += 1) subsets += features - size;
    return {
      method, features, keep, folds, subsets,
      candidateFits: subsets * folds, finalRefits: 1, total: subsets * folds + 1,
    };
  }
  if (method === 'rfe') {
    // Fit at each size from `features` down to keep + 1 to choose a removal,
    // then fit the retained model once. No cross-validation in this protocol.
    const rankingFits = features - keep;
    return {
      method, features, keep, folds: 1, subsets: rankingFits,
      candidateFits: rankingFits, finalRefits: 1, total: rankingFits + 1,
    };
  }
  throw new RangeError('The declared methods are forward selection and simple RFE.');
}

/* =========================================================== §4 permutation */

function checkRows(rows, width = 2) {
  if (!Array.isArray(rows) || rows.length < 2) throw new RangeError('Use at least two rows.');
  return rows.map((row, index) => {
    if (!Array.isArray(row) || row.length !== width) {
      throw new RangeError(`Row ${index} needs exactly ${width} input values.`);
    }
    return row.map((value, column) => checkFinite(value, `row ${index} input ${column + 1}`));
  });
}

/** A donor ordering must be a permutation: every source row used exactly once.
 * Fixed points are allowed, and are the reason a shuffle need not change every
 * case. */
export function checkDonor(donor, length) {
  if (!Array.isArray(donor) || donor.length !== length) {
    throw new RangeError(`The donor ordering needs one source row per assessed row (${length}).`);
  }
  const seen = new Set();
  donor.forEach(value => {
    checkInteger(value, 'a donor row');
    if (value < 0 || value >= length) throw new RangeError(`Donor rows run from 0 to ${length - 1}.`);
    if (seen.has(value)) throw new RangeError('Each source row must appear exactly once in the donor ordering.');
    seen.add(value);
  });
  return donor;
}

export function linearPredictions(rows, coefficients) {
  return rows.map(row => row.reduce((sum, value, column) => sum + value * coefficients[column], 0));
}

/** Shuffle one column, or a declared group of columns with the SAME donor
 * ordering, inside a FIXED linear predictor.
 *
 * Nothing is refitted: the coefficients are read, never updated. A group uses
 * one donor map so the within-group pairing survives while its relation to the
 * target and the remaining columns is disturbed. */
export function permutationExperiment({ rows, target, coefficients, donor, columns }) {
  const design = checkRows(rows, 2);
  if (!Array.isArray(target) || target.length !== design.length) throw new RangeError('Give one target per row.');
  const targets = target.map((value, index) => checkFinite(value, `the target of row ${index}`));
  if (!Array.isArray(coefficients) || coefficients.length !== 2) throw new RangeError('This fixed predictor has two coefficients.');
  const weights = coefficients.map((value, index) => checkFinite(value, `coefficient ${index + 1}`));
  checkDonor(donor, design.length);
  if (!Array.isArray(columns) || columns.length === 0 || columns.some(column => column !== 0 && column !== 1)) {
    throw new RangeError('Permute the first column, the second column, or both as one group.');
  }
  const chosen = [...new Set(columns)].sort();

  const basePredictions = linearPredictions(design, weights);
  const baseSquared = basePredictions.map((value, index) => (targets[index] - value) ** 2);
  const baseMse = baseSquared.reduce((sum, value) => sum + value, 0) / design.length;

  const altered = design.map((row, index) => row.map((value, column) =>
    (chosen.includes(column) ? design[donor[index]][column] : value)));
  const alteredPredictions = linearPredictions(altered, weights);
  const alteredSquared = alteredPredictions.map((value, index) => (targets[index] - value) ** 2);
  const alteredMse = alteredSquared.reduce((sum, value) => sum + value, 0) / design.length;

  return {
    rows: design, targets, coefficients: weights, donor, columns: chosen,
    grouped: chosen.length > 1,
    basePredictions, baseSquared, baseMse,
    altered, alteredPredictions, alteredSquared, alteredMse,
    // Difference of squares per row avoids subtracting two almost equal MSEs.
    increase: alteredPredictions.reduce((sum, value, index) => {
      const before = targets[index] - basePredictions[index];
      const after = targets[index] - value;
      return sum + (after - before) * (after + before);
    }, 0) / design.length,
    // A valid shuffle can leave rows untouched; that is not a broken shuffle.
    fixedPoints: donor.map((source, index) => (source === index ? index : -1)).filter(index => index >= 0),
    unchangedRows: design.map((row, index) => chosen.every(column => row[column] === altered[index][column]))
      .map((same, index) => (same ? index : -1)).filter(index => index >= 0),
    perRow: design.map((row, index) => ({
      index, donor: donor[index], original: row, hybrid: altered[index],
      changed: [0, 1].map(column => chosen.includes(column) && row[column] !== altered[index][column]),
      basePrediction: basePredictions[index], alteredPrediction: alteredPredictions[index],
      baseSquared: baseSquared[index], alteredSquared: alteredSquared[index],
      target: targets[index],
    })),
  };
}

/** The manuscript's three fixed predictors against the three perturbations. A
 * predictor that never reads a column cannot start reading it once another
 * column is corrupted; that is what the zeros in this table mean. */
export function permutationSuite({ rows, target, donor, models }) {
  return models.map(model => ({
    ...model,
    results: [[0], [1], [0, 1]].map(columns => {
      const run = permutationExperiment({ rows, target, coefficients: model.coefficients, donor, columns });
      return { columns, increase: run.increase, alteredMse: run.alteredMse, baseMse: run.baseMse, run };
    }),
  }));
}

/* ============================================================== §5 coalitions */

function checkInstance(instance) {
  if (!Array.isArray(instance) || instance.length === 0) throw new RangeError('Use one input vector.');
  const dimension = instance.length;
  checkRange(dimension, limits.players, 'the number of features');
  return instance.map((value, index) => checkFinite(value, `coordinate ${index + 1}`));
}

function checkBackground(background, dimension) {
  if (!Array.isArray(background) || background.length === 0) throw new RangeError('The reference needs at least one row.');
  return background.map((row, index) => {
    if (!Array.isArray(row) || row.length !== dimension) {
      throw new RangeError(`Reference row ${index} needs ${dimension} coordinates.`);
    }
    return row.map((value, column) => checkFinite(value, `reference row ${index} coordinate ${column + 1}`));
  });
}

/** One coalition's hybrid rows: retain the instance values in S, take every
 * other coordinate from the SAME donor row. Missing coordinates therefore stay
 * together. */
export function hybridRows(instance, background, mask) {
  const point = checkInstance(instance);
  const reference = checkBackground(background, point.length);
  return reference.map(row => row.map((value, column) => ((mask & (1 << column)) !== 0 ? point[column] : value)));
}

/** The complete background-replacement game: every coalition value is the mean
 * model output over the actual hybrid rows. */
export function coalitionGame({ predict, instance, background, keepHybrids = false }) {
  const point = checkInstance(instance);
  const reference = checkBackground(background, point.length);
  const dimension = point.length;
  const masks = [];
  for (let mask = 0; mask < (1 << dimension); mask += 1) {
    const rows = hybridRows(point, reference, mask);
    const outputs = predict(rows);
    if (!Array.isArray(outputs) || outputs.length !== reference.length) {
      throw new RangeError('predict must return one value per reference row.');
    }
    outputs.forEach((value, index) => checkFinite(value, `the prediction of hybrid row ${index}`));
    masks.push({
      mask,
      kept: Array.from({ length: dimension }, (_, column) => (mask & (1 << column)) !== 0),
      size: outputs.length,
      value: outputs.reduce((sum, value) => sum + value, 0) / outputs.length,
      outputs,
      rows: keepHybrids ? rows : undefined,
    });
  }
  return {
    dimension, instance: point, background: reference, masks,
    values: masks.map(entry => entry.value),
    baseline: masks[0].value,
    prediction: masks[masks.length - 1].value,
  };
}

/** Exact Shapley values from the 2^d coalition values, by the stated weights. */
export function shapleyValues(values) {
  if (!Array.isArray(values)) throw new RangeError('Give the coalition values as a list.');
  const dimension = Math.round(Math.log2(values.length));
  if ((1 << dimension) !== values.length) throw new RangeError('A game needs a power-of-two number of coalition values.');
  checkRange(dimension, limits.players, 'the number of features');
  values.forEach((value, index) => checkFinite(value, `coalition value ${index}`));
  const factorial = n => { let product = 1; for (let i = 2; i <= n; i += 1) product *= i; return product; };
  return Array.from({ length: dimension }, (_, player) => {
    let total = 0;
    for (let mask = 0; mask < values.length; mask += 1) {
      if ((mask & (1 << player)) !== 0) continue;
      let size = 0;
      for (let bit = 0; bit < dimension; bit += 1) if ((mask & (1 << bit)) !== 0) size += 1;
      const weight = factorial(size) * factorial(dimension - size - 1) / factorial(dimension);
      total += weight * (values[mask | (1 << player)] - values[mask]);
    }
    return total;
  });
}

/** Every arrival order and the increment each player takes in it. Averaging
 * these increments is the same number the weighted formula gives; both are
 * returned so the equality can be shown rather than asserted in prose. */
export function orderingPaths(values) {
  const dimension = Math.round(Math.log2(values.length));
  const orders = [];
  const walk = (remaining, sequence) => {
    if (remaining.length === 0) { orders.push(sequence); return; }
    remaining.forEach(player => walk(remaining.filter(other => other !== player), [...sequence, player]));
  };
  walk(Array.from({ length: dimension }, (_, index) => index), []);
  const paths = orders.map(order => {
    let mask = 0;
    const steps = order.map(player => {
      const before = mask;
      mask |= (1 << player);
      return { player, before, after: mask, from: values[before], to: values[mask], increment: values[mask] - values[before] };
    });
    return { order, steps, total: steps.reduce((sum, step) => sum + step.increment, 0) };
  });
  const averaged = Array.from({ length: dimension }, (_, player) =>
    paths.reduce((sum, path) => sum + path.steps.find(step => step.player === player).increment, 0) / paths.length);
  const weighted = shapleyValues(values);
  return {
    orders, paths, averaged, weighted,
    // A telescoping sum along every order gives v(F) - v(empty).
    agrees: averaged.every((value, index) => Math.abs(value - weighted[index]) <= 1e-12),
  };
}

/** f(a, b) = a + b + gamma * a * b, as a row-wise predictor. */
export function polynomialModel(gamma) {
  checkRange(gamma, limits.gamma, 'the interaction coefficient');
  return rows => rows.map(row => row[0] + row[1] + gamma * row[0] * row[1]);
}

/** The complete two-feature explanation: coalition values from actual hybrid
 * rows, both arrival orders, and the exact allocation. */
export function explainPolynomial({ instance, background, gamma }) {
  const predict = polynomialModel(gamma);
  const game = coalitionGame({ predict, instance, background, keepHybrids: true });
  const phi = shapleyValues(game.values);
  const paths = orderingPaths(game.values);
  const total = phi.reduce((sum, value) => sum + value, 0);
  return {
    ...game, gamma, phi, paths,
    total,
    reconstruction: game.baseline + total,
    efficiencyError: game.baseline + total - game.prediction,
    // The mean model output over the reference is not the model at the mean input.
    meanReferenceInput: game.background[0].map((_, column) =>
      game.background.reduce((sum, row) => sum + row[column], 0) / game.background.length),
    outputOfMeanInput: predict([game.background[0].map((_, column) =>
      game.background.reduce((sum, row) => sum + row[column], 0) / game.background.length)])[0],
    // With gamma = 0 no arrival order changes any increment.
    orderIndependent: paths.paths.every(path =>
      path.steps.every(step => Math.abs(step.increment - paths.paths[0].steps.find(other => other.player === step.player).increment) <= 1e-12)),
  };
}

/** Two missing-feature rules on the same model and instance: fair binary
 * duplicates X1 = X2 with f(x) = x1, explained at (1, 1).
 *
 * Conditional: learning either coordinate reveals both.
 * Replacement: the second coordinate never changes a coalition value, because
 * the model function never reads it.
 * Neither is a claim about changing a real instrument. */
export function dependentGames() {
  const conditionalValues = [0.5, 1, 1, 1];
  // The replacement game built from its own actual hybrid rows.
  const replacement = coalitionGame({
    predict: rows => rows.map(row => row[0]),
    instance: [1, 1],
    background: [[0, 0], [1, 1]],
    keepHybrids: true,
  });
  return {
    conditional: { values: conditionalValues, phi: shapleyValues(conditionalValues) },
    replacement: { values: replacement.values, phi: shapleyValues(replacement.values), game: replacement },
    // The impossible hybrid (1, 0) is part of the replacement game and is shown
    // rather than hidden: that is exactly what distinguishes the two rules.
    impossibleHybrids: replacement.masks.flatMap(entry =>
      (entry.rows ?? []).map((row, index) => ({ mask: entry.mask, index, row }))
        .filter(item => item.row[0] !== item.row[1])),
  };
}

/** Waterfall segments from a baseline and signed contributions. The last
 * segment's end is the reconstructed output, computed rather than placed. */
export function waterfall(baseline, contributions) {
  checkFinite(baseline, 'the baseline');
  let running = baseline;
  const segments = contributions.map((value, index) => {
    checkFinite(value, `contribution ${index + 1}`);
    const start = running;
    running += value;
    return {
      index, value, start, end: running,
      low: Math.min(start, running), high: Math.max(start, running),
      sign: value === 0 ? 0 : Math.sign(value),
    };
  });
  const extent = [baseline, running, ...segments.flatMap(segment => [segment.low, segment.high])];
  return {
    baseline, segments, total: running - baseline, reconstruction: running,
    minimum: Math.min(...extent), maximum: Math.max(...extent),
  };
}

/** A cumulative fraction of total absolute attribution. It ends at exactly one
 * when the total is positive; with a zero total it is undefined, and says so. */
export function cumulativeAttributionFraction(values) {
  const magnitudes = values.map((value, index) => Math.abs(checkFinite(value, `attribution ${index + 1}`)));
  const total = magnitudes.reduce((sum, value) => sum + value, 0);
  if (!(total > 0)) return { defined: false, total, fractions: values.map(() => null), ordered: [] };
  const ordered = magnitudes.map((magnitude, index) => ({ index, magnitude }))
    .sort((left, right) => right.magnitude - left.magnitude);
  let running = 0;
  const fractions = ordered.map(entry => {
    running += entry.magnitude;
    return { ...entry, cumulative: running, fraction: running / total };
  });
  return { defined: true, total, ordered, fractions, endsAtOne: Math.abs(fractions[fractions.length - 1].fraction - 1) <= 1e-12 };
}

/* ================================================== §6 the saved wine tree */

/** scikit-learn converts the input array to float32 before a tree compares it
 * with its stored double threshold. Reproducing that cast is the difference
 * between agreeing with the fitted model and disagreeing near a split. */
export function toFloat32(value) {
  return Math.fround(checkFinite(value, 'a measurement'));
}

function checkTree(tree) {
  const { childrenLeft, childrenRight, feature, threshold, value } = tree;
  if (!Array.isArray(childrenLeft) || childrenLeft.length !== childrenRight.length
    || feature.length !== childrenLeft.length || threshold.length !== childrenLeft.length
    || value.length !== childrenLeft.length) {
    throw new RangeError('The saved tree must have one entry per node in every array.');
  }
  return tree;
}

/** Walk one row down the saved tree, recording the actual comparison at each
 * split. `<=` takes the left child, as the fitted estimator does. */
export function treeDecision(tree, row) {
  checkTree(tree);
  if (!Array.isArray(row)) throw new RangeError('Give one measurement per input field.');
  const path = [];
  let node = 0;
  let guard = 0;
  while (tree.childrenLeft[node] !== -1) {
    guard += 1;
    if (guard > tree.childrenLeft.length) throw new RangeError('This saved tree does not terminate.');
    const field = tree.feature[node];
    const raw = checkFinite(row[field], `the measurement for field ${field}`);
    const cast = toFloat32(raw);
    const goesLeft = cast <= tree.threshold[node];
    path.push({
      node, field, raw, cast, threshold: tree.threshold[node], goesLeft,
      next: goesLeft ? tree.childrenLeft[node] : tree.childrenRight[node],
    });
    node = goesLeft ? tree.childrenLeft[node] : tree.childrenRight[node];
  }
  return { path, leaf: node, probabilities: tree.value[node], samples: tree.samples ? tree.samples[node] : null };
}

export function treeProbability(tree, row, classIndex = 0) {
  return treeDecision(tree, row).probabilities[classIndex];
}

/** The hard prediction: argmax over the leaf distribution, first maximum wins,
 * which is the estimator's own tie rule. */
export function treeClass(tree, row, classes) {
  const probabilities = treeDecision(tree, row).probabilities;
  let best = 0;
  for (let index = 1; index < probabilities.length; index += 1) {
    if (probabilities[index] > probabilities[best]) best = index;
  }
  return classes[best];
}

/** Weighted training-impurity decrease per field, normalised to sum to one.
 * This records how this fitted tree partitioned its training criterion. It is
 * not a held-out performance quantity and shares no axis with one. */
export function impurityImportance(tree, fieldCount) {
  checkTree(tree);
  const total = tree.samples[0];
  const raw = new Array(fieldCount).fill(0);
  const nodes = [];
  tree.childrenLeft.forEach((left, node) => {
    if (left === -1) return;
    const right = tree.childrenRight[node];
    const decrease = (tree.samples[node] * tree.impurity[node]
      - tree.samples[left] * tree.impurity[left]
      - tree.samples[right] * tree.impurity[right]) / total;
    raw[tree.feature[node]] += decrease;
    nodes.push({ node, field: tree.feature[node], decrease, samples: tree.samples[node] });
  });
  const sum = raw.reduce((running, value) => running + value, 0);
  return {
    raw, nodes, sum,
    normalized: raw.map(value => (sum > 0 ? value / sum : 0)),
    unit: 'normalized training-impurity decrease',
  };
}

/** The four-field coalition game on the saved tree and a declared reference.
 * Every coalition value is an average over the actual hybrid rows; the leaf
 * tally says which leaves those hybrids reached. */
export function explainTreeInstance({ tree, instance, background, classIndex = 0, keepHybrids = false }) {
  const game = coalitionGame({
    predict: rows => rows.map(row => treeProbability(tree, row, classIndex)),
    instance, background, keepHybrids,
  });
  const phi = shapleyValues(game.values);
  const total = phi.reduce((sum, value) => sum + value, 0);
  const leafTally = game.masks.map(entry => {
    const counts = new Map();
    const rows = entry.rows ?? hybridRows(instance, background, entry.mask);
    rows.forEach(row => {
      const leaf = treeDecision(tree, row).leaf;
      counts.set(leaf, (counts.get(leaf) ?? 0) + 1);
    });
    return {
      mask: entry.mask, value: entry.value,
      leaves: [...counts.entries()].sort((left, right) => left[0] - right[0])
        .map(([leaf, count]) => ({ leaf, count, probability: tree.value[leaf][classIndex] })),
    };
  });
  return {
    ...game, phi, total,
    reconstruction: game.baseline + total,
    efficiencyError: game.baseline + total - game.prediction,
    reconstructs: Math.abs(game.baseline + total - game.prediction) <= limits.reconstruction,
    leafTally,
    decision: treeDecision(tree, instance),
  };
}

/** The threshold to DRAW for a split, given the control a learner types into.
 *
 * A drawn threshold is a rule the learner will reason from, so it must not
 * disagree with the rule the tree applies at any value they can actually enter.
 * Rounding to a fixed number of decimals does not guarantee that: node 4's
 * threshold 1.5899999737739563 rounds to 1.59, but 1.59 casts to float32 ABOVE
 * it and takes the right branch, so a learner reading "≤ 1.59" and typing 1.59
 * is told their correct reasoning is wrong.
 *
 * Instead, return the largest value on the control's own grid that the tree
 * still sends left. Because Math.fround is monotonic, every grid value at or
 * below it goes left and every grid value above it goes right, so the drawn rule
 * and the applied rule agree at EVERY enterable value. The exact threshold is
 * always published beside the drawing.
 *
 * Returns null when no enterable value goes left, in which case there is no
 * grid-equivalent rule and the caller should show the exact threshold. */
export function drawnThreshold(threshold, control) {
  checkFinite(threshold, 'the threshold');
  const { minimum, maximum, decimals } = control;
  checkInteger(decimals, 'the control decimals');
  if (!(decimals >= 0 && decimals <= 9)) throw new RangeError('Use between zero and nine decimals.');
  if (!(maximum > minimum)) throw new RangeError('A control needs a positive range.');
  const scale = 10 ** decimals;
  const low = Math.round(minimum * scale);
  const high = Math.round(maximum * scale);
  const goesLeft = step => toFloat32(step / scale) <= threshold;
  if (!goesLeft(low)) return null;
  if (goesLeft(high)) return { value: high / scale, decimals, atCeiling: true };
  // Binary search for the last grid step that still goes left.
  let lower = low;
  let upper = high;
  while (upper - lower > 1) {
    const middle = Math.floor((lower + upper) / 2);
    if (goesLeft(middle)) lower = middle; else upper = middle;
  }
  return { value: lower / scale, decimals, atCeiling: false };
}

/** Does the drawn rule agree with the applied rule at every enterable value?
 * Returns the disagreements, so a verifier can assert the list is empty rather
 * than sampling points and hoping. */
export function thresholdDisagreements(threshold, control, drawn) {
  const { minimum, maximum, decimals } = control;
  const scale = 10 ** decimals;
  const misses = [];
  for (let step = Math.round(minimum * scale); step <= Math.round(maximum * scale); step += 1) {
    const value = step / scale;
    const applied = toFloat32(value) <= threshold;
    const asDrawn = value <= drawn;
    if (applied !== asDrawn) misses.push({ value, applied, asDrawn });
  }
  return misses;
}

/** Is an edited measurement outside the observed fitting column? An edit may
 * still be inspected; it is marked as extrapolation rather than presented as a
 * measured specimen. */
export function extrapolationFlags(values, ranges) {
  return values.map((value, index) => {
    checkFinite(value, `field ${index + 1}`);
    const [low, high] = ranges[index];
    return { index, value, low, high, below: value < low, above: value > high, outside: value < low || value > high };
  });
}

/* ==================================================== §7 and practice helpers */

export function sigmoid(z) {
  return 1 / (1 + Math.exp(-checkFinite(z, 'the margin')));
}

/** The probability of at least one rejection among independent valid null
 * tests. Dependence changes this arithmetic, and selection after seeing results
 * changes it again. */
export function familywiseProbability(tests, alpha) {
  checkInteger(tests, 'the number of tests');
  checkRange(alpha, { minimum: 0, maximum: 1 }, 'the level');
  if (tests < 1) throw new RangeError('Use at least one test.');
  return 1 - (1 - alpha) ** tests;
}

/** A unanimity game: value one only when every listed player is present. */
export function unanimityGame(players) {
  checkInteger(players, 'the number of players');
  checkRange(players, limits.players, 'the number of players');
  const full = (1 << players) - 1;
  const values = Array.from({ length: 1 << players }, (_, mask) => (mask === full ? 1 : 0));
  return { players, values, phi: shapleyValues(values) };
}

/** The same unanimity requirement with the players regrouped. Grouping changes
 * the possible arrival orders, so it is not the same as summing afterwards. */
export function groupedUnanimity(groupSizes) {
  if (!Array.isArray(groupSizes) || groupSizes.length < 2) throw new RangeError('Use at least two grouped players.');
  groupSizes.forEach((size, index) => {
    checkInteger(size, `group ${index + 1}`);
    if (size < 1) throw new RangeError('Every group needs at least one member.');
  });
  const members = groupSizes.reduce((sum, size) => sum + size, 0);
  const individual = unanimityGame(members);
  const grouped = unanimityGame(groupSizes.length);
  let offset = 0;
  const summed = groupSizes.map(size => {
    const value = individual.phi.slice(offset, offset + size).reduce((sum, entry) => sum + entry, 0);
    offset += size;
    return value;
  });
  return {
    groupSizes, members,
    individual: individual.phi,
    summedIndividual: summed,
    grouped: grouped.phi,
    // The two agree only when every group has the same size.
    agrees: summed.every((value, index) => Math.abs(value - grouped.phi[index]) <= 1e-12),
  };
}

/* ------------------------------------------------------- drawing arithmetic */

/** One shared scale for a panel of signed values. Two panels that measure
 * different things must call this separately and say so: an accuracy decrease
 * and a normalized impurity decrease never share an axis here. */
export function barScale(values, { minimumExtent = 1e-12 } = {}) {
  const magnitudes = values.map((value, index) => Math.abs(checkFinite(value, `value ${index + 1}`)));
  const extent = Math.max(...magnitudes, minimumExtent);
  return {
    extent,
    shares: values.map(value => value / extent),
    allZero: magnitudes.every(value => value === 0),
  };
}

/** Points for the three actually evaluated retained sizes. Nothing is
 * interpolated between them; the gaps are returned so a drawing can show that
 * the tested sizes are not equally spaced. */
export function sizePoints(candidates) {
  if (!Array.isArray(candidates) || candidates.length === 0) throw new RangeError('Give the evaluated sizes.');
  const sizes = candidates.map(entry => checkInteger(entry.k, 'a retained size'));
  const scores = candidates.map(entry => checkFinite(entry.meanAccuracy, 'a mean accuracy'));
  const best = scores.reduce((leader, value, index) => (value > scores[leader] ? index : leader), 0);
  return {
    sizes, scores,
    lowSize: Math.min(...sizes), highSize: Math.max(...sizes),
    lowScore: Math.min(...scores), highScore: Math.max(...scores),
    chosen: sizes[best], chosenIndex: best,
    evaluatedOnly: true,
  };
}

/** The per-fold retained masks as an explicit membership matrix. Row stability
 * and prediction quality are separate questions, and this is the first one. */
export function maskMatrix(folds, fieldCount) {
  return folds.map(fold => ({
    fold: fold.fold,
    selected: fold.selected,
    membership: Array.from({ length: fieldCount }, (_, column) => fold.selected.includes(column)),
  }));
}
