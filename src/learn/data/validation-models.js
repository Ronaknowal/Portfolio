/** Pure models for the cross-validation and hyperparameter-tuning lesson.
 *
 * The entities here are *fold plans*: which source row IDs sit in which fold,
 * under which grouping and which stratification, and what each fold's fitted
 * copy was allowed to see. Every function below computes its answer from the
 * arguments it is given. Nothing is interpolated, looked up or rounded into
 * place, and an input that does not describe a valid plan raises instead of
 * being quietly repaired.
 *
 * One vocabulary rule runs through the whole file. A score that *chose* a
 * setting is a selection score; a score produced by rows that were protected
 * from that choice is an assessment score. The two never share a field name.
 */

/* ------------------------------------------------------------------ *
 * Input guards
 * ------------------------------------------------------------------ */

const finite = (value, name) => {
  if (typeof value !== 'number' || !Number.isFinite(value)) {
    throw new RangeError(`${name} must be a finite number.`);
  }
  return value;
};

const wholeNumber = (value, name) => {
  finite(value, name);
  if (!Number.isInteger(value)) throw new RangeError(`${name} must be a whole number.`);
  return value;
};

const binaryLabel = (value, name) => {
  wholeNumber(value, name);
  if (value !== 0 && value !== 1) throw new RangeError(`${name} must be 0 or 1.`);
  return value;
};

const probability = (value, name) => {
  finite(value, name);
  if (value < 0 || value > 1) throw new RangeError(`${name} must lie between 0 and 1.`);
  return value;
};

/* ------------------------------------------------------------------ *
 * Fold plans
 * ------------------------------------------------------------------ */

/** `np.array_split(range(n), k)`: the first n mod k folds take one extra row,
 * so no remainder is dropped and every row keeps a validation turn. */
export function consecutiveFolds(n, k) {
  wholeNumber(n, 'the row count');
  wholeNumber(k, 'the fold count');
  if (!(k >= 2 && k <= n)) throw new RangeError('Require 2 <= k <= n.');
  const base = Math.floor(n / k);
  const extra = n % k;
  const folds = [];
  let cursor = 0;
  for (let index = 0; index < k; index += 1) {
    const size = base + (index < extra ? 1 : 0);
    folds.push(Array.from({ length: size }, (_, offset) => cursor + offset));
    cursor += size;
  }
  return folds;
}

/** Turn an explicit assignment into an inspectable plan.
 *
 * `rows` are the source row IDs in their original order; `folds` lists the
 * held-out IDs of each fold. The returned object answers, for every fold, which
 * IDs it assesses, which IDs its fitted copy may use, whether any group spans
 * that boundary and how the declared classes fell.
 *
 * `requirePartition` is the `cross_val_predict` contract: each row held out
 * exactly once. A forward-time plan legitimately fails it, so the flag can be
 * lowered and the coverage reported instead of the plan being rejected.
 */
export function foldPlan({
  rows, folds, groupOf = null, labelOf = null, requirePartition = true, name = 'fold plan',
}) {
  if (!Array.isArray(rows) || rows.length === 0) throw new RangeError('A plan needs at least one row.');
  if (new Set(rows).size !== rows.length) throw new RangeError('Row IDs must be distinct.');
  if (!Array.isArray(folds) || folds.length < 1) throw new RangeError('A plan needs at least one fold.');
  const known = new Set(rows);
  const heldOutCount = new Map(rows.map(id => [id, 0]));
  folds.forEach((fold, index) => {
    if (!Array.isArray(fold) || fold.length === 0) throw new RangeError(`Fold ${index + 1} is empty; every fold must assess at least one row.`);
    if (new Set(fold).size !== fold.length) throw new RangeError(`Fold ${index + 1} lists a row twice.`);
    fold.forEach(id => {
      if (!known.has(id)) throw new RangeError(`Fold ${index + 1} names row ${id}, which is not in this dataset.`);
      heldOutCount.set(id, heldOutCount.get(id) + 1);
    });
  });
  const unassessed = rows.filter(id => heldOutCount.get(id) === 0);
  const repeated = rows.filter(id => heldOutCount.get(id) > 1);
  if (requirePartition && (unassessed.length || repeated.length)) {
    throw new RangeError(
      `Every row must be held out exactly once. ${unassessed.length} row(s) are never assessed and ${repeated.length} are assessed more than once.`,
    );
  }

  const detail = folds.map((validation, index) => {
    const held = new Set(validation);
    const training = rows.filter(id => !held.has(id));
    return {
      index,
      label: `fit ${index + 1}`,
      validation: [...validation],
      training,
      assessed: validation.length,
      trainingSize: training.length,
    };
  });

  const grouping = { declared: Boolean(groupOf), groups: [], spanning: [], clean: true };
  if (groupOf) {
    grouping.groups = [...new Set(rows.map(id => groupOf(id)))];
    detail.forEach(fold => {
      const heldGroups = new Set(fold.validation.map(id => groupOf(id)));
      const trainGroups = new Set(fold.training.map(id => groupOf(id)));
      [...heldGroups].filter(group => trainGroups.has(group)).forEach(group => {
        grouping.spanning.push({ fold: fold.index, group });
      });
    });
    grouping.clean = grouping.spanning.length === 0;
  }

  const stratification = { declared: Boolean(labelOf), classes: [], perFold: [], missingClass: [] };
  if (labelOf) {
    stratification.classes = [...new Set(rows.map(id => labelOf(id)))].sort();
    const overall = Object.fromEntries(stratification.classes.map(cls => [cls, rows.filter(id => labelOf(id) === cls).length]));
    stratification.overall = overall;
    detail.forEach(fold => {
      const counts = Object.fromEntries(stratification.classes.map(cls => [cls, fold.validation.filter(id => labelOf(id) === cls).length]));
      stratification.perFold.push({ fold: fold.index, counts, share: Object.fromEntries(stratification.classes.map(cls => [cls, counts[cls] / fold.assessed])) });
      stratification.classes.filter(cls => counts[cls] === 0).forEach(cls => stratification.missingClass.push({ fold: fold.index, class: cls }));
    });
  }

  return {
    name,
    rows: [...rows],
    n: rows.length,
    k: folds.length,
    folds: detail,
    membership: Object.fromEntries(rows.map(id => [id, folds.findIndex(fold => fold.includes(id))])),
    coverage: { unassessed, repeated, partition: unassessed.length === 0 && repeated.length === 0 },
    grouping,
    stratification,
  };
}

/** The `cross_val_predict` table a plan can and cannot produce. Rows the plan
 * never holds out stay explicitly absent; they are never filled with an
 * in-sample output. */
export function outOfFoldCoverage(plan) {
  return plan.rows.map(id => {
    const holders = plan.folds.filter(fold => fold.validation.includes(id)).map(fold => fold.index);
    return {
      row: id,
      predictedBy: holders,
      status: holders.length === 0 ? 'no held-out prediction' : holders.length === 1 ? `fit ${holders[0] + 1}` : `assessed ${holders.length} times`,
      usable: holders.length === 1,
    };
  });
}

/* ------------------------------------------------------------------ *
 * One-nearest-neighbour cross-validation on a one-dimensional dataset
 * ------------------------------------------------------------------ */

/** Run a fresh 1-NN fit for every fold of a plan.
 *
 * Distances are absolute differences. Equal distances resolve to the smallest
 * eligible *source row ID*, which is what makes the trace reproducible when the
 * learner reorders or shuffles anything. A held-out row's own label is never
 * available to the fit that predicts it: `training` already excludes it.
 */
export function nearestNeighborRun(points, plan) {
  if (!Array.isArray(points) || points.length === 0) throw new RangeError('Provide at least one point.');
  const byId = new Map();
  points.forEach(point => {
    wholeNumber(point.id, 'a row ID');
    finite(point.x, `feature x of row ${point.id}`);
    binaryLabel(point.y, `label y of row ${point.id}`);
    if (byId.has(point.id)) throw new RangeError(`Row ${point.id} appears twice.`);
    byId.set(point.id, point);
  });
  plan.rows.forEach(id => { if (!byId.has(id)) throw new RangeError(`The plan names row ${id}, which has no measurement.`); });

  const folds = plan.folds.map(fold => {
    if (fold.training.length === 0) throw new RangeError(`Fit ${fold.index + 1} has no training rows; a 1-NN model cannot be fitted.`);
    const rows = fold.validation.map(id => {
      const point = byId.get(id);
      const contributions = fold.training.map(trainId => ({
        row: trainId,
        x: byId.get(trainId).x,
        label: byId.get(trainId).y,
        distance: Math.abs(point.x - byId.get(trainId).x),
      }));
      let winner = contributions[0];
      contributions.forEach(candidate => {
        if (candidate.distance < winner.distance - 1e-12) winner = candidate;
      });
      // Ties keep the smallest source ID; `training` is already in row order.
      const tied = contributions.filter(candidate => Math.abs(candidate.distance - winner.distance) <= 1e-12);
      winner = tied[0];
      return {
        row: id, x: point.x, truth: point.y,
        neighbor: winner.row, neighborX: winner.x, distance: winner.distance,
        prediction: winner.label,
        correct: winner.label === point.y,
        tied: tied.length > 1 ? tied.map(candidate => candidate.row) : [],
        contributions,
      };
    });
    const correct = rows.filter(row => row.correct).length;
    return {
      index: fold.index, training: fold.training, rows, correct, assessed: rows.length,
      accuracy: correct / rows.length,
    };
  });

  const totalCorrect = folds.reduce((sum, fold) => sum + fold.correct, 0);
  const totalAssessed = folds.reduce((sum, fold) => sum + fold.assessed, 0);
  return {
    folds,
    totalCorrect,
    totalAssessed,
    /** Each fold carries weight 1/K. */
    foldMean: folds.reduce((sum, fold) => sum + fold.accuracy, 0) / folds.length,
    /** Each assessed row carries weight 1/n. */
    pooled: totalCorrect / totalAssessed,
    /** The identity in the lesson: the pooled score is the size-weighted fold mean. */
    weightedFoldMean: folds.reduce((sum, fold) => sum + (fold.assessed / totalAssessed) * fold.accuracy, 0),
  };
}

/* ------------------------------------------------------------------ *
 * Selection on a criterion that contains no signal
 * ------------------------------------------------------------------ */

const checkPatterns = (candidates, cases) => {
  if (!Array.isArray(candidates) || candidates.length < 1) throw new RangeError('Keep at least one candidate rule.');
  if (candidates.length > 16) throw new RangeError('This enumeration supports at most sixteen candidate rules.');
  candidates.forEach((candidate, index) => {
    if (!Array.isArray(candidate.pattern) || candidate.pattern.length !== cases) {
      throw new RangeError(`Candidate ${index + 1} must predict all ${cases} validation cases.`);
    }
    candidate.pattern.forEach((value, position) => binaryLabel(value, `candidate ${index + 1} prediction at case ${position + 1}`));
  });
};

/** Score every candidate against one set of validation labels and apply the
 * declared tie rule: the first candidate in the enumeration wins. */
export function scoreCandidates(labels, candidates) {
  if (!Array.isArray(labels) || labels.length === 0) throw new RangeError('Provide the validation labels.');
  labels.forEach((value, index) => binaryLabel(value, `validation label ${index + 1}`));
  checkPatterns(candidates, labels.length);
  const scored = candidates.map((candidate, index) => {
    const matches = candidate.pattern.map((value, position) => value === labels[position]);
    const correct = matches.filter(Boolean).length;
    return {
      index, id: candidate.id ?? `candidate ${index + 1}`, pattern: [...candidate.pattern],
      matches, correct, validationAccuracy: correct / labels.length,
    };
  });
  const best = Math.max(...scored.map(row => row.correct));
  const winner = scored.find(row => row.correct === best);
  const tied = scored.filter(row => row.correct === best);
  return {
    rows: scored,
    winner,
    tieBrokenBy: tied.length > 1 ? 'first candidate in the declared enumeration' : null,
    tiedWith: tied.map(row => row.id),
    /** The score that chose the rule. Never an assessment of it. */
    selectedValidationAccuracy: winner.validationAccuracy,
  };
}

const allPatterns = cases => {
  const total = 2 ** cases;
  return Array.from({ length: total }, (_, index) =>
    Array.from({ length: cases }, (_, position) => (index >> (cases - 1 - position)) & 1));
};

/** Expected accuracy of a *fixed* prediction pattern on fresh independent fair
 * labels, computed by enumerating them rather than asserted. It is 1/2 for
 * every pattern, which is the point. */
export function futureAccuracy(pattern) {
  const targets = allPatterns(pattern.length);
  const total = targets.reduce((sum, target) =>
    sum + target.filter((value, position) => value === pattern[position]).length / pattern.length, 0);
  return total / targets.length;
}

/** Average selected validation score over all 2^cases equally likely label
 * patterns, with each pattern's own winner. */
export function enumerateSelection(candidates, cases = 4) {
  checkPatterns(candidates, cases);
  const patterns = allPatterns(cases);
  const rows = patterns.map(labels => {
    const outcome = scoreCandidates(labels, candidates);
    return { labels, bestCorrect: outcome.winner.correct, winner: outcome.winner.id };
  });
  const byCount = new Map();
  rows.forEach(row => {
    const ones = row.labels.filter(Boolean).length;
    const entry = byCount.get(ones) ?? { ones, patterns: 0, bestCorrect: new Set() };
    entry.patterns += 1;
    entry.bestCorrect.add(row.bestCorrect);
    byCount.set(ones, entry);
  });
  return {
    cases,
    patternCount: patterns.length,
    probabilityPerPattern: 1 / patterns.length,
    rows,
    meanSelectedAccuracy: rows.reduce((sum, row) => sum + row.bestCorrect, 0) / (rows.length * cases),
    byOnesCount: [...byCount.values()].sort((a, b) => a.ones - b.ones)
      .map(entry => ({ ones: entry.ones, patterns: entry.patterns, bestCorrect: [...entry.bestCorrect].sort((a, b) => a - b) })),
    /** Every candidate is a fixed rule, so each has the same expected accuracy
     * on fresh fair labels however it was chosen. */
    futureAccuracy: futureAccuracy(candidates[0].pattern),
  };
}

/* ------------------------------------------------------------------ *
 * A nested experiment small enough to trace every fit
 * ------------------------------------------------------------------ */

/** k-nearest-neighbour majority vote with an odd k, over a training list held
 * in increasing source-ID order. Equal distances resolve by that order, which
 * is the stable sort the author's reference function performs. */
function majorityVote(points, training, queryX, k) {
  const ranked = training
    .map((id, position) => ({ id, position, distance: Math.abs(queryX - points[id].x) }))
    .sort((a, b) => Math.abs(a.distance - b.distance) <= 1e-12
      ? a.id - b.id : a.distance - b.distance);
  const chosen = ranked.slice(0, k);
  const share = chosen.reduce((sum, item) => sum + points[item.id].y, 0) / k;
  return { prediction: share > 0.5 ? 1 : 0, neighbors: chosen, share };
}

/** The constructed sixteen-row nested experiment: two outer folds by row-ID
 * parity, two inner folds by alternating list position, candidates k = 1 and
 * k = 3, smaller k on a tie.
 *
 * Every number the page shows for this fixture comes out of this function, so
 * an edited label recomputes real distances and votes.
 */
export function nestedTrace({ x, y, neighborCounts = [1, 3] }) {
  if (!Array.isArray(x) || !Array.isArray(y) || x.length !== y.length) {
    throw new RangeError('Provide one label for every feature value.');
  }
  if (x.length < 4) throw new RangeError('The nested fixture needs at least four rows.');
  const points = x.map((value, id) => ({ id, x: finite(value, `feature x of row ${id}`), y: binaryLabel(y[id], `label y of row ${id}`) }));
  neighborCounts.forEach(k => {
    wholeNumber(k, 'a neighbour count');
    if (k < 1 || k % 2 === 0) throw new RangeError('Neighbour counts must be odd and at least 1, so a binary vote cannot tie.');
  });
  const ids = points.map(point => point.id);

  return [0, 1].map(held => {
    const test = ids.filter(id => id % 2 === held);
    const train = ids.filter(id => id % 2 !== held);
    const inner = [
      { train: train.filter((_, position) => position % 2 === 1), validation: train.filter((_, position) => position % 2 === 0) },
      { train: train.filter((_, position) => position % 2 === 0), validation: train.filter((_, position) => position % 2 === 1) },
    ];
    inner.forEach(split => {
      neighborCounts.forEach(k => {
        if (k > split.train.length) throw new RangeError(`Neighbour count ${k} exceeds the ${split.train.length} rows an inner fit may use.`);
      });
    });
    const candidates = neighborCounts.map(k => {
      const scores = inner.map(split => {
        const rows = split.validation.map(id => {
          const vote = majorityVote(points, split.train, points[id].x, k);
          return { row: id, truth: points[id].y, ...vote, correct: vote.prediction === points[id].y };
        });
        return { rows, correct: rows.filter(row => row.correct).length, accuracy: rows.filter(row => row.correct).length / rows.length };
      });
      return { k, innerScores: scores.map(score => score.accuracy), innerDetail: scores, mean: scores.reduce((sum, score) => sum + score.accuracy, 0) / scores.length };
    });
    const bestMean = Math.max(...candidates.map(candidate => candidate.mean));
    const selected = candidates.find(candidate => candidate.mean === bestMean);
    const refit = test.map(id => {
      const vote = majorityVote(points, train, points[id].x, selected.k);
      return { row: id, truth: points[id].y, ...vote, correct: vote.prediction === points[id].y };
    });
    return {
      outerFold: held,
      test, train, inner,
      candidates,
      selectedK: selected.k,
      tie: candidates.filter(candidate => candidate.mean === bestMean).length > 1,
      /** The inner mean that chose k. Selection evidence, not assessment. */
      selectionScore: selected.mean,
      prediction: refit.map(row => row.prediction),
      refit,
      /** Produced by rows that were protected from the choice above. */
      assessedCorrect: refit.filter(row => row.correct).length,
      assessedCount: refit.length,
      assessedAccuracy: refit.filter(row => row.correct).length / refit.length,
    };
  });
}

/* ------------------------------------------------------------------ *
 * Search coverage, risk curves and acquisition
 * ------------------------------------------------------------------ */

/** Probability that T independent draws land at least once in a region holding
 * probability mass p under the declared sampling distribution. This is mass,
 * not score proximity. */
export function hitProbability(p, trials) {
  probability(p, 'the sampling mass p');
  wholeNumber(trials, 'the number of draws');
  if (trials < 0) throw new RangeError('The number of draws cannot be negative.');
  return 1 - (1 - p) ** trials;
}

/** Smallest whole number of draws reaching a target hit probability. */
export function drawsForHitProbability(p, target) {
  probability(p, 'the sampling mass p');
  probability(target, 'the target probability');
  if (p === 0) throw new RangeError('A region with no sampling mass is never drawn.');
  if (target >= 1) throw new RangeError('No finite number of draws reaches certainty.');
  return Math.ceil(Math.log(1 - target) / Math.log(1 - p));
}

/** Exact risks of the mean predictor under the lesson's independent sampling
 * model: it predicts the training mean of m observations with variance sigma
 * squared. Both are analytic, not measured learning curves. */
export function meanPredictorRisk(varianceSigmaSquared, m) {
  finite(varianceSigmaSquared, 'the variance');
  if (varianceSigmaSquared < 0) throw new RangeError('A variance cannot be negative.');
  wholeNumber(m, 'the training size');
  if (m < 1) throw new RangeError('A training set needs at least one observation.');
  return {
    m,
    expectedNewLoss: varianceSigmaSquared * (1 + 1 / m),
    expectedTrainingLoss: varianceSigmaSquared * (1 - 1 / m),
    irreducible: varianceSigmaSquared,
    optimism: (2 * varianceSigmaSquared) / m,
  };
}

/** Variance of a leave-one-out accuracy for a rule that ignores its training
 * rows. The losses are independent because nothing connects them, however much
 * the training sets overlap. */
export function fixedRuleAccuracyVariance(n) {
  wholeNumber(n, 'the observation count');
  if (n < 1) throw new RangeError('Provide at least one observation.');
  return 1 / (4 * n);
}

/** Variance of a fold mean under equal variances and one common correlation.
 * Those are assumptions that explain the formula, not measurements of overlap. */
export function foldMeanVariance({ tauSquared, k, rho }) {
  finite(tauSquared, 'the per-fold variance');
  if (tauSquared < 0) throw new RangeError('A variance cannot be negative.');
  wholeNumber(k, 'the fold count');
  if (k < 2) throw new RangeError('Use at least two folds.');
  finite(rho, 'the common correlation');
  if (rho < -1 / (k - 1) || rho > 1) throw new RangeError(`With ${k} folds an equal correlation must lie between ${(-1 / (k - 1)).toFixed(4)} and 1.`);
  return (tauSquared * (1 + (k - 1) * rho)) / k;
}

/** Expected improvement of a candidate whose believed outcomes are a discrete
 * distribution. Y is the surrogate's belief about a loss, never a target. */
export function expectedImprovement(incumbentLoss, outcomes) {
  finite(incumbentLoss, 'the incumbent loss');
  if (!Array.isArray(outcomes) || outcomes.length === 0) throw new RangeError('Describe at least one believed outcome.');
  const total = outcomes.reduce((sum, outcome) => sum + probability(outcome.probability, 'an outcome probability'), 0);
  if (Math.abs(total - 1) > 1e-9) throw new RangeError('The believed outcome probabilities must sum to 1.');
  const rows = outcomes.map(outcome => {
    const loss = finite(outcome.loss, 'a believed loss');
    const improvement = Math.max(incumbentLoss - loss, 0);
    return { ...outcome, loss, improvement, contribution: outcome.probability * improvement };
  });
  return {
    incumbentLoss,
    rows,
    meanLoss: rows.reduce((sum, row) => sum + row.probability * row.loss, 0),
    expectedImprovement: rows.reduce((sum, row) => sum + row.contribution, 0),
  };
}

/** The TPE acquisition value the original paper derives, as a function of the
 * density ratio. A high l/g ratio raises it; gamma is the mass given to the
 * better-loss group. */
export function tpeAcquisition({ gamma, betterDensity, otherDensity }) {
  probability(gamma, 'gamma');
  finite(betterDensity, 'the better-group density');
  finite(otherDensity, 'the other-group density');
  if (betterDensity <= 0) throw new RangeError('The better-group density must be positive.');
  if (otherDensity < 0) throw new RangeError('A density cannot be negative.');
  return 1 / (gamma + (1 - gamma) * (otherDensity / betterDensity));
}

/* ------------------------------------------------------------------ *
 * Successive halving
 * ------------------------------------------------------------------ */

/** Run one halving schedule and record exactly what it paid to observe.
 *
 * `losses[i][s]` is candidate i's loss at budget `budgets[s]`. Values after a
 * candidate's elimination are never read by the schedule; they are returned
 * separately as hindsight so a page can keep the two apart.
 */
export function successiveHalving({ candidates, budgets, factor = 3, firstStage = 0 }) {
  if (!Array.isArray(budgets) || budgets.length < 2) throw new RangeError('A schedule needs at least two budgets.');
  budgets.forEach((budget, index) => {
    finite(budget, 'a budget');
    if (budget <= 0) throw new RangeError('Every budget must be positive.');
    if (index > 0 && budget <= budgets[index - 1]) throw new RangeError('Budgets must increase.');
  });
  if (!Array.isArray(candidates) || candidates.length < 2) throw new RangeError('Compare at least two candidates.');
  if (candidates.length > 9) throw new RangeError('This schedule view supports at most nine candidates.');
  candidates.forEach((candidate, index) => {
    if (!Array.isArray(candidate.losses) || candidate.losses.length !== budgets.length) {
      throw new RangeError(`Candidate ${index + 1} needs one loss for each of the ${budgets.length} budgets.`);
    }
    candidate.losses.forEach(loss => {
      finite(loss, 'a loss');
      if (loss < 0) throw new RangeError('A loss cannot be negative.');
    });
  });
  wholeNumber(factor, 'the survival factor');
  if (factor < 2) throw new RangeError('The survival factor must be at least 2.');
  wholeNumber(firstStage, 'the first comparison stage');
  if (firstStage < 0 || firstStage >= budgets.length - 1) {
    throw new RangeError('The first comparison must happen at a budget that still leaves a later stage.');
  }

  let active = candidates.map((candidate, index) => index);
  const stages = [];
  const observedUpTo = candidates.map(() => -1);
  let nominalCost = 0;
  let resumableCost = 0;
  let previousBudget = 0;
  for (let stage = firstStage; stage < budgets.length; stage += 1) {
    const budget = budgets[stage];
    const observed = active.map(index => ({
      index, id: candidates[index].id ?? `candidate ${index + 1}`, loss: candidates[index].losses[stage],
    }));
    active.forEach(index => { observedUpTo[index] = stage; });
    nominalCost += active.length * budget;
    resumableCost += active.length * (budget - previousBudget);
    previousBudget = budget;
    const keep = Math.max(1, Math.ceil(active.length / factor));
    const ranked = [...observed].sort((a, b) => (a.loss - b.loss) || (a.index - b.index));
    const survivors = stage === budgets.length - 1 ? ranked.slice(0, active.length) : ranked.slice(0, keep);
    stages.push({
      stage, budget, assessed: active.length, observed, ranked,
      keep: stage === budgets.length - 1 ? active.length : keep,
      survivors: survivors.map(item => item.index),
      eliminated: ranked.slice(survivors.length).map(item => item.index),
      nominalStageCost: active.length * budget,
    });
    active = survivors.map(item => item.index);
  }

  const survivorIndex = active[0];
  const finalStage = budgets.length - 1;
  const hindsightBest = candidates
    .map((candidate, index) => ({ index, loss: candidate.losses[finalStage] }))
    .sort((a, b) => (a.loss - b.loss) || (a.index - b.index))[0];
  return {
    budgets, factor, firstStage,
    stages,
    survivor: { index: survivorIndex, id: candidates[survivorIndex].id ?? `candidate ${survivorIndex + 1}`, losses: candidates[survivorIndex].losses },
    /** Per candidate, the stage index up to which the schedule actually paid
     * to look. Later entries were never consulted. */
    observedUpTo,
    cost: {
      nominal: nominalCost,
      resumable: resumableCost,
      fullAllocation: candidates.length * budgets[finalStage],
    },
    hindsight: {
      bestIndex: hindsightBest.index,
      bestId: candidates[hindsightBest.index].id ?? `candidate ${hindsightBest.index + 1}`,
      bestLoss: hindsightBest.loss,
      survivorFinalLoss: candidates[survivorIndex].losses[finalStage],
      // The question asks about minimum loss, not the identity picked by a tie rule.
      matchesSurvivor: hindsightBest.loss === candidates[survivorIndex].losses[finalStage],
      regret: candidates[survivorIndex].losses[finalStage] - hindsightBest.loss,
    },
  };
}

/* ------------------------------------------------------------------ *
 * Fit accounting
 * ------------------------------------------------------------------ */

/** Count the pipeline fits a plan actually costs. Fits at different levels
 * train on different numbers of rows, so this is a count and not a time. */
export function fitBudget({ candidates, outerFolds, innerFolds, refit = true, finalSearch = true }) {
  [['candidates', candidates], ['outerFolds', outerFolds], ['innerFolds', innerFolds]].forEach(([name, value]) => {
    wholeNumber(value, `the number of ${name}`);
    if (value < 1) throw new RangeError(`${name} must be at least 1.`);
  });
  const perOuter = candidates * innerFolds + (refit ? 1 : 0);
  const assessment = outerFolds * perOuter;
  const final = finalSearch ? candidates * innerFolds + (refit ? 1 : 0) : 0;
  return { perOuter, assessment, final, total: assessment + final };
}

/** A bootstrap resample of n positions leaves a given row out with probability
 * (1 - 1/n)^n, so the expected share of distinct original rows present is the
 * complement. */
export function bootstrapDistinctShare(n) {
  wholeNumber(n, 'the row count');
  if (n < 1) throw new RangeError('Provide at least one row.');
  const absent = (1 - 1 / n) ** n;
  return { n, absentProbability: absent, distinctShare: 1 - absent, limit: 1 - Math.exp(-1) };
}
