// Bounded teaching models. Counts, ratings, model scores and probabilities have
// separate contracts; a null rating is missing, including when zero is valid.
export const recommenderRatings = [[5, 3, null, 1, null, 4, null], [4, null, 4, 1, 2, null, 3], [null, 3, null, null, 4, 3, null], [1, null, null, 5, 4, null, 2], [null, 1, 5, 4, null, null, 3]];
export const recommenderCounts = [[10, 3, 0, 1, 0, 15, 0], [8, 0, 5, 1, 2, 0, 4], [0, 6, 0, 0, 9, 12, 0], [1, 0, 0, 20, 7, 0, 3], [0, 1, 14, 8, 0, 0, 6]];
function finite(value, minimum, maximum, label) {
  if (!Number.isFinite(value) || value < minimum || value > maximum) {
    throw new RangeError(`${label} must be finite and in [${minimum}, ${maximum}].`);
  }
  return value;
}
function integer(value, minimum, maximum, label) {
  finite(value, minimum, maximum, label);
  if (!Number.isInteger(value)) throw new TypeError(`${label} must be an integer.`);
  return value;
}
function dense(values, minimum, maximum, label) {
  if (!Array.isArray(values) || values.length < minimum || values.length > maximum) {
    throw new TypeError(`${label} has an unsupported length.`);
  }
  for (let index = 0; index < values.length; index += 1) {
    if (!Object.hasOwn(values, index)) throw new TypeError(`${label} must have every own element.`);
  }
  return values;
}
function vector(values, length, label) {
  dense(values, length, length, label).forEach(value => finite(value, -1e4, 1e4, label));
  return values;
}
function dot(left, right) {
  return left.reduce((sum, value, index) => sum + value * right[index], 0);
}
export function summarizeRatings(matrix = recommenderRatings) {
  dense(matrix, 1, 20, 'Rating rows');
  const columns = dense(matrix[0], 1, 30, 'Rating columns').length;
  const observed = [];
  const userMeans = matrix.map((row, user) => {
    dense(row, columns, columns, 'Rating row');
    const values = [];
    row.forEach((rating, item) => {
      if (rating === null) return;
      finite(rating, -10, 10, 'Observed rating');
      observed.push({
        user,
        item,
        rating
      });
      values.push(rating);
    });
    return values.length ? values.reduce((sum, value) => sum + value, 0) / values.length : null;
  });
  return {
    users: matrix.length,
    items: columns,
    observed,
    userMeans,
    globalMean: observed.length ? observed.reduce((sum, entry) => sum + entry.rating, 0) / observed.length : null,
    missing: matrix.length * columns - observed.length
  };
}
export function feedbackCell({
  rating = null,
  count = 0,
  prediction = .5,
  alpha = 2,
  mode = 'explicit'
} = {}) {
  if (rating !== null) finite(rating, -10, 10, 'Rating');
  finite(count, 0, 100, 'Count');
  finite(prediction, -10, 10, 'Prediction');
  finite(alpha, 0, 40, 'Confidence multiplier');
  if (!['explicit', 'implicit'].includes(mode)) throw new RangeError('Choose an explicit or implicit objective.');
  const included = mode === 'implicit' || rating !== null;
  const target = mode === 'implicit' ? Number(count > 0) : rating;
  const weight = mode === 'implicit' ? 1 + alpha * count : included ? 1 : 0;
  return {
    included,
    target,
    weight,
    loss: included ? weight * (target - prediction) ** 2 : 0
  };
}
export function neighborhoodPrediction({
  matrix = recommenderRatings,
  user = 0,
  item = 2,
  centered = true,
  minimumOverlap = 2,
  shrinkage = 2,
  neighbors = 3,
  signed = false,
  prior = 3
} = {}) {
  const summary = summarizeRatings(matrix);
  integer(user, 0, summary.users - 1, 'User');
  integer(item, 0, summary.items - 1, 'Item');
  integer(minimumOverlap, 1, 20, 'Minimum overlap');
  integer(neighbors, 1, 30, 'Neighbor count');
  finite(shrinkage, 0, 30, 'Shrinkage');
  finite(prior, -10, 10, 'No-data prior');
  if (typeof centered !== 'boolean' || typeof signed !== 'boolean') throw new TypeError('Centering and signed-weight choices must be Boolean.');
  if (matrix[user][item] !== null) throw new RangeError('Choose an unobserved target for this prediction.');
  const fallback = summary.userMeans[user] ?? summary.globalMean ?? prior;
  const evidence = [];
  for (let neighbor = 0; neighbor < summary.items; neighbor += 1) {
    if (neighbor === item || matrix[user][neighbor] === null) continue;
    const pairs = [];
    matrix.forEach((row, otherUser) => {
      if (row[item] === null || row[neighbor] === null) return;
      const center = centered ? summary.userMeans[otherUser] : 0;
      pairs.push({
        user: otherUser,
        left: row[item] - center,
        right: row[neighbor] - center,
        center
      });
    });
    const numerator = pairs.reduce((sum, pair) => sum + pair.left * pair.right, 0);
    const leftNorm = Math.hypot(...pairs.map(pair => pair.left));
    const rightNorm = Math.hypot(...pairs.map(pair => pair.right));
    const rawSimilarity = leftNorm > 0 && rightNorm > 0 ? Math.max(-1, Math.min(1, numerator / (leftNorm * rightNorm))) : 0;
    const supported = pairs.length >= minimumOverlap && leftNorm > 0 && rightNorm > 0;
    const similarity = supported ? rawSimilarity * pairs.length / (pairs.length + shrinkage) : 0;
    const signal = matrix[user][neighbor] - (centered ? fallback : 0);
    const usable = supported && (signed ? similarity !== 0 : similarity > 0);
    evidence.push({
      item: neighbor,
      pairs,
      numerator,
      leftNorm,
      rightNorm,
      rawSimilarity,
      similarity,
      supported,
      usable,
      signal,
      contribution: similarity * signal
    });
  }
  const selected = evidence.filter(entry => entry.usable).sort((left, right) => Math.abs(right.similarity) - Math.abs(left.similarity) || left.item - right.item).slice(0, neighbors);
  const denominator = selected.reduce((sum, entry) => sum + Math.abs(entry.similarity), 0);
  const numerator = selected.reduce((sum, entry) => sum + entry.contribution, 0);
  const prediction = denominator ? (centered ? fallback : 0) + numerator / denominator : fallback;
  return {
    ...summary,
    user,
    item,
    centered,
    signed,
    evidence,
    selected,
    numerator,
    denominator,
    prediction,
    fallback,
    usedFallback: denominator === 0
  };
}
export function explicitFactorStep({
  userFactors = [.4, -.2],
  itemFactors = [.5, .3],
  userBias = .1,
  itemBias = -.1,
  mean = 3,
  rating = 5,
  rate = .05,
  penalty = .1
} = {}) {
  const dimension = dense(userFactors, 1, 5, 'User factors').length;
  vector(userFactors, dimension, 'User factors');
  vector(itemFactors, dimension, 'Item factors');
  finite(userBias, -1e4, 1e4, 'User bias');
  finite(itemBias, -1e4, 1e4, 'Item bias');
  finite(mean, -10, 10, 'Global mean');
  finite(rating, -10, 10, 'Rating');
  finite(rate, 0, 1, 'Learning rate');
  finite(penalty, 0, 10, 'Penalty');
  const prediction = mean + userBias + itemBias + dot(userFactors, itemFactors);
  const error = rating - prediction;
  const gradients = {
    userFactors: userFactors.map((value, index) => -error * itemFactors[index] + penalty * value),
    itemFactors: itemFactors.map((value, index) => -error * userFactors[index] + penalty * value),
    userBias: -error + penalty * userBias,
    itemBias: -error + penalty * itemBias
  };
  const after = {
    userFactors: userFactors.map((value, index) => value - rate * gradients.userFactors[index]),
    itemFactors: itemFactors.map((value, index) => value - rate * gradients.itemFactors[index]),
    userBias: userBias - rate * gradients.userBias,
    itemBias: itemBias - rate * gradients.itemBias
  };
  const localLoss = (userVector, itemVector, biasUser, biasItem) => .5 * ((rating - mean - biasUser - biasItem - dot(userVector, itemVector)) ** 2 + penalty * (dot(userVector, userVector) + dot(itemVector, itemVector) + biasUser ** 2 + biasItem ** 2));
  return {
    before: {
      userFactors: [...userFactors],
      itemFactors: [...itemFactors],
      userBias,
      itemBias
    },
    after,
    mean,
    rating,
    prediction,
    error,
    rate,
    penalty,
    gradients,
    loss: localLoss(userFactors, itemFactors, userBias, itemBias),
    nextPrediction: mean + after.userBias + after.itemBias + dot(after.userFactors, after.itemFactors),
    nextLoss: localLoss(after.userFactors, after.itemFactors, after.userBias, after.itemBias)
  };
}
export function factorUpdateTrace({
  rate = .05,
  penalty = .1,
  rating = 5,
  steps = 8
} = {}) {
  integer(steps, 0, 15, 'Step count');
  let state = {
    userFactors: [.4, -.2],
    itemFactors: [.5, .3],
    userBias: .1,
    itemBias: -.1
  };
  const trace = [];
  let stopped = false;
  for (let index = 0; index < steps; index += 1) {
    const result = explicitFactorStep({
      ...state,
      rate,
      penalty,
      rating
    });
    const proposed = [...result.after.userFactors, ...result.after.itemFactors, result.after.userBias, result.after.itemBias];
    if (proposed.some(value => !Number.isFinite(value) || Math.abs(value) > 1e4)) {
      stopped = true;
      break;
    }
    trace.push(result);
    state = result.after;
  }
  // Validate even when no step was requested, and give the actual final state.
  const final = explicitFactorStep({
    ...state,
    rate: 0,
    penalty,
    rating
  });
  finite(rate, 0, 1, 'Learning rate');
  return {
    trace,
    final,
    stopped,
    completedSteps: trace.length,
    requestedSteps: steps
  };
}
export function rotatedFactors(angle = 0) {
  finite(angle, -180, 180, 'Rotation angle');
  const radians = angle * Math.PI / 180;
  const rotate = ([x, y]) => [Math.cos(radians) * x - Math.sin(radians) * y, Math.sin(radians) * x + Math.cos(radians) * y];
  const originalUser = [1, .5];
  const originalItems = [[1, 0], [0, 1], [.5, 1]];
  const user = rotate(originalUser);
  const items = originalItems.map(rotate);
  return {
    user,
    items,
    originalUser,
    originalItems,
    scores: items.map(item => dot(user, item)),
    originalScores: originalItems.map(item => dot(originalUser, item))
  };
}
export function implicitFactorBlock({
  counts = [2, 0, 1],
  alpha = 2,
  penalty = 1,
  includeMissing = true
} = {}) {
  dense(counts, 3, 3, 'Three item counts').forEach(count => finite(count, 0, 20, 'Count'));
  finite(alpha, 0, 40, 'Confidence multiplier');
  finite(penalty, .1, 10, 'Penalty');
  if (typeof includeMissing !== 'boolean') throw new TypeError('Missing-item inclusion must be Boolean.');
  const items = [[1, 0], [0, 1], [1, 1]];
  const targets = counts.map(count => Number(count > 0));
  const confidence = counts.map(count => count > 0 || includeMissing ? 1 + alpha * count : 0);
  const gram = [[2, 1], [1, 2]];
  const normal = [[penalty, 0], [0, penalty]];
  const right = [0, 0];
  items.forEach((item, index) => {
    for (let row = 0; row < 2; row += 1) {
      right[row] += confidence[index] * targets[index] * item[row];
      for (let column = 0; column < 2; column += 1) normal[row][column] += confidence[index] * item[row] * item[column];
    }
  });
  const l00 = Math.sqrt(normal[0][0]);
  const l10 = normal[1][0] / l00;
  const l11 = Math.sqrt(normal[1][1] - l10 * l10);
  const temporary = [right[0] / l00, (right[1] - l10 * right[0] / l00) / l11];
  const user = [(temporary[0] - l10 * temporary[1] / l11) / l00, temporary[1] / l11];
  const objective = position => items.reduce((sum, item, index) => sum + confidence[index] * (targets[index] - dot(position, item)) ** 2, penalty * dot(position, position));
  const contours = [.1, .5, 1].map(excess => ({
    excess,
    points: Array.from({
      length: 65
    }, (_, index) => {
      const theta = 2 * Math.PI * index / 64;
      const y = Math.sqrt(excess) * Math.sin(theta) / l11;
      const x = (Math.sqrt(excess) * Math.cos(theta) - l10 * y) / l00;
      return [user[0] + x, user[1] + y];
    })
  }));
  return {
    items,
    targets,
    confidence,
    gram,
    normal,
    right,
    user,
    scores: items.map(item => dot(user, item)),
    objective: objective(user),
    contours,
    penalty,
    alpha,
    includeMissing
  };
}
function softplus(value) {
  return Math.max(0, value) + Math.log1p(Math.exp(-Math.abs(value)));
}
export function bprPairStep({
  user = [1, .5],
  positive = [.5, 1],
  negative = [1, 0],
  rate = .1,
  penalty = .1
} = {}) {
  vector(user, 2, 'User factors');
  vector(positive, 2, 'Positive item');
  vector(negative, 2, 'Sampled item');
  finite(rate, 0, 1, 'Rate');
  finite(penalty, 0, 10, 'Penalty');
  const positiveScore = dot(user, positive);
  const negativeScore = dot(user, negative);
  const gap = positiveScore - negativeScore;
  const gradientMagnitude = gap >= 0 ? Math.exp(-gap) / (1 + Math.exp(-gap)) : 1 / (1 + Math.exp(gap));
  const after = {
    user: user.map((value, index) => value + rate * (gradientMagnitude * (positive[index] - negative[index]) - penalty * value)),
    positive: positive.map((value, index) => value + rate * (gradientMagnitude * user[index] - penalty * value)),
    negative: negative.map((value, index) => value + rate * (-gradientMagnitude * user[index] - penalty * value))
  };
  const loss = softplus(-gap) + penalty / 2 * (dot(user, user) + dot(positive, positive) + dot(negative, negative));
  const nextGap = dot(after.user, after.positive) - dot(after.user, after.negative);
  const nextLoss = softplus(-nextGap) + penalty / 2 * (dot(after.user, after.user) + dot(after.positive, after.positive) + dot(after.negative, after.negative));
  return {
    gap,
    positiveScore,
    negativeScore,
    gradientMagnitude,
    loss,
    after,
    nextGap,
    nextLoss
  };
}
export function evaluateRecommendationList({
  order = [2, 0, 4, 1, 3],
  grades = [1, 1, 0, 0, 0],
  cutoff = 3,
  eligible = null
} = {}) {
  dense(grades, 1, 50, 'Relevance grades').forEach(grade => integer(grade, 0, 3, 'Grade'));
  integer(cutoff, 1, 50, 'Cutoff');
  const checkIds = (ids, label) => {
    dense(ids, 0, grades.length, label).forEach(id => integer(id, 0, grades.length - 1, 'Item ID'));
    if (new Set(ids).size !== ids.length) throw new RangeError(`${label} cannot repeat an item.`);
  };
  const eligibleItems = eligible === null ? grades.map((_, item) => item) : eligible;
  checkIds(eligibleItems, 'Eligible items');
  checkIds(order, 'Ranking');
  if (order.some(item => !eligibleItems.includes(item))) throw new RangeError('The ranking contains an ineligible item.');
  const relevantCount = eligibleItems.filter(item => grades[item] > 0).length;
  let hits = 0;
  let sumPrecision = 0;
  let reciprocalRank = 0;
  const contributions = order.slice(0, cutoff).map((item, index) => {
    const rank = index + 1;
    const relevant = grades[item] > 0;
    hits += Number(relevant);
    if (relevant) {
      sumPrecision += hits / rank;
      if (!reciprocalRank) reciprocalRank = 1 / rank;
    }
    const gain = 2 ** grades[item] - 1;
    const discount = 1 / Math.log2(rank + 1);
    return {
      item,
      rank,
      relevant,
      grade: grades[item],
      gain,
      discount,
      dcg: gain * discount,
      prefixPrecision: hits / rank
    };
  });
  const dcg = contributions.reduce((sum, entry) => sum + entry.dcg, 0);
  const ideal = eligibleItems.map(item => grades[item]).sort((left, right) => right - left).slice(0, cutoff).reduce((sum, grade, index) => sum + (2 ** grade - 1) / Math.log2(index + 2), 0);
  const candidateRelevant = order.filter(item => grades[item] > 0).length;
  return {
    contributions,
    hits,
    relevantCount,
    dcg,
    ideal,
    cutoff,
    fillRate: contributions.length / cutoff,
    precision: hits / cutoff,
    recall: relevantCount ? hits / relevantCount : null,
    ndcg: ideal ? dcg / ideal : null,
    reciprocalRank: relevantCount ? reciprocalRank : null,
    averagePrecision: relevantCount ? sumPrecision / Math.min(cutoff, relevantCount) : null,
    candidateCoverage: relevantCount ? candidateRelevant / relevantCount : null,
    bestCandidateRecall: relevantCount ? Math.min(cutoff, candidateRelevant) / relevantCount : null
  };
}
export function exposurePolicy({
  loggingA = .8,
  targetA = .5,
  qualityA = .4,
  qualityB = .8,
  requests = 100
} = {}) {
  for (const [name, value] of Object.entries({
    loggingA,
    targetA,
    qualityA,
    qualityB
  })) finite(value, 0, 1, name);
  integer(requests, 1, 10000, 'Independent requests');
  const logging = [loggingA, 1 - loggingA];
  const target = [targetA, 1 - targetA];
  const quality = [qualityA, qualityB];
  const supported = logging.every((probability, index) => probability > 0 || target[index] === 0);
  const leaves = [];
  for (let item = 0; item < 2; item += 1) {
    for (const reward of [0, 1]) {
      const probability = logging[item] * (reward ? quality[item] : 1 - quality[item]);
      const weight = logging[item] > 0 ? target[item] / logging[item] : null;
      if (weight !== null && !Number.isFinite(weight)) throw new RangeError('The importance weight exceeds this numerical model\'s finite range.');
      leaves.push({
        item,
        reward,
        probability,
        weight,
        weightedReward: weight === null ? null : weight * reward
      });
    }
  }
  const expectedLogged = dot(logging, quality);
  const knownToyTargetValue = dot(target, quality);
  const expectation = supported ? leaves.reduce((sum, leaf) => sum + (leaf.weightedReward === null ? 0 : leaf.probability * leaf.weightedReward), 0) : null;
  // Cancel one logging factor analytically before multiplying: squaring a
  // large weight first can overflow even when the second moment is finite.
  const secondMoment = supported ? logging.reduce((sum, probability, item) => sum + (probability > 0 ? target[item] * quality[item] * (target[item] / probability) : 0), 0) : null;
  if (supported && !Number.isFinite(secondMoment)) throw new RangeError('The second moment exceeds this numerical model\'s finite range.');
  const variance = supported ? Math.max(0, secondMoment - expectation ** 2) : null;
  return {
    logging,
    target,
    quality,
    leaves,
    supported,
    expectedLogged,
    knownToyTargetValue,
    expectation,
    variance,
    meanStandardError: supported ? Math.sqrt(variance / requests) : null,
    requests
  };
}
