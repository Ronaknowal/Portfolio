/** Small, deterministic teaching models. Coordinates and probabilities are authored inputs. */
export const activeLearningFixtures = {
  hypotheses: Array.from({ length: 8 }, (_, index) => ({ threshold: index + 0.5, weight: 1 })),
  observations: [{ x: -1, label: 0 }, { x: 9, label: 1 }],
  queries: Array.from({ length: 9 }, (_, index) => index),
  opposing: [[0.95, 0.05], [0.05, 0.95]],
  shared: [[0.5, 0.5], [0.5, 0.5]],
  identical: [[0.95, 0.05], [0.95, 0.05]],
  anchors: [{ id: 'L1', x: 0, y: 0 }],
  candidates: [
    { id: 'A', x: 0, y: 1, probability: 0.5 },
    { id: 'B', x: 0.1, y: 1, probability: 0.52 },
    { id: 'C', x: 0, y: 4, probability: 0.7 },
    { id: 'D', x: 4, y: 0, probability: 0.72 },
  ],
};

export function assertFiniteRange(value, minimum, maximum, label) {
  if (!Number.isFinite(value) || value < minimum || value > maximum) {
    throw new Error(`${label} must be a finite number from ${minimum} to ${maximum}.`);
  }
}

export function entropy(probabilities) {
  if (!Array.isArray(probabilities) || probabilities.length < 2 || probabilities.length > 4) {
    throw new Error('Supply probabilities for two to four classes.');
  }
  probabilities.forEach(value => assertFiniteRange(value, 0, 1, 'Probability'));
  if (Math.abs(probabilities.reduce((sum, value) => sum + value, 0) - 1) > 1e-9) {
    throw new Error('Each probability row must sum to 1; use Normalize row if intended.');
  }
  return -probabilities.reduce((sum, value) => sum + (value === 0 ? 0 : value * Math.log(value)), 0);
}

export function uncertaintyScores(probabilities) {
  const entropyValue = entropy(probabilities);
  const sorted = [...probabilities].sort((left, right) => right - left);
  return { leastConfidence: 1 - sorted[0], margin: sorted[0] - sorted[1], entropy: entropyValue };
}

export function committeeDecomposition(rows) {
  if (rows.length < 2 || rows.length > 6 || rows.some(row => row.length !== rows[0].length)) {
    throw new Error('Use two to six members with the same number of classes.');
  }
  const memberEntropies = rows.map(entropy);
  const mean = rows[0].map((_, column) => rows.reduce((sum, row) => sum + row[column], 0) / rows.length);
  const predictiveEntropy = entropy(mean);
  const meanMemberEntropy = memberEntropies.reduce((sum, value) => sum + value, 0) / rows.length;
  const disagreement = Math.max(0, predictiveEntropy - meanMemberEntropy);
  const votes = mean.map(() => 0);
  rows.forEach(row => { votes[row.indexOf(Math.max(...row))] += 1; });
  return { mean, memberEntropies, predictiveEntropy, meanMemberEntropy, disagreement,
    votes, voteEntropy: entropy(votes.map(count => count / rows.length)) };
}

export function compareNumbers(value, reference, tolerance = 1e-10) {
  return Math.abs(value - reference) <= tolerance ? 'same' : value > reference ? 'larger' : 'smaller';
}

export function parseNumericRows(text, columns, label) {
  const lines = text.trim().split(/\n/).filter(line => line.trim());
  if (!lines.length) throw new Error(`${label} needs at least one row.`);
  return lines.map((line, index) => {
    const cells = line.trim().split(/[\s,]+/);
    if (cells.length !== columns || cells.some(cell => !Number.isFinite(Number(cell)))) {
      throw new Error(`${label} row ${index + 1} needs ${columns} finite numbers.`);
    }
    return cells.map(Number);
  });
}

export function validateThresholdInputs(hypotheses, observations, queries) {
  if (hypotheses.length < 2 || hypotheses.length > 16) throw new Error('Use 2–16 candidate thresholds.');
  if (new Set(hypotheses.map(row => row.threshold)).size !== hypotheses.length) throw new Error('Candidate thresholds must be distinct.');
  hypotheses.forEach(row => {
    assertFiniteRange(row.threshold, -10, 10, 'Threshold');
    if (!Number.isFinite(row.weight) || row.weight <= 0) throw new Error('Hypothesis weights must be positive and finite.');
  });
  if (observations.length > 24) throw new Error('Use at most 24 seed observations.');
  observations.forEach(row => {
    assertFiniteRange(row.x, -12, 12, 'Observation input');
    if (row.label !== 0 && row.label !== 1) throw new Error('An observed label must be 0 or 1.');
  });
  if (!queries.length || queries.length > 24 || new Set(queries).size !== queries.length) throw new Error('Use 1–24 distinct query values.');
  queries.forEach(value => assertFiniteRange(value, -12, 12, 'Query input'));
}

export function thresholdState(hypotheses, observations) {
  return hypotheses.filter(hypothesis => observations.every(row => Number(row.x >= hypothesis.threshold) === row.label));
}

export function thresholdQuestion(hypotheses, observations, query) {
  const remaining = thresholdState(hypotheses, observations);
  if (!remaining.length) throw new Error('No listed threshold is consistent; inspect the observations or reset.');
  const groups = [0, 1].map(label => remaining.filter(row => Number(query >= row.threshold) === label));
  const largestWeight = Math.max(...remaining.map(row => row.weight));
  const scaledWeight = remaining.reduce((sum, row) => sum + row.weight / largestWeight, 0);
  const probabilities = groups.map(rows => rows.reduce((sum, row) => sum + (row.weight / largestWeight) / scaledWeight, 0));
  return { remaining, groups, counts: groups.map(rows => rows.length), probabilities,
    expectedCount: groups.reduce((sum, rows, index) => sum + probabilities[index] * rows.length, 0) };
}

export function validateGeometry(anchors, candidates, budget) {
  if (anchors.length < 1 || anchors.length > 4) throw new Error('Use one to four existing anchors.');
  if (candidates.length < 2 || candidates.length > 24) throw new Error('Use two to 24 candidate items.');
  const ids = [...anchors, ...candidates].map(point => point.id);
  if (new Set(ids).size !== ids.length) throw new Error('Every point needs a distinct identifier.');
  [...anchors, ...candidates].forEach(point => {
    assertFiniteRange(point.x, -10, 10, `${point.id} x`);
    assertFiniteRange(point.y, -10, 10, `${point.id} y`);
  });
  candidates.forEach(point => assertFiniteRange(point.probability, 0, 1, `${point.id} probability`));
  if (!Number.isInteger(budget) || budget < 1 || budget > Math.min(6, candidates.length)) throw new Error('Budget must be 1–6 and no larger than the candidate count.');
}

export function coveringDistances(anchors, candidates, selectedIds = []) {
  const centers = [...anchors, ...candidates.filter(point => selectedIds.includes(point.id))];
  if (!centers.length) throw new Error('At least one center is required.');
  const assignments = candidates.map(point => {
    const nearest = centers.map(center => ({ center, distance: Math.hypot(point.x - center.x, point.y - center.y) }))
      .reduce((best, candidate) => candidate.distance < best.distance ? candidate : best);
    return { id: point.id, point, ...nearest };
  });
  return { assignments, radius: Math.max(...assignments.map(row => row.distance)) };
}

export function farthestFirst(anchors, candidates, budget) {
  validateGeometry(anchors, candidates, budget);
  const selected = [];
  const steps = [];
  for (let index = 0; index < budget; index += 1) {
    const before = coveringDistances(anchors, candidates, selected);
    const eligible = before.assignments.filter(row => !selected.includes(row.id));
    eligible.sort((left, right) => right.distance - left.distance || left.id.localeCompare(right.id, 'en'));
    const choice = eligible[0];
    selected.push(choice.id);
    steps.push({ id: choice.id, distanceBefore: choice.distance, selected: [...selected], ...coveringDistances(anchors, candidates, selected) });
  }
  return { selected, steps, ...coveringDistances(anchors, candidates, selected) };
}

export function entropyBatch(candidates, budget) {
  return candidates.map(point => ({ id: point.id, score: entropy([1 - point.probability, point.probability]) }))
    .sort((left, right) => Math.abs(right.score - left.score) <= 1e-12
      ? left.id.localeCompare(right.id, 'en') : right.score - left.score)
    .slice(0, budget).map(point => point.id);
}
