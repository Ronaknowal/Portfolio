// Small deterministic teaching fixtures; these are not measured benchmarks.
export const neighborRows = [
  { id: 'A1', point: [1, 2], label: 'A' },
  { id: 'A2', point: [1.5, 1.8], label: 'A' },
  { id: 'A3', point: [1.2, 2.5], label: 'A' },
  { id: 'B1', point: [5, 5], label: 'B' },
  { id: 'B2', point: [5.5, 4.8], label: 'B' },
  { id: 'B3', point: [4.8, 5.2], label: 'B' },
  { id: 'C1', point: [3, 3], label: 'C' },
  { id: 'C2', point: [3.2, 2.8], label: 'C' },
];

function boundedVector(vector, size) {
  if (!Array.isArray(vector) || vector.length !== size || vector.some(value => !Number.isFinite(value) || Math.abs(value) > 10000)) {
    throw new RangeError('Provide the expected number of finite coordinates in [-10000,10000].');
  }
}

export function metricDistance(first, second, metric = 'euclidean') {
  boundedVector(first, 2);
  boundedVector(second, 2);
  const differences = first.map((value, index) => Math.abs(value - second[index]));
  if (metric === 'euclidean') return Math.hypot(...differences);
  if (metric === 'manhattan') return differences[0] + differences[1];
  if (metric === 'maximum') return Math.max(...differences);
  if (metric === 'cosine') {
    const norm = Math.hypot(...first) * Math.hypot(...second);
    if (!norm) throw new RangeError('Cosine comparison requires two nonzero vectors.');
    return Math.max(0, Math.min(2, 1 - (first[0] * second[0] + first[1] * second[1]) / norm));
  }
  throw new RangeError('Unsupported distance.');
}

export function neighborReport({ query = [3.1, 2.9], k = 5, metric = 'euclidean', weights = 'uniform', candidates = neighborRows } = {}) {
  boundedVector(query, 2);
  if (!Number.isInteger(k) || k < 1 || k > candidates.length || !['uniform', 'distance'].includes(weights) || candidates.length > 8 || candidates.some(row => !neighborRows.includes(row))) {
    throw new RangeError('Use a nonempty fixture subset, legal k and supported weights.');
  }
  const ranked = candidates.map(row => ({ ...row, distance: metricDistance(row.point, query, metric) }));
  // The decimal fixture uses distances rounded to 12 places only for stable tie ordering.
  ranked.sort((first, second) => Number(first.distance.toFixed(12)) - Number(second.distance.toFixed(12)) || first.id.localeCompare(second.id));
  const neighbors = ranked.slice(0, k);
  const exact = neighbors.some(row => row.distance === 0);
  const mass = row => weights === 'uniform' ? 1 : exact ? Number(row.distance === 0) : 1 / row.distance;
  const denominator = neighbors.reduce((total, row) => total + mass(row), 0);
  const contributions = neighbors.map(row => ({ ...row, weight: mass(row) / denominator }));
  const votes = ['A', 'B', 'C'].map(label => ({ label, probability: contributions.filter(row => row.label === label).reduce((total, row) => total + row.weight, 0) }));
  const winner = votes.reduce((best, row) => row.probability > best.probability + 1e-12 ? row : best, votes[0]);
  return { ranked, neighbors: contributions, votes, label: winner.label, radius: neighbors.at(-1).distance, exact };
}

export const unitRows = [
  { id: 'R', point: [1, 100], label: 'repair' },
  { id: 'S', point: [3, 1000], label: 'service' },
  { id: 'T', point: [5, 1100], label: 'service' },
];
export function unitsReport(standardize = false) {
  if (typeof standardize !== 'boolean') throw new RangeError('Select raw or standardized coordinates.');
  const query = [1.2, 600];
  const means = [0, 1].map(column => unitRows.reduce((sum, row) => sum + row.point[column], 0) / unitRows.length);
  const scales = [0, 1].map(column => Math.sqrt(unitRows.reduce((sum, row) => sum + (row.point[column] - means[column]) ** 2, 0) / unitRows.length));
  const rows = unitRows.map(row => {
    const contributions = row.point.map((value, column) => ((value - query[column]) / (standardize ? scales[column] : 1)) ** 2);
    return { ...row, contributions, distance: Math.sqrt(contributions[0] + contributions[1]) };
  });
  rows.sort((first, second) => first.distance - second.distance);
  return { query, scales, rows, nearest: rows[0] };
}

export const regressionRows = [0, 1, 2, 3, 4, 5].map(value => ({ id: value, x: value, y: value ** 2 }));
export function localMean(query = 2.5, k = 3, weights = 'uniform') {
  if (!Number.isFinite(query) || query < -1 || query > 8 || !Number.isInteger(k) || k < 1 || k > 6 || !['uniform', 'distance'].includes(weights)) {
    throw new RangeError('Use query [-1,8], k1–6 and supported weights.');
  }
  const selected = regressionRows.map(row => ({ ...row, distance: Math.abs(row.x - query) })).sort((first, second) => first.distance - second.distance || first.id - second.id).slice(0, k);
  const hasExact = selected.some(row => row.distance === 0);
  const minimumDistance = selected[0].distance;
  const mass = row => weights === 'uniform' ? 1 : hasExact ? Number(row.distance === 0) : minimumDistance / row.distance;
  const total = selected.reduce((sum, row) => sum + mass(row), 0);
  const rows = selected.map(row => ({ ...row, weight: mass(row) / total }));
  return { rows, prediction: rows.reduce((sum, row) => sum + row.weight * row.y, 0) };
}

export function buildKdTree(rows = neighborRows, depth = 0) {
  if (!rows.length) return null;
  const axis = depth % 2;
  const sorted = [...rows].sort((first, second) => first.point[axis] - second.point[axis] || first.id.localeCompare(second.id));
  const middle = Math.floor(sorted.length / 2);
  return { row: sorted[middle], axis, left: buildKdTree(sorted.slice(0, middle), depth + 1), right: buildKdTree(sorted.slice(middle + 1), depth + 1) };
}

export function kdSearch(query = [1.3, 2.1]) {
  boundedVector(query, 2);
  const events = [];
  let best = null;
  function visit(node) {
    if (!node) return;
    const distance = metricDistance(query, node.row.point);
    if (!best || distance < best.distance || (distance === best.distance && node.row.id < best.id)) best = { ...node.row, distance };
    events.push({ kind: 'visit', id: node.row.id, axis: node.axis, split: node.row.point[node.axis], best: { ...best }, explanation: `Visit ${node.row.id}; best so far ${best.id}.` });
    const difference = query[node.axis] - node.row.point[node.axis];
    const near = difference <= 0 ? node.left : node.right;
    const far = difference <= 0 ? node.right : node.left;
    visit(near);
    if (far) {
      const lowerBound = Math.abs(difference);
      const search = lowerBound <= best.distance;
      events.push({ kind: search ? 'search-far' : 'prune', id: node.row.id, axis: node.axis, split: node.row.point[node.axis], lowerBound, best: { ...best }, explanation: `${search ? 'Search' : 'Prune'} the other side of ${node.row.id}: plane distance ${lowerBound.toFixed(3)} ${search ? '≤' : '>'} current best ${best.distance.toFixed(3)}.` });
      if (search) visit(far);
    }
  }
  visit(buildKdTree());
  return { events, best, visits: events.filter(event => event.kind === 'visit').length };
}

export function candidateReport(mode = 'all') {
  const subsets = { all: neighborRows, 'lose-close-c': neighborRows.filter(row => row.label !== 'C'), 'keep-close-c': neighborRows.filter(row => row.label !== 'B') };
  if (!subsets[mode]) throw new RangeError('Select a declared candidate pool.');
  const exact = neighborReport({ k: 3, weights: 'distance' });
  const candidate = neighborReport({ k: 3, weights: 'distance', candidates: subsets[mode] });
  const matched = candidate.neighbors.filter(row => exact.neighbors.some(other => other.id === row.id)).length;
  return { exact, candidate, candidates: subsets[mode], recall: matched / 3 };
}

export function volumeReport(dimension = 10, fraction = 0.01) {
  if (!Number.isInteger(dimension) || dimension < 1 || dimension > 100 || !Number.isFinite(fraction) || fraction <= 0 || fraction > 1) throw new RangeError('Use dimension1–100 and a positive volume fraction at most1.');
  return { side: fraction ** (1 / dimension), log10RowsForTenNeighborsAtSideOneTenth: 1 + dimension };
}
