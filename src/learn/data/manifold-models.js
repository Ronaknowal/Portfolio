/** Bounded, dependency-free mathematical models for the manifold lesson.
 * Distances and IDs are model values; drawing offsets never enter these inputs.
 */
export const U_POINTS = [
  { id: 'A', x: 0, y: 0 }, { id: 'B', x: 0, y: 1 },
  { id: 'C', x: 0, y: 2 }, { id: 'D', x: 1, y: 2 },
  { id: 'E', x: 2, y: 2 }, { id: 'F', x: 2, y: 1 },
  { id: 'G', x: 2, y: 0 },
];
const finite = (value, name) => { if (!Number.isFinite(value)) throw new Error(`${name} must be finite.`); };
const compareId = (a, b) => typeof a === 'number' && typeof b === 'number' ? a - b : String(a).localeCompare(String(b));
const squaredDistance = (a, b) => a.reduce((total, value, index) => total + (value - b[index]) ** 2, 0);

export function radiusGraph(points, radius, startId = 'A', endId = 'G') {
  finite(radius, 'Radius');
  if (radius <= 0 || !Array.isArray(points) || points.length < 2 || points.length > 7) throw new Error('Use 2–7 points and a positive radius.');
  const ids = points.map(point => point.id);
  if (new Set(ids).size !== points.length || !ids.includes(startId) || !ids.includes(endId) || startId === endId) throw new Error('Use unique point IDs and two distinct endpoints.');
  points.forEach(point => { finite(point.x, 'Point x'); finite(point.y, 'Point y'); });
  const edges = [];
  const adjacency = new Map(ids.map(id => [id, []]));
  for (let i = 0; i < points.length; i += 1) for (let j = i + 1; j < points.length; j += 1) {
    const length = Math.hypot(points[i].x - points[j].x, points[i].y - points[j].y);
    if (length === 0) throw new Error('Duplicate coordinates need a different zero-distance convention; separate these points.');
    if (length <= radius + 1e-12) {
      const edge = { source: ids[i], target: ids[j], length };
      edges.push(edge);
      adjacency.get(ids[i]).push({ id: ids[j], length });
      adjacency.get(ids[j]).push({ id: ids[i], length });
    }
  }
  // Positive edges and seven vertices permit enumerating simple paths. Sorting
  // complete paths resolves equal-length alternatives lexicographically.
  let distance = Infinity;
  let path = [];
  function visit(id, route, length) {
    if (length > distance + 1e-12) return;
    if (id === endId) {
      if (length < distance - 1e-12 || (Math.abs(length - distance) <= 1e-12 && route.join('\0') < path.join('\0'))) { distance = length; path = route; }
      return;
    }
    for (const neighbor of adjacency.get(id).slice().sort((a, b) => compareId(a.id, b.id))) if (!route.includes(neighbor.id)) visit(neighbor.id, [...route, neighbor.id], length + neighbor.length);
  }
  visit(startId, [startId], 0);
  let cumulative = 0;
  const routeEdges = path.slice(1).map((target, index) => {
    const source = path[index];
    const length = adjacency.get(source).find(item => item.id === target).length;
    cumulative += length;
    return { source, target, length, cumulative };
  });
  const start = points.find(point => point.id === startId), end = points.find(point => point.id === endId);
  return { points: points.map(point => ({ ...point })), edges, path, distance: Number.isFinite(distance) ? distance : null, routeEdges, directDistance: Math.hypot(start.x - end.x, start.y - end.y) };
}

export function gaussianRow(distances, sigma) {
  finite(sigma, 'Bandwidth');
  if (sigma <= 0 || !Array.isArray(distances) || distances.length < 1 || distances.length > 300) throw new Error('Use a positive bandwidth and a bounded distance row.');
  distances.forEach(distance => { finite(distance, 'Distance'); if (distance < 0) throw new Error('Distances cannot be negative.'); });
  const squared = distances.map(distance => distance ** 2), minimum = Math.min(...squared);
  const weights = squared.map(value => Math.exp(-value / (2 * sigma ** 2)));
  const stableWeights = squared.map(value => Math.exp(-(value - minimum) / (2 * sigma ** 2)));
  const total = stableWeights.reduce((a, b) => a + b, 0);
  const probabilities = stableWeights.map(value => value / total);
  const entropyBits = -probabilities.reduce((sum, value) => sum + (value > 0 ? value * Math.log2(value) : 0), 0);
  return { distances: [...distances], sigma, weights, probabilities, entropyBits, perplexity: 2 ** entropyBits };
}

export function directedMembership(distance, rho, sigma, retained = true) {
  [distance, rho, sigma].forEach(value => finite(value, 'Local input'));
  if (distance < 0 || rho < 0 || sigma <= 0) throw new Error('Distance/rho must be nonnegative and sigma positive.');
  return retained ? Math.exp(-Math.max(0, distance - rho) / sigma) : 0;
}
export function fuzzyUnion(forward, reverse) {
  [forward, reverse].forEach(value => { finite(value, 'Membership'); if (value < 0 || value > 1) throw new Error('Membership must be between 0 and 1.'); });
  return forward + reverse - forward * reverse;
}
export function umapConnection({ distance, rhoI, rhoJ, sigmaI, sigmaJ, retainedI = true, retainedJ = true }) {
  const forward = directedMembership(distance, rhoI, sigmaI, retainedI), reverse = directedMembership(distance, rhoJ, sigmaJ, retainedJ);
  return { forward, reverse, weight: fuzzyUnion(forward, reverse) };
}
export function idealPair(weight, separation, a = 1, b = 1) {
  [weight, separation, a, b].forEach(value => finite(value, 'Pair input'));
  if (weight < 0 || weight > 1 || separation < 0 || a <= 0 || b <= 0) throw new Error('Use weight 0–1, nonnegative separation and positive shape parameters.');
  const similarity = 1 / (1 + a * separation ** (2 * b));
  const cost = -(weight === 0 ? 0 : weight * Math.log(similarity)) - (weight === 1 ? 0 : (1 - weight) * Math.log1p(-similarity));
  const optimumSeparation = weight === 0 ? Infinity : ((1 / weight - 1) / a) ** (1 / (2 * b));
  return { similarity, cost, optimumSeparation };
}

/** Standard normalized symmetric t-SNE, no early exaggeration or optimizer. */
export function tsneObjective(P, Y) {
  const n = Y.length, dimensions = Y[0]?.length;
  if (n < 2 || n > 32 || !dimensions || dimensions > 3 || P.length !== n || Y.some(row => row.length !== dimensions || row.some(value => !Number.isFinite(value))) || P.some(row => row.length !== n)) throw new Error('Use a finite, bounded map and a matching probability matrix.');
  let mass = 0;
  P.forEach((row, i) => row.forEach((value, j) => { if (!Number.isFinite(value) || value < 0 || (i === j && value !== 0) || Math.abs(value - P[j][i]) > 1e-10) throw new Error('P must be symmetric, nonnegative and zero on its diagonal.'); mass += value; }));
  if (Math.abs(mass - 1) > 1e-8) throw new Error('P must sum to one over ordered pairs.');
  const kernel = Y.map((row, i) => Y.map((other, j) => i === j ? 0 : 1 / (1 + squaredDistance(row, other))));
  const normalization = kernel.flat().reduce((sum, value) => sum + value, 0);
  const Q = kernel.map(row => row.map(value => value / normalization));
  let cost = 0;
  const gradient = Y.map(() => Array(dimensions).fill(0));
  const pairContributions = Y.map((row, i) => Y.map((other, j) => row.map((value, axis) => {
    const contribution = -4 * (P[i][j] - Q[i][j]) * kernel[i][j] * (value - other[axis]);
    gradient[i][axis] -= contribution;
    return contribution;
  })));
  P.forEach((row, i) => row.forEach((value, j) => { if (value > 0) cost += value * Math.log(value / Q[i][j]); }));
  return { cost, gradient, Q, kernel, normalization, pairContributions };
}

export function neighborOrder(points, sourceIds = points.map((_, index) => index)) {
  if (!Array.isArray(points) || points.length < 2 || points.length > 300 || sourceIds.length !== points.length || new Set(sourceIds).size !== points.length) throw new Error('Use 2–300 observations with unique source IDs.');
  const dimensions = points[0].length;
  if (!dimensions || points.some(row => row.length !== dimensions || row.some(value => !Number.isFinite(value)))) throw new Error('All observations must have the same finite feature shape.');
  return points.map((point, i) => points.map((other, j) => ({ index: j, sourceRow: sourceIds[j], squared: squaredDistance(point, other) })).filter(item => item.index !== i).sort((a, b) => a.squared - b.squared || compareId(a.sourceRow, b.sourceRow)).map(({ index, sourceRow, squared }) => ({ index, sourceRow, distance: Math.sqrt(squared) })));
}
export function neighborAudit(inputPoints, mapPoints, k, sourceIds = inputPoints.map((_, index) => index)) {
  if (inputPoints.length !== mapPoints.length || !Number.isInteger(k) || k < 1 || k >= inputPoints.length / 2) throw new Error('Use matching observations and integer 1 ≤ k < n/2.');
  const inputOrder = neighborOrder(inputPoints, sourceIds), mapOrder = neighborOrder(mapPoints, sourceIds);
  let retainedTotal = 0, falsePenalty = 0, missingPenalty = 0;
  const local = inputOrder.map((row, i) => {
    const input = row.slice(0, k), map = mapOrder[i].slice(0, k);
    const retained = input.filter(item => map.some(other => other.sourceRow === item.sourceRow)).map(item => item.sourceRow);
    const missing = input.filter(item => !retained.includes(item.sourceRow)).map(item => item.sourceRow);
    const falseNeighbors = map.filter(item => !retained.includes(item.sourceRow)).map(item => item.sourceRow);
    retainedTotal += retained.length;
    falseNeighbors.forEach(id => { falsePenalty += row.findIndex(item => item.sourceRow === id) + 1 - k; });
    missing.forEach(id => { missingPenalty += mapOrder[i].findIndex(item => item.sourceRow === id) + 1 - k; });
    return { sourceRow: sourceIds[i], input, map, retained, missing, falseNeighbors, count: retained.length, retention: retained.length / k };
  });
  const factor = 2 / (inputPoints.length * k * (2 * inputPoints.length - 3 * k - 1));
  return { local, retention: retainedTotal / (inputPoints.length * k), trustworthiness: 1 - factor * falsePenalty, continuity: 1 - factor * missingPenalty, falsePenalty, missingPenalty, tieRule: 'ascending source-row ID' };
}
export function queryNeighbors(rows, coordinates, sourceRow, k) {
  const index = rows.findIndex(row => row.sourceRow === sourceRow);
  if (index < 0 || coordinates.length !== rows.length || !Number.isInteger(k) || k < 1 || k >= rows.length) throw new Error('Choose an existing source row and a valid neighbor count.');
  const ids = rows.map(row => row.sourceRow);
  // A query computes only two distance rows, not a full 300×300 audit.
  const rank = (points) => points.map((point, j) => ({ sourceRow: ids[j], distance: Math.sqrt(squaredDistance(points[index], point)), index: j })).filter(item => item.index !== index).sort((a, b) => a.distance - b.distance || a.sourceRow - b.sourceRow).slice(0, k);
  const input = rank(rows.map(row => row.pixels.map(value => value / 16))), map = rank(coordinates);
  const retained = input.filter(item => map.some(other => other.sourceRow === item.sourceRow)).map(item => item.sourceRow);
  return { input, map, retained, missing: input.filter(item => !retained.includes(item.sourceRow)).map(item => item.sourceRow), falseNeighbors: map.filter(item => !retained.includes(item.sourceRow)).map(item => item.sourceRow), count: retained.length, retention: retained.length / k };
}

/** Exact four-vertex Vietoris–Rips complex, including coincident identities. */
export function ripsComplex(points, epsilon) {
  finite(epsilon, 'Threshold');
  if (epsilon < 0 || points.length !== 4) throw new Error('This exact topology fixture has four points and a nonnegative threshold.');
  const edges = [], triangles = [], parent = [0, 1, 2, 3];
  const root = i => parent[i] === i ? i : (parent[i] = root(parent[i]));
  for (let i = 0; i < 4; i += 1) for (let j = i + 1; j < 4; j += 1) if (Math.sqrt(squaredDistance(points[i], points[j])) <= epsilon + 1e-12) { edges.push([i, j]); parent[root(i)] = root(j); }
  const has = (i, j) => edges.some(edge => edge[0] === i && edge[1] === j);
  for (let i = 0; i < 4; i += 1) for (let j = i + 1; j < 4; j += 1) for (let k = j + 1; k < 4; k += 1) if (has(i, j) && has(i, k) && has(j, k)) triangles.push([i, j, k]);
  const tetrahedra = edges.length === 6 ? [[0, 1, 2, 3]] : [];
  const beta0 = new Set(parent.map((_, i) => root(i))).size;
  // Up to four vertices, independent triangle boundaries have rank F−T.
  const beta1 = edges.length - 4 + beta0 - triangles.length + tetrahedra.length;
  return { edges, triangles, tetrahedra, beta0, beta1 };
}
