/** Pure teaching models for the DBSCAN lesson. Neighbourhoods are closed
 * (distance ≤ ε) and count the row itself; types are core, border or noise;
 * clusters are connected components of the core graph plus attached borders.
 * All fixtures use exactly representable coordinates so promised ties are real.
 * These are bounded teaching models, not a numerical library.
 */
import { irisRows, irisSpecies } from './dbscan-iris-data.js';

export const trailNames = 'ABCDEFGHIJ';
export const trailPoints = Object.freeze([-1.75, -1.5, -1.25, -1, 1, 1.25, 1.5, 1.75, 0, 4].map(x => Object.freeze([x, 0])));
export const variedGroups = Object.freeze({ left: [0, 0.125, 0.25, 0.375], middle: [0.75, 0.875, 1, 1.125], right: [5, 5.75, 6.5, 7.25], rightNull: [5, 5.125, 5.25, 5.375] });
export const cornerPoints = Object.freeze([[0, 0], [1, 0], [0, 2], [1, 2]].map(point => Object.freeze(point)));
export const fifthCorner = Object.freeze([3, 0]);
export const pointLimit = 200;

function checkPoints(points, bound = 1e6) {
  if (!Array.isArray(points) || points.length < 1 || points.length > pointLimit) throw new RangeError(`Use 1 to ${pointLimit} rows.`);
  const dimension = points[0].length;
  for (const point of points) {
    if (!Array.isArray(point) || point.length !== dimension || point.some(value => !Number.isFinite(value) || Math.abs(value) > bound)) {
      throw new RangeError('Every row needs the same number of finite coordinates.');
    }
  }
  return dimension;
}
export function distance(left, right) {
  return Math.hypot(...left.map((value, axis) => value - right[axis]));
}
/** Symmetric distance matrix. */
export function distanceMatrix(points) {
  checkPoints(points);
  return points.map(left => points.map(right => distance(left, right)));
}
/** DBSCAN with self-inclusive closed neighbourhoods. `order` lists the row
 * indices in visiting order; the result records types, cluster labels, core
 * components (canonical sorted ID sets), every border row's eligible components
 * and the deterministic visit order actually used. Optional positive weights
 * count as multiplicities for the core test. */
export function dbscan(points, eps, minimum, { order = null, weights = null } = {}) {
  checkPoints(points);
  if (!Number.isFinite(eps) || eps <= 0) throw new RangeError('Use a positive radius.');
  if (!Number.isInteger(minimum) || minimum < 1) throw new RangeError('Use a positive integer count.');
  const n = points.length;
  const visit = order ?? points.map((_, index) => index);
  if (visit.length !== n || new Set(visit).size !== n || visit.some(index => !Number.isInteger(index) || index < 0 || index >= n)) throw new RangeError('The visiting order must be a permutation of the rows.');
  const masses = weights ?? points.map(() => 1);
  if (masses.length !== n || masses.some(value => !Number.isFinite(value) || value <= 0)) throw new RangeError('Weights must be positive.');
  const matrix = distanceMatrix(points);
  const neighbors = matrix.map(row => row.flatMap((value, index) => value <= eps ? [index] : []));
  const counts = neighbors.map(list => list.reduce((sum, index) => sum + masses[index], 0));
  const core = counts.map(count => count >= minimum);
  const labels = points.map(() => -1);
  const componentOf = points.map(() => -1);
  let cluster = 0;
  const expansion = [];
  for (const seed of visit) {
    if (!core[seed] || labels[seed] !== -1) continue;
    labels[seed] = cluster;
    componentOf[seed] = cluster;
    const pending = [seed];
    while (pending.length) {
      const current = pending.pop();
      for (const other of neighbors[current]) {
        if (labels[other] !== -1) continue;
        labels[other] = cluster;
        expansion.push({ from: current, to: other, transmits: core[other] });
        if (core[other]) { componentOf[other] = cluster; pending.push(other); }
      }
    }
    cluster += 1;
  }
  const types = points.map((_, index) => core[index] ? 'core' : labels[index] !== -1 ? 'border' : 'noise');
  const components = Array.from({ length: cluster }, (_, id) => points.map((_, index) => index).filter(index => core[index] && componentOf[index] === id));
  const eligible = points.map((_, index) => core[index] ? [componentOf[index]] : [...new Set(neighbors[index].filter(other => core[other]).map(other => componentOf[other]))].sort((a, b) => a - b));
  return { eps, minimum, matrix, neighbors, counts, core, labels, types, components, eligible, clusters: cluster, expansion, visit,
    coreIds: points.map((_, i) => i).filter(i => types[i] === 'core'), borderIds: points.map((_, i) => i).filter(i => types[i] === 'border'), noiseIds: points.map((_, i) => i).filter(i => types[i] === 'noise') };
}
/** Sorted distances from one row including itself, and the m-th entry c_m (1-based). */
export function coreRadius(points, minimum) {
  const matrix = distanceMatrix(points);
  return matrix.map(row => { const sorted = [...row].sort((a, b) => a - b); return minimum <= sorted.length ? sorted[minimum - 1] : Infinity; });
}
export function sortedRoster(points, index) {
  const matrix = distanceMatrix(points);
  return matrix[index].map((value, other) => ({ index: other, distance: value })).sort((a, b) => a.distance - b.distance || a.index - b.index);
}
/** Standardize columns with population (ddof 0) scale, as StandardScaler does. */
export function standardize(rows) {
  const dimension = checkPoints(rows);
  const mean = Array.from({ length: dimension }, (_, j) => rows.reduce((sum, row) => sum + row[j], 0) / rows.length);
  const scale = Array.from({ length: dimension }, (_, j) => Math.sqrt(rows.reduce((sum, row) => sum + (row[j] - mean[j]) ** 2, 0) / rows.length) || 1);
  return { mean, scale, rows: rows.map(row => row.map((value, j) => (value - mean[j]) / scale[j])) };
}
/** Mean silhouette over the given rows and labels (two or more clusters, fewer than all rows). */
export function silhouette(points, labels) {
  const groups = [...new Set(labels)];
  if (groups.length < 2 || groups.length >= points.length) return null;
  const matrix = distanceMatrix(points);
  let total = 0;
  points.forEach((_, i) => {
    const own = labels[i];
    const ownMembers = labels.flatMap((label, j) => label === own && j !== i ? [j] : []);
    if (ownMembers.length === 0) return;
    const a = ownMembers.reduce((sum, j) => sum + matrix[i][j], 0) / ownMembers.length;
    const b = Math.min(...groups.filter(label => label !== own).map(label => { const members = labels.flatMap((value, j) => value === label ? [j] : []); return members.reduce((sum, j) => sum + matrix[i][j], 0) / members.length; }));
    total += (b - a) / Math.max(a, b);
  });
  return total / points.length;
}
/** Adjusted Rand index between two labelings of the same rows. */
export function adjustedRandIndex(truth, predicted) {
  if (truth.length !== predicted.length || truth.length === 0) throw new RangeError('Compare equal-length labelings.');
  const rowKeys = [...new Set(truth)], columnKeys = [...new Set(predicted)];
  const table = rowKeys.map(() => columnKeys.map(() => 0));
  truth.forEach((t, i) => { table[rowKeys.indexOf(t)][columnKeys.indexOf(predicted[i])] += 1; });
  const pairs = value => value * (value - 1) / 2;
  const sumCells = table.flat().reduce((sum, value) => sum + pairs(value), 0);
  const sumRows = table.map(row => row.reduce((s, v) => s + v, 0)).reduce((sum, value) => sum + pairs(value), 0);
  const sumColumns = columnKeys.map((_, j) => table.reduce((s, row) => s + row[j], 0)).reduce((sum, value) => sum + pairs(value), 0);
  const total = pairs(truth.length);
  const expected = sumRows * sumColumns / total;
  const maximum = (sumRows + sumColumns) / 2;
  if (maximum === expected) return 1;
  return (sumCells - expected) / (maximum - expected);
}
/** Full Iris report for one setting: fit, counts, coverage, conditional metrics. */
export function irisReport(eps, minimum, representation = 'standardized') {
  if (!['standardized', 'raw'].includes(representation)) throw new RangeError('Choose raw or standardized.');
  if (!Number.isFinite(eps) || eps < 0.1 || eps > 2) throw new RangeError('Use a radius between 0.1 and 2.');
  if (!Number.isInteger(minimum) || minimum < 1 || minimum > 20) throw new RangeError('Use a count between 1 and 20.');
  const space = representation === 'standardized' ? standardize(irisRows).rows : irisRows.map(row => [...row]);
  const fit = dbscan(space, eps, minimum);
  const assigned = fit.labels.map((label, i) => label >= 0 ? i : -1).filter(i => i >= 0);
  const assignedLabels = assigned.map(i => fit.labels[i]);
  const sizes = Array.from({ length: fit.clusters }, (_, id) => fit.labels.filter(label => label === id).length);
  return {
    eps, minimum, representation, fit, space,
    clusters: fit.clusters, sizes, coreCount: fit.coreIds.length, borderCount: fit.borderIds.length, noiseCount: fit.noiseIds.length,
    coverage: assigned.length / irisRows.length, assignedIds: assigned, noiseIds: fit.noiseIds,
    silhouetteAssigned: silhouette(assigned.map(i => space[i]), assignedLabels),
    ariAllRows: adjustedRandIndex(irisSpecies, fit.labels),
    ariAssigned: assigned.length ? adjustedRandIndex(assigned.map(i => irisSpecies[i]), assignedLabels) : null,
  };
}
/** Compare two reports on their common retained rows. */
export function compareReports(first, second) {
  const common = first.assignedIds.filter(i => second.assignedIds.includes(i));
  const ariCommon = common.length >= 2 ? adjustedRandIndex(common.map(i => first.fit.labels[i]), common.map(i => second.fit.labels[i])) : null;
  const speciesOnCommon = report => common.length >= 2 ? adjustedRandIndex(common.map(i => irisSpecies[i]), common.map(i => report.fit.labels[i])) : null;
  return { common, onlyFirst: first.assignedIds.filter(i => !second.assignedIds.includes(i)).length, onlySecond: second.assignedIds.filter(i => !first.assignedIds.includes(i)).length, ariCommon, ariSpeciesFirstCommon: speciesOnCommon(first), ariSpeciesSecondCommon: speciesOnCommon(second) };
}
/** The three intended groups of the density-conflict fixture and the exact
 * radius interval (if any) recovering all three with m = 3, under the ordering
 * assumptions stated in the specification. */
export function intervalFixture(offset = 0.75, spacing = 0.75) {
  if (!Number.isFinite(offset) || offset < 0.625 || offset > 2 || !Number.isFinite(spacing) || spacing < 0.125 || spacing > 1) throw new RangeError('Offset 0.625–2 and spacing 0.125–1.');
  const left = variedGroups.left, middle = variedGroups.left.map(x => x + offset), right = [0, 1, 2, 3].map(k => 5 + spacing * k);
  const points = [...left, ...middle, ...right].map(x => [x, 0]);
  const lower = Math.max(0.125, spacing);
  const upper = offset - 0.375;
  return { points, groups: { left, middle, right }, lower, upper, exists: lower < upper, interval: lower < upper ? [lower, upper] : null };
}
/** The 48 ring points: 12 at radius 1, 36 at radius 3, and their DBSCAN labels at ε = 0.6, m = 3. */
export function ringPoints() {
  const inner = Array.from({ length: 12 }, (_, k) => [Math.cos(2 * Math.PI * k / 12), Math.sin(2 * Math.PI * k / 12)]);
  const outer = Array.from({ length: 36 }, (_, k) => [3 * Math.cos(2 * Math.PI * k / 36), 3 * Math.sin(2 * Math.PI * k / 36)]);
  return { points: [...inner, ...outer], ring: [...inner.map(() => 0), ...outer.map(() => 1)] };
}
/** Scale coordinates by per-axis factors (metric change) and optionally the radius (unit change). */
export function transformRows(points, factors, eps, radiusFactor = 1) {
  return { points: points.map(point => point.map((value, axis) => value * factors[axis])), eps: eps * radiusFactor };
}
/** Neighbour matrices of two configurations are identical? Names the first differing pair. */
export function sameNeighborGraph(before, after) {
  const differing = [];
  before.neighbors.forEach((list, i) => { const other = after.neighbors[i]; if (list.join(',') !== other.join(',')) differing.push(i); });
  return { same: differing.length === 0, differing };
}
