/** Pure teaching models for the clustering-evaluation lesson: silhouettes,
 * pair-count agreement, information measures with an exact fixed-margin null,
 * an exact one-dimensional weighted split solver, and Iris helpers that reuse
 * natively fitted artifacts from clustering-evaluation-data.js. Bounded fixtures,
 * not a general library; every value is computed from the active inputs.
 */
import { irisRows, irisSpecies, irisScaler, irisPca, irisFits } from './clustering-evaluation-data.js';

const LN2 = Math.LN2;
const log2 = value => Math.log(value) / LN2;
const distance = (left, right) => Math.hypot(...left.map((value, axis) => value - right[axis]));

// ---- Silhouette ----------------------------------------------------------
export const sixPoints = Object.freeze([[0], [1], [2], [7], [8], [9]].map(point => Object.freeze(point)));
export const sixLabels = Object.freeze(['L', 'L', 'L', 'R', 'R', 'R']);
export const sixNames = Object.freeze(['A', 'B', 'C', 'D', 'E', 'F']);

function checkPoints(points, maximum = 200) {
  if (!Array.isArray(points) || points.length < 2 || points.length > maximum) throw new RangeError(`Use 2 to ${maximum} points.`);
  const dimension = points[0].length;
  for (const point of points) {
    if (!Array.isArray(point) || point.length !== dimension || point.some(value => !Number.isFinite(value) || Math.abs(value) > 1e6)) throw new RangeError('Every point needs the same number of finite coordinates.');
  }
}
/** Per-point silhouettes for a hard partition. Returns `undefined: true` when
 * the partition has fewer than two groups or every point is a singleton; a
 * singleton inside a valid partition gets s = 0 by convention; a = b = 0 gives 0. */
export function silhouetteSamples(points, labels, precomputed = null) {
  checkPoints(points);
  if (labels.length !== points.length) throw new RangeError('One label per point.');
  const groups = [...new Set(labels)];
  const n = points.length;
  if (groups.length < 2 || groups.length >= n) return { undefined: true, reason: groups.length < 2 ? 'fewer than two groups' : 'every observation is a singleton', groups, values: points.map(() => null), mean: null };
  const matrix = precomputed ?? points.map(left => points.map(right => distance(left, right)));
  const details = points.map((_, i) => {
    const own = labels[i];
    const ownMembers = labels.flatMap((label, j) => label === own && j !== i ? [j] : []);
    const foreign = groups.filter(group => group !== own).map(group => {
      const members = labels.flatMap((label, j) => label === group ? [j] : []);
      return { group, members, mean: members.reduce((sum, j) => sum + matrix[i][j], 0) / members.length };
    });
    const b = Math.min(...foreign.map(entry => entry.mean));
    const nearest = foreign.filter(entry => Math.abs(entry.mean - b) <= 1e-12).map(entry => entry.group);
    if (ownMembers.length === 0) return { a: null, b, s: 0, singleton: true, ownMembers, foreign, nearest, tie: nearest.length > 1 };
    const a = ownMembers.reduce((sum, j) => sum + matrix[i][j], 0) / ownMembers.length;
    const s = Math.max(a, b) > 0 ? (b - a) / Math.max(a, b) : 0;
    return { a, b, s, singleton: false, ownMembers, foreign, nearest, tie: nearest.length > 1 };
  });
  const values = details.map(entry => entry.s);
  return { undefined: false, groups, details, values, mean: values.reduce((sum, value) => sum + value, 0) / n, negative: values.filter(value => value < 0).length, matrix };
}
/** Sorted-within-group bar order: group by first appearance, descending s. */
export function silhouetteBarOrder(labels, values) {
  const groups = [...new Set(labels)];
  return groups.flatMap(group => labels.flatMap((label, index) => label === group ? [index] : []).sort((left, right) => values[right] - values[left]));
}
/** Six-point deeper indices (Euclidean): Calinski–Harabasz, Davies–Bouldin, Dunn. */
export function centroidIndices(points, labels) {
  checkPoints(points);
  const groups = [...new Set(labels)];
  const n = points.length, k = groups.length, d = points[0].length;
  const mean = members => Array.from({ length: d }, (_, axis) => members.reduce((sum, i) => sum + points[i][axis], 0) / members.length);
  const grand = mean(points.map((_, i) => i));
  const groupInfo = groups.map(group => { const members = labels.flatMap((label, i) => label === group ? [i] : []); const center = mean(members); return { group, members, center, scatter: members.reduce((sum, i) => sum + distance(points[i], center), 0) / members.length, within: members.reduce((sum, i) => sum + distance(points[i], center) ** 2, 0) }; });
  const W = groupInfo.reduce((sum, info) => sum + info.within, 0);
  const B = groupInfo.reduce((sum, info) => sum + info.members.length * distance(info.center, grand) ** 2, 0);
  const ch = k > 1 && k < n && W > 0 ? (B / (k - 1)) / (W / (n - k)) : null;
  const db = k > 1 ? groupInfo.reduce((sum, info) => sum + Math.max(...groupInfo.filter(other => other !== info).map(other => { const separation = distance(info.center, other.center); return separation > 0 ? (info.scatter + other.scatter) / separation : Infinity; })), 0) / k : null;
  let minBetween = Infinity, maxDiameter = 0;
  points.forEach((left, i) => points.forEach((right, j) => { if (i < j) { const value = distance(left, right); if (labels[i] === labels[j]) maxDiameter = Math.max(maxDiameter, value); else minBetween = Math.min(minBetween, value); } }));
  return { W, B, ch, db, dunn: maxDiameter > 0 && k > 1 ? minBetween / maxDiameter : null };
}

// ---- Pairs, contingency, RI and ARI ----------------------------------------
const choose2 = count => count * (count - 1) / 2;
export function contingency(u, v) {
  if (u.length !== v.length || u.length < 2) throw new RangeError('Compare two labelings of the same observations.');
  const rows = [...new Set(u)], columns = [...new Set(v)];
  const cells = rows.map(row => columns.map(column => u.flatMap((label, i) => label === row && v[i] === column ? [i] : [])));
  const rowSizes = rows.map(row => u.filter(label => label === row).length);
  const columnSizes = columns.map(column => v.filter(label => label === column).length);
  return { rows, columns, cells, rowSizes, columnSizes, n: u.length };
}
/** Pair counting: S together in both, A together in U, B together in V, M all pairs. */
export function pairAgreement(u, v) {
  const table = contingency(u, v);
  const S = table.cells.flat().reduce((sum, ids) => sum + choose2(ids.length), 0);
  const A = table.rowSizes.reduce((sum, size) => sum + choose2(size), 0);
  const B = table.columnSizes.reduce((sum, size) => sum + choose2(size), 0);
  const M = choose2(table.n);
  const TP = S, FN = A - S, FP = B - S, TN = M - A - B + S;
  const RI = (TP + TN) / M;
  const expectedS = A * B / M;
  const upper = (A + B) / 2;
  // Convention: both one group or both all singletons agree perfectly.
  const degenerate = Math.abs(upper - expectedS) < 1e-12;
  const ARI = degenerate ? (A === B ? 1 : 0) : (S - expectedS) / (upper - expectedS);
  const expectedRI = 1 - (A + B) / M + 2 * A * B / (M * M);
  const FM = TP + FP > 0 && TP + FN > 0 ? Math.sqrt(TP / (TP + FP) * TP / (TP + FN)) : null;
  return { table, S, A, B, M, TP, FN, FP, TN, RI, expectedS, expectedRI, upper, ARI, degenerate, FM };
}
/** Every unordered pair with its together/apart status in both labelings. */
export function pairBoard(u, v) {
  const pairs = [];
  for (let i = 0; i < u.length; i += 1) for (let j = i + 1; j < u.length; j += 1) {
    const togetherU = u[i] === u[j], togetherV = v[i] === v[j];
    pairs.push({ i, j, togetherU, togetherV, category: togetherU && togetherV ? 'TP' : togetherU ? 'FN' : togetherV ? 'FP' : 'TN' });
  }
  return pairs;
}

// ---- Information measures ---------------------------------------------------
function entropyOf(labels) {
  const counts = new Map();
  labels.forEach(label => counts.set(label, (counts.get(label) ?? 0) + 1));
  return [...counts.values()].reduce((sum, count) => { const p = count / labels.length; return sum - p * log2(p); }, 0);
}
export function informationMeasures(u, v) {
  const table = contingency(u, v);
  const n = table.n;
  const HU = entropyOf(u), HV = entropyOf(v);
  let I = 0;
  table.cells.forEach((row, r) => row.forEach((ids, c) => { if (ids.length) I += ids.length / n * log2(ids.length * n / (table.rowSizes[r] * table.columnSizes[c])); }));
  I = Math.max(0, I);
  const EMI = expectedMutualInformation(table.rowSizes, table.columnSizes, n);
  const bothConstant = HU === 0 && HV === 0, oneConstant = !bothConstant && (HU === 0 || HV === 0);
  const nmi = bothConstant ? 1 : oneConstant ? 0 : 2 * I / (HU + HV);
  const nmiGeometric = bothConstant ? 1 : oneConstant ? 0 : I / Math.sqrt(HU * HV);
  const denominator = (HU + HV) / 2 - EMI;
  const ami = bothConstant ? 1 : oneConstant ? 0 : Math.abs(denominator) < 1e-12 ? (Math.abs(I - EMI) < 1e-12 ? 1 : 0) : (I - EMI) / denominator;
  return { table, HU, HV, I, HUgivenV: HU - I, HVgivenU: HV - I, EMI, nmi, nmiGeometric, ami, homogeneity: HU === 0 ? 1 : I / HU, completeness: HV === 0 ? 1 : I / HV, vi: HU + HV - 2 * I, degenerateNull: bothConstant || oneConstant, bothConstant, oneConstant };
}
const logFactorial = (() => { const cache = [0]; return count => { while (cache.length <= count) cache.push(cache[cache.length - 1] + Math.log(cache.length)); return cache[count]; }; })();
const logChoose = (n, k) => logFactorial(n) - logFactorial(k) - logFactorial(n - k);
/** Exact expected mutual information (bits) under the fixed-margin
 * hypergeometric null, as in Vinh, Epps and Bailey. */
export function expectedMutualInformation(rowSizes, columnSizes, n) {
  let expected = 0;
  for (const a of rowSizes) for (const b of columnSizes) {
    const low = Math.max(1, a + b - n), high = Math.min(a, b);
    for (let r = low; r <= high; r += 1) {
      const probability = Math.exp(logChoose(a, r) + logChoose(n - a, b - r) - logChoose(n, b));
      expected += probability * (r / n) * log2(r * n / (a * b));
    }
  }
  return expected;
}
/** Enumerate every distinct assignment of V's label multiset to the IDs with U
 * fixed (the fixed-margin permutation null) for at most eight observations. */
export function fixedMarginNull(u, v) {
  if (u.length > 8) throw new RangeError('Enumerate the null for at most eight observations.');
  const counts = new Map();
  v.forEach(label => counts.set(label, (counts.get(label) ?? 0) + 1));
  const labels = [...counts.keys()];
  const assignments = [];
  const current = new Array(u.length);
  const remaining = labels.map(label => counts.get(label));
  const recurse = index => {
    if (index === u.length) { assignments.push([...current]); return; }
    labels.forEach((label, which) => { if (remaining[which] > 0) { remaining[which] -= 1; current[index] = label; recurse(index + 1); remaining[which] += 1; } });
  };
  recurse(0);
  // Overlap r counts IDs in U's first group that receive V's smallest label, so the histogram is stable across assignments.
  const smallest = [...labels].sort()[0];
  const scored = assignments.map(candidate => { const info = informationMeasures(u, candidate); const pairs = pairAgreement(u, candidate); const overlap = candidate.filter((label, i) => u[i] === u[0] && label === smallest).length; return { candidate, overlap, nmi: info.nmi, ami: info.ami, ari: pairs.ARI, mi: info.I }; });
  const overlapCounts = {};
  scored.forEach(entry => { overlapCounts[entry.overlap] = (overlapCounts[entry.overlap] ?? 0) + 1; });
  const mean = key => scored.reduce((sum, entry) => sum + entry[key], 0) / scored.length;
  return { count: scored.length, scored, overlapCounts, meanNmi: mean('nmi'), meanAmi: mean('ami'), meanAri: mean('ari'), meanMi: mean('mi') };
}

// ---- Exact one-dimensional weighted split solver ----------------------------
/** Sorted locations with strictly positive integer weights: choose the
 * minimum-cost split into k contiguous blocks; ties take the lexicographically
 * smaller ordered center tuple. Returns centers, cut positions and cost. */
export function exactLineCenters(locations, weights, k) {
  if (locations.length < 2 || locations.some((value, index) => !Number.isFinite(value) || (index > 0 && value <= locations[index - 1]))) throw new RangeError('Use strictly increasing finite locations.');
  if (weights.length !== locations.length || weights.some(weight => !Number.isInteger(weight) || weight < 1 || weight > 5)) throw new RangeError('Use integer weights from 1 to 5.');
  if (!Number.isInteger(k) || k < 1 || k > 3 || k > locations.length) throw new RangeError('Use k from 1 to 3.');
  const n = locations.length;
  const cutSets = [];
  const build = (start, chosen) => { if (chosen.length === k - 1) { cutSets.push(chosen); return; } for (let cut = start; cut <= n - (k - 1 - chosen.length); cut += 1) build(cut + 1, [...chosen, cut]); };
  build(1, []);
  const alternatives = cutSets.map(cuts => {
    const bounds = [0, ...cuts, n];
    const blocks = bounds.slice(0, -1).map((start, index) => locations.map((_, i) => i).slice(start, bounds[index + 1]));
    const centers = blocks.map(block => block.reduce((sum, i) => sum + weights[i] * locations[i], 0) / block.reduce((sum, i) => sum + weights[i], 0));
    const cost = blocks.reduce((sum, block, index) => sum + block.reduce((inner, i) => inner + weights[i] * (locations[i] - centers[index]) ** 2, 0), 0);
    return { cuts, blocks, centers, cost };
  });
  // Ties in cost (within 1e-9) are broken by the numerically smaller ordered centers, element by element, as Python's (cost, centers) tuple comparison in Program 5 does.
  const compareCenters = (a, b) => { for (let i = 0; i < a.length; i += 1) if (Math.abs(a[i] - b[i]) > 1e-12) return a[i] - b[i]; return 0; };
  alternatives.sort((left, right) => (Math.abs(left.cost - right.cost) > 1e-9 ? left.cost - right.cost : compareCenters(left.centers, right.centers)));
  const best = alternatives[0];
  const ties = alternatives.filter(entry => Math.abs(entry.cost - best.cost) <= 1e-9).length;
  const boundaries = best.centers.slice(1).map((center, index) => (best.centers[index] + center) / 2);
  const probeLabels = locations.map(location => { let bestIndex = 0; best.centers.forEach((center, index) => { if (Math.abs(location - center) < Math.abs(location - best.centers[bestIndex]) - 1e-12) bestIndex = index; }); return bestIndex; });
  return { ...best, ties, boundaries, probeLabels, alternatives };
}

// ---- Ring contrast ------------------------------------------------------------
export function ringContrast() {
  const points = [], ringLabels = [], sliceLabels = [], ids = [];
  for (const radius of [1, 2]) for (let j = 0; j < 16; j += 1) {
    const angle = 2 * Math.PI * (j + 0.25) / 16;
    points.push([radius * Math.cos(angle), radius * Math.sin(angle)]);
    ringLabels.push(radius === 1 ? 'inner' : 'outer');
    sliceLabels.push(Math.cos(angle) >= 0 ? 'right' : 'left');
    ids.push(`r${radius}-${String(j).padStart(2, '0')}`);
  }
  return { points, ids, ringLabels, sliceLabels, ring: silhouetteSamples(points, ringLabels), slice: silhouetteSamples(points, sliceLabels) };
}

// ---- Iris helpers -------------------------------------------------------------
const standardize = row => row.map((value, feature) => (value - irisScaler.mean[feature]) / irisScaler.scale[feature]);
/** Coordinates of every specimen in one of the four declared representations. */
export function irisRepresentation(name, weights = null) {
  if (!['raw4', 'scaled4', 'pca2', 'white2'].includes(name)) throw new RangeError('Unknown representation.');
  const scaled = irisRows.map(standardize);
  let coordinates;
  if (name === 'raw4') coordinates = irisRows.map(row => [...row]);
  else if (name === 'scaled4') coordinates = scaled;
  else {
    const projected = scaled.map(row => irisPca.components.map(component => component.reduce((sum, weight, feature) => sum + weight * row[feature], 0)));
    coordinates = name === 'pca2' ? projected : projected.map(row => row.map((value, axis) => value / Math.sqrt(irisPca.explainedVariance[axis])));
  }
  if (weights) {
    if (weights.length !== coordinates[0].length || weights.some(weight => !Number.isFinite(weight) || weight < 0.25 || weight > 4)) throw new RangeError('Use one weight from 0.25 to 4 per coordinate.');
    coordinates = coordinates.map(row => row.map((value, axis) => value * Math.sqrt(weights[axis])));
  }
  return coordinates;
}
/** A stored native fit, rescored in the browser from the CSV coordinates. */
export function irisFit(name, k, weights = null) {
  const key = `${name}-${k}`;
  if (!irisFits[key]) throw new RangeError('No stored fit for that representation and k.');
  const fit = irisFits[key];
  const coordinates = irisRepresentation(name, weights);
  const silhouette = silhouetteSamples(coordinates, fit.labels);
  const agreement = pairAgreement(irisSpecies, fit.labels);
  const information = informationMeasures(irisSpecies, fit.labels);
  return { ...fit, coordinates, silhouette, ari: agreement.ARI, ami: information.ami, agreement, information, stored: { silhouette: fit.silhouette, ari: fit.ari, ami: fit.ami, negative: fit.negative } };
}
