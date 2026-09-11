// Exact, bounded teaching fixtures; no empirical performance or timing claims.
export const inspectionRows = [
  { id: 'A', features: [1, 1], label: 0 }, { id: 'B', features: [2, 1], label: 0 },
  { id: 'C', features: [1, 3], label: 1 }, { id: 'D', features: [2, 3], label: 1 },
  { id: 'E', features: [4, 1], label: 1 }, { id: 'F', features: [5, 1], label: 1 },
  { id: 'G', features: [4, 3], label: 0 }, { id: 'H', features: [5, 3], label: 1 },
];
export const xorRows = [
  { id: 'A', features: [0, 0], label: 0 }, { id: 'B', features: [0, 1], label: 1 },
  { id: 'C', features: [1, 0], label: 1 }, { id: 'D', features: [1, 1], label: 0 },
];

function validateRows(rows) {
  if (!Array.isArray(rows) || !rows.length || rows.length > 64 || rows.some(row => !row || !Array.isArray(row.features) || row.features.length !== 2 || row.features.some(value => !Number.isFinite(value) || Math.abs(value) > 1000) || ![0, 1].includes(row.label))) throw new RangeError('Use 1–64 binary-labelled rows with two finite features in [−1000,1000].');
}
export function impurity(labels, criterion = 'gini') {
  if (!Array.isArray(labels) || labels.length > 64 || labels.some(label => ![0, 1].includes(label)) || !['gini', 'entropy'].includes(criterion)) throw new RangeError('Unsupported labels or impurity.');
  if (!labels.length) return 0;
  const probability = labels.reduce((sum, label) => sum + label, 0) / labels.length;
  if (criterion === 'gini') return 2 * probability * (1 - probability);
  return [probability, 1 - probability].reduce((sum, mass) => sum - (mass ? mass * Math.log2(mass) : 0), 0);
}
export function splitCandidates(rows, { criterion = 'gini', minimumLeaf = 1, features = [0, 1] } = {}) {
  validateRows(rows);
  if (!Number.isInteger(minimumLeaf) || minimumLeaf < 1 || minimumLeaf > 32 || !Array.isArray(features) || !features.length || features.some(feature => ![0, 1].includes(feature))) throw new RangeError('Unsupported split settings.');
  const parentImpurity = impurity(rows.map(row => row.label), criterion);
  const candidates = [];
  for (const feature of [...new Set(features)].sort()) {
    const values = [...new Set(rows.map(row => row.features[feature]))].sort((a, b) => a - b);
    for (let index = 0; index < values.length - 1; index++) {
      const midpoint = values[index] + (values[index + 1] - values[index]) / 2;
      const threshold = midpoint < values[index + 1] ? midpoint : values[index];
      const left = rows.filter(row => row.features[feature] <= threshold);
      const right = rows.filter(row => row.features[feature] > threshold);
      const leftImpurity = impurity(left.map(row => row.label), criterion);
      const rightImpurity = impurity(right.map(row => row.label), criterion);
      const weighted = (left.length * leftImpurity + right.length * rightImpurity) / rows.length;
      candidates.push({ feature, threshold, left, right, parentImpurity, leftImpurity, rightImpurity, weighted, gain: parentImpurity - weighted, allowed: left.length >= minimumLeaf && right.length >= minimumLeaf });
    }
  }
  return candidates;
}
export function growTree(rows = inspectionRows, { maxDepth = 3, minimumLeaf = 1, criterion = 'gini', allowZero = true, featureSelector = null } = {}) {
  validateRows(rows);
  if (!Number.isInteger(maxDepth) || maxDepth < 0 || maxDepth > 6 || typeof allowZero !== 'boolean') throw new RangeError('Use a depth from 0 to 6 and a boolean zero-gain policy.');
  splitCandidates(rows, { criterion, minimumLeaf });
  function grow(current, depth, path) {
    const probability = current.reduce((sum, row) => sum + row.label, 0) / current.length;
    const node = { path, depth, count: current.length, probability, label: Number(probability > .5), impurity: impurity(current.map(row => row.label), criterion), rows: current.map(row => row.id) };
    if (depth === maxDepth || node.impurity === 0 || current.length < 2 * minimumLeaf) return node;
    const features = featureSelector ? featureSelector(path) : [0, 1];
    const candidates = splitCandidates(current, { criterion, minimumLeaf, features }).filter(candidate => candidate.allowed);
    let best = null;
    for (const candidate of candidates) if (!best || candidate.gain > best.gain + 1e-12) best = candidate;
    if (!best || best.gain < -1e-12 || (!allowZero && best.gain <= 1e-12)) return node;
    return { ...node, feature: best.feature, threshold: best.threshold, gain: best.gain, consideredFeatures: features, left: grow(best.left, depth + 1, path + 'L'), right: grow(best.right, depth + 1, path + 'R') };
  }
  return grow(rows, 0, 'root');
}
export function treePrediction(tree, features) {
  if (!Array.isArray(features) || features.length !== 2 || features.some(value => !Number.isFinite(value) || Math.abs(value) > 1000)) throw new RangeError('Supply two bounded finite query values.');
  const path = [];
  let node = tree;
  while (node.left) {
    const left = features[node.feature] <= node.threshold;
    path.push({ path: node.path, feature: node.feature, threshold: node.threshold, value: features[node.feature], direction: left ? 'left' : 'right' });
    node = left ? node.left : node.right;
  }
  return { probability: node.probability, label: node.label, count: node.count, leafPath: node.path, rows: node.rows, path };
}
export function leafRegions(tree, bounds = [0, 6, 0, 4]) {
  if (!tree.left) return [{ ...tree, bounds }];
  const low = [...bounds]; const high = [...bounds];
  low[tree.feature * 2 + 1] = tree.threshold;
  high[tree.feature * 2] = tree.threshold;
  return [...leafRegions(tree.left, low), ...leafRegions(tree.right, high)];
}
export function prunedSubtrees(tree) {
  const leaf = { ...tree };
  delete leaf.left; delete leaf.right; delete leaf.feature; delete leaf.threshold; delete leaf.gain;
  if (!tree.left) return [leaf];
  return [leaf, ...prunedSubtrees(tree.left).flatMap(left => prunedSubtrees(tree.right).map(right => ({ ...tree, left, right })))];
}
export function pruningReport(alpha = .04) {
  if (!Number.isFinite(alpha) || alpha < 0 || alpha > .3) throw new RangeError('Use alpha in [0,.3].');
  const tree = growTree();
  const candidates = prunedSubtrees(tree).map((subtree, index) => {
    const leaves = leafRegions(subtree);
    const risk = leaves.reduce((sum, leaf) => sum + leaf.count * leaf.impurity / tree.count, 0);
    return { index, tree: subtree, leaves: leaves.length, risk, objective: risk + alpha * leaves.length };
  }).sort((a, b) => a.objective - b.objective || a.leaves - b.leaves || a.index - b.index);
  return { alpha, candidates, best: candidates[0] };
}
function randomGenerator(seed) {
  let state = seed >>> 0;
  return () => { state = (Math.imul(state, 1664525) + 1013904223) >>> 0; return state / 4294967296; };
}
export function forestReport({ trees = 6, selectedRow = 0, featureSampling = true } = {}) {
  if (!Number.isInteger(trees) || trees < 1 || trees > 12 || !Number.isInteger(selectedRow) || selectedRow < 0 || selectedRow > 7 || typeof featureSampling !== 'boolean') throw new RangeError('Use 1–12 trees and a fixture row 0–7.');
  const query = inspectionRows[selectedRow];
  const members = Array.from({ length: trees }, (_, index) => {
    const random = randomGenerator(29 + index * 101);
    const sampleIndices = Array.from({ length: 8 }, () => Math.floor(random() * 8));
    const sample = sampleIndices.map(rowIndex => inspectionRows[rowIndex]);
    const tree = growTree(sample, { maxDepth: 3, minimumLeaf: 1, featureSelector: featureSampling ? () => [Math.floor(random() * 2)] : null });
    return { index, sampleIndices, tree, omitted: !sampleIndices.includes(selectedRow), prediction: treePrediction(tree, query.features) };
  });
  const eligible = members.filter(member => member.omitted);
  const average = list => list.reduce((sum, member) => sum + member.prediction.probability, 0) / list.length;
  return { query, members, probability: average(members), oobCount: eligible.length, oobProbability: eligible.length ? average(eligible) : null };
}
export function ensembleVariance(trees, correlation) {
  if (!Number.isInteger(trees) || trees < 1 || trees > 200 || !Number.isFinite(correlation) || correlation < 0 || correlation > 1) throw new RangeError('Use B=1–200 and correlation in [0,1].');
  return correlation + (1 - correlation) / trees;
}
export function permutationRows(mode = 'copy') {
  if (!['copy', 'used', 'group'].includes(mode)) throw new RangeError('Choose a supported permutation.');
  const truth = [0, 0, 1, 1];
  const order = [2, 3, 0, 1];
  return truth.map((label, index) => {
    const first = mode === 'copy' ? label : truth[order[index]];
    const second = mode === 'used' ? label : truth[order[index]];
    return { id: index + 1, original: [label, label], transformed: [first, second], label, predicted: first, impossiblePair: first !== second };
  });
}
