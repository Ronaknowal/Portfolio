import assert from 'node:assert/strict';
import fs from 'node:fs';
import crypto from 'node:crypto';
import { clusteringPoints, rectanglePoints, assignClusters, moveClusterMeans, lloydTrace, seedingDistribution, hierarchyTrace, cutHierarchy, featureGeometry, palettePixels, quantizePalette } from '../src/learn/data/k-means-hierarchical-models.js';
const counts = {};
const record = name => {
  counts[name] = (counts[name] ?? 0) + 1;
};
function close(actual, expected, label) {
  assert(Math.abs(actual - expected) <= 2e-10 * Math.max(1, Math.abs(expected)), `${label}: ${actual} versus ${expected}`);
}
function distance(left, right) {
  return Math.hypot(...left.map((value, axis) => value - right[axis]));
}
// Pairwise identity: SSE(group) = sum_{i<j} ||xi-xj||² / group size.
// This does not compute a mean using the implementation's centroid helper.
function pairwiseSse(points, members) {
  let sum = 0;
  for (let left = 0; left < members.length; left += 1) {
    for (let right = left + 1; right < members.length; right += 1) {
      sum += distance(points[members[left]], points[members[right]]) ** 2;
    }
  }
  return members.length ? sum / members.length : 0;
}
function partitionSse(points, groups) {
  return groups.reduce((sum, group) => sum + pairwiseSse(points, group), 0);
}
function groupsFromLabels(labels) {
  return [...new Set(labels)].map(label => labels.flatMap((value, index) => value === label ? [index] : []));
}
let bestRectangle = Infinity;
const rectangleObjectives = [];
// Pin P0 to group0 to remove label permutations, while retaining every partition.
for (let mask = 1; mask < 8; mask += 1) {
  const labels = [0, ...[0, 1, 2].map(bit => mask >> bit & 1)];
  const score = partitionSse(rectanglePoints, groupsFromLabels(labels));
  rectangleObjectives.push(score);
  bestRectangle = Math.min(bestRectangle, score);
}
close(bestRectangle, 1, 'Enumerated rectangle global minimum');
for (const [indices, expected] of [[[0, 2], 1], [[0, 1], 9]]) {
  const trace = lloydTrace(rectanglePoints, indices.map(index => rectanglePoints[index]));
  close(trace.at(-1).sse, expected, 'Rectangle fixed point');
  assert.equal(trace.at(-1).status, 'fixed assignment');
  record('enumerated rectangle fixed points');
}
const fixtures = [clusteringPoints, rectanglePoints, [[0, 0], [0, 0], [2, 1], [4, 3], [5, 0]]];
for (const points of fixtures) {
  for (const indices of [[0, points.length - 1], [0, 1], [0, 0]]) {
    const trace = lloydTrace(points, indices.map(index => points[index]));
    let previous = Infinity;
    for (const state of trace.slice(1)) {
      const exact = points.reduce((sum, point, index) => sum + distance(point, state.centers[state.labels[index]]) ** 2, 0);
      close(state.sse, exact, 'Phase residuals use the actual centers and labels');
      assert(state.sse <= previous + 1e-10, 'Alternating objective must not increase');
      if (state.phase === 'move') close(state.sse, partitionSse(points, groupsFromLabels(state.labels)), 'Means attain pairwise SSE identity');
      previous = state.sse;
      record('phasewise residual and mean identities');
    }
  }
}
const tie = assignClusters([[1, 0]], [[0, 0], [2, 0]]);
assert.deepEqual(tie.labels, [0]);
const empty = moveClusterMeans([[1, 0]], [0], [[0, 0], [2, 0]]);
assert.deepEqual(empty.empty, [1]);
assert.deepEqual(empty.centers[1], [2, 0]);
record('explicit tie and empty-center policy');
for (const first of [0, 2, 5]) {
  for (const draw of [0, 0.001, 0.2, 0.5, 0.999]) {
    const result = seedingDistribution(clusteringPoints, [first], draw);
    const independent = clusteringPoints.map(point => distance(point, clusteringPoints[first]) ** 2);
    const total = independent.reduce((sum, value) => sum + value, 0);
    close(result.total, total, 'D squared normalizer');
    close(result.rows.reduce((sum, row) => sum + row.probability, 0), 1, 'Probability sum');
    let cumulative = 0,
      selected = null;
    independent.forEach((value, index) => {
      const start = cumulative;
      cumulative += value / total;
      if (draw >= start && draw < cumulative && value > 0) selected = index;
      close(result.rows[index].probability, value / total, 'Conditional row probability');
    });
    assert.equal(result.selected, selected);
    record('independent D squared cumulative draws');
  }
}
assert.equal(seedingDistribution([[2, 2], [2, 2]], [0]).selected, null);
assert.equal(seedingDistribution([[0, 0], [1, 0], [2, 0]], [0, 2], 0.5).selected, 1);
record('zero-distance and multi-center support');
const heights = {};
for (const linkage of ['single', 'complete', 'average', 'ward']) {
  for (const points of [clusteringPoints, [[0, 0], [1.1, 0.3], [2.4, 4], [4.2, 2], [8, 1]]]) {
    const tree = hierarchyTrace(points, linkage);
    let active = points.map((_, index) => ({
      id: index,
      members: [index]
    }));
    for (const merge of tree.merges) {
      const alternatives = [];
      for (let a = 0; a < active.length; a += 1) {
        for (let b = a + 1; b < active.length; b += 1) {
          const first = active[a],
            second = active[b];
          const allDistances = first.members.flatMap(i => second.members.map(j => distance(points[i], points[j])));
          const delta = pairwiseSse(points, [...first.members, ...second.members]) - pairwiseSse(points, first.members) - pairwiseSse(points, second.members);
          const height = linkage === 'ward' ? Math.sqrt(Math.max(0, 2 * delta)) : linkage === 'single' ? Math.min(...allDistances) : linkage === 'complete' ? Math.max(...allDistances) : allDistances.reduce((sum, value) => sum + value, 0) / allDistances.length;
          alternatives.push({
            left: first.id,
            right: second.id,
            delta,
            height
          });
        }
      }
      const selected = alternatives.find(pair => pair.left === merge.left && pair.right === merge.right);
      assert(selected);
      close(merge.height, Math.min(...alternatives.map(pair => pair.height)), 'Greedy minimum across independent candidate costs');
      close(merge.delta, selected.delta, 'Ward identity also reports actual SSE increase under every linkage');
      close(merge.sse, pairwiseSse(points, merge.members), 'Merged SSE');
      active = [...active.filter(group => group.id !== merge.left && group.id !== merge.right), {
        id: merge.id,
        members: merge.members
      }].sort((a, b) => a.id - b.id);
      record('independent all-candidate linkage minima and Ward identities');
    }
    close(tree.merges.reduce((sum, merge) => sum + merge.delta, 0), pairwiseSse(points, points.map((_, index) => index)), 'SSE increments telescope');
    const scaled = hierarchyTrace(points.map(point => point.map(value => 7 * value)), linkage);
    tree.merges.forEach((merge, index) => {
      close(scaled.merges[index].height, 7 * merge.height, 'Height scale units');
      close(scaled.merges[index].delta, 49 * merge.delta, 'SSE scale units');
      assert.deepEqual(scaled.merges[index].members, merge.members);
    });
    record('height and squared-error unit scaling');
    if (points === clusteringPoints) {
      heights[linkage] = tree.merges.map(merge => merge.height);
      assert.equal(cutHierarchy(tree, 'count', 4).count, 4);
      assert.equal(cutHierarchy(tree, 'height', Math.SQRT1_2).count, 3);
      const candidateHeights = [0, ...tree.merges.flatMap(merge => [Math.max(0, merge.height - 1e-8), merge.height, merge.height + 1e-8])];
      assert(candidateHeights.every(height => cutHierarchy(tree, 'height', height).count !== 4));
      record('equal-height cut cannot realize four clusters');
    }
  }
}
const raw = featureGeometry(1, 1),
  restored = featureGeometry(10, 0.01);
assert.deepEqual(raw.points, restored.points);
assert.deepEqual(raw.labels, restored.labels);
close(raw.sse, restored.sse, 'Unit compensation');
assert.notDeepEqual(featureGeometry(10, 1).labels, raw.labels);
for (const unit of [1, 10]) for (const weight of [0.01, 0.25, 1, 4]) {
  const state = featureGeometry(unit, weight);
  close(state.sse, partitionSse(state.points, groupsFromLabels(state.labels)), 'Transformed geometry SSE');
  record('unit and feature-weight states');
}
for (let count = 1; count <= 6; count += 1) {
  const palette = quantizePalette(count);
  assert.equal(palette.counts.reduce((sum, value) => sum + value, 0), 96);
  const unweighted = palettePixels.reduce((sum, pixel) => {
    const colorIndex = palette.colors.findIndex(color => color.every((value, axis) => value === pixel[axis]));
    return sum + distance(pixel, palette.centers[palette.labels[colorIndex]]) ** 2;
  }, 0);
  close(palette.sse, unweighted, 'Unique-color weights equal expanded pixel contributions');
  const rendered = palettePixels.reduce((sum, pixel, index) => sum + distance(pixel, palette.reconstructed[index]) ** 2, 0);
  close(palette.displayedSse, rendered, 'Rounded actual pixel error');
  close(palette.meanSquaredChannelError, rendered / 288, 'Channel denominator');
  record('expanded pixel versus weighted palette objectives');
}
close(quantizePalette(6).displayedSse, 0, 'All six original colors reconstruct exactly');
const invalid = [() => assignClusters([], [[0, 0]]), () => assignClusters([[0,,]], [[0, 0]]), () => assignClusters([[0, 0]], [[0]]), () => assignClusters([[0, Infinity]], [[0, 0]]), () => assignClusters([[0, 0]], [[0, 0]], [0]), () => moveClusterMeans([[0, 0]], [2], [[0, 0]]), () => lloydTrace([[0, 0]], [[0, 0]], null, 0), () => seedingDistribution([[0, 0]], [0], 1), () => seedingDistribution([[0, 0]], [0, 0]), () => hierarchyTrace(clusteringPoints, 'centroid'), () => cutHierarchy(hierarchyTrace(clusteringPoints), 'count', 0), () => featureGeometry(100, 1), () => quantizePalette(7)];
invalid.forEach(check => assert.throws(check, RangeError));
const sources = ['src/learn/data/k-means-hierarchical-models.js', 'src/learn/components/lesson-labs/KMeansHierarchicalLabs.jsx', 'src/learn/components/lesson-labs/k-means-hierarchical-labs.css'];
const hash = file => crypto.createHash('sha256').update(fs.readFileSync(file)).digest('hex');
const evidence = {
  checkedAt: new Date().toISOString(),
  sourceHashes: Object.fromEntries(sources.map(file => [file, hash(file)])),
  verifierHash: hash('scripts/verify-k-means-hierarchical-models.mjs'),
  counts,
  totalGroupedChecks: Object.values(counts).reduce((sum, value) => sum + value, 0),
  rejectedInputs: invalid.length,
  rectangleObjectives,
  fixtureHeights: heights,
  scope: 'Bounded independent pairwise-SSE, exhaustive four-point partitions, all-candidate linkage, expanded-pixel and unit/probability oracles. Component hashes identify the proposed UI, not a completed browser review.',
  limitations: ['No universal k-means global optimum or initialization guarantee is inferred.', 'Hierarchy input is bounded to 16 points; exact floating-point ties use deterministic IDs.', 'Numeric squared sRGB error is not perceptual image quality or encoded file size.', 'Root owns actual browser and lesson integration checks.'],
  passed: true
};
fs.writeFileSync('docs/teaching/evidence/k-means-hierarchical-models.json', JSON.stringify(evidence, null, 2) + '\n');
console.log(`PASS: ${evidence.totalGroupedChecks} grouped complementary model checks and ${invalid.length} rejection contracts.`);
