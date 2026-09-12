import assert from 'node:assert/strict';
import fs from 'node:fs';
import crypto from 'node:crypto';
import { clusteringPoints, rectanglePoints, chainPoints, assignClusters, moveClusterMeans, lloydTrace, enumerateTwoGroupPartitions, seedingDistribution, seedingFrequencies, hierarchyTrace, cutHierarchy, featureGeometry, paletteImages, paletteLimit, palettePixels, quantizePalette } from '../src/learn/data/k-means-hierarchical-models.js';

function verifyArithmeticRange() {
  const rejected = [
    ['distinct tiny assignment', () => assignClusters([[1e-200, 0]], [[0, 0]])],
    ['distinct tiny hierarchy', () => hierarchyTrace([[0], [1e-200]])],
    ['Ward factor erases positive subnormal separation', () => hierarchyTrace([[0], [2e-162]])],
    ['positive weighted assignment disappears', () => assignClusters([[1e-160]], [[0]], [1e-10])],
    ['positive weighted moved error disappears', () => moveClusterMeans([[0], [1e-160]], [0, 0], [[0]], [1e-10, 1e-10])],
    ['nonzero weighted coordinate disappears', () => moveClusterMeans([[1e-200]], [0], [[1e-200]], [1e-200])],
    ['positive D squared probability disappears', () => seedingDistribution([[0], [1e-160], [10000]], [0])],
    ['one nonzero coordinate square disappears', () => assignClusters([[1, 1e-200]], [[0, 0]])],
  ];
  rejected.forEach(([, check]) => assert.throws(check, RangeError));
  const accepted = [];
  for (const separation of [1e-150, 1e-160, 1e-161]) {
    const assigned = assignClusters([[separation]], [[0]]);
    const tree = hierarchyTrace([[0], [separation]]);
    assert(assigned.sse > 0 && Number.isFinite(assigned.sse));
    assert(tree.root.height > 0 && tree.root.delta > 0);
    accepted.push({ separation, sse: assigned.sse, height: tree.root.height });
  }
  assert.equal(assignClusters([[0], [0]], [[0]]).sse, 0);
  assert.equal(hierarchyTrace([[0], [0]]).root.height, 0);
  assert.equal(seedingDistribution([[2, 2], [2, 2]], [0]).stopped, true);
  return { rejected: rejected.map(([name]) => name), acceptedPositiveStates: accepted, genuineZeroCases: 3 };
}

if (process.argv.includes('--range-amendment')) {
  const archive = 'docs/teaching/archive/k-means-hierarchical-models-before-range-fix';
  const previous = await import(`../${archive}/k-means-hierarchical-models.js`);
  const prior = JSON.parse(fs.readFileSync(`${archive}/k-means-hierarchical-models.json`, 'utf8'));
  assert.equal(previous.assignClusters([[1e-200, 0]], [[0, 0]]).sse, 0);
  assert.equal(previous.hierarchyTrace([[0], [1e-200]]).root.height, 0);
  const boundary = verifyArithmeticRange();
  let unchanged = 0;
  for (const points of [clusteringPoints, rectanglePoints]) {
    for (const pair of [[0, points.length - 1], [0, 1], [0, 0]]) {
      const centers = pair.map(index => points[index]);
      assert.deepEqual(lloydTrace(points, centers), previous.lloydTrace(points, centers));
      unchanged += 1;
    }
  }
  for (const linkage of ['single', 'complete', 'average', 'ward']) {
    assert.deepEqual(hierarchyTrace(clusteringPoints, linkage), previous.hierarchyTrace(clusteringPoints, linkage));
    unchanged += 1;
  }
  for (const first of [0, 2, 5]) {
    assert.deepEqual(seedingDistribution(clusteringPoints, [first], 0.5), previous.seedingDistribution(clusteringPoints, [first], 0.5));
    unchanged += 1;
  }
  for (const [unit, weight] of [[1, 1], [10, 1], [10, 0.01]]) {
    assert.deepEqual(featureGeometry(unit, weight), previous.featureGeometry(unit, weight));
    unchanged += 1;
  }
  for (let count = 1; count <= 6; count += 1) {
    assert.deepEqual(quantizePalette(count), previous.quantizePalette(count));
    unchanged += 1;
  }
  const digest = file => crypto.createHash('sha256').update(fs.readFileSync(file)).digest('hex');
  const evidence = {
    checkedAt: new Date().toISOString(),
    sourceHashes: Object.fromEntries(Object.keys(prior.sourceHashes).map(file => [file, digest(file)])),
    verifierHash: digest('scripts/verify-k-means-hierarchical-models.mjs'),
    previousVerification: { record: `${archive}/k-means-hierarchical-models.json`, checkedAt: prior.checkedAt, sourceHashes: prior.sourceHashes, groupedChecks: prior.totalGroupedChecks, rejectedInputs: prior.rejectedInputs, reused: 'Prior126+13 campaign retained as historical evidence; it was not rerun for this amendment.' },
    reproducedPreviousFailures: ['Distinct1e-200 assignment SSE was zero.', 'Distinct1e-200 Ward hierarchy height was zero.'],
    boundary,
    unchangedFixtureOutputs: unchanged,
    changes: ['Reject nonzero squared separations and weighted terms or quotients that underflow to zero.', 'Reject positive separation lost in Ward size scaling.', 'Initial Lloyd caption describes an unassigned state.', 'Visible D squared key includes every positive probability, its point ID, color and selected marker.'],
    scope: 'Focused stage5 amendment: arithmetic range regressions and exact previous/final output conservation on existing finite fixtures. Root owns final actual-route UI review.',
    passed: true,
  };
  fs.writeFileSync('docs/teaching/evidence/k-means-hierarchical-models.json', JSON.stringify(evidence, null, 2) + '\n');
  console.log(`PASS amendment: ${boundary.rejected.length} range rejections, ${boundary.acceptedPositiveStates.length} representable tiny positive cases, ${boundary.genuineZeroCases} genuine-zero cases, ${unchanged} unchanged complete fixture outputs.`);
  process.exit(0);
}

const arithmeticRange = verifyArithmeticRange();
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

// Exhaustive two-group enumeration: every partition, exact pairwise SSE, sorted.
for (const points of [rectanglePoints, clusteringPoints, [[0], [1], [4], [5], [10]]]) {
  const enumerated = enumerateTwoGroupPartitions(points);
  assert.equal(enumerated.length, 2 ** (points.length - 1) - 1, 'Every two-group split appears once');
  enumerated.forEach((partition, index) => {
    close(partition.sse, partitionSse(points, partition.groups), 'Enumerated split SSE equals pairwise identity');
    if (index > 0) assert(partition.sse >= enumerated[index - 1].sse - 1e-12, 'Splits are sorted by SSE');
    record('exhaustive two-group partitions');
  });
}
for (const [unit, weight] of [[1, 1], [10, 1], [10, 0.25]]) {
  const state = featureGeometry(unit, weight);
  assert.equal(state.partitions.length, 7);
  assert.deepEqual(state.labels, state.partitions[0].labels);
  const lloydBest = Math.min(...[[0, 2], [0, 1]].map(indices => lloydTrace(state.points, indices.map(index => state.points[index])).at(-1).sse));
  assert(state.sse <= lloydBest + 1e-10, 'Exhaustive optimum is never worse than a Lloyd fixed point');
  record('exhaustive optimum bounds Lloyd fixed points');
}
// Seeded repeated draws reproduce the exact D² distribution within sampling error.
for (const selected of [[0], [4], [0, 5]]) {
  const frequencies = seedingFrequencies(clusteringPoints, selected, 2000, 1);
  const again = seedingFrequencies(clusteringPoints, selected, 2000, 1);
  assert.deepEqual(frequencies.counts, again.counts, 'Fixed seed reproduces counts');
  assert.equal(frequencies.counts.reduce((sum, value) => sum + value, 0), 2000);
  const distribution = seedingDistribution(clusteringPoints, selected, 0);
  distribution.rows.forEach(row => {
    assert(Math.abs(frequencies.frequencies[row.index] - row.probability) < 0.05, `Frequency ${frequencies.frequencies[row.index]} far from probability ${row.probability}`);
    if (row.probability === 0) assert.equal(frequencies.counts[row.index], 0, 'Zero-probability rows are never drawn');
  });
  assert.equal(frequencies.farthest, distribution.rows.reduce((best, row) => row.distance > best.distance ? row : best).index);
  close(frequencies.uniformProbability, 1 / (6 - selected.length), 'Uniform comparison');
  record('seeded frequency draws versus exact D² probabilities');
}
assert.equal(seedingFrequencies([[2, 2], [2, 2]], [0]).stopped, true);
// Chain fixture: single linkage keeps the six chain rows together at k = 2; complete breaks them.
const chainIntact = linkage => cutHierarchy(hierarchyTrace(chainPoints, linkage), 'count', 2).groups.some(group => [...group].sort((a, b) => a - b).join(',') === '0,1,2,3,4,5');
assert.equal(chainIntact('single'), true);
assert.equal(chainIntact('complete'), false);
assert.equal(chainIntact('average'), true);
assert.equal(chainIntact('ward'), true);
const chainSingle = hierarchyTrace(chainPoints, 'single');
assert.equal(chainSingle.merges.filter(merge => merge.height === 1).length, 5, 'Five chain merges share the exact height 1 under single linkage');
assert.equal(cutHierarchy(chainSingle, 'height', 1).count, 2);
assert.equal(cutHierarchy(chainSingle, 'height', 0.999).count, 7);
const chainComplete = hierarchyTrace(chainPoints, 'complete');
assert.deepEqual(cutHierarchy(chainComplete, 'count', 2).groups.map(group => [...group].sort((a, b) => a - b)), [[0, 1, 2, 3], [4, 5, 6, 7, 8]], 'Complete linkage attaches the triple to the right pair before the chain rejoins');
for (const linkage of ['single', 'complete', 'average', 'ward']) {
  const tree = hierarchyTrace(chainPoints, linkage);
  close(tree.merges.reduce((sum, merge) => sum + merge.delta, 0), pairwiseSse(chainPoints, chainPoints.map((_, index) => index)), 'Chain SSE increments telescope');
  record('chain fixture linkage contrast');
}
// Additional palette images: weighted objective equals expanded pixels; rounded error is real.
for (const imageId of Object.keys(paletteImages)) {
  const limit = paletteLimit(imageId);
  assert(limit >= 1 && limit <= 8);
  assert.throws(() => quantizePalette(limit + 1, imageId), RangeError);
  for (let count = 1; count <= limit; count += 1) {
    const palette = quantizePalette(count, imageId);
    assert.equal(palette.counts.reduce((sum, value) => sum + value, 0), palette.image.pixels.length);
    const expanded = palette.image.pixels.reduce((sum, pixel) => {
      const colorIndex = palette.colors.findIndex(color => color.every((value, axis) => value === pixel[axis]));
      return sum + distance(pixel, palette.centers[palette.labels[colorIndex]]) ** 2;
    }, 0);
    close(palette.sse, expanded, `Weighted ${imageId} objective equals expanded pixels`);
    const rendered = palette.image.pixels.reduce((sum, pixel, index) => sum + distance(pixel, palette.reconstructed[index]) ** 2, 0);
    close(palette.displayedSse, rendered, `Rounded ${imageId} pixel error`);
    if (count > 1) assert(palette.sse <= quantizePalette(count - 1, imageId).sse + 1e-9, `${imageId}: more entries never worsen this deterministic fit`);
    assert.equal(palette.status, 'fixed assignment', `${imageId} k=${count} reached a fixed point within the sweep budget`);
    record(`palette image ${imageId}`);
  }
}
assert.equal(quantizePalette(1, 'gradient').uniqueCount, 160);
assert.equal(quantizePalette(1, 'sky').counts.every(count => count >= 1), true);
const invalid = [() => assignClusters([], [[0, 0]]), () => assignClusters([[0,,]], [[0, 0]]), () => assignClusters([[0, 0]], [[0]]), () => assignClusters([[0, Infinity]], [[0, 0]]), () => assignClusters([[0, 0]], [[0, 0]], [0]), () => moveClusterMeans([[0, 0]], [2], [[0, 0]]), () => lloydTrace([[0, 0]], [[0, 0]], null, 0), () => seedingDistribution([[0, 0]], [0], 1), () => seedingDistribution([[0, 0]], [0, 0]), () => hierarchyTrace(clusteringPoints, 'centroid'), () => cutHierarchy(hierarchyTrace(clusteringPoints), 'count', 0), () => featureGeometry(100, 1), () => quantizePalette(7)];
invalid.forEach(check => assert.throws(check, RangeError));
const sources = ['src/learn/data/k-means-hierarchical-models.js', 'src/learn/data/k-means-hierarchical-faithful.js', 'src/learn/components/lesson-labs/KMeansHierarchicalLabs.jsx', 'src/learn/components/lesson-labs/k-means-hierarchical-labs.css', 'src/learn/components/lesson-labs/KMeansHierarchicalFigures.jsx'];
const hash = file => crypto.createHash('sha256').update(fs.readFileSync(file)).digest('hex');
const evidence = {
  checkedAt: new Date().toISOString(),
  sourceHashes: Object.fromEntries(sources.map(file => [file, hash(file)])),
  verifierHash: hash('scripts/verify-k-means-hierarchical-models.mjs'),
  counts,
  totalGroupedChecks: Object.values(counts).reduce((sum, value) => sum + value, 0),
  rejectedInputs: invalid.length,
  arithmeticRange,
  rectangleObjectives,
  fixtureHeights: heights,
  scope: 'Bounded independent pairwise-SSE, exhaustive two-group partitions, all-candidate linkage on the six-point and chain fixtures, seeded D² frequency draws, expanded-pixel objectives for three images and unit/probability oracles. Component hashes identify the proposed UI, not a completed browser review.',
  limitations: ['No universal k-means global optimum or initialization guarantee is inferred.', 'Hierarchy input is bounded to 16 points; exact floating-point ties use deterministic IDs.', 'Numeric squared sRGB error is not perceptual image quality or encoded file size.', 'Root owns actual browser and lesson integration checks.'],
  passed: true
};
fs.writeFileSync('docs/teaching/evidence/k-means-hierarchical-models.json', JSON.stringify(evidence, null, 2) + '\n');
console.log(`PASS: ${evidence.totalGroupedChecks} grouped complementary model checks and ${invalid.length} rejection contracts.`);
