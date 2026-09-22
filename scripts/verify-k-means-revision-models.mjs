/** Independent focused review of the pending K-Means revision. Read-only runtime
 * inspection: never imports old verifiers that overwrite historical evidence.
 * Run normally for models + native/data; --models-only reuses the source-bound
 * native receipt after checking its current input identities. */
import assert from 'node:assert/strict';
import fs from 'node:fs';
import crypto from 'node:crypto';
import { spawnSync } from 'node:child_process';
import * as models from '../src/learn/data/k-means-hierarchical-models.js';
import * as faithful from '../src/learn/data/k-means-hierarchical-faithful.js';
import { clusteringExamples as examples } from '../src/learn/data/k-means-hierarchical-examples.js';

const evidencePath = 'docs/teaching/evidence/k-means-revision-model-review.json';
const nativePath = 'docs/teaching/evidence/k-means-revision-native-review.json';
const hash = value => crypto.createHash('sha256').update(value).digest('hex');
const hashFile = file => hash(fs.readFileSync(file));
const sourceFiles = ['src/learn/data/k-means-hierarchical-models.js', 'src/learn/data/k-means-hierarchical-faithful.js', 'src/learn/data/k-means-hierarchical-examples.js'];
const nativeInputs = [sourceFiles[1], sourceFiles[2], 'scripts/verify-k-means-hierarchical-examples.py', 'scripts/verify-k-means-revision-native.py', 'docs/teaching/evidence/k-means-revision-faithful-source.csv'];
const sourceHashes = Object.fromEntries(sourceFiles.map(file => [file, hashFile(file)]));
fs.writeFileSync(evidencePath, JSON.stringify({ checkedAt: new Date().toISOString(), status: 'in-progress', sourceHashes }, null, 2) + '\n');
const groups = {};
const record = name => { groups[name] = (groups[name] ?? 0) + 1; };
const close = (actual, expected, label, tolerance = 1e-9) => assert(Math.abs(actual - expected) <= tolerance * Math.max(1, Math.abs(expected)), `${label}: ${actual} != ${expected}`);
const squaredDistance = (left, right) => Math.hypot(...left.map((value, axis) => value - right[axis])) ** 2;
// Pairwise variance identity is independent of the implementation's mean-based SSE.
function groupSse(points, members) {
  let sum = 0;
  for (let a = 0; a < members.length; a += 1) for (let b = a + 1; b < members.length; b += 1) sum += squaredDistance(points[members[a]], points[members[b]]);
  return sum / members.length;
}
const labelsToGroups = labels => [...new Set(labels)].map(label => labels.flatMap((value, index) => value === label ? [index] : []));
const groupSignature = groups => groups.map(group => [...group].sort((a, b) => a - b).join(',')).sort().join(';');
const { clusteringPoints, rectanglePoints, chainPoints, lloydTrace, enumerateTwoGroupPartitions, featureGeometry, hierarchyTrace, cutHierarchy, seedingDistribution, seedingFrequencies, quantizePalette, paletteImages, paletteLimit } = models;

for (const points of [clusteringPoints, rectanglePoints]) {
  for (let first = 0; first < points.length; first += 1) for (let second = 0; second < points.length; second += 1) {
    const trace = lloydTrace(points, [points[first], points[second]]);
    assert.equal(trace.at(-1).status, 'fixed assignment');
    let previous = Infinity;
    for (const state of trace.slice(1)) {
      const residual = points.reduce((sum, point, index) => sum + squaredDistance(point, state.centers[state.labels[index]]), 0);
      close(state.sse, residual, 'Every assignment/move uses its displayed labels and centers');
      assert(state.sse <= previous + 1e-9);
      if (state.phase === 'move') close(state.sse, labelsToGroups(state.labels).reduce((sum, members) => sum + groupSse(points, members), 0), 'Every updated mean attains the pairwise SSE identity');
      else {
        const expected = points.map(point => state.centers.map(center => squaredDistance(point, center))).map(row => row.indexOf(Math.min(...row)));
        assert.deepEqual(state.labels, expected);
      }
      previous = state.sse;
    }
    record(`all ordered ${points.length}-row seed pairs, including duplicate centers`);
  }
}

for (const unit of [1, 10]) for (const weight of [0.01, 0.25, 1, 4]) {
  const state = featureGeometry(unit, weight);
  const scored = [];
  for (let mask = 1; mask < 8; mask += 1) {
    const labels = [0, ...[0, 1, 2].map(bit => mask >> bit & 1)];
    const groups = labelsToGroups(labels);
    scored.push({ signature: groupSignature(groups), sse: groups.reduce((sum, group) => sum + groupSse(state.points, group), 0) });
  }
  assert.equal(new Set(state.partitions.map(partition => groupSignature(partition.groups))).size, 7);
  for (const item of state.partitions) close(item.sse, scored.find(row => row.signature === groupSignature(item.groups)).sse, 'All seven partitions score correctly');
  close(state.sse, Math.min(...scored.map(row => row.sse)), 'Reported partition is the exhaustive optimum');
  record('all eight unit/weight settings and every partition');
}
assert.deepEqual(featureGeometry(10, 0.01), { ...featureGeometry(1, 1) });
record('compensated units preserve every partition, center and objective');
for (const points of [[[0], [0]], [[0], [1], [1]], [[0, 0], [1, 3], [2, 1], [3, 4], [5, 2], [7, 3], [8, 1], [9, 4]]]) {
  const partitions = enumerateTwoGroupPartitions(points);
  assert.equal(partitions.length, 2 ** (points.length - 1) - 1);
  assert.equal(new Set(partitions.map(row => groupSignature(row.groups))).size, partitions.length);
  for (const row of partitions) close(row.sse, row.groups.reduce((sum, group) => sum + groupSse(points, group), 0), 'Changed fixture exhaustive objective');
  record('enumeration size boundaries and duplicate coordinates');
}

// The same named PRNG is reimplemented with exact BigInt 32-bit arithmetic.
// Select by integer mass cross-products, avoiding the production float CDF.
function integerDraws(seed, count) {
  const mask = 0xffffffffn;
  let state = BigInt(seed);
  return Array.from({ length: count }, () => {
    state = (state + 0x6d2b79f5n) & mask;
    let word = ((state ^ (state >> 15n)) * (state | 1n)) & mask;
    word ^= (word + (((word ^ (word >> 7n)) * (word | 61n)) & mask)) & mask;
    return (word ^ (word >> 14n)) & mask;
  });
}
const randomDraws = integerDraws(1, 200);
const frequencyQuestionMismatches = [];
const bucket = probability => probability >= 0.999 ? 'always' : probability >= 0.5 ? 'more' : 'less';
for (let mask = 1; mask < 64; mask += 1) {
  const selected = clusteringPoints.flatMap((_, index) => mask >> index & 1 ? [index] : []);
  const state = seedingDistribution(clusteringPoints, selected, 0);
  const frequencies = seedingFrequencies(clusteringPoints, selected, 200, 1);
  const integerMasses = clusteringPoints.map(point => BigInt(Math.round(4 * Math.min(...selected.map(index => squaredDistance(point, clusteringPoints[index]))))));
  const total = integerMasses.reduce((sum, value) => sum + value, 0n);
  assert.equal(state.stopped, total === 0n);
  close(state.total * 4, Number(total), 'D² normalizer');
  if (total === 0n) {
    assert.equal(frequencies.uniformProbability, 0);
    assert.equal(frequencies.farthest, null);
    assert.deepEqual(frequencies.counts, [0, 0, 0, 0, 0, 0]);
  } else {
    const expectedCounts = [0, 0, 0, 0, 0, 0];
    for (const draw of randomDraws) {
      let cumulative = 0n;
      const chosen = integerMasses.findIndex(mass => { cumulative += mass; return mass > 0n && draw * total < cumulative * 4294967296n; });
      assert(chosen >= 0);
      expectedCounts[chosen] += 1;
    }
    assert.deepEqual(frequencies.counts, expectedCounts);
    close(frequencies.uniformProbability, 1 / (6 - selected.length), 'Uniform remaining-row comparator');
    assert.equal(frequencies.counts.reduce((sum, count) => sum + count, 0), 200);
    state.rows.forEach((row, index) => {
      close(row.probability, Number(integerMasses[index]) / Number(total), 'Exact conditional row probability');
      if (row.probability > 0) assert.equal(seedingDistribution(clusteringPoints, selected, (row.start + row.end) / 2).selected, index);
      if (selected.includes(index)) assert.equal(row.probability, 0);
    });
    if (bucket(frequencies.farthestProbability) !== bucket(frequencies.frequencies[frequencies.farthest])) frequencyQuestionMismatches.push({ selected, farthest: frequencies.farthest, exactProbability: frequencies.farthestProbability, countOf200: frequencies.counts[frequencies.farthest] });
  }
  record('every nonempty selected subset: exact D² support and independent 200 draws');
}
assert.equal(frequencyQuestionMismatches.length, 17);
assert.deepEqual(seedingFrequencies([[2], [2]], [0]).counts, [0, 0]);
record('all-duplicate D² stops before every row is chosen');

for (const points of [clusteringPoints, chainPoints]) for (const linkage of ['single', 'complete', 'average', 'ward']) {
  const tree = hierarchyTrace(points, linkage);
  let active = points.map((_, index) => ({ id: index, members: [index] }));
  for (const merge of tree.merges) {
    const candidates = [];
    for (let a = 0; a < active.length; a += 1) for (let b = a + 1; b < active.length; b += 1) {
      const first = active[a], second = active[b];
      const delta = groupSse(points, [...first.members, ...second.members]) - groupSse(points, first.members) - groupSse(points, second.members);
      const lengths = first.members.flatMap(i => second.members.map(j => Math.sqrt(squaredDistance(points[i], points[j]))));
      const height = linkage === 'ward' ? Math.sqrt(Math.max(0, 2 * delta)) : linkage === 'single' ? Math.min(...lengths) : linkage === 'complete' ? Math.max(...lengths) : lengths.reduce((sum, value) => sum + value, 0) / lengths.length;
      candidates.push({ first: first.id, second: second.id, height, delta });
    }
    const actual = candidates.find(row => row.first === merge.left && row.second === merge.right);
    assert(actual);
    close(merge.height, Math.min(...candidates.map(row => row.height)), 'Selected merge is an independently calculated minimum');
    close(merge.delta, actual.delta, 'Merge SSE increase from pairwise identity');
    close(merge.sse, groupSse(points, merge.members), 'Merged SSE');
    active = [...active.filter(group => group.id !== merge.left && group.id !== merge.right), { id: merge.id, members: merge.members }];
  }
  for (let count = 1; count <= points.length; count += 1) {
    const cut = cutHierarchy(tree, 'count', count);
    assert.equal(cut.count, count);
    assert.deepEqual(cut.groups.flat().sort((a, b) => a - b), points.map((_, index) => index));
    close(cut.sse, cut.groups.reduce((sum, group) => sum + groupSse(points, group), 0), 'Cut SSE from independent partition identity');
  }
  for (const height of [0, ...new Set(tree.merges.map(row => row.height)), tree.root.height + 1]) {
    const cut = cutHierarchy(tree, 'height', height);
    assert.equal(cut.count, points.length - tree.merges.filter(merge => merge.height <= height).length);
    assert.deepEqual(cut.groups.flat().sort((a, b) => a - b), points.map((_, index) => index));
  }
  record('all-candidate hierarchy costs and every count/distinct-height cut');
}
assert.equal(cutHierarchy(hierarchyTrace(chainPoints, 'single'), 'height', 1).count, 2);
assert.equal(cutHierarchy(hierarchyTrace(chainPoints, 'single'), 'height', 1 - Number.EPSILON).count, 7);
record('five exact tied chain merges enter a height cut together');

const paletteSummaries = [];
for (const imageId of Object.keys(paletteImages)) for (let k = 1; k <= paletteLimit(imageId); k += 1) {
  const state = quantizePalette(k, imageId);
  assert.equal(state.status, 'fixed assignment');
  assert.equal(new Set(state.colors.map(color => color.join(','))).size, state.uniqueCount);
  assert.equal(state.counts.reduce((sum, count) => sum + count, 0), state.image.pixels.length);
  let floating = 0, integer = 0;
  for (let index = 0; index < state.image.pixels.length; index += 1) {
    const pixel = state.image.pixels[index];
    const colorIndex = state.colors.findIndex(color => color.every((value, axis) => value === pixel[axis]));
    floating += squaredDistance(pixel, state.centers[state.labels[colorIndex]]);
    integer += squaredDistance(pixel, state.reconstructed[index]);
    assert.deepEqual(state.reconstructed[index], state.roundedCenters[state.labels[colorIndex]]);
  }
  close(state.sse, floating, 'Unique weighted SSE equals full pixel residual sum');
  close(state.displayedSse, integer, 'Displayed reconstruction error uses actual integer pixels');
  close(state.meanSquaredChannelError, integer / (3 * state.image.pixels.length), 'Channel error normalization');
  if (k === paletteLimit(imageId)) paletteSummaries.push({ imageId, maximumEntries: k, uniqueColors: state.uniqueCount, pixelCount: state.image.pixels.length, displayedSse: state.displayedSse });
  record('every palette/image setting: weights, fixed point, displayed pixels and error');
}
assert.deepEqual(paletteSummaries.map(row => [row.uniqueColors, row.displayedSse]), [[6, 0], [37, 35945], [160, 165886]]);

const originalNative = JSON.parse(fs.readFileSync('docs/teaching/evidence/k-means-hierarchical-native.json'));
const nativeReuse = Object.fromEntries(Object.entries(examples).map(([name, item]) => {
  const old = originalNative.programs[name];
  const unchangedCode = hash(item.code) === old.codeHash;
  assert.equal(hash(item.expected), old.stdoutHash, `${name} displayed output changed`);
  if (!['lloyd', 'ward'].includes(name)) assert(unchangedCode);
  return [name, { codeSha256: hash(item.code), expectedSha256: hash(item.expected), unchangedCode, execution: unchangedCode ? 'Retained exact-code/output prior execution' : 'Fresh complete program, stdout conserved' }];
}));
let native;
if (process.argv.includes('--models-only')) {
  native = JSON.parse(fs.readFileSync(nativePath));
  assert(native.passed);
  for (const file of nativeInputs) assert.equal(native.sourceHashes[file], hashFile(file), `Native reuse stale: ${file}`);
} else {
  fs.writeFileSync(nativePath, JSON.stringify({ status: 'in-progress', passed: false }, null, 2) + '\n');
  const processResult = spawnSync('scratch/lesson-tools/Scripts/python.exe', ['-X', 'utf8', 'scripts/verify-k-means-revision-native.py'], { input: JSON.stringify({ examples, faithful }), encoding: 'utf8', maxBuffer: 4 * 1024 * 1024, timeout: 120000 });
  assert.equal(processResult.status, 0, processResult.stderr || processResult.error?.message);
  native = { ...JSON.parse(processResult.stdout), checkedAt: new Date().toISOString(), sourceHashes: Object.fromEntries(nativeInputs.map(file => [file, hashFile(file)])) };
  fs.writeFileSync(nativePath, JSON.stringify(native, null, 2) + '\n');
}
const evidence = { checkedAt: new Date().toISOString(), status: 'passed', sourceHashes, verifierHashes: Object.fromEntries(['scripts/verify-k-means-revision-models.mjs', 'scripts/verify-k-means-revision-native.py'].map(file => [file, hashFile(file)])), groups, groupedCases: Object.values(groups).reduce((sum, count) => sum + count, 0), frequencyQuestionMismatches, paletteSummaries, nativeReuse, nativeReceipt: { file: nativePath, sha256: hashFile(nativePath), checks: native.checks.length }, priorReceipts: ['docs/teaching/evidence/k-means-hierarchical-native.json', 'docs/teaching/evidence/k-means-hierarchical-models.json'].map(file => ({ file, sha256: hashFile(file) })), limitations: ['Only fixed teaching fixtures and documented controls are covered; this is not a general numerical-library proof.', 'Browser labels, prediction/reveal gates, charts and integration are owned by the root review.', 'The historical broad native campaign is reused for unchanged programs; only changed Lloyd/Ward and new Faithful bindings receive fresh numerical execution.'] };
fs.writeFileSync(evidencePath, JSON.stringify(evidence, null, 2) + '\n');
console.log(`PASS ${evidence.groupedCases} focused model groups; ${native.checks.length} native/data groups. Native source hashes sealed.`);
console.log(JSON.stringify(native.interpretive, null, 2));
