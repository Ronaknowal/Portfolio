// Bounded independent checks of the clustering-evaluation browser models against
// manuscript values, direct pair enumeration, exact null enumeration, an
// independent exhaustive partition search and the natively fitted Iris record.
import assert from 'node:assert/strict';
import fs from 'node:fs';
import crypto from 'node:crypto';
import { sixPoints, sixLabels, silhouetteSamples, silhouetteBarOrder, centroidIndices, pairAgreement, pairBoard, informationMeasures, expectedMutualInformation, fixedMarginNull, exactLineCenters, ringContrast, irisRepresentation, irisFit } from '../src/learn/data/clustering-evaluation-models.js';
import { irisRows, irisSpecies, irisFits } from '../src/learn/data/clustering-evaluation-data.js';

const counts = {};
const record = name => { counts[name] = (counts[name] ?? 0) + 1; };
const close = (actual, expected, label, tolerance = 1e-9) => assert(Math.abs(actual - expected) <= tolerance * Math.max(1, Math.abs(expected)), `${label}: ${actual} versus ${expected}`);
const author = JSON.parse(fs.readFileSync('docs/teaching/drafts/clustering-evaluation-validation-silhouette-ari-nmi/author-calculations.json', 'utf8'));
const visual = JSON.parse(fs.readFileSync('docs/teaching/drafts/clustering-evaluation-validation-silhouette-ari-nmi/visual-input-calculations.json', 'utf8'));

// Silhouette: six-point fixture and every specified state.
const base = silhouetteSamples(sixPoints, sixLabels);
base.values.forEach((value, i) => close(value, author.silhouette.base.values[i], `base s ${i}`));
close(base.mean, author.silhouette.base.mean, 'base mean');
close(base.details[2].a, 1.5, 'a(C)'); close(base.details[2].b, 6, 'b(C)');
const moved = silhouetteSamples(sixPoints, ['L', 'L', 'R', 'R', 'R', 'R']);
moved.values.forEach((value, i) => close(value, author.silhouette.move_row2.values[i], `moved s ${i}`));
close(moved.mean, author.silhouette.move_row2.mean, 'moved mean');
const shifted = silhouetteSamples([[0], [1], [3], [7], [8], [9]], sixLabels);
close(shifted.details[2].a, 2.5, 'shifted a'); close(shifted.details[2].b, 5, 'shifted b'); close(shifted.values[2], 0.5, 'shifted s');
const scaled = silhouetteSamples(sixPoints.map(point => [2 * point[0]]), sixLabels);
scaled.values.forEach((value, i) => close(value, base.values[i], 'common scale invariant'));
const singleton = silhouetteSamples(sixPoints, ['L', 'L', 'L', 'R', 'R', 'S']);
singleton.values.forEach((value, i) => close(value, author.silhouette.singleton.values[i], `singleton s ${i}`));
assert.equal(singleton.details[5].singleton, true);
const zero = silhouetteSamples([[0], [0], [0], [0], [0], [0]], sixLabels);
zero.values.forEach(value => assert.equal(value, 0));
assert.equal(silhouetteSamples(sixPoints, ['L', 'L', 'L', 'L', 'L', 'L']).undefined, true);
assert.equal(silhouetteSamples(sixPoints, ['a', 'b', 'c', 'd', 'e', 'f']).undefined, true);
const rejected = silhouetteSamples(sixPoints, ['0', '0', '-1', '1', '1', '1']);
close(rejected.mean, author.noise.all_label, 'rejection as singleton group');
const retained = silhouetteSamples(sixPoints.filter((_, i) => i !== 2), ['0', '0', '1', '1', '1']);
close(retained.mean, author.noise.conditional, 'conditional retained mean');
assert.deepEqual(silhouetteBarOrder(sixLabels, base.values), [1, 0, 2, 4, 5, 3]);
const indices = centroidIndices(sixPoints, sixLabels);
close(indices.ch, author.internal_other.CH, 'Calinski–Harabasz'); close(indices.db, author.internal_other.DB, 'Davies–Bouldin'); close(indices.dunn, 2.5, 'Dunn'); close(indices.W, 4, 'W'); close(indices.B, 73.5, 'B');
record('silhouette fixtures');
// Silhouette against a direct independent implementation on random clouds.
let seed = 7; const random = () => { seed = (seed * 1103515245 + 12345) % 2147483648; return seed / 2147483648; };
for (let trial = 0; trial < 5; trial += 1) {
  const points = Array.from({ length: 9 }, () => [random() * 10, random() * 10]);
  const labels = points.map((_, i) => ['x', 'y', 'z'][i % 3]);
  const result = silhouetteSamples(points, labels);
  points.forEach((point, i) => {
    const own = points.filter((_, j) => labels[j] === labels[i] && j !== i).map(other => Math.hypot(point[0] - other[0], point[1] - other[1]));
    const a = own.reduce((s, v) => s + v, 0) / own.length;
    const b = Math.min(...['x', 'y', 'z'].filter(g => g !== labels[i]).map(g => { const members = points.filter((_, j) => labels[j] === g); return members.reduce((s, other) => s + Math.hypot(point[0] - other[0], point[1] - other[1]), 0) / members.length; }));
    close(result.values[i], (b - a) / Math.max(a, b), `random cloud ${trial} point ${i}`);
  });
  record('independent silhouette clouds');
}
// Pairs: manuscript partitions, direct enumeration equals contingency shortcut.
const U = [0, 0, 0, 0, 1, 1, 1, 1];
for (const [name, v] of Object.entries({ same: [3, 3, 3, 3, 7, 7, 7, 7], cross: [0, 0, 1, 1, 0, 0, 1, 1], refine: [0, 0, 1, 1, 2, 2, 3, 3], constant: [0, 0, 0, 0, 0, 0, 0, 0], one_swap: [0, 0, 0, 1, 0, 1, 1, 1] })) {
  const pairs = pairAgreement(U, v), info = informationMeasures(U, v), board = pairBoard(U, v);
  close(pairs.RI, author.partitions[name].RI, `${name} RI`); close(pairs.ARI, author.partitions[name].ARI, `${name} ARI`);
  close(info.nmi, author.partitions[name].NMI, `${name} NMI`, 1e-8); close(info.ami, author.partitions[name].AMI, `${name} AMI`, 1e-8);
  assert.equal(board.filter(pair => pair.category === 'TP').length, pairs.TP); assert.equal(board.filter(pair => pair.category === 'TN').length, pairs.TN);
  assert.equal(board.length, 28);
  record(`partition ${name}`);
}
const swap = pairAgreement(U, [0, 0, 0, 1, 0, 1, 1, 1]);
assert.deepEqual([swap.S, swap.A, swap.B, swap.M], [6, 12, 12, 28]); close(swap.expectedRI, 25 / 49, 'expected RI'); close(swap.ARI, 0.125, 'swap ARI'); close(swap.FM, 0.5, 'Fowlkes–Mallows');
close(pairAgreement(U, [0, 0, 1, 1, 0, 0, 1, 1]).ARI, -1 / 6, 'cross ARI');
close(pairAgreement(U, [0, 0, 1, 1, 2, 2, 3, 3]).ARI, 4 / 11, 'refine ARI');
assert.equal(pairAgreement([0, 0, 0], [1, 1, 1]).ARI, 1, 'both single group');
assert.equal(pairAgreement([0, 1, 2], [5, 6, 7]).ARI, 1, 'both all singletons');
const refine = informationMeasures(U, [0, 0, 1, 1, 2, 2, 3, 3]);
close(refine.HU, 1, 'H(U)'); close(refine.HV, 2, 'H(V)'); close(refine.I, 1, 'I'); close(refine.nmi, 2 / 3, 'arithmetic NMI'); close(refine.nmiGeometric, Math.SQRT1_2, 'geometric NMI'); close(refine.homogeneity, 1, 'homogeneity'); close(refine.completeness, 0.5, 'completeness'); close(refine.vi, 1, 'VI');
const singletons = informationMeasures(U, [0, 1, 2, 3, 4, 5, 6, 7]);
close(singletons.nmi, 0.5, 'singleton NMI'); close(singletons.ami, 0, 'singleton AMI', 1e-9); close(pairAgreement(U, [0, 1, 2, 3, 4, 5, 6, 7]).ARI, 0, 'singleton ARI');
assert.equal(informationMeasures([0, 0, 0], [1, 1, 1]).nmi, 1); assert.equal(informationMeasures([0, 0, 1], [1, 1, 1]).nmi, 0); assert.equal(informationMeasures([0, 0, 1], [1, 1, 1]).ami, 0);
record('information measures');
// Exact null: 70 balanced assignments; hypergeometric expectation equals enumeration.
const nullBalanced = fixedMarginNull(U, [0, 0, 1, 1, 0, 0, 1, 1]);
assert.equal(nullBalanced.count, 70);
assert.deepEqual(nullBalanced.overlapCounts, { 0: 1, 1: 16, 2: 36, 3: 16, 4: 1 });
close(nullBalanced.meanNmi, author.exact_null.mean[1], 'mean NMI'); close(nullBalanced.meanAri, 0, 'mean ARI', 1e-9); close(nullBalanced.meanAmi, 0, 'mean AMI', 1e-9);
close(nullBalanced.meanMi, expectedMutualInformation([4, 4], [4, 4], 8), 'enumerated E[MI] equals hypergeometric');
const unbalanced = fixedMarginNull(U, [0, 0, 0, 1, 1, 1, 1, 1]);
assert.equal(unbalanced.count, 56);
close(unbalanced.meanMi, expectedMutualInformation([4, 4], [3, 5], 8), 'unbalanced E[MI]');
close(unbalanced.meanAri, 0, 'unbalanced mean ARI', 1e-9);
record('fixed-margin null');
// Exact line solver: manuscript states, independent brute force over all label assignments.
const left = exactLineCenters([0, 1, 4, 5, 8, 9], [3, 3, 1, 1, 1, 1], 2), right = exactLineCenters([0, 1, 4, 5, 8, 9], [1, 1, 1, 1, 3, 3], 2);
assert.deepEqual(left.centers, [0.5, 6.5]); assert.deepEqual(left.probeLabels, [0, 0, 1, 1, 1, 1]); close(left.cost, author.resamples.left.cost, 'left cost');
assert.deepEqual(right.centers, [2.5, 8.5]); assert.deepEqual(right.probeLabels, [0, 0, 0, 0, 1, 1]); close(right.cost, author.resamples.right.cost, 'right cost');
close(pairAgreement(left.probeLabels, right.probeLabels).ARI, -1 / 14, 'probe ARI');
const uniform = exactLineCenters([0, 1, 4, 5, 8, 9], [1, 1, 1, 1, 1, 1], 2);
close(uniform.cost, author.resamples.uniform.cost, 'uniform cost');
const changed = exactLineCenters([0, 1, 2, 8, 9, 10], [1, 1, 1, 1, 1, 1], 2);
assert.deepEqual(changed.centers, [1, 9]); close(changed.cost, 4, 'changed cost');
const tripled = exactLineCenters([0, 3, 6, 24, 27, 30], [1, 1, 1, 1, 1, 1], 2);
assert.deepEqual(tripled.centers, [3, 27]); close(tripled.cost, 36, 'tripled cost'); assert.deepEqual(tripled.probeLabels, changed.probeLabels);
assert.equal(exactLineCenters([0, 1, 4, 5, 8, 9], [2, 1, 4, 1, 1, 5], 1).centers.length, 1);
// Tie-breaking: among equal-cost contiguous splits the solver must keep the numerically smaller ordered centers,
// as Python's (cost, centers) tuple comparison does in Program 5. The enumeration below never calls the solver's ordering.
const independentBest = (locations, weights, k) => {
  const n = locations.length, out = [];
  const walk = (start, cuts) => { if (cuts.length === k - 1) { out.push(cuts); return; } for (let cut = start; cut <= n - (k - 1 - cuts.length); cut += 1) walk(cut + 1, [...cuts, cut]); };
  walk(1, []);
  const entries = out.map(cuts => {
    const bounds = [0, ...cuts, n], centers = [];
    const cost = bounds.slice(0, -1).reduce((sum, s, b) => {
      const idx = []; for (let i = s; i < bounds[b + 1]; i += 1) idx.push(i);
      const w = idx.reduce((a, i) => a + weights[i], 0), c = idx.reduce((a, i) => a + weights[i] * locations[i], 0) / w;
      centers.push(c);
      return sum + idx.reduce((a, i) => a + weights[i] * (locations[i] - c) ** 2, 0);
    }, 0);
    return { centers, cost };
  });
  const minCost = Math.min(...entries.map(e => e.cost));
  const tied = entries.filter(e => Math.abs(e.cost - minCost) <= 1e-9);
  tied.sort((a, b) => { for (let i = 0; i < a.centers.length; i += 1) if (Math.abs(a.centers[i] - b.centers[i]) > 1e-12) return a.centers[i] - b.centers[i]; return 0; });
  return { centers: tied[0].centers, ties: tied.length };
};
const negativeTie = exactLineCenters([-4, -2, 0], [1, 1, 1], 2);
assert.equal(negativeTie.ties, 2); assert.deepEqual(negativeTie.centers, [-4, -1]);
const symmetricTie = exactLineCenters([0, 2, 4], [1, 1, 1], 2);
assert.equal(symmetricTie.ties, 2); assert.deepEqual(symmetricTie.centers, [0, 3]);
let tieSeed = 20260912, tiedCases = 0;
const tieRand = () => { tieSeed = (tieSeed * 1103515245 + 12345) % 2147483648; return tieSeed / 2147483648; };
for (let trial = 0; trial < 3000; trial += 1) {
  const n = 3 + Math.floor(tieRand() * 4), k = 1 + Math.floor(tieRand() * Math.min(3, n));
  const set = new Set(); while (set.size < n) set.add(Math.floor(tieRand() * 81) - 40);
  const locations = [...set].sort((a, b) => a - b), weights = locations.map(() => 1 + Math.floor(tieRand() * 5));
  const solved = exactLineCenters(locations, weights, k), reference = independentBest(locations, weights, k);
  assert.equal(solved.ties, reference.ties, `tie count ${JSON.stringify([locations, weights, k])}`);
  solved.centers.forEach((c, i) => close(c, reference.centers[i], `tie-break centers ${JSON.stringify([locations, weights, k])}`, 1e-9));
  if (reference.ties > 1) tiedCases += 1;
}
assert.ok(tiedCases >= 3, `random search should meet tied optima, met ${tiedCases}`);
record('split tie-breaking');
// Brute force: every 2^6 labelings (any, not only contiguous) cannot beat the contiguous optimum.
for (const weights of [[3, 3, 1, 1, 1, 1], [1, 2, 5, 1, 4, 1], [5, 5, 5, 1, 1, 1]]) {
  const solution = exactLineCenters([0, 1, 4, 5, 8, 9], weights, 2);
  let best = Infinity;
  for (let mask = 1; mask < 63; mask += 1) {
    const groups = [[], []]; [0, 1, 2, 3, 4, 5].forEach(i => groups[mask >> i & 1].push(i));
    const cost = groups.reduce((sum, group) => { const w = group.reduce((s, i) => s + weights[i], 0); const c = group.reduce((s, i) => s + weights[i] * [0, 1, 4, 5, 8, 9][i], 0) / w; return sum + group.reduce((s, i) => s + weights[i] * ([0, 1, 4, 5, 8, 9][i] - c) ** 2, 0); }, 0);
    best = Math.min(best, cost);
  }
  close(solution.cost, best, `weights ${weights}: contiguous optimum equals brute force`);
  record('exact split brute force');
}
// Ring contrast matches the author's saved geometry.
const rings = ringContrast();
rings.points.forEach((point, i) => point.forEach((value, axis) => close(value, visual.ring_contrast.coordinates[i][axis], 'ring coordinate', 1e-12)));
close(rings.ring.mean, visual.ring_contrast.ring_mean, 'ring mean'); close(rings.slice.mean, visual.ring_contrast.slice_mean, 'slice mean');
record('ring contrast');
// Iris: browser rescoring reproduces the native record for all eight fits.
for (const key of Object.keys(irisFits)) {
  const fit = irisFit(irisFits[key].representation, irisFits[key].k);
  close(fit.silhouette.mean, fit.stored.silhouette, `${key} silhouette`, 1e-8);
  close(fit.ari, fit.stored.ari, `${key} ARI`, 1e-8); close(fit.ami, fit.stored.ami, `${key} AMI`, 1e-7);
  assert.equal(fit.silhouette.negative, fit.stored.negative, `${key} negative count`);
  record('iris fits');
}
assert.equal(pairAgreement(irisFits['scaled4-3'].labels, irisFits['pca2-3'].labels).ARI, 1, 'scaled4 and pca2 k=3 are the same partition');
const frozen = weights => silhouetteSamples(irisRepresentation('scaled4', weights), irisFits['scaled4-3'].labels);
close(frozen([1, 1, 1, 1]).mean, visual.fixed_partition_rescores.unit.mean, 'unit rescore');
close(frozen([1, 1, 1, 0.25]).mean, visual.fixed_partition_rescores.petal_width_quarter.mean, 'quarter rescore');
close(frozen([1, 1, 1, 4]).mean, visual.fixed_partition_rescores.petal_width_four.mean, 'four rescore');
close(frozen([4, 4, 4, 4]).mean, visual.fixed_partition_rescores.common_four.mean, 'common four rescore');
frozen([4, 4, 4, 4]).values.forEach((value, i) => close(value, frozen([1, 1, 1, 1]).values[i], 'common weight leaves every s unchanged'));
assert.equal(frozen([1, 1, 1, 4]).negative, visual.fixed_partition_rescores.petal_width_four.negative);
irisRepresentation('pca2').forEach((row, i) => row.forEach((value, axis) => close(value, visual.iris_view.unwhitened_pca2_coordinates[i][axis], 'pca2 view', 1e-9)));
assert.equal(irisRows.length, 150); assert.equal(irisSpecies.filter(label => label === 0).length, 50);
assert.throws(() => irisRepresentation('scaled4', [1, 1, 1, 9]), RangeError);
assert.throws(() => exactLineCenters([0, 1, 1], [1, 1, 1], 2), RangeError);
assert.throws(() => silhouetteSamples([[0], [1]], ['a']), RangeError);
record('iris rescoring');

const sources = ['src/learn/data/clustering-evaluation-models.js', 'src/learn/data/clustering-evaluation-data.js', 'src/learn/components/lesson-labs/ClusteringEvaluationLabs.jsx', 'src/learn/components/lesson-labs/ClusteringEvaluationFigures.jsx', 'src/learn/components/lesson-labs/clustering-evaluation-labs.css'];
const hash = file => crypto.createHash('sha256').update(fs.readFileSync(file)).digest('hex');
const evidence = {
  checkedAt: new Date().toISOString(),
  sourceHashes: Object.fromEntries(sources.filter(file => fs.existsSync(file)).map(file => [file, hash(file)])),
  verifierHash: hash('scripts/verify-clustering-evaluation-models.mjs'),
  counts, totalGroupedChecks: Object.values(counts).reduce((sum, value) => sum + value, 0),
  scope: 'Six-point silhouette states including singleton, zero, rejection and undefined; five random clouds against a direct implementation; five eight-ID partitions with direct pair enumeration; exact 70- and 56-assignment fixed-margin nulls against the hypergeometric expectation; exact split solver against brute force over all labelings; ring contrast; all eight native Iris fits rescored in the browser model within 1e-8; frozen-weight rescoring and the full-rank invariant.',
  limitations: ['Iris labels are natively fitted inputs, not browser k-means.', 'The null enumeration is bounded to eight observations.', 'Rendering, interaction and independent review are separate.'],
  passed: true,
};
fs.mkdirSync('docs/teaching/evidence', { recursive: true });
fs.writeFileSync('docs/teaching/evidence/clustering-evaluation-models.json', JSON.stringify(evidence, null, 2) + '\n');
console.log(`PASS: ${evidence.totalGroupedChecks} grouped clustering-evaluation model checks.`);
