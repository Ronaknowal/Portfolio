// Bounded independent checks of the DBSCAN browser models against the author
// probes, the manuscript's exact fixtures and hand-derived identities.
import assert from 'node:assert/strict';
import fs from 'node:fs';
import crypto from 'node:crypto';
import { trailPoints, variedGroups, cornerPoints, fifthCorner, dbscan, coreRadius, sortedRoster, standardize, silhouette, adjustedRandIndex, irisReport, compareReports, intervalFixture, ringPoints, transformRows, sameNeighborGraph } from '../src/learn/data/dbscan-models.js';
import { irisRows, irisSpecies, trailOptics, trailHdbscan, ringKmeans } from '../src/learn/data/dbscan-iris-data.js';

const author = JSON.parse(fs.readFileSync('docs/teaching/drafts/dbscan-density-based-clustering/author-calculations.json', 'utf8'));
const supplementary = JSON.parse(fs.readFileSync('docs/teaching/drafts/dbscan-density-based-clustering/supplementary-author-calculations.json', 'utf8'));
const counts = {};
const record = name => { counts[name] = (counts[name] ?? 0) + 1; };
const close = (actual, expected, label, tolerance = 1e-9) => assert(Math.abs(actual - expected) <= tolerance * Math.max(1, Math.abs(expected)), `${label}: ${actual} versus ${expected}`);
const sameSets = (left, right) => JSON.stringify(left.map(g => [...g].sort((a, b) => a - b)).sort()) === JSON.stringify(right.map(g => [...g].sort((a, b) => a - b)).sort());
const groupsFromLabels = labels => [...new Set(labels.filter(l => l >= 0))].map(l => labels.flatMap((v, i) => v === l ? [i] : []));

// Trail at five radii versus the author's sklearn probes.
for (const probe of author.street) {
  const fit = dbscan(trailPoints, probe.eps, probe.min_samples);
  assert.deepEqual(fit.coreIds, probe.core_ids, `core ids at ε=${probe.eps}`);
  assert.deepEqual(fit.borderIds, probe.border_ids, `border ids at ε=${probe.eps}`);
  assert.deepEqual(fit.noiseIds, probe.noise_ids, `noise ids at ε=${probe.eps}`);
  assert.equal(fit.clusters, probe.clusters);
  assert(sameSets(groupsFromLabels(fit.labels), groupsFromLabels(probe.labels)), `partition at ε=${probe.eps}`);
  record('trail radii versus sklearn');
}
const base = dbscan(trailPoints, 1, 4);
assert.deepEqual(base.counts, author.street_neighbor_counts_eps1);
assert.deepEqual(base.eligible[8], [0, 1], 'I is eligible for both components');
assert.equal(base.labels[8], base.labels[3], 'forward order attaches I to D’s component');
const reversed = dbscan(trailPoints, 1, 4, { order: [9, 8, 7, 6, 5, 4, 3, 2, 1, 0] });
assert.deepEqual(reversed.types, base.types, 'types are order-invariant');
assert(sameSets(reversed.components, base.components), 'core components are order-invariant');
assert.equal(reversed.labels[8], reversed.labels[4], 'reversed order attaches I to E’s component');
assert.deepEqual(author.reverse_order.labels_in_original_order.map(l => l >= 0 ? 'assigned' : 'noise'), reversed.labels.map(l => l >= 0 ? 'assigned' : 'noise'));
assert(base.expansion.every(step => step.transmits || base.types[step.to] === 'border'), 'only core rows transmit');
assert.deepEqual(coreRadius(trailPoints, 4), author.street_core_distance_m4);
assert.deepEqual([...coreRadius(trailPoints, 4)].sort((a, b) => a - b), [0.5, 0.5, 0.5, 0.5, 0.75, 0.75, 0.75, 0.75, 1.25, 2.75]);
assert.equal(coreRadius(trailPoints, 4).filter(v => v <= 1).length, 8);
assert.deepEqual(sortedRoster(trailPoints, 3).map(entry => entry.distance), [0, 0.25, 0.5, 0.75, 1, 2, 2.25, 2.5, 2.75, 5]);
assert.equal(coreRadius(trailPoints, 11)[0], Infinity, 'm beyond n has no core radius');
const moved = dbscan([...trailPoints.slice(0, 8), [0, 0.125], [4, 0]], 1, 4);
assert.equal(moved.types[8], 'noise', 'I moved off the line is noise under both orders');
assert.deepEqual(dbscan([[2, 0], [2, 0], [2, 0], [2, 0]], 0.125, 4).types, ['core', 'core', 'core', 'core']);
assert.equal(dbscan(trailPoints, 0.125, 1).clusters, 10, 'm = 1 makes every row its own component');
const weighted = dbscan([[0, 0], [2, 0]], 0.25, 3, { weights: [3, 1] });
assert.deepEqual(weighted.types, ['core', 'noise'], 'multiplicity weights restore the core');
assert.deepEqual(dbscan([[0, 0], [2, 0]], 0.25, 3).types, ['noise', 'noise']);
record('trail identities');
// Practice fixtures.
const five = dbscan([0, 0.25, 0.5, 0.75, 2].map(x => [x, 0]), 0.25, 3);
assert.deepEqual(five.types, ['border', 'core', 'core', 'border', 'noise']);
assert.deepEqual(dbscan([0, 0.25, 0.5, 0.75, 2].map(x => [x, 0]), 0.25, 4).types, Array(5).fill('noise'));
record('practice A');
// Density conflict fixture and null.
const varied = [...variedGroups.left, ...variedGroups.middle, ...variedGroups.right].map(x => [x, 0]);
for (const probe of author.varied) {
  const fit = dbscan(varied, probe.eps, 3);
  assert.deepEqual(fit.noiseIds, probe.noise_ids, `varied noise at ε=${probe.eps}`);
  assert(sameSets(groupsFromLabels(fit.labels), groupsFromLabels(probe.labels)));
  record('varied radii');
}
const nullFit = dbscan([...variedGroups.left, ...variedGroups.middle, ...variedGroups.rightNull].map(x => [x, 0]), 0.25, 3);
assert.equal(nullFit.clusters, 3); assert.equal(nullFit.noiseIds.length, 0);
assert.equal(intervalFixture(0.75, 0.75).exists, false, 'baseline has no radius');
assert.deepEqual(intervalFixture(0.75, 0.125).interval, [0.125, 0.375]);
assert.deepEqual(intervalFixture(0.75, 0.25).interval, [0.25, 0.375], 'practice F interval');
const repaired = intervalFixture(0.75, 0.25);
const repairedFit = dbscan(repaired.points, 0.25, 3);
assert.equal(repairedFit.clusters, 3); assert.deepEqual(repairedFit.types.slice(8), ['border', 'core', 'core', 'border']);
assert.equal(dbscan(repaired.points, 0.375, 3).clusters, 2, 'the closed boundary merges the dense groups at 0.375');
for (const [offset, spacing] of [[0.625, 0.125], [1, 0.5], [2, 1], [0.875, 0.375]]) {
  const fixture = intervalFixture(offset, spacing);
  if (fixture.exists) {
    const eps = (fixture.lower + fixture.upper) / 2;
    const fit = dbscan(fixture.points, eps, 3);
    assert.equal(fit.clusters, 3, `interval midpoint recovers three groups at offset ${offset}, spacing ${spacing}`);
    assert.equal(fit.noiseIds.length, 0);
  } else {
    for (let eps = 0.125; eps <= 2; eps += 0.125) assert.notEqual(dbscan(fixture.points, eps, 3).clusters === 3 && dbscan(fixture.points, eps, 3).noiseIds.length === 0, true, `no radius should give three complete groups at offset ${offset}, spacing ${spacing}`);
  }
  record('interval fixture agrees with the graph');
}
// Unit change versus metric change (spec L2 and practice E).
const corners = dbscan(cornerPoints, 1, 2);
assert.equal(corners.clusters, 2);
const halved = dbscan(transformRows(cornerPoints, [1, 0.5], 1).points, 1, 2);
assert.equal(halved.clusters, 1, 'halving y merges the pairs');
const extended = [...cornerPoints, fifthCorner];
const original = dbscan(extended, 1, 2);
assert.deepEqual(original.types, supplementary.unit_five_row_contrast.original.labels.map((l, i) => l < 0 ? 'noise' : original.types[i]));
const faulty = transformRows(extended, [1, 100], 1, 100);
const faultyFit = dbscan(faulty.points, faulty.eps, 2);
assert.equal(faultyFit.types[4], 'core', 'one-axis conversion turns the fifth row core');
assert.equal(sameNeighborGraph(original, faultyFit).same, false);
const uniform = transformRows(extended, [100, 100], 1, 100);
assert.equal(sameNeighborGraph(original, dbscan(uniform.points, uniform.eps, 2)).same, true, 'uniform conversion preserves the graph');
const fourFaulty = transformRows(cornerPoints, [1, 100], 1, 100);
assert.equal(sameNeighborGraph(corners, dbscan(fourFaulty.points, fourFaulty.eps, 2)).same, true, 'four corners accidentally survive');
record('units versus metric');
// Rings.
const rings = ringPoints();
const ringFit = dbscan(rings.points, 0.6, 3);
assert.equal(ringFit.coreIds.length, 48); assert.equal(ringFit.clusters, 2);
close(adjustedRandIndex(rings.ring, ringFit.labels), 1, 'DBSCAN ring ARI');
close(adjustedRandIndex(rings.ring, ringKmeans.labels), ringKmeans.ari, 'K-Means ring ARI matches the native record', 1e-9);
rings.points.forEach((point, i) => { const source = supplementary.shape.coordinates[i]; close(point[0], source[0], 'ring x', 1e-12); close(point[1], source[1], 'ring y', 1e-12); });
record('rings');
// Iris: browser DBSCAN, silhouette and ARI versus the seven author probes.
const scaled = standardize(irisRows);
irisRows[0].forEach((_, j) => { const mean = irisRows.reduce((s, r) => s + r[j], 0) / 150; close(scaled.mean[j], mean, 'standardize mean'); });
for (const probe of author.iris) {
  const report = irisReport(probe.eps, probe.min_samples, probe.representation.startsWith('four original') ? 'raw' : 'standardized');
  assert.equal(report.clusters, probe.clusters, `Iris clusters at ε=${probe.eps}, m=${probe.min_samples}, ${probe.representation}`);
  assert.deepEqual(report.noiseIds, probe.noise_ids, 'Iris noise ids');
  assert.deepEqual(report.fit.coreIds, probe.core_ids, 'Iris core ids');
  assert.deepEqual([...report.sizes].sort((a, b) => b - a), [...probe.sizes].sort((a, b) => b - a));
  close(report.coverage, probe.coverage, 'coverage');
  close(report.ariAllRows, probe.ari_all_rows_noise_is_one_label, 'all-row ARI', 1e-9);
  if (probe.ari_assigned !== null) close(report.ariAssigned, probe.ari_assigned, 'assigned ARI', 1e-9);
  if (probe.silhouette_assigned !== null) close(report.silhouetteAssigned, probe.silhouette_assigned, 'assigned silhouette', 1e-9);
  record('Iris settings versus sklearn');
}
const m5 = irisReport(0.5, 5), m10 = irisReport(0.5, 10);
const comparison = compareReports(m5, m10);
assert.equal(comparison.common.length <= 61, true);
assert.equal(comparison.common.length, 61, 'm = 10 retains a subset of the m = 5 rows');
close(comparison.ariSpeciesFirstCommon, 0.8476, 'species ARI of m = 5 on the 61 common rows', 5e-4);
close(comparison.ariSpeciesSecondCommon, 1, 'species ARI of m = 10 on the 61 common rows', 1e-9);
assert.equal(compareReports(irisReport(0.5, 5), irisReport(0.5, 5)).ariSpeciesFirstCommon, irisReport(0.5, 5).ariAssigned, 'common rows of identical snapshots are the assigned rows');
assert.equal(m10.assignedIds.length, 61); assert.equal(m5.assignedIds.length, 116);
assert.equal(irisReport(0.5, 1).noiseCount, 0, 'm = 1 removes noise');
assert.equal(silhouette([[0], [1], [2]], [0, 0, 0]), null, 'silhouette undefined with one cluster');
close(adjustedRandIndex([0, 0, 1, 1], [1, 1, 0, 0]), 1, 'ARI is permutation invariant');
close(silhouette([[0], [1], [4], [6]], [0, 0, 1, 1]), (0.75 + (4 - 1) / 4 * 0 + 0.6 + 0.6 + 0.6666666666666666) / 4 * 0 + 0.6537337662337662, 'four-point silhouette mean', 1e-9);
assert.deepEqual(trailOptics.eps1Labels, trailHdbscan.labels, 'embedded library results agree on the trail');
assert.throws(() => irisReport(3, 5), RangeError); assert.throws(() => dbscan(trailPoints, 0, 4), RangeError); assert.throws(() => intervalFixture(0.5, 0.5), RangeError);

const sources = ['src/learn/data/dbscan-models.js', 'src/learn/data/dbscan-iris-data.js', 'src/learn/components/lesson-labs/DbscanLabs.jsx', 'src/learn/components/lesson-labs/DbscanFigures.jsx', 'src/learn/components/lesson-labs/dbscan-labs.css'];
const hash = file => crypto.createHash('sha256').update(fs.readFileSync(file)).digest('hex');
const evidence = {
  checkedAt: new Date().toISOString(),
  sourceHashes: Object.fromEntries(sources.filter(fs.existsSync).map(file => [file, hash(file)])),
  verifierHash: hash('scripts/verify-dbscan-models.mjs'),
  counts, totalGroupedChecks: Object.values(counts).reduce((sum, value) => sum + value, 0),
  scope: 'Browser DBSCAN against sklearn probes on the trail (five radii, reversal), the density-conflict fixture and null, the analytic radius interval over several offsets/spacings, unit versus metric transformations, the 48-point rings, and all seven Iris settings (types, sizes, coverage, silhouette, ARI).',
  limitations: ['OPTICS and HDBSCAN values are embedded native results, not browser computations.', 'Rendering, interaction and independent review are separate.'],
  passed: true,
};
fs.mkdirSync('docs/teaching/evidence', { recursive: true });
fs.writeFileSync('docs/teaching/evidence/dbscan-models.json', JSON.stringify(evidence, null, 2) + '\n');
console.log(`PASS: ${evidence.totalGroupedChecks} grouped DBSCAN model checks.`);
