// Complementary review: invariances and information-preserving transformations.
// Reuse the unchanged author's broader numerical and browser evidence.
import assert from 'node:assert/strict';
import fs from 'node:fs';
import { createHash } from 'node:crypto';
import {
  multioutputMetrics, multioutputTruth, multioutputPredictions,
  labelJointDecisions, pooledLabelAssociation, sharedOutputStump,
  sharedFeatureShrinkage,
} from '../src/learn/data/multioutput-models.js';

const close = (actual, expected) => assert.ok(Math.abs(actual - expected) < 1e-11, `${actual} != ${expected}`);
const cases = [];
const hash = filename => createHash('sha256').update(fs.readFileSync(filename)).digest('hex');
const author = JSON.parse(fs.readFileSync('docs/teaching/evidence/multioutput-author-review.json', 'utf8'));
for (const [filename, expected] of Object.entries(author.sourceHashes)) assert.equal(hash(filename), expected);

const baseline = multioutputMetrics();
const aggregateKeys = ['tp', 'fp', 'fn', 'tn', 'hamming', 'f1', 'macroF1', 'observedSampleF1', 'subsetAccuracy'];
for (const order of [[3, 1, 0, 2], [2, 3, 1, 0]]) {
  const reordered = multioutputMetrics(order.map(i => multioutputTruth[i]), order.map(i => multioutputPredictions[i]));
  for (const key of aggregateKeys) close(reordered[key], baseline[key]);
  cases.push({ kind: 'whole-row permutation preserves aggregate metrics', order });
}
for (const order of [[2, 0, 1], [1, 2, 0]]) {
  const reorder = rows => rows.map(row => order.map(column => row[column]));
  const renamed = multioutputMetrics(reorder(multioutputTruth), reorder(multioutputPredictions));
  for (const key of aggregateKeys) close(renamed[key], baseline[key]);
  cases.push({ kind: 'consistent label permutation preserves aggregate metrics', order });
}
const unobserved = multioutputMetrics([[null, null], [null, null]], [[0, 1], [1, 0]]);
assert.equal(unobserved.observed, 0);
for (const key of ['f1', 'hamming', 'macroF1', 'observedSampleF1', 'subsetAccuracy']) assert.equal(unobserved[key], null);
cases.push({ kind: 'fully unknown annotations remain undefined, not perfect or zero loss' });

for (const counts of [[6, 5, 1, 8], [4, 1, 2, 3], [0, 0, 0, 1], [2, 2, 2, 2]]) {
  const original = labelJointDecisions(counts);
  const replicated = labelJointDecisions(counts.map(value => 2 * value));
  assert.deepEqual(replicated.mass, original.mass);
  assert.deepEqual(replicated.modes, original.modes);
  assert.deepEqual(replicated.greedy, original.greedy);
  assert.deepEqual(replicated.marginalDecision, original.marginalDecision);
  const selected = original.candidates.find(candidate => candidate.prediction.every((bit, i) => bit === original.marginalDecision[i]));
  close(selected.hammingRisk, Math.min(...original.candidates.map(candidate => candidate.hammingRisk)));
  cases.push({ kind: 'replicated counts preserve actions and marginal action minimizes Hamming risk', counts });
}
for (const [low, high, highShare] of [[.2, .7, .3], [.8, .1, .6], [.4, .4, .5], [0, 1, .25]]) {
  const mixture = pooledLabelAssociation({ low, high, highShare });
  close(mixture.covariance, highShare * (1 - highShare) * (high - low) ** 2);
  cases.push({ kind: 'pooled covariance equals variance of conditional mean', low, high, highShare });
}
for (const [temperatureShift, energyShift] of [[10, -50], [-7, 230]]) {
  const original = sharedOutputStump({ energyScale: 10 });
  const shifted = sharedOutputStump({ energyScale: 10, temperature: [0, 0, 2, 2].map(x => x + temperatureShift), energy: [0, 100, 100, 100].map(x => x + energyShift) });
  assert.equal(shifted.shared.threshold, original.shared.threshold);
  shifted.candidates.forEach((candidate, index) => close(candidate.scaledSse, original.candidates[index].scaledSse));
  cases.push({ kind: 'target translation changes means but preserves split objective', temperatureShift, energyShift });
}
for (const angle of [Math.PI / 6, Math.PI / 3, Math.PI / 2]) {
  const rotate = ([x, y]) => [Math.cos(angle) * x - Math.sin(angle) * y, Math.sin(angle) * x + Math.cos(angle) * y];
  const [first, second] = rotate([3, 4]);
  const rotated = sharedFeatureShrinkage({ first, second, penalty: 2 });
  rotate([1.8, 2.4]).forEach((value, index) => close(rotated.grouped[index], value));
  cases.push({ kind: 'group norm shrinkage is orthogonally equivariant', angle });
}

fs.mkdirSync('scratch/multioutput-independent', { recursive: true });
fs.writeFileSync('scratch/multioutput-independent/complementary-checks.json', JSON.stringify({
  checkedAt: new Date().toISOString(), sourceHashes: author.sourceHashes, cases,
  scope: 'Source identity plus complementary metamorphic checks; unchanged author execution and browser evidence reused.',
}, null, 2) + '\n');
console.log(`PASS: ${cases.length} complementary multi-output cases; all seven author source hashes match.`);
