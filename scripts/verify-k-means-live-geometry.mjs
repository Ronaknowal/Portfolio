import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { readFileSync, writeFileSync } from 'node:fs';
import { featureGeometry } from '../src/learn/data/k-means-hierarchical-models.js';

// Independent rectangle formula: two vertical columns cost h²; two horizontal
// rows cost 9. The two diagonal pairs cost 9+h², while each singleton/triple
// split costs 2(9+h²)/3, so neither improves on min(9,h²).
const cases = [];
for (const unit of [1, 10]) {
  for (let tick = 1; tick <= 400; tick += 1) {
    const weight = tick / 100;
    const state = featureGeometry(unit, weight);
    const expected = Math.min(9, unit * unit * weight);
    assert.ok(Math.abs(state.sse - expected) <= 1e-10, `Unit ${unit}, weight ${weight}: ${state.sse} versus ${expected}`);
    assert.equal(state.partitions.length, 7);
    assert.ok(state.points.flat().every(Number.isFinite));
    cases.push({ unit, weight, expected, actual: state.sse });
  }
}
assert.deepEqual(featureGeometry(10, 0.01), featureGeometry(1, 1));
for (const weight of [NaN, Infinity, -Infinity, -1, 0, 0.009, 4.001, '1']) {
  assert.throws(() => featureGeometry(1, weight), RangeError);
}
assert.throws(() => featureGeometry(2, 1), RangeError);
const sources = ['src/learn/data/k-means-hierarchical-models.js', 'src/learn/components/lesson-labs/KMeansHierarchicalLabs.jsx', 'scripts/verify-k-means-live-geometry.mjs'];
const evidence = {
  checkedAt: new Date().toISOString(),
  passed: true,
  contract: 'Every displayed .01–4 weight tick for both unit choices remains valid and matches the independent rectangle optimum.',
  cases,
  invalidCases: 9,
  unitRestoration: true,
  sourceHashes: Object.fromEntries(sources.map(path => [path, createHash('sha256').update(readFileSync(path)).digest('hex')])),
};
if (!process.argv.includes('--no-evidence')) writeFileSync('docs/teaching/evidence/k-means-live-geometry.json', `${JSON.stringify(evidence, null, 2)}\n`);
console.log(`Passed: ${cases.length} slider/unit settings, nine invalid inputs, exact unit restoration.`);
