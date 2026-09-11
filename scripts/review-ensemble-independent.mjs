// Complementary data/representation invariances; reuses the author native suite.
import assert from 'node:assert/strict';
import fs from 'node:fs';
import { createHash } from 'node:crypto';
import { votingState, fitBootstrapStump, oofOwnershipState, calibratedAverageLaw } from '../src/learn/data/ensemble-methods-models.js';

const close = (actual, expected) => assert.ok(Math.abs(actual - expected) < 1e-11, `${actual} != ${expected}`);
const cases = [];
for (const probabilities of [[.8, .4, .3], [.5, .5, .5], [0, 1, .75], [.1, .9, .2]]) {
  const weights = [1, 2, 3];
  const original = votingState(probabilities, weights);
  const scaled = votingState(probabilities, weights.map(value => value * 3));
  const reordered = votingState([probabilities[2], probabilities[0], probabilities[1]], [3, 1, 2]);
  for (const candidate of [scaled, reordered]) {
    close(candidate.probability, original.probability);
    close(candidate.ballotMass, original.ballotMass);
    assert.equal(candidate.hardClass, original.hardClass);
    assert.equal(candidate.softClass, original.softClass);
  }
  cases.push({ kind: 'member permutation and common weight scale preserve both votes', probabilities });
}
for (const draws of [[0, 0, 2, 3, 3, 5], [1, 2, 2, 4, 4, 5], [0, 1, 2, 3, 4, 5], [2, 2, 2]]) {
  const original = fitBootstrapStump(draws);
  const reordered = fitBootstrapStump([...draws].reverse());
  const repeated = fitBootstrapStump([...draws, ...draws]);
  for (const candidate of [reordered, repeated]) {
    assert.equal(candidate.threshold, original.threshold);
    candidate.predictions.forEach((value, index) => close(value, original.predictions[index]));
  }
  close(repeated.sse, original.sse * 2);
  cases.push({ kind: 'draw order and uniform multiplicity preserve a fitted stump', draws });
}
const complete = oofOwnershipState('honest', 3, 2.5);
for (const query of [0, .25, 1.5, 4.75, 5]) {
  const changed = oofOwnershipState('honest', 3, query);
  assert.deepEqual(changed.matrix, complete.matrix);
  close(changed.weightNearest, complete.weightNearest);
  close(changed.trainMse, complete.trainMse);
  cases.push({ kind: 'new query cannot refit the OOF combination', query });
}
for (const completed of [0, 1, 2]) {
  const partial = oofOwnershipState('honest', completed);
  assert.equal(partial.weightNearest, null);
  assert.equal(partial.ensemble, null);
  assert.equal(partial.matrix.filter(row => row.every(value => value === null)).length, 6 - 2 * completed);
  cases.push({ kind: 'incomplete OOF features cannot produce a trained combiner', completed });
}
const law = calibratedAverageLaw();
for (const key of ['forecastA', 'forecastB']) {
  for (const forecast of [.25, .75]) {
    const rows = law.filter(row => row[key] === forecast);
    close(rows.reduce((sum, row) => sum + row.mass * row.trueChance, 0) / rows.reduce((sum, row) => sum + row.mass, 0), forecast);
  }
  cases.push({ kind: 'exact law verifies individual calibration by conditioning', key });
}
const source = 'src/learn/data/ensemble-methods-models.js';
fs.mkdirSync('scratch/ensemble-independent', { recursive: true });
fs.writeFileSync('scratch/ensemble-independent/complementary-checks.json', JSON.stringify({ checkedAt: new Date().toISOString(), source, sha256: createHash('sha256').update(fs.readFileSync(source)).digest('hex'), cases }, null, 2) + '\n');
console.log(`PASS: ${cases.length} complementary ensemble cases.`);
