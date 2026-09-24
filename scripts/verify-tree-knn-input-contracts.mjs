// Focused closure of sparse-input findings; unchanged native/UI suites are reused.
import fs from 'node:fs';
import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import * as tree from '../src/learn/data/decision-tree-models.js';
import * as knn from '../src/learn/data/knn-models.js';
import { knnExamples } from '../src/learn/data/knn-examples.js';

const archive = 'docs/teaching/archive/tree-knn-input-amendment';
const oldModule = filename => import('data:text/javascript;base64,' + fs.readFileSync(`${archive}/${filename}`).toString('base64'));
const oldTree = await oldModule('decision-tree-models.js');
const oldKnn = await oldModule('knn-models.js');
const cases = [];
const rejects = (name, run) => { assert.throws(run, RangeError); cases.push({ kind: 'rejected unsupported input', name }); };
const inherited = values => {
  const array = Array(values.length);
  const prototype = Object.create(Array.prototype);
  values.forEach((value, index) => { prototype[index] = value; });
  Object.setPrototypeOf(array, prototype);
  return array;
};

for (const [name, values] of [['empty slots', Array(2)], ['one missing slot', [1, ,]], ['inherited coordinates', inherited([1, 2])]]) {
  rejects(`tree query: ${name}`, () => tree.treePrediction(tree.growTree(), values));
  rejects(`tree row features: ${name}`, () => tree.growTree([{ id: 'A', features: values, label: 1 }]));
  rejects(`KNN query: ${name}`, () => knn.neighborReport({ query: values }));
  rejects(`KD query: ${name}`, () => knn.kdSearch(values));
}
for (const values of [Array(2), inherited([0, 1])]) {
  rejects('tree impurity rejects absent own labels', () => tree.impurity(values));
  rejects('tree feature selection rejects absent own indices', () => tree.splitCandidates(tree.inspectionRows, { features: values }));
}
for (const rows of [Array(1), inherited([tree.inspectionRows[0]])]) rejects('tree requires own row entries', () => tree.growTree(rows));
for (const candidates of [Array(2), inherited([knn.neighborRows[0]]), [knn.neighborRows[0], knn.neighborRows[0]], null, {}]) {
  rejects('KNN candidate collection must be a dense distinct fixture subset', () => knn.neighborReport({ candidates, k: 1 }));
  rejects('KD construction requires its declared fixture subset', () => knn.buildKdTree(candidates));
}
assert.equal(knn.buildKdTree([]), null);

for (const maxDepth of [0, 1, 3, 6]) {
  const current = tree.growTree(tree.inspectionRows, { maxDepth });
  assert.deepEqual(current, oldTree.growTree(oldTree.inspectionRows, { maxDepth }));
  for (const query of [[1, 1], [2.5, 2], [6, 4]]) assert.deepEqual(tree.treePrediction(current, query), oldTree.treePrediction(current, query));
  cases.push({ kind: 'unchanged valid tree behavior', maxDepth });
}
for (const metric of ['euclidean', 'manhattan', 'maximum', 'cosine']) {
  for (const weights of ['uniform', 'distance']) {
    const current = knn.neighborReport({ metric, weights });
    const previous = oldKnn.neighborReport({ metric, weights });
    if (metric === 'cosine') {
      assert.equal(current.label, previous.label);
      assert.deepEqual(current.neighbors.map(row => row.id), previous.neighbors.map(row => row.id));
      current.ranked.forEach((row, index) => assert.ok(Math.abs(row.distance - previous.ranked[index].distance) < 1e-14));
      current.votes.forEach((row, index) => assert.ok(Math.abs(row.probability - previous.votes[index].probability) < 1e-12));
    } else assert.deepEqual(current, previous);
    cases.push({ kind: 'unchanged dense neighborhood behavior', metric, weights });
  }
}
for (const query of [[1.3, 2.1], [5.1, 4.9], [0, 0]]) assert.deepEqual(knn.kdSearch(query), oldKnn.kdSearch(query));
cases.push({ kind: 'unchanged exact-search fixture, three queries' });
for (const scale of [1e-300, 1e-200, 1, 10000]) {
  assert.equal(knn.metricDistance([scale, 0], [scale, 0], 'cosine'), 0);
  assert.ok(Math.abs(knn.metricDistance([scale, 0], [scale, scale], 'cosine') - (1 - 1 / Math.SQRT2)) < 1e-14);
  assert.equal(knn.metricDistance([scale, 0], [-scale, 0], 'cosine'), 2);
  cases.push({ kind: 'scale-invariant cosine with separately normalized vectors', scale });
}

const sources = ['src/learn/data/decision-tree-models.js', 'src/learn/data/knn-models.js', 'src/learn/data/knn-examples.js'];
const sourceHashes = Object.fromEntries(sources.map(filename => [filename, createHash('sha256').update(fs.readFileSync(filename)).digest('hex')]));
fs.mkdirSync('scratch/tree-knn-input-contracts', { recursive: true });
fs.writeFileSync('scratch/tree-knn-input-contracts/checks.json', JSON.stringify({ checkedAt: new Date().toISOString(), sourceHashes, cases, archive }, null, 2) + '\n');
fs.writeFileSync('scratch/tree-knn-input-contracts/knn-programs.json', JSON.stringify(knnExamples.filter(example => ['scratch-estimator', 'kd-tree'].includes(example.id)), null, 2) + '\n');
console.log(`PASS: ${cases.length} focused input/conservation cases; unaffected behavior reused.`);
