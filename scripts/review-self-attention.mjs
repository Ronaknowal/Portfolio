// Complementary review: new inputs, axis invariants and independent finite differences.
import assert from 'node:assert/strict';
import fs from 'node:fs';
import { createHash } from 'node:crypto';
import { attentionRead, multiHeadRead, attentionTrajectory, mixtures, attentionStorage } from '../src/learn/data/self-attention-models.js';

let seed = 7139;
const random = () => ((seed = (Math.imul(seed, 1664525) + 1013904223) >>> 0) / 2 ** 32);
const vector = n => Array.from({ length: n }, () => 4 * random() - 2);
const flat = x => x.flat(Infinity);
const close = (a, b, tol = 1e-9) => {
  const left = flat(a), right = flat(b);
  assert.equal(left.length, right.length);
  left.forEach((v, i) => assert.ok(Number.isFinite(v) && Math.abs(v - right[i]) < tol, `${v} != ${right[i]}`));
};
const scalarRead = (q, keys, values, allowed) => {
  // Independent unshifted expression, safe for these bounded small probes.
  const masses = keys.map((key, j) => allowed[j] ? Math.exp(key.reduce((s, x, d) => s + x * q[d], 0) / Math.sqrt(q.length)) : 0);
  const total = masses.reduce((a, b) => a + b, 0);
  return values[0].map((_, d) => masses.reduce((s, a, j) => s + a * values[j][d], 0) / total);
};
for (let trial = 0; trial < 80; trial++) {
  const n = 2 + trial % 7, d = 1 + trial % 4, q = vector(d), keys = Array.from({ length: n }, () => vector(d));
  const values = Array.from({ length: n }, () => vector(3)), allowed = keys.map((_, i) => i === 0 || random() > .3);
  const read = attentionRead(q, keys, values, { allowed });
  close(read.output, scalarRead(q, keys, values, allowed));
  close(read.output, attentionRead(q, keys, values, { allowed, offset: 1000 }).output);
  const reversed = attentionRead(q, [...keys].reverse(), [...values].reverse(), { allowed: [...allowed].reverse() });
  close(read.output, reversed.output);
  close(read.weights, [...reversed.weights].reverse());
  const target = trial % n, h = 1e-5;
  const changed = sign => values.map((v, j) => v.map((x, a) => x + (j === target && a === 1 ? sign * h : 0)));
  const plus = attentionRead(q, keys, changed(1), { allowed }).output[1];
  const minus = attentionRead(q, keys, changed(-1), { allowed }).output[1];
  close([(plus - minus) / (2 * h)], [read.weights[target]], 1e-9);
  for (let axis = 0; axis < d; axis++) {
    const derivative = keys.reduce((sum, key, j) => sum + read.weights[j] * (values[j][0] - read.output[0]) * key[axis] / Math.sqrt(d), 0);
    const moved = sign => q.map((x, a) => x + (a === axis ? sign * h : 0));
    const difference = (attentionRead(moved(1), keys, values, { allowed }).output[0] - attentionRead(moved(-1), keys, values, { allowed }).output[0]) / (2 * h);
    close([difference], [derivative], 1e-8);
  }
}
assert.equal(attentionRead([1], [[1]], [[4]], { allowed: [false] }).output, null);

for (let trial = 0; trial < 30; trial++) {
  const x = Array.from({ length: 4 }, () => vector(4));
  const maps = Object.fromEntries(['query', 'key', 'value', 'output'].map(name => [name, Array.from({ length: 4 }, () => vector(4))]));
  const y = multiHeadRead(x, maps, 2).output;
  // Relabel the heads, then permute the corresponding output-map input columns.
  const order = [2, 3, 0, 1];
  const permuted = Object.fromEntries(['query', 'key', 'value'].map(name => [name, order.map(i => maps[name][i])]));
  permuted.output = maps.output.map(row => order.map(i => row[i]));
  close(y, multiHeadRead(x, permuted, 2).output);
  close([...y].reverse(), multiHeadRead([...x].reverse(), maps, 2).output);
}
const assetPath = 'public/learn-assets/self-attention-multi-head-attention/trajectory-model.json';
const asset = JSON.parse(fs.readFileSync(assetPath));
const original = attentionTrajectory(asset.state_dict, asset.points);
for (let trial = 0; trial < 12; trial++) {
  const order = asset.points.map((_, i) => i).sort(() => random() - .5);
  const shuffled = attentionTrajectory(asset.state_dict, order.map(i => asset.points[i]));
  close(original.logits, shuffled.logits, 1e-9);
  // Exact excluded-padding invariance with arbitrary new points, not the author's fixture.
  const padded = [...asset.points, ...Array.from({ length: trial + 1 }, () => [random(), random()])];
  close(original.logits, attentionTrajectory(asset.state_dict, padded, padded.map((_, i) => i < 45)).logits, 1e-9);
}
close(original.logits, attentionTrajectory(asset.state_dict, asset.points.flatMap(p => [p, p])).logits, 1e-9);
assert.equal(attentionTrajectory(asset.state_dict, asset.points, asset.points.map(() => false)).error,
  'Keep at least one valid point for attention and pooling.');
const result = mixtures([.6, .3, .1], [.2, .3, .5], [4, 2, 4]);
close([result.outputA, result.outputB], [3.4, 3.4]);
assert.equal(mixtures([0, 0], [1, 1], [3, 4]).error, 'Each weight row needs a positive total.');
const count = attentionStorage(), doubled = attentionStorage({ length: 4096 });
assert.equal(count.matrixBytes, 128 * 2 ** 20);
assert.equal(count.cacheBytes, 96 * 2 ** 20);
assert.equal(doubled.matrixBytes, 4 * count.matrixBytes);
assert.equal(doubled.cacheBytes, 2 * count.cacheBytes);
const files = ['scripts/review-self-attention.mjs', 'src/learn/data/self-attention-models.js', assetPath];
const output = 'docs/teaching/deep-learning-completion/self-attention-multi-head-attention/independent-model-checks.json';
fs.writeFileSync(output, JSON.stringify({ passed: true, reviewer: '/root', checkedAt: new Date().toISOString(), checks: [
  '80 independently enumerated attention reads: random dimensions, masks, signed values; common score shift and donor reorder invariants',
  '80 value finite differences and all query-coordinate finite differences against the stated analytic gradient',
  '30 random two-head map relabelings with corresponding output columns; row permutation equivariance',
  '12 new frozen-model donor reorderings and arbitrary masked-padding cases, complete duplicate-point invariance, empty-pool rejection; native comparison reused from author evidence',
  'Empty legal set, zero-mass mixtures, distinct distributions with equal mixtures; independently derived storage units and length scaling'
], sourceHashes: Object.fromEntries(files.map(file => [file, createHash('sha256').update(fs.readFileSync(file)).digest('hex')])) }, null, 2) + '\n');
console.log(`PASS: ${output}`);
