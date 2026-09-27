// Complementary independent checks: fresh inputs and preserved-information laws.
// Does not certify the authored figures, reading flow or rendered interaction.
import fs from 'node:fs';
import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { kernelSummary, weightedMemory, deltaMemory, gooseUpdate, trajectoryMemoryForward } from '../src/learn/data/rwkv-memory-models.js';

const directory = 'docs/teaching/deep-learning-completion/rwkv-linear-attention-models';
const checks = [];
const near = (actual, expected, tolerance = 2e-10) => assert.ok(Math.abs(actual - expected) < tolerance, `${actual} versus ${expected}`);
let seed = 731;
const random = () => ((seed = (Math.imul(seed, 1664525) + 1013904223) >>> 0) / 2 ** 32);
const dot = (a, b) => a.reduce((sum, x, i) => sum + x * b[i], 0);

for (let trial = 0; trial < 60; trial++) {
  const length = 1 + trial % 12;
  const input = {
    queries: Array.from({ length }, () => [random() * 4, random() * 4]),
    keys: Array.from({ length }, () => [random() * 4, random() * 4]),
    values: Array.from({ length }, () => random() * 24 - 12),
    chunk: 1 + trial % 12,
  };
  const rows = kernelSummary(input);
  rows.forEach((row, t) => {
    const weights = input.keys.slice(0, t + 1).map(k => dot(input.queries[t], k));
    const total = weights.reduce((a, b) => a + b, 0);
    const answer = weights.reduce((sum, w, i) => sum + w / total * input.values[i], 0);
    near(row.output, answer); near(row.chunkOutput, answer);
  });
  const prefix = rows.slice(0, -1).map(row => row.output);
  input.keys[length - 1] = [0, 0]; input.values[length - 1] = -12;
  kernelSummary(input).slice(0, -1).forEach((row, i) => near(row.output, prefix[i]));
  input.queries[length - 1] = [0, 0];
  assert.equal(kernelSummary(input).at(-1).output, null);
}
checks.push({ name: '60 fresh positive-kernel sequences: independently normalized pair sums, unequal chunks, future-write invariance and undefined zero reads', passed: true });

for (let trial = 0; trial < 60; trial++) {
  const keys = Array.from({ length: 1 + trial % 12 }, () => random() * 16 - 8);
  const values = keys.map(() => random() * 24 - 12);
  const retention = trial % 3 === 0 ? 1 : .01 + .99 * random(), bonus = random() * 8 - 4;
  for (const offset of [-1000, 0, 1000]) {
    weightedMemory(keys, values, retention, bonus, offset).forEach((row, t) => {
      // Enumerate historical contribution ages; a current bonus never survives storage.
      const scores = keys.slice(0, t + 1).map((k, i) => k + (i === t ? bonus : (t - i - 1) * Math.log(retention)));
      const maximum = Math.max(...scores), weights = scores.map(s => Math.exp(s - maximum));
      near(row.output, dot(weights, values) / weights.reduce((a, b) => a + b, 0));
    });
  }
}
checks.push({ name: '180 fresh stable RWKV paths against enumerated historical weights, signed values, no forgetting and large common key shifts', passed: true });

for (let trial = 0; trial < 30; trial++) {
  const initial = [random() * 6 - 3, random() * 6 - 3], key = [random() * 4 - 2, random() * 4 - 2];
  const target = random() * 10 - 5, rate = random(), epsilon = 1e-5;
  const loss = state => .5 * (dot(state, key) - target) ** 2;
  const gradient = initial.map((_, i) => {
    const plus = [...initial], minus = [...initial]; plus[i] += epsilon; minus[i] -= epsilon;
    return (loss(plus) - loss(minus)) / (2 * epsilon);
  });
  deltaMemory([key], [target], rate, initial)[0].delta.forEach((value, i) => near(value, initial[i] - rate * gradient[i], 2e-8));
  const angle = random() * 2 * Math.PI, removal = [Math.cos(angle), Math.sin(angle)];
  const transition = [[.8 - .6 * removal[0] ** 2, -.2 * removal[0] * removal[1]], [-.6 * removal[1] * removal[0], .9 - .2 * removal[1] ** 2]];
  const initialMatrix = [[2, 7], [-1, 3]], write = [[5, 0], [2, 0]];
  const expected = initialMatrix.map((row, i) => [0, 1].map(j => row[0] * transition[0][j] + row[1] * transition[1][j] + write[i][j]));
  gooseUpdate(removal).next.flat().forEach((value, i) => near(value, expected.flat()[i]));
}
checks.push({ name: '30 nonunit-key delta gradients by finite differences and arbitrary-direction full-matrix RWKV-7 products', passed: true });

const asset = 'public/learn-assets/rwkv-linear-attention-models/trajectory-models.json';
const data = JSON.parse(fs.readFileSync(asset));
const original = JSON.parse(fs.readFileSync('docs/teaching/drafts/rwkv-linear-attention-models/trajectory-results.json'));
assert.deepEqual(data.specimens.map(row => row.sourceRow).sort((a, b) => a - b), [...original.roles.validation].sort((a, b) => a - b));
for (const kind of Object.keys(data.models)) for (const specimen of data.specimens.slice(0, 12)) {
  const unchanged = trajectoryMemoryForward(data.models[kind], specimen.points, kind);
  const changed = specimen.points.map((point, i) => i < 31 ? [...point] : [1 - point[0], 1 - point[1]]);
  const result = trajectoryMemoryForward(data.models[kind], changed, kind);
  assert.deepEqual(result.features.slice(0, 31), unchanged.features.slice(0, 31));
  near(unchanged.probabilities.reduce((a, b) => a + b, 0), 1);
  assert.ok(unchanged.probabilities.every(value => Number.isFinite(value) && value >= 0));
}
checks.push({ name: 'Published corpus equals the 50 validation IDs; both fitted models preserve all prefix features under later path edits on 12 specimens', passed: true });

const sourceFiles = ['src/learn/data/rwkv-memory-models.js', asset, 'scripts/review-rwkv-memory.mjs'];
const sourceHashes = Object.fromEntries(sourceFiles.map(file => [file, createHash('sha256').update(fs.readFileSync(file)).digest('hex')]));
fs.writeFileSync(`${directory}/independent-model-checks.json`, JSON.stringify({ passed: true, checks, sourceFiles, sourceHashes, scope: 'Complementary operator and causal-prefix review only; figures/browser reviewed separately.' }, null, 2) + '\n');
console.log(JSON.stringify({ passed: true, checkGroups: checks.length }));
