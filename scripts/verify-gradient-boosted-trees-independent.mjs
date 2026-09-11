import assert from 'node:assert/strict';
import fs from 'node:fs';
import crypto from 'node:crypto';
import { spawnSync } from 'node:child_process';
import * as model from '../src/learn/data/gradient-boosted-trees-models.js';
import { gradientBoostedTreeExamples as examples } from '../src/learn/data/gradient-boosted-trees-examples.js';

const packet = JSON.parse(fs.readFileSync('docs/teaching/evidence/gradient-boosted-trees-author-review.json'));
for (const file of packet.sourceHashes) assert.equal(crypto.createHash('sha256').update(fs.readFileSync(file.path)).digest('hex'), file.sha256);
const close = (a, b) => assert(Math.abs(a - b) <= 1e-9 * Math.max(1, Math.abs(a), Math.abs(b)), `${a} != ${b}`);
const counts = {};
function passed(name) { counts[name] = (counts[name] || 0) + 1; }
// A finite-population sampling variance oracle, rather than a second subset enumeration.
for (const [keep, draw] of [[1, 1], [1, 4], [2, 1], [2, 3], [3, 2], [3, 3]]) {
  const state = model.gossInvestigation({ keep, draw });
  const values = state.remaining.map(i => state.gradients[i]);
  const r = values.length;
  const mean = values.reduce((a, b) => a + b) / r;
  const variance = values.reduce((a, b) => a + (b - mean) ** 2, 0) / (r - 1);
  const predictedVariance = r ** 2 * (1 - draw / r) * variance / draw;
  close(state.averageSquare - state.average ** 2, predictedVariance);
  close(state.average, state.full);
  passed('finite-population variance law');
}
// Repeated x values remain inseparable, but feature/target affine transforms preserve partitions.
const x = [-3, -3, 0, 2, 4, 4];
const y = [1, -2, 4, 0, 7, 2];
for (const depth of [0, 1, 3]) {
  for (const rate of [0, .7, 1.4]) {
    const original = model.fitBoosting({ x, y, depth, rate, rounds: 4 });
    const shifted = model.fitBoosting({ x: x.map(v => 2 * v + 1), y: y.map(v => -3 * v + 11), depth, rate, rounds: 4 });
    for (const query of [-5, -3, -.4, 2, 3.8, 4, 8]) close(model.predictBoosting(shifted, 2 * query + 1), -3 * model.predictBoosting(original, query) + 11);
    passed('changed affine fit and inference');
  }
}
const native = spawnSync('scratch/lesson-tools/Scripts/python.exe', ['scripts/verify-gradient-boosted-trees-independent.py'], { input: JSON.stringify(examples), encoding: 'utf8', maxBuffer: 4e6 });
assert.equal(native.status, 0, native.stderr + native.stdout);
const output = { checkedAt: new Date().toISOString(), sourceHashes: packet.sourceHashes, authorFrozenAt: packet.frozenAt, counts, native: JSON.parse(native.stdout), limitations: 'Bounded complementary checks, not a rerun of the author suite. No GPU, library performance ranking or arbitrary-range arithmetic claim.' };
fs.mkdirSync('scratch/gradient-boosted-trees-independent', { recursive: true });
fs.writeFileSync('scratch/gradient-boosted-trees-independent/numerical-results.json', JSON.stringify(output, null, 2) + '\n');
console.log(JSON.stringify({ checkedAt: output.checkedAt, counts, native: output.native.counts }));
