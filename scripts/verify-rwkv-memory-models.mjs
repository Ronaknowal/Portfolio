import fs from 'node:fs';
import assert from 'node:assert/strict';
import { transform } from 'esbuild';
import { summaryDefaults, kernelSummary, weightedMemory, deltaMemory, gooseUpdate, trajectoryMemoryForward } from '../src/learn/data/rwkv-memory-models.js';
const id = 'rwkv-linear-attention-models', folder = `docs/teaching/deep-learning-completion/${id}`;
const checks = [];
const near = (a, b, tolerance = 1e-10) => assert.ok(Math.abs(a - b) <= tolerance, `${a} != ${b} (tolerance ${tolerance})`);
for (const chunk of [1, 2, 3, 4, 8]) {
  const rows = kernelSummary({ ...summaryDefaults(), chunk });
  rows.forEach((r, i) => { near(r.output, [3, 4 / 3, 10 / 3, 2.8][i]); near(r.output, r.chunkOutput); });
}
const zero = summaryDefaults(); zero.queries[2] = [0, 0]; assert.equal(kernelSummary(zero)[2].output, null);
const changed = summaryDefaults(); changed.values[3] = 11;
kernelSummary(changed).slice(0, 3).forEach((r, i) => near(r.output, kernelSummary(summaryDefaults())[i].output));
checks.push({ name: 'Chosen kernel: fresh exact answers, 5 chunk sizes, short remainder, zero denominator, future edit', passed: true });
const keys = [0, Math.log(2), 0, Math.log(4)], values = [2, 8, -1, 5];
for (const retention of [.01, .5, 1]) for (const offset of [-1000, 0, 1000]) {
  weightedMemory(keys, values, retention, Math.log(2), offset).forEach(r => near(r.output, r.direct));
}
const bonus = weightedMemory(keys, values), noBonus = weightedMemory(keys, values, .5, 0);
bonus.forEach((r, i) => { near(r.a, noBonus[i].a); near(r.b, noBonus[i].b); near(r.p, noBonus[i].p); });
weightedMemory(keys, [7, 7, 7, 7]).forEach(r => near(r.output, 7));
checks.push({ name: 'RWKV independent enumeration, large shifts, retention=1, bonus/state invariant, constant values', passed: true });
assert.deepEqual(deltaMemory([[1, 0], [0, 1], [1, 0]], [2, 7, 5], 1).at(-1).delta, [5, 7]);
const interference = deltaMemory([[1, 0], [.6, .8], [1, 0]], [2, 7, 5], 1).at(-1).delta;
near(interference[0] * .6 + interference[1] * .8, 6.712);
assert.deepEqual(deltaMemory([[1, 0]], [3], 0, [6, -2])[0].delta, [6, -2]);
gooseUpdate([.6, .8]).next.flat().forEach((v, i) => near(v, [4.152, 5.212, .552, 2.412][i]));
checks.push({ name: 'Orthogonal/correlated delta writes, nonempty zero-rate state, generalized matrix correction', passed: true });
const data = JSON.parse(fs.readFileSync(`public/learn-assets/${id}/trajectory-models.json`));
const fixtures = JSON.parse(fs.readFileSync(`${folder}/native-fixtures.json`));
assert.equal(data.specimens.length, 50); assert.equal(fixtures.length, 16);
let maximumPortError = 0;
for (const fixture of fixtures) {
  let points = data.specimens.find(row => row.sourceRow === fixture.sourceRow).points.map(p => [...p]);
  if (fixture.variant === 'reflect') points[fixture.cut][0] = 1 - points[fixture.cut][0];
  if (fixture.variant === 'reverse') points.reverse();
  const result = trajectoryMemoryForward(data.models[fixture.kind], points, fixture.kind, { cut: fixture.cut, reset: fixture.variant === 'reset' });
  result.logits.forEach((v, i) => { const error = Math.abs(v - fixture.logits[i]); maximumPortError = Math.max(maximumPortError, error); near(v, fixture.logits[i], 2e-5); });
}
checks.push({ name: '16 native PyTorch fixtures: both model kinds, 2 real validation rows, changed/carry/reset/reverse inputs', passed: true, maximumPortError });
for (const file of [`src/learn/data/topics/${id}.jsx`, 'src/learn/components/lesson-labs/RwkvMemoryLabs.jsx', 'src/learn/components/lesson-labs/RwkvMechanismFigures.jsx']) await transform(fs.readFileSync(file, 'utf8'), { loader: 'jsx' });
checks.push({ name: 'Topic and component JSX parse', passed: true });
fs.writeFileSync(`${folder}/model-checks.json`, JSON.stringify({ passed: true, checks, maximumPortError }, null, 2) + '\n');
console.log(JSON.stringify({ passed: true, checks: checks.length, nativeFixtures: fixtures.length, maximumPortError }));
