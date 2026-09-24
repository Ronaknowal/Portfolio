import assert from 'node:assert/strict';
import { mkdirSync, writeFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { spawnSync } from 'node:child_process';
import { einsumExamples } from '../src/learn/data/einsum-examples.js';
import { inspectContraction, attentionRow, attentionFixtures, basisChange, basisPresets, contractionOrders } from '../src/learn/data/einsum-models.js';

const directory = resolve('scratch/einsum-verification');
mkdirSync(directory, { recursive: true });
const python = process.env.LESSON_PYTHON || resolve('scratch/lesson-tools/Scripts/python.exe');
const env = { ...process.env, PYTHONIOENCODING: 'utf-8' };
for (const [key, example] of Object.entries(einsumExamples)) {
  const file = resolve(directory, key + '.py');
  writeFileSync(file, example.code);
  const run = spawnSync(python, ['-I', file], { encoding: 'utf8', env, timeout: 20000 });
  assert.equal(run.status, 0, key + ': ' + run.stderr);
  assert.equal(run.stdout.replaceAll('\r\n', '\n').trim(), example.expected.trim(), key);
}
const cases = { contractions: [], attention: [], bases: [], orders: [] };
const matrix = (rows, columns, shift = 0) => Array.from({ length: rows }, (_, i) => Array.from({ length: columns }, (_, j) => (i * 3 + j * 5 + shift) % 9 - 4));
function save(expression, operands) {
  cases.contractions.push({ expression, operands, result: inspectContraction(expression, operands) });
}
for (let m = 1; m <= 4; m++) {
  const vector = Array.from({ length: m }, (_, i) => i - 2);
  for (const expression of ['i->', 'i->i']) save(expression, [vector]);
  for (const expression of ['ii->', 'ii->i']) save(expression, [matrix(m, m)]);
  for (let n = 1; n <= 4; n++) {
    const A = matrix(m, n);
    for (const expression of ['ij->', 'ij->i', 'ij->j', 'ij->ij', 'ij->ji']) save(expression, [A]);
    for (const expression of ['ij,ij->', 'ij,ij->ij', 'ij,ij->i', 'ij,ij->j']) save(expression, [A, matrix(m, n, 2)]);
    save('i,j->ij', [vector, Array.from({ length: n }, (_, i) => 3 - i)]);
    for (let k = 1; k <= 4; k++) {
      for (const expression of ['ik,kj->ij', 'ik,kj->ji', 'ik,kj->ikj', 'ik,kj->', 'ik,kj->i']) save(expression, [matrix(m, k), matrix(k, n, 1)]);
    }
    save('bi,bi->b', [matrix(m, n), matrix(1, n)]);
    save('bi,bi->b', [matrix(1, n), matrix(m, n)]);
    save('ii,i->i', [matrix(1, 1), vector]);
  }
}
for (let batch = 0; batch < 2; batch++) for (let query = 0; query < 2; query++) {
  for (let mask = 1; mask < 8; mask++) for (const scaled of [false, true]) {
    const allowed = Array.from({ length: 3 }, (_, key) => Boolean(mask & (1 << key)));
    cases.attention.push({ fixture: attentionFixtures[batch], query, allowed, scaled, result: attentionRow(batch, query, allowed, scaled) });
  }
}
for (const basis of Object.keys(basisPresets)) for (let i = -4; i <= 4; i++) for (let j = -4; j <= 4; j++) {
  cases.bases.push({ vector: [i, j], result: basisChange(basis, [i, j]) });
}
for (let a = 1; a <= 6; a++) for (let b = 1; b <= 6; b++) for (let c = 1; c <= 6; c++) for (let d = 1; d <= 6; d++) {
  cases.orders.push({ dimensions: [a, b, c, d], result: contractionOrders([a, b, c, d]) });
}
const invalid = [
  () => inspectContraction('ij', [[[1, 2]]]),
  () => inspectContraction('ij->ii', [[[1, 2]]]),
  () => inspectContraction('ij->k', [[[1, 2]]]),
  () => inspectContraction('i->i', [[[1, 2]]]),
  () => inspectContraction('ii->i', [[[1, 2]]]),
  () => inspectContraction('i,i->', [[1, 2], [1, 2, 3]]),
  () => inspectContraction('ij->ij', [[[1], [2, 3]]]),
  () => inspectContraction('i->i', [[NaN]]),
  () => inspectContraction('i,i->', [[1]]),
  () => attentionRow(0, 0, [false, false, false]),
  () => attentionRow(2, 0, [true, true, true]),
  () => attentionRow(0, 3, [true, true, true]),
  () => attentionRow(0, 0, [1, 1, 1]),
  () => basisChange('singular', [1, 2]),
  () => basisChange('shear', [Infinity, 2]),
  ...[[1, 2, 3], [0, 2, 3, 4], [1, 2.5, 3, 4], [101, 2, 3, 4], [NaN, 2, 3, 4]].map(values => () => contractionOrders(values)),
];
invalid.forEach(action => assert.throws(action));
writeFileSync(resolve(directory, 'model-cases.json'), JSON.stringify(cases));
const native = spawnSync(python, ['-I', resolve('scripts/verify-einsum-native.py'), directory], { encoding: 'utf8', env, timeout: 60000 });
assert.equal(native.status, 0, native.stderr || native.stdout);
const evidence = { date: new Date().toISOString(), programs: Object.keys(einsumExamples).length, invalidContracts: invalid.length, ...JSON.parse(native.stdout) };
writeFileSync(resolve(directory, 'results.json'), JSON.stringify(evidence, null, 2));
console.log(JSON.stringify(evidence, null, 2));

