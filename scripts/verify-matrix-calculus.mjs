import assert from 'node:assert/strict';
import { mkdirSync, writeFileSync } from 'node:fs';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { spawnSync } from 'node:child_process';
import { matrixCalculusExamples } from '../src/learn/data/matrix-calculus-examples.js';
import { affineGradientState, chainDerivatives, differenceCheck, differenceSteps, localApproximation, polynomialJacobian, polynomialValue } from '../src/learn/data/matrix-calculus-models.js';

const repository = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const directory = resolve(repository, 'scratch/matrix-calculus-verification');
mkdirSync(directory, { recursive: true });
const python = process.env.LESSON_PYTHON || resolve(repository, 'scratch/lesson-tools/Scripts/python.exe');
const env = { ...process.env, PYTHONIOENCODING: 'utf-8' };
for (const [key, example] of Object.entries(matrixCalculusExamples)) {
  const filename = resolve(directory, key + '.py');
  writeFileSync(filename, example.code);
  const run = spawnSync(python, ['-I', filename], { encoding: 'utf8', env, timeout: 20000 });
  assert.equal(run.status, 0, key + ': ' + run.stderr);
  assert.equal(run.stdout.replaceAll('\r\n', '\n').trim(), example.expected.trim(), key + ': exact displayed stdout');
}

const cases = { local: [], chain: [], affine: [], differences: [] };
for (let first = -3; first <= 3; first += 0.5) {
  for (let second = -3; second <= 3; second += 0.5) {
    const input = [first, second];
    for (const direction of [[1, -2], [1, 0], [0, 1], [0, 0], [-2, 1]]) {
      for (const step of [-0.5, -0.25, 0, 0.01, 0.125, 0.25, 0.5]) {
        cases.local.push(localApproximation(input, direction, step));
      }
      for (const weights of [[1, -1], [1, 0], [1, 1], [0, 0], [-2, 3]]) {
        cases.chain.push(chainDerivatives(input, direction, weights));
      }
    }
  }
}
for (const fixture of ['ordinary', 'repeated', 'zero']) {
  for (const reduction of ['sum', 'mean']) cases.affine.push(affineGradientState(fixture, reduction));
}
for (const kind of ['cubic', 'absolute']) {
  for (const point of [0, 0.3, -0.3, 2, -2, 1e-8]) {
    for (const step of differenceSteps) cases.differences.push(differenceCheck(kind, point, step));
  }
}
const invalid = [
  () => polynomialValue([1]), () => polynomialJacobian([NaN, 1]),
  () => localApproximation([1, 2], [Infinity, 1], 1),
  () => localApproximation([1, 2], [1, 1], NaN),
  () => chainDerivatives([1, 2], [1, 1], [1]),
  () => affineGradientState('unknown'), () => affineGradientState('ordinary', 'median'),
  () => differenceCheck('unknown', 1, 0.1), () => differenceCheck('cubic', 1, 0),
  () => differenceCheck('cubic', 1, -1), () => differenceCheck('cubic', Infinity, 1),
];
invalid.forEach(action => assert.throws(action));
writeFileSync(resolve(directory, 'model-cases.json'), JSON.stringify(cases));
const native = spawnSync(python, ['-I', resolve(repository, 'scripts/verify-matrix-calculus-native.py'), directory], { encoding: 'utf8', env, timeout: 60000 });
assert.equal(native.status, 0, native.stderr || native.stdout);
const evidence = { date: new Date().toISOString(), displayedPrograms: Object.keys(matrixCalculusExamples).length, invalidContracts: invalid.length, ...JSON.parse(native.stdout) };
writeFileSync(resolve(directory, 'results.json'), JSON.stringify(evidence, null, 2));
console.log(JSON.stringify(evidence, null, 2));
