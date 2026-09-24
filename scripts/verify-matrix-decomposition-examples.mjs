import assert from 'node:assert/strict';
import { mkdirSync, writeFileSync } from 'node:fs';
import { resolve, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';
import { spawnSync } from 'node:child_process';
import { matrixDecompositionExamples } from '../src/learn/data/matrix-decomposition-examples.js';
import { covarianceGeometry, eliminationPresets, eliminationTrace, qrGeometry, svdGeometry } from '../src/learn/data/matrix-decomposition-models.js';

const repository = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const fixtureDirectory = resolve(repository, 'scratch/matrix-decomposition-verification');
mkdirSync(fixtureDirectory, { recursive: true });
const python = process.env.LESSON_PYTHON || resolve(repository, 'scratch/lesson-tools/Scripts/python.exe');
const environment = { ...process.env, PYTHONIOENCODING: 'utf-8' };
const failures = [];
for (const [name, example] of Object.entries(matrixDecompositionExamples)) {
  const source = resolve(fixtureDirectory, `${name}.py`);
  writeFileSync(source, example.code, 'utf8');
  const result = spawnSync(python, ['-I', source], { encoding: 'utf8', env: environment, timeout: 20000 });
  if (result.error || result.status !== 0 || result.stdout.replace(/\r\n/g, '\n').trim() !== example.expected.trim()) {
    failures.push({ name, error: result.error?.message, status: result.status, stderr: result.stderr,
      expected: example.expected, actual: result.stdout.replace(/\r\n/g, '\n').trim() });
  }
}
if (failures.length) {
  console.error(JSON.stringify(failures, null, 2));
  process.exit(1);
}
const fixtures = { elimination: Object.keys(eliminationPresets).map(eliminationTrace), qr: [], covariance: [], svd: [] };
for (let x = -4; x <= 4; x++) for (let y = -4; y <= 4; y++) fixtures.qr.push(qrGeometry([x / 2, y / 2]));
for (let index = -20; index <= 20; index++) fixtures.covariance.push(covarianceGeometry(index / 20));
for (const inputAngle of [-90, -30, 0, 30, 90]) {
  for (const outputAngle of [-90, -20, 0, 40, 90]) {
    for (const smallerScale of [0, 0.25, 1, 2.5, 3]) {
      for (const vectorAngle of [0, 45, 120, 270]) {
        for (const retained of [0, 1, 2]) fixtures.svd.push(svdGeometry({ inputAngle, outputAngle, smallerScale, vectorAngle, retained }));
      }
    }
  }
}
assert.throws(() => qrGeometry([Infinity, 0]));
assert.throws(() => qrGeometry([3, 0]));
assert.throws(() => covarianceGeometry(1.01));
assert.throws(() => covarianceGeometry(NaN));
assert.throws(() => svdGeometry({ smallerScale: -1 }));
assert.throws(() => svdGeometry({ retained: 0.5 }));
assert.throws(() => eliminationTrace('unknown'));
writeFileSync(resolve(fixtureDirectory, 'programs.json'), JSON.stringify(matrixDecompositionExamples));
writeFileSync(resolve(fixtureDirectory, 'models.json'), JSON.stringify(fixtures));
const oracle = spawnSync(python, ['-I', resolve(repository, 'scripts/verify-matrix-decomposition-native.py'), fixtureDirectory], {
  encoding: 'utf8', env: environment, timeout: 60000,
});
assert.equal(oracle.error, undefined, oracle.error?.message);
assert.equal(oracle.status, 0, oracle.stderr || oracle.stdout);
console.log(`Verified ${Object.keys(matrixDecompositionExamples).length} standalone Python programs and exact displayed outputs.`);
console.log(oracle.stdout.trim());
