import assert from 'node:assert/strict';
import { mkdirSync, writeFileSync } from 'node:fs';
import { resolve, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';
import { spawnSync } from 'node:child_process';
import { vectorTensorExamples } from '../src/learn/data/vector-tensor-examples.js';
import { applyMatrix, dot, linearMapPresets, matrixProduct, measurementTensor, parseSmallMatrix, productCell, projectVector, reduceMeasurementTensor } from '../src/learn/data/vector-tensor-models.js';

const repository = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const directory = resolve(repository, 'scratch/vector-tensor-verification');
mkdirSync(directory, { recursive: true });
const python = process.env.LESSON_PYTHON || resolve(repository, 'scratch/lesson-tools/Scripts/python.exe');
const environment = { ...process.env, PYTHONIOENCODING: 'utf-8' };
for (const [name, example] of Object.entries(vectorTensorExamples)) {
  const filename = resolve(directory, name + '.py');
  writeFileSync(filename, example.code);
  const result = spawnSync(python, ['-I', filename], { encoding: 'utf8', env: environment, timeout: 20000 });
  assert.equal(result.status, 0, name + ': ' + result.stderr);
  assert.equal(result.stdout.replace(/\r\n/g, '\n').trim(), example.expected.trim(), name + ': exact displayed stdout');
}

const cases = { projections: [], maps: [], products: [], reductions: [] };
for (let ux = -3; ux <= 3; ux++) {
  for (let uy = -3; uy <= 3; uy++) {
    for (let vx = -4; vx <= 4; vx++) {
      for (let vy = -4; vy <= 4; vy++) {
        const vector = [vx, vy], direction = [ux, uy];
        cases.projections.push({ vector, direction, result: projectVector(vector, direction) });
      }
    }
  }
}
for (const preset of Object.values(linearMapPresets)) {
  for (let x = -3; x <= 3; x++) for (let y = -3; y <= 3; y++) {
    cases.maps.push({ matrix: preset.matrix, vector: [x, y], output: applyMatrix(preset.matrix, [x, y]) });
  }
}
let seed = 92153;
const nextValue = () => {
  seed = (Math.imul(seed, 1664525) + 1013904223) >>> 0;
  return seed % 11 - 5;
};
for (let rows = 1; rows <= 3; rows++) {
  for (let inner = 1; inner <= 3; inner++) {
    for (let columns = 1; columns <= 3; columns++) {
      for (let trial = 0; trial < 20; trial++) {
        const left = Array.from({ length: rows }, () => Array.from({ length: inner }, nextValue));
        const right = Array.from({ length: inner }, () => Array.from({ length: columns }, nextValue));
        const cells = [];
        for (let row = 0; row < rows; row++) for (let column = 0; column < columns; column++) {
          for (let terms = 0; terms <= inner; terms++) cells.push({ row, column, terms, result: productCell(left, right, row, column, terms) });
        }
        cases.products.push({ left, right, output: matrixProduct(left, right), cells });
      }
    }
  }
}
const tensors = ['ramp', 'repeat', 'impulse'].map(measurementTensor);
for (let trial = 0; trial < 60; trial++) {
  tensors.push(Array.from({ length: 2 }, () => Array.from({ length: 2 }, () => Array.from({ length: 3 }, nextValue))));
}
for (const tensor of tensors) {
  for (let axis = 0; axis < 3; axis++) cases.reductions.push({ tensor, axis, result: reduceMeasurementTensor(tensor, axis) });
}
for (const invalid of ['', '1,;2,3', '1,2;3', '1,2,3,4', '1;2;3;4', 'Infinity', 'NaN', '21', '-21']) {
  assert.throws(() => parseSmallMatrix(invalid));
}
assert.deepEqual(parseSmallMatrix(' 1, -2.5 ; 3,0 '), [[1, -2.5], [3, 0]]);
assert.throws(() => matrixProduct([[1, 2]], [[1, 2]]));
assert.throws(() => dot([1], [1, 2]));
assert.throws(() => dot([Infinity], [1]));
assert.throws(() => productCell([[1]], [[2]], 0, 0, 2));
for (const axis of [-1, 3, 1.5]) assert.throws(() => reduceMeasurementTensor(measurementTensor(), axis));
assert.throws(() => reduceMeasurementTensor([[[1]]], 0));
assert.throws(() => measurementTensor('unknown'));

writeFileSync(resolve(directory, 'model-cases.json'), JSON.stringify(cases));
const oracle = spawnSync(python, ['-I', resolve(repository, 'scripts/verify-vector-tensor-native.py'), resolve(directory, 'model-cases.json')], {
  encoding: 'utf8', env: environment, timeout: 60000,
});
assert.equal(oracle.status, 0, oracle.stderr || oracle.stdout);
const evidence = { date: new Date().toISOString(), displayedPrograms: Object.keys(vectorTensorExamples).length, validationContracts: 19, ...JSON.parse(oracle.stdout) };
writeFileSync(resolve(directory, 'results.json'), JSON.stringify(evidence, null, 2));
console.log(JSON.stringify(evidence, null, 2));
