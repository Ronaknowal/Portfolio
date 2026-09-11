import fs from 'node:fs';
import path from 'node:path';
import crypto from 'node:crypto';
import assert from 'node:assert/strict';
import { spawnSync } from 'node:child_process';
import * as model from '../src/learn/data/abstract-algebra-models.js';
import { abstractAlgebraExamples } from '../src/learn/data/abstract-algebra-examples.js';

const directory = 'scratch/abstract-algebra-verification';
fs.mkdirSync(directory, { recursive: true });
const sourcePaths = [
  'src/learn/data/topics/abstract-algebra-groups-symmetry-actions.jsx',
  'src/learn/data/abstract-algebra-models.js',
  'src/learn/data/abstract-algebra-examples.js',
  'src/learn/components/lesson-labs/AbstractAlgebraLabs.jsx',
  'src/learn/components/lesson-labs/abstract-algebra-labs.css',
  'src/learn/data/curriculum/blueprints/abstract-algebra-groups-symmetry-actions.js',
];
const data = {
  productionSources: sourcePaths.map(source => ({ path: source, sha256: crypto.createHash('sha256').update(fs.readFileSync(source)).digest('hex') })),
  examples: abstractAlgebraExamples,
  compositions: [], permutations: [], matrices: [], inverse: [], subgroups: [], orbits: [], fixed: [], averages: [], equivariance: [], modular: [],
};
for (let g = 0; g < 8; g += 1) {
  data.permutations.push(model.squarePermutation(g));
  data.matrices.push(model.squareMatrix(g));
  data.inverse.push(model.inverseSquare(g));
  for (let h = 0; h < 8; h += 1) data.compositions.push(model.compositionState(g, h, (g + h) % 4));
}
for (let mask = 0; mask < 256; mask += 1) {
  const subset = model.squareGroup.filter(g => mask & (1 << g));
  let cosets = null;
  try { cosets = model.leftCosets(subset); } catch (error) { assert(error instanceof RangeError || error instanceof TypeError); }
  data.subgroups.push({ subset, generated: model.generatedSubgroup(subset), cosets });
}
for (let index = 0; index < 81; index += 1) {
  const colors = [0, 1, 2, 3].map(power => Math.floor(index / (3 ** power)) % 3);
  for (const only of [false, true]) data.orbits.push(model.orbitState(colors, only));
}
for (const q of [2, 3, 4]) for (const only of [false, true]) data.fixed.push(model.fixedColoringCount(q, only));
for (let seed = 0; seed < 32; seed += 1) {
  const matrix = Array.from({ length: 4 }, (_, i) => Array.from({ length: 4 }, (_, j) => ((17 * seed + 7 * i + 11 * j + i * j) % 33 - 16) / 4));
  data.averages.push({ matrix, full: model.averageSquareMap(matrix), rotations: model.averageSquareMap(matrix, true) });
  assert.deepEqual(model.averageSquareMap(model.averageSquareMap(matrix)), model.averageSquareMap(matrix));
  const values = [0, 1, 2, 3].map(i => ((seed * 13 + i * 7) % 41 - 20) / 4);
  const weights = [((seed % 9) - 4) / 4, (((seed * 3) % 9) - 4) / 4, (((seed * 5) % 9) - 4) / 4];
  for (const mode of ['raw', 'rotations', 'averaged', 'tied', 'shift']) {
    for (let g = 0; g < 8; g += 1) data.equivariance.push(model.equivarianceState(values, mode, g, weights));
  }
}
for (let n = 2; n <= 12; n += 1) for (let a = 0; a < n; a += 1) for (let b = 0; b < n; b += 1) data.modular.push(model.modularState(n, a, b));
let rejectionCount = 0;
for (const action of [
  () => model.composeSquare(-1, 0), () => model.composeSquare(0, 8), () => model.squarePermutation(NaN),
  () => model.orbitState([0, , 0, 1]), () => model.orbitState([0, 0, 0, 3]), () => model.orbitState([0, 0, 0, 1], 'yes'),
  () => model.leftCosets([]), () => model.leftCosets([0, 1]), () => model.generatedSubgroup([, 1]),
  () => model.fixedColoringCount(1), () => model.fixedColoringCount(2.5),
  () => model.averageSquareMap([[1], [2], [3], [4]]),
  () => model.averageSquareMap(Array.from({ length: 4 }, () => [1, 2, 3, Infinity])),
  () => model.equivarianceState([1, 2, 3, 4], 'toString'), () => model.equivarianceState([1, 2, , 4]),
  () => model.equivarianceState([1, 2, 3, 0.1]), () => model.parseSensorReadings('1,2,,4'),
  () => model.parseSensorReadings('1,2,3,1e-100'), () => model.parseSensorReadings('1,2,3,21'),
  () => model.modularState(5, 5, 1), () => model.modularState(6, 2, 1.5),
]) { assert.throws(action); rejectionCount += 1; }
assert.deepEqual(model.parseSensorReadings(' -1.25, +2, .5, 0 '), [-1.25, 2, 0.5, 0]);
assert.equal(model.cosetProductState(false, 0).sameOutput, true);
assert.equal(model.cosetProductState(false, 1).sameOutput, false);
for (let index = 0; index < 4; index += 1) assert(model.cosetProductState(true, index).sameOutput);
data.rejectionCount = rejectionCount;
fs.writeFileSync(`${directory}/payload.json`, JSON.stringify(data));
const run = spawnSync(path.resolve('scratch/lesson-tools/Scripts/python.exe'), ['scripts/verify-abstract-algebra.py'], { encoding: 'utf8', env: { ...process.env, PYTHONIOENCODING: 'utf-8' } });
process.stdout.write(run.stdout);
process.stderr.write(run.stderr);
process.exitCode = run.status ?? 1;
