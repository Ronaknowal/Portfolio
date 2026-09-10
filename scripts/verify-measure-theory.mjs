import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import { spawnSync } from 'node:child_process';
import * as model from '../src/learn/data/measure-theory-models.js';
import { measureTheoryExamples } from '../src/learn/data/measure-theory-examples.js';

const output = path.resolve('scratch/measure-theory-verification');
fs.mkdirSync(output, { recursive: true });
const counts = {};
const data = { examples: measureTheoryExamples };
data.events = Object.keys(model.INFORMATION_PARTITIONS).flatMap(partition =>
  Array.from({ length: 64 }, (_, mask) => model.informationState(partition, mask)));
data.preimages = Array.from({ length: 13 }, (_, i) => model.preimageState(i / 4));
data.mixtures = [];
for (let w = 0; w <= 20; w += 1) {
  for (let low = -1; low <= 5; low += 1) {
    for (let high = low; high <= 5; high += 1) {
      data.mixtures.push(model.mixedMeasureState(w / 20, low / 4, high / 4));
    }
  }
}
data.cantor = Array.from({ length: 7 }, (_, n) => model.cantorCoverState(n));
data.simple = Array.from({ length: 7 }, (_, n) => model.simpleIntegralState(n));
data.limits = [];
for (const mode of Object.keys(model.LIMIT_MODES)) {
  for (let n = 1; n <= 64; n += 1) {
    for (const x of [0, .01, .25, .5, 1, 1 / n, 1 / (n * n)]) {
      data.limits.push(model.limitIntegralState(mode, n, x));
    }
  }
}
data.joint = [];
for (let column = 0; column < 4; column += 1) {
  for (let row = 0; row < 4; row += 1) {
    for (let a = .5; a <= 3; a += .5) {
      for (let b = .5; b <= 3; b += .5) {
        data.joint.push(model.jointCellState(column, row, a, b));
      }
    }
  }
}
data.arrays = [];
for (let r = 1; r <= 12; r += 1) {
  for (let c = 1; c <= 13; c += 1) data.arrays.push(model.signedArrayState(r, c));
}
data.conditional = [];
for (let fixture = 0; fixture < 100; fixture += 1) {
  const values = Array.from({ length: 6 }, (_, i) => ((fixture * 17 + i * i * 13) % 201) - 100);
  for (const partition of Object.keys(model.INFORMATION_PARTITIONS)) {
    for (const sampling of ['fair', 'missing-six']) {
      data.conditional.push(model.conditionalMeanState(partition, sampling, values, fixture - 50));
    }
  }
}
data.ratios = ['fair', 'missing-six'].flatMap(source =>
  Object.keys(model.TARGET_MEASURES).map(target => model.reweightState(source, target)));

const invalid = [
  () => model.informationState('other', 0), () => model.informationState('pairs', -1),
  () => model.informationState('pairs', 64), () => model.informationState('pairs', 1.5),
  () => model.preimageState(NaN), () => model.preimageState(4),
  () => model.mixedMeasureState(-.1, 0, 1), () => model.mixedMeasureState(.5, 1, 0),
  () => model.mixedMeasureState(.5, 0, 2), () => model.mixedCdf(Infinity),
  () => model.cantorCoverState(7), () => model.cantorCoverState(.5),
  () => model.simpleIntegralState(-1), () => model.simpleIntegralState(7),
  () => model.limitFunction('unknown', 1, 0), () => model.limitFunction('spike', 0, 0),
  () => model.limitFunction('bounded', 1.2, .5), () => model.limitFunction('increasing', 65, .5),
  () => model.limitFunction('spike', 1, -.1), () => model.limitFunction('spike', 1, 1.1),
  () => model.jointCellState(4, 0), () => model.jointCellState(0, 4),
  () => model.jointCellState(0, 0, 0, 1), () => model.jointCellState(0, 0, 1, Infinity),
  () => model.signedArrayState(0, 1), () => model.signedArrayState(1, 14),
  () => model.parseLosses('1,2,3'), () => model.parseLosses('1,2,3,4,5,'),
  () => model.parseLosses('1,2,3,4,5,Infinity'), () => model.parseLosses('1,2,3,4,5,101'),
  () => model.conditionalMeanState('full', 'fair', [1]), () => model.conditionalMeanState('full', 'other'),
  () => model.conditionalMeanState('full', 'fair', [0, 0, 0, 0, 0, NaN]),
  () => model.reweightState('other', 'uniform'), () => model.reweightState('fair', 'other'),
];
invalid.forEach(run => assert.throws(run, RangeError));
assert.equal(model.measureNumber(1e-12), '1.00e-12');
assert.deepEqual(model.parseLosses(' -2, 0, 1e-3, 4, 5, 100 '), [-2, 0, .001, 4, 5, 100]);
counts.invalid = invalid.length;
for (const [key, value] of Object.entries(data)) counts[key] = Array.isArray(value) ? value.length : Object.keys(value).length;
fs.writeFileSync(path.join(output, 'model-states.json'), JSON.stringify(data, (_, value) => value === Infinity ? '+Infinity' : value === -Infinity ? '-Infinity' : value));
fs.writeFileSync(path.join(output, 'model-counts.json'), JSON.stringify(counts, null, 2));
const python = process.env.LESSON_PYTHON || path.resolve('scratch/lesson-tools/Scripts/python.exe');
const check = spawnSync(python, ['scripts/verify-measure-theory-native.py'], { encoding: 'utf8', maxBuffer: 5_000_000 });
process.stdout.write(check.stdout || '');
process.stderr.write(check.stderr || '');
assert.equal(check.status, 0, 'Independent native oracle failed.');
console.log('Model validation boundaries:', invalid.length);
