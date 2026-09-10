import assert from 'node:assert/strict';
import { mkdirSync, writeFileSync } from 'node:fs';
import { spawnSync } from 'node:child_process';
import * as model from '../src/learn/data/nonconvex-landscape-models.js';
import { nonconvexLandscapeExamples as examples } from '../src/learn/data/nonconvex-landscape-examples.js';

const folder = 'scratch/nonconvex-landscape-verification';
mkdirSync(folder, { recursive: true });
const cases = { wells: [], stationary: [], noise: [], symmetry: [], paths: [], interpolation: [] };
for (let tiltIndex = -5; tiltIndex <= 5; tiltIndex++) {
  for (let startIndex = -6; startIndex <= 6; startIndex++) {
    for (const rate of [0, 0.01, 0.06, 0.12, 0.2, 0.29, 0.3]) {
      cases.wells.push(model.wellState(tiltIndex * 0.05, startIndex * 0.25, rate));
    }
  }
}
for (const kind of Object.keys(model.stationaryPresets)) {
  for (let angle = 0; angle <= 360; angle += 15) {
    for (const radius of [0, 0.05, 0.2, 0.5, 0.95, 1]) cases.stationary.push(model.stationaryState(kind, angle, radius));
  }
}
for (const direction of ['none', 'stable', 'unstable', 'both']) {
  for (const rate of [0.02, 0.07, 0.12, 0.2, 0.25]) {
    for (const amplitude of [0, 0.05, 0.15, 0.5]) {
      for (const initialY of [-0.05, -0.001, 0, 0.001, 0.05]) cases.noise.push(model.saddleNoiseState(direction, rate, amplitude, initialY));
    }
  }
}
for (let exponent = -3; exponent <= 3; exponent += 0.25) {
  for (const displacement of [-0.3, -0.15, -0.01, 0, 0.01, 0.15, 0.3]) {
    for (const direction of ['normal', 'tangent']) cases.symmetry.push(model.symmetryState(exponent, displacement, direction));
  }
}
for (let index = 0; index <= 100; index++) cases.paths.push(model.modePaths(index / 100));
for (const parameter of [-3, -1, 0, 0.4, 2, 7]) {
  for (const input of [-2, -1, -0.25, 0, 0.5, 1, 2]) cases.interpolation.push({ parameter, input, value: model.interpolationValue(parameter, input) });
}
const invalid = [
  () => model.wellState(NaN), () => model.wellState(0.3), () => model.wellState(0, 2),
  () => model.wellState(0, 0, 0.31), () => model.wellState(0, 0, 0.1, 1.5),
  () => model.wellState(0, 0, 0.1, true), () => model.wellState(0, 0, 0.1, 81),
  () => model.stationaryState('missing'), () => model.stationaryState('constructor'),
  () => model.stationaryValue('toString', 0, 0), () => model.stationaryState('bowl', 361),
  () => model.stationaryState('bowl', 0, -1), () => model.saddleNoiseState('constructor'),
  () => model.saddleNoiseState('none', 0), () => model.saddleNoiseState('none', 0.1, 1),
  () => model.saddleNoiseState('none', 0.1, 0.1, Infinity),
  () => model.symmetryState(4), () => model.symmetryState(0, 0.4),
  () => model.symmetryState(0, 0, 'missing'), () => model.modePaths(-0.01),
  () => model.modePaths(true), () => model.landscapeNumber(Infinity),
];
invalid.forEach(call => assert.throws(call, RangeError));
assert.equal(model.landscapeNumber(1e-12), '1.000e-12');
assert.equal(model.landscapeNumber(0), '0');
for (const kind of ['saddle', 'flatSaddle']) {
  for (const angle of [45, 135, 225, 315]) {
    assert.equal(model.stationaryState(kind, angle, 0.5).actual, 0);
  }
}
const python = process.env.LESSON_PYTHON || 'scratch/lesson-tools/Scripts/python.exe';
const stdout = {};
for (const [key, example] of Object.entries(examples)) {
  const result = spawnSync(python, ['-X', 'utf8', '-c', example.code], { encoding: 'utf8' });
  assert.equal(result.status, 0, `${key}: ${result.stderr}`);
  const normalize = value => value.replaceAll('\r\n', '\n').trim();
  assert.equal(normalize(result.stdout), normalize(example.expected), `${key} displayed stdout`);
  stdout[key] = normalize(result.stdout);
}
writeFileSync(`${folder}/model-cases.json`, JSON.stringify(cases));
writeFileSync(`${folder}/examples.json`, JSON.stringify(examples));
writeFileSync(`${folder}/signs.json`, JSON.stringify(model.saddleSigns));
const native = spawnSync(python, ['-X', 'utf8', 'scripts/verify-nonconvex-landscape-native.py', folder], { encoding: 'utf8' });
assert.equal(native.status, 0, native.stderr + native.stdout);
const result = { at: new Date().toISOString(), displayedPrograms: Object.keys(examples).length, invalidModels: invalid.length, cases: Object.fromEntries(Object.entries(cases).map(([key, values]) => [key, values.length])), native: JSON.parse(native.stdout) };
writeFileSync(`${folder}/stdout.json`, JSON.stringify(stdout, null, 2));
writeFileSync(`${folder}/results.json`, JSON.stringify(result, null, 2));
console.log(JSON.stringify(result, null, 2));
