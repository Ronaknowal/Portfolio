import assert from 'node:assert/strict';
import { mkdirSync, writeFileSync } from 'node:fs';
import { spawnSync } from 'node:child_process';
import * as models from '../src/learn/data/duality-kkt-models.js';
import { dualityKktExamples } from '../src/learn/data/duality-kkt-examples.js';

const folder = 'scratch/duality-kkt-verification';
mkdirSync(folder, { recursive: true });
const cases = { projection: [], scalar: [], sensitivity: [], resource: [] };
for (const budget of [-2, 0, 2.5, 5, 7, 8, 10]) {
  for (const candidate of [[-3, -3], [0, 0], [2, 2], [2, 3], [3, 4], [4, 4], [8, 8]]) {
    for (const multiplier of [-2, -0.5, 0, 0.5, 2, 7, 12]) {
      cases.projection.push(models.projectionCertificateState(budget, candidate, multiplier));
    }
  }
}
for (let center = -2; center <= 2; center += 0.5) {
  for (let candidate = -2; candidate <= 4; candidate += 0.5) {
    for (let multiplier = -2; multiplier <= 6; multiplier += 0.5) {
      cases.scalar.push(models.scalarKktState(center, candidate, multiplier));
    }
  }
}
for (const mode of ['quadratic', 'kink']) {
  const bases = mode === 'quadratic' ? [0, 1, 4.5, 5, 6.5, 7, 7.5, 10] : [-2, -1, -0.25, 0, 0.25, 1, 2];
  for (const base of bases) for (const change of [-2, -1, -0.25, 0, 0.25, 1, 2]) {
    for (const price of [0, 0.25, 0.5, 0.75, 1]) cases.sensitivity.push(models.sensitivityState(mode, base, change, price));
  }
}
for (const budget of [0, 0.5, 2.5, 2.75, 3, 3.25, 4, 5, 7, 8]) {
  for (const rate of [0, 0.25, 1, 1.75, 2.5, 3, 4]) {
    for (const initial of [0, 2, 7, 20]) cases.resource.push(models.resourceDualAscentState(budget, rate, initial, 20));
  }
}
const invalid = [
  () => models.projectionCertificateState(NaN), () => models.projectionCertificateState(11),
  () => models.projectionCertificateState(5, [1]), () => models.projectionCertificateState(5, [1, Infinity]),
  () => models.projectionCertificateState(5, [0, 0], -3), () => models.projectionCertificateState(5, [true, 1]),
  () => models.scalarKktState(3), () => models.scalarKktState(0, 5), () => models.scalarKktState(0, 0, NaN),
  () => models.sensitivityState('missing'), () => models.sensitivityState('kink', 3),
  () => models.sensitivityState('quadratic', 5, 3), () => models.sensitivityState('kink', 0, 0, 2),
  () => models.resourceAllocationAtPrice(-1, 5), () => models.resourceAllocationAtPrice(Infinity, 5),
  () => models.resourceAllocationAtPrice(0, -1), () => models.resourceDualAscentState(5, -1),
  () => models.resourceDualAscentState(5, 1, 21), () => models.resourceDualAscentState(5, 1, 0, 1.5),
  () => models.resourceDualAscentState(5, 1, 0, true), () => models.resourceDualAscentState(5, 1, 0, 31),
  () => models.dualDomainState(5), () => models.degenerateDualState(-1),
];
invalid.forEach(call => assert.throws(call));
for (const multiplier of [-2, 0, 0.5, 1, 1 + Number.EPSILON, 2, 4]) {
  const state = models.dualDomainState(multiplier);
  assert.equal(state.dualValue, multiplier === 1 ? 1 : -Infinity);
}
for (const multiplier of [0, 1, 10, 100, 1000000]) {
  const state = models.degenerateDualState(multiplier);
  assert.equal(state.dualAttained, false);
  assert.ok(state.dualValue < 0);
  if (multiplier) assert.ok(Math.abs(1 + 2 * multiplier * state.minimizer) < 1e-12);
}
assert.equal(models.formatDualityNumber(0), '0');
assert.equal(models.formatDualityNumber(1e-12), '1.000e-12');
assert.equal(models.formatDualityNumber(null), 'not a certificate');
assert.throws(() => { cases.projection[0].candidate[0] = 10; });
assert.throws(() => { cases.resource[0].frames[0].allocation[0] = 10; });

const python = process.env.LESSON_PYTHON || 'scratch/lesson-tools/Scripts/python.exe';
const outputs = {};
for (const [name, example] of Object.entries(dualityKktExamples)) {
  const execution = spawnSync(python, ['-c', example.code], { encoding: 'utf8' });
  assert.equal(execution.status, 0, `${name}: ${execution.stderr}`);
  const normalize = value => value.replaceAll('\r\n', '\n').trimEnd();
  assert.equal(normalize(execution.stdout), normalize(example.expected), `${name} stdout`);
  outputs[name] = normalize(execution.stdout);
}
writeFileSync(`${folder}/model-cases.json`, JSON.stringify(cases));
writeFileSync(`${folder}/examples.json`, JSON.stringify(dualityKktExamples));
const native = spawnSync(python, ['scripts/verify-duality-kkt-native.py', folder], { encoding: 'utf8' });
assert.equal(native.status, 0, native.stderr + native.stdout);
const result = { at: new Date().toISOString(), programs: Object.keys(outputs).length, invalidModelGroups: invalid.length,
  projectionStates: cases.projection.length, scalarStates: cases.scalar.length, sensitivityStates: cases.sensitivity.length,
  resourceFrames: cases.resource.reduce((total, state) => total + state.frames.length, 0), native: JSON.parse(native.stdout) };
writeFileSync(`${folder}/stdout.json`, JSON.stringify(outputs, null, 2));
writeFileSync(`${folder}/results.json`, JSON.stringify(result, null, 2));
console.log(JSON.stringify(result, null, 2));
