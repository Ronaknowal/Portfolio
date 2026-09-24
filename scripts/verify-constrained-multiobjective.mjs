import assert from 'node:assert/strict';
import { mkdirSync, writeFileSync } from 'node:fs';
import { spawnSync } from 'node:child_process';
import * as model from '../src/learn/data/constrained-multiobjective-models.js';
import { constrainedExamples as examples } from '../src/learn/data/constrained-multiobjective-examples.js';

const folder = 'scratch/constrained-multiobjective-verification';
mkdirSync(folder, { recursive: true });
const cases = { projections: [], penalties: [], admm: [], decisions: [], continuous: [], units: [] };
for (const first of [-2, -0.6, 0, 0.5, 2, 4]) for (const second of [-2, -0.6, 0, 0.5, 3, 4]) {
  for (const budget of [0.5, 1, 2, 3]) for (const order of ['orthant-first', 'line-first']) {
    cases.projections.push(model.coupledProjectionState([first, second], budget, order));
  }
}
for (const mode of ['quadratic', 'hinge', 'barrier']) {
  const strengths = mode === 'quadratic' ? [0, 0.1, 1, 2, 19, 19.5, 20] : mode === 'hinge' ? [0, 0.1, 1.99, 2, 2.01, 3, 5] : [0.01, 0.1, 0.5, 1, 2, 3, 4];
  for (const strength of strengths) cases.penalties.push(model.constraintPenaltyState(mode, strength));
}
for (const target of [[2, -0.6], [1.5, -0.5], [-2, -2], [4, 4], [-2, 4], [0.5, 0.5]]) {
  for (const budget of [0.5, 1, 2, 3]) for (const rho of [0.1, 0.3, 1, 3, 10]) cases.admm.push(model.consensusAdmmState(target, budget, rho, 80));
}
for (const latency of [0, 7, 8, 14, 15, 18, 25, 70, 80]) for (const memory of [0, 31, 32, 40, 48, 96, 128, 384, 400]) {
  for (const price of [0, 0.01, 1 / 7, 0.2, 0.4, 1]) for (const method of ['error', 'weighted', 'latency']) cases.decisions.push(model.paretoDecisionState(latency, memory, price, method));
}
for (let index = 0; index <= 100; index++) for (const method of ['weighted', 'epsilon']) cases.continuous.push(model.continuousTradeoffState(method, index / 100, index / 25));
for (const item of model.deploymentCandidates) for (const price of [0, 0.01, 0.2, 0.4, 1]) cases.units.push({ item, price, ms: model.normalizedScore(item, 'ms', true, price), seconds: model.normalizedScore(item, 'seconds', true, price), wrong: model.normalizedScore(item, 'seconds', false, price) });
const invalid = [
  () => model.coupledProjectionState([NaN, 0]), () => model.coupledProjectionState([0]),
  () => model.coupledProjectionState([5, 0]), () => model.coupledProjectionState([0, 0], 0),
  () => model.coupledProjectionState([0, 0], 1, 'constructor'),
  () => model.constraintPenaltyState('constructor', 1), () => model.constraintPenaltyState('barrier', 0),
  () => model.constraintPenaltyState('quadratic', -1), () => model.constraintPenaltyState('hinge', 6),
  () => model.consensusAdmmState([0, 0], 1, 0), () => model.consensusAdmmState([0, 0], 1, 1, 0.5),
  () => model.consensusAdmmState([0, 0], 1, 1, true), () => model.consensusAdmmState([0, 0], 1, 1, 81),
  () => model.paretoDecisionState(-1), () => model.paretoDecisionState(25, -1),
  () => model.paretoDecisionState(25, 400, Infinity), () => model.paretoDecisionState(25, 400, 0.2, 'constructor'),
  () => model.paretoIndices([[1, NaN]]), () => model.paretoIndices([[1], [1, 2]]),
  () => model.dominatesMetrics([], []), () => model.continuousTradeoffState('unknown'),
  () => model.continuousTradeoffState('weighted', 2), () => model.continuousTradeoffState('epsilon', 0.5, -1),
  () => model.normalizedScore(model.deploymentCandidates[0], 'minutes'), () => model.constraintNumber(Infinity),
];
invalid.forEach(call => assert.throws(call, RangeError));
assert.equal(model.constraintNumber(1e-12), '1.000e-12');
assert.equal(model.modifiedConstraintCost('barrier', 1, 1), null);
assert.deepEqual(model.paretoIndices([[1, 1], [1, 1], [2, 1]]), [0, 1]);
assert.deepEqual(model.paretoIndices([]), []);
const python = process.env.LESSON_PYTHON || 'scratch/lesson-tools/Scripts/python.exe';
const stdout = {};
for (const [key, example] of Object.entries(examples)) {
  const result = spawnSync(python, ['-X', 'utf8', '-c', example.code], { encoding: 'utf8' });
  assert.equal(result.status, 0, `${key}: ${result.stderr}`);
  assert.equal(result.stdout.replaceAll('\r\n', '\n').trim(), example.expected.trim(), `${key}: visible expected output`);
  stdout[key] = example.expected;
}
writeFileSync(`${folder}/cases.json`, JSON.stringify({ cases, examples }));
const native = spawnSync(python, ['-X', 'utf8', 'scripts/verify-constrained-multiobjective-native.py', `${folder}/cases.json`], { encoding: 'utf8', maxBuffer: 8_000_000 });
assert.equal(native.status, 0, native.stdout + native.stderr);
const result = { timestamp: new Date().toISOString(), stdoutPrograms: Object.keys(stdout).length, invalidCases: invalid.length, modelCases: Object.fromEntries(Object.entries(cases).map(([key, value]) => [key, value.length])), native: JSON.parse(native.stdout), stdout };
writeFileSync(`${folder}/native-results.json`, JSON.stringify(result, null, 2));
console.log(JSON.stringify({ ...result, stdout: undefined }, null, 2));
