import fs from 'node:fs';
import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import * as model from '../src/learn/data/ordinary-differential-equations-models.js';
import * as original from '../scratch/ordinary-differential-equations-independent/author-sources/ordinary-differential-equations-models.js';
import { ordinaryDifferentialEquationsExamples as examples } from '../src/learn/data/ordinary-differential-equations-examples.js';

const directory = 'scratch/ordinary-differential-equations-independent';
const baseline = JSON.parse(fs.readFileSync(`${directory}/author-baseline.json`, 'utf8'));
const sources = baseline.sources.map(({ path }) => ({ path, sha256: createHash('sha256').update(fs.readFileSync(path)).digest('hex') }));
const fixtures = { cooling: [], logistic: [], matrices: [], forcing: [], stages: [], integrations: [], practice: [] };
let comparedFields = 0, maximumAbsoluteChange = 0, conservedStates = 0;
function compare(actual, expected) {
  if (typeof expected === 'number') {
    comparedFields += 1;
    const difference = Math.abs(actual - expected);
    assert(difference <= 1e-10 * Math.max(1, Math.abs(expected)), `${actual} != ${expected}`);
    maximumAbsoluteChange = Math.max(maximumAbsoluteChange, difference);
  } else if (expected && typeof expected === 'object') {
    assert.deepEqual(Object.keys(actual), Object.keys(expected));
    for (const key of Object.keys(expected)) compare(actual[key], expected[key]);
  } else assert.equal(actual, expected);
}
// Preserve author-checked inputs; changed small-number inputs have separate oracles below.
const authorFixtures = JSON.parse(fs.readFileSync('scratch/ordinary-differential-equations-verification/model-fixtures.json', 'utf8'));
const calls = {
  cooling: row => ['coolingState', [row.time, row.initial, row.rate, 10]],
  logistic: row => ['logisticState', [row.time, row.initial, row.rate, row.capacity]],
  waiting: row => ['waitingState', [row.time, row.departure]],
  matrices: row => ['exponential2', [row.matrix, row.time]],
  oscillators: row => ['oscillatorState', [row.time, row.damping, 1, row.velocity]],
  forcing: row => ['forcingTimeline', [row.time, 24, row.firstPower, row.secondPower, row.switchTime]],
  steps: row => ['odeStep', [row.method, (_time, state) => -row.rate * state, 0, 40, row.step]],
  integrations: row => ['numericalCooling', [row.method, row.step, 0.2, 5, 40]],
  amplification: row => ['coolingAmplification', [row.rate, row.step]],
  schedules: row => ['shearSchedule', [row.initial, row.first]],
};
for (const [kind, rows] of Object.entries(authorFixtures)) for (const row of rows) {
  const [name, args] = calls[kind](row);
  compare(model[name](...args), original[name](...args));
  conservedStates += 1;
}
for (const initial of [0, Number.MIN_VALUE, 1e-310, 1e-200, 2 ** -50, 3, 999999]) {
  for (const equilibrium of [0, 1e-200, 1, 1000]) for (const time of [0, 1e-200, 1e-14, 0.3, 19]) {
    fixtures.cooling.push({ initial, equilibrium, time, rate: 0.7, result: model.coolingState(time, initial, 0.7, equilibrium) });
  }
}
for (const initial of [Number.MIN_VALUE, 1e-300, 0.25, 7, 900]) for (const capacity of [0.01, 7, 91]) for (const time of [0.25, 0.75, 7]) {
  fixtures.logistic.push({ initial, capacity, time, rate: 0.3, result: model.logisticState(time, initial, 0.3, capacity) });
}
for (const diagonal of [-3, 0, 1]) for (const coupling of [-7, 0, 9]) for (const time of [-0.75, 0, 0.125, 1.75]) {
  const matrix = [[diagonal, coupling], [0, diagonal]];
  fixtures.matrices.push({ matrix, time, result: model.exponential2(matrix, time) });
}
for (const sign of [-1, 1]) for (const near of [0, 2 ** -35, -(2 ** -35)]) for (const time of [0.03125, 0.625, 3]) {
  const matrix = [[-2, sign * (4 + near)], [-sign, 2]];
  fixtures.matrices.push({ matrix, time, result: model.exponential2(matrix, time) });
}
for (const switchTime of [0, 0.75, 2.25, 6]) for (const time of [0.25, 0.75, 2.25, 5.75]) {
  fixtures.forcing.push({ switchTime, time, initial: 13, firstPower: 35, secondPower: 5,
    result: model.forcingTimeline(time, 13, 35, 5, switchTime) });
}
for (const method of ['euler', 'midpoint', 'rk4']) for (const time of [0, 0.375, 1.25]) for (const step of [0.0625, 0.25, 0.875]) {
  const state = 1.5, derivative = (t, y) => t - 2 * y;
  fixtures.stages.push({ method, time, step, state, result: model.odeStep(method, derivative, time, state, step) });
}
for (const method of ['euler', 'midpoint', 'rk4']) for (const step of [0.3, 0.15, 0.075]) {
  fixtures.integrations.push({ method, step, result: model.integrateScalar({ method, derivative: (t, y) => t - 2 * y, initial: 1.5, endTime: 1.7, step }) });
}
for (const endpoint of ['pi', 'half-pi']) for (const target of [-3, 0, 0.25, 2]) {
  const result = model.bvpFamily(endpoint, target, -1.5);
  assert.equal(result.classification, endpoint === 'half-pi' ? 'one solution' : target === 0 ? 'infinitely many solutions' : 'no solution');
}
const withInherited = (array, key, value) => {
  const prototype = Object.create(Array.prototype);
  prototype[key] = value;
  Object.setPrototypeOf(array, prototype);
  return array;
};
const badVectors = [[1, ,], withInherited([1, ,], 1, 2), [NaN, 1], [1, Infinity], [true, 1]];
let rejected = 0;
for (const vector of badVectors) for (const action of [
  () => model.applyOdeMatrix([[1, 0], [0, 1]], vector),
  () => model.linearSystemState([[1, 0], [0, 1]], vector, 1),
  () => model.shearSchedule(vector),
  () => model.exponential2([vector, [0, 1]], 1),
]) { assert.throws(action); rejected += 1; }
for (const action of [
  () => model.exponential2(withInherited([[1, 0], ,], 1, [0, 1]), 1),
  () => model.coolingState(1e-320, 0, 1e-5, 1e6),
  () => model.odeStep('rk4', () => Infinity, 0, 1, 0.1),
  () => model.odeStep('rk4', () => 1, 1e6, 1, Number.MIN_VALUE),
  () => model.bvpFamily('constructor', 0),
]) { assert.throws(action); rejected += 1; }
assert.equal(model.coolingState(0, 1e-200, 1, 1).state, 1e-200);
assert.equal(model.coolingState(1e-200, 1e-200, 1, 1).state, 2e-200);
const budget = model.integrateScalar({ derivative: () => 1, initial: 0, endTime: 10, step: 0.3, maximumSteps: 7 });
assert.match(budget.status, /exhausted/);
assert.equal(budget.steps, 7);
const stopped = model.integrateScalar({ derivative: () => NaN, initial: 0, endTime: 1, step: 0.1 });
assert.match(stopped.status, /nonfinite/);
assert.equal(stopped.steps, 0);
fs.writeFileSync(`${directory}/fixtures.json`, JSON.stringify({ at: new Date().toISOString(), sources, conservedStates, comparedFields, maximumAbsoluteChange, rejected, fixtures, examples }, null, 2));
console.log(JSON.stringify({ conservedStates, comparedFields, maximumAbsoluteChange, rejected, changedCases: Object.fromEntries(Object.entries(fixtures).map(([name, rows]) => [name, rows.length])) }));
