import assert from 'node:assert/strict';
import fs from 'node:fs';
import { parse } from '@babel/parser';
import katex from 'katex';
import * as model from '../src/learn/data/ordinary-differential-equations-models.js';
import { ordinaryDifferentialEquationsExamples } from '../src/learn/data/ordinary-differential-equations-examples.js';

const fixtures = { cooling: [], logistic: [], waiting: [], matrices: [], oscillators: [], forcing: [], steps: [], integrations: [], amplification: [], schedules: [] };
const body = fs.readFileSync('src/learn/data/topics/ordinary-differential-equations-linear-systems.jsx', 'utf8');
let equations = 0;
function inspect(node) {
  if (!node || typeof node !== 'object') return;
  if (node.type === 'TaggedTemplateExpression' && node.tag.type === 'MemberExpression' && node.tag.property.name === 'raw') {
    const formula = node.quasi.quasis.map(part => part.value.raw).join('');
    assert(!/[\x00-\x08\x0b\x0c\x0e-\x1f]/.test(formula));
    katex.renderToString(formula, { displayMode: true, throwOnError: true });
    equations += 1;
  }
  for (const value of Object.values(node)) {
    if (Array.isArray(value)) value.forEach(inspect);
    else if (value && typeof value === 'object') inspect(value);
  }
}
inspect(parse(body, { sourceType: 'module', plugins: ['jsx'] }));
for (const initial of [0, 1, 24, 40]) for (const rate of [0, 1e-12, 0.2, 4]) for (const time of [0, 0.1, 1, 10]) fixtures.cooling.push({ initial, rate, time, result: model.coolingState(time, initial, rate, 10) });
for (const initial of [0, Number.MIN_VALUE, 1e-310, 1e-12, 1, 10, 20, 1e6]) for (const rate of [0, 1, 100]) for (const time of [0, 1, 10]) for (const capacity of [1e-6, 10, 1e6]) fixtures.logistic.push({ initial, rate, time, capacity, result: model.logisticState(time, initial, rate, capacity) });
for (const departure of [0, 0.4, 1, 3]) for (const time of [0, departure, departure + 0.1, 4]) fixtures.waiting.push({ time, departure, result: model.waitingState(time, departure) });
for (const preset of Object.values(model.odeModePresets)) for (const time of [0, 1e-8, 0.4, 1, 3]) fixtures.matrices.push({ matrix: preset.matrix, time, result: model.exponential2(preset.matrix, time) });
let seed = 823;
function random() { seed = (Math.imul(seed, 1664525) + 1013904223) >>> 0; return seed / 2 ** 32; }
for (let index = 0; index < 180; index += 1) {
  const matrix = Array.from({ length: 2 }, () => Array.from({ length: 2 }, () => 8 * random() - 4));
  const time = 2 * random() - 1;
  fixtures.matrices.push({ matrix, time, result: model.exponential2(matrix, time) });
}
for (const damping of [0, 2, 4 - 1e-10, 4, 4 + 1e-10, 6, 10]) for (const velocity of [-2, 0, 2]) for (const time of [0, 1e-8, 0.5, 1, 6]) fixtures.oscillators.push({ damping, velocity, time, result: model.oscillatorState(time, damping, 1, velocity) });
for (const switchTime of [0, 1.3, 3, 6]) for (const firstPower of [0, 20, 30]) for (const secondPower of [0, 10]) for (const time of [0, switchTime, Math.min(6, switchTime + 0.1), 6]) fixtures.forcing.push({ time, switchTime, firstPower, secondPower, result: model.forcingTimeline(time, 24, firstPower, secondPower, switchTime) });
for (const method of ['euler', 'midpoint', 'rk4']) for (const rate of [0, 0.2, 4]) for (const step of [0.001, 0.4, 0.7]) fixtures.steps.push({ method, rate, step, result: model.odeStep(method, (_time, state) => -rate * state, 0, 40, step) });
for (const method of ['euler', 'midpoint', 'rk4']) for (const step of [0.7, 0.35, 0.175, 0.6]) {
  fixtures.integrations.push({ method, step, result: model.numericalCooling(method, step, 0.2, 5, 40) });
  const timeOnly = model.integrateScalar({ method, derivative: time => 2 * time, initial: 3, endTime: 2, step });
  if (method !== 'euler') assert(Math.abs(timeOnly.history.at(-1).state - 7) < 1e-12);
  const result = model.integrateScalar({ method, derivative: time => 3 * time ** 2, initial: 1, endTime: 1, step });
  if (method === 'rk4') assert(Math.abs(result.history.at(-1).state - 2) < 1e-12);
}
for (const rate of [0, 0.2, 4]) for (const step of [0, 0.25, 0.4, 0.5, 0.6, 1]) fixtures.amplification.push({ rate, step, result: model.coolingAmplification(rate, step) });
for (const initial of [[1,0], [0,1], [2,-3]]) for (const first of ['upper', 'lower']) fixtures.schedules.push({ initial, first, result: model.shearSchedule(initial, first) });
assert.equal(model.blowupState(0.5, 2).state, null);
assert.equal(model.blowupState(0.4, 2).status, 'inside the solution interval');
assert.equal(model.bvpFamily('pi', 0).classification, 'infinitely many solutions');
assert.equal(model.bvpFamily('pi', 1).classification, 'no solution');
assert.equal(model.bvpFamily('half-pi', 1).slope, 1);
const limited = model.integrateScalar({ derivative: (_t, y) => -y, initial: 1, endTime: 2, step: 0.1, maximumSteps: 3 });
assert(limited.status.includes('exhausted')); assert.equal(limited.steps, 3);
const nonfinite = model.integrateScalar({ derivative: () => Infinity, initial: 1, endTime: 1, step: 0.1 });
assert(nonfinite.status.includes('nonfinite')); assert.equal(nonfinite.steps, 0);
let rejected = 0;
for (const action of [
  () => model.exponential2([[0,,],[0,1]],1), () => model.exponential2([[0,1],,],1),
  () => model.logisticState(1,-1), () => model.logisticState(1,1,1,0),
  () => model.oscillatorState(1,-1), () => model.forcingTimeline(1,40,20,0,-1),
  () => model.shearSchedule([0,,]), () => model.odeStep('euler', () => 1, 1, 1, Number.MIN_VALUE),
  () => model.integrateScalar({ method: 'unknown', derivative: () => 0, initial: 0, endTime: 0, step: 1 }),
  () => model.integrateScalar({ derivative: () => 1, initial: 0, endTime: 1, step: 0 }),
  () => model.exponential2([[20,20],[20,20]],20), () => model.bvpFamily('approximate-pi',0),
]) { assert.throws(action); rejected += 1; }
assert.notEqual(model.formatOdeNumber(1e-12), '0');
const directory = 'scratch/ordinary-differential-equations-verification';
fs.mkdirSync(directory, { recursive: true });
fs.writeFileSync(`${directory}/model-fixtures.json`, JSON.stringify(fixtures));
fs.writeFileSync(`${directory}/examples.json`, JSON.stringify(ordinaryDifferentialEquationsExamples));
const result = { checkedAt: new Date().toISOString(), equations, rejected, cases: Object.fromEntries(Object.entries(fixtures).map(([key,value]) => [key,value.length])), status: 'passed' };
fs.writeFileSync(`${directory}/model-results.json`, JSON.stringify(result,null,2));
console.log(JSON.stringify(result));
