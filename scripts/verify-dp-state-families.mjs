import assert from 'node:assert/strict';
import fs from 'node:fs';
import { spawnSync } from 'node:child_process';
import * as model from '../src/learn/data/dp-state-families-models.js';
import { dpStateFamiliesExamples } from '../src/learn/data/dp-state-families-examples.js';

const fixtures = { examples: dpStateFamiliesExamples, matrices: [], balloons: [], trees: [], digits: [], prefixes: [] };
function arrays(length, choices, visit, prefix = []) {
  if (prefix.length === length) return visit(prefix);
  for (const value of choices) arrays(length, choices, visit, [...prefix, value]);
}
for (let length = 2; length <= 6; length += 1) arrays(length, [1, 2, 4], values => fixtures.matrices.push(model.matrixChainPlan(values)));
for (let length = 0; length <= 6; length += 1) arrays(length, [0, 1, 3], values => fixtures.balloons.push(model.balloonPlan(values)));
let seed = 563211;
function random(maximum) { seed = (Math.imul(seed, 1664525) + 1013904223) >>> 0; return seed % maximum; }
for (let index = 0; index < 650; index += 1) {
  const count = index % 10;
  const weights = Array.from({ length: count }, () => random(13) - 5);
  const edges = Array.from({ length: Math.max(0, count - 1) }, (_, child) => [random(child + 1), child + 1]);
  const root = count ? random(count) : 0;
  const tree = model.treeBoundaryPlan(weights, edges, root);
  const queries = [];
  for (let node = 0; node < count; node += 1) for (const parent of [false, true]) queries.push({ node, parent, ...model.treeBoundaryWitness(tree, node, parent) });
  fixtures.trees.push({ ...tree, queries });
}
for (let bound = 0; bound <= 2500; bound += 1) fixtures.digits.push({ bound, count: model.createDigitCounter(bound).total });
for (const bound of [0, 9, 99, 100, 102, 213, 999, 1098]) {
  const counter = model.createDigitCounter(bound);
  const prefixes = new Set(['']);
  for (let value = 0; value <= bound; value += 1) {
    const spelling = String(value).padStart(String(bound).length, '0');
    for (let length = 1; length <= spelling.length; length += 1) prefixes.add(spelling.slice(0, length));
  }
  for (const prefix of prefixes) {
    try { fixtures.prefixes.push(counter.inspect(prefix)); } catch (failure) {
      assert.match(failure.message, /digit already used|above bound/);
    }
  }
}
const invalid = [
  () => model.matrixChainPlan([]), () => model.matrixChainPlan([1]), () => model.matrixChainPlan([2, 0]),
  () => model.matrixChainPlan([2, NaN]), () => model.matrixChainPlan([2, 21]), () => model.matrixChainPlan(Array(3)),
  () => model.matrixChainPlan([true, 2]), () => model.matrixChainPlan('2,3'),
  () => model.balloonPlan([-1]), () => model.balloonPlan([Infinity]), () => model.balloonPlan(Array(2)),
  () => model.treeBoundaryPlan([1, 2], []), () => model.treeBoundaryPlan([1, 2], [[0, 0]]),
  () => model.treeBoundaryPlan([1, 2, 3], [[0, 1], [1, 0]]),
  () => model.treeBoundaryPlan([1, 2, 3, 4], [[0, 1], [1, 2], [2, 0]]),
  () => model.treeBoundaryPlan([1, 2], [[0, 2]]), () => model.treeBoundaryPlan(Array(2), [[0, 1]]),
  () => model.treeBoundaryPlan([1, 2], [Array(2)]), () => model.treeBoundaryPlan([], [], 1),
  () => model.treeBoundaryPlan([1], [], .2), () => model.treeBoundaryWitness(model.treeBoundaryPlan(), 0, 1),
  () => model.createDigitCounter(-1), () => model.createDigitCounter(1.2), () => model.createDigitCounter(1000000),
  () => model.createDigitCounter(NaN), () => model.createDigitCounter('213'),
  () => model.createDigitCounter(213).inspect('214'), () => model.createDigitCounter(213).inspect('11'),
  () => model.createDigitCounter(213).inspect('12 '), () => model.createDigitCounter(213).inspect('1234'),
];
for (const call of invalid) assert.throws(call);
assert.equal(model.createDigitCounter(213).inspect('21').remaining, 2);
assert.equal(model.createDigitCounter(213).inspect('12').remaining, 8);
assert(Object.isFrozen(model.matrixChainPlan().cells[0]));
assert(Object.isFrozen(model.balloonPlan().replay[0].live));
assert(Object.isFrozen(model.treeBoundaryPlan().children[0]));
assert(Object.isFrozen(model.createDigitCounter().inspect().branches[0]));
assert.throws(() => { model.matrixChainPlan().dimensions[0] = 3; });
fixtures.rejectedInputs = invalid.length;
fs.mkdirSync('scratch/dp-state-families-verification', { recursive: true });
fs.writeFileSync('scratch/dp-state-families-verification/fixtures.json', JSON.stringify(fixtures));
const result = spawnSync('scratch/lesson-tools/Scripts/python.exe', ['scripts/verify-dp-state-families-native.py'], { encoding: 'utf8', timeout: 120000 });
process.stdout.write(result.stdout);
process.stderr.write(result.stderr);
if (result.status !== 0) process.exit(result.status ?? 1);
