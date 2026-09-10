import fs from 'node:fs';
import assert from 'node:assert/strict';
import { execFileSync } from 'node:child_process';
import { matroidExchangeState, MATROID_PRESETS, assignmentTrace, knapsackSearchTrace, scaledKnapsackState, setCoverTrace, vertexCoverState } from '../src/learn/data/combinatorial-optimization-models.js';
const directory = 'scratch/combinatorial-optimization-review';
fs.mkdirSync(directory, { recursive: true });
let seed = 81732;
const random = maximum => { seed = (Math.imul(seed, 1664525) + 1013904223) >>> 0; return seed % maximum; };
const fixtures = { matroids: [], assignments: [], search: [], scaling: [], sets: [], covers: [] };
for (const kind of Object.keys(MATROID_PRESETS)) for (let sample = 0; sample < 30; sample += 1) {
  fixtures.matroids.push(matroidExchangeState({ kind, weights: MATROID_PRESETS[kind].weights.map(() => random(15) - 4), smaller: 1, larger: 6 }));
}
for (let sample = 0; sample < 240; sample += 1) {
  const workers = 1 + random(4), jobs = 1 + random(4);
  const costs = Array.from({ length: workers }, () => Array.from({ length: jobs }, () => random(5) === 0 ? null : random(21) - 10));
  const state = assignmentTrace({ costs, required: random(Math.min(workers, jobs) + 1) });
  assert.equal(state.negativeCycle, false);
  assert(state.residual.every(arc => arc.reducedCost >= 0));
  fixtures.assignments.push(state);
}
for (let sample = 0; sample < 220; sample += 1) {
  const items = Array.from({ length: random(7) }, (_, index) => ({ name: String(index), weight: 1 + random(9), value: random(30) }));
  const capacity = random(21);
  fixtures.search.push(knapsackSearchTrace({ items, capacity, maxExpanded: random(15) }));
  fixtures.scaling.push(scaledKnapsackState({ items, capacity, epsilonDenominator: [2, 4, 10, 20][random(4)] }));
}
for (let sample = 0; sample < 240; sample += 1) {
  const universeSize = random(9);
  const sets = Array.from({ length: random(7) }, (_, index) => ({ name: String(index), cost: random(12), elements: Array.from({ length: universeSize }, (_, element) => element).filter(() => random(3) === 0) }));
  fixtures.sets.push(setCoverTrace({ universeSize, sets, maximumSelections: sample % 2 ? random(sets.length + 1) : null }));
}
for (let sample = 0; sample < 180; sample += 1) {
  const costs = Array.from({ length: 1 + random(6) }, () => random(12));
  const edges = [];
  for (let a = 0; a < costs.length; a += 1) for (let b = a + 1; b < costs.length; b += 1) if (random(3) === 0) edges.push([a, b]);
  fixtures.covers.push(vertexCoverState({ costs, edges }));
}
const invalid = [
  () => assignmentTrace({ costs: [[NaN]] }), () => assignmentTrace({ costs: [[1], []] }),
  () => assignmentTrace({ costs: [[1]], required: 2 }), () => assignmentTrace({ costs: [] }),
  () => knapsackSearchTrace({ capacity: -1 }), () => knapsackSearchTrace({ items: [{ name: 'x', weight: 0, value: 2 }] }),
  () => knapsackSearchTrace({ maxExpanded: 512 }), () => scaledKnapsackState({ epsilonDenominator: 0 }),
  () => scaledKnapsackState({ items: Array(9).fill({ name: 'x', weight: 1, value: 2 }) }),
  () => setCoverTrace({ sets: [{ name: 'x', cost: 1, elements: [0, 0] }] }),
  () => setCoverTrace({ universeSize: 0, sets: [{ name: 'x', cost: 1, elements: [0] }] }),
  () => setCoverTrace({ maximumSelections: 4 }), () => vertexCoverState({ costs: [-1] }),
  () => vertexCoverState({ costs: [1, 1], edges: [[0, 1], [1, 0]] }),
  () => vertexCoverState({ costs: [1], edges: [[0, 0]] }),
  () => matroidExchangeState({ kind: 'unknown' }), () => matroidExchangeState({ weights: [1] }),
];
invalid.forEach(check => assert.throws(check, RangeError));
assert(Object.isFrozen(fixtures.assignments[0].costs[0]));
fs.writeFileSync(directory + '/model-fixtures.json', JSON.stringify(fixtures));
execFileSync('scratch/lesson-tools/Scripts/python.exe', ['scripts/verify-combinatorial-models.py'], { stdio: 'inherit' });
const report = { at: new Date().toISOString(), node: process.version, cases: Object.fromEntries(Object.entries(fixtures).map(([name, rows]) => [name, rows.length])), rejectedInputs: invalid.length, independent: JSON.parse(fs.readFileSync(directory + '/independent-model-results.json')) };
fs.writeFileSync(directory + '/model-results.json', JSON.stringify(report, null, 2) + '\n');
console.log(report.cases);
