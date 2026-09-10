import assert from 'node:assert/strict';
import { mkdirSync, writeFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { spawnSync } from 'node:child_process';
import { randomizedAlgorithmExamples } from '../src/learn/data/randomized-algorithm-examples.js';
import { amplification, expectedSelectionWork, productProbe, rejectionMap, reservoirDistribution, reservoirTrace, sampleBudget, shuffleTrace, quickselectTrace } from '../src/learn/data/randomized-algorithm-models.js';

const directory = resolve('scratch/randomized-algorithms-verification');
mkdirSync(directory, { recursive: true });
const python = resolve('scratch/lesson-tools/Scripts/python.exe');
for (const [name, example] of Object.entries(randomizedAlgorithmExamples)) {
  const file = resolve(directory, `${name}.py`);
  writeFileSync(file, example.code);
  const run = spawnSync(python, ['-X', 'utf8', '-I', file], { encoding: 'utf8', timeout: 15000 });
  assert.equal(run.status, 0, `${name}: ${run.stderr}`);
  assert.equal(run.stdout.replaceAll('\r\n', '\n').trim(), example.expected.trim(), name);
}
const evidence = { mappings: [], reservoirs: [], expectations: [], probes: [], majorities: [], budgets: [] };
for (let source = 2; source <= 32; source++) for (let target = 1; target <= source; target++) {
  for (const reject of [false, true]) evidence.mappings.push({ source, target, reject, ...rejectionMap(source, target, reject) });
}
for (let size = 1; size <= 8; size++) for (let capacity = 1; capacity <= size; capacity++) {
  evidence.reservoirs.push({ size, capacity, distribution: reservoirDistribution(size, capacity) });
}
for (let size = 1; size <= 20; size++) for (let rank = 0; rank < size; rank++) {
  const expectation = expectedSelectionWork(size, rank);
  assert(expectation <= 4 * size);
  evidence.expectations.push({ size, rank, expectation });
}
for (const fixture of ['correct', 'cancellation', 'even']) for (const bits of [[0, 0], [0, 1], [1, 0], [1, 1]]) {
  evidence.probes.push({ fixture, bits, ...productProbe(fixture, bits) });
}
for (const rounds of [1, 3, 5, 7, 9]) for (const failure of [0, 0.125, 0.25, 0.5, 0.75, 1]) {
  evidence.majorities.push({ rounds, failure, ...amplification(rounds, failure) });
}
for (const epsilon of [0.2, 0.1, 0.05, 0.02, 0.01]) for (const delta of [0.1, 0.05, 0.01, 0.001]) {
  evidence.budgets.push({ epsilon, delta, budget: sampleBudget(epsilon, delta) });
}
let traceStates = 0;
function paths(ranges, prefix = []) {
  if (!ranges.length) return [prefix];
  return Array.from({ length: ranges[0] }, (_, choice) => paths(ranges.slice(1), [...prefix, choice])).flat();
}
for (const choices of paths([4, 3, 2])) {
  const trace = shuffleTrace(choices);
  assert.equal(new Set(trace.at(-1).values).size, 4);
  traceStates += trace.length;
}
for (const capacity of [1, 2, 3]) for (const choices of paths(Array.from({ length: 6 - capacity }, (_, index) => capacity + index + 1))) {
  for (const state of reservoirTrace(capacity, choices)) {
    assert.equal(state.sample.length, capacity);
    assert.equal(new Set(state.sample).size, capacity);
    assert(state.sample.every(value => value < state.seen));
    traceStates++;
  }
}
for (const values of [[8, 1, 6, 3, 9, 2, 7, 4, 5], [4, 4, 1, 4, 2], [-3], [2, 2, 2]]) {
  for (let rank = 0; rank < values.length; rank++) {
    const queue = [[]];
    while (queue.length) {
      const choices = queue.pop();
      const state = quickselectTrace(values, rank, choices).at(-1);
      traceStates++;
      if (state.result !== null) assert.equal(state.result, [...values].sort((a, b) => a - b)[rank]);
      else for (let index = 0; index < state.active.length; index++) queue.push([...choices, index]);
    }
  }
}
const invalid = [() => rejectionMap(8, 9), () => reservoirDistribution(0, 1), () => reservoirTrace(2, [3]), () => shuffleTrace([4]), () => quickselectTrace([], 0, []), () => productProbe('correct', [0, 2]), () => sampleBudget(0, 0.1), () => amplification(0)];
invalid.forEach(run => assert.throws(run, RangeError));
writeFileSync(resolve(directory, 'examples.json'), JSON.stringify(randomizedAlgorithmExamples));
writeFileSync(resolve(directory, 'model-fixtures.json'), JSON.stringify(evidence));
const native = spawnSync(python, ['-X', 'utf8', '-I', resolve('scripts/verify-randomized-algorithms-native.py')], { encoding: 'utf8', timeout: 120000 });
assert.equal(native.status, 0, native.stderr || native.stdout);
const result = { checkedAt: new Date().toISOString(), completePrograms: Object.keys(randomizedAlgorithmExamples).length, traceStates, invalidCases: invalid.length, native: JSON.parse(native.stdout) };
writeFileSync(resolve(directory, 'results.json'), JSON.stringify(result, null, 2));
console.log(JSON.stringify(result, null, 2));
