import assert from 'node:assert/strict';
import { mkdir, writeFile } from 'node:fs/promises';
import { spawnSync } from 'node:child_process';
import { orderedPatternExamples } from '../src/learn/data/ordered-pattern-examples.js';
import { boundaryTrace, pairTrace, windowTrace, initialMergeState, mergeChoice, mergeRecords, rateSearch, rateState } from '../src/learn/data/ordered-pattern-models.js';

const directory = 'scratch/ordered-pattern-verification';
await mkdir(directory, { recursive: true });
const python = process.env.LESSON_PYTHON || 'scratch/lesson-tools/Scripts/python.exe';
for (const [name, example] of Object.entries(orderedPatternExamples)) {
  const path = `${directory}/${name}.py`;
  await writeFile(path, `${example.code}\n`);
  const result = spawnSync(python, [path], { encoding: 'utf8' });
  assert.equal(result.status, 0, `${name}: ${result.stderr}`);
  assert.equal(result.stdout.replace(/\r\n/g, '\n').trimEnd(), example.expected, `${name}: exact output mismatch`);
}
const models = { boundaries: [], pairs: [], windows: [], rates: [], merges: [] };
function sortedArrays(maximumLength, choices, visit, prefix = [], minimum = 0) {
  visit(prefix);
  if (prefix.length === maximumLength) return;
  for (let index = minimum; index < choices.length; index += 1) {
    sortedArrays(maximumLength, choices, visit, [...prefix, choices[index]], index);
  }
}
sortedArrays(6, [-2, 0, 2, 4], values => {
  for (let target = -3; target <= 6; target += 1) {
    for (const side of ['left', 'right']) {
      const trace = boundaryTrace(values, target, side);
      const expected = values.filter(value => side === 'left' ? value < target : value <= target).length;
      assert.equal(trace.boundary, expected);
      for (const step of trace.steps) {
        assert(values.slice(0, step.low).every(value => side === 'left' ? value < target : value <= target));
        assert(values.slice(step.high).every(value => side === 'left' ? value >= target : value > target));
        if (!step.done) assert(step.nextHigh - step.nextLow < step.high - step.low);
      }
      models.boundaries.push(trace);
    }
    models.pairs.push(pairTrace(values, target));
  }
});
function arrays(maximumLength, choices, visit, prefix = []) {
  visit(prefix);
  if (prefix.length === maximumLength) return;
  for (const value of choices) arrays(maximumLength, choices, visit, [...prefix, value]);
}
arrays(4, [0, 1, 3], values => {
  for (let target = 1; target <= 8; target += 1) models.windows.push(windowTrace(values, target));
});
for (let budget = 0; budget <= 18; budget += 1) {
  models.rates.push(rateSearch(budget));
  for (let speed = 1; speed <= 7; speed += 1) {
    const result = rateState(speed, budget);
    assert.equal(result.total, result.slots.reduce((sum, value) => sum + value, 0));
  }
}
function exploreMerge(state) {
  if (state.output.length === 6) {
    const keys = state.output.map(record => record.key);
    assert.deepEqual(keys, [1, 2, 4, 4, 4, 6]);
    const stable = state.output.filter(record => record.key === 4).map(record => record.id).join('') === 'BCE';
    assert.equal(state.stable, stable);
    models.merges.push(state);
    return;
  }
  for (const lane of ['left', 'right']) {
    const next = mergeChoice(state, lane);
    if (next.output.length > state.output.length) exploreMerge(next);
  }
}
exploreMerge(initialMergeState());
const rejected = mergeChoice(initialMergeState(), 'left');
assert.equal(rejected.output.length, 0);
for (const action of [() => boundaryTrace([2, 1], 1), () => boundaryTrace([NaN], 0), () => boundaryTrace([1], 1, 'middle'), () => pairTrace([0, -1], 0), () => windowTrace([1, -1, 5], 5), () => windowTrace([1], 0), () => rateState(0, 4), () => rateState(2, 4, []), () => rateSearch(-1)]) {
  assert.throws(action, RangeError);
}
await writeFile(`${directory}/programs.json`, JSON.stringify(orderedPatternExamples));
await writeFile(`${directory}/models.json`, JSON.stringify(models));
const native = spawnSync(python, ['scripts/verify-ordered-patterns-native.py', directory], { encoding: 'utf8' });
assert.equal(native.status, 0, native.stderr || native.stdout);
const summary = { programs: Object.keys(orderedPatternExamples).length, boundaryTraces: models.boundaries.length, pairTraces: models.pairs.length, windowTraces: models.windows.length, feasibilityBudgets: models.rates.length, mergePaths: models.merges.length, native: JSON.parse(native.stdout) };
await writeFile(`${directory}/results.json`, `${JSON.stringify(summary, null, 2)}\n`);
console.log(JSON.stringify(summary, null, 2));
