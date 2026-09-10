import assert from 'node:assert/strict';
import { mkdir, writeFile } from 'node:fs/promises';
import { spawnSync } from 'node:child_process';
import { dynamicProgrammingExamples } from '../src/learn/data/dynamic-programming-examples.js';
import { rewardTrace, gridPlan, lcsPlan, capacityTrace, subsetRoutePlan } from '../src/learn/data/dynamic-programming-models.js';

const directory = 'scratch/dynamic-programming-verification';
await mkdir(directory, { recursive: true });
const python = process.env.LESSON_PYTHON || 'scratch/lesson-tools/Scripts/python.exe';
for (const [name, example] of Object.entries(dynamicProgrammingExamples)) {
  const path = `${directory}/${name}.py`;
  await writeFile(path, `${example.code}\n`);
  const result = spawnSync(python, [path], { encoding: 'utf8' });
  assert.equal(result.status, 0, `${name}: ${result.stderr}`);
  assert.equal(result.stdout.replace(/\r\n/g, '\n').trimEnd(), example.expected, `${name}: exact stdout`);
}

function enumerateArrays(maximumLength, choices, visit, prefix = []) {
  visit(prefix);
  if (prefix.length < maximumLength) for (const value of choices) enumerateArrays(maximumLength, choices, visit, [...prefix, value]);
}
const fixtures = { rewards: [], grids: [], sequences: [], capacities: [], routes: [] };
enumerateArrays(5, [-2, 0, 3], values => {
  const memo = rewardTrace(values, 'memo');
  const table = rewardTrace(values, 'table');
  assert.equal(memo.result, table.result);
  for (const trace of [memo, table]) {
    for (const frame of trace.frames) {
      if (frame.phase === 'write') {
        assert.notEqual(frame.cache[frame.index + 1], null);
        assert.notEqual(frame.cache[frame.index + 2], null);
        assert.equal(frame.cache[frame.index], Math.max(frame.candidates.skip, frame.candidates.take));
      }
    }
  }
  fixtures.rewards.push({ values, result: memo.result, cache: table.frames.at(-1).cache });
});
for (let mask = 0; mask < 64; mask += 1) {
  const grid = [[1, -2, 3], [0, 4, 1]];
  const blocked = [];
  for (let index = 0; index < 6; index += 1) if (mask & (1 << index)) blocked.push(`${Math.floor(index / 3)},${index % 3}`);
  const model = gridPlan(grid, blocked);
  fixtures.grids.push(model);
}
const strings = [];
enumerateArrays(4, ['A', 'B'], values => strings.push(values.join('')));
for (const first of strings) for (const second of strings) fixtures.sequences.push(lcsPlan(first, second));
for (let capacity = 0; capacity <= 8; capacity += 1) {
  for (let firstWeight = 1; firstWeight <= 4; firstWeight += 1) {
    for (let secondWeight = 1; secondWeight <= 4; secondWeight += 1) {
      const items = [{ weight: firstWeight, value: 3 }, { weight: secondWeight, value: 4 }];
      for (const direction of ['ascending', 'descending']) {
        const trace = capacityTrace(items, capacity, direction);
        for (const frame of trace.frames.slice(1)) {
          if (direction === 'descending') assert.notEqual(frame.sourceGeneration, frame.item);
          for (let budget = 0; budget <= capacity; budget += 1) {
            const witness = frame.witnesses[budget];
            assert(witness.reduce((sum, index) => sum + items[index].weight, 0) <= budget);
            assert.equal(witness.reduce((sum, index) => sum + items[index].value, 0), frame.best[budget]);
            if (direction === 'descending') assert.equal(new Set(witness).size, witness.length);
          }
        }
        fixtures.capacities.push(trace);
      }
    }
  }
}
let seed = 432123;
function random(maximum) { seed = (Math.imul(seed, 1664525) + 1013904223) >>> 0; return seed % maximum; }
for (let caseIndex = 0; caseIndex < 120; caseIndex += 1) {
  const size = 1 + random(6);
  const costs = Array.from({ length: size }, (_, row) => Array.from({ length: size }, (_, column) => row === column ? 0 : random(5) === 0 ? null : random(15) - 5));
  const model = subsetRoutePlan(costs);
  fixtures.routes.push({ costs, best: model.best, result: model.result, path: model.path });
}
assert.throws(() => rewardTrace([100]));
assert.throws(() => rewardTrace([], 'wrong'));
assert.throws(() => gridPlan([[1], [2, 3]]));
assert.throws(() => lcsPlan('TOOLONG', 'ABC'));
assert.throws(() => lcsPlan('😀', 'A'));
assert.throws(() => capacityTrace([{ weight: 0, value: 3 }]));
assert.throws(() => subsetRoutePlan([[0, 1]]));
await writeFile(`${directory}/models.json`, JSON.stringify(fixtures));
const native = spawnSync(python, ['scripts/verify-dynamic-programming-native.py'], { encoding: 'utf8' });
assert.equal(native.status, 0, native.stderr || native.stdout);
console.log(native.stdout.trim());
const result = { checkedAt: new Date().toISOString(), exactPrograms: Object.keys(dynamicProgrammingExamples).length, modelCases: Object.fromEntries(Object.entries(fixtures).map(([key, values]) => [key, values.length])), native: JSON.parse(native.stdout) };
await writeFile(`${directory}/results.json`, JSON.stringify(result, null, 2));
console.log(JSON.stringify(result, null, 2));
