import assert from 'node:assert/strict';
import { mkdirSync, writeFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { spawnSync } from 'node:child_process';
import { algorithmCorrectnessExamples } from '../src/learn/data/algorithm-correctness-examples.js';
import {
  compactionTrace, euclidTrace, parseProofValues, partitionTrace,
  searchObligations, searchProofTrace,
} from '../src/learn/data/algorithm-correctness-models.js';

const directory = resolve('scratch/algorithm-correctness-verification');
mkdirSync(directory, { recursive: true });
const python = process.env.LESSON_PYTHON || resolve('scratch/lesson-tools/Scripts/python.exe');
const env = { ...process.env, PYTHONIOENCODING: 'utf-8' };
for (const [key, example] of Object.entries(algorithmCorrectnessExamples)) {
  const file = resolve(directory, `${key}.py`);
  writeFileSync(file, example.code);
  const run = spawnSync(python, ['-I', file], { encoding: 'utf8', env, timeout: 20000 });
  assert.equal(run.status, 0, `${key}: ${run.stderr}`);
  assert.equal(run.stdout.replaceAll('\r\n', '\n').trim(), example.expected.trim(), key);
}

function* arrays(alphabet, maximumLength) {
  let level = [[]];
  for (let length = 0; length <= maximumLength; length++) {
    yield* level;
    level = level.flatMap(prefix => alphabet.map(value => [...prefix, value]));
  }
}
const cases = { searches: [], compactions: [], partitions: [], euclids: [] };
for (const values of arrays([-1, 0, 1], 4)) {
  for (const target of [-1, 0, 1]) {
    for (const skip of [false, true]) {
      cases.searches.push({
        values, target, skip,
        states: searchProofTrace(values, target, skip),
        obligations: Object.fromEntries(['prefix', 'bounds', 'whole'].map(claim => [claim, searchObligations(values, target, claim, skip)])),
      });
    }
    cases.compactions.push({ values, removed: target, trace: compactionTrace(values, target) });
  }
}
for (const values of arrays([0, 1, 2], 6)) {
  for (const skipIncoming of [false, true]) {
    cases.partitions.push({ values, skipIncoming, states: partitionTrace(values, skipIncoming) });
  }
}
for (let first = 0; first <= 24; first++) {
  for (let second = 0; second <= 24; second++) {
    if (first + second > 0) cases.euclids.push({ first, second, states: euclidTrace(first, second) });
  }
}
for (const [first, second] of [[96, 0], [0, 96], [96, 95], [84, 30]]) {
  cases.euclids.push({ first, second, states: euclidTrace(first, second) });
}
const invalid = [
  () => parseProofValues('1,,2'), () => parseProofValues('1.5'),
  () => parseProofValues('NaN'), () => parseProofValues('1,2,3,4,5,6,7,8,9'),
  () => parseProofValues('21'), () => parseProofValues('0,3', { categories: true }),
  () => searchProofTrace([1], 0.5), () => searchProofTrace([Infinity], 0),
  () => searchObligations([], 0, 'unknown'), () => compactionTrace([], 21),
  () => partitionTrace([3]), () => euclidTrace(0, 0),
  () => euclidTrace(-1, 3), () => euclidTrace(97, 3), () => euclidTrace(1.5, 3),
];
invalid.forEach(action => assert.throws(action));
assert.deepEqual(parseProofValues(''), []);
assert.deepEqual(parseProofValues(' -20, 0, 20 '), [-20, 0, 20]);
writeFileSync(resolve(directory, 'model-cases.json'), JSON.stringify(cases));
const native = spawnSync(python, ['-I', resolve('scripts/verify-algorithm-correctness-native.py'), directory], {
  encoding: 'utf8', env, timeout: 60000,
});
assert.equal(native.status, 0, native.stderr || native.stdout);
const evidence = {
  checkedAt: new Date().toISOString(),
  programs: Object.keys(algorithmCorrectnessExamples).length,
  invalidModels: invalid.length,
  ...JSON.parse(native.stdout),
};
writeFileSync(resolve(directory, 'results.json'), JSON.stringify(evidence, null, 2));
console.log(JSON.stringify(evidence, null, 2));
