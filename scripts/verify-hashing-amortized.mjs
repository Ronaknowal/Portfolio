import assert from 'node:assert/strict';
import { mkdirSync, writeFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { spawnSync } from 'node:child_process';
import { hashingAmortizedExamples } from '../src/learn/data/hashing-amortized-examples.js';
import { defaultProbeProgram, denseSetTrace, hashFamilyState, parseProbeProgram, probeProgramTrace, resizeScenarios, resizeTrace } from '../src/learn/data/hashing-amortized-models.js';

const directory = resolve('scratch/hashing-amortized-verification');
mkdirSync(directory, { recursive: true });
const python = process.env.LESSON_PYTHON || resolve('scratch/lesson-tools/Scripts/python.exe');
const environment = { ...process.env, PYTHONIOENCODING: 'utf-8' };
for (const [key, example] of Object.entries(hashingAmortizedExamples)) {
  const file = resolve(directory, `${key}.py`);
  writeFileSync(file, example.code);
  const execution = spawnSync(python, ['-I', file], { encoding: 'utf8', env: environment, timeout: 20000 });
  assert.equal(execution.status, 0, `${key}: ${execution.stderr}`);
  assert.equal(execution.stdout.replaceAll('\r\n', '\n').trim(), example.expected.trim(), key);
}
let seed = 7919;
function random(limit) {
  seed = (Math.imul(seed, 1664525) + 1013904223) >>> 0;
  return seed % limit;
}
const probes = [];
for (const capacity of [4, 8, 16]) {
  const special = [defaultProbeProgram, '', 'put 3 1\nput 7 2\nput 11 3\ndel 7\nget 11\nput 11 9\nput 15 8', 'put -1 0\nget -1\ndel -1\nget -1\nrebuild 4'];
  for (let trial = 0; trial < 100; trial++) {
    const lines = [];
    for (let step = 0; step < 24; step++) {
      const key = (random(15) - 7) * 8 + 1;
      const kind = random(10);
      lines.push(kind < 5 ? `put ${key} ${random(21) - 10}` : kind < 7 ? `get ${key}` : kind < 9 ? `del ${key}` : `rebuild ${[4, 8, 16][random(3)]}`);
    }
    special.push(lines.join('\n'));
  }
  for (const program of special) probes.push({ capacity, program, trace: probeProgramTrace(program, capacity) });
}
const families = [];
for (let count = 1; count <= 8; count++) {
  for (let start = 0; start < 17; start++) {
    const keys = Array.from({ length: count }, (_, index) => (start + 3 * index) % 17);
    families.push(hashFamilyState(keys.join(','), 1 + random(16), random(17), keys[random(count)]));
  }
}
const resizes = [];
let level = [[]];
for (let length = 0; length <= 9; length++) {
  for (const operations of level) for (const policy of ['half', 'quarter', 'never']) {
    resizes.push({ operations, policy, growth: 'double', states: resizeTrace(operations, policy) });
  }
  level = level.flatMap(prefix => [[...prefix, '+'], [...prefix, '-']]);
}
for (const count of [1, 2, 3, 4, 8, 9, 16, 17, 24, 32, 33, 64]) {
  for (const growth of ['double', 'one']) resizes.push({ operations: Array(count).fill('+'), policy: 'never', growth, states: resizeTrace(Array(count).fill('+'), 'never', growth) });
}
for (const operations of Object.values(resizeScenarios)) for (const policy of ['half', 'quarter']) resizes.push({ operations, policy, growth: 'double', states: resizeTrace(operations, policy) });
const dense = [];
for (let length = 0; length <= 8; length++) {
  const values = Array.from({ length }, (_, index) => index * 3 - 10);
  for (const removed of [...values, 99]) dense.push({ values, removed, states: denseSetTrace(values, removed) });
}
const faults = probeProgramTrace(defaultProbeProgram, 8, true).states;
assert.ok(faults.some(state => state.issues.length > 0));
assert.ok(faults.some(state => state.agrees === false));
assert.throws(() => parseProbeProgram('put 1'));
const invalid = [
  () => parseProbeProgram('put 1 1.5'), () => parseProbeProgram('get NaN'),
  () => parseProbeProgram('put 1000 0'), () => parseProbeProgram('rebuild 3'),
  () => parseProbeProgram(Array(25).fill('get 1').join('\n')),
  () => probeProgramTrace('', 3), () => probeProgramTrace('', 8, 1),
  () => hashFamilyState('1,1'), () => hashFamilyState(''), () => hashFamilyState('17'),
  () => hashFamilyState('1,2', 0, 0, 1), () => hashFamilyState('1,2', 1, 17, 1),
  () => hashFamilyState('1,2', 1, 0, 3), () => resizeTrace(['?']),
  () => resizeTrace(Array(65).fill('+')), () => resizeTrace([], 'bad'),
  () => resizeTrace([], 'quarter', 'bad'), () => denseSetTrace([1, 1], 1),
  () => denseSetTrace([1.5], 1), () => denseSetTrace([], 0.5),
];
for (const check of invalid) assert.throws(check);
writeFileSync(resolve(directory, 'model-cases.json'), JSON.stringify({ probes, families, resizes, dense }));
const native = spawnSync(python, ['-I', resolve('scripts/verify-hashing-amortized-native.py'), directory], { encoding: 'utf8', env: environment, timeout: 120000 });
assert.equal(native.status, 0, native.stderr + native.stdout);
const result = { checkedAt: new Date().toISOString(), completePythonPrograms: Object.keys(hashingAmortizedExamples).length, invalidModelCases: invalid.length + 1, intentionalFaultDetected: true, native: JSON.parse(native.stdout) };
writeFileSync(resolve(directory, 'results.json'), JSON.stringify(result, null, 2));
console.log(JSON.stringify(result, null, 2));
