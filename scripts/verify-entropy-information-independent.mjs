import fs from 'node:fs';
import assert from 'node:assert/strict';
import { spawnSync } from 'node:child_process';
import { informationState, maxEntropyState, parseWeights } from '../src/learn/data/entropy-information-models.js';
import { entropyInformationExamples } from '../src/learn/data/entropy-information-examples.js';

const information = [];
for (const power of [3, 10, 22, 32, 42]) {
  for (const sign of [-1, 1]) {
    const epsilon = sign * 2 ** -power;
    for (const base of [2, Math.E]) {
      const p = [1, 3, 2, 2], q = [1 + epsilon, 3 - epsilon, 2, 2];
      information.push({ p, q, base, state: informationState(p, q, base) });
    }
  }
}
information.push({ p: [1, 1, 0, 0], q: [100, 1e-310, 0, 0], base: Math.E, state: informationState([1, 1, 0, 0], [100, 1e-310, 0, 0], Math.E) });
const maximum = [];
for (const mean of [0, 1e-6, 0.03, 0.5, 1, 1.75, 2 - 1e-6, 2]) {
  for (const fraction of [0, 0.15, 0.55, 1]) maximum.push(maxEntropyState(mean, fraction));
}
const rejections = [];
for (const mean of [1e-200, 1e-100, 0.5e-6, 2 - 0.5e-6]) {
  assert.throws(() => maxEntropyState(mean, 1), RangeError);
  rejections.push({ kind: 'maximum-entropy mean', input: mean });
}
for (const text of ['100,1e-322,0,0', '1,1e-9999,0,0']) {
  assert.throws(() => parseWeights(text), RangeError);
  rejections.push({ kind: 'positive probability underflow', input: text });
}
assert.equal(informationState([1, 1, 0, 0], parseWeights('100,0,0,0')).kl, Infinity);
const directory = 'scratch/entropy-independent-review';
fs.mkdirSync(directory, { recursive: true });
fs.writeFileSync(`${directory}/fixtures.json`, JSON.stringify({ information, maximum, rejections, examples: entropyInformationExamples }, (_, value) => Number.isFinite(value) || typeof value !== 'number' ? value : String(value), 2));
const result = spawnSync('scratch/lesson-tools/Scripts/python.exe', ['scripts/verify-entropy-information-independent.py'], { encoding: 'utf8' });
process.stdout.write(result.stdout);
process.stderr.write(result.stderr);
assert.equal(result.status, 0);
