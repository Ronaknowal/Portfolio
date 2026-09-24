import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import { spawnSync } from 'node:child_process';
import * as model from '../src/learn/data/entropy-information-models.js';
import { entropyInformationExamples as examples } from '../src/learn/data/entropy-information-examples.js';
const output = path.resolve('scratch/entropy-verification');
fs.mkdirSync(output, {
  recursive: true
});
const cases = {
  information: [],
  binary: [],
  prefix: [],
  conditional: [],
  logits: [],
  continuous: [],
  maximum: []
};
let seed = 173;
const random = () => {
  seed = Math.imul(seed, 1664525) + 1013904223 >>> 0;
  return seed / 2 ** 32;
};
for (let n = 0; n < 800; n++) {
  const p = Array.from({
    length: 4
  }, () => Math.floor(random() * 8));
  const q = Array.from({
    length: 4
  }, () => Math.floor(random() * 8));
  if (!p.some(Boolean)) p[0] = 1;
  if (!q.some(Boolean)) q[0] = 1;
  for (const base of [2, Math.E]) cases.information.push({
    p,
    q,
    base,
    state: model.informationState(p, q, base)
  });
}
for (let index = 0; index <= 100; index++) cases.binary.push(model.binaryEntropyState(index / 100));
function words(n) {
  return n === 0 ? [''] : words(n - 1).flatMap(word => model.SYMBOLS.map(symbol => word + symbol));
}
for (const codebook of Object.keys(model.CODEBOOKS)) for (let n = 1; n <= 3; n++) for (const word of words(n)) {
  const full = model.prefixCodeState(word, codebook);
  for (let consumed = 0; consumed <= full.bits.length; consumed++) cases.prefix.push({
    codebook,
    state: model.prefixCodeState(word, codebook, consumed)
  });
}
for (let i = 0; i <= 50; i++) for (const trust of [0, .01, .1, .5, .8, .9, .99, 1]) cases.conditional.push(model.conditionalLossState(i / 100, trust));
for (const gap of [0, .01, .5, 1, 2, 50, 1000]) for (const offset of [-1000, -800, 0, 1000]) for (let target = 0; target < 3; target++) cases.logits.push(model.logitsState(gap, offset, target));
for (let i = 1; i <= 32; i++) for (const scale of [1, 10, 100]) for (let bins = 2; bins <= 16; bins++) cases.continuous.push(model.continuousEntropyState(i / 8, scale, bins));
for (const mean of [...Array.from({
  length: 41
}, (_, i) => i / 20), 10 / 7, 1e-6, 2 - 1e-6]) for (let f = 0; f <= 20; f++) cases.maximum.push(model.maxEntropyState(mean, f / 20));
const invalid = [() => model.parseWeights(''), () => model.parseWeights('1,2,3'), () => model.parseWeights('1,2,3,4,5'), () => model.parseWeights('0,0,0,0'), () => model.parseWeights('1,-1,2,3'), () => model.parseWeights('1,NaN,2,3'), () => model.parseWeights('1,Infinity,2,3'), () => model.parseWeights('101,1,2,3'), () => model.informationState([1, 0], [1, 0, 0]), () => model.informationState([1, 0], [1, 0], 10), () => model.informationState([NaN, 1]), ...[-.1, 1.1, NaN, Infinity].map(p => () => model.binaryEntropyState(p)), ...['', 'A B', 'abc', 'ABCE', 'A'.repeat(17)].map(word => () => model.prefixCodeState(word)), () => model.prefixCodeState('A', 'unknown'), () => model.prefixCodeState('A', 'matched', 1.5), () => model.prefixCodeState('A', 'matched', 2), () => model.conditionalLossState(.6, .9), () => model.conditionalLossState(.1, -.1), () => model.logitsState(1001), () => model.logitsState(2, 1001), () => model.logitsState(2, 0, 3), () => model.continuousEntropyState(0), () => model.continuousEntropyState(.25, 0), () => model.continuousEntropyState(.25, 100, 1), () => model.continuousEntropyState(.25, 100, 2.5), () => model.maxEntropyState(-.1), () => model.maxEntropyState(2.1), () => model.maxEntropyState(1, -.1), () => model.maxEntropyState(1, 1.1)];
for (const mean of [1e-200, 1e-100, 1e-12, 2 - 1e-12]) invalid.push(() => model.maxEntropyState(mean, 1));
invalid.push(() => model.parseWeights('100,1e-9999,0,0'));
invalid.push(() => model.informationState([1, 1, 0, 0], [100, 1e-322, 0, 0], Math.E));
invalid.forEach(check => assert.throws(check));
cases.information.push({ p: [1, 1, 0, 0], q: [100, 1e-310, 0, 0], base: Math.E, state: model.informationState([1, 1, 0, 0], [100, 1e-310, 0, 0], Math.E) });
assert.equal(model.entropyNumber(1e-12), '1.00e-12');
const old = fs.readFileSync('scratch/entropy-authoring/original-lesson.jsx', 'utf8');
for (const key of ['coin', 'bernoulli']) assert(old.includes(examples[key].code), 'Original code bytes changed');
const nearMatch = Array.from({
  length: 40
}, (_, i) => {
  const eps = 2 ** (-i - 4);
  const q = [1 + eps, 1 - eps, 1, 1];
  return {
    eps,
    state: model.informationState([1, 1, 1, 1], q, Math.E)
  };
});
const payload = {
  cases,
  examples,
  nearMatch,
  invalidCases: invalid.length,
  codebooks: model.CODEBOOKS,
  originalPreserved: 2
};
fs.writeFileSync(path.join(output, 'cases.json'), JSON.stringify(payload, (_, value) => typeof value === 'number' && !Number.isFinite(value) ? String(value) : value));
const python = process.env.LESSON_PYTHON || path.resolve('scratch/lesson-tools/Scripts/python.exe');
const run = spawnSync(python, ['scripts/verify-entropy-information-native.py'], {
  stdio: 'inherit'
});
assert.equal(run.status, 0);
