import fs from 'node:fs';
import crypto from 'node:crypto';
import assert from 'node:assert/strict';
import { spawnSync } from 'node:child_process';
import * as model from '../src/learn/data/support-vector-machines-models.js';
import { svmExamples } from '../src/learn/data/support-vector-machines-examples.js';
import validation from '../src/learn/data/svm-validation-fixtures.js';

const directory = 'scratch/support-vector-machines-verification';
const hash = file => crypto.createHash('sha256').update(fs.readFileSync(file)).digest('hex');
const prior = JSON.parse(fs.readFileSync(`${directory}/program-results.json`));
for (const program of prior.programs) {
  assert.equal(hash(program.path), program.sha256, `Native source changed: ${program.path}`);
  assert.equal(svmExamples[program.key].code, fs.readFileSync(program.path, 'utf8').trim());
  assert.equal(svmExamples[program.key].expected, program.stdout);
}
const cases = { geometry: [], support: [], soft: [], kernels: [], pairs: [], tubes: [], sequences: [], validation };
const tiedDuplicate = model.svmPairStep([[-1, 0], [1, 0], [1, 0]], [-1, 1, 1], [.4, .1, .3], 1, 1, 2);
assert.equal(tiedDuplicate.q, 0);
assert.equal(tiedDuplicate.g, 0);
assert.equal(tiedDuplicate.bestDelta, 0);
assert.deepEqual(tiedDuplicate.after, [.4, .1, .3]);
cases.pairs.push(tiedDuplicate);
for (const angle of [-45, -25, 0, 17, 45]) for (const offset of [-.75, 0, .75]) for (const scale of [.5, 1, 3]) {
  cases.geometry.push({ angle, offset, scale, result: model.marginGeometry(angle, offset, scale) });
}
for (const moved of [.25, .5, .95, 1, 1.05, 2, 3]) cases.support.push({ moved, result: model.supportMotion(moved) });
for (const c of [.05, .25, .5, 1, 4]) for (const conflicting of [false, true]) for (const bias of [-1, -.3, 0, 1]) {
  cases.soft.push({ c, conflicting, bias, result: model.softPairState(c, conflicting, bias) });
}
for (const kind of ['linear', 'poly', 'rbf']) for (const c of [.05, .25, 1, 4]) for (const gamma of [.05, .5, 4]) {
  for (const query of [[-2, -2], [.5, -.7], [0, 0], [2, 2]]) {
    const result = model.xorKernelState(kind, c, gamma, query);
    assert.equal(model.xorScore(query, kind, c, gamma), result.score);
    cases.kernels.push({ kind, c, gamma, query, result });
  }
}
for (const duplicate of [false, true]) {
  const fixture = model.svmPairFixture(duplicate);
  let alpha = fixture.alpha;
  for (let step = 0; step < 16; step += 1) {
    const i = duplicate ? 0 : step % 2;
    const j = duplicate ? 1 : i + 1;
    const result = model.svmPairStep(fixture.points, fixture.labels, alpha, fixture.c, i, j);
    assert(result.after.every(value => value >= 0 && value <= fixture.c));
    assert(Math.abs(result.balance) <= 1e-9);
    assert(result.afterDual >= result.beforeDual - 1e-12);
    cases.pairs.push(result);
    alpha = result.after;
  }
}
for (let code = 0; code < 27; code += 1) {
  const points = [[-1, code % 3 - 1], [.5, Math.floor(code / 3) % 3 - 1], [Math.floor(code / 9) - 1, .5]];
  for (const pair of [[0, 1], [0, 2], [1, 2]]) cases.pairs.push(model.svmPairStep(points, [-1, 1, 1], [.4, .1, .3], 1, ...pair));
}
for (const amplitude of [1, 1.7, 3]) for (const epsilon of [0, .3, 1, 2]) for (const c of [.05, .25, 1, 4]) {
  cases.tubes.push({ amplitude, epsilon, c, result: model.svrTubeState(amplitude, epsilon, c) });
}
for (const word of ['', 'A', 'ACACA', 'CACAC', 'AACCA', 'CCCCCC']) for (const size of [1, 2, 3]) {
  cases.sequences.push({ word, size, result: model.spectrumCounts(word, size) });
}
const invalid = [
  () => model.marginGeometry(NaN), () => model.marginGeometry(0, Infinity),
  () => model.supportMotion(0), () => model.softPairState(.2, 'yes'),
  () => model.xorKernelState('bad'), () => model.xorKernelState('rbf', 1, .5, Array(2)),
  () => model.svmKernel([0, NaN], [0, 1]), () => model.svmKernel([0, 0], [0, 1], 'bad'),
  () => model.svmPairStep([[0, 0], [0, 0]], [-1, 1], [0, .2], 1, 0, 1),
  () => model.svmPairStep([Array(2), [0, 0]], [-1, 1], [0, 0], 1, 0, 1),
  () => model.svrTubeState(1, -1), () => model.spectrumCounts('ACG', 2),
];
invalid.forEach(run => assert.throws(run));
cases.invalidCount = invalid.length;
fs.writeFileSync(`${directory}/model-cases.json`, JSON.stringify(cases));
const result = spawnSync('scratch/lesson-tools/Scripts/python.exe', ['scripts/verify-support-vector-machines-native.py'], { encoding: 'utf8', timeout: 120000 });
process.stdout.write(result.stdout || '');
process.stderr.write(result.stderr || '');
assert.equal(result.status, 0, 'Independent finite verification failed');
const record = JSON.parse(fs.readFileSync(`${directory}/model-results.json`));
record.sourceHashes = [
  'src/learn/data/support-vector-machines-models.js',
  'src/learn/data/support-vector-machines-examples.js',
  'src/learn/data/svm-validation-fixtures.js',
].map(path => ({ path, sha256: hash(path) }));
record.executedPrograms = { count: prior.programCount, executedAt: prior.executedAt, unchangedSourcesAndDisplayedOutputs: true };
fs.writeFileSync(`${directory}/model-results.json`, JSON.stringify(record, null, 2) + '\n');
