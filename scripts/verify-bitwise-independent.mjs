import fs from 'node:fs';
import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { execFileSync } from 'node:child_process';
import * as models from '../src/learn/data/bitwise-foundations-models.js';
import { bitwiseExamples } from '../src/learn/data/bitwise-foundations-examples.js';
import { arrayMapExamples } from '../src/learn/data/array-map-foundations-examples.js';
import practice from '../src/learn/data/practice/arrays-strings-hash-maps.js';

const directory = 'scratch/bitwise-independent-review';
fs.mkdirSync(directory, { recursive: true });
const hash = bytes => createHash('sha256').update(bytes).digest('hex');
const author = JSON.parse(fs.readFileSync('docs/teaching/evidence/bitwise-foundations-author-review.json'));
const sources = author.sources.map(({ path, sha256, archive }) => {
  assert.equal(hash(fs.readFileSync(path)), sha256);
  assert.equal(hash(fs.readFileSync(archive)), sha256);
  return { path, sha256 };
});
const original = JSON.parse(fs.readFileSync('docs/teaching/evidence/bitwise-foundations-original-sources.json'));
for (const item of original.sources) assert.equal(hash(Buffer.from(item.base64, 'base64')), item.sha256);
const oldSource = path => Buffer.from(original.sources.find(item => item.path === path).base64, 'base64').toString('utf8');
const oldBody = oldSource(sources[0].path).replaceAll('\r\n', '\n');
const body = fs.readFileSync(sources[0].path, 'utf8').replaceAll('\r\n', '\n');
const start = '<H2>1. Choose';
const bridge = '<Prose>You can move on when';
assert.equal(body.slice(body.indexOf(start), body.indexOf('<H2>9. Store')).trim(), oldBody.slice(oldBody.indexOf(start), oldBody.indexOf(bridge)).trim());
assert(body.includes(oldBody.slice(oldBody.indexOf(bridge), oldBody.indexOf('<DsaPractice')).trim()));
for (const item of original.sources.slice(3)) assert.equal(hash(fs.readFileSync(item.path)), item.sha256);
const oldPractice = (await import('data:text/javascript;base64,' + Buffer.from(oldSource('src/learn/data/practice/arrays-strings-hash-maps.js')).toString('base64'))).default;
const problems = practice.groups.flatMap(group => group.problems);
for (const problem of oldPractice.groups.flatMap(group => group.problems)) assert.deepEqual(problems.find(item => item.number === problem.number), problem);
assert.deepEqual(Object.values(arrayMapExamples), Object.values((await import('data:text/javascript;base64,' + Buffer.from(oldSource('src/learn/data/array-map-foundations-examples.js')).toString('base64'))).arrayMapExamples));

const counts = { frozenSources: sources.length, originalArchiveFiles: original.sources.length, originalProblemObjects: 10, exactOriginalSectionBlock: 1, sourcePositionMaps: 0, setPermutationQueries: 0, parityPrefixes: 0, partitionWitnesses: 0, sparseRemovalWitnesses: 0, rejectedModelCalls: 0 };
let seed = 90371;
const random = limit => {
  seed = (seed * 1664525 + 1013904223) % 4294967296;
  return seed % limit;
};
const digits = (value, width) => value.toString(2).padStart(width, '0');
const fromDigits = value => Number.parseInt(value || '0', 2);
const unsignedParity = (values, width) => fromDigits(Array.from({ length: width }, (_, index) => values.reduce((sum, value) => sum + Number(digits(value, width)[index]), 0) % 2).join(''));
const items = (value, width) => [...digits(value, width)].flatMap((bit, index) => bit === '1' ? [width - index - 1] : []).sort((a, b) => a - b);

for (const width of [1, 3, 7, 9, 16]) {
  const modulus = 2 ** width;
  for (const value of [0, modulus - 1, modulus / 2, modulus / 2 - 1, ...Array.from({ length: 12 }, () => random(modulus))]) {
    const text = digits(value, width);
    for (let shift = 0; shift <= width; shift += 1) {
      const actual = models.wordInterpretation(value, width, shift);
      const logicalText = '0'.repeat(shift) + text.slice(0, width - shift);
      const arithmeticText = text[0].repeat(shift) + text.slice(0, width - shift);
      const leftText = text.slice(shift) + '0'.repeat(shift);
      assert.equal(actual.logical, fromDigits(logicalText));
      assert.equal(actual.arithmeticWord, fromDigits(arithmeticText));
      assert.equal(actual.arithmetic, fromDigits(arithmeticText) - (arithmeticText[0] === '1' ? modulus : 0));
      assert.equal(actual.leftWord, fromDigits(leftText));
      for (let column = 0; column < width; column += 1) {
        const right = actual.rightOrigins[column].source;
        const left = actual.leftOrigins[column].source;
        assert.equal(right === null ? '0' : text[width - right - 1], logicalText[column]);
        assert.equal(right === null ? text[0] : text[width - right - 1], arithmeticText[column]);
        assert.equal(left === null ? '0' : text[width - left - 1], leftText[column]);
      }
      counts.sourcePositionMaps += 1;
    }
  }
  for (let run = 0; run < 24; run += 1) {
    const a = random(modulus), b = random(modulus);
    const universe = Array.from({ length: width }, (_, index) => index);
    const A = new Set(items(a, width)), B = new Set(items(b, width));
    const permutation = universe.map(index => (index + 1) % width);
    const permute = set => [...set].reduce((sum, index) => sum + 2 ** permutation[index], 0);
    const conditions = {
      intersection: i => A.has(i) && B.has(i), union: i => A.has(i) || B.has(i),
      difference: i => A.has(i) && !B.has(i), symmetric: i => A.has(i) !== B.has(i), complement: i => !A.has(i),
    };
    for (const [operation, condition] of Object.entries(conditions)) {
      const expected = universe.filter(condition);
      assert.deepEqual(models.packedSetView(a, b, width, operation).resultMembers, expected);
      assert.equal(models.packedSetView(permute(A), permute(B), width, operation).result, permute(new Set(expected)));
      counts.setPermutationQueries += 1;
    }
  }
}

for (let run = 0; run < 240; run += 1) {
  const values = Array.from({ length: 1 + random(12) }, () => random(65536));
  if (run % 3 === 0) values.fill(values[0]);
  const before = [...values];
  const actual = models.parityTrace(values, 16);
  assert.deepEqual(values, before);
  for (const state of actual.states) {
    assert.equal(state.accumulator, unsignedParity(values.slice(0, state.consumed), 16));
    counts.parityPrefixes += 1;
  }
  const frequencies = [...new Set(values)].map(value => [value, values.filter(item => item === value).length]);
  for (const number of [1, 2]) assert.equal(actual[number === 1 ? 'oneSingletonPromise' : 'twoSingletonPromise'], frequencies.filter(([, count]) => count === 1).length === number && frequencies.every(([, count]) => count <= 2));
}
for (let run = 0; run < 120; run += 1) {
  const values = [run, 65535 - run, 1000 + run, 2000 + run, 1000 + run, 2000 + run];
  const part = models.twoSingletonPartition(values, 16);
  assert(part.promiseHolds);
  assert.deepEqual(part.answers.toSorted((a, b) => a - b), [run, 65535 - run]);
  assert.deepEqual(part.groups.flat().toSorted((a, b) => a - b), values.toSorted((a, b) => a - b));
  assert.equal(items(part.separatingBit, 16).length, 1);
  assert.notEqual(Math.floor(values[0] / part.separatingBit) % 2, Math.floor(values[1] / part.separatingBit) % 2);
  counts.partitionWitnesses += 1;
}
for (const value of [0, 65535, 32768, ...Array.from({ length: 150 }, () => random(65536))]) {
  const positions = items(value, 16);
  const actual = models.sparseBitTrace(value, 16);
  assert.equal(actual.population, positions.length);
  for (let index = 0; index < actual.states.length; index += 1) {
    const state = actual.states[index];
    assert.equal(state.current, positions.slice(index).reduce((sum, position) => sum + 2 ** position, 0));
    assert.equal(state.removed, index);
    assert.equal(state.clearedPosition, positions[index] ?? null);
    if (index < positions.length) assert.equal(state.next, positions.slice(index + 1).reduce((sum, position) => sum + 2 ** position, 0));
    counts.sparseRemovalWitnesses += 1;
  }
}
for (const call of [
  () => models.parityTrace([, 1]), () => models.parityTrace([undefined, 1]),
  () => models.wordBits(0, true), () => models.wordInterpretation(1, 16, 17),
  () => models.wordInterpretation(-1, 8, 1), () => models.sparseBitTrace(65536, 16),
  () => models.packedSetView(1, 1, 4, 'constructor'), () => models.toggleMember(1, 8, Infinity),
  () => models.twoSingletonPartition([0, 0]), () => models.parityTrace(new Uint8Array([1, 2])),
]) { assert.throws(call, RangeError); counts.rejectedModelCalls += 1; }

const payload = { checkedAt: new Date().toISOString(), status: 'passed', sources, counts, examples: { ...arrayMapExamples, ...bitwiseExamples }, bitwiseExamples };
fs.writeFileSync(`${directory}/model-results.json`, JSON.stringify({ ...payload, examples: undefined, bitwiseExamples: undefined }, null, 2) + '\n');
fs.writeFileSync(`${directory}/native-input.json`, JSON.stringify(payload));
const result = execFileSync('scratch/lesson-tools/Scripts/python.exe', ['scripts/verify-bitwise-independent.py'], { encoding: 'utf8' });
process.stdout.write(result);
