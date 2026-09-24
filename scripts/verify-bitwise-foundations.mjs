import fs from 'node:fs';
import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { execFileSync } from 'node:child_process';
import { parse } from '@babel/parser';
import { wordBits, selectedMembers, toggleMember, packedSetView, wordInterpretation,
  parityTrace, sparseBitTrace, twoSingletonPartition } from '../src/learn/data/bitwise-foundations-models.js';
import { bitwiseExamples } from '../src/learn/data/bitwise-foundations-examples.js';
import { arrayMapExamples } from '../src/learn/data/array-map-foundations-examples.js';
import practice from '../src/learn/data/practice/arrays-strings-hash-maps.js';

const directory = 'scratch/bitwise-author';
fs.mkdirSync(directory, { recursive: true });
const original = JSON.parse(fs.readFileSync('docs/teaching/evidence/bitwise-foundations-original.json'));
const hash = bytes => createHash('sha256').update(bytes).digest('hex');
const normalize = value => Array.isArray(value) ? value.map(normalize) : value && typeof value === 'object'
  ? Object.fromEntries(Object.entries(value).filter(([key]) => !['loc','start','end','extra','comments','leadingComments','trailingComments','innerComments'].includes(key)).map(([key, child]) => [key, normalize(child)])) : value;
const sourceFiles = [...original.sources.map(source => source.path),
  'src/learn/data/bitwise-foundations-models.js', 'src/learn/data/bitwise-foundations-examples.js',
  'src/learn/components/lesson-labs/BitwiseFoundationsLabs.jsx', 'src/learn/components/lesson-labs/bitwise-foundations-labs.css'];
const sources = sourceFiles.map(path => ({ path, sha256: hash(fs.readFileSync(path)), bytes: fs.statSync(path).size }));
for (const source of original.sources.slice(3)) assert.equal(hash(fs.readFileSync(source.path)), source.sha256);
assert.deepEqual(arrayMapExamples, original.originalExamples);
const oldPractice = (await import('../scratch/bitwise-foundations-original/src/learn/data/practice/arrays-strings-hash-maps.js')).default;
const allProblems = practice.groups.flatMap(group => group.problems);
for (const problem of oldPractice.groups.flatMap(group => group.problems)) assert.deepEqual(allProblems.find(item => item.number === problem.number), problem);
assert.equal(allProblems.length, 15);
assert.equal(new Set(allProblems.map(item => item.number)).size, 15);
const teachingNames = new Set(['H2','H3','Prose','PythonExample','Checkpoint','LessonTable','ArrayMovementLab','TextUnitsLab','HashBucketLab','ArrayAddressFigure']);
function teachingNodes(source) {
  const nodes = [];
  function visit(value) {
    if (!value || typeof value !== 'object') return;
    if (value.type === 'JSXElement' && teachingNames.has(value.openingElement.name.name)) nodes.push(JSON.stringify(normalize(value)));
    for (const child of Object.values(value)) if (Array.isArray(child)) child.forEach(visit); else visit(child);
  }
  visit(parse(source, { sourceType: 'module', plugins: ['jsx'] }));
  return nodes;
}
const oldBody = fs.readFileSync(original.sources[0].archive, 'utf8');
const newBody = fs.readFileSync(original.sources[0].path, 'utf8');
const oldNodes = teachingNodes(oldBody), newNodes = new Set(teachingNodes(newBody));
assert(oldNodes.every(node => newNodes.has(node)), 'Every original teaching subtree must survive.');

// Independent finite-set oracle: parse binary strings into ordinary Set values.
const members = (value, width) => new Set(value.toString(2).padStart(width, '0').split('').flatMap((bit, index) => bit === '1' ? [width - index - 1] : []));
const sorted = set => [...set].sort((a, b) => a - b);
let setChecks = 0, wordChecks = 0, sparseChecks = 0, parityChecks = 0, partitionChecks = 0;
const wordPayload = [], sparsePayload = [], parityPayload = [];
for (let width = 1; width <= 8; width += 1) {
  for (let left = 0; left < 2 ** width; left += 1) {
    const leftSet = members(left, width);
    assert.deepEqual(selectedMembers(left, width), sorted(leftSet));
    for (let position = 0; position < width; position += 1) {
      const changed = new Set(leftSet);
      if (changed.has(position)) changed.delete(position); else changed.add(position);
      assert.deepEqual(selectedMembers(toggleMember(left, width, position), width), sorted(changed));
    }
    for (let right = 0; right < 2 ** width; right += 1) {
      const rightSet = members(right, width), universe = Array.from({ length: width }, (_, index) => index);
      const expected = {
        intersection: universe.filter(index => leftSet.has(index) && rightSet.has(index)),
        union: universe.filter(index => leftSet.has(index) || rightSet.has(index)),
        difference: universe.filter(index => leftSet.has(index) && !rightSet.has(index)),
        symmetric: universe.filter(index => leftSet.has(index) !== rightSet.has(index)),
        complement: universe.filter(index => !leftSet.has(index)),
      };
      for (const operation of Object.keys(expected)) {
        const view = packedSetView(left, right, width, operation);
        assert.deepEqual(view.resultMembers, expected[operation]);
        assert.equal(view.result, expected[operation].reduce((sum, index) => sum + 2 ** index, 0));
        setChecks += 1;
      }
    }
    for (let shift = 0; shift <= width; shift += 1) {
      const view = wordInterpretation(left, width, shift);
      const bits = left.toString(2).padStart(width, '0');
      const signed = [...bits].reduce((sum, bit, index) => sum + Number(bit) * 2 ** (width - index - 1) * (index === 0 ? -1 : 1), 0);
      const logicalBits = ('0'.repeat(shift) + bits.slice(0, width - shift)).slice(-width);
      const arithmeticBits = (bits[0].repeat(shift) + bits.slice(0, width - shift)).slice(-width);
      const leftBits = (bits.slice(shift) + '0'.repeat(shift)).slice(-width);
      assert.equal(view.signed, signed);
      assert.equal(view.logical, parseInt(logicalBits, 2));
      assert.equal(view.arithmeticWord, parseInt(arithmeticBits, 2));
      assert.equal(view.arithmetic, Math.floor(signed / 2 ** shift));
      assert.equal(view.leftWord, parseInt(leftBits, 2));
      view.rightOrigins.forEach((origin, index) => assert.equal(origin.source, index < shift ? null : width - index - 1 + shift));
      wordPayload.push(view); wordChecks += 1;
    }
  }
}
for (let value = 0; value <= 65535; value += 1) {
  const trace = sparseBitTrace(value, 16);
  const initialOnes = members(value, 16);
  assert.equal(trace.population, initialOnes.size);
  assert.equal(trace.states.length, initialOnes.size + 1);
  trace.states.slice(0, -1).forEach((state, index) => {
    const remaining = members(state.current, 16), lowest = Math.min(...remaining);
    remaining.delete(lowest);
    assert.equal(state.clearedPosition, lowest);
    assert.deepEqual(selectedMembers(state.next, 16), sorted(remaining));
    assert.equal(state.removed, index);
    assert.equal(index + remaining.size + 1, initialOnes.size);
  });
  if (value < 256) sparsePayload.push(sparseBitTrace(value, 8));
  sparseChecks += 1;
}
for (let length = 1; length <= 6; length += 1) {
  for (let serial = 0; serial < 4 ** length; serial += 1) {
    const values = Array.from({ length }, (_, index) => Math.floor(serial / 4 ** index) % 4);
    const before = [...values], trace = parityTrace(values, 3);
    const frequencies = [0, 1, 2, 3].map(value => [value, values.filter(item => item === value).length]).filter(([, count]) => count);
    const expectedXor = count => [0, 1, 2].reduce((sum, position) => sum + (values.slice(0, count).filter(value => members(value, 3).has(position)).length % 2) * 2 ** position, 0);
    trace.states.forEach(state => assert.equal(state.accumulator, expectedXor(state.consumed)));
    assert.deepEqual(trace.frequencies, frequencies);
    assert.deepEqual(values, before);
    const singletons = frequencies.filter(([, count]) => count === 1).map(([value]) => value);
    const paired = frequencies.every(([, count]) => count === 1 || count === 2);
    assert.equal(trace.oneSingletonPromise, paired && singletons.length === 1);
    assert.equal(trace.twoSingletonPromise, paired && singletons.length === 2);
    if (trace.twoSingletonPromise) {
      const partition = twoSingletonPartition(values, 3);
      assert.deepEqual([...partition.answers].sort((a, b) => a - b), singletons);
      assert(partition.promiseHolds); partitionChecks += 1;
    }
    if (length <= 4) parityPayload.push(trace);
    parityChecks += 1;
  }
}
const badCalls = [() => wordBits(-1, 8), () => wordBits(256, 8), () => wordBits(1, 0), () => wordBits(1, 17),
  () => wordBits(true, 8), () => wordBits(NaN, 8), () => toggleMember(1, 8, 8), () => packedSetView(1, 2, 8, 'constructor'),
  () => wordInterpretation(1, 8, -1), () => wordInterpretation(1, 8, 9), () => wordInterpretation(1, 8, .5),
  () => parityTrace([]), () => parityTrace(new Array(2)), () => parityTrace(Array(13).fill(0)), () => parityTrace([null]),
  () => parityTrace([256]), () => sparseBitTrace(-1), () => sparseBitTrace(Infinity), () => twoSingletonPartition([2, 2])];
badCalls.forEach(call => assert.throws(call, RangeError));
const payload = { checkedAt: new Date().toISOString(), sources, originalTeachingSubtrees: oldNodes.length,
  originalExamples: arrayMapExamples, bitwiseExamples, originalProblems: 10, allProblems: 15,
  checks: { setChecks, wordChecks, sparseChecks, parityChecks, partitionChecks, rejectedInputs: badCalls.length },
  wordPayload, sparsePayload, parityPayload };
fs.writeFileSync(`${directory}/payload.json`, JSON.stringify(payload));
const python = process.env.LESSON_PYTHON || 'scratch/lesson-tools/Scripts/python.exe';
execFileSync(python, ['scripts/verify-bitwise-foundations-native.py'], { stdio: 'inherit', timeout: 120000 });
console.log(JSON.stringify({ ...payload.checks, originalTeachingSubtrees: oldNodes.length, sources: sources.length }));
