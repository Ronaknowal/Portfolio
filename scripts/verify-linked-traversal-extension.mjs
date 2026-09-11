import fs from 'node:fs';
import crypto from 'node:crypto';
import assert from 'node:assert/strict';
import { spawnSync } from 'node:child_process';
import * as linked from '../src/learn/data/linked-traversal-models.js';
import * as monotone from '../src/learn/data/monotonic-stack-models.js';
import { linkedTraversalExamples } from '../src/learn/data/linked-traversal-examples.js';
import { monotonicStackExamples } from '../src/learn/data/monotonic-stack-examples.js';
import { linkedExamples } from '../src/learn/data/linked-foundations-examples.js';
import practice from '../src/learn/data/practice/linked-lists-stacks-queues.js';
import { parse } from '@babel/parser';

const directory = 'scratch/linked-traversal-extension-verification';
fs.mkdirSync(directory, { recursive: true });
const paths = [
  'src/learn/data/topics/linked-lists-stacks-queues.jsx', 'src/learn/data/practice/linked-lists-stacks-queues.js',
  'src/learn/data/curriculum/blueprints/linked-lists-stacks-queues.js',
  'src/learn/data/linked-traversal-models.js', 'src/learn/data/monotonic-stack-models.js',
  'src/learn/data/linked-traversal-examples.js', 'src/learn/data/monotonic-stack-examples.js',
  'src/learn/components/lesson-labs/LinkedTraversalLabs.jsx', 'src/learn/components/lesson-labs/MonotonicStackLabs.jsx',
  'src/learn/components/lesson-labs/linked-traversal-labs.css', 'src/learn/components/lesson-labs/monotonic-stack-labs.css',
];
const fingerprints = () => paths.map(path => ({ path, sha256: crypto.createHash('sha256').update(fs.readFileSync(path)).digest('hex') }));
for (const path of paths.filter(path => /\.[jt]sx?$/.test(path))) parse(fs.readFileSync(path, 'utf8'), { sourceType: 'module', plugins: ['jsx'] });
const baseline = JSON.parse(fs.readFileSync('docs/teaching/evidence/linked-traversal-extension-original.json', 'utf8'));
for (const source of baseline.sources.filter(row => !paths.includes(row.path))) assert.equal(crypto.createHash('sha256').update(fs.readFileSync(source.path)).digest('hex'), source.sha256, source.path);
assert.deepEqual(linkedExamples, JSON.parse(fs.readFileSync('scratch/linked-traversal-extension-original/actual-examples.json', 'utf8')));
const oldPractice = JSON.parse(fs.readFileSync('scratch/linked-traversal-extension-original/practice.json', 'utf8'));
const problems = practice.groups.flatMap(group => group.problems);
for (const problem of oldPractice.groups.flatMap(group => group.problems)) {
  const current = problems.find(row => row.number === problem.number);
  assert(current);
  if (problem.number === 141) for (const key of Object.keys(problem).filter(key => key !== 'transfer')) assert.equal(current[key], problem[key]);
  else assert.deepEqual(current, problem);
}
// Keep every original explanation, complete-example placement and mechanism subtree.
function semantic(node) {
  if (Array.isArray(node)) return node.map(semantic);
  if (!node || typeof node !== 'object') return node;
  return Object.fromEntries(Object.entries(node).filter(([key]) => !['start', 'end', 'loc', 'extra', 'comments', 'leadingComments', 'trailingComments', 'innerComments'].includes(key)).map(([key, value]) => [key, semantic(value)]));
}
function meaningfulElements(source) {
  const elements = [];
  const visit = node => {
    if (!node || typeof node !== 'object') return;
    if (node.type === 'JSXElement' && ['Prose','H2','H3','Checkpoint','LessonTable','PythonExample','LinkedReversalLab','BracketStackLab','CircularQueueLab'].includes(node.openingElement.name.name)) elements.push(JSON.stringify(semantic(node)));
    for (const value of Object.values(node)) if (Array.isArray(value)) value.forEach(visit); else if (value && typeof value === 'object') visit(value);
  };
  visit(parse(source, { sourceType: 'module', plugins: ['jsx'] }));
  return elements;
}
const currentElements = meaningfulElements(fs.readFileSync(paths[0], 'utf8'));
const originalElements = meaningfulElements(fs.readFileSync(`scratch/linked-traversal-extension-original/${paths[0]}`, 'utf8'));
for (const element of originalElements) assert(currentElements.includes(element), 'Original teaching element missing');

function* products(alphabet, length, prefix = []) {
  if (!length) { yield prefix; return; }
  for (const value of alphabet) yield* products(alphabet, length - 1, [...prefix, value]);
}
const cycles = [];
for (let size = 0; size <= 4; size += 1) {
  const alphabet = [null, ...Array.from({ length: size }, (_, index) => index)];
  for (const next of products(alphabet, size)) for (const head of alphabet) cycles.push({ next, head, actual: linked.linkedCycleTrace(next, head) });
}
for (let size = 5; size <= 32; size += 1) for (const entry of [-1, 0, Math.floor(size / 2), size - 1]) {
  const next = linked.chainSuccessors(size, entry);
  cycles.push({ next, head: 0, actual: linked.linkedCycleTrace(next, 0) });
}
const arrays = [];
for (let size = 0; size <= 7; size += 1) for (const values of products([0, 1, 2], size)) {
  const strict = monotone.nextGreaterTrace(values);
  const inclusive = monotone.nextGreaterTrace(values, true);
  const histogram = monotone.histogramState(values);
  for (const [trace, equality] of [[strict, false], [inclusive, true]]) {
    assert(trace.pushes === size && trace.pops <= size);
    for (const frame of trace.frames) {
      assert(frame.stack.every((index, offset) => offset === 0 || (index > frame.stack[offset - 1] && (equality ? values[index] < values[frame.stack[offset - 1]] : values[index] <= values[frame.stack[offset - 1]]))));
    }
  }
  for (const trace of [histogram.leftTrace, histogram.rightTrace]) for (const frame of trace.frames) assert(frame.stack.every((index, offset) => offset === 0 || values[index] > values[frame.stack[offset - 1]]));
  arrays.push({ values, strict: strict.distances, inclusive: inclusive.distances, left: histogram.leftTrace.boundaries, right: histogram.rightTrace.boundaries, area: histogram.area, witness: histogram.best });
}
const signed = [...products([-2, 0, 2], 5)].map(values => ({ values, strict: monotone.nextGreaterTrace(values).distances, inclusive: monotone.nextGreaterTrace(values, true).distances }));
const middles = [];
for (let length = 0; length <= 12; length += 1) for (const policy of ['first','second']) for (const cut of [false,true]) middles.push(linked.middleSplitState(length, policy, cut));
let rejected = 0;
for (const attempt of [
  () => linked.linkedCycleTrace([,null]), () => linked.linkedCycleTrace([2]), () => linked.linkedCycleTrace([undefined]), () => linked.linkedCycleTrace([], 0),
  () => linked.chainSuccessors(0,0), () => linked.chainSuccessors(2.5,-1), () => linked.middleSplitState(5,'unknown'),
  () => monotone.nextGreaterTrace([,1]), () => monotone.nextGreaterTrace([NaN]), () => monotone.nextGreaterTrace([1], 'yes'),
  () => monotone.histogramState([-1]), () => monotone.histogramState([Infinity]), () => monotone.histogramState([1.5]),
  () => monotone.parseStackValues('1,,2'), () => monotone.parseStackValues('1e2'), () => monotone.parseStackValues('21'), () => monotone.parseStackValues('-1',true),
]) { assert.throws(attempt); rejected += 1; }
assert.deepEqual(monotone.parseStackValues(''), []);
assert.deepEqual(monotone.parseStackValues('-2, 0, 2'), [-2,0,2]);
const payload = { sources: fingerprints(), cycles, arrays, signed, middles, rejected, originalTeachingElements: originalElements.length, programs: { ...linkedTraversalExamples, ...monotonicStackExamples }, originalPrograms: linkedExamples, practice: problems.map(row => ({ number: row.number, slug: row.slug, difficulty: row.difficulty })) };
fs.writeFileSync(`${directory}/payload.json`, JSON.stringify(payload));
const result = spawnSync(process.env.LESSON_PYTHON || 'scratch/lesson-tools/Scripts/python.exe', ['scripts/verify-linked-traversal-extension.py'], { encoding: 'utf8' });
if (result.status !== 0) { console.error(result.stdout, result.stderr); process.exit(result.status || 1); }
assert.deepEqual(payload.sources, fingerprints());
console.log(result.stdout);
