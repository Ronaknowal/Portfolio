import assert from 'node:assert/strict';
import { spawnSync } from 'node:child_process';
import { mkdirSync, writeFileSync } from 'node:fs';
import { numpyFoundationsExamples } from '../src/learn/data/numpy-foundations-examples.js';
import { numpyReferenceExamples } from "../src/learn/data/numpy-reference-examples.js";
import { selectionCases, selectionResult, memoryState, broadcastCases, broadcastResult, reductionResult, product } from '../src/learn/data/numpy-foundations-model.js';

const python = process.env.NUMPY_PYTHON || 'C:/Users/ronak/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe';
const memories = ['view', 'copy', 'advanced'].flatMap(kind => [null, 0, 1, 2].map(index => ({ kind, index })));
const shapes = [[], [1], [2], [3], [0], [1, 1], [3, 1], [1, 2], [3, 2], [0, 2], [2, 1, 3], [1, 3]];
const generated = shapes.flatMap(aShape => shapes.map(bShape => ({ aShape, bShape,
  a: Array.from({ length: product(aShape) }, (_, i) => i + 7), b: Array.from({ length: product(bShape) }, (_, i) => i + 2) })));
const broadcasts = [...broadcastCases, ...generated];
const reductions = [0, 1].flatMap(axis => [false, true].map(keepdims => ({ axis, keepdims })));
const oracle = spawnSync(python, ['scripts/numpy-foundations-oracle.py'], { input: JSON.stringify({ selections: selectionCases, memory: memories, broadcast: broadcasts, reductions }), encoding: 'utf8' });
assert.equal(oracle.status, 0, oracle.stderr);
const actual = JSON.parse(oracle.stdout);
selectionCases.forEach((item, i) => {
  const result = selectionResult(item.id);
  for (const key of ['shape', 'values', 'source']) assert.deepEqual(result[key], actual.selections[i][key], `${item.id} ${key}`);
});
memories.forEach((item, i) => {
  const result = memoryState(item.kind, item.index);
  for (const key of ['original', 'picked', 'shares', 'offsets']) assert.deepEqual(result[key], actual.memory[i][key], `${item.kind} ${item.index} ${key}`);
});
broadcasts.forEach((item, i) => {
  const result = broadcastResult(item), expected = actual.broadcast[i];
  assert.equal(result.valid, expected.valid, `compatible ${JSON.stringify([item.aShape, item.bShape])}`);
  if (result.valid) {
    for (const key of ['shape', 'values']) assert.deepEqual(result[key], expected[key]);
    assert.deepEqual(result.cells.map(cell => cell.aIndex), expected.aIndices);
    assert.deepEqual(result.cells.map(cell => cell.bIndex), expected.bIndices);
  }
});
reductions.forEach((item, i) => {
  const result = reductionResult(item.axis, item.keepdims);
  for (const key of ['shape', 'values']) assert.deepEqual(result[key], actual.reductions[i][key]);
});
const retained = ['numpyMissing', 'numpySort', 'numpyAlgebra', 'numpyRandomIO', 'numpyProject'];
const examples = { ...numpyFoundationsExamples, ...Object.fromEntries(retained.map(name => [name, numpyReferenceExamples[name]])) };
const exampleResults = [];
for (const [name, example] of Object.entries(examples)) {
  const run = spawnSync(python, ['-c', example.code], { encoding: 'utf8' });
  assert.equal(run.status, 0, `${name}: ${run.stderr}`);
  assert.equal(run.stdout.replaceAll('\r\n', '\n').trim(), example.output.trim(), name);
  exampleResults.push({ name, status: 'passed', stderr: run.stderr.trim() });
}
const report = { checkedAt: new Date().toISOString(), runtime: actual.runtime, models: {
  selections: selectionCases.length, memoryStates: memories.length, broadcasts: broadcasts.length, reductions: reductions.length,
  allShapesAndOperandIndicesAgreeWithNumPy: true }, examples: exampleResults };
mkdirSync('scratch/numpy-foundations', { recursive: true });
writeFileSync('scratch/numpy-foundations/runtime-results.json', JSON.stringify(report, null, 2) + '\n');
console.log(JSON.stringify(report, null, 2));
