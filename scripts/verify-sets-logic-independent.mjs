import assert from 'node:assert/strict';
import fs from 'node:fs';
import * as model from '../src/learn/data/sets-logic-models.js';
import { setsLogicExamples } from '../src/learn/data/sets-logic-examples.js';

const directory = 'scratch/sets-logic-independent';
fs.mkdirSync(directory, { recursive: true });
const relations = [];
for (const size of [5, 6]) for (let seed = 0; seed < 48; seed++) {
  const permutation = Array.from({ length: size }, (_, i) => (size - 1 - i + seed) % size);
  const edges = [];
  for (let i = 0; i < size; i++) for (let j = i + 1; j < size; j++) {
    if ((seed * 19 + i * 23 + j * 11) % 7 < 3) edges.push([permutation[i], permutation[j]]);
  }
  // Generate an order by reachability, then verify against a separate Python oracle.
  const pairs = Array.from({ length: size }, (_, i) => [i, i]);
  for (let start = 0; start < size; start++) {
    const reached = new Set([start]);
    const frontier = [start];
    while (frontier.length) {
      const from = frontier.pop();
      for (const [a, b] of edges) if (a === from && !reached.has(b)) { reached.add(b); frontier.push(b); }
    }
    for (const target of reached) if (target !== start) pairs.push([start, target]);
  }
  relations.push({ size, pairs, result: model.inspectRelation(size, pairs) });
}
for (let seed = 0; seed < 36; seed++) {
  const size = 6;
  const pairs = [];
  for (let i = 0; i < size; i++) for (let j = 0; j < size; j++) if ((seed + 3 * i + 5 * j + i * j) % 11 < 5) pairs.push([i, j]);
  relations.push({ size, pairs, result: model.inspectRelation(size, pairs) });
}
const diagonals = [];
for (const size of [5, 6, 7, 8]) for (let seed = 0; seed < 16; seed++) {
  const matrix = Array.from({ length: size }, (_, i) => Array.from({ length: size }, (_, j) => (seed + 3 * i + 7 * j + i * j) % 9 < 4));
  diagonals.push({ matrix, result: model.diagonalSubset(matrix) });
}
const malformed = [
  () => model.inspectQuantifiers([[true, , false], [false, true, false], [false, false, true]], 3, 3),
  () => model.inspectQuantifiers([, [false, true, false], [false, false, true]], 0, 0),
  () => model.diagonalSubset([[, false], [false, true]]),
  () => model.diagonalSubset([, [false, true]]),
  () => model.inspectRelation(2, [[0, ,]]),
  () => model.inspectRelation(2, [[, 0]]),
];
const inputResults = malformed.map((run, index) => {
  try { run(); return { index, rejected: false }; } catch (error) { return { index, rejected: true, type: error.name }; }
});
fs.writeFileSync(directory + '/fixtures.json', JSON.stringify({ relations, diagonals, examples: setsLogicExamples, inputResults }));
assert(inputResults.every(result => result.rejected), JSON.stringify(inputResults));
console.log(`Independent exports: ${relations.length} larger relations, ${diagonals.length} larger diagonal boards, 6 malformed-input regressions.`);
