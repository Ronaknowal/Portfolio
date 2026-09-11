import assert from 'node:assert/strict';
import fs from 'node:fs';
import * as model from '../src/learn/data/category-theory-models.js';

const output = { compositions: [], naturality: [], products: [], probability: [], tangents: [], adjunctions: [] };
const names = Object.keys(model.finiteFunctionChoices);
for (const f of names) for (const g of names) for (const h of names) for (let input = 0; input < 3; input++) {
  output.compositions.push({ input, names: [f, g, h], ...model.compositionSnapshot(input, f, g, h) });
}
function tuples(size, length) {
  if (!length) return [[]];
  return tuples(size, length - 1).flatMap(values => Array.from({ length: size }, (_, value) => [...values, value]));
}
for (let length = 0; length <= 4; length++) for (const values of tuples(3, length)) {
  for (const operation of ['reverse', 'sort']) for (const mapping of names) {
    output.naturality.push({ values, operation, name: mapping, ...model.naturalitySnapshot(values, operation, mapping) });
  }
}
for (const mode of ['complete', 'missing', 'duplicate']) for (const row of [0, 1]) for (const column of [0, 1]) {
  output.products.push({ mode, ...model.productSnapshot(mode, row, column) });
}
for (let percent = 0; percent <= 100; percent++) output.probability.push({ percent, ...model.stochasticCopySnapshot(percent) });
for (const choice of Object.keys(model.tangentFunctionChoices)) for (let tenth = -30; tenth <= 30; tenth++) {
  for (const v of [-2, -1, 0, 1, 2]) for (const w of [-2, 0, 2]) {
    output.tangents.push({ choice, tenth, ...model.tangentSnapshot(tenth / 10, v, choice, w) });
  }
}
for (const mapping of tuples(3, 4)) for (let sourceMask = 0; sourceMask < 16; sourceMask++) for (let targetMask = 0; targetMask < 8; targetMask++) {
  output.adjunctions.push({ sourceMask, targetMask, ...model.adjunctionSnapshot(sourceMask, targetMask, mapping) });
}
const invalid = [
  () => model.validateFiniteMap([0, , 1], 3, 2),
  () => model.validateFiniteMap([0], 1, 0),
  () => model.validateFiniteMap([0], 2, 1),
  () => model.validateFiniteMap([NaN], 1, 2),
  () => model.compositionSnapshot(-1),
  () => model.compositionSnapshot(0, 'toString'),
  () => model.schemaSnapshot(0, [0, 2, 0]),
  () => model.naturalitySnapshot(Array(9).fill(0)),
  () => model.productSnapshot('other'),
  () => model.stochasticCopySnapshot(100.1),
  () => model.tangentSnapshot(Infinity),
  () => model.tangentSnapshot(0, 3),
  () => model.adjunctionSnapshot(16),
  () => model.multiplyStochastic([[0.2, 0.2]], [[1], [1]]),
  () => model.multiplyStochastic([[1]], [[1], [1]]),
];
invalid.forEach(check => assert.throws(check));
assert.deepEqual(model.validateFiniteMap([], 0, 0), []);
assert.deepEqual(model.composeFiniteMaps([], [], 0), []);
assert.deepEqual(model.pullbackPairs([0, 0, 1], [0, 1], 2), [[0, 0], [1, 0], [2, 1]]);
for (const first of [[[1, 0], [0, 1]], [[0.25, 0.75], [0.5, 0.5]]]) {
  assert.deepEqual(model.multiplyStochastic(first, [[1, 0], [0, 1]]), first);
}
fs.mkdirSync('scratch/category-theory-verification', { recursive: true });
fs.writeFileSync('scratch/category-theory-verification/model-fixtures.json', JSON.stringify(output));
console.log(JSON.stringify({ exported: Object.fromEntries(Object.entries(output).map(([key, values]) => [key, values.length])), invalidCases: invalid.length }));
