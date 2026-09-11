import assert from 'node:assert/strict';
import fs from 'node:fs';
import * as model from '../src/learn/data/sets-logic-models.js';
import { setsLogicExamples } from '../src/learn/data/sets-logic-examples.js';

const counts = { setStates: 0, argumentStates: 0, quantifierStates: 0, relations: 0, diagonalStates: 0, squareStates: 0, rejected: 0, sparseInputsRejected: 0 };
const range = size => Array.from({ length: size }, (_, index) => index);
const universe = new Set(model.rosterNames);
for (let first = 0; first < 16; first++) for (let second = 0; second < 16; second++) {
  const a = new Set(model.rosterNames.filter((_, index) => first & 2 ** index));
  const b = new Set(model.rosterNames.filter((_, index) => second & 2 ** index));
  const expected = {
    intersection: [...a].filter(x => b.has(x)), union: [...new Set([...a, ...b])],
    difference: [...a].filter(x => !b.has(x)), reverseDifference: [...b].filter(x => !a.has(x)),
    complement: [...universe].filter(x => !a.has(x)),
    symmetricDifference: [...new Set([...a, ...b])].filter(x => !(a.has(x) && b.has(x))),
  };
  for (const [operation, names] of Object.entries(expected)) {
    assert.deepEqual([...model.selectRoster(first, second, operation).selected].sort(), names.sort());
    counts.setStates++;
  }
}
// A truth set is a subset of the four assignments, allowing validity to be
// checked by containment rather than repeating the evaluator's switches.
const truthSets = { p: [2, 3], q: [1, 3], notP: [0, 1], notQ: [0, 2], implication: [0, 1, 3], converse: [0, 2, 3], contrapositive: [0, 1, 3], equivalence: [0, 3], conjunction: [3], disjunction: [1, 2, 3], exclusiveOr: [1, 2] };
const premiseOptions = ['p', 'q', 'notP', 'notQ', 'implication'];
for (let mask = 0; mask < 32; mask++) for (const conclusion of Object.keys(truthSets)) {
  const premises = premiseOptions.filter((_, index) => mask & 2 ** index);
  const admitted = range(4).filter(world => premises.every(name => truthSets[name].includes(world)));
  const rejected = admitted.filter(world => !truthSets[conclusion].includes(world));
  const result = model.inspectArgument(premises, conclusion);
  assert.equal(result.valid, rejected.length === 0);
  assert.equal(result.consistent, admitted.length > 0);
  assert.deepEqual(result.counterexamples.map(world => 2 * Number(world.p) + Number(world.q)), rejected);
  counts.argumentStates++;
}
for (let mask = 0; mask < 512; mask++) {
  const matrix = range(3).map(row => range(3).map(column => Boolean(mask & 2 ** (row * 3 + column))));
  for (let rows = 0; rows <= 3; rows++) for (let columns = 0; columns <= 3; columns++) {
    const result = model.inspectQuantifiers(matrix, rows, columns);
    const rowSums = range(rows).map(row => matrix[row].slice(0, columns).filter(Boolean).length);
    const columnSums = range(columns).map(column => range(rows).filter(row => matrix[row][column]).length);
    assert.equal(result.eachHasSomeone, rowSums.filter(sum => sum === 0).length === 0);
    assert.equal(result.someoneCoversAll, columnSums.filter(sum => sum === rows).length > 0);
    assert.equal(result.everyoneCoversAll, rowSums.reduce((a, b) => a + b, 0) === rows * columns);
    assert.equal(result.someoneCoversSomething, rowSums.reduce((a, b) => a + b, 0) > 0);
    for (let row = 0; row < rows; row++) assert.deepEqual(result.rowWitnesses[row], range(columns).filter(column => matrix[row][column]));
    for (let column = 0; column < columns; column++) {
      if (result.columnFailures[column] === null) assert.equal(columnSums[column], rows);
      else assert.equal(matrix[result.columnFailures[column]][column], false);
    }
    if (result.someoneCoversAll) assert(result.eachHasSomeone);
    counts.quantifierStates++;
  }
}
// Independent relation oracle: compose pair sets; recover equivalence classes
// from undirected components, and cover closure from reachability.
for (let size = 0; size <= 4; size++) for (let mask = 0; mask < 2 ** (size * size); mask++) {
  const nodes = range(size);
  const pairs = nodes.flatMap(a => nodes.flatMap(b => mask & 2 ** (a * size + b) ? [[a, b]] : []));
  const pairSet = new Set(pairs.map(([a, b]) => `${a},${b}`));
  const has = (a, b) => pairSet.has(`${a},${b}`);
  const composed = new Set();
  for (const [a, b] of pairs) for (const [c, d] of pairs) if (b === c) composed.add(`${a},${d}`);
  const result = model.inspectRelation(size, pairs);
  const properties = {
    reflexive: nodes.filter(x => !has(x, x)).length === 0,
    symmetric: pairs.filter(([a, b]) => !has(b, a)).length === 0,
    transitive: [...composed].filter(pair => !pairSet.has(pair)).length === 0,
    antisymmetric: pairs.filter(([a, b]) => a !== b && has(b, a)).length === 0,
  };
  assert.deepEqual(result.properties, properties);
  for (const [property, failure] of Object.entries(result.failures)) {
    assert.equal(failure === null, properties[property]);
    if (!failure) continue;
    const [a, b, c] = failure;
    if (property === 'reflexive') assert(!has(a, a));
    if (property === 'symmetric') assert(has(a, b) && !has(b, a));
    if (property === 'transitive') assert(has(a, b) && has(b, c) && !has(a, c));
    if (property === 'antisymmetric') assert(a !== b && has(a, b) && has(b, a));
  }
  if (result.equivalence) {
    assert.deepEqual(result.classes.flat().sort(), nodes);
    for (const group of result.classes) for (const a of nodes) for (const b of group) assert.equal(group.includes(a), has(a, b));
  } else assert.equal(result.classes, null);
  if (result.partialOrder) {
    const reach = nodes.map(a => nodes.map(b => a === b || result.covers.some(([c, d]) => c === a && d === b)));
    for (const k of nodes) for (const i of nodes) for (const j of nodes) reach[i][j] ||= reach[i][k] && reach[k][j];
    assert.deepEqual(reach, result.matrix);
    for (const [a, b] of result.covers) { assert(result.levels[b] > result.levels[a]); assert(!nodes.some(c => c !== a && c !== b && has(a, c) && has(c, b))); }
    assert.deepEqual(result.minimal, nodes.filter(a => pairs.filter(([b, c]) => c === a && b !== a).length === 0));
    assert.deepEqual(result.maximal, nodes.filter(a => pairs.filter(([b, c]) => b === a && c !== a).length === 0));
    assert.deepEqual(result.least, nodes.filter(a => pairs.filter(([b]) => b === a).length === size));
    assert.deepEqual(result.greatest, nodes.filter(a => pairs.filter(([, b]) => b === a).length === size));
  } else assert.equal(result.covers, null);
  counts.relations++;
}
for (let size = 0; size <= 3; size++) for (let mask = 0; mask < 2 ** (size * size); mask++) {
  const matrix = range(size).map(row => range(size).map(column => Boolean(mask & 2 ** (row * size + column))));
  const result = model.diagonalSubset(matrix);
  for (let index = 0; index < size; index++) {
    assert.notEqual(result.subset[index], matrix[index][index]);
    assert.notDeepEqual(result.subset, matrix[index]);
  }
  counts.diagonalStates++;
}
for (let size = 0; size <= 7; size++) {
  const result = model.oddSquareStep(size);
  assert.equal(new Set(result.cells.map(cell => `${cell.row},${cell.column}`)).size, (size + 1) ** 2);
  assert.equal(result.cells.filter(cell => cell.added).length, 2 * size + 1);
  assert.equal(result.cells.filter(cell => !cell.added).length, size ** 2);
  counts.squareStates++;
}
for (const run of [() => model.selectRoster(-1, 0, 'union'), () => model.selectRoster(0, 0, 'invalid'), () => model.evaluateProposition('p', 1, true), () => model.inspectArgument(['unknown'], 'p'), () => model.inspectQuantifiers([], 0, 0), () => model.inspectRelation(2, [[0, 2]]), () => model.inspectRelation(0, [[0, 0]]), () => model.oddSquareStep(8), () => model.diagonalSubset([[1]])]) {
  assert.throws(run); counts.rejected++;
}
// Array.some/forEach skip holes, so validating values through those callbacks
// alone does not establish a dense Boolean board or a complete ordered pair.
const sparseCases = [
  () => model.inspectQuantifiers([[true, , false], [false, true, false], [false, false, true]], 3, 3),
  () => model.inspectQuantifiers([, [false, true, false], [false, false, true]], 0, 0),
  () => model.inspectQuantifiers(Array(3), 0, 0),
  () => model.diagonalSubset([[, false], [false, true]]),
  () => model.diagonalSubset([[true, false], ,]),
  () => model.diagonalSubset(Array(2)),
  () => model.inspectRelation(2, [[0, ,]]),
  () => model.inspectRelation(2, [[, 1]]),
  () => model.inspectRelation(2, [Array(2)]),
  () => model.inspectRelation(2, Array(1)),
];
for (const run of sparseCases) {
  assert.throws(run, TypeError);
  counts.sparseInputsRejected++;
}
fs.mkdirSync('scratch/sets-logic-verification', { recursive: true });
fs.writeFileSync('scratch/sets-logic-verification/examples.json', JSON.stringify(setsLogicExamples));
fs.writeFileSync('scratch/sets-logic-verification/model-results.json', JSON.stringify({ checkedAt: new Date().toISOString(), passed: true, counts }, null, 2));
console.log(JSON.stringify(counts));
