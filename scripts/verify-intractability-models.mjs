import assert from 'node:assert/strict';
import { formulaPresets, parseFormula, evaluateFormula, truthTable, reduceFormulaToClique, inspectSelection, firstCliqueChoices, encodingCounts, parseCoverEdges, matchingCover, exactCover, minimumCover, isCover } from '../src/learn/data/intractability-models.js';
let randomState = 93271;
function integer(limit) {
  randomState = 1664525 * randomState + 1013904223 >>> 0;
  return randomState % limit;
}
function subsets(n) {
  return Array.from({
    length: 2 ** n
  }, (_, mask) => Array.from({
    length: n
  }, (_, index) => index).filter(index => (mask & 1 << index) !== 0));
}
function assignmentOracle(formula) {
  return subsets(3).map(chosen => [0, 1, 2].map(i => chosen.includes(i))).filter(bits => !formula.some(clause => clause.every(lit => bits[Math.abs(lit) - 1] !== lit > 0)));
}
const formulas = [...Object.values(formulaPresets), [], [[]], [[1], []], [[1, 1], [-1, -1]], [[1, -1]]];
for (let sample = 0; sample < 350; sample += 1) {
  formulas.push(Array.from({
    length: integer(4)
  }, () => Array.from({
    length: integer(4)
  }, () => (integer(2) ? 1 : -1) * (1 + integer(3)))));
}
let cliqueWitnesses = 0;
for (const formula of formulas) {
  const before = JSON.stringify(formula);
  const truth = assignmentOracle(formula);
  const graph = reduceFormulaToClique(formula);
  const edges = new Set(graph.edges.map(pair => [...pair].sort().join('|')));
  const allCliques = subsets(graph.vertices.length).filter(chosen => chosen.length === graph.target && chosen.every((u, i) => chosen.slice(i + 1).every(v => edges.has([graph.vertices[u].id, graph.vertices[v].id].sort().join('|')))));
  assert.equal(allCliques.length > 0, truth.length > 0);
  const returned = firstCliqueChoices(formula);
  assert.equal(returned !== null, truth.length > 0);
  const table = truthTable(formula);
  assert.equal(table.filter(row => row.satisfied).length, truth.length);
  for (const row of table) assert.equal(row.satisfied, truth.some(bits => JSON.stringify(bits) === JSON.stringify(row.assignment)));
  for (const clique of allCliques) {
    const selected = formula.map(() => null);
    for (const index of clique) {
      const vertex = graph.vertices[index];
      assert.equal(selected[vertex.clause], null);
      selected[vertex.clause] = vertex.id;
    }
    const recovered = inspectSelection(formula, selected);
    assert.equal(recovered.clique, true);
    assert.ok(truth.some(bits => JSON.stringify(bits) === JSON.stringify(recovered.assignment)));
    cliqueWitnesses += 1;
  }
  assert.equal(JSON.stringify(formula), before, 'Input mutation');
  for (const [a, b] of graph.edges) {
    const first = graph.vertices.find(v => v.id === a),
      second = graph.vertices.find(v => v.id === b);
    assert.notEqual(first.clause, second.clause);
    assert.notEqual(first.literal, -second.literal);
  }
}
const possibleEdges = [];
for (let u = 0; u < 6; u += 1) for (let v = u + 1; v < 6; v += 1) possibleEdges.push([u, v]);
const graphs = subsets(10).map(chosen => {
  const fiveEdges = possibleEdges.filter(([u, v]) => u < 5 && v < 5);
  return chosen.map(index => fiveEdges[index]);
});
for (let sample = 0; sample < 260; sample += 1) graphs.push(possibleEdges.filter(() => integer(3) === 0));
for (const edges of graphs) {
  const before = JSON.stringify(edges);
  const covers = subsets(6).filter(chosen => edges.every(([u, v]) => chosen.includes(u) || chosen.includes(v)));
  const optimum = Math.min(...covers.map(chosen => chosen.length));
  const approx = matchingCover(edges);
  assert.ok(isCover(edges, approx.selected));
  assert.ok(approx.selected.length <= 2 * optimum);
  assert.equal(new Set(approx.matching.flatMap(index => edges[index])).size, approx.matching.length * 2);
  assert.ok(approx.matching.length <= optimum);
  assert.deepEqual(approx.states.at(-1).selected, approx.selected);
  for (const snapshot of approx.states) {
    assert.equal(snapshot.selected.length, 2 * snapshot.matching.length);
    assert.equal(new Set(snapshot.matching.flatMap(index => edges[index])).size, snapshot.selected.length);
  }
  for (let budget = 0; budget <= 6; budget += 1) {
    const exact = exactCover(edges, budget);
    assert.equal(exact.selected !== null, optimum <= budget);
    assert.ok(exact.events.length <= 2 ** (budget + 1) - 1);
    if (exact.selected !== null) {
      assert.ok(isCover(edges, exact.selected));
      assert.ok(exact.selected.length <= budget);
      assert.equal(new Set(exact.selected).size, exact.selected.length);
    }
    for (const event of exact.events) {
      assert.equal(event.left + event.selected.length, budget);
      assert.deepEqual(event.remaining, edges.filter(edge => edge.every(vertex => !event.selected.includes(vertex))));
    }
  }
  assert.equal(minimumCover(edges).length, optimum);
  assert.equal(JSON.stringify(edges), before);
}
assert.deepEqual(parseFormula('1 -2\n3'), [[1, -2], [3]]);
for (const invalid of ['', '0', '4', '1 2 3 1', '1\n2\n3\n1', '1e0', '1,'.repeat(200)]) assert.throws(() => parseFormula(invalid));
assert.throws(() => evaluateFormula([[4]], [true, true, true]));
assert.throws(() => evaluateFormula([[1]], [1, false, false]));
assert.throws(() => inspectSelection([[1]], ['1:0']));
assert.deepEqual(parseCoverEdges('a b\nB A\nc d'), [[0, 1], [2, 3]]);
assert.deepEqual(parseCoverEdges(''), []);
for (const invalid of ['A A', 'G B', 'AB', 'A B C', 'A'.repeat(401)]) assert.throws(() => parseCoverEdges(invalid));
assert.throws(() => exactCover([], -1));
assert.throws(() => exactCover([], NaN));
assert.throws(() => matchingCover([[0, 1], [1, 0]]));
const saved = matchingCover([[0, 1], [2, 3]]);
saved.states[0].selected.push(5);
assert.deepEqual(saved.states.at(-1).selected, [0, 1, 2, 3]);
for (let exponent = 1; exponent <= 40; exponent += 1) {
  const count = encodingCounts(exponent);
  assert.equal(BigInt(`0b${count.binary}`), 1n << BigInt(exponent));
  assert.equal(count.bits, exponent + 1);
  assert.equal(BigInt(count.slots), BigInt(count.target) + 1n);
}
for (const invalid of [0, 41, 1.5, Infinity, NaN]) assert.throws(() => encodingCounts(invalid));
console.log(`Verified ${formulas.length} formulas, ${cliqueWitnesses} clique-to-assignment witnesses, ${graphs.length} exhaustive/random graph families at seven budgets, matching bounds, encoding counts and input/snapshot contracts.`);
