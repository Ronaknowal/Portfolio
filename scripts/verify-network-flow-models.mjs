import assert from 'node:assert/strict';
import { FLOW_EDGES, solveFlow, inspectFlow, nextAugmentation, residualArcs, cutCapacity, solveMatching, solvePixelCut, pixelEnergy } from '../src/learn/data/network-flow-models.js';
function minimumCut(n, edges, source = 0, sink = n - 1) {
  let best = Infinity;
  for (let mask = 0; mask < 2 ** n; mask++) {
    if (!(mask & 1 << source) || mask & 1 << sink) continue;
    const capacity = edges.reduce((sum, [u, v, cap]) => sum + (mask & 1 << u && !(mask & 1 << v) ? cap : 0), 0);
    best = Math.min(best, capacity);
  }
  return best;
}
function verifyFlow(n, edges) {
  const frozen = JSON.stringify(edges);
  const result = solveFlow(n, edges);
  assert.equal(result.value, minimumCut(n, edges));
  assert.equal(result.cut.capacity, result.value);
  assert.equal(JSON.stringify(edges), frozen);
  for (const flows of result.states) {
    const balances = Array(n).fill(0);
    edges.forEach(([u, v, cap], i) => {
      assert.ok(flows[i] >= 0 && flows[i] <= cap);
      balances[u] -= flows[i];
      balances[v] += flows[i];
    });
    assert.ok(balances.slice(1, -1).every(value => value === 0));
    assert.equal(balances[0] + balances[n - 1], 0);
  }
  result.steps.forEach((step, index) => {
    const original = result.states[index];
    const after = original.slice();
    let vertex = 0;
    assert.equal(step.delta, Math.min(...step.path.map(arc => arc.residual)));
    for (const arc of step.path) {
      assert.equal(arc.from, vertex);
      vertex = arc.to;
      const [u, v, capacity] = edges[arc.edgeId];
      assert.deepEqual([arc.from, arc.to], arc.direction === 1 ? [u, v] : [v, u]);
      assert.equal(arc.residual, arc.direction === 1 ? capacity - original[arc.edgeId] : original[arc.edgeId]);
      after[arc.edgeId] += arc.direction * step.delta;
    }
    assert.equal(vertex, n - 1);
    assert.deepEqual(after, result.states[index + 1]);
    assert.deepEqual(original, step.before);
  });
  return result;
}
let graphCases = 0;
const arcs = [[0, 1], [0, 2], [1, 0], [1, 2], [2, 0], [2, 1]];
for (let code = 0; code < 3 ** 6; code++) {
  let digits = code;
  const edges = arcs.map(([u, v]) => {
    const capacity = digits % 3;
    digits = Math.floor(digits / 3);
    return [u, v, capacity];
  });
  verifyFlow(3, edges);
  graphCases++;
}
let seed = 73429;
const random = upper => {
  seed = Math.imul(seed, 1664525) + 1013904223 >>> 0;
  return seed % upper;
};
for (let trial = 0; trial < 400; trial++) {
  const n = 2 + random(6);
  const edges = Array.from({
    length: random(21)
  }, () => [random(n), random(n), random(6)]);
  verifyFlow(n, edges);
  graphCases++;
}
const defaultResult = verifyFlow(6, FLOW_EDGES);
assert.deepEqual(defaultResult.flows, [1, 1, 0, 1, 1, 1, 1]);
assert.deepEqual(defaultResult.steps[1].path.map(arc => [arc.edgeId, arc.direction]), [[1, 1], [4, 1], [2, -1], [3, 1], [6, 1]]);
assert.deepEqual(defaultResult.states[1], [1, 0, 1, 0, 0, 1, 0]);
assert.equal(inspectFlow(6, FLOW_EDGES, [1, 0, 0, 0, 0, 0, 0]).feasible, false);
assert.equal(inspectFlow(6, FLOW_EDGES, [2, 0, 2, 0, 0, 2, 0]).feasible, false);
assert.deepEqual(residualArcs([[0, 1, 5], [1, 0, 4]], [3, 1]).map(arc => [arc.from, arc.to, arc.residual]), [[0, 1, 2], [1, 0, 3], [1, 0, 3], [0, 1, 1]]);
assert.throws(() => nextAugmentation(6, FLOW_EDGES, [1, 0, 0, 0, 0, 0, 0]));
assert.throws(() => solveFlow(2, [[0, 1, 0.5]]));
assert.throws(() => solveFlow(2, [[0, 1, NaN]]));
assert.throws(() => solveFlow(2, [], 0, 0));
assert.throws(() => cutCapacity(2, [[0, 1, 1]], [0, 1]));
for (let mask = 0; mask < 512; mask++) {
  const matrix = Array.from({
    length: 3
  }, (_, left) => Array.from({
    length: 3
  }, (_, right) => Boolean(mask & 1 << left * 3 + right)));
  const snapshot = JSON.stringify(matrix);
  const result = solveMatching(matrix);
  let maximum = 0;
  for (let code = 0; code < 64; code++) {
    let digits = code;
    const chosen = [];
    let valid = true;
    for (let left = 0; left < 3; left++) {
      const right = digits % 4 - 1;
      digits = Math.floor(digits / 4);
      if (right !== -1) {
        if (!matrix[left][right] || chosen.includes(right)) valid = false;
        chosen.push(right);
      }
    }
    if (valid) maximum = Math.max(maximum, chosen.length);
  }
  let minimumCover = 6;
  for (let chosen = 0; chosen < 64; chosen++) {
    if (matrix.every((row, left) => row.every((allowed, right) => !allowed || chosen & 1 << left || chosen & 1 << right + 3))) {
      minimumCover = Math.min(minimumCover, chosen.toString(2).replaceAll('0', '').length);
    }
  }
  assert.equal(result.matching.length, maximum);
  assert.equal(result.coverLeft.length + result.coverRight.length, minimumCover);
  assert.ok(matrix.every((row, left) => row.every((allowed, right) => !allowed || result.coverLeft.includes(left) || result.coverRight.includes(right))));
  assert.equal(new Set(result.matching.map(pair => pair[0])).size, maximum);
  assert.equal(new Set(result.matching.map(pair => pair[1])).size, maximum);
  const neighbors = [0, 1, 2].filter(right => result.reachableLeft.some(left => matrix[left][right]));
  assert.deepEqual(neighbors, result.reachableRight);
  assert.equal(result.deficiency, 3 - maximum);
  assert.equal(JSON.stringify(matrix), snapshot);
}
for (let penalty = 0; penalty <= 5; penalty++) {
  const result = solvePixelCut(penalty);
  let optimum = Infinity;
  for (let mask = 0; mask < 64; mask++) {
    const labels = Array.from({
      length: 6
    }, (_, i) => Boolean(mask & 1 << i));
    const unary = labels.reduce((sum, label, i) => sum + (label ? [6, 4, 0, 5, 1, 0][i] : [0, 1, 6, 0, 5, 6][i]), 0);
    let boundaries = 0;
    for (let row = 0; row < 2; row++) for (let column = 0; column < 3; column++) {
      const i = row * 3 + column;
      if (column < 2 && labels[i] !== labels[i + 1]) boundaries++;
      if (row < 1 && labels[i] !== labels[i + 3]) boundaries++;
    }
    assert.equal(pixelEnergy(labels, penalty).total, unary + penalty * boundaries);
    optimum = Math.min(optimum, unary + penalty * boundaries);
  }
  assert.equal(result.energy.total, optimum);
  assert.equal(result.flow.value, optimum);
}
console.log(`Network-flow models: ${graphCases + 1} flow/cut cases, 512 complete matching/cover/shortage cases, 384 binary label energies; traces, immutable inputs and invalid contracts passed.`);
