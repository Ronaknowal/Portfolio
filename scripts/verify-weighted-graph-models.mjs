import assert from 'node:assert/strict';
import { parseWeightedEdges, dijkstraTrace, bellmanFordTrace, recoverRoute, spanningForestTrace, dependencyState, directedCycle, criticalSchedule, ROUTE_TEXT, NEGATIVE_TEXT, FOREST_TEXT, DEPENDENCY_TEXT, JOB_DURATIONS } from '../src/learn/data/weighted-graph-models.js';
let seed = 71534;
function random(limit) {
  seed = Math.imul(seed, 1664525) + 1013904223 >>> 0;
  return seed % limit;
}
function edgesOf(triples) {
  return triples.map(([u, v, weight], id) => ({
    id,
    u,
    v,
    weight
  }));
}

// Independent oracle: enumerate simple paths and simple cycles, then use reachability
// to classify which source-target walks can repeat a negative cycle.
function pathOracle(n, edges, source) {
  const reach = Array.from({
    length: n
  }, (_, u) => Array.from({
    length: n
  }, (_, v) => u === v));
  edges.forEach(({
    u,
    v
  }) => {
    reach[u][v] = true;
  });
  for (let k = 0; k < n; k++) for (let u = 0; u < n; u++) for (let v = 0; v < n; v++) reach[u][v] ||= reach[u][k] && reach[k][v];
  const distances = Array(n).fill(Infinity);
  distances[source] = 0;
  function paths(u, visited, total) {
    distances[u] = Math.min(distances[u], total);
    for (const edge of edges.filter(edge => edge.u === u)) if (!visited.includes(edge.v)) paths(edge.v, [...visited, edge.v], total + edge.weight);
  }
  paths(source, [source], 0);
  const negative = new Set();
  for (let start = 0; start < n; start++) {
    function cycles(u, visited, total) {
      for (const edge of edges.filter(edge => edge.u === u)) {
        if (edge.v === start && total + edge.weight < 0) visited.forEach(vertex => negative.add(vertex));else if (!visited.includes(edge.v)) cycles(edge.v, [...visited, edge.v], total + edge.weight);
      }
    }
    cycles(start, [start], 0);
  }
  for (let v = 0; v < n; v++) if ([...negative].some(k => reach[source][k] && reach[k][v])) distances[v] = -Infinity;
  return distances;
}
function componentLabels(n, edges) {
  const labels = Array(n).fill(-1);
  for (let start = 0; start < n; start++) {
    if (labels[start] >= 0) continue;
    const todo = [start];
    labels[start] = start;
    while (todo.length) {
      const u = todo.pop();
      for (const edge of edges) {
        const v = edge.u === u ? edge.v : edge.v === u ? edge.u : -1;
        if (v >= 0 && labels[v] < 0) {
          labels[v] = start;
          todo.push(v);
        }
      }
    }
  }
  return labels;
}
function forestOracle(n, edges) {
  const original = componentLabels(n, edges);
  const needed = n - new Set(original).size;
  let best = Infinity;
  for (let mask = 0; mask < 2 ** edges.length; mask++) {
    const chosen = edges.filter((_, id) => mask & 1 << id);
    if (chosen.length !== needed) continue;
    const components = componentLabels(n, chosen);
    if (components.every((value, u) => components.every((other, v) => value === other === (original[u] === original[v])))) best = Math.min(best, chosen.reduce((sum, edge) => sum + edge.weight, 0));
  }
  return best;
}
function permutations(values) {
  if (!values.length) return [[]];
  return values.flatMap((value, index) => permutations(values.filter((_, i) => i !== index)).map(tail => [value, ...tail]));
}
assert.deepEqual(dijkstraTrace(6, parseWeightedEdges(ROUTE_TEXT), 0).at(-1).distances, [0, 2, 1, 4, 7, Infinity]);
assert.deepEqual(bellmanFordTrace(6, parseWeightedEdges(NEGATIVE_TEXT), 0).at(-1).distances, [0, -Infinity, -Infinity, -Infinity, Infinity, Infinity]);
assert.equal(spanningForestTrace(6, parseWeightedEdges(FOREST_TEXT, {
  directed: false
})).at(-1).total, 13);
assert.deepEqual(criticalSchedule(6, parseWeightedEdges(DEPENDENCY_TEXT, {
  weighted: false
}), JOB_DURATIONS).chain, [1, 3, 4, 5]);
let pathCases = 0;
for (let trial = 0; trial < 360; trial++) {
  const n = 1 + random(5);
  const edges = [];
  for (let u = 0; u < n; u++) for (let v = 0; v < n; v++) if (random(4) === 0) edges.push({
    id: edges.length,
    u,
    v,
    weight: random(10) - 4
  });
  for (let source = 0; source < n; source++) {
    assert.deepEqual(bellmanFordTrace(n, edges, source).at(-1).distances, pathOracle(n, edges, source));
    const positive = edges.map(edge => ({
      ...edge,
      weight: Math.abs(edge.weight)
    }));
    const trace = dijkstraTrace(n, positive, source);
    const final = trace.at(-1);
    assert.deepEqual(final.distances, pathOracle(n, positive, source));
    const settled = new Set();
    for (const state of trace) {
      assert.ok(Object.isFrozen(state.distances));
      if (state.kind === 'settle') {
        assert.ok(!settled.has(state.current));
        settled.add(state.current);
      }
      state.settled.forEach((value, vertex) => {
        if (value) assert.equal(state.distances[vertex], final.distances[vertex]);
      });
    }
    final.distances.forEach((distance, target) => {
      const witness = recoverRoute(final.parents, source, target);
      if (distance === Infinity) assert.equal(witness, null);else {
        assert.equal(witness.edgeIds.reduce((sum, id) => sum + positive[id].weight, 0), distance);
        witness.edgeIds.forEach((id, index) => {
          assert.equal(positive[id].u, witness.vertices[index]);
          assert.equal(positive[id].v, witness.vertices[index + 1]);
        });
      }
    });
    pathCases++;
  }
}
for (let trial = 0; trial < 320; trial++) {
  const n = 1 + random(5);
  const triples = Array.from({
    length: random(9)
  }, () => [random(n), random(n), random(13) - 6]);
  const edges = edgesOf(triples);
  const oracle = forestOracle(n, edges);
  for (const method of ['kruskal', 'prim']) {
    const final = spanningForestTrace(n, edges, method).at(-1);
    assert.equal(final.total, oracle);
    assert.equal(final.accepted.length, n - new Set(componentLabels(n, edges)).size);
    assert.equal(new Set(final.accepted).size, final.accepted.length);
  }
}
for (let trial = 0; trial < 200; trial++) {
  const n = 1 + random(5);
  const edges = [];
  for (let u = 0; u < n; u++) for (let v = 0; v < n; v++) if (random(5) === 0) edges.push({
    id: edges.length,
    u,
    v,
    weight: 1
  });
  const valid = permutations(Array.from({
    length: n
  }, (_, vertex) => vertex)).filter(order => edges.every(edge => order.indexOf(edge.u) < order.indexOf(edge.v)));
  const cycle = directedCycle(n, edges);
  assert.equal(cycle === null, valid.length > 0);
  if (cycle) {
    assert.equal(cycle[0], cycle.at(-1));
    cycle.slice(1).forEach((vertex, index) => assert.ok(edges.some(edge => edge.u === cycle[index] && edge.v === vertex)));
  }
  let state = dependencyState(n, edges);
  while (state.ready.length) state = dependencyState(n, edges, [...state.order, state.ready[random(state.ready.length)]]);
  assert.equal(state.complete, valid.length > 0);
  if (state.complete) {
    const durations = Array.from({
      length: n
    }, () => random(6));
    const schedule = criticalSchedule(n, edges, durations);
    // Enumerate all directed chains as an independent critical-path certificate.
    let longest = 0;
    function chains(vertex, total) {
      longest = Math.max(longest, total);
      edges.filter(edge => edge.u === vertex).forEach(edge => chains(edge.v, total + durations[edge.v]));
    }
    durations.forEach((duration, vertex) => chains(vertex, duration));
    assert.equal(schedule.makespan, longest);
    assert.equal(schedule.chain.reduce((sum, vertex) => sum + durations[vertex], 0), longest);
    edges.forEach(edge => assert.ok(schedule.starts[edge.v] >= schedule.finishes[edge.u]));
  }
}
for (const text of ['A A 1', 'A B NaN', 'A B 1.5', 'A B 26', 'A B 2\nA B 3', 'a B 3']) assert.throws(() => parseWeightedEdges(text));
assert.throws(() => parseWeightedEdges('A B -1', {
  nonnegative: true
}));
assert.equal(parseWeightedEdges('A B 1\nB A 2').length, 2);
assert.throws(() => parseWeightedEdges('A B 1\nB A 2', {
  directed: false
}));
assert.deepEqual(parseWeightedEdges(''), []);
console.log(`Weighted graphs: ${pathCases} signed/nonnegative source oracles; 320 exhaustive forests; 200 permutation/cycle/schedule checks; finite immutable states and input contracts passed.`);
