import assert from 'node:assert/strict';
import { GRAPH_VERTICES, GRAPH_EDGES, GRID_DEFAULT_WALLS, GRID_DEFAULT_SOURCES, GRID_DEFAULT_TARGET, parseGraphEdges, buildGraph, graphLayout, graphTraversalTrace, graphPath, directedCycleExamples, gridWavefrontTrace } from '../src/learn/data/graph-traversal-models.js';
const last = trace => trace.at(-1),
  key = cell => cell.join(',');
let graphCases = 0,
  gridCases = 0;
// Independent all-pairs dynamic programming: no queue or production adjacency lists.
function closure(vertices, edges, directed) {
  const distance = vertices.map((_, i) => vertices.map((_, j) => i === j ? 0 : Infinity));
  for (const [from, to] of edges) {
    const i = vertices.indexOf(from),
      j = vertices.indexOf(to);
    distance[i][j] = Math.min(distance[i][j], 1);
    if (!directed) distance[j][i] = Math.min(distance[j][i], 1);
  }
  for (let via = 0; via < vertices.length; via++) for (let from = 0; from < vertices.length; from++) for (let to = 0; to < vertices.length; to++) distance[from][to] = Math.min(distance[from][to], distance[from][via] + distance[via][to]);
  return distance;
}
function recursiveReference(vertices, edges, directed, source) {
  const seen = new Set(),
    entry = [],
    finish = [],
    parent = {},
    depth = {},
    entered = {},
    finished = {};
  let clock = 0;
  function visit(vertex, from = null, level = 0) {
    seen.add(vertex);
    parent[vertex] = from;
    depth[vertex] = level;
    entry.push(vertex);
    entered[vertex] = ++clock;
    const neighbors = vertices.filter(candidate => edges.some(([a, b]) => a === vertex && b === candidate || !directed && b === vertex && a === candidate)).sort();
    for (const neighbor of neighbors) if (!seen.has(neighbor)) visit(neighbor, vertex, level + 1);
    finish.push(vertex);
    finished[vertex] = ++clock;
  }
  visit(source);
  return {
    entry,
    finish,
    parent,
    depth,
    entered,
    finished
  };
}
function verifyGraph(edges, directed, vertices) {
  const graph = buildGraph(edges, directed, vertices),
    distances = closure(vertices, edges, directed),
    baseline = structuredClone(graph);
  for (let i = 0; i < vertices.length; i++) for (let j = 0; j < vertices.length; j++) {
    const expected = edges.some(([from, to]) => from === vertices[i] && to === vertices[j] || !directed && from === vertices[j] && to === vertices[i]);
    assert.equal(graph.matrix[i][j], Number(expected));
    assert.equal(graph.adjacency[vertices[i]].includes(vertices[j]), expected);
  }
  const layout = graphLayout(graph);
  assert.deepEqual(layout.nodes.map(node => node.vertex), vertices);
  assert.equal(layout.edges.length, graph.edges.length);
  for (const node of layout.nodes) {
    assert(node.x >= 31 && node.x + 31 <= layout.width);
    assert(node.y >= 72 && node.y + 44 <= layout.height);
  }
  for (const edge of layout.edges) assert(graph.edges.some(([from, to]) => from === edge.from && to === edge.to));
  for (const source of vertices) for (const method of ['bfs', 'dfs']) {
    const trace = graphTraversalTrace(graph, source, method),
      end = last(trace),
      sourceIndex = vertices.indexOf(source),
      reachable = vertices.filter((_, index) => Number.isFinite(distances[sourceIndex][index]));
    assert.deepEqual([...end.discovered].sort(), [...reachable].sort());
    assert.deepEqual([...end.finished].sort(), [...reachable].sort());
    assert.equal(end.result, 'complete');
    assert.deepEqual(end.queue, []);
    assert.deepEqual(end.frames, []);
    if (method === 'dfs') {
      const reference = recursiveReference(vertices, edges, directed, source);
      assert.deepEqual(end.entryOrder, reference.entry);
      assert.deepEqual(end.finishOrder, reference.finish);
      for (const vertex of reachable) {
        assert.equal(end.parent[vertex], reference.parent[vertex]);
        assert.equal(end.depth[vertex], reference.depth[vertex]);
        assert.equal(end.entryTime[vertex], reference.entered[vertex]);
        assert.equal(end.finishTime[vertex], reference.finished[vertex]);
      }
    }
    for (const vertex of vertices) {
      const expected = distances[sourceIndex][vertices.indexOf(vertex)],
        route = graphPath(end, vertex);
      if (!Number.isFinite(expected)) {
        assert.equal(route, null);
        assert.equal(end.distance[vertex], null);
        assert.equal(end.parent[vertex], null);
        continue;
      }
      assert.equal(route[0], source);
      assert.equal(route.at(-1), vertex);
      assert.equal(new Set(route).size, route.length);
      for (let index = 1; index < route.length; index++) assert(graph.adjacency[route[index - 1]].includes(route[index]));
      if (method === 'bfs') {
        assert.equal(end.distance[vertex], expected);
        assert.equal(route.length - 1, expected);
      } else assert.equal(route.length - 1, end.depth[vertex]);
    }
    let oldDiscovered = [];
    for (const frame of trace) {
      assert.deepEqual(frame.graph, graph);
      assert.deepEqual(frame.discovered.slice(0, oldDiscovered.length), oldDiscovered);
      assert.equal(new Set(frame.discovered).size, frame.discovered.length);
      assert.equal(new Set(frame.queue).size, frame.queue.length);
      for (const vertex of frame.queue) {
        assert(frame.discovered.includes(vertex));
        assert(!frame.finished.includes(vertex));
      }
      for (let index = 1; index < frame.queue.length; index++) assert(frame.distance[frame.queue[index - 1]] <= frame.distance[frame.queue[index]]);
      for (let index = 0; index < frame.frames.length; index++) {
        const item = frame.frames[index];
        assert(!frame.finished.includes(item.vertex));
        assert(item.nextNeighborIndex >= 0 && item.nextNeighborIndex <= graph.adjacency[item.vertex].length);
        if (index) assert.equal(frame.parent[item.vertex], frame.frames[index - 1].vertex);
      }
      oldDiscovered = frame.discovered;
    }
    assert.deepEqual(graph, baseline);
    graphCases++;
  }
}
const four = ['A', 'B', 'C', 'D'],
  undirectedPairs = four.flatMap((from, index) => four.slice(index + 1).map(to => [from, to]));
for (let mask = 0; mask < 2 ** undirectedPairs.length; mask++) verifyGraph(undirectedPairs.filter((_, index) => mask & 1 << index), false, four);
const three = ['A', 'B', 'C'],
  directedPairs = three.flatMap(from => three.map(to => [from, to]));
for (let mask = 0; mask < 2 ** directedPairs.length; mask++) verifyGraph(directedPairs.filter((_, index) => mask & 1 << index), true, three);
verifyGraph(GRAPH_EDGES, false, GRAPH_VERTICES);
verifyGraph(GRAPH_EDGES, true, GRAPH_VERTICES);
verifyGraph([['A', 'A'], ['A', 'B'], ['B', 'A'], ['A', 'B']], false, GRAPH_VERTICES);
verifyGraph([['A', 'A'], ['A', 'B'], ['B', 'A'], ['A', 'B']], true, GRAPH_VERTICES);
const defaultBfs = last(graphTraversalTrace()),
  defaultDfs = last(graphTraversalTrace(buildGraph(), 'A', 'dfs'));
assert.deepEqual(defaultBfs.entryOrder, ['A', 'B', 'C', 'D', 'E']);
assert.deepEqual(defaultDfs.entryOrder, ['A', 'B', 'D', 'C', 'E']);
assert.deepEqual(graphPath(defaultBfs, 'C'), ['A', 'C']);
assert.deepEqual(graphPath(defaultDfs, 'C'), ['A', 'B', 'D', 'C']);
assert.equal(graphPath(defaultBfs, 'H'), null);
assert.deepEqual(last(graphTraversalTrace(buildGraph(), 'F')).entryOrder, ['F', 'G']);
assert.deepEqual(last(graphTraversalTrace(buildGraph(), 'H')).entryOrder, ['H']);
assert.deepEqual(last(graphTraversalTrace(buildGraph(GRAPH_EDGES, true), 'E')).entryOrder, ['E']);
const duplicates = buildGraph([['A', 'B'], ['B', 'A'], ['A', 'A'], ['A', 'A']], false);
assert.equal(duplicates.edges.length, 2);
assert.deepEqual(duplicates.adjacency.A, ['A', 'B']);
assert.equal(duplicates.matrix[0][0], 1);
assert.equal(buildGraph([['A', 'B'], ['B', 'A']], true).edges.length, 2);
assert.equal(buildGraph([], false, []).vertices.length, 0);
assert.throws(() => buildGraph([['A', 'Z']]), TypeError);
assert.throws(() => buildGraph(Array(25).fill(['A', 'B'])), TypeError);
assert.throws(() => buildGraph([], false, ['A', 'A']), TypeError);
assert.throws(() => graphTraversalTrace(buildGraph(), 'Z'), TypeError);
assert.throws(() => graphTraversalTrace(buildGraph(), 'A', 'all'), TypeError);
for (const text of ['', 'A-B, B-C', 'A-A', 'A>B, B→A']) assert.equal(parseGraphEdges(text).valid, true, text);
for (const text of ['A-B,', 'a-b', 'A-Z', 'A', 'A-B-C', 'A-B,,B-C', Array(25).fill('A-B').join(',')]) assert.equal(parseGraphEdges(text).valid, false, text);
const cycleExamples = directedCycleExamples();
for (const graph of [cycleExamples.dag, cycleExamples.cycle]) {
  const compact = graphLayout(graph, { compact: true });
  assert.equal(compact.width, 330);
  assert.deepEqual(compact.edges, graphLayout(graph).edges);
  for (const node of compact.nodes) {
    assert(node.x >= 35 && node.x + 35 <= compact.width);
    assert(node.y >= 72 && node.y + 44 <= compact.height);
  }
}
const dagFrame = graphTraversalTrace(cycleExamples.dag, 'A', 'dfs').find(frame => frame.phase === 'Do not enter a discovered vertex again' && frame.edge?.join('') === 'CD');
assert(dagFrame.finished.includes('D'));
assert(!dagFrame.frames.some(frame => frame.vertex === 'D'));
const backFrame = graphTraversalTrace(cycleExamples.cycle, 'A', 'dfs').find(frame => frame.phase === 'Do not enter a discovered vertex again' && frame.edge?.join('') === 'DA');
assert(backFrame.frames.some(frame => frame.vertex === 'A'));
assert(!backFrame.finished.includes('A'));
const acyclicDistances = closure(cycleExamples.dag.vertices, cycleExamples.dag.edges, true);
assert(!Number.isFinite(acyclicDistances[3][0]));
const cycleDistances = closure(cycleExamples.cycle.vertices, cycleExamples.cycle.edges, true);
assert.equal(cycleDistances[3][0], 1);
function gridOracle(rows, columns, walls, sources) {
  const blocked = new Set(walls.map(key)),
    cells = Array.from({
      length: rows * columns
    }, (_, index) => [Math.floor(index / columns), index % columns]).filter(cell => !blocked.has(key(cell))),
    vertices = cells.map(key),
    edges = [];
  for (let first = 0; first < cells.length; first++) for (let second = first + 1; second < cells.length; second++) if (Math.abs(cells[first][0] - cells[second][0]) + Math.abs(cells[first][1] - cells[second][1]) === 1) edges.push([vertices[first], vertices[second]]);
  const all = closure(vertices, edges, false);
  return Array.from({
    length: rows
  }, (_, row) => Array.from({
    length: columns
  }, (_, column) => {
    const index = vertices.indexOf(key([row, column]));
    if (index < 0) return null;
    const best = Math.min(...sources.map(source => all[vertices.indexOf(key(source))][index]));
    return Number.isFinite(best) ? best : null;
  }));
}
function verifyGrid(options) {
  const baseline = structuredClone(options),
    trace = gridWavefrontTrace(options),
    end = last(trace),
    expected = gridOracle(end.rows, end.columns, end.walls, end.sources),
    blocked = new Set(end.walls.map(key)),
    sources = new Set(end.sources.map(key));
  assert.deepEqual(end.distance, expected);
  assert.equal(end.result, 'complete');
  assert.deepEqual(end.frontier, []);
  assert.deepEqual(options, baseline);
  for (const frame of trace) {
    assert.equal(new Set(frame.expanded).size, frame.expanded.length);
    assert.equal(new Set(frame.frontier.map(key)).size, frame.frontier.length);
    for (let row = 0; row < frame.rows; row++) for (let column = 0; column < frame.columns; column++) {
      const value = frame.distance[row][column],
        cell = key([row, column]);
      if (blocked.has(cell)) assert.equal(value, null);
      if (value === null) continue;
      assert.equal(value, expected[row][column]);
      if (value === 0) {
        assert(sources.has(cell));
        assert.equal(frame.parent[cell], null);
      } else {
        const previous = frame.parent[cell].split(',').map(Number);
        assert.equal(Math.abs(previous[0] - row) + Math.abs(previous[1] - column), 1);
        assert.equal(frame.distance[previous[0]][previous[1]], value - 1);
        assert.equal(frame.owner[cell], frame.owner[frame.parent[cell]]);
      }
    }
    if (frame.path) {
      assert.equal(key(frame.path.at(-1)), key(frame.target));
      assert(sources.has(key(frame.path[0])));
      assert.equal(frame.path.length - 1, frame.distance[frame.target[0]][frame.target[1]]);
      for (let i = 1; i < frame.path.length; i++) assert.equal(Math.abs(frame.path[i][0] - frame.path[i - 1][0]) + Math.abs(frame.path[i][1] - frame.path[i - 1][1]), 1);
    }
  }
  gridCases++;
  return end;
}
const candidates = Array.from({
  length: 9
}, (_, index) => [Math.floor(index / 3), index % 3]).filter(cell => !['0,0', '2,2'].includes(key(cell)));
for (let mask = 0; mask < 2 ** candidates.length; mask++) {
  const walls = candidates.filter((_, index) => mask & 1 << index);
  for (const sources of [[[0, 0]], [[0, 0], [2, 2]]]) verifyGrid({
    rows: 3,
    columns: 3,
    walls,
    sources,
    target: [2, 2]
  });
}
const single = verifyGrid({
  rows: 5,
  columns: 5,
  walls: GRID_DEFAULT_WALLS,
  sources: [GRID_DEFAULT_SOURCES[0]],
  target: GRID_DEFAULT_TARGET
});
assert.equal(single.distance[4][4], 8);
const multiple = verifyGrid({
  rows: 5,
  columns: 5,
  walls: GRID_DEFAULT_WALLS,
  sources: GRID_DEFAULT_SOURCES,
  target: [0, 4]
});
assert.equal(multiple.distance[4][4], 0);
assert.equal(multiple.distance[0][4], 4);
const selfTarget = verifyGrid({
  rows: 5,
  columns: 5,
  walls: [],
  sources: [[0, 0]],
  target: [0, 0]
});
assert.deepEqual(selfTarget.path, [[0, 0]]);
const isolated = verifyGrid({
  rows: 5,
  columns: 5,
  walls: [[0, 1], [1, 0]],
  sources: [[0, 0]],
  target: [4, 4]
});
assert.equal(isolated.path, null);
assert.equal(isolated.distance[4][4], null);
const noWalls = verifyGrid({
  rows: 5,
  columns: 5,
  walls: [],
  sources: [[0, 0]],
  target: [4, 4]
});
assert.equal(noWalls.distance[4][4], 8);
const duplicateRoots = verifyGrid({
  rows: 2,
  columns: 2,
  walls: [],
  sources: [[1, 1], [0, 0], [1, 1]],
  target: [0, 1]
});
assert.deepEqual(duplicateRoots.sources, [[0, 0], [1, 1]]);
for (const options of [{
  rows: 0
}, {
  rows: 7
}, {
  sources: []
}, {
  sources: [[9, 0]]
}, {
  walls: [[0, 0]]
}, {
  target: [9, 9]
}, {
  walls: [[4, 4]]
}]) assert.throws(() => gridWavefrontTrace(options), TypeError);
const graphIsolation = graphTraversalTrace(),
  firstGraph = structuredClone(graphIsolation[0]);
last(graphIsolation).parent.B = 'H';
assert.deepEqual(graphIsolation[0], firstGraph);
const gridIsolation = gridWavefrontTrace(),
  firstGrid = structuredClone(gridIsolation[0]);
last(gridIsolation).distance[0][0] = 99;
assert.deepEqual(gridIsolation[0], firstGrid);
console.log(`Graph models verified: ${graphCases} BFS/DFS runs against all-pairs shortest paths and independent recursive DFS; ${gridCases} wall/source configurations checked against all-pairs grid distances. Includes representation conservation, direction, self-loops, duplicate collapse, real DFS frames/times, parent routes, cycle-state figures, parsing, limits and snapshot isolation.`);
