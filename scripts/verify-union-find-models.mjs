import assert from 'node:assert/strict';
import {
  createUnionFind, traceUnion, traceFind, parentPath, unionFindGroups,
  buildUnionFind, CHAIN_UNIONS, createIslandGrid, activateIslandCell,
  activeIslandGroups, forestLayout,
} from '../src/learn/data/union-find-models.js';

// Independent oracle: close a Boolean adjacency relation, without parent pointers.
function connectivity(n, edges) {
  const reach = Array.from({ length: n }, (_, a) => Array.from({ length: n }, (_, b) => a === b));
  for (const [a, b] of edges) reach[a][b] = reach[b][a] = true;
  for (let k = 0; k < n; k++) {
    for (let a = 0; a < n; a++) {
      for (let b = 0; b < n; b++) reach[a][b] ||= reach[a][k] && reach[k][b];
    }
  }
  return reach;
}

let states = 0;
function checkState(state, edges) {
  const n = state.parent.length;
  const reach = connectivity(n, edges);
  const groups = unionFindGroups(state);
  assert.equal(state.count, groups.length);
  for (const group of groups) assert.equal(state.size[group.root], group.members.length);
  for (let a = 0; a < n; a++) {
    const path = parentPath(state, a);
    assert.ok(path.length <= n);
    const result = traceFind(state, a);
    assert.equal(result.root, path.at(-1));
    for (const frame of result.frames) {
      assert.deepEqual(unionFindGroups(frame.state), groups, 'compression preserves complete partition and roots');
    }
    assert.ok(parentPath(result.state, a).length <= 2);
    for (let b = 0; b < n; b++) assert.equal(path.at(-1) === parentPath(state, b).at(-1), reach[a][b]);
  }
  const layout = forestLayout(state);
  for (const position of Object.values(layout.positions)) {
    assert.ok(position.x >= 25 && position.x <= layout.width - 25);
    assert.ok(position.y >= 25 && position.y <= layout.height - 25);
  }
  states++;
}

const edges = [[0, 0], [0, 1], [0, 2], [1, 2], [2, 3], [1, 3]];
function enumerate(prefix, depth) {
  for (const policy of ['size', 'unweighted']) {
    for (const compress of [true, false]) {
      let state = createUnionFind(4);
      const seen = [];
      for (const edge of prefix) {
        const previous = JSON.stringify(state);
        const already = connectivity(4, seen)[edge[0]][edge[1]];
        const result = traceUnion(state, ...edge, policy, compress);
        assert.equal(result.merged, !already);
        assert.equal(JSON.stringify(state), previous, 'old snapshot is unchanged');
        assert.ok(Object.isFrozen(result.state.parent));
        state = result.state;
        seen.push(edge);
      }
      checkState(state, prefix);
    }
  }
  if (depth > 0) for (const edge of edges) enumerate([...prefix, edge], depth - 1);
}
enumerate([], 3);
checkState(createUnionFind(0), []);
const balanced = buildUnionFind();
assert.deepEqual(parentPath(balanced, 7), [7, 6, 4, 0]);
assert.equal(Math.max(...balanced.parent.map((_, node) => parentPath(balanced, node).length - 1)), 3);
assert.equal(parentPath(buildUnionFind(CHAIN_UNIONS, 'unweighted'), 0).length - 1, 7);
assert.equal(parentPath(buildUnionFind(CHAIN_UNIONS, 'size'), 7).length - 1, 1);
for (const bad of [-1, 1.2, NaN, Infinity, 26]) assert.throws(() => createUnionFind(bad));
for (const bad of [-1, 4, 1.2, NaN]) assert.throws(() => traceUnion(createUnionFind(4), 0, bad));

// Grid oracle: flood fill active cells, with no union/find operations.
function floodGroups(active) {
  const unseen = new Set(active.flatMap((open, index) => open ? [index] : []));
  const groups = [];
  while (unseen.size) {
    const start = unseen.values().next().value;
    unseen.delete(start);
    const queue = [start];
    for (let position = 0; position < queue.length; position++) {
      const node = queue[position];
      for (const candidate of [...unseen]) {
        if (Math.abs(Math.floor(candidate / 5) - Math.floor(node / 5)) + Math.abs(candidate % 5 - node % 5) === 1) {
          queue.push(candidate);
          unseen.delete(candidate);
        }
      }
    }
    groups.push(queue.sort((a, b) => a - b));
  }
  return groups.sort((a, b) => a[0] - b[0]);
}
let grids = 0;
const cells = [0, 1, 2, 5, 6, 7, 10, 11, 12];
for (let mask = 0; mask < 512; mask++) {
  let state = createIslandGrid();
  for (let bit = 0; bit < cells.length; bit++) {
    if (mask & (1 << bit)) state = activateIslandCell(state, cells[bit]).state;
  }
  const expected = floodGroups(state.active);
  const actual = activeIslandGroups(state).map(group => group.members).sort((a, b) => a[0] - b[0]);
  assert.deepEqual(actual, expected);
  assert.equal(state.count, expected.length);
  for (const cell of cells) {
    const before = JSON.stringify(state);
    const result = activateIslandCell(state, cell);
    assert.equal(JSON.stringify(state), before);
    assert.equal(result.state.count, floodGroups(result.state.active).length);
    if (state.active[cell]) assert.equal(result.state, state);
  }
  grids++;
}
console.log(`Union-Find models passed: ${states} partition/trace states, ${grids} grid subsets with every candidate opening, validation and pointer geometry.`);
