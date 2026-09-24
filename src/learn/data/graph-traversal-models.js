export const GRAPH_VERTICES = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H'];
export const GRAPH_EDGES = [['A', 'B'], ['A', 'C'], ['B', 'D'], ['C', 'D'], ['D', 'E'], ['F', 'G']];
export const GRID_DEFAULT_WALLS = [[0, 3], [1, 1], [1, 3], [2, 1], [3, 3]];
export const GRID_DEFAULT_SOURCES = [[0, 0], [4, 4]];
export const GRID_DEFAULT_TARGET = [4, 4];
const clone = value => structuredClone(value);
const cellKey = ([row, column]) => `${row},${column}`;
const fromCellKey = key => key.split(',').map(Number);
export function parseGraphEdges(text) {
  if (!text.trim()) return {
    valid: true,
    edges: [],
    error: null
  };
  const parts = text.split(',').map(part => part.trim());
  if (parts.length > 24 || parts.some(part => !/^([A-H])\s*(?:-|>|→)\s*([A-H])$/.test(part))) return {
    valid: false,
    edges: [],
    error: 'Use at most 24 comma-separated edges such as A-B, with uppercase endpoints A through H.'
  };
  return {
    valid: true,
    edges: parts.map(part => {
      const match = part.match(/^([A-H])\s*(?:-|>|→)\s*([A-H])$/);
      return [match[1], match[2]];
    }),
    error: null
  };
}
export function buildGraph(edges = GRAPH_EDGES, directed = false, vertices = GRAPH_VERTICES) {
  if (!Array.isArray(vertices) || new Set(vertices).size !== vertices.length || vertices.length > 8 || vertices.some(vertex => typeof vertex !== 'string')) throw new TypeError('Use at most eight distinct vertex labels.');
  if (edges.length > 24) throw new TypeError('Use at most 24 edge entries.');
  const known = new Set(vertices),
    unique = new Map(),
    adjacency = Object.fromEntries(vertices.map(vertex => [vertex, []]));
  for (const edge of edges) {
    if (!Array.isArray(edge) || edge.length !== 2 || edge.some(vertex => !known.has(vertex))) throw new TypeError('Every edge endpoint must name an existing vertex.');
    const [from, to] = !directed && edge[0] > edge[1] ? [edge[1], edge[0]] : edge;
    unique.set(`${from}\u0000${to}`, [from, to]);
  }
  const normalized = [...unique.values()].sort(([a, b], [c, d]) => a < c ? -1 : a > c ? 1 : b < d ? -1 : b > d ? 1 : 0);
  for (const [from, to] of normalized) {
    adjacency[from].push(to);
    if (!directed && from !== to) adjacency[to].push(from);
  }
  for (const vertex of vertices) adjacency[vertex].sort();
  const matrix = vertices.map(from => vertices.map(to => Number(adjacency[from].includes(to))));
  return {
    vertices: [...vertices],
    edges: normalized,
    directed: Boolean(directed),
    adjacency,
    matrix
  };
}
export function graphLayout(graph, { compact = false } = {}) {
  const fixed = {
    A: [65, 180],
    B: [205, 80],
    C: [205, 260],
    D: [350, 180],
    E: [495, 180],
    F: [95, 385],
    G: [245, 385],
    H: [465, 385]
  };
  const compactPositions = [[40, 170], [162, 80], [162, 260], [285, 170]];
  const useCompact = compact && graph.vertices.length <= compactPositions.length;
  const nodes = graph.vertices.map((vertex, index) => ({
    vertex,
    x: useCompact ? compactPositions[index][0] : fixed[vertex]?.[0] ?? 70 + index % 4 * 135,
    y: useCompact ? compactPositions[index][1] : fixed[vertex]?.[1] ?? 90 + Math.floor(index / 4) * 210
  }));
  return {
    width: useCompact ? 330 : Math.max(260, ...nodes.map(node => node.x + 65)),
    height: Math.max(180, ...nodes.map(node => node.y + 65)),
    nodes,
    edges: graph.edges.map(([from, to]) => ({
      from,
      to,
      reciprocal: graph.directed && from !== to && graph.adjacency[to].includes(from)
    }))
  };
}
export function graphTraversalTrace(graph = buildGraph(), source = 'A', method = 'bfs') {
  if (!graph.vertices.includes(source)) throw new TypeError('The source must be an existing vertex.');
  if (!['bfs', 'dfs'].includes(method)) throw new TypeError('Choose bfs or dfs.');
  const trace = [],
    discovered = [source],
    finished = [],
    queue = [],
    frames = [],
    entryOrder = [source],
    finishOrder = [];
  const parent = Object.fromEntries(graph.vertices.map(vertex => [vertex, null])),
    distance = {
      ...parent
    },
    depth = {
      ...parent
    },
    entryTime = {
      ...parent
    },
    finishTime = {
      ...parent
    };
  depth[source] = 0;
  if (method === 'bfs') distance[source] = 0;
  let clock = 0;
  const save = (phase, note, fields = {}) => trace.push(clone({
    graph,
    source,
    method,
    phase,
    note,
    discovered,
    finished,
    queue,
    frames,
    entryOrder,
    finishOrder,
    parent,
    distance,
    depth,
    entryTime,
    finishTime,
    activeVertex: null,
    edge: null,
    result: null,
    ...fields
  }));
  if (method === 'bfs') {
    queue.push(source);
    save('Discover and enqueue the source', `Mark ${source} before enqueueing it. Its distance is zero and it has no parent.`, {
      activeVertex: source
    });
    while (queue.length) {
      const vertex = queue.shift();
      save('Dequeue the oldest pending vertex', `${vertex} leaves the FIFO queue. Inspect its neighbors in ascending label order.`, {
        activeVertex: vertex
      });
      for (const neighbor of graph.adjacency[vertex]) {
        save('Inspect one outgoing adjacency', `Check ${vertex} → ${neighbor}. ${discovered.includes(neighbor) ? 'It is already discovered; do not enqueue it again.' : 'It is new: the next step assigns its first parent and distance.'}`, {
          activeVertex: vertex,
          edge: [vertex, neighbor]
        });
        if (!discovered.includes(neighbor)) {
          discovered.push(neighbor);
          parent[neighbor] = vertex;
          distance[neighbor] = distance[vertex] + 1;
          depth[neighbor] = depth[vertex] + 1;
          entryOrder.push(neighbor);
          queue.push(neighbor);
          save('Mark, assign and enqueue once', `${neighbor} gets parent ${vertex} and distance ${distance[neighbor]}. Marking now prevents another edge from adding a duplicate queue entry.`, {
            activeVertex: neighbor,
            edge: [vertex, neighbor]
          });
        }
      }
      finished.push(vertex);
      finishOrder.push(vertex);
      save('Finish this adjacency list', `Every outgoing adjacency of ${vertex} has been inspected. Pending vertices stay in FIFO order.`, {
        activeVertex: vertex
      });
    }
  } else {
    frames.push({
      vertex: source,
      nextNeighborIndex: 0
    });
    entryTime[source] = ++clock;
    save('Enter the source frame', `Begin ${source}'s call. A frame remembers the next neighbor to inspect after a child returns.`, {
      activeVertex: source
    });
    while (frames.length) {
      const frame = frames.at(-1),
        vertex = frame.vertex,
        neighbors = graph.adjacency[vertex];
      if (frame.nextNeighborIndex === neighbors.length) {
        frames.pop();
        finished.push(vertex);
        finishOrder.push(vertex);
        finishTime[vertex] = ++clock;
        save('Finish and return to the caller', `${vertex} has no uninspected neighbors. Record its finish time and remove its frame.`, {
          activeVertex: vertex
        });
        continue;
      }
      const neighbor = neighbors[frame.nextNeighborIndex++];
      save('Inspect the next saved neighbor', `Frame ${vertex} advances past ${neighbor}; it will resume at neighbor index ${frame.nextNeighborIndex} after any child call returns.`, {
        activeVertex: vertex,
        edge: [vertex, neighbor]
      });
      if (discovered.includes(neighbor)) {
        save('Do not enter a discovered vertex again', `${neighbor} is ${frames.some(item => item.vertex === neighbor) ? 'still active on the frame stack' : 'already finished'}. Skip another call; a generic revisit alone does not establish a directed cycle.`, {
          activeVertex: vertex,
          edge: [vertex, neighbor]
        });
        continue;
      }
      discovered.push(neighbor);
      parent[neighbor] = vertex;
      depth[neighbor] = depth[vertex] + 1;
      entryOrder.push(neighbor);
      entryTime[neighbor] = ++clock;
      frames.push({
        vertex: neighbor,
        nextNeighborIndex: 0
      });
      save('Enter one child frame', `Suspend ${vertex}; enter ${neighbor}. The new frame explores its entire reachable unfinished branch before ${vertex} resumes.`, {
        activeVertex: neighbor,
        edge: [vertex, neighbor]
      });
    }
  }
  save('Reachable search complete', `${discovered.length} of ${graph.vertices.length} vertices were reached from ${source}. Unreached vertices need another starting point; one run does not automatically cover every component.`, {
    result: 'complete'
  });
  return trace;
}
export function graphPath(state, target) {
  if (!state.graph.vertices.includes(target)) throw new TypeError('Target must be an existing vertex.');
  if (!state.discovered.includes(target)) return null;
  const path = [],
    seen = new Set();
  let current = target;
  while (current !== null) {
    if (seen.has(current)) throw new Error('Parent cycle.');
    seen.add(current);
    path.push(current);
    current = state.parent[current];
  }
  return path.reverse();
}
export function directedCycleExamples() {
  const vertices = ['A', 'B', 'C', 'D'];
  return {
    dag: buildGraph([['A', 'B'], ['A', 'C'], ['B', 'D'], ['C', 'D']], true, vertices),
    cycle: buildGraph([['A', 'B'], ['B', 'D'], ['D', 'A']], true, vertices),
    dagRevisit: ['C', 'D'],
    backEdge: ['D', 'A']
  };
}
export function gridWavefrontTrace({
  rows = 5,
  columns = 5,
  walls = GRID_DEFAULT_WALLS,
  sources = [GRID_DEFAULT_SOURCES[0]],
  target = GRID_DEFAULT_TARGET
} = {}) {
  if (!Number.isInteger(rows) || !Number.isInteger(columns) || rows < 1 || columns < 1 || rows > 6 || columns > 6) throw new TypeError('Use a grid of at most 6 by 6 cells.');
  const validCell = cell => Array.isArray(cell) && cell.length === 2 && cell.every(Number.isInteger) && cell[0] >= 0 && cell[0] < rows && cell[1] >= 0 && cell[1] < columns;
  if (!sources.length || sources.some(cell => !validCell(cell)) || walls.some(cell => !validCell(cell)) || !validCell(target)) throw new TypeError('Supply in-bounds sources, walls and target.');
  const blocked = new Set(walls.map(cellKey)),
    roots = [...new Set(sources.map(cellKey))].sort((a, b) => {
      const [ar, ac] = fromCellKey(a),
        [br, bc] = fromCellKey(b);
      return ar - br || ac - bc;
    });
  if (roots.some(root => blocked.has(root)) || blocked.has(cellKey(target))) throw new TypeError('Sources and target must be open cells.');
  const distance = Array.from({
      length: rows
    }, () => Array(columns).fill(null)),
    parent = {},
    owner = {},
    expanded = [],
    trace = [];
  let frontier = roots.map(fromCellKey),
    layer = 0;
  for (const root of roots) {
    const [row, column] = fromCellKey(root);
    distance[row][column] = 0;
    parent[root] = null;
    owner[root] = root;
  }
  const pathToTarget = () => {
    if (distance[target[0]][target[1]] === null) return null;
    const path = [];
    let current = cellKey(target);
    while (current !== null) {
      path.push(fromCellKey(current));
      current = parent[current];
    }
    return path.reverse();
  };
  const save = (phase, note, fields = {}) => trace.push(clone({
    rows,
    columns,
    walls: [...blocked].map(fromCellKey),
    sources: roots.map(fromCellKey),
    target,
    distance,
    parent,
    owner,
    expanded,
    frontier,
    layer,
    phase,
    note,
    path: pathToTarget(),
    result: null,
    ...fields
  }));
  save('Seed every source at distance zero', `${roots.length} source${roots.length === 1 ? ' starts' : 's start'} together. A cell is discovered when its distance is first assigned.`);
  while (frontier.length) {
    const currentLayer = frontier.map(cell => [...cell]),
      next = [];
    save('Expand the current distance layer', `Inspect all open four-neighbors of distance-${layer} cells. Neighbor order is up, left, right, down (row/column order); previously discovered cells keep their first parent.`);
    for (const [row, column] of currentLayer) {
      const key = cellKey([row, column]);
      expanded.push(key);
      for (const [dr, dc] of [[-1, 0], [0, -1], [0, 1], [1, 0]]) {
        const nr = row + dr,
          nc = column + dc,
          nextKey = cellKey([nr, nc]);
        if (nr < 0 || nr >= rows || nc < 0 || nc >= columns || blocked.has(nextKey) || distance[nr][nc] !== null) continue;
        distance[nr][nc] = layer + 1;
        parent[nextKey] = key;
        owner[nextKey] = owner[key];
        next.push([nr, nc]);
      }
    }
    frontier = next;
    layer++;
    save(next.length ? 'Discover the next layer' : 'No new cells beyond this layer', next.length ? `${next.length} new cell${next.length === 1 ? ' has' : 's have'} distance ${layer}. Their first parents lead back toward a nearest source.` : 'Every reachable open cell has been discovered. Walls and disconnected open cells are different states.');
  }
  save('Wavefront complete', distance[target[0]][target[1]] === null ? 'The target is open but unreachable from these sources.' : `The target needs ${distance[target[0]][target[1]]} moves from its nearest source. This path is one shortest route under the declared tie order.`, {
    result: 'complete'
  });
  return trace;
}
