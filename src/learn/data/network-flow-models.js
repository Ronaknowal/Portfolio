/** Exact, bounded integer models for the network-flow lesson. Geometry is separate. */
export const FLOW_LABELS = ['S', 'A', 'B', 'C', 'D', 'T'];
export const FLOW_EDGES = [[0, 1, 1], [0, 2, 1], [1, 3, 1], [1, 4, 1], [2, 3, 1], [3, 5, 1], [4, 5, 1]];
export const MATCHING_DEFAULT = [[true, true, false], [true, false, false], [false, false, true]];
export const MATCHING_DEFICIENT = [[true, true, false], [true, false, false], [false, true, false]];
export const PIXEL_BACKGROUND_COST = [0, 1, 6, 0, 5, 6];
export const PIXEL_FOREGROUND_COST = [6, 4, 0, 5, 1, 0];
export const PIXEL_NEIGHBORS = [[0, 1], [1, 2], [3, 4], [4, 5], [0, 3], [1, 4], [2, 5]];
function integer(value, lower, upper, label) {
  if (!Number.isSafeInteger(value) || value < lower || value > upper) {
    throw new Error(`${label} must be an integer from ${lower} to ${upper}.`);
  }
}
export function validateNetwork(n, edges, source, sink) {
  integer(n, 2, 16, 'Vertex count');
  integer(source, 0, n - 1, 'Source');
  integer(sink, 0, n - 1, 'Sink');
  if (source === sink) throw new Error('Source and sink must differ.');
  if (!Array.isArray(edges) || edges.length > 64) throw new Error('Use at most 64 original edges.');
  for (const edge of edges) {
    if (!Array.isArray(edge) || edge.length !== 3) throw new Error('An edge is [from, to, capacity].');
    integer(edge[0], 0, n - 1, 'Edge source');
    integer(edge[1], 0, n - 1, 'Edge destination');
    integer(edge[2], 0, 100, 'Capacity');
  }
}

/** Forward and reverse arcs stay distinct even for antiparallel original edges. */
export function residualArcs(edges, flows) {
  if (!Array.isArray(flows) || flows.length !== edges.length) throw new Error('One flow per original edge.');
  return edges.flatMap(([from, to, capacity], edgeId) => {
    integer(flows[edgeId], 0, capacity, 'Flow');
    return [{
      edgeId,
      direction: 1,
      from,
      to,
      residual: capacity - flows[edgeId]
    }, {
      edgeId,
      direction: -1,
      from: to,
      to: from,
      residual: flows[edgeId]
    }];
  });
}
export function inspectFlow(n, edges, flows, source = 0, sink = n - 1) {
  validateNetwork(n, edges, source, sink);
  if (!Array.isArray(flows) || flows.length !== edges.length) throw new Error('One proposed flow per edge.');
  const incoming = Array(n).fill(0);
  const outgoing = Array(n).fill(0);
  const badEdges = [];
  edges.forEach(([from, to, capacity], edgeId) => {
    integer(flows[edgeId], -100, 100, 'Proposed flow');
    if (flows[edgeId] < 0 || flows[edgeId] > capacity) badEdges.push(edgeId);
    outgoing[from] += flows[edgeId];
    incoming[to] += flows[edgeId];
  });
  const balance = incoming.map((amount, vertex) => amount - outgoing[vertex]);
  const badVertices = balance.flatMap((net, vertex) => vertex !== source && vertex !== sink && net !== 0 ? [vertex] : []);
  const sourceValue = outgoing[source] - incoming[source];
  const sinkValue = incoming[sink] - outgoing[sink];
  return {
    incoming,
    outgoing,
    balance,
    badEdges,
    badVertices,
    sourceValue,
    sinkValue,
    feasible: badEdges.length === 0 && badVertices.length === 0 && sourceValue === sinkValue
  };
}
export function nextAugmentation(n, edges, flows, source = 0, sink = n - 1) {
  const audit = inspectFlow(n, edges, flows, source, sink);
  if (!audit.feasible) throw new Error('Augmentation requires a feasible starting flow.');
  const arcs = residualArcs(edges, flows);
  const adjacency = Array.from({
    length: n
  }, () => []);
  arcs.forEach(arc => adjacency[arc.from].push(arc));
  const parents = Array(n).fill(null);
  const distance = Array(n).fill(null);
  const queue = [source];
  distance[source] = 0;
  for (let head = 0; head < queue.length; head++) {
    const vertex = queue[head];
    for (const arc of adjacency[vertex]) {
      if (arc.residual > 0 && distance[arc.to] === null) {
        parents[arc.to] = arc;
        distance[arc.to] = distance[vertex] + 1;
        queue.push(arc.to);
      }
    }
  }
  if (distance[sink] === null) return {
    path: [],
    delta: 0,
    distance,
    reachable: queue
  };
  const path = [];
  for (let vertex = sink; vertex !== source; vertex = parents[vertex].from) path.push({
    ...parents[vertex]
  });
  path.reverse();
  return {
    path,
    delta: Math.min(...path.map(arc => arc.residual)),
    distance,
    reachable: queue
  };
}
export function cutCapacity(n, edges, sourceSide, source = 0, sink = n - 1) {
  validateNetwork(n, edges, source, sink);
  const side = new Set(sourceSide);
  for (const vertex of side) integer(vertex, 0, n - 1, 'Cut vertex');
  if (!side.has(source) || side.has(sink)) throw new Error('A cut keeps the source inside and the sink outside.');
  const outgoingIds = [];
  const incomingIds = [];
  edges.forEach(([from, to], edgeId) => {
    if (side.has(from) && !side.has(to)) outgoingIds.push(edgeId);
    if (!side.has(from) && side.has(to)) incomingIds.push(edgeId);
  });
  return {
    outgoingIds,
    incomingIds,
    capacity: outgoingIds.reduce((total, edgeId) => total + edges[edgeId][2], 0)
  };
}
export function solveFlow(n, edges, source = 0, sink = n - 1) {
  validateNetwork(n, edges, source, sink);
  let flows = edges.map(() => 0);
  const states = [flows.slice()];
  const steps = [];
  while (true) {
    const search = nextAugmentation(n, edges, flows, source, sink);
    if (!search.path.length) {
      const cut = cutCapacity(n, edges, search.reachable, source, sink);
      return {
        flows,
        value: inspectFlow(n, edges, flows, source, sink).sourceValue,
        states,
        steps,
        reachable: search.reachable,
        cut
      };
    }
    const after = flows.slice();
    for (const arc of search.path) after[arc.edgeId] += arc.direction * search.delta;
    steps.push({
      ...search,
      before: flows.slice(),
      after: after.slice()
    });
    states.push(after.slice());
    flows = after;
    // The documented finite model has at most16vertices/64edges. This is a defect guard.
    if (steps.length > 4096) throw new Error('Unexpected augmentation bound exceeded.');
  }
}
export function matchingNetwork(matrix) {
  if (!Array.isArray(matrix) || matrix.length !== 3 || matrix.some(row => !Array.isArray(row) || row.length !== 3 || row.some(value => typeof value !== 'boolean'))) {
    throw new Error('The browser matching matrix must be 3 by 3 Boolean values.');
  }
  const edges = [[0, 1, 1], [0, 2, 1], [0, 3, 1]];
  const pairEdges = [];
  matrix.forEach((row, left) => row.forEach((allowed, right) => {
    if (allowed) {
      pairEdges.push({
        left,
        right,
        edgeId: edges.length
      });
      edges.push([left + 1, right + 4, 1]);
    }
  }));
  edges.push([4, 7, 1], [5, 7, 1], [6, 7, 1]);
  return {
    edges,
    pairEdges
  };
}
export function solveMatching(matrix) {
  const network = matchingNetwork(matrix);
  const flow = solveFlow(8, network.edges);
  const matching = network.pairEdges.filter(pair => flow.flows[pair.edgeId] === 1).map(({
    left,
    right
  }) => [left, right]);
  const leftMate = Array(3).fill(null);
  const rightMate = Array(3).fill(null);
  matching.forEach(([left, right]) => {
    leftMate[left] = right;
    rightMate[right] = left;
  });
  const visitedLeft = new Set(leftMate.flatMap((mate, left) => mate === null ? [left] : []));
  const visitedRight = new Set();
  const queue = [...visitedLeft];
  for (let head = 0; head < queue.length; head++) {
    const left = queue[head];
    matrix[left].forEach((allowed, right) => {
      if (!allowed || leftMate[left] === right || visitedRight.has(right)) return;
      visitedRight.add(right);
      const mate = rightMate[right];
      if (mate !== null && !visitedLeft.has(mate)) {
        visitedLeft.add(mate);
        queue.push(mate);
      }
    });
  }
  const reachableLeft = [...visitedLeft].sort((a, b) => a - b);
  const reachableRight = [...visitedRight].sort((a, b) => a - b);
  return {
    ...network,
    flow,
    matching,
    leftMate,
    rightMate,
    reachableLeft,
    reachableRight,
    coverLeft: [0, 1, 2].filter(left => !visitedLeft.has(left)),
    coverRight: reachableRight.slice(),
    deficiency: reachableLeft.length - reachableRight.length
  };
}
export function pixelEnergy(labels, penalty) {
  integer(penalty, 0, 5, 'Neighbor penalty');
  if (!Array.isArray(labels) || labels.length !== 6 || labels.some(value => typeof value !== 'boolean')) throw new Error('Six Boolean pixel labels are required.');
  const unary = labels.reduce((total, foreground, vertex) => total + (foreground ? PIXEL_FOREGROUND_COST[vertex] : PIXEL_BACKGROUND_COST[vertex]), 0);
  const boundary = PIXEL_NEIGHBORS.filter(([from, to]) => labels[from] !== labels[to]);
  return {
    unary,
    boundary,
    pairwise: boundary.length * penalty,
    total: unary + boundary.length * penalty
  };
}
export function solvePixelCut(penalty) {
  integer(penalty, 0, 5, 'Neighbor penalty');
  const edges = [];
  for (let vertex = 0; vertex < 6; vertex++) {
    edges.push([6, vertex, PIXEL_BACKGROUND_COST[vertex]], [vertex, 7, PIXEL_FOREGROUND_COST[vertex]]);
  }
  for (const [from, to] of PIXEL_NEIGHBORS) edges.push([from, to, penalty], [to, from, penalty]);
  const flow = solveFlow(8, edges, 6, 7);
  const labels = Array.from({
    length: 6
  }, (_, vertex) => flow.reachable.includes(vertex));
  return {
    labels,
    energy: pixelEnergy(labels, penalty),
    flow,
    edges
  };
}
