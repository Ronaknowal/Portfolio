// Bounded, exact-integer teaching models. Geometry and display sorting are not algorithm costs.
export const GRAPH_LABELS = Object.freeze(['A', 'B', 'C', 'D', 'E', 'F']);
export const ROUTE_TEXT = 'A B 10\nA C 1\nC B 1\nB D 2\nC D 8\nD E 3\nC E 20';
export const NEGATIVE_TEXT = 'A B 2\nB C -4\nC B 1\nC D 2\nE F -3\nF E 1';
export const FOREST_TEXT = 'A B 4\nA C 2\nB C 1\nB D 5\nC D 8\nC E 10\nD E 2\nD F 6\nE F 3';
export const DEPENDENCY_TEXT = 'A C\nB C\nB D\nC E\nD E\nE F';
export const DEPENDENCY_CYCLE_TEXT = `${DEPENDENCY_TEXT}\nE C`;
export const JOB_DURATIONS = Object.freeze([3, 2, 4, 6, 2, 1]);
export function parseWeightedEdges(text, {
  weighted = true,
  directed = true,
  nonnegative = false
} = {}) {
  if (typeof text !== 'string' || text.length > 1000) throw new Error('Use at most 12 edges and 1,000 characters.');
  const lines = text.trim() ? text.trim().split(/\n+/) : [];
  if (lines.length > 12) throw new Error('Use at most 12 edges.');
  const seen = new Set();
  return lines.map((line, id) => {
    const fields = line.trim().split(/\s+/);
    if (fields.length !== (weighted ? 3 : 2)) throw new Error(`Line ${id + 1}: use ${weighted ? 'A B 4' : 'A B'}.`);
    const [from, to, rawWeight] = fields;
    const u = GRAPH_LABELS.indexOf(from);
    const v = GRAPH_LABELS.indexOf(to);
    if (u < 0 || v < 0 || u === v) throw new Error(`Line ${id + 1}: choose two different vertices A–F.`);
    const key = directed ? `${u},${v}` : [u, v].sort().join(',');
    if (seen.has(key)) throw new Error(`Line ${id + 1}: duplicate edge. Keep one weight per pair.`);
    seen.add(key);
    const weight = weighted ? Number(rawWeight) : 1;
    if (weighted && (!/^-?\d+$/.test(rawWeight) || !Number.isInteger(weight) || weight < -25 || weight > 25)) {
      throw new Error(`Line ${id + 1}: weight must be an integer from −25 to 25.`);
    }
    if (nonnegative && weight < 0) throw new Error('Dijkstra requires nonnegative weights. Use the Bellman–Ford investigation for negative edges.');
    return Object.freeze({
      id,
      u,
      v,
      weight
    });
  });
}
function adjacency(n, edges, directed = true) {
  const lists = Array.from({
    length: n
  }, () => []);
  for (const edge of edges) {
    lists[edge.u].push({
      ...edge,
      to: edge.v
    });
    if (!directed) lists[edge.v].push({
      ...edge,
      to: edge.u
    });
  }
  return lists;
}
function freezeSnapshot(value) {
  const copy = structuredClone(value);
  function freeze(item) {
    if (item && typeof item === 'object') {
      Object.values(item).forEach(freeze);
      Object.freeze(item);
    }
  }
  freeze(copy);
  return copy;
}
export function dijkstraTrace(n, edges, source) {
  if (edges.some(edge => edge.weight < 0)) throw new Error('Nonnegative weights required.');
  const lists = adjacency(n, edges);
  const distances = Array(n).fill(Infinity);
  const parents = Array(n).fill(null);
  const settled = Array(n).fill(false);
  let ticket = 0;
  const queue = [{
    distance: 0,
    vertex: source,
    ticket: ticket++
  }];
  const states = [];
  distances[source] = 0;
  function record(kind, message, current = null, edgeId = null) {
    states.push(freezeSnapshot({
      kind,
      message,
      current,
      edgeId,
      distances,
      parents,
      settled,
      queue: [...queue].sort((a, b) => a.distance - b.distance || a.ticket - b.ticket)
    }));
  }
  record('initial', `Only ${GRAPH_LABELS[source] ?? source} has a known route: the empty route costs 0.`);
  while (queue.length) {
    // Small-model priority selection; the standalone Python implementation uses heapq.
    queue.sort((a, b) => a.distance - b.distance || a.ticket - b.ticket);
    const entry = queue.shift();
    const u = entry.vertex;
    if (entry.distance !== distances[u]) {
      record('stale', `Discard ${GRAPH_LABELS[u] ?? u} at ${entry.distance}: its current best is ${distances[u]}.`, u);
      continue;
    }
    settled[u] = true;
    record('settle', `Finalize ${GRAPH_LABELS[u] ?? u} at ${entry.distance}. No remaining nonnegative route can improve it.`, u);
    for (const edge of lists[u]) {
      const candidate = distances[u] + edge.weight;
      const improved = candidate < distances[edge.v];
      const old = distances[edge.v];
      if (improved) {
        distances[edge.v] = candidate;
        parents[edge.v] = {
          vertex: u,
          edgeId: edge.id
        };
        queue.push({
          distance: candidate,
          vertex: edge.v,
          ticket: ticket++
        });
      }
      record(improved ? 'improve' : 'keep', `${GRAPH_LABELS[u] ?? u} → ${GRAPH_LABELS[edge.v] ?? edge.v}: ${distances[u]} + ${edge.weight} = ${candidate}; ${improved ? `improve ${old === Infinity ? '∞' : old} to ${candidate} and queue a new entry.` : 'keep the current best.'}`, u, edge.id);
    }
  }
  record('done', 'The queue is empty. Finite distances are final; ∞ means unreachable.');
  return Object.freeze(states);
}
export function recoverRoute(parents, source, target) {
  const vertices = [target];
  const edgeIds = [];
  let current = target;
  while (current !== source && parents[current]) {
    edgeIds.push(parents[current].edgeId);
    current = parents[current].vertex;
    if (vertices.includes(current)) return null;
    vertices.push(current);
  }
  return current === source ? {
    vertices: vertices.reverse(),
    edgeIds: edgeIds.reverse()
  } : null;
}
export function bellmanFordTrace(n, edges, source) {
  let distances = Array(n).fill(Infinity);
  distances[source] = 0;
  const states = [freezeSnapshot({
    pass: 0,
    distances,
    previous: null,
    candidates: [],
    changed: [],
    affected: [],
    seeds: [],
    message: 'At most 0 edges: only the source has cost 0.'
  })];
  for (let pass = 1; pass <= n; pass += 1) {
    const previous = distances;
    const next = [...previous];
    const candidates = edges.map(edge => {
      const cost = previous[edge.u] + edge.weight;
      if (cost < next[edge.v]) next[edge.v] = cost;
      return {
        edgeId: edge.id,
        fromCost: previous[edge.u],
        cost
      };
    });
    const changed = next.flatMap((value, vertex) => value < previous[vertex] ? [vertex] : []);
    if (pass < n) {
      distances = next;
      states.push(freezeSnapshot({
        pass,
        distances,
        previous,
        candidates,
        changed,
        affected: [],
        seeds: [],
        message: `At most ${pass} edges. Every candidate reads only row ${pass - 1}; carry-forward allows fewer edges.`
      }));
    } else {
      const affected = new Set(changed);
      const queue = [...changed];
      const lists = adjacency(n, edges);
      for (let head = 0; head < queue.length; head += 1) {
        for (const edge of lists[queue[head]]) {
          if (!affected.has(edge.v)) {
            affected.add(edge.v);
            queue.push(edge.v);
          }
        }
      }
      distances = previous.map((value, vertex) => affected.has(vertex) ? -Infinity : value);
      states.push(freezeSnapshot({
        pass,
        distances,
        previous,
        candidates,
        changed,
        affected: [...affected],
        seeds: changed,
        message: `Pass ${n} is a detection pass. Strict improvements seed the downstream region with no finite minimum; unconnected cycles do not affect this source.`
      }));
    }
  }
  return Object.freeze(states);
}
export function spanningForestTrace(n, edges, method = 'kruskal') {
  const parents = Array.from({
    length: n
  }, (_, vertex) => vertex);
  const sizes = Array(n).fill(1);
  const accepted = [];
  const visited = Array(n).fill(false);
  const states = [];
  let total = 0;
  const find = vertex => {
    while (vertex !== parents[vertex]) vertex = parents[vertex];
    return vertex;
  };
  function record(message, edgeId = null, queue = [], kind = 'inspect') {
    states.push(freezeSnapshot({
      message,
      edgeId,
      queue,
      kind,
      accepted,
      total,
      visited,
      components: Array.from({
        length: n
      }, (_, vertex) => find(vertex))
    }));
  }
  function join(edge) {
    let a = find(edge.u);
    let b = find(edge.v);
    if (a === b) return false;
    if (sizes[a] < sizes[b]) [a, b] = [b, a];
    parents[b] = a;
    sizes[a] += sizes[b];
    accepted.push(edge.id);
    total += edge.weight;
    return true;
  }
  const compare = (a, b) => a.weight - b.weight || a.id - b.id;
  if (method === 'kruskal') {
    const remaining = [...edges].sort(compare);
    record('Start with one component per vertex. The next candidate is globally lightest.', null, remaining.map(edge => edge.id), 'initial');
    while (remaining.length) {
      const edge = remaining.shift();
      const added = join(edge);
      record(added ? `Accept ${GRAPH_LABELS[edge.u] ?? edge.u}—${GRAPH_LABELS[edge.v] ?? edge.v}: joins different components; add ${edge.weight}.` : `Reject ${GRAPH_LABELS[edge.u] ?? edge.u}—${GRAPH_LABELS[edge.v] ?? edge.v}: its endpoints already have an accepted route.`, edge.id, remaining.map(item => item.id), added ? 'accept' : 'reject');
    }
  } else if (method === 'prim') {
    const lists = adjacency(n, edges, false);
    const queue = [];
    const enter = vertex => {
      visited[vertex] = true;
      for (const edge of lists[vertex]) if (!visited[edge.to]) queue.push(edge);
      queue.sort(compare);
    };
    record('Start a new tree at the first unvisited vertex. Queue crossing edges, prioritized by their own weights.', null, [], 'initial');
    for (let start = 0; start < n; start += 1) {
      if (visited[start]) continue;
      enter(start);
      record(`Start the component containing ${GRAPH_LABELS[start] ?? start}.`, null, queue.map(edge => edge.id), 'start');
      while (queue.length) {
        const edge = queue.shift();
        if (visited[edge.u] && visited[edge.v]) {
          record('Discard this obsolete entry: both endpoints are already inside the growing tree.', edge.id, queue.map(item => item.id), 'reject');
          continue;
        }
        join(edge);
        enter(visited[edge.u] ? edge.v : edge.u);
        record(`Accept a lightest crossing edge of weight ${edge.weight}; grow this tree by one vertex.`, edge.id, queue.map(item => item.id), 'accept');
      }
    }
  } else throw new Error('Choose Kruskal or Prim.');
  record(`Finished: ${accepted.length} accepted edges, total ${total}. Isolated vertices remain components.`, null, [], 'done');
  return Object.freeze(states);
}
export function directedCycle(n, edges, excluded = []) {
  const lists = adjacency(n, edges);
  const status = Array(n).fill(0);
  excluded.forEach(vertex => {
    status[vertex] = 2;
  });
  const active = [];
  function visit(vertex) {
    status[vertex] = 1;
    active.push(vertex);
    for (const edge of lists[vertex]) {
      if (status[edge.v] === 1) return [...active.slice(active.indexOf(edge.v)), edge.v];
      if (status[edge.v] === 0) {
        const cycle = visit(edge.v);
        if (cycle) return cycle;
      }
    }
    active.pop();
    status[vertex] = 2;
    return null;
  }
  for (let vertex = 0; vertex < n; vertex += 1) {
    if (!status[vertex]) {
      const cycle = visit(vertex);
      if (cycle) return cycle;
    }
  }
  return null;
}
export function dependencyState(n, edges, order = []) {
  const remaining = new Set(Array.from({
    length: n
  }, (_, vertex) => vertex));
  const indegrees = Array(n).fill(0);
  edges.forEach(edge => {
    indegrees[edge.v] += 1;
  });
  for (const vertex of order) {
    if (!remaining.has(vertex) || indegrees[vertex] !== 0) throw new Error('Only a remaining zero-indegree vertex can be emitted.');
    remaining.delete(vertex);
    for (const edge of edges) if (edge.u === vertex) indegrees[edge.v] -= 1;
  }
  const ready = [...remaining].filter(vertex => indegrees[vertex] === 0);
  const cycle = ready.length === 0 && remaining.size ? directedCycle(n, edges, order) : null;
  return freezeSnapshot({
    order,
    indegrees,
    ready,
    remaining: [...remaining],
    cycle,
    complete: !remaining.size
  });
}
export function criticalSchedule(n, edges, durations) {
  if (durations.length !== n || durations.some(value => !Number.isFinite(value) || value < 0)) throw new Error('One finite nonnegative duration per job is required.');
  let state = dependencyState(n, edges);
  const starts = Array(n).fill(0);
  const finishes = Array(n).fill(0);
  const predecessor = Array(n).fill(null);
  while (state.ready.length) {
    const vertex = state.ready[0];
    finishes[vertex] = starts[vertex] + durations[vertex];
    for (const edge of edges) {
      if (edge.u === vertex && finishes[vertex] > starts[edge.v]) {
        starts[edge.v] = finishes[vertex];
        predecessor[edge.v] = vertex;
      }
    }
    state = dependencyState(n, edges, [...state.order, vertex]);
  }
  if (!state.complete) throw new Error('Dependencies must be acyclic.');
  const makespan = Math.max(0, ...finishes);
  const chain = [];
  let vertex = finishes.indexOf(makespan);
  while (vertex >= 0 && vertex !== null) {
    chain.unshift(vertex);
    vertex = predecessor[vertex];
  }
  return freezeSnapshot({
    starts,
    finishes,
    makespan,
    chain,
    order: state.order
  });
}
