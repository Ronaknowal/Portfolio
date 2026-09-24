// Small deterministic teaching models. Exhaustive references are deliberately
// bounded oracles, separate from each method's general complexity guarantee.
const freeze = value => {
  if (value && typeof value === 'object' && !Object.isFrozen(value)) {
    Object.values(value).forEach(freeze);
    Object.freeze(value);
  }
  return value;
};
const integer = (value, minimum, maximum, name) => {
  if (!Number.isSafeInteger(value) || value < minimum || value > maximum) {
    throw new RangeError(name + ' must be an integer from ' + minimum + ' to ' + maximum + '.');
  }
  return value;
};
const sum = values => values.reduce((total, value) => total + value, 0);
const indices = (mask, count) => Array.from({
  length: count
}, (_, index) => index).filter(index => mask & 1 << index);
const population = mask => {
  let count = 0;
  for (let bits = mask; bits; bits &= bits - 1) count += 1;
  return count;
};
export const MATROID_PRESETS = freeze({
  uniform: {
    label: 'Choose at most two items',
    names: ['A', 'B', 'C', 'D'],
    weights: [6, 4, 3, 1],
    rank: 2
  },
  graphic: {
    label: 'Choose a forest on four vertices',
    names: ['AB', 'BC', 'AC', 'CD', 'BD', 'AD'],
    weights: [6, 5, 4, 3, 2, 1],
    edges: [[0, 1], [1, 2], [0, 2], [2, 3], [1, 3], [0, 3]]
  },
  intervals: {
    label: 'Choose nonoverlapping intervals',
    names: ['A:[0,2)', 'B:[0,1)', 'C:[1,2)'],
    weights: [3, 2, 2],
    intervals: [[0, 2], [0, 1], [1, 2]]
  }
});
function independent(kind, mask) {
  const preset = MATROID_PRESETS[kind];
  const selected = indices(mask, preset.names.length);
  if (kind === 'uniform') return selected.length <= preset.rank;
  if (kind === 'intervals') return selected.every((first, index) => selected.slice(index + 1).every(second => {
    const a = preset.intervals[first],
      b = preset.intervals[second];
    return a[1] <= b[0] || b[1] <= a[0];
  }));
  const parent = [0, 1, 2, 3];
  const find = vertex => {
    while (parent[vertex] !== vertex) vertex = parent[vertex];
    return vertex;
  };
  for (const edge of selected) {
    const [a, b] = preset.edges[edge].map(find);
    if (a === b) return false;
    parent[a] = b;
  }
  return true;
}
export function matroidExchangeState({
  kind = 'intervals',
  weights = null,
  smaller = 1,
  larger = 6
} = {}) {
  const preset = MATROID_PRESETS[kind];
  if (!preset) throw new RangeError('Unknown independence system.');
  const size = preset.names.length,
    limit = (1 << size) - 1;
  integer(smaller, 0, limit, 'First selected set');
  integer(larger, 0, limit, 'Second selected set');
  const costs = weights === null ? [...preset.weights] : weights;
  if (!Array.isArray(costs) || costs.length !== size) throw new RangeError('One weight is needed for each element.');
  costs.forEach(weight => integer(weight, -20, 30, 'Element weight'));
  const feasible = Array.from({
    length: limit + 1
  }, (_, mask) => mask).filter(mask => independent(kind, mask));
  const value = mask => sum(indices(mask, size).map(index => costs[index]));
  const witness = (first, second) => indices(second & ~first, size).filter(index => independent(kind, first | 1 << index));
  const violations = [];
  for (const first of feasible) for (const second of feasible) {
    if (population(first) < population(second) && witness(first, second).length === 0) violations.push([first, second]);
  }
  let greedy = 0;
  const trace = [{
    mask: 0,
    value: 0,
    element: null,
    action: 'Start with the empty feasible set.'
  }];
  const order = Array.from({
    length: size
  }, (_, index) => index).sort((a, b) => costs[b] - costs[a] || a - b);
  for (const element of order) {
    const proposed = greedy | 1 << element;
    const accepted = costs[element] > 0 && independent(kind, proposed);
    if (accepted) greedy = proposed;
    trace.push({
      mask: greedy,
      value: value(greedy),
      element,
      accepted,
      action: costs[element] <= 0 ? 'Skip nonpositive weight for this unrestricted objective.' : accepted ? 'Add: feasibility remains valid.' : 'Reject: the enlarged set is infeasible.'
    });
  }
  const optimum = feasible.reduce((best, mask) => value(mask) > value(best) ? mask : best, 0);
  const applicable = independent(kind, smaller) && independent(kind, larger) && population(smaller) < population(larger);
  return freeze({
    kind,
    preset,
    weights: [...costs],
    smaller,
    larger,
    applicable,
    witness: applicable ? witness(smaller, larger) : [],
    feasible,
    augmentationHolds: violations.length === 0,
    violations,
    greedy,
    optimum,
    greedyValue: value(greedy),
    optimumValue: value(optimum),
    trace
  });
}
export const KNAPSACK_ITEMS = freeze([{
  name: 'A',
  weight: 2,
  value: 14
}, {
  name: 'B',
  weight: 3,
  value: 18
}, {
  name: 'C',
  weight: 4,
  value: 22
}, {
  name: 'D',
  weight: 5,
  value: 25
}, {
  name: 'E',
  weight: 7,
  value: 31
}]);
function validateItems(items, capacity) {
  if (!Array.isArray(items) || items.length > 8) throw new RangeError('Use at most eight items in this teaching model.');
  integer(capacity, 0, 1000, 'Capacity');
  items.forEach(item => {
    if (!item || typeof item.name !== 'string') throw new RangeError('Each item needs a name.');
    integer(item.weight, 1, 1000, 'Item weight');
    integer(item.value, 0, 1000, 'Item value');
  });
}
function knapsackOracle(items, capacity) {
  let best = {
    mask: 0,
    value: 0,
    weight: 0
  };
  for (let mask = 0; mask < 1 << items.length; mask += 1) {
    const selected = indices(mask, items.length);
    const weight = sum(selected.map(index => items[index].weight));
    const value = sum(selected.map(index => items[index].value));
    if (weight <= capacity && value > best.value) best = {
      mask,
      value,
      weight
    };
  }
  return best;
}
export function knapsackSearchTrace({
  items = KNAPSACK_ITEMS,
  capacity = 10,
  maxExpanded = 127
} = {}) {
  validateItems(items, capacity);
  integer(maxExpanded, 0, 511, 'Expansion limit');
  const order = items.map((item, index) => ({
    ...item,
    original: index
  })).sort((a, b) => b.value * a.weight - a.value * b.weight || a.original - b.original);
  const bound = node => {
    let available = capacity - node.weight,
      upper = node.value;
    for (let index = node.depth; index < order.length && available > 0; index += 1) {
      const item = order[index],
        used = Math.min(available, item.weight);
      upper += item.value * used / item.weight;
      available -= used;
    }
    return upper;
  };
  let nextId = 0;
  const make = (depth, mask, weight, value, parent, decision) => {
    const node = {
      id: nextId++,
      depth,
      mask,
      weight,
      value,
      parent,
      decision
    };
    return {
      ...node,
      upper: bound(node),
      status: 'open'
    };
  };
  const root = make(0, 0, 0, 0, null, 'Start');
  const nodes = [root],
    frontier = [root],
    trace = [];
  let incumbent = {
    mask: 0,
    weight: 0,
    value: 0
  };
  const snapshot = (expanded, action, current) => {
    const upper = Math.max(incumbent.value, ...frontier.map(node => node.upper));
    trace.push({
      expanded,
      action,
      current,
      incumbent: {
        ...incumbent
      },
      upper,
      gap: upper - incumbent.value,
      optimal: upper === incumbent.value,
      frontier: frontier.map(node => ({
        ...node
      })),
      nodes: nodes.map(node => ({
        ...node
      }))
    });
  };
  snapshot(0, 'The fractional root relaxation bounds every legal subset.', null);
  for (let expanded = 1; expanded <= maxExpanded && frontier.length; expanded += 1) {
    frontier.sort((a, b) => b.upper - a.upper || a.id - b.id);
    const node = frontier.shift();
    let action;
    if (node.upper <= incumbent.value) {
      node.status = 'bound-pruned';
      action = 'Its optimistic bound cannot beat the feasible incumbent.';
    } else if (node.depth === order.length) {
      node.status = 'leaf';
      if (node.value > incumbent.value) incumbent = {
        mask: node.mask,
        weight: node.weight,
        value: node.value
      };
      action = 'A complete feasible decision can update the incumbent.';
    } else {
      node.status = 'expanded';
      const item = order[node.depth];
      if (node.weight + item.weight <= capacity) {
        const child = make(node.depth + 1, node.mask | 1 << item.original, node.weight + item.weight, node.value + item.value, node.id, 'Include ' + item.name);
        nodes.push(child);
        frontier.push(child);
        if (child.value > incumbent.value) incumbent = {
          mask: child.mask,
          weight: child.weight,
          value: child.value
        };
      } else {
        nodes.push({
          id: nextId++,
          parent: node.id,
          depth: node.depth + 1,
          mask: node.mask | 1 << item.original,
          weight: node.weight + item.weight,
          value: node.value + item.value,
          upper: null,
          decision: 'Include ' + item.name,
          status: 'infeasible'
        });
      }
      const excluded = make(node.depth + 1, node.mask, node.weight, node.value, node.id, 'Exclude ' + item.name);
      nodes.push(excluded);
      frontier.push(excluded);
      action = 'Split into include and exclude; keeping no further items already gives a feasible candidate.';
    }
    snapshot(expanded, action, node.id);
  }
  return freeze({
    items: items.map(item => ({
      ...item
    })),
    capacity,
    order,
    trace,
    final: trace.at(-1),
    oracle: knapsackOracle(items, capacity)
  });
}
export function scaledKnapsackState({
  items = KNAPSACK_ITEMS,
  capacity = 10,
  epsilonDenominator = 4
} = {}) {
  validateItems(items, capacity);
  integer(epsilonDenominator, 2, 100, 'Inverse approximation error');
  const active = items.map((item, original) => ({
    ...item,
    original
  })).filter(item => item.weight <= capacity && item.value > 0);
  if (active.length === 0) return freeze({
    items: items.map(item => ({
      ...item
    })),
    capacity,
    epsilon: 1 / epsilonDenominator,
    scale: 0,
    active: [],
    selected: [],
    value: 0,
    weight: 0,
    oracle: knapsackOracle(items, capacity),
    rows: [[0]],
    columns: 1,
    operations: 0,
    guarantee: 0
  });
  const maximum = Math.max(...active.map(item => item.value));
  const divisor = active.length * epsilonDenominator;
  // All integer products fit far below 2^53 in this bounded model.
  const scaled = active.map(item => ({
    ...item,
    rounded: Math.floor(item.value * divisor / maximum)
  }));
  const total = sum(scaled.map(item => item.rounded));
  const rows = [Array(total + 1).fill(Infinity)];
  rows[0][0] = 0;
  let operations = 0;
  for (const item of scaled) {
    const previous = rows.at(-1),
      next = [...previous];
    for (let value = item.rounded; value <= total; value += 1) {
      next[value] = Math.min(next[value], previous[value - item.rounded] + item.weight);
      operations += 1;
    }
    rows.push(next);
  }
  let bestValue = total;
  while (rows.at(-1)[bestValue] > capacity) bestValue -= 1;
  const selected = [];
  for (let count = scaled.length, value = bestValue; count > 0; count -= 1) {
    if (rows[count][value] < rows[count - 1][value]) {
      selected.push(scaled[count - 1].original);
      value -= scaled[count - 1].rounded;
    }
  }
  selected.reverse();
  const oracle = knapsackOracle(items, capacity);
  return freeze({
    items: items.map(item => ({
      ...item
    })),
    capacity,
    epsilon: 1 / epsilonDenominator,
    scale: maximum / divisor,
    active: scaled,
    selected,
    value: sum(selected.map(index => items[index].value)),
    weight: sum(selected.map(index => items[index].weight)),
    oracle,
    rows,
    columns: total + 1,
    operations,
    guarantee: (1 - 1 / epsilonDenominator) * oracle.value,
    bestRoundedValue: bestValue
  });
}
export const COVER_SETS = freeze([{
  name: 'A',
  elements: [0, 1, 2, 3],
  cost: 1
}, {
  name: 'B',
  elements: [0, 1, 4],
  cost: 1
}, {
  name: 'C',
  elements: [2, 3, 5],
  cost: 1
}]);
export function setCoverTrace({
  universeSize = 6,
  sets = COVER_SETS,
  maximumSelections = null
} = {}) {
  integer(universeSize, 0, 12, 'Number of required elements');
  if (!Array.isArray(sets) || sets.length > 8) throw new RangeError('Use at most eight candidate sets.');
  if (maximumSelections !== null) integer(maximumSelections, 0, sets.length, 'Cardinality budget');
  const candidates = sets.map((candidate, index) => {
    if (!candidate || typeof candidate.name !== 'string' || !Array.isArray(candidate.elements)) throw new RangeError('Each candidate needs a name and element list.');
    integer(candidate.cost, 0, 1000, 'Set cost');
    if (new Set(candidate.elements).size !== candidate.elements.length) throw new RangeError('Elements within a set must be distinct.');
    candidate.elements.forEach(element => integer(element, 0, universeSize - 1, 'Element'));
    return {
      ...candidate,
      index,
      elements: [...candidate.elements],
      mask: candidate.elements.reduce((mask, element) => mask | 1 << element, 0)
    };
  });
  const full = (1 << universeSize) - 1;
  let covered = 0,
    cost = 0;
  const selected = [],
    charges = Array(universeSize).fill(0),
    trace = [];
  const snapshot = (chosen, added) => trace.push({
    chosen,
    added,
    covered,
    cost,
    selected: [...selected],
    charges: [...charges]
  });
  snapshot(null, []);
  while (covered !== full && (maximumSelections === null || selected.length < maximumSelections)) {
    const choices = candidates.filter(candidate => !selected.includes(candidate.index)).map(candidate => ({
      ...candidate,
      gain: population(candidate.mask & ~covered)
    })).filter(candidate => candidate.gain > 0);
    choices.sort((a, b) => maximumSelections === null ? a.cost * b.gain - b.cost * a.gain || a.index - b.index : b.gain - a.gain || a.index - b.index);
    if (choices.length === 0) break;
    const choice = choices[0],
      added = indices(choice.mask & ~covered, universeSize);
    added.forEach(element => {
      charges[element] = choice.cost / added.length;
    });
    covered |= choice.mask;
    cost += choice.cost;
    selected.push(choice.index);
    snapshot(choice.index, added);
  }
  let optimumCost = Infinity,
    optimumCoverage = 0,
    optimum = 0;
  for (let mask = 0; mask < 1 << candidates.length; mask += 1) {
    const chosen = indices(mask, candidates.length),
      union = chosen.reduce((bits, index) => bits | candidates[index].mask, 0);
    if (maximumSelections === null) {
      const objective = sum(chosen.map(index => candidates[index].cost));
      if (union === full && objective < optimumCost) {
        optimumCost = objective;
        optimum = mask;
      }
    } else if (chosen.length <= maximumSelections && population(union) > optimumCoverage) {
      optimumCoverage = population(union);
      optimum = mask;
    }
  }
  const maximumSetSize = Math.max(0, ...candidates.map(candidate => candidate.elements.length));
  const harmonic = sum(Array.from({
    length: maximumSetSize
  }, (_, index) => 1 / (index + 1)));
  return freeze({
    universeSize,
    sets: candidates,
    maximumSelections,
    selected,
    covered,
    coveredCount: population(covered),
    cost,
    trace,
    charges,
    feasible: covered === full,
    optimumCost,
    optimumCoverage,
    optimum,
    harmonic,
    maximumSetSize
  });
}
export const ASSIGNMENT_COSTS = freeze([[1, 2, 8], [2, 9, 8], [8, 8, 1]]);
export function assignmentTrace({
  costs = ASSIGNMENT_COSTS,
  required = null
} = {}) {
  if (!Array.isArray(costs) || costs.length < 1 || costs.length > 4 || !Array.isArray(costs[0]) || costs[0].length < 1 || costs[0].length > 4) throw new RangeError('Use one to four workers and jobs.');
  const workers = costs.length,
    jobs = costs[0].length;
  costs.forEach(row => {
    if (!Array.isArray(row) || row.length !== jobs) throw new RangeError('All workers need the same job columns.');
    row.forEach(cost => {
      if (cost !== null) integer(cost, -1000, 1000, 'Assignment cost');
    });
  });
  const target = required === null ? Math.min(workers, jobs) : required;
  integer(target, 0, Math.min(workers, jobs), 'Required assignments');
  const source = workers + jobs,
    sink = source + 1,
    vertices = sink + 1;
  const arcs = [],
    graph = Array.from({
      length: vertices
    }, () => []),
    assignments = [];
  const add = (from, to, cost, type, worker = null, job = null) => {
    const edge = arcs.length;
    arcs.push({
      from,
      to,
      cost,
      residual: 1,
      reverse: edge + 1,
      forward: true,
      type,
      worker,
      job
    });
    arcs.push({
      from: to,
      to: from,
      cost: -cost,
      residual: 0,
      reverse: edge,
      forward: false,
      type,
      worker,
      job
    });
    graph[from].push(edge);
    graph[to].push(edge + 1);
    return edge;
  };
  for (let worker = 0; worker < workers; worker += 1) add(source, worker, 0, 'worker-capacity');
  for (let worker = 0; worker < workers; worker += 1) for (let job = 0; job < jobs; job += 1) if (costs[worker][job] !== null) {
    assignments.push(add(worker, workers + job, costs[worker][job], 'assignment', worker, job));
  }
  for (let job = 0; job < jobs; job += 1) add(workers + job, sink, 0, 'job-capacity');
  const selected = () => assignments.filter(edge => arcs[edge].residual === 0).map(edge => ({
    worker: arcs[edge].worker,
    job: arcs[edge].job,
    cost: arcs[edge].cost
  }));
  const trace = [{
    flow: 0,
    cost: 0,
    pairs: [],
    path: [],
    action: 'No assignments. The forward layered network has no cycle.'
  }];
  let flow = 0;
  while (flow < target) {
    const distance = Array(vertices).fill(Infinity),
      predecessor = Array(vertices).fill(-1);
    distance[source] = 0;
    for (let pass = 0; pass < vertices - 1; pass += 1) {
      let changed = false;
      arcs.forEach((arc, index) => {
        if (arc.residual > 0 && distance[arc.from] + arc.cost < distance[arc.to]) {
          distance[arc.to] = distance[arc.from] + arc.cost;
          predecessor[arc.to] = index;
          changed = true;
        }
      });
      if (!changed) break;
    }
    if (!Number.isFinite(distance[sink])) break;
    const path = [];
    for (let vertex = sink; vertex !== source;) {
      const edge = predecessor[vertex];
      if (edge < 0 || path.length >= vertices) throw new Error('An invalid residual predecessor path was produced.');
      path.push(edge);
      vertex = arcs[edge].from;
    }
    path.reverse();
    const pathDescription = path.map(edge => ({
      ...arcs[edge],
      delta: arcs[edge].forward ? 'add' : 'undo'
    }));
    for (const edge of path) {
      arcs[edge].residual -= 1;
      arcs[arcs[edge].reverse].residual += 1;
    }
    flow += 1;
    const pairs = selected();
    trace.push({
      flow,
      cost: sum(pairs.map(pair => pair.cost)),
      pairs,
      path: pathDescription,
      pathCost: distance[sink],
      action: 'Send one unit along the cheapest residual path, refunding undone assignments.'
    });
  }
  // A zero-edge supersource reaches every residual component. Its Bellman-Ford
  // distances are a checkable potential, not just a successful augmenting run.
  const potentials = Array(vertices).fill(0);
  let negativeCycle = false;
  for (let pass = 0; pass < vertices; pass += 1) {
    let changed = false;
    for (const arc of arcs) if (arc.residual > 0 && potentials[arc.from] + arc.cost < potentials[arc.to]) {
      potentials[arc.to] = potentials[arc.from] + arc.cost;
      changed = true;
    }
    if (!changed) break;
    if (pass === vertices - 1) negativeCycle = true;
  }
  const residual = arcs.filter(arc => arc.residual > 0).map(arc => ({
    ...arc,
    reducedCost: arc.cost + potentials[arc.from] - potentials[arc.to]
  }));
  const reachable = new Set([source]),
    queue = [source];
  for (let cursor = 0; cursor < queue.length; cursor += 1) {
    for (const edge of graph[queue[cursor]]) if (arcs[edge].residual > 0 && !reachable.has(arcs[edge].to)) {
      reachable.add(arcs[edge].to);
      queue.push(arcs[edge].to);
    }
  }
  const pairs = selected();
  return freeze({
    costs: costs.map(row => [...row]),
    required: target,
    workers,
    jobs,
    source,
    sink,
    trace,
    flow,
    feasible: flow === target,
    cost: sum(pairs.map(pair => pair.cost)),
    pairs,
    potentials,
    residual,
    negativeCycle,
    reachable: [...reachable],
    maximumCardinalityProved: !reachable.has(sink)
  });
}
export const VERTEX_COVER_PRESETS = freeze({
  triangle: {
    label: 'Weighted triangle',
    costs: [2, 3, 4],
    edges: [[0, 1], [1, 2], [0, 2]]
  },
  cycle: {
    label: 'Bipartite four-cycle',
    costs: [1, 1, 1, 1],
    edges: [[0, 1], [1, 2], [2, 3], [0, 3]]
  },
  star: {
    label: 'Expensive hub, cheaper leaves',
    costs: [9, 2, 2, 2],
    edges: [[0, 1], [0, 2], [0, 3]]
  }
});
export function vertexCoverState({
  costs = [2, 3, 4],
  edges = [[0, 1], [1, 2], [0, 2]]
} = {}) {
  if (!Array.isArray(costs) || costs.length < 1 || costs.length > 6) throw new RangeError('Use one to six vertices.');
  costs.forEach(cost => integer(cost, 0, 1000, 'Vertex cost'));
  if (!Array.isArray(edges) || edges.length > 15) throw new RangeError('Use at most fifteen simple undirected edges.');
  const canonical = new Set();
  edges.forEach(edge => {
    if (!Array.isArray(edge) || edge.length !== 2) throw new RangeError('Each edge needs two vertices.');
    edge.forEach(vertex => integer(vertex, 0, costs.length - 1, 'Edge endpoint'));
    if (edge[0] === edge[1]) throw new RangeError('This cover model is loopless.');
    const key = [...edge].sort((a, b) => a - b).join(':');
    if (canonical.has(key)) throw new RangeError('Duplicate undirected edge.');
    canonical.add(key);
  });
  const count = costs.length;
  let exactCost = Infinity,
    exactMask = 0,
    fractionalCost = Infinity,
    fractional = [];
  for (let mask = 0; mask < 1 << count; mask += 1) if (edges.every(([a, b]) => mask & 1 << a || mask & 1 << b)) {
    const cost = sum(indices(mask, count).map(vertex => costs[vertex]));
    if (cost < exactCost) {
      exactCost = cost;
      exactMask = mask;
    }
  }
  // The lesson explains half integrality. This 3^n enumeration is solely the
  // bounded LP oracle in the picture; it is not a general-purpose LP solver.
  for (let code = 0; code < 3 ** count; code += 1) {
    let remaining = code;
    const values = Array.from({
      length: count
    }, () => {
      const value = remaining % 3 / 2;
      remaining = Math.floor(remaining / 3);
      return value;
    });
    if (!edges.every(([a, b]) => values[a] + values[b] >= 1)) continue;
    const objective = sum(values.map((value, index) => value * costs[index]));
    if (objective < fractionalCost || objective === fractionalCost && values.filter(value => value === .5).length > fractional.filter(value => value === .5).length) {
      fractionalCost = objective;
      fractional = values;
    }
  }
  const rounded = fractional.map((value, vertex) => value >= .5 ? vertex : null).filter(vertex => vertex !== null);
  const loads = Array(edges.length).fill(0),
    totals = Array(count).fill(0),
    selected = new Set(costs.map((cost, vertex) => cost === 0 ? vertex : null).filter(vertex => vertex !== null));
  const trace = [];
  const snapshot = (edge, increment) => trace.push({
    edge,
    increment,
    loads: [...loads],
    totals: [...totals],
    selected: [...selected],
    lower: sum(loads),
    cost: sum([...selected].map(vertex => costs[vertex])),
    uncovered: edges.map(([a, b], index) => !selected.has(a) && !selected.has(b) ? index : null).filter(index => index !== null)
  });
  snapshot(null, 0);
  while (trace.at(-1).uncovered.length) {
    const edge = trace.at(-1).uncovered[0],
      [a, b] = edges[edge];
    const increment = Math.min(costs[a] - totals[a], costs[b] - totals[b]);
    loads[edge] += increment;
    totals[a] += increment;
    totals[b] += increment;
    if (totals[a] === costs[a]) selected.add(a);
    if (totals[b] === costs[b]) selected.add(b);
    snapshot(edge, increment);
  }
  return freeze({
    costs: [...costs],
    edges: edges.map(edge => [...edge]),
    exactCost,
    exactMask,
    fractionalCost,
    fractional,
    rounded,
    roundedCost: sum(rounded.map(vertex => costs[vertex])),
    trace,
    final: trace.at(-1)
  });
}
