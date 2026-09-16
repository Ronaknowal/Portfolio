/** Pure models for the Bayesian Networks & Causal Graphical Models lesson.
 *
 * Three categories of number live here and are never mixed:
 *
 *   1. Exact declared constructions. The alarm network, the maintenance service
 *      model, the frontdoor example, the two structural causal models and the
 *      marginal-MAP masses are invented teaching quantities with exactly
 *      specified inputs. They are not measurements of crime, engineering,
 *      medicine or anything else.
 *   2. Recorded observations. The Wine cultivar results live in
 *      `bayesnet-data.js`, which is generated from the dataset this lesson
 *      serves. This module supplies the inference that reads them.
 *   3. Learner-edited hypotheticals. Anything an investigation lets a learner
 *      change is labelled as edited input at the point it is shown.
 *
 * Two conventions are load-bearing throughout.
 *
 *   * A probability that has no value is `null` together with a stated reason.
 *     It is never 0, never a uniform fallback and never a tiny denominator.
 *     Conditioning on an event of probability zero has no posterior.
 *   * A graph is a list of `[parent, child]` pairs. A *path* is a sequence of
 *     node names joined by edges in either direction; it is not restricted to
 *     directed ancestry. Every separation verdict and every drawn highlight in
 *     this lesson comes from one function, `dSeparation`, so the rule a figure
 *     draws and the rule a lab grades cannot drift apart.
 *
 * Anything a figure draws geometrically is computed here, not in the component,
 * so that scripts/verify-bayesnet-models.mjs can assert it: node positions,
 * edge endpoints trimmed to the node circles, arrowhead triangles, the polyline
 * a highlighted path follows, and the bar geometry of the two population lanes.
 */

/* ============================================================ input limits */

/**
 * The editable controls, declared once.
 *
 * The component reads these to configure its number fields and the verifier
 * reads the same object to check that every value the prose asks a learner to
 * type is actually reachable. Hard-coding the step in both places made the
 * check agree with a copy of the control rather than with the control, so a
 * changed step would not have been caught by the check written to catch it.
 */
export const controlSteps = {
  rootPrior: { decimals: 4, minimum: 0, maximum: 1, step: '0.0001' },
  callerRow: { decimals: 3, minimum: 0, maximum: 1, step: '0.001' },
  serviceProbability: { decimals: 3, minimum: 0, maximum: 1, step: '0.001' },
  measurement: { decimals: 2, minimum: 0, maximum: 100, step: '0.01' },
};

export const limits = {
  probability: { minimum: 0, maximum: 1 },
  probabilityStep: 0.001,
  measurement: { minimum: 0, maximum: 100 },
  maximumNodes: 8,
  maximumEdges: 12,
  maximumPathNodes: 8,
  tolerance: 1e-12,
};

/** A number a learner typed must be a real number before it enters a model. */
export function checkFinite(value, name) {
  if (typeof value !== 'number' || !Number.isFinite(value)) {
    throw new RangeError(`${name} must be a finite number, not ${value}.`);
  }
  return value;
}

/** A probability outside [0, 1] is not a probability. Refuse it; do not clamp. */
export function checkProbability(value, name) {
  checkFinite(value, name);
  if (value < limits.probability.minimum || value > limits.probability.maximum) {
    throw new RangeError(`${name} must lie between 0 and 1, not ${value}.`);
  }
  return value;
}

export function checkState(value, name) {
  if (value !== 0 && value !== 1) throw new RangeError(`${name} must be the state 0 or 1, not ${value}.`);
  return value;
}

/* =================================================== graphs and d-separation */

/** Every node named by any edge, in a deterministic order. */
export function graphNodes(edges) {
  const seen = [];
  edges.forEach(([from, to]) => {
    if (!seen.includes(from)) seen.push(from);
    if (!seen.includes(to)) seen.push(to);
  });
  return seen;
}

/** Refuse a self-edge, a repeated edge, a reversed duplicate or a directed cycle. */
export function checkGraph(edges) {
  if (!Array.isArray(edges)) throw new RangeError('A graph must be a list of [parent, child] pairs.');
  if (edges.length > limits.maximumEdges) {
    throw new RangeError(`This editor holds at most ${limits.maximumEdges} edges; ${edges.length} were given.`);
  }
  const seen = new Set();
  edges.forEach(([from, to]) => {
    if (typeof from !== 'string' || typeof to !== 'string' || !from || !to) {
      throw new RangeError('Every edge needs two node names.');
    }
    if (from === to) throw new RangeError(`${from} cannot point at itself.`);
    if (seen.has(`${from}>${to}`)) throw new RangeError(`The edge ${from} to ${to} is already present.`);
    if (seen.has(`${to}>${from}`)) throw new RangeError(`${to} to ${from} is already present, so ${from} to ${to} would be a two-node cycle.`);
    seen.add(`${from}>${to}`);
  });
  const nodes = graphNodes(edges);
  if (nodes.length > limits.maximumNodes) {
    throw new RangeError(`This editor holds at most ${limits.maximumNodes} nodes; ${nodes.length} were given.`);
  }
  // Kahn's algorithm: a graph with no removable source has a directed cycle.
  const remaining = new Set(nodes);
  const live = edges.slice();
  let progress = true;
  while (progress) {
    progress = false;
    for (const node of [...remaining]) {
      if (!live.some(([, to]) => to === node)) {
        remaining.delete(node);
        for (let index = live.length - 1; index >= 0; index -= 1) {
          if (live[index][0] === node) live.splice(index, 1);
        }
        progress = true;
      }
    }
  }
  if (remaining.size) {
    throw new RangeError(`These arrows form a directed cycle through ${[...remaining].sort().join(', ')}.`);
  }
  return edges;
}

export function parentsOf(edges, node) {
  return edges.filter(([, to]) => to === node).map(([from]) => from);
}

export function childrenOf(edges, node) {
  return edges.filter(([from]) => from === node).map(([, to]) => to);
}

/** Every node reachable from `node` by following arrows. `node` itself is excluded. */
export function descendantsOf(edges, node) {
  const found = new Set();
  const stack = [node];
  while (stack.length) {
    const current = stack.pop();
    childrenOf(edges, current).forEach(child => {
      if (!found.has(child)) {
        found.add(child);
        stack.push(child);
      }
    });
  }
  found.delete(node);
  return [...found].sort();
}

/** Every node that can reach `node` by following arrows. `node` itself is excluded. */
export function ancestorsOf(edges, node) {
  const found = new Set();
  const stack = [node];
  while (stack.length) {
    const current = stack.pop();
    parentsOf(edges, current).forEach(parent => {
      if (!found.has(parent)) {
        found.add(parent);
        stack.push(parent);
      }
    });
  }
  found.delete(node);
  return [...found].sort();
}

/** Every simple undirected path from `start` to `end`, in a deterministic order. */
export function simplePaths(edges, start, end) {
  const nodes = graphNodes(edges);
  if (!nodes.includes(start)) throw new RangeError(`${start} is not a node of this graph.`);
  if (!nodes.includes(end)) throw new RangeError(`${end} is not a node of this graph.`);
  if (start === end) throw new RangeError('Choose two different endpoints.');
  const neighbours = {};
  nodes.forEach(node => { neighbours[node] = []; });
  edges.forEach(([from, to]) => {
    if (!neighbours[from].includes(to)) neighbours[from].push(to);
    if (!neighbours[to].includes(from)) neighbours[to].push(from);
  });
  nodes.forEach(node => neighbours[node].sort());
  const found = [];
  const walk = path => {
    const last = path[path.length - 1];
    if (last === end) {
      found.push(path.slice());
      return;
    }
    if (path.length >= limits.maximumPathNodes) return;
    neighbours[last].forEach(next => {
      if (!path.includes(next)) walk([...path, next]);
    });
  };
  walk([start]);
  return found;
}

/**
 * Pearl's rule for one path, reported node by node so a figure can draw the
 * reason rather than only the verdict.
 *
 * An interior node is a **collider** on this path when both of its path
 * neighbours point into it. A collider lets the path through only if the
 * collider itself or one of its descendants is observed. Every other interior
 * node is a non-collider, and it blocks the path exactly when it is observed.
 */
export function pathStatus(edges, path, observed) {
  const observedSet = new Set(observed);
  const directed = new Set(edges.map(([from, to]) => `${from}>${to}`));
  const interior = [];
  for (let index = 1; index < path.length - 1; index += 1) {
    const left = path[index - 1];
    const middle = path[index];
    const right = path[index + 1];
    const collider = directed.has(`${left}>${middle}`) && directed.has(`${right}>${middle}`);
    const openDescendants = descendantsOf(edges, middle).filter(name => observedSet.has(name));
    const middleObserved = observedSet.has(middle);
    let blocks = false;
    let reason = '';
    if (collider) {
      blocks = !middleObserved && openDescendants.length === 0;
      reason = blocks
        ? `the collider ${middle} is unobserved and no descendant of it is observed`
        : middleObserved
          ? `the collider ${middle} is observed, which opens it`
          : `the collider ${middle} has the observed descendant ${openDescendants[0]}, which opens it`;
    } else {
      blocks = middleObserved;
      reason = blocks
        ? `the non-collider ${middle} is observed`
        : `the non-collider ${middle} is unobserved, so it passes`;
    }
    interior.push({
      node: middle, position: index, collider, observed: middleObserved,
      observedDescendants: openDescendants, blocks, reason,
    });
  }
  const blocker = interior.find(entry => entry.blocks) ?? null;
  return {
    path: path.slice(),
    interior,
    blocked: Boolean(blocker),
    blockerNode: blocker ? blocker.node : null,
    // An active path has to say why each of its nodes let the path through, not
    // merely that they did. "Every node passes" hides the collider-descendant
    // rule, which is the one a learner most often gets wrong.
    reason: blocker
      ? blocker.reason
      : interior.length === 0
        ? 'the two variables are joined by a single edge, so no node can block this path'
        : interior.map(entry => entry.reason).join('; '),
  };
}

/**
 * The one separation routine in this lesson. Both the highlighted drawing and
 * the graded verdict of the path investigation call it, so the rule that is
 * drawn is by construction the rule that is applied.
 */
export function dSeparation(edges, start, end, observed) {
  checkGraph(edges);
  const observedList = [...new Set(observed)].sort();
  const nodes = graphNodes(edges);
  observedList.forEach(node => {
    if (!nodes.includes(node)) throw new RangeError(`${node} is not a node of this graph.`);
  });
  if (observedList.includes(start) || observedList.includes(end)) {
    throw new RangeError('An endpoint of the query cannot also be an observed variable.');
  }
  const paths = simplePaths(edges, start, end).map(path => pathStatus(edges, path, observedList));
  const active = paths.filter(entry => !entry.blocked);
  return {
    start, end, observed: observedList, paths, active,
    separated: active.length === 0,
    verdict: active.length === 0 ? 'guaranteed-independent' : 'dependence-possible',
    summary: active.length === 0
      ? paths.length === 0
        ? `No path joins ${start} and ${end}, so the graph guarantees independence.`
        : `All ${paths.length} path${paths.length === 1 ? '' : 's'} between ${start} and ${end} are blocked, so the graph guarantees independence.`
      : `${active.length} of ${paths.length} path${paths.length === 1 ? '' : 's'} stay${active.length === 1 ? 's' : ''} active, so the graph leaves dependence possible.`,
  };
}

/** The paths the backdoor criterion inspects: those whose first arrow points into the treatment. */
export function backdoorPaths(edges, treatment, outcome) {
  const directed = new Set(edges.map(([from, to]) => `${from}>${to}`));
  return simplePaths(edges, treatment, outcome)
    .filter(path => directed.has(`${path[1]}>${path[0]}`));
}

/**
 * Pearl's sufficient backdoor criterion for the total effect of `treatment` on
 * `outcome`: the adjustment set may contain no descendant of the treatment, and
 * it must block every backdoor path.
 */
export function backdoorCriterion(edges, treatment, outcome, adjustment) {
  checkGraph(edges);
  const nodes = graphNodes(edges);
  const set = [...new Set(adjustment)].sort();
  set.forEach(node => {
    if (!nodes.includes(node)) throw new RangeError(`${node} is not a node of this graph.`);
  });
  if (set.includes(treatment) || set.includes(outcome)) {
    throw new RangeError('An adjustment set may not contain the treatment or the outcome itself.');
  }
  const treatmentDescendants = descendantsOf(edges, treatment);
  const descendantViolations = set.filter(node => treatmentDescendants.includes(node));
  const paths = backdoorPaths(edges, treatment, outcome)
    .map(path => pathStatus(edges, path, set));
  const openPaths = paths.filter(entry => !entry.blocked);
  const valid = descendantViolations.length === 0 && openPaths.length === 0;
  return {
    treatment, outcome, adjustment: set,
    treatmentDescendants, descendantViolations,
    backdoorPaths: paths, openPaths, valid,
    reason: descendantViolations.length
      ? `${descendantViolations.join(', ')} ${descendantViolations.length === 1 ? 'is a descendant' : 'are descendants'} of ${treatment}.`
      : openPaths.length
        ? `The backdoor path ${openPaths[0].path.join('–')} stays open: ${openPaths[0].reason}.`
        : paths.length === 0
          ? `No backdoor path leaves ${treatment}, so the empty set already satisfies the criterion.`
          : `No member is a descendant of ${treatment}, and all ${paths.length} backdoor path${paths.length === 1 ? ' is' : 's are'} blocked.`,
  };
}

/** Parents, children, and the other parents of those children. */
export function markovBlanket(edges, node) {
  checkGraph(edges);
  if (!graphNodes(edges).includes(node)) throw new RangeError(`${node} is not a node of this graph.`);
  const blanket = new Set();
  parentsOf(edges, node).forEach(name => blanket.add(name));
  const children = childrenOf(edges, node);
  children.forEach(child => {
    blanket.add(child);
    parentsOf(edges, child).forEach(coparent => blanket.add(coparent));
  });
  blanket.delete(node);
  return { node, children, blanket: [...blanket].sort() };
}

/** Node i with r_i states and q_i parent configurations contributes q_i(r_i − 1). */
export function freeParameters(specification) {
  const rows = specification.map(({ name, states, parentStates }) => {
    if (!Number.isInteger(states) || states < 2) throw new RangeError(`${name} needs at least two states.`);
    const configurations = parentStates.reduce((product, count) => {
      if (!Number.isInteger(count) || count < 2) throw new RangeError(`A parent of ${name} needs at least two states.`);
      return product * count;
    }, 1);
    return { name, states, configurations, stored: configurations * states, free: configurations * (states - 1) };
  });
  const totalStates = specification.reduce((product, entry) => product * entry.states, 1);
  return {
    rows,
    stored: rows.reduce((sum, row) => sum + row.stored, 0),
    free: rows.reduce((sum, row) => sum + row.free, 0),
    jointEntries: totalStates,
    jointFree: totalStates - 1,
  };
}

/* =========================================== binary networks and enumeration */

/** The parent configuration key a CPT row is stored under. */
export function parentKey(network, node, assignment) {
  return network.parents[node].map(parent => assignment[parent]).join(',');
}

/** P(node = 1 | its parents) for the given assignment. */
export function chanceOf(network, node, assignment) {
  const key = parentKey(network, node, assignment);
  const value = network.chance[node][key];
  if (value === undefined) throw new RangeError(`${node} has no table row for parent setting "${key}".`);
  return value;
}

/** The five factors one world uses, in network order, for the assembly figure. */
export function worldFactors(network, assignment) {
  return network.nodes.map(node => {
    const chance = chanceOf(network, node, assignment);
    const used = assignment[node] ? chance : 1 - chance;
    return {
      node,
      state: assignment[node],
      parents: network.parents[node],
      parentStates: network.parents[node].map(parent => assignment[parent]),
      parentKey: parentKey(network, node, assignment),
      chanceOfOne: chance,
      value: used,
    };
  });
}

export function jointProbability(network, assignment) {
  return worldFactors(network, assignment).reduce((product, factor) => product * factor.value, 1);
}

/** Every 2^n assignment, in a deterministic order, with its joint probability. */
export function enumerateWorlds(network) {
  const count = network.nodes.length;
  const worlds = [];
  for (let code = 0; code < 2 ** count; code += 1) {
    const assignment = {};
    network.nodes.forEach((node, index) => {
      assignment[node] = (code >> (count - 1 - index)) & 1;
    });
    worlds.push({ code, assignment, probability: jointProbability(network, assignment) });
  }
  return worlds;
}

/**
 * Marginalise the hidden variables, condition on the evidence, and report the
 * posterior over `target`. Evidence of probability zero has no posterior: the
 * result is `null` with the reason, never 0 and never one half.
 */
export function queryPosterior(network, evidence, target = 'B') {
  Object.entries(evidence).forEach(([node, state]) => {
    if (!network.nodes.includes(node)) throw new RangeError(`${node} is not a variable of this network.`);
    checkState(state, `the evidence on ${node}`);
  });
  if (!network.nodes.includes(target)) throw new RangeError(`${target} is not a variable of this network.`);
  if (target in evidence) throw new RangeError(`${target} is the query variable, so it cannot also be evidence.`);
  const mass = [0, 0];
  let compatible = 0;
  enumerateWorlds(network).forEach(world => {
    if (Object.entries(evidence).every(([node, state]) => world.assignment[node] === state)) {
      compatible += 1;
      mass[world.assignment[target]] += world.probability;
    }
  });
  const evidenceProbability = mass[0] + mass[1];
  const usable = evidenceProbability > 0;
  return {
    target, evidence: { ...evidence },
    mass, evidenceProbability, compatibleWorlds: compatible,
    posterior: usable ? mass[1] / evidenceProbability : null,
    undefinedBecause: usable ? null
      : 'this evidence has probability zero under the current tables, so there is no posterior to report',
  };
}

/* ================================================== factors and elimination */

/** A factor is a table over binary variables, indexed by counting in that scope's order. */
export function makeFactor(scope, values) {
  if (values.length !== 2 ** scope.length) {
    throw new RangeError(`A factor over ${scope.join(', ')} needs ${2 ** scope.length} cells, not ${values.length}.`);
  }
  return { scope: scope.slice(), values: values.slice() };
}

export function factorCells(factor) {
  return factor.values.map((value, index) => {
    const states = {};
    factor.scope.forEach((name, position) => {
      states[name] = (index >> (factor.scope.length - 1 - position)) & 1;
    });
    return { index, states, value };
  });
}

function factorAt(factor, states) {
  let index = 0;
  factor.scope.forEach(name => { index = index * 2 + states[name]; });
  return factor.values[index];
}

export function multiplyFactors(left, right) {
  const scope = [...left.scope, ...right.scope.filter(name => !left.scope.includes(name))];
  const values = new Array(2 ** scope.length).fill(0);
  for (let index = 0; index < values.length; index += 1) {
    const states = {};
    scope.forEach((name, position) => { states[name] = (index >> (scope.length - 1 - position)) & 1; });
    values[index] = factorAt(left, states) * factorAt(right, states);
  }
  return makeFactor(scope, values);
}

export function sumOut(factor, variable) {
  if (!factor.scope.includes(variable)) {
    throw new RangeError(`${variable} is not in the scope ${factor.scope.join(', ') || '(empty)'}.`);
  }
  const scope = factor.scope.filter(name => name !== variable);
  const values = new Array(2 ** scope.length).fill(0);
  for (let index = 0; index < values.length; index += 1) {
    const states = {};
    scope.forEach((name, position) => { states[name] = (index >> (scope.length - 1 - position)) & 1; });
    values[index] = [0, 1].reduce((sum, state) => sum + factorAt(factor, { ...states, [variable]: state }), 0);
  }
  return makeFactor(scope, values);
}

export function reduceByEvidence(factor, evidence) {
  const fixed = factor.scope.filter(name => name in evidence);
  if (!fixed.length) return { scope: factor.scope.slice(), values: factor.values.slice() };
  const scope = factor.scope.filter(name => !(name in evidence));
  const values = new Array(2 ** scope.length).fill(0);
  for (let index = 0; index < values.length; index += 1) {
    const states = { ...evidence };
    scope.forEach((name, position) => { states[name] = (index >> (scope.length - 1 - position)) & 1; });
    values[index] = factorAt(factor, states);
  }
  return makeFactor(scope, values);
}

/** The network's conditional probability tables as factors over child and parents. */
export function networkFactors(network) {
  return network.nodes.map(node => {
    const scope = [...network.parents[node], node];
    const values = new Array(2 ** scope.length).fill(0);
    for (let index = 0; index < values.length; index += 1) {
      const states = {};
      scope.forEach((name, position) => { states[name] = (index >> (scope.length - 1 - position)) & 1; });
      const chance = chanceOf(network, node, states);
      values[index] = states[node] ? chance : 1 - chance;
    }
    return { node, factor: makeFactor(scope, values) };
  });
}

/**
 * Variable elimination with a declared order. Each step collects every factor
 * that mentions the variable, multiplies them, sums the variable out and puts
 * the result back — the same four steps the prose lists.
 */
export function eliminationRun(network, evidence, order, target = 'B') {
  const eliminate = order.filter(name => name !== target && !(name in evidence));
  let factors = networkFactors(network).map(entry => reduceByEvidence(entry.factor, evidence));
  const steps = [];
  eliminate.forEach(variable => {
    const involved = factors.filter(factor => factor.scope.includes(variable));
    const untouched = factors.filter(factor => !factor.scope.includes(variable));
    if (!involved.length) {
      steps.push({ variable, inputScopes: [], resultScope: [], cells: [], note: 'no remaining factor mentions it' });
      return;
    }
    const product = involved.reduce((left, right) => multiplyFactors(left, right));
    const result = sumOut(product, variable);
    steps.push({
      variable,
      inputScopes: involved.map(factor => factor.scope.slice()),
      productScope: product.scope.slice(),
      productCells: product.values.length,
      resultScope: result.scope.slice(),
      cells: factorCells(result),
      note: '',
    });
    factors = [...untouched, result];
  });
  const final = factors.reduce((left, right) => multiplyFactors(left, right));
  const mass = [0, 1].map(state => factorAt(final, { [target]: state }));
  const total = mass[0] + mass[1];
  return {
    order: eliminate, steps, mass, evidenceProbability: total,
    posterior: total > 0 ? mass[1] / total : null,
    undefinedBecause: total > 0 ? null
      : 'this evidence has probability zero under the current tables, so there is no posterior to report',
  };
}

/** The induced width of one elimination order on the moral graph, and its largest dense factor. */
export function inducedWidth(network, order) {
  const adjacency = {};
  network.nodes.forEach(node => { adjacency[node] = new Set(); });
  network.nodes.forEach(node => {
    const family = [...network.parents[node], node];
    family.forEach(left => family.forEach(right => {
      if (left !== right) adjacency[left].add(right);
    }));
  });
  let width = 0;
  const removed = new Set();
  const fill = [];
  order.forEach(node => {
    const live = [...adjacency[node]].filter(name => !removed.has(name)).sort();
    width = live.length > width ? live.length : width;
    live.forEach(left => live.forEach(right => {
      if (left !== right && !adjacency[left].has(right)) {
        adjacency[left].add(right);
        adjacency[right].add(left);
        if (left < right) fill.push([left, right]);
      }
    }));
    removed.add(node);
  });
  return { width, fillEdges: fill, largestBinaryFactorCells: 2 ** (width + 1) };
}

/* =============================================== intervention and adjustment */

/**
 * The maintenance model Z → X → Y with Z → Y. `assignment[z]` is P(X = 1 | Z = z)
 * and `outcome[x][z]` is P(Y = 1 | X = x, Z = z).
 *
 * Observation selects units whose assignment mechanism produced X = x, so the
 * load mixture inside each group changes. Intervention replaces that mechanism
 * and keeps the population's own load mixture.
 */
export function serviceModel({ assignment, outcome, loadPrior = [0.5, 0.5] }) {
  assignment.forEach((value, index) => checkProbability(value, `P(X=1 | Z=${index})`));
  outcome.forEach((row, x) => row.forEach((value, z) => checkProbability(value, `P(Y=1 | X=${x}, Z=${z})`)));
  loadPrior.forEach((value, index) => checkProbability(value, `P(Z=${index})`));
  const priorSum = loadPrior[0] + loadPrior[1];
  if (priorSum <= 0 || Math.abs(priorSum - 1) > 1e-9) {
    throw new RangeError('The two load probabilities must sum to one.');
  }
  const lanes = [0, 1].map(x => {
    const weights = [0, 1].map(z => loadPrior[z] * (x ? assignment[z] : 1 - assignment[z]));
    const total = weights[0] + weights[1];
    const supported = total > 0;
    const shares = supported ? weights.map(weight => weight / total) : [null, null];
    return {
      treatment: x,
      observedWeights: weights,
      observedMass: total,
      observedShares: shares,
      observedRisk: supported ? shares.reduce((sum, share, z) => sum + share * outcome[x][z], 0) : null,
      interventionShares: loadPrior.slice(),
      interventionRisk: loadPrior.reduce((sum, share, z) => sum + share * outcome[x][z], 0),
      undefinedBecause: supported ? null
        : `no unit in this population receives treatment ${x}, so its observed risk has an empty denominator`,
    };
  });
  const bothObserved = lanes.every(lane => lane.observedRisk !== null);
  const associationDifference = bothObserved ? lanes[1].observedRisk - lanes[0].observedRisk : null;
  const causalDifference = lanes[1].interventionRisk - lanes[0].interventionRisk;
  return {
    lanes, assignment: assignment.slice(), outcome: outcome.map(row => row.slice()), loadPrior: loadPrior.slice(),
    associationDifference, causalDifference,
    bias: associationDifference === null ? null : associationDifference - causalDifference,
    positivity: assignment.every(value => value > 0 && value < 1),
    positivityNote: assignment.every(value => value > 0 && value < 1)
      ? 'Both treatment levels occur at both load values, so adjustment from observed data has the strata it needs.'
      : 'At least one load stratum never receives one of the treatment levels. The fully specified model still '
        + 'computes an interventional answer, but observed data alone cannot supply the missing response cells.',
  };
}

/**
 * The frontdoor example: U → X, U → Y, X → M → Y, with U hidden from the
 * observed summaries. Returns the observed joint, the inner mediator responses,
 * the frontdoor estimate, the truncated-factorisation answer computed from the
 * complete model, and the plain observational probabilities.
 */
export const declaredFrontdoorParameters = {
  latentPrior: 0.5, chanceX: [0.2, 0.8], chanceM: [0.1, 0.9],
  chanceY: [[0.05, 0.4], [0.5, 0.9]],
};
export function frontdoorModel({
  latentPrior = declaredFrontdoorParameters.latentPrior,
  chanceX = declaredFrontdoorParameters.chanceX,
  chanceM = declaredFrontdoorParameters.chanceM,
  chanceY = declaredFrontdoorParameters.chanceY,
} = {}) {
  checkProbability(latentPrior, 'P(U=1)');
  chanceX.forEach((value, u) => checkProbability(value, `P(X=1 | U=${u})`));
  chanceM.forEach((value, x) => checkProbability(value, `P(M=1 | X=${x})`));
  chanceY.forEach((row, m) => row.forEach((value, u) => checkProbability(value, `P(Y=1 | M=${m}, U=${u})`)));
  const joint = {};
  [0, 1].forEach(u => [0, 1].forEach(x => [0, 1].forEach(m => [0, 1].forEach(y => {
    joint[`${u}${x}${m}${y}`] =
      (u ? latentPrior : 1 - latentPrior)
      * (x ? chanceX[u] : 1 - chanceX[u])
      * (m ? chanceM[x] : 1 - chanceM[x])
      * (y ? chanceY[m][u] : 1 - chanceY[m][u]);
  }))));
  const observed = (x, m, y) => [0, 1].reduce((sum, u) => sum + joint[`${u}${x}${m}${y}`], 0);
  const marginX = [0, 1].map(x => [0, 1].reduce((sum, m) => sum + observed(x, m, 0) + observed(x, m, 1), 0));
  const inner = [0, 1].map(m => [0, 1].reduce((sum, xp) => {
    const cellMass = observed(xp, m, 0) + observed(xp, m, 1);
    if (cellMass <= 0) return sum;
    return sum + (observed(xp, m, 1) / cellMass) * marginX[xp];
  }, 0));
  const queries = [0, 1].map(x => {
    const mediatorGiven = [0, 1].map(m => (observed(x, m, 0) + observed(x, m, 1)) / marginX[x]);
    const frontdoor = [0, 1].reduce((sum, m) => sum + mediatorGiven[m] * inner[m], 0);
    const truncated = [0, 1].reduce((sum, u) => [0, 1].reduce((inner2, m) => inner2
      + (u ? latentPrior : 1 - latentPrior) * (m ? chanceM[x] : 1 - chanceM[x]) * chanceY[m][u], sum), 0);
    const observational = [0, 1].reduce((sum, m) => sum + observed(x, m, 1), 0) / marginX[x];
    return { treatment: x, mediatorGiven, frontdoor, truncated, observational };
  });
  return {
    joint, marginX, inner, queries,
    observedRiskGivenMediatorAndTreatment: [0, 1].map(m => [0, 1].map(x => {
      const cellMass = observed(x, m, 0) + observed(x, m, 1);
      return cellMass > 0 ? observed(x, m, 1) / cellMass : null;
    })),
    frontdoorDifference: queries[1].frontdoor - queries[0].frontdoor,
    observationalDifference: queries[1].observational - queries[0].observational,
    agreesWithTruncatedFactorisation: queries.every(q => Math.abs(q.frontdoor - q.truncated) < 1e-12),
  };
}

/** The three frontdoor conditions, checked on a stated graph rather than asserted. */
export function frontdoorConditions(edges, treatment, mediator, outcome) {
  checkGraph(edges);
  const directed = new Set(edges.map(([from, to]) => `${from}>${to}`));
  const directedPaths = [];
  const walk = path => {
    const last = path[path.length - 1];
    if (last === outcome) { directedPaths.push(path.slice()); return; }
    childrenOf(edges, last).forEach(child => { if (!path.includes(child)) walk([...path, child]); });
  };
  walk([treatment]);
  const intercepts = directedPaths.every(path => path.includes(mediator));
  const treatmentToMediator = simplePaths(edges, treatment, mediator)
    .filter(path => directed.has(`${path[1]}>${path[0]}`))
    .map(path => pathStatus(edges, path, []));
  const mediatorToOutcome = simplePaths(edges, mediator, outcome)
    .filter(path => directed.has(`${path[1]}>${path[0]}`))
    .map(path => pathStatus(edges, path, [treatment]));
  return {
    directedPaths, intercepts,
    treatmentToMediatorOpen: treatmentToMediator.filter(entry => !entry.blocked),
    mediatorToOutcomeOpen: mediatorToOutcome.filter(entry => !entry.blocked),
    satisfied: intercepts
      && treatmentToMediator.every(entry => entry.blocked)
      && mediatorToOutcome.every(entry => entry.blocked),
  };
}

/* ======================================== counterfactuals and query families */

/**
 * Two structural causal models with the same observational and population
 * interventional distributions over (X, Y), and different paired outcomes.
 * Model A sets Y = U; model B sets Y = X XOR U.
 */
export function counterfactualPair() {
  const units = [0, 1];
  const models = [
    { name: 'A', assignment: 'Y = U', respond: (x, u) => u },
    { name: 'B', assignment: 'Y = X XOR U', respond: (x, u) => (x === u ? 0 : 1) },
  ].map(model => {
    const rows = units.map(u => ({
      unit: u, weight: 0.5,
      outcomeUnderZero: model.respond(0, u),
      outcomeUnderOne: model.respond(1, u),
    }));
    return {
      ...model, rows,
      averageUnderZero: rows.reduce((sum, row) => sum + row.weight * row.outcomeUnderZero, 0),
      averageUnderOne: rows.reduce((sum, row) => sum + row.weight * row.outcomeUnderOne, 0),
    };
  });
  return { models, agreeOnAverages: models.every(m => m.averageUnderZero === 0.5 && m.averageUnderOne === 0.5) };
}

/** Abduction, action, prediction for one observed unit under both models. */
export function counterfactualOfUnit(observedTreatment, observedOutcome, newTreatment) {
  checkState(observedTreatment, 'the observed treatment');
  checkState(observedOutcome, 'the observed outcome');
  checkState(newTreatment, 'the changed treatment');
  return counterfactualPair().models.map(model => {
    const consistent = [0, 1].filter(u => (model.name === 'A' ? u : (observedTreatment === u ? 0 : 1)) === observedOutcome);
    if (consistent.length !== 1) {
      return { model: model.name, inferredExternalState: null, prediction: null,
        undefinedBecause: 'this observation does not pin down a single external state in this model' };
    }
    const u = consistent[0];
    const predicted = model.name === 'A' ? u : (newTreatment === u ? 0 : 1);
    return { model: model.name, inferredExternalState: u, prediction: predicted, undefinedBecause: null };
  });
}

/** A most probable world, a marginal-MAP answer, and the row masses that separate them. */
export function queryFamilies(masses) {
  const entries = Object.entries(masses).map(([key, value]) => {
    checkProbability(value, `the mass of ${key}`);
    return { query: Number(key[0]), hidden: Number(key[1]), value };
  });
  const total = entries.reduce((sum, entry) => sum + entry.value, 0);
  if (Math.abs(total - 1) > 1e-9) throw new RangeError(`These four masses sum to ${total}, not one.`);
  const best = entries.reduce((top, entry) => (entry.value > top.value ? entry : top));
  const rowMasses = [0, 1].map(q => entries.filter(entry => entry.query === q)
    .reduce((sum, entry) => sum + entry.value, 0));
  const marginalMap = rowMasses[1] > rowMasses[0] ? 1 : 0;
  return {
    entries, rowMasses,
    mostProbableWorld: { query: best.query, hidden: best.hidden, value: best.value },
    marginalMapQuery: marginalMap,
    marginalMapMass: rowMasses[marginalMap],
    agree: best.query === marginalMap,
  };
}

/* ======================================== reading a fitted classifier's tables */

/** Strictly above the fixed training median. Equality is below, not above. */
export function binariseSpecimen(values, medians) {
  if (values.length !== medians.length) throw new RangeError('A specimen needs one value per fitted median.');
  return values.map((value, index) => {
    checkFinite(value, `measurement ${index}`);
    return value > medians[index] ? 1 : 0;
  });
}

/**
 * Exact inference for the fitted naive-Bayes or tree-augmented network: keep
 * every one of the 2^4 feature states compatible with the visible measurements,
 * multiply the class prior by the four table entries, add, then normalise.
 * A hidden measurement is summed over, so its stored value cannot be read.
 */
export function classPosterior(model, states, visible) {
  const featureCount = model.parents.length;
  if (states.length !== featureCount) throw new RangeError('A state vector needs one entry per feature.');
  states.forEach((state, index) => checkState(state, `feature ${index}`));
  const visibleList = [...new Set(visible)].sort((a, b) => a - b);
  visibleList.forEach(index => {
    if (!Number.isInteger(index) || index < 0 || index >= featureCount) {
      throw new RangeError(`${index} is not one of this model's ${featureCount} measurements.`);
    }
  });
  const classCount = model.prior.length;
  const total = new Array(classCount).fill(0);
  let compatible = 0;
  for (let code = 0; code < 2 ** featureCount; code += 1) {
    const candidate = [];
    for (let index = 0; index < featureCount; index += 1) {
      candidate.push((code >> (featureCount - 1 - index)) & 1);
    }
    if (visibleList.some(index => candidate[index] !== states[index])) continue;
    compatible += 1;
    for (let klass = 0; klass < classCount; klass += 1) {
      let weight = model.prior[klass];
      for (let column = 0; column < featureCount; column += 1) {
        const parent = model.parents[column];
        const parentState = parent === null ? 0 : candidate[parent];
        weight *= model.tables[column][klass][parentState][candidate[column]];
      }
      total[klass] += weight;
    }
  }
  const mass = total.reduce((sum, value) => sum + value, 0);
  if (!(mass > 0)) {
    return { posterior: null, compatibleStates: compatible, visible: visibleList,
      undefinedBecause: 'every compatible state has probability zero under these tables' };
  }
  const posterior = total.map(value => value / mass);
  return {
    posterior, compatibleStates: compatible, visible: visibleList,
    leading: posterior.reduce((best, value, index) => (value > posterior[best] ? index : best), 0),
    entropyNats: entropyNats(posterior), undefinedBecause: null,
  };
}

/** Shannon entropy in nats, with the usual 0 log 0 = 0 convention. */
export function entropyNats(distribution) {
  return -distribution.reduce((sum, value) => (value > 0 ? sum + value * Math.log(value) : sum), 0);
}

/* ============================================== the rules the labs grade with
 *
 * Every graded comparison on this page reduces to one of the two functions
 * below, and they live here rather than in the component so that the verifier
 * can exercise them directly at their degenerate inputs. A grading rule that
 * only exists inside a React file is a rule nothing can assert.
 */

/** Numerical comparison tolerance, not a declaration of exact equality.
 * Structural support is tested against zero separately. */
export const unchangedTolerance = 1e-12;

/**
 * Did the graded quantity rise, fall, or stay within numerical tolerance?
 *
 * `null` means the quantity has no value, which is a fourth answer and not a
 * zero: a posterior conditioned on impossible evidence, or an observed risk
 * whose treatment group is empty. An absent new value is "undefined"; a newly
 * available value is "defined", without a numerical direction to an absent baseline.
 */
export function changeDirection(after, before) {
  if (after === null || after === undefined) return 'undefined';
  if (before === null || before === undefined) {
    checkFinite(after, 'the new value');
    return 'defined';
  }
  checkFinite(after, 'the new value');
  checkFinite(before, 'the previous value');
  const gap = after - before;
  const scale = Math.abs(before) > 1 ? Math.abs(before) : 1;
  if (Math.abs(gap) <= unchangedTolerance * scale) return 'unchanged';
  return gap > 0 ? 'higher' : 'lower';
}

/** Which of two candidate reveals leaves less uncertainty, or neither. */
export function compareUncertainty(firstNats, secondNats) {
  checkFinite(firstNats, 'the first remaining uncertainty');
  checkFinite(secondNats, 'the second remaining uncertainty');
  const gap = firstNats - secondNats;
  if (Math.abs(gap) <= unchangedTolerance) return 'equal';
  return gap < 0 ? 'first' : 'second';
}

/**
 * Reveal one candidate measurement or the other, and say which leaves less
 * uncertainty. Returns both resulting posteriors so the explanation shown to a
 * learner is drawn from the same call that graded them.
 */
export function purchaseComparison(model, states, visible, candidates) {
  const base = [...new Set(visible)].sort((a, b) => a - b);
  if (candidates[0] === candidates[1]) {
    const only = classPosterior(model, states, [...base, candidates[0]].sort((a, b) => a - b));
    return { first: only, second: only, outcome: 'equal', gap: 0 };
  }
  const first = classPosterior(model, states, [...base, candidates[0]].sort((a, b) => a - b));
  const second = classPosterior(model, states, [...base, candidates[1]].sort((a, b) => a - b));
  return {
    first, second,
    gap: first.entropyNats - second.entropyNats,
    outcome: compareUncertainty(first.entropyNats, second.entropyNats),
  };
}

/* ================================================== geometry a figure draws */

/**
 * A deterministic layered placement: a node's layer is its longest directed
 * distance from a root, and its position within the layer follows the order the
 * nodes were first named. No physics loop, no randomness, no animation.
 */
export function layeredLayout(edges, { width = 300, margin = 30, radius = 17, rowGap = 62 } = {}) {
  const nodes = graphNodes(edges);
  const layerOf = {};
  nodes.forEach(node => { layerOf[node] = 0; });
  for (let pass = 0; pass < nodes.length; pass += 1) {
    edges.forEach(([from, to]) => {
      if (layerOf[to] < layerOf[from] + 1) layerOf[to] = layerOf[from] + 1;
    });
  }
  const depth = nodes.reduce((most, node) => (layerOf[node] > most ? layerOf[node] : most), 0);
  const columns = [];
  for (let layer = 0; layer <= depth; layer += 1) {
    columns.push(nodes.filter(node => layerOf[node] === layer));
  }
  const widest = columns.reduce((most, column) => (column.length > most ? column.length : most), 1);
  const usableWidth = width - 2 * margin;
  const columnGap = widest > 1 ? usableWidth / (widest - 1) : usableWidth;
  // The circles shrink before they are allowed to touch. A fixed radius on an
  // eight-node graph either overlaps its neighbours or leaves an edge with no
  // drawable length, and both are silent failures in a figure that looks fine.
  const fitted = Math.min(radius, 0.42 * columnGap, 0.42 * rowGap);
  const height = 2 * margin + depth * rowGap;
  const positions = {};
  columns.forEach((column, layer) => {
    const y = margin + rowGap * layer;
    column.forEach((node, index) => {
      const x = column.length === 1
        ? width / 2
        : margin + (usableWidth * index) / (column.length - 1);
      positions[node] = { x, y, layer, indexInLayer: index };
    });
  });
  return { positions, columns, depth, radius: fitted, width, height, columnGap, rowGap };
}

/**
 * The shapes an edge has to miss.
 *
 * A node is not just its circle. A query endpoint carries a dashed badge five
 * units larger on every side, and an observed node carries a tag below it. The
 * first version of this router measured circles only, so an edge could clear
 * every circle and still cross a badge -- which is what the frontdoor figure's
 * U-to-Y edge did, visibly, while every assertion passed.
 */
export function nodeKeepOut(point, radius, { endpoint = false, observed = false } = {}) {
  const shapes = [{ kind: 'circle', x: point.x, y: point.y, radius }];
  if (endpoint) {
    shapes.push({ kind: 'rect', x: point.x, y: point.y, halfWidth: radius + 5, halfHeight: radius + 5 });
  }
  if (observed) {
    shapes.push({ kind: 'rect', x: point.x, y: point.y + radius + 10, halfWidth: 24, halfHeight: 7 });
  }
  return shapes;
}

/** Distance from a point to a shape's boundary; zero when the point is inside. */
export function distanceToShape(shape, x, y) {
  if (shape.kind === 'circle') {
    const gap = Math.hypot(x - shape.x, y - shape.y) - shape.radius;
    return gap > 0 ? gap : 0;
  }
  const dx = Math.abs(x - shape.x) - shape.halfWidth;
  const dy = Math.abs(y - shape.y) - shape.halfHeight;
  return Math.hypot(dx > 0 ? dx : 0, dy > 0 ? dy : 0);
}

/**
 * One drawn edge that is guaranteed to miss every shape it does not join.
 *
 * A layered layout puts a whole chain in one column, so an edge that skips a
 * layer runs straight through the nodes between them and through their labels:
 * correct in the DOM and wrong on screen. So an edge bows around them.
 *
 * Three things have to line up for the check on that bow to mean anything, and
 * the first version of this got all three wrong.
 *
 *   1. `trace` samples the curve the browser ACTUALLY draws. A quadratic
 *      trimmed to a sub-span has its own control point; reusing the full
 *      curve's control point describes a different curve, and the two diverged
 *      by up to nine pixels.
 *   2. Clearance is measured against the shapes that are drawn -- endpoint
 *      badges and observed tags as well as circles.
 *   3. The router's target and the verifier's threshold are one number,
 *      `clearance`, not two that can drift apart.
 *
 * If nothing in the ladder clears, the best bow is returned with
 * `cleared: false` rather than thrown: a crashed page is worse than a tight
 * edge, and the verifier asserts `cleared` on everything the lesson draws.
 */
export function edgeRoute(from, to, radius, obstacles = [], {
  arrowLength = 9, arrowWidth = 5.5, clearance = 3, samples = 64, prefer = 1,
} = {}) {
  const chord = Math.hypot(to.x - from.x, to.y - from.y);
  if (!(chord > 2 * radius)) {
    throw new RangeError('Two nodes are closer together than their own circles; the edge would have no visible length.');
  }
  const unitX = (to.x - from.x) / chord;
  const unitY = (to.y - from.y) / chord;
  const controlFor = bow => ({
    x: (from.x + to.x) / 2 - unitY * bow,
    y: (from.y + to.y) / 2 + unitX * bow,
  });
  const at = (control, t) => {
    const inverse = 1 - t;
    return {
      x: inverse * inverse * from.x + 2 * inverse * t * control.x + t * t * to.x,
      y: inverse * inverse * from.y + 2 * inverse * t * control.y + t * t * to.y,
    };
  };
  const slope = (control, t) => ({
    x: 2 * (1 - t) * (control.x - from.x) + 2 * t * (to.x - control.x),
    y: 2 * (1 - t) * (control.y - from.y) + 2 * t * (to.y - control.y),
  });
  /** The sub-arc over [t0, t1], as a quadratic in its own right. */
  const subCurve = (control, t0, t1) => {
    const startPoint = at(control, t0);
    const derivative = slope(control, t0);
    return {
      from: startPoint,
      control: {
        x: startPoint.x + (t1 - t0) * derivative.x / 2,
        y: startPoint.y + (t1 - t0) * derivative.y / 2,
      },
      to: at(control, t1),
    };
  };
  const sampleSub = (curve, count) => Array.from({ length: count + 1 }, (unused, index) => {
    const t = index / count;
    const inverse = 1 - t;
    return [
      inverse * inverse * curve.from.x + 2 * inverse * t * curve.control.x + t * t * curve.to.x,
      inverse * inverse * curve.from.y + 2 * inverse * t * curve.control.y + t * t * curve.to.y,
    ];
  });

  const ladder = [0];
  for (let step = 1; step <= 10; step += 1) ladder.push(prefer * step * 13, -prefer * step * 13);

  const build = bow => {
    const control = controlFor(bow);
    const distanceFrom = (target, t) => {
      const point = at(control, t);
      return Math.hypot(point.x - target.x, point.y - target.y);
    };
    const trim = (target, forward) => {
      let inside = forward ? 0 : 1;
      let outside = forward ? 1 : 0;
      for (let index = 0; index <= samples; index += 1) {
        const t = forward ? index / samples : 1 - index / samples;
        if (distanceFrom(target, t) >= radius) { outside = t; break; }
        inside = t;
      }
      for (let step = 0; step < 48; step += 1) {
        const middle = (inside + outside) / 2;
        if (distanceFrom(target, middle) >= radius) outside = middle; else inside = middle;
      }
      return outside;
    };
    const startT = trim(from, true);
    const endT = trim(to, false);
    const tip = at(control, endT);
    let low = startT;
    let high = endT;
    for (let step = 0; step < 48; step += 1) {
      const middle = (low + high) / 2;
      const point = at(control, middle);
      if (Math.hypot(point.x - tip.x, point.y - tip.y) <= arrowLength) high = middle; else low = middle;
    }
    const shaft = subCurve(control, startT, high);
    const direction = slope(control, endT);
    const length = Math.hypot(direction.x, direction.y) || 1;
    const dirX = direction.x / length;
    const dirY = direction.y / length;
    const arrow = [
      [tip.x, tip.y],
      [shaft.to.x - dirY * arrowWidth / 2, shaft.to.y + dirX * arrowWidth / 2],
      [shaft.to.x + dirY * arrowWidth / 2, shaft.to.y - dirX * arrowWidth / 2],
    ];
    // Everything the browser paints for this edge: the shaft it draws, and the
    // arrowhead triangle. Clearance is measured on exactly this and nothing else.
    const trace = [...sampleSub(shaft, samples), ...arrow.map(([x, y]) => [x, y])];
    let worst = Infinity;
    obstacles.forEach(shape => {
      trace.forEach(([x, y]) => {
        const gap = distanceToShape(shape, x, y);
        if (gap < worst) worst = gap;
      });
    });
    return {
      bow, start: shaft.from, tip, shaft, arrow, trace,
      worstClearance: obstacles.length ? worst : Infinity,
      unit: { x: dirX, y: dirY },
      path: `M ${shaft.from.x.toFixed(2)} ${shaft.from.y.toFixed(2)} `
        + `Q ${shaft.control.x.toFixed(2)} ${shaft.control.y.toFixed(2)} `
        + `${shaft.to.x.toFixed(2)} ${shaft.to.y.toFixed(2)}`,
    };
  };

  let best = null;
  for (const bow of ladder) {
    const candidate = build(bow);
    if (candidate.worstClearance >= clearance) return { ...candidate, cleared: true, clearance };
    if (!best || candidate.worstClearance > best.worstClearance) best = candidate;
  }
  return { ...best, cleared: false, clearance };
}

/** Route every edge of a drawn graph, so the drawing is one computed object. */
export function graphRoutes(edges, positions, radius, {
  endpoints = [], observed = [], ...options
} = {}) {
  const endpointSet = new Set(endpoints);
  const observedSet = new Set(observed);
  let alternate = 1;
  return edges.map(([from, to]) => {
    const shapes = Object.entries(positions)
      .filter(([name]) => name !== from && name !== to)
      .flatMap(([name, point]) => nodeKeepOut(point, radius, {
        endpoint: endpointSet.has(name), observed: observedSet.has(name),
      }));
    const route = edgeRoute(positions[from], positions[to], radius, shapes,
      { ...options, prefer: alternate });
    if (route.bow !== 0) alternate = -alternate;
    return { from, to, key: `${from}>${to}`, route };
  });
}

/** The polyline a highlighted path follows, from node centre to node centre. */
export function pathPolyline(path, positions) {
  return path.map(node => {
    const point = positions[node];
    if (!point) throw new RangeError(`${node} has no drawn position.`);
    return [point.x, point.y];
  });
}

/**
 * The two population lanes of the intervention figure. Each lane's segment
 * widths are proportional to the stratum weights actually used, so the drawing
 * cannot claim a mixture the arithmetic did not use.
 */
export function laneGeometry(result, { width = 300 } = {}) {
  return result.lanes.flatMap(lane => ([
    { lane: lane.treatment, kind: 'observation', shares: lane.observedShares, risk: lane.observedRisk },
    { lane: lane.treatment, kind: 'intervention', shares: lane.interventionShares, risk: lane.interventionRisk },
  ])).map(entry => {
    const shares = entry.shares[0] === null ? [0, 0] : entry.shares;
    const first = width * shares[0];
    return {
      ...entry,
      segments: [
        { load: 0, x: 0, width: first, share: shares[0] },
        { load: 1, x: first, width: width * shares[1], share: shares[1] },
      ],
      totalWidth: first + width * shares[1],
    };
  });
}

/* ==================================================== declared teaching inputs */

export const alarmNetwork = {
  nodes: ['B', 'E', 'A', 'J', 'M'],
  labels: { B: 'Burglary', E: 'Earthquake', A: 'Alarm sounds', J: 'John calls', M: 'Mary calls' },
  parents: { B: [], E: [], A: ['B', 'E'], J: ['A'], M: ['A'] },
  chance: {
    B: { '': 0.001 },
    E: { '': 0.002 },
    A: { '0,0': 0.001, '0,1': 0.29, '1,0': 0.94, '1,1': 0.95 },
    J: { 0: 0.05, 1: 0.9 },
    M: { 0: 0.01, 1: 0.7 },
  },
};

export const fixtures = {
  /** §1–2: the alarm network and its published evidence sets. */
  alarmEdges: [['B', 'A'], ['E', 'A'], ['A', 'J'], ['A', 'M']],
  assembledWorld: { B: 1, E: 0, A: 1, J: 1, M: 1 },
  publishedEvidence: [
    { label: 'None', evidence: {} },
    { label: 'John calls', evidence: { J: 1 } },
    { label: 'John and Mary call', evidence: { J: 1, M: 1 } },
    { label: 'Both call; no earthquake', evidence: { J: 1, M: 1, E: 0 } },
    { label: 'Alarm definitely sounds', evidence: { A: 1 } },
    { label: 'Alarm sounds; earthquake occurs', evidence: { A: 1, E: 1 } },
    { label: 'Alarm sounds; John calls', evidence: { A: 1, J: 1 } },
  ],
  /** §3: the elimination order the prose follows, and the four-node chain figure. */
  eliminationOrder: ['E', 'A', 'J', 'M'],
  eliminationEvidence: { J: 1, M: 1 },
  chainFactors: [['A', 'B'], ['B', 'C'], ['C', 'D']],
  chainEndpointOrder: ['A', 'D', 'B', 'C'],
  chainMiddleOrder: ['B', 'C', 'A', 'D'],
  /** §4: the path investigation's preset graphs. */
  secondRouteEdges: [['B', 'A'], ['E', 'A'], ['A', 'J'], ['A', 'M'], ['B', 'K'], ['K', 'E']],
  colliderChainEdges: [['R', 'S'], ['T', 'S'], ['S', 'V'], ['V', 'W']],
  /** The suggested setups of investigation 2, as data rather than markup, so
   *  that the verifier can require each one to be able to teach something: a
   *  preset on which every reachable observation set gives the same verdict is
   *  a fixture that cannot show the contrast it is placed there to show. The
   *  practice-5 graph is offered on S and T, not T and Y: T-to-Y is a single
   *  edge and a path with no interior node can never be blocked, so that pair
   *  is invariant — and it is also the wrong question, since d-separation
   *  counts the causal path the backdoor criterion deliberately sets aside. */
  pathPresets: [
    { label: 'The alarm graph', edges: [['B', 'A'], ['E', 'A'], ['A', 'J'], ['A', 'M']], start: 'B', end: 'E' },
    { label: 'Add a second route B \u2192 K \u2192 E',
      edges: [['B', 'A'], ['E', 'A'], ['A', 'J'], ['A', 'M'], ['B', 'K'], ['K', 'E']], start: 'B', end: 'E' },
    { label: 'A collider with a chain below it',
      edges: [['R', 'S'], ['T', 'S'], ['S', 'V'], ['V', 'W']], start: 'R', end: 'T' },
    { label: "Practice 5's graph: can status and training be separated?",
      edges: [['S', 'E'], ['S', 'Y'], ['E', 'T'], ['T', 'Y'], ['E', 'Y']], start: 'S', end: 'T' },
  ],
  /** §1 and practice 3: parameter counting. */
  alarmParameters: [
    { name: 'B', states: 2, parentStates: [] },
    { name: 'E', states: 2, parentStates: [] },
    { name: 'A', states: 2, parentStates: [2, 2] },
    { name: 'J', states: 2, parentStates: [2] },
    { name: 'M', states: 2, parentStates: [2] },
  ],
  practiceParameters: [
    { name: 'C', states: 3, parentStates: [] },
    { name: 'F1', states: 2, parentStates: [3] },
    { name: 'F2', states: 2, parentStates: [3, 2] },
    { name: 'F3', states: 2, parentStates: [3] },
  ],
  naiveBayesParameters: [
    { name: 'C', states: 3, parentStates: [] },
    { name: 'X1', states: 2, parentStates: [3] },
    { name: 'X2', states: 2, parentStates: [3] },
    { name: 'X3', states: 2, parentStates: [3] },
    { name: 'X4', states: 2, parentStates: [3] },
  ],
  treeAugmentedParameters: [
    { name: 'C', states: 3, parentStates: [] },
    { name: 'X1', states: 2, parentStates: [3] },
    { name: 'X2', states: 2, parentStates: [3, 2] },
    { name: 'X3', states: 2, parentStates: [3, 2] },
    { name: 'X4', states: 2, parentStates: [3, 2] },
  ],
  /** §6: the maintenance model, and the assignment change the investigation asks for. */
  service: { assignment: [0.2, 0.6], outcome: [[0.01, 0.1], [0.05, 0.2]] },
  serviceRandomised: { assignment: [0.4, 0.4], outcome: [[0.01, 0.1], [0.05, 0.2]] },
  serviceNoOverlap: { assignment: [0, 1], outcome: [[0.01, 0.1], [0.05, 0.2]] },
  serviceChangedResponse: { assignment: [0.2, 0.6], outcome: [[0.01, 0.1], [0.01, 0.2]] },
  serviceEdges: [['Z', 'X'], ['Z', 'Y'], ['X', 'Y']],
  /** Both service drawings use these fixed positions, so that cutting the
   *  assignment arrow changes one arrow and nothing else. A layered layout
   *  would re-rank X once its only parent is gone, and the two panels a reader
   *  is asked to compare would have different shapes. */
  servicePositions: {
    Z: { x: 40, y: 40 },
    X: { x: 180, y: 40 },
    Y: { x: 110, y: 150 },
  },
  serviceCanvas: { width: 220, height: 190, radius: 17 },
  /** §6 and practice 5: the adjustment-set graphs the destination note concerns. */
  educationEdges: [['S', 'E'], ['S', 'Y'], ['E', 'T'], ['T', 'Y']],
  educationWithDirectEffect: [['S', 'E'], ['S', 'Y'], ['E', 'T'], ['T', 'Y'], ['E', 'Y']],
  educationCandidateSets: [[], ['E'], ['S'], ['E', 'S']],
  educationLabels: { S: 'Status', E: 'Education', T: 'Training', Y: 'Salary' },
  /** §7: the frontdoor construction and the two counterfactual models. */
  frontdoorEdges: [['U', 'X'], ['U', 'Y'], ['X', 'M'], ['M', 'Y']],
  frontdoorWithLatentMediator: [['U', 'X'], ['U', 'Y'], ['X', 'M'], ['M', 'Y'], ['U', 'M']],
  frontdoorWithDirectEffect: [['U', 'X'], ['U', 'Y'], ['X', 'M'], ['M', 'Y'], ['X', 'Y']],
  counterfactualObservation: { treatment: 0, outcome: 0, changedTreatment: 1 },
  practiceCounterfactualObservation: { treatment: 1, outcome: 0, changedTreatment: 0 },
  /** §8: the marginal-MAP counterexample and the practice variation that coincides. */
  queryMasses: { '00': 0.3, '01': 0.3, 10: 0.39, 11: 0.01 },
  practiceQueryMasses: { '00': 0.2, '01': 0.25, 10: 0.4, 11: 0.15 },
  equivalenceClass: [
    { name: 'chain forward', edges: [['X', 'Z'], ['Z', 'Y']] },
    { name: 'chain reversed', edges: [['Y', 'Z'], ['Z', 'X']] },
    { name: 'fork', edges: [['Z', 'X'], ['Z', 'Y']] },
  ],
  collider: { name: 'collider', edges: [['X', 'Z'], ['Y', 'Z']] },
  /** §2 investigation: the edited caller tables whose consequences the prose states. */
  falseCallJohn: 0.2,
  uninformativeCallRow: 0.4,
  /** §5 investigation: the alcohol edits that cross and do not cross a fixed median. */
  alcoholEdits: [13.17, 12.9, 12.8],
  /** §8: the rejection-sampling illustration. */
  rejectionSampleSize: 100000,
};

/** A copy of the alarm network with one caller row replaced, for the evidence lab. */
export function withCallerRow(network, node, state, value) {
  checkProbability(value, `P(${node}=1 | A=${state})`);
  return {
    ...network,
    chance: { ...network.chance, [node]: { ...network.chance[node], [state]: value } },
  };
}

/** A copy of the network with one root probability replaced. */
export function withRootChance(network, node, value) {
  checkProbability(value, `P(${node}=1)`);
  if (network.parents[node].length) throw new RangeError(`${node} is not a root of this network.`);
  return { ...network, chance: { ...network.chance, [node]: { '': value } } };
}

/** A copy of the network with one alarm row replaced. */
export function withAlarmRow(network, key, value) {
  checkProbability(value, `P(A=1 | ${key})`);
  if (!(key in network.chance.A)) throw new RangeError(`The alarm table has no row "${key}".`);
  return { ...network, chance: { ...network.chance, A: { ...network.chance.A, [key]: value } } };
}
