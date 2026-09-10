// Small acyclic causal models for the lesson. Probabilities are model-calculated,
// not empirical evidence about a real intervention or a general identification engine.
function probability(value, label) {
  if (!Number.isFinite(value) || value < 0 || value > 1) {
    throw new RangeError(`${label} must be a finite probability from zero to one.`);
  }
}
function binary(value, label) {
  if (value !== 0 && value !== 1) throw new RangeError(`${label} must be zero or one.`);
}
function readonly(value) {
  if (value && typeof value === 'object') {
    Object.values(value).forEach(readonly);
    Object.freeze(value);
  }
  return value;
}
export function formatCausalNumber(value) {
  if (value === null) return 'Not identified by this calculation';
  if (!Number.isFinite(value)) return String(value);
  if (value === 0) return '0';
  if (Math.abs(value) < 0.00001) return value.toExponential(3);
  return Number(value.toFixed(6)).toString();
}
export function validateCausalGraph(graph) {
  const {
    nodes,
    edges
  } = graph;
  if (!Array.isArray(nodes) || !Array.isArray(edges) || nodes.length > 8 || !nodes.length || nodes.some(node => typeof node !== 'string') || new Set(nodes).size !== nodes.length) {
    throw new RangeError('Provide one through eight distinct named nodes.');
  }
  const edgeNames = new Set();
  for (const edge of edges) {
    if (!Array.isArray(edge) || edge.length !== 2 || !nodes.includes(edge[0]) || !nodes.includes(edge[1]) || edge[0] === edge[1]) throw new RangeError('Invalid directed edge.');
    const name = JSON.stringify(edge);
    if (edgeNames.has(name)) throw new RangeError('Duplicate directed edge.');
    edgeNames.add(name);
  }
  const visited = new Set(),
    active = new Set();
  function visit(node) {
    if (active.has(node)) throw new RangeError('The causal graph must be acyclic.');
    if (visited.has(node)) return;
    active.add(node);
    edges.filter(([source]) => source === node).forEach(([, target]) => visit(target));
    active.delete(node);
    visited.add(node);
  }
  nodes.forEach(visit);
  return graph;
}
function assertNodes(graph, values) {
  if (!Array.isArray(values) || values.some(value => !graph.nodes.includes(value)) || new Set(values).size !== values.length) throw new RangeError('Use distinct nodes from this graph.');
}
export function causalAncestors(graph, selected) {
  validateCausalGraph(graph);
  assertNodes(graph, selected);
  const ancestors = new Set(selected);
  let changed = true;
  while (changed) {
    changed = false;
    for (const [source, target] of graph.edges) {
      if (ancestors.has(target) && !ancestors.has(source)) {
        ancestors.add(source);
        changed = true;
      }
    }
  }
  return graph.nodes.filter(node => ancestors.has(node));
}
export function cutCausalGraph(graph, incoming = [], outgoing = []) {
  validateCausalGraph(graph);
  assertNodes(graph, incoming);
  assertNodes(graph, outgoing);
  const kept = graph.edges.filter(([source, target]) => !incoming.includes(target) && !outgoing.includes(source));
  return readonly({
    nodes: [...graph.nodes],
    edges: kept.map(edge => [...edge])
  });
}
export function traceCausalPaths(graph, start, end, conditioned = []) {
  validateCausalGraph(graph);
  assertNodes(graph, [start, end, ...conditioned]);
  const conditionedAncestors = new Set(causalAncestors(graph, conditioned));
  const hasArrow = (source, target) => graph.edges.some(edge => edge[0] === source && edge[1] === target);
  const paths = [];
  function search(path) {
    const current = path.at(-1);
    if (current === end) {
      const interior = path.slice(1, -1).map((node, index) => {
        const previous = path[index],
          next = path[index + 2];
        const collider = hasArrow(previous, node) && hasArrow(next, node);
        const open = collider ? conditionedAncestors.has(node) : !conditioned.includes(node);
        return {
          node,
          collider,
          open,
          reason: collider ? open ? 'Collider opened by conditioning on it or a descendant' : 'Collider and its descendants are not conditioned on' : open ? 'Unconditioned non-collider' : 'Conditioned non-collider blocks the path'
        };
      });
      paths.push({
        nodes: [...path],
        interior,
        active: interior.every(item => item.open)
      });
      return;
    }
    const neighbors = new Set(graph.edges.flatMap(([source, target]) => source === current ? [target] : target === current ? [source] : []));
    for (const neighbor of neighbors) if (!path.includes(neighbor)) search([...path, neighbor]);
  }
  search([start]);
  return readonly({
    paths,
    separated: !paths.some(path => path.active),
    conditioned: [...conditioned]
  });
}
export const CAUSAL_PATH_PRESETS = readonly({
  fork: {
    title: 'Common cause',
    nodes: ['X', 'Z', 'Y'],
    edges: [['Z', 'X'], ['Z', 'Y']],
    positions: {
      X: [55, 160],
      Z: [180, 55],
      Y: [305, 160]
    },
    allowed: ['Z'],
    description: 'Prior activity Z influences both offer X and outcome Y.'
  },
  chain: {
    title: 'Mediation chain',
    nodes: ['X', 'M', 'Y'],
    edges: [['X', 'M'], ['M', 'Y']],
    positions: {
      X: [55, 125],
      M: [180, 125],
      Y: [305, 125]
    },
    allowed: ['M'],
    description: 'X changes a mediator M, which changes Y. Blocking this chain removes its total-effect pathway from the conditional comparison.'
  },
  collider: {
    title: 'Selection collider',
    nodes: ['X', 'C', 'Y'],
    edges: [['X', 'C'], ['Y', 'C']],
    positions: {
      X: [55, 70],
      C: [180, 175],
      Y: [305, 70]
    },
    allowed: ['C'],
    description: 'Two independent causes can become associated after selecting on their common effect C.'
  },
  descendant: {
    title: 'Selected descendant',
    nodes: ['X', 'C', 'Y', 'D'],
    edges: [['X', 'C'], ['Y', 'C'], ['C', 'D']],
    positions: {
      X: [55, 60],
      C: [180, 125],
      Y: [305, 60],
      D: [180, 220]
    },
    allowed: ['C', 'D'],
    description: 'Conditioning on D, a descendant of the collider, can open the path even if C itself is not conditioned on.'
  }
});
export const CAUSAL_RULE_PRESETS = readonly({
  observation: {
    title: 'Rule 1: a blocked common-cause path',
    rule: 1,
    nodes: ['X', 'Y', 'Z', 'W'],
    edges: [['W', 'Z'], ['W', 'Y'], ['Z', 'X'], ['X', 'Y']],
    X: ['X'],
    Y: 'Y',
    Z: ['Z'],
    W: ['W'],
    positions: {
      W: [180, 50],
      Z: [55, 140],
      X: [180, 225],
      Y: [305, 140]
    },
    explanation: 'Cut Z→X for the existing action. Conditioning on W blocks the remaining Z←W→Y path. The observation of Z can be removed.'
  },
  observationFails: {
    title: 'Rule 1 fails: an added direct path',
    rule: 1,
    nodes: ['X', 'Y', 'Z', 'W'],
    edges: [['W', 'Z'], ['W', 'Y'], ['Z', 'X'], ['X', 'Y'], ['Z', 'Y']],
    X: ['X'],
    Y: 'Y',
    Z: ['Z'],
    W: ['W'],
    positions: {
      W: [180, 50],
      Z: [55, 140],
      X: [180, 225],
      Y: [305, 140]
    },
    explanation: 'The direct Z→Y path remains active. This graph does not justify deleting the observation.'
  },
  exchange: {
    title: 'Rule 2: exchange after controlling W',
    rule: 2,
    nodes: ['Z', 'Y', 'W'],
    edges: [['W', 'Z'], ['W', 'Y'], ['Z', 'Y']],
    X: [],
    Y: 'Y',
    Z: ['Z'],
    W: ['W'],
    positions: {
      W: [180, 50],
      Z: [55, 170],
      Y: [305, 170]
    },
    explanation: 'Remove outgoing Z→Y for the test. W blocks Z←W→Y. Setting Z and observing Z agree within W under these assumptions.'
  },
  exchangeFails: {
    title: 'Rule 2 fails: hidden common cause',
    rule: 2,
    nodes: ['Z', 'Y', 'U'],
    edges: [['U', 'Z'], ['U', 'Y'], ['Z', 'Y']],
    X: [],
    Y: 'Y',
    Z: ['Z'],
    W: [],
    hidden: ['U'],
    positions: {
      U: [180, 50],
      Z: [55, 170],
      Y: [305, 170]
    },
    explanation: 'Removing the outgoing treatment arrow leaves the unblocked hidden fork Z←U→Y. The action cannot be exchanged by this test.'
  },
  deletion: {
    title: 'Rule 3: an action downstream of Y',
    rule: 3,
    nodes: ['Y', 'Z', 'W'],
    edges: [['Y', 'Z'], ['Z', 'W']],
    X: [],
    Y: 'Y',
    Z: ['Z'],
    W: [],
    positions: {
      Y: [55, 125],
      Z: [180, 125],
      W: [305, 125]
    },
    explanation: 'No observations are conditioned on. Z(W) contains Z; cutting its incoming edge separates it from Y. Changing this downstream mechanism does not change Y.'
  },
  deletionFails: {
    title: 'Rule 3 fails: selection after the action',
    rule: 3,
    nodes: ['Y', 'Z', 'W'],
    edges: [['Y', 'Z'], ['Z', 'W']],
    X: [],
    Y: 'Y',
    Z: ['Z'],
    W: ['W'],
    positions: {
      Y: [55, 125],
      Z: [180, 125],
      W: [305, 125]
    },
    explanation: 'Z is an ancestor of the conditioned W, so Z(W) is empty. Do not cut Y→Z. The required separation fails: selecting after the action can change the composition of Y.'
  }
});
export function inspectDoRule(preset) {
  validateCausalGraph(preset);
  if (![1, 2, 3].includes(preset.rule)) throw new RangeError('Choose do-calculus rule one, two or three.');
  assertNodes(preset, [...preset.X, preset.Y, ...preset.Z, ...preset.W]);
  const base = cutCausalGraph(preset, preset.X);
  const eligibleActions = preset.rule === 3 ? preset.Z.filter(node => !causalAncestors(base, preset.W).includes(node)) : [];
  const transformed = preset.rule === 2 ? cutCausalGraph(base, [], preset.Z) : preset.rule === 3 ? cutCausalGraph(base, eligibleActions) : base;
  const tests = preset.Z.map(node => ({
    node,
    ...traceCausalPaths(transformed, preset.Y, node, [...preset.X, ...preset.W])
  }));
  const removed = preset.edges.filter(edge => !transformed.edges.some(item => item[0] === edge[0] && item[1] === edge[1]));
  return readonly({
    transformed,
    eligibleActions,
    removed,
    tests,
    valid: tests.every(test => test.separated)
  });
}
function conditionalMean(rows, selected, field = 'y') {
  const matching = rows.filter(row => Object.entries(selected).every(([key, value]) => row[key] === value));
  const mass = matching.reduce((sum, row) => sum + row.mass, 0);
  return mass === 0 ? null : matching.reduce((sum, row) => sum + row.mass * row[field], 0) / mass;
}
export function offerPopulation({
  highShare = .5,
  lowAssignment = .2,
  highAssignment = .8
} = {}) {
  [highShare, lowAssignment, highAssignment].forEach(value => probability(value, 'Population or assignment share'));
  const shares = [1 - highShare, highShare],
    assignment = [lowAssignment, highAssignment];
  const outcomes = [[.1, .2], [.3, .4]];
  const rows = [];
  for (const z of [0, 1]) for (const x of [0, 1]) for (const y of [0, 1]) {
    rows.push({
      z,
      x,
      y,
      mass: shares[z] * (x ? assignment[z] : 1 - assignment[z]) * (y ? outcomes[z][x] : 1 - outcomes[z][x])
    });
  }
  const strata = shares.map((share, z) => ({
    z,
    share,
    assignment: assignment[z],
    risks: [0, 1].map(x => conditionalMean(rows, {
      x,
      z
    })),
    treatedMass: share * assignment[z],
    untreatedMass: share * (1 - assignment[z])
  }));
  const observedRisks = [0, 1].map(x => conditionalMean(rows, {
    x
  }));
  const interventionRisks = [0, 1].map(x => shares.reduce((sum, share, z) => sum + share * outcomes[z][x], 0));
  const adjustedRisks = [0, 1].map(x => {
    if (strata.some(stratum => stratum.share > 0 && stratum.risks[x] === null)) return null;
    return strata.reduce((sum, stratum) => sum + stratum.share * (stratum.risks[x] ?? 0), 0);
  });
  const difference = values => values.includes(null) ? null : values[1] - values[0];
  const assignedMass = rows.filter(row => row.x === 1).reduce((sum, row) => sum + row.mass, 0);
  const unassignedMass = rows.filter(row => row.x === 0).reduce((sum, row) => sum + row.mass, 0);
  return readonly({
    rows,
    strata,
    shares,
    assignment,
    outcomes,
    observedRisks,
    interventionRisks,
    adjustedRisks,
    naiveDifference: difference(observedRisks),
    adjustedDifference: difference(adjustedRisks),
    causalDifference: difference(interventionRisks),
    assignedMass,
    treatedHighShare: assignedMass === 0 ? null : highShare * highAssignment / assignedMass,
    untreatedHighShare: unassignedMass === 0 ? null : highShare * (1 - highAssignment) / unassignedMass,
    overlap: strata.every(stratum => stratum.share === 0 || stratum.assignment > 0 && stratum.assignment < 1)
  });
}
export function latentCausalAmbiguity() {
  const mechanisms = [{
    name: 'Model A',
    risks: [[.2, .9], [.1, .8]]
  }, {
    name: 'Model B',
    risks: [[.2, .1], [.9, .8]]
  }];
  return readonly(mechanisms.map(model => {
    const observed = [];
    for (const u of [0, 1]) for (const y of [0, 1]) {
      observed.push({
        x: u,
        y,
        mass: .5 * (y ? model.risks[u][u] : 1 - model.risks[u][u])
      });
    }
    const interventionRisks = [0, 1].map(x => .5 * (model.risks[0][x] + model.risks[1][x]));
    return {
      ...model,
      observed,
      interventionRisks,
      effect: interventionRisks[1] - interventionRisks[0]
    };
  }));
}
export function frontdoorPopulation({
  mediatorLow = .1,
  mediatorHigh = .8,
  directEffect = 0
} = {}) {
  probability(mediatorLow, 'Mediator probability under X=0');
  probability(mediatorHigh, 'Mediator probability under X=1');
  if (!Number.isFinite(directEffect) || directEffect < 0 || directEffect > .1) throw new RangeError('Use direct effects from zero through .1.');
  const mediator = [mediatorLow, mediatorHigh],
    assignment = [.2, .8];
  const outcome = (u, m, x) => .1 + .5 * m + .2 * u + directEffect * x;
  const rows = [];
  for (const u of [0, 1]) for (const x of [0, 1]) for (const m of [0, 1]) for (const y of [0, 1]) {
    const risk = outcome(u, m, x);
    rows.push({
      u,
      x,
      m,
      y,
      mass: .5 * (x ? assignment[u] : 1 - assignment[u]) * (m ? mediator[x] : 1 - mediator[x]) * (y ? risk : 1 - risk)
    });
  }
  const outcomeTable = [0, 1].map(m => [0, 1].map(x => conditionalMean(rows, {
    m,
    x
  })));
  const mediatorRisks = [0, 1].map(m => outcomeTable[m].includes(null) ? null : .5 * (outcomeTable[m][0] + outcomeTable[m][1]));
  const frontdoorRisks = [0, 1].map(x => {
    const weights = [1 - mediator[x], mediator[x]];
    if (weights.some((weight, m) => weight > 0 && mediatorRisks[m] === null)) return null;
    return weights.reduce((sum, weight, m) => sum + weight * (mediatorRisks[m] ?? 0), 0);
  });
  const interventionRisks = [0, 1].map(x => [0, 1].reduce((sum, u) => sum + .5 * ((1 - mediator[x]) * outcome(u, 0, x) + mediator[x] * outcome(u, 1, x)), 0));
  const observedRisks = [0, 1].map(x => conditionalMean(rows, {
    x
  }));
  return readonly({
    rows,
    mediator,
    outcomeTable,
    mediatorRisks,
    frontdoorRisks,
    interventionRisks,
    observedRisks,
    directEffect,
    criterionSatisfied: directEffect === 0,
    hasRequiredSupport: !frontdoorRisks.includes(null)
  });
}
export function counterfactualResponseTypes({
  overlap = .125,
  observedTreatment = 1,
  observedOutcome = 1,
  intervention = 0
} = {}) {
  if (!Number.isFinite(overlap) || overlap < 0 || overlap > .25) throw new RangeError('Response overlap must be between zero and .25.');
  binary(observedTreatment, 'Observed treatment');
  binary(observedOutcome, 'Observed outcome');
  binary(intervention, 'New treatment');
  const types = [{
    label: '00',
    outcomes: [0, 0],
    prior: overlap
  }, {
    label: '01',
    outcomes: [0, 1],
    prior: .75 - overlap
  }, {
    label: '10',
    outcomes: [1, 0],
    prior: .25 - overlap
  }, {
    label: '11',
    outcomes: [1, 1],
    prior: overlap
  }];
  const compatibleMass = types.reduce((sum, type) => sum + (type.outcomes[observedTreatment] === observedOutcome ? type.prior : 0), 0);
  const posteriorTypes = types.map(type => ({
    ...type,
    compatible: type.outcomes[observedTreatment] === observedOutcome,
    posterior: type.outcomes[observedTreatment] === observedOutcome ? type.prior / compatibleMass : 0,
    counterfactualOutcome: type.outcomes[intervention]
  }));
  return readonly({
    overlap,
    observedTreatment,
    observedOutcome,
    intervention,
    types: posteriorTypes,
    compatibleMass,
    populationRisks: [.25, .75],
    populationEffect: .5,
    counterfactualRisk: posteriorTypes.reduce((sum, type) => sum + type.posterior * type.counterfactualOutcome, 0)
  });
}
export function augmentedEffectExpectation({
  assignment = [.2, .8],
  estimatedAssignment = [.2, .8],
  estimatedOutcomes = [[.1, .2], [.3, .4]],
  highShare = .5
} = {}) {
  if (assignment.length !== 2 || estimatedAssignment.length !== 2 || estimatedOutcomes.length !== 2 || estimatedOutcomes.some(row => row.length !== 2)) throw new RangeError('Provide two strata and two treatment outcomes.');
  [...assignment, ...estimatedAssignment, ...estimatedOutcomes.flat(), highShare].forEach(value => probability(value, 'Estimator input'));
  if (estimatedAssignment.some(value => value === 0 || value === 1)) throw new RangeError('Estimated propensities must be strictly between zero and one.');
  const population = offerPopulation({
    highShare,
    lowAssignment: assignment[0],
    highAssignment: assignment[1]
  });
  const augmented = population.rows.reduce((sum, row) => {
    const [mu0, mu1] = estimatedOutcomes[row.z],
      propensity = estimatedAssignment[row.z];
    const score = mu1 - mu0 + row.x * (row.y - mu1) / propensity - (1 - row.x) * (row.y - mu0) / (1 - propensity);
    return sum + row.mass * score;
  }, 0);
  const inverseWeighted = population.rows.reduce((sum, row) => sum + row.mass * (row.x * row.y / estimatedAssignment[row.z] - (1 - row.x) * row.y / (1 - estimatedAssignment[row.z])), 0);
  const standardized = population.shares.reduce((sum, share, z) => sum + share * (estimatedOutcomes[z][1] - estimatedOutcomes[z][0]), 0);
  return readonly({
    augmented,
    inverseWeighted,
    standardized,
    truth: population.causalDifference
  });
}
