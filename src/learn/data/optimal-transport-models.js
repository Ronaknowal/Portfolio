// Small deterministic teaching models. Costs are declared inputs, not benchmarks.
const sum = values => values.reduce((total, value) => total + value, 0);
const deepFreeze = value => {
  if (value && typeof value === 'object' && !Object.isFrozen(value)) {
    Object.values(value).forEach(deepFreeze);
    Object.freeze(value);
  }
  return value;
};
function bounded(value, minimum, maximum, name) {
  if (!Number.isFinite(value) || value < minimum || value > maximum) throw new RangeError(name + ' is outside the supported teaching range');
}
function probabilityWeights(weights, name) {
  if (!Array.isArray(weights) || weights.length < 1 || weights.length > 12) throw new RangeError(name + ' needs 1–12 weights');
  weights.forEach(value => bounded(value, 0, 1, name));
  if (Math.abs(sum(weights) - 1) > 1e-12) throw new RangeError(name + ' must sum to one');
}
function checkProblem(source, target, costs) {
  probabilityWeights(source, 'Source weights');
  probabilityWeights(target, 'Target weights');
  if (!Array.isArray(costs) || costs.length !== source.length) throw new RangeError('Cost rows must match source weights');
  for (const row of costs) {
    if (!Array.isArray(row) || row.length !== target.length) throw new RangeError('Cost columns must match target weights');
    row.forEach(value => bounded(value, 0, 10000, 'Ground cost'));
  }
}
export const TRANSPORT_COSTS = deepFreeze({
  distance: [[0, 1], [2, 1]],
  squared: [[0, 1], [4, 1]],
  indifferent: [[0, 1], [1, 2]],
  changed: [[2, 0], [0, 2]]
});
export function transportSummary(plan, source, target, costs) {
  const rowSums = plan.map(sum);
  const columnSums = target.map((_, column) => sum(plan.map(row => row[column])));
  const rowErrors = rowSums.map((value, index) => value - source[index]);
  const columnErrors = columnSums.map((value, index) => value - target[index]);
  return {
    rowSums,
    columnSums,
    rowErrors,
    columnErrors,
    residual: Math.max(...rowErrors.map(Math.abs), ...columnErrors.map(Math.abs)),
    cost: sum(plan.flatMap((row, i) => row.map((mass, j) => mass * costs[i][j]))),
    negativeEntropy: sum(plan.flatMap(row => row.map(mass => mass > 0 ? mass * (Math.log(mass) - 1) : 0)))
  };
}
export function twoLocationTransport({
  sourceFirst = 0.5,
  targetFirst = 0.5,
  fraction = 1,
  costKind = 'distance',
  dualPosition
} = {}) {
  bounded(sourceFirst, 0, 1, 'First source weight');
  bounded(targetFirst, 0, 1, 'First target weight');
  bounded(fraction, 0, 1, 'Feasible-plan position');
  const costs = TRANSPORT_COSTS[costKind];
  if (!costs) throw new RangeError('Unknown ground cost');
  const source = [sourceFirst, 1 - sourceFirst],
    target = [targetFirst, 1 - targetFirst];
  const lower = Math.max(0, sourceFirst + targetFirst - 1),
    upper = Math.min(sourceFirst, targetFirst);
  const makePlan = entry => [[entry, sourceFirst - entry], [targetFirst - entry, 1 - sourceFirst - targetFirst + entry]].map(row => row.map(value => Math.max(0, value)));
  const entry = lower + fraction * (upper - lower);
  const slope = costs[0][0] - costs[0][1] - costs[1][0] + costs[1][1];
  const optimumEntry = slope <= 0 ? upper : lower;
  const plan = makePlan(entry),
    optimum = makePlan(optimumEntry);
  // With f0=0, g_j=min(C0j,C1j-f1). The concave piecewise-linear
  // dual reaches its maximum at one of these two breakpoints.
  const breaks = costs[1].map((value, j) => value - costs[0][j]);
  const certificate = potential => {
    const sourcePotential = [0, potential];
    const targetPotential = costs[0].map((value, j) => Math.min(value, costs[1][j] - potential));
    const slack = costs.map((row, i) => row.map((value, j) => value - sourcePotential[i] - targetPotential[j]));
    const value = sum(source.map((mass, i) => mass * sourcePotential[i])) + sum(target.map((mass, j) => mass * targetPotential[j]));
    return {
      sourcePotential,
      targetPotential,
      slack,
      value
    };
  };
  const optimalCertificate = breaks.map(certificate).sort((left, right) => right.value - left.value)[0];
  if (dualPosition !== undefined) bounded(dualPosition, -5, 5, 'Source potential');
  const dual = dualPosition === undefined ? optimalCertificate : certificate(dualPosition);
  const summary = transportSummary(plan, source, target, costs);
  return deepFreeze({
    source,
    target,
    costs,
    lower,
    upper,
    entry,
    slope,
    plan,
    optimum,
    optimumEntry,
    optimumCost: transportSummary(optimum, source, target, costs).cost,
    dual,
    optimalCertificate,
    dualGap: summary.cost - dual.value,
    ...summary
  });
}
export function orderedTransport(sourceLocations, sourceWeights, targetLocations, targetWeights, order = 1) {
  probabilityWeights(sourceWeights, 'Source weights');
  probabilityWeights(targetWeights, 'Target weights');
  if (sourceLocations.length !== sourceWeights.length || targetLocations.length !== targetWeights.length) throw new RangeError('Locations and weights must have matching lengths');
  [...sourceLocations, ...targetLocations].forEach(value => bounded(value, -100, 100, 'Location'));
  bounded(order, 1, 4, 'Wasserstein order');
  const source = sourceLocations.map((position, index) => ({
    position,
    mass: sourceWeights[index],
    index
  })).sort((a, b) => a.position - b.position);
  const target = targetLocations.map((position, index) => ({
    position,
    mass: targetWeights[index],
    index
  })).sort((a, b) => a.position - b.position);
  const plan = sourceWeights.map(() => targetWeights.map(() => 0));
  let i = 0,
    j = 0,
    sourceLeft = source[0].mass,
    targetLeft = target[0].mass;
  while (i < source.length && j < target.length) {
    const mass = Math.min(sourceLeft, targetLeft);
    plan[source[i].index][target[j].index] += mass;
    sourceLeft -= mass;
    targetLeft -= mass;
    // Subtracting the smaller residual exhausts at least one side exactly.
    // A positive cutoff would discard rare mass whose displacement cost matters.
    if (sourceLeft <= 0) {
      i += 1;
      sourceLeft = source[i]?.mass ?? 0;
    }
    if (targetLeft <= 0) {
      j += 1;
      targetLeft = target[j]?.mass ?? 0;
    }
  }
  const costs = sourceLocations.map(x => targetLocations.map(y => Math.abs(x - y) ** order));
  const summary = transportSummary(plan, sourceWeights, targetWeights, costs);
  const knots = [...new Set([...sourceLocations, ...targetLocations])].sort((a, b) => a - b);
  const gaps = knots.slice(0, -1).map((left, index) => {
    const right = knots[index + 1];
    const sourceCdf = sum(sourceLocations.map((position, i) => position <= left ? sourceWeights[i] : 0));
    const targetCdf = sum(targetLocations.map((position, i) => position <= left ? targetWeights[i] : 0));
    return {
      left,
      right,
      sourceCdf,
      targetCdf,
      difference: sourceCdf - targetCdf,
      area: Math.abs(sourceCdf - targetCdf) * (right - left)
    };
  });
  return deepFreeze({
    plan,
    costs,
    ...summary,
    distance: summary.cost ** (1 / order),
    order,
    gaps,
    cdfArea: sum(gaps.map(gap => gap.area))
  });
}
export const CUMULATIVE_SCENARIOS = deepFreeze({
  nearby: {
    title: 'Move mass mainly to neighbours',
    weights: [0.1, 0.4, 0.3, 0.2]
  },
  far: {
    title: 'Move extra mass to the far end',
    weights: [0, 0.1, 0.2, 0.7]
  },
  identical: {
    title: 'Compare the same histogram',
    weights: [0.4, 0.1, 0.2, 0.3]
  }
});
export function cumulativeTransport(scenario = 'nearby', spacing = 1) {
  const selected = CUMULATIVE_SCENARIOS[scenario];
  if (!selected) throw new RangeError('Unknown cumulative comparison');
  bounded(spacing, 0.5, 3, 'Bin spacing');
  const locations = [0, 1, 2, 3].map(value => value * spacing);
  const source = [0.4, 0.1, 0.2, 0.3],
    target = selected.weights;
  return deepFreeze({
    locations,
    source,
    target,
    ...orderedTransport(locations, source, locations, target)
  });
}
function logSumExp(values) {
  const maximum = Math.max(...values);
  return maximum + Math.log(sum(values.map(value => Math.exp(value - maximum))));
}
export function sinkhornScaling({
  source = [0.5, 0.5],
  target = [0.5, 0.5],
  costs = TRANSPORT_COSTS.distance,
  epsilon = 0.5,
  iterations = 1000,
  tolerance = 1e-11,
  traceSteps = 0
} = {}) {
  checkProblem(source, target, costs);
  bounded(epsilon, 0.001, 100, 'Entropy scale');
  if (!Number.isInteger(iterations) || iterations < 1 || iterations > 10000) throw new RangeError('Iteration budget must be 1–10000');
  if (!Number.isInteger(traceSteps) || traceSteps < 0 || traceSteps > 100) throw new RangeError('Trace budget must be 0–100 half-steps');
  bounded(tolerance, 1e-14, 0.1, 'Marginal tolerance');
  const rows = source.map((mass, index) => mass > 0 ? index : -1).filter(index => index >= 0);
  const columns = target.map((mass, index) => mass > 0 ? index : -1).filter(index => index >= 0);
  const logKernel = rows.map(i => columns.map(j => -costs[i][j] / epsilon));
  const logSourceWeights = rows.map(i => Math.log(source[i]));
  const logTargetWeights = columns.map(j => Math.log(target[j]));
  const logSourceScale = rows.map(() => 0),
    logTargetScale = columns.map(() => 0);
  const restore = () => {
    const plan = source.map(() => target.map(() => 0));
    for (let i = 0; i < rows.length; i += 1) for (let j = 0; j < columns.length; j += 1) {
      plan[rows[i]][columns[j]] = Math.exp(logSourceScale[i] + logKernel[i][j] + logTargetScale[j]);
    }
    return plan;
  };
  const trace = [];
  const snapshot = phase => {
    const plan = restore();
    trace.push({
      phase,
      plan,
      ...transportSummary(plan, source, target, costs)
    });
  };
  if (traceSteps) snapshot('Initial kernel');
  let completedIterations = 0,
    plan;
  for (let iteration = 0; iteration < iterations; iteration += 1) {
    for (let i = 0; i < rows.length; i += 1) logSourceScale[i] = logSourceWeights[i] - logSumExp(columns.map((_, j) => logKernel[i][j] + logTargetScale[j]));
    if (trace.length && trace.length <= traceSteps) snapshot('Rows corrected');
    for (let j = 0; j < columns.length; j += 1) logTargetScale[j] = logTargetWeights[j] - logSumExp(rows.map((_, i) => logKernel[i][j] + logSourceScale[i]));
    if (trace.length && trace.length <= traceSteps) snapshot('Columns corrected');
    // Gauge recentering changes neither the plan nor the alternating updates.
    const gauge = sum(logTargetScale) / columns.length;
    for (let i = 0; i < rows.length; i += 1) logSourceScale[i] += gauge;
    for (let j = 0; j < columns.length; j += 1) logTargetScale[j] -= gauge;
    plan = restore();
    // Only feasibility is needed for stopping. Compute entropy/cost once at
    // the end (and for the bounded displayed trace), not on every sweep.
    let residual = 0;
    for (let i = 0; i < source.length; i += 1) residual = Math.max(residual, Math.abs(sum(plan[i]) - source[i]));
    for (let j = 0; j < target.length; j += 1) {
      let columnMass = 0;
      for (let i = 0; i < source.length; i += 1) columnMass += plan[i][j];
      residual = Math.max(residual, Math.abs(columnMass - target[j]));
    }
    completedIterations = iteration + 1;
    if (residual <= tolerance && (!traceSteps || trace.length > traceSteps)) break;
  }
  const summary = transportSummary(plan, source, target, costs);
  return deepFreeze({
    source: [...source],
    target: [...target],
    costs: costs.map(row => [...row]),
    epsilon,
    plan,
    ...summary,
    objective: summary.cost + epsilon * summary.negativeEntropy,
    converged: summary.residual <= tolerance,
    iterations: completedIterations,
    tolerance,
    trace,
    logSourceScale: source.map((_, index) => rows.includes(index) ? logSourceScale[rows.indexOf(index)] : null),
    logTargetScale: target.map((_, index) => columns.includes(index) ? logTargetScale[columns.indexOf(index)] : null)
  });
}
export function equalWeightEntropicPlan(costs, epsilon) {
  checkProblem([0.5, 0.5], [0.5, 0.5], costs);
  if (costs.length !== 2 || costs[0].length !== 2) throw new RangeError('The closed form requires two locations per distribution');
  bounded(epsilon, 0.001, 100, 'Entropy scale');
  const scaledSlope = (costs[0][0] - costs[0][1] - costs[1][0] + costs[1][1]) / (2 * epsilon);
  const ratio = Math.exp(-Math.abs(scaledSlope));
  const smaller = 0.5 * ratio / (1 + ratio),
    larger = 0.5 / (1 + ratio);
  const plan = scaledSlope <= 0 ? [[larger, smaller], [smaller, larger]] : [[smaller, larger], [larger, smaller]];
  const summary = transportSummary(plan, [0.5, 0.5], [0.5, 0.5], costs);
  return deepFreeze({
    plan,
    ...summary,
    objective: summary.cost + epsilon * summary.negativeEntropy
  });
}
export function sinkhornComparison(epsilon = 1, shift = 0, scale = 0.5) {
  bounded(shift, 0, 2, 'Target shift');
  bounded(scale, 0.5, 1.5, 'Target spread');
  const sourceLocations = [0, 2],
    targetLocations = [shift, 2 * scale + shift],
    weights = [0.5, 0.5];
  const cost = (left, right) => left.map(x => right.map(y => (x - y) ** 2));
  const solve = costs => ({
    ...sinkhornScaling({
      costs,
      epsilon,
      iterations: 5000
    }),
    analytic: equalWeightEntropicPlan(costs, epsilon)
  });
  const cross = solve(cost(sourceLocations, targetLocations));
  const sourceSelf = solve(cost(sourceLocations, sourceLocations));
  const targetSelf = solve(cost(targetLocations, targetLocations));
  const exactSquaredDistance = (shift ** 2 + (2 * scale + shift - 2) ** 2) / 2;
  return deepFreeze({
    sourceLocations,
    targetLocations,
    weights,
    cross,
    sourceSelf,
    targetSelf,
    divergence: cross.objective - 0.5 * sourceSelf.objective - 0.5 * targetSelf.objective,
    analyticDivergence: cross.analytic.objective - 0.5 * sourceSelf.analytic.objective - 0.5 * targetSelf.analytic.objective,
    exactSquaredDistance,
    exactDistance: Math.sqrt(exactSquaredDistance),
    converged: cross.converged && sourceSelf.converged && targetSelf.converged
  });
}
export function stabilityComparison(epsilon = 0.5, offset = 1000) {
  bounded(offset, 0, 1000, 'Uniform cost offset');
  const costs = TRANSPORT_COSTS.distance.map(row => row.map(cost => cost + offset));
  const kernel = costs.map(row => row.map(cost => Math.exp(-cost / epsilon)));
  const stable = sinkhornScaling({
    costs,
    epsilon
  });
  const reference = sinkhornScaling({
    epsilon
  });
  return deepFreeze({
    kernel,
    costs,
    offset,
    epsilon,
    vanishedEntries: kernel.flat().filter(value => value === 0).length,
    stable,
    reference,
    maximumPlanDifference: Math.max(...stable.plan.flat().map((value, index) => Math.abs(value - reference.plan.flat()[index])))
  });
}
