function finiteRange(value, low, high, name) {
  if (!Number.isFinite(value) || value < low || value > high) {
    throw new RangeError(`${name} must be a finite number from ${low} to ${high}.`);
  }
}
function pair(value, name) {
  if (!Array.isArray(value) || value.length !== 2 || value.some(item => !Number.isFinite(item))) {
    throw new RangeError(`${name} must contain two finite coordinates.`);
  }
}
export function constraintNumber(value) {
  if (!Number.isFinite(value)) throw new RangeError('Displayed values must be finite.');
  if (value === 0) return '0';
  if (Math.abs(value) < 1e-4 || Math.abs(value) >= 1e5) return value.toExponential(3);
  return Number(value.toFixed(5)).toString();
}
export function projectNonnegative(point) {
  pair(point, 'Point');
  return point.map(value => Math.max(0, value));
}
export function projectSumLine(point, budget) {
  pair(point, 'Point');
  finiteRange(budget, 0.5, 3, 'Budget');
  const correction = (point[0] + point[1] - budget) / 2;
  return point.map(value => value - correction);
}
export function projectSimplexPair(point, budget) {
  pair(point, 'Point');
  finiteRange(budget, 0.5, 3, 'Budget');
  const first = Math.max(0, Math.min(budget, (point[0] - point[1] + budget) / 2));
  return [first, budget - first];
}
function feasibility(point, budget) {
  return {
    nonnegativeViolation: Math.max(0, -point[0], -point[1]),
    equalityResidual: point[0] + point[1] - budget
  };
}
function squaredDistance(first, second) {
  return first.reduce((sum, value, index) => sum + (value - second[index]) ** 2, 0);
}
export function coupledProjectionState(target = [2, -0.6], budget = 1, order = 'orthant-first') {
  pair(target, 'Target');
  target.forEach(value => finiteRange(value, -2, 4, 'Target coordinate'));
  finiteRange(budget, 0.5, 3, 'Budget');
  if (!['orthant-first', 'line-first'].includes(order)) throw new RangeError('Choose a listed projection order.');
  const first = order === 'orthant-first' ? projectNonnegative(target) : projectSumLine(target, budget);
  const second = order === 'orthant-first' ? projectSumLine(first, budget) : projectNonnegative(first);
  const exact = projectSimplexPair(target, budget);
  const frames = [{
    point: [...target],
    label: 'Original target; no projection yet'
  }, {
    point: first,
    label: order === 'orthant-first' ? 'Clamp negative coordinates: satisfies C' : 'Subtract the shared sum correction: satisfies D'
  }, {
    point: second,
    label: order === 'orthant-first' ? 'Project the intermediate point onto D' : 'Project the intermediate point onto C'
  }].map((frame, index) => ({
    ...frame,
    step: index,
    ...feasibility(frame.point, budget),
    distanceSquared: squaredDistance(frame.point, target)
  }));
  return {
    target: [...target],
    budget,
    order,
    exact,
    exactDistanceSquared: squaredDistance(exact, target),
    frames
  };
}
export function originalConstraintCost(x) {
  return (x - 3) ** 2 / 2;
}
export function modifiedConstraintCost(mode, strength, x) {
  finiteRange(x, -10, 10, 'x');
  if (mode === 'quadratic') {
    finiteRange(strength, 0, 20, 'Quadratic penalty');
    return originalConstraintCost(x) + strength * Math.max(0, x - 1) ** 2 / 2;
  }
  if (mode === 'hinge') {
    finiteRange(strength, 0, 5, 'Hinge penalty');
    return originalConstraintCost(x) + strength * Math.max(0, x - 1);
  }
  if (mode === 'barrier') {
    finiteRange(strength, 0.01, 4, 'Barrier weight');
    if (x >= 1) return null;
    return originalConstraintCost(x) - strength * Math.log(1 - x);
  }
  throw new RangeError('Choose quadratic, hinge or barrier.');
}
export function constraintPenaltyState(mode = 'quadratic', strength = 2) {
  modifiedConstraintCost(mode, strength, 0);
  let optimum;
  if (mode === 'quadratic') optimum = 1 + 2 / (1 + strength);else if (mode === 'hinge') optimum = Math.max(1, 3 - strength);else optimum = 1 - strength / (Math.sqrt(1 + strength) + 1);
  return {
    mode,
    strength,
    optimum,
    originalCost: originalConstraintCost(optimum),
    modifiedCost: modifiedConstraintCost(mode, strength, optimum),
    violation: Math.max(0, optimum - 1),
    slack: 1 - optimum,
    hardOptimum: 1,
    hardCost: 2
  };
}
export function consensusAdmmState(target = [2, -0.6], budget = 1, rho = 1, steps = 40) {
  pair(target, 'Target');
  target.forEach(value => finiteRange(value, -2, 4, 'Target coordinate'));
  finiteRange(budget, 0.5, 3, 'Budget');
  finiteRange(rho, 0.1, 10, 'Fixed ADMM penalty');
  if (!Number.isInteger(steps) || steps < 0 || steps > 80) throw new RangeError('Steps must be an integer from 0 to 80.');
  let x = [budget / 2, budget / 2];
  let z = [...x];
  let u = [0, 0];
  const exact = projectSimplexPair(target, budget);
  const frames = [];
  for (let step = 0; step <= steps; step += 1) {
    let dualResidual = null;
    if (step > 0) {
      const previousZ = z;
      x = target.map((value, index) => Math.max(0, (value + rho * (z[index] - u[index])) / (1 + rho)));
      z = projectSumLine(x.map((value, index) => value + u[index]), budget);
      u = u.map((value, index) => value + x[index] - z[index]);
      // For A=I, B=-I, the signed stationarity residual is -rho*(z-new - z-old).
      dualResidual = z.map((value, index) => -rho * (value - previousZ[index]));
    }
    const primalResidual = x.map((value, index) => value - z[index]);
    const primalNorm = Math.hypot(...primalResidual);
    const dualNorm = dualResidual === null ? null : Math.hypot(...dualResidual);
    const primalTolerance = Math.SQRT2 * 1e-4 + 1e-3 * Math.max(Math.hypot(...x), Math.hypot(...z));
    const dualTolerance = Math.SQRT2 * 1e-4 + 1e-3 * rho * Math.hypot(...u);
    const repair = projectSimplexPair(z, budget);
    frames.push({
      step,
      x: [...x],
      z: [...z],
      u: [...u],
      primalResidual,
      dualResidual,
      primalNorm,
      dualNorm,
      primalTolerance,
      dualTolerance,
      stoppingPassed: step > 0 && primalNorm <= primalTolerance && dualNorm <= dualTolerance,
      xFeasibility: feasibility(x, budget),
      zFeasibility: feasibility(z, budget),
      repair,
      repairedCost: squaredDistance(repair, target) / 2
    });
  }
  return {
    target: [...target],
    budget,
    rho,
    exact,
    exactCost: squaredDistance(exact, target) / 2,
    frames
  };
}
export const deploymentCandidates = Object.freeze([Object.freeze({
  id: 'S',
  name: 'Small',
  error: 0.12,
  latencyMs: 8,
  memoryMb: 32
}), Object.freeze({
  id: 'C',
  name: 'Compact',
  error: 0.11,
  latencyMs: 15,
  memoryMb: 48
}), Object.freeze({
  id: 'M',
  name: 'Medium',
  error: 0.08,
  latencyMs: 18,
  memoryMb: 96
}), Object.freeze({
  id: 'L',
  name: 'Large',
  error: 0.07,
  latencyMs: 70,
  memoryMb: 384
}), Object.freeze({
  id: 'G',
  name: 'Legacy',
  error: 0.13,
  latencyMs: 22,
  memoryMb: 128
})]);
export function dominatesMetrics(first, second) {
  if (!Array.isArray(first) || !Array.isArray(second) || first.length === 0 || first.length !== second.length || [...first, ...second].some(value => !Number.isFinite(value))) {
    throw new RangeError('Objective vectors must have the same positive length and finite coordinates.');
  }
  return first.every((value, index) => value <= second[index]) && first.some((value, index) => value < second[index]);
}
export function paretoIndices(points) {
  if (!Array.isArray(points)) throw new RangeError('Supply a finite list of objective vectors.');
  points.forEach(point => dominatesMetrics(point, point));
  if (points.length && points.some(point => point.length !== points[0].length)) throw new RangeError('Objective dimensions must match.');
  return points.map((_, index) => index).filter(index => !points.some((point, other) => other !== index && dominatesMetrics(point, points[index])));
}
export function paretoDecisionState(latencyLimit = 25, memoryLimit = 400, price = 0.2, method = 'error') {
  finiteRange(latencyLimit, 0, 80, 'Latency limit');
  finiteRange(memoryLimit, 0, 400, 'Memory limit');
  finiteRange(price, 0, 1, 'Latency price');
  if (!['error', 'weighted', 'latency'].includes(method)) throw new RangeError('Choose a listed preference rule.');
  const candidates = deploymentCandidates.map(item => ({
    ...item,
    feasible: item.latencyMs <= latencyLimit && item.memoryMb <= memoryLimit,
    score: 100 * item.error + price * item.latencyMs
  }));
  const feasible = candidates.filter(item => item.feasible);
  const front = paretoIndices(feasible.map(item => [item.error, item.latencyMs])).map(index => feasible[index].id);
  const primary = item => method === 'weighted' ? item.score : method === 'latency' ? item.latencyMs : item.error;
  const best = feasible.length ? Math.min(...feasible.map(primary)) : null;
  const ties = feasible.filter(item => Math.abs(primary(item) - best) <= 1e-10).sort((first, second) => first.error - second.error || first.latencyMs - second.latencyMs || first.id.localeCompare(second.id));
  return {
    latencyLimit,
    memoryLimit,
    price,
    method,
    candidates,
    front,
    selected: ties[0]?.id ?? null,
    ties: ties.map(item => item.id)
  };
}
export function continuousObjectives(x) {
  finiteRange(x, -2, 4, 'Decision x');
  return [x * x, (x - 2) ** 2];
}
export function continuousTradeoffState(method = 'weighted', alpha = 0.5, epsilon = 1) {
  if (!['weighted', 'epsilon'].includes(method)) throw new RangeError('Choose weighted or epsilon.');
  finiteRange(alpha, 0, 1, 'Objective weight');
  finiteRange(epsilon, 0, 4, 'Second-objective limit');
  const optimum = method === 'weighted' ? 2 * (1 - alpha) : Math.max(0, 2 - Math.sqrt(epsilon));
  const objectives = continuousObjectives(optimum);
  return {
    method,
    alpha,
    epsilon,
    optimum,
    objectives,
    weightedScore: alpha * objectives[0] + (1 - alpha) * objectives[1],
    slack: epsilon - objectives[1]
  };
}
export function normalizedScore(candidate, latencyUnit = 'ms', convertPrice = true, price = 0.2) {
  if (!['ms', 'seconds'].includes(latencyUnit) || typeof convertPrice !== 'boolean') throw new RangeError('Choose declared latency units and a boolean conversion.');
  finiteRange(price, 0, 1, 'Latency price');
  const latency = latencyUnit === 'ms' ? candidate.latencyMs : candidate.latencyMs / 1000;
  const coefficient = latencyUnit === 'seconds' && convertPrice ? price * 1000 : price;
  return 100 * candidate.error + coefficient * latency;
}
